# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from sgg_benchmark.layers import ROIAlign
from sgg_benchmark.modeling.make_layers import make_conv3x3

from .utils import cat


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min
        

class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """
    # NOTE: cat_all_levels is added for relationship detection. We want to concatenate 
    # all levels, since detector is fixed in relation detection. Without concatenation
    # if there is any difference among levels, it can not be finetuned anymore. 
    def __init__(self, output_size, scales, sampling_ratio, in_channels=256, cat_all_levels=False):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()

        self.in_channels = in_channels
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        self.cat_all_levels = cat_all_levels
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)
        # reduce the channels
        if self.cat_all_levels:
            self.reduce_channel = make_conv3x3(in_channels * len(self.poolers), in_channels, dilation=1, stride=1, use_relu=True)

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)
        assert rois.size(0) > 0
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(boxes)

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        final_channels = num_channels * num_levels if self.cat_all_levels else num_channels
        result = torch.zeros(
            (num_rois, final_channels, output_size, output_size),
            dtype=dtype,
            device=device,
        )
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):

            if self.cat_all_levels:
                result[:,level*num_channels:(level+1)*num_channels,:,:] = pooler(per_level_feature, rois).to(dtype)
            else:
                idx_in_level = torch.nonzero(levels == level).squeeze(1)
                rois_per_level = rois[idx_in_level]
                result[idx_in_level] = pooler(per_level_feature, rois_per_level).to(dtype)
        if self.cat_all_levels:
            result = self.reduce_channel(result)
        return result
    
class PoolerYOLO(nn.Module):
    def __init__(self, output_size, sampling_ratio, in_channels=[256,512,512], out_channels=256, cat_all_levels=False):
        super(PoolerYOLO, self).__init__()
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.cat_all_levels = cat_all_levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = 3                   # features map depth, for YOLOV8m is 3, with size 20x20x256, 40x40x512, 80x80x512

        self.reduce_channel = make_conv3x3(self.out_channels * self.num_features, self.out_channels, dilation=1, stride=1, use_relu=True)


    def forward(self, x, boxes, reduce=False):
        dtype, device = x[0].dtype, x[0].device

        if reduce:
            # perform a conv 1x1 on all the feature maps one by one to reduce the channels
            for i in range(self.num_features):
                if x[i].shape[1] != self.out_channels:
                    x[i] = nn.Conv2d(x[i].shape[1], self.out_channels, kernel_size=1, stride=1, padding=0, device=device)(x[i])

        num_levels = len(x)
        assert num_levels <= self.num_features
        rois = self.convert_to_roi_format(boxes)
        # assert rois.size(0) > 0

        # Infer scales from the actual image
        scales = [boxes[0].size[0] / feature_map.size(-1) for feature_map in x]
        print(scales)
        poolers = [ROIAlign(self.output_size, spatial_scale=scale, sampling_ratio=self.sampling_ratio) for scale in scales]

        if num_levels == 1:
            return poolers[0](x[0], rois)
        
        map_size = [x[i].shape[2] for i in range(self.num_features)]
        # map_size = map_size[::-1]
        map_levels = LevelMapperYOLO(map_size)
        levels = map_levels(boxes)

        num_rois = rois.size(0)
        output_size = self.output_size[0]

        final_channels = self.out_channels * num_levels if self.cat_all_levels else self.out_channels
        result = torch.zeros(
            (num_rois, final_channels, output_size, output_size),
            dtype=dtype,
            device=device,
        )
        for level, (per_level_feature, pooler) in enumerate(zip(x, poolers)):
            if self.cat_all_levels:
                result[:,level*self.out_channels:(level+1)*self.out_channels,:,:] = pooler(per_level_feature, rois).to(dtype)
            else:
                idx_in_level = torch.nonzero(levels == level).squeeze(1)
                rois_per_level = rois[idx_in_level]
                result[idx_in_level] = pooler(per_level_feature, rois_per_level).to(dtype)

        if self.cat_all_levels:           
            result = self.reduce_channel(result)

        return result
    
    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

class LevelMapperYOLO(object):
    """Determine which level each RoI in a set of RoIs should map to based
    on a specific heuristic.
    """

    def __init__(self, map_size=[20,40,80]):
        self.map_scales = map_size

    def __call__(self, boxlists):
        """
        Assign each ROI to a feature map based on its area, to follow the YOLOV8 architecture with 3 different scales.
        Args:
            boxlists (list[BoxList])
        Returns:
            target_lvls (Tensor[N]): A tensor of the same length as rois, where each element is the target level of the corresponding ROI.
        """
        # Calculate the area of each ROI
        areas = torch.cat([boxlist.area() for boxlist in boxlists])

        # Assign each ROI to a feature map
        target_lvls = torch.zeros_like(areas)
        target_lvls[areas < self.map_scales[1]*self.map_scales[1]] = 0.0
        target_lvls[((areas >= self.map_scales[1]*self.map_scales[1]).bool() & \
                     (areas < self.map_scales[2]*self.map_scales[2]).bool())] = 1.0        
        target_lvls[areas >= self.map_scales[2]*self.map_scales[2]] = 2.0

        return target_lvls

def make_pooler(cfg, head_name):
    resolution = cfg.MODEL[head_name].POOLER_RESOLUTION
    scales = cfg.MODEL[head_name].POOLER_SCALES
    sampling_ratio = cfg.MODEL[head_name].POOLER_SAMPLING_RATIO
    pooler = Pooler(
        output_size=(resolution, resolution),
        scales=scales,
        sampling_ratio=sampling_ratio,
    )
    return pooler
