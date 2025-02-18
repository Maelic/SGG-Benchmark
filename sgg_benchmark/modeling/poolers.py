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
    def __init__(self, output_size, scales, sampling_ratio, in_channels=256, cat_all_levels=False, img_size=640):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(PoolerYOLO, self).__init__()

        self.in_channels = in_channels
        poolers = []
        self.scales = scales
        self.yolo_scales = [img_size / s for s in self.scales]

        for scale in self.yolo_scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        self.cat_all_levels = cat_all_levels

        self.map_levels = LevelMapperYOLO(scales)
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

class LevelMapperYOLO(object):
    def __init__(self, scales):
        self.scales = scales  # List of scales for YOLOv8, something like [80,40,20]
        self.idx_to_level = [(scales[0]**2)-1, 
                            (scales[0]**2) + (scales[1]**2)-1, 
                            (scales[0]**2) + (scales[1]**2) + (scales[2]**2) - 1 
                            ]  # Max index for each scale, something like [6399, 7999, 8399]

    def __call__(self, boxlists):
        # Calculate the square root of the areas of the boxes
        indices = cat([boxlist.get_field("feat_idx") for boxlist in boxlists], dim=0)

        target_lvls = torch.zeros_like(indices, dtype=torch.int64)

        # Assign values based on conditions
        target_lvls[indices < self.idx_to_level[0]] = 2
        target_lvls[(indices >= self.idx_to_level[0]) & (indices < self.idx_to_level[1])] = 1
        target_lvls[indices >= self.idx_to_level[1]] = 0

        # Convert to int64 and adjust for zero-based indexing
        return target_lvls.to(torch.int64)