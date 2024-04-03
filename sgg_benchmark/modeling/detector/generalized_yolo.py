# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from sgg_benchmark.structures.image_list import to_image_list
from sgg_benchmark.structures.bounding_box import BoxList
from sgg_benchmark.modeling.backbone import add_gt_proposals

from ..backbone import build_backbone
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedYOLO(nn.Module):
    """
    Main class for Generalized YOLO. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - heads: takes the features + the proposals from the YOLO head and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedYOLO, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.predcls = self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL

    def forward(self, images, targets=None, logger=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.roi_heads.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        images = to_image_list(images)
        with torch.no_grad():
            outputs, features = self.backbone(images.tensors, embed=True)
            proposals = self.backbone.postprocess(outputs, images.image_sizes)
        
        # if self.roi_heads.training and not self.predcls:
        #     proposals = add_gt_proposals(proposals, targets)

        if self.roi_heads:
            if self.predcls and self.roi_heads.training: # in predcls mode, we pass the targets as proposals
                for t in targets:
                    t.remove_field("image_path")
                x, result, detector_losses = self.roi_heads(features, proposals, targets, logger, targets)
            else:
                x, result, detector_losses = self.roi_heads(features, proposals, targets, logger, proposals)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.roi_heads.training:
            losses = {}
            losses.update(detector_losses)
            return losses
        return result