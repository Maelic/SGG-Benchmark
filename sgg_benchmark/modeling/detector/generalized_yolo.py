# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from sgg_benchmark.structures.image_list import to_image_list
from sgg_benchmark.structures.boxlist_ops import cat_boxlist

from ..backbone import build_backbone
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedYOLO(nn.Module):
    """
    Main class for Generalized YOLO. Currently supports boxes and masks.
    It consists of two main parts:
    - backbone
    - heads: takes the features + the proposals from the YOLO head and computes the relations from it.
    """

    def __init__(self, cfg):
        super(GeneralizedYOLO, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.predcls = self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        self.add_gt = self.cfg.MODEL.ROI_RELATION_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN

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

        if self.roi_heads.training and (targets is not None) and self.add_gt:
            proposals = self.add_gt_proposals(proposals,targets)

        # to avoid the empty list to be passed into roi_heads during testing and cause error in the pooler
        if not self.training and len(proposals[0].bbox) == 0:
            # add empty missing fields
            for p in proposals:
                p.add_field("pred_rel_scores", torch.tensor([], dtype=torch.float32, device=p.bbox.device))
                p.add_field("rel_pair_idxs", torch.tensor([], dtype=torch.int64, device=p.bbox.device))
            return proposals

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
        
    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        new_targets = []
        for t in targets:
            new_t = t.copy_with_fields(["labels"])
            new_t.add_field("pred_labels", t.get_field("labels"))
            new_t.add_field("pred_scores", torch.ones_like(t.get_field("labels"), dtype=torch.float32))
            new_targets.append(new_t)

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, new_targets)
        ]

        return proposals