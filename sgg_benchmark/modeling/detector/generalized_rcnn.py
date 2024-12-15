# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from sgg_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.export = False

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
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, logger)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
        
        if self.training:
            losses = {}
            losses.update(detector_losses)
            if not self.cfg.MODEL.RELATION_ON:
                # During the relationship training stage, the rpn_head should be fixed, and no loss. 
                losses.update(proposal_losses)
            return losses

        if self.export:
            boxes, rels = self.generate_detect_sg(result[0])
            return [boxes, rels] 

        return result

    def generate_detect_sg(self, predictions, obj_thres = 0.5):
        all_obj_labels = predictions.get_field('pred_labels')
        all_obj_scores = predictions.get_field('pred_scores')
        all_rel_pairs = predictions.get_field('rel_pair_idxs')
        all_rel_prob = predictions.get_field('pred_rel_scores')
        all_boxes = predictions.convert('xyxy').bbox

        all_rel_scores, all_rel_labels = all_rel_prob.max(-1)

        # filter objects and relationships
        all_obj_scores[all_obj_scores < obj_thres] = 0.0
        obj_mask = all_obj_scores >= obj_thres
        triplet_score = all_obj_scores[all_rel_pairs[:, 0]] * all_obj_scores[all_rel_pairs[:, 1]] * all_rel_scores
        rel_mask = ((all_rel_labels > 0) + (triplet_score > 0)) > 0

        # filter boxes
        all_boxes = all_boxes[obj_mask]

        # generate filterred result
        num_obj = obj_mask.shape[0]
        num_rel = rel_mask.shape[0]
        rel_matrix = torch.zeros((num_obj, num_obj))
        triplet_scores_matrix = torch.zeros((num_obj, num_obj))
        rel_scores_matrix = torch.zeros((num_obj, num_obj))
        for k in range(num_rel.item()):
            if rel_mask[k].item():
                k0 = all_rel_pairs[k, 0].long()
                k1 = all_rel_pairs[k, 1].long()
                rel_matrix[k0, k1], triplet_scores_matrix[k0, k1], rel_scores_matrix[k0, k1] = all_rel_labels[k], triplet_score[k], all_rel_scores[k]
        rel_matrix = rel_matrix[obj_mask][:, obj_mask].long()
        triplet_scores_matrix = triplet_scores_matrix[obj_mask][:, obj_mask].float()
        rel_scores_matrix = rel_scores_matrix[obj_mask][:, obj_mask].float()
        filter_obj = all_obj_labels[obj_mask]
        filter_pair = torch.nonzero(rel_matrix > 0)
        filter_rel = rel_matrix[filter_pair[:, 0], filter_pair[:, 1]]
        filter_scores = triplet_scores_matrix[filter_pair[:, 0], filter_pair[:, 1]]
        filter_rel_scores = rel_scores_matrix[filter_pair[:, 0], filter_pair[:, 1]]
        # assert that filter_rel and filter_scores are same shape:
        # assert(filter_rel.size() == filter_scores.size() == filter_rel_scores.size())
        
        # make 2 output tensors: one for boxes, one for relations
        # boxes tensor is shape (num_obj, 6) with columns (x1, y1, x2, y2, label, score)
        # rels tensor is shape (num_rel, 4) with columns (obj1, obj2, label, score)

        boxes = torch.cat((all_boxes, filter_obj[:, None], all_obj_scores[obj_mask][:, None]), dim=1)
        rels = torch.cat((filter_pair, filter_rel[:, None], filter_scores[:, None], filter_rel_scores[:, None]), dim=1)

        return boxes, rels
