# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from sgg_benchmark.structures.image_list import to_image_list
from sgg_benchmark.structures.boxlist_ops import boxlist_iou

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
        self.export = False

    def forward(self, images, targets=None, logger=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[Tensor]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[Tensor] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[Tensor] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.roi_heads.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        images = to_image_list(images)

        with torch.no_grad():
            outputs, features = self.backbone(images.tensors, visualize=False, embed=True)
            # get dino features
            proposals = self.backbone.postprocess(outputs, images.image_sizes)

        if self.roi_heads.training and (targets is not None) and self.add_gt:
            proposals = self.add_gt_proposals(proposals,targets)

        # to avoid the empty list to be passed into roi_heads during testing and cause error in the pooler
        if not self.training and len(proposals[0]) == 0:
            # add empty missing fields
            for p in proposals:
                p["pred_rel_scores"] = torch.tensor([], dtype=torch.float32, device=p.device)
                p["pred_rel_labels"] = torch.tensor([], dtype=torch.float32, device=p.device)
                p["rel_pair_idxs"] = torch.tensor([], dtype=torch.int64, device=p.device)
            return proposals

        if self.roi_heads:
            if self.predcls: # in predcls mode, we pass the targets as proposals
                for t in targets:
                    t["pred_labels"] = t["labels"]
                    t["pred_scores"] = torch.ones_like(t["labels"], dtype=torch.float32)
                x, result, detector_losses = self.roi_heads(features, proposals, targets, logger, targets)
            else:
                x, result, detector_losses = self.roi_heads(features, proposals, targets, logger, proposals)
        else:
            # RPN-only models don't have roi_heads
            result = proposals
            detector_losses = {}

        if self.roi_heads.training:
            losses = {}
            losses.update(detector_losses)
            return losses

        if self.export:
            boxes, rels = self.generate_detect_sg(result[0])
            return [boxes, rels]
        return result
        
    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[Tensor]
            targets: list[Tensor]
        """
        # new_targets = []
        # for t in targets:
        #     new_t = t.copy()
        #     new_t["pred_labels"] = t["labels"]
        #     new_t["pred_scores"] = torch.ones_like(t["labels"], dtype=torch.float32)
        #     new_targets.append(new_t)

        # compute iou
        for i in range(len(proposals)):
            target_boxes = targets[i][0] # only boxes from target
            ious = boxlist_iou(target_boxes, proposals[i])

            # get gt_boxes with iou < 0.5
            gt_boxes = target_boxes[ious.max(1).values < 0.5]
            # add one dim for fake conf in index 4, add the labels back after in index 5
            gt_boxes = torch.cat((gt_boxes[:, :4], torch.ones_like(gt_boxes[:, 0:1]), gt_boxes[:, 4:5]), dim=1)

            # add gt_boxes to proposals
            proposals[i] = torch.cat((proposals[i], gt_boxes), dim=0)

        return proposals
    
    def generate_detect_sg(self, predictions, obj_thres = 0.5):
        all_obj_labels = predictions["pred_labels"]
        all_obj_scores = predictions["pred_scores"]
        all_rel_pairs = predictions["rel_pair_idxs"]
        all_rel_prob = predictions["pred_rel_scores"]
        all_boxes = predictions["bbox"]

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
