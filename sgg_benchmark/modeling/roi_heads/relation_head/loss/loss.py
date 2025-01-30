# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from sgg_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from sgg_benchmark.modeling.box_coder import BoxCoder
from sgg_benchmark.modeling.matcher import Matcher
from sgg_benchmark.structures.boxlist_ops import boxlist_iou
from sgg_benchmark.modeling.utils import cat

from .reweight_loss import ReweightingCE, CEForSoftLabel, EdgeDensityLoss

class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        rel_loss,
        pred_weight,
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio

        self.criterion_loss_obj = nn.CrossEntropyLoss()

        if rel_loss == "LabelSmoothingRegression":
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        elif rel_loss == 'BCEWithLogitsLoss':
            self.criterion_loss = nn.BCEWithLogitsLoss()
        elif rel_loss == 'CEForSoftLabel':
            self.criterion_loss = CEForSoftLabel()
        elif rel_loss == "ReweightingCE":
            self.criterion_loss = ReweightingCE(pred_weight)
        elif rel_loss == "CrossEntropyLoss":
            self.criterion_loss = nn.CrossEntropyLoss()
        elif rel_loss == "EdgeDensityLoss":
            self.criterion_loss = EdgeDensityLoss()


    def __call__(self, proposals, rel_labels, relation_logits, refine_logits=None):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        relation_logits = cat(relation_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        loss_relation = self.criterion_loss(relation_logits, rel_labels)

        if refine_obj_logits is not None:
            refine_obj_logits = cat(refine_obj_logits, dim=0)
            loss_refine_obj = self.criterion_loss_obj(refine_obj_logits, fg_labels.long())

        # The following code is used to calculate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets, 
                                             fg_bg_sample=self.attribute_sampling, 
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            if refine_obj_logits is not None: 
                return loss_relation, (loss_refine_obj, loss_refine_att)
            else:
                return loss_relation, loss_refine_att
        elif refine_obj_logits is not None: 
            return loss_relation, loss_refine_obj
        else:
            return loss_relation, None

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class FocalLossFGBGNormalization(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, logits=True, fgbgnorm=True):
        super(FocalLossFGBGNormalization, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_loss = FocalLoss(alpha, gamma, logits, reduce=False)

    def forward(self, inputs, targets, reduce=True):
        loss = self.focal_loss(inputs, targets)
        
        loss = loss.sum(-1)
        loss /= (len(torch.nonzero(targets)) + 1)

        return loss.mean(-1)