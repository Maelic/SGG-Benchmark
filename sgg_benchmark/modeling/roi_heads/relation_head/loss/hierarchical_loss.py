
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

class RelationHierarchicalLossComputation(object):
    def __init__(
            self,
            attri_on,
            num_attri_cat,
            max_num_attri,
            attribute_sampling,
            attribute_bgfg_ratio,
            use_label_smoothing,
            predicate_proportion,
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
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5, ] + predicate_proportion)).cuda()

        # Assume NLL loss here.
        # TODO: is class_weight a pre-defined constant?
        # class_weight = 1 - relation_count / torch.sum(relation_count)
        self.criterion_loss = nn.CrossEntropyLoss()
        self.geo_criterion_loss = nn.NLLLoss()
        self.pos_criterion_loss = nn.NLLLoss()
        self.sem_criterion_loss = nn.NLLLoss()
        self.super_criterion_loss = nn.NLLLoss()
        # Hierarchical label
        self.geo_mapping = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 10: 7, 22: 8, 23: 9, 29: 10,
            31: 11, 32: 12, 33: 13, 43: 14
        }

        self.pos_mapping = {
            9: 0, 16: 1, 17: 2, 20: 3, 27: 4, 30: 5, 36: 6, 42: 7, 48: 8, 49: 9, 50: 10
        }
        self.sem_mapping = {
            7: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 18: 6, 19: 7, 21: 8, 24: 9, 25: 10,
            26: 11, 28: 12, 34: 13, 35: 14, 37: 15, 38: 16, 39: 17, 40: 18, 41: 19,
            44: 20, 45: 21, 46: 22, 47: 23
        }
        self.geo_label_tensor = torch.tensor([x for x in self.geo_mapping.keys()])
        self.pos_label_tensor = torch.tensor([x for x in self.pos_mapping.keys()])
        self.sem_label_tensor = torch.tensor([x for x in self.sem_mapping.keys()])

    # Assume no refine obj, only relation prediction
    # relation_logits is [geo, pos, sem, super]
    def __call__(self, proposals, rel_labels, rel_probs, refine_logits):
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
        rel1_prob, rel2_prob, rel3_prob, super_rel_prob = rel_probs
        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        refine_obj_logits = cat(refine_logits, dim=0)
        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        rel_labels = cat(rel_labels, dim=0)  # (rel, 1)
        rel1_prob = cat(rel1_prob, dim=0)  # (rel, 15)
        rel2_prob = cat(rel2_prob, dim=0)  # (rel, 11)
        rel3_prob = cat(rel3_prob, dim=0)  # (rel, 24)
        super_rel_prob = cat(super_rel_prob, dim=0)   # (rel, 4)
        cur_device = rel_labels.device

        # A mask to select labels within specific super category
        geo_label_tensor = self.geo_label_tensor.to(cur_device)
        pos_label_tensor = self.pos_label_tensor.to(cur_device)
        sem_label_tensor = self.sem_label_tensor.to(cur_device)
        # print(rel_labels.device)
        # print(self.geo_label_tensor.device)
        geo_label_mask = (rel_labels.unsqueeze(1) == geo_label_tensor).any(1)
        pos_label_mask = (rel_labels.unsqueeze(1) == pos_label_tensor).any(1)
        sem_label_mask = (rel_labels.unsqueeze(1) == sem_label_tensor).any(1)
        # Suppose 0 is geo, 1 is pos, 3 is sem
        # super_rel_label = pos_label_mask * 1 + sem_label_mask * 2
        # Suppose 0 is bg, 1 is geo, 2 is pos, 3 is sem
        super_rel_label = geo_label_mask + pos_label_mask * 2 + sem_label_mask * 3

        loss_relation = 0
        geo_labels = rel_labels[geo_label_mask]
        geo_labels = torch.tensor([self.geo_mapping[label.item()] for label in geo_labels]).to(cur_device)
        pos_labels = rel_labels[pos_label_mask]
        pos_labels = torch.tensor([self.pos_mapping[label.item()] for label in pos_labels]).to(cur_device)
        sem_labels = rel_labels[sem_label_mask]
        sem_labels = torch.tensor([self.sem_mapping[label.item()] for label in sem_labels]).to(cur_device)

        if geo_labels.shape[0] > 0:
            loss_relation += self.geo_criterion_loss(rel1_prob[geo_label_mask], geo_labels.long())
        if pos_labels.shape[0] > 0:
            loss_relation += self.pos_criterion_loss(rel2_prob[pos_label_mask], pos_labels.long())
        if sem_labels.shape[0] > 0:
            loss_relation += self.sem_criterion_loss(rel3_prob[sem_label_mask], sem_labels.long())
        if super_rel_label.shape[0] > 0:
            loss_relation += self.super_criterion_loss(super_rel_prob, super_rel_label.long())

        return loss_relation, loss_refine_obj