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

class EdgeDensityLoss(nn.Module):
    """
    Based on 
    [1] B. Knyazev, H. de Vries, C. Cangea, G.W. Taylor, A. Courville, E. Belilovsky.
    Graph Density-Aware Losses for Novel Compositions in Scene Graph Generation. BMVC 2020.
    https://arxiv.org/abs/2005.08230
    """

    def __init__(self, loss_weight=1.0):
        super(EdgeDensityLoss, self).__init__()
        self.loss_weight = loss_weight

        self.crit_loss =  nn.CrossEntropyLoss()

    def forward(self, input, target):
        loss = self.crit_loss(input, target)

        idx_fg = torch.nonzero(target > 0).data.view(-1)
        idx_bg = torch.nonzero(target == 0).data.view(-1)

        M_FG, M_BG, M = len(idx_fg), len(idx_bg), len(input)

        edge_weights = torch.ones(M).to(input)
        if M_FG > 0:
            edge_weights[idx_fg] = 1 / M_FG
        if M_BG > 0 and M_FG > 0:
            edge_weights[idx_bg] = 1 / M_FG

        loss = loss * torch.autograd.Variable(edge_weights)
        return torch.sum(loss)


class CEForSoftLabel(nn.Module):
    """
    Given a soft label, choose the class with max class as the GT.
    converting label like [0.1, 0.3, 0.6] to [0, 0, 1] and apply CrossEntropy
    """
    def __init__(self, reduction="mean"):
        super(CEForSoftLabel, self).__init__()
        self.reduction=reduction

    def forward(self, input, target, pos_weight=None):
        final_target = torch.zeros_like(target)
        final_target[torch.arange(0, target.size(0)), target.argmax(1)] = 1.
        target = final_target
        x = F.log_softmax(input, 1)
        loss = torch.sum(- x * target, dim=1)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')


class ReweightingCE(nn.Module):
    """
    Given a soft label, choose the class with max class as the GT.
    converting label like [0.1, 0.3, 0.6] to [0, 0, 1] and apply CrossEntropy
    """
    def __init__(self, pred_weight, reduction="mean"):
        super(ReweightingCE, self).__init__()
        self.reduction=reduction
        self.pred_weight = pred_weight

    def forward(self, input, target):
        """
        Args:
            input: the prediction
            target: [N, N_classes]. For each slice [weight, 0, 0, 1, 0, ...]
                we need to extract weight.
        Returns:

        """
        print(target)
        print(target[0].shape)
        final_target = torch.zeros_like(target)
        final_target[torch.arange(0, target.size(0)), target.argmax(0)] = 1.
        idxs = (target[:, 0] != 1).nonzero().squeeze()
        # add weight to the target
        weights = torch.ones_like(target[:, 0])
        weights[idxs] = self.pred_weight[target[idxs, 1]]
        target = final_target
        x = F.log_softmax(input, 1)
        loss = torch.sum(- x * target, dim=1)*weights
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')