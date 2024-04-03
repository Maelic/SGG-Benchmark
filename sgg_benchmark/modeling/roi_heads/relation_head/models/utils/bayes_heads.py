import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils_relation import layer_init

class BayesHead(nn.Module):
    """
    The prediction head with a hierarchical classification when the optional transformer encoder is used.
    """
    def __init__(self, input_dim=512, num_geometric=15, num_possessive=11, num_semantic=24, T1=1, T2=1, T3=1):
        super(BayesHead, self).__init__()
        self.fc3_1 = nn.Linear(input_dim, num_geometric)
        self.fc3_2 = nn.Linear(input_dim, num_possessive)
        self.fc3_3 = nn.Linear(input_dim, num_semantic)
        self.fc5 = nn.Linear(input_dim, 4)
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3

    def forward(self, h):
        super_relation = self.fc5(h)
        # super_relation = F.log_softmax(self.fc5(h), dim=1)

        # By Bayes rule, log p(relation_n, super_n) = log p(relation_1 | super_1) + log p(super_1)
        relation_1 = self.fc3_1(h)           # geometric
        # relation_1 = F.log_softmax(relation_1 / self.T1, dim=1) + super_relation[:, 0].view(-1, 1)
        relation_2 = self.fc3_2(h)           # possessive
        # relation_2 = F.log_softmax(relation_2 / self.T2, dim=1) + super_relation[:, 1].view(-1, 1)
        relation_3 = self.fc3_3(h)           # semantic
        # relation_3 = F.log_softmax(relation_3 / self.T3, dim=1) + super_relation[:, 2].view(-1, 1)
        return relation_1, relation_2, relation_3, super_relation

    def layer_init(self):
        layer_init(self.fc3_1, xavier=True)
        layer_init(self.fc3_2, xavier=True)
        layer_init(self.fc3_3, xavier=True)
        layer_init(self.fc5, xavier=True)


class BayesHeadProb(nn.Module):
    """
    The prediction head with a hierarchical classification when the optional transformer encoder is used.
    """
    def __init__(self, input_dim=512, num_geometric=15, num_possessive=11, num_semantic=24, T1=1, T2=1, T3=1):
        super(BayesHeadProb, self).__init__()
        self.fc3_1 = nn.Linear(input_dim, num_geometric)
        self.fc3_2 = nn.Linear(input_dim, num_possessive)
        self.fc3_3 = nn.Linear(input_dim, num_semantic)
        self.fc5 = nn.Linear(input_dim, 4)
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3

    def forward(self, h):
        super_relation = F.log_softmax(self.fc5(h), dim=1)

        # By Bayes rule, log p(relation_n, super_n) = log p(relation_1 | super_1) + log p(super_1)
        relation_1 = self.fc3_1(h)           # geometric
        relation_1 = F.log_softmax(relation_1 / self.T1, dim=1) + super_relation[:, 1].view(-1, 1)
        relation_2 = self.fc3_2(h)           # possessive
        relation_2 = F.log_softmax(relation_2 / self.T2, dim=1) + super_relation[:, 2].view(-1, 1)
        relation_3 = self.fc3_3(h)           # semantic
        relation_3 = F.log_softmax(relation_3 / self.T3, dim=1) + super_relation[:, 3].view(-1, 1)
        return relation_1, relation_2, relation_3, super_relation

    def layer_init(self):
        layer_init(self.fc3_1, xavier=True)
        layer_init(self.fc3_2, xavier=True)
        layer_init(self.fc3_3, xavier=True)
        layer_init(self.fc5, xavier=True)