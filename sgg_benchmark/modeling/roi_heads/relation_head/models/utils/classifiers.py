import math

import torch
from torch import nn
from torch.nn import functional as F, init

from sgg_benchmark.config import cfg


class WeightNormClassifier(nn.Module):
    """
    Hierarchical Category Context Modeling
    The FC classifier with the weight normalizations
    basically it just normalize the classifier weight while each classifierlassification process.
    """

    def __init__(self, input_dim=1024, num_class=1231, gamma_init=1.0):
        super(WeightNormClassifier, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_class, input_dim))
        self.gamma = nn.Parameter(torch.tensor([gamma_init]))
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.global_context)
        # nn.init.normal_(self.global_context, 0, 0.01)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, cls_feat):
        # Global Representation Normalization
        #  along the feature dimenstion
        # N, 1024
        normalized_global_context = F.normalize(self.weight, dim=1)
        updated_global_context = self.gamma * normalized_global_context
        # num_proposal * 1024) x (1024, num_class)
        cls_score = torch.matmul(cls_feat, updated_global_context.permute(1, 0))
        return cls_score  # cls_score


# ubc implementation
# sometimes occur NaN in backward
# class CosineSimilarityClassifier(nn.Module):
#     """
#     (2) classification score is based on cosine_similarity
#     """
#
#     def __init__(
#             self, input_size, num_classes,
#     ):
#         """
#         Args:
#             cfg: config
#             input_size (int): channels, or (channels, height, width)
#             num_classes (int): number of foreground classes
#             cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
#             box_dim (int): the dimension of bounding boxes.
#                 Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
#         """
#         super(CosineSimilarityClassifier, self).__init__()
#
#         if not isinstance(input_size, int):
#             input_size = np.prod(input_size)
#
#         # The prediction layer for num_classes foreground classes and one
#         self.cls_score = nn.Linear(input_size, num_classes, bias=False)
#         # self.scale = cfg.MODEL.ROI_HEADS.COSINE_SCALE
#         # if self.scale == -1:
#         # learnable global scaling factor
#         self.init_scale = 4.0
#         self.num_classes = num_classes
#         self.scale = nn.Parameter(torch.ones(1) * self.init_scale)
#         # num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
#         # self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
#
#         nn.init.normal_(self.cls_score.weight, std=0.01)
#         # nn.init.normal_(self.bbox_pred.weight, std=0.001)
#         # for l in [self.bbox_pred]:
#         #     nn.init.constant_(l.bias, 0)
#
#     def reset_parameters(self):
#         self.scale = nn.Parameter(torch.ones(1) * self.init_scale)
#         nn.init.normal_(self.cls_score.weight, std=0.01)
#
#     def forward(self, x):
#         # if x.dim() > 2:
#         #     x = torch.flatten(x, start_dim=1)
#
#         # normalize the input x along the `input_size` dimension
#         x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
#         x_normalized = x.div(x_norm + 1e-5)
#
#         # normalize weight
#         temp_norm = torch.norm(self.cls_score.weight.data, p=2, dim=1)\
#                          .unsqueeze(1)\
#                          .expand_as(self.cls_score.weight.data)
#         self.cls_score.weight.data = self.cls_score.weight.data.div(temp_norm + 1e-5)
#         cos_dist = self.cls_score(x_normalized)
#         scores = self.scale * cos_dist
#         return scores


class DotProductClassifier(nn.Module):
    def __init__(self, in_dims, num_class, bias=True, learnable_scale=False):
        super(DotProductClassifier, self).__init__()
        self.in_dims = in_dims
        self.weight = nn.Parameter(torch.Tensor(num_class, in_dims))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_class))
        self.scales = None
        if learnable_scale:
            self.scales = nn.Parameter(torch.ones(num_class))

        # if cfg.MODEL.ROI_RELATION_HEAD.FIX_CLASSIFIER_WEIGHT:
        #     self.fix_weights()
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def fix_weights(self, requires_grad=False):
        self.weight.requires_grad = requires_grad
        if self.bias is not None:
            self.bias.requires_grad = requires_grad

    def forward(self, input):
        output = F.linear(input, self.weight, self.bias)
        if self.scales is not None:
            output *= self.scales

        return output


class CosineSimilarityClassifier(nn.Module):
    """
    large-scale longtail classifier
    not only normalize the classifier weight, but also normalize the initial input features.
    """

    def __init__(self, in_dims, num_class, scale=4, margin=0.5, init_std=0.001):
        """

        :param in_dims: input feature dim
        :param num_class: category numbers
        :param scale:
        :param margin:
        :param init_std:
        """
        super(CosineSimilarityClassifier, self).__init__()
        self.in_dims = in_dims
        self.num_class = num_class
        self.init_scale = scale
        self.scale = nn.Parameter(torch.ones(num_class) * self.init_scale)
        self.margin = margin
        self.weight = nn.Parameter(torch.zeros((num_class, in_dims), device=torch.device('cuda')))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.scale = nn.Parameter(torch.ones(1) * self.init_scale)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input, 2, 1, keepdim=True)
        # x_normalized = input.div(norm_x + 1e-5)
        x_normalized = (norm_x / (1 + norm_x)) * (input / norm_x)
        w_normalized = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * x_normalized, w_normalized.t())


def build_classifier(input_dim, num_class, bias=True):
    if cfg.MODEL.ROI_RELATION_HEAD.CLASSIFIER == "weighted_norm":
        return WeightNormClassifier(input_dim, num_class)
    elif cfg.MODEL.ROI_RELATION_HEAD.CLASSIFIER == "cosine_similarity":
        return CosineSimilarityClassifier(input_dim, num_class)
    elif cfg.MODEL.ROI_RELATION_HEAD.CLASSIFIER == "linear":
        return DotProductClassifier(input_dim, num_class, bias)
    else:
        raise ValueError('invalid classifier type')