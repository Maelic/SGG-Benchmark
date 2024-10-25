# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from sgg_benchmark.modeling import registry
from sgg_benchmark.modeling.make_layers import make_fc
from sgg_benchmark.structures.boxlist_ops import boxlist_union
from sgg_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from sgg_benchmark.modeling.roi_heads.attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor

@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("RelationFeatureExtractor")
class RelationFeatureExtractor(nn.Module):
    """
    Heads for Motifs for relation triplet classification
    """
    def __init__(self, cfg, in_channels):
        super(RelationFeatureExtractor, self).__init__()
        self.cfg = cfg.clone()
        # should corresponding to obj_feature_map function in neural-motifs
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pool_all_levels = cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS

        self.use_spatial = cfg.MODEL.ROI_RELATION_HEAD.USE_SPATIAL_FEATURES
        self.use_union = cfg.MODEL.ROI_RELATION_HEAD.USE_UNION_FEATURES

        assert self.use_spatial or self.use_union, 'No features selected for the relation head'
        
        if cfg.MODEL.ATTRIBUTE_ON:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True, cat_all_levels=pool_all_levels)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels * 2
        else:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels

        # separate spatial
        self.separate_spatial = self.cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        if self.separate_spatial:
            input_size = self.feature_extractor.resize_channels
            out_dim = self.feature_extractor.out_channels
            self.spatial_fc = nn.Sequential(*[make_fc(input_size, out_dim//2), nn.ReLU(inplace=True),
                                              make_fc(out_dim//2, out_dim), nn.ReLU(inplace=True),
                                            ])

        # union rectangle size
        self.rect_size = resolution * 4 -1
        self.rect_conv = nn.Sequential(*[
            nn.Conv2d(2, in_channels //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
            ])


    def forward(self, x, proposals, rel_pair_idxs=None):
        device = x[0].device
        union_proposals = []
        rect_inputs = []
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]
            if self.use_union:
                union_proposal = boxlist_union(head_proposal, tail_proposal)
                union_proposals.append(union_proposal)

            if self.use_spatial:
                # use range to construct rectangle, sized (rect_size, rect_size)
                num_rel = rel_pair_idx.size(0)
                dummy_x_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size, self.rect_size)
                dummy_y_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size, self.rect_size)
                # resize bbox to the scale rect_size
                head_proposal = head_proposal.resize((self.rect_size, self.rect_size))
                tail_proposal = tail_proposal.resize((self.rect_size, self.rect_size))
                head_rect = ((dummy_x_range >= head_proposal.bbox[:,0].floor().view(-1,1,1).long()).bool() & \
                            (dummy_x_range <= head_proposal.bbox[:,2].ceil().view(-1,1,1).long()).bool() & \
                            (dummy_y_range >= head_proposal.bbox[:,1].floor().view(-1,1,1).long()).bool() & \
                            (dummy_y_range <= head_proposal.bbox[:,3].ceil().view(-1,1,1).long()).bool()).float()

                tail_rect = ((dummy_x_range >= tail_proposal.bbox[:,0].floor().view(-1,1,1).long()).bool() & \
                            (dummy_x_range <= tail_proposal.bbox[:,2].ceil().view(-1,1,1).long()).bool() & \
                            (dummy_y_range >= tail_proposal.bbox[:,1].floor().view(-1,1,1).long()).bool() & \
                            (dummy_y_range <= tail_proposal.bbox[:,3].ceil().view(-1,1,1).long()).bool()).float()

                rect_input = torch.stack((head_rect, tail_rect), dim=1) # (num_rel, 4, rect_size, rect_size)
                rect_inputs.append(rect_input)

        # merge two parts
        if self.separate_spatial and self.use_union and self.use_spatial:
            assert (self.use_union and self.use_spatial) # only support this mode if self.separate_spatial is True
            # rectangle feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
            rect_inputs = torch.cat(rect_inputs, dim=0)
            rect_features = self.rect_conv(rect_inputs)
            union_vis_features = self.feature_extractor.pooler(x, union_proposals)
            region_features = self.feature_extractor.forward_without_pool(union_vis_features)
            spatial_features = self.spatial_fc(rect_features.view(rect_features.size(0), -1))
            union_features = (region_features, spatial_features)
        else:
            union_vis_features = torch.tensor([]).to(device)
            rect_features = torch.tensor([]).to(device) 
            if self.use_union:
                union_vis_features = self.feature_extractor.pooler(x, union_proposals)
                union_features = union_vis_features 
            if self.use_spatial:
                rect_inputs = torch.cat(rect_inputs, dim=0)
                rect_features = self.rect_conv(rect_inputs)
                union_features = rect_features 
            if self.use_union and self.use_spatial:
                union_features = union_vis_features + rect_features
            union_features = self.feature_extractor.forward_without_pool(union_features) # (total_num_rel, out_channels)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            union_att_features = self.att_feature_extractor.pooler(x, union_proposals)
            union_features_att = union_att_features + rect_features
            union_features_att = self.att_feature_extractor.forward_without_pool(union_features_att)
            union_features = torch.cat((union_features, union_features_att), dim=-1)
            
        return union_features

def make_roi_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
