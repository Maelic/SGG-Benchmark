import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sgg_benchmark.modeling.poolers import Pooler
from sgg_benchmark.structures.bounding_box import BoxList
from sgg_benchmark.utils.txt_embeddings import obj_edge_vectors

from .utils_simrel_old import *

def assembling_auxiliary_query(
    auxiliary_rel, \
    class_logits_, pred_bboxes_, pro_features_, roi_features_, \
    so_class_logits, pred_bboxes_so, so_pro_features, so_roi_features, \
    rel_logits, rel_feat, vis_table, \
    new_arrival_ent_max_num=2, R=3, max_recall=100, iou_thres=0.5
):
    #class_logits = class_logits_.detach()
    #pred_bboxes = pred_bboxes_.detach()
    #pro_features = pro_features_.detach()
    class_logits = class_logits_
    pred_bboxes = pred_bboxes_
    pro_features = pro_features_
    roi_features = roi_features_
    
    N, nr_boxes = class_logits.shape[:2]
    nr_boxes = nr_boxes // 2
    nr_boxes_so = so_pro_features.shape[1]
    #final_new_arrival_mask, already_hit_mask = \
    final_new_arrival_mask = \
        auxiliary_branch(
            class_logits[:, 0::2, :], pred_bboxes[:, 0::2, :], \
            so_class_logits, pred_bboxes_so, so_pro_features, \
            rel_logits, vis_table, new_arrival_ent_max_num=new_arrival_ent_max_num, \
            R=R, max_recall=max_recall, iou_thres=iou_thres)
    
    vis_table = (vis_table | final_new_arrival_mask)
    final_new_arrival_mask = final_new_arrival_mask.view(-1)
    
    so_roi_features = so_roi_features.view(N, nr_boxes_so, -1)
    auxiliary_roi_feat = roi_features.view(N * pro_features.shape[1], -1)
    auxiliary_roi_feat = auxiliary_roi_feat[final_new_arrival_mask].view(N, new_arrival_ent_max_num, -1)
    auxiliary_s_feat = auxiliary_roi_feat
    #auxiliary_o_feat = auxiliary_roi_feat.clone().detach()
    auxiliary_o_feat = auxiliary_roi_feat.clone()
    auxiliary_o_feat = auxiliary_o_feat.view(N, new_arrival_ent_max_num, 2, -1)
    auxiliary_o_feat = torch.flip(auxiliary_o_feat, dims=[2])
    auxiliary_o_feat = auxiliary_o_feat.view(N, new_arrival_ent_max_num, -1)
    so_roi_features = torch.cat([so_roi_features, auxiliary_s_feat, auxiliary_o_feat], 1).to(class_logits.device)

    auxiliary_feat = pro_features.view(N * pro_features.shape[1], -1)
    auxiliary_feat = auxiliary_feat[final_new_arrival_mask].view(N, new_arrival_ent_max_num, -1)
    auxiliary_s_feat = auxiliary_feat
    #auxiliary_o_feat = auxiliary_feat.clone().detach()
    auxiliary_o_feat = auxiliary_feat.clone()
    auxiliary_o_feat = auxiliary_o_feat.view(N, new_arrival_ent_max_num, 2, -1)
    auxiliary_o_feat = torch.flip(auxiliary_o_feat, dims=[2])
    auxiliary_o_feat = auxiliary_o_feat.view(N, new_arrival_ent_max_num, -1)
    so_pro_features = torch.cat([so_pro_features, auxiliary_s_feat, auxiliary_o_feat], 1).to(class_logits.device)
    
    auxiliary_boxes = pred_bboxes.view(N * pro_features.shape[1], -1)
    auxiliary_boxes = auxiliary_boxes[final_new_arrival_mask].view(N, new_arrival_ent_max_num, 8)
    auxiliary_s_boxes = auxiliary_boxes
    #auxiliary_o_boxes = auxiliary_boxes.clone().detach()
    auxiliary_o_boxes = auxiliary_boxes.clone()
    auxiliary_o_boxes = auxiliary_o_boxes.view(N, new_arrival_ent_max_num, 2, 4)
    auxiliary_o_boxes = torch.flip(auxiliary_o_boxes, dims=[2])
    auxiliary_o_boxes = auxiliary_o_boxes.view(N, new_arrival_ent_max_num, 8)
    pred_bboxes_so = torch.cat([pred_bboxes_so, auxiliary_s_boxes, auxiliary_o_boxes], 1).to(class_logits.device)
    
    
    auxiliary_feat = class_logits
    auxiliary_feat = auxiliary_feat.view(N * pro_features.shape[1], -1)
    auxiliary_feat = auxiliary_feat[final_new_arrival_mask].view(N, new_arrival_ent_max_num, -1)
    auxiliary_s_feat = auxiliary_feat
    #auxiliary_o_feat = auxiliary_feat.clone().detach()
    auxiliary_o_feat = auxiliary_feat.clone()
    auxiliary_o_feat = auxiliary_o_feat.view(N, new_arrival_ent_max_num, 2, -1)
    auxiliary_o_feat = torch.flip(auxiliary_o_feat, dims=[2])
    auxiliary_o_feat = auxiliary_o_feat.view(N, new_arrival_ent_max_num, -1)
    so_class_logits = torch.cat([so_class_logits, auxiliary_s_feat, auxiliary_o_feat], 1).to(class_logits.device)
    
    auxiliary_rel = auxiliary_rel.view(N * nr_boxes, -1)
    new_arrival_auxi_rel_feat = auxiliary_rel[final_new_arrival_mask].view(N, new_arrival_ent_max_num, 2, -1)
    new_arrival_auxi_rel_feat_s = new_arrival_auxi_rel_feat[:, :, 0, :]
    new_arrival_auxi_rel_feat_o = new_arrival_auxi_rel_feat[:, :, 1, :]
    rel_feat = torch.cat([rel_feat, new_arrival_auxi_rel_feat_s, new_arrival_auxi_rel_feat_o], 1).to(class_logits.device)
    return so_pro_features, pred_bboxes_so, so_class_logits, rel_feat, so_roi_features, vis_table

def auxiliary_branch(
    class_logits, pred_bboxes, \
    so_class_logits, pred_bboxes_so, so_pro_features, \
    rel_logits, vis_table, new_arrival_ent_max_num=10, \
    R=3, max_recall=100, iou_thres=0.5
):
    N, nr_boxes = class_logits.shape[:2]
    nr_boxes_so = so_pro_features.shape[1]
    
    #already_hit_mask = torch.zeros_like(class_logits)

    final_new_arrival_mask = torch.zeros(N, nr_boxes, dtype=torch.bool)
    with torch.no_grad():
        _, extended_class, extended_boxes, tri_mask = extend_and_select_tri(\
            so_class_logits, pred_bboxes_so, rel_logits, R=R, max_recall=max_recall)
        selected_class, selected_boxes = extended_class[tri_mask], extended_boxes[tri_mask]
        selected_class = selected_class.view(N, 1, max_recall * 2)
        selected_boxes = selected_boxes.view(N, max_recall * 2, 4)
        
        R = 1
        
        iou_mat = batch_box_iou(pred_bboxes, selected_boxes)[0] # N, nr_boxes*2, M
        iou_mat = (iou_mat >= iou_thres)
        iou_mat = iou_mat.unsqueeze(2).repeat(1,1,R,1) # N, nr_boxes*2, R, M
        iou_mat = iou_mat.view(N, nr_boxes * R, -1)
        
        pure_ent_cls_score = torch.sigmoid(class_logits) # N, nr_boxes*2, 150
        _, pure_ent_class = torch.topk(pure_ent_cls_score, k=R, dim=-1, sorted=True) # N, nr_boxes*2, R
        pure_ent_class = pure_ent_class.view(N, -1) # N, nr_boxes*2 * R
        cls_equal_mat = (pure_ent_class.unsqueeze(2) == selected_class)
        
        new_ent_mask_mat = (iou_mat & cls_equal_mat) # N, nr_boxes*2 * R, M
        new_ent_mask_mat = new_ent_mask_mat.view(N, nr_boxes, -1)
        new_ent_mask_mat = (new_ent_mask_mat.sum(-1) == 0) # N, nr_boxes * 2
        
        
        pure_ent_cls_max_score, _ = torch.max(pure_ent_cls_score, dim=-1) # N, nr_boxes*2
        for i in range(N):
            pure_ent_cls_max_score[i][~new_ent_mask_mat[i]] -= 2.
            pure_ent_cls_max_score[i][vis_table[i]] -= 2.
            
            _, sorting_idx = torch.sort(pure_ent_cls_max_score[i], dim=0, descending=True)
            sorting_idx = sorting_idx[:new_arrival_ent_max_num]
            final_new_arrival_mask[i][sorting_idx] = True
            

    return final_new_arrival_mask 
    
    
def extend_and_select_tri(pre_obj_logits, obj_boxes, rel_logits, R=3, max_recall=100):
    N, nr_boxes = rel_logits.shape[:2]
    tri_mask = torch.zeros(N, nr_boxes * R * R, dtype=torch.bool)
    with torch.no_grad():
        pre_obj_scores = torch.sigmoid(pre_obj_logits)
        rel_score = torch.sigmoid(rel_logits) # rel_score: N, nr_boxes, num_rel_class
        pre_obj_scores = pre_obj_scores.view(N, 2 * nr_boxes, -1)
        obj_boxes = obj_boxes.view(N, 2 * nr_boxes, 4)
        
        pre_obj_scores_s = pre_obj_scores[:, 0::2] # N, nr_boxes, cls_num
        pre_obj_scores_o = pre_obj_scores[:, 1::2]
        s_score, s_class = torch.topk(pre_obj_scores_s, k=R, dim=-1, sorted=True) # N, nr_boxes, R
        o_score, o_class = torch.topk(pre_obj_scores_o, k=R, dim=-1, sorted=True)
        s_boxes = obj_boxes[:, 0::2, :] # N, nr_boxes, 4
        o_boxes = obj_boxes[:, 1::2, :]
        
        X, Y = np.meshgrid(np.arange(R), np.arange(R), indexing='ij')
        X = X.reshape(-1)
        Y = Y.reshape(-1)
        
        s_score_d = s_score[:, :, X] # N, nr_boxes, R*R
        o_score_d = o_score[:, :, Y]
        score_d = torch.cat([s_score_d.unsqueeze(3), \
            o_score_d.unsqueeze(3)], 3) # N, nr_boxes, R*R, 2
        
        s_class_d = s_class[:, :, X] # N, nr_boxes, R*R
        o_class_d = o_class[:, :, Y]
        class_d = torch.cat([s_class_d.unsqueeze(3), \
            o_class_d.unsqueeze(3)], 3) # N, nr_boxes, R*R, 2
        
        s_boxes = s_boxes.unsqueeze(2).repeat(1,1,R,1) # N, nr_boxes, R, 4
        o_boxes = o_boxes.unsqueeze(2).repeat(1,1,R,1)
        s_boxes_d = s_boxes[:, :, X, :] # N, nr_boxes, R*R, 4
        o_boxes_d = o_boxes[:, :, Y, :]
        boxes_d = torch.cat([s_boxes_d.unsqueeze(3), \
            o_boxes_d.unsqueeze(3)], 3) # N, nr_boxes, R*R, 2, 4
        
        rel_d = rel_score.unsqueeze(2).repeat(1, 1, R**2, 1) # N, nr_boxes, R*R, num_rel_class
        
        
        score_d = score_d.view(N, -1) # N, nr_boxes * R*R * 2
        rel_d = rel_d.view(N, nr_boxes * R * R, -1)

        for i in range(N):
            per_tri_rel_scores, _ = rel_d[i].max(dim=-1) # nr_boxes * R*R
            triple_scores = per_tri_rel_scores * score_d[i][0::2] * score_d[i][1::2]
            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
            sorting_idx = sorting_idx[:max_recall]
            tri_mask[i, sorting_idx] = True
        
        tri_mask = tri_mask.view(N * nr_boxes * R * R)
        boxes_d = boxes_d.view(N * nr_boxes * R * R, 8) # N, nr_boxes, R, R, 8
        class_d = class_d.view(N * nr_boxes * R * R, 2) # N, nr_boxes, R, R, 2
        score_d = score_d.view(N * nr_boxes * R * R, 2) # N, nr_boxes, R, R, 2
        
    return score_d, class_d, boxes_d, tri_mask

    
class Rel_head(nn.Module):
    def __init__(self, dim_in, dim_q, dim_v, dim_in_submap, num_class, 
                num_rel_class, num_pre_cls_rel_layers, 
                dim_feedforward=2048, hidden_dim=64, 
                num_head=8, dropout=0.1, 
                scale_clamp=math.log(100000.0 / 16), 
                bbox_weights=(2.0, 2.0, 1.0, 1.0),
                use_focal=True, prior_prob=0.01, 
                dynamic_conv_num=2, resolution=7,
                num_pre_cls_layers=1, num_pre_reg_layers=3,
                use_cross_obj_feat_fusion=True, 
                use_siamese_head=False, freq_bias=None,
                embed_vecs=None, obj_word_embed_dim=None,
                posi_encode_dim=None, dim_in_rel=256, 
                enable_rel_x2y=False, num_rel_proposals=150,
                dim_rank=128, enable_mask_branch=False, 
                enable_query_reverse=False, use_refine_obj=True,
                enable_auxiliary_branch=False,
                start_auxi_branch=0,
                causal_effect_analsis=False,
                use_only_obj2rel=False,
                diable_rel_fusion=False,
                enable_one_rel_conv=False,
                num_batch_reduction=250,
                enable_batch_reduction=False,
                cur_layer_id=0,
                use_last_relness=False,):
        super(Rel_head, self).__init__()
        self.num_head = num_head
        self.dim_in = dim_in
        self.dim_q = dim_q
        self.dim_v = dim_v
        self.dim_in_submap = dim_in_submap
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights
        self.use_focal = use_focal
        self.enable_query_reverse = enable_query_reverse
        self.num_class = num_class
        self.num_rel_class = num_rel_class
        self.enable_auxiliary_branch = enable_auxiliary_branch
        self.start_auxi_branch = start_auxi_branch
        self.use_refine_obj = use_refine_obj
        self.causal_effect_analsis = causal_effect_analsis
        self.use_only_obj2rel = use_only_obj2rel
        self.diable_rel_fusion = diable_rel_fusion
        self.enable_one_rel_conv = enable_one_rel_conv
        self.enable_batch_reduction = enable_batch_reduction
        self.cur_layer_id = cur_layer_id
        if self.use_focal: 
            self.num_class -= 1 ###!!!
            self.num_rel_class -= 1
        
        self.use_last_relness = use_last_relness
        #if use_last_relness:
        #    self.num_rel_class += 1
        
        self.self_atten_rel = \
            Instance_self_disentangle_atten(dim_in_rel, dim_q, dim_in_rel, 
                num_head=num_head, dropout=dropout, use_disentangle=False) #
        
        if enable_one_rel_conv:
            if enable_batch_reduction:
                self.dynamic_conv_rel = \
                    Shrink_Dynamic_conv(dim_in_rel, dim_in_submap, vh=3 * resolution, \
                        vw=resolution, hidden_dim=hidden_dim, dropout=dropout)
            else:
                self.dynamic_conv_rel = \
                    Dynamic_conv(dim_in_rel, dim_in_submap, vh=3 * resolution, 
                        vw=resolution, hidden_dim=hidden_dim, 
                        dropout=dropout, dynamic_conv_num=dynamic_conv_num)
        else:
            self.dynamic_conv_rel = \
                Dynamic_conv(dim_in_rel, dim_in_submap, vh=resolution, hidden_dim=hidden_dim, 
                    dropout=dropout, dynamic_conv_num=dynamic_conv_num)
            self.dynamic_conv_rel_obj = \
                Dynamic_conv(dim_in_rel, dim_in_submap, vh=2 * resolution, 
                    vw=resolution, hidden_dim=hidden_dim, 
                    dropout=dropout, dynamic_conv_num=dynamic_conv_num)
        
        if not self.diable_rel_fusion:
            self.combine_rel_s = nn.Sequential(
                    nn.Linear(dim_in, dim_in // 4),
                    nn.LayerNorm(dim_in // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim_in // 4, dim_in_rel),)
            self.combine_rel_o = copy.deepcopy(self.combine_rel_s)
            self.layer_norm_fusion = nn.LayerNorm(dim_in_rel)
        
        self.FFN_rel = nn.Sequential(
            nn.Linear(dim_in_rel, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, dim_in_rel),) #
        self.layer_norm_rel = nn.LayerNorm(dim_in_rel)
        
        self.cls_rel = nn.Sequential()
        for i in range(0, 3 * num_pre_cls_rel_layers, 3):
            self.cls_rel.add_module(str(i), nn.Linear(dim_in_rel, dim_in_rel, False))
            self.cls_rel.add_module(str(i+1), nn.LayerNorm(dim_in_rel))
            if i+2 < 3 * num_pre_cls_rel_layers - 1:
                self.cls_rel.add_module(str(i+2), nn.ReLU(inplace=True))
        self.cls_rel_logits = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(dim_in_rel, self.num_rel_class))
        
        self.freq_bias_w=None
        if freq_bias is not None:
            self.freq_bias_w = freq_bias
            self.prior_score = nn.Linear(dim_in_rel, self.num_rel_class, bias=False)
            
        
        self.obj_word_embed = None
        if embed_vecs is not None:
            self.obj_word_embed = nn.Embedding(num_class, obj_word_embed_dim)
            with torch.no_grad():
                self.obj_word_embed.weight.copy_(embed_vecs, non_blocking=True)

            self.obj_embed_w = nn.Linear(obj_word_embed_dim, dim_in)
        
        self.posi_encode_dim = posi_encode_dim
        if posi_encode_dim is not None:
            self.posi_embedding = nn.Sequential(
                    nn.Linear(posi_encode_dim * 8, posi_encode_dim * 8 // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(posi_encode_dim * 8 // 2, dim_in_rel),)
        
        self.enable_rel_x2y = enable_rel_x2y
        if enable_rel_x2y:
            cls = nn.Sequential()
            for i in range(0, 3 * num_pre_cls_layers, 3):
                cls.add_module(str(i), nn.Linear(dim_in, dim_in, False))
                cls.add_module(str(i+1), nn.LayerNorm(dim_in))
                if i+2 < 3 * num_pre_cls_layers - 1:
                    cls.add_module(str(i+2), nn.ReLU(inplace=True))
            
            self.cls_rel_obj_s = copy.deepcopy(cls)
            self.cls_rel_obj_o = copy.deepcopy(cls)
            self.cls_rel_logits_obj = nn.Sequential(
                                nn.ReLU(inplace=True),
                                nn.Linear(dim_in, self.num_rel_class),)
        
        self.register_buffer("averge_rel_feat", torch.zeros(dim_in_rel))
        self.average_ratio = 0.0005
        
    def forward(self, class_logits, pred_bboxes, pro_features, roi_features, \
      so_class_logits, pred_bboxes_so, so_pro_features, so_roi_features, \
      mask_class_logits, ent_mask_features, rel_features, img_size, pooler, features, \
      posi_encoding, disable_union=False, disable_forward_x2y=False, causal_conducting=False, \
      features_detach=None):
        
        rel_pred_logits, rel_features, prod_pred, obj2rel_logits = \
            self.relation_inference(class_logits, pred_bboxes, pro_features, roi_features, \
                so_class_logits, pred_bboxes_so, so_pro_features, so_roi_features, \
                mask_class_logits, ent_mask_features, rel_features, img_size, pooler, \
                features, posi_encoding, disable_union=disable_union, \
                disable_forward_x2y=disable_forward_x2y, causal_conducting=causal_conducting, \
                features_detach=features_detach)
        
        
        if self.enable_query_reverse:
            assert self.enable_rel_x2y, 'Not enable_rel_x2y'
            assert self.freq_bias_w is None, 'Using freq_bias_w'
            if self.use_refine_obj:
                so_class_logits, pred_bboxes_so, rel_pred_logits, \
                obj2rel_logits, mask_class_logits = \
                    self.reverse_pair(so_class_logits, pred_bboxes_so, rel_pred_logits, \
                        obj2rel_logits, rel_pred_logits_prime, ent_feat, mask_class_logits)
            else:
                class_logits, pred_bboxes, rel_pred_logits, \
                obj2rel_logits, mask_class_logits = \
                    self.reverse_pair(class_logits, pred_bboxes, rel_pred_logits, \
                        obj2rel_logits, rel_pred_logits_prime, ent_feat, mask_class_logits)
        
        return class_logits, pred_bboxes, rel_pred_logits, pro_features, rel_features, \
            prod_pred, obj2rel_logits, roi_features, mask_class_logits, ent_mask_features, \
            so_class_logits, pred_bboxes_so, so_pro_features
    
    def relation_inference(self, class_logits, pred_bboxes, pro_features, roi_features, \
      so_class_logits, pred_bboxes_so, so_pro_features, so_roi_features, \
      mask_class_logits, ent_mask_features, rel_features, img_size, pooler, features, \
      posi_encoding, disable_union=False, disable_forward_x2y=False, causal_conducting=False, \
      features_detach=None):
        
        N, nr_boxes = pro_features.shape[:2]
        nr_boxes_so = nr_boxes
        dim_in_submap = self.dim_in_submap
        if self.use_refine_obj:
            nr_boxes_so = so_pro_features.shape[1]
        
        if self.use_refine_obj:
            detached_pred_bboxes = pred_bboxes_so.detach()
            ent_feat = so_pro_features.detach()
            ent_feat = ent_feat.contiguous().view(N * nr_boxes_so * 2, -1)
            
            en_roi_feat_map = so_roi_features.detach() # B*100*2, dim_in, 49
        else:
            detached_pred_bboxes = pred_bboxes.detach()
            ent_feat = pro_features.detach()
            ent_feat = ent_feat.contiguous().view(N * nr_boxes * 2, -1)
        
            en_roi_feat_map = roi_features.detach() # B*100*2, dim_in, 49
            #en_roi_feat_map = roi_features
        
        
        if self.enable_rel_x2y and (self.training or (not disable_forward_x2y)):
            ent_feat_rel_s = self.cls_rel_obj_s(ent_feat[0::2])
            ent_feat_rel_o = self.cls_rel_obj_o(ent_feat[1::2])
            obj2rel_logits = self.fusion(ent_feat_rel_s, ent_feat_rel_o)
            obj2rel_logits = self.cls_rel_logits_obj(obj2rel_logits)
        else:
            obj2rel_logits = torch.empty(rel_pred_logits.shape[0], 0)
            obj2rel_logits = obj2rel_logits.view(N, nr_boxes_so, -1)
        
        if self.use_only_obj2rel:
            rel_pred_logits = obj2rel_logits
            
            obj2rel_logits = obj2rel_logits.view(N, nr_boxes_so, -1)
            prod_pred = rel_pred_logits.view(N, nr_boxes_so, -1)
            rel_pred_logits = rel_pred_logits.view(N, nr_boxes_so, -1)
            rel_features = rel_features.view(N, nr_boxes_so, -1)
        
            return rel_pred_logits, rel_features, prod_pred, obj2rel_logits
        
        
        if self.freq_bias_w is not None:
            pred_ent_label = class_logits.detach()
            pred_ent_label = pred_ent_label.contiguous().view(N * nr_boxes * 2, -1)
            if self.use_focal:
                pred_ent_label = torch.sigmoid(pred_ent_label)
            else:
                pred_ent_label = F.softmax(pred_ent_label, dim=-1)
            pred_ent_label_score, pred_ent_label = torch.max(pred_ent_label, dim=-1)
            if self.use_focal:
                pred_ent_label += 1
        
        
        if self.training:
            self.averge_rel_feat = self.moving_average(self.averge_rel_feat, \
                rel_features.contiguous().view(N * nr_boxes_so, -1))
        
        
        rel_features = rel_features.view(N, nr_boxes_so, -1)
        rel_features = self.self_atten_rel(
            rel_features, rel_features, v=rel_features)
        
        rel_features = rel_features.contiguous().view(N * nr_boxes_so, -1)
        
        
        en_roi_feat_map = en_roi_feat_map.view(N, nr_boxes_so, 2, dim_in_submap, -1)
        en_roi_feat_map = en_roi_feat_map.permute(0, 1, 3, 2, 4).contiguous()
        en_roi_feat_map = en_roi_feat_map.view(N * nr_boxes_so, dim_in_submap, -1)
        if not self.enable_one_rel_conv:
            rel_features = self.dynamic_conv_rel_obj(rel_features, en_roi_feat_map)    
        
        proposal_boxes_union = list()
        for b in range(N):
            union_boxes = detached_pred_bboxes[b].view(-1, 4)
            union_boxes = self.get_rel_rois(union_boxes[0::2], union_boxes[1::2])
            proposal_boxes_union.append(BoxList(union_boxes, img_size[b]))
        
        if features_detach is not None: union_roi_features = pooler(features_detach, proposal_boxes_union)
        else: union_roi_features = pooler(features, proposal_boxes_union)
        union_roi_features = union_roi_features.contiguous().view(N * nr_boxes_so, dim_in_submap, -1)
        if not self.enable_one_rel_conv:
            rel_features = self.dynamic_conv_rel(rel_features, union_roi_features)
        else:
            cat_union_roi_features = torch.cat([en_roi_feat_map, union_roi_features], -1).to(union_roi_features.device)
            rel_features = self.dynamic_conv_rel(rel_features, cat_union_roi_features)
        
        posi_encode_dim = self.posi_encode_dim

        if not self.diable_rel_fusion:
            if posi_encoding is not None:
                posi_embed = posi_encoding(detached_pred_bboxes)
                posi_cls = self.posi_embedding(posi_embed.view(-1, posi_encode_dim * 8))
                rel_features = rel_features + posi_cls    
            
            rel_features = rel_features.contiguous().view(N * nr_boxes_so, -1)
            ent_feat_in = self.combine_rel_s(ent_feat[0::2]) + \
                        self.combine_rel_o(ent_feat[1::2])
            rel_features = self.layer_norm_fusion(rel_features + ent_feat_in)

        rel_features = self.layer_norm_rel(rel_features + self.FFN_rel(rel_features))
        rel_pred_logits_pre = self.cls_rel(rel_features)
        rel_pred_logits_prime = self.cls_rel_logits(rel_pred_logits_pre)
        rel_pred_logits = rel_pred_logits_prime
        
        prod_pred = rel_pred_logits_prime
        prod_pred = prod_pred.view(N, nr_boxes_so, -1)
        
        if self.enable_rel_x2y and (self.training or (not disable_forward_x2y)):
            rel_pred_logits = rel_pred_logits + obj2rel_logits
            obj2rel_logits = obj2rel_logits.view(N, nr_boxes_so, -1)
        else:
            obj2rel_logits = torch.empty(rel_pred_logits.shape[0], 0)
            obj2rel_logits = obj2rel_logits.view(N, nr_boxes_so, -1)
        
        
        rel_pred_logits = rel_pred_logits.view(N, nr_boxes_so, -1)
        rel_features = rel_features.view(N, nr_boxes_so, -1)
        
        return rel_pred_logits, rel_features, prod_pred, obj2rel_logits
    
    def reverse_pair(self, class_logits, pred_bboxes, rel_pred_logits, obj2rel_logits, \
      rel_pred_logits_prime, ent_feat, mask_class_logits=None):
        N, nr_boxes = class_logits.shape[:2]
        nr_rel_boxes = rel_pred_logits.shape[1]
        class_logits_rvs = class_logits.view(N, nr_boxes, 2, -1)
        class_logits_rvs = torch.flip(class_logits_rvs, dims=[2]) #class_logits_rvs[:, :, [1,0], :]
        class_logits_rvs = class_logits_rvs.view(N, nr_boxes, -1)
        pred_bboxes_rvs = pred_bboxes.view(N, nr_boxes, 2, -1)
        pred_bboxes_rvs = torch.flip(pred_bboxes_rvs, dims=[2]) #pred_bboxes_rvs[:, :, [1,0], :]
        pred_bboxes_rvs = pred_bboxes_rvs.view(N, nr_boxes, -1)
        
        class_logits = torch.cat([class_logits, class_logits_rvs], 1).to(class_logits.device)
        pred_bboxes = torch.cat([pred_bboxes, pred_bboxes_rvs], 1).to(pred_bboxes.device)
        
        if mask_class_logits is not None:
            mask_class_logits_rvs = mask_class_logits.view(N, nr_rel_boxes, 2, -1)
            mask_class_logits_rvs = torch.flip(mask_class_logits_rvs, dims=[2]) #mask_class_logits_rvs[:, :, [1,0], :]
            mask_class_logits_rvs = mask_class_logits_rvs.view(N, nr_rel_boxes, -1)
            mask_class_logits = torch.cat([mask_class_logits, mask_class_logits_rvs], 1).to(mask_class_logits.device)
        
        rel_pred_logits_prime = rel_pred_logits_prime.view(N * nr_rel_boxes, -1)
        
        ent_feat_rel_s = self.cls_rel_obj_s(ent_feat[1::2])
        ent_feat_rel_o = self.cls_rel_obj_o(ent_feat[0::2])
        obj2rel_logits_rvs = self.fusion(ent_feat_rel_s, ent_feat_rel_o)
        obj2rel_logits_rvs = self.cls_rel_logits_obj(obj2rel_logits_rvs)
        rel_pred_logits_rvs = rel_pred_logits_prime + obj2rel_logits_rvs
        obj2rel_logits_rvs = obj2rel_logits_rvs.view(N, nr_rel_boxes, -1)
        obj2rel_logits = torch.cat([obj2rel_logits, obj2rel_logits_rvs], 1).to(class_logits.device)
        
        rel_pred_logits_rvs = rel_pred_logits_rvs.view(N, nr_rel_boxes, -1)
        rel_pred_logits = torch.cat([rel_pred_logits, rel_pred_logits_rvs], 1).to(class_logits.device)
        return class_logits, pred_bboxes, rel_pred_logits, obj2rel_logits, mask_class_logits
    
    def get_rel_rois(self, rois1, rois2):
        device_id = rois1.device
        rois1 = rois1.data.cpu().numpy()
        rois2 = rois2.data.cpu().numpy()
        xmin = np.minimum(rois1[:, 0], rois2[:, 0])
        ymin = np.minimum(rois1[:, 1], rois2[:, 1])
        xmax = np.maximum(rois1[:, 2], rois2[:, 2])
        ymax = np.maximum(rois1[:, 3], rois2[:, 3])
        ans = np.vstack((xmin, ymin, xmax, ymax)).transpose()
        ans = torch.from_numpy(ans).to(device_id)
        return ans
    
    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder
    
    def fusion(self, x, y):
        return F.relu(x + y, inplace=True) - (x - y) ** 2
    

class obj_feature_fuser(nn.Module):
    def __init__(self, dim_in, dim_q, dim_v, dim_in_submap, num_class, 
            num_rel_class, num_pre_cls_rel_layers, 
            num_head=8, dim_feedforward=2048, hidden_dim=64, 
            dropout=0.1, resolution=7, scales=1./16., sampling_ratio=0., cat_all_levels=False,
            scale_clamp = math.log(100000.0 / 16), bbox_weights=(2.0, 2.0, 1.0, 1.0), 
            use_focal=True, prior_prob=0.01, stack_num=6, rel_stack_num=6,
            return_intermediate=True, pooler_in_channels=512, dynamic_conv_num=2, 
            num_pre_cls_layers=1, num_pre_reg_layers=3,
            use_cross_obj_feat_fusion=True, prior_prob_rel=0.01,
            use_siamese_head=False, 
            posi_encode_dim=256, posi_embed_dim=256, obj_word_embed_dim=200,
            obj_classes_list=[], word_embed_weight_path='', freq_bias=None,
            dim_in_rel=256, enable_rel_x2y=False, num_rel_proposals=150, 
            dim_rank=128, enable_mask_branch=False, enable_query_reverse=False,
            use_refine_obj=True, enable_auxiliary_branch=False, new_arrival_ent_max_num=2,
            pair_group=1, start_auxi_branch=0, use_cross_rank=False,
            causal_effect_analsis=False, use_fusion_kq_selfatten=False,
            use_only_obj2rel=False, diable_rel_fusion=False, dim_ent_pre_cls=1024, 
            dim_ent_pre_reg=1024, enable_one_rel_conv=False, num_batch_reduction=250, 
            enable_batch_reduction=False, use_pure_objdet=False, use_last_relness=False,
            num_proposals_list=None):
        super(obj_feature_fuser, self).__init__()
        self.return_intermediate = return_intermediate
        
        self.use_last_relness = use_last_relness
        if use_last_relness:
            num_rel_class += 1
        
        if word_embed_weight_path == '':
            embed_vecs = None
        else:
            embed_vecs = obj_edge_vectors(obj_classes_list, wv_type=self.cfg.MODEL.TEXT_EMBEDDING,  wv_dir=word_embed_weight_path, wv_dim=obj_word_embed_dim)
        
        if posi_encode_dim is not None:
            self.posi_encoding = PositionalEncoding(posi_encode_dim)
        else:
            self.posi_encoding = None
        
        self.num_proposals_list = num_proposals_list
        
        pair_roi_head = Pair_roi_head(
            dim_in, dim_q, dim_v, dim_in_submap, 
            num_class, num_rel_class, num_pre_cls_rel_layers, 
            dim_feedforward=dim_feedforward, 
            hidden_dim=hidden_dim, num_head=num_head, dropout=dropout, 
            scale_clamp=scale_clamp, bbox_weights=bbox_weights,
            use_focal=use_focal, prior_prob=prior_prob, 
            dynamic_conv_num=dynamic_conv_num, resolution=resolution,
            num_pre_cls_layers=num_pre_cls_layers,
            num_pre_reg_layers=num_pre_reg_layers,
            use_cross_obj_feat_fusion=use_cross_obj_feat_fusion,
            use_siamese_head=use_siamese_head,
            freq_bias=freq_bias, embed_vecs=embed_vecs,
            obj_word_embed_dim=obj_word_embed_dim,
            posi_encode_dim=posi_encode_dim, dim_in_rel=dim_in_rel,
            enable_rel_x2y=enable_rel_x2y, num_rel_proposals=num_rel_proposals,
            dim_rank=dim_rank, enable_mask_branch=enable_mask_branch,
            enable_query_reverse=enable_query_reverse, use_refine_obj=use_refine_obj, 
            enable_auxiliary_branch=enable_auxiliary_branch,
            start_auxi_branch=start_auxi_branch,
            pair_group=pair_group,
            use_cross_rank=use_cross_rank,
            causal_effect_analsis=causal_effect_analsis,
            use_fusion_kq_selfatten=use_fusion_kq_selfatten,
            dim_ent_pre_cls=dim_ent_pre_cls,
            dim_ent_pre_reg=dim_ent_pre_reg,
            num_batch_reduction=num_batch_reduction,
            enable_batch_reduction=enable_batch_reduction,
            cur_layer_id=0, new_arrival_ent_max_num=new_arrival_ent_max_num, \
            use_pure_objdet=use_pure_objdet)
        
        self.use_pure_objdet = use_pure_objdet
        if not use_pure_objdet:
            rel_head = Rel_head(
                dim_in, dim_q, dim_v, dim_in_submap, 
                num_class, num_rel_class, num_pre_cls_rel_layers, 
                dim_feedforward=dim_feedforward, 
                hidden_dim=hidden_dim, num_head=num_head, dropout=dropout, 
                scale_clamp=scale_clamp, bbox_weights=bbox_weights,
                use_focal=use_focal, prior_prob=prior_prob, 
                dynamic_conv_num=dynamic_conv_num, resolution=resolution,
                num_pre_cls_layers=num_pre_cls_layers,
                num_pre_reg_layers=num_pre_reg_layers,
                use_cross_obj_feat_fusion=use_cross_obj_feat_fusion,
                use_siamese_head=use_siamese_head,
                freq_bias=freq_bias, embed_vecs=embed_vecs,
                obj_word_embed_dim=obj_word_embed_dim,
                posi_encode_dim=posi_encode_dim, dim_in_rel=dim_in_rel,
                enable_rel_x2y=enable_rel_x2y, num_rel_proposals=num_rel_proposals,
                dim_rank=dim_rank, enable_mask_branch=enable_mask_branch,
                enable_query_reverse=enable_query_reverse, use_refine_obj=use_refine_obj, 
                enable_auxiliary_branch=enable_auxiliary_branch,
                start_auxi_branch=start_auxi_branch,
                causal_effect_analsis=causal_effect_analsis,
                use_only_obj2rel=use_only_obj2rel,
                diable_rel_fusion=diable_rel_fusion,
                enable_one_rel_conv=enable_one_rel_conv,
                num_batch_reduction=num_batch_reduction,
                enable_batch_reduction=enable_batch_reduction,
                cur_layer_id=0, use_last_relness=use_last_relness,)
        
        self.head_series = _get_clones(pair_roi_head, stack_num)
        if not use_pure_objdet:
            self.rel_head_series = _get_clones(rel_head, rel_stack_num)
        
        self.enable_query_reverse = enable_query_reverse
        self.use_refine_obj = use_refine_obj
        self.enable_auxiliary_branch = enable_auxiliary_branch
        self.new_arrival_ent_max_num = new_arrival_ent_max_num
        self.start_auxi_branch = start_auxi_branch
        self.pair_group = pair_group
        self.use_cross_rank = use_cross_rank
        self.causal_effect_analsis = causal_effect_analsis
        
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            in_channels=pooler_in_channels,
            cat_all_levels=cat_all_levels,)
        self.pooler = pooler
        
        self.use_focal = use_focal
        self.num_class = num_class
        self.num_rel_class = num_rel_class
        if use_focal:
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.bias_value_rel = -math.log((1 - prior_prob) / prior_prob)
            self.num_class -= 1
            self.num_rel_class -= 1
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        # init all parameters.
        for n, p in self.named_parameters():
            if n.find('freq_bias') >= 0:
                print('freq_bias found.')
                continue
            if n.find('obj_word_embed') >= 0:
                print('obj_word_embed found.')
                continue
            if n.find('posi_encoding') >= 0:
                print('posi_encoding found.')
                continue    
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            if self.use_focal and p.shape[-1] == self.num_class:
                nn.init.constant_(p, self.bias_value)
            if self.use_focal and p.shape[-1] == self.num_rel_class:
                nn.init.constant_(p, self.bias_value_rel)
            if p.shape[-1] == 1:
                nn.init.constant_(p, self.bias_value)
            #nn.init.normal_(p, std=0.01)
    
    def forward(self, features, init_bboxes, init_features, init_rel_features, \
      img_size, init_ent_mask_features=None, only_ent=False, init_so_pro_features=None, init_so_bboxes=None, \
      auxiliary_rel=None, causal_conducting=False, ori_images=None):
        """
            :param init_bboxes: (nr_boxes, 8)
            :param init_features: (N, nr_boxes, 2 * dim_in)
            
            return: [1, N, nr_boxes (* 2), ...]
        """
        
        #exam_svd(self)
        
        class_logits, pred_bboxes, roi_features = None, None, None
        so_class_logits, pred_bboxes_so, so_roi_features = None, None, None
        
        inter_mask_class_logits = []
        inter_class_logits = []
        inter_rel_pred_logits = []
        inter_rel_prod_pred = []
        inter_obj2rel_logits = []
        inter_pred_bboxes = []
        
        inter_so_pred_bboxes = []
        inter_so_class_logits = []
        
        
        
        bs = len(features[0])
        
        
        bboxes = init_bboxes
        init_features = init_features[None].repeat(bs, 1, 1)
        proposal_features = init_features.clone()
        
        init_rel_features = init_rel_features[None].repeat(bs, 1, 1)
        rel_features = init_rel_features.clone()
        
        
        if init_ent_mask_features is not None:
            init_ent_mask_features = init_ent_mask_features[None].repeat(bs, 1, 1)
            ent_mask_features = init_ent_mask_features.clone()
        else:
            ent_mask_features = torch.tensor([])
            
        if init_so_pro_features is not None:
            init_so_pro_features = init_so_pro_features[None].repeat(bs, 1, 1)
            so_pro_features = init_so_pro_features.clone()
        else:
            so_pro_features = torch.tensor([])
        so_bboxes = init_so_bboxes
        
        auxiliary_rel_ = auxiliary_rel
        if auxiliary_rel is not None:
            auxiliary_rel = auxiliary_rel[None].repeat(bs, 1, 1)
            auxiliary_rel_ = auxiliary_rel.clone()

        rel_pred_logits, rel_prod_pred, obj2rel_logits = None, None, None
        if self.enable_auxiliary_branch:
            vis_table = torch.zeros(bs, bboxes.shape[1], dtype=torch.bool)
            rel_pred_logits = torch.ones(bs, rel_features.shape[1], self.num_rel_class).to(rel_features.device)
        
        counterfact_rel_logits = None

        
        tuple_img_size = [tuple(img_size[b].data.cpu().numpy()) for b in range(img_size.shape[0])]
        
        inter_tri_score = []
        

        if self.use_pure_objdet:
            inter_proposal_features = []
            inter_roi_features = []
            for idx, rcnn_head in enumerate(self.head_series):
                class_logits, pred_bboxes, proposal_features, roi_features, \
                mask_class_logits, ent_mask_features, \
                so_class_logits, pred_bboxes_so, so_pro_features, so_roi_features = \
                    rcnn_head(features, bboxes, proposal_features, self.pooler, \
                                tuple_img_size, rel_features=rel_features, \
                                posi_encoding=self.posi_encoding, only_ent=True, \
                                ent_mask_features=ent_mask_features,
                                so_pro_features=so_pro_features, 
                                so_bboxes=so_bboxes,
                                auxiliary_rel=auxiliary_rel_,
                                causal_conducting=causal_conducting,
                                flg_add_auxi=False)
                N, nr_boxes = class_logits.shape[:2]
                inter_class_logits.append(class_logits.view(N, 2*nr_boxes, -1))
                inter_pred_bboxes.append(pred_bboxes.view(N, 2*nr_boxes, -1))
                inter_proposal_features.append(proposal_features.view(N, 2*nr_boxes, -1))
                inter_roi_features.append(roi_features.view(N, 2*nr_boxes, -1))
                bboxes = pred_bboxes.detach()
            
            return inter_class_logits, inter_pred_bboxes, inter_proposal_features, inter_roi_features
        
        features_detach = tuple([feature.detach() for feature in features])
        
        for idx, (rcnn_head, rel_rcnn_head) in enumerate(zip(self.head_series, self.rel_head_series)):
            flg_add_auxi = False
            
            if self.num_proposals_list is not None:
                if idx > 0 and self.num_proposals_list[idx] < self.num_proposals_list[idx-1]:
                    so_pro_features, so_bboxes, rel_features = \
                        self.relative_rank_embedding(so_pro_features, \
                            so_bboxes, rel_features, so_class_logits, \
                            rel_pred_logits, self.num_proposals_list[idx], idx)
            
            class_logits, pred_bboxes, proposal_features, roi_features, \
            mask_class_logits, ent_mask_features, \
            so_class_logits, pred_bboxes_so, so_pro_features, so_roi_features = \
                rcnn_head(features, bboxes, proposal_features, self.pooler, \
                            tuple_img_size, rel_features=rel_features, \
                            posi_encoding=self.posi_encoding, only_ent=only_ent, \
                            ent_mask_features=ent_mask_features,
                            so_pro_features=so_pro_features, 
                            so_bboxes=so_bboxes,
                            auxiliary_rel=auxiliary_rel_,
                            causal_conducting=causal_conducting,
                            flg_add_auxi=flg_add_auxi, 
                            features_detach=features_detach)
            
            if self.training and idx >= self.start_auxi_branch and self.enable_auxiliary_branch:
                assert self.use_refine_obj, 'No use_refine_obj'
                flg_add_auxi = True
                #so_pro_features, so_bboxes, _, _, _, _ = \
                so_pro_features, pred_bboxes_so, \
                so_class_logits, rel_features, so_roi_features, vis_table = \
                    assembling_auxiliary_query(
                        auxiliary_rel_, \
                        class_logits, pred_bboxes, proposal_features, roi_features, \
                        so_class_logits, pred_bboxes_so, so_pro_features, so_roi_features, \
                        rel_pred_logits, rel_features, vis_table, \
                        new_arrival_ent_max_num=self.new_arrival_ent_max_num, \
                        R=3, max_recall=100, iou_thres=0.5)
            
            if not only_ent:
                class_logits, pred_bboxes, rel_pred_logits, proposal_features, rel_features, \
                rel_prod_pred, obj2rel_logits, roi_features, mask_class_logits, ent_mask_features, \
                so_class_logits, pred_bboxes_so, so_pro_features = \
                    rel_rcnn_head(class_logits, pred_bboxes, proposal_features, roi_features, \
                        so_class_logits, pred_bboxes_so, so_pro_features, so_roi_features, \
                        mask_class_logits, ent_mask_features, rel_features, tuple_img_size, \
                        self.pooler, features, self.posi_encoding, causal_conducting=causal_conducting, \
                        features_detach=features_detach)
            

            
            if self.return_intermediate:
                if self.use_refine_obj and self.enable_auxiliary_branch:
                    if self.training:
                        #inter_class_logits.append(class_logits)
                        #inter_pred_bboxes.append(pred_bboxes)
                        inter_class_logits.append(class_logits[:, 0::2, :])
                        inter_pred_bboxes.append(pred_bboxes[:, 0::2, :])
                    else:
                        inter_class_logits.append(class_logits)
                        inter_pred_bboxes.append(pred_bboxes)
                else:
                    inter_class_logits.append(class_logits)
                    inter_pred_bboxes.append(pred_bboxes)
                
                inter_rel_pred_logits.append(rel_pred_logits)
                inter_obj2rel_logits.append(obj2rel_logits)
                inter_rel_prod_pred.append(rel_prod_pred)
                inter_mask_class_logits.append(mask_class_logits)
                inter_so_class_logits.append(so_class_logits)
                inter_so_pred_bboxes.append(pred_bboxes_so)
                
                
            if self.enable_query_reverse:
                if self.use_refine_obj:
                    nr_boxes = so_pro_features.shape[1]
                    so_bboxes = pred_bboxes_so[:, :nr_boxes, :].detach()
                    bboxes = pred_bboxes.detach()
                else:
                    nr_boxes = proposal_features.shape[1]
                    bboxes = pred_bboxes[:, :nr_boxes, :].detach()
                    so_bboxes = pred_bboxes_so.detach()
            else:
                if self.training or (not self.use_refine_obj):
                    bboxes = pred_bboxes.detach()
                so_bboxes = pred_bboxes_so.detach()
        
        
        if self.return_intermediate:
            return inter_class_logits, \
                    inter_pred_bboxes, \
                    inter_rel_pred_logits, \
                    inter_obj2rel_logits, \
                    inter_rel_prod_pred, \
                    inter_mask_class_logits, \
                    inter_so_class_logits, \
                    inter_so_pred_bboxes, \
                    counterfact_rel_logits

        return class_logits[None], pred_bboxes[None], \
            rel_pred_logits[None], obj2rel_logits[None], \
            rel_prod_pred[None], \
            mask_class_logits[None], \
            so_class_logits[None], \
            pred_bboxes_so[None], \
            counterfact_rel_logits[None]
    
    def relative_rank_embedding(self, so_pro_features, so_bboxes, rel_features, soobjlogit, rellogit, R, stage_idx):
        with torch.no_grad():
            N, nr_boxes_so = so_bboxes.shape[:2]
            soobjlogit = soobjlogit.contiguous().view(N, nr_boxes_so*2, -1)
            obj_score, _ = torch.max(soobjlogit, dim=-1)
            obj_score = F.sigmoid(obj_score)
            obj_score = obj_score.contiguous().view(N, nr_boxes_so, 2)
            obj_score = obj_score[:, :, 0] * obj_score[:, :, 1]
            rel_score, _ = torch.max(rellogit, dim=-1)
            rel_score = F.sigmoid(rel_score)
            tri_score = rel_score * obj_score
            _, selectid = torch.topk(tri_score, k=R, dim=-1)
            batchid = torch.arange(N).view(-1, 1).expand(N, R)
            selectid = selectid.view(-1)
            batchid = batchid.view(-1)
            so_pro_features = so_pro_features[batchid, selectid, :].view(N, len(selectid), -1)
            so_bboxes = so_bboxes[batchid, selectid, :].view(N, len(selectid), -1)
            rel_features = rel_features[batchid, selectid, :].view(N, len(selectid), -1)
        return so_pro_features, so_bboxes, rel_features

class Pair_roi_head(nn.Module):
    def __init__(self, dim_in, dim_q, dim_v, dim_in_submap, num_class, 
                    num_rel_class, num_pre_cls_rel_layers, 
                    dim_feedforward=2048, hidden_dim=64, 
                    num_head=8, dropout=0.1, 
                    scale_clamp=math.log(100000.0 / 16), 
                    bbox_weights=(2.0, 2.0, 1.0, 1.0),
                    use_focal=True, prior_prob=0.01, 
                    dynamic_conv_num=2, resolution=7,
                    num_pre_cls_layers=1, num_pre_reg_layers=3,
                    use_cross_obj_feat_fusion=True, 
                    use_siamese_head=False, freq_bias=None,
                    embed_vecs=None, obj_word_embed_dim=None,
                    posi_encode_dim=None, dim_in_rel=256, 
                    enable_rel_x2y=False, num_rel_proposals=150,
                    dim_rank=128, enable_mask_branch=False, 
                    enable_query_reverse=False, use_refine_obj=True,
                    enable_auxiliary_branch=False,
                    start_auxi_branch=0, pair_group=1,
                    use_cross_rank=False,
                    causal_effect_analsis=False,
                    use_fusion_kq_selfatten=False,
                    dim_ent_pre_cls=1024, dim_ent_pre_reg=1024,
                    num_batch_reduction=250,
                    enable_batch_reduction=False, cur_layer_id=0,
                    new_arrival_ent_max_num=5, use_pure_objdet=False):
        super(Pair_roi_head, self).__init__()
        self.num_head = num_head
        self.dim_in_rel = dim_in_rel
        self.dim_in = dim_in
        self.dim_q = dim_q
        self.dim_v = dim_v
        self.dim_in_submap = dim_in_submap
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights
        self.cur_layer_id = cur_layer_id
        self.new_arrival_ent_max_num = new_arrival_ent_max_num
        self.use_pure_objdet = use_pure_objdet
        
        self.use_focal = use_focal
        
        self.enable_query_reverse = enable_query_reverse
        
        self.num_class = num_class
        self.num_rel_class = num_rel_class
        if self.use_focal: 
            self.num_class -= 1 ###!!!
            self.num_rel_class -= 1
        
        self.self_atten = \
            Instance_self_disentangle_atten(dim_in, dim_q, dim_v, 
                num_head=num_head, dropout=dropout, use_disentangle=False) #
        
        if enable_batch_reduction:
            self.dynamic_conv = \
                Shrink_Dynamic_conv(dim_in, dim_in_submap, vh=resolution, \
                    hidden_dim=hidden_dim, dropout=dropout)
        else:
            self.dynamic_conv = \
                Dynamic_conv(dim_in, dim_in_submap, vh=resolution, hidden_dim=hidden_dim, 
                    dropout=dropout, dynamic_conv_num=dynamic_conv_num)
        
        self.use_cross_obj_feat_fusion = use_cross_obj_feat_fusion
        
        self.FFN = nn.Sequential(
            nn.Linear(dim_in, dim_feedforward),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_in),)
            #nn.Dropout(dropout, inplace=True),
            
        self.layer_norm = nn.LayerNorm(dim_in)
        
        self.cls = nn.Sequential()
        for i in range(0, 3 * num_pre_cls_layers, 3):
            if i == 0:
                self.cls.add_module(str(i), nn.Linear(dim_in, dim_ent_pre_cls, False))
            else:
                self.cls.add_module(str(i), nn.Linear(dim_ent_pre_cls, dim_ent_pre_cls, False))
            self.cls.add_module(str(i+1), nn.LayerNorm(dim_ent_pre_cls))
            if i+2 < 3 * num_pre_cls_layers - 1:
                self.cls.add_module(str(i+2), nn.ReLU(inplace=True))
        #self.cls.add_module(str(3 * num_pre_cls_layers), nn.Linear(dim_in, self.num_class))
        self.cls_logits = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(dim_ent_pre_cls, self.num_class),)
        
        self.reg = nn.Sequential()
        for i in range(0, 3 * num_pre_reg_layers, 3):
            if i == 0:
                self.reg.add_module(str(i), nn.Linear(dim_in, dim_ent_pre_reg, False))
            else:
                self.reg.add_module(str(i), nn.Linear(dim_ent_pre_reg, dim_ent_pre_reg, False))
            self.reg.add_module(str(i+1), nn.LayerNorm(dim_ent_pre_reg))
            self.reg.add_module(str(i+2), nn.ReLU(inplace=True))
        #self.reg.add_module(str(3 * num_pre_reg_layers), nn.Linear(dim_in, 4))
        self.reg_deltas = nn.Linear(dim_ent_pre_reg, 4)
        
        
        self.enable_mask_branch = enable_mask_branch
        if enable_mask_branch:
            self.self_atten_mask = \
                Instance_self_disentangle_atten(dim_in, dim_q, dim_v, 
                    num_head=num_head, dropout=dropout, use_disentangle=False)
            self.FFN_mask = nn.Sequential(
                nn.Linear(dim_rank, int(dim_feedforward / dim_in * dim_rank)),
                nn.ReLU(inplace=True),
                nn.Linear(int(dim_feedforward / dim_in * dim_rank), dim_rank),)
            self.layer_norm_mask = nn.LayerNorm(dim_rank)
            self.cls_mask = nn.Sequential()
            for i in range(0, 3 * num_pre_cls_layers, 3):
                self.cls_mask.add_module(str(i), nn.Linear(dim_rank, dim_rank, False))
                self.cls_mask.add_module(str(i+1), nn.LayerNorm(dim_rank))
                if i+2 < 3 * num_pre_cls_layers - 1:
                    self.cls_mask.add_module(str(i+2), nn.ReLU(inplace=True))
            self.cls_mask_logits = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(dim_rank, self.num_class),)
            
        self.use_refine_obj = use_refine_obj
        if self.use_refine_obj:
            self.use_fusion_kq_selfatten = use_fusion_kq_selfatten
            if use_fusion_kq_selfatten:
                self.more_spatial = nn.Sequential(
                    nn.Linear(posi_encode_dim * 8, dim_in * 2),)
                self.more_so_vec = nn.Sequential(
                        nn.Linear(dim_in * 2, dim_in * 2 // 4),
                        nn.LayerNorm(dim_in * 2 // 4),
                        nn.ReLU(inplace=True),
                        nn.Linear(dim_in * 2 // 4, dim_in * 2),)
        

        
        
        self.enable_auxiliary_branch = enable_auxiliary_branch
        self.start_auxi_branch = start_auxi_branch
        self.pair_group = pair_group
        self.use_cross_rank = use_cross_rank
        self.causal_effect_analsis = causal_effect_analsis
        
        self.obj_word_embed = None
        if embed_vecs is not None:
            self.obj_word_embed = nn.Embedding(num_class, obj_word_embed_dim)
            with torch.no_grad():
                self.obj_word_embed.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed_w = nn.Linear(obj_word_embed_dim, dim_in)
        
        self.average_ratio = 0.0005
        self.register_buffer("averge_proposal_feat", torch.zeros(dim_in))
        
        self.enable_batch_reduction = enable_batch_reduction
        
    def forward(self, features, bboxes, pro_features, pooler, img_size, \
      rel_features=None, posi_encoding=None, only_ent=False, ent_mask_features=None, \
      so_pro_features=None, so_bboxes=None, auxiliary_rel=None, causal_conducting=False, \
      flg_add_auxi=False, features_detach=None):
        """
            #:param bboxes: (N, nr_boxes, 4)
            #:param pro_features: (N, nr_boxes, dim_in)
            
            N == B
            
            :param bboxes: (N, nr_boxes, 8)
            :param pro_features: (N, nr_boxes, 2 * dim_in)
            
            return: (N * nr_boxes * 2, dim_in)
        """
        N, nr_boxes = pro_features.shape[:2]
        nr_boxes_so = nr_boxes
        dim_in_submap = self.dim_in_submap
        

        if self.training or (not self.use_refine_obj) or self.use_pure_objdet:
            # roi_feature.
            proposal_boxes = list()
            for b in range(N):
                proposal_boxes.append(BoxList(bboxes[b].view(-1, 4), img_size[b]))
            roi_features = pooler(features, proposal_boxes)
            roi_features = roi_features.view(
                N * nr_boxes * 2, dim_in_submap, -1) # B*100*2, dim_in, 49
        
        so_roi_features = None
        if self.use_refine_obj:
            bboxes_posi_embed = posi_encoding(bboxes).view(N, nr_boxes, -1)
        
        if self.training or (not self.use_refine_obj) or self.use_pure_objdet:
            # latent feature vector
            pro_features = pro_features.view(N, nr_boxes * 2, -1)
            selfatten_input_vec = pro_features
            
            
            pro_features = self.self_atten(
                selfatten_input_vec, selfatten_input_vec, v=pro_features)
            pro_features = pro_features.contiguous().view(N * nr_boxes * 2, -1)
            pro_features = self.dynamic_conv(pro_features, roi_features)
            pro_features = self.layer_norm(pro_features + self.FFN(pro_features))
            
            class_logits_pre = self.cls(pro_features)
            class_logits = self.cls_logits(class_logits_pre)
            if self.use_refine_obj or only_ent:
                bboxes_deltas_pre = self.reg(pro_features)
                bboxes_deltas = self.reg_deltas(bboxes_deltas_pre)
                pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))
            
            
        if only_ent:
            class_logits = class_logits.view(N, nr_boxes, -1)
            pred_bboxes = pred_bboxes.view(N, nr_boxes, -1)
            roi_features = roi_features.view(N, nr_boxes, -1)
            pro_features = pro_features.view(N, nr_boxes, -1)
            return class_logits, pred_bboxes, pro_features, roi_features, \
                torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), \
                torch.tensor([]), torch.tensor([])
                
        
        if self.use_refine_obj and self.obj_word_embed is not None:
            pro_max_cls_score, pro_cls_id = torch.max(class_logits, dim=-1)
            pro_cls_embed = self.obj_word_embed(pro_cls_id) * pro_max_cls_score.unsqueeze(-1)
            pro_cls_embed = self.obj_embed_w(pro_cls_embed)
        else:
            pro_cls_embed = 0.
        
        if not self.use_refine_obj:
            so_class_logits = torch.tensor([])
        
        
        if self.use_refine_obj:
            nr_boxes_so = so_pro_features.shape[1]
            so_bboxes_posi_embed = posi_encoding(so_bboxes).view(N, nr_boxes_so, -1)
            
            if self.use_fusion_kq_selfatten:
                so_pro_features = so_pro_features.view(N, nr_boxes_so, -1)
                cross_posi_embed = self.more_spatial(so_bboxes_posi_embed).view(N, nr_boxes_so * 2, -1)
                cross_so_vec = self.more_so_vec(so_pro_features).view(N, nr_boxes_so * 2, -1)
                so_pro_features = so_pro_features.view(N, nr_boxes_so * 2, -1)
                selfatten_input_vec = so_pro_features + cross_posi_embed + cross_so_vec
            else:
                so_pro_features = so_pro_features.view(N, nr_boxes_so * 2, -1)
                selfatten_input_vec = so_pro_features

            so_pro_features = self.self_atten(
                selfatten_input_vec, selfatten_input_vec, v=so_pro_features)
            
            
            so_proposal_boxes = list()
            for b in range(N):
                so_proposal_boxes.append(BoxList(so_bboxes[b].view(-1, 4), img_size[b]))
            if features_detach is not None: so_roi_features = pooler(features_detach, so_proposal_boxes)
            else: so_roi_features = pooler(features, so_proposal_boxes)
            so_roi_features = so_roi_features.view(
                N * nr_boxes_so * 2, dim_in_submap, -1)
            
            so_pro_features = so_pro_features.contiguous().view(N * nr_boxes_so * 2, -1)
            so_pro_features = self.dynamic_conv(so_pro_features, so_roi_features)
            so_pro_features = self.layer_norm(so_pro_features + self.FFN(so_pro_features))
            
            if self.training:
                self.averge_proposal_feat = \
                    self.moving_average(self.averge_proposal_feat, so_pro_features)
            elif self.causal_effect_analsis and causal_conducting:
                so_pro_features = self.averge_proposal_feat.view(1, -1).expand(N * nr_boxes_so * 2, -1)
                so_pro_features = so_pro_features.contiguous()
            
            so_class_logits_pre = self.cls(so_pro_features)
            
            
            so_class_logits = self.cls_logits(so_class_logits_pre)

            
        if self.enable_mask_branch:
            ent_mask_features = ent_mask_features.view(N, nr_boxes_so * 2, -1)
            ent_mask_features = self.self_atten_mask(ent_mask_features, ent_mask_features)
            
            if self.use_refine_obj:
                ent_mask_features = ent_mask_features.contiguous().view(N * nr_boxes_so * 2, -1)
                ent_mask_features = self.dynamic_conv(ent_mask_features, so_roi_features)
            else:
                ent_mask_features = ent_mask_features.contiguous().view(N * nr_boxes_so * 2, -1)
                ent_mask_features = self.dynamic_conv(ent_mask_features, roi_features)
            

            ent_mask_features = self.layer_norm_mask(ent_mask_features + self.FFN_mask(ent_mask_features))
            mask_class_logits_pre = self.cls_mask(ent_mask_features)
            mask_class_logits = self.cls_mask_logits(mask_class_logits_pre)
            
            mask_class_logits = mask_class_logits.view(N, nr_boxes_so, -1)
            ent_mask_features = ent_mask_features.view(N, nr_boxes_so, -1)
        else:
            mask_class_logits = torch.tensor([])
        
        if self.use_refine_obj:
            so_pro_features_reg = so_pro_features
            
            
            so_pro_features_reg = so_pro_features_reg.contiguous().view(N * nr_boxes_so * 2, -1)
            bboxes_deltas_pre2 = self.reg(so_pro_features_reg)
            bboxes_deltas2 = self.reg_deltas(bboxes_deltas_pre2)
        else:
            pro_features = pro_features.contiguous().view(N * nr_boxes * 2, -1)
            bboxes_deltas_pre2 = self.reg(pro_features)
            bboxes_deltas2 = self.reg_deltas(bboxes_deltas_pre2)
        
        if self.training or (not self.use_refine_obj):
            pro_features = pro_features.view(N, nr_boxes, -1)
            roi_features = roi_features.view(N, nr_boxes, -1)
        
        if self.use_refine_obj:
            if self.training:
                class_logits = class_logits.view(N, 2 * nr_boxes, -1)
                pred_bboxes = pred_bboxes.view(N, 2 * nr_boxes, -1)
            else:
                class_logits = None
                pred_bboxes = None
                roi_features = None
                
            so_class_logits = so_class_logits.view(N, nr_boxes_so, -1)
            so_pro_features = so_pro_features.view(N, nr_boxes_so, -1)
            pred_bboxes_so = self.apply_deltas(bboxes_deltas2, so_bboxes.view(-1, 4))
            pred_bboxes_so = pred_bboxes_so.view(N, nr_boxes_so, -1) # N,nr_boxes_so,8
        else:
            class_logits = class_logits.view(N, nr_boxes, -1) # N,nr_boxes,num_obj*2
            pred_bboxes = self.apply_deltas(bboxes_deltas2, bboxes.view(-1, 4))
            pred_bboxes = pred_bboxes.view(N, nr_boxes, -1) # N,nr_boxes,8
            pred_bboxes_so = torch.tensor([])
        
        return class_logits, pred_bboxes, pro_features, roi_features, \
            mask_class_logits, ent_mask_features, \
            so_class_logits, pred_bboxes_so, so_pro_features, so_roi_features
    
    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder
    
    def fusion(self, x, y):
        return F.relu(x + y, inplace=True) - (x - y) ** 2
    
    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes

class Instance_self_disentangle_atten(nn.Module):
    def __init__(self, dim_in, dim_q, dim_v, num_head=8, dropout=0.1, use_disentangle=False, use_nn=True, enable_ln=True):
        """
            preform in multi batch
        """
        super(Instance_self_disentangle_atten, self).__init__()
        self.use_disentangle = use_disentangle
        self.num_head = num_head
        self.temper = np.power(dim_q, 0.5)
        
        assert dim_q % num_head == 0, 'dim_q % num_head != 0'
        assert dim_v % num_head == 0, 'dim_v % num_head != 0'
        
        
        assert not (use_disentangle is True and use_nn is True)
        self.use_nn = use_nn
        
        if use_nn:
            #assert dim_in == dim_q
            #assert dim_in == dim_v
            self.self_attn = nn.MultiheadAttention(dim_in, num_head, dropout=dropout)
        else:
            self.glo_Wq = nn.Linear(dim_in, dim_q, bias=True)
            self.glo_Wk = nn.Linear(dim_in, dim_q, bias=True)
            self.glo_Wv = nn.Linear(dim_in, dim_v, bias=True)
            self.fc = nn.Linear(dim_v, dim_in)
            
            nn.init.xavier_normal_(self.fc.weight)
            nn.init.normal_(self.glo_Wq.weight, mean=0, std=np.sqrt(2.0 / (dim_in + dim_q)))
            nn.init.normal_(self.glo_Wk.weight, mean=0, std=np.sqrt(2.0 / (dim_in + dim_q)))
            nn.init.normal_(self.glo_Wv.weight, mean=0, std=np.sqrt(2.0 / (dim_in + dim_v)))
            
            nn.init.zeros_(self.glo_Wq.bias)
            nn.init.zeros_(self.glo_Wk.bias)
            nn.init.zeros_(self.glo_Wv.bias)
            nn.init.zeros_(self.fc.bias)
            
            if use_disentangle:
                self.glo_Wm = nn.Linear(dim_in, num_head, bias=True)
                nn.init.normal_(self.glo_Wm.weight, mean=0, std=np.sqrt(2.0 / (dim_in + num_head)))
                nn.init.zeros_(self.glo_Wm.bias)
        
        #self.dropout = nn.Dropout(dropout, inplace=True)
        self.enable_ln = enable_ln
        if enable_ln:
            self.layer_norm = nn.LayerNorm(dim_in)
        
    def forward(self, x, y, v=None, mask=None, pairwise_score=None):
        """
            x: [B, N, P]
            y: [B, Nk, P]
        """
        if v is None:
            v = y
        if self.use_nn:
            x = x.contiguous().permute(1, 0, 2) #N,B,P
            y = y.contiguous().permute(1, 0, 2) 
            v = v.contiguous().permute(1, 0, 2) 
            ans = self.self_attn(x, y, value=v)[0] #N,B,P
            ans = ans.contiguous().permute(1, 0, 2)
            x = x.contiguous().permute(1, 0, 2)
        else:
            num_head, temper = self.num_head, self.temper
            q = self.glo_Wq(x)
            k = self.glo_Wk(y)
            v = self.glo_Wv(v)
            if self.use_disentangle: m = self.glo_Wm(x)
            else: m = None
            ans = self.mh_disent_atten(q, k, v, m=m, num_head=num_head, \
                    temper=temper, mask=mask, pairwise_score=pairwise_score)
            ans = self.fc(ans)
            
        if self.enable_ln:
            #ans = self.dropout(ans)
            ans = self.layer_norm(x + ans)
        
        return ans
        
    def mh_disent_atten(self, q, k, v, m=None, num_head=1, temper=1.0, mask=None, pairwise_score=None):
        """
            m: [B, Nk, num_head]: m = x * Wm
                    Wm: [P, num_head]
            
            q: [B, N, P]
            k: [B, Nk, P]: k = x * Wk
            v: [B, Nk, V]: v = x * Wv
            
            ans: [num_head, N, V]
        """
        B, Nq = q.shape[:2]
        B, Nk = k.shape[:2]
        
        q = q.view(B, Nq, num_head, -1)
        k = k.view(B, Nk, num_head, -1)
        v = v.view(B, Nk, num_head, -1)
        
        q = q.permute(0, 2, 1, 3).contiguous() # B, num_head, N, P//num_head
        k = k.permute(0, 2, 3, 1).contiguous() # B, num_head, P//num_head, Nk
        v = v.permute(0, 2, 1, 3).contiguous() # B, num_head, Nk, P//num_head
        
        if m is not None:
            m = m.view(B, Nk, num_head, -1)
            m = m.permute(0, 2, 3, 1).contiguous() # B, num_head, 1, Nk
            score_pair = torch.matmul(
                q - torch.mean(q, dim=-2, keepdims=True), 
                k - torch.mean(k, dim=-1, keepdims=True)
            ) / temper # B, num_head, N, Nk
            score_unary = F.softmax(m, dim=-1)
        else:
            score_pair = torch.matmul(q, k) / temper # B, num_head, N, Nk
        
        if pairwise_score is not None:
            score_pair = score_pair + pairwise_score
        
        if mask is not None:
            mask = mask.repeat(num_head, 1, 1) # (n*b) x .. x ..
            score_pair = score_pair.masked_fill(mask, -np.inf)
        
        score_pair = F.softmax(score_pair, dim=-1)
        
        if m is not None:
            ans = torch.matmul(score_pair + score_unary, v)
        else:
            ans = torch.matmul(score_pair, v)
        
        ans = ans.view(num_head, B, Nq, -1)
        ans = ans.permute(1, 2, 0, 3).contiguous().view(B, Nq, -1)
        return ans

class Dynamic_conv(nn.Module):
    def __init__(self, dim_in, dim_in_v, vh=7, vw=None, hidden_dim=64, dropout=0.1, dynamic_conv_num=2):
        super(Dynamic_conv, self).__init__()
        dim_q = dim_in_v * hidden_dim
        self.dim_q = dim_q
        self.hidden_dim_2 = hidden_dim**2
        self.hidden_dim = hidden_dim
        self.dynamic_conv_num = dynamic_conv_num
        if vw is None: vw = vh
        
        self.Wq = nn.Linear(dim_in, 
            dim_q*2 + (hidden_dim**2) * (dynamic_conv_num-2), bias=True)
        
        self.dynamic_conv_norm = torch.nn.ModuleList()
        for i in range(dynamic_conv_num-1):
            self.dynamic_conv_norm.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),))
        self.dynamic_conv_norm.append(
                nn.Sequential(
                    nn.LayerNorm(dim_in_v),
                    nn.ReLU(inplace=True),))
                    
        self.Wv = nn.Sequential(
            nn.Linear(dim_in_v * vh * vw, dim_in, bias=True),
            nn.LayerNorm(dim_in),
            nn.ReLU(inplace=True),)
            ##nn.Linear(dim_in, dim_in, bias=True),
            #nn.Dropout(dropout),)
        self.layer_norm = nn.LayerNorm(dim_in)
        
    def forward(self, x, v):
        """
            q: [N, dim_in]
            v: [N, V, Nk], ENSURE: V == P // outdim
            
            return: [N, dim_in]
        """
        v = v.permute(0, 2, 1).contiguous()
        q = self.Wq(x)
        v = self.dynamic_conv(q, v)
        v = self.layer_norm(x + self.Wv(v))
        return v
            
    def dynamic_conv(self, q, v):
        """
            num_head == hidden_dim
            
            q: [N, dim_q*2 + ((dim_q // num_head)**2) * (dynamic_conv_num-2)]
            v: [N, Nk, V], ENSURE: V == P // num_head == P // hidden_dim
            
            ans: [N, num_head, Nk]
        """
        dim_q, hidden_dim_2, hidden_dim = \
            self.dim_q, self.hidden_dim_2, self.hidden_dim
        N = q.shape[0]
        dynamic_conv_num = self.dynamic_conv_num
        param = q[:, :dim_q].view(N, -1, hidden_dim)
        v = torch.bmm(v, param)
        v = self.dynamic_conv_norm[0](v) # N, 7*7, hidden_dim
        for i in range(1, dynamic_conv_num-1):
            st, ed = dim_q + hidden_dim_2 * (i-1), dim_q + hidden_dim_2 * i
            param = q[:, st:ed].view(N, -1, hidden_dim)
            v = torch.bmm(v, param)
            v = self.dynamic_conv_norm[i](v) # N, 7*7, hidden_dim
        last_q = q[:, -dim_q:].view(N, hidden_dim, -1)
        v = torch.bmm(v, last_q)
        v = self.dynamic_conv_norm[-1](v)
        v = v.view(N, -1)
        return v



class Shrink_Dynamic_conv(nn.Module):
    def __init__(self, dim_in, dim_in_v, vh=7, vw=None, hidden_dim=64, dropout=0.1):
        super(Shrink_Dynamic_conv, self).__init__()
        dim_q = dim_in_v * hidden_dim
        self.dim_q = dim_q
        self.hidden_dim_2 = hidden_dim**2
        self.hidden_dim = hidden_dim
        if vw is None: vw = vh
        self.vw = vw
        self.vh = vh
        
        num_head = max(1, dim_in // dim_in_v)
        self.num_head = num_head
        self.dim_group_q = dim_in_v * hidden_dim // num_head
        assert dim_in % num_head == 0, 'Shrink Dynamic conv, dim_in % num_head != 0'
        
        internal_hidden_dim = hidden_dim
        self.internal_hidden_dim = internal_hidden_dim
        
        
        #self.W_dynamic_bottleneck_in = nn.Linear(dim_in, internal_hidden_dim * hidden_dim, bias=True)
        #self.W_static_in = nn.Conv1d(internal_hidden_dim, \
        #    dim_in_v, 1, stride=1, padding=0, groups=1, bias=True)
        
        #if num_head > 1:
        #    self.W_in = nn.Linear(dim_in, dim_in, bias=True)
        
        self.W_dynamic_bottleneck_in = nn.Sequential(
            nn.Linear(dim_in, dim_in // 4, bias=True),
            nn.LayerNorm(dim_in // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in // 4, dim_in_v * hidden_dim + dim_in_v * hidden_dim, bias=True),)
        #self.W_dynamic_bottleneck_in = nn.Linear(dim_in_v, 2 * dim_in_v * hidden_dim, bias=True)
        
        #self.W_dynamic_bottleneck_out = nn.Linear(dim_in, hidden_dim * dim_in_v, bias=True)
        
        self.dynamic_conv_norm = torch.nn.ModuleList()
        self.dynamic_conv_norm.append(
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),))
        self.dynamic_conv_norm.append(
            nn.Sequential(
                nn.LayerNorm(dim_in_v),
                nn.ReLU(inplace=True),))
        
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(dim_in_v, 8*dim_in_v, (vh, vw), \
                stride=1, padding=0, groups=dim_in_v, bias=True),
            nn.SiLU(inplace=True),)
        self.depthwise_Wv = nn.Sequential(
            nn.Linear(8*dim_in_v, dim_in, bias=True),
            nn.LayerNorm(dim_in),
            nn.ReLU(inplace=True),)
        #self.Wv = nn.Sequential(
        #    nn.Linear(dim_in_v * vh * vw, dim_in, bias=True),
        #    nn.LayerNorm(dim_in),
        #    nn.ReLU(inplace=True),)
        self.layer_norm = nn.LayerNorm(dim_in)
        
    def forward(self, x, v):
        """
            x: [N, dim_in], (600, 1024)
            v: [N, V, Nk], ENSURE: V == P // outdim, (600, 256, 49)
            
            return: [N, dim_in]
        """
        v = v.permute(0, 2, 1).contiguous()
        
        vw, vh, num_head, dim_q, dim_group_q = self.vw, self.vh, self.num_head, self.dim_q, self.dim_group_q
        N, internal_hidden_dim, hidden_dim = x.shape[0], self.internal_hidden_dim, self.hidden_dim
        

        bottleneck_filters = self.W_dynamic_bottleneck_in(x)
        bottleneck_filter1 = bottleneck_filters[:, :dim_q]
        bottleneck_filter2 = bottleneck_filters[:, dim_q:]

        bottleneck_filter1 = bottleneck_filter1.view(N, -1, hidden_dim) # N, dim_in_v, hidden_dim
        v = torch.bmm(v, bottleneck_filter1) # N, 7*7, hidden_dim
        v = self.dynamic_conv_norm[0](v)
        
        bottleneck_filter2 = bottleneck_filter2.view(N, hidden_dim, -1)
        v = torch.bmm(v, bottleneck_filter2)
        v = self.dynamic_conv_norm[-1](v) # N, 7*7, dim_in_v
        
        v = v.permute(0, 2, 1).contiguous()
        v = v.view(N, -1, vh, vw).contiguous()
        v = self.pointwise_conv(v)
        v = v.view(N, -1)
        v = self.depthwise_Wv(v)
        
        #v = self.layer_norm(x + self.Wv(v))
        v = self.layer_norm(x + v)
        return v



class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim=64, sentence_len=64, scale=None, is_normalize=True):
        super(PositionalEncoding, self).__init__()
        self.sentence_len = sentence_len
        self.embedding_dim = embedding_dim
        
        if scale is not None and is_normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.is_normalize = is_normalize
        self.scale = scale
        if scale is None:
            self.scale = 2 * math.pi
        
        if is_normalize is False:
            self.positionalEncoding = torch.zeros((sentence_len, embedding_dim)).float()
            for pos in range(0, sentence_len):
                for i in range(0, embedding_dim // 2):
                    if 2 * i < embedding_dim:
                        self.positionalEncoding[pos, 2 * i] = \
                            math.sin(pos / math.pow(10000, 2 * i / embedding_dim))
                    if 2 * i + 1 < embedding_dim:
                        self.positionalEncoding[pos, 2 * i + 1] = \
                            math.cos(pos / math.pow(10000, 2 * i / embedding_dim))
        else:
            self.positionalEncoding = torch.zeros((1, embedding_dim)).float()
            for pos in range(0, 1):
                for i in range(0, embedding_dim // 2):
                    if 2 * i < embedding_dim:
                        self.positionalEncoding[pos, 2 * i] = \
                            math.pow(10000, 2 * i / embedding_dim)
                    if 2 * i + 1 < embedding_dim:
                        self.positionalEncoding[pos, 2 * i + 1] = \
                            math.pow(10000, 2 * i / embedding_dim)
        
        self.register_buffer('positional_encoding', self.positionalEncoding)
        
        
    @torch.no_grad()
    def forward(self, relative_roi):
        sentence_len = self.sentence_len
        
        #x = relative_roi[:, 1:]
        x = relative_roi
        N, NUM_CO = x.shape[:2]
		
        x = x.view(-1, 4).type(torch.int64)
        
        x = x // 16
        x = torch.clamp(x, 0, sentence_len-1)
        
        if self.is_normalize:
            x = x / (sentence_len - 1) * self.scale
            x = x.view(-1, 1)
            x = x / self.positional_encoding
            ans = torch.stack([x[:, 0::2].sin(), x[:, 1::2].cos()], dim=2).flatten(1)
        else:
            x = x.reshape(-1)
            ans = self.positional_encoding[x, :].to(relative_roi.device)
        
        ans = ans.view(N, NUM_CO, -1)
        #ans = ans.view(N, NUM_CO * self.embedding_dim)
        return ans

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def batch_box_iou(boxes1, boxes2):
    """
        boxes1: [B, N, 4]
        boxes2: [B, M, 4]
        
        return: [B, N, M]
    """
    area1 = (boxes1[:, :, 3] - boxes1[:, :, 1] + 1) * (boxes1[:, :, 2] - boxes1[:, :, 0] + 1)
    area2 = (boxes2[:, :, 3] - boxes2[:, :, 1] + 1) * (boxes2[:, :, 2] - boxes2[:, :, 0] + 1)
    lt = torch.max(boxes1[:, :, None, :2], boxes2[:, None, :, :2].type(boxes1.dtype))  # [B, N,M,2]
    rb = torch.min(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:].type(boxes1.dtype))  # [B, N,M,2]
    wh = (rb - lt).clamp(min=0)  # [B, N,M,2]
    inter = wh[:, :, :, 0] * wh[:, :, :, 1]  # [B, N,M]
    union = area1[:, :, None] + area2[:, None, :] - inter
    iou = inter / union
    return iou, union

def batch_pairwise_posi(boxes1, boxes2):
    """
        boxes1: [B, N, 4]
        boxes2: [B, M, 4]
        
        return: [B, N, M, 4]
    """
    device = boxes1.device
    boxes2[:, :, 2] = boxes2[:, :, 2] - boxes2[:, :, 0] + 1 #x1,y1,w,h
    boxes2[:, :, 3] = boxes2[:, :, 3] - boxes2[:, :, 1] + 1 #x1,y1,w,h
    
    boxes1[:, :, 2] = boxes1[:, :, 2] - boxes1[:, :, 0] + 1 #x1,y1,w,h
    boxes1[:, :, 3] = boxes1[:, :, 3] - boxes1[:, :, 1] + 1 #x1,y1,w,h
    
    a1 = torch.log((torch.abs(boxes1[:, :, None, 0] - boxes2[:, None, :, 0])+1e-3) / (boxes1[:, :, None, 2]+1e-3))
    a2 = torch.log((torch.abs(boxes1[:, :, None, 1] - boxes2[:, None, :, 1])+1e-3) / (boxes1[:, :, None, 3]+1e-3))
    a3 = torch.log((boxes2[:, None, :, 2]+1e-3) / (boxes1[:, :, None, 2]+1e-3))
    a4 = torch.log((boxes2[:, None, :, 3]+1e-3) / (boxes1[:, :, None, 3]+1e-3))
    
    ret = torch.stack([a1, a2, a3, a4], -1)
    return ret