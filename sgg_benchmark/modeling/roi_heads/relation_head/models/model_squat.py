import torch
from torch import nn
from torch.nn import functional as F
from sgg_benchmark.modeling.roi_heads.relation_head.models.utils.pairwise_feats import PairwiseFeatureExtractor

import copy

def set_diff(a, b):
    combined = torch.cat((a, b))
    uniques, counts = combined.unique(return_counts=True)
    diff = uniques[counts == 1]
    
    return diff

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1)
        )
    
    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=0, keepdim=True).expand(z_local.shape[0], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out

    
class P2PDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, unary_output=False):
        super(P2PDecoder, self).__init__() 
        if num_layers == 0:
            self.layers = []
        else:
            self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers 
        self.norm = norm
        self.unary_output = unary_output
        
    def forward(self, tgt, memory, ind, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        pair = tgt 
        if self.num_layers == 0: 
            if self.unary_output:
                return memory[0], tgt 
            else: return tgt
        
        for mod in self.layers:
            unary, pair = mod(pair, memory, ind, tgt_mask=tgt_mask,
                               memory_mask=memory_mask, 
                               tgt_key_padding_mask=tgt_key_padding_mask, 
                               memory_key_padding_mask=memory_key_padding_mask)
            
        if self.norm is not None:
            pair = self.norm(pair)
            unary = self.norm(unary)
            
        if self.unary_output: 
            return unary, pair
        
        return pair
    
    
class P2PDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, norm_first=False):
        super(P2PDecoderLayer, self).__init__() 
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_node = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_e2e  = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_e2n = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_n2e  = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_n2n = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.linear1_unary = nn.Linear(d_model, dim_feedforward)
        self.dropout_unary = nn.Dropout(dropout)
        self.linear2_unary = nn.Linear(dim_feedforward, d_model)
        
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm1_unary = nn.LayerNorm(d_model)
        self.norm2_unary = nn.LayerNorm(d_model)
        self.norm3_unary = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout1_unary = nn.Dropout(dropout)
        self.dropout2_unary = nn.Dropout(dropout)
        self.dropout3_unary = nn.Dropout(dropout)
        
        self.activation = activation
        
    def forward(self, tgt, memory, ind, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        sparsified_pair = tgt
        sparsified_unary, entire_pair = memory 
        ind_pair, ind_e2e, ind_n2e = ind 
        
        sparsified_pair = self.norm1(sparsified_pair + self._sa_block(sparsified_pair, None, None))
        sparsified_pair = sparsified_pair.to(entire_pair.dtype)
        sparsified_unary = self.norm1_unary(sparsified_unary + self._sa_node_block(sparsified_unary, None, None))
        
        pair = torch.zeros_like(entire_pair, dtype=entire_pair.dtype, device=entire_pair.device)
        ind_ = torch.logical_not((torch.arange(pair.size(0), device=entire_pair.device).unsqueeze(1) == ind_pair).any(1))
        
        pair[ind_pair] = sparsified_pair
        pair[ind_] = entire_pair[ind_]

        pair_e2e = pair[ind_e2e]
        pair_n2e = pair[ind_n2e]
        
        updated_pair = self.norm2(sparsified_pair + self._mha_e2e(sparsified_pair, pair_e2e, None, None) \
                                                  + self._mha_e2n(sparsified_pair, sparsified_unary, None, None)) 
        updated_pair = self.norm3(updated_pair + self._ff_block_edge(updated_pair)) 
        
        updated_unary = self.norm2(sparsified_unary + self._mha_n2e(sparsified_unary, pair_n2e, None, None) \
                                                    + self._mha_n2n(sparsified_unary, sparsified_unary, None, None)) 
        updated_unary = self.norm3(updated_unary + self._ff_block_node(updated_unary)) 
        
        return updated_unary, updated_pair

    def _sa_block(self, x, attn_mask, key_padding_mask): 
        x = self.self_attn(x, x, x, attn_mask=attn_mask, 
                           key_padding_mask=key_padding_mask, 
                           need_weights=False)[0]
        
        return self.dropout1(x)

    def _sa_node_block(self, x, attn_mask, key_padding_mask): 
        x = self.self_attn_node(x, x, x, attn_mask=attn_mask, 
                                key_padding_mask=key_padding_mask, 
                                need_weights=False)[0]
        
        return self.dropout1_unary(x)
    
    def _mha_e2n(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn_e2n(x, mem, mem, 
                                      attn_mask=attn_mask, 
                                      key_padding_mask=key_padding_mask, 
                                      need_weights=False)[0]
        
        return self.dropout2(x)
    
    def _mha_e2e(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn_e2e(x, mem, mem, 
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask, 
                                     need_weights=False)[0]
        
        return self.dropout2(x)
    
    def _mha_n2e(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn_n2e(x, mem, mem, 
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask, 
                                     need_weights=False)[0]
        
        return self.dropout2_unary(x)
    
    def _mha_n2n(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn_n2n(x, mem, mem, 
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask, 
                                     need_weights=False)[0]
        
        return self.dropout2_unary(x)

    def _ff_block_node(self, x): 
        x = self.linear2_unary(self.dropout_unary(self.activation(self.linear1_unary(x))))
        return self.dropout3_unary(x)
    
    def _ff_block_edge(self, x): 
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
        
        
class SquatContext(nn.Module):
    def __init__(self, config, in_channels, hidden_dim=512, num_iter=3):
        super(SquatContext, self).__init__()

        self.cfg = config
        self.hidden_dim = hidden_dim
        self.num_iter = num_iter
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.pairwise_feature_extractor = PairwiseFeatureExtractor(config, in_channels)
        self.pooling_dim = self.pairwise_feature_extractor.pooling_dim

        self.obj_embedding = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.rel_embedding = nn.Sequential(
            nn.Linear(in_channels * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.pooling_dim),
            nn.ReLU()
        )
            
        norm_first = config.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.PRE_NORM
        decoder_layer = P2PDecoderLayer(self.pooling_dim, 8, self.hidden_dim * 2, norm_first=norm_first)
        num_layer = config.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.NUM_DECODER
        self.m2m_decoder = P2PDecoder(decoder_layer, num_layer)
        self.obj_classifier = nn.Linear(self.hidden_dim, self.num_obj_cls)
        self.rel_classifier = nn.Linear(self.pooling_dim, self.num_rel_cls)
        
        self.mask_predictor = MaskPredictor(self.pooling_dim, self.hidden_dim)
        self.mask_predictor_e2e = MaskPredictor(self.pooling_dim, self.hidden_dim)
        self.mask_predictor_n2e = MaskPredictor(self.pooling_dim, self.hidden_dim) 
        self.rho = config.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.RHO
        self.beta = config.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.BETA

    def set_pretrain_pre_clser_mode(self, val=True):
        self.pretrain_pre_clser_mode = val

    def _get_map_idx(self, proposals, rel_pair_idxs):
        device = rel_pair_idxs[0].device
        rel_inds = []
        offset = 0
        obj_num = sum([len(proposal) for proposal in proposals])
        obj_obj_map = torch.zeros(obj_num, obj_num)

        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            obj_obj_map_i = (1 - torch.eye(len(proposal))).float()
            obj_obj_map[offset:offset + len(proposal), offset:offset + len(proposal)] = obj_obj_map_i
            rel_ind_i = rel_pair_idx + offset
            offset += len(proposal)
            rel_inds.append(rel_ind_i)

        rel_inds = torch.cat(rel_inds, dim=0)
        obj_obj_map = obj_obj_map.to(device)

        return rel_inds, obj_obj_map, obj_num

    def forward(self, roi_features, proposals, union_features, rel_pair_idxs, rel_gt_binarys=None, logger=None, use_union=True):
        num_rels = [rel_pair_idx.size(0) for rel_pair_idx in rel_pair_idxs]
        # rel_inds, obj_obj_map, obj_num = self._get_map_idx(proposals, rel_pair_idxs)
        
        feat_obj = self.obj_embedding(roi_features)
        augment_obj_feat, feat_pred = self.pairwise_feature_extractor(
            roi_features,
            union_features,
            proposals,
            rel_pair_idxs,
            use_union,
        )
        
        feat_pred_batch_key = torch.split(feat_pred, num_rels)
        masks = [self.mask_predictor(k).squeeze(1) for k in feat_pred_batch_key] # num_rel X 1 
        top_inds = [torch.topk(mask, int(mask.size(0) * self.rho))[1] for mask in masks]
        feat_pred_batch_query = [k[top_ind] for k, top_ind in zip(feat_pred_batch_key, top_inds)]
        
        masks_e2e = [self.mask_predictor_e2e(k).squeeze(1) for k in feat_pred_batch_key]
        top_inds_e2e = [torch.topk(mask, int(mask.size(0) * self.beta))[1] for mask in masks_e2e]
        
        masks_n2e = [self.mask_predictor_n2e(k).squeeze(1) for k in feat_pred_batch_key]
        top_inds_n2e = [torch.topk(mask, int(mask.size(0) * self.beta))[1] for mask in masks_n2e]
        
        augment_obj_feat = torch.split(augment_obj_feat, [len(proposal) for proposal in proposals])
        
        feat_pred_batch = [self.m2m_decoder(q.unsqueeze(1), (u.unsqueeze(1), p.unsqueeze(1)), (ind, ind_e2e, ind_n2e)).squeeze(1) \
                           for p, u, q, ind, ind_e2e, ind_n2e in \
                           zip(feat_pred_batch_key, augment_obj_feat, feat_pred_batch_query, top_inds, top_inds_e2e, top_inds_n2e)]
                    
        feat_pred_batch_ = []
        for idx, (top_ind, k, out) in enumerate(zip(top_inds, feat_pred_batch_key, feat_pred_batch)):
            remaining_ind = set_diff(torch.arange(k.size(0), device=k.device), top_ind)
            feat_pred_ = torch.zeros_like(k, dtype=k.dtype, device=k.device)
            out = out.to(k.dtype)

            feat_pred_[top_ind] = out 
            feat_pred_[remaining_ind] = k[remaining_ind]

            feat_pred_batch_.append(feat_pred_)

        feat_pred_ = torch.cat(feat_pred_batch_, dim=0)
        
        score_obj = self.obj_classifier(feat_obj)        
        score_pred = self.rel_classifier(feat_pred_)
        
        return score_obj, score_pred, (masks, masks_e2e, masks_n2e)

    def set_pretrain_pre_clser_mode(self, val=True):
        self.pretrain_pre_clser_mode = val