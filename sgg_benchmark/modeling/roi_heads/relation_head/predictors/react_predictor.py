import numpy as np
import torch
from sgg_benchmark.modeling import registry
from torch import nn

from sgg_benchmark.layers import MLP
from sgg_benchmark.modeling.utils import cat
from sgg_benchmark.utils.txt_embeddings import rel_vectors, obj_edge_vectors

from sgg_benchmark.modeling.roi_heads.relation_head.predictors.default_predictors import BasePredictor
from sgg_benchmark.modeling.roi_heads.relation_head.models.utils.utils_motifs import to_onehot

import torch.nn.functional as F

@registry.ROI_RELATION_PREDICTOR.register("REACTPredictor")
class REACTPredictor(BasePredictor):
    def __init__(self, config, in_channels):
        super().__init__(config, in_channels)        

        self.num_obj_classes = len(self.obj_classes)
        dropout_p = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE

        self.mlp_dim = self.cfg.MODEL.ROI_RELATION_HEAD.MLP_HEAD_DIM
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM

        self.use_union = self.cfg.MODEL.ROI_RELATION_HEAD.USE_UNION_FEATURES or self.cfg.MODEL.ROI_RELATION_HEAD.USE_SPATIAL_FEATURES
        self.text_only = self.cfg.MODEL.ROI_RELATION_HEAD.TEXTUAL_FEATURES_ONLY

        self.post_emb = nn.Linear(in_channels, self.mlp_dim * 2) 

        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim*2, self.mlp_dim)  
        self.gate_obj = nn.Linear(self.mlp_dim*2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim*2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim*2, self.mlp_dim)
        ])

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)

        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)

        if self.use_union:
            self.gate_pred = nn.Linear(self.mlp_dim*2, self.mlp_dim)
        
        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_type=self.cfg.MODEL.TEXT_EMBEDDING, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)  # load Glove for objects
        rel_embed_vecs = rel_vectors(self.rel_classes, wv_type=self.cfg.MODEL.TEXT_EMBEDDING, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)

        with torch.no_grad():
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
        self.so_linear_layer = nn.Linear(self.mlp_dim*2, self.mlp_dim)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        if self.text_only:
            return self.text_only_forward(proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger)
        
        add_losses = {}

        entity_dists, entity_preds = self.encode_obj_labels(proposals)

        entity_rep = self.post_emb(roi_features)   # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)    # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)    # xo

        entity_embeds = self.obj_embed(entity_preds) # obtaining the word embedding of entities with GloVe 

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []

        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed in zip(rel_pair_idxs, sub_reps, obj_reps, entity_preds, entity_embeds):

            s_embed = self.W_sub(entity_embed.index_select(0, pair_idx[:, 0]))  #  Ws x ts
            o_embed = self.W_obj(entity_embed.index_select(0, pair_idx[:, 1]))  #  Wo x to

            sem_sub = self.vis2sem(sub_rep.index_select(0, pair_idx[:, 0]))  # h(xs)
            sem_obj = self.vis2sem(obj_rep.index_select(0, pair_idx[:, 1]))  # h(xo)
            
            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)

            fusion_so.append(F.relu(self.so_linear_layer(cat((sub, obj), dim=-1)))) # F(s, o)

        fusion_so = cat(fusion_so, dim=0)  

        if self.use_union:
            sem_pred = self.vis2sem(union_features)  # h(xu)
            gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp

            rel_rep = fusion_so - sem_pred * gate_sem_pred  #  F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        else:
            rel_rep = fusion_so
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes
        
        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if self.training:
            ### Prototype Regularization  ---- cosine similarity
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach() 
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (self.num_rel_cls*self.num_rel_cls)  
            add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
            
            ### Prototype Regularization  ---- Euclidean distance
            gamma2 = 7.0
            predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1) 
            predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(self.num_rel_cls, -1, -1)
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1   # obtain d-, where k2 = 1
            dist_loss = torch.max(torch.zeros(self.num_rel_cls).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
            add_losses.update({"dist_loss2": dist_loss})

            ###  Prototype-based Learning  ---- Euclidean distance
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1)  # r
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
            mask_neg = torch.ones(rel_labels.size(0), self.num_rel_cls).cuda()  
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
            loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
            add_losses.update({"loss_dis": loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)

        return entity_dists, rel_dists, add_losses
    

    def text_only_forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        add_losses = {}

        entity_dists, entity_preds = self.encode_obj_labels(proposals)

        entity_embeds = self.obj_embed(entity_preds) # obtaining the word embedding of entities with GloVe 

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]

        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []

        for pair_idx, entity_pred, entity_embed in zip(rel_pair_idxs, entity_preds, entity_embeds):

            s_embed = self.W_sub(entity_embed.index_select(0, pair_idx[:, 0]))  #  Ws x ts
            o_embed = self.W_obj(entity_embed.index_select(0, pair_idx[:, 1]))  #  Wo x to

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(s_embed))) + s_embed)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(o_embed))) + o_embed)
            #####

            fusion_so.append(F.relu(self.so_linear_layer(cat((sub, obj), dim=-1)))) # F(s, o)

        rel_rep = cat(fusion_so, dim=0)  

        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes
        
        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if self.training:
            ### Prototype Regularization  ---- cosine similarity
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach() 
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (self.num_rel_cls*self.num_rel_cls)  
            add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
            
            ### Prototype Regularization  ---- Euclidean distance
            gamma2 = 7.0
            predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1) 
            predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(self.num_rel_cls, -1, -1)
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1   # obtain d-, where k2 = 1
            dist_loss = torch.max(torch.zeros(self.num_rel_cls).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
            add_losses.update({"dist_loss2": dist_loss})

            ###  Prototype-based Learning  ---- Euclidean distance
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1)  # r
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
            mask_neg = torch.ones(rel_labels.size(0), self.num_rel_cls).cuda()  
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
            loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
            add_losses.update({"loss_dis": loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)

        return entity_dists, rel_dists, add_losses
    
    def encode_obj_labels(self, proposals):
        obj_labels = cat([proposal[:, 5] for proposal in proposals], dim=0)
        obj_labels = obj_labels.long()
        obj_dists = to_onehot(obj_labels, self.num_obj_classes)
        return obj_dists, obj_labels