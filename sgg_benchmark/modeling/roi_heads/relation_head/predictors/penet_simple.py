import numpy as np
import torch
from sgg_benchmark.modeling import registry
from torch import nn

from sgg_benchmark.layers import MLP
from sgg_benchmark.modeling.utils import cat
from sgg_benchmark.utils.txt_embeddings import rel_vectors

from sgg_benchmark.modeling.roi_heads.relation_head.models.model_penet import PENetContext
from sgg_benchmark.modeling.roi_heads.relation_head.predictors.default_predictors import BasePredictor

@registry.ROI_RELATION_PREDICTOR.register("PENetSimplePredictor")
class PrototypeEmbeddingNetwork(BasePredictor):
    def __init__(self, config, in_channels):
        super().__init__(config, in_channels)        

        self.num_obj_classes = len(self.obj_classes)
        dropout_p = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE

        self.context_layer = PENetContext(config, self.obj_classes, self.rel_classes, in_channels, dropout_p=dropout_p)

        self.mlp_dim = self.cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM

        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        rel_embed_vecs = rel_vectors(self.rel_classes, wv_type=self.cfg.MODEL.TEXT_EMBEDDING, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        add_losses = {}

        entity_dists, _, fusion_so, _ = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        rel_rep = fusion_so # - sem_pred * gate_sem_pred  #  F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
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

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if self.training:
            ### Prototype Regularization  ---- cosine similarity
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach() 
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (self.num_rel_cls*self.num_rel_cls)  
            add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
            ### end
            
            ### Prototype Regularization  ---- Euclidean distance
            gamma2 = 7.0
            predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1) 
            predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(self.num_rel_cls, -1, -1)
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1   # obtain d-, where k2 = 1
            dist_loss = torch.max(torch.zeros(self.num_rel_cls).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
            add_losses.update({"dist_loss2": dist_loss})
            ### end 

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
            ### end 
 
        return entity_dists, rel_dists, add_losses