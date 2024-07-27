import torch
import torch.nn as nn
import torch.nn.functional as F
from sgg_benchmark.layers import fusion_func
from sgg_benchmark.modeling.utils import cat
from sgg_benchmark.utils.txt_embeddings import obj_edge_vectors
from sgg_benchmark.layers import MLP
from sgg_benchmark.modeling.make_layers import make_fc
from ..models.utils.utils_motifs import to_onehot, encode_box_info
from .utils.utils_relation import nms_per_cls

class PENetContext(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels, dropout_p=0.2):
        super().__init__()        

        self.num_obj_classes = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.cfg = config

        self.pooling_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM

        self.mlp_dim = self.cfg.MODEL.ROI_RELATION_HEAD.MLP_HEAD_DIM
        self.post_emb = nn.Linear(in_channels, self.mlp_dim * 2)  

        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_type=self.cfg.MODEL.TEXT_EMBEDDING, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)  # load Glove for objects
        self.obj_embed = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
       
        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim*2, self.mlp_dim)  
        self.gate_obj = nn.Linear(self.mlp_dim*2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim*2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim*2, self.mlp_dim)
        ])

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        
        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)

        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes) 
        self.lin_obj_cyx = make_fc(in_channels+ self.embed_dim + 128, self.hidden_dim)
        
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.obj_decode = not (self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX or self.cfg.MODEL.BACKBONE.FREEZE)

        self.use_text_features_only = self.cfg.MODEL.ROI_RELATION_HEAD.TEXTUAL_FEATURES_ONLY
        self.use_visual_features_only = self.cfg.MODEL.ROI_RELATION_HEAD.VISUAL_FEATURES_ONLY

    def forward(self, roi_features, proposals, rel_pair_idxs, logger=None):
        # refine object labels
        if self.obj_decode:
            entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        else:
            entity_dists, entity_preds = self.encode_obj_labels(proposals)

        entity_rep = self.post_emb(roi_features)   # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)    # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)    # xo

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)

        if not self.use_visual_features_only:
            entity_embeds = self.obj_embed(entity_preds) # obtaining the word embedding of entities with GloVe 
            entity_embeds = entity_embeds.split(num_objs, dim=0)
        else:
            # tensor of zeros of size (num_objs, embed_dim)
            entity_embeds = torch.zeros((sum(num_objs), self.embed_dim), device=roi_features.device)

        entity_preds = entity_preds.split(num_objs, dim=0)

        fusion_so = []

        for pair_idx, sub_rep, obj_rep, entity_embed in zip(rel_pair_idxs, sub_reps, obj_reps, entity_embeds):
            if self.use_text_features_only:
                s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  #  Ws x ts
                o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  #  Wo x to

                sub = s_embed  # s = Ws x ts
                obj = o_embed  # o = Wo x to

            elif self.use_visual_features_only:
                sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
                sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)
  
                sub = sem_sub
                obj = sem_obj

            else: # original full model
                s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  #  Ws x ts
                o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  #  Wo x to

                sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])
                sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])

                gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))
                gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))

                sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
                obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)

            fusion_so.append(fusion_func(sub, obj)) # F(s, o)

        fusion_so = cat(fusion_so, dim=0)  

        return entity_dists, entity_preds, fusion_so, None
    
    def encode_obj_labels(self, proposals):
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        obj_labels = obj_labels.long()
        obj_dists = to_onehot(obj_labels, self.num_obj_classes)
        return obj_dists, obj_labels

    def refine_obj_labels(self, roi_features, proposals):
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
        obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        obj_dists = self.out_obj(obj_pre_rep_for_pred)
        use_decoder_nms = not self.training
        if use_decoder_nms:
            boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
            obj_preds = nms_per_cls(obj_dists, boxes_per_cls, num_objs, self.nms_thresh).long()
        else:
            obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()
        
        return obj_dists, obj_preds