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

        self.mlp_dim = self.cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_type=self.cfg.MODEL.TEXT_EMBEDDING, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)  # load Glove for objects
        self.obj_embed = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
       
        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        
        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
  
    def forward(self, roi_features, proposals, rel_pair_idxs, logger=None):

        entity_dists, entity_preds = self.encode_obj_labels(proposals)

        num_objs = [len(b) for b in proposals]

        entity_embeds = self.obj_embed(entity_preds) # obtaining the word embedding of entities with GloVe 
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        entity_preds = entity_preds.split(num_objs, dim=0)

        fusion_so = []

        for pair_idx, entity_embed in zip(rel_pair_idxs, entity_embeds):
            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  #  Ws x ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  #  Wo x to

            sub = s_embed  # s = Ws x ts
            obj = o_embed  # o = Wo x to

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