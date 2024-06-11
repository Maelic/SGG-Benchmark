'''Rectified Identity Cell'''

import torch
from torch import nn
import torch.nn.functional as F
from sgg_benchmark.modeling.utils import cat
from sgg_benchmark.utils.txt_embeddings import obj_edge_vectors
from .utils.utils_co_attention import Self_Attention_Encoder, Cross_Attention_Encoder
from .utils.utils_motifs import to_onehot, encode_box_info
from .utils.utils_relation import nms_per_cls

class Self_Attention_Cell(nn.Module):
    def __init__(self, config, hidden_dim=None):
        super(Self_Attention_Cell, self).__init__()
        self.cfg = config
        if hidden_dim is None:
            self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        else:
            self.hidden_dim = hidden_dim
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.SA_transformer_encoder = Self_Attention_Encoder(self.num_head, self.k_dim,
                                              self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)


    def forward(self, x, textual_feats=None, num_objs=None):
        assert num_objs is not None
        outp = self.SA_transformer_encoder(x, num_objs)

        return outp

class Cross_Attention_Cell(nn.Module):
    def __init__(self, config, hidden_dim=None):
        super(Cross_Attention_Cell, self).__init__()
        self.cfg = config
        if hidden_dim is None:
            self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        else:
            self.hidden_dim = hidden_dim
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.CA_transformer_encoder = Cross_Attention_Encoder(self.num_head, self.k_dim,
                                self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)


    def forward(self, x, textual_feats, num_objs=None):
        assert num_objs is not None
        outp = self.CA_transformer_encoder(x, textual_feats, num_objs)

        return outp

class Single_Layer_Hybrid_Attention(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config):
        super().__init__()
        self.SA_Cell_vis = Self_Attention_Cell(config)
        self.SA_Cell_txt = Self_Attention_Cell(config)
        self.CA_Cell_vis = Cross_Attention_Cell(config)
        self.CA_Cell_txt = Cross_Attention_Cell(config)

    def forward(self, visual_feats, text_feats, num_objs):
        tsa = self.SA_Cell_txt(text_feats, num_objs=num_objs)
        tca = self.CA_Cell_txt(text_feats, visual_feats, num_objs=num_objs)
        vsa = self.SA_Cell_vis(visual_feats, num_objs=num_objs)
        vca = self.CA_Cell_vis(visual_feats, text_feats, num_objs=num_objs)
        textual_output = tsa + tca
        visual_output = vsa + vca

        return visual_output, textual_output

class SHA_Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config, n_layers):
        super().__init__()
        self.cfg = config
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM
        self.cross_module = nn.ModuleList([
            Single_Layer_Hybrid_Attention(config)
            for _ in range(n_layers)])

    def forward(self, visual_feats, text_feats, num_objs):
        visual_output = visual_feats
        textual_output = text_feats

        for enc_layer in self.cross_module:
            visual_output, textual_output = enc_layer(visual_output, textual_output, num_objs)

        visual_output = visual_output + textual_output

        return visual_output, textual_output

class SHA_Context(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super().__init__()
        self.cfg = config
        # setting parameters
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER
        self.edge_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER

        # the following word embedding layer should be initalize by glove.6B before using
        embed_vecs = obj_edge_vectors(self.obj_classes, wv_type=self.cfg.MODEL.TEXT_EMBEDDING, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)

        # position embedding
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])
        self.lin_obj_visual = nn.Linear(self.in_channels + 128, self.hidden_dim)
        self.lin_obj_textual = nn.Linear(self.embed_dim, self.hidden_dim)

        self.lin_edge_visual = nn.Linear(self.hidden_dim + self.in_channels, self.hidden_dim)
        self.lin_edge_textual = nn.Linear(self.embed_dim, self.hidden_dim)

        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_cls)

        self.context_obj = SHA_Encoder(config, self.obj_layer)
        self.context_edge = SHA_Encoder(config, self.edge_layer)

        self.obj_decode = not (self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX or self.cfg.MODEL.BACKBONE.FREEZE)

        self.use_text_features_only = self.cfg.MODEL.ROI_RELATION_HEAD.TEXTUAL_FEATURES_ONLY
        self.use_visual_features_only = self.cfg.MODEL.ROI_RELATION_HEAD.VISUAL_FEATURES_ONLY

    def forward(self, roi_features, proposals, rel_pair_idxs, logger=None):
        obj_labels = None
        if self.obj_decode: # backbone is completely frozen and we consider predictions as GT
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight
        else:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            obj_embed = self.obj_embed1(obj_labels.long())

        # bbox embedding will be used as input
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.bbox_embed(encode_box_info(proposals))

        # encode objects with transformer
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_vis = cat((roi_features, pos_embed), -1)
        obj_pre_rep_vis = self.lin_obj_visual(obj_pre_rep_vis)
        obj_pre_rep_txt = obj_embed
        obj_pre_rep_txt = self.lin_obj_textual(obj_pre_rep_txt)
        obj_feats_vis, _, = self.context_obj(obj_pre_rep_vis, obj_pre_rep_txt, num_objs)
        obj_feats = obj_feats_vis

        # predict obj_dists and obj_preds
        if not self.obj_decode:
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)

        else:
            obj_dists = self.out_obj(obj_feats)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1
        
        edge_pre_rep_vis = cat((roi_features, obj_feats), dim=-1)
        edge_pre_rep_txt = self.obj_embed2(obj_preds)

        # edge context
        edge_pre_rep_vis = self.lin_edge_visual(edge_pre_rep_vis)
        edge_pre_rep_txt = self.lin_edge_textual(edge_pre_rep_txt)
        edge_ctx_vis, _ = self.context_edge(edge_pre_rep_vis, edge_pre_rep_txt, num_objs)
        edge_ctx = edge_ctx_vis

        return obj_dists, obj_preds, edge_ctx, None