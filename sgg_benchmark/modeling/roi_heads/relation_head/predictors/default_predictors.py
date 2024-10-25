import numpy as np
import torch
from sgg_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F

from sgg_benchmark.modeling.utils import cat

from ..models.model_msg_passing import IMPContext
from ..models.model_vtranse import VTransEFeature
from ..models.model_vctree import VCTreeLSTMContext
from ..models.model_motifs import LSTMContext, LSTMContext_RNN, FrequencyBias # LSTMContext_RNN correspond to original motifs, LSTMContext is without the object detection part (decoder RNN)
from ..models.model_motifs_with_attribute import AttributeLSTMContext
from ..models.model_transformer import TransformerContext
from ..models.model_gpsnet import GPSNetContext
from ..models.model_penet import PENetContext
from ..models.model_squat import SquatContext

from ..models.utils.utils_relation import layer_init, get_box_info, get_box_pair_info
from ..models.utils.classifiers import build_classifier
from ..models.utils.utils_motifs import to_onehot
from ..models.utils.utils_relation import obj_prediction_nms

from sgg_benchmark.data import get_dataset_statistics
class BasePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(BasePredictor, self).__init__()

        # load base parameters
        self.cfg = config

        self.statistics = get_dataset_statistics(self.cfg)

        self.obj_classes = self.statistics['obj_classes']
        self.rel_classes = self.statistics['rel_classes']

        self.pred_freq = self.statistics['pred_freq']

        self.attribute_on = config.MODEL.ATTRIBUTE_ON

        self.num_obj_cls = self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = self.cfg.MODEL.ROI_RELATION_HEAD.USE_FREQUENCY_BIAS

        assert in_channels is not None

        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.use_vision = self.cfg.MODEL.ROI_RELATION_HEAD.USE_UNION_FEATURES or self.cfg.MODEL.ROI_RELATION_HEAD.USE_SPATIAL_FEATURES

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        self.freeze_backbone = self.cfg.MODEL.BACKBONE.FREEZE
        self.obj_decode = not (self.freeze_backbone or self.mode == "predcls")

        if self.use_bias:
            self.freq_bias = FrequencyBias(self.cfg, self.statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        raise NotImplementedError
    
@registry.ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(BasePredictor):
    def __init__(self, config, in_channels):
        super(BasePredictor, self).__init__(config, in_channels)

        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses

@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(BasePredictor):
    def __init__(self, config, in_channels):
        super().__init__(config, in_channels)

        # module construct
        self.context_layer = TransformerContext(config, self.obj_classes, self.rel_classes, in_channels)

        # post decoding
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

            rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
        else:
            rel_dists = self.ctx_compress(prod_rep)
                
        # use frequence bias
        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred)

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses

@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(BasePredictor):
    def __init__(self, config, in_channels):
        super().__init__(config, in_channels)

        # init contextual lstm encoding
        if self.attribute_on:
            att_classes = self.statistics['att_classes']
            self.context_layer = AttributeLSTMContext(config, self.obj_classes, att_classes, self.rel_classes, in_channels)
        else:
            if self.freeze_backbone:
                self.context_layer = LSTMContext(config, self.obj_classes, self.rel_classes, in_channels)
            else:
                self.context_layer = LSTMContext_RNN(config, self.obj_classes, self.rel_classes, in_channels)

        # post decoding
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_objs = [len(b) for b in proposals]
        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            if self.freeze_backbone:
                obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)
            else:
                obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)
                obj_dists = obj_dists.split(num_objs, dim=0)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses

@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(BasePredictor):
    def __init__(self, config, in_channels):
        super().__init__(config, in_channels)

        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, self.obj_classes, self.rel_classes, self.statistics, in_channels)

        # post decoding
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        layer_init(self.ctx_compress, xavier=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        if self.use_vision:
            if self.union_single_not_match:
                union_features = self.up_dim(union_features)

            ctx_dists = self.ctx_compress(prod_rep * union_features)
        else:
            ctx_dists = self.ctx_compress(prod_rep)
        if self.use_bias:
            rel_dists = ctx_dists + self.freq_bias.index_with_labels(pair_pred.long())
        else:
            rel_dists = ctx_dists

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses

@registry.ROI_RELATION_PREDICTOR.register("GPSNetPredictor")
class GPSNetPredictor(BasePredictor):
    def __init__(self, config, in_channels):
        super().__init__(config, in_channels)

        self.context_layer = GPSNetContext(
            config,
            self.obj_classes, 
            self.rel_classes, 
            in_channels,
            hidden_dim=self.hidden_dim,
            num_iter=2,
        )

        self.rel_feature_type = "fusion"

        self.use_obj_recls_logits = not self.freeze_backbone

        # post classification
        self.rel_classifier = build_classifier(self.pooling_dim, self.num_rel_cls)

        self.rel_classifier.reset_parameters()

    def forward(
        self,
        inst_proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        """

        :param inst_proposals:
        :param rel_pair_idxs:
        :param rel_labels:
        :param rel_binarys:
            the box pairs with that match the ground truth [num_prp, num_prp]
        :param roi_features:
        :param union_features:
        :param logger:

        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        obj_pred_logits, obj_pred_labels, rel_feats, _ = self.context_layer(roi_features, union_features, inst_proposals, rel_pair_idxs, rel_binarys)

        rel_cls_logits = self.rel_classifier(rel_feats)

        num_objs = [len(b) for b in inst_proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_pred_labels = obj_pred_labels.split(num_objs, dim=0)
            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_pred_labels):
                pair_preds.append(
                    torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
                )
            pair_pred = cat(pair_preds, dim=0)
            rel_cls_logits = rel_cls_logits + self.freq_bias.index_with_labels(
                pair_pred.long()
            )

        obj_pred_logits = obj_pred_logits.split(num_objs, dim=0)
        rel_cls_logits = rel_cls_logits.split(num_rels, dim=0)

        add_losses = {}

        return obj_pred_logits, rel_cls_logits, add_losses

@registry.ROI_RELATION_PREDICTOR.register("SquatPredictor")
class SquatPredictor(BasePredictor): 
    def __init__(self, config, in_channels):
        super().__init__(config, in_channels)

        self.loss_coef = 1.0

        # self.split_context_model4inst_rel = config.MODEL.ROI_RELATION_HEAD.GRCNN_MODULE.SPLIT_GRAPH4OBJ_REL

        self.context_layer = SquatContext(config, in_channels, hidden_dim=self.hidden_dim)

        self.obj_recls_logits_update_manner = "replace"
        assert self.obj_recls_logits_update_manner in ["replace", "add"]
        
    def forward(self, inst_proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None): 
        """
        :param inst_proposals:
        :param rel_pair_idxs:
        :param rel_labels:
        :param rel_binarys:
            the box pairs with that match the ground truth [num_prp, num_prp]
        :param roi_features:
        :param union_features:
        :param logger:

        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        score_obj, score_rel, masks = self.context_layer(
            roi_features, inst_proposals, union_features, rel_pair_idxs, rel_binarys, self.use_vision
        ) # masks : [list[Tensor]]
        rel_cls_logits = score_rel
        
        if not self.obj_decode:
            obj_labels = cat(
                [proposal.get_field("labels") for proposal in inst_proposals], dim=0
            )
            refined_obj_logits = to_onehot(obj_labels, self.num_obj_cls)
        else:
            refined_obj_logits = score_obj

        num_objs = [len(b) for b in inst_proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        
        # using the object results, update the pred label and logits
        if self.obj_decode:
            obj_pred_logits = cat(
                [each_prop.get_field("predict_logits") for each_prop in inst_proposals], dim=0
            )

            boxes_per_cls = cat(
                [proposal.get_field("boxes_per_cls") for proposal in inst_proposals], dim=0
            )  # comes from post process of box_head
            # here we use the logits refinements by adding
            if self.obj_recls_logits_update_manner == "add":
                obj_pred_logits = refined_obj_logits + obj_pred_logits
            if self.obj_recls_logits_update_manner == "replace":
                obj_pred_logits = refined_obj_logits
            refined_obj_pred_labels = obj_prediction_nms(
                boxes_per_cls, obj_pred_logits, nms_thresh=0.5
            )
            obj_pred_labels = refined_obj_pred_labels
        else:
            obj_pred_labels = cat(
                [each_prop.get_field("labels") for each_prop in inst_proposals], dim=0
            )
            obj_pred_logits = refined_obj_logits
            
        if self.use_bias:
            obj_pred_labels = obj_pred_labels.split(num_objs, dim=0)
            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_pred_labels):
                pair_preds.append(
                    torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
                )
            pair_pred = cat(pair_preds, dim=0)
            rel_cls_logits = rel_cls_logits + self.freq_bias.index_with_labels(
                pair_pred.long()
            )
        
        add_losses = {}
        losses = []

        if self.training:
            masks_q, masks_e2e, masks_n2e = masks 

            for mask, rel_binary, rel_pair_idx in zip(masks_q, rel_binarys, rel_pair_idxs):
                target = rel_binary[rel_pair_idx[:, 0], rel_pair_idx[:, 1]]
                target = target.float()
                loss = F.binary_cross_entropy_with_logits(mask, target)
                losses.append(loss)
            losses = sum(losses) / len(losses)
            add_losses['loss_mask_query'] = losses / 3. * self.loss_coef
            
            losses = []
            for mask, rel_binary, rel_pair_idx in zip(masks_e2e, rel_binarys, rel_pair_idxs):
                target = rel_binary[rel_pair_idx[:, 0], rel_pair_idx[:, 1]]
                target = target.float()
                loss = F.binary_cross_entropy_with_logits(mask, target)
                losses.append(loss)
            losses = sum(losses) / len(losses)
            add_losses['loss_mask_e2e'] = losses / 3. * self.loss_coef
            
            losses = []
            for mask, rel_binary, rel_pair_idx in zip(masks_n2e, rel_binarys, rel_pair_idxs):
                target = rel_binary[rel_pair_idx[:, 0], rel_pair_idx[:, 1]]
                target = target.float()
                loss = F.binary_cross_entropy_with_logits(mask, target)
                losses.append(loss)
            losses = sum(losses) / len(losses)
            add_losses['loss_mask_n2e'] = losses / 3. * self.loss_coef
            
        obj_pred_logits = obj_pred_logits.split(num_objs, dim=0)
        rel_cls_logits = rel_cls_logits.split(num_rels, dim=0)
        
        return obj_pred_logits, rel_cls_logits, add_losses
    
@registry.ROI_RELATION_PREDICTOR.register("VETOPredictor")
class VETOPredictor(BasePredictor):
    def __init__(self, config, in_channels):
        super().__init__(config, in_channels)

        self.use_norm = False
        self.pcpl = False

        self.FC_SIZE_CLASS = self.cfg.MODEL.ROI_RELATION_HEAD.VETOTRANSFORMER.T_INPUT_DIM
        self.FC_SIZE_LOC = self.cfg.MODEL.ROI_RELATION_HEAD.VETOTRANSFORMER.T_INPUT_DIM
        self.LOC_INPUT_SIZE = 256
        self.obj_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.use_embed = False
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        # post decoding
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(len(self.obj_classes), self.embed_dim)
        classme_input_dim = 200  # 151 #self.embed_dim if self.use_embed else len(self.obj_classes)
        self.class_projection = nn.Sequential(
            nn.Linear(classme_input_dim * 2, self.FC_SIZE_CLASS),
            nn.ReLU(inplace=True))

        with torch.no_grad():
            self.obj_embed.weight.copy_(embed_vecs, non_blocking=True)
        # self.decoder_lin = nn.Linear(self.obj_dim * 2 + self.embed_dim + 128, len(self.obj_classes))

        # position embedding
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])

        self.pos_embed = nn.Sequential(*[
            nn.BatchNorm1d(4, momentum=0.001),
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        ])

        self.location_projection = nn.Sequential(
            nn.Linear(self.LOC_INPUT_SIZE, self.FC_SIZE_LOC),
            nn.ReLU(inplace=True))

        self.fusion_transformer = VETOTransformer(config=config, in_channels=256)
        features_size = self.cfg.MODEL.ROI_RELATION_HEAD.VETOTRANSFORMER.T_INPUT_DIM
        # -- Final FC layer which predicts the relations
        self.rel_out = xavier_init(nn.Linear(features_size, self.num_rel_cls, bias=True))
        self.beta_loss = self.cfg.GLOBAL_SETTING.BETA_LOSS
        if self.beta_loss:
            rel_counts = self.statistics['pred_freq']
            rel_counts[::-1].sort()
            beta = 0.999  # (class_volume - 1.0) / class_volume
            rel_class_weights = (1.0 - beta) / (1 - (beta ** rel_counts))
            rel_class_weights *= float(self.num_rel_cls) / np.sum(rel_class_weights)
            rel_class_weights = torch.FloatTensor(rel_class_weights).cuda()
        else:
            rel_class_weights = np.ones((self.num_rel_cls,))
            rel_class_weights = torch.from_numpy(rel_class_weights).float()
        self.criterion_loss_rel = nn.CrossEntropyLoss(weight=rel_class_weights)
        self.criterion_loss = nn.CrossEntropyLoss()


    def forward(self, proposals,
                    rel_pair_idxs,
                    rel_labels,
                    logger,
                    roi_features=None,
                    roi_depth_features=None, rel_binarys=None):

            if self.mode == "predcls":
                obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            else:
                obj_labels = None

            if self.mode == "predcls":
                obj_logits = obj_labels
                obj_embed = self.obj_embed(obj_labels.long())
                obj_dists = F.one_hot(obj_labels.long(), self.num_obj_cls).float()

            else:
                obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
                obj_labels = cat([proposal.get_field("pred_labels") for proposal in proposals], dim=0).detach()
                obj_dists = F.one_hot(obj_labels.long(), self.num_obj_cls).float()
                obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight

            if proposals[0].mode == 'xyxy':
                centor_proposals = [p.convert('xywh') for p in proposals]
            else:
                centor_proposals = proposals

            pos_embed = self.pos_embed(cat([art.center_xywh(p.bbox) for p in centor_proposals], dim=0))

            proposal_count_per_img = [len(x) for x in proposals]
            rel_count_per_img = [len(x) for x in rel_pair_idxs]
            subj_inds = torch.zeros(sum(rel_count_per_img), dtype=torch.long)
            obj_inds = torch.zeros(sum(rel_count_per_img), dtype=torch.long)
            start = 0
            cumulative_proposals_count = 0
            for i, irel_pair in enumerate(rel_pair_idxs):
                end = start+len(irel_pair)
                subj_inds[start: end] = irel_pair[:, 0] + cumulative_proposals_count
                obj_inds[start: end] = irel_pair[:, 1] + cumulative_proposals_count
                cumulative_proposals_count += proposal_count_per_img[i]
                start = end

            # -- Create a pairwise relation vector out of location features
            rel_location = torch.cat((pos_embed[subj_inds], pos_embed[obj_inds]), dim=1)
            rel_location = self.location_projection(rel_location)
            rel_class = torch.cat((obj_embed[subj_inds], obj_embed[obj_inds]), dim=1)
            rel_class = self.class_projection(rel_class)
            rel_visual = torch.cat((roi_features[subj_inds], roi_features[obj_inds]), 1)
            rel_depth = torch.cat((roi_depth_features[subj_inds], roi_depth_features[obj_inds]), 1)
            rel_logits_raw = self.fusion_transformer(rel_depth, rel_visual, rel_location, rel_class)
            rel_dists = self.rel_out(
                rel_logits_raw)

            add_losses = {}
            if self.training:
                if self.mode != "predcls":
                    fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                    loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                    add_losses['obj_loss'] = loss_refine_obj
                rel_labels = cat(rel_labels, dim=0)
                add_losses['rel_loss'] = self.criterion_loss_rel(rel_dists, rel_labels)
                return None, None, add_losses, None, None, None
            obj_dists = obj_dists.split(proposal_count_per_img, dim=0)
            rel_dists = rel_dists.split(rel_count_per_img, dim=0)

            return obj_dists, rel_dists, add_losses, None, None, None