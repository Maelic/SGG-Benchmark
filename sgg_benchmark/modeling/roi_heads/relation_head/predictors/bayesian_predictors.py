# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from sgg_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F

from sgg_benchmark.layers import Label_Smoothing_Regression
from sgg_benchmark.modeling.utils import cat
from ..models.model_msg_passing import IMPContext
from ..models.model_vtranse import VTransEFeature
from ..models.model_vctree import VCTreeLSTMContext
from ..models.model_motifs import LSTMContext, FrequencyBias
from ..models.model_motifs_with_attribute import AttributeLSTMContext
from ..models.model_transformer import TransformerContext
from ..models.utils.utils_relation import layer_init, get_box_info, get_box_pair_info
from sgg_benchmark.data import get_dataset_statistics

from ..models.utils.bayes_heads import BayesHead, BayesHeadProb


@registry.ROI_RELATION_PREDICTOR.register("TransformerHierPredictor")
class TransformerHierPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerHierPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.USE_UNION_FEATURES
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']

        if self.attribute_on:
            att_classes = statistics['att_classes']
            assert self.num_att_cls == len(att_classes)

        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # module construct
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        # self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)
        self.rel_compress = BayesHead(self.pooling_dim)
        self.ctx_compress = BayesHead(self.hidden_dim * 2)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        # layer_init(self.rel_compress, xavier=True)
        # layer_init(self.ctx_compress, xavier=True)
        self.rel_compress.layer_init()
        self.ctx_compress.layer_init()

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

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
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features
        rel_rel1, rel_rel2, rel_rel3, rel_super = self.rel_compress(visual_rep)
        ctx_rel1, ctx_rel2, ctx_rel3, ctx_super = self.ctx_compress(prod_rep)
        rel1_logits = rel_rel1 + ctx_rel1
        rel2_logits = rel_rel2 + ctx_rel2
        rel3_logits = rel_rel3 + ctx_rel3
        super_logits = rel_super + ctx_super

        super_relation = F.log_softmax(super_logits, dim=1)
        relation_1 = F.log_softmax(rel1_logits, dim=1) + super_relation[:, 1].view(-1, 1)
        relation_2 = F.log_softmax(rel2_logits, dim=1) + super_relation[:, 2].view(-1, 1)
        relation_3 = F.log_softmax(rel3_logits, dim=1) + super_relation[:, 3].view(-1, 1)

        obj_dists = obj_dists.split(num_objs, dim=0)
        relation1_dist = relation_1.split(num_rels, dim=0)
        relation2_dist = relation_2.split(num_rels, dim=0)
        relation3_dist = relation_3.split(num_rels, dim=0)
        superrelation_dist = super_relation.split(num_rels, dim=0)

        add_losses = {}

        return obj_dists, [relation1_dist, relation2_dist, relation3_dist, superrelation_dist], add_losses

@registry.ROI_RELATION_PREDICTOR.register("MotifHierarchicalPredictor")
class MotifHierarchicalPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifHierarchicalPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.USE_UNION_FEATURES
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        
        if self.attribute_on:
            att_classes = statistics['att_classes']
            assert self.num_att_cls == len(att_classes)

        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = BayesHead(input_dim=self.pooling_dim)
        # self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        self.rel_compress.layer_init()
        # layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        self.geo_label = ['1', '2', '3', '4', '5', '6', '8', '10', '22', '23', '29', '31', '32', '33', '43']
        self.pos_label = ['9', '16', '17', '20', '27', '30', '36', '42', '48', '49', '50']
        self.sem_label = ['7', '11', '12', '13', '14', '15', '18', '19', '21', '24', '25', '26', '28', '34', '35', '37',
                          '38', '39', '40', '41', '44', '45', '46', '47']
        self.geo_label_tensor = torch.tensor([int(x) for x in self.geo_label])
        self.pos_label_tensor = torch.tensor([int(x) for x in self.pos_label])
        self.sem_label_tensor = torch.tensor([int(x) for x in self.sem_label])

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
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

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel1_logits, rel2_logits, rel3_logits, super_logits = self.rel_compress(prod_rep)
        if self.use_bias:
            # bias dimension is 51, already include the background(0)
            bias = self.freq_bias.index_with_labels(pair_pred.long())  # (rel, 51)
            rel1_bias = bias[:, self.geo_label_tensor]  # (rel, 15)
            rel2_bias = bias[:, self.pos_label_tensor]  # (rel, 11)
            rel3_bias = bias[:, self.sem_label_tensor]  # (rel, 24)
            super_bias = torch.stack(
                (
                    torch.exp(rel1_bias).sum(dim=1),
                    torch.exp(rel2_bias).sum(dim=1),
                    torch.exp(rel3_bias).sum(dim=1),
                ),
                dim=1  # Stack along the second dimension to match the shape (rel, 3)
            )

            super_bias = torch.log(super_bias)

            rel1_logits = rel1_logits + rel1_bias
            rel2_logits = rel2_logits + rel2_bias
            rel3_logits = rel3_logits + rel3_bias
            super_logits[:, 1:] = super_logits[:, 1:] + super_bias  # ignore the background class

        # SOFTMAX
        super_relation = F.log_softmax(super_logits, dim=1)
        relation_1 = F.log_softmax(rel1_logits, dim=1) + super_relation[:, 1].view(-1, 1)
        relation_2 = F.log_softmax(rel2_logits, dim=1) + super_relation[:, 2].view(-1, 1)
        relation_3 = F.log_softmax(rel3_logits, dim=1) + super_relation[:, 3].view(-1, 1)

        obj_dists = obj_dists.split(num_objs, dim=0)
        relation1_dist = relation_1.split(num_rels, dim=0)
        relation2_dist = relation_2.split(num_rels, dim=0)
        relation3_dist = relation_3.split(num_rels, dim=0)
        superrelation_dist = super_relation.split(num_rels, dim=0)

        add_losses = {}

        return obj_dists, [relation1_dist, relation2_dist, relation3_dist, superrelation_dist], add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreeHierPredictor")
class VCTreeHierPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreeHierPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        
        if self.attribute_on:
            att_classes = statistics['att_classes']
            assert self.num_att_cls == len(att_classes)

        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        # self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = BayesHeadProb(self.pooling_dim)
        self.ctx_compress.layer_init()
        # self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # layer_init(self.uni_gate, xavier=True)
        # layer_init(self.frq_gate, xavier=True)
        # layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs,
                                                                          logger)

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
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        # uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        # frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # ctx_dists = self.ctx_compress(prod_rep * union_features)
        relation_1, relation_2, relation_3, super_relation = self.ctx_compress(prod_rep * union_features)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        obj_dists = obj_dists.split(num_objs, dim=0)
        relation1_dist = relation_1.split(num_rels, dim=0)
        relation2_dist = relation_2.split(num_rels, dim=0)
        relation3_dist = relation_3.split(num_rels, dim=0)
        superrelation_dist = super_relation.split(num_rels, dim=0)

        add_losses = {}
        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, [relation1_dist, relation2_dist, relation3_dist, superrelation_dist], add_losses


@registry.ROI_RELATION_PREDICTOR.register("CausalAnalysisHierPredictor")
class CausalAnalysisHierPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(CausalAnalysisHierPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.USE_SPATIAL_FEATURES
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        if config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse":
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        # TODO(zhijunz): Also modify for VTranse
        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                            nn.ReLU(inplace=True), ])
            # self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
            # layer_init(self.ctx_compress, xavier=True)
            self.ctx_compress = BayesHead(input_dim=self.pooling_dim)
            self.ctx_compress.layer_init()
        # self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # layer_init(self.vis_compress, xavier=True)
        self.vis_compress = BayesHead(input_dim=self.pooling_dim)
        self.vis_compress.layer_init()

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        if not self.use_vtranse:
            layer_init(self.post_cat[0], xavier=True)

        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(self.hidden_dim, self.pooling_dim),
                                           nn.ReLU(inplace=True)
                                           ])
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

        self.geo_label = ['1', '2', '3', '4', '5', '6', '8', '10', '22', '23', '29', '31', '32', '33', '43']
        self.pos_label = ['9', '16', '17', '20', '27', '30', '36', '42', '48', '49', '50']
        self.sem_label = ['7', '11', '12', '13', '14', '15', '18', '19', '21', '24', '25', '26', '28', '34', '35', '37',
                          '38', '39', '40', '41', '44', '45', '46', '47']
        self.geo_mapping = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 10: 7, 22: 8, 23: 9, 29: 10,
            31: 11, 32: 12, 33: 13, 43: 14
        }

        self.pos_mapping = {
            9: 0, 16: 1, 17: 2, 20: 3, 27: 4, 30: 5, 36: 6, 42: 7, 48: 8, 49: 9, 50: 10
        }
        self.sem_mapping = {
            7: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 18: 6, 19: 7, 21: 8, 24: 9, 25: 10,
            26: 11, 28: 12, 34: 13, 35: 14, 37: 15, 38: 16, 39: 17, 40: 18, 41: 19,
            44: 20, 45: 21, 46: 22, 47: 23
        }
        self.geo_label_tensor = torch.tensor([int(x) for x in self.geo_label])
        self.pos_label_tensor = torch.tensor([int(x) for x in self.pos_label])
        self.sem_label_tensor = torch.tensor([int(x) for x in self.sem_label])

    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger,
                              ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs,
                                                                          logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps,
                                                                             obj_preds, obj_boxs, obj_prob_list):
            if self.use_vtranse:
                ctx_reps.append(head_rep[pair_idx[:, 0]] - tail_rep[pair_idx[:, 1]])
            else:
                ctx_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_obj_probs.append(torch.stack((obj_prob[pair_idx[:, 0]], obj_prob[pair_idx[:, 1]]), dim=2))
            pair_bboxs_info.append(get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]]))
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list

    def calculate_hier_label_mask(self, rel_labels):
        cur_device = rel_labels.device
        geo_label_tensor = self.geo_label_tensor.to(cur_device)
        pos_label_tensor = self.pos_label_tensor.to(cur_device)
        sem_label_tensor = self.sem_label_tensor.to(cur_device)
        geo_label_mask = (rel_labels.unsqueeze(1) == geo_label_tensor).any(1)
        pos_label_mask = (rel_labels.unsqueeze(1) == pos_label_tensor).any(1)
        sem_label_mask = (rel_labels.unsqueeze(1) == sem_label_tensor).any(1)
        super_rel_label = geo_label_mask + pos_label_mask * 2 + sem_label_mask * 3

        geo_labels = rel_labels[geo_label_mask]
        geo_labels = torch.tensor([self.geo_mapping[label.item()] for label in geo_labels]).to(cur_device)
        pos_labels = rel_labels[pos_label_mask]
        pos_labels = torch.tensor([self.pos_mapping[label.item()] for label in pos_labels]).to(cur_device)
        sem_labels = rel_labels[sem_label_mask]
        sem_labels = torch.tensor([self.sem_mapping[label.item()] for label in sem_labels]).to(cur_device)

        return geo_label_mask, pos_label_mask, sem_label_mask, geo_labels, pos_labels, sem_labels, super_rel_label


    def calculate_loss_hier(self, rel1_logits, rel2_logits, rel3_logits, super_rel_logits,
                            geo_label_mask, pos_label_mask, sem_label_mask, geo_labels, pos_labels, sem_labels, super_rel_label):
        loss_relation = 0
        if geo_labels.shape[0] > 0:
            loss_relation += F.cross_entropy(rel1_logits[geo_label_mask], geo_labels)
        if pos_labels.shape[0] > 0:
            loss_relation += F.cross_entropy(rel2_logits[pos_label_mask], pos_labels)
        if sem_labels.shape[0] > 0:
            loss_relation += F.cross_entropy(rel3_logits[sem_label_mask], sem_labels)
        if super_rel_label.shape[0] > 0:
            loss_relation += F.cross_entropy(super_rel_logits[super_rel_label], super_rel_label)

        return loss_relation

    def calculate_bias_hier(self, pair_pred):
        """
        :param pair_pred(or frq_rep): [num_rels, ?]
        """
        bias = self.freq_bias.index_with_labels(pair_pred.long())  # (rel, 51)
        bg_bias = bias[:, 0]  # (rel, 1)
        rel1_bias = bias[:, self.geo_label_tensor]  # (rel, 15)
        rel2_bias = bias[:, self.pos_label_tensor]  # (rel, 11)
        rel3_bias = bias[:, self.sem_label_tensor]  # (rel, 24)
        super_bias = torch.stack(
            (
                torch.exp(bg_bias),
                torch.exp(rel1_bias).sum(dim=1),
                torch.exp(rel2_bias).sum(dim=1),
                torch.exp(rel3_bias).sum(dim=1),
            ),
            dim=1  # Stack along the second dimension to match the shape (rel, 3)
        )
        # print(super_bias.shape)
        # print(super_bias)
        # print(torch.exp(bias[:, 0]))self
        super_bias = torch.log(super_bias)
        # print(super_bias.shape)
        # print(super_bias)

        return rel1_bias, rel2_bias, rel3_bias, super_bias

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        # TODO(zhijunz): now only use index_with_labels
        # rel1_bias, rel2_bias, rel3_bias, super_bias = self.calculate_bias_hier(frq_rep)

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        # vis_dists = self.vis_compress(vis_rep)
        # ctx_dists = self.ctx_compress(ctx_rep)
        vis_rel1_logits, vis_rel2_logits, vis_rel3_logits, vis_super_logits = self.vis_compress(vis_rep)
        ctx_rel1_logits, ctx_rel2_logits, ctx_rel3_logits, ctx_super_logits = self.ctx_compress(ctx_rep)

        # TODO(zhijunz): now only use sum
        # elif self.fusion_type == 'sum':
        #     union_dists = vis_dists + ctx_dists + frq_dists

        # union_rel1_logits = vis_rel1_logits + ctx_rel1_logits + rel1_bias
        # union_rel2_logits = vis_rel2_logits + ctx_rel2_logits + rel2_bias
        # uninon_rel3_logits = vis_rel3_logits + ctx_rel3_logits + rel3_bias
        # super_logits = vis_super_logits + ctx_super_logits + super_bias

        union_rel1_logits = vis_rel1_logits + ctx_rel1_logits
        union_rel2_logits = vis_rel2_logits + ctx_rel2_logits
        uninon_rel3_logits = vis_rel3_logits + ctx_rel3_logits
        super_logits = vis_super_logits + ctx_super_logits

        return union_rel1_logits, union_rel2_logits, uninon_rel3_logits, super_logits


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Args:
            proposals: list[BoxList], len(proposals) == batch_size
            rel_pair_idxs: list[Tensor], len(rel_pair_idxs) == batch_size
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(
            roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _ = self.pair_feature_generate(roi_features,
                                                                                                   proposals,
                                                                                                   rel_pair_idxs,
                                                                                                   num_objs, obj_boxs,
                                                                                                   logger,
                                                                                                   ctx_average=True)

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats

        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        # rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
        # rel_dist_list = rel_dists.split(num_rels, dim=0)
        rel1_logits, rel2_logits, rel3_logits, super_logits = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            # branch constraint: make sure each branch can predict independently
            vis_rel1_logits, vis_rel2_logits, vis_rel3_logits, vis_super_logits = self.vis_compress(union_features)
            ctx_rel1_logits, ctx_rel2_logits, ctx_rel3_logits, ctx_super_logits = self.ctx_compress(post_ctx_rep)
            geo_label_mask, pos_label_mask, sem_label_mask, geo_labels, pos_labels, sem_labels, super_rel_label = \
                self.calculate_hier_label_mask(rel_labels)
            add_losses['auxiliary_ctx'] = \
            self.calculate_loss_hier(vis_rel1_logits, vis_rel2_logits, vis_rel3_logits, vis_super_logits,
                                      geo_label_mask, pos_label_mask, sem_label_mask, geo_labels, pos_labels,
                                      sem_labels, super_rel_label)
            if not (self.fusion_type == 'gate'):
                add_losses['auxiliary_vis'] = \
                self.calculate_loss_hier(ctx_rel1_logits, ctx_rel2_logits, ctx_rel3_logits, ctx_super_logits,
                                            geo_label_mask, pos_label_mask, sem_label_mask, geo_labels, pos_labels,
                                            sem_labels, super_rel_label)
                # add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()),rel_labels)

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1,
                                                                                          -1) if self.separate_spatial else avg_ctx_rep
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == 'TDE':  # TDE of CTX
                # TODO(zhijunz): now only use pair_pred and TDE
                # rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(
                #     union_features, avg_ctx_rep, pair_obj_probs)
                rel1_logits_1, rel2_logits_1, rel3_logits_1, super_logits_1 = self.calculate_logits(union_features, post_ctx_rep, pair_pred, False)
                rel1_logits_2, rel2_logits_2, rel3_logits_2, super_logits_2 = self.calculate_logits(union_features, avg_ctx_rep, pair_pred, False)
                rel1_logits = rel1_logits_1 - rel1_logits_2
                rel2_logits = rel2_logits_1 - rel2_logits_2
                rel3_logits = rel3_logits_1 - rel3_logits_2
                super_logits = super_logits_1 - super_logits_2

            elif self.effect_type == 'NIE':  # NIE of FRQ
                rel_dists = self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) - self.calculate_logits(
                    union_features, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == 'TE':  # Total Effect
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(
                    union_features, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == 'none'
                pass
            # rel_dist_list = rel_dists.split(num_rels, dim=0)

        # SOFTMAX
        super_relation = F.log_softmax(super_logits, dim=1)
        relation_1 = F.log_softmax(rel1_logits, dim=1) + super_relation[:, 1].view(-1, 1)
        relation_2 = F.log_softmax(rel2_logits, dim=1) + super_relation[:, 2].view(-1, 1)
        relation_3 = F.log_softmax(rel3_logits, dim=1) + super_relation[:, 3].view(-1, 1)

        relation1_dist = relation_1.split(num_rels, dim=0)
        relation2_dist = relation_2.split(num_rels, dim=0)
        relation3_dist = relation_3.split(num_rels, dim=0)
        superrelation_dist = super_relation.split(num_rels, dim=0)

        return obj_dist_list, [relation1_dist, relation2_dist, relation3_dist, superrelation_dist], add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder


    def binary_ce_loss(self, logits, gt):
        batch_size, num_cat = logits.shape
        answer = torch.zeros((batch_size, num_cat), device=gt.device).float()
        answer[torch.arange(batch_size, device=gt.device), gt.long()] = 1.0
        return F.binary_cross_entropy_with_logits(logits, answer) * num_cat

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)