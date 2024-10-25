from torch import nn
from torch.nn import functional as F
from sgg_benchmark.modeling.utils import cat
from sgg_benchmark.modeling.make_layers import make_fc
from sgg_benchmark.utils.txt_embeddings import obj_edge_vectors
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from .utils_motifs import encode_box_info
from sgg_benchmark.data import get_dataset_statistics
import torch

class PairwiseFeatureExtractor(nn.Module):
    """
    extract the pairwise features from the object pairs and union features.
    most pipeline keep same with the motifs instead the lstm massage passing process
    """

    def __init__(self, config, in_channels):
        super(PairwiseFeatureExtractor, self).__init__()
        self.cfg = config
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        # self.cfg.MODEL.ROI_RELATION_HEAD.REL_PROP = statistics['pred_freq']
        self.num_obj_classes = len(obj_classes)
        self.num_rel_classes = len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # features augmentation for instance features
        # word embedding
        # add language prior representation according to the prediction distribution
        # of objects
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.obj_dim = in_channels
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.spatial_for_vision = self.cfg.MODEL.ROI_RELATION_HEAD.USE_SPATIAL_FEATURES

        self.word_embed_feats_on = True
        if self.word_embed_feats_on:
            obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_type=self.cfg.MODEL.TEXT_EMBEDDING,  wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
            self.obj_embed_on_prob_dist = nn.Embedding(self.num_obj_classes, self.embed_dim)
            self.obj_embed_on_pred_label = nn.Embedding(self.num_obj_classes, self.embed_dim)
            with torch.no_grad():
                self.obj_embed_on_prob_dist.weight.copy_(obj_embed_vecs, non_blocking=True)
                self.obj_embed_on_pred_label.weight.copy_(obj_embed_vecs, non_blocking=True)
        else:
            self.embed_dim = 0

        # features augmentation for rel pairwise features
        self.rel_feature_type = "obj_pair" if self.cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "SquatPredictor" else "fusion"

        # the input dimension is ROI head MLP, but the inner module is pooling dim, so we need
        # to decrease the dimension first.
        if self.pooling_dim != in_channels:
            self.rel_feat_dim_not_match = True
            self.rel_feature_up_dim = make_fc(in_channels, self.pooling_dim)
            layer_init(self.rel_feature_up_dim, xavier=True)
        else:
            self.rel_feat_dim_not_match = False

        self.pairwise_obj_feat_updim_fc = make_fc(self.hidden_dim + self.obj_dim + self.embed_dim,
                                                  self.hidden_dim * 2)

        self.outdim = self.pooling_dim
        # position embedding
        # encode the geometry information of bbox in relationships
        self.geometry_feat_dim = 128
        self.pos_embed = nn.Sequential(*[
            make_fc(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            make_fc(32, self.geometry_feat_dim), nn.ReLU(inplace=True),
        ])

        if self.rel_feature_type in ["obj_pair", "fusion"]:
            if self.spatial_for_vision:
                self.spt_emb = nn.Sequential(*[make_fc(32, self.hidden_dim),
                                               nn.ReLU(inplace=True),
                                               make_fc(self.hidden_dim, self.hidden_dim * 2),
                                               nn.ReLU(inplace=True)
                                               ])
                layer_init(self.spt_emb[0], xavier=True)
                layer_init(self.spt_emb[2], xavier=True)

            self.pairwise_rel_feat_finalize_fc = nn.Sequential(
                make_fc(self.hidden_dim * 2, self.pooling_dim),
                nn.ReLU(inplace=True),
            )

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.obj_hidden_linear = make_fc(self.obj_dim + self.embed_dim + self.geometry_feat_dim, self.hidden_dim)

        self.obj_feat_aug_finalize_fc = nn.Sequential(
            make_fc(self.hidden_dim + self.obj_dim + self.embed_dim, self.pooling_dim),
            nn.ReLU(inplace=True),
        )

        # untreated average features

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def pairwise_rel_features(self, augment_obj_feat, union_features, rel_pair_idxs, inst_proposals):
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in inst_proposals]
        num_objs = [len(p) for p in inst_proposals]
        # post decode
        # (num_objs, hidden_dim) -> (num_objs, hidden_dim * 2)
        # going to split single object representation to sub-object role of relationship
        pairwise_obj_feats_fused = self.pairwise_obj_feat_updim_fc(augment_obj_feat)
        pairwise_obj_feats_fused = pairwise_obj_feats_fused.view(pairwise_obj_feats_fused.size(0), 2, self.hidden_dim)
        head_rep = pairwise_obj_feats_fused[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = pairwise_obj_feats_fused[:, 1].contiguous().view(-1, self.hidden_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        # generate the pairwise object for relationship representation
        # (num_objs, hidden_dim) <rel pairing > (num_objs, hidden_dim)
        #   -> (num_rel, hidden_dim * 2)
        #   -> (num_rel, hidden_dim)
        obj_pair_feat4rel_rep = []
        pair_bboxs_info = []

        for pair_idx, head_rep, tail_rep, obj_box in zip(rel_pair_idxs, head_reps, tail_reps, obj_boxs):
            obj_pair_feat4rel_rep.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_bboxs_info.append(get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]]))
        pair_bbox_geo_info = cat(pair_bboxs_info, dim=0)
        obj_pair_feat4rel_rep = cat(obj_pair_feat4rel_rep, dim=0)  # (num_rel, hidden_dim * 2)
        if self.spatial_for_vision:
            obj_pair_feat4rel_rep = obj_pair_feat4rel_rep * self.spt_emb(pair_bbox_geo_info)

        obj_pair_feat4rel_rep = self.pairwise_rel_feat_finalize_fc(obj_pair_feat4rel_rep)  # (num_rel, hidden_dim)

        return obj_pair_feat4rel_rep

    def forward(self, inst_roi_feats, union_features, inst_proposals, rel_pair_idxs, use_union=True):
        """

        :param inst_roi_feats: instance ROI features, list(Tensor)
        :param inst_proposals: instance proposals, list(BoxList())
        :param rel_pair_idxs:
        :return:
            obj_pred_logits obj_pred_labels 2nd time instance classification results
            obj_representation4rel, the objects features ready for the represent the relationship
        """
        # using label or logits do the label space embeddings
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX or self.cfg.MODEL.BACKBONE.FREEZE:
            obj_labels = cat([proposal.get_field("labels") for proposal in inst_proposals], dim=0)
        else:
            obj_labels = None

        if self.word_embed_feats_on:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL or self.cfg.MODEL.BACKBONE.FREEZE:
                obj_embed_by_pred_dist = self.obj_embed_on_prob_dist(obj_labels.long())
            else:
                obj_logits = cat([proposal.get_field("predict_logits") for proposal in inst_proposals], dim=0).detach()
                obj_embed_by_pred_dist = F.softmax(obj_logits, dim=1) @ self.obj_embed_on_prob_dist.weight

        # box positive geometry embedding
        assert inst_proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(inst_proposals))

        # word embedding refine
        if self.word_embed_feats_on:
            obj_pre_rep = cat((inst_roi_feats, obj_embed_by_pred_dist, pos_embed), -1)
        else:
            obj_pre_rep = cat((inst_roi_feats, pos_embed), -1)
        # object level contextual feature
        augment_obj_feat = self.obj_hidden_linear(obj_pre_rep)  # map to hidden_dim

        # todo reclassify on the fused object features
        # Decode in order
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL or self.cfg.MODEL.BACKBONE.FREEZE:
            assert obj_labels is not None
            obj_pred_labels = obj_labels
        else:
            # todo: currently no redo classification on embedding representation,
            #       we just use the first stage object prediction
            obj_pred_labels = cat([each_prop.get_field("predict_logits").argmax(-1) for each_prop in inst_proposals], dim=0)

        # object labels space embedding from the prediction labels
        if self.word_embed_feats_on:
            obj_embed_by_pred_labels = self.obj_embed_on_pred_label(obj_pred_labels.long())

        # average action in test phrase for causal effect analysis
        if self.word_embed_feats_on:
            augment_obj_feat = cat((obj_embed_by_pred_labels, inst_roi_feats, augment_obj_feat), -1)
        else:
            augment_obj_feat = cat((inst_roi_feats, augment_obj_feat), -1)

        if use_union:
            if self.rel_feature_type == "obj_pair" or self.rel_feature_type == "fusion":
                rel_features = self.pairwise_rel_features(augment_obj_feat, union_features,
                                                        rel_pair_idxs, inst_proposals)
                if self.rel_feature_type == "fusion":
                    if self.rel_feat_dim_not_match:
                        union_features = self.rel_feature_up_dim(union_features)
                    rel_features = union_features + rel_features

            elif self.rel_feature_type == "union":
                if self.rel_feat_dim_not_match:
                    union_features = self.rel_feature_up_dim(union_features)
                rel_features = union_features

            else:
                assert False
        else:
            rel_features = self.pairwise_rel_features(augment_obj_feat, None,
                                                        rel_pair_idxs, inst_proposals)
        # mapping to hidden
        augment_obj_feat = self.obj_feat_aug_finalize_fc(augment_obj_feat)

        return augment_obj_feat, rel_features