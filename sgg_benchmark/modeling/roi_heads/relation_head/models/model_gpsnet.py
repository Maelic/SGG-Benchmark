import copy

# import ipdb
import torch
import torch.nn as nn
from torch.nn import functional as F

from sgg_benchmark.modeling.make_layers import make_fc
from sgg_benchmark.structures.boxlist_ops import squeeze_tensor
from sgg_benchmark.modeling.utils import cat
from sgg_benchmark.modeling.roi_heads.relation_head.models.utils.classifiers import build_classifier
from sgg_benchmark.modeling.roi_heads.relation_head.models.utils.utils_motifs import to_onehot
from sgg_benchmark.modeling.roi_heads.relation_head.models.utils.utils_relation import obj_prediction_nms
from sgg_benchmark.modeling.roi_heads.relation_head.models.utils.pairwise_feats import PairwiseFeatureExtractor
class GatingModel(nn.Module):
    def __init__(self, entity_input_dim, union_input_dim, hidden_dim, filter_dim=32):
        super(GatingModel, self).__init__()

        self.entity_input_dim = entity_input_dim
        self.union_input_dim = union_input_dim
        self.hidden_dim = hidden_dim

        self.ws = nn.Sequential(
            # nn.BatchNorm1d(self.entity_input_dim),
            make_fc(self.entity_input_dim, self.hidden_dim, ),
            nn.ReLU(),
        )
        self.wo = nn.Sequential(
            # nn.BatchNorm1d(self.entity_input_dim),
            make_fc(self.entity_input_dim, self.hidden_dim, ),
            nn.ReLU(),
        )

        self.wu = nn.Sequential(
            # nn.BatchNorm1d(self.union_input_dim),
            make_fc(self.union_input_dim, self.hidden_dim, ),
            nn.ReLU(),
        )

        self.w = nn.Sequential(
            # nn.BatchNorm1d(self.hidden_dim),
            make_fc(self.hidden_dim, filter_dim, ),
            nn.ReLU(),
        )

    def forward(self, subj_feat, obj_feat, rel_feat):
        prod = self.ws(subj_feat) * self.wo(obj_feat)
        atten_f = self.w(prod * self.wu(rel_feat))

        # average the nodes attention between the nodes
        if atten_f.shape[1] > 1:
            atten_f = atten_f.mean(1)

        return squeeze_tensor(atten_f)


def multichnl_matmul(tensor3d, mat):
    """
    tensor3d N x M x C
    mat M x N

    return:  N x C * N
    """
    out = []
    for i in range(tensor3d.shape[-1]):  # for each channel
        out.append(torch.mm(tensor3d[:, :, i], mat))
    return torch.cat(out, -1)


class MessageGenerator(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(MessageGenerator, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        self.output_fc = nn.Sequential(nn.Linear(self.input_dims, self.input_dims // 4),
                                       nn.LayerNorm(self.input_dims // 4),
                                       nn.ReLU(),
                                       nn.Linear(self.input_dims // 4, self.hidden_dim),
                                       nn.ReLU(), )

        self.message_fc = nn.Sequential(
            nn.Linear(self.input_dims, self.input_dims // 2))  # down dim for the bidirectional message

    def forward(self, source_features, weighting_gate, rel_pair_idx, relness_score=None):
        n_nodes = source_features.shape[0]

        # apply a masked softmax on the attention logits
        def masked_softmax(weighting_gate, rel_pair_idx):
            atten_mat_ex = torch.zeros((n_nodes, n_nodes), dtype=source_features.dtype, device=source_features.device)
            atten_mat_mask = torch.zeros((n_nodes, n_nodes), dtype=source_features.dtype, device=source_features.device)
            atten_mat_mask[rel_pair_idx[:, 0], rel_pair_idx[:, 1]] = 1.
            atten_mat_ex[rel_pair_idx[:, 0], rel_pair_idx[:, 1]] = weighting_gate

            atten_mat_ex = (atten_mat_ex - atten_mat_ex.max()).exp() * atten_mat_mask  # e^x of each position in vaild pairs
            atten_mat = atten_mat_ex / (atten_mat_ex.sum(1).unsqueeze(1) + 1e-6)
            return atten_mat

        # the softmax is better than sigmoid,
        # the competition between the nodes when msp may decrease the search space
        def masked_sigmoid(weighting_gate, rel_pair_idx):
            atten_mat_ex = torch.zeros((n_nodes, n_nodes), dtype=source_features.dtype, device=source_features.device)
            atten_mat_mask = torch.zeros((n_nodes, n_nodes), dtype=source_features.dtype, device=source_features.device)
            atten_mat_mask[rel_pair_idx[:, 0], rel_pair_idx[:, 1]] = 1.
            atten_mat_ex[rel_pair_idx[:, 0], rel_pair_idx[:, 1]] = weighting_gate

            atten_mat = torch.sigmoid(atten_mat_ex) * atten_mat_mask
            return atten_mat

        atten_mat = masked_softmax(weighting_gate, rel_pair_idx)

        # apply the relness scores
        if relness_score is not None:
            relness_mat = torch.zeros((n_nodes, n_nodes), dtype=source_features.dtype, device=source_features.device)
            relness_mat[rel_pair_idx[:, 0], rel_pair_idx[:, 1]] = relness_score
            atten_mat *= relness_mat

        # bidirectional msp attention
        atten_mat_t = atten_mat.transpose(1, 0)
        atten_mat_bidi = torch.stack((atten_mat, atten_mat_t), -1)

        # N x N x 2 mul N x M --> N x M
        vaild_msg_idx = squeeze_tensor(atten_mat.sum(1).nonzero())
        message_feats = multichnl_matmul(atten_mat_bidi, self.message_fc(source_features))

        padded_msg_feat = torch.zeros((message_feats.shape[0], self.hidden_dim),
                                      dtype=source_features.dtype, device=source_features.device)
        padded_msg_feat[vaild_msg_idx] += self.output_fc(torch.index_select(message_feats, 0, vaild_msg_idx))

        return padded_msg_feat


class MessagePassingUnit(nn.Module):
    def __init__(self, hidden_dim, filter_dim=128):
        """

        Args:
            input_dim:
            filter_dim: the channel number of attention between the nodes
        """
        super(MessagePassingUnit, self).__init__()

        self.w = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, filter_dim, bias=True),
        )
        self.fea_size = hidden_dim
        self.filter_size = filter_dim

    def forward(self, unary_term, pair_term):

        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.size()[1])

        paired_feats = torch.cat([unary_term, pair_term], 1)

        try:
            gate = torch.sigmoid(self.w(paired_feats))
        except ValueError:
            # less than one element, don't use the bn
            gate = torch.sigmoid(self.w[1:](paired_feats))

        if gate.shape[1] > 1:
            gate = gate.mean(1)  # average the nodes attention between the nodes

        # print 'gate', gate
        output = pair_term * gate.view(-1, 1).expand(gate.size()[0], pair_term.size()[1])

        return output, gate


class UpdateUnit(nn.Module):
    def __init__(self, input_dim_ih, input_dim_hh, output_dim, dropout=False):
        super(UpdateUnit, self).__init__()

        # don't add dropout
        self.wih = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim_ih, output_dim, bias=True)
        )
        self.whh = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim_hh, output_dim, bias=True)
        )


    def forward(self, input_feat, hidden_feat):
        output = self.wih(input_feat) + self.whh(hidden_feat)

        return output


class GPSNetContext(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels, hidden_dim=512, num_iter=2, dropout=False, ):
        super(GPSNetContext, self).__init__()
        self.cfg = config
        self.filter_the_mp_instance = False
        self.relness_weighting_mp = False

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        ########## constants ########
        self.hidden_dim = hidden_dim
        self.update_step = num_iter
        self.pooling_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.num_rel_cls = self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.num_obj_cls = self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        if self.update_step < 1:
            print(
                "WARNING: the update_step should be greater than 0, current: ", + self.update_step)

        self.pairwise_feature_extractor = PairwiseFeatureExtractor(self.cfg, in_channels)

        self.pairwise_obj_feat_updim_fc = nn.Sequential(
            make_fc(self.pooling_dim, self.hidden_dim * 2),
            nn.ReLU()
        )

        self.pairwise_rel_feat_finalize_fc = nn.Sequential(
            make_fc(self.hidden_dim * 2, self.pooling_dim),
            nn.ReLU(),
        )

        self.obj2obj_gating_model = GatingModel(self.pooling_dim, self.pooling_dim, self.hidden_dim)
        self.obj2obj_msg_gen = MessageGenerator(self.pooling_dim, self.hidden_dim)

        self.sub2pred_msp = MessagePassingUnit(self.hidden_dim, 64)
        self.obj2pred_msp = MessagePassingUnit(self.hidden_dim, 64)

        self.rel_feat_update_downdim_fc = nn.Sequential(
            nn.BatchNorm1d(self.pooling_dim),
            nn.ReLU(),
            nn.Linear(self.pooling_dim, self.hidden_dim)
        )

        self.rel_feat_update_inst_feat_downdim_fc = nn.Sequential(
            nn.BatchNorm1d(self.pooling_dim),
            nn.ReLU(),
            make_fc(self.pooling_dim, self.hidden_dim * 2),
        )
        self.rel_feat_update_unit = UpdateUnit(self.hidden_dim, self.pooling_dim,
                                               self.pooling_dim, dropout=False)

        # here has a FFN bottleneck for each step of msp
        self.inst_feat_down_dim_fcs = nn.ModuleList(
            [
                nn.Sequential(
                    make_fc(self.pooling_dim, self.hidden_dim),
                    nn.ReLU()
                ) for _ in range(self.update_step)

            ]
        )

        self.obj2obj_msg_fuse = nn.Sequential(
            make_fc(self.hidden_dim, self.pooling_dim),
            nn.ReLU()
        )

        self.obj_classifier = build_classifier(self.pooling_dim, self.num_obj_cls)
        self.obj_classifier.reset_parameters()
        self.obj_decode = not (self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX or self.cfg.MODEL.BACKBONE.FREEZE)

    def _pre_predciate_classification(self, relatedness_scores, proposals, rel_pair_inds,
                                      refine_iter, refine_rel_feats_each_iters):
        if self.mp_pair_refine_iter > 1:
            pre_cls_logits, \
            pred_relatedness_scores = self.pre_rel_classifier[refine_iter](refine_rel_feats_each_iters[-1],
                                                                           proposals, rel_pair_inds)
        else:
            pre_cls_logits, \
            pred_relatedness_scores = self.pre_rel_classifier(refine_rel_feats_each_iters[-1],
                                                              proposals, rel_pair_inds)

        # update the relness container for output
        for idx, pairs in enumerate(rel_pair_inds):
            relatedness_scores[idx][pairs[:, 0], pairs[:, 1]] = pred_relatedness_scores[idx][pairs[:, 0], pairs[:, 1]]

        updated_relness_score = [each[:] for each in relatedness_scores]

        return pre_cls_logits, updated_relness_score

    def _prepare_adjacency_matrix(self, num_proposals, valid_inst_idx, rel_pair_idxs, relatedness):
        """
        prepare the index of how subject and object related to the union boxes
        Args:
            num_proposals:
            valid_inst_idx:  todo: use for filter the invalid entities
            rel_pair_idxs:
            relatedness:

        return:
            rel_inds,
                extent the instances pairing matrix to the batch wised (num_rel, 2)
            subj_pred_map,
                how the instances related to the relation predicates as the subject (num_inst, rel_pair_num)
            obj_pred_map
                how the instances related to the relation predicates as the object (num_inst, rel_pair_num)
            rel_prop_pairs_relness_batch_cat,
                relness score for selected rel pairs (num_rel, )
            selected_rel_prop_pairs_idx:
                the valid rel prop pairs indexs for the msp (num_vaild_rel, )

        """

        rel_inds = []
        offset = 0
        rel_prop_pairs_relness_batch = []

        for idx, (prop_num, rel_ind_i) in enumerate(zip(num_proposals, rel_pair_idxs, )):
            if self.filter_the_mp_instance:
                assert relatedness is not None
                related_matrix = relatedness[idx]
                rel_prop_pairs_relness = related_matrix[rel_ind_i[:, 0], rel_ind_i[:, 1]]
                # get the valid relation pair for the message passing
                rel_prop_pairs_relness_batch.append(rel_prop_pairs_relness)
            rel_ind_i = copy.deepcopy(rel_ind_i)

            rel_ind_i += offset
            offset += prop_num
            rel_inds.append(rel_ind_i)

        rel_inds = torch.cat(rel_inds, 0)

        subj_pred_map = rel_inds.new(sum(num_proposals), rel_inds.shape[0]).fill_(0).float().detach()
        obj_pred_map = rel_inds.new(sum(num_proposals), rel_inds.shape[0]).fill_(0).float().detach()
        # only message passing on valid pairs

        subj_pred_map.scatter_(0, (rel_inds[:, 0].contiguous().view(1, -1)), 1)
        obj_pred_map.scatter_(0, (rel_inds[:, 1].contiguous().view(1, -1)), 1)
        rel_prop_pairs_relness_batch_cat = None
        selected_rel_prop_pairs_idx = torch.arange(len(rel_inds), dtype=torch.int64).to(rel_inds.device)

        return rel_inds, subj_pred_map, obj_pred_map, rel_prop_pairs_relness_batch_cat, selected_rel_prop_pairs_idx

    def prepare_message(self, target_features, source_features, rel_feat, rel_pair_idx,
                        gate_module: GatingModel, message_gener: MessageGenerator, relness_score=None):
        """
        build up the adjacency matrix for indicating how the instance and predicates connect,
        Then the message passing process can be
        :param target_features: (num_inst, dim)
        :param source_features: (num_rel, dim)
        :param select_mat:  (num_rel, 2)
        :param gate_module:
        :param relness_score: (num_rel, )
        :return:
        """

        source_indices = rel_pair_idx[:, 1]
        target_indices = rel_pair_idx[:, 0]

        source_f = torch.index_select(source_features, 0, source_indices)
        target_f = torch.index_select(target_features, 0, target_indices)
        weighting_gate = gate_module(target_f, source_f, rel_feat)

        message = message_gener(source_features, weighting_gate, rel_pair_idx, relness_score)

        return message

    def pairwise_rel_features(self, augment_obj_feat, rel_pair_idxs):
        pairwise_obj_feats_fused = self.pairwise_obj_feat_updim_fc(augment_obj_feat)
        pairwise_obj_feats_fused = pairwise_obj_feats_fused.view(pairwise_obj_feats_fused.size(0), 2, self.hidden_dim)
        head_rep = pairwise_obj_feats_fused[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = pairwise_obj_feats_fused[:, 1].contiguous().view(-1, self.hidden_dim)
        obj_pair_feat4rel_rep = torch.cat((head_rep[rel_pair_idxs[:, 0]], tail_rep[rel_pair_idxs[:, 1]]), dim=-1)
        obj_pair_feat4rel_rep = self.pairwise_rel_feat_finalize_fc(obj_pair_feat4rel_rep)  # (num_rel, hidden_dim)
        return obj_pair_feat4rel_rep

    def _update_rel_feats(self, curr_inst_feats, curr_rel_feats, batchwise_rel_pair_inds, valid_inst_idx):
        indices_sub = batchwise_rel_pair_inds[:, 0]
        indices_obj = batchwise_rel_pair_inds[:, 1]  # num_rel, 1

        downdim_inst_feats = self.rel_feat_update_inst_feat_downdim_fc(curr_inst_feats)

        downdim_inst_feats = downdim_inst_feats.view(downdim_inst_feats.shape[0], 2, self.hidden_dim)
        head_rep = downdim_inst_feats[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = downdim_inst_feats[:, 1].contiguous().view(-1, self.hidden_dim)

        if self.filter_the_mp_instance:
            # here we only pass massage from the fg boxes to the predicates
            valid_sub_inst_in_pairs = valid_inst_idx[indices_sub]
            valid_obj_inst_in_pairs = valid_inst_idx[indices_obj]
            valid_inst_pair_inds = squeeze_tensor(((valid_sub_inst_in_pairs) & (valid_obj_inst_in_pairs)).nonzero())
            # num_rel(vaild rel pair idx), 1

            # num_rel(vaild sub inst), 1
            indices_sub = indices_sub[valid_inst_pair_inds]
            # num_rel(vaild obj inst), 1
            indices_obj = indices_obj[valid_inst_pair_inds]

            # obj to pred on all pairs
            feat_sub2pred = torch.index_select(head_rep, 0, indices_sub)
            feat_obj2pred = torch.index_select(tail_rep, 0, indices_obj)

            # num_rel(vaild obj inst), hidden_dim
            vaild_rel_pairs_feats = torch.index_select(curr_rel_feats, 0, valid_inst_pair_inds)

            if vaild_rel_pairs_feats.shape[0] > 1:
                downdim_rel_feats = self.rel_feat_update_downdim_fc(vaild_rel_pairs_feats)
            else:
                downdim_rel_feats = self.rel_feat_update_downdim_fc[1:](vaild_rel_pairs_feats)

            sub2rel_feat, sub2pred_gate_weight = self.sub2pred_msp(downdim_rel_feats,
                                                                   feat_sub2pred)
            obj2rel_feat, obj2pred_gate_weight = self.obj2pred_msp(downdim_rel_feats,
                                                                   feat_obj2pred)

            # the gating machinsim has done in the msp model, we just sum them
            entit2rel_feat = (sub2rel_feat + obj2rel_feat) / 2.
            next_stp_rel_feature4iter = self.rel_feat_update_unit(entit2rel_feat,
                                                                  vaild_rel_pairs_feats)

            padded_next_stp_rel_feats = curr_rel_feats.clone()
            # residual add with the initial feature and incoming updated features
            padded_next_stp_rel_feats[valid_inst_pair_inds] += next_stp_rel_feature4iter

            return  padded_next_stp_rel_feats
        else:

            # obj to pred on all pairs
            feat_sub2pred = torch.index_select(head_rep, 0, indices_sub)
            feat_obj2pred = torch.index_select(tail_rep, 0, indices_obj)

            if curr_rel_feats.shape[0] > 1:
                downdim_rel_feats = self.rel_feat_update_downdim_fc(curr_rel_feats)
            else:
                downdim_rel_feats = self.rel_feat_update_downdim_fc[1:](curr_rel_feats)


            sub2rel_feat, sub2pred_gate_weight = self.sub2pred_msp(downdim_rel_feats, feat_sub2pred)
            obj2rel_feat, obj2pred_gate_weight = self.obj2pred_msp(downdim_rel_feats, feat_obj2pred)
            entit2rel_feat = (sub2rel_feat + obj2rel_feat) / 2.

            return  curr_rel_feats + self.rel_feat_update_unit(entit2rel_feat, curr_rel_feats)


    def forward(self, inst_features, rel_union_features, proposals, rel_pair_inds, relatedness=None):
        num_inst_proposals = [len(b) for b in proposals]
        # first, augment the entities and predicate features by pairwise results
        augment_obj_feat, rel_feats = self.pairwise_feature_extractor(inst_features, rel_union_features, proposals, rel_pair_inds)

        refine_rel_feats_each_iters = [rel_feats]
        refine_entit_feats_each_iters = [augment_obj_feat]

        rel_graph_iter_feat = []
        obj_graph_iter_feat = []

        for _ in range(1):
            valid_inst_idx = []
            curr_iter_relatedness = None
            # filter the instance
            valid_inst_idx = None
            if self.mode == "sgdet":
                score_thresh = 0.02
                while score_thresh >= -1e-06:  # a negative value very close to 0.0
                    valid_inst_idx_batch = []
                    size_require = []
                    for p in proposals:
                        valid_inst_idx = p.get_field('pred_scores') >= score_thresh
                        valid_inst_idx_batch.append(valid_inst_idx)
                        size_require.append(len(squeeze_tensor(torch.nonzero(valid_inst_idx))) > 3)

                    valid_inst_idx = torch.cat(valid_inst_idx_batch, 0)
                    if all(size_require) :
                        break

                    # print('Got {} rel_rois when score_thresh={}, changing to {}'.format(
                    #     valid_len, score_thresh, score_thresh - 0.01))
                    score_thresh -= 0.01


            # prepare the adjacency matrix
            batchwise_rel_pair_inds, subj2pred_inds, obj2pred_inds, \
            relness_batchcat, vaild_rel_pairs_idx = self._prepare_adjacency_matrix(num_inst_proposals,
                                                                                   valid_inst_idx, rel_pair_inds,
                                                                                   curr_iter_relatedness)

            # do msp
            msp_inst_feats_each_iters = [augment_obj_feat]
            msp_rel_feats_each_iters = [rel_feats]

            for t in range(self.update_step):
                # msp on entities features
                curr_inst_feats = msp_inst_feats_each_iters[-1]
                curr_rel_feats = msp_rel_feats_each_iters[-1]

                selected_relness = None
                if self.relness_weighting_mp:
                    assert relness_batchcat is not None
                    selected_relness = relness_batchcat[vaild_rel_pairs_idx]
                # key: instance feature(subj)
                # query: instance feature(obj) + rel union feature
                # value: instance feature(obj)
                message = self.prepare_message(curr_inst_feats, curr_inst_feats,
                                               curr_rel_feats[vaild_rel_pairs_idx],
                                               torch.index_select(batchwise_rel_pair_inds, 0, vaild_rel_pairs_idx),
                                               self.obj2obj_gating_model, self.obj2obj_msg_gen,
                                               selected_relness)

                assert message.shape[0] == curr_inst_feats.shape[0]

                fused_inst_feat = message + self.inst_feat_down_dim_fcs[t](curr_inst_feats)

                update_inst_feats = self.obj2obj_msg_fuse(fused_inst_feat)

                # if torch.isnan(update_inst_feats).any():
                #     ipdb.set_trace()

                msp_inst_feats_each_iters.append(update_inst_feats)

                # update predicate features from entities features
                # updated_rel_feats = self._update_rel_feats(curr_inst_feats=update_inst_feats,
                #                                            curr_rel_feats=curr_rel_feats,
                #                                            batchwise_rel_pair_inds=batchwise_rel_pair_inds,
                #                                            valid_inst_idx=valid_inst_idx)
                # msp_rel_feats_each_iters.append(updated_rel_feats)

            rel_graph_iter_feat.append(msp_rel_feats_each_iters)
            obj_graph_iter_feat.append(msp_inst_feats_each_iters)

            refine_entit_feats_each_iters.append(msp_inst_feats_each_iters[-1])

            # the instance feature has been fuse into relationship feature during the msp, we directly use
            # them here
            # refine_rel_feats_each_iters.append(msp_rel_feats_each_iters[-1])

            paired_inst_feats = self.pairwise_rel_features(msp_inst_feats_each_iters[-1], batchwise_rel_pair_inds)
            refine_rel_feats_each_iters.append(paired_inst_feats + msp_rel_feats_each_iters[-1])

        refined_inst_features = refine_entit_feats_each_iters[-1]

        refined_rel_features = refine_rel_feats_each_iters[-1]

        # decode object labels
        if self.obj_decode:
            refined_obj_logits = self.obj_classifier(refined_inst_features)
            boxes_per_cls = cat(
                [proposal.get_field("boxes_per_cls") for proposal in proposals], dim=0
            )
            refined_obj_pred_labels = obj_prediction_nms(
                boxes_per_cls, refined_obj_logits, nms_thresh=0.5
            )
            obj_labels = refined_obj_pred_labels
        else:
            refined_obj_logits = refined_inst_features
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            obj_labels = obj_labels.long()

        return refined_obj_logits, obj_labels, refined_rel_features, None