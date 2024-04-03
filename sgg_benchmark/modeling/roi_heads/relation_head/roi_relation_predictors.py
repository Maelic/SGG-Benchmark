# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
from sgg_benchmark.modeling import registry
from torch.nn import functional as F
import time

from .predictors.default_predictors import *
from .predictors.GCL_predictors import *
from .predictors.bayesian_predictors import *

def make_roi_relation_predictor_GCL(cfg, in_channels):
    result_str = '---'*20
    if cfg.GLOBAL_SETTING.USE_BIAS:
        result_str += ('\nwe use [ bias ]!')
    else:
        result_str += ('\nwe do [ not ] use bias!')
    result_str += ('\nthe model we use is [ %s ]' % cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR)
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == True and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == True:
        result_str += ('\ntraining mode is [ predcls ]')
    elif cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == False and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == True:
        result_str += ('\ntraining mode is [ sgcls ]')
    elif cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == False and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == False:
        result_str += ('\ntraining mode is [ sgdet ]')
    else:
        exit('wrong training mode!')
    result_str += ('\nlearning rate is [ %.5f ]' % cfg.SOLVER.BASE_LR)
    result_str += ('\nthe knowledge distillation strategy is [ %s ]' % cfg.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE)
    assert cfg.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE in ['None', 'KL_logit_Neighbor', 'KL_logit_None',
                                               'KL_logit_TopDown', 'KL_logit_BottomUp', 'KL_logit_BiDirection']
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR in ['TransLike_GCL', 'TransLikePredictor']:
        result_str += ('\nrel labels=0 is use [ %s ] to process' % cfg.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE)
        assert cfg.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE in ['rand_insert', 'rand_choose', 'all_include']
        assert cfg.GLOBAL_SETTING.BASIC_ENCODER in ['Self-Attention', 'Cross-Attention', 'Hybrid-Attention']
        result_str += ('\n-----Transformer layer is [ %d ] in obj and [ %d ] in rel' %
                       (cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER,
                        cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER))
        result_str += ('\n-----Transformer mode is [ %s ]' % cfg.GLOBAL_SETTING.BASIC_ENCODER)
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR in ['MotifsLike_GCL', 'MotifsLikePredictor']:
        assert cfg.GLOBAL_SETTING.BASIC_ENCODER in ['Motifs', 'VTransE']
        result_str += ('\n-----Model mode is [ %s ]' % cfg.GLOBAL_SETTING.BASIC_ENCODER)

    #num_of_group_element_list, predicate_stage_count = get_group_splits(cfg.DATASETS.NAME, cfg.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE)
    #max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
    #incre_idx_list, max_elemnt_list, group_matrix, kd_matrix = get_current_predicate_idx(
    #    num_of_group_element_list, cfg.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_PENALTY, cfg.DATASETS.NAME)
    #result_str += ('\n   the number of elements in each group is {}'.format(incre_idx_list))
    #result_str += ('\n   incremental stage list is {}'.format(num_of_group_element_list))
    #result_str += ('\n   the length of each line in group is {}'.format(predicate_stage_count))
    #result_str += ('\n   the max number of elements in each group is {}'.format(max_group_element_number_list))
    result_str += ('\n   the knowledge distillation strategy is [ %s ]' % cfg.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE)
    result_str += ('\n   the penalty for whole distillation loss is [ %.2f ]' % cfg.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT)
    with open(os.path.join(cfg.OUTPUT_DIR, 'control_info.txt'), 'w') as outfile:
        outfile.write(result_str)
    result_str += '\n\n'
    result_str += '---'*20
    print(result_str)
    time.sleep(2)
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)


def make_roi_relation_predictor(cfg, in_channels):
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR in ['TransLike_GCL', 'TransLikePredictor', 'MotifsLike_GCL', 'MotifsLikePredictor']:
        return make_roi_relation_predictor_GCL(cfg, in_channels)
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
