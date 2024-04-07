# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
import numpy as np
import math

from ...box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from ...box_head.roi_box_predictors import make_roi_box_predictor
from ...box_head.inference import make_roi_box_post_processor
from ...box_head.loss import make_roi_box_loss_evaluator
from ...box_head.sampling import make_roi_box_samp_processor

from ..simrel_head.model_simrel import *
from ..simrel_head.utils_simrel_old import *
from ..simrel_head.loss_simrel import SetCriterion
from ..simrel_head.loss_simrel import HungarianMatcher
from ..simrel_head.simrel_inference import *
from ..simrel_head.util import box_ops
from ..simrel_head.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from ..simrel_head.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


from sgg_benchmark.data import get_dataset_statistics
from sgg_benchmark.modeling.roi_heads.relation_head.model_motifs import FrequencyBias

def batch_box_iou_no_1pix(boxes1, boxes2):
    area1 = (boxes1[:, 3] - boxes1[:, 1] + 1e-3) * (boxes1[:, 2] - boxes1[:, 0] + 1e-3)
    area2 = (boxes2[:, 3] - boxes2[:, 1] + 1e-3) * (boxes2[:, 2] - boxes2[:, 0] + 1e-3)
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt) * (rb - lt >= 0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union

def data_duplicate_filter(sub_bbox, obj_bbox, sub_label, obj_label, rel_label, bbox, entity_label, iou_thres=0.85):
    s = np.concatenate([sub_bbox.reshape(-1, 4), sub_label.reshape(-1, 1)], -1)
    o = np.concatenate([obj_bbox.reshape(-1, 4), obj_label.reshape(-1, 1)], -1)
    t = np.concatenate([s, o, rel_label.reshape(-1, 1)], -1)
    obj5 = np.concatenate([bbox.reshape(-1, 4), entity_label.reshape(-1, 1)], -1)
    print(obj5)
    iou, _ = batch_box_iou_no_1pix(bbox.reshape(-1, 4), bbox.reshape(-1, 4))
    obj_overlap_mask = (iou >= iou_thres) & (entity_label.reshape(-1, 1) == entity_label.reshape(1, -1))
    N = np.arange(len(entity_label))
    obj_overlap_mask[N, N] = False
    obj_overlap1, obj_overlap2 = np.where(obj_overlap_mask)
    print(len(obj_overlap2), obj_overlap1, obj_overlap2)
    print()

def relax_topk_cost_kl_perbatch(
    output_bg_per_batch, pure_ent_logit, pure_ent_box, distri_score, \
    image_size_xyxy, X, Y, top_k=10, cls_cost=1., iou_cost=1., reg_cost=2.5, \
    random_num=3000, alpha=0.25, gamma=2.
):
    """
        tri_ent_logit: N, 2C
        pure_ent_logit: M, C
        
    """
    tri_ent_logit = output_bg_per_batch['pred_logits'].squeeze(0)
    tri_box = output_bg_per_batch['pred_boxes'].squeeze(0)
    
    
    num_queries = tri_ent_logit.shape[0]
    out_prob = tri_ent_logit.view(num_queries*2, -1).sigmoid() # N, C
    target_prob = pure_ent_logit.sigmoid() # M, C
    
    kl_x = out_prob
    
    log_kl_x = torch.log(kl_x + 1e-8)
    log_kl_x_s = log_kl_x[0::2] # N, C
    log_kl_x_o = log_kl_x[1::2] # N, C
    one_minus_log_kl_x = torch.log(1 - kl_x + 1e-8)
    one_minus_log_kl_x_s = one_minus_log_kl_x[0::2] # N, C
    one_minus_log_kl_x_o = one_minus_log_kl_x[1::2] # N, C
    

    p_log_kl_x_s = torch.mm(log_kl_x_s, target_prob.t()) # N, M
    p_log_kl_x_o = torch.mm(log_kl_x_o, target_prob.t())
    p_one_minus_log_kl_x_s = torch.mm(one_minus_log_kl_x_s, 1-target_prob.t())
    p_one_minus_log_kl_x_o = torch.mm(one_minus_log_kl_x_o, 1-target_prob.t())
    
    p_log_p = (target_prob * torch.log(target_prob)).sum(-1).view(1, -1)
    one_minus_p_log_p = ((1 - target_prob) * torch.log(1 - target_prob)).sum(-1).view(1, -1)
    
    cost_cls_s = (p_log_p - p_log_kl_x_s) + (one_minus_p_log_p - p_one_minus_log_kl_x_s) # N, M 
    cost_cls_o = (p_log_p - p_log_kl_x_o) + (one_minus_p_log_p - p_one_minus_log_kl_x_o)
    
    neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-torch.log(1 - out_prob + 1e-8))
    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-torch.log(out_prob + 1e-8)) # N, C
    

    p_distri_score = distri_score
    
    cost_dist_cls_s = torch.mm(pos_cost_class[0::2], p_distri_score.t()) - \
                        torch.mm(neg_cost_class[0::2], p_distri_score.t())
    cost_dist_cls_o = torch.mm(pos_cost_class[1::2], p_distri_score.t()) - \
                        torch.mm(neg_cost_class[1::2], p_distri_score.t())
    cost_cls_s =  cost_dist_cls_s
    cost_cls_o =  cost_dist_cls_o
    
    
    image_size_out = torch.from_numpy(image_size_xyxy).type_as(out_prob).to(out_prob.device).view(-1, 4)
    cost_bbox_s = torch.cdist(tri_box[:, :4] / image_size_out, pure_ent_box / image_size_out, p=1)
    cost_bbox_o = torch.cdist(tri_box[:, 4:] / image_size_out, pure_ent_box / image_size_out, p=1)
    
    cost_giou_s = -generalized_box_iou(tri_box[:, :4], pure_ent_box)
    cost_giou_o = -generalized_box_iou(tri_box[:, 4:], pure_ent_box)
    
    cost = cls_cost * (cost_cls_s[:, X] + cost_cls_o[:, Y]) + \
        iou_cost * (cost_giou_s[:, X] + cost_giou_o[:, Y]) + \
        reg_cost * (cost_bbox_s[:, X] + cost_bbox_o[:, Y]) # N, MM
    
    
    s_score, s_class = torch.topk(cost, k=top_k, largest=False, dim=-1) # 
    s_score, sorting_idx = torch.sort(s_score.view(-1), dim=0, descending=False)
    s_class = s_class.view(-1)
    s_class = s_class[sorting_idx]
    s_class = s_class.data.cpu().numpy()
    _, unique_indices = np.unique(s_class, return_index=True)
    unique_indices = np.sort(unique_indices)
    s_class = s_class[unique_indices]
    s_class = s_class[:random_num]
    remain_mask = np.ones(len(X), dtype=np.bool)
    remain_mask[s_class] = False

    return X[s_class], Y[s_class], X[remain_mask], Y[remain_mask], s_class, remain_mask

def select_bg_pair_in_one_layer(outputs, targets, indices):
    bs, num_queries = outputs["pred_boxes"].shape[:2]
    mask = [np.ones(num_queries, dtype=np.bool) for i in range(bs)]
    for idx, (i, j) in enumerate(indices):
        mask[idx][i] = False
    output_bg_per_batch_list = list()
    for idx, target in enumerate(targets):
        output_bg_per_batch = dict()
        for k, v in outputs.items():
            if k == 'aux_outputs': continue
            if k == 'tri_score': continue
            m = v[idx].view(1, num_queries, -1)
            output_bg_per_batch[k] = m[:, mask[idx]]
        output_bg_per_batch_list.append(output_bg_per_batch)
    return output_bg_per_batch_list


def make_pair_prediction(
    targets, class_logits, pred_bboxes, output_bg_per_batch_list,
    pure_ent_indices=None, random_num=3000, 
    ent_freq_weight=None, bg_freq_weight=None, 
    top_k=10, cls_cost=1., iou_cost=1., reg_cost=2.5,
    use_hard_label_klmatch=False, use_only_last_detection=True,
):
    hungarian_targets = make_pair_prediction_unit(
        targets, class_logits[-1], pred_bboxes[-1], output_bg_per_batch_list,
        pure_ent_indices=pure_ent_indices[0], random_num=random_num, 
        ent_freq_weight=ent_freq_weight, bg_freq_weight=bg_freq_weight, 
        top_k=top_k, cls_cost=cls_cost, iou_cost=iou_cost, reg_cost=reg_cost,
        use_hard_label_klmatch=use_hard_label_klmatch)
    
    if use_only_last_detection:
        return hungarian_targets
    else:
        aux_outputs = []
        for i in range(len(class_logits[:-1])):
            aux_hungarian_targets = make_pair_prediction_unit(
                targets, class_logits[i], pred_bboxes[i], output_bg_per_batch_list,
                pure_ent_indices=pure_ent_indices[1][i], random_num=random_num, 
                ent_freq_weight=ent_freq_weight, bg_freq_weight=bg_freq_weight, 
                top_k=top_k, cls_cost=cls_cost, iou_cost=iou_cost, reg_cost=reg_cost,
                use_hard_label_klmatch=use_hard_label_klmatch)
            aux_outputs.append(aux_hungarian_targets)
        
        for i in range(len(hungarian_targets)):
            temp_aux_outputs = []
            for j in range(len(aux_outputs)):
                temp_aux_outputs.append(aux_outputs[j][i])
            hungarian_targets[i]['aux_outputs'] = temp_aux_outputs
            
        return hungarian_targets

def make_pair_prediction_unit(
    targets, class_logits, pred_bboxes, output_bg_per_batch_list,
    pure_ent_indices=None, random_num=3000, 
    ent_freq_weight=None, bg_freq_weight=None, 
    top_k=10, cls_cost=1., iou_cost=1., reg_cost=2.5,
    use_hard_label_klmatch=False,
):
    hungarian_targets = list()
    with torch.no_grad():
        sum_ent_psedo_label = 0
        for i, target in enumerate(targets):
            ent_labels = target['entity_label']
            ent_boxes = torch.from_numpy(target['entity_bbox']).type_as(class_logits).to(class_logits.device)
            sub_id = target['sub_id']
            obj_id = target['obj_id']
            pred_bboxes_i = pred_bboxes[i].detach().view(-1, 4)
            class_logits_i = class_logits[i].detach().view(pred_bboxes_i.shape[0], -1)
            
            score_class_i = torch.sigmoid(class_logits_i)
            max_score_class_i, max_score_class_id = torch.max(score_class_i, dim=-1)
            
            if ent_freq_weight is not None:
                select_weight = bg_freq_weight * np.ones(pred_bboxes_i.shape[0], dtype=np.float)
            else:
                select_weight = None
                
            if pure_ent_indices is not None:
                prediction_ent_list, gt_ent_list = pure_ent_indices[i]
                if len(gt_ent_list) >= len(ent_labels):
                    pred_sub_id = prediction_ent_list[sub_id]
                    pred_obj_id = prediction_ent_list[obj_id]
                else:
                    filter_query_mask = -1 * np.ones(len(ent_labels), dtype=np.int)
                    filter_query_mask[gt_ent_list] = prediction_ent_list
                    pre_pred_sub_id = filter_query_mask[sub_id]
                    pre_pred_obj_id = filter_query_mask[obj_id]
                    and_subobj_id = (pre_pred_sub_id > -1) & (pre_pred_obj_id > -1)
                    pred_sub_id = pre_pred_sub_id[and_subobj_id]
                    pred_obj_id = pre_pred_obj_id[and_subobj_id]
                
                cls_logits_i_gt = torch.zeros_like(class_logits_i)
                cls_logits_i_gt[prediction_ent_list, ent_labels[gt_ent_list]] = 1.
                distri_score = cls_logits_i_gt
                substitute_class_logits_i = distri_score.clone().detach()
                
                
                contain_gt_pred_bboxes_i = -1. * torch.ones_like(pred_bboxes_i)
                contain_gt_pred_bboxes_i[prediction_ent_list, :] = ent_boxes[gt_ent_list]
                substitute_pred_bboxes_i = pred_bboxes_i.clone().detach()
                substitute_pred_bboxes_i[prediction_ent_list, :] = ent_boxes[gt_ent_list]
                
                count_ent_like_bg_num = np.zeros(pred_bboxes_i.shape[0])
                count_ent_like_bg_num[prediction_ent_list] = 1
                count_ent_like_bg_num = count_ent_like_bg_num.reshape(-1,1)
                
                if ent_freq_weight is not None:
                    select_weight[prediction_ent_list] = ent_freq_weight[ent_labels[gt_ent_list]]
                
                if use_hard_label_klmatch:
                    pred_bboxes_i = substitute_pred_bboxes_i
                    score_class_i = substitute_class_logits_i
                    max_score_class_i, max_score_class_id = torch.max(score_class_i, dim=-1)
                    
            else:
                distri_score = score_class_i
                contain_gt_pred_bboxes_i = pred_bboxes_i
                count_ent_like_bg_num = np.ones(pred_bboxes_i.shape[0])
                count_ent_like_bg_num = count_ent_like_bg_num.reshape(-1,1)

            
            
            max_score_class_id = max_score_class_id.data.cpu().numpy()
            
            
            image_size_xyxy = target['image_size_xyxy']
            
            cur_ent_num = pred_bboxes_i.shape[0]
            X, Y = np.meshgrid(np.arange(pred_bboxes_i.shape[0]), \
                    np.arange(pred_bboxes_i.shape[0]), indexing='ij')
            X, Y = X.reshape(-1), Y.reshape(-1)
            if pure_ent_indices is not None:
                bgmask = np.ones((pred_bboxes_i.shape[0], pred_bboxes_i.shape[0]), dtype=np.bool)
                bgmask[pred_sub_id, pred_obj_id] = False
                bgmask = bgmask.reshape(-1)
                X, Y = X[bgmask], Y[bgmask]
            
            rmv_self_conn_id = np.where(X!=Y)[0]
            X, Y = X[rmv_self_conn_id], Y[rmv_self_conn_id]
            

            pure_ent_pred_bboxes_i = pred_bboxes_i
            pred_bboxes_i = torch.cat((pred_bboxes_i[X], pred_bboxes_i[Y]), -1)
            contain_gt_pred_bboxes_i = torch.cat((contain_gt_pred_bboxes_i[X], contain_gt_pred_bboxes_i[Y]), -1)
            count_ent_like_bg_num = np.hstack((count_ent_like_bg_num[X], count_ent_like_bg_num[Y]))
            
            
            soft_bg_score = torch.min(1 - distri_score, torch.sigmoid(class_logits_i))
            soft_bg_score = torch.cat((soft_bg_score[X], soft_bg_score[Y]), -1)
            
            distri_score = torch.cat((distri_score[X], distri_score[Y]), -1)
            
            max_score_class_i_logits = torch.log(max_score_class_i + 1e-8)
            max_score_class_i_logits = max_score_class_i_logits.view(-1, 1)
            max_score_class_i_logits = \
                torch.cat((max_score_class_i_logits[X], max_score_class_i_logits[Y]), -1)

            max_score_class_i_logits_neg = torch.log(1 - max_score_class_i + 1e-8)
            max_score_class_i_logits_neg = max_score_class_i_logits_neg.view(-1, 1)
            max_score_class_i_logits_neg = \
                torch.cat((max_score_class_i_logits_neg[X], max_score_class_i_logits_neg[Y]), -1)

            max_score_class_i = max_score_class_i.view(-1, 1)
            max_score_class_i = torch.cat((max_score_class_i[X], max_score_class_i[Y]), -1)
            max_score_class_id = max_score_class_id.reshape(-1, 1)
            max_score_class_id = np.hstack((max_score_class_id[X], max_score_class_id[Y]))
            
            
            
            #score_class_i_logits = torch.log(score_class_i + 1e-8)
            #score_class_i_logits = torch.cat((score_class_i_logits[X], score_class_i_logits[Y]), -1)
            #score_class_i_logits_neg = torch.log(1 - score_class_i + 1e-8)
            #score_class_i_logits_neg = torch.cat((score_class_i_logits_neg[X], score_class_i_logits_neg[Y]), -1)
            #pair_score_class_i = torch.cat((score_class_i[X], score_class_i[Y]), -1)
                
            
            image_size_xyxy_tgt = np.tile(image_size_xyxy[None], (pred_bboxes_i.shape[0], 1))
            pure_ent_image_size_xyxy_tgt = np.tile(image_size_xyxy[None], (pure_ent_pred_bboxes_i.shape[0], 1))
            hungarian_target = \
                dict(labels=max_score_class_id,
                    distri_score=distri_score,
                    soft_bg_score=soft_bg_score,
                    score_logits=max_score_class_i_logits,
                    score_logits_neg=max_score_class_i_logits_neg,
                    score=max_score_class_i,
                    boxes_xyxy=pred_bboxes_i, 
                    distri_boxes_xyxy=contain_gt_pred_bboxes_i, 
                    image_size_xyxy=image_size_xyxy, 
                    image_size_xyxy_tgt=image_size_xyxy_tgt,
                    count_ent_like_bg_num=count_ent_like_bg_num,
                    gt_entity_num=len(ent_labels),
                    gt_rel_num=len(sub_id),
                    X=X, Y=Y,
                    pure_ent_pred_bboxes_i=pure_ent_pred_bboxes_i,
                    pure_ent_image_size_xyxy_tgt=pure_ent_image_size_xyxy_tgt,
                    pure_ent_score_class_i=score_class_i,
                    pure_ent_score_dist_label_i=substitute_class_logits_i,
                    ) #X=X+sum_ent_psedo_label, Y=Y+sum_ent_psedo_label,
            hungarian_targets.append(hungarian_target)
            sum_ent_psedo_label += cur_ent_num
    return hungarian_targets

def prob_freq(rel_mat, X, Y):
    out_deg = np.sum(rel_mat > 0, 1)
    in_deg = np.sum(rel_mat > 0, 0)
    tot_deg = in_deg + out_deg
    tot_deg_prob_mat = tot_deg.reshape(-1, 1) + tot_deg.reshape(1, -1)
    tot_deg_prob_mat = np.exp(-tot_deg_prob_mat)
    select_list = tot_deg_prob_mat[X, Y]
    select_list = select_list / select_list.sum()
    return select_list

def make_target(targets, cfg, num_proposals=300, use_focal=True, fg_triplet_rate=0.15):
    with torch.no_grad():
        hungarian_targets = list()
        complem_hungarian_targets = list()
        pure_ent_hungarian_targets = list()
        images_whwh = list()
        for i, target in enumerate(targets):
            #print(target.bbox)
            
            #bbox = target.bbox
            #bbox = bbox.data.cpu().numpy()
            bbox = target.bbox.data.cpu().numpy()
            
            w, h = target.size
            if use_focal:
                entity_label = target.get_field("labels").long().data.cpu().numpy() - 1
            else:
                entity_label = target.get_field("labels").long().data.cpu().numpy()
            if cfg.DEBUG.PURE_SPARSE_RCNN and \
              cfg.DEBUG.DUPLICATE_OBJ_BOXES:
                bbox = np.tile(bbox, (3, 1))
                entity_label = np.tile(entity_label, 3)
                image_size_xyxy = np.array([w, h, w, h], dtype=bbox.dtype)
            else:
                rel_matrix = target.get_field("relation").long().data.cpu().numpy()
                sub_id, obj_id = np.where(rel_matrix > 0)
                
                if use_focal:
                    rel_label = rel_matrix[sub_id, obj_id] - 1
                else:
                    rel_label = rel_matrix[sub_id, obj_id]
                sub_label = entity_label[sub_id]
                obj_label = entity_label[obj_id]
                sub_bbox = bbox[sub_id]
                obj_bbox = bbox[obj_id]
                
                #data_duplicate_filter(sub_bbox, obj_bbox, sub_label, obj_label, rel_label, bbox, entity_label)
                
                extra_entities_mask = \
                    (np.sum(rel_matrix, 0) < 1) & \
                    (np.sum(rel_matrix, 1) < 1)
                extra_bbox = bbox[extra_entities_mask]
                extra_entity_label = entity_label[extra_entities_mask]
                extra_entities_id = np.where(extra_entities_mask)[0]
                
                mask_rel_mat = (rel_matrix > 0).astype(np.int32) + np.eye(len(rel_matrix))
                complem_sub_id, complem_obj_id = np.where(mask_rel_mat <= 0)
                if len(complem_sub_id) > int(fg_triplet_rate * num_proposals) - len(sub_label):
                    p_list = None
                    #p_list = prob_freq(mask_rel_mat, complem_sub_id, complem_obj_id)
                    complem_id_id = np.random.choice(len(complem_sub_id), \
                        max(0, int(fg_triplet_rate * num_proposals) - len(sub_label)), \
                        replace=False, p=p_list)
                    complem_sub_id = complem_sub_id[complem_id_id]
                    complem_obj_id = complem_obj_id[complem_id_id]
                
                complem_sub_label = entity_label[complem_sub_id]
                complem_obj_label = entity_label[complem_obj_id]
                complem_label = np.stack([complem_sub_label, complem_obj_label]).T
                complem_sub_bbox = bbox[complem_sub_id]
                complem_obj_bbox = bbox[complem_obj_id]
                complem_box = np.hstack((complem_sub_bbox, complem_obj_bbox))
                
                sub_obj_box = np.hstack((sub_bbox, obj_bbox))
                sub_obj_label = np.stack([sub_label, obj_label]).T
                image_size_xyxy = np.array([w, h, w, h], dtype=bbox.dtype)
                image_size_xyxy_tgt = np.tile(image_size_xyxy[None], (len(sub_id), 1))
                extra_image_size_xyxy_tgt = np.tile(image_size_xyxy[None], (len(extra_bbox), 1))
                complem_image_size_xyxy_tgt = np.tile(image_size_xyxy[None], (len(complem_sub_id), 1))
            
            entity_image_size_xyxy_tgt_ent = np.tile(image_size_xyxy[None], (len(bbox), 1))
            
            complem_hungarian_target = \
                dict(image_size_xyxy=image_size_xyxy,  
                    image_size_xyxy_tgt=complem_image_size_xyxy_tgt,
                    boxes_xyxy=complem_box,
                    labels=complem_label,)
            
            hungarian_target = \
                dict(labels = sub_obj_label,
                    boxes_xyxy = sub_obj_box, 
                    image_size_xyxy = image_size_xyxy, 
                    image_size_xyxy_tgt = image_size_xyxy_tgt, 
                    rel_label = rel_label,
                    entity_image_size_xyxy_tgt_ent=entity_image_size_xyxy_tgt_ent,
                    entity_bbox=bbox,
                    entity_label=entity_label,
                    extra_bbox=extra_bbox,
                    extra_entity_label=extra_entity_label,
                    extra_image_size_xyxy_tgt=extra_image_size_xyxy_tgt,
                    sub_id=sub_id,
                    obj_id=obj_id,
                    extra_entities_id=extra_entities_id,
                    complem_label=complem_label,
                    complem_box=complem_box,
                    complem_image_size_xyxy_tgt=complem_image_size_xyxy_tgt,
                    )
            
            pure_ent_hungarian_target = \
                dict(image_size_xyxy=image_size_xyxy,  
                    image_size_xyxy_tgt=entity_image_size_xyxy_tgt_ent,
                    boxes_xyxy=bbox,
                    labels=entity_label,)
            
            hungarian_targets.append(hungarian_target)
            pure_ent_hungarian_targets.append(pure_ent_hungarian_target)
            complem_hungarian_targets.append(complem_hungarian_target)
            images_whwh.append(image_size_xyxy)
        
        images_whwh = np.stack(images_whwh)
        images_whwh = torch.from_numpy(images_whwh).type_as(targets[0].bbox).to(targets[0].bbox.device)
    return images_whwh, pure_ent_hungarian_targets, hungarian_targets, complem_hungarian_targets

def add_predict_logits(proposals, class_logits):
    slice_idxs = [0]
    for i in range(len(proposals)):
        slice_idxs.append(len(proposals[i])+slice_idxs[-1])
        proposals[i].add_field("predict_logits", class_logits[slice_idxs[i]:slice_idxs[i+1]])
    return proposals

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.cfg = cfg.clone()
        self.use_relation_fusion = cfg.MODEL.SimrelRCNN.USE_RELATION_FUSION_FOR_OBJECT
        self.use_post_hungarian_loss = cfg.MODEL.SimrelRCNN.USE_HUNGARIAN_LOSS
        self.use_pure_objdet = cfg.MODEL.SimrelRCNN.USE_PURE_OBJDET
        self.use_pure_sparsercnn_as_the_objdet_when_rel = \
            cfg.MODEL.SimrelRCNN.USE_SPARSERCNN_AS_THE_OBJDET_BASE_WHEN_REL
        if self.use_relation_fusion or self.use_post_hungarian_loss:
            self.dim_in = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
            self.dim_q = in_channels
            self.dimv = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
            self.dim_in_submap = in_channels
            self.num_obj_class = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
            self.num_proposals = cfg.MODEL.SimrelRCNN.NUM_PROPOSALS
            self.use_refine_obj = cfg.MODEL.SimrelRCNN.USE_REFINE_OBJ_FEATURE
            
            self.query_gradual_reduction = cfg.MODEL.SimrelRCNN.QUERY_GRADUAL_REDUCTION
            self.num_proposals_list = None
            if self.query_gradual_reduction is not None and self.use_refine_obj:
                num_proposals_list = list(self.query_gradual_reduction)
                self.num_proposals_list = num_proposals_list
                assert self.num_proposals == num_proposals_list[0]
            
            self.num_rel_class = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
            self.num_pre_cls_rel_layers = cfg.MODEL.SimrelRCNN.NUM_CLS_REL
            self.dim_in_rel = cfg.MODEL.SimrelRCNN.REL_DIM
            
            self.enable_rel_x2y = cfg.MODEL.SimrelRCNN.ENABLE_REL_X2Y
            self.pure_ent_num_proposals = cfg.MODEL.SimrelRCNN.PURE_ENT_NUM_PROPOSALS
            
            self.word_embed_weight_path = self.cfg.GLOVE_DIR
            self.posi_encode_dim = cfg.MODEL.SimrelRCNN.POSI_ENCODE_DIM
            self.posi_embed_dim = cfg.MODEL.SimrelRCNN.POSI_EMBED_DIM
            self.obj_word_embed_dim = cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
            statistics = get_dataset_statistics(cfg)
            obj_classes_list, rel_classes, att_classes = \
                statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
            if cfg.MODEL.SimrelRCNN.ENABLE_FREQ:
                freq_bias = FrequencyBias(cfg, statistics)
            else:
                freq_bias = None
            
            self.enable_mask_branch = cfg.MODEL.SimrelRCNN.ENABLE_MASK_BRANCH
            self.enable_query_reverse = cfg.MODEL.SimrelRCNN.ENABLE_QUERY_REVERSE
            
            self.freeze_pure_objdet = cfg.MODEL.SimrelRCNN.FREEZE_PUREE_OBJDET
            
            self.enable_fake_true_label = cfg.MODEL.SimrelRCNN.ENABLE_FAKE_TRUE_LABEL #to-do
            
            self.enable_kl_branch = cfg.MODEL.SimrelRCNN.ENABLE_KL_BRANCH
            self.mainbody_ent_det_only_fg = cfg.MODEL.SimrelRCNN.ENT_DET_ONLY_FG
            
            self.enable_auxiliary_branch = cfg.MODEL.SimrelRCNN.AUXILIARY_BRANCH
            self.new_arrival_ent_max_num = cfg.MODEL.SimrelRCNN.AUXILIARY_BRANCH_SELECT_ENT_MAX_NUM
            self.start_auxi_branch = cfg.MODEL.SimrelRCNN.AUXILIARY_BRANCH_START
            self.pair_group = cfg.MODEL.SimrelRCNN.PAIR_GROUP
            
            self.use_only_obj2rel = cfg.MODEL.SimrelRCNN.USE_ONLY_OBJ2REL
            
            self.effect_analysis = cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
            self.use_fusion_kq_selfatten = not cfg.MODEL.SimrelRCNN.DISABLE_KQ_FUSION_SELFATTEN
            
            self.diable_rel_fusion = cfg.MODEL.SimrelRCNN.DISABLE_REL_FUSION
            self.use_triplet_nms = cfg.MODEL.SimrelRCNN.USE_TRIPLET_NMS
            self.pairs_random_num_for_kl = cfg.MODEL.SimrelRCNN.PAIRS_RANDOM_NUM_FOR_KL
            self.use_hard_label_klmatch = cfg.MODEL.SimrelRCNN.USE_HARD_LABEL_KLMATCH
            
            self.use_last_relness = cfg.MODEL.SimrelRCNN.USE_LAST_RELNESS
            self.use_only_last_detection = cfg.MODEL.SimrelRCNN.USE_LAST_DET_FOR_KL_LABELASSIGN
            self.disable_obj2rel_loss = cfg.MODEL.SimrelRCNN.DISABLE_OBJ2REL_LOSS
            
            self.enable_batch_reduction = cfg.MODEL.SimrelRCNN.ENABLE_BATCH_REDUCTION
            self.num_batch_reduction = cfg.MODEL.SimrelRCNN.NUM_BATCH_REDUCTION
            self.enable_one_rel_conv = cfg.MODEL.SimrelRCNN.ONE_REL_CONV
            self.dim_ent_pre_cls = cfg.MODEL.SimrelRCNN.DIM_ENT_PRE_CLS
            self.dim_ent_pre_reg = cfg.MODEL.SimrelRCNN.DIM_ENT_PRE_REG
            if self.dim_ent_pre_cls is None: self.dim_ent_pre_cls = self.dim_in
            if self.dim_ent_pre_reg is None: self.dim_ent_pre_reg = self.dim_in
            
            
            dim_rank = self.dim_in
            
            obj_feature_fuser_ins = obj_feature_fuser(
                self.dim_in, self.dim_q, self.dimv, 
                self.dim_in_submap, self.num_obj_class, # dim_in_submap, num_class
                self.num_rel_class, self.num_pre_cls_rel_layers, 
                num_head=cfg.MODEL.SimrelRCNN.NHEADS, 
                dim_feedforward=cfg.MODEL.SimrelRCNN.DIM_FEEDFORWARD, 
                hidden_dim=cfg.MODEL.SimrelRCNN.DIM_DYNAMIC, 
                dropout=cfg.MODEL.SimrelRCNN.DROPOUT, 
                resolution=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION, 
                scales=cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES, 
                sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO, 
                cat_all_levels=False,
                scale_clamp = math.log(100000.0 / 16), 
                bbox_weights=(2.0, 2.0, 1.0, 1.0), 
                use_focal=cfg.MODEL.SimrelRCNN.USE_FOCAL, 
                prior_prob=cfg.MODEL.SimrelRCNN.PRIOR_PROB, 
                stack_num=cfg.MODEL.SimrelRCNN.NUM_HEADS, 
                rel_stack_num=cfg.MODEL.SimrelRCNN.REL_STACK_NUM, 
                return_intermediate=cfg.MODEL.SimrelRCNN.DEEP_SUPERVISION, 
                pooler_in_channels=512,
                dynamic_conv_num=cfg.MODEL.SimrelRCNN.NUM_DYNAMIC,
                num_pre_cls_layers=cfg.MODEL.SimrelRCNN.NUM_CLS,
                num_pre_reg_layers=cfg.MODEL.SimrelRCNN.NUM_REG,
                use_cross_obj_feat_fusion=cfg.MODEL.SimrelRCNN.CROSS_OBJ_FEAT_FUSION,
                prior_prob_rel=cfg.MODEL.SimrelRCNN.PRIOR_PROB_REL,
                use_siamese_head=cfg.MODEL.SimrelRCNN.SIAMESE_HEAD,
                posi_encode_dim=self.posi_encode_dim, 
                posi_embed_dim=self.posi_embed_dim, 
                obj_word_embed_dim=self.obj_word_embed_dim,
                obj_classes_list=obj_classes_list, 
                word_embed_weight_path=self.word_embed_weight_path,
                freq_bias=freq_bias, dim_in_rel=self.dim_in_rel, 
                enable_rel_x2y=self.enable_rel_x2y, num_rel_proposals=self.num_proposals,
                dim_rank=dim_rank,
                enable_mask_branch=self.enable_mask_branch,
                enable_query_reverse=self.enable_query_reverse,
                use_refine_obj=self.use_refine_obj,
                enable_auxiliary_branch=self.enable_auxiliary_branch,
                new_arrival_ent_max_num=self.new_arrival_ent_max_num,
                pair_group=self.pair_group,
                start_auxi_branch=self.start_auxi_branch,
                use_cross_rank=cfg.MODEL.SimrelRCNN.USE_CROSS_RANK,
                causal_effect_analsis=self.effect_analysis,
                use_fusion_kq_selfatten=self.use_fusion_kq_selfatten,
                use_only_obj2rel=self.use_only_obj2rel,
                diable_rel_fusion=self.diable_rel_fusion,
                dim_ent_pre_cls=self.dim_ent_pre_cls,
                dim_ent_pre_reg=self.dim_ent_pre_reg,
                enable_one_rel_conv=self.enable_one_rel_conv,
                num_batch_reduction=self.num_batch_reduction,
                enable_batch_reduction=self.enable_batch_reduction,
                use_pure_objdet=self.use_pure_objdet,
                use_last_relness=self.use_last_relness,
                num_proposals_list=self.num_proposals_list,)
            
            init_rel_features = nn.Embedding(self.num_proposals, self.dim_in_rel)
            
            if self.cfg.MODEL.RELATION_ON and (not self.use_pure_sparsercnn_as_the_objdet_when_rel):
                self.obj_feature_fuser = obj_feature_fuser_ins
                self.init_rel_features = init_rel_features
            else:
                self.obj_feature_fuser_pure_ent = obj_feature_fuser_ins
                self.init_rel_features_pure_ent = init_rel_features
            
            if self.enable_mask_branch:
                self.init_ent_mask_proposal_features = nn.Embedding(self.num_proposals, 2 * dim_rank)
            
            if self.use_refine_obj:
                self.init_so_proposal_features = nn.Embedding(self.num_proposals, 2 * self.dim_in)
                self.init_so_proposal_boxes = nn.Embedding(self.num_proposals, 2 * 4)
                nn.init.constant_(self.init_so_proposal_boxes.weight[:, 0:2], 0.5)
                nn.init.constant_(self.init_so_proposal_boxes.weight[:, 2:4], 1.0)
                nn.init.constant_(self.init_so_proposal_boxes.weight[:, 4:6], 0.5)
                nn.init.constant_(self.init_so_proposal_boxes.weight[:, 6:8], 1.0)
                
                self.num_proposals = cfg.MODEL.SimrelRCNN.PURE_ENT_NUM_PROPOSALS
            
            init_proposal_features = nn.Embedding(self.num_proposals, 2 * self.dim_in)
            init_proposal_boxes = nn.Embedding(self.num_proposals, 2 * 4)
            nn.init.constant_(init_proposal_boxes.weight[:, 0:2], 0.5)
            nn.init.constant_(init_proposal_boxes.weight[:, 2:4], 1.0)
            nn.init.constant_(init_proposal_boxes.weight[:, 4:6], 0.5)
            nn.init.constant_(init_proposal_boxes.weight[:, 6:8], 1.0)
            if self.cfg.MODEL.RELATION_ON and (not self.use_pure_sparsercnn_as_the_objdet_when_rel):
                self.init_proposal_features = init_proposal_features
                self.init_proposal_boxes = init_proposal_boxes
            else:
                self.init_proposal_features_pure_ent = init_proposal_features
                self.init_proposal_boxes_pure_ent = init_proposal_boxes
            
            if self.enable_auxiliary_branch:
                self.init_auxiliary_rel_feat = nn.Embedding(self.num_proposals, 2 * self.dim_in_rel)
                #self.init_auxiliary_rel_feat = None
            
            
            if cfg.MODEL.SimrelRCNN.FREEZE_LATENT_VECTORS:
                self.init_rel_features.weight.requires_grad = False
                self.init_proposal_features.weight.requires_grad = False
                if self.enable_mask_branch:
                    self.init_ent_mask_proposal_features.requires_grad = False
                if self.enable_auxiliary_branch:
                    if self.init_auxiliary_rel_feat is not None:
                        self.init_auxiliary_rel_feat.requires_grad = False
                if self.use_refine_obj:
                    self.init_so_proposal_features.weight.requires_grad = False
            
            if cfg.MODEL.SimrelRCNN.FREEZE_LATENT_BOXES:
                if self.use_refine_obj:
                    self.init_so_proposal_boxes.weight.requires_grad = False
                self.init_proposal_boxes.weight.requires_grad = False
            
            
            if self.use_refine_obj:
                self.num_proposals = cfg.MODEL.SimrelRCNN.NUM_PROPOSALS
                
            # Loss parameters:
            self.use_equ_loss = cfg.MODEL.SimrelRCNN.USE_EQU_LOSS
            
            rel_weight = cfg.MODEL.SimrelRCNN.REL_CLASS_WEIGHT
            tri_w = cfg.MODEL.SimrelRCNN.TRIPLET_MASK_WEIGHT
            
            class_weight = cfg.MODEL.SimrelRCNN.CLASS_WEIGHT
            giou_weight = cfg.MODEL.SimrelRCNN.GIOU_WEIGHT
            l1_weight = cfg.MODEL.SimrelRCNN.L1_WEIGHT
            no_object_weight = cfg.MODEL.SimrelRCNN.NO_OBJECT_WEIGHT
            self.deep_supervision = cfg.MODEL.SimrelRCNN.DEEP_SUPERVISION
            self.use_focal = cfg.MODEL.SimrelRCNN.USE_FOCAL
            alpha = cfg.MODEL.SimrelRCNN.ALPHA
            gamma = cfg.MODEL.SimrelRCNN.GAMMA
            # Build Criterion.
            matcher = HungarianMatcher(alpha=alpha, 
                                    gamma=gamma, 
                                    cost_class=class_weight, 
                                    cost_bbox=l1_weight, 
                                    cost_giou=giou_weight, 
                                    use_focal=self.use_focal,
                                    rel_w=rel_weight,
                                    tri_w=tri_w)
            self.matcher = matcher
            weight_dict = {"loss_ce": class_weight, 
                            "loss_bbox": l1_weight, 
                            "loss_giou": giou_weight, 
                            "loss_rel": rel_weight,
                            "loss_rel_freq_pro": rel_weight,
                            "loss_rel_obj2rel": rel_weight,
                            "loss_tri_score": tri_w,  ## must keep in line with 'out_loss_name'
                            }
            if self.deep_supervision:
                aux_weight_dict = {}
                for i in range(cfg.MODEL.SimrelRCNN.NUM_HEADS - 1):
                    aux_weight_dict.update({
                        k + f"_{i}": v for k, v in weight_dict.items()
                    })
                weight_dict.update(aux_weight_dict)

            losses = ["labels", "boxes"]
            if (not cfg.DEBUG.PURE_SPARSE_RCNN) and (not self.use_pure_objdet):
                losses.append("rel_pred_logits")
                if self.enable_rel_x2y and (not self.use_only_obj2rel) and (not self.disable_obj2rel_loss):
                    losses.append("obj2rel_logits")
                if self.enable_mask_branch:
                    losses.append("tri_score")
            
            self.enable_bg_obj=cfg.MODEL.SimrelRCNN.ENABLE_BG_OBJ
            
            vg_rel_freq = cfg.MODEL.ROI_RELATION_HEAD.REL_PROP
            vg_rel_freq = torch.tensor(vg_rel_freq, dtype=torch.float)
            #vg_rel_freq = vg_rel_freq.log() - (1 - vg_rel_freq).log()
            vg_rel_freq = vg_rel_freq.log()
            self.register_buffer("vg_rel_freq", vg_rel_freq)
            self.register_buffer("average_rel_logits", torch.zeros(self.num_rel_class-1))
            self.register_buffer("var_rel_logits", torch.zeros(self.num_rel_class-1))
            self.vg_logits_adjustment = cfg.MODEL.SimrelRCNN.REL_LOGITS_ADJUSTMENT
            self.logit_adj_tau = cfg.MODEL.SimrelRCNN.LOGIT_ADJ_TAU
            self.average_ratio = 0.0005
            
            c_ent_freq = cfg.MODEL.SimrelRCNN.VG_ENT_PROP
            if self.num_obj_class == 602:
                c_ent_freq = cfg.MODEL.SimrelRCNN.OI_REL_06_ENT_PROP
            if self.num_obj_class == 58:
                c_ent_freq = cfg.MODEL.SimrelRCNN.OI_REL_04_ENT_PROP
            if self.num_obj_class == 101:
                c_ent_freq = cfg.MODEL.SimrelRCNN.VRD_ENT_PROP
            
            self.ent_freq_mu = cfg.MODEL.SimrelRCNN.ENT_FREQ_MU
            self.ent_freq = None
            if cfg.MODEL.SimrelRCNN.ENABLE_ENT_PROP:
                self.ent_freq = c_ent_freq

            self.ent_freq_for_smaple = np.array(c_ent_freq, dtype=np.float)
            self.ent_freq_mean_for_smaple = np.mean(self.ent_freq_for_smaple)
            self.ent_freq_for_smaple = np.exp(-self.ent_freq_for_smaple)
            self.ent_freq_mean_for_smaple = np.exp(-self.ent_freq_mean_for_smaple)
            
            self.criterion = SetCriterion(num_classes=self.num_obj_class,
                                          matcher=matcher,
                                          weight_dict=weight_dict,
                                          eos_coef=no_object_weight,
                                          losses=losses,
                                          use_focal=self.use_focal,
                                          num_rel_classes=self.num_rel_class,
                                          alpha=alpha, 
                                          gamma=gamma,
                                          debug_rmv_bg=cfg.DEBUG.REMOVE_BG_SAMPLE,
                                          use_equ_loss=self.use_equ_loss,
                                          enable_query_reverse=self.enable_query_reverse,
                                          ent_freq=self.ent_freq, ent_freq_mu=self.ent_freq_mu,
                                          use_last_relness=self.use_last_relness)
            if self.use_refine_obj:
                losses_pure_ent = ["labels", "boxes"] 
                weight_dict_pure_ent = \
                    {"loss_ce": cfg.MODEL.SimrelRCNN.PURE_ENT_CLASS_WEIGHT, 
                        "loss_bbox": cfg.MODEL.SimrelRCNN.PURE_ENT_L1_WEIGHT, 
                        "loss_giou": cfg.MODEL.SimrelRCNN.PURE_ENT_GIOU_WEIGHT}
                if self.deep_supervision:
                    pure_ent_aux_weight_dict = {}
                    for i in range(cfg.MODEL.SimrelRCNN.NUM_HEADS - 1):
                        pure_ent_aux_weight_dict.update({
                            k + f"_{i}": v for k, v in weight_dict_pure_ent.items()
                        })
                    weight_dict_pure_ent.update(pure_ent_aux_weight_dict)
                pure_ent_matcher = HungarianMatcher(alpha=alpha, gamma=gamma, 
                                    cost_class=cfg.MODEL.SimrelRCNN.PURE_ENT_CLASS_WEIGHT, 
                                    cost_bbox=cfg.MODEL.SimrelRCNN.PURE_ENT_L1_WEIGHT, 
                                    cost_giou=cfg.MODEL.SimrelRCNN.PURE_ENT_GIOU_WEIGHT, 
                                    use_focal=self.use_focal, rel_w=None, tri_w=None)
                self.pure_ent_criterion = SetCriterion(num_classes=self.num_obj_class,
                                          matcher=pure_ent_matcher,
                                          weight_dict=weight_dict_pure_ent,
                                          eos_coef=no_object_weight,
                                          losses=losses_pure_ent,
                                          use_focal=self.use_focal,
                                          num_rel_classes=self.num_rel_class,
                                          alpha=alpha, gamma=gamma,)
                
                if self.enable_kl_branch:
                    losses_kl = ["labels", "boxes"]
                    kl_branch_weight = cfg.MODEL.SimrelRCNN.KL_BRANCH_WEIGHT
                    
                    weight_dict_kl = \
                        {"loss_ce": weight_dict["loss_ce"] * kl_branch_weight, 
                        "loss_bbox": weight_dict["loss_bbox"] * kl_branch_weight, 
                        "loss_giou": weight_dict["loss_giou"] * kl_branch_weight}
                    
                    if self.deep_supervision:
                        kl_aux_weight_dict = {}
                        for i in range(cfg.MODEL.SimrelRCNN.NUM_HEADS - 1):
                            kl_aux_weight_dict.update({
                                k + f"_{i}": v for k, v in weight_dict_kl.items()
                            })
                        weight_dict_kl.update(kl_aux_weight_dict)
                    
                    kl_matcher = HungarianMatcher(alpha=alpha, gamma=gamma, 
                                    cost_class=weight_dict["loss_ce"] * kl_branch_weight, 
                                    cost_bbox=weight_dict["loss_bbox"] * kl_branch_weight, 
                                    cost_giou=weight_dict["loss_giou"] * kl_branch_weight, 
                                    use_focal=self.use_focal, rel_w=None, tri_w=tri_w)
                    
                    self.kl_criterion = SetCriterion(
                                num_classes=self.num_obj_class,
                                matcher=kl_matcher,
                                weight_dict=weight_dict_kl,
                                eos_coef=no_object_weight,
                                losses=losses_kl,
                                use_focal=self.use_focal,
                                num_rel_classes=self.num_rel_class,
                                alpha=alpha, gamma=gamma,
                                enable_kl_div=True,
                                ent_freq=self.ent_freq, 
                                ent_freq_mu=self.ent_freq_mu)

                        
            self._reset_parameters()
        else:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, 
                in_channels, half_out=self.cfg.MODEL.ATTRIBUTE_ON)
            self.predictor = make_roi_box_predictor(
                cfg, self.feature_extractor.out_channels)
                
            self.post_processor = make_roi_box_post_processor(cfg)
            self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
            self.samp_processor = make_roi_box_samp_processor(cfg)
    
    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder
    
    def _reset_parameters(self):
        # init all parameters.
        for n, p in self.named_parameters():
            if self.freeze_pure_objdet:
                if n.find('_so') < 0 and n.find('_mask') < 0 and n.find('_rel') < 0:
                    if n.find('self_atten') >= 0 or n.find('dynamic_conv') >= 0 \
                      or n.find('FFN') >= 0 or n.find('layer_norm') >= 0 \
                      or n.find('reg') >= 0 or n.find('reg_deltas') >= 0 \
                      or n.find('cls') >= 0 or n.find('cls_logits') >= 0:
                        p.requires_grad = False
                if n.find('init_proposal_boxes') >= 0 \
                  or n.find('init_proposal_features') >= 0:
                    p.requires_grad = False
    
    def forward(self, features, proposals, targets=None, ori_images=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        ###################################################################
        # box head specifically for relation prediction model
        ###################################################################
        if self.cfg.MODEL.RELATION_ON and (not self.use_pure_sparsercnn_as_the_objdet_when_rel):
            if self.use_post_hungarian_loss:
                if targets is not None:
                    images_whwh, pure_ent_hungarian_targets, \
                    hungarian_targets, complem_hungarian_targets = \
                        make_target(targets, self.cfg, 
                            num_proposals=self.num_proposals, 
                            use_focal=self.use_focal)
                    
                    if self.cfg.DEBUG.PURE_SPARSE_RCNN or self.use_pure_objdet:
                        hungarian_targets = pure_ent_hungarian_targets
                    
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                # use ground truth box as proposals
                proposals = [target.copy_with_fields(["labels", "attributes"]) for target in targets]
                
                if self.use_relation_fusion or self.use_post_hungarian_loss:
                    ###!!! to-do
                    proposal_boxes = self.init_proposal_boxes.weight.clone()
                    nr_boxes = proposal_boxes.shape[0]
                    proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes.view(-1, 4))
                    proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]
                    proposal_boxes = proposal_boxes.view(images_whwh.shape[0], nr_boxes, -1)
                    class_logits, pred_bboxes, rel_pred_logits = self.obj_feature_fuser(features, 
                        proposal_boxes, self.init_proposal_features.weight, self.init_rel_features.weight, 
                        images_whwh[:,:2])
                    ###!!! to-do
                    return x, proposal_boxes, {} ###!!! to-do
                else:
                    x = self.feature_extractor(features, proposals)
                    if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                        # mode==predcls
                        # return gt proposals and no loss even during training
                        return x, proposals, {}
                    else:
                        # mode==sgcls
                        # add field:class_logits into gt proposals, note field:labels is still gt
                        class_logits, _ = self.predictor(x)
                        proposals = add_predict_logits(proposals, class_logits)
                        return x, proposals, {}
            else:
                # mode==sgdet
                if not self.use_post_hungarian_loss:
                    if self.training or not self.cfg.TEST.CUSTUM_EVAL:
                        # in sparse RCNN, maybe removable!
                        proposals = self.samp_processor.assign_label_to_proposals(proposals, targets) 
                
                if self.use_relation_fusion:
                    proposal_boxes = self.init_proposal_boxes.weight.clone()
                    nr_boxes = proposal_boxes.shape[0]
                    proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes.view(-1, 4))
                    proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]
                    proposal_boxes = proposal_boxes.view(images_whwh.shape[0], nr_boxes, -1)
                    
                    init_so_pro_features = None
                    so_proposal_boxes = None
                    if self.use_refine_obj:
                        so_proposal_boxes = self.init_so_proposal_boxes.weight.clone()
                        nr_boxes = so_proposal_boxes.shape[0]
                        so_proposal_boxes = box_cxcywh_to_xyxy(so_proposal_boxes.view(-1, 4))
                        so_proposal_boxes = so_proposal_boxes[None] * images_whwh[:, None, :]
                        so_proposal_boxes = so_proposal_boxes.view(images_whwh.shape[0], nr_boxes, -1)
                        init_so_pro_features = self.init_so_proposal_features.weight
                    

                    if self.enable_auxiliary_branch:
                        init_auxiliary_rel_feat = None
                        if self.init_auxiliary_rel_feat is not None:
                            init_auxiliary_rel_feat = self.init_auxiliary_rel_feat.weight
                        
                    
                    init_ent_mask_features = None
                    if self.enable_mask_branch:
                        init_ent_mask_features=self.init_ent_mask_proposal_features.weight

                    #print('avg')
                    #for i in self.average_rel_logits: print(i.item())
                    #print('var')
                    #for i in self.var_rel_logits: print(i.item())
                    #assert False
                    
                    class_logits, pred_bboxes, rel_pred_logits, obj2rel_logits, \
                    rel_prod_freq, mask_class_logits, \
                    so_class_logits, pred_bboxes_so, \
                    counterfact_rel_logits = \
                        self.obj_feature_fuser(features, proposal_boxes, 
                            self.init_proposal_features.weight, 
                            self.init_rel_features.weight, 
                            images_whwh[:,:2],
                            init_ent_mask_features=init_ent_mask_features,
                            only_ent=self.cfg.DEBUG.PURE_SPARSE_RCNN,
                            init_so_pro_features=init_so_pro_features,
                            init_so_bboxes=so_proposal_boxes,
                            auxiliary_rel=init_auxiliary_rel_feat,
                            ori_images=ori_images)
                    
                    if self.effect_analysis and not self.training:
                        _, _, counterfact_rel_logits, _, _, _, \
                        _, _, _ = \
                            self.obj_feature_fuser(features, proposal_boxes, 
                                self.init_proposal_features.weight, 
                                self.init_rel_features.weight, 
                                images_whwh[:,:2],
                                init_ent_mask_features=init_ent_mask_features,
                                only_ent=self.cfg.DEBUG.PURE_SPARSE_RCNN,
                                init_so_pro_features=init_so_pro_features,
                                init_so_bboxes=so_proposal_boxes,
                                auxiliary_rel=init_auxiliary_rel_feat,
                                causal_conducting=True)
                        counterfact_rel_logits = counterfact_rel_logits[-1]
                    elif self.vg_logits_adjustment and not self.training:
                        counterfact_rel_logits = self.logit_adj_tau * self.vg_rel_freq
                    
                    if self.training:
                        cur_rel_logits = rel_pred_logits[-1].view(-1, rel_pred_logits[-1].shape[-1])
                        if self.use_last_relness:
                            cur_rel_logits = cur_rel_logits[:, :-1]
                        tmp = self.moving_average(self.average_rel_logits, cur_rel_logits)
                        self.average_rel_logits = tmp
                        with torch.no_grad():
                            self.var_rel_logits = self.var_rel_logits * (1 - self.average_ratio) + \
                                self.average_ratio * cur_rel_logits.var(0).view(-1)
                        

                    if self.cfg.DEBUG.PURE_SPARSE_RCNN:
                        layer_num, N, nr_boxes, num_obj = class_logits.shape
                        class_logits = class_logits.view(layer_num, N, 2 * nr_boxes, -1)
                        pred_bboxes = pred_bboxes.view(layer_num, N, 2 * nr_boxes, -1)
                        
                        output = {'pred_logits': class_logits[-1], 
                                'pred_boxes': pred_bboxes[-1]}
                    else:
                        if self.use_refine_obj:
                            pure_ent_output = {'pred_logits': class_logits[-1], 
                                'pred_boxes': pred_bboxes[-1]}

                            pred_logits = so_class_logits[-1]
                            pred_boxes = pred_bboxes_so[-1]
                        else:
                            pred_logits = class_logits[-1]
                            pred_boxes = pred_bboxes[-1]
                        
                        output = {'pred_logits': pred_logits, 
                                'rel_pred_logits': rel_pred_logits[-1], 
                                'rel_prod_freq': rel_prod_freq[-1], 
                                'obj2rel_logits': obj2rel_logits[-1], 
                                'pred_boxes': pred_boxes, 
                             }
                        if self.enable_mask_branch:
                            output['tri_score'] = mask_class_logits[-1]
                        
                        
                    if self.deep_supervision:
                        if self.cfg.DEBUG.PURE_SPARSE_RCNN:
                            output['aux_outputs'] = \
                                [{'pred_logits': a, 'pred_boxes': b} \
                                    for a, b in zip(class_logits[:-1], \
                                                pred_bboxes[:-1])]
                        else:
                            if self.use_refine_obj:
                                pure_ent_pred_logits = class_logits[:-1]
                                pure_ent_pred_boxes = pred_bboxes[:-1]
                                pred_boxes = pred_bboxes_so[:-1]
                                pred_logits = so_class_logits[:-1]
                                
                                pure_ent_output['aux_outputs'] = \
                                [{'pred_logits': a, 'pred_boxes': b} \
                                    for a, b in zip(pure_ent_pred_logits, \
                                                pure_ent_pred_boxes)]

                            else:
                                pred_boxes = pred_bboxes[:-1]
                                pred_logits = class_logits[:-1]
                        
                            if self.enable_mask_branch:
                                output['aux_outputs'] = \
                                [{'pred_logits': a, 'pred_boxes': b, \
                                'rel_pred_logits': c, 'obj2rel_logits': d, \
                                'rel_prod_freq': e, 'tri_score': f} \
                                    for a, b, c, d, e, f in zip(pred_logits, \
                                                pred_boxes, \
                                                rel_pred_logits[:-1], \
                                                obj2rel_logits[:-1], \
                                                rel_prod_freq[:-1], \
                                                mask_class_logits[:-1])]
                            else:
                                output['aux_outputs'] = \
                                [{'pred_logits': a, 'pred_boxes': b, \
                                'rel_pred_logits': c, 'obj2rel_logits': d, \
                                'rel_prod_freq': e} \
                                    for a, b, c, d, e in zip(pred_logits, \
                                                pred_boxes, \
                                                rel_pred_logits[:-1], \
                                                obj2rel_logits[:-1], \
                                                rel_prod_freq[:-1])]
                             
                else:
                    x = self.feature_extractor(features, proposals)
                    class_logits, box_regression = self.predictor(x)
                    proposals = add_predict_logits(proposals, class_logits)
                
                if self.use_relation_fusion or self.use_post_hungarian_loss:
                    if self.training and targets is not None:
                        if self.cfg.DEBUG.PURE_SPARSE_RCNN:
                            ent_det_only_fg = False
                        else: 
                            ent_det_only_fg = self.mainbody_ent_det_only_fg
                            #ent_det_only_fg = True
                        loss_dict, indices_list = \
                            self.criterion(output, hungarian_targets, \
                                self.enable_bg_obj, ent_det_only_fg=ent_det_only_fg)
                        weight_dict = self.criterion.weight_dict
                        for k in loss_dict.keys():
                            if k in weight_dict:
                                loss_dict[k] *= weight_dict[k]
                        
                        if self.use_refine_obj:
                            pure_ent_loss_dict, pure_ent_indices = \
                                self.pure_ent_criterion(pure_ent_output, \
                                    pure_ent_hungarian_targets, False, ent_det_only_fg=False)
                            
                            
                            pure_ent_weight_dict = self.pure_ent_criterion.weight_dict
                            for k in pure_ent_loss_dict.keys():
                                if k in pure_ent_weight_dict:
                                    pure_ent_loss_dict[k] *= pure_ent_weight_dict[k]
                                    loss_dict['pure_ent_'+k] = pure_ent_loss_dict[k]
                                    
                                    
                            if self.enable_kl_branch:
                                kl_weight_dict = self.kl_criterion.weight_dict
                                outputs_bg_pair_list = select_bg_pair_in_one_layer(output, hungarian_targets, indices_list[0])
                                if self.deep_supervision:
                                    d = list()
                                    for idx, aux_dict in enumerate(output['aux_outputs']):
                                        dd = select_bg_pair_in_one_layer(aux_dict, hungarian_targets, indices_list[1][idx])
                                        d.append(dd)
                                
                                    for idx in range(len(outputs_bg_pair_list)):
                                        outputs_bg_pair_list[idx]['aux_outputs'] = list()
                                        for layer_id, dd in enumerate(d):
                                            outputs_bg_pair_list[idx]['aux_outputs'].append(dd[idx])
                                
                                pair_prediction_targets = \
                                    make_pair_prediction(hungarian_targets, 
                                        class_logits, pred_bboxes, 
                                        outputs_bg_pair_list,
                                        pure_ent_indices=pure_ent_indices,
                                        random_num=self.num_proposals,
                                        ent_freq_weight=self.ent_freq_for_smaple, 
                                        bg_freq_weight=self.ent_freq_mean_for_smaple,
                                        top_k=self.num_proposals, 
                                        cls_cost=kl_weight_dict['loss_ce'], 
                                        iou_cost=kl_weight_dict['loss_giou'], 
                                        reg_cost=kl_weight_dict['loss_bbox'],
                                        use_hard_label_klmatch=self.use_hard_label_klmatch,
                                        use_only_last_detection=self.use_only_last_detection)
                                #random_num=self.pairs_random_num_for_kl,
                                #top_k=(self.pairs_random_num_for_kl//self.num_proposals),
                                #top_k=self.pure_ent_num_proposals, 
                                
                                for idx, output_i in enumerate(outputs_bg_pair_list):
                                    kl_loss_dict, _ = self.kl_criterion(output_i, \
                                        [pair_prediction_targets[idx]], False, ent_det_only_fg=False)
                                    for k in kl_loss_dict.keys():
                                        if k in kl_weight_dict:
                                            kl_loss_dict[k] *= kl_weight_dict[k]
                                            loss_dict['kl_'+k] = kl_loss_dict[k] / (1. * len(pair_prediction_targets))
                                
                        return None, None, loss_dict
                    else:
                        if self.use_refine_obj:
                            pred_bboxes = pred_bboxes_so[-1]
                            class_logits = so_class_logits[-1]
                        else:
                            pred_bboxes = pred_bboxes[-1]
                            class_logits = class_logits[-1]
                        rel_pred_logits = rel_pred_logits[-1]
                        if self.enable_mask_branch:
                            tri_score = F.sigmoid(mask_class_logits[-1])
                        else:
                            tri_score=None
                        result = inference_pair(
                            pred_bboxes, class_logits, 
                            rel_pred_logits, images_whwh[:,:2], 
                            tri_score=tri_score, 
                            counterfact_rel_logits=counterfact_rel_logits, 
                            use_triplet_nms=self.use_triplet_nms, 
                            use_last_relness=self.use_last_relness)
                        return None, result, {}
                else:
                    # post process:
                    # filter proposals using nms, keep original bbox, 
                    # add a field 'boxes_per_cls' of size (#nms, #cls, 4)
                    x, result = self.post_processor((x, class_logits, box_regression), 
                                                        proposals, relation_mode=True)
                    # note x is not matched with processed_proposals, so sharing x is not permitted
                    return x, result, {}

        #####################################################################
        # Original box head (relation_on = False)
        #####################################################################
        if self.use_post_hungarian_loss:
            hungarian_targets = None
            if targets is not None:
                images_whwh, pure_ent_hungarian_targets, \
                hungarian_targets, complem_hungarian_targets = \
                    make_target(targets, self.cfg, 
                        num_proposals=self.num_proposals, 
                        use_focal=self.use_focal)
                
                if self.cfg.DEBUG.PURE_SPARSE_RCNN or self.use_pure_objdet:
                    hungarian_targets = pure_ent_hungarian_targets
            
            
            proposal_boxes = self.init_proposal_boxes_pure_ent.weight.clone()
            nr_boxes = proposal_boxes.shape[0]
            proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes.view(-1, 4))
            proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]
            proposal_boxes = proposal_boxes.view(images_whwh.shape[0], nr_boxes, -1)
            
            init_so_pro_features = None
            so_proposal_boxes = None
            init_ent_mask_features = None
            
            class_logits, pred_bboxes, proposal_features, roi_features = \
                self.obj_feature_fuser_pure_ent(features, proposal_boxes, 
                    self.init_proposal_features_pure_ent.weight, 
                    self.init_rel_features_pure_ent.weight, 
                    images_whwh[:,:2],
                    init_ent_mask_features=init_ent_mask_features,
                    only_ent=True,
                    init_so_pro_features=init_so_pro_features,
                    init_so_bboxes=so_proposal_boxes,
                    auxiliary_rel=None,
                    ori_images=ori_images)
            
            output = {'pred_logits': class_logits[-1], 
                    'pred_boxes': pred_bboxes[-1],}
            output['aux_outputs'] = \
                [{'pred_logits': a, 'pred_boxes': b} \
                    for a, b in zip(class_logits[:-1], \
                                pred_bboxes[:-1])]
            img_size = images_whwh[:,:2]
            tuple_img_size = [tuple(img_size[b].data.cpu().numpy()) for b in range(img_size.shape[0])]
            proposals = list()
            N = class_logits[-1].shape[0]
            for b in range(N):
                proposals.append(BoxList(pred_bboxes[-1][b].view(-1, 4), tuple_img_size[b]))
            
            all_hit_targets_list = None
            if (self.training and targets is not None) or self.use_pure_sparsercnn_as_the_objdet_when_rel:
                loss_dict, indices_list = \
                    self.criterion(output, hungarian_targets, \
                        self.enable_bg_obj, ent_det_only_fg=False)
                all_hit_targets_list, aux_indicess_list = indices_list
                        
            if (self.training and targets is not None) and (not self.use_pure_sparsercnn_as_the_objdet_when_rel):
                weight_dict = self.criterion.weight_dict
                for k in loss_dict.keys():
                    if k in weight_dict:
                        loss_dict[k] *= weight_dict[k]
                return proposal_features[-1], proposals, loss_dict
            else:
                if self.training or (not self.use_pure_sparsercnn_as_the_objdet_when_rel):
                    num_limit=None
                else:
                    num_limit=self.cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
                with torch.no_grad():
                    if not self.use_pure_sparsercnn_as_the_objdet_when_rel:
                        hungarian_targets = None
                    
                    x, result = inference_obj(proposal_features[-1], \
                        pred_bboxes[-1], class_logits[-1], images_whwh[:,:2], \
                        num_limit=num_limit, \
                        indices=all_hit_targets_list, targets=hungarian_targets)
                return x, result, {}
        else:
            if self.training:
                # Faster R-CNN subsamples during training the proposals with a fixed
                # positive / negative ratio
                with torch.no_grad():
                    proposals = self.samp_processor.subsample(proposals, targets)

            # extract features that will be fed to the final classifier. The
            # feature_extractor generally corresponds to the pooler + heads
            x = self.feature_extractor(features, proposals)
            # final classifier that converts the features into predictions
            class_logits, box_regression = self.predictor(x)
            
            if not self.training:
                x, result = self.post_processor((x, class_logits, box_regression), proposals)

                # if we want to save the proposals, we need sort them by confidence first.
                if self.cfg.TEST.SAVE_PROPOSALS:
                    _, sort_ind = result.get_field("pred_scores").view(-1).sort(dim=0, descending=True)
                    x = x[sort_ind]
                    result = result[sort_ind]
                    result.add_field("features", x.cpu().numpy())

                return x, result, {}

            loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression], proposals)

            return x, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)

def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)