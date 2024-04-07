import numpy as np
import torch
import torch.nn.functional as F

from sgg_benchmark.structures.bounding_box import BoxList

from torchvision.ops.boxes import box_area

def mymatcher(triplet_outputs, hungarian_targets, matcher, basedd_C=None, return_C=False):
    if return_C:
        rel_pred_logits = triplet_outputs['rel_pred_logits'].data.cpu().numpy()
        C_list, C_max_id = matcher(triplet_outputs, hungarian_targets, return_C=return_C, basedd_C=basedd_C)
        indices = []
        for i, C in enumerate(C_list):
            relscore = np.max(rel_pred_logits[i], axis=-1)
            rel_id = np.argsort(relscore)
            
            min_c_id = np.argmin(C[rel_id], axis=-1)
            uniqued_ent_id, uniqued_tri_id = np.unique(min_c_id, return_index=True)
            
            indices.append((rel_id[uniqued_tri_id], uniqued_ent_id))
    else:
        indices, _, _ = matcher(triplet_outputs, hungarian_targets, return_C=return_C, basedd_C=basedd_C)
    
    return indices

def pred_hungar(pure_ent, pure_ent_boxes, triplet_outputs, img_size_xyxy, matcher, use_sig=True):
    N, ent_nr_boxes = pure_ent.shape[:2]
    sub_id, obj_id = np.meshgrid(np.arange(ent_nr_boxes), np.arange(ent_nr_boxes), indexing='ij')
    sub_id, obj_id = sub_id.reshape(-1), obj_id.reshape(-1)
    rmv_self_id = np.where(sub_id != obj_id)[0]
    sub_id = sub_id[rmv_self_id]
    obj_id = obj_id[rmv_self_id]
    
    if use_sig:
        pure_ent_scores = pure_ent.view(N, ent_nr_boxes, -1)
        pre_max_pure_ent_scores = torch.sigmoid(pure_ent_scores)
        pre_max_pure_ent_scores = -pre_max_pure_ent_scores.log()
        max_pure_ent_scores, pure_ent_labels = torch.max(pre_max_pure_ent_scores, -1)
    else:
        pure_ent_scores = pure_ent.view(N, ent_nr_boxes, -1)
        pre_max_pure_ent_scores = F.softmax(pure_ent_scores, -1)
        pre_max_pure_ent_scores = -pre_max_pure_ent_scores.log()
        max_pure_ent_scores, pure_ent_labels = torch.max(pre_max_pure_ent_scores, -1)
    
    max_pure_ent_scores_s = max_pure_ent_scores[:, sub_id]
    max_pure_ent_scores_s = max_pure_ent_scores_s.view(1, -1)
    max_pure_ent_scores_o = max_pure_ent_scores[:, obj_id]
    max_pure_ent_scores_o = max_pure_ent_scores_o.view(1, -1)
    max_pure_ent_scores = (max_pure_ent_scores_s, max_pure_ent_scores_o)
    
    box_pair = triplet_outputs['pred_boxes'].clone().detach()
    cls_score_pair = triplet_outputs['pred_logits']
    
    pure_ent_boxes = pure_ent_boxes.data.cpu().numpy()
    pure_ent_labels = pure_ent_labels.data.cpu().numpy()
    pure_ent_scores = pure_ent_scores.data.cpu().numpy()
    hungarian_targets = list()
    for i in range(N):
        w, h = img_size_xyxy[i, :2]
        
        
        sub_label = pure_ent_labels[i, sub_id]
        obj_label = pure_ent_labels[i, obj_id]
        sub_obj_label = np.stack([sub_label, obj_label]).T
        
        sub_bbox = pure_ent_boxes[i, sub_id]
        obj_bbox = pure_ent_boxes[i, obj_id]
        sub_obj_box = np.hstack((sub_bbox, obj_bbox))
        
        sub_label_scores = pure_ent_scores[i, sub_id]
        obj_label_scores = pure_ent_scores[i, obj_id]
        label_scores = np.hstack((sub_label_scores, obj_label_scores))
        
        image_size_xyxy = np.array([w, h, w, h], dtype=sub_obj_box.dtype)
        image_size_xyxy_tgt = np.tile(image_size_xyxy[None], (len(sub_id), 1))
        hungarian_targets.append(dict(labels = sub_obj_label,
                                    label_scores = label_scores,
                                    boxes_xyxy = sub_obj_box, 
                                    image_size_xyxy = image_size_xyxy, 
                                    image_size_xyxy_tgt = image_size_xyxy_tgt, 
                                    sub_id=sub_id,
                                    obj_id=obj_id,))
    
    indices = mymatcher(triplet_outputs, hungarian_targets, matcher, basedd_C=max_pure_ent_scores)

    for idx, (i,j) in enumerate(indices):   
        ent_boxes = hungarian_targets[idx]['boxes_xyxy']
        ent_scores = hungarian_targets[idx]['label_scores']
        
        select_ent_score = torch.from_numpy(ent_scores[j]).to(pure_ent.device)
        select_ent_score = select_ent_score.view(2 * select_ent_score.shape[0], -1)
        score_w, _ = torch.max(select_ent_score, dim=-1, keepdims=True)
        
        hit_score = cls_score_pair[idx, i]
        hit_score = hit_score.view(hit_score.shape[0] * 2, -1)
        hit_score += score_w
        hit_score = hit_score.view(len(i), -1)
        
        cls_score_pair[idx, i] = hit_score
        
        
        
        #cls_score_pair[idx, i] = torch.from_numpy(ent_scores[j]).to(pure_ent.device)
        #box_pair[idx, i] = torch.from_numpy(ent_boxes[j]).to(pure_ent.device)
        
    return cls_score_pair, box_pair
    
def rmv_non_overlap_pair(sorting_idx, ori_obj_boxes_i, thre=1e-5):
    idmask_rmv_self = ~(pairwise_iou(ori_obj_boxes_i[0::2], ori_obj_boxes_i[1::2]) < thre)
    sorting_idx = sorting_idx[idmask_rmv_self]
    return sorting_idx

def rmv_impossible_pair(sorting_idx, ori_obj_scores_i, score_thre=0.15):
    filter_score = score_thre
    while filter_score >= -1e-5:
        saved_id = torch.where((ori_obj_scores_i[0::2] > filter_score) & \
            (ori_obj_scores_i[1::2] > filter_score))[0]
        if len(saved_id) < 1:
            filter_score -= 0.01
            continue
        sorting_idx = sorting_idx[saved_id]
        break
    
    return sorting_idx

def rmv_self_connection(sorting_idx, ori_obj_labels_i, ori_obj_boxes_i, thre=0.95):
    idmask_rmv_self = ~((ori_obj_labels_i[0::2] == ori_obj_labels_i[1::2]) & \
        (pairwise_iou(ori_obj_boxes_i[0::2], ori_obj_boxes_i[1::2]) > thre))
    sorting_idx = sorting_idx[idmask_rmv_self]
    return sorting_idx

def pairwise_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, :2], boxes2[:, :2].type(boxes1.dtype))  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:].type(boxes1.dtype))  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N, ]
    union = area1 + area2 - inter
    iou = inter / union
    return iou

def batch_box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 3] - boxes1[:, 1] + 1) * (boxes1[:, 2] - boxes1[:, 0] + 1)
    area2 = (boxes2[:, 3] - boxes2[:, 1] + 1) * (boxes2[:, 2] - boxes2[:, 0] + 1)
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt) * (rb - lt >= 0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union

    
def rmv_pre_duplicate_pair(ori_obj_boxes, ori_obj_labels, ori_obj_scores, ori_rel_labels, ori_rel_score, N, nr_boxes, R=2, score_thre=0.15, low_thre=0.1, use_times_rmv=True, rmv_times=2.2):
    """
        return:
        
        ori_obj_boxes: N, nr_boxes * R*R * 2, 4
        ori_obj_labels: N, nr_boxes * R*R * 2
        ori_obj_scores: N, nr_boxes * R*R * 2
        ori_rel_labels: N, nr_boxes * R*R * 2
        ori_rel_score: N, nr_boxes * R * R, -1
    """
    ori_obj_boxes = ori_obj_boxes.view(N, nr_boxes, R*R, 2, 4)
    ori_obj_labels = ori_obj_labels.view(N, nr_boxes, R*R, 2)
    ori_obj_scores = ori_obj_scores.view(N, nr_boxes, R*R, 2)
    ori_rel_labels = ori_rel_labels.view(N, nr_boxes, R*R)
    ori_rel_score = ori_rel_score.view(N, nr_boxes, R*R, -1)
    
    obj_boxes = []
    obj_labels = []
    obj_scores = []
    rel_labels = []
    rel_score = []
    
    for ii in range(N):
        if use_times_rmv:
            select_idx = np.zeros((nr_boxes, R*R), dtype=np.bool)
            for i in range(0, R*R):
                s_standa = ori_obj_scores[ii, :, 0, 0]
                o_standa = ori_obj_scores[ii, :, 0, 1]
                cand_s_score = ori_obj_scores[ii, :, i, 0]
                cand_o_score = ori_obj_scores[ii, :, i, 1]
                valid_s_mask = (cand_s_score * rmv_times >= s_standa)
                valid_o_mask = (cand_o_score * rmv_times >= o_standa)
                valid_mask = (valid_s_mask & valid_o_mask)
                select_nr_boxes = torch.where(valid_mask)[0]
                select_nr_boxes = select_nr_boxes.data.cpu().numpy()
                select_idx[select_nr_boxes, i] = True
        else:
            low_thre_N = low_thre
            score_thre_N = score_thre
            while low_thre_N >= -1e-5:
                select_idx = np.zeros((nr_boxes, R*R), dtype=np.bool)
                for i in range(0, R*R):
                    s_standa = ori_obj_scores[ii, :, 0, 0]
                    o_standa = ori_obj_scores[ii, :, 0, 1]
                    cand_s_score = ori_obj_scores[ii, :, i, 0]
                    cand_o_score = ori_obj_scores[ii, :, i, 1]
                    high_s_standa = (s_standa < score_thre_N) & (s_standa >= low_thre_N)
                    high_o_standa = (o_standa < score_thre_N) & (o_standa >= low_thre_N)
                    low_cand_s_score = (cand_s_score >= score_thre_N)
                    low_cand_o_score = (cand_o_score >= score_thre_N)
                    valid_s_mask = (high_s_standa | low_cand_s_score)
                    valid_o_mask = (high_o_standa | low_cand_o_score)
                    valid_mask = (valid_s_mask & valid_o_mask)
                    select_nr_boxes = torch.where(valid_mask)[0]
                    select_nr_boxes = select_nr_boxes.data.cpu().numpy()
                    #select_i = i * torch.ones_like(select_nr_boxes)
                    #select_idx[select_nr_boxes, select_i] = True
                    select_idx[select_nr_boxes, i] = True
                if select_idx.sum() <= 0:
                    low_thre_N -= 0.05
                    score_thre_N -= 0.05
                else:
                    break
    
        obj_boxes.append(ori_obj_boxes[ii][select_idx].view(-1, 4))
        obj_labels.append(ori_obj_labels[ii][select_idx].view(-1))
        obj_scores.append(ori_obj_scores[ii][select_idx].view(-1))
        rel_labels.append(ori_rel_labels[ii][select_idx].view(-1))
        rel_score.append(ori_rel_score[ii][select_idx])
    return obj_boxes, obj_labels, obj_scores, rel_labels, rel_score

    
def self_score_constraint(cls_score_pair):
    ans = cls_score_pair**2 / torch.sum(cls_score_pair**2, dim=-1, keepdims=True)
    return ans

def inference_obj(features, box, cls_score, img_size_xyxy, num_limit=256, indices=None, targets=None):
    N, nr_boxes = cls_score.shape[:2]
    cls_logit = cls_score
    obj_score_padding = torch.zeros(N, nr_boxes, 1).to(cls_score.device)
    obj_logit_padding = -11. * torch.ones(N, nr_boxes, 1).to(cls_score.device)
    cls_score = F.sigmoid(cls_score)
    pre_obj_scores = cls_score.view(N, nr_boxes, -1)
    pre_obj_scores = torch.cat([obj_score_padding, pre_obj_scores], -1)
    pre_obj_logits = torch.cat([obj_logit_padding, cls_logit], -1)
    obj_scores, obj_labels = torch.max(pre_obj_scores, -1)
    obj_boxes = box.view(N, nr_boxes, -1)
    R = 3
    num_cls = pre_obj_scores.shape[-1]

    s_score, s_class = torch.topk(pre_obj_scores, k=R, dim=-1, sorted=True) # N, nr_boxes, R
    extend_obj_boxes_per_class = obj_boxes.view(N, nr_boxes, 1, 4)
    extend_obj_boxes_per_class = extend_obj_boxes_per_class.expand(N, nr_boxes, num_cls, 4)
    select_boxes = []
    select_feats = []
    select_logit_vecs = []
    select_boxes_pre_class = []
    pred_labels = []
    pred_scores = []
    
    add_labels = None
    if targets is not None:
        add_labels = []
    
    if num_limit is None:
        num_limit = nr_boxes
        
    for i in range(N):
        flatten_s_score = s_score[i].view(-1)
        flatten_s_label = s_class[i].view(-1)
        _, sorting_idx = torch.sort(flatten_s_score, dim=0, descending=True)
        sorting_idx = sorting_idx[:num_limit]
        select_box = obj_boxes[i, sorting_idx//R]
        #print(select_box)
        select_features = features[i, sorting_idx//R]
        select_logit_vec = pre_obj_logits[i, sorting_idx//R]
        select_box_pre_class = extend_obj_boxes_per_class[i, sorting_idx//R]
        select_feats.append(select_features)
        select_boxes.append(select_box)
        select_boxes_pre_class.append(select_box_pre_class)
        select_logit_vecs.append(select_logit_vec)
        pred_labels.append(flatten_s_label[sorting_idx])
        pred_scores.append(flatten_s_score[sorting_idx])
        if targets is not None:
            label = targets[i]['labels'] + 1
            pred_id, gt_id = indices[i]
            correspond_label_vec = np.zeros(nr_boxes)
            correspond_label_vec[pred_id] = label[gt_id]
            correspond_label_vec = torch.tensor(correspond_label_vec).long().to(cls_score.device)
            
            add_label = correspond_label_vec[sorting_idx//R]
            gt_box = targets[i]['boxes_xyxy']
            iou, _ = batch_box_iou(select_box.data.cpu().numpy(), gt_box)
            iou = torch.tensor(iou).float().to(cls_score.device)
            label_comp = torch.tensor(label).long().to(cls_score.device)
            hit_mask = (label_comp.view(1, -1) == add_label.view(-1, 1)) & (iou > 0.5)
            
            #hung_label_match = torch.zeros(len(sorting_idx), len(label)).long().to(cls_score.device)
            #hung_label_match[pred_id, gt_id] = 1
            #hit_mask = (hung_label_match > 0) & hit_mask
            new_pred_id, new_gt_id = torch.where(hit_mask)
            
            add_label = torch.zeros(len(sorting_idx)).long().to(cls_score.device)
            add_label[new_pred_id] = label_comp[new_gt_id]
            add_labels.append(add_label)
    
    results = list()
    for i in range(N):
        boxlist = BoxList(select_boxes[i], tuple(img_size_xyxy[i].data.cpu().numpy()))
        boxlist.add_field('pred_labels', pred_labels[i]) # (#obj, )
        boxlist.add_field('pred_scores', pred_scores[i]) # (#obj, )
        boxlist.add_field('predict_logits', select_logit_vecs[i])
        boxlist.add_field('boxes_per_cls', select_boxes_pre_class[i])
        if add_labels is not None:
            boxlist.add_field('labels', add_labels[i])
        results.append(boxlist)
    
    select_feats = torch.cat(select_feats, 0)
    
    return select_feats, results

def inference_pair(box_pair, cls_score_pair, rel_score, img_size_xyxy, tri_score=None, \
  use_sig=True, use_all_obj_score=True, enable_vis_obj_det_metric=False, counterfact_rel_logits=None, \
  use_triplet_nms=False, use_last_relness=False):
    """
        box_pair: N, nr_boxes, 8
        cls_score_pair: N, nr_boxes, 2 * num_class
        rel_score: N, nr_boxes, num_rel_class
        img_size_xyxy: N, 4
    """
    N, nr_boxes = cls_score_pair.shape[:2]
    if use_sig:
        obj_score_padding = torch.zeros(N, 2 * nr_boxes, 1).to(cls_score_pair.device)
        rel_score_padding = torch.zeros(N, nr_boxes, 1).to(cls_score_pair.device)
        
        cls_score_pair = F.sigmoid(cls_score_pair)
        pre_obj_scores = cls_score_pair.view(N, 2 * nr_boxes, -1)
        
        
        if tri_score is not None:
            pre_obj_scores = pre_obj_scores * tri_score.view(N, 2 * nr_boxes, -1)
        
        
        pre_obj_scores = torch.cat([obj_score_padding, pre_obj_scores], -1)
        obj_scores, obj_labels = torch.max(pre_obj_scores, -1)
        obj_boxes = box_pair.view(N, 2 * nr_boxes, -1)
        
        if counterfact_rel_logits is not None:
            if use_last_relness:
                if rel_score.shape[-1] == counterfact_rel_logits.shape[-1]:
                    counterfact_rel_logits = counterfact_rel_logits[:, :, -1]
                
                rel_score[:, :, :-1] = rel_score[:, :, :-1] - counterfact_rel_logits
            else:
                rel_score = rel_score - counterfact_rel_logits
        
        rel_score = F.sigmoid(rel_score)
        if use_last_relness:
            rel_score = rel_score[:, :, :-1] * (rel_score[:, :, [-1]])
        
        rel_score = torch.cat([rel_score_padding, rel_score], -1)
        _, rel_labels = torch.max(rel_score, -1)
    else:
        cls_score_pair = F.softmax(cls_score_pair, -1)
        pre_obj_scores = cls_score_pair.view(N, 2 * nr_boxes, -1)
        
        if tri_score is not None:
            pre_obj_scores = pre_obj_scores * tri_score.view(N, 2 * nr_boxes, -1)
        
        obj_scores, obj_labels = torch.max(pre_obj_scores, -1)
        
        obj_boxes = box_pair.view(N, 2 * nr_boxes, -1)
        
        rel_score = F.softmax(rel_score, -1)
        _, rel_labels = torch.max(rel_score, -1)
    
    R = 5
    R = min(max(1, R), pre_obj_scores.shape[-1]-1)
    
    if use_all_obj_score:
        pre_obj_scores_s = pre_obj_scores[:, 0::2] # N, nr_boxes, cls_num
        pre_obj_scores_o = pre_obj_scores[:, 1::2]
        s_score, s_class = torch.topk(pre_obj_scores_s, k=R, dim=-1, sorted=True) # N, nr_boxes, R
        o_score, o_class = torch.topk(pre_obj_scores_o, k=R, dim=-1, sorted=True)
        s_boxes = obj_boxes[:, 0::2, :] # N, nr_boxes, 4
        o_boxes = obj_boxes[:, 1::2, :]
        
        ori_obj_boxes = []
        ori_obj_labels = []
        ori_obj_scores = []
        ori_rel_labels = []
        ori_rel_score = [] # rel_score: N, nr_boxes, num_rel_class

        
        X, Y = torch.meshgrid(torch.arange(R), torch.arange(R))
        X = X.reshape(-1)
        Y = Y.reshape(-1)
        
        s_score_d = s_score[:, :, X] # N, nr_boxes, R*R
        o_score_d = o_score[:, :, Y]
        score_d = torch.cat([s_score_d.unsqueeze(3), \
            o_score_d.unsqueeze(3)], 3).to(rel_score.device) # N, nr_boxes, R*R, 2
        
        s_class_d = s_class[:, :, X] # N, nr_boxes, R*R
        o_class_d = o_class[:, :, Y]
        class_d = torch.cat([s_class_d.unsqueeze(3), \
            o_class_d.unsqueeze(3)], 3).to(rel_score.device) # N, nr_boxes, R*R, 2
        
        s_boxes = s_boxes.unsqueeze(2).repeat(1,1,R,1) # N, nr_boxes, R, 4
        o_boxes = o_boxes.unsqueeze(2).repeat(1,1,R,1)
        s_boxes_d = s_boxes[:, :, X, :] # N, nr_boxes, R*R, 4
        o_boxes_d = o_boxes[:, :, Y, :]
        boxes_d = torch.cat([s_boxes_d.unsqueeze(3), \
            o_boxes_d.unsqueeze(3)], 3).to(rel_score.device) # N, nr_boxes, R*R, 2, 4
        
        rel_d = rel_score.unsqueeze(2).repeat(1, 1, R**2, 1) # N, nr_boxes, R*R, num_rel_class
        
        
        _, aug_rel_label = torch.max(rel_d, -1) # N, nr_boxes, R*R
        
        boxes_d = boxes_d.view(N, -1, 4)
        class_d = class_d.view(N, -1) # N, nr_boxes * R*R * 2
        score_d = score_d.view(N, -1)
        aug_rel_label = aug_rel_label.view(N, -1)
        rel_d = rel_d.view(N, nr_boxes * R * R, -1)
        
        ori_obj_boxes = boxes_d
        ori_obj_labels = class_d
        ori_obj_scores = score_d
        ori_rel_labels = aug_rel_label
        ori_rel_score = rel_d
    else:
        ori_obj_boxes = obj_boxes #N, 2 * nr_boxes, -1
        ori_obj_labels = obj_labels
        ori_obj_scores = obj_scores
        ori_rel_labels = rel_labels
        ori_rel_score = rel_score
        
    # sorting
    
    obj_boxes = []
    obj_labels = []
    obj_scores = []
    rel_labels = []
    rel_score = []
    rel_pair_idx_list = []
    
    for i in range(N):
        per_tri_rel_scores, per_tri_rel_class = ori_rel_score[i][:, 1:].max(dim=-1) # nr_boxes * R*R
        per_tri_rel_class = per_tri_rel_class + 1
        
        triple_scores = per_tri_rel_scores * ((ori_obj_scores[i][0::2] * ori_obj_scores[i][1::2]))
            
        _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
        
        
        if use_triplet_nms:
            sorting_idx = enable_triple_nms(ori_obj_labels[i][0::2], ori_obj_labels[i][1::2], \
                ori_rel_labels[i], ori_obj_boxes[i][0::2], ori_obj_boxes[i][1::2], sorting_idx, thresh=0.6)
        
        
        sorting_idx = sorting_idx[:100] 
        
        obj_sorting_idx = torch.stack([2*sorting_idx, 2*sorting_idx+1]).t().to(ori_rel_score[i].device)
        obj_sorting_idx = obj_sorting_idx.reshape(-1)
        
        top_ent_box = ori_obj_boxes[i][obj_sorting_idx]
        top_ent_label = ori_obj_labels[i][obj_sorting_idx]
        top_ent_score = ori_obj_scores[i][obj_sorting_idx]
        rel_pair_idx = torch.arange(0, len(ori_obj_boxes[i][obj_sorting_idx])).view(-1, 2)
        
        if enable_vis_obj_det_metric:
            top_ent_label, top_ent_box, top_ent_score, rel_pair_idx = \
                visualize_object_det_rough(\
                    top_ent_score[0::2], top_ent_score[1::2], \
                    top_ent_label[0::2], top_ent_label[1::2], \
                    top_ent_box[0::2], top_ent_box[1::2])
        
        obj_boxes.append(top_ent_box)
        obj_labels.append(top_ent_label)
        obj_scores.append(top_ent_score)
        rel_score.append(ori_rel_score[i][sorting_idx])
        rel_labels.append(ori_rel_labels[i][sorting_idx])
        rel_pair_idx_list.append(rel_pair_idx)
    
    
    results = list()
    for i in range(N):
        boxlist = BoxList(obj_boxes[i], tuple(img_size_xyxy[i].data.cpu().numpy()))
        boxlist.add_field('pred_labels', obj_labels[i]) # (#obj, )
        boxlist.add_field('pred_scores', obj_scores[i]) # (#obj, )
        boxlist.add_field('rel_pair_idxs', rel_pair_idx_list[i]) # (#rel, 2)
        boxlist.add_field('pred_rel_scores', rel_score[i]) # (#rel, #rel_class)
        boxlist.add_field('pred_rel_labels', rel_labels[i]) # (#rel, )
        results.append(boxlist)
    return results
    
    
def visualize_object_det_rough(sub_score, obj_score, sub_cls, obj_cls, sub_boxes, obj_boxes, thresh=0.5):
    device_id = sub_cls.device

    if isinstance(sub_score, torch.Tensor):
        sub_score = sub_score.data.cpu().numpy()
    if isinstance(obj_score, torch.Tensor):
        obj_score = obj_score.data.cpu().numpy()
    if isinstance(sub_boxes, torch.Tensor):
        sub_boxes = sub_boxes.data.cpu().numpy()
    if isinstance(obj_boxes, torch.Tensor):
        obj_boxes = obj_boxes.data.cpu().numpy()    
    if isinstance(obj_cls, torch.Tensor):
        obj_cls = obj_cls.data.cpu().numpy() 
    if isinstance(sub_cls, torch.Tensor):
        sub_cls = sub_cls.data.cpu().numpy()
    
    sub_score = sub_score.reshape(-1, 1)
    obj_score = obj_score.reshape(-1, 1)
    sub_score = np.concatenate([sub_score, obj_score], -1)
    sub_score = sub_score.reshape(-1)
    
    sub_cls = sub_cls.reshape(-1, 1)
    obj_cls = obj_cls.reshape(-1, 1)
    sub_cls = np.concatenate([sub_cls, obj_cls], -1)
    sub_cls = sub_cls.reshape(-1)
    
    sub_boxes = np.concatenate([sub_boxes, obj_boxes], -1)
    sub_boxes = sub_boxes.reshape(-1, 4)
    
    
    sub_cls_row = sub_cls.reshape(-1, 1)
    sub_cls_col = sub_cls.reshape(1, -1)
    sub_cls_correct = (sub_cls_row == sub_cls_col)
    
    s_iou_correct = (batch_box_iou(sub_boxes, sub_boxes)[0] >= thresh)
    
    ent_id = np.arange(len(sub_cls)).reshape(-1, 1) + 1
    correct = sub_cls_correct * s_iou_correct * ent_id
    
    X, Y = np.where(correct <= 0)
    correct[X, Y] = len(sub_cls) + 10
    correct -= 1
    nms_id_map = np.min(correct, axis=0)
    
    
    
    mp_id = dict()
    lst = -1
    new_ent_score = []
    new_ent_box = []
    new_ent_label = []
    new_map_id = []
    for i in range(len(nms_id_map)):
        if nms_id_map[i] in mp_id:
            new_map_id.append(mp_id[nms_id_map[i]])
        else:
            lst += 1
            mp_id[nms_id_map[i]] = lst
            
            new_ent_label.append(sub_cls[nms_id_map[i]])
            new_ent_box.append(sub_boxes[nms_id_map[i]])
            new_ent_score.append(sub_score[nms_id_map[i]])
            new_map_id.append(lst)
    
    
    new_ent_label = np.array(new_ent_label, dtype=np.int32)
    new_ent_box = np.stack(new_ent_box, 0)
    new_ent_score = np.array(new_ent_score, dtype=np.float32)
    new_map_id = np.array(new_map_id, dtype=np.int32).reshape(-1, 2)
    
    new_ent_label = torch.from_numpy(new_ent_label).to(device_id)
    new_ent_box = torch.from_numpy(new_ent_box).to(device_id)
    new_ent_score = torch.from_numpy(new_ent_score).to(device_id)
    new_map_id = torch.from_numpy(new_map_id).to(device_id)
    
    return new_ent_label, new_ent_box, new_ent_score, new_map_id 
    
def rough_unique(sub_cls, obj_cls, pred_cls, sub_boxes, obj_boxes, sorting_idx, resize_co=10.):
    sub_cls = sub_cls[sorting_idx]
    obj_cls = obj_cls[sorting_idx]
    pred_cls = pred_cls[sorting_idx]
    sub_boxes = sub_boxes[sorting_idx]
    obj_boxes = obj_boxes[sorting_idx]
    if isinstance(sub_boxes, torch.Tensor):
        sub_boxes = sub_boxes.data.cpu().numpy()
    if isinstance(obj_boxes, torch.Tensor):
        obj_boxes = obj_boxes.data.cpu().numpy()    
    if isinstance(pred_cls, torch.Tensor):
        pred_cls = pred_cls.data.cpu().numpy() 
    if isinstance(obj_cls, torch.Tensor):
        obj_cls = obj_cls.data.cpu().numpy() 
    if isinstance(sub_cls, torch.Tensor):
        sub_cls = sub_cls.data.cpu().numpy()
    
    resized_sub_boxes = (sub_boxes / resize_co).astype(np.int32)
    resized_obj_boxes = (obj_boxes / resize_co).astype(np.int32)
    
    triplet_feat = np.concatenate([sub_cls.reshape(-1, 1), obj_cls.reshape(-1, 1), pred_cls.reshape(-1, 1), \
        resized_sub_boxes.reshape(-1, 4), resized_obj_boxes.reshape(-1, 4)], -1).astype(np.int32)
    _, keep = np.unique(triplet_feat, return_index=True, axis=0)
    
    keep = np.sort(keep)
    return sorting_idx[torch.from_numpy(keep).long()]
    
def enable_triple_nms(sub_cls, obj_cls, pred_cls, sub_boxes, obj_boxes, sorting_idx, thresh=0.5, is_fast=False):
    #sorting_idx = sorting_idx[:3600]
    sorting_idx = sorting_idx[:1200]
    sub_cls = sub_cls[sorting_idx]
    obj_cls = obj_cls[sorting_idx]
    pred_cls = pred_cls[sorting_idx]
    sub_boxes = sub_boxes[sorting_idx]
    obj_boxes = obj_boxes[sorting_idx]
    if is_fast:
        keep = fast_triplet_nms(sub_cls, obj_cls, pred_cls, sub_boxes, obj_boxes, thresh=thresh)
    else:
        keep = triplet_nms(sub_cls, obj_cls, pred_cls, sub_boxes, obj_boxes, thresh=thresh)
    
    #return sorting_idx[torch.from_numpy(keep).long()]
    return sorting_idx[keep]

def fast_triplet_nms(sub_cls, obj_cls, pred_cls, sub_boxes, obj_boxes, thresh=0.4):
    if isinstance(sub_boxes, torch.Tensor):
        sub_boxes = sub_boxes.data.cpu().numpy()
    if isinstance(obj_boxes, torch.Tensor):
        obj_boxes = obj_boxes.data.cpu().numpy()    
    if isinstance(pred_cls, torch.Tensor):
        pred_cls = pred_cls.data.cpu().numpy() 
    if isinstance(obj_cls, torch.Tensor):
        obj_cls = obj_cls.data.cpu().numpy() 
    if isinstance(sub_cls, torch.Tensor):
        sub_cls = sub_cls.data.cpu().numpy()
        
    sub_cls_row = sub_cls.reshape(-1, 1)
    sub_cls_col = sub_cls.reshape(1, -1)
    sub_cls_correct = (sub_cls_row == sub_cls_col)
    
    obj_cls_row = obj_cls.reshape(-1, 1)
    obj_cls_col = obj_cls.reshape(1, -1)
    obj_cls_correct = (obj_cls_row == obj_cls_col)
    
    pred_cls_row = pred_cls.reshape(-1, 1)
    pred_cls_col = pred_cls.reshape(1, -1)
    pred_cls_correct = (pred_cls_row == pred_cls_col)
    
    s_iou_correct = (batch_box_iou(sub_boxes, sub_boxes)[0] >= thresh)
    o_iou_correct = (batch_box_iou(obj_boxes, obj_boxes)[0] >= thresh)
    
    ent_id = np.arange(len(sub_cls)).reshape(-1, 1) + 1
    correct = sub_cls_correct * s_iou_correct * o_iou_correct * obj_cls_correct * pred_cls_correct * ent_id
    
    X, Y = np.where(correct <= 0)
    correct[X, Y] = len(sub_cls) + 10
    correct -= 1
    keep = np.min(correct, axis=0)
    keep = np.unique(keep)
    return keep

def triplet_nms(sub_cls, obj_cls, pred_cls, sub_boxes, obj_boxes, thresh=0.4):
    if isinstance(sub_boxes, torch.Tensor):
        sub_boxes = sub_boxes.data.cpu().numpy()
    if isinstance(obj_boxes, torch.Tensor):
        obj_boxes = obj_boxes.data.cpu().numpy()    
    if isinstance(pred_cls, torch.Tensor):
        pred_cls = pred_cls.data.cpu().numpy() 
    if isinstance(obj_cls, torch.Tensor):
        obj_cls = obj_cls.data.cpu().numpy() 
    if isinstance(sub_cls, torch.Tensor):
        sub_cls = sub_cls.data.cpu().numpy() 
    
    sub_x1 = sub_boxes[:, 0]
    sub_y1 = sub_boxes[:, 1]
    sub_x2 = sub_boxes[:, 2]
    sub_y2 = sub_boxes[:, 3]
    obj_x1 = obj_boxes[:, 0]
    obj_y1 = obj_boxes[:, 1]
    obj_x2 = obj_boxes[:, 2]
    obj_y2 = obj_boxes[:, 3]


    sub_areas = (sub_x2 - sub_x1 + 1) * (sub_y2 - sub_y1 + 1)
    obj_areas = (obj_x2 - obj_x1 + 1) * (obj_y2 - obj_y1 + 1)
    order = np.array(range(len(sub_cls)))

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        sub_xx1 = np.maximum(sub_x1[i], sub_x1[order[1:]])
        sub_yy1 = np.maximum(sub_y1[i], sub_y1[order[1:]])
        sub_xx2 = np.minimum(sub_x2[i], sub_x2[order[1:]])
        sub_yy2 = np.minimum(sub_y2[i], sub_y2[order[1:]])
        sub_id = sub_cls[i]
        obj_xx1 = np.maximum(obj_x1[i], obj_x1[order[1:]])
        obj_yy1 = np.maximum(obj_y1[i], obj_y1[order[1:]])
        obj_xx2 = np.minimum(obj_x2[i], obj_x2[order[1:]])
        obj_yy2 = np.minimum(obj_y2[i], obj_y2[order[1:]])
        obj_id = obj_cls[i]
        pred_id = pred_cls[i]

        w = np.maximum(0.0, sub_xx2 - sub_xx1 + 1)
        h = np.maximum(0.0, sub_yy2 - sub_yy1 + 1)
        inter = w * h
        sub_ovr = inter / (sub_areas[i] + sub_areas[order[1:]] - inter)

        w = np.maximum(0.0, obj_xx2 - obj_xx1 + 1)
        h = np.maximum(0.0, obj_yy2 - obj_yy1 + 1)
        inter = w * h
        obj_ovr = inter / (obj_areas[i] + obj_areas[order[1:]] - inter)
        inds = np.where( (sub_ovr <= thresh) |
                    (obj_ovr <= thresh) |
                    (sub_cls[order[1:]] != sub_id) |
                    (obj_cls[order[1:]] != obj_id) |
                    (pred_cls[order[1:]] != pred_id) )[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)

