#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SparseRCNN model and criterion classes.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
#from fvcore.nn import sigmoid_focal_loss_jit
import numpy as np

from .util import box_ops
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou

from scipy.optimize import linear_sum_assignment

from .eqloss import lambda_cal
from .eqloss import get_counted_freq
from .eqloss import directly_get_counted_freq
from .eqloss import eql_loss
from .simrel_inference import *
from .utils_simrel_old import *

class SetCriterion(nn.Module):
    """ This class computes the loss for SparseRCNN.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, use_focal, \
      num_rel_classes, alpha = 0.25, gamma = 2.0, debug_rmv_bg = False, use_equ_loss=False, \
      enable_query_reverse=False, enable_kl_div=False, ent_freq=None, ent_freq_mu=4., use_last_relness=False, \
      kl_fg_reweight=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.debug_rmv_bg = debug_rmv_bg
        self.num_classes = num_classes
        self.num_rel_classes = num_rel_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal
        self.enable_kl_div = enable_kl_div
        self.use_last_relness = use_last_relness
        self.kl_fg_reweight = kl_fg_reweight
        if ent_freq is not None:
            ent_freq = torch.tensor(ent_freq)
            ent_freq_gamma = 3. - (1-ent_freq)**ent_freq_mu * (-(ent_freq+1e-8).log())**(1./ent_freq_mu)
            ent_freq_gamma = torch.clamp(ent_freq_gamma, min=0., max=2.)
            self.register_buffer('ent_freq_gamma', ent_freq_gamma)
        else:
            self.ent_freq_gamma = gamma
        
        if self.use_focal:
            self.focal_loss_alpha = alpha
            self.focal_loss_gamma = gamma
            self.num_classes -= 1
            self.num_rel_classes -= 1
        else:
            empty_weight = torch.ones(self.num_classes)
            empty_weight_rel = torch.ones(self.num_rel_classes)
            #empty_weight[-1] = self.eos_coef
            empty_weight[0] = self.eos_coef
            empty_weight_rel[0] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)
            self.register_buffer('empty_weight_rel', empty_weight_rel)
        
        
        freq_info, num_info = directly_get_counted_freq('vg')
        lambda_ = lambda_cal(freq_info, num_info, num_rel_classes-1, T=100)
        self.freq_info = freq_info
        self.lambda_ = lambda_
        self.use_equ_loss = use_equ_loss
        self.enable_query_reverse = enable_query_reverse
        
    def loss_rel_labels(self, outputs, targets, indices, num_boxes, so_id=None, loss_name='rel_pred_logits', out_loss_name='loss_rel', log=False):
        assert loss_name in outputs
        src_logits = outputs[loss_name]
        idx = self._get_src_permutation_idx(indices)
        
        #target_classes_o = torch.cat([t["rel_label"][J] for t, (_, J) in zip(targets, indices)])
        target_classes_o = np.concatenate([t["rel_label"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = np.full(src_logits.shape[:2], self.num_rel_classes, dtype=np.int32)
        
        target_classes[idx] = target_classes_o
        
        if self.use_focal:
            src_logits = src_logits.flatten(0, 1)
            target_classes = target_classes.reshape(-1)
            if not self.use_equ_loss:
                pos_inds = np.nonzero(target_classes != self.num_rel_classes)[0]
                labels = torch.zeros_like(src_logits)
                labels[pos_inds, target_classes[pos_inds]] = 1
                if self.use_last_relness:
                    labels[pos_inds, -1] = 1
                    
                class_loss = sigmoid_focal_loss(
                    src_logits, labels,
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="sum",
                ) / num_boxes
            else:
                target_classes[target_classes==self.num_rel_classes] = -1
                target_classes += 1
                class_loss = eql_loss(src_logits, target_classes, \
                                self.freq_info, self.lambda_, \
                                is_bce=self.use_focal, \
                                cls_is_sigmoid=self.use_focal)
            
            losses = {out_loss_name: class_loss}
        else:
            target_classes = torch.from_numpy(target_classes).type(torch.int64).to(src_logits.device)
            
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_rel)
            losses = {out_loss_name: loss_ce}
        return losses, target_classes
        
    def loss_labels(self, outputs, targets, indices, num_boxes, so_id=None, loss_name='pred_logits', out_loss_name='loss_ce', ent_det_only_fg=True, ent_bg_type='all', log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert loss_name in outputs
        src_logits = outputs[loss_name] # [batch_size, num_queries, num_classes * 2]
        
        idx = self._get_src_permutation_idx(indices)
        
        if self.enable_kl_div:
            target_classes_o = np.concatenate([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes_o = torch.from_numpy(target_classes_o).long().to(src_logits.device)
            target_classes_o = target_classes_o.view(-1, 1)
            
            target_classes_o_score = torch.cat([t["distri_score"][J] for t, (_, J) in zip(targets, indices)])
            target_classes_o_score = target_classes_o_score.type_as(src_logits).to(src_logits.device)
            target_classes_o_score = target_classes_o_score.view(-1, self.num_classes)
            
            src_logits_select = src_logits[idx]
            src_logits_select = src_logits_select.view(-1, self.num_classes)
            #src_logits_select = torch.gather(src_logits_select, -1, target_classes_o)
            
            alpha, gamma =self.focal_loss_alpha, self.ent_freq_gamma
            
            class_loss = sigmoid_focal_loss(
                src_logits_select, 
                target_classes_o_score,
                alpha=alpha,
                gamma=gamma,
                reduction="sum",
                fg_reweight = self.kl_fg_reweight,
            ) / num_boxes
            losses = {out_loss_name: class_loss}
            return losses, None
            
        target_classes_o = np.concatenate([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        if src_logits.shape[-1] > self.num_classes:
            if not self.use_focal:
                target_classes = np.full((src_logits.shape[0], src_logits.shape[1], 2), self.num_classes, dtype=np.int32)
                
            #target_classes_sub = torch.full(src_logits.shape[:2], 
            #            self.num_classes, dtype=torch.int64, device=src_logits.device) # [batch_size, num_queries]
            #target_classes_obj = torch.full(src_logits.shape[:2], 
            #            self.num_classes, dtype=torch.int64, device=src_logits.device) # [batch_size, num_queries]
            target_classes_sub = np.full(src_logits.shape[:2], self.num_classes, dtype=np.int32)
            target_classes_obj = np.full(src_logits.shape[:2], self.num_classes, dtype=np.int32)
        else:
            #target_classes = torch.full(src_logits.shape[:2], 
            #            self.num_classes, dtype=torch.int64, device=src_logits.device) # [batch_size, num_queries]
            target_classes = np.full(src_logits.shape[:2], self.num_classes, dtype=np.int32)
            
            target_classes[idx] = target_classes_o # [batch_size, num_queries, 2]
        
        if src_logits.shape[-1] > self.num_classes:
            batch_idx, idx_in_batch = idx
            if so_id is not None:
                so_idx = self._get_subobj_permutation_idx(so_id)
                batch_idx_s, idx_in_batch_s = batch_idx[~so_idx], idx_in_batch[~so_idx]
                batch_idx_o, idx_in_batch_o = batch_idx[so_idx], idx_in_batch[so_idx]
                target_classes_sub[batch_idx_s, idx_in_batch_s] = target_classes_o[~so_idx, 0]
                target_classes_obj[batch_idx_o, idx_in_batch_o] = target_classes_o[so_idx, 1]
            else:
                batch_idx_s, idx_in_batch_s = batch_idx, idx_in_batch
                batch_idx_o, idx_in_batch_o = batch_idx, idx_in_batch
                target_classes_sub[batch_idx_s, idx_in_batch_s] = target_classes_o[:, 0]
                target_classes_obj[batch_idx_o, idx_in_batch_o] = target_classes_o[:, 1]
            target_classes = (target_classes_sub, target_classes_obj)
            
        if self.use_focal:
            alpha, gamma =self.focal_loss_alpha, self.focal_loss_gamma
            if src_logits.shape[-1] > self.num_classes:
                gamma = self.ent_freq_gamma
                ##src_logits = src_logits.view(-1, self.num_classes)
                ##target_classes = target_classes.view(-1)
                ##pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
                ##labels = torch.zeros_like(src_logits) # [batch_size * num_queries * 2, num_classes]
                ##labels[pos_inds, target_classes[pos_inds]] = 1

                pos_inds, pos_inds_pred_box_id = np.nonzero(target_classes_sub != self.num_classes)
                label_s = np.zeros((src_logits.shape[0], src_logits.shape[1], self.num_classes), dtype=np.float32)
                
                label_s[pos_inds, pos_inds_pred_box_id, target_classes_sub[pos_inds, pos_inds_pred_box_id]] = 1
                
                label_s = label_s[batch_idx_s, idx_in_batch_s].reshape(-1, self.num_classes)
                label_s = torch.from_numpy(label_s).type_as(src_logits).to(src_logits.device)
                
                pos_inds, pos_inds_pred_box_id = np.nonzero(target_classes_obj != self.num_classes)
                
                label_o = np.zeros((src_logits.shape[0], src_logits.shape[1], self.num_classes), dtype=np.float32)
                
                label_o[pos_inds, pos_inds_pred_box_id, target_classes_obj[pos_inds, pos_inds_pred_box_id]] = 1
                
                label_o = label_o[batch_idx_o, idx_in_batch_o].reshape(-1, self.num_classes)
                label_o = torch.from_numpy(label_o).type_as(src_logits).to(src_logits.device)
                
                
                class_loss_s = sigmoid_focal_loss(
                    src_logits[batch_idx_s, idx_in_batch_s, :self.num_classes].view(-1, self.num_classes), 
                    label_s,
                    alpha=alpha,
                    gamma=gamma,
                    reduction="sum",
                )
                class_loss_o = sigmoid_focal_loss(
                    src_logits[batch_idx_o, idx_in_batch_o, self.num_classes:].view(-1, self.num_classes), 
                    label_o,
                    alpha=alpha,
                    gamma=gamma,
                    reduction="sum",
                )
                
                
                class_loss = (class_loss_s + class_loss_o) / num_boxes
                if not ent_det_only_fg:
                    bg_mask = np.ones((src_logits.shape[0], src_logits.shape[1]), dtype=np.bool)
                    bg_mask[idx] = False
                    if self.enable_query_reverse:
                        rvs_pre_mask = idx[1] < src_logits.shape[1]//2
                        rvs_post_mask = idx[1] >= src_logits.shape[1]//2
                        idx_rvs_p = (idx[0][rvs_pre_mask], idx[1][rvs_pre_mask] + src_logits.shape[1]//2)
                        idx_rvs_b = (idx[0][rvs_post_mask], idx[1][rvs_post_mask] - src_logits.shape[1]//2)
                        bg_mask[idx_rvs_p] = False
                        bg_mask[idx_rvs_b] = False
                    bg_mask = bg_mask.reshape(-1)
                    src_logits_bg = src_logits.view(-1, 2 * self.num_classes)[bg_mask]
                    src_logits_bg = src_logits_bg.view(-1, self.num_classes)
                    if ent_bg_type == 'random':
                        select_bg_mask = np.random.choice(2, len(src_logits_bg)).astype(np.bool)
                        src_logits_bg = src_logits_bg[select_bg_mask]
                    elif ent_bg_type == 'obj':
                        src_logits_bg = src_logits_bg[1::2]
                    
                    label_bg = torch.zeros_like(src_logits_bg)
                    
                    class_loss_bg = sigmoid_focal_loss(
                        src_logits_bg, 
                        label_bg,
                        alpha=alpha,
                        gamma=gamma,
                        reduction="sum",
                    ) / num_boxes
                    
                    if self.enable_query_reverse:
                        class_loss_bg = 0.5 * class_loss_bg
                        
                    class_loss = class_loss + class_loss_bg
            else:
                src_logits = src_logits.flatten(0, 1)
                
                target_classes = target_classes.reshape(-1)
                pos_inds = np.nonzero(target_classes != self.num_classes)[0]
                
                labels = torch.zeros_like(src_logits)
                labels[pos_inds, target_classes[pos_inds]] = 1
                if self.debug_rmv_bg:
                    fg_sample_id = torch.where(labels.sum(1) > 0)[0]
                    src_logits = src_logits[fg_sample_id]
                    labels = labels[fg_sample_id]
                
                # comp focal loss.
                if ent_det_only_fg:
                    class_loss = sigmoid_focal_loss(
                        src_logits[pos_inds],
                        labels[pos_inds],
                        alpha=alpha,
                        gamma=gamma,
                        reduction="sum",
                    ) / num_boxes
                else:
                    class_loss = sigmoid_focal_loss(
                        src_logits,
                        labels,
                        alpha=alpha,
                        gamma=gamma,
                        reduction="sum",
                    ) / num_boxes
            losses = {out_loss_name: class_loss}
        else:
            if src_logits.shape[-1] > self.num_classes:
                src_logits_s = src_logits[:, :, :self.num_classes]
                src_logits_o = src_logits[:, :, self.num_classes:]
                loss_ce_s = F.cross_entropy(src_logits.transpose(1, 2), 
                        target_classes[:, :, 0], self.empty_weight, reduction="none")
                loss_ce_o = F.cross_entropy(src_logits.transpose(1, 2), 
                        target_classes[:, :, 1], self.empty_weight, reduction="none")
                loss_ce = loss_ce_s[~so_idx] + loss_ce_o[so_idx]
            else:
                loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
            losses = {out_loss_name: loss_ce}

        return losses, target_classes


    def loss_boxes(self, outputs, targets, indices, num_boxes, so_id=None, loss_name='boxes', out_loss_name='loss_boxes'):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
           
           so_id: 0/1 mask
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        
        if self.enable_kl_div:
            boxes_xyxy_name = 'distri_boxes_xyxy'
        else:
            boxes_xyxy_name = 'boxes_xyxy'
        
        if isinstance(targets[0][boxes_xyxy_name], torch.Tensor):
            target_boxes = torch.cat([t[boxes_xyxy_name][i] for t, (_, i) in zip(targets, indices)])
        else:
            target_boxes = np.concatenate([t[boxes_xyxy_name][i] for t, (_, i) in zip(targets, indices)])
            target_boxes = torch.from_numpy(target_boxes).type_as(src_boxes).to(src_boxes.device)
        
        if so_id is not None:
            so_idx = self._get_subobj_permutation_idx(so_id)
        
        losses = {}
        if self.enable_kl_div:
            target_boxes_flat = target_boxes.view(-1, 4)
            src_boxes_flat = src_boxes.view(-1, 4)
            fg_id = torch.where(target_boxes_flat.sum(-1) >= 0)[0]
            
            loss_giou = 1 - torch.diag(\
                box_ops.generalized_box_iou(src_boxes_flat[fg_id], target_boxes_flat[fg_id]))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        else:
            if target_boxes.shape[-1] > 5:
                if so_id is not None:
                    loss_giou_s = 1 - torch.diag(\
                            box_ops.generalized_box_iou(src_boxes[~so_idx][:, :4], 
                                                    target_boxes[~so_idx][:, :4]))
                    loss_giou_o = 1 - torch.diag(\
                            box_ops.generalized_box_iou(src_boxes[so_idx][:, 4:], 
                                                    target_boxes[so_idx][:, 4:]))
                else:
                    loss_giou_s = 1 - torch.diag(\
                        box_ops.generalized_box_iou(src_boxes[:, :4], target_boxes[:, :4]))
                    loss_giou_o = 1 - torch.diag(\
                        box_ops.generalized_box_iou(src_boxes[:, 4:], target_boxes[:, 4:]))
                
                loss_giou = loss_giou_s.sum() + loss_giou_o.sum()
                losses['loss_giou'] = loss_giou / num_boxes
            else:
                loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes))
                losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        if isinstance(targets[0]["image_size_xyxy_tgt"], torch.Tensor):
            image_size = torch.cat(\
                [v["image_size_xyxy_tgt"][:len(indices[idx][0])] for idx, v in enumerate(targets)])
        else:
            #image_size = np.concatenate([v["image_size_xyxy_tgt"] for v in targets])
            image_size = np.concatenate(\
                [v["image_size_xyxy_tgt"][:len(indices[idx][0])] for idx, v in enumerate(targets)]) ###!!!
            image_size = torch.from_numpy(image_size).type_as(src_boxes).to(src_boxes.device)
        
        if target_boxes.shape[-1] > 5:
            image_size = image_size.repeat(1, 2)
        
        src_boxes_ = src_boxes / image_size
        target_boxes_ = target_boxes / image_size
        if self.enable_kl_div:
            target_boxes_flat_ = target_boxes_.view(-1, 4)
            src_boxes_flat_ = src_boxes_.view(-1, 4)
            loss_bbox = F.l1_loss(src_boxes_flat_[fg_id], target_boxes_flat_[fg_id], reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        else:
            if target_boxes.shape[-1] > 5:
                if so_id is not None:
                    loss_bbox_s = F.l1_loss(\
                        src_boxes_[~so_idx][:, :4], target_boxes_[~so_idx][:, :4], reduction='none')
                    loss_bbox_o = F.l1_loss(\
                        src_boxes_[so_idx][:, 4:], target_boxes_[so_idx][:, 4:], reduction='none')
                else:
                    loss_bbox_s = F.l1_loss(src_boxes_[:, :4], target_boxes_[:, :4], reduction='none')
                    loss_bbox_o = F.l1_loss(src_boxes_[:, 4:], target_boxes_[:, 4:], reduction='none')
                
                
                loss_bbox = loss_bbox_s.sum() + loss_bbox_o.sum()
                losses['loss_bbox'] = loss_bbox / num_boxes
                
            else:
                loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction='none')
                losses['loss_bbox'] = loss_bbox.sum() / num_boxes
                
                #print(src_boxes_)
                #print(target_boxes_)
                #print('num_boxes: {}'.format(num_boxes))
                #print('loss_bbox: {}'.format(loss_bbox.sum() / num_boxes))
                #print('obj l1')
                #print()

        return losses, target_boxes

    
    def _get_subobj_permutation_idx(self, so_id):
        #src_idx = torch.cat([src for src in so_id])
        src_idx = np.concatenate([src for src in so_id])
        return src_idx
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        #batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        #src_idx = torch.cat([src for (src, _) in indices])
        batch_idx = np.concatenate([np.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = np.concatenate([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        #batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        #tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        batch_idx = np.concatenate([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = np.concatenate([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, so_id, num_ent_boxes, ent_det_only_fg, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'rel_pred_logits': self.loss_rel_labels,
            'rel_prod_freq': self.loss_rel_labels,
            'obj2rel_logits': self.loss_rel_labels,
            'tri_score': self.ranking_loss,
        }
        
        in_loss_name_map = {
            'labels': 'pred_logits',
            'boxes': 'boxes',
            'rel_pred_logits': 'rel_pred_logits',
            'rel_prod_freq': 'rel_prod_freq',
            'obj2rel_logits': 'obj2rel_logits',
            'tri_score': 'tri_score',
        }
        
        out_loss_name_map = {
            'labels': 'loss_ce',
            'boxes': 'loss_boxes',
            'rel_pred_logits': 'loss_rel',
            'rel_prod_freq': 'loss_rel_freq_pro',
            'obj2rel_logits': 'loss_rel_obj2rel',
            'tri_score': 'loss_tri_score',
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        if loss.find('rel') >= 0:
            return loss_map[loss](outputs, targets, indices, num_boxes, so_id, in_loss_name_map[loss], out_loss_name_map[loss], **kwargs)
        else:
            if loss == 'labels':
                return loss_map[loss](outputs, targets, indices, num_ent_boxes, so_id, in_loss_name_map[loss], out_loss_name_map[loss], ent_det_only_fg, **kwargs)
            else:
                return loss_map[loss](outputs, targets, indices, num_ent_boxes, so_id, in_loss_name_map[loss], out_loss_name_map[loss], **kwargs)
            
    def forward(self, outputs, targets, enable_bg_obj=False, ent_det_only_fg=True):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        #all_hit_targets_list = list()

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, so_id, pair_gt_upper_bound_list = \
            self.matcher(outputs_without_aux, targets, enable_kl_div=self.enable_kl_div)
        
        all_hit_targets_list = indices
        
        if targets[0]['boxes_xyxy'].shape[1] > 4:
            num_times = 2
        else:
            num_times = 1
        
        if self.enable_kl_div:
            target_classes_o_score = \
                np.concatenate([t["count_ent_like_bg_num"][J] for t, (_, J) in zip(targets, indices)])
            num_ent_boxes = target_classes_o_score.sum() + 1
            for t in targets:
                num_ent_boxes += 2*t['gt_rel_num']
            num_ent_boxes = torch.as_tensor([num_ent_boxes], \
                dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_ent_boxes)
            num_ent_boxes = torch.clamp(num_ent_boxes / get_world_size(), min=1).item()
        else:
            if enable_bg_obj:

                extra_output, extra_indices, extra_hungarian_targets, _ = self.complement_allocate(outputs_without_aux, targets, indices)
                #num_ent_boxes = sum(num_times*len(t["labels"])+num_times*len(t["complem_label"]) for t in targets)
                num_ent_boxes = sum(num_times*len(t["labels"])+len(t["extra_entity_label"]) for t in targets)
                
                #num_ent_boxes = sum(num_times*len(t["labels"]) for t in targets)
                #num_ent_boxes = sum(num_times*len(t["labels"])+len(t["entity_label"]) for t in targets)
                
                num_ent_boxes = torch.as_tensor([num_ent_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
                if is_dist_avail_and_initialized():
                    torch.distributed.all_reduce(num_ent_boxes)
                num_ent_boxes = torch.clamp(num_ent_boxes / get_world_size(), min=1).item()
            else:
                #num_ent_boxes = sum(num_times*len(t["labels"]) for t in targets)
                num_ent_boxes = sum(num_times*min(len(t["labels"]), len(indices[idx][0])) for idx,t in enumerate(targets)) ###!!!
                num_ent_boxes = torch.as_tensor([num_ent_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
                if is_dist_avail_and_initialized():
                    torch.distributed.all_reduce(num_ent_boxes)
                num_ent_boxes = torch.clamp(num_ent_boxes / get_world_size(), min=1).item()
            
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        #num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = sum(min(len(t["labels"]), len(indices[idx][0])) for idx,t in enumerate(targets)) ###!!!
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        #print('---------------------')
        
        # Compute all the requested losses
        losses = {}
        hit_targets_list = list()
        for loss in self.losses:
            loss_dict, hit_targets = self.get_loss(loss, outputs, targets, indices, num_boxes, so_id, num_ent_boxes, ent_det_only_fg)
            losses.update(loss_dict)
            #hit_targets_list.append(hit_targets)
        #all_hit_targets_list.append(hit_targets_list)
        if enable_bg_obj:
            for i, target in enumerate(targets):
                extra_loss_ce, _ = self.loss_labels(extra_output[i], [extra_hungarian_targets[i]], [extra_indices[i]], num_ent_boxes, loss_name='pred_logits', out_loss_name='loss_ce', ent_det_only_fg=True)
                losses['loss_ce'] += extra_loss_ce['loss_ce']
                
                extra_loss_box, _ = self.loss_boxes(extra_output[i], [extra_hungarian_targets[i]], [extra_indices[i]], num_ent_boxes, loss_name='boxes', out_loss_name='loss_boxes')
                losses['loss_giou'] += extra_loss_box['loss_giou']
                losses['loss_bbox'] += extra_loss_box['loss_bbox']
                

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        aux_indicess_list = list()
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if self.enable_kl_div and 'aux_outputs' in targets:
                    ans_targets = []
                    for target in targets:
                        ans_targets.append(target['aux_outputs'][i])
                    targets = ans_targets
                    
                indices, so_id, pair_gt_upper_bound_list = \
                    self.matcher(aux_outputs, targets, enable_kl_div=self.enable_kl_div) ###!!!
                
                if self.enable_kl_div: #and 'aux_outputs' in targets:
                    target_classes_o_score = \
                        np.concatenate([t["count_ent_like_bg_num"][J] for t, (_, J) in zip(targets, indices)])
                    num_ent_boxes = target_classes_o_score.sum() + 1
                    for t in targets:
                        num_ent_boxes += 2*t['gt_rel_num']
                    num_ent_boxes = torch.as_tensor([num_ent_boxes], \
                        dtype=torch.float, device=next(iter(outputs.values())).device)
                    if is_dist_avail_and_initialized():
                        torch.distributed.all_reduce(num_ent_boxes)
                    num_ent_boxes = torch.clamp(num_ent_boxes / get_world_size(), min=1).item()
                
                aux_indicess_list.append(indices)
                if enable_bg_obj:
                    extra_output, extra_indices, extra_hungarian_targets, _ = self.complement_allocate(aux_outputs, targets, indices)
                
                #hit_targets_list = list()
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict, hit_targets = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, so_id, num_ent_boxes, ent_det_only_fg, **kwargs)
                    #hit_targets_list.append(hit_targets)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                #all_hit_targets_list.append(hit_targets_list)
                if enable_bg_obj:
                    for ii, target in enumerate(targets):
                        extra_loss_ce, _ = self.loss_labels(extra_output[ii], [extra_hungarian_targets[ii]], [extra_indices[ii]], num_ent_boxes, loss_name='pred_logits', out_loss_name='loss_ce', ent_det_only_fg=True)
                        losses['loss_ce_{}'.format(i)] += extra_loss_ce['loss_ce']
                        
                        extra_loss_box, _ = self.loss_boxes(extra_output[ii], [extra_hungarian_targets[ii]], [extra_indices[ii]], num_ent_boxes, loss_name='boxes', out_loss_name='loss_boxes')
                        losses['loss_giou_{}'.format(i)] += extra_loss_box['loss_giou']
                        losses['loss_bbox_{}'.format(i)] += extra_loss_box['loss_bbox']
                        
                        
        return losses, (all_hit_targets_list, aux_indicess_list)
    
    
    def pure_obj_hungarian(self, outputs, targets, indices, threshold=0.5):
        """
            pred_boxes: N, nr_boxes, 2 * 4
            mask: N, nr_boxes, 2;    True means bg
        """
        bs, num_queries = outputs["pred_boxes"].shape[:2]
        
        mask = [np.ones(2 * num_queries, dtype=np.bool) for i in range(bs)]
        for idx, (i, j) in enumerate(indices):
            mask[idx][2 * i] = False
            mask[idx][2 * i + 1] = False
        
        extra_output_list = list()
        for idx, target in enumerate(targets):
            pred_boxes = outputs["pred_boxes"][idx].view(num_queries*2, 4)
            pred_logits = outputs["pred_logits"][idx].view(num_queries*2, -1)
            pred_boxes = pred_boxes[mask[idx]].unsqueeze(0)
            pred_logits = pred_logits[mask[idx]].unsqueeze(0)
            pred_boxes = pred_boxes.view(1, -1, 8)
            pred_logits = pred_logits.view(1, pred_boxes.shape[1], -1)
            extra_output = dict(pred_boxes=pred_boxes, pred_logits=pred_logits)
            
            extra_output_list.append(extra_output)

        return extra_output_list
    
    def disentagled_pair_allocate(self, extra_output_list, pair_gt_upper_bound_list, targets):
        extra_hungarian_targets = list()
        for i, target in enumerate(targets):
            hungarian_target = \
                dict(boxes_xyxy=target["boxes_xyxy"],
                    labels=target["labels"],
                    image_size_xyxy=target["image_size_xyxy"], 
                    image_size_xyxy_tgt =target["image_size_xyxy_tgt"],)
            extra_hungarian_targets.append(hungarian_target)
        
        extra_indices = list()
        disentagled_pair_targets = list()
        for i, target in enumerate(extra_hungarian_targets):
            A = pair_gt_upper_bound_list[i]
            A = A[None]
            C, C_max_id = self.matcher(extra_output_list[i], [target], return_C=True, enable_kl_div=self.enable_kl_div)
            
            #mask = ((C < A).sum(1) > 0)
            mask = ((C >= A).sum(1) > 0)
            if C.shape[0] * C.shape[1] <= 0:
                valid_id = np.empty_like(C).astype(np.int32)
            else:
                #valid_id = np.argmin(C, 1)
                valid_id = np.argmax(C, 1)
            
            valid_id = valid_id[mask]
            C_max_id = C_max_id[mask]
            pred_valid_id = np.where(mask)[0]
            
            C_max_id = C_max_id[np.arange(len(valid_id)), valid_id]
            
            pred_valid_id = pred_valid_id * 2 + C_max_id
            
            extra_indices.append(pred_valid_id)
            
        return extra_indices
    
    def pure_loss_bg_label(self, outputs, idx, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # [batch_size, num_queries, num_classes * 2]
        bs, nr_boxes = src_logits.shape[:2]
        
            
        if self.use_focal:
            if src_logits.shape[-1] > self.num_classes:
                src_logits = src_logits.view(nr_boxes * 2, -1)
                src_logits_bg = src_logits[idx]
                label_bg = torch.zeros_like(src_logits_bg)
                class_loss_bg = sigmoid_focal_loss(
                    src_logits_bg, 
                    label_bg,
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="sum",
                ) / num_boxes
                class_loss = class_loss_bg
            else:
                src_logits = src_logits.flatten(0, 1)
                src_logits_bg = src_logits[idx]
                label_bg = torch.zeros_like(src_logits_bg)
                class_loss = sigmoid_focal_loss(
                    src_logits_bg,
                    label_bg,
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="sum",
                ) / num_boxes
            losses = {'loss_ce': class_loss}
        else:
            if src_logits.shape[-1] > self.num_classes:
                src_logits = src_logits.view(nr_boxes * 2, -1)
                src_logits_bg = src_logits[idx]
                label_bg = torch.zeros(len(src_logits_bg)).long()
                loss_ce = F.cross_entropy(src_logits_bg, label_bg, self.empty_weight)
            else:
                src_logits = src_logits.flatten(0, 1)
                src_logits_bg = src_logits[idx]
                label_bg = torch.zeros(len(src_logits_bg)).long()
                loss_ce = F.cross_entropy(src_logits_bg, label_bg, self.empty_weight)
            losses = {'loss_ce': loss_ce}
        return losses
    
    def complement_allocate(self, outputs, targets, indices, use_single_ent=True, enable_multi_matchr=False):
        bs, num_queries = outputs["pred_boxes"].shape[:2]
        mask = [np.ones(2 * num_queries, dtype=np.bool) for i in range(bs)]
        for idx, (i, j) in enumerate(indices):
            mask[idx][2 * i] = False
            mask[idx][2 * i + 1] = False
        
        bg_pair_mask = list()
        extra_hungarian_targets = list()
        extra_indices = list()
        extra_output_list = list()
        for idx, target in enumerate(targets):
            pred_boxes = outputs["pred_boxes"][idx].view(num_queries*2, 4)
            pred_logits = outputs["pred_logits"][idx].view(num_queries*2, -1)
            
            pred_boxes = pred_boxes[mask[idx]].unsqueeze(0)
            pred_logits = pred_logits[mask[idx]].unsqueeze(0)
            
            if not use_single_ent:
                pred_boxes = pred_boxes.view(1, -1, 8)
                pred_logits = pred_logits.view(1, pred_boxes.shape[1], -1)
            
            extra_output = dict(pred_boxes=pred_boxes, pred_logits=pred_logits)
            if "tri_score" in outputs:
                tri_score = outputs["tri_score"][idx].view(num_queries*2, -1)
                tri_score = tri_score[mask[idx]].unsqueeze(0)
                if use_single_ent:
                    tri_score = tri_score.view(1, pred_boxes.shape[1], -1)
                extra_output.update(tri_score=tri_score)
            
            extra_output_list.append(extra_output)
            
            if not use_single_ent:
                hungarian_target = \
                    dict(boxes_xyxy=target["complem_box"],
                        labels=target["complem_label"],
                        image_size_xyxy=target["image_size_xyxy"], 
                        image_size_xyxy_tgt=target["complem_image_size_xyxy_tgt"],)
                one_bs_indices, _, _ = self.matcher(extra_output, [hungarian_target], enable_kl_div=self.enable_kl_div)
            else:
                if enable_multi_matchr:
                    hungarian_target = \
                        dict(boxes_xyxy=target["entity_bbox"],
                            labels=target["entity_label"],
                            image_size_xyxy=target["image_size_xyxy"], 
                            image_size_xyxy_tgt =target["entity_image_size_xyxy_tgt_ent"],)
                
                    one_bs_indices = self.iou_matcher(extra_output, [hungarian_target])
                else:
                    hungarian_target = \
                        dict(boxes_xyxy=target["extra_bbox"],
                            labels=target["extra_entity_label"],
                            image_size_xyxy=target["image_size_xyxy"], 
                            image_size_xyxy_tgt =target["extra_image_size_xyxy_tgt"],)
                    
                    one_bs_indices, _, _ = self.matcher(extra_output, [hungarian_target], enable_kl_div=self.enable_kl_div)
            
            hungarian_target["image_size_xyxy_tgt"] = \
                hungarian_target["image_size_xyxy_tgt"][one_bs_indices[0][1]]
            
            extra_indices.append(one_bs_indices[0])
            extra_hungarian_targets.append(hungarian_target)
            
            s_bg_pair_mask = np.ones((pred_boxes.shape[1], 2), dtype=np.bool)
            s_bg_pair_mask[one_bs_indices[0][0]] = False
            s_bg_pair_mask = s_bg_pair_mask.reshape(pred_boxes.shape[1] * 2)
            
            bg_pair_mask.append(s_bg_pair_mask)
            
        return extra_output_list, extra_indices, extra_hungarian_targets, bg_pair_mask
    
    def iou_matcher(self, outputs, targets, iou_thres=0.85):
        src_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        N, nr_boxes = src_logits.shape[:2]
        
        neg_boxes = pred_boxes
    
        target_boxes = np.concatenate([t['boxes_xyxy'] for t in targets])
        target_boxes = torch.from_numpy(target_boxes).type_as(neg_boxes).to(neg_boxes.device)
        pos_boxes = target_boxes
        
        neg_boxes = neg_boxes.view(-1, 4)
        pos_boxes = pos_boxes.view(-1, 4)
        tot_iou = box_iou(neg_boxes, pos_boxes)[0]
        tot_iou = tot_iou.cpu()
        
        sizes = [len(t['boxes_xyxy']) for t in targets]
            
        sum_sizes = [0, ]
        for i,s in enumerate(sizes):
            sum_sizes.append(sum_sizes[i]+s)
        
        neg_sizes = [nr_boxes for i in range(N)]
        neg_sum_sizes = [0, ]
        for i,s in enumerate(neg_sizes):
            neg_sum_sizes.append(neg_sum_sizes[i]+s)
            
        multi_indices = []
        for i, c in enumerate(tot_iou.split(sizes, -1)):
            A = c[neg_sum_sizes[i]:neg_sum_sizes[i+1]].data.numpy()
            A_score = np.max(A, axis=-1)
            cls_id = np.argmax(A, axis=-1)
            pos_id = np.where(A_score >= iou_thres)[0]
            multi_indices.append((pos_id, cls_id[pos_id]))
        
        return multi_indices
    
    def ranking_loss(self, outputs, targets, indices, num_boxes, so_id=None, loss_name='tri_score', out_loss_name='loss_tri_score', margin=0.1):
        losses, _ = self.loss_labels(outputs, targets, indices, num_boxes, so_id=so_id, \
            loss_name=loss_name, out_loss_name=out_loss_name, ent_det_only_fg=False, ent_bg_type='all')
        #    loss_name=loss_name, out_loss_name=out_loss_name, ent_det_only_fg=True, ent_bg_type='all')   
        return losses, []
            
    def ranking_loss_nms(self, outputs, targets, indices, num_boxes, so_id=None, loss_name='tri_score', out_loss_name='loss_tri_score', thresh=0.5):
        src_logits = outputs['pred_logits']
        rel_pred_logits = outputs['rel_pred_logits']
        pred_boxes = outputs['pred_boxes']
        N, nr_boxes = src_logits.shape[:2]
        
        if self.use_focal:
            score_src_logits = F.sigmoid(src_logits)
            score_rel_pred_logits = F.sigmoid(rel_pred_logits)
        else:
            score_rel_pred_logits = F.softmax(rel_pred_logits, dim=-1)
            score_rel_pred_logits = score_rel_pred_logits[:,:,1:]
            
            score_src_logits = F.softmax(src_logits, dim=-1)
            score_src_logits = score_src_logits.view(N, 2*nr_boxes, -1)
            score_src_logits = score_src_logits[:,:,1:]
            score_src_logits = score_src_logits.view(N, nr_boxes, -1)
            
        loss_count = []
        for idx, (i, j) in enumerate(indices):
            target_boxes = targets[idx]['boxes_xyxy']
            target_labels = targets[idx]['labels']
            target_scores = np.ones_like(target_labels) + 11
            target_rel_labels = targets[idx]['rel_label']
            target_rel_scores = np.ones_like(target_rel_labels) + 11
        
            bg_mask = np.ones(nr_boxes, dtype=np.bool)
            bg_mask[i] = False
            
            pair_score = score_src_logits[idx].data.cpu().numpy()
            sub_score = pair_score[:, :self.num_classes]
            obj_score = pair_score[:, self.num_classes:]
            sub_class = np.argmax(sub_score, -1)
            sub_score = np.max(sub_score, -1)
            obj_class = np.argmax(obj_score, -1)
            obj_score = np.max(obj_score, -1)
            
            rel_score = score_rel_pred_logits[idx].data.cpu().numpy()
            rel_class = np.argmax(rel_score, -1)
            rel_score = np.max(rel_score, -1)
            
            pair_box = pred_boxes[idx].data.cpu().numpy()
            sub_box = pair_box[:, :4]
            obj_box = pair_box[:, 4:]
            
            
            sub_class = np.concatenate([sub_class, target_labels[:, 0]])
            sub_score = np.concatenate([sub_score, target_scores[:, 0]])
            obj_class = np.concatenate([obj_class, target_labels[:, 1]])
            obj_score = np.concatenate([obj_score, target_scores[:, 1]])
            rel_class = np.concatenate([rel_class, target_rel_labels])
            rel_score = np.concatenate([rel_score, target_rel_scores])
            sub_box = np.concatenate([sub_box, target_boxes[:, :4]])
            obj_box = np.concatenate([obj_box, target_boxes[:, 4:]])
            
            tri_score = (rel_score * obj_score * sub_score).reshape(-1)
            tri_score[i] += 10
            sorting_idx = np.argsort(tri_score, axis=0)
            sorting_idx = sorting_idx[::-1]
            
            
            sorting_idx = enable_triple_nms(sub_class, obj_class, \
                rel_class, sub_box, obj_box, sorting_idx, thresh=thresh)
            
            
            sorting_idx = sorting_idx[sorting_idx < nr_boxes]
            
            
            bg_mask[sorting_idx] = False
            labels = torch.zeros_like(src_logits[idx][bg_mask])
            loss = sigmoid_focal_loss(
                src_logits[idx][bg_mask], labels,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            ) / num_boxes
            
            
            loss_count.append(loss)
        
        loss = torch.stack(loss_count, 0)
        loss = loss.sum()
        losses = {out_loss_name: loss}
        return losses, []
    
    def ranking_loss_iou(self, outputs, targets, indices, num_boxes, so_id=None, loss_name='tri_score', out_loss_name='loss_tri_score', margin=0.1, iou_thres=0.4, use_double=True):
        #losses, _ = self.loss_labels(outputs, targets, indices, num_boxes, so_id=so_id, \
        #    loss_name=loss_name, out_loss_name=out_loss_name, ent_det_only_fg=False, ent_bg_type='all')
        
        src_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        N, nr_boxes = src_logits.shape[:2]
        
        idx = self._get_src_permutation_idx(indices)
        
        
        bg_mask = np.ones(src_logits.shape[:2], dtype=np.bool)
        bg_mask[idx] = False
        idx_neg = np.where(bg_mask)
        
        
        neg_boxes = pred_boxes[idx_neg]
        
        #pos_boxes = pred_boxes[idx]
        
        target_boxes = np.concatenate([t['boxes_xyxy'][J] for t, (_, J) in zip(targets, indices)])
        target_boxes = torch.from_numpy(target_boxes).type_as(neg_boxes).to(neg_boxes.device)
        pos_boxes = target_boxes
        
        if use_double:
            pos_boxes = torch.cat([pos_boxes, pred_boxes[idx]]).to(pos_boxes.device)
        
        if src_logits.shape[-1] > self.num_classes:
            tot_iou_s = box_iou(neg_boxes[:, :4], pos_boxes[:, :4])[0]
            tot_iou_o = box_iou(neg_boxes[:, 4:], pos_boxes[:, 4:])[0]
            tot_iou = torch.minimum(tot_iou_s, tot_iou_o)
        else:
            tot_iou = box_iou(neg_boxes, pred_boxes[idx])[0]
        
        tot_iou = tot_iou.cpu()
        

        if use_double:
            sizes = [2*len(i) for i,_ in indices]
        else:
            sizes = [len(i) for i,_ in indices]
            
        sum_sizes = [0, ]
        for i,s in enumerate(sizes):
            sum_sizes.append(sum_sizes[i]+s)
        
        
        neg_sizes = [len(np.where(idx_neg[0] == i)[0]) for i in range(N)]
        neg_sum_sizes = [0, ]
        for i,s in enumerate(neg_sizes):
            neg_sum_sizes.append(neg_sum_sizes[i]+s)
        
        
        select_id, select_gt_class = [], []
        for i, c in enumerate(tot_iou.split(sizes, -1)):
            A = c[neg_sum_sizes[i]:neg_sum_sizes[i+1]].data.numpy()
            relative_neg_id, relative_pos_id = np.where(A > iou_thres)
            select_id.append(relative_neg_id)
            
            _, J = indices[i]
            labels = targets[i]["labels"][J]
            if use_double:
                if src_logits.shape[-1] > self.num_classes:
                    labels = np.tile(labels, (2, 1))
                else:
                    labels = np.tile(labels, (2, ))
            
            labels = labels[relative_pos_id]
            
            select_gt_class.append(labels)
        
        
        ans = []
        if src_logits.shape[-1] > self.num_classes:
            for i, (relative_neg_id, labels) in enumerate(zip(select_id, select_gt_class)):
                label = labels.reshape(-1, 2)
                ##print(src_logits[idx_neg].shape)
                ##print(neg_sum_sizes)
                ##print(relative_neg_id)
                ##print(label)
                
                bg_candi_s = src_logits[idx_neg][neg_sum_sizes[i]:neg_sum_sizes[i+1]][relative_neg_id, label[:, 0]]
                bg_candi_o = src_logits[idx_neg][neg_sum_sizes[i]:neg_sum_sizes[i+1]][relative_neg_id, self.num_classes+label[:, 1]]
                ans.append(bg_candi_s)
                ans.append(bg_candi_o)
                
                #bg_candi_o = src_logits[idx_neg][neg_sum_sizes[i]:neg_sum_sizes[i+1]][relative_neg_id, :]
                #ans.append(bg_candi_o)
        else:
            for i, (relative_neg_id, labels) in enumerate(zip(select_id, select_gt_class)):
                label = labels.reshape(-1)
                bg_candi = src_logits[idx_neg][neg_sum_sizes[i]:neg_sum_sizes[i+1]][relative_neg_id, label]
                ans.append(bg_candi)
        
        src_logits = torch.cat(ans).to(src_logits.device)
        labels = torch.zeros_like(src_logits)
        
        #print(src_logits)
        
        loss = sigmoid_focal_loss(
            src_logits, labels,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        #) / (1. * labels.shape[0] + 1e-6)
        ) / num_boxes
        losses = {out_loss_name: loss}
        
        return losses, []
    
    
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, use_focal: bool = False, rel_w: float = 1., tri_w = 1.):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.rel_w = rel_w
        self.tri_w = tri_w
        if self.use_focal:
            self.focal_loss_alpha = alpha
            self.focal_loss_gamma = gamma
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
    
    @torch.no_grad()
    def forward(self, outputs, targets, return_C=False, basedd_C=None, enable_kl_div=False):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes * 2] with the classification logits
                 # "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 # "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                 
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 8] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes, 2] (s_label, o_label) (where num_target_boxes 
                 # "labels": Tensor of dim [num_target_boxes] (where num_target_boxes 
                           is the number of ground-truth objects in the target) containing the class labels
                 # "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "boxes": Tensor of dim [num_target_boxes, 8] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_boxes"].shape[:2]
        
        # We flatten to compute the cost matrices in a batch
        if self.use_focal:
            if outputs["pred_boxes"].shape[-1] > 5:
                out_prob = outputs["pred_logits"].view(bs, num_queries*2, -1).flatten(0, 1).sigmoid()
            else:
                out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size*num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        else:
            if outputs["pred_boxes"].shape[-1] > 5:
                out_prob = outputs["pred_logits"].view(bs, num_queries*2, -1).flatten(0, 1).softmax(-1)
            else:
                out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1) # [batch_size*num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = np.concatenate([v["labels"] for v in targets])
        if enable_kl_div:
            boxes_xyxy_name = "pure_ent_pred_bboxes_i"
            image_size_xyxy_tgt_name = "pure_ent_image_size_xyxy_tgt"
            X = np.concatenate([v["X"] for v in targets])
            X = torch.from_numpy(X).long().to(out_bbox.device)
            Y = np.concatenate([v["Y"] for v in targets])
            Y = torch.from_numpy(Y).long().to(out_bbox.device)
        else:
            boxes_xyxy_name = "boxes_xyxy"
            image_size_xyxy_tgt_name = "image_size_xyxy_tgt"
            
        if isinstance(targets[0][boxes_xyxy_name], torch.Tensor):
            tgt_bbox = torch.cat([v[boxes_xyxy_name] for v in targets])
        else:
            tgt_bbox = np.concatenate([v[boxes_xyxy_name] for v in targets])
            tgt_bbox = torch.from_numpy(tgt_bbox).type_as(out_bbox).to(out_bbox.device)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal:
            # Compute the classification cost.
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma

            if enable_kl_div:
                tgt_score = torch.cat([v["pure_ent_score_class_i"] for v in targets]) # M, C
                tgt_label_onehot = torch.cat([v["pure_ent_score_dist_label_i"] for v in targets]) # M, C

                distri_score = torch.cat([v["distri_score"] for v in targets])
                
                neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-torch.log(1 - out_prob + 1e-8))
                pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-torch.log(out_prob + 1e-8)) # N, C
                cost_class_s = torch.mm(pos_cost_class[0::2], tgt_label_onehot.t()) - \
                    torch.mm(neg_cost_class[0::2], ((2*tgt_label_onehot-1)*tgt_score).t())
                cost_class_o = torch.mm(pos_cost_class[1::2], tgt_label_onehot.t()) - \
                    torch.mm(neg_cost_class[1::2], ((2*tgt_label_onehot-1)*tgt_score).t())
                cost_class_s = cost_class_s[:, X]
                cost_class_o = cost_class_o[:, Y]

            else:
                neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                if outputs["pred_boxes"].shape[-1] > 5:
                    cost_class_s = pos_cost_class[0::2, tgt_ids[:, 0]] - neg_cost_class[0::2, tgt_ids[:, 0]]
                    cost_class_o = pos_cost_class[1::2, tgt_ids[:, 1]] - neg_cost_class[1::2, tgt_ids[:, 1]]
                    #print('g', cost_class_s)
                else:
                    cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids] # N, M
        else:
            if outputs["pred_boxes"].shape[-1] > 5:
                cost_class_s = -out_prob[0::2, tgt_ids[:, 0]]
                cost_class_o = -out_prob[1::2, tgt_ids[:, 1]]
            else:
                cost_class = -out_prob[:, tgt_ids]
        
        # Compute the L1 cost between boxes
        if isinstance(targets[0]["image_size_xyxy"], torch.Tensor):
            image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
            image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
        else:
            image_size_out = np.concatenate([v["image_size_xyxy"][None] for v in targets])
            image_size_out = np.tile(image_size_out[:, None, :], (1, num_queries, 1)).reshape(-1, 4)
            image_size_out = torch.from_numpy(image_size_out).type_as(out_bbox).to(out_bbox.device)
        
        if isinstance(targets[0][image_size_xyxy_tgt_name], torch.Tensor):
            image_size_tgt = torch.cat([v[image_size_xyxy_tgt_name] for v in targets])
        else:
            image_size_tgt = np.concatenate([v[image_size_xyxy_tgt_name] for v in targets])
            image_size_tgt = torch.from_numpy(image_size_tgt).type_as(out_bbox).to(out_bbox.device)
        
        assert outputs["pred_boxes"].shape[-1] == out_bbox.shape[-1], \
            'outputs["pred_boxes"].shape[-1] != out_bbox.shape[-1]'
        
        if enable_kl_div:
            image_size_out = image_size_out.repeat(1, 2)
        else:
            if out_bbox.shape[-1] > 5:
                image_size_out = image_size_out.repeat(1, 2)
                image_size_tgt = image_size_tgt.repeat(1, 2)
        
        out_bbox_ = out_bbox / image_size_out
        tgt_bbox_ = tgt_bbox / image_size_tgt
        if enable_kl_div:
            cost_bbox_s = torch.cdist(out_bbox_[:, :4], tgt_bbox_, p=1)
            cost_bbox_o = torch.cdist(out_bbox_[:, 4:], tgt_bbox_, p=1)
            cost_bbox_s = cost_bbox_s[:, X]
            cost_bbox_o = cost_bbox_o[:, Y]
        else:
            if out_bbox.shape[-1] > 5:
                cost_bbox_s = torch.cdist(out_bbox_[:, :4], tgt_bbox_[:, :4], p=1)
                cost_bbox_o = torch.cdist(out_bbox_[:, 4:], tgt_bbox_[:, 4:], p=1)
            else:
                cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)
                #cost_bbox = torch.abs(out_bbox_.view(-1, 1, 4) - tgt_bbox_.view(1, -1, 4)).sum(-1)

        # Compute the giou cost betwen boxes
        if enable_kl_div:
            cost_giou_s = -generalized_box_iou(out_bbox[:, :4], tgt_bbox)
            cost_giou_o = -generalized_box_iou(out_bbox[:, 4:], tgt_bbox)
            cost_giou_s = cost_giou_s[:, X]
            cost_giou_o = cost_giou_o[:, Y]
        else:
            if out_bbox.shape[-1] > 5:
                cost_giou_s = -generalized_box_iou(out_bbox[:, :4], tgt_bbox[:, :4])
                cost_giou_o = -generalized_box_iou(out_bbox[:, 4:], tgt_bbox[:, 4:])
            else:
                cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
        
        so_id = None
        # Final cost matrix
        if out_bbox.shape[-1] > 5:
            C_s = self.cost_bbox * cost_bbox_s + self.cost_class * cost_class_s + self.cost_giou * cost_giou_s
            C_o = self.cost_bbox * cost_bbox_o + self.cost_class * cost_class_o + self.cost_giou * cost_giou_o
            
            if basedd_C is not None and len(basedd_C) >= 2:
                C_s = C_s + basedd_C[0]
                C_o = C_o + basedd_C[1]
            
            C = torch.stack([C_s, C_o], 0)
            if tgt_bbox.shape[0] < 1 or tgt_bbox.shape[1] < 1:
                C = torch.sum(C, 0)
                C_max_id = torch.zeros_like(C).long()
            else:
                _, C_max_id = torch.max(C, 0)
                _ *= 2
                C = torch.sum(C, 0)
            
            
            #if C.shape[-1] < 1000:
            #    exam_cost_mat(cost_class_s)
            
        else:
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        
            if basedd_C is not None and len(basedd_C) < 2:
                C = C + basedd_C
        
        C0 = C
        C1 = C
        if out_bbox.shape[-1] > 5:
            if "rel_label" in targets[0]:
                tgt_ids_rel = np.concatenate([v["rel_label"] for v in targets])
                if self.use_focal:
                    out_prob_rel = outputs['rel_pred_logits'].flatten(0, 1).sigmoid()
                    
                    alpha, gamma = self.focal_loss_alpha, self.focal_loss_gamma
                    neg_cost_class_rel = (1 - alpha) * (out_prob_rel ** gamma) * (-(1 - out_prob_rel + 1e-8).log())
                    pos_cost_class_rel = alpha * ((1 - out_prob_rel) ** gamma) * (-(out_prob_rel + 1e-8).log())
                    cost_class_rel = pos_cost_class_rel[:, tgt_ids_rel] - neg_cost_class_rel[:, tgt_ids_rel]
                else:
                    out_prob_rel = outputs['rel_pred_logits'].flatten(0, 1).softmax(-1)
                    cost_class_rel = -out_prob_rel[:, tgt_ids_rel]
                
                C = C + self.rel_w * cost_class_rel
        
        if "tri_score" in outputs:
            out_prob_rel = outputs['tri_score']
            if outputs["pred_boxes"].shape[-1] > 5:
                out_prob_rel = out_prob_rel.view(bs, num_queries*2, -1)
            out_prob_rel = out_prob_rel.flatten(0, 1)
            
            tgt_ids_rel = np.concatenate([v["labels"] for v in targets])
            
            if self.use_focal:
                out_prob_rel = out_prob_rel.sigmoid()
                alpha, gamma = self.focal_loss_alpha, self.focal_loss_gamma
                neg_cost_class_rel = (1 - alpha) * (out_prob_rel ** gamma) * (-(1 - out_prob_rel + 1e-8).log())
                pos_cost_class_rel = alpha * ((1 - out_prob_rel) ** gamma) * (-(out_prob_rel + 1e-8).log())
                
                if outputs["pred_boxes"].shape[-1] > 5:
                    cost_class_rel = pos_cost_class_rel[0::2, tgt_ids_rel[:, 0]] - neg_cost_class_rel[0::2, tgt_ids_rel[:, 0]]
                    cost_class_rel += pos_cost_class_rel[1::2, tgt_ids_rel[:, 1]] - neg_cost_class_rel[1::2, tgt_ids_rel[:, 1]]
                else:
                    cost_class_rel = pos_cost_class_rel[:, tgt_ids_rel] - neg_cost_class_rel[:, tgt_ids_rel]
            else:
                out_prob_rel = out_prob_rel.softmax(-1)
                if outputs["pred_boxes"].shape[-1] > 5:
                    cost_class_rel = -out_prob_rel[::2, tgt_ids_rel[:, 0]]
                    cost_class_rel += -out_prob_rel[1::2, tgt_ids_rel[:, 1]]
                else:
                    cost_class_rel = -out_prob_rel[:, tgt_ids_rel]
                    
            C = C + self.tri_w * cost_class_rel
            
        sizes = [len(v["boxes_xyxy"]) for v in targets] # sum(sizes) == C.shape[-1]
        C = C.view(bs, num_queries, -1).cpu() # batchsize, query_num, gt_num
        
        C0 = C0.view(bs, num_queries, -1).cpu() # batchsize, query_num, gt_num
        
        
        if return_C:
            C = [c[i].data.cpu().numpy() for i, c in enumerate(C.split(sizes, -1))]
            C_max_id = [c[i].data.cpu().numpy() for i, c in enumerate(C_max_id.split(sizes, -1))]
            return C, C_max_id
        
        
        
        if enable_kl_div:
            pre_indices = []
            C1 = C1.view(bs, num_queries, -1)
            for i, c in enumerate(C1.split(sizes, -1)):
                cost_mat = c[i]
                cost_mat, cost_col_id_map = topk_km_imbalance_matsize_torch(cost_mat)
                ans_hung_row_ind, ans_hung_col_ind = linear_sum_assignment(cost_mat)
                pre_indices.append((ans_hung_row_ind, cost_col_id_map[ans_hung_col_ind]))
        else:
            pre_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        if so_id is not None:
            so_id = so_id.bool().view(bs, num_queries, -1).cpu()
            so_id = so_id.split(sizes, -1)
            
        indices = list()
        if so_id is not None:
            so_id_list = list()
        else:
            so_id_list = None
        for i, v in enumerate(pre_indices):
            pred_id, gt_id = v
            permute_id = np.argsort(gt_id)
            indices.append((pred_id[permute_id], gt_id[permute_id]))
            if so_id is not None:
                so_id_list.append(so_id[i][i][pred_id[permute_id], gt_id[permute_id]])
          
        c0_list = [c[i].data.numpy() for i, c in enumerate(C0.split(sizes, -1))]
        gt_upper_bound_list = list()
        for idx,(i,j) in enumerate(indices):
            cc0 = c0_list[idx][i, j]
            gt_upper_bound_list.append(cc0)
        
        #return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        return indices, so_id_list, gt_upper_bound_list
    
def binary_kl_loss(
    inputs,
    targets,
    reduction: str = "none",
):
    p = torch.sigmoid(inputs)
    bkl_loss = F.kl_div(p.log(), targets, reduction="none") + \
        F.kl_div((1-p).log(), 1-targets, reduction="none")
    if reduction == "mean":
        loss = bkl_loss.mean()
    elif reduction == "sum":
        loss = bkl_loss.sum()
    return loss
    

def sigmoid_focal_loss(
    inputs,
    targets,
    alpha = -1,
    gamma = 2,
    reduction: str = "none",
    fg_reweight = None,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    
    if fg_reweight is not None:
        onehot_weight = targets.sum(-1).view(-1, 1)
        onehot_weight = onehot_weight * fg_reweight
        ce_loss = ce_loss * onehot_weight
    
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
    
def loss_format_func(x, y):
    return ((x + y) + 2 * (x - y) ** 2) / 2.

    
def topk_km_imbalance_matsize(cost):
    n, m = cost.shape
    l, r = 2, min(n, m) #max(n//m, m//n)
    s_class = np.argpartition(cost, r)
    s_class = s_class[:, :r]
    ans_id = np.unique(s_class.reshape(-1))
    if m > n:
        while l <= r:
            mid = (l + r) // 2
            s_class = np.argpartition(cost, mid)
            s_class = s_class[:, :mid]
            unique_ans = np.unique(s_class.reshape(-1))
            if len(unique_ans) >= min(n, m):
                r = mid - 1
                ans_id = unique_ans
            else:
                l = mid + 1
    new_cost = cost[:, ans_id]
    return new_cost, ans_id
    
def topk_km_imbalance_matsize_torch(cost):
    n, m = cost.shape
    l, r = 2, min(n, m) #max(n//m, m//n)
    _, s_class = torch.topk(cost, k=r, largest=False, dim=-1)
    ans_id = torch.unique(s_class.view(-1))
    if m > n:
        while l <= r:
            mid = (l + r) // 2
            _, s_class = torch.topk(cost, k=mid, largest=False, dim=-1)
            unique_ans = torch.unique(s_class.view(-1))
            if len(unique_ans) >= min(n, m):
                r = mid - 1
                ans_id = unique_ans
            else:
                l = mid + 1
    new_cost = cost[:, ans_id]
    return new_cost.data.cpu().numpy(), ans_id.data.cpu().numpy()