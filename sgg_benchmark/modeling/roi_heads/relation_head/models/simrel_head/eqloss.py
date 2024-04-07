import json
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from torch.autograd import Variable


VG_REL_PROP = [0.01858, 0.00057, 0.00051, 0.00109, 0.00150, 0.00489, 0.00432, 0.02913, 0.00245, 0.00121, 
               0.00404, 0.00110, 0.00132, 0.00172, 0.00005, 0.00242, 0.00050, 0.00048, 0.00208, 0.15608,
               0.02650, 0.06091, 0.00900, 0.00183, 0.00225, 0.00090, 0.00028, 0.00077, 0.04844, 0.08645,
               0.31621, 0.00088, 0.00301, 0.00042, 0.00186, 0.00100, 0.00027, 0.01012, 0.00010, 0.01286,
               0.00647, 0.00084, 0.01077, 0.00132, 0.00069, 0.00376, 0.00214, 0.11424, 0.01205, 0.02958]
NUM_VG_OBJOBJ_REL_INSTANCES=439063

def lambda_cal(freq_info, num_info, n_c, T=100):
    freq_num_sum = np.sum(num_info)
    ans = 0.00177
    l, r, t = 0.0, 1.0, 0
    while abs(r - l) > 1e-6 and t <= T:
        mid = (l + r) / 2
        s = np.sum((freq_info < mid) * num_info)
        s /= (freq_num_sum + 1e-6)
        if s > 0.02 and s < 0.1:
            ans = mid
        elif s >= 0.1:
            r = mid
        else:
            l = mid
        t += 1
    return ans

def get_counted_freq(counted_num_path):
    with open(counted_num_path, 'r') as f: #(36, )
        num_info = json.load(f)
        f.close()
    num_info = np.array(num_info, dtype=np.float32)
    freq_info = num_info / num_info.sum()
    return freq_info, num_info

def directly_get_counted_freq(data_name):
    if data_name.find('vg') >= 0:
        freq_info = np.array(VG_REL_PROP, dtype=np.float32)
        num_info = NUM_VG_OBJOBJ_REL_INSTANCES * freq_info
        return freq_info, num_info 
    else:
        assert False, 'no such dataset'

def exclude_func(gt_classes, bg_ind=0, n_c=36, n_i=512):
    # instance-level weight
    #bg_ind = n_c
    weight = (gt_classes != bg_ind).float()
    weight = weight.view(n_i, 1).expand(n_i, n_c)
    return weight
    
def threshold_func(pred_class_logits, freq_info, lambda_, n_c=36, n_i=512, is_bce=True):
    # class-level weight
    weight = pred_class_logits.new_zeros(n_c)
    if is_bce:
        weight[freq_info < lambda_] = 1
    else:
        weight[1:][freq_info < lambda_] = 1
    weight = weight.view(1, n_c).expand(n_i, n_c)
    return weight

def eql_loss(cls_score, label_int32, freq_info, lambda_, reduction='sum', is_bce=True, cls_is_sigmoid=True):
    
    if not isinstance(label_int32, torch.Tensor):
        rois_label = Variable(torch.from_numpy(label_int32.astype('int64'))).to(cls_score.device)
    else:
        rois_label = label_int32
    
    
    def expand_label(pred, gt_classes, n_c=36, n_i=512, is_bce=True):
        if is_bce:
            target = pred.new_zeros(n_i, n_c + 1)
            target[torch.arange(n_i), gt_classes] = 1
            #return target[:, :n_c]
            return target[:, 1:]
        else:
            target = pred.new_zeros(n_i, n_c)
            target[torch.arange(n_i), gt_classes] = 1
            return target
    
    if is_bce:
        if cls_is_sigmoid: 
            pred_class_logits = cls_score
        else: 
            pred_class_logits = cls_score[:, 1:]
    else:
        pred_class_logits = cls_score
    n_i, n_c = pred_class_logits.size()    
        
    target = expand_label(pred_class_logits, rois_label, n_c, n_i, is_bce)

    eql_w = 1 - exclude_func(rois_label, 0, n_c, n_i) * \
            threshold_func(pred_class_logits, freq_info, lambda_, n_c, n_i, is_bce) * \
            (1 - target)
    
    if is_bce:
        cls_loss = F.binary_cross_entropy_with_logits(pred_class_logits, target,
                                                  reduction='none')
        if reduction == 'sum':
            cls_loss = torch.sum(cls_loss * eql_w) / n_i
            #cls_loss = torch.sum(cls_loss * eql_w)
        elif reduction == 'none':
            cls_loss = cls_loss * eql_w
    else:
        exp_pred_class_logits = pred_class_logits.exp()
        cls_loss = pred_class_logits - \
                    ((exp_pred_class_logits * eql_w).sum(dim=1, keepdim=True) + 1e-6).log()
        if reduction == 'sum':
            cls_loss = -torch.sum(cls_loss * target) / n_i
            #cls_loss = -torch.sum(cls_loss * target)
        elif reduction == 'none':
            cls_loss = -cls_loss * eql_w
    
    return cls_loss