import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class FrequencyBias_GCL(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, cfg, statistics, eps=1e-3, predicate_all_list=None):
        super(FrequencyBias_GCL, self).__init__()
        assert predicate_all_list is not None
        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        # self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = max(predicate_all_list) + 1
        old_matrix = statistics['fg_matrix'].float()

        fg_matrix = torch.zeros([self.num_obj_cls, self.num_obj_cls, self.num_rel_cls],
                                dtype=old_matrix.dtype, device=old_matrix.device)

        lines = 0
        assert len(predicate_all_list) == 51 or len(predicate_all_list) == 101
        for i in range(len(predicate_all_list)):
            if i == 0 or predicate_all_list[i] > 0:
                fg_matrix[:, :, lines] = old_matrix[:, :, i]
                lines = lines + 1

        assert lines == self.num_rel_cls

        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        # pred_dist = statistics['pred_dist'].float()
        assert pred_dist.size(0) == pred_dist.size(1)

        self.num_objs = pred_dist.size(0)
        self.num_rels = pred_dist.size(2)
        pred_dist = pred_dist.view(-1, self.num_rels)

        self.obj_baseline = nn.Embedding(self.num_objs * self.num_objs, self.num_rels)
        with torch.no_grad():
            self.obj_baseline.weight.copy_(pred_dist, non_blocking=True)

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2]
        :return:
        """
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def index_with_probability(self, pair_prob):
        """
        :param labels: [batch_size, num_obj, 2]
        :return:
        """
        batch_size, num_obj, _ = pair_prob.shape

        joint_prob = pair_prob[:, :, 0].contiguous().view(batch_size, num_obj, 1) * pair_prob[:, :,
                                                                                    1].contiguous().view(batch_size, 1,
                                                                                                         num_obj)

        return joint_prob.view(batch_size, num_obj * num_obj) @ self.obj_baseline.weight

    def forward(self, labels):
        # implement through index_with_labels
        return self.index_with_labels(labels)

def KL_divergence(logits_p, logits_q, reduce=True):
    # p = softmax(logits_p)
    # q = softmax(logits_q)
    # KL(p||q)
    # suppose that p/q is in shape of [bs, num_classes]

    p = F.softmax(logits_p, dim=1)
    q = F.softmax(logits_q, dim=1)

    shape = list(p.size())
    _shape = list(q.size())
    assert shape == _shape
    #print(shape)
    num_classes = shape[1]
    epsilon = 1e-8
    _p = (p + epsilon * Variable(torch.ones(*shape).cuda())) / (1.0 + num_classes * epsilon)
    _q = (q + epsilon * Variable(torch.ones(*shape).cuda())) / (1.0 + num_classes * epsilon)
    if reduce:
        return torch.mean(torch.sum(_p * torch.log(_p / _q), 1))
    else:
        return torch.sum(_p * torch.log(_p / _q), 1)

def generate_current_predicate_set(incremental_stage_list, current_training_stage):
    outp = []
    formerp = []
    current_chosen_vector = []
    former_chosen_vector = []
    for i in range(current_training_stage + 1):
        outp.extend(incremental_stage_list[i])
    for i in range(current_training_stage):
        formerp.extend(incremental_stage_list[i])
    for i in range(len(outp)+1):
        if i in incremental_stage_list[current_training_stage]:
            current_chosen_vector.append(1)
        else:
            current_chosen_vector.append(0)
    for i in range(len(outp)+1):
        if i in formerp:
            former_chosen_vector.append(1)
        else:
            former_chosen_vector.append(0)
    num_stage_vector = []
    n_p = 0
    for isl in incremental_stage_list:
        n_p += len(isl)
        num_stage_vector.append(n_p)

    return outp, formerp, current_chosen_vector, former_chosen_vector, num_stage_vector

def generate_num_stage_vector(incremental_stage_list):
    num_stage_vector = []
    n_p = 0
    for isl in incremental_stage_list:
        n_p += len(isl)
        num_stage_vector.append(n_p)

    return num_stage_vector

def get_current_predicate_idx(incremental_stage_list, zeros_vector_penalty, num_predicate):
    data_long = num_predicate

    outp = []
    for i in range(data_long):
        outp.append(0)
    for i in range(len(incremental_stage_list)):
        for num in incremental_stage_list[i]:
            outp[num] = i+1
    max_p = []
    for i in incremental_stage_list:
        max_p.append(max(i))

    idx_search_p = []
    kd_p = []
    for i in range(len(incremental_stage_list)):
        p1 = []
        p2 = []
        for j in range(data_long):
            p1.append(0)
            p2.append(zeros_vector_penalty)
        max_l = max_p[i]
        for j in range(max_l):
            p1[j+1] = j+1
            p2[j+1] = 1.0
        idx_search_p.append(p1)
        kd_p.append(p2)

    return outp, max_p, idx_search_p, kd_p

def generate_onehot_vector(incremental_stage_list, current_training_stage, num_class):
    data_long = num_class

    one_hot_vector = []
    if current_training_stage == -1:
        one_hot_vector.append(0)
        for i in range(data_long-1):
            one_hot_vector.append(1)
        return one_hot_vector
    for i in range(data_long):
        one_hot_vector.append(0)
    for i in range(current_training_stage+1):
        if i+1 == current_training_stage:
            for idx in incremental_stage_list[i]:
                if idx != 1 and idx != 2:
                    one_hot_vector[idx] = 1
                else:
                    one_hot_vector[idx] = -1
        elif i == current_training_stage:
            for idx in incremental_stage_list[i]:
                one_hot_vector[idx] = 1
        else:
            for idx in incremental_stage_list[i]:
                one_hot_vector[idx] = -1

    return one_hot_vector

def generate_sample_rate_vector(num_stage_predicate, predicate_new_order_count):
    outp = []
    for i in range(len(num_stage_predicate)):
        opiece = []
        for j in range(len(predicate_new_order_count)):
            opiece.append(0.0)
        num_list = predicate_new_order_count[0:(num_stage_predicate[i]+1)]
        median = np.median(num_list[1:])
        for j in range(len(num_list)):
            if num_list[j] > median:
                num = median / num_list[j]
                if j == 0:
                    num = num * 10.0
                if num < 0.01:
                    num = 0.01
                opiece[j] = num
            else:
                opiece[j] = 1.0
        outp.append(opiece)
    return outp

def generate_current_group_sequence_for_bias(current_set, num_class):
    data_long = num_class

    outp = []
    for i in range(data_long):
        outp.append(0)
    for i in current_set:
        outp[i] = i
    return outp

def generate_current_sequence_for_bias(incremental_stage_list, num_class):
    data_long = num_class

    outp = []
    for i in range(len(incremental_stage_list)):
        opiece = []
        for j in range(data_long):
            opiece.append(0)
        for j in range(i+1):
            for k in incremental_stage_list[j]:
                opiece[k] = k
        outp.append(opiece)

    return outp