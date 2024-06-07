import array
import os
import zipfile
import six
import torch
import numpy as np
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm
import sys
from sgg_benchmark.modeling.utils import cat

import clip

def normalize_sigmoid_logits(orig_logits):
    orig_logits = torch.sigmoid(orig_logits)
    orig_logits = orig_logits / (orig_logits.sum(1).unsqueeze(-1) + 1e-12)
    return orig_logits

def generate_attributes_target(attributes, device, max_num_attri, num_attri_cat):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert max_num_attri == attributes.shape[1]
        num_obj = attributes.shape[0]

        with_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, num_attri_cat), device=device).float()

        for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
            for k in range(max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, with_attri_idx

def transpose_packed_sequence_inds(lengths):
    """
    Get a TxB indices from sorted lengths. 
    Fetch new_inds, split by new_lens, padding to max(new_lens), and stack.
    Returns:
        new_inds (np.array) [sum(lengths), ]
        new_lens (list(np.array)): number of elements of each time step, descending
    """
    new_inds = []
    new_lens = []
    cum_add = np.cumsum([0] + lengths)
    max_len = lengths[0]
    length_pointer = len(lengths) - 1
    for i in range(max_len):
        while length_pointer > 0 and lengths[length_pointer] <= i:
            length_pointer -= 1
        new_inds.append(cum_add[:(length_pointer+1)].copy())
        cum_add[:(length_pointer+1)] += 1
        new_lens.append(length_pointer+1)
    new_inds = np.concatenate(new_inds, 0)
    return new_inds, new_lens


def sort_by_score(proposals, scores):
    """
    We'll sort everything scorewise from Hi->low, BUT we need to keep images together
    and sort LSTM from l
    :param im_inds: Which im we're on
    :param scores: Goodness ranging between [0, 1]. Higher numbers come FIRST
    :return: Permutation to put everything in the right order for the LSTM
             Inverse permutation
             Lengths for the TxB packed sequence.
    """
    num_rois = [len(b) for b in proposals]
    num_im = len(num_rois)

    scores = scores.split(num_rois, dim=0)
    ordered_scores = []
    for i, (score, num_roi) in enumerate(zip(scores, num_rois)):
        ordered_scores.append( score + 2.0 * float(num_roi * 2 * num_im + i) )
    ordered_scores = cat(ordered_scores, dim=0)
    _, perm = torch.sort(ordered_scores, 0, descending=True)

    num_rois = sorted(num_rois, reverse=True)
    inds, ls_transposed = transpose_packed_sequence_inds(num_rois)  # move it to TxB form
    inds = torch.LongTensor(inds).to(scores[0].device)
    ls_transposed = torch.LongTensor(ls_transposed)
    
    perm = perm[inds] # (batch_num_box, )
    _, inv_perm = torch.sort(perm)

    return perm, inv_perm, ls_transposed

def to_onehot(vec, num_classes, fill=1000):
    """
    Creates a [size, num_classes] torch FloatTensor where
    one_hot[i, vec[i]] = fill
    
    :param vec: 1d torch tensor
    :param num_classes: int
    :param fill: value that we want + and - things to be.
    :return: 
    """
    onehot_result = vec.new(vec.size(0), num_classes).float().fill_(-fill)
    arange_inds = torch.arange(0, vec.size(0), device=vec.device).long()

    onehot_result.view(-1)[vec.long() + num_classes*arange_inds] = fill
    return onehot_result


def get_dropout_mask(dropout_probability, tensor_shape, device):
    """
    once get, it is fixed all the time
    """
    binary_mask = (torch.rand(tensor_shape) > dropout_probability)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().to(device).div(1.0 - dropout_probability)
    return dropout_mask

def center_x(proposals):
    assert proposals[0].mode == 'xyxy'
    boxes = cat([p.bbox for p in proposals], dim=0)
    c_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    return c_x.view(-1)
    
def encode_box_info(proposals):
    """
    encode proposed box information (x1, y1, x2, y2) to 
    (cx/wid, cy/hei, w/wid, h/hei, x1/wid, y1/hei, x2/wid, y2/hei, wh/wid*hei)
    """
    assert proposals[0].mode == 'xyxy'
    boxes_info = []
    for proposal in proposals:
        boxes = proposal.bbox
        img_size = proposal.size
        wid = img_size[0]
        hei = img_size[1]
        wh = boxes[:, 2:] - boxes[:, :2] + 1.0
        xy = boxes[:, :2] + 0.5 * wh
        w, h = wh.split([1,1], dim=-1)
        x, y = xy.split([1,1], dim=-1)
        x1, y1, x2, y2 = boxes.split([1,1,1,1], dim=-1)
        # assert wid * hei != 0
        info = torch.cat([w/wid, h/hei, x/wid, y/hei, x1/wid, y1/hei, x2/wid, y2/hei,
                          w*h/(wid*hei)], dim=-1).view(-1, 9)
        boxes_info.append(info)

    return torch.cat(boxes_info, dim=0)