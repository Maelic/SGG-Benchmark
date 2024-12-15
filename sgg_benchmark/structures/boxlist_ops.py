# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import scipy.linalg

from sgg_benchmark.layers import nms as _box_nms

def squeeze_tensor(tensor):
    tensor = torch.squeeze(tensor)
    try:
        len(tensor)
    except TypeError:
        tensor.unsqueeze_(0)
    return tensor

def boxlist_nms(boxes, scores, nms_thresh, max_proposals=-1):
    """
    Performs non-maximum suppression on a set of boxes, with scores specified
    in a tensor.

    Arguments:
        boxes (Tensor)
        scores (Tensor)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
    """
    if nms_thresh <= 0:
        return boxes, scores
    keep = _box_nms(boxes, scores, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxes = boxes[keep]
    scores = scores[keep]
    return boxes, scores


def remove_small_boxes(boxes, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxes (Tensor)
        min_size (int)
    """
    _, _, ws, hs = boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxes[keep]

def boxlist_iou(boxlist1, boxlist2):
    return compute_iou(boxlist1[:, :4], boxlist2[:, :4])

def compute_iou(boxes1, boxes2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      boxes1: (Tensor) bounding boxes, sized [N,4].
      boxes2: (Tensor) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].
    """

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    iou = inter / (area1[:, None] + area2 - inter)

    return iou


def boxlist_union(boxes1, boxes2):
    """
    Compute the union region of two set of boxes

    Arguments:
      boxes1: (Tensor) bounding boxes, sized [N,4].
      boxes2: (Tensor) bounding boxes, sized [N,4].

    Returns:
      (Tensor) union, sized [N,4].
    """
    union_box = torch.cat((
        torch.min(boxes1[:,:2], boxes2[:,:2]),
        torch.max(boxes1[:,2:], boxes2[:,2:])
        ),dim=1)
    return union_box

def boxlist_intersection(boxes1, boxes2):
    """
    Compute the intersection region of two set of boxes

    Arguments:
      boxes1: (Tensor) bounding boxes, sized [N,4].
      boxes2: (Tensor) bounding boxes, sized [N,4].

    Returns:
      (Tensor) intersection, sized [N,4].
    """
    inter_box = torch.cat((
        torch.max(boxes1[:,:2], boxes2[:,:2]),
        torch.min(boxes1[:,2:], boxes2[:,2:])
        ),dim=1)
    invalid_bbox = torch.max((inter_box[:,0] >= inter_box[:,2]).long(), (inter_box[:,1] >= inter_box[:,3]).long())
    inter_box[invalid_bbox > 0] = 0
    return inter_box


def cat_boxlist(boxes_list):
    """
    Concatenates a list of boxes (having the same image size) into a
    single tensor

    Arguments:
        boxes_list (list[Tensor])
    """
    return torch.cat(boxes_list, dim=0)

def resize_boxes(boxes, size, orig_size):
    """
    Resize boxes to the new image size

    Arguments:
        boxes (Tensor)
        size (tuple)
        orig_size (tuple)
    """
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) / o
        for s, o in zip(size, orig_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)