import torch
from sgg_benchmark.structures.bounding_box import BoxList
from sgg_benchmark.structures.boxlist_ops import cat_boxlist

def add_gt_proposals(proposals, targets):
    """
    Arguments:
        proposals: list[BoxList]
        targets: list[BoxList]
    """

    gt_boxes = [target.copy_with_fields(["labels"]) for target in targets]

    proposals = [
        cat_boxlist((proposal, gt_box))
        for proposal, gt_box in zip(proposals, gt_boxes)
    ]

    return proposals