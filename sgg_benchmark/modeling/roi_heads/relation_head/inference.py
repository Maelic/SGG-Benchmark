# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from sgg_benchmark.structures.bounding_box import BoxList
from sgg_benchmark.modeling.box_coder import BoxCoder
from .models.utils.utils_relation import obj_prediction_nms

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        attribute_on,
        use_gt_box=False,
        proposals_as_gt=False,
        later_nms_pred_thres=0.3,
    ):
        """
        Arguments:

        """
        super(PostProcessor, self).__init__()
        self.attribute_on = attribute_on
        self.use_gt_box = use_gt_box
        self.proposals_as_gt = proposals_as_gt
        self.later_nms_pred_thres = later_nms_pred_thres

    def forward(self, x, rel_pair_idxs, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the relation logits
                and finetuned object logits from the relation model.
            rel_pair_idxs （list[tensor]): subject and object indice of each relation,
                the size of tensor is (num_rel, 2)
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for each image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        if not self.proposals_as_gt:
            relation_logits, refine_logits = x
            finetune_obj_logits = refine_logits
        else:
            relation_logits = x
        
        if self.attribute_on:
            if isinstance(refine_logits[0], (list, tuple)):
                finetune_obj_logits, finetune_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attribute_on = False
                finetune_obj_logits = refine_logits

        results = []
        if not self.proposals_as_gt:
            it_dict = zip(relation_logits, finetune_obj_logits, rel_pair_idxs, boxes)
        else:
            it_dict = zip(relation_logits, rel_pair_idxs, boxes)

        for i, current_it in enumerate(it_dict):
            if self.attribute_on:
                att_logit = finetune_att_logits[i]
                att_prob = torch.sigmoid(att_logit)
        
            if not self.proposals_as_gt:
                rel_logit, obj_logit, rel_pair_idx, box = current_it
                obj_class_prob = F.softmax(obj_logit, -1)
                obj_class_prob[:, 0] = 0  # set background score to 0
                num_obj_bbox = obj_class_prob.shape[0]
                num_obj_class = obj_class_prob.shape[1]
                if self.use_gt_box:
                    obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
                    obj_pred = obj_pred + 1
                else:
                    # NOTE: by kaihua, apply late nms for object prediction
                    obj_pred = obj_prediction_nms(box.get_field('boxes_per_cls'), obj_logit, self.later_nms_pred_thres)
                    obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                    obj_scores = obj_class_prob.view(-1)[obj_score_ind]
                assert obj_scores.shape[0] == num_obj_bbox
            else:
                rel_logit, rel_pair_idx, box = current_it
                obj_scores = box.get_field('pred_scores')
                obj_pred = box.get_field('pred_labels')
                obj_pred = obj_pred + 1
           
            obj_class = obj_pred

            if self.use_gt_box or self.proposals_as_gt:
                boxlist = box
            else:
                # mode==sgdet
                # apply regression based on finetuned object class
                device = obj_class.device
                batch_size = obj_class.shape[0]
                regressed_box_idxs = obj_class
                boxlist = BoxList(box.get_field('boxes_per_cls')[torch.arange(batch_size, device=device), regressed_box_idxs], box.size, 'xyxy')
            boxlist.add_field('pred_labels', obj_class) # (#obj, )
            boxlist.add_field('pred_scores', obj_scores) # (#obj, )

            if self.attribute_on:
                boxlist.add_field('pred_attributes', att_prob)
            
            # sorting triples according to score production
            obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
            rel_class_prob = F.softmax(rel_logit, -1)
            rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
            rel_class = rel_class + 1
            # TODO Kaihua: how about using weighted some here?  e.g. rel*1 + obj *0.8 + obj*0.8
            triple_scores = rel_scores * obj_scores0 * obj_scores1
            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
            rel_pair_idx = rel_pair_idx[sorting_idx]
            rel_class_prob = rel_class_prob[sorting_idx]
            rel_labels = rel_class[sorting_idx]

            boxlist.add_field('rel_pair_idxs', rel_pair_idx) # (#rel, 2)
            boxlist.add_field('pred_rel_scores', rel_class_prob) # (#rel, #rel_class)
            boxlist.add_field('pred_rel_labels', rel_labels) # (#rel, )
            # should have fields : rel_pair_idxs, pred_rel_class_prob, pred_rel_labels, pred_labels, pred_scores
            # Note
            # TODO Kaihua: add a new type of element, which can have different length with boxlist (similar to field, except that once 
            # the boxlist has such an element, the slicing operation should be forbidden.)
            # it is not safe to add fields about relation into boxlist!
            results.append(boxlist)
        return results
    


class HierarchPostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
            self,
            attribute_on,
            use_gt_box=False,
            later_nms_pred_thres=0.3,
    ):
        """
        Arguments:

        """
        super(HierarchPostProcessor, self).__init__()
        self.attribute_on = attribute_on
        self.use_gt_box = use_gt_box
        self.later_nms_pred_thres = later_nms_pred_thres
        self.geo_label = [1, 2, 3, 4, 5, 6, 8, 10, 22, 23, 29, 31, 32, 33, 43]
        self.pos_label = [9, 16, 17, 20, 27, 30, 36, 42, 48, 49, 50]
        self.sem_label = [7, 11, 12, 13, 14, 15, 18, 19, 21, 24, 25, 26, 28, 34, 35, 37, 38, 39, 40, 41, 44, 45, 46, 47]
        self.geo_label_tensor = torch.tensor(self.geo_label)
        self.pos_label_tensor = torch.tensor(self.pos_label)
        self.sem_label_tensor = torch.tensor(self.sem_label)

    def forward(self, x, rel_pair_idxs, boxes):
        """
        Arguments:
            x: rel1_prob, rel2_prob, rel3_prob, super_rel_prob, refine_logits
            rel_pair_idxs （list[tensor]): subject and object indice of each relation,
                the size of tensor is (num_rel, 2)
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        relations_probs, refine_logits = x
        rel1_probs, rel2_probs, rel3_probs, super_rel_probs = relations_probs

        # Assume no attr
        finetune_obj_logits = refine_logits

        results = []
        for i, (rel1_prob, rel2_prob, rel3_prob, super_rel_prob, obj_logit,
                rel_pair_idx, box) in enumerate(zip(
                rel1_probs, rel2_probs, rel3_probs, super_rel_probs,
                finetune_obj_logits, rel_pair_idxs, boxes
        )):
            # i: index of image
            obj_class_prob = F.softmax(obj_logit, -1)
            obj_class_prob[:, 0] = 0  # set background score to 0
            num_obj_bbox = obj_class_prob.shape[0]
            num_obj_class = obj_class_prob.shape[1]

            if self.use_gt_box:
                obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
                obj_pred = obj_pred + 1
            else:
                # NOTE: by kaihua, apply late nms for object prediction
                obj_pred = obj_prediction_nms(box.get_field('boxes_per_cls'),
                                              obj_logit,
                                              self.later_nms_pred_thres)
                obj_score_ind = torch.arange(num_obj_bbox,
                                             device=obj_logit.device) * num_obj_class + obj_pred
                obj_scores = obj_class_prob.view(-1)[obj_score_ind]

            assert obj_scores.shape[0] == num_obj_bbox
            obj_class = obj_pred
            device = obj_class.device

            if self.use_gt_box:
                boxlist = box
            else:
                # mode==sgdet
                # apply regression based on finetuned object class
                batch_size = obj_class.shape[0]
                regressed_box_idxs = obj_class
                boxlist = BoxList(
                    box.get_field('boxes_per_cls')[torch.arange(batch_size,
                                                                device=device), regressed_box_idxs],
                    box.size, 'xyxy')
            boxlist.add_field('pred_labels', obj_class)  # (#obj, )
            boxlist.add_field('pred_scores', obj_scores)  # (#obj, )

            # sorting triples according to score production
            obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
            
            self.geo_label_tensor = self.geo_label_tensor.to(device)
            self.pos_label_tensor = self.pos_label_tensor.to(device)
            self.sem_label_tensor = self.sem_label_tensor.to(device)
            rel1_prob = torch.exp(rel1_prob)
            rel2_prob = torch.exp(rel2_prob)
            rel3_prob = torch.exp(rel3_prob)

            # For Bayesian classification head, we predict three edges for one pair(each edge for one super category),
            # then gather all the predictions for ranking.
            rel1_scores, rel1_class = rel1_prob.max(dim=1)
            rel1_class = self.geo_label_tensor[rel1_class]
            rel2_scores, rel2_class = rel2_prob.max(dim=1)
            rel2_class = self.pos_label_tensor[rel2_class]
            rel3_scores, rel3_class = rel3_prob.max(dim=1)
            rel3_class = self.sem_label_tensor[rel3_class]

            cat_class_prob = torch.cat((rel1_prob, rel2_prob, rel3_prob), dim=1)
            cat_class_prob = torch.cat((cat_class_prob, cat_class_prob, cat_class_prob), dim=0)
            cat_rel_pair_idx = torch.cat((rel_pair_idx, rel_pair_idx, rel_pair_idx), dim=0)
            cat_obj_score0 = torch.cat((obj_scores0, obj_scores0, obj_scores0), dim=0)
            cat_obj_score1 = torch.cat((obj_scores1, obj_scores1, obj_scores1), dim=0)
            cat_labels = torch.cat((rel1_class, rel2_class, rel3_class), dim=0)
            cat_scores = torch.cat((rel1_scores, rel2_scores, rel3_scores), dim=0)

            triple_scores = cat_scores * cat_obj_score0 * cat_obj_score1
            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
            rel_pair_idx = cat_rel_pair_idx[sorting_idx]
            rel_class_prob = cat_class_prob[sorting_idx]
            rel_labels = cat_labels[sorting_idx]

            #############################################################
            # # query llm about top k triplets for commonsense validation
            # llm_responses = self.llm.query(rel_pair_idx[:self.llm.top_k, :], rel_labels[:self.llm.top_k])
            # rel_class_prob[:self.llm.top_k, :][llm_responses == -1] = -math.inf
            #
            # # resort the triplets
            # _, sorting_idx = torch.sort(rel_class_prob, dim=0, descending=True)
            # rel_pair_idx = rel_pair_idx[sorting_idx]
            # rel_class_prob = rel_class_prob[sorting_idx]
            # rel_labels = rel_labels[sorting_idx]
            #############################################################

            boxlist.add_field('rel_pair_idxs', rel_pair_idx)  # (#rel, 2)
            boxlist.add_field('pred_rel_scores', rel_class_prob)  # (#rel, #rel_class)
            boxlist.add_field('pred_rel_labels', rel_labels)  # (#rel, )

            # should have fields : rel_pair_idxs, pred_rel_class_prob, pred_rel_labels, pred_labels, pred_scores
            results.append(boxlist)
        return results


def make_roi_relation_post_processor(cfg):
    attribute_on = cfg.MODEL.ATTRIBUTE_ON
    use_gt_box = cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX
    proposals_as_gt = cfg.MODEL.BACKBONE.FREEZE
    later_nms_pred_thres = cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

    if "Hierarchical" in cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR:
        postprocessor = HierarchPostProcessor(
            attribute_on,
            use_gt_box,
            later_nms_pred_thres,
        )
    else:
        postprocessor = PostProcessor(
            attribute_on,
            use_gt_box,
            proposals_as_gt,
            later_nms_pred_thres,
        )
    return postprocessor
