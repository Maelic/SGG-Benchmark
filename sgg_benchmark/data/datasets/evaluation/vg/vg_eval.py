import os
import torch
import numpy as np
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from sgg_benchmark.data.datasets.evaluation.vg.sgg_eval import *
from sgg_benchmark.config.paths_catalog import DatasetCatalog


def do_vg_evaluation(
        cfg,
        dataset,
        dataset_name,
        predictions,
        output_folder,
        logger,
        iou_types,
        informative=False,
):
    # Control which metric to evaluate, recall need to be here BY DEFAULT
    metrics_to_eval = {'relations': ['recall', 'mean_recall', 'f1_score'], 'bbox': ['mAP']}

    if cfg.TEST.INFORMATIVE:
        metrics_to_eval['relations'].extend(['informative_recall'])

    metrics_map = {'recall': SGRecall, 'recall_nogc': SGNoGraphConstraintRecall, 'zeroshot_recall': SGZeroShotRecall,
                   'ng_zeroshot_recall': SGNGZeroShotRecall, 'informative_recall': SGInformativeRecall,
                   'mean_recall': SGMeanRecall, 'recall_relative': SGRecallRelative,
                   'mean_recall_relative': SGMeanRecallRelative, 'f1_score': SGF1Score,
                   'weighted_recall': SGWeightedRecall, 'weighted_mean_recall': SGWeightedMeanRecall}

    metrics_to_eval = {k: v for k, v in metrics_map.items() if k in metrics_to_eval['relations']}

    # get zeroshot triplet
    if "relations" in iou_types:
        data_dir = DatasetCatalog.DATA_DIR
        try:
            zero_shot_file = DatasetCatalog.DATASETS[dataset_name]['zeroshot_file']
            zeroshot_triplet = torch.load(os.path.join(data_dir, zero_shot_file),
                                          map_location=torch.device("cpu")).long().numpy()
        except KeyError:
            zeroshot_triplet = np.array([])

    # attribute_on = cfg.MODEL.ATTRIBUTE_ON
    # num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
    # extract evaluation settings from cfg
    # mode = cfg.TEST.RELATION.EVAL_MODE
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and not cfg.SGDET_TEST:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            mode = 'predcls'
        else:
            mode = 'sgcls'
    else:
        mode = 'sgdet'

    num_rel_category = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
    multiple_preds = cfg.TEST.RELATION.MULTIPLE_PREDS
    iou_thres = cfg.TEST.RELATION.IOU_THRESHOLD
    assert mode in {'predcls', 'sgdet', 'sgcls', 'phrdet', 'preddet'}

    groundtruths = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        # recover original size which is before transform
        predictions[image_id] = prediction.resize((image_width, image_height))

        gt = dataset.get_groundtruth(image_id, evaluation=True)
        groundtruths.append(gt)

    if informative and not 'informative_rels' in groundtruths[0].extra_fields.keys():
        logger.info(
            'Dataset does not have informative_rels, skipping informative evaluation for dataset: %s' % dataset_name)
        informative = False

    result_str = '\n' + '=' * 100 + '\n'
    if "bbox" in iou_types:
        # create a Coco-like object that we can use to evaluate detection!
        anns = []
        for image_id, gt in enumerate(groundtruths):
            labels = gt.get_field('labels').tolist() # integer
            boxes = gt.bbox.tolist() # xyxy
            for cls, box in zip(labels, boxes):
                anns.append({
                    'area': (box[3] - box[1]) * (box[2] - box[0]),
                    'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]], # xywh
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': image_id,
                    'iscrowd': 0,
                })
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'use coco script for vg detection evaluation'},
            'images': [{'id': i} for i in range(len(groundtruths))],
            'categories': [
                {'supercategory': 'person', 'id': i, 'name': name}
                for i, name in enumerate(dataset.ind_to_classes) if name != '__background__'
                ],
            'annotations': anns,
        }
        fauxcoco.createIndex()

        # format predictions to coco-like
        cocolike_predictions = []
        for image_id, prediction in enumerate(predictions):
            box = prediction.convert('xywh').bbox.detach().cpu().numpy() # xywh
            score = prediction.get_field('pred_scores').detach().cpu().numpy() # (#objs,)
            label = prediction.get_field('pred_labels').detach().cpu().numpy() # (#objs,)
            # for predcls, we set label and score to groundtruth
            if mode == 'predcls':
                label = prediction.get_field('labels').detach().cpu().numpy()
                score = np.ones(label.shape[0])
                assert len(label) == len(box)
            image_id = np.asarray([image_id]*len(box))
            cocolike_predictions.append(
                np.column_stack((image_id, box, score, label))
                )
            # logger.info(cocolike_predictions)
        cocolike_predictions = np.concatenate(cocolike_predictions, 0)
        # evaluate via coco API
        res = fauxcoco.loadRes(cocolike_predictions)
        coco_eval = COCOeval(fauxcoco, res, 'bbox')
        coco_eval.params.imgIds = list(range(len(groundtruths)))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAp = coco_eval.stats[1]

        result_str += 'Detection evaluation mAp=%.4f\n' % mAp
        result_str += '=' * 100 + '\n'

    if "relations" in iou_types:
        result_dict = {}
        evaluator = {}

        for k, v in metrics_to_eval.items():
            if "mean" in k:
                cur_metric = v(result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
            else:
                cur_metric = v(result_dict)
            cur_metric.register_container(mode)
            evaluator["eval_" + k] = cur_metric

        # prepare all inputs
        global_container = {}
        global_container['zeroshot_triplet'] = zeroshot_triplet
        global_container['result_dict'] = result_dict
        global_container['mode'] = mode
        global_container['multiple_preds'] = multiple_preds
        global_container['num_rel_category'] = num_rel_category
        global_container['iou_thres'] = iou_thres
        # global_container['attribute_on'] = attribute_on
        # global_container['num_attributes'] = num_attributes

        if informative:
            stats = dataset.get_statistics()
            global_container['ind_to_predicates'] = stats['rel_classes']
            global_container['ind_to_classes'] = stats['obj_classes']

        for groundtruth, prediction in tqdm(zip(groundtruths, predictions), desc='Evaluating', total=len(groundtruths),
                                            disable=not (informative)):
            evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator,
                                           informative=informative)

        for k, v in evaluator.items():
            if "mean" in k:
                v.calculate(global_container, None, mode)
            if "f1" not in k:
                result_str += v.generate_print_string(mode)

        if "eval_f1_score" in evaluator.keys():  # make sure we do f1 at the end
            evaluator['eval_f1_score'].calculate(global_container['result_dict'], None, mode)
            result_str += evaluator['eval_f1_score'].generate_print_string(mode)
        result_str += '=' * 100 + '\n'

    logger.info(result_str)

    if "relations" in iou_types:
        # result_dict['detector_mAP@50'] = mAp
        if output_folder:
            out_file = os.path.join(output_folder, 'eval_results_top_' + str(cfg.TEST.TOP_K) + '.json')

            if "test" in dataset_name:
                result_file = os.path.join(output_folder, 'results.json')

                with open(result_file, 'r') as f:
                    res = json.load(f)

                # res['mAP@50'] = mAp

                # compute overall recall@20, 50, 100
                for k in [20, 50, 100]:
                    recall = result_dict[mode + '_recall'][k]
                    res[mode + '_recall@' + str(k)] = np.mean(recall)

                for k in [20, 50, 100]:
                    mean_recall = result_dict[mode + '_mean_recall'][k]
                    res[mode + '_mean_recall@' + str(k)] = np.mean(mean_recall)

                for k in [20, 50, 100]:
                    f1_score = result_dict[mode + '_f1_score'][k]
                    res[mode + '_f1_score@' + str(k)] = np.mean(f1_score)

                with open(result_file, 'w') as f:
                    json.dump(res, f)

            with open(out_file, 'w') as f:
                json.dump(result_dict, f)
            # torch.save(result_dict, os.path.join(output_folder, 'result_dict.pytorch'))
        return result_dict
    # elif "bbox" in iou_types:
    #     return {'mAP': float(mAp)}
    else:
        return -1


def save_output(output_folder, groundtruths, predictions, dataset):
    if output_folder:
        torch.save({'groundtruths': groundtruths, 'predictions': predictions},
                   os.path.join(output_folder, "eval_results.pytorch"))

        # with open(os.path.join(output_folder, "result.txt"), "w") as f:
        #    f.write(result_str)
        # visualization information
        visual_info = []
        for image_id, (groundtruth, prediction) in enumerate(zip(groundtruths, predictions)):
            img_file = os.path.abspath(dataset.filenames[image_id])
            groundtruth = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]]  # xyxy, str
                for b, l in zip(groundtruth.bbox.tolist(), groundtruth.get_field('labels').tolist())
            ]
            prediction = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]]  # xyxy, str
                for b, l in zip(prediction.bbox.tolist(), prediction.get_field('pred_labels').tolist())
            ]
            visual_info.append({
                'img_file': img_file,
                'groundtruth': groundtruth,
                'prediction': prediction
            })
        with open(os.path.join(output_folder, "visual_info.json"), "w") as f:
            json.dump(visual_info, f)


def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator, informative=False):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    # unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = groundtruth.get_field('relation_tuple').long().detach().cpu().numpy()

    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = groundtruth.convert('xyxy').bbox.detach().cpu().numpy()  # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth.get_field('labels').long().detach().cpu().numpy()  # (#gt_objs, )

    # about relations
    local_container['pred_rel_inds'] = prediction.get_field(
        'rel_pair_idxs').long().detach().cpu().numpy()  # (#pred_rels, 2)
    local_container['rel_scores'] = prediction.get_field(
        'pred_rel_scores').detach().cpu().numpy()  # (#pred_rels, num_pred_class)

    # about objects
    local_container['pred_boxes'] = prediction.convert('xyxy').bbox.detach().cpu().numpy()  # (#pred_objs, 4)
    local_container['pred_classes'] = prediction.get_field(
        'pred_labels').long().detach().cpu().numpy()  # (#pred_objs, )
    local_container['obj_scores'] = prediction.get_field('pred_scores').detach().cpu().numpy()  # (#pred_objs, )

    if informative:
        local_container['informative_rels'] = groundtruth.get_field('informative_rels')

    # to calculate accuracy, only consider those gt pairs
    # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing"
    # for sgcls and predcls
    if mode != 'sgdet':
        if "eval_pair_accuracy" in evaluator.keys():
            evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)

    # to calculate the prior label based on statistics for zero-shot
    for k, v in evaluator.items():
        if "zeroshot" in k:
            v.prepare_zeroshot(global_container, local_container)

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])
    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode not in ['sgdet', 'phrdet', 'predcls', 'sgcls']:
        raise ValueError('invalid mode')

    # check if any of the local container is empty
    for k, v in local_container.items():
        if isinstance(v, np.ndarray) and v.size == 0:
            return
        if isinstance(v, list) and len(v) == 0:
            return
        if torch.is_tensor(v) and v.shape[0] == 0:
            return

    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate(global_container, local_container, mode)

    for k, v in evaluator.items():
        if "mean" in k:
            v.collect_mean_recall_items(global_container, local_container, mode)
        elif "f1" not in k:  # meanRecall and F1 need to be computed at the end
            if k == "eval_recall":
                continue
            v.calculate(global_container, local_container, mode)

    return


def convert_relation_matrix_to_triplets(relation):
    triplets = []
    for i in range(len(relation)):
        for j in range(len(relation)):
            if relation[i, j] > 0:
                triplets.append((i, j, relation[i, j]))
    return torch.LongTensor(triplets)  # (num_rel, 3)


def generate_attributes_target(attributes, num_attributes):
    """
    from list of attribute indexs to [1,0,1,0,...,0,1] form
    """
    max_att = attributes.shape[1]
    num_obj = attributes.shape[0]

    with_attri_idx = (attributes.sum(-1) > 0).long()
    without_attri_idx = 1 - with_attri_idx
    num_pos = int(with_attri_idx.sum())
    num_neg = int(without_attri_idx.sum())
    assert num_pos + num_neg == num_obj

    attribute_targets = torch.zeros((num_obj, num_attributes), device=attributes.device).float()

    for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
        for k in range(max_att):
            att_id = int(attributes[idx, k])
            if att_id == 0:
                break
            else:
                attribute_targets[idx, att_id] = 1

    return attribute_targets