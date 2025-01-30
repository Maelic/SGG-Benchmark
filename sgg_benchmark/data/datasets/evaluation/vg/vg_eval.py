import os
import torch
import numpy as np
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from PIL import Image
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns

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
    metric_names = {'relations': ['recall', 'mean_recall', 'f1_score'], 'bbox': ['mAP']}

    relation_mAP = True # whether to evaluate relation mAP, i.e. compute the mAP only on boxes with at least one relation

    if cfg.TEST.INFORMATIVE:
        metric_names['relations'].extend(['informative_recall'])
    
    metrics_map = {'recall': SGRecall, 'recall_nogc': SGNoGraphConstraintRecall, 'zeroshot_recall': SGZeroShotRecall, 'ng_zeroshot_recall': SGNGZeroShotRecall, 'informative_recall': SGInformativeRecall, 'mean_recall': SGMeanRecall, 'recall_relative': SGRecallRelative, 'mean_recall_relative': SGMeanRecallRelative, 'f1_score': SGF1Score, 'weighted_recall': SGWeightedRecall, 'weighted_mean_recall': SGWeightedMeanRecall, 'CLIPScore': CLIPScoreMatching, 'BLIPScore': BLIPScoreMatching, 'SGLIPScore': SIGLIPScoreMatching}

    metrics_to_eval = {k: v for k,v in metrics_map.items() if k in metric_names['relations']}
    #metrics_to_eval['bbox'] = [] # 'mAP'

    # get zeroshot triplet
    if "relations" in iou_types:
        data_dir = DatasetCatalog.DATA_DIR
        try:
            zero_shot_file = DatasetCatalog.DATASETS[dataset_name]['zeroshot_file']
            zeroshot_triplet = torch.load(os.path.join(data_dir, zero_shot_file), map_location=torch.device("cpu")).long().numpy()
        except KeyError:
            zeroshot_triplet = np.array([])

    #attribute_on = cfg.MODEL.ATTRIBUTE_ON
    #num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
    # extract evaluation settings from cfg
    # mode = cfg.TEST.RELATION.EVAL_MODE
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            mode = 'predcls'
        else:
            mode = 'sgcls'
    else:
        mode = 'sgdet'

    num_rel_category = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
    multiple_preds = cfg.TEST.RELATION.MULTIPLE_PREDS
    iou_thres = cfg.TEST.RELATION.IOU_THRESHOLD
    pred_sampling = False #cfg.TEST.RELATION.PREDICATE_SAMPLING
    top_k = cfg.TEST.TOP_K

    if pred_sampling:
        # sampling space over top-k with step 5
        sampling_space = list(range(0, top_k+1, 5))
        print('Sampling space:', sampling_space)
        # linear space for interpolation after
        x = np.linspace(0, top_k, num=top_k+1)
    else:
        sampling_space = [cfg.TEST.TOP_K]

    assert mode in {'predcls', 'sgdet', 'sgcls', 'phrdet', 'preddet'}

    global_result_dict = {k :{} for k in sampling_space}
    
    if "relations" in iou_types:
        for k in sampling_space:
            groundtruths = []
            logger.info('Evaluating top-%d' % k)

            new_preds = [None] * len(predictions)

            for image_id, prediction in enumerate(predictions):
                img_info = dataset.get_img_info(image_id)
                image_width = img_info["width"]
                image_height = img_info["height"]
                # recover original size which is before transform
                preds = prediction.copy_with_fields(['pred_scores', 'pred_labels', 'rel_pair_idxs', 'pred_rel_scores', 'pred_rel_labels'])
                if preds.size != (image_width, image_height):
                    preds = preds.resize((image_width, image_height))

                # remove all boxes not in the top k
                pred_box_scores = preds.get_field('pred_scores')
                if len(pred_box_scores) < k:
                    new_preds[image_id] = preds
                    gt = dataset.get_groundtruth(image_id, evaluation=True)
                    groundtruths.append(gt)
                    continue

                preds.bbox = preds.bbox[:k]
                preds.extra_fields['pred_scores'] = preds.extra_fields['pred_scores'][:k]
                preds.extra_fields['pred_labels'] = preds.extra_fields['pred_labels'][:k]
                
                # remove all relations that contain boxes not in the top k
                pred_rel_inds = preds.get_field('rel_pair_idxs')
                pred_rel_scores = preds.get_field('pred_rel_scores')
                keep = (pred_rel_inds < k).all(1)

                preds.extra_fields['rel_pair_idxs'] = pred_rel_inds[keep]
                preds.extra_fields['pred_rel_scores'] = pred_rel_scores[keep]
                preds.extra_fields['pred_rel_labels'] = preds.extra_fields['pred_rel_labels'][keep]
                
                new_preds[image_id] = preds
                gt = dataset.get_groundtruth(image_id, evaluation=True)
                groundtruths.append(gt)

            if informative and not 'informative_rels' in groundtruths[0].extra_fields.keys():
                # logger.info('Dataset does not have informative_rels, skipping informative evaluation for dataset: %s' % dataset_name)
                informative = False

            result_str = '\n' + '=' * 100 + '\n'

            if "relations" in iou_types:
                result_dict = {}
                evaluator = {}

                for m, v in metrics_to_eval.items():
                    if "mean" in m:
                        cur_metric = v(result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
                    else:
                        cur_metric = v(result_dict)
                    cur_metric.register_container(mode)
                    evaluator["eval_"+m] = cur_metric

                # prepare all inputs
                global_container = {}
                global_container['zeroshot_triplet'] = zeroshot_triplet
                global_container['result_dict'] = result_dict
                global_container['mode'] = mode
                global_container['multiple_preds'] = multiple_preds
                global_container['num_rel_category'] = num_rel_category
                global_container['iou_thres'] = iou_thres
                #global_container['attribute_on'] = attribute_on
                #global_container['num_attributes'] = num_attributes

                #if informative:
                stats = dataset.get_statistics()
                global_container['ind_to_predicates'] = stats['rel_classes']
                global_container['ind_to_classes'] = stats['obj_classes']

                fg_matrix = stats['fg_matrix']

                global_container['fg_triplets'] = convert_relation_matrix_to_triplets(fg_matrix)

                i = 0
            
                for groundtruth, prediction in tqdm(zip(groundtruths, new_preds), desc='Evaluating', total=len(groundtruths)):
                    evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator, informative=informative)
                    i+= 1
                    # if i == 1000:
                    #     break

                for m,v in evaluator.items():
                    if "mean" in m:
                        v.calculate(global_container, None, mode)
                    if "f1" not in m:
                        result_str += v.generate_print_string(mode)

                if "eval_f1_score" in evaluator.keys(): # make sure we do f1 at the end
                    evaluator['eval_f1_score'].calculate(global_container['result_dict'], None, mode)
                    result_str += evaluator['eval_f1_score'].generate_print_string(mode)
                result_str += '=' * 100 + '\n'

                global_result_dict[k]['recall'] = np.mean([np.mean(result_dict[mode + '_recall'][100]), np.mean(result_dict[mode + '_recall'][50]), np.mean(result_dict[mode + '_recall'][20])])
                global_result_dict[k]['mean_recall'] = np.mean([np.mean(result_dict[mode + '_mean_recall'][100]), np.mean(result_dict[mode + '_mean_recall'][50]), np.mean(result_dict[mode + '_mean_recall'][20])])
                global_result_dict[k]['f1_score'] = np.mean([np.mean(result_dict[mode + '_f1_score'][100]), np.mean(result_dict[mode + '_f1_score'][50]), np.mean(result_dict[mode + '_f1_score'][20])])

            if len(sampling_space) == 1:
                logger.info(result_str)
            elif k == 100:
                logger.info('Evaluation finished for top-%d' % k)
                logger.info(result_str)
                result_str = ''
            else:
                logger.info('Evaluation finished for top-%d' % k)
                result_str = ''
        
        if len(sampling_space) > 1:
            # Extract metrics
            metrics = list(global_result_dict[sampling_space[0]].keys())
            optimum_per_metric = {}

            # Create a dictionary to store interpolation functions for each metric
            interpolation_functions = {}
            derivatives = {}

            # Generate interpolation functions for each metric
            for metric in metrics:
                # Extract values for the current metric
                metric_values = [global_result_dict[k][metric] for k in sampling_space]
                metric_values[0] = 0.0
                
                # Interpolate the data using scipy interp1d to handle the case where the data is not monotonically increasing
                interpolation_function = interp1d(sampling_space, metric_values, kind='cubic', fill_value="extrapolate")
                interpolated_values = interpolation_function(x)
                interpolation_functions[metric] = interpolated_values

                # Calculate the derivative
                derivative = np.gradient(interpolated_values, x)
                derivatives[metric] = derivative

                # Find the point where the derivative is below a certain threshold
                thres = 1e-5
                optimum_index = np.where(np.abs(derivative) < thres)[0][0]
                
                # Store the optimum top-k for the current metric
                optimum_per_metric[metric] = {'index': int(x[optimum_index]), 'value': interpolated_values[optimum_index]}

                # Generate nice curve for plotting
                if metric == 'relation_mAP':
                    name = 'Relation mAP'
                elif metric == 'mAP':
                    name = 'Detection mAP'
                elif metric == 'recall':
                    name = 'Recall@K'
                elif metric == 'mean_recall':
                    name = 'meanRecall@K'
                elif metric == 'f1_score':
                    # save the interpolation_functions[metric] and derivatives[metric] to a json file
                    final_dict = {'interpolation': interpolation_functions[metric].tolist(), 'derivative': derivatives[metric].tolist(), 'optimum': optimum_per_metric[metric]}
                    with open(os.path.join(output_folder, f'{metric}_vs_Top-K.json'), 'w') as f:
                        json.dump(final_dict, f)
                    name = 'F1@K'
                
                sns.set_theme(style="whitegrid")
                sns.set_context("paper", font_scale=1.5)

                plt.figure(figsize=(10, 6))
                sns.lineplot(x=x, y=interpolation_functions[metric], label=f'{metric} (Interpolated)')
                sns.lineplot(x=x, y=derivatives[metric], label=f'{metric} (Derivative)', linestyle='--')

                # display the value for the optimum top-k on the plot
                plt.axvline(x=optimum_per_metric[metric]['index'], color='r', linestyle='--', label=f'Optimum top-{optimum_per_metric[metric]["index"]} with value {optimum_per_metric[metric]["value"]}')

                plt.xlabel('Top-K', fontsize=14)
                plt.ylabel(name, fontsize=14)
                plt.title(f'{name} vs Top-K', fontsize=16)
                plt.legend()
                plt.savefig(os.path.join(output_folder, f'{metric}_vs_Top-K.png'))
                plt.close()

        # display the optimum top-k for each metric, in a nice way
        if len(sampling_space) > 1:
            logger.info('Optimum top-k for each metric:')
            for metric, optimum in optimum_per_metric.items():
                logger.info(f'{metric}: top-{optimum["index"]} with value {optimum["value"]}')

        metric_to_optimize = "f1_score"
        # redo evaluation with optimal top-k for the metric to optimize
        if len(sampling_space) > 1:
            logger.info(f'Optimizing for metric: {metric_to_optimize}')
            top_k = optimum_per_metric[metric_to_optimize]["index"]
            logger.info(f'Optimum top-k: {top_k}')

            groundtruths = []
            new_preds = [None] * len(predictions)

            for image_id, prediction in enumerate(predictions):
                img_info = dataset.get_img_info(image_id)
                image_width = img_info["width"]
                image_height = img_info["height"]
                # recover original size which is before transform
                preds = prediction.copy_with_fields(['pred_scores', 'pred_labels', 'rel_pair_idxs', 'pred_rel_scores', 'pred_rel_labels'])
                if preds.size != (image_width, image_height):
                    preds = preds.resize((image_width, image_height))

                # remove all boxes not in the top k
                pred_box_scores = preds.get_field('pred_scores')
                if len(pred_box_scores) < top_k:
                    new_preds[image_id] = preds
                    gt = dataset.get_groundtruth(image_id, evaluation=True)
                    groundtruths.append(gt)
                    continue

                preds.bbox = preds.bbox[:top_k]
                preds.extra_fields['pred_scores'] = preds.extra_fields['pred_scores'][:top_k]
                preds.extra_fields['pred_labels'] = preds.extra_fields['pred_labels'][:top_k]
                
                # remove all relations that contain boxes not in the top k
                pred_rel_inds = preds.get_field('rel_pair_idxs')
                pred_rel_scores = preds.get_field('pred_rel_scores')
                keep = (pred_rel_inds < top_k).all(1)

                preds.extra_fields['rel_pair_idxs'] = pred_rel_inds[keep]
                preds.extra_fields['pred_rel_scores'] = pred_rel_scores[keep]
                preds.extra_fields['pred_rel_labels'] = preds.extra_fields['pred_rel_labels'][keep]
                
                new_preds[image_id] = preds
                gt = dataset.get_groundtruth(image_id, evaluation=True)
                groundtruths.append(gt)

            result_dict = {}
            evaluator = {}

            for m, v in metrics_to_eval.items():
                if "mean" in m:
                    cur_metric = v(result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
                else:
                    cur_metric = v(result_dict)
                cur_metric.register_container(mode)
                evaluator["eval_"+m] = cur_metric

            # prepare all inputs
            global_container = {}
            global_container['zeroshot_triplet'] = zeroshot_triplet
            global_container['result_dict'] = result_dict
            global_container['mode'] = mode
            global_container['multiple_preds'] = multiple_preds
            global_container['num_rel_category'] = num_rel_category
            global_container['iou_thres'] = iou_thres
            #global_container['attribute_on'] = attribute_on
            #global_container['num_attributes'] = num_attributes

            #if informative:
            stats = dataset.get_statistics()
            global_container['ind_to_predicates'] = stats['rel_classes']
            global_container['ind_to_classes'] = stats['obj_classes']
            
            for groundtruth, prediction in tqdm(zip(groundtruths, new_preds), desc='Evaluating', total=len(groundtruths)):
                evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator, informative=informative)
            
            for m,v in evaluator.items():
                if "mean" in m:
                    v.calculate(global_container, None, mode)
                if "f1" not in m:
                    result_str += v.generate_print_string(mode)

            if "eval_f1_score" in evaluator.keys(): # make sure we do f1 at the end
                evaluator['eval_f1_score'].calculate(global_container['result_dict'], None, mode)
                result_str += evaluator['eval_f1_score'].generate_print_string(mode)
            result_str += '=' * 100 + '\n'

            global_result_dict[k]['recall'] = np.mean([np.mean(result_dict[mode + '_recall'][100]), np.mean(result_dict[mode + '_recall'][50]), np.mean(result_dict[mode + '_recall'][20])])
            global_result_dict[k]['mean_recall'] = np.mean([np.mean(result_dict[mode + '_mean_recall'][100]), np.mean(result_dict[mode + '_mean_recall'][50]), np.mean(result_dict[mode + '_mean_recall'][20])])
            global_result_dict[k]['f1_score'] = np.mean([np.mean(result_dict[mode + '_f1_score'][100]), np.mean(result_dict[mode + '_f1_score'][50]), np.mean(result_dict[mode + '_f1_score'][20])])

    if "bbox" in iou_types and "mAP" in metric_names['bbox']:
        mAp, result_dict, result_str = compute_map(groundtruths, new_preds, dataset, mode, result_dict, result_str, relation_mAP=relation_mAP)

    logger.info(result_str)
        
    if "relations" in iou_types:
        if output_folder:
            result_dict['detector_mAP@50'] = mAp

            out_file = os.path.join(output_folder, 'eval_results_top_'+str(cfg.TEST.TOP_K)+'.json')

            #if "test" in dataset_name:
            result_file = os.path.join(output_folder, 'results.json')

            with open(result_file, 'r') as f:
                res = json.load(f)

            res['mAP@50'] = mAp

            # compute overall recall@20, 50, 100
            for k in [20, 50, 100]:
                recall = result_dict[mode + '_recall'][k]
                res[mode + '_recall@'+str(k)] = np.mean(recall)

            for k in [20, 50, 100]:
                mean_recall = result_dict[mode + '_mean_recall'][k]
                res[mode + '_mean_recall@'+str(k)] = np.mean(mean_recall)

            for k in [20, 50, 100]:
                f1_score = result_dict[mode + '_f1_score'][k]
                res[mode + '_f1_score@'+str(k)] = np.mean(f1_score)

            with open(result_file, 'w') as f:
                json.dump(res, f)

            with open(out_file, 'w') as f:
                json.dump(result_dict, f)
            torch.save(result_dict, os.path.join(output_folder, 'result_dict.pytorch'))
        return result_dict
    elif "bbox" in iou_types:
        return {'mAP': float(mAp)}
    else:
        return -1

def save_output(output_folder, groundtruths, predictions, dataset):
    if output_folder:
        torch.save({'groundtruths':groundtruths, 'predictions':predictions}, os.path.join(output_folder, "eval_results.pytorch"))

        #with open(os.path.join(output_folder, "result.txt"), "w") as f:
        #    f.write(result_str)
        # visualization information
        visual_info = []
        for image_id, (groundtruth, prediction) in enumerate(zip(groundtruths, predictions)):
            img_file = os.path.abspath(dataset.filenames[image_id])
            groundtruth = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]] # xyxy, str
                for b, l in zip(groundtruth.bbox.tolist(), groundtruth.get_field('labels').tolist())
                ]
            prediction = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]] # xyxy, str
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
    #unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = groundtruth.get_field('relation_tuple').long().detach().cpu().numpy()

    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = groundtruth.convert('xyxy').bbox.detach().cpu().numpy()                   # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth.get_field('labels').long().detach().cpu().numpy()           # (#gt_objs, )

    # about relations
    local_container['pred_rel_inds'] = prediction.get_field('rel_pair_idxs').long().detach().cpu().numpy()  # (#pred_rels, 2)
    local_container['rel_scores'] = prediction.get_field('pred_rel_scores').detach().cpu().numpy()          # (#pred_rels, num_pred_class)

    # about objects
    local_container['pred_boxes'] = prediction.convert('xyxy').bbox.detach().cpu().numpy()                  # (#pred_objs, 4)
    local_container['pred_classes'] = prediction.get_field('pred_labels').long().detach().cpu().numpy()     # (#pred_objs, )
    local_container['obj_scores'] = prediction.get_field('pred_scores').detach().cpu().numpy()              # (#pred_objs, )
    
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
    for k,v in local_container.items():
        if isinstance(v, np.ndarray) and v.size == 0:
            return
        if isinstance(v, list) and len(v) == 0:
            return
        if torch.is_tensor(v) and v.shape[0] == 0:
            return
    
    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate(global_container, local_container, mode)

    # clip score
    if "eval_CLIPScore" in evaluator.keys() or "eval_BLIPScore" in evaluator.keys() or "eval_SGLIPScore" in evaluator.keys():
        # get gt image
        img_path = groundtruth.get_field('image_path')
        img = Image.open(img_path).convert('RGB')
        local_container['gt_image'] = img
        local_container['img_path'] = img_path

    for k, v in evaluator.items():
        if "mean" in k:
            v.collect_mean_recall_items(global_container, local_container, mode)
        elif "f1" not in k: # meanRecall and F1 need to be computed at the end
            if k=="eval_recall":
                continue
            else:
                v.calculate(global_container, local_container, mode)            
    return 

def convert_relation_matrix_to_triplets(relation):
    triplets = []
    for i in range(len(relation)):
        for j in range(len(relation)):
            for k in range(len(relation[i, j])):
                if relation[i, j, k] > 0:
                    triplets.append((i, j, k))
    return torch.LongTensor(triplets) # (num_rel, 3)


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

def compute_map(groundtruths, predictions, dataset, mode, result_dict, result_str, relation_mAP=False):
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
            label = prediction.get_field('pred_labels').detach().cpu().numpy()
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
    # if len(sampling_space) == 1:
    coco_eval.summarize()
    mAp = coco_eval.stats[1]

    result_dict['mAP'] = float(mAp)
    
    result_str += 'Detection evaluation mAp=%.4f\n' % mAp
    result_str += '=' * 100 + '\n'

    if relation_mAP == True:
        # create a Coco-like object that we can use to evaluate detection!
        anns = []
        for image_id, gt in enumerate(groundtruths):

            labels = gt.get_field('labels').tolist() # integer
            boxes = gt.bbox.tolist() # xyxy
            rels = gt.get_field('relation_tuple').tolist() # (num_rel, 3)
            # remove all boxes with no relations
            has_rel = np.zeros(len(labels), dtype=bool)
            for rel in rels:
                has_rel[rel[0]] = True
                has_rel[rel[1]] = True
            labels = [l for l, hr in zip(labels, has_rel) if hr]
            boxes = [b for b, hr in zip(boxes, has_rel) if hr]

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
                label = prediction.get_field('pred_labels').detach().cpu().numpy()
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
        # if len(sampling_space) == 1:
        coco_eval.summarize()
        mAp = coco_eval.stats[1]

        result_dict['relation_mAP'] = float(mAp)
        
        result_str += 'Relation-based detection evaluation mAp=%.4f\n' % mAp
        result_str += '=' * 100 + '\n'

    return mAp, result_dict, result_str