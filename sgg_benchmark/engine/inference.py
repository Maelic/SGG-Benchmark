# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import numpy as np

import json
import torch
from tqdm import tqdm

from sgg_benchmark.config import cfg
from sgg_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from .bbox_aug import im_detect_bbox_aug
import networkx as nx
import datetime


def compute_on_dataset(model, data_loader, device, synchronize_gather=True, timer=None, silence=False,
                       informative=False):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    timings = []

    if informative:
        obj_classes, pred_classes, informative_rels = init_informative_post_process()
        starter2, ender2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings2 = []

    for i, batch in enumerate(tqdm(data_loader, disable=silence)):
        # if i == 100:
        #     break
        with torch.no_grad():
            images, targets, image_ids = batch
            targets = [target.to(device) for target in targets]

            if timer:
                starter.record()
                if informative:
                    starter2.record()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                # compute GFLOPS
                output = model(images.to(device), targets)
            if timer:
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings.append(curr_time)

            if informative:
                for boxlist in output:
                    informative_post_process(boxlist=boxlist, obj_classes=obj_classes, pred_classes=pred_classes,
                                             informative_rels=informative_rels)
                if timer:
                    ender2.record()
                    torch.cuda.synchronize()
                    curr_time = starter2.elapsed_time(ender2)
                    timings2.append(curr_time)

            output = [o.to(cpu_device) for o in output]

            # add image_path to output
            if output[0].has_field('image_path'):
                for i, o in enumerate(output):
                    o.add_field('image_path', targets[i].get_field('image_path'))

        if synchronize_gather:
            synchronize()
            multi_gpu_predictions = all_gather({img_id: result for img_id, result in zip(image_ids, output)})
            if is_main_process():
                for p in multi_gpu_predictions:
                    results_dict.update(p)
        else:
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
            # detected_sgg = custom_sgg_post_precessing(results_dict)
            # clean_graph = generate_detect_sg(detected_sgg, vg_dict)
            # save the detected sgg to npy file
            # np.save(os.path.join(save_dir, 'sgg_{}.npy'.format(image_id)), clean_graph)
    torch.cuda.empty_cache()
    if not informative: timings2 = timings
    return results_dict, timings, timings2


def init_informative_post_process():
    classes_dict = "./datasets/IndoorVG_4/VG-SGG-dicts.json"
    with open(classes_dict, 'r') as f:
        classes = json.load(f)

    informative_path = "./similarity_mpnet.json"

    with open(informative_path, 'r') as f:
        informative_rels = json.load(f)

    obj_classes = classes['idx_to_label']
    obj_classes = {int(k): v for k, v in obj_classes.items()}
    pred_classes = classes['idx_to_predicate']
    pred_classes = {int(k): v for k, v in pred_classes.items()}

    return obj_classes, pred_classes, informative_rels


def informative_post_process(boxlist, obj_classes, pred_classes, informative_rels, top_n=100):
    scores = boxlist.get_field('pred_rel_scores')[:top_n]
    labels = boxlist.get_field('pred_rel_labels')[:top_n]
    pd_rels = boxlist.get_field('rel_pair_idxs')[:top_n]

    if len(pd_rels) == 0:
        return

    pd_labels = boxlist.get_field('pred_labels')

    pd_s_labels = pd_labels[pd_rels[:, 0]]
    pd_o_labels = pd_labels[pd_rels[:, 1]]

    # Pre-compute mappings for subject and object labels
    subj_strs = [obj_classes[idx.item()] for idx in pd_s_labels]
    obj_strs = [obj_classes[idx.item()] for idx in pd_o_labels]
    pred_strs = [pred_classes[idx.item()] for idx in labels]

    # tensor of size subj_strs
    # info_scores = torch.zeros(len(subj_strs))

    nx_graph = nx.DiGraph()

    # Pre-compute the values of sub_id, obj_id, sub, and obj outside the loop
    sub_ids = pd_rels[:, 0]
    obj_ids = pd_rels[:, 1]

    # Compute inform_score and add edges to nx_graph
    inform_scores = np.array([informative_rels.get(f"{sub_str} {pred} {obj_str}", 0.0) for sub_str, obj_str, pred in
                              zip(subj_strs, obj_strs, pred_strs)])
    sub_strs = np.array([str(sub_id.item()) + subj_str for sub_id, subj_str in zip(sub_ids, subj_strs)])
    obj_strs = np.array([str(obj_id.item()) + obj_str for obj_id, obj_str in zip(obj_ids, obj_strs)])
    nx_graph.add_edges_from(
        [(sub, obj, {'weight': inform_score, 'distance': 1 - inform_score}) for sub, obj, inform_score in
         zip(sub_strs, obj_strs, inform_scores)])

    # Compute edge betweenness centrality and normalize values_norm
    betw = nx.edge_betweenness_centrality(nx_graph, normalized=True, weight='distance')
    # normalize between 0 and 1
    values_norm = torch.nn.functional.normalize(torch.tensor(list(betw.values())), dim=0)
    values_norm = values_norm.tolist()

    # Compute triple_scores
    # max_scores = scores[:,1:].max(1)[0].tolist()
    triple_scores = (values_norm + inform_scores) / 2  # 0.5 * (values_norm + inform_scores)

    # Sort the scores in descending order and get the sorting indices
    sorting_idx = list(np.argsort(triple_scores)[::-1])
    # full_idx = torch.arange(0,len(scores),1)

    # # replace the first len(sorting_idx) elements of full_idx with sorting_idx
    # full_idx[:len(sorting_idx)] = sorting_idx

    boxlist.remove_field('pred_rel_scores')
    boxlist.remove_field('pred_rel_labels')
    boxlist.remove_field('rel_pair_idxs')

    boxlist.add_field('pred_rel_scores', scores[sorting_idx])
    boxlist.add_field('pred_rel_labels', labels[sorting_idx])
    boxlist.add_field('rel_pair_idxs', pd_rels[sorting_idx])


def generate_detect_sg(predictions, vg_dict, obj_thres=0.5):
    all_obj_labels = predictions.get_field('pred_labels')
    all_obj_scores = predictions.get_field('pred_scores')
    all_rel_pairs = predictions.get_field('rel_pair_idxs')
    all_rel_prob = predictions.get_field('pred_rel_scores')
    all_rel_scores, all_rel_labels = all_rel_prob.max(-1)

    # filter objects and relationships
    all_obj_scores[all_obj_scores < obj_thres] = 0.0
    obj_mask = all_obj_scores >= obj_thres
    triplet_score = all_obj_scores[all_rel_pairs[:, 0]] * all_obj_scores[all_rel_pairs[:, 1]] * all_rel_scores
    rel_mask = ((all_rel_labels > 0) + (triplet_score > 0)) > 0

    # generate filterred result
    num_obj = obj_mask.shape[0]
    num_rel = rel_mask.shape[0]
    rel_matrix = torch.zeros((num_obj, num_obj))
    triplet_scores_matrix = torch.zeros((num_obj, num_obj))
    rel_scores_matrix = torch.zeros((num_obj, num_obj))
    for k in range(num_rel):
        if rel_mask[k]:
            rel_matrix[int(all_rel_pairs[k, 0]), int(all_rel_pairs[k, 1])], triplet_scores_matrix[
                int(all_rel_pairs[k, 0]), int(all_rel_pairs[k, 1])], rel_scores_matrix[
                int(all_rel_pairs[k, 0]), int(all_rel_pairs[k, 1])] = all_rel_labels[k], triplet_score[k], \
            all_rel_scores[k]
    rel_matrix = rel_matrix[obj_mask][:, obj_mask].long()
    triplet_scores_matrix = triplet_scores_matrix[obj_mask][:, obj_mask].float()
    rel_scores_matrix = rel_scores_matrix[obj_mask][:, obj_mask].float()
    filter_obj = all_obj_labels[obj_mask]
    filter_pair = torch.nonzero(rel_matrix > 0)
    filter_rel = rel_matrix[filter_pair[:, 0], filter_pair[:, 1]]
    filter_scores = triplet_scores_matrix[filter_pair[:, 0], filter_pair[:, 1]]
    filter_rel_scores = rel_scores_matrix[filter_pair[:, 0], filter_pair[:, 1]]
    # assert that filter_rel and filter_scores are same shape:
    assert (filter_rel.size() == filter_scores.size() == filter_rel_scores.size())
    # generate labels
    pred_objs = [vg_dict['idx_to_label'][str(i)] for i in filter_obj.tolist()]
    pred_rels = [[i[0], i[1], vg_dict['idx_to_predicate'][str(j)], s, z] for i, j, s, z in
                 zip(filter_pair.tolist(), filter_rel.tolist(), filter_scores.tolist(), filter_rel_scores.tolist())]

    output = [{'entities': pred_objs, 'relations': pred_rels}, ]

    return output


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, synchronize_gather=True):
    if not synchronize_gather:
        all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return

    if synchronize_gather:
        predictions = predictions_per_gpu
    else:
        # merge the list of dicts
        predictions = {}
        for p in all_predictions:
            predictions.update(p)

    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "WARNING! WARNING! WARNING! WARNING! WARNING! WARNING!"
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        logger=None,
        informative=False,
        silence=False,
):
    load_prediction_from_cache = cfg.TEST.ALLOW_LOAD_FROM_CACHE and output_folder is not None and os.path.exists(
        os.path.join(output_folder, "predictions.pth"))

    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    if logger is None:
        logger = logging.getLogger("sgg_benchmark.inference")
    dataset = data_loader.dataset

    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    # get dataset name
    p = dataset_name.rfind("_")
    name, split = dataset_name[:p], dataset_name[p + 1:]

    if load_prediction_from_cache:
        pred_path = os.path.join(output_folder, "predictions.pth")
        predictions = torch.load(pred_path, map_location=torch.device("cpu"))
        logger.info("Loaded predictions from cache in {}".format(pred_path))
    else:
        predictions, timings, timings2 = compute_on_dataset(model, data_loader, device,
                                                            synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER,
                                                            timer=True, silence=silence, informative=False)
        # wait for all processes to complete before measuring the time
        synchronize()
        batch_size = int(cfg.TEST.IMS_PER_BATCH)

        total_time_str = str(datetime.timedelta(seconds=int(sum(timings2) / 1000)))
        avg_per_image = np.mean(timings2) / batch_size
        logger.info(
            "Total run time: {} ({} ms / img per device, on {} devices)".format(
                total_time_str, avg_per_image, num_devices
            )
        )

        # get batch size
        mean_syn = np.mean(timings) / batch_size
        mean_std = np.std(timings) / batch_size
        logger.info(
            "Average latency per image: {}ms".format(mean_syn)
        )
        logger.info(
            "Standard deviation of latency: {}ms".format(mean_std)
        )

    if not load_prediction_from_cache:
        predictions = _accumulate_predictions_from_multiple_gpus(predictions,
                                                                 synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER)

    if not is_main_process():
        return -1.0

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        informative=informative,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    # save preditions to .pth file
    if output_folder is not None and not load_prediction_from_cache:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
        latency = {
            'mean_syn': mean_syn,
            'mean_std': mean_std,
            'latency_raw': avg_per_image
        }
        file_name = "results.json"
        # if file exists, load it and update it
        data = latency
        if os.path.exists(os.path.join(output_folder, file_name)):
            with open(os.path.join(output_folder, file_name), 'r') as f:
                data = json.load(f)
            data.update(latency)
        json.dump(data, open(os.path.join(output_folder, file_name), "w"))

    if cfg.TEST.CUSTUM_EVAL:
        detected_sgg = custom_sgg_post_precessing(predictions)
        with open(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_prediction.json'), 'w') as outfile:
            json.dump(detected_sgg, outfile)
        print('=====> ' + str(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_prediction.json')) + ' SAVED !')
        return -1.0

    # return None
    return evaluate(cfg=cfg,
                    dataset=dataset,
                    dataset_name=name,
                    predictions=predictions,
                    output_folder=output_folder,
                    logger=logger,
                    **extra_args)


def custom_sgg_post_precessing(predictions):
    output_dict = {}
    for idx, boxlist in enumerate(predictions):
        xyxy_bbox = boxlist.convert('xyxy').bbox
        # current sgg info
        current_dict = {}
        # sort bbox based on confidence
        sortedid, id2sorted = get_sorted_bbox_mapping(boxlist.get_field('pred_scores').tolist())
        # sorted bbox label and score
        bbox = []
        bbox_labels = []
        bbox_scores = []
        for i in sortedid:
            bbox.append(xyxy_bbox[i].tolist())
            bbox_labels.append(boxlist.get_field('pred_labels')[i].item())
            bbox_scores.append(boxlist.get_field('pred_scores')[i].item())
        current_dict['bbox'] = bbox
        current_dict['bbox_labels'] = bbox_labels
        current_dict['bbox_scores'] = bbox_scores
        # sorted relationships
        rel_sortedid, _ = get_sorted_bbox_mapping(boxlist.get_field('pred_rel_scores')[:, 1:].max(1)[0].tolist())
        # sorted rel
        rel_pairs = []
        rel_labels = []
        rel_scores = []
        rel_all_scores = []
        for i in rel_sortedid:
            rel_labels.append(boxlist.get_field('pred_rel_scores')[i][1:].max(0)[1].item() + 1)
            rel_scores.append(boxlist.get_field('pred_rel_scores')[i][1:].max(0)[0].item())
            rel_all_scores.append(boxlist.get_field('pred_rel_scores')[i].tolist())
            old_pair = boxlist.get_field('rel_pair_idxs')[i].tolist()
            rel_pairs.append([id2sorted[old_pair[0]], id2sorted[old_pair[1]]])
        current_dict['rel_pairs'] = rel_pairs
        current_dict['rel_labels'] = rel_labels
        current_dict['rel_scores'] = rel_scores
        current_dict['rel_all_scores'] = rel_all_scores
        output_dict[idx] = current_dict
    return output_dict


def get_sorted_bbox_mapping(score_list):
    sorted_scoreidx = sorted([(s, i) for i, s in enumerate(score_list)], reverse=True)
    sorted2id = [item[1] for item in sorted_scoreidx]
    id2sorted = [item[1] for item in sorted([(j, i) for i, j in enumerate(sorted2id)])]
    return sorted2id, id2sorted