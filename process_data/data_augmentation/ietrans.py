import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm

from sgg_benchmark.config import cfg
from sgg_benchmark.data import make_data_loader
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.utils.logger import setup_logger, logger_step
from sgg_benchmark.structures.boxlist_ops import boxlist_iou
from sgg_benchmark.structures.boxlist_ops import cat_boxlist

import pickle

def add_gt_proposals(proposals, targets):
    """
    Arguments:
        proposals: list[BoxList]
        targets: list[BoxList]
    """
    new_targets = []
    for t in targets:
        new_t = t.copy_with_fields(["labels"])
        new_t.add_field("pred_labels", t.get_field("labels"))
        new_t.add_field("pred_scores", torch.ones_like(t.get_field("labels"), dtype=torch.float32))
        new_targets.append(new_t)

    proposals = [
        cat_boxlist((proposal, gt_box))
        for proposal, gt_box in zip(proposals, new_targets)
    ]

    return proposals

def process(path, output_file=None, categories=False):
    config_file = path

    cfg.merge_from_file(config_file)
    cfg.TEST.IMS_PER_BATCH = 1
    cfg.freeze()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    logger = setup_logger("sgg_benchmark", verbose="INFO", steps=True)

    logger_step(logger, 'Building model...')
    model = build_detection_model(cfg) 

    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    last_check = checkpointer.get_checkpoint_file()
    logger.info("Loading best checkpoint from {}...".format(last_check))
    _ = checkpointer.load(last_check)

    # dataset_to_test can be customize to test or val to perform data-transfer on the rest of the data
    data_loader_train = make_data_loader(cfg=cfg, mode="test", is_distributed=distributed, dataset_to_test='train')
    data_loader_train = data_loader_train[0]
    model.eval()

    ############
    # for loop #
    ############
    gt_rels_count = 0
    internal_trans_count = 0
    external_trans_count = 0
    dataset = data_loader_train.dataset
    stats = dataset.get_statistics()
    out_data = {}

    device = torch.device(cfg.MODEL.DEVICE)

    # show the number of triplets that have a frequency higher than 0 (not 0-shot triplet)
    print('triplet_freq: ', len([k for k, v in stats['triplet_freq'].items() if v > 0]))

    # triplet cat
    if categories:
        triplet_cat_path = "./process_data/data_augmentation/triplets_categories.pkl"
        with open(triplet_cat_path, 'rb') as f:
            triplet_cat = pickle.load(f)

        # transfer rules
        # {'functional': 0, 'topological': 1, 'attribute': 2, 'part-whole': 3}
        transfer_rules = {0: [0, 1], 1: [1, 0, 3], 2: [2], 3: [3]}

    pbar = tqdm(total=len(data_loader_train))
    for batch in data_loader_train:
        pbar.update(1)
        with torch.no_grad():
            images, targets, image_ids = batch
            targets = [target.to(device) for target in targets]

            outputs, features = model.backbone(images.tensors.to(device), embed=True)
            proposals = model.backbone.postprocess(outputs, images.image_sizes)

            proposals = add_gt_proposals(proposals,targets)

            _, predictions, _ = model.roi_heads(features, proposals, targets, logger, proposals)

        img_name = targets[0].get_field('image_path').split('/')[-1]
        out_data[img_name] = []
        img_info = dataset.get_img_info(image_ids[0])
        image_width = img_info["width"]
        image_height = img_info["height"]
        # recover original size which is before transform
        predictions = predictions[0]
        predictions = predictions.resize((image_width, image_height)).convert('xyxy').to(device)

        gt = dataset.get_groundtruth(image_ids[0], evaluation=True).to(device)
        assert dataset.filenames[image_ids[0]].split('/')[-1] == img_name, 'Image name does not match: %s %s' % (dataset.filenames[image_ids[0]], img_name)

        gt_labels = gt.get_field('labels') # integer
        gt_rels = gt.get_field('relation_tuple')

        # gt_rels is a tensor of shape (num_boxes, num_boxes) with 0 if no rel or the rel index otherwise
        # we need to convert it to a list of subject object pair
        sub_obj_pair_list = []
        for (s,o,r) in gt_rels:
            sub_obj_pair_list.append([s,o])

        # get gt_no_rels
        gt_no_rels = []
        for s in range(len(gt_labels)):  # subject
            for o in range(len(gt_labels)):  # object
                if s == o:
                    continue
                if [s, o] not in sub_obj_pair_list:
                    gt_no_rels.append([s, o, 0])
        gt_no_rels = np.array(gt_no_rels)

        # get pred
        pd_rels = predictions.get_field('rel_pair_idxs')
        pd_rel_dists = predictions.get_field('pred_rel_scores')
        pd_labels = predictions.get_field('pred_labels')
        
        # get iou between gt and pd
        ious = boxlist_iou(gt, predictions)

        ##################
        # internal trans #
        ##################
        gt_rels_count += gt_rels.shape[0]
        for i in range(gt_rels.shape[0]):
            gt_s_idx, gt_o_idx, gt_r_label = gt_rels[i]
            gt_s_label = gt_labels[gt_s_idx]
            gt_o_label = gt_labels[gt_o_idx]

            pd_s_labels = pd_labels[pd_rels[:, 0]]
            pd_o_labels = pd_labels[pd_rels[:, 1]]
            s_ious = ious[gt_s_idx, pd_rels[:, 0]]
            o_ious = ious[gt_o_idx, pd_rels[:, 1]]

            mask = (gt_s_label == pd_s_labels) & (gt_o_label == pd_o_labels) & (s_ious > 0.5) & (o_ious > 0.5)
            pd_r_dists_list = pd_rel_dists[mask]

            if pd_r_dists_list.size(0) > 0:
                pd_r_dists = pd_r_dists_list.mean(axis=0)

                if gt_r_label != torch.argmax(pd_r_dists):
                    r_sort = torch.argsort(pd_r_dists, descending=True)
                    gt_r_idx = (r_sort == gt_r_label.item()).nonzero(as_tuple=True)[0].item()

                    confusion_r_labels = r_sort[:gt_r_idx]
                    # we get then ratio of frequence of the triplet in original data over the frequence of the predicate alone              
                    if stats['triplet_freq'][(gt_s_label.item(), gt_o_label.item(), gt_r_label.item())] > 0:      
                        ori_attr = stats['triplet_freq'][(gt_s_label.item(), gt_o_label.item(), gt_r_label.item())] / stats['pred_freq'][gt_r_label.item()]
                    else:
                        ori_attr = 0.0

                    for c_r_label in confusion_r_labels:
                        if c_r_label != 0:
                            if stats['triplet_freq'][(gt_s_label.item(), gt_o_label.item(), c_r_label.item())] > 0: # zero-shot triplet
                                new_attr = stats['triplet_freq'][(gt_s_label.item(), gt_o_label.item(), c_r_label.item())] / stats['pred_freq'][c_r_label.item()]
                                if new_attr > ori_attr:
                                    # check transfer rules
                                    if categories:
                                        ori_cat = triplet_cat[(gt_s_label.item(), gt_o_label.item(), gt_r_label.item())]
                                        new_cat = triplet_cat[(gt_s_label.item(), gt_o_label.item(), c_r_label.item())]

                                        if new_cat in transfer_rules[ori_cat]:
                                            out_data[img_name].append([gt_s_idx.item(), gt_o_idx.item(), c_r_label.item()])
                                            internal_trans_count += 1
                                            break
                                    else:
                                        out_data[img_name].append([gt_s_idx.item(), gt_o_idx.item(), c_r_label.item()])
                                        internal_trans_count += 1
                                        break

        if len(gt_no_rels) > 0: # it is possible that there is no no-rels
            gt_s_idx = gt_no_rels[:, 0]
            gt_o_idx = gt_no_rels[:, 1]
            gt_r_label = gt_no_rels[:, 2]
            gt_s_label = gt_labels[gt_s_idx]
            gt_o_label = gt_labels[gt_o_idx]

            pd_s_idx = pd_rels[:, 0]
            pd_o_idx = pd_rels[:, 1]
            pd_s_label = pd_labels[pd_s_idx]
            pd_o_label = pd_labels[pd_o_idx]
            pd_r_dists = pd_rel_dists

            s_iou = ious[gt_s_idx[:, None], pd_s_idx]
            o_iou = ious[gt_o_idx[:, None], pd_o_idx]

            # Vectorize operations
            mask = (gt_s_label[:, None] == pd_s_label) & (gt_o_label[:, None] == pd_o_label) & (s_iou > 0.5) & (o_iou > 0.5) & (ious[gt_s_idx[:, None], pd_o_idx] > 0.1)
            
            pd_r_dists_list = [pd_r_dists[m] for m in mask]

            for i, pd_r_dists in enumerate(pd_r_dists_list):
                if len(pd_r_dists) > 0:
                    pd_r_dists = pd_r_dists.mean(axis=0)
                    if gt_r_label[i] != torch.argmax(pd_r_dists):
                        r_sort = torch.argsort(pd_r_dists, descending=True)
                        gt_r_idx = (r_sort == gt_r_label[i].item()).nonzero(as_tuple=True)[0].item()
                        confusion_r_labels = r_sort[:gt_r_idx]

                        attr_sort = [stats['triplet_freq'][(gt_s_label[i].item(), gt_o_label[i].item(), r_label.item())] / stats['pred_freq'][r_label.item()]  if stats['pred_freq'][r_label.item()] > 0 else 0.0 for r_label in confusion_r_labels]

                        confusion_r_labels = [x for _, x in sorted(zip(attr_sort, confusion_r_labels), key=lambda pair: pair[0], reverse=False)]
                        for c_r_label in confusion_r_labels:
                            # if int(c_r_label.item()) not in [25, 15, 1, 17, 23]: # remove some too frequent categories, i.e. [on, has, above, in, near]
                            if stats['triplet_freq'][(gt_s_label[i].item(), gt_o_label[i].item(), c_r_label.item())] > 0: 
                                out_data[img_name].append([gt_s_idx[i].item(), gt_o_idx[i].item(), c_r_label.item()])
                                external_trans_count += 1
                                break
                        # c_r_label = confusion_r_labels[0]
                        # out_data[img_name].append([gt_s_idx[i].item(), gt_o_idx[i].item(), c_r_label.item()])
                        # external_trans_count += 1


        pbar.set_description('int: %d, ext: %d' % (internal_trans_count, external_trans_count))

    print('gt_rels_count: ', gt_rels_count)
    print('internal_trans_count: ', internal_trans_count)
    print('external_trans_count: ', external_trans_count)
    if output_file is not None:
        fo = open(output_file, 'w')
        json.dump(out_data, fo)

if __name__ == '__main__':
    config_file = sys.argv[1]
    output_file = sys.argv[2]
    categories = True

    process(config_file, output_file, categories)