import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sgg_benchmark.config import cfg
from sgg_benchmark.data import make_data_loader
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.utils.logger import setup_logger, logger_step
from sgg_benchmark.utils.miscellaneous import mkdir, save_config
from sgg_benchmark.structures.boxlist_ops import boxlist_iou


def process(path, output_file=None):
    base_path = path
    categories_path = base_path+"/categories_gpt3_Indoorvg4.csv"
    stats_path = base_path+"/VG-SGG-dicts.json"
    config_file = base_path+"/config.yml"

    categories = {}
    with open(categories_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split(',')
            try:
                categories[line[0]] = line[1]
            except:
                print(line)
                print(i)
                exit(0)
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    idx_to_label = stats['idx_to_label']
    idx_to_predicate = stats['idx_to_predicate']

    cfg.merge_from_file(config_file)
    cfg.freeze()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    logger = setup_logger("sgg_benchmark", verbose=True, steps=True)

    logger_step(logger, 'Building model...')
    model = build_detection_model(cfg) 

    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    last_check = checkpointer.get_checkpoint_file()
    logger.info("Loading best checkpoint from {}...".format(last_check))
    _ = checkpointer.load(last_check)

    dataset_names = cfg.DATASETS.TEST

    # This variable enables the script to run the test on any dataset split.
    if cfg.DATASETS.TO_TEST:
        assert cfg.DATASETS.TO_TEST in {'train', 'val', 'test', None}
        if cfg.DATASETS.TO_TEST == 'train':
            dataset_names = cfg.DATASETS.TRAIN
        elif cfg.DATASETS.TO_TEST == 'val':
            dataset_names = cfg.DATASETS.VAL

    output_folders = [None] * len(cfg.DATASETS.TEST)

    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loader_train = make_data_loader(cfg=cfg, mode="test", is_distributed=distributed, dataset_to_test='train')
    #data_loader_train = make_data_loader(cfg=cfg, mode="test", is_distributed=distributed)
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

    # show the number of triplets that ahev a frequency higher than 0
    print('triplet_freq: ', len([k for k, v in stats['triplet_freq'].items() if v > 0]))

    pbar = tqdm(total=len(data_loader_train))
    for batch in data_loader_train:
        pbar.update(1)
        with torch.no_grad():
            images, targets, image_ids = batch
            targets = [target.to(device) for target in targets]

            predictions = model(images.to(device), targets)

        out_data[image_ids[0]] = []
        img_info = dataset.get_img_info(image_ids[0])
        image_width = img_info["width"]
        image_height = img_info["height"]
        # recover original size which is before transform
        predictions = predictions[0]
        predictions = predictions.resize((image_width, image_height)).convert('xyxy').to(device)

        gt = dataset.get_groundtruth(image_ids[0], evaluation=True).to(device)

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
        pd_rel_dists = predictions.get_field('pred_rel_scores').tolist()

        # pd_rel_dists, pd_rel_labels = all_rel_prob.max(-1)
        # pd_rel_dists = []
        # for i in range(pd_rel_labels.shape[0]):
        #     pd_rel_dists.append(predictions.get_field('pred_rel_scores')[i][1:].max(0)[1].item() + 1)
        pd_labels = predictions.get_field('pred_labels').tolist()
        
        # get iou between gt and pd
        ious = boxlist_iou(gt, predictions)

        ##################
        # internal trans #
        ##################
        gt_rels_count += gt_rels.shape[0]
        for i in range(gt_rels.shape[0]):
            gt_s_idx = gt_rels[i][0]
            gt_o_idx = gt_rels[i][1]
            gt_r_label = gt_rels[i][2]-1
            gt_s_label = gt_labels[gt_s_idx]
            gt_o_label = gt_labels[gt_o_idx]
            pd_r_dists_list = []
            for j in range(pd_rels.shape[0]):
                pd_s_idx = pd_rels[j][0]
                pd_o_idx = pd_rels[j][1]
                pd_s_label = int(pd_labels[pd_s_idx])
                pd_o_label = int(pd_labels[pd_o_idx])
                pd_r_dists = pd_rel_dists[j]
                s_iou = ious[gt_s_idx, pd_s_idx]
                o_iou = ious[gt_o_idx, pd_o_idx]
                if gt_s_label == pd_s_label and gt_o_label == pd_o_label and \
                    s_iou > 0.5 and o_iou > 0.5:
                    pd_r_dists_list.append(pd_r_dists[1:])
            if len(pd_r_dists_list) > 0:
                pd_r_dists = np.stack(pd_r_dists_list, axis=0)
                pd_r_dists = pd_r_dists.mean(axis=0)

                if gt_r_label != np.argmax(pd_r_dists):
                    r_sort = np.argsort(pd_r_dists)[::-1]
                    gt_r_idx = np.where(r_sort == gt_r_label.item())[0].item()

                    confusion_r_labels = r_sort[:gt_r_idx]
                    # we get then ratio of frequence of the triplet in original data over the frequence of the relation                    
                    ori_attr = stats['triplet_freq'][(gt_s_label.item(), gt_o_label.item(), gt_r_label.item()+1)] / stats['pred_freq'][gt_r_label.item()]
                    ori_rel = idx_to_label[str(gt_s_label.item())] + ' ' + idx_to_predicate[str(gt_r_label.item()+1)] + ' ' + idx_to_label[str(gt_o_label.item())]

                    counter_cat = []
                    for c_r_label in confusion_r_labels:
                        if c_r_label != 0:
                            if stats['triplet_freq'][(gt_s_label.item(), gt_o_label.item(), c_r_label+1)] == 0:
                                continue
                            new_attr = stats['triplet_freq'][(gt_s_label.item(), gt_o_label.item(), c_r_label+1)] / stats['pred_freq'][c_r_label]

                            new_cat = categories[idx_to_label[str(gt_s_label.item())] + ' ' + idx_to_predicate[str(c_r_label+1)] + ' ' + idx_to_label[str(gt_o_label.item())]]

                            if new_attr < ori_attr and new_cat not in counter_cat:
                                counter_cat.append(new_cat)
                                full_rel = idx_to_label[str(gt_s_label.item())] + ' ' + idx_to_predicate[str(c_r_label+1)] + ' ' + idx_to_label[str(gt_o_label.item())]
                                out_data[image_ids[0]].append([gt_s_idx.item(), gt_o_idx.item(), int(c_r_label+1)])
                                internal_trans_count += 1
                                # print('Transfered from %s to %s' % (ori_rel, full_rel))

        ##################
        # external trans #
        ##################
        for i in range(gt_no_rels.shape[0]):
            gt_s_idx = gt_no_rels[i][0]
            gt_o_idx = gt_no_rels[i][1]
            gt_r_label = gt_no_rels[i][2]
            gt_s_label = gt_labels[gt_s_idx]
            gt_o_label = gt_labels[gt_o_idx]
            pd_r_dists_list = []
            for j in range(pd_rels.shape[0]):
                pd_s_idx = pd_rels[j][0]
                pd_o_idx = pd_rels[j][1]
                pd_s_label = pd_labels[pd_s_idx]
                pd_o_label = pd_labels[pd_o_idx]
                pd_r_dists = pd_rel_dists[j]
                s_iou = ious[gt_s_idx, pd_s_idx]
                o_iou = ious[gt_o_idx, pd_o_idx]
                if gt_s_label == pd_s_label and gt_o_label == pd_o_label and \
                    s_iou > 0.5 and o_iou > 0.5:
                    # get iou between gt_s_idx and gt_o_idx
                    if ious[gt_s_idx, pd_o_idx] > 0.1:
                        pd_r_dists_list.append(pd_r_dists)
            if len(pd_r_dists_list) > 0:
                pd_r_dists = np.stack(pd_r_dists_list, axis=0)
                # TODO: use weighted average
                pd_r_dists = pd_r_dists.mean(axis=0)
                if gt_r_label != np.argmax(pd_r_dists):
                    r_sort = np.argsort(pd_r_dists)[::-1]
                    gt_r_idx = np.where(r_sort == gt_r_label.item())[0].item()
                    confusion_r_labels = r_sort[:gt_r_idx]
                    # re-ranking by attraction
                    attr_sort = [stats['triplet_freq'][(gt_s_label.item(), gt_o_label.item(), r_label)] / stats['pred_freq'][r_label-1] for r_label in confusion_r_labels]
                    confusion_r_labels = [x for _, x in sorted(zip(attr_sort, confusion_r_labels), key=lambda pair: pair[0], reverse=True)]

                    # counter_cat = []
                    for i, c_r_label in enumerate(confusion_r_labels):
                        if c_r_label != 0:
                            new_attr = stats['triplet_freq'][(gt_s_label.item(), gt_o_label.item(), c_r_label)]
                            if new_attr > 0:
                                # new_cat = categories[idx_to_label[str(gt_s_label.item())] + ' ' + idx_to_predicate[str(c_r_label)] + ' ' + idx_to_label[str(gt_o_label.item())]]
                                # if new_cat not in counter_cat:
                                #     counter_cat.append(new_cat)
                                out_data[image_ids[0]].append([gt_s_idx.item(), gt_o_idx.item(), int(c_r_label)])
                                external_trans_count += 1
                                full_rel = idx_to_label[str(gt_s_label.item())] + ' ' + idx_to_predicate[str(c_r_label)] + ' ' + idx_to_label[str(gt_o_label.item())]
                                # print('Adding new rel %s ' % (full_rel))
                                break

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
    process(config_file, output_file)
    