import os
import sys
import logging
import torch
import h5py
import json
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import pickle
from sgg_benchmark.structures.bounding_box import BoxList
from sgg_benchmark.structures.boxlist_ops import cat_boxlist, split_boxlist
from sgg_benchmark.data.datasets.visual_genome import get_VG_statistics, get_VG_statistics_wo_sample

class GQADataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split, img_dir, dict_file, train_file, test_file, transforms=None, 
                 filter_empty_rels=True, num_im=-1, num_val_im=5000,
                 filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False, custom_eval=False, custom_path=''):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        # for debug
        # num_im = 10000
        # num_val_im = 4

        assert split in {'train', 'val', 'test'}
        self.cfg = cfg
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.train_file = train_file
        self.test_file = test_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        logger = logging.getLogger("maskrcnn_benchmark.dataset")
        self.logger = logger

        self.logger.info('\nwe change the gqa get ground-truth!\n')

        self.ind_to_classes, self.ind_to_predicates = load_info(dict_file) # contiguous 151, 51 containing __background__
        self.categories = {i : self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        if self.split == 'train':
            self.filenames, self.img_info, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
                self.train_file, self.split)
        else:
            self.filenames, self.img_info, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
                self.test_file, self.split)

        self.idx_list = list(range(len(self.filenames)))
        self.repeat_dict = None
        
    def __getitem__(self, index):
        if self.repeat_dict is not None:
            index = self.idx_list[index]
        img = Image.open(os.path.join(self.img_dir, self.filenames[index])).convert("RGB")
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('=' * 20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']),
                  ' ', str(self.img_info[index]['height']), ' ', '=' * 20)
        target = self.get_groundtruth(index, False)
        fp_name = self.filenames[index].split('/')[-1]
        targets_len = len(target)
        target.add_field("fp_name", fp_name)
        target.add_field("src_w", img.size[0])
        target.add_field("src_h", img.size[1])
        pre_compute_box = tgt_record = pre_comp_record = pre_comp_result = None
        if self.cfg.DATASETS.DIR_LOAD_PRECOMPUTE_DETECTION_BOX and (self.split == 'train'):
            pre_comp_result = torch.load(os.path.join(self.cfg.DATASETS.GQA_BBOX_DIR, fp_name))
        if pre_comp_result is not None:
            boxes_arr = torch.as_tensor(pre_comp_result['bbox']).reshape(-1, 4)
            pre_compute_box = BoxList(boxes_arr, img.size, mode='xyxy')
            tgt_record = target.remove_all_fields()
            target = cat_boxlist([target, pre_compute_box])
            pre_comp_record = {
                'pred_scores': pre_comp_result['pred_scores'],
                'pred_labels': pre_comp_result['pred_labels'],
                'predict_logits': pre_comp_result['predict_logits'],
                'labels': pre_comp_result['labels'],
            }
        if self.split == 'train' and (self.cfg.MODEL.INFER_TRAIN or self.cfg.DATASETS.INFER_BBOX):
            img, target = self.transforms(img, target)
            if pre_comp_record is not None:
                target = self.split_target(target, targets_len, len(pre_compute_box), tgt_record, pre_comp_record)
        elif self.split == 'train' and self.cfg.MODEL.STAGE == "stage1":
            img1, target1 = self.transforms(img, target)
            img2, target2 = self.transforms(img, target)
            if pre_comp_record is not None:
                target1 = self.split_target(target1, targets_len, len(pre_compute_box), tgt_record, pre_comp_record)
                target2 = self.split_target(target2, targets_len, len(pre_compute_box), tgt_record, pre_comp_record)
            return [img1, img2], [target1, target2], index
        elif self.transforms is not None:
            img, target = self.transforms(img, target)
            if pre_comp_record is not None:
                target = self.split_target(target, targets_len, len(pre_compute_box), tgt_record, pre_comp_record)
        
        return img, target, index

    @staticmethod
    def split_target(all_boxes, targets_len, pre_compute_len, tgt_record, pre_comp_record):
        resized_boxes = split_boxlist(all_boxes, (targets_len, targets_len + pre_compute_len))
        target = resized_boxes[0]
        pre_compute_box = resized_boxes[1]
        target.add_all_fields(tgt_record[0], tgt_record[1])
        pre_compute_box.add_field("pred_scores", pre_comp_record['pred_scores'])
        pre_compute_box.add_field("pred_labels", pre_comp_record['pred_labels'])
        pre_compute_box.add_field("predict_logits", pre_comp_record['predict_logits'])
        pre_compute_box.add_field("labels", pre_comp_record['labels'])
        target = (target, pre_compute_box)
        return target

    def get_img_info(self, index):
        return self.img_info[index]

    def get_statistics(self):
        _, bg_matrix, pred_matrix = get_VG_statistics(self, must_overlap=True)
        fg_matrix = get_VG_statistics_wo_sample(self)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)
        down_rate, generate_rate = None, None
        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'pred_matrix': pred_matrix,
            'data_length': len(self.idx_list),
            'down_rate': down_rate,
            'generate_rate': generate_rate,
            'repeat_dict': self.repeat_dict,
        }
        return result

    def get_custom_imgs(self, path):
        self.custom_files = []
        self.img_info = []
        for file_name in os.listdir(path):
            self.custom_files.append(os.path.join(path, file_name))
            img = Image.open(os.path.join(path, file_name)).convert("RGB")
            self.img_info.append({'width':int(img.width), 'height':int(img.height)})

    def get_groundtruth(self, index, evaluation=False, flip_img=False):
        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        box = self.gt_boxes[index]
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes

        if flip_img:
            new_xmin = w - box[:,2]
            new_xmax = w - box[:,0]
            box[:,0] = new_xmin
            box[:,2] = new_xmax
        target = BoxList(box, (w, h), 'xyxy') # xyxy

        tgt_labels = torch.from_numpy(self.gt_classes[index])
        target.add_field("labels", tgt_labels.long())

        relation = self.relationships[index].copy()  # (num_rel, 3)

        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
        target.add_field("relation", relation_map, is_triplet=True)
        target.add_field("image_index", index)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation))  # for evaluation
            return target
        else:
            target = target.clip_to_image(remove_empty=True)
            return target

    def __len__(self):
        return len(self.idx_list)



def get_GQA_statistics(img_dir, train_file, dict_file, must_overlap=True):
    train_data = GQADataset(split='train', img_dir=img_dir, train_file=train_file,
                           dict_file=dict_file, test_file=None, num_val_im=5000,
                           filter_duplicate_rels=False)
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

    for ex_ind in tqdm(range(len(train_data))):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
            fg_matrix[o1, o2, gtr] += 1
        # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes

def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:, :, :2], boxes2.reshape([1, num_box2, -1])[:, :, :2])  # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:, :, 2:], boxes2.reshape([1, num_box2, -1])[:, :, 2:])  # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter

def correct_img_info(img_dir, image_file):
    with open(image_file, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
    with open(image_file, 'w') as outfile:
        json.dump(data, outfile)

def load_info(dict_file):
    info = json.load(open(dict_file, 'r'))
    ind_to_classes = info['ind_to_classes']
    ind_to_predicates = info['ind_to_predicates']
    return ind_to_classes, ind_to_predicates

def load_graphs(data_json_file, split):
    data_info_all = json.load(open(data_json_file, 'r'))
    filenames = data_info_all['filenames_all']
    img_info = data_info_all['img_info_all']
    gt_boxes = data_info_all['gt_boxes_all']
    gt_classes = data_info_all['gt_classes_all']
    relationships = data_info_all['relationships_all']

    output_filenames = []
    output_img_info = []
    output_boxes = []
    output_classes = []
    output_relationships = []

    items = 0
    for filename, imginfo, gt_b, gt_c, gt_r in zip(filenames, img_info, gt_boxes, gt_classes, relationships):
        len_obj = len(gt_b)
        items += 1

        if split == 'val' or split == 'test':
            if items == 5580:
                continue

        if filename in ['285743.jpg', '61530.jpg', '285761.jpg', '150344.jpg', '286093.jpg',
                        '286065.jpg', '61564.jpg', '498098.jpg', '285665.jpg', '150417.jpg']:
            continue

        if len(gt_r) > 0 and len_obj > 0:
            output_filenames.append(filename)
            output_img_info.append(imginfo)
            output_boxes.append(np.array(gt_b))
            output_classes.append(np.array(gt_c))
            output_relationships.append(np.array(gt_r))


    if split == 'val':
        output_filenames = output_filenames[:5000]
        output_img_info = output_img_info[:5000]
        output_boxes = output_boxes[:5000]
        output_classes = output_classes[:5000]
        output_relationships = output_relationships[:5000]

    return output_filenames, output_img_info, output_boxes, output_classes, output_relationships


def load_image_filenames(img_dir, filenames, img_info):
    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info_new = []
    for i, img, info in enumerate(zip(filenames, img_info)):
        basename = img
        if basename in corrupted_ims:
            continue
        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            img_info_new.append(info)
    print("filename... ", len(fns))
    print("img_info... ", len(img_info_new))
    return fns, img_info_new