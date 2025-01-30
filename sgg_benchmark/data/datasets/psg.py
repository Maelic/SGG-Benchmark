import random
from collections import defaultdict

import numpy as np
import torch

import json
import cv2
import os

from sgg_benchmark.structures.bounding_box import BoxList
from tqdm import tqdm

class PSGDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            split,  # {"train", "test"}
            img_dir,
            ann_file,
            filter_empty_rels=True,
            filter_duplicate_rels=True,
            transforms=None,
            informative_file=None,
            all_bboxes: bool = True,  # load all bboxes (thing, stuff) for SG
            box_size=(640,640),
    ):
        self.ann_file = ann_file
        self.transforms = transforms
        self.filter_empty_rels = filter_empty_rels
        self.filter_duplicate_rels = filter_duplicate_rels
        self.img_prefix = img_dir

        self.box_size = box_size

        self.all_bboxes = all_bboxes
        self.split = split        

        dataset = self.__load_annotations__(ann_file)

        for d in dataset['data']:
            # NOTE: 0-index for object class labels
            # for s in d['segments_info']:
            #     s['category_id'] += 1

            # for a in d['annotations']:
            #     a['category_id'] += 1

            # NOTE: 1-index for predicate class labels
            for r in d['relations']:
                r[2] += 1

        # NOTE: Filter out images with zero relations. 
        # Comment out this part for competition files
        if self.filter_empty_rels:
            dataset['data'] = [
                d for d in dataset['data'] if len(d['relations']) != 0
            ]

        # Get split
        assert split in {'train', 'test', 'val', 'all'}
        if split == 'train':
            self.data = [
                d for d in dataset['data']
                if d['image_id'] not in dataset['test_image_ids']
            ]
            # slice 1000 images for validation
            self.data = self.data[1000:]
        elif split == 'test':
            self.data = [
                d for d in dataset['data']
                if d['image_id'] in dataset['test_image_ids']
            ]
        elif split == 'val':
            self.data = [
                d for d in dataset['data']
                if d['image_id'] not in dataset['test_image_ids']
            ]
            self.data = self.data[:1000]
        elif split == 'all':
            self.data = dataset['data']

        # Init image infos
        self.data_infos = []
        for d in self.data:
            self.data_infos.append({
                'filename': d['file_name'],
                'height': d['height'],
                'width': d['width'],
                'id': d['image_id'],
            })
        self.img_ids = [d['filename'] for d in self.data_infos]

        # Define classes, 0-index
        # NOTE: Class ids should range from 0 to (num_classes - 1)
        # original, with -merged and -other
        # self.THING_CLASSES = dataset['thing_classes']
        # self.STUFF_CLASSES = dataset['stuff_classes']
        # self.CLASSES = self.THING_CLASSES + self.STUFF_CLASSES

        # new, without -merged and -other
        # get path of ann_file
        anno_path = os.path.dirname(ann_file)
        with open(os.path.join(anno_path, 'obj_classes.txt'), 'r') as f:
            self.CLASSES = f.read().splitlines()
        
        self.PREDICATES = dataset['predicate_classes']

        label_to_idx = {label: idx+1 for idx, label in enumerate(self.CLASSES)}
        predicate_to_idx = {label: idx+1 for idx, label in enumerate(self.PREDICATES)}

        label_to_idx['__background__'] = 0
        predicate_to_idx['__background__'] = 0

        self.ind_to_classes = sorted(label_to_idx, key=lambda k: label_to_idx[k])
        self.ind_to_predicates = sorted(predicate_to_idx, key=lambda k: predicate_to_idx[k])
        self.categories = {i : self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        if informative_file != "" and os.path.exists(informative_file):
            self.informative_graphs = json.load(open(informative_file, 'r'))
        else:
            self.informative_graphs = None
    
    def get_img_info(self, index):
        return self.data_infos[index]

    def __load_annotations__(self, ann_file):
        with open(ann_file, 'r') as f:
            dataset = json.load(f)
        return dataset
    
    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, index):
        img_path = self.img_prefix + '/' + self.img_ids[index]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = self.get_groundtruth(index)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        target.add_field("image_path", img_path, is_triplet=True)

        return img, target, index

    def get_groundtruth(self, idx, evaluation=False):
        d = self.data[idx]

        # get img size
        w, h = d['width'], d['height']

        if self.all_bboxes:
            # NOTE: Get all the bbox annotations (thing + stuff)
            gt_bboxes = np.array([a['bbox'] for a in d['annotations']],
                                 dtype=np.float32)
            gt_labels = np.array([a['category_id'] for a in d['annotations']],
                                 dtype=np.int64)

        else:
            gt_bboxes = []
            gt_labels = []

            # FIXME: Do we have to filter out `is_crowd`?
            # Do not train on `is_crowd`,
            # i.e just follow the mmdet dataset classes
            # Or treat them as stuff classes?
            # Can try and train on datasets with iscrowd
            # and without and see the difference

            for a, s in zip(d['annotations'], d['segments_info']):
                # NOTE: Only thing bboxes are loaded
                if s['isthing']:
                    gt_bboxes.append(a['bbox'])
                    gt_labels.append(a['category_id'])

            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)

        # add 1 for bg
        gt_labels += 1

        gt_bboxes = torch.from_numpy(gt_bboxes).reshape(-1, 4)

        target = BoxList(gt_bboxes, (w, h), 'xyxy') # xyxy
        target.add_field("labels", torch.from_numpy(gt_labels.copy()))
        del gt_labels

        # Process relationship annotations
        gt_rels = d['relations'].copy()

        # Filter out dupes!
        if self.split == 'train' and self.filter_duplicate_rels:
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in gt_rels:
                all_rel_sets[(o0, o1)].append(r)
            gt_rels = [(k[0], k[1], np.random.choice(v))
                       for k, v in all_rel_sets.items()]
            gt_rels = np.array(gt_rels, dtype=np.int32)
        else:
            # for test or val set, filter the duplicate triplets,
            # but allow multiple labels for each pair
            all_rel_sets = []
            for (o0, o1, r) in gt_rels:
                if (o0, o1, r) not in all_rel_sets:
                    all_rel_sets.append((o0, o1, r))
            gt_rels = np.array(all_rel_sets, dtype=np.int32)

        # add relation to target
        num_box = len(gt_bboxes)
        relation_map = np.zeros((num_box, num_box), dtype=np.int64)
        for i in range(gt_rels.shape[0]):
            # If already exists a relation?
            if relation_map[int(gt_rels[i, 0]), int(gt_rels[i, 1])] > 0:
                if random.random() > 0.5:
                    relation_map[int(gt_rels[i, 0]),
                                 int(gt_rels[i, 1])] = int(gt_rels[i, 2])
            else:
                relation_map[int(gt_rels[i, 0]),
                             int(gt_rels[i, 1])] = int(gt_rels[i, 2])
                
        relation_map = torch.from_numpy(relation_map)
        target.add_field("relation", relation_map, is_triplet=True)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(gt_rels)) # for evaluation
            if self.informative_graphs is not None:
                target.add_field("informative_rels", self.informative_graphs[str(self.data_infos[idx]['id'])])
            target.add_field("image_path", self.img_prefix + '/' + self.img_ids[idx])
            return target
        else:
            target = target.clip_to_image(remove_empty=True)
            return target
        
    def get_statistics(self):
        fg_matrix, bg_matrix, predicate_new_order, predicate_new_order_count, pred_prop, triplet_freq, pred_weight = self.get_PSG_statistics()
        eps = 1e-3
        bg_matrix += 1
        # fg_matrix[:, :, 0] = bg_matrix
        fg_sum = fg_matrix.sum(2)[:, :, None]
        # Avoid division by zero by using np.where
        pred_dist = np.log(np.where(fg_sum > 0, fg_matrix / fg_sum, 1e-10) + eps)  # Use a small value if sum is zero

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'predicate_new_order': predicate_new_order,
            'predicate_new_order_count': predicate_new_order_count,
            'pred_freq': pred_prop,
            'triplet_freq': triplet_freq,
            'pred_weight': pred_weight
        }

        return result
    
    def get_PSG_statistics(self):
        num_obj_classes = len(self.CLASSES)+1
        num_rel_classes = len(self.PREDICATES)+1

        fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
        bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

        dat = PSGDataset('all', self.img_prefix, self.ann_file, filter_empty_rels=False, filter_duplicate_rels=False, informative_file="").data

        for d in tqdm(dat):
            gt_classes = np.array([a['category_id'] for a in d['annotations']])
            gt_relations =np.array(d['relations'])
            gt_boxes = np.array([a['bbox'] for a in d['annotations']])

            # For the foreground, we'll just look at everything
            o1o2 = gt_classes[gt_relations[:, :2]]
            for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
                fg_matrix[o1, o2, gtr] += 1
            # For the background, get all of the things that overlap.
            o1o2_total = gt_classes[np.array(box_filter(gt_boxes, must_overlap=True), dtype=int)]
            for (o1, o2) in o1o2_total:
                bg_matrix[o1, o2] += 1
        
        stats_pred = {i: 0 for i in range(num_rel_classes)}
        for k in fg_matrix:
            for p in k:
                for i, x in enumerate(p):
                    stats_pred[i] += x

        pred_freq = [stats_pred[i] / sum(stats_pred.values()) for i in range(num_rel_classes)]
        
        # weight is the inverse frequency normalized by the median
        pred_weights = torch.tensor(np.sum(fg_matrix, axis=(0, 1)))
        pred_weights[0] = -1.0
        pred_weights = (1./pred_weights) * torch.median(pred_weights)

        # add background value
        stats_pred[0] = len(bg_matrix.flatten())
        stats_pred = dict(sorted(stats_pred.items(), key=lambda x: x[1], reverse=True))
        predicate_new_order = list(stats_pred.keys())
        predicate_new_order_count = list(stats_pred.values())

        triplet_freq = {}

        # Compute the total count of all triplets
        total_count = fg_matrix.sum()
        # Loop over each element in the fg_matrix
        for i in range(fg_matrix.shape[0]):
            for j in range(fg_matrix.shape[1]):
                for k in range(fg_matrix.shape[2]):
                    # The triplet is (i, j, k)
                    triplet = (i, j, k)
                    # The frequency is the value in the fg_matrix divided by the total count
                    freq = fg_matrix[i, j, k] / total_count
                    # Add the triplet and its frequency to the dictionary
                    triplet_freq[triplet] = freq

        return fg_matrix, bg_matrix, predicate_new_order, predicate_new_order_count, pred_freq, triplet_freq, pred_weights

def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""

    overlaps = bbox_overlaps(boxes.astype(float), boxes.astype(float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=bool)
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

    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter