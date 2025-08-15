import os
import sys
import torch
import h5py
import json
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import cv2

from sgg_benchmark.structures.bounding_box import BoxList
from sgg_benchmark.structures.boxlist_ops import boxlist_iou

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class RelationDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, img_dir, transforms=None,
                filter_empty_rels=True, num_im=-1,
                filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False):
        """
        COCO format dataset for Scene Graph Generation
        Parameters:
            annotation_file: JSON file in COCO format with relationships (one file per split)
            img_dir: folder containing all images
            filter_empty_rels: True if we filter out images without relationships
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
        """
        
        self.flip_aug = flip_aug
        self.img_dir = img_dir
        self.annotation_file = annotation_file
        self.filter_non_overlap = filter_non_overlap
        self.filter_duplicate_rels = filter_duplicate_rels
        self.transforms = transforms
        
        # Load COCO format data
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build category mappings
        self._build_category_mappings()
        
        # Process COCO data to extract required information
        self._process_coco_data(num_im, filter_empty_rels)
    
    def _build_category_mappings(self):
        """Build category mappings from COCO categories"""
        # Extract categories
        categories = self.coco_data['categories']
        
        # Build object class mappings
        self.id_to_classes = {}
        self.classes_to_id = {}
        for cat in categories:
            self.id_to_classes[cat['id']] = cat['name']
            self.classes_to_id[cat['name']] = cat['id']
        
        # Add background class
        self.id_to_classes[0] = '__background__'
        self.classes_to_id['__background__'] = 0
        
        # Create indexed mappings (sorted by id for consistency)
        self.ind_to_classes = [self.id_to_classes[i] for i in sorted(self.id_to_classes.keys())]

        rel_categories = self.coco_data['rel_categories'] if 'rel_categories' in self.coco_data else []

        # Create indexed mappings with background class at index 0
        self.ind_to_predicates = {0: '__background__'}
        for i, pred in enumerate(rel_categories):
            # Extract name if pred is a dictionary, otherwise use pred directly
            pred_name = pred['name'] if isinstance(pred, dict) and 'name' in pred else str(pred)
            self.ind_to_predicates[i + 1] = pred_name
            
        print(f"RelationDataset: Total relation classes: {len(self.ind_to_predicates)}")
        print(f"RelationDataset: Predicate mappings: {self.ind_to_predicates}")

    def _process_coco_data(self, num_im, filter_empty_rels):
        """Process COCO data to extract boxes, classes, and relationships"""
        images = self.coco_data['images']
        annotations = self.coco_data['annotations']
        
        # Build annotation lookup by image_id
        ann_by_img = defaultdict(list)
        for ann in annotations:
            ann_by_img[ann['image_id']].append(ann)
        
        # Build relationship lookup by image_id
        rel_by_img = defaultdict(list)
        if 'rel_annotations' in self.coco_data:
            for rel in self.coco_data['rel_annotations']:
                rel_by_img[rel['image_id']].append(rel)
        
        # Filter images based on criteria (no split needed since one file = one split)
        filtered_images = []
        if filter_empty_rels:
            # Only include images with relationships
            for img in images:
                img_id = img['id']
                if img_id in rel_by_img and len(rel_by_img[img_id]) > 0:
                    filtered_images.append(img)
        else:
            filtered_images = images
        
        # Apply image limits
        if num_im > 0:
            filtered_images = filtered_images[:num_im]
        
        # Process each image
        self.filenames = []
        self.img_info = []
        self.gt_boxes = []
        self.gt_classes = []
        self.relationships = []
        
        for img_data in filtered_images:
            img_id = img_data['id']
            
            # Get image info
            img_filename = os.path.join(self.img_dir, img_data['file_name'])
            if not os.path.exists(img_filename):
                continue
                
            self.filenames.append(img_filename)
            self.img_info.append({
                'width': img_data['width'],
                'height': img_data['height'],
                'image_id': img_id
            })
            
            # Get annotations for this image
            img_annotations = ann_by_img[img_id]
            
            # Process bounding boxes and classes
            boxes = []
            classes = []
            ann_id_to_idx = {}  # Map annotation id to box index
            
            for idx, ann in enumerate(img_annotations):
                # Convert COCO bbox format [x, y, w, h] to [x1, y1, x2, y2]
                x, y, w, h = ann['bbox']
                box = [x, y, x + w, y + h]
                boxes.append(box)
                
                # Map category_id to our class index
                cat_id = ann['category_id']
                class_name = self.id_to_classes[cat_id]
                class_idx = self.ind_to_classes.index(class_name)
                classes.append(class_idx)
                
                ann_id_to_idx[ann['id']] = idx
            
            # Process relationships
            relations = []
            img_relationships = rel_by_img.get(img_id, [])
            for rel in img_relationships:
                subj_id = rel['subject_id']
                obj_id = rel['object_id']
                predicate = rel['predicate_id'] + 1  # Add 1 to account for background class at index 0
                
                # Map to box indices
                if subj_id in ann_id_to_idx and obj_id in ann_id_to_idx:
                    subj_idx = ann_id_to_idx[subj_id]
                    obj_idx = ann_id_to_idx[obj_id]
                    
                    # Ensure predicate is within valid range
                    if predicate < len(self.ind_to_predicates):
                        relations.append([subj_idx, obj_idx, predicate])
            
            # Apply overlap filtering if needed
            if self.filter_non_overlap and len(relations) > 0:
                boxes_array = np.array(boxes, dtype=np.float32)
                relations_array = np.array(relations, dtype=np.int32)
                
                # Check overlaps
                boxes_tensor = torch.from_numpy(boxes_array)
                boxes_obj = BoxList(boxes_tensor, (img_data['width'], img_data['height']), 'xyxy')
                inters = boxlist_iou(boxes_obj, boxes_obj)
                rel_overs = inters[relations_array[:, 0], relations_array[:, 1]]
                inc = np.where(rel_overs > 0.0)[0]
                
                if inc.size > 0:
                    relations = [relations[i] for i in inc]
                else:
                    # Skip this image if no overlapping relationships
                    self.filenames.pop()
                    self.img_info.pop()
                    continue
            
            # Store boxes in original COCO format (no scaling needed)
            if len(boxes) > 0:
                boxes_array = np.array(boxes, dtype=np.float32)
                self.gt_boxes.append(boxes_array)
            else:
                self.gt_boxes.append(np.zeros((0, 4), dtype=np.float32))
            
            self.gt_classes.append(np.array(classes, dtype=np.int32))
            if len(relations) > 0:
                relations_array = np.array(relations, dtype=np.int32)
                # Debug predicate range
                if len(relations_array) > 0:
                    pred_range = (relations_array[:, 2].min(), relations_array[:, 2].max())
                    if pred_range[1] >= len(self.ind_to_predicates):
                        print(f"WARNING: Predicate index {pred_range[1]} is out of range. Max allowed: {len(self.ind_to_predicates)-1}")
                self.relationships.append(relations_array)
            else:
                self.relationships.append(np.zeros((0, 3), dtype=np.int32))
    
    def __getitem__(self, index):
            
        flip_img = (random.random() > 0.5) and self.flip_aug
        
        target = self.get_groundtruth(index, flip_img)
        
        img = cv2.imread(self.filenames[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if flip_img:
            img = Image.fromarray(img)
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
            img = np.array(img)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        target.add_field("image_path", self.filenames[index], is_triplet=True)
        
        return img, target, index
    
    def get_groundtruth(self, index, flip_img=False, evaluation=False):
        img_info = self.get_img_info(index)
        w, h = img_info['width'], img_info['height']
        
        # Use boxes directly (no scaling needed for COCO format)
        box = torch.from_numpy(self.gt_boxes[index]).reshape(-1, 4)  # guard against no boxes
        
        if flip_img:
            new_xmin = w - box[:, 2]
            new_xmax = w - box[:, 0]
            box[:, 0] = new_xmin
            box[:, 2] = new_xmax
        
        target = BoxList(box, (w, h), 'xyxy')
        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        
        relation = self.relationships[index].copy()
        if self.filter_duplicate_rels and len(relation) > 0:
            # Filter out dupes!
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)
        
        # Add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
        target.add_field("relation", relation_map, is_triplet=True)
        
        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation))
            target.add_field("image_path", self.filenames[index], is_triplet=False)
        else:
            target = target.clip_to_image(remove_empty=True)
        
        return target
    
    def get_img_info(self, index):
        return self.img_info[index]
    
    def get_custom_imgs(self, path):
        """Same implementation as VGDataset for custom evaluation"""
        self.custom_files = []
        self.img_info = []
        if not os.path.exists(path):
            return
        if os.path.isdir(path):
            files = os.listdir(path)
            img_exts = ['.jpg', '.jpeg', '.png']
            if not any([f.endswith(tuple(img_exts)) for f in files]):
                return
            for file_name in os.listdir(path):
                self.custom_files.append(os.path.join(path, file_name))
                img = Image.open(os.path.join(path, file_name)).convert("RGB")
                self.img_info.append({'width': int(img.width), 'height': int(img.height), 'image_id': str(file_name.split('.')[0])})
        if os.path.isfile(path):
            file_list = json.load(open(path))
            for file in file_list:
                self.custom_files.append(file)
                img = Image.open(file).convert("RGB")
                self.img_info.append({'width': int(img.width), 'height': int(img.height), 'image_id': str(file.split('/')[-1].split('.')[0])})
    
    def get_statistics(self):
        """Compute statistics for the dataset"""
        fg_matrix, bg_matrix, predicate_new_order, predicate_new_order_count, pred_freq, triplet_freq, pred_weight = get_relation_statistics(
            dataset=self, must_overlap=True)
        
        eps = 1e-3
        bg_matrix += 1
        fg_sum = fg_matrix.sum(2)[:, :, None]
        pred_dist = np.log(np.where(fg_sum > 0, fg_matrix / fg_sum, 1e-10) + eps)
        
        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'predicate_new_order': predicate_new_order,
            'predicate_new_order_count': predicate_new_order_count,
            'pred_freq': pred_freq,
            'triplet_freq': triplet_freq,
            'pred_weight': pred_weight,
        }
        
        return result
    
    def __len__(self):
        return len(self.filenames)

def get_relation_statistics(dataset, must_overlap=True):
    """
    Compute statistics for RelationDataset (COCO format)
    Similar to get_VG_statistics but works with RelationDataset
    """
    num_obj_classes = len(dataset.ind_to_classes)
    num_rel_classes = len(dataset.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)
    
    for ex_ind in range(len(dataset)):
        gt_classes = dataset.gt_classes[ex_ind].copy()
        gt_relations = dataset.relationships[ex_ind].copy()
        gt_boxes = dataset.gt_boxes[ex_ind].copy()
        
        if len(gt_relations) == 0 or len(gt_classes) == 0:
            continue
        
        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
            fg_matrix[o1, o2, gtr] += 1
        
        # For the background, get all of the things that overlap.
        if len(gt_boxes) > 0:
            o1o2_total = gt_classes[np.array(box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
            for (o1, o2) in o1o2_total:
                bg_matrix[o1, o2] += 1
    
    # Compute predicate statistics
    stats_pred = {i: 0 for i in range(num_rel_classes)}
    for k in fg_matrix:
        for p in k:
            for i, x in enumerate(p):
                stats_pred[i] += x
    
    pred_freq = [stats_pred[i] / sum(stats_pred.values()) if sum(stats_pred.values()) > 0 else 0 for i in range(num_rel_classes)]
    
    # Weight is the inverse frequency normalized by the median
    pred_weights = torch.tensor(np.sum(fg_matrix, axis=(0, 1)))
    pred_weights[0] = -1.0
    non_zero_weights = pred_weights[pred_weights > 0]
    if len(non_zero_weights) > 0:
        pred_weights = (1./pred_weights) * torch.median(non_zero_weights)
    
    # Add background value
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
                triplet = (i, j, k)
                freq = fg_matrix[i, j, k]
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
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

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
            logger.debug("Wrong image size for %s", filename)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
    with open(image_file, 'w') as outfile:  
        json.dump(data, outfile)
