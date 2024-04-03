import torch
import json
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import numpy as np
import time

from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.structures.image_list import to_image_list
from sgg_benchmark.config import cfg

from sgg_benchmark.data.build import build_transforms
from sgg_benchmark.utils.logger import setup_logger

from ultralytics.engine.results import Boxes

import cv2

class SGG_Model(object):
    def __init__(self, config, dict_classes, tracking=False, logging_level="INFO") -> None:
        cfg.merge_from_file(config)
        cfg.freeze()
        self.cfg = cfg

        self.stats = json.load(open(dict_classes, 'r'))

        logger = setup_logger("sgg_demo")
        logger.remove()
        
        self.model = None
        self.checkpointer = None
        self.device = None
        self.tracking = tracking
        self.last_time = 0

        # can choose between BYTETracker or OCSORT, in my experience OCSORT works a little bit better
        if self.tracking:
            from boxmot import BYTETracker,OCSORT

            self.tracker = OCSORT(
                per_class=False,
                det_thresh=0,
                max_age=20,
                min_hits=1,
                asso_threshold=0.2,
                delta_t=2,
                asso_func='giou',
                inertia=0.2,
                use_byte=True,
            )
            
        self.load_model()
        self.model.roi_heads.eval()
        self.model.backbone.eval()

    def load_model(self):
        self.model = build_detection_model(self.cfg)
        self.model.to(self.cfg.MODEL.DEVICE)

        self.checkpointer = DetectronCheckpointer(self.cfg, self.model)
        last_check = self.checkpointer.get_checkpoint_file()
        if last_check == "":
            last_check = self.cfg.MODEL.WEIGHT
        _ = self.checkpointer.load(last_check)
        self.device = torch.device(self.cfg.MODEL.DEVICE)

    def predict(self, image, visu=False):
        out_img = image.copy()
        self.last_time = time.time()
        img_list, target = self._pre_processing(image)

        with torch.no_grad():
            targets = [target.to(self.device)]
            img_list = img_list.to(self.device)
            outputs, features = self.model.backbone(img_list.tensors, embed=True)
            proposals = self.model.backbone.postprocess(outputs, img_list.image_sizes)

            _, predictions, _ = self.model.roi_heads(features, proposals, targets, None, proposals)

        predictions = self._post_process_yolo(predictions[0], orig_size=img_list.image_sizes[0])
        
        # update tracker
        if self.tracking:
            predictions['track_id'] = [None for _ in range(len(predictions['bbox']))]
            # check if there is bbox to track
            yolo_bbox = predictions['yolo_bboxes'].data.numpy()
            if len(yolo_bbox) > 0:
                tracks = self.tracker.update(yolo_bbox, image)
                for i, cur_id in enumerate(tracks[:,6]):
                    cur_id = int(cur_id)
                    # update the track id in the predictions
                    predictions['track_id'][cur_id] = int(tracks[i][4])
                    # update the box
                    predictions['bbox'][cur_id] = tracks[i][:4]

            for i, track_id in enumerate(predictions['track_id']):
                if track_id is None:
                    # add dummy id to not break the code
                    predictions['track_id'][i] = max([p for p in predictions['track_id'] if p != None]) + 1

        if visu:
            for i, bbox in enumerate(predictions['bbox']):
                bbox = [int(b) for b in bbox]
                cv2.rectangle(out_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                if 'track_id' in predictions:
                    cv2.putText(out_img, f"{predictions['track_id'][i]}_{predictions['bbox_labels'][i]}", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(out_img, f"{predictions['bbox_labels'][i]}", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # show fps
                cv2.putText(out_img, f"FPS: {1/(time.time()-self.last_time):.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            img_graph = self.visualize_graph(predictions)
            return out_img, img_graph
        return predictions
    
    def visualize_graph(self, predictions, color='blue'):
        G = nx.MultiDiGraph()
        for i, r_label in enumerate(predictions['rel_labels']):
            r = predictions['rel_pairs'][i]
            if 'track_id' in predictions:
                subj = f"{predictions['track_id'][r[0]]}_{predictions['bbox_labels'][r[0]]}"
                obj = f"{predictions['track_id'][r[1]]}_{predictions['bbox_labels'][r[1]]}"
            else:
                subj = predictions['bbox_labels'][r[0]]
                obj = predictions['bbox_labels'][r[1]]
            G.add_edge(str(subj), str(obj), label=r_label)

        # draw networkx graph with graphviz, display edge labels
        G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
        G.graph['graph'] = {'scale': '2'}
        G.graph['node'] = {'shape': 'rectangle'}
        # all graph color to blue
        G.graph['edge']['color'] = color
        G.graph['node']['color'] = color
        # draw graph
        img_graph = to_agraph(G)
        img_graph.layout('dot')
        img_graph = img_graph.draw(format='png', args='-Gdpi=100')
        # to cv2
        img_graph = cv2.imdecode(np.fromstring(img_graph, np.uint8), cv2.IMREAD_COLOR)

        return img_graph

    def _pre_processing(self, image):
        # to cv2 format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # reshape to 480,640
        image = cv2.resize(image, (640, 480))

        target = torch.LongTensor([-1])
        transform = build_transforms(self.cfg, is_train=False)

        image, target = transform(image, target)
        # image = image[None,:] # add batch dimension
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)

        return image_list, target
    
    def _post_process_yolo(self, boxlist, rel_threshold=0.01, orig_size=(480,640)):
        height, width = orig_size
        boxlist = boxlist.resize((width, height))

        xyxy_bbox = boxlist.bbox
        # current sgg info
        current_dict = {'bbox': [], 
                        'bbox_labels': [], 
                        'bbox_scores': [], 
                        'rel_pairs': [], 
                        'rel_labels': [], 
                        'rel_scores': [], 
                        'rel_all_scores': []
        }
        
        # sort boxes based on confidence
        sortedid, id2sorted = self.get_sorted_bbox_mapping(boxlist.get_field('pred_scores').tolist())

        for i in sortedid:
            current_dict['bbox'].append([int(round(b)) for b in xyxy_bbox[i].tolist()])
            current_dict['bbox_labels'].append(boxlist.get_field('pred_labels')[i].item())
            current_dict['bbox_scores'].append(boxlist.get_field('pred_scores')[i].item())

        current_dict['bbox_labels'] = [c for c in current_dict['bbox_labels']]
        
        # transform bbox, bbox_labels and bbox_scores to a single tensor of shape (N, 6)
        bboxes_tensor = torch.cat([torch.tensor(current_dict['bbox']), torch.tensor(current_dict['bbox_scores']).unsqueeze(1), torch.arange(len(current_dict['bbox'])).unsqueeze(1)], dim=1)

        bboxes = Boxes(bboxes_tensor, (width, height))

        # sorted relationships
        rel_sortedid, _ = self.get_sorted_bbox_mapping(boxlist.get_field('pred_rel_scores')[:,1:].max(1)[0].tolist())

        # remove all relationships with score < rel_threshold
        rel_sortedid = [i for i in rel_sortedid if boxlist.get_field('pred_rel_scores')[i][1:].max(0)[0] > rel_threshold]

        # sorted rel
        for i in rel_sortedid:
            old_pair = boxlist.get_field('rel_pair_idxs')[i].tolist()
            # don't add if the subject or object is not in the sortedid
            if old_pair[0] not in id2sorted or old_pair[1] not in id2sorted:
                continue
            
            current_dict['rel_labels'].append(boxlist.get_field('pred_rel_scores')[i][1:].max(0)[1].item() + 1)
            current_dict['rel_scores'].append(boxlist.get_field('pred_rel_scores')[i][1:].max(0)[0].item())
            current_dict['rel_pairs'].append([id2sorted[old_pair[0]], id2sorted[old_pair[1]]])

        current_dict['bbox_labels'] = [self.stats['idx_to_label'][str(i)] for i in current_dict['bbox_labels']]
        current_dict['rel_labels'] = [self.stats['idx_to_predicate'][str(i)] for i in current_dict['rel_labels']]

        current_dict['yolo_bboxes'] = bboxes

        return current_dict
    
    def get_sorted_bbox_mapping(self, score_list):
        sorted_scoreidx = sorted([(s, i) for i, s in enumerate(score_list)], reverse=True)
        sorted2id = [item[1] for item in sorted_scoreidx]
        id2sorted = [item[1] for item in sorted([(j,i) for i, j in enumerate(sorted2id)])]
        return sorted2id, id2sorted