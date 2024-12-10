import torch
import numpy as np
import time

from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.structures.image_list import to_image_list
from sgg_benchmark.config import cfg

from sgg_benchmark.data.build import build_transforms
from sgg_benchmark.utils.logger import setup_logger
from sgg_benchmark.data import get_dataset_statistics

import cv2
import seaborn as sns
import os

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

class SGG_Model(object):
    def __init__(self, config, weights, dcs=100, tracking=False, rel_conf=0.1, box_conf=0.5, show_fps=True) -> None:
        cfg.merge_from_file(config)
        cfg.TEST.CUSTUM_EVAL = True
        cfg.OUTPUT_DIR = os.path.dirname(config)

        # to force SGDET mode /!\ careful though, if the model hasn't been trained in sgdet mode, this will break the code
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = False

        cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = dcs
        # cfg.MODEL.BACKBONE.NMS_THRESH = 0.267
        self.cfg = cfg
        self.show_fps = show_fps

        # for visu
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.text_padding = 2  # Padding around the text

        self.stats = get_dataset_statistics(self.cfg)

        self.obj_class_colors = sns.color_palette('Paired', len(self.stats['obj_classes'])+2)
        # to cv2 format
        self.obj_class_colors = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in self.obj_class_colors]

        logger = setup_logger("sgg_demo")
        logger.remove()
        
        self.model = None
        self.model_weights = weights
        self.checkpointer = None
        self.device = None
        self.tracking = tracking
        self.last_time = 0

        self.rel_conf = rel_conf
        self.box_conf = box_conf

        # can choose between BYTETracker or OCSORT, in my experience OCSORT works a little bit better
        if self.tracking:
            from boxmot import OcSort

            self.tracker = OcSort(
                per_class=True,
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
        self.model.eval()

        self.pre_time_bench = []
        self.detec_time_bench = []
        self.post_time_bench = []

    def load_model(self):
        self.model = build_detection_model(self.cfg)
        self.model.to(self.cfg.MODEL.DEVICE)

        self.checkpointer = DetectronCheckpointer(self.cfg, self.model)
        # last_check = self.checkpointer.get_checkpoint_file()
        # if last_check == "":
        #     last_check = self.cfg.MODEL.WEIGHT
        ckpt = self.checkpointer.load(self.model_weights)
        self.device = torch.device(self.cfg.MODEL.DEVICE)

        if self.cfg.MODEL.BACKBONE.TYPE == "yolov8world":
            names = self.stats['obj_classes'].values()
            self.model.backbone.load_txt_feats(names)

    def predict(self, image, visu_type='image'):
        self.model.roi_heads.eval()
        self.model.backbone.eval()

        out_img = image.copy()
        self.last_time = time.time()
        img_list, _ = self._pre_processing(image)
        img_list = img_list.to(self.device)
        targets = None
        pre_process_time =(time.time()-self.last_time)*1000
        self.pre_time_bench.append(pre_process_time)
        
        with torch.no_grad():
            t_start = time.time()
            predictions = self.model(img_list, targets)
            det_time = time.time()-t_start # in second
            det_time *= 1000 # in milisecond
            self.detec_time_bench.append(det_time)
        
        t_start2 = time.time()
        bboxes, rels = self._post_process2(predictions[0], orig_size=image.shape[:2], rel_threshold=self.rel_conf)
        bboxes = bboxes.cpu().numpy()
        rels = rels.cpu().numpy()
        post_process_time = time.time()
        
        print("Objects detected: ", len(bboxes))
        print("Relationships detected: ", len(rels))

        # update tracker
        if self.tracking and len(bboxes) > 0:
            # check if there is bbox to track
            if len(bboxes) > 0:
                tracks = self.tracker.update(bboxes, image)
                # add one dim to bboxes
                bboxes = np.concatenate((bboxes, np.zeros((len(bboxes), 1))), axis=1)
                if len(tracks) > 0:
                    for i, cur_id in enumerate(tracks[:,7]):
                        cur_id = int(cur_id)
                        # update the track id in the predictions
                        class_label = str(int(bboxes[cur_id][5].item()))
                        
                        bboxes[cur_id][6] = int(class_label + str(int(tracks[i][4]))) # we track by class, 140, 141, 142, ...
                        # update the box coordinates
                        bboxes[cur_id][0] = tracks[i][0]
                        bboxes[cur_id][1] = tracks[i][1]
                        bboxes[cur_id][2] = tracks[i][2]
                        bboxes[cur_id][3] = tracks[i][3]
        if visu_type == 'video':
            out_img = self.draw_full_graph(out_img, bboxes, rels)
               # Assuming out_img is the image and predictions is the dictionary containing the predictions
            image_height, image_width = out_img.shape[:2]
            
            # Calculate the font scale based on the image width
            max_text_width = 0.2 * image_width
            font_scale = max_text_width / 250  # Adjust the divisor to fine-tune the font size
            
            # Calculate the height of the text
            (text_width, text_height), baseline = cv2.getTextSize("Sample Text", cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)
            text_height += baseline
            
            # Calculate the positions for the text
            positions = {
                "fps": (10, text_height * 1),
                "objects": (10, image_height - text_height * 1 - 100),
                "relationships": (10, image_height - text_height * 2 - 120),
                "detection": (10, text_height * 2 + 10),
                "pre_process": (10, text_height * 3 + 10),
                "post_process": (10, text_height * 4 + 10)
            }
            
            # Draw the text on the image
            if self.show_fps:
                true_fps = 1/(post_process_time-self.last_time) #({1/(time.time()-self.last_time):.2f})
                cv2.putText(out_img, f"FPS: {true_fps:.2f}", positions["fps"], cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), 2)
            
            # cv2.putText(out_img, f"Objects: {len(bboxes)}", positions["objects"], cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), 2)
            # cv2.putText(out_img, f"Relationships: {len(rels)}", positions["relationships"], cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), 2)
            
            cv2.putText(out_img, f"Detection: {det_time:.2f}ms", positions["detection"], cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), 2)

            # in milisecond
            post_process_time = post_process_time - t_start2
            post_process_time *= 1000
            self.post_time_bench.append(post_process_time)

            cv2.putText(out_img, f"Pre process: {pre_process_time:.2f}ms", positions["pre_process"], cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), 2)

            cv2.putText(out_img, f"Post process: {post_process_time:.2f}ms", positions["post_process"], cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), 2)

            # Convert the image to RGB
            out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
            return out_img, None
        
        elif visu_type == 'image':
            graph_img = self.visualize_graph(rels, bboxes, image.shape[:2])
            out_img = self.draw_boxes_image(bboxes, out_img)

            return out_img, graph_img
        
        return predictions, None
    
    def draw_boxes_image(self, bboxes, out_img):
        bbox_labels = [self.stats['obj_classes'][int(b[5])] for b in bboxes]

        for i, bbox in enumerate(bboxes):
            bbox = [int(b) for b in bbox[:4]]
            label = bbox_labels[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_padding = 2  # Padding around the text
            
            # Draw bounding box
            cv2.rectangle(out_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
            
            text = f"{str(i)}_{label}"
            
            # Calculate text size (width, height) and baseline
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # Calculate rectangle coordinates for the background
            rect_start = (bbox[0], bbox[1] - text_height - text_padding )
            rect_end = (bbox[0] + text_width + text_padding * 2, bbox[1] + text_padding )
            
            # Draw background rectangle
            cv2.rectangle(out_img, rect_start, rect_end, (255, 0, 0), cv2.FILLED)
            
            # Draw text
            cv2.putText(out_img, text, (bbox[0] + text_padding, bbox[1] - text_padding ), font, font_scale, (255, 255, 255), font_thickness)
        
        return cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    
    def draw_bbox(self, img, bbox, label):
        # Convert bbox to integer
        bbox = [int(b) for b in bbox]

        color = self.obj_class_colors[self.stats['obj_classes'].index(label)]

        # Draw bounding box
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Determine the text to be drawn
        text = f"{label}"

        # Calculate text size (width, height) and baseline
        (text_width, text_height), baseline = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)

        # Calculate rectangle coordinates such that the rectangle is inside the box, top left
        rect_start = (bbox[0], bbox[1] - text_height - 2 * self.text_padding)
        rect_end = (bbox[0] + text_width + 2 * self.text_padding, bbox[1])
        # if negative, move the rectangle to the left
        if rect_start[0] < 0:
            rect_start = (0, rect_start[1])
            rect_end = (text_width + 2 * self.text_padding, rect_end[1])
        if rect_end[0] > img.shape[1]:
            rect_start = (img.shape[1] - text_width - 2 * self.text_padding, rect_start[1])
            rect_end = (img.shape[1], rect_end[1])
        if rect_start[1] < 0:
            rect_start = (rect_start[0], 0)
            rect_end = (rect_end[0], text_height + 2 * self.text_padding)
        if rect_end[1] > img.shape[0]:
            rect_start = (rect_start[0], img.shape[0] - text_height - 2 * self.text_padding)
            rect_end = (rect_end[0], img.shape[0])

        # Draw background rectangle
        cv2.rectangle(img, rect_start, rect_end, color, cv2.FILLED)

        # Draw text
        cv2.putText(img, text, (rect_start[0] + self.text_padding, rect_end[1] - self.text_padding), self.font, self.font_scale, (255, 255, 255), self.font_thickness)

        # Return coordinates of the center of the bbox
        return (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2

    def draw_full_graph(self, img, bboxes, rels):
        # Convert bboxes and rels to CPU and then to NumPy arrays
        # bboxes is of type [x1, y1, x2, y2, score, class, id]
        
        # Precompute class labels
        bbox_labels = [self.stats['obj_classes'][int(b[5])] for b in bboxes]
        color = self.obj_class_colors[-1]
        
        for s, o, r, _ in rels:
            s,o,r = int(s), int(o), int(r)
            if len(bboxes[0]) > 6:
                subj = f"{bboxes[s][6]}_{bbox_labels[s]}"
                obj = f"{bboxes[o][6]}_{bbox_labels[o]}"
            else:
                subj = bbox_labels[s]
                obj = bbox_labels[o]
        
            #color = self.obj_class_colors[int(bboxes[s][5])]

            c_sub = self.draw_bbox(img, bboxes[s][:4], subj)
            c_obj = self.draw_bbox(img, bboxes[o][:4], obj)
        
            # Draw the relation between center of sub c_sub and center of obj c_obj
            cv2.line(img, c_sub, c_obj, color, 2)
        
            r_label = self.stats['rel_classes'][r]
            font_scale = 0.5

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(r_label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 1)
            # Draw background
            rect_start = ((c_sub[0] + c_obj[0]) // 2-2, ((c_sub[1] + c_obj[1]) // 2) - text_height - 2 * self.text_padding)
            rect_end = ((c_sub[0] + c_obj[0]) // 2 + text_width + 2 * self.text_padding, (c_sub[1] + c_obj[1]) // 2)

            # draw a rectange with rounded corners
            self.draw_rounded_rectangle(img, rect_start, rect_end, color, cv2.FILLED, 5)
            
            # Draw the relation label
            cv2.putText(img, r_label, ((c_sub[0] + c_obj[0]) // 2, (c_sub[1] + c_obj[1]) // 2 - 5), cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC, font_scale, (255, 255, 255), 1)

        return img
    
    def draw_rounded_rectangle(self, img, top_left, bottom_right, color, thickness, radius):
        # Draw the four straight edges
        cv2.rectangle(img, (top_left[0] + radius, top_left[1]), (bottom_right[0] - radius, bottom_right[1]), color, thickness)
        cv2.rectangle(img, (top_left[0], top_left[1] + radius), (bottom_right[0], bottom_right[1] - radius), color, thickness)

        # Draw the four rounded corners
        cv2.ellipse(img, (top_left[0] + radius, top_left[1] + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (bottom_right[0] - radius, top_left[1] + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (bottom_right[0] - radius, bottom_right[1] - radius), (radius, radius), 0, 0, 90, color, thickness)
        cv2.ellipse(img, (top_left[0] + radius, bottom_right[1] - radius), (radius, radius), 90, 0, 90, color, thickness)

    def visualize_graph(self, rels, bboxes, color='blue'):
        bbox_labels = [self.stats['obj_classes'][int(b[5])] for b in bboxes]
        G = nx.MultiDiGraph()
        for i, r_label in enumerate(rels[:, 2]):
            label_rel = self.stats['rel_classes'][int(r_label)]
            r = rels[i]
            # to int
            r = r.astype(int)
            subj = str(r[0])+'_'+bbox_labels[int(r[0])]
            obj = str(r[1])+'_'+bbox_labels[int(r[1])]
            G.add_edge(str(subj), str(obj), label=label_rel, color=color)

        # draw networkx graph with graphviz, display edge labels
        G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
        G.graph['graph'] = {'scale': '2'}
        G.graph['node'] = {'shape': 'rectangle'}
        # all graph color to blue
        G.graph['edge']['color'] = color
        G.graph['node']['color'] = color

        img_graph = to_agraph(G)
        # Layout the graph
        img_graph.layout('dot')

        # Draw the graph directly to a byte array
        png_byte_array = img_graph.draw(format='png', prog='dot')

        # Convert the byte array to an OpenCV image without redundant conversion
        img_cv2 = cv2.imdecode(np.frombuffer(png_byte_array, np.uint8), cv2.IMREAD_COLOR)

        return img_cv2
    
    def nice_plot(self, img, graph):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        graph = cv2.cvtColor(graph, cv2.COLOR_BGR2RGB)

        out_img = np.zeros((img.shape[0]+graph.shape[0], max(img.shape[1], graph.shape[1]), 3), dtype=np.uint8)
        out_img[:img.shape[0], :img.shape[1]] = img
        out_img[img.shape[0]:, :graph.shape[1]] = graph
        
        return out_img


    def _pre_processing(self, image):
        # to cv2 format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # reshape to 480,640
        image = cv2.resize(image, (640, 640))

        target = torch.LongTensor([-1])
        transform = build_transforms(self.cfg, is_train=False)

        image, target = transform(image, target)
        # image = image[None,:] # add batch dimension
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)

        return image_list, target
    
    def _post_process(self, boxlist, rel_threshold=0.1, box_thres=0.1, orig_size=(640,640)):
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
        # filter by box thres
        sortedid = [i for i in sortedid if boxlist.get_field('pred_scores')[i] > box_thres]
        id2sorted = {v: k for k, v in enumerate(sortedid)}

        for i in sortedid:
            current_dict['bbox'].append([int(round(b)) for b in xyxy_bbox[i].tolist()])
            current_dict['bbox_labels'].append(boxlist.get_field('pred_labels')[i].item())
            current_dict['bbox_scores'].append(boxlist.get_field('pred_scores')[i].item())

        current_dict['bbox_labels'] = [c for c in current_dict['bbox_labels']]
        
        # transform bbox, bbox_labels and bbox_scores to a single tensor of shape (N, 6)
        bboxes_tensor = torch.cat([torch.tensor(current_dict['bbox']), torch.tensor(current_dict['bbox_scores']).unsqueeze(1), torch.arange(len(current_dict['bbox_labels'])).unsqueeze(1)], dim=1)

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
            rel_s = boxlist.get_field('pred_rel_scores')[i][1:].max(0)[0].item()
            # rel score is sub_score * obj_score * rel_score
            rel_s = rel_s * boxlist.get_field('pred_scores')[old_pair[0]].item() * boxlist.get_field('pred_scores')[old_pair[1]].item()
            current_dict['rel_scores'].append(rel_s)
            current_dict['rel_pairs'].append([id2sorted[old_pair[0]], id2sorted[old_pair[1]]])
        current_dict['bbox_labels'] = [self.stats['obj_classes'][i] for i in current_dict['bbox_labels']]
        current_dict['rel_labels'] = [self.stats['rel_classes'][i] for i in current_dict['rel_labels']]

        current_dict['yolo_bboxes'] = bboxes_tensor

        return current_dict
    
    def get_sorted_bbox_mapping(self, score_list):
        sorted_scoreidx = sorted([(s, i) for i, s in enumerate(score_list)], reverse=True)
        sorted2id = [item[1] for item in sorted_scoreidx]
        id2sorted = [item[1] for item in sorted([(j,i) for i, j in enumerate(sorted2id)])]
        return sorted2id, id2sorted
    
    def _post_process2(self, boxlist, rel_threshold=0.1, box_thres=0.1, orig_size=(640,640)):
        height, width = orig_size
        boxlist = boxlist.resize((width, height))
        
        xyxy_bbox = boxlist.bbox

        rel_scores = boxlist.get_field('pred_rel_scores')
        pairs = boxlist.get_field('rel_pair_idxs')
        rel_labels = boxlist.get_field('pred_rel_labels')
        
        # Remove the background and get the max scores
        rel_scores = rel_scores[:, 1:].max(dim=1)[0]

        # rel_score = rel_score * subj_score * obj_score
        rel_scores = rel_scores * boxlist.get_field('pred_scores')[pairs[:, 0]] * boxlist.get_field('pred_scores')[pairs[:, 1]]
        
        # Stack the pairs, labels, and rel_scores
        filtered_rels = torch.cat((pairs.int(), rel_labels.unsqueeze(1).int(), rel_scores.unsqueeze(1)), dim=1)

        filtered_rels = filtered_rels[filtered_rels[:, 3] > rel_threshold]
        
        if filtered_rels.size(0) == 0:
            return torch.tensor([]), torch.tensor([])
        
        # Get unique values in the filtered_rels[:, :2]
        unique_values = torch.unique(filtered_rels[:, :2].reshape(-1))
        
        # Get corresponding boxes and scores
        bboxes = xyxy_bbox[unique_values.long()].int()
        scores = boxlist.get_field('pred_scores')[unique_values.long()]
        labels = boxlist.get_field('pred_labels')[unique_values.long()].int()
        
        # Combine boxes, scores, and labels into a single tensor
        bboxes_tensor = torch.cat([bboxes, scores.unsqueeze(1), labels.unsqueeze(1)], dim=1)
        
        # Modify the subj, obj to match the new indexes
        subj_indices = torch.searchsorted(unique_values, filtered_rels[:, 0])
        obj_indices = torch.searchsorted(unique_values, filtered_rels[:, 1])
        filtered_rels[:, 0] = subj_indices
        filtered_rels[:, 1] = obj_indices

        return bboxes_tensor, filtered_rels
    
    def post_process_rels(self, boxlist):
        rel_scores = boxlist.get_field('pred_rel_scores')
        pairs = boxlist.get_field('rel_pair_idxs')
        labels = boxlist.get_field('pred_rel_labels')
        
        # Remove the background and get the max scores
        rel_scores = rel_scores[:, 1:].max(dim=1)[0]

        # rel_score = rel_score * subj_score * obj_score
        # rel_scores = rel_scores * boxlist.get_field('pred_scores')[pairs[:, 0]] * boxlist.get_field('pred_scores')[pairs[:, 1]]
        
        # Stack the pairs, labels, and rel_scores
        all_rels = torch.cat((pairs.int(), labels.unsqueeze(1).int(), rel_scores.unsqueeze(1)), dim=1)
        
        # Remove all entries with subj == obj
        #all_rels = all_rels[all_rels[:, 0] != all_rels[:, 1]]
        
        # Check if there are relations
        if all_rels.size(0) == 0:
            return all_rels

        return all_rels
    
    def get_latency(self):
        print(f"Preprocessing time: {np.mean(self.pre_time_bench):.2f} ms")
        print(f"Detection time: {np.mean(self.detec_time_bench):.2f} ms")
        print(f"Post processing time: {np.mean(self.post_time_bench):.2f} ms")
        print(f"Total time: {np.mean(self.pre_time_bench) + np.mean(self.detec_time_bench) + np.mean(self.post_time_bench):.2f} ms")