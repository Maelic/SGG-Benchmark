import torch
import torch.nn.functional as F
import numpy as np
from functools import reduce

from PIL import Image

from sgg_benchmark.utils.miscellaneous import intersect_2d, argsort_desc, bbox_overlaps
# from sentence_transformers import SentenceTransformer, util
import time
from sgg_benchmark.structures.bounding_box import BoxList

from PIL import Image, ImageDraw, ImageFont

from abc import ABC, abstractmethod

from collections import Counter
from transformers import AutoProcessor, AutoModel

# sim_model = SentenceTransformer('all-mpnet-base-v2',trust_remote_code=True) # clip-ViT-B-32

class SceneGraphEvaluation(ABC):
    def __init__(self, result_dict):
        super().__init__()
        self.result_dict = result_dict
 
    @abstractmethod
    def register_container(self, mode):
        print("Register Result Container")
        pass
    
    @abstractmethod
    def generate_print_string(self, mode):
        print("Generate Print String")
        pass

    def calculate(self, global_container, local_container, mode):
        pass


class BLIPScoreMatching(SceneGraphEvaluation):
    def __init__(self, result_dict):
        from lavis.models import load_model_and_preprocess
        from lavis.processors import load_processor
        super(BLIPScoreMatching, self).__init__(result_dict)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=self.device, is_eval=True)
        self.fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 12)

        self.confusion_triplets = Counter()
        self.deviation = []

        self.ids = 0
        self.triplets_scores = {}

        self.images_path = "/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/checkpoints/PSG/SGDET/react-yolov8m/images_boxes_blip/"
        self.out_path = "/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/checkpoints/PSG/SGDET/react-yolov8m/triplet_scores_blip.json"

        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)

    def register_container(self, mode):
        self.result_dict[mode + '_blip_score_matching'] = []
        self.result_dict[mode + '_blip_itm_matching'] = []
        self.result_dict[mode + '_blip_gt_match_cos'] = []

    def generate_print_string(self, mode):
        match_cos = str(self.result_dict[mode + '_blip_gt_match_cos'].count(1)) + " / " + str(len(self.result_dict[mode + '_blip_gt_match_cos']))
        result_str = 'SGG eval: '
        result_str += '    BLIP COSINE Score Matching: %.4f; ' % np.mean(self.result_dict[mode + '_blip_score_matching'])
        result_str += '    BLIP NUMBER OF GT MATCHES COSINE: %s; ' % match_cos
        result_str += ' for mode=%s, type=BLIP Score Matching.' % mode
        result_str += '\n'
        print(self.confusion_triplets.most_common(20))
        print("Mean Deviation: ", np.mean(self.deviation))
        # save to csv
        with open("/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/checkpoints/PSG/SGDET/react-yolov8m/confusion_triplets.csv", "w") as f:
            for key, value in self.confusion_triplets.items():
                f.write("%s,%s,%s\n"%(key[0],key[1],value))
        
        with open(self.out_path, "w") as f:
            json.dump(self.triplets_scores, f)

        return result_str
    
    def preprocess_img(self, image):
        img = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        return img
    
    def compute_similarity(self, text, image, method='itm'):
        if type(text) == str:
            text = [text]
        txt = []

        for t in text:
            #t = "a photo of a " + t
            txt.append(self.text_processors["eval"](t))
        images = torch.stack([image for item in txt]).squeeze(1).to(self.device)
        
        if method == 'cosine':
            with torch.no_grad():
                score = self.model({"image": images, "text_input": txt}, match_head='itc')
                score = score.cpu()
        elif method == 'itm':
            with torch.no_grad():
                itm_output = self.model({"image": images, "text_input": txt}, match_head="itm")
                itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
            score = itm_scores[:, 1].cpu()
        else:
            print("Not supported method")
            return None
        return score
    
    def calculate(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        fg_triplets = global_container['fg_triplets']

        gt_image = local_container['gt_image'] 

        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
        pred_scores = rel_scores[:,1:].max(1)

        # get predicted pairs which have two matching boxes
        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)
        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)

        #pred_triplet_boxes = pred_triplet_boxes[:100]
        # get matching box indices
        matching_boxes, gt_matching_boxes = compute_box_matches(gt_triplet_boxes, pred_triplet_boxes, iou_thresh=iou_thres)

        num_match = 0
        overall_score = 0
        

        orig_img = gt_image.copy()

        for l, i in enumerate(matching_boxes):
            # check if the subject and object labels are matching the gt
            if gt_triplets[gt_matching_boxes[l]][0] != pred_triplets[i][0] or gt_triplets[gt_matching_boxes[l]][2] != pred_triplets[i][2]:
                continue

            box_sub = pred_triplet_boxes[i][:4]
            box_obj = pred_triplet_boxes[i][4:]

            # get union box
            union_box = [min(box_sub[0], box_obj[0]), min(box_sub[1], box_obj[1]), max(box_sub[2], box_obj[2]), max(box_sub[3], box_obj[3])]

            # to int
            union_box = [int(i) for i in union_box]

            # crop the PIL Image
            union_image = gt_image.crop(union_box)

            image_features = self.preprocess_img(union_image)

            # compute score for GT triplet:
            gt_triplet = gt_triplets[gt_matching_boxes[l]]
            sub_str = global_container['ind_to_classes'][gt_triplet[0]]
            obj_str = global_container['ind_to_classes'][gt_triplet[2]]
            gt_triplet = sub_str + " "+ str(global_container['ind_to_predicates'][gt_triplet[1]]) + " "+ obj_str

            # get all other possible triplets with the same subject and object
            possible_triplets = fg_triplets[(fg_triplets[:, 0] == pred_triplets[i][0]) & (fg_triplets[:, 1] == pred_triplets[i][2])]
            possible_triplets = possible_triplets.tolist()
            # if pred_triplets[i][2] != pred_triplets[i][0]:
            #     possible_triplets.extend(fg_triplets[(fg_triplets[:, 0] == pred_triplets[i][2]) & (fg_triplets[:, 1] == pred_triplets[i][0])])
            
            possible_triplets = [str(global_container['ind_to_classes'][triplet[0]]) + " "+ str(global_container['ind_to_predicates'][triplet[2]]) + " "+ str(global_container['ind_to_classes'][triplet[1]]) for triplet in possible_triplets]
            # construct list of str
            # possible_triplets = [str(global_container['ind_to_classes'][gt_triplets[gt_matching_boxes[l]][0]]) + " "+ str(global_container['ind_to_predicates'][p]) + " "+ str(global_container['ind_to_classes'][gt_triplets[gt_matching_boxes[l]][2]]) for p in range(1, len(global_container['ind_to_predicates']))]
            # possible_triplets.remove(gt_triplet)

            triplet = str(global_container['ind_to_classes'][pred_triplets[i][0]]) + " "+ str(global_container['ind_to_predicates'][pred_triplets[i][1]]) + " "+ str(global_container['ind_to_classes'][pred_triplets[i][2]])

            text_list = [gt_triplet]
            # text_list.append(triplet)
            text_list.extend(possible_triplets)

            scores = self.compute_similarity(text_list, image_features, method='itm')

            # compute std
            std = torch.std(scores, dim=0).item()
            self.deviation.append(std)

            # get max indice
            max_ind = torch.argmax(scores, dim=0).item()

            if max_ind == 0:
                self.result_dict[mode + '_blip_gt_match_cos'].append(1)
            else:
                self.result_dict[mode + '_blip_gt_match_cos'].append(0)
                tuple_confused = (gt_triplet, text_list[max_ind])
                self.confusion_triplets.update([tuple_confused])
            
            self.triplets_scores[self.ids] = {t: float(s) for t, s in zip(text_list, scores.numpy())}

            # draw bounding boxes on image and save to folder
            dest_folder = self.images_path
            base_image = orig_img.copy()
            base_image = draw_boundig_boxes(base_image, [box_sub, box_obj], [sub_str, obj_str], color=(255, 0, 0), thickness=2)

            # save
            base_image.save(dest_folder + str(self.ids) + ".png")

            self.ids += 1

            num_match += 1
            overall_score += 0 #scores[1].item()

            # remove allocated memory
            del union_image
            del image_features
            # free memory
            torch.cuda.empty_cache()

        if num_match > 0:
            res = overall_score / num_match
        else:
            res = 0

        self.result_dict[mode + '_blip_score_matching'].append(res)

class SIGLIPScoreMatching(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SIGLIPScoreMatching, self).__init__(result_dict)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = "google/siglip-so400m-patch14-384" # "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14", "google/siglip-so400m-patch14-384", "facebook/metaclip-h14-fullcc2.5b", "Salesforce/blip2-opt-2.7b"
        self.fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 12)
        # self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        # self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id
        )
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        # self.processor = AutoProcessor.from_pretrained("facebook/metaclip-h14-fullcc2.5b")
        # self.model = AutoModel.from_pretrained("facebook/metaclip-h14-fullcc2.5b")
        # self.processor = AutoProcessor.from_pretrained(
        #     "Salesforce/blip2-opt-2.7b",
        #     torch_dtype=torch.bfloat16,
        #     _attn_implementation="flash_attention_2",
        #     low_cpu_mem_usage=True,
        # )
        # self.model = AutoModel.from_pretrained(            
        #     "Salesforce/blip2-opt-2.7b",
        #     torch_dtype=torch.bfloat16,
        #     low_cpu_mem_usage=True,
        # )

        # to device
        self.model.to(self.device)

        self.confusion_triplets = Counter()
        self.alpha = 2.5
        self.deviation = []

    def register_container(self, mode):
        self.result_dict[mode + '_siglip_score_matching'] = []
        self.result_dict[mode + '_siglip_score_matching2'] = []
        self.result_dict[mode + '_siglip_gt_match_cos'] = []

    def generate_print_string(self, mode):
        match_cos = str(self.result_dict[mode + '_siglip_gt_match_cos'].count(1)) + " / " + str(len(self.result_dict[mode + '_siglip_gt_match_cos']))
        result_str = 'SGG eval: '
        result_str += '    SIGLIP COSINE Score Matching: %.4f; ' % np.mean(self.result_dict[mode + '_siglip_score_matching'])
        result_str += '    SIGLIP COSINE Score Matching WEIGHTED: %.4f; ' % np.mean(self.result_dict[mode + '_siglip_score_matching2'])
        result_str += '    SIGLIP NUMBER OF GT MATCHES COSINE: %s; ' % match_cos
        result_str += ' for mode=%s, type=SGLIP Score Matching.' % mode
        result_str += '\n'
        print("Mean Deviation: ", np.mean(self.deviation))
        print(self.confusion_triplets.most_common(20))
        
        return result_str
    
    def preprocess(self, image, texts):
        # texts = torch.tensor(texts).to(self.device)
        inputs = self.processor(text=texts, images=image, padding="max_length", return_tensors="pt")
        return inputs
    
    def compute_similarity(self, text, image):
        if type(text) == str:
            text = [text]
        text = [f"This is a photo of {t}" for t in text]
        inputs = self.preprocess(image, text)
        # to device
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        #     image_features = outputs.image_embeds
        #     text_features = outputs.text_embeds

        #     image_features /= image_features.norm(dim=-1, keepdim=True)
        #     text_features /= text_features.norm(dim=-1, keepdim=True)

        #     cos_scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # return cos_scores[0].cpu()
        
        logits = outputs.logits_per_image
        probs = torch.sigmoid(logits)
        return probs[0]
    
    def calculate(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        fg_triplets = global_container['fg_triplets']

        gt_image = local_container['gt_image'] 

        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
        pred_scores = rel_scores[:,1:].max(1)

        # get predicted pairs which have two matching boxes
        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)
        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)

        # get matching box indices
        matching_boxes, gt_matching_boxes = compute_box_matches(gt_triplet_boxes, pred_triplet_boxes, iou_thresh=iou_thres)

        num_match = 0
        score = 0
        max_rels = len(gt_boxes) * (len(gt_boxes) - 1)

        for l, i in enumerate(matching_boxes):
            # check if the subject and object labels are matching the gt
            if gt_triplets[gt_matching_boxes[l]][0] != pred_triplets[i][0] or gt_triplets[gt_matching_boxes[l]][2] != pred_triplets[i][2]:
                continue

            box_sub = pred_triplet_boxes[i][:4]
            box_obj = pred_triplet_boxes[i][4:]

            # get union box
            union_box = [min(box_sub[0], box_obj[0]), min(box_sub[1], box_obj[1]), max(box_sub[2], box_obj[2]), max(box_sub[3], box_obj[3])]

            # crop the PIL Image
            union_image = gt_image.crop(union_box)

            # get all other possible triplets with the same subject and object
            possible_triplets = fg_triplets[(fg_triplets[:, 0] == pred_triplets[i][0]) & (fg_triplets[:, 1] == pred_triplets[i][2])]
            possible_triplets = possible_triplets.tolist()
            # if pred_triplets[i][2] != pred_triplets[i][0]:
            #     possible_triplets.extend(fg_triplets[(fg_triplets[:, 0] == pred_triplets[i][2]) & (fg_triplets[:, 1] == pred_triplets[i][0])])

            possible_triplets = [str(global_container['ind_to_classes'][triplet[0]]) + " "+ str(global_container['ind_to_predicates'][triplet[2]]) + " "+ str(global_container['ind_to_classes'][triplet[1]]) for triplet in possible_triplets]

            triplet = str(global_container['ind_to_classes'][pred_triplets[i][0]]) + " "+ str(global_container['ind_to_predicates'][pred_triplets[i][1]]) + " "+ str(global_container['ind_to_classes'][pred_triplets[i][2]])

            gt_triplet = gt_triplets[gt_matching_boxes[l]]
            gt_triplet = str(global_container['ind_to_classes'][gt_triplet[0]]) + " "+ str(global_container['ind_to_predicates'][gt_triplet[1]]) + " "+ str(global_container['ind_to_classes'][gt_triplet[2]])

            # construct text list
            text_list = [gt_triplet]
            text_list.append(triplet)
            text_list.extend(possible_triplets)

            probs = self.compute_similarity(text_list, union_image)
            self.deviation.append(torch.std(probs, dim=0).item())

            # get max indice
            max_ind = torch.argmax(probs, dim=0).item()

            if max_ind == 0:
                self.result_dict[mode + '_siglip_gt_match_cos'].append(1)
            else:
                self.result_dict[mode + '_siglip_gt_match_cos'].append(0)
                tuple_confused = (gt_triplet, text_list[max_ind])
                self.confusion_triplets.update([tuple_confused])

            #score += probs[1].item()
            score += (self.alpha * probs[1].item()) / max(1, np.log(np.log(i)))
            num_match += 1

            # remove allocated memory
            del union_image
            # free memory
            torch.cuda.empty_cache()

        if num_match > 0:
            max_rels = 0.5*max_rels
            res2 = self.alpha * ((score / num_match) / max(1, np.log(max_rels-num_match)))
            res = score / num_match
        else:
            res = 0
            res2 = 0

        self.result_dict[mode + '_siglip_score_matching'].append(res)
        self.result_dict[mode + '_siglip_score_matching2'].append(res2)

class CLIPScoreMatching(SceneGraphEvaluation):
    def __init__(self, result_dict):
        import open_clip
        import clip

        super(CLIPScoreMatching, self).__init__(result_dict)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.clip_model, _, self.preprocess =  open_clip.create_model_and_transforms('ViT-B-32', pretrained="/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/negCLIP.pt", device=self.device)
        #self.clip_model, _, self.preprocess =  open_clip.create_model_and_transforms('ViT-H-14-quickgelu', pretrained="dfn5b", device=self.device)
        #self.clip_model, _, self.preprocess =  open_clip.create_model_and_transforms('ViT-H-14', pretrained="/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/h14_v1.2_altogether.pt", device=self.device)

        #self.clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        #self.tokenizer = open_clip.get_tokenizer('ViT-H-14')

        self.clip_model, self.preprocess = clip.load('ViT-L/14', device=self.device)
        self.tokenizer = clip.tokenize
        self.confusion_triplets = Counter()
        self.alpha = 2.5
        self.deviation = []

        self.out_path = "/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/checkpoints/PSG/SGDET/react-yolov8m/triplet_scores.json"
        self.triplets_scores = {}
        self.ids = 0
        self.images_path = "/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/checkpoints/PSG/SGDET/react-yolov8m/images_boxes/"

    def register_container(self, mode):
        self.result_dict[mode + '_clip_score_matching'] = []
        self.result_dict[mode + '_clip_score_matching2'] = []
        self.result_dict[mode + '_clip_gt_match_cos'] = []

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        result_str += '    CLIP Score Matching WEIGHTED: %.4f; ' % np.mean(self.result_dict[mode + '_clip_score_matching'])
        result_str += '    CLIP Score Matching ORIG: %.4f; ' % np.mean(self.result_dict[mode + '_clip_score_matching2'])

        result_str += '    CLIP NUMBER OF GT MATCHES: %s; ' % str(self.result_dict[mode + '_clip_gt_match_cos'].count(1)) + " / " + str(len(self.result_dict[mode + '_clip_gt_match_cos']))
        result_str += ' for mode=%s, type=CLIP Score Matching.' % mode
        result_str += '\n'
        print(self.confusion_triplets.most_common(20))
        print("Mean Deviation: ", np.mean(self.deviation))
        # save to csv
        with open("/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/checkpoints/PSG/SGDET/react-yolov8m/confusion_triplets.csv", "w") as f:
            for key, value in self.confusion_triplets.items():
                f.write("%s,%s,%s\n"%(key[0],key[1],value))

        # triplet_scores = json.dumps(self.triplets_scores)
        with open(self.out_path, "w") as f:
            json.dump(self.triplets_scores, f)

        return result_str
    
    def compute_similarity(self, text, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        if type(text) == str:
            text = [text]
        #for t in text:
            # t = "a photo of a " + t
        t = self.tokenizer(text).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(t)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            cos_scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        return cos_scores[0].cpu()
    
    def weight_function(self, position, max_position, mode="linear"):
        if mode == "linear":
            return (max_position - position) / max_position
        if mode == "log": # normalized log
            return np.log(max_position - position + 1) / np.log(max_position + 1)
    
    def calculate(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        nb_boxes= len(gt_boxes)
        max_rel = nb_boxes * (nb_boxes - 1)

        fg_triplets = global_container['fg_triplets']

        gt_image = local_container['gt_image'] 

        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
        pred_scores = rel_scores[:,1:].max(1)

        # get predicted pairs which have two matching boxes
        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)
        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)

        for i in range(len(gt_triplets)):
            sub_str = global_container['ind_to_classes'][gt_triplets[i][0]]
            obj_str = global_container['ind_to_classes'][gt_triplets[i][2]]
            gt_triplet = sub_str + " "+ str(global_container['ind_to_predicates'][gt_triplets[i][1]]) + " "+ obj_str

            possible_triplets = fg_triplets[(fg_triplets[:, 0] == gt_triplets[i][0]) & (fg_triplets[:, 1] == gt_triplets[i][2])]
            possible_triplets = possible_triplets.tolist()

            possible_triplets = [str(global_container['ind_to_classes'][triplet[0]]) + " "+ str(global_container['ind_to_predicates'][triplet[2]]) + " "+ str(global_container['ind_to_classes'][triplet[1]]) for triplet in possible_triplets]

            text_list = [gt_triplet]
            text_list.extend(possible_triplets)

            box_sub = gt_triplet_boxes[i][:4]
            box_obj = gt_triplet_boxes[i][4:]

            # get union box
            union_box = [min(box_sub[0], box_obj[0]), min(box_sub[1], box_obj[1]), max(box_sub[2], box_obj[2]), max(box_sub[3], box_obj[3])]
            union_image = gt_image.crop(union_box)

            scores = self.compute_similarity(text_list, union_image)

            # compute std
            std = torch.std(scores, dim=0).item()
            self.deviation.append(std)

            # get max indice
            max_ind = torch.argmax(scores, dim=0).item()

            if max_ind == 0:
                self.result_dict[mode + '_clip_gt_match_cos'].append(1)
            else:
                self.result_dict[mode + '_clip_gt_match_cos'].append(0)
                tuple_confused = (gt_triplet, text_list[max_ind])
                self.confusion_triplets.update([tuple_confused])
            num_match = 0


        # # get matching box indices
        # matching_boxes, gt_matching_boxes = compute_box_matches(gt_triplet_boxes, pred_triplet_boxes, iou_thresh=iou_thres)

        # # # create a list of boxes for all possible pairs of gt_boxes, except the ones with the same index
        # # gt_boxes = torch.tensor(gt_boxes)

        # # i_indices, j_indices = torch.meshgrid(torch.arange(nb_boxes), torch.arange(nb_boxes), indexing='ij')
        # # mask = i_indices != j_indices
        # # gt_triplet_boxes = torch.cat((gt_boxes[i_indices[mask]], gt_boxes[j_indices[mask]]), dim=1)

        # # pred_triplet_boxes = pred_triplet_boxes[:100]
        # # #print(len(pred_triplet_boxes))
        # # # get matching box indices
        # # matching_boxes, gt_matching_boxes = compute_box_matches(gt_triplet_boxes, pred_triplet_boxes, iou_thresh=iou_thres)

        # num_match = 0
        # score = 0
        # orig_img = gt_image.copy()

        # for l, i in enumerate(matching_boxes):
        #     if gt_triplets[gt_matching_boxes[l]][0] != pred_triplets[i][0] or gt_triplets[gt_matching_boxes[l]][2] != pred_triplets[i][2]:
        #         continue

        #     box_sub = pred_triplet_boxes[i][:4]
        #     box_obj = pred_triplet_boxes[i][4:]

        #     # get union box
        #     union_box = [min(box_sub[0], box_obj[0]), min(box_sub[1], box_obj[1]), max(box_sub[2], box_obj[2]), max(box_sub[3], box_obj[3])]

        #     # to int
        #     union_box = [int(i) for i in union_box]

        #     # crop the PIL Image
        #     union_image = gt_image.crop(union_box)

        #     # get all other possible triplets with the same subject and object
        #     possible_triplets = fg_triplets[(fg_triplets[:, 0] == pred_triplets[i][0]) & (fg_triplets[:, 1] == pred_triplets[i][2])]
        #     possible_triplets = possible_triplets.tolist()
        #     # if pred_triplets[i][2] != pred_triplets[i][0]:
        #     #     possible_triplets.extend(fg_triplets[(fg_triplets[:, 0] == pred_triplets[i][2]) & (fg_triplets[:, 1] == pred_triplets[i][0])])

        #     # construct list of str
        #     possible_triplets = [str(global_container['ind_to_classes'][triplet[0]]) + " "+ str(global_container['ind_to_predicates'][triplet[2]]) + " "+ str(global_container['ind_to_classes'][triplet[1]]) for triplet in possible_triplets]
            
        #     gt_triplet = gt_triplets[gt_matching_boxes[l]]
        #     sub_str = global_container['ind_to_classes'][gt_triplet[0]]
        #     obj_str = global_container['ind_to_classes'][gt_triplet[2]]
        #     gt_triplet = sub_str + " "+ str(global_container['ind_to_predicates'][gt_triplet[1]]) + " "+ obj_str

        #     triplet = str(global_container['ind_to_classes'][pred_triplets[i][0]]) + " "+ str(global_container['ind_to_predicates'][pred_triplets[i][1]]) + " "+ str(global_container['ind_to_classes'][pred_triplets[i][2]])

        #     text_list = [gt_triplet]
        #     # text_list.append(triplet)
        #     text_list.extend(possible_triplets)

        #     # base_image = orig_img.copy()
        #     # base_image = draw_boundig_boxes(base_image, [box_sub, box_obj], [sub_str, obj_str], color=(255, 0, 0), thickness=2)
        #     # base_image = base_image.crop(union_box)

        #     scores = self.compute_similarity(text_list, union_image)

        #     # compute std
        #     std = torch.std(scores, dim=0).item()
        #     self.deviation.append(std)

        #     # get max indice
        #     max_ind = torch.argmax(scores, dim=0).item()

        #     if max_ind == 0:
        #         self.result_dict[mode + '_clip_gt_match_cos'].append(1)
        #     else:
        #         self.result_dict[mode + '_clip_gt_match_cos'].append(0)
        #         tuple_confused = (gt_triplet, text_list[max_ind])
        #         self.confusion_triplets.update([tuple_confused])
        #         # scores = scores.cpu().numpy()

        #         # self.triplets_scores[self.ids] = {t: float(s) for t, s in zip(text_list, scores)}

        #         # # draw bounding boxes on image and save to folder
        #         # dest_folder = self.images_path
        #         # base_image = orig_img.copy()
        #         # base_image = draw_boundig_boxes(base_image, [box_sub, box_obj], [sub_str, obj_str], color=(255, 0, 0), thickness=2)

        #         # # save
        #         # base_image.save(dest_folder + str(self.ids) + ".png")

        #         self.ids += 1

        #     # plot under the base_image the triplet and the score
        #     # draw_str.append("PREDICTED: "+ triplet + " : " + str(round(cos*100, 2)))

        #     # if pred_cos > gt_cos:
        #     #     self.result_dict[mode + '_clip_gt_match_cos'].append(0)
        #     #     tuple_confused = (gt_triplet, triplet)
        #     #     self.confusion_triplets.update([tuple_confused])    
        #     # else:
        #     #     self.result_dict[mode + '_clip_gt_match_cos'].append(1)

        #     # draw_str.append("OTHER POSSIBLE TRIPLETS:")

        #     # padding = 50
        #     # for other_triplet in possible_triplets:
        #     #     cos = self.compute_similarity(other_triplet, image_features)

        #     #     padding += 20
        #     #     draw_str.append(other_triplet + " : " + str(round(cos*100, 2)))

        #         # if cos > gt_cos:
        #         #     self.result_dict[mode + '_clip_gt_match_cos'][-1] = 0
        #         #     tuple_confused = (gt_triplet, other_triplet)
        #         #     self.confusion_triplets.update([tuple_confused])             

        #     # create a new blank image with the text in black
            
        #     # text_image = Image.new('RGB', (base_image.width, base_image.height + padding), color = (255, 255, 255))
        #     # d = ImageDraw.Draw(text_image)
        #     # # select good font
        #     # fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 12)
        #     # text_width = min(100, base_image.width)
        #     # txt = "\n".join(draw_str)

        #     # d.text((10, 10), txt, fill=(0, 0, 0), font=fnt)

        #     # # combine the two images
        #     # final_img = Image.new('RGB', (base_image.width, base_image.height + padding), color = (255, 255, 255))
        #     # final_img.paste(base_image, (0, 0))
        #     # final_img.paste(text_image, (0, base_image.height))

        #     # # save the drawing to the folder
        #     # img_id = local_container['img_path'].split("/")[-1].split(".")[0]
        #     # final_img.save(dest_folder + img_id +"_"+str(l) + ".png")

        #     score += 0 #(self.alpha * scores[1].item()) / max(1, np.log(np.log(i)))
            
        #     num_match += 1

        #     # remove allocated memory
        #     del union_image
        #     del triplet
        #     # free memory
        #     torch.cuda.empty_cache()

        if num_match > 0:
            # to compute the average score, we penalize too few or too many predictions, by using the max_rel value
            # log of max_rel
            # print((score / num_match))
            max_rel = 0.5*max_rel
            res = (score / num_match) / max(1, np.log(max_rel-num_match))
            res2 = score / num_match
            # print(res, score, num_match, max_rel, np.log(max_rel-num_match))
            # print(len(gt_boxes), len(pred_boxes), len(matching_boxes))
        else:
            res = 0
            res2 = 0
            
        self.result_dict[mode + '_clip_score_matching'].append(res)
        self.result_dict[mode + '_clip_score_matching2'].append(res2)


def draw_boundig_boxes(image, boxes, labels, color=(255, 0, 0), thickness=2):
    fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 12)
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        draw.rectangle(box, outline=color, width=thickness)
        # draw the label as a text with a rectangle background
        draw.rectangle([box[0], box[1], box[0]+len(label)*8, box[1]+20], fill=color)
        draw.text((box[0], box[1]), label, fill=(255, 255, 255), font=fnt)

    return image

class SGF1Score(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGF1Score, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_f1_score'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k in self.result_dict[mode + '_f1_score']:
            result_str += '    F1 @ %d: %.4f; ' % (k, self.result_dict[mode + '_f1_score'][k])
        result_str += ' for mode=%s, type=F1.' % mode
        result_str += '\n'
        return result_str

    def calculate(self, global_container, local_container, mode):
        for k in global_container[mode + '_recall']:
            recall_k = np.mean(global_container[mode + '_recall'][k])
            mean_reacall_k = np.mean(global_container[mode + '_mean_recall'][k])

            if recall_k + mean_reacall_k > 0:
                f1 = 2 * recall_k * mean_reacall_k / (recall_k + mean_reacall_k)
            else:
                f1 = 0
            self.result_dict[mode + '_f1_score'][k] = f1

class SGRecallRelative(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGRecallRelative, self).__init__(result_dict)
        
    def register_container(self, mode):
        self.result_dict[mode + '_recall_relative'] = []

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        result_str += '    R @ %s: %.4f; ' % ('relative', np.mean(self.result_dict[mode + '_recall_relative']))
        result_str += ' for mode=%s, type=Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate(self, global_container, local_container, mode):
        gt_rels = local_container['gt_rels']

        pred_to_gt = local_container['pred_to_gt']

        # get number of gt relationships
        k = gt_rels.shape[0]
        match = reduce(np.union1d, pred_to_gt[:k])
        rec_i = float(len(match)) / float(gt_rels.shape[0])
        self.result_dict[mode + '_recall_relative'].append(rec_i)

        return local_container
    
class SGMeanRecallRelative(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=True):
        super(SGMeanRecallRelative, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:] # remove __background__

    def register_container(self, mode):
        self.result_dict[mode + '_mean_recall_relative'] = {'relative': 0.0}
        self.result_dict[mode + '_mean_recall_collect_relative'] = {'relative': [[] for i in range(self.num_rel)]}
        self.result_dict[mode + '_mean_recall_list_relative'] = {'relative': []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_mean_recall_relative'].items():
            result_str += '   mR @ %s: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Mean Recall.' % mode
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_mean_recall_list_relative']['relative']):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        k = gt_rels.shape[0]

        # the following code are copied from Neural-MOTIFS
        match = reduce(np.union1d, pred_to_gt[:k])
        # NOTE: by kaihua, calculate Mean Recall for each category independently
        # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
        recall_hit = [0] * self.num_rel
        recall_count = [0] * self.num_rel
        for idx in range(gt_rels.shape[0]):
            local_label = gt_rels[idx,2]
            recall_count[int(local_label)] += 1
            recall_count[0] += 1

        for idx in range(len(match)):
            local_label = gt_rels[int(match[idx]),2]
            recall_hit[int(local_label)] += 1
            recall_hit[0] += 1
        
        for n in range(self.num_rel):
            if recall_count[n] > 0:
                self.result_dict[mode + '_mean_recall_collect_relative']['relative'][n].append(float(recall_hit[n] / recall_count[n]))

    def calculate(self, global_container, local_container, mode):
        sum_recall = 0
        num_rel_no_bg = self.num_rel - 1
        for idx in range(num_rel_no_bg):
            if len(self.result_dict[mode + '_mean_recall_collect_relative']['relative'][idx+1]) == 0:
                tmp_recall = 0.0
            else:
                tmp_recall = np.mean(self.result_dict[mode + '_mean_recall_collect_relative']['relative'][idx+1])
            self.result_dict[mode + '_mean_recall_list_relative']['relative'].append(tmp_recall)
            sum_recall += tmp_recall

        self.result_dict[mode + '_mean_recall_relative']['relative'] = sum_recall / float(num_rel_no_bg)
        return

class SGInformativeRecallWeighted(SceneGraphEvaluation):
    def __init__(self, result_dict, sim='mpnet'):
        super(SGInformativeRecall, self).__init__(result_dict)

        self.sim_options = ['glove', 'uae_large', 'bert_large', 'minilm', 'mpnet', 'clip']
        if sim not in self.sim_options:
            raise ValueError('sim must be in %s' % self.sim_options)
        self.similarity = sim

        # load embeddings according to similarity value
        if self.similarity == 'glove':
            self.sim_model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
        elif self.similarity == 'uae_large':
            self.sim_model = SentenceTransformer('WhereIsAI/UAE-Large-V1')
        elif self.similarity == 'bert_large':
            self.sim_model = SentenceTransformer('bert-large-nli-mean-tokens')
        elif self.similarity == 'minilm':
            self.sim_model = SentenceTransformer('all-MiniLM-L12-v2')
        elif self.similarity == "mpnet":
            self.sim_model = SentenceTransformer('all-mpnet-base-v2')
        elif self.similarity == "clip":
            self.sim_model = SentenceTransformer('CLIP-ViT-B-32')

    def register_container(self, mode):
        self.result_dict[mode + '_informative_recall'] = {100: []} # 5: [], 10: [], 20: [], 50: [], 

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_informative_recall'].items():
            result_str += '    IR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Informative Recall.' % mode
        result_str += '\n'
        return result_str
    
    def similarity_match(self, gt_triplets, pred_triplets, cosine_thres=0.9):
        pred_to_gt = [[] for _ in range(len(pred_triplets))]

        if len(gt_triplets) == 0 or len(pred_triplets) == 0:
            return pred_to_gt

        gt_triplets_embeddings = self.sim_model.encode(gt_triplets)
        pred_triplets_embeddings = self.sim_model.encode(pred_triplets)

        # Compute cosine similarity for all combinations at once
        cos_sim_matrix = util.cos_sim(pred_triplets_embeddings, gt_triplets_embeddings)

        # Iterate over each pred_triplet's cosine similarity scores
        for i, cos_sim_scores in enumerate(cos_sim_matrix):
            # Find the index of the maximum cosine similarity score for the current pred_triplet
            max_sim_score, max_index = torch.max(cos_sim_scores, dim=0)

            # If the highest cosine similarity score is above the threshold, consider it a match
            if max_sim_score > cosine_thres:
                pred_to_gt[i].append(max_index.item())

        return pred_to_gt

    def calculate(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        gt_relationships = local_container['informative_rels']

        pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
        pred_scores = rel_scores[:,1:].max(1)

        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        pred_triplets = [str(global_container['ind_to_classes'][triplet[0]]) + " "+ str(global_container['ind_to_predicates'][triplet[1]]) + " "+ str(global_container['ind_to_classes'][triplet[2]]) for triplet in pred_triplets]

        pred_to_gt = self.similarity_match(gt_relationships, pred_triplets, cosine_thres=0.9)

        for k in self.result_dict[mode + '_informative_recall']:
            weighted_sum = 0
            # get indices of pred_to_gt that are not empty
            indices = [i for i, x in enumerate(pred_to_gt[:k]) if x]
            # apply a lambda function to remove the number of previous items in the list, such that a perfect ranking = 0
            indices = list(map(lambda x, y: x - y, indices, range(len(indices))))
            for idx in indices:
                # Compute weight for the current position
                weight = self.weight_function(idx, k)
                # Add weighted match to the weighted sum
                weighted_sum += weight 

            # Normalize the weighted recall score
            rec_i_weighted = weighted_sum / len(gt_relationships)
            self.result_dict[mode + '_informative_recall'][k].append(rec_i_weighted)

        return local_container
    
    def weight_function(self, position, max_position, mode="linear"):
        if mode == "linear":
            return (max_position - position) / max_position
        if mode == "log": # normalized log
            return np.log(max_position - position + 1) / np.log(max_position + 1)


class SGInformativeRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, sim='mpnet'):
        super(SGInformativeRecall, self).__init__(result_dict)

        self.sim_options = ['glove', 'uae_large', 'bert_large', 'minilm', 'mpnet', 'clip']
        if sim not in self.sim_options:
            raise ValueError('sim must be in %s' % self.sim_options)
        self.similarity = sim

        # load embeddings according to similarity value
        if self.similarity == 'glove':
            self.sim_model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
        elif self.similarity == 'uae_large':
            self.sim_model = SentenceTransformer('WhereIsAI/UAE-Large-V1')
        elif self.similarity == 'bert_large':
            self.sim_model = SentenceTransformer('bert-large-nli-mean-tokens')
        elif self.similarity == 'minilm':
            self.sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        elif self.similarity == "mpnet":
            self.sim_model = SentenceTransformer('all-mpnet-base-v2')
        elif self.similarity == "clip":
            self.sim_model = SentenceTransformer('CLIP-ViT-B-32')

    def register_container(self, mode):
        self.result_dict[mode + '_informative_recall'] = {5: [], 10: [], 20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_informative_recall'].items():
            result_str += '    IR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Informative Recall.' % mode
        result_str += '\n'
        return result_str

    def similarity_match(self, gt_triplets, pred_triplets, cosine_thres=0.8):
        """
        Perform cosine similarity between gt_triplets list of strings and pred_triplets list of strings
        For each pred_triplet, find the gt_triplet with the highest cosine similarity score
        If the highest cosine similarity score is above a threshold, then consider the pred_triplet to be a match
        Return:
            pred_to_gt [List of List]
        """
        pred_to_gt = [[] for _ in range(len(pred_triplets))]

        if len(gt_triplets) == 0 or len(pred_triplets) == 0:
            return pred_to_gt

        gt_triplets_embeddings = self.sim_model.encode(gt_triplets, batch_size=256, device='cuda')
        pred_triplets_embeddings = self.sim_model.encode(pred_triplets, batch_size=256, device='cuda')

        # Compute cosine similarity for all combinations at once
        cos_sim_matrix = util.cos_sim(pred_triplets_embeddings, gt_triplets_embeddings)

        # Iterate over each pred_triplet's cosine similarity scores
        for i, cos_sim_scores in enumerate(cos_sim_matrix):
            # Find the index of the maximum cosine similarity score for the current pred_triplet
            max_sim_score, max_index = torch.max(cos_sim_scores, dim=0)

            # If the highest cosine similarity score is above the threshold, consider it a match
            if max_sim_score > cosine_thres:
                pred_to_gt[i].append(max_index.item())

        return pred_to_gt

    def calculate(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        gt_relationships = local_container['informative_rels']

        pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
        pred_scores = rel_scores[:,1:].max(1)

        pred_triplets, _, _ = _triplet(pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        pred_triplets = [str(global_container['ind_to_classes'][triplet[0]]) + " "+ str(global_container['ind_to_predicates'][triplet[1]]) + " "+ str(global_container['ind_to_classes'][triplet[2]]) for triplet in pred_triplets]

        pred_to_gt = self.similarity_match(gt_relationships, pred_triplets, cosine_thres=0.8)

        for k in self.result_dict[mode + '_informative_recall']:
            # check if pred_to_gt_inf is empty
            if all(len(x) == 0 for x in pred_to_gt):
                rec_i = 0.0
            else:
                match = reduce(np.union1d, pred_to_gt[:k])
                if len(gt_relationships) > 0 and len(match) > 0:
                    rec_i = float(len(match)) / float(len(gt_relationships))
                else:
                    rec_i = 0.0
                self.result_dict[mode + '_informative_recall'][k].append(rec_i)

        return local_container

"""
Traditional Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""
class SGRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGRecall, self).__init__(result_dict)
        self.scores = {20: [], 50: [], 100: []}
        self.number_matches = 0

    def register_container(self, mode):
        self.result_dict[mode + '_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall'].items():
            result_str += '    R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Recall (Main).' % mode
        result_str += '\n'
        
        # print average of scores
        for k, v in self.scores.items():
            result_str += '    Avg Score @ %d: %.4f; ' % (k, np.mean(v))
        result_str += '\n'
        print("Number of matches: ", self.number_matches)
        return result_str

    def calculate(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
        pred_scores = rel_scores[:,1:].max(1)

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)

        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes

        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
                pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        # apply [0] * [1] * [2] to compute the overall score of the triplet
        pred_triplet_scores = pred_triplet_scores[:, 0] * pred_triplet_scores[:, 1] * pred_triplet_scores[:, 2]
        # to single dim
        pred_triplet_scores = pred_triplet_scores.squeeze() 

        # Compute recall. It's most efficient to match once and then do recall after
        pred_to_gt = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            global_container,
            phrdet=mode=='phrdet',
        )
        match = reduce(np.union1d, pred_to_gt)
        self.number_matches += len(match)
        
        local_container['pred_to_gt'] = pred_to_gt

        for k in self.result_dict[mode + '_recall']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            
            # accumulate scores from matching predictions
            indices = [i for i, x in enumerate(pred_to_gt[:k]) if x]
            if len(indices) > 0 and len(pred_triplet_scores) > 0:
                self.scores[k].extend(pred_triplet_scores[indices])

            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall'][k].append(rec_i)

        return local_container

class SGWeightedRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGWeightedRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_weighted_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_weighted_recall'].items():
            result_str += '    R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Weighted Recall (Main).' % mode
        result_str += '\n'
        return result_str

    def calculate(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
        pred_scores = rel_scores[:,1:].max(1)

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes

        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
                pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        # Compute recall. It's most efficient to match once and then do recall after
        pred_to_gt = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            global_container,
            phrdet=mode=='phrdet',
        )
        local_container['pred_to_gt'] = pred_to_gt

        # Modify the recall calculation loop to include weights
        for k in self.result_dict[mode + '_weighted_recall']:
            weighted_sum = 0
            # get indices of pred_to_gt that are not empty
            indices = [i for i, x in enumerate(pred_to_gt[:k]) if x]
            # apply a lambda function to remove the number of previous items in the list, such that a perfect ranking = 0
            indices = list(map(lambda x, y: x - y, indices, range(len(indices))))
            for idx in indices:
                # Compute weight for the current position
                weight = self.weight_function(idx, k)
                # Add weighted match to the weighted sum
                weighted_sum += weight 

            # Normalize the weighted recall score
            rec_i_weighted = weighted_sum / len(gt_triplets)
            self.result_dict[mode + '_weighted_recall'][k].append(rec_i_weighted)

        return local_container
    
    def weight_function(self, position, max_position, mode="linear"):
        if mode == "linear":
            return (max_position - position) / max_position
        if mode == "log": # normalized log
            return np.log(max_position - position + 1) / np.log(max_position + 1)
    
"""
No Graph Constraint Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""
class SGNoGraphConstraintRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGNoGraphConstraintRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_recall_nogc'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall_nogc'].items():
            result_str += ' ng-R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=No Graph Constraint Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate(self, global_container, local_container, mode):
        obj_scores = local_container['obj_scores']
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_boxes = local_container['pred_boxes']
        pred_classes = local_container['pred_classes']
        gt_rels = local_container['gt_rels']

        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        nogc_overall_scores = obj_scores_per_rel[:,None] * rel_scores[:,1:]
        nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        nogc_pred_rels = np.column_stack((pred_rel_inds[nogc_score_inds[:,0]], nogc_score_inds[:,1]+1))
        nogc_pred_scores = rel_scores[nogc_score_inds[:,0], nogc_score_inds[:,1]+1]

        nogc_pred_triplets, nogc_pred_triplet_boxes, _ = _triplet(
                nogc_pred_rels, pred_classes, pred_boxes, nogc_pred_scores, obj_scores
        )

        # No Graph Constraint
        gt_triplets = local_container['gt_triplets']
        gt_triplet_boxes = local_container['gt_triplet_boxes']
        iou_thres = global_container['iou_thres']

        nogc_pred_to_gt = _compute_pred_matches(
            gt_triplets,
            nogc_pred_triplets,
            gt_triplet_boxes,
            nogc_pred_triplet_boxes,
            iou_thres,
            global_container,
            phrdet=mode=='phrdet',
        )

        local_container['nogc_pred_to_gt'] = nogc_pred_to_gt

        for k in self.result_dict[mode + '_recall_nogc']:
            match = reduce(np.union1d, nogc_pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall_nogc'][k].append(rec_i)

        return local_container

"""
Zero Shot Scene Graph
Only calculate the triplet that not occurred in the training set
"""
class SGZeroShotRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGZeroShotRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_zeroshot_recall'] = {20: [], 50: [], 100: []} 

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_zeroshot_recall'].items():
            result_str += '   zR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Zero Shot Recall.' % mode
        result_str += '\n'
        return result_str

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        zeroshot_triplets = global_container['zeroshot_triplet']

        sub_id, ob_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[ob_id], pred_label))  # num_rel, 3

        if len(zeroshot_triplets) == 0: # no zero shot triplets
            self.zeroshot_idx = []
        else:
            self.zeroshot_idx = np.where( intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0 )[0].tolist()

    def calculate(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']

        for k in self.result_dict[mode + '_zeroshot_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.zeroshot_idx) + len(match_list) - len(set(self.zeroshot_idx + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.zeroshot_idx))
                self.result_dict[mode + '_zeroshot_recall'][k].append(zero_rec_i)
                
"""
No Graph Constraint Mean Recall
"""
class SGNGZeroShotRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGNGZeroShotRecall, self).__init__(result_dict)
    
    def register_container(self, mode):
        self.result_dict[mode + '_ng_zeroshot_recall'] = {20: [], 50: [], 100: []} 

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_ng_zeroshot_recall'].items():
            result_str += 'ng-zR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=No Graph Constraint Zero Shot Recall.' % mode
        result_str += '\n'
        return result_str

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        zeroshot_triplets = global_container['zeroshot_triplet']

        sub_id, ob_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[ob_id], pred_label))  # num_rel, 3

        if len(zeroshot_triplets) == 0: # no zero shot triplets
            self.zeroshot_idx = []
        else:
            self.zeroshot_idx = np.where( intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0 )[0].tolist()

    def calculate(self, global_container, local_container, mode):
        pred_to_gt = local_container['nogc_pred_to_gt']

        for k in self.result_dict[mode + '_ng_zeroshot_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.zeroshot_idx) + len(match_list) - len(set(self.zeroshot_idx + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.zeroshot_idx))
                self.result_dict[mode + '_ng_zeroshot_recall'][k].append(zero_rec_i)


"""
Give Ground Truth Object-Subject Pairs
Calculate Recall for SG-Cls and Pred-Cls
Only used in https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
"""
class SGPairAccuracy(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGPairAccuracy, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_accuracy_hit'] = {20: [], 50: [], 100: []}
        self.result_dict[mode + '_accuracy_count'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        # mean everything
        for k in self.result_dict[mode + '_accuracy_hit']:
            self.result_dict[mode + '_accuracy_hit'][k] = np.mean(self.result_dict[mode + '_accuracy_hit'][k])
            self.result_dict[mode + '_accuracy_count'][k] = np.mean(self.result_dict[mode + '_accuracy_count'][k])

        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accuracy_hit'].items():
            a_hit = v
            a_count = self.result_dict[mode + '_accuracy_count'][k]
            result_str += '    A @ %d: %.4f; ' % (k, a_hit/a_count)
        result_str += ' for mode=%s, type=TopK Accuracy.' % mode
        result_str += '\n'
        return result_str

    def prepare_gtpair(self, local_container):
        pred_pair_idx = local_container['pred_rel_inds'][:, 0] * 1024 + local_container['pred_rel_inds'][:, 1]
        gt_pair_idx = local_container['gt_rels'][:, 0] * 1024 + local_container['gt_rels'][:, 1]
        self.pred_pair_in_gt = (pred_pair_idx[:, None] == gt_pair_idx[None, :]).sum(-1) > 0

    def calculate(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_accuracy_hit']:
            # to calculate accuracy, only consider those gt pairs
            # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
            # for sgcls and predcls
            if mode != 'sgdet':
                gt_pair_pred_to_gt = []
                for p, flag in zip(pred_to_gt, self.pred_pair_in_gt):
                    if flag:
                        gt_pair_pred_to_gt.append(p)
                if len(gt_pair_pred_to_gt) > 0:
                    gt_pair_match = reduce(np.union1d, gt_pair_pred_to_gt[:k])
                else:
                    gt_pair_match = []
                self.result_dict[mode + '_accuracy_hit'][k].append(float(len(gt_pair_match)))
                self.result_dict[mode + '_accuracy_count'][k].append(float(gt_rels.shape[0]))

"""
Mean Recall: Proposed in:
https://arxiv.org/pdf/1812.01880.pdf CVPR, 2019
"""
class SGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=True):
        super(SGMeanRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:] # remove __background__

    def register_container(self, mode):
        #self.result_dict[mode + '_recall_hit'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        #self.result_dict[mode + '_recall_count'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        self.result_dict[mode + '_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}
        self.result_dict[mode + '_mean_recall_list'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            result_str += '   mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Mean Recall.' % mode
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_mean_recall_collect']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx,2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]),2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1
            
            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + '_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))
 
    def calculate(self, global_container, local_container, mode):
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_mean_recall_collect'][k][idx+1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_mean_recall_collect'][k][idx+1])
                self.result_dict[mode + '_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[mode + '_mean_recall'][k] = sum_recall / float(num_rel_no_bg)
        return

class SGWeightedMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=True):
        super(SGWeightedMeanRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:] # remove __background__

    def register_container(self, mode):
        #self.result_dict[mode + '_recall_hit'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        #self.result_dict[mode + '_recall_count'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        self.result_dict[mode + '_weighted_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_weighted_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}
        self.result_dict[mode + '_weighted_mean_recall_list'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_weighted_mean_recall'].items():
            result_str += '   mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Weighted Mean Recall.' % mode
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_weighted_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        return result_str
    
    def weight_function(self, position, max_position, mode="linear"):
        if mode == "linear":
            return (max_position - position) / max_position
        if mode == "log": # normalized log
            return np.log(max_position - position + 1) / np.log(max_position + 1)

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_weighted_mean_recall_collect']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel

            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx,2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                weight = self.weight_function(idx, k)
                local_label = gt_rels[int(match[idx]),2]
                recall_hit[int(local_label)] += weight
                recall_hit[0] += weight

            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + '_weighted_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))

    def calculate(self, global_container, local_container, mode):
        for k, v in self.result_dict[mode + '_weighted_mean_recall'].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_weighted_mean_recall_collect'][k][idx+1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_weighted_mean_recall_collect'][k][idx+1])
                self.result_dict[mode + '_weighted_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[mode + '_weighted_mean_recall'][k] = sum_recall / float(num_rel_no_bg)
        return

"""
No Graph Constraint Mean Recall
"""
class SGNGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=True):
        super(SGNGMeanRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:] # remove __background__

    def register_container(self, mode):
        self.result_dict[mode + '_ng_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_ng_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}
        self.result_dict[mode + '_ng_mean_recall_list'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_ng_mean_recall'].items():
            result_str += 'ng-mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=No Graph Constraint Mean Recall.' % mode
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_ng_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container['nogc_pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_ng_mean_recall_collect']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx,2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]),2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1
            
            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + '_ng_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))
 

    def calculate(self, global_container, local_container, mode):
        for k, v in self.result_dict[mode + '_ng_mean_recall'].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_ng_mean_recall_collect'][k][idx+1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_ng_mean_recall_collect'][k][idx+1])
                self.result_dict[mode + '_ng_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[mode + '_ng_mean_recall'][k] = sum_recall / float(num_rel_no_bg)
        return

"""
Accumulate Recall:
calculate recall on the whole dataset instead of each image
"""
class SGAccumulateRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGAccumulateRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_accumulate_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            result_str += '   aR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Accumulate Recall.' % mode
        result_str += '\n'
        return result_str

    def calculate(self, global_container, local_container, mode):
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            self.result_dict[mode + '_accumulate_recall'][k] = float(self.result_dict[mode + '_recall_hit'][k][0]) / float(self.result_dict[mode + '_recall_count'][k][0] + 1e-10)
        return 

def _triplet(relations, classes, boxes, predicate_scores=None, class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        boxes (#objs, 4)
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns: 
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplets_boxes (#rel, 8) array of boxes for the parts
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]

    try:
        triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    except:
        print('sub_id:', sub_id)
        print('ob_id:', ob_id)
        print('classes:', classes)
        print('relations:', relations)
        print('classes[sub_id]:', classes[sub_id])
        print('classes[ob_id]:', classes[ob_id])
        print(classes.shape)
        print(classes)
        raise

    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[sub_id], predicate_scores, class_scores[ob_id],
        ))

    return triplets, triplet_boxes, triplet_scores


def compute_box_matches(gt_boxes, pred_boxes, iou_thresh=0.5):
    """ Compute IoU between two boxlists, and return the matches
    Parameters:
        gt_boxes (ndarray): n x 8
        pred_boxes (ndarray): m x 8
        iou_thresh (float): threshold to consider as matched
    Returns:
    """

    # for each triplet, calculate the IoU of the union box
    matches = []
    gt_matches = []
    for i, gt_box in enumerate(gt_boxes):
        # compute IoU with every predicted box
        # if IoU > threshold, then consider as matched

        sub_iou = bbox_overlaps(gt_box[None,:4], pred_boxes[:, :4])
        obj_iou = bbox_overlaps(gt_box[None,4:], pred_boxes[:, 4:])

        inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        if inds.any():
            matches.append(inds.argmax())
            gt_matches.append(i)

    return matches, gt_matches

def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thres, global_container, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)

    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    i=0
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        i+=1
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thres

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thres) & (obj_iou >= iou_thres)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt

def _compute_pred_matches2(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thres, global_container, phrdet=False, sim='clip', threshold=0.9):

    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
        
    # transform gt_triplets and pred_triplets to full string
    pred_triplets_full = [str(global_container['ind_to_classes'][triplet[0]]) + " "+ str(global_container['ind_to_predicates'][triplet[1]]) + " "+ str(global_container['ind_to_classes'][triplet[2]]) for triplet in pred_triplets]

    gt_triplets_full = [str(global_container['ind_to_classes'][triplet[0]]) + " "+ str(global_container['ind_to_predicates'][triplet[1]]) + " "+ str(global_container['ind_to_classes'][triplet[2]]) for triplet in gt_triplets]

    for i in range(len(gt_triplets)):
        gt_triplets_full[i] = gt_triplets_full[i].lower()
    for i in range(len(pred_triplets)):
        pred_triplets_full[i] = pred_triplets_full[i].lower()

    pred_to_gt = [[] for x in range(len(pred_triplets_full))]

    if len(gt_triplets_full) == 0 or len(pred_triplets_full) == 0:
        return pred_to_gt

    gt_triplets_embeddings = sim_model.encode(gt_triplets_full, batch_size=128)
    pred_triplets_embeddings = sim_model.encode(pred_triplets_full, batch_size=128)

    # Convert the lists to PyTorch tensors
    gt_triplets_embeddings = torch.tensor(gt_triplets_embeddings)
    pred_triplets_embeddings = torch.tensor(pred_triplets_embeddings)

    # # Normalize the embeddings
    # gt_triplets_embeddings = F.normalize(gt_triplets_embeddings, p=2, dim=1)
    # pred_triplets_embeddings = F.normalize(pred_triplets_embeddings, p=2, dim=1)

    # Compute the cosine similarity for all pairs of triplets
    cos_sim = F.cosine_similarity(gt_triplets_embeddings[:, None, :], pred_triplets_embeddings[None, :, :], dim=2)

    # Get the indices of the top-1 most similar predicted triplet for each ground truth triplet
    top_sim_values, top_sim_indices = cos_sim.topk(1, dim=1)

    # Get the indices above the threshold
    temp_list = torch.argwhere(top_sim_values >= threshold)[:,:1].squeeze().tolist()

    match_indices_gt = temp_list if isinstance(temp_list, list) else [temp_list]

    tmp_pred = top_sim_indices[match_indices_gt].squeeze().tolist()
    match_indices_pred = tmp_pred if isinstance(tmp_pred, list) else [tmp_pred]

    # concat gt_boxes[match_indices_gt][:, :4] and gt_boxes[match_indices_gt][:, 4:] to get the full box
    sub_iou_all = bbox_overlaps(gt_boxes[match_indices_gt][:, :4], pred_boxes[:, :4])
    sub_iou_all = np.maximum(sub_iou_all, bbox_overlaps(gt_boxes[match_indices_gt][:, 4:], pred_boxes[:, :4]))
    obj_iou_all = bbox_overlaps(gt_boxes[match_indices_gt][:, :4], pred_boxes[:, 4:])
    obj_iou_all = np.maximum(obj_iou_all, bbox_overlaps(gt_boxes[match_indices_gt][:, 4:], pred_boxes[:, 4:]))

    # print(sub_iou_all)
    # print(len(sub_iou_all))

    # Iterate over each GT box and its corresponding matches.
    for gt_ind, sub_iou, obj_iou, keep in zip(match_indices_gt, sub_iou_all, obj_iou_all, match_indices_pred):
        # Apply threshold to both subject and object IOUs.
        inds = (sub_iou[keep] >= iou_thres) & (obj_iou[keep] >= iou_thres)

        # If any indices meet the threshold, append to pred_to_gt.
        if inds.any():
            pred_to_gt[keep].extend(int(gt_ind) for i in inds.nonzero()[0])

    return pred_to_gt
