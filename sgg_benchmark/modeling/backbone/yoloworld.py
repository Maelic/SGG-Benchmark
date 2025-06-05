import torch
import torch.nn.functional as F
from ultralytics.nn.tasks import WorldModel
from sgg_benchmark.data.transforms import LetterBox

from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import ops
from ultralytics.engine.results import Results

from sgg_benchmark.structures.bounding_box import BoxList
from sgg_benchmark.utils.txt_embeddings import obj_edge_vectors
from sgg_benchmark.modeling.backbone.utils import non_max_suppression

import numpy as np

from ultralytics.nn.modules import (
    C2fAttn,
    ImagePoolingAttn,
    WorldDetect,
)

class YoloWorldModel(WorldModel):
    def __init__(self, cfg, ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        yolo_cfg = cfg.MODEL.YOLO.SIZE+'.yaml'
        self.cfg = cfg
        super().__init__(yolo_cfg, nc=nc, verbose=verbose)
        # self.features_layers = [len(self.model) - 2]
        self.conf_thres = cfg.MODEL.BACKBONE.NMS_THRESH
        self.iou_thres = cfg.MODEL.ROI_HEADS.NMS
        self.device = cfg.MODEL.DEVICE
        self.input_size = cfg.INPUT.MIN_SIZE_TRAIN
        self.nc = nc
        self.max_det = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG

    def forward(self, x, profile=False, txt_feats=None, visualize=False, embed=None):
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)
        ori_txt_feats = txt_feats.clone()
        y, dt, feature_maps = [], [], []  # outputs
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            """
            We extract features from the following layers:
            15: 80x80
            18: 40x40
            21: 20x20
            For different object scales, as in original YOLOV8 implementation.
            """
            if embed and m.i in {15, 18, 21}: # if current layer is one of the feature extraction layers
                feature_maps.append(x)
        if embed:
            return x, feature_maps
        else:
            return x

    def load(self, weights_path: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """

        weights, _ = attempt_load_one_weight(weights_path)

        if weights:
            super().load(weights)
        self.txt_feats = weights.txt_feats

    def load_txt_feats(self, names):
        txt_feats = obj_edge_vectors(names, wv_type='clip')
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.txt_feats = txt_feats.reshape(-1, len(names), txt_feats.shape[-1])

    def prepare_input(self, image, input_shape=(640,640), stride=32, auto=True):
        not_tensor = not isinstance(image, torch.Tensor)
        if not_tensor:
            same_shapes = all(x.shape == im[0].shape for x in image)
            letterbox = LetterBox(input_shape, auto=same_shapes, stride=self.model.stride)(image=image)
            im = np.stack([letterbox(image=x) for x in im])
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device).float()
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0

        return im
    
    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        if not(torch.is_tensor(batch)):
            batch['img'] = batch['img'].to(self.device, non_blocking=True)
            batch['img'] = (batch['img'].half() if self.half else batch['img'].float()) / 255
            for k in ['batch_idx', 'cls', 'bboxes']:
                batch[k] = batch[k].to(self.device)

            nb = len(batch['img'])
            self.lb = [torch.cat([batch['cls'], batch['bboxes']], dim=-1)[batch['batch_idx'] == i]
                    for i in range(nb)] if self.save_hybrid else []  # for autolabelling
        else:
            batch = batch.to(self.device, non_blocking=True)
            batch = (batch.half() if self.half else batch.float()) / 255
            nb = len(batch)

        return batch
    
    def visualize(self, preds, orig_imgs):
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
    
        # get model input size
        imgsz = (self.input_size, self.input_size)
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(imgsz, pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
    
    def postprocess(self, preds, image_sizes):
        """Post-processes predictions and returns a list of Results objects."""
        preds, indices = non_max_suppression(
            preds,
            nc=self.nc,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            max_det=self.max_det,
        )

        if len(preds) == 0:
            # return a dummy box with size of all image
            boxes = torch.tensor([[0, 0, image_sizes[0][1], image_sizes[0][0]]], device=self.device)
            scores = torch.tensor([0.0], device=self.device)
            labels = torch.tensor([0], device=self.device)
            boxlist = BoxList(boxes, image_sizes[0], mode="xyxy")
            boxlist.add_field("pred_labels", labels)
            boxlist.add_field("pred_scores", scores)
            boxlist.add_field("labels", labels)
            boxlist.add_field("feat_idx", torch.tensor([0], device=self.device))
            return [boxlist]
        
        results = []
        for i, (pred, idx) in enumerate(zip(preds, indices)):
            # flip
            out_img_size = image_sizes[i]

            boxes = pred[:, :4]
            boxes = ops.scale_boxes((self.input_size, self.input_size), boxes, (out_img_size[1], out_img_size[0]))

            boxlist = BoxList(boxes, out_img_size, mode="xyxy")

            scores = pred[:, 4]
            labels = pred[:, 5].long()
            boxlist.add_field("pred_labels", labels.detach().clone())
            # add 1 to all labels to account for background class
            labels += 1
            boxlist.add_field("pred_scores", scores)
            boxlist.add_field("labels", labels)
            boxlist.add_field("feat_idx", idx.long())

            results.append(boxlist)
        return results

    
    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}