import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel

from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import ops
from ultralytics.utils.plotting import feature_visualization
from pathlib import Path

from sgg_benchmark.structures.boxlist_ops import clip_boxes
class YoloV8(DetectionModel):
    def __init__(self, cfg, ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        yolo_cfg = cfg.MODEL.YOLO.SIZE+'.yaml'
        if cfg.VERBOSE in ["DEBUG", "INFO"]:
            verbose = True
        else:
            verbose = False
        super().__init__(yolo_cfg, nc=nc, verbose=False)
        # self.features_layers = [len(self.model) - 2]
        self.conf_thres = cfg.MODEL.BACKBONE.NMS_THRESH
        self.iou_thres = cfg.MODEL.ROI_HEADS.NMS
        self.device = cfg.MODEL.DEVICE
        self.input_size = cfg.INPUT.MIN_SIZE_TRAIN
        self.nc = nc
        self.max_det = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG

        if self.end2end:
            self.layers_to_extract = [16, 19, 22]
        else:
            self.layers_to_extract = [15, 18, 21]

    # custom implementation of forward method based on
    # https://github.com/ultralytics/ultralytics/blob/3df9d278dce67eec7fdb4fddc0aab22fee62588f/ultralytics/nn/tasks.py#L122
    def forward(self, x, profile=False, visualize=False, embed=None):
        y, feature_maps = [], []  # outputs
        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            """
            We extract features from the following layers:
            15: 80x80
            18: 40x40
            21: 20x20
            For different object scales, as in original YOLOV8 implementation.
            """
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=Path('/home/maelic/Documents/PhD/MyModel/SGG-Benchmark/demo/test_custom/results'))
            if embed:
                if i in self.layers_to_extract:  # if current layer is one of the feature extraction layers
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
    
    def postprocess(self, preds, image_sizes):
        """Post-processes predictions and returns a list of Results objects."""

        if self.end2end:
            preds = preds[0]
            mask = preds[..., 4] > self.conf_thres
            preds = [p[mask[idx]] for idx, p in enumerate(preds)]
            # sort by confidence
            preds = [p[p[:, 4].argsort(descending=True)] for p in preds]
            preds = [p[:self.max_det] for p in preds]
        else:
            preds = ops.non_max_suppression(
                preds,
                nc=self.nc,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                max_det=self.max_det,
            )

        if len(preds) == 0:
            # return a dummy box with size of all image
            empty_pred = torch.zeros((1, 6), device=self.device)
            return [empty_pred]
        
        results = []
        for i, pred in enumerate(preds):
            pred[:, :4] = clip_boxes(pred[:, :4], image_sizes[i])

            img_size = image_sizes[i]
            pred[:,5] = pred[:,5] + 1
            # add img_size to the prediction at index 6 and 7

            img_size_repeated_0 = torch.full((pred.shape[0], 1), img_size[0], device=self.device)
            img_size_repeated_1 = torch.full((pred.shape[0], 1), img_size[1], device=self.device)
            
            pred = torch.cat((pred, img_size_repeated_0, img_size_repeated_1), dim=1)
            results.append(pred)
        return results
    
    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}