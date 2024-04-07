import torch
from torch import nn

from sgg_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..roi_heads.roi_heads import build_roi_heads


class SimrelRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(SimrelRCNN, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        
    def forward(self, images, targets=None, logger=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
            
            # 
            targets[0]: BoxList(num_boxes=26, image_width=800, image_height=600, mode=xyxy)
                        targets[0].size[0] == 800, targets[0].size[0] == 600, 
            targets: [BoxList(num_boxes=26, image_width=800, image_height=600, mode=xyxy)]
            targets[0].bbox.shape: torch.Size([26, 4]); x < image_width, y < image_height
            targets[0].get_field("labels").long().shape: torch.Size([26])
            targets[0].get_field("relation").shape: torch.Size([26, 26]); elements >= 0

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        
        x, result, detector_losses = self.roi_heads(features, None, targets, logger, images.tensors)

        if self.training:
            losses = {}
            losses.update(detector_losses)

            return losses

        return result