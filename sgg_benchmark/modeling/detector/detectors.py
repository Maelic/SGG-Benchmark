# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .generalized_yolo import GeneralizedYOLO


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN, "GeneralizedYOLO": GeneralizedYOLO}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
