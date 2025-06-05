# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn
import torch

from sgg_benchmark.modeling import registry
from sgg_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from . import vgg
from .yolo import YoloModel
from .yoloworld import YoloWorldModel
from .yoloe import YOLOEDetectionModel

@registry.BACKBONES.register("dinov2")
def build_dinov2_backbone(cfg):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.out_channels = 768
    return model

@registry.BACKBONES.register("yolo")
def build_yolo_backbone(cfg):
    nc = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES -1

    model = YoloModel(cfg, nc=nc)
    model.out_channels = cfg.MODEL.YOLO.OUT_CHANNELS[0]

    return model

@registry.BACKBONES.register("yoloworld")
def build_yoloworld_backbone(cfg):
    nc = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES -1

    model = YoloWorldModel(cfg, nc=nc)
    model.out_channels = cfg.MODEL.YOLO.OUT_CHANNELS[0]

    return model

@registry.BACKBONES.register("yoloe")
def build_yoloworld_backbone(cfg):
    nc = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES -1

    model = YOLOEDetectionModel(cfg, nc=nc)
    model.out_channels = cfg.MODEL.YOLO.OUT_CHANNELS[0]

    return model

@registry.BACKBONES.register("yolov5")
def build_yolov5_backbone(cfg):
    nc = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES -1

    model = YoloModel(cfg, nc=nc)
    model.out_channels = cfg.MODEL.YOLO.OUT_CHANNELS[0]

    return model

@registry.BACKBONES.register("VGG-16")
def build_vgg_fpn_backbone(cfg):
    body = vgg.VGG16(cfg)
    out_channels = cfg.MODEL.VGG.VGG16_OUT_CHANNELS
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE is not None:
        assert cfg.MODEL.BACKBONE.TYPE in registry.BACKBONES, \
            "cfg.MODEL.BACKBONE.TYPE: {} are not registered in registry".format(
                cfg.MODEL.BACKBONE.TYPE
            )
        return registry.BACKBONES[cfg.MODEL.BACKBONE.TYPE](cfg)
