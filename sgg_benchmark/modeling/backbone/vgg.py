# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
OR:
    model = ResNet(
        "StemWithGN",
        "BottleneckWithGN",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""

from torch import nn

import torchvision.models as models

class VGG16(nn.Module):
    def __init__(self, cfg):
        super(VGG16, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.conv_body = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    def forward(self, x):
        output = []
        output.append(self.conv_body(x))
        return output

