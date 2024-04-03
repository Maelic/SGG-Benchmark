# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from sgg_benchmark import _C

from torch.cuda.amp import custom_fwd

# Only valid with fp32 inputs - give AMP the hint
nms = custom_fwd(_C.nms)

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""