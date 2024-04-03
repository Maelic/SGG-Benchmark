
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.GLOBAL_SETTING = CN()
_C.GLOBAL_SETTING.BASIC_ENCODER = 'Hybrid-Attention'
_C.GLOBAL_SETTING.USE_BIAS = True

_C.GLOBAL_SETTING.GCL_SETTING = CN()
_C.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE = 'divide4'
_C.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT = 1.0
_C.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE = 'KL_logit_TopDown'
_C.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN = True
_C.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE = 'rand_insert'
_C.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_PENALTY = 0.1
