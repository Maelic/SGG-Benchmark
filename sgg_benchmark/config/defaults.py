# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

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

_C.MODEL = CN()
_C.MODEL.FLIP_AUG = False
_C.MODEL.RPN_ONLY = False
_C.MODEL.MASK_ON = False
_C.MODEL.ATTRIBUTE_ON = False
_C.MODEL.RELATION_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN" # GeneralizedYOLO
_C.MODEL.CLS_AGNOSTIC_BBOX_REG = False

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ""

# checkpoint of detector, for relation prediction
_C.MODEL.PRETRAINED_DETECTOR_CKPT = ""

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = 800  # (800,)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = True
_C.INPUT.FLIP_PROB_TRAIN = 0.5

# For YOLOV8
_C.INPUT.PADDING = False

# Image ColorJitter
_C.INPUT.BRIGHTNESS = 0.0
_C.INPUT.CONTRAST = 0.0
_C.INPUT.SATURATION = 0.0
_C.INPUT.HUE = 0.0

_C.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for val, as present in paths_catalog.py
# Note that except dataset names, all remaining val configs reuse those of test
_C.DATASETS.VAL = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
_C.DATASETS.NAME = ""
_C.DATASETS.TO_TEST = ""

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True


# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
_C.MODEL.BACKBONE.TYPE = "R-50-C4" # Can be yolo or yolov5 or VGG-16 or R-50-C4 or R-50-C5 or R-101-C4 or R-101-C5 or R-50-FPN or R-101-FPN or R-152-FPN
_C.MODEL.BACKBONE.EXTRA_CONFIG = ""
# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2
# NMS threshold used on backbone proposals
_C.MODEL.BACKBONE.NMS_THRESH = 0.7
_C.MODEL.BACKBONE.FREEZE = False # to control whether to freeze the backbone for relations prediction, i.e. the preds will be counted as GT

# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
_C.MODEL.FPN.USE_GN = False
_C.MODEL.FPN.USE_RELU = False


# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
_C.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
_C.MODEL.GROUP_NORM.EPSILON = 1e-5


# ---------------------------------------------------------------------------- #
# YOLO options
# ---------------------------------------------------------------------------- #
_C.MODEL.YOLO = CN()
_C.MODEL.YOLO.WEIGHTS = ""
_C.MODEL.YOLO.SIZE = "yolov8l" # Can be nano, small, medium or large
_C.MODEL.YOLO.IMG_SIZE = 640 # input image size
_C.MODEL.YOLO.OUT_CHANNELS = [192,384,576] # dim for the last layer of the YOLOv8 head
# for YOLOV8, 9, 10: [192,384,576]
# for YOLOV11: [256,512,512]

_C.MODEL.BOX_HEAD = True # for Yolov8 we do not need box head

# ---------------------------------------------------------------------------- #
# Text Embeddings options (for object and relation encoding)
# ---------------------------------------------------------------------------- #

_C.MODEL.TEXT_EMBEDDING = "glove.6B" # clip, glove.6B

# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.USE_FPN = False
_C.MODEL.RPN.RPN_MID_CHANNEL = 512
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
_C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
_C.MODEL.RPN.ANCHOR_STRIDE = (16,)
# RPN anchor aspect ratios
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.STRADDLE_THRESH = 0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
# Total number of RPN examples per image
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000

# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.RPN.MIN_SIZE = 0
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000
# Apply the post NMS per batch (default) or per image during training
# (default is True to be consistent with Detectron, see Issue #672)
_C.MODEL.RPN.FPN_POST_NMS_PER_BATCH = True
# Custom rpn head, empty to use default conv or separable conv
_C.MODEL.RPN.RPN_HEAD = "SingleConvRPNHead"


# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.3

# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
_C.MODEL.ROI_HEADS.SCORE_THRESH = 0.01
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.ROI_HEADS.NMS = 0.3
_C.MODEL.ROI_HEADS.POST_NMS_PER_CLS_TOPN = 300
# Remove duplicated assigned labels for a single bbox in nms
_C.MODEL.ROI_HEADS.NMS_FILTER_DUPLICATES = False 
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100


_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 151
# Hidden layer dimension when using an MLP for the RoI box head
_C.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 2048
# GN
_C.MODEL.ROI_BOX_HEAD.USE_GN = False
# Dilation
_C.MODEL.ROI_BOX_HEAD.DILATION = 1
_C.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
_C.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4



_C.MODEL.ROI_ATTRIBUTE_HEAD = CN()
_C.MODEL.ROI_ATTRIBUTE_HEAD.FEATURE_EXTRACTOR = "FPN2MLPFeatureExtractor"
_C.MODEL.ROI_ATTRIBUTE_HEAD.PREDICTOR = "FPNPredictor"
_C.MODEL.ROI_ATTRIBUTE_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
# Add attributes to each box
_C.MODEL.ROI_ATTRIBUTE_HEAD.USE_BINARY_LOSS = True
_C.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_LOSS_WEIGHT = 0.1
_C.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES = 201
_C.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES = 10  # max number of attribute per bbox
_C.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE = True
_C.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO = 3
_C.MODEL.ROI_ATTRIBUTE_HEAD.POS_WEIGHT = 5.0


_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
_C.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
# Whether or not resize and translate masks to the input image.
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS = False
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5
# Dilation
_C.MODEL.ROI_MASK_HEAD.DILATION = 1
# GN
_C.MODEL.ROI_MASK_HEAD.USE_GN = False


_C.MODEL.ROI_RELATION_HEAD = CN()
# share box feature extractor should be set False for neural-motifs
_C.MODEL.ROI_RELATION_HEAD.PREDICTOR = "MotifPredictor"
_C.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR = "RelationFeatureExtractor"
_C.MODEL.ROI_RELATION_HEAD.USE_UNION_FEATURES = True
_C.MODEL.ROI_RELATION_HEAD.USE_SPATIAL_FEATURES = True
_C.MODEL.ROI_RELATION_HEAD.TEXTUAL_FEATURES_ONLY = False
_C.MODEL.ROI_RELATION_HEAD.VISUAL_FEATURES_ONLY = False
_C.MODEL.ROI_RELATION_HEAD.LOGIT_ADJUSTMENT = False
_C.MODEL.ROI_RELATION_HEAD.LOGIT_ADJUSTMENT_TAU = 0.3

_C.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS = True
_C.MODEL.ROI_RELATION_HEAD.NUM_CLASSES = 51
_C.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE = 64
_C.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION = 0.25
_C.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = True
_C.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
_C.MODEL.ROI_RELATION_HEAD.EMBED_DIM = 200
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE = 0.2
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM = 512
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM = 4096
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER = 1  # assert >= 1
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER = 1  # assert >= 1
_C.MODEL.ROI_RELATION_HEAD.MLP_HEAD_DIM = 2048 # for PENET

_C.MODEL.ROI_RELATION_HEAD.LOSS = "CrossEntropyLoss" # can be "CrossEntropyLoss", "FocalLoss", "ReweightingCE"

_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER = CN()
# for TransformerPredictor only
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE = 0.1   
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER = 4        
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER = 2        
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD = 8         
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM = 2048     
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM = 64         
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM = 64

_C.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE = CN()
_C.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.PRE_NORM = False
_C.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.NUM_DECODER = 3
_C.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.RHO = 0.35
_C.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.BETA = 0.7

_C.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.PRETRAIN_MASK = False
_C.MODEL.ROI_RELATION_HEAD.SQUAT_MODULE.PRETRAIN_MASK_EPOCH = 1


_C.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS = False
_C.MODEL.ROI_RELATION_HEAD.USE_FREQUENCY_BIAS = False
_C.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP = True
_C.MODEL.ROI_RELATION_HEAD.NUM_SAMPLE_PER_GT_REL = 4  # when sample fg relationship from gt, the max number of corresponding proposal pairs

# in sgdet, to make sure the detector won't missing any ground truth bbox, 
# we add grount truth box to the output of RPN proposals during Training
_C.MODEL.ROI_RELATION_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN = False

_C.MODEL.ROI_RELATION_HEAD.CLASSIFIER = "linear"  # weighted_norm, linear, cosine_similarity

_C.MODEL.ROI_RELATION_HEAD.CAUSAL = CN()
# direct and indirect effect analysis
_C.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS = False
# Fusion
_C.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE = 'sum'
# causal context feature layer
_C.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER = 'motifs'
# separate spatial in union feature
_C.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL = False

_C.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE = 'none' # 'TDE', 'TIE', 'TE'

### LEGACY BUT KEEPING IT FOR BACKWARD COMPATIBILITY
_C.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION = False
_C.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION = False


_C.MODEL.VGG = CN()
_C.MODEL.VGG.VGG16_OUT_CHANNELS= 512

# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
_C.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

_C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

_C.MODEL.RESNETS.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.RESNETS.WITH_MODULATED_DCN = False
_C.MODEL.RESNETS.DEFORMABLE_GROUPS = 1


# ---------------------------------------------------------------------------- #
# FBNet options
# ---------------------------------------------------------------------------- #
_C.MODEL.FBNET = CN()
_C.MODEL.FBNET.ARCH = "default"
# custom arch
_C.MODEL.FBNET.ARCH_DEF = ""
_C.MODEL.FBNET.BN_TYPE = "bn"
_C.MODEL.FBNET.SCALE_FACTOR = 1.0
# the output channels will be divisible by WIDTH_DIVISOR
_C.MODEL.FBNET.WIDTH_DIVISOR = 1
_C.MODEL.FBNET.DW_CONV_SKIP_BN = True
_C.MODEL.FBNET.DW_CONV_SKIP_RELU = True

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.DET_HEAD_LAST_SCALE = 1.0
_C.MODEL.FBNET.DET_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.DET_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.KPTS_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.KPTS_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.KPTS_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.MASK_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.MASK_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.MASK_HEAD_STRIDE = 0

# 0 to use all blocks defined in arch_def
_C.MODEL.FBNET.RPN_HEAD_BLOCKS = 0
_C.MODEL.FBNET.RPN_BN_TYPE = ""


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000
_C.SOLVER.MAX_EPOCH = 100

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 1

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0
_C.SOLVER.CLIP_NORM = 5.0

_C.SOLVER.GAMMA = 0.5
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.SCHEDULE = CN()
_C.SOLVER.SCHEDULE.TYPE = "WarmupMultiStepLR"  # "WarmupReduceLROnPlateau"
# the following paramters are only used for WarmupReduceLROnPlateau
_C.SOLVER.SCHEDULE.PATIENCE = 2
_C.SOLVER.SCHEDULE.THRESHOLD = 1e-4
_C.SOLVER.SCHEDULE.COOLDOWN = 1
_C.SOLVER.SCHEDULE.FACTOR = 0.5
_C.SOLVER.SCHEDULE.MAX_DECAY_STEP = 7


_C.SOLVER.CHECKPOINT_PERIOD = 2500

_C.SOLVER.GRAD_NORM_CLIP = 5.0

_C.SOLVER.PRINT_GRAD_FREQ = 5000
# whether validate and validate period
_C.SOLVER.TO_VAL = True
_C.SOLVER.PRE_VAL = True
_C.SOLVER.VAL_PERIOD = 2500

# update schedule
# when load from a previous model, if set to True
# only maintain the iteration number and all the other settings of the 
# schedule will be changed
_C.SOLVER.UPDATE_SCHEDULE_DURING_LOAD = False

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 8
_C.SOLVER.OPTIMIZER = "SGD"  # "ADAMW"

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 8
# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 100
_C.TEST.INFORMATIVE = False

# ---------------------------------------------------------------------------- #
# Test-time augmentations for bounding box detection
# See configs/test_time_aug/e2e_mask_rcnn_R-50-FPN_1x.yaml for an example
# ---------------------------------------------------------------------------- #
_C.TEST.BBOX_AUG = CN()

# Enable test-time augmentation for bounding box detection if True
_C.TEST.BBOX_AUG.ENABLED = False

# Horizontal flip at the original scale (id transform)
_C.TEST.BBOX_AUG.H_FLIP = False

# Each scale is the pixel size of an image's shortest side
_C.TEST.BBOX_AUG.SCALES = ()

# Max pixel size of the longer side
_C.TEST.BBOX_AUG.MAX_SIZE = 4000

# Horizontal flip at each scale
_C.TEST.BBOX_AUG.SCALE_H_FLIP = False

_C.TEST.SAVE_PROPOSALS = False
# Settings for relation testing
_C.TEST.RELATION = CN()
_C.TEST.RELATION.MULTIPLE_PREDS = False
_C.TEST.RELATION.IOU_THRESHOLD = 0.5
_C.TEST.RELATION.REQUIRE_OVERLAP = True
# when predict the label of bbox, run nms on each cls
_C.TEST.RELATION.LATER_NMS_PREDICTION_THRES = 0.3 
# synchronize_gather, used for sgdet, otherwise test on multi-gpu will cause out of memory
_C.TEST.RELATION.SYNC_GATHER = False

_C.TEST.ALLOW_LOAD_FROM_CACHE = True

_C.TEST.CUSTUM_EVAL = False
_C.TEST.CUSTUM_PATH = '.'
_C.TEST.TOP_K = 100

_C.TEST.COMPUTE_DCS = False


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.DETECTED_SGG_DIR = "."
_C.GLOVE_DIR = "."
_C.VERBOSE = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
_C.SEED = 42 # -1 to desactivate

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
_C.PATHS_DATA = os.path.join(os.path.dirname(__file__), "/home/Documents/Datasets/VG/")

_C.METRIC_TO_TRACK = "mR" # mR (mean Recall), R (Recall), zR (zero-shot Recall), ng-R (no graph constraint Recall),
                          # ng-mR (no graph constraint mean Recall), ng-zR (no graph constraint zero-shot Recall), topA (Top K Accuracy)

# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #

# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"