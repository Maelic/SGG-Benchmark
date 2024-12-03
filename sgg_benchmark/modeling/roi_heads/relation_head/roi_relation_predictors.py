from sgg_benchmark.modeling import registry

from .predictors.default_predictors import *
from .predictors.GCL_predictors import *
from .predictors.bayesian_predictors import *
from .predictors.regularized_predictors import *
from .predictors.react_predictor import *

def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
