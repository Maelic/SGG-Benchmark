from torch import nn

class BaseContext(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super().__init__()
        self.cfg = config
        # setting parameters
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        self.obj_decode = not (self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX or self.cfg.MODEL.BACKBONE.FREEZE)

    def forward(self, roi_features, proposals, rel_pair_idxs, logger=None):
        pass