# Models Map

from .model_gpsnet import GPSNetContext
from .model_Cross_Attention import CA_Context
from .model_Hybrid_Attention import SHA_Context
from .model_penet import PENetContext
from .model_motifs import LSTMContext, LSTMContext_RNN
from .model_msg_passing import IMPContext
from .model_transformer import TransformerContext
from .model_vctree import VCTreeLSTMContext
from .model_vtranse import VTransEFeature

MODEL_MAP = {
    "gpsnet": GPSNetContext,
    "cross_att": CA_Context,
    "stacked_att": SHA_Context,
    "penet": PENetContext,
    "motifs": LSTMContext,
    "imp": IMPContext,
    "transformer": TransformerContext,
    "vctree": VCTreeLSTMContext,
    "vtranse": VTransEFeature,
}