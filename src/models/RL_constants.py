from src.models.feature_rl import (
    FeatureRL,
    DecayFeatureRL,
)
from collections import namedtuple
import scipy.stats as ss

ModelInfo = namedtuple(
    "ModelInfo",
    ["model", "prior_distributions", "bounds", "modelname", "parameter_names"],
)


MODELS = {
    "featurerl": ModelInfo(
        FeatureRL,
        [ss.beta(2, 2), ss.gamma(4.82, scale=0.88), ss.norm(0, 1)],
        [(1e-9, 1 - (1e-9)), (1e-9, 10), (-20, 20)],
        "FeatureRL",
        ["eta", "beta", "resp_st"],
    ),
    "decayfeaturerl": ModelInfo(
        DecayFeatureRL,
        [ss.beta(2, 2), ss.gamma(3, scale=0.2), ss.beta(2, 2), ss.norm(0, 1)],
        [(1e-9, 1 - (1e-9)), (1e-9, 10), (1e-9, 1 - (1e-9)), (-20, 20)],
        "DecayFeatureRL",
        ["eta", "beta", "decay", "resp_st"],
    ),
}
