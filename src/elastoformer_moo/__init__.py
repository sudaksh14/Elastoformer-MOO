"""Nested shared-weight elastic inference for decoder-only small language models."""

from .arch import SUPPORTED_MODEL_TYPES, ArchInfo, UnsupportedArchitectureError
from .config import ElasticConfig, make_grid, smallest_and_largest
from .losses import distillation_kl, lm_cross_entropy, monotonicity_penalty
from .nested import ElasticController, make_elastic
from .sampling import SandwichSampler

__version__ = "0.1.0"

__all__ = [
    "SUPPORTED_MODEL_TYPES",
    "ArchInfo",
    "ElasticConfig",
    "ElasticController",
    "SandwichSampler",
    "UnsupportedArchitectureError",
    "distillation_kl",
    "lm_cross_entropy",
    "make_elastic",
    "make_grid",
    "monotonicity_penalty",
    "smallest_and_largest",
]
