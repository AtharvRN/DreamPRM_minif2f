"""Model definitions and utilities for DreamPRM.

This module contains model building functions, loss functions,
and PRM-specific utilities for the DreamPRM training pipeline.
"""

from models.prm_model import build_prm_model
from models.losses import (
    make_step_rewards,
    extract_step_probabilities,
    compute_aggregate_score,
    PRMLossFunction
)

__all__ = [
    "build_prm_model",
    "make_step_rewards",
    "extract_step_probabilities",
    "compute_aggregate_score",
    "PRMLossFunction"
]
