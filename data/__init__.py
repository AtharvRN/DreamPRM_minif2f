"""Data handling module for DreamPRM.

This module contains dataset classes, data processing utilities,
and collators for the DreamPRM training pipeline.
"""

from .dataset import PRMDataset
from .processing import (
    read_jsonl,
    extract_steps_from_cot_response,
    extract_rewards_from_cot_steps,
    DataCollatorPRM
)

__all__ = [
    "PRMDataset",
    "read_jsonl",
    "extract_steps_from_cot_response", 
    "extract_rewards_from_cot_steps",
    "DataCollatorPRM"
]
