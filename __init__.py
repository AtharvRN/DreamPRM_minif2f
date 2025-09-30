"""DreamPRM ATP - Automated Theorem Proving with Process Reward Models

This package provides a comprehensive framework for training Process Reward Models
using bilevel optimization for automated theorem proving applications.

Key Features:
- Modular architecture with clear separation of concerns
- Bilevel optimization for instance reweighting
- LoRA fine-tuning for efficient adaptation
- Comprehensive metrics tracking and checkpointing
- Support for multiple model architectures
- Robust error handling and logging

Modules:
- config: Configuration management and argument parsing
- data: Dataset loading and preprocessing utilities
- models: Model building and PRM-specific functions
- training: Training loop and optimization utilities
- utils: Common utilities and helpers

Usage:
    from ATP import TrainingConfig, BilevelTrainer
    from ATP.models import build_prm_model
    from ATP.data import PRMDataset
"""

__version__ = "2.0.0"
__author__ = "DreamPRM Team"
__description__ = "Automated Theorem Proving with Process Reward Models"

# Import main components for easy access
from .config import TrainingConfig, parse_arguments
from .data import PRMDataset, DataCollatorPRM
from .models import build_prm_model
from .training import BilevelTrainer
from .utils import setup_logging, set_random_seed

__all__ = [
    # Configuration
    "TrainingConfig",
    "parse_arguments",
    
    # Data handling
    "PRMDataset", 
    "DataCollatorPRM",
    
    # Model building
    "build_prm_model",
    
    # Training
    "BilevelTrainer",
    
    # Utilities
    "setup_logging",
    "set_random_seed",
    
    # Metadata
    "__version__",
    "__author__",
    "__description__"
]
