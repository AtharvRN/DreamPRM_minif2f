"""Configuration management for DreamPRM training.

This module handles all command-line arguments and configuration
settings for the DreamPRM training pipeline.
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    
    # Data and model paths
    train_json_file: str = "train_split.json"
    meta_json_file: str = "meta_training.json"
    weights_path: str = "./checkpoints"
    model_name: str = "Qwen/Qwen2.5-Math-PRM-7B"
    
    # Training configuration
    max_epochs: int = 10
    total_steps: int = 2000
    inner_steps: int = 1
    save_every_steps: int = 500
    eval_every_steps: int = 100
    log_every_steps: int = 10
    
    # Optimization hyperparameters
    batch_size: int = 1
    meta_batch_size: int = 1
    lr: float = 2e-4
    meta_lr: float = 1e-1
    weight_decay: float = 0.01
    meta_weight_decay: float = 1e-3
    gradient_clipping: float = 1.0
    warmup_ratio: float = 0.05
    
    # Model configuration
    max_length: int = 4096
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Multi-GPU configuration
    device: str = "cuda"
    use_ddp: bool = False
    local_rank: int = -1
    world_size: int = 1
    dist_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    gradient_accumulation_steps: int = 1
    
    # System configuration
    precision: str = "bf16"
    seed: int = 42
    num_workers: int = 0
    
    # Logging and experiment tracking
    project_name: str = "DreamPRM-Text-Reasoning"
    experiment_name: Optional[str] = None
    log_level: str = "INFO"
    disable_wandb: bool = False
    
    # Resume and checkpointing
    resume_from: Optional[str] = None
    resume_step: int = 0
    
    # Special modes
    debug: bool = False
    dry_run: bool = False
    baseline: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure paths exist
        Path(self.weights_path).mkdir(parents=True, exist_ok=True)
        
        # Validate numeric values
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")
        if self.total_steps <= 0:
            raise ValueError("Total steps must be positive")


def parse_arguments() -> TrainingConfig:
    """Parse command line arguments and return a TrainingConfig object.
    
    Returns:
        TrainingConfig: Parsed configuration object
    """
    parser = argparse.ArgumentParser(
        description="DreamPRM Text Reasoning Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and model paths
    data_group = parser.add_argument_group("Data and Model")
    data_group.add_argument(
        '--train_json_file', type=str, default="train_split.json",
        help="Training data file path"
    )
    data_group.add_argument(
        '--meta_json_file', type=str, default="meta_training.json",
        help="Meta data file path"
    )
    data_group.add_argument(
        '--weights_path', type=str, default="./checkpoints",
        help="Model weights save path"
    )
    data_group.add_argument(
        '--model_name', type=str, default="Qwen/Qwen2.5-Math-PRM-7B",
        help="Base model name from HuggingFace"
    )
    
    # Training configuration
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument(
        "--max_epochs", type=int, default=10,
        help="Maximum training epochs"
    )
    training_group.add_argument(
        "--total_steps", type=int, default=2000,
        help="Total training steps"
    )
    training_group.add_argument(
        "--inner_steps", type=int, default=1,
        help="Inner loop steps per outer step"
    )
    training_group.add_argument(
        "--save_every_steps", type=int, default=500,
        help="Save checkpoint every N steps"
    )
    training_group.add_argument(
        "--eval_every_steps", type=int, default=100,
        help="Evaluate every N steps"
    )
    training_group.add_argument(
        "--log_every_steps", type=int, default=10,
        help="Log every N steps"
    )
    
    # Optimization hyperparameters
    optim_group = parser.add_argument_group("Optimization")
    optim_group.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for training"
    )
    optim_group.add_argument(
        "--meta_batch_size", type=int, default=1,
        help="Meta batch size for bilevel optimization"
    )
    optim_group.add_argument(
        "--lr", type=float, default=2e-4,
        help="Inner loop learning rate"
    )
    optim_group.add_argument(
        "--meta_lr", type=float, default=1e-1,
        help="Outer loop (meta) learning rate"
    )
    optim_group.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="Weight decay for inner loop"
    )
    optim_group.add_argument(
        "--meta_weight_decay", type=float, default=1e-3,
        help="Weight decay for meta loop"
    )
    optim_group.add_argument(
        "--gradient_clipping", type=float, default=1.0,
        help="Gradient clipping value"
    )
    optim_group.add_argument(
        "--warmup_ratio", type=float, default=0.05,
        help="Warmup ratio for scheduler"
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--max_length", type=int, default=4096,
        help="Maximum sequence length"
    )
    model_group.add_argument(
        "--use_lora", action="store_true",
        help="Use LoRA for parameter-efficient fine-tuning"
    )
    model_group.add_argument(
        "--no_lora", dest="use_lora", action="store_false",
        help="Disable LoRA (full fine-tuning)"
    )
    model_group.add_argument(
        "--lora_r", type=int, default=16,
        help="LoRA rank (dimensionality of adaptation)"
    )
    model_group.add_argument(
        "--lora_alpha", type=int, default=32,
        help="LoRA alpha (scaling parameter)"
    )
    model_group.add_argument(
        "--lora_dropout", type=float, default=0.05,
        help="LoRA dropout rate"
    )
    
    # System configuration
    system_group = parser.add_argument_group("System Configuration")
    system_group.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda/cpu)"
    )
    system_group.add_argument(
        "--use_ddp", action="store_true",
        help="Use DistributedDataParallel for multi-GPU training"
    )
    system_group.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed training"
    )
    system_group.add_argument(
        "--world_size", type=int, default=1,
        help="Number of GPUs for distributed training"
    )
    system_group.add_argument(
        "--dist_backend", type=str, default="nccl",
        help="Distributed backend (nccl/gloo)"
    )
    system_group.add_argument(
        "--ddp_find_unused_parameters", action="store_true",
        help="Find unused parameters in DDP (slower but more robust)"
    )
    system_group.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Gradient accumulation steps for larger effective batch size"
    )
    system_group.add_argument(
        "--precision", type=str, default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Training precision mode"
    )
    system_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    system_group.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader number of workers"
    )
    
    # Logging and experiment tracking
    logging_group = parser.add_argument_group("Logging and Tracking")
    logging_group.add_argument(
        "--project_name", type=str, default="DreamPRM-Text-Reasoning",
        help="Wandb project name"
    )
    logging_group.add_argument(
        "--experiment_name", type=str, default=None,
        help="Experiment name (auto-generated if None)"
    )
    logging_group.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    logging_group.add_argument(
        "--disable_wandb", action="store_true",
        help="Disable wandb logging"
    )
    
    # Resume and checkpointing
    resume_group = parser.add_argument_group("Resume and Checkpointing")
    resume_group.add_argument(
        "--resume_from", type=str, default=None,
        help="Resume from checkpoint path"
    )
    resume_group.add_argument(
        "--resume_step", type=int, default=0,
        help="Resume from specific step number"
    )
    
    # Special modes
    modes_group = parser.add_argument_group("Special Modes")
    modes_group.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode with detailed error reporting"
    )
    modes_group.add_argument(
        "--dry_run", action="store_true",
        help="Dry run without actual training"
    )
    modes_group.add_argument(
        "--baseline", action="store_true",
        help="Run baseline without bilevel optimization"
    )
    
    # Set defaults for boolean flags
    parser.set_defaults(use_lora=True)
    
    args = parser.parse_args()
    
    # Convert to dataclass
    config = TrainingConfig(**vars(args))
    
    return config
