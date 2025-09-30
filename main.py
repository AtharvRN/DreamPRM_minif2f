#!/usr/bin/env python3
"""Main entry point for DreamPRM training.

This script orchestrates the entire DreamPRM training pipeline,
from configuration parsing to model training completion.

Usage:
    python main.py [arguments]
    
Examples:
    # Basic training
    python main.py --train_json_file data/train.jsonl --meta_json_file data/meta.jsonl
    
    # Training with custom configuration
    python main.py --model_name Qwen/Qwen2.5-Math-PRM-7B --lr 1e-4 --total_steps 5000
    
    # Resume from checkpoint
    python main.py --resume_from checkpoints/checkpoint_step_1000.pt
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import our modular components
from .config import parse_arguments, TrainingConfig
from .data import PRMDataset, DataCollatorPRM
from .models import build_prm_model
from .training import BilevelTrainer
from .utils import setup_logging, set_random_seed, get_device, check_precision_support

logger = logging.getLogger(__name__)


def validate_data_files(config: TrainingConfig) -> None:
    """Validate that required data files exist.
    
    Args:
        config: Training configuration
        
    Raises:
        FileNotFoundError: If required data files are missing
    """
    train_path = Path(config.train_json_file)
    meta_path = Path(config.meta_json_file)
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data file not found: {config.train_json_file}")
    
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta data file not found: {config.meta_json_file}")
    
    logger.info(f"Data files validated: train={train_path}, meta={meta_path}")


def setup_experiment_tracking(config: TrainingConfig) -> None:
    """Setup experiment tracking with Weights & Biases.
    
    Args:
        config: Training configuration
    """
    if config.disable_wandb:
        logger.info("Weights & Biases logging disabled")
        return
    
    experiment_name = config.experiment_name or f"dreamprm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        wandb.init(
            project=config.project_name,
            name=experiment_name,
            config=vars(config),
            tags=["dreamprm", "prm", "bilevel-optimization"]
        )
        logger.info(f"Weights & Biases initialized: {experiment_name}")
    except Exception as e:
        logger.warning(f"Failed to initialize Weights & Biases: {e}")
        config.disable_wandb = True


def load_tokenizer_and_model(config: TrainingConfig) -> tuple[AutoTokenizer, torch.nn.Module]:
    """Load tokenizer and build the PRM model.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (tokenizer, model)
    """
    logger.info(f"Loading tokenizer and model: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, 
        trust_remote_code=True
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("Set pad_token to eos_token")
    
    # Get step separator token ID
    try:
        step_separator_id = tokenizer.encode("<extra_0>")[0]
        logger.info(f"Step separator token ID: {step_separator_id}")
    except (IndexError, KeyError):
        raise ValueError("Tokenizer does not support <extra_0> token")
    
    # Build model
    model = build_prm_model(
        model_name=config.model_name,
        use_bf16=(config.precision == "bf16"),
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout
    )
    
    return tokenizer, model, step_separator_id


def create_data_loaders(
    config: TrainingConfig, 
    tokenizer: AutoTokenizer
) -> tuple[DataLoader, DataLoader]:
    """Create training and meta data loaders.
    
    Args:
        config: Training configuration
        tokenizer: Tokenizer for text processing
        
    Returns:
        Tuple of (train_loader, meta_loader)
    """
    logger.info("Creating datasets and data loaders...")
    
    # Create datasets
    train_dataset = PRMDataset(
        jsonl_path=config.train_json_file,
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    
    meta_dataset = PRMDataset(
        jsonl_path=config.meta_json_file,
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Meta: {len(meta_dataset)}")
    
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty!")
    
    if len(meta_dataset) == 0:
        raise ValueError("Meta dataset is empty!")
    
    # Create data collator
    collator = DataCollatorPRM(tokenizer.pad_token_id)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    meta_loader = DataLoader(
        meta_dataset,
        batch_size=config.meta_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, meta_loader


def log_system_info(config: TrainingConfig) -> None:
    """Log system and configuration information.
    
    Args:
        config: Training configuration
    """
    logger.info("=== System Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    logger.info("=== Training Configuration ===")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Precision: {config.precision}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.lr}")
    logger.info(f"Total steps: {config.total_steps}")
    logger.info(f"LoRA config: r={config.lora_r}, alpha={config.lora_alpha}")


def main() -> int:
    """Main training function.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse configuration
        config = parse_arguments()
        
        # Setup logging
        setup_logging(config.log_level)
        logger.info("DreamPRM Training Started")
        
        # Set random seed for reproducibility
        set_random_seed(config.seed)
        
        # Validate configuration and environment
        validate_data_files(config)
        
        # Check device and precision support
        device = get_device(config.device)
        config.precision = check_precision_support(config.precision)
        config.device = str(device)
        
        # Log system information
        log_system_info(config)
        
        # Setup experiment tracking
        setup_experiment_tracking(config)
        
        # Load tokenizer and model
        tokenizer, model, step_separator_id = load_tokenizer_and_model(config)
        
        # Create data loaders
        train_loader, meta_loader = create_data_loaders(config, tokenizer)
        
        # Dry run check
        if config.dry_run:
            logger.info("Dry run completed successfully!")
            return 0
        
        # Create trainer and start training
        trainer = BilevelTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
            meta_loader=meta_loader,
            step_separator_id=step_separator_id
        )
        
        # Execute training
        trainer.train()
        
        logger.info("Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if hasattr(config, 'debug') and config.debug:
            raise
        return 1
    finally:
        # Cleanup
        if 'config' in locals() and not config.disable_wandb:
            try:
                wandb.finish()
            except:
                pass
        logger.info("DreamPRM Training Finished")


if __name__ == "__main__":
    sys.exit(main())
