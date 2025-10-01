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
import os

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import our modular components
from config import parse_arguments, TrainingConfig
from data import PRMDataset, DataCollatorPRM
from models import build_prm_model
from training import BilevelTrainer
from training.distributed import (
    setup_distributed_training, 
    cleanup_distributed_training,
    is_main_process,
    get_rank,
    get_world_size
)
from utils import setup_logging, set_random_seed, get_device, check_precision_support

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
    # Only initialize wandb on main process in distributed training
    if config.disable_wandb or not is_main_process():
        if not is_main_process():
            logger.info("Weights & Biases disabled on non-main process")
        else:
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
        use_lora=config.use_lora,
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
    if is_main_process():
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
    
    if is_main_process():
        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Meta: {len(meta_dataset)}")
    
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty!")
    
    if len(meta_dataset) == 0:
        raise ValueError("Meta dataset is empty!")
    
    # Create data collator
    collator = DataCollatorPRM(tokenizer.pad_token_id)
    
    # Import distributed utilities
    from training.distributed import create_distributed_dataloader
    
    # Create data loaders with distributed support
    train_loader = create_distributed_dataloader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    meta_loader = create_distributed_dataloader(
        dataset=meta_dataset,
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
    # Only log from main process in distributed training
    if not is_main_process():
        return
        
    logger.info("=== System Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    if config.use_ddp:
        logger.info(f"Distributed training: rank={get_rank()}/{get_world_size()}")
    
    logger.info("=== Training Configuration ===")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Precision: {config.precision}")
    logger.info(f"Batch size: {config.batch_size} (per process)")
    logger.info(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps * get_world_size()}")
    logger.info(f"Learning rate: {config.lr}")
    logger.info(f"Total steps: {config.total_steps}")
    logger.info(f"LoRA enabled: {config.use_lora}")
    if config.use_lora:
        logger.info(f"LoRA config: r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")


def main() -> int:
    """Main training function.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse configuration
        config = parse_arguments()
        
        # Setup distributed training if requested
        rank, world_size, is_distributed = setup_distributed_training(
            local_rank=config.local_rank,
            world_size=config.world_size,
            backend=config.dist_backend
        )
        
        # Update config with distributed info
        config.use_ddp = is_distributed
        config.local_rank = rank if is_distributed else -1
        config.world_size = world_size
        
        # Setup logging (only detailed logging on main process)
        log_level = config.log_level if is_main_process() else "WARNING"
        setup_logging(log_level)
        
        if is_main_process():
            logger.info("DreamPRM Training Started")
        
        # Set random seed for reproducibility
        set_random_seed(config.seed + rank)  # Different seed per process
        
        # Validate configuration and environment
        validate_data_files(config)
        
        # Check device and precision support
        device = get_device(config.device)
        if is_distributed:
            device = torch.device(f"cuda:{rank}")
        else:
            # Force to use only GPU 0 to avoid multi-GPU conflicts
            if torch.cuda.is_available() and config.device == "cuda":
                device = torch.device("cuda:0")
                torch.cuda.set_device(0)
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
            if is_main_process():
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
        
        if is_main_process():
            logger.info("Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        if is_main_process():
            logger.info("Training interrupted by user")
        return 0
    except Exception as e:
        if is_main_process():
            logger.error(f"Training failed: {e}")
        if hasattr(config, 'debug') and config.debug:
            raise
        return 1
    finally:
        # Cleanup
        try:
            if is_main_process() and 'config' in locals() and not config.disable_wandb:
                wandb.finish()
        except:
            pass
        
        # Cleanup distributed training
        if 'config' in locals() and config.use_ddp:
            cleanup_distributed_training()
        
        if is_main_process():
            logger.info("DreamPRM Training Finished")


if __name__ == "__main__":
    sys.exit(main())
