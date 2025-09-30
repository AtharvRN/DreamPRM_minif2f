"""Common utility functions for DreamPRM.

This module contains general-purpose utilities that are used
across different parts of the codebase.
"""

import logging
import random
from typing import Optional
import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.getLogger(__name__).info(f"Set random seed to {seed}")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = "training.log"
) -> logging.Logger:
    """Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path. If None, only console output.
        
    Returns:
        Logger instance for the calling module
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup handlers
    handlers = [logging.StreamHandler()]  # Console output
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Return logger for the calling module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level {log_level}")
    
    return logger


def get_device(preferred_device: str = "cuda") -> torch.device:
    """Get the best available device for computation.
    
    Args:
        preferred_device: Preferred device ('cuda' or 'cpu')
        
    Returns:
        torch.device object for the selected device
    """
    if preferred_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logging.getLogger(__name__).info(f"Using GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        logging.getLogger(__name__).info("Using CPU")
    
    return device


def check_precision_support(precision: str) -> str:
    """Check if the requested precision is supported and return the best available.
    
    Args:
        precision: Requested precision ('bf16', 'fp16', 'fp32')
        
    Returns:
        Supported precision string
    """
    logger = logging.getLogger(__name__)
    
    if precision == "bf16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            logger.info("Using bfloat16 precision")
            return "bf16"
        else:
            logger.warning("BF16 not supported, falling back to FP16")
            precision = "fp16"
    
    if precision == "fp16":
        if torch.cuda.is_available():
            logger.info("Using float16 precision")
            return "fp16"
        else:
            logger.warning("FP16 requires CUDA, falling back to FP32")
            precision = "fp32"
    
    logger.info("Using float32 precision")
    return "fp32"


def format_number(num: int) -> str:
    """Format large numbers with commas for readability.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted number string
    """
    return f"{num:,}"


def get_memory_usage() -> dict:
    """Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage information
    """
    stats = {}
    
    if torch.cuda.is_available():
        stats['cuda_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
        stats['cuda_reserved'] = torch.cuda.memory_reserved() / 1024**3   # GB
        stats['cuda_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    return stats
