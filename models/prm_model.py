"""Process Reward Model building utilities.

This module contains functions for building and configuring
Process Reward Models with LoRA fine-tuning.
"""

import logging
from typing import Optional
import torch
from transformers import AutoModel, PreTrainedModel
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


def build_prm_model(
    model_name: str,
    use_bf16: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    device_map: str = "auto"
) -> PreTrainedModel:
    """Build a Process Reward Model with LoRA fine-tuning.
    
    Args:
        model_name: HuggingFace model name/path
        use_bf16: Whether to use bfloat16 precision
        lora_r: LoRA rank (dimensionality of adaptation)
        lora_alpha: LoRA alpha scaling parameter
        lora_dropout: LoRA dropout rate
        device_map: Device mapping strategy for model loading
        
    Returns:
        PeftModel: Model with LoRA adapters attached
        
    Raises:
        ValueError: If model loading or LoRA configuration fails
    """
    logger.info(f"Building PRM model from {model_name}")
    
    # Determine precision
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    logger.info(f"Using precision: {dtype}")
    
    try:
        # Load base model
        base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map
        )
        
        # Configure model for training
        _configure_model_for_training(base_model)
        
        # Set up LoRA configuration
        lora_config = _create_lora_config(
            lora_r, lora_alpha, lora_dropout
        )
        
        # Apply LoRA to the model
        peft_model = get_peft_model(base_model, lora_config)
        
        # Log model information
        _log_model_info(peft_model)
        
        return peft_model
        
    except Exception as e:
        raise ValueError(f"Failed to build PRM model: {e}") from e


def _configure_model_for_training(model: PreTrainedModel) -> None:
    """Configure model settings for training.
    
    Args:
        model: The model to configure
    """
    # Disable caching for training
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
        logger.debug("Disabled model caching for training")
    
    # Enable gradient checkpointing for memory efficiency
    try:
        # Try new gradient checkpointing API first
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.debug("Enabled gradient checkpointing (non-reentrant)")
    except Exception:
        # Fallback to old API
        try:
            model.gradient_checkpointing_enable()
            logger.debug("Enabled gradient checkpointing (fallback)")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")


def _create_lora_config(
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float
) -> LoraConfig:
    """Create LoRA configuration for the model.
    
    Args:
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout rate
        
    Returns:
        LoraConfig: Configured LoRA settings
    """
    # Define target modules for LoRA (common transformer attention/MLP layers)
    target_modules = [
        "q_proj",    # Query projection
        "k_proj",    # Key projection
        "v_proj",    # Value projection
        "o_proj",    # Output projection
        "gate_proj", # Gate projection (for some architectures)
        "up_proj",   # Up projection (MLP)
        "down_proj"  # Down projection (MLP)
    ]
    
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=target_modules
    )
    
    logger.info(
        f"LoRA config: r={lora_r}, alpha={lora_alpha}, "
        f"dropout={lora_dropout}, targets={target_modules}"
    )
    
    return config


def _log_model_info(model: PreTrainedModel) -> None:
    """Log information about the model parameters.
    
    Args:
        model: The model to analyze
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(
        f"Model parameters - Total: {total_params:,}, "
        f"Trainable: {trainable_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )


def count_model_parameters(model: PreTrainedModel) -> tuple[int, int]:
    """Count total and trainable parameters in a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Tuple of (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params
