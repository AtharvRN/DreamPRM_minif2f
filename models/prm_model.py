"""Process Reward Model building utilities.

This module contains functions for building and configuring
Process Reward Models with LoRA fine-tuning.
"""

import logging
from typing import Optional
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, PreTrainedModel
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


def build_prm_model(
    model_name: str,
    use_bf16: bool = True,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    device_map: Optional[str] = "auto"
) -> PreTrainedModel:
    """Build a Process Reward Model with optional LoRA fine-tuning.
    
    Args:
        model_name: HuggingFace model name/path
        use_bf16: Whether to use bfloat16 precision
        use_lora: Whether to use LoRA for parameter-efficient fine-tuning
        lora_r: LoRA rank (dimensionality of adaptation)
        lora_alpha: LoRA alpha scaling parameter
        lora_dropout: LoRA dropout rate
        device_map: Device mapping strategy for model loading
        
    Returns:
        PreTrainedModel or PeftModel: Model with optional LoRA adapters
        
    Raises:
        ValueError: If model loading or LoRA configuration fails
    """
    logger.info(f"Building PRM model from {model_name} (LoRA: {use_lora})")
    
    # Determine precision
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    logger.info(f"Using precision: {dtype}")
    
    try:
        # Load base model using official approach
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "device_map": device_map,  # Use device_map (default "auto" for multi-GPU)
        }
        
        # Load the model directly with AutoModel (official approach)
        base_model = AutoModel.from_pretrained(
            model_name,
            **model_kwargs
        )
        logger.info(f"Loaded model type: {type(base_model).__name__}")
        
        # Log device mapping if available
        if hasattr(base_model, 'hf_device_map'):
            logger.info(f"Model device map: {base_model.hf_device_map}")
        else:
            logger.info("No device mapping information available")
        
        # Configure model for training
        _configure_model_for_training(base_model)
        
        # Ensure model is in training mode and parameters require gradients
        base_model.train()
        for param in base_model.parameters():
            param.requires_grad = True
        logger.info("Ensured all model parameters require gradients")
        
        if use_lora:
            # Set up LoRA configuration
            lora_config = _create_lora_config(
                lora_r, lora_alpha, lora_dropout
            )
            
            # Apply LoRA to the model
            model = get_peft_model(base_model, lora_config)
            logger.info("LoRA adapters applied successfully")
        else:
            # Use full fine-tuning
            model = base_model
            logger.info("Using full fine-tuning (no LoRA)")
            
            # For full fine-tuning without device_map, we'll handle multi-GPU in trainer
            logger.info("Model will use PyTorch native multi-GPU support")
        
        # Log model information
        _log_model_info(model)
        
        return model
        
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
