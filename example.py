#!/usr/bin/env python3
"""Example script demonstrating the new modular DreamPRM architecture.

This script shows how to use individual components of the restructured
DreamPRM codebase for custom training scenarios or research.
"""

import torch
from transformers import AutoTokenizer

# Import our modular components
from config import TrainingConfig
from data import PRMDataset, DataCollatorPRM
from models import build_prm_model, count_model_parameters
from training import BilevelTrainer
from utils import setup_logging, set_random_seed

def example_basic_usage():
    """Example of basic usage with the new architecture."""
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting DreamPRM example")
    
    # Create configuration (you can also use parse_arguments() for CLI)
    config = TrainingConfig(
        train_json_file="data/train_small.jsonl",
        meta_json_file="data/meta_small.jsonl",
        model_name="Qwen/Qwen2.5-Math-PRM-7B",
        total_steps=100,  # Small example
        batch_size=1,
        lr=1e-4,
        dry_run=True  # Just validate, don't actually train
    )
    
    # Set seed for reproducibility
    set_random_seed(config.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get step separator token
    step_separator_id = tokenizer.encode("<extra_0>")[0]
    logger.info(f"Step separator token ID: {step_separator_id}")
    
    # Create datasets
    logger.info("Loading datasets...")
    try:
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
        
        logger.info(f"Loaded {len(train_dataset)} training samples")
        logger.info(f"Loaded {len(meta_dataset)} meta samples")
        
    except FileNotFoundError as e:
        logger.warning(f"Data files not found: {e}")
        logger.info("This is expected in the example - create your data files to run actual training")
        return
    
    # Build model
    logger.info("Building PRM model...")
    model = build_prm_model(
        model_name=config.model_name,
        use_bf16=(config.precision == "bf16"),
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout
    )
    
    # Count parameters
    total_params, trainable_params = count_model_parameters(model)
    logger.info(f"Model has {total_params:,} total parameters")
    logger.info(f"Model has {trainable_params:,} trainable parameters")
    
    logger.info("Example completed successfully!")


def example_custom_training():
    """Example of setting up custom training with the modular components."""
    
    logger = setup_logging()
    logger.info("Custom training example")
    
    # You can create custom configurations programmatically
    config = TrainingConfig()
    config.model_name = "microsoft/DialoGPT-medium"  # Smaller model for example
    config.total_steps = 50
    config.lr = 5e-5
    config.batch_size = 2
    
    # Or customize specific aspects
    config.lora_r = 8  # Smaller LoRA rank
    config.precision = "fp32"  # Use FP32 for compatibility
    
    logger.info(f"Using custom config: {config}")


def example_individual_components():
    """Example of using individual components separately."""
    
    logger = setup_logging()
    logger.info("Individual components example")
    
    # Example: Just test the data processing
    try:
        from data.processing import read_jsonl, extract_steps_from_cot_response
        
        # Simulate some data
        sample_data = {
            "cot_response": """
            ### Step 1: First, analyze the problem
            We need to understand what's being asked.
            
            ### Step 2: Apply the relevant theorem
            Using the fundamental theorem, we can proceed.
            
            ### Step 3: Solve the equation
            Substituting the values gives us the answer.
            """
        }
        
        steps = extract_steps_from_cot_response(sample_data["cot_response"])
        logger.info(f"Extracted {len(steps)} steps from CoT response")
        for i, step in enumerate(steps, 1):
            logger.info(f"Step {i}: {step[:50]}...")
            
    except Exception as e:
        logger.error(f"Error in data processing example: {e}")
    
    # Example: Test model building without loading weights
    try:
        from models.prm_model import _create_lora_config
        
        lora_config = _create_lora_config(lora_r=16, lora_alpha=32, lora_dropout=0.1)
        logger.info(f"Created LoRA config: {lora_config}")
        
    except Exception as e:
        logger.error(f"Error in model example: {e}")


if __name__ == "__main__":
    print("=== DreamPRM Modular Architecture Examples ===\n")
    
    print("1. Basic usage example:")
    example_basic_usage()
    
    print("\n2. Custom training configuration:")
    example_custom_training()
    
    print("\n3. Individual components:")
    example_individual_components()
    
    print("\n=== Examples completed ===")
    print("\nTo run actual training, ensure you have:")
    print("1. Training data in JSONL format")
    print("2. Meta-learning data in JSONL format") 
    print("3. Required dependencies installed (see requirements.txt)")
    print("4. Sufficient GPU memory for the chosen model")
    print("\nThen run: python main.py --train_json_file <path> --meta_json_file <path>")