# DreamPRM MinIF2F

A specialized implementation of DreamPRM (Process Reward Model) for the MinIF2F (Minimalist International Mathematical Olympiad) benchmark dataset.

## Overview

This project implements a Process Reward Model using bilevel optimization to train on mathematical reasoning tasks from the MinIF2F dataset. The model learns to evaluate the quality of reasoning steps in mathematical proofs.

## 🏗️ Architecture

The new architecture follows Python best practices with clear separation of concerns:

```
ATP/
├── __init__.py              # Package initialization
├── main.py                  # Clean entry point and orchestration
├── config.py                # Configuration management
├── data/                    # Data handling module
│   ├── __init__.py
│   ├── dataset.py           # Dataset classes (PRMDataset)
│   └── processing.py        # Data processing utilities
├── models/                  # Model definitions and utilities
│   ├── __init__.py
│   ├── prm_model.py         # PRM model building with LoRA
│   └── losses.py            # Loss functions and PRM utilities
├── training/                # Training logic and utilities
│   ├── __init__.py
│   ├── trainer.py           # Main bilevel training loop
│   └── utils.py             # Training utilities (checkpoints, metrics)
└── utils/                   # Common utilities
    ├── __init__.py
    └── common.py            # Logging, seeding, device management
```

## 🚀 Key Improvements

### 1. **Modular Design**
- **Separation of Concerns**: Each module has a clear responsibility
- **Reusable Components**: Easy to use individual components in other projects
- **Clean Interfaces**: Well-defined APIs between modules

### 2. **Better Error Handling**
- **Comprehensive Exception Handling**: Graceful handling of errors at all levels
- **Detailed Logging**: Structured logging with different levels
- **Validation**: Input validation and early error detection

### 3. **Enhanced Configuration**
- **Type-Safe Configuration**: Using dataclasses for configuration management
- **Argument Grouping**: Logical grouping of command-line arguments
- **Validation**: Configuration validation with helpful error messages

### 4. **Improved Data Handling**
- **Robust Dataset Class**: Better error handling and validation
- **Flexible Data Processing**: Modular data processing pipeline
- **Memory Efficient**: Optimized data loading and collation

### 5. **Modern Training Loop**
- **Clean Trainer Class**: Object-oriented training with clear methods
- **Comprehensive Metrics**: Advanced metrics tracking and logging
- **Checkpoint Management**: Robust checkpoint saving/loading with cleanup

### 6. **Documentation & Type Hints**
- **Comprehensive Docstrings**: Detailed documentation for all functions and classes
- **Type Hints**: Full type annotations for better IDE support and error detection
- **Examples**: Usage examples in docstrings

## 📖 Usage

### Basic Training
```bash
python main.py \
    --train_json_file data/train.jsonl \
    --meta_json_file data/meta.jsonl \
    --model_name Qwen/Qwen2.5-Math-PRM-7B
```

### Advanced Configuration
```bash
python main.py \
    --train_json_file data/train.jsonl \
    --meta_json_file data/meta.jsonl \
    --model_name Qwen/Qwen2.5-Math-PRM-7B \
    --lr 1e-4 \
    --meta_lr 5e-2 \
    --total_steps 5000 \
    --batch_size 2 \
    --lora_r 32 \
    --precision bf16 \
    --project_name "my-dreamprm-experiment"
```

### Resume Training
```bash
python main.py \
    --resume_from checkpoints/checkpoint_step_1000.pt \
    --resume_step 1000
```

### Dry Run (Validation)
```bash
python main.py --dry_run --debug
```

## 🔧 Configuration Options

### Data and Model
- `--train_json_file`: Training data path (JSONL format)
- `--meta_json_file`: Meta-learning data path (JSONL format)  
- `--model_name`: HuggingFace model name (default: Qwen/Qwen2.5-Math-PRM-7B)
- `--weights_path`: Checkpoint save directory

### Training Parameters
- `--total_steps`: Total training steps (default: 2000)
- `--batch_size`: Training batch size (default: 1)
- `--meta_batch_size`: Meta-learning batch size (default: 1)
- `--lr`: Inner loop learning rate (default: 2e-4)
- `--meta_lr`: Outer loop learning rate (default: 1e-1)

### Model Configuration
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--lora_dropout`: LoRA dropout (default: 0.05)
- `--max_length`: Maximum sequence length (default: 4096)

### System Settings
- `--device`: Device to use (cuda/cpu)
- `--precision`: Precision mode (bf16/fp16/fp32)
- `--seed`: Random seed for reproducibility
- `--num_workers`: DataLoader workers

### Experiment Tracking
- `--project_name`: Wandb project name
- `--experiment_name`: Custom experiment name
- `--disable_wandb`: Disable Weights & Biases logging

### Special Modes
- `--debug`: Enable debug mode with detailed error reporting
- `--dry_run`: Validate configuration without training
- `--baseline`: Run without bilevel optimization

## 📊 Features

### Advanced Metrics Tracking
- **Real-time Monitoring**: Live metrics during training
- **Best Metrics Tracking**: Automatic tracking of best values
- **Weights & Biases Integration**: Automatic experiment logging
- **Comprehensive Summaries**: Detailed training summaries

### Robust Checkpoint Management
- **Automatic Cleanup**: Keeps only recent checkpoints
- **Resume Training**: Easy resume from any checkpoint
- **Comprehensive State**: Saves all training state including optimizers

### Error Recovery
- **Graceful Degradation**: Continues training despite individual batch failures
- **Memory Management**: Automatic memory cleanup
- **Interrupt Handling**: Clean shutdown on Ctrl+C

## 🔬 Architecture Details

### Data Flow
1. **Configuration**: Parse and validate all settings
2. **Data Loading**: Load and preprocess JSONL data
3. **Model Setup**: Build PRM model with LoRA adapters
4. **Training**: Execute bilevel optimization loop
5. **Monitoring**: Track metrics and save checkpoints

### Bilevel Optimization
- **Inner Loop**: Train PRM parameters with instance weighting
- **Outer Loop**: Update instance weights based on meta-learning objectives
- **Dynamic Weighting**: Adaptive instance importance based on meta performance

### Process Reward Model (PRM)
- **Step-wise Evaluation**: Evaluate reasoning quality at each step
- **Qwen Integration**: Uses official Qwen PRM methodology
- **Flexible Architecture**: Supports different base models

## 🛠️ Development

### Code Quality
- **Type Safety**: Full type annotations
- **Documentation**: Comprehensive docstrings
- **Error Handling**: Robust error management
- **Logging**: Structured logging throughout

### Extensibility
- **Plugin Architecture**: Easy to add new components
- **Configuration System**: Flexible configuration management
- **Modular Design**: Independent, reusable modules

## 📝 Data Format

The training data should be in JSONL format with the following structure:

```json
{
    "cot_response": "### Step 1: Analyze the problem...\n### Step 2: Apply theorem...",
    "cot_steps": [
        {"pi": 0.8},
        {"pi": 0.9},
        {"pi": 0.7},
        {"pi": 0.85}
    ],
    "meta_label": 1.0,
    "informal_prefix": "Prove that...",
    "formal_statement": "∀ x ∈ ℝ, ..."
}
```

## 🎯 Benefits of Restructuring

1. **Maintainability**: Clear module boundaries make maintenance easier
2. **Testability**: Individual components can be unit tested
3. **Reusability**: Components can be used in other projects
4. **Readability**: Code is self-documenting with clear intentions
5. **Extensibility**: Easy to add new features without breaking existing code
6. **Debugging**: Better error messages and logging for troubleshooting
7. **Performance**: Optimized data loading and memory management

## 🔄 Migration from Old Code

The restructured code maintains full compatibility with the original functionality while providing significant improvements in code organization and maintainability. All command-line arguments remain the same, so existing scripts will continue to work.

## 🤝 Contributing

When contributing to this codebase:

1. **Follow the Architecture**: Respect module boundaries
2. **Add Documentation**: Include comprehensive docstrings
3. **Type Annotations**: Use type hints for all functions
4. **Error Handling**: Include appropriate error handling
5. **Logging**: Add structured logging for debugging
6. **Tests**: Consider adding unit tests for new functionality

This restructured codebase provides a solid foundation for future development and research in automated theorem proving with process reward models.