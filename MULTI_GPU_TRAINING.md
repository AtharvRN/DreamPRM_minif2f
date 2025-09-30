# Multi-GPU Training Guide for DreamPRM

This guide explains how to set up and run DreamPRM training on single or multiple GPUs with flexible LoRA configuration.

## 🚀 Quick Start

### Single GPU Training
```bash
# With LoRA (recommended for limited VRAM)
python launch_training.py --use_lora --batch_size 2

# Without LoRA (full fine-tuning)
python launch_training.py --no_lora --batch_size 1 --gradient_accumulation_steps 4
```

### Multi-GPU Training
```bash
# 4 GPUs with LoRA
python launch_training.py --num_gpus 4 --use_lora --batch_size 1

# 8 GPUs without LoRA (full fine-tuning)
python launch_training.py --num_gpus 8 --no_lora --batch_size 1 --gradient_accumulation_steps 8
```

## 📋 Configuration Options

### Training Modes

| Mode | Memory Usage | Training Speed | Performance |
|------|-------------|----------------|-------------|
| **LoRA** | Low (~8-12GB per GPU) | Fast | Good |
| **Full Fine-tuning** | High (~40-80GB per GPU) | Slower | Best |

### GPU Memory Requirements

| Configuration | Memory per GPU | Recommended Setup |
|---------------|----------------|-------------------|
| LoRA + Batch Size 1 | ~8GB | Single RTX 3080/4070 |
| LoRA + Batch Size 2 | ~12GB | Single RTX 4080/4090 |
| Full + Batch Size 1 | ~40GB | Single A100/H100 |
| Full + Accumulation 8 | ~16GB + accumulation | 4x RTX 4090 |

## 🛠️ Command Line Options

### Launcher Options
- `--num_gpus`: Number of GPUs (default: 1)
- `--nnodes`: Number of nodes for multi-node training (default: 1)
- `--node_rank`: Rank of current node (default: 0)
- `--master_addr`: Master node address (default: localhost)

### LoRA Configuration
- `--use_lora`: Enable LoRA (default: True)
- `--no_lora`: Disable LoRA (full fine-tuning)
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha scaling (default: 32)
- `--lora_dropout`: LoRA dropout rate (default: 0.05)

### Training Parameters
- `--batch_size`: Batch size per GPU (default: 1)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 1)
- `--lr`: Learning rate (default: 2e-4)
- `--total_steps`: Total training steps (default: 2000)
- `--precision`: Training precision (bf16/fp16/fp32, default: bf16)

## 💡 Recommended Configurations

### For Different Hardware Setups

#### Single RTX 3080/4070 (10-12GB VRAM)
```bash
python launch_training.py \
    --use_lora \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --precision bf16
```

#### Single RTX 4090 (24GB VRAM)
```bash
# LoRA mode
python launch_training.py \
    --use_lora \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --precision bf16

# Or small full fine-tuning
python launch_training.py \
    --no_lora \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --precision bf16
```

#### Dual RTX 4090 Setup
```bash
python launch_training.py \
    --num_gpus 2 \
    --use_lora \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --precision bf16
```

#### High-End Server (4x A100/H100)
```bash
# Full fine-tuning
python launch_training.py \
    --num_gpus 4 \
    --no_lora \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --precision bf16

# Or high-throughput LoRA
python launch_training.py \
    --num_gpus 4 \
    --use_lora \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --precision bf16
```

## 🔧 Advanced Usage

### Custom Model and Data
```bash
python launch_training.py \
    --num_gpus 2 \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --train_json_file "custom_train.jsonl" \
    --meta_json_file "custom_meta.jsonl" \
    --max_length 2048 \
    --use_lora
```

### Fine-tuning Existing Checkpoint
```bash
python launch_training.py \
    --num_gpus 4 \
    --resume_from "checkpoints/checkpoint_step_1000.pt" \
    --lr 1e-5 \
    --use_lora
```

### Multi-Node Training (2 nodes, 4 GPUs each)
```bash
# Node 0 (master)
python launch_training.py \
    --num_gpus 4 \
    --nnodes 2 \
    --node_rank 0 \
    --master_addr "192.168.1.100"

# Node 1
python launch_training.py \
    --num_gpus 4 \
    --nnodes 2 \
    --node_rank 1 \
    --master_addr "192.168.1.100"
```

## 📊 Performance Optimization

### LoRA vs Full Fine-tuning

**LoRA Advantages:**
- Much lower memory usage (8-12GB vs 40-80GB per GPU)
- Faster training iteration
- Good performance for most tasks
- Easy to merge and deploy

**Full Fine-tuning Advantages:**
- Maximum model performance
- No architectural constraints
- Better for domain adaptation
- More thorough parameter updates

### Batch Size and Gradient Accumulation

Calculate effective batch size: `batch_size × gradient_accumulation_steps × num_gpus`

Examples:
- Single GPU: `batch_size=4, accumulation=4` → effective batch size = 16
- 4 GPUs: `batch_size=2, accumulation=2` → effective batch size = 16
- 8 GPUs: `batch_size=1, accumulation=2` → effective batch size = 16

### Memory Optimization Tips

1. **Use gradient checkpointing** (automatically enabled)
2. **Choose appropriate precision:**
   - `bf16`: Best performance on modern GPUs (A100, RTX 40XX)
   - `fp16`: Good compatibility with older GPUs
   - `fp32`: Maximum precision, highest memory usage

3. **Adjust sequence length:**
   - `max_length=2048`: Halves memory usage vs 4096
   - `max_length=1024`: Quarters memory usage vs 4096

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
--batch_size 1

# Increase gradient accumulation
--gradient_accumulation_steps 8

# Use LoRA instead of full fine-tuning
--use_lora

# Reduce sequence length
--max_length 2048
```

#### Distributed Training Issues
```bash
# Check GPU visibility
nvidia-smi

# Verify PyTorch distributed
python -c "import torch; print(torch.cuda.device_count())"

# Use fallback launcher
python -m torch.distributed.launch --nproc_per_node=4 main.py --use_ddp
```

#### Slow Training
```bash
# Increase batch size if memory allows
--batch_size 2

# Use mixed precision
--precision bf16

# Optimize data loading
--num_workers 4
```

### Monitoring and Logging

#### Check Training Progress
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# View training logs
tail -f logs/training.log

# Monitor with Weights & Biases
# (automatically enabled if not disabled)
```

## 📈 Scaling Guidelines

### Effective Batch Size Scaling
- Start with single GPU baseline: batch_size=2, accumulation=4 (effective=8)
- Scale to 4 GPUs: batch_size=1, accumulation=2 (effective=8)
- Scale to 8 GPUs: batch_size=1, accumulation=1 (effective=8)

### Learning Rate Scaling
- **Linear scaling:** `lr = base_lr × sqrt(num_gpus)`
- **Conservative:** Keep base learning rate constant
- **Aggressive:** `lr = base_lr × num_gpus`

Example:
```bash
# Single GPU baseline
python launch_training.py --lr 2e-4

# 4 GPU scaling (conservative)
python launch_training.py --num_gpus 4 --lr 2e-4

# 4 GPU scaling (linear)
python launch_training.py --num_gpus 4 --lr 4e-4
```

## 🎯 Production Deployment

### Saving and Loading Models

The training automatically saves checkpoints. After training:

```python
# Load LoRA model
from peft import PeftModel
from transformers import AutoModel

base_model = AutoModel.from_pretrained("Qwen/Qwen2.5-Math-PRM-7B")
model = PeftModel.from_pretrained(base_model, "checkpoints/lora_adapters")

# Load full fine-tuned model
model = AutoModel.from_pretrained("checkpoints/final_model")
```

### Merging LoRA Adapters
```python
# Merge and save as single model
merged_model = model.merge_and_unload()
merged_model.save_pretrained("final_merged_model")
```

This comprehensive setup allows you to scale DreamPRM training from single GPU development to large-scale multi-GPU production training with flexible LoRA configuration! 🚀