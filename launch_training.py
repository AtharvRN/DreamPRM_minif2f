#!/usr/bin/env python3
"""
Multi-GPU Training Launcher for DreamPRM

This script provides easy launching of multi-GPU training using torchrun
or python -m torch.distributed.launch.

Examples:
    # Single GPU training
    python launch_training.py --config configs/single_gpu.yaml
    
    # Multi-GPU training (4 GPUs)
    python launch_training.py --config configs/multi_gpu.yaml --num_gpus 4
    
    # Multi-GPU with specific arguments
    python launch_training.py --num_gpus 2 --batch_size 2 --lr 1e-4 --use_lora
    
    # Multi-GPU without LoRA (full fine-tuning)
    python launch_training.py --num_gpus 4 --no_lora --batch_size 1 --gradient_accumulation_steps 8
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def find_free_port():
    """Find a free port for distributed training."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def launch_single_gpu(args, extra_args):
    """Launch single GPU training."""
    cmd = [
        sys.executable, "main.py",
        "--device", "cuda",
        *extra_args
    ]
    
    print(f"Launching single GPU training...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return e.returncode


def launch_multi_gpu(args, extra_args):
    """Launch multi-GPU training using torchrun."""
    # Find a free port for distributed training
    master_port = find_free_port()
    
    # Build torchrun command
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(args.num_gpus),
        "--master_port", str(master_port),
    ]
    
    # Add additional torchrun arguments if specified
    if args.nnodes > 1:
        cmd.extend(["--nnodes", str(args.nnodes)])
    if args.node_rank > 0:
        cmd.extend(["--node_rank", str(args.node_rank)])
    if args.master_addr != "localhost":
        cmd.extend(["--master_addr", args.master_addr])
    
    # Add the main script and arguments
    cmd.extend([
        "main.py",
        "--use_ddp",
        "--world_size", str(args.num_gpus * args.nnodes),
        *extra_args
    ])
    
    print(f"Launching {args.num_gpus} GPU training...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return e.returncode


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU Training Launcher for DreamPRM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Launcher-specific arguments
    launcher_group = parser.add_argument_group("Launcher Configuration")
    launcher_group.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs to use for training"
    )
    launcher_group.add_argument(
        "--nnodes", type=int, default=1,
        help="Number of nodes (machines) for distributed training"
    )
    launcher_group.add_argument(
        "--node_rank", type=int, default=0,
        help="Rank of the current node"
    )
    launcher_group.add_argument(
        "--master_addr", type=str, default="localhost",
        help="Master node address for multi-node training"
    )
    
    # Common training arguments (subset)
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument(
        "--config", type=str, default=None,
        help="Configuration file path (YAML or JSON)"
    )
    training_group.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-Math-PRM-7B",
        help="Model name or path"
    )
    training_group.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size per GPU"
    )
    training_group.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Gradient accumulation steps"
    )
    training_group.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate"
    )
    training_group.add_argument(
        "--total_steps", type=int, default=2000,
        help="Total training steps"
    )
    training_group.add_argument(
        "--precision", type=str, default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Training precision"
    )
    
    # LoRA configuration
    lora_group = parser.add_argument_group("LoRA Configuration")
    lora_group.add_argument(
        "--use_lora", action="store_true", default=True,
        help="Use LoRA for parameter-efficient fine-tuning"
    )
    lora_group.add_argument(
        "--no_lora", dest="use_lora", action="store_false",
        help="Disable LoRA (full fine-tuning)"
    )
    lora_group.add_argument(
        "--lora_r", type=int, default=16,
        help="LoRA rank"
    )
    lora_group.add_argument(
        "--lora_alpha", type=int, default=32,
        help="LoRA alpha"
    )
    lora_group.add_argument(
        "--lora_dropout", type=float, default=0.05,
        help="LoRA dropout rate"
    )
    
    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--train_json_file", type=str, default="train_split.json",
        help="Training data file"
    )
    data_group.add_argument(
        "--meta_json_file", type=str, default="meta_training.json",
        help="Meta training data file"
    )
    data_group.add_argument(
        "--max_length", type=int, default=4096,
        help="Maximum sequence length"
    )
    
    # Parse known args to allow passing additional arguments
    args, unknown_args = parser.parse_known_args()
    
    # Validate arguments
    if args.num_gpus < 1:
        print("Error: num_gpus must be at least 1")
        return 1
    
    if args.num_gpus > 1 and not torch_available():
        print("Error: PyTorch with CUDA support required for multi-GPU training")
        return 1
    
    # Build extra arguments to pass to main script
    extra_args = []
    
    # Add training arguments
    extra_args.extend(["--model_name", args.model_name])
    extra_args.extend(["--batch_size", str(args.batch_size)])
    extra_args.extend(["--gradient_accumulation_steps", str(args.gradient_accumulation_steps)])
    extra_args.extend(["--lr", str(args.lr)])
    extra_args.extend(["--total_steps", str(args.total_steps)])
    extra_args.extend(["--precision", args.precision])
    
    # Add LoRA arguments
    if args.use_lora:
        extra_args.append("--use_lora")
        extra_args.extend(["--lora_r", str(args.lora_r)])
        extra_args.extend(["--lora_alpha", str(args.lora_alpha)])
        extra_args.extend(["--lora_dropout", str(args.lora_dropout)])
    else:
        extra_args.append("--no_lora")
    
    # Add data arguments
    extra_args.extend(["--train_json_file", args.train_json_file])
    extra_args.extend(["--meta_json_file", args.meta_json_file])
    extra_args.extend(["--max_length", str(args.max_length)])
    
    # Add unknown arguments
    extra_args.extend(unknown_args)
    
    # Launch training
    if args.num_gpus == 1:
        return launch_single_gpu(args, extra_args)
    else:
        return launch_multi_gpu(args, extra_args)


def torch_available():
    """Check if PyTorch with CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def print_usage_examples():
    """Print usage examples."""
    examples = [
        "# Single GPU training with LoRA",
        "python launch_training.py --use_lora --batch_size 2",
        "",
        "# Multi-GPU training (4 GPUs) with LoRA",
        "python launch_training.py --num_gpus 4 --use_lora --batch_size 1",
        "",
        "# Multi-GPU full fine-tuning (no LoRA)",
        "python launch_training.py --num_gpus 2 --no_lora --batch_size 1 --gradient_accumulation_steps 8",
        "",
        "# Large-scale training with gradient accumulation",
        "python launch_training.py --num_gpus 8 --batch_size 1 --gradient_accumulation_steps 16 --precision bf16",
        "",
        "# Custom learning rate and model",
        "python launch_training.py --num_gpus 2 --lr 1e-4 --model_name microsoft/DialoGPT-medium",
    ]
    
    print("Usage Examples:")
    print("=" * 50)
    for example in examples:
        print(example)


if __name__ == "__main__":
    if len(sys.argv) == 1 or "--help" in sys.argv or "-h" in sys.argv:
        print_usage_examples()
        print()
    
    exit_code = main()
    sys.exit(exit_code)