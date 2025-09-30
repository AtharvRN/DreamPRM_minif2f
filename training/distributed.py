"""Distributed training utilities for multi-GPU support.

This module provides utilities for setting up and managing distributed
training across multiple GPUs using PyTorch's DistributedDataParallel.
"""

import os
import logging
from typing import Optional, Any
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


def setup_distributed_training(
    local_rank: int = -1,
    world_size: int = 1,
    backend: str = "nccl"
) -> tuple[int, int, bool]:
    """Setup distributed training environment.
    
    Args:
        local_rank: Local rank of the current process
        world_size: Total number of processes
        backend: Distributed backend to use
        
    Returns:
        Tuple of (rank, world_size, is_distributed)
    """
    # Check if distributed training should be used
    if local_rank == -1 or world_size <= 1:
        return 0, 1, False
    
    # Set up environment variables for distributed training
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(local_rank)
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = str(world_size)
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(local_rank)
    
    # Initialize distributed training
    try:
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                rank=local_rank,
                world_size=world_size
            )
        
        # Set device for current process
        torch.cuda.set_device(local_rank)
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        logger.info(
            f"Distributed training initialized: rank={rank}/{world_size}, "
            f"local_rank={local_rank}, backend={backend}"
        )
        
        return rank, world_size, True
        
    except Exception as e:
        logger.error(f"Failed to initialize distributed training: {e}")
        return 0, 1, False


def cleanup_distributed_training() -> None:
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training cleaned up")


def wrap_model_for_distributed(
    model: torch.nn.Module,
    device_ids: Optional[list] = None,
    find_unused_parameters: bool = False
) -> torch.nn.Module:
    """Wrap model with DistributedDataParallel if in distributed mode.
    
    Args:
        model: Model to wrap
        device_ids: List of device IDs for the model
        find_unused_parameters: Whether to find unused parameters
        
    Returns:
        Wrapped model or original model if not distributed
    """
    if not dist.is_initialized():
        return model
    
    try:
        # Get current device
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device_ids is None:
            device_ids = [local_rank]
        
        # Wrap with DDP
        ddp_model = DDP(
            model,
            device_ids=device_ids,
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters
        )
        
        logger.info(f"Model wrapped with DDP on device {local_rank}")
        return ddp_model
        
    except Exception as e:
        logger.error(f"Failed to wrap model with DDP: {e}")
        return model


def create_distributed_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    collate_fn: Optional[Any] = None,
    drop_last: bool = False
) -> DataLoader:
    """Create a DataLoader with distributed sampling if in distributed mode.
    
    Args:
        dataset: Dataset to create loader for
        batch_size: Batch size per process
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        collate_fn: Collate function for batching
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        DataLoader with appropriate sampler
    """
    # Create distributed sampler if in distributed mode
    sampler = None
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last
        )
        shuffle = False  # Sampler handles shuffling
        logger.info("Using DistributedSampler for multi-GPU training")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last
    )
    
    return dataloader


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0).
    
    Returns:
        True if main process, False otherwise
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get the world size (total number of processes).
    
    Returns:
        World size
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get the rank of current process.
    
    Returns:
        Process rank
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce_tensor(tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
    """All-reduce a tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation ("mean", "sum")
        
    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor
    
    # Clone tensor to avoid in-place modification
    reduced_tensor = tensor.clone()
    
    # All-reduce
    dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
    
    # Average if requested
    if op == "mean":
        reduced_tensor /= get_world_size()
    
    return reduced_tensor


def reduce_metrics(metrics: dict) -> dict:
    """Reduce metrics across all processes.
    
    Args:
        metrics: Dictionary of metrics to reduce
        
    Returns:
        Reduced metrics dictionary
    """
    if not dist.is_initialized():
        return metrics
    
    reduced_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            tensor = torch.tensor(value, dtype=torch.float32)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            reduced_tensor = all_reduce_tensor(tensor, op="mean")
            reduced_metrics[key] = reduced_tensor.item()
        else:
            reduced_metrics[key] = value
    
    return reduced_metrics


def save_checkpoint_distributed(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    best_metric: float = 0.0,
    extra_state: Optional[dict] = None
) -> None:
    """Save checkpoint only from main process in distributed training.
    
    Args:
        checkpoint_path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler
        epoch: Current epoch
        step: Current step
        best_metric: Best metric achieved
        extra_state: Additional state to save
    """
    if not is_main_process():
        return
    
    # Extract model state dict (handle DDP wrapper)
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    # Prepare checkpoint
    checkpoint = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if extra_state is not None:
        checkpoint.update(extra_state)
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint_distributed(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    device: str = "cuda"
) -> tuple[int, int, float]:
    """Load checkpoint in distributed training.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler
        device: Device to map tensors to
        
    Returns:
        Tuple of (epoch, step, best_metric)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state (handle DDP wrapper)
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    best_metric = checkpoint.get("best_metric", 0.0)
    
    logger.info(f"Checkpoint loaded: epoch={epoch}, step={step}, best_metric={best_metric}")
    
    return epoch, step, best_metric