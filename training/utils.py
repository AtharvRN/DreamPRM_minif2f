"""Training utilities for DreamPRM.

This module contains checkpoint management, metrics tracking,
and other training-related helper classes.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
import wandb

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints and saving/loading.
    
    This class handles saving and loading model checkpoints during training,
    including automatic cleanup of old checkpoints.
    """
    
    def __init__(
        self,
        save_dir: str,
        max_checkpoints: int = 5,
        prefix: str = "checkpoint"
    ):
        """Initialize the checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            prefix: Prefix for checkpoint filenames
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.prefix = prefix
        
        logger.info(f"CheckpointManager initialized: {save_dir} (max: {max_checkpoints})")
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        meta_optimizer: Optional[torch.optim.Optimizer],
        instance_weights: torch.Tensor,
        step: int,
        loss: float,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save a training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Main optimizer
            meta_optimizer: Meta optimizer (can be None)
            instance_weights: Instance weights tensor
            step: Current training step
            loss: Current loss value
            metrics: Additional metrics to save
            
        Returns:
            Path to the saved checkpoint
        """
        checkpoint_path = self.save_dir / f"{self.prefix}_step_{step}.pt"
        
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'meta_optimizer_state_dict': (
                meta_optimizer.state_dict() if meta_optimizer else None
            ),
            'instance_weights': instance_weights.data.clone(),
            'loss': loss,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__
        }
        
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint at step {step}: {checkpoint_path}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        meta_optimizer: Optional[torch.optim.Optimizer] = None,
        instance_weights: Optional[torch.Tensor] = None,
        device: str = "cpu"
    ) -> tuple[int, float, Dict[str, Any]]:
        """Load a training checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            model: Model to load state into
            optimizer: Main optimizer (optional)
            meta_optimizer: Meta optimizer (optional)
            instance_weights: Instance weights tensor (optional)
            device: Device to map tensors to
            
        Returns:
            Tuple of (step, loss, metrics)
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer states if provided
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if meta_optimizer and 'meta_optimizer_state_dict' in checkpoint:
                if checkpoint['meta_optimizer_state_dict'] is not None:
                    meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
            
            # Load instance weights if provided
            if instance_weights is not None and 'instance_weights' in checkpoint:
                instance_weights.data.copy_(checkpoint['instance_weights'])
            
            step = checkpoint['step']
            loss = checkpoint.get('loss', 0.0)
            metrics = checkpoint.get('metrics', {})
            
            logger.info(f"Loaded checkpoint from step {step}: {checkpoint_path}")
            
            return step, loss, metrics
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints keeping only the latest ones."""
        checkpoints = list(self.save_dir.glob(f"{self.prefix}_step_*.pt"))
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by step number
        def extract_step(path: Path) -> int:
            try:
                return int(path.stem.split('_')[-1])
            except (ValueError, IndexError):
                return 0
        
        checkpoints.sort(key=extract_step)
        
        # Remove oldest checkpoints
        for old_checkpoint in checkpoints[:-self.max_checkpoints]:
            try:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")
    
    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints.
        
        Returns:
            List of checkpoint paths sorted by step number
        """
        checkpoints = list(self.save_dir.glob(f"{self.prefix}_step_*.pt"))
        
        def extract_step(path: Path) -> int:
            try:
                return int(path.stem.split('_')[-1])
            except (ValueError, IndexError):
                return 0
        
        return sorted(checkpoints, key=extract_step)


class MetricsTracker:
    """Track and log training metrics.
    
    This class handles metric collection, averaging, and logging
    to both console and Weights & Biases.
    """
    
    def __init__(self, use_wandb: bool = True):
        """Initialize the metrics tracker.
        
        Args:
            use_wandb: Whether to log metrics to Weights & Biases
        """
        self.use_wandb = use_wandb
        self.metrics: Dict[str, List[float]] = {}
        self.best_metrics: Dict[str, float] = {}
        
        logger.info(f"MetricsTracker initialized (wandb: {use_wandb})")
    
    def update(
        self,
        metrics_dict: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Update metrics with new values.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            step: Optional step number for wandb logging
        """
        for key, value in metrics_dict.items():
            # Store in history
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(float(value))
            
            # Track best metrics
            self._update_best_metric(key, float(value))
        
        # Log to wandb if enabled
        if self.use_wandb and step is not None:
            try:
                wandb.log(metrics_dict, step=step)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")
    
    def _update_best_metric(self, key: str, value: float) -> None:
        """Update best metric tracking.
        
        Args:
            key: Metric name
            value: Metric value
        """
        # For loss metrics, track minimum
        if 'loss' in key.lower():
            if key not in self.best_metrics or value < self.best_metrics[key]:
                self.best_metrics[key] = value
        # For accuracy/performance metrics, track maximum
        elif any(term in key.lower() for term in ['accuracy', 'precision', 'recall', 'f1']):
            if key not in self.best_metrics or value > self.best_metrics[key]:
                self.best_metrics[key] = value
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get the latest value for a metric.
        
        Args:
            key: Metric name
            
        Returns:
            Latest metric value or None if not found
        """
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return None
    
    def get_average(self, key: str, last_n: int = 10) -> Optional[float]:
        """Get the average of the last N values for a metric.
        
        Args:
            key: Metric name
            last_n: Number of recent values to average
            
        Returns:
            Average of recent values or None if not found
        """
        if key not in self.metrics or not self.metrics[key]:
            return None
        
        recent_values = self.metrics[key][-last_n:]
        return float(np.mean(recent_values))
    
    def get_best(self, key: str) -> Optional[float]:
        """Get the best value for a metric.
        
        Args:
            key: Metric name
            
        Returns:
            Best metric value or None if not found
        """
        return self.best_metrics.get(key)
    
    def log_summary(self) -> None:
        """Log a summary of all tracked metrics."""
        logger.info("=== Training Metrics Summary ===")
        
        for key, value in self.best_metrics.items():
            logger.info(f"Best {key}: {value:.6f}")
        
        # Log current values
        logger.info("=== Current Values ===")
        for key in self.metrics:
            latest = self.get_latest(key)
            if latest is not None:
                logger.info(f"Current {key}: {latest:.6f}")
        
        # Log to wandb
        if self.use_wandb:
            try:
                wandb.log({f"best_{k}": v for k, v in self.best_metrics.items()})
            except Exception as e:
                logger.warning(f"Failed to log summary to wandb: {e}")
    
    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.metrics.clear()
        self.best_metrics.clear()
        logger.info("Reset all metrics")
