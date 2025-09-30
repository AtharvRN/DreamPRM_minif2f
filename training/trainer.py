"""Main training logic for DreamPRM bilevel optimization.

This module contains the core training loop for the bilevel optimization
process in DreamPRM, including inner and outer loop updates.
"""

import logging
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_cosine_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm

from ..config import TrainingConfig
from ..models.losses import PRMLossFunction, extract_step_probabilities, compute_aggregate_score
from .utils import CheckpointManager, MetricsTracker

logger = logging.getLogger(__name__)


class BilevelTrainer:
    """Bilevel trainer for Process Reward Model optimization.
    
    This class implements the bilevel optimization algorithm for training
    Process Reward Models with instance reweighting.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_loader: DataLoader,
        meta_loader: DataLoader,
        step_separator_id: int
    ):
        """Initialize the bilevel trainer.
        
        Args:
            config: Training configuration
            model: The PRM model to train
            train_loader: Training data loader
            meta_loader: Meta-learning data loader  
            step_separator_id: Token ID for step separators
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.meta_loader = meta_loader
        self.step_separator_id = step_separator_id
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Initialize loss function
        self.loss_fn = PRMLossFunction()
        
        # Initialize instance weights
        self.instance_weights = nn.Parameter(
            torch.ones(len(train_loader.dataset), device=self.device, requires_grad=True)
        )
        
        # Setup optimizers
        self._setup_optimizers()
        
        # Setup tracking and checkpointing
        self.checkpoint_manager = CheckpointManager(config.weights_path)
        self.metrics_tracker = MetricsTracker(use_wandb=not config.disable_wandb)
        
        # Cache tokenizer to avoid repeated loading
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        
        logger.info("BilevelTrainer initialized")
    
    def _setup_optimizers(self) -> None:
        """Setup optimizers and learning rate schedulers."""
        # Inner loop optimizer (for model parameters)
        self.optimizer_inner = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        # Outer loop optimizer (for instance weights)
        self.optimizer_outer = AdamW(
            [self.instance_weights],
            lr=self.config.meta_lr,
            weight_decay=self.config.meta_weight_decay
        )
        
        # Learning rate schedulers
        warmup_steps = int(self.config.warmup_ratio * self.config.total_steps)
        
        self.scheduler_inner = get_cosine_schedule_with_warmup(
            self.optimizer_inner,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.config.total_steps
        )
        
        self.scheduler_outer = get_cosine_schedule_with_warmup(
            self.optimizer_outer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.config.total_steps
        )
        
        logger.info(f"Optimizers configured with warmup_steps={warmup_steps}")
    
    def train(self) -> None:
        """Execute the main training loop."""
        logger.info("Starting bilevel training...")
        
        # Resume from checkpoint if specified
        start_step = self._maybe_resume_training()
        
        # Training loop
        self.model.train()
        progress_bar = tqdm(
            range(start_step, self.config.total_steps),
            desc="Training",
            initial=start_step,
            total=self.config.total_steps
        )
        
        try:
            for step in progress_bar:
                step_metrics = self._training_step(step)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'inner_loss': step_metrics.get('inner_loss', 0),
                    'meta_loss': step_metrics.get('meta_loss', 0)
                })
                
                # Logging
                if step % self.config.log_every_steps == 0:
                    self._log_step_metrics(step, step_metrics)
                
                # Save checkpoint
                if step % self.config.save_every_steps == 0 and step > 0:
                    self._save_checkpoint(step, step_metrics)
                
                # Memory cleanup
                if step % 100 == 0:
                    torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            if self.config.debug:
                raise
        finally:
            self._finalize_training()
    
    def _training_step(self, step: int) -> Dict[str, float]:
        """Execute one training step.
        
        Args:
            step: Current training step
            
        Returns:
            Dictionary containing step metrics
        """
        step_metrics = {}
        
        # Inner loop: Train PRM parameters
        inner_loss = self._inner_loop_update()
        if inner_loss is not None:
            step_metrics['inner_loss'] = inner_loss
            step_metrics['inner_lr'] = self.scheduler_inner.get_last_lr()[0]
        
        # Outer loop: Update instance weights
        if not self.config.baseline and step % self.config.eval_every_steps == 0:
            meta_metrics = self._outer_loop_update()
            step_metrics.update(meta_metrics)
        
        # Update metrics tracker
        self.metrics_tracker.update(step_metrics, step=step)
        
        return step_metrics
    
    def _inner_loop_update(self) -> Optional[float]:
        """Execute inner loop update (train model parameters).
        
        Returns:
            Average inner loss or None if no valid batches
        """
        inner_losses = []
        
        for inner_step in range(self.config.inner_steps):
            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    loss = self._process_training_batch(batch)
                    if loss is not None:
                        inner_losses.append(loss)
                        
                except Exception as e:
                    logger.warning(f"Error in inner loop batch {batch_idx}: {e}")
                    if self.config.debug:
                        raise
                    continue
        
        return sum(inner_losses) / len(inner_losses) if inner_losses else None
    
    def _process_training_batch(self, batch: Dict[str, Any]) -> Optional[float]:
        """Process a single training batch.
        
        Args:
            batch: Training batch data
            
        Returns:
            Batch loss or None if processing failed
        """
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract step probabilities
        step_probs = extract_step_probabilities(
            outputs.logits, input_ids, self.step_separator_id, self.tokenizer
        )
        
        if not step_probs or all(len(p) == 0 for p in step_probs):
            logger.warning("No valid step probabilities found")
            return None
        
        # Compute step-wise loss
        step_loss = self.loss_fn.compute_step_loss(
            step_probs, batch["rewards_list"], self.device
        )
        
        if step_loss.item() == 0:
            return None
        
        # Apply instance weights if not baseline
        if self.config.baseline:
            loss = step_loss
        else:
            weights = self.instance_weights[batch["idxs"].to(self.device)]
            loss = (weights * step_loss).mean()
        
        # Backward pass
        self.optimizer_inner.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clipping
        )
        
        # Update parameters
        self.optimizer_inner.step()
        self.scheduler_inner.step()
        
        return loss.item()
    
    def _outer_loop_update(self) -> Dict[str, float]:
        """Execute outer loop update (update instance weights).
        
        Returns:
            Dictionary with meta-learning metrics
        """
        self.model.eval()
        meta_losses = []
        aggregate_scores = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.meta_loader):
                try:
                    meta_metrics = self._process_meta_batch(batch)
                    if meta_metrics:
                        meta_losses.append(meta_metrics['loss'])
                        aggregate_scores.extend(meta_metrics['scores'])
                        
                except Exception as e:
                    logger.warning(f"Error in outer loop batch {batch_idx}: {e}")
                    if self.config.debug:
                        raise
                    continue
        
        # Update instance weights based on meta loss
        metrics = {}
        if meta_losses:
            meta_loss_mean = sum(meta_losses) / len(meta_losses)
            metrics['meta_loss'] = meta_loss_mean
            metrics['meta_lr'] = self.scheduler_outer.get_last_lr()[0]
            metrics['avg_aggregate_score'] = (
                sum(aggregate_scores) / len(aggregate_scores) if aggregate_scores else 0
            )
            metrics['instance_weight_mean'] = self.instance_weights.mean().item()
            metrics['instance_weight_std'] = self.instance_weights.std().item()
            
            # Simple meta-learning update (placeholder for full bilevel optimization)
            if not self.config.dry_run:
                self._update_instance_weights(meta_loss_mean)
        
        self.model.train()
        return metrics
    
    def _process_meta_batch(self, batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single meta-learning batch.
        
        Args:
            batch: Meta batch data
            
        Returns:
            Dictionary with loss and scores or None if processing failed
        """
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract step probabilities
        step_probs = extract_step_probabilities(
            outputs.logits, input_ids, self.step_separator_id, self.tokenizer
        )
        
        # Compute aggregate scores
        aggregate_scores_list = []
        for probs in step_probs:
            if len(probs) > 0:
                score = compute_aggregate_score(probs)
                aggregate_scores_list.append(score)
        
        if not aggregate_scores_list:
            return None
        
        # Compute meta loss
        aggregate_tensor = torch.stack(aggregate_scores_list)
        meta_labels = batch["meta_labels"].to(self.device)
        meta_loss = self.loss_fn.compute_meta_loss(aggregate_tensor, meta_labels)
        
        return {
            'loss': meta_loss.item(),
            'scores': [score.item() for score in aggregate_scores_list]
        }
    
    def _update_instance_weights(self, meta_loss: float) -> None:
        """Update instance weights based on meta loss.
        
        Args:
            meta_loss: Current meta loss value
        """
        with torch.no_grad():
            # Simple heuristic update (could be replaced with proper bilevel optimization)
            if meta_loss > 0.1:
                self.instance_weights.data *= 0.99
            else:
                self.instance_weights.data *= 1.01
            
            # Clamp weights to reasonable range
            self.instance_weights.data.clamp_(0.1, 10.0)
    
    def _maybe_resume_training(self) -> int:
        """Resume training from checkpoint if specified.
        
        Returns:
            Starting step number
        """
        if self.config.resume_from:
            try:
                step, _, _ = self.checkpoint_manager.load_checkpoint(
                    self.config.resume_from,
                    self.model,
                    self.optimizer_inner,
                    self.optimizer_outer,
                    self.instance_weights,
                    str(self.device)
                )
                return max(step, self.config.resume_step)
            except Exception as e:
                logger.error(f"Failed to resume from checkpoint: {e}")
                if self.config.debug:
                    raise
        
        return 0
    
    def _log_step_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Log metrics for a training step.
        
        Args:
            step: Current step number
            metrics: Metrics dictionary
        """
        log_msg = f"Step {step}:"
        for key, value in metrics.items():
            log_msg += f" {key}={value:.6f}"
        logger.info(log_msg)
    
    def _save_checkpoint(self, step: int, metrics: Dict[str, float]) -> None:
        """Save a training checkpoint.
        
        Args:
            step: Current step number
            metrics: Current metrics
        """
        try:
            self.checkpoint_manager.save_checkpoint(
                self.model,
                self.optimizer_inner,
                self.optimizer_outer,
                self.instance_weights,
                step,
                metrics.get('inner_loss', 0.0),
                metrics
            )
        except Exception as e:
            logger.error(f"Failed to save checkpoint at step {step}: {e}")
    
    def _finalize_training(self) -> None:
        """Finalize training with cleanup and summary."""
        logger.info("Training completed!")
        
        # Log final summary
        self.metrics_tracker.log_summary()
        
        # Save final model if not in dry run mode
        if not self.config.dry_run:
            try:
                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer_inner,
                    self.optimizer_outer,
                    self.instance_weights,
                    self.config.total_steps,
                    self.metrics_tracker.get_latest('inner_loss') or 0.0,
                    self.metrics_tracker.best_metrics
                )
                logger.info("Final model saved")
            except Exception as e:
                logger.error(f"Failed to save final model: {e}")