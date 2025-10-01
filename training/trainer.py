"""Main training logic for DreamPRM bilevel optimization.

This module contains the core training loop for the bilevel optimization
process in DreamPRM, including inner and outer loop updates with multi-GPU support.
"""

import logging
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm

from config import TrainingConfig
from models.losses import PRMLossFunction, extract_step_probabilities, compute_aggregate_score
from training.utils import CheckpointManager, MetricsTracker
from training.distributed import (
    wrap_model_for_distributed,
    create_distributed_dataloader,
    is_main_process,
    barrier,
    reduce_metrics,
    save_checkpoint_distributed,
    load_checkpoint_distributed
)

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
        self.step_separator_id = step_separator_id
        
        # Check if model is already distributed with device_map
        self.model_is_distributed = hasattr(model, 'hf_device_map') and model.hf_device_map is not None
        
        # Setup device
        if config.use_ddp:
            self.device = torch.device(f"cuda:{config.local_rank}")
        else:
            self.device = torch.device(config.device)
            # Ensure we're using the specified device
            if "cuda" in config.device and torch.cuda.is_available():
                torch.cuda.set_device(self.device)
        
        # Handle model device placement
        if self.model_is_distributed:
            # Model is already distributed with device_map, don't move it
            logger.info(f"Model is pre-distributed with device_map: {model.hf_device_map}")
            self.model = model
            
            # CRITICAL: Ensure training mode and gradients after device mapping
            self.model.train()
            for name, param in self.model.named_parameters():
                param.requires_grad = True
            logger.info("Forced training mode and gradients for distributed model")
        else:
            # Move model to device and wrap for distributed training if needed
            model = model.to(self.device)
            
            # For multi-GPU without device_map, use DataParallel
            if torch.cuda.device_count() > 1 and not config.use_ddp:
                logger.info(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
                model = torch.nn.DataParallel(model)
                
            if config.use_ddp:
                self.model = wrap_model_for_distributed(
                    model, 
                    find_unused_parameters=config.ddp_find_unused_parameters
                )
            else:
                self.model = model
        
        # Store original data loaders or create distributed ones
        if config.use_ddp:
            # Recreate with distributed samplers - this should be done in create_data_loaders
            # but we handle it here in case it wasn't
            self.train_loader = train_loader
            self.meta_loader = meta_loader
        else:
            self.train_loader = train_loader
            self.meta_loader = meta_loader
        
        # Initialize loss function
        self.loss_fn = PRMLossFunction()
        
        # Initialize instance weights
        if self.model_is_distributed:
            # For distributed models, put instance weights on the first available GPU
            first_device = f"cuda:{min(model.hf_device_map.values())}"
            self.instance_weights = nn.Parameter(
                torch.ones(len(train_loader.dataset), device=first_device, requires_grad=True)
            )
        else:
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
        
        # Gradient accumulation state
        self.accumulated_loss = 0.0
        self.accumulation_steps = 0
        
        if is_main_process():
            logger.info("BilevelTrainer initialized")
            
        # Log detailed parameter information
        self._log_parameter_details()
    
    def _log_parameter_details(self):
        """Log detailed information about model parameters and memory usage."""
        if not is_main_process():
            return
            
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
            else:
                frozen_params += param_count
        
        logger.info(f"=== Parameter Analysis ===")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {frozen_params:,}")
        logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        # Log instance weights info
        logger.info(f"Instance weights: {self.instance_weights.numel():,} parameters")
        logger.info(f"Instance weights device: {self.instance_weights.device}")
        logger.info(f"Instance weights requires_grad: {self.instance_weights.requires_grad}")
        
        # Log memory info if CUDA is available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"GPU {i} - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
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
        if is_main_process():
            logger.info("Starting bilevel training...")
        
        # Resume from checkpoint if specified
        start_step = self._maybe_resume_training()
        
        # Training loop
        self.model.train()
        
        # Setup progress bar only on main process
        if is_main_process():
            progress_bar = tqdm(
                range(start_step, self.config.total_steps),
                desc="Training",
                initial=start_step,
                total=self.config.total_steps
            )
        else:
            progress_bar = range(start_step, self.config.total_steps)
        
        try:
            for step in progress_bar:
                # Set epoch for distributed sampler
                if self.config.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(step)
                
                step_metrics = self._training_step(step)
                
                # Reduce metrics across processes
                if self.config.use_ddp:
                    step_metrics = reduce_metrics(step_metrics)
                
                # Update progress bar (main process only)
                if is_main_process() and hasattr(progress_bar, 'set_postfix'):
                    progress_bar.set_postfix({
                        'inner_loss': step_metrics.get('inner_loss', 0),
                        'meta_loss': step_metrics.get('meta_loss', 0)
                    })
                
                # Logging (main process only)
                if is_main_process() and step % self.config.log_every_steps == 0:
                    self._log_step_metrics(step, step_metrics)
                
                # Save checkpoint (main process only)
                if step % self.config.save_every_steps == 0 and step > 0:
                    if self.config.use_ddp:
                        barrier()  # Synchronize before saving
                    if is_main_process():
                        self._save_checkpoint(step, step_metrics)
                
                # Memory cleanup
                if step % 100 == 0:
                    torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            if is_main_process():
                logger.info("Training interrupted by user")
        except Exception as e:
            if is_main_process():
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
        self.model.train()  # Ensure training mode
        
        # Force all parameters to require gradients
        for param in self.model.parameters():
            param.requires_grad = True
        
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
        # Handle device placement based on model distribution
        if self.model_is_distributed:
            # For distributed models, we need to move inputs to the device of the embedding layer
            # The embedding layer is typically on the first device in the device map
            embed_device = None
            for name, device_id in self.model.hf_device_map.items():
                if 'embed_tokens' in name:
                    embed_device = f"cuda:{device_id}"
                    break
            
            if embed_device is None:
                # Fallback to first device in map
                embed_device = f"cuda:{min(self.model.hf_device_map.values())}"
            
            input_ids = batch["input_ids"].to(embed_device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(embed_device, non_blocking=True)
        else:
            # For single-device models, move tensors to the model device
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        
        rewards_list = batch["rewards_list"]
        idxs = batch["idxs"]
        
        # Ensure model is in training mode and gradients are enabled
        self.model.train()
        if self.model_is_distributed:
            for param in self.model.parameters():
                param.requires_grad = True
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Re-enable gradients after forward pass for distributed models
        if self.model_is_distributed:
            for param in self.model.parameters():
                param.requires_grad = True
        
        # Extract step probabilities
        step_probs = extract_step_probabilities(
            outputs.logits, input_ids, self.step_separator_id, self.tokenizer
        )
        
        if not step_probs or all(len(p) == 0 for p in step_probs):
            logger.warning("No valid step probabilities found")
            return None
        
        # Compute step-wise loss
        step_loss = self.loss_fn.compute_step_loss(
            step_probs, rewards_list, self.device if not self.model_is_distributed else None
        )
        
        if step_loss.item() == 0:
            return None
        
        # Apply instance weights if not baseline
        if self.config.baseline:
            loss = step_loss
        else:
            if self.model_is_distributed:
                # Move idxs to the same device as instance weights
                weights_device = self.instance_weights.device
                weights = self.instance_weights[idxs.to(weights_device, non_blocking=True)]
                # Move weights to the same device as step_loss
                weights = weights.to(step_loss.device)
            else:
                weights = self.instance_weights[idxs.to(self.device, non_blocking=True)]
            loss = (weights * step_loss).mean()
        
        # Backward pass
        self.optimizer_inner.zero_grad()
        
        # For device_map models, ensure all model parameters have gradients enabled
        if self.model_is_distributed:
            for param in self.model.parameters():
                if param.requires_grad:
                    param.retain_grad()
        
        loss.backward()
        
        # Log instance weights periodically
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 1
            
        if self._step_count % 50 == 1 and is_main_process():  # Log every 20 batches
            logger.info(f"Instance weights (step {self._step_count}): {self.instance_weights.data}")
            self._log_gradient_info()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clipping
        )
        
        # Update parameters
        self.optimizer_inner.step()
        self.scheduler_inner.step()
        
        return loss.item()
    
    def _log_gradient_info(self):
        """Log essential gradient information."""
        if not is_main_process():
            return
            
        params_with_grad = sum(1 for p in self.model.parameters() if p.grad is not None)
        total_params = sum(1 for p in self.model.parameters())
        
        # logger.info(f"Gradients: {params_with_grad}/{total_params} parameters updated")
        
        # Log instance weights gradient
        if self.instance_weights.grad is not None:
            inst_grad_norm = self.instance_weights.grad.data.norm(2).item()
            # logger.info(f"Instance weights gradient norm: {inst_grad_norm:.6f}")
    
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
        
        # Restore training mode and gradients after outer loop
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        
        return metrics
    
    def _process_meta_batch(self, batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single meta-learning batch.
        
        Args:
            batch: Meta batch data
            
        Returns:
            Dictionary with loss and scores or None if processing failed
        """
        # Handle device placement based on model distribution
        if self.model_is_distributed:
            # For distributed models, move to embedding device
            embed_device = None
            for name, device_id in self.model.hf_device_map.items():
                if 'embed_tokens' in name:
                    embed_device = f"cuda:{device_id}"
                    break
            
            if embed_device is None:
                # Fallback to first device in map
                embed_device = f"cuda:{min(self.model.hf_device_map.values())}"
            
            input_ids = batch["input_ids"].to(embed_device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(embed_device, non_blocking=True)
            meta_labels = batch["meta_labels"].to(embed_device, non_blocking=True)
        else:
            # For single-device models, move tensors to the model device
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            meta_labels = batch["meta_labels"].to(self.device, non_blocking=True)
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # CRITICAL: For device_map models, ensure output tensors require gradients
        if self.model_is_distributed and hasattr(outputs, 'logits'):
            outputs.logits.requires_grad_(True)
        
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
        
        # Ensure meta_labels is on the same device as aggregate_tensor
        if self.model_is_distributed:
            meta_labels = meta_labels.to(aggregate_tensor.device)
        
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