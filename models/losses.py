"""Loss functions and PRM-specific utilities.

This module contains loss functions, reward extraction methods,
and scoring utilities for Process Reward Model training.
"""

import logging
from typing import List, Optional
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Loss functions
bce_loss = nn.BCEWithLogitsLoss(reduction="none")
mse_loss = nn.MSELoss()


def make_step_rewards(
    logits: torch.Tensor,
    token_masks: torch.Tensor
) -> List[List[float]]:
    """Extract step rewards using Qwen2.5-Math PRM approach.
    
    This function follows the official Qwen PRM methodology for extracting
    step-wise reward probabilities from model logits.
    
    Args:
        logits: Model logits with shape [batch_size, seq_len, num_classes]
                where num_classes=2 for binary classification
        token_masks: Boolean mask indicating step separator positions
                    with shape [batch_size, seq_len]
                    
    Returns:
        List of lists containing reward probabilities for each sample
    """
    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1)  # [batch_size, seq_len, num_classes]
    
    # Mask probabilities to only keep step separator positions
    probabilities = probabilities * token_masks.unsqueeze(-1)  # Broadcasting mask
    
    all_scores = []
    
    for batch_idx in range(probabilities.size(0)):
        sample_probs = probabilities[batch_idx]  # [seq_len, num_classes]
        valid_positions = token_masks[batch_idx]  # [seq_len]
        
        if valid_positions.sum() > 0:
            # Extract probabilities at valid positions
            step_probs = sample_probs[valid_positions]  # [num_steps, num_classes]
            # Get probability of positive class (index 1)
            positive_probs = step_probs[:, 1]  # [num_steps]
            step_scores = positive_probs.cpu().tolist()
        else:
            step_scores = []
            
        all_scores.append(step_scores)
    
    return all_scores


def extract_step_probabilities(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    step_separator_id: int,
    tokenizer: Optional[PreTrainedTokenizer] = None
) -> List[torch.Tensor]:
    """Extract step probabilities at separator positions.
    
    Args:
        logits: Model output logits
        input_ids: Input token sequences
        step_separator_id: Token ID for step separators (e.g., <extra_0>)
        tokenizer: Optional tokenizer for debugging
        
    Returns:
        List of probability tensors for each sample in the batch
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Logits shape: {logits.shape}")
    
    # Create masks for step separator positions
    token_masks = (input_ids == step_separator_id)  # [batch_size, seq_len]
    
    if logger.isEnabledFor(logging.DEBUG):
        sep_count = token_masks.sum().item()
        logger.debug(f"Found {sep_count} step separator positions")
    
    # Extract step rewards using the official method
    step_rewards = make_step_rewards(logits, token_masks)
    
    # Convert to tensor format while preserving gradients
    step_prob_tensors = []
    device = logits.device
    requires_grad = torch.is_grad_enabled() and logits.requires_grad
    
    for rewards in step_rewards:
        if rewards:
            tensor = torch.tensor(
                rewards,
                dtype=torch.float32,
                device=device,
                requires_grad=requires_grad
            )
        else:
            tensor = torch.tensor(
                [],
                dtype=torch.float32,
                device=device,
                requires_grad=requires_grad
            )
        step_prob_tensors.append(tensor)
    
    return step_prob_tensors


def compute_aggregate_score(step_probabilities: torch.Tensor) -> torch.Tensor:
    """Compute aggregate score from step probabilities.
    
    This function computes a single score representing the overall
    quality of a reasoning chain based on step-wise probabilities.
    
    Args:
        step_probabilities: Tensor of step probabilities
        
    Returns:
        Scalar tensor representing the aggregate score
    """
    if len(step_probabilities) == 0:
        device = step_probabilities.device if hasattr(step_probabilities, 'device') else 'cpu'
        return torch.tensor(0.0, device=device)
    
    # Compute log-odds sum (logit space aggregation)
    # log(p / (1-p)) for each step, then sum
    epsilon = 1e-8  # Prevent division by zero
    log_odds = torch.log(step_probabilities / (1 - step_probabilities + epsilon))
    aggregate = torch.sum(log_odds)
    
    return aggregate


class PRMLossFunction:
    """Process Reward Model loss function handler.
    
    This class encapsulates the loss computation logic for training
    Process Reward Models with step-wise supervision.
    """
    
    def __init__(self, reduction: str = "mean"):
        """Initialize the PRM loss function.
        
        Args:
            reduction: Reduction method for loss computation ('mean', 'sum', 'none')
        """
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.MSELoss(reduction=reduction)
    
    def compute_step_loss(
        self,
        predicted_probs: List[torch.Tensor],
        target_rewards: List[torch.Tensor],
        device: torch.device
    ) -> torch.Tensor:
        """Compute step-wise BCE loss.
        
        Args:
            predicted_probs: List of predicted step probabilities
            target_rewards: List of target reward values
            device: Device for computation
            
        Returns:
            Computed loss tensor
        """
        losses = []
        
        for pred, target in zip(predicted_probs, target_rewards):
            if len(pred) == 0:
                continue
                
            pred = pred.to(device)
            target = target.to(device)
            
            # Ensure same length
            min_length = min(len(pred), len(target))
            if min_length == 0:
                continue
                
            pred = pred[:min_length]
            target = target[:min_length]
            
            # Compute BCE loss
            step_loss = self.bce(pred, target)
            losses.append(step_loss.mean())
        
        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        losses_tensor = torch.stack(losses)
        
        if self.reduction == "mean":
            return losses_tensor.mean()
        elif self.reduction == "sum":
            return losses_tensor.sum()
        else:
            return losses_tensor
    
    def compute_meta_loss(
        self,
        aggregate_scores: torch.Tensor,
        meta_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute meta-learning MSE loss.
        
        Args:
            aggregate_scores: Predicted aggregate scores
            meta_labels: Target meta labels
            
        Returns:
            Computed meta loss
        """
        # Apply sigmoid to aggregate scores before comparison
        predicted_labels = torch.sigmoid(aggregate_scores)
        return self.mse(predicted_labels, meta_labels)
