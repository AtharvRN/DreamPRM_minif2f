"""Dataset classes for DreamPRM training.

This module contains the main dataset class for loading and processing
Process Reward Model training data.
"""

import logging
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .processing import (
    read_jsonl,
    extract_steps_from_cot_response,
    extract_rewards_from_cot_steps
)

logger = logging.getLogger(__name__)


class PRMDataset(Dataset):
    """Dataset for Process Reward Model training.
    
    This dataset loads chain-of-thought reasoning examples with step-wise
    rewards and prepares them for training a Process Reward Model.
    
    The dataset expects JSONL files where each line contains:
    - cot_response: Chain-of-thought reasoning text with step markers
    - cot_steps: Step-wise annotations with reward values (pi)
    - meta_label: Binary label for meta-learning (0 or 1)
    - informal_prefix/formal_statement: Problem statement
    """
    
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 4096,
        expected_steps: int = 5,
        min_rewards: int = 4
    ):
        """Initialize the PRM dataset.
        
        Args:
            jsonl_path: Path to the JSONL data file
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length for tokenization
            expected_steps: Expected number of reasoning steps
            min_rewards: Minimum number of reward values required
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.expected_steps = expected_steps
        self.min_rewards = min_rewards
        
        # Get the step separator token ID
        try:
            self.step_sep_id = tokenizer.encode("<extra_0>")[0]
        except (IndexError, KeyError):
            raise ValueError("Tokenizer does not support <extra_0> token")
        
        # Storage for processed data
        self.inputs: List[Dict[str, torch.Tensor]] = []
        self.rewards: List[torch.Tensor] = []
        self.meta_labels: List[float] = []
        
        # Load and process the data
        self._load_data(jsonl_path)
        
        logger.info(
            f"Loaded {len(self)} samples from {jsonl_path} "
            f"(expected {expected_steps} steps, min {min_rewards} rewards)"
        )
    
    def _load_data(self, jsonl_path: str) -> None:
        """Load and process data from JSONL file.
        
        Args:
            jsonl_path: Path to the data file
        """
        processed_count = 0
        skipped_count = 0
        
        for record in read_jsonl(jsonl_path):
            if self._process_record(record):
                processed_count += 1
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            logger.warning(
                f"Skipped {skipped_count} records due to insufficient steps/rewards "
                f"or tokenization issues"
            )
    
    def _process_record(self, record: Dict[str, Any]) -> bool:
        """Process a single record from the dataset.
        
        Args:
            record: Dictionary containing the record data
            
        Returns:
            True if the record was successfully processed, False otherwise
        """
        # Extract steps and rewards
        steps = extract_steps_from_cot_response(record.get("cot_response", ""))
        rewards = extract_rewards_from_cot_steps(record.get("cot_steps", []))
        meta_label = record.get("meta_label", 1.0)
        
        # Check if we have enough steps and rewards
        if len(steps) != self.expected_steps or len(rewards) < self.min_rewards:
            return False
        
        # Skip examples where all rewards are 0 (poor quality reasoning)
        if all(reward == 0.0 for reward in rewards[:self.min_rewards]):
            return False
        
        # Create the assistant response with step separators
        assistant_response = "<extra_0>".join(steps)
        
        # Get the problem statement
        user_text = (
            record.get("informal_prefix", "") or 
            record.get("formal_statement", "") or 
            "Theorem"
        )
        
        # Format as conversation
        conversation = [
            {"role": "system", "content": "Expert mathematician reasoning step by step."},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_response},
        ]
        
        # Apply chat template or fallback formatting
        try:
            formatted_text = self.tokenizer.apply_chat_template(
                conversation, tokenize=False
            )
        except Exception:
            # Fallback to simple formatting
            formatted_text = (
                f"{conversation[0]['content']}\n"
                f"USER:\n{conversation[1]['content']}\n"
                f"ASSISTANT:\n{assistant_response}"
            )
        
        # Tokenize the text
        try:
            encoding = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )
            
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}")
            return False
        
        # Verify step separator count
        step_sep_count = (input_ids == self.step_sep_id).sum().item()
        if step_sep_count != self.min_rewards:
            return False
        
        # Store the processed data
        self.inputs.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        self.rewards.append(torch.tensor(rewards[:self.min_rewards], dtype=torch.float32))
        self.meta_labels.append(float(meta_label))
        
        return True
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.inputs)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing:
            - input_ids: Token sequence
            - attention_mask: Attention mask
            - rewards: Step-wise reward values
            - meta_label: Meta-learning label
            - idx: Sample index
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")
        
        return {
            **self.inputs[index],
            "rewards": self.rewards[index],
            "meta_label": self.meta_labels[index],
            "idx": index
        }
    
    def get_step_separator_id(self) -> int:
        """Get the token ID for step separators.
        
        Returns:
            Token ID for the <extra_0> separator
        """
        return self.step_sep_id
