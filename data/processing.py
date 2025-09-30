"""Data processing utilities for DreamPRM.

This module contains functions for reading JSONL files, extracting steps
from chain-of-thought responses, and data collation utilities.
"""

import json
import re
from typing import Iterator, Dict, Any, List, Union
import torch
from torch.utils.data import DataLoader

# Regular expression to match step headers in CoT responses
STEP_HEADER_RE = re.compile(r"^###\s*Step\s+(\d+)\s*:", re.IGNORECASE)


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Read JSONL file line by line.
    
    Args:
        path: Path to the JSONL file
        
    Yields:
        Dict containing parsed JSON objects
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num} in {path}: {e}")
                        continue
    except FileNotFoundError:
        raise FileNotFoundError(f"JSONL file not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Error reading JSONL file {path}: {e}")


def extract_steps_from_cot_response(cot_response: str) -> List[str]:
    """Extract reasoning steps from a chain-of-thought response.
    
    Parses the response text to identify step-by-step reasoning
    sections marked with step headers (e.g., "### Step 1:").
    
    Args:
        cot_response: The chain-of-thought response text
        
    Returns:
        List of extracted step contents (without headers)
    """
    if not isinstance(cot_response, str):
        return []
    
    lines = cot_response.splitlines()
    steps = []
    current_step = []
    in_step = False
    
    for line in lines:
        # Check if this line starts a new step
        if STEP_HEADER_RE.match(line.strip()):
            # Save the previous step if we were in one
            if in_step and current_step:
                steps.append(current_step)
                current_step = []
            in_step = True
        elif in_step:
            # Add content to current step
            current_step.append(line)
    
    # Don't forget the last step
    if in_step and current_step:
        steps.append(current_step)
    
    # Join lines within each step and filter out empty steps
    processed_steps = []
    for step_lines in steps:
        step_content = "\n".join(step_lines).strip()
        if step_content:  # Only keep non-empty steps
            processed_steps.append(step_content)
    
    return processed_steps


def extract_rewards_from_cot_steps(cot_steps: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[float]:
    """Extract reward values from chain-of-thought step annotations.
    
    Args:
        cot_steps: Either a dictionary mapping step indices to step data,
                  or a list of step dictionaries containing reward information
                  
    Returns:
        List of extracted reward values (pi values)
    """
    rewards = []
    
    # Convert dict to list if needed
    if isinstance(cot_steps, dict):
        # Sort by keys to maintain order
        step_list = [cot_steps[k] for k in sorted(cot_steps.keys())]
    elif isinstance(cot_steps, list):
        step_list = cot_steps
    else:
        return rewards
    
    # Extract rewards from each step
    for item in step_list:
        if isinstance(item, dict) and "pi" in item:
            try:
                reward_value = float(item["pi"])
                rewards.append(reward_value)
            except (ValueError, TypeError):
                # Skip invalid reward values
                continue
    
    return rewards


class DataCollatorPRM:
    """Data collator for Process Reward Model training.
    
    Handles batching of variable-length sequences with proper padding
    and maintains reward, meta-label, and index information.
    """
    
    def __init__(self, pad_token_id: int):
        """Initialize the collator.
        
        Args:
            pad_token_id: Token ID to use for padding sequences
        """
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples.
        
        Args:
            batch: List of sample dictionaries from the dataset
            
        Returns:
            Dictionary containing batched tensors:
            - input_ids: Padded input token sequences
            - attention_mask: Attention masks for padded sequences
            - rewards_list: List of reward tensors (variable length)
            - meta_labels: Meta-learning labels
            - idxs: Sample indices
        """
        if not batch:
            raise ValueError("Cannot collate empty batch")
        
        # Find maximum sequence length in the batch
        max_length = max(sample["input_ids"].size(0) for sample in batch)
        
        # Initialize lists for batched data
        input_ids_batch = []
        attention_mask_batch = []
        rewards_list = []
        meta_labels = []
        indices = []
        
        for sample in batch:
            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"]
            
            # Pad sequences to max length
            current_length = input_ids.size(0)
            if current_length < max_length:
                padding_length = max_length - current_length
                
                # Pad input_ids with pad_token_id
                padded_input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), self.pad_token_id, dtype=input_ids.dtype)
                ])
                
                # Pad attention_mask with zeros
                padded_attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(padding_length, dtype=attention_mask.dtype)
                ])
            else:
                padded_input_ids = input_ids
                padded_attention_mask = attention_mask
            
            input_ids_batch.append(padded_input_ids)
            attention_mask_batch.append(padded_attention_mask)
            rewards_list.append(sample["rewards"])
            meta_labels.append(sample["meta_label"])
            indices.append(sample["idx"])
        
        return {
            "input_ids": torch.stack(input_ids_batch),
            "attention_mask": torch.stack(attention_mask_batch),
            "rewards_list": rewards_list,  # Keep as list due to variable lengths
            "meta_labels": torch.tensor(meta_labels, dtype=torch.float),
            "idxs": torch.tensor(indices, dtype=torch.long)
        }
