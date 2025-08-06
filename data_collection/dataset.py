#!/usr/bin/env python3
import os
from datasets import load_dataset, Dataset
from typing import Union, Optional

# Import configuration
from config import DATASET_CONFIG

def load_math_dataset(
    dataset_name: str = DATASET_CONFIG["dataset_name"], 
    split: str = DATASET_CONFIG["dataset_split"], 
    num_problems: Union[int, str] = DATASET_CONFIG["num_problems"]
) -> Dataset:
    """
    Loads the specified math dataset from Hugging Face datasets.

    Args:
        dataset_name: The name of the dataset on Hugging Face Hub.
        split: The dataset split to load (e.g., 'train', 'test').
        num_problems: The number of problems to load. Can be an integer or 'all' to load the entire dataset split.

    Returns:
        The loaded dataset as a Hugging Face Dataset object.
        
    Raises:
        ValueError: If num_problems is invalid.
    """
    print(f"Loading dataset {dataset_name} (split: {split})...")
    
    # Load the full dataset split first
    full_dataset = load_dataset(dataset_name, split=split)
    
    # Handle 'all' or specific number of problems
    if isinstance(num_problems, str) and num_problems.lower() == 'all':
        print(f"Loaded all {len(full_dataset)} problems from split '{split}'.")
        return full_dataset
    elif isinstance(num_problems, int) and num_problems > 0:
        limit = min(num_problems, len(full_dataset))
        print(f"Loading the first {limit} problems.")
        return full_dataset.select(range(limit))
    elif isinstance(num_problems, int) and num_problems <= 0:
        print(f"Warning: num_problems ({num_problems}) must be positive or 'all'. Loading all problems.")
        return full_dataset
    else:
        raise ValueError(f"Invalid value for num_problems: {num_problems}. Must be a positive integer or 'all'.")
