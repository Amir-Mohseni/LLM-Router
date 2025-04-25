#!/usr/bin/env python3
import pytest
from data_collection.dataset import load_math_dataset
from datasets import Dataset

# Import config defaults for testing
from data_collection.config import DATASET_NAME, DATASET_SPLIT, NUM_PROBLEMS


def test_load_default_config():
    """Test loading dataset with default configuration."""
    print("\n--- Testing load with default config ---")
    try:
        dataset = load_math_dataset()
        assert isinstance(dataset, Dataset)
        # Check if it loaded the default number of problems (or less if dataset is smaller)
        assert len(dataset) <= NUM_PROBLEMS if isinstance(NUM_PROBLEMS, int) else True
        print(f"Loaded {len(dataset)} problems successfully (default num_problems={NUM_PROBLEMS}).")
        print(f"Dataset features: {dataset.features}")
        if len(dataset) > 0:
            print(f"First example:\n{dataset[0]}")
    except Exception as e:
        pytest.fail(f"Loading with default config failed: {e}")

def test_load_specific_number():
    """Test loading a specific number of problems."""
    print("\n--- Testing load with specific number (10) ---")
    num_to_load = 10
    try:
        dataset = load_math_dataset(num_problems=num_to_load)
        assert isinstance(dataset, Dataset)
        assert len(dataset) <= num_to_load # Should load at most 10 problems
        print(f"Loaded {len(dataset)} problems successfully (requested {num_to_load}).")
    except Exception as e:
        pytest.fail(f"Loading {num_to_load} problems failed: {e}")

def test_load_all_problems():
    """Test loading all problems from the split."""
    print("\n--- Testing load with 'all' problems ---")
    try:
        # Load a small known dataset split to check count easily if possible, else use default
        # For now, just use the default config dataset/split and load all
        dataset = load_math_dataset(num_problems='all')
        assert isinstance(dataset, Dataset)
        # We don't know the exact size of the default split beforehand, but it should load successfully
        assert len(dataset) > 0 
        print(f"Loaded all {len(dataset)} problems successfully.")
    except Exception as e:
        pytest.fail(f"Loading 'all' problems failed: {e}")

def test_load_invalid_num_problems():
    """Test loading with invalid num_problems values."""
    print("\n--- Testing load with invalid num_problems ---")
    with pytest.raises(ValueError):
        load_math_dataset(num_problems=-5)
    with pytest.raises(ValueError):
        load_math_dataset(num_problems="invalid_string")
    # 0 or negative should load all with a warning, check if we want to raise error instead
    # Current implementation prints warning and loads all for <= 0, let's test that behavior
    try:
        print("Testing num_problems=0 (should load all with warning)")
        dataset_zero = load_math_dataset(num_problems=0)
        assert len(dataset_zero) > 0 # Should load the full dataset
        print(f"Loaded {len(dataset_zero)} problems for num_problems=0")
    except Exception as e:
        pytest.fail(f"Loading with num_problems=0 failed unexpectedly: {e}")
        
# You might need to configure PYTHONPATH or use `python -m pytest` for imports to work
# Example: export PYTHONPATH=$PYTHONPATH:/path/to/your/project/root 