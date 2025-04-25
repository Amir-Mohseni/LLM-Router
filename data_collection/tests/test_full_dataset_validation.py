#!/usr/bin/env python3
"""
Tests for validating the entire dataset structure and consistency.
These tests load the full dataset and can be slow.
"""
import sys
import pytest
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import Counter

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from datasets import Dataset
from data_collection.dataset import load_math_dataset

# Mark all tests in this file as slow
pytestmark = pytest.mark.slow

# --- Fixtures ---

@pytest.fixture(scope="module")
def full_dataset() -> Dataset:
    """Fixture to load the *entire* dataset split for validation."""
    print(f"\nLoading FULL dataset for validation (this may take time)...")
    try:
        # Use default dataset/split from config, but load ALL problems
        dataset = load_math_dataset(num_problems='all')
        assert len(dataset) > 0, "Full dataset loaded is empty!"
        print(f"Loaded {len(dataset)} problems for full validation.")
        return dataset
    except Exception as e:
        pytest.fail(f"Failed to load full dataset: {e}")

# --- Test Functions ---

def test_full_dataset_consistency(full_dataset: Dataset):
    """✓ Check consistency and validity across the full dataset."""
    print(f"\n--- Validating {len(full_dataset)} dataset entries ---")
    unique_ids: Set[str] = set()
    errors: List[str] = [] 
    id_counts = Counter()

    for i, item in enumerate(full_dataset):
        # 1. Check Unique ID
        uid = item.get('unique_id')
        if not uid or not isinstance(uid, str):
            errors.append(f"Entry {i}: Missing or invalid unique_id: {uid}")
        else:
            id_counts[uid] += 1
            if uid in unique_ids:
                 # Error added later after counting all duplicates
                 pass
            unique_ids.add(uid)

        # 2. Check Question
        question = item.get('question')
        if not question or not isinstance(question, str):
            errors.append(f"Entry {i} (ID: {uid}): Missing or invalid question.")

        # 3. Check Category
        category = item.get('category')
        if not category or not isinstance(category, str):
             # Warning rather than error, maybe allow empty category?
             # errors.append(f"Entry {i} (ID: {uid}): Missing or invalid category.")
             pass 

        # 4. Check Consistency based on MCQ type
        choices = item.get('choices')
        choice_index = item.get('choice_index_correct')
        answer = item.get('answer_correct')

        is_mcq = choices is not None

        if is_mcq:
            # MCQ Checks
            if not isinstance(choices, list) or len(choices) < 2:
                errors.append(f"Entry {i} (ID: {uid}): Invalid MCQ choices: {choices}")
            elif not all(isinstance(c, str) for c in choices):
                errors.append(f"Entry {i} (ID: {uid}): Non-string found in MCQ choices: {choices}")
                
            if not isinstance(choice_index, int) or not (0 <= choice_index < len(choices)):
                 errors.append(f"Entry {i} (ID: {uid}): Invalid choice_index_correct ({choice_index}) for choices length {len(choices) if choices else 'N/A'}.")
            elif answer != choices[choice_index]:
                 errors.append(f"Entry {i} (ID: {uid}): answer_correct ('{answer}') does not match choices[{choice_index}] ('{choices[choice_index]}').")
        else:
            # Non-MCQ Checks
            if choices is not None:
                errors.append(f"Entry {i} (ID: {uid}): Non-MCQ item has non-null choices: {choices}")
            if choice_index is not None:
                 errors.append(f"Entry {i} (ID: {uid}): Non-MCQ item has non-null choice_index_correct: {choice_index}")
            if not answer or not isinstance(answer, str):
                 errors.append(f"Entry {i} (ID: {uid}): Missing or invalid answer_correct for non-MCQ.")
                 
    # Check for duplicate IDs gathered during the loop
    duplicates = {uid for uid, count in id_counts.items() if count > 1}
    if duplicates:
        errors.append(f"Found {len(duplicates)} duplicate unique_ids: {list(duplicates)[:10]}... (Total occurrences: {[(uid, id_counts[uid]) for uid in duplicates][:5]}...) ")

    # Assert at the end
    assert not errors, f"Found {len(errors)} consistency errors in the full dataset:\n" + "\n".join([f"- {e}" for e in errors[:20]]) # Show first 20 errors
    
    print(f"\nFull dataset consistency validation passed for {len(full_dataset)} entries.") 