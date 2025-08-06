"""
Tests for dataset loading, structure, and prompt formatting.
"""
import sys
import pytest
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from datasets import Dataset
from dataset import load_math_dataset
from config import DATASET_CONFIG
from prompts import (
    MATH_PROMPT, MCQ_PROMPT_TEMPLATE, DEFAULT_SYSTEM_PROMPT,
    env as jinja_env
)

# Required fields based on DATA_FORMAT.md
REQUIRED_FIELDS = [
    'unique_id',
    'question',
    'category',
    'choices',
    'choice_index_correct',
    'explanation_correct',
    'answer_correct'
]

# --- Fixtures ---

@pytest.fixture(scope="module")
def sample_dataset() -> Dataset:
    """Load a small sample of the dataset for testing."""
    sample_size = 20  # Enough to likely include both MCQ and non-MCQ examples
    dataset = load_math_dataset(num_problems=sample_size)
    assert len(dataset) > 0, "Dataset loaded is empty!"
    return dataset

@pytest.fixture(scope="module")
def mcq_example(sample_dataset: Dataset) -> Dict[str, Any]:
    """Get the first MCQ example from the sample dataset."""
    for item in sample_dataset:
        if item.get("choices") is not None:
            return item
    pytest.skip("No MCQ example found in the dataset sample.")

@pytest.fixture(scope="module")
def non_mcq_example(sample_dataset: Dataset) -> Dict[str, Any]:
    """Get the first non-MCQ example from the sample dataset."""
    for item in sample_dataset:
        if item.get("choices") is None:
            return item
    pytest.skip("No non-MCQ example found in the dataset sample.")

# --- Helper Functions ---

def format_mcq_choices(choices: List[str]) -> List[str]:
    """Format choices with option letters (A, B, C, etc.)"""
    option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [f"{option_letters[i]}. {choice}" for i, choice in enumerate(choices)]

def format_mcq_prompt(question: str, choices: List[str]) -> str:
    """Format an MCQ question using Jinja2 templating"""
    template = jinja_env.from_string(MCQ_PROMPT_TEMPLATE)
    return template.render(question=question, choices=choices)

def format_math_prompt(question: str) -> str:
    """Format a non-MCQ question using Jinja2 templating"""
    template = jinja_env.from_string(MATH_PROMPT)
    return template.render(question=question)

# --- Test Classes ---

class TestDatasetStructure:
    """Tests for dataset structure and integrity."""
    
    def test_required_fields_present(self, sample_dataset: Dataset):
        """✓ Dataset contains all required fields."""
        actual_fields = list(sample_dataset.features.keys())
        missing_fields = [field for field in REQUIRED_FIELDS if field not in actual_fields]
        assert not missing_fields, f"Missing required fields: {missing_fields}"
    
    def test_unique_ids(self, sample_dataset: Dataset):
        """✓ All unique_ids are actually unique."""
        ids = [item['unique_id'] for item in sample_dataset]
        assert len(ids) == len(set(ids)), "Found duplicate unique_ids in the dataset"


class TestPromptFormatting:
    """Tests for prompt formatting functionality."""
    
    def test_mcq_prompt_formatting(self, mcq_example: Dict[str, Any]):
        """✓ MCQ prompts are correctly formatted."""
        question = mcq_example['question']
        choices = mcq_example['choices']
        
        # Verify input data
        assert question, "MCQ question is empty"
        assert choices and isinstance(choices, list) and len(choices) > 0, "Invalid MCQ choices"
        
        # Test formatting
        formatted_choices = format_mcq_choices(choices)
        formatted_prompt = format_mcq_prompt(question, formatted_choices)
        
        # Verify output
        assert question in formatted_prompt, "Question missing from formatted prompt"
        assert all(choice in formatted_prompt for choice in formatted_choices), "Choices missing from prompt"
        
        # Test full prompt with system prompt
        full_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{formatted_prompt}"
        assert DEFAULT_SYSTEM_PROMPT in full_prompt, "System prompt missing"
    
    def test_non_mcq_prompt_formatting(self, non_mcq_example: Dict[str, Any]):
        """✓ Non-MCQ prompts are correctly formatted."""
        question = non_mcq_example['question']
        assert question, "Non-MCQ question is empty"
        
        # Test formatting
        formatted_prompt = format_math_prompt(question)
        
        # Verify output
        assert question in formatted_prompt, "Question missing from formatted prompt"
        
        # Test full prompt with system prompt
        full_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{formatted_prompt}"
        assert DEFAULT_SYSTEM_PROMPT in full_prompt, "System prompt missing"