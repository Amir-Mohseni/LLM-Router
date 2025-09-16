#!/usr/bin/env python3
"""
Example demonstrating how to use the new answer_column_name configuration
for different datasets with different answer field names.
"""

from config import DATASET_CONFIG

# Example configurations for different datasets
AIME_CONFIG = {
    **DATASET_CONFIG,
    "dataset_name": "di-zhang-fdu/AIME_1983_2024",
    "question_column_name": "Question",
    "answer_column_name": "Answer",  # AIME uses "Answer" field
}

BIG_MATH_CONFIG = {
    **DATASET_CONFIG,
    "dataset_name": "AmirMohseni/Big-Math-RL-filtered",
    "question_column_name": "problem",
    "answer_column_name": "answer",  # Big-Math-RL uses "answer" field
}

MATH_500_CONFIG = {
    **DATASET_CONFIG,
    "dataset_name": "HuggingFaceH4/MATH-500",
    "question_column_name": "problem",
    "answer_column_name": "answer",  # MATH-500 uses "answer" field
}

# Standardized format (after normalization)
STANDARDIZED_CONFIG = {
    **DATASET_CONFIG,
    "dataset_name": "HPC-Boys/AIME_1983_2024",
    "question_column_name": "question",
    "answer_column_name": "answer_correct",  # Standardized format uses "answer_correct"
}

print("Example configurations for different datasets:")
print("=" * 60)
print("AIME dataset (original):")
print(f"  Question field: {AIME_CONFIG['question_column_name']}")
print(f"  Answer field: {AIME_CONFIG['answer_column_name']}")
print()
print("Big-Math-RL dataset (original):")
print(f"  Question field: {BIG_MATH_CONFIG['question_column_name']}")
print(f"  Answer field: {BIG_MATH_CONFIG['answer_column_name']}")
print()
print("MATH-500 dataset (original):")
print(f"  Question field: {MATH_500_CONFIG['question_column_name']}")
print(f"  Answer field: {MATH_500_CONFIG['answer_column_name']}")
print()
print("Standardized format (after normalization):")
print(f"  Question field: {STANDARDIZED_CONFIG['question_column_name']}")
print(f"  Answer field: {STANDARDIZED_CONFIG['answer_column_name']}")
print()
print("To use a different dataset, update config.py with the appropriate field names:")
print("DATASET_CONFIG = {")
print('    "dataset_name": "your-dataset-name",')
print('    "question_column_name": "your_question_field",')
print('    "answer_column_name": "your_answer_field",')
print("    # ... other config options")
print("}")
