#!/usr/bin/env python3
"""
Test script to demonstrate how prompts are formatted for MCQ and non-MCQ questions
from an actual dataset
"""
import sys
import json
from pathlib import Path

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from datasets import load_dataset
from data_collection.config import DATASET_NAME, DATASET_SPLIT
from data_collection.prompts import (
    MATH_PROMPT, MCQ_PROMPT_TEMPLATE, DEFAULT_SYSTEM_PROMPT,
    env as jinja_env
)

def format_mcq_choices(choices):
    """Format choices with option letters (A, B, C, etc.)"""
    option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [f"{option_letters[i]}. {choice}" for i, choice in enumerate(choices)]

def format_mcq_prompt(question, choices):
    """Format an MCQ question using Jinja2 templating"""
    # Use the environment from prompts.py
    formatted_prompt = jinja_env.from_string(MCQ_PROMPT_TEMPLATE).render(
        question=question,
        choices=choices
    )
    return formatted_prompt

def format_math_prompt(question):
    """Format a non-MCQ question using Jinja2 templating"""
    # Use the environment from prompts.py
    return jinja_env.from_string(MATH_PROMPT).render(question=question)

def main():
    # Load a sample of the dataset
    print(f"Loading dataset {DATASET_NAME}...")
    try:
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        
        # Limit to a small sample
        sample_size = 5
        dataset = dataset.select(range(min(sample_size, len(dataset))))
        
        print(f"Loaded {len(dataset)} sample problems")
        
        # Display dataset fields for the first item to understand structure
        print("\nDataset Fields:")
        print("-"*80)
        for key, value in dataset[0].items():
            print(f"{key}: {type(value).__name__} - {value if len(str(value)) < 100 else str(value)[:100]+'...'}")
        
        # Find an MCQ and a non-MCQ example
        mcq_example = None
        non_mcq_example = None
        
        for item in dataset:
            is_mcq = item.get("choices") is not None
            
            if is_mcq and mcq_example is None:
                mcq_example = item
            elif not is_mcq and non_mcq_example is None:
                non_mcq_example = item
                
            # Break if we found both types
            if mcq_example is not None and non_mcq_example is not None:
                break
        
        # Format and display the MCQ example
        if mcq_example:
            print("\n" + "="*80)
            print("MCQ EXAMPLE:")
            print("="*80)
            print(f"Question: {mcq_example['question']}")
            print(f"Choices: {mcq_example['choices']}")
            print(f"Correct Answer: {mcq_example.get('answer_correct', '')}")
            print(f"Correct Index: {mcq_example.get('choice_index_correct', '')}")
            print(f"Category: {mcq_example.get('category', 'Unknown')}")
            
            # Format the choices
            formatted_choices = format_mcq_choices(mcq_example['choices'])
            
            # Format the prompt
            formatted_prompt = format_mcq_prompt(mcq_example['question'], formatted_choices)
            
            # Add system prompt
            full_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{formatted_prompt}"
            
            print("\nFormatted MCQ Prompt:")
            print("-"*80)
            print(full_prompt)
        else:
            print("\nNo MCQ example found in the dataset sample")
        
        # Format and display the non-MCQ example
        if non_mcq_example:
            print("\n" + "="*80)
            print("NON-MCQ EXAMPLE:")
            print("="*80)
            print(f"Question: {non_mcq_example['question']}")
            print(f"Correct Answer: {non_mcq_example.get('answer_correct', '')}")
            print(f"Category: {non_mcq_example.get('category', 'Unknown')}")
            
            # Format the prompt
            formatted_prompt = format_math_prompt(non_mcq_example['question'])
            
            # Add system prompt
            full_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{formatted_prompt}"
            
            print("\nFormatted Non-MCQ Prompt:")
            print("-"*80)
            print(full_prompt)
        else:
            print("\nNo non-MCQ example found in the dataset sample")
        
        # Save examples to a JSON file for reference
        output_dir = Path(__file__).resolve().parent
        output_file = output_dir / "prompt_examples.json"
        
        examples = {
            "mcq_example": mcq_example if mcq_example else None,
            "non_mcq_example": non_mcq_example if non_mcq_example else None,
            "mcq_formatted_prompt": full_prompt if mcq_example else None,
            "non_mcq_formatted_prompt": full_prompt if non_mcq_example else None
        }
        
        with open(output_file, "w") as f:
            # Convert dataset items to dict for JSON serialization
            if examples["mcq_example"]:
                examples["mcq_example"] = dict(examples["mcq_example"])
            if examples["non_mcq_example"]:
                examples["non_mcq_example"] = dict(examples["non_mcq_example"])
            
            json.dump(examples, f, indent=2)
        
        print(f"\nExamples saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nCannot load the dataset. Make sure the dataset is properly configured in config.py.")
        print("You can still run test_jinja_templates.py to see how the templates are formatted.")

if __name__ == "__main__":
    main() 