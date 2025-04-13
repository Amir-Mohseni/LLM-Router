#!/usr/bin/env python3
"""
Test script to demonstrate Jinja template rendering for MCQ and non-MCQ questions
This script uses fixed examples to show exact formatting regardless of the dataset
"""
import sys
import os
from pathlib import Path
import jinja2

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_collection.prompts import MCQ_PROMPT, MATH_PROMPT, DEFAULT_SYSTEM_PROMPT

# Sample problems
MCQ_QUESTION = "What is the capital of France?"
MCQ_CHOICES = ["London", "Paris", "Rome", "Berlin"]
MCQ_ANSWER = "B"  # Paris

NON_MCQ_QUESTION = "What is the derivative of f(x) = 3x² + 2x - 5?"
NON_MCQ_ANSWER = "6x + 2"

def format_mcq_choices(choices):
    """Format choices with option letters (A, B, C, etc.)"""
    option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [f"{option_letters[i]}. {choice}" for i, choice in enumerate(choices)]

def format_mcq_prompt(question, choices):
    """Format an MCQ question using Jinja2 templating"""
    env = jinja2.Environment()
    template = env.from_string(MCQ_PROMPT)
    
    formatted_prompt = template.render(
        question=question,
        choices=choices
    )
    return formatted_prompt

def format_math_prompt(question):
    """Format a non-MCQ question"""
    return MATH_PROMPT.replace("{{ question }}", question)

def main():
    print("\n" + "="*80)
    print("JINJA TEMPLATE TEST")
    print("="*80)
    
    # Format and display the MCQ example
    print("\nMCQ EXAMPLE:")
    print("-"*80)
    print(f"Question: {MCQ_QUESTION}")
    raw_choices = MCQ_CHOICES
    print(f"Choices: {raw_choices}")
    
    # Format the choices
    formatted_choices = format_mcq_choices(raw_choices)
    print("\nFormatted Choices:")
    for choice in formatted_choices:
        print(f"  {choice}")
    
    # Format the prompt using Jinja2
    formatted_prompt = format_mcq_prompt(MCQ_QUESTION, formatted_choices)
    
    # Add system prompt
    full_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{formatted_prompt}"
    
    print("\nComplete MCQ Prompt:")
    print("-"*80)
    print(full_prompt)
    
    # Format and display the non-MCQ example
    print("\n" + "="*80)
    print("NON-MCQ EXAMPLE:")
    print("-"*80)
    print(f"Question: {NON_MCQ_QUESTION}")
    
    # Format the prompt
    formatted_prompt = format_math_prompt(NON_MCQ_QUESTION)
    
    # Add system prompt
    full_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{formatted_prompt}"
    
    print("\nComplete Non-MCQ Prompt:")
    print("-"*80)
    print(full_prompt)
    
    print("\nTemplate Test Complete")

if __name__ == "__main__":
    main() 