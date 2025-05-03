#!/usr/bin/env python3
"""
Push extracted answer datasets to Hugging Face Hub.

This script takes a folder containing extracted answers in JSONL format
(with train/val/test splits) and pushes it to the Hugging Face Hub under
the HPC-Boys organization.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import datasets
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login
from getpass import getpass
from dotenv import load_dotenv

load_dotenv(override=True)

def load_jsonl_file(file_path: str) -> List[Dict]:
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line in {file_path}: {e}")
    
    print(f"Loaded {len(data)} examples from {file_path}")
    return data

def process_data_for_upload(data: List[Dict]) -> List[Dict]:
    """
    Process the data to create a row for each question with columns for responses.
    Also add total count and correct count for each question.
    """
    processed_data = []
    
    for entry in data:
        # Extract responses and create a new entry
        responses = entry.get("responses", [])
        
        # Create a new processed entry
        processed_entry = {
            # Keep original question fields
            "unique_id": entry.get("unique_id", ""),
            "problem": entry.get("problem", ""),
            "is_mcq": entry.get("is_mcq", False),
            "choices": entry.get("choices", []),
            "choice_index_correct": entry.get("choice_index_correct"),
            "explanation_correct": entry.get("explanation_correct", ""),
            "answer_correct": entry.get("answer_correct", ""),
            "category": entry.get("category", ""),
        }
        
        # Add each response as a separate column
        total_responses = len(responses)
        correct_count = 0
        
        for i, response in enumerate(responses):
            # Add the response
            processed_entry[f"response_{i+1}"] = response.get("full_response", "")
            
            # Add extracted answer if available
            if "extracted_answer" in response:
                processed_entry[f"extracted_answer_{i+1}"] = response.get("extracted_answer", "")
            
            # Add correctness if available
            is_correct = response.get("is_correct", False)
            processed_entry[f"is_correct_{i+1}"] = is_correct
            
            # Count correct responses
            if is_correct:
                correct_count += 1
        
        # Add total counts
        processed_entry["total_responses"] = total_responses
        processed_entry["correct_responses"] = correct_count
        processed_entry["accuracy"] = correct_count / total_responses if total_responses > 0 else 0
        
        processed_data.append(processed_entry)
    
    return processed_data

def create_dataset_from_folder(folder_path: str) -> DatasetDict:
    """
    Create a DatasetDict from a folder containing train.jsonl, val.jsonl, and test.jsonl files.
    Process the data to have a row per question with columns for responses.
    """
    folder_path = Path(folder_path)
    
    # Check that the folder exists
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Folder {folder_path} does not exist or is not a directory")
    
    # Expected files
    train_file = folder_path / "train.jsonl"
    val_file = folder_path / "val.jsonl"
    test_file = folder_path / "test.jsonl"
    
    # Create dataset dictionary
    dataset_dict = {}
    
    # Load train split if it exists
    if train_file.exists():
        train_data = load_jsonl_file(train_file)
        processed_train_data = process_data_for_upload(train_data)
        dataset_dict["train"] = Dataset.from_list(processed_train_data)
    else:
        print(f"Warning: {train_file} not found")
    
    # Load validation split if it exists (already named val.jsonl)
    if val_file.exists():
        val_data = load_jsonl_file(val_file)
        processed_val_data = process_data_for_upload(val_data)
        dataset_dict["validation"] = Dataset.from_list(processed_val_data)
    else:
        print(f"Warning: {val_file} not found")
    
    # Load test split if it exists
    if test_file.exists():
        test_data = load_jsonl_file(test_file)
        processed_test_data = process_data_for_upload(test_data)
        dataset_dict["test"] = Dataset.from_list(processed_test_data)
    else:
        print(f"Warning: {test_file} not found")
    
    # Check if we have any data
    if not dataset_dict:
        raise ValueError(f"No valid JSONL files found in {folder_path}")
    
    return DatasetDict(dataset_dict)

def push_to_hub(dataset: DatasetDict, repo_name: str, organization: str = "HPC-Boys", 
                token: Optional[str] = None, private: bool = False):
    """Push a dataset to the Hugging Face Hub."""
    # Ensure we're logged in
    if token:
        login(token=token, add_to_git_credential=True)
    
    # Full repository name
    full_repo_name = f"{organization}/{repo_name}"
    
    # Push the dataset to the hub
    dataset.push_to_hub(
        full_repo_name,
        private=private,
        token=token
    )
    
    print(f"Successfully pushed dataset to {full_repo_name}")
    print(f"You can view it at: https://huggingface.co/datasets/{full_repo_name}")

def parse_model_name(input_folder: str) -> str:
    """Extract a default model name from the input folder path."""
    # Get the last part of the folder path
    folder_name = os.path.basename(os.path.normpath(input_folder))
    # Replace underscores with spaces and capitalize words
    return folder_name.replace('_', ' ').title()

def main():
    parser = argparse.ArgumentParser(description="Push extracted answers to Hugging Face Hub")
    parser.add_argument("--folder", "-f", type=str, required=True,
                       help="Folder containing the extracted answer files (train.jsonl, val.jsonl, test.jsonl)")
    parser.add_argument("--repo-name", "-r", type=str, 
                       help="Repository name to use on Hugging Face Hub (default: based on folder name)")
    parser.add_argument("--org", "-o", type=str, default="HPC-Boys",
                       help="Organization name on Hugging Face Hub (default: HPC-Boys)")
    parser.add_argument("--private", action="store_true",
                       help="Make the dataset private (default: public)")
    parser.add_argument("--token", "-t", type=str, 
                       help="Hugging Face API token (if not provided, will prompt or use cached token)")
    
    args = parser.parse_args()
    
    # Get a default repo name if none provided
    if not args.repo_name:
        default_name = parse_model_name(args.folder)
        args.repo_name = input(f"Enter dataset repository name [default: {default_name}]: ")
        if not args.repo_name:
            args.repo_name = default_name
    
    # Make sure the repo name is valid for HF Hub (lowercase, no spaces)
    args.repo_name = args.repo_name.lower().replace(' ', '-')
    
    # Ensure we have a valid token
    args.token = os.environ.get("HF_TOKEN") if os.environ.get("HF_TOKEN") else getpass("Enter your Hugging Face API token (input will be hidden): ")
    
    # Double-check before pushing
    print(f"\nPreparing to push dataset:")
    print(f"  - Source folder: {args.folder}")
    print(f"  - Destination: {args.org}/{args.repo_name}")
    print(f"  - Visibility: {'Private' if args.private else 'Public'}")
    
    confirmation = input("\nProceed with upload? (y/n): ")
    if confirmation.lower() != 'y':
        print("Upload cancelled.")
        return
    
    # Create and push the dataset
    try:
        dataset = create_dataset_from_folder(args.folder)
        push_to_hub(
            dataset=dataset, 
            repo_name=args.repo_name, 
            organization=args.org,
            token=args.token,
            private=args.private
        )
    except Exception as e:
        print(f"Error pushing dataset to hub: {e}")
        return
    
    print("\nDataset upload completed successfully!")

if __name__ == "__main__":
    main() 