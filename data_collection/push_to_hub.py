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
from typing import Dict, List, Optional, Any, Tuple
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
    Now handles LLM judge fields: extraction_method and judge_explanation.
    Ensures consistent schema across all splits.
    """
    processed_data = []
    
    # First pass: determine the maximum number of responses across all entries
    max_responses = 0
    for entry in data:
        responses = entry.get("responses", [])
        max_responses = max(max_responses, len(responses))
    
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
        
        # Add each response as a separate column (pad to max_responses for consistency)
        total_responses = len(responses)
        correct_count = 0
        
        # Track extraction method statistics
        extraction_method_counts = {}
        
        # Process all response slots up to max_responses to ensure consistent schema
        for i in range(max_responses):
            if i < len(responses):
                response = responses[i]
                
                # Add the response
                processed_entry[f"response_{i+1}"] = response.get("full_response", "")
                
                # Add extracted answer if available
                processed_entry[f"extracted_answer_{i+1}"] = response.get("extracted_answer", "")
                
                # Add correctness if available
                is_correct = response.get("is_correct", False)
                processed_entry[f"is_correct_{i+1}"] = is_correct
                
                # Add extraction method if available (new field)
                extraction_method = response.get("extraction_method", "unknown")
                processed_entry[f"extraction_method_{i+1}"] = extraction_method
                
                # Count extraction methods
                extraction_method_counts[extraction_method] = extraction_method_counts.get(extraction_method, 0) + 1
                
                # Add judge explanation if available (new field) - always add field for consistency
                processed_entry[f"judge_explanation_{i+1}"] = response.get("judge_explanation", "")
                
                # Count correct responses
                if is_correct:
                    correct_count += 1
            else:
                # Pad with empty values for consistent schema
                processed_entry[f"response_{i+1}"] = ""
                processed_entry[f"extracted_answer_{i+1}"] = ""
                processed_entry[f"is_correct_{i+1}"] = False
                processed_entry[f"extraction_method_{i+1}"] = ""
                processed_entry[f"judge_explanation_{i+1}"] = ""
        
        # Add total counts
        processed_entry["total_responses"] = total_responses
        processed_entry["correct_responses"] = correct_count
        processed_entry["accuracy"] = correct_count / total_responses if total_responses > 0 else 0
        
        # Add extraction method statistics
        processed_entry["extraction_method_counts"] = extraction_method_counts
        
        # Add individual extraction method counts for easier analysis
        for method in ["math_verify", "llm_judge", "regex_fallback"]:
            processed_entry[f"{method}_count"] = extraction_method_counts.get(method, 0)
            processed_entry[f"{method}_accuracy"] = (
                sum(1 for i, response in enumerate(responses) 
                    if response.get("extraction_method") == method and response.get("is_correct", False))
                / extraction_method_counts.get(method, 1)  # Avoid division by zero
                if extraction_method_counts.get(method, 0) > 0 else 0.0
            )
        
        processed_data.append(processed_entry)
    
    return processed_data

def standardize_schemas_across_splits(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Ensure all splits have the same schema by finding the global maximum number of responses
    and reprocessing all splits with consistent field structure.
    """
    # Find global maximum responses across all splits
    global_max_responses = 0
    all_data = [train_data, val_data, test_data]
    
    for data_split in all_data:
        if data_split:  # Check if split exists
            for entry in data_split:
                responses = entry.get("responses", [])
                global_max_responses = max(global_max_responses, len(responses))
    
    print(f"Global maximum responses across all splits: {global_max_responses}")
    
    # Reprocess all splits with the global maximum
    standardized_splits = []
    for data_split in all_data:
        if data_split:
            standardized_split = process_data_for_upload_with_max_responses(data_split, global_max_responses)
            standardized_splits.append(standardized_split)
        else:
            standardized_splits.append([])
    
    return tuple(standardized_splits)

def process_data_for_upload_with_max_responses(data: List[Dict], max_responses: int) -> List[Dict]:
    """
    Process data with a specified maximum number of response columns for schema consistency.
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
        
        # Add each response as a separate column (pad to max_responses for consistency)
        total_responses = len(responses)
        correct_count = 0
        
        # Track extraction method statistics
        extraction_method_counts = {}
        
        # Process all response slots up to max_responses to ensure consistent schema
        for i in range(max_responses):
            if i < len(responses):
                response = responses[i]
                
                # Add the response
                processed_entry[f"response_{i+1}"] = response.get("full_response", "")
                
                # Add extracted answer if available
                processed_entry[f"extracted_answer_{i+1}"] = response.get("extracted_answer", "")
                
                # Add correctness if available
                is_correct = response.get("is_correct", False)
                processed_entry[f"is_correct_{i+1}"] = is_correct
                
                # Add extraction method if available (new field)
                extraction_method = response.get("extraction_method", "unknown")
                processed_entry[f"extraction_method_{i+1}"] = extraction_method
                
                # Count extraction methods
                extraction_method_counts[extraction_method] = extraction_method_counts.get(extraction_method, 0) + 1
                
                # Add judge explanation if available (new field) - always add field for consistency
                processed_entry[f"judge_explanation_{i+1}"] = response.get("judge_explanation", "")
                
                # Count correct responses
                if is_correct:
                    correct_count += 1
            else:
                # Pad with empty values for consistent schema
                processed_entry[f"response_{i+1}"] = ""
                processed_entry[f"extracted_answer_{i+1}"] = ""
                processed_entry[f"is_correct_{i+1}"] = False
                processed_entry[f"extraction_method_{i+1}"] = ""
                processed_entry[f"judge_explanation_{i+1}"] = ""
        
        # Add total counts
        processed_entry["total_responses"] = total_responses
        processed_entry["correct_responses"] = correct_count
        processed_entry["accuracy"] = correct_count / total_responses if total_responses > 0 else 0.0
        
        # Add extraction method statistics
        processed_entry["extraction_method_counts"] = extraction_method_counts
        
        # Add individual extraction method counts for easier analysis
        for method in ["math_verify", "llm_judge", "regex_fallback"]:
            processed_entry[f"{method}_count"] = extraction_method_counts.get(method, 0)
            processed_entry[f"{method}_accuracy"] = (
                sum(1 for i, response in enumerate(responses) 
                    if response.get("extraction_method") == method and response.get("is_correct", False))
                / extraction_method_counts.get(method, 1)  # Avoid division by zero
                if extraction_method_counts.get(method, 0) > 0 else 0.0
            )
        
        processed_data.append(processed_entry)
    
    return processed_data

def create_dataset_from_folder(folder_path: str) -> DatasetDict:
    """
    Create a DatasetDict from a folder containing train.jsonl, val.jsonl, and test.jsonl files.
    Process the data to have a row per question with columns for responses.
    Ensures consistent schema across all splits.
    """
    folder_path = Path(folder_path)
    
    # Check that the folder exists
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Folder {folder_path} does not exist or is not a directory")
    
    # Expected files
    train_file = folder_path / "train.jsonl"
    val_file = folder_path / "val.jsonl"
    test_file = folder_path / "test.jsonl"
    
    # Load raw data first
    train_data = load_jsonl_file(train_file) if train_file.exists() else []
    val_data = load_jsonl_file(val_file) if val_file.exists() else []
    test_data = load_jsonl_file(test_file) if test_file.exists() else []
    
    # Ensure consistent schemas across all splits
    processed_train_data, processed_val_data, processed_test_data = standardize_schemas_across_splits(
        train_data, val_data, test_data
    )
    
    # Create dataset dictionary
    dataset_dict = {}
    
    if processed_train_data:
        dataset_dict["train"] = Dataset.from_list(processed_train_data)
        print(f"✅ Created train split with {len(processed_train_data)} examples")
    else:
        print(f"⚠️  Warning: No train data found")
    
    if processed_val_data:
        dataset_dict["validation"] = Dataset.from_list(processed_val_data)
        print(f"✅ Created validation split with {len(processed_val_data)} examples")
    else:
        print(f"⚠️  Warning: No validation data found")
    
    if processed_test_data:
        dataset_dict["test"] = Dataset.from_list(processed_test_data)
        print(f"✅ Created test split with {len(processed_test_data)} examples")
    else:
        print(f"⚠️  Warning: No test data found")
    
    # Check if we have any data
    if not dataset_dict:
        raise ValueError(f"No valid JSONL files found in {folder_path}")
    
    # Print schema info for verification
    if dataset_dict:
        first_split = list(dataset_dict.values())[0]
        print(f"📋 Schema info: {len(first_split.column_names)} columns")
        print(f"🔢 Response columns: {sum(1 for col in first_split.column_names if col.startswith('response_'))}")
    
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