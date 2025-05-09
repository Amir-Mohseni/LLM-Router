#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import logging
from collections import defaultdict
import dataset

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import config before other imports to ensure it's fully loaded
from config import (
    DATASET_NAME, DATASET_SPLIT,
    NUM_PROBLEMS, K_RESPONSES, TEMPERATURE, MAX_TOKENS, OUTPUT_DIR,
    PROBLEM_BATCH_SIZE, API_MODE, API_BASE, API_KEY_NAME,
    MODEL_NAME, GENERATION_KWARGS,
    MAX_CONCURRENT_REQUESTS
)

from run_inference import main as run_inference_main

def parse_args():
    parser = argparse.ArgumentParser(description="Run selective inference on problems a weaker model failed on")
    
    parser.add_argument("--weak_model_results", type=str, required=True,
                        help="Path to the processed results from the weaker model (JSONL)")
    parser.add_argument("--failure_threshold", type=float, default=0.5,
                        help="Threshold for considering a question failed (e.g., 0.5 means >50% of runs were wrong)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help=f"Directory to save results (default: {OUTPUT_DIR})")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Filename for selective inference results")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help=f"Stronger model to use for inference (default: {MODEL_NAME})")
    parser.add_argument("--api_mode", type=str, choices=["local", "remote"], default=API_MODE,
                        help=f"API mode to use (default: {API_MODE})")
    
    # Add any additional run_inference.py parameters with defaults from config
    parser.add_argument("--k_responses", type=int, default=K_RESPONSES,
                        help=f"Number of responses per problem (default: {K_RESPONSES})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Sampling temperature (default: {TEMPERATURE})")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS,
                        help=f"Maximum number of tokens for generation (default: {MAX_TOKENS})")
    parser.add_argument("--api_base", type=str, default=API_BASE,
                        help=f"Base URL for the API (default: {API_BASE})")
    parser.add_argument("--api_key", type=str, default=None,
                        help=f"API key (default: loaded from .env file as {API_KEY_NAME})")
    parser.add_argument("--max_concurrent", type=int, default=MAX_CONCURRENT_REQUESTS,
                        help=f"Maximum concurrent API requests (default: {MAX_CONCURRENT_REQUESTS})")
    parser.add_argument("--batch_size", type=int, default=PROBLEM_BATCH_SIZE,
                        help=f"Number of problems per batch for checkpointing (default: {PROBLEM_BATCH_SIZE})")
    return parser.parse_args()

def filter_failed_problems(results_file, failure_threshold=0.5, k_responses=K_RESPONSES):
    """
    Extract problems that the weaker model failed on more than failure_threshold percent of the time.
    
    Args:
        results_file: Path to the processed results JSONL file
        failure_threshold: Minimum failure rate to include a problem (0.5 = 50%)
        k_responses: Maximum number of responses to consider per problem
        
    Returns:
        List of failed problem dictionaries with their unique IDs
    """
    failed_problems = []
    
    # Check if file exists
    if not os.path.exists(results_file):
        logger.error(f"Results file not found: {results_file}")
        return []
    
    # Parse the JSONL file
    with open(results_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                problem_data = json.loads(line)
                problem_id = problem_data.get("unique_id", "")
                
                if not problem_id:
                    logger.warning(f"Found problem without unique_id, skipping")
                    continue
                
                # Check the responses for correctness
                responses = problem_data.get("responses", [])
                if not responses:
                    logger.warning(f"Problem {problem_id} has no responses, skipping")
                    continue
                
                # If there are more than k responses, keep only the top k
                # Sort by correctness (correct first), and take the first k
                if len(responses) > k_responses:
                    logger.info(f"Problem {problem_id} has {len(responses)} responses, limiting to top {k_responses}")
                    # Sort responses with correct ones first
                    sorted_responses = sorted(responses, key=lambda r: r.get("is_correct", False), reverse=True)
                    responses = sorted_responses[:k_responses]
                
                # Count total responses and correct responses
                total_count = len(responses)
                correct_count = sum(1 for resp in responses if resp.get("is_correct", False))
                
                # Calculate failure rate
                failure_rate = 1.0 - (correct_count / total_count) if total_count > 0 else 0
                
                # If failure rate exceeds threshold, add to list
                if failure_rate > failure_threshold:
                    failed_problems.append({
                        "unique_id": problem_id,
                        "problem": problem_data.get("problem", ""),
                        "is_mcq": problem_data.get("is_mcq", False),
                        "choices": problem_data.get("choices", None),
                        "choice_index_correct": problem_data.get("choice_index_correct", None),
                        "explanation_correct": problem_data.get("explanation_correct", ""),
                        "answer_correct": problem_data.get("answer_correct", ""),
                        "category": problem_data.get("category", ""),
                        "failure_rate": failure_rate
                    })
                    
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON line: {line[:100]}...")
                continue
                
    logger.info(f"Found {len(failed_problems)} problems with failure rate > {failure_threshold*100}%")
    return failed_problems

def create_failed_problem_dict(problem_data, failure_rate):
    """Helper function to create a standard problem dictionary for failed problems"""
    return {
        "unique_id": problem_data.get("unique_id", ""),
        "problem": problem_data.get("problem", ""),
        "is_mcq": problem_data.get("is_mcq", False),
        "choices": problem_data.get("choices", None),
        "choice_index_correct": problem_data.get("choice_index_correct", None),
        "explanation_correct": problem_data.get("explanation_correct", ""),
        "answer_correct": problem_data.get("answer_correct", ""),
        "category": problem_data.get("category", ""),
        "failure_rate": failure_rate
    }

def create_filtered_dataset(failed_problems, output_path):
    """
    Create a temporary dataset file with just the failed problems
    
    Args:
        failed_problems: List of problem dictionaries
        output_path: Path to save the filtered dataset
        
    Returns:
        Path to the filtered dataset file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the filtered dataset
    with open(output_path, 'w') as f:
        for problem in failed_problems:
            f.write(json.dumps(problem) + '\n')
    
    logger.info(f"Created filtered dataset with {len(failed_problems)} problems at {output_path}")
    return output_path

def main():
    args = parse_args()
    
    # Extract problems the weaker model failed on
    logger.info(f"Analyzing results from weaker model: {args.weak_model_results}")
    
    failed_problems = filter_failed_problems(
        args.weak_model_results, 
        args.failure_threshold,
        args.k_responses
    )
    
    if not failed_problems:
        logger.error("No failed problems found. Exiting.")
        return
    
    # Create a temporary dataset file with just the failed problems
    temp_dataset_path = os.path.join(args.output_dir, "temp_failed_problems_dataset.jsonl")
    create_filtered_dataset(failed_problems, temp_dataset_path)
    
    # Prepare arguments for run_inference
    # We'll use sys.argv to modify the command line arguments before calling run_inference_main
    import sys
    original_argv = sys.argv.copy()
    
    # Build new arguments for run_inference using config-based defaults
    new_argv = [
        "run_inference.py",  # Script name
        f"--model={args.model}",
        f"--num_problems=all",  # Process all problems in our filtered dataset
        f"--k_responses={args.k_responses}",
        f"--temperature={args.temperature}",
        f"--max_tokens={args.max_tokens}",
        f"--output_dir={args.output_dir}",
        f"--output_file={args.output_file}",
        f"--batch_size={args.batch_size}",
        f"--api_mode={args.api_mode}",
        f"--max_concurrent={args.max_concurrent}",
    ]
    
    # Add API base URL if provided
    if args.api_base:
        new_argv.append(f"--api_base={args.api_base}")
    
    # Handle API key based on mode
    if args.api_mode == "remote":
        api_key = args.api_key or os.environ.get(API_KEY_NAME)
        if api_key:
            new_argv.append(f"--api_key={api_key}")
    
    # Override sys.argv temporarily
    sys.argv = new_argv
    
    # Import module for dataset patching
    original_load_math_dataset = dataset.load_math_dataset
    
    def patched_load_math_dataset(dataset_name=None, split=None, num_problems=None):
        """Patched version that loads our filtered dataset instead"""
        logger.info(f"Loading filtered dataset from {temp_dataset_path}")
        
        # Load the JSONL file
        problems = []
        with open(temp_dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        
        logger.info(f"Loaded {len(problems)} problems from filtered dataset")
        return problems
    
    # Apply the monkey patch
    dataset.load_math_dataset = patched_load_math_dataset
    
    try:
        # Run inference on the filtered dataset
        logger.info(f"Running inference with model {args.model} on {len(failed_problems)} failed problems")
        run_inference_main()
    finally:
        # Restore original state
        sys.argv = original_argv
        dataset.load_math_dataset = original_load_math_dataset
        
        # Clean up temporary dataset if desired
        # os.remove(temp_dataset_path)  # Uncomment to remove temp dataset
    
    logger.info(f"Selective inference completed. Results saved to {os.path.join(args.output_dir, args.output_file)}")
    
if __name__ == "__main__":
    main() 