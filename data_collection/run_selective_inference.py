#!/usr/bin/env python3
import os
import json
import argparse
import subprocess
from pathlib import Path
import logging
from collections import defaultdict

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

from dataset import load_math_dataset

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

def run_inference_direct(args, problems, output_file):
    """
    Run inference directly on the problems without using the run_inference.py script
    """
    logger.info(f"Running direct inference with model {args.model} on {len(problems)} problems")
    
    # Import necessary modules
    import asyncio
    from LLM import create_llm, REMOTE_API_PARAMS, LOCAL_VLLM_PARAMS
    from prompts import (
        MATH_PROMPT, MCQ_PROMPT_TEMPLATE, DEFAULT_SYSTEM_PROMPT, 
        env as jinja_env
    )
    from tqdm import tqdm
    
    # Define helper functions from run_inference.py
    def format_mcq_prompt(question, choices):
        formatted_prompt = jinja_env.from_string(MCQ_PROMPT_TEMPLATE).render(
            question=question,
            choices=choices
        )
        return formatted_prompt

    def format_prompt(question, choices=None):
        if choices:
            return format_mcq_prompt(question, choices)
        else:
            return jinja_env.from_string(MATH_PROMPT).render(question=question)
    
    def format_choices(choices):
        option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return [f"{option_letters[i]}. {choice}" for i, choice in enumerate(choices)]
    
    # Get filtered kwargs based on API mode
    def get_filtered_kwargs(api_mode):
        if api_mode.lower() == "remote":
            return {k: v for k, v in GENERATION_KWARGS.items() if k in REMOTE_API_PARAMS}
        else:
            return GENERATION_KWARGS
    
    # Get the API key based on mode
    if args.api_mode == "remote":
        api_key = args.api_key or os.environ.get(API_KEY_NAME)
        if not api_key:
            raise ValueError(f"API key is required for remote API mode. Set it with --api_key or in .env file as {API_KEY_NAME}.")
    else:
        api_key = "EMPTY"  # For local vLLM server
    
    # Filter generation kwargs based on API mode
    filtered_kwargs = get_filtered_kwargs(args.api_mode)
    
    # Initialize LLM
    print(f"Initializing LLM...")
    try:
        llm = create_llm(
            model_name=args.model,
            api_mode=args.api_mode,
            api_key=api_key, 
            api_base=args.api_base,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            **filtered_kwargs
        )
        print(f"LLM initialized")
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        if args.api_mode == "local":
            print("\nFor local mode, make sure the vLLM server is running. You can start it with:")
            print(f"python -m data_collection.serve_llm --model {args.model}")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Define async functions for generation
    async def async_generate_response(llm, prompt, prompt_idx, semaphore, k_responses=1):
        async with semaphore:
            try:
                responses = []
                error_count = 0
                
                # Generate k responses
                for i in range(k_responses):
                    try:
                        # Use the LLM module to generate a response
                        response_text = await llm.ainvoke(prompt)
                        responses.append({
                            "full_response": response_text.strip()
                        })
                    except Exception as e:
                        error_count += 1
                        error_message = str(e)
                        logger.error(f"Request {i+1}/{k_responses} for prompt {prompt_idx} failed: {error_message}")
                            
                        responses.append({
                            "full_response": f"Error generating response: {error_message}"
                        })
                
                # Check if all responses failed
                if error_count == k_responses:
                    logger.error(f"All {k_responses} responses failed for prompt {prompt_idx}")
                    return {
                        "prompt": prompt,
                        "responses": responses,
                        "success": False,
                        "prompt_idx": prompt_idx
                    }
                
                return {
                    "prompt": prompt,
                    "responses": responses,
                    "success": True,
                    "prompt_idx": prompt_idx
                }
                
            except Exception as e:
                error_message = str(e)
                logger.error(f"Error generating responses for prompt {prompt_idx}: {error_message}")
                
                # Add empty responses to maintain count
                responses = [{"full_response": f"Error generating response: {error_message}"} for _ in range(k_responses)]
                return {
                    "prompt": prompt,
                    "responses": responses,
                    "success": False,
                    "prompt_idx": prompt_idx
                }
    
    async def generate_responses_async(llm, prompts, k_responses, max_concurrent):
        # Create a semaphore to limit the number of concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create a progress bar for the batch
        pbar = tqdm(total=len(prompts), desc="Generating batch responses", leave=True)
        
        # Modified async_generate_response that updates progress
        async def async_generate_with_progress(llm, prompt, prompt_idx, semaphore, k_responses):
            result = await async_generate_response(
                llm=llm, 
                prompt=prompt, 
                prompt_idx=prompt_idx, 
                semaphore=semaphore,
                k_responses=k_responses
            )
            pbar.update(1)  # Update progress bar after each task completes
            return result
        
        # Generate responses for each prompt in parallel
        tasks = []
        for i, prompt in enumerate(prompts):
            task = async_generate_with_progress(
                llm=llm,
                prompt=prompt,
                prompt_idx=i,
                semaphore=semaphore,
                k_responses=k_responses
            )
            tasks.append(task)
        
        print(f"Submitting all {len(tasks)} tasks in batch and waiting for all to complete...")
        
        # Submit all tasks at once and wait for all to complete
        responses = await asyncio.gather(*tasks)
        
        # Close progress bar
        pbar.close()
        
        # Sort responses by prompt index to maintain order  
        responses.sort(key=lambda x: x["prompt_idx"])
        
        return [{
            "prompt": response["prompt"],
            "responses": response["responses"],
        } for response in responses]
    
    # Save results to the output file
    def save_results(results, output_file):
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
        # Write results to the file
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        logger.info(f"Saved {len(results)} results to {output_file}")
    
    # Process in batches
    async def process_problems():
        batch_size = args.batch_size
        k_responses = args.k_responses
        
        # Split problems into batches
        all_results = []
        total_batches = (len(problems) + batch_size - 1) // batch_size  # Ceiling division
        
        # Process each batch
        with tqdm(total=len(problems), desc="Processing problems") as pbar:
            for batch_idx, batch_start in enumerate(range(0, len(problems), batch_size)):
                batch_end = min(batch_start + batch_size, len(problems))
                # Get the problems for the current batch
                batch_problems = problems[batch_start:batch_end]
                
                print(f"\nBatch {batch_idx+1}/{total_batches}: Processing {len(batch_problems)} problems...")
                
                # Prepare prompts for this batch
                batch_prompts = []
                
                for problem in batch_problems:
                    # Get the question and determine if it's MCQ
                    question = problem["problem"]
                    is_mcq = problem.get("is_mcq", False)
                    
                    # Format prompt based on question type
                    if is_mcq:
                        # Format choices with letters (A, B, C, etc.)
                        formatted_choices = format_choices(problem["choices"])
                        formatted_prompt = format_prompt(question, formatted_choices)
                    else:
                        formatted_prompt = format_prompt(question)
                    
                    # Prepend system prompt
                    final_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{formatted_prompt}"
                    batch_prompts.append(final_prompt)
                
                # Generate responses for this batch
                batch_responses = await generate_responses_async(
                    llm=llm,
                    prompts=batch_prompts,
                    k_responses=k_responses,
                    max_concurrent=args.max_concurrent
                )
                
                # Combine with problem info
                batch_results = []
                for i, (problem, response_set) in enumerate(zip(batch_problems, batch_responses)):
                    batch_results.append({
                        "unique_id": problem["unique_id"],
                        "problem": problem["problem"],
                        "is_mcq": problem["is_mcq"],
                        "choices": problem["choices"],
                        "choice_index_correct": problem["choice_index_correct"],
                        "explanation_correct": problem["explanation_correct"],
                        "answer_correct": problem["answer_correct"],
                        "category": problem["category"],
                        "responses": response_set["responses"]
                    })
                
                # Save batch results
                if output_file:
                    # For the first batch, write to the file directly
                    # For subsequent batches, append to the file
                    mode = "w" if batch_idx == 0 else "a"
                    with open(output_file, mode) as f:
                        for result in batch_results:
                            f.write(json.dumps(result) + "\n")
                
                # Add to overall results
                all_results.extend(batch_results)
                
                # Update progress
                pbar.update(len(batch_problems))
        
        return all_results
    
    # Run the main async function
    asyncio.run(process_problems())
    
    logger.info(f"Direct inference completed. Results saved to {output_file}")

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
    temp_dataset_dir = os.path.join(args.output_dir, "temp_dataset")
    os.makedirs(temp_dataset_dir, exist_ok=True)
    temp_dataset_path = os.path.join(temp_dataset_dir, "failed_problems.jsonl")
    create_filtered_dataset(failed_problems, temp_dataset_path)
    
    logger.info(f"Filtered dataset contains {len(failed_problems)} problems")
    
    # Run inference directly on the problems
    output_file = os.path.join(args.output_dir, args.output_file)
    run_inference_direct(args, failed_problems, output_file)
    
    logger.info(f"Selective inference completed. Results saved to {output_file}")
    
if __name__ == "__main__":
    main() 