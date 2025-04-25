#!/usr/bin/env python3
import os
import json
import time
import argparse
import asyncio
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path
import jinja2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import OpenAI client
from openai import AsyncOpenAI

# Import dataset handling
from datasets import load_dataset

# Import local modules
from config import (
    DATASET_NAME, DATASET_SPLIT,
    NUM_PROBLEMS, K_RESPONSES, TEMPERATURE, MAX_TOKENS, OUTPUT_DIR,
    PROBLEM_BATCH_SIZE, API_MODE, API_BASE, 
    MODEL_NAME, GENERATION_KWARGS,
    MAX_CONCURRENT_REQUESTS
)
from prompts import (
    MATH_PROMPT, MCQ_PROMPT_TEMPLATE, DEFAULT_SYSTEM_PROMPT,
    env as jinja_env
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LLMs on math problems")
    parser.add_argument("--model", type=str, default=MODEL_NAME, 
                        help=f"Model to use for inference (default: {MODEL_NAME})")
    parser.add_argument("--num_problems", type=str, default=str(NUM_PROBLEMS),
                        help=f"Number of problems to test or 'all' for entire dataset (default: {NUM_PROBLEMS})")
    parser.add_argument("--k_responses", type=int, default=K_RESPONSES,
                        help=f"Number of responses per problem (default: {K_RESPONSES})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Sampling temperature (default: {TEMPERATURE})")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS,
                        help=f"Maximum number of tokens for generation (default: {MAX_TOKENS})")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help=f"Directory to save results (default: {OUTPUT_DIR})")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Filename for results (required)")
    parser.add_argument("--retry_failed", action="store_true",
                        help="Only retry questions that previously failed")
    parser.add_argument("--batch_size", type=int, default=PROBLEM_BATCH_SIZE,
                        help=f"Number of problems per batch for checkpointing (default: {PROBLEM_BATCH_SIZE})")
    parser.add_argument("--api_mode", type=str, choices=["local", "remote"], default=API_MODE,
                        help=f"API mode to use (local: vLLM server, remote: OpenAI API) (default: {API_MODE})")
    parser.add_argument("--api_base", type=str, default=API_BASE,
                        help=f"Base URL for the API (default: {API_BASE})")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (default: loaded from .env file)")
    parser.add_argument("--max_concurrent", type=int, default=MAX_CONCURRENT_REQUESTS,
                        help=f"Maximum concurrent API requests (default: {MAX_CONCURRENT_REQUESTS})")
    return parser.parse_args()

def format_mcq_prompt(question: str, choices: List[str]) -> str:
    """Format an MCQ question using the template from prompts.py"""
    # Use the environment from prompts.py that has all needed filters
    
    # Render the template with the question and choices
    formatted_prompt = jinja_env.from_string(MCQ_PROMPT_TEMPLATE).render(
        question=question,
        choices=choices
    )
    
    return formatted_prompt

def format_prompt(question: str, choices: Optional[List[str]] = None) -> str:
    """Format the question using the appropriate template from prompts.py"""
    # Check if it's an MCQ question
    if choices:
        return format_mcq_prompt(question, choices)
    else:
        # For non-MCQ questions, use the regular prompt with proper Jinja rendering
        return jinja_env.from_string(MATH_PROMPT).render(question=question)

async def async_generate_response(client, prompt, model_name, max_tokens, temperature, k_responses, prompt_idx, semaphore):
    """Generate responses for a single prompt using the OpenAI API client asynchronously"""
    async with semaphore:
        try:
            # Prepare generation parameters
            params = {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "n": k_responses,
                **GENERATION_KWARGS  # Include advanced generation settings from config
            }
            
            # Use 'n' parameter to generate k responses in one API call
            completion = await client.completions.create(**params)
            
            # Extract and store all responses
            responses = []
            for choice in completion.choices:
                responses.append({
                    "full_response": choice.text.strip()
                })
            
            return {
                "prompt": prompt,
                "responses": responses,
                "success": True,
                "prompt_idx": prompt_idx
            }
            
        except Exception as e:
            print(f"Error generating responses for prompt {prompt_idx}: {e}")
            # Add empty responses to maintain count
            responses = [{"full_response": f"Error generating response: {str(e)}"} for _ in range(k_responses)]
            return {
                "prompt": prompt,
                "responses": responses,
                "success": False,
                "prompt_idx": prompt_idx
            }

async def generate_responses_async(client, prompts, k_responses, temperature, max_tokens, model_name, max_concurrent, skip_failed=True):
    """Generate k responses for each prompt using the OpenAI API client with efficient parallelization"""
    
    # Create a semaphore to limit the number of concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create a progress bar for the batch
    pbar = tqdm(total=len(prompts), desc="Generating batch responses", leave=True)
    
    # Modified async_generate_response that updates progress
    async def async_generate_with_progress(client, prompt, model_name, max_tokens, temperature, k_responses, prompt_idx, semaphore):
        result = await async_generate_response(client, prompt, model_name, max_tokens, temperature, k_responses, prompt_idx, semaphore)
        pbar.update(1)  # Update progress bar after each task completes
        return result
    
    # Generate responses for each prompt in parallel
    tasks = []
    for i, prompt in enumerate(prompts):
        task = async_generate_with_progress(
            client=client,
            prompt=prompt,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            k_responses=k_responses,
            prompt_idx=i,
            semaphore=semaphore
        )
        tasks.append(task)
    
    print(f"Submitting all {len(tasks)} tasks in batch and waiting for all to complete...")
    
    # Submit all tasks at once and wait for all to complete
    responses = await asyncio.gather(*tasks)
    
    # Close progress bar
    pbar.close()
    
    # Sort responses by prompt index to maintain order  
    responses.sort(key=lambda x: x["prompt_idx"])
    
    # Filter out failed responses if skip_failed is True
    if skip_failed:
        responses = [response for response in responses if response["success"]]
    
    # Return the responses (without the prompt_idx and success fields)
    return [{
        "prompt": response["prompt"],
        "responses": response["responses"]
    } for response in responses]

def save_results(results: List[Dict], model_name: str, num_problems: int, 
                k_responses: int, output_dir: str) -> str:
    """Save results to a JSONL file with checkpointing"""
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a unique filename
    cleaned_model_name = model_name.replace('/', '_')
    output_file = os.path.join(
        output_dir, 
        f"{cleaned_model_name}_{num_problems}problems_{k_responses}k_{int(time.time())}.jsonl"
    )
    
    # Save the results in JSONL format (one JSON object per line)
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    return output_file

def save_batch(batch_results: List[Dict], output_file: str):
    """Save a batch of results to the output file in JSONL format."""
    with open(output_file, "a") as f:
        for result in batch_results:
            f.write(json.dumps(result) + "\n")

def format_choices(choices: List[str]) -> List[str]:
    """Format the choices with option letters (A, B, C, etc.)"""
    option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [f"{option_letters[i]}. {choice}" for i, choice in enumerate(choices)]

async def main_async():
    # Parse command line arguments
    args = parse_args()
    
    # Set up API client with the specified settings
    api_base = args.api_base
    api_mode = args.api_mode
    
    # Handle API key based on mode
    if api_mode == "local":
        # For local vLLM server, use a dummy API key if none provided
        api_key = args.api_key or os.environ.get("VLLM_API_KEY")
        print("Using local vLLM server mode")
    else:
        # For remote API, require a real API key
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key is required for remote API mode. Set it with --api_key or in .env file as OPENAI_API_KEY.")
        
    model_name = args.model
    max_concurrent = args.max_concurrent
    
    # Log the configuration
    print(f"Running inference with the following settings:")
    print(f"  API Mode: {api_mode}")
    print(f"  API Base URL: {api_base}")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {DATASET_NAME} (split: {DATASET_SPLIT})")
    
    # Handle 'all' or specific number of problems
    if args.num_problems.lower() == 'all':
        num_problems_str = "all"
        num_problems = -1  # Special value to indicate all problems
    else:
        try:
            num_problems = int(args.num_problems)
            num_problems_str = str(num_problems)
            if num_problems <= 0:
                print("Warning: num_problems must be positive or 'all'. Using all problems.")
                num_problems = -1
                num_problems_str = "all"
        except ValueError:
            print(f"Warning: Invalid num_problems value '{args.num_problems}'. Using default {NUM_PROBLEMS}.")
            num_problems = NUM_PROBLEMS
            num_problems_str = str(num_problems)
    
    print(f"  Problems: {num_problems_str}")
    print(f"  Responses per problem: {args.k_responses}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max concurrent requests: {max_concurrent}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Output file: {args.output_file}")
    
    # Load the dataset
    print(f"Loading dataset {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    
    # Limit to the desired number of problems
    if num_problems > 0:
        dataset = dataset.select(range(min(num_problems, len(dataset))))
    
    print(f"Loaded {len(dataset)} problems")
    
    # Initialize the async OpenAI client
    print(f"Initializing async OpenAI client...")
    start_time = time.time()
    
    # Create OpenAI client with async support
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    
    model_init_time = time.time() - start_time
    print(f"Client initialized in {model_init_time:.2f} seconds")
    
    if api_mode == "local":
        print(f"NOTE: Make sure vLLM server is running with command: vllm serve {model_name}")
    
    # Create output file path
    output_file = os.path.join(args.output_dir, args.output_file)
    
    # Track successful and failed questions
    successful_ids = set()
    failed_ids = set()
    
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists.")
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    question_id = data.get("unique_id", "")
                    
                    if question_id:
                        # Check if there were any valid responses
                        all_failed = all(
                            resp.get("full_response", "").startswith("Error generating response:") 
                            for resp in data.get("responses", [])
                        )
                        
                        if all_failed:
                            # If all responses failed, mark for retry
                            failed_ids.add(question_id)
                        else:
                            # If at least one response was successful, mark as done
                            successful_ids.add(question_id)
                except json.JSONDecodeError:
                    continue
        
        print(f"Found {len(successful_ids)} successfully processed questions")
        print(f"Found {len(failed_ids)} questions with failed responses")
    else:
        # Create the output directory if it doesn't exist
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process in batches for checkpointing
    batch_size = args.batch_size
    total_batches = (len(dataset) + batch_size - 1) // batch_size  # Ceiling division
    
    # Create a tqdm progress bar for the overall dataset
    with tqdm(total=len(dataset), desc="Processing dataset") as pbar:
        for batch_idx, batch_start in enumerate(range(0, len(dataset), batch_size)):
            batch_end = min(batch_start + batch_size, len(dataset))
            print(f"\nBatch {batch_idx+1}/{total_batches}: Processing problems {batch_start+1}-{batch_end} of {len(dataset)}...")
            
            # Prepare prompts for this batch
            batch_prompts = []
            batch_problems_info = []
            batch_indices = []  # Track the original dataset indices
            
            skipped_count = 0
            for i in range(batch_start, batch_end):
                problem = dataset[i]
                
                # Get or generate a unique ID for this question
                question_id = problem.get("unique_id", f"q{i}")
                
                # Skip if already processed successfully
                if question_id in successful_ids:
                    print(f"Skipping question {question_id} - already processed successfully")
                    skipped_count += 1
                    continue
                
                # If retry_failed is set, only process questions that failed before
                if args.retry_failed and question_id not in failed_ids and os.path.exists(output_file):
                    print(f"Skipping question {question_id} - not in failed questions list")
                    skipped_count += 1
                    continue
                
                # Get the question and determine if it's MCQ
                question = problem["question"]
                is_mcq = problem.get("choices") is not None
                
                # Format prompt based on question type
                if is_mcq:
                    # Format choices with letters (A, B, C, etc.)
                    formatted_choices = format_choices(problem["choices"])
                    formatted_prompt = format_prompt(question, formatted_choices)
                else:
                    formatted_prompt = format_prompt(question)
                
                # Prepend system prompt
                final_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{formatted_prompt}"
                
                # Store all problem information
                batch_problems_info.append({
                    "unique_id": question_id,
                    "problem": question,
                    "is_mcq": is_mcq,
                    "choices": problem.get("choices", None),
                    "choice_index_correct": problem.get("choice_index_correct", None),
                    "explanation_correct": problem.get("explanation_correct", ""),
                    "answer_correct": problem.get("answer_correct", ""),
                    "category": problem.get("category", "")
                })
                
                batch_prompts.append(final_prompt)
                batch_indices.append(i)
            
            # Skip this batch if all questions have been processed or should be skipped
            if not batch_prompts:
                print(f"Skipping batch {batch_idx+1}/{total_batches} - all questions already processed or skipped")
                pbar.update(batch_end - batch_start)  # Update progress bar for skipped batch
                continue
            
            # Generate responses for this batch using async processing
            batch_start_time = time.time()
            
            print(f"Generating responses for {len(batch_prompts)} prompts with up to {max_concurrent} concurrent requests...")
            batch_responses = await generate_responses_async(
                client=client,
                prompts=batch_prompts,
                k_responses=args.k_responses,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                model_name=model_name,
                max_concurrent=max_concurrent,
                skip_failed=False  # Don't skip failed responses, we're tracking them
            )
            
            batch_time = time.time() - batch_start_time
            print(f"Batch {batch_idx+1}/{total_batches} completed in {batch_time:.2f} seconds ({len(batch_prompts) / batch_time:.2f} prompts/second)")
            
            # Combine problem info with responses for this batch
            batch_results = []
            for problem_info, response_set in zip(batch_problems_info, batch_responses):
                question_id = problem_info["unique_id"]
                
                # Check if all responses were errors
                all_failed = all(resp["full_response"].startswith("Error generating response:") 
                                for resp in response_set["responses"])
                
                # Track success/failure for next run
                if not all_failed:
                    successful_ids.add(question_id)
                    if question_id in failed_ids:
                        failed_ids.remove(question_id)
                else:
                    # If server connection error, mark for retry
                    failed_ids.add(question_id)
                    print(f"Warning: All responses failed for question {question_id}, will retry in next run")
                
                batch_results.append({
                    "unique_id": problem_info["unique_id"],
                    "problem": problem_info["problem"],
                    "is_mcq": problem_info["is_mcq"],
                    "choices": problem_info["choices"],
                    "choice_index_correct": problem_info["choice_index_correct"],
                    "explanation_correct": problem_info["explanation_correct"],
                    "answer_correct": problem_info["answer_correct"],
                    "category": problem_info["category"],
                    "responses": response_set["responses"]
                })
            
            # Save this batch as a checkpoint
            save_batch(batch_results, output_file)
            
            # Update progress description to show completion status
            pbar.set_description(f"Processing dataset - {batch_end}/{len(dataset)} problems ({batch_idx+1}/{total_batches} batches)")
            pbar.update(batch_end - batch_start)
            
            print(f"Checkpoint saved, {batch_end}/{len(dataset)} problems processed")
            print(f"Success status: {len(successful_ids)} questions successfully processed")
            print(f"Failure status: {len(failed_ids)} questions with failed responses")
    
    # Print summary
    print(f"\nInference completed:")
    print(f"Results saved to {output_file}")
    print(f"Successfully processed {len(successful_ids)} questions")
    print(f"Failed to process {len(failed_ids)} questions")
    
    if failed_ids:
        print("\nTo retry failed questions, run:")
        print(f"python -m data_collection.run_inference --retry_failed --output_file {os.path.basename(output_file)}")
    
    print("\nTo extract and analyze answers, use the answer_extraction.py script:")
    print(f"python -m data_collection.answer_extraction --input {output_file}")
    print(f"Results will be saved in the 'extracted_answers' directory by default.")

def main():
    """Main entry point that sets up the asyncio event loop"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 