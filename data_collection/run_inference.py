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

# Import OpenAI client
from openai import AsyncOpenAI

# Import dataset handling
from datasets import load_dataset

# Import local modules
from config import (
    DATASET_NAME, DATASET_SPLIT,
    NUM_PROBLEMS, K_RESPONSES, TEMPERATURE, MAX_TOKENS, OUTPUT_DIR,
    PROBLEM_BATCH_SIZE, API_MODE, API_BASE, API_KEY, 
    MODEL_NAME, MAX_ATTEMPTS_PER_QUESTION, GENERATION_KWARGS
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
                        help=f"Maximum number of tokens for generation and model context (default: {MAX_TOKENS})")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help=f"Directory to save results (default: {OUTPUT_DIR})")
    parser.add_argument("--output_file", type=str, default=None,
                        help=f"Custom filename for results (default: model_dataset.jsonl)")
    parser.add_argument("--max_attempts", type=int, default=MAX_ATTEMPTS_PER_QUESTION,
                        help=f"Maximum number of attempts per question (default: {MAX_ATTEMPTS_PER_QUESTION})")
    parser.add_argument("--batch_size", type=int, default=PROBLEM_BATCH_SIZE,
                        help=f"Number of problems to process in each batch (default: {PROBLEM_BATCH_SIZE})")
    parser.add_argument("--api_mode", type=str, choices=["local", "remote"], default=API_MODE,
                        help=f"API mode to use (local: vLLM server, remote: OpenAI API) (default: {API_MODE})")
    parser.add_argument("--api_base", type=str, default=API_BASE,
                        help=f"Base URL for the API (default: {API_BASE})")
    parser.add_argument("--api_key", type=str, default=API_KEY,
                        help=f"API key (default: {API_KEY})")
    parser.add_argument("--max_concurrent", type=int, default=10,
                        help="Maximum number of concurrent API requests (default: 10)")
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
    
    # Generate responses for each prompt in parallel
    tasks = []
    for i, prompt in enumerate(prompts):
        task = async_generate_response(
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
    
    # Wait for all tasks to complete with a progress bar
    responses = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating responses", leave=False):
        responses.append(await f)
    
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
    api_key = args.api_key
    model_name = args.model
    api_mode = args.api_mode
    max_concurrent = args.max_concurrent
    
    # Log the configuration
    print(f"Running inference with the following settings:")
    print(f"  API Mode: {api_mode}")
    print(f"  API Base URL: {api_base}")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {DATASET_NAME} (split: {DATASET_SPLIT})")
    print(f"  Max concurrent requests: {max_concurrent}")
    
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
    print(f"  Output directory: {args.output_dir}")
    print(f"  Maximum attempts per question: {args.max_attempts}")
    if args.output_file:
        print(f"  Custom output filename: {args.output_file}")
    
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
    
    # Create a unique output file path
    if args.output_file:
        output_file = os.path.join(args.output_dir, args.output_file)
    else:
        # Use a more consistent filename based on the model and dataset
        cleaned_model_name = model_name.replace('/', '_')
        dataset_name = DATASET_NAME.split('/')[-1]
        output_file = os.path.join(
            args.output_dir, 
            f"{cleaned_model_name}_{dataset_name}.jsonl"
        )
    
    # Load attempt tracking data from existing output file
    attempts_by_id = {}
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
                        
                        # Only count as an attempt if there was at least one valid response
                        if not all_failed:
                            # Increment the attempt counter for this question
                            attempts_by_id[question_id] = attempts_by_id.get(question_id, 0) + 1
                except json.JSONDecodeError:
                    continue
        
        print(f"Found data for {len(attempts_by_id)} previously processed questions with valid responses")
    else:
        # Create the output directory if it doesn't exist
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process in batches for checkpointing
    batch_size = args.batch_size
    
    # Create a tqdm progress bar for the overall dataset
    with tqdm(total=len(dataset), desc="Processing dataset") as pbar:
        for batch_start in range(0, len(dataset), batch_size):
            batch_end = min(batch_start + batch_size, len(dataset))
            print(f"Processing problems {batch_start+1}-{batch_end} of {len(dataset)}...")
            
            # Prepare prompts for this batch
            batch_prompts = []
            batch_problems_info = []
            batch_indices = []  # Track the original dataset indices
            
            skipped_count = 0
            for i in range(batch_start, batch_end):
                problem = dataset[i]
                
                # Get or generate a unique ID for this question
                question_id = problem.get("unique_id", f"q{i}")
                
                # Skip if we've already attempted this question the maximum number of times
                if attempts_by_id.get(question_id, 0) >= args.max_attempts:
                    print(f"Skipping question {question_id} - reached max attempts ({args.max_attempts})")
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
            
            # Skip this batch if all questions have reached max attempts
            if not batch_prompts:
                print(f"Skipping batch {batch_start+1}-{batch_end} - all questions have reached max attempts")
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
                skip_failed=False  # Don't skip failed responses, we're tracking attempts
            )
            
            batch_time = time.time() - batch_start_time
            print(f"Batch completed in {batch_time:.2f} seconds ({len(batch_prompts) / batch_time:.2f} prompts/second)")
            
            # Combine problem info with responses for this batch
            batch_results = []
            for problem_info, response_set in zip(batch_problems_info, batch_responses):
                question_id = problem_info["unique_id"]
                
                # Check if all responses were errors
                all_failed = all(resp["full_response"].startswith("Error generating response:") 
                                for resp in response_set["responses"])
                
                # Only update attempt counter if we got at least one valid response
                if not all_failed:
                    attempts_by_id[question_id] = attempts_by_id.get(question_id, 0) + 1
                else:
                    # If server connection error, don't count this as an attempt
                    print(f"Warning: All responses failed for question {question_id}, not counting as an attempt")
                
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
            
            print(f"Checkpoint saved, {batch_end}/{len(dataset)} problems processed")
            print(f"Current attempts status: {len(attempts_by_id)} questions have been attempted at least once")
            
            # Update the progress bar with the number of processed problems in this batch
            pbar.update(batch_end - batch_start)
    
    # Print summary
    print(f"\nInference completed:")
    print(f"Results saved to {output_file}")
    print(f"Processed {len(attempts_by_id)} unique questions")
    
    # Print distribution of attempts
    attempt_counts = {}
    for attempt_count in attempts_by_id.values():
        attempt_counts[attempt_count] = attempt_counts.get(attempt_count, 0) + 1
    
    print("\nAttempt distribution:")
    for attempts, count in sorted(attempt_counts.items()):
        print(f"  {attempts} attempt(s): {count} questions")
    
    print("\nTo extract and analyze answers, use the answer_extraction.py script:")
    print(f"python -m data_collection.answer_extraction --input {output_file}")

def main():
    """Main entry point that sets up the asyncio event loop"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 