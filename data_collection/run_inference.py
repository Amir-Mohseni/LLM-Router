#!/usr/bin/env python3
import os
import json
import time
import argparse
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from tqdm import tqdm
from pathlib import Path
import jinja2
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import local modules
from config import (
    LLM_CONFIG, DATASET_CONFIG, GENERATION_CONFIG, PROCESSING_CONFIG, OUTPUT_CONFIG, DEFAULT_SAMPLING_PARAMS
)
from prompts import (
    MATH_PROMPT, MCQ_PROMPT_TEMPLATE, DEFAULT_SYSTEM_PROMPT,
    env as jinja_env
)
# Import dataset loading function
from dataset import load_math_dataset

# Import our LLM module
from LLM import create_llm, BaseLLM

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LLMs on math problems")
    parser.add_argument("--model", type=str, default=LLM_CONFIG["model_name"], 
                        help=f"Model to use for inference (default: {LLM_CONFIG['model_name']})")
    parser.add_argument("--num_problems", type=str, default=str(DATASET_CONFIG["num_problems"]),
                        help=f"Number of problems to test or 'all' for entire dataset (default: {DATASET_CONFIG['num_problems']})")
    parser.add_argument("--k_responses", type=int, default=GENERATION_CONFIG["k_responses"],
                        help=f"Number of responses per problem (default: {GENERATION_CONFIG['k_responses']})")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_CONFIG["output_dir"],
                        help=f"Directory to save results (default: {OUTPUT_CONFIG['output_dir']})")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Filename for results (required)")
    parser.add_argument("--retry_failed", action="store_true",
                        help="Only retry questions that previously failed")
    parser.add_argument("--batch_size", type=int, default=PROCESSING_CONFIG["problem_batch_size"],
                        help=f"Number of problems per batch for checkpointing (default: {PROCESSING_CONFIG['problem_batch_size']})")
    parser.add_argument("--api_base", type=str, default=LLM_CONFIG["base_url"],
                        help=f"Base URL for the API (default: {LLM_CONFIG['base_url']})")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (default: loaded from .env file)")
    parser.add_argument("--max_concurrent", type=int, default=PROCESSING_CONFIG["max_concurrent_requests"],
                        help=f"Maximum concurrent API requests (default: {PROCESSING_CONFIG['max_concurrent_requests']})")
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

async def async_generate_response(llm: BaseLLM, prompt: str, prompt_idx: int, semaphore, k_responses: int = 1) -> Dict:
    """Generate responses for a single prompt using the LLM module asynchronously"""
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

async def generate_responses_async(llm: BaseLLM, prompts: List[str], k_responses: int, max_concurrent: int, skip_failed: bool = True):
    """Generate k responses for each prompt using the LLM module with efficient parallelization"""
    
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
    
    # Filter out failed responses if skip_failed is True
    if skip_failed:
        responses = [response for response in responses if response["success"]]
    
    return [{
        "prompt": response["prompt"],
        "responses": response["responses"],
    } for response in responses]

def save_batch(batch_results: List[Dict], output_file: str):
    """Save a batch of results to the output file in JSONL format."""
    # Ensure the directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    # Append the batch results to the file
    with open(output_file, "a") as f:
        for result in batch_results:
            f.write(json.dumps(result) + "\n")
    
    logger.info(f"Saved {len(batch_results)} results to {output_file}")

def format_choices(choices: List[str]) -> List[str]:
    """Format the choices with option letters (A, B, C, etc.)"""
    option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [f"{option_letters[i]}. {choice}" for i, choice in enumerate(choices)]

async def main_async():
    # Parse command line arguments
    args = parse_args()
    
    # Set up LLM parameters
    api_base = args.api_base
    model_name = args.model
    max_concurrent = args.max_concurrent
    
    # Handle API key - always required now
    api_key = args.api_key or os.environ.get(LLM_CONFIG["api_key_name"])
    if not api_key:
        raise ValueError(f"API key is required. Set it with --api_key or in .env file as {LLM_CONFIG['api_key_name']}.")
    
    # Use default sampling parameters from config
    sampling_params = DEFAULT_SAMPLING_PARAMS
        
    # Log the configuration
    print(f"Running inference with the following settings:")
    print(f"  API Base URL: {api_base}")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {DATASET_CONFIG['dataset_name']} (split: {DATASET_CONFIG['dataset_split']})")
    
    # Handle 'all' or specific number of problems from args
    num_problems_arg = args.num_problems
    if num_problems_arg.lower() != 'all':
        try:
            num_problems_val = int(num_problems_arg)
            if num_problems_val <= 0:
                print(f"Warning: num_problems must be positive or 'all'. Using default from config: {DATASET_CONFIG['num_problems']}")
                num_problems_val = DATASET_CONFIG["num_problems"]
        except ValueError:
            print(f"Warning: Invalid num_problems value '{num_problems_arg}'. Using default from config: {DATASET_CONFIG['num_problems']}")
            num_problems_val = DATASET_CONFIG["num_problems"]
    else:
        num_problems_val = 'all' # Use 'all' string
        
    print(f"  Problems to process: {num_problems_val}")
    print(f"  Responses per problem: {args.k_responses}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max concurrent requests: {max_concurrent}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Output file: {args.output_file}")
    print(f"  Sampling parameters: {sampling_params}")
    
    # Load the dataset using the new function and parsed argument
    dataset = load_math_dataset(
        dataset_name=DATASET_CONFIG["dataset_name"], 
        split=DATASET_CONFIG["dataset_split"], 
        num_problems=num_problems_val
    )
    
    # Initialize LLM
    print(f"Initializing LLM...")
    
    try:
        # Use our LLM module to create the LLM instance
        llm = create_llm(
            model_name=model_name,
            api_key=api_key, 
            api_base=api_base,
            system_prompt=LLM_CONFIG["system_prompt"],
            sampling_params=sampling_params
        )
        if LLM_CONFIG["system_prompt"]:
            print(f"LLM initialized with system prompt: {LLM_CONFIG['system_prompt']}")
        else:
            print(f"LLM initialized")
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return
    
    # Create output file path
    output_file = os.path.join(args.output_dir, args.output_file)
    
    # Track successful and failed questions
    successful_ids = set()
    fully_failed_ids = set()  # Questions where all responses failed
    partial_failed_ids = {}   # Dict mapping question_id -> list of failed response indices
    
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists.")
        with open(output_file, 'r') as f:
            # Load all existing results first to avoid file IO in the loop
            existing_results = [json.loads(line) for line in f if line.strip()]
            
            for data in existing_results:
                try:
                    question_id = data.get("unique_id", "")
                    
                    if question_id:
                        # Check each response individually
                        failed_indices = []
                        for i, resp in enumerate(data.get("responses", [])):
                            response_text = resp.get("full_response", "")
                            if response_text.startswith("Error generating response:") or not response_text.strip():
                                failed_indices.append(i)
                        
                        if len(failed_indices) == len(data.get("responses", [])):
                            # All responses failed, mark for complete retry
                            fully_failed_ids.add(question_id)
                        elif failed_indices:
                            # Some responses failed, mark for partial retry
                            partial_failed_ids[question_id] = failed_indices
                            # Still consider it successful overall since we have some good responses
                            successful_ids.add(question_id)
                        else:
                            # All responses were good
                            successful_ids.add(question_id)
                except Exception as e:
                    logger.error(f"Error parsing result: {e}")
                    continue
        
        print(f"Found {len(successful_ids)} questions with at least one successful response")
        print(f"Found {len(fully_failed_ids)} questions with all responses failed")
        print(f"Found {len(partial_failed_ids)} questions with some responses failed")
        
        # Calculate total failed responses
        total_failed_responses = len(fully_failed_ids) * args.k_responses + sum(len(indices) for indices in partial_failed_ids.values())
        print(f"Total failed responses to retry: {total_failed_responses}")
    else:
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine which questions need processing
    indices_to_process = []
    partial_retry_info = {}  # Map index -> (question_id, failed_indices)
    
    for i, problem in enumerate(dataset):
        question_id = problem.get("unique_id", f"q{i}")
        
        # Case 1: Completely new question (no successful responses)
        if question_id not in successful_ids and question_id not in fully_failed_ids:
            indices_to_process.append(i)
            continue
            
        # Case 2: Fully failed question (all responses failed)
        if question_id in fully_failed_ids:
            if args.retry_failed:
                indices_to_process.append(i)
            continue
            
        # Case 3: Partially failed question (some responses failed)
        if question_id in partial_failed_ids and args.retry_failed:
            indices_to_process.append(i)
            # Store which response indices need to be regenerated
            partial_retry_info[i] = (question_id, partial_failed_ids[question_id])
            
    print(f"Identified {len(indices_to_process)} questions to process out of {len(dataset)} total.")
    if partial_retry_info:
        print(f"Of these, {len(partial_retry_info)} questions will have partial response regeneration.")
    
    # Process in batches for checkpointing
    batch_size = args.batch_size
    total_batches = (len(indices_to_process) + batch_size - 1) // batch_size  # Ceiling division
    
    # Create a tqdm progress bar for the questions to be processed
    with tqdm(total=len(indices_to_process), desc="Processing dataset") as pbar:
        for batch_idx, batch_start in enumerate(range(0, len(indices_to_process), batch_size)):
            batch_end = min(batch_start + batch_size, len(indices_to_process))
            # Get the indices for the current batch
            batch_indices = indices_to_process[batch_start:batch_end]
            
            print(f"\nBatch {batch_idx+1}/{total_batches}: Processing {len(batch_indices)} questions (indices {batch_indices[0]} to {batch_indices[-1]})...")
            
            # Prepare prompts for this batch
            batch_prompts = []
            batch_problems_info = []
            batch_retry_info = []  # Track which prompts are partial retries
            
            for i in batch_indices:
                # Fetch the problem using the index
                problem = dataset[i] 
                question_id = problem.get("unique_id", f"q{i}") # Get ID for info storage
                
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
                
                # Store which responses need to be regenerated (if this is a partial retry)
                if i in partial_retry_info:
                    batch_retry_info.append(partial_retry_info[i])
                else:
                    # Generate all responses for this problem
                    batch_retry_info.append((question_id, list(range(args.k_responses))))
                
                batch_prompts.append(final_prompt)
            
            # Skip this batch if it somehow ended up empty (shouldn't happen with new logic)
            if not batch_prompts:
                print(f"Skipping batch {batch_idx+1}/{total_batches} - unexpectedly empty")
                pbar.update(len(batch_indices)) # Update progress bar for skipped batch
                continue
            
            # Generate responses for this batch using async processing
            batch_start_time = time.time()
            
            print(f"Generating responses for {len(batch_prompts)} prompts with up to {max_concurrent} concurrent requests...")
            
            # Use async processing for all cases
            batch_responses = await generate_responses_async(
                llm=llm,
                prompts=batch_prompts,
                k_responses=args.k_responses,
                max_concurrent=max_concurrent,
                skip_failed=False  # Don't skip failed responses, we're tracking them
            )
            
            batch_time = time.time() - batch_start_time
            print(f"Batch {batch_idx+1}/{total_batches} completed in {batch_time:.2f} seconds ({len(batch_prompts) / batch_time:.2f} prompts/second)")
            
            # Combine problem info with responses for this batch
            batch_results = []
            successful_responses_in_batch = 0
            
            # Load existing data for partial updates
            existing_data_by_id = {}
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        q_id = data.get("unique_id", "")
                        if q_id:
                            existing_data_by_id[q_id] = data
            
            for idx, (problem_info, response_set, retry_info) in enumerate(zip(batch_problems_info, batch_responses, batch_retry_info)):
                question_id = problem_info["unique_id"]
                question_id, retry_indices = retry_info
                
                # For partial retries, merge with existing responses
                if question_id in existing_data_by_id:
                    existing_data = existing_data_by_id[question_id]
                    existing_responses = existing_data.get("responses", [])
                    
                    # Count how many acceptable responses we already have
                    acceptable_responses = []
                    failed_indices = []
                    for i, resp in enumerate(existing_responses):
                        response_text = resp.get("full_response", "")
                        if not response_text.startswith("Error generating response:") and response_text.strip():
                            acceptable_responses.append(resp)
                        else:
                            failed_indices.append(i)
                    
                    # Determine how many new responses we need
                    # If we have enough acceptable responses (>= k), we just replace failed ones
                    # If we don't have enough, we need to generate at least k - len(acceptable) new ones
                    required_new_responses = max(len(failed_indices), args.k_responses - len(acceptable_responses))
                    
                    # For questions that were fully failed, make sure we generate at least k responses
                    # or the original number, whichever is greater
                    if question_id in fully_failed_ids:
                        required_new_responses = max(required_new_responses, args.k_responses, len(existing_responses))
                    
                    # If we're doing a retry but didn't generate enough responses, log a warning
                    if len(response_set["responses"]) < required_new_responses:
                        logger.warning(f"Generated fewer responses than needed for {question_id}: got {len(response_set['responses'])}, need {required_new_responses}")
                    
                    # Decide how to merge responses:
                    # 1. If only some responses failed, replace just those (preserving original array size)
                    # 2. If we don't have enough responses overall, add the new ones to existing good ones
                    if len(acceptable_responses) >= args.k_responses:
                        # We have enough good responses, just replace failed ones
                        merged_responses = list(existing_responses)  # Make a copy
                        
                        # Replace failed responses with new ones, up to what we have
                        for i, new_resp in enumerate(response_set["responses"]):
                            if i < len(failed_indices) and failed_indices[i] < len(merged_responses):
                                merged_responses[failed_indices[i]] = new_resp
                    else:
                        # We don't have enough good responses, keep all good ones and add new ones
                        merged_responses = acceptable_responses
                        # Add all new responses
                        merged_responses.extend(response_set["responses"])
                    
                    # Check if all responses are now valid
                    all_valid = all(
                        not resp.get("full_response", "").startswith("Error generating response:") and 
                        resp.get("full_response", "").strip()
                        for resp in merged_responses
                    )
                    
                    if all_valid:
                        successful_ids.add(question_id)
                        successful_responses_in_batch += 1
                        if question_id in fully_failed_ids:
                            fully_failed_ids.remove(question_id)
                        if question_id in partial_failed_ids:
                            partial_failed_ids.pop(question_id, None)  # Remove from partial failures
                    else:
                        # Some responses still failed, update partial_failed_ids
                        new_failed_indices = []
                        for i, resp in enumerate(merged_responses):
                            response_text = resp.get("full_response", "")
                            if response_text.startswith("Error generating response:") or not response_text.strip():
                                new_failed_indices.append(i)
                        
                        if len(new_failed_indices) == len(merged_responses):
                            fully_failed_ids.add(question_id)
                            if question_id in successful_ids:
                                successful_ids.remove(question_id)
                            logger.warning(f"All responses still failed for question {question_id}")
                        else:
                            partial_failed_ids[question_id] = new_failed_indices
                            successful_ids.add(question_id)  # Still consider it successful overall
                    
                    # Create result with merged responses
                    batch_results.append({
                        "unique_id": problem_info["unique_id"],
                        "problem": problem_info["problem"],
                        "is_mcq": problem_info["is_mcq"],
                        "choices": problem_info["choices"],
                        "choice_index_correct": problem_info["choice_index_correct"],
                        "explanation_correct": problem_info["explanation_correct"],
                        "answer_correct": problem_info["answer_correct"],
                        "category": problem_info["category"],
                        "responses": merged_responses
                    })
                else:
                    # This is a new question - check all responses
                    all_failed = all(resp["full_response"].startswith("Error generating response:") or not resp["full_response"].strip()
                                    for resp in response_set["responses"])
                    
                    if not all_failed:
                        successful_ids.add(question_id)
                        successful_responses_in_batch += 1
                        if question_id in fully_failed_ids:
                            fully_failed_ids.remove(question_id)
                    else:
                        fully_failed_ids.add(question_id)
                        if question_id in successful_ids:
                            successful_ids.remove(question_id)
                        logger.warning(f"All responses failed or were empty for question {question_id}, will retry in next run")
                    
                    # Create standard result
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
            
            # Create a temporary file with updated results
            temp_output_file = f"{output_file}.temp"
            updated_ids = {result["unique_id"] for result in batch_results}
            
            # First copy all non-updated entries from original file
            if os.path.exists(output_file):
                with open(output_file, 'r') as infile, open(temp_output_file, 'w') as outfile:
                    for line in infile:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        q_id = data.get("unique_id", "")
                        # Skip entries that we're updating
                        if q_id not in updated_ids:
                            outfile.write(line)
            else:
                # Create an empty temp file if original doesn't exist
                open(temp_output_file, 'w').close()
            
            # Then append our new batch results
            with open(temp_output_file, 'a') as f:
                for result in batch_results:
                    f.write(json.dumps(result) + "\n")
            
            # Replace the original file with our updated version
            os.replace(temp_output_file, output_file)
            
            print(f"Checkpoint saved with {successful_responses_in_batch} successful questions")
            
            # Ask if user wants to continue or abort
            if successful_responses_in_batch == 0:
                try:
                    response = input("\nAll requests in this batch failed. Continue with next batch? (y/n): ").strip().lower()
                    if response != 'y':
                        print("Aborting batch processing. Saving current results...")
                        break
                except KeyboardInterrupt:
                    print("\nAborting batch processing. Saving current results...")
                    break
            
            # Update progress description to show completion status
            pbar.set_description(f"Processing dataset - {batch_end}/{len(indices_to_process)} problems ({batch_idx+1}/{total_batches} batches)")
            # Update progress bar by the number of items processed in this batch
            pbar.update(len(batch_indices)) 
            
            print(f"Batch complete, {batch_end}/{len(indices_to_process)} problems processed")
            print(f"Success status: {len(successful_ids)} questions successfully processed")
            print(f"Failure status: {len(fully_failed_ids)} questions with failed responses")
    
    # Print summary
    print(f"\nInference completed:")
    print(f"Results saved to {output_file}")
    print(f"Successfully processed {len(successful_ids)} questions")
    print(f"Failed to process {len(fully_failed_ids)} questions completely")
    if partial_failed_ids:
        print(f"Partially failed: {len(partial_failed_ids)} questions with some failed responses")
    
    retry_needed = bool(fully_failed_ids or partial_failed_ids)
    if retry_needed:
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