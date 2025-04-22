#!/usr/bin/env python3
import os
import json
import time
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path
import jinja2

# Import OpenAI client
from openai import OpenAI

# Import dataset handling
from datasets import load_dataset

# Import local modules
from config import (
    DEFAULT_MODEL, DATASET_NAME, DATASET_SPLIT,
    NUM_PROBLEMS, K_RESPONSES, TEMPERATURE, MAX_TOKENS, OUTPUT_DIR,
    PROMPT_BATCH_SIZE, PROBLEM_BATCH_SIZE,
    API_MODE, API_BASE, API_KEY, MODEL_NAME
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
    parser.add_argument("--batch_size", type=int, default=PROBLEM_BATCH_SIZE,
                        help=f"Number of problems to process in each batch (default: {PROBLEM_BATCH_SIZE})")
    parser.add_argument("--api_mode", type=str, choices=["local", "remote"], default=API_MODE,
                        help=f"API mode to use (local: vLLM server, remote: OpenAI API) (default: {API_MODE})")
    parser.add_argument("--api_base", type=str, default=API_BASE,
                        help=f"Base URL for the API (default: {API_BASE})")
    parser.add_argument("--api_key", type=str, default=API_KEY,
                        help=f"API key (default: {API_KEY})")
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

def generate_responses(client: OpenAI, prompts: List[str], k_responses: int, temperature: float, max_tokens: int, model_name: str) -> List[Dict]:
    """Generate k responses for each prompt using the OpenAI API client with n=k parameter"""
    
    # Generate responses for each prompt
    all_responses = []
    
    # Get the batch size for prompt processing
    batch_size = PROMPT_BATCH_SIZE
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing prompt batches"):
        batch_prompts = prompts[i:i+batch_size]
        batch_responses = []
        
        # Process each prompt in the batch, generating all k responses at once
        for prompt_idx, prompt in enumerate(batch_prompts):
            try:
                # Use 'n' parameter to generate k responses in one API call
                completion = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=k_responses
                )
                
                # Extract and store all responses
                responses = []
                for choice in completion.choices:
                    responses.append({
                        "full_response": choice.text.strip()
                    })
                
                batch_responses.append({
                    "prompt": prompt,
                    "responses": responses
                })
                
            except Exception as e:
                print(f"Error generating responses for prompt {i+prompt_idx}: {e}")
                # Add empty responses to maintain count
                responses = [{"full_response": f"Error generating response: {str(e)}"} for _ in range(k_responses)]
                batch_responses.append({
                    "prompt": prompt,
                    "responses": responses
                })
        
        all_responses.extend(batch_responses)
    
    return all_responses

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

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up API client with the specified settings
    api_base = args.api_base
    api_key = args.api_key
    model_name = args.model
    api_mode = args.api_mode
    
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
    print(f"  Output directory: {args.output_dir}")
    
    # Load the dataset
    print(f"Loading dataset {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    
    # Limit to the desired number of problems
    if num_problems > 0:
        dataset = dataset.select(range(min(num_problems, len(dataset))))
    
    print(f"Loaded {len(dataset)} problems")
    
    # Initialize the OpenAI client
    print(f"Initializing OpenAI client...")
    start_time = time.time()
    
    # Create OpenAI client
    client = OpenAI(api_key=api_key, base_url=api_base)
    
    model_init_time = time.time() - start_time
    print(f"Client initialized in {model_init_time:.2f} seconds")
    
    if api_mode == "local":
        print(f"NOTE: Make sure vLLM server is running with command: vllm serve {model_name}")
    
    # Create a unique output file path
    cleaned_model_name = model_name.replace('/', '_')
    dataset_size_str = "full" if num_problems_str == "all" else f"{len(dataset)}"
    output_file = os.path.join(
        args.output_dir, 
        f"{cleaned_model_name}_{dataset_size_str}problems_{args.k_responses}k_{int(time.time())}.jsonl"
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process in batches for checkpointing
    batch_size = args.batch_size
    batch_results = []
    
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        print(f"Processing problems {batch_start+1}-{batch_end} of {len(dataset)}...")
        
        # Prepare prompts for this batch
        batch_prompts = []
        batch_problems_info = []
        
        for i in range(batch_start, batch_end):
            problem = dataset[i]
            
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
                "unique_id": problem.get("unique_id", ""),
                "problem": question,
                "is_mcq": is_mcq,
                "choices": problem.get("choices", None),
                "choice_index_correct": problem.get("choice_index_correct", None),
                "explanation_correct": problem.get("explanation_correct", ""),
                "answer_correct": problem.get("answer_correct", ""),
                "category": problem.get("category", "")
            })
            
            batch_prompts.append(final_prompt)
        
        # Generate responses for this batch
        batch_start_time = time.time()
        
        batch_responses = generate_responses(
            client=client,
            prompts=batch_prompts,
            k_responses=args.k_responses,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            model_name=model_name
        )
        
        batch_time = time.time() - batch_start_time
        print(f"Batch completed in {batch_time:.2f} seconds")
        
        # Combine problem info with responses for this batch
        for problem_info, response_set in zip(batch_problems_info, batch_responses):
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
        
        # Clear the batch results for the next batch
        batch_results = []
    
    print(f"Results saved to {output_file}")
    print("\nTo extract and analyze answers, use the answer_extraction.py script:")
    print(f"python -m data_collection.answer_extraction --input {output_file}")

if __name__ == "__main__":
    main() 