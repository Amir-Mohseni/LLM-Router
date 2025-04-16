#!/usr/bin/env python3
import os
import json
import time
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path
import jinja2

# Import vLLM for efficient inference
from vllm import LLM, SamplingParams

# Import dataset handling
from datasets import load_dataset

# Import local modules
from config import (
    DEFAULT_MODEL, DATASET_NAME, DATASET_SPLIT,
    NUM_PROBLEMS, K_RESPONSES, TEMPERATURE, MAX_TOKENS, OUTPUT_DIR
)
from prompts import MATH_PROMPT, MCQ_PROMPT, DEFAULT_SYSTEM_PROMPT

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LLMs on math problems")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                        help=f"Model to use for inference (default: {DEFAULT_MODEL})")
    parser.add_argument("--num_problems", type=int, default=NUM_PROBLEMS,
                        help=f"Number of problems to test (default: {NUM_PROBLEMS})")
    parser.add_argument("--k_responses", type=int, default=K_RESPONSES,
                        help=f"Number of responses per problem (default: {K_RESPONSES})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Sampling temperature (default: {TEMPERATURE})")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help=f"Directory to save results (default: {OUTPUT_DIR})")
    return parser.parse_args()

def format_mcq_prompt(question: str, choices: List[str]) -> str:
    """Format an MCQ question using the template from prompts.py"""
    # Use Jinja2 to handle the template with the for loop
    env = jinja2.Environment()
    template = env.from_string(MCQ_PROMPT)
    
    # Render the template with the question and choices
    formatted_prompt = template.render(
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
        # For non-MCQ questions, use the regular prompt
        return MATH_PROMPT.replace("{{ question }}", question)

def generate_responses(llm: LLM, prompts: List[str], k_responses: int, temperature: float) -> List[Dict]:
    """Generate k responses for each prompt using vLLM"""
    # Set sampling parameters for diverse responses
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        n=k_responses  # Generate k responses per prompt
    )
    
    # Generate responses for each prompt
    all_responses = []
    for _, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        outputs = llm.generate(prompt, sampling_params)
        
        responses = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            
            # Store just the full response text without extraction
            responses.append({
                "full_response": generated_text
            })
        
        all_responses.append({
            "prompt": prompt,
            "responses": responses
        })
    
    return all_responses

def save_results(results: List[Dict], model_name: str, num_problems: int, 
                k_responses: int, output_dir: str) -> str:
    """Save results to a JSON file"""
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a unique filename
    cleaned_model_name = model_name.replace('/', '_')
    output_file = os.path.join(
        output_dir, 
        f"{cleaned_model_name}_{num_problems}problems_{k_responses}k_{int(time.time())}.json"
    )
    
    # Save the results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return output_file

def format_choices(choices: List[str]) -> List[str]:
    """Format the choices with option letters (A, B, C, etc.)"""
    option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [f"{option_letters[i]}. {choice}" for i, choice in enumerate(choices)]

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Log the configuration
    print(f"Running inference with the following settings:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {DATASET_NAME} (split: {DATASET_SPLIT})")
    print(f"  Problems: {args.num_problems}")
    print(f"  Responses per problem: {args.k_responses}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Output directory: {args.output_dir}")
    
    # Load the dataset
    print(f"Loading dataset {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    
    # Limit to the desired number of problems
    if args.num_problems > 0:
        dataset = dataset.select(range(min(args.num_problems, len(dataset))))
    
    print(f"Loaded {len(dataset)} problems")
    
    # Prepare prompts for each problem
    prompts = []
    problems_info = []
    
    for problem in dataset:
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
        problems_info.append({
            "unique_id": problem.get("unique_id", ""),
            "problem": question,
            "is_mcq": is_mcq,
            "choices": problem.get("choices", None),
            "choice_index_correct": problem.get("choice_index_correct", None),
            "explanation_correct": problem.get("explanation_correct", ""),
            "answer_correct": problem.get("answer_correct", ""),
            "category": problem.get("category", "")
        })
        
        prompts.append(final_prompt)
    
    # Initialize the LLM
    print(f"Initializing model {args.model}...")
    start_time = time.time()
    
    llm = LLM(model=args.model, max_model_len=2048, dtype="auto")
    
    model_init_time = time.time() - start_time
    print(f"Model initialized in {model_init_time:.2f} seconds")
    
    # Generate responses
    print(f"Generating {args.k_responses} responses for each problem...")
    generation_start_time = time.time()
    
    responses = generate_responses(
        llm=llm,
        prompts=prompts,
        k_responses=args.k_responses,
        temperature=args.temperature
    )
    
    generation_time = time.time() - generation_start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    
    # Combine problem info with responses
    results = []
    for i, (problem_info, response_set) in enumerate(zip(problems_info, responses)):
        results.append({
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
    
    # Save results
    output_file = save_results(
        results=results,
        model_name=args.model,
        num_problems=len(dataset),
        k_responses=args.k_responses,
        output_dir=args.output_dir
    )
    
    print(f"Results saved to {output_file}")
    print("\nTo extract and analyze answers, use the answer_extraction.py script:")
    print(f"python -m data_collection.answer_extraction --input {output_file}")

if __name__ == "__main__":
    main() 