#!/usr/bin/env python3
import os
import json
import time
import re
import argparse
from typing import List, Dict
from tqdm import tqdm
from pathlib import Path

# Import vLLM for efficient inference
from vllm import LLM, SamplingParams

# Import dataset handling
from datasets import load_dataset

# Import local modules
from data_collection.config import (
    DEFAULT_MODEL, DATASET_NAME, DATASET_SPLIT,
    NUM_PROBLEMS, K_RESPONSES, TEMPERATURE, MAX_TOKENS, OUTPUT_DIR
)
from data_collection.prompts import PROMPT, DEFAULT_SYSTEM_PROMPT

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

def format_prompt(question: str) -> str:
    """Format the question using the template from prompts.py"""
    # Replace the placeholder with the actual question
    formatted_prompt = PROMPT.replace("{{ question }}", question)
    
    # Return the formatted prompt as a string
    return formatted_prompt

def extract_answer(generated_text: str) -> str:
    """Extract the answer from the <answer>...</answer> tags"""
    answer_match = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL)
    return answer_match.group(1).strip() if answer_match else "No answer found in formatted output"

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
    for i, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        outputs = llm.generate(prompt, sampling_params)
        
        responses = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            extracted_answer = extract_answer(generated_text)
            
            responses.append({
                "full_response": generated_text,
                "extracted_answer": extracted_answer
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
        # Format prompt using the template and add system instructions
        question = problem["problem"]
        formatted_prompt = format_prompt(question)
        
        # Prepend system prompt if needed (for models that support it, like Claude)
        # Or just use the formatted prompt as is for most models
        final_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{formatted_prompt}"
        
        # Store the problem information - MATH-500 has 'problem', 'solution', and 'answer' fields
        problems_info.append({
            "problem": question,
            "solution": problem.get("solution", ""),  # Use get() in case field doesn't exist
            "correct_answer": problem.get("answer", "")
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
            "problem": problem_info["problem"],
            "solution": problem_info["solution"],
            "correct_answer": problem_info["correct_answer"],
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
    
    # Calculate basic metrics
    correct_count = 0
    total_responses = 0
    
    for result in results:
        correct_answer = result["correct_answer"]
        
        for response in result["responses"]:
            extracted_answer = response["extracted_answer"]
            # Simple exact match for now
            if extracted_answer.strip() == correct_answer.strip():
                correct_count += 1
            total_responses += 1
    
    accuracy = correct_count / total_responses if total_responses > 0 else 0
    print(f"Overall accuracy: {accuracy:.2%} ({correct_count}/{total_responses})")
    
    # Display a sample of results
    print("\nSample of results:")
    for i, result in enumerate(results[:2]):  # Show first two problems
        print(f"\nProblem {i+1}: {result['problem'][:100]}...")
        print(f"Correct answer: {result['correct_answer']}")
        print("Generated answers:")
        for j, response in enumerate(result["responses"][:2]):  # Show first two responses
            print(f"  Response {j+1}: {response['extracted_answer']}")

if __name__ == "__main__":
    main() 