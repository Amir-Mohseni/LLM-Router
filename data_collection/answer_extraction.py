#!/usr/bin/env python3
"""
Utility module for extracting and verifying answers from model responses
"""
import re
import json
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from math_verify import parse, verify
from tqdm import tqdm

def extract_answer(generated_text: str) -> str:
    """Extract the answer from the <answer>...</answer> tags"""
    answer_match = list(re.finditer(r'<answer>(.*?)</answer>', generated_text, re.DOTALL))
    if answer_match:
        answer_text = answer_match[-1].group(1).strip()

        # Find all \boxed{...} instances with proper handling of nested braces
        boxed_pattern = r'\\boxed\{((?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*)\}'
        boxed_match = list(re.finditer(boxed_pattern, answer_text))
        dollar_math_match = list(re.finditer(r'(\$.*?\$)', answer_text))
        if boxed_match:
            # Return the full boxed expression with the braces
            boxed_text = boxed_match[-1].group(0).strip()
            return boxed_text
        elif dollar_math_match:
            return dollar_math_match[-1].group(1).strip()
        else:
            return answer_text
    else:
        # Also update the pattern for finding boxed expressions outside of answer tags
        boxed_pattern = r'\\boxed\{((?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*)\}'
        boxed_match = list(re.finditer(boxed_pattern, generated_text))
        dollar_math_match = list(re.finditer(r'(\$.*?\$)', generated_text))
        if boxed_match:
            # Return the full boxed expression with the braces
            boxed_text = boxed_match[-1].group(0).strip()
            return boxed_text
        elif dollar_math_match:
            return dollar_math_match[-1].group(1).strip()
        else:
            return "No answer found in formatted output"

def verify_answer(ground_truth: str, generated_answer: str) -> bool:
    """Verify if the generated answer is correct"""
    try:
        gt = parse(ground_truth)
        ga = parse(generated_answer)
        
        clean_ga = re.sub(r'\\boxed\{(.*?)\}', r'\1', generated_answer)
        return verify(gt, ga) or verify(ga, gt) or verify(gt, clean_ga) or verify(clean_ga, gt)
    except Exception as e:
        print(f"Error verifying answers: {e}")
        return False

def process_results_file(input_file: str, output_file: Optional[str] = None) -> Tuple[List[Dict], int, int]:
    """
    Process a results file to extract and verify answers
    
    Args:
        input_file: Path to the input JSONL file with model responses
        output_file: Path to save the processed results (None to skip saving)
        
    Returns:
        Tuple containing (processed_results, correct_count, total_responses)
    """
    # Load the results from JSONL format (one JSON object per line)
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                results.append(json.loads(line))
        
    processed_results = []
    correct_count = 0
    total_responses = 0
    
    # Process each problem result with a progress bar
    for result in tqdm(results, desc=f"Processing {os.path.basename(input_file)}", leave=False):
        problem_result = {
            "unique_id": result.get("unique_id"),
            "problem": result["problem"],
            "is_mcq": result.get("is_mcq", False),
            "choices": result.get("choices", None),
            "choice_index_correct": result.get("choice_index_correct", None),
            "explanation_correct": result.get("explanation_correct", ""),
            "answer_correct": result.get("answer_correct", ""),
            "category": result.get("category", ""),
            "responses": []
        }
        
        # Process each response
        for response in result["responses"]:
            extracted_answer = extract_answer(response["full_response"])
            if result.get("is_mcq", False):
                correct_answer = chr(result["choice_index_correct"] + 65)  # Convert 0-based index to uppercase letter
                correct_answer = f'\\boxed{{{correct_answer}}}' # Wrap the correct answer in \boxed{}
            else:
                correct_answer = result["answer_correct"]
            is_correct = verify_answer(
                ground_truth=correct_answer, 
                generated_answer=extracted_answer,
            )
            
            if is_correct:
                correct_count += 1
            total_responses += 1
            
            # Add the processed response
            processed_response = {
                "full_response": response["full_response"],
                "extracted_answer": extracted_answer,
                "is_correct": is_correct
            }
            problem_result["responses"].append(processed_response)
            
        processed_results.append(problem_result)
    
    # Save the processed results if an output file is specified
    if output_file:
        # Only try to create directory if there's a directory component
        dir_name = os.path.dirname(output_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        # Make sure output file has .jsonl extension
        if not output_file.endswith('.jsonl'):
            output_file = output_file + '.jsonl'
            
        # Save in JSONL format (one JSON object per line)
        with open(output_file, 'w') as f:
            for result in processed_results:
                f.write(json.dumps(result) + '\n')
            
    return processed_results, correct_count, total_responses

def analyze_by_category(results: List[Dict]) -> Dict[str, Dict[str, Union[int, float]]]:
    """Analyze results by category"""
    category_metrics = {}
    
    for result in results:
        category = result.get("category", "unknown")
        
        if category not in category_metrics:
            category_metrics[category] = {
                "total_questions": 0,
                "total_responses": 0,
                "correct_responses": 0
            }
        
        category_metrics[category]["total_questions"] += 1
        
        for response in result["responses"]:
            category_metrics[category]["total_responses"] += 1
            if response["is_correct"]:
                category_metrics[category]["correct_responses"] += 1
    
    # Calculate accuracy for each category
    for category, metrics in category_metrics.items():
        metrics["accuracy"] = (
            metrics["correct_responses"] / metrics["total_responses"] 
            if metrics["total_responses"] > 0 else 0
        )
    
    return category_metrics

def process_directory(input_dir: str, output_dir: Optional[str] = None) -> Dict[str, Dict]:
    """
    Process all result files in a directory
    
    Args:
        input_dir: Directory containing result JSONL files
        output_dir: Directory to save processed results (None to skip saving)
        
    Returns:
        Dictionary mapping model names to their metrics
    """
    metrics = {}
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSONL files in the directory
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    
    # Process each file in the directory with a progress bar
    for filename in tqdm(jsonl_files, desc="Processing files", leave=True):
        input_path = os.path.join(input_dir, filename)
        
        # Determine output path if needed
        output_path = None
        if output_dir:
            output_filename = f"processed_{filename}"
            output_path = os.path.join(output_dir, output_filename)
        
        # Extract model name from filename
        model_name = filename.split('_')[0]
        
        # Process the file
        processed_results, correct, total = process_results_file(input_path, output_path)
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
        
        # Analyze by category
        category_metrics = analyze_by_category(processed_results)
        
        # Store metrics
        metrics[model_name] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "by_category": category_metrics
        }
        
        print(f"Processed {filename}: Accuracy {accuracy:.2%} ({correct}/{total})")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and verify answers from model responses")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file or directory to process")
    parser.add_argument("--output", "-o", help="Output file or directory for processed results (defaults to 'extracted_answers' directory)")
    parser.add_argument("--by-category", action="store_true", help="Show results broken down by category")
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    # Create extracted_answers directory 
    os.makedirs("data_collection/extracted_answers", exist_ok=True)
    
    # Set default output directory to 'extracted_answers' if not specified
    if not output_path:
        if os.path.isdir(input_path):
            output_path = "data_collection/extracted_answers"
        else:
            # For single file, create the directory and use the same filename with 'processed_' prefix
            output_path = os.path.join("data_collection/extracted_answers", f"processed_{os.path.basename(input_path)}")
    else:
        # If output_path is just a filename without directory component, place it in extracted_answers
        if not os.path.dirname(output_path):
            output_path = os.path.join("data_collection/extracted_answers", output_path)
    
    if os.path.isdir(input_path):
        # Process a directory of result files
        metrics = process_directory(input_path, output_path)
        
        # Print summary
        print("\nSummary of all models:")
        for model, result in metrics.items():
            print(f"{model}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
            
            # Show category breakdown if requested
            if args.by_category and "by_category" in result:
                print("  Results by category:")
                for category, cat_metrics in result["by_category"].items():
                    print(f"    {category}: {cat_metrics['accuracy']:.2%} ({cat_metrics['correct_responses']}/{cat_metrics['total_responses']})")
    else:
        # Process a single file
        processed_results, correct, total = process_results_file(input_path, output_path)
        accuracy = correct / total if total > 0 else 0
        
        # Print summary
        print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{total})")
        
        # Show category breakdown if requested
        if args.by_category:
            category_metrics = analyze_by_category(processed_results)
            print("\nResults by category:")
            
            # Sort categories by accuracy (descending)
            sorted_categories = sorted(
                category_metrics.items(),
                key=lambda x: x[1]['accuracy'],
                reverse=True
            )
            
            for category, metrics in sorted_categories:
                print(f"  {category}: {metrics['accuracy']:.2%} ({metrics['correct_responses']}/{metrics['total_responses']})")
        
        if output_path:
            print(f"Processed results saved to {output_path}") 