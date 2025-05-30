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
from pydantic import BaseModel, Field
import asyncio
from dataclasses import dataclass

# Import LLM functionality
from LLM import create_llm

# Pydantic models for structured output
class JudgeResponse(BaseModel):
    """Structured response from the LLM judge"""
    is_match: bool = Field(description="Whether the answer matches the correct answer")
    final_answer: str = Field(description="The extracted final answer from the response")
    explanation: str = Field(description="Explanation of the judgment")

class MCQJudgeResponse(BaseModel):
    """Structured response from the LLM judge for MCQ questions"""
    is_match: bool = Field(description="Whether the answer matches the correct answer")
    final_answer: str = Field(description="The extracted final answer from the response")
    explanation: str = Field(description="Explanation of the judgment")

@dataclass
class ExtractionResult:
    """Result of answer extraction including method used"""
    extracted_answer: str
    is_correct: bool
    extraction_method: str  # "regex" or "llm_judge"
    judge_explanation: Optional[str] = None

# Initialize LLM judge instance (will be set up when needed)
_llm_judge = None

def get_llm_judge():
    """Get or create the LLM judge instance"""
    global _llm_judge
    if _llm_judge is None:
        try:
            # Get API key from environment
            import os
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            _llm_judge = create_llm(
                model_name="gemini-2.0-flash",
                api_mode="remote",
                api_key=api_key,
                api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                temperature=0.5,  # Use low temperature for consistent judgments
            )
        except Exception as e:
            print(f"Warning: Could not initialize LLM judge: {e}")
            _llm_judge = None
    return _llm_judge

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

async def llm_judge_extract_and_verify(
    question: str, 
    generated_answer: str, 
    correct_answer: str, 
    is_mcq: bool = False, 
    choices: Optional[List[str]] = None
) -> Tuple[str, bool, str]:
    """
    Use LLM judge to extract answer and verify correctness with structured output.
    
    Args:
        question: The original question
        generated_answer: The model's response
        correct_answer: The correct answer
        is_mcq: Whether this is a multiple choice question
        choices: List of choices for MCQ questions
        
    Returns:
        Tuple of (extracted_answer, is_correct, explanation)
    """
    judge = get_llm_judge()
    if judge is None:
        return "Judge not available", False, "LLM judge could not be initialized"
    
    try:
        if is_mcq:
            # Use MCQ judge prompt
            from prompts import MCQ_JUDGE_PROMPT, env
            
            # Format choices with letters
            formatted_choices = []
            for i, choice in enumerate(choices or []):
                formatted_choices.append(f"{chr(65 + i)}. {choice}")
            
            prompt = env.from_string(MCQ_JUDGE_PROMPT).render(
                question=question,
                choices=formatted_choices,
                answer=generated_answer,
                correct_answer=correct_answer
            )
            
            response = await judge.ainvoke_structured(prompt, MCQJudgeResponse)
        else:
            # Use general judge prompt
            from prompts import JUDGE_PROMPT, env
            
            prompt = env.from_string(JUDGE_PROMPT).render(
                question=question,
                answer=generated_answer,
                correct_answer=correct_answer
            )
            
            response = await judge.ainvoke_structured(prompt, JudgeResponse)
        
        return response.final_answer, response.is_match, response.explanation
        
    except Exception as e:
        print(f"Error in LLM judge: {e}")
        return "Judge error", False, f"Error during LLM judgment: {str(e)}"

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

async def extract_and_verify_with_fallback(
    question: str,
    full_response: str,
    correct_answer: str,
    is_mcq: bool = False,
    choices: Optional[List[str]] = None
) -> ExtractionResult:
    """
    Extract and verify answer with LLM judge fallback.
    
    Args:
        question: The original question
        full_response: The model's full response
        correct_answer: The correct answer
        is_mcq: Whether this is a multiple choice question
        choices: List of choices for MCQ questions
        
    Returns:
        ExtractionResult with extracted answer, correctness, and method used
    """
    # First try regex extraction
    extracted_answer = extract_answer(full_response)
    
    # Check if regex extraction was successful
    regex_failed = (
        extracted_answer == "No answer found in formatted output" or
        extracted_answer.strip() == "" or
        extracted_answer is None
    )
    
    if not regex_failed:
        # Try to verify with regex-extracted answer
        is_correct = verify_answer(correct_answer, extracted_answer)
        
        # If verification passes, use regex result
        if is_correct:
            return ExtractionResult(
                extracted_answer=extracted_answer,
                is_correct=True,
                extraction_method="regex"
            )
    
    # If regex failed or verification failed, use LLM judge
    try:
        llm_extracted, llm_is_correct, llm_explanation = await llm_judge_extract_and_verify(
            question, full_response, correct_answer, is_mcq, choices
        )
        
        return ExtractionResult(
            extracted_answer=llm_extracted,
            is_correct=llm_is_correct,
            extraction_method="llm_judge",
            judge_explanation=llm_explanation
        )
    except Exception as e:
        # If LLM judge also fails, return regex result with error info
        return ExtractionResult(
            extracted_answer=extracted_answer if not regex_failed else "Extraction failed",
            is_correct=False if regex_failed else verify_answer(correct_answer, extracted_answer),
            extraction_method="regex_fallback",
            judge_explanation=f"LLM judge failed: {str(e)}"
        )

def extract_and_verify_math_only(
    full_response: str,
    correct_answer: str
) -> Tuple[str, bool, bool]:
    """
    Extract answer using regex and verify with math_verify only.
    
    Args:
        full_response: The model's full response
        correct_answer: The correct answer
        
    Returns:
        Tuple of (extracted_answer, is_correct, needs_llm_judge)
    """
    # First try regex extraction
    extracted_answer = extract_answer(full_response)
    
    # Check if regex extraction was successful
    regex_failed = (
        extracted_answer == "No answer found in formatted output" or
        extracted_answer.strip() == "" or
        extracted_answer is None
    )
    
    if not regex_failed:
        # Try to verify with regex-extracted answer
        is_correct = verify_answer(correct_answer, extracted_answer)
        
        # If verification passes, we're done
        if is_correct:
            return extracted_answer, True, False
    
    # Need LLM judge
    return extracted_answer if not regex_failed else "Extraction failed", False, True

@dataclass
class LLMJudgeTask:
    """Task for LLM judge processing"""
    problem_idx: int
    response_idx: int
    question: str
    full_response: str
    correct_answer: str
    is_mcq: bool
    choices: Optional[List[str]]
    regex_extracted_answer: str

async def process_llm_judge_batch(tasks: List[LLMJudgeTask], max_workers: int = 10) -> List[ExtractionResult]:
    """
    Process a batch of LLM judge tasks in parallel with controlled concurrency.
    
    Args:
        tasks: List of LLMJudgeTask objects to process
        max_workers: Maximum number of concurrent API calls (default: 10)
        
    Returns:
        List of ExtractionResult objects in the same order as tasks
    """
    if not tasks:
        return []
    
    # Use semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(max_workers)
    
    async def process_single_task(task: LLMJudgeTask) -> Tuple[str, bool, str]:
        """Process a single LLM judge task with semaphore control"""
        async with semaphore:
            return await llm_judge_extract_and_verify(
                task.question,
                task.full_response, 
                task.correct_answer,
                task.is_mcq,
                task.choices
            )
    
    # Create coroutines for all LLM judge calls with semaphore control
    judge_coroutines = [process_single_task(task) for task in tasks]
    
    # Run all LLM judge calls in parallel with controlled concurrency
    try:
        judge_results = await asyncio.gather(*judge_coroutines, return_exceptions=True)
    except Exception as e:
        print(f"Error in batch LLM judge processing: {e}")
        judge_results = [None] * len(tasks)
    
    # Convert results to ExtractionResult objects
    extraction_results = []
    for i, (task, judge_result) in enumerate(zip(tasks, judge_results)):
        if isinstance(judge_result, Exception):
            # Handle exception
            result = ExtractionResult(
                extracted_answer=task.regex_extracted_answer,
                is_correct=False,
                extraction_method="regex_fallback",
                judge_explanation=f"LLM judge failed: {str(judge_result)}"
            )
        elif judge_result is None:
            # Handle None result
            result = ExtractionResult(
                extracted_answer=task.regex_extracted_answer,
                is_correct=False,
                extraction_method="regex_fallback", 
                judge_explanation="LLM judge returned None"
            )
        else:
            # Success case
            llm_extracted, llm_is_correct, llm_explanation = judge_result
            result = ExtractionResult(
                extracted_answer=llm_extracted,
                is_correct=llm_is_correct,
                extraction_method="llm_judge",
                judge_explanation=llm_explanation
            )
        
        extraction_results.append(result)
    
    return extraction_results

async def process_results_file_async(
    input_file: str, 
    output_file: Optional[str] = None,
    batch_size: int = 20,
    max_workers: int = 40
) -> Tuple[List[Dict], int, int]:
    """
    Async version of process_results_file that supports LLM judge fallback with parallel processing
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to save the processed results (None to skip saving)
        batch_size: Number of LLM judge tasks to process in each batch (default: 20)
        max_workers: Maximum number of concurrent API calls per batch (default: 10)
    """
    # Load the results from JSONL format (one JSON object per line)
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                results.append(json.loads(line))
    
    print(f"📊 Processing {len(results)} problems...")
    print(f"⚙️  Configuration: batch_size={batch_size}, max_workers={max_workers}")
    
    # Phase 1: Do math_verify for all responses (sequential, fast)
    print("🔍 Phase 1: Running math verification...")
    processed_results = []
    llm_judge_tasks = []
    math_verify_results = []  # Store results temporarily
    
    for problem_idx, result in enumerate(tqdm(results, desc="Math verification", leave=False)):
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
        
        problem_math_results = []
        
        # Process each response with math verification only
        for response_idx, response in enumerate(result["responses"]):
            if result.get("is_mcq", False):
                correct_answer = chr(result["choice_index_correct"] + 65)  # Convert 0-based index to uppercase letter
                correct_answer = f'\\boxed{{{correct_answer}}}' # Wrap the correct answer in \boxed{}
            else:
                correct_answer = result["answer_correct"]
            
            # First phase: math verification only
            extracted_answer, is_correct, needs_llm_judge = extract_and_verify_math_only(
                response["full_response"],
                correct_answer
            )
            
            problem_math_results.append({
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "needs_llm_judge": needs_llm_judge,
                "full_response": response["full_response"]
            })
            
            # If needs LLM judge, add to the queue
            if needs_llm_judge:
                task = LLMJudgeTask(
                    problem_idx=problem_idx,
                    response_idx=response_idx,
                    question=result["problem"],
                    full_response=response["full_response"],
                    correct_answer=correct_answer,
                    is_mcq=result.get("is_mcq", False),
                    choices=result.get("choices", None),
                    regex_extracted_answer=extracted_answer
                )
                llm_judge_tasks.append(task)
        
        math_verify_results.append(problem_math_results)
        processed_results.append(problem_result)
    
    print(f"✅ Math verification complete. {len(llm_judge_tasks)} responses need LLM judge.")
    
    # Phase 2: Run LLM judge in parallel for failed cases
    llm_judge_results = {}
    if llm_judge_tasks:
        print(f"🧠 Phase 2: Running LLM judge for {len(llm_judge_tasks)} responses...")
        print(f"   📦 Processing in batches of {batch_size} with {max_workers} concurrent workers per batch")
        
        llm_results = []
        
        for i in tqdm(range(0, len(llm_judge_tasks), batch_size), desc="LLM judge batches", leave=False):
            batch = llm_judge_tasks[i:i + batch_size]
            batch_results = await process_llm_judge_batch(batch, max_workers=max_workers)
            llm_results.extend(batch_results)
        
        # Map results back to (problem_idx, response_idx) 
        for task, result in zip(llm_judge_tasks, llm_results):
            key = (task.problem_idx, task.response_idx)
            llm_judge_results[key] = result
        
        print("✅ LLM judge processing complete.")
    
    # Phase 3: Combine results
    print("🔗 Phase 3: Combining results...")
    correct_count = 0
    total_responses = 0
    
    for problem_idx, (problem_result, problem_math_results) in enumerate(zip(processed_results, math_verify_results)):
        for response_idx, math_result in enumerate(problem_math_results):
            total_responses += 1
            
            # Check if we have LLM judge result for this response
            key = (problem_idx, response_idx)
            if key in llm_judge_results:
                # Use LLM judge result
                llm_result = llm_judge_results[key]
                processed_response = {
                    "full_response": math_result["full_response"],
                    "extracted_answer": llm_result.extracted_answer,
                    "is_correct": llm_result.is_correct,
                    "extraction_method": llm_result.extraction_method,
                }
                if llm_result.judge_explanation:
                    processed_response["judge_explanation"] = llm_result.judge_explanation
                
                if llm_result.is_correct:
                    correct_count += 1
            else:
                # Use math verification result - both success and failure are "math_verify"
                processed_response = {
                    "full_response": math_result["full_response"],
                    "extracted_answer": math_result["extracted_answer"],
                    "is_correct": math_result["is_correct"],
                    "extraction_method": "math_verify",  # Both regex+math_verify success and failure
                }
                
                if math_result["is_correct"]:
                    correct_count += 1
            
            problem_result["responses"].append(processed_response)
    
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
    
    print(f"🎉 Processing complete! {correct_count}/{total_responses} correct ({correct_count/total_responses:.1%})")
    return processed_results, correct_count, total_responses

def process_results_file(
    input_file: str, 
    output_file: Optional[str] = None, 
    use_llm_judge: bool = True,
    batch_size: int = 20,
    max_workers: int = 10
) -> Tuple[List[Dict], int, int]:
    """
    Process a results file to extract and verify answers
    
    Args:
        input_file: Path to the input JSONL file with model responses
        output_file: Path to save the processed results (None to skip saving)
        use_llm_judge: Whether to use LLM judge for fallback (default: True)
        batch_size: Number of LLM judge tasks to process in each batch (default: 20)
        max_workers: Maximum number of concurrent API calls per batch (default: 10)
        
    Returns:
        Tuple containing (processed_results, correct_count, total_responses)
    """
    if use_llm_judge:
        # Use async version with LLM judge support
        return asyncio.run(process_results_file_async(input_file, output_file, batch_size, max_workers))
    
    # Original synchronous version (kept for backward compatibility)
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
            
            # Add the processed response (legacy format without extraction method)
            processed_response = {
                "full_response": response["full_response"],
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "extraction_method": "math_verify"  # Mark as regex for legacy compatibility
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

def analyze_extraction_methods(results: List[Dict]) -> Dict[str, Dict[str, Union[int, float]]]:
    """Analyze results by extraction method"""
    method_metrics = {}
    
    for result in results:
        for response in result["responses"]:
            method = response.get("extraction_method", "unknown")
            
            if method not in method_metrics:
                method_metrics[method] = {
                    "total_responses": 0,
                    "correct_responses": 0
                }
            
            method_metrics[method]["total_responses"] += 1
            if response["is_correct"]:
                method_metrics[method]["correct_responses"] += 1
    
    # Calculate accuracy for each method
    for method, metrics in method_metrics.items():
        metrics["accuracy"] = (
            metrics["correct_responses"] / metrics["total_responses"] 
            if metrics["total_responses"] > 0 else 0
        )
    
    return method_metrics

def process_directory(input_dir: str, output_dir: Optional[str] = None, use_llm_judge: bool = True, batch_size: int = 20, max_workers: int = 10) -> Dict[str, Dict]:
    """
    Process all result files in a directory
    
    Args:
        input_dir: Directory containing result JSONL files
        output_dir: Directory to save processed results (None to skip saving)
        use_llm_judge: Whether to use LLM judge for fallback (default: True)
        batch_size: Number of LLM judge tasks to process in each batch (default: 20)
        max_workers: Maximum number of concurrent API calls per batch (default: 10)
        
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
        processed_results, correct, total = process_results_file(input_path, output_path, use_llm_judge, batch_size, max_workers)
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
        
        # Analyze by category
        category_metrics = analyze_by_category(processed_results)
        
        # Analyze by extraction method
        extraction_metrics = analyze_extraction_methods(processed_results)
        
        # Store metrics
        metrics[model_name] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "by_category": category_metrics,
            "by_extraction_method": extraction_metrics
        }
        
        print(f"Processed {filename}: Accuracy {accuracy:.2%} ({correct}/{total})")
        
        # Show extraction method breakdown
        if use_llm_judge:
            print("  Extraction method breakdown:")
            for method, method_metrics in extraction_metrics.items():
                print(f"    {method}: {method_metrics['accuracy']:.2%} ({method_metrics['correct_responses']}/{method_metrics['total_responses']})")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and verify answers from model responses")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file or directory to process")
    parser.add_argument("--output", "-o", help="Output file or directory for processed results (defaults to 'extracted_answers' directory)")
    parser.add_argument("--by-category", action="store_true", help="Show results broken down by category")
    parser.add_argument("--no-llm-judge", action="store_true", help="Disable LLM judge fallback (use only regex extraction)")
    parser.add_argument("--show-extraction-methods", action="store_true", help="Show results broken down by extraction method")
    parser.add_argument("--batch-size", type=int, default=200, help="Number of LLM judge tasks to process in each batch (default: 20)")
    parser.add_argument("--max-workers", type=int, default=40, help="Maximum number of concurrent API calls per batch (default: 40)")
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    use_llm_judge = not args.no_llm_judge
    batch_size = args.batch_size
    max_workers = args.max_workers
    
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
    
    if use_llm_judge:
        print("🧠 LLM Judge enabled: Using Gemini 2.0 Flash for answer extraction fallback")
        print(f"⚙️  Parallelization: batch_size={batch_size}, max_workers={max_workers}")
    else:
        print("📝 LLM Judge disabled: Using only regex-based extraction")
    
    if os.path.isdir(input_path):
        # Process a directory of result files
        metrics = process_directory(input_path, output_path, use_llm_judge, batch_size, max_workers)
        
        # Print summary
        print("\nSummary of all models:")
        for model, result in metrics.items():
            print(f"{model}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
            
            # Show category breakdown if requested
            if args.by_category and "by_category" in result:
                print("  Results by category:")
                for category, cat_metrics in result["by_category"].items():
                    print(f"    {category}: {cat_metrics['accuracy']:.2%} ({cat_metrics['correct_responses']}/{cat_metrics['total_responses']})")
            
            # Show extraction method breakdown if requested
            if args.show_extraction_methods and "by_extraction_method" in result:
                print("  Results by extraction method:")
                for method, method_metrics in result["by_extraction_method"].items():
                    print(f"    {method}: {method_metrics['accuracy']:.2%} ({method_metrics['correct_responses']}/{method_metrics['total_responses']})")
    else:
        # Process a single file
        processed_results, correct, total = process_results_file(input_path, output_path, use_llm_judge, batch_size, max_workers)
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
        
        # Show extraction method breakdown if requested
        if args.show_extraction_methods or use_llm_judge:
            extraction_metrics = analyze_extraction_methods(processed_results)
            print("\nResults by extraction method:")
            
            # Sort methods by total responses (descending)
            sorted_methods = sorted(
                extraction_metrics.items(),
                key=lambda x: x[1]['total_responses'],
                reverse=True
            )
            
            for method, metrics in sorted_methods:
                print(f"  {method}: {metrics['accuracy']:.2%} ({metrics['correct_responses']}/{metrics['total_responses']})")
        
        if output_path:
            print(f"Processed results saved to {output_path}") 