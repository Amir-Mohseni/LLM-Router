# Data Collection for LLM Router

This directory contains tools for collecting and analyzing model responses to problems.

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have a valid Hugging Face API token to access the models.

## File Structure

- `config.py`: Configuration settings for datasets, models, and generation parameters
- `prompts.py`: Prompt templates for generating responses from models
- `run_inference.py`: Main script for running inference on a single model
- `answer_extraction.py`: Script for extracting and evaluating answers from the model outputs

## Workflow

The typical workflow is a two-step process:

1. **Generate responses**: Use `run_inference.py` to generate model responses 
2. **Extract answers**: Use `answer_extraction.py` to extract answers and evaluate them

This separation allows for more efficient processing and makes it easier to experiment with different answer extraction methods without re-running the models.

## Supported Question Types

The system supports two types of questions:

1. **Multiple Choice Questions (MCQs)**: Questions with a set of predefined choices
2. **Open-ended Questions**: Questions requiring a free-form answer

Each question type uses a different prompt template and answer extraction method.

## Dataset Format

The dataset is expected to have the following fields:

- `unique_id`: String that uniquely identifies each question
- `question`: The problem text
- `choices`: List of choices for MCQ questions, or null for non-MCQ questions
- `choice_index_correct`: Index of the correct choice (0-based) for MCQ questions, or null
- `explanation_correct`: Explanation of the correct answer
- `answer_correct`: The correct answer text
- `category`: Category or subject area of the question

## Usage

### Step 1: Generate Model Responses

```bash
python -m data_collection.run_inference --model "google/gemma-3-1b-it" --num_problems 5 --k_responses 3
```

Options:
- `--model`: Model ID to use for inference (default: "google/gemma-3-1b-it")
- `--num_problems`: Number of problems to test (default: 5)
- `--k_responses`: Number of responses to generate per problem (default: 5)
- `--temperature`: Sampling temperature (default: 0.7)
- `--output_dir`: Directory to save results (default: "inference_results")

### Step 2: Extract and Evaluate Answers

Process a single result file:
```bash
python -m data_collection.answer_extraction --input inference_results/google_gemma-3-1b-it_5problems_3k_1234567890.json
```

Process all result files in a directory:
```bash
python -m data_collection.answer_extraction --input inference_results --output processed_results
```

Options:
- `--input`, `-i`: Input file or directory to process (required)
- `--output`, `-o`: Output file or directory for processed results (optional)
- `--by-category`: Show results broken down by category

## Output Format

### Inference Results Format
```json
[
  {
    "unique_id": "question123",
    "problem": "Problem text",
    "is_mcq": true,
    "choices": ["Option A", "Option B", "Option C", "Option D"],
    "choice_index_correct": 2,
    "explanation_correct": "Explanation of the correct answer",
    "answer_correct": "C",
    "category": "Mathematics",
    "responses": [
      {
        "full_response": "Model's full response"
      }
    ]
  }
]
```

### Processed Results Format
```json
[
  {
    "unique_id": "question123",
    "problem": "Problem text",
    "is_mcq": true,
    "choices": ["Option A", "Option B", "Option C", "Option D"],
    "choice_index_correct": 2,
    "explanation_correct": "Explanation of the correct answer",
    "answer_correct": "C",
    "category": "Mathematics",
    "responses": [
      {
        "full_response": "Model's full response",
        "extracted_answer": "C",
        "is_correct": true
      }
    ]
  }
]
```

## Customization

You can modify the following files to customize the behavior:

- `config.py`: Change default models, dataset, number of problems, etc.
- `prompts.py`: Modify the prompt templates for both MCQ and non-MCQ questions
- `answer_extraction.py`: Customize answer extraction and evaluation methods 