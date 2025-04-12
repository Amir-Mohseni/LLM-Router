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
- `run_tests.py`: Wrapper script to test multiple models sequentially

## Usage

### Single Model Testing

To test a single model:

```bash
python -m data_collection.run_inference --model "google/gemma-3-1b-it" --num_problems 5 --k_responses 3
```

Options:
- `--model`: Model ID to use for inference (default: "google/gemma-3-1b-it")
- `--num_problems`: Number of problems to test (default: 5)
- `--k_responses`: Number of responses to generate per problem (default: 5)
- `--temperature`: Sampling temperature (default: 0.7)
- `--output_dir`: Directory to save results (default: "data_collection/inference_results")

This will run each model in sequence and save the results to separate files.

## Dataset

The scripts are currently configured to use the HuggingFaceH4/MATH-500 dataset's test split, which contains mathematical problems with solutions and answers.

## Output Format

Results are saved in JSON files with the following structure:

```json
[
  {
    "problem": "Problem text",
    "solution": "Reference solution",
    "correct_answer": "Reference answer",
    "responses": [
      {
        "full_response": "Model's full response",
        "extracted_answer": "Answer extracted from <answer> tags"
      },
      ...
    ]
  },
  ...
]
```

## Customization

You can modify the following files to customize the behavior:

- `config.py`: Change default models, dataset, number of problems, etc.
- `prompts.py`: Modify the prompt templates used for generation 