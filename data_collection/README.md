# Data Collection for LLM Router

This directory contains tools for collecting and analyzing model responses to problems.

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure API settings in `config.py` if you want to use remote APIs.

3. Environment Variables:
   ```bash
   # Copy the example environment file
   cp env.example .env
   
   # Edit the .env file with your API keys and preferred settings
   nano .env
   ```

## File Structure

- `config.py`: Configuration settings for datasets, models, generation parameters, API settings and batching
- `prompts.py`: Prompt templates for generating responses from models
- `run_inference.py`: Main script for running inference on a single model
- `answer_extraction.py`: Script for extracting and evaluating answers from the model outputs

## Workflow

The typical workflow is a two-step process:

1. **Generate responses**: Use `run_inference.py` to generate model responses 
2. **Extract answers**: Use `answer_extraction.py` to extract answers and evaluate them

This separation allows for more efficient processing and makes it easier to experiment with different answer extraction methods without re-running the models.

## Features

- **Asynchronous processing**: Efficient parallel API calls for maximum throughput
- **Unified API interface**: Single API approach for both local (vLLM) and remote (OpenAI) services
- **Optimized parallelization**: Uses batched requests with n=k parameter for efficient response generation
- **Batched processing**: Problems are processed in batches, with results saved after each batch
- **Checkpointing**: Progress is saved periodically, allowing recovery from interruptions
- **JSONL format**: Results are stored in JSONL format (one JSON object per line) for efficient storage and processing
- **Multiple responses**: For each problem, k sample responses are generated and preserved regardless of correctness
- **Flexible dataset handling**: Process either a specific number of problems or the entire dataset

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

### API Modes

The script supports two API modes which share the same configuration format:

1. **Local mode**: Uses a local vLLM server (requires starting the server separately)
2. **Remote mode**: Uses remote APIs like OpenAI (requires API key)

You can configure the API settings in `config.py` or use command line arguments.

### Local Mode Setup (Default)

Before running the inference script in local mode, start the vLLM server:

```bash
# Start the vLLM server with your model
vllm serve meta-llama/Llama-3.2-1B
```

The server will be available at the URL specified in config.py (default: http://localhost:8000/v1).

### Generate Model Responses

```bash
# Basic usage with default settings
python -m data_collection.run_inference

# Local API mode with custom model
python -m data_collection.run_inference --model "meta-llama/Llama-3.2-1B"

# Remote API mode
python -m data_collection.run_inference --api_mode remote --model "gpt-3.5-turbo-instruct"

# Customize batch size and responses per problem
python -m data_collection.run_inference --batch_size 10 --k_responses 3

# Control parallel processing (higher values = more throughput but higher resource usage)
python -m data_collection.run_inference --max_concurrent 20

# Custom API settings
python -m data_collection.run_inference --api_base "http://localhost:9000/v1" --api_key "your-api-key-here"
```

Options:
- `--model`: Model ID to use for inference (default from config.py)
- `--api_mode`: API mode to use (local or remote, default from config.py)
- `--api_base`: Base URL for the API (default from config.py)
- `--api_key`: API key (default from config.py)
- `--num_problems`: Number of problems to test or 'all' for entire dataset (default from config.py)
- `--k_responses`: Number of responses to generate per problem (default from config.py)
- `--temperature`: Sampling temperature (default from config.py)
- `--max_tokens`: Maximum number of tokens for generation (default from config.py)
- `--batch_size`: Number of problems to process in each batch (default from config.py)
- `--output_dir`: Directory to save results (default from config.py)
- `--max_concurrent`: Maximum number of concurrent API requests (default: 10)

### Extract and Evaluate Answers

Process a single result file:
```bash
python -m data_collection.answer_extraction --input inference_results/model_name_5problems_3k_1234567890.jsonl
```

Process all result files in a directory:
```bash
python -m data_collection.answer_extraction --input inference_results --output processed_results
```

Options:
- `--input`, `-i`: Input JSONL file or directory to process (required)
- `--output`, `-o`: Output file or directory for processed results (optional)
- `--by-category`: Show results broken down by category

## Output Format

### Inference Results Format (JSONL)
```json
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
```

### Processed Results Format (JSONL)
```json
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
```

## Performance Optimization

The code uses asynchronous processing to make multiple API calls concurrently, greatly improving throughput. You can adjust the level of parallelism with the `--max_concurrent` parameter. Higher values will process more requests simultaneously but will require more resources. The optimal value depends on your hardware, network capacity, and the API endpoint's capabilities.

## Customization

You can modify the following files to customize the behavior:

- `config.py`: Change default models, API settings, dataset, batching parameters, etc.
- `prompts.py`: Modify the prompt templates for both MCQ and non-MCQ questions
- `answer_extraction.py`: Customize answer extraction and evaluation methods 