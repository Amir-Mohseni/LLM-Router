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
- **JSONL format**: Results are stored in JSONL format for efficient storage and processing
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

Before running the inference script in local mode, you can start the vLLM server using either the command line or the provided script:

#### Option 1: Direct command line
```bash
# Start the vLLM server with your model
vllm serve meta-llama/Llama-3.2-1B --model-impl transformers --max_model_len 8192
```

#### Option 2: Using the serve_llm.py script (recommended)
```bash
# Start the server with default settings from config.py
python -m data_collection.serve_llm

# Or customize the model and parameters at runtime
python -m data_collection.serve_llm --model "meta-llama/Llama-3.2-1B" --max-model-len 4096 --port 8080
```

The server will use the settings defined in `config.py` by default. You can modify these settings in the config file or override them with command-line arguments as shown above.

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

# Use a custom output filename (useful for resuming interrupted runs)
python -m data_collection.run_inference --output_file "my_custom_results.jsonl"

# Set a maximum number of attempts per question (default: 3)
python -m data_collection.run_inference --max_attempts 5

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
- `--output_file`: Custom filename for results (default: auto-generated)
- `--max_attempts`: Maximum number of attempts per question (default: 3)
- `--max_concurrent`: Maximum number of concurrent API requests (default: 10)

### Handling Failed API Calls

The script tracks the number of attempts made for each question. If an API call fails, the question will be retried in subsequent runs until it reaches the maximum number of attempts specified by `--max_attempts`. This ensures that:

1. Questions with successful responses aren't retried unnecessarily
2. Questions with failed responses get additional attempts
3. Questions that consistently fail after reaching the maximum attempts are skipped

```bash
# Set a higher max attempts for challenging questions
python -m data_collection.run_inference --max_attempts 5 --output_file "difficult_questions.jsonl"
```

### Resume Interrupted Runs

If you need to interrupt a run, you can resume it later by using the same output filename:

```bash
# Start a run with a custom filename
python -m data_collection.run_inference --output_file "my_results.jsonl"

# Later, resume the run with the same filename
python -m data_collection.run_inference --output_file "my_results.jsonl"
```

The script will automatically detect how many problems have already been processed and continue from where it left off.

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