# Data Collection Module

This module provides tools for collecting language model responses on math problems.

## Components

1. **LLM.py**: Unified LLM interface supporting both local and remote models
   - `BaseLLM`: Abstract base class with common functionality
   - `RemoteLLM`: Implementation for API-based models
   - `LocalLLM`: Implementation for local vLLM models
   - `create_llm()`: Factory function to instantiate the appropriate LLM type
2. **run_inference.py**: Script to run inference on math datasets
3. **serve_llm.py**: Script to start a local vLLM server
4. **dataset.py**: Functions to load datasets from Hugging Face
5. **config.py**: Configuration settings for model, API, dataset, etc.

## Usage Guide

### Polymorphic LLM Architecture

The system uses a polymorphic design to handle both local and remote LLMs through a unified interface:

- **BaseLLM**: Abstract base class that defines a common interface for all LLM types
- **RemoteLLM**: Implementation for remote API-based models (OpenAI, Google, etc.)
- **LocalLLM**: Implementation for local models running on a vLLM server
- **Parameter Compatibility**: Automatic filtering of incompatible parameters between model types

This architecture provides:
1. Clean separation between model types
2. Unified interface for all LLM operations
3. Proper handling of parameter incompatibilities
4. Appropriate error reporting and logging

### Running Inference

The `run_inference.py` script can run inference using either remote API models or local vLLM models.

#### Basic Usage:

```bash
# Run inference with default settings
python -m data_collection.run_inference --output_file results.jsonl

# Run with custom model and parameters
python -m data_collection.run_inference \
  --model gemma-3-27b-it \
  --api_mode remote \
  --temperature 0.7 \
  --max_tokens 4096 \
  --k_responses 5 \
  --output_file custom_results.jsonl
```

#### Using Local Mode (vLLM):

To use a local model with vLLM:

1. First, start the vLLM server:
   ```bash
   python -m data_collection.serve_llm --model meta-llama/Llama-2-13b-chat-hf
   ```

2. Then run inference in a separate terminal:
   ```bash
   python -m data_collection.run_inference \
     --api_mode local \
     --api_base http://localhost:8000/v1 \
     --model meta-llama/Llama-2-13b-chat-hf \
     --output_file local_results.jsonl
   ```

#### Using Remote Mode:

For remote API services like OpenAI or Google:

```bash
# Set your API key in environment variables
export GOOGLE_API_KEY="your-api-key"

# Run inference
python -m data_collection.run_inference \
  --api_mode remote \
  --api_base https://generativelanguage.googleapis.com/v1beta/openai/ \
  --model gemma-3-27b-it \
  --output_file google_results.jsonl
```

### Parameter Compatibility

The system automatically filters generation parameters based on the API mode:

- **Remote Models**: Only the parameters supported by remote APIs (`temperature`, `max_tokens`, etc.)
- **Local Models**: All vLLM parameters (`stop`, `temperature`, `max_tokens`, `repetition_penalty`, etc.)

This handling happens automatically in `run_inference.py` without requiring manual parameter management.

### Configuration

You can configure default values in `config.py`:

```python
# Set your preferred default model and API mode
MODEL_NAME = "gemma-3-27b-it"
API_MODE = "remote"  # or "local"
API_KEY_NAME = "GOOGLE_API_KEY"  # env var name for API key
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Generation parameters (will be filtered based on API mode)
GENERATION_KWARGS = {
    "temperature": 0.7,
    "max_tokens": 2048,
    "echo": True,              # Only used for local models
    "repetition_penalty": 1.2, # Only used for local models
    "stop": ["<end_of_turn>"]  # Only used for local models
}

# For local vLLM server
VLLM_HOST = "0.0.0.0"
VLLM_PORT = 8000
VLLM_MAX_MODEL_LEN = 8192
VLLM_TENSOR_PARALLEL_SIZE = 2  # Number of GPUs to use for tensor parallelism
```

### Advanced Usage

#### Processing Large Datasets:

For large datasets, use batch processing:

```bash
python -m data_collection.run_inference \
  --num_problems all \
  --batch_size 20 \
  --max_concurrent 5 \
  --output_file large_dataset.jsonl
```

#### Retrying Failed Queries:

If some queries fail, you can retry just those:

```bash
python -m data_collection.run_inference \
  --retry_failed \
  --output_file previous_results.jsonl
```

#### Extracting Answers:

After collecting responses, extract structured answers:

```bash
python -m data_collection.answer_extraction --input results.jsonl
```

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure API settings in `config.py` if you want to use remote APIs.

3. Environment Variables:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit the .env file with your API keys and preferred settings
   nano .env
   ```

## File Structure

- `config.py`: Configuration settings for datasets, models, generation parameters, API settings and batching
- `prompts.py`: Prompt templates for generating responses from models
- `LLM.py`: Polymorphic LLM implementation with support for remote and local models
- `run_inference.py`: Main script for running inference on a single model
- `answer_extraction.py`: Script for extracting and evaluating answers from the model outputs
- `serve_llm.py`: Script to start and manage a local vLLM server

## Workflow

The typical workflow is a two-step process:

1. **Generate responses**: Use `run_inference.py` to generate model responses 
2. **Extract answers**: Use `answer_extraction.py` to extract answers and evaluate them

This separation allows for more efficient processing and makes it easier to experiment with different answer extraction methods without re-running the models.

## Features

- **Polymorphic LLM design**: Unified interface for all LLM types with proper inheritance
- **Parameter compatibility**: Automatic filtering of incompatible parameters between model types
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

The script supports two API modes which share the same interface:

1. **Local mode**: Uses a local vLLM server (requires starting the server separately)
2. **Remote mode**: Uses remote APIs like OpenAI (requires API key)

You can configure the API settings in `config.py` or use command line arguments.

### Local Mode Setup (Default)

Before running the inference script in local mode, you can start the vLLM server using either the command line or the provided script:

#### Option 1: Direct command line
```bash
# Start the vLLM server with your model
vllm serve meta-llama/Llama-3.2-1B --model-impl transformers --max_model_len 8192 --tensor-parallel-size 2
```

#### Option 2: Using the serve_llm.py script (recommended)
```bash
# Start the server with default settings from config.py (runs in background automatically)
python -m data_collection.serve_llm

# Or customize the model and parameters at runtime
python -m data_collection.serve_llm --model "meta-llama/Llama-3.2-1B" --max-model-len 4096

# Increase GPU utilization (default is 95%)
python -m data_collection.serve_llm --gpu-util 0.98

# Start a server for Gemma models with necessary flags
python -m data_collection.serve_llm --model "google/gemma-3-4b-it"

# Use tensor parallelism to distribute model across multiple GPUs
python -m data_collection.serve_llm --model "meta-llama/Llama-3-70b" --tensor-parallel-size 4
```

#### Tensor Parallelism

For large models that don't fit in a single GPU's memory, you can use tensor parallelism to distribute the model across multiple GPUs:

```bash
# Distribute a 70B model across 4 GPUs
python -m data_collection.serve_llm --model "meta-llama/Llama-3-70b" --tensor-parallel-size 4
```

The tensor parallel size can be configured in three ways:
1. In `config.py` by setting `VLLM_TENSOR_PARALLEL_SIZE`
2. Via command line with `--tensor-parallel-size`
3. In Docker Compose by setting the `VLLM_TENSOR_PARALLEL_SIZE` environment variable

Note that tensor parallelism requires:
- Multiple CUDA-capable GPUs with sufficient memory
- NVIDIA GPU driver version 470.42.01 or higher
- CUDA 11.4 or higher

Performance benefits:
- Load larger models than would fit on a single GPU
- Improved inference speed for large models
- Better utilization of multi-GPU systems

The server will use the settings defined in `config.py` by default. You can modify these settings in the config file or override them with command-line arguments as shown above. The server always runs in the background and will display its Process ID (PID) so you can stop it later if needed.

#### Special Token Handling

The system includes optimizations to prevent issues with special tokens like `<end_of_turn>` being repeatedly generated:

1. **Stop Sequences**: Generation automatically stops when encountering special tokens like `<end_of_turn>` or `<|im_end|>`.
2. **Eager Execution**: The vLLM server uses eager execution mode for better token handling.

These optimizations ensure clean responses without repetitive end tokens or other artifacts.

#### Google Colab Support
The server script is designed to work seamlessly in Google Colab. When running in Colab with the `!` command prefix, the script will:

1. Start the server in the background
2. Wait until the server is fully online and responding
3. Return control to the next cell after confirmation

Example usage in Colab:
```python
# Cell 1: Start the server with high GPU utilization
!python data_collection/serve_llm.py --model "google/gemma-3-4b-it" --gpu-util 0.98

# Cell 2: Run inference (will execute once the server is ready)
!python data_collection/run_inference.py --model "google/gemma-3-4b-it"
```

The server will be available at the URL specified in config.py (default: http://localhost:8000/v1).

### Generate Model Responses

```bash
# Basic usage with default settings
python -m data_collection.run_inference

# Local API mode with custom model
python -m data_collection.run_inference --model "meta-llama/Llama-3.2-1B"

# Remote API mode
python -m data_collection.run_inference --api_mode remote --model "gemini-2.0-flash"

# Customize batch size and responses per problem
python -m data_collection.run_inference --batch_size 10 --k_responses 3

# Control parallel processing (higher values = more throughput but higher resource usage)
python -m data_collection.run_inference --max_concurrent 20

# Use a custom output filename (useful for consistent result storage across runs)
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

### Continuous Inference and Error Handling

The inference system is designed for robustness and continuation:

1. **Consistent File Naming**: By default, results are saved with a consistent filename based on the model and dataset name, allowing automatic continuation across runs.

2. **Attempt Tracking**: The system tracks the number of attempts made for each question. Failed questions will be retried in subsequent runs until they reach the maximum number of attempts.

3. **Connection Error Handling**: If all responses for a question fail due to server connection issues, the system will not count it as an attempt, ensuring that temporary outages don't waste your attempt quota.

4. **Progress Tracking**: The system automatically tracks which questions have been processed and continues from where it left off in case of interruptions.

Example workflow:
```bash
# 1. Start the vLLM server (runs in background automatically)
python -m data_collection.serve_llm

# 2. Run inference (first batch)
python -m data_collection.run_inference

# 3. If the process is interrupted or some questions fail, just run it again
# to continue from where it left off
python -m data_collection.run_inference

# 4. When done, analyze the results
python -m data_collection.answer_extraction --input data_collection/inference_results/google_gemma-3-4b-it_MATH_500_MMLU_Pro.jsonl
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