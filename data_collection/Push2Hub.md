# Pushing Model Results to HuggingFace Hub

This document explains how to use the `push_to_hub.py` script to upload extracted model answers to the HuggingFace Hub under the HPC-Boys organization.

## Requirements

Make sure you have the necessary libraries installed:

```bash
pip install datasets huggingface_hub
```

You'll also need a HuggingFace account and API token with write access to the HPC-Boys organization.

## Basic Usage

```bash
# Run the script with a folder containing train.jsonl, val.jsonl, and test.jsonl files
python -m data_collection.push_to_hub --folder data_collection/extracted_answers/your_model_folder
```

## Example: Uploading Gemini Flash 2.0 Results

```bash
# Upload Gemini Flash 2.0 results
python -m data_collection.push_to_hub --folder data_collection/extracted_answers/gemini_flash_2
```

The script will:
1. Prompt you for a repository name (default: "Gemini Flash 2")
2. Ask for your HuggingFace API token (if not already logged in)
3. Show a summary of what will be uploaded
4. Ask for confirmation before proceeding

After uploading, the dataset will be available at:
`https://huggingface.co/datasets/HPC-Boys/gemini-flash-2`

## Example: Uploading with Custom Name and Privacy Setting

```bash
# Upload with custom repository name and make it private
python -m data_collection.push_to_hub \
  --folder data_collection/extracted_answers/gemini_flash_2 \
  --repo-name gemini-flash-2-math-results \
  --private \
  --token YOUR_HF_TOKEN
```

## All Available Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--folder` | `-f` | Folder containing JSONL files | (required) |
| `--repo-name` | `-r` | Repository name on HuggingFace | (derived from folder name) |
| `--org` | `-o` | Organization name | "HPC-Boys" |
| `--private` | - | Make the dataset private | False (public) |
| `--token` | `-t` | HuggingFace API token | (will prompt if not provided) |

## Expected Folder Structure

The script expects the following structure in your folder:

```
your_model_folder/
├── train.jsonl   # Training split
├── val.jsonl     # Validation split (note: this is "val" not "validation")
└── test.jsonl    # Test split
```

Each JSONL file should contain one JSON object per line, with each object representing a single example.

## Data Processing

The script processes the input data to make it more analysis-friendly. For each question:

1. **Original Question Data**: All original fields are preserved (unique_id, problem, choices, etc.)

2. **Response Columns**: Each of the model's responses is added as a separate column
   - `response_1`, `response_2`, ...: Full text of each response
   - `extracted_answer_1`, `extracted_answer_2`, ...: Extracted answers (if available)
   - `is_correct_1`, `is_correct_2`, ...: Boolean indicating correctness of each response

3. **Summary Statistics**: Additional columns to track performance
   - `total_responses`: Total number of responses for this question
   - `correct_responses`: Number of correct responses
   - `accuracy`: Ratio of correct responses (correct_responses / total_responses)

This format makes it easy to analyze model performance question by question.

## Advanced Options for Running Large Models

When generating results with large models (like Mixtral or Llama-3-70B), you can use additional options to optimize inference:

```bash
# Run a large mixture-of-experts model with optimizations
python -m data_collection.serve_llm \
  --model mistralai/Mixtral-8x22B-v0.1 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8

# Automatically use all available GPUs
python -m data_collection.serve_llm \
  --model meta-llama/Llama-3-70b \
  --use-all-gpus

# Run inference after server is ready
python -m data_collection.run_inference \
  --api_mode local \
  --model mistralai/Mixtral-8x22B-v0.1 \
  --output_file mixtral_results.jsonl
```

### Memory Optimization Options

- **Automatic GPU Detection**: The system automatically detects available CUDA devices
  - `--use-all-gpus`: Automatically use all available GPUs
  - The script warns if you request more GPUs than available

- **Tensor Parallelism**: Distributes model weight tensors across multiple GPUs
  - `--tensor-parallel-size 4`: Splits the model across 4 GPUs

- **Expert Parallelism**: For mixture-of-experts (MoE) models, enables parallel execution of expert modules
  - `--enable-expert-parallel`: Enables expert parallelism for models like Mixtral

- **KV Cache Data Type**: Controls memory usage by setting key-value cache precision
  - `--kv-cache-dtype fp8`: Uses FP8 format for KV cache to reduce memory usage (options: auto, fp8, fp16, bf16)

These options allow you to run larger models that wouldn't otherwise fit on your hardware.

## Dataset on HuggingFace

After uploading, you can access your dataset using the datasets library:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("HPC-Boys/gemini-flash-2")

# Access a specific split
training_data = dataset["train"]
validation_data = dataset["validation"]  # Note: "validation" not "val" in the loaded dataset
test_data = dataset["test"]

# Print the first example with its stats
print(f"Question: {training_data[0]['problem']}")
print(f"Total responses: {training_data[0]['total_responses']}")
print(f"Correct responses: {training_data[0]['correct_responses']}")
print(f"Accuracy: {training_data[0]['accuracy']:.2f}")

# Access individual responses
print(f"Response 1: {training_data[0]['response_1']}")
print(f"Correct? {training_data[0]['is_correct_1']}")
```

## Sample Analysis

The uploaded dataset can be used for various analyses:

```python
# Calculate average accuracy across all questions
average_accuracy = sum(example['accuracy'] for example in dataset["test"]) / len(dataset["test"])
print(f"Average accuracy: {average_accuracy:.2f}")

# Find questions with 100% accuracy
perfect_questions = [ex for ex in dataset["test"] if ex['accuracy'] == 1.0]
print(f"Questions with 100% accuracy: {len(perfect_questions)}")

# Find questions with 0% accuracy
failed_questions = [ex for ex in dataset["test"] if ex['accuracy'] == 0.0]
print(f"Questions with 0% accuracy: {len(failed_questions)}")
``` 