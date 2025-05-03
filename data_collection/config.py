# Configuration settings for model inference

#
# Model & API settings
#
MODEL_NAME = "google/gemma-3-27b-it"  # The model identifier to use for inference
API_MODE = "local"                   # "local" (vLLM server) or "remote" (OpenAI API)
API_BASE = "http://localhost:8000/v1"  # Base URL for API
API_KEY_NAME = "VLLM_API_KEY"

#MODEL_NAME = "gemini-2.0-flash"
#API_MODE = "remote"
#API_KEY_NAME = "GOOGLE_API_KEY"
#API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

# vLLM server settings (only used when starting the server, not by run_inference.py)
VLLM_HOST = "0.0.0.0"
VLLM_PORT = 8000
VLLM_MODEL_IMPL = "transformers"
VLLM_MAX_MODEL_LEN = 8192
VLLM_TENSOR_PARALLEL_SIZE = 1  # Number of GPUs to use for tensor parallelism (default: 1)
# Note: For gemma models, we need to disable multimodal preprocessing with --disable-mm-preprocessor-cache

#
# Dataset settings
#
DATASET_NAME = "HPC-Boys/MATH_500_MMLU_Pro"
DATASET_SPLIT = "train"
NUM_PROBLEMS = 5  # Number of problems to test, or 'all' for entire dataset

#
# Generation settings
#
K_RESPONSES = 5       # Number of responses per question
TEMPERATURE = 0.7     # Sampling temperature for diversity
MAX_TOKENS = 2048     # Maximum tokens per response

# Advanced generation settings
GENERATION_KWARGS = {
    "stop": ["<end_of_turn>", "<|end_of_turn|>", "<|im_end|>"],
    "logprobs": None,  # Don't return token logprobs to save bandwidth
    "echo": False,     # Don't echo the prompt in the response
}

#
# Processing settings
#
# Batch processing controls
PROBLEM_BATCH_SIZE = 100  # Problems per batch for checkpointing
MAX_CONCURRENT_REQUESTS = 100  # Maximum concurrent API requests

#
# Output settings
#
OUTPUT_DIR = "data_collection/inference_results" 