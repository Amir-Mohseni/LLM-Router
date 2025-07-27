# Configuration settings for model inference

#
# Model & API settings
#
#MODEL_NAME = "google/gemma-3-27b-it"  # The model identifier to use for inference
#API_MODE = "local"                   # "local" (vLLM server) or "remote" (OpenAI API)
#API_BASE = "http://localhost:8000/v1"  # Base URL for API
#API_KEY_NAME = "VLLM_API_KEY"

#MODEL_NAME = "gemini-2.0-flash"
#API_MODE = "remote"
#API_KEY_NAME = "GEMINI_API_KEY"
#API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

MODEL_NAME = "qwen/qwen3-8b:free"
API_MODE = "remote"
API_KEY_NAME = "OPENROUTER_API_KEY"
API_BASE = "https://openrouter.ai/api/v1"
SYSTEM_PROMPT = "You are a helpful assistant. /no_think"

# System prompt settings - used to control model behavior
# Set to None for no system prompt, or specify a string to apply to all models
#SYSTEM_PROMPT = None  # Default: no system prompt

# vLLM server settings (only used when starting the server, not by run_inference.py)
VLLM_HOST = "0.0.0.0"
VLLM_PORT = 8000
VLLM_MODEL_IMPL = "transformers"
VLLM_MAX_MODEL_LEN = 8192
VLLM_TENSOR_PARALLEL_SIZE = 1  # Number of GPUs to use for tensor parallelism (default: 1)
VLLM_ENABLE_EXPERT_PARALLEL = False  # Enable expert parallelism for MoE models
VLLM_KV_CACHE_DTYPE = "auto"  # Data type for KV cache: "auto", "fp8", "fp16", "bf16", etc.
# Note: For gemma models, we need to disable multimodal preprocessing with --disable-mm-preprocessor-cache

#
# Dataset settings
#
#DATASET_NAME = "HPC-Boys/MATH_500_MMLU_Pro"
DATASET_NAME = "HPC-Boys/AIME_1983_2024"
DATASET_SPLIT = "train"
NUM_PROBLEMS = 'all'  # Number of problems to test, or 'all' for entire dataset

#
# Generation settings
#
K_RESPONSES = 3       # Number of responses per question
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