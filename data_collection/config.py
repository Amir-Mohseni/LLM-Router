# Configuration settings for model inference

# Model settings
DEFAULT_MODEL = "google/gemma-3-4b-it"

# API settings
API_MODE = "local"  # "local" or "remote"

# Local API settings (for vLLM server)
API_BASE = "http://localhost:8000/v1"  # Base URL 
API_KEY = "EMPTY"  # API key for local server (usually not needed)
MODEL_NAME = "gemma-3-4b-it"

# vLLM server settings
VLLM_HOST = "0.0.0.0"  # Host to bind the server to
VLLM_PORT = 8000  # Port to run the server on
VLLM_MODEL_IMPL = "transformers"  # Model implementation (transformers or vllm)
VLLM_MAX_MODEL_LEN = 8192  # Maximum model context length
# Note: For gemma models, we need to disable multimodal preprocessing with --disable-mm-preprocessor-cache
# to avoid the "Cannot find `mm_limits` for model" error

# Dataset settings
DATASET_NAME = "HPC-Boys/MATH_500_MMLU_Pro"
DATASET_SPLIT = "train"
NUM_PROBLEMS = 5  # Number of problems to test

# Generation settings
K_RESPONSES = 5  # Number of responses per question
TEMPERATURE = 0.7  # Sampling temperature for diversity
MAX_TOKENS = 2048  # Maximum tokens per response (for non-reasoning models -> 2048 and for reasoning models -> 4096)

# Output settings
OUTPUT_DIR = "data_collection/inference_results"
CUSTOM_OUTPUT_FILENAME = None  # Custom filename for results (None = auto-generate)
MAX_ATTEMPTS_PER_QUESTION = 3  # Maximum number of attempts per question before giving up

# Batching settings
PROMPT_BATCH_SIZE = 8  # Number of prompts to process at once in vLLM
PROBLEM_BATCH_SIZE = 5  # Number of problems to process in each batch for checkpointing 