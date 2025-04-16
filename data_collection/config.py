# Configuration settings for model inference

# Model settings
DEFAULT_MODEL = "google/gemma-3-4b-it"

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