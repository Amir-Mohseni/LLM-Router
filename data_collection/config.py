# Configuration settings for model inference

# Model settings
DEFAULT_MODEL = "google/gemma-3-4b-it"

# Dataset settings
DATASET_NAME = "HuggingFaceH4/MATH-500"
DATASET_SPLIT = "test"
NUM_PROBLEMS = 5  # Number of problems to test

# Generation settings
K_RESPONSES = 5  # Number of responses per question
TEMPERATURE = 0.7  # Sampling temperature for diversity
MAX_TOKENS = 2048  # Maximum tokens per response

# Output settings
OUTPUT_DIR = "data_collection/inference_results" 