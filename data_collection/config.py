# Configuration settings for model inference

#
# LLM Configuration
#
# LLM_CONFIG = {
#     "model_name": "qwen/qwen3-8b",
#     "base_url": "https://openrouter.ai/api/v1",
#     "api_key_name": "OPENROUTER_API_KEY",
#     "system_prompt": None,  # Set to None for no system prompt, or specify a string
# }

# Alternative LLM configurations (commented out)
LLM_CONFIG = {
    "model_name": "Qwen/Qwen3-8B",
    "base_url": "http://localhost:8000/v1",
    "api_key_name": "VLLM_API_KEY",
    "system_prompt": None,
}

# LLM_CONFIG = {
#     "model_name": "gemini-2.0-flash",
#     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
#     "api_key_name": "GEMINI_API_KEY",
#     "system_prompt": None,
# }

#
# Sampling Presets
#
THINKING_PARAMS = {
    "chat_template_kwargs": {"enable_thinking": True},
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0,
    "max_tokens": None,
}

NON_THINKING_PARAMS = {
    "chat_template_kwargs": {"enable_thinking": False},
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
    "max_tokens": None,
}

# Default sampling parameters (empty by default)
DEFAULT_SAMPLING_PARAMS = NON_THINKING_PARAMS

#
# Judge LLM Configuration (for answer extraction verification)
#
JUDGE_LLM_CONFIG = {
    "model_name": "gemini-2.0-flash",
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "api_key_name": "GEMINI_API_KEY",
    "system_prompt": None,
}

# Judge sampling parameters (for consistent judgments)
JUDGE_SAMPLING_PARAMS = {
    "temperature": 0.5,
    "max_tokens": 4096,
}

#
# Dataset Configuration
#
DATASET_CONFIG = {
    "dataset_name": "HPC-Boys/AIME_1983_2024",  # Alternative: "HPC-Boys/MATH_500_MMLU_Pro"
    "dataset_split": "train",
    "num_problems": 'all',  # Number of problems to test, or 'all' for entire dataset
}

#
# Generation Configuration
#
GENERATION_CONFIG = {
    "k_responses": 1,      # Number of responses per question
}

#
# Processing Configuration
#
PROCESSING_CONFIG = {
    "problem_batch_size": 30,        # Problems per batch for checkpointing
    "max_concurrent_requests": 30,   # Maximum concurrent API requests
}

#
# Output Configuration
#
OUTPUT_CONFIG = {
    "output_dir": "data_collection/inference_results",
}