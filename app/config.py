"""
Configuration settings for the LLM Router application.
"""
import os
import json

# Default API settings
LLM_CONFIG = {
    # Base URLs for different providers
    "base_urls": {
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1"
    },
    
    # Environment variable names for API keys
    "api_keys": {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY"
    },
    
    # Default reasoning tokens limit
    "reasoning_max_tokens": 2000,
    
    # Maximum conversation history to maintain
    "max_history_length": 10,
}

def _load_env_overrides():
    """
    Load configuration overrides from environment variables (if any)
    
    Supported environment variables (all optional):
    - LLM_CONFIG_JSON: JSON string with config overrides
    - LLM_BASE_URL_OPENROUTER: Override OpenRouter base URL
    - LLM_BASE_URL_OPENAI: Override OpenAI base URL
    """
    # Check for JSON config override (optional)
    if os.getenv("LLM_CONFIG_JSON"):
        try:
            overrides = json.loads(os.getenv("LLM_CONFIG_JSON"))
            for key, value in overrides.items():
                if key in LLM_CONFIG and isinstance(LLM_CONFIG[key], dict) and isinstance(value, dict):
                    # Merge dictionaries for nested values
                    LLM_CONFIG[key].update(value)
                else:
                    # Direct override for simple values
                    LLM_CONFIG[key] = value
        except json.JSONDecodeError:
            print("Warning: Invalid JSON in LLM_CONFIG_JSON environment variable")
    
    # Individual base URL overrides (optional)
    if os.getenv("LLM_BASE_URL_OPENROUTER"):
        LLM_CONFIG["base_urls"]["openrouter"] = os.getenv("LLM_BASE_URL_OPENROUTER")
    
    if os.getenv("LLM_BASE_URL_OPENAI"):
        LLM_CONFIG["base_urls"]["openai"] = os.getenv("LLM_BASE_URL_OPENAI")

def get_config():
    """Returns the current configuration settings with environment overrides (if any)"""
    _load_env_overrides()
    return LLM_CONFIG 