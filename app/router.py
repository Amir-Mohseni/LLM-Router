import os
# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .classifier import Classifier


class ModelRouter:
    """
    Router that determines which model to use based on input content
    """
    
    def __init__(self):
        # Define model options with provider information
        self.models = {
            # OpenRouter models - user requested models
            "google/gemini-2.5-pro-preview": {
                "display_name": "Gemini 2.5 Pro",
                "provider": "openrouter"
            },
            "google/gemini-2.0-flash-001": {
                "display_name": "Gemini 2.0 Flash",
                "provider": "openrouter"
            },
            "qwen/qwen3-14b": {
                "display_name": "Qwen 3 14B",
                "provider": "openrouter"
            }
        }
        self.classifier = Classifier(model_name='AmirMohseni/BERT-Router-base')
    
    def select_model(self, message, history):
        """
        Select the best model based on the input message and conversation history
        
        Args:
            message (str): The user's current message
        Returns:
            str: The model key to use
        """        
        # Classify the message (small_llm, large_llm)
        model_type = self.classifier.classify(message)
        
        if model_type == 'large_llm':
            return 'google/gemini-2.5-pro-preview'  # Large powerful model
        elif model_type == 'small_llm':
            return 'google/gemini-2.0-flash-001'    # Smaller, faster model
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
    def get_model_display_name(self, model_key):
        """Get the display name for a model key"""
        model_info = self.models.get(model_key, {})
        return model_info.get("display_name", "Unknown Model")
    
    def get_model_provider(self, model_key):
        """Get the provider for a model key"""
        model_info = self.models.get(model_key, {})
        return model_info.get("provider", "openrouter")  # Default to openrouter
    
    def get_model_key_from_display_name(self, display_name):
        """Convert display name back to model key"""
        for key, info in self.models.items():
            if info.get("display_name") == display_name:
                return key
        return None
    
    def get_available_models(self):
        """Return list of available model display names"""
        return [info["display_name"] for info in self.models.values()] 