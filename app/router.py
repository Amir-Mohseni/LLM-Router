import os
# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .classifier import Classifier


class ModelRouter:
    """
    Router that determines which model to use based on input content
    """
    
    def __init__(self):
        # Define model options
        self.models = {
            "google/gemma-3-27b-it": "Gemma 3 27B",
            "meta-llama/Llama-3.2-1B-Instruct": "Llama 3.2 1B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "Distill R1 1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "Distill R1 32B",
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
            return 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
        elif model_type == 'small_llm':
            return 'meta-llama/Llama-3.2-1B-Instruct'
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
    def get_model_display_name(self, model_key):
        """Get the display name for a model key"""
        return self.models.get(model_key, "Unknown Model")
    
    def get_model_key_from_display_name(self, display_name):
        """Convert display name back to model key"""
        for key, name in self.models.items():
            if name == display_name:
                return key
        return None
    
    def get_available_models(self):
        """Return list of available model display names"""
        return list(self.models.values()) 