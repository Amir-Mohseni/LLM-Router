from openai import OpenAI
import os
from router import ModelRouter

class LLMHandler:
    """Handler for LLM models to generate responses using Hugging Face's API"""
    
    def __init__(self):
        # API Key from environment variable
        self.api_key = os.getenv('HF_TOKEN')
        if not self.api_key:
            print("Warning: HF_TOKEN environment variable not set. API calls will fail.")
        
        # Initialize the model router
        self.router = ModelRouter()
        
        # Cache for OpenAI clients
        self.client_cache = {}
        
        # Maximum conversation history to maintain
        self.max_history_length = 5
    
    def get_available_models(self):
        """Return list of available models"""
        return self.router.get_available_models()
    
    def _get_client(self, model_key):
        """Get or create an OpenAI client for the specified model"""
        if model_key not in self.client_cache:
            self.client_cache[model_key] = OpenAI(
                base_url=f"https://router.huggingface.co/hf-inference/models/{model_key}/v1",
                api_key=self.api_key
            )
        return self.client_cache[model_key]
    
    def _format_messages(self, message, history):
        """Format the conversation history into messages for the API"""
        messages = []
        
        # Only use the last few turns to avoid context length issues
        recent_history = history[-self.max_history_length:] if history else []
        
        # Add conversation history
        for user_msg, assistant_msg in recent_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add the current message
        messages.append({"role": "user", "content": message})
        
        return messages
    
    def generate_response(self, message, history, model_selection):
        """Generate a response from the selected model"""
        try:
            # Handle automatic model selection
            if model_selection == "automatic":
                model_key = self.router.select_model(message, history)
                model_info = f"Using {self.router.get_model_display_name(model_key)} (automatic selection)"
            else:
                # Convert display name back to model key
                model_key = self.router.get_model_key_from_display_name(model_selection)
                if not model_key:
                    return "Error: Model not found"
                model_info = f"Using {model_selection}"
            
            # Get the client for this model
            client = self._get_client(model_key)
            
            # Format messages for the API
            messages = self._format_messages(message, history)
            
            # Call the API
            completion = client.chat.completions.create(
                model=model_key,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                max_tokens=500,
            )
            
            # Extract the response
            response = completion.choices[0].message.content
            
            # Update history with the new interaction
            history.append((message, response))
            
            return history
            
        except Exception as e:
            # Handle any API errors
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            history.append((message, f"Sorry, I encountered an error: {str(e)}"))
            return history 