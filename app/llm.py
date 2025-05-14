from openai import OpenAI
import os
from dotenv import load_dotenv

from .router import ModelRouter
from .config import get_config

load_dotenv()

class LLMHandler:
    """Handler for LLM models to generate responses using different providers"""
    
    def __init__(self):
        # Get configuration
        self.config = get_config()
        
        # Initialize the model router
        self.router = ModelRouter()
        
        # Cache for OpenAI clients
        self.client_cache = {}
        
        # Maximum conversation history to maintain
        self.max_history_length = self.config["max_history_length"]
        
        # Check if required API keys exist and print warnings if missing
        self._check_api_keys()
    
    def _check_api_keys(self):
        """Check if required API keys exist in the environment"""
        for provider, key_name in self.config["api_keys"].items():
            if not os.getenv(key_name):
                print(f"WARNING: Missing {key_name} environment variable - {provider} calls will fail")
                print(f"Please add {key_name} to your .env file")
    
    def get_available_models(self):
        """Return list of available models"""
        return self.router.get_available_models()
    
    def _get_client(self, provider):
        """Get or create an API client for the specified provider"""
        if provider not in self.client_cache:
            # Get the base URL for this provider
            base_url = self.config["base_urls"].get(provider)
            if not base_url:
                raise ValueError(f"Unknown provider: {provider}")
            
            # Get the appropriate API key for this provider
            api_key_name = self.config["api_keys"].get(provider)
            api_key = os.getenv(api_key_name)
            
            if not api_key:
                raise ValueError(f"Missing API key: {api_key_name} environment variable not set")
            
            self.client_cache[provider] = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
        return self.client_cache[provider]
    
    def call_model(self, provider, model_name, messages, stream=False, max_tokens=2048):
        """
        Call an LLM model with the specified provider and model name
        
        Args:
            provider (str): Provider name ('openai' or 'openrouter')
            model_name (str): Name of the model to use
            messages (list): List of message dictionaries (role, content)
            stream (bool): Whether to return a streaming response
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            Response from the model or a stream of responses
        """
        try:
            # Get the client for this provider
            client = self._get_client(provider)
            
            # Call the API
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=stream,
                max_tokens=max_tokens,
            )
            
            if not stream:
                # Return the full response content for non-streaming requests
                content = response.choices[0].message.content
                # Check if we're getting HTML error response
                if content.strip().startswith("<!DOCTYPE html>") or content.strip().startswith("<html"):
                    print(f"[ERROR] Received HTML error response")
                    return "Sorry, the service is currently unavailable. Please try again later."
                return content
            else:
                # Return the stream for streaming requests
                return response
                
        except Exception as e:
            # Handle any API errors
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            
            # Check if the error contains HTML (service unavailable response)
            error_text = str(e)
            if "<!DOCTYPE html>" in error_text or "<html" in error_text:
                return "Sorry, the service is currently unavailable. Please try again later."
            else:
                return f"Sorry, I encountered an error: {str(e)}"
    
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
        """
        Generate a response from the selected model
        Returns the complete response as a string
        """
        try:
            # Handle automatic model selection
            if model_selection == "automatic":
                model_key = self.router.select_model(message, history)
            else:
                # Convert display name back to model key
                model_key = self.router.get_model_key_from_display_name(model_selection)
                if not model_key:
                    return "Error: Model not found"
            
            # Get the provider for this model
            provider = self.router.get_model_provider(model_key)
            
            # Format messages for the API
            messages = self._format_messages(message, history)
            
            # Call the model using the appropriate provider
            return self.call_model(provider, model_key, messages, stream=False)
            
        except Exception as e:
            # Handle any errors
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return f"Sorry, I encountered an error: {str(e)}"
    
    def generate_streaming_response(self, message, history, model_selection):
        """
        Generate a streaming response from the selected model
        Yields chunks of text as they are received from the API
        """
        try:
            # Handle automatic model selection
            if model_selection == "automatic":
                model_key = self.router.select_model(message, history)
                model_info = f"Using {self.router.get_model_display_name(model_key)} (automatic selection)"
                # First yield the model info
                yield model_info
            else:
                # Convert display name back to model key
                model_key = self.router.get_model_key_from_display_name(model_selection)
                if not model_key:
                    yield "Error: Model not found"
                    return
            
            # Get the provider for this model
            provider = self.router.get_model_provider(model_key)
            
            # Format messages for the API
            messages = self._format_messages(message, history)
            
            # Call the model with streaming enabled using the appropriate provider
            stream = self.call_model(provider, model_key, messages, stream=True)
            
            # Handle error responses (strings)
            if isinstance(stream, str):
                yield stream
                return
                        
            # Stream the response chunks
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    # Yield the full accumulated response so far
                    yield full_response
            
        except Exception as e:
            # Handle any errors
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            yield f"Sorry, I encountered an error: {str(e)}"
    
    def call_model_directly(self, provider, model_name, message, history=None, stream=False, max_tokens=2048):
        """
        Call a model directly with a simpler interface
        
        Args:
            provider (str): Provider name ('openai' or 'openrouter')
            model_name (str): Name of the model to use
            message (str): User message to send
            history (list): Optional conversation history list of (user, assistant) tuples
            stream (bool): Whether to return a streaming response
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            Response from the model or a stream of responses
        """
        try:
            # Format messages
            messages = self._format_messages(message, history or [])
            
            # Call the model
            return self.call_model(provider, model_name, messages, stream, max_tokens)
            
        except ValueError as ve:
            # Handle missing API keys or unknown providers
            error_msg = str(ve)
            print(f"Error calling model: {error_msg}")
            return f"Configuration error: {error_msg}"
            
        except Exception as e:
            # Handle other errors
            error_msg = str(e)
            print(f"Error calling model: {error_msg}")
            return f"Error: {error_msg}"

if __name__ == "__main__":
    llm = LLMHandler()
    print("Available models:")
    for model in llm.get_available_models():
        print(f"- {model}")
    
    print("\nTesting with default model")
    try:
        # Test calling a model directly
        response = llm.call_model_directly("openrouter", "google/gemini-2.0-flash-001", "What is the capital of France?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Test failed: {str(e)}")
        print("Make sure you have OPENROUTER_API_KEY in your .env file")
