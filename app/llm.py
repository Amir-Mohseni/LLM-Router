from openai import OpenAI
import os
# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .router import ModelRouter

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
            
            # Get the client for this model
            client = self._get_client(model_key)
            
            # Format messages for the API
            messages = self._format_messages(message, history)
            
            # Call the API without streaming
            response = client.chat.completions.create(
                model=model_key,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                max_tokens=1000,
            )
            
            # Return the full response
            content = response.choices[0].message.content
            # Check if we're getting HTML error response
            if content.strip().startswith("<!DOCTYPE html>") or content.strip().startswith("<html"):
                print(f"[ERROR] Received HTML error response")
                return "Sorry, the service is currently unavailable. Please try again later or select a different model."
            
            return content
            
        except Exception as e:
            # Handle any API errors
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            
            # Check if the error contains HTML (service unavailable response)
            error_text = str(e)
            if "<!DOCTYPE html>" in error_text or "<html" in error_text:
                return "Sorry, the Hugging Face service is currently unavailable. Please try again later or select a different model."
            else:
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
            
            # Get the client for this model
            client = self._get_client(model_key)
            
            # Format messages for the API
            messages = self._format_messages(message, history)
            
            # Call the API with streaming enabled
            stream = client.chat.completions.create(
                model=model_key,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                stream=True,
                max_tokens=500,
            )
                        
            # Stream the response chunks - FIXED to accumulate the full response
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    # Check if we're getting HTML error response
                    if content.strip().startswith("<!DOCTYPE html>") or content.strip().startswith("<html"):
                        print(f"[ERROR] Received HTML error response")
                        yield "Sorry, the service is currently unavailable. Please try again later or select a different model."
                        return
                    full_response += content
                    # Yield the full accumulated response so far
                    yield full_response
            
        except Exception as e:
            # Handle any API errors
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            
            # Check if the error contains HTML (service unavailable response)
            error_text = str(e)
            if "<!DOCTYPE html>" in error_text or "<html" in error_text:
                yield "Sorry, the Hugging Face service is currently unavailable. Please try again later or select a different model."
            else:
                yield f"Sorry, I encountered an error: {str(e)}"
        
if __name__ == "__main__":
    llm = LLMHandler()
    print(llm.get_available_models())
    print("\n--------------------------------")
    print("Streaming response:")
    # Stream the response
    for chunk in llm.generate_streaming_response("What is the capital of France?", [], "automatic"):
        print(chunk, end="")
    print("\n--------------------------------")  
    # Generate the response
    response = llm.generate_response("What is the capital of France?", [], "automatic")
    print(response)
