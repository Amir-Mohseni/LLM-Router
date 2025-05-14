from openai import OpenAI
import os
from dotenv import load_dotenv
import json

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
    
    def call_model(self, provider, model_name, messages, stream=False, max_tokens=4096, enable_reasoning=None):
        """
        Call an LLM model with the specified provider and model name
        
        Args:
            provider (str): Provider name ('openai' or 'openrouter')
            model_name (str): Name of the model to use
            messages (list): List of message dictionaries (role, content)
            stream (bool): Whether to return a streaming response
            max_tokens (int): Maximum number of tokens to generate
            enable_reasoning (bool): Whether to enable reasoning for supported models.
                                     If None, will use reasoning for models that support it.
            
        Returns:
            Response from the model or a stream of responses
        """
        try:
            # Get the client for this provider
            client = self._get_client(provider)
            
            # Create the base API params
            api_params = {
                "model": model_name,
                "messages": messages,
                "stream": stream,
                "max_tokens": max_tokens,
            }
            
            # Always enable reasoning for OpenRouter calls regardless of model
            if provider == "openrouter":
                # Check if this model supports reasoning according to our router configuration
                supports_reasoning = self.router.model_supports_reasoning(model_name)
                
                if supports_reasoning:
                    print(f"[DEBUG] Enabling reasoning for {model_name} (supports reasoning)")
                    api_params["reasoning"] = {
                        "max_tokens": self.config.get("reasoning_max_tokens", 2000),
                        "exclude": False  # Important: Ensure reasoning is included in the response
                    }
                else:
                    print(f"[DEBUG] Skipping reasoning for {model_name} (not supported according to configuration)")
            
            # Call the API with error handling for unsupported parameters
            try:
                print(f"[DEBUG] Calling {provider}/{model_name} with params: {api_params}")
                response = client.chat.completions.create(**api_params)
                
                # For non-streaming responses, log some information about the response
                if not stream and provider == "openrouter":
                    try:
                        print(f"[DEBUG] Response keys: {dir(response)}")
                        print(f"[DEBUG] Response choices: {response.choices}")
                        print(f"[DEBUG] Response first choice: {dir(response.choices[0])}")
                        print(f"[DEBUG] Response message keys: {dir(response.choices[0].message)}")
                        # Try to access reasoning directly
                        try:
                            reasoning = response.choices[0].message.reasoning
                            print(f"[DEBUG] Found reasoning: {reasoning[:100]}...")
                        except Exception as e:
                            print(f"[DEBUG] No reasoning attribute: {str(e)}")
                            
                        # Try to access message as dict
                        try:
                            message_dict = response.choices[0].message.model_dump()
                            print(f"[DEBUG] Message as dict: {message_dict.keys()}")
                            if 'reasoning' in message_dict:
                                print(f"[DEBUG] Found reasoning in dict: {message_dict['reasoning'][:100]}...")
                        except Exception as e:
                            print(f"[DEBUG] Could not convert message to dict: {str(e)}")
                    except Exception as e:
                        print(f"[DEBUG] Error examining response: {str(e)}")
                
            except TypeError as e:
                # If 'reasoning' parameter causes an error, remove it and try again
                if "unexpected keyword argument 'reasoning'" in str(e) or "not present in schema" in str(e):
                    print(f"Warning: Reasoning parameter not supported for {provider}/{model_name}. Removing it.")
                    if "reasoning" in api_params:
                        del api_params["reasoning"]
                    # Try again without the reasoning parameter
                    response = client.chat.completions.create(**api_params)
                else:
                    # Re-raise other TypeError issues
                    raise
            
            # Process the response appropriately
            if not stream:
                # Extract content from the response
                content = response.choices[0].message.content
                
                # Check if we're getting HTML error response
                if content.strip().startswith("<!DOCTYPE html>") or content.strip().startswith("<html"):
                    print(f"[ERROR] Received HTML error response")
                    return {"content": "Sorry, the service is currently unavailable. Please try again later."}
                
                # Prepare result with content
                result = {"content": content}
                
                # Try to extract reasoning from OpenRouter response
                if provider == "openrouter":
                    print(f"[DEBUG] Attempting to extract reasoning from {model_name} response")
                    
                    # Method 1: Try to access reasoning attribute directly
                    try:
                        reasoning = getattr(response.choices[0].message, "reasoning", None)
                        if reasoning:
                            print(f"[DEBUG] Found reasoning via attribute: {reasoning[:50]}...")
                            result["reasoning"] = reasoning
                            result["has_reasoning"] = True
                            return result
                    except Exception as e:
                        print(f"[DEBUG] Method 1 failed: {str(e)}")
                    
                    # Method 2: Try to access via __dict__
                    try:
                        message_dict = vars(response.choices[0].message)
                        if 'reasoning' in message_dict:
                            reasoning = message_dict['reasoning']
                            print(f"[DEBUG] Found reasoning via __dict__: {reasoning[:50]}...")
                            result["reasoning"] = reasoning
                            result["has_reasoning"] = True
                            return result
                    except Exception as e:
                        print(f"[DEBUG] Method 2 failed: {str(e)}")
                    
                    # Method 3: Try to convert to dict with model_dump
                    try:
                        message_dict = response.choices[0].message.model_dump()
                        if 'reasoning' in message_dict:
                            reasoning = message_dict['reasoning']
                            print(f"[DEBUG] Found reasoning via model_dump: {reasoning[:50]}...")
                            result["reasoning"] = reasoning
                            result["has_reasoning"] = True
                            return result
                    except Exception as e:
                        print(f"[DEBUG] Method 3 failed: {str(e)}")
                    
                    # Method 4: Try to access as a custom field
                    try:
                        reasoning = response.choices[0].message.custom_fields.get('reasoning')
                        if reasoning:
                            print(f"[DEBUG] Found reasoning via custom_fields: {reasoning[:50]}...")
                            result["reasoning"] = reasoning
                            result["has_reasoning"] = True
                            return result
                    except Exception as e:
                        print(f"[DEBUG] Method 4 failed: {str(e)}")
                    
                    print(f"[DEBUG] Could not extract reasoning from response")
                
                return result
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
                return {"content": "Sorry, the service is currently unavailable. Please try again later."}
            else:
                return {"content": f"Sorry, I encountered an error: {str(e)}"}
    
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
    
    def generate_response(self, message, history, model_selection, enable_reasoning=None):
        """
        Generate a response from the selected model
        Returns the complete response as a string or dictionary with reasoning
        
        Args:
            message (str): User's message
            history (list): Conversation history as list of (user, assistant) tuples
            model_selection (str): Model selection (display name or "automatic")
            enable_reasoning (bool): Whether to enable reasoning for supported models
        
        Returns:
            str or dict: Response content, or dict with content and reasoning
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
            result = self.call_model(provider, model_key, messages, stream=False, enable_reasoning=enable_reasoning)
            
            return result
            
        except Exception as e:
            # Handle any errors
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return f"Sorry, I encountered an error: {str(e)}"
    
    def generate_streaming_response(self, message, history, model_selection, enable_reasoning=None):
        """
        Generate a streaming response from the selected model
        Yields chunks of text as they are received from the API
        
        Args:
            message (str): User's message
            history (list): Conversation history as list of (user, assistant) tuples
            model_selection (str): Model selection (display name or "automatic")
            enable_reasoning (bool): Whether to enable reasoning for supported models
        
        Yields:
            dict: Response chunks with content and optional reasoning chunks
        """
        try:
            # Handle automatic model selection
            if model_selection == "automatic":
                model_key = self.router.select_model(message, history)
                model_info = f"Using {self.router.get_model_display_name(model_key)} (automatic selection)"
                print(f"[DEBUG] Auto-selected model: {model_key}")
                # First yield the model info
                yield {"content": model_info}
            else:
                # Convert display name back to model key
                model_key = self.router.get_model_key_from_display_name(model_selection)
                print(f"[DEBUG] User-selected model: {model_key}")
                if not model_key:
                    yield {"content": "Error: Model not found"}
                    return
            
            # Get the provider for this model
            provider = self.router.get_model_provider(model_key)
            print(f"[DEBUG] Using provider: {provider}")
            
            # Format messages for the API
            messages = self._format_messages(message, history)
            
            # Always enable reasoning for provider-supported models
            supports_reasoning = provider == "openrouter"
            has_reasoning = supports_reasoning
            
            # If provider supports reasoning, indicate it in first response
            if has_reasoning:
                print(f"[DEBUG] Model supports reasoning, indicating in response")
                yield {"content": "", "has_reasoning": True, "reasoning": ""}
            
            # Call the model with streaming enabled using the appropriate provider
            # Always try with reasoning for OpenRouter
            stream = self.call_model(
                provider, 
                model_key, 
                messages, 
                stream=True, 
                enable_reasoning=True  # Always try reasoning with OpenRouter
            )
            
            # Handle error responses (strings or dicts with error content)
            if not hasattr(stream, '__iter__') and not hasattr(stream, '__next__'):
                if isinstance(stream, str):
                    yield {"content": stream}
                elif isinstance(stream, dict):
                    yield stream
                return
                        
            # Stream the response chunks
            full_content = ""
            full_reasoning = ""
            
            for chunk in stream:
                updated = False
                
                # Check if we received delta content
                if hasattr(chunk.choices[0], "delta"):
                    delta = chunk.choices[0].delta
                    
                    # Extract content from delta
                    if hasattr(delta, "content") and delta.content is not None:
                        content = delta.content
                        full_content += content
                        updated = True
                    
                    # Extract reasoning from delta (for OpenRouter)
                    if has_reasoning and hasattr(delta, "reasoning") and delta.reasoning is not None:
                        reasoning = delta.reasoning
                        full_reasoning += reasoning
                        updated = True
                
                # Only yield if something was updated
                if updated:
                    # Return complete buffers in each result
                    result = {"content": full_content}
                    if has_reasoning and full_reasoning:
                        result["reasoning"] = full_reasoning
                        result["has_reasoning"] = True
                    yield result
            
            # Yield final content with reasoning flag if applicable
            final_result = {"content": full_content}
            if has_reasoning and full_reasoning:
                final_result["reasoning"] = full_reasoning
                final_result["has_reasoning"] = True
            
            yield final_result
            
        except Exception as e:
            # Handle any errors
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            yield {"content": f"Sorry, I encountered an error: {str(e)}"}
    
    def call_model_directly(self, provider, model_name, message, history=None, stream=False, max_tokens=4096, enable_reasoning=None):
        """
        Call a model directly with a simpler interface
        
        Args:
            provider (str): Provider name ('openai' or 'openrouter')
            model_name (str): Name of the model to use
            message (str): User message to send
            history (list): Optional conversation history list of (user, assistant) tuples
            stream (bool): Whether to return a streaming response
            max_tokens (int): Maximum number of tokens to generate
            enable_reasoning (bool): Whether to enable reasoning for supported models
            
        Returns:
            Response from the model or a stream of responses
        """
        try:
            # Format messages
            messages = self._format_messages(message, history or [])
            
            # Call the model
            return self.call_model(provider, model_name, messages, stream, max_tokens, enable_reasoning)
            
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
    
    print("\nTesting with thinking model")
    try:
        # Test calling a model directly with reasoning enabled
        response = llm.call_model_directly(
            "openrouter", 
            "google/gemini-2.5-pro-preview", 
            "What's the most efficient algorithm for sorting a large dataset?",
            enable_reasoning=True
        )
        
        if isinstance(response, dict):
            if "reasoning" in response:
                print("\nReasoning:")
                print(response["reasoning"])
            print("\nResponse:")
            print(response["content"])
        else:
            print(f"Response: {response}")
    except Exception as e:
        print(f"Test failed: {str(e)}")
        print("Make sure you have OPENROUTER_API_KEY in your .env file")
