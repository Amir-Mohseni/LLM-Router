"""
LLM abstraction layer that provides a unified interface for language models.

This module provides:
1. Standard text-based LLM interactions
2. Structured output generation from LLMs
3. Vision capabilities with image inputs

It uses LangChain under the hood but provides a simplified interface.
"""

import os
import json
import logging
import base64
import requests
from typing import Any, Dict, List, Optional, Type, Union, TypeVar
from abc import ABC, abstractmethod
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
from io import BytesIO
from urllib.parse import urlparse

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLMOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv(override=True)

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Image functionality will be limited.")

# Define type variable for generic class
T = TypeVar('T', bound=BaseModel)

# Define parameter sets for different API types
REMOTE_API_PARAMS = {"temperature", "max_tokens"}
LOCAL_VLLM_PARAMS = {"stop", "temperature", "max_tokens", "repetition_penalty", "min_tokens"}


def _is_url(string: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def _process_image(image: Union[str, Path, "Image.Image"]) -> Dict[str, Any]:
    """
    Process a single image into the format expected by LangChain.
    
    Args:
        image: Image as file path, Path object, URL string, or PIL Image
        
    Returns:
        Dict containing the image data in LangChain format
    """
    if isinstance(image, str) and _is_url(image):
        # Handle URL case - return URL directly
        return {
            "type": "image_url",
            "image_url": {"url": image}
        }
        
    elif isinstance(image, (str, Path)):
        # Handle local file case
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        ext = image_path.suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", 
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp"
        }.get(ext, "image/jpeg")
        
        with image_path.open("rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
            
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
        }
            
    elif PIL_AVAILABLE and isinstance(image, Image.Image):
        # Default to JPEG encoding for PIL images
        mime_type = "image/jpeg"
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
        }
        
    else:
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for Image.Image objects. Install with: pip install Pillow")
        raise TypeError("Image must be a file path (str/Path), URL (str), or a PIL.Image.Image object.")


def _build_message_content(prompt: str, images: Optional[List[Union[str, Path, "Image.Image"]]] = None) -> List[Dict[str, Any]]:
    """
    Build message content with text and optional images.
    
    Args:
        prompt: Text prompt
        images: Optional list of images (file paths, URLs, Path objects, or PIL Images, can be empty list)
        
    Returns:
        List of content items for LangChain message
    """
    content = [{"type": "text", "text": prompt}]
    
    if images:
        for image in images:
            content.append(_process_image(image))
            
    return content


class BaseLLM(ABC):
    """Base abstract LLM class that provides a unified interface for language models."""
    
    def __init__(
        self, 
        model_name: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize a base LLM instance.
        
        Args:
            model_name (str): The specific model name
            temperature (float): Temperature setting for generation
            max_tokens (int, optional): Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.model = None  # Will be initialized by subclasses
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the model - to be implemented by subclasses"""
        pass
    
    def invoke(self, prompt: str, images: Optional[List[Union[str, Path, "Image.Image"]]] = None) -> str:
        """
        Invoke the model with a text prompt and optional images.
        
        Args:
            prompt (str): The prompt to send to the model
            images (List, optional): List of images (file paths, URLs, Path objects, or PIL Images)
            
        Returns:
            str: The model's response
        """
        if not self.model:
            raise ValueError("Model has not been initialized")
            
        try:
            if images:
                # Use vision-capable message format
                content = _build_message_content(prompt, images)
                message = HumanMessage(content=content)
                response = self.model.invoke([message])
            else:
                # Use simple text prompt
                response = self.model.invoke(prompt)
                
            if hasattr(response, 'content'):
                return response.content
            return response
        except Exception as e:
            logger.error(f"Error invoking model: {str(e)}")
            raise
    
    async def ainvoke(self, prompt: str, images: Optional[List[Union[str, Path, "Image.Image"]]] = None) -> str:
        """
        Asynchronously invoke the model with a text prompt and optional images.
        
        Args:
            prompt (str): The prompt to send to the model
            images (List, optional): List of images (file paths, URLs, Path objects, or PIL Images)
            
        Returns:
            str: The model's response
        """
        if not self.model:
            raise ValueError("Model has not been initialized")
            
        try:
            if images:
                # Use vision-capable message format
                content = _build_message_content(prompt, images)
                message = HumanMessage(content=content)
                response = await self.model.ainvoke([message])
            else:
                # Use simple text prompt
                response = await self.model.ainvoke(prompt)
                
            if hasattr(response, 'content'):
                return response.content
            return response
        except Exception as e:
            logger.error(f"Error in async invocation: {str(e)}")
            raise
            
    def invoke_structured(self, prompt: str, output_schema: Type[T], images: Optional[List[Union[str, Path, "Image.Image"]]] = None) -> T:
        """
        Invoke the model with structured output and optional images.
        
        Args:
            prompt: The prompt to send to the model
            output_schema: Pydantic model class for structured output
            images (List, optional): List of images (file paths, URLs, Path objects, or PIL Images)
            
        Returns:
            An instance of the output_schema class
        """
        if not self.model:
            raise ValueError("Model has not been initialized")
            
        try:
            # Create a structured LLM from the current LLM
            structured_llm = self.model.with_structured_output(output_schema)
            
            # Get response from structured LLM
            if images:
                # Use vision-capable message format
                content = _build_message_content(prompt, images)
                message = HumanMessage(content=content)
                response = structured_llm.invoke([message])
            else:
                # Use simple text prompt
                response = structured_llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Failed to generate structured output: {str(e)}")
            raise
            
    async def ainvoke_structured(self, prompt: str, output_schema: Type[T], images: Optional[List[Union[str, Path, "Image.Image"]]] = None) -> T:
        """
        Asynchronously invoke the model with structured output and optional images.
        
        Args:
            prompt: The prompt to send to the model
            output_schema: Pydantic model class for structured output
            images (List, optional): List of images (file paths, URLs, Path objects, or PIL Images)
            
        Returns:
            An instance of the output_schema class
        """
        if not self.model:
            raise ValueError("Model has not been initialized")
            
        try:
            # Create a structured LLM from the current LLM
            structured_llm = self.model.with_structured_output(output_schema)
            
            # Get response from structured LLM
            if images:
                # Use vision-capable message format
                content = _build_message_content(prompt, images)
                message = HumanMessage(content=content)
                response = await structured_llm.ainvoke([message])
            else:
                # Use simple text prompt
                response = await structured_llm.ainvoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Failed to generate structured output: {str(e)}")
            raise


class RemoteLLM(BaseLLM):
    """Class for remote API-based LLMs like OpenAI models"""
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize a RemoteLLM instance.
        
        Args:
            model_name (str): The model name to use
            api_key (str): API key for the provider
            base_url (str, optional): Base URL for API requests
            temperature (float): Temperature setting for generation
            max_tokens (int, optional): Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        """
        # Log warnings about incompatible parameters but don't filter them
        unsupported_params = set(kwargs.keys()) - REMOTE_API_PARAMS
        if unsupported_params:
            logger.warning(f"Some parameters may not be supported by remote API: {unsupported_params}")
            
        super().__init__(model_name, temperature, max_tokens, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the OpenAI model"""
        # Build config for the model
        model_config = {
            "temperature": self.temperature,
            "api_key": self.api_key,
            "model": self.model_name,
        }
        
        # Add base URL if provided
        if self.base_url:
            model_config["base_url"] = self.base_url
            
        # Add max_tokens if provided
        if self.max_tokens is not None:
            model_config["max_tokens"] = self.max_tokens
            
        # Add any additional kwargs
        try:
            model_config.update(self.kwargs)
            
            # Initialize the ChatOpenAI model
            self.model = ChatOpenAI(**model_config)
            logger.info(f"Initialized remote LLM: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing remote LLM: {str(e)}")
            raise


class LocalLLM(BaseLLM):
    """Class for local LLMs using vLLM"""
    
    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:8000/v1",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize a LocalLLM instance.
        
        Args:
            model_name (str): The model name to use (HuggingFace model ID)
            api_base (str): Base URL for the local vLLM server
            temperature (float): Temperature setting for generation
            max_tokens (int, optional): Maximum tokens to generate
            **kwargs: Additional model-specific parameters
        """
        # Log warnings about incompatible parameters but don't filter them
        unsupported_params = set(kwargs.keys()) - LOCAL_VLLM_PARAMS
        if unsupported_params:
            logger.warning(f"Some parameters may not be supported by vLLM: {unsupported_params}")
            
        super().__init__(model_name, temperature, max_tokens, **kwargs)
        self.api_base = api_base
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the vLLM model"""
        # Build config for the model
        model_config = {
            "openai_api_key": "EMPTY",  # Not used but required by the API
            "openai_api_base": self.api_base,
            "model_name": self.model_name,
            "temperature": self.temperature,
        }
        
        # Add max_tokens if provided
        if self.max_tokens is not None:
            model_config["max_tokens"] = self.max_tokens
            
        # Add model kwargs, which might include unsupported parameters
        # The LangChain VLLMOpenAI class will handle these appropriately
        try:
            if self.kwargs:
                model_config["model_kwargs"] = self.kwargs
                
            # Initialize the VLLMOpenAI model
            self.model = VLLMOpenAI(**model_config)
            logger.info(f"Initialized local vLLM: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing local vLLM: {str(e)}")
            raise


# Factory function to create an LLM instance based on the mode
def create_llm(
    model_name: str,
    api_mode: str = "remote", 
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    **kwargs
) -> BaseLLM:
    """
    Create an LLM instance with the specified configuration.
    
    Args:
        model_name (str): The specific model name
        api_mode (str): "remote" for API-based models or "local" for vLLM models
        api_key (str, optional): API key for remote providers
        api_base (str, optional): Base URL for API requests
        temperature (float): Temperature setting for generation
        max_tokens (int, optional): Maximum tokens to generate
        **kwargs: Additional provider-specific parameters
        
    Returns:
        BaseLLM: Configured LLM instance (either RemoteLLM or LocalLLM)
    """
    if api_mode.lower() == "remote":
        if not api_key:
            raise ValueError("API key must be provided for remote LLM")
        return RemoteLLM(
            model_name=model_name,
            api_key=api_key,
            base_url=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif api_mode.lower() == "local":
        # Default api_base for local vLLM if not provided
        local_api_base = api_base or "http://localhost:8000/v1"
        return LocalLLM(
            model_name=model_name,
            api_base=local_api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported API mode: {api_mode}. Use 'remote' or 'local'.")


def generate(prompt: str, image: Optional[Union[str, Path, "Image.Image", List[Union[str, Path, "Image.Image"]]]] = None, llm: Optional[BaseLLM] = None) -> str:
    """
    Convenience function for generating responses with optional image support.
    Compatible with the original generate function signature.
    
    Args:
        prompt (str): The text prompt
        image (optional): Single image or list of images (file paths, URLs, Path objects, or PIL Images)
        llm (BaseLLM, optional): The LLM instance to use
        
    Returns:
        str: The model's response
        
    Raises:
        ValueError: If no LLM instance is provided
    """
    if llm is None:
        raise ValueError("LLM instance must be provided")
    
    # Handle the case where image is a single item (convert to list)
    images = None
    if image is not None:
        if isinstance(image, list):
            images = image if image else None  # Handle empty list
        else:
            images = [image]  # Convert single image to list
    
    return llm.invoke(prompt, images=images) 