"""
LLM abstraction layer that provides a unified interface for language models.

This module provides:
1. Standard text-based LLM interactions
2. Structured output generation from LLMs
3. Vision capabilities with image inputs

It uses OpenAI API directly.
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
import asyncio

# Import OpenAI components
import openai
from openai import OpenAI, AsyncOpenAI

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

def _is_url(string: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def _process_image(image: Union[str, Path, "Image.Image"]) -> Dict[str, Any]:
    """
    Process a single image into the format expected by OpenAI.
    
    Args:
        image: Image as file path, Path object, URL string, or PIL Image
        
    Returns:
        Dict containing the image data in OpenAI format
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
        List of content items for OpenAI message
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
        system_prompt: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize a base LLM instance.
        
        Args:
            model_name (str): The specific model name
            system_prompt (str, optional): System prompt to prepend to all conversations
            extra_body (dict, optional): Extra body parameters to pass to the API
            **kwargs: Additional provider-specific parameters including temperature and max_tokens
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.extra_body = extra_body
        self.kwargs = kwargs
        self.client = None  # Will be initialized by subclasses
        self.async_client = None  # Async client
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the model - to be implemented by subclasses"""
        pass
    
    def _build_messages(self, prompt: str, images: Optional[List[Union[str, Path, "Image.Image"]]] = None) -> List[Dict[str, Any]]:
        """
        Build messages list with optional system prompt.
        
        Args:
            prompt (str): The user prompt
            images (List, optional): List of images for the user message
            
        Returns:
            List of messages for the model
        """
        messages = []
        
        # Add system message if system_prompt is set
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add user message with optional images
        if images:
            content = _build_message_content(prompt, images)
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
            
        return messages
    
    def invoke(self, prompt: str, images: Optional[List[Union[str, Path, "Image.Image"]]] = None) -> str:
        """
        Invoke the model with a text prompt and optional images.
        
        Args:
            prompt (str): The prompt to send to the model
            images (List, optional): List of images (file paths, URLs, Path objects, or PIL Images)
            
        Returns:
            str: The model's response
        """
        if not self.client:
            raise ValueError("Model has not been initialized")
            
        try:
            messages = self._build_messages(prompt, images)
            
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "messages": messages,
                **self.kwargs
            }
            
            # Add extra_body if provided
            if self.extra_body:
                request_params["extra_body"] = self.extra_body
                logger.debug(f"Using extra_body parameters: {self.extra_body}")
            
            logger.debug(f"Request parameters: {request_params}")
            response = self.client.chat.completions.create(**request_params)
            return response.choices[0].message.content
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
        if not self.async_client:
            raise ValueError("Async client has not been initialized")
            
        try:
            messages = self._build_messages(prompt, images)
            
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "messages": messages,
                **self.kwargs
            }
            
            # Add extra_body if provided
            if self.extra_body:
                request_params["extra_body"] = self.extra_body
            
            response = await self.async_client.chat.completions.create(**request_params)
            return response.choices[0].message.content
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
        if not self.client:
            raise ValueError("Model has not been initialized")
            
        try:
            messages = self._build_messages(prompt, images)
            
            # Build request parameters with response_format for structured output
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "response_format": {"type": "json_object"},
                **self.kwargs
            }
            
            # Add extra_body if provided
            if self.extra_body:
                request_params["extra_body"] = self.extra_body
            
            response = self.client.chat.completions.create(**request_params)
            content = response.choices[0].message.content
            
            # Parse JSON response and validate with Pydantic
            try:
                data = json.loads(content)
                return output_schema(**data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {content}")
                raise ValueError(f"Invalid JSON response: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to validate with schema: {str(e)}")
                raise ValueError(f"Response doesn't match schema: {str(e)}")
                
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
        if not self.async_client:
            raise ValueError("Async client has not been initialized")
            
        try:
            messages = self._build_messages(prompt, images)
            
            # Build request parameters with response_format for structured output
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "response_format": {"type": "json_object"},
                **self.kwargs
            }
            
            # Add extra_body if provided
            if self.extra_body:
                request_params["extra_body"] = self.extra_body
            
            response = await self.async_client.chat.completions.create(**request_params)
            content = response.choices[0].message.content
            
            # Parse JSON response and validate with Pydantic
            try:
                data = json.loads(content)
                return output_schema(**data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {content}")
                raise ValueError(f"Invalid JSON response: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to validate with schema: {str(e)}")
                raise ValueError(f"Response doesn't match schema: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to generate structured output: {str(e)}")
            raise


class RemoteLLM(BaseLLM):
    """Class for remote API-based LLMs using OpenAI-compatible APIs"""
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize a RemoteLLM instance.
        
        Args:
            model_name (str): The model name to use
            api_key (str): API key for the provider
            base_url (str, optional): Base URL for API requests
            system_prompt (str, optional): System prompt to prepend to all conversations
            extra_body (dict, optional): Extra body parameters to pass to the API
            **kwargs: Additional provider-specific parameters including temperature and max_tokens
        """
        super().__init__(model_name, system_prompt, extra_body, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the OpenAI client"""
        try:
            # Build config for the client
            client_config = {
                "api_key": self.api_key,
            }
            
            # Add base URL if provided
            if self.base_url:
                client_config["base_url"] = self.base_url
                
            # Initialize both sync and async clients
            self.client = OpenAI(**client_config)
            self.async_client = AsyncOpenAI(**client_config)
            
            logger.info(f"Initialized remote LLM: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing remote LLM: {str(e)}")
            raise

# Factory function to create an LLM instance (always remote)
def create_llm(
    model_name: str,
    api_key: str,
    api_base: str,
    system_prompt: Optional[str] = None,
    sampling_params: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseLLM:
    """
    Create an LLM instance with the specified configuration.
    
    Args:
        model_name (str): The specific model name
        api_key (str): API key for the provider
        api_base (str): Base URL for API requests
        system_prompt (str, optional): System prompt to prepend to all conversations
        sampling_params (dict, optional): Sampling parameters (temperature, max_tokens, etc.)
        extra_body (dict, optional): Extra body parameters to pass to the API
        **kwargs: Additional provider-specific parameters
        
    Returns:
        BaseLLM: Configured RemoteLLM instance
    """
    if not api_key:
        raise ValueError("API key must be provided")
    
    # Merge sampling_params into extra_body instead of passing as kwargs
    if sampling_params:
        if extra_body is None:
            extra_body = {}
        extra_body.update(sampling_params)
    
    return RemoteLLM(
        model_name=model_name,
        api_key=api_key,
        base_url=api_base,
        system_prompt=system_prompt,
        extra_body=extra_body,
        **kwargs
    )


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