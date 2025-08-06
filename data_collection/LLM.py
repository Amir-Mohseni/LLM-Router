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
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a base LLM instance.
        
        Args:
            model_name (str): The specific model name
            system_prompt (str, optional): System prompt to prepend to all conversations
            **kwargs: Additional provider-specific parameters including temperature and max_tokens
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.kwargs = kwargs
        self.model = None  # Will be initialized by subclasses
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the model - to be implemented by subclasses"""
        pass
    
    def _build_messages(self, prompt: str, images: Optional[List[Union[str, Path, "Image.Image"]]] = None) -> List:
        """
        Build messages list with optional system prompt.
        
        Args:
            prompt (str): The user prompt
            images (List, optional): List of images for the user message
            
        Returns:
            List of messages for the model
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        
        messages = []
        
        # Add system message if system_prompt is set
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        
        # Add user message with optional images
        if images:
            content = _build_message_content(prompt, images)
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=prompt))
            
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
        if not self.model:
            raise ValueError("Model has not been initialized")
            
        try:
            if self.system_prompt or images:
                # Use message-based format when system prompt is set or images are provided
                messages = self._build_messages(prompt, images)
                response = self.model.invoke(messages)
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
            if self.system_prompt or images:
                # Use message-based format when system prompt is set or images are provided
                messages = self._build_messages(prompt, images)
                response = await self.model.ainvoke(messages)
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
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a RemoteLLM instance.
        
        Args:
            model_name (str): The model name to use
            api_key (str): API key for the provider
            base_url (str, optional): Base URL for API requests
            system_prompt (str, optional): System prompt to prepend to all conversations
            **kwargs: Additional provider-specific parameters including temperature and max_tokens
        """
        super().__init__(model_name, system_prompt, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the OpenAI model"""
        # Build config for the model
        model_config = {
            "api_key": self.api_key,
            "model": self.model_name,
        }
        
        # Add base URL if provided
        if self.base_url:
            model_config["base_url"] = self.base_url
            
        # Add any additional kwargs (including temperature and max_tokens)
        try:
            model_config.update(self.kwargs)
            
            # Initialize the ChatOpenAI model
            self.model = ChatOpenAI(**model_config)
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
        **kwargs: Additional provider-specific parameters
        
    Returns:
        BaseLLM: Configured RemoteLLM instance
    """
    if not api_key:
        raise ValueError("API key must be provided")
    
    # Pass sampling parameters as kwargs
    if sampling_params:
        kwargs.update(sampling_params)
    
    return RemoteLLM(
        model_name=model_name,
        api_key=api_key,
        base_url=api_base,
        system_prompt=system_prompt,
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