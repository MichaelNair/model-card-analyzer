"""
LiteLLM Driver Module

A flexible driver for interacting with various LLM APIs through LiteLLM.
Supports text queries, vision models with image inputs, and custom parameters.
"""

from typing import Optional, Dict, Any, List, Union
import base64
import os
import warnings
from pathlib import Path
from litellm import completion
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress Pydantic serialization warnings from LiteLLM
# These are harmless warnings about mismatched field counts in nested response objects
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


class LiteLLMDriver:
    """
    A driver class for interacting with LLM APIs through LiteLLM.
    
    Supports multiple model providers with a unified interface.
    """
    
    def __init__(self):
        """Initialize the LiteLLM driver."""
        pass
    
    def query(
        self,
        model_name: str,
        query_text: str,
        temperature: float = 0.3,
        image_path: Optional[str] = None,
        additional_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a query to an LLM model via LiteLLM.
        
        Args:
            model_name: The name of the model (e.g., 'gpt-4', 'claude-3-opus-20240229')
            query_text: The text query/prompt to send to the model
            temperature: Temperature value for response randomness (0.0-1.0+)
            image_path: Optional path to an image file for vision models
            additional_args: Optional dictionary of additional arguments to pass to the API
                           (e.g., max_tokens, top_p, frequency_penalty, etc.)
        
        Returns:
            Dictionary containing the response and metadata
            
        Raises:
            FileNotFoundError: If image_path is provided but file doesn't exist
            Exception: For API errors or other issues
        """
        try:
            # Build the messages list
            messages = self._build_messages(query_text, image_path)
            
            # Prepare arguments for the completion call
            completion_args = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature
            }
            
            # Add any additional arguments
            if additional_args:
                completion_args.update(additional_args)
            
            # Make the API call
            response = completion(**completion_args)
            
            # Format and return the response
            return self._format_response(response)
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image file not found: {e}")
        except Exception as e:
            raise Exception(f"Error querying LLM: {str(e)}")
    
    def _build_messages(
        self,
        query_text: str,
        image_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Build the messages array for the API call.
        
        Args:
            query_text: The text query
            image_path: Optional path to an image
            
        Returns:
            List of message dictionaries
        """
        if image_path:
            # For vision models, include the image in the content
            # Handle image encoding
            image_data = self._encode_image(image_path)
            
            content = [
                {
                    "type": "image_url",
                    "image_url": image_data
                },
                {"type": "text", "text": query_text}
                
            ]
            
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
        else:
            # Standard text-only query
            messages = [
                {
                    "role": "user",
                    "content": query_text
                }
            ]
        
        return messages
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64 or return URL.
        
        Args:
            image_path: Path to the image file or URL
            
        Returns:
            Image data in appropriate format
        """
        # Check if it's a URL
        if image_path.startswith(('http://', 'https://')):
            return image_path
        
        # Otherwise, treat as local file
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Read and encode the image
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        # Get the file extension for mime type
        ext = path.suffix.lower().lstrip('.')
        mime_type = f"image/{ext}" if ext in ['jpeg', 'jpg', 'png', 'gif', 'webp'] else "image/jpeg"
        
        return f"data:{mime_type};base64,{encoded_string}"
    
    def _format_response(self, response: Any) -> Dict[str, Any]:
        """
        Format the LiteLLM response into a standardized dictionary.
        
        Args:
            response: The raw response from LiteLLM
            
        Returns:
            Formatted response dictionary with essential fields
        """
        # Extract the essential fields directly from the response object
        # Following LiteLLM best practices from DataCamp tutorial
        
        # Extract message content from first choice
        content = response.choices[0].message.content if response.choices else None
        
        # Extract finish reason
        finish_reason = response.choices[0].finish_reason if response.choices else None
        
        # Extract model name
        model = response.model
        
        # Extract token usage statistics
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        # Build the formatted response dictionary
        response_dict = {
            "content": content,
            "model": model,
            "usage": usage,
            "finish_reason": finish_reason,
        }
        
        # Include optional metadata if available
        if hasattr(response, 'id') and response.id:
            response_dict["id"] = response.id
        if hasattr(response, 'created') and response.created:
            response_dict["created"] = response.created
        
        return response_dict


# Convenience function for quick queries
def quick_query(
    model_name: str,
    query_text: str,
    temperature: float = 0.7,
    image_path: Optional[str] = None,
    additional_args: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convenience function for quick queries that returns just the text response.
    
    Args:
        model_name: The name of the model
        query_text: The text query/prompt
        temperature: Temperature value (default: 0.7)
        image_path: Optional path to an image file
        additional_args: Optional additional arguments
        
    Returns:
        The text content of the response
    """
    driver = LiteLLMDriver()
    response = driver.query(model_name, query_text, temperature, image_path, additional_args)
    return response["content"]


# Example usage
if __name__ == "__main__":
    # Initialize the driver
    driver = LiteLLMDriver()
    
    # Example 1: Simple text query
    # print("Example 1: Simple text query")
    # try:
    #     response = driver.query(
    #         model_name="gpt-5.2",
    #         query_text="What is the capital of France?",
    #         temperature=0.5
    #     )
    #     print(f"Response: {response['content']}")
    #     print(f"Tokens used: {response['usage']['total_tokens']}\n")
    # except Exception as e:
    #     print(f"Error: {e}\n")
    
    # Example 2: Query with additional arguments
    print("Example 2: Query with additional arguments")
    try:
        response = driver.query(
            model_name="gpt-5.2",
            query_text="Write a haiku about the summer",
            temperature=0.9,
            additional_args={
                "max_tokens": 100
            }
        )
        print(f"Response: {response['content']}")
        print(f"Tokens used: {response['usage']['total_tokens']}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 3: Vision model with image (uncomment if you have an image)
    # print("Example 3: Vision model with image")
    # try:
    #     response = driver.query(
    #         model_name="gpt-5.2",
    #         query_text="What's in this image?",
    #         temperature=0.7,
    #         image_path="path/to/your/image.jpg"
    #     )
    #     print(f"Response: {response['content']}\n")
    # except Exception as e:
    #     print(f"Error: {e}\n")
    
    # Example 4: Using the convenience function
    print("Example 4: Using quick_query convenience function")
    try:
        response_text = quick_query(
            model_name="gpt-5.2",
            query_text="Say hello in 3 languages",
            temperature=0.5
        )
        print(f"Response: {response_text}\n")
    except Exception as e:
        print(f"Error: {e}\n")
