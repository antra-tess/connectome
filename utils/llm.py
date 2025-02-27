"""
LiteLLM Utilities
Provides integration with LiteLLM for making LLM API calls.
"""

import logging
from typing import Dict, Any, List, Optional

import litellm
from litellm import completion

from config import LLM_API_KEY, LLM_MODEL, LLM_PROVIDER, LLM_BASE_URL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_litellm() -> None:
    """
    Initialize LiteLLM with configuration from environment variables.
    """
    # Set LiteLLM API key
    litellm.api_key = LLM_API_KEY
    
    # Set custom base URL if provided
    if LLM_BASE_URL:
        litellm.api_base = LLM_BASE_URL
    
    # Set any other LiteLLM configurations here
    
    logger.info(f"LiteLLM initialized with model: {LLM_MODEL}, provider: {LLM_PROVIDER}")


def send_to_llm(messages: List[Dict[str, str]], 
               additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Send a request to the LLM using LiteLLM.
    
    Args:
        messages: List of messages in the conversation 
                 (each with 'role' and 'content')
        additional_params: Optional additional parameters for the LLM call
        
    Returns:
        Raw response from the LLM
    """
    try:
        # Combine default parameters with additional parameters
        params = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        
        # Add any additional parameters
        if additional_params:
            params.update(additional_params)
        
        # Log the request (excluding sensitive data)
        logger.info(f"Sending request to LLM: model={LLM_MODEL}, "
                   f"message_count={len(messages)}")
        
        # Call LiteLLM
        response = completion(**params)
        
        # Extract the response content
        if hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                content = response.choices[0].message.content
            else:
                content = str(response.choices[0])
        else:
            content = str(response)
        
        # Log the response (truncated for brevity)
        logger.info(f"Received response from LLM: {content[:100]}...")
        
        return content
    
    except Exception as e:
        logger.error(f"Error calling LLM: {str(e)}")
        # Return a graceful error message
        return f"I apologize, but I encountered an error while processing your request: {str(e)}"


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the current LLM model.
    
    Returns:
        Dictionary with model information
    """
    return {
        "model": LLM_MODEL,
        "provider": LLM_PROVIDER,
        "has_tool_calling": _model_supports_tool_calling(LLM_MODEL),
    }


def _model_supports_tool_calling(model_name: str) -> bool:
    """
    Check if the specified model supports native function/tool calling.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if the model supports tool calling, False otherwise
    """
    # This is a simple heuristic based on known models
    # In a production environment, you might want to maintain a list or check with the provider's API
    
    # Most OpenAI models with function calling support
    if any(model in model_name.lower() for model in [
        "gpt-4", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", 
        "gpt-4-0613", "gpt-4-1106-preview", "gpt-4-turbo"
    ]):
        return True
    
    # Anthropic Claude models with function calling support
    if "claude" in model_name.lower() and any(version in model_name.lower() for version in [
        "opus", "sonnet", "3", "3.5"
    ]):
        return True
    
    # Add other models as they become available with function calling
    
    return False 