"""
LiteLLM Provider Implementation

Concrete implementation of the LLM provider interface using LiteLLM.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union

try:
    import litellm
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

from bot_framework.llm.provider_interface import (
    LLMProvider, 
    LLMMessage, 
    LLMToolDefinition,
    LLMToolCall,
    LLMResponse
)


class LiteLLMProvider(LLMProvider):
    """LiteLLM implementation of LLMProvider."""
    
    def __init__(self, default_model: str = "gpt-4", **kwargs):
        """
        Initialize the LiteLLM provider.
        
        Args:
            default_model: The default model to use
            **kwargs: Additional configuration for LiteLLM
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "LiteLLM is not installed. Please install it with: pip install litellm"
            )
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.default_model = default_model
        self.config = kwargs
        
        # Apply any custom configuration
        for key, value in kwargs.items():
            if hasattr(litellm, key):
                setattr(litellm, key, value)
                
        self.logger.info(f"LiteLLM provider initialized with default model: {default_model}")
    
    def complete(self, 
                 messages: List[LLMMessage],
                 model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 tools: Optional[List[LLMToolDefinition]] = None,
                 **kwargs) -> LLMResponse:
        """
        Generate a completion using LiteLLM.
        
        Args:
            messages: The conversation history
            model: The model to use (defaults to provider's default)
            temperature: The sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional list of tools that the model can use
            **kwargs: Additional provider-specific parameters
            
        Returns:
            A response from the LLM
        """
        self.logger.info(f"Generating completion with model: {model or self.default_model}")
        
        # Format messages for LiteLLM
        litellm_messages = self._format_messages(messages)
        
        # Format tools if provided
        litellm_functions = None
        if tools:
            litellm_functions = [self.format_tool_for_provider(tool) for tool in tools]
        
        # Prepare parameters
        params = {
            "model": model or self.default_model,
            "messages": litellm_messages,
        }
        
        # Add optional parameters if provided
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if litellm_functions:
            params["functions"] = litellm_functions
        
        # Add any additional kwargs
        params.update(kwargs)
        
        try:
            # Call LiteLLM
            response = completion(**params)
            return self.parse_response(response)
        except Exception as e:
            self.logger.error(f"Error generating completion: {e}")
            return LLMResponse(
                content=f"Error: {str(e)}",
                finish_reason="error"
            )
    
    def format_tool_for_provider(self, tool: LLMToolDefinition) -> Dict[str, Any]:
        """
        Format a tool definition for LiteLLM.
        
        Args:
            tool: Tool definition
            
        Returns:
            LiteLLM-compatible function definition
        """
        # Convert our tool format to LiteLLM function format
        function_def = {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
        
        # Add parameters
        required_params = []
        for param_name, param_info in tool.parameters.items():
            if isinstance(param_info, dict):
                function_def["parameters"]["properties"][param_name] = param_info
                if param_info.get("required", False):
                    required_params.append(param_name)
            else:
                # Simple string description
                function_def["parameters"]["properties"][param_name] = {
                    "type": "string",
                    "description": param_info
                }
        
        # Add required parameters list if any were found
        if required_params:
            function_def["parameters"]["required"] = required_params
            
        return function_def
    
    def parse_response(self, raw_response: Any) -> LLMResponse:
        """
        Parse LiteLLM response into standardized LLMResponse.
        
        Args:
            raw_response: LiteLLM response
            
        Returns:
            Standardized LLMResponse
        """
        # Extract the response message
        response_message = raw_response.choices[0].message
        
        # Extract content
        content = response_message.get("content")
        
        # Extract tool calls if present
        tool_calls = []
        if "function_call" in response_message:
            # Single function call format
            function_call = response_message["function_call"]
            tool_name = function_call["name"]
            
            # Parse arguments (handle potential JSON parsing errors)
            try:
                parameters = json.loads(function_call["arguments"])
            except json.JSONDecodeError:
                parameters = {"raw_arguments": function_call["arguments"]}
                
            tool_calls.append(LLMToolCall(tool_name=tool_name, parameters=parameters))
        
        # Handle newer format with multiple tool calls
        elif "tool_calls" in response_message:
            for tool_call in response_message["tool_calls"]:
                if tool_call["type"] == "function":
                    function_call = tool_call["function"]
                    tool_name = function_call["name"]
                    
                    # Parse arguments
                    try:
                        parameters = json.loads(function_call["arguments"])
                    except json.JSONDecodeError:
                        parameters = {"raw_arguments": function_call["arguments"]}
                        
                    tool_calls.append(LLMToolCall(tool_name=tool_name, parameters=parameters))
        
        # Extract finish reason
        finish_reason = raw_response.choices[0].finish_reason
        
        # Extract usage info if available
        usage = None
        if hasattr(raw_response, "usage"):
            usage = {
                "prompt_tokens": raw_response.usage.prompt_tokens,
                "completion_tokens": raw_response.usage.completion_tokens,
                "total_tokens": raw_response.usage.total_tokens
            }
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage
        )
    
    def _format_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """
        Format messages for LiteLLM.
        
        Args:
            messages: List of LLMMessage objects
            
        Returns:
            List of LiteLLM-compatible message dictionaries
        """
        litellm_messages = []
        
        for message in messages:
            msg_dict = {
                "role": message.role,
                "content": message.content
            }
            
            # Add name if provided (for function messages)
            if message.name:
                msg_dict["name"] = message.name
                
            litellm_messages.append(msg_dict)
            
        return litellm_messages 