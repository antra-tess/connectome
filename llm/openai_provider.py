"""
OpenAI LLM Provider Implementation
"""

import os
import logging
import json # Added for parsing tool arguments JSON string
from typing import Dict, Any, List, Optional

# Use AsyncOpenAI for async completion
from openai import AsyncOpenAI, APIError 
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from .provider_interface import (
    LLMProvider,
    LLMMessage,
    LLMToolDefinition,
    LLMToolCall,
    LLMResponse
)

logger = logging.getLogger(__name__)

class OpenAIProvider(LLMProvider):
    """LLM Provider implementation for OpenAI's API (v1+)."""

    DEFAULT_MODEL = "gpt-4o" # Or specify another default like "gpt-3.5-turbo"

    def __init__(self, api_key: Optional[str] = None, default_model: Optional[str] = None):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, attempts to read from OPENAI_API_KEY env var.
            default_model: Default model to use if not specified in calls.
        """
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set.")
            
        # Use AsyncOpenAI client
        self.async_client = AsyncOpenAI(api_key=resolved_api_key) 
        self.default_model = default_model or self.DEFAULT_MODEL
        logger.info(f"OpenAIProvider initialized with default model: {self.default_model}")

    async def complete(self, 
                 messages: List[LLMMessage],
                 model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 tools: Optional[List[LLMToolDefinition]] = None,
                 **kwargs) -> LLMResponse:
        """
        Generate a completion using the OpenAI API (asynchronously).
        """
        target_model = model or self.default_model
        
        # Format messages for OpenAI API
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        # Format tools for OpenAI API if provided
        openai_tools = None
        if tools:
            openai_tools = [self.format_tool_for_provider(tool) for tool in tools]
            
        # Prepare arguments, filtering out None values
        api_kwargs = {
            "model": target_model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if openai_tools:
             api_kwargs["tools"] = openai_tools
             api_kwargs["tool_choice"] = "auto" # Let the model decide
             
        api_kwargs.update(kwargs)
        filtered_kwargs = {k: v for k, v in api_kwargs.items() if v is not None}

        logger.debug(f"Calling OpenAI API async: Model={target_model}, NumMessages={len(openai_messages)}, NumTools={len(openai_tools or [])}")
        
        try:
            # --- Use await with the async client --- 
            completion: ChatCompletion = await self.async_client.chat.completions.create(**filtered_kwargs)
            # -----------------------------------------

            logger.debug(f"OpenAI Response received: FinishReason={completion.choices[0].finish_reason}")
            return self.parse_response(completion)
            
        except APIError as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            return LLMResponse(content=f"(Error: OpenAI API call failed - {e.status_code} {e.type})", finish_reason="error")
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI call: {e}", exc_info=True)
            return LLMResponse(content=f"(Error: Unexpected error during OpenAI call)", finish_reason="error")

    def format_tool_for_provider(self, tool: LLMToolDefinition) -> Dict[str, Any]:
        """
        Format our LLMToolDefinition into the OpenAI tool format.
        OpenAI expects {'type': 'function', 'function': {'name': ..., 'description': ..., 'parameters': ...}}
        """
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters # Assume parameters are already in JSON Schema format
            }
        }
    
    def parse_response(self, raw_response: ChatCompletion) -> LLMResponse:
        """
        Parse the raw OpenAI ChatCompletion object into our LLMResponse.
        """
        if not raw_response or not raw_response.choices:
            return LLMResponse(content="(Error: Invalid or empty OpenAI response)", finish_reason="error")

        choice: Choice = raw_response.choices[0]
        message: ChatCompletionMessage = choice.message

        parsed_content: Optional[str] = message.content
        parsed_tool_calls: Optional[List[LLMToolCall]] = None

        if message.tool_calls:
            parsed_tool_calls = []
            for tool_call in message.tool_calls:
                if not isinstance(tool_call, ChatCompletionMessageToolCall):
                    logger.warning(f"Skipping unexpected item in tool_calls: {type(tool_call)}")
                    continue
                    
                if tool_call.type == 'function':
                    try:
                        # OpenAI returns parameters as a JSON string, needs parsing
                        params_dict = json.loads(tool_call.function.arguments)
                        parsed_tool_calls.append(LLMToolCall(
                            tool_name=tool_call.function.name,
                            parameters=params_dict
                        ))
                    except json.JSONDecodeError as e:
                         logger.error(f"Failed to parse JSON arguments for tool call {tool_call.function.name}: {e}. Raw args: {tool_call.function.arguments}")
                         # How to handle? Skip tool call? Add error message?
                         # Add an error message in content? Modify finish_reason?
                         # Let's add a note to content for now and keep finish_reason
                         error_note = f" (Note: Failed to parse args for tool '{tool_call.function.name}')"
                         parsed_content = (parsed_content or "") + error_note
                else:
                     logger.warning(f"Unsupported tool call type from OpenAI: {tool_call.type}")
        
        # Parse usage stats if available
        usage_stats = None
        if raw_response.usage:
             usage_stats = {
                  "prompt_tokens": raw_response.usage.prompt_tokens,
                  "completion_tokens": raw_response.usage.completion_tokens,
                  "total_tokens": raw_response.usage.total_tokens
             }

        return LLMResponse(
            content=parsed_content,
            tool_calls=parsed_tool_calls,
            finish_reason=choice.finish_reason,
            usage=usage_stats
        )

# Example Usage (Conceptual - would happen during HUD initialization/injection)
# try:
#     openai_provider = OpenAIProvider()
#     # Inject openai_provider into HUDComponent
# except ValueError as e:
#     print(f"Failed to initialize OpenAI Provider: {e}") 