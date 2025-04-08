"""
Anthropic Claude LLM Provider Implementation
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional, Union

# Use AsyncAnthropic
from anthropic import AsyncAnthropic, APIError 
from anthropic.types import Message, ToolUseBlock

from .provider_interface import (
    LLMProvider,
    LLMMessage,
    LLMToolDefinition,
    LLMToolCall,
    LLMResponse
)

logger = logging.getLogger(__name__)

class AnthropicProvider(LLMProvider):
    """LLM Provider implementation for Anthropic's Claude API."""

    DEFAULT_MODEL = "claude-3-opus-20240229" # Or specify another default

    def __init__(self, api_key: Optional[str] = None, default_model: Optional[str] = None):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key. If None, attempts to read from ANTHROPIC_API_KEY env var.
            default_model: Default model to use if not specified in calls.
        """
        resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_api_key:
            raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set.")
            
        self.async_client = AsyncAnthropic(api_key=resolved_api_key)
        self.default_model = default_model or self.DEFAULT_MODEL
        logger.info(f"AnthropicProvider initialized with default model: {self.default_model}")

    async def complete(self, 
                 messages: List[LLMMessage],
                 model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = 1024, # Anthropic requires max_tokens
                 tools: Optional[List[LLMToolDefinition]] = None,
                 **kwargs) -> LLMResponse:
        """
        Generate a completion using the Anthropic Claude API (asynchronously).
        """
        target_model = model or self.default_model
        
        # Separate system prompt (if any)
        system_prompt: Optional[str] = None
        if messages and messages[0].role == "system":
            system_prompt = messages[0].content
            anthropic_messages = [self.format_message_for_provider(msg) for msg in messages[1:]]
        else:
            anthropic_messages = [self.format_message_for_provider(msg) for msg in messages]
        
        # Format tools for Anthropic API if provided
        anthropic_tools = None
        if tools:
            anthropic_tools = [self.format_tool_for_provider(tool) for tool in tools]
            
        # Prepare arguments
        api_kwargs = {
            "model": target_model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens, # Required by Anthropic
            "temperature": temperature,
        }
        if system_prompt:
             api_kwargs["system"] = system_prompt
        if anthropic_tools:
             api_kwargs["tools"] = anthropic_tools
             # Anthropic doesn't have an explicit "auto" tool choice like OpenAI;
             # it decides based on whether tools are provided.
             
        # Add any extra provider-specific kwargs passed in
        api_kwargs.update(kwargs)
        filtered_kwargs = {k: v for k, v in api_kwargs.items() if v is not None}

        logger.debug(f"Calling Anthropic API async: Model={target_model}, NumMessages={len(anthropic_messages)}, NumTools={len(anthropic_tools or [])}")
        
        try:
            completion: Message = await self.async_client.messages.create(**filtered_kwargs)
            logger.debug(f"Anthropic Response received: StopReason={completion.stop_reason}")
            return self.parse_response(completion)
            
        except APIError as e:
            logger.error(f"Anthropic API error: {e}", exc_info=True)
            return LLMResponse(content=f"(Error: Anthropic API call failed - {e.status_code} {e.type})", finish_reason="error")
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic call: {e}", exc_info=True)
            return LLMResponse(content=f"(Error: Unexpected error during Anthropic call)", finish_reason="error")

    def format_message_for_provider(self, message: LLMMessage) -> Dict[str, Any]:
         """Formats an LLMMessage for the Anthropic API."""
         # TODO: Handle tool results messages correctly if needed by Anthropic API v3
         # This basic version assumes user/assistant text messages.
         if message.role not in ["user", "assistant"]:
              logger.warning(f"Anthropic API expects alternating user/assistant roles. Role '{message.role}' might cause issues.")
         return {"role": message.role, "content": message.content}
         
    def format_tool_for_provider(self, tool: LLMToolDefinition) -> Dict[str, Any]:
        """
        Format our LLMToolDefinition into the Anthropic tool format.
        Anthropic expects {'name': ..., 'description': ..., 'input_schema': ...}
        """
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters # Assume parameters are already in JSON Schema format
        }
    
    def parse_response(self, raw_response: Message) -> LLMResponse:
        """
        Parse the raw Anthropic Message object into our LLMResponse.
        """
        parsed_content: Optional[str] = None
        parsed_tool_calls: Optional[List[LLMToolCall]] = None
        finish_reason = raw_response.stop_reason

        # Anthropic returns content as a list of blocks (e.g., TextBlock, ToolUseBlock)
        assistant_reply_started = False # Flag to gather consecutive text blocks
        text_content_parts = []

        for block in raw_response.content:
            if block.type == "text":
                text_content_parts.append(block.text)
                assistant_reply_started = True # Start gathering text
            elif block.type == "tool_use":
                 if not parsed_tool_calls: parsed_tool_calls = []
                 # Check if it's a valid ToolUseBlock (it should be)
                 if isinstance(block, ToolUseBlock):
                     parsed_tool_calls.append(LLMToolCall(
                         # Need to store tool_use_id if we need to send results back
                         tool_name=block.name,
                         parameters=block.input # Anthropic provides input as dict
                     ))
                 else:
                      logger.warning(f"Received unexpected block type within tool_use: {type(block)}")
                 # Tool use likely implies assistant finished text part, reset flag
                 assistant_reply_started = False 
            else:
                 logger.warning(f"Unsupported content block type from Anthropic: {block.type}")
                 assistant_reply_started = False # Reset on unknown block

        if text_content_parts:
             parsed_content = "".join(text_content_parts)
             
        # Determine overall finish reason - if tool use, Anthropic sets stop_reason="tool_use"
        # We map this back for consistency if needed, or keep Anthropic's reason.
        # finish_reason = "tool_calls" if finish_reason == "tool_use" else finish_reason 
        
        # Parse usage stats
        usage_stats = None
        if raw_response.usage:
             usage_stats = {
                  "prompt_tokens": raw_response.usage.input_tokens,
                  "completion_tokens": raw_response.usage.output_tokens,
                  "total_tokens": raw_response.usage.input_tokens + raw_response.usage.output_tokens
             }

        return LLMResponse(
            content=parsed_content,
            tool_calls=parsed_tool_calls,
            finish_reason=finish_reason,
            usage=usage_stats
        ) 