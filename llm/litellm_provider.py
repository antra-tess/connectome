"""
LiteLLM Provider Implementation

Concrete implementation of the LLM provider interface using LiteLLM.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union

from opentelemetry import trace

try:
    import litellm
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

from llm.provider_interface import (
    LLMProvider,
    LLMMessage,
    LLMToolDefinition,
    LLMToolCall,
    LLMResponse
)

from host.observability import get_tracer
from .scaffolding_observer import ScaffoldingObserver

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

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
        model_to_use = model or self.default_model
        with tracer.start_as_current_span("litellm.complete", attributes={
            "llm.model": model_to_use,
            "llm.prompt.message_count": len(messages),
            "llm.provider": "litellm"
        }) as span:
            self.logger.info(f"Generating completion with model: {model_to_use}")

            # Format messages for LiteLLM
            litellm_messages = self._format_messages(messages)

            # Format tools if provided
            litellm_functions = None
            if tools:
                self.logger.debug(f"Formatting {len(tools)} tools for LiteLLM")
                litellm_functions = []
                for tool in tools:
                    try:
                        # Validate tool first
                        validation_issues = validate_llm_tool_definition(tool)
                        if validation_issues:
                            self.logger.error(f"Tool '{getattr(tool, 'name', 'UNKNOWN')}' validation failed: {'; '.join(validation_issues)}")
                            continue

                        formatted_tool = self.format_tool_for_provider(tool)
                        litellm_functions.append(formatted_tool)
                        self.logger.debug(f"Successfully formatted tool: {tool.name}")
                    except Exception as e:
                        self.logger.error(f"Error formatting tool '{getattr(tool, 'name', 'UNKNOWN')}': {e}")
                        continue

                self.logger.debug(f"Formatted {len(litellm_functions)} tools successfully")
                if self.logger.isEnabledFor(logging.DEBUG):
                    # Only stringify tools for debug if debug logging is enabled
                    self.logger.debug(f"LiteLLM tool definitions: {json.dumps(litellm_functions, indent=2)}")

            # Prepare parameters
            params = {
                "model": model_to_use,
                "messages": litellm_messages,
            }

            # Add optional parameters if provided
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            if litellm_functions:
                params["tools"] = litellm_functions

            # Add any additional kwargs, but filter out scaffolding-specific parameters
            # that are not meant for the underlying LLM provider
            scaffolding_params = {'original_context_data'}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in scaffolding_params}

            # Log filtered parameters for debugging
            if scaffolding_params.intersection(kwargs.keys()):
                filtered_out = scaffolding_params.intersection(kwargs.keys())
                self.logger.debug(f"Filtered out scaffolding-specific parameters: {filtered_out}")

            params.update(filtered_kwargs)

            observer = ScaffoldingObserver()

            try:
                # Add request data to the span
                span.add_event(
                    "LLM Request",
                    attributes={"llm.request.body": json.dumps(params, default=str)}
                )

                # Call ScaffoldingObserver to record a request to LLM
                observer.observe_request(
                    messages,
                    model_to_use,
                    temperature,
                    max_tokens,
                    tools,
                    kwargs.get('original_context_data', None)
                )

                # Call LiteLLM
                response = completion(**params)
                parsed_response = self.parse_response(response)

                # Add response data to the span
                response_log_data = {
                    "content": parsed_response.content,
                    "tool_calls": [tc.__dict__ for tc in parsed_response.tool_calls],
                    "finish_reason": parsed_response.finish_reason,
                    "usage": parsed_response.usage,
                }
                span.add_event(
                    "LLM Response",
                    attributes={"llm.response.body": json.dumps(response_log_data, default=str)}
                )

                # Call ScaffoldingObserver to record a response from LLM
                observer.observe_response(parsed_response.content)

                # Add response attributes to the span
                if parsed_response.usage:
                    span.set_attribute("llm.usage.prompt_tokens", parsed_response.usage.get("prompt_tokens", 0))
                    span.set_attribute("llm.usage.completion_tokens", parsed_response.usage.get("completion_tokens", 0))
                    span.set_attribute("llm.usage.total_tokens", parsed_response.usage.get("total_tokens", 0))

                span.set_attribute("llm.finish_reason", parsed_response.finish_reason or "unknown")
                if parsed_response.tool_calls:
                    span.set_attribute("llm.tool_calls_count", len(parsed_response.tool_calls))

                span.set_status(trace.Status(trace.StatusCode.OK))

                return parsed_response

            except Exception as e:
                self.logger.error(f"Error generating completion: {e}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, f"LLM completion failed: {e}"))
                return LLMResponse(
                    content=f"Error: {str(e)}",
                    finish_reason="error"
                )

    def format_tool_for_provider(self, tool: LLMToolDefinition) -> Dict[str, Any]:
        """
        Format a tool definition for LiteLLM.

        Args:
            tool: Tool definition with JSON schema parameters

        Returns:
            LiteLLM-compatible function definition
        """
        # Validate tool input
        if not isinstance(tool, LLMToolDefinition):
            raise TypeError(f"Expected LLMToolDefinition, got {type(tool)}")

        if not tool.name or not isinstance(tool.name, str):
            raise ValueError(f"Tool name must be a non-empty string, got: {tool.name}")

        if not isinstance(tool.parameters, dict):
            raise ValueError(f"Tool parameters must be a dict (JSON schema), got {type(tool.parameters)}: {tool.parameters}")

        # tool.parameters should be a JSON schema like:
        # {
        #   "type": "object",
        #   "properties": {
        #     "param_name": {"type": "string", "description": "..."},
        #     ...
        #   },
        #   "required": ["param1", "param2"]
        # }

        # Build LiteLLM function definition
        function_def = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters.copy()  # Use the JSON schema directly
            }
        }

        # Validate the JSON schema structure
        if "type" not in tool.parameters:
            self.logger.warning(f"Tool '{tool.name}' parameters missing 'type' field, adding 'object'")
            function_def["function"]["parameters"]["type"] = "object"

        if "properties" not in tool.parameters:
            self.logger.warning(f"Tool '{tool.name}' parameters missing 'properties' field, adding empty object")
            function_def["function"]["parameters"]["properties"] = {}

        # Ensure required is a list if present
        if "required" in tool.parameters and not isinstance(tool.parameters["required"], list):
            self.logger.warning(f"Tool '{tool.name}' has non-list 'required' field: {tool.parameters['required']}")
            function_def["function"]["parameters"]["required"] = []

        # Validate the function definition is JSON serializable
        try:
            json.dumps(function_def)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Tool '{tool.name}' produced non-JSON-serializable definition: {e}")

        return function_def

    def parse_response(self, raw_response: Any) -> LLMResponse:
        """
        Parse LiteLLM response into standardized LLMResponse.

        Args:
            raw_response: LiteLLM response

        Returns:
            Standardized LLMResponse
        """
        try:
            # Validate response structure
            if not hasattr(raw_response, 'choices') or not raw_response.choices:
                self.logger.error(f"Invalid response structure: missing or empty choices")
                return LLMResponse(
                    content="Error: Invalid response structure from LLM",
                    finish_reason="error"
                )

            # Extract the response message
            response_message = raw_response.choices[0].message

            # Extract content - handle both dict and object formats
            if hasattr(response_message, 'content'):
                content = response_message.content
            elif isinstance(response_message, dict):
                content = response_message.get("content")
            else:
                content = str(response_message) if response_message else None

            # Extract tool calls if present
            tool_calls = []

            # Handle single function call format (older OpenAI format)
            function_call = None
            if hasattr(response_message, 'function_call'):
                function_call = response_message.function_call
            elif isinstance(response_message, dict) and "function_call" in response_message:
                function_call = response_message["function_call"]

            if function_call:
                tool_name = function_call.get("name") if isinstance(function_call, dict) else getattr(function_call, "name", None)
                arguments = function_call.get("arguments") if isinstance(function_call, dict) else getattr(function_call, "arguments", None)

                if tool_name and arguments:
                    # Parse arguments (handle potential JSON parsing errors)
                    try:
                        if isinstance(arguments, str):
                            parameters = json.loads(arguments)
                        elif isinstance(arguments, dict):
                            parameters = arguments
                        else:
                            parameters = {"raw_arguments": str(arguments)}
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse function call arguments as JSON: {e}")
                        parameters = {"raw_arguments": str(arguments)}

                    tool_calls.append(LLMToolCall(tool_name=tool_name, parameters=parameters))

            # Handle newer format with multiple tool calls
            tool_calls_list = None
            if hasattr(response_message, 'tool_calls'):
                tool_calls_list = response_message.tool_calls
            elif isinstance(response_message, dict) and "tool_calls" in response_message:
                tool_calls_list = response_message["tool_calls"]

            if tool_calls_list:
                for tool_call in tool_calls_list:
                    if isinstance(tool_call, dict):
                        tool_type = tool_call.get("type")
                        if tool_type == "function" and "function" in tool_call:
                            function_call = tool_call["function"]
                            tool_name = function_call.get("name")
                            arguments = function_call.get("arguments")
                    else:
                        # Handle object format
                        tool_type = getattr(tool_call, "type", None)
                        if tool_type == "function" and hasattr(tool_call, "function"):
                            function_call = tool_call.function
                            tool_name = getattr(function_call, "name", None)
                            arguments = getattr(function_call, "arguments", None)
                        else:
                            continue

                    if tool_name and arguments is not None:
                        # Parse arguments
                        try:
                            if isinstance(arguments, str):
                                parameters = json.loads(arguments)
                            elif isinstance(arguments, dict):
                                parameters = arguments
                            else:
                                parameters = {"raw_arguments": str(arguments)}
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse tool call arguments as JSON: {e}")
                            parameters = {"raw_arguments": str(arguments)}

                        tool_calls.append(LLMToolCall(tool_name=tool_name, parameters=parameters))

            # Extract finish reason
            finish_reason = None
            if hasattr(raw_response.choices[0], 'finish_reason'):
                finish_reason = raw_response.choices[0].finish_reason
            elif isinstance(raw_response.choices[0], dict):
                finish_reason = raw_response.choices[0].get('finish_reason')

            # Extract usage info if available
            usage = None
            if hasattr(raw_response, "usage") and raw_response.usage:
                usage_obj = raw_response.usage
                usage = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                    "total_tokens": getattr(usage_obj, "total_tokens", 0)
                }

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage or {}
            )

        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}", exc_info=True)
            return LLMResponse(
                content=f"Error parsing response: {str(e)}",
                finish_reason="error"
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
                "content": message.content  # Can now be string or list for multimodal
            }

            # Add name if provided (for function messages)
            if message.name:
                msg_dict["name"] = message.name

            litellm_messages.append(msg_dict)

        return litellm_messages

def validate_llm_tool_definition(tool: LLMToolDefinition) -> List[str]:
    """
    Validate an LLMToolDefinition and return a list of issues found.

    Args:
        tool: The tool definition to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []

    if not isinstance(tool, LLMToolDefinition):
        issues.append(f"Expected LLMToolDefinition, got {type(tool)}")
        return issues

    # Check name
    if not tool.name or not isinstance(tool.name, str):
        issues.append(f"Tool name must be a non-empty string, got: {repr(tool.name)}")

    # Check description
    if tool.description is not None and not isinstance(tool.description, str):
        issues.append(f"Tool description must be a string or None, got: {type(tool.description)}")

    # Check parameters
    if not isinstance(tool.parameters, dict):
        issues.append(f"Tool parameters must be a dict (JSON schema), got {type(tool.parameters)}")
        return issues

    # Check JSON schema structure
    if "type" not in tool.parameters:
        issues.append("Parameters schema missing 'type' field")
    elif tool.parameters["type"] != "object":
        issues.append(f"Parameters schema 'type' should be 'object', got: {tool.parameters['type']}")

    if "properties" not in tool.parameters:
        issues.append("Parameters schema missing 'properties' field")
    elif not isinstance(tool.parameters["properties"], dict):
        issues.append(f"Parameters 'properties' must be a dict, got {type(tool.parameters['properties'])}")

    if "required" in tool.parameters:
        if not isinstance(tool.parameters["required"], list):
            issues.append(f"Parameters 'required' must be a list, got {type(tool.parameters['required'])}")
        elif tool.parameters["properties"]:
            # Check that required params exist in properties
            properties = tool.parameters["properties"]
            for req_param in tool.parameters["required"]:
                if req_param not in properties:
                    issues.append(f"Required parameter '{req_param}' not found in properties")

    # Test JSON serialization
    try:
        json.dumps(tool.parameters)
    except (TypeError, ValueError) as e:
        issues.append(f"Parameters not JSON serializable: {e}")

    return issues