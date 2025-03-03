"""
Function Calling Protocol Implementation
Implements protocol for models with native function/tool calling capabilities.
"""

import logging
import json
from typing import Dict, Any, List, Tuple, Optional

from interface.protocol.base_protocol import BaseProtocol

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FunctionCallingProtocol(BaseProtocol):
    """
    Implementation of protocol using native function/tool calling capabilities.
    
    This protocol leverages the model's built-in function calling capabilities
    instead of parsing text-based tool calls like the ReAct pattern.
    """
    
    def extract_tool_calls(self, llm_response: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from the LLM response.
        
        In the function calling protocol, tool calls are typically returned
        as structured data directly from the model API. This method handles
        both JSON-formatted tool calls in the text response and the case where
        tool calls were already extracted by LiteLLM.
        
        Args:
            llm_response: Raw response from the LLM
            
        Returns:
            List of extracted tool calls with their parameters
        """
        # Check if the response is a dictionary (already parsed by LiteLLM)
        if isinstance(llm_response, dict):
            tool_calls = []
            
            # Handle OpenAI format
            if 'function_call' in llm_response:
                function_call = llm_response['function_call']
                
                # Parse parameters
                arguments = function_call.get('arguments', '{}')
                if isinstance(arguments, str):
                    try:
                        parameters = json.loads(arguments)
                    except json.JSONDecodeError:
                        parameters = {"raw_args": arguments}
                else:
                    parameters = arguments
                    
                tool_calls.append({
                    'name': function_call.get('name', 'unknown_function'),
                    'parameters': parameters
                })
                
            # Handle multiple tool calls (OpenAI's tool_calls format)
            elif 'tool_calls' in llm_response:
                for tool_call in llm_response['tool_calls']:
                    function_info = tool_call.get('function', {})
                    
                    # Parse parameters
                    arguments = function_info.get('arguments', '{}')
                    if isinstance(arguments, str):
                        try:
                            parameters = json.loads(arguments)
                        except json.JSONDecodeError:
                            parameters = {"raw_args": arguments}
                    else:
                        parameters = arguments
                        
                    tool_calls.append({
                        'name': function_info.get('name', 'unknown_function'),
                        'parameters': parameters
                    })
                    
            return tool_calls
            
        # If it's a string, try to parse tool calls from the text
        else:
            tool_calls = []
            
            # Try to find JSON blocks that might contain tool calls
            import re
            json_blocks = re.findall(r'```json\n(.*?)\n```', llm_response, re.DOTALL)
            
            for block in json_blocks:
                try:
                    data = json.loads(block)
                    
                    # Look for common formats
                    if 'function' in data and 'arguments' in data['function']:
                        tool_calls.append({
                            'name': data['function'].get('name', 'unknown_function'),
                            'parameters': data['function']['arguments']
                        })
                    elif 'name' in data and 'parameters' in data:
                        tool_calls.append({
                            'name': data['name'],
                            'parameters': data['parameters']
                        })
                except json.JSONDecodeError:
                    continue
                    
            return tool_calls
    
    def format_tool_result(self, tool_name: str, result: Any) -> str:
        """
        Format the result of a tool execution.
        
        Args:
            tool_name: Name of the tool that was executed
            result: Result returned by the tool
            
        Returns:
            Formatted tool result
        """
        # Format the result as a JSON string for the model
        if isinstance(result, (dict, list)):
            result_json = json.dumps(result)
            return f"Tool '{tool_name}' returned: {result_json}"
        else:
            return f"Tool '{tool_name}' returned: {result}"
    
    def extract_final_response(self, llm_response: str) -> Optional[str]:
        """
        Extract the final user-facing response from the LLM output.
        
        Args:
            llm_response: Raw response from the LLM
            
        Returns:
            Extracted final response or None if no final response found
        """
        # Check if the response is a dictionary (already parsed by LiteLLM)
        if isinstance(llm_response, dict):
            # Check if there's a content field (OpenAI format)
            if 'content' in llm_response:
                return llm_response['content']
                
            # Check if it's directly a message with text field (Claude format)
            if 'text' in llm_response:
                return llm_response['text']
                
            # No recognized format
            logger.warning("No recognized response format in LLM output dictionary")
            return None
            
        # If it's a string and doesn't contain tool calls, it's probably the final response
        else:
            # Check if there are any tool calls in the response
            import re
            
            # Look for common function calling patterns
            has_tool_call = any([
                '```json' in llm_response and ('function' in llm_response or 'tool' in llm_response),
                re.search(r'I need to use the .* tool', llm_response, re.IGNORECASE),
                re.search(r'I should call the .* function', llm_response, re.IGNORECASE)
            ])
            
            if has_tool_call:
                # Try to extract just the actual response part
                response_parts = re.split(r'```json.*?```', llm_response, flags=re.DOTALL)
                if response_parts:
                    # Join the non-JSON parts
                    return '\n'.join(part.strip() for part in response_parts if part.strip())
                    
                return None
            else:
                # No tool calls, return the entire response
                return llm_response
    
    def format_for_litellm(self, base_prompt: str, messages: List[Dict[str, str]], 
                           tools: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Format the complete request for LiteLLM with function calling capabilities.
        
        Args:
            base_prompt: The base system prompt
            messages: List of conversation messages
            tools: List of available tools
            
        Returns:
            Tuple of (formatted_messages, additional_params) where additional_params
            includes the function definitions for the LLM
        """
        # Check if model supports tool format or function format
        model_prefers_tool_format = self._prefers_tool_format()
        model_prefers_tool_role = self._prefers_tool_role()
        
        # Format the system prompt
        system_prompt = self.format_system_prompt(base_prompt, tools)
        
        # Create the formatted messages array
        formatted_messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Process conversation messages
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'assistant':
                formatted_role = 'assistant'
            elif role == 'tool_result':
                # Format tool results differently based on model preference
                if model_prefers_tool_role:
                    formatted_role = 'tool'
                    # Keep only the result, not the formatting
                    formatted_messages.append({
                        "role": formatted_role,
                        "content": content
                    })
                    continue
                else:
                    # Use assistant role with special formatting
                    formatted_role = 'assistant'
                    content = f"Tool result: {content}"
            else:
                formatted_role = 'user'
            
            formatted_messages.append({
                "role": formatted_role,
                "content": content
            })
        
        # Prepare additional parameters for function calling
        additional_params = {}
        
        # Convert tools to the format expected by the model
        if tools:
            if model_prefers_tool_format:
                # OpenAI format with "tools"
                openai_tools = []
                
                for tool in tools:
                    # Convert to OpenAI function format
                    function_def = {
                        "type": "function",
                        "function": {
                            "name": tool.get('name', 'unknown_tool'),
                            "description": tool.get('description', ''),
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                        }
                    }
                    
                    # Add parameters
                    for param_name, param_info in tool.get('parameters', {}).items():
                        function_def["function"]["parameters"]["properties"][param_name] = {
                            "type": "string",
                            "description": param_info.get('description', '')
                        }
                        
                        if param_info.get('required', False):
                            function_def["function"]["parameters"]["required"].append(param_name)
                    
                    openai_tools.append(function_def)
                
                additional_params["tools"] = openai_tools
            else:
                # Original OpenAI format with "functions"
                function_defs = []
                
                for tool in tools:
                    # Convert to OpenAI function format
                    function_def = {
                        "name": tool.get('name', 'unknown_tool'),
                        "description": tool.get('description', ''),
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                    
                    # Add parameters
                    for param_name, param_info in tool.get('parameters', {}).items():
                        function_def["parameters"]["properties"][param_name] = {
                            "type": "string",
                            "description": param_info.get('description', '')
                        }
                        
                        if param_info.get('required', False):
                            function_def["parameters"]["required"].append(param_name)
                    
                    function_defs.append(function_def)
                
                additional_params["functions"] = function_defs
        
        return formatted_messages, additional_params
    
    def _prefers_tool_format(self) -> bool:
        """
        Check if the model prefers the newer 'tools' format over 'functions'.
        
        Returns:
            True if the model prefers the 'tools' format, False otherwise
        """
        model_name = self.model_info.get('model', '').lower()
        
        # Models known to use the newer 'tools' format
        tool_format_models = [
            'gpt-4-1106-preview',
            'gpt-4-vision-preview',
            'gpt-4-turbo',
            'gpt-4-0125-preview',
        ]
        
        return any(model in model_name for model in tool_format_models)
    
    def _prefers_tool_role(self) -> bool:
        """
        Check if the model prefers the 'tool' role for tool responses.
        
        Returns:
            True if the model supports the 'tool' role, False otherwise
        """
        model_name = self.model_info.get('model', '').lower()
        
        # Only certain models support the 'tool' role
        tool_role_models = [
            'gpt-4-1106-preview',
            'gpt-4-vision-preview',
            'gpt-4-turbo',
            'gpt-4-0125-preview',
        ]
        
        return any(model in model_name for model in tool_role_models)
    
    @property
    def protocol_prompt_format(self) -> str:
        return """TOOL USAGE INSTRUCTIONS:
You have access to various tools to assist users. The system will handle the formatting of tool calls.

{tool_descriptions}

When you need to use a tool:
1. Clearly indicate that you need to call a specific tool
2. The system will execute the tool and provide the result
3. Use the tool result to inform your final response

IMPORTANT:
- Only call tools when necessary to fulfill the user's request
- You can call multiple tools if needed
- Make sure to provide a final response that addresses the user's request
"""