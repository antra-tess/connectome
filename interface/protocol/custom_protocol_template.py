"""
Custom Protocol Template

This template can be copied to create new protocol classes.
To create a new protocol:
1. Copy this file with a descriptive name (e.g., my_protocol.py)
2. Rename the class to match your protocol name (e.g., MyProtocol)
3. Implement the required methods
4. Update the configuration to use your new protocol

Your protocol will be automatically discovered by the framework.
"""

from typing import Dict, Any, List, Tuple, Optional
from interface.protocol.base_protocol import BaseProtocol


class CustomProtocolTemplate(BaseProtocol):
    """
    Template for creating custom protocols.
    
    This class serves as a starting point for creating new tool-usage protocols.
    """
    
    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from the LLM response according to the protocol format.
        
        Args:
            text: The LLM response text
            
        Returns:
            List of extracted tool calls, each containing 'name' and 'args' keys
        """
        # Implement your tool call extraction logic here
        # For example, using regex to parse a specific format
        tool_calls = []
        
        # Example implementation (you should replace this with your own logic):
        import re
        pattern = r"TOOL: (\w+)\nARGS: (.+?)(?=\nTOOL:|$)"
        matches = re.findall(pattern, text, re.DOTALL)
        
        for name, args_str in matches:
            # Parse args_str into a dictionary
            args = {}
            for arg_pair in args_str.strip().split('\n'):
                if ':' in arg_pair:
                    key, value = arg_pair.split(':', 1)
                    args[key.strip()] = value.strip()
            
            tool_calls.append({
                'name': name,
                'args': args
            })
        
        return tool_calls
    
    def format_tool_result(self, tool_name: str, result: Any) -> str:
        """
        Format the result of a tool execution according to the protocol.
        
        Args:
            tool_name: Name of the tool that was executed
            result: The result of the tool execution
            
        Returns:
            Formatted tool result
        """
        # Format the result according to your protocol's expected format
        if isinstance(result, dict):
            result_str = "\n".join([f"{key}: {value}" for key, value in result.items()])
        else:
            result_str = str(result)
            
        return f"RESULT FROM {tool_name}:\n{result_str}\n"
    
    def extract_final_response(self, text: str) -> Optional[str]:
        """
        Extract the final response from the LLM output.
        
        Args:
            text: The LLM response text
            
        Returns:
            Extracted final response or None if no final response was found
        """
        # Example implementation (replace with your own logic):
        import re
        match = re.search(r"FINAL RESPONSE:(.*?)$", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def format_for_litellm(self, messages: List[Dict[str, Any]], 
                          tool_descriptions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Format the complete request for LiteLLM according to the protocol.
        
        Args:
            messages: List of message objects
            tool_descriptions: List of tool descriptions
            
        Returns:
            Tuple of (formatted_messages, additional_params) where additional_params 
            may include protocol-specific parameters for the LLM call
        """
        # Implement your formatting logic for LiteLLM
        # This method should prepare the messages for the LiteLLM API
        formatted_messages = messages.copy()
        
        # You might add protocol-specific parameters for the LLM call
        additional_params = {}
        
        # Example: If your protocol needs structured tool descriptions
        # additional_params["tools"] = [...]
        
        return formatted_messages, additional_params
    
    @property
    def protocol_prompt_format(self) -> str:
        """
        Get the prompt format for the protocol.
        
        Returns:
            Protocol-specific prompt format
        """
        return """
TOOL USAGE INSTRUCTIONS:
To use a tool, format your response as follows:

TOOL: tool_name
ARGS:
param1: value1
param2: value2

Wait for the result before proceeding.

RESULT FROM tool_name:
(tool execution result will appear here)

To provide your final response to the user:

FINAL RESPONSE:
Your response to the user here.

Available tools:
{tool_descriptions}
"""


# Notes:
# 1. Do not instantiate your protocol class here
# 2. Your protocol will be discovered automatically
# 3. Configuration in config.py will determine if and how your protocol is used 