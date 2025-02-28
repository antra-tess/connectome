"""
ReAct Protocol Implementation
Implements the ReAct (Reasoning, Action, Observation) pattern for tool usage.
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional

from interface.protocol.base_protocol import BaseProtocol
from interface.prompt_library.protocol_prompts import (
    REACT_PROMPT_FORMAT,
    format_tool_description
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReactProtocol(BaseProtocol):
    """
    Implementation of the ReAct (Reasoning, Action, Observation) pattern.
    
    The ReAct pattern instructs the LLM to generate text in this format:
    
    1. Thought: Reasoning about the current situation
    2. Action: tool_name(param1=value1, param2=value2)
    3. Observation: Result of the tool execution
    ...
    Final Answer: Final response to the user
    """
    
    def format_system_prompt(self, base_prompt: str, tools: List[Dict[str, Any]]) -> str:
        """
        Format the system prompt with ReAct pattern instructions and tool descriptions.
        
        Args:
            base_prompt: The base system prompt
            tools: List of available tools with their descriptions
            
        Returns:
            Formatted system prompt with ReAct instructions and tool descriptions
        """
        # Format tool descriptions for the prompt
        tool_descriptions = "\n\n".join([format_tool_description(tool) for tool in tools])
        
        # Format the ReAct instructions with tool descriptions
        react_instructions = REACT_PROMPT_FORMAT.format(tool_descriptions=tool_descriptions)
        
        # Combine base prompt with ReAct instructions
        return f"{base_prompt}\n\n{react_instructions}"
    
    def extract_tool_calls(self, llm_response: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from the LLM response according to the ReAct pattern.
        
        Args:
            llm_response: Raw response from the LLM
            
        Returns:
            List of extracted tool calls with their parameters
        """
        # Regular expression to match ReAct pattern tool calls
        # Format: Action: tool_name(param1=value1, param2=value2)
        action_pattern = r'Action: (\w+)\((.*?)\)'
        
        # Find all matches
        matches = re.findall(action_pattern, llm_response)
        
        tool_calls = []
        for match in matches:
            tool_name = match[0]
            params_str = match[1]
            
            # Parse parameters
            params = {}
            if params_str:
                # Split by commas, but only commas outside of quotes
                param_pairs = self._split_params(params_str)
                
                for param_pair in param_pairs:
                    if '=' in param_pair:
                        key, value = param_pair.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                            
                        params[key] = value
            
            tool_calls.append({
                'name': tool_name,
                'parameters': params
            })
        
        logger.info(f"Extracted {len(tool_calls)} tool calls from LLM response")
        return tool_calls
    
    def format_tool_result(self, tool_name: str, result: Any) -> str:
        """
        Format the result of a tool execution according to the ReAct pattern.
        
        Args:
            tool_name: Name of the tool that was executed
            result: Result returned by the tool
            
        Returns:
            Formatted tool result to be included in the next prompt
        """
        # Format according to ReAct pattern
        return f"Observation: {result}"
    
    def extract_final_response(self, llm_response: str) -> Optional[str]:
        """
        Extract the final user-facing response from the LLM output.
        
        Args:
            llm_response: Raw response from the LLM
            
        Returns:
            Extracted final response or None if no final response found
        """
        # Look for "Final Answer:" pattern
        final_answer_pattern = r'Final Answer:(.*?)(?:\n\s*$|$)'
        match = re.search(final_answer_pattern, llm_response, re.DOTALL)
        
        if match:
            answer = match.group(1).strip()
            logger.info("Final answer extracted from LLM response")
            return answer
        
        # If no "Final Answer:" pattern, check if the whole response might be a direct answer
        # (no tool calls or thought process)
        if not re.search(r'(Thought:|Action:|Observation:)', llm_response):
            logger.info("No ReAct patterns found, treating entire response as final answer")
            return llm_response.strip()
        
        logger.warning("No final answer found in LLM response")
        return None
    
    def format_for_litellm(self, base_prompt: str, messages: List[Dict[str, str]], 
                          tools: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Format the complete request for LiteLLM according to the ReAct pattern.
        
        Args:
            base_prompt: The base system prompt
            messages: List of conversation messages
            tools: List of available tools
            
        Returns:
            Tuple of (formatted_messages, additional_params) where additional_params
            may include protocol-specific parameters for the LLM call
        """
        # First format the system prompt with tool descriptions
        system_prompt = self.format_system_prompt(base_prompt, tools)
        
        # Create the formatted messages array
        formatted_messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add conversation messages
        for msg in messages:
            role = msg.get('role', 'user')
            
            # Map internal roles to LiteLLM roles
            if role == 'assistant':
                formatted_role = 'assistant'
            elif role == 'tool_result':
                # Tool results are added as assistant messages containing observations
                formatted_role = 'assistant'
                msg['content'] = f"Observation: {msg['content']}"
            else:
                formatted_role = 'user'
            
            formatted_messages.append({
                "role": formatted_role,
                "content": msg.get('content', '')
            })
        
        # No additional parameters needed for ReAct
        additional_params = {}
        
        return formatted_messages, additional_params
    
    def _split_params(self, params_str: str) -> List[str]:
        """
        Split parameter string by commas, but respect quoted values.
        
        Args:
            params_str: String containing parameters
            
        Returns:
            List of parameter pairs
        """
        params = []
        current_param = ""
        in_quotes = False
        quote_char = None
        
        for char in params_str:
            if char in ['"', "'"]:
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                current_param += char
            elif char == ',' and not in_quotes:
                params.append(current_param.strip())
                current_param = ""
            else:
                current_param += char
                
        if current_param:
            params.append(current_param.strip())
            
        return params 