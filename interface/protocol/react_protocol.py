"""
ReAct Protocol Implementation
Implements the ReAct (Reasoning, Action, Observation) pattern for tool usage.
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional

from interface.protocol.base_protocol import BaseProtocol

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
    
    @property
    def protocol_prompt_format(self) -> str:
        """
        Get the prompt format for ReAct protocol.
        
        Returns:
            Tool usage instructions for ReAct protocol
        """
        return """
REASONING AND ACTING INSTRUCTIONS:
- First, think through a problem step by step.
- Use the Action/Observation pattern when you need to use tools.
- Format for using tools:
  Action: tool_name(param1="value1", param2="value2")
  
- After each Action, wait for an Observation with the tool result.
- Once you have all the information, provide your Final Answer.

Example:
Thought: I need to find information about X.
Action: search_web(query="information about X")
Observation: [Result from search tool]
Thought: Now I know about X. The user asked about Y which relates to X...
Final Answer: Based on the information, Y is...
"""

    def format_tools(self, tools: List[Dict[str, Any]]) -> str:
        """
        Format tools for the ReAct protocol (descriptive text format).
        
        Args:
            tools: List of tool descriptions
            
        Returns:
            Formatted tool descriptions as a string for ReAct prompting
        """
        if not tools:
            return ""
        
        descriptions = ["# Available Tools\n"]
        
        for tool in tools:
            tool_name = tool.get("name", "")
            tool_desc = tool.get("description", "")
            
            descriptions.append(f"## {tool_name}\n{tool_desc}")
            
            parameters = tool.get("parameters", {})
            if parameters:
                descriptions.append("\nParameters:")
                for param_name, param_desc in parameters.items():
                    descriptions.append(f"- {param_name}: {param_desc}")
            
            # Add usage example
            param_names = list(parameters.keys())
            if param_names:
                param_str = ", ".join([f"{param}='value'" for param in param_names])
                descriptions.append(f"\nUsage: Action: {tool_name}({param_str})")
            else:
                descriptions.append(f"\nUsage: Action: {tool_name}()")
            
            descriptions.append("\n")
        
        return "\n".join(descriptions)