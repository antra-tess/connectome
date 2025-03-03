"""
Base Protocol for Tool Usage
Provides an abstract base class for implementing different tool-usage protocols.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from interface.prompt_library.protocol_prompts import format_tool_description


class BaseProtocol(ABC):
    """
    Abstract base class for tool-usage protocols.
    
    Different protocols (ReAct, Function Calling, etc.) can be implemented
    as subclasses of this class, ensuring a consistent interface.
    """
    
    def __init__(self, model_info: Dict[str, Any] = None):
        """
        Initialize the protocol with model information.
        
        Args:
            model_info: Dictionary containing information about the LLM model being used
        """
        self.model_info = model_info or {}
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging for this protocol instance."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initialized {self.__class__.__name__} with model {self.model_info.get('model', 'unknown')}")
    
    def format_system_prompt(self, base_prompt: str, tools: List[Dict[str, Any]]) -> str:
        """
        Format the system prompt with tool descriptions according to the protocol.
        
        Args:
            base_prompt: The base system prompt
            tools: List of available tools with their descriptions
            
        Returns:
            Formatted system prompt with protocol instructions and tool descriptions
        """
        # Format tool descriptions for the prompt
        tool_descriptions = "\n\n".join([format_tool_description(tool) for tool in tools])
        
        # Format the protocol instructions with tool descriptions
        protocol_instructions = self.protocol_prompt_format.format(tool_descriptions=tool_descriptions)
        
        # Combine base prompt with protocol instructions
        return f"{base_prompt}\n\n{protocol_instructions}"
    
    @abstractmethod
    def extract_tool_calls(self, llm_response: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from the LLM response according to the protocol format.
        
        Args:
            llm_response: Raw response from the LLM
            
        Returns:
            List of extracted tool calls with their parameters
        """
        pass
    
    @abstractmethod
    def format_tool_result(self, tool_name: str, result: Any) -> str:
        """
        Format the result of a tool execution according to the protocol.
        
        Args:
            tool_name: Name of the tool that was executed
            result: Result returned by the tool
            
        Returns:
            Formatted tool result to be included in the next prompt
        """
        pass
    
    @abstractmethod
    def extract_final_response(self, llm_response: str) -> Optional[str]:
        """
        Extract the final user-facing response from the LLM output.
        
        Args:
            llm_response: Raw response from the LLM
            
        Returns:
            Extracted final response or None if no final response found
        """
        pass
    
    @abstractmethod
    def format_for_litellm(self, base_prompt: str, messages: List[Dict[str, str]], 
                           tools: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Format the complete request for LiteLLM according to the protocol.
        
        Args:
            base_prompt: The base system prompt
            messages: List of conversation messages
            tools: List of available tools
            
        Returns:
            Tuple of (prompt, additional_params) where additional_params
            may include protocol-specific parameters for the LLM call
        """
        pass 

    @abstractmethod
    @property
    def protocol_prompt_format(self) -> str:
        """
        Get the prompt format for the protocol.
        """
        pass