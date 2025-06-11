"""
LLM Provider Interface

Abstract base class defining the interface for LLM providers.
This allows the bot framework to switch between different LLM backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union


class LLMMessage:
    """Representation of a message in a conversation with an LLM."""
    
    def __init__(self, role: str, content: Union[str, List[Dict[str, Any]]], name: Optional[str] = None):
        """
        Initialize a message.
        
        Args:
            role: The role of the message sender (e.g., "user", "assistant", "system")
            content: The content of the message. Can be:
                     - str: Plain text content
                     - List[Dict]: Multimodal content with parts like:
                       [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]
            name: Optional name for function calls
        """
        self.role = role
        self.content = content
        self.name = name
    
    def is_multimodal(self) -> bool:
        """Check if this message contains multimodal content."""
        return isinstance(self.content, list)
    
    def get_text_content(self) -> str:
        """Extract text content from the message, regardless of format."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            # Extract all text parts
            text_parts = []
            for part in self.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            return " ".join(text_parts)
        else:
            return str(self.content)
    
    def get_attachment_count(self) -> int:
        """Get the number of non-text attachments in this message."""
        if not isinstance(self.content, list):
            return 0
        return len([part for part in self.content if isinstance(part, dict) and part.get("type") != "text"])


class LLMToolDefinition:
    """Definition of a tool that can be used by an LLM."""
    
    def __init__(self, 
                 name: str,
                 description: str,
                 parameters: Dict[str, Any]):
        """
        Initialize a tool definition.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            parameters: The parameters the tool accepts
        """
        self.name = name
        self.description = description
        self.parameters = parameters

    def __str__(self):
        return f"ToolDefinition(name={self.name}, description={self.description}, parameters={self.parameters})"


class LLMToolCall:
    """Representation of a tool call made by an LLM."""
    
    def __init__(self, 
                 tool_name: str,
                 parameters: Dict[str, Any]):
        """
        Initialize a tool call.
        
        Args:
            tool_name: The name of the tool being called
            parameters: The parameters for the tool call
        """
        self.tool_name = tool_name
        self.parameters = parameters


class LLMResponse:
    """Representation of a response from an LLM."""
    
    def __init__(self, 
                 content: Optional[str] = None,
                 tool_calls: Optional[List[LLMToolCall]] = None,
                 finish_reason: Optional[str] = None,
                 usage: Optional[Dict[str, int]] = None):
        """
        Initialize a response.
        
        Args:
            content: The text content of the response (may be None if tool_calls is provided)
            tool_calls: Optional list of tool calls
            finish_reason: The reason the generation finished (e.g., "stop", "length", "tool_calls")
            usage: Token usage information
        """
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.usage = usage or {}


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def complete(self, 
                 messages: List[LLMMessage],
                 model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 tools: Optional[List[LLMToolDefinition]] = None,
                 **kwargs) -> LLMResponse:
        """
        Generate a completion from the LLM.
        
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
        pass
    
    @abstractmethod
    def format_tool_for_provider(self, tool: Dict[str, Any]) -> Any:
        """
        Format a tool definition for this specific provider.
        
        Args:
            tool: Generic tool definition
            
        Returns:
            Provider-specific tool definition
        """
        pass
    
    @abstractmethod
    def parse_response(self, raw_response: Any) -> LLMResponse:
        """
        Parse raw provider response into standardized LLMResponse.
        
        Args:
            raw_response: Provider-specific response
            
        Returns:
            Standardized LLMResponse
        """
        pass 