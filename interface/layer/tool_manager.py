"""
Tool Manager Component

Handles the registration and execution of privileged tools for the interface layer.
"""

import functools
import logging
from typing import Dict, Any, Optional, Callable, List

# Configure logging
logger = logging.getLogger(__name__)


class PrivilegedToolManager:
    """
    Manages privileged tools that operate directly on the interface layer.
    
    Privileged tools are different from environment tools as they have direct
    access to the interface layer's functionality and can operate on the agent's
    cognitive process itself.
    """
    
    def __init__(self):
        """Initialize the privileged tool manager."""
        self.privileged_tools: Dict[str, Dict[str, Any]] = {}
    
    def register_tool(self, 
                      name: str, 
                      func: Callable,
                      description: str = "", 
                      parameter_descriptions: Optional[Dict[str, str]] = None) -> str:
        """
        Register a privileged tool.
        
        Args:
            name: Name of the tool
            func: Function to call when the tool is executed
            description: Description of the tool
            parameter_descriptions: Descriptions of the tool's parameters
            
        Returns:
            Name of the registered tool
        """
        if parameter_descriptions is None:
            parameter_descriptions = {}
            
        # Store the tool metadata
        self.privileged_tools[name] = {
            "function": func,
            "description": description,
            "parameter_descriptions": parameter_descriptions
        }
        
        logger.info(f"Registered privileged tool: {name}")
        return name
    
    def register_tool_decorator(self, 
                               name: str, 
                               description: str = "", 
                               parameter_descriptions: Optional[Dict[str, str]] = None) -> Callable:
        """
        Decorator for registering privileged tools.
        
        Args:
            name: Name of the tool
            description: Description of the tool
            parameter_descriptions: Descriptions of the tool's parameters
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            # Create tool metadata
            self.register_tool(
                name=name,
                func=func,
                description=description,
                parameter_descriptions=parameter_descriptions
            )
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a privileged tool.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Result of the tool execution
            
        Raises:
            ValueError: If the tool is not found
        """
        if tool_name not in self.privileged_tools:
            raise ValueError(f"Privileged tool '{tool_name}' not found")
        
        tool = self.privileged_tools[tool_name]
        
        try:
            result = tool["function"](**kwargs)
            logger.info(f"Executed privileged tool: {tool_name}")
            return result
        except Exception as e:
            logger.error(f"Error executing privileged tool '{tool_name}': {str(e)}")
            raise
    
    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        Get descriptions of all privileged tools.
        
        Returns:
            List of tool descriptions
        """
        tool_descriptions = []
        
        for name, tool in self.privileged_tools.items():
            # Format the parameters for the description
            params = {}
            for param_name, param_desc in tool["parameter_descriptions"].items():
                params[param_name] = param_desc
                
            tool_descriptions.append({
                "name": name,
                "description": tool["description"],
                "parameters": params
            })
            
        return tool_descriptions 