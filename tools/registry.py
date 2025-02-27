"""
Tool Registry
Implements the tool registration and execution logic using a decorator pattern.
"""

import functools
import inspect
from typing import Dict, Any, Callable, List, Optional


class ToolRegistry:
    """
    Registry for tools that can be used by the agent.
    
    Provides a decorator for registering tools and methods for listing
    and executing registered tools.
    """
    
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: Optional[str] = None, 
                description: str = "", 
                parameter_descriptions: Optional[Dict[str, str]] = None) -> Callable:
        """
        Decorator for registering a function as a tool.
        
        Args:
            name: Optional custom name for the tool (defaults to function name)
            description: Description of what the tool does
            parameter_descriptions: Optional descriptions for each parameter
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            
            # Get parameter info from function signature
            sig = inspect.signature(func)
            parameters = {}
            
            for param_name, param in sig.parameters.items():
                param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
                param_desc = parameter_descriptions.get(param_name, "") if parameter_descriptions else ""
                
                parameters[param_name] = {
                    "type": param_type,
                    "description": param_desc,
                    "required": param.default == inspect.Parameter.empty
                }
            
            # Register the tool
            self._tools[tool_name] = {
                "function": func,
                "description": description,
                "parameters": parameters
            }
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def execute(self, tool_name: str, **params) -> Any:
        """
        Execute a registered tool by name with the provided parameters.
        
        Args:
            tool_name: Name of the tool to execute
            **params: Parameters to pass to the tool
            
        Returns:
            Result of the tool execution
            
        Raises:
            ValueError: If the tool is not found or required parameters are missing
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        tool = self._tools[tool_name]
        
        # Validate required parameters
        for param_name, param_info in tool["parameters"].items():
            if param_info["required"] and param_name not in params:
                raise ValueError(f"Missing required parameter '{param_name}' for tool '{tool_name}'")
        
        try:
            return tool["function"](**params)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
    
    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        Get descriptions of all registered tools in a format suitable for LLMs.
        
        Returns:
            List of tool descriptions with name, description, and parameters
        """
        tool_descriptions = []
        
        for name, tool in self._tools.items():
            tool_desc = {
                "name": name,
                "description": tool["description"],
                "parameters": {}
            }
            
            for param_name, param_info in tool["parameters"].items():
                tool_desc["parameters"][param_name] = {
                    "description": param_info["description"],
                    "required": param_info["required"]
                }
            
            tool_descriptions.append(tool_desc)
        
        return tool_descriptions

# Create a global instance of the registry
registry = ToolRegistry()

# Convenience function for registering tools
def register_tool(name=None, description="", parameter_descriptions=None):
    """
    Decorator for registering a function as a tool.
    
    Args:
        name: Optional custom name for the tool (defaults to function name)
        description: Description of what the tool does
        parameter_descriptions: Optional descriptions for each parameter
        
    Returns:
        Decorator function
    """
    return registry.register(name, description, parameter_descriptions)

# Convenience function for executing tools
def execute_tool(tool_name, **params):
    """
    Execute a registered tool by name with the provided parameters.
    
    Args:
        tool_name: Name of the tool to execute
        **params: Parameters to pass to the tool
        
    Returns:
        Result of the tool execution
    """
    return registry.execute(tool_name, **params)

# Convenience function for getting tool descriptions
def get_tool_descriptions():
    """
    Get descriptions of all registered tools in a format suitable for LLMs.
    
    Returns:
        List of tool descriptions
    """
    return registry.get_tool_descriptions() 