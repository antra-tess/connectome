"""
ToolProvider Component
Component for registering and executing tools for Elements.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Set
import uuid
import time
import functools

from .base_component import Component

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ToolProvider(Component):
    """
    Component for registering and executing tools for Elements.
    
    The ToolProvider is responsible for:
    - Registering tools that can be called on an Element
    - Executing tools when requested
    - Tracking tool execution for the timeline
    """
    
    # Component unique type identifier
    COMPONENT_TYPE: str = "tool_provider"
    
    # Event types this component handles
    HANDLED_EVENT_TYPES: List[str] = [
        "tool_executed"
    ]
    
    def __init__(self, element=None):
        """
        Initialize the ToolProvider component.
        
        Args:
            element: The Element this component is attached to
        """
        super().__init__(element)
        
        # Initialize state
        self._state = {
            "tools": {},  # name -> tool function
            "tool_descriptions": {},  # name -> description and parameters
            "execution_history": []  # list of tool executions
        }
    
    def register_tool(self, name: str, description: str, parameter_descriptions: Dict[str, str]) -> Callable:
        """
        Register a tool with this Element.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            parameter_descriptions: Dictionary mapping parameter names to descriptions
            
        Returns:
            Decorator to apply to the tool function
        """
        def decorator(func: Callable) -> Callable:
            self._state["tools"][name] = func
            self._state["tool_descriptions"][name] = {
                "description": description,
                "parameters": parameter_descriptions
            }
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
                
            return wrapper
            
        return decorator
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """
        Execute a tool by name.
        
        Args:
            name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Result of the tool execution
        """
        if not self._is_initialized or not self._is_enabled:
            logger.warning(f"Cannot execute tool: ToolProvider component {self.id} is not initialized or enabled")
            return {"error": "Component not initialized or enabled"}
            
        # Check if the tool exists
        if name not in self._state["tools"]:
            error = f"Tool not found: {name}"
            logger.warning(error)
            return {"error": error}
            
        # Get the tool function
        tool_func = self._state["tools"][name]
        
        # Record the execution
        execution_record = {
            "tool": name,
            "timestamp": int(time.time() * 1000),
            "parameters": kwargs,
            "status": "started"
        }
        self._state["execution_history"].append(execution_record)
        
        # Execute the tool
        try:
            result = tool_func(**kwargs)
            
            # Update the execution record
            execution_record["status"] = "completed"
            execution_record["result"] = result
            
            # Create a tool execution event
            self._record_tool_execution(name, kwargs, result)
            
            return result
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            
            # Update the execution record
            execution_record["status"] = "error"
            execution_record["error"] = str(e)
            
            return {"error": str(e)}
    
    def _record_tool_execution(self, name: str, parameters: Dict[str, Any], result: Any) -> None:
        """
        Record a tool execution in the timeline.
        
        Args:
            name: Name of the executed tool
            parameters: Parameters passed to the tool
            result: Result of the tool execution
        """
        # Create an event for this tool execution
        event = {
            "event_type": "tool_executed",
            "tool_name": name,
            "parameters": parameters,
            "result": result,
            "timestamp": int(time.time() * 1000),
            "element_id": self.element.id if self.element else None
        }
        
        # Notify the Element to handle this event
        if self.element:
            self.element.handle_event(event, {"timeline_id": "primary"})
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary of tool name to tool description
        """
        return self._state["tool_descriptions"].copy()
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        Get the tool execution history.
        
        Returns:
            List of tool execution records
        """
        return self._state["execution_history"].copy()
    
    def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handle tool-related events.
        
        Args:
            event: Event data
            timeline_context: Timeline context for this event
            
        Returns:
            True if the event was handled, False otherwise
        """
        # Nothing specific to do here yet
        return False 