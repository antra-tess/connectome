"""
Tool Provider Component
Manages the registration and execution of tools (actions) for an Element.
Sibling components (e.g., Action Handlers) register their tools here.
The AgentLoopComponent (or similar) uses this component to discover and execute tools.
"""
import logging
from typing import Dict, Any, Callable, List, Optional, Tuple

from ..base import Component, BaseElement

logger = logging.getLogger(__name__)

# Define a type for the stored tool information
ToolDefinition = Dict[str, Any] # Stores description, params_schema, function

class ToolProviderComponent(Component):
    """
    A component that acts as a registry and execution point for tools on an Element.
    """
    COMPONENT_TYPE = "ToolProviderComponent"

    def __init__(self, element_id: str, name: str = "ToolProvider", **kwargs):
        super().__init__(element_id, name, **kwargs)
        # _tools will store:
        # {
        #   "tool_name": {
        #       "description": "...",
        #       "parameter_descriptions": {"param1": "desc1", ...},
        #       "function": callable_function
        #   }, ...
        # }
        self._tools: Dict[str, ToolDefinition] = {}
        logger.debug(f"ToolProviderComponent initialized for Element {self.owner_id if self.owner else 'unowned'}")

    def initialize(self, **kwargs) -> None:
        """Initializes the component."""
        super().initialize(**kwargs)
        # No specific initialization actions needed beyond base class for now.

    def register_tool(self, name: str, description: str, parameter_descriptions: Dict[str, str]) -> Callable:
        """
        Decorator to register a callable as a tool.

        Args:
            name: The unique name of the tool.
            description: A human-readable description of what the tool does.
            parameter_descriptions: A dictionary describing each parameter the tool function expects.
                                    Key: parameter name, Value: parameter description.
        
        Returns:
            The decorator that registers the function.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Tool name must be a non-empty string.")
        if not isinstance(description, str):
            raise ValueError("Tool description must be a string.")
        if not isinstance(parameter_descriptions, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in parameter_descriptions.items()):
            raise ValueError("Parameter descriptions must be a dictionary of string keys and string values.")

        def decorator(func: Callable) -> Callable:
            if name in self._tools:
                logger.warning(f"Tool '{name}' is being re-registered on Element {self.owner_id if self.owner else 'unknown'}. Overwriting previous definition.")
            
            self._tools[name] = {
                "description": description,
                "parameter_descriptions": parameter_descriptions,
                "function": func
            }
            logger.info(f"Tool '{name}' registered for Element {self.owner.id if self.owner else 'unknown'}.")
            return func
        return decorator

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Returns a list of available tools with their descriptions and parameter schemas.
        Formatted for easy consumption by an LLM or agent.
        """
        available_tools_list = []
        for tool_name, tool_info in self._tools.items():
            available_tools_list.append({
                "name": tool_name,
                "description": tool_info["description"],
                "parameters": tool_info["parameter_descriptions"] # LLM usually expects schema of params
            })
        return available_tools_list

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes a registered tool by its name with the given keyword arguments.

        Args:
            tool_name: The name of the tool to execute.
            **kwargs: Keyword arguments to pass to the tool's function.

        Returns:
            A dictionary containing the result of the tool execution.
            Typically includes a "success": True/False field and either "result" or "error".
        """
        if tool_name not in self._tools:
            logger.error(f"Tool '{tool_name}' not found on Element {self.owner.id if self.owner else 'unknown'}.")
            return {"success": False, "error": f"Tool '{tool_name}' not found."}

        tool_info = self._tools[tool_name]
        tool_func = tool_info["function"]
        
        # Basic parameter validation (checking if all expected params are present can be added here if desired)
        # For now, assume the caller (e.g., AgentLoopComponent after LLM output) provides correct params.
        
        try:
            logger.info(f"Executing tool '{tool_name}' on Element {self.owner.id if self.owner else 'unknown'} with params: {kwargs}")
            result = tool_func(**kwargs)
            
            # Ensure result is a dictionary, as this is a common contract
            if not isinstance(result, dict):
                logger.warning(f"Tool '{tool_name}' did not return a dictionary. Wrapping result. Original: {result}")
                # If it's a successful call but not a dict, it's often good to wrap it for consistency
                if result is not None: # Don't wrap None as a success
                    return {"success": True, "result": result}
                # If result is None, it's ambiguous, treat as possible non-error but no explicit success data
                return {"success": True, "status": "Tool executed, no explicit result returned."}

            # If it is a dict, ensure it has a success flag or add one.
            if 'success' not in result:
                logger.debug(f"Tool '{tool_name}' result dict missing 'success' flag. Assuming success for now.")
                result['success'] = True # Default to success if not specified and no exception

            return result
        except TypeError as te: # Often indicates parameter mismatch
            logger.error(f"TypeError executing tool '{tool_name}' on Element {self.owner.id if self.owner else 'unknown'}: {te}", exc_info=True)
            return {"success": False, "error": f"Parameter mismatch for tool '{tool_name}': {te}"}
        except Exception as e:
            logger.error(f"Exception executing tool '{tool_name}' on Element {self.owner.id if self.owner else 'unknown'}: {e}", exc_info=True)
            return {"success": False, "error": f"Error executing tool '{tool_name}': {str(e)}"}

    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Retrieves the full definition of a tool, including its function.
        Used internally or by advanced components.
        """
        return self._tools.get(tool_name)

    # Optional: Method to unregister a tool, if dynamic unregistration is needed.
    # def unregister_tool(self, tool_name: str) -> bool: ... 