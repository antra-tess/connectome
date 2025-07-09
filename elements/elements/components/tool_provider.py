"""
Tool Provider Component
Manages the registration and execution of tools (actions) for an Element.
Sibling components (e.g., Action Handlers) register their tools here.
The AgentLoopComponent (or similar) uses this component to discover and execute tools.
"""
import logging
from typing import Dict, Any, Callable, List, Optional, Tuple, TypedDict
import asyncio
import inspect

from ..base import Component, BaseElement
# Import the registry decorator
from elements.component_registry import register_component
# Import LLM Tool Definition structure
from llm.provider_interface import LLMToolDefinition

logger = logging.getLogger(__name__)

# --- Define ToolParameterSchema structure ---
class ToolParameterProperty(TypedDict):
    type: str # JSON Schema types: string, integer, number, boolean, array, object
    description: str
    # For array type, can specify items schema
    items: Optional[Dict[str, Any]] # e.g., {"type": "string"} or {"type": "object", "properties": {...}}
    # For object type, can specify properties schema
    properties: Optional[Dict[str, Any]] 
    # For enum
    enum: Optional[List[Any]]

class ToolParameter(TypedDict):
    name: str
    description: str
    type: str # JSON Schema types: string, integer, number, boolean, array, object
    required: bool
    # Optional: for more complex types like arrays of objects or enums
    items: Optional[Dict[str, Any]] # Schema for items if type is "array"
    properties: Optional[Dict[str, Any]] # Schema for properties if type is "object"
    enum: Optional[List[Any]] # List of possible values if type is "string" or "number" with enum restriction

# Stored tool information will now use List[ToolParameter] for its parameter schema
class StoredToolDefinition(TypedDict):
    description: str
    parameters_schema: List[ToolParameter] # List of parameter definition dicts
    function: Callable

@register_component
class ToolProviderComponent(Component):
    """
    A component that acts as a registry and execution point for tools on an Element.
    """
    COMPONENT_TYPE = "ToolProviderComponent"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # _tools will store:
        # {
        #   "tool_name": StoredToolDefinition, ...
        # }
        self._tools: Dict[str, StoredToolDefinition] = {}
        logger.debug(f"ToolProviderComponent initialized")

    def initialize(self, **kwargs) -> None:
        """Initializes the component."""
        super().initialize(**kwargs)
        # No specific initialization actions needed beyond base class for now.

    def register_tool(self, name: str, description: str, parameters_schema: List[ToolParameter]) -> Callable:
        """
        Decorator to register a callable as a tool.

        Args:
            name: The unique name of the tool.
            description: A human-readable description of what the tool does.
            parameters_schema: A list of dictionaries, where each dictionary defines a parameter
                             according to ToolParameter TypedDict structure.
        
        Returns:
            The decorator that registers the function.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Tool name must be a non-empty string.")
        if not isinstance(description, str):
            raise ValueError("Tool description must be a string.")
        if not isinstance(parameters_schema, list):
            raise ValueError("Parameters must be a list of parameter definition dictionaries.")
        for param in parameters_schema:
            if not isinstance(param, dict) or not all(k in param for k in ['name', 'type', 'description', 'required']):
                raise ValueError(f"Invalid parameter definition: {param}. Must include name, type, description, required.")
            if not isinstance(param['name'], str) or not param['name']:
                 raise ValueError(f"Parameter name must be a non-empty string in {param}")
            if not isinstance(param['type'], str) or param['type'] not in ["string", "integer", "number", "boolean", "array", "object"]:
                 raise ValueError(f"Parameter type must be a valid JSON schema type in {param}")
            if not isinstance(param['description'], str):
                 raise ValueError(f"Parameter description must be a string in {param}")
            if not isinstance(param['required'], bool):
                 raise ValueError(f"Parameter required must be a boolean in {param}")

        def decorator(func: Callable) -> Callable:
            self.register_tool_function(name, description, parameters_schema, func)
            return func
        return decorator

    def register_tool_function(self, name: str, description: str, parameters_schema: List[ToolParameter], tool_func: Callable) -> None:
        """
        Registers a tool function programmatically.

        Args:
            name: The unique name of the tool.
            description: A human-readable description of what the tool does.
            parameters_schema: A list of dictionaries defining the tool's parameters.
            tool_func: The callable function that implements the tool.
        """
        if name in self._tools:
            logger.warning(f"Tool '{name}' is being re-registered on Element {self.owner.id if self.owner else 'unknown'}. Overwriting previous definition.")
        
        # Validate parameters_schema (could reuse validation from decorator or add more here)
        for param in parameters_schema:
             if not isinstance(param, dict) or not all(k in param for k in ['name', 'type', 'description', 'required']):
                 logger.error(f"Parameters schema: {parameters_schema}")
                 logger.error(f"Invalid parameter definition during register_tool_function for tool '{name}': {param}")
                 # Potentially raise an error or skip registration
                 return 

        self._tools[name] = {
            "description": description,
            "parameters_schema": parameters_schema,
            "function": tool_func
        }
        logger.info(f"Tool '{name}' registered via function for Element {self.owner.id if self.owner else 'unknown'}.")

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Returns a list of available tools with their descriptions and parameter schemas.
        Formatted for easy consumption.
        
        DEPRECATED: Use get_llm_tool_definitions() for LLMToolDefinition objects
        or get_available_tool_names() for simple name lists.
        This method is maintained for backward compatibility.
        """
        available_tools_list = []
        for tool_name, tool_info in self._tools.items():
            # Format parameters for simpler display if needed, or pass full schema
            formatted_params = []
            for param_schema in tool_info["parameters_schema"]:
                formatted_params.append({
                    "name": param_schema["name"],
                    "type": param_schema["type"],
                    "description": param_schema["description"],
                    "required": param_schema["required"],
                    "details": {k:v for k,v in param_schema.items() if k not in ['name', 'type', 'description', 'required'] and v is not None}
                })
            available_tools_list.append({
                "name": tool_name,
                "description": tool_info["description"],
                "parameters": formatted_params 
            })
        return available_tools_list

    def get_available_tool_names(self) -> List[str]:
        """
        Returns a list of names of all registered tools.
        
        NEW: Backward compatibility method for components that expect simple tool name lists.
        This is the same as list_tools() but with a more descriptive name.
        """
        return list(self._tools.keys())

    def get_enhanced_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        NEW: Returns enhanced tool definitions with complete metadata for VEIL emission.
        
        This provides the rich tool information needed for tool aggregation and rendering
        while maintaining all original tool metadata including target element information.
        
        Returns:
            List of enhanced tool definition dictionaries with complete metadata
        """
        enhanced_definitions = []
        for tool_name, tool_info in self._tools.items():
            # Build the complete tool definition with all metadata
            enhanced_def = {
                "name": tool_name,
                "description": tool_info["description"],
                "parameters": self._build_json_schema_from_parameters(tool_info["parameters_schema"]),
                "target_element_id": self.owner.id if self.owner else None,
                "element_name": getattr(self.owner, 'name', 'Unknown Element') if self.owner else 'Unknown Element',
                "element_type": self.owner.__class__.__name__ if self.owner else 'Unknown',
                "original_tool_name": tool_name  # For aggregation logic
            }
            enhanced_definitions.append(enhanced_def)
        return enhanced_definitions

    def _build_json_schema_from_parameters(self, parameters_schema: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        NEW: Build JSON schema from parameter definitions for enhanced tool definitions.
        
        This is extracted from get_llm_tool_definitions() to avoid code duplication.
        """
        json_schema_properties = {}
        required_params = []

        for param_def in parameters_schema:
            param_name = param_def["name"]
            prop_schema = {
                "type": param_def["type"],
                "description": param_def["description"]
            }
            # Add optional fields if present
            if param_def.get("items"):
                prop_schema["items"] = param_def["items"]
            if param_def.get("properties"): # For object type
                prop_schema["properties"] = param_def["properties"]
            if param_def.get("enum"):
                 prop_schema["enum"] = param_def["enum"]
            
            json_schema_properties[param_name] = prop_schema
            
            if param_def["required"]:
                required_params.append(param_name)
        
        final_json_schema = {
            "type": "object",
            "properties": json_schema_properties
        }
        if required_params: # Only add 'required' key if there are required parameters
            final_json_schema["required"] = required_params
        
        return final_json_schema

    def get_llm_tool_definitions(self) -> List[LLMToolDefinition]:
        """
        Returns a list of available tools formatted as LLMToolDefinition objects,
        suitable for passing to the LLM provider.
        Constructs JSON schema from the detailed parameter definitions.
        """
        llm_tools = []
        for tool_name, tool_info in self._tools.items():
            # Use the new helper method to build JSON schema
            final_json_schema = self._build_json_schema_from_parameters(tool_info["parameters_schema"])
            
            llm_tools.append(LLMToolDefinition(
                name=tool_name,
                description=tool_info["description"],
                parameters=final_json_schema
            ))
        return llm_tools

    async def execute_tool(self, tool_name: str, calling_context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes a registered tool by its name with the given keyword arguments.

        Args:
            tool_name: The name of the tool to execute.
            calling_context: Optional dictionary containing context about the caller.
            **kwargs: Keyword arguments to pass to the tool's function.

        Returns:
            A dictionary containing the result of the tool execution.
            Typically includes a "success": True/False field and either "result" or "error".
        """
        if tool_name not in self._tools:
            logger.error(f"Tool '{tool_name}' not found on Element {self.owner.id if self.owner else 'unknown'}.")
            logger.error(f"Available tools: {self.list_tools()}")
            return {"success": False, "error": f"Tool '{tool_name}' not found."}

        tool_info = self._tools[tool_name]
        tool_func = tool_info["function"]
        # Inspect the tool function's signature to see if it accepts calling_context
        tool_func_sig = inspect.signature(tool_func)
        params_to_pass = kwargs.copy() # Start with parameters from LLM

        if 'calling_context' in tool_func_sig.parameters:
            params_to_pass['calling_context'] = calling_context
            logger.debug(f"Passing calling_context to tool '{tool_name}'. Context keys: {list(calling_context.keys()) if calling_context else 'None'}")
        elif calling_context:
            logger.debug(f"Tool '{tool_name}' does not accept calling_context, but it was provided. It will not be passed. Context keys: {list(calling_context.keys())}")

        try:
            logger.info(f"Executing tool '{tool_name}' on Element {self.owner.id if self.owner else 'unknown'} with params: {kwargs}. Context provided: {bool(calling_context)}")
            
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**params_to_pass)
            else:
                result = tool_func(**params_to_pass)
            
            if not isinstance(result, dict):
                logger.warning(f"Tool '{tool_name}' did not return a dictionary. Wrapping result. Original: {result}")
                if result is not None:
                    return {"success": True, "result": result}
                return {"success": True, "status": "Tool executed, no explicit result returned."}

            if 'success' not in result:
                logger.debug(f"Tool '{tool_name}' result dict missing 'success' flag. Assuming success for now.")
                result['success'] = True 

            return result
        except TypeError as te: 
            logger.error(f"TypeError executing tool '{tool_name}' on Element {self.owner.id if self.owner else 'unknown'}: {te}", exc_info=True)
            return {"success": False, "error": f"Parameter mismatch for tool '{tool_name}': {te}"}
        except Exception as e:
            logger.error(f"Exception executing tool '{tool_name}' on Element {self.owner.id if self.owner else 'unknown'}: {e}", exc_info=True)
            return {"success": False, "error": f"Error executing tool '{tool_name}': {str(e)}"}

    def get_tool_definition(self, tool_name: str) -> Optional[StoredToolDefinition]:
        """
        Retrieves the full definition of a tool, including its function.
        Used internally or by advanced components.
        """
        return self._tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """Returns a list of names of all registered tools."""
        return list(self._tools.keys())

    # Optional: Method to unregister a tool, if dynamic unregistration is needed.
    # def unregister_tool(self, tool_name: str) -> bool: ... 