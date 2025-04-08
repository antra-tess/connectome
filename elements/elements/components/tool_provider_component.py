"""
ToolProvider Component

Component for registering and retrieving tool definitions for an Element.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Set
import uuid
import time
import functools

from ..base import BaseElement, Component # Correct relative import
from .tool_provider_component import ToolDefinition # Import own definition
from .space.container_component import ContainerComponent # Dependency
from collections import defaultdict # For mounted tool tracking

logger = logging.getLogger(__name__)

# Define a structure for tool definitions (can be expanded)
class ToolDefinition:
    def __init__(self, name: str, description: str, parameters: Dict, execution_info: Dict):
        self.name = name
        self.description = description
        self.parameters = parameters # e.g., JSON Schema for parameters
        self.execution_info = execution_info # {"type": "direct_call" | "action_request", ...details}
        
    def to_dict(self) -> Dict[str, Any]:
         return {
              "name": self.name,
              "description": self.description,
              "parameters": self.parameters,
              "execution_info": self.execution_info
         }

class ToolProviderComponent(Component):
    """
    Manages the registration and retrieval of ToolDefinitions for the Element.
    It does not execute tools itself; execution is handled by the AgentLoopComponent
    based on the retrieved execution_info.
    Includes aggregation of tools from mounted elements.
    """
    
    COMPONENT_TYPE: str = "tool_provider"
    DEPENDENCIES: Set[str] = {ContainerComponent.COMPONENT_TYPE}
    HANDLED_EVENT_TYPES: List[str] = [] # Doesn't handle events directly
    
    def __init__(self, element: BaseElement): # Use BaseElement type hint
        super().__init__(element)
        # Store ToolDefinition objects keyed by tool name
        self._tool_definitions: Dict[str, ToolDefinition] = {}
        # Track tools originating from mounted elements: {mount_id: {tool_name_in_provider: original_tool_name}}
        self._mounted_element_tools: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._container_comp: Optional[ContainerComponent] = None # Store ref to container
        logger.debug(f"ToolProviderComponent initialized for element {element.id}")

    def _initialize(self, config: Dict = None) -> bool:
        """Register listeners on initialize."""
        if not super()._initialize(config):
             return False # Don't proceed if base initialization failed

        # Ensure element is set before accessing components
        if not self.element:
            logger.error("ToolProviderComponent cannot initialize: element is not set.")
            return False

        self._container_comp = self.element.get_component(ContainerComponent)
        if self._container_comp:
            try:
                self._container_comp.add_mount_listener(self._handle_element_mount)
                self._container_comp.add_unmount_listener(self._handle_element_unmount)
                # Process already mounted elements
                for mount_id, element in self._container_comp.get_mounted_elements().items():
                     self._handle_element_mount(mount_id, element)
                logger.info(f"ToolProviderComponent registered listeners with ContainerComponent on {self.element.id}")
                return True
            except AttributeError as e:
                 logger.error(f"ToolProviderComponent failed to register listeners. Container missing methods? {e}", exc_info=True)
                 return False # Fail initialization if methods missing
        else:
            logger.error(f"ToolProviderComponent failed to get ContainerComponent dependency on {self.element.id}")
            return False # Fail initialization if dependency missing

    def _on_cleanup(self) -> bool:
        """Unregister listeners on cleanup."""
        if self._container_comp:
             try:
                 self._container_comp.remove_mount_listener(self._handle_element_mount)
                 self._container_comp.remove_unmount_listener(self._handle_element_unmount)
                 logger.info(f"ToolProviderComponent unregistered listeners from ContainerComponent on {self.element.id}")
             except Exception as e:
                 logger.error(f"Error unregistering container listeners: {e}")
        self._container_comp = None # Clear reference
        # Unregister all tools sourced from mounted elements just in case
        all_mounted_tool_names = []
        for mount_id in list(self._mounted_element_tools.keys()):
             all_mounted_tool_names.extend(list(self._mounted_element_tools[mount_id].keys()))
             del self._mounted_element_tools[mount_id] # Clear tracking
        for tool_name in all_mounted_tool_names:
             if tool_name in self._tool_definitions:
                  del self._tool_definitions[tool_name]
                  logger.debug(f"Cleaned up mounted tool {tool_name} on shutdown.")
                  
        return super()._on_cleanup()


    def _handle_element_mount(self, mount_id: str, mounted_element: BaseElement):
        """Callback when an element is mounted in the container."""
        if not mounted_element:
             logger.warning(f"_handle_element_mount called with None element for mount_id {mount_id}")
             return
             
        logger.info(f"Handling mount of element {mounted_element.id} at mount_id {mount_id}")
        # Check if the mounted element has its own ToolProvider (use class, not string)
        mounted_tool_provider = mounted_element.get_component(ToolProviderComponent)
        if mounted_tool_provider:
            logger.debug(f"Mounted element {mounted_element.id} has a ToolProviderComponent. Aggregating tools...")
            original_definitions = mounted_tool_provider.get_all_tool_definitions()
            for original_name, tool_def in original_definitions.items():
                # Create a prefixed name for the tool in this provider
                prefixed_tool_name = f"{mount_id}.{original_name}" # Use dot separator?

                # --- Attempt Direct Call --- (Or consider safer action_request later)
                success = self.register_tool(
                    name=prefixed_tool_name,
                    description=f"[{mounted_element.name}@{mount_id}] {tool_def.description}", # Add context
                    parameters=tool_def.parameters,
                    execution_info=tool_def.execution_info # Use original info directly
                )
                if success:
                    # Track the mapping from prefixed name to original name for this mount_id
                    self._mounted_element_tools[mount_id][prefixed_tool_name] = original_name
                else:
                     logger.warning(f"Failed to register aggregated tool {prefixed_tool_name} from {mounted_element.id}")
        else:
             logger.debug(f"Mounted element {mounted_element.id} does not have a ToolProviderComponent.")

    def _handle_element_unmount(self, mount_id: str, element_being_unmounted: BaseElement):
        """Callback when an element is about to be unmounted."""
        if not element_being_unmounted:
             logger.warning(f"_handle_element_unmount called with None element for mount_id {mount_id}")
             return
             
        logger.info(f"Handling unmount of element {element_being_unmounted.id} from mount_id {mount_id}")
        if mount_id in self._mounted_element_tools:
            tool_names_to_remove = list(self._mounted_element_tools[mount_id].keys()) # Get keys before modifying
            logger.debug(f"Unregistering {len(tool_names_to_remove)} tools from unmounted element {element_being_unmounted.id}")
            for tool_name in tool_names_to_remove:
                self.unregister_tool(tool_name) # Let unregister handle removal from main dict
            # Clean up tracking dict (handled within unregister_tool now)
            # del self._mounted_element_tools[mount_id]

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        execution_info: Dict[str, Any]
    ) -> bool:
        if name in self._tool_definitions:
             logger.warning(f"Tool '{name}' already registered for element {self.element.id}. Overwriting.")
             # If overwriting, check if it was a mounted tool and remove old tracking
             for mount_id, tools in list(self._mounted_element_tools.items()):
                  if name in tools:
                       del self._mounted_element_tools[mount_id][name]
                       if not self._mounted_element_tools[mount_id]:
                           del self._mounted_element_tools[mount_id]
                       break

        if not execution_info or 'type' not in execution_info:
             logger.error(f"Cannot register tool '{name}': execution_info must contain a 'type' key.")
             return False
             
        try:
            definition = ToolDefinition(
                 name=name,
                 description=description,
                 parameters=parameters,
                 execution_info=execution_info
            )
            self._tool_definitions[name] = definition
            logger.info(f"Registered tool '{name}' for element {self.element.id}")
            return True
        except Exception as e:
            logger.error(f"Error creating ToolDefinition for '{name}': {e}", exc_info=True)
            return False

    def unregister_tool(self, name: str) -> bool:
        """Removes a tool definition and cleans up mounted tool tracking."""
        if name in self._tool_definitions:
            del self._tool_definitions[name]
            logger.info(f"Unregistered tool '{name}' for element {self.element.id}")
            # Check if it was a tracked mounted tool and remove tracking
            for mount_id, tools in list(self._mounted_element_tools.items()):
                 if name in tools:
                      del self._mounted_element_tools[mount_id][name]
                      if not self._mounted_element_tools[mount_id]:
                          del self._mounted_element_tools[mount_id]
                      logger.debug(f"Removed tracking for mounted tool {name} from {mount_id}")
                      break
            return True
        else:
            logger.warning(f"Tool '{name}' not found for unregistration on element {self.element.id}")
            return False

    def get_tool_definition(self, name: str) -> Optional[ToolDefinition]:
        """Gets the definition for a specific tool by name."""
        return self._tool_definitions.get(name)

    def get_all_tool_definitions(self) -> Dict[str, ToolDefinition]:
        """Gets all registered tool definitions."""
        return self._tool_definitions.copy()
        
    def get_llm_tool_schemas(self) -> List[Dict[str, Any]]:
         """
         Generates a list of tool schemas suitable for passing to an LLM 
         that supports function/tool calling (like OpenAI).
         Excludes execution_info.
         """
         schemas = []
         for name, definition in self._tool_definitions.items():
              schema = {
                   # Adjust based on specific LLM API requirements (e.g., OpenAI wants 'function')
                   "type": "function", 
                   "function": {
                        "name": name,
                        "description": definition.description,
                        "parameters": definition.parameters # Assuming parameters are JSON Schema
                   }
              }
              schemas.append(schema)
         return schemas

    # Removed execute_tool, _record_tool_execution, get_execution_history, _on_event

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