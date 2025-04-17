"""
Core Tools Component

Provides fundamental tools available to the agent, potentially related to 
internal state, introspection, or core functionalities not tied to specific 
external interactions or mounted elements.
"""

import logging
from typing import Dict, Any, Optional, List, Callable

from ..base import Component, BaseElement
from ..tool_provider_component import ToolProviderComponent
# Import context manager only for type hinting if needed, not strict dependency yet
from .context_manager_component import ContextManagerComponent
# Import ContainerComponent to find mounted elements
from .space.container_component import ContainerComponent
# Import TextStateComponent to modify and read notebook state
from .state.text_state_component import TextStateComponent
from .element_factory_component import ElementFactoryComponent
from ....host.event_loop import OutgoingActionCallback

logger = logging.getLogger(__name__)

class CoreToolsComponent(Component):
    COMPONENT_TYPE = "core_tools"
    # Add Container dependency needed for write/read_notebook_impl
    DEPENDENCIES = {ContainerComponent.COMPONENT_TYPE, ElementFactoryComponent.COMPONENT_TYPE}
    
    # Store the outgoing callback (might be needed for future core tools)
    _outgoing_action_callback: Optional[OutgoingActionCallback] = None

    def set_outgoing_action_callback(self, callback: OutgoingActionCallback):
        """Sets the callback for enqueuing outgoing actions."""
        self._outgoing_action_callback = callback
        logger.debug(f"Outgoing action callback set for {self.__class__.__name__} on {self.element.id}")

    def register_tools(self, tool_provider: ToolProviderComponent):
        """Registers all core tools with the provider."""
        if not tool_provider:
             logger.error(f"Cannot register core tools: ToolProvider component is missing on {self.element.id}")
             return
             
        logger.debug(f"Registering core tools for element {self.element.id}...")
        self._register_run_memory_processing_tool(tool_provider)
        self._register_set_timer_tool(tool_provider)
        self._register_write_to_notebook_tool(tool_provider)
        self._register_create_notebook_tool(tool_provider)
        self._register_read_from_notebook_tool(tool_provider)
        # Add more core tools here...

    # --- Tool Definitions --- 

    def _register_run_memory_processing_tool(self, tool_provider: ToolProviderComponent):
        tool_name = "run_memory_processing"
        description = "Initiates a cycle of processing older history into structured memory summaries. Useful when notified that unprocessed history is large."
        parameters = {
            "type": "object",
            "properties": {
                "process_one_chunk": {
                    "type": "boolean",
                    "description": "If true (default), process only the single oldest unprocessed chunk. If false, may process multiple chunks.",
                    "default": True
                },
                "max_chunks": {
                    "type": "integer",
                    "description": "(Optional) Maximum number of chunks to process if process_one_chunk is false."
                },
                "generation_mechanism": {
                     "type": "string",
                     "description": "The mechanism to use for generating the memory summary.",
                     "enum": ["self_query", "curated"],
                     "default": "self_query"
                }
            },
            "required": [] # No required params if defaults are used
        }
        # Execution Target: This needs refinement. Let's assume the AgentLoop handles it via a specific action type.
        # Alternatively, a dedicated MemoryOrchestrator module/component.
        execution_info = {
            "type": "action_request",
            "target_module": "AgentLoop", # Or MemoryOrchestrator
            "action_type": "trigger_memory_processing",
            "payload": {} # Arguments will be passed in standard payload
        }
        tool_provider.register_tool(tool_name, description, parameters, execution_info)

    def _register_set_timer_tool(self, tool_provider: ToolProviderComponent):
        tool_name = "set_timer"
        description = "Sets a timer to deliver a message back to the agent after a specified delay."
        parameters = {
            "type": "object",
            "properties": {
                "delay_seconds": {
                    "type": "number",
                    "description": "The delay in seconds before the timer fires."
                },
                "message": {
                    "type": "string",
                    "description": "The message to deliver when the timer fires."
                }
            },
            "required": ["delay_seconds", "message"]
        }
        # Execution Target: Needs a Host-level Timer module or integration with the event loop.
        execution_info = {
            "type": "action_request",
            "target_module": "TimerService", # Hypothetical module
            "action_type": "set_agent_timer",
            "payload": {} 
        }
        tool_provider.register_tool(tool_name, description, parameters, execution_info)

    def _register_write_to_notebook_tool(self, tool_provider: ToolProviderComponent):
        tool_name = "write_to_notebook"
        description = "Writes or appends content to a personal notebook element within the agent's InnerSpace."
        parameters = {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The text content to write."
                },
                "notebook_id": {
                    "type": "string",
                    "description": "The ID of the notebook element (e.g., a TextElement) within the InnerSpace. Defaults to 'main_notebook'.",
                    "default": "main_notebook"
                },
                 "append": {
                    "type": "boolean",
                    "description": "If true, append to existing content. If false (default), overwrite.",
                    "default": False
                 }
            },
            "required": ["content"]
        }
        # Use direct_call, pointing to the implementation method
        execution_info = {
            "type": "direct_call",
            "function": self._write_notebook_impl
        }
        tool_provider.register_tool(tool_name, description, parameters, execution_info)

    def _register_create_notebook_tool(self, tool_provider: ToolProviderComponent):
        """Registers the tool for creating new notebook TextElements."""
        tool_name = "create_notebook"
        description = "Creates a new, empty notebook element (TextElement) within the agent's InnerSpace."
        parameters = {
            "type": "object",
            "properties": {
                "notebook_id": {
                    "type": "string", 
                    "description": "A unique ID for the new notebook element (e.g., 'project_notes')."
                },
                "notebook_name": {
                     "type": "string",
                     "description": "(Optional) A display name for the notebook element."
                }
            },
            "required": ["notebook_id"]
        }
        # Use direct_call for internal creation logic
        execution_info = {
            "type": "direct_call",
            "function": self._create_notebook_impl
        }
        tool_provider.register_tool(tool_name, description, parameters, execution_info)

    def _register_read_from_notebook_tool(self, tool_provider: ToolProviderComponent):
        """Registers the tool for reading content from a notebook element."""
        tool_name = "read_from_notebook"
        description = "Reads the current content from a specified notebook element within the agent's InnerSpace."
        parameters = {
            "type": "object",
            "properties": {
                "notebook_id": {
                    "type": "string",
                    "description": "The ID of the notebook element (e.g., a TextElement) to read from. Defaults to 'main_notebook'.",
                    "default": "main_notebook"
                }
            },
            "required": [] # notebook_id has a default
        }
        # Use direct_call for internal reading logic
        execution_info = {
            "type": "direct_call",
            "function": self._read_notebook_impl
        }
        tool_provider.register_tool(tool_name, description, parameters, execution_info)

    # --- Direct Call Implementations (if any needed in the future) --- 
    # None currently defined

    # def _register_memory_query_tool(self, tool_provider: ToolProviderComponent):
    #    tool_name = "query_memory"
    #    description = "Searches the agent's long-term memory."
    #    parameters = { ... }
    #    # This might be a direct call to a MemoryComponent
    #    execution_info = {"type": "direct_call", "function": self._query_memory_impl}
    #    tool_provider.register_tool(tool_name, description, parameters, execution_info)
        
    # def _query_memory_impl(self, query: str): ... 

    def _write_notebook_impl(self, content: str, notebook_id: str = "main_notebook", append: bool = False) -> str:
        """Implementation logic for the write_to_notebook tool."""
        logger.info(f"Executing write_to_notebook: notebook_id='{notebook_id}', append={append}")
        
        if not self.element:
            msg = "Cannot execute write_to_notebook: CoreToolsComponent has no parent element (InnerSpace)."
            logger.error(msg)
            return f"Error: {msg}"
            
        container = self.element.get_component(ContainerComponent)
        if not container:
            msg = f"Cannot execute write_to_notebook: Parent element {self.element.id} has no ContainerComponent."
            logger.error(msg)
            return f"Error: {msg}"
            
        # Assume notebook_id is the mount_id or element_id within the container
        # TODO: Clarify if notebook_id refers to mount_id or element.id
        target_element = container.get_element_by_id(notebook_id) 
        if not target_element:
            # Try finding by name as a fallback? Or just fail?
            # Alternative: Find first element with TextStateComponent? Risky.
            msg = f"Cannot execute write_to_notebook: Element with ID '{notebook_id}' not found in container {self.element.id}."
            logger.error(msg)
            return f"Error: {msg}" # Return error message as tool result
            
        text_state_comp = target_element.get_component(TextStateComponent)
        if not text_state_comp:
            msg = f"Cannot execute write_to_notebook: Target element '{notebook_id}' ({target_element.id}) does not have TextStateComponent."
            logger.error(msg)
            return f"Error: {msg}"
            
        try:
            if append:
                text_state_comp.append_text(content)
                result_msg = f"Successfully appended content to notebook '{notebook_id}'."
            else:
                text_state_comp.set_text(content)
                result_msg = f"Successfully wrote content to notebook '{notebook_id}'."
            logger.info(result_msg)
            return result_msg # Return success message as tool result
        except Exception as e:
             msg = f"Error while writing to notebook '{notebook_id}': {e}"
             logger.error(msg, exc_info=True)
             return f"Error: {msg}"

    def _create_notebook_impl(self, notebook_id: str, notebook_name: Optional[str] = None) -> str:
        """Implementation logic for the create_notebook tool."""
        logger.info(f"Executing create_notebook: notebook_id='{notebook_id}', name='{notebook_name}'")
        
        if not self.element:
            msg = "Cannot execute create_notebook: CoreToolsComponent has no parent element (InnerSpace)."
            logger.error(msg)
            return f"Error: {msg}"
            
        factory_comp = self.element.get_component(ElementFactoryComponent)
        if not factory_comp:
            msg = f"Cannot execute create_notebook: Parent element {self.element.id} has no ElementFactoryComponent."
            logger.error(msg)
            return f"Error: {msg}"
            
        # Check if element with this ID already exists in the container
        container = self.element.get_component(ContainerComponent)
        if container and container.get_element_by_id(notebook_id):
             msg = f"Cannot create notebook: Element with ID '{notebook_id}' already exists in {self.element.id}."
             logger.warning(msg)
             return f"Error: {msg}"

        # Use the factory component to create and mount the TextElement
        # We assume the ElementFactory is configured to know about 'standard/text_element'
        # We pass an empty initial state for the TextStateComponent
        try:
            created_element = factory_comp.create_element(
                 prefab_name="standard/text_element", # Use the PREFAB_NAME derived from path
                 element_id=notebook_id,      # Use the user-provided ID
                 name=notebook_name or notebook_id, # Use ID as name if not provided
                 initial_state=None # TextStateComponent is defined in the prefab now
            )
            
            if created_element:
                 # The factory component should handle mounting within the InnerSpace's container
                 result_msg = f"Successfully created notebook element '{notebook_id}' (ID: {created_element.id})."
                 logger.info(result_msg)
                 return result_msg
            else:
                 # Error should have been logged by factory_comp.create_element
                 msg = f"Failed to create notebook element '{notebook_id}' using factory."
                 logger.error(msg)
                 return f"Error: {msg}"
                 
        except Exception as e:
             msg = f"Error during notebook creation for '{notebook_id}': {e}"
             logger.error(msg, exc_info=True)
             return f"Error: {msg}"

    def _read_notebook_impl(self, notebook_id: str = "main_notebook") -> str:
        """Implementation logic for the read_from_notebook tool."""
        logger.info(f"Executing read_from_notebook: notebook_id='{notebook_id}'")

        if not self.element:
            msg = "Cannot execute read_from_notebook: CoreToolsComponent has no parent element (InnerSpace)."
            logger.error(msg)
            return f"Error: {msg}"

        container = self.element.get_component(ContainerComponent)
        if not container:
            msg = f"Cannot execute read_from_notebook: Parent element {self.element.id} has no ContainerComponent."
            logger.error(msg)
            return f"Error: {msg}"

        target_element = container.get_element_by_id(notebook_id)
        if not target_element:
            msg = f"Cannot execute read_from_notebook: Element with ID '{notebook_id}' not found in container {self.element.id}."
            logger.error(msg)
            return f"Error: {msg}" # Return error message as tool result

        text_state_comp = target_element.get_component(TextStateComponent)
        if not text_state_comp:
            msg = f"Cannot execute read_from_notebook: Target element '{notebook_id}' ({target_element.id}) does not have TextStateComponent."
            logger.error(msg)
            return f"Error: {msg}"

        try:
            # Assume TextStateComponent has a get_text() method
            content = text_state_comp.get_text()
            logger.info(f"Successfully read content from notebook '{notebook_id}'.")
            # The content itself is the result
            return content
        except AttributeError:
             msg = f"Error reading notebook '{notebook_id}': TextStateComponent does not have a 'get_text' method."
             logger.error(msg)
             return f"Error: {msg}"
        except Exception as e:
             msg = f"Error while reading from notebook '{notebook_id}': {e}"
             logger.error(msg, exc_info=True)
             return f"Error: {msg}"