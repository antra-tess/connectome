"""
Core Tools Component

Provides fundamental tools available to the agent, potentially related to 
internal state, introspection, or core functionalities not tied to specific 
external interactions or mounted elements.
"""

import logging
from typing import Dict, Any, Optional, List, Callable

from ..base import Component
from ..tool_provider_component import ToolProviderComponent
from ....host.event_loop import OutgoingActionCallback

logger = logging.getLogger(__name__)

class CoreToolsComponent(Component):
    COMPONENT_TYPE = "core_tools"
    DEPENDENCIES = set() 
    
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
             
        logger.debug(f"Registering core tools for element {self.element.id} (currently none defined)...")
        # --- Add registration calls for actual CORE tools here in the future ---
        # Example: self._register_memory_query_tool(tool_provider)
        # Example: self._register_set_reminder_tool(tool_provider)
        pass # No core tools defined yet

    # --- Tool Definitions for CORE tools would go here --- 
    
    # def _register_memory_query_tool(self, tool_provider: ToolProviderComponent):
    #    tool_name = "query_memory"
    #    description = "Searches the agent's long-term memory."
    #    parameters = { ... }
    #    # This might be a direct call to a MemoryComponent
    #    execution_info = {"type": "direct_call", "function": self._query_memory_impl}
    #    tool_provider.register_tool(tool_name, description, parameters, execution_info)
        
    # def _query_memory_impl(self, query: str): ... 