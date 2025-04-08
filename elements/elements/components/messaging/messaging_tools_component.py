"""
Messaging Tools Component

Provides tools specifically for interacting with messaging platforms 
(sending, editing, deleting messages, reactions).
"""

import logging
from typing import Dict, Any, Optional, List, Callable

# Core component imports
from ..base import Component
from ..tool_provider_component import ToolProviderComponent # Adjusted relative import
# Callback type
from ....host.event_loop import OutgoingActionCallback

logger = logging.getLogger(__name__)

class MessagingToolsComponent(Component):
    """
    Component holding tools related to external messaging actions.
    Registers tools that typically generate action_requests for ActivityClient.
    """
    COMPONENT_TYPE = "messaging_tools"
    # Does not directly depend on other components for its own state,
    # but relies on ToolProvider being present on the element for registration.
    DEPENDENCIES = set()

    # This component doesn't *need* the callback itself, as the tools only
    # define the action request; the AgentLoopComponent uses its callback.
    # However, keeping the pattern might be useful if tools evolve.
    _outgoing_action_callback: Optional[OutgoingActionCallback] = None

    # --- Tool Registration (called by InnerSpace) --- 
    def register_tools(self, tool_provider: ToolProviderComponent):
        """Registers all messaging tools with the provider."""
        if not tool_provider:
             logger.error(f"Cannot register messaging tools: ToolProvider component is missing on {self.element.id}")
             return
             
        logger.debug(f"Registering messaging tools for element {self.element.id}")
        self._register_send_message_tool(tool_provider)
        self._register_edit_message_tool(tool_provider)
        self._register_delete_message_tool(tool_provider)
        self._register_react_tool(tool_provider, "add_reaction")
        self._register_react_tool(tool_provider, "remove_reaction")

    # --- Tool Definitions (Moved from CoreToolsComponent) --- 

    def _register_send_message_tool(self, tool_provider: ToolProviderComponent):
        tool_name = "send_message"
        description = "Sends a text message to a specific conversation via a specified adapter."
        parameters = {
            "type": "object",
            "properties": {
                "adapter_id": {"type": "string", "description": "The ID of the adapter to send through (e.g., 'discord_1')."},
                "conversation_id": {"type": "string", "description": "The ID of the conversation/channel/DM to send to."},
                "text": {"type": "string", "description": "The text content of the message to send."}
            },
            "required": ["adapter_id", "conversation_id", "text"]
        }
        execution_info = {
            "type": "action_request",
            "target_module": "ActivityClient",
            "action_type": "send_message", 
            "payload": {} 
        }
        tool_provider.register_tool(tool_name, description, parameters, execution_info)

    def _register_edit_message_tool(self, tool_provider: ToolProviderComponent):
        tool_name = "edit_message"
        description = "Edits the text content of a previously sent message."
        parameters = {
            "type": "object",
            "properties": {
                "adapter_id": {"type": "string", "description": "The ID of the adapter the message exists on."},
                "conversation_id": {"type": "string", "description": "The ID of the conversation containing the message."},
                "message_id": {"type": "string", "description": "The ID of the specific message to edit."},
                "text": {"type": "string", "description": "The new text content for the message."}
            },
            "required": ["adapter_id", "conversation_id", "message_id", "text"]
        }
        execution_info = {
            "type": "action_request",
            "target_module": "ActivityClient",
            "action_type": "edit_message",
            "payload": {}
        }
        tool_provider.register_tool(tool_name, description, parameters, execution_info)

    def _register_delete_message_tool(self, tool_provider: ToolProviderComponent):
        tool_name = "delete_message"
        description = "Deletes a previously sent message."
        parameters = {
            "type": "object",
            "properties": {
                "adapter_id": {"type": "string", "description": "The ID of the adapter the message exists on."},
                "conversation_id": {"type": "string", "description": "The ID of the conversation containing the message."},
                "message_id": {"type": "string", "description": "The ID of the specific message to delete."}
            },
            "required": ["adapter_id", "conversation_id", "message_id"]
        }
        execution_info = {
            "type": "action_request",
            "target_module": "ActivityClient",
            "action_type": "delete_message",
            "payload": {}
        }
        tool_provider.register_tool(tool_name, description, parameters, execution_info)

    def _register_react_tool(self, tool_provider: ToolProviderComponent, action: str):
        tool_name = action # e.g., "add_reaction" or "remove_reaction"
        description = f"{'Adds' if action == 'add_reaction' else 'Removes'} an emoji reaction to/from a specific message."
        parameters = {
            "type": "object",
            "properties": {
                "adapter_id": {"type": "string", "description": "The ID of the adapter the message exists on."},
                "conversation_id": {"type": "string", "description": "The ID of the conversation containing the message."},
                "message_id": {"type": "string", "description": "The ID of the specific message to react to."},
                "emoji": {"type": "string", "description": "The emoji to add or remove (e.g., '👍', '🎉')."}
            },
            "required": ["adapter_id", "conversation_id", "message_id", "emoji"]
        }
        execution_info = {
            "type": "action_request",
            "target_module": "ActivityClient",
            "action_type": action, # Use the specific action type
            "payload": {}
        }
        tool_provider.register_tool(tool_name, description, parameters, execution_info) 