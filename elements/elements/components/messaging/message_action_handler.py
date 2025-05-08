"""
Message Action Handler Component
Provides tools/actions for interacting with messaging elements (e.g., sending messages).
"""
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

from ...base import Component

# Assuming MessageListComponent is used for context, but not strictly required by handler itself
from .message_list import MessageListComponent 

if TYPE_CHECKING:
    from ...base import BaseElement
    from ..tool_provider import ToolProviderComponent # In components directory
    from host.event_loop import OutgoingActionCallback # In host directory
    from elements.elements.inner_space import InnerSpace # To check if owner is InnerSpace for DMs
    from elements.elements.space import Space # To check if owner is Space (SharedSpace context)

logger = logging.getLogger(__name__)

class MessageActionHandler(Component):
    """
    Provides actions (tools) for messaging functionality, like sending messages.
    Registers these tools with the owning element's ToolProviderComponent.
    Uses the outgoing_action_callback to send actions to external systems.
    """
    COMPONENT_TYPE = "MessageActionHandler"

    # Define dependencies that InnerSpace or other parent Elements should inject.
    # We need the callback to send things out.
    INJECTED_DEPENDENCIES = {
        'outgoing_action_callback': '_outgoing_action_callback'
    }
    
    def __init__(self, element_id: str, name: str, outgoing_action_callback: Optional['OutgoingActionCallback'] = None, **kwargs):
        super().__init__(element_id, name, **kwargs)
        self._outgoing_action_callback = outgoing_action_callback
        if not self._outgoing_action_callback:
             logger.warning(f"MessageActionHandler {self.id} initialized without an outgoing_action_callback. Cannot send external messages.")

    def initialize(self, **kwargs) -> None:
        """Registers tools after initialization."""
        super().initialize(**kwargs)
        self._register_messaging_tools()
        logger.debug(f"MessageActionHandler initialized and tools registered for Element {self.owner.id}")

    def set_outgoing_action_callback(self, callback: 'OutgoingActionCallback'):
        """Allows setting the callback after initialization if needed."""
        self._outgoing_action_callback = callback
        logger.debug(f"Outgoing action callback set for MessageActionHandler {self.id}")

    def _register_messaging_tools(self):
        """Registers messaging-related tools with the ToolProviderComponent."""
        tool_provider: Optional['ToolProviderComponent'] = self.get_sibling_component("ToolProviderComponent")
        if not tool_provider:
            logger.error(f"[{self.owner.id}] Cannot register messaging tools: ToolProviderComponent not found.")
            return

        # --- Register send_message Tool --- 
        @tool_provider.register_tool(
            name="send_message",
            description="Sends a message to the chat context represented by this element.",
            parameter_descriptions={
                "text": "The content of the message to send (required, string)",
                "reply_to_external_id": "Optional external ID of the message being replied to (string)"
            }
        )
        def send_message_tool(text: str, reply_to_external_id: Optional[str] = None) -> Dict[str, Any]:
            """Handles sending a message, determining if it's DM or channel based on context."""
            if not self._outgoing_action_callback:
                return {"success": False, "error": "Outgoing action callback is not configured."}
            if not text:
                 return {"success": False, "error": "Message text cannot be empty."}

            # Determine context: DM (in InnerSpace) or Channel (in SharedSpace)?
            # This requires knowing the nature of the owning element and its space.
            adapter_id = None
            conversation_id = None # This might be channel ID or recipient ID for DMs
            is_dm = False

            # Check if owner element is for DMs (e.g., MyDiscordDMsElement)
            # We need a better way to know this - maybe a property/component on the owner?
            # For now, let's use a naming convention or check parent space type.
            if hasattr(self.owner, 'parent') and self.owner.parent and hasattr(self.owner.parent, 'IS_INNER_SPACE') and self.owner.parent.IS_INNER_SPACE:
                 # Assume elements directly in InnerSpace handling messages are for DMs or specific agent interactions
                 is_dm = True
                 # How to get adapter_id and conversation_id (recipient) for DMs?
                 # The owning element (e.g., MyDiscordDMsElement) should provide this info.
                 if hasattr(self.owner, 'get_adapter_id'): adapter_id = self.owner.get_adapter_id()
                 if hasattr(self.owner, 'get_dm_recipient_id'): conversation_id = self.owner.get_dm_recipient_id()
                 logger.debug(f"[{self.owner.id}] send_message detected DM context.")
            elif hasattr(self.owner, 'parent') and self.owner.parent and hasattr(self.owner.parent, 'IS_SPACE') and not getattr(self.owner.parent, 'IS_INNER_SPACE', False):
                 # Assume elements in other Spaces (SharedSpaces) are for channels
                 is_dm = False
                 # How to get adapter_id and channel_id for SharedSpace?
                 # The owning Space should provide this info based on its metadata.
                 if hasattr(self.owner.parent, 'metadata'): # Assuming metadata is stored
                     adapter_id = self.owner.parent.metadata.get('source_adapter')
                     conversation_id = self.owner.parent.metadata.get('external_channel_id')
                 logger.debug(f"[{self.owner.id}] send_message detected Channel context.")
            else:
                 return {"success": False, "error": "Cannot determine context (DM or Channel) for sending message."}

            if not adapter_id or not conversation_id:
                return {"success": False, "error": f"Could not determine adapter_id ({adapter_id}) or conversation_id ({conversation_id}) for sending."}
            
            # Construct the action request for the HostEventLoop outgoing queue
            # This structure should match what ActivityClient.handle_outgoing_action expects
            action_request = {
                "target_module": "ActivityClient",
                "action_type": "send_message", # Corresponds to ActivityClient handling
                "payload": {
                    "adapter_id": adapter_id,
                    "conversation_id": conversation_id, # Channel ID or Recipient ID
                    "text": text,
                    "reply_to_external_id": reply_to_external_id,
                    # Include agent ID/name if needed by ActivityClient for history recording
                    "requesting_element_id": self.owner.id, # ID of the element initiating the send
                    "requesting_agent_id": self.owner.parent.id if (hasattr(self.owner, 'parent') and self.owner.parent and hasattr(self.owner.parent, 'IS_INNER_SPACE')) else None # Best guess for agent ID
                }
            }
            
            try:
                self._outgoing_action_callback(action_request)
                logger.info(f"[{self.owner.id}] Dispatched 'send_message' action to adapter '{adapter_id}' for conversation '{conversation_id}'.")
                return {"success": True, "status": "Message sent to outgoing queue."} 
            except Exception as e:
                 logger.error(f"[{self.owner.id}] Error dispatching send_message action via callback: {e}", exc_info=True)
                 return {"success": False, "error": f"Error dispatching message: {e}"}

        # --- Register add_reaction Tool (Example) --- 
        # @tool_provider.register_tool(...)
        # def add_reaction_tool(message_external_id: str, emoji: str) -> Dict[str, Any]:
        #     # Similar logic to determine context (adapter_id, conversation_id)
        #     # Construct action_request for "add_reaction"
        #     # Use self._outgoing_action_callback
        #     pass

        # --- Register other tools as needed ---
