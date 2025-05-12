"""
Message Action Handler Component
Provides tools/actions for interacting with messaging elements (e.g., sending messages).
"""
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple

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

        # --- Register delete_message Tool --- 
        @tool_provider.register_tool(
            name="delete_message",
            description="Deletes a message specified by its external ID.",
            parameter_descriptions={
                "message_external_id": "The external ID of the message to delete (required, string)"
            }
        )
        def delete_message_tool(message_external_id: str) -> Dict[str, Any]:
            """Handles deleting a message using its external ID."""
            if not self._outgoing_action_callback:
                return {"success": False, "error": "Outgoing action callback is not configured."}
            if not message_external_id:
                return {"success": False, "error": "message_external_id is required."} 
                
            # Determine context (adapter_id, conversation_id)
            adapter_id, conversation_id = self._get_message_context()
            if not adapter_id or not conversation_id:
                return {"success": False, "error": f"Could not determine context for deleting message."} 
                
            action_request = {
                "target_module": "ActivityClient",
                "action_type": "delete_message",
                "payload": {
                    "adapter_id": adapter_id,
                    "conversation_id": conversation_id,
                    "message_external_id": message_external_id,
                    "requesting_element_id": self.owner.id,
                    "requesting_agent_id": self.owner.parent.id if (hasattr(self.owner, 'parent') and self.owner.parent and hasattr(self.owner.parent, 'IS_INNER_SPACE')) else None
                }
            }
            
            try:
                self._outgoing_action_callback(action_request)
                logger.info(f"[{self.owner.id}] Dispatched 'delete_message' action for ID '{message_external_id}' to adapter '{adapter_id}'.")
                return {"success": True, "status": "Delete request sent to outgoing queue."} 
            except Exception as e:
                 logger.error(f"[{self.owner.id}] Error dispatching delete_message action: {e}", exc_info=True)
                 return {"success": False, "error": f"Error dispatching delete request: {e}"}
                 
        # --- Register edit_message Tool --- 
        @tool_provider.register_tool(
            name="edit_message",
            description="Edits an existing message specified by its external ID.",
            parameter_descriptions={
                "message_external_id": "The external ID of the message to edit (required, string)",
                "new_text": "The new text content for the message (required, string)"
            }
        )
        def edit_message_tool(message_external_id: str, new_text: str) -> Dict[str, Any]:
            """Handles editing a message using its external ID."""
            if not self._outgoing_action_callback:
                return {"success": False, "error": "Outgoing action callback is not configured."}
            if not message_external_id or not new_text:
                return {"success": False, "error": "message_external_id and new_text are required."} 
                
            # Determine context (adapter_id, conversation_id)
            adapter_id, conversation_id = self._get_message_context()
            if not adapter_id or not conversation_id:
                return {"success": False, "error": f"Could not determine context for editing message."} 
                
            action_request = {
                "target_module": "ActivityClient",
                "action_type": "edit_message",
                "payload": {
                    "adapter_id": adapter_id,
                    "conversation_id": conversation_id,
                    "message_external_id": message_external_id,
                    "new_text": new_text,
                    "requesting_element_id": self.owner.id,
                    "requesting_agent_id": self.owner.parent.id if (hasattr(self.owner, 'parent') and self.owner.parent and hasattr(self.owner.parent, 'IS_INNER_SPACE')) else None
                }
            }
            
            try:
                self._outgoing_action_callback(action_request)
                logger.info(f"[{self.owner.id}] Dispatched 'edit_message' action for ID '{message_external_id}' to adapter '{adapter_id}'.")
                return {"success": True, "status": "Edit request sent to outgoing queue."} 
            except Exception as e:
                 logger.error(f"[{self.owner.id}] Error dispatching edit_message action: {e}", exc_info=True)
                 return {"success": False, "error": f"Error dispatching edit request: {e}"}

        # --- Register add_reaction Tool --- 
        @tool_provider.register_tool(
            name="add_reaction",
            description="Adds an emoji reaction to a message specified by its external ID.",
            parameter_descriptions={
                "message_external_id": "The external ID of the message to react to (required, string)",
                "emoji": "The emoji to add as a reaction (required, string, e.g., '👍', ':smile:')"
            }
        )
        def add_reaction_tool(message_external_id: str, emoji: str) -> Dict[str, Any]:
            """Handles adding a reaction to a message using its external ID."""
            if not self._outgoing_action_callback:
                return {"success": False, "error": "Outgoing action callback is not configured."}
            if not message_external_id or not emoji:
                return {"success": False, "error": "message_external_id and emoji are required."} 
                
            # Determine context (adapter_id, conversation_id)
            adapter_id, conversation_id = self._get_message_context()
            if not adapter_id or not conversation_id:
                return {"success": False, "error": f"Could not determine context for adding reaction."} 
                
            action_request = {
                "target_module": "ActivityClient",
                "action_type": "add_reaction",
                "payload": {
                    "adapter_id": adapter_id,
                    "conversation_id": conversation_id,
                    "message_external_id": message_external_id,
                    "emoji": emoji,
                    "requesting_element_id": self.owner.id,
                    "requesting_agent_id": self.owner.parent.id if (hasattr(self.owner, 'parent') and self.owner.parent and hasattr(self.owner.parent, 'IS_INNER_SPACE')) else None
                }
            }
            
            try:
                self._outgoing_action_callback(action_request)
                logger.info(f"[{self.owner.id}] Dispatched 'add_reaction' ({emoji}) action for ID '{message_external_id}' to adapter '{adapter_id}'.")
                return {"success": True, "status": "Add reaction request sent to outgoing queue."} 
            except Exception as e:
                 logger.error(f"[{self.owner.id}] Error dispatching add_reaction action: {e}", exc_info=True)
                 return {"success": False, "error": f"Error dispatching add reaction request: {e}"}

        # --- Register remove_reaction Tool --- 
        @tool_provider.register_tool(
            name="remove_reaction",
            description="Removes an emoji reaction (previously added by this agent/bot) from a message specified by its external ID.",
            parameter_descriptions={
                "message_external_id": "The external ID of the message to remove the reaction from (required, string)",
                "emoji": "The emoji reaction to remove (required, string, e.g., '👍', ':smile:')"
            }
        )
        def remove_reaction_tool(message_external_id: str, emoji: str) -> Dict[str, Any]:
            """Handles removing a reaction from a message using its external ID."""
            if not self._outgoing_action_callback:
                return {"success": False, "error": "Outgoing action callback is not configured."}
            if not message_external_id or not emoji:
                return {"success": False, "error": "message_external_id and emoji are required."} 
                
            # Determine context (adapter_id, conversation_id)
            adapter_id, conversation_id = self._get_message_context()
            if not adapter_id or not conversation_id:
                return {"success": False, "error": f"Could not determine context for removing reaction."} 
                
            action_request = {
                "target_module": "ActivityClient",
                "action_type": "remove_reaction",
                "payload": {
                    "adapter_id": adapter_id,
                    "conversation_id": conversation_id,
                    "message_external_id": message_external_id,
                    "emoji": emoji,
                    "requesting_element_id": self.owner.id,
                    "requesting_agent_id": self.owner.parent.id if (hasattr(self.owner, 'parent') and self.owner.parent and hasattr(self.owner.parent, 'IS_INNER_SPACE')) else None
                }
            }
            
            try:
                self._outgoing_action_callback(action_request)
                logger.info(f"[{self.owner.id}] Dispatched 'remove_reaction' ({emoji}) action for ID '{message_external_id}' to adapter '{adapter_id}'.")
                return {"success": True, "status": "Remove reaction request sent to outgoing queue."} 
            except Exception as e:
                 logger.error(f"[{self.owner.id}] Error dispatching remove_reaction action: {e}", exc_info=True)
                 return {"success": False, "error": f"Error dispatching remove reaction request: {e}"}

    # --- Helper to get context --- (Could be improved)
    def _get_message_context(self, use_external_conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Helper to determine adapter_id and conversation_id based on owner/parent context."""
        adapter_id = None
        conversation_id = None
        try:
            if use_external_conversation_id:
                conversation_id = use_external_conversation_id
            if hasattr(self.owner, 'parent') and self.owner.parent:
                parent_space = self.owner.parent
                if hasattr(parent_space, 'IS_INNER_SPACE') and parent_space.IS_INNER_SPACE:
                    # DM Context - Owning element needs to provide details
                    if hasattr(self.owner, 'get_adapter_id'): adapter_id = self.owner.get_adapter_id()
                    if hasattr(self.owner, 'get_dm_recipient_id'): conversation_id = self.owner.get_dm_recipient_id()
                elif hasattr(parent_space, 'IS_SPACE'): # SharedSpace Context
                    if hasattr(parent_space, 'metadata'):
                        adapter_id = parent_space.metadata.get('source_adapter')
                        conversation_id = parent_space.metadata.get('external_channel_id')
            if not adapter_id or not conversation_id:
                logger.warning(f"[{self.owner.id}] Failed to determine valid message context (adapter: {adapter_id}, conversation: {conversation_id})")
                return {"success": False, "error": f"Failed to determine adapter context for conversation {conversation_id}"}
        except Exception as e:
            logger.error(f"[{self.owner.id}] Error determining message context: {e}", exc_info=True)
            return {"success": False, "error": f"Error determining message context: {e}"}
            
        return {"success": True, "adapter_id": adapter_id, "conversation_id": conversation_id}
        
    def handle_fetch_history(self, 
                             conversation_id: str,
                             before_ms: Optional[int] = None,
                             after_ms: Optional[int] = None,
                             limit: Optional[int] = 100, 
                             calling_context: Dict[str, Any] = None):
        """
        Tool to request historical messages for a conversation from the adapter.

        Args:
            conversation_id: The external ID of the conversation/channel.
            before_ms: Fetch messages before this timestamp (milliseconds UTC).
            after_ms: Fetch messages after this timestamp (milliseconds UTC).
            limit: Maximum number of messages to fetch.
            calling_context: Context from the loop component calling the tool.

        Returns:
            Result of the action dispatch (e.g., confirmation or error).
        """
        if not calling_context:
            logger.warning(f"[{self.owner.id}] handle_fetch_history called without calling_context.")
            # Potentially raise error or return failure? For now, proceed but log.
            calling_context = {}
            
        # --- Determine Context (Adapter ID) ---
        # Use the helper, assuming it can find the context for the owner element
        context_result = self._get_message_context(use_external_conversation_id=conversation_id)
        if not context_result['success']:
             logger.error(f"[{self.owner.id}] Failed to get message context for handle_fetch_history: {context_result['error']}")
             # TODO: Return a structured error to the LLM?
             return { "status": "error", "message": f"Failed to determine adapter context for conversation {conversation_id}" }
        
        adapter_id = context_result['adapter_id']
        # We already have conversation_id from args
        
        logger.info(f"[{self.owner.id}] Preparing fetch_history action for adapter '{adapter_id}', conv '{conversation_id}'.")
        
        payload = {
            "adapter_id": adapter_id,
            "conversation_id": conversation_id,
            "before": before_ms, # Pass along Nones if not provided
            "after": after_ms,
            "limit": limit,
            # Pass necessary context for potential response handling/history recording
            "requesting_element_id": self.owner.id, 
            "calling_loop_id": calling_context.get('loop_component_id')
        }
        
        return self._dispatch_action("fetch_history", payload, calling_context)

    def handle_get_attachment(self, 
                              attachment_id: str,
                              conversation_id: Optional[str] = None, # Optional, try context first
                              message_external_id: Optional[str] = None, # Optional, for context
                              calling_context: Dict[str, Any] = None):
        """
        Tool to request the content of a specific attachment from the adapter.

        Args:
            attachment_id: The unique ID of the attachment to fetch (required).
            conversation_id: Optional external ID of the conversation (if context is ambiguous).
            message_external_id: Optional external ID of the message the attachment belongs to (for context).
            calling_context: Context from the loop component calling the tool.
        
        Returns:
            Result of the action dispatch (e.g., confirmation or error).
        """
        if not calling_context:
            logger.warning(f"[{self.owner.id}] handle_get_attachment called without calling_context.")
            calling_context = {}
        if not attachment_id:
            return { "status": "error", "message": "attachment_id is required."}

        # --- Determine Context (Adapter ID & Conversation ID) ---
        adapter_id = None
        conv_id_to_use = conversation_id # Prioritize explicitly passed ID
        
        if not conv_id_to_use:
            # If not passed, try to get from context
            context_result = self._get_message_context()
            if not context_result['success']:
                logger.error(f"[{self.owner.id}] Failed to get message context for handle_get_attachment: {context_result['error']}")
                return { "status": "error", "message": f"Failed to determine adapter context for attachment {attachment_id}" }
            adapter_id = context_result['adapter_id']
            conv_id_to_use = context_result['conversation_id']
        else:
            # If conv_id was passed, we still need the adapter_id from context
            context_result = self._get_message_context(use_external_conversation_id=conv_id_to_use)
            if not context_result['success']:
                logger.error(f"[{self.owner.id}] Failed to get adapter context for handle_get_attachment (conv_id provided): {context_result['error']}")
                return { "status": "error", "message": f"Failed to determine adapter context for conversation {conv_id_to_use}" }
            adapter_id = context_result['adapter_id']
        
        if not adapter_id or not conv_id_to_use:
            return { "status": "error", "message": f"Could not determine adapter_id ({adapter_id}) or conversation_id ({conv_id_to_use}) for getting attachment."}

        logger.info(f"[{self.owner.id}] Preparing get_attachment action for adapter '{adapter_id}', conv '{conv_id_to_use}', attachment '{attachment_id}'.")

        payload = {
            "adapter_id": adapter_id,
            "conversation_id": conv_id_to_use,
            "attachment_id": attachment_id,
            "message_external_id": message_external_id, # Pass along if provided
            "requesting_element_id": self.owner.id, 
            "calling_loop_id": calling_context.get('loop_component_id')
        }
        
        return self._dispatch_action("get_attachment", payload, calling_context)

    # --- Register other tools as needed ---
