"""
Message Action Handler Component
Provides tools/actions for interacting with messaging elements (e.g., sending messages).
"""
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, List

from ...base import Component
from ..tool_provider import ToolParameter # Import the new ToolParameter type

# Assuming MessageListComponent is used for context, but not strictly required by handler itself
from .message_list import MessageListComponent 

if TYPE_CHECKING:
    from ...base import BaseElement
    from ..tool_provider import ToolProviderComponent # In components directory
    from host.event_loop import OutgoingActionCallback # In host directory
    from elements.elements.inner_space import InnerSpace # To check if owner is InnerSpace for DMs
    from elements.elements.space import Space # To check if owner is Space (SharedSpace context)
    from elements.elements.uplink import UplinkProxy # For type checking

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
    
    def __init__(self, element: Optional[BaseElement] = None, outgoing_action_callback: Optional['OutgoingActionCallback'] = None, **kwargs): # Added element, updated kwargs
        super().__init__(element, **kwargs) # Pass element and kwargs
        self._outgoing_action_callback = outgoing_action_callback
        if not self._outgoing_action_callback:
             logger.warning(f"MessageActionHandler {self.id if self.id else 'UNKNOWN_ID'} initialized without an outgoing_action_callback. Cannot send external messages.")

    def initialize(self, **kwargs) -> None:
        """Registers tools after initialization."""
        super().initialize(**kwargs)
        self._register_messaging_tools()
        logger.debug(f"MessageActionHandler initialized and tools registered for Element {self.owner.id if self.owner else 'Unknown'}")

    def set_outgoing_action_callback(self, callback: 'OutgoingActionCallback'):
        """Allows setting the callback after initialization if needed."""
        self._outgoing_action_callback = callback
        logger.debug(f"Outgoing action callback set for MessageActionHandler {self.id if self.id else 'UNKNOWN_ID'}")

    def _register_messaging_tools(self):
        """Registers messaging-related tools with the ToolProviderComponent."""
        tool_provider: Optional['ToolProviderComponent'] = self.get_sibling_component("ToolProviderComponent")
        if not tool_provider:
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Cannot register messaging tools: ToolProviderComponent not found.")
            return

        # --- Define Parameter Schemas ---
        send_message_params: List[ToolParameter] = [
            {"name": "text", "type": "string", "description": "The content of the message to send.", "required": True},
            {"name": "reply_to_external_id", "type": "string", "description": "Optional external ID of the message being replied to.", "required": False},
            {
                "name": "attachments", "type": "array", "description": "Optional list of attachment objects to send.", "required": False,
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL of the attachment."},
                        "filename": {"type": "string", "description": "Filename of the attachment."},
                        "attachment_type": {"type": "string", "description": "MIME type (e.g., 'image/png')."}
                    },
                    "required": ["url"] # Example: URL is required for an attachment object
                }
            }
        ]

        delete_message_params: List[ToolParameter] = [
            {"name": "message_external_id", "type": "string", "description": "The external ID of the message to delete.", "required": True}
        ]

        edit_message_params: List[ToolParameter] = [
            {"name": "message_external_id", "type": "string", "description": "The external ID of the message to edit.", "required": True},
            {"name": "new_text", "type": "string", "description": "The new text content for the message.", "required": True}
        ]

        add_reaction_params: List[ToolParameter] = [
            {"name": "message_external_id", "type": "string", "description": "The external ID of the message to react to.", "required": True},
            {"name": "emoji", "type": "string", "description": "The emoji to add as a reaction (e.g., '👍', ':smile:').", "required": True}
        ]
        
        remove_reaction_params: List[ToolParameter] = [
            {"name": "message_external_id", "type": "string", "description": "The external ID of the message to remove reaction from.", "required": True},
            {"name": "emoji", "type": "string", "description": "The emoji reaction to remove (e.g., '👍', ':smile:').", "required": True}
        ]

        fetch_history_params: List[ToolParameter] = [
            {"name": "conversation_id", "type": "string", "description": "The external ID of the conversation/channel to fetch history from.", "required": True},
            {"name": "before_ms", "type": "integer", "description": "Fetch messages before this UTC timestamp in milliseconds. (Optional)", "required": False},
            {"name": "after_ms", "type": "integer", "description": "Fetch messages after this UTC timestamp in milliseconds. (Optional)", "required": False},
            {"name": "limit", "type": "integer", "description": "Maximum number of messages to fetch (e.g., 100). (Optional)", "required": False}
        ]

        get_message_attachment_content_params: List[ToolParameter] = [
            {"name": "message_external_id", "type": "string", "description": "The external ID of the message containing the attachment.", "required": True},
            {"name": "attachment_id", "type": "string", "description": "The unique ID of the attachment to fetch.", "required": True}
        ]

        # --- Register send_message Tool --- 
        @tool_provider.register_tool(
            name="send_message",
            description="Sends a message to the chat context represented by this element.",
            parameters=send_message_params
        )
        def send_message_tool(text: str, 
                              reply_to_external_id: Optional[str] = None, 
                              attachments: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
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
                "action_type": "send_message", # Generic action type for ActivityClient
                "payload": {
                    "adapter_id": adapter_id,
                    "conversation_id": conversation_id,
                    "text": text,
                    "reply_to_external_id": reply_to_external_id, # Pass along for threading
                    "attachments": attachments or [], # Pass along attachments
                    "requesting_element_id": self.owner.id, # For context/logging in ActivityClient
                    # Add agent_id if the owner is in an InnerSpace for better tracking
                    "requesting_agent_id": self.owner.parent.id if (hasattr(self.owner, 'parent') and self.owner.parent and hasattr(self.owner.parent, 'IS_INNER_SPACE')) else None
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
            description="Deletes a message specified by its external ID from the conversation this element represents.",
            parameters=delete_message_params
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
            parameters=edit_message_params
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
            parameters=add_reaction_params
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
            parameters=remove_reaction_params
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

        # --- Register fetch_history Tool ---
        @tool_provider.register_tool(
            name="fetch_message_history",
            description="Fetches historical messages for a specific conversation from the adapter.",
            parameters=fetch_history_params
        )
        def fetch_history_tool(conversation_id: str, # Explicitly required by tool
                                 before_ms: Optional[int] = None,
                                 after_ms: Optional[int] = None,
                                 limit: Optional[int] = 100) -> Dict[str, Any]:
            # This tool might be called by an agent loop that doesn't have a `calling_context`
            # in the same way a direct user tool might.
            # For now, we pass an empty dict for calling_context.
            # The agent loop should ideally provide its own ID.
            return self.handle_fetch_history(
                conversation_id=conversation_id,
                before_ms=before_ms,
                after_ms=after_ms,
                limit=limit,
                calling_context={} # Placeholder
            )

        # --- Register get_message_attachment_content Tool (modified from get_attachment_tool) --- 
        @tool_provider.register_tool(
            name="get_message_attachment_content",
            description="Retrieves the content of a specific attachment. If not locally cached, initiates a fetch from the adapter.",
            parameters=get_message_attachment_content_params # Use updated params
        )
        def get_message_attachment_content_tool(message_external_id: str, attachment_id: str) -> Dict[str, Any]:
            if not self._outgoing_action_callback: # _dispatch_action checks this, but good for early exit
                return {"success": False, "error": "Outgoing action callback is not configured."}

            logger.debug(f"[{self.owner.id}] Tool 'get_message_attachment_content' called for msg_ext_id: {message_external_id}, att_id: {attachment_id}")

            msg_list_comp = self.get_sibling_component(MessageListComponent)
            if not msg_list_comp:
                return {"success": False, "error": "MessageListComponent not found on element."}

            # Find the message in MessageListComponent
            target_message: Optional[Dict[str, Any]] = None
            for msg in msg_list_comp.get_messages(): # get_messages returns List[MessageType]
                if msg.get('original_external_id') == message_external_id:
                    target_message = msg
                    break
            
            if not target_message:
                return {"success": False, "error": f"Message with external ID '{message_external_id}' not found."}

            # Find the attachment in the message
            target_attachment: Optional[Dict[str, Any]] = None
            for att in target_message.get('attachments', []):
                if att.get('attachment_id') == attachment_id:
                    target_attachment = att
                    break
            
            if not target_attachment:
                return {"success": False, "error": f"Attachment with ID '{attachment_id}' not found in message '{message_external_id}'."}

            # Check if content is already available
            if target_attachment.get('content') is not None:
                logger.info(f"[{self.owner.id}] Attachment content for '{attachment_id}' found directly in MessageListComponent.")
                return {
                    "success": True, 
                    "status": "content_retrieved",
                    "filename": target_attachment.get('filename'),
                    "content_type": target_attachment.get('content_type'),
                    "content": target_attachment.get('content')
                }

            # If content not available, but URL is, initiate fetch
            attachment_url = target_attachment.get('url')
            if not attachment_url:
                return {"success": False, "error": f"Attachment '{attachment_id}' in message '{message_external_id}' has no content and no URL to fetch from."}

            # Determine context for dispatching the fetch action
            adapter_id, conversation_id = self._get_message_context()
            if not adapter_id or not conversation_id:
                 # _get_message_context logs error if it fails
                return {"success": False, "error": "Failed to determine adapter/conversation context for fetching attachment."}

            logger.info(f"[{self.owner.id}] Attachment content for '{attachment_id}' not cached. Initiating fetch from URL: {attachment_url}")
            
            action_payload = {
                "adapter_id": adapter_id,
                "conversation_id": conversation_id, # Context for where the original message was
                "message_external_id": message_external_id, # Original message ID
                "attachment_id": attachment_id,
                "attachment_url": attachment_url, # URL to fetch from
                "requesting_element_id": self.owner.id,
                # Pass through original is_dm flag from context if available in _get_message_context, ActivityClient might need it
                # "is_dm": self._get_message_context().get('is_dm_context', False) # Assuming _get_message_context can provide this
            }

            # Use _dispatch_action to send to ActivityClient
            dispatch_result = self._dispatch_action(
                action_type="fetch_attachment_content", # New action type for ActivityClient to handle
                payload=action_payload
            )
            
            # _dispatch_action already formats success/error, but we can add status
            if dispatch_result.get("success"):
                return {
                    "success": True, 
                    "status": "fetch_initiated", 
                    "message_id": message_external_id, 
                    "attachment_id": attachment_id,
                    "detail": dispatch_result.get("status", "Fetch request sent to outgoing queue.")
                }
            else:
                return dispatch_result # Return the error from _dispatch_action

    def _get_message_context(self, use_external_conversation_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Determines the adapter_id and conversation_id based on the owner element's context.
        Can be an InnerSpace (for DMs via DM Elements), a SharedSpace, or an UplinkProxy.

        Args:
            use_external_conversation_id: If provided, this specific conversation_id is used (e.g., for fetch_history).
                                           Otherwise, derived from context.
        
        Returns:
            A tuple (adapter_id, conversation_id). Both can be None if context is unclear.
        """
        from elements.elements.inner_space import InnerSpace # Local import for type check
        from elements.elements.space import Space # Local import for type check
        from elements.elements.uplink import UplinkProxy # Local import for type check

        owner = self.owner
        adapter_id: Optional[str] = None
        conversation_id: Optional[str] = None
        # is_dm_context = False # Optional: to pass to ActivityClient if needed

        if not owner:
            logger.error(f"MessageActionHandler cannot determine context: No owner element.")
            return None, None

        # Case 1: Owner is an UplinkProxy
        if isinstance(owner, UplinkProxy):
            # UplinkProxy's remote_space_info should contain adapter_id and external_conversation_id of the remote space
            if hasattr(owner, 'remote_space_info') and owner.remote_space_info:
                remote_info = owner.remote_space_info.get('info', {})
                adapter_id = remote_info.get('adapter_id')
                # For uplinks, conversation_id is the remote space's external_conversation_id
                conversation_id = use_external_conversation_id if use_external_conversation_id else remote_info.get('external_conversation_id')
                # if remote_info.get('is_dm_space'): is_dm_context = True # Or however remote DM spaces are identified
                logger.debug(f"[{owner.id}] Uplink context: adapter='{adapter_id}', conv='{conversation_id}'. Remote info: {owner.remote_space_info}")
            else:
                logger.warning(f"[{owner.id}] UplinkProxy owner missing or has empty remote_space_info.")
        
        # Case 2: Owner's parent is InnerSpace (implies DM Element context)
        # This assumes DM Elements are direct children of InnerSpace, or their parent has this info.
        elif hasattr(owner, 'parent') and owner.parent and isinstance(owner.parent, InnerSpace):
            # The DM Element itself should store adapter_id and the other user's ID as conversation_id
            # This requires the DM Element (e.g. created by DirectMessageManagerComponent) to have these.
            # Let's assume the owner (DM Element) has .adapter_id and .external_conversation_id (user_id)
            if hasattr(owner, 'adapter_id'): adapter_id = owner.adapter_id
            if hasattr(owner, 'external_conversation_id'): # This would be the other user's ID
                conversation_id = use_external_conversation_id if use_external_conversation_id else owner.external_conversation_id
            # is_dm_context = True
            logger.debug(f"[{owner.id}] DM Element context in InnerSpace: adapter='{adapter_id}', conv(user_id)='{conversation_id}'.")

        # Case 3: Owner is a Space (e.g. a SharedSpace directly, less common for MessageActionHandler to be on SharedSpace itself)
        # More likely, MessageActionHandler is on an element *within* a SharedSpace.
        # If MessageActionHandler is on an element *mounted inside* a SharedSpace:
        elif hasattr(owner, 'parent') and owner.parent and isinstance(owner.parent, Space) and not isinstance(owner.parent, InnerSpace):
            # The parent Space (SharedSpace) should have adapter_id and external_conversation_id
            shared_space = owner.parent
            if hasattr(shared_space, 'adapter_id'): adapter_id = shared_space.adapter_id
            if hasattr(shared_space, 'external_conversation_id'): # This is the channel ID
                conversation_id = use_external_conversation_id if use_external_conversation_id else shared_space.external_conversation_id
            logger.debug(f"[{owner.id}] Element in SharedSpace context: adapter='{adapter_id}', conv(channel_id)='{conversation_id}'.")
        
        # Case 4: Owner is a Space directly (e.g. SharedSpace itself)
        elif isinstance(owner, Space) and not isinstance(owner, InnerSpace): # Should be a SharedSpace
            if hasattr(owner, 'adapter_id'): adapter_id = owner.adapter_id
            if hasattr(owner, 'external_conversation_id'):
                conversation_id = use_external_conversation_id if use_external_conversation_id else owner.external_conversation_id
            logger.debug(f"[{owner.id}] SharedSpace direct context: adapter='{adapter_id}', conv(channel_id)='{conversation_id}'.")
        else:
            logger.warning(f"[{owner.id if owner else 'NoOwner'}] MessageActionHandler cannot determine messaging context (adapter/conversation ID). Owner type: {type(owner)}")

        if not adapter_id or not conversation_id:
            logger.error(f"[{owner.id if owner else 'NoOwner'}] Failed to determine adapter_id ('{adapter_id}') or conversation_id ('{conversation_id}') for messaging action.")
            return None, None 
            
        return adapter_id, conversation_id

    def handle_fetch_history(self, 
                             conversation_id: str,
                             before_ms: Optional[int] = None,
                             after_ms: Optional[int] = None,
                             limit: Optional[int] = 100, 
                             calling_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: # Added Optional to calling_context
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
        calling_context = calling_context or {} # Ensure calling_context is a dict
            
        context_result = self._get_message_context(use_external_conversation_id=conversation_id)
        if not context_result.get('success'):
             return { "success": False, "error": context_result.get('error', f"Failed to determine adapter context for conversation {conversation_id}") }
        
        adapter_id = context_result['adapter_id']
        
        logger.info(f"[{self.owner.id if self.owner else 'Unknown'}] Preparing fetch_history action for adapter '{adapter_id}', conv '{conversation_id}'.")
        
        payload = {
            "adapter_id": adapter_id,
            "conversation_id": conversation_id, # This is the external_id
            "before_timestamp_ms": before_ms, 
            "after_timestamp_ms": after_ms,
            "limit": limit,
            "requesting_element_id": self.owner.id if self.owner else None,
            "calling_loop_id": calling_context.get('loop_component_id') # From AgentLoop
        }
        
        return self._dispatch_action("fetch_message_history", payload) # "fetch_message_history" is the ActivityClient action

    def handle_get_attachment(self, 
                              attachment_id: str,
                              conversation_id: Optional[str] = None, 
                              message_external_id: Optional[str] = None, 
                              calling_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: # Added Optional to calling_context
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
        calling_context = calling_context or {}
        if not attachment_id: # Should be caught by schema validation by LLM call
            return { "success": False, "error": "attachment_id is required."}

        context_result = self._get_message_context(use_external_conversation_id=conversation_id)
        if not context_result.get('success'):
            return { "success": False, "error": context_result.get('error', f"Failed to determine adapter context for attachment {attachment_id}") }
        
        adapter_id = context_result['adapter_id']
        # If conversation_id was passed to tool, it's used. Otherwise, context_result['conversation_id'] is used.
        actual_conversation_id = conversation_id if conversation_id else context_result['conversation_id']
        
        if not adapter_id or not actual_conversation_id:
            return { "success": False, "error": f"Could not determine adapter_id ({adapter_id}) or conversation_id ({actual_conversation_id}) for getting attachment."}

        logger.info(f"[{self.owner.id if self.owner else 'Unknown'}] Preparing get_attachment action for adapter '{adapter_id}', conv '{actual_conversation_id}', attachment '{attachment_id}'.")

        payload = {
            "adapter_id": adapter_id,
            "conversation_id": actual_conversation_id,
            "attachment_id": attachment_id,
            "message_external_id": message_external_id,
            "requesting_element_id": self.owner.id if self.owner else None,
            "calling_loop_id": calling_context.get('loop_component_id') # From AgentLoop
        }
        
        return self._dispatch_action("get_attachment_content", payload) # "get_attachment_content" is ActivityClient action

    def _dispatch_action(self, action_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to dispatch an action via the outgoing_action_callback."""
        if not self._outgoing_action_callback:
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Cannot dispatch '{action_type}': Outgoing action callback not configured.")
            return {"success": False, "error": "Outgoing action callback is not configured."}

        action_request = {
            "target_module": "ActivityClient",
            "action_type": action_type, # This is the type ActivityClient expects
            "payload": payload
        }
        try:
            self._outgoing_action_callback(action_request)
            logger.info(f"[{self.owner.id if self.owner else 'Unknown'}] Dispatched '{action_type}' to ActivityClient with payload: {payload}")
            return {"success": True, "status": f"'{action_type}' request sent to outgoing queue."}
        except Exception as e:
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Error dispatching '{action_type}' action: {e}", exc_info=True)
            return {"success": False, "error": f"Error dispatching '{action_type}' request: {e}"}
