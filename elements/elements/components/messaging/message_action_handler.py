"""
Message Action Handler Component
Provides tools/actions for interacting with messaging elements (e.g., sending messages).
"""
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING, Tuple, List
import uuid # Added for generating unique request IDs
import time # Added for timestamp

from ...base import Component, BaseElement
from ..tool_provider import ToolParameter # Import the new ToolParameter type

# Assuming MessageListComponent is used for context, but not strictly required by handler itself
from .message_list import MessageListComponent
from elements.component_registry import register_component
from opentelemetry import propagate

if TYPE_CHECKING:
    # from ...base import BaseElement # Removed from here
    from ..tool_provider import ToolProviderComponent # In components directory
    from host.event_loop import OutgoingActionCallback # In host directory
    from elements.elements.inner_space import InnerSpace # To check if owner is InnerSpace for DMs
    from elements.elements.space import Space # To check if owner is Space (SharedSpace context)
    from elements.elements.uplink import UplinkProxy # For type checking

logger = logging.getLogger(__name__)

@register_component
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

    def __init__(self, element: Optional[Any] = None, outgoing_action_callback: Optional['OutgoingActionCallback'] = None, **kwargs): # Added element, updated kwargs
        super().__init__(element, **kwargs) # Pass element and kwargs
        self._outgoing_action_callback = outgoing_action_callback


    def initialize(self, **kwargs) -> None:
        """Registers tools after initialization."""
        super().initialize(**kwargs)

        owner_id_for_log = "UnknownOwner"
        if self.owner and hasattr(self.owner, 'id'):
            owner_id_for_log = self.owner.id
        if not self._outgoing_action_callback or self._outgoing_action_callback is None:
             logger.warning(f"MessageActionHandler {self.id if self.id else 'UNKNOWN_ID'} on Element '{owner_id_for_log}' initialized without an outgoing_action_callback. Cannot send external messages.")
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
            {"name": "target_element_id", "type": "string", "description": "Optional specific element ID to target for sending. If not provided, will use the element this tool is attached to.", "required": False},
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
            {"name": "before", "type": "integer", "description": "Fetch messages before this UTC timestamp in seconds. For example, this param can be set to int(datetime.now().timestamp()). (Either this, or after param must be submitted)", "required": False},
            {"name": "after", "type": "integer", "description": "Fetch messages after this UTC timestamp in seconds. (Either this, or before param must be submitted)", "required": False},
            {"name": "limit", "type": "integer", "description": "Maximum number of messages to fetch (e.g., 100). (Optional)", "required": False}
        ]

        # get_message_attachment_content_params: List[ToolParameter] = [
        #     {"name": "message_external_id", "type": "string", "description": "The external ID of the message containing the attachment.", "required": True},
        #     {"name": "attachment_id", "type": "string", "description": "The unique ID of the attachment to fetch.", "required": True}
        # ]

        # --- Register send_message Tool ---
        @tool_provider.register_tool(
            name="send_message",
            description="Sends a message to the current conversation (DM or channel). If used via an Uplink to a SharedSpace, this requests the SharedSpace to send the message.",
            parameters_schema=send_message_params
        )
        async def send_message_tool(text: str,
                                    attachments: Optional[List[Dict[str, Any]]] = None,
                                    reply_to_external_id: Optional[str] = None,
                                    target_element_id: Optional[str] = None,
                                    calling_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """
            Tool function to send a message.
            Uses local outgoing_action_callback.

            Args:
                text: Message content to send
                attachments: Optional list of attachment objects
                reply_to_external_id: Optional ID of message being replied to
                target_element_id: Optional specific element ID to target. Usually not needed
                                  as the agent loop will automatically route to the correct element.
                calling_context: Context from the calling component
            """
            logger.info(f"[{self.owner.id}] MessageActionHandler.send_message_tool called. Text: '{text[:50]}...', Target: {target_element_id or 'auto'}")

            # Note: target_element_id parameter is included for completeness but typically
            # the agent loop handles routing automatically based on tool aggregation

            retrieved_adapter_id, retrieved_conversation_id = self._get_message_context()
            requesting_agent_id = self._get_requesting_agent_id(calling_context) # Pass context
            agent_name = self._get_requesting_agent_name(calling_context) # Pass context

            if not retrieved_adapter_id or not retrieved_conversation_id:
                error_msg = f"Cannot determine context (adapter_id: {retrieved_adapter_id}, conversation_id: {retrieved_conversation_id}) for sending message."
                logger.error(f"[{self.owner.id}] {error_msg}")
                return {"success": False, "error": error_msg, "message_id": None}

            internal_request_id = self._get_internal_request_id()

            # --- Standard dispatch using _outgoing_action_callback (e.g., for DMs or SharedSpace's chat_interface) ---
            if self._outgoing_action_callback is None:
                error_msg = "Outgoing action callback is not set. Cannot send message."
                logger.error(f"[{self.owner.id}] {error_msg}")
                return {"success": False, "error": error_msg, "message_id": None}

            msg_list_comp = self.owner.get_component_by_type("MessageListComponent")
            if msg_list_comp:
                # Agent name might be different from owner name if tools are on InnerSpace directly
                # For DMs, requesting_agent_id (self.owner.agent_id) and agent_name are relevant.
                final_attachments = []
                if attachments:
                    if isinstance(attachments, list):
                        for att in attachments:
                            if isinstance(att, dict):
                                final_attachments.append(att)
                            else:
                                logger.warning(f"[{self.owner.id}] send_message_tool: Skipping non-dict attachment: {att}")
                    else:
                        logger.warning(f"[{self.owner.id}] send_message_tool: Attachments argument was not a list: {attachments}")

                msg_list_comp.add_pending_message(
                    internal_request_id=internal_request_id,
                    text=text,
                    sender_id=requesting_agent_id or "unknown_agent",
                    sender_name=agent_name or "Unknown Agent",
                    timestamp=time.time(),
                    attachments=final_attachments,
                    reply_to_external_id=reply_to_external_id,
                    adapter_id=retrieved_adapter_id # Should be correct for DM context
                )
                logger.info(f"[{self.owner.id}] Added pending message (req_id: {internal_request_id}) to MessageListComponent for direct send.")
            else:
                logger.warning(f"[{self.owner.id}] MessageListComponent not found. Cannot add pending message locally before direct sending.")

            action_request = {
                "target_module": "ActivityClient",
                "action_type": "send_message",
                "payload": {
                    "internal_request_id": internal_request_id,
                    "adapter_id": retrieved_adapter_id,
                    "conversation_id": retrieved_conversation_id,
                    "text": text,
                    "reply_to_external_id": reply_to_external_id,
                    "attachments": attachments or [],
                    "requesting_element_id": self.owner.id,
                    "requesting_agent_id": requesting_agent_id,
                    "target_element_id": target_element_id
                }
            }
            logger.debug(f"[{self.owner.id}] Dispatching direct send_message action request: {action_request}")

            try:
                assert self._outgoing_action_callback is not None
                dispatch_result = await self._outgoing_action_callback(action_request)
                if dispatch_result and dispatch_result.get("success"):
                    logger.info(f"[{self.owner.id}] Direct send_message action successfully dispatched to ActivityClient for req_id: {internal_request_id}.")
                    return {"success": True, "status": "pending_confirmation", "internal_request_id": internal_request_id, "message_id": dispatch_result.get("message_id")}
                else:
                    error_msg = f"Failed to dispatch direct send_message action to ActivityClient. Result: {dispatch_result}"
                    logger.error(f"[{self.owner.id}] {error_msg} for req_id: {internal_request_id}")
                    return {"success": False, "error": error_msg, "message_id": None}
            except Exception as e:
                error_msg = f"Exception during direct send_message dispatch: {e}"
                logger.exception(f"[{self.owner.id}] {error_msg} for req_id: {internal_request_id}")
                return {"success": False, "error": error_msg, "message_id": None}

        # --- Register delete_message Tool ---
        @tool_provider.register_tool(
            name="delete_message",
            description="Deletes a message specified by its external ID from the conversation this element represents.",
            parameters_schema=delete_message_params
        )
        async def delete_message_tool(message_external_id: str, calling_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Handles deleting a message using its external ID."""
            if not self._outgoing_action_callback:
                return {"success": False, "error": "Outgoing action callback is not configured."}
            if not message_external_id:
                return {"success": False, "error": "message_external_id is required."}

            adapter_id, conversation_id = self._get_message_context()
            if not adapter_id or not conversation_id:
                logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] delete_message_tool: _get_message_context failed.")
                return {"success": False, "error": f"Could not determine context for deleting message."}

            requesting_agent_id = self._get_requesting_agent_id(calling_context)

            # NEW: Immediately update local state before external dispatch
            msg_list_comp = self.get_sibling_component(MessageListComponent)
            if msg_list_comp:
                local_update_success = msg_list_comp.mark_message_pending_delete(message_external_id, requesting_agent_id or "unknown_agent")
                if local_update_success:
                    logger.info(f"[{self.owner.id}] Immediately marked message '{message_external_id}' as pending deletion in local state")

                    # Trigger VEIL update to show pending state immediately
                    veil_producer = self.get_sibling_component("MessageListVeilProducer")
                    if veil_producer:
                        veil_producer.emit_delta()
                else:
                    logger.warning(f"[{self.owner.id}] Could not mark message '{message_external_id}' as pending delete in local state")
            else:
                logger.warning(f"[{self.owner.id}] MessageListComponent not found for immediate local state update")

            action_request = {
                "target_module": "ActivityClient",
                "action_type": "delete_message",
                "payload": {
                    "internal_request_id": self._get_internal_request_id(),
                    "adapter_id": adapter_id,
                    "conversation_id": conversation_id,
                    "message_external_id": message_external_id,
                    "requesting_element_id": self.owner.id,
                    "requesting_agent_id": requesting_agent_id
                }
            }
            try:
                await self._outgoing_action_callback(action_request)
                logger.info(f"[{self.owner.id}] Dispatched 'delete_message' action for ID '{message_external_id}' to adapter '{adapter_id}'.")
                return {"success": True, "status": "Delete request sent. Message marked as pending deletion in conversation.", "message_external_id": message_external_id}
            except Exception as e:
                # NEW: Restore message state if dispatch fails
                if msg_list_comp:
                    msg_list_comp.restore_message_from_pending_state(message_external_id, "delete")
                    if veil_producer:
                        veil_producer.emit_delta()  # Update VEIL to show restore
                logger.error(f"[{self.owner.id}] Error dispatching delete_message action: {e}", exc_info=True)
                return {"success": False, "error": f"Error dispatching delete request: {e}"}

        # --- Register edit_message Tool ---
        @tool_provider.register_tool(
            name="edit_message",
            description="Edits an existing message specified by its external ID.",
            parameters_schema=edit_message_params
        )
        async def edit_message_tool(message_external_id: str, new_text: str, calling_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Handles editing a message using its external ID."""
            if not self._outgoing_action_callback:
                return {"success": False, "error": "Outgoing action callback is not configured."}
            if not message_external_id or not new_text:
                return {"success": False, "error": "message_external_id and new_text are required."}

            adapter_id, conversation_id = self._get_message_context()
            if not adapter_id or not conversation_id:
                logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] edit_message_tool: _get_message_context failed.")
                return {"success": False, "error": f"Could not determine context for editing message."}

            requesting_agent_id = self._get_requesting_agent_id(calling_context)

            # NEW: Immediately update local state before external dispatch
            msg_list_comp = self.get_sibling_component(MessageListComponent)
            if msg_list_comp:
                local_update_success = msg_list_comp.mark_message_pending_edit(message_external_id, new_text, requesting_agent_id or "unknown_agent")
                if local_update_success:
                    logger.info(f"[{self.owner.id}] Immediately marked message '{message_external_id}' as pending edit in local state")

                    # Trigger VEIL update to show pending state immediately
                    veil_producer = self.get_sibling_component("MessageListVeilProducer")
                    if veil_producer:
                        veil_producer.emit_delta()
                else:
                    logger.warning(f"[{self.owner.id}] Could not mark message '{message_external_id}' as pending edit in local state")
            else:
                logger.warning(f"[{self.owner.id}] MessageListComponent not found for immediate local state update")

            action_request = {
                "target_module": "ActivityClient",
                "action_type": "edit_message",
                "payload": {
                    "internal_request_id": self._get_internal_request_id(),
                    "adapter_id": adapter_id,
                    "conversation_id": conversation_id,
                    "message_external_id": message_external_id,
                    "new_text": new_text,
                    "requesting_element_id": self.owner.id,
                    "requesting_agent_id": requesting_agent_id
                }
            }

            try:
                await self._outgoing_action_callback(action_request)
                logger.info(f"[{self.owner.id}] Dispatched 'edit_message' action for ID '{message_external_id}' to adapter '{adapter_id}'.")
                return {"success": True, "status": "Edit request sent. Message updated with new text in conversation.", "message_external_id": message_external_id}
            except Exception as e:
                 # NEW: Restore message state if dispatch fails
                 if msg_list_comp:
                     msg_list_comp.restore_message_from_pending_state(message_external_id, "edit")
                     if veil_producer:
                         veil_producer.emit_delta()  # Update VEIL to show restore
                 logger.error(f"[{self.owner.id}] Error dispatching edit_message action: {e}", exc_info=True)
                 return {"success": False, "error": f"Error dispatching edit request: {e}"}

        # --- Register add_reaction Tool ---
        @tool_provider.register_tool(
            name="add_reaction",
            description="Adds an emoji reaction to a message specified by its external ID.",
            parameters_schema=add_reaction_params
        )
        async def add_reaction_tool(message_external_id: str, emoji: str, calling_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Handles adding a reaction to a message using its external ID."""
            if not self._outgoing_action_callback:
                return {"success": False, "error": "Outgoing action callback is not configured."}
            if not message_external_id or not emoji:
                return {"success": False, "error": "message_external_id and emoji are required."}

            adapter_id, conversation_id = self._get_message_context()
            if not adapter_id or not conversation_id:
                logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] add_reaction_tool: _get_message_context failed.")
                return {"success": False, "error": f"Could not determine context for adding reaction."}

            requesting_agent_id = self._get_requesting_agent_id(calling_context)

            # NEW: Immediately update local state before external dispatch
            msg_list_comp = self.get_sibling_component(MessageListComponent)
            if msg_list_comp:
                local_update_success = msg_list_comp.add_pending_reaction(message_external_id, emoji, requesting_agent_id or "unknown_agent")
                if local_update_success:
                    logger.info(f"[{self.owner.id}] Immediately added pending reaction '{emoji}' to message '{message_external_id}' in local state")

                    # Trigger VEIL update to show pending state immediately
                    veil_producer = self.get_sibling_component("MessageListVeilProducer")
                    if veil_producer:
                        veil_producer.emit_delta()
                else:
                    logger.warning(f"[{self.owner.id}] Could not add pending reaction '{emoji}' to message '{message_external_id}' in local state")
            else:
                logger.warning(f"[{self.owner.id}] MessageListComponent not found for immediate local state update")

            action_request = {
                "target_module": "ActivityClient",
                "action_type": "add_reaction",
                "payload": {
                    "internal_request_id": self._get_internal_request_id(),
                    "adapter_id": adapter_id,
                    "conversation_id": conversation_id,
                    "message_external_id": message_external_id,
                    "emoji": emoji,
                    "requesting_element_id": self.owner.id,
                    "requesting_agent_id": requesting_agent_id
                }
            }

            try:
                await self._outgoing_action_callback(action_request)
                logger.info(f"[{self.owner.id}] Dispatched 'add_reaction' ({emoji}) action for ID '{message_external_id}' to adapter '{adapter_id}'.")
                return {"success": True, "status": f"Reaction '{emoji}' added to message in conversation.", "message_external_id": message_external_id}
            except Exception as e:
                # NEW: Restore message state if dispatch fails
                if msg_list_comp:
                    msg_list_comp.restore_message_from_pending_state(message_external_id, "add_reaction")
                    if veil_producer:
                        veil_producer.emit_delta()  # Update VEIL to show restore
                logger.error(f"[{self.owner.id}] Error dispatching add_reaction action: {e}", exc_info=True)
                return {"success": False, "error": f"Error dispatching add reaction request: {e}"}

        # --- Register remove_reaction Tool ---
        @tool_provider.register_tool(
            name="remove_reaction",
            description="Removes an emoji reaction (previously added by this agent/bot) from a message specified by its external ID.",
            parameters_schema=remove_reaction_params
        )
        async def remove_reaction_tool(message_external_id: str, emoji: str, calling_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Handles removing a reaction from a message using its external ID."""
            if not self._outgoing_action_callback:
                return {"success": False, "error": "Outgoing action callback is not configured."}
            if not message_external_id or not emoji:
                return {"success": False, "error": "message_external_id and emoji are required."}

            adapter_id, conversation_id = self._get_message_context()
            if not adapter_id or not conversation_id:
                logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] remove_reaction_tool: _get_message_context failed.")
                return {"success": False, "error": f"Could not determine context for removing reaction."}

            requesting_agent_id = self._get_requesting_agent_id(calling_context)

            # NEW: Immediately update local state before external dispatch
            msg_list_comp = self.get_sibling_component(MessageListComponent)
            if msg_list_comp:
                local_update_success = msg_list_comp.remove_pending_reaction(message_external_id, emoji, requesting_agent_id or "unknown_agent")
                if local_update_success:
                    logger.info(f"[{self.owner.id}] Immediately removed pending reaction '{emoji}' from message '{message_external_id}' in local state")

                    # Trigger VEIL update to show pending state immediately
                    veil_producer = self.get_sibling_component("MessageListVeilProducer")
                    if veil_producer:
                        veil_producer.emit_delta()
                else:
                    logger.warning(f"[{self.owner.id}] Could not remove pending reaction '{emoji}' from message '{message_external_id}' in local state")
            else:
                logger.warning(f"[{self.owner.id}] MessageListComponent not found for immediate local state update")

            action_request = {
                "target_module": "ActivityClient",
                "action_type": "remove_reaction",
                "payload": {
                    "internal_request_id": self._get_internal_request_id(),
                    "adapter_id": adapter_id,
                    "conversation_id": conversation_id,
                    "message_external_id": message_external_id,
                    "emoji": emoji,
                    "requesting_element_id": self.owner.id,
                    "requesting_agent_id": requesting_agent_id
                }
            }

            try:
                await self._outgoing_action_callback(action_request)
                logger.info(f"[{self.owner.id}] Dispatched 'remove_reaction' ({emoji}) action for ID '{message_external_id}' to adapter '{adapter_id}'.")
                return {"success": True, "status": f"Reaction '{emoji}' removed from message in conversation.", "message_external_id": message_external_id}
            except Exception as e:
                # NEW: Restore message state if dispatch fails
                if msg_list_comp:
                    msg_list_comp.restore_message_from_pending_state(message_external_id, "remove_reaction")
                    if veil_producer:
                        veil_producer.emit_delta()  # Update VEIL to show restore
                logger.error(f"[{self.owner.id}] Error dispatching remove_reaction action: {e}", exc_info=True)
                return {"success": False, "error": f"Error dispatching remove reaction request: {e}"}

        # --- Register fetch_history Tool ---
        @tool_provider.register_tool(
            name="fetch_history",
            description="Fetches historical messages for a specific conversation from the adapter.",
            parameters_schema=fetch_history_params
        )
        async def fetch_history_tool(conversation_id: str, # Explicitly required by tool
                                 before: Optional[int] = None,
                                 after: Optional[int] = None,
                                 limit: Optional[int] = 100,
                                 calling_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            # This tool might be called by an agent loop that doesn't have a `calling_context`
            # in the same way a direct user tool might.
            # The agent loop should ideally provide its own ID.
            return await self.handle_fetch_history(
                conversation_id=conversation_id,
                before=before,
                after=after,
                limit=limit,
                calling_context=calling_context # Pass context through
            )

        # --- Register get_message_attachment_content Tool (modified from get_attachment_tool) ---
        # @tool_provider.register_tool(
        #     name="get_message_attachment_content",
        #     description="Retrieves the content of a specific attachment. If not locally cached, initiates a fetch from the adapter.",
        #     parameters_schema=get_message_attachment_content_params # Use updated params
        # )
        # async def get_message_attachment_content_tool(message_external_id: str, attachment_id: str, calling_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        #     if not self._outgoing_action_callback: # _dispatch_action checks this, but good for early exit
        #         return {"success": False, "error": "Outgoing action callback is not configured."}

        #     logger.debug(f"[{self.owner.id}] Tool 'get_message_attachment_content' called for msg_ext_id: {message_external_id}, att_id: {attachment_id}")

        #     msg_list_comp = self.get_sibling_component(MessageListComponent)
        #     if not msg_list_comp:
        #         return {"success": False, "error": "MessageListComponent not found on element."}

        #     # Find the message in MessageListComponent
        #     target_message: Optional[Dict[str, Any]] = None
        #     for msg in msg_list_comp.get_messages(): # get_messages returns List[MessageType]
        #         if msg.get('original_external_id') == message_external_id:
        #             target_message = msg
        #             break

        #     if not target_message:
        #         return {"success": False, "error": f"Message with external ID '{message_external_id}' not found."}

        #     # Find the attachment in the message
        #     target_attachment: Optional[Dict[str, Any]] = None
        #     for att in target_message.get('attachments', []):
        #         if att.get('attachment_id') == attachment_id:
        #             target_attachment = att
        #             break

        #     if not target_attachment:
        #         return {"success": False, "error": f"Attachment with ID '{attachment_id}' not found in message '{message_external_id}'."}

        #     # Check if content is already available
        #     if target_attachment.get('content') is not None:
        #         logger.info(f"[{self.owner.id}] Attachment content for '{attachment_id}' found directly in MessageListComponent.")
        #         return {
        #             "success": True,
        #             "status": "content_retrieved",
        #             "filename": target_attachment.get('filename'),
        #             "content_type": target_attachment.get('content_type'),
        #             "content": target_attachment.get('content')
        #         }

        #     # If content not available, but URL is, initiate fetch
        #     attachment_url = target_attachment.get('url')
        #     if not attachment_url:
        #         return {"success": False, "error": f"Attachment '{attachment_id}' in message '{message_external_id}' has no content and no URL to fetch from."}

        #     # Determine context for dispatching the fetch action
        #     adapter_id, conversation_id = self._get_message_context(use_external_conversation_id=conversation_id)
        #     if not adapter_id or not conversation_id:
        #          # _get_message_context logs error if it fails
        #         logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] get_message_attachment_content_tool: _get_message_context failed for fetching attachment.")
        #         return {"success": False, "error": "Failed to determine adapter/conversation context for fetching attachment."}

        #     # Generate internal request ID for tracking like send_message does
        #     internal_request_id = f"fetch_att_req_{self.owner.id}_{uuid.uuid4().hex[:12]}"

        #     logger.info(f"[{self.owner.id}] Attachment content for '{attachment_id}' not cached. Initiating fetch from URL: {attachment_url}")

        #     action_payload = {
        #         "internal_request_id": internal_request_id,
        #         "adapter_id": adapter_id,
        #         "conversation_id": conversation_id, # Context for where the original message was
        #         "message_external_id": message_external_id, # Original message ID
        #         "attachment_id": attachment_id,
        #         "attachment_url": attachment_url, # URL to fetch from
        #         "requesting_element_id": self.owner.id,
        #         "requesting_agent_id": self._get_requesting_agent_id(calling_context) # Pass context
        #         # Pass through original is_dm flag from context if available in _get_message_context, ActivityClient might need it
        #         # "is_dm": self._get_message_context().get('is_dm_context', False) # Assuming _get_message_context can provide this
        #     }

        #     # Use _dispatch_action to send to ActivityClient
        #     dispatch_result = await self._dispatch_action(
        #         action_type="fetch_attachment_content", # New action type for ActivityClient to handle
        #         payload=action_payload
        #     )

        #     # _dispatch_action already formats success/error, but we can add status
        #     if dispatch_result.get("success"):
        #         return {
        #             "success": True,
        #             "status": "fetch_initiated",
        #             "message_id": message_external_id,
        #             "attachment_id": attachment_id,
        #             "detail": dispatch_result.get("status", "Fetch request sent to outgoing queue.")
        #         }
        #     else:
        #         return dispatch_result # Return the error from _dispatch_action

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

        # Get the parent object of the owner element
        parent_element_obj = owner.get_parent_object()

        # Case 1: Owner is an UplinkProxy (does not need parent for its primary info like adapter_id/conv_id)
        if isinstance(owner, UplinkProxy):
            if hasattr(owner, 'remote_space_info') and owner.remote_space_info:
                remote_info = owner.remote_space_info
                adapter_id = remote_info.get('adapter_id')
                conversation_id = use_external_conversation_id if use_external_conversation_id else remote_info.get('external_conversation_id')
                logger.debug(f"[{owner.id}] Uplink context: adapter='{adapter_id}', conv='{conversation_id}'. Remote info: {owner.remote_space_info}")
            else:
                logger.warning(f"[{owner.id}] UplinkProxy owner missing or has empty remote_space_info.")

        # Case 2: Owner is a DM Element (has dm_adapter_id, dm_external_conversation_id directly)
        # Parent object (InnerSpace) is primarily for agent_id, not adapter/conversation for the DM itself.
        elif hasattr(owner, 'dm_adapter_id') and hasattr(owner, 'dm_external_conversation_id'):
            adapter_id = owner.dm_adapter_id
            conversation_id = use_external_conversation_id if use_external_conversation_id else owner.dm_external_conversation_id
            logger.debug(f"[{owner.id}] DM Element direct attributes context: adapter='{adapter_id}', conv(user_id)='{conversation_id}'.")
            # We expect parent_element_obj to be an InnerSpace here if all is correct.
            if parent_element_obj and not isinstance(parent_element_obj, InnerSpace):
                logger.warning(f"[{owner.id}] DM Element's parent object (ID: {parent_element_obj.id if parent_element_obj else 'None'}) is not an InnerSpace as expected.")

        # Case 2b: Owner is a chat element with generic attributes (created by updated ChatManagerComponent)
        elif hasattr(owner, 'adapter_id') and hasattr(owner, 'external_conversation_id'):
            adapter_id = owner.adapter_id
            conversation_id = use_external_conversation_id if use_external_conversation_id else owner.external_conversation_id
            logger.debug(f"[{owner.id}] Chat Element direct attributes context: adapter='{adapter_id}', conv='{conversation_id}'.")
            # We expect parent_element_obj to be an InnerSpace here if all is correct.
            if parent_element_obj and not isinstance(parent_element_obj, InnerSpace):
                logger.warning(f"[{owner.id}] Chat Element's parent object (ID: {parent_element_obj.id if parent_element_obj else 'None'}) is not an InnerSpace as expected.")

        # Case 3: Owner is an element whose parent is a SharedSpace providing the context
        elif parent_element_obj and isinstance(parent_element_obj, Space) and not isinstance(parent_element_obj, InnerSpace):
            shared_space_parent = parent_element_obj
            if hasattr(shared_space_parent, 'adapter_id'): adapter_id = shared_space_parent.adapter_id
            if hasattr(shared_space_parent, 'external_conversation_id'):
                conversation_id = use_external_conversation_id if use_external_conversation_id else shared_space_parent.external_conversation_id
            logger.debug(f"[{owner.id}] Element in SharedSpace context (Parent: {parent_element_obj.id}): adapter='{adapter_id}', conv(channel_id)='{conversation_id}'.")

        # Case 4: Owner is a Space itself (e.g. SharedSpace, less common for MessageActionHandler to be directly on it)
        elif isinstance(owner, Space) and not isinstance(owner, InnerSpace):
            if hasattr(owner, 'adapter_id'): adapter_id = owner.adapter_id
            if hasattr(owner, 'external_conversation_id'):
                conversation_id = use_external_conversation_id if use_external_conversation_id else owner.external_conversation_id
            logger.debug(f"[{owner.id}] SharedSpace direct context: adapter='{adapter_id}', conv(channel_id)='{conversation_id}'.")
        else:
            parent_id_for_log = owner.get_parent_info().get('parent_id') if owner.get_parent_info() else "Unknown"
            logger.warning(f"[{owner.id if owner else 'NoOwner'}] MessageActionHandler cannot determine full messaging context. Owner type: {type(owner)}, Parent ID: {parent_id_for_log}, Parent Obj: {type(parent_element_obj) if parent_element_obj else 'None'}")

        if not adapter_id or not conversation_id:
            logger.error(f"[{owner.id if owner else 'NoOwner'}] Failed to determine adapter_id ('{adapter_id}') or conversation_id ('{conversation_id}') for messaging action.")
            return None, None

        return adapter_id, conversation_id

    def _get_requesting_agent_id(self, calling_context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        from elements.elements.inner_space import InnerSpace
        """Helper to get agent_id.
           Priority: calling_context, then owner's InnerSpace context."""
        calling_context = calling_context or {}

        # 1. Try to get from calling_context (e.g., for remote actions)
        context_agent_id = calling_context.get('source_agent_id')
        if context_agent_id:
            logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] Retrieved agent_id '{context_agent_id}' from calling_context.")
            return context_agent_id

        # 2. Fallback to owner/parent InnerSpace context (e.g., for DMs)
        if not self.owner:
            logger.warning(f"[{self.id if hasattr(self, 'id') else 'UnknownMAH'}] _get_requesting_agent_id: No owner element.")
            return None

        parent_obj = self.owner.get_parent_object()
        if parent_obj and isinstance(parent_obj, InnerSpace) and hasattr(parent_obj, 'agent_id'):
            logger.debug(f"[{self.owner.id}] Retrieved agent_id '{parent_obj.agent_id}' from parent InnerSpace.")
            return parent_obj.agent_id

        if isinstance(self.owner, InnerSpace) and hasattr(self.owner, 'agent_id'):
            logger.debug(f"[{self.owner.id}] Retrieved agent_id '{self.owner.agent_id}' from owner InnerSpace.")
            return self.owner.agent_id

        logger.warning(f"[{self.owner.id if self.owner else 'Unknown'}] Could not determine requesting_agent_id through calling_context or InnerSpace hierarchy.")
        return None

    def _get_requesting_agent_name(self, calling_context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        from elements.elements.inner_space import InnerSpace
        """Helper to get agent_name.
           Priority: calling_context, then owner's InnerSpace context."""
        calling_context = calling_context or {}

        # 1. Try to get from calling_context (e.g., for remote actions)
        context_agent_name = calling_context.get('source_agent_name')
        if context_agent_name:
            logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] Retrieved agent_name '{context_agent_name}' from calling_context.")
            return context_agent_name

        # 2. Fallback to owner/parent InnerSpace context (e.g., for DMs)
        if not self.owner:
            logger.warning(f"[{self.id if hasattr(self, 'id') else 'UnknownMAH'}] _get_requesting_agent_name: No owner element.")
            return None

        parent_obj = self.owner.get_parent_object()
        if parent_obj and isinstance(parent_obj, InnerSpace) and hasattr(parent_obj, 'agent_name'): # Check for agent_name
            logger.debug(f"[{self.owner.id}] Retrieved agent_name '{parent_obj.agent_name}' from parent InnerSpace.")
            return parent_obj.agent_name

        if isinstance(self.owner, InnerSpace) and hasattr(self.owner, 'agent_name'): # Check for agent_name
            logger.debug(f"[{self.owner.id}] Retrieved agent_name '{self.owner.agent_name}' from owner InnerSpace.")
            return self.owner.agent_name # Return agent_name, not agent_id

        logger.warning(f"[{self.owner.id if self.owner else 'Unknown'}] Could not determine requesting_agent_name through calling_context or InnerSpace hierarchy.")
        return None

    def _get_internal_request_id(self) -> str:
        """Builds an internal request id

        Returns:
            internal request ID
        """
        return f"msg_req_{self.owner.id if self.owner else 'unknown'}_{uuid.uuid4().hex[:12]}"

    async def handle_fetch_history(self,
                             conversation_id: str,
                             before: Optional[int] = None,
                             after: Optional[int] = None,
                             limit: Optional[int] = 100,
                             calling_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: # Added Optional to calling_context
        """
        Tool to request historical messages for a conversation from the adapter.

        Args:
            conversation_id: The external ID of the conversation/channel.
            before: Fetch messages before this timestamp (seconds UTC).
            after: Fetch messages after this timestamp (seconds UTC).
            limit: Maximum number of messages to fetch.
            calling_context: Context from the loop component calling the tool.

        Returns:
            Result of the action dispatch (e.g., confirmation or error).
        """
        calling_context = calling_context or {} # Ensure calling_context is a dict

        context_adapter_id, context_conversation_id = self._get_message_context(use_external_conversation_id=conversation_id) # Renamed vars to avoid conflict

        if not context_adapter_id: # Check if adapter_id was successfully retrieved
             return { "success": False, "error": f"Failed to determine adapter context for conversation {conversation_id}" }

        adapter_id = context_adapter_id # Assign to original variable name

        # Generate internal request ID for tracking like send_message does
        internal_request_id = self._get_internal_request_id()

        logger.info(f"[{self.owner.id if self.owner else 'Unknown'}] Preparing fetch_history action for adapter '{adapter_id}', conv '{conversation_id}'.")

        payload = {
            "internal_request_id": internal_request_id,
            "adapter_id": adapter_id,
            "conversation_id": conversation_id, # This is the external_id
            "before_timestamp_ms": before,
            "after_timestamp_ms": after,
            "limit": limit,
            "requesting_element_id": self.owner.id if self.owner else None,
            "calling_loop_id": calling_context.get('loop_component_id') # From AgentLoop
        }

        return await self._dispatch_action("fetch_history", payload) # "fetch_history" is the ActivityClient action

    async def handle_get_attachment(self,
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

        context_adapter_id, context_conv_id_from_context = self._get_message_context(use_external_conversation_id=conversation_id) # Renamed vars

        if not context_adapter_id: # Check if adapter_id was successfully retrieved
            return { "success": False, "error": f"Failed to determine adapter context for attachment {attachment_id}" }

        adapter_id = context_adapter_id # Assign to original variable name
        # If conversation_id was passed to tool, it's used. Otherwise, context_result['conversation_id'] is used.
        actual_conversation_id = conversation_id if conversation_id else context_conv_id_from_context

        if not adapter_id or not actual_conversation_id:
            return { "success": False, "error": f"Could not determine adapter_id ({adapter_id}) or conversation_id ({actual_conversation_id}) for getting attachment."}

        # Generate internal request ID for tracking like send_message does
        internal_request_id = f"attach_req_{self.owner.id if self.owner else 'unknown'}_{uuid.uuid4().hex[:12]}"

        logger.info(f"[{self.owner.id if self.owner else 'Unknown'}] Preparing get_attachment action for adapter '{adapter_id}', conv '{actual_conversation_id}', attachment '{attachment_id}'.")

        payload = {
            "internal_request_id": internal_request_id,
            "adapter_id": adapter_id,
            "conversation_id": actual_conversation_id,
            "attachment_id": attachment_id,
            "message_external_id": message_external_id,
            "requesting_element_id": self.owner.id if self.owner else None,
            "calling_loop_id": calling_context.get('loop_component_id') # From AgentLoop
        }

        return await self._dispatch_action("get_attachment_content", payload) # "get_attachment_content" is ActivityClient action

    async def _dispatch_action(self, action_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to dispatch an action via the outgoing_action_callback."""
        if not self._outgoing_action_callback:
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Cannot dispatch '{action_type}': Outgoing action callback not configured.")
            return {"success": False, "error": "Outgoing action callback is not configured."}

        action_request = {
            "target_module": "ActivityClient",
            "action_type": action_type, # This is the type ActivityClient expects
            "payload": payload
        }

        # Inject the current trace context to be propagated across the event loop
        carrier = {}
        propagate.inject(carrier)
        action_request["telemetry_context"] = carrier

        try:
            await self._outgoing_action_callback(action_request)
            logger.info(f"[{self.owner.id if self.owner else 'Unknown'}] Dispatched '{action_type}' to ActivityClient with payload: {payload}")
            return {"success": True, "status": f"'{action_type}' request sent to outgoing queue."}
        except Exception as e:
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Error dispatching '{action_type}' action: {e}", exc_info=True)
            return {"success": False, "error": f"Error dispatching '{action_type}' request: {e}"}
