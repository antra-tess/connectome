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
        msg_params: List[ToolParameter] = [
            {"name": "inner_content", "type": "string", "description": "The content of the message to send.", "required": True},
        ]

        get_attachment_params: List[ToolParameter] = [
            {"name": "attachment_id", "type": "string", "description": "The unique ID of the attachment to fetch.", "required": True},
            {"name": "conversation_id", "type": "string", "description": "The conversation ID (optional - will use current context if not provided).", "required": False},
            {"name": "message_external_id", "type": "string", "description": "The external ID of the message containing the attachment (optional).", "required": False}
        ]

        # --- Register msg Tool ---
        @tool_provider.register_tool(
            name="msg",
            description="Sends a message to the conversation (DM or channel).",
            parameters_schema=msg_params
        )
        async def msg_tool(inner_content: str,
                                    # attachments: Optional[List[Dict[str, Any]]] = None,
                                    # reply_to_external_id: Optional[str] = None,
                                    # target_element_id: Optional[str] = None,
                                    calling_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """
            Tool function to send a message.
            Uses local outgoing_action_callback.

            Args:
                inner_content: Message content to send
                attachments: Optional list of attachment objects
                reply_to_external_id: Optional ID of message being replied to
                target_element_id: Optional specific element ID to target. Usually not needed
                                  as the agent loop will automatically route to the correct element.
                calling_context: Context from the calling component
            """
            logger.info(f"[{self.owner.id}] MessageActionHandler.msg_tool called. Text: '{inner_content[:50]}...', Target: {'auto'}")

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
                # final_attachments = []
                # if attachments:
                #     if isinstance(attachments, list):
                #         for att in attachments:
                #             if isinstance(att, dict):
                #                 final_attachments.append(att)
                #             else:
                #                 logger.warning(f"[{self.owner.id}] msg_tool: Skipping non-dict attachment: {att}")
                #     else:
                #         logger.warning(f"[{self.owner.id}] msg_tool: Attachments argument was not a list: {attachments}")

                msg_list_comp.add_pending_message(
                    internal_request_id=internal_request_id,
                    text=inner_content,
                    sender_id=requesting_agent_id or "unknown_agent",
                    sender_name=agent_name or "Unknown Agent",
                    timestamp=time.time(),
                    # attachments=final_attachments,
                    # reply_to_external_id=reply_to_external_id,
                    adapter_id=retrieved_adapter_id, # Should be correct for DM context
                    is_from_current_agent=True,  # FIXED: Mark messages from current agent for deduplication
                    is_internal_origin=True  # NEW: Mark as tool-generated to prevent synthetic agent responses
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
                    "text": inner_content,
                    # "reply_to_external_id": reply_to_external_id,
                    # "attachments": attachments or [],
                    "requesting_element_id": self.owner.id,
                    "requesting_agent_id": requesting_agent_id,
                    # "target_element_id": target_element_id
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

        # --- Register get_attachment Tool ---
        @tool_provider.register_tool(
            name="get_attachment",
            description="Retrieves the content of a specific attachment by its ID. Returns attachment content in base64 format along with metadata. Note: This feature is currently experimental - attachment content may be available directly in message data.",
            parameters_schema=get_attachment_params
        )
        async def get_attachment_tool(attachment_id: str, 
                                    conversation_id: Optional[str] = None,
                                    message_external_id: Optional[str] = None,
                                    calling_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """
            Tool function to get attachment content with enhanced validation.
            
            Security and validation features:
            - Input sanitization and length limits
            - Context validation 
            - Enhanced error handling
            - Rate limiting considerations (attachment_id length check prevents abuse)
            """
            logger.info(f"[{self.owner.id}] get_attachment tool called for attachment_id: {attachment_id}")
            
            # Convert to string if needed and validate
            attachment_id = str(attachment_id) if attachment_id is not None else ""
            
            # Security validation
            if not attachment_id or len(attachment_id.strip()) == 0:
                return {"success": False, "error": "attachment_id cannot be empty"}
            
            # Prevent potential abuse with extremely long IDs
            if len(attachment_id) > 100:
                return {"success": False, "error": "attachment_id too long (max 100 characters)"}
            
            # Basic format validation - attachment IDs should be alphanumeric with some special chars
            import re
            if not re.match(r'^[a-zA-Z0-9_\-\.]+$', attachment_id):
                return {"success": False, "error": "attachment_id contains invalid characters (only alphanumeric, underscore, dash, and dot allowed)"}
            
            try:
                # Use the existing handle_get_attachment method with improved error handling
                result = await self.handle_get_attachment(
                    attachment_id=attachment_id,
                    conversation_id=conversation_id, 
                    message_external_id=message_external_id,
                    calling_context=calling_context
                )
                
                # Add additional validation and context to results
                if result.get("success", False):
                    logger.info(f"[{self.owner.id}] get_attachment succeeded for {attachment_id}")
                    # Add security note to successful results
                    if "status" in result:
                        result["security_note"] = "Attachment fetched successfully. Content should be validated before use."
                else:
                    logger.warning(f"[{self.owner.id}] get_attachment failed for {attachment_id}: {result.get('error', 'Unknown error')}")
                    # Add helpful guidance for failures
                    if "error" in result and "adapter" in result["error"].lower():
                        result["suggestion"] = "Check if the Discord adapter is running and properly configured for attachment processing."
                
                return result
                
            except Exception as e:
                error_msg = f"Unexpected error in get_attachment tool: {str(e)}"
                logger.error(f"[{self.owner.id}] {error_msg}", exc_info=True)
                return {"success": False, "error": error_msg, "suggestion": "This may be due to system configuration issues. Check logs for details."}

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

        # Generate internal request ID for tracking like msg does
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
                              calling_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Tool to request the content of a specific attachment from the adapter.
        Enhanced with robust error handling to prevent irregularities.

        Args:
            attachment_id: The unique ID of the attachment to fetch (required).
            conversation_id: Optional external ID of the conversation (if context is ambiguous).
            message_external_id: Optional external ID of the message the attachment belongs to (for context).
            calling_context: Context from the loop component calling the tool.

        Returns:
            Result of the action dispatch with enhanced error information.
        """
        # Enhanced input validation
        calling_context = calling_context or {}
        
        if not attachment_id or not isinstance(attachment_id, str) or not attachment_id.strip():
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Invalid attachment_id: {attachment_id}")
            return {"success": False, "error": "attachment_id is required and must be a non-empty string."}
        
        # Sanitize attachment_id to prevent injection issues
        attachment_id = attachment_id.strip()
        if len(attachment_id) > 100:  # Reasonable limit
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] attachment_id too long: {len(attachment_id)} characters")
            return {"success": False, "error": "attachment_id is too long (max 100 characters)."}

        # Enhanced context resolution with better error handling
        try:
            context_adapter_id, context_conv_id_from_context = self._get_message_context(use_external_conversation_id=conversation_id)
        except Exception as e:
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Error getting message context: {e}", exc_info=True)
            return {"success": False, "error": f"Failed to determine message context: {str(e)}"}

        if not context_adapter_id:
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] No adapter_id resolved for attachment {attachment_id}")
            return {"success": False, "error": f"Failed to determine adapter context for attachment {attachment_id}. Check element configuration."}

        adapter_id = context_adapter_id
        actual_conversation_id = conversation_id if conversation_id else context_conv_id_from_context

        # Enhanced validation of resolved context
        if not adapter_id or not isinstance(adapter_id, str):
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Invalid adapter_id: {adapter_id}")
            return {"success": False, "error": "Invalid adapter_id resolved from context."}
            
        if not actual_conversation_id or not isinstance(actual_conversation_id, str):
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Invalid conversation_id: {actual_conversation_id}")
            return {"success": False, "error": "Invalid conversation_id resolved from context."}

        # Generate internal request ID with better uniqueness
        try:
            owner_id = self.owner.id if self.owner and hasattr(self.owner, 'id') else 'unknown'
            internal_request_id = f"attach_req_{owner_id}_{uuid.uuid4().hex[:12]}"
        except Exception as e:
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Error generating request ID: {e}")
            internal_request_id = f"attach_req_fallback_{uuid.uuid4().hex[:8]}"

        logger.info(f"[{self.owner.id if self.owner else 'Unknown'}] Preparing get_attachment action for adapter '{adapter_id}', conv '{actual_conversation_id}', attachment '{attachment_id}'.")

        # Enhanced payload with validation
        try:
            payload = {
                "internal_request_id": internal_request_id,
                "adapter_id": str(adapter_id),  # Ensure string
                "conversation_id": str(actual_conversation_id),  # Ensure string
                "attachment_id": str(attachment_id),  # Ensure string
                "message_external_id": str(message_external_id) if message_external_id else None,
                "requesting_element_id": self.owner.id if self.owner else None,
                "calling_loop_id": calling_context.get('loop_component_id') if isinstance(calling_context, dict) else None,
                "timestamp": time.time()  # Add timestamp for tracking
            }
            
            # Validate payload doesn't have None values where strings are expected
            required_string_fields = ["internal_request_id", "adapter_id", "conversation_id", "attachment_id"]
            for field in required_string_fields:
                if not payload.get(field) or not isinstance(payload[field], str):
                    logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Invalid {field} in payload: {payload.get(field)}")
                    return {"success": False, "error": f"Invalid {field} for attachment request."}
                    
        except Exception as e:
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Error creating attachment payload: {e}", exc_info=True)
            return {"success": False, "error": f"Error preparing attachment request: {str(e)}"}

        # Enhanced dispatch with better error context
        try:
            result = await self._dispatch_action("fetch_attachment_content", payload)
            
            # Add success logging
            if result.get("success"):
                logger.info(f"[{self.owner.id if self.owner else 'Unknown'}] Successfully initiated attachment fetch for {attachment_id}")
            else:
                logger.warning(f"[{self.owner.id if self.owner else 'Unknown'}] Attachment fetch dispatch failed for {attachment_id}: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Unexpected error dispatching get_attachment_content: {e}", exc_info=True)
            return {"success": False, "error": f"Unexpected error during attachment request: {str(e)}"}

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
