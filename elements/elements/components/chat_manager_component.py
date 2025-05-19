import logging
from typing import Dict, Any, Optional, TYPE_CHECKING, List

from .base_component import Component
from elements.component_registry import register_component

if TYPE_CHECKING:
    from ..base import BaseElement
    from ..inner_space import InnerSpace # Owner can be InnerSpace
    from ..space import Space # Or a generic Space (like SharedSpace)
    from .factory_component import ElementFactoryComponent
    from .tool_provider import ToolProviderComponent, ToolParameter

logger = logging.getLogger(__name__)

DM_SESSION_PREFAB_NAME = "direct_message_session" # Renamed for clarity
STANDARD_CHAT_INTERFACE_PREFAB_NAME = "standard_chat_interface"

@register_component
class ChatManagerComponent(Component):
    COMPONENT_TYPE = "chat_manager"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # State for DM sessions when owner is InnerSpace
        self._state.setdefault("dm_sessions_map", {}) # (adapter_id, external_conv_id) -> {"element_id": str, "mount_id": str}
        # No specific state needed for SharedSpace mode, as the chat interface is usually singular and conventional.

    def initialize(self, **kwargs) -> None:
        from ..inner_space import InnerSpace # For type checking owner
        super().initialize(**kwargs)
        self._register_tools() # Modified to be conditional
        logger.info(f"ChatManagerComponent initialized for Space {self.owner.id if self.owner else 'Unknown'}")

    def _get_element_factory(self) -> Optional['ElementFactoryComponent']:
        if self.owner and hasattr(self.owner, 'get_element_factory') and callable(getattr(self.owner, 'get_element_factory')):
            factory = self.owner.get_element_factory()
            if factory:
                return factory
        logger.error(f"[{self.owner.id if self.owner else 'UnknownChatManager'}/{self.COMPONENT_TYPE}] Could not get ElementFactoryComponent from owner.")
        return None

    def _register_tools(self):
        from ..inner_space import InnerSpace # For type checking owner
        # Only register DM-specific tools if the owner is an InnerSpace
        if isinstance(self.owner, InnerSpace):
            tool_provider: Optional['ToolProviderComponent'] = self.owner.get_tool_provider() if self.owner else None
            if not tool_provider:
                logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] ToolProviderComponent not found on InnerSpace owner. Cannot register DM tools.")
                return

            initiate_dm_params: List[ToolParameter] = [
                {"name": "adapter_id", "type": "string", "description": "The ID of the adapter (e.g., 'discord', 'slack').", "required": True},
                {"name": "user_external_id", "type": "string", "description": "The external ID of the user on the adapter.", "required": True},
                {"name": "user_display_name", "type": "string", "description": "The display name of the user.", "required": False},
                {"name": "external_conversation_id", "type": "string", "description": "Explicit external conversation ID if known and different from user_external_id.", "required": False}
            ]

            @tool_provider.register_tool(
                name="initiate_direct_message_session",
                description="Ensures a DM session element exists for a given user on an adapter, creating it if necessary. Returns the element_id and mount_id.",
                parameters_schema=initiate_dm_params
            )
            def initiate_direct_message_session_tool(adapter_id: str, user_external_id: str, user_display_name: Optional[str] = None, external_conversation_id: Optional[str] = None) -> Dict[str, Any]:
                # This tool is only relevant for InnerSpace context for initiating DMs
                conv_id_to_use = external_conversation_id if external_conversation_id else user_external_id
                dm_element, mount_id = self._ensure_dm_chat_element(
                    adapter_id, 
                    conv_id_to_use, # This is the user_id for DMs
                    user_external_id, # The actual user being DM'd
                    user_display_name or user_external_id
                )
                if dm_element and mount_id:
                    return {"success": True, "status": "ensured", "element_id": dm_element.id, "mount_id": mount_id}
                else:
                    return {"success": False, "error": "Failed to ensure DM chat element."}
            
            logger.info(f"DM tools registered for ChatManagerComponent on InnerSpace {self.owner.id}")
        else:
            logger.info(f"ChatManagerComponent on Space {self.owner.id} (not InnerSpace) will not register DM tools.")

    def _generate_short_uuid(self):
        import uuid
        return str(uuid.uuid4())[:8]

    def _ensure_dm_chat_element(self, adapter_id: str, conv_id: str, user_ext_id_for_naming: str, user_display_name_for_naming: str) -> tuple[Optional['BaseElement'], Optional[str]]:
        """Ensures a DM chat element exists for the given user, creating if necessary. Owner must be InnerSpace."""
        if not self.owner or not hasattr(self.owner, 'get_mounted_element'): # Basic check for Space-like owner
            logger.error(f"[{self.COMPONENT_TYPE}] Owner not set or not a Space. Cannot ensure DM element.")
            return None, None

        session_key = (adapter_id, conv_id)
        session_info = self._state["dm_sessions_map"].get(session_key)
        
        if session_info:
            mounted_element = self.owner.get_mounted_element(session_info["mount_id"])
            if mounted_element and mounted_element.id == session_info["element_id"]:
                logger.debug(f"DM chat element for {session_key} already exists and mounted: {session_info['element_id']}")
                return mounted_element, session_info["mount_id"]
            else:
                logger.warning(f"DM session for {session_key} in map but element not found/mismatched. Will recreate.")
                self._state["dm_sessions_map"].pop(session_key, None) # Clean up stale entry

        logger.info(f"DM chat element for {session_key} not found or needs recreation. Attempting creation...")
        element_factory = self._get_element_factory()
        if not element_factory:
            return None, None # Error logged by _get_element_factory

        element_name = f"DM with {user_display_name_for_naming or user_ext_id_for_naming} ({adapter_id})"
        element_description = f"DM session with {user_display_name_for_naming or user_ext_id_for_naming} on {adapter_id} (channel: {conv_id})"
        
        # Sanitize user_ext_id_for_naming for mount_id and element_id parts
        safe_user_id_part = user_ext_id_for_naming.replace(':', '_').replace('@', '_').replace('.', '_') if user_ext_id_for_naming else self._generate_short_uuid()
        
        mount_id = f"dm_{adapter_id}_{safe_user_id_part}"
        new_element_id = f"dm_elem_{adapter_id}_{safe_user_id_part}_{self._generate_short_uuid()}"

        element_config = {
            "name": element_name,
            "description": element_description,
            "dm_adapter_id": adapter_id,
            "dm_external_conversation_id": conv_id, # This is the user's ID from adapter's PoV for DM
            "dm_recipient_info": { # Info about the person the agent is DMing
                "external_user_id": user_ext_id_for_naming, 
                "display_name": user_display_name_for_naming
            }
        }
        
        creation_result = element_factory.handle_create_element_from_prefab(
            prefab_name=DM_SESSION_PREFAB_NAME,
            element_id=new_element_id,
            mount_id_override=mount_id,
            element_config=element_config
        )

        if creation_result and creation_result.get('success') and creation_result.get('element'):
            new_dm_element = creation_result['element']
            actual_mount_id = creation_result.get('mount_id', mount_id)
            self._state["dm_sessions_map"][session_key] = {"element_id": new_dm_element.id, "mount_id": actual_mount_id}
            logger.info(f"Successfully created and mounted DM chat element '{new_dm_element.id}' (mounted as '{actual_mount_id}') for {session_key}.")
            return new_dm_element, actual_mount_id
        else:
            error_msg = creation_result.get('error', "Failed to create DM element.") if creation_result else "Factory error."
            logger.error(f"Failed to create DM chat element for {session_key}: {error_msg}")
            return None, None

    def _ensure_shared_chat_element(self) -> Optional['BaseElement']:
        """Ensures the primary shared chat interface element exists. Owner must be SharedSpace."""
        if not self.owner or not hasattr(self.owner, 'adapter_id') or not hasattr(self.owner, 'external_conversation_id'):
            logger.error(f"[{self.COMPONENT_TYPE}] Owner is not a properly configured SharedSpace. Cannot ensure shared chat element.")
            return None

        # Conventional mount ID for the shared chat interface
        chat_element_mount_id = f"{self.owner.id}_chat_interface"
        
        target_chat_element = self.owner.get_mounted_element(chat_element_mount_id)
        if target_chat_element:
            logger.debug(f"Shared chat interface '{chat_element_mount_id}' already exists for {self.owner.id}.")
            return target_chat_element

        logger.info(f"Shared chat interface '{chat_element_mount_id}' not found for {self.owner.id}. Creating...")
        element_factory = self._get_element_factory()
        if not element_factory:
            return None # Error logged

        element_config = {
            "name": f"Chat Interface for {self.owner.name}",
            "description": f"Handles messages for shared space {self.owner.name}",
            "adapter_id": self.owner.adapter_id,
            "external_conversation_id": self.owner.external_conversation_id
        }
        # Unique element ID for the chat interface itself
        chat_element_id = f"chat_elem_{self.owner.id}_{self._generate_short_uuid()}"

        creation_result = element_factory.handle_create_element_from_prefab(
            prefab_name=STANDARD_CHAT_INTERFACE_PREFAB_NAME,
            element_id=chat_element_id,
            mount_id_override=chat_element_mount_id,
            element_config=element_config
        )

        if creation_result and creation_result.get('success') and creation_result.get('element'):
            created_element = creation_result['element']
            logger.info(f"Successfully created shared chat interface '{created_element.id}' (mounted as '{chat_element_mount_id}') for {self.owner.id}.")
            return created_element
        else:
            error_msg = creation_result.get('error', "Failed to create shared chat interface.") if creation_result else "Factory error."
            logger.error(f"Failed to create shared chat interface for {self.owner.id}: {error_msg}")
            return None

    def handle_event(self, event_node: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handles events from the parent Space. If it's a message_received event,
        ensures the appropriate chat element (DM or shared) exists and forwards the event.
        """
        from ..inner_space import InnerSpace # For type checking owner
        from ..space import Space # For type checking owner

        event_payload_from_space = event_node.get('payload', {})
        connectome_event_type = event_payload_from_space.get("event_type")

        if connectome_event_type == "message_received":
            inner_message_payload = event_payload_from_space.get("payload", {}) # This is the EER payload
            if not isinstance(inner_message_payload, dict):
                logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] 'message_received' event has invalid inner payload.")
                return False

            is_dm = inner_message_payload.get("is_dm", False)
            source_adapter_id = event_payload_from_space.get("source_adapter_id")
            # external_conversation_id is the channel_id or user_id from the adapter perspective
            external_conversation_id = event_payload_from_space.get("external_conversation_id")
            if not source_adapter_id or not external_conversation_id:
                logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] 'message_received' missing source_adapter_id or conversation_id in outer payload. Event: {event_payload_from_space}")
                return False

            target_chat_element: Optional['BaseElement'] = None
            
            if isinstance(self.owner, InnerSpace):
                if is_dm:
                    # external_conversation_id from event_payload_from_space is the DM partner's ID
                    # sender_id from inner_message_payload is also the DM partner's ID for an incoming DM
                    dm_partner_user_id = inner_message_payload.get("sender_id", external_conversation_id)
                    dm_partner_display_name = inner_message_payload.get("sender_name", dm_partner_user_id)
                    
                    target_chat_element, _ = self._ensure_dm_chat_element(
                        source_adapter_id,
                        external_conversation_id, # This is the conversation_id with the DM partner
                        dm_partner_user_id,       # User ID for naming/config
                        dm_partner_display_name   # Display name for naming/config
                    )
                else: # Non-DM on InnerSpace
                    logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Received non-DM message on InnerSpace. ChatManagerComponent ignoring. Event: {event_node}")
                    return False
            elif isinstance(self.owner, Space): # General Space, likely a SharedSpace
                if not is_dm:
                    target_chat_element = self._ensure_shared_chat_element()
                else: # DM on SharedSpace
                    logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Received DM message on SharedSpace. ChatManagerComponent ignoring. Event: {event_node}")
                    return False
            else: # Should not happen if component is on a Space/InnerSpace
                logger.error(f"[{self.COMPONENT_TYPE}] Owner is not InnerSpace or Space. Cannot process chat event.")
                return False

            if target_chat_element and hasattr(target_chat_element, 'receive_event'):
                # Forward the full event_node that ChatManagerComponent received from its parent's timeline
                target_chat_element.receive_event(event_node, timeline_context)
                return True # Event handled by forwarding
            elif target_chat_element:
                logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Target chat element '{target_chat_element.id}' found but has no receive_event method.")
            else: # target_chat_element is None
                logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Could not ensure or find target chat element. Message not forwarded for event: {event_node.get('id')}")
            return False # Indicate event not fully handled if forwarding failed
            
        return False # Event not a message_received, or not handled

    # Keep this method if ChatManager on InnerSpace needs to provide easy access to DM elements for other components.
    # Or, other components can use tools provided by ChatManager.
    def get_dm_element_for_user(self, adapter_id: str, external_user_or_conv_id: str) -> Optional['BaseElement']:
        """
        Retrieves the mounted DM session element for a given user/conversation on an adapter.
        Primarily for use when ChatManagerComponent is on an InnerSpace.
        """
        from ..inner_space import InnerSpace
        if not isinstance(self.owner, InnerSpace):
            logger.warning(f"[{self.COMPONENT_TYPE}] get_dm_element_for_user called when owner is not InnerSpace. This method is for DM management.")
            return None

        session_key = (adapter_id, external_user_or_conv_id)
        session_info = self._state["dm_sessions_map"].get(session_key)
        
        if session_info and self.owner:
            mounted_element = self.owner.get_mounted_element(session_info["mount_id"])
            if mounted_element and mounted_element.id == session_info["element_id"]:
                return mounted_element
            else: # Stale entry
                logger.warning(f"DM element for {session_key} (id: {session_info['element_id']}, mount: {session_info['mount_id']}) in map but not found/mismatched. Cleaning up.")
                self._state["dm_sessions_map"].pop(session_key, None)
        return None 