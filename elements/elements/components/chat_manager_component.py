import logging
from typing import Dict, Any, Optional, TYPE_CHECKING, List
from datetime import datetime

from .base_component import Component
from elements.component_registry import register_component
from elements.utils.element_id_generator import ElementIdGenerator

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

    def _generate_deterministic_element_id(self, adapter_id: str, conv_id: str, is_dm: bool) -> str:
        """
        Generate a deterministic element ID based on conversation parameters.
        
        DEPRECATED: Use ElementIdGenerator.generate_chat_element_id() directly.
        This method is kept for backward compatibility.
        """
        return ElementIdGenerator.generate_chat_element_id(
            adapter_id=adapter_id,
            conversation_id=conv_id,
            is_dm=is_dm,
            owner_space_id=self.owner.id if self.owner else None
        )

    def _generate_short_uuid(self):
        """Legacy method kept for compatibility - prefer _generate_deterministic_element_id"""
        import uuid
        return str(uuid.uuid4())[:8]


    def _ensure_chat_element(self, adapter_id: str, conv_id: str, user_ext_id_for_naming: str, user_display_name_for_naming: str, is_dm: bool) -> tuple[Optional['BaseElement'], Optional[str]]:
        """Ensures a chat element exists for the given user, creating if necessary. Owner must be InnerSpace."""
        if not self.owner or not hasattr(self.owner, 'get_mounted_element'): # Basic check for Space-like owner
            logger.error(f"[{self.COMPONENT_TYPE}] Owner not set or not a Space. Cannot ensure chat element.")
            return None, None

        session_key = (adapter_id, conv_id)
        session_info = self._state["dm_sessions_map"].get(session_key)
        
        if session_info:
            mounted_element = self.owner.get_mounted_element(session_info["mount_id"])
            if mounted_element and mounted_element.id == session_info["element_id"]:
                logger.debug(f"Chat element for {session_key} already exists and mounted: {session_info['element_id']}")
                return mounted_element, session_info["mount_id"]
            else:
                logger.warning(f"Chat session for {session_key} in map but element not found/mismatched. Will recreate.")
                self._state["dm_sessions_map"].pop(session_key, None) # Clean up stale entry

        # NEW: Check if deterministic element already exists (handles replay/restart scenarios)
        deterministic_element_id = self._generate_deterministic_element_id(adapter_id, conv_id, is_dm)
        
        # First try to find by deterministic element ID in mounted elements
        if self.owner:
            for mount_id, element in self.owner.get_mounted_elements().items():
                if element.id == deterministic_element_id:
                    logger.info(f"Found existing chat element with deterministic ID: {deterministic_element_id} (mount: {mount_id})")
                    # Update the session mapping to point to the found element
                    self._state["dm_sessions_map"][session_key] = {"element_id": element.id, "mount_id": mount_id}
                    return element, mount_id

        logger.info(f"Chat element for {session_key} not found or needs recreation. Attempting creation...")
        element_factory = self._get_element_factory()
        if not element_factory:
            return None, None # Error logged by _get_element_factory

        session_description = "DM" if is_dm else "Chat"
        element_name = f"{session_description} session ID {user_display_name_for_naming or user_ext_id_for_naming} ({adapter_id})"
        element_description = f"{session_description} session with {user_display_name_for_naming or user_ext_id_for_naming} on {adapter_id} (channel: {conv_id})"
        
        # NEW: Use deterministic element ID instead of random UUID
        new_element_id = deterministic_element_id
        mount_id = new_element_id  # Use same ID for both (simplified approach)

        element_config = {
            "name": element_name,
            "description": element_description,
            "adapter_id": adapter_id,  # Generic attribute instead of dm_adapter_id
            "external_conversation_id": conv_id,  # Generic attribute instead of dm_external_conversation_id
            # Store recipient info for display/context but not as core attributes for tools
            "recipient_info": { # Info about the person/conversation for display
                "external_user_id": user_ext_id_for_naming, 
                "display_name": user_display_name_for_naming,
                "is_dm": is_dm  # Store DM flag for context
            }
        }
        
        creation_result = element_factory.handle_create_element_from_prefab(
            prefab_name=STANDARD_CHAT_INTERFACE_PREFAB_NAME,  # Use generic prefab
            element_id=new_element_id,
            mount_id_override=mount_id,
            element_config=element_config
        )

        if creation_result and creation_result.get('success') and creation_result.get('element'):
            new_chat_element = creation_result['element']  # Renamed from new_dm_element
            actual_mount_id = creation_result.get('mount_id', mount_id)
            self._state["dm_sessions_map"][session_key] = {"element_id": new_chat_element.id, "mount_id": actual_mount_id}
            
            # NEW: Record component state change for replay
            self._record_state_change("dm_session_created", {
                "session_key": session_key,
                "element_id": new_chat_element.id,
                "mount_id": actual_mount_id,
                "adapter_id": adapter_id,
                "conv_id": conv_id,
                "user_ext_id": user_ext_id_for_naming,
                "user_display_name": user_display_name_for_naming,
                "deterministic_id": deterministic_element_id  # Include for debugging
            })
            
            logger.debug(f"Successfully created and mounted chat element '{new_chat_element.id}' (mounted as '{actual_mount_id}') for {session_key}.")
            return new_chat_element, actual_mount_id
        else:
            error_msg = creation_result.get('error', "Failed to create chat element.") if creation_result else "Factory error."
            logger.error(f"Failed to create chat element for {session_key}: {error_msg}")
            return None, None


    def handle_event(self, event_node: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handles events from the parent Space. If it's a message_received event,
        ensures the appropriate chat element (DM or shared) exists and forwards the event.
        """
        from ..inner_space import InnerSpace # For type checking owner
        from ..space import Space # For type checking owner
        
        event_payload_from_space = event_node.get('payload', {})
        connectome_event_type = event_payload_from_space.get("event_type")

        # NEW: Handle component state restoration during replay
        if connectome_event_type == "component_state_updated":
            is_replay_mode = timeline_context.get('replay_mode', False)
            if is_replay_mode:
                return self._handle_state_restoration(event_payload_from_space)

        # NEW: Handle message events (including replay events)
        if connectome_event_type in ["message_received", "historical_message_received", "agent_message_confirmed"]:
            inner_message_payload = event_payload_from_space.get("payload", {}) # This is the EER payload
            if not isinstance(inner_message_payload, dict):
                logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] '{connectome_event_type}' event has invalid inner payload.")
                return False

            is_dm = inner_message_payload.get("is_dm", False)
            source_adapter_id = event_payload_from_space.get("source_adapter_id")
            # external_conversation_id is the channel_id or user_id from the adapter perspective
            external_conversation_id = event_payload_from_space.get("external_conversation_id")
            if not source_adapter_id or not external_conversation_id:
                logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] '{connectome_event_type}' missing source_adapter_id or conversation_id in outer payload. Event: {event_payload_from_space}")
                return False

            target_chat_element: Optional['BaseElement'] = None
            
            if isinstance(self.owner, InnerSpace):
                # SIMPLIFIED: Unified message routing logic
                # Use external_conversation_id as the primary key for finding/creating chat elements
                # The sender info is used for display names and element configuration
                
                # For display purposes, get sender information
                sender_display_name = inner_message_payload.get("sender_display_name", "Unknown")
                sender_external_id = inner_message_payload.get("sender_external_id", external_conversation_id)
                
                # For DMs, the external_conversation_id is typically the conversation/user ID
                # For channels, it's the channel ID
                # In both cases, we use it as the key for finding the right chat element
                target_chat_element, _ = self._ensure_chat_element(
                    source_adapter_id,
                    external_conversation_id,  # Always use this as the conversation key
                    sender_external_id,        # For naming/display purposes
                    sender_display_name,       # For naming/display purposes
                    is_dm=is_dm,              # DM flag determines behavior, not routing
                )
            else: # Should not happen if component is on a Space/InnerSpace
                logger.error(f"[{self.COMPONENT_TYPE}] Owner is not InnerSpace or Space. Cannot process chat event.")
                return False

            if target_chat_element and hasattr(target_chat_element, 'receive_event'):
                # Forward the full event_node that ChatManagerComponent received from its parent's timeline
                logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Routing {connectome_event_type} to chat element {target_chat_element.id}")
                target_chat_element.receive_event(event_node, timeline_context)
                return True # Event handled by forwarding
            elif target_chat_element:
                logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Target chat element '{target_chat_element.id}' found but has no receive_event method.")
            else: # target_chat_element is None
                logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Could not ensure or find target chat element. Message not forwarded for event: {event_node.get('id')}")
            return False # Indicate event not fully handled if forwarding failed
            
        return False # Event not a message_received/historical_message_received/agent_message_confirmed, or not handled

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
        

    def _record_state_change(self, change_type: str, change_data: Dict[str, Any]) -> None:
        """
        Record a component state change to the owner's timeline for replay purposes.
        
        Args:
            change_type: Type of state change (e.g., "dm_session_created")
            change_data: Data describing the state change
        """
        if not self.owner or not hasattr(self.owner, 'add_event_to_primary_timeline'):
            return
            
        event_payload = {
            "event_type": "component_state_updated",
            "target_element_id": self.owner.id,
            "is_replayable": True,  # Component state changes should be replayed
            "data": {
                "component_id": self.id,
                "component_type": self.COMPONENT_TYPE,
                "change_type": change_type,
                "change_data": change_data,
                "timestamp": f"{datetime.now().isoformat()}"
            }
        }
        
        try:
            self.owner.add_event_to_primary_timeline(event_payload)
            logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Recorded state change: {change_type}")
        except Exception as e:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Error recording state change: {e}")

    def _handle_state_restoration(self, event_payload: Dict[str, Any]) -> bool:
        """
        Handle component state restoration during event replay.
        
        Args:
            event_payload: The component_state_updated event payload
            
        Returns:
            True if state restoration was successful, False otherwise
        """
        try:
            event_data = event_payload.get('data', {})
            component_id = event_data.get('component_id')
            component_type = event_data.get('component_type')
            change_type = event_data.get('change_type')
            change_data = event_data.get('change_data', {})
            
            # Only handle events for this component
            if component_id != self.id or component_type != self.COMPONENT_TYPE:
                logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Ignoring state restoration for different component: {component_id}")
                return True  # Not an error, just not for us
                
            logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] REPLAY: Restoring state - {change_type}")
            
            if change_type == "dm_session_created":
                # Restore DM session mapping
                adapter_id = change_data.get('adapter_id')
                conv_id = change_data.get('conv_id')
                element_id = change_data.get('element_id')
                mount_id = change_data.get('mount_id')
                
                if adapter_id and conv_id and element_id and mount_id:
                    session_key = (adapter_id, conv_id)
                    
                    # Verify the element actually exists before restoring state
                    if self.owner and self.owner.get_mounted_element(mount_id):
                        self._state["dm_sessions_map"][session_key] = {
                            "element_id": element_id, 
                            "mount_id": mount_id
                        }
                        logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] ✓ Restored DM session mapping: {session_key} -> {element_id}")
                        return True
                    else:
                        logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Cannot restore DM session mapping: element {mount_id} not found")
                        return False
                else:
                    logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Incomplete DM session data for restoration: {change_data}")
                    return False
            else:
                logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Unknown state change type for restoration: {change_type}")
                return False
                
        except Exception as e:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Error during state restoration: {e}", exc_info=True)
            return False 