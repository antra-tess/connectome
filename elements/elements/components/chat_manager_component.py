import logging
from typing import Dict, Any, Optional, TYPE_CHECKING, List
from datetime import datetime
import time

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
        # Post-refactor: no tool registration; creation-only
        logger.info(f"ChatManagerComponent initialized for Space {self.owner.id if self.owner else 'Unknown'} (creation-only)")

    def _get_element_factory(self) -> Optional['ElementFactoryComponent']:
        if self.owner and hasattr(self.owner, 'get_element_factory') and callable(getattr(self.owner, 'get_element_factory')):
            factory = self.owner.get_element_factory()
            if factory:
                return factory
        logger.error(f"[{self.owner.id if self.owner else 'UnknownChatManager'}/{self.COMPONENT_TYPE}] Could not get ElementFactoryComponent from owner.")
        return None

    def _register_tools(self):
        # Post-refactor: no tools to register; kept for compatibility
        return

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
            },
            # NEW: Add placeholders for conversation metadata
            "adapter_type": None,  # Will be set by _apply_conversation_metadata_to_element if available
            "server_name": None,
            "conversation_name": None
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

    def _ensure_chat_element_with_metadata(self, adapter_id: str, conv_id: str, user_ext_id_for_naming: str, 
                                         user_display_name_for_naming: str, is_dm: bool, 
                                         conversation_metadata: Optional[Dict[str, Any]] = None) -> tuple[Optional['BaseElement'], Optional[str]]:
        """
        Enhanced version of _ensure_chat_element that includes conversation metadata.
        """
        # Call existing _ensure_chat_element method
        element, mount_id = self._ensure_chat_element(adapter_id, conv_id, user_ext_id_for_naming, 
                                                    user_display_name_for_naming, is_dm)
        
        # If element was created/found and we have metadata, apply it
        if element and conversation_metadata:
            self._apply_conversation_metadata_to_element(element, conversation_metadata)
            
        return element, mount_id
    
    def _apply_conversation_metadata_to_element(self, element: 'BaseElement', metadata: Dict[str, Any]) -> None:
        """
        Apply conversation metadata to a chat element for later access by components.
        """
        # Store metadata as element attributes for component access
        for key, value in metadata.items():
            setattr(element, key, value)
        
        # Update element name to use conversation_name if available
        conversation_name = metadata.get("conversation_name")
        if conversation_name and conversation_name != "Unknown Conversation":
            element.name = conversation_name
            
        logger.debug(f"Applied conversation metadata to element {element.id}: adapter_type={element.adapter_type}, server={element.server_name}")


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

        # Post-refactor: bulk history is broadcast by Space directly to components; ChatManager ignores it
        if connectome_event_type == "bulk_history_fetched":
            return False

        # NEW: Handle conversation started event
        if connectome_event_type == "conversation_started":
            return self._handle_conversation_started_event(event_payload_from_space, timeline_context)

        # Post-refactor: ChatManager does not route message/update/reaction/action events
        if connectome_event_type in [
            "message_received", "historical_message_received", "agent_message_confirmed",
            "connectome_message_updated", "connectome_message_deleted",
            "connectome_reaction_added", "connectome_reaction_removed",
            "attachment_content_available",
            "connectome_action_success", "connectome_action_failure"
        ]:
            return False

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

    def _handle_bulk_history_fetched(self, event_payload: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handles bulk history fetched events for efficient bulk processing and reconciliation.

        Args:
            event_payload: The bulk_history_fetched event payload
            timeline_context: Timeline context for the event

        Returns:
            True if handled successfully, False otherwise
        """
        from ..inner_space import InnerSpace
        try:
            # Extract bulk history data
            history_messages = event_payload.get("payload", {}).get("history_messages", [])
            source_adapter_id = event_payload.get("source_adapter_id")
            external_conversation_id = event_payload.get("external_conversation_id")
            is_dm = event_payload.get("payload", {}).get("is_dm", False)
            total_message_count = event_payload.get("payload", {}).get("total_message_count", len(history_messages))

            logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Processing bulk history: {total_message_count} messages for conversation '{external_conversation_id}'")

            if not source_adapter_id or not external_conversation_id:
                logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Bulk history event missing required fields. Event: {event_payload}")
                return False

            if not isinstance(self.owner, InnerSpace):
                logger.error(f"[{self.COMPONENT_TYPE}] Bulk history processing only supported on InnerSpace owners.")
                return False

            # Find existing chat element (should already exist from conversation_started)
            session_key = (source_adapter_id, external_conversation_id)
            session_info = self._state["dm_sessions_map"].get(session_key)

            if session_info:
                mounted_element = self.owner.get_mounted_element(session_info["mount_id"])
                if mounted_element and mounted_element.id == session_info["element_id"]:
                    target_chat_element = mounted_element
                    logger.debug(f"Found existing chat element {target_chat_element.id} for bulk history on conversation {external_conversation_id}")
                else:
                    logger.warning(f"Chat session for {session_key} in map but element not found/mismatched for bulk history.")
                    # Clean up stale entry
                    self._state["dm_sessions_map"].pop(session_key, None)
                    target_chat_element = None
            else:
                target_chat_element = None

            # FALLBACK: If no existing element found, create one without metadata
            # This handles edge cases where conversation_started might have been missed
            if not target_chat_element:
                logger.warning(f"No existing chat element found for bulk history on conversation {external_conversation_id}. Creating fallback element (conversation_started may have been missed).")
                
                # Use the first message for sender information if available
                sender_external_id = external_conversation_id  # Default fallback
                sender_display_name = "Unknown"

                if history_messages and isinstance(history_messages[0], dict):
                    first_msg_sender = history_messages[0].get('sender', {})
                    if first_msg_sender:
                        sender_external_id = first_msg_sender.get("user_id", external_conversation_id)
                        sender_display_name = first_msg_sender.get("display_name", "Unknown")

                target_chat_element, mount_id = self._ensure_chat_element(
                    source_adapter_id,
                    external_conversation_id,
                    sender_external_id,
                    sender_display_name,
                    is_dm=is_dm
                )

            if not target_chat_element:
                logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Could not ensure chat element for bulk history processing. Conversation: {external_conversation_id}")
                return False
            # Create a bulk history event for the MessageListComponent to handle
            bulk_event_for_message_list = {
                "event_type": "bulk_history_received",  # Different event type for MessageListComponent
                "event_id": f"bulk_msg_list_{external_conversation_id}_{int(time.time()*1000)}",
                "source_adapter_id": source_adapter_id,
                "external_conversation_id": external_conversation_id,
                "target_element_id": target_chat_element.id,
                "is_replayable": True,
                "payload": {
                    "event_type": "bulk_history_received",
                    "source_adapter_id": source_adapter_id,
                    "external_conversation_id": external_conversation_id,
                    "is_dm": is_dm,
                    "history_messages": history_messages,
                    "total_message_count": total_message_count,
                    "timestamp": time.time(),
                    "bulk_processing": True  # Flag to indicate bulk processing mode
                }
            }

            # Route the bulk event to the chat element
            if hasattr(target_chat_element, 'receive_event'):
                target_chat_element.receive_event(bulk_event_for_message_list, timeline_context)
                logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Successfully routed bulk history to MessageListComponent")
                return True
            else:
                logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Chat element '{target_chat_element.id}' does not support receive_event")
                return False

        except Exception as e:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Error processing bulk history: {e}", exc_info=True)
            return False

    def _handle_conversation_started_event(self, event_payload: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handles conversation_started events to ensure chat elements exist with proper metadata.
        
        Args:
            event_payload: The conversation_started event payload
            timeline_context: Timeline context for the event
            
        Returns:
            True if handled successfully, False otherwise
        """
        from ..inner_space import InnerSpace
        
        try:
            inner_payload = event_payload.get("payload", {})
            source_adapter_id = event_payload.get("source_adapter_id")
            external_conversation_id = event_payload.get("external_conversation_id")
            conversation_metadata = inner_payload.get("conversation_metadata", {})
            is_dm = inner_payload.get("is_dm", False)
            
            if not source_adapter_id or not external_conversation_id:
                logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] conversation_started missing required fields")
                return False
                
            if not isinstance(self.owner, InnerSpace):
                logger.error(f"[{self.COMPONENT_TYPE}] conversation_started only supported on InnerSpace owners")
                return False
            
            # Extract display information from metadata
            conversation_name = conversation_metadata.get("conversation_name", "Unknown Conversation")
            sender_display_name = conversation_name  # Use conversation name as display name
            sender_external_id = external_conversation_id  # Fallback for sender ID

            
            # Ensure chat element exists with enhanced metadata
            target_chat_element, mount_id = self._ensure_chat_element_with_metadata(
                source_adapter_id,
                external_conversation_id,
                sender_external_id,
                sender_display_name,
                is_dm=is_dm,
                conversation_metadata=conversation_metadata  # Pass metadata for element config
            )
            
            if target_chat_element:
                logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Successfully ensured chat element for conversation_started: {target_chat_element.id}")
                return True
            else:
                logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Failed to ensure chat element for conversation_started")
                return False
                
        except Exception as e:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Error handling conversation_started: {e}", exc_info=True)
            return False
