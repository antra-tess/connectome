import logging
from typing import Dict, Any, Optional, TYPE_CHECKING, List

from .base_component import Component
from elements.component_registry import register_component # For registration

if TYPE_CHECKING:
    from ..base import BaseElement
    from ..inner_space import InnerSpace # Owner is InnerSpace
    from .factory_component import ElementFactoryComponent
    from .tool_provider import ToolProviderComponent, ToolParameter # Import ToolParameter

logger = logging.getLogger(__name__)

DM_ELEMENT_PREFAB_NAME = "direct_message_session"

@register_component
class DirectMessageManagerComponent(Component):
    COMPONENT_TYPE = "direct_message_manager"
    # Does not directly depend on other components for its own init,
    # but will use ToolProvider (for tools) and ElementFactory (for creation) from its owner (InnerSpace)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Map: (adapter_id, external_conversation_id) -> element_id_of_dm_session_element
        self._state.setdefault("active_dm_sessions", {})
        self._state.setdefault("dm_element_mount_ids", {}) # (adapter_id, external_conv_id) -> mount_id

    def initialize(self, **kwargs) -> None:
        super().initialize(**kwargs)
        self._register_dm_tools()
        logger.info(f"DirectMessageManagerComponent initialized for InnerSpace {self.owner.id}")

    def _get_element_factory(self) -> Optional['ElementFactoryComponent']:
        if self.owner and hasattr(self.owner, 'get_element_factory'):
            return self.owner.get_element_factory()
        logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Could not get ElementFactoryComponent from owner.")
        return None

    def _register_dm_tools(self):
        tool_provider: Optional['ToolProviderComponent'] = self.owner.get_tool_provider() if self.owner else None
        if not tool_provider:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] ToolProviderComponent not found on InnerSpace. Cannot register DM tools.")
            return

        initiate_dm_params: List[ToolParameter] = [
            {"name": "adapter_id", "type": "string", "description": "The ID of the adapter (e.g., 'discord', 'slack').", "required": True},
            {"name": "user_external_id", "type": "string", "description": "The external ID of the user on the adapter.", "required": True},
            {"name": "user_display_name", "type": "string", "description": "The display name of the user.", "required": False},
            {"name": "external_conversation_id", "type": "string", "description": "Explicit external conversation ID if known and different from user_external_id.", "required": False}
        ]

        @tool_provider.register_tool(
            name="initiate_direct_message_session",
            description="Ensures a DM session element exists for a given user on an adapter, creating it if necessary. Returns the element_id.",
            parameters_schema=initiate_dm_params
        )
        def initiate_direct_message_session_tool(adapter_id: str, user_external_id: str, user_display_name: Optional[str] = None, external_conversation_id: Optional[str] = None) -> Dict[str, Any]:
            conv_id_to_use = external_conversation_id if external_conversation_id else user_external_id
            
            session_key = (adapter_id, conv_id_to_use)
            existing_element_id = self._state["active_dm_sessions"].get(session_key)

            if existing_element_id and self.owner.get_mounted_element(self._state["dm_element_mount_ids"].get(session_key)):
                logger.info(f"DM session element for {session_key} already exists: {existing_element_id}")
                return {"success": True, "status": "existing", "element_id": existing_element_id, "mount_id": self._state["dm_element_mount_ids"].get(session_key)}

            logger.info(f"DM session for {session_key} not found or element unmounted. Attempting to create/mount...")
            
            element_factory = self._get_element_factory()
            if not element_factory:
                return {"success": False, "error": "ElementFactoryComponent unavailable."}

            element_name = f"DM with {user_display_name or user_external_id} ({adapter_id})"
            element_description = f"Direct Message session with {user_display_name or user_external_id} on {adapter_id} (channel: {conv_id_to_use})"
            mount_id = f"dm_{adapter_id}_{user_external_id.replace(':', '_').replace('@', '_')}" # Make a safe mount_id

            element_config = {
                "name": element_name,
                "description": element_description,
                "dm_adapter_id": adapter_id,
                "dm_external_conversation_id": conv_id_to_use,
                "dm_recipient_info": {
                    "external_user_id": user_external_id,
                    "display_name": user_display_name or user_external_id
                }
            }
            
            # Use a unique element_id for the new DM element
            new_element_id = f"dm_elem_{adapter_id}_{user_external_id.replace(':', '_').replace('@', '_')}_{self._generate_short_uuid()}"


            new_dm_element_result = element_factory.handle_create_element_from_prefab(
                prefab_name=DM_ELEMENT_PREFAB_NAME,
                element_id=new_element_id, # The element's own unique ID
                mount_id_override=mount_id,  # The desired alias for mounting
                element_config=element_config
            )

            if new_dm_element_result and new_dm_element_result.get('success') and new_dm_element_result.get('element'):
                new_dm_element = new_dm_element_result['element']
                final_mount_id = new_dm_element_result.get('mount_id', mount_id) # Use mount_id from result if available
                self._state["active_dm_sessions"][session_key] = new_dm_element.id
                self._state["dm_element_mount_ids"][session_key] = final_mount_id # Store the mount_id used
                logger.info(f"Successfully created and mounted DM session element '{new_dm_element.id}' (mounted as '{final_mount_id}') for {session_key}.")
                return {"success": True, "status": "created", "element_id": new_dm_element.id, "mount_id": final_mount_id}
            else:
                error_msg = new_dm_element_result.get('error', "Failed to create DM session element.") if new_dm_element_result else "Factory returned None or error."
                logger.error(f"Failed to create DM session element for {session_key} using prefab: {error_msg}")
                return {"success": False, "error": f"Failed to create DM session element: {error_msg}"}
        
        logger.info(f"DM tools registered for DirectMessageManagerComponent on {self.owner.id}")
        
    def _generate_short_uuid(self):
        import uuid
        return str(uuid.uuid4())[:8]

    def handle_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handles events, specifically looking for incoming DM messages to ensure
        a corresponding DM session element exists and routes the event to it.
        """
        event_payload = event.get('payload', {})
        connectome_event_type = event_payload.get("event_type")

        # We are interested in "message_received" events that are DMs and routed to InnerSpace
        # (potentially to a generic target_element_id like "dms_[adapter_id]" or no specific target yet)
        if connectome_event_type == "message_received":
            message_data = event_payload.get("payload", {}) # The actual message content is nested further
            is_dm = message_data.get("is_dm", False)
            source_adapter_id = message_data.get("source_adapter_id")
            # external_channel_id on the message_data is the adapter's conversation ID for the DM
            external_conversation_id = message_data.get("external_channel_id") 
            
            # This component only cares about DMs that were routed to its owner (InnerSpace)
            # and might not yet have a specific target DM element.
            # If ExternalEventRouter already targeted a specific DM element, this component might not even see it
            # unless InnerSpace broadly dispatches to all its components.
            # For now, let's assume it gets a chance to see all "message_received" events on InnerSpace.

            if is_dm and source_adapter_id and external_conversation_id:
                logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Received DM event on InnerSpace: adapter='{source_adapter_id}', conv_id='{external_conversation_id}'.")
                
                session_key = (source_adapter_id, external_conversation_id)
                dm_element_id = self._state["active_dm_sessions"].get(session_key)
                dm_element_mount_id = self._state["dm_element_mount_ids"].get(session_key)
                target_dm_element: Optional['BaseElement'] = None
                if dm_element_id and dm_element_mount_id:
                    target_dm_element = self.owner.get_mounted_element(dm_element_mount_id)
                    if not target_dm_element or target_dm_element.mount_id != dm_element_mount_id:
                        logger.warning(f"DM session element for {session_key} was in map (id: {dm_element_id}, mount: {dm_element_mount_id}) but not found/mismatched when fetched. Will try to recreate.")
                        target_dm_element = None # Force recreation
                
                if not target_dm_element:
                    logger.info(f"No active DM session element found for {session_key}. Creating one.")
                    element_factory = self._get_element_factory()
                    if not element_factory:
                        logger.error(f"Cannot create DM element for {session_key}: ElementFactory unavailable.")
                        return False # Cannot handle further

                    # Extract recipient info (sender of this incoming message is the recipient for the agent)
                    sender_external_id = message_data.get("sender_external_id")
                    sender_display_name = message_data.get("sender_display_name", sender_external_id)
                    
                    element_name = f"DM with {sender_display_name or sender_external_id} ({source_adapter_id})"
                    element_description = f"DM session with {sender_display_name or sender_external_id} on {source_adapter_id} (channel: {external_conversation_id})"
                    mount_id = f"dm_{source_adapter_id}_{sender_external_id.replace(':', '_').replace('@', '_') if sender_external_id else self._generate_short_uuid()}"
                    
                    # Use a unique element_id for the new DM element
                    new_element_id_for_dm = f"dm_elem_{source_adapter_id}_{(sender_external_id.replace(':', '_').replace('@', '_') if sender_external_id else self._generate_short_uuid())}_{self._generate_short_uuid()}"


                    element_config = {
                        "name": element_name,
                        "description": element_description,
                        "dm_adapter_id": source_adapter_id,
                        "dm_external_conversation_id": external_conversation_id,
                        "dm_recipient_info": {
                            "external_user_id": sender_external_id,
                            "display_name": sender_display_name
                        }
                    }
                    
                    creation_result = element_factory.handle_create_element_from_prefab(
                        prefab_name=DM_ELEMENT_PREFAB_NAME,
                        element_id=new_element_id_for_dm, # Element's own unique ID
                        mount_id_override=mount_id, # The desired alias for mounting
                        element_config=element_config
                    )

                    if creation_result and creation_result.get('success') and creation_result.get('element'):
                        target_dm_element = creation_result['element']
                        final_mount_id = creation_result.get('mount_id', mount_id) # Use mount_id from result
                        self._state["active_dm_sessions"][session_key] = target_dm_element.id
                        self._state["dm_element_mount_ids"][session_key] = final_mount_id
                        logger.info(f"Created and mounted new DM session element '{target_dm_element.id}' (mounted as '{final_mount_id}') for {session_key}.")
                    else:
                        error_msg = creation_result.get('error') if creation_result else "Creation failed"
                        logger.error(f"Failed to create DM session element for {session_key} on incoming message: {error_msg}")
                        return False # Critical failure
                # Now, forward the original event (with its full payload) to the target DM element
                if target_dm_element and hasattr(target_dm_element, 'receive_event'):
                    logger.debug(f"Forwarding DM event (original id: {event.get('id')}) to element {target_dm_element.id} (mount_id: {self._state['dm_element_mount_ids'].get(session_key)})")
                    
                    # Construct the "full event node" to send to the DM Element.
                    # We use the event ID and timeline_id from the event InnerSpace (owner) recorded,
                    # as this forwarded event is part of the same causal chain.
                    # 'event' here is the full_event_node that DMManagerComponent.handle_event received.
                    # The original event (from ExternalEventRouter) had target_element_id like "dms_adapter_discord"
                    # or was generic to InnerSpace. We now retarget it.
                    # The payload within event_payload should be correct (message_received with its own payload)
                    
                    # This is the inner payload, correctly targeted at the DM Element
                    retargeted_inner_event_payload = {
                        "event_type": connectome_event_type, # e.g., "message_received"
                        "target_element_id": target_dm_element.id, # Crucially, retargeted
                        "source_adapter_id": source_adapter_id, # Keep original source adapter
                        # Potentially copy other relevant fields from event_payload (the one DMManager got) if needed here
                        "payload": message_data # The actual adapter data (text, sender, attachments etc.)
                    }

                    # Construct the "full event node" to send to the DM Element.
                    # We use the event ID and timeline_id from the event InnerSpace (owner) recorded,
                    # as this forwarded event is part of the same causal chain.
                    # 'event' here is the full_event_node that DMManagerComponent.handle_event received.
                    reconstructed_full_event_node_for_dm_element = {
                        "id": event.get("id"), # Event ID from InnerSpace's timeline event
                        "timeline_id": event.get("timeline_id"), # Timeline ID from InnerSpace's timeline event
                        "payload": retargeted_inner_event_payload
                    }
                    
                    # Use the same timeline_context as the one InnerSpace received
                    # as it contains the relevant timeline_id for the DMElement to potentially use if it also logged.
                    target_dm_element.receive_event(reconstructed_full_event_node_for_dm_element, timeline_context)
                    return True # DMManager handled this event by routing it
                else:
                    logger.error(f"DM element {dm_element_id} found for {session_key} but could not forward event.")
            
        return False # Event not fully handled by this component if not a DM or error occurred 

    def get_dm_element_for_user(self, adapter_id: str, external_user_or_conv_id: str) -> Optional['BaseElement']:
        """
        Retrieves the mounted DM session element for a given user/conversation on an adapter.

        Args:
            adapter_id: The ID of the adapter.
            external_user_or_conv_id: The external user ID or conversation ID on the adapter.

        Returns:
            The BaseElement instance if found and mounted, otherwise None.
        """
        session_key = (adapter_id, external_user_or_conv_id)
        dm_element_id = self._state["active_dm_sessions"].get(session_key)
        dm_element_mount_id = self._state["dm_element_mount_ids"].get(session_key)
        if dm_element_id and dm_element_mount_id and self.owner:
            # Check if the DM element is mounted on the owner
            mounted_element = self.owner.get_mounted_element(dm_element_mount_id) # USE MOUNT ID HERE
            if mounted_element and mounted_element.id == dm_element_id:
                return mounted_element
            else:
                logger.warning(f"DM element for {session_key} (id: {dm_element_id}, mount: {dm_element_mount_id}) in map but not found/mismatched on owner.")
                # Clean up stale entry if element is no longer there or ID mismatch
                self._state["active_dm_sessions"].pop(session_key, None)
                self._state["dm_element_mount_ids"].pop(session_key, None)
        return None 