"""
External Event Router Module

Responsible for routing events originating from external adapters (via ActivityClient)
to the appropriate InnerSpace or SharedSpace.
"""
import logging
import time # For timestamps if not provided by adapter
from typing import Dict, Any, Optional

# Assuming elements.elements.space.Space is the base class for SharedSpace
# and elements.elements.inner_space.InnerSpace inherits from it.
from elements.elements.space import Space
from elements.elements.inner_space import InnerSpace # For type hinting
from elements.space_registry import SpaceRegistry # For finding/creating spaces

logger = logging.getLogger(__name__)

class ExternalEventRouter:
    """
    Routes normalized events from external adapters (via HostEventLoop)
    to the appropriate InnerSpace or SharedSpace.
    """

    def __init__(self, space_registry: SpaceRegistry):
        """
        Initializes the ExternalEventRouter.

        Args:
            space_registry: An instance of SpaceRegistry to find or create Spaces.
        """
        if not isinstance(space_registry, SpaceRegistry):
            raise TypeError("ExternalEventRouter requires an instance of SpaceRegistry.")
        self.space_registry = space_registry
        logger.info("ExternalEventRouter initialized.")

    async def route_external_event(self, event_data_from_activity_client: Dict[str, Any], original_timeline_context: Dict[str, Any]):
        """
        Main entry point for processing an event received from the HostEventLoop,
        originating from ActivityClient.

        Args:
            event_data_from_activity_client: The event dictionary enqueued by ActivityClient.
                                             Expected to contain 'source_adapter_id' and 'payload'
                                             (which is the original normalized event from the adapter).
            original_timeline_context: The timeline_context passed by HostEventLoop.
                                       Currently an empty dict from ActivityClient.
                                       This router will construct the actual timeline_context
                                       for the target Space.
        """
        if not isinstance(event_data_from_activity_client, dict):
            logger.error(f"ExternalEventRouter received non-dict event_data: {type(event_data_from_activity_client)}")
            return

        source_adapter_id = event_data_from_activity_client.get("source_adapter_id")
        adapter_payload = event_data_from_activity_client.get("payload") # Original normalized event from adapter

        if not source_adapter_id or not isinstance(adapter_payload, dict):
            logger.error(f"Event from ActivityClient missing 'source_adapter_id' or valid 'payload': {event_data_from_activity_client}")
            return

        event_type_from_adapter = adapter_payload.get("event_type") # e.g., "direct_message", "channel_message"
        
        logger.debug(f"ExternalEventRouter routing event: Adapter='{source_adapter_id}', Type='{event_type_from_adapter}', AdapterPayload='{adapter_payload}'")

        if event_type_from_adapter == "direct_message":
            await self._handle_direct_message(source_adapter_id, adapter_payload)
        elif event_type_from_adapter == "channel_message":
            await self._handle_channel_message(source_adapter_id, adapter_payload)
        elif event_type_from_adapter == "user_added_to_channel": # Or similar system event
            await self._handle_system_notification(source_adapter_id, adapter_payload)
        # Add more elif blocks for other event_type_from_adapter as needed
        else:
            logger.warning(f"ExternalEventRouter: Unhandled event type '{event_type_from_adapter}' from adapter '{source_adapter_id}'. Payload: {adapter_payload}")

    async def _construct_timeline_context_for_space(self, target_space: Space) -> Dict[str, Any]:
        """
        Constructs a basic timeline context for appending an event to a space.
        Assumes appending to the primary timeline's latest point.
        Space.receive_event and its TimelineComponent should handle actual parenting.
        """
        primary_timeline_id = None
        if hasattr(target_space, 'get_primary_timeline') and callable(target_space.get_primary_timeline):
            primary_timeline_id = target_space.get_primary_timeline()

        if not primary_timeline_id:
            # Fallback or error if no primary timeline concept or method
            # This depends on how Space/TimelineComponent is designed to handle default appends
            logger.warning(f"Target space {target_space.id} does not have a primary timeline reported. Defaulting to None.")
            # It's better if Space.receive_event can handle a None timeline_id by using its default.
            # For now, let's assume TimelineComponent can handle it or uses a default "main" if ID is None
        
        # For appending new external events, we often don't specify parent_event_id,
        # letting the TimelineComponent append to the head of the specified timeline.
        return {"timeline_id": primary_timeline_id}


    async def _handle_direct_message(self, source_adapter_id: str, dm_payload: Dict[str, Any]):
        """
        Handles a direct message, routing it to the recipient agent's InnerSpace.
        """
        recipient_agent_id = dm_payload.get("recipient_connectome_agent_id")
        if not recipient_agent_id:
            logger.error(f"DM from adapter '{source_adapter_id}' missing 'recipient_connectome_agent_id'. Payload: {dm_payload}")
            return

        target_inner_space: Optional[InnerSpace] = self.space_registry.get_inner_space_for_agent(recipient_agent_id)
        if not target_inner_space:
            logger.error(f"Could not route DM: InnerSpace for agent_id '{recipient_agent_id}' not found.")
            return

        dm_handler_element_id = f"dms_{source_adapter_id}" # Convention for target element in InnerSpace

        connectome_dm_event = {
            # Header for Space.receive_event
            "event_type": "connectome_internal_dm", # Connectome's internal event type for the DAG
            "target_element_id": dm_handler_element_id, # Element within InnerSpace to handle this

            # Actual payload for the DM handler component
            "payload": {
                "source_adapter_id": source_adapter_id,
                "timestamp": dm_payload.get("timestamp", time.time()),
                "sender_external_id": dm_payload.get("sender_external_id"),
                "sender_display_name": dm_payload.get("sender_display_name"),
                "text": dm_payload.get("text"),
                "original_message_id_external": dm_payload.get("message_id_external"),
                # Include the full original payload if downstream components might need other fields
                "original_adapter_payload": dm_payload 
            }
        }
        
        timeline_context = await self._construct_timeline_context_for_space(target_inner_space)

        try:
            target_inner_space.receive_event(connectome_dm_event, timeline_context)
            logger.info(f"DM from '{source_adapter_id}' for agent '{recipient_agent_id}' routed to InnerSpace.")
        except Exception as e:
            logger.error(f"Error in InnerSpace {target_inner_space.id} receiving DM: {e}", exc_info=True)

    async def _handle_channel_message(self, source_adapter_id: str, channel_msg_payload: Dict[str, Any]):
        """
        Handles a public channel message, routing it to a SharedSpace.
        """
        external_channel_id = channel_msg_payload.get("channel_id_external")
        if not external_channel_id:
            logger.error(f"Channel message from adapter '{source_adapter_id}' missing 'channel_id_external'. Payload: {channel_msg_payload}")
            return

        shared_space_identifier = f"shared_{source_adapter_id}_{external_channel_id}"
        shared_space_name = channel_msg_payload.get("channel_name_external", f"{source_adapter_id}-{external_channel_id}")
        
        target_shared_space: Optional[Space] = self.space_registry.get_or_create_shared_space(
            identifier=shared_space_identifier,
            name=shared_space_name,
            description=f"Shared space for {source_adapter_id} channel {external_channel_id}",
            metadata={"source_adapter": source_adapter_id, "external_channel_id": external_channel_id}
        )
        
        if not target_shared_space:
            logger.error(f"Failed to get or create SharedSpace '{shared_space_identifier}'. Message cannot be routed.")
            return
        
        chat_element_id_in_shared_space = f"chat_{external_channel_id}" # Convention for target element in SharedSpace

        connectome_channel_event = {
            # Header for Space.receive_event
            "event_type": "connectome_internal_channel_message",
            "target_element_id": chat_element_id_in_shared_space,

            # Actual payload
            "payload": {
                "source_adapter_id": source_adapter_id,
                "timestamp": channel_msg_payload.get("timestamp", time.time()),
                "sender_external_id": channel_msg_payload.get("sender_external_id"),
                "sender_display_name": channel_msg_payload.get("sender_display_name"),
                "text": channel_msg_payload.get("text"),
                "original_message_id_external": channel_msg_payload.get("message_id_external"),
                "external_channel_id": external_channel_id,
                "original_adapter_payload": channel_msg_payload
            }
        }
        
        timeline_context = await self._construct_timeline_context_for_space(target_shared_space)

        try:
            target_shared_space.receive_event(connectome_channel_event, timeline_context)
            logger.info(f"Channel message from '{source_adapter_id}' channel '{external_channel_id}' routed to SharedSpace '{shared_space_identifier}'.")
        except Exception as e:
            logger.error(f"Error in SharedSpace {target_shared_space.id} receiving channel message: {e}", exc_info=True)

    async def _handle_system_notification(self, source_adapter_id: str, notification_payload: Dict[str, Any]):
        """
        Handles system notifications, routing them to the affected agent's InnerSpace.
        """
        affected_agent_id = notification_payload.get("affected_connectome_agent_id")
        if not affected_agent_id:
            logger.error(f"System notification from '{source_adapter_id}' missing 'affected_connectome_agent_id'. Payload: {notification_payload}")
            return

        target_inner_space: Optional[InnerSpace] = self.space_registry.get_inner_space_for_agent(affected_agent_id)
        if not target_inner_space:
            logger.warning(f"Could not route system notification: InnerSpace for agent_id '{affected_agent_id}' not found.")
            return

        system_notification_handler_element_id = "system_notifications_handler" # Convention for target element

        connectome_system_event = {
            # Header for Space.receive_event
            "event_type": "connectome_internal_system_notification",
            "target_element_id": system_notification_handler_element_id,
            
            # Actual payload
            "payload": {
                "source_adapter_id": source_adapter_id,
                "notification_type_from_adapter": notification_payload.get("system_event_subtype"),
                "timestamp": notification_payload.get("timestamp", time.time()),
                "details": notification_payload.get("details", {}),
                "original_adapter_payload": notification_payload
            }
        }
        
        timeline_context = await self._construct_timeline_context_for_space(target_inner_space)
        
        try:
            target_inner_space.receive_event(connectome_system_event, timeline_context)
            logger.info(f"System notification from '{source_adapter_id}' for agent '{affected_agent_id}' routed to InnerSpace.")
        except Exception as e:
            logger.error(f"Error in InnerSpace {target_inner_space.id} receiving system notification: {e}", exc_info=True)

# Example of how HostEventLoop would call this (conceptual):
# async def some_method_in_host_event_loop(self, event_from_ac: Dict[str,Any], timeline_ctx_from_ac: Dict[str,Any]):
#    if self.external_event_router:
#        await self.external_event_router.route_external_event(event_from_ac, timeline_ctx_from_ac)
