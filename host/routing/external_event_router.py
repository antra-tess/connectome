"""
External Event Router Module

Responsible for routing events originating from external adapters (via ActivityClient)
to the appropriate InnerSpace or SharedSpace.
"""
import logging
import time # For timestamps if not provided by adapter
from typing import Dict, Any, Optional, Callable, List # Added List
import uuid
from opentelemetry import trace
import json

from host.observability import get_tracer

# Assuming elements.elements.space.Space is the base class for SharedSpace
# and elements.elements.inner_space.InnerSpace inherits from it.
from elements.elements.space import Space
from elements.elements.inner_space import InnerSpace # For type hinting
from elements.space_registry import SpaceRegistry # For finding/creating spaces
from elements.utils.element_id_generator import ElementIdGenerator

# --- Agent cycle callback type removed - AgentLoop will self-trigger ---

# NEW: Type hint for AgentConfig (or import it if stable)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from host.config import AgentConfig

# NEW: Connectome internal event types for message send lifecycle
# (Mirrors definitions in MessageListComponent for clarity, though not strictly imported)
CONNECTOME_MESSAGE_SEND_CONFIRMED = "connectome_message_send_confirmed"
CONNECTOME_MESSAGE_SEND_FAILED = "connectome_message_send_failed"

logger = logging.getLogger(__name__)

# Initialize the tracer for this module
tracer = get_tracer(__name__)

class ExternalEventRouter:
    """
    Routes external events from ActivityClient to appropriate Space/Element structures.
    Handles incoming message events and outgoing action confirmations with proper context management.
    """

    def __init__(self, space_registry: 'SpaceRegistry', agent_configs: List['AgentConfig']):
        """
        Initializes the ExternalEventRouter.

        Args:
            space_registry: The SpaceRegistry instance for looking up Spaces by ID.
            agent_configs: A list of all configured agents.
        """
        if not isinstance(space_registry, SpaceRegistry):
            raise TypeError("ExternalEventRouter requires an instance of SpaceRegistry.")
        if not isinstance(agent_configs, list):
            raise TypeError("ExternalEventRouter requires a list of agent_configs.")
             
        self.space_registry = space_registry
        # --- Agent cycle callback removed - AgentLoop will self-trigger ---
        self.agent_configs = agent_configs
        self._activity_client = None  # Will be set via set_activity_client()
        
        # NEW: Store full context for pending outgoing actions (for proper confirmation routing)
        # Key: internal_request_id, Value: Full action context including routing info
        self._pending_action_contexts: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"ExternalEventRouter initialized with {len(agent_configs)} agent configs.")

    async def _create_chat_message_details_payload(self, source_adapter_id: str, adapter_data: Dict[str, Any], is_dm: bool, adapter_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Creates a standardized dictionary containing the core details of a message,
        intended to be the 'payload' part of the event that ChatManagerComponent processes.
        
        Args:
            source_adapter_id: ID of the source adapter
            adapter_data: Raw data from the adapter
            is_dm: Whether this is a direct message
            adapter_type: Type of adapter (e.g., 'zulip', 'discord') for adapter mapping registration
        """
        sender_info = adapter_data.get('sender', {})
        # The conversation_id from adapter_data is the key for external_conversation_id
        external_conv_id = adapter_data.get("conversation_id")

        details_payload = {
            "source_adapter_id": source_adapter_id,
            "external_conversation_id": external_conv_id,
            "is_dm": is_dm,
            "text": adapter_data.get("text", ""), # Standardized to text_content
            "sender_external_id": sender_info.get("user_id"),     # Standardized to sender_id
            "sender_display_name": sender_info.get("display_name"), # Standardized to sender_name
            "timestamp": adapter_data.get("timestamp", time.time()),
            "original_message_id_external": adapter_data.get("message_id"),
            "mentions": adapter_data.get("mentions", []),
            "attachments": adapter_data.get("attachments", []),
            "original_adapter_data": adapter_data, # Keep the raw adapter data
            # NEW: Include adapter_type for adapter mapping registration
            "adapter_type": adapter_type
        }
        return details_payload
    
    def _get_agent_id_by_alias(self, adapter_name: str) -> Optional[str]:
        """
        Retrieves the agent_id associated with a given adapter_name.
        """
        logger.debug(f"Looking for agent with adapter_name: '{adapter_name}'")
        for agent_config in self.agent_configs:
            logger.debug(f"Checking agent '{agent_config.agent_id}' with platform_aliases: {agent_config.platform_aliases}")
            if adapter_name in agent_config.platform_aliases.values():
                logger.info(f"Found agent '{agent_config.agent_id}' for adapter_name '{adapter_name}'")
                return agent_config.agent_id
        logger.warning(f"No agent found for adapter_name '{adapter_name}'. Available aliases: {[list(ac.platform_aliases.values()) for ac in self.agent_configs]}")
        return None

    async def route_external_event(self, event_data_from_activity_client: Dict[str, Any], original_timeline_context: Dict[str, Any]):
        """
        Main entry point for processing an event received from the HostEventLoop,
        originating from ActivityClient.
        """
        source_adapter_id = event_data_from_activity_client.get("source_adapter_id", "unknown_adapter")
        payload = event_data_from_activity_client.get("payload", {})
        event_type_from_adapter = payload.get("event_type_from_adapter", "unknown_event_type")


        with tracer.start_as_current_span("route_external_event", attributes={
            "event.type": event_type_from_adapter,
            "adapter.id": source_adapter_id,
        }) as span:
            # Add critical routing information to the span attributes
            adapter_data = payload.get("adapter_data", {})
            span.set_attribute("routing.is_dm", adapter_data.get("is_direct_message", False))
            span.set_attribute("routing.mentions", adapter_data.get("mentions", []))

            span.add_event("Processing external event", attributes={"event.data": str(event_data_from_activity_client)})
            
            if not isinstance(event_data_from_activity_client, dict):
                logger.error(f"ExternalEventRouter received non-dict event_data: {type(event_data_from_activity_client)}")
                span.set_attribute("routing.error", "Non-dict event data")
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Invalid event data format"))
                return

            adapter_type = event_data_from_activity_client.get("adapter_type")  # NEW: Extract adapter_type
            
            if not source_adapter_id or not isinstance(payload, dict):
                logger.error(f"Event from ActivityClient missing 'source_adapter_id' or valid 'payload' dict: {event_data_from_activity_client}")
                span.set_attribute("routing.error", "Missing source_adapter_id or payload")
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Invalid event structure"))
                return

            adapter_name = adapter_data.get("adapter_name")
            agent_id = self._get_agent_id_by_alias(adapter_name)
            adapter_data["recipient_connectome_agent_id"] = agent_id

            if not event_type_from_adapter or not isinstance(adapter_data, dict):
                logger.error(f"Event payload missing 'event_type_from_adapter' or valid 'adapter_data' dict: {payload}")
                span.set_attribute("routing.error", "Missing event_type_from_adapter or adapter_data")
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Invalid payload structure"))
                return

            logger.debug(f"ExternalEventRouter routing event: Adapter='{source_adapter_id}', Type='{event_type_from_adapter}', AdapterData='{adapter_data}'")
            span.add_event("Routing logic started")
            
            try:
                # --- Routing Logic based on event_type_from_adapter ---
                if event_type_from_adapter == "message_received":
                    await self._handle_direct_message(source_adapter_id, adapter_data, adapter_type)
                elif event_type_from_adapter == "message_updated":
                    await self._handle_message_updated(source_adapter_id, adapter_data)
                elif event_type_from_adapter == "message_deleted":
                    await self._handle_message_deleted(source_adapter_id, adapter_data)
                elif event_type_from_adapter == "reaction_added":
                    await self._handle_reaction_added(source_adapter_id, adapter_data)
                elif event_type_from_adapter == "reaction_removed":
                    await self._handle_reaction_removed(source_adapter_id, adapter_data)
                elif event_type_from_adapter == "conversation_started":
                    await self._handle_conversation_started(source_adapter_id, adapter_data)
                elif event_type_from_adapter == "connectome_history_received":
                    await self._handle_history_received(source_adapter_id, payload.get("conversation_id"), payload.get("messages", []), adapter_data)
                elif event_type_from_adapter == "connectome_attachment_received":
                    await self._handle_attachment_received(source_adapter_id, payload.get("conversation_id"), payload)
                elif event_type_from_adapter == "connectome_attachment_data_received":
                    await self._handle_attachment_data_received(source_adapter_id, adapter_data)
                # --- NEW: Handle Generic Action Confirmations/Failures ---
                elif event_type_from_adapter == "adapter_action_success": # Generic success for any action
                    await self._handle_action_success_ack(source_adapter_id, adapter_data)
                elif event_type_from_adapter == "adapter_action_failure": # Generic failure for any action
                    await self._handle_action_failure_ack(source_adapter_id, adapter_data)
                # --- DEPRECATED: Keep old handlers for backward compatibility ---
                elif event_type_from_adapter == "adapter_send_success_ack": # Legacy - redirect to generic handler
                    logger.warning(f"Received legacy 'adapter_send_success_ack' event. Redirecting to generic handler.")
                    await self._handle_action_success_ack(source_adapter_id, adapter_data)
                elif event_type_from_adapter == "adapter_send_failure_ack": # Legacy - redirect to generic handler
                    logger.warning(f"Received legacy 'adapter_send_failure_ack' event. Redirecting to generic handler.")
                    await self._handle_action_failure_ack(source_adapter_id, adapter_data)
                else:
                    logger.warning(f"ExternalEventRouter: Unhandled event type '{event_type_from_adapter}' from adapter '{source_adapter_id}'. Data: {adapter_data}")
                    span.set_attribute("routing.status", "unhandled_event_type")
                
                span.set_status(trace.Status(trace.StatusCode.OK))
            
            except Exception as e:
                logger.error(f"Error during event routing for type '{event_type_from_adapter}': {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, f"Routing failed: {e}"))

    # TEMPORARY: DEPRECATED: This method is part of the old SharedSpace routing logic.
    def _is_direct_message(self, adapter_data: Dict[str, Any]) -> bool:
        """
        🚨 DEPRECATED: This method is part of the old SharedSpace routing logic.
        
        New implementation uses unified agent-based routing to InnerSpaces only.
        SharedSpace logic is shelved for good and should not be used.
        
        Determines if a message is a Direct Message based on the adapter data.

        Checks for an explicit 'is_direct_message' boolean flag in the adapter_data.
        If the flag is present, its value is returned.
        If the flag is absent, defaults to False (assuming channel message).

        Args:
            adapter_data: The dictionary containing data specific to the adapter event.

        Returns:
            True if the event is identified as a direct message, False otherwise.
        """
        if 'is_direct_message' in adapter_data:
            is_dm = adapter_data.get('is_direct_message')
            if isinstance(is_dm, bool):
                logger.debug(f"Found 'is_direct_message': {is_dm} in adapter_data for conversation: {adapter_data.get('conversation_id')}")
                return is_dm
            else:
                logger.warning(f"Found 'is_direct_message' key in adapter_data, but its value is not a boolean: {is_dm}. Treating as non-DM.")
                return False
        else:
            # Defaulting to False if the key is not present
            logger.debug(f"'is_direct_message' key not found in adapter_data for conversation: {adapter_data.get('conversation_id')}. Assuming channel message.")
            return False

    # TEMPORARY: DEPRECATED: This method is part of the old SharedSpace routing logic.
    async def _find_target_space_for_conversation(self, source_adapter_id: str, conversation_id: str, adapter_data: Dict[str, Any]) -> Optional[Space]:
        """
        🚨 DEPRECATED: This method is part of the old SharedSpace routing logic.
        
        New implementation uses unified agent-based routing to InnerSpaces only.
        Use _find_target_inner_space_for_agent() instead.
        SharedSpace logic is shelved for good and should not be used.
        
        Finds the target InnerSpace or SharedSpace for an event based on 
        adapter ID, conversation ID, and the provided adapter_data.

        Uses the 'is_direct_message' flag in adapter_data if present.
        If True, finds the agent configured to handle DMs for the source_adapter_id 
        and returns their InnerSpace.
        If False or absent, constructs the expected SharedSpace ID and attempts to retrieve it.
        
        Args:
            source_adapter_id: The unique ID of the adapter connection.
            conversation_id: The ID of the conversation from the external platform.
            adapter_data: The raw data dictionary from the adapter, expected to contain 
                          'is_direct_message' (optional).

        Returns:
            The target Space instance (InnerSpace or SharedSpace) or None if not found/ambiguous.
        """
        logger.warning(f"🚨 DEPRECATED: _find_target_space_for_conversation called for {conversation_id}. This method uses old SharedSpace logic which is deprecated.")
        
        is_dm = adapter_data.get('is_direct_message')

        if is_dm is True: 
            # --- Handle Direct Message Routing --- 
            logger.debug(f"Finding InnerSpace for DM: adapter={source_adapter_id}, conv={conversation_id}")
            responsible_agent_id = None
            found_agents = []

            # Find agent(s) responsible for DMs from this adapter
            for agent_config in self.agent_configs:
                if source_adapter_id in agent_config.handles_direct_messages_from_adapter_ids:
                    found_agents.append(agent_config.agent_id)
            
            if len(found_agents) == 1:
                responsible_agent_id = found_agents[0]
                logger.debug(f"Found responsible agent: {responsible_agent_id}")
            elif len(found_agents) == 0:
                logger.error(f"Cannot route DM for conv '{conversation_id}' from adapter '{source_adapter_id}': No agent configured to handle DMs from this adapter.")
                return None
            else: # More than one agent found - configuration error
                logger.error(f"Ambiguous DM target for conv '{conversation_id}' from adapter '{source_adapter_id}': Multiple agents handle DMs: {found_agents}")
                return None
            
            # Retrieve the InnerSpace for the identified agent
            target_inner_space = self.space_registry.get_inner_space_for_agent(responsible_agent_id)
            if not target_inner_space:
                logger.error(f"Found agent '{responsible_agent_id}' responsible for DM conv '{conversation_id}', but their InnerSpace could not be retrieved.")
                return None
            
            return target_inner_space
            
        else: 
            # --- Handle Channel/Shared Space Routing --- 
            logger.warning(f"🚨 DEPRECATED: SharedSpace routing requested for {conversation_id}. SharedSpace logic is shelved and should not be used.")
            if is_dm is False:
                 logger.debug(f"Finding SharedSpace (is_dm=False): adapter={source_adapter_id}, conv={conversation_id}")
            else: # is_dm flag was missing
                 logger.debug(f"Finding SharedSpace (is_dm flag missing): adapter={source_adapter_id}, conv={conversation_id}")
                 
            shared_space_id = f"shared_{source_adapter_id}_{conversation_id}"
            target_shared_space = self.space_registry.get_space(shared_space_id)
            
            if target_shared_space:
                logger.debug(f"Found existing SharedSpace: {shared_space_id}")
                return target_shared_space
            else:
                # Important: Do not create SharedSpaces on the fly for update/delete/reaction events
                # as context might be missing. The space should exist from a prior message_received.
                logger.warning(f"Could not find SharedSpace '{shared_space_id}' for non-DM event. Event may be ignored if space context is required.")
                return None
        
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

    async def _find_target_inner_space_for_agent(self, adapter_data: Dict[str, Any]) -> Optional[InnerSpace]:
        """
        Unified method to find target InnerSpace for any message-related event.
        Uses the same agent-based routing logic as _handle_direct_message.
        
        Args:
            adapter_data: Raw adapter data containing recipient_connectome_agent_id
            
        Returns:
            Target InnerSpace for the agent, or None if not found
        """
        recipient_agent_id = adapter_data.get("recipient_connectome_agent_id")
        if not recipient_agent_id:
            logger.error(f"Event missing 'recipient_connectome_agent_id' in adapter_data. Cannot route to InnerSpace.")
            return None
        
        target_inner_space = self.space_registry.get_inner_space_for_agent(recipient_agent_id)
        if not target_inner_space:
            logger.error(f"Could not find InnerSpace for agent_id '{recipient_agent_id}'.")
            return None
        
        return target_inner_space

    async def _handle_direct_message(self, source_adapter_id: str, adapter_data: Dict[str, Any], adapter_type: Optional[str] = None):
        """
        Handles a direct message, routing it to the recipient agent's InnerSpace
        and notifying HostEventLoop to potentially trigger a cycle.
        (Assumes _is_direct_message already determined this is a DM)
        """
        # Need a way to map conversation_id/adapter_name to the target agent_id
        # This logic needs to be defined based on how adapters report DMs.
        # Let's assume adapter_data contains 'recipient_connectome_agent_id' for now.
        recipient_agent_id = adapter_data.get("recipient_connectome_agent_id") 
        if not recipient_agent_id:
            # Fallback: try to infer from conversation_id? Needs helper function.
            # recipient_agent_id = self._map_dm_conversation_to_agent(source_adapter_id, adapter_data.get('conversation_id'))
            logger.error(f"Message from adapter '{source_adapter_id}' missing 'recipient_connectome_agent_id' in adapter_data. Cannot route DM. Data: {adapter_data}")
            return

        target_inner_space: Optional[InnerSpace] = self.space_registry.get_inner_space_for_agent(recipient_agent_id)
        if not target_inner_space:
            logger.error(f"Could not route DM: InnerSpace for agent_id '{recipient_agent_id}' not found.")
            return

        # Construct the standardized inner payload for ChatManagerComponent
        is_dm_flag = adapter_data.get("is_direct_message", False)
        chat_message_details = await self._create_chat_message_details_payload(source_adapter_id, adapter_data, is_dm=is_dm_flag, adapter_type=adapter_type)
        
        # Construct the event to be recorded on the InnerSpace's timeline
        # The 'payload' of this event will be what ChatManagerComponent processes.
        dm_event_for_timeline = {
            "event_type": "message_received", # Connectome internal type
            "event_id": adapter_data.get("message_id", f"dm_msg_{uuid.uuid4()}"), # Unique ID for the timeline event
            "source_adapter_id": source_adapter_id, # Context for InnerSpace
            # Use the conversation_id from adapter_data, which is the DM's conversation ID
            "external_conversation_id": adapter_data.get("conversation_id"), 
            "is_replayable": True,  # Message content should be replayed for state restoration
            "payload": chat_message_details # Standardized inner payload
        }
        
        timeline_context = await self._construct_timeline_context_for_space(target_inner_space)

        try:
            target_inner_space.receive_event(dm_event_for_timeline, timeline_context)
            logger.info(f"DM from '{source_adapter_id}' for agent '{recipient_agent_id}' routed to InnerSpace with standardized payload.")
            
        except Exception as e:
            logger.error(f"Error in InnerSpace {target_inner_space.id} receiving DM: {e}", exc_info=True)

    # TEMPORARY: DEPRECATED: This method is part of the old SharedSpace routing logic.
    async def _handle_channel_message(self, source_adapter_id: str, adapter_data: Dict[str, Any], adapter_type: Optional[str] = None):
        """
        🚨 DEPRECATED: This method is part of the old SharedSpace routing logic.
        
        New implementation uses unified agent-based routing to InnerSpaces only.
        All message events now route through _handle_direct_message using agent-based selection.
        SharedSpace logic is shelved for good and should not be used.
        
        Handles a public channel message, routing it to a SharedSpace.
        Then, checks ALL configured agents to see if they were mentioned and ensures an uplink.
        (Assumes _is_direct_message already determined this is NOT a DM)
        """
        logger.warning(f"🚨 DEPRECATED: _handle_channel_message called for adapter '{source_adapter_id}'. This method uses old SharedSpace logic which is deprecated.")
        
        external_channel_id = adapter_data.get("conversation_id") # Channel ID is in conversation_id
        if not external_channel_id:
            logger.error(f"Channel message from adapter '{source_adapter_id}' missing 'conversation_id' in adapter_data. Data: {adapter_data}")
            return

        shared_space_identifier = f"shared_{source_adapter_id}_{external_channel_id}"
        shared_space_name = adapter_data.get("channel_name", f"{source_adapter_id}-{external_channel_id}")
        
        target_shared_space: Optional[Space] = self.space_registry.get_or_create_shared_space(
            identifier=shared_space_identifier,
            name=shared_space_name,
            description=f"Shared space for {source_adapter_id} channel {external_channel_id}",
            adapter_id=source_adapter_id,             # EXPLICITLY PASS
            external_conversation_id=external_channel_id, # EXPLICITLY PASS
            metadata={"source_adapter": source_adapter_id, "external_channel_id": external_channel_id} # Keep metadata for other uses if any
        )
        
        if not target_shared_space:
            logger.error(f"Failed to get or create SharedSpace '{shared_space_identifier}'. Message cannot be routed.")
            return
        
        # Construct the standardized inner payload for ChatManagerComponent
        is_dm_flag = False # This is a channel message
        chat_message_details = await self._create_chat_message_details_payload(source_adapter_id, adapter_data, is_dm=is_dm_flag, adapter_type=adapter_type)

        # Construct the event to be recorded on the SharedSpace's timeline
        channel_event_for_timeline = {
            "event_type": "message_received", # Connectome internal type
            "event_id": adapter_data.get("message_id", f"ch_msg_{uuid.uuid4()}"), # Unique ID for the timeline event
            "source_adapter_id": source_adapter_id, # Context for SharedSpace
            "external_conversation_id": external_channel_id, # Context for SharedSpace (same as in chat_message_details)
            "is_replayable": True,  # Message content should be replayed for state restoration
            "payload": chat_message_details # Standardized inner payload
        }
        
        timeline_context = await self._construct_timeline_context_for_space(target_shared_space)

        try:
            # Pass the STANDARDIZED event data to the SharedSpace
            target_shared_space.receive_event(channel_event_for_timeline, timeline_context)
            logger.info(f"Channel message from '{source_adapter_id}' channel '{external_channel_id}' routed to SharedSpace '{shared_space_identifier}' with standardized payload structure.")
            
            # --- Refined Uplink and Cycling Logic ---
            # 1. Ensure uplinks for all agents configured for this adapter (for passive awareness)
            agents_configured_for_adapter = set()
            for agent_cfg in self.agent_configs:
                # TODO: Refine this condition. Should there be a separate config for channel visibility/uplinks?
                # For now, using handles_direct_messages_from_adapter_ids implies general interest in the adapter.
                if source_adapter_id in agent_cfg.handles_direct_messages_from_adapter_ids:
                    agents_configured_for_adapter.add(agent_cfg.agent_id)
            
            logger.info(f"Agents configured for adapter '{source_adapter_id}' (will ensure uplinks): {agents_configured_for_adapter}")
            for agent_id_for_uplink in agents_configured_for_adapter:
                agent_inner_space = self.space_registry.get_inner_space_for_agent(agent_id_for_uplink)
                if agent_inner_space:
                    logger.debug(f"Ensuring uplink to '{target_shared_space.id}' for agent '{agent_id_for_uplink}' in InnerSpace '{agent_inner_space.id}'.")
                    uplink_manager = agent_inner_space.get_uplink_manager()
                    if uplink_manager:
                        uplink_element = uplink_manager.ensure_uplink_to_shared_space(
                            shared_space_id=target_shared_space.id,
                            shared_space_name=target_shared_space.name,
                            shared_space_description=target_shared_space.description
                        )
                        if uplink_element:
                            logger.info(f"Successfully ensured uplink (ID: {uplink_element.id}) for agent '{agent_id_for_uplink}' to SharedSpace '{target_shared_space.id}'.")
                        else:
                            logger.error(f"Failed to ensure uplink for agent '{agent_id_for_uplink}' to SharedSpace '{target_shared_space.id}'.")
                    else:
                        logger.error(f"Agent '{agent_id_for_uplink}'s InnerSpace does not have UplinkManager. Cannot ensure uplink.")
                else:
                    logger.warning(f"Could not find InnerSpace for agent '{agent_id_for_uplink}' to ensure uplink to '{target_shared_space.id}'.")

            # 2. Mark agents for cycle ONLY IF they were mentioned
            mentioned_agent_ids = {mention.get("user_id") for mention in adapter_data.get("mentions", []) if mention.get("user_id")}

            if mentioned_agent_ids:
                logger.info(f"Channel message mentioned agents: {mentioned_agent_ids}. Note: Agent cycle marking removed - AgentLoop will self-trigger.")
            else:
                logger.debug("No specific agent mentions in channel message. Note: Agent cycle marking removed - AgentLoop will self-trigger.")
            
        except Exception as e:
            logger.error(f"Error in SharedSpace {target_shared_space.id} receiving channel message or during agent uplink/mention check: {e}", exc_info=True)

    # --- NEW HANDLERS for Send ACKs ---
    async def _handle_action_success_ack(self, source_adapter_id: str, adapter_data: Dict[str, Any]):
        """
        ENHANCED: Handles action success acknowledgments with full context lookup and proper routing.
        
        Uses stored context from handle_outgoing_action to create properly routed events 
        with all necessary fields for ChatManagerComponent routing.
        """
        internal_request_id = adapter_data.get("internal_request_id")
        
        if not internal_request_id:
            logger.error(f"[_handle_action_success_ack] Missing 'internal_request_id' in adapter_data: {adapter_data}")
            return

        # NEW: Lookup stored context instead of relying on ActivityClient
        stored_context = self._pending_action_contexts.pop(internal_request_id, None)
        
        if not stored_context:
            logger.warning(f"[_handle_action_success_ack] No stored context found for internal_request_id '{internal_request_id}'. Action may have failed during dispatch or timed out.")
            return
            
        # Extract routing info from stored context
        target_element_id = stored_context["target_element_id"]
        action_type = stored_context["action_type"]
        source_adapter_id_from_context = stored_context["source_adapter_id"]
        external_conversation_id = stored_context["external_conversation_id"]
        
        logger.info(f"[_handle_action_success_ack] Processing {action_type} success for req_id '{internal_request_id}' using stored context: adapter={source_adapter_id_from_context}, conv={external_conversation_id}")

        # Get the target element using deep search
        target_element = self.space_registry.find_element_deep(target_element_id)
        if not target_element:
            logger.error(f"[_handle_action_success_ack] Could not find target element '{target_element_id}' for {action_type} confirmation using deep search.")
            return

        # Get the parent space of the target element
        parent_space_of_target_element = target_element.get_parent_object()
        if not parent_space_of_target_element:
            logger.error(f"[_handle_action_success_ack] Could not find parent Space for element '{target_element_id}'.")
            return
        
        # Ensure the parent is actually a Space capable of receiving events
        if not hasattr(parent_space_of_target_element, 'receive_event') or not hasattr(parent_space_of_target_element, 'id'):
            logger.error(f"[_handle_action_success_ack] Parent object for '{target_element_id}' is not a routable Space.")
            return

        # Create properly routed connectome event with all necessary fields
        connectome_event_type = "connectome_action_success"
        
        confirmation_payload = {
            "internal_request_id": internal_request_id,
            "action_type": action_type,
            "confirmed_timestamp": adapter_data.get("confirmed_timestamp", time.time()),
            "adapter_response_data": adapter_data.get("adapter_response_data", {}),
            "raw_adapter_response": adapter_data.get("raw_adapter_response", {})
        }

        # NEW: Create event with proper routing fields
        connectome_event = {
            "event_type": connectome_event_type,
            "source_adapter_id": source_adapter_id_from_context,        # ✅ For ChatManagerComponent routing
            "external_conversation_id": external_conversation_id,       # ✅ For ChatManagerComponent routing
            "target_element_id": target_element_id,
            "is_replayable": True,  # Action confirmations should be replayed for state consistency
            "payload": confirmation_payload
        }
        
        timeline_context = await self._construct_timeline_context_for_space(parent_space_of_target_element) 

        try:
            parent_space_of_target_element.receive_event(connectome_event, timeline_context)
            logger.info(f"{connectome_event_type} event for {action_type} req_id '{internal_request_id}' routed via Space '{parent_space_of_target_element.id}' to Element '{target_element_id}' with full routing context.")
        except Exception as e:
            logger.error(f"Error routing {connectome_event_type} via Space '{parent_space_of_target_element.id}': {e}", exc_info=True)

    async def _handle_action_failure_ack(self, source_adapter_id: str, adapter_data: Dict[str, Any]):
        """
        ENHANCED: Handles action failure acknowledgments with full context lookup and proper routing.
        
        Uses stored context from handle_outgoing_action to create properly routed events 
        with all necessary fields for ChatManagerComponent routing.
        """
        internal_request_id = adapter_data.get("internal_request_id")
        
        if not internal_request_id:
            logger.error(f"[_handle_action_failure_ack] Missing 'internal_request_id' in adapter_data: {adapter_data}")
            return

        # NEW: Lookup stored context instead of relying on ActivityClient
        stored_context = self._pending_action_contexts.pop(internal_request_id, None)
        
        if not stored_context:
            logger.warning(f"[_handle_action_failure_ack] No stored context found for internal_request_id '{internal_request_id}'. Action may have failed during dispatch or timed out.")
            return
            
        # Extract routing info from stored context
        target_element_id = stored_context["target_element_id"]
        action_type = stored_context["action_type"]
        source_adapter_id_from_context = stored_context["source_adapter_id"]
        external_conversation_id = stored_context["external_conversation_id"]
        
        # Extract error information from adapter response
        error_message = adapter_data.get("error_message", "Unknown failure from adapter")
        if not isinstance(error_message, str):
            error_message = str(error_message)
        failed_timestamp = adapter_data.get("failed_timestamp", time.time())

        logger.info(f"[_handle_action_failure_ack] Processing {action_type} failure for req_id '{internal_request_id}' using stored context: adapter={source_adapter_id_from_context}, conv={external_conversation_id}. Error: {error_message}")

        # Get the target element using deep search
        target_element = self.space_registry.find_element_deep(target_element_id)
        if not target_element:
            logger.error(f"[_handle_action_failure_ack] Could not find target element '{target_element_id}' for {action_type} failure ack using deep search.")
            return

        # Get the parent space of the target element
        parent_space_of_target_element = target_element.get_parent_object()
        if not parent_space_of_target_element:
            logger.error(f"[_handle_action_failure_ack] Could not find parent Space for element '{target_element_id}'.")
            return

        # Ensure the parent is actually a Space capable of receiving events
        if not hasattr(parent_space_of_target_element, 'receive_event') or not hasattr(parent_space_of_target_element, 'id'):
            logger.error(f"[_handle_action_failure_ack] Parent object for '{target_element_id}' is not a routable Space.")
            return

        # Create properly routed connectome event with all necessary fields
        connectome_event_type = "connectome_action_failure"
        
        failure_payload = {
            "internal_request_id": internal_request_id,
            "action_type": action_type,
            "error_message": error_message,
            "failed_timestamp": failed_timestamp,
            "adapter_response_data": adapter_data.get("adapter_response_data", {}),
            "raw_adapter_response": adapter_data.get("raw_adapter_response", {})
        }

        # NEW: Create event with proper routing fields
        connectome_event = {
            "event_type": connectome_event_type,
            "source_adapter_id": source_adapter_id_from_context,        # ✅ For ChatManagerComponent routing
            "external_conversation_id": external_conversation_id,       # ✅ For ChatManagerComponent routing
            "target_element_id": target_element_id,
            "is_replayable": True,  # Action failures should be replayed for state consistency
            "payload": failure_payload
        }
        
        timeline_context = await self._construct_timeline_context_for_space(parent_space_of_target_element)

        try:
            parent_space_of_target_element.receive_event(connectome_event, timeline_context)
            logger.info(f"{connectome_event_type} event for {action_type} req_id '{internal_request_id}' routed via Space '{parent_space_of_target_element.id}' to Element '{target_element_id}' with full routing context.")
        except Exception as e:
            logger.error(f"Error routing {connectome_event_type} via Space '{parent_space_of_target_element.id}': {e}", exc_info=True)

    async def _handle_message_updated(self, source_adapter_id: str, adapter_data: Dict[str, Any]):
        """
        Handles a message update event, routing it to the recipient agent's InnerSpace.
        Uses unified agent-based routing for consistency with all message events.
        """
        logger.info(f"Handling 'message_updated' event from {source_adapter_id}. Data: {adapter_data}")
        
        # Extract recipient agent ID like _handle_direct_message does
        adapter_name = adapter_data.get("adapter_name")
        agent_id = self._get_agent_id_by_alias(adapter_name)
        adapter_data["recipient_connectome_agent_id"] = agent_id
        
        # Use unified InnerSpace routing
        target_inner_space = await self._find_target_inner_space_for_agent(adapter_data)
        if not target_inner_space:
            logger.error(f"Cannot route message_updated: Target InnerSpace not found.")
            return
        
        conversation_id = adapter_data.get("conversation_id")
        message_id = adapter_data.get("message_id")
        if not conversation_id or not message_id:
            logger.error("message_updated event missing conversation_id or message_id.")
            return
        
        # Determine if this is a DM for proper target element ID generation
        is_dm = adapter_data.get("is_direct_message", False)
        target_element_id = self._generate_target_element_id(source_adapter_id, conversation_id, is_dm, target_inner_space)
        
        # Construct event payload
        event_payload = {
            "source_adapter_id": source_adapter_id,
            "original_message_id_external": message_id,
            "external_conversation_id": conversation_id,
            "new_text": adapter_data.get("new_text"),
            "timestamp": adapter_data.get("timestamp", time.time()),
            "original_adapter_data": adapter_data
        }
        
        connectome_event = {
            "event_type": "connectome_message_updated",
            "target_element_id": target_element_id,
            "is_replayable": True,  # Message edits should be replayed for message state
            "payload": event_payload
        }
        
        timeline_context = await self._construct_timeline_context_for_space(target_inner_space)
        try:
            target_inner_space.receive_event(connectome_event, timeline_context)
            logger.info(f"message_updated event routed to InnerSpace {target_inner_space.id}")
        except Exception as e:
            logger.error(f"Error routing message_updated event: {e}", exc_info=True)

    async def _handle_message_deleted(self, source_adapter_id: str, adapter_data: Dict[str, Any]):
        """
        Handles a message deletion event, routing it to the recipient agent's InnerSpace.
        Uses unified agent-based routing for consistency with all message events.
        """
        logger.info(f"Handling 'message_deleted' event from {source_adapter_id}. Data: {adapter_data}")
        
        # Extract recipient agent ID like _handle_direct_message does
        adapter_name = adapter_data.get("adapter_name")
        agent_id = self._get_agent_id_by_alias(adapter_name)
        adapter_data["recipient_connectome_agent_id"] = agent_id
        
        # Use unified InnerSpace routing
        target_inner_space = await self._find_target_inner_space_for_agent(adapter_data)
        if not target_inner_space:
            logger.error(f"Cannot route message_deleted: Target InnerSpace not found.")
            return
        
        conversation_id = adapter_data.get("conversation_id")
        message_id = adapter_data.get("message_id")
        if not conversation_id or not message_id:
            logger.error("message_deleted event missing conversation_id or message_id.")
            return
        
        # Determine if this is a DM for proper target element ID generation
        is_dm = adapter_data.get("is_direct_message", False)
        target_element_id = self._generate_target_element_id(source_adapter_id, conversation_id, is_dm, target_inner_space)
        
        event_payload = {
            "source_adapter_id": source_adapter_id,
            "original_message_id_external": message_id,
            "external_conversation_id": conversation_id,
            "timestamp": adapter_data.get("timestamp", time.time()),
            "original_adapter_data": adapter_data
        }
        
        connectome_event = {
            "event_type": "connectome_message_deleted",
            "target_element_id": target_element_id,
            "is_replayable": True,  # Message deletions should be replayed for message state
            "payload": event_payload
        }
        
        timeline_context = await self._construct_timeline_context_for_space(target_inner_space)
        
        try:
            target_inner_space.receive_event(connectome_event, timeline_context)
            logger.info(f"message_deleted event routed to InnerSpace {target_inner_space.id}")
        except Exception as e:
            logger.error(f"Error routing message_deleted event: {e}", exc_info=True)

    async def _handle_reaction_added(self, source_adapter_id: str, adapter_data: Dict[str, Any]):
        """
        Handles a reaction added event, routing it to the recipient agent's InnerSpace.
        Uses unified agent-based routing for consistency with all message events.
        """
        logger.info(f"Handling 'reaction_added' event from {source_adapter_id}. Data: {adapter_data}")
        
        # Extract recipient agent ID like _handle_direct_message does
        adapter_name = adapter_data.get("adapter_name")
        agent_id = self._get_agent_id_by_alias(adapter_name)
        adapter_data["recipient_connectome_agent_id"] = agent_id
        
        # Use unified InnerSpace routing
        target_inner_space = await self._find_target_inner_space_for_agent(adapter_data)
        if not target_inner_space:
            logger.error(f"Cannot route reaction_added: Target InnerSpace not found.")
            return
        
        conversation_id = adapter_data.get("conversation_id")
        message_id = adapter_data.get("message_id")
        emoji = adapter_data.get("emoji")
        if not conversation_id or not message_id or not emoji:
            logger.error("reaction_added event missing conversation_id, message_id, or emoji.")
            return
        
        # Determine if this is a DM for proper target element ID generation
        is_dm = adapter_data.get("is_direct_message", False)
        target_element_id = self._generate_target_element_id(source_adapter_id, conversation_id, is_dm, target_inner_space)
        
        event_payload = {
            "source_adapter_id": source_adapter_id,
            "original_message_id_external": message_id,
            "external_conversation_id": conversation_id,
            "emoji": emoji,
            "user_external_id": adapter_data.get('user', {}).get('user_id'),
            "user_display_name": adapter_data.get('user', {}).get('display_name'),
            "timestamp": adapter_data.get("timestamp", time.time()),
            "original_adapter_data": adapter_data
        }
        
        connectome_event = {
            "event_type": "connectome_reaction_added",
            "target_element_id": target_element_id,
            "is_replayable": True,  # Reaction additions should be replayed for message state
            "payload": event_payload
        }
        
        timeline_context = await self._construct_timeline_context_for_space(target_inner_space)
        
        try:
            target_inner_space.receive_event(connectome_event, timeline_context)
            logger.info(f"reaction_added event routed to InnerSpace {target_inner_space.id}")
        except Exception as e:
            logger.error(f"Error routing reaction_added event: {e}", exc_info=True)

    async def _handle_reaction_removed(self, source_adapter_id: str, adapter_data: Dict[str, Any]):
        """
        Handles a reaction removed event, routing it to the recipient agent's InnerSpace.
        Uses unified agent-based routing for consistency with all message events.
        """
        logger.info(f"Handling 'reaction_removed' event from {source_adapter_id}. Data: {adapter_data}")
        
        # Extract recipient agent ID like _handle_direct_message does
        adapter_name = adapter_data.get("adapter_name")
        agent_id = self._get_agent_id_by_alias(adapter_name)
        adapter_data["recipient_connectome_agent_id"] = agent_id
        
        # Use unified InnerSpace routing
        target_inner_space = await self._find_target_inner_space_for_agent(adapter_data)
        if not target_inner_space:
            logger.error(f"Cannot route reaction_removed: Target InnerSpace not found.")
            return
        
        conversation_id = adapter_data.get("conversation_id")
        message_id = adapter_data.get("message_id")
        emoji = adapter_data.get("emoji")
        if not conversation_id or not message_id or not emoji:
            logger.error("reaction_removed event missing conversation_id, message_id, or emoji.")
            return
        
        # Determine if this is a DM for proper target element ID generation
        is_dm = adapter_data.get("is_direct_message", False)
        target_element_id = self._generate_target_element_id(source_adapter_id, conversation_id, is_dm, target_inner_space)
        
        event_payload = {
            "source_adapter_id": source_adapter_id,
            "original_message_id_external": message_id,
            "external_conversation_id": conversation_id,
            "emoji": emoji,
            "user_external_id": adapter_data.get('user', {}).get('user_id'),
            "user_display_name": adapter_data.get('user', {}).get('display_name'),
            "timestamp": adapter_data.get("timestamp", time.time()),
            "original_adapter_data": adapter_data
        }
        
        connectome_event = {
            "event_type": "connectome_reaction_removed",
            "target_element_id": target_element_id,
            "is_replayable": True,  # Reaction removals should be replayed for message state
            "payload": event_payload
        }
        
        timeline_context = await self._construct_timeline_context_for_space(target_inner_space)
        
        try:
            target_inner_space.receive_event(connectome_event, timeline_context)
            logger.info(f"reaction_removed event routed to InnerSpace {target_inner_space.id}")
        except Exception as e:
            logger.error(f"Error routing reaction_removed event: {e}", exc_info=True)

    async def _handle_conversation_started(self, source_adapter_id: str, adapter_data: Dict[str, Any]):
        """
        Handles the 'conversation_started' event, typically received when joining
        a new channel or starting a connection, containing message history.

        NEW: Routes to InnerSpaces following the refactored architecture where
        all conversations are handled within InnerSpaces, not SharedSpaces.
        Uses the same agent selection logic as _handle_direct_message.
        """
        logger.info(f"Handling 'conversation_started' event from {source_adapter_id}. Data keys: {adapter_data.keys()}")
        conversation_id = adapter_data.get("conversation_id")
        history = adapter_data.get("history") # List of message dicts
        is_dm = adapter_data.get("is_direct_message", False)
        
        # NEW: Reuse the same agent selection logic as _handle_direct_message
        recipient_agent_id = adapter_data.get("recipient_connectome_agent_id")
        if not recipient_agent_id:
            logger.error(f"conversation_started event from adapter '{source_adapter_id}' missing 'recipient_connectome_agent_id' in adapter_data. Cannot route.")
            return

        if not conversation_id:
            logger.error("conversation_started event missing 'conversation_id'. Cannot process.")
            return
        if not isinstance(history, list):
            logger.error(f"conversation_started event for '{conversation_id}' missing 'history' list or it's not a list. Cannot process history.")
            history = [] # Process without history if it's invalid

        # NEW: Get the target InnerSpace for the agent (same as _handle_direct_message)
        target_inner_space = self.space_registry.get_inner_space_for_agent(recipient_agent_id)
        if not target_inner_space:
            logger.error(f"Could not route conversation_started: InnerSpace for agent_id '{recipient_agent_id}' not found.")
            return

        logger.info(f"Processing conversation_started for agent '{recipient_agent_id}' InnerSpace '{target_inner_space.id}' with {len(history)} history messages...")

        # --- Process History for the Target InnerSpace ---
        timeline_context = await self._construct_timeline_context_for_space(target_inner_space)
        processed_count = 0
        error_count = 0

        for message_dict in history:
            if not isinstance(message_dict, dict):
                logger.warning(f"Skipping invalid history item (not a dict) in '{conversation_id}': {message_dict}")
                error_count += 1
                continue
                
            try:
                # Construct the standard message_received payload from the history item
                sender_info = message_dict.get('sender', {})
                history_message_payload = {
                    "source_adapter_id": source_adapter_id,
                    "timestamp": message_dict.get("timestamp", time.time()),
                    "sender_external_id": sender_info.get("user_id"),
                    "sender_display_name": sender_info.get("display_name", "Unknown Sender"),
                    "text": message_dict.get("text"),
                    "is_dm": is_dm,
                    "mentions": message_dict.get("mentions", []), 
                    "original_message_id_external": message_dict.get("message_id"),
                    "external_conversation_id": conversation_id,
                    "original_adapter_data": message_dict, # Store original history item
                    "attachments": message_dict.get("attachments", []),
                    # NEW: Add recipient context for InnerSpace routing (same as _handle_direct_message)
                    "recipient_connectome_agent_id": recipient_agent_id,
                    "external_channel_id": conversation_id if not is_dm else None,  # Only for channel messages
                }

                connectome_history_event = {
                    "event_type": "historical_message_received", # Use historical event type to prevent activation
                    "event_id": f"history_{conversation_id}_{message_dict.get('message_id', processed_count)}_{recipient_agent_id}",
                    "source_adapter_id": source_adapter_id,
                    "external_conversation_id": conversation_id,
                    "target_element_id": self._generate_target_element_id(source_adapter_id, conversation_id, is_dm, target_inner_space),  # NEW: Use generated ID
                    "is_replayable": True,  # Historical messages should be replayed for state restoration
                    "payload": history_message_payload
                }
                
                # Receive the historical event onto the InnerSpace's timeline
                target_inner_space.receive_event(connectome_history_event, timeline_context)
                processed_count += 1

            except Exception as e:
                logger.error(f"Error processing history message for agent '{recipient_agent_id}' in '{conversation_id}': {e}. Message data: {message_dict}", exc_info=True)
                error_count += 1
        
        logger.info(f"Finished processing conversation_started history for agent '{recipient_agent_id}' in '{conversation_id}'. Processed: {processed_count}, Errors: {error_count}")

    # --- NEW HANDLER for History Response --- 
    async def _handle_history_received(self, source_adapter_id: str, conversation_id: str, messages: List[Dict[str, Any]], adapter_data: Dict[str, Any]):
        """
        Handles the internally routed 'connectome_history_received' event.
        Uses unified agent-based routing for consistency with all message events.
        """
        logger.info(f"Handling 'connectome_history_received' event for conv '{conversation_id}' from {source_adapter_id} with {len(messages)} messages.")

        # Extract recipient agent ID like _handle_direct_message does
        adapter_name = adapter_data.get("adapter_name")
        agent_id = self._get_agent_id_by_alias(adapter_name)
        adapter_data["recipient_connectome_agent_id"] = agent_id

        # Use unified InnerSpace routing
        target_inner_space = await self._find_target_inner_space_for_agent(adapter_data)
        if not target_inner_space:
            logger.error(f"Failed to find target InnerSpace for history received for conv '{conversation_id}'. History cannot be processed.")
            return

        logger.info(f"Found target InnerSpace '{target_inner_space.id}'. Processing {len(messages)} history messages...")

        # Process History for the Target InnerSpace
        timeline_context = await self._construct_timeline_context_for_space(target_inner_space)
        processed_count = 0
        error_count = 0

        # Determine if context is DM based on adapter_data
        is_dm_context = adapter_data.get("is_direct_message", False)

        for message_dict in messages:
            if not isinstance(message_dict, dict):
                logger.warning(f"Skipping invalid history item (not a dict) in '{conversation_id}': {message_dict}")
                error_count += 1
                continue
                
            try:
                # Construct the standard message_received payload from the history item
                sender_info = message_dict.get('sender', {})
                history_message_payload = {
                    "source_adapter_id": source_adapter_id, 
                    "timestamp": message_dict.get("timestamp", time.time()),
                    "sender_external_id": sender_info.get("user_id"),
                    "sender_display_name": sender_info.get("display_name", "Unknown Sender"),
                    "text": message_dict.get("text"),
                    "is_dm": is_dm_context,
                    "mentions": message_dict.get("mentions", []), 
                    "original_message_id_external": message_dict.get("message_id"),
                    "external_conversation_id": conversation_id, 
                    "original_adapter_data": message_dict,
                    "attachments": message_dict.get("attachments", []),
                }

                connectome_history_event = {
                    "source_adapter_id": source_adapter_id,
                    "conversation_id": conversation_id,
                    "event_type": "historical_message_received",
                    "target_element_id": self._generate_target_element_id(source_adapter_id, conversation_id, is_dm_context, target_inner_space),
                    "is_replayable": True,  # Historical messages should be replayed for state restoration
                    "payload": history_message_payload
                }
                
                target_inner_space.receive_event(connectome_history_event, timeline_context)
                processed_count += 1

            except Exception as e:
                logger.error(f"Error processing history message for '{conversation_id}': {e}. Message data: {message_dict}", exc_info=True)
                error_count += 1
        
        logger.info(f"Finished processing history for '{conversation_id}'. Processed: {processed_count}, Errors: {error_count}")

    # --- NEW HANDLER for Attachment Data --- 
    async def _handle_attachment_received(self, source_adapter_id: str, conversation_id: str, attachment_event_payload: Dict[str, Any]):
        """
        Handles the internally routed 'connectome_attachment_received' event.
        Uses unified agent-based routing for consistency with all message events.
        """
        adapter_data = attachment_event_payload.get("adapter_data", {})
        attachment_id = adapter_data.get("attachment_id")
        logger.info(f"Handling 'connectome_attachment_received' for conv '{conversation_id}' from {source_adapter_id}, attachment '{attachment_id}'.")

        # Extract recipient agent ID like _handle_direct_message does
        adapter_name = adapter_data.get("adapter_name")
        agent_id = self._get_agent_id_by_alias(adapter_name)
        adapter_data["recipient_connectome_agent_id"] = agent_id

        # Use unified InnerSpace routing
        target_inner_space = await self._find_target_inner_space_for_agent(adapter_data)
        if not target_inner_space:
            logger.error(f"Failed to find target InnerSpace for attachment received for conv '{conversation_id}'. Attachment event ignored.")
            return

        logger.info(f"Found target InnerSpace '{target_inner_space.id}'. Routing attachment event.")

        # Route the attachment event onto the timeline
        timeline_context = await self._construct_timeline_context_for_space(target_inner_space)
        
        # Determine if this is a DM for proper target element ID generation
        is_dm = adapter_data.get("is_direct_message", False)
        target_element_id = self._generate_target_element_id(source_adapter_id, conversation_id, is_dm, target_inner_space)
        
        connectome_attachment_event = {
            "event_type": "connectome_attachment_received", 
            "source_adapter_id": source_adapter_id,
            "target_element_id": target_element_id,
            "is_replayable": True,  # Attachment events should be replayed for message state restoration
            "payload": attachment_event_payload
        }

        try:
            target_inner_space.receive_event(connectome_attachment_event, timeline_context)
            logger.info(f"Routed connectome_attachment_received event for attachment '{attachment_id}' to InnerSpace '{target_inner_space.id}'.")
        except Exception as e:
            logger.error(f"Error routing connectome_attachment_received event: {e}", exc_info=True)

    # --- NEW HANDLER for Fetched Attachment Data ---
    async def _handle_attachment_data_received(self, source_adapter_id: str, adapter_data_with_content: Dict[str, Any]):
        """
        Handles an event from ActivityClient containing the actual fetched content of an attachment.
        Uses unified agent-based routing for consistency with all message events.

        Args:
            source_adapter_id: The adapter that fetched the data.
            adapter_data_with_content: The payload from ActivityClient. Expected to contain:
                - conversation_id: ID of the original conversation.
                - original_message_id_external: ID of the message this attachment belongs to.
                - attachment_id: ID of the attachment.
                - filename: Name of the file.
                - content_type: MIME type of the content.
                - content: The actual attachment content (e.g., bytes or base64 string).
                - adapter_data: Crucially, this nested dict must contain routing info
                                including adapter_name for agent identification.
        """
        conversation_id = adapter_data_with_content.get("conversation_id")
        original_message_id = adapter_data_with_content.get("original_message_id_external")
        attachment_id = adapter_data_with_content.get("attachment_id")
        
        logger.info(f"Handling 'connectome_attachment_data_received' for conv '{conversation_id}', msg '{original_message_id}', attachment '{attachment_id}' from {source_adapter_id}.")

        if not all([conversation_id, original_message_id, attachment_id]):
            logger.error(f"_handle_attachment_data_received: Missing key identifiers in payload. Data: {adapter_data_with_content}")
            return

        # This nested adapter_data is crucial for routing to the correct agent
        routing_adapter_data = adapter_data_with_content.get("adapter_data")
        if not isinstance(routing_adapter_data, dict):
            logger.error(f"_handle_attachment_data_received: 'adapter_data' for routing is missing or not a dict in payload. Data: {adapter_data_with_content}")
            return

        # Extract recipient agent ID like _handle_direct_message does
        adapter_name = routing_adapter_data.get("adapter_name")
        agent_id = self._get_agent_id_by_alias(adapter_name)
        routing_adapter_data["recipient_connectome_agent_id"] = agent_id

        # Use unified InnerSpace routing
        target_inner_space = await self._find_target_inner_space_for_agent(routing_adapter_data)
        if not target_inner_space:
            logger.error(f"_handle_attachment_data_received: Target InnerSpace not found for conv '{conversation_id}'. Attachment data for '{attachment_id}' cannot be delivered.")
            return

        # Prepare payload for the internal event to be stored in the space
        internal_event_payload = {
            "source_adapter_id": source_adapter_id,
            "external_conversation_id": conversation_id,
            "original_message_id_external": original_message_id,
            "attachment_id": attachment_id,
            "filename": adapter_data_with_content.get("filename"),
            "content_type": adapter_data_with_content.get("content_type"),
            "content": adapter_data_with_content.get("content"),
            "status": "content_available",
            "timestamp": adapter_data_with_content.get("timestamp", time.time())
        }

        # Determine if this is a DM for proper target element ID generation
        is_dm = routing_adapter_data.get("is_direct_message", False)
        target_element_id = self._generate_target_element_id(source_adapter_id, conversation_id, is_dm, target_inner_space)

        connectome_internal_event = {
            "event_type": "attachment_content_available",
            "target_element_id": target_element_id,
            "is_replayable": True,  # Attachment content should be replayed for message state restoration
            "payload": internal_event_payload
        }

        timeline_context = await self._construct_timeline_context_for_space(target_inner_space)
        try:
            target_inner_space.receive_event(connectome_internal_event, timeline_context)
            logger.info(f"Routed 'attachment_content_available' for attachment '{attachment_id}' to InnerSpace '{target_inner_space.id}'.")
        except Exception as e:
            logger.error(f"Error routing 'attachment_content_available' event to InnerSpace '{target_inner_space.id}': {e}", exc_info=True)

    def _generate_target_element_id(self, source_adapter_id: str, conversation_id: str, is_dm: bool, target_space: Optional[Space] = None) -> str:
        """
        Generate a consistent target element ID for event routing using the shared utility.
        
        Args:
            source_adapter_id: The adapter ID (e.g., 'zulip_adapter')
            conversation_id: The conversation/channel ID from the external platform
            is_dm: Whether this is a direct message
            target_space: The target space (for additional context)
            
        Returns:
            Deterministic target element ID that matches actual element IDs
        """
        owner_space_id = target_space.id if target_space else None
        return ElementIdGenerator.generate_target_element_id(
            adapter_id=source_adapter_id,
            conversation_id=conversation_id,
            is_dm=is_dm,
            owner_space_id=owner_space_id
        )

    async def handle_outgoing_action(self, action_request: Dict[str, Any]):
        """
        ENHANCED: Handles outgoing actions with full context storage and message-specific preprocessing.
        
        This method serves as the context keeper that:
        1. Stores full action context (including routing info) for confirmations
        2. Handles message/action-specific logic (validation, context, formatting)
        3. Dispatches clean, generic actions to ActivityClient (thin I/O layer)
        
        ActivityClient becomes a pure I/O layer while this handles all business logic.
        """
        action_type = action_request.get("action_type")
        payload = action_request.get("payload", {})
        target_module = action_request.get("target_module")
        
        logger.info(f"ExternalEventRouter preprocessing outgoing {action_type} action for {target_module}")
        
        # Only preprocess actions targeting ActivityClient for now
        if target_module != "ActivityClient":
            logger.debug(f"Action target '{target_module}' not ActivityClient, skipping preprocessing")
            return
            
        # Extract key fields for tracking and validation
        requesting_element_id = payload.get("requesting_element_id")
        internal_request_id = payload.get("internal_request_id")
        adapter_id = payload.get("adapter_id")
        conversation_id = payload.get("conversation_id")
        
        # Basic validation
        if not adapter_id:
            logger.error(f"Cannot preprocess {action_type}: Missing 'adapter_id' in payload")
            return
        if not requesting_element_id:
            logger.error(f"Cannot preprocess {action_type}: Missing 'requesting_element_id' in payload")
            return
        if not internal_request_id:
            logger.error(f"Cannot preprocess {action_type}: Missing 'internal_request_id' in payload")
            return
            
        # NEW: Store full context for confirmation routing
        self._pending_action_contexts[internal_request_id] = {
            "source_adapter_id": adapter_id,
            "external_conversation_id": conversation_id,
            "action_type": action_type,
            "target_element_id": requesting_element_id,
            "original_action_request": action_request,
            "timestamp": time.time()
        }
        logger.debug(f"Stored action context for req_id {internal_request_id}: adapter={adapter_id}, conv={conversation_id}")
            
        # Create clean, action-specific payload for ActivityClient
        clean_payload = {
            "action_type": action_type,
            "adapter_id": adapter_id,
            "internal_request_id": internal_request_id,
            "requesting_element_id": requesting_element_id,
        }
        
        # Add action-specific fields with proper mapping and validation
        if action_type == "send_message":
            clean_payload.update({
                "conversation_id": conversation_id,
                "text": payload.get("text"),
                "attachments": payload.get("attachments", []),
                "reply_to_external_id": payload.get("reply_to_external_id")
            })
            
        elif action_type == "delete_message":
            clean_payload.update({
                "conversation_id": conversation_id,
                "message_external_id": payload.get("message_external_id")
            })
            
        elif action_type == "edit_message":
            clean_payload.update({
                "conversation_id": conversation_id,
                "message_external_id": payload.get("message_external_id"),
                "new_text": payload.get("new_text")
            })
            
        elif action_type in ["add_reaction", "remove_reaction"]:
            clean_payload.update({
                "conversation_id": conversation_id,
                "message_external_id": payload.get("message_external_id"),
                "emoji": payload.get("emoji")
            })
            
        elif action_type == "fetch_message_history":
            clean_payload.update({
                "conversation_id": conversation_id,
                "before_timestamp_ms": payload.get("before_timestamp_ms"),
                "after_timestamp_ms": payload.get("after_timestamp_ms"),
                "limit": payload.get("limit", 100)
            })
            
        elif action_type == "fetch_attachment_content":
            clean_payload.update({
                "conversation_id": conversation_id,
                "message_external_id": payload.get("message_external_id"),
                "attachment_id": payload.get("attachment_id"),
                "attachment_url": payload.get("attachment_url")
            })
            
        else:
            logger.warning(f"Unknown action_type '{action_type}' in outgoing action preprocessing. Passing through with minimal changes.")
            clean_payload.update(payload)  # Pass through unknown actions
        
        # Create clean action request for ActivityClient (thin I/O layer)
        clean_action_request = {
            "action_type": action_type,
            "target_module": "ActivityClient",
            "payload": clean_payload
        }
        
        logger.debug(f"Dispatching clean {action_type} action to ActivityClient: {clean_action_request}")
        
        # Dispatch to ActivityClient as thin I/O layer
        try:
            if not self._activity_client:
                logger.error("ExternalEventRouter doesn't have ActivityClient reference. Cannot dispatch action.")
                return
                
            await self._activity_client.handle_outgoing_action(clean_action_request)
            logger.info(f"Successfully dispatched {action_type} action to ActivityClient")
            
        except Exception as e:
            logger.error(f"Error dispatching {action_type} action to ActivityClient: {e}", exc_info=True)
            # Clean up stored context since action failed to dispatch
            self._pending_action_contexts.pop(internal_request_id, None)
            
    def set_activity_client(self, activity_client):
        """
        Sets the ActivityClient reference for outgoing action dispatch.
        Called during initialization to resolve circular dependencies.
        """
        self._activity_client = activity_client
        logger.debug("ExternalEventRouter: ActivityClient reference set for outgoing action dispatch")
