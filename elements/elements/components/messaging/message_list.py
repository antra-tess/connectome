"""
Message List Component
Manages the state of a list of messages within an Element.
Acts as a cache/materialized view based on events from the TimelineComponent's DAG.
"""
import logging
import time
from typing import Dict, Any, Optional, List

from ...base import Component
from elements.component_registry import register_component

logger = logging.getLogger(__name__)

# Define the structure of a message stored in the component's state
# This can be expanded later (e.g., with reactions, read_status, edits)
MessageType = Dict[str, Any]

# Internal event types for message send lifecycle
CONNECTOME_MESSAGE_SEND_CONFIRMED = "connectome_message_send_confirmed"
CONNECTOME_MESSAGE_SEND_FAILED = "connectome_message_send_failed"

@register_component
class MessageListComponent(Component):
    """
    Maintains an ordered list of messages based on events recorded in the timeline.
    """
    COMPONENT_TYPE = "MessageListComponent"

    # Events this component reacts to - Updated to handle the router's output
    HANDLED_EVENT_TYPES = [
        "message_received",                   # Unified handler for DMs/Channel messages
        "historical_message_received",        # NEW: For conversation history without activation
        "bulk_history_received",              # NEW: For bulk history processing from ChatManagerComponent
        "agent_message_confirmed",            # NEW: For confirmed agent outgoing messages (replay)
        "connectome_message_deleted",         # Use Connectome-defined types for delete/edit
        "connectome_message_updated",         # Use Connectome-defined types for delete/edit
        "connectome_reaction_added",          # For handling added reactions
        "connectome_reaction_removed",        # For handling removed reactions
        "attachment_content_available",       # NEW: For when fetched attachment content arrives
        "connectome_action_success",          # NEW: Generic action success (replaces specific events)
        "connectome_action_failure",          # NEW: Generic action failure (replaces specific events)
    ]

    def initialize(self, max_messages: Optional[int] = None, **kwargs) -> None:
        """
        Initializes the component state.

        Args:
            max_messages: Optional limit on the number of messages to store.
        """
        super().initialize(**kwargs)
        self._state.setdefault('_messages', []) # List of MessageType dictionaries
        self._state.setdefault('_message_map', {}) # Optional: Map internal_msg_id -> index in _messages for faster updates/deletes
        # NEW: Track operations for VEIL generation
        self._state.setdefault('_pending_veil_operations', [])
        self._max_messages = max_messages
        logger.debug(f"MessageListComponent initialized for Element {self.owner.id}. Max messages: {max_messages}")

    def handle_event(self, event_node: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Processes relevant timeline events to update the message list.
        Expects `event_node['payload']['event_type']` to contain the event type and `event_node['payload']` to contain the actual event details.
        """
        # FIXED: Get event_type from inside payload, where Space actually puts it
        event_payload = event_node.get('payload', {})  # This is what Space provides
        event_type = event_payload.get('event_type')    # Space puts event_type inside payload

        # Check if this is a replay event to avoid activation during startup
        is_replay_mode = timeline_context.get('replay_mode', False)

        if event_type in self.HANDLED_EVENT_TYPES:
            logger.debug(f"[{self.owner.id}] MessageListComponent handling event: {event_type} (replay: {is_replay_mode})")

            # FIXED: For events from ExternalEventRouter via Space, the actual content is nested in event_payload['payload']
            # The structure is: event_node['payload']['payload'] contains the actual message content
            actual_content_payload = event_payload.get('payload', {})  # This contains the actual message content

            if event_type == "message_received":
                self._handle_new_message(actual_content_payload)
            elif event_type == "historical_message_received":
                # Handle historical messages the same way but don't trigger activation
                self._handle_new_message(actual_content_payload)
            elif event_type == "bulk_history_received":
                # Handle bulk history processing
                self._handle_bulk_history_received(event_payload, timeline_context)
            elif event_type == "connectome_message_deleted":
                self._handle_delete_message(actual_content_payload)
            elif event_type == "connectome_message_updated":
                self._handle_edit_message(actual_content_payload)
            elif event_type == "connectome_reaction_added":
                self._handle_reaction_added(actual_content_payload)
            elif event_type == "connectome_reaction_removed":
                self._handle_reaction_removed(actual_content_payload)
            elif event_type == "attachment_content_available":
                self._handle_attachment_content_available(actual_content_payload)
            elif event_type == "connectome_action_success":
                self._handle_action_success(actual_content_payload)
            elif event_type == "connectome_action_failure":
                self._handle_action_failure(actual_content_payload)
            elif event_type == "agent_message_confirmed":
                self._handle_agent_message_confirmed(actual_content_payload)
            else:
                logger.warning(f"[{self.owner.id}] No specific handler implemented for event type '{event_type}' in MessageListComponent.")
                return False # Event type not handled by this component

            # Emit VEIL delta after handling the event (only during normal operation, not replay)
            veil_producer = self.get_sibling_component("MessageListVeilProducer")
            activation_reason = self._get_activation_reason(event_type, actual_content_payload)

            if veil_producer:
                veil_producer.emit_delta()

                if activation_reason and not is_replay_mode:
                    self._emit_activation_call(activation_reason, event_type, actual_content_payload)
                    veil_producer.emit_delta()

            return True

        return False # Event type not in HANDLED_EVENT_TYPES

    def _get_activation_reason(self, event_type: str, content_payload: Dict[str, Any]) -> str:
        """
        Decides if agent activation is needed after processing an event.
        If needed, emits an "activation_call" event to the timeline.

        Args:
            event_type: The type of event that was processed
            content_payload: The content payload that was processed

        Returns:
            The reason for activation, or None if no activation is needed
        """
        activation_reason = None

        if event_type == "message_received":
            # Check for direct messages first
            is_dm = content_payload.get('is_dm', False)
            if is_dm:
                activation_reason = "direct_message_received"
                logger.debug(f"[{self.owner.id}] Activation check: DM received, triggering agent activation")

            # NEW: Check for mentions if not already activated by DM
            if not activation_reason:
                mentions = content_payload.get('mentions', [])
                if mentions:
                    # Check if any mention is for our agent
                    parent_space = self.owner.get_parent_object() if hasattr(self.owner, 'get_parent_object') else None
                    if parent_space and hasattr(parent_space, 'is_mention_for_agent'):
                        if parent_space.is_mention_for_agent(mentions):
                            activation_reason = "agent_mentioned"
                            logger.debug(f"[{self.owner.id}] Activation check: Agent mentioned in {mentions}, triggering agent activation")
                        else:
                            logger.debug(f"[{self.owner.id}] Mentions detected {mentions} but none are for our agent")
                    else:
                        logger.debug(f"[{self.owner.id}] Mentions detected {mentions} but cannot check if for our agent (no parent space or method)")

        return activation_reason

    def _emit_activation_call(self, reason: str, triggering_event_type: str, triggering_payload: Dict[str, Any]) -> None:
        """
        Enhanced to signal focus change to VeilProducer for historical focus tracking.

        Emits an "activation_call" event to the parent space's timeline.
        This is a non-replayable event that signals AgentLoop to consider running a cycle.

        Args:
            reason: Why activation was triggered (e.g., "direct_message_received", "agent_mentioned")
            triggering_event_type: The event type that caused this activation
            triggering_payload: The payload of the triggering event
        """
        # STEP 2: Continue with existing activation_call logic
        if not self.owner:
            logger.warning(f"[MessageListComponent] Cannot emit activation_call: No owner element")
            return

        parent_space = self.owner.get_parent_object()
        if not parent_space or not hasattr(parent_space, 'receive_event'):
            logger.warning(f"[{self.owner.id}] Cannot emit activation_call: Parent space not found or not event-capable")
            return

        # NEW: Include focus context for targeted rendering
        focus_context = {
            "focus_element_id": self.owner.id,  # The element that should be rendered
            "focus_element_type": self.owner.__class__.__name__,
            "focus_element_name": getattr(self.owner, 'name', 'Unknown'),
            "conversation_context": {
                "adapter_id": getattr(self.owner, 'adapter_id', None),
                "external_conversation_id": getattr(self.owner, 'external_conversation_id', None),
                "is_dm": triggering_payload.get('is_dm', False),
                "conversation_id": triggering_payload.get('external_conversation_id'),
                "recent_sender": triggering_payload.get('sender_display_name'),
                "recent_message_preview": triggering_payload.get('text', '')[:100] if triggering_payload.get('text') else None,
                # NEW: Include mention information for agent_mentioned activations
                "activation_reason": reason,
                "mentions": triggering_payload.get('mentions', []) if reason == "agent_mentioned" else None
            }
        }

        activation_event = {
            "event_type": "activation_call",
            "event_id": f"activation_{self.owner.id}_{int(time.time()*1000)}",
            "source_element_id": self.owner.id,
            "activation_reason": reason,
            "triggering_event_type": triggering_event_type,
            "timestamp": time.time(),
            "is_replayable": True,
            "focus_context": focus_context,  # NEW: Context for focused rendering
            "payload": {
                "reason": reason,
                "source_element_id": self.owner.id,
                "triggering_event_type": triggering_event_type,
                "focus_context": focus_context,  # Also include in payload for easy access
                # Don't include full triggering_payload to keep event lightweight
                "conversation_id": triggering_payload.get('external_conversation_id'),
                "sender_id": triggering_payload.get('sender_external_id'),
                # NEW: Include mention information
                "mentions": triggering_payload.get('mentions', []) if reason == "agent_mentioned" else None
            }
        }

        # Use basic timeline context (let parent space handle specifics)
        timeline_context = {"timeline_id": self.owner.get_parent_object().get_primary_timeline()}  # Default timeline

        try:
            parent_space.receive_event(activation_event, timeline_context)
            logger.info(f"[{self.owner.id}] Emitted focused activation_call event to parent space. Reason: {reason}, Focus: {self.owner.id}")
        except Exception as e:
            logger.error(f"[{self.owner.id}] Error emitting activation_call event: {e}", exc_info=True)

    def _handle_new_message(self, message_content: Dict[str, Any]) -> bool:
        """Adds a new message to the list. message_content is the actual message data (e.g., adapter_data)."""

        # Extract relevant fields from the message_content
        internal_message_id = f"msg_{self.owner.id}_{int(time.time()*1000)}_{len(self._state['_messages'])}"

        processed_attachments = []
        for att_data in message_content.get('attachments', []):
            if isinstance(att_data, dict):
                processed_attachments.append({
                    "attachment_id": att_data.get("attachment_id"),
                    "filename": att_data.get("filename"),
                    "content_type": att_data.get("content_type"),
                    "size": att_data.get("size"),
                    "url": att_data.get("url"), # Keep original URL if present
                    "content": att_data.get("content"), # Store inline content if provided
                    # Add other metadata fields from att_data if necessary
                })
            else:
                logger.warning(f"[{self.owner.id}] Skipping non-dict attachment data: {att_data}")

        # NEW: Determine if this message was sent by the current agent
        is_from_current_agent = self._is_message_from_current_agent(message_content)

        new_message: MessageType = {
            'internal_id': internal_message_id,
            'timestamp': message_content.get('timestamp', time.time()),
            'sender_id': message_content.get('sender_external_id'),
            'sender_name': message_content.get('sender_display_name', 'Unknown Sender'),
            'text': message_content.get('text'),
            'original_external_id': message_content.get('original_message_id_external'),
            'adapter_id': message_content.get('source_adapter_id'), # Assumes this is in message_content
            'is_edited': False,
            'reactions': {},
            'read_by': [],
            'attachments': processed_attachments,
            'status': "received", # Default for incoming messages
            'internal_request_id': None, # Not applicable for incoming
            'error_details': None, # Not applicable for incoming
            'is_from_current_agent': is_from_current_agent,  # NEW: Set agent flag for historical detection
            'is_internal_origin': False  # NEW: External messages are not from internal tool calls
        }

        self._state['_messages'].append(new_message)
        self._state['_message_map'][internal_message_id] = len(self._state['_messages']) - 1

        # Optional: Enforce max_messages limit
        if self._max_messages and len(self._state['_messages']) > self._max_messages:
            oldest_message = self._state['_messages'].pop(0)
            del self._state['_message_map'][oldest_message['internal_id']]
            # Need to update indices in _message_map after pop(0) - this makes pop(0) expensive.
            # Using a deque or different structure might be better if max_messages is small and frequent.
            # For simplicity now, we'll rebuild map (inefficient for large lists!)
            if self._state['_messages']: # Ensure list is not empty before trying to rebuild map from a popped message
                self._rebuild_message_map() # Inefficient!
            logger.debug(f"[{self.owner.id}] Pruned oldest message due to max_messages limit.")

        logger.info(f"[{self.owner.id}] New message added. Total messages: {len(self._state['_messages'])}")

        # Record state change for significant milestone messages (every 10th message)
        if len(self._state['_messages']) % 10 == 0:
            self._record_message_state_change("message_count_milestone", {
                "total_messages": len(self._state['_messages']),
                "latest_message_id": internal_message_id,
                "sender": new_message.get('sender_name'),
                "timestamp": new_message.get('timestamp')
            })

        # TODO: Could this component emit a local event like "message_list_updated"?
        return True

    def _handle_delete_message(self, delete_content: Dict[str, Any]) -> bool:
        """
        Handles message deletion events from external sources.

        Two scenarios:
        1. External deletion (other user deleted message) → Mark as deleted with tombstone
        2. Confirmation of agent pending deletion → Remove pending state, mark as confirmed deleted

        Args:
            delete_content: The actual data for deletion (e.g., from event_payload['payload']).
        """
        original_external_id = delete_content.get('original_message_id_external')

        if not original_external_id:
            logger.warning(f"[{self.owner.id}] Message deletion event lacked necessary identifier. Payload: {delete_content}")
            return False

        # Find the message by external ID
        message_to_update = None
        message_index = -1
        for idx, msg in enumerate(self._state['_messages']):
            if msg.get('original_external_id') == original_external_id:
                message_to_update = msg
                message_index = idx
                break

        if not message_to_update:
            logger.warning(f"[{self.owner.id}] Could not delete message: External ID '{original_external_id}' not found.")
            return False

        # NEW: Record delete operation for VEIL generation BEFORE making changes
        original_text = message_to_update.get('original_text_before_pending_delete', message_to_update.get('text'))
        self._record_veil_operation({
            "operation_type": "delete",
            "veil_id": message_to_update['internal_id'],
            "conversation_context": self._get_conversation_metadata(),
            "sender_info": {
                "sender_id": message_to_update.get('sender_id'),
                "sender_name": message_to_update.get('sender_name')
            },
            "delete_details": {
                "original_text": original_text,
                "delete_timestamp": delete_content.get('timestamp', time.time()),
                "original_preview": self._create_text_preview(original_text or "", 50),
                "deletion_source": "external" if not message_to_update.get('status') == "pending_delete" else "agent"
            }
        })

        current_status = message_to_update.get('status', 'received')

        # if current_status == "pending_delete":
        #     # Scenario 2: This is confirmation of our agent's pending deletion
        #     pending_agent_id = message_to_update.get('pending_delete_by_agent_id')
        #     logger.info(f"[{self.owner.id}] Confirmed deletion of message '{original_external_id}' that was pending delete by agent '{pending_agent_id}'")

        #     # Update to confirmed deleted state (keep as tombstone for context)
        #     message_to_update['text'] = "[🗑️ Message deleted]"
        #     message_to_update['status'] = "deleted"
        #     message_to_update['deleted_timestamp'] = delete_content.get('timestamp', time.time())
        #     message_to_update['confirmed_deleted'] = True

        #     # Clean up pending state
        #     message_to_update.pop('original_text_before_pending_delete', None)
        #     message_to_update.pop('pending_delete_by_agent_id', None)
        #     message_to_update.pop('pending_delete_timestamp', None)

        # else:
            # Scenario 1: External deletion (another user deleted the message)
        logger.debug(f"[{self.owner.id}] External deletion of message '{original_external_id}' (was status: {current_status})")

        self._state['_message_map'].pop(message_to_update['internal_id'])
        self._state['_messages'].pop(message_index)

        # IMPORTANT: Keep message in list as tombstone for conversation context
        # Agents need to see that messages existed but were deleted
        logger.info(f"[{self.owner.id}] Message '{original_external_id}' removed from list")
        return True

    def _handle_edit_message(self, edit_content: Dict[str, Any]) -> bool:
        """
        Handles message edit events from external sources.

        Two scenarios:
        1. External edit (other user edited message) → Apply edit immediately
        2. Confirmation of agent pending edit → Remove pending state, apply confirmed edit

        Args:
            edit_content: The actual data for edit (e.g., from event_payload['payload']).
        """
        original_external_id = edit_content.get('original_message_id_external')
        new_text = edit_content.get('new_text')
        edit_timestamp = edit_content.get('timestamp', time.time())

        if not original_external_id or new_text is None:
             logger.warning(f"[{self.owner.id}] Message edit event lacked necessary identifier or new_text. Payload: {edit_content}")
             return False

        # Find message by external ID
        message_to_edit = None
        for msg in self._state['_messages']:
            if msg.get('original_external_id') == original_external_id:
                message_to_edit = msg
                break

        if not message_to_edit:
            logger.warning(f"[{self.owner.id}] Could not edit message: External ID '{original_external_id}' not found.")
            return False

        # NEW: Record edit operation for VEIL generation BEFORE making changes
        original_text = message_to_edit.get('original_text_before_pending_edit', message_to_edit.get('text'))
        self._record_veil_operation({
            "operation_type": "edit",
            "veil_id": message_to_edit['internal_id'],
            "conversation_context": self._get_conversation_metadata(),
            "sender_info": {
                "sender_id": message_to_edit.get('sender_id'),
                "sender_name": message_to_edit.get('sender_name')
            },
            "edit_details": {
                "original_text": original_text,
                "new_text": new_text,
                "edit_timestamp": edit_timestamp,
                "original_preview": self._create_text_preview(original_text or "", 50),
                "new_preview": self._create_text_preview(new_text, 50)
            }
        })

        current_status = message_to_edit.get('status', 'received')

        if current_status == "pending_edit":
            # Scenario 2: This is confirmation of our agent's pending edit
            pending_agent_id = message_to_edit.get('pending_edit_by_agent_id')
            pending_new_text = message_to_edit.get('pending_new_text')

            # Check if confirmed edit matches what agent requested
            if new_text.strip() == pending_new_text.strip():
                logger.info(f"[{self.owner.id}] Confirmed edit of message '{original_external_id}' that was pending edit by agent '{pending_agent_id}'")

                # Apply confirmed edit (remove pending indicator)
                message_to_edit['text'] = new_text
                message_to_edit['status'] = "received"  # Back to normal status
                message_to_edit['is_edited'] = True
                message_to_edit['last_edited_timestamp'] = edit_timestamp
                message_to_edit['confirmed_edited'] = True

                # Clean up pending state
                message_to_edit.pop('original_text_before_pending_edit', None)
                message_to_edit.pop('pending_edit_by_agent_id', None)
                message_to_edit.pop('pending_edit_timestamp', None)
                message_to_edit.pop('pending_new_text', None)
            else:
                logger.warning(f"[{self.owner.id}] Edit confirmation text mismatch for '{original_external_id}'. Expected: '{pending_new_text}', Got: '{new_text}'. Applying external version.")

                # External edit took precedence - apply it and clear pending
                message_to_edit['text'] = new_text
                message_to_edit['status'] = "received"
                message_to_edit['is_edited'] = True
                message_to_edit['last_edited_timestamp'] = edit_timestamp
                message_to_edit['externally_overridden'] = True

                # Clean up pending state
                message_to_edit.pop('original_text_before_pending_edit', None)
                message_to_edit.pop('pending_edit_by_agent_id', None)
                message_to_edit.pop('pending_edit_timestamp', None)
                message_to_edit.pop('pending_new_text', None)
        else:
            # Scenario 1: External edit (another user edited the message)
            logger.info(f"[{self.owner.id}] External edit of message '{original_external_id}' (was status: {current_status})")

            # Store original text if not already stored
            if 'original_text_before_edit' not in message_to_edit and not message_to_edit.get('is_edited', False):
                message_to_edit['original_text_before_edit'] = message_to_edit.get('text')

            # Apply external edit
            message_to_edit['text'] = new_text
            message_to_edit['is_edited'] = True
            message_to_edit['last_edited_timestamp'] = edit_timestamp
            message_to_edit['externally_edited'] = True

        logger.info(f"[{self.owner.id}] Message '{original_external_id}' edit processed successfully")
        return True

    def _handle_reaction_added(self, reaction_content: Dict[str, Any]) -> bool:
        """
        Handles reaction addition events from external sources.

        Two scenarios:
        1. External reaction (other user added reaction) → Add immediately
        2. Confirmation of agent pending reaction → Remove pending state, add confirmed reaction

        Args:
            reaction_content: The actual data for the reaction (e.g., from event_payload['payload']).
        """
        original_external_id = reaction_content.get('original_message_id_external')
        emoji = reaction_content.get('emoji')
        user_id = reaction_content.get('user_external_id')
        user_name = reaction_content.get('user_display_name', 'Unknown User')

        if not original_external_id or not emoji:
            logger.warning(f"[{self.owner.id}] Reaction event lacked message ID or emoji. Payload: {reaction_content}")
            return False

        message_to_update = None
        for msg in self._state['_messages']:
            if msg.get('original_external_id') == original_external_id:
                message_to_update = msg
                break

        if not message_to_update:
            logger.warning(f"[{self.owner.id}] Could not add reaction: Message with external ID '{original_external_id}' not found.")
            return False

        # NEW: Record reaction addition for VEIL generation BEFORE making changes
        self._record_veil_operation({
            "operation_type": "reaction_added",
            "veil_id": message_to_update['internal_id'],
            "conversation_context": self._get_conversation_metadata(),
            "reaction_details": {
                "emoji": emoji,
                "user_id": user_id,
                "user_name": user_name,
                "timestamp": reaction_content.get('timestamp', time.time()),
                "original_message_id": original_external_id
            }
        })

        if 'reactions' not in message_to_update:
            message_to_update['reactions'] = {}

        if emoji not in message_to_update['reactions']:
            message_to_update['reactions'][emoji] = []

        # Check if this is confirmation of a pending reaction by checking for pending markers
        pending_reactions = message_to_update.get('pending_reactions', {})
        pending_key = None
        for key, pending_info in pending_reactions.items():
            if pending_info.get('emoji') == emoji and user_id and pending_info.get('agent_id') == user_id:
                pending_key = key
                break

        if pending_key:
            # Scenario 2: This is confirmation of our agent's pending reaction
            logger.info(f"[{self.owner.id}] Confirmed reaction '{emoji}' addition to message '{original_external_id}' that was pending by agent '{user_id}'")

            # Remove the pending marker and add the confirmed user ID
            pending_marker = f"pending_{user_id}"
            if pending_marker in message_to_update['reactions'][emoji]:
                message_to_update['reactions'][emoji].remove(pending_marker)

            # Add the confirmed user ID if not already present
            if user_id and user_id not in message_to_update['reactions'][emoji]:
                message_to_update['reactions'][emoji].append(user_id)

            # Clean up pending tracking
            del pending_reactions[pending_key]
            if not pending_reactions:
                message_to_update.pop('pending_reactions', None)

            logger.info(f"[{self.owner.id}] Reaction '{emoji}' by agent '{user_name}' confirmed and finalized on message '{original_external_id}'")
        else:
            # Scenario 1: External reaction (another user added the reaction)
            logger.info(f"[{self.owner.id}] External reaction '{emoji}' added to message '{original_external_id}' by user '{user_name}'")

            # Add user to list if not already present (some platforms might send redundant events)
            if user_id and user_id not in message_to_update['reactions'][emoji]:
                message_to_update['reactions'][emoji].append(user_id)
            elif not user_id:
                # If user_id is not provided by the adapter, add anonymous marker
                message_to_update['reactions'][emoji].append("anonymous_reaction")
                logger.debug(f"[{self.owner.id}] Added anonymous reaction '{emoji}' to message {original_external_id}")

        logger.info(f"[{self.owner.id}] Reaction '{emoji}' by '{user_name if user_id else 'anonymous'}' processed for message '{original_external_id}'")
        return True

    def _handle_reaction_removed(self, reaction_content: Dict[str, Any]) -> bool:
        """
        Handles reaction removal events from external sources.

        Two scenarios:
        1. External reaction removal (other user removed reaction) → Remove immediately
        2. Confirmation of agent pending reaction removal → Finalize removal

        Args:
            reaction_content: The actual data for reaction removal (e.g., from event_payload['payload']).
        """
        original_external_id = reaction_content.get('original_message_id_external')
        emoji = reaction_content.get('emoji')
        user_id = reaction_content.get('user_external_id')
        user_name = reaction_content.get('user_display_name', 'Unknown User')

        if not original_external_id or not emoji:
            logger.warning(f"[{self.owner.id}] Reaction removal event lacked message ID or emoji. Payload: {reaction_content}")
            return False

        message_to_update = None
        for msg in self._state['_messages']:
            if msg.get('original_external_id') == original_external_id:
                message_to_update = msg
                break

        if not message_to_update or 'reactions' not in message_to_update or emoji not in message_to_update['reactions']:
            logger.warning(f"[{self.owner.id}] Could not remove reaction: Message '{original_external_id}' not found or no such reaction '{emoji}'.")
            return False

        # NEW: Record reaction removal for VEIL generation BEFORE making changes
        self._record_veil_operation({
            "operation_type": "reaction_removed",
            "veil_id": message_to_update['internal_id'],
            "conversation_context": self._get_conversation_metadata(),
            "reaction_details": {
                "emoji": emoji,
                "user_id": user_id,
                "user_name": user_name,
                "timestamp": reaction_content.get('timestamp', time.time()),
                "original_message_id": original_external_id
            }
        })

        # Check if this is confirmation of a pending reaction removal
        pending_removals = message_to_update.get('pending_reaction_removals', {})
        pending_removal_key = None
        for key, pending_info in pending_removals.items():
            if pending_info.get('emoji') == emoji and user_id and pending_info.get('agent_id') == user_id:
                pending_removal_key = key
                break

        if pending_removal_key:
            # Scenario 2: This is confirmation of our agent's pending reaction removal
            logger.info(f"[{self.owner.id}] Confirmed reaction '{emoji}' removal from message '{original_external_id}' that was pending by agent '{user_id}'")

            # The reaction should already be removed from the local state by remove_pending_reaction
            # Just clean up the pending removal tracking
            del pending_removals[pending_removal_key]
            if not pending_removals:
                message_to_update.pop('pending_reaction_removals', None)

            logger.info(f"[{self.owner.id}] Reaction '{emoji}' removal by agent '{user_name}' confirmed and finalized for message '{original_external_id}'")
        else:
            # Scenario 1: External reaction removal (another user removed their reaction)
            logger.info(f"[{self.owner.id}] External reaction '{emoji}' removal from message '{original_external_id}' by user '{user_name}'")

            reactions_list = message_to_update['reactions'][emoji]
            if user_id:
                if user_id in reactions_list:
                    reactions_list.remove(user_id)
                    logger.info(f"[{self.owner.id}] Reaction '{emoji}' by '{user_name}' removed from message '{original_external_id}'")
                else:
                    logger.debug(f"[{self.owner.id}] User '{user_name}' did not have reaction '{emoji}' on message '{original_external_id}' to remove")
                    return False # User hadn't reacted with this emoji
            else:
                # If user_id is None, try to remove a generic "anonymous_reaction" marker if present
                if "anonymous_reaction" in reactions_list:
                    reactions_list.remove("anonymous_reaction")
                    logger.info(f"[{self.owner.id}] Anonymous reaction '{emoji}' removed from message '{original_external_id}'")
                else:
                    logger.debug(f"[{self.owner.id}] No anonymous reaction '{emoji}' on message '{original_external_id}' to remove")
                    return False # No anonymous reaction to remove

            # If the list for this emoji is now empty, remove the emoji key itself
            if not reactions_list:
                del message_to_update['reactions'][emoji]
                logger.debug(f"[{self.owner.id}] Emoji '{emoji}' removed from message '{original_external_id}' as no users left")

        logger.info(f"[{self.owner.id}] Reaction '{emoji}' removal by '{user_name if user_id else 'anonymous'}' processed for message '{original_external_id}'")
        return True

    def _rebuild_message_map(self):
        """Inefficiently rebuilds the internal ID to index map."""
        self._state['_message_map'] = {msg['internal_id']: idx for idx, msg in enumerate(self._state['_messages']) if 'internal_id' in msg}

    def _record_message_state_change(self, change_type: str, change_data: Dict[str, Any]) -> None:
        """
        Record a message state change to the owner's timeline for replay purposes.

        Args:
            change_type: Type of state change (e.g., "message_count_milestone")
            change_data: Data describing the state change
        """
        if not self.owner or not hasattr(self.owner.get_parent_object() if hasattr(self.owner, 'get_parent_object') else None, 'add_event_to_primary_timeline'):
            return

        parent_space = self.owner.get_parent_object() if hasattr(self.owner, 'get_parent_object') else None
        if not parent_space:
            return

        event_payload = {
            "event_type": "component_state_updated",
            "target_element_id": self.owner.id,
            "is_replayable": True,  # Message state changes should be replayed
            "data": {
                "component_id": self.id,
                "component_type": self.COMPONENT_TYPE,
                "change_type": change_type,
                "change_data": change_data,
                "timestamp": time.time()
            }
        }

        try:
            parent_space.add_event_to_primary_timeline(event_payload)
            logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Recorded message state change: {change_type}")
        except Exception as e:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Error recording message state change: {e}")

    # --- Accessor Methods ---
    def get_messages(self, limit: Optional[int] = None) -> List[MessageType]:
        """Returns the current list of messages, optionally limited."""
        messages = self._state.get('_messages', [])
        if limit and limit > 0:
            return messages[-limit:] # Return the last 'limit' messages
        return messages

    def get_message_by_internal_id(self, internal_id: str) -> Optional[MessageType]:
        """Retrieves a message by its internal component ID."""
        idx = self._state.get('_message_map', {}).get(internal_id)
        if idx is not None and 0 <= idx < len(self._state['_messages']):
            return self._state['_messages'][idx]
        return None

    def get_message_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current message state for debugging and monitoring.

        Returns:
            Dictionary containing message count, senders, time range, etc.
        """
        messages = self._state.get('_messages', [])
        if not messages:
            return {
                "total_messages": 0,
                "unique_senders": 0,
                "time_range": None,
                "last_message_time": None,
                "message_types": {}
            }

        senders = set()
        message_types = {}
        timestamps = []

        for msg in messages:
            if msg.get('sender_id'):
                senders.add(msg['sender_id'])

            msg_status = msg.get('status', 'unknown')
            message_types[msg_status] = message_types.get(msg_status, 0) + 1

            if msg.get('timestamp'):
                timestamps.append(msg['timestamp'])

        timestamps.sort()
        time_range = None
        if len(timestamps) >= 2:
            time_range = {
                "earliest": timestamps[0],
                "latest": timestamps[-1],
                "span_seconds": timestamps[-1] - timestamps[0]
            }

        return {
            "total_messages": len(messages),
            "unique_senders": len(senders),
            "time_range": time_range,
            "last_message_time": timestamps[-1] if timestamps else None,
            "message_types": message_types,
            "max_messages_limit": self._max_messages
        }

    def verify_replay_integrity(self) -> Dict[str, Any]:
        """
        Verify that replayed messages maintain proper ordering and consistency.

        Returns:
            Dictionary with integrity check results
        """
        messages = self._state.get('_messages', [])
        issues = []

        # Check timestamp ordering
        timestamps = [msg.get('timestamp', 0) for msg in messages]
        if timestamps != sorted(timestamps):
            issues.append("Messages not in chronological order")

        # Check for duplicate internal IDs
        internal_ids = [msg.get('internal_id') for msg in messages if msg.get('internal_id')]
        if len(internal_ids) != len(set(internal_ids)):
            issues.append("Duplicate internal message IDs found")

        # Check message map consistency
        map_size = len(self._state.get('_message_map', {}))
        if map_size != len(messages):
            issues.append(f"Message map size ({map_size}) doesn't match message count ({len(messages)})")

        return {
            "integrity_ok": len(issues) == 0,
            "issues": issues,
            "total_messages": len(messages),
            "message_map_size": map_size
        }

    def _handle_attachment_content_available(self, attachment_payload_content: Dict[str, Any]) -> bool:
        """
        Updates an existing message's attachment with fetched content.
        Triggered by 'attachment_content_available' event.
        attachment_payload_content is the actual data for this event (e.g., from event_payload['payload']).
        """
        original_message_external_id = attachment_payload_content.get('original_message_id_external')
        attachment_id_to_update = attachment_payload_content.get('attachment_id')
        content = attachment_payload_content.get('content')

        if not all([original_message_external_id, attachment_id_to_update]):
            logger.warning(f"[{self.owner.id}] 'attachment_content_available' event missing 'original_message_id_external' or 'attachment_id'. Payload: {attachment_payload_content}")
            return False

        message_to_update = None
        for msg in self._state['_messages']:
            if msg.get('original_external_id') == original_message_external_id:
                message_to_update = msg
                break

        if not message_to_update:
            logger.warning(f"[{self.owner.id}] Cannot update attachment content: Message with external ID '{original_message_external_id}' not found.")
            return False

        attachment_found = False
        for att in message_to_update.get('attachments', []):
            if att.get('attachment_id') == attachment_id_to_update:
                att['content'] = content
                # Optionally update other fields like 'status' or 'content_retrieved_timestamp'
                att['status'] = 'content_available'
                logger.info(f"[{self.owner.id}] Content for attachment '{attachment_id_to_update}' added to message '{original_message_external_id}'.")
                attachment_found = True
                break

        if not attachment_found:
            logger.warning(f"[{self.owner.id}] Cannot update attachment content: Attachment with ID '{attachment_id_to_update}' not found in message '{original_message_external_id}'.")
            return False

        return True

    # --- NEW: Method for adding a pending outgoing message ---
    def add_pending_message(self,
                            internal_request_id: str,
                            text: str,
                            sender_id: str, # Agent's own ID
                            sender_name: str, # Agent's display name
                            timestamp: float,
                            attachments: Optional[List[Dict[str, Any]]] = None,
                            reply_to_external_id: Optional[str] = None, # If agent is replying
                            adapter_id: Optional[str] = None, # The adapter this message is going to
                            is_from_current_agent: bool = False, # FIXED: Whether this message is from the current agent
                            is_internal_origin: bool = False # NEW: Whether this message originated from internal tool calls
                           ) -> Optional[str]:
        """
        Adds a new message initiated by the local agent to the list with 'pending_send' status.
        This is typically called by MessageActionHandler before dispatching to ActivityClient.
        Returns the internal_id of the added message, or None if failed.
        """
        internal_message_id = f"msg_pending_{self.owner.id}_{int(time.time()*1000)}_{len(self._state['_messages'])}"
        processed_attachments = []
        if attachments:
            for att_data in attachments:
                if isinstance(att_data, dict):
                    processed_attachments.append({
                        # Using fields expected by MessageType, adapt if attachment structure differs for outgoing
                        "attachment_id": att_data.get("attachment_id"), # Might be generated by adapter later
                        "filename": att_data.get("filename"),
                        "content_type": att_data.get("content_type", att_data.get("attachment_type")), # Handle both
                        "size": att_data.get("size"),
                        "url": att_data.get("url"),
                        "content": att_data.get("content"),
                    })
                else:
                    logger.warning(f"[{self.owner.id}] Skipping non-dict attachment data in add_pending_message: {att_data}")

        new_message: MessageType = {
            'internal_id': internal_message_id,
            'timestamp': timestamp,
            'sender_id': sender_id,
            'sender_name': sender_name,
            'text': text,
            'original_external_id': None, # Will be filled upon confirmation
            'reply_to_external_id': reply_to_external_id, # Store if it's a reply
            'adapter_id': adapter_id,
            'is_edited': False,
            'reactions': {},
            'read_by': [],
            'attachments': processed_attachments,
            'status': "pending_send", # Key change for outgoing messages
            'internal_request_id': internal_request_id, # For matching confirmation
            'error_details': None,
            'is_from_current_agent': is_from_current_agent,  # FIXED: Store agent flag for deduplication
            'is_internal_origin': is_internal_origin  # NEW: Track if message originated from internal tool calls
        }
        self._state['_messages'].append(new_message)
        self._state['_message_map'][internal_message_id] = len(self._state['_messages']) - 1

        logger.info(f"[{self.owner.id}] Pending outgoing message (req_id: {internal_request_id}) added. Total messages: {len(self._state['_messages'])}")

        # Optional: Enforce max_messages limit (might prune old confirmed messages)
        if self._max_messages and len(self._state['_messages']) > self._max_messages:
            oldest_message = self._state['_messages'].pop(0)
            if oldest_message and 'internal_id' in oldest_message:
                 del self._state['_message_map'][oldest_message['internal_id']]
                 self._rebuild_message_map() # Inefficient!
                 logger.debug(f"[{self.owner.id}] Pruned oldest message due to max_messages limit while adding pending message.")

        return internal_message_id

    # --- NEW: Handlers for outgoing message lifecycle ---
    def _handle_message_send_confirmed(self, confirm_content: Dict[str, Any]) -> bool:
        """
        Updates a pending message to 'sent' status upon confirmation from adapter.
        Now handles generic action success payload structure.
        """

        internal_req_id = confirm_content.get('internal_request_id')
        adapter_response_data = confirm_content.get('adapter_response_data', {})

        # Extract external message IDs from adapter response data (action-specific logic)
        external_ids = adapter_response_data.get('message_ids', [])

        if not internal_req_id or not external_ids:
            logger.warning(f"[{self.owner.id}] Message send confirmation missing 'internal_request_id' or 'message_ids' in adapter_response_data. Payload: {confirm_content}")
            return False

        message_to_update: Optional[MessageType] = None
        # Find by internal_request_id (can be slow, consider map if performance is an issue)
        idx_to_update = -1
        for idx, msg in enumerate(self._state['_messages']):
            if msg.get('internal_request_id') == internal_req_id and msg.get('status') == "pending_send":
                message_to_update = msg
                idx_to_update = idx
                break
            elif msg.get('internal_request_id') == internal_req_id and msg.get('status') != "pending_send":
                logger.warning(f"[{self.owner.id}] Found message with internal_request_id '{internal_req_id}' but it's not in 'pending_send' status. Current status: {msg.get('status')}")
        if message_to_update:
            message_to_update['status'] = "sent"
            # Assuming the first ID is the primary one for now
            message_to_update['original_external_id'] = external_ids[0]
            if 'confirmed_timestamp' in confirm_content and confirm_content['confirmed_timestamp']:
                message_to_update['timestamp'] = confirm_content['confirmed_timestamp']

            # Update in the list directly (if Python list mutation reflects, otherwise reassign)
            # self._state['_messages'][idx_to_update] = message_to_update # Not strictly needed if dict is mutated in place

            logger.info(f"[{self.owner.id}] Pending message (req_id: {internal_req_id}) confirmed as sent. External ID: {external_ids[0]}.")

            # NEW: Emit replayable timeline event for agent outgoing message
            self._emit_agent_message_confirmed_event(message_to_update, confirm_content)

            return True
        else:
            logger.warning(f"[{self.owner.id}] Could not find 'pending_send' message with internal_request_id '{internal_req_id}' to confirm.")
            return False

    def _handle_message_send_failed(self, failure_content: Dict[str, Any]) -> bool:
        """
        Updates a pending message to 'failed_to_send' status.
        failure_content is expected to contain:
        - internal_request_id: str
        - error_message: str
        - failed_timestamp: Optional[float]
        """
        internal_req_id = failure_content.get('internal_request_id')
        error_msg = failure_content.get('error_message')

        if not internal_req_id:
            logger.warning(f"[{self.owner.id}] Message send failure event missing 'internal_request_id'. Payload: {failure_content}")
            return False

        message_to_update: Optional[MessageType] = None
        idx_to_update = -1

        for idx, msg in enumerate(self._state['_messages']):
            if msg.get('internal_request_id') == internal_req_id and msg.get('status') == "pending_send":
                message_to_update = msg
                idx_to_update = idx
                break
            elif msg.get('internal_request_id') == internal_req_id and msg.get('status') != "pending_send":
                logger.warning(f"[{self.owner.id}] Found message with internal_request_id '{internal_req_id}' but it's not in 'pending_send' status. Current status: {msg.get('status')}")

        if message_to_update:
            message_to_update['status'] = "failed_to_send"
            message_to_update['error_details'] = error_msg or "Unknown send failure"
            if 'failed_timestamp' in failure_content and failure_content['failed_timestamp']:
                message_to_update['timestamp'] = failure_content['failed_timestamp'] # Update to failure time

            logger.error(f"[{self.owner.id}] Agent message failed to send (req_id: {internal_req_id}). Error: {error_msg}")

            # NEW: Trigger agent activation for message send failure
            # This allows the agent to respond to the failure, potentially retry, or take alternative action
            self._emit_send_failure_activation_call(message_to_update, failure_content)

            return True
        else:
            logger.warning(f"[{self.owner.id}] Could not find 'pending_send' message with internal_request_id '{internal_req_id}' to mark as failed.")
            return False

    # --- Other potential methods ---
    # def mark_message_read(self, internal_id: str, user_id: str): ...
    # def add_reaction(self, internal_id: str, reaction: str, user_id: str): ...
    # def remove_reaction(self, internal_id: str, reaction: str, user_id: str): ...

    def _emit_agent_message_confirmed_event(self, confirmed_message: MessageType, confirm_content: Dict[str, Any]) -> None:
        """
        Emit a replayable timeline event for a confirmed agent outgoing message.
        This ensures agent messages are restored during replay for complete conversation history.

        Args:
            confirmed_message: The message that was just confirmed
            confirm_content: The confirmation payload from the adapter
        """
        if not self.owner:
            logger.warning(f"[MessageListComponent] Cannot emit agent message event: No owner element")
            return

        parent_space = self.owner.get_parent_object()
        if not parent_space or not hasattr(parent_space, 'receive_event'):
            logger.warning(f"[{self.owner.id}] Cannot emit agent message event: Parent space not found or not event-capable")
            return

        # Get conversation context from the chat element itself
        # The chat element should have adapter_id and external_conversation_id set
        adapter_id = getattr(self.owner, 'adapter_id', confirmed_message.get('adapter_id', 'unknown'))
        external_conversation_id = getattr(self.owner, 'external_conversation_id', confirm_content.get('conversation_id', 'unknown'))

        # Determine if this is a DM from the chat element's recipient_info
        is_dm = False
        if hasattr(self.owner, 'recipient_info'):
            recipient_info = getattr(self.owner, 'recipient_info', {})
            is_dm = recipient_info.get('is_dm', False)
        else:
            # Fallback: try to infer from element name or other attributes
            # For now, assume it's a DM if we can't determine otherwise
            is_dm = True  # Conservative default for agent messages

        # Create a message_received-like event for the agent's confirmed message
        # This allows it to be replayed during system startup
        agent_message_payload = {
            "source_adapter_id": adapter_id,
            "external_conversation_id": external_conversation_id,
            "is_dm": is_dm,  # Use the preserved DM flag
            "text": confirmed_message.get('text', ''),
            "sender_external_id": confirmed_message.get('sender_id'),
            "sender_display_name": confirmed_message.get('sender_name', 'Agent'),
            "timestamp": confirmed_message.get('timestamp', time.time()),
            "original_message_id_external": confirmed_message.get('original_external_id'),
            "mentions": [],  # Agents typically don't mention others in their messages
            "attachments": confirmed_message.get('attachments', []),
            "internal_request_id": confirmed_message.get('internal_request_id'),
            "message_source": "agent_outgoing",  # Mark as agent-originated
            "is_internal_origin": confirmed_message.get('is_internal_origin', True),  # NEW: Preserve internal origin flag for replay
            "original_adapter_data": confirm_content  # Store original confirmation data
        }

        agent_message_event = {
            "event_type": "agent_message_confirmed",  # NEW: Specific event type for agent messages
            "event_id": f"agent_msg_{self.owner.id}_{confirmed_message.get('internal_request_id', int(time.time()*1000))}",
            "source_element_id": self.owner.id,
            "source_adapter_id": adapter_id,
            "external_conversation_id": external_conversation_id,  # Use preserved conversation ID
            "timestamp": confirmed_message.get('timestamp', time.time()),
            "is_replayable": True,  # CRITICAL: Agent messages must be replayable for conversation restoration
            "payload": agent_message_payload
        }

        # Use basic timeline context (let parent space handle specifics)
        timeline_context = {"timeline_id": parent_space.get_primary_timeline() if hasattr(parent_space, 'get_primary_timeline') else None}

        try:
            parent_space.receive_event(agent_message_event, timeline_context)
            logger.info(f"[{self.owner.id}] Emitted replayable agent_message_confirmed event for req_id: {confirmed_message.get('internal_request_id')} (is_dm: {is_dm}, conv_id: {external_conversation_id})")
        except Exception as e:
            logger.error(f"[{self.owner.id}] Error emitting agent message event: {e}", exc_info=True)

    def _handle_agent_message_confirmed(self, agent_message_content: Dict[str, Any]) -> bool:
        """
        Handles agent_message_confirmed events during replay.
        This restores agent outgoing messages to maintain complete conversation history.

        FIXED: Now checks if message already exists to prevent duplicates during live confirmations.

        Args:
            agent_message_content: The agent message payload from the timeline event

        Returns:
            True if handled successfully, False otherwise
        """
        # FIXED: Check if this message already exists to prevent duplicates
        # This can happen when:
        # 1. Live confirmation: message exists as pending, just got confirmed, and then this event is processed
        # 2. Replay: message doesn't exist yet and needs to be restored from timeline

        internal_request_id = agent_message_content.get('internal_request_id')
        external_message_id = agent_message_content.get('original_message_id_external')

        # Check if message already exists by internal_request_id or external_id
        existing_message = None
        if internal_request_id:
            for msg in self._state['_messages']:
                if msg.get('internal_request_id') == internal_request_id:
                    existing_message = msg
                    break

        # If not found by internal_request_id, try external_id
        if not existing_message and external_message_id:
            for msg in self._state['_messages']:
                if msg.get('original_external_id') == external_message_id:
                    existing_message = msg
                    break

        if existing_message:
            # Message already exists (live confirmation scenario) - just ensure it's marked correctly
            logger.debug(f"[{self.owner.id}] Agent message with req_id '{internal_request_id}' already exists. Status: {existing_message.get('status')}. Skipping duplicate addition.")

            # Ensure the existing message has the correct external_id if it was missing
            if not existing_message.get('original_external_id') and external_message_id:
                existing_message['original_external_id'] = external_message_id
                logger.debug(f"[{self.owner.id}] Updated existing message with external_id: {external_message_id}")

            return True

        # Message doesn't exist - this is a replay scenario, add it
        logger.info(f"[{self.owner.id}] REPLAY: Adding agent message with req_id '{internal_request_id}' from timeline")

        # During replay, we want to add the agent message to the conversation
        # as if it were a regular message (which it is, just from the agent)

        internal_message_id = f"msg_agent_{self.owner.id}_{int(time.time()*1000)}_{len(self._state['_messages'])}"

        # Process attachments if present
        processed_attachments = []
        for att_data in agent_message_content.get('attachments', []):
            if isinstance(att_data, dict):
                processed_attachments.append({
                    "attachment_id": att_data.get("attachment_id"),
                    "filename": att_data.get("filename"),
                    "content_type": att_data.get("content_type"),
                    "size": att_data.get("size"),
                    "url": att_data.get("url"),
                    "content": att_data.get("content"),
                })

        # Create the agent message entry
        agent_message: MessageType = {
            'internal_id': internal_message_id,
            'timestamp': agent_message_content.get('timestamp', time.time()),
            'sender_id': agent_message_content.get('sender_external_id'),
            'sender_name': agent_message_content.get('sender_display_name', 'Agent'),
            'text': agent_message_content.get('text', ''),
            'original_external_id': agent_message_content.get('original_message_id_external'),
            'adapter_id': agent_message_content.get('source_adapter_id'),
            'is_edited': False,
            'reactions': {},
            'read_by': [],
            'attachments': processed_attachments,
            'status': "sent",  # Agent messages are confirmed as sent
            'internal_request_id': agent_message_content.get('internal_request_id'),
            'error_details': None,
            'message_source': "agent_outgoing",  # Mark as agent-originated for debugging
            'is_from_current_agent': True,  # FIXED: Agent messages during replay are from current agent
            'is_internal_origin': agent_message_content.get('is_internal_origin', True)  # NEW: Preserve internal origin flag (default True for agent messages)
        }

        self._state['_messages'].append(agent_message)
        self._state['_message_map'][internal_message_id] = len(self._state['_messages']) - 1

        # Optional: Enforce max_messages limit
        if self._max_messages and len(self._state['_messages']) > self._max_messages:
            oldest_message = self._state['_messages'].pop(0)
            if oldest_message and 'internal_id' in oldest_message:
                del self._state['_message_map'][oldest_message['internal_id']]
                self._rebuild_message_map()
                logger.debug(f"[{self.owner.id}] Pruned oldest message due to max_messages limit during agent message replay.")

        logger.info(f"[{self.owner.id}] REPLAY: Restored agent outgoing message '{agent_message_content.get('text', '')[:50]}...' ({len(self._state.get('_messages', []))} total messages)")

        return True

    def _emit_send_failure_activation_call(self, failed_message: MessageType, failure_content: Dict[str, Any]) -> None:
        """
        Emits an "activation_call" event when an agent's message fails to send.
        This allows the agent to respond to the failure, potentially retry, or take alternative action.

        Args:
            failed_message: The message that failed to send
            failure_content: The failure payload from the adapter
        """
        if not self.owner:
            logger.warning(f"[MessageListComponent] Cannot emit send failure activation: No owner element")
            return

        parent_space = self.owner.get_parent_object()
        if not parent_space or not hasattr(parent_space, 'receive_event'):
            logger.warning(f"[{self.owner.id}] Cannot emit send failure activation: Parent space not found or not event-capable")
            return

        # Get conversation context from the chat element itself
        adapter_id = getattr(self.owner, 'adapter_id', failed_message.get('adapter_id', 'unknown'))
        external_conversation_id = getattr(self.owner, 'external_conversation_id', failure_content.get('conversation_id', 'unknown'))

        # Determine if this is a DM from the chat element's recipient_info
        is_dm = False
        if hasattr(self.owner, 'recipient_info'):
            recipient_info = getattr(self.owner, 'recipient_info', {})
            is_dm = recipient_info.get('is_dm', False)
        else:
            # Fallback: assume it's a DM if we can't determine otherwise
            is_dm = True  # Conservative default for agent messages

        # Create focused activation context for the failed message
        focus_context = {
            "focus_element_id": self.owner.id,  # The element that should be rendered
            "focus_element_type": self.owner.__class__.__name__,
            "focus_element_name": getattr(self.owner, 'name', 'Unknown'),
            "conversation_context": {
                "adapter_id": adapter_id,
                "external_conversation_id": external_conversation_id,
                "is_dm": is_dm,
                "conversation_id": external_conversation_id,
                "activation_reason": "message_send_failed",
                "failed_message_text": failed_message.get('text', '')[:100],  # Preview of failed message
                "error_message": failure_content.get('error_message', 'Unknown send failure'),
                "internal_request_id": failed_message.get('internal_request_id'),
                "failed_timestamp": failure_content.get('failed_timestamp', time.time())
            }
        }

        activation_event = {
            "event_type": "activation_call",
            "event_id": f"activation_send_failure_{self.owner.id}_{int(time.time()*1000)}",
            "source_element_id": self.owner.id,
            "activation_reason": "message_send_failed",
            "triggering_event_type": "connectome_message_send_failed",
            "timestamp": time.time(),
            "is_replayable": False,  # Activation calls are runtime-only
            "focus_context": focus_context,  # Context for focused rendering
            "payload": {
                "reason": "message_send_failed",
                "source_element_id": self.owner.id,
                "triggering_event_type": "connectome_message_send_failed",
                "focus_context": focus_context,
                "conversation_id": external_conversation_id,
                "failed_message": {
                    "internal_request_id": failed_message.get('internal_request_id'),
                    "text": failed_message.get('text'),
                    "timestamp": failed_message.get('timestamp'),
                    "error_details": failed_message.get('error_details')
                },
                "error_message": failure_content.get('error_message', 'Unknown send failure'),
                "adapter_id": adapter_id
            }
        }

        # Use basic timeline context (let parent space handle specifics)
        timeline_context = {"timeline_id": parent_space.get_primary_timeline() if hasattr(parent_space, 'get_primary_timeline') else None}

        try:
            parent_space.receive_event(activation_event, timeline_context)
            logger.warning(f"[{self.owner.id}] Emitted activation_call for message send failure. Reason: message_send_failed, Error: {failure_content.get('error_message', 'Unknown')}")
        except Exception as e:
            logger.error(f"[{self.owner.id}] Error emitting send failure activation: {e}", exc_info=True)

    # # --- NEW: Methods for immediate local state updates (called by tools before external confirmation) ---
    # def mark_message_pending_delete(self, external_message_id: str, requesting_agent_id: str) -> bool:
    #     """
    #     Immediately marks a message as pending deletion in local state.
    #     Called by delete_message tool before external confirmation.

    #     Args:
    #         external_message_id: External ID of message to mark as pending delete
    #         requesting_agent_id: ID of agent requesting the deletion

    #     Returns:
    #         True if message found and marked, False otherwise
    #     """
    #     message_to_update = None
    #     for msg in self._state['_messages']:
    #         if msg.get('original_external_id') == external_message_id:
    #             message_to_update = msg
    #             break

    #     if message_to_update:
    #         # Store original text for potential restore if deletion fails
    #         if 'original_text_before_pending_delete' not in message_to_update:
    #             message_to_update['original_text_before_pending_delete'] = message_to_update.get('text')

    #         message_to_update['text'] = "[🗑️ Deleting message...]"
    #         message_to_update['status'] = "pending_delete"
    #         message_to_update['pending_delete_by_agent_id'] = requesting_agent_id
    #         message_to_update['pending_delete_timestamp'] = time.time()

    #         logger.info(f"[{self.owner.id}] Marked message '{external_message_id}' as pending deletion by agent '{requesting_agent_id}'")
    #         return True
    #     else:
    #         logger.warning(f"[{self.owner.id}] Cannot mark message '{external_message_id}' as pending delete: Message not found")
    #         return False

    # def mark_message_pending_edit(self, external_message_id: str, new_text: str, requesting_agent_id: str) -> bool:
    #     """
    #     Immediately shows edited text with pending status in local state.
    #     Called by edit_message tool before external confirmation.

    #     Args:
    #         external_message_id: External ID of message to edit
    #         new_text: New text content to show
    #         requesting_agent_id: ID of agent requesting the edit

    #     Returns:
    #         True if message found and updated, False otherwise
    #     """
    #     message_to_update = None
    #     for msg in self._state['_messages']:
    #         if msg.get('original_external_id') == external_message_id:
    #             message_to_update = msg
    #             break

    #     if message_to_update:
    #         # Store original text for potential restore if edit fails
    #         if 'original_text_before_pending_edit' not in message_to_update:
    #             message_to_update['original_text_before_pending_edit'] = message_to_update.get('text')

    #         message_to_update['text'] = f"{new_text} ✏️"  # Show new text with edit indicator
    #         message_to_update['status'] = "pending_edit"
    #         message_to_update['pending_edit_by_agent_id'] = requesting_agent_id
    #         message_to_update['pending_edit_timestamp'] = time.time()
    #         message_to_update['pending_new_text'] = new_text  # Store clean version

    #         logger.info(f"[{self.owner.id}] Marked message '{external_message_id}' as pending edit by agent '{requesting_agent_id}'")
    #         return True
    #     else:
    #         logger.warning(f"[{self.owner.id}] Cannot mark message '{external_message_id}' as pending edit: Message not found")
    #         return False

    def add_pending_reaction(self, external_message_id: str, emoji: str, requesting_agent_id: str) -> bool:
        """
        Immediately adds a reaction with pending status in local state.
        Called by add_reaction tool before external confirmation.

        Args:
            external_message_id: External ID of message to react to
            emoji: Emoji to add
            requesting_agent_id: ID of agent adding the reaction

        Returns:
            True if message found and reaction added, False otherwise
        """
        message_to_update = None
        for msg in self._state['_messages']:
            if msg.get('original_external_id') == external_message_id:
                message_to_update = msg
                break

        if message_to_update:
            if 'reactions' not in message_to_update:
                message_to_update['reactions'] = {}

            if emoji not in message_to_update['reactions']:
                message_to_update['reactions'][emoji] = []

            # Add reaction with pending marker
            pending_reaction_id = f"pending_{requesting_agent_id}"
            if pending_reaction_id not in message_to_update['reactions'][emoji]:
                message_to_update['reactions'][emoji].append(pending_reaction_id)

                # Track pending reactions for cleanup
                if 'pending_reactions' not in message_to_update:
                    message_to_update['pending_reactions'] = {}
                message_to_update['pending_reactions'][f"{emoji}_{requesting_agent_id}"] = {
                    "emoji": emoji,
                    "agent_id": requesting_agent_id,
                    "timestamp": time.time()
                }

                logger.info(f"[{self.owner.id}] Added pending reaction '{emoji}' by agent '{requesting_agent_id}' to message '{external_message_id}'")
                return True

        logger.warning(f"[{self.owner.id}] Cannot add pending reaction to message '{external_message_id}': Message not found")
        return False

    def remove_pending_reaction(self, external_message_id: str, emoji: str, requesting_agent_id: str) -> bool:
        """
        Immediately removes a reaction with pending status in local state.
        Called by remove_reaction tool before external confirmation.

        Args:
            external_message_id: External ID of message to remove reaction from
            emoji: Emoji to remove
            requesting_agent_id: ID of agent removing the reaction

        Returns:
            True if message found and reaction removed, False otherwise
        """
        message_to_update = None
        for msg in self._state['_messages']:
            if msg.get('original_external_id') == external_message_id:
                message_to_update = msg
                break

        if message_to_update and 'reactions' in message_to_update and emoji in message_to_update['reactions']:
            # Look for existing reaction by this agent (could be pending or confirmed)
            reactions_list = message_to_update['reactions'][emoji]
            agent_reaction_found = False

            # Remove confirmed reaction or pending reaction
            for reaction_marker in [requesting_agent_id, f"pending_{requesting_agent_id}"]:
                if reaction_marker in reactions_list:
                    reactions_list.remove(reaction_marker)
                    agent_reaction_found = True
                    break

            if agent_reaction_found:
                # If no reactions left for this emoji, remove the emoji
                if not reactions_list:
                    del message_to_update['reactions'][emoji]

                # Add pending removal marker
                if 'pending_reaction_removals' not in message_to_update:
                    message_to_update['pending_reaction_removals'] = {}
                message_to_update['pending_reaction_removals'][f"{emoji}_{requesting_agent_id}"] = {
                    "emoji": emoji,
                    "agent_id": requesting_agent_id,
                    "timestamp": time.time()
                }

                logger.info(f"[{self.owner.id}] Removed pending reaction '{emoji}' by agent '{requesting_agent_id}' from message '{external_message_id}'")
                return True

        logger.warning(f"[{self.owner.id}] Cannot remove pending reaction from message '{external_message_id}': Message or reaction not found")
        return False

    def restore_message_from_pending_state(self, external_message_id: str, operation_type: str) -> bool:
        """
        Restores a message from pending state if external operation fails.

        Args:
            external_message_id: External ID of message to restore
            operation_type: Type of operation that failed ("delete", "edit", "add_reaction", "remove_reaction")

        Returns:
            True if message found and restored, False otherwise
        """
        message_to_restore = None
        for msg in self._state['_messages']:
            if msg.get('original_external_id') == external_message_id:
                message_to_restore = msg
                break

        if not message_to_restore:
            logger.warning(f"[{self.owner.id}] Cannot restore message '{external_message_id}': Message not found")
            return False

        if operation_type == "delete":
            if 'original_text_before_pending_delete' in message_to_restore:
                message_to_restore['text'] = message_to_restore['original_text_before_pending_delete']
                del message_to_restore['original_text_before_pending_delete']
            message_to_restore['status'] = "received"  # Restore to normal
            message_to_restore.pop('pending_delete_by_agent_id', None)
            message_to_restore.pop('pending_delete_timestamp', None)

        elif operation_type == "edit":
            if 'original_text_before_pending_edit' in message_to_restore:
                message_to_restore['text'] = message_to_restore['original_text_before_pending_edit']
                del message_to_restore['original_text_before_pending_edit']
            message_to_restore['status'] = "received"  # Restore to normal
            message_to_restore.pop('pending_edit_by_agent_id', None)
            message_to_restore.pop('pending_edit_timestamp', None)
            message_to_restore.pop('pending_new_text', None)

        # Clear pending reaction markers
        message_to_restore.pop('pending_reactions', None)
        message_to_restore.pop('pending_reaction_removals', None)

        logger.info(f"[{self.owner.id}] Restored message '{external_message_id}' from pending {operation_type} state")
        return True

    def _handle_action_success(self, success_content: Dict[str, Any]) -> bool:
        """
        Handles generic action success events and routes to action-specific handlers.

        Args:
            success_content: The action success payload containing action_type and adapter_response_data

        Returns:
            True if handled successfully, False otherwise
        """
        action_type = success_content.get('action_type')

        if action_type == "send_message":
            return self._handle_message_send_confirmed(success_content)
        elif action_type == "delete_message":
            return self._handle_delete_message_confirmed(success_content)
        elif action_type == "edit_message":
            return self._handle_edit_message_confirmed(success_content)
        elif action_type in ["add_reaction", "remove_reaction"]:
            return self._handle_reaction_action_confirmed(success_content)
        else:
            logger.warning(f"[{self.owner.id}] Unknown action_type '{action_type}' in action success. Ignoring.")
            return False

    def _handle_action_failure(self, failure_content: Dict[str, Any]) -> bool:
        """
        Handles generic action failure events and routes to action-specific handlers.

        Args:
            failure_content: The action failure payload containing action_type and error details

        Returns:
            True if handled successfully, False otherwise
        """
        action_type = failure_content.get('action_type')

        if action_type == "send_message":
            return self._handle_message_send_failed(failure_content)
        elif action_type in ["delete_message", "edit_message", "add_reaction", "remove_reaction"]:
            return self._handle_message_action_failed(failure_content)
        else:
            logger.warning(f"[{self.owner.id}] Unknown action_type '{action_type}' in action failure. Ignoring.")
            return False

    def _handle_delete_message_confirmed(self, success_content: Dict[str, Any]) -> bool:
        """
        Handles confirmation of delete_message action success.
        """
        internal_req_id = success_content.get('internal_request_id')
        adapter_response_data = success_content.get('adapter_response_data', {})

        # For delete confirmations, we just need to know it succeeded
        logger.info(f"[{self.owner.id}] Delete message action confirmed for req_id: {internal_req_id}")

        # The actual deletion should have already been handled by _handle_delete_message
        # This confirmation just means the adapter successfully processed our request
        return True

    def _handle_edit_message_confirmed(self, success_content: Dict[str, Any]) -> bool:
        """
        Handles confirmation of edit_message action success.
        """
        internal_req_id = success_content.get('internal_request_id')
        adapter_response_data = success_content.get('adapter_response_data', {})

        # For edit confirmations, we just need to know it succeeded
        logger.info(f"[{self.owner.id}] Edit message action confirmed for req_id: {internal_req_id}")

        # The actual edit should have already been handled by _handle_edit_message
        # This confirmation just means the adapter successfully processed our request
        return True

    def _handle_reaction_action_confirmed(self, success_content: Dict[str, Any]) -> bool:
        """
        Handles confirmation of add_reaction or remove_reaction action success.
        """
        action_type = success_content.get('action_type')
        internal_req_id = success_content.get('internal_request_id')
        adapter_response_data = success_content.get('adapter_response_data', {})

        logger.info(f"[{self.owner.id}] Reaction action '{action_type}' confirmed for req_id: {internal_req_id}")

        # The actual reaction add/remove should have already been handled by
        # _handle_reaction_added or _handle_reaction_removed
        # This confirmation just means the adapter successfully processed our request
        return True

    def _handle_message_action_failed(self, failure_content: Dict[str, Any]) -> bool:
        """
        Handles failure of message action (delete, edit, reaction).
        Restores the message from pending state if needed.
        """
        action_type = failure_content.get('action_type')
        internal_req_id = failure_content.get('internal_request_id')
        error_msg = failure_content.get('error_message')
        adapter_response_data = failure_content.get('adapter_response_data', {})

        logger.warning(f"[{self.owner.id}] Message action '{action_type}' failed for req_id: {internal_req_id}. Error: {error_msg}")

        # Try to extract affected message ID for restoration
        affected_message_id = adapter_response_data.get('affected_message_id')
        if affected_message_id:
            # Restore message from pending state since the action failed
            self.restore_message_from_pending_state(affected_message_id, action_type.replace('_message', '').replace('_reaction', ''))

        return True

    def _handle_bulk_history_received(self, bulk_history_content: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handles bulk history processing with sophisticated reconciliation logic.

        NEW: Uses the reconciliation engine to properly handle:
        - Empty MessageList scenarios
        - Overlap detection and processing
        - Edit reconciliation (history has higher order-of-truth)
        - Deletion reconciliation (missing from history = deleted)
        - Gap detection and system message management

        Args:
            bulk_history_content: The bulk history payload containing history_messages
            timeline_context: Timeline context for the event

        Returns:
            True if handled successfully, False otherwise
        """
        try:
            history_messages = bulk_history_content.get("history_messages", [])
            source_adapter_id = bulk_history_content.get("source_adapter_id")
            external_conversation_id = bulk_history_content.get("external_conversation_id")
            is_dm = bulk_history_content.get("is_dm", False)
            is_replay_mode = timeline_context.get('replay_mode', False)

            logger.info(f"[{self.owner.id}] Processing bulk history with reconciliation: {len(history_messages)} messages from {source_adapter_id}")

            if not history_messages:
                logger.info(f"[{self.owner.id}] No history messages to process")
                return True

            # NEW: Use the sophisticated reconciliation engine
            reconciliation_results = self._reconcile_history_with_existing_messages(history_messages, source_adapter_id)

            # Log detailed reconciliation results
            logger.info(f"[{self.owner.id}] Bulk history reconciliation complete:")
            logger.info(f"  - Processed: {reconciliation_results['processed_count']} messages")
            logger.info(f"  - Added: {reconciliation_results['added_count']} new messages")
            logger.info(f"  - Edited: {reconciliation_results['edited_count']} messages")
            logger.info(f"  - Deleted: {reconciliation_results['deleted_count']} messages")
            logger.info(f"  - Gap markers added: {reconciliation_results['gap_messages_added']}")
            logger.info(f"  - Gap markers removed: {reconciliation_results['gap_messages_removed']}")
            logger.info(f"  - Total messages now: {len(self._state.get('_messages', []))}")

            # Log any errors that occurred during reconciliation
            if reconciliation_results['errors']:
                logger.warning(f"[{self.owner.id}] Reconciliation errors: {reconciliation_results['errors']}")

            # NOTE: VEIL delta emission is handled by handle_event() for all events uniformly
            # No need to emit delta here to avoid double emission

            # During replay, provide summary of restoration
            if is_replay_mode:
                total_changes = (reconciliation_results['added_count'] +
                               reconciliation_results['edited_count'] +
                               reconciliation_results['deleted_count'])
                logger.info(f"[{self.owner.id}] REPLAY: Reconciled {total_changes} changes from bulk history "
                           f"({len(self._state.get('_messages', []))} total messages)")

            return True

        except Exception as e:
            logger.error(f"[{self.owner.id}] Error in bulk history reconciliation: {e}", exc_info=True)
            return False

    # --- NEW: History Reconciliation Helper Methods ---
    def _get_message_timestamp_range(self) -> Optional[Dict[str, float]]:
        """
        Get the timestamp range of messages in the current MessageList.

        Returns:
            Dict with 'min' and 'max' timestamps, or None if no messages
        """
        messages = self._state.get('_messages', [])
        if not messages:
            return None

        timestamps = [msg.get('timestamp') for msg in messages if msg.get('timestamp')]
        if not timestamps:
            return None

        return {
            'min': min(timestamps),
            'max': max(timestamps)
        }

    def _get_history_timestamp_range(self, history_messages: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """
        Get the timestamp range of messages in the history batch.

        Args:
            history_messages: List of history message dictionaries

        Returns:
            Dict with 'min' and 'max' timestamps, or None if no messages
        """
        if not history_messages:
            return None

        timestamps = [msg.get('timestamp') for msg in history_messages if msg.get('timestamp')]
        if not timestamps:
            return None

        return {
            'min': min(timestamps),
            'max': max(timestamps)
        }

    def _detect_history_overlap(self, history_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect overlap type between history batch and existing MessageList.

        Args:
            history_messages: List of history message dictionaries

        Returns:
            Dict containing overlap analysis:
            - overlap_type: 'no_existing', 'no_overlap_earlier', 'no_overlap_later', 'has_overlap'
            - messagelist_range: timestamp range of existing messages
            - history_range: timestamp range of history messages
            - gap_info: information about gaps if applicable
        """
        messagelist_range = self._get_message_timestamp_range()
        history_range = self._get_history_timestamp_range(history_messages)

        if not messagelist_range:
            return {
                'overlap_type': 'no_existing',
                'messagelist_range': None,
                'history_range': history_range,
                'gap_info': None
            }

        if not history_range:
            return {
                'overlap_type': 'no_history',
                'messagelist_range': messagelist_range,
                'history_range': None,
                'gap_info': None
            }

        # Check for no overlap scenarios
        if history_range['max'] < messagelist_range['min']:
            # History is entirely earlier than existing messages
            return {
                'overlap_type': 'no_overlap_earlier',
                'messagelist_range': messagelist_range,
                'history_range': history_range,
                'gap_info': {
                    'gap_start': history_range['max'],
                    'gap_end': messagelist_range['min'],
                    'gap_type': 'between_history_and_existing'
                }
            }
        elif history_range['min'] > messagelist_range['max']:
            # History is entirely later than existing messages
            return {
                'overlap_type': 'no_overlap_later',
                'messagelist_range': messagelist_range,
                'history_range': history_range,
                'gap_info': {
                    'gap_start': messagelist_range['max'],
                    'gap_end': history_range['min'],
                    'gap_type': 'between_existing_and_history'
                }
            }
        else:
            # There is some overlap
            return {
                'overlap_type': 'has_overlap',
                'messagelist_range': messagelist_range,
                'history_range': history_range,
                'gap_info': None
            }

    def _create_system_gap_message(self, gap_start: float, gap_end: float, gap_type: str) -> Dict[str, Any]:
        """
        Create a system message indicating missing messages in a time range.

        Args:
            gap_start: Start timestamp of the gap
            gap_end: End timestamp of the gap
            gap_type: Type of gap for context

        Returns:
            MessageType dictionary for the system gap message
        """
        import datetime

        # Format timestamps for display
        start_time = datetime.datetime.fromtimestamp(gap_start).strftime('%Y-%m-%d %H:%M:%S')
        end_time = datetime.datetime.fromtimestamp(gap_end).strftime('%Y-%m-%d %H:%M:%S')

        gap_duration = gap_end - gap_start
        if gap_duration < 3600:  # Less than 1 hour
            duration_text = f"{int(gap_duration / 60)} minutes"
        elif gap_duration < 86400:  # Less than 1 day
            duration_text = f"{gap_duration / 3600:.1f} hours"
        else:
            duration_text = f"{gap_duration / 86400:.1f} days"

        internal_message_id = f"gap_msg_{int(gap_start)}_{int(gap_end)}"
        gap_timestamp = (gap_start + gap_end) / 2  # Middle of gap

        gap_message = {
            'internal_id': internal_message_id,
            'timestamp': gap_timestamp,
            'sender_id': 'SYSTEM',
            'sender_name': 'SYSTEM',
            'text': f"📭 Messages from {start_time} to {end_time} are not available ({duration_text} gap)",
            'original_external_id': None,  # System messages don't have external IDs
            'adapter_id': None,
            'is_edited': False,
            'reactions': {},
            'read_by': [],
            'attachments': [],
            'status': 'system_gap_marker',
            'internal_request_id': None,
            'error_details': None,
            'message_source': 'system_generated',
            'gap_start': gap_start,
            'gap_end': gap_end,
            'gap_type': gap_type
        }

        logger.info(f"[{self.owner.id}] Created system gap message for {duration_text} gap from {start_time} to {end_time}")
        return gap_message

    def _remove_existing_gap_messages(self, overlap_range_start: float, overlap_range_end: float) -> int:
        """
        Remove existing gap messages that are now filled by new history.

        Args:
            overlap_range_start: Start of the range now covered by history
            overlap_range_end: End of the range now covered by history

        Returns:
            Number of gap messages removed
        """
        messages_to_remove = []

        for idx, msg in enumerate(self._state['_messages']):
            if (msg.get('status') == 'system_gap_marker' and
                msg.get('message_source') == 'system_generated'):

                gap_start = msg.get('gap_start', 0)
                gap_end = msg.get('gap_end', 0)

                # Check if this gap is now covered by the new history
                if (gap_start >= overlap_range_start and gap_end <= overlap_range_end):
                    messages_to_remove.append(idx)
                    logger.info(f"[{self.owner.id}] Removing gap message {msg.get('internal_id')} - gap now filled by history")

        # Remove messages in reverse order to maintain indices
        for idx in reversed(messages_to_remove):
            removed_msg = self._state['_messages'].pop(idx)
            if removed_msg.get('internal_id') in self._state.get('_message_map', {}):
                del self._state['_message_map'][removed_msg['internal_id']]

        # Rebuild message map if we removed any messages
        if messages_to_remove:
            self._rebuild_message_map()

        return len(messages_to_remove)

    def _reconcile_history_with_existing_messages(self, history_messages: List[Dict[str, Any]], source_adapter_id: str) -> Dict[str, Any]:
        """
        Main reconciliation engine that applies history messages against existing MessageList.
        Implements the sophisticated reconciliation rules with higher order-of-truth for history.

        Args:
            history_messages: List of history message dictionaries from adapter
            source_adapter_id: ID of the adapter providing the history

        Returns:
            Dict with reconciliation results:
            - processed_count: number of messages processed
            - added_count: number of new messages added
            - edited_count: number of messages edited
            - deleted_count: number of messages deleted
            - gap_messages_added: number of gap markers added
            - gap_messages_removed: number of gap markers removed
        """
        results = {
            'processed_count': 0,
            'added_count': 0,
            'edited_count': 0,
            'deleted_count': 0,
            'gap_messages_added': 0,
            'gap_messages_removed': 0,
            'errors': []
        }

        if not history_messages:
            logger.info(f"[{self.owner.id}] No history messages to reconcile")
            return results

        # Step 1: Analyze overlap between history and existing messages
        overlap_analysis = self._detect_history_overlap(history_messages)
        overlap_type = overlap_analysis['overlap_type']

        logger.info(f"[{self.owner.id}] History reconciliation: {overlap_type}, {len(history_messages)} history messages")

        # Step 2: Handle different overlap scenarios
        if overlap_type == 'no_existing':
            # MessageList is empty - apply all history as-is
            logger.info(f"[{self.owner.id}] Empty MessageList - applying all {len(history_messages)} history messages")
            for message_dict in history_messages:
                if self._apply_history_message_as_new(message_dict, source_adapter_id):
                    results['added_count'] += 1
                else:
                    results['errors'].append(f"Failed to add history message: {message_dict.get('message_id', 'unknown')}")
                results['processed_count'] += 1

        elif overlap_type == 'no_history':
            # No history provided - nothing to reconcile
            logger.info(f"[{self.owner.id}] No history messages provided")

        elif overlap_type in ['no_overlap_earlier', 'no_overlap_later']:
            # No overlap - add gap message and process all history
            gap_info = overlap_analysis['gap_info']

            # Create and add gap message
            gap_message = self._create_system_gap_message(
                gap_info['gap_start'],
                gap_info['gap_end'],
                gap_info['gap_type']
            )
            self._add_message_to_list(gap_message)
            results['gap_messages_added'] = 1

            # Add all history messages
            for message_dict in history_messages:
                if self._apply_history_message_as_new(message_dict, source_adapter_id):
                    results['added_count'] += 1
                else:
                    results['errors'].append(f"Failed to add history message: {message_dict.get('message_id', 'unknown')}")
                results['processed_count'] += 1

        elif overlap_type == 'has_overlap':
            # Complex case - need to reconcile overlapping messages

            # Step 2a: Remove gap messages that are now filled
            history_range = overlap_analysis['history_range']
            removed_gaps = self._remove_existing_gap_messages(history_range['min'], history_range['max'])
            results['gap_messages_removed'] = removed_gaps

            # Step 2b: Create maps for efficient lookup
            existing_messages_by_external_id = {}
            for msg in self._state['_messages']:
                external_id = msg.get('original_external_id')
                if external_id:
                    existing_messages_by_external_id[external_id] = msg

            history_messages_by_external_id = {}
            for hist_msg in history_messages:
                external_id = hist_msg.get('message_id')
                if external_id:
                    history_messages_by_external_id[external_id] = hist_msg

            # Step 2c: Process history messages (add/edit)
            for message_dict in history_messages:
                external_id = message_dict.get('message_id')
                if not external_id:
                    # Skip messages without external ID - can't reconcile
                    results['errors'].append("Skipping history message without external ID")
                    results['processed_count'] += 1
                    continue

                existing_msg = existing_messages_by_external_id.get(external_id)
                if existing_msg:
                    # Message exists in both - check for edits
                    if self._should_apply_history_edit(existing_msg, message_dict):
                        if self._apply_history_edit(existing_msg, message_dict):
                            results['edited_count'] += 1
                        else:
                            results['errors'].append(f"Failed to edit message: {external_id}")
                else:
                    # Message in history but not in MessageList - add it
                    if self._apply_history_message_as_new(message_dict, source_adapter_id):
                        results['added_count'] += 1
                    else:
                        results['errors'].append(f"Failed to add history message: {external_id}")

                results['processed_count'] += 1

            # Step 2d: Find messages to delete (in MessageList but not in history, within overlap range)
            for msg in list(self._state['_messages']):  # Copy list to allow modification
                external_id = msg.get('original_external_id')
                if not external_id:
                    if self._apply_history_deletion(msg):
                        results['deleted_count'] += 1
                    else:
                        results['errors'].append(f"Failed to delete message: {external_id}")
                    continue

                msg_timestamp = msg.get('timestamp')
                if not msg_timestamp:
                    continue
                # Check if this message is within the history range but not in history
                if (history_range['min'] <= msg_timestamp <= history_range['max'] and
                    external_id not in history_messages_by_external_id):

                    # This message should be deleted (exists in MessageList but not in history)
                    if self._apply_history_deletion(msg):
                        results['deleted_count'] += 1
                    else:
                        results['errors'].append(f"Failed to delete message: {external_id}")

        # Step 3: Sort messages by timestamp to maintain chronological order
        self._sort_messages_by_timestamp()

        logger.info(f"[{self.owner.id}] History reconciliation complete: "
                   f"{results['added_count']} added, {results['edited_count']} edited, "
                   f"{results['deleted_count']} deleted, {results['gap_messages_added']} gaps added, "
                   f"{results['gap_messages_removed']} gaps removed, {len(results['errors'])} errors")

        return results

    def _apply_history_message_as_new(self, message_dict: Dict[str, Any], source_adapter_id: str) -> bool:
        """
        Apply a history message as a new message in the MessageList.

        Args:
            message_dict: History message dictionary
            source_adapter_id: Source adapter ID

        Returns:
            True if successful, False otherwise
        """
        try:
            sender_info = message_dict.get('sender', {})

            message_payload = {
                "source_adapter_id": source_adapter_id,
                "timestamp": message_dict.get("timestamp", time.time()),
                "sender_external_id": sender_info.get("user_id"),
                "sender_display_name": sender_info.get("display_name", "Unknown Sender"),
                "text": message_dict.get("text"),
                "original_message_id_external": message_dict.get("message_id"),
                "mentions": message_dict.get("mentions", []),
                "attachments": message_dict.get("attachments", [])
            }

            return self._handle_new_message(message_payload)

        except Exception as e:
            logger.error(f"[{self.owner.id}] Error applying history message as new: {e}", exc_info=True)
            return False

    def _should_apply_history_edit(self, existing_msg: Dict[str, Any], history_msg: Dict[str, Any]) -> bool:
        """
        Determine if a history message represents an edit that should be applied.

        Args:
            existing_msg: Existing message in MessageList
            history_msg: History message from adapter

        Returns:
            True if edit should be applied, False otherwise
        """
        # Check if history message indicates it was edited
        return (history_msg.get('edited', False) or history_msg.get('edit_timestamp') is not None)

    def _apply_history_edit(self, existing_msg: Dict[str, Any], history_msg: Dict[str, Any]) -> bool:
        """
        Apply an edit from history to an existing message.

        Args:
            existing_msg: Existing message in MessageList
            history_msg: History message with edit information

        Returns:
            True if successful, False otherwise
        """
        try:
            edit_content = {
                'original_message_id_external': history_msg.get('message_id'),
                'new_text': history_msg.get('text'),
                'timestamp': history_msg.get('edit_timestamp') or history_msg.get('timestamp', time.time())
            }

            logger.info(f"[{self.owner.id}] Applying history edit to message {history_msg.get('message_id')}")
            return self._handle_edit_message(edit_content)

        except Exception as e:
            logger.error(f"[{self.owner.id}] Error applying history edit: {e}", exc_info=True)
            return False

    def _apply_history_deletion(self, existing_msg: Dict[str, Any]) -> bool:
        """
        Apply a deletion for a message that exists in MessageList but not in history.

        Args:
            existing_msg: Message that should be deleted

        Returns:
            True if successful, False otherwise
        """
        try:
            delete_content = {
                'original_message_id_external': existing_msg.get('original_external_id'),
                'timestamp': time.time()
            }

            logger.info(f"[{self.owner.id}] Applying history deletion to message {existing_msg.get('original_external_id')}")
            return self._handle_delete_message(delete_content)

        except Exception as e:
            logger.error(f"[{self.owner.id}] Error applying history deletion: {e}", exc_info=True)
            return False

    def _add_message_to_list(self, message: Dict[str, Any]) -> bool:
        """
        Add a message directly to the MessageList (used for system messages).

        Args:
            message: Message dictionary to add

        Returns:
            True if successful, False otherwise
        """
        try:
            self._state['_messages'].append(message)
            internal_id = message.get('internal_id')
            if internal_id:
                self._state['_message_map'][internal_id] = len(self._state['_messages']) - 1
            return True
        except Exception as e:
            logger.error(f"[{self.owner.id}] Error adding message to list: {e}", exc_info=True)
            return False

    def _sort_messages_by_timestamp(self) -> None:
        """
        Sort all messages in the MessageList by timestamp to maintain chronological order.
        Rebuilds the message map after sorting.
        """
        try:
            self._state['_messages'].sort(key=lambda msg: msg.get('timestamp', 0))
            self._rebuild_message_map()
            logger.debug(f"[{self.owner.id}] Sorted {len(self._state['_messages'])} messages by timestamp")
        except Exception as e:
            logger.error(f"[{self.owner.id}] Error sorting messages by timestamp: {e}", exc_info=True)

    # --- NEW: Helper Methods for VEIL Operation Tracking ---
    def _record_veil_operation(self, operation_data: Dict[str, Any]) -> None:
        """Record operation for VEIL generation."""
        self._state['_pending_veil_operations'].append(operation_data)
        logger.debug(f"[{self.owner.id}] Recorded VEIL operation: {operation_data.get('operation_type')} for {operation_data.get('veil_id')}")

    def get_pending_veil_operations(self) -> List[Dict[str, Any]]:
        """Get and clear pending VEIL operations."""
        operations = list(self._state.get('_pending_veil_operations', []))
        self._state['_pending_veil_operations'].clear()
        return operations

    def _create_text_preview(self, text: str, max_length: int) -> Dict[str, Any]:
        """Create text preview with truncation info."""
        if not text:
            return {"preview": "", "truncated": False, "original_length": 0}

        if len(text) <= max_length:
            return {"preview": text, "truncated": False, "original_length": len(text)}
        else:
            truncated_count = len(text) - max_length
            return {
                "preview": text[:max_length],
                "truncated": True,
                "original_length": len(text),
                "truncated_count": truncated_count
            }

    def _is_message_from_current_agent(self, message_content: Dict[str, Any]) -> bool:
        """
        Determine if a message was sent by the current agent.

        This is crucial for historical message processing to identify agent messages
        that need synthetic agent_response facets for proper turn structure.

        Args:
            message_content: Message content from adapter/history

        Returns:
            True if message was sent by current agent, False otherwise
        """
        try:
            # Get agent information from parent space
            parent_space = self.owner.get_parent_object() if hasattr(self.owner, 'get_parent_object') else None
            if not parent_space:
                logger.debug(f"[{self.owner.id}] No parent space found - cannot determine agent identity")
                return False

            agent_name = getattr(parent_space, 'agent_name', None)
            agent_external_id = getattr(parent_space, 'agent_external_id', None)  # If available
            alias = getattr(self.owner, 'alias', None)

            # Get sender information from message
            sender_name = message_content.get('sender_display_name', '')
            sender_id = message_content.get('sender_external_id', '')

            # Method 1: Compare sender_id with agent_external_id (most reliable)
            if agent_external_id and sender_id:
                if sender_id == agent_external_id:
                    logger.debug(f"[{self.owner.id}] Message from current agent (matched external ID: {sender_id})")
                    return True

            # Method 2: Compare sender_name with agent_name (fallback)
            if agent_name and sender_name:
                if sender_name == agent_name:
                    logger.debug(f"[{self.owner.id}] Message from current agent (matched name: {sender_name})")
                    return True

            if alias and sender_name:
                if sender_name == alias:
                    logger.debug(f"[{self.owner.id}] Message from current agent (matched alias: {sender_name})")
                    return True

            # Method 3: Check for agent-specific markers in message metadata
            # Some adapters might include additional metadata about message source
            message_source = message_content.get('message_source')
            if message_source == "agent_outgoing":
                logger.debug(f"[{self.owner.id}] Message from current agent (marked as agent_outgoing)")
                return True

            # Not from current agent
            logger.debug(f"[{self.owner.id}] Message NOT from current agent (sender: {sender_name}, agent: {agent_name})")
            return False

        except Exception as e:
            logger.error(f"[{self.owner.id}] Error determining if message is from current agent: {e}", exc_info=True)
            # Safe default: assume not from agent to avoid false positives
            return False

    def _get_conversation_metadata(self) -> Dict[str, Any]:
        """Get conversation metadata from the owner element."""
        metadata = {}
        if self.owner:
            metadata.update({
                "adapter_type": getattr(self.owner, 'adapter_type', None),
                "server_name": getattr(self.owner, 'server_name', None),
                "conversation_name": getattr(self.owner, 'conversation_name', None),
                "adapter_id": getattr(self.owner, 'adapter_id', None),
                "external_conversation_id": getattr(self.owner, 'external_conversation_id', None),
                "alias": getattr(self.owner, 'alias', None)
            })
        return metadata
