"""
Message List Component
Manages the state of a list of messages within an Element.
Acts as a cache/materialized view based on events from the TimelineComponent's DAG.
"""
import logging
import time
from typing import Dict, Any, Optional, List

from ...base import Component

logger = logging.getLogger(__name__)

# Define the structure of a message stored in the component's state
# This can be expanded later (e.g., with reactions, read_status, edits)
MessageType = Dict[str, Any] 

class MessageListComponent(Component):
    """
    Maintains an ordered list of messages based on events recorded in the timeline.
    """
    COMPONENT_TYPE = "MessageListComponent"

    # Events this component reacts to
    HANDLED_EVENT_TYPES = [
        "connectome_internal_dm",             # Handled by _handle_new_message
        "connectome_internal_channel_message", # Handled by _handle_new_message
        "message_deleted",                    # Handled by _handle_delete_message
        "message_edited",                     # Handled by _handle_edit_message
        # Add other relevant event types like "reaction_added", "message_pinned" later
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
        self._max_messages = max_messages
        logger.debug(f"MessageListComponent initialized for Element {self.owner.id}. Max messages: {max_messages}")

    def handle_event(self, event_node: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Processes relevant timeline events to update the message list.
        Expects `event_node['payload']` to contain the actual event details.
        """
        event_payload = event_node.get('payload', {})
        event_type = event_payload.get('event_type')
        target_element = event_payload.get('target_element_id')

        # Ensure the event is targeted at the element this component is attached to
        if target_element != self.owner.id:
            return False # Not for us

        if event_type in self.HANDLED_EVENT_TYPES:
            logger.debug(f"[{self.owner.id}] MessageListComponent handling event: {event_type}")
            if event_type in ["connectome_internal_dm", "connectome_internal_channel_message"]:
                return self._handle_new_message(event_payload.get('payload', {}))
            elif event_type == "message_deleted":
                return self._handle_delete_message(event_payload.get('payload', {}))
            elif event_type == "message_edited":
                return self._handle_edit_message(event_payload.get('payload', {}))
            # Add handlers for other types here
            else:
                logger.warning(f"[{self.owner.id}] No specific handler implemented for event type '{event_type}' in MessageListComponent.")
        
        return False # Event type not handled by this component

    def _handle_new_message(self, message_data: Dict[str, Any]) -> bool:
        """Adds a new message to the list."""
        # Extract relevant fields from the message_data (which is the payload of the connectome event)
        internal_message_id = f"msg_{self.owner.id}_{int(time.time()*1000)}_{len(self._state['_messages'])}" # Generate unique ID within this list
        new_message: MessageType = {
            'internal_id': internal_message_id,
            'timestamp': message_data.get('timestamp', time.time()),
            'sender_id': message_data.get('sender_external_id'), # Or sender_connectome_id if available
            'sender_name': message_data.get('sender_display_name', 'Unknown Sender'),
            'text': message_data.get('text'),
            'original_external_id': message_data.get('original_message_id_external'), # For correlation
            'adapter_id': message_data.get('source_adapter_id'),
            # Add more fields as needed: reactions, read_status, attachments, is_edited etc.
            'is_edited': False,
            'reactions': {},
            'read_by': []
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
            self._rebuild_message_map()
            logger.debug(f"[{self.owner.id}] Pruned oldest message due to max_messages limit.")
            
        logger.info(f"[{self.owner.id}] New message added. Total messages: {len(self._state['_messages'])}")
        # TODO: Could this component emit a local event like "message_list_updated"?
        return True

    def _handle_delete_message(self, delete_data: Dict[str, Any]) -> bool:
        """
        Removes a message based on its ID.
        Requires an event payload containing message identifiers.
        """
        # We need a reliable way to identify the message to delete.
        # Option 1: Use original_external_id if the delete event provides it.
        # Option 2: Use an internal ID if the original event returned one that was stored.
        original_external_id = delete_data.get('original_message_id_external')
        internal_id_to_delete = None
        
        if original_external_id:
            # Find internal message by external ID (requires iterating - slow)
            found_idx = -1
            for idx, msg in enumerate(self._state['_messages']):
                if msg.get('original_external_id') == original_external_id:
                    internal_id_to_delete = msg.get('internal_id')
                    found_idx = idx
                    break
            if found_idx != -1:
                del self._state['_messages'][found_idx]
                if internal_id_to_delete: del self._state['_message_map'][internal_id_to_delete]
                self._rebuild_message_map() # Inefficient!
                logger.info(f"[{self.owner.id}] Message with external ID '{original_external_id}' deleted.")
                return True
            else:
                logger.warning(f"[{self.owner.id}] Could not delete message: External ID '{original_external_id}' not found.")
                return False
        else:
            # TODO: Implement deletion by internal ID if available in delete_data
            logger.warning(f"[{self.owner.id}] Message deletion event lacked necessary identifier. Payload: {delete_data}")
            return False

    def _handle_edit_message(self, edit_data: Dict[str, Any]) -> bool:
        """
        Updates the content of an existing message.
        Requires message identifier and new text in payload.
        """
        original_external_id = edit_data.get('original_message_id_external')
        new_text = edit_data.get('new_text')
        
        if not original_external_id or new_text is None:
             logger.warning(f"[{self.owner.id}] Message edit event lacked necessary identifier or new_text. Payload: {edit_data}")
             return False
             
        # Find message by external ID (inefficient)
        message_to_edit = None
        for msg in self._state['_messages']:
            if msg.get('original_external_id') == original_external_id:
                message_to_edit = msg
                break
        
        if message_to_edit:
            message_to_edit['text'] = new_text
            message_to_edit['is_edited'] = True
            message_to_edit['last_edited_timestamp'] = edit_data.get('edit_timestamp', time.time())
            logger.info(f"[{self.owner.id}] Message with external ID '{original_external_id}' edited.")
            # TODO: Emit event?
            return True
        else:
            logger.warning(f"[{self.owner.id}] Could not edit message: External ID '{original_external_id}' not found.")
            return False

    def _rebuild_message_map(self):
        """Inefficiently rebuilds the internal ID to index map."""
        self._state['_message_map'] = {msg['internal_id']: idx for idx, msg in enumerate(self._state['_messages']) if 'internal_id' in msg}

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
        
    # --- Other potential methods ---
    # def mark_message_read(self, internal_id: str, user_id: str): ...
    # def add_reaction(self, internal_id: str, reaction: str, user_id: str): ...
    # def remove_reaction(self, internal_id: str, reaction: str, user_id: str): ...
