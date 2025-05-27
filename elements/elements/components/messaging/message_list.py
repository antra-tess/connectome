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
        "connectome_message_deleted",         # Use Connectome-defined types for delete/edit
        "connectome_message_updated",         # Use Connectome-defined types for delete/edit
        "connectome_reaction_added",          # For handling added reactions
        "connectome_reaction_removed",        # For handling removed reactions
        "attachment_content_available",       # NEW: For when fetched attachment content arrives
        CONNECTOME_MESSAGE_SEND_CONFIRMED,    # NEW: For outgoing message success
        CONNECTOME_MESSAGE_SEND_FAILED        # NEW: For outgoing message failure
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
        event_payload = event_node.get('payload', {}) # This is the inner payload
        event_type = event_payload.get('event_type')

        if event_type in self.HANDLED_EVENT_TYPES:
            logger.debug(f"[{self.owner.id}] MessageListComponent handling event: {event_type}")
            # Message handlers now expect the *actual content* payload, 
            # which is event_payload['payload'] if event_payload itself has a nested structure,
            # or just event_payload if it's flat.
            # For "message_received" from a Space, event_payload['payload'] is the adapter_data.
            
            actual_content_payload = event_payload.get('payload', event_payload) # Default to event_payload if no deeper 'payload' key
            if event_type == "message_received":
                self._handle_new_message(actual_content_payload)
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
            elif event_type == CONNECTOME_MESSAGE_SEND_CONFIRMED: # NEW
                self._handle_message_send_confirmed(actual_content_payload)
            elif event_type == CONNECTOME_MESSAGE_SEND_FAILED: # NEW
                self._handle_message_send_failed(actual_content_payload)
            else:
                logger.warning(f"[{self.owner.id}] No specific handler implemented for event type '{event_type}' in MessageListComponent.")
                return False # Event type not handled by this component
            veil_producer = self.get_sibling_component("MessageListVeilProducer")
            veil_producer.emit_delta()

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
            'error_details': None # Not applicable for incoming
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
        # TODO: Could this component emit a local event like "message_list_updated"?
        return True

    def _handle_delete_message(self, delete_content: Dict[str, Any]) -> bool:
        """
        Removes a message based on its ID.
        delete_content is the actual data for deletion (e.g., from event_payload['payload']).
        """
        original_external_id = delete_content.get('original_message_id_external')
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
            logger.warning(f"[{self.owner.id}] Message deletion event lacked necessary identifier. Payload: {delete_content}")
            return False

    def _handle_edit_message(self, edit_content: Dict[str, Any]) -> bool:
        """
        Updates the content of an existing message.
        edit_content is the actual data for edit (e.g., from event_payload['payload']).
        """
        original_external_id = edit_content.get('original_message_id_external')
        new_text = edit_content.get('new_text')
        edit_timestamp = edit_content.get('timestamp', time.time())
        
        if not original_external_id or new_text is None:
             logger.warning(f"[{self.owner.id}] Message edit event lacked necessary identifier or new_text. Payload: {edit_content}")
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
            message_to_edit['last_edited_timestamp'] = edit_timestamp
            logger.info(f"[{self.owner.id}] Message with external ID '{original_external_id}' edited.")
            # TODO: Emit event?
            return True
        else:
            logger.warning(f"[{self.owner.id}] Could not edit message: External ID '{original_external_id}' not found.")
            return False

    def _handle_reaction_added(self, reaction_content: Dict[str, Any]) -> bool:
        """
        Adds a reaction to an existing message.
        reaction_content is the actual data for the reaction (e.g., from event_payload['payload']).
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
        
        if message_to_update:
            if 'reactions' not in message_to_update:
                message_to_update['reactions'] = {}
            
            if emoji not in message_to_update['reactions']:
                message_to_update['reactions'][emoji] = []
            
            # Add user to list if not already present (some platforms might send redundant events)
            if user_id and user_id not in message_to_update['reactions'][emoji]:
                message_to_update['reactions'][emoji].append(user_id)
            elif not user_id:
                # If user_id is not provided by the adapter, we might just increment a counter
                # or store a generic reaction without specific user attribution.
                # For now, let's assume a reaction implies one user if ID is missing.
                # This could be a list of anonymous markers, or just a count if we change structure.
                message_to_update['reactions'][emoji].append("anonymous_reaction") 
                logger.debug(f"[{self.owner.id}] Added anonymous reaction '{emoji}' to message {original_external_id}")

            logger.info(f"[{self.owner.id}] Reaction '{emoji}' by '{user_name if user_id else 'anonymous'}' added to message '{original_external_id}'.")
            # TODO: Emit event?
            return True
        else:
            logger.warning(f"[{self.owner.id}] Could not add reaction: Message with external ID '{original_external_id}' not found.")
            return False

    def _handle_reaction_removed(self, reaction_content: Dict[str, Any]) -> bool:
        """
        Removes a reaction from an existing message.
        reaction_content is the actual data for reaction removal (e.g., from event_payload['payload']).
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

        if message_to_update and 'reactions' in message_to_update and emoji in message_to_update['reactions']:
            if user_id:
                if user_id in message_to_update['reactions'][emoji]:
                    message_to_update['reactions'][emoji].remove(user_id)
                    logger.info(f"[{self.owner.id}] Reaction '{emoji}' by '{user_name}' removed from message '{original_external_id}'.")
                else:
                    logger.debug(f"[{self.owner.id}] User '{user_name}' did not have reaction '{emoji}' on message '{original_external_id}' to remove.")
                    return False # User hadn't reacted with this emoji
            else:
                # If user_id is None, try to remove a generic "anonymous_reaction" marker if present
                if "anonymous_reaction" in message_to_update['reactions'][emoji]:
                    message_to_update['reactions'][emoji].remove("anonymous_reaction")
                    logger.info(f"[{self.owner.id}] Anonymous reaction '{emoji}' removed from message '{original_external_id}'.")
                else:
                    logger.debug(f"[{self.owner.id}] No anonymous reaction '{emoji}' on message '{original_external_id}' to remove.")
                    return False # No anonymous reaction to remove

            # If the list for this emoji is now empty, remove the emoji key itself
            if not message_to_update['reactions'][emoji]:
                del message_to_update['reactions'][emoji]
                logger.debug(f"[{self.owner.id}] Emoji '{emoji}' removed from message '{original_external_id}' as no users left.")
            
            # TODO: Emit event?
            return True
        else:
            logger.warning(f"[{self.owner.id}] Could not remove reaction: Message '{original_external_id}' not found or no such reaction '{emoji}'.")
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
                            adapter_id: Optional[str] = None # The adapter this message is going to
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
            'error_details': None
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
        confirm_content is expected to contain:
        - internal_request_id: str
        - external_message_ids: List[str] (IDs assigned by the external platform)
        - confirmed_timestamp: Optional[float]
        """
        internal_req_id = confirm_content.get('internal_request_id')
        external_ids = confirm_content.get('external_message_ids')

        if not internal_req_id or not external_ids:
            logger.warning(f"[{self.owner.id}] Message send confirmation missing 'internal_request_id' or 'external_message_ids'. Payload: {confirm_content}")
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
        for msg in self._state['_messages']:
            if msg.get('internal_request_id') == internal_req_id and msg.get('status') == "pending_send":
                message_to_update = msg
                break
        
        if message_to_update:
            message_to_update['status'] = "failed_to_send"
            message_to_update['error_details'] = error_msg or "Unknown send failure"
            if 'failed_timestamp' in failure_content and failure_content['failed_timestamp']:
                message_to_update['timestamp'] = failure_content['failed_timestamp'] # Update to failure time
            logger.info(f"[{self.owner.id}] Pending message (req_id: {internal_req_id}) marked as failed_to_send. Error: {error_msg}")
            return True
        else:
            logger.warning(f"[{self.owner.id}] Could not find 'pending_send' message with internal_request_id '{internal_req_id}' to mark as failed.")
            return False

    # --- Other potential methods ---
    # def mark_message_read(self, internal_id: str, user_id: str): ...
    # def add_reaction(self, internal_id: str, reaction: str, user_id: str): ...
    # def remove_reaction(self, internal_id: str, reaction: str, user_id: str): ...
