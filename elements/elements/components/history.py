import logging
from typing import List, Dict, Any, Optional, Tuple
import time

from .base import Component
from ..base import BaseElement # Corrected import path assuming BaseElement is in ..base

logger = logging.getLogger(__name__)

# Define expected event types for clarity
EVENT_MESSAGE_RECEIVED = "message_received"
EVENT_AGENT_MESSAGE_SENT = "agent_message_sent" # Assuming this is the event type for agent sends
EVENT_MESSAGE_EDITED = "message_edited"
EVENT_MESSAGE_DELETED = "message_deleted"
EVENT_REACTION_ADDED = "reaction_added"
EVENT_REACTION_REMOVED = "reaction_removed"
EVENT_SYSTEM_NOTIFICATION = "system_notification"
EVENT_TOOL_RESULT_AVAILABLE = "tool_result_available" # Added
# Add other relevant types if needed, e.g., system notifications

HISTORY_EVENT_TYPES = {
    EVENT_MESSAGE_RECEIVED,
    EVENT_AGENT_MESSAGE_SENT,
    EVENT_MESSAGE_EDITED,
    EVENT_MESSAGE_DELETED,
    EVENT_REACTION_ADDED,
    EVENT_REACTION_REMOVED,
    EVENT_SYSTEM_NOTIFICATION,
    EVENT_TOOL_RESULT_AVAILABLE, # Added
}

class HistoryComponent(Component):
    """
    Manages the history of messages and related events for an element,
    partitioning history by conversation_id and supporting edits,
    deletions, and reactions within each partition.
    Tracks the most recently active conversation.
    """

    COMPONENT_TYPE = "history"
    # No direct component dependencies needed for internal logic,
    # but relies on event routing to receive relevant events.
    DEPENDENCIES = set() 

    def __init__(self, element: 'BaseElement', max_history_size_per_conversation: Optional[int] = 1000):
        super().__init__(element)
        # Partitioned history: {conversation_id: {message_id: message_entry}}
        self._history: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Partitioned timestamps: {conversation_id: {message_id: original_timestamp}}
        self._entry_timestamps: Dict[str, Dict[str, int]] = {}
        # Track most recently active conversation
        self._active_conversation_id: Optional[str] = None
        self.max_history_size_per_conversation = max_history_size_per_conversation
        logger.debug(f"HistoryComponent initialized for element {element.id}")

    def handle_event(self, event: Dict[str, Any], timeline_context: Optional[Dict[str, Any]] = None) -> bool:
        """Handles incoming events, routing them to the correct conversation partition."""
        event_type = event.get("event_type")
        payload = event.get("payload", {})
        
        # --- Get Conversation ID - Crucial for Partitioning ---
        # Events like system notifications might not have one.
        conversation_id = payload.get("conversation_id")
        adapter_id = payload.get("adapter_id") # Needed for context sometimes
        message_id = payload.get("message_id")

        if event_type not in HISTORY_EVENT_TYPES:
            return False # Event type not relevant

        # If it's a message-related event, conversation_id is mandatory
        is_message_event = event_type in {EVENT_MESSAGE_RECEIVED, EVENT_AGENT_MESSAGE_SENT, EVENT_MESSAGE_EDITED, EVENT_MESSAGE_DELETED, EVENT_REACTION_ADDED, EVENT_REACTION_REMOVED}
        if is_message_event and not conversation_id:
             logger.warning(f"Skipping {event_type} event for {self.element.id}: missing 'conversation_id' in payload.")
             return False
             
        # --- Update Active Conversation --- 
        if conversation_id:
            # Update active conversation if this event pertains to one
            if self._active_conversation_id != conversation_id:
                 logger.debug(f"[{self.element.id}] Setting active conversation to: {conversation_id}")
                 self._active_conversation_id = conversation_id
            
            # --- Ensure Partition Exists --- 
            if conversation_id not in self._history:
                 logger.info(f"[{self.element.id}] Creating new history partition for conversation_id: {conversation_id}")
                 self._history[conversation_id] = {}
                 self._entry_timestamps[conversation_id] = {}
        elif event_type == EVENT_SYSTEM_NOTIFICATION:
             # Handle system notifications (existing logic)
             return True
        elif event_type == EVENT_TOOL_RESULT_AVAILABLE:
             # Tool results might not always have a direct conversation_id in their payload
             # depending on how the tool execution context is managed. 
             # Let's assume for now it *does* have conversation_id if it's relevant to a specific chat.
             # If not, we might need a different way to store/route tool results.
             if not conversation_id:
                  logger.warning(f"Skipping {event_type} for {self.element.id}: 'conversation_id' missing. Cannot store result in history partition.")
                  return False
             # Ensure partition exists if we have a conversation_id
             if conversation_id not in self._history:
                  logger.info(f"[{self.element.id}] Creating new history partition for conversation_id: {conversation_id} for tool result.")
                  self._history[conversation_id] = {}
                  self._entry_timestamps[conversation_id] = {}
        else:
             # Other event types might need conversation_id later
             logger.warning(f"Skipping {event_type} event for {self.element.id}: 'conversation_id' is required but missing.")
             return False
        # --- End Partitioning Logic ---
        
        # Use tool_call_id as the message_id for tool results
        message_id = payload.get("message_id") or payload.get("tool_call_id")

        logger.debug(f"HistoryComponent on {self.element.id} handling event: {event_type} for Conv: {conversation_id}, ID: {message_id}")
        current_time_ms = int(time.time() * 1000)

        # --- Handle different event types within the partition ---
        if event_type == EVENT_MESSAGE_RECEIVED or event_type == EVENT_AGENT_MESSAGE_SENT:
            if not message_id:
                 logger.warning(f"Skipping {event_type} for Conv '{conversation_id}': missing 'message_id'.")
                 return False
            self._add_new_message(conversation_id, event, message_id, payload, timeline_context, current_time_ms)
            self._enforce_history_limit(conversation_id)
            return True

        elif event_type == EVENT_MESSAGE_EDITED:
            if not message_id:
                 logger.warning(f"Skipping {event_type} for Conv '{conversation_id}': missing 'message_id'.")
                 return False
            return self._edit_message(conversation_id, message_id, payload, current_time_ms)

        elif event_type == EVENT_MESSAGE_DELETED:
            if not message_id:
                 logger.warning(f"Skipping {event_type} for Conv '{conversation_id}': missing 'message_id'.")
                 return False
            return self._delete_message(conversation_id, message_id, current_time_ms)

        elif event_type == EVENT_REACTION_ADDED or event_type == EVENT_REACTION_REMOVED:
            if not message_id or not payload.get("emoji"):
                logger.warning(f"Skipping {event_type} for Conv '{conversation_id}': missing 'message_id' or 'emoji'.")
                return False
            is_add = event_type == EVENT_REACTION_ADDED
            return self._update_reaction(conversation_id, message_id, payload.get("emoji"), is_add, payload.get("user_id"))

        elif event_type == EVENT_TOOL_RESULT_AVAILABLE:
            # Store the tool result
            tool_call_id = payload.get("tool_call_id")
            tool_result = payload.get("result") # Assuming result is in payload
            tool_name = payload.get("tool_name", "unknown_tool") # Get tool name if available
            
            if not tool_call_id:
                 logger.warning(f"Skipping {event_type} for Conv '{conversation_id}': missing 'tool_call_id'.")
                 return False
            if tool_result is None:
                 logger.warning(f"Skipping {event_type} for Conv '{conversation_id}' (ToolCall ID: {tool_call_id}): missing 'result'.")
                 # Store an error maybe? Or just skip?
                 # Let's store an entry indicating missing result.
                 tool_result = {"error": "Result missing in event payload"}
                 
            self._add_tool_result(conversation_id, tool_call_id, tool_name, tool_result, timeline_context, current_time_ms)
            # Decide whether to enforce history limit on tool results? Yes, treat like other entries.
            self._enforce_history_limit(conversation_id)
            return True

        return False # Fallback

    def _add_new_message(self, conversation_id: str, event: Dict[str, Any], message_id: str, payload: Dict[str, Any], timeline_context: Optional[Dict[str, Any]], timestamp_ms: int):
        """Adds a new message entry to the specific conversation's history."""
        conv_history = self._history[conversation_id]
        conv_timestamps = self._entry_timestamps[conversation_id]
        
        if message_id in conv_history:
             logger.warning(f"[{self.element.id}/Conv:{conversation_id}] Duplicate message_id '{message_id}' received. Overwriting.")

        entry = {
            "message_id": message_id,
            "conversation_id": conversation_id, # Store for convenience
            "event_id": event.get("event_id"),
            "event_type": event.get("event_type"),
            "timestamp": payload.get("timestamp", timestamp_ms),
            "timeline_id": timeline_context.get("timeline_id") if timeline_context else None,
            "data": payload,
            "deleted": False,
            "edited_timestamp": None,
            "deleted_timestamp": None,
            "reactions": {}
        }
        conv_history[message_id] = entry
        conv_timestamps[message_id] = entry["timestamp"]
        logger.debug(f"Added/Updated message '{message_id}' in history for Conv:{conversation_id}.")

    def _edit_message(self, conversation_id: str, message_id: str, payload: Dict[str, Any], timestamp_ms: int) -> bool:
        """Updates an existing message entry within a conversation partition."""
        entry = self._history.get(conversation_id, {}).get(message_id)
        if not entry:
            logger.warning(f"Cannot edit message '{message_id}' for Conv '{conversation_id}': Not found.")
            return False
        new_text = payload.get("text")
        if new_text is None:
            logger.warning(f"Edit event for message '{message_id}' / Conv '{conversation_id}' missing 'text'.")
            return False
        entry["data"]["text"] = new_text
        entry["edited_timestamp"] = timestamp_ms
        logger.debug(f"Edited message '{message_id}' in history for Conv:{conversation_id}.")
        return True

    def _delete_message(self, conversation_id: str, message_id: str, timestamp_ms: int) -> bool:
        """Marks an existing message entry as deleted within a conversation partition."""
        entry = self._history.get(conversation_id, {}).get(message_id)
        if not entry:
            logger.warning(f"Cannot delete message '{message_id}' for Conv '{conversation_id}': Not found.")
            return False
        entry["deleted"] = True
        entry["deleted_timestamp"] = timestamp_ms
        logger.debug(f"Marked message '{message_id}' as deleted in history for Conv:{conversation_id}.")
        return True

    def _update_reaction(self, conversation_id: str, message_id: str, emoji: str, is_add: bool, user_id: Optional[str] = None) -> bool:
        """Adds or removes a reaction count for a message within a conversation partition."""
        entry = self._history.get(conversation_id, {}).get(message_id)
        if not entry:
            logger.warning(f"Cannot update reaction on message '{message_id}' for Conv '{conversation_id}': Not found.")
            return False
        reactions = entry.setdefault("reactions", {})
        current_count = reactions.get(emoji, 0)
        if is_add:
            reactions[emoji] = current_count + 1
            logger.debug(f"Added reaction '{emoji}' to msg '{message_id}' (Conv:{conversation_id}). New: {reactions[emoji]}. User: {user_id}")
        else:
            if current_count > 0:
                reactions[emoji] = current_count - 1
                if reactions[emoji] == 0: del reactions[emoji]
                logger.debug(f"Removed reaction '{emoji}' from msg '{message_id}' (Conv:{conversation_id}). New: {reactions.get(emoji, 0)}. User: {user_id}")
            else:
                logger.warning(f"Cannot remove reaction '{emoji}' from msg '{message_id}' (Conv:{conversation_id}): Not present.")
        return True

    def _add_tool_result(self, conversation_id: str, tool_call_id: str, tool_name: str, result: Any, timeline_context: Optional[Dict[str, Any]], timestamp_ms: int):
        """Adds a tool result entry to the specific conversation's history."""
        conv_history = self._history[conversation_id]
        conv_timestamps = self._entry_timestamps[conversation_id]

        # Use tool_call_id as the unique key for this history entry
        message_id = tool_call_id 
        
        if message_id in conv_history:
             # This might happen if a tool result is somehow processed twice. 
             # Overwriting might be okay, or log a more severe warning.
             logger.warning(f"[{self.element.id}/Conv:{conversation_id}] Duplicate tool_call_id '{message_id}' received as history entry key. Overwriting.")

        entry = {
            # Treat tool_call_id as the message_id for history purposes
            "message_id": message_id, 
            "conversation_id": conversation_id,
            # Use a specific event type if desired, or keep the incoming one
            "event_type": EVENT_TOOL_RESULT_AVAILABLE, 
            "timestamp": timestamp_ms, # Use processing time
            "timeline_id": timeline_context.get("timeline_id") if timeline_context else None,
            # Structure the data payload for tool results clearly
            "data": { 
                "role": "tool", # Standard role for tool results
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "content": result # Store the actual result here
            },
            # Metadata fields (not typically edited/deleted/reacted to)
            "deleted": False,
            "edited_timestamp": None,
            "deleted_timestamp": None,
            "reactions": {}
        }
        conv_history[message_id] = entry
        conv_timestamps[message_id] = entry["timestamp"]
        logger.debug(f"Added tool result '{tool_call_id}' to history for Conv:{conversation_id}.")

    def _enforce_history_limit(self, conversation_id: str):
        """Removes the oldest entries for a specific conversation if it exceeds the limit."""
        if self.max_history_size_per_conversation is None:
            return
            
        conv_history = self._history.get(conversation_id)
        conv_timestamps = self._entry_timestamps.get(conversation_id)
        if not conv_history or not conv_timestamps:
             return # Should not happen if called after adding a message
             
        current_size = len(conv_history)
        if current_size <= self.max_history_size_per_conversation:
            return

        num_to_remove = current_size - self.max_history_size_per_conversation
        if num_to_remove <= 0: return

        sorted_ids_by_time: List[Tuple[str, int]] = sorted(conv_timestamps.items(), key=lambda item: item[1])
        ids_to_remove = [msg_id for msg_id, ts in sorted_ids_by_time[:num_to_remove]]

        for msg_id in ids_to_remove:
            del conv_history[msg_id]
            del conv_timestamps[msg_id]
            logger.debug(f"Pruned old history entry '{msg_id}' from Conv:{conversation_id} due to size limit.")

    # --- Public Accessor Methods ---
    
    def get_active_conversation_id(self) -> Optional[str]:
         """Returns the ID of the most recently interacted-with conversation."""
         return self._active_conversation_id

    def get_history(self, conversation_id: str, include_deleted=False, sort_by_timestamp=True) -> List[Dict[str, Any]]:
        """
        Returns the history for a specific conversation partition.
        Can optionally filter out deleted messages and sort by timestamp.
        Returns an empty list if the conversation_id is unknown.
        """
        conv_history = self._history.get(conversation_id)
        if not conv_history:
            return []
            
        entries = list(conv_history.values())
        if not include_deleted:
            entries = [entry for entry in entries if not entry.get("deleted")]
        if sort_by_timestamp:
            entries.sort(key=lambda x: x.get("timestamp", 0))
        return entries

    def get_message_by_id(self, conversation_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific message entry by its conversation and message ID."""
        return self._history.get(conversation_id, {}).get(message_id)

    def clear_history(self, conversation_id: Optional[str] = None):
        """Clears the history for a specific conversation or all conversations."""
        if conversation_id:
            if conversation_id in self._history:
                self._history.pop(conversation_id, None)
                self._entry_timestamps.pop(conversation_id, None)
                logger.info(f"History cleared for conversation {conversation_id} on element {self.element.id}")
                if self._active_conversation_id == conversation_id:
                     self._active_conversation_id = None # Reset if active was cleared
            else:
                 logger.warning(f"Cannot clear history for unknown conversation_id: {conversation_id}")
        else:
            self._history = {}
            self._entry_timestamps = {}
            self._active_conversation_id = None
            logger.info(f"All history partitions cleared for element {self.element.id}")

    def list_conversations(self) -> List[str]:
         """Returns a list of conversation IDs currently tracked in history."""
         return list(self._history.keys())

    # Optional: Methods to get specific types of history or search history
    # Might need adjustments based on the new structure