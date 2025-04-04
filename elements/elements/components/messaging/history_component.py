"""
Messaging History Component
Component for managing message history for ChatElement.
"""

import logging
from typing import Dict, Any, Optional, List
import uuid
import time

from ..base_component import Component
from ..space.timeline_component import TimelineComponent # Needed for timeline interaction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HistoryComponent(Component):
    """
    Manages the message history for a ChatElement, respecting timelines.
    """
    
    COMPONENT_TYPE: str = "messaging_history"
    DEPENDENCIES: List[str] = ["timeline"] # Depends on the element having a TimelineComponent
    
    # Events this component handles to update history
    HANDLED_EVENT_TYPES: List[str] = [
        "message_received", 
        "message_sent", 
        "message_updated", 
        "message_deleted",
        "clear_context" # Might clear history for a timeline
    ]
    
    # Messages starting with these markers are often ignored
    IGNORE_MARKERS = ["."]
    
    def __init__(self, element=None):
        super().__init__(element)
        # State stores messages per timeline: { timeline_id: { message_id: message_data } }
        self._state = {
            "messages": {} 
        }
    
    def _get_timeline_comp(self) -> Optional[TimelineComponent]:
        """Helper to get the associated TimelineComponent."""
        if not self.element:
            return None
        return self.element.get_component_by_type("timeline")

    def _ensure_timeline_history(self, timeline_id: str):
        """Ensures the messages dictionary exists for a given timeline."""
        if timeline_id not in self._state["messages"]:
            self._state["messages"][timeline_id] = {}
            
    def add_message(self, message_data: Dict[str, Any], timeline_id: str) -> bool:
        """
        Adds a message to the history for a specific timeline.
        
        Args:
            message_data: The message content and metadata.
            timeline_id: The ID of the timeline to add the message to.
            
        Returns:
            True if added successfully, False otherwise.
        """
        if not self._is_initialized or not self._is_enabled:
            return False
            
        # Basic validation
        message_id = message_data.get("message_id")
        if not message_id:
            message_id = f"msg_{uuid.uuid4().hex[:8]}"
            message_data["message_id"] = message_id
            
        if "timestamp" not in message_data:
            message_data["timestamp"] = int(time.time() * 1000)
        
        # Ensure history structure exists
        self._ensure_timeline_history(timeline_id)
        
        # Check for ignore markers (simple check)
        text = message_data.get("text", "")
        if any(text.startswith(marker) for marker in self.IGNORE_MARKERS):
            logger.debug(f"Ignoring message {message_id} due to ignore marker.")
            # Decide if ignored messages should still be stored or not.
            # For now, let's store them but maybe flag them.
            message_data["ignored"] = True 
            
        # Store the message
        self._state["messages"][timeline_id][message_id] = message_data
        logger.debug(f"Added message {message_id} to history for timeline {timeline_id}")
        return True
        
    def update_message(self, message_id: str, update_data: Dict[str, Any], timeline_id: str) -> bool:
        """
        Updates an existing message in a specific timeline.
        
        Args:
            message_id: The ID of the message to update.
            update_data: Dictionary containing updates.
            timeline_id: The ID of the timeline where the message exists.
            
        Returns:
            True if updated successfully, False otherwise.
        """
        if not self._is_initialized or not self._is_enabled:
            return False
            
        if timeline_id not in self._state["messages"] or message_id not in self._state["messages"][timeline_id]:
            logger.warning(f"Message {message_id} not found in timeline {timeline_id} for update.")
            return False
            
        self._state["messages"][timeline_id][message_id].update(update_data)
        # Ensure timestamp reflects update time
        self._state["messages"][timeline_id][message_id]["last_updated"] = int(time.time() * 1000)
        logger.debug(f"Updated message {message_id} in timeline {timeline_id}")
        return True
        
    def delete_message(self, message_id: str, timeline_id: str) -> bool:
        """
        Deletes a message from a specific timeline.
        
        Args:
            message_id: The ID of the message to delete.
            timeline_id: The ID of the timeline where the message exists.
            
        Returns:
            True if deleted successfully, False otherwise.
        """
        if not self._is_initialized or not self._is_enabled:
            return False
            
        if timeline_id not in self._state["messages"] or message_id not in self._state["messages"][timeline_id]:
            logger.warning(f"Message {message_id} not found in timeline {timeline_id} for deletion.")
            return False
            
        del self._state["messages"][timeline_id][message_id]
        logger.debug(f"Deleted message {message_id} from timeline {timeline_id}")
        return True
        
    def get_messages(self, timeline_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves messages for a specific timeline, sorted by timestamp.
        
        Args:
            timeline_id: The ID of the timeline.
            limit: Optional maximum number of messages to return.
            
        Returns:
            A list of message dictionaries, sorted chronologically.
        """
        if timeline_id not in self._state["messages"]:
            return []
            
        messages = list(self._state["messages"][timeline_id].values())
        # Sort by timestamp
        messages.sort(key=lambda m: m.get("timestamp", 0))
        
        if limit is not None:
            return messages[-limit:]
        return messages

    def clear_history(self, timeline_id: str) -> bool:
        """
        Clears the message history for a specific timeline.
        
        Args:
            timeline_id: The ID of the timeline to clear.
        
        Returns:
            True if cleared successfully, False otherwise.
        """
        if timeline_id in self._state["messages"]:
            self._state["messages"][timeline_id] = {}
            logger.info(f"Cleared message history for timeline {timeline_id}")
            return True
        return False

    def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handle events to update the message history.
        """
        event_type = event.get("event_type")
        timeline_id = timeline_context.get("timeline_id")
        
        if not timeline_id:
            logger.warning(f"{self.COMPONENT_TYPE}: Received event without timeline_id: {event_type}")
            # Try to get primary timeline from TimelineComponent if available
            timeline_comp = self._get_timeline_comp()
            if timeline_comp:
                primary_timeline = timeline_comp.get_primary_timeline()
                if primary_timeline:
                    timeline_id = primary_timeline
                    logger.debug(f"{self.COMPONENT_TYPE}: Using primary timeline {timeline_id} for event {event_type}")
                else:
                    return False # Cannot process without a timeline context
            else:
                return False

        data = event.get("data", {})
        message_id = data.get("message_id")

        if event_type == "message_received" or event_type == "message_sent":
            # Assume 'data' contains the full message_data structure
            return self.add_message(data, timeline_id)
            
        elif event_type == "message_updated":
            if message_id:
                # Assume 'data' contains the fields to update
                update_fields = {k: v for k, v in data.items() if k != "message_id"}
                return self.update_message(message_id, update_fields, timeline_id)
            else:
                logger.warning(f"{self.COMPONENT_TYPE}: Received {event_type} without message_id.")
                return False
                
        elif event_type == "message_deleted":
            if message_id:
                return self.delete_message(message_id, timeline_id)
            else:
                logger.warning(f"{self.COMPONENT_TYPE}: Received {event_type} without message_id.")
                return False
        
        elif event_type == "clear_context":
            # This event type might signify clearing history
            return self.clear_history(timeline_id)

        # Could add handling for reactions here as well if needed
        # elif event_type == "reaction_added": ...
        # elif event_type == "reaction_removed": ...
            
        return False # Event type not handled for history update 