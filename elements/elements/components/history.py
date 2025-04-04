import logging
from typing import List, Dict, Any, Optional
import time

from .base import Component

logger = logging.getLogger(__name__)

class HistoryComponent(Component):
    """
    Manages the history of events or messages for an element.
    Provides methods to add, retrieve, and clear history.
    """

    COMPONENT_TYPE = "history"
    DEPENDENCIES = {"timeline"} # Often depends on timeline for ordering/context

    def __init__(self, element: 'BaseElement', max_history_size: int = 100):
        super().__init__(element)
        self.max_history_size = max_history_size
        self._history: List[Dict[str, Any]] = [] # Stores the actual event/message data
        logger.debug(f"HistoryComponent initialized for element {element.id}")

    def handle_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """Handles incoming events relevant to history tracking."""
        event_type = event.get("event_type")
        
        # --- Define relevant event types --- 
        # TODO: Make this configurable or based on element type?
        relevant_types = {"message_received", "message_sent", "agent_action", "system_notification"}
        # -----------------------------------
        
        if event_type in relevant_types:
             logger.debug(f"HistoryComponent on {self.element.id} handling event: {event_type}")
             # --- Extract and Store Information --- 
             # Basic: Store the entire event data (or relevant parts)
             history_entry = {
                 "event_id": event.get("event_id"),
                 "event_type": event_type,
                 "timestamp": event.get("timestamp", int(time.time() * 1000)),
                 "timeline_id": timeline_context.get("timeline_id"),
                 "data": event.get("data", event) # Store relevant data payload
             }
             # You might want more specific extraction based on event_type here
             # e.g., for messages: sender, recipient, text
             # e.g., for actions: action_name, parameters, result
             # ------------------------------------
             
             self.add_history_entry(history_entry)
             return True # Indicate the event was handled
             
        return False # Event not relevant to this component

    def add_history_entry(self, entry: Dict[str, Any]):
        """Adds an entry to the history, managing size limits."""
        self._history.append(entry)
        # Trim history if it exceeds the maximum size
        if len(self._history) > self.max_history_size:
            self._history.pop(0)
        logger.debug(f"Added entry to history for {self.element.id}. Size: {len(self._history)}")

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the current history."""
        return self._history

    def clear_history(self):
        """Clears the history."""
        self._history = []
        logger.info(f"History cleared for element {self.element.id}")

    # Optional: Methods to get specific types of history or search history
    # def get_messages(self) -> List[Dict[str, Any]]:
    #     return [entry for entry in self._history if entry.get("event_type", "").startswith("message_")] 