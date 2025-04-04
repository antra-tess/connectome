"""
Global Attention Component
Placeholder component for managing attention requests across spaces.
"""

import logging
from typing import Dict, Any, Optional, List
import time

from ..base_component import Component

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GlobalAttentionComponent(Component):
    """
    Manages attention requests originating from potentially multiple spaces/elements.
    This component would typically reside on the InnerSpace.
    (Placeholder implementation - needs refinement)
    """
    
    COMPONENT_TYPE: str = "global_attention_manager"
    DEPENDENCIES: List[str] = [] 
    
    HANDLED_EVENT_TYPES: List[str] = [
        "attention_requested",
        "attention_cleared"
    ]
    
    def __init__(self, element=None):
        super().__init__(element)
        # State stores current attention requests: { request_id: request_data }
        # request_id could be f"{space_id}_{element_id}_{timestamp}" or similar
        self._state = {
            "attention_requests": {} 
        }

    def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handle attention requests and clears.
        """
        event_type = event.get("event_type")
        data = event.get("data", {})
        space_id = data.get("space_id")
        element_id = data.get("element_id")
        # Use source_element_id if provided (e.g., for nested attention requests)
        source_element_id = data.get("source_element_id", element_id) 
        
        if not space_id or not source_element_id:
             logger.warning(f"{self.COMPONENT_TYPE}: Received {event_type} without space_id or source_element_id.")
             return False
             
        # Generate a reasonably unique key for the request
        request_key = f"{space_id}::{source_element_id}"

        if event_type == "attention_requested":
            request_data = data.get("request_data", {})
            timestamp = data.get("timestamp", int(time.time() * 1000))
            
            if request_key in self._state["attention_requests"]:
                 logger.debug(f"Updating existing attention request for {request_key}")
            else:
                 logger.info(f"New attention request from {request_key}")
                 
            self._state["attention_requests"][request_key] = {
                "space_id": space_id,
                "element_id": element_id,
                "source_element_id": source_element_id,
                "request_data": request_data,
                "timestamp": timestamp,
                "timeline_context": timeline_context # Store context for potential HUD use
            }
            # TODO: Need mechanism to signal this attention to the HUD/Shell
            return True
            
        elif event_type == "attention_cleared":
            if request_key in self._state["attention_requests"]:
                 logger.info(f"Clearing attention request for {request_key}")
                 del self._state["attention_requests"][request_key]
                 # TODO: Need mechanism to signal clearance to HUD/Shell
                 return True
            else:
                 logger.debug(f"Received attention_cleared for non-existent request key: {request_key}")
                 return False # Not technically handled if key didn't exist
                 
        return False
        
    def get_attention_requests(self) -> Dict[str, Any]:
         """Returns the current dictionary of attention requests."""
         return self._state["attention_requests"].copy()

    def clear_all_requests(self) -> None:
         """Clears all current attention requests."""
         if self._state["attention_requests"]:
              logger.info("Clearing all attention requests.")
              self._state["attention_requests"] = {}
              # TODO: Signal clearance to HUD/Shell 