"""
Uplink Connection Component
Component for managing the connection state and spans of an UplinkProxy.
"""

import logging
from typing import Dict, Any, Optional, List
import uuid
import time
from datetime import datetime, timedelta

from ..base_component import Component
from ..space.timeline_component import TimelineComponent # Needed to record connection events

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UplinkConnectionComponent(Component):
    """
    Manages the connection lifecycle, state, and spans for an Uplink element.
    """
    
    COMPONENT_TYPE: str = "uplink_connection"
    DEPENDENCIES: List[str] = ["timeline"] # Needs timeline to record its own events
    
    # Events this component handles internally or listens for
    HANDLED_EVENT_TYPES: List[str] = [
        "uplink_connected", 
        "uplink_disconnected",
        "uplink_error",
        # Might listen for external triggers 
        # "connect_request", 
        # "disconnect_request"
    ]
    
    def __init__(self, element=None, remote_space_id: Optional[str] = None, sync_interval: int = 60):
        super().__init__(element)
        self.remote_space_id = remote_space_id or "unknown_remote"
        # State stores connection details and history
        self._state = {
            "connected": False,
            "last_connection_attempt": None,
            "last_successful_connection": None,
            "last_disconnection_time": None,
            "last_sync_request_time": None, # Managed by cache component?
            "sync_interval": sync_interval,  # seconds
            "error": None,
            "connection_history": [], # List of {'type': 'connect'/'disconnect'/'error', 'timestamp': ..., 'details': ...}
            "connection_spans": [], # List of {'start_time': ..., 'end_time': ...}
            "current_span_start": None
        }

    def _get_timeline_comp(self) -> Optional[TimelineComponent]:
        """Helper to get the associated TimelineComponent."""
        if not self.element:
            return None
        return self.element.get_component_by_type("timeline")

    def connect(self) -> bool:
        """
        Attempts to establish a connection to the remote space.
        (Simulated for now - real implementation would involve network calls)
        
        Returns:
            True if connection was successful (or already connected), False on failure.
        """
        if not self._is_initialized or not self._is_enabled:
             logger.warning(f"{self.COMPONENT_TYPE}: Cannot connect, component not ready.")
             return False
             
        if self._state["connected"]:
            logger.debug(f"Already connected to {self.remote_space_id}")
            return True

        timestamp = int(time.time() * 1000)
        self._state["last_connection_attempt"] = timestamp
        logger.info(f"Attempting to connect to remote space: {self.remote_space_id}")
        
        # --- Simulate connection attempt --- 
        # In a real implementation, this would involve network calls, authentication etc.
        # For now, let's assume it succeeds.
        connection_successful = True 
        error_details = None
        # ---------------------------------
        
        if connection_successful:
            self._state["connected"] = True
            self._state["last_successful_connection"] = timestamp
            self._state["error"] = None
            self._state["connection_history"].append({
                "type": "connect", 
                "timestamp": timestamp
            })
            # Start a new connection span
            if self._state["current_span_start"] is None:
                 self._state["current_span_start"] = timestamp
                 
            logger.info(f"Successfully connected to {self.remote_space_id}")
            # Record event in the element's *own* timeline
            self._record_timeline_event("uplink_connected", {"remote_space_id": self.remote_space_id})
            return True
        else:
            self._state["connected"] = False
            self._state["error"] = error_details or "Connection failed (simulated)"
            self._state["connection_history"].append({
                "type": "error", 
                "timestamp": timestamp, 
                "details": self._state["error"]
            })
            # End any active connection span on failure
            self._end_current_span(timestamp)
            logger.error(f"Failed to connect to {self.remote_space_id}: {self._state['error']}")
            # Record error event in the element's *own* timeline
            self._record_timeline_event("uplink_error", {"remote_space_id": self.remote_space_id, "error": self._state["error"]})
            return False

    def disconnect(self) -> bool:
        """
        Disconnects from the remote space.
        
        Returns:
            True if disconnected successfully (or already disconnected).
        """
        if not self._state["connected"]:
            logger.debug(f"Already disconnected from {self.remote_space_id}")
            return True
            
        timestamp = int(time.time() * 1000)
        logger.info(f"Disconnecting from remote space: {self.remote_space_id}")
        
        # --- Simulate disconnection --- 
        # Real implementation might involve notifying remote etc.
        disconnection_successful = True
        # ----------------------------
        
        if disconnection_successful:
             self._state["connected"] = False
             self._state["last_disconnection_time"] = timestamp
             self._state["connection_history"].append({
                 "type": "disconnect", 
                 "timestamp": timestamp
             })
             # End the current connection span
             self._end_current_span(timestamp)
             logger.info(f"Successfully disconnected from {self.remote_space_id}")
             # Record event in the element's *own* timeline
             self._record_timeline_event("uplink_disconnected", {"remote_space_id": self.remote_space_id})
             return True
        else:
            # Handle disconnection failure (less common unless network issue during notification)
            error_msg = "Disconnection failed (simulated)"
            self._state["error"] = error_msg
            self._state["connection_history"].append({"type": "error", "timestamp": timestamp, "details": error_msg})
            logger.error(f"Failed to cleanly disconnect from {self.remote_space_id}")
            # Record error event
            self._record_timeline_event("uplink_error", {"remote_space_id": self.remote_space_id, "error": error_msg})
            return False
            
    def _end_current_span(self, end_time: int):
        """Ends the current connection span if one is active."""
        if self._state["current_span_start"] is not None:
             self._state["connection_spans"].append({
                 "start_time": self._state["current_span_start"],
                 "end_time": end_time
             })
             self._state["current_span_start"] = None
             logger.debug(f"Ended connection span for {self.remote_space_id}")

    def get_connection_state(self) -> Dict[str, Any]:
        """Returns the current connection status."""
        return {
             "remote_space_id": self.remote_space_id,
             "connected": self._state["connected"],
             "last_successful_connection": self._state["last_successful_connection"],
             "last_disconnection_time": self._state["last_disconnection_time"],
             "error": self._state["error"]
        }

    def get_connection_spans(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
         """
         Returns recorded connection spans.
         Optionally includes the currently active span if connected.
         """
         spans = self._state["connection_spans"].copy()
         if self._state["connected"] and self._state["current_span_start"] is not None:
              spans.append({
                   "start_time": self._state["current_span_start"],
                   "end_time": None # Indicates active span
              })
         # Sort by start time just in case
         spans.sort(key=lambda s: s["start_time"])
         if limit:
              return spans[-limit:]
         return spans
         
    def _record_timeline_event(self, event_type: str, data: Dict[str, Any]):
        """Records an uplink connection event in the element's own timeline."""
        timeline_comp = self._get_timeline_comp()
        if not timeline_comp:
             logger.warning(f"Cannot record uplink event {event_type}, missing TimelineComponent.")
             return
             
        primary_timeline = timeline_comp.get_primary_timeline()
        if not primary_timeline:
             logger.warning(f"Cannot record uplink event {event_type}, no primary timeline found.")
             return

        event_data = {
            "event_id": f"{event_type}_{uuid.uuid4().hex[:8]}",
            "event_type": event_type,
            "timestamp": int(time.time() * 1000),
            "element_id": self.element.id if self.element else None,
            "data": data
        }
        timeline_comp.add_event_to_timeline(event_data, {"timeline_id": primary_timeline})

    # _on_event could handle external requests if needed
    # def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
    #     event_type = event.get("event_type")
    #     if event_type == "connect_request":
    #         return self.connect()
    #     elif event_type == "disconnect_request":
    #         return self.disconnect()
    #     return False

    def _on_cleanup(self) -> bool:
         """Ensures disconnection on cleanup."""
         if self._state["connected"]:
              logger.info(f"Cleaning up UplinkConnectionComponent: attempting disconnect from {self.remote_space_id}")
              self.disconnect()
         return True 