"""
Uplink Connection Component
Component for managing the connection state and spans of an UplinkProxy.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
import uuid
import time
from datetime import datetime, timedelta
import asyncio # Added for async sleep

# Type hinting imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ....space_registry import SpaceRegistry
    from ....elements.space import Space
    from elements.elements.uplink import UplinkProxy

from ..base_component import Component
from ..space.timeline_component import TimelineComponent # Needed to record connection events
# Import the registry decorator
from elements.component_registry import register_component

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@register_component
class UplinkConnectionComponent(Component):
    """
    Manages the connection lifecycle, state, and spans for an Uplink element.
    """
    COMPONENT_TYPE = "UplinkConnectionComponent"
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
    
    def __init__(self, remote_space_id: Optional[str] = None,
                 sync_interval: int = 60,
                 space_registry: Optional['SpaceRegistry'] = None,
                 delta_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None, **kwargs):
        super().__init__(**kwargs)
        self.remote_space_id = remote_space_id or "unknown_remote"
        self._space_registry = space_registry
        self._delta_callback = delta_callback
        self._remote_space_ref: Optional['Space'] = None
        
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
        # _space_registry and _delta_callback are passed in, no need to get from owner element initially

    def initialize(self, **kwargs) -> None:
        super().initialize(**kwargs)
        # Initialization logic that depends on self.owner can go here
        # For example, if _space_registry or _delta_callback were NOT passed via __init__:
        # if self.owner and hasattr(self.owner, '_space_registry'):
        #     self._space_registry = self.owner._space_registry
        # if self.owner and hasattr(self.owner, '_delta_callback'):
        #      self._delta_callback = self.owner._delta_callback # Assuming owner (UplinkProxy) has it
        logger.debug(f"UplinkConnectionComponent initialized for {self.owner.id if self.owner else 'UnknownOwner'}")

    def _get_timeline_comp(self) -> Optional[TimelineComponent]:
        """Helper to get the associated TimelineComponent."""
        if not self.owner: # Changed from self.element
            return None
        return self.owner.get_component_by_type("timeline") # Changed from self.element

    @property
    def is_connected(self) -> bool:
        """Returns True if the connection is established."""
        return self._state["connected"]

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
             
        if self.is_connected:
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

            # --- NEW: Register listener with remote space --- 
            if self._space_registry and self._delta_callback:
                try:
                     self._remote_space_ref = self._space_registry.get_space(self.remote_space_id)
                     if self._remote_space_ref:
                         logger.info(f"Registering delta listener with remote space {self.remote_space_id}")
                         self._remote_space_ref.register_uplink_listener(self._delta_callback)
                     else:
                          logger.error(f"Could not register listener: Remote space {self.remote_space_id} not found in registry.")
                          # TODO: Should connection fail if remote space cannot be found?
                except Exception as e:
                     logger.error(f"Error registering uplink listener with remote space {self.remote_space_id}: {e}", exc_info=True)
            elif not self._space_registry:
                 logger.warning("Cannot register uplink listener: SpaceRegistry not provided.")
            elif not self._delta_callback:
                 logger.warning("Cannot register uplink listener: Delta callback not provided.")
            # --- END NEW --- 

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
        # --- NEW: Unregister listener --- 
        if self._remote_space_ref and self._delta_callback:
             try:
                  logger.info(f"Unregistering delta listener from remote space {self.remote_space_id}")
                  self._remote_space_ref.unregister_uplink_listener(self._delta_callback)
             except Exception as e:
                  logger.error(f"Error unregistering uplink listener from {self.remote_space_id}: {e}", exc_info=True)
        self._remote_space_ref = None # Clear reference regardless
        # --- END NEW --- 

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
            "element_id": self.owner.id if self.owner else None, # Changed from self.element
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

    async def send_event_to_remote_space(self, event_payload: Dict[str, Any]) -> bool:
        """
        Sends an event (typically an action request) to the connected remote space.

        Args:
            event_payload: The event data to send to the remote space.
                         This payload should be structured for the remote space to understand,
                         e.g., containing action_name, parameters, and target element info.

        Returns:
            True if the event was successfully dispatched to the remote space, False otherwise.
        """
        if not self._state["connected"]:
            logger.error(f"[{self.owner.id if self.owner else 'N/A'}/{self.COMPONENT_TYPE}] Cannot send event: Not connected to remote space '{self.remote_space_id}'.")
            return False
        
        if not self._space_registry:
            logger.error(f"[{self.owner.id if self.owner else 'N/A'}/{self.COMPONENT_TYPE}] Cannot send event: SpaceRegistry not available.")
            return False

        remote_space = self._space_registry.get_space(self.remote_space_id)
        if not remote_space:
            logger.error(f"[{self.owner.id if self.owner else 'N/A'}/{self.COMPONENT_TYPE}] Cannot send event: Remote space '{self.remote_space_id}' not found in registry.")
            return False
        
        try:
            # Assuming remote_space.receive_event is synchronous based on previous reversions
            # If it were async, this would need to be awaited, and this method would need to be async.
            # Based on recent changes, Space.receive_event is sync and handles async sub-tasks with create_task.
            timeline_context = {} # Default/empty timeline context for now
            primary_timeline = remote_space.get_primary_timeline()
            if primary_timeline:
                timeline_context['timeline_id'] = primary_timeline
            
            remote_space.receive_event(event_payload, timeline_context)
            logger.info(f"[{self.owner.id if self.owner else 'N/A'}/{self.COMPONENT_TYPE}] Successfully dispatched event to remote space '{self.remote_space_id}'. Event type: {event_payload.get('event_type')}")
            return True
        except Exception as e:
            logger.error(f"[{self.owner.id if self.owner else 'N/A'}/{self.COMPONENT_TYPE}] Error dispatching event to remote space '{self.remote_space_id}': {e}", exc_info=True)
            return False

    async def fetch_remote_public_tool_definitions(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetches the public tool definitions from the connected remote space.

        Returns:
            A list of tool definition dictionaries, or None if fetching fails or not connected.
        """
        if not self._state["connected"]:
            logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Cannot fetch tool definitions: Not connected to remote space '{self.remote_space_id}'.")
            return None
        
        if not self._space_registry:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Cannot fetch tool definitions: SpaceRegistry not available.")
            return None

        remote_space_obj: Optional['Space'] = self._space_registry.get_space(self.remote_space_id)
        if not remote_space_obj:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Cannot fetch tool definitions: Remote space '{self.remote_space_id}' not found in registry.")
            return None
        
        if not hasattr(remote_space_obj, 'get_public_tool_definitions') or not callable(getattr(remote_space_obj, 'get_public_tool_definitions')):
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Remote space '{self.remote_space_id}' (type: {type(remote_space_obj)}) does not have a callable 'get_public_tool_definitions' method.")
            return None
            
        try:
            # Space.get_public_tool_definitions() is currently synchronous.
            # If it became async, this would need `await`.
            tool_definitions = remote_space_obj.get_public_tool_definitions()
            logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Successfully fetched {len(tool_definitions)} tool definitions from remote space '{self.remote_space_id}'.")
            return tool_definitions
        except Exception as e:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Error calling get_public_tool_definitions on remote space '{self.remote_space_id}': {e}", exc_info=True)
            return None 