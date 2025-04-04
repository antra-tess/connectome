"""
Activity Listener (Formerly Message Listener)

Receives normalized events pushed by external adapters and routes them 
into the core system via the SpaceRegistry.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Callable
import uuid
import time

# Assuming SpaceRegistry is accessible via relative import or added to sys.path
from elements.space_registry import SpaceRegistry 
# Import HostEventLoop for type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from host.event_loop import HostEventLoop 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ActivityListener:
    """
    Receives normalized events, determines their timeline context, and enqueues
    them into the HostEventLoop for processing.
    
    This class is responsible for:
    1. Receiving normalized events (e.g., via an API endpoint it exposes)
    2. Determining the timeline context based on event data
    3. Enqueuing the event and context into the HostEventLoop
    
    The Listener assumes incoming data is *already normalized* by external adapters.
    """
    
    # Note: Removed config from __init__ for now, can be re-added if needed for server setup
    def __init__(self, space_registry: SpaceRegistry):
        """
        Initialize the activity listener.
        
        Args:
            space_registry: Registry (needed for context determination, potentially).
        """
        self.space_registry = space_registry # Keep for potential future use in context determination
        self._event_loop: Optional['HostEventLoop'] = None
        logger.info("Initialized ActivityListener")

    def set_event_loop(self, event_loop: 'HostEventLoop'):
        """Sets the HostEventLoop reference for enqueuing events."""
        self._event_loop = event_loop
        logger.info("HostEventLoop reference set in ActivityListener.")
    
    def _determine_timeline_context(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the timeline context for an incoming normalized event.
        
        Args:
            event_data: The normalized event data received.
            
        Returns:
            Dictionary containing timeline context information.
        """
        # Extract relevant fields for timeline context from the *normalized* event
        # The exact fields depend on the agreed-upon normalized format.
        adapter_type = event_data.get('adapter_type', 'unknown') 
        conversation_id = event_data.get('conversation_id')
        thread_id = event_data.get('thread_id') # Optional thread info
        
        # Generate a timeline ID based on normalized identifiers
        timeline_id = None
        if conversation_id: 
            # Use conversation_id potentially prefixed by adapter type if needed for uniqueness
            timeline_id = f"{adapter_type}_{conversation_id}" 
            if thread_id: # Append thread ID if present for finer granularity
                 timeline_id += f"_thread_{thread_id}" 
        else:
            # Fallback if conversation_id is missing in normalized event (should ideally not happen)
            timeline_id = f"timeline_{str(uuid.uuid4())[:8]}"
            logger.warning(f"Missing 'conversation_id' in normalized event. Generated random timeline ID: {timeline_id}")
        
        # Construct the timeline context
        timeline_context = {
            "timeline_id": timeline_id,
            "is_primary": True,  # Assume primary unless specified otherwise?
            "last_event_id": None, # Updated by TimelineComponent
            "timestamp": event_data.get("timestamp", int(time.time() * 1000)) # Use event timestamp if available
        }
        
        logger.debug(f"Determined timeline context for event {event_data.get('event_id', '?')}: {timeline_context}")
        return timeline_context
    
    def handle_incoming_event(self, normalized_event_data: Dict[str, Any]) -> bool:
        """
        Handles a normalized incoming event by validating it, determining context,
        and enqueuing it into the HostEventLoop.
        """
        if not self._event_loop:
             logger.error("Cannot handle incoming event: HostEventLoop reference not set.")
             return False
             
        try:
            # Validate, ensure IDs/timestamps (Existing logic retained)
            logger.debug(f"Handling incoming event: {normalized_event_data.get('event_type', 'unknown')}")
            if not isinstance(normalized_event_data, dict) or 'event_type' not in normalized_event_data:
                logger.error(f"Invalid normalized event format: {normalized_event_data}")
                return False
            if "event_id" not in normalized_event_data:
                 normalized_event_data["event_id"] = f"evt_{uuid.uuid4().hex[:8]}"
            if "timestamp" not in normalized_event_data:
                 normalized_event_data["timestamp"] = int(time.time() * 1000)
                 
            # Determine timeline context
            timeline_context = self._determine_timeline_context(normalized_event_data)
            
            # Enqueue the event and its context into the host event loop
            self._event_loop.enqueue_event(normalized_event_data, timeline_context)
            logger.debug(f"Enqueued event {normalized_event_data['event_id']} into HostEventLoop.")
            
            return True # Indicate successful receipt and enqueueing
            
        except Exception as e:
            logger.error(f"Error handling incoming event: {e}", exc_info=True)
            return False

    # Removed handle_clear_context - should likely be a specific event type handled above
    # Removed _standardize_message_format - assuming input is already normalized
    
    # --- TODO: Add mechanism to expose handle_incoming_event --- 
    # This class needs a way to receive events. Options:
    # 1. Run an HTTP server (e.g., Flask, FastAPI) with an endpoint calling handle_incoming_event.
    # 2. Connect to a message queue (e.g., RabbitMQ, Kafka) and consume messages.
    # 3. Be called directly by adapter processes if running in the same environment.
    # Example (conceptual):
    # def start_listening(self, host='0.0.0.0', port=5000):
    #     from flask import Flask, request, jsonify
    #     app = Flask(__name__)
    #     @app.route('/event', methods=['POST'])
    #     def event_endpoint():
    #         if not request.json:
    #             return jsonify({"error": "Invalid JSON"}), 400
    #         success = self.handle_incoming_event(request.json)
    #         return jsonify({"success": success}), 200 if success else 500
    #     logger.info(f"Starting Activity Listener server on {host}:{port}")
    #     app.run(host=host, port=port)
    # ----------------------------------------------------------- 