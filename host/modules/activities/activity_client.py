"""
Activity Client Module (Client Model)

Connects as a client (e.g., Socket.IO) to external Adapter API endpoints 
to send outgoing actions and receive normalized incoming events.
"""

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Callable
import time # Added for timestamp

import socketio # Using python-socketio

# Event Loop for queuing incoming events
from host.event_loop import HostEventLoop 

logger = logging.getLogger(__name__)

# Example config defaults (move to config loading later)
SOCKET_RECONNECTION_ATTEMPTS = 3
SOCKET_RECONNECTION_DELAY = 5
SOCKET_TIMEOUT = 10

class ActivityClient:
    """
    Connects to external Adapter APIs (assumed Socket.IO for now), 
    sends standardized actions, and receives normalized events, 
    queuing them onto the HostEventLoop.
    """
    
    def __init__(self, host_event_loop: HostEventLoop, adapter_api_configs: List[Dict[str, Any]]):
        """
        Initialize the Activity Client.
        
        Args:
            host_event_loop: The main HostEventLoop instance.
            adapter_api_configs: List of configurations for external adapter APIs.
                                 Each dict should contain at least 'id' and 'url'.
        """
        self._host_event_loop = host_event_loop
        # Stores adapter configs {adapter_id: config_dict}
        self.adapter_configs: Dict[str, Dict[str, Any]] = {}
        # Stores connected async client instances {adapter_id: socketio.AsyncClient}
        self.clients: Dict[str, socketio.AsyncClient] = {}
        # Tracks connection state {adapter_id: bool}
        self.connected_adapters: Dict[str, bool] = {}

        self._load_adapter_configs(adapter_api_configs)
        logger.info(f"ActivityClient initialized with {len(self.adapter_configs)} adapter API configs.")

    def _load_adapter_configs(self, adapter_api_configs: List[Dict[str, Any]]) -> None:
        """Loads and validates adapter API configurations."""
        for config in adapter_api_configs:
            if "id" in config and "url" in config:
                adapter_id = config["id"]
                self.adapter_configs[adapter_id] = config
                self.connected_adapters[adapter_id] = False
                logger.debug(f"Loaded adapter config: {adapter_id} -> {config.get('url')}")
            else:
                logger.warning(f"Skipping invalid adapter config: {config}")

    async def start_connections(self) -> None:
        """Establish connections to all configured adapter APIs."""
        logger.info("Attempting connections to external Adapter APIs...")
        connect_tasks = []
        for adapter_id, config in self.adapter_configs.items():
            connect_tasks.append(self._connect_to_adapter_api(adapter_id, config))
        
        if connect_tasks:
            results = await asyncio.gather(*connect_tasks, return_exceptions=True)
            for i, adapter_id in enumerate(self.adapter_configs.keys()):
                 if isinstance(results[i], Exception):
                      logger.error(f"Failed to connect to adapter '{adapter_id}': {results[i]}")
                 elif results[i] is False: # Explicit failure from _connect
                      logger.error(f"Failed to connect to adapter '{adapter_id}' (check logs).")
        logger.info("Finished connection attempts to Adapter APIs.")

    async def _connect_to_adapter_api(self, adapter_id: str, config: Dict[str, Any]) -> bool:
        """Connect to a specific adapter API endpoint using socketio.AsyncClient."""
        url = config.get("url")
        if not url:
            logger.error(f"No URL specified for adapter API '{adapter_id}'")
            return False

        if adapter_id in self.clients and self.connected_adapters.get(adapter_id):
             logger.info(f"Already connected to adapter '{adapter_id}'. Skipping.")
             return True
             
        client = socketio.AsyncClient(
            reconnection=True,
            reconnection_attempts=SOCKET_RECONNECTION_ATTEMPTS,
            reconnection_delay=SOCKET_RECONNECTION_DELAY,
            request_timeout=SOCKET_TIMEOUT
        )
        self.clients[adapter_id] = client
        self.connected_adapters[adapter_id] = False # Assume disconnected until confirmed

        # --- Define Event Handlers --- 
        # Use closures to capture adapter_id for logging/handling
        
        @client.event
        async def connect():
            logger.info(f"Successfully connected to adapter API '{adapter_id}' at {url}")
            self.connected_adapters[adapter_id] = True
            # Send registration/hello message if needed by adapter API
            # await self._register_with_adapter_api(adapter_id, config)

        @client.event
        async def disconnect():
            logger.info(f"Disconnected from adapter API '{adapter_id}'")
            self.connected_adapters[adapter_id] = False

        @client.event
        async def connect_error(data):
            logger.error(f"Connection error with adapter API '{adapter_id}': {data}")
            self.connected_adapters[adapter_id] = False
            # No need to disconnect client instance here, library handles retries

        # --- Handler for Incoming Normalized Events --- 
        @client.on("normalized_event") 
        async def handle_normalized_event(data: Dict[str, Any]):
            if not isinstance(data, dict):
                 logger.warning(f"Received non-dict normalized_event from '{adapter_id}': {data}")
                 return
            
            # Add the ID of the connection this came from for routing/history
            data['source_adapter_id'] = adapter_id 
            
            logger.debug(f"Received normalized_event from '{adapter_id}': Type={data.get('event_type')}. Enqueuing...")
            # Enqueue the *entire received data structure* onto the main loop
            # Pass an empty dict for timeline_context as ActivityClient doesn't determine it.
            self._host_event_loop.enqueue_incoming_event(data, {})
            
        # --- Connect --- 
        try:
            auth_data = config.get("auth") # Expect auth dict if needed
            logger.info(f"Connecting to adapter API '{adapter_id}' at {url}...")
            await client.connect(url, auth=auth_data, namespaces=["/"])
            # Note: Connection status set by async connect event handler
            return True # Indicates connection attempt initiated successfully
        except socketio.exceptions.ConnectionError as e:
             logger.error(f"Failed to connect to adapter '{adapter_id}': {e}")
             # Clean up client instance if connection fails definitively?
             if adapter_id in self.clients: del self.clients[adapter_id]
             self.connected_adapters[adapter_id] = False
             return False
        except Exception as e:
             logger.error(f"Unexpected error during connection attempt to '{adapter_id}': {e}", exc_info=True)
             if adapter_id in self.clients: del self.clients[adapter_id]
             self.connected_adapters[adapter_id] = False
             return False

    async def handle_outgoing_action(self, action: Dict[str, Any]):
        """
        Handles an outgoing action routed from the HostEventLoop.
        Finds the target adapter API client, emits the action externally,
        and then queues an internal 'agent_message_sent' event for history recording.
        """
        action_type = action.get("action_type")
        payload = action.get("payload", {}) 
        # Assume these are passed in the action payload from AgentLoop
        requesting_element_id = payload.get("requesting_element_id")
        agent_name = payload.get("agent_name", "Unknown Agent") 
        target_adapter_id = payload.get("adapter_id") 
        
        if not target_adapter_id:
            logger.error(f"Cannot handle outgoing action '{action_type}': Missing 'adapter_id' in payload.")
            return # TODO: Maybe return failure status?
        if not requesting_element_id:
             logger.error(f"Cannot handle outgoing action '{action_type}': Missing 'requesting_element_id' in payload (needed for history update).")
             return
            
        client = self.clients.get(target_adapter_id)
        if not client or not self.connected_adapters.get(target_adapter_id):
            logger.error(f"Cannot handle outgoing action '{action_type}': Target adapter API '{target_adapter_id}' not found or not connected.")
            # TODO: Send failure back?
            return
            
        # --- Structure the data to be emitted externally --- 
        outgoing_event_type = action_type
        outgoing_data = {} # Start fresh for explicit mapping
        is_message_sent = False # Flag to check if we should record history
        message_text = None # Store text for history
        conversation_id = None # Store conversation for history
        
        # Map payload fields to expected outgoing data structure per action_type
        if outgoing_event_type == "send_message":
             conversation_id = payload.get("conversation_id")
             message_text = payload.get("text")
             outgoing_data = {
                  "conversation_id": conversation_id,
                  "text": message_text
                  # TODO: Add threadId, attachments, mentions from payload if needed by adapter API
             }
             is_message_sent = True # We want to record this in history
             
        elif outgoing_event_type == "edit_message":
             outgoing_data = {
                  "conversation_id": payload.get("conversation_id"),
                  "message_id": payload.get("message_id"),
                  "text": payload.get("text")
             }
             # We probably don't record edits as new history turns
             
        elif outgoing_event_type == "delete_message":
             outgoing_data = {
                  "conversation_id": payload.get("conversation_id"),
                  "message_id": payload.get("message_id")
             }
             # We probably don't record deletions as new history turns
             
        elif outgoing_event_type in ["add_reaction", "remove_reaction"]:
             outgoing_data = {
                  "conversation_id": payload.get("conversation_id"),
                  "message_id": payload.get("message_id"),
                  "emoji": payload.get("emoji")
             }
             # Don't record reactions as history turns either
             
        else:
             logger.warning(f"Outgoing action type '{outgoing_event_type}' has no specific data mapping. Sending raw payload args minus internal fields.")
             # Filter out internal fields before sending raw tool_args
             internal_keys = {'adapter_id', 'tool_name', 'tool_args', 'tool_call_id', 
                              'requesting_element_id', 'requesting_agent_id', 'agent_name'}
             outgoing_data = {k: v for k, v in payload.get('tool_args', {}).items() if k not in internal_keys}

        # Construct the final object to emit to the external adapter API
        data_to_emit = {
             "event_type": outgoing_event_type,
             "data": outgoing_data
        }
        
        # Emit using the "bot_response" event name to the adapter
        event_name_to_emit = "bot_response"
        logger.debug(f"Sending '{event_name_to_emit}' event to adapter API '{target_adapter_id}' with data: {data_to_emit}")
        
        try:
            # --- Emit Externally --- 
            await client.emit(event_name_to_emit, data_to_emit)
            logger.debug(f"Successfully emitted '{event_name_to_emit}' to '{target_adapter_id}'")
            
            # --- Enqueue Internal Event for History (if applicable) --- 
            if is_message_sent and message_text is not None and conversation_id is not None:
                history_event_payload = {
                    # Routing Key: ID of the InnerSpace that sent the message
                    "target_element_id": requesting_element_id, 
                    # Data needed by the history handler:
                    "adapter_id": target_adapter_id,
                    "conversation_id": conversation_id,
                    "text": message_text,
                    "agent_name": agent_name, 
                    "timestamp": int(time.time() * 1000) # Timestamp of successful send
                }
                history_event = {
                    "event_type": "agent_message_sent",
                    "payload": history_event_payload
                }
                logger.debug(f"Enqueuing internal event: {history_event['event_type']}")
                self._host_event_loop.enqueue_internal_event(history_event) # Use a separate queue/method?
                
        except Exception as e:
            logger.error(f"Error emitting action '{action_type}' to adapter '{target_adapter_id}': {e}", exc_info=True)
            # TODO: Send feedback/error event back to originator (InnerSpace)?
            # Maybe enqueue an 'action_failed' internal event?

    async def shutdown(self):
        """Disconnects from all adapter APIs."""
        logger.info("Disconnecting from all Adapter APIs...")
        disconnect_tasks = []
        for adapter_id, client in self.clients.items():
             if self.connected_adapters.get(adapter_id):
                  logger.info(f"Disconnecting from adapter API '{adapter_id}'...")
                  disconnect_tasks.append(asyncio.create_task(client.disconnect()))
             
        if disconnect_tasks:
            results = await asyncio.gather(*disconnect_tasks, return_exceptions=True)
            # Log any errors during disconnection
            adapter_ids = list(self.clients.keys())
            for i, res in enumerate(results):
                 if isinstance(res, Exception):
                      logger.error(f"Error disconnecting from adapter '{adapter_ids[i]}': {res}")
                      
        self.clients.clear()
        self.connected_adapters.clear()
        logger.info("All adapter API connections closed.") 