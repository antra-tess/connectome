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

        # --- Handler for Incoming Events from Adapter API --- 
        # Renamed from "normalized_event" to "bot_request"
        @client.on("bot_request") 
        async def handle_bot_request(raw_payload: Dict[str, Any]):
            """Handles incoming events (structured as per user spec) from the adapter API."""
            
            if not isinstance(raw_payload, dict):
                 logger.warning(f"Received non-dict bot_request from '{adapter_id}': {raw_payload}")
                 return
            
            # Extract core fields from the raw payload
            raw_event_type = raw_payload.get('event_type')
            raw_data = raw_payload.get('data')
            # adapter_type = raw_payload.get('adapter_type') # Could store/use this if needed

            if not raw_event_type or not isinstance(raw_data, dict):
                 logger.warning(f"Received bot_request from '{adapter_id}' missing 'event_type' or valid 'data': {raw_payload}")
                 return

            # Extract the adapter ID (which is adapter_name in the raw data)
            # Use the connection's adapter_id as fallback/confirmation?
            # For now, trust the data payload.
            source_adapter_id_from_data = raw_data.get('adapter_name')
            if not source_adapter_id_from_data:
                 logger.warning(f"Received bot_request data from '{adapter_id}' missing 'adapter_name': {raw_data}")
                 # Use the connection's adapter_id as fallback?
                 source_adapter_id_from_data = adapter_id 
                 # Maybe return an error?
                 
            # Log if the adapter_name in data doesn't match the connection ID - indicates potential issue
            if source_adapter_id_from_data != adapter_id:
                 logger.warning(f"Adapter name '{source_adapter_id_from_data}' in bot_request data does not match connection ID '{adapter_id}'. Using data value.")

            logger.debug(f"Received bot_request from '{adapter_id}': Type={raw_event_type}. Enqueuing...")
            
            # Construct the event structure expected by HostEventLoop/ExternalEventRouter
            event_to_enqueue = {
                # Set the reliable source ID from the connection
                "source_adapter_id": adapter_id, 
                # Pass the raw event type and data payload for the router to interpret
                "payload": {
                     "event_type_from_adapter": raw_event_type, # e.g., "message_received", "message_updated"
                     "adapter_data": raw_data # The entire 'data' dict from the raw payload
                }
            }
            
            # Enqueue the event onto the main loop
            # Pass an empty dict for timeline_context initially.
            self._host_event_loop.enqueue_incoming_event(event_to_enqueue, {})
            
        # --- Handler for Incoming History Response --- 
        # REMOVED @client.on("history_response") handler

        # --- NEW Handler for Generic Success Responses ---
        @client.on("request_success")
        async def handle_request_success(raw_payload: Dict[str, Any]):
            """Handles generic success responses from the adapter API."""
            if not isinstance(raw_payload, dict):
                 logger.warning(f"Received non-dict request_success from '{adapter_id}': {raw_payload}")
                 return
            
            request_type = raw_payload.get('request_type')
            status = raw_payload.get('status') # Should be "success"
            data = raw_payload.get('data')

            logger.debug(f"Received request_success from '{adapter_id}'. Type: {request_type}, Status: {status}, Data keys: {list(data.keys()) if isinstance(data, dict) else None}")

            if status != "success":
                logger.warning(f"Received request_success event from '{adapter_id}' but status was not 'success': {status}")
                # Optionally handle non-success statuses if they use this event type

            # --- Check if it's a response to fetch_history ---
            if request_type == "history" and isinstance(data, dict):
                conversation_id = data.get('conversation_id')
                messages = data.get('messages') # List of message dicts

                if not conversation_id or not isinstance(messages, list):
                     logger.warning(f"Received history success response from '{adapter_id}' missing 'conversation_id' or valid 'messages' list in data: {data}")
                     return

                logger.info(f"Received history success response from '{adapter_id}' for conv '{conversation_id}' with {len(messages)} messages. Enqueuing...")
                
                event_to_enqueue = {
                    "source_adapter_id": adapter_id, 
                    "event_type": "connectome_history_received", 
                    "payload": { 
                         "conversation_id": conversation_id,
                         "messages": messages, 
                         "adapter_data": data 
                    }
                }
                self._host_event_loop.enqueue_incoming_event(event_to_enqueue, {})
            
            # --- Check if it's a response to get_attachment ---
            elif request_type == "attachment" and isinstance(data, dict):
                # 'data' here is the payload from the adapter for the fetched attachment
                # It should contain conversation_id, attachment_id, filename, content_type, content etc.
                conversation_id_from_attach = data.get('conversation_id') # Adapter should echo this back
                attachment_id_from_attach = data.get('attachment_id')   # Adapter should echo this back

                if not conversation_id_from_attach or not attachment_id_from_attach:
                    logger.warning(f"Received attachment success response from '{adapter_id}' missing 'conversation_id' or 'attachment_id' in data: {data}")
                    return

                logger.info(f"Received attachment success response from '{adapter_id}' for conv '{conversation_id_from_attach}', attachment '{attachment_id_from_attach}'. Enqueuing...")
                
                event_to_enqueue = {
                    "source_adapter_id": adapter_id, 
                    "event_type": "connectome_attachment_received",
                    "payload": {
                        # Pass the conversation_id as identified in the attachment response data
                        "conversation_id": conversation_id_from_attach,
                        # Pass the entire 'data' from the adapter as 'adapter_data'
                        # This will contain attachment_id, filename, content, content_type etc.
                        "adapter_data": data 
                    }
                }
                self._host_event_loop.enqueue_incoming_event(event_to_enqueue, {})

            elif request_type == "send_message":
                # Log success, maybe extract message ID if needed later for correlation
                external_msg_id = data.get('message_external_id') if isinstance(data, dict) else None
                logger.info(f"Received success confirmation for 'send_message' from '{adapter_id}'. External ID: {external_msg_id}")
                # TODO: Optionally enqueue an internal "agent_message_confirmed" event?
            
            else:
                # Handle success confirmations for other types if needed
                logger.info(f"Received unhandled request_success confirmation for type '{request_type}' from '{adapter_id}'.")

        # --- Handler for Incoming Attachment Data (separate from message events) ---
        # Placeholder - assuming adapter sends attachment data via a specific event
        # Example: @client.on("attachment_data")
        # async def handle_attachment_data(raw_payload: Dict[str, Any]):
        #    ...
        #    self._host_event_loop.enqueue_incoming_event(event_to_enqueue, {})

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
        # is_message_sent = False # Removed history event logic for now
        # message_text = None # Removed history event logic for now
        # conversation_id = None # Removed history event logic for now
        
        # Map payload fields to expected outgoing data structure per action_type
        # Based on user-provided examples.
        try:
            if outgoing_event_type == "send_message":
                outgoing_data = {
                    "conversation_id": payload["conversation_id"], # Required
                    "text": payload["text"],                 # Required
                    # Optional fields if adapters support them
                    # "reply_to_external_id": payload.get("reply_to_external_id") 
                }
                # is_message_sent = True
                # message_text = payload["text"]
                # conversation_id = payload["conversation_id"]
                
            elif outgoing_event_type == "edit_message":
                outgoing_data = {
                    "conversation_id": payload["conversation_id"],     # Required
                    "message_id": payload["message_external_id"], # Required - maps from message_external_id
                    "text": payload["new_text"]                   # Required - maps from new_text
                }
                
            elif outgoing_event_type == "delete_message":
                outgoing_data = {
                    "conversation_id": payload["conversation_id"],     # Required
                    "message_id": payload["message_external_id"]  # Required - maps from message_external_id
                }
                
            elif outgoing_event_type == "add_reaction":
                outgoing_data = {
                    "conversation_id": payload["conversation_id"],     # Required
                    "message_id": payload["message_external_id"], # Required - maps from message_external_id
                    "emoji": payload["emoji"]                     # Required
                }
                
            elif outgoing_event_type == "remove_reaction":
                outgoing_data = {
                    "conversation_id": payload["conversation_id"],     # Required
                    "message_id": payload["message_external_id"], # Required - maps from message_external_id
                    "emoji": payload["emoji"]                     # Required
                }
                
            elif outgoing_event_type == "fetch_history":
                outgoing_data = {
                    "conversation_id": payload["conversation_id"], # Required
                    "before": payload.get("before"),           # Optional
                    "after": payload.get("after"),             # Optional
                    "limit": payload.get("limit", 100)         # Optional, with default
                }
                # Note: No internal history event is generated for the request itself.
                # The response (history) will generate events later.

            elif outgoing_event_type == "get_attachment":
                outgoing_data = {
                    "conversation_id": payload["conversation_id"], # Required
                    "attachment_id": payload["attachment_id"]    # Required
                    # Optional: pass message_id if adapter needs it for context
                    # "message_id": payload.get("message_external_id") 
                }

            else:
                logger.warning(f"Outgoing action type '{outgoing_event_type}' has no specific data mapping. Sending empty data field.")
                outgoing_data = {} # Send empty data if type is unknown

        except KeyError as e:
            logger.error(f"Missing required key '{e}' in payload for action type '{outgoing_event_type}'. Cannot construct outgoing data.")
            return # Abort if payload is missing required fields for the specific action

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
            # Removed for now
            # if is_message_sent and message_text is not None and conversation_id is not None:
            #     history_event_payload = { ... }
            #     history_event = { ... }
            #     self._host_event_loop.enqueue_internal_event(history_event)
                
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