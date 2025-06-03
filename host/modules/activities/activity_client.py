"""
Activity Client Module - Thin I/O Layer

SIMPLIFIED ARCHITECTURE: ActivityClient now serves as a thin I/O layer that handles 
generic adapter communication. All action-specific logic has been moved to 
ExternalEventRouter for better scalability.

Responsibilities:
- Manages WebSocket connections to external adapters
- Dispatches generic actions received from ExternalEventRouter
- Receives adapter responses and routes them back to ExternalEventRouter
- Tracks pending requests for confirmation/failure handling
- Maintains connection status and health

Action-specific logic (validation, formatting, business rules) is handled by:
- ExternalEventRouter (current)
- Future: Specialized routers (MessageRouter, etc.) as the system scales
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
SOCKET_TIMEOUT = 30  # seconds
PENDING_REQUEST_TIMEOUT_SECONDS = 300  # 5 minutes

class ActivityClient:
    """
    Pure thin I/O layer for WebSocket communication with external adapters.
    
    Handles only:
    - WebSocket connection management  
    - Request/response correlation via internal_request_id
    - Raw message dispatching and routing
    
    All business logic, context storage, and action-specific handling is done by ExternalEventRouter.
    """
    
    def __init__(self, host_event_loop: HostEventLoop, adapter_api_configs: List[Dict[str, Any]]):
        self.host_event_loop = host_event_loop
        self.adapters = {}  # adapter_id -> {config, client, connected}
        
        # SIMPLIFIED: Only store minimal I/O correlation data
        # Key: internal_request_id, Value: Basic correlation info for I/O matching only
        self._pending_io_requests: Dict[str, Dict[str, Any]] = {}
        
        self._load_adapter_configs(adapter_api_configs)
        logger.info(f"ActivityClient initialized with {len(self.adapters)} adapter API configs.")

    def _load_adapter_configs(self, adapter_api_configs: List[Dict[str, Any]]) -> None:
        """Load and validate adapter configurations."""
        for config in adapter_api_configs:
            if "id" in config and "url" in config:
                adapter_id = config["id"]
                self.adapters[adapter_id] = {
                    "config": config,
                    "client": None,
                    "connected": False
                }
                logger.debug(f"Loaded adapter config: {adapter_id} -> {config.get('url')}")
            else:
                logger.warning(f"Invalid adapter config (missing 'id' or 'url'): {config}")

    async def connect_to_all_adapters(self) -> bool:
        """Attempt to connect to all configured adapter APIs."""
        if not self.adapters:
            logger.warning("No adapter APIs configured. Skipping connections.")
            return True

        logger.info("Attempting connections to external Adapter APIs...")
        connect_tasks = []
        for adapter_id, adapter_info in self.adapters.items():
            connect_tasks.append(self._connect_to_adapter_api(adapter_id, adapter_info["config"]))
        
        if connect_tasks:
            results = await asyncio.gather(*connect_tasks, return_exceptions=True)
            for i, adapter_id in enumerate(self.adapters.keys()):
                 if isinstance(results[i], Exception):
                      logger.error(f"Failed to connect to adapter '{adapter_id}': {results[i]}")
                 elif results[i]:
                      logger.info(f"Successfully connected to adapter '{adapter_id}'")
                 else:
                      logger.error(f"Failed to connect to adapter '{adapter_id}' (unknown error)")
        
        connected_count = sum(1 for adapter_info in self.adapters.values() if adapter_info["connected"])
        logger.info(f"Connected to {connected_count}/{len(self.adapters)} adapter APIs.")
        return connected_count > 0

    async def _connect_to_adapter_api(self, adapter_id: str, config: Dict[str, Any]) -> bool:
        """Connect to a specific adapter API endpoint using socketio.AsyncClient."""
        url = config.get("url")
        if not url:
            logger.error(f"No URL specified for adapter API '{adapter_id}'")
            return False

        if adapter_id in self.adapters and self.adapters[adapter_id]["connected"]:
             logger.info(f"Already connected to adapter '{adapter_id}'. Skipping.")
             return True
             
        # Connect and set up handlers
        try:
            client = socketio.AsyncClient(
                logger=False,  
                reconnection=True,
                reconnection_attempts=SOCKET_RECONNECTION_ATTEMPTS,
                reconnection_delay=SOCKET_RECONNECTION_DELAY,
                request_timeout=SOCKET_TIMEOUT
            )
            self.adapters[adapter_id]["client"] = client
            self.adapters[adapter_id]["connected"] = False # Assume disconnected until confirmed

            # --- Define Event Handlers ---
            @client.event
            async def connect():
                logger.info(f"Successfully connected to adapter API '{adapter_id}' at {url}")
                self.adapters[adapter_id]["connected"] = True
                # Send registration/hello message if needed by adapter API
                # await self._register_with_adapter_api(adapter_id, config)

            @client.event  
            async def disconnect():
                logger.info(f"Disconnected from adapter API '{adapter_id}'")
                self.adapters[adapter_id]["connected"] = False

            @client.event
            async def connect_error(data):
                logger.error(f"Connection error with adapter API '{adapter_id}': {data}")
                self.adapters[adapter_id]["connected"] = False
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
                self.host_event_loop.enqueue_incoming_event(event_to_enqueue, {})
                
            # --- Handler for Incoming History Response --- 
            # REMOVED @client.on("history_response") handler

            # --- NEW Handler for request_queued from Adapter ---
            @client.on("request_queued")
            async def handle_request_queued(raw_payload: Dict[str, Any]):
                """
                Handles the 'request_queued' event from the adapter.
                This event confirms the adapter has received our request and gives us its internal ID for it.
                It should also echo back the internal_request_id we sent from MessageActionHandler.
                """
                if not isinstance(raw_payload, dict):
                    logger.warning(f"Received non-dict request_queued from '{adapter_id}': {raw_payload}")
                    return

                
                adapter_specific_request_id = raw_payload.get('request_id') # Adapter's own ID for the operation
                # Adapter *must* echo back the internal_request_id we sent with the original bot_response
                mah_internal_request_id_echo = raw_payload.get('internal_request_id') 

                if not adapter_specific_request_id or not mah_internal_request_id_echo:
                    logger.error(f"Received request_queued from '{adapter_id}' missing 'request_id' or 'internal_request_id'. Cannot link. Payload: {raw_payload}")
                    return

                # SIMPLIFIED: Don't create duplicate entries. 
                # The adapter success/failure will use internal_request_id (which we already have stored)
                # We just log the adapter's request_id for debugging but don't need to re-key anything
                pending_info = self._pending_io_requests.get(mah_internal_request_id_echo)
                if pending_info:
                    # Just add the adapter's request ID to the existing entry for debugging
                    pending_info["adapter_request_id"] = adapter_specific_request_id
                    logger.info(f"Request '{mah_internal_request_id_echo}' successfully queued by adapter '{adapter_id}'. Adapter RequestID: '{adapter_specific_request_id}'.")
                else:
                    logger.warning(f"Received request_queued for MAH ID '{mah_internal_request_id_echo}' but no initial pending record found. Possible timing issue or prior error.")

                # No event is enqueued to HostEventLoop from here; this is an intermediate step.
                # The status in MessageListComponent remains 'pending_send'.

            # --- SIMPLIFIED Handler for Generic Success Responses ---
            @client.on("request_success")
            async def handle_request_success(raw_payload: Dict[str, Any]):
                """SIMPLIFIED: Thin I/O layer for adapter success responses. Routes to ExternalEventRouter."""
                logger.critical(f"Received request_success from '{adapter_id}': {raw_payload}")
                if not isinstance(raw_payload, dict):
                     logger.warning(f"Received non-dict request_success from '{adapter_id}': {raw_payload}")
                     return
                 
                adapter_request_id = raw_payload.get('request_id') 
                internal_request_id = raw_payload.get('internal_request_id')  # FIXED: Use internal_request_id for correlation
                data = raw_payload.get('data', {})

                logger.debug(f"ActivityClient (thin I/O) received request_success from '{adapter_id}'. Internal Request ID: {internal_request_id}")

                if not internal_request_id:
                    logger.warning(f"Received request_success from '{adapter_id}' but missing 'internal_request_id'. Cannot correlate. Payload: {raw_payload}")
                    return

                # FIXED: Retrieve our internal tracking info using the internal_request_id (not adapter's request_id)
                pending_request_info = self._pending_io_requests.pop(internal_request_id, None)

                if not pending_request_info:
                    logger.warning(f"Received request_success for internal_request_id '{internal_request_id}' from '{adapter_id}', but no matching pending request found.")
                    return
                
                # Create minimal generic success event for ExternalEventRouter to process
                success_event = {
                    "source_adapter_id": adapter_id,
                    "adapter_type": "unknown",  # We don't track this at I/O level
                    "payload": {
                        "event_type_from_adapter": "adapter_action_success",
                        "adapter_data": {
                            # Pass through essential correlation data
                            "internal_request_id": internal_request_id,
                            "target_element_id_for_confirmation": pending_request_info.get("target_element_id_for_confirmation"),
                            "action_type": pending_request_info.get("action_type"),
                            "conversation_id": pending_request_info.get("conversation_id"),  # NEW: Pass through conversation_id for routing
                            "adapter_request_id": adapter_request_id,
                            # Pass through raw adapter response for ExternalEventRouter to process
                            "raw_adapter_response": raw_payload,
                            "adapter_response_data": data,
                            "confirmed_timestamp": data.get("timestamp", time.time())
                        }
                    }
                }
                
                logger.info(f"ActivityClient routing success for {pending_request_info.get('action_type')} (req_id: {internal_request_id}) to ExternalEventRouter")
                self.host_event_loop.enqueue_incoming_event(success_event, {})

            # --- SIMPLIFIED Handler for Generic Failure Responses ---
            @client.on("request_failed")
            async def handle_request_failure(raw_payload: Dict[str, Any]):
                """SIMPLIFIED: Thin I/O layer for adapter failure responses. Routes to ExternalEventRouter."""
                if not isinstance(raw_payload, dict):
                    logger.warning(f"Received non-dict request_failed from '{adapter_id}': {raw_payload}")
                    return
                    
                adapter_request_id = raw_payload.get('request_id')
                internal_request_id = raw_payload.get('internal_request_id')  # FIXED: Use internal_request_id for correlation
                data = raw_payload.get('data', {})

                logger.debug(f"ActivityClient (thin I/O) received request_failed from '{adapter_id}'. Internal Request ID: {internal_request_id}")

                if not internal_request_id:
                    logger.warning(f"Received request_failed from '{adapter_id}' but missing 'internal_request_id'. Cannot correlate. Payload: {raw_payload}")
                    return

                # FIXED: Retrieve our internal tracking info using the internal_request_id (not adapter's request_id)
                pending_request_info = self._pending_io_requests.pop(internal_request_id, None)

                if not pending_request_info:
                    logger.warning(f"Received request_failed for internal_request_id '{internal_request_id}' from '{adapter_id}', but no matching pending request found.")
                    return
                
                # Extract comprehensive error information from adapter response
                error_message = (
                    data.get('message') or 
                    data.get('error') or 
                    raw_payload.get('error') or 
                    raw_payload.get('message') or 
                    "Unknown error from adapter"
                )
                
                # Create minimal generic failure event for ExternalEventRouter to process
                failure_event = {
                    "source_adapter_id": adapter_id,
                    "adapter_type": "unknown",  # We don't track this at I/O level
                    "payload": {
                        "event_type_from_adapter": "adapter_action_failure",
                        "adapter_data": {
                            # Pass through essential correlation data
                            "internal_request_id": internal_request_id,
                            "target_element_id_for_confirmation": pending_request_info.get("target_element_id_for_confirmation"),
                            "action_type": pending_request_info.get("action_type"),
                            "conversation_id": pending_request_info.get("conversation_id"),  # NEW: Pass through conversation_id for routing
                            "adapter_request_id": adapter_request_id,
                            # Pass through raw adapter response for ExternalEventRouter to process
                            "raw_adapter_response": raw_payload,
                            "adapter_response_data": data,
                            "error_message": error_message,
                            "failed_timestamp": data.get("timestamp", time.time())
                        }
                    }
                }
                
                logger.info(f"ActivityClient routing failure for {pending_request_info.get('action_type')} (req_id: {internal_request_id}) to ExternalEventRouter")
                self.host_event_loop.enqueue_incoming_event(failure_event, {})

            # --- Handler for Incoming Attachment Data (separate from message events) ---
            # Placeholder - assuming adapter sends attachment data via a specific event
            # Example: @client.on("attachment_data")
            # async def handle_attachment_data(raw_payload: Dict[str, Any]):
            #    ...
            #    self._host_event_loop.enqueue_incoming_event(event_to_enqueue, {})
            
            # --- Connect --- 
            auth_data = config.get("auth") # Expect auth dict if needed
            logger.info(f"Connecting to adapter API '{adapter_id}' at {url}...")
            await client.connect(url, auth=auth_data, namespaces=["/"])
            # Note: Connection status set by async connect event handler
            return True # Indicates connection attempt initiated successfully
        except socketio.exceptions.ConnectionError as e:
             logger.error(f"Failed to connect to adapter '{adapter_id}': {e}")
             # Clean up client instance if connection fails definitively?
             if adapter_id in self.adapters: del self.adapters[adapter_id]
             self.adapters[adapter_id]["connected"] = False
             return False
        except Exception as e:
             logger.error(f"Unexpected error during connection attempt to '{adapter_id}': {e}", exc_info=True)
             if adapter_id in self.adapters: del self.adapters[adapter_id]
             self.adapters[adapter_id]["connected"] = False
             return False

    async def handle_outgoing_action(self, action: Dict[str, Any]):
        """
        SIMPLIFIED: Thin I/O layer for dispatching actions to adapters.
        
        ExternalEventRouter now handles all action-specific preprocessing.
        This method focuses purely on adapter communication.
        """
        logger.critical(f"ActivityClient (thin I/O) dispatching action: {action}")
        payload = action.get("payload", {})
        
        # Extract required fields from payload
        target_adapter_id = payload.get("adapter_id")
        internal_action_type = payload.get("action_type")
        internal_request_id = payload.get("internal_request_id")
        requesting_element_id = payload.get("requesting_element_id")
        
        # Basic validation
        if not target_adapter_id:
            logger.error(f"Cannot dispatch action: Missing 'adapter_id' in payload")
            return
        if not internal_action_type:
            logger.error(f"Cannot dispatch action: Missing 'action_type' in payload")
            return
            
        logger.info(f"ActivityClient dispatching {internal_action_type} to adapter '{target_adapter_id}'")
            
        # Check if adapter is connected
        if target_adapter_id not in self.adapters:
            logger.error(f"Adapter '{target_adapter_id}' not found in connected clients")
            return
        if not self.adapters[target_adapter_id]["connected"]:
            logger.error(f"Adapter '{target_adapter_id}' is not connected")
            return

        client = self.adapters[target_adapter_id]["client"]
        
        try:
            # Store minimal pending request for correlation tracking (action-agnostic)
            if internal_request_id:
                # ENHANCED: Include conversation_id for routing confirmations back to correct chat
                conversation_id = payload.get("conversation_id")  # Extract from payload for routing
                
                # MINIMAL: Only store what's needed for I/O correlation + routing
                self._pending_io_requests[internal_request_id] = {
                    "adapter_id": target_adapter_id,
                    "action_type": internal_action_type,  # Basic categorization only
                    "target_element_id_for_confirmation": requesting_element_id,
                    "conversation_id": conversation_id,  # NEW: Store for routing confirmations
                    "request_time": time.monotonic()
                    # NOTE: No action-specific data stored - ExternalEventRouter handles that
                }
                logger.debug(f"Registered pending request {internal_request_id} for {internal_action_type} (thin I/O tracking)")

            # Create adapter payload in the structure the adapter expects
            # Based on working format: { "event_type": "send_message", "internal_request_id": "...", "data": {...} }
            data_payload = payload.copy()
            data_payload.pop("action_type", None)  # Remove internal Connectome field
            data_payload.pop("requesting_element_id", None)  # Remove internal Connectome field
            # Keep internal_request_id in data_payload for adapter tracking
            
            adapter_payload = {
                "event_type": internal_action_type,  # Adapter needs this to know operation type
                "internal_request_id": internal_request_id,  # Top-level for response correlation
                "data": data_payload  # Nested data structure as adapter expects
            }
            
            # Dispatch to adapter using generic event emit
            logger.debug(f"Emitting {internal_action_type} to adapter '{target_adapter_id}': {adapter_payload}")
            
            # Additional connection health check before emit
            if not hasattr(client, 'connected') or not client.connected:
                raise ConnectionError(f"SocketIO client for adapter '{target_adapter_id}' reports not connected (client.connected = {getattr(client, 'connected', 'N/A')})")
            
            # Double-check our connection tracking
            if not self.adapters[target_adapter_id]["connected"]:
                logger.warning(f"Our connection tracking shows adapter '{target_adapter_id}' as disconnected, but proceeding with emit attempt")
            
            # Use "bot_response" as the consistent event name for all outgoing actions
            await client.emit("bot_response", adapter_payload)
            logger.info(f"Successfully dispatched {internal_action_type} to adapter '{target_adapter_id}' (req_id: {internal_request_id})")
            
        except ConnectionError as ce:
            logger.error(f"Connection error dispatching {internal_action_type} to adapter '{target_adapter_id}': {ce}")
            # Mark adapter as disconnected and clean up
            self.adapters[target_adapter_id]["connected"] = False
            if internal_request_id:
                self._pending_io_requests.pop(internal_request_id, None)
                self._emit_connection_failure_event(target_adapter_id, internal_action_type, internal_request_id, requesting_element_id, str(ce))
        except Exception as e:
            logger.error(f"Error dispatching {internal_action_type} to adapter '{target_adapter_id}': {e}", exc_info=True)
            # Remove from pending if it was added, as it didn't reach adapter successfully
            if internal_request_id:
                self._pending_io_requests.pop(internal_request_id, None)
                # Enqueue an immediate failure event to be routed back via ExternalEventRouter
                logger.info(f"Emit failed for {internal_action_type} (req_id: {internal_request_id}). Enqueuing immediate failure event.")
                failure_event = {
                    "event_type": "adapter_action_failure",
                    "adapter_id": target_adapter_id,
                    "adapter_data": {
                        "internal_request_id": internal_request_id,
                        "error_message": f"Failed to emit to adapter: {str(e)}",
                        "failed_timestamp": time.time(),
                        "adapter_response_data": {},
                        "raw_adapter_response": {}
                    }
                }
                self.host_event_loop.enqueue_incoming_event(failure_event, {})
                
    def _emit_connection_failure_event(self, adapter_id: str, action_type: str, internal_request_id: str, requesting_element_id: str, error_message: str):
        """Helper to emit a connection failure event when adapter connection is lost."""
        failure_event = {
            "source_adapter_id": adapter_id,
            "adapter_type": "unknown",
            "payload": {
                "event_type_from_adapter": "adapter_action_failure",
                "adapter_data": {
                    "action_type": action_type,
                    "internal_request_id": internal_request_id,
                    "target_element_id_for_confirmation": requesting_element_id,
                    "error_message": f"Connection lost to adapter: {error_message}",
                    "failed_timestamp": time.time(),
                    "raw_adapter_response": {"connection_error": True},
                    "adapter_response_data": {"connection_lost": True}
                }
            }
        }
        self.host_event_loop.enqueue_incoming_event(failure_event, {})

    async def _cleanup_timed_out_pending_requests(self):
        """Periodically cleans up requests in _pending_io_requests that have timed out."""
        # This should be run as a periodic background task if ActivityClient has its own async loop
        # or called by HostEventLoop periodically.
        now = time.monotonic()
        timed_out_ids = []
        for req_id, info in self._pending_io_requests.items():
            if (now - info["request_time"]) > PENDING_REQUEST_TIMEOUT_SECONDS:
                timed_out_ids.append(req_id)
        
        for req_id in timed_out_ids:
            info = self._pending_io_requests.pop(req_id)
            action_type = info.get("action_type", "unknown")
            target_element_id = info.get("target_element_id_for_confirmation")
            adapter_id = info.get("adapter_id", "unknown")
            
            logger.warning(f"Outgoing request {req_id} (action: {action_type}) timed out waiting for adapter '{adapter_id}' confirmation.")
            
            # Send timeout failure event via ExternalEventRouter
            timeout_failure_event = {
                "source_adapter_id": adapter_id,
                "adapter_type": "unknown",  # We don't track this at I/O level
                "payload": {
                    "event_type_from_adapter": "adapter_action_failure",
                    "adapter_data": {
                        "internal_request_id": req_id,  # Use req_id as internal_request_id
                        "target_element_id_for_confirmation": target_element_id,
                        "action_type": action_type,
                        "error_message": "Request timed out waiting for adapter confirmation.",
                        "failed_timestamp": time.time(),
                        # Pass through timeout info for ExternalEventRouter to process
                        "raw_adapter_response": {"timeout": True},
                        "adapter_response_data": {"timeout_seconds": PENDING_REQUEST_TIMEOUT_SECONDS}
                    }
                }
            }
            self.host_event_loop.enqueue_incoming_event(timeout_failure_event, {})

    async def shutdown(self):
        """Disconnects from all adapter APIs."""
        logger.info("Disconnecting from all Adapter APIs...")
        disconnect_tasks = []
        for adapter_id, adapter_info in self.adapters.items():
             if adapter_info.get("connected") and adapter_info.get("client"):
                  logger.info(f"Disconnecting from adapter API '{adapter_id}'...")
                  disconnect_tasks.append(asyncio.create_task(adapter_info["client"].disconnect()))
             
        if disconnect_tasks:
            results = await asyncio.gather(*disconnect_tasks, return_exceptions=True)
            # Log any errors during disconnection
            adapter_ids = list(self.adapters.keys())
            for i, res in enumerate(results):
                 if isinstance(res, Exception):
                      logger.error(f"Error disconnecting from adapter '{adapter_ids[i]}': {res}")
                      
        self.adapters.clear()
        logger.info("All adapter API connections closed.") 