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
from opentelemetry import trace, propagate, context

import socketio # Using python-socketio

from host.observability import get_tracer

# Event Loop for queuing incoming events
from host.event_loop import HostEventLoop 

# Initialize the tracer for this module
tracer = get_tracer(__name__)

logger = logging.getLogger(__name__)

# Example config defaults (move to config loading later)
SOCKET_RECONNECTION_ATTEMPTS = 3
SOCKET_RECONNECTION_DELAY = 5
SOCKET_TIMEOUT = 30  # seconds
PENDING_REQUEST_TIMEOUT_SECONDS = 300  # 5 minutes

# NEW: Enhanced timeouts for robustness
EMIT_TIMEOUT_SECONDS = 30          # Allow 30s for emit (agent thinking time)
AGENT_RESPONSE_GRACE_PERIOD = 45   # Extra grace for slow agent responses
CONNECTION_HEALTH_CHECK_INTERVAL = 60  # Check connection health every minute

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
        
        # NEW: Keepalive and idle state management
        self._keepalive_interval = 120  # Send keepalive every 2 minutes
        self._idle_threshold = 300  # Consider connection idle after 5 minutes
        self._last_activity: Dict[str, float] = {}  # adapter_id -> timestamp of last activity
        self._keepalive_tasks: Dict[str, asyncio.Task] = {}  # adapter_id -> keepalive task
        
        # NEW: Connection health tracking
        self._connection_health_tasks: Dict[str, asyncio.Task] = {}  # adapter_id -> health check task
        self._successful_operations: Dict[str, int] = {}  # Track successful operations to reset timeout counters
        
        self._load_adapter_configs(adapter_api_configs)
        logger.info(f"ActivityClient initialized with {len(self.adapters)} adapter API configs.")
        logger.info(f"Keepalive interval: {self._keepalive_interval}s, idle threshold: {self._idle_threshold}s")

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
            async def connect(*args):
                logger.info(f"Successfully connected to adapter API '{adapter_id}' at {url}")
                self.adapters[adapter_id]["connected"] = True
                # Record initial adapter signal
                self._record_activity(adapter_id)
                # Start keepalive task for this adapter
                self._start_keepalive(adapter_id)
                # Send registration/hello message if needed by adapter API
                # await self._register_with_adapter_api(adapter_id, config)

            @client.event  
            async def disconnect(*args):
                logger.info(f"Disconnected from adapter API '{adapter_id}'")
                self.adapters[adapter_id]["connected"] = False
                # Stop keepalive task
                self._stop_keepalive(adapter_id)

            @client.event
            async def connect_error(data):
                logger.error(f"Connection error with adapter API '{adapter_id}': {data}")
                self.adapters[adapter_id]["connected"] = False
                # Stop keepalive task
                self._stop_keepalive(adapter_id)
                # No need to disconnect client instance here, library handles retries

            # --- Handler for Incoming Events from Adapter API --- 
            # Renamed from "normalized_event" to "bot_request"
            @client.on("bot_request") 
            async def handle_bot_request(raw_payload: Dict[str, Any]):
                """Handles incoming events (structured as per user spec) from the adapter API."""
                
                # This is the start of a new trace for an unsolicited incoming event.
                with tracer.start_as_current_span("activity_client.handle_bot_request") as span:
                    # Record activity for this outgoing action
                    self._record_activity(adapter_id)
                    logger.critical(f"Received bot_request from '{adapter_id}': {raw_payload}")
                    
                    if not isinstance(raw_payload, dict):
                        logger.warning(f"Received non-dict bot_request from '{adapter_id}': {raw_payload}")
                        span.set_attribute("event.error", "Non-dict payload")
                        return
                    
                    # Extract core fields from the raw payload
                    raw_event_type = raw_payload.get('event_type', 'unknown')
                    span.set_attribute("adapter.event_type", raw_event_type)
                    raw_data = raw_payload.get('data')
                    
                    if not raw_event_type or not isinstance(raw_data, dict):
                        logger.warning(f"Received bot_request from '{adapter_id}' missing 'event_type' or valid 'data': {raw_payload}")
                        span.set_attribute("event.error", "Missing event_type or data")
                        return

                    source_adapter_id_from_data = raw_data.get('adapter_name')
                    if not source_adapter_id_from_data:
                        logger.warning(f"Received bot_request data from '{adapter_id}' missing 'adapter_name': {raw_data}")
                        source_adapter_id_from_data = adapter_id 
                        
                    if source_adapter_id_from_data != adapter_id:
                        logger.warning(f"Adapter name '{source_adapter_id_from_data}' in bot_request data does not match connection ID '{adapter_id}'. Using data value.")

                    logger.debug(f"Received bot_request from '{adapter_id}': Type={raw_event_type}. Enqueuing...")
                    
                    event_to_enqueue = {
                        "source_adapter_id": adapter_id, 
                        "payload": {
                            "event_type_from_adapter": raw_event_type,
                            "adapter_data": raw_data
                        }
                    }
                    
                    # Inject telemetry context for propagation across the event loop
                    carrier = {}
                    propagate.inject(carrier)
                    event_to_enqueue["telemetry_context"] = carrier
                    span.add_event("Injecting new trace context into carrier for event loop")
                    
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
                # Record activity for this outgoing action
                self._record_activity(adapter_id)
                
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
                
                # Record activity for this outgoing action
                self._record_activity(adapter_id)
                
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
                
                # Extract the parent trace context to continue the trace from the specific outgoing action
                parent_carrier = pending_request_info.get("telemetry_context", {})
                parent_ctx = propagate.extract(parent_carrier)
                
                with tracer.start_as_current_span("activity_client.handle_request_success", context=parent_ctx) as span:
                    span.set_attribute("adapter.request_id", adapter_request_id)
                    span.set_attribute("adapter.internal_request_id", internal_request_id)
                    
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
                    
                    # Inject this continued trace's context for downstream processing
                    carrier = {}
                    propagate.inject(carrier)
                    success_event["telemetry_context"] = carrier
                    span.add_event("Injecting continued trace context into carrier for event loop")
                    
                    logger.info(f"ActivityClient routing success for {pending_request_info.get('action_type')} (req_id: {internal_request_id}) to ExternalEventRouter")
                    # Note: We are no longer propagating context to the event loop from here
                    self.host_event_loop.enqueue_incoming_event(success_event, {})
                    
                    # NEW: Track successful response to improve connection health assessment
                    self._record_successful_operation(adapter_id)

            # --- SIMPLIFIED Handler for Generic Failure Responses ---
            @client.on("request_failed")
            async def handle_request_failure(raw_payload: Dict[str, Any]):
                """SIMPLIFIED: Thin I/O layer for adapter failure responses. Routes to ExternalEventRouter."""
                
                # Record activity for this outgoing action
                self._record_activity(adapter_id)
                
                if not isinstance(raw_payload, dict):
                    logger.warning(f"Received non-dict request_failed from '{adapter_id}': {raw_payload}")
                    return
                    
                adapter_request_id = raw_payload.get('request_id')
                internal_request_id = raw_payload.get('internal_request_id')  # FIXED: Use internal_request_id for correlation
                data = raw_payload.get('data') or {}  # Ensure data is always a dict, not None

                logger.debug(f"ActivityClient (thin I/O) received request_failed from '{adapter_id}'. Internal Request ID: {internal_request_id}")

                if not internal_request_id:
                    logger.warning(f"Received request_failed from '{adapter_id}' but missing 'internal_request_id'. Cannot correlate. Payload: {raw_payload}")
                    return

                # FIXED: Retrieve our internal tracking info using the internal_request_id (not adapter's request_id)
                pending_request_info = self._pending_io_requests.pop(internal_request_id, None)

                if not pending_request_info:
                    logger.warning(f"Received request_failed for internal_request_id '{internal_request_id}' from '{adapter_id}', but no matching pending request found.")
                    return
                
                # Extract the parent trace context to continue the trace
                parent_carrier = pending_request_info.get("telemetry_context", {})
                parent_ctx = propagate.extract(parent_carrier)

                with tracer.start_as_current_span("activity_client.handle_request_failure", context=parent_ctx) as span:
                    span.set_attribute("adapter.request_id", adapter_request_id)
                    span.set_attribute("adapter.internal_request_id", internal_request_id)

                    # Extract comprehensive error information from adapter response
                    error_message = (
                        data.get('message') or 
                        data.get('error') or 
                        raw_payload.get('error') or 
                        raw_payload.get('message') or 
                        "Unknown error from adapter"
                    )
                    span.set_attribute("error.message", error_message)
                    
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
                    
                    # Inject this continued trace's context for downstream processing
                    carrier = {}
                    propagate.inject(carrier)
                    failure_event["telemetry_context"] = carrier
                    span.add_event("Injecting continued trace context into carrier for event loop")
                    
                    logger.info(f"ActivityClient routing failure for {pending_request_info.get('action_type')} (req_id: {internal_request_id}) to ExternalEventRouter")
                    # Note: We are no longer propagating context to the event loop from here
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
        payload = action.get("payload", {})
        target_adapter_id = payload.get("adapter_id")
        internal_action_type = payload.get("action_type")

        with tracer.start_as_current_span("activity_client.handle_outgoing_action", attributes={
            "adapter.id": target_adapter_id,
            "action.type": internal_action_type
        }) as span:
            # Extract required fields from payload
            internal_request_id = payload.get("internal_request_id")
            requesting_element_id = payload.get("requesting_element_id")
            
            # Basic validation
            if not target_adapter_id:
                logger.error(f"Cannot dispatch action: Missing 'adapter_id' in payload")
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Missing adapter_id"))
                return
            if not internal_action_type:
                logger.error(f"Cannot dispatch action: Missing 'action_type' in payload")
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Missing action_type"))
                return
                
            logger.info(f"ActivityClient dispatching {internal_action_type} to adapter '{target_adapter_id}'")
            span.add_event("Dispatching action")
                
            # Check if adapter is connected
            if target_adapter_id not in self.adapters:
                logger.error(f"Adapter '{target_adapter_id}' not found in connected clients")
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Adapter not found"))
                return
            if not self.adapters[target_adapter_id]["connected"]:
                logger.error(f"Adapter '{target_adapter_id}' is not connected")
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Adapter not connected"))
                return

            client = self.adapters[target_adapter_id]["client"]
            
            try:
                # NEW: Check for idle connection and proactively refresh if needed
                current_time = time.time()
                last_activity = self._last_activity.get(target_adapter_id, 0)
                idle_duration = current_time - last_activity
                
                if idle_duration > self._idle_threshold:
                    logger.warning(f"Adapter '{target_adapter_id}' has been idle for {idle_duration:.1f}s (threshold: {self._idle_threshold}s). Proactively refreshing connection before send.")
                    
                    # Attempt to refresh the connection proactively
                    refresh_success = await self._refresh_idle_connection(target_adapter_id)
                    if not refresh_success:
                        logger.error(f"Failed to refresh idle connection for adapter '{target_adapter_id}'")
                        if internal_request_id:
                            self._pending_io_requests.pop(internal_request_id, None)
                            self._emit_connection_failure_event(target_adapter_id, internal_action_type, internal_request_id, requesting_element_id, "Failed to refresh idle connection")
                        return
                
                # Record activity for this outgoing action
                self._record_activity(target_adapter_id)
                
                # Store minimal pending request for correlation tracking (action-agnostic)
                if internal_request_id:
                    # ENHANCED: Include conversation_id for routing confirmations back to correct chat
                    conversation_id = payload.get("conversation_id")  # Extract from payload for routing
                    
                    # Manually capture and store the trace context for this specific request
                    carrier = {}
                    propagate.inject(carrier)
                    
                    # MINIMAL: Only store what's needed for I/O correlation + routing
                    self._pending_io_requests[internal_request_id] = {
                        "adapter_id": target_adapter_id,
                        "action_type": internal_action_type,  # Basic categorization only
                        "target_element_id_for_confirmation": requesting_element_id,
                        "conversation_id": conversation_id,  # NEW: Store for routing confirmations
                        "request_time": time.monotonic(),
                        "telemetry_context": carrier # Store the context
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
                
                # Enhanced connection health checks before emit
                if not hasattr(client, 'connected') or not client.connected:
                    raise ConnectionError(f"SocketIO client for adapter '{target_adapter_id}' reports not connected (client.connected = {getattr(client, 'connected', 'N/A')})")
                
                # NEW: Comprehensive queue and connection state validation
                connection_issues = []
                
                # Check engine.io connection state if available
                if hasattr(client, 'eio') and client.eio:
                    if hasattr(client.eio, 'state') and client.eio.state != 'connected':
                        connection_issues.append(f"Engine.IO state is '{client.eio.state}', not 'connected'")
                    
                    # Check packet queue state more thoroughly
                    if hasattr(client.eio, 'queue'):
                        if hasattr(client.eio.queue, 'empty') and client.eio.queue.empty():
                            connection_issues.append("Engine.IO packet queue is empty")
                        elif hasattr(client.eio.queue, 'qsize') and client.eio.queue.qsize() == 0:
                            connection_issues.append("Engine.IO packet queue size is 0")
                        
                        # Check if transport is available and healthy
                        if hasattr(client.eio, 'transport') and client.eio.transport:
                            if hasattr(client.eio.transport, 'state') and client.eio.transport.state != 'open':
                                connection_issues.append(f"Transport state is '{client.eio.transport.state}', not 'open'")
                        else:
                            connection_issues.append("Engine.IO transport is missing or None")
                
                # If we detected connection issues, attempt proactive refresh
                if connection_issues:
                    logger.warning(f"Detected connection issues for adapter '{target_adapter_id}': {'; '.join(connection_issues)}. Attempting connection refresh.")
                    
                    # Try to refresh the connection
                    refresh_success = await self._refresh_idle_connection(target_adapter_id, force=True)
                    if not refresh_success:
                        error_msg = f"Connection validation failed and refresh unsuccessful: {'; '.join(connection_issues)}"
                        raise ConnectionError(error_msg)
                    
                    # Update client reference after refresh
                    client = self.adapters[target_adapter_id]["client"]
                    logger.info(f"Connection refreshed successfully for adapter '{target_adapter_id}', proceeding with emit")
                
                # Double-check our connection tracking
                if not self.adapters[target_adapter_id]["connected"]:
                    logger.warning(f"Our connection tracking shows adapter '{target_adapter_id}' as disconnected, but proceeding with emit attempt")
                
                # Dispatch to adapter using generic event emit
                logger.debug(f"Emitting {internal_action_type} to adapter '{target_adapter_id}': {adapter_payload}")
                span.add_event("Emitting to adapter")
                
                # Enhanced emit with timeout and retry logic
                try:
                    # NEW: Use longer timeout for emit to accommodate agent thinking time
                    await asyncio.wait_for(
                        client.emit("bot_response", adapter_payload),
                        timeout=EMIT_TIMEOUT_SECONDS  # Increased from 10s to 30s
                    )
                    logger.info(f"Successfully dispatched {internal_action_type} to adapter '{target_adapter_id}' (req_id: {internal_request_id})")
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    
                    # NEW: Track successful operation to reset timeout counters
                    self._record_successful_operation(target_adapter_id)
                except asyncio.TimeoutError:
                    # ENHANCED: More nuanced timeout handling
                    logger.warning(f"Emit timeout for {internal_action_type} to adapter '{target_adapter_id}' after {EMIT_TIMEOUT_SECONDS}s")
                    
                    # Check if this is a pattern (multiple timeouts) or isolated incident
                    timeout_count = getattr(self, f'_timeout_count_{target_adapter_id}', 0) + 1
                    setattr(self, f'_timeout_count_{target_adapter_id}', timeout_count)
                    
                    # NEW: Check recent success history to determine if this is truly a connection issue
                    recent_successes = self._successful_operations.get(target_adapter_id, 0)
                    
                    if timeout_count >= 3 and recent_successes < 2:
                        # Multiple timeouts with few recent successes suggest real connection issue
                        logger.error(f"Multiple emit timeouts ({timeout_count}) with low success rate for adapter '{target_adapter_id}' - marking as disconnected")
                        self.adapters[target_adapter_id]["connected"] = False
                        raise ConnectionError(f"Multiple emit timeouts - connection to adapter '{target_adapter_id}' unstable")
                    elif timeout_count >= 5:
                        # Hard limit: too many timeouts regardless of successes
                        logger.error(f"Excessive emit timeouts ({timeout_count}) for adapter '{target_adapter_id}' - marking as disconnected")
                        self.adapters[target_adapter_id]["connected"] = False
                        raise ConnectionError(f"Excessive emit timeouts - connection to adapter '{target_adapter_id}' unstable")
                    else:
                        # Be more tolerant if we've had recent successes
                        tolerance_reason = f"recent successes: {recent_successes}" if recent_successes > 0 else "within tolerance"
                        logger.info(f"Emit timeout ({timeout_count}) for adapter '{target_adapter_id}' - continuing ({tolerance_reason})")
                        # Don't raise ConnectionError immediately, let the pending request timeout handle it
                        logger.info(f"Dispatched {internal_action_type} to adapter '{target_adapter_id}' (req_id: {internal_request_id}) - timeout tolerance applied")
                    
                except (socketio.exceptions.DisconnectedError, socketio.exceptions.ConnectionError) as se:
                    # SocketIO specific connection errors
                    logger.error(f"SocketIO connection error during emit to adapter '{target_adapter_id}': {se}")
                    self.adapters[target_adapter_id]["connected"] = False
                    raise ConnectionError(f"SocketIO connection lost during emit: {se}")
                except Exception as emit_error:
                    # Check if it's a packet queue error or similar low-level issue
                    error_str = str(emit_error).lower()
                    if 'packet queue' in error_str or 'queue is empty' in error_str:
                        logger.error(f"Packet queue error for adapter '{target_adapter_id}': {emit_error}")
                        # Mark adapter as disconnected to trigger reconnection
                        self.adapters[target_adapter_id]["connected"] = False
                        raise ConnectionError(f"Packet queue error - connection to adapter '{target_adapter_id}' unstable: {emit_error}")
                    else:
                        # Re-raise other emit errors
                        raise emit_error

            except ConnectionError as ce:
                logger.error(f"Connection error dispatching {internal_action_type} to adapter '{target_adapter_id}': {ce}")
                span.record_exception(ce)
                span.set_status(trace.Status(trace.StatusCode.ERROR, "ConnectionError"))
                # Mark adapter as disconnected and clean up
                self.adapters[target_adapter_id]["connected"] = False
                if internal_request_id:
                    self._pending_io_requests.pop(internal_request_id, None)
                    self._emit_connection_failure_event(target_adapter_id, internal_action_type, internal_request_id, requesting_element_id, str(ce))
                
                # Try to reconnect if it's a connection stability issue
                logger.info(f"Attempting to reconnect to adapter '{target_adapter_id}' after connection error...")
                asyncio.create_task(self._attempt_reconnect(target_adapter_id))
                
            except Exception as e:
                logger.error(f"Error dispatching {internal_action_type} to adapter '{target_adapter_id}': {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Dispatch error"))
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
        
        # Stop all keepalive tasks first
        for adapter_id in list(self._keepalive_tasks.keys()):
            self._stop_keepalive(adapter_id)
        
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
        self._last_activity.clear()
        self._keepalive_tasks.clear()
        
        # NEW: Clean up connection health tracking
        self._successful_operations.clear()
        # Clear timeout counters
        for attr_name in list(vars(self).keys()):
            if attr_name.startswith('_timeout_count_'):
                delattr(self, attr_name)
                
        logger.info("All adapter API connections closed.")

    async def _attempt_reconnect(self, adapter_id: str, max_attempts: int = 3, delay: float = 2.0):
        """
        Attempt to reconnect to an adapter that experienced connection issues.
        
        Args:
            adapter_id: ID of the adapter to reconnect to
            max_attempts: Maximum number of reconnection attempts
            delay: Delay between attempts in seconds
        """
        if adapter_id not in self.adapters:
            logger.error(f"Cannot reconnect to adapter '{adapter_id}': not in adapter list")
            return
        
        adapter_info = self.adapters[adapter_id]
        config = adapter_info.get("config")
        if not config:
            logger.error(f"Cannot reconnect to adapter '{adapter_id}': missing config")
            return
        
        logger.info(f"Starting reconnection attempts for adapter '{adapter_id}' (max {max_attempts} attempts)")
        
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Reconnection attempt {attempt}/{max_attempts} for adapter '{adapter_id}'")
                
                # Disconnect existing client if still connected
                if adapter_info.get("client"):
                    try:
                        await adapter_info["client"].disconnect()
                        await asyncio.sleep(0.5)  # Brief pause for clean disconnect
                    except Exception as disconnect_error:
                        logger.debug(f"Error during disconnect cleanup for '{adapter_id}': {disconnect_error}")
                
                # Reset connection state
                adapter_info["client"] = None
                adapter_info["connected"] = False
                
                # Attempt reconnection
                success = await self._connect_to_adapter_api(adapter_id, config)
                if success:
                    logger.info(f"Successfully reconnected to adapter '{adapter_id}' on attempt {attempt}")
                    return True
                else:
                    logger.warning(f"Reconnection attempt {attempt}/{max_attempts} failed for adapter '{adapter_id}'")
                    
            except Exception as reconnect_error:
                logger.error(f"Error during reconnection attempt {attempt} for adapter '{adapter_id}': {reconnect_error}")
            
            # Wait before next attempt (except on last attempt)
            if attempt < max_attempts:
                logger.info(f"Waiting {delay} seconds before next reconnection attempt for adapter '{adapter_id}'")
                await asyncio.sleep(delay)
                delay *= 1.5  # Exponential backoff
        
        logger.error(f"Failed to reconnect to adapter '{adapter_id}' after {max_attempts} attempts")
        return False 

    def _record_activity(self, adapter_id: str):
        """Record activity timestamp for an adapter to track idle state."""
        self._last_activity[adapter_id] = time.time()

    def _start_keepalive(self, adapter_id: str):
        """Start keepalive task for an adapter to prevent idle timeout."""
        # Stop existing task if any
        self._stop_keepalive(adapter_id)
        
        # Start new keepalive task
        task = asyncio.create_task(self._keepalive_loop(adapter_id))
        self._keepalive_tasks[adapter_id] = task
        logger.debug(f"Started keepalive task for adapter '{adapter_id}'")

    def _stop_keepalive(self, adapter_id: str):
        """Stop keepalive task for an adapter."""
        if adapter_id in self._keepalive_tasks:
            task = self._keepalive_tasks.pop(adapter_id)
            if not task.done():
                task.cancel()
            logger.debug(f"Stopped keepalive task for adapter '{adapter_id}'")

    async def _keepalive_loop(self, adapter_id: str):
        """Keepalive task that sends keepalive messages to prevent idle timeout."""
        try:
            while True:
                await asyncio.sleep(self._keepalive_interval)
                
                # Check if adapter is still connected
                if (adapter_id not in self.adapters or 
                    not self.adapters[adapter_id].get("connected") or 
                    not self.adapters[adapter_id].get("client")):
                    logger.debug(f"Stopping keepalive task for adapter '{adapter_id}' - no longer connected")
                    break
                
                try:
                    client = self.adapters[adapter_id]["client"]
                    
                    # Send lightweight ping to keep connection alive
                    await client.emit("ping", {"timestamp": time.time(), "keepalive": True})
                    logger.debug(f"Sent keepalive ping to adapter '{adapter_id}'")
                    
                    # Record this as activity
                    self._record_activity(adapter_id)
                    
                except Exception as e:
                    logger.warning(f"Keepalive ping failed for adapter '{adapter_id}': {e}")
                    # Don't break the loop immediately - connection might recover
                    
        except asyncio.CancelledError:
            logger.debug(f"Keepalive task for adapter '{adapter_id}' was cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in keepalive loop for adapter '{adapter_id}': {e}")

    async def _refresh_idle_connection(self, adapter_id: str, force: bool = False) -> bool:
        """
        Refresh an idle connection to prevent packet queue issues.
        
        Args:
            adapter_id: ID of the adapter to refresh
            force: If True, refresh even if not technically idle
            
        Returns:
            True if refresh was successful, False otherwise
        """
        if adapter_id not in self.adapters:
            logger.error(f"Cannot refresh connection for unknown adapter '{adapter_id}'")
            return False
        
        adapter_info = self.adapters[adapter_id]
        config = adapter_info.get("config")
        
        if not config:
            logger.error(f"Cannot refresh connection for adapter '{adapter_id}': missing config")
            return False
        
        # Check if refresh is needed (unless forced)
        if not force:
            current_time = time.time()
            last_activity = self._last_activity.get(adapter_id, 0)
            idle_duration = current_time - last_activity
            
            if idle_duration < self._idle_threshold:
                logger.debug(f"Adapter '{adapter_id}' not idle ({idle_duration:.1f}s < {self._idle_threshold}s), skipping refresh")
                return True
        
        logger.info(f"Refreshing {'idle' if not force else 'problematic'} connection for adapter '{adapter_id}'")
        
        try:
            # Stop keepalive task during refresh
            self._stop_keepalive(adapter_id)
            
            # Gracefully disconnect existing client
            old_client = adapter_info.get("client")
            if old_client:
                try:
                    await old_client.disconnect()
                    await asyncio.sleep(0.5)  # Brief pause for clean disconnect
                except Exception as disconnect_error:
                    logger.debug(f"Error during disconnect in refresh for '{adapter_id}': {disconnect_error}")
            
            # Reset state
            adapter_info["client"] = None
            adapter_info["connected"] = False
            
            # Reconnect
            success = await self._connect_to_adapter_api(adapter_id, config)
            if success:
                logger.info(f"Successfully refreshed connection for adapter '{adapter_id}'")
                # Activity and keepalive are started in connect event handler
                return True
            else:
                logger.error(f"Failed to refresh connection for adapter '{adapter_id}'")
                return False
                
        except Exception as refresh_error:
            logger.error(f"Error during connection refresh for adapter '{adapter_id}': {refresh_error}")
            return False 

    def _record_successful_operation(self, adapter_id: str):
        """Record a successful operation for an adapter to reset timeout counters."""
        if adapter_id in self._successful_operations:
            self._successful_operations[adapter_id] += 1
        else:
            self._successful_operations[adapter_id] = 1
            
        # Reset timeout counter if we have enough recent successes
        if self._successful_operations[adapter_id] >= 3:
            if hasattr(self, f'_timeout_count_{adapter_id}'):
                old_count = getattr(self, f'_timeout_count_{adapter_id}', 0)
                if old_count > 0:
                    logger.info(f"Resetting timeout counter for adapter '{adapter_id}' after {self._successful_operations[adapter_id]} successful operations")
                setattr(self, f'_timeout_count_{adapter_id}', 0)
                # Reset success counter to prevent it from growing indefinitely
                self._successful_operations[adapter_id] = 0 