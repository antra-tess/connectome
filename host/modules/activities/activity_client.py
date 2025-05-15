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
PENDING_REQUEST_TIMEOUT_SECONDS = 60 # NEW: Timeout for internal request tracking

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

        # NEW: For tracking outgoing requests that need confirmation from the adapter
        # Key: Adapter-specific request ID (from adapter's immediate response if any, or a generated one)
        # Value: Dict containing { "internal_request_id", "target_element_id_for_confirmation", 
        #                        "original_conversation_id", "request_time" }
        self._pending_outgoing_requests: Dict[str, Dict[str, Any]] = {}

        # NEW: For linking MAH's internal_request_id to adapter's requestId if they differ
        # Key: MAH's internal_request_id 
        # Value: Adapter's requestId (from request_queued)
        self._mah_to_adapter_request_id_link: Dict[str, str] = {}

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

            data = raw_payload.get('data', {})
            adapter_specific_request_id = data.get('requestId') # Adapter's own ID for the operation
            # Adapter *must* echo back the internal_request_id we sent with the original bot_response
            mah_internal_request_id_echo = data.get('internal_request_id') 

            if not adapter_specific_request_id or not mah_internal_request_id_echo:
                logger.error(f"Received request_queued from '{adapter_id}' missing 'requestId' or 'internal_request_id'. Cannot link. Payload: {raw_payload}")
                return

            # Now we can link MAH's internal_request_id to the adapter's specific_request_id
            # And prepare the entry in _pending_outgoing_requests, keyed by adapter_specific_request_id.
            
            # Retrieve the original pending info that was temporarily keyed by mah_internal_request_id_echo
            # (This assumes handle_outgoing_action initially stores it keyed by mah_internal_request_id)
            original_pending_info = self._pending_outgoing_requests.pop(mah_internal_request_id_echo, None)

            if not original_pending_info:
                logger.warning(f"Received request_queued for MAH ID '{mah_internal_request_id_echo}' but no initial pending record found. Possible timing issue or prior error.")
                return

            # Re-key the pending information with the adapter_specific_request_id
            self._pending_outgoing_requests[adapter_specific_request_id] = original_pending_info
            # Store the link for potential debugging or other uses, though primary key is now adapter_specific_request_id
            self._mah_to_adapter_request_id_link[mah_internal_request_id_echo] = adapter_specific_request_id

            logger.info(f"Request '{mah_internal_request_id_echo}' successfully queued by adapter '{adapter_id}'. Adapter RequestID: '{adapter_specific_request_id}'. Now awaiting final success/failure.")
            # No event is enqueued to HostEventLoop from here; this is an intermediate step.
            # The status in MessageListComponent remains 'pending_send'.

        # --- NEW Handler for Generic Success Responses ---
        @client.on("request_success")
        async def handle_request_success(raw_payload: Dict[str, Any]):
            """Handles generic success responses from the adapter API."""
            if not isinstance(raw_payload, dict):
                 logger.warning(f"Received non-dict request_success from '{adapter_id}': {raw_payload}")
                 return
            
            # For successful message sends, other actions (edit, delete etc.)
            adapter_request_id = raw_payload.get('requestId') # Assuming adapter returns the original requestId
            data = raw_payload.get('data', {}) # The actual content data from the adapter response

            logger.debug(f"Received request_success from '{adapter_id}'. Adapter Request ID: {adapter_request_id}, Data: {data}")

            if not adapter_request_id:
                logger.warning(f"Received request_success from '{adapter_id}' but missing 'requestId'. Cannot correlate. Payload: {raw_payload}")
                # Check if this is a history or attachment response not tied to a specific prior request_id from us
                # This part might need refinement based on how adapters send unsolicited history/attachments vs responses.
                # The previous history/attachment logic was based on specific event types like "connectome_history_received"
                # which would be constructed by this handler if the raw_payload indicated it was such.
                
                # Let's assume for now that general 'request_success' is for prior requests we made.
                # If it's a history or attachment *response* to a fetch/get request we made,
                # it SHOULD have the adapter_request_id from that initial fetch/get action.
                return

            # Retrieve our internal tracking info using the adapter's request_id
            pending_request_info = self._pending_outgoing_requests.pop(adapter_request_id, None)

            if not pending_request_info:
                logger.warning(f"Received request_success for adapter_request_id '{adapter_request_id}' from '{adapter_id}', but no matching pending request found. Might be a late response or for a non-tracked action.")
                # It could also be a success for an action that doesn't need this specific ack routing (e.g. fetch_history that gets data directly)
                # Let's check based on the original action_type if available in pending_request_info, or assume it's a message send ack
                # For now, this handler is primarily for message send success.
                return
            
            # We found a pending request, now process it as a send success
            internal_req_id = pending_request_info["internal_request_id"]
            target_element_id = pending_request_info["target_element_id_for_confirmation"]
            original_conv_id = pending_request_info["original_conversation_id"]
            
            # Extract external message IDs from the adapter's success response data
            external_message_ids = data.get('message_ids') # Assuming adapter returns this for send_message success
            if not external_message_ids or not isinstance(external_message_ids, list):
                logger.warning(f"request_success for send_message (req_id: {internal_req_id}) from '{adapter_id}' missing valid 'message_ids' in data: {data}")
                # We might still proceed to mark it as success but without external IDs if that makes sense
                external_message_ids = [] # Or handle as error if IDs are critical

            logger.info(f"Correlated request_success for send_message. Internal Req ID: {internal_req_id}, Adapter Req ID: {adapter_request_id}. External IDs: {external_message_ids}")

            event_to_enqueue = {
                "source_adapter_id": adapter_id, # The adapter that sent this success ack
                "payload": {
                    "event_type_from_adapter": "adapter_send_success_ack", # For ExternalEventRouter
                    "adapter_data": {
                        "internal_request_id": internal_req_id,
                        "target_element_id_for_confirmation": target_element_id,
                        "conversation_id": original_conv_id, # Original context
                        "external_message_ids": external_message_ids,
                        "confirmed_timestamp": data.get("timestamp", time.time())
                    }
                }
            }
            self._host_event_loop.enqueue_incoming_event(event_to_enqueue, {})

        # --- NEW Handler for Generic Failure Responses ---
        @client.on("request_failure") # Or whatever the adapter calls its general failure event
        async def handle_request_failure(raw_payload: Dict[str, Any]):
            """Handles generic failure responses from the adapter API."""
            if not isinstance(raw_payload, dict):
                logger.warning(f"Received non-dict request_failure from '{adapter_id}': {raw_payload}")
                return

            adapter_request_id = raw_payload.get('requestId')
            data = raw_payload.get('data', {}) # Error details are often in 'data' or a 'message' field
            error_message_from_adapter = data.get('message', "Unknown error from adapter.")
            if not isinstance(error_message_from_adapter, str):
                error_message_from_adapter = str(error_message_from_adapter)

            logger.debug(f"Received request_failure from '{adapter_id}'. Adapter Request ID: {adapter_request_id}, Error: {error_message_from_adapter}")

            if not adapter_request_id:
                logger.warning(f"Received request_failure from '{adapter_id}' but missing 'requestId'. Cannot correlate. Payload: {raw_payload}")
                return

            pending_request_info = self._pending_outgoing_requests.pop(adapter_request_id, None)

            if not pending_request_info:
                logger.warning(f"Received request_failure for adapter_request_id '{adapter_request_id}' from '{adapter_id}', but no matching pending request found.")
                return
            
            internal_req_id = pending_request_info["internal_request_id"]
            target_element_id = pending_request_info["target_element_id_for_confirmation"]
            original_conv_id = pending_request_info["original_conversation_id"]

            logger.info(f"Correlated request_failure. Internal Req ID: {internal_req_id}, Adapter Req ID: {adapter_request_id}. Error: {error_message_from_adapter}")

            event_to_enqueue = {
                "source_adapter_id": adapter_id,
                "payload": {
                    "event_type_from_adapter": "adapter_send_failure_ack", # For ExternalEventRouter
                    "adapter_data": {
                        "internal_request_id": internal_req_id,
                        "target_element_id_for_confirmation": target_element_id,
                        "conversation_id": original_conv_id,
                        "error_message": error_message_from_adapter,
                        "failed_timestamp": data.get("timestamp", time.time())
                    }
                }
            }
            self._host_event_loop.enqueue_incoming_event(event_to_enqueue, {})

        # --- Old request_success handler content (partially relevant for other types) ---
        # The following was the previous content of request_success, parts of which might be 
        # adapted if adapters send history/attachment data via this generic success event
        # and IF those are also tied to a specific adapter_request_id that we tracked.
        # For now, this is commented out as the primary focus is send_message acks.
        
        # request_type = raw_payload.get('request_type') # This was from old structure
        # status = raw_payload.get('status') # Should be "success"
        # data_old = raw_payload.get('data')

        # if status != "success":
        #     logger.warning(f"Received request_success event from '{adapter_id}' but status was not 'success': {status}")

        # if request_type == "history" and isinstance(data_old, dict):
        #     conversation_id = data_old.get('conversation_id')
        #     messages = data_old.get('messages')
        #     if conversation_id and isinstance(messages, list):
        #         event_to_enqueue = { ... "event_type": "connectome_history_received" ... }
        #         self._host_event_loop.enqueue_incoming_event(event_to_enqueue, {})
        # elif request_type == "attachment" and isinstance(data_old, dict):
        #     conversation_id_from_attach = data_old.get('conversation_id')
        #     attachment_id_from_attach = data_old.get('attachment_id')
        #     if conversation_id_from_attach and attachment_id_from_attach:
        #          event_to_enqueue = { ... "event_type": "connectome_attachment_received" ...}
        #          self._host_event_loop.enqueue_incoming_event(event_to_enqueue, {})
        # elif request_type == "send_message":
        #      # This specific case is now handled above by correlating adapter_request_id
        #      pass 
        # else:
        #     logger.info(f"Received unhandled request_success for type '{request_type}' from '{adapter_id}'.")


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
        internal_request_id_from_mah = payload.get("internal_request_id") # Get this from MAH payload
        
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
                        "attachments": payload.get("attachments"),     # Optional
                        "reply_to_external_id": payload.get("reply_to_external_id"), # Optional
                        "internal_request_id": payload.get("internal_request_id") # Required for tracking
                    }
                    
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
                        "limit": payload.get("limit", 20)          # Optional, with default
                    }
                    # The response (history) will generate events later.

            elif outgoing_event_type == "get_attachment":
             outgoing_data = {
                    "conversation_id": payload["conversation_id"], # Required
                    "attachment_id": payload["attachment_id"]    # Required
                }
             
            else: # This covers the case where the action_type is unknown
                logger.warning(f"Outgoing action type '{outgoing_event_type}' has no specific data mapping. Sending empty data field.")
                outgoing_data = {} # Send empty data if type is unknown

        except KeyError as e:
            logger.error(f"Missing required key '{e}' in payload for action type '{outgoing_event_type}'. Cannot construct outgoing data.")
            return # Abort if payload is missing required fields for the specific action
        # The try-except block for KeyError is now complete.

        # Construct the final object to emit to the external adapter API
        # This is the structure the *adapter* expects for its "bot_response" type event.
        # It may or may not include our internal_request_id_from_mah. 
        # The adapter is more likely to return its *own* requestId in its immediate response.
        data_to_emit = {
             "event_type": outgoing_event_type, # e.g., "send_message"
             "data": { 
                 **outgoing_data, 
                 # CRUCIAL: Send our internal_request_id so adapter can echo it in request_queued
                 "internal_request_id_for_echo": internal_request_id_from_mah 
            }
        }
        
        # Emit using the "bot_response" event name to the adapter
        event_name_to_emit = "bot_response"
        logger.debug(f"Sending '{event_name_to_emit}' event to adapter API '{target_adapter_id}' with data: {data_to_emit}")
        
        try:
            # --- Emit Externally and await adapter's immediate response if it provides one ---
            # The adapter should ideally respond with its own request_id for this emission.
            # Let's assume emit can be awaited and might return such a response, or we use a callback.
            # For python-socketio, emit can have a callback for acknowledgment.
            
            adapter_response = await client.emit(event_name_to_emit, data_to_emit) # Adapters might not return directly
            
            # SIMPLIFICATION: For now, we assume the adapter might return an immediate ack with its own requestId.
            # If not, this model is flawed and needs adapter-specific logic for how to get that initial adapter_request_id.
            # Example: If adapter_response = {"requestId": "adapter_req_xyz"}
            
            # For robust tracking, we need the adapter to give us an ID for *this specific emit*.
            # If `client.emit` doesn't directly return this, we might need to:
            # 1. Use `client.call` if the adapter supports request/response for this.
            # 2. Have the adapter immediately emit back a "request_received_ack" with its own ID.
            # 3. Generate a UUID on our side, send it *in* `data_to_emit`, and hope adapter echoes it in its async success/failure.

            # Let's proceed with option 3 for now as it's client-driven, though it requires adapter cooperation.
            # We will add our internal_request_id_from_mah to the data_to_emit, and hope the adapter
            # includes this in its `requestId` field when it sends `request_success` or `request_failure`.
            # This makes `internal_request_id_from_mah` serve as the `adapter_request_id` for correlation.
            # This is a temporary workaround if adapters don't give their own unique ID per request. 
            
            # If the action was 'send_message', we should store it for later confirmation.
            if outgoing_event_type == "send_message" and internal_request_id_from_mah:
                # The key for _pending_outgoing_requests should be what the ADAPTER will use 
                # to identify this request in its async ack (i.e., its `requestId` field).
                # If the adapter simply echoes back our `internal_request_id_from_mah` in its `requestId` field,
                # then we can use `internal_request_id_from_mah` as the key here.
                
                # INITIAL REGISTRATION: Keyed by our internal_request_id_from_mah.
                # This will be re-keyed by adapter's requestId upon receiving "request_queued".
                self._pending_outgoing_requests[internal_request_id_from_mah] = {
                    "internal_request_id": internal_request_id_from_mah, # Storing it again for consistency in value structure
                    "target_element_id_for_confirmation": requesting_element_id,
                    "original_conversation_id": payload.get("conversation_id"),
                    "request_time": time.monotonic()
                }
                logger.info(f"Stored initial pending outgoing 'send_message' with key: {internal_request_id_from_mah} for adapter {target_adapter_id}. Awaiting request_queued from adapter.")
            else:
                logger.debug(f"Action '{outgoing_event_type}' not a send_message or missing internal_request_id. Not adding to pending request tracking for adapter ack.")

            logger.debug(f"Successfully emitted '{event_name_to_emit}' to '{target_adapter_id}'")
                
        except Exception as e:
            logger.error(f"Error emitting action '{action_type}' to adapter '{target_adapter_id}': {e}", exc_info=True)
            # If emit fails, we might want to immediately mark the pending message in MessageListComponent as failed.
            if outgoing_event_type == "send_message" and internal_request_id_from_mah:
                # Remove from pending if it was added, as it didn't even reach adapter successfully
                self._pending_outgoing_requests.pop(internal_request_id_from_mah, None)
                # Enqueue an immediate failure event to be routed back to MessageListComponent
                logger.info(f"Emit failed for send_message (req_id: {internal_request_id_from_mah}). Enqueuing immediate failure.")
                failure_event_to_enqueue = {
                    "source_adapter_id": target_adapter_id,
                    "payload": {
                        "event_type_from_adapter": "adapter_send_failure_ack",
                        "adapter_data": {
                            "internal_request_id": internal_request_id_from_mah,
                            "target_element_id_for_confirmation": requesting_element_id,
                            "conversation_id": payload.get("conversation_id"),
                            "error_message": f"Emit to adapter failed: {e}",
                            "failed_timestamp": time.time()
                        }
                    }
                }
                self._host_event_loop.enqueue_incoming_event(failure_event_to_enqueue, {})

    async def _cleanup_timed_out_pending_requests(self):
        """Periodically cleans up requests in _pending_outgoing_requests that have timed out."""
        # This should be run as a periodic background task if ActivityClient has its own async loop
        # or called by HostEventLoop periodically.
        now = time.monotonic()
        timed_out_ids = []
        for req_id, info in self._pending_outgoing_requests.items():
            if (now - info["request_time"]) > PENDING_REQUEST_TIMEOUT_SECONDS:
                timed_out_ids.append(req_id)
        
        for req_id in timed_out_ids:
            info = self._pending_outgoing_requests.pop(req_id)
            logger.warning(f"Outgoing request {req_id} (internal: {info['internal_request_id']}) timed out waiting for adapter confirmation.")
            # Optionally, send a failure event back to MessageListComponent for these timeouts
            timeout_failure_event = {
                "source_adapter_id": "ActivityClientInternal", # Or derive original adapter if stored
                "payload": {
                    "event_type_from_adapter": "adapter_send_failure_ack",
                    "adapter_data": {
                        "internal_request_id": info["internal_request_id"],
                        "target_element_id_for_confirmation": info["target_element_id_for_confirmation"],
                        "conversation_id": info["original_conversation_id"],
                        "error_message": "Request timed out waiting for adapter confirmation.",
                        "failed_timestamp": time.time()
                    }
                }
            }
            self._host_event_loop.enqueue_incoming_event(timeout_failure_event, {})

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