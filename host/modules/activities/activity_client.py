"""
Activity Client (Formerly Socket.IO Client)

Sends standardized outgoing messages/actions to the external adapter API.
(Currently uses Socket.IO based on original code, but could be adapted 
 to REST, gRPC, etc. based on the API contract with external adapters).
"""

import json
import logging
import socketio # Keep dependency for now based on original code
from typing import Dict, Any, List, Optional, Callable

# TODO: Replace with proper config loading
# from config import (
#     DEFAULT_ADAPTERS, 
#     ADDITIONAL_ADAPTERS,
#     SOCKET_RECONNECTION_ATTEMPTS,
#     SOCKET_RECONNECTION_DELAY,
#     SOCKET_TIMEOUT
# )
# Placeholder config values
DEFAULT_ADAPTERS = [] 
ADDITIONAL_ADAPTERS = "[]"
SOCKET_RECONNECTION_ATTEMPTS = 3
SOCKET_RECONNECTION_DELAY = 5
SOCKET_TIMEOUT = 10

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ActivityClient:
    """
    Sends standardized outgoing events to the external adapter API.
    Assumes external adapters expose an API (e.g., Socket.IO endpoint) to receive these.
    """
    
    def __init__(self, adapter_configs: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the Activity Client.
        
        Args:
            adapter_configs: List of configurations for known external adapter APIs.
                             Each dict should contain at least 'id' and 'url'.
        """
        # Note: Removed dependency on message_handler, client only sends.
        self.clients: Dict[str, socketio.Client] = {}
        self.adapters: Dict[str, Dict[str, Any]] = {}
        self.connected_adapters: Dict[str, bool] = {}
        
        # Load adapter configurations
        self._load_adapter_configs(adapter_configs)
        
        logger.info(f"ActivityClient initialized with {len(self.adapters)} adapter API configurations")
    
    def _load_adapter_configs(self, adapter_configs: Optional[List[Dict[str, Any]]]) -> None:
        """
        Load adapter API configurations.
        """
        configs_to_load = adapter_configs or []
        # TODO: Add logic for loading DEFAULT_ADAPTERS and ADDITIONAL_ADAPTERS if needed
            
        for adapter in configs_to_load:
             if "id" in adapter and "url" in adapter: # Basic validation
                  adapter_id = adapter["id"]
                  self.adapters[adapter_id] = adapter
                  self.connected_adapters[adapter_id] = False
                  logger.debug(f"Loaded adapter config: {adapter_id}") # Fixed f-string
             else:
                  logger.warning(f"Skipping invalid adapter config: {adapter}")

    def connect_to_adapter_apis(self) -> None:
        """
        Establish connections to all configured adapter APIs.
        """
        for adapter_id, adapter_config in self.adapters.items():
            self._connect_to_adapter_api(adapter_id, adapter_config)
    
    def _connect_to_adapter_api(self, adapter_id: str, adapter_config: Dict[str, Any]) -> None:
        """
        Connect to a specific adapter API endpoint.
        (Currently assumes Socket.IO based on original code)
        
        Args:
            adapter_id: Unique identifier for the adapter API.
            adapter_config: Configuration for the connection (e.g., URL, auth).
        """
        url = adapter_config.get("url")
        if not url:
            logger.error(f"No URL specified for adapter API {adapter_id}")
            return
        
        # --- Assuming Socket.IO based on original --- 
        # TODO: Adapt this if the API contract uses REST, gRPC, etc.
        client = socketio.Client(
            reconnection=True,
            reconnection_attempts=SOCKET_RECONNECTION_ATTEMPTS,
            reconnection_delay=SOCKET_RECONNECTION_DELAY,
            request_timeout=SOCKET_TIMEOUT
        )
        
        # Register basic connection/disconnection handlers for status tracking
        @client.event
        def connect():
            logger.info(f"Connected to adapter API {adapter_id} at {url}")
            self.connected_adapters[adapter_id] = True
            # Send registration/hello message?
            self._register_with_adapter_api(adapter_id, adapter_config)

        @client.event
        def disconnect():
            logger.info(f"Disconnected from adapter API {adapter_id}")
            self.connected_adapters[adapter_id] = False
        
        @client.event
        def connect_error(data):
            logger.error(f"Connection error with adapter API {adapter_id}: {data}")
            self.connected_adapters[adapter_id] = False
        # -------------------------------------------
        
        # Connect to the adapter API endpoint
        try:
            auth = {}
            if "auth_token" in adapter_config and adapter_config["auth_token"]:
                auth["token"] = adapter_config["auth_token"]
            
            logger.info(f"Attempting connection to adapter API {adapter_id} at {url}...")
            client.connect( url, auth=auth, namespaces=["/"], wait_timeout=10 )
            self.clients[adapter_id] = client 
            # Connection status set by event handler upon success
            
        except Exception as e:
            logger.error(f"Failed initial connect attempt to adapter API {adapter_id} at {url}: {str(e)}")
            self.connected_adapters[adapter_id] = False
    
    def _register_with_adapter_api(self, adapter_id: str, adapter_config: Dict[str, Any]) -> None:
        """
        Send a registration or hello message after connecting if needed by API contract.
        (Currently assumes Socket.IO based on original code)
        """
        if adapter_id not in self.clients or not self.connected_adapters.get(adapter_id, False):
            logger.warning(f"Cannot register with adapter API {adapter_id} - not connected")
            return
        
        # Example registration data (adjust based on actual API contract)
        registration_data = {
            "client_id": "connectome_host_process",
            "client_type": "core_system",
            "capabilities": ["receive_normalized_events", "send_standardized_actions"]
        }
        
        try:
            # Assuming Socket.IO emit
            self.clients[adapter_id].emit("register_client", registration_data)
            logger.info(f"Sent registration to adapter API {adapter_id}")
        except Exception as e:
            logger.error(f"Failed to send registration to adapter API {adapter_id}: {str(e)}")
    
    def send_event_to_adapter(self, event: Dict[str, Any]) -> bool:
        """
        Sends a standardized event object to the specified external adapter API.
        
        Args:
            event: Standardized event data, including an 'adapter_id' field 
                   indicating which external adapter API to target.
                   Example format (adjust based on actual contract):
                   {
                       "event_type": "send_message", # or "edit_message", "typing_indicator" etc.
                       "payload": { ... }, # Standardized payload
                       "adapter_id": "discord_adapter_1" 
                   }
                
        Returns:
            True if the event was sent successfully, False otherwise.
        """
        adapter_id = event.get("adapter_id")
        if not adapter_id:
            logger.error("No adapter_id specified in event to send")
            return False
        
        if adapter_id not in self.clients or not self.connected_adapters.get(adapter_id, False):
            logger.error(f"Cannot send event to adapter API {adapter_id} - not connected")
            return False
        
        try:
            # --- Assuming Socket.IO based on original --- 
            # TODO: Adapt this if the API contract uses REST, gRPC, etc.
            # The event name ("standard_event_from_core"?) depends on the API contract
            self.clients[adapter_id].emit("standard_event_from_core", event)
            # -------------------------------------------
            logger.debug(f"Sent event {event.get('event_type')} to adapter API {adapter_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send event to adapter API {adapter_id}: {str(e)}")
            return False
            
    # Removed specific methods like send_response, send_error, send_typing_indicator
    # Replaced by the generic send_event_to_adapter which expects standardized events.

    def close_connections(self) -> None:
        """Close all active connections to adapter APIs."""
        logger.info("Closing connections to adapter APIs...")
        for adapter_id, client in self.clients.items():
            if self.connected_adapters.get(adapter_id, False):
                try:
                    client.disconnect()
                    logger.info(f"Disconnected from adapter API {adapter_id}")
                except Exception as e:
                    logger.error(f"Error disconnecting from adapter API {adapter_id}: {str(e)}")
        self.clients.clear()
        self.connected_adapters.clear() 