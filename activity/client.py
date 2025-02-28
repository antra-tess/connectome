"""
Socket.IO Client
Manages connections to multiple normalizing adapter Socket.IO servers.
"""

import json
import logging
import socketio
from typing import Dict, Any, List, Optional, Callable

from config import (
    DEFAULT_ADAPTERS, 
    ADDITIONAL_ADAPTERS,
    SOCKET_RECONNECTION_ATTEMPTS,
    SOCKET_RECONNECTION_DELAY,
    SOCKET_TIMEOUT
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SocketIOClient:
    """
    Manages Socket.IO client connections to multiple normalizing adapter servers.
    
    This class is responsible for:
    - Establishing and maintaining connections to adapter servers
    - Sending messages to adapters
    - Registering event handlers for incoming messages
    """
    
    def __init__(self, message_handler: Callable):
        """
        Initialize the Socket.IO client manager.
        
        Args:
            message_handler: Callback function to handle incoming events
        """
        self.message_handler = message_handler
        self.clients: Dict[str, socketio.Client] = {}
        self.adapters: Dict[str, Dict[str, Any]] = {}
        self.connected_adapters: Dict[str, bool] = {}
        
        # Load adapter configurations
        self._load_adapter_configs()
        
        logger.info(f"SocketIOClient initialized with {len(self.adapters)} adapter configurations")
    
    def _load_adapter_configs(self) -> None:
        """
        Load adapter configurations from environment variables or config.
        """
        # Start with default adapters
        for adapter in DEFAULT_ADAPTERS:
            self.adapters[adapter["id"]] = adapter
            self.connected_adapters[adapter["id"]] = False
        
        # Add additional adapters from environment variable if present
        try:
            additional_adapters = json.loads(ADDITIONAL_ADAPTERS)
            if isinstance(additional_adapters, list):
                for adapter in additional_adapters:
                    if "id" in adapter and "url" in adapter:
                        self.adapters[adapter["id"]] = adapter
                        self.connected_adapters[adapter["id"]] = False
                    else:
                        logger.warning(f"Skipping invalid adapter config: {adapter}")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse ADDITIONAL_ADAPTERS: {ADDITIONAL_ADAPTERS}")
    
    def connect_to_adapters(self) -> None:
        """
        Establish connections to all configured adapters.
        """
        for adapter_id, adapter_config in self.adapters.items():
            self._connect_to_adapter(adapter_id, adapter_config)
    
    def _connect_to_adapter(self, adapter_id: str, adapter_config: Dict[str, Any]) -> None:
        """
        Connect to a specific adapter server.
        
        Args:
            adapter_id: Unique identifier for the adapter
            adapter_config: Configuration for the adapter connection
        """
        url = adapter_config.get("url")
        if not url:
            logger.error(f"No URL specified for adapter {adapter_id}")
            return
        
        # Create a new Socket.IO client
        client = socketio.Client(
            reconnection=True,
            reconnection_attempts=SOCKET_RECONNECTION_ATTEMPTS,
            reconnection_delay=SOCKET_RECONNECTION_DELAY,
            request_timeout=SOCKET_TIMEOUT
        )
        
        # Register event handlers
        self._register_event_handlers(client, adapter_id)
        
        # Connect to the adapter server
        try:
            auth = {}
            if "auth_token" in adapter_config and adapter_config["auth_token"]:
                auth["token"] = adapter_config["auth_token"]
            
            client.connect(
                url,
                auth=auth,
                namespaces=["/"],
                wait_timeout=10
            )
            
            # Store the client
            self.clients[adapter_id] = client
            self.connected_adapters[adapter_id] = True
            
            # Register with the adapter
            self._register_with_adapter(adapter_id, adapter_config)
            
            logger.info(f"Connected to adapter {adapter_id} at {url}")
        except Exception as e:
            logger.error(f"Failed to connect to adapter {adapter_id} at {url}: {str(e)}")
    
    def _register_event_handlers(self, client: socketio.Client, adapter_id: str) -> None:
        """
        Register event handlers for a Socket.IO client.
        
        Args:
            client: Socket.IO client instance
            adapter_id: Identifier for the adapter
        """
        @client.event
        def connect():
            logger.info(f"Connected to adapter {adapter_id}")
            self.connected_adapters[adapter_id] = True
        
        @client.event
        def disconnect():
            logger.info(f"Disconnected from adapter {adapter_id}")
            self.connected_adapters[adapter_id] = False
        
        @client.event
        def connect_error(data):
            logger.error(f"Connection error with adapter {adapter_id}: {data}")
            self.connected_adapters[adapter_id] = False
        
        @client.on("chat_message")
        def on_chat_message(data):
            logger.info(f"Received message from adapter {adapter_id}")
            # Add adapter ID to the message data
            if isinstance(data, dict):
                data["adapter_id"] = adapter_id
            # Process the message and get a response
            response = self.message_handler("message", data)
            # If there's a response, send it back
            if response:
                self.send_message(response)
        
        @client.on("clear_context")
        def on_clear_context(data):
            logger.info(f"Received clear context request from adapter {adapter_id}")
            # Add adapter ID to the data
            if isinstance(data, dict):
                data["adapter_id"] = adapter_id
            # Process the request and get a response
            response = self.message_handler("clear_context", data)
            # If there's a response, send it back
            if response:
                self.send_message(response)
        
        @client.on("registration_success")
        def on_registration_success(data):
            logger.info(f"Successfully registered with adapter {adapter_id}: {data}")
        
        @client.on("registration_error")
        def on_registration_error(data):
            logger.error(f"Failed to register with adapter {adapter_id}: {data}")
    
    def _register_with_adapter(self, adapter_id: str, adapter_config: Dict[str, Any]) -> None:
        """
        Register with the adapter after connecting.
        
        Args:
            adapter_id: Identifier for the adapter
            adapter_config: Configuration for the adapter
        """
        if adapter_id not in self.clients or not self.connected_adapters.get(adapter_id, False):
            logger.warning(f"Cannot register with adapter {adapter_id} - not connected")
            return
        
        # Prepare registration data
        registration_data = {
            "bot_id": "bot_framework",
            "bot_name": "Bot Framework",
            "capabilities": [
                "text_messages",
                "multi_user_chat",
                "context_management"
            ]
        }
        
        # Send registration event
        try:
            self.clients[adapter_id].emit("register_bot", registration_data)
            logger.info(f"Sent registration to adapter {adapter_id}")
        except Exception as e:
            logger.error(f"Failed to send registration to adapter {adapter_id}: {str(e)}")
    
    def send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send a message to the appropriate adapter.
        
        Args:
            message: Message data to send, including adapter_id
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        adapter_id = message.get("adapter_id")
        if not adapter_id:
            logger.error("No adapter_id specified in message")
            return False
        
        if adapter_id not in self.clients or not self.connected_adapters.get(adapter_id, False):
            logger.error(f"Cannot send message to adapter {adapter_id} - not connected")
            return False
        
        try:
            # Send the message to the adapter
            self.clients[adapter_id].emit("bot_response", message)
            logger.info(f"Sent message to adapter {adapter_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send message to adapter {adapter_id}: {str(e)}")
            return False
    
    def close_connections(self) -> None:
        """
        Close all adapter connections.
        """
        for adapter_id, client in self.clients.items():
            try:
                client.disconnect()
                logger.info(f"Disconnected from adapter {adapter_id}")
            except Exception as e:
                logger.error(f"Error disconnecting from adapter {adapter_id}: {str(e)}")
        
        self.clients = {}
        self.connected_adapters = {adapter_id: False for adapter_id in self.adapters} 