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
    
    def __init__(self, message_handler):
        """
        Initialize the Socket.IO client manager.
        
        Args:
            message_handler: MessageHandler instance to process incoming messages
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
            # Add adapter info to the data
            data['adapter_id'] = adapter_id
            
            # Call the message handler's handle_message method directly
            result = self.message_handler.handle_message(data)
            
            # Process the result if needed
            if result:
                logger.info(f"Message handled: {result}")
        
        @client.on("clear_context")
        def on_clear_context(data):
            # Add adapter info to the data
            data['adapter_id'] = adapter_id
            
            # Call the message handler's handle_clear_context method directly
            result = self.message_handler.handle_clear_context(data)
            
            # Process the result if needed
            if result:
                logger.info(f"Context cleared: {result}")
        
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
        Send a message to an adapter server.
        
        Args:
            message: Dictionary containing message data including:
                - adapter_id: Identifier for the target adapter
                - event: Event type (defaults to 'bot_response')
                - data: Message data
                
        Returns:
            True if the message was sent successfully, False otherwise
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
    
    def send_response(self, user_id: str, message_text: str, message_id: Optional[str] = None, 
                    platform: Optional[str] = None, adapter_id: Optional[str] = None) -> bool:
        """
        Send a response message to a user through an adapter.
        
        Args:
            user_id: ID of the user to send the response to
            message_text: Content of the response
            message_id: Optional ID for the message (for threading)
            platform: Optional platform identifier
            adapter_id: Optional adapter ID (if not provided, tries to determine from platform)
            
        Returns:
            True if the response was sent successfully, False otherwise
        """
        # Determine adapter ID if not provided
        if not adapter_id and platform:
            # Find adapter for this platform
            for aid, config in self.adapters.items():
                if config.get('platform') == platform:
                    adapter_id = aid
                    break
        
        if not adapter_id:
            logger.error(f"No adapter found for response to user {user_id}")
            return False
        
        # Create response data
        response_data = {
            "adapter_id": adapter_id,
            "event": "bot_response",
            "data": {
                "user_id": user_id,
                "message": message_text,
                "message_id": message_id
            }
        }
        
        # Send the response
        return self.send_message(response_data)
    
    def send_error(self, adapter_id: str, chat_id: str, error_message: str) -> bool:
        """
        Send an error message to an adapter.
        
        Args:
            adapter_id: ID of the adapter
            chat_id: ID of the chat/conversation
            error_message: Error message to send
            
        Returns:
            True if the error was sent successfully, False otherwise
        """
        error_data = {
            "adapter_id": adapter_id,
            "event": "error",
            "data": {
                "chat_id": chat_id,
                "error": error_message
            }
        }
        
        return self.send_message(error_data)
    
    def send_typing_indicator(self, adapter_id: str, chat_id: str, is_typing: bool = True) -> bool:
        """
        Send a typing indicator to an adapter.
        
        Args:
            adapter_id: ID of the adapter
            chat_id: ID of the chat/conversation
            is_typing: Whether the bot is typing (True) or not (False)
            
        Returns:
            True if the indicator was sent successfully, False otherwise
        """
        typing_data = {
            "adapter_id": adapter_id,
            "event": "typing_indicator",
            "data": {
                "chat_id": chat_id,
                "is_typing": is_typing
            }
        }
        
        return self.send_message(typing_data)
    
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