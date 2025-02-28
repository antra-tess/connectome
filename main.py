#!/usr/bin/env python3
"""
Bot Framework Main Entry Point
Initializes the Socket.IO clients and connects to normalizing adapters.
"""

import signal
import sys
import time
import logging
from threading import Event

from config import DEFAULT_ADAPTERS
from utils.llm import initialize_litellm
# Agent functionality is now part of InterfaceLayer
from activity.client import SocketIOClient
from activity.listener import create_message_handler
from activity.sender import initialize_sender

# Import environment-related components
from environments.base import Environment
from environments.system import SystemEnvironment
from environments.web import WebEnvironment
from environments.messaging import MessagingEnvironment
from environments.file import FileEnvironment
from environments.manager import EnvironmentManager
from interface.layer import InterfaceLayer
# ContextEnvironment is now handled by InterfaceLayer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_event = Event()


def signal_handler(sig, frame):
    """
    Handle shutdown signals.
    """
    logger.info("Shutdown signal received, closing connections...")
    shutdown_event.set()


def initialize_environments():
    """
    Initialize the environment system.
    
    Returns:
        Tuple of (environment_manager, interface_layer)
    """
    # Create environment manager
    environment_manager = EnvironmentManager()
    
    # Create system environment (root)
    system_env = SystemEnvironment()
    environment_manager.register_environment(system_env)
    
    # Create and register other environments
    web_env = WebEnvironment()
    environment_manager.register_environment(web_env)
    
    messaging_env = MessagingEnvironment()
    environment_manager.register_environment(messaging_env)
    
    file_env = FileEnvironment()
    environment_manager.register_environment(file_env)
    
    # Mount environments to the system environment
    system_env.mount_environment("web")
    system_env.mount_environment("messaging")
    system_env.mount_environment("file")
    
    # Create interface layer with context management capabilities
    interface_layer = InterfaceLayer(environment_manager)
    
    # Set the environment manager in the system environment
    system_env.set_environment_manager(environment_manager)
    
    logger.info("Environment system initialized")
    return environment_manager, interface_layer


if __name__ == '__main__':
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize LiteLLM
    initialize_litellm()
    
    # Initialize environment system
    environment_manager, interface_layer = initialize_environments()
    
    # Create message handler function that uses the environment manager instead of interface layer
    message_handler = create_message_handler(environment_manager)
    
    # Initialize Socket.IO client
    socket_client = SocketIOClient(message_handler)
    
    # Initialize message sender
    initialize_sender(socket_client)
    
    # Set up response callback in the environment manager
    # This allows sending responses back through the Activity Layer
    def response_callback(response_data):
        socket_client.send_message(response_data)
    
    environment_manager.set_response_callback(response_callback)
    
    # Connect to adapters
    logger.info(f"Connecting to {len(DEFAULT_ADAPTERS)} normalizing adapters...")
    socket_client.connect_to_adapters()
    
    try:
        # Keep the main thread running
        logger.info("Bot Framework is running. Press Ctrl+C to exit.")
        while not shutdown_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        # Clean up connections
        socket_client.close_connections()
        logger.info("Bot Framework shutdown complete.") 