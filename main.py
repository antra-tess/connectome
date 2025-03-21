#!/usr/bin/env python3
"""
Bot Framework Main Entry Point
Initializes the Socket.IO clients and connects to normalizing adapters.
"""

import signal
import sys
import time
import logging
import json
import os
import importlib
import inspect
from threading import Event

# Import configuration
from config import ENVIRONMENTS, PROTOCOLS, DEFAULT_PROTOCOL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_event = Event()

# Import modules
from activity.client import SocketIOClient
from activity.listener import MessageHandler
from elements.system import SystemEnvironment
from bot_framework.elements.space_registry import EnvironmentManager, SpaceRegistry
from bot_framework.interface.layer import InterfaceLayer

from elements.base import Environment


def discover_environment_classes():
    """
    Dynamically discover all environment classes from the environments directory.
    
    Returns:
        dict: A dictionary mapping class names to environment classes
    """
    env_classes = {}
    environments_dir = os.path.join(os.path.dirname(__file__), 'environments/environment_classes/')
    
    logger.info(f"Discovering environment classes from: {environments_dir}")
    
    # List all Python files in the environments directory
    try:
        for filename in os.listdir(environments_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]  # Remove the .py extension
                
                try:
                    # Import the module dynamically
                    module_path = f"environments.{module_name}"
                    module = importlib.import_module(module_path)
                    
                    # Find all classes in the module that inherit from Environment
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, Environment) and 
                            obj is not Environment):
                            logger.info(f"Discovered environment class: {name}")
                            env_classes[name] = obj
                            
                except Exception as e:
                    logger.error(f"Error importing module {module_name}: {str(e)}")
    except Exception as e:
        logger.error(f"Error scanning environments directory: {str(e)}")
    
    return env_classes


def signal_handler(sig, frame):
    """
    Handle shutdown signals.
    """
    logger.info("Shutdown signal received, closing connections...")
    shutdown_event.set()


def initialize_environments():
    """
    Initialize the environment system based on configuration.
    
    Returns:
        Tuple of (environment_manager, interface_layer)
    """
    # Create environment manager
    environment_manager = EnvironmentManager()
    
    # Create system environment (root) - this is always required
    system_env = SystemEnvironment()
    environment_manager.register_environment(system_env)
    
    # Dictionary to store created environments by ID for later mounting
    created_environments = {"system": system_env}
    
    # Create and register configured environments
    logger.info("Initializing configured environments...")
    
    # Dynamically discover available environment classes
    env_classes = discover_environment_classes()
    
    # Always ensure SystemEnvironment is available
    env_classes["SystemEnvironment"] = SystemEnvironment
    
    # Initialize environments from configuration
    for env_config in ENVIRONMENTS:
        env_id = env_config.get("id")
        env_class_name = env_config.get("class")
        enabled = env_config.get("enabled", True)
        
        if not enabled:
            logger.info(f"Environment '{env_id}' is disabled in configuration, skipping...")
            continue
            
        if env_class_name not in env_classes:
            logger.warning(f"Unknown environment class '{env_class_name}' for '{env_id}', skipping... Available classes: {list(env_classes.keys())}")
            continue
            
        try:
            # Create the environment instance
            logger.info(f"Creating environment: {env_id} ({env_class_name})")
            env_class = env_classes[env_class_name]
            environment = env_class(env_id=env_id)
            
            # Register with the manager
            environment_manager.register_environment(environment)
            created_environments[env_id] = environment
        except Exception as e:
            logger.error(f"Failed to initialize environment '{env_id}': {str(e)}")
    
    # Mount environments to the system environment
    logger.info("Mounting environments to system environment...")
    for env_config in ENVIRONMENTS:
        env_id = env_config.get("id")
        enabled = env_config.get("enabled", True)
        mount_point = env_config.get("mount_point")
        
        if not enabled or env_id not in created_environments:
            continue
            
        try:
            logger.info(f"Mounting environment '{env_id}' to system environment")
            system_env.mount_environment(env_id, mount_point)
        except Exception as e:
            logger.error(f"Failed to mount environment '{env_id}': {str(e)}")
    
    # Determine which protocol to use based on configuration
    protocol_name = DEFAULT_PROTOCOL
    
    # Check if the default protocol is enabled in configuration
    default_protocol_enabled = False
    for protocol_config in PROTOCOLS:
        if protocol_config.get("name") == DEFAULT_PROTOCOL and protocol_config.get("enabled", True):
            default_protocol_enabled = True
            break
    
    # If default protocol is not enabled, use the first enabled protocol
    if not default_protocol_enabled:
        for protocol_config in PROTOCOLS:
            if protocol_config.get("enabled", True):
                protocol_name = protocol_config.get("name")
                logger.info(f"Default protocol '{DEFAULT_PROTOCOL}' not enabled, using '{protocol_name}' instead")
                break
    
    logger.info(f"Using protocol: {protocol_name}")
    
    # Create interface layer with context management capabilities and the selected protocol
    interface_layer = InterfaceLayer(environment_manager, protocol_name=protocol_name)
    
    # Set the environment manager in the system environment
    system_env.set_environment_manager(environment_manager)
    
    logger.info("Environment system initialized")
    return environment_manager, interface_layer


if __name__ == '__main__':
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize environment system (LLMProcessor will handle LiteLLM initialization)
    environment_manager, interface_layer = initialize_environments()

    # Initialize SpaceRegistry if not already initialized
    space_registry = SpaceRegistry()

    # Create message handler with the space registry
    message_handler = MessageHandler(space_registry)

    # Initialize Socket.IO client with the message handler
    socket_client = SocketIOClient(message_handler)
    
    # Connect the socket client to the space registry
    space_registry.set_socket_client(socket_client)
    
    # Set space registry on all elements that need to send outgoing messages
    for env_id, environment in environment_manager.get_all_environments().items():
        if hasattr(environment, 'set_registry'):
            logger.info(f"Setting space registry for environment: {env_id}")
            environment.set_registry(space_registry)
    
    # Define the response callback function and set it on the environment manager
    def response_callback(response_data):
        # Extract data from the response
        user_id = response_data.get('user_id')
        message = response_data.get('message')
        message_id = response_data.get('message_id')
        platform = response_data.get('platform')
        adapter_id = response_data.get('adapter_id')
        
        # Use the send_response method which handles formatting
        socket_client.send_response(
            user_id=user_id,
            message_text=message,
            message_id=message_id,
            platform=platform,
            adapter_id=adapter_id
        )

    # Set response callback directly on the environment manager
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