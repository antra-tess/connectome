#!/usr/bin/env python3
"""
Bot Framework Main Entry Point
Initializes the components and connects to normalizing adapters.
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
from config import (
    SOCKET_RECONNECTION_ATTEMPTS, 
    SOCKET_RECONNECTION_DELAY,
    SOCKET_TIMEOUT,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_BASE_URL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    DEFAULT_ADAPTERS,
    AGENT_NAME,
    AGENT_DESCRIPTION
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_event = Event()

# Import modules
from activity.client import SocketIOClient
from activity.listener import MessageHandler
from elements.space_registry import SpaceRegistry
from elements.elements.inner_space import InnerSpace
from elements.elements.messaging import ChatElement
from shell import SinglePhaseShell, TwoPhaseShell
from shell.hud import HUD
from shell.context_manager import ContextManager
from rendering import RenderingFormat, RenderingImportance


def signal_handler(sig, frame):
    """
    Handle shutdown signals.
    """
    logger.info("Shutdown signal received, closing connections...")
    shutdown_event.set()


def configure_llm():
    """
    Configure the LLM provider based on environment variables.
    
    Returns:
        dict: LLM configuration
    """
    # Create LLM configuration
    llm_config = {
        "type": LLM_PROVIDER,
        "default_model": LLM_MODEL,
        "api_key": LLM_API_KEY,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "max_sequential_tool_calls": 5,
    }
    
    # Add base URL if provided
    if LLM_BASE_URL:
        llm_config["base_url"] = LLM_BASE_URL
    
    logger.info(f"Configured LLM provider: {LLM_PROVIDER}, model: {LLM_MODEL}")
    return llm_config


def initialize_shell(shell_type=None):
    """
    Initialize the Shell and supporting components.
    
    Args:
        shell_type: Type of shell to initialize ("single_phase" or "two_phase")
        
    Returns:
        BaseShell: The initialized shell instance
    """
    # Get shell type from environment variable if not provided
    if shell_type is None:
        shell_type = os.getenv('SHELL_TYPE', 'single_phase')
    
    # Get chat model from environment variable
    chat_model = os.getenv('CHAT_MODEL', 'direct')  # 'direct' or 'uplinked'
    
    # Create SpaceRegistry
    space_registry = SpaceRegistry()
    logger.info("Space registry initialized")
    
    # Create HUD
    hud = HUD()
    logger.info("HUD initialized")
    
    # Create context manager
    context_manager = ContextManager()
    logger.info("Context manager initialized")
    
    # Configure LLM
    llm_config = configure_llm()
    
    # Model info for the shell
    model_info = {
        "name": AGENT_NAME,
        "description": AGENT_DESCRIPTION,
        "capabilities": ["text", "tool_use", "memory"],
    }
    
    # Initialize the appropriate shell type
    if shell_type.lower() == "two_phase":
        logger.info("Initializing TwoPhaseShell")
        shell = TwoPhaseShell(
            registry=space_registry,
            hud=hud,
            context_manager=context_manager,
            model_info=model_info,
            llm_config=llm_config
        )
    else:  # Default to single phase
        logger.info("Initializing SinglePhaseShell")
        shell = SinglePhaseShell(
            registry=space_registry,
            hud=hud,
            context_manager=context_manager,
            model_info=model_info,
            llm_config=llm_config
        )
    
    # Initialize chat elements based on the selected model
    if chat_model.lower() == 'uplinked':
        # Model 2: Uplinked chat elements
        logger.info("Using uplinked chat model")
        
        # Use the factory function to create uplinked chat setup
        from elements import create_uplinked_chat_setup
        
        # For each adapter we want to support
        for adapter_info in DEFAULT_ADAPTERS:
            platform = adapter_info.get('platform', 'unknown')
            adapter_id = adapter_info.get('adapter_id', 'unknown')
            
            # Create the uplinked chat setup
            shared_space, uplink, chat_element = create_uplinked_chat_setup(
                shell.inner_space, 
                space_registry,
                platform=platform,
                adapter_id=adapter_id
            )
            
            logger.info(f"Created uplinked chat for {platform} (adapter: {adapter_id})")
        
        logger.info("Initialized uplinked chat model with shared spaces")
    else:
        # Model 1: Direct chat elements
        logger.info("Using direct chat model")
        
        # Use the factory function to create direct chat elements
        from elements import create_direct_chat_element
        
        # For each adapter we want to support
        for adapter_info in DEFAULT_ADAPTERS:
            platform = adapter_info.get('platform', 'unknown')
            adapter_id = adapter_info.get('adapter_id', 'unknown')
            
            # Create the direct chat element
            chat_element = create_direct_chat_element(
                f"{platform}_chat",
                f"{platform.capitalize()} Chat",
                f"Interface for {platform} messages",
                platform=platform,
                adapter_id=adapter_id,
                registry=space_registry
            )
            
            # Mount in the inner space
            shell.inner_space.mount_element(chat_element)
            
            logger.info(f"Created direct chat for {platform} (adapter: {adapter_id})")
        
        logger.info("Initialized direct chat model")
    
    logger.info(f"Shell and elements initialized (using {shell_type} shell with {chat_model} chat model)")
    return shell, space_registry


def process_message(message_data, shell):
    """
    Process an incoming message through the shell.
    
    Args:
        message_data: Message data from the activity layer
        shell: The shell instance to process the message
        
    Returns:
        dict: Result of message processing
    """
    logger.info(f"Processing message: {message_data.get('message_id')} from {message_data.get('user_id')}")
    
    # Create event data for shell processing
    event = {
        "type": "user_message",
        "event_type": "message_received",
        "user_id": message_data.get("user_id"),
        "message_id": message_data.get("message_id"),
        "content": message_data.get("message"),
        "platform": message_data.get("platform"),
        "timestamp": int(time.time() * 1000),
        "metadata": {
            "adapter_id": message_data.get("adapter_id"),
            "raw_event": message_data.get("raw_event", {})
        }
    }
    
    # Process through shell
    result = shell.handle_external_event(event)
    logger.debug(f"Shell processing result: {result}")
    
    return result


if __name__ == '__main__':
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize shell and space registry
        shell, space_registry = initialize_shell()
        
        # Create message handler with the space registry
        message_handler = MessageHandler(space_registry, shell)
        
        # Initialize Socket.IO client with the message handler
        socket_client = SocketIOClient(
            message_handler, 
            reconnection_attempts=SOCKET_RECONNECTION_ATTEMPTS,
            reconnection_delay=SOCKET_RECONNECTION_DELAY,
            timeout=SOCKET_TIMEOUT
        )
        
        # Connect the socket client to the space registry
        space_registry.set_socket_client(socket_client)
        
        # Define the message processing callback
        def message_callback(message_data):
            try:
                # Process message through shell
                result = process_message(message_data, shell)
                
                # Check for responses in the result
                if "actions_executed" in result:
                    for action in result["actions_executed"]:
                        if action.get("type") == "message" and action.get("success"):
                            # Format and send response
                            socket_client.send_response(
                                user_id=message_data.get("user_id"),
                                message_text=action.get("content", ""),
                                message_id=message_data.get("message_id"),
                                platform=message_data.get("platform"),
                                adapter_id=message_data.get("adapter_id")
                            )
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
        
        # Set message callback on message handler
        message_handler.set_message_callback(message_callback)
        
        # Connect to adapters
        logger.info(f"Connecting to {len(DEFAULT_ADAPTERS)} normalizing adapters...")
        socket_client.connect_to_adapters(DEFAULT_ADAPTERS)
        
        logger.info("Bot Framework is running. Press Ctrl+C to exit.")
        while not shutdown_event.is_set():
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        # Clean up connections
        if 'socket_client' in locals():
            socket_client.close_connections()
        logger.info("Bot Framework shutdown complete.") 