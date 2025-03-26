#!/usr/bin/env python3
"""
Bot Framework Main Entry Point
Initializes the components and connects to normalizing adapters.
"""

import json
import logging
import os
import signal
import time
from threading import Event

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    ADDITIONAL_ADAPTERS,
    AGENT_NAME,
    AGENT_DESCRIPTION
)

# Global flag for graceful shutdown
shutdown_event = Event()

# Import modules
from activity.client import SocketIOClient
from activity.listener import MessageHandler
from elements.space_registry import SpaceRegistry
from elements.elements.inner_space import InnerSpace
from shell import SinglePhaseShell, TwoPhaseShell
from shell.hud import HUD
from shell.context_manager import ContextManager


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
    
    # Validate that the shell has an inner_space
    if not hasattr(shell, 'inner_space') or shell.inner_space is None:
        logger.error("Shell initialization failed: No InnerSpace available")
        raise RuntimeError("Shell initialization failed: No InnerSpace available")
    
    # Verify that the inner_space is properly registered
    if not space_registry.get_space(shell.inner_space.id):
        logger.error(f"InnerSpace {shell.inner_space.id} not registered with SpaceRegistry")
        raise RuntimeError(f"InnerSpace {shell.inner_space.id} not registered with SpaceRegistry")
    
    # Verify registry's inner_space reference matches shell's inner_space
    if space_registry.get_inner_space() != shell.inner_space:
        logger.error("Registry inner_space reference doesn't match shell's inner_space")
        # Try to fix by updating the registry's reference
        if isinstance(shell.inner_space, InnerSpace):
            logger.warning("Attempting to fix by registering shell's inner_space with registry")
            space_registry.register_inner_space(shell.inner_space)
            if space_registry.get_inner_space() != shell.inner_space:
                raise RuntimeError("Failed to align registry's inner_space with shell's inner_space")
            logger.info("Fixed: Registry's inner_space now matches shell's inner_space")
        else:
            raise RuntimeError("Inner space type mismatch - not an InnerSpace instance")
    
    logger.info(f"Confirmed InnerSpace '{shell.inner_space.id}' is properly initialized")
    
    # Initialize chat elements based on the selected model
    if chat_model.lower() == 'uplinked':
        # Model 2: Uplinked chat elements
        logger.info("Using uplinked chat model")
        
        # Use the factory function to create uplinked chat setup
        from elements import create_uplinked_chat_setup
        
        # Track created uplinks for validation
        created_uplinks = []
        
        # For each adapter we want to support
        for adapter_info in DEFAULT_ADAPTERS:
            platform = adapter_info.get('platform', 'unknown')
            adapter_id = adapter_info.get('id', 'unknown')
            
            try:
                # Create the uplinked chat setup
                shared_space, uplink, chat_element = create_uplinked_chat_setup(
                    shell.inner_space, 
                    space_registry,
                    platform=platform,
                    adapter_id=adapter_id
                )
                
                # Verify the uplink was mounted correctly
                if uplink.id not in [e.id for e in shell.inner_space.get_elements()]:
                    logger.warning(f"Uplink {uplink.id} not properly mounted to InnerSpace")
                    # Attempt to mount it again
                    shell.inner_space.mount_element(uplink)
                    logger.info(f"Re-mounted uplink {uplink.id} to InnerSpace")
                
                created_uplinks.append(uplink.id)
                logger.info(f"Created uplinked chat for {platform} (adapter: {adapter_id})")
            except Exception as e:
                logger.error(f"Failed to create uplinked chat for {platform}: {e}")
                # Continue with other adapters despite this error
        
        # Validate we created at least one uplink
        if not created_uplinks:
            logger.error("No uplinks were successfully created")
            raise RuntimeError("Failed to create any uplinks for chat communication")
            
        logger.info(f"Initialized uplinked chat model with {len(created_uplinks)} shared spaces")
    else:
        # Model 1: Direct chat elements
        logger.info("Using direct chat model")
        
        # Use the factory function to create direct chat elements
        from elements import create_direct_chat_element
        
        # Track created chat elements for validation
        created_elements = []
        
        # For each adapter we want to support
        for adapter_info in DEFAULT_ADAPTERS:
            platform = adapter_info.get('platform', 'unknown')
            adapter_id = adapter_info.get('id', 'unknown')
            
            try:
                # Create the direct chat element
                element_id = f"{platform}_chat"
                chat_element = create_direct_chat_element(
                    element_id,
                    f"{platform.capitalize()} Chat",
                    f"Interface for {platform} messages",
                    platform=platform,
                    adapter_id=adapter_id,
                    registry=space_registry
                )
                
                # Mount in the inner space and verify it worked
                shell.inner_space.mount_element(chat_element)
                
                # Verify the element was mounted correctly
                if element_id not in [e.id for e in shell.inner_space.get_elements()]:
                    logger.warning(f"Chat element {element_id} not properly mounted to InnerSpace")
                    # Try mounting again
                    shell.inner_space.mount_element(chat_element)
                    
                created_elements.append(element_id)
                logger.info(f"Created direct chat for {platform} (adapter: {adapter_id})")
            except Exception as e:
                logger.error(f"Failed to create direct chat element for {platform}: {e}")
                # Continue with other adapters despite this error
        
        # Validate we created at least one chat element
        if not created_elements:
            logger.error("No chat elements were successfully created")
            raise RuntimeError("Failed to create any chat elements for communication")
            
        logger.info(f"Initialized direct chat model with {len(created_elements)} elements")
    
    # Final validation - make sure inner space has elements
    inner_space_elements = shell.inner_space.get_elements()
    if not inner_space_elements:
        logger.error("InnerSpace has no elements mounted - communication will fail")
        raise RuntimeError("No elements mounted in InnerSpace")
        
    logger.info(f"Shell and elements initialized (using {shell_type} shell with {chat_model} chat model)")
    logger.info(f"InnerSpace has {len(inner_space_elements)} elements mounted")
    
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
    
    # Create timeline context for the event
    timeline_context = {
        "timeline_id": f"{message_data.get('platform')}_{message_data.get('user_id')}",
        "is_primary": True,
        "last_event_id": None  # This should be retrieved from previous state
    }
    
    # Process through shell with timeline context
    result = shell.handle_external_event(event, timeline_context)
    logger.debug(f"Shell processing result: {result}")
    
    return result


def get_combined_adapters():
    """
    Combine default and additional adapters.
    
    Returns:
        List of combined adapter configurations
    """
    try:
        additional_adapters = json.loads(ADDITIONAL_ADAPTERS)
        combined_adapters = DEFAULT_ADAPTERS.copy()
        combined_adapters.extend(additional_adapters)
        return combined_adapters
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse ADDITIONAL_ADAPTERS: {ADDITIONAL_ADAPTERS}")
        return DEFAULT_ADAPTERS.copy()


if __name__ == '__main__':
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize shell and space registry
        shell, space_registry = initialize_shell()
        
        # Create message handler with the space registry
        message_handler = MessageHandler(space_registry)
        
        # Initialize Socket.IO client with the message handler
        socket_client = SocketIOClient(message_handler)
        
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
                        action_type = action.get("type")
                        
                        if action_type == "message" and action.get("success"):
                            # Format and send response
                            socket_client.send_response(
                                user_id=message_data.get("user_id"),
                                message_text=action.get("content", ""),
                                message_id=message_data.get("message_id"),
                                platform=message_data.get("platform"),
                                adapter_id=message_data.get("adapter_id")
                            )
                        elif action_type == "element_action" and action.get("success"):
                            # Handle element action results
                            logger.info(f"Element action executed: {action.get('name')}")
                        elif action_type == "shell_tool" and action.get("success"):
                            # Handle shell tool results
                            logger.info(f"Shell tool executed: {action.get('name')}")
                        else:
                            logger.debug(f"Unhandled or failed action: {action_type}")
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
        
        # Register callback with SpaceRegistry
        space_registry.set_response_callback(message_callback)
        
        # Get combined adapters for logging purposes
        combined_adapters = get_combined_adapters()
        
        # Connect to adapters (uses internal adapter list)
        logger.info(f"Connecting to {len(combined_adapters)} normalizing adapters...")
        socket_client.connect_to_adapters()
        
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