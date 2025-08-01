"""
Main entry point for the Host process.
Initializes infrastructure modules, SpaceRegistry, and InnerSpace instances for agents.
"""

import logging
import asyncio
import sys
from typing import Dict, Any, Optional # Added Optional
import os

# Basic logging until we load configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Imports ---
# Infrastructure
from elements.space_registry import SpaceRegistry
from llm.provider_factory import LLMProviderFactory # Use factory
from host.event_loop import HostEventLoop
from host.routing.external_event_router import ExternalEventRouter
from host.modules.routing.host_router import HostRouter
# Removed ActivityListener as ActivityClient handles incoming on its connections
# from host.modules.activities.activity_listener import ActivityListener
from host.modules.activities.activity_client import ActivityClient

# Inspector module
from host.modules.inspector.inspector_server import InspectorServer

# Agent/InnerSpace
from elements.elements.inner_space import InnerSpace
from elements.elements.components.agent_loop import SimpleRequestResponseLoopComponent, ToolTextParsingLoopComponent # Import loop types
# Import other loop types if needed for config
from elements.component_registry import COMPONENT_REGISTRY # Import registry to look up loop types

# Import the component scanner
from elements.component_registry import scan_and_load_components

# Configuration Loading
from host.config import load_settings, HostSettings

# ---------------

def configure_logging(log_level: str = "CRITICAL", log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                     log_to_file: bool = False, log_file_path: str = "logs/connectome.log", 
                     max_lines_per_file: int = 5000, max_log_files: int = 10):
    """
    Configure logging with the specified level, format, and optional rolling file logging.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log message format string
        log_to_file: Whether to enable file logging
        log_file_path: Path to log file (directory will be created if needed)
        max_lines_per_file: Maximum lines per log file before rotation
        max_log_files: Maximum number of log files to keep
    """
    try:
        # Get numeric log level from string
        numeric_level = getattr(logging, log_level.upper())
        
        # Clear existing handlers and reconfigure
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Always add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Add file handler if requested
        if log_to_file:
            try:
                import os
                from logging.handlers import RotatingFileHandler
                
                # Create log directory if it doesn't exist
                log_dir = os.path.dirname(log_file_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                # Calculate approximate file size limit based on lines
                # Estimate ~100 characters per log line on average
                estimated_chars_per_line = 100
                max_bytes = max_lines_per_file * estimated_chars_per_line
                
                # Create rotating file handler
                file_handler = RotatingFileHandler(
                    filename=log_file_path,
                    maxBytes=max_bytes,
                    backupCount=max_log_files - 1,  # -1 because current file + backups = total
                    encoding='utf-8'
                )
                file_handler.setLevel(numeric_level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
                
                print(f"✓ Rolling file logging enabled: {log_file_path} ({max_lines_per_file} lines/file, {max_log_files} files max)")
                
            except Exception as file_error:
                print(f"⚠ Failed to setup file logging: {file_error}")
                print("Continuing with console logging only...")
        
        # Set root logger level
        root_logger.setLevel(numeric_level)
        
        # Update our logger reference
        global logger
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured: level={log_level.upper()}, console=✓, file={'✓' if log_to_file else '✗'}")
        
    except AttributeError:
        logger.error(f"Invalid log level: {log_level}. Using INFO instead.")
        # Fallback configuration
        logging.basicConfig(level=logging.INFO, format=log_format, force=True)

async def amain():
    """Asynchronous main entry point."""
    settings = load_settings()
    
    # Configure logging with settings from configuration
    configure_logging(
        log_level=settings.log_level, 
        log_format=settings.log_format,
        log_to_file=settings.log_to_file,
        log_file_path=settings.log_file_path,
        max_lines_per_file=settings.log_max_lines_per_file,
        max_log_files=settings.log_max_files
    )

    # --- Scan for all components FIRST ---
    # This ensures the COMPONENT_REGISTRY is populated before any other
    # module needs to look up a component type by name.
    logger.info("Scanning for all registered components...")
    scan_and_load_components()
    logger.info(f"Component scan complete. Registry contains: {list(COMPONENT_REGISTRY.keys())}")
    
    # Get the SpaceRegistry instance and initialize its storage FIRST
    space_registry = SpaceRegistry.get_instance()
    logger.info("Initializing SpaceRegistry with storage for coordinated replay...")
    
    # NEW: Initialize SpaceRegistry storage first - this will recreate SharedSpaces
    try:
        registry_storage_success = await space_registry.initialize_storage()
        if registry_storage_success:
            logger.info("✓ SpaceRegistry storage initialized - SharedSpaces recreated from storage")
        else:
            logger.warning("⚠ SpaceRegistry storage initialization failed - will proceed without persistence")
    except Exception as e:
        logger.error(f"SpaceRegistry storage initialization error: {e}", exc_info=True)
        logger.warning("Proceeding without SpaceRegistry persistence")

    # Initialize LLM provider using factory and proper config structure
    logger.info("Initializing LLM provider...")
    if not settings.llm_api_key:
        logger.error("LLM API key not set. Please set CONNECTOME_LLM_API_KEY in your .env file.")
        return
        
    try:
        # Convert LLMConfig to dict for factory
        llm_config_dict = {"type": settings.llm_type, "default_model": settings.llm_default_model, "api_key": settings.llm_api_key}
        llm_provider = LLMProviderFactory.create_from_config(llm_config_dict)
        logger.info(f"LLM Provider created: {settings.llm_type} with model {settings.llm_default_model}")
    except Exception as e:
        logger.error(f"Failed to create LLM provider: {e}", exc_info=True)
        return
    
    # Create HostRouter (simple dependency)
    host_router = HostRouter()
    
    # Create a minimal HostEventLoop first
    logger.info("Initializing HostEventLoop...")
    event_loop = HostEventLoop(
        host_router=host_router,
        activity_client=None,  # Will be set after ActivityClient creation
        external_event_router=None,  # Will be set after ExternalEventRouter creation
        space_registry=space_registry
    )
    
    # Now create ExternalEventRouter
    logger.info("Initializing ExternalEventRouter...")
    external_event_router = ExternalEventRouter(
        space_registry=space_registry,
        agent_configs=settings.agents
    )
    
    # Set the external_event_router reference in event_loop
    event_loop.external_event_router = external_event_router
    
    # Initialize Activity Client with proper parameters
    logger.info("Initializing ActivityClient...")
    adapter_configs_as_dicts = [config.model_dump() for config in settings.activity_client_adapter_configs]
    activity_client = ActivityClient(host_event_loop=event_loop, adapter_api_configs=adapter_configs_as_dicts)
    
    # Set activity_client reference in event_loop after creation
    event_loop.activity_client = activity_client
    
    # NEW: Set ActivityClient reference in ExternalEventRouter for outgoing action dispatch
    external_event_router.set_activity_client(activity_client)
    logger.info("ExternalEventRouter configured with ActivityClient reference for action preprocessing")
    
    # After component scanning, we can validate agent configurations
    logger.info("Processing agent configurations...")
    agents_processed = 0
    for agent_config in settings.agents:
        try:
            agent_loop_type = agent_config.agent_loop_component_type_name
            if agent_loop_type not in COMPONENT_REGISTRY:
                logger.error(f"Agent loop component type '{agent_loop_type}' not found in registry. Available types: {list(COMPONENT_REGISTRY.keys())}")
                continue
            
            # Get the component class from registry
            agent_loop_component_class = COMPONENT_REGISTRY[agent_loop_type]
            
            # Create InnerSpace for this agent directly
            logger.info(f"Creating InnerSpace for agent '{agent_config.name}' ({agent_config.agent_id})")
            inner_space = InnerSpace(
                element_id=f"{agent_config.agent_id}_inner_space",
                name=f"{agent_config.name}'s Mind", 
                agent_name=agent_config.name,
                description=f"Inner space for agent {agent_config.name}",
                agent_description=agent_config.description,
                agent_id=agent_config.agent_id,
                llm_provider=llm_provider,
                agent_loop_component_type=agent_loop_component_class,
                outgoing_action_callback=event_loop.get_outgoing_action_callback(),
                agent_purpose=agent_config.description
            )
            
            # Register the InnerSpace with SpaceRegistry
            space_registry.register_inner_space(inner_space, agent_config.agent_id)
            
            agents_processed += 1
            logger.info(f"✓ Agent '{agent_config.name}' ({agent_config.agent_id}) InnerSpace created and registered successfully")
                
        except Exception as e:
            logger.exception(f"Error processing agent config {agent_config.agent_id}: {e}")
    
    if agents_processed == 0:
        logger.error("No agents were successfully configured. Check your CONNECTOME_AGENTS_JSON configuration.")
        return
        
    logger.info(f"Successfully configured {agents_processed} agent(s)")
    
    # In this coordinated approach:
    # 1. ✓ SpaceRegistry storage initialized first - SharedSpaces recreated
    # 2. ✓ Agents initialized - InnerSpace replay can now connect to existing SharedSpaces  
    # 3. ✓ ActivityClient connections happen after internal state restoration
    
    # Start Activity Client (external connections)
    logger.info("Starting Activity Client connections...")
    await activity_client.connect_to_all_adapters()

    # Initialize Inspector Server (if enabled)
    inspector_server = None
    if settings.inspector_enabled:
        logger.info(f"Starting Inspector Server on port {settings.inspector_port}...")
        try:
            # Create a host instance reference for the inspector
            host_instance = type('HostInstance', (), {
                'space_registry': space_registry,
                'activity_client': activity_client,
                'event_loop': event_loop,
                'external_event_router': external_event_router,
                'settings': settings
            })()
            
            inspector_server = InspectorServer(host_instance, port=settings.inspector_port)
            await inspector_server.start()
        except Exception as e:
            logger.error(f"Failed to start Inspector Server: {e}", exc_info=True)
            inspector_server = None

    try:
        logger.info("Starting Host Event Loop...")
        await event_loop.run() # Run the main async loop

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Initiating shutdown...")
    except Exception as e:
         logger.exception(f"Host Event Loop encountered a critical error: {e}")
    finally:
         logger.info("Host process shutting down...")
         
         # NEW: Graceful shutdown with coordinated persistence
         if space_registry:
             await shutdown_all_spaces_gracefully(space_registry)
             # NEW: SpaceRegistry persistence during shutdown
             await space_registry.shutdown_with_persistence()
         
         if event_loop: 
             event_loop.stop() 
         if activity_client:
             await activity_client.shutdown()
         if inspector_server:
             await inspector_server.stop()
         logger.info("Shutdown sequence complete.")

async def shutdown_all_spaces_gracefully(space_registry):
    """
    Gracefully shutdown all spaces in the registry, including VEIL snapshot storage.
    
    Args:
        space_registry: The SpaceRegistry instance containing all spaces
    """
    try:
        logger.info("Beginning graceful shutdown of all spaces...")
        
        # Get all registered spaces
        all_spaces = []
        
        # Get all spaces (includes both InnerSpaces and SharedSpaces)
        spaces_dict = space_registry.get_spaces()
        if spaces_dict:
            all_spaces.extend(spaces_dict.values())
            logger.info(f"Found {len(spaces_dict)} total spaces for shutdown")
        
        if not all_spaces:
            logger.info("No spaces found for shutdown")
            return
        
        logger.info(f"Shutting down {len(all_spaces)} spaces...")
        
        # Shutdown each space
        for space in all_spaces:
            try:
                space_name = getattr(space, 'name', 'Unknown')
                space_id = getattr(space, 'id', 'Unknown')
                space_type = space.__class__.__name__
                
                # Check if this space has VEIL snapshot capability
                if hasattr(space, 'shutdown_with_veil_snapshot'):
                    logger.info(f"Shutting down {space_type} with VEIL snapshot: {space_name} ({space_id})")
                    success = await space.shutdown_with_veil_snapshot()
                    if success:
                        logger.info(f"✓ Successfully shut down space: {space_name}")
                    else:
                        logger.warning(f"⚠ Space shutdown completed with issues: {space_name}")
                else:
                    # Fallback for spaces without snapshot capability
                    logger.info(f"Shutting down {space_type} (basic): {space_name} ({space_id})")
                    # Could add other cleanup here if needed
                    
            except Exception as e:
                space_name = getattr(space, 'name', 'Unknown')
                logger.error(f"Error shutting down space {space_name}: {e}", exc_info=True)
                # Continue with other spaces even if one fails
        
        logger.info("Graceful shutdown of all spaces completed")
        
    except Exception as e:
        logger.error(f"Error during graceful spaces shutdown: {e}", exc_info=True)

def main():
    """Synchronous entry point."""
    try:
        asyncio.run(amain())
    except Exception as e:
        logger.critical(f"Critical error during host execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 