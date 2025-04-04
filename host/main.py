"""
Main entry point for the Host process.
Initializes modules, elements, and starts the event loop.
"""

import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Imports for Core Components ---
# Core elements & registry
from elements.elements.inner_space import InnerSpace
from elements.space_registry import SpaceRegistry

# LLM abstraction
from llm.openai_provider import OpenAIProvider # Or your chosen provider

# Host components
from host.event_loop import HostEventLoop
from host.modules.activities.activity_listener import ActivityListener
from host.modules.activities.activity_client import ActivityClient

# Configuration (assuming a basic config loading mechanism exists or is added)
# from .config import load_config # Example

# ---------------------------------

async def amain():
    """Asynchronous main entry point."""
    logger.info("Starting Host Process (Async)..." + "\n" + "-"*20)

    # --- Configuration (Placeholder) ---
    # TODO: Implement robust configuration loading (e.g., from YAML/env vars)
    llm_api_key = "YOUR_OPENAI_API_KEY" # IMPORTANT: Load securely!
    # Example configs - replace with actual loaded values
    activity_listener_config = {"host": "localhost", "port": 5000} # If listener runs a server
    activity_client_adapter_configs = [
         {"id": "adapter_api_1", "url": "http://localhost:5001", "auth_token": None}
    ]

    if llm_api_key == "YOUR_OPENAI_API_KEY":
        logger.warning("LLM API Key is set to placeholder!")
    # ------------------------------------

    # --- Initialize Core Services/Modules ---
    try:
        # 1. Initialize LLM Provider
        logger.info("Initializing LLM Provider...")
        llm_provider = OpenAIProvider(api_key=llm_api_key)
        logger.info(f"LLM Provider Initialized: OpenAI") # Adjust if using different provider

        # 2. Initialize Space Registry
        logger.info("Initializing Space Registry...")
        space_registry = SpaceRegistry()
        logger.info("Space Registry Initialized.")

        # TODO: Initialize Persistence Module

    except Exception as e:
        logger.exception(f"Failed to initialize core services: {e}")
        return # Cannot proceed
    # ------------------------------------

    # --- Initialize InnerSpace ---
    try:
        logger.info("Initializing InnerSpace...")
        # InnerSpace needs the LLM provider. It will get the outgoing callback later.
        inner_space = InnerSpace(
            id="agent_inner_space_main",
            name="Agent Mind",
            llm_provider=llm_provider
        )
        logger.info(f"InnerSpace instance created: {inner_space.id}")
        
        # Register InnerSpace with the Registry AFTER creation
        if not space_registry.register_inner_space(inner_space):
            logger.error("Failed to register InnerSpace with the registry. Exiting.")
            return
        logger.info(f"InnerSpace registered: {inner_space.id}")

    except Exception as e:
         logger.exception(f"Failed to initialize InnerSpace: {e}")
         return # Cannot proceed
    # ------------------------------------

    # --- Initialize Activity Modules ---
    try:
        logger.info("Initializing Activity Modules...")
        # Pass space_registry to listener
        activity_listener = ActivityListener(
            space_registry=space_registry # Removed config for now
        )
        activity_client = ActivityClient(
             adapter_configs=activity_client_adapter_configs
        )
        logger.info("Activity Listener and Client Initialized.")
        # TODO: Start listener server/consumer task if needed
        # TODO: Start client connection task if needed

    except Exception as e:
        logger.exception(f"Failed to initialize Activity modules: {e}")
        return # Decide if this is fatal
    # ------------------------------------

    # --- Initialize and Start Event Loop ---
    try:
        logger.info("Initializing Host Event Loop...")
        event_loop = HostEventLoop(
            registry=space_registry,
            inner_space=inner_space,
            activity_listener=activity_listener,
            activity_client=activity_client
        )

        # Give the listener the callback to enqueue INCOMING events
        activity_listener.set_event_loop(event_loop)
        
        # Get the callback for OUTGOING actions from the loop
        outgoing_action_callback = event_loop.get_outgoing_action_callback()
        
        # NOW Inject the OUTGOING action callback into InnerSpace
        # This requires InnerSpace to have a method like `set_outgoing_action_callback`
        inner_space.set_outgoing_action_callback(outgoing_action_callback)
        logger.info("Outgoing action callback injected into InnerSpace.")

        logger.info("Starting Host Event Loop...")
        await event_loop.run() # Run the main async loop

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Initiating shutdown...")
    except Exception as e:
         logger.exception(f"Host Event Loop encountered a critical error: {e}")
    finally:
         logger.info("Host process shutting down...")
         # --- Graceful Shutdown ---
         if 'event_loop' in locals() and hasattr(event_loop, 'stop'):
             event_loop.stop() # Signal loop to stop

         # TODO: Add graceful shutdown for listener, client, persistence etc.
         # if 'activity_listener' in locals() and hasattr(activity_listener, 'shutdown'):
         #     await activity_listener.shutdown()
         # if 'activity_client' in locals() and hasattr(activity_client, 'close_connections'):
         #     activity_client.close_connections() # May need async version
         logger.info("Shutdown sequence complete.")
    # ------------------------------------

def main():
    """Synchronous entry point."""
    try:
        asyncio.run(amain())
    except Exception as e:
        # Catch final errors during startup/shutdown outside amain's try/except
        logger.critical(f"Critical error during host execution: {e}", exc_info=True)
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main() 