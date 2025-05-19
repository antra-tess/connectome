"""
Main entry point for the Host process.
Initializes infrastructure modules, SpaceRegistry, and InnerSpace instances for agents.
"""

import logging
import asyncio
import sys
from typing import Dict, Any, Optional # Added Optional

# Configure logging
# Consider moving to a dedicated config file/setup function
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Imports ---
# Infrastructure
from elements.space_registry import SpaceRegistry
from llm.provider_factory import LLMProviderFactory # Use factory
from host.event_loop import HostEventLoop
from host.routing.external_event_router import ExternalEventRouter
# Removed ActivityListener as ActivityClient handles incoming on its connections
# from host.modules.activities.activity_listener import ActivityListener
from host.modules.activities.activity_client import ActivityClient

# Agent/InnerSpace
from elements.elements.inner_space import InnerSpace
from elements.elements.agent_loop import SimpleRequestResponseLoopComponent, MultiStepToolLoopComponent # Import loop types
# Import other loop types if needed for config
from elements.component_registry import COMPONENT_REGISTRY # Import registry to look up loop types

# Import the component scanner
from elements.component_registry import scan_and_load_components

# Configuration Loading
from host.config import load_settings, HostSettings

# --------------- 

async def amain():
    """Asynchronous main entry point."""
    logger.info("Starting Host Process (Async - Refactored)..." + "\n" + "-"*20)

    # --- Scan and load components early ---
    logger.info("Scanning for registered components...")
    try:
        scan_and_load_components()
        logger.info(f"Component scanning complete. Registry: {list(COMPONENT_REGISTRY.keys())}")
    except Exception as e:
        logger.critical(f"Failed to scan and load components: {e}", exc_info=True)
        return # Cannot proceed without components
    # --------------------------------------

    # --- Configuration ---
    try:
        settings: HostSettings = load_settings()
    except ValueError as e: # If load_settings re-raises
        logger.critical(f"Halting due to configuration loading error: {e}")
        return
    # -------------------------------------

    # --- Initialize Core Infrastructure ---
    llm_provider = None
    space_registry = None
    activity_client = None
    external_event_router = None
    event_loop = None

    try:
        logger.info("Initializing Core Infrastructure...")
        
        # 1. LLM Provider (using factory and settings)
        if not settings.llm_provider:
             raise ValueError("LLM provider configuration ('llm_provider') is missing from settings.")
        llm_provider_dict = settings.llm_provider.model_dump(exclude_none=True) # Exclude None for cleaner config
        llm_provider = LLMProviderFactory.create_from_config(llm_provider_dict)
        logger.info(f"LLM Provider created: {settings.llm_provider.type}")

        # 2. Space Registry
        space_registry = SpaceRegistry.get_instance() # Use singleton getter
        logger.info("SpaceRegistry initialized (singleton instance).")

        # 3. Activity Client (Placeholder loop, injected later)
        parsed_adapter_configs = [adapter.model_dump(exclude_none=True) for adapter in settings.activity_client_adapter_configs]
        activity_client = ActivityClient(
            host_event_loop=None, # Pass None initially
            adapter_api_configs=parsed_adapter_configs
        )
        logger.info(f"ActivityClient initialized with {len(settings.activity_client_adapter_configs)} adapter configs.")
        
        # --- IMPORTANT: Initialization order ---
        # Event loop needs dependencies, but some dependencies (like ExternalEventRouter)
        # need callbacks from the event loop. Initialize loop partially, then router, then finalize loop.

        # 4. Host Event Loop (Initial Pass - without Router callback yet)
        # We need the loop instance to get the mark_agent_for_cycle callback for the router.
        # Also need agent_configs for the loop.
        event_loop = HostEventLoop(
            host_router=None, # HostRouter seems obsolete, passing None
            activity_client=activity_client,
            external_event_router=None, # Pass None initially
            space_registry=space_registry,
            agent_configs=settings.agents # Pass parsed agent configs
        )
        logger.info("HostEventLoop partially initialized (pre-router).")

        # 5. External Event Router (Inject SpaceRegistry and Loop Callback)
        if not event_loop: raise RuntimeError("Event loop failed initialization") # Should not happen
        
        external_event_router = ExternalEventRouter(
            space_registry=space_registry,
            mark_agent_for_cycle_callback=event_loop.mark_agent_for_cycle, # Pass the method
            agent_configs=settings.agents # <<< Pass agent configs
        )
        logger.info("ExternalEventRouter initialized and linked to HostEventLoop callback.")
        
        # 6. Finalize Host Event Loop (Inject Router)
        event_loop.external_event_router = external_event_router # Inject the router instance
        logger.info("HostEventLoop finalized with ExternalEventRouter reference.")
        
        # 7. Inject correct HostEventLoop instance into ActivityClient
        if not activity_client: raise RuntimeError("Activity client failed initialization") # Should not happen
        activity_client._host_event_loop = event_loop 
        space_registry.response_callback = event_loop.get_outgoing_action_callback()
        logger.info("HostEventLoop reference injected into ActivityClient.")
        
    except Exception as e:
        logger.exception(f"Failed to initialize core infrastructure: {e}")
        return 
    # -------------------------------------

    # --- Initialize InnerSpaces for Agents ---
    if not settings.agents:
         logger.warning("No agents configured in settings.")
    else:
         logger.info(f"Initializing {len(settings.agents)} agent(s)...")
         for agent_config in settings.agents:
              agent_id = agent_config.agent_id
              agent_name = agent_config.name
              if not agent_id:
                   logger.error("Skipping agent config with missing 'agent_id'")
                   continue
                   
              logger.info(f"Initializing InnerSpace for agent: {agent_name} ({agent_id})")
              try:
                   # Determine Agent Loop Component Type from Registry
                   agent_loop_type_name = agent_config.agent_loop_component_type_name
                   agent_loop_type = COMPONENT_REGISTRY.get(agent_loop_type_name)
                   
                   if not agent_loop_type:
                       logger.error(f"Agent loop component type '{agent_loop_type_name}' not found in registry or not a subclass of BaseAgentLoopComponent for agent {agent_id}. Skipping.")
                       continue
                   
                   # Get LLM Provider (use default or agent-specific override - TODO)
                   agent_llm_provider = llm_provider # Default
                   
                   # Get callbacks from the event loop
                   outgoing_callback = event_loop.get_outgoing_action_callback()
                   mark_agent_callback = event_loop.mark_agent_for_cycle # Get the callback
                   
                   # Create InnerSpace instance, passing agent_id and mark_agent_callback
                   inner_space = InnerSpace(
                       element_id=f"innerspace_{agent_id}",
                       name=f"{agent_name}_InnerSpace",
                       agent_name=agent_name,
                       description=agent_config.description,
                       agent_id=agent_id, # <<< Pass agent_id
                       llm_provider=agent_llm_provider,
                       agent_loop_component_type=agent_loop_type,
                       outgoing_action_callback=outgoing_callback,
                       space_registry=space_registry,
                       mark_agent_for_cycle_callback=mark_agent_callback, # <<< Pass callback
                       system_prompt_template=agent_config.system_prompt_template # <<< Pass template
                       # components_to_add=agent_config.inner_space_extra_components # If using this config
                   )
                   
                   # Register InnerSpace with SpaceRegistry
                   registration_success = space_registry.register_inner_space(inner_space, agent_id)
                   
                   if registration_success:
                       logger.info(f"Successfully initialized and registered InnerSpace for {agent_name} ({agent_id}).")
                   else:
                       logger.error(f"Failed to register InnerSpace for {agent_name} ({agent_id}). Skipping agent.")
              
              except Exception as e:
                   logger.exception(f"Error initializing InnerSpace for agent {agent_id}: {e}")
    # -------------------------------------

    # --- Start Activity Client Connections --- (After EventLoop is fully ready)
    if activity_client:
        logger.info("Starting ActivityClient connections...")
        await activity_client.start_connections()
        logger.info("ActivityClient connections started.")
    # -----------------------------------------

    # --- Start Event Loop --- 
    if not event_loop:
         logger.critical("HostEventLoop failed to initialize. Cannot run.")
         return
         
    try:
        logger.info("Starting Host Event Loop...")
        await event_loop.run() # Run the main async loop

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Initiating shutdown...")
    except Exception as e:
         logger.exception(f"Host Event Loop encountered a critical error: {e}")
    finally:
         logger.info("Host process shutting down...")
         if event_loop: 
             event_loop.stop() 
         if activity_client:
             await activity_client.shutdown()
         # Add shutdown for other components if needed (e.g., SpaceRegistry persistence)
         logger.info("Shutdown sequence complete.")
    # ------------------------------------

def main():
    """Synchronous entry point."""
    try:
        asyncio.run(amain())
    except Exception as e:
        logger.critical(f"Critical error during host execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 