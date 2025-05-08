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
from elements.elements.agent_loop import SimpleRequestResponseLoopComponent # Import default loop
# Import other loop types if needed for config

# Configuration Loading (Placeholder - replace with actual config loading)
# Example: from config import load_host_config 

# --------------- 

async def amain():
    """Asynchronous main entry point."""
    logger.info("Starting Host Process (Async - Refactored)..." + "\n" + "-"*20)

    # --- Configuration (Placeholder - Adapt as needed) ---
    # config = load_host_config() # Load from file/env
    host_config = {
        # LLM Provider config (using factory)
        "llm_provider": {
            "type": "litellm", # Example: using LiteLLM
            # Config specific to the provider type
            "default_model": "gpt-4", # Example for LiteLLM
            # Add API keys, base URLs etc. securely (e.g., from env vars)
            # "api_key": os.environ.get("OPENAI_API_KEY") 
        },
        "activity_client_adapter_configs": [
            # Example: Needs actual adapter URLs/auth
            {"id": "discord_adapter_1", "url": "http://localhost:5001", "auth": None},
            # {"id": "telegram_adapter_main", "url": "http://localhost:5002", "auth": {"token": "secret"}}
        ],
        "agents": [
            {
                "agent_id": "agent_001",
                "name": "Default Agent",
                "description": "The primary agent instance.",
                "agent_loop_component_type_name": "SimpleRequestResponseLoopComponent", # Name to select loop type
                # Agent-specific LLM overrides (optional)
                # "llm_provider_config_override": { "default_model": "gpt-3.5-turbo" }
                "inner_space_extra_components": [] # List any extra Component types
            },
            # Add more agent configurations if running multi-agent host
            # {
            #     "agent_id": "agent_002",
            #     "name": "Research Agent",
            #     "description": "Specialized research agent.",
            #     "agent_loop_component_type_name": "ResearchLoopComponent", # Requires this class to exist
            # }
        ]
    }
    # -------------------------------------

    # --- Initialize Core Infrastructure ---
    llm_provider = None
    space_registry = None
    activity_client = None
    external_event_router = None
    event_loop = None

    try:
        logger.info("Initializing Core Infrastructure...")
        
        # 1. LLM Provider (using factory)
        llm_provider_config = host_config.get("llm_provider")
        if not llm_provider_config:
            raise ValueError("LLM provider configuration ('llm_provider') is missing.")
        llm_provider = LLMProviderFactory.create_from_config(llm_provider_config)
        logger.info(f"LLM Provider created: {llm_provider_config.get('type')}")

        # 2. Space Registry
        space_registry = SpaceRegistry()
        logger.info("SpaceRegistry initialized.")

        # 3. Activity Client (needs HostEventLoop callback later)
        activity_client_configs = host_config.get("activity_client_adapter_configs", [])
        # Pass placeholder HostEventLoop for now, will inject callback later
        temp_event_loop_for_ac_init = None # ActivityClient needs the loop instance itself
        activity_client = ActivityClient(host_event_loop=temp_event_loop_for_ac_init, adapter_api_configs=activity_client_configs) 
        logger.info(f"ActivityClient initialized with {len(activity_client_configs)} adapter configs.")

        # 4. External Event Router
        external_event_router = ExternalEventRouter(space_registry=space_registry)
        logger.info("ExternalEventRouter initialized.")

        # 5. Host Event Loop (Inject dependencies)
        # Removed ActivityListener dependency
        event_loop = HostEventLoop(
            host_router=None, # HostRouter might be obsolete now, pass None or remove param
            space_registry=space_registry,
            external_event_router=external_event_router,
            activity_client=activity_client
        )
        logger.info("HostEventLoop initialized.")
        
        # 6. Inject correct HostEventLoop instance/callback into ActivityClient
        # Option A: Give ActivityClient the whole loop instance if it needs more interaction
        activity_client._host_event_loop = event_loop 
        # Option B: If ActivityClient only needs the enqueue method (more decoupled)
        # activity_client.set_incoming_event_callback(event_loop.enqueue_incoming_event) 
        logger.info("HostEventLoop reference injected into ActivityClient.")

    except Exception as e:
        logger.exception(f"Failed to initialize core infrastructure: {e}")
        return
    # -------------------------------------

    # --- Initialize InnerSpaces for Agents ---
    agent_configs = host_config.get("agents", [])
    if not agent_configs:
         logger.warning("No agents configured in host_config.")
    else:
         logger.info(f"Initializing {len(agent_configs)} agent(s)...")
         for agent_config in agent_configs:
              agent_id = agent_config.get("agent_id")
              agent_name = agent_config.get("name", f"Agent_{agent_id}")
              if not agent_id:
                   logger.error("Skipping agent config with missing 'agent_id'")
                   continue
                   
              logger.info(f"Initializing InnerSpace for agent: {agent_name} ({agent_id})")
              try:
                   # Determine Agent Loop Component Type
                   agent_loop_type_name = agent_config.get("agent_loop_component_type_name", "SimpleRequestResponseLoopComponent")
                   # Simple lookup for now, could use dynamic import or a registry
                   if agent_loop_type_name == "SimpleRequestResponseLoopComponent":
                       agent_loop_type = SimpleRequestResponseLoopComponent
                   # Add elif for other loop types...
                   # elif agent_loop_type_name == "ResearchLoopComponent":
                   #     from path.to.research_loop import ResearchLoopComponent
                   #     agent_loop_type = ResearchLoopComponent 
                   else:
                       logger.error(f"Unknown agent_loop_component_type_name: '{agent_loop_type_name}' for agent {agent_id}. Skipping agent.")
                       continue
                   
                   # Get LLM Provider (use default or agent-specific override)
                   agent_llm_provider = llm_provider # Default
                   # TODO: Add logic for agent-specific LLM overrides if needed
                   # if agent_config.get("llm_provider_config_override"):
                   #    agent_llm_provider = LLMProviderFactory.create_from_config(agent_config["llm_provider_config_override"])
                   
                   # Get the outgoing action callback from the event loop
                   outgoing_callback = event_loop.get_outgoing_action_callback()
                   
                   # Create InnerSpace instance
                   inner_space = InnerSpace(
                       id=f"innerspace_{agent_id}", # Generate InnerSpace ID
                       name=f"{agent_name}_InnerSpace",
                       description=agent_config.get("description", "Agent Inner Space"),
                       llm_provider=agent_llm_provider,
                       agent_loop_component_type=agent_loop_type,
                       outgoing_action_callback=outgoing_callback,
                       # components_to_add=agent_config.get("inner_space_extra_components", []) # Pass extra components if needed
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
        await activity_client.start_connections()
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