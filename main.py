#!/usr/bin/env python3
"""
Main application entry point for the Connectome Host.
Initializes core components, modules, and starts the event loop.
"""

import logging
import asyncio
import os
import signal
from typing import Dict, Any, List

# Host components
from host.event_loop import HostEventLoop
from host.modules.routing.host_router import HostRouter

# Modules
from host.modules.shell.shell_module import ShellModule
from host.modules.activities.activity_client import ActivityClient
# Example Activity Adapters (replace with actual/mock implementations)
from host.modules.activities.adapters.mock_adapter import MockActivityAdapter 

# LLM Provider
from llm.openai_provider import OpenAIProvider # Using OpenAI for now
# from llm.litellm_provider import LiteLLMProvider # Or use LiteLLM
from llm.provider_interface import LLMProvider

# Basic Config Loading (replace with a proper system like Dynaconf, Pydantic Settings, etc.)
def load_config() -> Dict[str, Any]:
    """Loads basic configuration. Placeholder implementation."""
    logger.info("Loading configuration (using placeholder values)...")
    return {
        "logging_level": "DEBUG",
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "agents": [
            {
                "id": "agent_001",
                "name": "Assistant",
                "loop_type": "simple", # or "two_step"
                "routing_keys": {
                    "adapter_id": ["mock_adapter_api_1"], # Match adapter ID below
                    "conversation_id": ["general", "random"]
                }
            }
        ],
        # Renamed to adapter_apis to reflect client model
        "adapter_apis": [
            {
             "type": "mock_server", # Indicates the type of external API
             "id": "mock_adapter_api_1", # ID for this connection
             "name": "Mock Adapter API",
             "url": "http://localhost:5678", # Example URL where mock server runs
             "auth": None # Optional auth details
            }
            # Add configs for real adapter APIs here (Discord, Slack etc.)
            # {"type": "discord", "id": "discord_1", "url": "...", "auth": {"token": "..."}}
        ]
    }

# Logging Setup
def setup_logging(level_str: str = "INFO"):
    """Configures application logging."""
    log_level = logging.getLevelName(level_str.upper()) 
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Optionally, set lower levels for specific noisy libraries
    # logging.getLogger("openai").setLevel(logging.WARNING)
    logger.info(f"Logging configured at level: {logging.getLevelName(log_level)}")

# Main Async Function
async def main():
    """Initializes and runs the Connectome Host."""
    
    # 1. Load Config
    config = load_config()

    # 2. Setup Logging
    setup_logging(config.get("logging_level", "INFO"))

    # 3. Instantiate Core Host Components
    logger.info("Initializing Host components...")
    host_event_loop = HostEventLoop()
    host_router = HostRouter(host_event_loop)
    logger.info("HostEventLoop and HostRouter created.")

    # 4. Instantiate LLM Provider
    logger.info("Initializing LLM provider...")
    llm_provider: LLMProvider
    try:
        llm_provider = OpenAIProvider(api_key=config.get("openai_api_key"))
        logger.info(f"LLM Provider ({llm_provider.__class__.__name__}) initialized.")
    except ValueError as e:
        logger.error(f"Failed to initialize LLM Provider: {e}. Check API keys/config.")
        return 
    except Exception as e:
        logger.error(f"Unexpected error initializing LLM Provider: {e}", exc_info=True)
        return

    # 5. Instantiate Modules
    logger.info("Initializing Host Modules...")
    modules: Dict[str, Any] = {}

    # --- ActivityClient Module --- 
    # Pass the list of API configs directly
    adapter_api_configs = config.get("adapter_apis", [])
    try:
        activity_client = ActivityClient(host_event_loop, adapter_api_configs)
        modules["ActivityClient"] = activity_client
        logger.info("ActivityClient module initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize ActivityClient: {e}", exc_info=True)
        return # Cannot proceed if ActivityClient fails

    # --- ShellModule(s) --- 
    agent_shells: List[ShellModule] = []
    for agent_conf in config.get("agents", []):
        agent_id = agent_conf.get("id")
        agent_name = agent_conf.get("name")
        loop_type_str = agent_conf.get("loop_type", "simple")
        if not agent_id or not agent_name: continue # Skip invalid
        
        AgentLoopType = None
        if loop_type_str == "simple":
             from elements.elements.components.simple_loop import SimpleRequestResponseLoopComponent
             AgentLoopType = SimpleRequestResponseLoopComponent
        elif loop_type_str == "two_step":
             from elements.elements.components.two_step_loop import TwoStepLoopComponent
             AgentLoopType = TwoStepLoopComponent
    else:
            logger.error(f"Unknown agent loop type '{loop_type_str}' for agent {agent_id}. Skipping agent.")
            continue
            
        logger.info(f"Creating ShellModule for agent '{agent_name}' ({agent_id}) with loop: {loop_type_str}")
        try:
            shell = ShellModule(
                agent_id=agent_id,
                agent_name=agent_name,
                host_config=config,
                llm_provider=llm_provider,
                host_router=host_router,
                host_event_loop=host_event_loop,
                agent_loop_component_type=AgentLoopType
            )
            modules[f"Shell_{agent_id}"] = shell 
            agent_shells.append(shell)
            logger.info(f"ShellModule created for agent {agent_id}.")
            except Exception as e:
            logger.error(f"Failed to create ShellModule for agent {agent_id}: {e}", exc_info=True)

    if not agent_shells:
         logger.error("No agent ShellModules were successfully created. Exiting.")
         return
         
    # 6. Register Modules with Event Loop
    logger.info("Registering modules with HostEventLoop...")
    host_event_loop.register_module("ActivityClient", activity_client)
    for shell in agent_shells:
         host_event_loop.register_module(f"Shell_{shell.agent_id}", shell)
    logger.info("Modules registered.")

    # 7. Start Connections & Event Loop
    logger.info("Starting Adapter API connections...")
    await activity_client.start_connections()
    # TODO: Check connection status before starting loop?
    
    logger.info("Starting Host Event Loop...")
    # Add signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received.")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Run the main loop until stop signal
    try:
        await host_event_loop.run(stop_event)
    except Exception as e:
        logger.critical(f"HostEventLoop crashed: {e}", exc_info=True)
    finally:
        logger.info("Host Event Loop stopped. Performing shutdown...")
        # 8. Handle Shutdown
        # Shutdown ActivityClient first
        if "ActivityClient" in modules:
             try:
                  logger.info("Shutting down ActivityClient connections...")
                  await modules["ActivityClient"].shutdown()
             except Exception as e:
                  logger.error(f"Error shutting down ActivityClient: {e}", exc_info=True)
                  
        # Shutdown Shells
        for module_name, module_instance in modules.items():
             if module_name == "ActivityClient": continue # Already handled
             if hasattr(module_instance, 'shutdown') and asyncio.iscoroutinefunction(module_instance.shutdown):
                  try:
                       logger.info(f"Shutting down module: {module_name}...")
                       await module_instance.shutdown()
                  except Exception as e:
                       logger.error(f"Error shutting down module {module_name}: {e}", exc_info=True)
             elif hasattr(module_instance, 'shutdown'):
                  try:
                       logger.info(f"Shutting down module: {module_name} (sync)...")
                       module_instance.shutdown()
                  except Exception as e:
                       logger.error(f"Error shutting down module {module_name}: {e}", exc_info=True)
                       
        logger.info("Application shutdown complete.")

# Run the main function
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user.") 