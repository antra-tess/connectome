"""
Main entry point for the Host process.
Initializes infrastructure modules and ShellModules for agents.
"""

import logging
import asyncio
import sys
from typing import Dict # Added Dict for typing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Imports ---
# Infrastructure
# Removed SpaceRegistry import
from llm.openai_provider import OpenAIProvider # Or your chosen provider
from host.event_loop import HostEventLoop
from host.modules.routing.host_router import HostRouter # Added
from host.modules.activities.activity_listener import ActivityListener
from host.modules.activities.activity_client import ActivityClient
# Agent/Shell
from host.modules.shell.shell_module import ShellModule
# TODO: Import specific AgentLoopComponent types when created

# Configuration
# from .config import load_config
# --------------- 

async def amain():
    """Asynchronous main entry point."""
    logger.info("Starting Host Process (Async)..." + "\n" + "-"*20)

    # --- Configuration (Placeholder) ---
    host_config = {
         "llm_api_key": "YOUR_OPENAI_API_KEY", 
         "activity_listener_config": {},
         "activity_client_adapter_configs": [
              {"id": "adapter_api_1", "url": "http://localhost:5001", "auth_token": None}
         ],
         "agents": [
              {
                   "id": "agent_001",
                   "name": "Default Agent",
                   # Example routing keys for HostRouter
                   "routing_keys": {
                        "conversation_id": ["conv_abc", "conv_xyz"],
                        "adapter_id": "adapter_api_1" # Route all from this adapter to agent_001
                   }
                   # "agent_loop_type_name": "SimpleRequestResponseLoopComponent"
              }
         ]
    }
    if host_config["llm_api_key"] == "YOUR_OPENAI_API_KEY":
        logger.warning("LLM API Key is set to placeholder!")
    # ------------------------------------

    # --- Initialize Core Infrastructure ---
    try:
        logger.info("Initializing Core Infrastructure...")
        llm_provider = OpenAIProvider(api_key=host_config["llm_api_key"])
        host_router = HostRouter() # Initialize HostRouter
        # Removed SpaceRegistry initialization
        activity_listener = ActivityListener(host_router=host_router) # Pass router if needed for context?
        activity_client = ActivityClient(adapter_configs=host_config["activity_client_adapter_configs"])
        
        # HostEventLoop needs router and shell_modules dict (created below)
        # Initialize later after shell_modules are created
        event_loop = None 
        
        logger.info("Core Infrastructure Initialized (except EventLoop)." + 
                    " EventLoop will be created after ShellModules.")
        
    except Exception as e:
        logger.exception(f"Failed to initialize core infrastructure: {e}")
        return 
    # ------------------------------------

    # --- Initialize ShellModules for Agents ---
    shell_modules: Dict[str, ShellModule] = {} # Use Dict
    if not host_config.get("agents"):
         logger.warning("No agents configured.")
    else:
         for agent_config in host_config["agents"]:
              agent_id = agent_config.get("id")
              agent_name = agent_config.get("name", f"Agent_{agent_id}")
              if not agent_id:
                   logger.error("Skipping agent config with missing 'id'")
                   continue
                   
              agent_loop_type = None # Placeholder
              
              logger.info(f"Initializing ShellModule for agent: {agent_name} ({agent_id})")
              try:
                   # Shell needs router and event loop (to get callback later)
                   # Temporarily create event loop here just to pass, then replace
                   temp_event_loop_ref_for_shell_init = HostEventLoop(host_router, {}, activity_listener, activity_client)
                   
                   shell = ShellModule(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        host_config=host_config, 
                        llm_provider=llm_provider,
                        host_router=host_router, # Pass router
                        host_event_loop=temp_event_loop_ref_for_shell_init, # Pass temporary loop ref
                        agent_loop_component_type=agent_loop_type
                   )
                   if shell.get_inner_space(): 
                        shell_modules[agent_id] = shell # Add to dict
                        logger.info(f"Successfully initialized ShellModule for {agent_id}")
                   else:
                        logger.error(f"ShellModule initialization failed for {agent_id}. Skipping agent.")
              except Exception as e:
                   logger.exception(f"Error initializing ShellModule for {agent_id}: {e}")
                   
    if not shell_modules:
         logger.critical("No agent ShellModules initialized successfully. Host cannot run.")
         return
    # ------------------------------------
    
    # --- Finalize Infrastructure Setup (EventLoop) ---
    try:
         logger.info("Finalizing HostEventLoop initialization...")
         event_loop = HostEventLoop(
             host_router=host_router,
             shell_modules=shell_modules, # Pass the final dict
             activity_listener=activity_listener,
             activity_client=activity_client
         )
         # Inject incoming event callback into listener
         activity_listener.set_event_loop(event_loop)
         logger.info("HostEventLoop fully initialized.")
         
         # NOW, give the actual event loop reference back to the shells if needed 
         # (e.g., if shell needs to directly enqueue outgoing actions, which it shouldn't usually)
         for shell in shell_modules.values():
             shell._host_event_loop = event_loop # Update the reference
             # Also re-inject callback from the *correct* event loop instance
             outgoing_callback = event_loop.get_outgoing_action_callback()
             if shell._inner_space:
                 shell._inner_space.set_outgoing_action_callback(outgoing_callback)
             
    except Exception as e:
         logger.exception(f"Failed to finalize HostEventLoop: {e}")
         return
    # --------------------------------------------------
    
    # --- Start Event Loop --- 
    try:
        logger.info("Starting Host Event Loop...")
        await event_loop.run() # Run the main async loop

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Initiating shutdown...")
    except Exception as e:
         logger.exception(f"Host Event Loop encountered a critical error: {e}")
    finally:
         logger.info("Host process shutting down...")
         if 'event_loop' in locals() and event_loop is not None:
             event_loop.stop() 
         for shell in shell_modules.values(): # Iterate dict
              if hasattr(shell, 'shutdown'):
                   try:
                        await shell.shutdown()
                   except Exception as shell_shutdown_e:
                        logger.error(f"Error shutting down ShellModule for {shell.agent_id}: {shell_shutdown_e}")
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