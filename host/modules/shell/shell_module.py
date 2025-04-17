"""
Shell Module

Represents the configuration and runtime setup for a specific agent 
within the Host environment.
"""

import logging
import asyncio # Added for async methods
from typing import Dict, Any, Optional, Type

# Core element imports
from elements.elements.inner_space import InnerSpace
# Import BaseAgentLoopComponent from agent_loop
from elements.elements.components.agent_loop import BaseAgentLoopComponent
# Import SimpleRequestResponseLoopComponent from simple_loop
from elements.elements.components.simple_loop import SimpleRequestResponseLoopComponent

# LLM provider interface
from llm.provider_interface import LLMProviderInterface

# Host infrastructure imports
from host.event_loop import HostEventLoop, OutgoingActionCallback
from host.modules.routing.host_router import HostRouter # Added

logger = logging.getLogger(__name__)

class ShellModule:
    """
    Manages the instantiation and configuration of a single agent's InnerSpace
    and its core components, acting as the bridge between the Host environment
    and the agent's subjective world.
    """
    
    def __init__(self,
                 agent_id: str, # Unique ID for this agent/shell instance
                 agent_name: str,
                 host_config: Dict[str, Any], # Full or partial host config
                 llm_provider: LLMProviderInterface,
                 host_router: HostRouter, # Added
                 host_event_loop: HostEventLoop,
                 # Make the type hint more specific
                 agent_loop_component_type: Optional[Type[BaseAgentLoopComponent]] = None 
                ):
        """
        Initializes the ShellModule for a specific agent.
        
        Args:
            agent_id: A unique identifier for this agent instance.
            agent_name: A human-readable name for the agent.
            host_config: Configuration relevant to this shell/agent.
            llm_provider: The LLM provider instance.
            host_router: The HostRouter instance.
            host_event_loop: The main HostEventLoop.
            agent_loop_component_type: The specific AgentLoopComponent class to use.
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self._host_config = host_config
        self._llm_provider = llm_provider
        self._host_router = host_router # Store router
        self._host_event_loop = host_event_loop
        self._inner_space: Optional[InnerSpace] = None
        
        # Use provided type or default to SimpleRequestResponseLoopComponent
        self._agent_loop_type = agent_loop_component_type or SimpleRequestResponseLoopComponent
        
        logger.info(f"Initializing ShellModule for agent: {self.agent_name} ({self.agent_id})")
        
        self._initialize_agent()
        
        # Register routes after agent is initialized (e.g., using config)
        self._register_routing()
        
    def _initialize_agent(self):
        """Creates and configures the agent's InnerSpace."""
        logger.info(f"[{self.agent_id}] Initializing InnerSpace...")
        try:
            # Pass the selected agent_loop_component_type to InnerSpace
            self._inner_space = InnerSpace(
                id=f"{self.agent_id}_inner_space",
                name=f"{self.agent_name}'s Mind",
                llm_provider=self._llm_provider,
                agent_loop_component_type=self._agent_loop_type 
            )
            logger.info(f"[{self.agent_id}] InnerSpace instance created: {self._inner_space.id}")

            # Inject Outgoing Callback (happens after InnerSpace and its components are created)
            outgoing_action_callback = self._host_event_loop.get_outgoing_action_callback()
            if self._inner_space: # Check if InnerSpace creation succeeded
                 self._inner_space.set_outgoing_action_callback(outgoing_action_callback)
                 logger.info(f"[{self.agent_id}] Outgoing action callback injected into InnerSpace.")
            else:
                 # Fail initialization if InnerSpace couldn't be created
                 raise ValueError("InnerSpace creation failed, cannot inject callback.")
                 
            # TODO: Perform any other Shell-specific setup for the agent here
            # e.g., loading specific memories, setting initial state via components?
            
            logger.info(f"[{self.agent_id}] Agent InnerSpace initialization complete.")

        except Exception as e:
            logger.exception(f"[{self.agent_id}] Critical error during agent initialization: {e}")
            self._inner_space = None # Ensure partial state isn't left

    def _register_routing(self):
        """Registers routes for this agent with the HostRouter based on configuration.
        Primarily registers based on 'adapter_id' found in routing_keys.
        """
        if not self._inner_space: # Don't register if init failed
            logger.warning(f"[{self.agent_id}] Skipping route registration: InnerSpace not initialized.")
            return

        logger.info(f"[{self.agent_id}] Registering routes...")
        agent_config = next((agent for agent in self._host_config.get('agents', []) if agent.get('id') == self.agent_id), None)

        if not agent_config:
            logger.warning(f"[{self.agent_id}] No configuration found for agent. Cannot register routes.")
            return

        routing_config = agent_config.get('routing_keys', {})
        registered_any = False

        # --- Handle specific 'adapter_id' key --- 
        adapter_id_val = routing_config.get("adapter_id")
        if adapter_id_val:
            if isinstance(adapter_id_val, str):
                context_key = "adapter_id" # The key to register under
                self._host_router.register_agent_route(self.agent_id, context_key, adapter_id_val)
                logger.info(f"[{self.agent_id}] Registered route for adapter_id: {adapter_id_val}")
                registered_any = True
                # Remove so it's not processed generically
                routing_config.pop("adapter_id", None) 
            elif isinstance(adapter_id_val, list):
                 # Allow registering multiple adapters if needed
                 context_key = "adapter_id"
                 for ad_id in adapter_id_val:
                      if isinstance(ad_id, str):
                           self._host_router.register_agent_route(self.agent_id, context_key, ad_id)
                           logger.info(f"[{self.agent_id}] Registered route for adapter_id: {ad_id}")
                           registered_any = True
                      else:
                           logger.warning(f"[{self.agent_id}] Invalid adapter_id found in list: {ad_id}. Skipping.")
                 routing_config.pop("adapter_id", None)
            else:
                 logger.warning(f"[{self.agent_id}] 'adapter_id' key in routing_keys has invalid type {type(adapter_id_val)}. Expected string or list. Skipping.")
                 routing_config.pop("adapter_id", None)

        # --- Handle other generic routing keys --- 
        logger.debug(f"[{self.agent_id}] Processing remaining generic routing keys: {list(routing_config.keys())}")
        for context_key, context_values in routing_config.items():
            if isinstance(context_values, list):
                for value in context_values:
                    if value is not None:
                        self._host_router.register_agent_route(self.agent_id, context_key, value)
                        registered_any = True
            elif context_values is not None:
                self._host_router.register_agent_route(self.agent_id, context_key, context_values)
                registered_any = True

        if not registered_any:
            logger.warning(f"[{self.agent_id}] No valid routes were registered based on configuration.")

    async def handle_incoming_event(self, event_data: Dict[str, Any], timeline_context: Dict[str, Any]):
        """
        Handles an incoming event routed to this agent's shell.
        Delegates processing to the InnerSpace.
        """
        if not self._inner_space:
            logger.error(f"[{self.agent_id}] Cannot handle event: InnerSpace not initialized.")
            return
            
        logger.debug(f"[{self.agent_id}] Shell handling incoming event: {event_data.get('event_type')}")
        try:
            # Delegate to InnerSpace's event handling logic
            # InnerSpace.receive_event is synchronous based on current code
            self._inner_space.receive_event(event_data, timeline_context)
        except Exception as e:
             logger.error(f"[{self.agent_id}] Error during InnerSpace event processing: {e}", exc_info=True)
             # Decide how to handle errors during event processing

    async def trigger_agent_cycle(self):
        """
        Triggers the agent's internal reasoning cycle.
        Gets the AgentLoopComponent from InnerSpace and calls its run_cycle.
        """
        if not self._inner_space:
            logger.error(f"[{self.agent_id}] Cannot trigger cycle: InnerSpace not initialized.")
            return
            
        logger.debug(f"[{self.agent_id}] Triggering agent cycle via ShellModule...")
        try:
            # Use the helper method on InnerSpace
            agent_loop_component = self._inner_space.get_agent_loop_component() 
            
            if agent_loop_component:
                 logger.info(f"[{self.agent_id}] Found AgentLoopComponent: {agent_loop_component.__class__.__name__}. Running cycle...")
                 await agent_loop_component.run_cycle()
                 logger.info(f"[{self.agent_id}] Agent cycle execution finished.")
            else:
                 logger.error(f"[{self.agent_id}] Cannot run cycle: AgentLoopComponent not found on InnerSpace.")
                      
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error during agent cycle execution triggered by Shell: {e}", exc_info=True)

    def get_inner_space(self) -> Optional[InnerSpace]:
        """Returns the managed InnerSpace instance."""
        return self._inner_space

    async def shutdown(self):
        """Perform graceful shutdown procedures for this agent/shell."""
        logger.info(f"[{self.agent_id}] Shutting down ShellModule...")
        # TODO: Unregister routes?
        if self._inner_space:
             # Example: unregister routes associated with this agent
             agent_specific_config = next((agent for agent in self._host_config.get('agents', []) if agent.get('id') == self.agent_id), None)
             if agent_specific_config and 'routing_keys' in agent_specific_config:
                 for context_key, context_values in agent_specific_config['routing_keys'].items():
                      if isinstance(context_values, list):
                           for value in context_values:
                                self._host_router.unregister_agent_route(context_key, value)
                      else:
                           self._host_router.unregister_agent_route(context_key, context_values)
        logger.info(f"[{self.agent_id}] ShellModule shutdown complete.") 