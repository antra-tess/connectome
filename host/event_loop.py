"""
Host's main event loop.

Manages incoming/outgoing queues, timing, and triggers ShellModules based on routing.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Tuple, Callable, Optional, Set, List

# Type hinting imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from host.modules.routing.host_router import HostRouter
    from host.modules.activities.activity_client import ActivityClient
    from host.routing.external_event_router import ExternalEventRouter
    from elements.space_registry import SpaceRegistry
    from host.config import AgentConfig

logger = logging.getLogger(__name__)

# Callback type alias
OutgoingActionCallback = Callable[[Dict[str, Any]], None]

# Cycle triggering constants
AGENT_CYCLE_DEBOUNCE_SECONDS = 0.5

# MODIFIED: TRIGGERING_EVENT_TYPES - more specific now
# We will rely on mark_agent_for_cycle to decide based on content
MESSAGE_EVENT_TYPE = "message_received" # Example type for incoming messages
TOOL_RESULT_EVENT_TYPE = "tool_result_received" # Example type for tool results targeting an agent

class HostEventLoop:
    
    def __init__(self,
                 host_router: 'HostRouter',
                 activity_client: 'ActivityClient',
                 external_event_router: 'ExternalEventRouter',
                 space_registry: 'SpaceRegistry',
                 agent_configs: List['AgentConfig']): # NEW: Add agent_configs
        self.running = False
        self.host_router = host_router
        self.activity_client = activity_client
        self.external_event_router = external_event_router
        self.space_registry = space_registry
        
        # NEW: Store agent_configs as a dict for easy lookup
        self.agent_configs: Dict[str, 'AgentConfig'] = {ac.agent_id: ac for ac in agent_configs if ac.agent_id}
        if len(self.agent_configs) != len(agent_configs):
            logger.warning("Some agent configs were missing agent_id and were not stored in HostEventLoop.")

        self._incoming_event_queue: asyncio.Queue[Tuple[Dict[str, Any], Dict[str, Any]]] = asyncio.Queue()
        self._outgoing_action_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        
        self._trigger_event_received_time: Dict[str, float] = {}
        self._pending_agent_cycles: Set[str] = set()
        self._last_agent_cycle_time: Dict[str, float] = {}
        
        logger.info("HostEventLoop initialized.") # Updated log message

    # --- Enqueue Methods (Unchanged) ---
    def enqueue_incoming_event(self, event_data: Dict[str, Any], timeline_context: Dict[str, Any]):
        """Adds an event from the ActivityListener to the incoming processing queue."""
        try:
             self._incoming_event_queue.put_nowait((event_data, timeline_context))
             logger.debug(f"Incoming event {event_data.get('event_id')} enqueued.")
        except asyncio.QueueFull:
             logger.error("HostEventLoop incoming queue is full! Event dropped.")
        except Exception as e:
             logger.error(f"Error enqueuing incoming event: {e}", exc_info=True)

    def get_outgoing_action_callback(self) -> OutgoingActionCallback:
         return self.enqueue_outgoing_action

    async def enqueue_outgoing_action(self, action_request: Dict[str, Any]) -> Dict[str, Any]:
        """Adds an outgoing action request from an InnerSpace component to the queue."""
        # TODO: Add validation for action_request format?
        try:
            self._outgoing_action_queue.put_nowait(action_request)
            logger.debug(f"Outgoing action enqueued: {action_request.get('action_type')}")
            return {"success": True, "status": "Action enqueued successfully."}
        except asyncio.QueueFull:
             logger.error("HostEventLoop outgoing queue is full! Action dropped.")
             return {"success": False, "error": "Outgoing queue full."}
        except Exception as e:
             logger.error(f"Error enqueuing outgoing action: {e}", exc_info=True)
             return {"success": False, "error": f"Error enqueuing action: {e}"}
             
    # --- NEW: Method for conditional agent cycle marking --- 
    def mark_agent_for_cycle(self, agent_id: str, event_payload: Dict[str, Any], current_time: float):
        """
        Conditionally marks an agent for a cycle based on event content (DM/mention).
        This method is intended to be called by ExternalEventRouter or other systems
        that have determined an event is relevant to a specific agent's InnerSpace.
        """
        if not agent_id:
            logger.warning("mark_agent_for_cycle called with no agent_id.")
            return

        agent_config = self.agent_configs.get(agent_id)
        if not agent_config:
            logger.warning(f"Agent config not found for {agent_id}. Cannot determine mention aliases.")
            # Default to triggering if we can't find config? Or not? For now, don't trigger.
            return

        event_type = event_payload.get("event_type")
        payload_data = event_payload.get("payload", {}) # The actual content from the adapter
        is_dm = payload_data.get("is_dm", False)
        text_content = payload_data.get("text_content", "").lower() # Ensure text_content is string
        # Mentions should be a list of IDs/names from the adapter
        mentions = payload_data.get("mentions", []) 
        source_adapter_id = event_payload.get("source_adapter_id")

        should_trigger = False

        if event_type == TOOL_RESULT_EVENT_TYPE:
            # Tool results for an agent should always trigger their cycle if the result is routed to their InnerSpace
            # The ExternalEventRouter should ensure this event_payload is associated with the correct agent_id
            # when calling mark_agent_for_cycle.
            logger.debug(f"Tool result received for agent {agent_id}. Marking for cycle.")
            should_trigger = True
        elif event_type == MESSAGE_EVENT_TYPE:
            if is_dm:
                logger.debug(f"DM received for agent {agent_id}. Marking for cycle.")
                should_trigger = True
            elif source_adapter_id and agent_config.platform_aliases:
                agent_alias_on_platform = agent_config.platform_aliases.get(source_adapter_id)
                if agent_alias_on_platform:
                    agent_alias_lower = agent_alias_on_platform.lower()
                    # Check simple text mention (e.g., @BotName text, BotName: text)
                    # More sophisticated mention parsing might be needed depending on platform specifics
                    # Combine the conditions into a single boolean check
                    is_mentioned_in_text = (
                        f"@{agent_alias_lower}" in text_content or
                        text_content.startswith(f"{agent_alias_lower}:") or
                        agent_alias_lower in text_content # General check, might be too broad
                    )
                    if is_mentioned_in_text:
                        logger.debug(f"Agent {agent_id} (alias: {agent_alias_on_platform}) mentioned in text. Marking for cycle.")
                        should_trigger = True
                    
                    # Check structured mentions if provided by adapter
                    # Assuming mentions is a list of strings (user IDs or names)
                    if not should_trigger and isinstance(mentions, list):
                        for mentioned_user in mentions:
                            if isinstance(mentioned_user, str) and agent_alias_lower == mentioned_user.lower():
                                logger.debug(f"Agent {agent_id} (alias: {agent_alias_on_platform}) found in structured mentions. Marking for cycle.")
                                should_trigger = True
                                break 
                else:
                    logger.debug(f"No alias configured for agent {agent_id} on adapter {source_adapter_id}. Cannot check for text mentions.")
            else:
                if not source_adapter_id:
                    logger.debug(f"Event for agent {agent_id} has no source_adapter_id. Cannot check platform alias.")
                if not agent_config.platform_aliases:
                    logger.debug(f"Agent {agent_id} has no platform_aliases configured.")
        else:
            # For other event types, assume they don't trigger a cycle unless explicitly handled
            logger.debug(f"Event type {event_type} for agent {agent_id} is not a direct message or tool result. Not marking for cycle based on content.")


        if should_trigger:
            self._trigger_event_received_time[agent_id] = current_time
            self._pending_agent_cycles.add(agent_id)
            logger.info(f"Agent {agent_id} marked for cycle trigger due to event: {event_type}.")
        else:
            logger.debug(f"Event {event_type} for agent {agent_id} did not meet criteria for cycle trigger.")

    async def _process_incoming_event_queue(self) -> None:
        """Processes incoming events, routes them, and uses mark_agent_for_cycle."""
        # current_time = time.monotonic() # Moved local, passed to mark_agent_for_cycle
        try:
            while not self._incoming_event_queue.empty():
                event_data, timeline_context = await self._incoming_event_queue.get()
                event_type = event_data.get("event_type")
                event_id = event_data.get("event_id", "unknown")
                payload = event_data.get("payload", {}) # Get payload safely
                logger.debug(f"Processing incoming event {event_id} ({event_type}) from queue. Source Adapter: {event_data.get('source_adapter_id')}")

                # Routing via ExternalEventRouter is primary for adapter events
                if event_data.get("source_adapter_id"):
                    try:
                        # ExternalEventRouter will call self.mark_agent_for_cycle if needed
                        await self.external_event_router.route_external_event(event_data, timeline_context)
                    except Exception as ext_router_err:
                        logger.error(f"Error in ExternalEventRouter for event {event_id}: {ext_router_err}", exc_info=True)
                        self._incoming_event_queue.task_done()

                # Handling for internal events (e.g., tool results not from an adapter)
                # These events must contain enough info to identify the target agent_id
                # Example: a tool result being placed back onto an InnerSpace timeline
                # The InnerSpace itself (or component that recorded it) should then call mark_agent_for_cycle
                # This loop no longer tries to guess the target agent for internal events.
                # If an internal event is put on the queue and isn't an adapter event, it's assumed 
                # that the system component that enqueued it is also responsible for calling mark_agent_for_cycle
                # if that event is supposed to trigger a cycle.
                # The old logic for internal event routing is removed for simplification here.
                # It needs to be handled by the component that generates such internal events.
                # For example, InnerSpace.execute_element_action, when it records a tool_result_received,
                # should call self.host_event_loop.mark_agent_for_cycle(...).
                # This requires InnerSpace to have a reference to the HostEventLoop.

                # For now, if an event reaches here and is not from an adapter, we log it.
                # The component responsible for this internal event needs to be updated
                # to call mark_agent_for_cycle itself.
                if not event_data.get("source_adapter_id"):
                    logger.warning(f"Internal event {event_id} received in HEL queue. Cycle triggering for this event must be handled by the event source by calling mark_agent_for_cycle.")

                    self._incoming_event_queue.task_done()
        except Exception as e:
            logger.exception("Exception during incoming event queue processing.")

    async def _process_outgoing_action_queue(self) -> None:
        """Processes all actions currently in the outgoing queue, routing to target modules."""
        try:
            while not self._outgoing_action_queue.empty():
                action_request = await self._outgoing_action_queue.get()
                action_type = action_request.get("action_type")
                target_module_name = action_request.get("target_module") # Use name for clarity
                # payload = action_request.get("payload") # Payload is part of the full request dict
                logger.debug(f"Processing outgoing action: Type='{action_type}', Target='{target_module_name}'")
                
                # --- Routing Logic --- 
                handler_called = False
                if target_module_name == "ActivityClient": # <<< Added handler routing
                     if hasattr(self.activity_client, 'handle_outgoing_action') and callable(getattr(self.activity_client, 'handle_outgoing_action')):
                          try:
                               await self.activity_client.handle_outgoing_action(action_request) # Pass full request
                               handler_called = True
                          except Exception as client_error:
                               logger.error(f"Error in ActivityClient handling action '{action_type}': {client_error}", exc_info=True)
                     else:
                          logger.error(f"ActivityClient module does not have a callable 'handle_outgoing_action' method.")
                # Add routing for other target modules here...
                # elif target_module_name == "PersistenceModule":
                #     await self.persistence_module.handle_action(...) 
                # elif target_module_name == "TimerService":
                #     await self.timer_service.handle_action(...)
                
                if not handler_called:
                     logger.warning(f"No handler found or registered for outgoing action target module: '{target_module_name}'")
                     
                self._outgoing_action_queue.task_done()
        except Exception as e:
            logger.exception("Exception during outgoing action queue processing.")
            
    # --- Main Loop --- 
    async def run(self):
        """Runs the main event loop."""
        logger.info("Starting Host Event Loop...")
        self.running = True
        
        while self.running:
            try:
                now = time.monotonic() 
                
                # 1. Process incoming events (updates _pending_agent_cycles)
                await self._process_incoming_event_queue()
                
                # 2. Process outgoing actions 
                await self._process_outgoing_action_queue()
                
                # 3. Check for internal timers/scheduled actions (TODO)
                # ...
                
                # 4. Trigger agent cycles for pending agents after debounce
                agents_to_run_now = set()
                for agent_id in list(self._pending_agent_cycles): # Iterate copy for safe removal
                     if self._should_agent_run_cycle(agent_id, now):
                          agents_to_run_now.add(agent_id)
                          self._pending_agent_cycles.remove(agent_id)
                          
                for agent_id in agents_to_run_now:
                    # target_shell = self.shell_modules.get(agent_id) # Removed
                    target_inner_space = self.space_registry.get_inner_space_for_agent(agent_id)
                    if target_inner_space:
                        agent_loop_component = target_inner_space.get_agent_loop_component()
                        if agent_loop_component:
                            logger.info(f"Agent cycle triggered for {agent_id} via InnerSpace's AgentLoopComponent.")
                        self._last_agent_cycle_time[agent_id] = now
                        try:
                                # Assuming AgentLoopComponent has a method like trigger_cycle or run_cognitive_cycle
                                await agent_loop_component.trigger_cycle() 
                        except Exception as cycle_error:
                                logger.error(f"Error during agent cycle trigger for {agent_id} via AgentLoopComponent: {cycle_error}", exc_info=True)
                        else:
                            logger.error(f"Could not trigger agent cycle: AgentLoopComponent not found in InnerSpace for {agent_id}.")
                    else:
                        logger.error(f"Could not trigger agent cycle: InnerSpace for {agent_id} not found in SpaceRegistry.")

                # NEW STEP 5: Process on_frame_end for all registered spaces
                try:
                    all_spaces = self.space_registry.get_spaces() # Returns Dict[str, Space]
                    if all_spaces:
                        logger.debug(f"Processing on_frame_end for {len(all_spaces)} spaces.")
                        for space_id, space in all_spaces.items():
                            try:
                                if hasattr(space, 'on_frame_end') and callable(space.on_frame_end):
                                    # If on_frame_end is async, it needs to be awaited.
                                    # Assuming it's synchronous for now based on previous Space impl.
                                    # If it becomes async in Space, this will need `await space.on_frame_end()`
                                    space.on_frame_end()
                                    logger.debug(f"Called on_frame_end for space: {space.name} ({space_id})")
                                else:
                                    logger.warning(f"Space {space.name} ({space_id}) does not have a callable on_frame_end method.")
                            except Exception as frame_end_err:
                                logger.error(f"Error calling on_frame_end for space {space.name} ({space_id}): {frame_end_err}", exc_info=True)
                    else:
                        logger.debug("No spaces registered to process on_frame_end.")
                except Exception as e_spaces:
                    logger.error(f"Error retrieving spaces from registry for on_frame_end: {e_spaces}", exc_info=True)

                # Prevent busy-waiting
                await asyncio.sleep(0.01) # Adjust sleep time as necessary
            except KeyboardInterrupt:
                 logger.info("Host Event Loop task cancelled.")
                 self.running = False 
            except Exception as e:
                 logger.exception("Exception in event loop iteration.")
            
        logger.info("Host Event Loop finished.")

    def stop(self):
        """Signals the event loop to stop gracefully."""
        if self.running:
            self.running = False
            logger.info("Stopping Host Event Loop...")
        else:
            logger.info("Host Event Loop already stopped or stopping.")
        
    def _should_agent_run_cycle(self, agent_id: str, current_time: float) -> bool:
         """ 
         Determine if a specific agent's cycle should run based on 
         triggering events and a debounce timer for that agent.
         
         Note: This assumes agent_id is in _pending_agent_cycles.
         """
         last_trigger_time = self._trigger_event_received_time.get(agent_id)
         if last_trigger_time is None:
              # Should not happen if agent_id is in _pending_agent_cycles, but check anyway
              logger.warning(f"Checking cycle for {agent_id} but no trigger time recorded.")
              return False 
              
         # Check if debounce period has passed since the *last* trigger event for this agent
         if (current_time - last_trigger_time) >= AGENT_CYCLE_DEBOUNCE_SECONDS:
             logger.debug(f"Debounce period passed for agent {agent_id}. Triggering cycle.")
             return True
             
         # Still within debounce period
         return False 