"""
Host's main event loop.

Manages incoming/outgoing queues, timing, and triggers ShellModules based on routing.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Tuple, Callable, Optional, Set

# Type hinting imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from host.modules.routing.host_router import HostRouter
    # from host.modules.shell.shell_module import ShellModule # Removed
    from host.modules.activities.activity_listener import ActivityListener
    from host.modules.activities.activity_client import ActivityClient
    from host.routing.external_event_router import ExternalEventRouter
    from elements.space_registry import SpaceRegistry # Added import

logger = logging.getLogger(__name__)

# Callback type alias
OutgoingActionCallback = Callable[[Dict[str, Any]], None]

# Cycle triggering constants
AGENT_CYCLE_DEBOUNCE_SECONDS = 0.5
TRIGGERING_EVENT_TYPES = {"message_received", "tool_result_available"}

class HostEventLoop:
    
    def __init__(self,
                 host_router: 'HostRouter', # May become vestigial if ExternalEventRouter handles all routing
                 # shell_modules: Dict[str, 'ShellModule'], # Removed
                 activity_listener: 'ActivityListener',
                 activity_client: 'ActivityClient',
                 external_event_router: 'ExternalEventRouter',
                 space_registry: 'SpaceRegistry'): # Added parameter
        self.running = False
        # Infrastructure Modules
        self.host_router: 'HostRouter' = host_router # Keep for now, reassess its role later
        # self.shell_modules: Dict[str, 'ShellModule'] = shell_modules # Removed
        self.activity_listener: 'ActivityListener' = activity_listener
        self.activity_client: 'ActivityClient' = activity_client
        self.external_event_router: 'ExternalEventRouter' = external_event_router
        self.space_registry: 'SpaceRegistry' = space_registry # Added instance variable
        
        # Queues
        self._incoming_event_queue: asyncio.Queue[Tuple[Dict[str, Any], Dict[str, Any]]] = asyncio.Queue()
        self._outgoing_action_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        
        # Cycle triggering state
        self._trigger_event_received_time: Dict[str, float] = {}
        self._pending_agent_cycles: Set[str] = set()
        self._last_agent_cycle_time: Dict[str, float] = {}
        
        logger.info("HostEventLoop initialized with HostRouter and ShellModules.")

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

    def enqueue_outgoing_action(self, action_request: Dict[str, Any]):
        """Adds an outgoing action request from an InnerSpace component to the queue."""
        # TODO: Add validation for action_request format?
        try:
            self._outgoing_action_queue.put_nowait(action_request)
            logger.debug(f"Outgoing action enqueued: {action_request.get('action_type')}")
        except asyncio.QueueFull:
             logger.error("HostEventLoop outgoing queue is full! Action dropped.")
        except Exception as e:
             logger.error(f"Error enqueuing outgoing action: {e}", exc_info=True)
             
    # --- Queue Processing Methods --- 
    async def _process_incoming_event_queue(self) -> None:
         """Processes incoming events, routes them to ShellModules, and marks cycle triggers."""
         now = time.monotonic()
         try:
             while not self._incoming_event_queue.empty():
                 event_data, timeline_context = await self._incoming_event_queue.get()
                 event_type = event_data.get("event_type")
                 event_id = event_data.get("event_id", "unknown")
                 payload = event_data.get("payload", {}) # Get payload safely
                 logger.debug(f"Processing incoming event {event_id} ({event_type}) from queue. Source Adapter: {event_data.get('source_adapter_id')}")

                 # --- New Routing Logic --- 
                 if event_data.get("source_adapter_id"):
                     # This event came from ActivityClient, route through ExternalEventRouter
                     try:
                         # ExternalEventRouter needs the raw event_data from ActivityClient (which includes payload)
                         # and potentially the timeline_context if it's to be used or modified further.
                         # Given ActivityClient sends {} for timeline_context, ExternalEventRouter must handle that.
                         await self.external_event_router.route_external_event(event_data, timeline_context)
                     except Exception as ext_router_err:
                         logger.error(f"Error in ExternalEventRouter for event {event_id}: {ext_router_err}", exc_info=True)
                     self._incoming_event_queue.task_done()
                     continue # Move to next event in queue
                 # --- End New Routing Logic ---

                 # 1. Find target agent via HostRouter (Original logic for non-adapter events)
                 # This part needs to be re-evaluated. If an event isn't from an adapter,
                 # how do we know which agent to trigger? Internal events should carry agent context.
                 # For now, we assume cycle triggering is primarily based on external_event_router determining the agent
                 # or internal events explicitly marking an agent for a cycle.
                 
                 # If ExternalEventRouter or other logic determines an agent needs a cycle based on this event:
                 # This logic would now likely live within ExternalEventRouter which calls back to HEL, or HEL uses a flag from it.
                 # For simplicity, let's assume ExternalEventRouter's processing might update _pending_agent_cycles
                 # or that internal events (like tool_results) directly add to it.

                 # The following original logic is now largely superseded by ExternalEventRouter for adapter events
                 # and needs a clear path for internally generated, agent-specific events.
                 if not event_data.get("source_adapter_id"): # Only if NOT handled by ExternalEventRouter
                     target_agent_id_for_internal = None
                     # Option 1: payload directly contains target_agent_id for internal events
                     if isinstance(payload, dict):
                         target_agent_id_for_internal = payload.get("target_connectome_agent_id")

                     # Option 2: Use host_router if it's still relevant for some internal event types
                     if not target_agent_id_for_internal and self.host_router:
                         routing_context_internal = payload 
                         if timeline_context: 
                             for key, value in timeline_context.items():
                                 routing_context_internal.setdefault(key, value)
                         try:
                             target_agent_id_for_internal = self.host_router.get_target_agent_id(routing_context_internal)
                         except Exception as router_err:
                             logger.error(f"Error calling host_router.get_target_agent_id for internal event: {router_err}", exc_info=True)

                     if target_agent_id_for_internal:
                         # This internal event might be a direct request for a shell to handle something OR a cycle trigger.
                         # If it's a direct event for a component (e.g. tool result), it should be routed via its InnerSpace.receive_event.
                         # This HostEventLoop shouldn't directly call ShellModule.handle_incoming_event anymore.
                         # Instead, such internal events should target an InnerSpace element.
                         # For cycle triggering based on internal events:
                         if event_type in TRIGGERING_EVENT_TYPES:
                             logger.debug(f"Internal event {event_id} ({event_type}) is a cycle trigger for agent {target_agent_id_for_internal}.")
                             self._trigger_event_received_time[target_agent_id_for_internal] = now
                             self._pending_agent_cycles.add(target_agent_id_for_internal)
                     else:
                         logger.warning(f"Internal event {event_id} ({event_type}) could not be routed to a specific agent for cycle triggering.")

                 # The old call to target_shell.handle_incoming_event(event_data, timeline_context) is removed.
                 # Events are now routed by ExternalEventRouter to Space.receive_event for DAG recording and component processing.

                 # Cycle triggering based on event_type (original logic for reference, needs integration with new routing)
                 # If an event (external or internal) has been processed and determined to affect an agent, 
                 # that agent ID should be added to _pending_agent_cycles. 
                 # ExternalEventRouter might need a way to signal back to HostEventLoop or directly add to _pending_agent_cycles.
                 # For now, let's assume if ExternalEventRouter successfully routes a TRIGGERING_EVENT_TYPE to an InnerSpace,
                 # the InnerSpace itself (or a component) might be responsible for adding its agent_id to a list HEL checks,
                 # or ExternalEventRouter informs HEL. This is a subtle point.
                 # A simpler flow: ExternalEventRouter identifies the agent_id and if the event type is triggering, it directly calls a method on HEL
                 # like self.mark_agent_for_cycle(agent_id, now)

                 self._incoming_event_queue.task_done()
         except Exception as e:
              logger.exception("Exception during incoming event queue processing.")
         # No return value needed now, state is tracked in _pending_agent_cycles

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

                # Prevent busy-waiting
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
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