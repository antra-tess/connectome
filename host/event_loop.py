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
    from host.modules.shell.shell_module import ShellModule
    from host.modules.activities.activity_listener import ActivityListener
    from host.modules.activities.activity_client import ActivityClient

logger = logging.getLogger(__name__)

# Callback type alias
OutgoingActionCallback = Callable[[Dict[str, Any]], None]

# Cycle triggering constants
AGENT_CYCLE_DEBOUNCE_SECONDS = 0.5
TRIGGERING_EVENT_TYPES = {"message_received", "tool_result_available"}

class HostEventLoop:
    
    def __init__(self,
                 host_router: 'HostRouter',
                 shell_modules: Dict[str, 'ShellModule'],
                 activity_listener: 'ActivityListener',
                 activity_client: 'ActivityClient'):
        self.running = False
        # Infrastructure Modules
        self.host_router: 'HostRouter' = host_router
        self.shell_modules: Dict[str, 'ShellModule'] = shell_modules
        self.activity_listener: 'ActivityListener' = activity_listener
        self.activity_client: 'ActivityClient' = activity_client
        
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
                 logger.debug(f"Processing incoming event {event_id} ({event_type}) from queue.")
                 
                 # 1. Find target agent via HostRouter
                 target_agent_id = self.host_router.get_target_agent_id(timeline_context)
                 
                 if not target_agent_id:
                      # TODO: Implement default routing or error handling for unroutable events
                      logger.warning(f"Could not route event {event_id}: No target agent found for context {timeline_context}")
                      self._incoming_event_queue.task_done()
                      continue
                      
                 # 2. Get the corresponding ShellModule
                 target_shell = self.shell_modules.get(target_agent_id)
                 if not target_shell:
                      logger.error(f"Could not route event {event_id}: Target agent '{target_agent_id}' not found in shell_modules.")
                      self._incoming_event_queue.task_done()
                      continue
                      
                 # 3. Call the ShellModule's handler
                 try:
                      # Pass the event handling off to the specific shell
                      await target_shell.handle_incoming_event(event_data, timeline_context)
                 except Exception as shell_event_error:
                      logger.error(f"Error in ShellModule {target_agent_id} while handling event {event_id}: {shell_event_error}", exc_info=True)
                      # Decide if this error should prevent cycle trigger
                      
                 # 4. Check if this event type should trigger an agent cycle for this agent
                 if event_type in TRIGGERING_EVENT_TYPES:
                     logger.debug(f"Event {event_id} ({event_type}) is a cycle trigger for agent {target_agent_id}.")
                     self._trigger_event_received_time[target_agent_id] = now
                     self._pending_agent_cycles.add(target_agent_id) # Mark agent for potential cycle
                         
                 self._incoming_event_queue.task_done()
         except Exception as e:
              logger.exception("Exception during incoming event queue processing.")
         # No return value needed now, state is tracked in _pending_agent_cycles

    async def _process_outgoing_action_queue(self) -> None:
        """Processes all actions currently in the outgoing queue."""
        try:
            while not self._outgoing_action_queue.empty():
                action_request = await self._outgoing_action_queue.get()
                logger.debug(f"Processing outgoing action: {action_request.get('action_type')}")
                action_type = action_request.get("action_type")
                target_module = action_request.get("target_module")
                payload = action_request.get("payload")
                if target_module == "activity_client" and action_type == "send_external_event":
                     if isinstance(payload, dict):
                          success = self.activity_client.send_event_to_adapter(payload)
                          if not success:
                               logger.warning(f"Failed to send external event via ActivityClient: {payload.get('event_type')}")
                     else:
                          logger.error(f"Invalid payload format for send_external_event: {payload}")
                else:
                     logger.warning(f"No handler found for outgoing action: Type='{action_type}', Target='{target_module}'")
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
                     target_shell = self.shell_modules.get(agent_id)
                     if target_shell:
                          logger.info(f"Agent cycle triggered for {agent_id}.")
                          self._last_agent_cycle_time[agent_id] = now
                          try:
                               await target_shell.trigger_agent_cycle()
                          except Exception as cycle_error:
                               logger.error(f"Error during agent cycle trigger for {agent_id}: {cycle_error}", exc_info=True)
                     else:
                          logger.error(f"Could not trigger agent cycle: ShellModule for {agent_id} not found.")

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