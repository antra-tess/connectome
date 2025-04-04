"""
Host's main event loop.

Coordinates activities, agent cycles, and event processing for both
incoming external events and outgoing internal action requests.
"""

import logging
import asyncio
import time
from collections import deque
from typing import Dict, Any, Tuple, Callable, Optional

# Assuming imports for type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from elements.space_registry import SpaceRegistry
    from elements.elements.inner_space import InnerSpace
    from elements.elements.components.hud_component import HUDComponent
    from host.modules.activities.activity_listener import ActivityListener
    from host.modules.activities.activity_client import ActivityClient

logger = logging.getLogger(__name__)

# Define a type alias for the outgoing action callback signature
OutgoingActionCallback = Callable[[Dict[str, Any]], None]

# --- Constants for Cycle Triggering --- 
AGENT_CYCLE_DEBOUNCE_SECONDS = 0.5 # Wait this long after a trigger event before running cycle
TRIGGERING_EVENT_TYPES = {"message_received", "tool_result_available"} # Events that prompt agent thought
# ------------------------------------

class HostEventLoop:
    
    def __init__(self,
                 registry: 'SpaceRegistry',
                 inner_space: 'InnerSpace',
                 activity_listener: 'ActivityListener',
                 activity_client: 'ActivityClient'):
        self.running = False
        # Store references to needed components/modules
        self.registry: 'SpaceRegistry' = registry
        self.inner_space: 'InnerSpace' = inner_space 
        self.activity_listener: 'ActivityListener' = activity_listener
        self.activity_client: 'ActivityClient' = activity_client
        
        # Queue for incoming events from external adapters (via Listener)
        self._incoming_event_queue: asyncio.Queue[Tuple[Dict[str, Any], Dict[str, Any]]] = asyncio.Queue()
        self._last_event_processed_time = 0
        
        # Queue for outgoing action requests from InnerSpace components
        self._outgoing_action_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        
        # Cycle triggering state
        self._trigger_event_received_time: Optional[float] = None # Tracks specific trigger events
        self._agent_cycle_pending = False # Flag if cycle is waiting for debounce
        self._last_agent_cycle_time: float = 0
        
        logger.info("HostEventLoop initialized with incoming and outgoing queues.")

    # --- Incoming Event Handling ---
    def enqueue_incoming_event(self, event_data: Dict[str, Any], timeline_context: Dict[str, Any]):
        """Adds an event from the ActivityListener to the incoming processing queue."""
        try:
             self._incoming_event_queue.put_nowait((event_data, timeline_context))
             logger.debug(f"Incoming event {event_data.get('event_id')} enqueued.")
        except asyncio.QueueFull:
             logger.error("HostEventLoop incoming queue is full! Event dropped.")
        except Exception as e:
             logger.error(f"Error enqueuing incoming event: {e}", exc_info=True)

    async def _process_incoming_event_queue(self) -> bool:
         """Processes incoming events, returns True if a trigger event was processed."""
         processed_trigger_event = False
         now = time.monotonic()
         try:
             while not self._incoming_event_queue.empty():
                 event_data, timeline_context = await self._incoming_event_queue.get()
                 event_type = event_data.get("event_type")
                 event_id = event_data.get("event_id", "unknown")
                 logger.debug(f"Processing incoming event {event_id} ({event_type}) from queue.")
                 
                 success = self.registry.route_event(event_data, timeline_context)
                 if success:
                      self._last_event_processed_time = now
                      # Check if this specific event type should trigger an agent cycle
                      if event_type in TRIGGERING_EVENT_TYPES:
                           logger.debug(f"Event {event_id} ({event_type}) is a cycle trigger.")
                           self._trigger_event_received_time = now
                           self._agent_cycle_pending = True # Mark that a cycle is potentially needed
                           processed_trigger_event = True
                 else:
                      logger.warning(f"Incoming event routing failed for {event_id}")
                 self._incoming_event_queue.task_done()
         except Exception as e:
              logger.exception("Exception during incoming event queue processing.")
         return processed_trigger_event # Return if any trigger event was processed in this batch
         
    # --- Outgoing Action Handling ---
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
             
    def get_outgoing_action_callback(self) -> OutgoingActionCallback:
         """Returns a reference to the enqueue_outgoing_action method."""
         return self.enqueue_outgoing_action

    async def _process_outgoing_action_queue(self) -> None:
        """Processes all actions currently in the outgoing queue."""
        try:
            while not self._outgoing_action_queue.empty():
                action_request = await self._outgoing_action_queue.get()
                logger.debug(f"Processing outgoing action: {action_request.get('action_type')}")
                
                action_type = action_request.get("action_type")
                target_module = action_request.get("target_module")
                payload = action_request.get("payload")
                
                # --- Dispatch Logic --- 
                if target_module == "activity_client" and action_type == "send_external_event":
                     if isinstance(payload, dict):
                          success = self.activity_client.send_event_to_adapter(payload)
                          if not success:
                               logger.warning(f"Failed to send external event via ActivityClient: {payload.get('event_type')}")
                     else:
                          logger.error(f"Invalid payload format for send_external_event: {payload}")
                else:
                     logger.warning(f"No handler found for outgoing action: Type='{action_type}', Target='{target_module}'")
                # --- End Dispatch Logic ---
                
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
                now = time.monotonic() # Get current time once per loop iteration
                
                # 1. Process incoming events 
                await self._process_incoming_event_queue()
                
                # 2. Process outgoing actions
                await self._process_outgoing_action_queue()
                
                # 3. Check for internal timers/scheduled actions (TODO)
                # ...
                
                # 4. Check if agent needs to run a cycle 
                agent_should_run = self._agent_needs_cycle(now)
                if agent_should_run:
                    logger.info("Agent cycle triggered.") # Use info level for cycle start
                    self._agent_cycle_pending = False # Consume the pending state
                    self._last_agent_cycle_time = now
                    
                    hud_component: 'HUDComponent' = self.inner_space.get_hud()
                    if hud_component and hud_component.is_enabled:
                        try:
                             await hud_component.run_agent_cycle()
                             logger.info("Agent cycle completed.")
                        except Exception as cycle_error:
                             logger.error(f"Error during agent cycle execution: {cycle_error}", exc_info=True)
                    elif not hud_component:
                         logger.warning("Agent needs cycle, but HUD component not found on InnerSpace.")
                    # Don't reset _trigger_event_received_time here, let debounce handle it

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
        
    def _agent_needs_cycle(self, current_time: float) -> bool:
         """ 
         Determine if the agent's reasoning cycle should run based on 
         triggering events and a debounce timer.
         """
         if not self._agent_cycle_pending:
             return False # No trigger event received since last cycle check
             
         # Check if debounce period has passed since the *last* trigger event
         if self._trigger_event_received_time is not None and \
            (current_time - self._trigger_event_received_time) >= AGENT_CYCLE_DEBOUNCE_SECONDS:
             logger.debug(f"Debounce period passed. Triggering agent cycle.")
             return True
             
         # Still within debounce period or no trigger time recorded (shouldn't happen if pending is true)
         return False 