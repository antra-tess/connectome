"""
Host's main event loop.

Manages incoming/outgoing queues, timing, and triggers ShellModules based on routing.
"""

import logging
import asyncio
from typing import Dict, Any, Tuple, Callable, Optional, List

# Type hinting imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from host.modules.routing.host_router import HostRouter
    from host.modules.activities.activity_client import ActivityClient
    from host.routing.external_event_router import ExternalEventRouter
    from elements.space_registry import SpaceRegistry

logger = logging.getLogger(__name__)

# Callback type alias
OutgoingActionCallback = Callable[[Dict[str, Any]], None]

# --- Agent cycle constants removed - AgentLoop will self-trigger ---

class HostEventLoop:
    
    def __init__(self,
                 host_router: 'HostRouter',
                 activity_client: 'ActivityClient',
                 external_event_router: 'ExternalEventRouter',
                 space_registry: 'SpaceRegistry'):
        self.running = False
        self.host_router = host_router
        self.activity_client = activity_client
        self.external_event_router = external_event_router
        self.space_registry = space_registry

        self._incoming_event_queue: asyncio.Queue[Tuple[Dict[str, Any], Dict[str, Any]]] = asyncio.Queue()
        self._outgoing_action_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        
        # --- Agent cycle tracking state removed - AgentLoop will self-trigger ---
        
        logger.info("HostEventLoop initialized.")

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
             
    # --- Agent cycle management removed - AgentLoop will self-trigger ---
             
    async def _process_incoming_event_queue(self) -> None:
        """Processes incoming events and routes them via ExternalEventRouter."""
        try:
            while not self._incoming_event_queue.empty():
                event_data, timeline_context = await self._incoming_event_queue.get()
                event_type = event_data.get("event_type")
                event_id = event_data.get("event_id", "unknown")
                logger.debug(f"Processing incoming event {event_id} ({event_type}) from queue. Source Adapter: {event_data.get('source_adapter_id')}")

                # Routing via ExternalEventRouter is primary for adapter events
                if event_data.get("source_adapter_id"):
                    try:
                        await self.external_event_router.route_external_event(event_data, timeline_context)
                    except Exception as ext_router_err:
                        logger.error(f"Error in ExternalEventRouter for event {event_id}: {ext_router_err}", exc_info=True)
                        self._incoming_event_queue.task_done()

                # For internal events without source_adapter_id, log warning
                if not event_data.get("source_adapter_id"):
                    logger.warning(f"Internal event {event_id} received in HEL queue. Processing internal events not yet implemented.")

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
                
                # --- Enhanced Routing Logic with ExternalEventRouter as Preprocessor --- 
                handler_called = False
                if target_module_name == "ActivityClient":
                    # NEW: Route through ExternalEventRouter first for action-specific processing
                    if hasattr(self.external_event_router, 'handle_outgoing_action') and callable(getattr(self.external_event_router, 'handle_outgoing_action')):
                        try:
                            # ExternalEventRouter processes action-specific logic and routes to ActivityClient
                            await self.external_event_router.handle_outgoing_action(action_request)
                            handler_called = True
                        except Exception as router_error:
                            logger.error(f"Error in ExternalEventRouter preprocessing action '{action_type}': {router_error}", exc_info=True)
                    else:
                        # Fallback to direct ActivityClient (for backward compatibility during transition)
                        logger.warning(f"ExternalEventRouter does not have 'handle_outgoing_action'. Falling back to direct ActivityClient routing.")
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
                # 1. Process incoming events and route them
                await self._process_incoming_event_queue()
                
                # 2. Process outgoing actions 
                await self._process_outgoing_action_queue()

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
            
    # --- Agent cycle helper methods removed - AgentLoop will self-trigger ---