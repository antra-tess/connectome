import logging
import asyncio
import time
from typing import Dict, Any, Callable, Optional, List

# Base Host Module
from ..base_module import BaseHostModule
# Event loop integration (assuming a way to add tasks/timers)
from ..event_loop import HostEventLoop, OutgoingActionCallback

logger = logging.getLogger(__name__)

class TimerServiceModule(BaseHostModule):
    """
    A Host-level module providing timer functionality for agents.
    Listens for 'set_agent_timer' action requests and schedules callbacks.
    """
    MODULE_NAME = "TimerService"

    def __init__(self, event_loop: HostEventLoop, outgoing_action_sender: OutgoingActionCallback):
        super().__init__(event_loop)
        self._outgoing_action_sender = outgoing_action_sender
        self._active_timers: Dict[str, asyncio.Task] = {}
        logger.info("TimerServiceModule initialized.")

    def get_supported_actions(self) -> List[str]:
        return ["set_agent_timer"]

    async def handle_action_request(self, action_type: str, payload: Dict[str, Any]) -> None:
        """Handles incoming action requests targeted at this module."""
        if action_type == "set_agent_timer":
            await self._handle_set_timer(payload)
        else:
            logger.warning(f"Received unsupported action request type: {action_type}")

    async def _handle_set_timer(self, payload: Dict[str, Any]):
        """Processes the set_agent_timer request."""
        tool_args = payload.get("tool_args", {})
        delay_seconds = tool_args.get("delay_seconds")
        message = tool_args.get("message")
        requesting_element_id = payload.get("requesting_element_id") # InnerSpace ID
        tool_call_id = payload.get("tool_call_id") # From original tool call

        if not isinstance(delay_seconds, (int, float)) or delay_seconds <= 0:
            logger.error(f"Invalid delay_seconds for set_agent_timer: {delay_seconds}")
            # TODO: Send error back to agent? Requires knowing the requesting agent ID
            return
        if not isinstance(message, str) or not message:
             logger.error(f"Invalid or empty message for set_agent_timer.")
             return
        if not requesting_element_id:
            logger.error("Cannot set timer: Missing requesting_element_id (InnerSpace ID) in payload.")
            return

        timer_id = f"timer_{requesting_element_id}_{tool_call_id}_{int(time.time()*1000)}"
        logger.info(f"Scheduling timer '{timer_id}' for {delay_seconds}s for element {requesting_element_id} with message: '{message[:50]}...'")

        # Use asyncio.create_task to schedule the callback
        timer_task = asyncio.create_task(self._timer_callback(delay_seconds, timer_id, requesting_element_id, message))
        self._active_timers[timer_id] = timer_task
        
        # Add callback to remove from active_timers when done
        timer_task.add_done_callback(lambda t: self._active_timers.pop(timer_id, None))

    async def _timer_callback(self, delay: float, timer_id: str, target_element_id: str, message: str):
        """The actual coroutine that waits and then sends the event."""
        try:
            await asyncio.sleep(delay)
            logger.info(f"Timer '{timer_id}' fired for element {target_element_id}.")
            
            # Construct the event to inject back into the target element (InnerSpace)
            # This event should be handled by ContextManagerComponent to add to history.
            event_payload = {
                 "event_type": "agent_timer_fired",
                 "timestamp": int(time.time() * 1000),
                 "data": {
                     "timer_id": timer_id,
                     "message": message
                 }
            }
            
            # We need a way to dispatch this event *to* the specific InnerSpace element.
            # This likely involves the HostEventLoop or the outgoing_action_sender.
            # Option 1: Use outgoing_action_sender if it can route internal events
            # Option 2: Have event_loop.dispatch_to_element(target_element_id, event_payload)
            # Assuming Option 2 for now (needs HostEventLoop implementation)
            if hasattr(self._event_loop, 'dispatch_to_element'):
                 await self._event_loop.dispatch_to_element(target_element_id, event_payload)
            else:
                 logger.error(f"HostEventLoop does not support dispatch_to_element. Cannot deliver timer {timer_id}.")
                 # Fallback: Maybe try sending via outgoing_action_sender if it can act as a message bus?
                 # self._outgoing_action_sender({"target_module": "InnerSpaceHandler", "action_type": "deliver_event", ...})

        except asyncio.CancelledError:
            logger.info(f"Timer '{timer_id}' cancelled.")
        except Exception as e:
            logger.error(f"Error in timer callback for '{timer_id}': {e}", exc_info=True)

    async def shutdown(self): 
        """Cancel any active timers on shutdown."""
        logger.info(f"Shutting down TimerServiceModule. Cancelling {len(self._active_timers)} active timers...")
        for timer_id, task in list(self._active_timers.items()): # Iterate over copy
            if not task.done():
                task.cancel()
                try:
                     await task # Allow cancellation to propagate
                except asyncio.CancelledError:
                     pass # Expected
        self._active_timers.clear()
        logger.info("TimerServiceModule shutdown complete.")

    # TODO: Add mechanism to cancel timers via action request?
    # async def handle_cancel_timer(self, payload: Dict[str, Any]): ... 