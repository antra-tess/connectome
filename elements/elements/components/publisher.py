import logging
from typing import Dict, Any, Optional, Callable # Added Callable

from .base import Component

# Assuming HostEventLoop defines this type alias, or define it here
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Define type alias locally for clarity, referencing HostEventLoop's definition
    OutgoingActionCallback = Callable[[Dict[str, Any]], None]

logger = logging.getLogger(__name__)

class PublisherComponent(Component):
    """
    Component responsible for formatting and publishing outgoing messages/events 
    originating from its parent Element to external systems via the HostEventLoop.
    """
    
    def __init__(self, element: 'BaseElement'):
        super().__init__(element)
        self._outgoing_action_callback: Optional['OutgoingActionCallback'] = None
        logger.debug(f"PublisherComponent initialized for element {element.id}")
        
    def set_outgoing_action_callback(self, callback: 'OutgoingActionCallback'):
         """Sets the callback function used to enqueue outgoing actions."""
         self._outgoing_action_callback = callback
         logger.debug(f"Outgoing action callback set for PublisherComponent on {self.element.id}")

    def publish_message(self, 
                        event_type: str, 
                        payload: Dict[str, Any],
                        adapter_id: Optional[str] = None) -> bool:
        """
        Formats and publishes an outgoing message by enqueuing an action request.
        
        Args:
            event_type: The type of event to send (e.g., "send_message", "edit_message", "typing").
            payload: The data payload for the event, specific to the event_type.
                     Should contain necessary info like conversation_id, text, etc.
            adapter_id: The specific adapter API target (required by ActivityClient).

        Returns:
            True if the action was successfully enqueued, False otherwise.
        """
        if not self._outgoing_action_callback:
            logger.error(f"Cannot publish message from {self.element.id}: Outgoing action callback not set.")
            return False
            
        if not adapter_id:
             logger.error(f"Cannot publish message from {self.element.id}: Target adapter_id is required.")
             return False

        # --- Construct the Standardized Event Payload for ActivityClient --- 
        # This payload is what ActivityClient.send_event_to_adapter expects.
        # It must include the adapter_id itself.
        external_event_payload = {
            "event_type": event_type,
            "payload": payload, # The specific data for the event
            "adapter_id": adapter_id,
            "source_element_id": self.element.id # Add source for traceability
        }
        # ------------------------------------------------------------------

        # --- Construct the Action Request for HostEventLoop --- 
        action_request = {
            "target_module": "activity_client",
            "action_type": "send_external_event", # The action HostEventLoop understands
            "payload": external_event_payload # The data for ActivityClient
        }
        # -----------------------------------------------------
        
        try:
            self._outgoing_action_callback(action_request)
            logger.info(f"Enqueued outgoing '{event_type}' action from {self.element.id} targeting adapter {adapter_id}")
            return True
        except Exception as e:
            logger.error(f"Error enqueuing outgoing action from {self.element.id}: {e}", exc_info=True)
            return False

    # Example usage (might be called by the parent element's action handler):
    # def handle_send_chat_message_action(self, text: str, conversation_id: str, adapter_id: str):
    #     payload = {
    #         "conversation_id": conversation_id,
    #         "text": text
    #     }
    #     self.publish_message("send_message", payload, adapter_id) 