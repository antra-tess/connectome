"""
Space Registry

Manages the collection of active Spaces and Elements.
Provides routing capabilities based on various criteria.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Type
import traceback
import time
import uuid

from elements.elements.base import BaseElement
from .elements.space import Space
from .elements.inner_space import InnerSpace

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpaceRegistry:
    """
    Manages spaces and facilitates event routing.
    (Placeholder implementation)
    """
    def __init__(self):
        self._spaces: Dict[str, Space] = {}
        self._elements: Dict[str, BaseElement] = {}
        self._inner_space: Optional[InnerSpace] = None
        # TODO: Add mapping for conversation_id/adapter_type to space_id
        self._routing_map: Dict[str, str] = {}
        logger.info("SpaceRegistry initialized.")

    def register_space(self, space: Space) -> bool:
        """Registers a Space element."""
        if not isinstance(space, Space):
            logger.error(f"Attempted to register non-Space object: {type(space)}")
            return False
        if space.id in self._spaces:
            logger.warning(f"Space {space.id} already registered. Overwriting.")
        self._spaces[space.id] = space
        self._elements[space.id] = space # Also register as a general element
        logger.info(f"Registered Space: {space.name} ({space.id})")
        # TODO: Update routing map based on space properties?
        return True
        
    def register_inner_space(self, inner_space: InnerSpace) -> bool:
        """Specifically registers the agent's InnerSpace."""
        if self._inner_space and self._inner_space.id != inner_space.id:
             logger.warning(f"Replacing previously registered InnerSpace {self._inner_space.id} with {inner_space.id}")
        self._inner_space = inner_space
        # Also register it as a general space/element
        return self.register_space(inner_space)

    def get_space(self, space_id: str) -> Optional[Space]:
        """Gets a registered Space by its ID."""
        return self._spaces.get(space_id)
        
    def get_element(self, element_id: str) -> Optional[BaseElement]:
         """Gets any registered Element (Space or Object) by its ID."""
         return self._elements.get(element_id)
         
    def get_inner_space(self) -> Optional[InnerSpace]:
         """Gets the registered InnerSpace."""
         return self._inner_space

    def route_event(self, event_data: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Routes an incoming event to the appropriate Space.
        (Placeholder routing logic)
        """
        target_space_id = None
        conversation_id = event_data.get("conversation_id")
        adapter_type = event_data.get("adapter_type")
        timeline_id_from_context = timeline_context.get("timeline_id")

        # --- Placeholder Routing Logic --- 
        # TODO: Implement robust routing based on conversation_id, adapter_type,
        # potentially mapping these to specific Space IDs (e.g., UplinkProxies or ChatElements
        # mounted within InnerSpace).
        
        # Example: If timeline ID directly matches a known space ID?
        if timeline_id_from_context and timeline_id_from_context in self._spaces:
            target_space_id = timeline_id_from_context
        # Example: Simple fallback to InnerSpace for now?
        elif self._inner_space:
            target_space_id = self._inner_space.id
            logger.debug(f"Routing event {event_data.get('event_id')} to InnerSpace (fallback). Context: {timeline_context}")
        # ----------------------------------

        if target_space_id:
            target_space = self.get_space(target_space_id)
            if target_space:
                logger.debug(f"Routing event {event_data.get('event_id')} to Space {target_space_id}")
                # The Space's receive_event handles timeline management and processing
                target_space.receive_event(event_data, timeline_context)
                return True
            else:
                logger.error(f"Routing failed: Target Space {target_space_id} not found in registry.")
        else:
            logger.warning(f"Routing failed: Could not determine target Space for event. Timeline Context: {timeline_context}")
            
        return False

    def unregister_space(self, space_id: str) -> bool:
        """
        Unregister a space from the registry.
        
        Args:
            space_id: ID of the space to unregister
            
        Returns:
            True if unregistration was successful, False otherwise
        """
        # Cannot unregister inner space
        if self._inner_space and space_id == self._inner_space.id:
            logger.error(f"Cannot unregister inner space: {space_id}")
            return False
            
        # Check if the space exists
        if space_id not in self._spaces:
            logger.warning(f"Space with ID {space_id} not found")
            return False
            
        # Get space before removing
        space = self._spaces[space_id]
        
        # Remove the space
        del self._spaces[space_id]
        logger.info(f"Unregistered space: {space.name} ({space_id})")
        
        # Notify observers
        self._notify_observers("space_unregistered", {
            "space_id": space_id,
            "space_name": space.name,
            "space_type": space.__class__.__name__
        })
        
        return True
    
    def get_spaces(self) -> Dict[str, Space]:
        """
        Get all registered spaces.
        
        Returns:
            Dictionary mapping space IDs to space objects
        """
        return self._spaces.copy()
    
    def is_inner_space(self, space_id: str) -> bool:
        """
        Check if a space is the inner space.
        
        Args:
            space_id: ID of the space to check
            
        Returns:
            True if the space is the inner space, False otherwise
        """
        return self._inner_space is not None and space_id == self._inner_space.id
    
    def route_message(self, event_data: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Route a message to the appropriate space(s).
        
        Args:
            event_data: Event data to route
            timeline_context: Timeline context for this message
            
        Returns:
            True if routing was successful, False otherwise
        """
        try:
            event_type = event_data.get("event_type")
            
            # Validate timeline context
            if not timeline_context.get("timeline_id"):
                timeline_context["timeline_id"] = f"timeline_{str(uuid.uuid4())[:8]}"
                logger.warning(f"Generated random timeline ID: {timeline_context['timeline_id']}")
                
            if "is_primary" not in timeline_context:
                timeline_context["is_primary"] = True
                
            logger.info(f"Routing {event_type} event in timeline {timeline_context['timeline_id']}")
            
            # Special handling for clear context events
            if event_type == "clear_context":
                return self._handle_clear_context(event_data, timeline_context)
                
            # Route to the appropriate space(s)
            routed = False
            
            # First try to route to a specific space if adapter_type and conversation_id exist
            adapter_type = event_data.get("adapter_type")
            conversation_id = event_data.get("conversation_id")
            
            if adapter_type and conversation_id:
                # Check if we have a space for this conversation
                for space_id, space in self._spaces.items():
                    # Spaces will have methods for checking if they handle specific conversations
                    if hasattr(space, "handles_conversation") and space.handles_conversation(adapter_type, conversation_id):
                        # Route to this space with timeline context
                        space.receive_event(event_data, timeline_context)
                        routed = True
                        break
                        
            # If not routed to a specific space, route to inner space
            if not routed and self._inner_space:
                self._inner_space.receive_event(event_data, timeline_context)
                routed = True
                
            return routed
            
        except Exception as e:
            logger.error(f"Error routing message: {e}")
            return False
    
    def route_to_inner_space(self, event_data: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Route a message specifically to the inner space.
        
        Args:
            event_data: Event data to route
            timeline_context: Timeline context for the event
            
        Returns:
            True if the message was successfully routed, False otherwise
        """
        if self._inner_space is None:
            logger.error("Cannot route to inner space: No inner space registered")
            return False
            
        return self._inner_space.update_state(event_data, timeline_context)
    
    def update_space_state(self, space_id: str, event_data: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Update the state of a space.
        
        Args:
            space_id: ID of the space to update
            event_data: Event data to update with
            timeline_context: Timeline context for the update
            
        Returns:
            True if the state was successfully updated, False otherwise
        """
        # Get the space
        space = self.get_space(space_id)
        if not space:
            logger.error(f"Cannot update state: Space {space_id} not found")
            return False
            
        # Update the space state
        success = space.update_state(event_data, timeline_context)
        
        # Notify observers
        self._notify_observers("space_state_updated", {
            "space_id": space_id,
            "event_type": event_data.get("event_type"),
            "timeline_context": timeline_context,
            "success": success
        })
        
        return success
    
    def set_context_manager(self, context_manager) -> None:
        """
        Set the context manager.
        
        Args:
            context_manager: Context manager instance
        """
        self._context_manager = context_manager
        
    def register_observer(self, space_id: str, observer) -> None:
        """
        Register an observer for a specific space.
        
        Args:
            space_id: Space ID to observe
            observer: Observer object to notify
        """
        space = self.get_space(space_id)
        if space:
            space.register_observer(observer)
    
    def unregister_observer(self, space_id: str, observer) -> None:
        """
        Remove an observer from a specific space.
        
        Args:
            space_id: Space ID
            observer: Observer to remove
        """
        space = self.get_space(space_id)
        if space:
            space.unregister_observer(observer)
    
    def send_response(self, user_id, message_text, message_id=None, platform=None):
        """Send a response back through the response callback"""
        if self.response_callback:
            response_data = {
                "user_id": user_id,
                "message": message_text,
                "message_id": message_id,
                "platform": platform
            }
            self.response_callback(response_data)
            return True
        else:
            logger.warning("No response callback set in SpaceRegistry")
            return False
    
    def set_response_callback(self, callback: Callable) -> None:
        """
        Set the callback for sending responses back to the Activity Layer.
        
        Args:
            callback: Function to call with response data
        """
        self.response_callback = callback
        logger.debug("Response callback set in SpaceRegistry")
    
    def add_observer(self, event_type: str, callback: Callable) -> None:
        """
        Add an observer for specific event types.
        
        Args:
            event_type: Type of event to observe
            callback: Function to call when the event occurs
        """
        if event_type not in self.space_observers:
            self.space_observers[event_type] = []
            
        if callback not in self.space_observers[event_type]:
            self.space_observers[event_type].append(callback)
            logger.debug(f"Added observer for event type: {event_type}")
    
    def remove_observer(self, event_type: str, callback: Callable) -> None:
        """
        Remove an observer for a specific event type.
        
        Args:
            event_type: Type of event
            callback: Function to remove
        """
        if event_type in self.space_observers and callback in self.space_observers[event_type]:
            self.space_observers[event_type].remove(callback)
            logger.debug(f"Removed observer for event type: {event_type}")
    
    def _notify_observers(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Notify all observers of a specific event type.
        
        Args:
            event_type: Type of event
            event_data: Data about the event
        """
        if event_type in self.space_observers:
            for callback in self.space_observers[event_type]:
                try:
                    callback(event_data)
                except Exception as e:
                    logger.error(f"Error in observer callback for {event_type}: {e}")
        
        # Special handling for attention-related events to notify the Shell
        if event_type in ["attention_requested", "attention_cleared", 
                          "inner_space_attention_requested", "inner_space_attention_cleared"]:
            self._propagate_attention_event(event_type, event_data)
    
    def _propagate_attention_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Propagate attention-related events to the Shell.
        
        This method handles all attention events from Spaces and the InnerSpace,
        ensuring they reach the Shell through the proper channel.
        
        Args:
            event_type: Type of attention event
            event_data: Data about the event
        """
        # Shell is typically connected via the response_callback
        if self.response_callback:
            # Normalize the event to a format that Shell understands
            shell_event = {
                "type": "attention_event",
                "event_type": event_type,
                "timestamp": int(time.time() * 1000)
            }
            
            # Include source element ID for proper routing
            if "source_element_id" in event_data:
                shell_event["source_element_id"] = event_data["source_element_id"]
            elif "element_id" in event_data:
                shell_event["source_element_id"] = event_data["element_id"]
            
            # Include space information
            if "space_id" in event_data:
                shell_event["space_id"] = event_data["space_id"]
            elif "inner_space_id" in event_data:
                shell_event["space_id"] = event_data["inner_space_id"]
                shell_event["is_inner_space"] = True
            
            # Include timeline context if available
            if "timeline_context" in event_data:
                shell_event["timeline_context"] = event_data["timeline_context"]
            
            # Include the original event data
            shell_event["event_data"] = event_data
            
            # Notify the Shell
            try:
                self.response_callback(shell_event)
                logger.debug(f"Propagated attention event to Shell: {event_type}")
            except Exception as e:
                logger.error(f"Error propagating attention event to Shell: {e}")
    
    def set_socket_client(self, socket_client) -> None:
        """
        Set the socket client for external communication.
        
        Args:
            socket_client: SocketIOClient instance
        """
        self.socket_client = socket_client
        logger.info("Socket client set in SpaceRegistry")
    
    def send_external_message(self, message_data: Dict[str, Any]) -> bool:
        """
        Send a message to an external system.
        
        This method serves as a central point for all outgoing messages,
        routing them through the socket client.
        
        Args:
            message_data: Message data to send, with format:
                {
                    "event_type": "send_message",  # Or other event types
                    "data": {
                        "conversation_id": "C123",
                        "text": "Hello world",
                        # Other event-specific fields
                    },
                    "adapter_id": "adapter_id"
                }
                
        Returns:
            True if the message was sent successfully, False otherwise
        """
        if not self.socket_client:
            logger.error("Cannot send external message: No socket client set")
            return False
            
        # Log the outgoing message
        event_type = message_data.get("event_type", "unknown")
        adapter_id = message_data.get("adapter_id", "unknown")
        logger.debug(f"Sending external {event_type} message via adapter {adapter_id}")
        
        # Send through the socket client
        return self.socket_client.send_message(message_data)
    
    def send_typing_indicator(self, adapter_id: str, chat_id: str, is_typing: bool = True) -> bool:
        """
        Send a typing indicator to an external system.
        
        Args:
            adapter_id: ID of the adapter to send through
            chat_id: ID of the chat/conversation
            is_typing: Whether typing is active (True) or not (False)
            
        Returns:
            True if the indicator was sent successfully, False otherwise
        """
        if not self.socket_client:
            logger.error("Cannot send typing indicator: No socket client set")
            return False
            
        logger.debug(f"Sending typing indicator ({is_typing}) for chat {chat_id} via adapter {adapter_id}")
        
        return self.socket_client.send_typing_indicator(adapter_id, chat_id, is_typing)
    
    def send_error(self, adapter_id: str, chat_id: str, error_message: str) -> bool:
        """
        Send an error message to an external system.
        
        Args:
            adapter_id: ID of the adapter to send through
            chat_id: ID of the chat/conversation
            error_message: Error message to send
            
        Returns:
            True if the error was sent successfully, False otherwise
        """
        if not self.socket_client:
            logger.error("Cannot send error message: No socket client set")
            return False
            
        logger.debug(f"Sending error message for chat {chat_id} via adapter {adapter_id}: {error_message}")
        
        return self.socket_client.send_error(adapter_id, chat_id, error_message)
    
    def propagate_message(self, message: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Propagate a message from an element to the activity layer.
        
        Args:
            message: Message to propagate
            timeline_context: Timeline context for this message
            
        Returns:
            True if propagation was successful, False otherwise
        """
        # Only propagate from primary timeline
        if not timeline_context.get("is_primary", False):
            logger.info(f"Not propagating message from non-primary timeline: {timeline_context.get('timeline_id')}")
            return False
            
        # Send to socket client if available
        if self.socket_client:
            try:
                self.socket_client.send_message(message)
                logger.info("Propagated message to activity layer")
                return True
            except Exception as e:
                logger.error(f"Error propagating message: {e}")
                return False
        else:
            logger.warning("No socket client available for message propagation")
            return False