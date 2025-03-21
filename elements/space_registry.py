"""
Space Registry
Manages the lifecycle of spaces and routes messages to appropriate spaces in the Bot Framework.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Type
import traceback

from elements.elements.base import BaseElement
from .elements.space import Space
from .elements.inner_space import InnerSpace

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpaceRegistry:
    """
    Manages the lifecycle of spaces and routes messages to appropriate spaces.
    
    The SpaceRegistry is responsible for:
    1. Registering and managing spaces
    2. Routing messages to appropriate spaces based on event type
    3. Managing space lifecycle (registration, unregistration)
    4. Handling space state updates and observers
    5. Routing outgoing messages to the activity layer
    """
    
    def __init__(self, context_manager=None):
        """
        Initialize the space registry.
        
        Args:
            context_manager: Optional context manager instance
        """
        self.spaces: Dict[str, Space] = {}
        self._context_manager = context_manager
        self.response_callback = None
        
        # Map of event types to observer callbacks
        self.space_observers: Dict[str, List[Callable]] = {}
        
        # InnerSpace specific attributes
        self.inner_space: Optional[InnerSpace] = None
        
        # Activity layer connection
        self.socket_client = None
        
        logger.info("Space registry initialized")
    
    def register_space(self, space: Space) -> bool:
        """
        Register a space in the registry.
        
        Args:
            space: Space to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        if not isinstance(space, Space):
            logger.error(f"Cannot register {space.__class__.__name__} as it is not a Space")
            return False
            
        space_id = space.id
        
        # Check if a space with this ID already exists
        if space_id in self.spaces:
            logger.warning(f"Space with ID {space_id} already registered")
            return False
            
        # Register the space
        self.spaces[space_id] = space
        
        # Set the registry reference on the space
        if hasattr(space, 'set_registry'):
            space.set_registry(self)
            
        logger.info(f"Registered space: {space.name} ({space_id})")
        
        # Notify observers
        self._notify_observers("space_registered", {
            "space_id": space_id,
            "space_name": space.name,
            "space_type": space.__class__.__name__
        })
        
        return True
    
    def register_inner_space(self, space: InnerSpace) -> bool:
        """
        Register the inner space in the registry.
        
        Args:
            space: InnerSpace to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        if not isinstance(space, InnerSpace):
            logger.error(f"Cannot register {space.__class__.__name__} as inner space. Must be an InnerSpace.")
            return False
            
        # Check if an inner space is already registered
        if self.inner_space is not None:
            logger.warning(f"Inner space already registered: {self.inner_space.id}")
            return False
            
        # Register as inner space
        self.inner_space = space
        
        # Also register as a regular space
        return self.register_space(space)
    
    def unregister_space(self, space_id: str) -> bool:
        """
        Unregister a space from the registry.
        
        Args:
            space_id: ID of the space to unregister
            
        Returns:
            True if unregistration was successful, False otherwise
        """
        # Cannot unregister inner space
        if self.inner_space and space_id == self.inner_space.id:
            logger.error(f"Cannot unregister inner space: {space_id}")
            return False
            
        # Check if the space exists
        if space_id not in self.spaces:
            logger.warning(f"Space with ID {space_id} not found")
            return False
            
        # Get space before removing
        space = self.spaces[space_id]
        
        # Remove the space
        del self.spaces[space_id]
        logger.info(f"Unregistered space: {space.name} ({space_id})")
        
        # Notify observers
        self._notify_observers("space_unregistered", {
            "space_id": space_id,
            "space_name": space.name,
            "space_type": space.__class__.__name__
        })
        
        return True
    
    def get_space(self, space_id: str) -> Optional[Space]:
        """
        Get a space by ID.
        
        Args:
            space_id: ID of the space to get
            
        Returns:
            The space if found, None otherwise
        """
        return self.spaces.get(space_id)
    
    def get_inner_space(self) -> Optional[InnerSpace]:
        """
        Get the inner space.
        
        Returns:
            The inner space if registered, None otherwise
        """
        return self.inner_space
    
    def get_spaces(self) -> Dict[str, Space]:
        """
        Get all registered spaces.
        
        Returns:
            Dictionary mapping space IDs to space objects
        """
        return self.spaces.copy()
    
    def is_inner_space(self, space_id: str) -> bool:
        """
        Check if a space is the inner space.
        
        Args:
            space_id: ID of the space to check
            
        Returns:
            True if the space is the inner space, False otherwise
        """
        return self.inner_space is not None and space_id == self.inner_space.id
    
    def route_message(self, event_data: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Route a message to the appropriate space(s).
        
        Args:
            event_data: Event data to route
            timeline_context: Timeline context for the event
            
        Returns:
            True if the message was successfully routed, False otherwise
        """
        # Extract the event type
        event_type = event_data.get("event_type")
        if not event_type:
            logger.error("Missing event_type in event data")
            return False
            
        # Try to route to inner space first if it's an inner space event
        inner_space_events = ["agent_thought", "agent_action", "internal_event"]
        if event_type in inner_space_events and self.inner_space:
            return self.route_to_inner_space(event_data, timeline_context)
            
        # Route to the appropriate space based on event type
        for space_id, space in self.spaces.items():
            if event_type in space.EVENT_TYPES:
                return self.update_space_state(space_id, event_data, timeline_context)
                
        # If no space handles this event type, log and notify
        logger.warning(f"No space found to handle event type: {event_type}")
        self._notify_observers("unhandled_event", {
            "event_type": event_type,
            "timeline_context": timeline_context
        })
        
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
        if self.inner_space is None:
            logger.error("Cannot route to inner space: No inner space registered")
            return False
            
        return self.inner_space.update_state(event_data, timeline_context)
    
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
        Add an observer for registry events.
        
        Args:
            event_type: Type of event to observe
            callback: Callback function to call when the event occurs
        """
        if event_type not in self.space_observers:
            self.space_observers[event_type] = []
            
        self.space_observers[event_type].append(callback)
        logger.debug(f"Added observer for {event_type} events")
    
    def remove_observer(self, event_type: str, callback: Callable) -> bool:
        """
        Remove an observer for registry events.
        
        Args:
            event_type: Type of event to stop observing
            callback: Callback function to remove
            
        Returns:
            True if the observer was removed, False otherwise
        """
        if event_type not in self.space_observers:
            return False
            
        try:
            self.space_observers[event_type].remove(callback)
            logger.debug(f"Removed observer for {event_type} events")
            return True
        except ValueError:
            return False
    
    def _notify_observers(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Notify observers of an event.
        
        Args:
            event_type: Type of event that occurred
            event_data: Data for the event
        """
        if event_type in self.space_observers:
            for callback in self.space_observers[event_type]:
                try:
                    callback(event_data)
                except Exception as e:
                    logger.error(f"Error in observer callback: {e}")
    
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
        Propagate a message from the inner space to external systems.
        
        This is called by InnerSpace.send_message() when a message needs to be 
        propagated externally. It handles routing the message to the appropriate 
        external adapter based on message type.
        
        Args:
            message: The message to propagate
            timeline_context: The timeline context for this message
            
        Returns:
            True if the message was propagated successfully, False otherwise
        """
        logger.info(f"Propagating message externally: {message.get('type')}")
        
        message_type = message.get("type")
        
        # Format message based on type
        if message_type == "agent_message":
            # Agent message should be sent to external adapter
            conversation_id = message.get("conversationId")
            text = message.get("content", "")
            adapter_id = message.get("adapterId")
            
            # Build external message format
            external_message = {
                "event_type": "send_message",
                "data": {
                    "conversation_id": conversation_id,
                    "text": text,
                    "message_id": message.get("id")
                },
                "adapter_id": adapter_id
            }
            
            # Send through external message system
            return self.send_external_message(external_message)
            
        elif message_type == "tool_call":
            # Tool call is handled internally by the Shell
            # Just notify observers
            self._notify_observers("tool_call", {
                "tool_name": message.get("toolName"),
                "tool_args": message.get("toolArgs"),
                "message_id": message.get("id")
            })
            return True
            
        else:
            logger.warning(f"Unknown message type for propagation: {message_type}")
            return False