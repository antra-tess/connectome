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

from elements.elements.base import BaseElement, MountType
# Removed direct imports of Space and InnerSpace from here

# TYPE_CHECKING block for Space and InnerSpace imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
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
    _instance: Optional["SpaceRegistry"] = None  # Class variable to hold the singleton instance

    @classmethod
    def get_instance(cls) -> "SpaceRegistry":
        """Gets the singleton instance of SpaceRegistry, creating it if necessary."""
        if cls._instance is None:
            logger.info("Creating new SpaceRegistry singleton instance.")
            cls._instance = cls()  # Call __init__ indirectly here
        return cls._instance

    def __init__(self):
        """Initialize the SpaceRegistry. Should typically only be called once via get_instance()."""
        # Optional: Enforce singleton nature strictly
        if SpaceRegistry._instance is not None and SpaceRegistry._instance is not self:
            # This condition check `is not self` is to allow the first call from get_instance to proceed.
            raise RuntimeError("SpaceRegistry is a singleton. Use SpaceRegistry.get_instance() to access it.")
        
        self._spaces: Dict[str, "Space"] = {}
        self._elements: Dict[str, BaseElement] = {}
        self._agent_inner_spaces: Dict[str, "InnerSpace"] = {}
        # TODO: Add mapping for conversation_id/adapter_type to space_id
        self._routing_map: Dict[str, str] = {}
        self.response_callback: Optional[Callable] = None
        self.space_observers: Dict[str, List[Callable]] = {}
        self.socket_client: Optional[Any] = None
        logger.info("SpaceRegistry initialized.")

    def register_space(self, space: "Space") -> bool:
        """Registers a Space element."""
        from .elements.space import Space
        if not isinstance(space, Space):
            logger.error(f"Attempted to register non-Space object: {type(space)}")
            return False
        if space.id in self._spaces:
            return True
            logger.warning(f"Space {space.id} already registered. Overwriting.")
        self._spaces[space.id] = space
        self._elements[space.id] = space # Also register as a general element
        logger.info(f"Registered Space: {space.name} ({space.id})")
        # TODO: Update routing map based on space properties?
        return True
        
    def register_inner_space(self, inner_space: "InnerSpace", agent_id: str) -> bool:
        """Specifically registers an agent's InnerSpace."""
        from .elements.inner_space import InnerSpace
        if not isinstance(inner_space, InnerSpace):
            logger.error(f"Attempted to register non-InnerSpace object as InnerSpace: {type(inner_space)}")
            return False
        if agent_id in self._agent_inner_spaces and self._agent_inner_spaces[agent_id].id != inner_space.id:
             logger.warning(f"Replacing previously registered InnerSpace for agent {agent_id} (ID: {self._agent_inner_spaces[agent_id].id}) with new InnerSpace (ID: {inner_space.id})")
        
        self._agent_inner_spaces[agent_id] = inner_space
        logger.info(f"Registered InnerSpace for agent_id '{agent_id}': {inner_space.name} ({inner_space.id})")
        # Also register it as a general space/element
        return self.register_space(inner_space)

    def get_space(self, space_id: str) -> Optional["Space"]:
        """Gets a registered Space by its ID."""
        return self._spaces.get(space_id)
        
    def get_element(self, element_id: str) -> Optional[BaseElement]:
         """Gets any registered Element (Space or Object) by its ID."""
         return self._elements.get(element_id)
         
    def get_inner_space_for_agent(self, agent_id: str) -> Optional["InnerSpace"]:
        """Gets the InnerSpace registered for a specific agent_id."""
        return self._agent_inner_spaces.get(agent_id)

    def find_element_deep(self, element_id: str) -> Optional[BaseElement]:
        """
        Finds an element by its ID, searching top-level elements and then
        recursively within each registered Space.
        """
        # 1. Check top-level elements (which includes all registered Spaces)
        element = self._elements.get(element_id)
        if element:
            return element

        # 2. If not found, search within each registered Space
        for space in self._spaces.values():
            # Space.get_element_by_id checks the space itself and its direct children
            found_in_space = space.get_element_by_id(element_id)
            if found_in_space:
                return found_in_space
        
        logger.warning(f"[SpaceRegistry] Element with ID '{element_id}' not found through deep search.")
        return None

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
        # This fallback logic needs to be reconsidered in a multi-agent setup.
        # It should probably try to route to an InnerSpace relevant to the event context
        # or not route at all if no specific target is found.
        # For now, if there's only one InnerSpace, it might be okay.
        # Let's assume ExternalEventRouter will handle the primary routing decision.
        # elif self._agent_inner_spaces: # Check if any inner spaces exist
        #     # This is problematic: which InnerSpace to route to?
        #     # Commenting out this general fallback for now.
        #     # first_agent_inner_space = next(iter(self._agent_inner_spaces.values()))
        #     # target_space_id = first_agent_inner_space.id
        #     logger.debug(f"Routing event {event_data.get('event_id')} to an InnerSpace (fallback - needs agent context). Context: {timeline_context}")
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
        # Check if the space to be unregistered is any of the agent's InnerSpaces
        agent_id_for_this_space = None
        for aid, inner_s in self._agent_inner_spaces.items():
            if inner_s.id == space_id:
                agent_id_for_this_space = aid
                break
        
        if agent_id_for_this_space:
            logger.info(f"Unregistering InnerSpace for agent {agent_id_for_this_space} (Space ID: {space_id})")
            del self._agent_inner_spaces[agent_id_for_this_space]
            # Continue to remove from general _spaces and _elements
            
        # Check if the space exists in the general list
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
    
    def get_spaces(self) -> Dict[str, "Space"]:
        """
        Get all registered spaces.
        
        Returns:
            Dictionary mapping space IDs to space objects
        """
        return self._spaces.copy()
    
    def is_inner_space(self, space_id: str) -> bool:
        """
        Check if a space is an inner space for any registered agent.
        
        Args:
            space_id: ID of the space to check
            
        Returns:
            True if the space is an inner space, False otherwise
        """
        return any(inner_s.id == space_id for inner_s in self._agent_inner_spaces.values())
    
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
            # This needs to be agent-specific now.
            # The ExternalEventRouter will handle routing to the correct InnerSpace.
            # This general fallback is problematic.
            # if not routed and self._agent_inner_spaces:
            #     # Which InnerSpace? For now, this method might be deprecated for external routing.
            #     # first_agent_inner_space = next(iter(self._agent_inner_spaces.values()))
            #     # first_agent_inner_space.receive_event(event_data, timeline_context)
            #     # routed = True
            #     logger.warning("SpaceRegistry.route_message fallback to InnerSpace is ambiguous in multi-agent setup.")
                
            return routed
            
        except Exception as e:
            logger.error(f"Error routing message: {e}")
            return False
    
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

    def get_or_create_shared_space(self, 
                                   identifier: str, 
                                   name: Optional[str] = None, 
                                   description: Optional[str] = "Shared Space",
                                   adapter_id: Optional[str] = None,             # NEW
                                   external_conversation_id: Optional[str] = None, # NEW
                                   metadata: Optional[Dict[str, Any]] = None) -> Optional["Space"]:
        """
        Gets a SharedSpace by its identifier, or creates it if it doesn't exist.
        A SharedSpace is an instance of the base Space class.

        Args:
            identifier: Unique identifier for the SharedSpace.
            name: Human-readable name for the SharedSpace if created. Defaults to identifier.
            description: Description for the SharedSpace if created.
            adapter_id: Optional adapter ID for the space. If not provided, attempts to parse from identifier.
            external_conversation_id: Optional external conversation ID. If not provided, attempts to parse from identifier.
            metadata: Optional additional metadata for the space if created (currently not directly used by Space constructor).

        Returns:
            The existing or newly created Space instance, or None if creation failed.
        """
        from .elements.space import Space
        existing_space = self.get_space(identifier)
        if existing_space:
            # Ensure the chat interface element exists even for existing spaces if somehow missed
            # This is a bit of a safeguard, ideally it's created with the space.
            # Re-evaluate if this check is still needed or how it should work with factory
            if not existing_space.get_mounted_element(f"{identifier}_chat_interface"):
                logger.warning(f"Existing SharedSpace {identifier} missing chat_interface. Attempting to add it via factory.")
                self._create_chat_interface_in_shared_space_with_factory(existing_space)
            return existing_space

        logger.info(f"SharedSpace with identifier '{identifier}' not found. Creating new one.")
        space_name = name if name else identifier
        
        # Attempt to parse adapter_id and external_conversation_id from identifier
        # if not explicitly provided.
        parsed_adapter_id = adapter_id
        parsed_external_conv_id = external_conversation_id

        if not parsed_adapter_id and not parsed_external_conv_id:
            parts = identifier.split('_')
            # Expected format: "shared_adapterid_conversationid" (at least 3 parts)
            if len(parts) >= 3 and parts[0] == "shared":
                parsed_adapter_id = parts[1]
                parsed_external_conv_id = "_".join(parts[2:]) # Join remaining parts for conversation ID
                logger.info(f"Parsed adapter_id='{parsed_adapter_id}' and external_conv_id='{parsed_external_conv_id}' from identifier '{identifier}'.")
            else:
                logger.warning(f"Could not parse adapter_id and external_conversation_id from identifier '{identifier}'. They will be None for the new Space.")

        try:
            new_shared_space = Space(
                element_id=identifier, 
                name=space_name, 
                description=description,
                adapter_id=parsed_adapter_id,                 # PASS PARSED/PROVIDED
                external_conversation_id=parsed_external_conv_id # PASS PARSED/PROVIDED
            )

            # metadata handling could be added here if Space constructor or a setter uses it
            # if metadata and hasattr(new_shared_space, 'set_metadata'):
            #    new_shared_space.set_metadata(metadata)


            if self.register_space(new_shared_space):
                logger.info(f"Successfully created and registered SharedSpace: {space_name} ({identifier})")
                # NEW: Create and mount the chat interface element using ElementFactoryComponent
                self._create_chat_interface_in_shared_space_with_factory(new_shared_space)
                return new_shared_space
            else:
                # register_space should log its own errors
                logger.error(f"Failed to register newly created SharedSpace: {space_name} ({identifier})")
                return None
        except Exception as e:
            logger.error(f"Exception during SharedSpace creation for identifier '{identifier}': {e}", exc_info=True)
            return None

    def _create_chat_interface_in_shared_space_with_factory(self, shared_space: "Space") -> None:
        """
        Uses an ElementFactoryComponent on the SharedSpace to create and mount a chat interface.
        Assumes SharedSpace now has ElementFactoryComponent by default and it's configured
        with the necessary outgoing_action_callback (which should be self.response_callback).
        """
        from .elements.components.factory_component import ElementFactoryComponent
        if not shared_space or not shared_space.id or not shared_space.adapter_id or not shared_space.external_conversation_id:
            logger.error(f"Cannot create chat interface: SharedSpace is None or missing critical attributes (id, adapter_id, external_conversation_id). Space: {shared_space}")
            return

        chat_interface_id = f"{shared_space.id}_chat_interface"
        if shared_space.get_mounted_element(chat_interface_id):
            logger.debug(f"Chat interface '{chat_interface_id}' already exists in {shared_space.id}. Skipping creation.")
            return

        logger.info(f"Creating chat interface for SharedSpace '{shared_space.id}' using its ElementFactoryComponent.")

        # 1. Get the ElementFactoryComponent (should exist by default on SharedSpace now)
        factory_component = shared_space.get_component_by_type(ElementFactoryComponent)
        if not factory_component:
            # This is now unexpected if Space.__init__ guarantees it.
            logger.error(f"CRITICAL: ElementFactoryComponent not found on SharedSpace '{shared_space.id}'. Cannot create chat interface.")
            return
        
        # Ensure the factory has the correct outgoing_action_callback (SpaceRegistry's response_callback)
        # Space.__init__ should have passed it if provided at SharedSpace creation. 
        # If not, or if we need to be certain for SharedSpaces created by SpaceRegistry itself:
        if hasattr(self, 'response_callback') and self.response_callback:
            if not (hasattr(factory_component, '_outgoing_action_callback_for_created') and \
                    factory_component._outgoing_action_callback_for_created == self.response_callback):
                if hasattr(factory_component, 'set_outgoing_action_callback') and callable(getattr(factory_component, 'set_outgoing_action_callback')):
                    factory_component.set_outgoing_action_callback(self.response_callback)
                    logger.info(f"Ensured/Set SpaceRegistry.response_callback on ElementFactoryComponent for SharedSpace '{shared_space.id}'.")
                else:
                    logger.warning(f"ElementFactoryComponent on SharedSpace '{shared_space.id}' does not have set_outgoing_action_callback. Cannot ensure it has registry's callback.")
        else:
            logger.warning(f"SpaceRegistry does not have response_callback. Chat interface in SharedSpace '{shared_space.id}' may not send messages.")

        # 2. Use the factory to create the chat interface from prefab
        element_config_for_chat_interface = {
            "name": f"Chat Interface for {shared_space.name}",
            "description": f"Handles messages and interactions for shared space {shared_space.name}",
            "adapter_id": shared_space.adapter_id, 
            "external_conversation_id": shared_space.external_conversation_id
        }

        creation_result = factory_component.handle_create_element_from_prefab(
            prefab_name="standard_chat_interface",
            element_id=chat_interface_id,
            element_config=element_config_for_chat_interface,
            mount_id_override=chat_interface_id
        )

        if creation_result and creation_result.get("success"):
            new_chat_element = creation_result.get("element")
            if new_chat_element:
                logger.info(f"Successfully created and mounted chat interface '{new_chat_element.id}' in SharedSpace '{shared_space.id}' using factory.")
                # The ElementFactoryComponent should have injected the outgoing_action_callback
                # into the MessageActionHandler of the new_chat_element if the prefab specifies it
                # and the factory has the callback stored. No need for manual injection here anymore.
            else:
                logger.error(f"Chat interface creation reported success but no element returned for SharedSpace '{shared_space.id}'. Result: {creation_result}")
        else:
            error_msg = creation_result.get("error", "Unknown error from ElementFactoryComponent") if creation_result else "No result from ElementFactoryComponent"
            logger.error(f"Failed to create chat interface in SharedSpace '{shared_space.id}' using factory: {error_msg}")