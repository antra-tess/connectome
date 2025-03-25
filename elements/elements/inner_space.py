"""
Inner Space
Special space element that represents the agent's subjective experience.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Type
import uuid
import time
import importlib
import inspect

from .space import Space
from .base import BaseElement, MountType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InnerSpace(Space):
    """
    Special space element that represents the agent's subjective experience.
    
    The InnerSpace is a privileged space that:
    1. Contains the agent's subjective timeline
    2. Serves as the agent's primary interface to the world
    3. Manages uplinks to external spaces
    4. Provides tools for the agent to interact with itself and external spaces
    
    Each Shell has exactly one InnerSpace that cannot be replaced or destroyed.
    """
    
    # Event types that can be processed by the inner space
    EVENT_TYPES = [
        "user_message",
        "system_message",
        "tool_response",
        "agent_message",
        "state_update"
    ]
    
    def __init__(self, element_id: str, name: str, registry):
        """
        Initialize the inner space.
        
        Args:
            element_id: Unique identifier for this inner space
            name: Human-readable name for this inner space
            registry: SpaceRegistry instance
        """
        super().__init__(element_id, name, "Agent's Inner Space")
        
        # Register with registry
        self._registry = registry
        registry.register_inner_space(self)
        
        # Element registry for creating new elements
        self._element_registry: Dict[str, Type[BaseElement]] = {}
        
        # Register standard element types
        self._register_standard_element_types()
        
        # Track elements requesting attention across all spaces
        self._attention_requests: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Created inner space: {name} ({element_id})")
    
    def get_space(self, space_id: str) -> Optional[Space]:
        """
        Get a space by ID.
        
        Args:
            space_id: ID of the space to retrieve
            
        Returns:
            The space, or None if not found
        """
        # First check spaces we've already loaded
        if space_id in self.spaces:
            return self.spaces[space_id]
            
        # If not found, try to get from registry
        if self._registry:
            space = self._registry.get_space(space_id)
            if space:
                self.spaces[space_id] = space
                return space
                
        return None
    
    def execute_element_action(self, 
                               space_id: Optional[str], 
                               element_id: str, 
                               action_name: str, 
                               parameters: Dict[str, Any],
                               timeline_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action on an element.
        
        Args:
            space_id: ID of the space containing the element (if None, use this space)
            element_id: ID of the element to act on
            action_name: Name of the action to execute
            parameters: Parameters for the action
            timeline_context: Timeline context for this action
            
        Returns:
            Result of the action execution
        """
        if not space_id or space_id == self.id:
            # Action on an element in this space
            return self.execute_action_on_element(element_id, action_name, parameters)
            
        # Action on an element in another space
        target_space = self.get_space(space_id)
        if not target_space:
            return {"error": f"Space not found: {space_id}"}
            
        return target_space.execute_action_on_element(element_id, action_name, parameters)
    
    def _register_inner_space_tools(self) -> None:
        """Register tools specific to inner space."""
        
        @self.register_tool(
            name="mount_element",
            description="Mount a new element in this inner space",
            parameter_descriptions={
                "element_type": "Type of element to mount",
                "element_id": "ID to assign to the new element (optional)",
                "name": "Name for the new element",
                "description": "Description of the element's purpose",
                "mount_id": "Identifier for the mount point (optional)",
                "mount_type": "Type of mounting (inclusion or uplink)",
                "parameters": "Additional parameters for element initialization"
            }
        )
        def mount_element(element_type: str, name: str, description: str, 
                         element_id: Optional[str] = None, mount_id: Optional[str] = None,
                         mount_type: str = "inclusion", 
                         parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """
            Tool to mount a new element in this inner space.
            
            Args:
                element_type: Type of element to mount
                name: Name for the new element
                description: Description of the element's purpose
                element_id: ID to assign to the new element (optional)
                mount_id: Identifier for the mount point (optional)
                mount_type: Type of mounting (inclusion or uplink)
                parameters: Additional parameters for element initialization
                
            Returns:
                Result of the mounting operation
            """
            # Validate mount type
            try:
                mt = MountType.INCLUSION if mount_type == "inclusion" else MountType.UPLINK
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid mount type: {mount_type}"
                }
                
            # Generate element ID if not provided
            if not element_id:
                element_id = f"{element_type.lower()}_{str(uuid.uuid4())[:8]}"
                
            # Check if an element with this ID already exists
            for mount_info in self._mounted_elements.values():
                existing_element = mount_info["element"]
                if existing_element.id == element_id:
                    return {
                        "success": False,
                        "error": f"Element with ID {element_id} already exists"
                    }
            
            # Get element class
            element_class = self._get_element_class(element_type)
            if not element_class:
                return {
                    "success": False,
                    "error": f"Element type {element_type} not found"
                }
                
            # Create the element
            try:
                # Initialize parameters if not provided
                params = parameters or {}
                
                # Create the element
                element = element_class(element_id=element_id, name=name, 
                                       description=description, **params)
                                       
                # Mount the element
                mount_result = self.mount_element(element, mount_id, mt)
                
                if not mount_result:
                    return {
                        "success": False,
                        "error": f"Failed to mount element {element_id}"
                    }
                    
                return {
                    "success": True,
                    "element_id": element_id,
                    "mount_id": mount_id or element_id,
                    "element_type": element_type,
                    "mount_type": mount_type
                }
            except Exception as e:
                logger.error(f"Error creating element of type {element_type}: {e}")
                return {
                    "success": False,
                    "error": f"Error creating element: {str(e)}"
                }
        
        @self.register_tool(
            name="unmount_element",
            description="Unmount an element from this inner space",
            parameter_descriptions={
                "mount_id": "Identifier for the mount point"
            }
        )
        def unmount_element_tool(mount_id: str) -> Dict[str, Any]:
            """
            Tool to unmount an element from this inner space.
            
            Args:
                mount_id: Identifier for the mount point
                
            Returns:
                Result of the unmounting operation
            """
            # Get element info before unmounting
            element_info = {}
            if mount_id in self._mounted_elements:
                element = self._mounted_elements[mount_id]["element"]
                element_info = {
                    "element_id": element.id,
                    "element_name": element.name,
                    "element_type": element.__class__.__name__
                }
            
            # Unmount the element
            success = self.unmount_element(mount_id)
            
            return {
                "success": success,
                "mount_id": mount_id,
                **element_info
            }
        
        @self.register_tool(
            name="register_element_type",
            description="Register a new element type",
            parameter_descriptions={
                "element_type": "Name of the element type",
                "element_class": "Fully qualified class name of the element"
            }
        )
        def register_element_type_tool(element_type: str, element_class: str) -> Dict[str, Any]:
            """
            Tool to register a new element type.
            
            Args:
                element_type: Name of the element type
                element_class: Fully qualified class name of the element
                
            Returns:
                Result of the registration operation
            """
            try:
                # Import the module and get the class
                module_path, class_name = element_class.rsplit('.', 1)
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                
                # Check if it's a BaseElement subclass
                if not issubclass(cls, BaseElement):
                    return {
                        "success": False,
                        "error": f"{element_class} is not a subclass of BaseElement"
                    }
                    
                # Register the element type
                self.register_element_type(element_type, cls)
                
                return {
                    "success": True,
                    "element_type": element_type,
                    "element_class": element_class
                }
            except Exception as e:
                logger.error(f"Error registering element type {element_type}: {e}")
                return {
                    "success": False,
                    "error": f"Error registering element type: {str(e)}"
                }
        
        @self.register_tool(
            name="get_registered_elements",
            description="Get all registered element types",
            parameter_descriptions={}
        )
        def get_registered_elements_tool() -> Dict[str, Any]:
            """
            Tool to get all registered element types.
            
            Returns:
                Dictionary of registered element types
            """
            return {
                "success": True,
                "element_types": self.get_registered_elements()
            }
    
    def register_element_type(self, element_type: str, element_class: Type[BaseElement]) -> bool:
        """
        Register a new element type.
        
        Args:
            element_type: Name of the element type
            element_class: Class of the element
            
        Returns:
            True if registration was successful, False otherwise
        """
        # Validate the element class
        if not inspect.isclass(element_class) or not issubclass(element_class, BaseElement):
            logger.error(f"Cannot register {element_class.__name__} as it is not a subclass of BaseElement")
            return False
            
        # Register the element type
        self._element_registry[element_type] = element_class
        logger.info(f"Registered element type {element_type}")
        
        return True
    
    def get_registered_elements(self) -> Dict[str, str]:
        """
        Get all registered element types.
        
        Returns:
            Dictionary mapping element type names to class names
        """
        return {
            element_type: element_class.__name__ 
            for element_type, element_class in self._element_registry.items()
        }
    
    def _get_element_class(self, element_type: str) -> Optional[Type[BaseElement]]:
        """
        Get the class for an element type.
        
        Args:
            element_type: Name of the element type
            
        Returns:
            Element class if found, None otherwise
        """
        # Check if the element type is registered
        if element_type in self._element_registry:
            return self._element_registry[element_type]
            
        # Try to import the element type
        try:
            # First try as a module path
            if '.' in element_type:
                module_path, class_name = element_type.rsplit('.', 1)
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                
                # Check if it's a BaseElement subclass
                if issubclass(cls, BaseElement):
                    # Register for future use
                    self._element_registry[element_type] = cls
                    return cls
                    
            # Try as a class in the elements module
            module = importlib.import_module('elements.elements')
            if hasattr(module, element_type):
                cls = getattr(module, element_type)
                
                # Check if it's a BaseElement subclass
                if issubclass(cls, BaseElement):
                    # Register for future use
                    self._element_registry[element_type] = cls
                    return cls
        except (ImportError, AttributeError) as e:
            logger.error(f"Error importing element type {element_type}: {e}")
            
        logger.warning(f"Element type {element_type} not found")
        return None
    
    def _record_timeline_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Record an event in the timeline.
        
        Args:
            event_type: Type of the event
            event_data: Data for the event
        """
        # Create the event data
        full_event_data = {
            "event_type": event_type,
            "timestamp": int(time.time() * 1000),
            **event_data
        }
        
        # Use primary timeline if available, otherwise create a new one
        timeline_id = self._timeline_state["primary_timeline"]
        if timeline_id is None:
            timeline_id = str(uuid.uuid4())
            self._timeline_state["primary_timeline"] = timeline_id
            self._timeline_state["active_timelines"].add(timeline_id)
            self._timeline_state["events"][timeline_id] = []
            self._timeline_state["timeline_metadata"][timeline_id] = {
                "created_at": int(time.time() * 1000),
                "last_updated": int(time.time() * 1000)
            }
            
        # Update the state
        self.update_state(full_event_data, {"timeline_id": timeline_id})
    
    def _get_element_by_id(self, element_id: str) -> Optional[BaseElement]:
        """
        Get an element by ID.
        
        This method is used by the mount_element tool to find elements by ID.
        If needed, it can be extended to look up elements in a registry
        outside this space.
        
        Args:
            element_id: ID of the element to get
            
        Returns:
            The element if found, None otherwise
        """
        # Check mounted elements
        for mount_info in self._mounted_elements.values():
            element = mount_info["element"]
            if element.id == element_id:
                return element
                
        return None
    
    def receive_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """
        Receive an event from the Activity Layer.
        
        Args:
            event: The event to receive
            timeline_context: The timeline context for this event
        """
        logger.info(f"Receiving event in timeline {timeline_context.get('timeline_id')}: {event.get('type')}")
        
        # Store the event in the timeline DAG
        self.add_event_to_timeline(event, timeline_context)
        
        # Process the event
        try:
            event_type = event.get("type")
            if event_type in self.EVENT_TYPES:
                self._process_event(event, timeline_context)
            
            # Route to appropriate element if specified
            if "targetElement" in event:
                target_element_id = event["targetElement"]
                
                if target_element_id in self.mounted_elements:
                    element = self.mounted_elements[target_element_id]
                    element.receive_event(event, timeline_context)
                else:
                    logger.warning(f"Event targeted non-existent element: {target_element_id}")
        except Exception as e:
            logger.error(f"Error processing event: {e}", exc_info=True)
    
    def add_event_to_timeline(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> str:
        """
        Add an event to the timeline DAG.
        
        Args:
            event: The event to add
            timeline_context: The timeline context for this event
            
        Returns:
            ID of the newly created event
        """
        logger.info(f"Adding event to timeline {timeline_context.get('timeline_id')}: {event.get('type')}")
        
        # Create event object with metadata
        event_id = event.get("id", str(uuid.uuid4()))
        event["id"] = event_id
        event["timestamp"] = event.get("timestamp", int(time.time() * 1000))
        
        # Get branch ID from context
        branch_id = timeline_context.get("timeline_id")
        if not branch_id:
            logger.warning("No branch ID in timeline context, using primary branch")
            branch_id = "primary-branch-001"
        
        # Link to parent event if available
        parent_event_id = timeline_context.get("last_event_id")
        if parent_event_id:
            event["parent_id"] = parent_event_id
        
        # Store in timeline state
        if branch_id not in self._timeline_state["events"]:
            self._timeline_state["events"][branch_id] = {}
            
        self._timeline_state["events"][branch_id][event_id] = event
        
        # Update timeline context with this event as the new last event
        timeline_context["last_event_id"] = event["id"]
        
        # Notify any observers
        self._notify_observers("timeline_updated", {
            "element_id": self.id,
            "event_id": event_id,
            "event_type": event.get("type"),
            "timeline_id": branch_id
        })
        
        return event["id"]
    
    def send_message(self, message: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """
        Send a message from the agent to external systems.
        
        This passes through the Inner Space, to the Activity Layer, and then to external adapters.
        
        Args:
            message: The message to send
            timeline_context: The timeline context for this message
        """
        logger.info(f"Sending message in timeline {timeline_context.get('timeline_id')}: {message.get('type')}")
        
        # Add the message to the timeline as an event
        message_id = self.add_event_to_timeline(message, timeline_context)
        
        # Only propagate externally if in primary timeline
        if timeline_context.get("is_primary", False):
            # In a real implementation, this would call Activity Layer to propagate externally
            logger.info(f"Propagating message externally: {message_id}")
            
            # If we have an associated registry, propagate through it
            if self._registry:
                self._registry.propagate_message(message, timeline_context)
        else:
            logger.info(f"Message not propagated externally (non-primary timeline): {message_id}")
    
    def _process_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """
        Process an event based on its type.
        
        Args:
            event: The event to process
            timeline_context: The timeline context for this event
        """
        event_type = event.get("type")
        
        if event_type == "user_message":
            # Process user message
            self._handle_user_message(event, timeline_context)
        elif event_type == "system_message":
            # Process system message
            self._handle_system_message(event, timeline_context)
        elif event_type == "tool_response":
            # Process tool response
            self._handle_tool_response(event, timeline_context)
        elif event_type == "agent_message":
            # Process agent message
            self._handle_agent_message(event, timeline_context)
        elif event_type == "state_update":
            # Process state update
            self._handle_state_update(event, timeline_context)
        else:
            logger.warning(f"Unknown event type: {event_type}")
            
    def _handle_user_message(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """Handle a user message event"""
        # In a complete implementation, this would process the user message
        # and potentially update the state of the inner space
        logger.info(f"Handling user message: {event.get('content', '')[:30]}...")
        
    def _handle_system_message(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """Handle a system message event"""
        # Process system message
        logger.info(f"Handling system message: {event.get('content', '')[:30]}...")
        
    def _handle_tool_response(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """Handle a tool response event"""
        # Process tool response
        logger.info(f"Handling tool response: {event.get('toolName', '')} - {event.get('status', '')}")
        
    def _handle_agent_message(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """Handle an agent message event"""
        # Process agent message
        logger.info(f"Handling agent message: {event.get('content', '')[:30]}...")
        
    def _handle_state_update(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """Handle a state update event"""
        # Process state update
        logger.info(f"Handling state update: {event.get('stateType', '')}")
    
    def mount_element(self, element: BaseElement, mount_type: MountType = MountType.CHILD) -> None:
        """
        Mount an element to this space.
        
        Args:
            element: The element to mount
            mount_type: The type of mount
        """
        logger.info(f"Mounting element {element.element_id} to inner space with mount type {mount_type}")
        
        # Store the element
        self.mounted_elements[element.element_id] = element
        
        # Update the element's parent reference
        element.parent = self 

    def _on_element_state_changed(self, element_id: str, state_data: Dict[str, Any]) -> None:
        """
        Handle element state change notification.
        
        Args:
            element_id: ID of the element that changed
            state_data: Data about the state change
        """
        logger.debug(f"Element state changed: {element_id}")
        
        # Check for attention request
        event_type = state_data.get("type")
        if event_type == "attention_requested":
            self._handle_attention_request(element_id, state_data)
        elif event_type == "attention_cleared":
            self._handle_attention_cleared(element_id)
    
    def _handle_attention_request(self, element_id: str, request_data: Dict[str, Any]) -> None:
        """
        Handle an attention request from a Space or Element.
        
        Args:
            element_id: ID of the Space or Element requesting attention
            request_data: Data about the attention request
        """
        # Add or update the attention request
        self._attention_requests[element_id] = {
            "timestamp": request_data.get("timestamp", int(time.time() * 1000)),
            "data": request_data,
            "source_element_id": request_data.get("source_element_id", element_id)
        }
        
        logger.info(f"Element/Space {element_id} attention request registered in InnerSpace")
        
        # Notify registry that a component needs attention
        if self._registry:
            self._registry._notify_observers("inner_space_attention_requested", {
                "inner_space_id": self.id,
                "element_id": element_id,
                "source_element_id": request_data.get("source_element_id", element_id),
                "request_data": request_data
            })
    
    def _handle_attention_cleared(self, element_id: str) -> None:
        """
        Handle attention cleared notification from a Space or Element.
        
        Args:
            element_id: ID of the Space or Element clearing attention
        """
        # Remove the attention request if it exists
        if element_id in self._attention_requests:
            source_element_id = self._attention_requests[element_id].get("source_element_id", element_id)
            del self._attention_requests[element_id]
            
            # Notify registry that this element no longer needs attention
            if self._registry:
                self._registry._notify_observers("inner_space_attention_cleared", {
                    "inner_space_id": self.id,
                    "element_id": element_id,
                    "source_element_id": source_element_id
                })
                
            logger.info(f"Element/Space {element_id} attention cleared in InnerSpace")
    
    def get_elements_requesting_attention(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all elements currently requesting attention in the InnerSpace.
        
        Returns:
            Dictionary mapping element IDs to their attention request data
        """
        return self._attention_requests.copy()
    
    def handle_element_observer_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle events from observed elements.
        
        Args:
            event_data: Data about the event
        """
        element_id = event_data.get("element_id")
        if not element_id:
            logger.warning("Received element event without element_id")
            return
            
        event_type = event_data.get("type")
        
        # Handle attention related events
        if event_type == "attention_requested":
            # This is coming from a Space that has already evaluated the need
            self._handle_attention_request(element_id, event_data)
        elif event_type == "attention_cleared":
            self._handle_attention_cleared(element_id)
        elif event_type == "element_state_changed":
            if event_data.get("state_change") == "attention_needed":
                # Handle direct attention requests from elements mounted in InnerSpace
                # For elements in sub-spaces, the Space should handle this
                if element_id in self._mounted_elements:
                    self._handle_attention_request(element_id, event_data)
            else:
                logger.debug(f"Element state changed: {element_id} - {event_data.get('state_change')}")
        else:
            logger.debug(f"Received element event: {event_type} from {element_id}") 