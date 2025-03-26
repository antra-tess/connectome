"""
Base Element
Base class for all elements in the Bot Framework.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Set, Tuple, Literal
import uuid
import time
import enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MountType(enum.Enum):
    """Type of element mounting"""
    INCLUSION = "inclusion"  # Lifecycle managed by containing element
    UPLINK = "uplink"        # Connection without lifecycle management


class ElementState(enum.Enum):
    """State of an element"""
    OPEN = "open"    # Element is fully mounted and active
    CLOSED = "closed"  # Element is mounted but in reduced functionality state
    HIDDEN = "hidden"  # Element is hidden (not visible)


class BaseElement:
    """
    Base class for all elements in the Bot Framework.
    
    Provides core functionality and interfaces that all elements must implement.
    
    Elements have two states:
    - Open: The element is fully active and its interior is accessible
    - Closed: The element is in a reduced state with only exterior accessible
    
    Elements also have two views:
    - Interior: The full content and functionality (required)
    - Exterior: A compact representation for closed state (optional)
    
    Elements can be mounted inside other elements in two ways:
    - Inclusion: The element's lifecycle is managed by the containing element
    - Uplink: The element maintains its own lifecycle but is connected
    """
    
    # Define which event types this element can handle
    EVENT_TYPES: List[str] = []
    
    # Whether this element can be opened/closed
    SUPPORTS_OPEN_CLOSE: bool = True
    
    # Whether this element has an exterior representation
    HAS_EXTERIOR: bool = False
    
    # Whether this element is a Space (contains other elements) or an Object
    IS_SPACE: bool = False
    
    def __init__(self, element_id: str, name: str, description: str):
        """
        Initialize the base element.
        
        Args:
            element_id: Unique identifier for this element
            name: Human-readable name for this element
            description: Description of this element's purpose
        """
        self.id = element_id
        self.name = name
        self.description = description
        
        # Element state (open/closed)
        self._state = ElementState.CLOSED
        
        # Mounted elements (if this is a Space)
        self._mounted_elements: Dict[str, Dict[str, Any]] = {}
        
        # Parent element (where this element is mounted)
        self._parent_element: Optional[Tuple[str, MountType]] = None
        
        # Initialize timeline management
        self._timeline_state = {
            "events": {},  # timeline_id -> list of events
            "active_timelines": set(),  # Set of active timeline IDs
            "timeline_metadata": {},  # timeline_id -> metadata
            "timeline_relationships": {},  # timeline_id -> {parent_id, is_primary, fork_point}
            "primary_timeline": None,  # ID of the primary timeline
            "element_states": {},  # timeline_id -> {event_id: event_data}
            "entangled_timelines": set()  # Set of entangled timeline IDs
        }
        
        # Attention tracking
        self._attention_requested = False
        self._attention_request_data = None
        self._last_attention_request_time = None
        
        # Initialize observer callbacks
        self._observers: List[Callable] = []
        
        # Initialize tools
        self._tools: Dict[str, Callable] = {}
        self._tool_descriptions: Dict[str, Dict[str, Any]] = {}
        
        # Registry reference (for sending messages to activity layer)
        self._registry = None
        
        # Element delegate for rendering
        self._delegate = None
        
        logger.info(f"Created element: {name} ({element_id})")
    
    def set_registry(self, registry) -> None:
        """
        Set the SpaceRegistry reference.
        
        This allows elements to send messages to the activity layer
        through the SpaceRegistry.
        
        Args:
            registry: SpaceRegistry instance
        """
        self._registry = registry
        logger.debug(f"Set registry reference for element {self.id}")
    
    def get_registry(self):
        """
        Get the SpaceRegistry reference.
        
        Returns:
            The SpaceRegistry instance if set, None otherwise
        """
        return self._registry
    
    def open(self) -> bool:
        """
        Open the element.
        
        This makes the element fully active and provides access to its interior.
        
        Returns:
            True if the operation was successful, False otherwise
        """
        if not self.SUPPORTS_OPEN_CLOSE:
            logger.warning(f"Element {self.id} does not support open/close operations")
            return False
            
        if self._state == ElementState.OPEN:
            logger.debug(f"Element {self.id} is already open")
            return True
            
        prev_state = self._state
        self._state = ElementState.OPEN
        
        # Create a state change event
        event_data = {
            "event_type": "element_opened",
            "prev_state": prev_state.value,
            "new_state": self._state.value,
            "timestamp": int(time.time() * 1000)
        }
        
        # Get primary timeline if available
        timeline_id = self._timeline_state.get("primary_timeline")
        if timeline_id:
            # Update state in timeline
            self.update_state(event_data, {"timeline_id": timeline_id})
        
        # Notify any observers
        self.notify_observers({
            "type": "element_state_changed",
            "element_id": self.id,
            "state_change": "opened",
            "prev_state": prev_state.value,
            "new_state": self._state.value,
            "timestamp": int(time.time() * 1000)
        })
        
        # Open any included elements
        if self.IS_SPACE:
            for mount_id, mount_info in self._mounted_elements.items():
                if mount_info.get("mount_type") == MountType.INCLUSION:
                    element = mount_info.get("element")
                    if element and element.SUPPORTS_OPEN_CLOSE:
                        element.open()
        
        # Execute custom open logic
        self._on_open()
        
        logger.info(f"Opened element {self.id}")
        return True
    
    def close(self) -> bool:
        """
        Close the element.
        
        This reduces the element's functionality and only exposes its exterior.
        
        Returns:
            True if the operation was successful, False otherwise
        """
        if not self.SUPPORTS_OPEN_CLOSE:
            logger.warning(f"Element {self.id} does not support open/close operations")
            return False
            
        if self._state == ElementState.CLOSED:
            logger.debug(f"Element {self.id} is already closed")
            return True
            
        prev_state = self._state
        self._state = ElementState.CLOSED
        
        # Create a state change event
        event_data = {
            "event_type": "element_closed",
            "prev_state": prev_state.value,
            "new_state": self._state.value,
            "timestamp": int(time.time() * 1000)
        }
        
        # Get primary timeline if available
        timeline_id = self._timeline_state.get("primary_timeline")
        if timeline_id:
            # Update state in timeline
            self.update_state(event_data, {"timeline_id": timeline_id})
        
        # Notify any observers
        self.notify_observers({
            "type": "element_state_changed",
            "element_id": self.id,
            "state_change": "closed",
            "prev_state": prev_state.value,
            "new_state": self._state.value,
            "timestamp": int(time.time() * 1000)
        })
        
        # Close any included elements
        if self.IS_SPACE:
            for mount_id, mount_info in self._mounted_elements.items():
                if mount_info.get("mount_type") == MountType.INCLUSION:
                    element = mount_info.get("element")
                    if element and element.SUPPORTS_OPEN_CLOSE:
                        element.close()
        
        # Execute custom close logic
        self._on_close()
        
        logger.info(f"Closed element {self.id}")
        return True
    
    def hide(self) -> bool:
        """
        Hide the element.
        
        This makes the element invisible but still mounted.
        
        Returns:
            True if the operation was successful, False otherwise
        """
        if self._state == ElementState.HIDDEN:
            logger.debug(f"Element {self.id} is already hidden")
            return True
            
        prev_state = self._state
        self._state = ElementState.HIDDEN
        
        # Create a state change event
        event_data = {
            "event_type": "element_hidden",
            "prev_state": prev_state.value,
            "new_state": self._state.value,
            "timestamp": int(time.time() * 1000)
        }
        
        # Get primary timeline if available
        timeline_id = self._timeline_state.get("primary_timeline")
        if timeline_id:
            # Update state in timeline
            self.update_state(event_data, {"timeline_id": timeline_id})
        
        # Notify any observers
        self.notify_observers({
            "type": "element_state_changed",
            "element_id": self.id,
            "state_change": "hidden",
            "prev_state": prev_state.value,
            "new_state": self._state.value,
            "timestamp": int(time.time() * 1000)
        })
        
        logger.info(f"Hidden element {self.id}")
        return True
    
    def show(self) -> bool:
        """
        Show the element (make it visible again).
        
        This reverses the hide operation.
        
        Returns:
            True if the operation was successful, False otherwise
        """
        if self._state != ElementState.HIDDEN:
            logger.debug(f"Element {self.id} is not hidden")
            return True
            
        # Restore previous state (default to closed)
        prev_state = self._state
        self._state = ElementState.CLOSED
        
        # Create a state change event
        event_data = {
            "event_type": "element_shown",
            "prev_state": prev_state.value,
            "new_state": self._state.value,
            "timestamp": int(time.time() * 1000)
        }
        
        # Get primary timeline if available
        timeline_id = self._timeline_state.get("primary_timeline")
        if timeline_id:
            # Update state in timeline
            self.update_state(event_data, {"timeline_id": timeline_id})
        
        # Notify any observers
        self.notify_observers({
            "type": "element_state_changed",
            "element_id": self.id,
            "state_change": "shown",
            "prev_state": prev_state.value,
            "new_state": self._state.value,
            "timestamp": int(time.time() * 1000)
        })
        
        logger.info(f"Showed element {self.id}")
        return True
    
    def _on_open(self) -> None:
        """
        Custom logic to execute when element is opened.
        
        Override in subclasses to provide specific open behavior.
        """
        pass
        
    def _on_close(self) -> None:
        """
        Custom logic to execute when element is closed.
        
        Override in subclasses to provide specific close behavior.
        """
        pass
    
    def is_open(self) -> bool:
        """
        Check if the element is open.
        
        Returns:
            True if the element is open, False if closed
        """
        return self._state == ElementState.OPEN
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the element.
        
        Returns:
            Dictionary representation of the current state
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "state": self._state.value,
            "timeline_state": self._timeline_state,
            "is_space": self.IS_SPACE,
            "supports_open_close": self.SUPPORTS_OPEN_CLOSE,
            "has_exterior": self.HAS_EXTERIOR
        }
    
    def get_interior_state(self) -> Dict[str, Any]:
        """
        Get the interior state of the element.
        
        This provides the full representation of the element when open.
        Must be implemented by subclasses.
        
        Returns:
            Dictionary representation of the interior state
        """
        # Base implementation provides minimal info
        # Subclasses should override with full interior state
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "event_types": self.EVENT_TYPES,
            "mounted_elements": self._get_mounted_elements_info() if self.IS_SPACE else None
        }
    
    def get_exterior_state(self) -> Optional[Dict[str, Any]]:
        """
        Get the exterior state of the element.
        
        This provides a compact representation of the element when closed.
        May be None if the element has no exterior.
        
        Returns:
            Dictionary representation of the exterior state, or None
        """
        # Base implementation provides minimal info if HAS_EXTERIOR is True
        # Subclasses should override with specific exterior state
        if not self.HAS_EXTERIOR:
            return None
            
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description
        }
    
    def mount_element(self, element: 'BaseElement', mount_id: str = None, 
                      mount_type: MountType = MountType.INCLUSION) -> bool:
        """
        Mount an element inside this element (if this is a Space).
        
        Args:
            element: Element to mount
            mount_id: Optional identifier for the mount point, defaults to element.id
            mount_type: Type of mounting (inclusion or uplink)
            
        Returns:
            True if mounting was successful, False otherwise
        """
        if not self.IS_SPACE:
            logger.error(f"Cannot mount elements in {self.id} as it is not a Space")
            return False
            
        # Use element ID as mount ID if not provided
        mount_id = mount_id or element.id
        
        # Check if mount point is already in use
        if mount_id in self._mounted_elements:
            logger.warning(f"Mount point {mount_id} is already in use in space {self.id}")
            return False
            
        # Check for circular mounting
        if self._would_create_circular_mount(element):
            logger.error(f"Cannot mount element {element.id} in {self.id} as it would create a circular reference")
            return False
            
        # Store the element
        self._mounted_elements[mount_id] = {
            "element": element,
            "mount_type": mount_type,
            "mounted_at": int(time.time() * 1000)
        }
        
        # Store parent reference in the element
        element._set_parent(self.id, mount_type)
        
        # Record the mounting event in the timeline
        self._record_mount_event(element, mount_id, mount_type)
        
        # If inclusion, element state follows space state
        if mount_type == MountType.INCLUSION:
            if self.is_open():
                element.open()
            else:
                element.close()
        
        logger.info(f"Mounted element {element.id} in space {self.id} at mount point {mount_id} as {mount_type.value}")
        
        # Notify observers
        self.notify_observers({
            "type": "element_mounted",
            "space_id": self.id,
            "element_id": element.id,
            "mount_id": mount_id,
            "mount_type": mount_type.value
        })
        
        return True
    
    def unmount_element(self, mount_id: str) -> bool:
        """
        Unmount an element from this element.
        
        Args:
            mount_id: Identifier for the mount point
            
        Returns:
            True if unmounting was successful, False otherwise
        """
        if not self.IS_SPACE:
            logger.error(f"Cannot unmount elements from {self.id} as it is not a Space")
            return False
            
        # Check if mount point exists
        if mount_id not in self._mounted_elements:
            logger.warning(f"Mount point {mount_id} does not exist in space {self.id}")
            return False
            
        # Get element and mount type
        element = self._mounted_elements[mount_id]["element"]
        mount_type = self._mounted_elements[mount_id]["mount_type"]
        
        # Clear parent reference in the element
        element._clear_parent()
        
        # Remove from mounted elements
        del self._mounted_elements[mount_id]
        
        # Record the unmounting event in the timeline
        self._record_unmount_event(element, mount_id, mount_type)
        
        logger.info(f"Unmounted element {element.id} from space {self.id} at mount point {mount_id}")
        
        # Notify observers
        self.notify_observers({
            "type": "element_unmounted",
            "space_id": self.id,
            "element_id": element.id,
            "mount_id": mount_id
        })
        
        return True
    
    def get_mounted_element(self, mount_id: str) -> Optional['BaseElement']:
        """
        Get an element mounted at a specific mount point.
        
        Args:
            mount_id: Identifier for the mount point
            
        Returns:
            The mounted element if found, None otherwise
        """
        if not self.IS_SPACE or mount_id not in self._mounted_elements:
            return None
            
        return self._mounted_elements[mount_id]["element"]
    
    def get_mounted_elements(self) -> Dict[str, 'BaseElement']:
        """
        Get all elements mounted in this space.
        
        Returns:
            Dictionary mapping mount IDs to mounted elements
        """
        if not self.IS_SPACE:
            return {}
            
        return {mount_id: info["element"] for mount_id, info in self._mounted_elements.items()}
    
    def _get_mounted_elements_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all mounted elements.
        
        Returns:
            Dictionary mapping mount IDs to element information
        """
        if not self.IS_SPACE:
            return {}
            
        result = {}
        for mount_id, info in self._mounted_elements.items():
            element = info["element"]
            result[mount_id] = {
                "element_id": element.id,
                "element_name": element.name,
                "element_type": element.__class__.__name__,
                "mount_type": info["mount_type"].value,
                "mounted_at": info["mounted_at"],
                "is_open": element.is_open(),
                "is_space": element.IS_SPACE
            }
        return result
    
    def _would_create_circular_mount(self, element: 'BaseElement') -> bool:
        """
        Check if mounting an element would create a circular reference.
        
        Args:
            element: Element to check
            
        Returns:
            True if mounting would create a circular reference, False otherwise
        """
        # Check if the element is this element
        if element.id == self.id:
            return True
            
        # Check if this element is already mounted in the element (directly or indirectly)
        current = element
        while current._parent_element is not None:
            parent_id, _ = current._parent_element
            if parent_id == self.id:
                return True
                
            # Get the parent element
            # This might be None if the parent was destroyed
            parent_space = None
            for space in getattr(current, "_spaces_registry", {}).values():
                if space.id == parent_id:
                    parent_space = space
                    break
                    
            if parent_space is None:
                break
                
            current = parent_space
            
        return False
    
    def _set_parent(self, parent_id: str, mount_type: MountType) -> None:
        """
        Set the parent element reference.
        
        Args:
            parent_id: ID of the parent element
            mount_type: Type of mounting (inclusion or uplink)
        """
        self._parent_element = (parent_id, mount_type)
    
    def _clear_parent(self) -> None:
        """Clear the parent element reference."""
        self._parent_element = None
    
    def get_parent_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the parent element.
        
        Returns:
            Dictionary with parent information if available, None otherwise
        """
        if self._parent_element is None:
            return None
            
        parent_id, mount_type = self._parent_element
        return {
            "parent_id": parent_id,
            "mount_type": mount_type.value
        }
    
    def _record_mount_event(self, element: 'BaseElement', mount_id: str, mount_type: MountType) -> None:
        """
        Record a mount event in the timeline.
        
        Args:
            element: Mounted element
            mount_id: Mount ID
            mount_type: Type of mounting
        """
        event_data = {
            "event_type": "element_mounted",
            "element_id": element.id,
            "element_name": element.name,
            "element_type": element.__class__.__name__,
            "mount_id": mount_id,
            "mount_type": mount_type.value,
            "timestamp": int(time.time() * 1000)
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
            
        # Add the event
        self.update_state(event_data, {"timeline_id": timeline_id})
    
    def _record_unmount_event(self, element: 'BaseElement', mount_id: str, mount_type: MountType) -> None:
        """
        Record an unmount event in the timeline.
        
        Args:
            element: Unmounted element
            mount_id: Mount ID
            mount_type: Type of mounting
        """
        event_data = {
            "event_type": "element_unmounted",
            "element_id": element.id,
            "element_name": element.name,
            "element_type": element.__class__.__name__,
            "mount_id": mount_id,
            "mount_type": mount_type.value,
            "timestamp": int(time.time() * 1000)
        }
        
        # Use primary timeline if available
        timeline_id = self._timeline_state["primary_timeline"]
        if timeline_id is not None:
            self.update_state(event_data, {"timeline_id": timeline_id})
    
    def register_tool(self, name: str, description: str, parameter_descriptions: Dict[str, str]) -> Callable:
        """
        Decorator to register a tool with the element.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            parameter_descriptions: Dictionary mapping parameter names to descriptions
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            self._tools[name] = func
            self._tool_descriptions[name] = {
                "description": description,
                "parameters": parameter_descriptions
            }
            return func
        return decorator
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered tools and their descriptions.
        
        Returns:
            Dictionary mapping tool names to their descriptions and parameters
        """
        return self._tool_descriptions
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """
        Execute a registered tool.
        
        Args:
            name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Result of tool execution
            
        Raises:
            KeyError: If tool is not registered
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return self._tools[name](**kwargs)
    
    def register_observer(self, callback: Callable) -> None:
        """
        Register an observer callback.
        
        Args:
            callback: Function to call when state changes
        """
        if callback not in self._observers:
            self._observers.append(callback)
            logger.debug(f"Registered observer for element {self.id}")
    
    def unregister_observer(self, callback: Callable) -> None:
        """
        Remove an observer callback.
        
        Args:
            callback: Function to remove
        """
        if callback in self._observers:
            self._observers.remove(callback)
            logger.debug(f"Unregistered observer for element {self.id}")
    
    def notify_observers(self, update_data: Dict[str, Any]) -> None:
        """
        Notify all observers of a state change.
        
        Args:
            update_data: Data about the state change
        """
        for observer in self._observers:
            try:
                observer(update_data)
            except Exception as e:
                logger.error(f"Error in observer callback: {e}")
    
    def update_state(self, update_data: Dict[str, Any], timeline_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the element state based on incoming event data.
        
        This method handles the core state updates and timeline recording logic.
        It should not be overridden by subclasses directly - instead override
        _update_state_internal for custom state update logic.
        
        Args:
            update_data: Event data to update with
            timeline_context: Optional timeline context for the update
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Validate timeline context
            validated_timeline_context = self._validate_timeline_context(timeline_context)
            
            # Check for timeline entanglement - verify if this element is entangled with events in this timeline
            if not self._can_update_in_timeline(validated_timeline_context):
                logger.warning(f"Element {self.id} cannot update state in timeline {validated_timeline_context.get('timeline_id')}: Timeline entanglement check failed")
                return False
            
            # Get timeline ID from context
            timeline_id = validated_timeline_context.get('timeline_id')
            if not timeline_id:
                timeline_id = update_data.get('timeline_id')
                
            if not timeline_id:
                logger.warning(f"No timeline ID provided for element {self.id} update")
                # Use primary timeline as fallback
                timeline_id = self._timeline_state.get("primary_timeline")
                if not timeline_id:
                    # Generate a new timeline ID as last resort
                    timeline_id = f"timeline_{str(uuid.uuid4())[:8]}"
                    logger.warning(f"Generated new timeline ID for element {self.id}: {timeline_id}")
            
            # Initialize timeline state if needed
            self._ensure_timeline_exists(timeline_id, validated_timeline_context)
            
            # Handle standard element state transitions from event
            event_type = update_data.get('event_type')
            
            # Process explicit state transition events
            if event_type == 'element_opened':
                if self._state != ElementState.OPEN and self.SUPPORTS_OPEN_CLOSE:
                    prev_state = self._state
                    self._state = ElementState.OPEN
                    logger.debug(f"Element {self.id} state changed to OPEN via event")
                    self._on_open()
                    
                    # Add state transition metadata to update_data if not present
                    if 'state_transition' not in update_data:
                        update_data['state_transition'] = {
                            'prev_state': prev_state.value,
                            'new_state': self._state.value
                        }
            
            elif event_type == 'element_closed':
                if self._state != ElementState.CLOSED and self.SUPPORTS_OPEN_CLOSE:
                    prev_state = self._state
                    self._state = ElementState.CLOSED
                    logger.debug(f"Element {self.id} state changed to CLOSED via event")
                    self._on_close()
                    
                    # Add state transition metadata to update_data if not present
                    if 'state_transition' not in update_data:
                        update_data['state_transition'] = {
                            'prev_state': prev_state.value,
                            'new_state': self._state.value
                        }
            
            elif event_type == 'element_hidden':
                if self._state != ElementState.HIDDEN:
                    prev_state = self._state
                    self._state = ElementState.HIDDEN
                    logger.debug(f"Element {self.id} state changed to HIDDEN via event")
                    
                    # Add state transition metadata to update_data if not present
                    if 'state_transition' not in update_data:
                        update_data['state_transition'] = {
                            'prev_state': prev_state.value,
                            'new_state': self._state.value
                        }
            
            elif event_type == 'element_shown' and self._state == ElementState.HIDDEN:
                prev_state = self._state
                self._state = ElementState.CLOSED  # Default to closed when shown
                logger.debug(f"Element {self.id} state changed to CLOSED via show event")
                
                # Add state transition metadata to update_data if not present
                if 'state_transition' not in update_data:
                    update_data['state_transition'] = {
                        'prev_state': prev_state.value,
                        'new_state': self._state.value
                    }
            
            # If we're closed and this is an external event that's not a state transition event,
            # then we should reject it unless it's marked as internal
            closed_state_events = ['element_opened', 'element_closed', 'element_hidden', 'element_shown']
            if (self._state == ElementState.CLOSED and 
                event_type not in closed_state_events and
                not update_data.get('is_internal', False)):
                # Check if the element explicitly allows this event type while closed
                if not hasattr(self, 'ALLOWED_CLOSED_EVENTS') or event_type not in self.ALLOWED_CLOSED_EVENTS:
                    logger.warning(f"Element {self.id} is closed and cannot process external event: {event_type}")
                    return False
            
            # Handle attention request/clear events
            if event_type == 'attention_requested' or update_data.get('state_change') == 'attention_needed':
                self._attention_requested = True
                self._attention_request_data = update_data
                self._last_attention_request_time = int(time.time() * 1000)
                logger.debug(f"Attention requested for element {self.id}")
            elif event_type == 'attention_cleared':
                self._attention_requested = False
                self._attention_request_data = None
                logger.debug(f"Attention cleared for element {self.id}")
            
            # Record in timeline
            if timeline_id and event_type:
                # Generate a unique event ID if not provided
                event_id = update_data.get('event_id') or f"evt_{str(uuid.uuid4())[:8]}"
                
                # Make sure we have the events structure
                if 'events' not in self._timeline_state:
                    self._timeline_state['events'] = {}
                if timeline_id not in self._timeline_state['events']:
                    self._timeline_state['events'][timeline_id] = []
                
                # Add the event ID to the timeline
                self._timeline_state['events'][timeline_id].append(event_id)
                
                # Store the complete event data
                if 'event_data' not in self._timeline_state:
                    self._timeline_state['event_data'] = {}
                if timeline_id not in self._timeline_state['event_data']:
                    self._timeline_state['event_data'][timeline_id] = {}
                
                # Store event data with timestamp
                if 'timestamp' not in update_data:
                    update_data['timestamp'] = int(time.time() * 1000)
                self._timeline_state['event_data'][timeline_id][event_id] = update_data.copy()
                
                # Mark this element as entangled with this timeline
                if 'entangled_timelines' not in self._timeline_state:
                    self._timeline_state['entangled_timelines'] = set()
                self._timeline_state['entangled_timelines'].add(timeline_id)
                
                logger.debug(f"Added event {event_id} of type {event_type} to timeline {timeline_id} for element {self.id}")
            
            # Call subclass-specific update logic
            self._update_state_internal(update_data, validated_timeline_context)
            
            # Notify observers of state change
            notification = {
                "type": "element_state_changed",
                "element_id": self.id,
                "event_type": event_type,
                "state_change": update_data.get('state_change', event_type),
                "element_state": self._state.value,
                "attention_state": {
                    "needs_attention": self._attention_requested,
                    "request_data": self._attention_request_data
                },
                "timestamp": update_data.get('timestamp', int(time.time() * 1000)),
                "timeline_context": validated_timeline_context
            }
            
            # Include state transition data if present
            if 'state_transition' in update_data:
                notification['state_transition'] = update_data['state_transition']
            
            # Notify observers
            try:
                self.notify_observers(notification)
            except Exception as e:
                logger.error(f"Error notifying observers for element {self.id}: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Error updating element {self.id} state: {e}", exc_info=True)
            return False
    
    def _update_state_internal(self, update_data: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """
        Internal method for custom state update logic.
        
        This should be overridden by subclasses to implement their specific
        state update behavior.
        
        Args:
            update_data: Event data to update with
            timeline_context: Timeline context for the update
        """
        pass
    
    def signal_attention_need(self, request_data: Dict[str, Any], timeline_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Signal that this element needs attention from the agent.
        
        Unlike the previous request_attention method, this creates a proper state update
        that will flow through the standard event processing mechanism.
        
        Args:
            request_data: Data about the attention need
            timeline_context: Timeline context for this update
            
        Returns:
            True if the signal was processed, False otherwise
        """
        attention_event = {
            "event_type": "needs_attention",
            "request_data": request_data,
            "timestamp": int(time.time() * 1000)
        }
        return self.update_state(attention_event, timeline_context)
    
    def clear_attention_need(self, timeline_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Clear any pending attention need.
        
        Args:
            timeline_context: Timeline context for this update
            
        Returns:
            True if the clear operation was processed, False otherwise
        """
        clear_event = {
            "event_type": "attention_cleared",
            "timestamp": int(time.time() * 1000)
        }
        return self.update_state(clear_event, timeline_context)
        
    def has_attention_need(self) -> bool:
        """Check if this element has a pending attention need."""
        return self._attention_requested
    
    def get_attention_data(self) -> Optional[Dict[str, Any]]:
        """Get data associated with the attention need."""
        return self._attention_request_data if self._attention_requested else None
    
    def _validate_timeline_context(self, timeline_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate and normalize timeline context.
        
        Args:
            timeline_context: Timeline context to validate
            
        Returns:
            Validated timeline context
        """
        context = timeline_context or {}
        if "timeline_id" not in context:
            # Use primary timeline if available
            if self._timeline_state["primary_timeline"]:
                context["timeline_id"] = self._timeline_state["primary_timeline"]
            else:
                context["timeline_id"] = f"timeline_{str(uuid.uuid4())[:8]}"
        
        if "is_primary" not in context:
            context["is_primary"] = (context["timeline_id"] == self._timeline_state["primary_timeline"])
            
        if "last_event_id" not in context:
            context["last_event_id"] = None
            
        return context
    
    def create_timeline_fork(self, source_timeline_id: str, fork_point_event_id: str, is_primary: bool = False) -> Optional[str]:
        """
        Create a new timeline fork from an existing timeline.
        
        Args:
            source_timeline_id: ID of the timeline to fork from
            fork_point_event_id: ID of the event to fork from
            is_primary: Whether this fork should be the primary timeline
            
        Returns:
            ID of the new timeline if successful, None otherwise
        """
        try:
            # Validate source timeline exists
            if source_timeline_id not in self._timeline_state["events"]:
                logger.error(f"Source timeline {source_timeline_id} not found")
                return None
                
            # Find fork point event
            fork_point = None
            for event in self._timeline_state["events"][source_timeline_id]:
                if event["id"] == fork_point_event_id:
                    fork_point = event
                    break
                    
            if not fork_point:
                logger.error(f"Fork point event {fork_point_event_id} not found in timeline {source_timeline_id}")
                return None
                
            # Create new timeline
            new_timeline_id = str(uuid.uuid4())
            self._timeline_state["events"][new_timeline_id] = []
            self._timeline_state["active_timelines"].add(new_timeline_id)
            
            # Copy events up to fork point
            for event in self._timeline_state["events"][source_timeline_id]:
                if event["id"] == fork_point_event_id:
                    break
                self._timeline_state["events"][new_timeline_id].append(event)
                
            # Set up timeline relationships
            self._timeline_state["timeline_relationships"][new_timeline_id] = {
                "parent_id": source_timeline_id,
                "is_primary": is_primary,
                "fork_point": fork_point_event_id
            }
            
            # Update primary timeline if needed
            if is_primary:
                self._timeline_state["primary_timeline"] = new_timeline_id
                
            # Initialize metadata
            self._timeline_state["timeline_metadata"][new_timeline_id] = {
                "created_at": int(time.time() * 1000),
                "last_updated": int(time.time() * 1000),
                "forked_from": source_timeline_id,
                "fork_point": fork_point_event_id
            }
            
            logger.info(f"Created new timeline {new_timeline_id} forked from {source_timeline_id}")
            return new_timeline_id
            
        except Exception as e:
            logger.error(f"Error creating timeline fork: {e}")
            return None
    
    def designate_primary_timeline(self, timeline_id: str) -> bool:
        """
        Designate a timeline as the primary timeline.
        
        Args:
            timeline_id: ID of the timeline to designate as primary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate timeline exists
            if timeline_id not in self._timeline_state["events"]:
                logger.error(f"Timeline {timeline_id} not found")
                return False
                
            # Update primary timeline
            self._timeline_state["primary_timeline"] = timeline_id
            
            # Update timeline relationship
            if timeline_id in self._timeline_state["timeline_relationships"]:
                self._timeline_state["timeline_relationships"][timeline_id]["is_primary"] = True
                
            logger.info(f"Designated timeline {timeline_id} as primary")
            return True
            
        except Exception as e:
            logger.error(f"Error designating primary timeline: {e}")
            return False
    
    def get_timeline_relationships(self, timeline_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the relationships for a specific timeline.
        
        Args:
            timeline_id: ID of the timeline to get relationships for
            
        Returns:
            Dictionary of relationships if found, None otherwise
        """
        return self._timeline_state["timeline_relationships"].get(timeline_id)
    
    def get_primary_timeline(self) -> Optional[str]:
        """
        Get the ID of the primary timeline.
        
        Returns:
            ID of the primary timeline if one exists, None otherwise
        """
        return self._timeline_state["primary_timeline"]
    
    def is_primary_timeline(self, timeline_id: str) -> bool:
        """
        Check if a timeline is the primary timeline.
        
        Args:
            timeline_id: ID of the timeline to check
            
        Returns:
            True if the timeline is primary, False otherwise
        """
        return timeline_id == self._timeline_state["primary_timeline"]
    
    def get_delegate(self):
        """
        Get the element delegate for rendering this element.
        
        The delegate is responsible for transforming the element's state
        into renderable content for the agent.
        
        If no custom delegate has been set, a default delegate is created
        and returned.
        
        Returns:
            Element delegate instance
        """
        if not self._delegate:
            # Import here to avoid circular imports
            try:
                from rendering.delegates import DefaultDelegate
                self._delegate = DefaultDelegate(self)
            except ImportError:
                logger.warning(f"Could not create default delegate for element {self.id}")
                return None
                
        return self._delegate
    
    def set_delegate(self, delegate) -> None:
        """
        Set a custom element delegate for rendering.
        
        Args:
            delegate: Element delegate instance
        """
        self._delegate = delegate
        logger.debug(f"Set custom delegate for element {self.id}") 