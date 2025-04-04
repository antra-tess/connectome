"""
Base Element
Base class for all elements in the Component-based architecture.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Set, Tuple, Literal, Type
import uuid
import time
import enum

from .components import Component

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
    
    In the component-based architecture, Elements are minimal structural nodes
    that primarily maintain ID, name, and parent references. All state and behavior
    are implemented by Components attached to the Element.
    
    Elements:
    - Have a unique ID and name
    - Can be mounted in a parent element 
    - Can have multiple Components attached to provide functionality
    - Are structural nodes in the scene graph
    - Delegate most functionality to their attached Components
    """
    
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
        
        # Parent element (where this element is mounted)
        self._parent_element: Optional[Tuple[str, MountType]] = None
        
        # Registry reference (for sending messages to activity layer)
        self._registry = None
        
        # Components
        self._components: Dict[str, Any] = {}
        
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
    
    def add_component(self, component_type: Type, **kwargs) -> Optional[Any]:
        """
        Add a component to this element.
        
        Args:
            component_type: Type of component to add
            **kwargs: Additional arguments to pass to the component constructor
        
        Returns:
            The added component, or None if the component could not be added
        """
        # Create the component
        try:
            component = component_type(element=self, **kwargs)
        except Exception as e:
            logger.error(f"Error creating component of type {component_type.__name__}: {e}")
            return None
        
        # Check if we already have a component of this type
        for existing in self._components.values():
            if existing.COMPONENT_TYPE == component.COMPONENT_TYPE:
                logger.warning(f"Element {self.id} already has a component of type {component.COMPONENT_TYPE}")
                return None
        
        # Validate dependencies
        if not component.validate_dependencies(self._components):
            logger.error(f"Component {component.COMPONENT_TYPE} has unsatisfied dependencies")
            return None
        
        # Initialize the component
        if not component.initialize():
            logger.error(f"Failed to initialize component {component.COMPONENT_TYPE}")
            return None
        
        # Add to components dictionary
        self._components[component.id] = component
        
        logger.debug(f"Added component {component.COMPONENT_TYPE} to element {self.id}")
        return component
    
    def remove_component(self, component_id: str) -> bool:
        """
        Remove a component from this element.
        
        Args:
            component_id: ID of the component to remove
            
        Returns:
            True if the component was removed, False otherwise
        """
        if component_id not in self._components:
            return False
        
        component = self._components[component_id]
        
        # Check for dependents
        for other_comp in self._components.values():
            if component.COMPONENT_TYPE in other_comp.DEPENDENCIES:
                logger.warning(f"Cannot remove component {component.COMPONENT_TYPE}: it is a dependency for {other_comp.COMPONENT_TYPE}")
                return False
        
        # Cleanup and remove
        if not component.cleanup():
            logger.warning(f"Component {component.COMPONENT_TYPE} reported cleanup failure")
        
        del self._components[component_id]
        logger.debug(f"Removed component {component.COMPONENT_TYPE} from element {self.id}")
        
        return True
    
    def get_component(self, component_id: str) -> Optional[Any]:
        """
        Get a component by ID.
        
        Args:
            component_id: ID of the component to get
            
        Returns:
            The component, or None if not found
        """
        return self._components.get(component_id)
    
    def get_component_by_type(self, component_type: str) -> Optional[Any]:
        """
        Get a component by type.
        
        Args:
            component_type: Type of component to get
            
        Returns:
            The first component of the specified type, or None if not found
        """
        for component in self._components.values():
            if component.COMPONENT_TYPE == component_type:
                return component
            return None
    
    def get_components(self) -> Dict[str, Any]:
        """
        Get all components.
        
        Returns:
            Dictionary of component ID to component
        """
        return self._components.copy()
    
    def handle_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handle an event by delegating to components.
        
        Args:
            event: Event data
            timeline_context: Timeline context for this event
            
        Returns:
            True if the event was handled by any component, False otherwise
        """
        # Delegate event to all components
        handled = False
        
        for component in self._components.values():
            if component.handle_event(event, timeline_context):
                handled = True
                
        return handled
    
    def _set_parent(self, parent_id: str, mount_type: MountType) -> None:
        """
        Set the parent element.
        
        Args:
            parent_id: ID of the parent element
            mount_type: Type of mounting
        """
        self._parent_element = (parent_id, mount_type)
    
    def _clear_parent(self) -> None:
        """Clear the parent element reference."""
        self._parent_element = None
    
    def get_parent_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the parent element.
        
        Returns:
            Dictionary with parent ID and mount type, or None if not mounted
        """
        if not self._parent_element:
            return None
            
        parent_id, mount_type = self._parent_element
        return {
            "parent_id": parent_id,
            "mount_type": mount_type.value
        }
    
    def cleanup(self) -> bool:
        """
        Clean up the element and all its components.
        
        Returns:
            True if cleanup was successful, False otherwise
        """
        success = True
        
        # Clean up all components
        for component_id in list(self._components.keys()):
            if not self.remove_component(component_id):
                success = False
        
        return success 