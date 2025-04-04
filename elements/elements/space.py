"""
Space
Base class for space elements that can contain other elements.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Type, Set
import uuid
import time

from .base import BaseElement, MountType
from .components.space import ContainerComponent, TimelineComponent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Space(BaseElement):
    """
    Base class for space elements that can contain other elements.
    
    Spaces are structural containers that:
    - Can contain other elements (Objects or Spaces)
    - Manage their own timeline (Loom DAG)
    - Process events and route them to contained elements
    
    In the component-based architecture, Space functionality is implemented
    by components like ContainerComponent and TimelineComponent.
    """
    
    # Space is a container for other elements
    IS_SPACE = True
    
    def __init__(self, element_id: str, name: str, description: str):
        """
        Initialize the space.
        
        Args:
            element_id: Unique identifier for this space
            name: Human-readable name for this space
            description: Description of this space's purpose
        """
        super().__init__(element_id, name, description)
        
        # Initialize space with required components
        self._container = self.add_component(ContainerComponent)
        self._timeline = self.add_component(TimelineComponent)
        
        if not self._container or not self._timeline:
            logger.error(f"Failed to initialize required components for space {element_id}")
        
        logger.info(f"Created space: {name} ({element_id})")
    
    def mount_element(self, element: BaseElement, mount_id: Optional[str] = None, mount_type: MountType = MountType.INCLUSION) -> bool:
            """
        Mount an element in this space.
            
            Args:
            element: Element to mount
            mount_id: ID to use for the mount point (defaults to element.id)
                mount_type: Type of mounting (inclusion or uplink)
        
        Returns:
            True if the element was mounted, False otherwise
        """
            container = self.get_component_by_type("container")
            if not container:
                logger.error(f"Cannot mount element: Space {self.id} has no container component")
                return False
                
            return container.mount_element(element, mount_id, mount_type)
    
    def unmount_element(self, mount_id: str) -> bool:
        """
        Unmount an element.
        
        Args:
            mount_id: ID of the mount point
            
        Returns:
            True if the element was unmounted, False otherwise
        """
        container = self.get_component_by_type("container")
        if not container:
            logger.error(f"Cannot unmount element: Space {self.id} has no container component")
            return False
        
        return container.unmount_element(mount_id)
    
    def get_mounted_element(self, mount_id: str) -> Optional[BaseElement]:
        """
        Get a mounted element by ID.
        
        Args:
            mount_id: ID of the mount point
            
        Returns:
            The mounted element, or None if not found
        """
        container = self.get_component_by_type("container")
        if not container:
            logger.error(f"Cannot get mounted element: Space {self.id} has no container component")
        return None
    
        return container.get_mounted_element(mount_id)
    
    def get_mounted_elements(self) -> Dict[str, BaseElement]:
        """
        Get all mounted elements.
        
        Returns:
            Dictionary of mount ID to element
        """
        container = self.get_component_by_type("container")
        if not container:
            logger.error(f"Cannot get mounted elements: Space {self.id} has no container component")
            return {}
            
        return container.get_mounted_elements()
    
    def receive_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """
        Receive an event and process it in the appropriate timeline.
        
        Args:
            event: Event data
            timeline_context: Timeline context information
        """
        # First add the event to the timeline
        timeline = self.get_component_by_type("timeline")
        if not timeline:
            logger.error(f"Cannot receive event: Space {self.id} has no timeline component")
            return
            
        event_id = timeline.add_event_to_timeline(event, timeline_context)
        if not event_id:
            logger.error(f"Failed to add event to timeline: {timeline_context.get('timeline_id')}")
            return
            
        # Process the event
        self._process_event(event, timeline_context)
    
    def _process_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """
        Process an event.
        
        Args:
            event: Event data
            timeline_context: Timeline context information
        """
        # Handle the event in this element
        self.handle_event(event, timeline_context)
        
        # Route the event to any mounted elements that can handle it
        container = self.get_component_by_type("container")
        if not container:
            return
            
        for element in container.get_mounted_elements().values():
            # Check if the element can handle this event type
            if hasattr(element, 'EVENT_TYPES') and event.get('event_type') in element.EVENT_TYPES:
                element.handle_event(event, timeline_context)
    
    def add_event_to_timeline(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> Optional[str]:
        """
        Add an event to a timeline.
        
        Args:
            event: Event data to add
            timeline_context: Timeline context information
            
        Returns:
            ID of the created event, or None if it could not be added
        """
        timeline = self.get_component_by_type("timeline")
        if not timeline:
            logger.error(f"Cannot add event to timeline: Space {self.id} has no timeline component")
            return None
            
        return timeline.add_event_to_timeline(event, timeline_context)
    
    def update_state(self, update_data: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Update state in a specific timeline.
        
        Args:
            update_data: Data to update
            timeline_context: Timeline context information
            
        Returns:
            True if the state was updated, False otherwise
        """
        timeline = self.get_component_by_type("timeline")
        if not timeline:
            logger.error(f"Cannot update state: Space {self.id} has no timeline component")
            return False
            
        return timeline.update_state(update_data, timeline_context)
    
    def create_timeline_fork(self, source_timeline_id: str, fork_point_event_id: str, 
                           is_primary: bool = False) -> Optional[str]:
        """
        Create a new timeline forked from an existing one.
        
        Args:
            source_timeline_id: ID of the timeline to fork from
            fork_point_event_id: ID of the event to fork from
            is_primary: Whether this should be the primary timeline
            
        Returns:
            ID of the new timeline, or None if it could not be created
        """
        timeline = self.get_component_by_type("timeline")
        if not timeline:
            logger.error(f"Cannot create timeline fork: Space {self.id} has no timeline component")
            return None
            
        return timeline.create_timeline_fork(source_timeline_id, fork_point_event_id, is_primary)
    
    def designate_primary_timeline(self, timeline_id: str) -> bool:
        """
        Designate a timeline as the primary timeline.
        
        Args:
            timeline_id: ID of the timeline to designate as primary
            
        Returns:
            True if the timeline was designated as primary, False otherwise
        """
        timeline = self.get_component_by_type("timeline")
        if not timeline:
            logger.error(f"Cannot designate primary timeline: Space {self.id} has no timeline component")
            return False
            
        return timeline.designate_primary_timeline(timeline_id)
    
    def get_timeline_relationships(self, timeline_id: str) -> Optional[Dict[str, Any]]:
        """
        Get relationships for a timeline.
        
        Args:
            timeline_id: ID of the timeline
            
        Returns:
            Dictionary of relationship information, or None if not found
        """
        timeline = self.get_component_by_type("timeline")
        if not timeline:
            logger.error(f"Cannot get timeline relationships: Space {self.id} has no timeline component")
        return None
            
        return timeline.get_timeline_relationships(timeline_id)
    
    def get_primary_timeline(self) -> Optional[str]:
        """
        Get the ID of the primary timeline.
        
        Returns:
            ID of the primary timeline, or None if not set
        """
        timeline = self.get_component_by_type("timeline")
        if not timeline:
            logger.error(f"Cannot get primary timeline: Space {self.id} has no timeline component")
            return None
            
        return timeline.get_primary_timeline()
    
    def is_primary_timeline(self, timeline_id: str) -> bool:
        """
        Check if a timeline is the primary timeline.
        
        Args:
            timeline_id: ID of the timeline to check
            
        Returns:
            True if the timeline is primary, False otherwise
        """
        timeline = self.get_component_by_type("timeline")
        if not timeline:
            logger.error(f"Cannot check if timeline is primary: Space {self.id} has no timeline component")
            return False
            
        return timeline.is_primary_timeline(timeline_id)
        
    def get_timeline_events(self, timeline_id: str) -> List[Dict[str, Any]]:
        """
        Get all events in a timeline.
        
        Args:
            timeline_id: ID of the timeline
            
        Returns:
            List of events in the timeline
        """
        timeline = self.get_component_by_type("timeline")
        if not timeline:
            logger.error(f"Cannot get timeline events: Space {self.id} has no timeline component")
            return []
            
        return timeline.get_timeline_events(timeline_id)
    
    def execute_action_on_element(self, element_id: str, action_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action on an element in this space.
        
        Args:
            element_id: ID of the element to act on
            action_name: Name of the action to execute
            parameters: Parameters for the action
            
        Returns:
            Result of the action execution
        """
        # Check if this is an action on this space itself
        if element_id == self.id:
            tool = self.get_component_by_type("tool_provider")
            if tool:
                return tool.execute_tool(action_name, **parameters)
            else:
                return {"error": f"Space {self.id} does not have a tool provider component"}
                
        # Otherwise, look for the element in mounted elements
        container = self.get_component_by_type("container")
        if not container:
            return {"error": f"Space {self.id} has no container component"}
            
        # Try to find by direct ID
        element = None
        for mounted in container.get_mounted_elements().values():
            if mounted.id == element_id:
                element = mounted
                break
                
        # If not found, try to find by mount ID
        if not element:
            element = container.get_mounted_element(element_id)
            
        if not element:
            return {"error": f"Element not found: {element_id}"}
            
        # Execute the action on the element
        tool = element.get_component_by_type("tool_provider")
        if tool:
            return tool.execute_tool(action_name, **parameters)
        else:
            return {"error": f"Element {element_id} does not have a tool provider component"} 