"""
Container Component
Component for managing mounted elements within a Space.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Set, Tuple
import uuid
import time

from ..base_component import Component
from ...base import MountType, BaseElement
from collections import defaultdict # Needed for listener storage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContainerComponent(Component):
    """
    Component for managing mounted elements within a Space.
    
    This component provides the ability to:
    - Mount and unmount elements
    - Track mounted elements
    - Provide access to mounted elements
    """
    
    # Component unique type identifier
    COMPONENT_TYPE: str = "container"
    
    # Event types this component handles
    HANDLED_EVENT_TYPES: List[str] = [
        "element_mounted",
        "element_unmounted"
    ]
    
    def __init__(self, element=None):
        """
        Initialize the container component.
        
        Args:
            element: The Element this component is attached to
        """
        super().__init__(element)
        
        # Mounted elements
        self._state = {
            "mounted_elements": {}  # Dict[str, Dict[str, Any]]
        }
        # Listener lists (not part of loomable state)
        self._mount_listeners: List[Callable[[str, BaseElement], None]] = []
        self._unmount_listeners: List[Callable[[str, BaseElement], None]] = []
    
    # --- Listener Management ---
    def add_mount_listener(self, listener: Callable[[str, BaseElement], None]):
        """Adds a listener to be called after an element is mounted.
           The listener receives (mount_id, mounted_element)."""
        if listener not in self._mount_listeners:
            self._mount_listeners.append(listener)

    def remove_mount_listener(self, listener: Callable[[str, BaseElement], None]):
        """Removes a mount listener."""
        try:
            self._mount_listeners.remove(listener)
        except ValueError:
            pass # Listener not found

    def add_unmount_listener(self, listener: Callable[[str, BaseElement], None]):
        """Adds a listener to be called before an element is unmounted.
           The listener receives (mount_id, element_being_unmounted)."""
        if listener not in self._unmount_listeners:
            self._unmount_listeners.append(listener)

    def remove_unmount_listener(self, listener: Callable[[str, BaseElement], None]):
        """Removes an unmount listener."""
        try:
            self._unmount_listeners.remove(listener)
        except ValueError:
            pass

    def _notify_mount_listeners(self, mount_id: str, element: BaseElement):
        """Calls all registered mount listeners."""
        logger.debug(f"Notifying {len(self._mount_listeners)} mount listeners about {mount_id}")
        for listener in self._mount_listeners[:]: # Iterate copy in case listener removes itself
            try:
                listener(mount_id, element)
            except Exception as e:
                logger.error(f"Error calling mount listener {listener.__name__}: {e}", exc_info=True)

    def _notify_unmount_listeners(self, mount_id: str, element: BaseElement):
        """Calls all registered unmount listeners."""
        logger.debug(f"Notifying {len(self._unmount_listeners)} unmount listeners about {mount_id}")
        for listener in self._unmount_listeners[:]: # Iterate copy
            try:
                listener(mount_id, element)
            except Exception as e:
                logger.error(f"Error calling unmount listener {listener.__name__}: {e}", exc_info=True)

    # --- Mounting/Unmounting Logic ---
    def mount_element(self, element: BaseElement, mount_id: Optional[str] = None, 
                     mount_type: MountType = MountType.INCLUSION) -> bool:
        """
        Mount an element in this container.
        
        Args:
            element: Element to mount
            mount_id: ID to use for the mount point (defaults to element.id)
            mount_type: Type of mounting (inclusion or uplink)
            
        Returns:
            True if the element was mounted, False otherwise
        """
        if not self._is_initialized or not self._is_enabled:
            logger.warning(f"Cannot mount element: Container component {self.id} is not initialized or enabled")
            return False
            
        # Use element ID as mount ID if not provided
        mount_id = mount_id or element.id
        
        # Check if mount ID is already in use
        if mount_id in self._state["mounted_elements"]:
            logger.warning(f"Mount ID {mount_id} is already in use")
            return False
            
        # Check for circular mounting
        if self._would_create_circular_mount(element):
            logger.warning(f"Cannot mount element {element.id} as it would create a circular mount")
            return False
            
        # Set parent reference
        element._set_parent(self.element.id, mount_type)
        
        # Store the mounted element
        self._state["mounted_elements"][mount_id] = {
            "element": element,
            "mount_type": mount_type,
            "mount_time": int(time.time() * 1000)
        }
        
        # Record the mount event
        self._record_mount_event(element, mount_id, mount_type)
        
        # *** Notify Listeners AFTER successful mount ***
        self._notify_mount_listeners(mount_id, element)
        
        logger.info(f"Mounted element {element.id} at {mount_id} in {self.element.id}")
        return True
    
    def unmount_element(self, mount_id: str) -> bool:
        """
        Unmount an element.
        
        Args:
            mount_id: ID of the mount point
            
        Returns:
            True if the element was unmounted, False otherwise
        """
        if not self._is_initialized or not self._is_enabled:
            logger.warning(f"Cannot unmount element: Container component {self.id} is not initialized or enabled")
            return False
            
        # Check if mount ID exists
        if mount_id not in self._state["mounted_elements"]:
            logger.warning(f"Mount ID {mount_id} not found")
            return False
            
        # Get the element
        mount_info = self._state["mounted_elements"][mount_id]
        element = mount_info["element"]
        mount_type = mount_info["mount_type"]
        
        # *** Notify Listeners BEFORE unmounting ***
        self._notify_unmount_listeners(mount_id, element)
        
        # Record the unmount event
        self._record_unmount_event(element, mount_id, mount_type)
        
        # Clear parent reference
        element._clear_parent()
        
        # Remove the mount
        del self._state["mounted_elements"][mount_id]
        
        logger.info(f"Unmounted element {element.id} from {mount_id} in {self.element.id}")
        return True
    
    def get_mounted_element(self, mount_id: str) -> Optional[BaseElement]:
        """
        Get a mounted element by ID.
        
        Args:
            mount_id: ID of the mount point
            
        Returns:
            The mounted element, or None if not found
        """
        if mount_id not in self._state["mounted_elements"]:
            return None
            
        return self._state["mounted_elements"][mount_id]["element"]
    
    def get_mounted_elements(self) -> Dict[str, BaseElement]:
        """
        Get all mounted elements.
        
        Returns:
            Dictionary of mount ID to element
        """
        return {
            mount_id: mount_info["element"]
            for mount_id, mount_info in self._state["mounted_elements"].items()
        }
    
    def get_mounted_elements_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all mounted elements.
        
        Returns:
            Dictionary of mount ID to mount info
        """
        return {
            mount_id: {
                "element_id": mount_info["element"].id,
                "element_name": mount_info["element"].name,
                "element_type": mount_info["element"].__class__.__name__,
                "mount_type": mount_info["mount_type"].value,
                "mount_time": mount_info["mount_time"]
            }
            for mount_id, mount_info in self._state["mounted_elements"].items()
        }
    
    def _would_create_circular_mount(self, element: BaseElement) -> bool:
        """
        Check if mounting an element would create a circular mount.
        
        Args:
            element: Element to check
            
        Returns:
            True if mounting would create a circular mount, False otherwise
        """
        # Check if this element is the same as the one to mount
        if element.id == self.element.id:
            return True
            
        # If the element to mount has no parent, it can't create a circular mount
        parent_info = element.get_parent_info()
        if not parent_info:
            return False
            
        # If the parent of the element to mount is this element, it's directly safe
        if parent_info["parent_id"] == self.element.id:
            return False
            
        # Now we need to check if this element's parent chain includes the element to mount
        current_element = self.element
        visited_elements = set([current_element.id])
        
        while True:
            parent_info = current_element.get_parent_info()
            if not parent_info:
                # Reached the top of the chain, no circular mount
                return False
                
            parent_id = parent_info["parent_id"]
            
            # Check if we've seen this parent before (cycle detection)
            if parent_id in visited_elements:
                # Found a cycle, but not necessarily involving element to mount
                logger.warning(f"Detected cycle in parent chain of {self.element.id}")
                break
                
            visited_elements.add(parent_id)
            
            # If the parent is the element to mount, it would create a circular mount
            if parent_id == element.id:
                return True
                
            # Get the parent element
            registry = self.element.get_registry()
            if not registry:
                # Can't check further without registry
                logger.warning(f"Cannot check for circular mount: No registry reference")
                break
                
            parent_element = registry.get_element(parent_id)
            if not parent_element:
                # Can't check further without parent element
                logger.warning(f"Cannot check for circular mount: Parent element {parent_id} not found")
                break
                
            current_element = parent_element
            
        # If we can't determine for sure, err on the side of caution
        return False
    
    def _record_mount_event(self, element: BaseElement, mount_id: str, mount_type: MountType) -> None:
        """
        Record an element mount event in the timeline.
        
        Args:
            element: Element that was mounted
            mount_id: ID of the mount point
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
        
        # Notify the Element to handle this event
        self.element.handle_event(event_data, {"timeline_id": "primary"})
    
    def _record_unmount_event(self, element: BaseElement, mount_id: str, mount_type: MountType) -> None:
        """
        Record an element unmount event in the timeline.
        
        Args:
            element: Element that was unmounted
            mount_id: ID of the mount point
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
        
        # Notify the Element to handle this event
        self.element.handle_event(event_data, {"timeline_id": "primary"})
        
    def _on_cleanup(self) -> bool:
        """
        Clean up this component.
        
        Returns:
            True if cleanup was successful, False otherwise
        """
        # Unmount all elements
        for mount_id in list(self._state["mounted_elements"].keys()):
            self.unmount_element(mount_id)
            
        return True 