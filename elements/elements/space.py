"""
Space Element
Base class for all space elements in the Bot Framework.
"""

import logging
from typing import Dict, Any, Optional, List, Set, Type, Callable
import uuid
import time

from .base import BaseElement, MountType, ElementState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Space(BaseElement):
    """
    Base class for all space elements.
    
    Spaces are elements that can contain other elements (both Spaces and Objects).
    They provide the foundation for creating complex environments and hierarchies.
    
    Spaces can be used to:
    - Create containers for grouping related elements
    - Define logical boundaries for elements
    - Implement specialized environments like InnerSpace, ChatSpace, etc.
    """
    
    # Spaces can contain other elements
    IS_SPACE = True
    
    # Events that all spaces handle
    EVENT_TYPES = [
        "element_mounted", 
        "element_unmounted", 
        "element_opened", 
        "element_closed",
        "space_focus_changed"
    ]
    
    def __init__(self, element_id: str, name: str, description: str):
        """
        Initialize the space.
        
        Args:
            element_id: Unique identifier for this space
            name: Human-readable name for this space
            description: Description of this space's purpose
        """
        super().__init__(element_id, name, description)
        
        # Focused element ID (if any)
        self._focused_element_id: Optional[str] = None
        
        # Initialize space-specific tools
        self._register_space_tools()
        
        logger.info(f"Created space: {name} ({element_id})")
    
    def _register_space_tools(self) -> None:
        """Register tools specific to spaces."""
        
        @self.register_tool(
            name="mount_element",
            description="Mount an element in this space",
            parameter_descriptions={
                "element_id": "ID of the element to mount",
                "mount_id": "Optional identifier for the mount point, defaults to element_id",
                "mount_type": "Type of mounting (inclusion or uplink)"
            }
        )
        def mount_element_tool(element_id: str, mount_id: Optional[str] = None, 
                              mount_type: str = "inclusion") -> Dict[str, Any]:
            """
            Tool to mount an element in this space.
            
            Args:
                element_id: ID of the element to mount
                mount_id: Optional identifier for the mount point
                mount_type: Type of mounting (inclusion or uplink)
                
            Returns:
                Result of the mounting operation
            """
            # Get element from registry (implementation depends on context)
            # This would typically be handled by a method in a concrete subclass
            element = self._get_element_by_id(element_id)
            if not element:
                return {
                    "success": False,
                    "error": f"Element with ID {element_id} not found"
                }
            
            # Determine mount type
            try:
                mt = MountType.INCLUSION if mount_type == "inclusion" else MountType.UPLINK
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid mount type: {mount_type}"
                }
            
            # Mount the element
            success = self.mount_element(element, mount_id, mt)
            
            return {
                "success": success,
                "mount_id": mount_id or element_id,
                "element_id": element_id,
                "mount_type": mount_type
            }
        
        @self.register_tool(
            name="unmount_element",
            description="Unmount an element from this space",
            parameter_descriptions={
                "mount_id": "Identifier for the mount point"
            }
        )
        def unmount_element_tool(mount_id: str) -> Dict[str, Any]:
            """
            Tool to unmount an element from this space.
            
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
                    "element_name": element.name
                }
            
            # Unmount the element
            success = self.unmount_element(mount_id)
            
            return {
                "success": success,
                "mount_id": mount_id,
                **element_info
            }
        
        @self.register_tool(
            name="focus_element",
            description="Focus on a specific element in this space",
            parameter_descriptions={
                "mount_id": "Identifier for the mount point"
            }
        )
        def focus_element_tool(mount_id: str) -> Dict[str, Any]:
            """
            Tool to focus on a specific element in this space.
            
            Args:
                mount_id: Identifier for the mount point
                
            Returns:
                Result of the focus operation
            """
            # Check if the element exists
            if mount_id not in self._mounted_elements:
                return {
                    "success": False,
                    "error": f"No element mounted at {mount_id}"
                }
            
            # Get the element
            element = self._mounted_elements[mount_id]["element"]
            
            # Set focus
            self._set_focus(mount_id)
            
            return {
                "success": True,
                "mount_id": mount_id,
                "element_id": element.id,
                "element_name": element.name
            }
        
        @self.register_tool(
            name="clear_focus",
            description="Clear focus from any element",
            parameter_descriptions={}
        )
        def clear_focus_tool() -> Dict[str, Any]:
            """
            Tool to clear focus from any element.
            
            Returns:
                Result of the clear focus operation
            """
            previous_focus = self._focused_element_id
            
            # Clear focus
            self._clear_focus()
            
            return {
                "success": True,
                "previous_focus": previous_focus
            }
        
        @self.register_tool(
            name="list_mounted_elements",
            description="List all elements mounted in this space",
            parameter_descriptions={}
        )
        def list_mounted_elements_tool() -> Dict[str, Any]:
            """
            Tool to list all elements mounted in this space.
            
            Returns:
                List of mounted elements
            """
            return {
                "success": True,
                "mounted_elements": self._get_mounted_elements_info()
            }
        
        @self.register_tool(
            name="get_element_info",
            description="Get information about a specific mounted element",
            parameter_descriptions={
                "mount_id": "Identifier for the mount point"
            }
        )
        def get_element_info_tool(mount_id: str) -> Dict[str, Any]:
            """
            Tool to get information about a specific mounted element.
            
            Args:
                mount_id: Identifier for the mount point
                
            Returns:
                Information about the element
            """
            # Check if the element exists
            if mount_id not in self._mounted_elements:
                return {
                    "success": False,
                    "error": f"No element mounted at {mount_id}"
                }
            
            # Get element info
            element_info = self._get_mounted_elements_info().get(mount_id, {})
            
            return {
                "success": True,
                "mount_id": mount_id,
                "element_info": element_info
            }
    
    def get_interior_state(self) -> Dict[str, Any]:
        """
        Get the interior state of the space.
        
        Returns:
            Dictionary representation of the interior state
        """
        # Get basic interior state from BaseElement
        interior = super().get_interior_state()
        
        # Add space-specific state
        interior.update({
            "focused_element_id": self._focused_element_id,
        })
        
        return interior
    
    def get_exterior_state(self) -> Optional[Dict[str, Any]]:
        """
        Get the exterior state of the space.
        
        Returns:
            Dictionary representation of the exterior state, or None
        """
        if not self.HAS_EXTERIOR:
            return None
            
        # Get basic exterior state from BaseElement
        exterior = super().get_exterior_state()
        
        # Add space-specific state
        exterior.update({
            "element_count": len(self._mounted_elements),
            "focused_element_id": self._focused_element_id
        })
        
        return exterior
    
    def _set_focus(self, mount_id: str) -> bool:
        """
        Set focus on a specific element.
        
        Args:
            mount_id: Identifier for the mount point
            
        Returns:
            True if focus was set successfully, False otherwise
        """
        # Check if the element exists
        if mount_id not in self._mounted_elements:
            logger.warning(f"Cannot focus on non-existent mount point {mount_id}")
            return False
        
        # Change focus
        previous_focus = self._focused_element_id
        self._focused_element_id = mount_id
        
        # Record the focus change event
        event_data = {
            "event_type": "space_focus_changed",
            "space_id": self.id,
            "previous_focus": previous_focus,
            "current_focus": mount_id,
            "timestamp": int(time.time() * 1000)
        }
        
        # Get primary timeline
        timeline_id = self._timeline_state["primary_timeline"]
        if timeline_id is not None:
            self.update_state(event_data, {"timeline_id": timeline_id})
        
        logger.info(f"Set focus in space {self.id} to element at mount point {mount_id}")
        
        # Notify observers
        self.notify_observers({
            "type": "focus_changed",
            "space_id": self.id,
            "previous_focus": previous_focus,
            "current_focus": mount_id
        })
        
        return True
    
    def _clear_focus(self) -> None:
        """Clear focus from any element."""
        if self._focused_element_id is None:
            return
        
        previous_focus = self._focused_element_id
        self._focused_element_id = None
        
        # Record the focus change event
        event_data = {
            "event_type": "space_focus_changed",
            "space_id": self.id,
            "previous_focus": previous_focus,
            "current_focus": None,
            "timestamp": int(time.time() * 1000)
        }
        
        # Get primary timeline
        timeline_id = self._timeline_state["primary_timeline"]
        if timeline_id is not None:
            self.update_state(event_data, {"timeline_id": timeline_id})
        
        logger.info(f"Cleared focus in space {self.id}")
        
        # Notify observers
        self.notify_observers({
            "type": "focus_changed",
            "space_id": self.id,
            "previous_focus": previous_focus,
            "current_focus": None
        })
    
    def get_focused_element(self) -> Optional[BaseElement]:
        """
        Get the currently focused element.
        
        Returns:
            The focused element if any, None otherwise
        """
        if self._focused_element_id is None:
            return None
            
        return self.get_mounted_element(self._focused_element_id)
    
    def _get_element_by_id(self, element_id: str) -> Optional[BaseElement]:
        """
        Get an element by ID.
        
        This is a placeholder method that should be overridden by concrete subclasses
        to provide access to the element registry.
        
        Args:
            element_id: ID of the element to get
            
        Returns:
            The element if found, None otherwise
        """
        # This method should be overridden by concrete subclasses
        logger.warning("_get_element_by_id called on base Space class")
        return None
    
    def _on_open(self) -> None:
        """
        Hook called when the space is opened.
        
        Opens all elements mounted as inclusions.
        """
        for mount_id, info in self._mounted_elements.items():
            if info["mount_type"] == MountType.INCLUSION:
                element = info["element"]
                element.open()
    
    def _on_close(self) -> None:
        """
        Hook called when the space is closed.
        
        Closes all elements mounted as inclusions.
        """
        for mount_id, info in self._mounted_elements.items():
            if info["mount_type"] == MountType.INCLUSION:
                element = info["element"]
                element.close() 