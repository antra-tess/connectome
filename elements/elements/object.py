"""
Object Element
Base class for all object elements in the Bot Framework.
"""

import logging
from typing import Dict, Any, Optional, List, Set, Type, Callable
import uuid
import time

from .base import BaseElement, MountType, ElementState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Object(BaseElement):
    """
    Base class for all object elements.
    
    Objects are elements that represent tools, resources, or entities that can be
    interacted with but cannot contain other elements.
    
    Unlike Spaces, Objects:
    - Cannot be entered by participants
    - Cannot contain other elements (no mounting)
    - Often represent specific tools or resources
    - Can have their own specialized interaction mechanisms
    
    Examples of Objects include:
    - Chat elements
    - Document viewers
    - Tools and utilities
    - Data visualizations
    """
    
    # Objects cannot contain other elements
    IS_SPACE = False
    
    # Events that all objects handle
    EVENT_TYPES = [
        "object_interacted", 
        "object_property_changed"
    ]
    
    def __init__(self, element_id: str, name: str, description: str):
        """
        Initialize the object.
        
        Args:
            element_id: Unique identifier for this object
            name: Human-readable name for this object
            description: Description of this object's purpose
        """
        super().__init__(element_id, name, description)
        
        # Object properties (can be extended by subclasses)
        self._properties: Dict[str, Any] = {}
        
        # Initialize object-specific tools
        self._register_object_tools()
        
        logger.info(f"Created object: {name} ({element_id})")
    
    def _register_object_tools(self) -> None:
        """Register tools specific to objects."""
        
        @self.register_tool(
            name="get_properties",
            description="Get all properties of this object",
            parameter_descriptions={}
        )
        def get_properties_tool() -> Dict[str, Any]:
            """
            Tool to get all properties of this object.
            
            Returns:
                Dictionary of object properties
            """
            return {
                "success": True,
                "properties": self._properties
            }
        
        @self.register_tool(
            name="get_property",
            description="Get a specific property of this object",
            parameter_descriptions={
                "property_name": "Name of the property to get"
            }
        )
        def get_property_tool(property_name: str) -> Dict[str, Any]:
            """
            Tool to get a specific property of this object.
            
            Args:
                property_name: Name of the property to get
                
            Returns:
                Value of the property if found, error otherwise
            """
            if property_name not in self._properties:
                return {
                    "success": False,
                    "error": f"Property {property_name} not found"
                }
                
            return {
                "success": True,
                "property_name": property_name,
                "value": self._properties[property_name]
            }
        
        @self.register_tool(
            name="set_property",
            description="Set a property of this object",
            parameter_descriptions={
                "property_name": "Name of the property to set",
                "value": "Value to set the property to"
            }
        )
        def set_property_tool(property_name: str, value: Any) -> Dict[str, Any]:
            """
            Tool to set a property of this object.
            
            Args:
                property_name: Name of the property to set
                value: Value to set the property to
                
            Returns:
                Result of the operation
            """
            # Set the property
            old_value = self._properties.get(property_name)
            self._properties[property_name] = value
            
            # Record the property change event
            event_data = {
                "event_type": "object_property_changed",
                "object_id": self.id,
                "property_name": property_name,
                "old_value": old_value,
                "new_value": value,
                "timestamp": int(time.time() * 1000)
            }
            
            # Get primary timeline
            timeline_id = self._timeline_state["primary_timeline"]
            if timeline_id is not None:
                self.update_state(event_data, {"timeline_id": timeline_id})
            
            logger.info(f"Set property {property_name} of object {self.id} to {value}")
            
            # Notify observers
            self.notify_observers({
                "type": "property_changed",
                "object_id": self.id,
                "property_name": property_name,
                "old_value": old_value,
                "new_value": value
            })
            
            return {
                "success": True,
                "property_name": property_name,
                "old_value": old_value,
                "new_value": value
            }
        
        @self.register_tool(
            name="interact",
            description="Interact with this object in a generic way",
            parameter_descriptions={
                "interaction_type": "Type of interaction",
                "parameters": "Parameters for the interaction"
            }
        )
        def interact_tool(interaction_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """
            Tool to interact with this object in a generic way.
            
            Args:
                interaction_type: Type of interaction
                parameters: Parameters for the interaction
                
            Returns:
                Result of the interaction
            """
            # Call the interaction handler (to be implemented by subclasses)
            result = self._handle_interaction(interaction_type, parameters or {})
            
            # Record the interaction event
            event_data = {
                "event_type": "object_interacted",
                "object_id": self.id,
                "interaction_type": interaction_type,
                "parameters": parameters or {},
                "result": result,
                "timestamp": int(time.time() * 1000)
            }
            
            # Get primary timeline
            timeline_id = self._timeline_state["primary_timeline"]
            if timeline_id is not None:
                self.update_state(event_data, {"timeline_id": timeline_id})
            
            logger.info(f"Interaction {interaction_type} with object {self.id}")
            
            # Notify observers
            self.notify_observers({
                "type": "object_interacted",
                "object_id": self.id,
                "interaction_type": interaction_type,
                "parameters": parameters or {},
                "result": result
            })
            
            return result
    
    def get_interior_state(self) -> Dict[str, Any]:
        """
        Get the interior state of the object.
        
        Returns:
            Dictionary representation of the interior state
        """
        # Get basic interior state from BaseElement
        interior = super().get_interior_state()
        
        # Add object-specific state
        interior.update({
            "properties": self._properties
        })
        
        return interior
    
    def get_exterior_state(self) -> Optional[Dict[str, Any]]:
        """
        Get the exterior state of the object.
        
        Returns:
            Dictionary representation of the exterior state, or None
        """
        if not self.HAS_EXTERIOR:
            return None
            
        # Get basic exterior state from BaseElement
        exterior = super().get_exterior_state()
        
        # Add object-specific state (simplified version of properties)
        # Typically only includes essential information for the closed state
        exterior.update({
            "property_count": len(self._properties),
            "has_properties": len(self._properties) > 0
        })
        
        return exterior
    
    def set_property(self, property_name: str, value: Any) -> None:
        """
        Set a property of this object.
        
        Args:
            property_name: Name of the property to set
            value: Value to set the property to
        """
        # Set the property
        old_value = self._properties.get(property_name)
        self._properties[property_name] = value
        
        # Record the property change event
        event_data = {
            "event_type": "object_property_changed",
            "object_id": self.id,
            "property_name": property_name,
            "old_value": old_value,
            "new_value": value,
            "timestamp": int(time.time() * 1000)
        }
        
        # Get primary timeline
        timeline_id = self._timeline_state["primary_timeline"]
        if timeline_id is not None:
            self.update_state(event_data, {"timeline_id": timeline_id})
        
        logger.info(f"Set property {property_name} of object {self.id} to {value}")
        
        # Notify observers
        self.notify_observers({
            "type": "property_changed",
            "object_id": self.id,
            "property_name": property_name,
            "old_value": old_value,
            "new_value": value
        })
    
    def get_property(self, property_name: str, default: Any = None) -> Any:
        """
        Get a property of this object.
        
        Args:
            property_name: Name of the property to get
            default: Default value to return if property not found
            
        Returns:
            Value of the property if found, default otherwise
        """
        return self._properties.get(property_name, default)
    
    def _handle_interaction(self, interaction_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an interaction with this object.
        
        This is a placeholder method that should be overridden by concrete subclasses
        to provide specialized interaction handling.
        
        Args:
            interaction_type: Type of interaction
            parameters: Parameters for the interaction
            
        Returns:
            Result of the interaction
        """
        # This method should be overridden by concrete subclasses
        logger.warning(f"_handle_interaction called on base Object class with type {interaction_type}")
        return {
            "success": False,
            "error": f"Interaction type {interaction_type} not supported by this object",
            "object_id": self.id,
            "object_type": self.__class__.__name__
        }
    
    def mount_element(self, element: 'BaseElement', mount_id: Optional[str] = None, 
                     mount_type: MountType = MountType.INCLUSION) -> bool:
        """
        Objects cannot mount elements (override to prevent mounting).
        
        Args:
            element: Element to mount
            mount_id: Optional identifier for the mount point
            mount_type: Type of mounting
            
        Returns:
            Always False for Objects
        """
        logger.error(f"Cannot mount elements in object {self.id} - Objects cannot contain other elements")
        return False 