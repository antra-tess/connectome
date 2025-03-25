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
        
        # Track elements requesting attention
        self._elements_requesting_attention: Dict[str, Dict[str, Any]] = {}
        
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
    
    def element_observer_callback(self, update_data: Dict[str, Any]) -> None:
        """
        Handle element state change notifications.
        
        This method processes notifications from elements mounted in this space
        and determines whether Shell attention is needed.
        
        Args:
            update_data: Data about the state change
        """
        element_id = update_data.get("element_id")
        if not element_id or element_id not in self._mounted_elements:
            return
            
        event_type = update_data.get("type")
        
        # Get the timeline context
        timeline_context = update_data.get("timeline_context")
        
        # Handle attention state in element updates
        if event_type == "element_state_updated":
            attention_state = update_data.get("attention_state", {})
            if attention_state.get("needs_attention", False):
                self._handle_element_attention_need(element_id, update_data, timeline_context)
            elif element_id in self._elements_requesting_attention:
                # Element no longer needs attention
                self._handle_element_attention_cleared(element_id, timeline_context)
    
    def _handle_element_attention_need(self, element_id: str, update_data: Dict[str, Any], 
                                     timeline_context: Optional[Dict[str, Any]] = None) -> None:
        """
        Process an element's notification that it needs attention.
        
        This method evaluates whether the element's need should be escalated
        to a formal attention request to the Shell.
        
        Args:
            element_id: ID of the element needing attention
            update_data: Data about the attention need
            timeline_context: Timeline context for this notification
        """
        # Validate the timeline context
        validated_timeline_context = self._validate_timeline_context(timeline_context)
        
        # Store the attention request
        self._elements_requesting_attention[element_id] = {
            "element": self._mounted_elements[element_id]["element"],
            "timestamp": update_data.get("timestamp", int(time.time() * 1000)),
            "data": update_data.get("attention_state", {}).get("request_data", {}),
            "timeline_context": validated_timeline_context
        }
        
        logger.info(f"Element {element_id} needs attention in space {self.id} timeline {validated_timeline_context.get('timeline_id', 'unknown')}")
        
        # Propagate the attention request to the registry if applicable
        if self._registry:
            # This is the official attention request to the Shell via the registry
            self._registry._notify_observers("attention_requested", {
                "space_id": self.id,
                "element_id": element_id,
                "source_element_id": element_id,
                "request_data": update_data.get("attention_state", {}).get("request_data", {}),
                "timeline_context": validated_timeline_context
            })
    
    def _handle_element_attention_cleared(self, element_id: str, timeline_context: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle when an element no longer needs attention.
        
        Args:
            element_id: ID of the element clearing its attention need
            timeline_context: Timeline context for this notification
        """
        # Validate the timeline context
        validated_timeline_context = self._validate_timeline_context(timeline_context)
        
        if element_id in self._elements_requesting_attention:
            del self._elements_requesting_attention[element_id]
            
            # Notify registry that this element no longer needs attention
            if self._registry:
                self._registry._notify_observers("attention_cleared", {
                    "space_id": self.id,
                    "element_id": element_id,
                    "timeline_context": validated_timeline_context
                })
            
            logger.info(f"Element {element_id} attention cleared in space {self.id} timeline {validated_timeline_context.get('timeline_id', 'unknown')}")
    
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
        
    def receive_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Receive an event from the Activity Layer or SpaceRegistry.
        
        This method routes events to the Space and its contained elements,
        ensuring that all events pass through the Space layer.
        
        Args:
            event: Event data to process
            timeline_context: Timeline context for this event
            
        Returns:
            True if the event was successfully processed, False otherwise
        """
        # Validate timeline context
        validated_timeline_context = self._validate_timeline_context(timeline_context)
        
        # Verify timeline coherence
        if not self._verify_timeline_coherence(validated_timeline_context):
            logger.warning(f"Space {self.id}: Timeline coherence check failed for {validated_timeline_context.get('timeline_id')}")
            return False
            
        # Check if this is a primary timeline
        if not validated_timeline_context.get("is_primary", False):
            # For non-primary timelines, check if we can interact with this timeline
            if not self._can_interact_with_timeline(validated_timeline_context):
                logger.warning(f"Space {self.id}: Cannot interact with non-primary timeline: {validated_timeline_context.get('timeline_id')}")
                return False
        
        # Record the event in the timeline DAG
        event_id = self._record_event_in_timeline(event, validated_timeline_context)
        if event_id:
            logger.debug(f"Space {self.id}: Recorded event {event_id} in timeline {validated_timeline_context.get('timeline_id')}")
        
        # Update our state with this event
        success = self.update_state(event, validated_timeline_context)
        if not success:
            logger.warning(f"Space {self.id}: Failed to update space state for event: {event.get('event_type')}")
            return False
        
        # Check for targeted routing
        target_element_id = event.get("target_element_id")
        if target_element_id:
            # If this is a targeted event, only route to that element
            target_element = self.get_mounted_element(target_element_id)
            if target_element:
                event_type = event.get("event_type")
                if hasattr(target_element, "EVENT_TYPES") and (event_type in target_element.EVENT_TYPES or "*" in target_element.EVENT_TYPES):
                    target_element.update_state(event, validated_timeline_context)
                    logger.debug(f"Space {self.id}: Routed event {event_type} to targeted element {target_element_id}")
                else:
                    logger.warning(f"Space {self.id}: Target element {target_element_id} does not handle event type {event_type}")
            else:
                logger.warning(f"Space {self.id}: Target element {target_element_id} not found")
        else:
            # Otherwise, pass to any elements that can handle this event type
            event_type = event.get("event_type")
            for mount_id, mount_info in self._mounted_elements.items():
                element = mount_info["element"]
                if hasattr(element, "EVENT_TYPES") and (event_type in element.EVENT_TYPES or "*" in element.EVENT_TYPES):
                    element.update_state(event, validated_timeline_context)
                    logger.debug(f"Space {self.id}: Routed event {event_type} to element {element.id}")
        
        logger.debug(f"Space {self.id}: Processed event {event.get('event_type')} in timeline {validated_timeline_context.get('timeline_id')}")
        return True
        
    def _record_event_in_timeline(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> Optional[str]:
        """
        Record an event in the timeline DAG.
        
        This ensures that all events that pass through a Space are properly
        recorded in the timeline.
        
        Args:
            event: Event data to record
            timeline_context: Timeline context for this event
            
        Returns:
            Event ID if successfully recorded, None otherwise
        """
        # Get timeline ID
        timeline_id = timeline_context.get("timeline_id")
        if not timeline_id:
            logger.warning(f"Space {self.id}: Cannot record event in timeline: No timeline_id provided")
            return None
            
        # Ensure the event has an ID
        event_id = event.get("event_id")
        if not event_id:
            event_id = f"evt_{str(uuid.uuid4())[:8]}"
            event["event_id"] = event_id
            
        # Ensure the event has a timestamp
        if "timestamp" not in event:
            event["timestamp"] = int(time.time() * 1000)
            
        # Record the event in our timeline structure
        if "timelines" not in self._timeline_state:
            self._timeline_state["timelines"] = {}
            
        if timeline_id not in self._timeline_state["timelines"]:
            self._timeline_state["timelines"][timeline_id] = {
                "created_at": int(time.time() * 1000),
                "events": [],
                "event_data": {}
            }
            
        # Add to events list
        self._timeline_state["timelines"][timeline_id]["events"].append(event_id)
        
        # Store full event data
        self._timeline_state["timelines"][timeline_id]["event_data"][event_id] = event.copy()
        
        return event_id
    
    def _verify_timeline_coherence(self, timeline_context: Dict[str, Any]) -> bool:
        """
        Verify that the timeline context is coherent with the space's timeline state.
        
        This checks if the timeline exists and if the last_event_id is consistent.
        
        Args:
            timeline_context: Timeline context to verify
            
        Returns:
            True if the timeline context is coherent, False otherwise
        """
        timeline_id = timeline_context.get("timeline_id")
        if not timeline_id:
            logger.warning("Missing timeline_id in context")
            return False
            
        # Check if this timeline exists in our known timelines
        if timeline_id not in self._timeline_state["timelines"]:
            # New timeline - acceptable if it's being created
            if timeline_context.get("create_timeline", False):
                logger.info(f"Creating new timeline: {timeline_id}")
                self._timeline_state["timelines"][timeline_id] = {
                    "events": [],
                    "created_at": int(time.time() * 1000),
                    "parent_timeline": timeline_context.get("parent_timeline"),
                    "fork_point": timeline_context.get("fork_point")
                }
                return True
            else:
                logger.warning(f"Unknown timeline: {timeline_id}")
                return False
                
        # Check last_event_id if provided
        last_event_id = timeline_context.get("last_event_id")
        if last_event_id:
            timeline_events = self._timeline_state["timelines"][timeline_id]["events"]
            if timeline_events and timeline_events[-1] != last_event_id:
                logger.warning(f"Inconsistent last_event_id: {last_event_id} vs {timeline_events[-1]}")
                return False
                
        return True
    
    def _can_interact_with_timeline(self, timeline_context: Dict[str, Any]) -> bool:
        """
        Check if we can interact with a timeline.
        
        According to the architecture, Loom tree sections with forks but no primary 
        timeline cannot be interacted with. Agents that are causally entangled in 
        these regions remain unavailable until a primary timeline is established.
        
        Args:
            timeline_context: Timeline context to check
            
        Returns:
            True if we can interact with this timeline, False otherwise
        """
        timeline_id = timeline_context.get("timeline_id")
        if not timeline_id:
            return False
            
        # Always allow interaction with primary timeline
        if timeline_id == self._timeline_state["primary_timeline"]:
            return True
            
        # Check if this timeline has a parent with primary designation
        parent_timeline = self._get_parent_timeline(timeline_id)
        if not parent_timeline:
            # Orphaned timeline with no parent cannot be interacted with
            return False
            
        # Check if there's a path to a primary timeline
        while parent_timeline:
            if parent_timeline == self._timeline_state["primary_timeline"]:
                return True
            parent_timeline = self._get_parent_timeline(parent_timeline)
            
        # No path to primary timeline found
        return False
    
    def _get_parent_timeline(self, timeline_id: str) -> Optional[str]:
        """
        Get the parent timeline of a timeline.
        
        Args:
            timeline_id: ID of the timeline
            
        Returns:
            Parent timeline ID, or None if not found
        """
        if timeline_id in self._timeline_state["timelines"]:
            return self._timeline_state["timelines"][timeline_id].get("parent_timeline")
        return None
    
    def get_elements_requesting_attention(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all elements currently requesting attention in this space.
        
        Returns:
            Dictionary mapping element IDs to their attention request data
        """
        return self._elements_requesting_attention.copy()
    
    def mount_element(self, element: BaseElement, mount_id: Optional[str] = None, 
                     mount_type: MountType = MountType.INCLUSION) -> bool:
        """
        Mount an element in this space.
        
        Args:
            element: Element to mount
            mount_id: Optional identifier for the mount point
            mount_type: Type of mounting
            
        Returns:
            True if the element was mounted successfully, False otherwise
        """
        # Implementation of element mounting
        # ... [existing mount_element code] ...
        
        # Register as observer for the element's state changes
        element.register_observer(self.element_observer_callback)
        
        return True
    
    def unmount_element(self, mount_id: str) -> bool:
        """
        Unmount an element from this space.
        
        Args:
            mount_id: Identifier for the mount point
            
        Returns:
            True if the element was unmounted successfully, False otherwise
        """
        # Get the element before unmounting
        if mount_id in self._mounted_elements:
            element = self._mounted_elements[mount_id]["element"]
            
            # Unregister as observer
            element.unregister_observer(self.element_observer_callback)
            
            # Clear any attention requests for this element
            if mount_id in self._elements_requesting_attention:
                self._handle_element_attention_cleared(mount_id)
        
        # Rest of unmount implementation
        # ... [existing unmount_element code] ...
        
        return True
    
    def execute_action_on_element(self, element_id: str, action_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action on an element in this space.
        
        All agent actions on elements should go through this method to ensure
        they are properly recorded in the timeline DAG.
        
        Args:
            element_id: ID of the element to act on
            action_name: Name of the action to execute
            parameters: Parameters for the action (may include timeline_context)
            
        Returns:
            Result of the action execution
        """
        # Get the element
        element = self.get_mounted_element(element_id)
        if not element:
            return {"error": f"Element not found in space {self.id}: {element_id}"}
        
        # Extract timeline context if available
        timeline_context = parameters.get("timeline_context")
        
        # If we don't have a timeline context, use the primary timeline
        if not timeline_context:
            primary_timeline_id = self._timeline_state.get("primary_timeline")
            if primary_timeline_id:
                timeline_context = {
                    "timeline_id": primary_timeline_id,
                    "is_primary": True
                }
            else:
                logger.warning(f"Space {self.id}: No timeline context provided for action and no primary timeline available")
        
        # Now validate the timeline coherence
        if timeline_context:
            validated_timeline_context = self._validate_timeline_context(timeline_context)
            
            # Validate timeline coherence
            if not self._verify_timeline_coherence(validated_timeline_context):
                return {
                    "error": f"Timeline coherence check failed: {validated_timeline_context.get('timeline_id')}",
                    "element_id": element_id,
                    "action_name": action_name
                }
            
            # Check if we can interact with this timeline
            if not validated_timeline_context.get("is_primary", False) and not self._can_interact_with_timeline(validated_timeline_context):
                return {
                    "error": f"Cannot interact with non-primary timeline: {validated_timeline_context.get('timeline_id')}",
                    "element_id": element_id,
                    "action_name": action_name
                }
                
            # Record agent action as event in timeline DAG
            event_id = self._record_action_in_timeline(element_id, action_name, parameters, validated_timeline_context)
            logger.debug(f"Space {self.id}: Recorded action '{action_name}' on element {element_id} as event {event_id}")
        
        # Execute the action
        try:
            if hasattr(element, "execute_tool"):
                # Ensure we pass the validated timeline context in parameters
                if timeline_context:
                    action_parameters = parameters.copy()
                    action_parameters["timeline_context"] = validated_timeline_context
                else:
                    action_parameters = parameters
                    
                result = element.execute_tool(action_name, **action_parameters)
                
                # Enhance the result with execution metadata
                if isinstance(result, dict):
                    result["action_executed"] = {
                        "element_id": element_id,
                        "action_name": action_name,
                        "timestamp": int(time.time() * 1000)
                    }
                    if timeline_context:
                        result["timeline_id"] = validated_timeline_context.get("timeline_id")
                
                return result
            else:
                return {"error": f"Element {element_id} does not support actions"}
        except Exception as e:
            logger.error(f"Error executing action {action_name} on element {element_id}: {e}", exc_info=True)
            return {"error": str(e)}
            
    def _record_action_in_timeline(self, element_id: str, action_name: str, 
                                 parameters: Dict[str, Any], timeline_context: Dict[str, Any]) -> str:
        """
        Record an agent action in the timeline DAG.
        
        According to the architecture, agent actions should be recorded as events in the timeline.
        This ensures that every action taken by the agent is properly recorded in the timeline DAG.
        
        Args:
            element_id: ID of the element the action is performed on
            action_name: Name of the action
            parameters: Parameters for the action
            timeline_context: Timeline context for this action
            
        Returns:
            The ID of the created event
        """
        # Generate a unique event ID
        event_id = f"action_{str(uuid.uuid4())[:8]}"
        
        # Create an event for this action
        action_event = {
            "event_id": event_id,
            "event_type": "agent_action",
            "element_id": element_id,
            "action_name": action_name,
            "parameters": {k: v for k, v in parameters.items() if k != "timeline_context"},  # Don't include timeline_context
            "timestamp": int(time.time() * 1000),
            "agent_action": True
        }
        
        # Get timeline ID
        timeline_id = timeline_context.get("timeline_id")
        if not timeline_id:
            logger.warning(f"Space {self.id}: Cannot record action in timeline: No timeline_id provided")
            return event_id
            
        # Ensure timeline structure exists
        if "timelines" not in self._timeline_state:
            self._timeline_state["timelines"] = {}
            
        if timeline_id not in self._timeline_state["timelines"]:
            self._timeline_state["timelines"][timeline_id] = {
                "created_at": int(time.time() * 1000),
                "events": [],
                "event_data": {}
            }
            
        # Add the event to the timeline
        self._timeline_state["timelines"][timeline_id]["events"].append(event_id)
            
        # Store the event data
        if "event_data" not in self._timeline_state["timelines"][timeline_id]:
            self._timeline_state["timelines"][timeline_id]["event_data"] = {}
        self._timeline_state["timelines"][timeline_id]["event_data"][event_id] = action_event
        
        # Also update our state to reflect this action
        self.update_state(action_event, timeline_context)
        
        logger.debug(f"Space {self.id}: Recorded agent action '{action_name}' on element {element_id} in timeline {timeline_id}")
        return event_id 