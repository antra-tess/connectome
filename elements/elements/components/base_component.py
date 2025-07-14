"""
Base Component
Base class for all components in the Component-based architecture.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Type
import uuid
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Component:
    """
    Base class for all components in the Component-based architecture.
    
    Components provide specific behaviors/functionality to Elements.
    Each Component is attached to exactly one Element and cannot exist independently.
    
    Components:
    - Store state that was previously directly in Elements
    - Implement behavior that was previously in Element methods
    - Have a clear lifecycle tied to their owning Element
    - Can communicate with other Components through their Element
    """
    
    # Define which event types this component can handle
    HANDLED_EVENT_TYPES: List[str] = []
    
    # Component unique type identifier
    COMPONENT_TYPE: str = "base_component"
    
    # Define dependencies - other component types that must be present
    DEPENDENCIES: List[str] = []
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the component.
        
        The 'owner' attribute (referencing the Element this component is attached to)
        is expected to be set by the owning Element after construction.
        """
        self.id = f"{self.COMPONENT_TYPE}_{uuid.uuid4().hex[:8]}"
        self._state = {}
        
        # Initialize lifecycle state
        self._is_initialized = False
        self._is_enabled = True
        
        logger.debug(f"Created component: {self.COMPONENT_TYPE} ({self.id})")

    def get_sibling_component(self, component_type) -> Optional['Component']:
        return self.owner.get_component_by_type(component_type)
    
    def initialize(self) -> bool:
        """
        Initialize the component after it has been attached to an element.
        
        This is called when the component is first attached to an element.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if self._is_initialized:
            return True
            
        # Perform any required initialization
        initialization_result = self._on_initialize()
        
        if initialization_result:
            self._is_initialized = True
            logger.debug(f"Initialized component: {self.COMPONENT_TYPE} ({self.id})")
        
        return initialization_result
    
    def _on_initialize(self) -> bool:
        """
        Custom initialization logic.
        
        Override this method in subclasses to provide custom initialization.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        return True
    
    def enable(self) -> bool:
        """
        Enable the component.
        
        Returns:
            True if the component was enabled, False otherwise
        """
        if not self._is_initialized:
            logger.warning(f"Cannot enable uninitialized component: {self.COMPONENT_TYPE} ({self.id})")
            return False
            
        if self._is_enabled:
            return True
            
        enable_result = self._on_enable()
        
        if enable_result:
            self._is_enabled = True
            logger.debug(f"Enabled component: {self.COMPONENT_TYPE} ({self.id})")
        
        return enable_result
    
    def _on_enable(self) -> bool:
        """
        Custom enable logic.
        
        Override this method in subclasses to provide custom enable behavior.
        
        Returns:
            True if enabling was successful, False otherwise
        """
        return True
    
    def disable(self) -> bool:
        """
        Disable the component.
        
        Returns:
            True if the component was disabled, False otherwise
        """
        if not self._is_enabled:
            return True
            
        disable_result = self._on_disable()
        
        if disable_result:
            self._is_enabled = False
            logger.debug(f"Disabled component: {self.COMPONENT_TYPE} ({self.id})")
        
        return disable_result
    
    def _on_disable(self) -> bool:
        """
        Custom disable logic.
        
        Override this method in subclasses to provide custom disable behavior.
        
        Returns:
            True if disabling was successful, False otherwise
        """
        return True
    
    def cleanup(self) -> bool:
        """
        Clean up the component when it is removed from its element.
        
        Returns:
            True if cleanup was successful, False otherwise
        """
        cleanup_result = self._on_cleanup()
        
        if cleanup_result:
            logger.debug(f"Cleaned up component: {self.COMPONENT_TYPE} ({self.id})")
        
        return cleanup_result
    
    def _on_cleanup(self) -> bool:
        """
        Custom cleanup logic.
        
        Override this method in subclasses to provide custom cleanup behavior.
        
        Returns:
            True if cleanup was successful, False otherwise
        """
        return True
    
    def is_enabled(self) -> bool:
        """
        Check if the component is enabled.
        
        Returns:
            True if the component is enabled, False otherwise
        """
        return self._is_enabled
    
    def is_initialized(self) -> bool:
        """
        Check if the component is initialized.
        
        Returns:
            True if the component is initialized, False otherwise
        """
        return self._is_initialized
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the component's state.
        
        Returns:
            Dictionary representation of the component's state
        """
        return self._state.copy()
    
    def update_state(self, state_update: Dict[str, Any]) -> bool:
        """
        Update the component's state.
        
        Args:
            state_update: Dictionary containing state updates
            
        Returns:
            True if the state was updated, False otherwise
        """
        if not self._is_initialized or not self._is_enabled:
            return False
            
        # Apply the update
        self._state.update(state_update)
        
        # Call update handler
        self._on_state_updated(state_update)
        
        return True
    
    def _on_state_updated(self, state_update: Dict[str, Any]) -> None:
        """
        Custom state update logic.
        
        Override this method in subclasses to react to state updates.
        
        Args:
            state_update: Dictionary containing state updates
        """
        pass
    
    def handle_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handle an event.
        
        Args:
            event: Event data
            timeline_context: Timeline context for this event
            
        Returns:
            True if the event was handled, False otherwise
        """
        if not self._is_initialized or not self._is_enabled:
            return False
            
        # Check if this component handles this event type
        event_type = event.get("event_type")
        if event_type not in self.HANDLED_EVENT_TYPES:
            return False
            
        # Call the event handler
        return self._on_event(event, timeline_context)
    
    def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Custom event handling logic.
        
        Override this method in subclasses to handle events.
        
        Args:
            event: Event data
            timeline_context: Timeline context for this event
            
        Returns:
            True if the event was handled, False otherwise
        """
        return False
    
    def validate_dependencies(self, components: Dict[str, 'Component']) -> bool:
        """
        Validate that all required dependencies are present.
        
        Args:
            components: Dictionary of components attached to the same element
            
        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        for dependency in self.DEPENDENCIES:
            if not any(comp.COMPONENT_TYPE == dependency for comp in components.values()):
                logger.error(f"Missing dependency {dependency} for component {self.COMPONENT_TYPE}")
                return False
                
        return True 
    
class VeilProducer(Component):
    """
    Base class for components that produce VEIL representations.
    """
    COMPONENT_TYPE = "VeilProducer"
    
    def emit_delta(self) -> None:
        """
        Emits the calculated delta to the owning element's timeline.
        """
        delta_operations = self.calculate_delta()
        if delta_operations:
            self.owner.receive_delta(delta_operations)

    def calculate_delta(self) -> Optional[List]:
        """
        Base implementation for calculate_delta.
        Subclasses should override this to provide specific delta calculation logic.
        This method should also be responsible for updating the producer's internal state 
        (e.g., _last_ids, _has_produced_root_add_before) after determining the deltas.
        
        NEW: Return type is now VEILFacetOperation only.
        All legacy delta operation support has been removed for clean VEILFacet architecture.
        
        Returns:
            List[VEILFacetOperation] instances for the VEILFacet system
        """
        logger.warning(f"[{self.owner.id if self.owner else 'Unknown'}/{self.COMPONENT_TYPE}] calculate_delta() not implemented.")
        return None
    
    def _add_owner_tracking(self, veil_node: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add owner_id tracking to a VEIL node for efficient filtering.
        
        This enables granular access to VEIL nodes without full tree reconstruction
        by allowing direct filtering of the Space's flat cache by owner_id.
        
        Args:
            veil_node: VEIL node dictionary to add tracking to
            
        Returns:
            The same VEIL node with owner_id added to properties
        """
        if not self.owner:
            logger.warning(f"[{self.COMPONENT_TYPE}] Cannot add owner tracking: no owner set")
            return veil_node
        
        # Ensure properties exist
        if "properties" not in veil_node:
            veil_node["properties"] = {}
        
        # Add owner tracking for granular filtering
        veil_node["properties"]["owner_id"] = self.owner.id
        veil_node["properties"]["producer_type"] = self.COMPONENT_TYPE
        
        return veil_node
    
    def _add_owner_tracking_to_delta_ops(self, delta_operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add owner_id tracking to all nodes in delta operations.
        
        Args:
            delta_operations: List of delta operations to process
            
        Returns:
            The same delta operations with owner_id added to all nodes
        """
        if not delta_operations:
            return delta_operations
        
        for delta_op in delta_operations:
            if delta_op.get("op") == "add_node" and "node" in delta_op:
                self._add_owner_tracking(delta_op["node"])
            elif delta_op.get("op") == "update_node" and "properties" in delta_op:
                # For update operations, also add owner tracking to the properties being updated
                if not self.owner:
                    continue
                delta_op["properties"]["owner_id"] = self.owner.id
                delta_op["properties"]["producer_type"] = self.COMPONENT_TYPE
        
        return delta_operations
