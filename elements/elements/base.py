"""
Base Element
Base class for all elements in the Component-based architecture.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Set, Tuple, Literal, Type, TYPE_CHECKING, Union
import uuid
import time
import enum
import inspect # For tool registration

from .components.base_component import Component

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Forward reference for type hint
if TYPE_CHECKING:
    from .space import Space


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
            descriptfion: Description of this element's purpose
        """
        self.id = element_id
        self.name = name
        self.description = description
        
        # Parent element (where this element is mounted)
        self._parent_element: Optional[Tuple[str, MountType]] = None
        
        # Registry reference (for sending messages to activity layer)
        self._registry = None
        
        # Components
        self._components: Dict[str, 'Component'] = {}
        
        logger.info(f"Created element: {name} ({element_id})")
        
        # Note: Components are typically added by subclasses or factories *after* base init.
        # Therefore, calling validation here might be too early. 
        # Let's add the validation method here, but the call should likely be moved
        # to the end of the subclass __init__ or the factory construction process.
        # self._validate_all_component_dependencies() # <<< Tentatively place call here, but see note.
    
    def get_registry(self):
        """
        Get the SpaceRegistry reference.
        
        Returns:
            The SpaceRegistry instance if set, None otherwise
        """
        from ..space_registry import SpaceRegistry
        return SpaceRegistry.get_instance()
    
    def add_component(self, component_type: Type['Component'], **kwargs) -> Optional['Component']:
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
            # Pass self (the element) to the component constructor
            # component = component_type(owner=self, **kwargs) # Old way
            component = component_type(**kwargs) # Instantiate without owner
            component.owner = self # Set owner attribute directly
        except Exception as e:
            logger.error(f"Error creating component of type {component_type.__name__} for {self.id}: {e}, kwargs: {kwargs}", exc_info=True)
            return None
        
        # Check if we already have a component of this type
        if self.get_component_by_type(component.COMPONENT_TYPE):
             logger.warning(f"Element {self.id} already has a component of type {component.COMPONENT_TYPE}. Cannot add duplicate.")
             return None
        
        # Initialize the component
        try:
            component.initialize()
        except BaseException as e:    
            logger.error(f"Failed to initialize component {component.COMPONENT_TYPE}: {e}", exc_info=True)
            return None
        
        # Add to components dictionary using component type as key for easier lookup?
        # Or use component.id? Let's stick with component.id for now as it seems to be the current pattern elsewhere.
        comp_id = component.id 
        self._components[comp_id] = component
        
        logger.debug(f"Added component {component.COMPONENT_TYPE} ({comp_id}) to element {self.id}")
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
    
    def get_component(self, component_id: str) -> Optional['Component']:
        """
        Get a component by ID.
        
        Args:
            component_id: ID of the component to get
            
        Returns:
            The component, or None if not found
        """
        return self._components.get(component_id)
    
    def get_component_by_type(self, component_type: Union[str, Type[Component]]) -> Optional[Component]:
        """Get a component by its type (class or string identifier)."""
        target_type_name = component_type if isinstance(component_type, str) else component_type.COMPONENT_TYPE
        return_comp = None
        for comp in self._components.values():
            # Check class hierarchy for matches if component_type is a class
            if not isinstance(component_type, str) and isinstance(comp, component_type):
                return_comp = comp
            # Fallback to checking COMPONENT_TYPE string
            if comp.COMPONENT_TYPE == target_type_name:
                return_comp = comp
        return return_comp

    def get_components_by_type(self, component_type: Union[str, Type[Component]]) -> List[Component]:
        """Get all components matching a given type (class or string identifier)."""
        matches = []
        target_type_name = component_type if isinstance(component_type, str) else component_type.COMPONENT_TYPE
        for comp in self._components.values():
            if not isinstance(component_type, str) and isinstance(comp, component_type):
                matches.append(comp)
            elif hasattr(comp, 'COMPONENT_TYPE') and comp.COMPONENT_TYPE == target_type_name:
                matches.append(comp)
        return matches
    
    def get_components(self) -> Dict[str, 'Component']:
        """
        Get all components.
        
        Returns:
            Dictionary of component ID to component instance.
        """
        return self._components
    
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

    def get_parent_object(self) -> Optional['BaseElement']: # Or Optional['Space'] might be more accurate if parent is always Space
        """
        Retrieves the actual parent object of this element using the stored parent_id and the SpaceRegistry.
        
        Returns:
            The parent BaseElement (likely a Space instance) or None if not found or if registry is unavailable.
        """
        if not self._parent_element:
            logger.debug(f"[{self.id}] No parent information tuple (_parent_element) available.")
            return None
        
        parent_id, _ = self._parent_element # mount_type is not needed for lookup here
        
        if not parent_id:
            logger.debug(f"[{self.id}] Parent ID is not set in _parent_element tuple.")
            return None

        registry = self.get_registry()
        if not registry:
            logger.warning(f"[{self.id}] Cannot get parent object: SpaceRegistry not set on this element.")
            return None

        # Attempt to get the parent as a Space first, as this is the common case for context.
        # SpaceRegistry.get_space() returns Optional[Space], and Space is a BaseElement.
        parent_obj = registry.get_space(parent_id)
        if parent_obj:
            return parent_obj
        
        # Optional: Fallback if the parent might be a non-Space element also registered.
        # For messaging context, parent is almost always a Space.
        # if hasattr(registry, 'get_element'):
        #     parent_obj = registry.get_element(parent_id)
        #     if parent_obj:
        #         logger.debug(f"[{self.id}] Found parent object with ID '{parent_id}' via registry's get_element method.")
        #         return parent_obj

        logger.warning(f"[{self.id}] Could not find parent object (Space) with ID '{parent_id}' via registry's get_space method.")
        return None
    
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

    def _validate_all_component_dependencies(self) -> bool:
        """
        Validates dependencies for ALL attached components after they have been added.

        Returns:
            True if all components satisfy their dependencies, False otherwise.
        """
        logger.debug(f"Validating dependencies for all components on element {self.id}...")
        all_valid = True
        component_map_by_type = {comp.COMPONENT_TYPE: comp for comp in self._components.values() if hasattr(comp, 'COMPONENT_TYPE')}
        
        for comp_id, component in self._components.items():
            if hasattr(component, 'DEPENDENCIES') and component.DEPENDENCIES:
                # Check if DEPENDENCIES is a set or list
                dependencies_list = component.DEPENDENCIES
                if isinstance(dependencies_list, set):
                     pass # Already a set
                elif isinstance(dependencies_list, list):
                     dependencies_list = set(dependencies_list) # Convert list to set for efficient lookup
                else:
                     logger.error(f"Component {comp_id} ({component.COMPONENT_TYPE}) has invalid DEPENDENCIES attribute (not a set or list). Skipping validation.")
                     all_valid = False
                     continue
                     
                for required_type in dependencies_list:
                    if required_type not in component_map_by_type:
                        logger.error(f"Dependency validation FAILED for {comp_id} ({component.COMPONENT_TYPE}): Missing required component type '{required_type}'.")
                        all_valid = False
                        # Optionally, raise an exception here to halt initialization immediately
                        # raise ValueError(f"Missing dependency '{required_type}' for component {component.COMPONENT_TYPE} on element {self.id}")
            # Check if the component has the validate_dependencies method (it should from Component base)
            # if hasattr(component, 'validate_dependencies') and callable(component.validate_dependencies):
            #     if not component.validate_dependencies(self._components):
            #         # Error already logged by component.validate_dependencies
            #         all_valid = False
            #         # Optionally raise an exception here
            #         # raise ValueError(f"Dependency validation failed for {component.COMPONENT_TYPE} on element {self.id}")
            # else:
            #     logger.warning(f"Component {comp_id} ({component.COMPONENT_TYPE}) missing validate_dependencies method.")

        if all_valid:
            logger.debug(f"All component dependencies successfully validated for element {self.id}.")
        else:
            logger.error(f"Dependency validation failed for one or more components on element {self.id}.")
            
        return all_valid 

    def finalize_setup(self) -> None:
        """
        Called after the element and all its initial components have been added and configured.
        Allows the element to perform any final initialization steps, such as registering
        tools from its components if it has a ToolProvider.
        """
        # Example: If this element has a ToolProvider, it might register tools from other components.
        # tool_provider = self.get_component_by_type("ToolProviderComponent")
        # if tool_provider and hasattr(tool_provider, 'discover_and_register_tools'):
        #     tool_provider.discover_and_register_tools(self) # Pass self to scan
        pass

    def _register_component_tools(self) -> None:
        """
        Iterates through attached components and registers methods 
        marked as tools (e.g., starting with 'handle_') with this 
        element's own ToolProviderComponent (if present).
        """
        # Find the ToolProviderComponent on this element
        from .components.tool_provider import ToolProviderComponent # Needed for registration target
        tool_provider = self.get_component_by_type(ToolProviderComponent)
        if not tool_provider:
            # No tool provider on this element, nothing to do.
            # Log this only if other components *have* handle_ methods?
            # For now, just return silently.
            return

        logger.debug(f"[{self.id}] Scanning attached components for local tools...")
        registered_count = 0
        skipped_count = 0

        for comp_id, component in self.get_components().items():
            # Skip the tool provider itself
            if component is tool_provider:
                continue
                
            # Inspect methods
            for attr_name in dir(component):
                if attr_name.startswith("handle_"):
                    try:
                        attr_value = getattr(component, attr_name)
                        if callable(attr_value) and not attr_name.startswith('_'):
                            tool_name = attr_name.replace("handle_", "")
                            docstring = inspect.getdoc(attr_value) or ""
                            description = docstring.split('\n')[0] if docstring else f"Tool '{tool_name}' from {component.COMPONENT_TYPE}"
                            param_descriptions = {} # Placeholder
                            
                            # Check if already registered *on this element's provider*
                            if tool_name in tool_provider.list_tools():
                                logger.debug(f"Tool '{tool_name}' already registered on {self.id}. Skipping registration from {component.COMPONENT_TYPE}.")
                                skipped_count += 1
                                continue
                                
                            tool_provider.register_tool_function(
                                name=tool_name,
                                description=description,
                                parameter_descriptions=param_descriptions,
                                tool_func=attr_value
                            )
                            registered_count += 1
                            logger.debug(f"Registered tool '{tool_name}' on {self.id} from {component.COMPONENT_TYPE}")
                    except Exception as e:
                        logger.error(f"Error inspecting/registering method {attr_name} from component {comp_id} on element {self.id}: {e}", exc_info=True)
        
        if registered_count > 0 or skipped_count > 0:
            logger.info(f"[{self.id}] Local tool registration complete. Registered: {registered_count}, Skipped: {skipped_count}.")

    def _on_cleanup(self) -> bool:
        """
        Called when the element is being destroyed or removed.
        """
        # Implement cleanup logic here
        return True

    def receive_event(self, event_payload: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """
        Receive an event and dispatch it to components.
        This is the standard entry point for an element to process an event.
        Spaces will override this to also record to their timeline first.
        
        Args:
            event_payload: The core data of the event.
            timeline_context: Timeline context (may be less relevant for non-Space elements
                              unless they need to act based on which timeline triggered this).
        """
        logger.debug(f"[{self.id}] BaseElement received event: Type='{event_payload.get('event_type')}', Target='{event_payload.get('target_element_id')}'")
        # Simply delegate to the existing handle_event which dispatches to components.
        # Components will receive this event_payload directly.
        self.handle_event(event_payload, timeline_context)

    def handle_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handle an event by delegating to components.
        
        Args:
            event: Event data (this is the event_payload received by receive_event)
            timeline_context: Timeline context for this event
            
        Returns:
            True if the event was handled by any component, False otherwise
        """
        handled = False
        for component_id, component in self._components.items(): # Iterate with ID for better logging
            if hasattr(component, 'handle_event') and callable(component.handle_event):
                try:
                    if component.handle_event(event, timeline_context):
                        logger.debug(f"[{self.id}] Event '{event.get('event_type')}' handled by component: {component_id} ({component.COMPONENT_TYPE if hasattr(component, 'COMPONENT_TYPE') else 'UnknownComponentType'})")
                        handled = True
                        # Allow multiple components to handle if needed, or break if exclusive
                except Exception as comp_err:
                     logger.error(f"[{self.id}] Error in component '{component_id} ({component.COMPONENT_TYPE if hasattr(component, 'COMPONENT_TYPE') else 'UnknownComponentType'})' handling event: {comp_err}", exc_info=True)
        return handled

    def _set_mount_id(self, mount_id: str) -> None:
        """
        Set the mount ID for this element.
        """
        self._mount_id = mount_id
        logger.debug(f"[{self.id}] Set mount ID to '{mount_id}'")

    @property
    def mount_id(self) -> str:
        """
        Get the mount ID for this element.
        """
        return self._mount_id

    def receive_delta(self, delta: List[Dict[str, Any]]) -> None:
        parent = self.get_parent_object()
        if parent and hasattr(parent, 'receive_delta'):
            parent.receive_delta(delta)
        else:
            logger.warning(f"[{self.id}] No owner or receive_delta method on owner for element {self.id}")
