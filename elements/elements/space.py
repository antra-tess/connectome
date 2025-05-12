"""
Space
Base class for space elements that can contain other elements.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Type, Set
import uuid
import time

from .base import BaseElement, MountType, Component, VeilProducer
from .components.space import ContainerComponent, TimelineComponent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Space(BaseElement):
    """
    Base class for space elements that can contain other elements.
    
    Spaces are structural containers that manage their own timeline (Loom DAG)
    and contained elements. Functionality is primarily delegated to components
    like ContainerComponent and TimelineComponent.
    """
    
    # Space is a container for other elements
    IS_SPACE = True
    
    def __init__(self, element_id: str, name: str, description: str, **kwargs):
        """
        Initialize the space.
        
        Args:
            element_id: Unique identifier for this space
            name: Human-readable name for this space
            description: Description of this space's purpose
            **kwargs: Passthrough for BaseElement
        """
        super().__init__(element_id=element_id, name=name, description=description, **kwargs)
        
        # Initialize space with required components
        # Store references for easier delegation, though get_component could also be used.
        self._container: Optional[ContainerComponent] = self.add_component(ContainerComponent)
        self._timeline: Optional[TimelineComponent] = self.add_component(TimelineComponent)
        
        # NEW: Initialize uplink listeners set and delta cache
        self._uplink_listeners: Set[Callable[[List[Dict[str, Any]]], None]] = set()
        self._cached_deltas: List[Dict[str, Any]] = []
        
        if not self._container or not self._timeline:
            # BaseElement.add_component logs errors, but we might want to raise here
            logger.critical(f"CRITICAL: Failed to initialize required components (Container/Timeline) for space {self.id}. Space may be unstable.")
            # raise RuntimeError(f"Failed to initialize required components for space {self.id}")
        
        logger.info(f"Created space: {name} ({self.id})")
    
    # --- Container Methods (Delegated) ---
    def mount_element(self, element: BaseElement, mount_id: Optional[str] = None, mount_type: MountType = MountType.INCLUSION) -> bool:
        """Mount an element in this space (delegates to ContainerComponent)."""
        if not self._container:
            logger.error(f"[{self.id}] Cannot mount element: ContainerComponent unavailable.")
            return False
        return self._container.mount_element(element, mount_id, mount_type)
    
    def unmount_element(self, mount_id: str) -> bool:
        """Unmount an element (delegates to ContainerComponent)."""
        if not self._container:
            logger.error(f"[{self.id}] Cannot unmount element: ContainerComponent unavailable.")
            return False
        return self._container.unmount_element(mount_id)
    
    def get_mounted_element(self, mount_id: str) -> Optional[BaseElement]:
        """Get a mounted element by ID (delegates to ContainerComponent)."""
        if not self._container:
            logger.error(f"[{self.id}] Cannot get mounted element: ContainerComponent unavailable.")
        return None
        return self._container.get_mounted_element(mount_id)
    
    def get_mounted_elements(self) -> Dict[str, BaseElement]:
        """Get all mounted elements (delegates to ContainerComponent)."""
        if not self._container:
            logger.error(f"[{self.id}] Cannot get mounted elements: ContainerComponent unavailable.")
            return {}
        return self._container.get_mounted_elements()
        
    def get_mounted_elements_info(self) -> Dict[str, Dict[str, Any]]:
         """Get metadata for mounted elements (delegates to ContainerComponent)."""
         if not self._container:
             logger.error(f"[{self.id}] Cannot get mounted elements info: ContainerComponent unavailable.")
             return {}
         return self._container.get_mounted_elements_info()

    # --- Timeline Methods (Delegated) ---
    def add_event_to_timeline(self, event_payload: Dict[str, Any], timeline_context: Dict[str, Any]) -> Optional[str]:
        """Add an event to a timeline (delegates to TimelineComponent)."""
        if not self._timeline:
            logger.error(f"[{self.id}] Cannot add event to timeline: TimelineComponent unavailable.")
            return None
        return self._timeline.add_event_to_timeline(event_payload, timeline_context)
    
    def get_primary_timeline(self) -> Optional[str]:
        """Get the ID of the primary timeline (delegates to TimelineComponent)."""
        if not self._timeline:
            logger.error(f"[{self.id}] Cannot get primary timeline: TimelineComponent unavailable.")
            return None
        return self._timeline.get_primary_timeline()
    
    def designate_primary_timeline(self, timeline_id: str) -> bool:
        """Designate a timeline as the primary timeline (delegates to TimelineComponent)."""
        if not self._timeline:
            logger.error(f"[{self.id}] Cannot designate primary timeline: TimelineComponent unavailable.")
            return False
        return self._timeline.designate_primary_timeline(timeline_id)
        
    def get_timeline_events(self, timeline_id: str, start_event_id: Optional[str] = None, limit: int = 0) -> List[Dict[str, Any]]:
        """Get events in a timeline (delegates to TimelineComponent)."""
        if not self._timeline:
            logger.error(f"[{self.id}] Cannot get timeline events: TimelineComponent unavailable.")
            return []
        return self._timeline.get_timeline_events(timeline_id, start_event_id, limit)
        
    # --- Core Event Processing --- 
    def receive_event(self, event_payload: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """
        Receive an event, record it in the timeline, and dispatch it to components
        and potentially mounted child elements.
        
        Args:
            event_payload: The core data of the event (e.g., {'event_type': ..., 'target_element_id': ..., 'payload': ...}).
            timeline_context: Timeline context information (e.g., {'timeline_id': ...}).
        """
        event_type = event_payload.get("event_type")
        target_element_id = event_payload.get("target_element_id")
        logger.debug(f"[{self.id}] Receiving event: Type='{event_type}', Target='{target_element_id}', Timeline='{timeline_context.get('timeline_id')}'")

        # 1. Add event to the timeline via TimelineComponent
        new_event_id = self.add_event_to_timeline(event_payload, timeline_context)
        if not new_event_id:
            # Error already logged by add_event_to_timeline
            logger.error(f"[{self.id}] Failed to add event to timeline. Aborting processing for this event.")
            return
            
        # Construct the full event node as it exists in the DAG for handlers
        # (Alternatively, components could query the timeline if they need the full node)
        full_event_node = { 
             'id': new_event_id,
             'timeline_id': timeline_context.get('timeline_id', self.get_primary_timeline()),
             # We don't easily know parent_ids/timestamp here without querying timeline again,
             # so components should rely on the payload for now, or query if needed.
             'payload': event_payload 
        }

        # 2. Dispatch event to *own* components
        # Iterate through all components attached to this Space element
        handled_by_component = False
        for comp_name, component in self.get_components().items():
            if hasattr(component, 'handle_event') and callable(component.handle_event):
                try:
                    # Pass the full event node as recorded (or just payload if preferred)
                    # Passing full_event_node gives components access to event ID etc.
                    if component.handle_event(full_event_node, timeline_context):
                         logger.debug(f"[{self.id}] Event '{new_event_id}' ({event_type}) handled by component: {comp_name}")
                         handled_by_component = True
                         # Allow multiple components to handle the same event
                except Exception as comp_err:
                     logger.error(f"[{self.id}] Error in component '{comp_name}' handling event '{new_event_id}': {comp_err}", exc_info=True)
        
        # 3. Dispatch event to targeted *child* element, if specified
        if target_element_id:
            mounted_element = self.get_mounted_element(target_element_id)
            if mounted_element:
                 if hasattr(mounted_element, 'receive_event') and callable(mounted_element.receive_event):
                      logger.debug(f"[{self.id}] Routing event '{new_event_id}' to mounted element: {target_element_id}")
                      try:
                           # Child element receives the same event payload and timeline context
                           mounted_element.receive_event(event_payload, timeline_context) 
                      except Exception as mounted_err:
                           logger.error(f"[{self.id}] Error in mounted element '{target_element_id}' receiving event '{new_event_id}': {mounted_err}", exc_info=True)
                 else:
                      logger.warning(f"[{self.id}] Mounted element '{target_element_id}' found but has no receive_event method.")
            else:
                 # This could happen if the event targets an element nested deeper
                 # We rely on parent elements routing downwards. If it wasn't found here,
                 # it means the target wasn't a direct child.
                 logger.debug(f"[{self.id}] Event '{new_event_id}' targets element '{target_element_id}', which is not directly mounted here.")
        else:
             # Event was not targeted at a specific child element
             logger.debug(f"[{self.id}] Event '{new_event_id}' ({event_type}) processed by Space components.")

    # NEW methods for listener registration
    def register_uplink_listener(self, callback: Callable[[List[Dict[str, Any]]], None]):
        """Registers a callback function to be notified when new VEIL deltas are generated."""
        if callable(callback):
            self._uplink_listeners.add(callback)
            logger.debug(f"[{self.id}] Registered uplink listener: {callback}")
        else:
            logger.warning(f"[{self.id}] Attempted to register non-callable uplink listener: {callback}")

    def unregister_uplink_listener(self, callback: Callable[[List[Dict[str, Any]]], None]):
        """Unregisters a previously registered uplink listener callback."""
        self._uplink_listeners.discard(callback)
        logger.debug(f"[{self.id}] Unregistered uplink listener: {callback}")

    # NEW: Frame end processing method
    def on_frame_end(self):
        """
        Processes the end of a frame: calculates VEIL deltas for all child elements
        with VeilProducers, caches them, and notifies listeners.
        """
        logger.debug(f"[{self.id}] Starting on_frame_end processing...")
        all_deltas_in_frame: List[Dict[str, Any]] = []
        processed_elements = set() # Track elements to avoid duplicate processing if mounted multiple times

        mounted_elements = self.get_mounted_elements() # Use the delegation method

        for mount_id, element in mounted_elements.items():
            if element.id in processed_elements:
                continue # Skip if already processed via another mount point
            processed_elements.add(element.id)

            # Find VeilProducer components on this element
            veil_producers = element.get_components_by_type(VeilProducer) # Get all instances

            if not veil_producers:
                 # This debug message might be too verbose if many elements don't have VeilProducers
                 # logger.debug(f"[{self.id}] Element {element.name} ({element.id}) has no VeilProducer components.")
                 continue

            for producer_component in veil_producers:
                if hasattr(producer_component, 'calculate_delta') and callable(producer_component.calculate_delta):
                    try:
                        delta = producer_component.calculate_delta()
                        if delta: # Only collect non-empty deltas
                            if isinstance(delta, list):
                                all_deltas_in_frame.extend(delta)
                                logger.debug(f"[{self.id}] Collected {len(delta)} delta operations from {producer_component.__class__.__name__} on element {element.id}")
                            else:
                                 logger.warning(f"[{self.id}] VeilProducer {producer_component.__class__.__name__} on element {element.id} returned non-list delta: {type(delta)}. Skipping.")
                    except Exception as e:
                        logger.error(f"[{self.id}] Error calling calculate_delta on {producer_component.__class__.__name__} for element {element.id}: {e}", exc_info=True)
                else:
                     logger.warning(f"[{self.id}] Found VeilProducer component {producer_component.__class__.__name__} on {element.id} without a callable calculate_delta method.")


        # Cache the collected deltas for this frame
        # TODO: Decide on caching strategy - replace, append, timestamp? Replacing for now.
        self._cached_deltas = all_deltas_in_frame
        logger.debug(f"[{self.id}] Total deltas collected in frame: {len(self._cached_deltas)}")

        # Notify listeners if new deltas were generated
        if self._cached_deltas and self._uplink_listeners:
            logger.info(f"[{self.id}] Notifying {len(self._uplink_listeners)} uplink listeners of new VEIL deltas.")
            # Pass a copy of the deltas to listeners
            deltas_to_send = list(self._cached_deltas)
            for listener in list(self._uplink_listeners): # Iterate over a copy in case listener modifies the set
                try:
                    # Consider if listeners should PULL the data instead?
                    # Passing deltas directly for now.
                    listener(deltas_to_send)
                except Exception as e:
                    logger.error(f"[{self.id}] Error notifying uplink listener {listener}: {e}", exc_info=True)
                    # Optionally unregister faulty listeners?
                    # self.unregister_uplink_listener(listener)

        logger.debug(f"[{self.id}] Finished on_frame_end processing.")

    # Optional: Method for listeners to pull cached deltas if needed
    def get_cached_deltas(self) -> List[Dict[str, Any]]:
        """Returns the VEIL deltas calculated in the last frame."""
        # Consider clearing cache after retrieval? Or timestamping?
        return list(self._cached_deltas) # Return a copy

    # --- Action Execution --- 
    def execute_action_on_element(self, element_id: str, action_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action on an element in this space.
        Finds the element (either self or a mounted child) and delegates to its ToolProviderComponent.
        """
        target_element: Optional[BaseElement] = None

        # Check if this is an action on this space itself
        if element_id == self.id:
            target_element = self
        else:
            # Otherwise, look for the element in mounted elements (using ContainerComponent)
            target_element = self.get_mounted_element(element_id)
            
        if not target_element:
            # If not found by mount_id, maybe element_id is a direct ID of a mounted element?
            # This is less common for actions but could happen.
            for elem in self.get_mounted_elements().values():
                if elem.id == element_id:
                     target_element = elem
                     logger.debug(f"[{self.id}] Found target element by element_id '{element_id}' instead of mount_id.")
                     break
            
            if not target_element:
                return {"success": False, "error": f"[{self.id}] Target element '{element_id}' not found in this space."} 

        # Execute the action on the found element
        tool_provider = target_element.get_component_by_type("ToolProviderComponent") # Use string for safety
        if tool_provider and hasattr(tool_provider, 'execute_tool'):
            logger.info(f"[{self.id}] Executing action '{action_name}' on element '{target_element.name}' ({target_element.id}).")
            try:
                # Assume execute_tool handles parameter unpacking etc.
                result = tool_provider.execute_tool(action_name, **parameters)
                
                # Record action execution event
                action_event_payload = {
                    'event_type': 'action_executed',
                    'target_element_id': target_element.id, # Record which element was acted upon
                    'payload': {
                        'action_name': action_name,
                        'parameters': parameters,
                        'result': result # Include the result
                    }
                }
                # Use default timeline context (primary timeline)
                timeline_context = {}
                primary_timeline = self.get_primary_timeline()
                if primary_timeline:
                    timeline_context['timeline_id'] = primary_timeline
                self.add_event_to_timeline(action_event_payload, timeline_context)
                
                return result if isinstance(result, dict) else {"success": True, "result": result}
            except Exception as e:
                 logger.error(f"[{self.id}] Error executing action '{action_name}' on element '{target_element.id}': {e}", exc_info=True)
                 return {"success": False, "error": f"Error executing action: {e}"} 
        else:
            return {"success": False, "error": f"Element '{target_element.name}' ({target_element.id}) does not have a runnable ToolProviderComponent."} 