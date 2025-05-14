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
    
    Spaces are structural containers that manage their own timeline (Loom DAG)
    and contained elements. Functionality is primarily delegated to components
    like ContainerComponent and TimelineComponent.
    """
    
    # Space is a container for other elements
    IS_SPACE = True

    EVENT_TYPES = []
    
    def __init__(self, element_id: str, name: str, description: str, 
                 adapter_id: Optional[str] = None, 
                 external_conversation_id: Optional[str] = None, 
                 **kwargs):
        """
        Initialize the space.
        
        Args:
            element_id: Unique identifier for this space
            name: Human-readable name for this space
            description: Description of this space's purpose
            adapter_id: Optional ID of the adapter this space might be associated with (e.g., for SharedSpaces).
            external_conversation_id: Optional external ID (e.g., channel ID) this space might represent.
            **kwargs: Passthrough for BaseElement
        """
        super().__init__(element_id=element_id, name=name, description=description, **kwargs)
        
        # Store adapter and external conversation IDs if provided (especially for SharedSpaces)
        self.adapter_id = adapter_id
        self.external_conversation_id = external_conversation_id
        
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
    def get_timeline(self) -> Optional[TimelineComponent]:
        """Get the timeline component."""
        return self.get_component_by_type(TimelineComponent)

    def add_event_to_primary_timeline(self, event_payload: Dict[str, Any]) -> Optional[str]:
        """Add an event to the primary timeline (delegates to TimelineComponent)."""
        if not self._timeline:
            logger.error(f"[{self.id}] Cannot add event to timeline: TimelineComponent unavailable.")
            return None
        return self._timeline.add_event_to_primary_timeline(event_payload)

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
        
    # --- NEW: Method to find an element by ID ---
    def get_element_by_id(self, element_id: str) -> Optional[BaseElement]:
        """
        Finds an element by its ID. 
        Checks if it's the Space itself or one of its directly mounted elements.
        """
        if self.id == element_id:
            return self
        
        # Check mounted elements by their actual element ID (not just mount_id)
        if self._container:
            for mounted_element in self._container.get_mounted_elements().values():
                if mounted_element.id == element_id:
                    return mounted_element
        
        logger.debug(f"[{self.id}] Element with ID '{element_id}' not found in this space or its direct children.")
        return None

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
                    raise ValueError(f"Mounted element '{target_element_id}' has no receive_event method.")
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
    def on_frame_end(self) -> None:
        """
        Called by the HostEventLoop (or owning Space) after its processing frame.
        Sends cached VEIL deltas to all registered uplink listeners.
        """
        if self._cached_deltas:
            logger.debug(f"[{self.id}] on_frame_end: Sending {len(self._cached_deltas)} cached deltas to {len(self._uplink_listeners)} listeners.")
            for listener_callback in self._uplink_listeners:
                try:
                    listener_callback(self._cached_deltas)
                except Exception as e:
                    logger.error(f"[{self.id}] Error calling uplink listener: {e}", exc_info=True)
            self._cached_deltas.clear()
        else:
            logger.debug(f"[{self.id}] on_frame_end: No deltas to send.")
            
    # --- Uplink Support Methods ---
    def get_space_metadata_for_uplink(self) -> Dict[str, Any]:
        """
        Provides essential metadata about this space for an UplinkProxy 
        during its initial synchronization.
            
        Returns:
            A dictionary containing space ID, name, description, and potentially
            adapter-specific identifiers if available from a component.
        """
        metadata = {
            "space_id": self.id,
            "name": self.name,
            "description": self.description,
            "element_type": self.__class__.__name__,
            "adapter_id": self.adapter_id, 
            "external_conversation_id": self.external_conversation_id
        }
        # Example: If a space represents an external channel, it might have a component
        # holding this info. For now, this is illustrative.
        # channel_info_comp = self.get_component_by_type("ChannelInfoComponent")
        # if channel_info_comp:
        #     metadata["adapter_id"] = channel_info_comp.get_adapter_id()
        #     metadata["external_conversation_id"] = channel_info_comp.get_external_id()
            
        logger.debug(f"[{self.id}] Providing metadata for uplink: {metadata}")
        return metadata

    def get_full_veil_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Generates and returns a full VEIL snapshot of this space and its contents.
        Delegates to the SpaceVeilProducerComponent if present.
            
        Returns:
            A dictionary representing the full VEIL state, or None if generation fails.
        """
        veil_producer = self.get_component_by_type("SpaceVeilProducerComponent")
        if veil_producer and hasattr(veil_producer, 'produce_veil'):
            logger.debug(f"[{self.id}] Generating full VEIL snapshot via SpaceVeilProducerComponent.")
            try:
                # Assuming produce_veil() with no arguments means full snapshot
                # and it needs the current element_states which it can get from self.element
                # or the producer is already observing the space.
                # The VeilProducer's produce_veil usually takes element_states.
                # For a SpaceVeilProducer, it might need to gather states from mounted elements.
                # Let's assume its produce_veil can operate on its owner (the Space) directly.
                return veil_producer.produce_veil() 
            except Exception as e:
                logger.error(f"[{self.id}] Error generating VEIL snapshot from SpaceVeilProducerComponent: {e}", exc_info=True)
                return None
        else:
            logger.warning(f"[{self.id}] SpaceVeilProducerComponent not found or 'produce_veil' method missing. Cannot generate full VEIL snapshot.")
            # Fallback: return basic space info if no producer
            return {
                "element_id": self.id,
                "name": self.name,
                "description": self.description,
                "element_type": self.__class__.__name__,
                "mounted_elements": {mount_id: elem.name for mount_id, elem in self.get_mounted_elements().items()},
                "components": [comp.COMPONENT_TYPE for comp in self.get_components().values()],
                "veil_status": "producer_missing"
            }

    # --- Lifecycle & Debug --- 
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, name={self.name}, description={self.description})"

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