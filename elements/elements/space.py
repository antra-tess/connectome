"""
Space
Base class for space elements that can contain other elements.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Type, Set
import uuid
import time
import asyncio

from .base import BaseElement, MountType
from .components.space import ContainerComponent, TimelineComponent
from .components.tool_provider import ToolProviderComponent
from .components.factory_component import ElementFactoryComponent
from .components.chat_manager_component import ChatManagerComponent
from .components.tool_provider import ToolProviderComponent 

# Type checking imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from host.event_loop import OutgoingActionCallback # For type hint

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
    IS_UPLINK_SPACE = False
    
    EVENT_TYPES = []
    
    def __init__(self, element_id: str, name: str, description: str, 
                 adapter_id: Optional[str] = None, 
                 external_conversation_id: Optional[str] = None,
                 outgoing_action_callback: Optional['OutgoingActionCallback'] = None,
                 **kwargs):
        """
        Initialize the space.
        
        Args:
            element_id: Unique identifier for this space
            name: Human-readable name for this space
            description: Description of this space's purpose
            adapter_id: Optional ID of the adapter this space might be associated with (e.g., for SharedSpaces).
            external_conversation_id: Optional external ID (e.g., channel ID) this space might represent.
            outgoing_action_callback: Optional callback for dispatching actions externally.
            **kwargs: Passthrough for BaseElement
        """
        super().__init__(element_id=element_id, name=name, description=description, **kwargs)
        
        # Store adapter and external conversation IDs if provided (especially for SharedSpaces)
        self.adapter_id = adapter_id
        self.external_conversation_id = external_conversation_id
        self._outgoing_action_callback = outgoing_action_callback
        
        # Initialize space with required components
        # Store references for easier delegation, though get_component could also be used.
        self._container: Optional[ContainerComponent] = self.add_component(ContainerComponent)
        self._timeline: Optional[TimelineComponent] = self.add_component(TimelineComponent)
        
        
        # NEW: Add ElementFactoryComponent by default
        factory_kwargs = {}
        if self._outgoing_action_callback:
            # If ElementFactoryComponent is updated to take this in __init__
            factory_kwargs['outgoing_action_callback'] = self._outgoing_action_callback
        self._tool_provider = self.add_component(ToolProviderComponent)
        
        if not self.IS_UPLINK_SPACE:    
            self._element_factory: Optional[ElementFactoryComponent] = self.add_component(ElementFactoryComponent, **factory_kwargs)
            self.add_component(ChatManagerComponent)
        # assert  False, self._element_factory
        # NEW: Initialize uplink listeners set and delta cache
        self._uplink_listeners: Set[Callable[[List[Dict[str, Any]]], None]] = set()
        self._cached_deltas: List[Dict[str, Any]] = []
        
        # NEW: Add ChatManagerComponent by default to all Spaces
        
        
        if not self._container or not self._timeline:
            # BaseElement.add_component logs errors, but we might want to raise here
            logger.critical(f"CRITICAL: Failed to initialize required components (Container/Timeline) for space {self.id}. Space may be unstable.")
            # raise RuntimeError(f"Failed to initialize required components for space {self.id}")
        if not self.IS_UPLINK_SPACE and not self._element_factory:
            logger.critical(f"CRITICAL: Failed to initialize ElementFactoryComponent for space {self.id}. Space may be unstable.")
            # raise RuntimeError(f"Failed to initialize ElementFactoryComponent for space {self.id}")
        
        if not self.IS_UPLINK_SPACE:
            # If ElementFactoryComponent uses a setter instead of __init__ for callback:
            if self._element_factory and self._outgoing_action_callback and not factory_kwargs.get('outgoing_action_callback'):
                if hasattr(self._element_factory, 'set_outgoing_action_callback') and callable(getattr(self._element_factory, 'set_outgoing_action_callback')):
                    self._element_factory.set_outgoing_action_callback(self._outgoing_action_callback)

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
        

    def get_tool_provider(self) -> Optional[ToolProviderComponent]:
        """Get the tool provider component."""
        return self._tool_provider
        
    def get_element_factory(self) -> Optional[ElementFactoryComponent]:
        """Get the element factory component."""
        return self._element_factory

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
        # Use the original event_payload (the first argument to this method) for dispatching to child
        original_target_element_id = event_payload.get("target_element_id") 

        if original_target_element_id:
            # Ensure lookup is by actual element ID if mounted_elements keys are mount_ids that might differ
            # For now, assuming get_mounted_element can handle element_id or mount_id.
            # Or, iterate values:
            mounted_element = None
            if self._container: # Check if ContainerComponent exists
                direct_child = self._container.get_mounted_element(original_target_element_id)
                if direct_child and direct_child.id == original_target_element_id:
                    mounted_element = direct_child
                else: # Fallback: iterate if mount_id != element_id
                    for elem in self._container.get_mounted_elements().values():
                        if elem.id == original_target_element_id:
                            mounted_element = elem
                            break
            
            if mounted_element:
                if hasattr(mounted_element, 'receive_event') and callable(mounted_element.receive_event):
                    logger.debug(f"[{self.id}] Routing event (ID: '{new_event_id}', Type: '{event_payload.get('event_type')}') to mounted element: {mounted_element.id}")
                    try:
                        # Child element receives the original event_payload and timeline context
                        mounted_element.receive_event(event_payload, timeline_context)
                    except Exception as mounted_err:
                        logger.error(f"[{self.id}] Error in mounted element '{mounted_element.id}' receiving event '{new_event_id}': {mounted_err}", exc_info=True)
                else:
                    logger.warning(f"[{self.id}] Mounted element '{original_target_element_id}' found but has no receive_event method.")
                    raise ValueError(f"Mounted element '{original_target_element_id}' has no receive_event method.")
            else:
                 # This could happen if the event targets an element nested deeper
                 # We rely on parent elements routing downwards. If it wasn't found here,
                 # it means the target wasn't a direct child.
                 logger.debug(f"[{self.id}] Event '{new_event_id}' targets element '{original_target_element_id}', which is not directly mounted here.")
        else:
             # Event was not targeted at a specific child element
             # Ensure this logging happens only if no specific child target AND not handled by own components in a way that stops propagation.
             # For now, this is fine as a general log if no specific target_element_id was in the original event_payload.
             if not original_target_element_id and not handled_by_component:
                 logger.debug(f"[{self.id}] Event '{new_event_id}' ({event_type}) processed by Space components (or no specific child target and no component handled it).")
             elif not original_target_element_id and handled_by_component:
                 logger.debug(f"[{self.id}] Event '{new_event_id}' ({event_type}) handled by Space component(s); no specific child target.")

        # --- NEW: Handle action_request_for_remote --- 
        if event_type == "action_request_for_remote":
            logger.info(f"[{self.id}] Processing event type 'action_request_for_remote' (Event ID: {new_event_id})")
            target_id_for_action = event_payload.get("remote_target_element_id")
            action_to_execute = event_payload.get("action_name")
            params_for_action = event_payload.get("action_parameters")
            source_uplink_id = event_payload.get("source_uplink_id")
            source_agent_id = event_payload.get("source_agent_id")

            if not target_id_for_action or not action_to_execute or params_for_action is None: # params can be empty dict
                logger.error(f"[{self.id}] Malformed 'action_request_for_remote': missing target_id ('{target_id_for_action}'), action_name ('{action_to_execute}'), or parameters. Payload: {event_payload}")
                return # Stop processing this specific path

            logger.info(f"[{self.id}] Attempting to execute remote action '{action_to_execute}' on target element '{target_id_for_action}' with params: {params_for_action}")
            
            # Prepare calling context for the action execution
            action_calling_context = {
                "source_type": "remote_uplink_request",
                "source_uplink_id": source_uplink_id,
                "source_agent_id": source_agent_id, # The original agent who initiated via uplink
                "original_event_id": new_event_id, # ID of the action_request_for_remote event itself
                "timeline_id": timeline_context.get('timeline_id')
            }

            # Use the existing execute_action_on_element method
            # This method is async, so if receive_event becomes async, this should be awaited.
            # For now, if execute_action_on_element needs to be async and called from sync receive_event,
            # it would require asyncio.create_task or similar, and handling of its result might be deferred.
            # Assuming Space.execute_action_on_element can be called and completes sufficiently for now.
            # If execute_action_on_element becomes truly async, receive_event might need to be async too.
            # For now, let's assume it is okay to call it like this from a synchronous method
            # or that execute_action_on_element handles its async nature appropriately (e.g. for tool calls)
            # If execute_action_on_element is async, this needs to be: `asyncio.create_task(self.execute_action_on_element(...))`
            # And this `receive_event` cannot directly return its result.
            # Re-checking `execute_action_on_element`: it is indeed async.
            # This means `receive_event` should ideally be async if it wants to `await` this.
            # For now, we will log a warning if it's not awaited and proceed. This is a larger refactor point.
            # TODO: Refactor receive_event to be async if it needs to await async operations like execute_action_on_element.
            
            # This is a significant architectural point: event handlers might need to become async.
            # import asyncio # Keep for create_task
            try:
                # Use asyncio.create_task to run the async method from this sync method
                asyncio.create_task(self.execute_action_on_element(
                    target_id_for_action, 
                    action_to_execute, 
                    params_for_action, 
                    action_calling_context
                ))
                logger.info(f"[{self.id}] Task created for remote action '{action_to_execute}' on '{target_id_for_action}'."
                            f" IMPORTANT: Result of this async action is not directly handled by this synchronous receive_event.")
            except RuntimeError as e:
                # This can happen if no event loop is running, e.g. in some test setups or sync parts of code
                logger.error(f"[{self.id}] Could not create task for async execute_action_on_element: {e}. "
                             f"The action '{action_to_execute}' might not be executed. Ensure an event loop is running.")
            except Exception as e: # Catching general Exception for safety
                logger.error(f"[{self.id}] Error creating task for async execute_action_on_element for remote action: {e}", exc_info=True)

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

    def receive_delta(self, delta: List[Dict[str, Any]]) -> None:
        """
        Receives a VEIL delta from an UplinkProxy and adds it to the Space's timeline.
        """
        event_payload = {
            "event_type": "Veil Delta Calculated",
            "delta": delta
        }
        self.add_event_to_timeline(event_payload, {"timeline_id": self.get_primary_timeline()})
        self._cached_deltas.extend(delta)

    # --- Action Execution --- 
    async def execute_action_on_element(self, element_id: str, action_name: str, parameters: Dict[str, Any], calling_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Allows this Space to execute an action (tool) on one of its mounted child elements.
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
        tool_provider = target_element.get_component_by_type("ToolProviderComponent")
        if not tool_provider:
            err_msg = f"Cannot execute action '{action_name}' on element '{element_id}': Target element does not have a ToolProviderComponent."
            logger.error(f"[{self.id}] {err_msg}")
            return {"success": False, "error": err_msg}

        # Execute the tool via the ToolProviderComponent
        logger.info(f"[{self.id}] Executing action '{action_name}' on element '{target_element.name}' ({target_element.id}). Params: {parameters}. Context: {calling_context}")
        action_result = await tool_provider.execute_tool(action_name, calling_context=calling_context, **parameters)
        
        # TODO: Consider recording the action_result or a summary as an event on the Space's timeline?
        # For example, if an agent successfully sends a message, that could be a timeline event.
        
        # Record action execution event
        action_event_payload = {
            'event_type': 'action_executed',
            'target_element_id': target_element.id, # Record which element was acted upon
            'payload': {
                'action_name': action_name,
                'parameters': parameters,
                'result': action_result # Include the result
            }
        }
        # Use default timeline context (primary timeline)
        timeline_context = {}
        primary_timeline = self.get_primary_timeline()
        if primary_timeline:
            timeline_context['timeline_id'] = primary_timeline
        self.add_event_to_timeline(action_event_payload, timeline_context)
        
        return action_result if isinstance(action_result, dict) else {"success": True, "result": action_result} 

    def get_public_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Collects tool definitions from this Space and its mounted child elements
        that have a ToolProviderComponent.

        Returns:
            A list of dictionaries, where each dictionary contains:
            - 'provider_element_id': The ID of the element providing the tool.
            - 'tool_name': The name of the tool.
            - 'description': The tool's description.
            - 'parameters_schema': The schema for the tool's parameters.
        """
        all_tool_definitions: List[Dict[str, Any]] = []

        # 1. Get tools from this Space itself
        space_tool_provider = self.get_component_by_type(ToolProviderComponent)
        if space_tool_provider:
            raw_tools = space_tool_provider.get_llm_tool_definitions() # Returns List[LLMToolDefinition]
            for tool_def in raw_tools:
                all_tool_definitions.append({
                    "provider_element_id": self.id,
                    "tool_name": tool_def.name,
                    "description": tool_def.description,
                    "parameters_schema": tool_def.parameters
                })
            logger.debug(f"[{self.id}] Found {len(raw_tools)} tools on Space itself.")

        # 2. Get tools from mounted child elements
        if self._container: # Ensure ContainerComponent exists
            for child_element in self._container.get_mounted_elements().values():
                child_tool_provider = child_element.get_component_by_type(ToolProviderComponent)
                if child_tool_provider:
                    child_raw_tools = child_tool_provider.get_llm_tool_definitions()
                    for tool_def in child_raw_tools:
                        all_tool_definitions.append({
                            "provider_element_id": child_element.id,
                            "tool_name": tool_def.name,
                            "description": tool_def.description,
                            "parameters_schema": tool_def.parameters
                        })
                    logger.debug(f"[{self.id}] Found {len(child_raw_tools)} tools on child element '{child_element.id}'.")
        
        logger.info(f"[{self.id}] Collected {len(all_tool_definitions)} public tool definitions in total.")
        return all_tool_definitions 