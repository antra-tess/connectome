"""
Space
Base class for space elements that can contain other elements.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Type, Set, Tuple
import uuid
import time
import asyncio
import copy
import os
from enum import Enum

from .base import BaseElement, MountType
from .components.space import ContainerComponent, TimelineComponent, SpaceVeilProducer
from .components.tool_provider import ToolProviderComponent
from .components.factory_component import ElementFactoryComponent
from .components.chat_manager_component import ChatManagerComponent
from .components.tool_provider import ToolProviderComponent 
from .components.base_component import VeilProducer

# Type checking imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from host.event_loop import OutgoingActionCallback # For type hint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NEW: Event replay configuration and modes
class EventReplayMode(Enum):
    DISABLED = "disabled"
    ENABLED = "enabled" 
    SELECTIVE = "selective"  # Future: replay only specific event types
    ENABLED_WITH_VEIL_SNAPSHOT = "enabled_with_veil_snapshot"  # NEW: Structural replay + VEIL cache restoration

# NEW: Default replayability for event types (fallback when is_replayable flag not set)
# These are only used when events don't have explicit is_replayable flag
DEFAULT_REPLAYABLE_EVENTS = {
    # Structural events - generally safe to replay
    'element_mounted', 'element_unmounted', 'component_initialized', 
    'tool_provider_registered', 'element_created_from_prefab',
    'component_state_updated',
    
    # Message events - content restoration
    'message_received', 'historical_message_received', 'agent_message_confirmed', 'connectome_message_deleted', 'connectome_message_updated', 
    'connectome_reaction_added', 'connectome_reaction_removed', 'attachment_content_available',
    'connectome_message_send_confirmed', 'connectome_message_send_failed',
}

DEFAULT_NON_REPLAYABLE_EVENTS = {
    # Runtime-only events that shouldn't be replayed
    'timeline_created', 'veil_delta_operations_received_by_space', 'action_executed',
    'activation_call', 'agent_context_generated', 'tool_action_dispatched', 'tool_result_received',
    'final_action_dispatched', 'llm_response_processed', 'structured_tool_action_dispatched',
}

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
    IS_INNER_SPACE = False

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
        
        # NEW: Event replay configuration
        self._event_replay_mode = self._determine_replay_mode()
        self._replay_in_progress = False
        self._replayed_event_ids = set()  # Track which events have been replayed
        
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
            self._veil_producer: Optional[SpaceVeilProducer] = self.add_component(SpaceVeilProducer)
        # assert  False, self._element_factory
        # NEW: Initialize uplink listeners set and delta cache
        self._uplink_listeners: Set[Callable[[List[Dict[str, Any]]], None]] = set()
        self._cached_deltas: List[Dict[str, Any]] = []
        
        # NEW: Initialize internal hierarchical VEIL cache (Phase 1, Step 1.1)
        self._flat_veil_cache: Dict[str, Any] = {}
        # NEW: For accumulating deltas from children in the "emit on change" model
        self._deltas_accumulated_this_frame: List[Dict[str, Any]] = []
        self._ensure_own_veil_presence_initialized()
        
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

        # NEW: Schedule event replay after component initialization
        if self._event_replay_mode != EventReplayMode.DISABLED:
            # We need to schedule replay after storage is initialized
            # This will be triggered when TimelineComponent completes its async initialization
            logger.info(f"[{self.id}] Event replay mode: {self._event_replay_mode.value}")

        logger.info(f"Created space: {name} ({self.id})")

    def _determine_replay_mode(self) -> EventReplayMode:
        """Determine event replay mode from environment variables or configuration."""
        replay_mode = os.environ.get('CONNECTOME_EVENT_REPLAY_MODE', 'disabled').lower()
        
        if replay_mode == 'enabled':
            return EventReplayMode.ENABLED
        elif replay_mode == 'enabled_with_veil_snapshot':
            return EventReplayMode.ENABLED_WITH_VEIL_SNAPSHOT
        elif replay_mode == 'selective':
            return EventReplayMode.SELECTIVE
        else:
            return EventReplayMode.DISABLED

    def _ensure_own_veil_presence_initialized(self) -> None:
        """
        Ensures that the Space's own root VEIL node is present in self._flat_veil_cache.
        This is crucial to call before processing any deltas that might reference it as a parent,
        and also ensures it's there if on_frame_end triggers its own producer.
        """
        # Ensure _flat_veil_cache itself is initialized (should be by __init__)
        if not hasattr(self, '_flat_veil_cache'):
            logger.error(f"[{self.id}] Critical: _flat_veil_cache attribute not found during _ensure_own_veil_presence_initialized. Initializing to empty dict.")
            self._flat_veil_cache = {}

        space_root_id = f"{self.id}_space_root"

        if space_root_id not in self._flat_veil_cache:
            logger.warning(f"[{self.id}] Root VEIL node '{space_root_id}' not found in cache. Initializing in _ensure_own_veil_presence_initialized.")
            if not self.IS_UPLINK_SPACE:
                self._veil_producer.emit_delta()
            logger.info(f"[{self.id}] Initialized own root VEIL node '{space_root_id}' in _flat_veil_cache via _ensure_own_veil_presence_initialized.")
        else:
            logger.debug(f"[{self.id}] Own root VEIL node '{space_root_id}' already present in _flat_veil_cache.")


    def _apply_deltas_to_internal_cache(self, deltas: List[Dict[str, Any]]) -> None:
        """
        Applies a list of VEIL delta operations to the internal flat VEIL cache.
        Modifies self._flat_veil_cache in place.
        It checks for a top-level 'parent_id' in 'add_node' and 'update_node' operations
        and ensures this 'parent_id' is stored within the node's 'properties'.
        """
        if not isinstance(deltas, list):
            logger.warning(f"[{self.id}] Invalid deltas format received by _apply_deltas_to_internal_cache: not a list.")
            return
        logger.debug(f"[{self.id}] Applying {len(deltas)} deltas to internal flat cache.")

        for operation in deltas:
            op_type = operation.get("op")
            node_id_from_op = operation.get("veil_id") # Used by update_node, remove_node
            node_data_from_op = operation.get("node")   # Used by add_node
            top_level_parent_id = operation.get("parent_id") # Check for parent_id at the operation level

            if op_type == "add_node":
                if not node_data_from_op or not isinstance(node_data_from_op, dict) or "veil_id" not in node_data_from_op:
                    logger.warning(f"[{self.id}] Invalid 'add_node' operation (missing node, node not dict, or missing veil_id): {operation}")
                    continue
                
                new_node_id = node_data_from_op["veil_id"]
                
                # Ensure 'properties' dictionary exists in the node data
                if "properties" not in node_data_from_op or not isinstance(node_data_from_op.get("properties"), dict):
                    node_data_from_op["properties"] = {}
                
                # If parent_id is provided at the top level of the delta op, inject it into node's properties
                if top_level_parent_id:
                    node_data_from_op["properties"]["parent_id"] = top_level_parent_id
                    logger.debug(f"[{self.id}] Cache: Injecting parent_id '{top_level_parent_id}' into properties of node '{new_node_id}' during add_node.")

                if new_node_id in self._flat_veil_cache:
                    logger.debug(f"[{self.id}] Cache: Updating existing node {new_node_id} via add_node delta in _flat_veil_cache.")
                else:
                    logger.debug(f"[{self.id}] Cache: Adding new node {new_node_id} via add_node delta to _flat_veil_cache.")
                
                if "children" not in node_data_from_op or not isinstance(node_data_from_op.get("children"), list):
                    node_data_from_op["children"] = [] 
                
                self._flat_veil_cache[new_node_id] = node_data_from_op

            elif op_type == "update_node":
                if not node_id_from_op:
                    logger.warning(f"[{self.id}] Invalid 'update_node' operation (missing veil_id): {operation}")
                    continue
                
                if node_id_from_op in self._flat_veil_cache:
                    logger.debug(f"[{self.id}] Cache: Updating properties for node {node_id_from_op} in _flat_veil_cache.")
                    
                    # Ensure 'properties' dictionary exists in the cached node data
                    if "properties" not in self._flat_veil_cache[node_id_from_op] or \
                       not isinstance(self._flat_veil_cache[node_id_from_op].get("properties"), dict):
                         self._flat_veil_cache[node_id_from_op]["properties"] = {}

                    # Apply property updates from the delta
                    properties_to_update = operation.get("properties")
                    if properties_to_update is not None: # Can be an empty dict to clear properties
                        self._flat_veil_cache[node_id_from_op]["properties"].update(properties_to_update)
                    else:
                        # If "properties" is not in the delta, it's not an error, just means no property changes.
                        # However, if "properties" is explicitly null, it could mean clear all, but our current
                        # model uses an empty dict for that if updating. Let's log if it's missing but was expected.
                        # For now, if properties_to_update is None, we only care about potential parent_id change.
                        pass 

                    # If parent_id is provided at the top level of the update_node op, update it in node's properties
                    if top_level_parent_id:
                        self._flat_veil_cache[node_id_from_op]["properties"]["parent_id"] = top_level_parent_id
                        logger.debug(f"[{self.id}] Cache: Injecting/updating parent_id '{top_level_parent_id}' into properties of node '{node_id_from_op}' during update_node.")
                else:
                    logger.warning(f"[{self.id}] Cache: 'update_node' for {node_id_from_op} but node not found in _flat_veil_cache.")
            
            elif op_type == "remove_node":
                if not node_id_from_op:
                    logger.warning(f"[{self.id}] Invalid 'remove_node' operation (missing veil_id): {operation}")
                    continue

                if node_id_from_op in self._flat_veil_cache:
                    logger.debug(f"[{self.id}] Cache: Removing node {node_id_from_op} from _flat_veil_cache.")
                    del self._flat_veil_cache[node_id_from_op]
                    # No need to remove from parent's children list here;
                    # SpaceVeilProducer will handle this during reconstruction.
                else:
                    logger.warning(f"[{self.id}] Cache: 'remove_node' for {node_id_from_op} but node not found in _flat_veil_cache.")
            else:
                logger.warning(f"[{self.id}] Cache: Unsupported delta operation '{op_type}' received. Skipping.")

    # --- Container Methods (Delegated) ---
    def mount_element(self, element: BaseElement, mount_id: Optional[str] = None, mount_type: MountType = MountType.INCLUSION, creation_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        """Mount an element in this space (delegates to ContainerComponent)."""
        if not self._container:
            logger.error(f"[{self.id}] Cannot mount element: ContainerComponent unavailable.")
            return False, None
        # Pass self.id as parent_space_id to the element being mounted
        mount_successful, final_mount_id = self._container.mount_element(element, mount_id, mount_type, creation_data)
        return mount_successful, final_mount_id
    
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

    def _register_adapter_mapping_from_event(self, event_payload: Dict[str, Any]) -> None:
        """
        Register adapter mapping from a message event for mention detection.
        Extracts agent adapter information from the event payload and registers it if it matches this agent.
        
        Args:
            event_payload: The message event payload containing original_adapter_data
        """
        try:
            # Only register for InnerSpaces that have the registration method
            if not hasattr(self, 'register_adapter_mapping') or not callable(getattr(self, 'register_adapter_mapping')):
                return
            
            # Extract data from the event payload (nested in payload.payload)
            inner_payload = event_payload.get('payload', {})
            original_adapter_data = inner_payload.get('original_adapter_data', {})
            
            if not original_adapter_data:
                logger.debug(f"[{self.id}] No original_adapter_data found in event for adapter mapping registration")
                return
            
            # Extract agent information from the original adapter data
            agent_adapter_name = original_adapter_data.get('adapter_name')  # e.g., "Alena Bot 2"
            agent_adapter_id = original_adapter_data.get('adapter_id')      # e.g., "756"
            adapter_type = inner_payload.get('adapter_type')               # e.g., "zulip" (might be None)
            
            # Use a fallback if adapter_type is missing
            if not adapter_type:
                # Try to infer from source_adapter_id or use 'unknown'
                source_adapter_id = inner_payload.get('source_adapter_id', '')
                if 'zulip' in source_adapter_id.lower():
                    adapter_type = 'zulip'
                elif 'discord' in source_adapter_id.lower():
                    adapter_type = 'discord'
                else:
                    adapter_type = 'unknown'
                
            if agent_adapter_name and agent_adapter_id:
                logger.debug(f"[{self.id}] Registering adapter mapping from event: {adapter_type}/{agent_adapter_name} -> {agent_adapter_id}")
                self.register_adapter_mapping(adapter_type, agent_adapter_name, agent_adapter_id)
            else:
                logger.debug(f"[{self.id}] Incomplete adapter mapping data in event: adapter_name={agent_adapter_name}, adapter_id={agent_adapter_id}")
                
        except Exception as e:
            logger.warning(f"[{self.id}] Error registering adapter mapping from event: {e}")
            
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
        
        # NEW: Check if this is a replay mode to prevent double-recording
        is_replay_mode = timeline_context.get('replay_mode', False)
        
        logger.debug(f"[{self.id}] Receiving event: Type='{event_type}', Target='{target_element_id}', Timeline='{timeline_context.get('timeline_id')}', Replay={is_replay_mode}")
        
        # 1. Add event to the timeline via TimelineComponent (unless in replay mode)
        new_event_id = None
        if not is_replay_mode:
            new_event_id = self.add_event_to_timeline(event_payload, timeline_context)
            if not new_event_id:
                logger.error(f"[{self.id}] Failed to add event to timeline. Aborting processing for this event. Event: {event_payload}")
                return
        else:
            # In replay mode, use a dummy event ID for processing
            new_event_id = f"replay_{event_type}_{int(time.time() * 1000)}"
        
        # NEW: Handle adapter mapping registration for message events (only for InnerSpaces)
        if self.IS_INNER_SPACE:
            self._register_adapter_mapping_from_event(event_payload)
            
        # Construct the full event node as it exists in the DAG for handlers
        full_event_node = { 
             'id': new_event_id,
             'timeline_id': timeline_context.get('timeline_id', self.get_primary_timeline()),
             # We don't easily know parent_ids/timestamp here without querying timeline again,
             # so components should rely on the payload for now, or query if needed.
             'payload': event_payload.copy() # PASS A SHALLOW COPY TO COMPONENTS
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
            action_name_check = event_payload.get("action_name")
            
            logger.info(f"[{self.id}] Processing event type 'action_request_for_remote' (Event ID: {new_event_id})")
            target_id_for_action = event_payload.get("remote_target_element_id")
            action_name_from_payload = event_payload.get("action_name")
            parameters_from_payload = event_payload.get("action_parameters", {})

            if target_id_for_action and action_name_from_payload:
                # Construct calling_context for the action on the remote element
                remote_action_calling_context = {
                    "source_agent_id": event_payload.get("source_agent_id"),
                    "source_agent_name": event_payload.get("source_agent_name"),
                    "source_uplink_id": event_payload.get("source_uplink_id"),
                    "original_event_id": new_event_id # Link back to this triggering event
                }
                logger.info(f"[{self.id}] Executing remote action on element '{target_id_for_action}', action: '{action_name_from_payload}', context: {remote_action_calling_context}")
                
                # Execute the action.
                # This needs to be async if execute_action_on_element can be.
                # For now, assuming it might involve async tool execution eventually.
                asyncio.create_task(
                    self.execute_action_on_element(
                        element_id=target_id_for_action,
                        action_name=action_name_from_payload,
                        parameters=parameters_from_payload,
                        calling_context=remote_action_calling_context # PASS THE CONSTRUCTED CONTEXT
                    )
                )
            else:
                logger.error(f"[{self.id}] 'action_request_for_remote' event (ID: {new_event_id}) missing remote_target_element_id or action_name. Payload: {event_payload}")
            return # Event processed by this specific handler
        # --- End handle action_request_for_remote ---

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
        In the new "emit on change" model:
        1. Triggers its own SpaceVeilProducer to emit deltas (which will be cached via receive_delta).
        2. Sends all deltas accumulated in _deltas_accumulated_this_frame to uplink listeners.
        """
        # 1. Trigger this Space's own VeilProducer to emit its delta
        # This ensures the Space's own root node state is up-to-date in the cache and accumulated deltas.
        if not self.IS_UPLINK_SPACE and self._veil_producer and hasattr(self._veil_producer, 'emit_delta'):
            try:
                logger.debug(f"[{self.id}] on_frame_end: Triggering emit_delta for own SpaceVeilProducer.")
                self._veil_producer.emit_delta() # This will call calculate_delta and then self.receive_delta
            except Exception as e:
                logger.error(f"[{self.id}] on_frame_end: Error triggering emit_delta for SpaceVeilProducer: {e}", exc_info=True)

        # 2. Send accumulated deltas to listeners
        if self._deltas_accumulated_this_frame:
            logger.debug(f"[{self.id}] on_frame_end: Sending {len(self._deltas_accumulated_this_frame)} accumulated deltas to {len(self._uplink_listeners)} listeners.")
            
            # Create a copy for sending, as listeners might modify it or processing might take time.
            deltas_to_send = list(self._deltas_accumulated_this_frame) 
            
            all_listeners_notified_successfully = True
            for listener_callback in self._uplink_listeners:
                try:
                    listener_callback(deltas_to_send)
                except Exception as e:
                    all_listeners_notified_successfully = False
                    logger.error(f"[{self.id}] Error calling uplink listener {listener_callback}: {e}", exc_info=True)
            
            # Clear the accumulated deltas for the next frame, regardless of listener success for now.
            # If listener success is critical for not clearing, that logic would be more complex (e.g., retry, dead-letter queue for deltas).
            self._deltas_accumulated_this_frame.clear()
        else:
            logger.debug(f"[{self.id}] on_frame_end: No accumulated deltas to send.")
            
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
        Delegates the construction of the hierarchy to its SpaceVeilProducer,
        providing it with the Space's internal flat cache.
        """
        if self.IS_UPLINK_SPACE: # UplinkSpaces might not have a standard _veil_producer
            logger.warning(f"[{self.id}] get_full_veil_snapshot() called on an UplinkSpace. UplinkSpaces typically do not produce their own VEIL in this manner.")
            # Or, they might have a different producer type. For now, return None or basic info.
            # Consider what an UplinkSpace should return here.
            # For now, let's provide its own root if the cache has it.
            space_root_id = f"{self.id}_space_root"
            if self._flat_veil_cache and space_root_id in self._flat_veil_cache:
                 return copy.deepcopy(self._flat_veil_cache[space_root_id]) # Just its own node
            return {"veil_id": space_root_id, "properties": {"name": self.name, "status": "UplinkSpace - snapshot not applicable"}, "children": []}


        if not self._veil_producer:
            logger.error(f"[{self.id}] Cannot generate full VEIL snapshot: _veil_producer (SpaceVeilProducer) not found.")
            # Fallback: return basic space info if producer is missing
            space_root_id = f"{self.id}_space_root"
            return {
                "veil_id": space_root_id,
                "node_type": "space_root", # VEIL_SPACE_ROOT_TYPE, # - might not be defined here
                "properties": {
                    "element_id": self.id,
                    "element_name": self.name,
                    "description": self.description,
                    "element_type": self.__class__.__name__,
                    "veil_status": "veil_producer_missing_on_space"
                },
                "children": []
            }

        # The SpaceVeilProducer's get_full_veil method now handles building from this Space's cache.
        logger.debug(f"[{self.id}] get_full_veil_snapshot: Delegating to SpaceVeilProducer to build VEIL from internal cache.")
        try:
            # SpaceVeilProducer.get_full_veil() will use self.owner._flat_veil_cache
            # and self.owner.id (which are this Space's attributes)
            full_reconstructed_veil = self._veil_producer.get_full_veil() 
            
            if full_reconstructed_veil:
                return full_reconstructed_veil # Already a deepcopy from the producer's builder
            else:
                # This case implies an issue within the producer's get_full_veil or its builder
                logger.error(f"[{self.id}] SpaceVeilProducer.get_full_veil() returned None or empty. Snapshot generation failed.")
                space_root_id = f"{self.id}_space_root"
                return {
                    "veil_id": space_root_id,
                    "node_type": "space_root",
                    "properties": {
                        "element_id": self.id,
                        "element_name": self.name,
                        "description": self.description,
                        "element_type": self.__class__.__name__,
                        "veil_status": "veil_producer_failed_to_build_hierarchy"
                     },
                    "children": list(self._flat_veil_cache.values()) if self._flat_veil_cache else [] 
                }
        except Exception as e:
            logger.error(f"[{self.id}] Error calling SpaceVeilProducer.get_full_veil() for snapshot: {e}", exc_info=True)
            space_root_id = f"{self.id}_space_root"
            return {
                "veil_id": space_root_id,
                "node_type": "space_root",
                "properties": {
                    "element_id": self.id,
                    "element_name": self.name,
                    "description": self.description,
                    "element_type": self.__class__.__name__,
                    "veil_status": f"error_delegating_to_veil_producer: {e}"
                },
                "children": []
            }

    # --- Lifecycle & Debug --- 
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, name={self.name}, description={self.description})"

    # Optional: Method for listeners to pull cached deltas if needed
    def get_cached_deltas(self) -> List[Dict[str, Any]]:
        """Returns the VEIL deltas calculated in the last frame."""
        # Consider clearing cache after retrieval? Or timestamping?
        return list(self._cached_deltas) # Return a copy

    def receive_delta(self, delta_operations: List[Dict[str, Any]]) -> None:
        """
        Receives VEIL delta operations, typically from a child element's VeilProducer
        (via its emit_delta -> owner.receive_delta -> this_method chain).
        Applies these deltas to the Space's internal flat VEIL cache and
        accumulates them for dispatch by on_frame_end.
        """
        if not isinstance(delta_operations, list) or not delta_operations:
            logger.debug(f"[{self.id}] Space.receive_delta called with no valid delta operations. Skipping.")
            return

        logger.debug(f"[{self.id}] Space.receive_delta: Processing {len(delta_operations)} delta operations.")

        # 1. Apply to internal flat VEIL cache
        try:
            self._apply_deltas_to_internal_cache(delta_operations)
            logger.debug(f"[{self.id}] Space.receive_delta: Applied {len(delta_operations)} operations to _flat_veil_cache.")
        except Exception as e:
            logger.error(f"[{self.id}] Space.receive_delta: Error applying deltas to internal cache: {e}", exc_info=True)
            # Decide if we should still accumulate if caching fails. For now, let's still accumulate.

        # 2. Accumulate for on_frame_end dispatch
        self._deltas_accumulated_this_frame.extend(delta_operations)
        logger.debug(f"[{self.id}] Space.receive_delta: Added {len(delta_operations)} operations to _deltas_accumulated_this_frame. Total accumulated: {len(self._deltas_accumulated_this_frame)}")

        # Optional: Record an event to the timeline (as before, but maybe less critical now that cache is updated)
        # This might be too noisy if every producer emitting causes an event here.
        # Consider if this specific event is still needed or if the cache changes are sufficient.
        # For now, let's keep it to see its utility.
        event_payload = {
            "event_type": "veil_delta_operations_received_by_space", 
            "payload": { 
                "source": "child_producer_emission", # Indicates source
                "delta_operation_count": len(delta_operations)
            }
        }
        timeline_context = {} # TODO: ensure this gets proper context from caller if needed
        primary_timeline = self.get_primary_timeline()
        if primary_timeline:
            timeline_context['timeline_id'] = primary_timeline
        try:
            self.add_event_to_timeline(event_payload, timeline_context) # Commented out to reduce noise, can be re-enabled if useful for debugging.
        except Exception as e:
            logger.error(f"[{self.id}] Space.receive_delta: Error adding event to timeline: {e}", exc_info=True)

    def get_flat_veil_snapshot(self) -> Dict[str, Any]:
        """
        Returns a deep copy of the Space's internal flat VEIL node cache.
        The cache is a dictionary where keys are veil_ids and values are node data.
        Hierarchy is not explicitly stored here; it's reconstructed by producers.
        """
        logger.debug(f"[{self.id}] Providing flat VEIL cache snapshot. Size: {len(self._flat_veil_cache)}")
        return copy.deepcopy(self._flat_veil_cache)

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

    # --- NEW: Event Replay Methods ---
    
    async def replay_events_from_timeline(self) -> bool:
        """
        Replay timeline events to reconstruct system state.
        Should be called after storage is initialized but before normal operation.
        
        Returns:
            True if replay completed successfully, False otherwise
        """
        if self._event_replay_mode == EventReplayMode.DISABLED:
            logger.debug(f"[{self.id}] Event replay disabled, skipping")
            return True
            
        if self._replay_in_progress:
            logger.warning(f"[{self.id}] Event replay already in progress, skipping")
            return False
            
        if not self._timeline:
            logger.error(f"[{self.id}] Cannot replay events: TimelineComponent not available")
            return False
            
        try:
            self._replay_in_progress = True
            logger.info(f"[{self.id}] Starting event replay for system state reconstruction")
            
            # Get all events from the primary timeline in chronological order
            primary_timeline_id = self._timeline.get_primary_timeline()
            if not primary_timeline_id:
                logger.info(f"[{self.id}] No primary timeline found, nothing to replay")
                return True
                
            # Get events in chronological order (oldest first for replay)
            all_events = self._timeline.get_timeline_events(primary_timeline_id, limit=0)
            
            # Reverse to get chronological order (oldest first)
            chronological_events = list(reversed(all_events))
            
            if not chronological_events:
                logger.info(f"[{self.id}] No events found in timeline, nothing to replay")
                return True
                
            logger.info(f"[{self.id}] Found {len(chronological_events)} events to potentially replay")
            
            replayed_count = 0
            skipped_count = 0
            
            for event_node in chronological_events:
                event_id = event_node.get('id')
                event_payload = event_node.get('payload', {})
                event_type = event_payload.get('event_type')
                
                # Skip if already replayed (shouldn't happen, but safety check)
                if event_id in self._replayed_event_ids:
                    logger.debug(f"[{self.id}] Skipping already replayed event {event_id}")
                    skipped_count += 1
                    continue
                    
                # Check if this event type should be replayed
                if not self._should_replay_event(event_type, event_payload):
                    logger.debug(f"[{self.id}] Skipping non-replayable event {event_id} ({event_type})")
                    skipped_count += 1
                    continue
                    
                # Log what we're about to replay
                logger.info(f"[{self.id}] Replaying event {event_id}: {event_type}")
                
                # Replay the event
                success = await self._replay_single_event(event_node)
                if success:
                    self._replayed_event_ids.add(event_id)
                    replayed_count += 1
                    logger.info(f"[{self.id}] ✓ Successfully replayed event {event_id} ({event_type})")
                else:
                    logger.error(f"[{self.id}] ✗ Failed to replay event {event_id} ({event_type})")
                    # Continue with other events even if one fails
                    skipped_count += 1
                    
            logger.info(f"[{self.id}] Event replay completed: {replayed_count} replayed, {skipped_count} skipped")
            
            # NEW: For VEIL snapshot mode, restore VEIL cache after structural replay
            if self._event_replay_mode == EventReplayMode.ENABLED_WITH_VEIL_SNAPSHOT:
                logger.info(f"[{self.id}] Attempting VEIL cache restoration after structural replay")
                veil_restoration_success = await self._restore_veil_snapshot()
                if not veil_restoration_success:
                    logger.warning(f"[{self.id}] VEIL cache restoration failed, regenerating from current state")
                    await self._regenerate_veil_state_after_replay()
            elif replayed_count > 0:
                # Only regenerate VEIL state if we actually replayed events
                # For new agents with no events, let normal initialization handle VEIL setup
                logger.info(f"[{self.id}] Regenerating VEIL state after replaying {replayed_count} events")
                await self._regenerate_veil_state_after_replay()
            else:
                logger.info(f"[{self.id}] No events replayed, skipping VEIL regeneration - normal initialization will handle VEIL setup")
            
            # NEW: After VEIL regeneration, ensure uplinks are connected (crucial for InnerSpaces)
            if replayed_count > 0:
                logger.info(f"[{self.id}] Connecting uplinks after timeline replay")
                await self._connect_uplinks_after_replay()
            
            # NEW: Verify replay integrity for message components
            await self._verify_replay_integrity()
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.id}] Error during event replay: {e}", exc_info=True)
            return False
        finally:
            self._replay_in_progress = False

    async def _regenerate_veil_state_after_replay(self) -> bool:
        """
        Regenerate VEIL state after event replay to ensure VEIL cache reflects recreated elements.
        
        Returns:
            True if regeneration was successful, False otherwise
        """
        try:
            logger.info(f"[{self.id}] Regenerating VEIL state after event replay")
            
            # Clear the existing VEIL cache to force regeneration
            self._flat_veil_cache.clear()
            
            # Ensure own VEIL presence is initialized
            self._ensure_own_veil_presence_initialized()
            
            # Trigger VEIL emission for all mounted elements
            if self._container:
                mounted_elements = self._container.get_mounted_elements()
                logger.info(f"[{self.id}] Triggering VEIL regeneration for {len(mounted_elements)} mounted elements")
                
                for mount_id, element in mounted_elements.items():
                    try:
                        logger.debug(f"[{self.id}] Processing mounted element: {mount_id} (type: {type(element).__name__})")
                        
                        # Check if element is a Space with on_frame_end method
                        if hasattr(element, 'on_frame_end') and callable(element.on_frame_end):
                            element.on_frame_end()
                            logger.debug(f"[{self.id}] ✓ Triggered on_frame_end for Space element {mount_id}")
                        else:
                            # For regular BaseElement instances, find and trigger VeilProducer components
                            veil_producers_found = 0
                            if hasattr(element, 'get_components') and callable(element.get_components):
                                components = element.get_components()
                                for component in components.values():
                                    # Check if component is a VeilProducer (has emit_delta method)
                                    if hasattr(component, 'emit_delta') and callable(component.emit_delta):
                                        try:
                                            component.emit_delta()
                                            veil_producers_found += 1
                                            logger.info(f"[{self.id}] ✓ Triggered emit_delta for VeilProducer {type(component).__name__} on element {mount_id}")
                                        except Exception as e:
                                            logger.error(f"[{self.id}] Error calling emit_delta on {type(component).__name__} for element {mount_id}: {e}")
                            
                            if veil_producers_found == 0:
                                logger.debug(f"[{self.id}] No VeilProducer components found on element {mount_id}")
                            else:
                                logger.debug(f"[{self.id}] ✓ Triggered {veil_producers_found} VeilProducer(s) on element {mount_id}")
                                
                    except Exception as e:
                        logger.warning(f"[{self.id}] Error triggering VEIL emission for element {mount_id}: {e}")
            else:
                logger.warning(f"[{self.id}] No container component found for VEIL regeneration")
            
            # Trigger own VEIL producer to ensure space root is updated
            if hasattr(self, 'on_frame_end') and callable(self.on_frame_end):
                self.on_frame_end()
                
            logger.info(f"[{self.id}] VEIL state regeneration completed. Cache size: {len(self._flat_veil_cache)}")
            return True
            
        except Exception as e:
            logger.error(f"[{self.id}] Error during VEIL state regeneration: {e}", exc_info=True)
            return False
    
    def _should_replay_event(self, event_type: str, event_payload: Dict[str, Any]) -> bool:
        """
        Determine if an event should be replayed based on its is_replayable flag or type.
        
        Args:
            event_type: The type of event
            event_payload: The full event payload
            
        Returns:
            True if the event should be replayed, False otherwise
        """
        if self._event_replay_mode == EventReplayMode.DISABLED:
            return False
            
        # NEW: Check for explicit is_replayable flag on the event first
        is_replayable = event_payload.get('is_replayable')
        if is_replayable is not None:
            logger.debug(f"[{self.id}] Event {event_type} has explicit is_replayable={is_replayable}")
            return bool(is_replayable)
            
        # Fallback to default behavior based on event type
        # Never replay explicitly non-replayable events
        if event_type in DEFAULT_NON_REPLAYABLE_EVENTS:
            logger.debug(f"[{self.id}] Event {event_type} is in DEFAULT_NON_REPLAYABLE_EVENTS")
            return False
            
        if self._event_replay_mode == EventReplayMode.ENABLED:
            # Replay events in default replayable list
            should_replay = event_type in DEFAULT_REPLAYABLE_EVENTS
            logger.debug(f"[{self.id}] Event {event_type} replayable by default: {should_replay}")
            return should_replay
            
        elif self._event_replay_mode == EventReplayMode.SELECTIVE:
            # Future: implement selective replay logic
            # For now, same as ENABLED
            should_replay = event_type in DEFAULT_REPLAYABLE_EVENTS  
            logger.debug(f"[{self.id}] Event {event_type} replayable (selective mode): {should_replay}")
            return should_replay
            
        elif self._event_replay_mode == EventReplayMode.ENABLED_WITH_VEIL_SNAPSHOT:
            # NEW: Structural replay + VEIL cache restoration - replay everything that's not explicitly non-replayable
            should_replay = event_type not in DEFAULT_NON_REPLAYABLE_EVENTS
            logger.debug(f"[{self.id}] Event {event_type} replayable (VEIL snapshot mode): {should_replay}")
            return should_replay
            
        return False
    
    async def _replay_single_event(self, event_node: Dict[str, Any]) -> bool:
        """
        Replay a single event to reconstruct system state.
        
        Args:
            event_node: The complete event node from timeline
            
        Returns:
            True if replay was successful, False otherwise
        """
        event_payload = event_node.get('payload', {})
        event_type = event_payload.get('event_type')
        event_data = event_payload.get('data', {})
        
        try:
            # For most events, we can use the generic receive_event mechanism with replay mode
            # This allows components to handle replay events the same way as live events
            timeline_context = {
                'timeline_id': event_node.get('timeline_id'),
                'replay_mode': True,  # Mark this as a replay to prevent double-recording
                'original_event_id': event_node.get('id'),
                'original_timestamp': event_node.get('timestamp')
            }
            
            # Use receive_event for most replay scenarios
            # This ensures components get the chance to handle replayed events
            logger.debug(f"[{self.id}] Replaying event {event_type} via receive_event mechanism")
            self.receive_event(event_payload, timeline_context)
            
            # For events that need special handling beyond component processing:
            if event_type == 'element_mounted':
                return await self._replay_element_mounted(event_data)
                
            elif event_type == 'element_unmounted':
                return await self._replay_element_unmounted(event_data)
                
            elif event_type == 'element_created_from_prefab':
                return await self._replay_element_created_from_prefab(event_data)
                
            # For events that are handled sufficiently by component processing:
            elif event_type in ['component_initialized', 'tool_provider_registered']:
                # These events are primarily informational or handled by component loading
                return True
                
            else:
                logger.debug(f"[{self.id}] Event type {event_type} replayed via component mechanism")
                return True
                
        except Exception as e:
            logger.error(f"[{self.id}] Error replaying {event_type} event: {e}", exc_info=True)
            return False
    
    async def _replay_element_mounted(self, event_data: Dict[str, Any]) -> bool:
        """Replay element_mounted event - recreate the mounted element if possible."""
        mount_id = event_data.get('mount_id')
        element_id = event_data.get('element_id')
        element_name = event_data.get('element_name')
        element_type = event_data.get('element_type')
        mount_type_str = event_data.get('mount_type', 'INCLUSION')
        
        # NEW: Extract creation data if available
        creation_data = event_data.get('creation_data', {})
        prefab_name = creation_data.get('prefab_name')
        element_config = creation_data.get('element_config', {})
        component_config = creation_data.get('component_config_overrides', {})
        
        logger.info(f"[{self.id}] REPLAY: element_mounted - {element_name} ({element_id}) as {mount_id}")
        
        # Check if element is already mounted (avoid double-mounting)
        if self._container and self._container.get_mounted_element(mount_id):
            logger.debug(f"[{self.id}] Element {mount_id} already mounted, skipping replay")
            return True
            
        # Try to recreate the element
        if not self._element_factory:
            logger.warning(f"[{self.id}] Cannot replay element mounting: No ElementFactoryComponent")
            return False
            
        try:
            # Convert mount_type string back to enum
            from .base import MountType
            if hasattr(MountType, mount_type_str):
                mount_type = getattr(MountType, mount_type_str)
            else:
                mount_type = MountType.INCLUSION
                
            if prefab_name:
                # Recreate from prefab if prefab info is available
                logger.info(f"[{self.id}] Recreating element from prefab: {prefab_name}")
                
                # Use ElementFactoryComponent to recreate from prefab
                result = self._element_factory.handle_create_element_from_prefab(
                    prefab_name=prefab_name,
                    element_id=element_id,
                    element_config=element_config if element_config else None,
                    component_config_overrides=component_config if component_config else None,
                    mount_id_override=mount_id
                )
                
                if result.get('success'):
                    logger.info(f"[{self.id}] ✓ Successfully recreated element {element_id} from prefab {prefab_name}")
                    return True
                else:
                    logger.warning(f"[{self.id}] Failed to recreate element from prefab: {result.get('error')}")
                    return False
                    
            else:
                # Try to recreate using basic element information
                # This is more limited but handles cases where prefab info isn't available
                logger.info(f"[{self.id}] Attempting basic element recreation for {element_type}")
                
                # For now, we can't fully recreate arbitrary elements without more information
                # This would require a more sophisticated element serialization system
                logger.warning(f"[{self.id}] Basic element recreation not fully implemented for {element_type}")
                return False
                
        except Exception as e:
            logger.error(f"[{self.id}] Error during element recreation: {e}", exc_info=True)
            return False
    
    async def _replay_element_unmounted(self, event_data: Dict[str, Any]) -> bool:
        """Replay element_unmounted event."""
        mount_id = event_data.get('mount_id')
        logger.info(f"[{self.id}] REPLAY: element_unmounted - {mount_id}")
        
        # Check if element is still mounted and unmount it
        if self._container and self._container.get_mounted_element(mount_id):
            return self._container.unmount_element(mount_id)
        else:
            logger.debug(f"[{self.id}] Element {mount_id} not mounted, nothing to unmount")
            return True
    
    async def _replay_element_created_from_prefab(self, event_data: Dict[str, Any]) -> bool:
        """Replay element_created_from_prefab event."""
        prefab_name = event_data.get('prefab_name')
        element_id = event_data.get('new_element_id')
        mount_id = event_data.get('mount_id')
        logger.info(f"[{self.id}] REPLAY: element_created_from_prefab - {prefab_name} -> {element_id}")
        
        # Check if element already exists
        if self._container and self._container.get_mounted_element(mount_id):
            logger.debug(f"[{self.id}] Element {mount_id} already exists, skipping prefab replay")
            return True
            
        # Try to recreate using factory
        if not self._element_factory:
            logger.warning(f"[{self.id}] Cannot replay prefab creation: No ElementFactoryComponent")
            return False
            
        # Note: This would require re-calling the factory with original parameters
        # which would need to be stored in the event data
        logger.warning(f"[{self.id}] Prefab creation replay not fully implemented")
        return False  # For now, return False until full implementation
    
    def is_replay_in_progress(self) -> bool:
        """Check if event replay is currently in progress."""
        return self._replay_in_progress
    
    def get_replay_mode(self) -> EventReplayMode:
        """Get the current event replay mode."""
        return self._event_replay_mode
    
    def get_replayed_event_count(self) -> int:
        """Get the number of events that have been replayed."""
        return len(self._replayed_event_ids) 

    async def store_veil_snapshot(self) -> bool:
        """
        Store current VEIL cache as a snapshot to storage.
        Called during shutdown or periodically to preserve VEIL state.
        
        Returns:
            True if snapshot was stored successfully, False otherwise
        """
        if self._event_replay_mode not in [EventReplayMode.ENABLED_WITH_VEIL_SNAPSHOT]:
            # Only store snapshots when the mode supports it
            return True
            
        try:
            if not self._timeline or not hasattr(self._timeline, '_storage') or not self._timeline._storage:
                logger.debug(f"[{self.id}] Cannot store VEIL snapshot: Timeline storage not available")
                return False
                
            # Create snapshot data
            snapshot_data = {
                'veil_cache': self._flat_veil_cache.copy(),
                'snapshot_timestamp': time.time(),
                'space_id': self.id,
                'space_type': self.__class__.__name__,
                'element_count': len(self._container.get_mounted_elements()) if self._container else 0,
                'cache_size': len(self._flat_veil_cache),
                'metadata': {
                    'is_inner_space': self.IS_INNER_SPACE,
                    'is_uplink_space': self.IS_UPLINK_SPACE,
                    'adapter_id': getattr(self, 'adapter_id', None),
                    'external_conversation_id': getattr(self, 'external_conversation_id', None)
                }
            }
            
            # Store snapshot using timeline's storage
            storage_key = f"veil_snapshot_{self.id}"
            success = await self._timeline._storage.store_system_state(storage_key, snapshot_data)
            
            if success:
                logger.info(f"[{self.id}] VEIL snapshot stored successfully. Cache size: {len(self._flat_veil_cache)}")
            else:
                logger.error(f"[{self.id}] Failed to store VEIL snapshot")
                
            return success
            
        except Exception as e:
            logger.error(f"[{self.id}] Error storing VEIL snapshot: {e}", exc_info=True)
            return False

    async def _restore_veil_snapshot(self) -> bool:
        """
        Restore VEIL cache from stored snapshot.
        Called during replay to restore previous VEIL state.
        
        Returns:
            True if snapshot was restored successfully, False otherwise
        """
        try:
            if not self._timeline or not hasattr(self._timeline, '_storage') or not self._timeline._storage:
                logger.debug(f"[{self.id}] Cannot restore VEIL snapshot: Timeline storage not available")
                return False
                
            # Load snapshot from storage
            storage_key = f"veil_snapshot_{self.id}"
            snapshot_data = await self._timeline._storage.load_system_state(storage_key)
            
            if not snapshot_data:
                logger.info(f"[{self.id}] No VEIL snapshot found for restoration")
                return False
                
            # Validate snapshot compatibility
            if not self._validate_veil_snapshot(snapshot_data):
                logger.warning(f"[{self.id}] VEIL snapshot validation failed, cannot restore")
                return False
                
            # Restore VEIL cache
            restored_cache = snapshot_data.get('veil_cache', {})
            if not isinstance(restored_cache, dict):
                logger.error(f"[{self.id}] Invalid VEIL cache format in snapshot")
                return False
                
            self._flat_veil_cache.clear()
            self._flat_veil_cache.update(restored_cache)
            
            snapshot_timestamp = snapshot_data.get('snapshot_timestamp', 0)
            cache_size = len(self._flat_veil_cache)
            
            logger.info(f"[{self.id}] VEIL snapshot restored successfully. Cache size: {cache_size}, "
                       f"Snapshot age: {time.time() - snapshot_timestamp:.1f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.id}] Error restoring VEIL snapshot: {e}", exc_info=True)
            return False

    def _validate_veil_snapshot(self, snapshot_data: Dict[str, Any]) -> bool:
        """
        Validate that a VEIL snapshot is compatible with current space state.
        
        Args:
            snapshot_data: The snapshot data to validate
            
        Returns:
            True if snapshot is valid and compatible, False otherwise
        """
        try:
            # Check basic snapshot structure
            required_fields = ['veil_cache', 'snapshot_timestamp', 'space_id']
            for field in required_fields:
                if field not in snapshot_data:
                    logger.warning(f"[{self.id}] VEIL snapshot missing required field: {field}")
                    return False
                    
            # Verify space ID matches
            if snapshot_data['space_id'] != self.id:
                logger.warning(f"[{self.id}] VEIL snapshot space ID mismatch: {snapshot_data['space_id']} != {self.id}")
                return False
                
            # Check snapshot age (don't restore very old snapshots)
            max_age_seconds = 7 * 24 * 3600  # 7 days
            snapshot_age = time.time() - snapshot_data.get('snapshot_timestamp', 0)
            if snapshot_age > max_age_seconds:
                logger.warning(f"[{self.id}] VEIL snapshot too old: {snapshot_age:.1f}s (max: {max_age_seconds}s)")
                return False
                
            # Validate VEIL cache structure
            veil_cache = snapshot_data.get('veil_cache', {})
            if not isinstance(veil_cache, dict):
                logger.warning(f"[{self.id}] VEIL snapshot cache is not a dictionary")
                return False
                
            # Basic validation: check that cache contains expected root node
            expected_root_id = f"{self.id}_space_root"
            if expected_root_id not in veil_cache:
                logger.warning(f"[{self.id}] VEIL snapshot missing expected root node: {expected_root_id}")
                # Don't fail validation for this - root can be regenerated
                
            logger.debug(f"[{self.id}] VEIL snapshot validation passed. Cache size: {len(veil_cache)}")
            return True
            
        except Exception as e:
            logger.error(f"[{self.id}] Error validating VEIL snapshot: {e}", exc_info=True)
            return False

    async def shutdown_with_veil_snapshot(self) -> bool:
        """
        Perform shutdown operations including storing VEIL snapshot if enabled.
        This should be called before system shutdown to preserve VEIL state.
        
        Returns:
            True if shutdown completed successfully, False otherwise
        """
        try:
            logger.info(f"[{self.id}] Beginning shutdown sequence")
            
            # Store VEIL snapshot if enabled
            if self._event_replay_mode == EventReplayMode.ENABLED_WITH_VEIL_SNAPSHOT:
                logger.info(f"[{self.id}] Storing VEIL snapshot before shutdown")
                snapshot_success = await self.store_veil_snapshot()
                if not snapshot_success:
                    logger.warning(f"[{self.id}] Failed to store VEIL snapshot during shutdown")
                else:
                    logger.info(f"[{self.id}] VEIL snapshot stored successfully during shutdown")
            
            # Perform any additional shutdown operations here
            # (component cleanup, connection closure, etc.)
            
            logger.info(f"[{self.id}] Shutdown sequence completed")
            return True
            
        except Exception as e:
            logger.error(f"[{self.id}] Error during shutdown: {e}", exc_info=True)
            return False
    
    def has_veil_snapshot_enabled(self) -> bool:
        """
        Check if VEIL snapshot functionality is enabled for this space.
        
        Returns:
            True if VEIL snapshots are enabled, False otherwise
        """
        return self._event_replay_mode == EventReplayMode.ENABLED_WITH_VEIL_SNAPSHOT 

    async def _connect_uplinks_after_replay(self) -> bool:
        """
        Connect uplinks after event replay to ensure recreated uplinks are connected to SharedSpaces and sync their content.
        This is crucial because during timeline replay, uplinks are recreated but not automatically connected.
        
        Returns:
            True if connection attempts were made, False if no uplinks found or errors occurred
        """
        try:
            logger.info(f"[{self.id}] Searching for uplink elements to connect after replay")
            
            if not self._container:
                logger.debug(f"[{self.id}] No container component, skipping uplink connection")
                return True
                
            mounted_elements = self._container.get_mounted_elements()
            uplink_count = 0
            connected_count = 0
            
            for mount_id, element in mounted_elements.items():
                # Check if this is an uplink element
                if hasattr(element, 'IS_UPLINK_SPACE') and element.IS_UPLINK_SPACE:
                    uplink_count += 1
                    remote_space_id = getattr(element, 'remote_space_id', 'Unknown')
                    logger.info(f"[{self.id}] Found uplink element {mount_id} -> {remote_space_id}")
                    
                    # Get the connection component and attempt to connect
                    connection_component = getattr(element, '_connection_component', None)
                    if connection_component and hasattr(connection_component, 'connect'):
                        try:
                            success = connection_component.connect()
                            if success:
                                connected_count += 1
                                logger.info(f"[{self.id}] ✓ Successfully connected uplink {mount_id} to {remote_space_id}")
                                
                                # After connection, trigger a sync to populate cache
                                cache_component = getattr(element, '_cache_component', None)
                                if cache_component and hasattr(cache_component, 'sync_remote_state'):
                                    sync_success = cache_component.sync_remote_state()
                                    if sync_success:
                                        logger.info(f"[{self.id}] ✓ Successfully synced cache for uplink {mount_id}")
                                    else:
                                        logger.warning(f"[{self.id}] ⚠ Failed to sync cache for uplink {mount_id}")
                                else:
                                    logger.debug(f"[{self.id}] No cache component found for uplink {mount_id}")
                            else:
                                logger.warning(f"[{self.id}] ✗ Failed to connect uplink {mount_id} to {remote_space_id}")
                        except Exception as e:
                            logger.error(f"[{self.id}] Error connecting uplink {mount_id}: {e}", exc_info=True)
                    else:
                        logger.warning(f"[{self.id}] Uplink element {mount_id} missing connection component")
            
            if uplink_count > 0:
                logger.info(f"[{self.id}] Uplink connection results: {connected_count}/{uplink_count} connected successfully")
            else:
                logger.debug(f"[{self.id}] No uplink elements found to connect")
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.id}] Error during uplink connection after replay: {e}", exc_info=True)
            return False 

    async def _verify_replay_integrity(self) -> bool:
        """
        Verify replay integrity across all components that should have state restored.
        
        Returns:
            True if verification was successful, False otherwise
        """
        try:
            logger.info(f"[{self.id}] Verifying replay integrity after event replay")
            
            # Check mounted elements for message components
            if self._container:
                mounted_elements = self._container.get_mounted_elements()
                logger.info(f"[{self.id}] Checking {len(mounted_elements)} mounted elements for replay integrity")
                
                for mount_id, element in mounted_elements.items():
                    try:
                        # Check if element has MessageListComponent
                        from .components.messaging.message_list import MessageListComponent
                        message_list_comp = element.get_component_by_type(MessageListComponent)
                        
                        if message_list_comp:
                            # Verify message integrity
                            stats = message_list_comp.get_message_statistics()
                            integrity = message_list_comp.verify_replay_integrity()
                            
                            logger.info(f"[{self.id}] Element {mount_id} has {stats['total_messages']} messages from {stats['unique_senders']} senders")
                            
                            if not integrity['integrity_ok']:
                                logger.warning(f"[{self.id}] Element {mount_id} message integrity issues: {integrity['issues']}")
                            else:
                                logger.debug(f"[{self.id}] Element {mount_id} message integrity verified")
                        else:
                            logger.debug(f"[{self.id}] Element {mount_id} has no MessageListComponent")
                            
                    except Exception as e:
                        logger.warning(f"[{self.id}] Error checking element {mount_id} replay integrity: {e}")
            else:
                logger.warning(f"[{self.id}] No container component found for integrity verification")
            
            logger.info(f"[{self.id}] Replay integrity verification completed")
            return True
            
        except Exception as e:
            logger.error(f"[{self.id}] Error during replay integrity verification: {e}", exc_info=True)
            return False 

    def get_flat_veil_cache_snapshot(self) -> Dict[str, Any]:
        """
        Returns a deep copy of the entire flat VEIL cache for external analysis.
        
        Returns:
            Deep copy of the flat VEIL cache
        """
        logger.debug(f"[{self.id}] Providing flat VEIL cache snapshot. Size: {len(self._flat_veil_cache)}")
        return copy.deepcopy(self._flat_veil_cache)
    
    def get_veil_nodes_by_owner(self, owner_id: str) -> Dict[str, Any]:
        """
        Get all VEIL nodes belonging to a specific owner element.
        
        This enables efficient granular access without full VEIL tree reconstruction.
        
        Args:
            owner_id: Element ID to filter by
            
        Returns:
            Dictionary of {veil_id: veil_node} for nodes owned by the specified element
        """
        filtered_nodes = {}
        
        for veil_id, node_data in self._flat_veil_cache.items():
            if isinstance(node_data, dict):
                props = node_data.get("properties", {})
                if props.get("owner_id") == owner_id:
                    filtered_nodes[veil_id] = copy.deepcopy(node_data)
        
        logger.debug(f"[{self.id}] Filtered {len(filtered_nodes)} VEIL nodes for owner {owner_id}")
        return filtered_nodes
    
    def get_veil_nodes_by_type(self, node_type: str, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all VEIL nodes of a specific type, optionally filtered by owner.
        
        Args:
            node_type: VEIL node type to filter by
            owner_id: Optional owner ID to further filter by
            
        Returns:
            Dictionary of {veil_id: veil_node} matching the criteria
        """
        filtered_nodes = {}
        
        for veil_id, node_data in self._flat_veil_cache.items():
            if isinstance(node_data, dict):
                if node_data.get("node_type") == node_type:
                    # If owner_id is specified, also filter by owner
                    if owner_id is None:
                        filtered_nodes[veil_id] = copy.deepcopy(node_data)
                    else:
                        props = node_data.get("properties", {})
                        if props.get("owner_id") == owner_id:
                            filtered_nodes[veil_id] = copy.deepcopy(node_data)
        
        logger.debug(f"[{self.id}] Filtered {len(filtered_nodes)} VEIL nodes of type '{node_type}'" + 
                    (f" for owner {owner_id}" if owner_id else ""))
        return filtered_nodes
    
    def get_veil_nodes_by_content_nature(self, content_nature: str, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all VEIL nodes with specific content_nature, optionally filtered by owner.
        
        Args:
            content_nature: Content nature to filter by (e.g., "chat_message", "attachment_content")
            owner_id: Optional owner ID to further filter by
            
        Returns:
            Dictionary of {veil_id: veil_node} matching the criteria
        """
        filtered_nodes = {}
        
        for veil_id, node_data in self._flat_veil_cache.items():
            if isinstance(node_data, dict):
                props = node_data.get("properties", {})
                if props.get("content_nature") == content_nature:
                    # If owner_id is specified, also filter by owner
                    if owner_id is None:
                        filtered_nodes[veil_id] = copy.deepcopy(node_data)
                    else:
                        if props.get("owner_id") == owner_id:
                            filtered_nodes[veil_id] = copy.deepcopy(node_data)
        
        logger.debug(f"[{self.id}] Filtered {len(filtered_nodes)} VEIL nodes with content_nature '{content_nature}'" + 
                    (f" for owner {owner_id}" if owner_id else ""))
        return filtered_nodes
    
    def has_multimodal_content(self, owner_id: Optional[str] = None) -> bool:
        """
        Check if the space contains multimodal content (attachment nodes with content).
        
        Args:
            owner_id: Optional owner ID to filter by
            
        Returns:
            True if multimodal content is found, False otherwise
        """
        for veil_id, node_data in self._flat_veil_cache.items():
            if isinstance(node_data, dict):
                props = node_data.get("properties", {})
                
                # Check if it's an attachment content node
                if (props.get("content_nature", "").startswith("image") or 
                    props.get("structural_role") == "attachment_content" or
                    node_data.get("node_type") == "attachment_content_item"):
                    
                    # If owner_id filter is specified, check ownership
                    if owner_id is None or props.get("owner_id") == owner_id:
                        logger.debug(f"[{self.id}] Found multimodal content in node {veil_id}")
                        return True
        
        return False