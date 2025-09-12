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

# NEW: Event phase classification for two-phase replay
# Phase 1: Structural events (system skeleton)
STRUCTURAL_EVENTS = {
    'element_mounted', 'element_unmounted', 'component_initialized',
    'element_created_from_prefab', 'tool_provider_registered',
    'component_state_updated', 'activation_call'
}

# Phase 2: Content events (data and conversations)
CONTENT_EVENTS = {
    'message_received', 'historical_message_received', 'agent_message_confirmed',
    'message_deleted', 'message_updated',
    'reaction_added', 'reaction_removed',
    'attachment_content_available', 'connectome_message_send_confirmed',
    'connectome_message_send_failed', 'agent_response_generated'
}

# NEW: Default replayability for event types (fallback when is_replayable flag not set)
# These are only used when events don't have explicit is_replayable flag
DEFAULT_REPLAYABLE_EVENTS = STRUCTURAL_EVENTS | CONTENT_EVENTS

DEFAULT_NON_REPLAYABLE_EVENTS = {
    # Runtime-only events that shouldn't be replayed
    'timeline_created', 'veil_delta_operations_received_by_space', 'action_executed',
    'agent_context_generated', 'tool_action_dispatched', 'tool_result_received',
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

        if not self.IS_UPLINK_SPACE:
            self._veil_producer: Optional[SpaceVeilProducer] = self.add_component(SpaceVeilProducer)

        # NEW: Event replay configuration
        self._event_replay_mode = self._determine_replay_mode()
        self._replay_in_progress = False
        self._replayed_event_ids = set()  # Track which events have been replayed

        # NEW: Two-phase replay state tracking
        self._current_replay_phase = None  # Track current phase: 'structural', 'content', or None
        self._structural_phase_complete = False
        self._content_phase_complete = False

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
        # NEW: Initialize uplink listeners set for VEILFacet operations
        self._uplink_listeners: Set[Callable[[List], None]] = set()  # Now handles VEILFacetOperations
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
    
    def on_registered(self) -> None:
        """
        Called by SpaceRegistry when this space is successfully registered.
        
        This triggers the Godot-inspired ready cascade, ensuring all child elements
        and their components become ready in proper dependency order.
        """
        logger.info(f"[{self.id}] Space registered in SpaceRegistry, triggering ready cascade")
        self._ready()

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

    def _classify_event_phase(self, event_type: str, event_payload: Optional[Dict[str, Any]] = None) -> str:
        """
        Classify an event into structural or content phase.

        Args:
            event_type: The type of event to classify
            event_payload: Optional event payload for override checking

        Returns:
            'structural' or 'content' phase classification
        """
        # Check for explicit phase override in event payload
        if event_payload and 'replay_phase' in event_payload:
            override_phase = event_payload['replay_phase']
            if override_phase in ['structural', 'content']:
                logger.debug(f"[{self.id}] Event {event_type} has explicit phase override: {override_phase}")
                return override_phase

        # Automatic classification based on event type
        if event_type in STRUCTURAL_EVENTS:
            return 'structural'
        elif event_type in CONTENT_EVENTS:
            return 'content'
        else:
            # Default: treat unknown events as content events to maintain safety
            logger.debug(f"[{self.id}] Unknown event type {event_type}, classifying as content")
            return 'content'

    def _ensure_own_veil_presence_initialized(self) -> None:
        """
        Ensures that the Space's own root VEIL node is present via SpaceVeilProducer.
        This is crucial to call before processing any deltas that might reference it as a parent,
        and also ensures it's there if on_frame_end triggers its own producer.
        """
        if not self.IS_UPLINK_SPACE and self._veil_producer:
            space_root_id = f"{self.id}_space_root"

            # Check if root exists in SpaceVeilProducer's cache
            veil_cache = self._veil_producer.get_flat_veil_cache()
            if space_root_id not in veil_cache:
                logger.warning(f"[{self.id}] Root VEIL node '{space_root_id}' not found in SpaceVeilProducer cache. Triggering emit_delta.")
                self._veil_producer.emit_delta()
                logger.info(f"[{self.id}] Initialized own root VEIL node '{space_root_id}' via SpaceVeilProducer.")
            else:
                logger.debug(f"[{self.id}] Own root VEIL node '{space_root_id}' already present in SpaceVeilProducer cache.")
        elif self.IS_UPLINK_SPACE:
            logger.debug(f"[{self.id}] Skipping VEIL initialization for UplinkSpace.")
        else:
            logger.warning(f"[{self.id}] Cannot ensure VEIL presence: SpaceVeilProducer not available.")

    # --- Container Methods (Delegated) ---
    def mount_element(self,
                      element: BaseElement,
                      mount_id: Optional[str] = None,
                      mount_type: MountType = MountType.INCLUSION,
                      creation_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
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

    # --- NEW: VEIL Operations ---
    def receive_delta(self, delta_operations: List[Dict[str, Any]]) -> None:
        """
        Receive VEIL facet operations from child elements and route to SpaceVeilProducer.
        
        Args:
            delta_operations: List of VEILFacetOperation instances
        """
        if not self._veil_producer:
            logger.critical(f"🔨🏠 [{self.id}] No SpaceVeilProducer available - VEIL operations lost!")
            return
            
        # Route to SpaceVeilProducer for processing
        try:
            import asyncio
            
            # SpaceVeilProducer.receive_facet_operations is async, so we need to handle it properly
            loop = asyncio.get_running_loop()
            loop.create_task(self._veil_producer.receive_facet_operations(delta_operations))
            
        except Exception as e:
            logger.error(f"Error routing VEIL operations to SpaceVeilProducer: {e}", exc_info=True)

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
        """
        event_type = event_payload.get("event_type")

        # NEW: Check if this is a replay mode to prevent double-recording
        is_replay_mode = timeline_context.get('replay_mode', False)
        replay_phase = timeline_context.get('replay_phase', 'unknown')

        logger.debug(f"[{self.id}] Receiving event: Type='{event_type}', Timeline='{timeline_context.get('timeline_id')}', Replay={is_replay_mode}, Phase={replay_phase}")

        # 1. Add event to the timeline via TimelineComponent (unless in replay mode)
        new_event_id = None
        if not is_replay_mode:
            # Decider classification (persist decisions for determinism)
            try:
                # Prefer InterruptDecider for pre-classification
                interrupt_decider = None
                activation_decider = None
                for _name, comp in self.get_components().items():
                    ctype = getattr(comp, 'COMPONENT_TYPE', '')
                    if ctype == 'InterruptDeciderComponent':
                        interrupt_decider = comp
                    elif ctype == 'ActivationDeciderComponent':
                        activation_decider = comp
                decider = activation_decider or interrupt_decider
            except Exception:
                decider = None
            if interrupt_decider and isinstance(event_payload, dict):
                try:
                    interrupt_decision = interrupt_decider.classify_interrupt(event_payload)
                    if isinstance(interrupt_decision, dict):
                        # Store under event_payload.decider.interrupt_class
                        event_payload.setdefault('decider', {}).update(interrupt_decision)
                except Exception:
                    pass
            # ActivationDecider can also produce activation hints per event
            if activation_decider and isinstance(event_payload, dict):
                try:
                    act_decision = activation_decider.classify_event(event_payload)
                    if isinstance(act_decision, dict):
                        event_payload.setdefault('decider', {}).update(act_decision)
                except Exception:
                    pass

            new_event_id = self.add_event_to_timeline(event_payload, timeline_context)
            if not new_event_id:
                logger.error(f"[{self.id}] Failed to add event to timeline. Aborting processing for this event. Event: {event_payload}")
                return
            # Remember decisions and routing context
            try:
                if activation_decider and event_payload.get('decider') and hasattr(activation_decider, 'remember_decision'):
                    activation_decider.remember_decision(new_event_id, event_payload.get('decider'))
                if activation_decider and hasattr(activation_decider, 'remember_event_context'):
                    ctx = {
						'adapter_id': event_payload.get('source_adapter_id') or event_payload.get('payload', {}).get('source_adapter_id'),
						'conversation_id': event_payload.get('external_conversation_id') or event_payload.get('payload', {}).get('external_conversation_id'),
						'is_dm': bool(event_payload.get('payload', {}).get('is_dm', False)),
						# NEW: sender info for conditional rules like typing_from_last_activator
						'sender_id': event_payload.get('payload', {}).get('sender_id') or event_payload.get('payload', {}).get('user_id') or event_payload.get('payload', {}).get('author_id')
					}
                activation_decider.remember_event_context(new_event_id, ctx)
            except Exception:
                pass
        else:
			# In replay mode, use a dummy event ID for processing
            new_event_id = f"replay_{event_type}_{int(time.time() * 1000)}"
			# Remember persisted activation decision for ack-based activation
            try:
                activation_decider = None
                for _name, comp in self.get_components().items():
                    if getattr(comp, 'COMPONENT_TYPE', '') == 'ActivationDeciderComponent':
                        activation_decider = comp
                    break
                if activation_decider and isinstance(event_payload, dict) and event_payload.get('decider'):
                    if hasattr(activation_decider, 'remember_decision'):
                        activation_decider.remember_decision(new_event_id, event_payload.get('decider'))
                    if hasattr(activation_decider, 'remember_event_context'):
                        ctx = {
							'adapter_id': event_payload.get('source_adapter_id') or event_payload.get('payload', {}).get('source_adapter_id'),
							'conversation_id': event_payload.get('external_conversation_id') or event_payload.get('payload', {}).get('external_conversation_id'),
							'is_dm': bool(event_payload.get('payload', {}).get('is_dm', False)),
							# NEW: sender info for conditional rules like typing_from_last_activator
							'sender_id': event_payload.get('payload', {}).get('sender_id') or event_payload.get('payload', {}).get('user_id') or event_payload.get('payload', {}).get('author_id')
						}
                        activation_decider.remember_event_context(new_event_id, ctx)
            except Exception:
                pass

        # NEW: Handle adapter mapping registration for message events (only for InnerSpaces)
        if self.IS_INNER_SPACE:
            self._register_adapter_mapping_from_event(event_payload)

        # Construct the full event node as it exists in the DAG for handlers
        full_event_node = {
             'id': new_event_id,
             'timeline_id': timeline_context.get('timeline_id', self.get_primary_timeline()),
             'payload': event_payload.copy()
        }

        # NEW: Handle system events directly (bypass heartbeat classification)
        # These events represent internal system decisions/acknowledgments, not external content
        if event_type in ["activation_call", "component_processed"]:
            self.process_event_for_components(full_event_node, timeline_context)
            return

        # Route event to Heartbeat queues instead of immediate broadcast
        try:
            heartbeat = None
            for _name, comp in self.get_components().items():
                if getattr(comp, 'COMPONENT_TYPE', '') == 'HeartbeatComponent':
                    heartbeat = comp
                    break
            if heartbeat:
                interrupt_class = (event_payload or {}).get('decider', {}).get('interrupt_class') if isinstance(event_payload, dict) else None
                
                loop = asyncio.get_running_loop()
                if interrupt_class == 'interrupt':
                    loop.create_task(heartbeat.handle_interrupt_now(full_event_node, timeline_context))
                else:
                    loop.create_task(heartbeat.enqueue_normal(full_event_node, timeline_context))
            else:
                # Fallback to immediate processing if heartbeat missing
                self.process_event_for_components(full_event_node, timeline_context)
        except Exception as e:
            logger.debug(f"[{self.id}] Heartbeat routing error, falling back to immediate processing: {e}")
            self.process_event_for_components(full_event_node, timeline_context)

    # New method: process event by broadcasting to own components and all mounted elements
    def process_event_for_components(self, event_payload: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        try:
            # 2. Dispatch event to own components
            handled_by_component = False
            for comp_name, component in self.get_components().items():
                if hasattr(component, 'handle_event') and callable(component.handle_event):
                    try:
                        if component.handle_event(event_payload, timeline_context):
                            handled_by_component = True
                    except Exception as comp_err:
                        logger.error(f"[{self.id}] Error in component '{comp_name}' handling event: {comp_err}", exc_info=True)
            # 3. Broadcast to all mounted child elements
            try:
                mounted_elements = self._container.get_mounted_elements() if self._container else {}  
                for elem in mounted_elements.values():
                    if hasattr(elem, 'receive_event') and callable(elem.receive_event):
                        elem.receive_event(event_payload, timeline_context)
            except Exception as e:
                logger.error(f"[{self.id}] Error broadcasting event to mounted elements: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"[{self.id}] process_event_for_components error: {e}", exc_info=True)

    async def execute_action_on_element(self,
                                        element_id: str,
                                        action_name: str,
                                        parameters: Dict[str, Any],
                                        calling_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

    # --- NEW: Event Replay Methods ---

    async def replay_events_from_timeline(self) -> bool:
        """
        NEW: Two-phase replay timeline events to reconstruct system state with proper chronological ordering.

        Phase 1: Structural Restoration - Mount elements and initialize components
        Phase 2: Content Restoration - Process content events chronologically

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
            logger.info(f"[{self.id}] Starting two-phase event replay for system state reconstruction")

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

            logger.info(f"[{self.id}] Found {len(chronological_events)} events for two-phase replay")

            # Phase 1: Structural Replay
            logger.info(f"[{self.id}] === Phase 1: Structural Replay Starting ===")
            structural_success = await self._replay_structural_phase(chronological_events)
            if not structural_success:
                logger.error(f"[{self.id}] Phase 1: Structural replay failed")
                return False

            # Verify structural integrity before content
            logger.info(f"[{self.id}] Verifying structural integrity before content phase")
            structure_ready = await self._verify_structural_integrity()
            if not structure_ready:
                logger.error(f"[{self.id}] Structural integrity verification failed")
                return False

            # Phase 2: Content Replay
            logger.info(f"[{self.id}] === Phase 2: Content Replay Starting ===")
            content_success = await self._replay_content_phase(chronological_events)
            if not content_success:
                logger.error(f"[{self.id}] Phase 2: Content replay failed")
                return False

            # Post-replay operations
            await self._complete_two_phase_replay()

            logger.info(f"[{self.id}] ✓ Two-phase replay completed successfully")
            return True

        except Exception as e:
            logger.error(f"[{self.id}] Error during two-phase replay: {e}", exc_info=True)
            return False
        finally:
            self._replay_in_progress = False
            self._current_replay_phase = None

    async def _replay_structural_phase(self, all_events: List[Dict]) -> bool:
        """
        Phase 1: Replay only structural events to establish system skeleton.

        Args:
            all_events: All chronological events from timeline

        Returns:
            True if structural phase completed successfully, False otherwise
        """
        try:
            self._current_replay_phase = 'structural'

            # Filter to only structural events
            structural_events = [
                event for event in all_events
                if self._classify_event_phase(
                    event.get('payload', {}).get('event_type'),
                    event.get('payload', {})
                ) == 'structural'
            ]

            logger.info(f"[{self.id}] Phase 1: Processing {len(structural_events)} structural events")

            replayed_count = 0
            skipped_count = 0

            for event_node in structural_events:
                event_id = event_node.get('id')
                event_payload = event_node.get('payload', {})
                event_type = event_payload.get('event_type')

                # Check if this event should be replayed
                if not self._should_replay_event(event_type, event_payload):
                    logger.debug(f"[{self.id}] Phase 1: Skipping non-replayable event {event_id} ({event_type})")
                    skipped_count += 1
                    continue

                logger.info(f"[{self.id}] Phase 1: Replaying structural event {event_id}: {event_type}")

                # Replay the structural event
                success = await self._replay_single_event(event_node, phase="structural")
                if success:
                    self._replayed_event_ids.add(event_id)
                    replayed_count += 1
                    logger.info(f"[{self.id}] Phase 1: ✓ Successfully replayed {event_id} ({event_type})")
                else:
                    logger.error(f"[{self.id}] Phase 1: ✗ Failed to replay {event_id} ({event_type})")
                    return False  # Fail fast for structural issues

            self._structural_phase_complete = True
            logger.info(f"[{self.id}] Phase 1: Structural replay completed - {replayed_count} replayed, {skipped_count} skipped")
            return True

        except Exception as e:
            logger.error(f"[{self.id}] Phase 1: Error during structural replay: {e}", exc_info=True)
            return False

    async def _replay_content_phase(self, all_events: List[Dict]) -> bool:
        """
        Phase 2: Replay only content events chronologically with proper VEIL facet timing.

        Args:
            all_events: All chronological events from timeline

        Returns:
            True if content phase completed successfully, False otherwise
        """
        try:
            self._current_replay_phase = 'content'

            # Filter to only content events (preserve chronological order)
            content_events = [
                event for event in all_events
                if self._classify_event_phase(
                    event.get('payload', {}).get('event_type'),
                    event.get('payload', {})
                ) == 'content'
            ]

            logger.info(f"[{self.id}] Phase 2: Processing {len(content_events)} content events chronologically")

            replayed_count = 0
            skipped_count = 0

            for event_node in content_events:
                event_id = event_node.get('id')
                event_payload = event_node.get('payload', {})
                event_type = event_payload.get('event_type')

                # Check if this event should be replayed
                if not self._should_replay_event(event_type, event_payload):
                    logger.debug(f"[{self.id}] Phase 2: Skipping non-replayable event {event_id} ({event_type})")
                    skipped_count += 1
                    continue

                logger.info(f"[{self.id}] Phase 2: Replaying content event {event_id}: {event_type}")

                # Replay the content event
                success = await self._replay_single_event(event_node, phase="content")
                if success:
                    self._replayed_event_ids.add(event_id)
                    replayed_count += 1
                    logger.info(f"[{self.id}] Phase 2: ✓ Successfully replayed {event_id} ({event_type})")
                else:
                    logger.warning(f"[{self.id}] Phase 2: ⚠ Failed to replay {event_id} ({event_type}) - continuing")
                    skipped_count += 1
                    # Continue with other content events even if one fails

            self._content_phase_complete = True
            logger.info(f"[{self.id}] Phase 2: Content replay completed - {replayed_count} replayed, {skipped_count} skipped")
            return True

        except Exception as e:
            logger.error(f"[{self.id}] Phase 2: Error during content replay: {e}", exc_info=True)
            return False

    async def _verify_structural_integrity(self) -> bool:
        """
        Verify all expected elements and components exist before content replay.

        Returns:
            True if structural integrity is verified, False otherwise
        """
        try:
            logger.info(f"[{self.id}] Verifying structural integrity after Phase 1")

            # Check that all expected chat elements exist based on timeline events
            expected_elements = self._extract_expected_elements_from_timeline()

            missing_elements = []
            incomplete_elements = []

            for element_info in expected_elements:
                mount_id = element_info.get('mount_id')
                element_type = element_info.get('element_type', 'Unknown')

                element = self.get_mounted_element(mount_id) if mount_id else None

                if not element:
                    missing_elements.append(f"{mount_id} ({element_type})")
                    continue

                # Verify element has expected components based on type
                if element_type in ['ConversationElement', 'ChatElement']:
                    required_components = ['MessageListComponent', 'MessageListVeilProducer']
                    for comp_type_name in required_components:
                        # Try to get component by name (more flexible than type)
                        comp = element.get_component(comp_type_name)
                        if not comp:
                            incomplete_elements.append(f"{mount_id} missing {comp_type_name}")

            # Report structural integrity results
            if missing_elements:
                logger.error(f"[{self.id}] Missing expected elements: {missing_elements}")
                return False

            if incomplete_elements:
                logger.warning(f"[{self.id}] Elements with missing components: {incomplete_elements}")
                # Don't fail for missing components - they might be optional

            total_elements = len(expected_elements)
            ready_elements = total_elements - len(missing_elements)

            logger.info(f"[{self.id}] Structural integrity verified: {ready_elements}/{total_elements} elements ready")
            return True

        except Exception as e:
            logger.error(f"[{self.id}] Structural integrity verification failed: {e}", exc_info=True)
            return False

    def _extract_expected_elements_from_timeline(self) -> List[Dict[str, Any]]:
        """
        Extract expected element information from timeline mount events.

        Returns:
            List of dictionaries with element information
        """
        expected_elements = []

        try:
            # Look for element_mounted events that were successfully replayed
            for event_id in self._replayed_event_ids:
                # This is a simplified approach - in a full implementation,
                # we'd need to track the events we just processed
                pass

            # For now, check what's currently mounted as a proxy
            if self._container:
                mounted_elements_info = self._container.get_mounted_elements_info()
                for mount_id, element_info in mounted_elements_info.items():
                    expected_elements.append({
                        'mount_id': mount_id,
                        'element_type': element_info.get('element_type', 'Unknown'),
                        'element_id': element_info.get('element_id')
                    })

        except Exception as e:
            logger.warning(f"[{self.id}] Error extracting expected elements: {e}")

        return expected_elements

    async def _complete_two_phase_replay(self) -> bool:
        """
        Complete post-replay operations with simplified VEIL handling.

        NEW: VEIL facets are now built in real-time during content phase,
        so no complex regeneration is needed!

        Returns:
            True if completion was successful, False otherwise
        """
        try:
            total_replayed = len(self._replayed_event_ids)
            logger.info(f"[{self.id}] Completing two-phase replay with {total_replayed} total events replayed")

            # Check final VEIL cache size (built during content phase)
            final_cache_size = self._veil_producer.get_facet_cache_size() if self._veil_producer else 0
            logger.info(f"[{self.id}] VEIL cache size after chronological replay: {final_cache_size}")

            # NEW: Simplified VEIL handling
            if self._event_replay_mode == EventReplayMode.ENABLED_WITH_VEIL_SNAPSHOT:
                # For snapshot mode, still try to restore stored VEIL
                logger.info(f"[{self.id}] Attempting VEIL cache restoration in snapshot mode")
                veil_restoration_success = await self._restore_veil_snapshot()
                if not veil_restoration_success:
                    logger.info(f"[{self.id}] VEIL cache restoration failed, using chronologically built VEIL from content phase")
            else:
                # For regular mode, VEIL was built chronologically during content phase - no regeneration needed!
                logger.info(f"[{self.id}] ✓ VEIL state built chronologically during content phase - no regeneration required")

            # Connect uplinks after replay (crucial for InnerSpaces)
            if total_replayed > 0:
                logger.info(f"[{self.id}] Connecting uplinks after two-phase replay")
                await self._connect_uplinks_after_replay()

            # Verify replay integrity for message components
            await self._verify_replay_integrity()

            return True

        except Exception as e:
            logger.error(f"[{self.id}] Error during two-phase replay completion: {e}", exc_info=True)
            return False

    async def _regenerate_veil_state_after_replay(self) -> bool:
        """
        Regenerate VEIL state after event replay to ensure VEIL cache reflects recreated elements.

        Returns:
            True if regeneration was successful, False otherwise
        """
        try:
            logger.info(f"[{self.id}] Regenerating VEIL state after event replay")

            # FIXED: Don't clear the entire cache - agent responses were already replayed!
            # Only regenerate VEIL for mounted elements that might not have been replayed
            # (e.g., status facets for containers)

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

            cache_size = self._veil_producer.get_facet_cache_size() if self._veil_producer else 0
            logger.info(f"[{self.id}] VEIL state regeneration completed. Cache size: {cache_size}")
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

    async def _replay_single_event(self, event_node: Dict[str, Any], phase: str = "unknown") -> bool:
        """
        Replay a single event to reconstruct system state with phase awareness.

        Args:
            event_node: The complete event node from timeline
            phase: The current replay phase ('structural', 'content', or 'unknown')

        Returns:
            True if replay was successful, False otherwise
        """
        event_payload = event_node.get('payload', {})
        event_type = event_payload.get('event_type')
        event_data = event_payload.get('data', {})

        try:
            # NEW: Enhanced timeline context with phase information
            timeline_context = {
                'timeline_id': event_node.get('timeline_id'),
                'replay_mode': True,  # Mark this as a replay to prevent double-recording
                'replay_phase': phase,  # NEW: Include current phase for component awareness
                'original_event_id': event_node.get('id'),
                'original_timestamp': event_node.get('timestamp')
            }

            # Handle activation_call events during replay (extract focus context)
            if event_type == "activation_call":
                self._handle_activation_call_during_replay(event_node)
                return True  # Don't replay the activation_call itself, just extract focus

            # Use receive_event for most replay scenarios
            # This ensures components get the chance to handle replayed events with phase awareness
            logger.debug(f"[{self.id}] Replaying {phase} event {event_type} via receive_event mechanism")
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
                logger.debug(f"[{self.id}] Event type {event_type} replayed via component mechanism in {phase} phase")
                return True

        except Exception as e:
            logger.error(f"[{self.id}] Error replaying {event_type} event in {phase} phase: {e}", exc_info=True)
            return False

    def _handle_activation_call_during_replay(self, event_node: Dict[str, Any]) -> None:
        """
        NEW: Extract and signal focus StatusFacet from activation_call during replay.

        Args:
            event_node: The activation_call event node containing focus context
        """
        try:
            focus_context = event_node.get('payload', {}).get('focus_context', {})
            focus_element_id = focus_context.get('focus_element_id')

            if focus_element_id:
                # Find the target element and trigger focus signal
                target_element = self.get_element_by_id(focus_element_id)
                if target_element:
                    # Get the MessageList component and trigger focus signal
                    message_list_component = target_element.get_component_by_type("MessageListComponent")
                    if message_list_component:
                        hud_component = message_list_component.owner.get_parent_object().get_hud()
                        if hud_component:
                            hud_component.record_focus_change(focus_context)
                            logger.debug(f"[REPLAY] Signaled focus change for {focus_element_id}")
                        else:
                            logger.debug(f"[REPLAY] HUDComponent not found for element {focus_element_id}")
                    else:
                        logger.debug(f"[REPLAY] MessageListComponent not found for element {focus_element_id}")
                else:
                    logger.debug(f"[REPLAY] Element {focus_element_id} not found for focus signal")
            else:
                logger.debug(f"[REPLAY] No focus_element_id in activation_call focus_context")

        except Exception as e:
            logger.error(f"[{self.id}] Error handling activation_call during replay: {e}", exc_info=True)

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
            veil_cache = self._veil_producer.get_flat_veil_cache() if self._veil_producer else {}
            snapshot_data = {
                'veil_cache': veil_cache,
                'snapshot_timestamp': time.time(),
                'space_id': self.id,
                'space_type': self.__class__.__name__,
                'element_count': len(self._container.get_mounted_elements()) if self._container else 0,
                'cache_size': len(veil_cache),
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
                cache_size = self._veil_producer.get_facet_cache_size() if self._veil_producer else 0
                logger.info(f"[{self.id}] VEIL snapshot stored successfully. Cache size: {cache_size}")
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

            if self._veil_producer:
                self._veil_producer.update_flat_veil_cache(restored_cache)
            else:
                logger.error(f"[{self.id}] Cannot restore VEIL cache: SpaceVeilProducer not available")
                return False

            snapshot_timestamp = snapshot_data.get('snapshot_timestamp', 0)
            cache_size = len(restored_cache)

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
        if self._veil_producer:
            veil_cache = self._veil_producer.get_flat_veil_cache()
            logger.debug(f"[{self.id}] Providing flat VEIL cache snapshot. Size: {len(veil_cache)}")
            return veil_cache
        else:
            logger.warning(f"[{self.id}] Cannot provide VEIL cache snapshot: SpaceVeilProducer not available")
            return {}

    def get_facets_by_owner(self, owner_id: str) -> Dict[str, 'VEILFacet']:
        """
        Get all VEILFacets belonging to a specific owner element.

        This enables efficient granular access to facets by owner.

        Args:
            owner_id: Element ID to filter by

        Returns:
            Dictionary of {facet_id: VEILFacet} for facets owned by the specified element
        """
        if self._veil_producer:
            return self._veil_producer.get_facets_by_owner(owner_id)
        else:
            logger.warning(f"[{self.id}] Cannot filter VEILFacets by owner: SpaceVeilProducer not available")
            return {}

    def get_veil_nodes_by_type(self, node_type: str, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all VEIL nodes of a specific type, optionally filtered by owner.

        Args:
            node_type: VEIL node type to filter by
            owner_id: Optional owner ID to further filter by

        Returns:
            Dictionary of {veil_id: veil_node} matching the criteria
        """
        if self._veil_producer:
            return self._veil_producer.get_veil_nodes_by_type(node_type, owner_id)
        else:
            logger.warning(f"[{self.id}] Cannot filter VEIL nodes by type: SpaceVeilProducer not available")
            return {}

    def get_veil_nodes_by_content_nature(self, content_nature: str, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all VEIL nodes with specific content_nature, optionally filtered by owner.

        Args:
            content_nature: Content nature to filter by (e.g., "chat_message", "attachment_content")
            owner_id: Optional owner ID to further filter by

        Returns:
            Dictionary of {veil_id: veil_node} matching the criteria
        """
        if self._veil_producer:
            return self._veil_producer.get_veil_nodes_by_content_nature(content_nature, owner_id)
        else:
            logger.warning(f"[{self.id}] Cannot filter VEIL nodes by content nature: SpaceVeilProducer not available")
            return {}

    def has_multimodal_content(self, owner_id: Optional[str] = None) -> bool:
        """
        Check if the space contains multimodal content (attachment nodes with content).

        Args:
            owner_id: Optional owner ID to filter by

        Returns:
            True if multimodal content is found, False otherwise
        """
        if self._veil_producer:
            return self._veil_producer.has_multimodal_content(owner_id)
        else:
            logger.warning(f"[{self.id}] Cannot check for multimodal content: SpaceVeilProducer not available")
            return False