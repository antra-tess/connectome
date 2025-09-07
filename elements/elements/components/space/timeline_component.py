"""
Timeline Component
Manages the Loom DAG (event history) for a Space.
"""
import logging
import time
import uuid
import asyncio
from typing import Dict, Any, Optional, List, Set

from ...base import Component
# Import the registry decorator
from elements.component_registry import register_component

# NEW: Import storage system
from storage import create_storage_from_env, StorageInterface

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMELINE_ID = "primary"

@register_component
class TimelineComponent(Component):
    """
    Manages the Loom DAG (event history) for the owning Space element.
    Handles adding new events, maintaining parent links, and tracking the primary timeline.

    NOTE: This is a simplified implementation focusing on a single primary timeline.
          Full Loom features (forking, merging, complex DAG traversal) are not yet implemented.

    NEW: Integrates with pluggable storage system to persist timeline events and DAG structure.
    """
    COMPONENT_TYPE = "TimelineComponent"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # NEW: Storage integration
        self._storage: Optional[StorageInterface] = None
        self._storage_initialized = False
        self._space_id: Optional[str] = None  # For storage key naming

    def initialize(self, **kwargs) -> None:
        """Initializes the component state, creating a default primary timeline."""
        super().initialize(**kwargs)
        # _timelines: { timeline_id: { "is_primary": bool, "head_event_ids": set[str] } }
        self._state.setdefault('_timelines', {})
        # _all_events: { event_id: { "id": str, "timestamp": float, "parent_ids": list[str], "timeline_id": str, "payload": dict } }
        self._state.setdefault('_all_events', {})
        self._state.setdefault('_primary_timeline_id', None)

        # NEW: Get Space ID for storage
        if self.owner:
            self._space_id = f"space_{self.owner.id}"

        # Create default primary timeline if none exists
        if not self._state['_timelines']:
            self._create_timeline(DEFAULT_TIMELINE_ID, is_primary=True)
            logger.info(f"TimelineComponent initialized for Element {self.owner.id}. Created default timeline '{DEFAULT_TIMELINE_ID}'.")
        else:
            logger.debug(f"TimelineComponent initialized for Element {self.owner.id}. Existing state loaded.")

        # NEW: Schedule async storage initialization
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._initialize_storage_async())
            logger.debug(f"Scheduled async storage initialization for TimelineComponent {self.id}")
        except RuntimeError:
            logger.debug(f"No event loop running, will initialize storage on first use for TimelineComponent {self.id}")

        return True

    async def _initialize_storage_async(self) -> bool:
        """Initialize storage backend asynchronously."""
        try:
            logger.debug(f"Initializing storage backend for TimelineComponent {self.id}")

            # Create storage from environment configuration
            self._storage = create_storage_from_env()

            # Initialize the storage backend
            success = await self._storage.initialize()
            if not success:
                logger.error(f"Failed to initialize storage backend for TimelineComponent")
                return False

            self._storage_initialized = True
            logger.info(f"Storage backend successfully initialized for TimelineComponent {self.id}")

            # Load existing timeline data from storage
            await self._load_timeline_from_storage()

            return True

        except Exception as e:
            logger.error(f"Error initializing storage backend for TimelineComponent: {e}", exc_info=True)
            return False

    async def _ensure_storage_ready(self) -> bool:
        """Ensure storage backend is initialized before use."""
        if self._storage_initialized:
            return True

        if self._storage is None:
            logger.debug(f"Storage not yet initialized for TimelineComponent {self.id}, initializing now...")
            success = await self._initialize_storage_async()
            return success

        return self._storage_initialized

    async def _load_timeline_from_storage(self) -> bool:
        """Load existing timeline data from storage."""
        if not self._storage_initialized or not self._space_id:
            return False

        try:
            logger.debug(f"Loading timeline data for space {self._space_id}")

            # Load timeline state from system state storage
            timeline_state = await self._storage.load_system_state(f"timeline_state_{self._space_id}")
            if timeline_state:
                # Restore timeline metadata, converting head_event_ids back to sets
                timelines = timeline_state.get('_timelines', {})
                for timeline_id, timeline_info in timelines.items():
                    head_event_ids = timeline_info.get('head_event_ids', [])
                    # Convert list back to set if needed
                    if isinstance(head_event_ids, list):
                        timeline_info['head_event_ids'] = set(head_event_ids)
                    elif isinstance(head_event_ids, str):
                        # Handle case where it was accidentally saved as string
                        timeline_info['head_event_ids'] = {head_event_ids}
                    elif not isinstance(head_event_ids, set):
                        timeline_info['head_event_ids'] = set()

                self._state['_timelines'] = timelines
                self._state['_primary_timeline_id'] = timeline_state.get('_primary_timeline_id')
                logger.info(f"Loaded timeline state with {len(self._state['_timelines'])} timelines")

            # Load all timeline events from storage
            timeline_events = await self._storage.load_system_state(f"timeline_events_{self._space_id}")
            if timeline_events:
                self._state['_all_events'] = timeline_events.get('_all_events', {})
                logger.info(f"Loaded {len(self._state['_all_events'])} timeline events")

            # NEW: Only trigger event replay if there are events to replay AND replay is enabled
            has_events_to_replay = len(self._state.get('_all_events', {})) > 0
            if has_events_to_replay and hasattr(self.owner, 'replay_events_from_timeline') and callable(self.owner.replay_events_from_timeline):
                logger.info(f"Triggering event replay for space {self._space_id} ({len(self._state['_all_events'])} events)")
                replay_success = await self.owner.replay_events_from_timeline()
                if replay_success:
                    logger.info(f"Event replay completed successfully for space {self._space_id}")
                else:
                    logger.warning(f"Event replay failed for space {self._space_id}")
            elif has_events_to_replay:
                logger.info(f"Event replay disabled for space {self._space_id} ({len(self._state['_all_events'])} events found but not replaying)")
            else:
                logger.info(f"No events found for space {self._space_id}, skipping replay (fresh agent)")

            return True

        except Exception as e:
            logger.error(f"Error loading timeline data from storage: {e}", exc_info=True)
            return False

    async def _persist_timeline_state(self) -> bool:
        """Persist current timeline state to storage."""
        if not await self._ensure_storage_ready() or not self._space_id:
            return False

        try:
            # Store timeline state (metadata about timelines)
            # Convert sets to lists for JSON serialization
            timelines_serializable = {}
            for timeline_id, timeline_info in self._state['_timelines'].items():
                timeline_copy = timeline_info.copy()
                # Convert set to list for serialization
                if isinstance(timeline_copy.get('head_event_ids'), set):
                    timeline_copy['head_event_ids'] = list(timeline_copy['head_event_ids'])
                timelines_serializable[timeline_id] = timeline_copy

            timeline_state = {
                '_timelines': timelines_serializable,
                '_primary_timeline_id': self._state['_primary_timeline_id'],
                'last_updated': time.time()
            }
            await self._storage.store_system_state(f"timeline_state_{self._space_id}", timeline_state)

            # Store all events (the actual event data)
            timeline_events = {
                '_all_events': self._state['_all_events'],
                'last_updated': time.time()
            }
            await self._storage.store_system_state(f"timeline_events_{self._space_id}", timeline_events)

            logger.debug(f"Persisted timeline state with {len(self._state['_all_events'])} events")
            return True

        except Exception as e:
            logger.error(f"Error persisting timeline state: {e}", exc_info=True)
            return False

    async def shutdown(self) -> bool:
        """Gracefully shutdown the timeline component and storage."""
        try:
            # Persist final state before shutdown
            if self._storage_initialized:
                await self._persist_timeline_state()
                logger.info(f"Shutting down storage backend for TimelineComponent {self.id}")
                await self._storage.shutdown()
            return True
        except Exception as e:
            logger.error(f"Error during TimelineComponent shutdown: {e}", exc_info=True)
            return False

    def _create_timeline(self, timeline_id: str, is_primary: bool = False, fork_from: Optional[Dict[str, str]] = None):
        """Internal helper to create timeline metadata and root event."""
        if timeline_id in self._state['_timelines']:
            logger.warning(f"[{self.owner.id}] Timeline '{timeline_id}' already exists.")
            return

        root_event_id = str(uuid.uuid4())
        root_event = {
            'id': root_event_id,
            'timestamp': time.time(),
            'parent_ids': [],
            'timeline_id': timeline_id,
            'payload': {'event_type': 'timeline_created', 'timeline_id': timeline_id}
        }
        # If forking, add parent link to fork point event
        if fork_from and fork_from.get('source_event_id'):
            root_event['parent_ids'] = [fork_from['source_event_id']]
            root_event['payload']['fork_source_timeline'] = fork_from.get('source_timeline_id')
            root_event['payload']['fork_source_event'] = fork_from.get('source_event_id')
            logger.info(f"[{self.owner.id}] Creating timeline '{timeline_id}' forked from event '{fork_from['source_event_id']}'.")
        else:
            logger.info(f"[{self.owner.id}] Creating new root timeline '{timeline_id}'.")

        self._state['_all_events'][root_event_id] = root_event
        self._state['_timelines'][timeline_id] = {
            'is_primary': False, # Set explicitly later if needed
            'head_event_ids': {root_event_id} # Start with the root event as head
        }

        if is_primary:
            self.designate_primary_timeline(timeline_id)

    def add_event_to_primary_timeline(self, event_payload: Dict[str, Any]) -> Optional[str]:
        """
        Adds an event to the primary timeline.
        """
        event_id = self.add_event_to_timeline(event_payload, timeline_context={'timeline_id': self._state['_primary_timeline_id']})

        # NEW: Schedule async persistence (non-blocking)
        if event_id and self._storage_initialized:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._persist_timeline_state())
            except RuntimeError:
                pass  # No event loop running, persistence will happen on shutdown

        return event_id

    def add_event_to_timeline(self, event_payload: Dict[str, Any], timeline_context: Dict[str, Any]) -> Optional[str]:
        """
        Adds a new event to the specified timeline, linking it to the current head(s).

        Args:
            event_payload: The dictionary containing the event data (e.g., {'event_type': ..., 'payload': ...}).
            timeline_context: Dictionary specifying the target timeline.
                             Expected keys:
                               - 'timeline_id': ID of the target timeline (defaults to primary).
                               - (Optional) 'parent_event_ids': Explicit parents. If omitted, uses current heads.

        Returns:
            The ID of the newly created event, or None if failed.
        """
        target_timeline_id = timeline_context.get('timeline_id', self._state['_primary_timeline_id'])
        if not target_timeline_id:
            logger.error(f"[{self.owner.id}] Cannot add event: No target timeline specified and no primary timeline set.")
            return None

        timeline_info = self._state['_timelines'].get(target_timeline_id)
        if not timeline_info:
            logger.error(f"[{self.owner.id}] Cannot add event: Target timeline '{target_timeline_id}' not found.")
            return None

        # Determine parent event IDs
        parent_ids = timeline_context.get('parent_event_ids')
        if not parent_ids:
            # Default to current head(s) of the target timeline
            parent_ids = list(timeline_info.get('head_event_ids', []))
            if not parent_ids:
                 logger.error(f"[{self.owner.id}] Cannot add event to timeline '{target_timeline_id}': No head events found (timeline empty or corrupt?).")
                 return None
        elif not isinstance(parent_ids, list):
             logger.warning(f"[{self.owner.id}] Provided parent_event_ids is not a list ({type(parent_ids)}). Converting to list.")
             parent_ids = [str(parent_ids)] # Attempt conversion

        # Validate parent events exist? (Optional, can be slow)
        # for p_id in parent_ids:
        #     if p_id not in self._state['_all_events']:
        #         logger.error(f"Cannot add event: Parent event ID '{p_id}' not found.")
        #         return None

        new_event_id = str(uuid.uuid4())
        new_event_node = {
            'id': new_event_id,
            'timestamp': time.time(),
            'parent_ids': parent_ids,
            'timeline_id': target_timeline_id, # Mark which timeline it was added to
            'payload': event_payload # Store the full original payload
        }

        self._state['_all_events'][new_event_id] = new_event_node

        # Update head events for this timeline: the new event becomes the sole head.
        # For merges, this logic would be more complex.
        timeline_info['head_event_ids'] = {new_event_id}

        logger.debug(f"[{self.owner.id}] Event '{new_event_id}' added to timeline '{target_timeline_id}' (Parents: {parent_ids}). Payload type: {event_payload.get('event_type')}")

        # NEW: Schedule async persistence for the new event (non-blocking)
        if self._storage_initialized:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._persist_timeline_state())
            except RuntimeError:
                pass  # No event loop running, persistence will happen on shutdown

        return new_event_id

    def get_primary_timeline(self) -> Optional[str]:
        """Returns the ID of the primary timeline, if one is designated."""
        return self._state.get('_primary_timeline_id')

    def designate_primary_timeline(self, timeline_id: str) -> bool:
        """
        Sets a timeline as the primary one.

        Args:
            timeline_id: The ID of the timeline to set as primary.

        Returns:
            True if successful, False otherwise.
        """
        if timeline_id not in self._state['_timelines']:
            logger.error(f"[{self.owner.id}] Cannot designate primary timeline: Timeline '{timeline_id}' does not exist.")
            return False

        old_primary = self._state['_primary_timeline_id']
        if old_primary and old_primary in self._state['_timelines']:
            self._state['_timelines'][old_primary]['is_primary'] = False

        self._state['_timelines'][timeline_id]['is_primary'] = True
        self._state['_primary_timeline_id'] = timeline_id
        logger.info(f"[{self.owner.id}] Timeline '{timeline_id}' designated as primary.")
        return True

    def get_timeline_events(self, timeline_id: str, start_event_id: Optional[str] = None, limit: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieves events for a specific timeline, traversing backwards from heads or start_event_id.
        NOTE: Basic implementation traverses backwards from current heads. Does not handle complex DAGs efficiently.

        Args:
            timeline_id: The ID of the timeline to retrieve events from.
            start_event_id: If provided, start traversal from this event backwards.
            limit: Maximum number of events to return (0 for all reachable).

        Returns:
            A list of event node dictionaries, ordered newest to oldest.
        """
        timeline_info = self._state['_timelines'].get(timeline_id)
        if not timeline_info:
            logger.warning(f"[{self.owner.id}] Cannot get events: Timeline '{timeline_id}' not found.")
            return []

        events = []
        queue: List[str] = []
        visited: Set[str] = set()

        if start_event_id:
             if start_event_id in self._state['_all_events']:
                  queue.append(start_event_id)
             else:
                  logger.warning(f"[{self.owner.id}] Start event ID '{start_event_id}' not found.")
        else:
            # Start from current head(s) if no start_event_id
            queue.extend(list(timeline_info.get('head_event_ids', [])))

        queue.sort(key=lambda eid: self._state['_all_events'].get(eid, {}).get('timestamp', 0), reverse=True)

        while queue:
            current_event_id = queue.pop(0)
            if current_event_id in visited:
                continue
            visited.add(current_event_id)

            event_node = self._state['_all_events'].get(current_event_id)
            if not event_node:
                logger.warning(f"[{self.owner.id}] Traversal error: Event ID '{current_event_id}' not found in _all_events.")
                continue

            events.append(event_node)

            if limit > 0 and len(events) >= limit:
                break

            # Add parents to queue for further traversal (simple backward chain)
            parent_ids = event_node.get('parent_ids', [])
            # Basic handling: only add parents not already visited
            parents_to_add = [p_id for p_id in parent_ids if p_id not in visited and p_id in self._state['_all_events']]
            # In a real DAG, sorting parents by timestamp before adding might be needed for specific orders
            queue.extend(parents_to_add)
            # Re-sort queue? Or process depth-first? Simple list extension implies BFS-like layer by layer backwards.
            # For now, keep it simple.

        # Events are currently newest -> oldest due to traversal order
        return events

    def get_last_relevant_event(
        self,
        timeline_id: Optional[str] = None,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Finds the most recent event on a timeline matching the filter criteria.
        Searches backwards from the timeline head(s).

        Args:
            timeline_id: The ID of the timeline to search. Defaults to the primary timeline.
            filter_criteria: A dictionary specifying conditions the event must meet.
                             Supports filtering by 'event_type' in payload, and keys within payload['data'].
                             Example:
                                 {'payload.event_type': 'tool_result_received'}
                                 {'payload.data.loop_component_id': 'abc-123'}
                                 {'payload.event_type__in': ['type1', 'type2']}

        Returns:
            The full event dictionary of the most recent matching event, or None if not found.
        """
        target_timeline_id = timeline_id or self.get_primary_timeline()
        if not target_timeline_id:
            logger.warning(f"[{self.owner.id}] Cannot get last relevant event: No timeline specified and no primary found.")
            return None

        timeline_info = self._state['_timelines'].get(target_timeline_id)
        if not timeline_info:
            logger.warning(f"[{self.owner.id}] Cannot get last relevant event: Timeline '{target_timeline_id}' not found.")
            return None

        if not filter_criteria:
             logger.warning(f"[{self.owner.id}] get_last_relevant_event called without filter_criteria.")
             # Optionally return the absolute last event? For now, return None.
             return None

        # Use a similar backward traversal logic as get_timeline_events, but stop at first match
        queue: List[str] = []
        visited: Set[str] = set()

        # Start from current head(s)
        queue.extend(list(timeline_info.get('head_event_ids', [])))
        # Process newest heads first
        queue.sort(key=lambda eid: self._state['_all_events'].get(eid, {}).get('timestamp', 0), reverse=True)

        while queue:
            current_event_id = queue.pop(0)
            if current_event_id in visited:
                continue
            visited.add(current_event_id)

            event_node = self._state['_all_events'].get(current_event_id)
            if not event_node:
                continue # Should not happen normally

            # --- Check if event matches filter criteria ---
            match = True
            # payload = event_node.get('payload', {}) # No longer needed here
            # data = payload.get('data', {}) # No longer needed here
            for key, expected_value in filter_criteria.items():
                actual_value = None
                is_match = False

                # Handle special suffixes first (e.g., __in)
                if key.endswith('__in'):
                    base_key = key[:-4] # Remove '__in'
                    actual_value = self._get_nested_value(event_node, base_key)
                    if isinstance(expected_value, list) and actual_value in expected_value:
                        is_match = True
                    # else: is_match remains False

                # Handle standard equality check
                else:
                    actual_value = self._get_nested_value(event_node, key)
                    if actual_value == expected_value:
                        is_match = True
                    # else: is_match remains False

                # If any criterion fails, the event doesn't match
                if not is_match:
                    match = False
                    break

            if match:
                logger.debug(f"Found matching event '{current_event_id}' for criteria: {filter_criteria}")
                return event_node # Found the most recent match
            # --- End Match Check ---

            # If not matched, add parents to queue (newest first)
            parent_ids = event_node.get('parent_ids', [])
            parents_to_add = []
            for p_id in parent_ids:
                 if p_id not in visited and p_id in self._state['_all_events']:
                      parents_to_add.append((self._state['_all_events'][p_id].get('timestamp', 0), p_id))

            # Sort parents by timestamp descending to check newer branches first
            parents_to_add.sort(key=lambda item: item[0], reverse=True)
            # Insert parents at the beginning of the queue to continue depth-first search backwards
            queue = [p_id for _, p_id in parents_to_add] + queue

        logger.debug(f"No matching event found for criteria: {filter_criteria}")
        return None # No match found in history

    # --- Helper for nested dictionary access ---
    def _get_nested_value(self, data: Dict, key_path: str, default: Any = None) -> Any:
        """Safely retrieve a value from a nested dictionary using dot notation."""
        keys = key_path.split('.')
        current_level = data
        for key in keys:
            if isinstance(current_level, dict):
                current_level = current_level.get(key)
            else:
                return default # Cannot traverse further
            if current_level is None:
                return default # Key not found at this level
        return current_level

    # --- Methods for future Loom features (Forking, Merging) ---
    # def create_timeline_fork(...)
    # def merge_timelines(...)
    # def get_timeline_relationships(...)

    def create_timeline_fork(
        self,
        source_timeline_id: str,
        fork_point_event_id: str,
        new_timeline_id: Optional[str] = None,
        is_primary: bool = False
    ) -> Optional[str]:
        """
        Creates a new timeline forked from an existing one at a specific event.

        Args:
            source_timeline_id: The ID of the timeline to fork from.
            fork_point_event_id: The ID of the event in the source timeline to fork from.
                                 The new timeline's root event will have this as its parent.
            new_timeline_id: Optional specific ID for the new timeline. If None, a UUID is generated.
            is_primary: Whether the new forked timeline should become the primary one.

        Returns:
            The ID of the new timeline if successful, None otherwise.
        """
        # Validate source timeline exists
        if source_timeline_id not in self._state['_timelines']:
            logger.error(f"[{self.owner.id}] Cannot fork: Source timeline '{source_timeline_id}' not found.")
            return None

        # Validate fork point event exists
        if fork_point_event_id not in self._state['_all_events']:
            logger.error(f"[{self.owner.id}] Cannot fork: Fork point event '{fork_point_event_id}' not found.")
            return None
        # Optionally, validate the fork point event is actually part of the source timeline history (more complex)

        final_new_timeline_id = new_timeline_id if new_timeline_id else f"timeline_{uuid.uuid4()}"
        if final_new_timeline_id in self._state['_timelines']:
            logger.error(f"[{self.owner.id}] Cannot fork: New timeline ID '{final_new_timeline_id}' already exists.")
            return None

        fork_info = {
            'source_timeline_id': source_timeline_id,
            'source_event_id': fork_point_event_id
        }

        self._create_timeline(final_new_timeline_id, is_primary=is_primary, fork_from=fork_info)

        # Check if creation was successful (timeline should now exist)
        if final_new_timeline_id in self._state['_timelines']:
            logger.info(f"[{self.owner.id}] Successfully created timeline fork '{final_new_timeline_id}' from event '{fork_point_event_id}' on timeline '{source_timeline_id}'.")
            return final_new_timeline_id
        else:
            # _create_timeline should log errors if it fails internally
            logger.error(f"[{self.owner.id}] Failed to create timeline fork '{final_new_timeline_id}'.")
            return None

    def merge_timeline_simple(
        self,
        source_timeline_id: str,
        target_timeline_id: str,
        merge_event_payload: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Performs a simple merge by creating a new merge event in the target timeline
        that has the heads of both source and target timelines as parents.
        This does NOT interleave or rewrite history, it just marks the merge point.

        Args:
            source_timeline_id: The ID of the timeline whose head(s) will be merged from.
            target_timeline_id: The ID of the timeline to merge into. Its head(s) will also be parents.
                                This timeline will contain the new merge event.
            merge_event_payload: Optional payload for the merge event itself (e.g., merge strategy notes).

        Returns:
            The ID of the new merge event if successful, None otherwise.
        """
        source_timeline_info = self._state['_timelines'].get(source_timeline_id)
        target_timeline_info = self._state['_timelines'].get(target_timeline_id)

        if not source_timeline_info:
            logger.error(f"[{self.owner.id}] Cannot merge: Source timeline '{source_timeline_id}' not found.")
            return None
        if not target_timeline_info:
            logger.error(f"[{self.owner.id}] Cannot merge: Target timeline '{target_timeline_id}' not found.")
            return None

        source_head_ids = list(source_timeline_info.get('head_event_ids', []))
        target_head_ids = list(target_timeline_info.get('head_event_ids', []))

        if not source_head_ids:
            logger.warning(f"[{self.owner.id}] Source timeline '{source_timeline_id}' has no head events to merge from.")
            # Depending on desired semantics, could allow merging an empty timeline?
            # For now, let's require source heads.
            return None
        if not target_head_ids:
             logger.error(f"[{self.owner.id}] Cannot merge into timeline '{target_timeline_id}': No head events found.")
             return None

        # Combine parents from both heads, ensuring uniqueness
        all_parent_ids = list(set(source_head_ids + target_head_ids))

        # Create the payload for the merge event
        final_merge_payload = {
            'event_type': 'timelines_merged_simple',
            'payload': {
                'source_timeline_id': source_timeline_id,
                'target_timeline_id': target_timeline_id,
                'merged_head_ids': all_parent_ids,
                **(merge_event_payload or {})
            }
        }

        # Add the merge event to the target timeline
        timeline_context = {
            'timeline_id': target_timeline_id,
            'parent_event_ids': all_parent_ids # Explicitly set parents
        }

        merge_event_id = self.add_event_to_timeline(final_merge_payload, timeline_context)

        if merge_event_id:
            logger.info(f"[{self.owner.id}] Successfully created simple merge event '{merge_event_id}' in timeline '{target_timeline_id}', merging from '{source_timeline_id}'.")
            # The target timeline's head is now automatically the merge event ID due to add_event_to_timeline logic.
            return merge_event_id
        else:
            logger.error(f"[{self.owner.id}] Failed to create merge event while merging '{source_timeline_id}' into '{target_timeline_id}'.")
            return None
