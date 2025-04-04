"""
Timeline Component
Component for managing timeline functionality for Space elements.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Set
import uuid
import time

from ..base_component import Component

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimelineComponent(Component):
    """
    Component for managing timeline functionality for Space elements.
    
    This component provides the Loom system functionality:
    - Timeline management with DAG structure
    - Event recording and retrieval
    - Timeline forking and merging
    - Timeline state management
    """
    
    # Component unique type identifier
    COMPONENT_TYPE: str = "timeline"
    
    # Event types this component handles
    HANDLED_EVENT_TYPES: List[str] = [
        "timeline_created",
        "timeline_forked",
        "timeline_designated_primary",
        "state_update"
    ]
    
    def __init__(self, element=None):
        """
        Initialize the timeline component.
        
        Args:
            element: The Element this component is attached to
        """
        super().__init__(element)
        
        # Initialize timeline state
        self._state = {
            "events": {},  # timeline_id -> list of events
            "active_timelines": set(),  # Set of active timeline IDs
            "timeline_metadata": {},  # timeline_id -> metadata
            "timeline_relationships": {},  # timeline_id -> {parent_id, is_primary, fork_point}
            "primary_timeline": None,  # ID of the primary timeline
            "element_states": {},  # timeline_id -> {event_id: event_data}
            "entangled_timelines": set()  # Set of entangled timeline IDs
        }
    
    def _on_initialize(self) -> bool:
        """
        Initialize the timeline component.
        
        Creates a primary timeline if none exists.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        # Create a primary timeline if none exists
        if not self._state["primary_timeline"]:
            primary_timeline_id = f"timeline_{uuid.uuid4().hex[:8]}"
            
            # Initialize the timeline
            self._state["events"][primary_timeline_id] = []
            self._state["active_timelines"].add(primary_timeline_id)
            self._state["timeline_metadata"][primary_timeline_id] = {
                "created_at": int(time.time() * 1000),
                "last_updated": int(time.time() * 1000),
                "name": "Primary Timeline",
                "description": "Default timeline"
            }
            
            # Set as primary
            self._state["primary_timeline"] = primary_timeline_id
            self._state["timeline_relationships"][primary_timeline_id] = {
                "parent_id": None,
                "is_primary": True,
                "fork_point": None
            }
            
            # Initialize element states for this timeline
            self._state["element_states"][primary_timeline_id] = {}
            
            logger.info(f"Created primary timeline: {primary_timeline_id}")
        
        return True
    
    def add_event_to_timeline(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> Optional[str]:
        """
        Add an event to a timeline.
        
        Args:
            event: Event data to add
            timeline_context: Timeline context information
            
        Returns:
            ID of the created event, or None if it could not be added
        """
        if not self._is_initialized or not self._is_enabled:
            logger.warning(f"Cannot add event: Timeline component {self.id} is not initialized or enabled")
            return None
            
        # Extract timeline ID from context
        timeline_id = timeline_context.get("timeline_id")
        if not timeline_id:
            logger.warning(f"Cannot add event: No timeline ID provided")
            return None
            
        # Check if timeline exists
        if timeline_id not in self._state["events"]:
            logger.warning(f"Cannot add event: Timeline {timeline_id} does not exist")
            return None
            
        # Generate event ID if not provided
        event_id = event.get("event_id")
        if not event_id:
            event_id = f"event_{uuid.uuid4().hex[:8]}"
            event["event_id"] = event_id
            
        # Add timestamp if not provided
        if "timestamp" not in event:
            event["timestamp"] = int(time.time() * 1000)
            
        # Add the event to the timeline
        self._state["events"][timeline_id].append(event)
        
        # Update timeline metadata
        self._state["timeline_metadata"][timeline_id]["last_updated"] = int(time.time() * 1000)
        
        logger.debug(f"Added event {event_id} to timeline {timeline_id}")
        return event_id
    
    def update_state(self, update_data: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Update state in a specific timeline.
        
        Args:
            update_data: Data to update
            timeline_context: Timeline context information
            
        Returns:
            True if the state was updated, False otherwise
        """
        if not self._is_initialized or not self._is_enabled:
            logger.warning(f"Cannot update state: Timeline component {self.id} is not initialized or enabled")
            return False
            
        # Extract timeline ID from context
        timeline_id = timeline_context.get("timeline_id")
        if not timeline_id:
            logger.warning(f"Cannot update state: No timeline ID provided")
            return False
            
        # Check if timeline exists
        if timeline_id not in self._state["element_states"]:
            logger.warning(f"Cannot update state: Timeline {timeline_id} does not exist")
            return False
            
        # Create the state update event
        event_id = f"state_update_{uuid.uuid4().hex[:8]}"
        event = {
            "event_id": event_id,
            "event_type": "state_update",
            "timestamp": int(time.time() * 1000),
            "element_id": self.element.id,
            "data": update_data
        }
        
        # Add the event to the timeline
        self.add_event_to_timeline(event, timeline_context)
        
        # Store the state update
        self._state["element_states"][timeline_id][event_id] = update_data
        
        return True
    
    def create_timeline_fork(self, source_timeline_id: str, fork_point_event_id: str, 
                           is_primary: bool = False, name: Optional[str] = None,
                           description: Optional[str] = None) -> Optional[str]:
        """
        Create a new timeline forked from an existing one.
        
        Args:
            source_timeline_id: ID of the timeline to fork from
            fork_point_event_id: ID of the event to fork from
            is_primary: Whether this should be the primary timeline
            name: Name for the new timeline
            description: Description for the new timeline
            
        Returns:
            ID of the new timeline, or None if it could not be created
        """
        if not self._is_initialized or not self._is_enabled:
            logger.warning(f"Cannot fork timeline: Timeline component {self.id} is not initialized or enabled")
            return None
            
        # Check if source timeline exists
        if source_timeline_id not in self._state["events"]:
            logger.warning(f"Cannot fork timeline: Source timeline {source_timeline_id} does not exist")
            return None
            
        # Check if fork point exists in source timeline
        source_events = self._state["events"][source_timeline_id]
        fork_point_index = None
        for i, event in enumerate(source_events):
            if event.get("event_id") == fork_point_event_id:
                fork_point_index = i
                break
                
        if fork_point_index is None:
            logger.warning(f"Cannot fork timeline: Fork point event {fork_point_event_id} not found in timeline {source_timeline_id}")
            return None
            
        # Generate new timeline ID
        new_timeline_id = f"timeline_{uuid.uuid4().hex[:8]}"
        
        # Create the new timeline
        self._state["events"][new_timeline_id] = source_events[:fork_point_index + 1].copy()
        self._state["active_timelines"].add(new_timeline_id)
        self._state["timeline_metadata"][new_timeline_id] = {
            "created_at": int(time.time() * 1000),
            "last_updated": int(time.time() * 1000),
            "name": name or f"Fork of {self._state['timeline_metadata'][source_timeline_id].get('name', source_timeline_id)}",
            "description": description or f"Forked from {source_timeline_id} at event {fork_point_event_id}"
        }
        
        # Set relationship
        self._state["timeline_relationships"][new_timeline_id] = {
            "parent_id": source_timeline_id,
            "is_primary": is_primary,
            "fork_point": fork_point_event_id
        }
        
        # Initialize element states for this timeline (copy from source)
        self._state["element_states"][new_timeline_id] = {}
        if source_timeline_id in self._state["element_states"]:
            # Copy only states relevant up to the fork point
            for event_id, state in self._state["element_states"][source_timeline_id].items():
                # Only include states that are from events up to the fork point
                source_event_ids = [e.get("event_id") for e in source_events[:fork_point_index + 1]]
                if event_id in source_event_ids:
                    self._state["element_states"][new_timeline_id][event_id] = state.copy()
        
        # If this is the primary timeline, update the primary reference
        if is_primary:
            self.designate_primary_timeline(new_timeline_id)
            
        # Record the fork event
        fork_event = {
            "event_id": f"fork_{uuid.uuid4().hex[:8]}",
            "event_type": "timeline_forked",
            "timestamp": int(time.time() * 1000),
            "source_timeline_id": source_timeline_id,
            "fork_point_event_id": fork_point_event_id,
            "is_primary": is_primary
        }
        
        self.add_event_to_timeline(fork_event, {"timeline_id": new_timeline_id})
        
        logger.info(f"Created timeline fork {new_timeline_id} from {source_timeline_id} at event {fork_point_event_id}")
        return new_timeline_id
    
    def designate_primary_timeline(self, timeline_id: str) -> bool:
        """
        Designate a timeline as the primary timeline.
        
        Args:
            timeline_id: ID of the timeline to designate as primary
            
        Returns:
            True if the timeline was designated as primary, False otherwise
        """
        if not self._is_initialized or not self._is_enabled:
            logger.warning(f"Cannot designate primary timeline: Timeline component {self.id} is not initialized or enabled")
            return False
            
        # Check if timeline exists
        if timeline_id not in self._state["events"]:
            logger.warning(f"Cannot designate primary timeline: Timeline {timeline_id} does not exist")
            return False
            
        # Update relationships
        if self._state["primary_timeline"]:
            # Clear primary flag on previous primary timeline
            prev_primary = self._state["primary_timeline"]
            if prev_primary in self._state["timeline_relationships"]:
                self._state["timeline_relationships"][prev_primary]["is_primary"] = False
        
        # Set new primary timeline
        self._state["primary_timeline"] = timeline_id
        
        # Update relationship
        if timeline_id in self._state["timeline_relationships"]:
            self._state["timeline_relationships"][timeline_id]["is_primary"] = True
        else:
            self._state["timeline_relationships"][timeline_id] = {
                "parent_id": None,
                "is_primary": True,
                "fork_point": None
            }
        
        # Record the designation event
        designation_event = {
            "event_id": f"primary_designation_{uuid.uuid4().hex[:8]}",
            "event_type": "timeline_designated_primary",
            "timestamp": int(time.time() * 1000),
            "previous_primary": self._state["primary_timeline"],
            "new_primary": timeline_id
        }
        
        self.add_event_to_timeline(designation_event, {"timeline_id": timeline_id})
        
        logger.info(f"Designated timeline {timeline_id} as primary")
        return True
    
    def get_timeline_relationships(self, timeline_id: str) -> Optional[Dict[str, Any]]:
        """
        Get relationships for a timeline.
        
        Args:
            timeline_id: ID of the timeline
            
        Returns:
            Dictionary of relationship information, or None if not found
        """
        return self._state["timeline_relationships"].get(timeline_id)
    
    def get_primary_timeline(self) -> Optional[str]:
        """
        Get the ID of the primary timeline.
        
        Returns:
            ID of the primary timeline, or None if not set
        """
        return self._state["primary_timeline"]
    
    def is_primary_timeline(self, timeline_id: str) -> bool:
        """
        Check if a timeline is the primary timeline.
        
        Args:
            timeline_id: ID of the timeline to check
            
        Returns:
            True if the timeline is primary, False otherwise
        """
        return (
            self._state["primary_timeline"] == timeline_id and
            timeline_id in self._state["timeline_relationships"] and
            self._state["timeline_relationships"][timeline_id].get("is_primary", False)
        )
    
    def get_timeline_events(self, timeline_id: str) -> List[Dict[str, Any]]:
        """
        Get all events in a timeline.
        
        Args:
            timeline_id: ID of the timeline
            
        Returns:
            List of events in the timeline
        """
        return self._state["events"].get(timeline_id, []) 