import pytest
import time
import uuid
from unittest.mock import MagicMock, patch, ANY
from typing import Dict, Any, Optional

try:
    # Class to test
    from elements.elements.components.space.timeline_component import TimelineComponent
    # Dependencies
    from elements.elements.base import BaseElement, Component

except ImportError as e:
    pytest.skip(f"Skipping TimelineComponent tests due to import error: {e}", allow_module_level=True)

# --- Mocks and Fixtures ---

class MockTimelineElement(BaseElement):
    """Mock element to host the TimelineComponent."""
    def __init__(self, element_id="mock_timeline_host", name="TimelineHost"):
        super().__init__(element_id, name, "Host for Timeline") # Call BaseElement init
        self._components: Dict[str, Component] = {}

    def add_component(self, component: Component):
        key = getattr(component, 'COMPONENT_TYPE', component.__class__.__name__)
        self._components[key] = component
        component.element = self

    def get_component_by_type(self, comp_type: str) -> Optional[Component]:
         return self._components.get(comp_type)

@pytest.fixture
def host_element():
    return MockTimelineElement()

@pytest.fixture
def timeline_component(host_element):
    """Instance of the component under test, initialized."""
    component = TimelineComponent(element=host_element)
    # Manually initialize - this creates the primary timeline
    initialized = component._on_initialize()
    assert initialized is True
    component._is_enabled = True # Enable for direct calls
    host_element.add_component(component)
    return component

# Helper to get the initial primary timeline ID created by the fixture
@pytest.fixture
def primary_timeline_id(timeline_component):
    return timeline_component._state["primary_timeline"]

# --- Test Cases ---

def test_initialization_creates_primary(timeline_component, primary_timeline_id):
    """Test _on_initialize creates a primary timeline correctly."""
    assert primary_timeline_id is not None
    assert primary_timeline_id.startswith("timeline_")
    assert primary_timeline_id in timeline_component._state["events"]
    assert primary_timeline_id in timeline_component._state["active_timelines"]
    assert primary_timeline_id in timeline_component._state["timeline_metadata"]
    assert primary_timeline_id in timeline_component._state["timeline_relationships"]
    assert primary_timeline_id in timeline_component._state["element_states"]

    metadata = timeline_component._state["timeline_metadata"][primary_timeline_id]
    assert metadata["name"] == "Primary Timeline"
    assert "created_at" in metadata
    assert "last_updated" in metadata

    rels = timeline_component._state["timeline_relationships"][primary_timeline_id]
    assert rels["parent_id"] is None
    assert rels["is_primary"] is True
    assert rels["fork_point"] is None

def test_initialization_idempotent(timeline_component, primary_timeline_id):
    """Test calling _on_initialize again doesn't create another primary timeline."""
    initial_timeline_count = len(timeline_component._state["active_timelines"])
    
    # Call initialize again
    timeline_component._on_initialize()
    
    assert timeline_component._state["primary_timeline"] == primary_timeline_id
    assert len(timeline_component._state["active_timelines"]) == initial_timeline_count

# --- add_event_to_timeline Tests ---

def test_add_event_success(timeline_component, primary_timeline_id):
    """Test adding a valid event."""
    event_data = {"event_type": "test", "data": "foo"}
    context = {"timeline_id": primary_timeline_id}
    initial_update_time = timeline_component._state["timeline_metadata"][primary_timeline_id]["last_updated"]

    # Allow time to pass for timestamp check
    time.sleep(0.01)
    
    event_id = timeline_component.add_event_to_timeline(event_data.copy(), context)

    assert event_id is not None
    assert event_id.startswith("event_")
    timeline_events = timeline_component._state["events"][primary_timeline_id]
    assert len(timeline_events) == 1
    added_event = timeline_events[0]
    assert added_event["event_id"] == event_id
    assert added_event["event_type"] == "test"
    assert added_event["data"] == "foo"
    assert "timestamp" in added_event

    # Check metadata updated
    assert timeline_component._state["timeline_metadata"][primary_timeline_id]["last_updated"] > initial_update_time

def test_add_event_with_existing_id_and_timestamp(timeline_component, primary_timeline_id):
    """Test adding event with provided ID and timestamp."""
    event_id = "custom_event_1"
    timestamp = 1000
    event_data = {"event_id": event_id, "event_type": "custom", "timestamp": timestamp}
    context = {"timeline_id": primary_timeline_id}

    added_event_id = timeline_component.add_event_to_timeline(event_data.copy(), context)

    assert added_event_id == event_id
    timeline_events = timeline_component._state["events"][primary_timeline_id]
    assert len(timeline_events) == 1
    added_event = timeline_events[0]
    assert added_event["event_id"] == event_id
    assert added_event["timestamp"] == timestamp

def test_add_event_fail_no_timeline_id(timeline_component, caplog):
    """Test adding event fails if context lacks timeline_id."""
    with caplog.at_level(logging.WARNING):
        event_id = timeline_component.add_event_to_timeline({}, {})
    assert event_id is None
    assert "No timeline ID provided" in caplog.text

def test_add_event_fail_invalid_timeline_id(timeline_component, caplog):
    """Test adding event fails if timeline_id doesn't exist."""
    with caplog.at_level(logging.WARNING):
        event_id = timeline_component.add_event_to_timeline({}, {"timeline_id": "non_existent"})
    assert event_id is None
    assert "Timeline non_existent does not exist" in caplog.text

def test_add_event_fail_disabled(timeline_component, primary_timeline_id):
    """Test adding event fails if component disabled."""
    timeline_component._is_enabled = False
    event_id = timeline_component.add_event_to_timeline({}, {"timeline_id": primary_timeline_id})
    assert event_id is None

# --- update_state Tests ---

def test_update_state_success(timeline_component, primary_timeline_id):
    """Test successfully updating state."""
    update_data = {"key": "value", "other": 123}
    context = {"timeline_id": primary_timeline_id}

    # Patch add_event_to_timeline to verify its call
    with patch.object(timeline_component, 'add_event_to_timeline', wraps=timeline_component.add_event_to_timeline) as mock_add:
        success = timeline_component.update_state(update_data, context)

    assert success is True
    # Check add_event was called with correct structure
    mock_add.assert_called_once()
    call_args, _ = mock_add.call_args
    event_arg, context_arg = call_args
    assert context_arg == context
    assert event_arg["event_type"] == "state_update"
    assert event_arg["element_id"] == timeline_component.element.id
    assert event_arg["data"] == update_data
    state_event_id = event_arg["event_id"]

    # Check state storage
    assert primary_timeline_id in timeline_component._state["element_states"]
    timeline_states = timeline_component._state["element_states"][primary_timeline_id]
    assert state_event_id in timeline_states
    assert timeline_states[state_event_id] == update_data

def test_update_state_fail_no_timeline_id(timeline_component, caplog):
    """Test update_state fails without timeline_id."""
    with caplog.at_level(logging.WARNING):
        success = timeline_component.update_state({}, {})
    assert success is False
    assert "No timeline ID provided" in caplog.text

# --- create_timeline_fork Tests ---

def test_create_timeline_fork_success(timeline_component, primary_timeline_id):
    """Test successfully forking a timeline."""
    # Add some events to the primary timeline
    ev1_id = timeline_component.add_event_to_timeline({"event_type": "e1"}, {"timeline_id": primary_timeline_id})
    ev2_id = timeline_component.add_event_to_timeline({"event_type": "e2"}, {"timeline_id": primary_timeline_id})
    ev3_id = timeline_component.add_event_to_timeline({"event_type": "e3"}, {"timeline_id": primary_timeline_id})
    # Add some state associated with an event
    timeline_component.update_state({"state1": "a"}, {"timeline_id": primary_timeline_id}) # ev4
    timeline_component.update_state({"state2": "b"}, {"timeline_id": primary_timeline_id}) # ev5 - fork point
    fork_point_event_id = timeline_component._state["events"][primary_timeline_id][-1]["event_id"]
    timeline_component.update_state({"state3": "c"}, {"timeline_id": primary_timeline_id}) # ev6 - after fork

    fork_name = "My Fork"
    fork_desc = "Testing fork"
    
    new_timeline_id = timeline_component.create_timeline_fork(
        source_timeline_id=primary_timeline_id,
        fork_point_event_id=fork_point_event_id,
        is_primary=False,
        name=fork_name,
        description=fork_desc
    )

    assert new_timeline_id is not None
    assert new_timeline_id != primary_timeline_id
    assert new_timeline_id in timeline_component._state["active_timelines"]
    assert new_timeline_id in timeline_component._state["events"]
    assert new_timeline_id in timeline_component._state["timeline_metadata"]
    assert new_timeline_id in timeline_component._state["timeline_relationships"]
    assert new_timeline_id in timeline_component._state["element_states"]

    # Check events copied correctly (up to and including fork point)
    forked_events = timeline_component._state["events"][new_timeline_id]
    primary_events = timeline_component._state["events"][primary_timeline_id]
    assert len(forked_events) == 5 # Initial + e1, e2, e3, state1, state2(fork)
    assert forked_events[-1]["event_id"] == fork_point_event_id
    assert len(primary_events) == 6 # Should have the event after the fork

    # Check metadata and relationships
    assert timeline_component._state["timeline_metadata"][new_timeline_id]["name"] == fork_name
    assert timeline_component._state["timeline_metadata"][new_timeline_id]["description"] == fork_desc
    rels = timeline_component._state["timeline_relationships"][new_timeline_id]
    assert rels["parent_id"] == primary_timeline_id
    assert rels["is_primary"] is False
    assert rels["fork_point"] == fork_point_event_id

    # Check element states copied correctly (only those up to fork point)
    forked_states = timeline_component._state["element_states"][new_timeline_id]
    primary_states = timeline_component._state["element_states"][primary_timeline_id]
    state_event_ids = [e["event_id"] for e in forked_events if e["event_type"] == "state_update"]
    assert len(forked_states) == 2 # state1, state2
    assert all(eid in forked_states for eid in state_event_ids)
    assert len(primary_states) == 3 # state1, state2, state3

def test_create_timeline_fork_fail_bad_source(timeline_component, primary_timeline_id, caplog):
    """Test fork fails if source timeline doesn't exist."""
    with caplog.at_level(logging.WARNING):
        new_id = timeline_component.create_timeline_fork("bad_source", "any_event")
    assert new_id is None
    assert "Source timeline bad_source does not exist" in caplog.text

def test_create_timeline_fork_fail_bad_fork_point(timeline_component, primary_timeline_id, caplog):
    """Test fork fails if fork point event doesn't exist in source."""
    with caplog.at_level(logging.WARNING):
        new_id = timeline_component.create_timeline_fork(primary_timeline_id, "bad_event_id")
    assert new_id is None
    assert "Fork point event bad_event_id not found" in caplog.text

# --- designate_primary_timeline Tests ---

def test_designate_primary_timeline_success(timeline_component, primary_timeline_id):
    """Test designating an existing non-primary timeline as primary."""
    # Create a fork first
    ev1_id = timeline_component.add_event_to_timeline({"type": "e1"}, {"timeline_id": primary_timeline_id})
    fork_id = timeline_component.create_timeline_fork(primary_timeline_id, ev1_id)
    assert fork_id is not None
    assert timeline_component._state["primary_timeline"] == primary_timeline_id
    assert timeline_component._state["timeline_relationships"][primary_timeline_id]["is_primary"] is True
    assert timeline_component._state["timeline_relationships"][fork_id]["is_primary"] is False

    success = timeline_component.designate_primary_timeline(fork_id)

    assert success is True
    assert timeline_component._state["primary_timeline"] == fork_id
    assert timeline_component._state["timeline_relationships"][primary_timeline_id]["is_primary"] is False
    assert timeline_component._state["timeline_relationships"][fork_id]["is_primary"] is True

def test_designate_primary_timeline_already_primary(timeline_component, primary_timeline_id):
    """Test designating the current primary as primary again (should be no-op)."""
    success = timeline_component.designate_primary_timeline(primary_timeline_id)
    assert success is True # Still successful
    assert timeline_component._state["primary_timeline"] == primary_timeline_id
    assert timeline_component._state["timeline_relationships"][primary_timeline_id]["is_primary"] is True

def test_designate_primary_timeline_fail_not_found(timeline_component, caplog):
    """Test designating fails if timeline ID doesn't exist."""
    with caplog.at_level(logging.WARNING):
        success = timeline_component.designate_primary_timeline("non_existent_id")
    assert success is False
    assert "Timeline non_existent_id not found" in caplog.text

# --- Getter Tests ---

def test_get_timeline_events(timeline_component, primary_timeline_id):
    """Test retrieving events for a specific timeline."""
    ev1 = {"event_id": "ev1", "type": "t1"}
    ev2 = {"event_id": "ev2", "type": "t2"}
    timeline_component.add_event_to_timeline(ev1, {"timeline_id": primary_timeline_id})
    timeline_component.add_event_to_timeline(ev2, {"timeline_id": primary_timeline_id})

    events = timeline_component.get_timeline_events(primary_timeline_id)
    assert len(events) == 2
    assert events[0]["event_id"] == "ev1"
    assert events[1]["event_id"] == "ev2"

    empty_events = timeline_component.get_timeline_events("non_existent")
    assert empty_events == []

def test_get_primary_timeline(timeline_component, primary_timeline_id):
    """Test retrieving the primary timeline ID."""
    assert timeline_component.get_primary_timeline() == primary_timeline_id

def test_is_primary_timeline(timeline_component, primary_timeline_id):
    """Test checking if a timeline is primary."""
    assert timeline_component.is_primary_timeline(primary_timeline_id) is True
    
    # Create a fork
    ev1_id = timeline_component.add_event_to_timeline({}, {"timeline_id": primary_timeline_id})
    fork_id = timeline_component.create_timeline_fork(primary_timeline_id, ev1_id)
    assert timeline_component.is_primary_timeline(fork_id) is False
    assert timeline_component.is_primary_timeline("non_existent") is False

def test_get_timeline_relationships(timeline_component, primary_timeline_id):
     """Test retrieving timeline relationship info."""
     rels_primary = timeline_component.get_timeline_relationships(primary_timeline_id)
     assert rels_primary["parent_id"] is None
     assert rels_primary["is_primary"] is True
     
     ev1_id = timeline_component.add_event_to_timeline({}, {"timeline_id": primary_timeline_id})
     fork_id = timeline_component.create_timeline_fork(primary_timeline_id, ev1_id)
     rels_fork = timeline_component.get_timeline_relationships(fork_id)
     assert rels_fork["parent_id"] == primary_timeline_id
     assert rels_fork["is_primary"] is False
     assert rels_fork["fork_point"] == ev1_id
     
     assert timeline_component.get_timeline_relationships("non_existent") is None
