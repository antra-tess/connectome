import pytest
import time
import uuid
from unittest.mock import MagicMock, patch

try:
    from elements.elements.base import BaseElement, Component
    from elements.elements.components.messaging.history_component import HistoryComponent
    from elements.elements.components.space.timeline_component import TimelineComponent
except ImportError as e:
    pytest.skip(f"Skipping HistoryComponent tests due to import error: {e}", allow_module_level=True)

# --- Mocks and Fixtures ---

class MockElementWithTimeline(BaseElement):
    """Mock element that can hold components, including a mock TimelineComponent."""
    def __init__(self, element_id="mock_element"):
        self.id = element_id
        self._components = {}
        # Add a mock timeline component by default
        self._mock_timeline_comp = MagicMock(spec=TimelineComponent)
        self._mock_timeline_comp.get_primary_timeline.return_value = "primary_timeline_id"
        self._components[TimelineComponent.COMPONENT_TYPE] = self._mock_timeline_comp

    def add_component(self, component: Component):
        self._components[component.COMPONENT_TYPE] = component
        component.element = self

    def get_component_by_type(self, component_type: str):
        return self._components.get(component_type)

    def get_component(self, component_type_or_class):
         if isinstance(component_type_or_class, type):
              return self.get_component_by_type(component_type_or_class.COMPONENT_TYPE)
         else:
              return self.get_component_by_type(component_type_or_class)

@pytest.fixture
def mock_timeline_component():
    """Fixture for a mocked TimelineComponent."""
    mock = MagicMock(spec=TimelineComponent)
    mock.COMPONENT_TYPE = "timeline"
    mock.get_primary_timeline.return_value = "primary_timeline_1"
    return mock

@pytest.fixture
def mock_element(mock_timeline_component):
    """Fixture for a mocked BaseElement containing a mocked TimelineComponent."""
    element = MagicMock(spec=BaseElement)
    element.get_component_by_type.return_value = mock_timeline_component
    return element

@pytest.fixture
def history_component(mock_element):
    """Fixture for a HistoryComponent instance associated with a mock element."""
    comp = HistoryComponent(element=mock_element)
    comp._initialize()
    comp._enable()
    # Clear state just in case
    comp._state = {"messages": {}}
    return comp

@pytest.fixture
def sample_message_data():
    """Fixture for sample message data."""
    return lambda msg_id=None, text="Hello", timestamp=None: {
        "message_id": msg_id or f"msg_{uuid.uuid4().hex[:8]}",
        "text": text,
        "sender": "user",
        "timestamp": timestamp or int(time.time() * 1000)
    }

# --- Test Cases ---

def test_history_component_initialization(history_component, mock_element):
    """Test that the HistoryComponent initializes correctly."""
    assert history_component.element == mock_element
    assert history_component._state == {"messages": {}}
    assert history_component._is_initialized
    assert history_component._is_enabled
    assert history_component.COMPONENT_TYPE == "messaging_history"
    assert "timeline" in history_component.DEPENDENCIES

def test_get_timeline_comp(history_component, mock_timeline_component):
    """Test the helper method to retrieve the TimelineComponent."""
    assert history_component._get_timeline_comp() == mock_timeline_component

def test_ensure_timeline_history(history_component):
    """Test that the history structure is created for a timeline."""
    timeline_id = "timeline_test_1"
    assert timeline_id not in history_component._state["messages"]
    history_component._ensure_timeline_history(timeline_id)
    assert timeline_id in history_component._state["messages"]
    assert history_component._state["messages"][timeline_id] == {}
    # Call again, should not change anything
    history_component._ensure_timeline_history(timeline_id)
    assert history_component._state["messages"][timeline_id] == {}

# --- Direct Method Tests ---

def test_add_message_success(history_component, sample_message_data):
    """Test adding a message successfully via direct call."""
    timeline_id = "timeline_add_1"
    msg_data = sample_message_data(text="First message")
    msg_id = msg_data["message_id"]

    added = history_component.add_message(msg_data, timeline_id)

    assert added is True
    assert timeline_id in history_component._state["messages"]
    assert msg_id in history_component._state["messages"][timeline_id]
    stored_msg = history_component._state["messages"][timeline_id][msg_id]
    assert stored_msg["text"] == "First message"
    assert stored_msg["sender"] == "user"
    assert "timestamp" in stored_msg
    assert "ignored" not in stored_msg

def test_add_message_generates_id_and_timestamp(history_component):
    """Test that message ID and timestamp are generated if missing."""
    timeline_id = "timeline_add_2"
    msg_data = {"text": "Auto ID", "sender": "system"}

    start_time = int(time.time() * 1000)
    added = history_component.add_message(msg_data.copy(), timeline_id) # Use copy
    end_time = int(time.time() * 1000)

    assert added is True
    assert len(history_component._state["messages"][timeline_id]) == 1
    stored_msg = list(history_component._state["messages"][timeline_id].values())[0]
    assert "message_id" in stored_msg
    assert stored_msg["message_id"].startswith("msg_")
    assert "timestamp" in stored_msg
    assert start_time <= stored_msg["timestamp"] <= end_time
    assert stored_msg["text"] == "Auto ID"

def test_add_message_ignored_marker(history_component, sample_message_data):
    """Test adding a message with an ignore marker."""
    timeline_id = "timeline_add_3"
    msg_data = sample_message_data(text=".ignored command")
    msg_id = msg_data["message_id"]

    added = history_component.add_message(msg_data, timeline_id)

    assert added is True
    assert timeline_id in history_component._state["messages"]
    assert msg_id in history_component._state["messages"][timeline_id]
    stored_msg = history_component._state["messages"][timeline_id][msg_id]
    assert stored_msg["text"] == ".ignored command"
    assert stored_msg.get("ignored") is True

def test_add_message_disabled(history_component, sample_message_data):
    """Test adding a message fails when component is disabled."""
    history_component._disable()
    added = history_component.add_message(sample_message_data(), "timeline_fail_1")
    assert added is False
    assert "timeline_fail_1" not in history_component._state["messages"]

def test_update_message_success(history_component, sample_message_data):
    """Test updating an existing message."""
    timeline_id = "timeline_update_1"
    msg_data = sample_message_data(text="Original text")
    msg_id = msg_data["message_id"]
    history_component.add_message(msg_data, timeline_id)

    update_data = {"text": "Updated text", "status": "edited"}
    start_time = int(time.time() * 1000)
    updated = history_component.update_message(msg_id, update_data, timeline_id)
    end_time = int(time.time() * 1000)

    assert updated is True
    stored_msg = history_component._state["messages"][timeline_id][msg_id]
    assert stored_msg["text"] == "Updated text"
    assert stored_msg["status"] == "edited"
    assert stored_msg["sender"] == "user" # Original field preserved
    assert "last_updated" in stored_msg
    assert start_time <= stored_msg["last_updated"] <= end_time

def test_update_message_not_found(history_component):
    """Test updating a non-existent message."""
    timeline_id = "timeline_update_2"
    history_component._ensure_timeline_history(timeline_id)
    updated = history_component.update_message("non_existent_id", {"text": "Fail"}, timeline_id)
    assert updated is False

def test_update_message_wrong_timeline(history_component, sample_message_data):
    """Test updating a message in the wrong timeline."""
    timeline_id_1 = "timeline_update_3a"
    timeline_id_2 = "timeline_update_3b"
    msg_data = sample_message_data()
    msg_id = msg_data["message_id"]
    history_component.add_message(msg_data, timeline_id_1)
    history_component._ensure_timeline_history(timeline_id_2)

    updated = history_component.update_message(msg_id, {"text": "Fail"}, timeline_id_2)
    assert updated is False
    assert history_component._state["messages"][timeline_id_1][msg_id]["text"] != "Fail"

def test_delete_message_success(history_component, sample_message_data):
    """Test deleting an existing message."""
    timeline_id = "timeline_delete_1"
    msg_data = sample_message_data()
    msg_id = msg_data["message_id"]
    history_component.add_message(msg_data, timeline_id)
    assert msg_id in history_component._state["messages"][timeline_id]

    deleted = history_component.delete_message(msg_id, timeline_id)
    assert deleted is True
    assert msg_id not in history_component._state["messages"][timeline_id]

def test_delete_message_not_found(history_component):
    """Test deleting a non-existent message."""
    timeline_id = "timeline_delete_2"
    history_component._ensure_timeline_history(timeline_id)
    deleted = history_component.delete_message("non_existent_id", timeline_id)
    assert deleted is False

def test_get_messages_empty(history_component):
    """Test getting messages from an empty or non-existent timeline."""
    assert history_component.get_messages("non_existent_timeline") == []
    timeline_id = "timeline_get_1"
    history_component._ensure_timeline_history(timeline_id)
    assert history_component.get_messages(timeline_id) == []

def test_get_messages_sorted_and_limited(history_component, sample_message_data):
    """Test getting messages, ensuring sorting and limit."""
    timeline_id = "timeline_get_2"
    msg1_data = sample_message_data(text="Msg1", timestamp=1000)
    msg2_data = sample_message_data(text="Msg2", timestamp=500)
    msg3_data = sample_message_data(text="Msg3", timestamp=1500)

    history_component.add_message(msg1_data, timeline_id)
    history_component.add_message(msg2_data, timeline_id)
    history_component.add_message(msg3_data, timeline_id)

    # Get all, check sorting
    all_msgs = history_component.get_messages(timeline_id)
    assert len(all_msgs) == 3
    assert all_msgs[0]["text"] == "Msg2" # timestamp 500
    assert all_msgs[1]["text"] == "Msg1" # timestamp 1000
    assert all_msgs[2]["text"] == "Msg3" # timestamp 1500

    # Get limited (latest 2)
    limited_msgs = history_component.get_messages(timeline_id, limit=2)
    assert len(limited_msgs) == 2
    assert limited_msgs[0]["text"] == "Msg1" # timestamp 1000
    assert limited_msgs[1]["text"] == "Msg3" # timestamp 1500

    # Get limited (more than available)
    limited_msgs_over = history_component.get_messages(timeline_id, limit=5)
    assert len(limited_msgs_over) == 3
    assert limited_msgs_over[0]["text"] == "Msg2"

def test_clear_history_success(history_component, sample_message_data):
    """Test clearing history for a timeline."""
    timeline_id = "timeline_clear_1"
    history_component.add_message(sample_message_data(), timeline_id)
    history_component.add_message(sample_message_data(), timeline_id)
    assert len(history_component._state["messages"][timeline_id]) == 2

    cleared = history_component.clear_history(timeline_id)
    assert cleared is True
    assert timeline_id in history_component._state["messages"] # Key remains
    assert history_component._state["messages"][timeline_id] == {}
    assert history_component.get_messages(timeline_id) == []

def test_clear_history_non_existent(history_component):
    """Test clearing history for a non-existent timeline."""
    cleared = history_component.clear_history("non_existent_timeline")
    assert cleared is False

# --- Event Handling Tests ---

@pytest.mark.parametrize("event_type", ["message_received", "message_sent"])
def test_on_event_add_message(history_component, sample_message_data, event_type):
    """Test handling message received/sent events."""
    timeline_id = "timeline_event_add"
    msg_data = sample_message_data(text="Event message")
    msg_id = msg_data["message_id"]
    event = {"event_type": event_type, "data": msg_data}
    timeline_context = {"timeline_id": timeline_id}

    handled = history_component._on_event(event, timeline_context)

    assert handled is True
    assert msg_id in history_component._state["messages"][timeline_id]
    assert history_component._state["messages"][timeline_id][msg_id]["text"] == "Event message"

def test_on_event_update_message(history_component, sample_message_data):
    """Test handling message updated event."""
    timeline_id = "timeline_event_update"
    msg_data = sample_message_data(text="Original")
    msg_id = msg_data["message_id"]
    history_component.add_message(msg_data, timeline_id)

    update_data = {"message_id": msg_id, "text": "Updated via event", "status": "edited"}
    event = {"event_type": "message_updated", "data": update_data}
    timeline_context = {"timeline_id": timeline_id}

    handled = history_component._on_event(event, timeline_context)

    assert handled is True
    stored_msg = history_component._state["messages"][timeline_id][msg_id]
    assert stored_msg["text"] == "Updated via event"
    assert stored_msg["status"] == "edited"
    assert "last_updated" in stored_msg

def test_on_event_update_message_no_id(history_component):
    """Test handling message updated event without message_id."""
    event = {"event_type": "message_updated", "data": {"text": "fail"}}
    timeline_context = {"timeline_id": "timeline_event_update_fail"}
    handled = history_component._on_event(event, timeline_context)
    assert handled is False

def test_on_event_delete_message(history_component, sample_message_data):
    """Test handling message deleted event."""
    timeline_id = "timeline_event_delete"
    msg_data = sample_message_data()
    msg_id = msg_data["message_id"]
    history_component.add_message(msg_data, timeline_id)

    event = {"event_type": "message_deleted", "data": {"message_id": msg_id}}
    timeline_context = {"timeline_id": timeline_id}

    handled = history_component._on_event(event, timeline_context)

    assert handled is True
    assert msg_id not in history_component._state["messages"][timeline_id]

def test_on_event_delete_message_no_id(history_component):
    """Test handling message deleted event without message_id."""
    event = {"event_type": "message_deleted", "data": {}}
    timeline_context = {"timeline_id": "timeline_event_delete_fail"}
    handled = history_component._on_event(event, timeline_context)
    assert handled is False

def test_on_event_clear_context(history_component, sample_message_data):
    """Test handling clear context event."""
    timeline_id = "timeline_event_clear"
    history_component.add_message(sample_message_data(), timeline_id)
    assert len(history_component._state["messages"][timeline_id]) > 0

    event = {"event_type": "clear_context", "data": {}}
    timeline_context = {"timeline_id": timeline_id}

    handled = history_component._on_event(event, timeline_context)

    assert handled is True
    assert history_component._state["messages"][timeline_id] == {}

def test_on_event_no_timeline_id_uses_primary(history_component, sample_message_data, mock_timeline_component):
    """Test event handling uses primary timeline ID if not in context."""
    primary_timeline_id = "primary_timeline_1"
    msg_data = sample_message_data(text="Primary timeline message")
    msg_id = msg_data["message_id"]
    event = {"event_type": "message_received", "data": msg_data}
    timeline_context = {} # No timeline_id here

    handled = history_component._on_event(event, timeline_context)

    assert handled is True
    # Check that the mock was called to get the primary timeline
    mock_timeline_component.get_primary_timeline.assert_called_once()
    # Check message was added to the primary timeline
    assert primary_timeline_id in history_component._state["messages"]
    assert msg_id in history_component._state["messages"][primary_timeline_id]
    assert history_component._state["messages"][primary_timeline_id][msg_id]["text"] == "Primary timeline message"

def test_on_event_no_timeline_id_and_no_primary(history_component, sample_message_data, mock_timeline_component):
    """Test event handling fails if no timeline_id and no primary timeline."""
    mock_timeline_component.get_primary_timeline.return_value = None # Simulate no primary
    event = {"event_type": "message_received", "data": sample_message_data()}
    timeline_context = {}

    handled = history_component._on_event(event, timeline_context)

    assert handled is False
    mock_timeline_component.get_primary_timeline.assert_called_once()
    assert history_component._state["messages"] == {} # No messages added

def test_on_event_no_timeline_component(sample_message_data):
    """Test event handling fails if element has no TimelineComponent."""
    # Create component without a mock element initially
    comp_no_element = HistoryComponent(element=None)
    comp_no_element._initialize()
    comp_no_element._enable()

    event = {"event_type": "message_received", "data": sample_message_data()}
    timeline_context = {}

    # Mock element without timeline component
    element_no_timeline = MagicMock(spec=BaseElement)
    element_no_timeline.get_component_by_type.return_value = None
    comp_no_element.element = element_no_timeline

    handled = comp_no_element._on_event(event, timeline_context)

    assert handled is False
    assert comp_no_element._state["messages"] == {}

def test_on_event_unhandled_type(history_component):
    """Test that unhandled event types are ignored."""
    event = {"event_type": "some_other_event", "data": {}}
    timeline_context = {"timeline_id": "timeline_event_unhandled"}
    handled = history_component._on_event(event, timeline_context)
    assert handled is False 