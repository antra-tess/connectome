import pytest
import time
from unittest.mock import MagicMock

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
def mock_element():
    return MockElementWithTimeline()

@pytest.fixture
def history_component(mock_element):
    component = HistoryComponent(element=mock_element)
    # Manually initialize and enable for testing direct method calls
    component._is_initialized = True
    component._is_enabled = True
    return component

# --- Test Cases ---

def test_initialization(history_component, mock_element):
    """Test basic initialization."""
    assert history_component is not None
    assert history_component.COMPONENT_TYPE == "messaging_history"
    assert history_component.element == mock_element
    assert "messages" in history_component._state
    assert not history_component._state["messages"] # Initially empty

def test_add_message_success(history_component):
    """Test successfully adding a message."""
    timeline_id = "timeline1"
    msg_data = {"text": "Hello world", "sender": "user"}
    
    added = history_component.add_message(msg_data.copy(), timeline_id)
    assert added is True
    assert timeline_id in history_component._state["messages"]
    
    messages = history_component._state["messages"][timeline_id]
    assert len(messages) == 1
    message_id = list(messages.keys())[0]
    assert message_id.startswith("msg_")
    
    stored_message = messages[message_id]
    assert stored_message["text"] == "Hello world"
    assert stored_message["sender"] == "user"
    assert "timestamp" in stored_message
    assert "message_id" in stored_message
    assert message_id == stored_message["message_id"]

def test_add_message_with_id(history_component):
    """Test adding a message that already has an ID."""
    timeline_id = "timeline2"
    msg_id = "custom_msg_id_123"
    msg_data = {"message_id": msg_id, "text": "Test"}
    
    added = history_component.add_message(msg_data.copy(), timeline_id)
    assert added is True
    assert timeline_id in history_component._state["messages"]
    assert msg_id in history_component._state["messages"][timeline_id]
    assert history_component._state["messages"][timeline_id][msg_id]["text"] == "Test"

def test_add_message_ignored(history_component):
    """Test adding a message starting with an ignore marker."""
    timeline_id = "timeline1"
    msg_data = {"text": ".ignore this"}
    
    added = history_component.add_message(msg_data.copy(), timeline_id)
    assert added is True
    message_id = list(history_component._state["messages"][timeline_id].keys())[0]
    stored_message = history_component._state["messages"][timeline_id][message_id]
    assert stored_message.get("ignored") is True

def test_add_message_disabled(history_component):
    """Test adding fails if component is disabled."""
    history_component._is_enabled = False
    added = history_component.add_message({"text": "test"}, "timeline1")
    assert added is False

def test_update_message_success(history_component):
    """Test successfully updating a message."""
    timeline_id = "timeline_upd"
    msg_id = "msg_to_update"
    initial_data = {"message_id": msg_id, "text": "Initial text", "timestamp": time.time() * 1000}
    history_component.add_message(initial_data.copy(), timeline_id)
    
    update_data = {"text": "Updated text", "edited": True}
    updated = history_component.update_message(msg_id, update_data, timeline_id)
    
    assert updated is True
    stored_message = history_component._state["messages"][timeline_id][msg_id]
    assert stored_message["text"] == "Updated text"
    assert stored_message["edited"] is True
    assert "last_updated" in stored_message
    assert stored_message["last_updated"] > initial_data["timestamp"]

def test_update_message_not_found(history_component):
    """Test updating a non-existent message."""
    updated = history_component.update_message("not_a_real_id", {"text": "update"}, "timeline1")
    assert updated is False

def test_delete_message_success(history_component):
    """Test successfully deleting a message."""
    timeline_id = "timeline_del"
    msg_id = "msg_to_delete"
    history_component.add_message({"message_id": msg_id, "text": "delete me"}, timeline_id)
    assert msg_id in history_component._state["messages"][timeline_id]
    
    deleted = history_component.delete_message(msg_id, timeline_id)
    assert deleted is True
    assert msg_id not in history_component._state["messages"][timeline_id]

def test_delete_message_not_found(history_component):
    """Test deleting a non-existent message."""
    deleted = history_component.delete_message("not_a_real_id", "timeline1")
    assert deleted is False

def test_get_messages_sorted(history_component):
    """Test get_messages returns messages sorted by timestamp."""
    timeline_id = "timeline_get"
    msg1 = {"message_id": "m1", "text": "first", "timestamp": 1000}
    msg3 = {"message_id": "m3", "text": "third", "timestamp": 3000}
    msg2 = {"message_id": "m2", "text": "second", "timestamp": 2000}
    history_component.add_message(msg1, timeline_id)
    history_component.add_message(msg3, timeline_id)
    history_component.add_message(msg2, timeline_id)
    
    retrieved = history_component.get_messages(timeline_id)
    assert len(retrieved) == 3
    assert retrieved[0]["message_id"] == "m1"
    assert retrieved[1]["message_id"] == "m2"
    assert retrieved[2]["message_id"] == "m3"

def test_get_messages_limit(history_component):
    """Test get_messages respects the limit."""
    timeline_id = "timeline_limit"
    msg1 = {"message_id": "m1", "text": "first", "timestamp": 1000}
    msg2 = {"message_id": "m2", "text": "second", "timestamp": 2000}
    msg3 = {"message_id": "m3", "text": "third", "timestamp": 3000}
    history_component.add_message(msg1, timeline_id)
    history_component.add_message(msg2, timeline_id)
    history_component.add_message(msg3, timeline_id)
    
    retrieved = history_component.get_messages(timeline_id, limit=2)
    assert len(retrieved) == 2
    assert retrieved[0]["message_id"] == "m2" # Should get the latest 2
    assert retrieved[1]["message_id"] == "m3"

def test_get_messages_empty(history_component):
    """Test get_messages returns empty list for unknown timeline."""
    retrieved = history_component.get_messages("non_existent_timeline")
    assert retrieved == []

def test_clear_history(history_component):
    """Test clearing history for a timeline."""
    timeline_id = "timeline_clear"
    history_component.add_message({"text": "message 1"}, timeline_id)
    assert len(history_component._state["messages"][timeline_id]) == 1
    
    cleared = history_component.clear_history(timeline_id)
    assert cleared is True
    assert len(history_component._state["messages"][timeline_id]) == 0
    # Ensure other timelines are unaffected
    history_component.add_message({"text": "other msg"}, "other_timeline")
    assert len(history_component._state["messages"]["other_timeline"]) == 1

def test_clear_history_non_existent(history_component):
    """Test clearing history for a timeline that doesn't exist."""
    cleared = history_component.clear_history("non_existent_timeline")
    assert cleared is False

# --- Event Handling Tests ---

@pytest.mark.parametrize("event_type", ["message_received", "message_sent"])
def test_on_event_add(history_component, event_type):
    """Test _on_event adding messages."""
    timeline_id = "event_timeline"
    msg_data = {"text": f"From {event_type}"}
    event = {"event_type": event_type, "data": msg_data}
    timeline_context = {"timeline_id": timeline_id}
    
    handled = history_component._on_event(event, timeline_context)
    assert handled is True
    messages = history_component.get_messages(timeline_id)
    assert len(messages) == 1
    assert messages[0]["text"] == f"From {event_type}"

def test_on_event_update(history_component):
    """Test _on_event updating a message."""
    timeline_id = "event_timeline_upd"
    msg_id = "event_msg_upd"
    history_component.add_message({"message_id": msg_id, "text": "initial"}, timeline_id)
    
    update_data = {"message_id": msg_id, "text": "updated by event", "status": "edited"}
    event = {"event_type": "message_updated", "data": update_data}
    timeline_context = {"timeline_id": timeline_id}
    
    handled = history_component._on_event(event, timeline_context)
    assert handled is True
    message = history_component.get_messages(timeline_id)[0]
    assert message["text"] == "updated by event"
    assert message["status"] == "edited"

def test_on_event_delete(history_component):
    """Test _on_event deleting a message."""
    timeline_id = "event_timeline_del"
    msg_id = "event_msg_del"
    history_component.add_message({"message_id": msg_id, "text": "to be deleted"}, timeline_id)
    assert len(history_component.get_messages(timeline_id)) == 1
    
    delete_data = {"message_id": msg_id}
    event = {"event_type": "message_deleted", "data": delete_data}
    timeline_context = {"timeline_id": timeline_id}
    
    handled = history_component._on_event(event, timeline_context)
    assert handled is True
    assert len(history_component.get_messages(timeline_id)) == 0

def test_on_event_clear_context(history_component):
    """Test _on_event clearing history."""
    timeline_id = "event_timeline_clear"
    history_component.add_message({"text": "msg1"}, timeline_id)
    assert len(history_component.get_messages(timeline_id)) == 1
    
    event = {"event_type": "clear_context", "data": {}}
    timeline_context = {"timeline_id": timeline_id}
    
    handled = history_component._on_event(event, timeline_context)
    assert handled is True
    assert len(history_component.get_messages(timeline_id)) == 0
    
def test_on_event_missing_timeline_id_uses_primary(history_component, mock_element):
    """Test _on_event uses primary timeline ID if context lacks one."""
    primary_id = "primary_from_mock"
    mock_element._mock_timeline_comp.get_primary_timeline.return_value = primary_id
    
    event = {"event_type": "message_received", "data": {"text": "hello"}}
    timeline_context = {} # No timeline_id
    
    handled = history_component._on_event(event, timeline_context)
    assert handled is True
    # Check if message was added to the primary timeline
    messages = history_component.get_messages(primary_id)
    assert len(messages) == 1
    assert messages[0]["text"] == "hello"

def test_on_event_missing_timeline_id_and_no_primary(history_component, mock_element):
    """Test _on_event fails if timeline ID is missing and no primary exists."""
    mock_element._mock_timeline_comp.get_primary_timeline.return_value = None # Simulate no primary
    
    event = {"event_type": "message_received", "data": {"text": "hello"}}
    timeline_context = {} # No timeline_id
    
    handled = history_component._on_event(event, timeline_context)
    assert handled is False

def test_on_event_unhandled_type(history_component):
    """Test _on_event returns False for unhandled event types."""
    event = {"event_type": "some_other_event", "data": {}}
    timeline_context = {"timeline_id": "timeline1"}
    handled = history_component._on_event(event, timeline_context)
    assert handled is False 