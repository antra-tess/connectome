import pytest
import time
from unittest.mock import MagicMock, patch, call

from elements.elements.components.global_attention_component import GlobalAttentionComponent
from elements.elements.base_element import BaseElement

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_element():
    """Fixture for a mocked BaseElement (optional, component doesn't use it heavily)."""
    element = MagicMock(spec=BaseElement)
    element.id = "attention_manager_element"
    return element

@pytest.fixture
def attention_component(mock_element):
    """Fixture for a GlobalAttentionComponent instance."""
    comp = GlobalAttentionComponent(element=mock_element)
    comp._initialize()
    comp._enable()
    # Reset state for clean tests
    comp._state = {"attention_requests": {}}
    return comp

# --- Helper to create events ---
def create_attention_event(event_type, space_id, source_element_id, element_id=None, request_data=None, timestamp=None):
    """Helper to create consistent attention event structures."""
    event_data = {
        "space_id": space_id,
        "source_element_id": source_element_id,
        "element_id": element_id or source_element_id, # Default element_id to source if not provided
        "request_data": request_data or {},
        "timestamp": timestamp or int(time.time() * 1000)
    }
    return {"event_type": event_type, "data": event_data}

# --- Test Cases ---

def test_attention_component_initialization(attention_component, mock_element):
    """Test correct initialization."""
    assert attention_component.element == mock_element
    assert attention_component._is_initialized
    assert attention_component._is_enabled
    assert attention_component.COMPONENT_TYPE == "global_attention_manager"
    assert not attention_component.DEPENDENCIES
    assert attention_component._state == {"attention_requests": {}}
    assert attention_component.HANDLED_EVENT_TYPES == ["attention_requested", "attention_cleared"]

# --- _on_event Tests ---

@patch('time.time', return_value=1700007000.0)
def test_on_event_attention_requested_new(mock_time, attention_component):
    """Test handling a new attention_requested event."""
    space = "space_A"
    source_el = "element_1"
    req_data = {"focus": "input_field"}
    timeline_ctx = {"timeline_id": "timeline_X"}
    event = create_attention_event("attention_requested", space, source_el, request_data=req_data)
    expected_key = f"{space}::{source_el}"

    with patch('logging.Logger.info') as mock_log_info:
        handled = attention_component._on_event(event, timeline_ctx)

    assert handled is True
    assert expected_key in attention_component._state["attention_requests"]
    stored_request = attention_component._state["attention_requests"][expected_key]
    assert stored_request["space_id"] == space
    assert stored_request["source_element_id"] == source_el
    assert stored_request["element_id"] == source_el # Defaulted
    assert stored_request["request_data"] == req_data
    assert stored_request["timestamp"] == 1700007000000
    assert stored_request["timeline_context"] == timeline_ctx
    mock_log_info.assert_called_once_with(f"New attention request from {expected_key}")

@patch('time.time', side_effect=[1700008000.0, 1700008010.0]) # Two timestamps
def test_on_event_attention_requested_update(mock_time, attention_component):
    """Test handling an update to an existing attention request."""
    space = "space_B"
    source_el = "element_2"
    timeline_ctx = {"timeline_id": "timeline_Y"}
    expected_key = f"{space}::{source_el}"

    # First request
    event1 = create_attention_event("attention_requested", space, source_el, request_data={"initial": True})
    attention_component._on_event(event1, timeline_ctx)
    assert attention_component._state["attention_requests"][expected_key]["timestamp"] == 1700008000000
    assert attention_component._state["attention_requests"][expected_key]["request_data"] == {"initial": True}


    # Second (update) request
    event2 = create_attention_event("attention_requested", space, source_el, request_data={"updated": True})
    with patch('logging.Logger.debug') as mock_log_debug:
        handled = attention_component._on_event(event2, timeline_ctx)

    assert handled is True
    assert expected_key in attention_component._state["attention_requests"]
    stored_request = attention_component._state["attention_requests"][expected_key]
    assert stored_request["request_data"] == {"updated": True} # Data updated
    assert stored_request["timestamp"] == 1700008010000 # Timestamp updated
    mock_log_debug.assert_called_once_with(f"Updating existing attention request for {expected_key}")


@pytest.mark.parametrize("event_type", ["attention_requested", "attention_cleared"])
@pytest.mark.parametrize("missing_field", ["space_id", "source_element_id"])
def test_on_event_missing_ids(attention_component, event_type, missing_field):
    """Test events fail if space_id or source_element_id are missing."""
    event_data = {
        "space_id": "space_C",
        "source_element_id": "element_3"
    }
    event_data[missing_field] = None # Make one field None
    event = {"event_type": event_type, "data": event_data}

    with patch('logging.Logger.warning') as mock_log_warning:
        handled = attention_component._on_event(event, {})

    assert handled is False
    assert not attention_component._state["attention_requests"] # State should remain empty
    mock_log_warning.assert_called_once_with(
       f"{attention_component.COMPONENT_TYPE}: Received {event_type} without space_id or source_element_id."
    )

def test_on_event_attention_cleared_success(attention_component):
    """Test clearing an existing attention request."""
    space = "space_D"
    source_el = "element_4"
    expected_key = f"{space}::{source_el}"

    # Add a request first
    add_event = create_attention_event("attention_requested", space, source_el)
    attention_component._on_event(add_event, {})
    assert expected_key in attention_component._state["attention_requests"]

    # Now clear it
    clear_event = create_attention_event("attention_cleared", space, source_el)
    with patch('logging.Logger.info') as mock_log_info:
        handled = attention_component._on_event(clear_event, {})

    assert handled is True
    assert expected_key not in attention_component._state["attention_requests"]
    mock_log_info.assert_called_once_with(f"Clearing attention request for {expected_key}")

def test_on_event_attention_cleared_non_existent(attention_component):
    """Test clearing a non-existent attention request."""
    space = "space_E"
    source_el = "element_5"
    expected_key = f"{space}::{source_el}"

    assert expected_key not in attention_component._state["attention_requests"]
    clear_event = create_attention_event("attention_cleared", space, source_el)

    with patch('logging.Logger.debug') as mock_log_debug:
        handled = attention_component._on_event(clear_event, {})

    assert handled is False # Should return False if key didn't exist
    assert expected_key not in attention_component._state["attention_requests"]
    mock_log_debug.assert_called_once_with(f"Received attention_cleared for non-existent request key: {expected_key}")

def test_on_event_unhandled_type(attention_component):
    """Test that unhandled event types return False."""
    event = {"event_type": "some_other_event", "data": {"space_id": "s", "source_element_id": "e"}}
    handled = attention_component._on_event(event, {})
    assert handled is False

# --- Getter and Clearer Method Tests ---

def test_get_attention_requests_empty(attention_component):
    """Test getting requests when none exist."""
    assert attention_component.get_attention_requests() == {}

def test_get_attention_requests_populated(attention_component):
    """Test getting requests after adding some."""
    # Add two requests
    event1 = create_attention_event("attention_requested", "space_F", "element_6")
    event2 = create_attention_event("attention_requested", "space_G", "element_7")
    attention_component._on_event(event1, {})
    attention_component._on_event(event2, {})

    requests = attention_component.get_attention_requests()
    assert len(requests) == 2
    assert "space_F::element_6" in requests
    assert "space_G::element_7" in requests

    # Verify it's a copy
    requests["new_key"] = "test"
    assert "new_key" not in attention_component._state["attention_requests"]


def test_clear_all_requests_populated(attention_component):
    """Test clearing all requests when some exist."""
     # Add two requests
    event1 = create_attention_event("attention_requested", "space_H", "element_8")
    event2 = create_attention_event("attention_requested", "space_I", "element_9")
    attention_component._on_event(event1, {})
    attention_component._on_event(event2, {})
    assert len(attention_component._state["attention_requests"]) == 2

    with patch('logging.Logger.info') as mock_log_info:
        attention_component.clear_all_requests()

    assert attention_component._state["attention_requests"] == {}
    mock_log_info.assert_called_once_with("Clearing all attention requests.")

def test_clear_all_requests_empty(attention_component):
    """Test clearing requests when the state is already empty."""
    assert not attention_component._state["attention_requests"]
    with patch('logging.Logger.info') as mock_log_info:
         attention_component.clear_all_requests()
    assert attention_component._state["attention_requests"] == {}
    mock_log_info.assert_not_called() # Should not log if nothing to clear 