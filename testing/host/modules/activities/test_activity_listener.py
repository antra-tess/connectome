import pytest
import time
from unittest.mock import MagicMock, patch, call, ANY
from typing import Dict, Any, Optional

try:
    # Class to test
    from host.modules.activities.activity_listener import ActivityListener
    # Dependencies to mock
    from elements.space_registry import SpaceRegistry
    from host.event_loop import HostEventLoop

except ImportError as e:
    pytest.skip(f"Skipping ActivityListener tests due to import error: {e}", allow_module_level=True)

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_space_registry():
    # Not currently used by listener, but required by __init__
    return MagicMock(spec=SpaceRegistry)

@pytest.fixture
def mock_event_loop():
    loop = MagicMock(spec=HostEventLoop)
    # Mock the actual enqueue method name
    loop.enqueue_incoming_event = MagicMock()
    return loop

@pytest.fixture
def activity_listener(mock_space_registry):
    # Initialize without event loop initially
    listener = ActivityListener(space_registry=mock_space_registry)
    return listener

# --- Test Cases ---

def test_initialization(activity_listener, mock_space_registry):
    """Test constructor stores dependencies."""
    assert activity_listener.space_registry == mock_space_registry
    assert activity_listener._event_loop is None

def test_set_event_loop(activity_listener, mock_event_loop):
    """Test setting the event loop reference."""
    assert activity_listener._event_loop is None
    activity_listener.set_event_loop(mock_event_loop)
    assert activity_listener._event_loop == mock_event_loop

# --- Timeline Context Determination Tests ---

def test_determine_timeline_context_full_ids(activity_listener):
    """Test context determination with conversation and thread IDs."""
    event_data = {
        "adapter_type": "discord",
        "conversation_id": "channel123",
        "thread_id": "thread456",
        "timestamp": 1234567890000
    }
    context = activity_listener._determine_timeline_context(event_data)
    assert context["timeline_id"] == "discord_channel123_thread_thread456"
    assert context["is_primary"] is True
    assert context["last_event_id"] is None
    assert context["timestamp"] == 1234567890000

def test_determine_timeline_context_conversation_only(activity_listener):
    """Test context determination with only conversation ID."""
    event_data = {
        "adapter_type": "slack",
        "conversation_id": "general"
    }
    context = activity_listener._determine_timeline_context(event_data)
    assert context["timeline_id"] == "slack_general"
    assert "timestamp" in context # Should add current time if missing

def test_determine_timeline_context_unknown_adapter(activity_listener):
    """Test context determination with missing adapter_type."""
    event_data = {
        "conversation_id": "conv999"
    }
    context = activity_listener._determine_timeline_context(event_data)
    assert context["timeline_id"] == "unknown_conv999"

def test_determine_timeline_context_missing_conversation_id(activity_listener, caplog):
    """Test fallback timeline ID generation when conversation_id is missing."""
    event_data = {
        "adapter_type": "email"
        # Missing conversation_id
    }
    with caplog.at_level(logging.WARNING):
        context = activity_listener._determine_timeline_context(event_data)

    assert context["timeline_id"].startswith("timeline_")
    assert len(context["timeline_id"]) == len("timeline_") + 8 # Check length of random part
    assert "Missing 'conversation_id' in normalized event" in caplog.text

# --- Event Handling Tests ---

def test_handle_incoming_event_success(activity_listener, mock_event_loop):
    """Test successfully handling and enqueuing a valid event."""
    activity_listener.set_event_loop(mock_event_loop)
    event_data = {
        "event_type": "message_received",
        "adapter_type": "test_adapter",
        "conversation_id": "test_conv",
        "data": {"text": "hello"}
    }
    # Keep a copy because the handler might modify it (add IDs/timestamps)
    original_event_data = event_data.copy()

    # Patch context determination to verify its call and control output
    expected_context = {"timeline_id": "test_adapter_test_conv", "is_primary": True, "last_event_id": None, "timestamp": ANY}
    with patch.object(activity_listener, '_determine_timeline_context', return_value=expected_context) as mock_determine_context:
        success = activity_listener.handle_incoming_event(event_data)

    assert success is True
    mock_determine_context.assert_called_once()
    # Event data might have ID/timestamp added, check enqueue args carefully
    # Use ANY for timestamp as it's generated if missing
    mock_event_loop.enqueue_incoming_event.assert_called_once() # Corrected method name
    call_args, _ = mock_event_loop.enqueue_incoming_event.call_args
    enqueued_event, enqueued_context = call_args
    
    # Check basic structure and that original data is present
    assert enqueued_event["event_type"] == original_event_data["event_type"]
    assert enqueued_event["data"] == original_event_data["data"]
    assert "event_id" in enqueued_event # Should be added if missing
    assert "timestamp" in enqueued_event # Should be added if missing
    
    assert enqueued_context == expected_context

def test_handle_incoming_event_no_loop(activity_listener, caplog):
    """Test handling fails if event loop isn't set."""
    event_data = {"event_type": "message_received"}
    # Event loop is NOT set on the listener fixture initially
    with caplog.at_level(logging.ERROR):
        success = activity_listener.handle_incoming_event(event_data)

    assert success is False
    assert "HostEventLoop reference not set" in caplog.text

def test_handle_incoming_event_invalid_format_not_dict(activity_listener, mock_event_loop, caplog):
    """Test handling fails for non-dict input."""
    activity_listener.set_event_loop(mock_event_loop)
    with caplog.at_level(logging.ERROR):
        success = activity_listener.handle_incoming_event("not a dict")
    assert success is False
    assert "Invalid normalized event format" in caplog.text
    mock_event_loop.enqueue_incoming_event.assert_not_called()

def test_handle_incoming_event_invalid_format_no_type(activity_listener, mock_event_loop, caplog):
    """Test handling fails for dict missing 'event_type'"""
    activity_listener.set_event_loop(mock_event_loop)
    event_data = {"data": "something"} # Missing event_type
    with caplog.at_level(logging.ERROR):
        success = activity_listener.handle_incoming_event(event_data)
    assert success is False
    assert "Invalid normalized event format" in caplog.text
    mock_event_loop.enqueue_incoming_event.assert_not_called()

def test_handle_incoming_event_adds_missing_ids(activity_listener, mock_event_loop):
    """Test that missing event_id and timestamp are added."""
    activity_listener.set_event_loop(mock_event_loop)
    event_data = {
        "event_type": "test_event",
        "adapter_type": "test",
        "conversation_id": "conv1"
        # Missing event_id and timestamp
    }
    
    success = activity_listener.handle_incoming_event(event_data.copy()) # Pass copy
    
    assert success is True
    mock_event_loop.enqueue_incoming_event.assert_called_once()
    call_args, _ = mock_event_loop.enqueue_incoming_event.call_args
    enqueued_event, _ = call_args
    assert "event_id" in enqueued_event
    assert enqueued_event["event_id"].startswith("evt_")
    assert "timestamp" in enqueued_event
    assert isinstance(enqueued_event["timestamp"], int) 