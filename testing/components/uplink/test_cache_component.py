import pytest
import time
from unittest.mock import MagicMock, patch, call

from elements.elements.components.uplink.cache_component import RemoteStateCacheComponent
from elements.elements.components.uplink.connection_component import UplinkConnectionComponent
from elements.elements.base_element import BaseElement

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_connection_component():
    """Fixture for a mocked UplinkConnectionComponent."""
    mock = MagicMock(spec=UplinkConnectionComponent)
    mock.COMPONENT_TYPE = "uplink_connection"
    # Default to connected state
    mock.get_connection_state.return_value = {"connected": True, "status": "active"}
    return mock

@pytest.fixture
def mock_element(mock_connection_component):
    """Fixture for a mocked BaseElement containing mocked components."""
    element = MagicMock(spec=BaseElement)
    element.get_component_by_type.return_value = mock_connection_component
    element.handle_event = MagicMock() # Mock event handling
    return element

@pytest.fixture
def cache_component(mock_element):
    """Fixture for a CacheComponent instance."""
    comp = RemoteStateCacheComponent(element=mock_element, cache_ttl=10, remote_space_id="remote_test_space")
    comp._initialize()
    comp._enable()
    # Reset state for clean tests
    comp._state = {
        "remote_state_cache": {},
        "history_bundles": {},
        "processed_remote_events": set(),
        "last_successful_sync": None,
        "last_sync_attempt": None,
        "cache_expiry": {},
        "cache_ttl": 10 # Match constructor for consistency
    }
    return comp

# --- Test Cases ---

def test_cache_component_initialization(cache_component, mock_element):
    """Test correct initialization."""
    assert cache_component.element == mock_element
    assert cache_component.remote_space_id == "remote_test_space"
    assert cache_component._state["cache_ttl"] == 10
    assert cache_component._is_initialized
    assert cache_component._is_enabled
    assert cache_component.COMPONENT_TYPE == "remote_state_cache"
    assert "uplink_connection" in cache_component.DEPENDENCIES

def test_initialization_default_remote_id():
    """Test initialization uses default remote ID if none provided."""
    comp = RemoteStateCacheComponent()
    assert comp.remote_space_id == "unknown_remote"

def test_get_connection_comp(cache_component, mock_connection_component):
    """Test retrieving the dependency component."""
    assert cache_component._get_connection_comp() == mock_connection_component

@patch('time.time', return_value=1700000000.0) # Mock time for predictable expiry
def test_sync_remote_state_success(mock_time, cache_component, mock_element):
    """Test successful state synchronization simulation."""
    assert not cache_component._state["remote_state_cache"]
    assert cache_component._state["last_successful_sync"] is None

    # Mock uuid to make generated state predictable if needed
    with patch('uuid.uuid4') as mock_uuid:
        mock_uuid.return_value.hex = '1234567890abcdef' # Simplified hex
        success = cache_component.sync_remote_state()

    assert success is True
    assert cache_component._state["last_sync_attempt"] == 1700000000000
    assert cache_component._state["last_successful_sync"] == 1700000000000

    # Check cache content (based on simulation in component)
    assert "remote_element_1234" in cache_component._state["remote_state_cache"]
    assert cache_component._state["remote_state_cache"]["remote_element_1234"]["name"] == "Remote Item"
    assert "span_1699990000000_1700000000000" in cache_component._state["history_bundles"]

    # Check expiry update
    expected_expiry = 1700000000000 + (10 * 1000) # ttl is 10 seconds
    assert cache_component._state["cache_expiry"]["full_state"] == expected_expiry

    # Check event emission
    mock_element.handle_event.assert_called_once()
    call_args, call_kwargs = mock_element.handle_event.call_args
    assert call_args[0]["event_type"] == "uplink_state_synced"
    assert call_args[0]["data"]["remote_space_id"] == "remote_test_space"
    assert call_args[0]["data"]["cache_size"] > 0
    assert call_args[1] == {"timeline_id": "primary"}

def test_sync_remote_state_not_connected(cache_component, mock_connection_component):
    """Test sync fails if connection component reports not connected."""
    mock_connection_component.get_connection_state.return_value = {"connected": False}
    success = cache_component.sync_remote_state()
    assert success is False
    assert cache_component._state["last_successful_sync"] is None
    assert not cache_component._state["remote_state_cache"] # Cache remains empty

def test_sync_remote_state_no_connection_component(cache_component, mock_element):
    """Test sync fails if connection component is missing."""
    mock_element.get_component_by_type.return_value = None # Simulate missing component
    success = cache_component.sync_remote_state()
    assert success is False
    assert cache_component._state["last_successful_sync"] is None

def test_sync_remote_state_disabled(cache_component):
    """Test sync fails if component is disabled."""
    cache_component._disable()
    success = cache_component.sync_remote_state()
    assert success is False

def test_is_cache_stale_no_expiry(cache_component):
    """Test cache is stale if no expiry is set."""
    assert "full_state" not in cache_component._state["cache_expiry"]
    assert cache_component._is_cache_stale() is True

@patch('time.time', return_value=1700000000.0)
def test_is_cache_stale_not_expired(mock_time, cache_component):
    """Test cache is not stale if expiry is in the future."""
    cache_component._state["cache_expiry"]["full_state"] = 1700000015000 # Expires in 15s
    assert cache_component._is_cache_stale() is False

@patch('time.time', return_value=1700000020.0)
def test_is_cache_stale_expired(mock_time, cache_component):
    """Test cache is stale if expiry is in the past."""
    cache_component._state["cache_expiry"]["full_state"] = 1700000015000 # Expired 5s ago
    assert cache_component._is_cache_stale() is True

@patch.object(RemoteStateCacheComponent, 'sync_remote_state', return_value=True)
def test_get_synced_remote_state_no_sync_needed(mock_sync, cache_component):
    """Test getting state when cache is fresh, no sync triggered."""
    # Simulate fresh cache
    with patch.object(cache_component, '_is_cache_stale', return_value=False):
        cache_component._state["remote_state_cache"] = {"key": "value"}
        state = cache_component.get_synced_remote_state()

    assert state == {"key": "value"}
    mock_sync.assert_not_called()

@patch.object(RemoteStateCacheComponent, 'sync_remote_state', return_value=True)
def test_get_synced_remote_state_stale_cache(mock_sync, cache_component):
    """Test getting state triggers sync when cache is stale."""
    # Simulate stale cache
    with patch.object(cache_component, '_is_cache_stale', return_value=True):
         # Simulate sync adding data
        def side_effect(*args, **kwargs):
             cache_component._state["remote_state_cache"] = {"new_key": "new_value"}
             return True
        mock_sync.side_effect = side_effect

        state = cache_component.get_synced_remote_state()

    assert state == {"new_key": "new_value"}
    mock_sync.assert_called_once()

@patch.object(RemoteStateCacheComponent, 'sync_remote_state', return_value=True)
def test_get_synced_remote_state_force_sync(mock_sync, cache_component):
    """Test getting state triggers sync when force_sync=True, even if cache is fresh."""
     # Simulate fresh cache
    with patch.object(cache_component, '_is_cache_stale', return_value=False):
        # Simulate sync adding data
        def side_effect(*args, **kwargs):
             cache_component._state["remote_state_cache"] = {"forced_key": "forced_value"}
             return True
        mock_sync.side_effect = side_effect

        state = cache_component.get_synced_remote_state(force_sync=True)

    assert state == {"forced_key": "forced_value"}
    mock_sync.assert_called_once()

def test_get_history_bundles(cache_component):
    """Test retrieving history bundles."""
    cache_component._state["history_bundles"] = {
        "span1": [{"event": "A"}],
        "span2": [{"event": "B"}],
        "span3": [{"event": "C"}],
    }

    # Get all
    all_bundles = cache_component.get_history_bundles()
    assert all_bundles == {"span1": [{"event": "A"}], "span2": [{"event": "B"}], "span3": [{"event": "C"}]}

    # Get specific subset
    subset_bundles = cache_component.get_history_bundles(span_ids=["span1", "span3", "span_missing"])
    assert subset_bundles == {"span1": [{"event": "A"}], "span3": [{"event": "C"}]} # span_missing ignored

    # Get empty
    empty_bundles = cache_component.get_history_bundles(span_ids=["span_missing"])
    assert empty_bundles == {}

# Note: Testing actual timer logic of enable/disable_auto_sync requires async framework/threading mocks
# We will test the state changes and logging associated with the placeholders.

@patch('logging.Logger.info')
@patch('logging.Logger.warning')
def test_enable_auto_sync(mock_log_warning, mock_log_info, cache_component):
    """Test enabling auto sync placeholder."""
    assert not cache_component._auto_sync_enabled
    cache_component.enable_auto_sync(interval=60)
    assert cache_component._auto_sync_enabled
    assert cache_component._state["cache_ttl"] == 60
    mock_log_info.assert_any_call(f"Enabling auto-sync for remote_test_space every 60 seconds.")
    mock_log_warning.assert_called_with("Auto-sync timer implementation required (e.g., asyncio).")
    # Call again, should not log enabling again if already enabled
    mock_log_info.reset_mock()
    cache_component.enable_auto_sync() # Use existing TTL
    assert cache_component._auto_sync_enabled
    mock_log_info.assert_not_called()

@patch('logging.Logger.info')
@patch('logging.Logger.warning')
def test_disable_auto_sync(mock_log_warning, mock_log_info, cache_component):
    """Test disabling auto sync placeholder."""
    # First enable it
    cache_component.enable_auto_sync(interval=30)
    assert cache_component._auto_sync_enabled

    # Now disable
    cache_component.disable_auto_sync()
    assert not cache_component._auto_sync_enabled
    mock_log_warning.assert_called_with("Auto-sync timer cancellation required.")
    mock_log_info.assert_any_call(f"Disabled auto-sync for remote_test_space.")

    # Call again, should not log if already disabled
    mock_log_info.reset_mock()
    cache_component.disable_auto_sync()
    assert not cache_component._auto_sync_enabled
    mock_log_info.assert_not_called()

@patch.object(RemoteStateCacheComponent, 'sync_remote_state', return_value=True)
def test_on_event_sync_request(mock_sync, cache_component):
    """Test handling of the 'sync_request' event."""
    event = {"event_type": "sync_request", "data": {}}
    timeline_context = {"timeline_id": "irrelevant"}

    handled = cache_component._on_event(event, timeline_context)
    assert handled is True
    mock_sync.assert_called_once()

def test_on_event_other_event(cache_component):
    """Test that other events are not handled."""
    event = {"event_type": "other_event", "data": {}}
    timeline_context = {"timeline_id": "irrelevant"}
    handled = cache_component._on_event(event, timeline_context)
    assert handled is False

@patch.object(RemoteStateCacheComponent, 'disable_auto_sync')
def test_on_cleanup(mock_disable_sync, cache_component):
    """Test that cleanup disables auto-sync."""
    # Enable first to ensure disable is meaningful
    cache_component.enable_auto_sync()
    mock_disable_sync.reset_mock() # Reset mock after enable call

    cleaned_up = cache_component._on_cleanup()
    assert cleaned_up is True
    mock_disable_sync.assert_called_once() 