import pytest
import asyncio
from unittest.mock import MagicMock, patch, call, ANY
from typing import Dict, Any, Optional, List

try:
    # Class to test
    from host.modules.activities.activity_client import ActivityClient
    # Dependencies
    from host.event_loop import HostEventLoop
    # Mock socketio - IMPORTANT: We need to mock the *class* before import if methods are defined at class level
    # Or mock specific instances and methods after import
    import socketio # Import the actual library to mock its classes/exceptions
except ImportError as e:
    pytest.skip(f"Skipping ActivityClient tests due to import error: {e} (is python-socketio installed?)", allow_module_level=True)

# --- Mocks and Fixtures ---

ADAPTER_ID_1 = "adapter_api_1"
ADAPTER_URL_1 = "http://localhost:5001"
ADAPTER_ID_2 = "adapter_api_2"
ADAPTER_URL_2 = "http://localhost:5002"

@pytest.fixture
def mock_host_event_loop():
    loop = MagicMock(spec=HostEventLoop)
    loop.enqueue_incoming_event = MagicMock()
    return loop

@pytest.fixture
def mock_adapter_configs():
    return [
        {"id": ADAPTER_ID_1, "url": ADAPTER_URL_1, "auth": {"token": "abc"}},
        {"id": ADAPTER_ID_2, "url": ADAPTER_URL_2}, # No auth
        {"url": "no_id_url"}, # Invalid config
        {"id": "no_url_id"}, # Invalid config
    ]

@pytest.fixture
def mock_async_client_cls():
     # Create a mock class for socketio.AsyncClient
     mock_cls = MagicMock(spec=socketio.AsyncClient)

     # Mock the instance methods needed
     mock_instance = MagicMock(spec=socketio.AsyncClient)
     mock_instance.connect = AsyncMock() # connect is async
     mock_instance.disconnect = AsyncMock()
     mock_instance.emit = AsyncMock()
     mock_instance.event = MagicMock() # Mock the decorator itself
     mock_instance.on = MagicMock() # Mock the decorator itself
     
     # Configure decorators to just return the function they decorate
     # This allows us to test if the handler *functions* are defined
     def event_decorator(func):
         # Store the decorated function for later retrieval/call if needed
         # Use a known attribute name on the mock instance
         if not hasattr(mock_instance, '_event_handlers'): mock_instance._event_handlers = {}
         mock_instance._event_handlers[func.__name__] = func
         return func
     mock_instance.event.side_effect = event_decorator
     
     def on_decorator(event_name):
         def decorator(func):
             # Store the decorated function by event name
             if not hasattr(mock_instance, '_on_handlers'): mock_instance._on_handlers = {}
             mock_instance._on_handlers[event_name] = func
             return func
         return decorator
     mock_instance.on.side_effect = on_decorator

     # Add dicts to store handlers registered via decorators
     # Initialize here to ensure they exist even if no decorators are called in a test path
     mock_instance._event_handlers = {}
     mock_instance._on_handlers = {}

     # Make the mock class return this configured instance
     mock_cls.return_value = mock_instance
     return mock_cls

@pytest.fixture
def activity_client(mock_host_event_loop, mock_adapter_configs):
    # Initialize with configs, event loop will be used later
    client = ActivityClient(host_event_loop=mock_host_event_loop, adapter_api_configs=mock_adapter_configs)
    return client

# --- Test Cases ---

def test_initialization_loads_configs(activity_client, mock_adapter_configs):
    """Test __init__ loads valid configs and ignores invalid ones."""
    assert len(activity_client.adapter_configs) == 2 # Only 2 valid configs
    assert ADAPTER_ID_1 in activity_client.adapter_configs
    assert ADAPTER_ID_2 in activity_client.adapter_configs
    assert activity_client.adapter_configs[ADAPTER_ID_1]["url"] == ADAPTER_URL_1
    assert activity_client.adapter_configs[ADAPTER_ID_2]["url"] == ADAPTER_URL_2
    assert ADAPTER_ID_1 in activity_client.connected_adapters
    assert ADAPTER_ID_2 in activity_client.connected_adapters
    assert activity_client.connected_adapters[ADAPTER_ID_1] is False # Initially disconnected
    assert activity_client.connected_adapters[ADAPTER_ID_2] is False

@pytest.mark.asyncio
@patch('host.modules.activities.activity_client.socketio.AsyncClient') # Patch the class used inside
async def test_connect_to_adapter_api_success(MockAsyncClientCls, activity_client, mock_async_client_cls): # Inject the fixture too
    """Test successful connection attempt and handler registration."""
    # Configure the mock class to return our pre-configured mock instance
    MockAsyncClientCls.return_value = mock_async_client_cls.return_value
    mock_client_instance = MockAsyncClientCls.return_value
    mock_client_instance.connect.return_value = None # Simulate connect completing without error

    config = activity_client.adapter_configs[ADAPTER_ID_1]
    success = await activity_client._connect_to_adapter_api(ADAPTER_ID_1, config)

    assert success is True
    assert ADAPTER_ID_1 in activity_client.clients
    assert activity_client.clients[ADAPTER_ID_1] == mock_client_instance
    # Status becomes True only when 'connect' event handler runs, so still False here
    assert activity_client.connected_adapters[ADAPTER_ID_1] is False

    # Check socketio.AsyncClient constructor was called correctly
    MockAsyncClientCls.assert_called_once_with(
        reconnection=True,
        reconnection_attempts=ANY,
        reconnection_delay=ANY,
        request_timeout=ANY
    )
    # Check connect was called
    mock_client_instance.connect.assert_awaited_once_with(
        ADAPTER_URL_1,
        auth=config.get("auth"),
        namespaces=["/"]
    )
    # Check handlers were registered (by checking the side effect storage)
    assert 'connect' in mock_client_instance._event_handlers
    assert 'disconnect' in mock_client_instance._event_handlers
    assert 'connect_error' in mock_client_instance._event_handlers
    assert 'normalized_event' in mock_client_instance._on_handlers

@pytest.mark.asyncio
@patch('host.modules.activities.activity_client.socketio.AsyncClient')
async def test_connect_to_adapter_api_connection_error(MockAsyncClientCls, activity_client, mock_async_client_cls, caplog):
    """Test handling of socketio.exceptions.ConnectionError during connect."""
    MockAsyncClientCls.return_value = mock_async_client_cls.return_value
    mock_client_instance = MockAsyncClientCls.return_value
    # Simulate connect raising ConnectionError
    mock_client_instance.connect.side_effect = socketio.exceptions.ConnectionError("Connection refused")

    config = activity_client.adapter_configs[ADAPTER_ID_1]
    with caplog.at_level(logging.ERROR):
        success = await activity_client._connect_to_adapter_api(ADAPTER_ID_1, config)

    assert success is False
    assert ADAPTER_ID_1 not in activity_client.clients # Client instance should be removed on definite failure
    assert activity_client.connected_adapters[ADAPTER_ID_1] is False
    assert "Failed to connect" in caplog.text
    assert "Connection refused" in caplog.text

@pytest.mark.asyncio
@patch('host.modules.activities.activity_client.socketio.AsyncClient')
async def test_start_connections_runs_gather(MockAsyncClientCls, activity_client):
     """Test start_connections calls _connect_to_adapter_api for each config."""
     # Need to patch the method *within* the instance for this test
     with patch.object(activity_client, '_connect_to_adapter_api', new_callable=AsyncMock) as mock_connect:
         mock_connect.return_value = True # Assume success for the calls
         await activity_client.start_connections()

         assert mock_connect.call_count == len(activity_client.adapter_configs)
         expected_calls = [
             call(ADAPTER_ID_1, activity_client.adapter_configs[ADAPTER_ID_1]),
             call(ADAPTER_ID_2, activity_client.adapter_configs[ADAPTER_ID_2]),
         ]
         mock_connect.assert_has_calls(expected_calls, any_order=True)

# --- Test Internal Event Handlers ---

@pytest.mark.asyncio
@patch('host.modules.activities.activity_client.socketio.AsyncClient')
async def test_connect_event_handler(MockAsyncClientCls, activity_client, mock_async_client_cls):
    """Test the 'connect' event handler logic."""
    MockAsyncClientCls.return_value = mock_async_client_cls.return_value
    mock_client_instance = MockAsyncClientCls.return_value
    # Need to run connect first to register handlers
    config = activity_client.adapter_configs[ADAPTER_ID_1]
    await activity_client._connect_to_adapter_api(ADAPTER_ID_1, config)
    
    # Get the registered connect handler and call it
    connect_handler = mock_client_instance._event_handlers.get('connect')
    assert connect_handler is not None
    
    assert activity_client.connected_adapters[ADAPTER_ID_1] is False # Before handler
    await connect_handler() # Simulate the event being called
    assert activity_client.connected_adapters[ADAPTER_ID_1] is True # After handler

@pytest.mark.asyncio
@patch('host.modules.activities.activity_client.socketio.AsyncClient')
async def test_disconnect_event_handler(MockAsyncClientCls, activity_client, mock_async_client_cls):
    """Test the 'disconnect' event handler logic."""
    MockAsyncClientCls.return_value = mock_async_client_cls.return_value
    mock_client_instance = MockAsyncClientCls.return_value
    config = activity_client.adapter_configs[ADAPTER_ID_1]
    await activity_client._connect_to_adapter_api(ADAPTER_ID_1, config)
    
    # Simulate being connected first
    activity_client.connected_adapters[ADAPTER_ID_1] = True
    
    disconnect_handler = mock_client_instance._event_handlers.get('disconnect')
    assert disconnect_handler is not None
    await disconnect_handler() # Simulate disconnect event
    assert activity_client.connected_adapters[ADAPTER_ID_1] is False

@pytest.mark.asyncio
@patch('host.modules.activities.activity_client.socketio.AsyncClient')
async def test_normalized_event_handler(MockAsyncClientCls, activity_client, mock_host_event_loop, mock_async_client_cls):
    """Test the 'normalized_event' handler logic."""
    MockAsyncClientCls.return_value = mock_async_client_cls.return_value
    mock_client_instance = MockAsyncClientCls.return_value
    config = activity_client.adapter_configs[ADAPTER_ID_1]
    await activity_client._connect_to_adapter_api(ADAPTER_ID_1, config)

    event_handler = mock_client_instance._on_handlers.get('normalized_event')
    assert event_handler is not None

    # Simulate receiving event data
    received_data = {"event_type": "message", "text": "Data from adapter"}
    await event_handler(received_data.copy()) # Pass copy

    # Check event loop enqueue was called with correct data
    mock_host_event_loop.enqueue_incoming_event.assert_called_once() 
    call_args, _ = mock_host_event_loop.enqueue_incoming_event.call_args
    enqueued_data = call_args[0] # First argument is the event data dictionary
    # Second argument (context) isn't generated by client, check if needed
    # assert len(call_args) == 1 or call_args[1] is None # Or whatever context is expected here
    
    assert enqueued_data["event_type"] == "message"
    assert enqueued_data["text"] == "Data from adapter"
    assert enqueued_data["source_adapter_id"] == ADAPTER_ID_1 # ID should be added

# --- Outgoing Action Handling ---

@pytest.mark.asyncio
@patch('host.modules.activities.activity_client.socketio.AsyncClient')
async def test_handle_outgoing_action_success(MockAsyncClientCls, activity_client, mock_async_client_cls):
     """Test successfully sending an outgoing action."""
     MockAsyncClientCls.return_value = mock_async_client_cls.return_value
     mock_client_instance = MockAsyncClientCls.return_value
     # Simulate connection to ADAPTER_ID_1
     config = activity_client.adapter_configs[ADAPTER_ID_1]
     await activity_client._connect_to_adapter_api(ADAPTER_ID_1, config)
     # Manually set connected state for the test
     activity_client.connected_adapters[ADAPTER_ID_1] = True
     client_instance = activity_client.clients[ADAPTER_ID_1]

     action = {
         "action_type": "send_message", # This determines the formatting
         "payload": {
             "adapter_id": ADAPTER_ID_1, # Target adapter
             "conversation_id": "conv123",
             "text": "Hello from bot"
             # Other potential internal fields to be filtered
         }
     }
     await activity_client.handle_outgoing_action(action)

     # Verify emit call on the correct client instance
     expected_data_to_emit = {
         "event_type": "send_message",
         "data": {
             "conversation_id": "conv123",
             "text": "Hello from bot"
         }
     }
     client_instance.emit.assert_awaited_once_with("bot_response", expected_data_to_emit)

@pytest.mark.asyncio
async def test_handle_outgoing_action_missing_adapter_id(activity_client, caplog):
    """Test outgoing action handling fails if adapter_id is missing in payload."""
    action = {"action_type": "send_message", "payload": {"text": "No adapter"}}
    with caplog.at_level(logging.ERROR):
        await activity_client.handle_outgoing_action(action)
    assert "Missing 'adapter_id' in payload" in caplog.text

@pytest.mark.asyncio
async def test_handle_outgoing_action_adapter_not_connected(activity_client, caplog):
    """Test outgoing action handling fails if target adapter is not connected."""
    # Ensure adapter is marked as not connected
    activity_client.connected_adapters[ADAPTER_ID_1] = False
    # Add a mock client instance just to get past the initial check
    activity_client.clients[ADAPTER_ID_1] = MagicMock()
    
    action = {"action_type": "send_message", "payload": {"adapter_id": ADAPTER_ID_1}}
    with caplog.at_level(logging.ERROR):
        await activity_client.handle_outgoing_action(action)
    assert f"Target adapter API '{ADAPTER_ID_1}' not found or not connected" in caplog.text

# --- Shutdown ---

@pytest.mark.asyncio
@patch('host.modules.activities.activity_client.socketio.AsyncClient')
async def test_shutdown(MockAsyncClientCls, activity_client, mock_async_client_cls):
    """Test shutdown disconnects connected clients."""
    # Simulate connections
    mock_client_1 = MagicMock(spec=socketio.AsyncClient); mock_client_1.disconnect = AsyncMock()
    mock_client_2 = MagicMock(spec=socketio.AsyncClient); mock_client_2.disconnect = AsyncMock()
    activity_client.clients = {ADAPTER_ID_1: mock_client_1, ADAPTER_ID_2: mock_client_2}
    activity_client.connected_adapters = {ADAPTER_ID_1: True, ADAPTER_ID_2: False} # Only 1 is connected

    await activity_client.shutdown()

    # Verify disconnect called only for the connected client
    mock_client_1.disconnect.assert_awaited_once()
    mock_client_2.disconnect.assert_not_called()

    # Verify state cleared
    assert not activity_client.clients
    assert not activity_client.connected_adapters 