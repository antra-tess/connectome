import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch, call, ANY
from typing import Dict, Any, Optional, Tuple

try:
    # Class to test
    from host.event_loop import HostEventLoop, OutgoingActionCallback, AGENT_CYCLE_DEBOUNCE_SECONDS, TRIGGERING_EVENT_TYPES
    # Dependencies
    from host.modules.routing.host_router import HostRouter
    from host.modules.shell.shell_module import ShellModule
    from host.modules.activities.activity_listener import ActivityListener
    from host.modules.activities.activity_client import ActivityClient

except ImportError as e:
    pytest.skip(f"Skipping HostEventLoop tests due to import error: {e}", allow_module_level=True)

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_host_router():
    router = MagicMock(spec=HostRouter)
    router.get_target_agent_id = MagicMock(return_value=None) # Default: unroutable
    return router

@pytest.fixture
def mock_shell_module():
    shell = MagicMock(spec=ShellModule)
    shell.agent_id = "agent_test_1"
    shell.handle_incoming_event = AsyncMock()
    shell.trigger_agent_cycle = AsyncMock()
    return shell

@pytest.fixture
def mock_shell_modules(mock_shell_module):
    # Fixture providing a dictionary of shell modules
    return {mock_shell_module.agent_id: mock_shell_module}

@pytest.fixture
def mock_activity_listener():
    return MagicMock(spec=ActivityListener)

@pytest.fixture
def mock_activity_client():
    client = MagicMock(spec=ActivityClient)
    client.send_event_to_adapter = MagicMock(return_value=True) # Default: success
    return client

@pytest.fixture
def event_loop(mock_host_router, mock_shell_modules, mock_activity_listener, mock_activity_client):
    # Use the dictionary fixture here
    loop = HostEventLoop(
        host_router=mock_host_router,
        shell_modules=mock_shell_modules,
        activity_listener=mock_activity_listener,
        activity_client=mock_activity_client
    )
    # Clear queues just in case
    while not loop._incoming_event_queue.empty(): loop._incoming_event_queue.get_nowait()
    while not loop._outgoing_action_queue.empty(): loop._outgoing_action_queue.get_nowait()
    return loop

# --- Test Cases ---

def test_initialization(event_loop, mock_host_router, mock_shell_modules, mock_activity_listener, mock_activity_client):
    """Test constructor stores dependencies."""
    assert event_loop.host_router == mock_host_router
    assert event_loop.shell_modules == mock_shell_modules
    assert event_loop.activity_listener == mock_activity_listener
    assert event_loop.activity_client == mock_activity_client
    assert event_loop.running is False
    assert event_loop._incoming_event_queue.empty()
    assert event_loop._outgoing_action_queue.empty()

def test_get_outgoing_action_callback(event_loop):
    """Test that the correct enqueue method is returned as callback."""
    callback = event_loop.get_outgoing_action_callback()
    assert callback == event_loop.enqueue_outgoing_action

@pytest.mark.asyncio
async def test_enqueue_incoming_event(event_loop):
    """Test enqueuing an incoming event."""
    event = {"event_type": "test"}
    context = {"timeline_id": "t1"}
    assert event_loop._incoming_event_queue.empty()
    event_loop.enqueue_incoming_event(event, context)
    assert event_loop._incoming_event_queue.qsize() == 1
    dequeued_event, dequeued_context = await event_loop._incoming_event_queue.get()
    assert dequeued_event == event
    assert dequeued_context == context

@pytest.mark.asyncio
async def test_enqueue_outgoing_action(event_loop):
    """Test enqueuing an outgoing action."""
    action = {"action_type": "send"}
    assert event_loop._outgoing_action_queue.empty()
    event_loop.enqueue_outgoing_action(action)
    assert event_loop._outgoing_action_queue.qsize() == 1
    dequeued_action = await event_loop._outgoing_action_queue.get()
    assert dequeued_action == action

# --- Incoming Queue Processing ---

@pytest.mark.asyncio
async def test_process_incoming_routable_triggering(event_loop, mock_host_router, mock_shell_module):
    """Test processing a routable event that triggers a cycle."""
    agent_id = mock_shell_module.agent_id
    trigger_event_type = list(TRIGGERING_EVENT_TYPES)[0] # Get a valid trigger type
    event = {"event_type": trigger_event_type, "event_id": "ev1"}
    context = {"timeline_id": "t_route"}
    mock_host_router.get_target_agent_id.return_value = agent_id

    event_loop.enqueue_incoming_event(event, context)
    await event_loop._process_incoming_event_queue()

    mock_host_router.get_target_agent_id.assert_called_once_with(context)
    mock_shell_module.handle_incoming_event.assert_awaited_once_with(event, context)
    assert agent_id in event_loop._pending_agent_cycles
    assert agent_id in event_loop._trigger_event_received_time

@pytest.mark.asyncio
async def test_process_incoming_routable_non_triggering(event_loop, mock_host_router, mock_shell_module):
    """Test processing a routable event that does NOT trigger a cycle."""
    agent_id = mock_shell_module.agent_id
    event = {"event_type": "non_trigger_event", "event_id": "ev2"}
    context = {"timeline_id": "t_route2"}
    mock_host_router.get_target_agent_id.return_value = agent_id

    event_loop.enqueue_incoming_event(event, context)
    await event_loop._process_incoming_event_queue()

    mock_host_router.get_target_agent_id.assert_called_once_with(context)
    mock_shell_module.handle_incoming_event.assert_awaited_once_with(event, context)
    assert agent_id not in event_loop._pending_agent_cycles # Should not be added
    assert agent_id not in event_loop._trigger_event_received_time

@pytest.mark.asyncio
async def test_process_incoming_unroutable(event_loop, mock_host_router, mock_shell_module, caplog):
    """Test processing an unroutable event."""
    event = {"event_type": "some_event", "event_id": "ev3"}
    context = {"timeline_id": "t_unroute"}
    mock_host_router.get_target_agent_id.return_value = None # Simulate unroutable

    event_loop.enqueue_incoming_event(event, context)
    with caplog.at_level(logging.WARNING):
        await event_loop._process_incoming_event_queue()

    mock_host_router.get_target_agent_id.assert_called_once_with(context)
    mock_shell_module.handle_incoming_event.assert_not_called() # Shell handler not called
    assert "Could not route event ev3" in caplog.text

@pytest.mark.asyncio
async def test_process_incoming_unknown_agent(event_loop, mock_host_router, caplog):
    """Test processing event for an agent ID not in shell_modules."""
    agent_id = "unknown_agent"
    event = {"event_type": "trigger_event", "event_id": "ev4"}
    context = {"timeline_id": "t_unknown"}
    mock_host_router.get_target_agent_id.return_value = agent_id

    event_loop.enqueue_incoming_event(event, context)
    with caplog.at_level(logging.ERROR):
        await event_loop._process_incoming_event_queue()

    mock_host_router.get_target_agent_id.assert_called_once_with(context)
    assert f"Target agent '{agent_id}' not found" in caplog.text

# --- Outgoing Queue Processing ---

@pytest.mark.asyncio
async def test_process_outgoing_valid_action(event_loop, mock_activity_client):
    """Test processing a valid outgoing action for activity client."""
    payload = {"event_type": "send_message", "data": "hello"}
    action = {
        "target_module": "activity_client",
        "action_type": "send_external_event",
        "payload": payload
    }
    event_loop.enqueue_outgoing_action(action)
    await event_loop._process_outgoing_action_queue()

    mock_activity_client.send_event_to_adapter.assert_called_once_with(payload)

@pytest.mark.asyncio
async def test_process_outgoing_unhandled_action(event_loop, mock_activity_client, caplog):
    """Test processing an unhandled outgoing action."""
    action = {"target_module": "unknown_module", "action_type": "do_stuff"}
    event_loop.enqueue_outgoing_action(action)
    with caplog.at_level(logging.WARNING):
        await event_loop._process_outgoing_action_queue()

    mock_activity_client.send_event_to_adapter.assert_not_called()
    assert "No handler found for outgoing action" in caplog.text

# --- Cycle Triggering Logic ---

def test_should_agent_run_cycle_debounce_passed(event_loop):
    """Test _should_agent_run_cycle returns True when debounce passes."""
    agent_id = "agent_deb"
    trigger_time = time.monotonic() - (AGENT_CYCLE_DEBOUNCE_SECONDS + 0.1)
    event_loop._trigger_event_received_time[agent_id] = trigger_time

    should_run = event_loop._should_agent_run_cycle(agent_id, time.monotonic())
    assert should_run is True

def test_should_agent_run_cycle_debounce_not_passed(event_loop):
    """Test _should_agent_run_cycle returns False when within debounce period."""
    agent_id = "agent_deb"
    trigger_time = time.monotonic() - (AGENT_CYCLE_DEBOUNCE_SECONDS - 0.1)
    event_loop._trigger_event_received_time[agent_id] = trigger_time

    should_run = event_loop._should_agent_run_cycle(agent_id, time.monotonic())
    assert should_run is False

def test_should_agent_run_cycle_no_trigger_time(event_loop):
     """Test _should_agent_run_cycle handles missing trigger time (shouldn't happen if logic correct)."""
     agent_id = "agent_no_time"
     # Ensure no trigger time is set
     event_loop._trigger_event_received_time.pop(agent_id, None)
     should_run = event_loop._should_agent_run_cycle(agent_id, time.monotonic())
     assert should_run is False # Cannot run without trigger time

# --- Run Loop Simulation (Simplified) ---

@pytest.mark.asyncio
@patch('asyncio.sleep', return_value=None) # Prevent actual sleeping
async def test_run_loop_iteration_triggers_cycle(
    mock_sleep, # Patched asyncio.sleep
    event_loop,
    mock_host_router,
    mock_shell_module
):
    """Simulate one loop iteration where an event triggers a cycle after debounce."""
    agent_id = mock_shell_module.agent_id
    trigger_event_type = list(TRIGGERING_EVENT_TYPES)[0]
    event = {"event_type": trigger_event_type, "event_id": "ev_trigger"}
    context = {"timeline_id": "t_cycle"}
    mock_host_router.get_target_agent_id.return_value = agent_id

    # 1. Enqueue event
    event_loop.enqueue_incoming_event(event, context)

    # 2. Run _process_incoming (simulates first part of loop) -> adds agent to pending
    await event_loop._process_incoming_event_queue()
    assert agent_id in event_loop._pending_agent_cycles

    # 3. Patch time.monotonic to simulate time passing beyond debounce
    with patch('time.monotonic', return_value=time.monotonic() + AGENT_CYCLE_DEBOUNCE_SECONDS + 0.1):
        # 4. Manually call the part of the run loop that checks/triggers cycles
        now = time.monotonic()
        agents_to_run_now = set()
        for loop_agent_id in list(event_loop._pending_agent_cycles): # Iterate copy for safe removal
            if event_loop._should_agent_run_cycle(loop_agent_id, now):
                 agents_to_run_now.add(loop_agent_id)
                 event_loop._pending_agent_cycles.remove(loop_agent_id)

        assert agent_id in agents_to_run_now # Should be marked to run
        
        # Simulate the trigger call
        for run_agent_id in agents_to_run_now:
             target_shell = event_loop.shell_modules.get(run_agent_id)
             if target_shell:
                 await target_shell.trigger_agent_cycle()

    # 5. Verify shell cycle was triggered
    mock_shell_module.trigger_agent_cycle.assert_awaited_once()
    assert agent_id not in event_loop._pending_agent_cycles # Should be removed after check

# --- Stop Method ---

def test_stop_method(event_loop):
    """Test the stop method sets running flag."""
    event_loop.running = True
    event_loop.stop()
    assert event_loop.running is False 