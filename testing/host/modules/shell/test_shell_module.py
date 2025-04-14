import pytest
import asyncio
from unittest.mock import MagicMock, patch, call, ANY
from typing import Dict, Any, Optional, Type

try:
    # Class to test
    from host.modules.shell.shell_module import ShellModule
    # Dependencies to mock
    from elements.elements.inner_space import InnerSpace
    from llm.provider_interface import LLMProviderInterface
    from host.modules.routing.host_router import HostRouter
    from host.event_loop import HostEventLoop, OutgoingActionCallback
    from elements.elements.components.agent_loop import BaseAgentLoopComponent
    from elements.elements.components.simple_loop import SimpleRequestResponseLoopComponent

except ImportError as e:
    pytest.skip(f"Skipping ShellModule tests due to import error: {e}", allow_module_level=True)

# --- Mocks and Fixtures ---

AGENT_ID = "test_agent_001"
AGENT_NAME = "TestAgent"

@pytest.fixture
def mock_llm_provider():
    return MagicMock(spec=LLMProviderInterface)

@pytest.fixture
def mock_host_router():
    router = MagicMock(spec=HostRouter)
    router.register_agent_route = MagicMock()
    router.unregister_agent_route = MagicMock()
    return router

@pytest.fixture
def mock_host_event_loop():
    loop = MagicMock(spec=HostEventLoop)
    loop.get_outgoing_action_callback = MagicMock(return_value=MagicMock(spec=OutgoingActionCallback))
    return loop

@pytest.fixture
def mock_inner_space_instance():
    space = MagicMock(spec=InnerSpace)
    space.id = f"{AGENT_ID}_inner_space"
    space.set_outgoing_action_callback = MagicMock()
    space.receive_event = MagicMock()
    # Mock the agent loop component retrieval
    mock_loop_comp = MagicMock(spec=BaseAgentLoopComponent)
    mock_loop_comp.run_cycle = AsyncMock() # run_cycle is async
    space.get_agent_loop_component = MagicMock(return_value=mock_loop_comp)
    return space

@pytest.fixture
def mock_host_config():
    # Provide a sample config for testing routing registration/unregistration
    return {
        "agents": [
            {
                "id": AGENT_ID,
                "name": AGENT_NAME,
                "routing_keys": {
                    "conversation_id": ["conv1", "conv2"],
                    "user_id": "user123"
                }
            },
            {
                "id": "other_agent",
                "name": "Other",
                "routing_keys": {"conversation_id": ["conv3"]}
            }
        ]
        # Other host config...
    }

# Patch the InnerSpace class itself for testing constructor calls
@pytest.fixture(autouse=True) # Apply this patch to all tests in the module
def patch_inner_space_class(monkeypatch, mock_inner_space_instance):
    # Make InnerSpace(...) return our mock instance
    mock_inner_space_class = MagicMock(return_value=mock_inner_space_instance)
    monkeypatch.setattr('host.modules.shell.shell_module.InnerSpace', mock_inner_space_class)
    return mock_inner_space_class # Return the mock class for assertion checks

# --- Test Cases ---

def test_shell_initialization_success(
    mock_llm_provider,
    mock_host_router,
    mock_host_event_loop,
    mock_host_config,
    patch_inner_space_class, # Fixture to patch InnerSpace class
    mock_inner_space_instance # Fixture providing the instance returned by patched class
):
    """Test successful initialization of ShellModule."""
    shell = ShellModule(
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        host_config=mock_host_config,
        llm_provider=mock_llm_provider,
        host_router=mock_host_router,
        host_event_loop=mock_host_event_loop
    )

    # Assertions
    assert shell.agent_id == AGENT_ID
    assert shell.agent_name == AGENT_NAME
    assert shell._llm_provider == mock_llm_provider
    assert shell._host_router == mock_host_router
    assert shell._host_event_loop == mock_host_event_loop
    assert shell._inner_space == mock_inner_space_instance

    # Check InnerSpace instantiation
    patch_inner_space_class.assert_called_once_with(
        id=f"{AGENT_ID}_inner_space",
        name=f"{AGENT_NAME}'s Mind",
        llm_provider=mock_llm_provider,
        agent_loop_component_type=SimpleRequestResponseLoopComponent # Default
    )

    # Check outgoing callback injection
    mock_host_event_loop.get_outgoing_action_callback.assert_called_once()
    mock_inner_space_instance.set_outgoing_action_callback.assert_called_once_with(
        mock_host_event_loop.get_outgoing_action_callback.return_value
    )

    # Check routing registration
    expected_route_calls = [
        call(AGENT_ID, "conversation_id", "conv1"),
        call(AGENT_ID, "conversation_id", "conv2"),
        call(AGENT_ID, "user_id", "user123")
    ]
    mock_host_router.register_agent_route.assert_has_calls(expected_route_calls, any_order=True)
    assert mock_host_router.register_agent_route.call_count == 3

def test_shell_initialization_custom_loop_type(
    mock_llm_provider,
    mock_host_router,
    mock_host_event_loop,
    mock_host_config,
    patch_inner_space_class
):
    """Test initialization with a custom AgentLoopComponent type."""
    class CustomLoopComponent(BaseAgentLoopComponent):
        COMPONENT_TYPE = "custom_loop"
        async def run_cycle(self): pass

    shell = ShellModule(
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        host_config=mock_host_config,
        llm_provider=mock_llm_provider,
        host_router=mock_host_router,
        host_event_loop=mock_host_event_loop,
        agent_loop_component_type=CustomLoopComponent # Pass custom type
    )

    # Check InnerSpace instantiation used the custom type
    patch_inner_space_class.assert_called_once_with(
        id=ANY, name=ANY, llm_provider=ANY,
        agent_loop_component_type=CustomLoopComponent
    )

def test_shell_initialization_no_routing_keys(
    mock_llm_provider,
    mock_host_router,
    mock_host_event_loop,
    patch_inner_space_class
):
    """Test initialization when host_config has no routing keys for the agent."""
    empty_config = {"agents": [{"id": AGENT_ID, "name": AGENT_NAME}]} # No routing_keys
    shell = ShellModule(
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        host_config=empty_config,
        llm_provider=mock_llm_provider,
        host_router=mock_host_router,
        host_event_loop=mock_host_event_loop
    )
    # Ensure no routing calls were made
    mock_host_router.register_agent_route.assert_not_called()


def test_shell_initialization_inner_space_fails(
    mock_llm_provider,
    mock_host_router,
    mock_host_event_loop,
    mock_host_config,
    patch_inner_space_class, # Use the patched class fixture
    caplog
):
    """Test initialization handles failure during InnerSpace creation."""
    # Make the patched InnerSpace constructor raise an error
    patch_inner_space_class.side_effect = ValueError("InnerSpace Boom!")

    with caplog.at_level(logging.ERROR):
         shell = ShellModule(
            agent_id=AGENT_ID, agent_name=AGENT_NAME, host_config=mock_host_config,
            llm_provider=mock_llm_provider, host_router=mock_host_router,
            host_event_loop=mock_host_event_loop
        )

    # Assertions
    assert shell._inner_space is None # Should be None after failure
    assert "Critical error during agent initialization: InnerSpace Boom!" in caplog.text
    # Check that subsequent steps like routing weren't attempted
    mock_host_router.register_agent_route.assert_not_called()
    # Reset side effect if the mock class is reused
    patch_inner_space_class.side_effect = None

@pytest.mark.asyncio
async def test_handle_incoming_event_success(
    shell_module_initialized, # Use a fixture that provides an initialized shell
    mock_inner_space_instance
):
    """Test successful delegation of incoming event to InnerSpace."""
    event = {"event_type": "test_event", "data": {"value": 1}}
    context = {"timeline_id": "t1"}

    await shell_module_initialized.handle_incoming_event(event, context)

    mock_inner_space_instance.receive_event.assert_called_once_with(event, context)

@pytest.mark.asyncio
async def test_handle_incoming_event_no_inner_space(
    shell_module_initialized, # Fixture providing initialized shell
    caplog
):
    """Test event handling logs error if InnerSpace failed init."""
    # Simulate InnerSpace being None
    shell_module_initialized._inner_space = None

    with caplog.at_level(logging.ERROR):
        await shell_module_initialized.handle_incoming_event({}, {})

    assert "Cannot handle event: InnerSpace not initialized" in caplog.text

@pytest.mark.asyncio
async def test_trigger_agent_cycle_success(
     shell_module_initialized,
     mock_inner_space_instance
):
    """Test successfully triggering the agent cycle."""
    # Get the mock loop component from the mock inner space
    mock_loop_comp = mock_inner_space_instance.get_agent_loop_component.return_value

    await shell_module_initialized.trigger_agent_cycle()

    mock_inner_space_instance.get_agent_loop_component.assert_called_once()
    mock_loop_comp.run_cycle.assert_awaited_once()

@pytest.mark.asyncio
async def test_trigger_agent_cycle_no_inner_space(
    shell_module_initialized,
    caplog
):
    """Test triggering cycle logs error if InnerSpace failed init."""
    shell_module_initialized._inner_space = None
    with caplog.at_level(logging.ERROR):
        await shell_module_initialized.trigger_agent_cycle()
    assert "Cannot trigger cycle: InnerSpace not initialized" in caplog.text

@pytest.mark.asyncio
async def test_trigger_agent_cycle_no_loop_component(
    shell_module_initialized,
    mock_inner_space_instance,
    caplog
):
    """Test triggering cycle logs error if InnerSpace has no loop component."""
    # Make get_agent_loop_component return None
    mock_inner_space_instance.get_agent_loop_component.return_value = None

    with caplog.at_level(logging.ERROR):
        await shell_module_initialized.trigger_agent_cycle()

    mock_inner_space_instance.get_agent_loop_component.assert_called_once()
    assert "AgentLoopComponent not found on InnerSpace" in caplog.text

@pytest.mark.asyncio
async def test_shutdown_success(
    shell_module_initialized,
    mock_host_router,
    mock_host_config # Need config again to know which routes to unregister
):
    """Test shutdown unregisters routes."""
    await shell_module_initialized.shutdown()

    # Verify unregister calls based on mock_host_config
    expected_unregister_calls = [
        call("conversation_id", "conv1"),
        call("conversation_id", "conv2"),
        call("user_id", "user123")
    ]
    mock_host_router.unregister_agent_route.assert_has_calls(expected_unregister_calls, any_order=True)
    assert mock_host_router.unregister_agent_route.call_count == 3

@pytest.mark.asyncio
async def test_shutdown_no_inner_space(
    shell_module_initialized,
    mock_host_router
):
    """Test shutdown does nothing if InnerSpace wasn't initialized."""
    shell_module_initialized._inner_space = None
    await shell_module_initialized.shutdown()
    mock_host_router.unregister_agent_route.assert_not_called()

# --- Helper Fixture for Initialized Shell ---
# Avoids repeating init logic in every test function

@pytest.fixture
def shell_module_initialized(
    mock_llm_provider,
    mock_host_router,
    mock_host_event_loop,
    mock_host_config,
    patch_inner_space_class, # Ensures InnerSpace is patched
    mock_inner_space_instance # Ensures we have the instance
):
    """Provides a successfully initialized ShellModule instance."""
    # Reset mocks potentially modified by error tests
    patch_inner_space_class.side_effect = None
    patch_inner_space_class.return_value = mock_inner_space_instance
    
    shell = ShellModule(
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        host_config=mock_host_config,
        llm_provider=mock_llm_provider,
        host_router=mock_host_router,
        host_event_loop=mock_host_event_loop
    )
    # Double-check init didn't fail silently in fixture setup
    assert shell._inner_space is not None
    return shell 