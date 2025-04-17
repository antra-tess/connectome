import pytest
from unittest.mock import MagicMock, patch, call, AsyncMock
import logging

# Component under test
from elements.elements.components.two_step_loop import TwoStepLoopComponent

# Dependencies and related classes
from elements.elements.components.base_agent_loop import BaseAgentLoopComponent # Base class
from elements.elements.components.hud_component import HUDComponent
from elements.elements.components.context_manager_component import ContextManagerComponent
from elements.elements.components.tool_provider_component import ToolProviderComponent
from elements.elements.base_element import BaseElement
from elements.llm.provider import LLMProvider # Interface for mocking
from elements.llm.response import LLMResponse, LLMToolCall # For mock responses

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_llm_provider():
    """Fixture for a mocked LLMProvider."""
    mock = MagicMock(spec=LLMProvider)
    # Use AsyncMock for the async 'complete' method
    mock.complete = AsyncMock()
    return mock

@pytest.fixture
def mock_hud_component(mock_llm_provider):
    """Fixture for a mocked HUDComponent."""
    mock = MagicMock(spec=HUDComponent)
    mock.COMPONENT_TYPE = "hud"
    mock._llm_provider = mock_llm_provider # Inject mocked provider
    mock._state = { # Mock state needed by the loop
        "model": "test-model",
        "temperature": 0.5,
        "max_tokens": 1000
    }
    # Mock the message preparation helper
    mock._prepare_llm_messages_from_string = MagicMock(return_value=[{"role": "user", "content": "Prepared Context"}])
    return mock

@pytest.fixture
def mock_context_manager():
    """Fixture for a mocked ContextManagerComponent."""
    mock = MagicMock(spec=ContextManagerComponent)
    mock.COMPONENT_TYPE = "context_manager"
    mock.build_context = MagicMock(return_value="Current context string.")
    mock.add_history_turn = MagicMock()
    return mock

@pytest.fixture
def mock_tool_provider():
    """Fixture for a mocked ToolProviderComponent."""
    mock = MagicMock(spec=ToolProviderComponent)
    mock.COMPONENT_TYPE = "tool_provider"
    # Mock schema retrieval for the action step
    mock.get_llm_tool_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "test_tool", "description": "A test tool", "parameters": {}}}
    ])
    return mock

@pytest.fixture
def mock_element(mock_hud_component, mock_context_manager, mock_tool_provider):
    """Fixture for a mocked BaseElement with necessary components."""
    element = MagicMock(spec=BaseElement)
    element.id = "two_step_agent_1"

    # Mock get_component_by_type to return the correct mock based on type request
    def get_component_side_effect(comp_type: str):
        if comp_type == "hud":
            return mock_hud_component
        elif comp_type == "context_manager":
            return mock_context_manager
        elif comp_type == "tool_provider":
            return mock_tool_provider
        else:
            # Explicitly return None if an unexpected component is requested
            # Find specific components by type
            for comp in [mock_hud_component, mock_context_manager, mock_tool_provider]:
                 if hasattr(comp, 'COMPONENT_TYPE') and comp.COMPONENT_TYPE == comp_type:
                      return comp
            return None # Default for others
    element.get_component_by_type = MagicMock(side_effect=get_component_side_effect)

    # Patch get_component to do the same for simpler lookups if used by _get_dependency
    element.get_component = MagicMock(side_effect=lambda comp_cls: get_component_side_effect(comp_cls.COMPONENT_TYPE))

    return element

@pytest.fixture
def two_step_loop(mock_element):
    """Fixture for the TwoStepLoopComponent instance."""
    # Mock methods assumed to be implemented/inherited (e.g., from BaseAgentLoop)
    # These handle the *output* of the action step
    with patch.object(TwoStepLoopComponent, '_execute_tool', return_value=None) as mock_exec, \
         patch.object(TwoStepLoopComponent, '_publish_final_message', return_value=None) as mock_pub:

        comp = TwoStepLoopComponent(element=mock_element)
        comp._initialize()
        comp._enable()
        # Store mocks for assertion access in tests
        comp._mock_execute_tool = mock_exec
        comp._mock_publish_final_message = mock_pub
        yield comp # Yield to allow tests to run

# --- Helper Function ---
def create_llm_response(content: Optional[str] = None, tool_calls: Optional[List[LLMToolCall]] = None) -> LLMResponse:
    """Creates a mock LLMResponse object."""
    response = MagicMock(spec=LLMResponse)
    response.content = content
    response.tool_calls = tool_calls
    # Mock to_dict if needed for history logging
    if tool_calls:
         for tc in tool_calls:
              tc.to_dict = MagicMock(return_value={"id": tc.id, "type": "function", "function": {"name": tc.tool_name, "arguments": tc.parameters}})
    return response

# --- Test Cases ---

def test_two_step_loop_initialization(two_step_loop, mock_element):
    """Test basic initialization."""
    assert two_step_loop.element == mock_element
    assert two_step_loop.COMPONENT_TYPE == "agent_loop.two_step"
    assert two_step_loop._is_initialized
    assert two_step_loop._is_enabled
    assert "_contemplation_prompt_suffix" in two_step_loop.__dict__
    assert "_action_prompt_suffix" in two_step_loop.__dict__

@pytest.mark.asyncio
async def test_run_cycle_missing_dependency(two_step_loop, mock_element):
    """Test cycle abortion if a dependency is missing."""
    # Make ContextManager missing
    mock_element.get_component_by_type.side_effect = lambda ct: None if ct == "context_manager" else MagicMock()
    mock_element.get_component.side_effect = lambda c: None if c == ContextManagerComponent else MagicMock()


    with patch('logging.Logger.error') as mock_log_error:
        await two_step_loop.run_cycle()

    mock_log_error.assert_called_with(
        f"[{two_step_loop.element.id}] Cycle aborted: Missing dependencies (HUD, ContextManager, or ToolProvider)."
    )

@pytest.mark.asyncio
async def test_run_cycle_missing_llm_provider(two_step_loop, mock_hud_component):
    """Test cycle abortion if HUD has no LLM provider."""
    mock_hud_component._llm_provider = None # Simulate missing provider

    with patch('logging.Logger.error') as mock_log_error:
        await two_step_loop.run_cycle()

    mock_log_error.assert_called_with(
        f"[{two_step_loop.element.id}] Cycle aborted: HUDComponent has no LLMProvider."
    )

@pytest.mark.asyncio
async def test_run_cycle_contemplation_then_message(two_step_loop, mock_llm_provider, mock_context_manager):
    """Test a cycle resulting in contemplation then a final message."""
    contemplation_text = "Thinking about the user query."
    final_message = "This is the final answer."

    # Configure LLM mock responses
    mock_llm_provider.complete.side_effect = [
        # First call (contemplation)
        create_llm_response(content=contemplation_text),
        # Second call (action)
        create_llm_response(content=final_message)
    ]

    await two_step_loop.run_cycle()

    # --- Assertions ---
    # 1. LLM called twice
    assert mock_llm_provider.complete.call_count == 2
    contemplation_call_args, action_call_args = mock_llm_provider.complete.call_args_list

    # 2. Contemplation call checks
    cont_args, cont_kwargs = contemplation_call_args
    assert cont_kwargs['tools'] is None # No external tools for contemplation
    assert two_step_loop._contemplation_prompt_suffix in cont_kwargs['messages'][0]['content']

    # 3. History updated after contemplation
    mock_context_manager.add_history_turn.assert_any_call(role="internal_monologue", content=contemplation_text)

    # 4. Action call checks (context manager was called again before action)
    assert mock_context_manager.build_context.call_count == 2
    act_args, act_kwargs = action_call_args
    assert act_kwargs['tools'] is not None # Tools provided for action
    assert two_step_loop._action_prompt_suffix in act_kwargs['messages'][0]['content']

    # 5. History updated after action (assistant message)
    mock_context_manager.add_history_turn.assert_any_call(role="assistant", content={"content": final_message})

    # 6. Final message published
    two_step_loop._mock_publish_final_message.assert_called_once_with(mock_context_manager, final_message)
    two_step_loop._mock_execute_tool.assert_not_called()


@pytest.mark.asyncio
async def test_run_cycle_contemplation_then_tool_call(two_step_loop, mock_llm_provider, mock_context_manager, mock_tool_provider):
    """Test a cycle resulting in contemplation then a tool call."""
    contemplation_text = "Plan: Need to use the test tool."
    tool_name = "test_tool"
    tool_params = {"arg1": "value1"}
    tool_call_id = "call_abc123"

    # Mock LLMToolCall
    mock_tool_call = MagicMock(spec=LLMToolCall)
    mock_tool_call.id = tool_call_id
    mock_tool_call.tool_name = tool_name
    mock_tool_call.parameters = tool_params

    # Configure LLM mock responses
    mock_llm_provider.complete.side_effect = [
        # First call (contemplation)
        create_llm_response(content=contemplation_text),
        # Second call (action) - returns tool call, no direct content
        create_llm_response(content=None, tool_calls=[mock_tool_call])
    ]

    await two_step_loop.run_cycle()

    # --- Assertions ---
    # 1. LLM called twice
    assert mock_llm_provider.complete.call_count == 2

    # 2. History updated after contemplation
    mock_context_manager.add_history_turn.assert_any_call(role="internal_monologue", content=contemplation_text)

    # 3. History updated after action (assistant tool call)
    expected_tool_call_dict = {"id": tool_call_id, "type": "function", "function": {"name": tool_name, "arguments": tool_params}}
    mock_context_manager.add_history_turn.assert_any_call(role="assistant", content={"tool_calls": [expected_tool_call_dict]})

    # 4. Tool executed, message not published
    two_step_loop._mock_execute_tool.assert_called_once_with(mock_tool_provider, tool_name, tool_params)
    two_step_loop._mock_publish_final_message.assert_not_called()

@pytest.mark.asyncio
async def test_run_cycle_contemplation_fails(two_step_loop, mock_llm_provider, mock_context_manager):
    """Test cycle when the contemplation step fails."""
    # Configure LLM mock to raise error on first call
    mock_llm_provider.complete.side_effect = [
        ValueError("LLM Error during contemplation"), # Fails first call
        create_llm_response(content="Action despite contemplation failure") # Second call (should still happen)
    ]

    with patch('logging.Logger.error') as mock_log_error:
        await two_step_loop.run_cycle()

    # Assert contemplation error logged
    mock_log_error.assert_any_call(
        f"[{two_step_loop.element.id}] Error during Contemplation LLM call: LLM Error during contemplation", exc_info=True
    )
    # Default thought added to history
    mock_context_manager.add_history_turn.assert_any_call(role="internal_monologue", content="(No internal thought generated)")
    # Action step still ran
    assert mock_llm_provider.complete.call_count == 2
    # Final message from action step should have been published
    two_step_loop._mock_publish_final_message.assert_called_once_with(mock_context_manager, "Action despite contemplation failure")
    two_step_loop._mock_execute_tool.assert_not_called()

@pytest.mark.asyncio
async def test_run_cycle_action_fails(two_step_loop, mock_llm_provider, mock_context_manager):
    """Test cycle when the action step fails."""
    contemplation_text = "Contemplation succeeded."
     # Configure LLM mock responses
    mock_llm_provider.complete.side_effect = [
        create_llm_response(content=contemplation_text), # Contemplation OK
        ValueError("LLM Error during action") # Action fails
    ]

    with patch('logging.Logger.error') as mock_log_error, \
         patch('logging.Logger.warning') as mock_log_warning:
        await two_step_loop.run_cycle()

    # Assert action error logged
    mock_log_error.assert_any_call(
        f"[{two_step_loop.element.id}] Error during Action LLM call: LLM Error during action", exc_info=True
    )
    # Warning logged about action step failure
    mock_log_warning.assert_any_call(f"[{two_step_loop.element.id}] Action step failed or returned no response.")
    # History updated with contemplation and system message for action failure
    mock_context_manager.add_history_turn.assert_any_call(role="internal_monologue", content=contemplation_text)
    mock_context_manager.add_history_turn.assert_any_call(role="system", content="(Action generation step failed)")

    # No tool or message output
    two_step_loop._mock_publish_final_message.assert_not_called()
    two_step_loop._mock_execute_tool.assert_not_called()

@pytest.mark.asyncio
async def test_run_cycle_action_no_output(two_step_loop, mock_llm_provider, mock_context_manager):
    """Test cycle when action step returns no content or tool calls."""
    contemplation_text = "Contemplation ok."
    # Configure LLM mock responses
    mock_llm_provider.complete.side_effect = [
        create_llm_response(content=contemplation_text), # Contemplation OK
        create_llm_response(content=None, tool_calls=None) # Action returns nothing
    ]
    with patch('logging.Logger.warning') as mock_log_warning:
        await two_step_loop.run_cycle()

    # Warning logged about no action output
    mock_log_warning.assert_any_call(f"[{two_step_loop.element.id}] Action LLM response had no tool calls and no content.")
    # History shows contemplation and system message for no action output
    mock_context_manager.add_history_turn.assert_any_call(role="internal_monologue", content=contemplation_text)
    mock_context_manager.add_history_turn.assert_any_call(role="system", content="(Action step produced no actionable output)")
    # No tool or message output
    two_step_loop._mock_publish_final_message.assert_not_called()
    two_step_loop._mock_execute_tool.assert_not_called() 