import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, call
import logging
from typing import Optional, List, Dict, Any

# Adjust imports based on actual project structure
# Assuming base classes and interfaces are accessible
try:
    from elements.elements.base import BaseElement
    from elements.elements.components.agent_loop import BaseAgentLoopComponent
    from elements.elements.components.simple_loop import SimpleRequestResponseLoopComponent
    from elements.elements.components.hud_component import HUDComponent
    from elements.elements.components.context_manager_component import ContextManagerComponent
    from elements.elements.components.tool_provider_component import ToolProviderComponent
    from llm.provider_interface import LLMResponse, LLMToolCall
    from elements.elements.components.memory.self_query_memory_generator import SelfQueryMemoryGenerationComponent
    from elements.elements.components.memory.curated_memory_generator import CuratedMemoryGenerationComponent
except ImportError as e:
    pytest.skip(f"Skipping tests due to import error: {e}", allow_module_level=True)


# --- Mocks and Fixtures ---

# Mock for the Element (e.g., InnerSpace) the component attaches to
class MockElement(BaseElement):
    def __init__(self, element_id="mock_inner_space"):
        self.id = element_id
        self.name = "MockInnerSpace"
        self._components = {} # Store attached components

    def add_component(self, component: BaseAgentLoopComponent):
        # Simplified add for testing
        self._components[component.COMPONENT_TYPE] = component
        component.element = self # Link back to element

    def get_component(self, component_type: str):
        return self._components.get(component_type)

    # Mock the dependency retrieval mechanism used by _get_dependency
    def get_dependency(self, component_type: str):
         if component_type == HUDComponent.COMPONENT_TYPE:
             return self._mock_hud
         elif component_type == ContextManagerComponent.COMPONENT_TYPE:
             return self._mock_context_manager
         elif component_type == ToolProviderComponent.COMPONENT_TYPE:
             return self._mock_tool_provider
         return None

@pytest.fixture
def mock_element():
    element = MockElement()
    # Attach mock components directly for the element to find via get_dependency
    element._mock_hud = AsyncMock(spec=HUDComponent)
    element._mock_context_manager = MagicMock(spec=ContextManagerComponent)
    element._mock_tool_provider = MagicMock(spec=ToolProviderComponent)
    return element

@pytest.fixture
def mock_hud(mock_element):
    return mock_element._mock_hud

@pytest.fixture
def mock_context_manager(mock_element):
    return mock_element._mock_context_manager

@pytest.fixture
def mock_tool_provider(mock_element):
    return mock_element._mock_tool_provider

@pytest.fixture
def loop_component(mock_element):
    # Instantiate the component
    component = SimpleRequestResponseLoopComponent()
    # Attach it to the mock element
    mock_element.add_component(component)
    
    # Patch the inherited methods we want to observe/control directly in tests
    # Keep the original implementation unless specifically mocked in a test case
    component._execute_tool = MagicMock(wraps=component._execute_tool)
    component._publish_final_message = MagicMock(wraps=component._publish_final_message)
    
    component.logger = logging.getLogger("TestSimpleLoop") # Use a specific logger for capture
    return component

# --- Test Cases ---

def test_initialization(loop_component, mock_element):
    """Test that the component initializes correctly."""
    assert loop_component is not None
    assert loop_component.COMPONENT_TYPE == "agent_loop.simple"
    assert loop_component.element == mock_element

@pytest.mark.asyncio
async def test_run_cycle_final_message(loop_component, mock_hud, mock_context_manager, caplog):
    """Test cycle when LLM returns only content (final message)."""
    final_content = "This is the final response from the LLM."
    mock_llm_response = LLMResponse(content=final_content, tool_calls=None)
    mock_hud.prepare_and_call_llm.return_value = mock_llm_response

    with caplog.at_level(logging.INFO):
        await loop_component.run_cycle()

    # Assertions
    mock_hud.prepare_and_call_llm.assert_awaited_once()
    # Check history addition
    mock_context_manager.add_history_turn.assert_called_once_with(
        role="assistant",
        content={"content": final_content}
    )
    # Check that _publish_final_message was called (indirectly via wrapper)
    loop_component._publish_final_message.assert_called_once_with(mock_context_manager, final_content)
    # Check that _execute_tool was NOT called
    loop_component._execute_tool.assert_not_called()
    assert "Publishing final message..." in caplog.text

@pytest.mark.asyncio
async def test_run_cycle_tool_call(loop_component, mock_hud, mock_context_manager, mock_tool_provider, caplog):
    """Test cycle when LLM returns a tool call."""
    tool_call = LLMToolCall(tool_call_id="call_123", tool_name="test_tool", parameters={"arg": "value"})
    mock_llm_response = LLMResponse(content=None, tool_calls=[tool_call])
    mock_hud.prepare_and_call_llm.return_value = mock_llm_response

    # Mock the actual tool execution part within the inherited _execute_tool
    # We assume _execute_tool finds the tool and calls it; we mock the provider's part
    mock_tool_provider.get_tool_execution_info.return_value = {
        'type': 'direct_call', # or adapter
        'function': MagicMock(return_value="Tool Result String") # Mock the tool function itself
    }

    with caplog.at_level(logging.INFO):
         # Patch _execute_tool for finer control *within this test* if needed,
         # otherwise rely on the wrapped mock from the fixture to check calls
        await loop_component.run_cycle()

    # Assertions
    mock_hud.prepare_and_call_llm.assert_awaited_once()
    # Check history addition
    mock_context_manager.add_history_turn.assert_called_once_with(
        role="assistant",
        content={"tool_calls": [tool_call.to_dict()]}
    )
    # Check _execute_tool was called (indirectly via wrapper)
    loop_component._execute_tool.assert_called_once_with(mock_tool_provider, tool_call.tool_name, tool_call.parameters)
    # Check _publish_final_message was NOT called
    loop_component._publish_final_message.assert_not_called()
    assert "Executing 1 tool call(s)..." in caplog.text
    # Optionally, check if the mocked tool function inside get_tool_execution_info was called
    # mock_tool_provider.get_tool_execution_info.return_value['function'].assert_called_once_with({'arg': 'value'})


@pytest.mark.asyncio
async def test_run_cycle_tool_call_and_content(loop_component, mock_hud, mock_context_manager, mock_tool_provider, caplog):
    """Test cycle when LLM returns both content and a tool call (should execute tool)."""
    final_content = "I can call the tool for you."
    tool_call = LLMToolCall(tool_call_id="call_456", tool_name="another_tool", parameters={})
    mock_llm_response = LLMResponse(content=final_content, tool_calls=[tool_call])
    mock_hud.prepare_and_call_llm.return_value = mock_llm_response

    mock_tool_provider.get_tool_execution_info.return_value = {
        'type': 'direct_call',
        'function': MagicMock(return_value="Another Tool Result")
    }

    with caplog.at_level(logging.INFO):
        await loop_component.run_cycle()

    # Assertions
    mock_hud.prepare_and_call_llm.assert_awaited_once()
    # Check history addition includes both
    mock_context_manager.add_history_turn.assert_called_once_with(
        role="assistant",
        content={'content': final_content, 'tool_calls': [tool_call.to_dict()]}
    )
    # Check tool was executed
    loop_component._execute_tool.assert_called_once_with(mock_tool_provider, tool_call.tool_name, tool_call.parameters)
    # Check final message was NOT published (tool call takes precedence)
    loop_component._publish_final_message.assert_not_called()
    assert "Executing 1 tool call(s)..." in caplog.text

@pytest.mark.asyncio
async def test_run_cycle_no_llm_response(loop_component, mock_hud, caplog):
    """Test cycle when the HUD returns no LLM response."""
    mock_hud.prepare_and_call_llm.return_value = None

    with caplog.at_level(logging.WARNING):
        await loop_component.run_cycle()

    # Assertions
    mock_hud.prepare_and_call_llm.assert_awaited_once()
    loop_component._execute_tool.assert_not_called()
    loop_component._publish_final_message.assert_not_called()
    assert "Cycle aborted: LLM call failed or returned no response." in caplog.text

@pytest.mark.asyncio
async def test_run_cycle_hud_missing(loop_component, mock_element, caplog):
    """Test cycle handles missing HUD dependency gracefully."""
    # Make the mock element return None for HUD
    original_get_dependency = mock_element.get_dependency
    mock_element.get_dependency = MagicMock(side_effect=lambda type: None if type == HUDComponent.COMPONENT_TYPE else original_get_dependency(type))
    
    with caplog.at_level(logging.ERROR):
        await loop_component.run_cycle()

    assert "Cycle aborted: HUDComponent missing." in caplog.text
    # Restore original method if mock_element is used in other tests
    mock_element.get_dependency = original_get_dependency


@pytest.mark.asyncio
async def test_run_cycle_no_content_or_tool_calls(loop_component, mock_hud, mock_context_manager, caplog):
    """Test cycle when LLM returns neither content nor tool calls."""
    mock_llm_response = LLMResponse(content=None, tool_calls=None)
    mock_hud.prepare_and_call_llm.return_value = mock_llm_response

    with caplog.at_level(logging.WARNING):
        await loop_component.run_cycle()

    # Assertions
    mock_hud.prepare_and_call_llm.assert_awaited_once()
    # Check history addition (should still add an empty assistant turn)
    mock_context_manager.add_history_turn.assert_called_once_with(
        role="assistant",
        content={}
    )
    loop_component._execute_tool.assert_not_called()
    loop_component._publish_final_message.assert_not_called()
    assert "LLM response had no tool calls and no content." in caplog.text

# TODO: Add tests for error handling within _execute_tool if BaseAgentLoopComponent's version needs testing.
# TODO: Add tests for error handling within _publish_final_message if BaseAgentLoopComponent's version needs testing.
# TODO: Test interaction with PublisherComponent if _publish_final_message uses it. 

# --- Test Cases for Memory Processing Handler --- 

@pytest.mark.asyncio
async def test_handle_memory_processing_self_query_success(loop_component, mock_element):
    """Test successful memory processing using the default 'self_query' mechanism."""
    ctx_mgr_mock = mock_element.get_component(ContextManagerComponent.COMPONENT_TYPE)
    self_query_gen_mock = mock_element.get_component(SelfQueryMemoryGenerationComponent.COMPONENT_TYPE)
    curated_gen_mock = mock_element.get_component(CuratedMemoryGenerationComponent.COMPONENT_TYPE)
    
    # Simulate one chunk of history being returned
    mock_chunk = [{'role': 'user', 'content': 'hello', 'timestamp': 1000}]
    ctx_mgr_mock.get_unprocessed_history_chunks.return_value = [mock_chunk]
    self_query_gen_mock.generate_memory_for_chunk.return_value = "mem_sq_abc"
    
    action_payload = {
        'tool_call_id': 'test_tool_call_1',
        'tool_args': {
            'process_one_chunk': True,
            # 'generation_mechanism': 'self_query' # Implicit default
        }
    }
    
    await loop_component.handle_action_request("trigger_memory_processing", action_payload)
    
    # Assertions
    ctx_mgr_mock.get_unprocessed_history_chunks.assert_awaited_once_with(limit=1) # Process one chunk = limit 1
    self_query_gen_mock.generate_memory_for_chunk.assert_awaited_once_with(mock_chunk)
    curated_gen_mock.generate_memory_for_chunk.assert_not_awaited() # Curated should not be called
    ctx_mgr_mock.update_processed_marker.assert_called_once_with(1000) # Called with timestamp of last message
    # Check history logging
    ctx_mgr_mock.add_history_turn.assert_called_once_with(
        role="system", 
        content="Memory processing via 'self_query' complete for test_tool_call_1. Generated 1 memories. History processed up to timestamp 1000."
    )

@pytest.mark.asyncio
async def test_handle_memory_processing_curated_success(loop_component, mock_element):
    """Test successful memory processing using the explicit 'curated' mechanism."""
    ctx_mgr_mock = mock_element.get_component(ContextManagerComponent.COMPONENT_TYPE)
    self_query_gen_mock = mock_element.get_component(SelfQueryMemoryGenerationComponent.COMPONENT_TYPE)
    curated_gen_mock = mock_element.get_component(CuratedMemoryGenerationComponent.COMPONENT_TYPE)
    
    mock_chunk = [{'role': 'user', 'content': 'details', 'timestamp': 2000}]
    ctx_mgr_mock.get_unprocessed_history_chunks.return_value = [mock_chunk]
    curated_gen_mock.generate_memory_for_chunk.return_value = "mem_cur_xyz"
    
    action_payload = {
        'tool_call_id': 'test_tool_call_2',
        'tool_args': {
            'process_one_chunk': True,
            'generation_mechanism': 'curated' # Explicitly select curated
        }
    }
    
    await loop_component.handle_action_request("trigger_memory_processing", action_payload)
    
    # Assertions
    ctx_mgr_mock.get_unprocessed_history_chunks.assert_awaited_once_with(limit=1)
    curated_gen_mock.generate_memory_for_chunk.assert_awaited_once_with(mock_chunk)
    self_query_gen_mock.generate_memory_for_chunk.assert_not_awaited()
    ctx_mgr_mock.update_processed_marker.assert_called_once_with(2000)
    ctx_mgr_mock.add_history_turn.assert_called_once_with(
        role="system", 
        content="Memory processing via 'curated' complete for test_tool_call_2. Generated 1 memories. History processed up to timestamp 2000."
    )

@pytest.mark.asyncio
async def test_handle_memory_processing_invalid_mechanism(loop_component, mock_element):
    """Test that an invalid mechanism defaults to self_query."""
    ctx_mgr_mock = mock_element.get_component(ContextManagerComponent.COMPONENT_TYPE)
    self_query_gen_mock = mock_element.get_component(SelfQueryMemoryGenerationComponent.COMPONENT_TYPE)
    
    mock_chunk = [{'role': 'user', 'content': 'test', 'timestamp': 3000}]
    ctx_mgr_mock.get_unprocessed_history_chunks.return_value = [mock_chunk]
    self_query_gen_mock.generate_memory_for_chunk.return_value = "mem_sq_def"
    
    action_payload = {
        'tool_call_id': 'test_tool_call_3',
        'tool_args': {
            'generation_mechanism': 'invalid_option' # Pass an invalid value
        }
    }
    
    await loop_component.handle_action_request("trigger_memory_processing", action_payload)
    
    # Assert self_query (default) was used
    self_query_gen_mock.generate_memory_for_chunk.assert_awaited_once_with(mock_chunk)
    ctx_mgr_mock.update_processed_marker.assert_called_once_with(3000)
    ctx_mgr_mock.add_history_turn.assert_called_once_with(
        role="system", 
        content="Memory processing via 'self_query' complete for test_tool_call_3. Generated 1 memories. History processed up to timestamp 3000."
    )

@pytest.mark.asyncio
async def test_handle_memory_processing_no_chunks(loop_component, mock_element):
    """Test handling when no unprocessed chunks are returned."""
    ctx_mgr_mock = mock_element.get_component(ContextManagerComponent.COMPONENT_TYPE)
    self_query_gen_mock = mock_element.get_component(SelfQueryMemoryGenerationComponent.COMPONENT_TYPE)
    curated_gen_mock = mock_element.get_component(CuratedMemoryGenerationComponent.COMPONENT_TYPE)
    
    ctx_mgr_mock.get_unprocessed_history_chunks.return_value = [] # Simulate no chunks
    
    action_payload = {'tool_call_id': 'test_tool_call_4', 'tool_args': {}}
    
    await loop_component.handle_action_request("trigger_memory_processing", action_payload)
    
    # Assertions
    ctx_mgr_mock.get_unprocessed_history_chunks.assert_awaited_once_with(limit=1) # Still called
    self_query_gen_mock.generate_memory_for_chunk.assert_not_awaited()
    curated_gen_mock.generate_memory_for_chunk.assert_not_awaited()
    ctx_mgr_mock.update_processed_marker.assert_not_called()
    ctx_mgr_mock.add_history_turn.assert_called_once_with(
        role="system", 
        content="Memory processing triggered (test_tool_call_4), but no unprocessed history found."
    )

@pytest.mark.asyncio
async def test_handle_memory_processing_generation_failure(loop_component, mock_element):
    """Test handling when memory generation fails for a chunk."""
    ctx_mgr_mock = mock_element.get_component(ContextManagerComponent.COMPONENT_TYPE)
    self_query_gen_mock = mock_element.get_component(SelfQueryMemoryGenerationComponent.COMPONENT_TYPE)
    
    mock_chunk = [{'role': 'user', 'content': 'fail_test', 'timestamp': 5000}]
    ctx_mgr_mock.get_unprocessed_history_chunks.return_value = [mock_chunk]
    self_query_gen_mock.generate_memory_for_chunk.return_value = None # Simulate failure
    
    action_payload = {'tool_call_id': 'test_tool_call_5', 'tool_args': {}}
    
    await loop_component.handle_action_request("trigger_memory_processing", action_payload)
    
    # Assertions
    ctx_mgr_mock.get_unprocessed_history_chunks.assert_awaited_once_with(limit=1)
    self_query_gen_mock.generate_memory_for_chunk.assert_awaited_once_with(mock_chunk)
    ctx_mgr_mock.update_processed_marker.assert_not_called() # Marker not updated on failure
    ctx_mgr_mock.add_history_turn.assert_called_once_with(
        role="system", 
        content="Memory processing via 'self_query' failed for all 1 chunks attempted for test_tool_call_5."
    )
    
@pytest.mark.asyncio
async def test_handle_memory_processing_missing_generator(loop_component, mock_element):
    """Test handling when the required memory generator component is missing."""
    ctx_mgr_mock = mock_element.get_component(ContextManagerComponent.COMPONENT_TYPE)
    # Simulate the component being missing by making get_dependency return None
    with patch.object(loop_component, '_get_dependency', return_value=None) as mock_get_dep:
        # Ensure ContextManager is still found initially
        mock_get_dep.side_effect = lambda dep_type: ctx_mgr_mock if dep_type == ContextManagerComponent.COMPONENT_TYPE else None
        
        action_payload = {
            'tool_call_id': 'test_tool_call_6',
            'tool_args': {'generation_mechanism': 'curated'}
        }
        
        await loop_component.handle_action_request("trigger_memory_processing", action_payload)
        
        # Assertions
        # Check _get_dependency was called for both ContextManager and the generator
        assert call(ContextManagerComponent.COMPONENT_TYPE) in mock_get_dep.call_args_list
        assert call(CuratedMemoryGenerationComponent.COMPONENT_TYPE) in mock_get_dep.call_args_list
        
        ctx_mgr_mock.get_unprocessed_history_chunks.assert_not_awaited()
        ctx_mgr_mock.update_processed_marker.assert_not_called()
        # Check error logged to history
        ctx_mgr_mock.add_history_turn.assert_called_once_with(
            role="system", 
            content=f"Error handling memory processing for test_tool_call_6: Requested memory generator '{CuratedMemoryGenerationComponent.COMPONENT_TYPE}' not found."
        )
        
# TODO: Add tests for run_cycle if needed (separate from memory handling) 