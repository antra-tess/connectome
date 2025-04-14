import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, call
from typing import Optional, List, Dict, Any

try:
    # Class to test
    from elements.elements.components.hud_component import HUDComponent
    # Dependencies and base classes
    from elements.elements.base import BaseElement, Component
    from elements.elements.components.context_manager_component import ContextManagerComponent
    from elements.elements.components.tool_provider_component import ToolProviderComponent
    from llm.provider_interface import (
        LLMProviderInterface,
        LLMMessage,
        LLMToolDefinition,
        LLMToolCall,
        LLMResponse
    )
except ImportError as e:
    pytest.skip(f"Skipping HUDComponent tests due to import error: {e}", allow_module_level=True)

# --- Mocks and Fixtures ---

# Mock Element to host the HUDComponent
class MockHostElement(BaseElement):
    def __init__(self, element_id="host_for_hud"):
        self.id = element_id
        self.name = "MockHostForHUD"
        self._components = {}
        # Add mocks for dependencies HUDComponent will look for
        self._mock_context_manager = MagicMock(spec=ContextManagerComponent)
        self._mock_tool_provider = MagicMock(spec=ToolProviderComponent)
        self._components[ContextManagerComponent.COMPONENT_TYPE] = self._mock_context_manager
        self._components[ToolProviderComponent.COMPONENT_TYPE] = self._mock_tool_provider
        
    def add_component(self, component: Component):
        self._components[component.COMPONENT_TYPE] = component
        component.element = self

    def get_component(self, component_type_or_class) -> Optional[Component]:
        component_type = component_type_or_class
        if isinstance(component_type_or_class, type):
             component_type = getattr(component_type_or_class, 'COMPONENT_TYPE', None)
             if not component_type: # Fallback for base classes?
                 for comp in self._components.values():
                     if isinstance(comp, component_type_or_class): return comp
                 return None
        return self._components.get(component_type)

@pytest.fixture
def mock_host_element():
    return MockHostElement()

@pytest.fixture
def mock_llm_provider():
    # Use AsyncMock for the 'complete' method
    provider = MagicMock(spec=LLMProviderInterface)
    provider.complete = AsyncMock() 
    return provider

@pytest.fixture
def mock_context_manager(mock_host_element):
    return mock_host_element._mock_context_manager

@pytest.fixture
def mock_tool_provider(mock_host_element):
    return mock_host_element._mock_tool_provider

@pytest.fixture
def hud_component(mock_host_element, mock_llm_provider):
    # Inject the mock LLM provider during init
    component = HUDComponent(element=mock_host_element, llm_provider=mock_llm_provider)
    # Manually initialize/enable for testing
    component._initialize()
    component._is_enabled = True
    return component

# --- Test Cases ---

def test_initialization_success(hud_component, mock_host_element, mock_llm_provider):
    """Test successful initialization with an injected LLM provider."""
    assert hud_component.element == mock_host_element
    assert hud_component._llm_provider == mock_llm_provider
    assert hud_component._state["system_prompt"] == "You are a helpful AI assistant."
    assert hud_component._state["history_representation"] == "user_message"

def test_initialization_no_provider(mock_host_element, caplog):
    """Test initialization logs error if LLM provider is missing."""
    with caplog.at_level(logging.ERROR):
        component = HUDComponent(element=mock_host_element, llm_provider=None)
    assert component._llm_provider is None
    assert "HUDComponent requires an LLMProvider instance" in caplog.text

def test_prepare_tool_definitions_success(hud_component, mock_tool_provider):
    """Test preparing tool definitions when ToolProvider exists."""
    mock_schema = [{"type": "function", "function": {"name": "test_tool"}}]
    mock_tool_provider.get_llm_tool_schemas.return_value = mock_schema
    
    definitions = hud_component._prepare_tool_definitions()
    
    assert definitions == mock_schema
    mock_tool_provider.get_llm_tool_schemas.assert_called_once()

def test_prepare_tool_definitions_no_provider(hud_component, mock_host_element):
    """Test preparing tool definitions when ToolProvider is missing."""
    # Remove the mock tool provider from the element
    original_provider = mock_host_element._components.pop(ToolProviderComponent.COMPONENT_TYPE, None)
    
    definitions = hud_component._prepare_tool_definitions()
    
    assert definitions is None
    # Restore provider if needed for other tests using the same element instance
    if original_provider:
         mock_host_element._components[ToolProviderComponent.COMPONENT_TYPE] = original_provider

def test_prepare_llm_messages_user_message_format(hud_component, mock_context_manager):
    """Test preparing messages using the default 'user_message' format."""
    system_prompt = "System test prompt"
    context_str = "Built context string"
    hud_component._state["system_prompt"] = system_prompt
    hud_component._state["history_representation"] = "user_message"
    mock_context_manager.build_context.return_value = context_str
    
    messages = hud_component._prepare_llm_messages_from_string(context_str)
    
    assert messages is not None
    assert len(messages) == 2
    assert messages[0] == LLMMessage(role="system", content=system_prompt)
    assert messages[1] == LLMMessage(role="user", content=context_str)

def test_prepare_llm_messages_list_format_fallback(hud_component, mock_context_manager, caplog):
    """Test 'message_list' format falls back and logs warning."""
    system_prompt = "System prompt"
    context_str = "Context string"
    hud_component._state["system_prompt"] = system_prompt
    hud_component._state["history_representation"] = "message_list" # Set unsupported format
    mock_context_manager.build_context.return_value = context_str

    with caplog.at_level(logging.WARNING):
        messages = hud_component._prepare_llm_messages_from_string(context_str)
        
    assert "'message_list' history representation not yet fully implemented" in caplog.text
    # Should fall back to user_message format
    assert messages is not None
    assert len(messages) == 2
    assert messages[0] == LLMMessage(role="system", content=system_prompt)
    assert messages[1] == LLMMessage(role="user", content=context_str)

@pytest.mark.asyncio
async def test_prepare_and_call_llm_success_content_only(hud_component, mock_context_manager, mock_tool_provider, mock_llm_provider):
    """Test the full LLM call cycle returning only content."""
    context_str = "Input context"
    response_content = "LLM final answer"
    mock_context_manager.build_context.return_value = context_str
    mock_tool_provider.get_llm_tool_schemas.return_value = [] # No tools defined
    
    # Configure the mock LLM provider's response
    expected_response = LLMResponse(content=response_content, tool_calls=None)
    mock_llm_provider.complete.return_value = expected_response
    
    result = await hud_component.prepare_and_call_llm()
    
    assert result == expected_response
    mock_context_manager.build_context.assert_called_once()
    mock_tool_provider.get_llm_tool_schemas.assert_called_once()
    # Check args passed to llm_provider.complete
    mock_llm_provider.complete.assert_awaited_once()
    call_args, call_kwargs = mock_llm_provider.complete.call_args
    assert len(call_kwargs['messages']) == 2 # System + User
    assert call_kwargs['messages'][1].content == context_str
    assert call_kwargs['tools'] == []
    assert hud_component._state["last_llm_response"] == expected_response

@pytest.mark.asyncio
async def test_prepare_and_call_llm_success_tool_call(hud_component, mock_context_manager, mock_tool_provider, mock_llm_provider):
    """Test the full LLM call cycle returning a tool call."""
    context_str = "Context leading to tool use"
    mock_tools_schema = [{"type": "function", "function": {"name": "some_tool"}}]
    mock_context_manager.build_context.return_value = context_str
    mock_tool_provider.get_llm_tool_schemas.return_value = mock_tools_schema
    
    tool_call = LLMToolCall(tool_call_id="tc1", tool_name="some_tool", parameters={})
    expected_response = LLMResponse(content=None, tool_calls=[tool_call])
    mock_llm_provider.complete.return_value = expected_response
    
    result = await hud_component.prepare_and_call_llm()
    
    assert result == expected_response
    mock_llm_provider.complete.assert_awaited_once()
    call_args, call_kwargs = mock_llm_provider.complete.call_args
    assert call_kwargs['tools'] == mock_tools_schema
    assert hud_component._state["last_llm_response"] == expected_response

@pytest.mark.asyncio
async def test_prepare_and_call_llm_no_provider(hud_component, mock_host_element, caplog):
    """Test prepare_and_call_llm fails gracefully if provider is None."""
    hud_component._llm_provider = None # Simulate missing provider
    with caplog.at_level(logging.ERROR):
        result = await hud_component.prepare_and_call_llm()
    assert result is None
    assert "LLMProvider is not configured" in caplog.text

@pytest.mark.asyncio
async def test_prepare_and_call_llm_no_context_manager(hud_component, mock_host_element, caplog):
    """Test prepare_and_call_llm fails gracefully if ContextManager is missing."""
    # Remove context manager
    original_cm = mock_host_element._components.pop(ContextManagerComponent.COMPONENT_TYPE, None)
    with caplog.at_level(logging.ERROR):
        result = await hud_component.prepare_and_call_llm()
    assert result is None
    assert "ContextManagerComponent missing" in caplog.text
    # Restore if needed
    if original_cm: mock_host_element._components[ContextManagerComponent.COMPONENT_TYPE] = original_cm

@pytest.mark.asyncio
async def test_prepare_and_call_llm_provider_error(hud_component, mock_context_manager, mock_tool_provider, mock_llm_provider, caplog):
    """Test handling of exceptions during the LLM provider call."""
    mock_context_manager.build_context.return_value = "Context"
    mock_tool_provider.get_llm_tool_schemas.return_value = []
    mock_llm_provider.complete.side_effect = Exception("LLM API Error")
    
    with caplog.at_level(logging.ERROR):
        result = await hud_component.prepare_and_call_llm()
        
    assert result is None
    assert "Error during LLM completion: LLM API Error" in caplog.text
    assert hud_component._state["last_llm_response"] is None # Should clear on error

# Test event handling if needed, requires async test runner setup usually
@pytest.mark.asyncio
async def test_on_event_run_request(hud_component):
     """Test that 'run_agent_cycle_request' event triggers the call."""
     event = {"event_type": "run_agent_cycle_request"}
     timeline_context = {} # Not used by HUD's handler
     
     # Patch the method that gets called
     with patch.object(hud_component, 'prepare_and_call_llm', new_callable=AsyncMock) as mock_call:
         handled = hud_component._on_event(event, timeline_context)
         
         # Check the _on_event logic returns True/False if it matters
         # assert handled is True # Or False depending on design
         
         # Ensure the target method was called
         await asyncio.sleep(0) # Allow coro to run if event handler doesn't await
         mock_call.assert_called_once()
