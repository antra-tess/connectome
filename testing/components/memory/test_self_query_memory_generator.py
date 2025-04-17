import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch, call

# Import the component to test
from elements.elements.components.memory.self_query_memory_generator import SelfQueryMemoryGenerationComponent
# Import dependencies to mock
from elements.elements.base import BaseElement, Component
from elements.elements.components.context_manager_component import ContextManagerComponent
from elements.elements.components.memory.structured_memory_component import StructuredMemoryComponent
from elements.elements.components.hud_component import HUDComponent
from elements.elements.llm.provider_interface import LLMProviderInterface, LLMResponse

# Mock Element to host the component and its dependencies
class MockSelfQueryHostElement(BaseElement):
    def __init__(self, element_id="sq_host"):
        self.id = element_id
        self.name = "MockSQHost"
        self._components = {}
        # Mock dependencies
        self.mock_ctx_mgr = AsyncMock(spec=ContextManagerComponent)
        self.mock_mem_store = AsyncMock(spec=StructuredMemoryComponent) # Use AsyncMock if add_memory is async
        self.mock_hud = MagicMock(spec=HUDComponent)
        self.mock_llm_provider = AsyncMock(spec=LLMProviderInterface)
        
        # Setup HUD mock to provide the LLM provider
        self.mock_hud._llm_provider = self.mock_llm_provider
        self.mock_hud._state = {"model": "default_agent_model"} # Default model from HUD
        
        self._components[ContextManagerComponent.COMPONENT_TYPE] = self.mock_ctx_mgr
        self._components[StructuredMemoryComponent.COMPONENT_TYPE] = self.mock_mem_store
        self._components[HUDComponent.COMPONENT_TYPE] = self.mock_hud
        
    def add_component(self, component: Component):
        self._components[component.COMPONENT_TYPE] = component
        component.element = self
        
    def get_component(self, component_type_or_class):
        component_type = component_type_or_class
        if isinstance(component_type_or_class, type):
            component_type = getattr(component_type_or_class, 'COMPONENT_TYPE', None)
        return self._components.get(component_type)

@pytest.fixture
def mock_host_element():
    element = MockSelfQueryHostElement()
    # Configure default mock returns
    element.mock_ctx_mgr.get_history_tail.return_value = [] # Default: empty tail
    element.mock_mem_store.add_memory.return_value = "new_mem_id_123" # Default: success
    element.mock_llm_provider.complete.return_value = AsyncMock(spec=LLMResponse, content="LLM Summary") # Default: success
    return element

@pytest.fixture
def self_query_generator(mock_host_element):
    # Use default init values
    component = SelfQueryMemoryGenerationComponent(element=mock_host_element)
    # Manually initialize 
    component._initialize()
    component._is_enabled = True
    return component

# --- Test Cases ---

@pytest.mark.asyncio
async def test_generate_memory_success_default_model(self_query_generator, mock_host_element):
    """Test successful generation using default settings and HUD model."""
    mock_ctx_mgr = mock_host_element.mock_ctx_mgr
    mock_mem_store = mock_host_element.mock_mem_store
    mock_llm_provider = mock_host_element.mock_llm_provider
    
    chunk_messages = [
        {"role": "user", "content": "Query 1", "timestamp": 1000, "id": "msg1"},
        {"role": "assistant", "content": "Response 1", "timestamp": 1100, "id": "msg2"}
    ]
    history_tail = [{"role": "system", "content": "Old message", "timestamp": 500}]
    mock_ctx_mgr.get_history_tail.return_value = history_tail
    mock_llm_provider.complete.return_value = AsyncMock(spec=LLMResponse, content="Generated Summary")
    mock_mem_store.add_memory.return_value = "mem_success_1" 
    
    memory_id = await self_query_generator.generate_memory_for_chunk(chunk_messages)
    
    assert memory_id == "mem_success_1"
    
    # Verify dependencies called correctly
    mock_ctx_mgr.get_history_tail.assert_awaited_once_with(
        before_timestamp=1000, 
        token_limit=self_query_generator._state["tail_token_limit"]
    )
    
    # Verify LLM call
    mock_llm_provider.complete.assert_awaited_once()
    call_args, call_kwargs = mock_llm_provider.complete.call_args
    assert call_kwargs.get('model') == "default_agent_model" # Used HUD default
    prompt_messages = call_kwargs.get('messages')
    assert len(prompt_messages) == 1
    prompt_content = prompt_messages[0]["content"]
    # Check essential parts of the prompt
    assert self_query_generator._state["primer_content"] in prompt_content
    assert json.dumps(history_tail) in prompt_content
    assert "<user_turn><to_remember>" in prompt_content
    assert json.dumps(chunk_messages[0]["content"]) in prompt_content
    assert "<assistant_turn><to_remember>" in prompt_content
    assert json.dumps(chunk_messages[1]["content"]) in prompt_content
    assert "Chunk to Remember" in prompt_content
    # Check that the specific prompt string is in the call
    # This relies on the prompt string being defined in the tested module
    # A better approach might be to mock the constant or check for keywords
    assert "select the most relevant and important information" in prompt_content 
    
    # Verify memory store call
    mock_mem_store.add_memory.assert_called_once()
    add_mem_args, _ = mock_mem_store.add_memory.call_args
    memory_data = add_mem_args[0]
    assert memory_data["content"] == [{"role": "assistant", "content": "Generated Summary"}]
    assert memory_data["metadata"] == {"generation_method": "self_query", "agent_model": "default_agent_model"}
    assert memory_data["source_info"] == {
        "type": "chunk_summary",
        "chunk_message_ids": ["msg1", "msg2"],
        "chunk_start_timestamp": 1000,
        "chunk_end_timestamp": 1100
    }
    assert "timestamp" in memory_data # Timestamp of creation

@pytest.mark.asyncio
async def test_generate_memory_success_override_model(mock_host_element):
    """Test successful generation using an overridden agent model."""
    # Create component with override
    generator = SelfQueryMemoryGenerationComponent(
        element=mock_host_element, 
        agent_model_override="specific_model_for_sq"
    )
    generator._initialize()
    generator._is_enabled = True
    
    mock_llm_provider = mock_host_element.mock_llm_provider
    mock_mem_store = mock_host_element.mock_mem_store
    chunk_messages = [{"role": "user", "content": "Q", "timestamp": 100, "id": "m1"}]
    
    await generator.generate_memory_for_chunk(chunk_messages)
    
    # Verify LLM was called with the override model
    mock_llm_provider.complete.assert_awaited_once()
    _, call_kwargs = mock_llm_provider.complete.call_args
    assert call_kwargs.get('model') == "specific_model_for_sq"
    
    # Verify metadata reflects the override model
    mock_mem_store.add_memory.assert_called_once()
    add_mem_args, _ = mock_mem_store.add_memory.call_args
    assert add_mem_args[0]["metadata"]["agent_model"] == "specific_model_for_sq"

@pytest.mark.asyncio
async def test_generate_memory_empty_chunk(self_query_generator):
    """Test that generation is skipped for an empty chunk."""
    memory_id = await self_query_generator.generate_memory_for_chunk([])
    assert memory_id is None
    # Check dependencies were NOT called
    self_query_generator.element.mock_ctx_mgr.get_history_tail.assert_not_awaited()
    self_query_generator.element.mock_llm_provider.complete.assert_not_awaited()
    self_query_generator.element.mock_mem_store.add_memory.assert_not_called()

@pytest.mark.asyncio
async def test_generate_memory_missing_dependency(self_query_generator, mock_host_element):
    """Test handling when a dependency is missing."""
    # Simulate missing ContextManager
    original_ctx_mgr = mock_host_element._components.pop(ContextManagerComponent.COMPONENT_TYPE)
    
    chunk_messages = [{"role": "user", "content": "Q", "timestamp": 100}]
    memory_id = await self_query_generator.generate_memory_for_chunk(chunk_messages)
    assert memory_id is None
    
    # Restore for other tests
    mock_host_element._components[ContextManagerComponent.COMPONENT_TYPE] = original_ctx_mgr 

@pytest.mark.asyncio
async def test_generate_memory_llm_failure(self_query_generator, mock_host_element):
    """Test handling when the LLM call fails."""
    mock_llm_provider = mock_host_element.mock_llm_provider
    mock_mem_store = mock_host_element.mock_mem_store
    
    # Simulate LLM error
    mock_llm_provider.complete.side_effect = Exception("LLM API Error")
    
    chunk_messages = [{"role": "user", "content": "Q", "timestamp": 100}]
    memory_id = await self_query_generator.generate_memory_for_chunk(chunk_messages)
    
    assert memory_id is None
    mock_llm_provider.complete.assert_awaited_once() # Verify it was called
    mock_mem_store.add_memory.assert_not_called() # Verify add_memory wasn't called
    # Reset side effect
    mock_llm_provider.complete.side_effect = None
    
@pytest.mark.asyncio
async def test_generate_memory_storage_failure(self_query_generator, mock_host_element):
    """Test handling when storing the memory fails."""
    mock_mem_store = mock_host_element.mock_mem_store
    
    # Simulate storage error
    mock_mem_store.add_memory.return_value = None 
    
    chunk_messages = [{"role": "user", "content": "Q", "timestamp": 100}]
    memory_id = await self_query_generator.generate_memory_for_chunk(chunk_messages)
    
    assert memory_id is None
    mock_mem_store.add_memory.assert_called_once() # Verify it was called


</rewritten_file> 