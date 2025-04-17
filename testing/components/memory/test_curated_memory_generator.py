import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch, call

# Import the component to test
from elements.elements.components.memory.curated_memory_generator import CuratedMemoryGenerationComponent
# Import dependencies to mock
from elements.elements.base import BaseElement, Component
from elements.elements.components.context_manager_component import ContextManagerComponent
from elements.elements.components.memory.structured_memory_component import StructuredMemoryComponent
from elements.elements.components.hud_component import HUDComponent
from elements.elements.llm.provider_interface import LLMProviderInterface, LLMResponse

# Mock Element to host the component and its dependencies
class MockCuratedHostElement(BaseElement):
    def __init__(self, element_id=\"cur_host\"):
        self.id = element_id
        self.name = \"MockCuratedHost\"
        self._components = {}
        # Mock dependencies
        self.mock_ctx_mgr = AsyncMock(spec=ContextManagerComponent)
        self.mock_mem_store = AsyncMock(spec=StructuredMemoryComponent) 
        self.mock_hud = MagicMock(spec=HUDComponent)
        self.mock_llm_provider = AsyncMock(spec=LLMProviderInterface)
        
        # Setup HUD mock to provide the LLM provider
        self.mock_hud._llm_provider = self.mock_llm_provider
        self.mock_hud._state = {"model": "default_agent_model"} # Default agent model from HUD
        
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
    element = MockCuratedHostElement()
    # Configure default mock returns for successful path
    element.mock_ctx_mgr.get_history_tail.return_value = [] 
    element.mock_mem_store.add_memory.return_value = \"new_cur_mem_id_123\" 
    # Simulate successful responses for each step
    element.mock_llm_provider.complete.side_effect = [
        AsyncMock(spec=LLMResponse, content=\"Extracted Quote\"),      # 1. Quote
        AsyncMock(spec=LLMResponse, content=\"Agent Perspective\"), # 2. Perspective
        AsyncMock(spec=LLMResponse, content=\"CM Analysis\"),       # 3. Analysis
        AsyncMock(spec=LLMResponse, content=\"Refined Perspective\") # 4. Refinement
    ]
    return element

@pytest.fixture
def curated_generator(mock_host_element):
    # Use default init values (includes specific CM model)
    component = CuratedMemoryGenerationComponent(element=mock_host_element)
    component._initialize()
    component._is_enabled = True
    return component

# --- Test Cases ---

@pytest.mark.asyncio
async def test_generate_memory_success_default_models(curated_generator, mock_host_element):
    \"\"\"Test successful generation using default agent/CM models.\"\"\"
    mock_ctx_mgr = mock_host_element.mock_ctx_mgr
    mock_mem_store = mock_host_element.mock_mem_store
    mock_llm_provider = mock_host_element.mock_llm_provider
    
    chunk_messages = [
        {\"role\": \"user\", \"content\": \"Curated Q1\", \"timestamp\": 2000, \"id\": \"cm1\"},
        {\"role\": \"assistant\", \"content\": \"Curated R1\", \"timestamp\": 2100, \"id\": \"cm2\"}
    ]
    history_tail = [{\"role\": \"system\", \"content\": \"Old curated context\", \"timestamp\": 1500}]
    mock_ctx_mgr.get_history_tail.return_value = history_tail
    
    # Reset side effect for predictable responses in this test
    llm_responses = [
        AsyncMock(spec=LLMResponse, content=\"Test Quote\"), 
        AsyncMock(spec=LLMResponse, content=\"Test Perspective\"),
        AsyncMock(spec=LLMResponse, content=\"Test Analysis\"),
        AsyncMock(spec=LLMResponse, content=\"Test Refined Perspective\")
    ]
    mock_llm_provider.complete.side_effect = llm_responses
    mock_mem_store.add_memory.return_value = \"cur_mem_success_1\" 
    
    memory_id = await curated_generator.generate_memory_for_chunk(chunk_messages)
    
    assert memory_id == \"cur_mem_success_1\"
    
    # Verify Context Manager call
    mock_ctx_mgr.get_history_tail.assert_awaited_once_with(
        before_timestamp=2000, 
        token_limit=curated_generator._state[\"tail_token_limit"]
    )
    
    # Verify ALL LLM calls were made
    assert mock_llm_provider.complete.await_count == 4
    calls = mock_llm_provider.complete.await_args_list
    
    # --- Check Call 1: Quote --- 
    args1, kwargs1 = calls[0]
    assert kwargs1.get('model') == \"default_agent_model\" # Default Agent model
    prompt1 = kwargs1.get('messages')[0][\"content\"]
    assert curated_generator._state[\"primer_content\"] in prompt1
    assert json.dumps(history_tail) in prompt1
    assert \"Curated Q1\" in prompt1
    assert \"Curated R1\" in prompt1
    assert \"extract a direct quote\" in prompt1 # Check specific prompt part

    # --- Check Call 2: Perspective --- 
    args2, kwargs2 = calls[1]
    assert kwargs2.get('model') == \"default_agent_model\" # Default Agent model
    prompt2 = kwargs2.get('messages')[0][\"content\"]
    assert \"Extracted Quote: Test Quote\" in prompt2 # Check quote included
    assert \"what was your perspective\" in prompt2 # Check specific prompt part
    
    # --- Check Call 3: Analysis --- 
    args3, kwargs3 = calls[2]
    # Should use the specific CM model defined in component defaults
    expected_cm_model = curated_generator._state.get(\"cm_model_override\")
    assert kwargs3.get('model') == expected_cm_model 
    prompt3 = kwargs3.get('messages')[0][\"content\"]
    assert \"Extracted Quote: Test Quote\" in prompt3
    assert \"Agent Perspective: Test Perspective\" in prompt3 # Check perspective included
    assert \"objective analytical summary\" in prompt3 # Check specific prompt part
    
    # --- Check Call 4: Refinement --- 
    args4, kwargs4 = calls[3]
    assert kwargs4.get('model') == \"default_agent_model\" # Default Agent model
    prompt4 = kwargs4.get('messages')[0][\"content\"]
    assert \"Extracted Quote: Test Quote\" in prompt4
    assert \"Initial Perspective: Test Perspective\" in prompt4
    assert \"Analytical Summary: Test Analysis\" in prompt4 # Check analysis included
    assert \"Refine your perspective\" in prompt4 # Check specific prompt part
    
    # Verify memory store call
    mock_mem_store.add_memory.assert_called_once()
    add_mem_args, _ = mock_mem_store.add_memory.call_args
    memory_data = add_mem_args[0]
    # Check assembled content
    assert len(memory_data[\"content\"]) == 6
    assert memory_data[\"content\"][0][\"content\"] == f\"Quote regarding period {chunk_messages[0][\"timestamp\"]}:\"
    assert memory_data[\"content\"][1][\"content\"] == \"Test Quote\"
    assert memory_data[\"content\"][2][\"content\"] == \"Agent perspective on this period:\"
    assert memory_data[\"content\"][3][\"content\"] == \"Test Perspective\"
    assert \"Analysis: Test Analysis\" in memory_data[\"content\"][4][\"content\"] # Check analysis formatting
    assert memory_data[\"content\"][5][\"content\"] == \"Test Refined Perspective\"
    # Check metadata
    assert memory_data[\"metadata\"] == {
        \"generation_method\": \"curated_6_step\", 
        \"agent_model\": \"default_agent_model\", 
        \"cm_model\": expected_cm_model
    }
    # Check source info
    assert memory_data[\"source_info\"] == {
        \"type\": \"chunk_summary\",
        \"chunk_message_ids\": [\"cm1\", \"cm2\"],
        \"chunk_start_timestamp\": 2000,
        \"chunk_end_timestamp\": 2100
    }
    assert \"timestamp\" in memory_data 

@pytest.mark.asyncio
async def test_generate_memory_success_override_models(mock_host_element):
    \"\"\"Test successful generation using overridden agent/CM models.\"\"\"
    agent_override = \"agent-override-model\"
    cm_override = \"cm-override-model\"
    generator = CuratedMemoryGenerationComponent(
        element=mock_host_element, 
        agent_model_override=agent_override,
        cm_model_override=cm_override
    )
    generator._initialize()
    generator._is_enabled = True
    
    mock_llm_provider = mock_host_element.mock_llm_provider
    mock_mem_store = mock_host_element.mock_mem_store
    chunk_messages = [{\"role\": \"user\", \"content\": \"Q\", \"timestamp\": 100, \"id\": \"m1\"}]
    
    # Reset side effect for predictable responses
    llm_responses = [
        AsyncMock(spec=LLMResponse, content=\"OQuote\"), 
        AsyncMock(spec=LLMResponse, content=\"OPersp\"),
        AsyncMock(spec=LLMResponse, content=\"OAnalysis\"),
        AsyncMock(spec=LLMResponse, content=\"ORefined\")
    ]
    mock_llm_provider.complete.side_effect = llm_responses
    
    await generator.generate_memory_for_chunk(chunk_messages)
    
    # Verify models used in LLM calls
    assert mock_llm_provider.complete.await_count == 4
    calls = mock_llm_provider.complete.await_args_list
    assert calls[0][1].get('model') == agent_override # Quote
    assert calls[1][1].get('model') == agent_override # Perspective
    assert calls[2][1].get('model') == cm_override    # Analysis (CM Override)
    assert calls[3][1].get('model') == agent_override # Refinement
    
    # Verify metadata reflects the override models
    mock_mem_store.add_memory.assert_called_once()
    add_mem_args, _ = mock_mem_store.add_memory.call_args
    assert add_mem_args[0][\"metadata\"][\"agent_model\"] == agent_override
    assert add_mem_args[0][\"metadata\"][\"cm_model\"] == cm_override

@pytest.mark.asyncio
async def test_generate_memory_empty_chunk(curated_generator):
    \"\"\"Test that generation is skipped for an empty chunk.\"\"\"
    memory_id = await curated_generator.generate_memory_for_chunk([])
    assert memory_id is None
    curated_generator.element.mock_llm_provider.complete.assert_not_awaited()
    curated_generator.element.mock_mem_store.add_memory.assert_not_called()

@pytest.mark.asyncio
async def test_generate_memory_llm_failure_step1(curated_generator, mock_host_element):
    \"\"\"Test handling when the first LLM call (quote) fails.\"\"\"
    mock_llm_provider = mock_host_element.mock_llm_provider
    mock_mem_store = mock_host_element.mock_mem_store
    
    # Simulate LLM error only on the first call
    mock_llm_provider.complete.side_effect = Exception(\"Quote Error\")
    
    chunk_messages = [{\"role\": \"user\", \"content\": \"Q\", \"timestamp\": 100}]
    memory_id = await curated_generator.generate_memory_for_chunk(chunk_messages)
    
    assert memory_id is None
    mock_llm_provider.complete.assert_awaited_once() # Only first call attempted
    mock_mem_store.add_memory.assert_not_called()
    
    # Restore side effect for other tests if fixture is reused heavily
    # (May need more sophisticated fixture setup for side effects)
    mock_llm_provider.complete.side_effect = None 