import pytest
import asyncio # Added for async tests
import time
from unittest.mock import MagicMock, patch, call, AsyncMock # Added AsyncMock
from typing import Optional, Dict, Any, List

try:
    # Import the class to test
    from elements.elements.components.context_manager_component import ContextManagerComponent, estimate_tokens
    # Import base classes and dependencies
    from elements.elements.base import BaseElement, Component
    from elements.elements.components.space.container_component import ContainerComponent
    from elements.elements.components.global_attention_component import GlobalAttentionComponent
    from elements.elements.components.veil_producer_component import VeilProducer
    from elements.elements.uplink import UplinkProxy # Assume UplinkProxy is importable
    from elements.elements.components.uplink.cache_component import RemoteStateCacheComponent
    # Import memory component dependency
    from elements.elements.components.memory.structured_memory_component import StructuredMemoryComponent
except ImportError as e:
    pytest.skip(f"Skipping ContextManagerComponent tests due to import error: {e}", allow_module_level=True)

# --- Mocks and Fixtures ---

# Mock Element to host the component
class MockHostElement(BaseElement):
    def __init__(self, element_id="host_for_cmc"):
        self.id = element_id
        self.name = "MockHostElement"
        self._components = {}
        self._mock_container = MagicMock(spec=ContainerComponent)
        self._mock_attention = MagicMock(spec=GlobalAttentionComponent)
        # Add mock for memory store dependency
        self._mock_memory_store = MagicMock(spec=StructuredMemoryComponent)
        self._components[ContainerComponent.COMPONENT_TYPE] = self._mock_container
        self._components[GlobalAttentionComponent.COMPONENT_TYPE] = self._mock_attention
        self._components[StructuredMemoryComponent.COMPONENT_TYPE] = self._mock_memory_store # Register mock

    def add_component(self, component: Component):
        self._components[component.COMPONENT_TYPE] = component
        component.element = self

    def get_component(self, component_type_or_class) -> Optional[Component]:
        component_type = component_type_or_class
        if isinstance(component_type_or_class, type):
            component_type = getattr(component_type_or_class, 'COMPONENT_TYPE', None)
            if not component_type: # Fallback for base classes?
                # This logic might need adjustment based on how base types are looked up
                for comp in self._components.values():
                    if isinstance(comp, component_type_or_class):
                        return comp
                return None

        return self._components.get(component_type)
        
    def has_component(self, component_type_or_class) -> bool:
        # Simple check for testing _gather_information
        if self.get_component(component_type_or_class):
            return True
        return False


# Mock Mounted Elements
class MockPlainMountedElement(BaseElement):
    def __init__(self, element_id="plain_el", name="Plain"):
        self.id = element_id
        self.name = name
        self._components = {}
    def get_component(self, *args, **kwargs): return None
    def has_component(self, *args, **kwargs): return False

class MockVeilMountedElement(BaseElement):
    def __init__(self, element_id="veil_el", name="VeilProducerElement"):
        self.id = element_id
        self.name = name
        self._components = {}
        self._mock_veil_producer = MagicMock(spec=VeilProducer)
        self._mock_veil_producer.produce_veil.return_value = {"type": "veil", "content": "Veil representation"}
        self._components[VeilProducer.COMPONENT_TYPE] = self._mock_veil_producer
    def get_component(self, component_type_or_class):
        if isinstance(component_type_or_class, type) and component_type_or_class == VeilProducer:
             return self._mock_veil_producer
        elif component_type_or_class == VeilProducer.COMPONENT_TYPE:
             return self._mock_veil_producer
        return None
    def has_component(self, component_type_or_class):
        if isinstance(component_type_or_class, type) and component_type_or_class == VeilProducer:
            return True
        elif component_type_or_class == VeilProducer.COMPONENT_TYPE:
            return True
        return False
        
class MockUplinkElement(UplinkProxy): # Inherit to satisfy isinstance check
     def __init__(self, element_id="uplink_el", name="Uplink"):
        self.id = element_id
        self.name = name
        self._components = {}
        self._mock_cache_comp = MagicMock(spec=RemoteStateCacheComponent)
        self._mock_cache_comp.get_synced_remote_state.return_value = {"remote": "cached_state"}
        self._components[RemoteStateCacheComponent.COMPONENT_TYPE] = self._mock_cache_comp
     def get_component(self, component_type_or_class):
        if isinstance(component_type_or_class, type) and component_type_or_class == RemoteStateCacheComponent:
             return self._mock_cache_comp
        elif component_type_or_class == RemoteStateCacheComponent.COMPONENT_TYPE:
             return self._mock_cache_comp
        return None
     def has_component(self, *args, **kwargs): return False # Doesn't have VeilProducer

@pytest.fixture
def mock_host_element():
    return MockHostElement()

@pytest.fixture
def mock_container(mock_host_element):
    return mock_host_element._mock_container

@pytest.fixture
def mock_attention(mock_host_element):
    return mock_host_element._mock_attention

@pytest.fixture
def mock_memory_store(mock_host_element):
    # Provide default return values for methods used by ContextManager
    mock_store = mock_host_element._mock_memory_store
    mock_store.get_memories.return_value = [] # Default: no memories
    return mock_store

@pytest.fixture
def context_manager(mock_host_element):
    # Use default init values unless overridden in test
    component = ContextManagerComponent(
        element=mock_host_element,
        # Ensure tiktoken doesn't interfere if not installed/mocked
        # We'll mock _get_token_count directly in tests where needed
        tokenizer_model='test_model_for_init_only' 
    )
    component._tokenizer = None # Force fallback estimator unless mocked
    # Manually initialize for testing direct calls
    component._initialize()
    component._is_enabled = True
    return component

# --- Initialization Tests ---

def test_initialization_defaults(context_manager, mock_host_element):
    """Test initialization with default parameters."""
    assert context_manager.element == mock_host_element
    assert context_manager._state["token_budget"] == 4000
    assert context_manager._state["compression_strategy"] == "truncate_recent"
    assert context_manager._state["history_max_turns"] == 20
    assert context_manager._state["max_memories_in_context"] == 5 # Check new default
    assert not context_manager._state["conversation_history"]
    assert context_manager._last_built_context is None

def test_initialization_custom_params(mock_host_element):
    """Test initialization with custom parameters."""
    component = ContextManagerComponent(
        element=mock_host_element,
        token_budget=1000,
        compression_strategy="summarize", # Example custom strategy
        history_max_turns=5,
        max_memories_in_context=10 # Custom memory count
    )
    component._initialize() # Need to call this manually
    assert component._state["token_budget"] == 1000
    assert component._state["compression_strategy"] == "summarize"
    assert component._state["history_max_turns"] == 5
    assert component._state["max_memories_in_context"] == 10 # Check custom value

# --- History Management Tests (Including Memory Marker) ---

def test_add_history_turn(context_manager):
    """Test adding turns to conversation history."""
    assert len(context_manager._state["conversation_history"]) == 0
    context_manager.add_history_turn(role="user", content="Hello")
    assert len(context_manager._state["conversation_history"]) == 1
    turn1 = context_manager._state["conversation_history"][0]
    assert turn1["role"] == "user"
    assert turn1["content"] == "Hello"
    assert "timestamp" in turn1

    context_manager.add_history_turn(role="assistant", content="Hi there", name="Bot")
    assert len(context_manager._state["conversation_history"]) == 2
    turn2 = context_manager._state["conversation_history"][1]
    assert turn2["role"] == "assistant"
    assert turn2["content"] == "Hi there"
    assert turn2["name"] == "Bot"

def test_clear_history(context_manager):
    """Test clearing conversation history."""
    context_manager.add_history_turn(role="user", content="Message 1")
    context_manager.add_history_turn(role="assistant", content="Message 2")
    assert len(context_manager._state["conversation_history"]) == 2
    context_manager.clear_history()
    assert len(context_manager._state["conversation_history"]) == 0

def test_get_recent_history(context_manager):
    """Test retrieving recent history respects history_max_turns."""
    context_manager._state["history_max_turns"] = 3
    for i in range(5):
        context_manager.add_history_turn(role="user", content=f"Message {i+1}")
    
    recent_history = context_manager._get_recent_history()
    assert len(recent_history) == 3
    assert recent_history[0]["content"] == "Message 3"
    assert recent_history[1]["content"] == "Message 4"
    assert recent_history[2]["content"] == "Message 5"

def test_update_processed_marker(context_manager):
    """Test updating the processed history marker."""
    assert context_manager._state["processed_history_marker_timestamp"] is None
    
    context_manager.update_processed_marker(1000)
    assert context_manager._state["processed_history_marker_timestamp"] == 1000
    
    # Update with newer timestamp
    context_manager.update_processed_marker(1500)
    assert context_manager._state["processed_history_marker_timestamp"] == 1500
    
    # Try updating with older timestamp (should be ignored)
    context_manager.update_processed_marker(1200)
    assert context_manager._state["processed_history_marker_timestamp"] == 1500
    
    # Clear history should reset marker
    context_manager.clear_history()
    assert context_manager._state["processed_history_marker_timestamp"] is None

@pytest.mark.asyncio
async def test_get_unprocessed_history_chunks_no_marker(context_manager):
    """Test getting chunks when no marker is set (get all history)."""
    # Mock token counting for simplicity (1 token per char + 10 overhead)
    with patch.object(context_manager, '_get_token_count', side_effect=lambda t: len(t)):
        context_manager.add_history_turn(role="user", content="Msg1", timestamp=100) # 4 + 10 = 14 tokens
        context_manager.add_history_turn(role="asst", content="Msg2", timestamp=200) # 4 + 10 = 14 tokens
        context_manager.add_history_turn(role="user", content="Msg3", timestamp=300) # 4 + 10 = 14 tokens
        
        # Max chunk tokens = 30 -> should split after 2 messages
        chunks = await context_manager.get_unprocessed_history_chunks(limit=None, max_chunk_tokens=30)
        
        assert len(chunks) == 2
        assert len(chunks[0]) == 2 # Msg1, Msg2 (14 + 14 = 28 <= 30)
        assert chunks[0][0]["content"] == "Msg1"
        assert chunks[0][1]["content"] == "Msg2"
        assert len(chunks[1]) == 1 # Msg3 (14 <= 30)
        assert chunks[1][0]["content"] == "Msg3"
        
@pytest.mark.asyncio
async def test_get_unprocessed_history_chunks_with_marker(context_manager):
    """Test getting chunks only after the marker timestamp."""
    with patch.object(context_manager, '_get_token_count', side_effect=lambda t: len(t)):
        context_manager.add_history_turn(role="user", content="Processed1", timestamp=100)
        context_manager.add_history_turn(role="asst", content="Processed2", timestamp=200)
        context_manager.update_processed_marker(200) # Mark first two as processed
        context_manager.add_history_turn(role="user", content="Unprocessed1", timestamp=300) # 12 + 10 = 22 tokens
        context_manager.add_history_turn(role="asst", content="Unprocessed2", timestamp=400) # 12 + 10 = 22 tokens
        
        # Max chunk tokens = 50
        chunks = await context_manager.get_unprocessed_history_chunks(limit=None, max_chunk_tokens=50)
        
        assert len(chunks) == 1
        assert len(chunks[0]) == 2 # Both fit (22 + 22 = 44 <= 50)
        assert chunks[0][0]["content"] == "Unprocessed1"
        assert chunks[0][1]["content"] == "Unprocessed2"

@pytest.mark.asyncio
async def test_get_unprocessed_history_chunks_limit(context_manager):
    """Test the limit parameter for chunks."""
    with patch.object(context_manager, '_get_token_count', side_effect=lambda t: len(t)):
        context_manager.add_history_turn(role="user", content="A", timestamp=100) # 11 tokens
        context_manager.add_history_turn(role="asst", content="B", timestamp=200) # 11 tokens
        context_manager.add_history_turn(role="user", content="C", timestamp=300) # 11 tokens
        
        # Max chunk tokens = 15 -> each message is a chunk
        chunks = await context_manager.get_unprocessed_history_chunks(limit=2, max_chunk_tokens=15)
        
        assert len(chunks) == 2 # Respect limit
        assert len(chunks[0]) == 1 
        assert chunks[0][0]["content"] == "A"
        assert len(chunks[1]) == 1
        assert chunks[1][0]["content"] == "B"
        # Chunk C is excluded due to limit

@pytest.mark.asyncio
async def test_get_unprocessed_history_chunks_no_unprocessed(context_manager):
    """Test when all history is marked as processed."""
    context_manager.add_history_turn(role="user", content="Msg", timestamp=100)
    context_manager.update_processed_marker(100)
    chunks = await context_manager.get_unprocessed_history_chunks()
    assert len(chunks) == 0

# --- Context Building Tests (Including Memory) ---

@pytest.fixture
def setup_gather_mocks(mock_container, mock_attention):
    # Setup mock elements to be "mounted"
    plain_el = MockPlainMountedElement(element_id="plain1", name="PlainElement")
    veil_el = MockVeilMountedElement(element_id="veil1", name="VeilElement")
    uplink_el = MockUplinkElement(element_id="uplink1", name="UplinkElement")
    
    mounted_elements = {
        "mount_plain": plain_el,
        "mount_veil": veil_el,
        "mount_uplink": uplink_el,
    }
    mock_container.get_mounted_elements.return_value = mounted_elements
    mock_attention.get_state.return_value = {"current_focus": "veil1", "requests": []}
    
    return mounted_elements # Return for potential verification

def test_gather_information_with_memories(context_manager, mock_memory_store, setup_gather_mocks):
    """Test _gather_information includes memory retrieval."""
    context_manager._state["max_memories_in_context"] = 3 # Set limit for test
    mock_memories = [
        {"memory_id": "m1", "content": "Memory 1"},
        {"memory_id": "m2", "content": "Memory 2"}
    ]
    mock_memory_store.get_memories.return_value = mock_memories
    
    raw_info = context_manager._gather_information()
    
    # Verify memory store was called
    mock_memory_store.get_memories.assert_called_once_with(limit=3)
    
    # Check memories are in the result
    assert "memories" in raw_info
    assert raw_info["memories"] == mock_memories
    
    # Check other parts are still gathered (basic check)
    assert "history" in raw_info
    assert "elements" in raw_info
    assert "attention" in raw_info

def test_filter_and_prioritize_with_memories(context_manager):
    """Test prioritization includes memories correctly."""
    raw_info = {
        "memories": [
            {"memory_id": "mem1", "timestamp": 100},
            {"memory_id": "mem2", "timestamp": 200} # More recent memory
        ],
        "history": [
            {"role": "user", "content": "H1", "timestamp": 300},
            {"role": "asst", "content": "H2", "timestamp": 400} # More recent history
        ],
        "elements": {
            "el1": {"id": "el1", "type": "TypeA"}
        },
        "attention": {}
    }
    
    prioritized = context_manager._filter_and_prioritize(raw_info)
    
    # Check order (Memories first (most recent first within memories), then History (most recent first), then Elements)
    assert len(prioritized) == 5 # 2 mem + 2 hist + 1 el
    assert prioritized[0][1] == "memory" and prioritized[0][2]["memory_id"] == "mem2" # Most recent mem first
    assert prioritized[1][1] == "memory" and prioritized[1][2]["memory_id"] == "mem1"
    assert prioritized[2][1] == "history" and prioritized[2][2]["content"] == "H2" # Most recent hist first
    assert prioritized[3][1] == "history" and prioritized[3][2]["content"] == "H1"
    assert prioritized[4][1] == "element" 

def test_format_intermediate_with_memories(context_manager):
    """Test _format_intermediate correctly formats memory items."""
    # Create prioritized items including a memory
    memory_item = {
        "memory_id": "mem_abc", "timestamp": 12345,
        "content": [
            {"role": "system", "content": "Quote: ..."},
            {"role": "assistant", "content": "My perspective..."}
        ]
    }
    history_item = {"role": "user", "content": "User asks question"}
    
    prioritized_items = [
        (-1001, "memory", memory_item), # Memory first
        (-101, "history", history_item)
    ]
    
    formatted_str = context_manager._format_intermediate(prioritized_items)
    
    # Basic checks for formatting
    assert "--- Memory (ID: mem_abc, Timestamp: 12345) ---" in formatted_str
    assert "System: Quote: ..." in formatted_str
    assert "Assistant: My perspective..." in formatted_str
    assert "--- End Memory ---" in formatted_str
    assert "User: User asks question" in formatted_str
    # Check spacing between sections
    assert "--- End Memory ---\n\nUser: User asks question" in formatted_str

# --- Other Tests ---

@patch('elements.elements.components.context_manager_component.estimate_tokens', side_effect=lambda t: len(t)) # Simple 1 char = 1 token mock
def test_compress_information_truncate(mock_estimate, context_manager):
    """Test the basic truncate_recent compression strategy."""
    context_manager._state["token_budget"] = 20
    context_manager._state["compression_strategy"] = "truncate_recent"
    
    # Input longer than budget
    long_formatted_str = "This is a very long string that exceeds the budget." # Length > 20
    compressed_str = context_manager._compress_information(long_formatted_str)
    assert len(compressed_str) <= 20 
    assert compressed_str == long_formatted_str[-20:] # Should take the last 20 chars
    
    # Input within budget
    short_formatted_str = "Short string." # Length < 20
    compressed_str = context_manager._compress_information(short_formatted_str)
    assert compressed_str == short_formatted_str # Should remain unchanged

@patch('elements.elements.components.context_manager_component.estimate_tokens', side_effect=lambda t: len(t))
def test_compress_information_unsupported_strategy(mock_estimate, context_manager):
     """Test handling of an unknown compression strategy (should likely fallback or warn)."""
     context_manager._state["token_budget"] = 10
     context_manager._state["compression_strategy"] = "unknown_magic"
     
     formatted_str = "Some input string."
     # Current basic implementation falls back to truncate
     compressed_str = context_manager._compress_information(formatted_str)
     assert compressed_str == formatted_str[-10:] # Falls back to truncate

def test_build_context_calls_pipeline(context_manager):
     """Test that build_context calls the internal methods in order."""
     with patch.object(context_manager, '_gather_information', return_value={"history": [], "elements": {}}) as mock_gather, \
          patch.object(context_manager, '_filter_and_prioritize', return_value=[]) as mock_filter, \
          patch.object(context_manager, '_format_intermediate', return_value="Formatted") as mock_format, \
          patch.object(context_manager, '_compress_information', return_value="Compressed") as mock_compress:
          
          result = context_manager.build_context()
          
          assert result == "Compressed"
          mock_gather.assert_called_once()
          mock_filter.assert_called_once_with(mock_gather.return_value)
          mock_format.assert_called_once_with(mock_filter.return_value)
          mock_compress.assert_called_once_with(mock_format.return_value)
          assert context_manager._last_built_context == "Compressed"
          assert context_manager._state["last_build_time"] is not None
