import pytest
import time
from unittest.mock import MagicMock

# Import the component to test
from elements.elements.components.memory.structured_memory_component import StructuredMemoryComponent
# Import base classes/dependencies for context
from elements.elements.base import BaseElement

# Mock Element to host the component
class MockMemHostElement(BaseElement):
    def __init__(self, element_id="mem_host"):
        self.id = element_id
        self.name = "MockMemHost"
        self._components = {}

    def add_component(self, component):
        self._components[component.COMPONENT_TYPE] = component
        component.element = self

@pytest.fixture
def mock_host_element():
    return MockMemHostElement()

@pytest.fixture
def memory_store(mock_host_element):
    # Use default init values
    component = StructuredMemoryComponent(element=mock_host_element)
    # Manually initialize 
    component._initialize()
    component._is_enabled = True
    return component

# --- Test Cases ---

def test_add_memory_basic(memory_store):
    """Test adding a basic memory entry."""
    assert len(memory_store._state["memories"]) == 0
    memory_data = {
        "content": [{"role": "assistant", "content": "Summary 1"}],
        "source_info": {"type": "chunk", "chunk_id": "a"},
        "metadata": {"method": "self_query"}
    }
    
    memory_id = memory_store.add_memory(memory_data)
    
    assert memory_id is not None
    assert isinstance(memory_id, str)
    assert len(memory_store._state["memories"]) == 1
    
    stored_memory = memory_store._state["memories"][0]
    assert stored_memory["memory_id"] == memory_id
    assert "timestamp" in stored_memory
    assert isinstance(stored_memory["timestamp"], int)
    assert stored_memory["content"] == memory_data["content"]
    assert stored_memory["source_info"] == memory_data["source_info"]
    assert stored_memory["metadata"] == memory_data["metadata"]

def test_add_memory_predefined_id_timestamp(memory_store):
    """Test adding memory with predefined ID and timestamp."""
    predefined_id = "mem_test_123"
    predefined_ts = int(time.time() * 1000) - 10000 # 10 seconds ago
    memory_data = {
        "memory_id": predefined_id,
        "timestamp": predefined_ts,
        "content": "Simple content",
    }
    
    memory_id = memory_store.add_memory(memory_data)
    
    assert memory_id == predefined_id
    stored_memory = memory_store.get_memory_by_id(predefined_id)
    assert stored_memory is not None
    assert stored_memory["timestamp"] == predefined_ts

def test_get_memories_empty(memory_store):
    """Test getting memories when the store is empty."""
    assert memory_store.get_memories() == []

def test_get_memories_all_and_sorted(memory_store):
    """Test getting all memories, ensuring they are sorted by timestamp."""
    ts_now = int(time.time() * 1000)
    mem1 = {"content": "M1", "timestamp": ts_now - 200}
    mem2 = {"content": "M2", "timestamp": ts_now - 100}
    mem3 = {"content": "M3", "timestamp": ts_now - 300} # Oldest
    
    id1 = memory_store.add_memory(mem1)
    id2 = memory_store.add_memory(mem2)
    id3 = memory_store.add_memory(mem3)
    
    all_memories = memory_store.get_memories()
    assert len(all_memories) == 3
    # Check sorting (oldest first)
    assert all_memories[0]["memory_id"] == id3
    assert all_memories[1]["memory_id"] == id1
    assert all_memories[2]["memory_id"] == id2

def test_get_memories_limit(memory_store):
    """Test the limit parameter for get_memories."""
    ts_now = int(time.time() * 1000)
    ids = []
    for i in range(5):
        ids.append(memory_store.add_memory({"content": f"M{i}", "timestamp": ts_now + i * 10}))
        
    limited_memories = memory_store.get_memories(limit=3)
    assert len(limited_memories) == 3
    # Should return the last 3 added (most recent)
    assert limited_memories[0]["memory_id"] == ids[2]
    assert limited_memories[1]["memory_id"] == ids[3]
    assert limited_memories[2]["memory_id"] == ids[4]

def test_get_memories_time_filter(memory_store):
    """Test the since_timestamp and before_timestamp filters."""
    ts_now = int(time.time() * 1000)
    ids = []
    timestamps = [ts_now - 300, ts_now - 200, ts_now - 100, ts_now, ts_now + 100]
    for i, ts in enumerate(timestamps):
        ids.append(memory_store.add_memory({"content": f"M{i}", "timestamp": ts}))
    
    # Since filter (>= ts_now - 100)
    since_mem = memory_store.get_memories(since_timestamp=ts_now - 100)
    assert len(since_mem) == 3
    assert since_mem[0]["memory_id"] == ids[2]
    assert since_mem[1]["memory_id"] == ids[3]
    assert since_mem[2]["memory_id"] == ids[4]
    
    # Before filter (< ts_now)
    before_mem = memory_store.get_memories(before_timestamp=ts_now)
    assert len(before_mem) == 3
    assert before_mem[0]["memory_id"] == ids[0]
    assert before_mem[1]["memory_id"] == ids[1]
    assert before_mem[2]["memory_id"] == ids[2]
    
    # Combined filter (>= ts_now - 200 and < ts_now + 100)
    combined_mem = memory_store.get_memories(since_timestamp=ts_now - 200, before_timestamp=ts_now + 100)
    assert len(combined_mem) == 3
    assert combined_mem[0]["memory_id"] == ids[1]
    assert combined_mem[1]["memory_id"] == ids[2]
    assert combined_mem[2]["memory_id"] == ids[3]

def test_get_memory_by_id_found(memory_store):
    """Test retrieving a specific memory by ID when it exists."""
    memory_data = {"content": "Specific Memory"}
    memory_id = memory_store.add_memory(memory_data)
    
    retrieved_memory = memory_store.get_memory_by_id(memory_id)
    assert retrieved_memory is not None
    assert retrieved_memory["memory_id"] == memory_id
    assert retrieved_memory["content"] == "Specific Memory"
    
    # Ensure it returns a copy
    retrieved_memory["content"] = "Modified"
    original_stored = memory_store._state["memories"][0]
    assert original_stored["content"] == "Specific Memory"
    
def test_get_memory_by_id_not_found(memory_store):
    """Test retrieving a specific memory by ID when it does not exist."""
    memory_store.add_memory({"content": "Some Memory"})
    retrieved_memory = memory_store.get_memory_by_id("non_existent_id")
    assert retrieved_memory is None

def test_clear_memories(memory_store):
    """Test clearing all memories from the store."""
    memory_store.add_memory({"content": "Mem 1"})
    memory_store.add_memory({"content": "Mem 2"})
    assert len(memory_store._state["memories"]) == 2
    
    memory_store.clear_memories()
    assert len(memory_store._state["memories"]) == 0
    assert memory_store.get_memories() == [] 