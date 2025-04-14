import pytest
import time
from unittest.mock import MagicMock, patch, call
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
        self._components[ContainerComponent.COMPONENT_TYPE] = self._mock_container
        self._components[GlobalAttentionComponent.COMPONENT_TYPE] = self._mock_attention

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
def context_manager(mock_host_element):
    # Use default init values unless overridden in test
    component = ContextManagerComponent(element=mock_host_element)
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
    assert not context_manager._state["conversation_history"]
    assert context_manager._last_built_context is None

def test_initialization_custom_params(mock_host_element):
    """Test initialization with custom parameters."""
    component = ContextManagerComponent(
        element=mock_host_element,
        token_budget=1000,
        compression_strategy="summarize", # Example custom strategy
        history_max_turns=5
    )
    component._initialize() # Need to call this manually
    assert component._state["token_budget"] == 1000
    assert component._state["compression_strategy"] == "summarize"
    assert component._state["history_max_turns"] == 5

# --- History Management Tests ---

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

# --- Context Building Tests ---

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

def test_gather_information(context_manager, mock_container, mock_attention, setup_gather_mocks):
    """Test the _gather_information method."""
    context_manager.add_history_turn(role="user", content="User message")
    
    raw_info = context_manager._gather_information()
    
    # Verify calls to dependencies
    mock_attention.get_state.assert_called_once()
    mock_container.get_mounted_elements.assert_called_once()
    
    # Check history
    assert "history" in raw_info
    assert len(raw_info["history"]) == 1
    assert raw_info["history"][0]["content"] == "User message"
    
    # Check attention
    assert "attention" in raw_info
    assert raw_info["attention"]["current_focus"] == "veil1"
    
    # Check elements gathered
    assert "elements" in raw_info
    elements = raw_info["elements"]
    # Should include InnerSpace itself (host element)
    assert context_manager.element.id in elements
    assert elements[context_manager.element.id]["type"] == "MockHostElement" 
    
    # Check mounted elements
    assert "plain1" in elements
    assert elements["plain1"]["mount_id"] == "mount_plain"
    assert elements["plain1"]["veil"] is None # Plain element has no veil
    assert elements["plain1"]["uplink_cache"] is None

    assert "veil1" in elements
    assert elements["veil1"]["mount_id"] == "mount_veil"
    assert elements["veil1"]["veil"] == {"type": "veil", "content": "Veil representation"} # Veil element has veil
    assert elements["veil1"]["uplink_cache"] is None
    
    assert "uplink1" in elements
    assert elements["uplink1"]["mount_id"] == "mount_uplink"
    assert elements["uplink1"]["veil"] is None # Uplink mock doesn't have veil
    assert elements["uplink1"]["uplink_cache"] == {"remote": "cached_state"} # Uplink has cache

def test_filter_and_prioritize(context_manager):
    """Test the prioritization logic."""
    # Simplified raw_info for testing priority
    raw_info = {
        "history": [
            {"role": "user", "content": "msg1"}, # Priority -2
            {"role": "assistant", "content": "msg2"} # Priority -1
        ],
        "attention": {"current_focus": "el_focused"},
        "elements": {
            "host_for_cmc": {"id": "host_for_cmc", "type": "InnerSpace", "name": "Host"}, # Priority 1
            "el_normal": {"id": "el_normal", "type": "SomeElement", "name": "Normal"},     # Priority 100
            "el_focused": {"id": "el_focused", "type": "FocusElement", "name": "Focused"}, # Priority 5
            "el_chat": {"id": "el_chat", "type": "ChatElement", "name": "Chat"},         # Priority 10
            "el_uplink": {"id": "el_uplink", "type": "UplinkProxy", "name": "Uplink"},     # Priority 15
        }
    }
    
    prioritized = context_manager._filter_and_prioritize(raw_info)
    
    # Expected order: history, InnerSpace, focused, chat, uplink, normal
    expected_ids_order = ["msg1", "msg2", "host_for_cmc", "el_focused", "el_chat", "el_uplink", "el_normal"]
    
    actual_ids_order = []
    for priority, type, data in prioritized:
        if type == "history":
            actual_ids_order.append(data["content"])
        elif type == "element":
            actual_ids_order.append(data["id"])
            
    assert actual_ids_order == expected_ids_order

def test_format_intermediate_basic(context_manager):
    """Test the basic _format_intermediate which currently only handles history."""
    prioritized_items = [
        (-2, "history", {"role": "user", "content": "Question?"}),
        (-1, "history", {"role": "assistant", "content": "Answer."}),
        (5, "element", {"id": "el1", "name": "SomeElement", "type": "TypeA"}), # Should be ignored by basic formatter
    ]
    
    formatted_str = context_manager._format_intermediate(prioritized_items)
    
    # Check basic formatting (exact format depends on implementation detail, check presence)
    assert "user: Question?" in formatted_str 
    assert "assistant: Answer." in formatted_str
    # Ensure element data isn't included in this basic version
    assert "SomeElement" not in formatted_str 
    assert "TypeA" not in formatted_str

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
