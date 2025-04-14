import pytest
from unittest.mock import MagicMock, call
import logging

try:
    from elements.elements.base import BaseElement, Component
    from elements.elements.components.core_tools import CoreToolsComponent
    from elements.elements.components.tool_provider_component import ToolProviderComponent
    # Assuming OutgoingActionCallback is importable or can be mocked simply
    from host.event_loop import OutgoingActionCallback 
except ImportError as e:
    pytest.skip(f"Skipping tests due to import error: {e}", allow_module_level=True)

# --- Mocks and Fixtures ---

class MockCoreElement(BaseElement):
    def __init__(self, element_id="mock_core_element"):
        self.id = element_id
        self._components = {}

    def add_component(self, component: Component):
        self._components[component.COMPONENT_TYPE] = component
        component.element = self

    def get_component(self, component_type: str):
        return self._components.get(component_type)

@pytest.fixture
def mock_element():
    return MockCoreElement()

@pytest.fixture
def mock_tool_provider():
    # Mock ToolProviderComponent with the necessary register_tool method
    provider = MagicMock(spec=ToolProviderComponent)
    provider.register_tool = MagicMock()
    return provider
    
@pytest.fixture
def mock_outgoing_callback():
    # Simple mock for the callback function signature
    return MagicMock(spec=OutgoingActionCallback)

@pytest.fixture
def core_tools_component(mock_element):
    component = CoreToolsComponent()
    mock_element.add_component(component)
    component.logger = logging.getLogger("TestCoreTools") # For log capture
    return component

# --- Test Cases ---

def test_initialization(core_tools_component, mock_element):
    """Test that the component initializes correctly."""
    assert core_tools_component is not None
    assert core_tools_component.COMPONENT_TYPE == "core_tools"
    assert core_tools_component.element == mock_element
    assert core_tools_component._outgoing_action_callback is None

def test_set_outgoing_action_callback(core_tools_component, mock_outgoing_callback):
    """Test setting the outgoing action callback."""
    core_tools_component.set_outgoing_action_callback(mock_outgoing_callback)
    assert core_tools_component._outgoing_action_callback == mock_outgoing_callback

def test_register_tools_success(core_tools_component, mock_tool_provider, caplog):
    """Test register_tools runs successfully with a valid provider (currently does nothing)."""
    with caplog.at_level(logging.DEBUG):
        core_tools_component.register_tools(mock_tool_provider)
    
    # Currently, no tools are registered, so the mock provider's register_tool shouldn't be called.
    mock_tool_provider.register_tool.assert_not_called() 
    assert f"Registering core tools for element {core_tools_component.element.id}" in caplog.text

def test_register_tools_provider_missing(core_tools_component, caplog):
    """Test register_tools handles a missing tool provider."""
    with caplog.at_level(logging.ERROR):
        core_tools_component.register_tools(None)
        
    assert f"Cannot register core tools: ToolProvider component is missing on {core_tools_component.element.id}" in caplog.text

# Add more tests here if/when actual core tools are implemented in CoreToolsComponent 