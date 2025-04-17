import pytest
import unittest
from unittest.mock import MagicMock, patch, call

# Import the component to test
from elements.elements.components.core_tools import CoreToolsComponent
# Import dependencies needed for mocking or setup
from elements.elements.base import BaseElement
from elements.elements.components.tool_provider_component import ToolProviderComponent
from elements.elements.components.space.container_component import ContainerComponent
from elements.elements.components.element_factory_component import ElementFactoryComponent
from elements.elements.components.state.text_state_component import TextStateComponent

# Mock the Element and other components for isolated testing
class MockElement(BaseElement):
    def __init__(self, element_id="test_element", name="TestElement"):
        super().__init__(element_id, name)
        # Add mocked components needed by CoreToolsComponent dependencies or implementations
        self._components = {
            ContainerComponent.COMPONENT_TYPE: MagicMock(spec=ContainerComponent),
            ElementFactoryComponent.COMPONENT_TYPE: MagicMock(spec=ElementFactoryComponent),
            TextStateComponent.COMPONENT_TYPE: MagicMock(spec=TextStateComponent) # For notebook element
        }
        # Add the component under test
        self.core_tools = CoreToolsComponent(element=self)
        self._components[CoreToolsComponent.COMPONENT_TYPE] = self.core_tools

    def get_component(self, component_type: str):
        return self._components.get(component_type)

    def has_component(self, component_type: str) -> bool:
         return component_type in self._components

@pytest.fixture
def mock_element():
    element = MockElement()
    # Ensure dependencies are properly mocked if needed by implementations
    element.get_component(ContainerComponent.COMPONENT_TYPE).get_element_by_id.return_value = None # Default: element not found
    element.get_component(ElementFactoryComponent.COMPONENT_TYPE).create_element.return_value = MagicMock(spec=BaseElement) # Simulate successful creation
    return element

@pytest.fixture
def core_tools(mock_element): # Use the mock_element fixture
    return mock_element.core_tools

@pytest.fixture
def mock_tool_provider():
    provider = MagicMock(spec=ToolProviderComponent)
    # Use a dictionary to store registered tools for assertion
    registered_tools = {}
    def mock_register(name, description, parameters, execution_info):
        registered_tools[name] = {
            'description': description,
            'parameters': parameters,
            'execution_info': execution_info
        }
        return True # Simulate successful registration

    provider.register_tool = MagicMock(side_effect=mock_register)
    provider._tool_definitions = registered_tools # Allow access for assertions
    return provider

# --- Test Cases ---

def test_register_run_memory_processing_tool(core_tools, mock_tool_provider):
    """Test that run_memory_processing tool is registered with correct parameters."""
    core_tools._register_run_memory_processing_tool(mock_tool_provider)

    # Check if the tool was registered
    mock_tool_provider.register_tool.assert_called_once()
    args, kwargs = mock_tool_provider.register_tool.call_args

    # Assert on the arguments passed to register_tool
    assert args[0] == "run_memory_processing" # Tool name
    assert isinstance(args[1], str) # Description
    parameters = args[2]
    execution_info = args[3]

    # Check parameters structure
    assert parameters['type'] == 'object'
    assert 'process_one_chunk' in parameters['properties']
    assert parameters['properties']['process_one_chunk']['type'] == 'boolean'
    assert parameters['properties']['process_one_chunk']['default'] is True
    assert 'max_chunks' in parameters['properties']
    assert parameters['properties']['max_chunks']['type'] == 'integer'

    # Check the NEW generation_mechanism parameter
    assert 'generation_mechanism' in parameters['properties']
    gen_mech_param = parameters['properties']['generation_mechanism']
    assert gen_mech_param['type'] == 'string'
    assert gen_mech_param['description'] # Check description exists
    assert gen_mech_param['enum'] == ["self_query", "curated"]
    assert gen_mech_param['default'] == "self_query"

    # Check execution info
    assert execution_info['type'] == 'action_request'
    assert execution_info['target_module'] == 'AgentLoop'
    assert execution_info['action_type'] == 'trigger_memory_processing'

# TODO: Add tests for write_to_notebook, create_notebook, read_from_notebook
# Example test structure for a direct_call tool
# def test_write_to_notebook_impl_success(core_tools, mock_element):
#     # ... setup mocks for Container, TextElement, TextStateComponent ...
#     result = core_tools._write_notebook_impl(content="Test content", notebook_id="test_nb")
#     # ... assert on mocks and result ...

# Add more tests as needed... 