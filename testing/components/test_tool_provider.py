import pytest
from unittest.mock import MagicMock, call, ANY
import logging

try:
    # Core classes
    from elements.elements.base import BaseElement, Component
    from elements.elements.components.tool_provider_component import ToolProviderComponent, ToolDefinition
    from elements.elements.components.space.container_component import ContainerComponent
except ImportError as e:
    pytest.skip(f"Skipping tests due to import error: {e}", allow_module_level=True)

# --- Mocks and Fixtures ---

# Mock Element to host the ToolProviderComponent
class MockHostElement(BaseElement):
    def __init__(self, element_id="host_element"):
        self.id = element_id
        self._components = {}
        self._mock_container = None # Will hold the mock container

    def add_component(self, component: Component):
        self._components[component.COMPONENT_TYPE] = component
        component.element = self

    def get_component(self, component_type_or_class) -> Optional[Component]:
        # Handle class type for lookup
        if isinstance(component_type_or_class, type):
            component_type = component_type_or_class.COMPONENT_TYPE
        else:
            component_type = component_type_or_class
            
        if component_type == ContainerComponent.COMPONENT_TYPE:
            return self._mock_container
        return self._components.get(component_type)

# Mock Element to be mounted
class MockMountedElement(BaseElement):
     def __init__(self, element_id="mounted_element", name="MountedElement", has_provider=False):
        self.id = element_id
        self.name = name
        self._components = {}
        self._mock_provider = None
        if has_provider:
             # If it has a provider, create a mock one for it
             self._mock_provider = MagicMock(spec=ToolProviderComponent)
             self._mock_provider.get_all_tool_definitions = MagicMock(return_value={})
             self._components[ToolProviderComponent.COMPONENT_TYPE] = self._mock_provider
             
     def get_component(self, component_type_or_class) -> Optional[Component]:
         # Handle class type for lookup
         if isinstance(component_type_or_class, type):
             component_type = component_type_or_class.COMPONENT_TYPE
         else:
             component_type = component_type_or_class
             
         return self._components.get(component_type)

@pytest.fixture
def mock_container():
    container = MagicMock(spec=ContainerComponent)
    container.add_mount_listener = MagicMock()
    container.add_unmount_listener = MagicMock()
    container.remove_mount_listener = MagicMock()
    container.remove_unmount_listener = MagicMock()
    container.get_mounted_elements = MagicMock(return_value={})
    # Store listeners to simulate calling them
    container._mount_listeners = []
    container._unmount_listeners = []
    container.add_mount_listener.side_effect = lambda listener: container._mount_listeners.append(listener)
    container.add_unmount_listener.side_effect = lambda listener: container._unmount_listeners.append(listener)
    return container

@pytest.fixture
def host_element(mock_container):
    element = MockHostElement()
    element._mock_container = mock_container # Link mock container
    return element

@pytest.fixture
def tool_provider(host_element):
    provider = ToolProviderComponent(element=host_element)
    provider.logger = logging.getLogger("TestToolProvider")
    # Manually call initialize since we don't have a full component lifecycle simulation
    initialized = provider._initialize()
    assert initialized is True # Ensure initialization succeeds with mock container
    return provider

# Helper to simulate mounting
def simulate_mount(container_mock: MagicMock, mount_id: str, element_to_mount: BaseElement):
    # Update the container's internal state
    current_elements = container_mock.get_mounted_elements.return_value
    current_elements[mount_id] = element_to_mount
    container_mock.get_mounted_elements.return_value = current_elements
    # Call registered mount listeners
    for listener in container_mock._mount_listeners:
        listener(mount_id, element_to_mount)

# Helper to simulate unmounting
def simulate_unmount(container_mock: MagicMock, mount_id: str, element_to_unmount: BaseElement):
     # Update the container's internal state
    current_elements = container_mock.get_mounted_elements.return_value
    if mount_id in current_elements: del current_elements[mount_id]
    container_mock.get_mounted_elements.return_value = current_elements
    # Call registered unmount listeners
    for listener in container_mock._unmount_listeners:
         listener(mount_id, element_to_unmount)

# --- Test Cases ---

def test_initialization_and_listeners(tool_provider, mock_container):
    """Test initialization registers listeners with the container."""
    assert tool_provider is not None
    assert tool_provider.COMPONENT_TYPE == "tool_provider"
    assert tool_provider._container_comp == mock_container
    mock_container.add_mount_listener.assert_called_once_with(ANY) # Check a listener was added
    mock_container.add_unmount_listener.assert_called_once_with(ANY) # Check a listener was added

def test_cleanup_removes_listeners(tool_provider, mock_container):
     """Test cleanup removes listeners."""
     tool_provider._on_cleanup()
     mock_container.remove_mount_listener.assert_called_once_with(ANY)
     mock_container.remove_unmount_listener.assert_called_once_with(ANY)
     assert tool_provider._container_comp is None # Should clear ref

def test_register_tool_success(tool_provider):
    """Test registering a tool directly."""
    name = "my_tool"
    desc = "Does something cool."
    params = {"type": "object", "properties": {"arg1": {"type": "string"}}}
    exec_info = {"type": "direct_call", "function": "some_function"}
    
    success = tool_provider.register_tool(name, desc, params, exec_info)
    assert success is True
    definition = tool_provider.get_tool_definition(name)
    assert definition is not None
    assert definition.name == name
    assert definition.description == desc
    assert definition.parameters == params
    assert definition.execution_info == exec_info
    assert name in tool_provider.get_all_tool_definitions()

def test_register_tool_overwrite(tool_provider, caplog):
    """Test overwriting an existing tool registration."""
    name = "overwrite_tool"
    tool_provider.register_tool(name, "desc1", {}, {"type": "direct_call"})
    with caplog.at_level(logging.WARNING):
        success = tool_provider.register_tool(name, "desc2", {}, {"type": "action_request"})
    
    assert success is True
    assert f"Tool '{name}' already registered" in caplog.text
    definition = tool_provider.get_tool_definition(name)
    assert definition.description == "desc2"
    assert definition.execution_info["type"] == "action_request"

def test_register_tool_missing_exec_type(tool_provider, caplog):
    """Test registering fails if execution_info lacks 'type'."""
    with caplog.at_level(logging.ERROR):
        success = tool_provider.register_tool("bad_tool", "desc", {}, {"details": "abc"})
    assert success is False
    assert "execution_info must contain a 'type' key" in caplog.text
    assert tool_provider.get_tool_definition("bad_tool") is None

def test_unregister_tool_success(tool_provider):
    """Test unregistering an existing tool."""
    name = "unregister_me"
    tool_provider.register_tool(name, "desc", {}, {"type": "direct_call"})
    assert tool_provider.get_tool_definition(name) is not None
    success = tool_provider.unregister_tool(name)
    assert success is True
    assert tool_provider.get_tool_definition(name) is None
    assert name not in tool_provider.get_all_tool_definitions()

def test_unregister_tool_not_found(tool_provider, caplog):
    """Test unregistering a tool that doesn't exist."""
    with caplog.at_level(logging.WARNING):
        success = tool_provider.unregister_tool("does_not_exist")
    assert success is False
    assert "Tool 'does_not_exist' not found" in caplog.text

def test_get_llm_tool_schemas(tool_provider):
    """Test formatting tool definitions into LLM schemas."""
    tool_provider.register_tool("tool1", "desc1", {"type": "object", "properties": {}}, {"type": "direct_call"})
    tool_provider.register_tool("tool2", "desc2", {"type": "object", "properties": {"p": {"type": "integer"}}}, {"type": "action_request"})
    
    schemas = tool_provider.get_llm_tool_schemas()
    assert len(schemas) == 2
    
    schema1 = next(s for s in schemas if s["function"]["name"] == "tool1")
    schema2 = next(s for s in schemas if s["function"]["name"] == "tool2")
    
    assert schema1["type"] == "function"
    assert schema1["function"]["description"] == "desc1"
    assert schema1["function"]["parameters"] == {"type": "object", "properties": {}}
    
    assert schema2["type"] == "function"
    assert schema2["function"]["name"] == "tool2"
    assert schema2["function"]["parameters"]["properties"]["p"]["type"] == "integer"

# --- Tests for Mounting/Unmounting and Tool Aggregation ---

@pytest.fixture
def mounted_element_with_provider():
    element = MockMountedElement(element_id="mount1", name="ElementWithTools", has_provider=True)
    # Define some tools for its internal provider
    tool_def1 = ToolDefinition("internal_tool", "Does internal stuff", {"type": "object"}, {"type": "direct_call"})
    element._mock_provider.get_all_tool_definitions.return_value = {"internal_tool": tool_def1}
    return element
    
@pytest.fixture
def mounted_element_without_provider():
    return MockMountedElement(element_id="mount2", name="ElementWithoutTools", has_provider=False)

def test_mount_element_with_provider(tool_provider, mock_container, mounted_element_with_provider):
    """Test mounting an element that has its own ToolProvider."""
    mount_id = "slot1"
    simulate_mount(mock_container, mount_id, mounted_element_with_provider)
    
    # Check if the tool was aggregated with the correct prefix
    aggregated_name = f"{mount_id}.internal_tool"
    definition = tool_provider.get_tool_definition(aggregated_name)
    assert definition is not None
    assert definition.name == aggregated_name
    assert definition.description.startswith(f"[{mounted_element_with_provider.name}@{mount_id}]")
    assert definition.parameters == {"type": "object"}
    assert definition.execution_info == {"type": "direct_call"} # Should use original exec info
    # Check internal tracking
    assert mount_id in tool_provider._mounted_element_tools
    assert tool_provider._mounted_element_tools[mount_id][aggregated_name] == "internal_tool"

def test_mount_element_without_provider(tool_provider, mock_container, mounted_element_without_provider):
    """Test mounting an element that does NOT have its own ToolProvider."""
    mount_id = "slot2"
    initial_tool_count = len(tool_provider.get_all_tool_definitions())
    
    simulate_mount(mock_container, mount_id, mounted_element_without_provider)
    
    # No tools should have been added
    assert len(tool_provider.get_all_tool_definitions()) == initial_tool_count
    assert mount_id not in tool_provider._mounted_element_tools

def test_unmount_element_with_provider(tool_provider, mock_container, mounted_element_with_provider):
    """Test unmounting an element removes its aggregated tools."""
    mount_id = "slot1"
    aggregated_name = f"{mount_id}.internal_tool"
    
    # Mount it first
    simulate_mount(mock_container, mount_id, mounted_element_with_provider)
    assert tool_provider.get_tool_definition(aggregated_name) is not None
    assert mount_id in tool_provider._mounted_element_tools

    # Now unmount it
    simulate_unmount(mock_container, mount_id, mounted_element_with_provider)
    
    # Check the tool is gone
    assert tool_provider.get_tool_definition(aggregated_name) is None
    assert mount_id not in tool_provider._mounted_element_tools

def test_register_tool_overwrites_mounted_tool_tracking(tool_provider, mock_container, mounted_element_with_provider, caplog):
    """Test direct registration correctly removes old tracking if it overwrites a mounted tool name."""
    mount_id = "slot1"
    aggregated_name = f"{mount_id}.internal_tool"
    
    # Mount it first
    simulate_mount(mock_container, mount_id, mounted_element_with_provider)
    assert tool_provider.get_tool_definition(aggregated_name) is not None
    assert mount_id in tool_provider._mounted_element_tools
    assert tool_provider._mounted_element_tools[mount_id][aggregated_name] == "internal_tool"

    # Now directly register a tool with the same name
    with caplog.at_level(logging.WARNING):
         success = tool_provider.register_tool(aggregated_name, "New Desc", {}, {"type":"action_request"})
         
    assert success is True
    assert f"Tool '{aggregated_name}' already registered" in caplog.text
    # Check tracking for the mounted element is gone for this name
    assert mount_id not in tool_provider._mounted_element_tools # The mount_id entry should be gone if it's empty
    
    definition = tool_provider.get_tool_definition(aggregated_name)
    assert definition.description == "New Desc"

def test_cleanup_removes_mounted_tools(tool_provider, mock_container, mounted_element_with_provider):
    """Test that _on_cleanup removes tools aggregated from mounted elements."""
    mount_id = "slot1"
    aggregated_name = f"{mount_id}.internal_tool"
    
    # Mount it first
    simulate_mount(mock_container, mount_id, mounted_element_with_provider)
    assert tool_provider.get_tool_definition(aggregated_name) is not None
    assert mount_id in tool_provider._mounted_element_tools

    # Call cleanup
    tool_provider._on_cleanup()
    
    # Check tool is gone and tracking is gone
    assert tool_provider.get_tool_definition(aggregated_name) is None
    assert mount_id not in tool_provider._mounted_element_tools
    # Check listeners were removed
    mock_container.remove_mount_listener.assert_called_once()
    mock_container.remove_unmount_listener.assert_called_once() 