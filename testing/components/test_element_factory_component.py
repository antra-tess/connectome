import pytest
from unittest.mock import MagicMock, patch, call
import logging

# Class imports
from elements.elements.factory import ElementFactory
from elements.elements.base import BaseElement
from elements.elements.components.element_factory_component import ElementFactoryComponent
from elements.elements.base_component import Component # Import base component

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_central_factory():
    """Fixture for a mocked central ElementFactory."""
    mock = MagicMock(spec=ElementFactory)
    # Mock the create_element method
    mock.create_element = MagicMock()
    return mock

@pytest.fixture
def mock_created_element():
    """Fixture for a mocked element returned by the factory."""
    element = MagicMock(spec=BaseElement)
    element.id = "created_element_id_1"
    element.name = "PrefabName" # Default name often comes from prefab
    element.description = "Prefab description"
    # Mock components if needed for other tests, but not strictly required here
    element.components = []
    element.get_component_by_type = MagicMock(return_value=None)
    return element

@pytest.fixture
def factory_component(mock_central_factory):
    """Fixture for an ElementFactoryComponent instance with a mocked central factory."""
    # Pass the mocked factory during initialization
    comp = ElementFactoryComponent(element=None, central_factory=mock_central_factory)
    comp._initialize() # Although simple, call lifecycle methods
    comp._enable()
    return comp

# --- Test Cases ---

def test_factory_component_initialization_success(factory_component, mock_central_factory):
    """Test successful initialization with a provided central factory."""
    assert factory_component._is_initialized
    assert factory_component._is_enabled
    assert factory_component.COMPONENT_TYPE == "element_factory"
    assert factory_component.central_factory == mock_central_factory
    assert not factory_component.DEPENDENCIES

@patch('elements.elements.factory.ElementFactory') # Patch the class itself
def test_factory_component_initialization_no_factory(MockElementFactoryClass):
    """Test initialization warns and creates a default factory if none provided."""
    mock_instance = MockElementFactoryClass.return_value # The instance created by the constructor

    with patch('logging.Logger.warning') as mock_log_warning:
        comp = ElementFactoryComponent(element=None, central_factory=None)
        comp._initialize()

    mock_log_warning.assert_called_once_with(
        "Central ElementFactory not provided to ElementFactoryComponent. Creating a default one. This might lead to inconsistencies."
    )
    MockElementFactoryClass.assert_called_once() # Check default constructor was called
    assert comp.central_factory == mock_instance

def test_central_factory_property(factory_component, mock_central_factory):
    """Test the central_factory property getter."""
    assert factory_component.central_factory == mock_central_factory

# Test create_element method
def test_create_element_success_no_overrides(factory_component, mock_central_factory, mock_created_element):
    """Test successful element creation using the central factory without overrides."""
    prefab = "test_prefab"
    el_id = "test_id_1"
    init_state = {"SomeComponent": {"config": "value"}}

    # Configure the mock factory to return our mock element
    mock_central_factory.create_element.return_value = mock_created_element

    created_el = factory_component.create_element(
        prefab_name=prefab,
        element_id=el_id,
        initial_state=init_state
    )

    # Assert factory was called correctly
    mock_central_factory.create_element.assert_called_once_with(
        element_id=el_id,
        prefab_name=prefab,
        initial_state=init_state
    )

    # Assert the returned element is the one from the factory
    assert created_el == mock_created_element
    # Assert overrides were NOT applied
    assert created_el.name == "PrefabName" # Assuming default name wasn't changed
    assert created_el.description == "Prefab description"

def test_create_element_success_with_overrides(factory_component, mock_central_factory, mock_created_element):
    """Test successful element creation with name and description overrides."""
    prefab = "test_prefab_override"
    new_name = "My Custom Name"
    new_desc = "My Custom Description"

    mock_central_factory.create_element.return_value = mock_created_element

    with patch('logging.Logger.debug') as mock_log_debug:
        created_el = factory_component.create_element(
            prefab_name=prefab,
            element_id=None, # Let factory handle ID generation (or pass None)
            initial_state=None,
            name=new_name,
            description=new_desc
        )

     # Assert factory was called (id and state are None)
    mock_central_factory.create_element.assert_called_once_with(
        element_id=None,
        prefab_name=prefab,
        initial_state=None
    )

    assert created_el == mock_created_element
    # Assert overrides WERE applied
    assert created_el.name == new_name
    assert created_el.description == new_desc

    # Check debug logs for overrides
    expected_calls = [
        call(f"Overrode element {mock_created_element.id} name to '{new_name}'"),
        call(f"Overrode element {mock_created_element.id} description")
    ]
    mock_log_debug.assert_has_calls(expected_calls, any_order=True)


def test_create_element_factory_returns_none(factory_component, mock_central_factory):
    """Test create_element when the central factory fails and returns None."""
    prefab = "failing_prefab"

    # Configure mock factory to return None
    mock_central_factory.create_element.return_value = None

    # Suppress expected info log, check for no error logs
    with patch('logging.Logger.info'), patch('logging.Logger.error') as mock_log_error:
        created_el = factory_component.create_element(prefab_name=prefab)

    assert created_el is None
    mock_central_factory.create_element.assert_called_once_with(
        element_id=None, prefab_name=prefab, initial_state=None
    )
    mock_log_error.assert_not_called() # Error should be logged by factory, not component

def test_create_element_factory_raises_exception(factory_component, mock_central_factory):
    """Test create_element when the central factory raises an exception."""
    prefab = "exception_prefab"
    test_exception = ValueError("Prefab definition error")

    # Configure mock factory to raise an exception
    mock_central_factory.create_element.side_effect = test_exception

    with patch('logging.Logger.error') as mock_log_error:
        created_el = factory_component.create_element(prefab_name=prefab)

    assert created_el is None
    mock_central_factory.create_element.assert_called_once_with(
        element_id=None, prefab_name=prefab, initial_state=None
    )
    # Assert that the component caught the exception and logged an error
    mock_log_error.assert_called_once()
    args, kwargs = mock_log_error.call_args
    assert f"Error calling central factory to create prefab '{prefab}'" in args[0]
    assert kwargs.get('exc_info') is True

# Note: Event handling (_on_event) is commented out in the source, so no tests for it yet.
# If uncommented, add tests similar to other components' _on_event tests, ensuring
# it parses event data correctly and calls self.create_element. 