import pytest
import time
from unittest.mock import MagicMock, call
from typing import Optional, Dict, Any, List, Callable # Added List, Callable

try:
    # Class to test
    from elements.elements.components.space.container_component import ContainerComponent
    # Dependencies
    from elements.elements.base import BaseElement, MountType, Component # Added Component

except ImportError as e:
    pytest.skip(f"Skipping ContainerComponent tests due to import error: {e}", allow_module_level=True)

# --- Mocks and Fixtures ---

class MockElement(BaseElement):
    """Mock BaseElement for testing mounting."""
    def __init__(self, element_id="mock_el", name="MockElement"):
        self.id = element_id
        self.name = name
        self._components: Dict[str, Component] = {} # Added type hint
        self._parent_id: Optional[str] = None
        self._mount_type: Optional[MountType] = None
        # Mock methods called by ContainerComponent
        self._set_parent = MagicMock()
        self._clear_parent = MagicMock()
        self.get_parent_info = MagicMock(return_value=None) # Default: no parent
        # Keep simple get_component mock unless needed for complex interactions
        self.get_component = MagicMock(return_value=None) 

    def add_component(self, component: Component): # Basic add for testing
        # Use component type if available, otherwise class name as key fallback
        key = getattr(component, 'COMPONENT_TYPE', component.__class__.__name__)
        self._components[key] = component
        component.element = self

    def get_component_by_type(self, comp_type: str) -> Optional[Component]: # Basic get for testing
         return self._components.get(comp_type)

@pytest.fixture
def host_element():
    """The element hosting the ContainerComponent."""
    element = MockElement(element_id="host_element", name="Host")
    # Override the simple get_component mock for the host
    element.get_component = lambda c_type_or_class: element._components.get(getattr(c_type_or_class, 'COMPONENT_TYPE', None) if isinstance(c_type_or_class, type) else c_type_or_class)
    return element

@pytest.fixture
def container_component(host_element):
    """Instance of the component under test."""
    component = ContainerComponent(element=host_element)
    # Patch the internal event recording methods as they are placeholders
    component._record_mount_event = MagicMock()
    component._record_unmount_event = MagicMock()
    # Manually initialize/enable
    component._initialize() # Use internal initialize
    component._is_enabled = True
    host_element.add_component(component) # Add to host
    return component

@pytest.fixture
def element_to_mount():
    """A standard element to be mounted."""
    return MockElement(element_id="child_element", name="Child")

@pytest.fixture
def mock_mount_listener():
    # Define the expected signature for the mock
    listener: Callable[[str, BaseElement], None] = MagicMock()
    return listener

@pytest.fixture
def mock_unmount_listener():
    listener: Callable[[str, BaseElement], None] = MagicMock()
    return listener


# --- Test Cases ---

def test_initialization(container_component, host_element):
    """Test initial state."""
    assert container_component.element == host_element
    assert not container_component._state["mounted_elements"]
    assert not container_component._mount_listeners
    assert not container_component._unmount_listeners

# --- Mounting Tests ---

def test_mount_element_success_defaults(container_component, host_element, element_to_mount):
    """Test successful mount using default mount_id and type."""
    success = container_component.mount_element(element_to_mount)

    assert success is True
    mount_id = element_to_mount.id # Default mount_id is element_id
    assert mount_id in container_component._state["mounted_elements"]
    mount_info = container_component._state["mounted_elements"][mount_id]
    assert mount_info["element"] == element_to_mount
    assert mount_info["mount_type"] == MountType.INCLUSION # Default type
    assert "mount_time" in mount_info

    # Check interactions with child element
    element_to_mount._set_parent.assert_called_once_with(host_element.id, MountType.INCLUSION)
    container_component._record_mount_event.assert_called_once_with(element_to_mount, mount_id, MountType.INCLUSION)

def test_mount_element_success_explicit_id_type(container_component, host_element, element_to_mount):
    """Test successful mount using explicit mount_id and type."""
    mount_id = "custom_mount_point"
    mount_type = MountType.UPLINK
    success = container_component.mount_element(element_to_mount, mount_id=mount_id, mount_type=mount_type)

    assert success is True
    assert mount_id in container_component._state["mounted_elements"]
    assert element_to_mount.id not in container_component._state["mounted_elements"] # Ensure default wasn't used
    mount_info = container_component._state["mounted_elements"][mount_id]
    assert mount_info["element"] == element_to_mount
    assert mount_info["mount_type"] == mount_type

    element_to_mount._set_parent.assert_called_once_with(host_element.id, mount_type)
    container_component._record_mount_event.assert_called_once_with(element_to_mount, mount_id, mount_type)

def test_mount_element_fail_id_in_use(container_component, element_to_mount):
    """Test mount fails if mount_id is already used."""
    mount_id = "duplicate_mount"
    container_component.mount_element(element_to_mount, mount_id=mount_id) # Mount first time
    element_to_mount._set_parent.reset_mock() # Reset mock for second call check

    another_element = MockElement(element_id="another_el")
    success = container_component.mount_element(another_element, mount_id=mount_id)

    assert success is False
    assert len(container_component._state["mounted_elements"]) == 1 # Should still only have the first one
    another_element._set_parent.assert_not_called()

def test_mount_element_fail_circular_self(container_component, host_element):
    """Test mount fails if attempting to mount the host element into itself."""
    # Temporarily override _would_create_circular_mount for simplicity if needed,
    # or rely on its internal check against self.element.id
    success = container_component.mount_element(host_element)
    assert success is False
    host_element._set_parent.assert_not_called()

def test_mount_element_fail_circular_parent(container_component, host_element, element_to_mount):
    """Test mount fails if attempting to mount the host's parent."""
    # Make the element_to_mount behave as if it's the host's parent
    host_element.get_parent_info.return_value = {"parent_id": element_to_mount.id, "mount_type": MountType.INCLUSION}
    # The _would_create_circular_mount checks the host's parent hierarchy
    
    success = container_component.mount_element(element_to_mount)
    assert success is False
    element_to_mount._set_parent.assert_not_called()
    # Reset host parent info if fixture is reused
    host_element.get_parent_info.return_value = None

# --- Unmounting Tests ---

def test_unmount_element_success(container_component, element_to_mount):
    """Test successful unmounting."""
    mount_id = element_to_mount.id
    container_component.mount_element(element_to_mount) # Mount first
    assert mount_id in container_component._state["mounted_elements"]

    success = container_component.unmount_element(mount_id)

    assert success is True
    assert mount_id not in container_component._state["mounted_elements"]
    element_to_mount._clear_parent.assert_called_once()
    container_component._record_unmount_event.assert_called_once_with(element_to_mount, mount_id, MountType.INCLUSION)

def test_unmount_element_not_found(container_component):
    """Test unmounting a non-existent mount_id."""
    success = container_component.unmount_element("non_existent_mount")
    assert success is False

# --- Retrieval Tests ---

def test_get_mounted_element(container_component, element_to_mount):
    """Test retrieving a specific mounted element."""
    mount_id = "test_mount"
    container_component.mount_element(element_to_mount, mount_id=mount_id)
    
    retrieved = container_component.get_mounted_element(mount_id)
    assert retrieved == element_to_mount
    
    not_found = container_component.get_mounted_element("wrong_id")
    assert not_found is None

def test_get_mounted_elements(container_component, element_to_mount):
    """Test retrieving all mounted elements."""
    mount_id1 = "mount1"
    el1 = element_to_mount
    mount_id2 = "mount2"
    el2 = MockElement(element_id="child2")
    
    assert not container_component.get_mounted_elements() # Initially empty
    
    container_component.mount_element(el1, mount_id=mount_id1)
    container_component.mount_element(el2, mount_id=mount_id2)
    
    all_mounted = container_component.get_mounted_elements()
    assert len(all_mounted) == 2
    assert all_mounted[mount_id1] == el1
    assert all_mounted[mount_id2] == el2

def test_get_mounted_elements_info(container_component, element_to_mount):
    """Test retrieving detailed info about mounted elements."""
    mount_id = "info_mount"
    mount_type = MountType.UPLINK
    container_component.mount_element(element_to_mount, mount_id=mount_id, mount_type=mount_type)
    
    info = container_component.get_mounted_elements_info()
    assert len(info) == 1
    assert mount_id in info
    mount_info = info[mount_id]
    assert mount_info["element_id"] == element_to_mount.id
    assert mount_info["element_name"] == element_to_mount.name
    assert mount_info["element_type"] == element_to_mount.__class__.__name__ # MockElement
    assert mount_info["mount_type"] == mount_type.value # Should be the enum value ('uplink')
    assert "mount_time" in mount_info

# --- Listener Tests ---

def test_add_remove_listeners(container_component, mock_mount_listener, mock_unmount_listener):
    """Test adding and removing listeners."""
    assert not container_component._mount_listeners
    assert not container_component._unmount_listeners

    container_component.add_mount_listener(mock_mount_listener)
    container_component.add_unmount_listener(mock_unmount_listener)
    assert mock_mount_listener in container_component._mount_listeners
    assert mock_unmount_listener in container_component._unmount_listeners
    
    # Add again (should not duplicate)
    container_component.add_mount_listener(mock_mount_listener)
    assert len(container_component._mount_listeners) == 1
    
    container_component.remove_mount_listener(mock_mount_listener)
    container_component.remove_unmount_listener(mock_unmount_listener)
    assert mock_mount_listener not in container_component._mount_listeners
    assert mock_unmount_listener not in container_component._unmount_listeners
    
    # Remove again (should not raise error)
    container_component.remove_mount_listener(mock_mount_listener)

def test_mount_notify_listeners(container_component, element_to_mount, mock_mount_listener):
    """Test mount listeners are notified AFTER successful mount."""
    mount_id = "notify_mount"
    container_component.add_mount_listener(mock_mount_listener)
    
    success = container_component.mount_element(element_to_mount, mount_id=mount_id)
    
    assert success is True
    mock_mount_listener.assert_called_once_with(mount_id, element_to_mount)

def test_unmount_notify_listeners(container_component, element_to_mount, mock_unmount_listener):
    """Test unmount listeners are notified BEFORE successful unmount."""
    mount_id = "notify_unmount"
    container_component.mount_element(element_to_mount, mount_id=mount_id) # Mount first
    
    container_component.add_unmount_listener(mock_unmount_listener)
    success = container_component.unmount_element(mount_id)
    
    assert success is True
    mock_unmount_listener.assert_called_once_with(mount_id, element_to_mount)

def test_listener_error_handling(container_component, element_to_mount, caplog):
    """Test that an error in one listener doesn't stop others."""
    mount_id = "listener_error"
    bad_listener = MagicMock(side_effect=ValueError("Listener Error!"))
    good_listener = MagicMock()
    
    container_component.add_mount_listener(bad_listener)
    container_component.add_mount_listener(good_listener)
    
    with caplog.at_level(logging.ERROR):
        container_component.mount_element(element_to_mount, mount_id=mount_id)
        
    assert "Error calling mount listener" in caplog.text
    assert "Listener Error!" in caplog.text
    bad_listener.assert_called_once_with(mount_id, element_to_mount)
    good_listener.assert_called_once_with(mount_id, element_to_mount) # Good one should still run 