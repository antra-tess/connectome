import pytest
from unittest.mock import MagicMock, patch, call, ANY
from typing import Dict, Any, Optional, List

try:
    # Class to test
    from elements.elements.space import Space
    # Base class and dependencies
    from elements.elements.base import BaseElement, MountType, Component
    from elements.elements.components.space import ContainerComponent, TimelineComponent

except ImportError as e:
    pytest.skip(f"Skipping Space tests due to import error: {e}", allow_module_level=True)

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_container_component():
    comp = MagicMock(spec=ContainerComponent)
    # Mock methods called by Space
    comp.mount_element = MagicMock(return_value=True)
    comp.unmount_element = MagicMock(return_value=True)
    comp.get_mounted_element = MagicMock(return_value=None)
    comp.get_mounted_elements = MagicMock(return_value={})
    return comp

@pytest.fixture
def mock_timeline_component():
    comp = MagicMock(spec=TimelineComponent)
    # Mock methods called by Space
    comp.add_event_to_timeline = MagicMock(return_value="new_event_id_123") # Return success
    comp.update_state = MagicMock(return_value=True)
    comp.create_timeline_fork = MagicMock(return_value="new_timeline_id_456")
    comp.designate_primary_timeline = MagicMock(return_value=True)
    comp.get_timeline_relationships = MagicMock(return_value={"parent": "p1"})
    comp.get_primary_timeline = MagicMock(return_value="primary_id")
    comp.is_primary_timeline = MagicMock(return_value=True)
    comp.get_timeline_events = MagicMock(return_value=[])
    return comp

@pytest.fixture
def space_instance(monkeypatch, mock_container_component, mock_timeline_component):
    """Provides a Space instance with mocked components added."""

    # Mock BaseElement.add_component to control what's added
    added_components = {}
    def mock_add_component(self, CompCls, *args, **kwargs):
        # Determine the key (COMPONENT_TYPE or class name)
        key = getattr(CompCls, 'COMPONENT_TYPE', CompCls.__name__)
        # Return the pre-made mock instance if it's Container or Timeline
        if key == ContainerComponent.COMPONENT_TYPE:
            instance = mock_container_component
        elif key == TimelineComponent.COMPONENT_TYPE:
            instance = mock_timeline_component
        else: # Should not happen in Space init
            instance = MagicMock(spec=CompCls)

        instance.COMPONENT_TYPE = key # Ensure mock has type
        instance.element = self # Link back element
        added_components[key] = instance # Track added
        # Simulate BaseElement storing components internally for get_component_by_type
        if not hasattr(self, '_components'): self._components = {}
        self._components[key] = instance 
        return instance # Return the mock

    # Mock get_component_by_type to return our mocks
    def mock_get_component_by_type(self, comp_type_str):
         # Use the _components dict populated by mock_add_component
         return self._components.get(comp_type_str)

    monkeypatch.setattr(BaseElement, "add_component", mock_add_component)
    monkeypatch.setattr(BaseElement, "get_component_by_type", mock_get_component_by_type)
    # Patch handle_event on BaseElement so we can check calls to Space's own handler
    monkeypatch.setattr(BaseElement, "handle_event", MagicMock())

    # Instantiate Space - this will now use our mocked add_component
    space = Space(element_id="test_space", name="Test Space", description="A space for testing")
    # Store the mocks on the instance for easy access in tests if needed
    space._mock_container_comp = mock_container_component
    space._mock_timeline_comp = mock_timeline_component
    return space


# --- Mock Child Element for Event Routing ---
class MockChildElement(BaseElement):
     # Class attribute defining handled types
     EVENT_TYPES = {"child_event", "common_event"}

     def __init__(self, element_id="child1", name="Child"):
          super().__init__(element_id, name, "Mock Child") # Call BaseElement init
          # Mock the handle_event method to track calls
          self.handle_event = MagicMock()

class MockOtherChildElement(BaseElement):
     EVENT_TYPES = {"other_event", "common_event"}
     def __init__(self, element_id="child2", name="OtherChild"):
          super().__init__(element_id, name, "Mock Other Child")
          self.handle_event = MagicMock()

class MockNonHandlingChildElement(BaseElement):
     EVENT_TYPES = {"another_type"} # Explicitly doesn't handle common_event
     def __init__(self, element_id="child4", name="NonHandler"):
          super().__init__(element_id, name, "Mock Non Handler")
          self.handle_event = MagicMock()

# --- Test Cases ---

def test_initialization(space_instance, mock_container_component, mock_timeline_component):
    """Test Space.__init__ adds Container and Timeline components."""
    assert space_instance.IS_SPACE is True
    # Check if the mocks were retrieved correctly via get_component_by_type
    assert space_instance.get_component_by_type(ContainerComponent.COMPONENT_TYPE) == mock_container_component
    assert space_instance.get_component_by_type(TimelineComponent.COMPONENT_TYPE) == mock_timeline_component
    # Check internal references (if needed, though above check is better)
    # These might fail depending on exact Space.__init__ implementation details
    # assert space_instance._container == mock_container_component
    # assert space_instance._timeline == mock_timeline_component

# --- Delegation Method Tests ---

def test_mount_element_delegates(space_instance, mock_container_component):
    """Test space.mount_element calls container.mount_element."""
    element = MagicMock(spec=BaseElement)
    mount_id = "m1"
    mount_type = MountType.UPLINK
    result = space_instance.mount_element(element, mount_id=mount_id, mount_type=mount_type)
    assert result is True # Should return container's result
    mock_container_component.mount_element.assert_called_once_with(element, mount_id, mount_type)

def test_unmount_element_delegates(space_instance, mock_container_component):
    """Test space.unmount_element calls container.unmount_element."""
    mount_id = "m1"
    result = space_instance.unmount_element(mount_id)
    assert result is True # Should return container's result
    mock_container_component.unmount_element.assert_called_once_with(mount_id)

def test_get_mounted_element_delegates(space_instance, mock_container_component):
    """Test space.get_mounted_element calls container.get_mounted_element."""
    mount_id = "m1"
    mock_element = MagicMock(spec=BaseElement)
    mock_container_component.get_mounted_element.return_value = mock_element
    result = space_instance.get_mounted_element(mount_id)
    assert result == mock_element
    mock_container_component.get_mounted_element.assert_called_once_with(mount_id)

def test_get_mounted_elements_delegates(space_instance, mock_container_component):
    """Test space.get_mounted_elements calls container.get_mounted_elements."""
    mock_dict = {"m1": MagicMock()}
    mock_container_component.get_mounted_elements.return_value = mock_dict
    result = space_instance.get_mounted_elements()
    assert result == mock_dict
    mock_container_component.get_mounted_elements.assert_called_once()

def test_add_event_to_timeline_delegates(space_instance, mock_timeline_component):
    """Test space.add_event_to_timeline calls timeline.add_event_to_timeline."""
    event = {"type": "test"}
    context = {"id": "t1"}
    result = space_instance.add_event_to_timeline(event, context)
    assert result == "new_event_id_123" # From mock
    mock_timeline_component.add_event_to_timeline.assert_called_once_with(event, context)

def test_create_timeline_fork_delegates(space_instance, mock_timeline_component):
    """Test space.create_timeline_fork calls timeline.create_timeline_fork."""
    result = space_instance.create_timeline_fork("source_t", "source_e", is_primary=True)
    assert result == "new_timeline_id_456" # From mock
    mock_timeline_component.create_timeline_fork.assert_called_once_with("source_t", "source_e", True)

def test_designate_primary_timeline_delegates(space_instance, mock_timeline_component):
    """Test space.designate_primary_timeline calls timeline.designate_primary_timeline."""
    result = space_instance.designate_primary_timeline("t_new_primary")
    assert result is True # From mock
    mock_timeline_component.designate_primary_timeline.assert_called_once_with("t_new_primary")

# Add similar delegation tests for:
# - update_state
# - get_timeline_relationships
# - get_primary_timeline
# - is_primary_timeline
# - get_timeline_events

# --- Event Handling Tests ---

def test_receive_event_success(space_instance, mock_timeline_component):
    """Test receive_event adds to timeline then calls _process_event."""
    event = {"event_type": "test_receive"}
    context = {"timeline_id": "t_recv"}
    # Patch _process_event to check it gets called
    with patch.object(space_instance, '_process_event') as mock_process:
        space_instance.receive_event(event, context)

    mock_timeline_component.add_event_to_timeline.assert_called_once_with(event, context)
    mock_process.assert_called_once_with(event, context)

def test_receive_event_timeline_add_fails(space_instance, mock_timeline_component, caplog):
    """Test receive_event handles failure when adding event to timeline."""
    mock_timeline_component.add_event_to_timeline.return_value = None # Simulate failure
    event = {"event_type": "test_fail"}
    context = {"timeline_id": "t_fail"}
    with patch.object(space_instance, '_process_event') as mock_process:
         with caplog.at_level(logging.ERROR):
            space_instance.receive_event(event, context)

    mock_timeline_component.add_event_to_timeline.assert_called_once_with(event, context)
    assert "Failed to add event to timeline" in caplog.text
    mock_process.assert_not_called() # Should not process if add fails

def test_process_event_routing(space_instance, mock_container_component):
    """Test _process_event calls space's handle_event and routes to relevant children."""
    event_type = "common_event"
    event = {"event_type": event_type}
    context = {"timeline_id": "t_proc"}

    # Mock mounted elements
    child1 = MockChildElement(element_id="c1") # Handles "common_event"
    child2 = MockOtherChildElement(element_id="c2") # Handles "common_event"
    child3 = MockChildElement(element_id="c3") # Handles "common_event" but different class?
    child4 = MockNonHandlingChildElement(element_id="c4") # No EVENT_TYPES defined or doesn't handle
    
    mock_container_component.get_mounted_elements.return_value = {
        "m1": child1, "m2": child2, "m3": child3, "m4": child4
    }

    # Get the mocked handle_event on the space instance itself
    space_handle_event_mock = space_instance.handle_event

    # Call the method under test
    space_instance._process_event(event, context)

    # Verify space's own handler was called
    space_handle_event_mock.assert_called_once_with(event, context)
    # Verify container was queried
    mock_container_component.get_mounted_elements.assert_called_once()
    # Verify relevant children handlers were called
    child1.handle_event.assert_called_once_with(event, context)
    child2.handle_event.assert_called_once_with(event, context)
    child3.handle_event.assert_called_once_with(event, context)
    # Verify child4 handler was NOT called
    child4.handle_event.assert_not_called()

def test_process_event_no_container(space_instance):
    """Test _process_event works okay if container component is missing."""
    # Simulate container being missing by making get_component_by_type return None for it
    original_get = space_instance.get_component_by_type
    def get_comp_override(comp_type_str):
        if comp_type_str == ContainerComponent.COMPONENT_TYPE:
            return None
        return original_get(comp_type_str)
    space_instance.get_component_by_type = get_comp_override
    
    space_handle_event_mock = space_instance.handle_event # Get mock again
    
    event = {"event_type": "test"}
    context = {"timeline_id": "t_no_cont"}
    
    # Should run without error, just won't route to children
    space_instance._process_event(event, context)

    # Verify space's own handler was still called
    space_handle_event_mock.assert_called_once_with(event, context)
    # Restore original method if fixture is reused elsewhere
    space_instance.get_component_by_type = original_get 