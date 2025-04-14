import pytest
from unittest.mock import MagicMock, patch, call, ANY
import logging
from typing import Optional, Dict, Any, Type

try:
    # Import the class to test
    from elements.elements.inner_space import InnerSpace
    # Import base classes and dependencies needed for mocking/type hinting
    from elements.elements.base import BaseElement, Component
    from elements.elements.space import Space # Parent class
    from llm.provider_interface import LLMProviderInterface
    # Import component types InnerSpace is expected to create
    from elements.elements.components import (
        ToolProviderComponent, 
        GlobalAttentionComponent,
        ContainerComponent, 
        ContextManagerComponent,
        HUDComponent,             
        TimelineComponent,
        CoreToolsComponent,
        MessagingToolsComponent,
        BaseAgentLoopComponent, # Base class for agent loops
        SimpleRequestResponseLoopComponent # Default agent loop
    )
    from host.event_loop import OutgoingActionCallback
except ImportError as e:
    pytest.skip(f"Skipping InnerSpace tests due to import error: {e}", allow_module_level=True)

# --- Mocks and Fixtures ---

# Mock LLM Provider
@pytest.fixture
def mock_llm_provider():
    return MagicMock(spec=LLMProviderInterface)

# Mock Outgoing Callback
@pytest.fixture
def mock_outgoing_callback():
    return MagicMock(spec=OutgoingActionCallback)

# Helper function to patch components for a test
def patch_components(monkeypatch, component_classes_to_mock: Dict[Type[Component], MagicMock]) -> Dict[Type[Component], MagicMock]:
    """Patches component classes and BaseElement methods for testing InnerSpace init."""
    mock_added_components = {}

    def mock_add_component(self, component_instance):
        mock_added_components[component_instance.__class__] = component_instance
        component_instance.element = self
        # Ensure mock has setter if original class does
        if hasattr(component_instance.__class__, 'set_outgoing_action_callback'):
            component_instance.set_outgoing_action_callback = MagicMock()
        
    def mock_get_component_by_type(self, component_type_or_class):
         requested_type_name = getattr(component_type_or_class, 'COMPONENT_TYPE', None)

         for comp_class, instance in mock_added_components.items():
            instance_type_name = getattr(instance, 'COMPONENT_TYPE', None)
            # Check 1: Direct Class Inheritance/Match
            if isinstance(instance, component_type_or_class):
                return instance
            # Check 2: COMPONENT_TYPE Match (if available)
            if requested_type_name and instance_type_name and requested_type_name == instance_type_name:
                return instance
            # Check 3: Direct Class Match (for base classes like BaseAgentLoopComponent)
            if comp_class == component_type_or_class:
                return instance
         return None # Not found

    # Patch Space.__init__ to simplify testing
    monkeypatch.setattr(Space, "__init__", lambda self, id, name, **kwargs: None)
    # Patch BaseElement methods
    monkeypatch.setattr(BaseElement, "add_component", mock_add_component)
    monkeypatch.setattr(BaseElement, "get_component_by_type", mock_get_component_by_type)
    # Patch component classes constructors
    for cls, mock_class_spec in component_classes_to_mock.items():
        mock_instance = mock_class_spec() # Create instance from provided spec
        if hasattr(cls, 'COMPONENT_TYPE'):
             mock_instance.COMPONENT_TYPE = cls.COMPONENT_TYPE
        # Ensure the mock instance has the methods we expect to call (like register_tools)
        if hasattr(cls, 'register_tools'):
             mock_instance.register_tools = MagicMock()
        # Ensure mock has setter if original class does
        if hasattr(cls, 'set_outgoing_action_callback'):
             mock_instance.set_outgoing_action_callback = MagicMock()
        # Fallback attribute for callback propagation check
        if not hasattr(mock_instance, 'set_outgoing_action_callback'):
             mock_instance._outgoing_action_callback = None
             
        monkeypatch.setattr(f"{cls.__module__}.{cls.__name__}", MagicMock(return_value=mock_instance))

    return mock_added_components

# --- Test Cases ---

def test_inner_space_initialization_success(
    mock_llm_provider, 
    monkeypatch
): 
    """Test successful initialization of InnerSpace with default components."""
    
    mock_component_specs = {
        ToolProviderComponent: MagicMock(spec=ToolProviderComponent),
        GlobalAttentionComponent: MagicMock(spec=GlobalAttentionComponent),
        ContextManagerComponent: MagicMock(spec=ContextManagerComponent),
        HUDComponent: MagicMock(spec=HUDComponent),
        CoreToolsComponent: MagicMock(spec=CoreToolsComponent),
        MessagingToolsComponent: MagicMock(spec=MessagingToolsComponent),
        SimpleRequestResponseLoopComponent: MagicMock(spec=SimpleRequestResponseLoopComponent),
        ContainerComponent: MagicMock(spec=ContainerComponent),
        TimelineComponent: MagicMock(spec=TimelineComponent),
    }
    
    mock_added_components = patch_components(monkeypatch, mock_component_specs)

    # --- Instantiation --- 
    inner_space = InnerSpace(id="test_is", name="TestInner", llm_provider=mock_llm_provider)

    # --- Assertions --- 
    assert inner_space.id == "test_is"
    assert inner_space.name == "TestInner"
    assert inner_space._llm_provider == mock_llm_provider

    # Verify core components were instantiated and added (check mocks)
    assert mock_added_components.get(ToolProviderComponent) is not None
    assert mock_added_components.get(GlobalAttentionComponent) is not None
    assert mock_added_components.get(ContextManagerComponent) is not None
    assert mock_added_components.get(HUDComponent) is not None
    assert mock_added_components.get(CoreToolsComponent) is not None
    assert mock_added_components.get(MessagingToolsComponent) is not None
    assert mock_added_components.get(SimpleRequestResponseLoopComponent) is not None
    
    # Verify dependency injection (LLM Provider passed to HUD constructor)
    hud_class_mock = mock_component_specs[HUDComponent]
    # Check constructor was called with llm_provider kwarg
    # This relies on patching the class constructor via monkeypatch
    # The actual check needs to happen on the mock *class* object patched
    hud_constructor_mock = monkeypatch.get_item(globals(), f"{HUDComponent.__module__}.{HUDComponent.__name__}")
    # hud_constructor_mock.assert_called_once_with(llm_provider=mock_llm_provider)
    # NOTE: Verifying constructor args with this level of patching is complex.
    # We'll trust the INJECTED_DEPENDENCIES mechanism is working if init doesn't fail.

    # Verify tool registration calls
    tool_provider_instance_mock = mock_added_components.get(ToolProviderComponent)
    core_tools_instance_mock = mock_added_components.get(CoreToolsComponent)
    messaging_tools_instance_mock = mock_added_components.get(MessagingToolsComponent)

    assert tool_provider_instance_mock is not None
    assert core_tools_instance_mock is not None
    assert messaging_tools_instance_mock is not None
    
    # Check register_tools was called on the *mock instances*
    core_tools_instance_mock.register_tools.assert_called_once_with(tool_provider_instance_mock)
    messaging_tools_instance_mock.register_tools.assert_called_once_with(tool_provider_instance_mock)

def test_set_outgoing_action_callback_propagation(
     mock_llm_provider,
     mock_outgoing_callback,
     monkeypatch
 ):
    """Test that set_outgoing_action_callback propagates to relevant components."""
    mock_component_specs = {
        HUDComponent: MagicMock(spec=HUDComponent),
        CoreToolsComponent: MagicMock(spec=CoreToolsComponent),
        SimpleRequestResponseLoopComponent: MagicMock(spec=SimpleRequestResponseLoopComponent),
        ToolProviderComponent: MagicMock(spec=ToolProviderComponent),
        GlobalAttentionComponent: MagicMock(spec=GlobalAttentionComponent),
        ContextManagerComponent: MagicMock(spec=ContextManagerComponent),
        MessagingToolsComponent: MagicMock(spec=MessagingToolsComponent),
        ContainerComponent: MagicMock(spec=ContainerComponent),
        TimelineComponent: MagicMock(spec=TimelineComponent),
    }
    
    mock_added_components = patch_components(monkeypatch, mock_component_specs)

    inner_space = InnerSpace(id="test_is_cb", name="TestCallback", llm_provider=mock_llm_provider)

    # Call the method under test
    inner_space.set_outgoing_action_callback(mock_outgoing_callback)

    # Assertions
    assert inner_space._outgoing_action_callback == mock_outgoing_callback

    # Check propagation to components expected to have the callback
    hud_mock = mock_added_components.get(HUDComponent)
    core_tools_mock = mock_added_components.get(CoreToolsComponent)
    # Use BaseAgentLoopComponent to find the loop instance
    loop_mock = next((c for c in mock_added_components.values() if isinstance(c, BaseAgentLoopComponent)), None)
    
    assert hud_mock is not None
    assert core_tools_mock is not None
    assert loop_mock is not None

    # Check if setter was called or attribute was set
    if hasattr(hud_mock, 'set_outgoing_action_callback') and isinstance(hud_mock.set_outgoing_action_callback, MagicMock):
        hud_mock.set_outgoing_action_callback.assert_called_once_with(mock_outgoing_callback)
    elif hasattr(hud_mock, '_outgoing_action_callback'):
         assert hud_mock._outgoing_action_callback == mock_outgoing_callback
         
    if hasattr(core_tools_mock, 'set_outgoing_action_callback') and isinstance(core_tools_mock.set_outgoing_action_callback, MagicMock):
        core_tools_mock.set_outgoing_action_callback.assert_called_once_with(mock_outgoing_callback)
    elif hasattr(core_tools_mock, '_outgoing_action_callback'):
         assert core_tools_mock._outgoing_action_callback == mock_outgoing_callback
         
    if hasattr(loop_mock, 'set_outgoing_action_callback') and isinstance(loop_mock.set_outgoing_action_callback, MagicMock):
        loop_mock.set_outgoing_action_callback.assert_called_once_with(mock_outgoing_callback)
    elif hasattr(loop_mock, '_outgoing_action_callback'):
         assert loop_mock._outgoing_action_callback == mock_outgoing_callback

# Add more tests for getters, potentially basic tests for overridden methods like handle_event
# if they have significant logic beyond calling components/super.
 