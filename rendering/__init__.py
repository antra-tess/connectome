"""
Rendering System

Provides the core rendering capabilities for translating element states into
renderable context for agents. This package includes the Rendering API,
Element Delegates, and Widget API.
"""

from .api import (
    RenderingOptions,
    RenderingResult,
    RenderingMetadata,
    RenderingImportance,
    RenderingFormat,
    CompressionHint,
    SceneNode,
    RenderingContext,
    RenderingRegistry,
    registry
)

from .delegates import (
    ElementDelegate,
    FunctionDelegate,
    StaticDelegate,
    DefaultDelegate,
    render_element_decorator
)

from .widgets import (
    Widget,
    TextWidget,
    CardWidget,
    ListWidget,
    TableWidget,
    StatusWidget,
    CodeWidget,
    CompoundWidget,
    WidgetDelegate
)

__all__ = [
    # API
    'RenderingOptions',
    'RenderingResult',
    'RenderingMetadata',
    'RenderingImportance',
    'RenderingFormat',
    'CompressionHint',
    'SceneNode',
    'RenderingContext',
    'RenderingRegistry',
    'registry',
    
    # Delegates
    'ElementDelegate',
    'FunctionDelegate',
    'StaticDelegate',
    'DefaultDelegate',
    'render_element_decorator',
    
    # Widgets
    'Widget',
    'TextWidget',
    'CardWidget',
    'ListWidget',
    'TableWidget',
    'StatusWidget',
    'CodeWidget',
    'CompoundWidget',
    'WidgetDelegate'
] 