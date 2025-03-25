"""
Element Delegates

Provides the interfaces and base implementations for Element Delegates, which 
transform element state into rendered text for the agent context.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
import inspect

from .api import (
    RenderingOptions, 
    RenderingResult, 
    RenderingMetadata,
    RenderingImportance,
    RenderingFormat,
    CompressionHint,
    registry
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ElementDelegate(ABC):
    """
    Base class for all Element Delegates.
    
    Element Delegates are responsible for transforming element state into
    renderable content for the agent's context. They provide the bridge
    between the element's internal state and its external representation.
    """
    
    def __init__(self, element=None):
        """
        Initialize the element delegate.
        
        Args:
            element: Optional reference to the associated element
        """
        self.element = element
    
    @abstractmethod
    def render(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the element state into a text representation.
        
        Args:
            state: Current state of the element
            options: Rendering options
            
        Returns:
            Rendering result containing the rendered content and metadata
        """
        pass
    
    def render_exterior(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the element's exterior view (when closed).
        
        Args:
            state: Current state of the element
            options: Rendering options
            
        Returns:
            Rendering result containing the exterior view
        """
        # Default implementation returns a compact representation
        element_id = self.get_element_id()
        element_type = self.get_element_type()
        
        if options.format == RenderingFormat.MARKDOWN:
            content = f"**{element_type}**: {element_id}"
        else:
            content = f"{element_type}: {element_id}"
            
        metadata = self.create_metadata(
            importance=RenderingImportance.LOW,
            format=options.format,
            compression_hint=CompressionHint.OMIT
        )
        
        return RenderingResult(
            content=content,
            metadata=metadata
        )
    
    def render_interior(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the element's interior view (when open).
        
        Args:
            state: Current state of the element
            options: Rendering options
            
        Returns:
            Rendering result containing the interior view
        """
        # Default implementation calls the main render method
        return self.render(state, options)
    
    def get_element_id(self) -> str:
        """Get the ID of the associated element."""
        if self.element and hasattr(self.element, 'id'):
            return self.element.id
        return "unknown_element"
    
    def get_element_type(self) -> str:
        """Get the type of the associated element."""
        if self.element:
            return self.element.__class__.__name__
        return "unknown_type"
    
    def create_metadata(self, 
                       importance: RenderingImportance = RenderingImportance.MEDIUM,
                       format: RenderingFormat = RenderingFormat.TEXT,
                       compression_hint: CompressionHint = CompressionHint.SUMMARIZE,
                       related_elements: List[str] = None,
                       attributes: Dict[str, Any] = None) -> RenderingMetadata:
        """
        Create metadata for the rendering result.
        
        Args:
            importance: Importance level
            format: Format of the content
            compression_hint: Hint for compression
            related_elements: Related element IDs
            attributes: Additional attributes
            
        Returns:
            Rendering metadata
        """
        return RenderingMetadata(
            element_id=self.get_element_id(),
            element_type=self.get_element_type(),
            importance=importance,
            format=format,
            compression_hint=compression_hint,
            related_elements=related_elements or [],
            attributes=attributes or {}
        )
    
    def register(self) -> None:
        """Register this delegate with the global registry."""
        element_type = self.get_element_type()
        
        def renderer_func(state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
            # Check if element is open/closed
            if self.element and hasattr(self.element, 'is_open'):
                if self.element.is_open():
                    return self.render_interior(state, options)
                else:
                    return self.render_exterior(state, options)
            # Fall back to main render method if state not available
            return self.render(state, options)
            
        registry.register_renderer(element_type, renderer_func)
        logger.debug(f"Registered delegate for element type: {element_type}")


class FunctionDelegate:
    """
    Function-based delegate that uses a function to render element state.
    """
    
    def __init__(self, element_type: str, render_func: Callable):
        """
        Initialize the function delegate.
        
        Args:
            element_type: Type of element this delegate handles
            render_func: Function to render the element
        """
        self.element_type = element_type
        self.render_func = render_func
    
    def register(self) -> None:
        """Register this delegate with the global registry."""
        registry.register_renderer(self.element_type, self.render_func)
        logger.debug(f"Registered function delegate for element type: {self.element_type}")


class StaticDelegate(ElementDelegate):
    """
    Delegate that renders static content for an element.
    """
    
    def __init__(self, element=None, static_content: str = "",
                importance: RenderingImportance = RenderingImportance.LOW,
                format: RenderingFormat = RenderingFormat.TEXT):
        """
        Initialize the static delegate.
        
        Args:
            element: Associated element
            static_content: Static content to render
            importance: Importance level for the content
            format: Format of the content
        """
        super().__init__(element)
        self.static_content = static_content
        self.importance = importance
        self.format = format
    
    def render(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the static content.
        
        Args:
            state: Current state (ignored)
            options: Rendering options
            
        Returns:
            Rendering result with static content
        """
        metadata = self.create_metadata(
            importance=self.importance,
            format=self.format,
            compression_hint=CompressionHint.OMIT
        )
        
        return RenderingResult(
            content=self.static_content,
            metadata=metadata
        )


class DefaultDelegate(ElementDelegate):
    """
    Default delegate that renders a generic representation of element state.
    """
    
    def render(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the element state in a generic way.
        
        Args:
            state: Current state of the element
            options: Rendering options
            
        Returns:
            Rendering result with generic representation
        """
        element_id = self.get_element_id()
        element_type = self.get_element_type()
        
        if options.format == RenderingFormat.MARKDOWN:
            content = f"## {element_type}: {element_id}\n\n"
            
            # Add state summary
            content += "### State\n\n"
            for key, value in state.items():
                if isinstance(value, dict) or isinstance(value, list):
                    content += f"- **{key}**: *complex data*\n"
                else:
                    content += f"- **{key}**: {value}\n"
        else:
            content = f"{element_type}: {element_id}\n\n"
            
            # Add state summary
            content += "State:\n"
            for key, value in state.items():
                if isinstance(value, dict) or isinstance(value, list):
                    content += f"- {key}: (complex data)\n"
                else:
                    content += f"- {key}: {value}\n"
        
        metadata = self.create_metadata(
            importance=RenderingImportance.LOW,
            format=options.format,
            compression_hint=CompressionHint.SUMMARIZE
        )
        
        return RenderingResult(
            content=content,
            metadata=metadata
        )
    
    def render_exterior(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render a compact exterior view of the element.
        
        Args:
            state: Current state of the element
            options: Rendering options
            
        Returns:
            Rendering result with compact representation
        """
        element_id = self.get_element_id()
        element_type = self.get_element_type()
        
        # Get a brief summary of state if available
        state_summary = ""
        if state:
            # Try to get a meaningful summary field
            summary_fields = ["name", "description", "status", "type"]
            for field in summary_fields:
                if field in state:
                    state_summary = f" - {state[field]}"
                    break
        
        if options.format == RenderingFormat.MARKDOWN:
            content = f"**{element_type}**: {element_id}{state_summary}"
        else:
            content = f"{element_type}: {element_id}{state_summary}"
        
        metadata = self.create_metadata(
            importance=RenderingImportance.LOW,
            format=options.format,
            compression_hint=CompressionHint.OMIT
        )
        
        return RenderingResult(
            content=content,
            metadata=metadata
        )


def render_element_decorator(element_type: str = None, **kwargs):
    """
    Decorator to register a function as an element renderer.
    
    Args:
        element_type: Type of element this function renders
        **kwargs: Additional options for the renderer
        
    Returns:
        Decorator function
    """
    def decorator(func):
        nonlocal element_type
        
        # If element_type not provided, try to derive from function name
        if not element_type:
            func_name = func.__name__
            if func_name.startswith("render_"):
                element_type = func_name[7:]  # Remove "render_" prefix
            else:
                element_type = func_name
        
        # Create a wrapper function that matches the expected signature
        @registry.register_renderer(element_type)
        def wrapper(state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
            return func(state, options, **kwargs)
            
        return func
        
    return decorator


# Register a default renderer for unknown element types
@render_element_decorator("default")
def render_default(state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
    """Default renderer for unknown element types."""
    delegate = DefaultDelegate()
    return delegate.render(state, options) 