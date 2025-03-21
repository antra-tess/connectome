"""
Widget API

Provides a set of reusable UI components (widgets) for rendering elements.
These widgets extend the basic Rendering API with higher-level,
optionally stateful primitives.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
import json

from .api import (
    RenderingOptions, 
    RenderingResult, 
    RenderingMetadata,
    RenderingImportance,
    RenderingFormat,
    CompressionHint
)

from .delegates import ElementDelegate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Widget(ABC):
    """
    Base class for all widgets.
    
    Widgets are reusable UI components that can be used to render
    complex element states in a consistent manner.
    """
    
    def __init__(self, element_id: Optional[str] = None, element_type: Optional[str] = None):
        """
        Initialize the widget.
        
        Args:
            element_id: Optional ID of the associated element
            element_type: Optional type of the associated element
        """
        self.element_id = element_id
        self.element_type = element_type
    
    @abstractmethod
    def render(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the widget with the given state.
        
        Args:
            state: State data for the widget
            options: Rendering options
            
        Returns:
            Rendering result
        """
        pass
    
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
            element_id=self.element_id or "widget",
            element_type=self.element_type or self.__class__.__name__,
            importance=importance,
            format=format,
            compression_hint=compression_hint,
            related_elements=related_elements or [],
            attributes=attributes or {}
        )


class TextWidget(Widget):
    """
    Simple text widget for displaying formatted text.
    """
    
    def __init__(self, text: str = "", element_id: Optional[str] = None,
                format: RenderingFormat = RenderingFormat.TEXT):
        """
        Initialize the text widget.
        
        Args:
            text: Initial text content
            element_id: Optional element ID
            format: Text format
        """
        super().__init__(element_id, "TextWidget")
        self.text = text
        self.format = format
    
    def render(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the text widget.
        
        Args:
            state: State data (may contain a "text" field to override initial text)
            options: Rendering options
            
        Returns:
            Rendering result
        """
        # Use text from state if available, otherwise use initial text
        text = state.get("text", self.text)
        
        metadata = self.create_metadata(
            format=options.format or self.format,
            importance=RenderingImportance.MEDIUM,
            compression_hint=CompressionHint.TRUNCATE
        )
        
        return RenderingResult(
            content=text,
            metadata=metadata
        )


class CardWidget(Widget):
    """
    Card widget for displaying titled content with optional sections.
    """
    
    def __init__(self, title: str = "", content: str = "", 
                element_id: Optional[str] = None,
                importance: RenderingImportance = RenderingImportance.MEDIUM):
        """
        Initialize the card widget.
        
        Args:
            title: Card title
            content: Card content
            element_id: Optional element ID
            importance: Importance level
        """
        super().__init__(element_id, "CardWidget")
        self.title = title
        self.content = content
        self.importance = importance
    
    def render(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the card widget.
        
        Args:
            state: State data (may contain "title" and "content" fields)
            options: Rendering options
            
        Returns:
            Rendering result
        """
        title = state.get("title", self.title)
        content = state.get("content", self.content)
        
        if options.format == RenderingFormat.MARKDOWN:
            rendered = f"## {title}\n\n{content}"
        else:
            rendered = f"{title}\n\n{content}"
        
        metadata = self.create_metadata(
            format=options.format,
            importance=state.get("importance", self.importance),
            compression_hint=CompressionHint.SUMMARIZE
        )
        
        result = RenderingResult(
            content=rendered,
            metadata=metadata
        )
        
        # Add sections if present in state
        if "sections" in state and isinstance(state["sections"], dict):
            for name, section_content in state["sections"].items():
                section_metadata = self.create_metadata(
                    format=options.format,
                    importance=RenderingImportance.LOW,
                    compression_hint=CompressionHint.OMIT
                )
                
                section_result = RenderingResult(
                    content=section_content,
                    metadata=section_metadata
                )
                
                result.with_section(name, section_result)
        
        return result


class ListWidget(Widget):
    """
    List widget for displaying ordered or unordered lists.
    """
    
    def __init__(self, items: List[str] = None, ordered: bool = False,
                element_id: Optional[str] = None,
                importance: RenderingImportance = RenderingImportance.MEDIUM):
        """
        Initialize the list widget.
        
        Args:
            items: List items
            ordered: Whether the list is ordered
            element_id: Optional element ID
            importance: Importance level
        """
        super().__init__(element_id, "ListWidget")
        self.items = items or []
        self.ordered = ordered
        self.importance = importance
    
    def render(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the list widget.
        
        Args:
            state: State data (may contain "items" and "ordered" fields)
            options: Rendering options
            
        Returns:
            Rendering result
        """
        items = state.get("items", self.items)
        ordered = state.get("ordered", self.ordered)
        
        if not items:
            rendered = "(empty list)"
        else:
            if options.format == RenderingFormat.MARKDOWN:
                if ordered:
                    rendered = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
                else:
                    rendered = "\n".join(f"- {item}" for item in items)
            else:
                if ordered:
                    rendered = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
                else:
                    rendered = "\n".join(f"* {item}" for item in items)
        
        metadata = self.create_metadata(
            format=options.format,
            importance=state.get("importance", self.importance),
            compression_hint=CompressionHint.TRUNCATE,
            attributes={"item_count": len(items)}
        )
        
        return RenderingResult(
            content=rendered,
            metadata=metadata
        )


class TableWidget(Widget):
    """
    Table widget for displaying tabular data.
    """
    
    def __init__(self, headers: List[str] = None, rows: List[List[Any]] = None,
                element_id: Optional[str] = None,
                importance: RenderingImportance = RenderingImportance.HIGH):
        """
        Initialize the table widget.
        
        Args:
            headers: Column headers
            rows: Data rows
            element_id: Optional element ID
            importance: Importance level
        """
        super().__init__(element_id, "TableWidget")
        self.headers = headers or []
        self.rows = rows or []
        self.importance = importance
    
    def render(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the table widget.
        
        Args:
            state: State data (may contain "headers" and "rows" fields)
            options: Rendering options
            
        Returns:
            Rendering result
        """
        headers = state.get("headers", self.headers)
        rows = state.get("rows", self.rows)
        
        if not headers or not rows:
            rendered = "(empty table)"
        else:
            if options.format == RenderingFormat.MARKDOWN:
                # Create header row
                header_row = "| " + " | ".join(str(h) for h in headers) + " |"
                
                # Create separator row
                separator = "| " + " | ".join("---" for _ in headers) + " |"
                
                # Create data rows
                data_rows = []
                for row in rows:
                    # Ensure row has the same number of columns as headers
                    padded_row = row + [""] * (len(headers) - len(row))
                    data_rows.append("| " + " | ".join(str(cell) for cell in padded_row[:len(headers)]) + " |")
                
                # Combine all rows
                rendered = "\n".join([header_row, separator] + data_rows)
            else:
                # Simple text table
                # Calculate column widths
                col_widths = [max(len(str(h)), max([len(str(row[i])) if i < len(row) else 0 for row in rows]) if rows else 0) 
                             for i, h in enumerate(headers)]
                
                # Create header row
                header_row = "  ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
                
                # Create separator
                separator = "  ".join("-" * w for w in col_widths)
                
                # Create data rows
                data_rows = []
                for row in rows:
                    # Ensure row has the same number of columns as headers
                    padded_row = row + [""] * (len(headers) - len(row))
                    data_rows.append("  ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(padded_row[:len(headers)])))
                
                # Combine all rows
                rendered = "\n".join([header_row, separator] + data_rows)
        
        metadata = self.create_metadata(
            format=options.format,
            importance=state.get("importance", self.importance),
            compression_hint=CompressionHint.SUMMARIZE,
            attributes={
                "row_count": len(rows),
                "column_count": len(headers)
            }
        )
        
        return RenderingResult(
            content=rendered,
            metadata=metadata
        )


class StatusWidget(Widget):
    """
    Status widget for displaying the status of an element.
    """
    
    def __init__(self, status: str = "unknown", details: str = "",
                element_id: Optional[str] = None,
                importance: RenderingImportance = RenderingImportance.HIGH):
        """
        Initialize the status widget.
        
        Args:
            status: Status value
            details: Additional details
            element_id: Optional element ID
            importance: Importance level
        """
        super().__init__(element_id, "StatusWidget")
        self.status = status
        self.details = details
        self.importance = importance
    
    def render(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the status widget.
        
        Args:
            state: State data (may contain "status" and "details" fields)
            options: Rendering options
            
        Returns:
            Rendering result
        """
        status = state.get("status", self.status)
        details = state.get("details", self.details)
        
        if options.format == RenderingFormat.MARKDOWN:
            if details:
                rendered = f"**Status**: {status} - {details}"
            else:
                rendered = f"**Status**: {status}"
        else:
            if details:
                rendered = f"Status: {status} - {details}"
            else:
                rendered = f"Status: {status}"
        
        # Status is usually important
        metadata = self.create_metadata(
            format=options.format,
            importance=state.get("importance", self.importance),
            compression_hint=CompressionHint.NONE,
            attributes={"status": status}
        )
        
        return RenderingResult(
            content=rendered,
            metadata=metadata
        )


class CodeWidget(Widget):
    """
    Code widget for displaying formatted code snippets.
    """
    
    def __init__(self, code: str = "", language: str = "",
                element_id: Optional[str] = None,
                importance: RenderingImportance = RenderingImportance.HIGH):
        """
        Initialize the code widget.
        
        Args:
            code: Code content
            language: Programming language
            element_id: Optional element ID
            importance: Importance level
        """
        super().__init__(element_id, "CodeWidget")
        self.code = code
        self.language = language
        self.importance = importance
    
    def render(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the code widget.
        
        Args:
            state: State data (may contain "code" and "language" fields)
            options: Rendering options
            
        Returns:
            Rendering result
        """
        code = state.get("code", self.code)
        language = state.get("language", self.language)
        
        if options.format == RenderingFormat.MARKDOWN:
            if language:
                rendered = f"```{language}\n{code}\n```"
            else:
                rendered = f"```\n{code}\n```"
        else:
            lines = code.split("\n")
            rendered = "\n".join([f"    {line}" for line in lines])
        
        metadata = self.create_metadata(
            format=options.format,
            importance=state.get("importance", self.importance),
            compression_hint=CompressionHint.NONE if len(code) < 500 else CompressionHint.TRUNCATE,
            attributes={"language": language, "line_count": code.count("\n") + 1}
        )
        
        return RenderingResult(
            content=rendered,
            metadata=metadata
        )


class CompoundWidget(Widget):
    """
    Compound widget for combining multiple widgets.
    """
    
    def __init__(self, widgets: Dict[str, Widget] = None, 
                layout: str = "vertical",
                element_id: Optional[str] = None):
        """
        Initialize the compound widget.
        
        Args:
            widgets: Dictionary of named widgets
            layout: Layout direction ("vertical" or "horizontal")
            element_id: Optional element ID
        """
        super().__init__(element_id, "CompoundWidget")
        self.widgets = widgets or {}
        self.layout = layout
    
    def add_widget(self, name: str, widget: Widget) -> None:
        """
        Add a widget to the compound.
        
        Args:
            name: Widget name
            widget: Widget instance
        """
        self.widgets[name] = widget
    
    def render(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the compound widget.
        
        Args:
            state: State data (may contain named states for sub-widgets)
            options: Rendering options
            
        Returns:
            Rendering result
        """
        sections = {}
        contents = []
        
        # Render each sub-widget
        for name, widget in self.widgets.items():
            # Get state for this widget if available
            widget_state = state.get(name, {})
            
            # Render the widget
            result = widget.render(widget_state, options)
            
            # Add rendered content
            contents.append(result.content)
            
            # Add as a section
            sections[name] = result
        
        # Combine rendered content based on layout
        if self.layout == "horizontal":
            separator = " | "
        else:  # vertical
            separator = "\n\n"
            
        combined_content = separator.join(contents)
        
        # Determine overall importance based on sub-widgets
        importances = [section.metadata.importance for section in sections.values()]
        highest_importance = max(importances, key=lambda i: i.value) if importances else RenderingImportance.MEDIUM
        
        metadata = self.create_metadata(
            format=options.format,
            importance=highest_importance,
            compression_hint=CompressionHint.SUMMARIZE,
            attributes={"widget_count": len(self.widgets), "layout": self.layout}
        )
        
        result = RenderingResult(
            content=combined_content,
            metadata=metadata,
            sections=sections
        )
        
        return result


class WidgetDelegate(ElementDelegate):
    """
    Element delegate that uses a widget to render element state.
    """
    
    def __init__(self, element=None, widget: Optional[Widget] = None):
        """
        Initialize the widget delegate.
        
        Args:
            element: Associated element
            widget: Widget to use for rendering
        """
        super().__init__(element)
        self.widget = widget
    
    def render(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the element state using the widget.
        
        Args:
            state: Current state of the element
            options: Rendering options
            
        Returns:
            Rendering result
        """
        if self.widget:
            return self.widget.render(state, options)
        else:
            # Create a default card widget if none is specified
            widget = CardWidget(
                title=f"{self.get_element_type()}: {self.get_element_id()}",
                element_id=self.get_element_id()
            )
            return widget.render(state, options) 