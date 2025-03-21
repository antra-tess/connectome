"""
Rendering API

Provides the core rendering capabilities for translating element states into
renderable context for agents. This API defines the interfaces and base classes
for the rendering system.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RenderingImportance(Enum):
    """Importance levels for rendered content."""
    CRITICAL = 100  # Must never be compressed or removed
    HIGH = 75       # Try to preserve in full
    MEDIUM = 50     # Default importance
    LOW = 25        # Can be heavily compressed or summarized
    MINIMAL = 10    # Can be removed if needed


class RenderingFormat(Enum):
    """Supported rendering formats."""
    TEXT = "text"               # Plain text
    MARKDOWN = "markdown"       # Markdown formatting
    JSON = "json"               # JSON structured data
    HTML = "html"               # HTML formatting (if supported by target)
    STRUCTURED = "structured"   # Structured data with specific schema


class CompressionHint(Enum):
    """Hints for how content can be compressed."""
    NONE = "none"               # Do not compress this content
    SUMMARIZE = "summarize"     # Content can be summarized
    TRUNCATE = "truncate"       # Content can be truncated
    OMIT = "omit"               # Content can be omitted if needed
    REFERENCE = "reference"     # Replace with a reference/pointer


class RenderingMetadata:
    """
    Metadata for a rendered element, including importance, relationships,
    and compression hints.
    """
    
    def __init__(self,
                 element_id: str,
                 element_type: str,
                 importance: RenderingImportance = RenderingImportance.MEDIUM,
                 format: RenderingFormat = RenderingFormat.TEXT,
                 compression_hint: CompressionHint = CompressionHint.SUMMARIZE,
                 related_elements: List[str] = None,
                 attributes: Dict[str, Any] = None):
        """
        Initialize rendering metadata.
        
        Args:
            element_id: Unique identifier for the element
            element_type: Type of the element
            importance: Importance level for compression decisions
            format: Format of the rendered content
            compression_hint: Hint for how content can be compressed
            related_elements: List of related element IDs
            attributes: Additional attributes for rendering
        """
        self.element_id = element_id
        self.element_type = element_type
        self.importance = importance
        self.format = format
        self.compression_hint = compression_hint
        self.related_elements = related_elements or []
        self.attributes = attributes or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a dictionary."""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "importance": self.importance.name,
            "format": self.format.value,
            "compression_hint": self.compression_hint.value,
            "related_elements": self.related_elements,
            "attributes": self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RenderingMetadata':
        """Create metadata from a dictionary."""
        return cls(
            element_id=data["element_id"],
            element_type=data["element_type"],
            importance=RenderingImportance[data["importance"]],
            format=RenderingFormat(data["format"]),
            compression_hint=CompressionHint(data["compression_hint"]),
            related_elements=data.get("related_elements", []),
            attributes=data.get("attributes", {})
        )


class RenderingResult:
    """
    Result of rendering an element, including content and metadata.
    """
    
    def __init__(self,
                 content: str,
                 metadata: RenderingMetadata,
                 sections: Dict[str, 'RenderingResult'] = None,
                 timestamp: Optional[int] = None):
        """
        Initialize a rendering result.
        
        Args:
            content: Rendered content as string
            metadata: Metadata for the rendered content
            sections: Named sub-sections of the rendering
            timestamp: Optional timestamp for when the rendering was created
        """
        self.content = content
        self.metadata = metadata
        self.sections = sections or {}
        self.timestamp = timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rendering result to a dictionary."""
        sections_dict = {}
        for name, section in self.sections.items():
            sections_dict[name] = section.to_dict()
            
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "sections": sections_dict,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RenderingResult':
        """Create rendering result from a dictionary."""
        metadata = RenderingMetadata.from_dict(data["metadata"])
        
        sections = {}
        for name, section_data in data.get("sections", {}).items():
            sections[name] = cls.from_dict(section_data)
            
        return cls(
            content=data["content"],
            metadata=metadata,
            sections=sections,
            timestamp=data.get("timestamp")
        )
    
    def with_section(self, name: str, section: 'RenderingResult') -> 'RenderingResult':
        """
        Add or replace a section in the rendering result.
        
        Args:
            name: Name of the section
            section: Rendering result for the section
            
        Returns:
            Self for method chaining
        """
        self.sections[name] = section
        return self
    
    def to_json(self) -> str:
        """Convert rendering result to JSON string."""
        return json.dumps(self.to_dict())


class RenderingOptions:
    """
    Options for controlling how an element is rendered.
    """
    
    def __init__(self,
                 format: RenderingFormat = RenderingFormat.TEXT,
                 max_length: Optional[int] = None,
                 include_details: bool = True,
                 target_audience: str = "agent",
                 timeline_id: Optional[str] = None,
                 custom_options: Dict[str, Any] = None):
        """
        Initialize rendering options.
        
        Args:
            format: Desired output format
            max_length: Maximum length of the rendering
            include_details: Whether to include detailed information
            target_audience: Who will consume the rendering
            timeline_id: Optional timeline context for rendering
            custom_options: Additional custom options
        """
        self.format = format
        self.max_length = max_length
        self.include_details = include_details
        self.target_audience = target_audience
        self.timeline_id = timeline_id
        self.custom_options = custom_options or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert options to a dictionary."""
        return {
            "format": self.format.value,
            "max_length": self.max_length,
            "include_details": self.include_details,
            "target_audience": self.target_audience,
            "timeline_id": self.timeline_id,
            "custom_options": self.custom_options
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RenderingOptions':
        """Create options from a dictionary."""
        return cls(
            format=RenderingFormat(data.get("format", RenderingFormat.TEXT.value)),
            max_length=data.get("max_length"),
            include_details=data.get("include_details", True),
            target_audience=data.get("target_audience", "agent"),
            timeline_id=data.get("timeline_id"),
            custom_options=data.get("custom_options", {})
        )


class SceneNode:
    """
    Represents a node in the scene graph for rendering.
    Used to build a representation of the element hierarchy.
    """
    
    def __init__(self,
                 element_id: str,
                 element_type: str,
                 parent_id: Optional[str] = None,
                 children: List['SceneNode'] = None,
                 references: List[str] = None,
                 state: Dict[str, Any] = None,
                 visibility: bool = True):
        """
        Initialize a scene node.
        
        Args:
            element_id: Unique identifier for the element
            element_type: Type of the element
            parent_id: ID of the parent element
            children: Child nodes
            references: Referenced element IDs
            state: Element state data
            visibility: Whether the element is visible
        """
        self.element_id = element_id
        self.element_type = element_type
        self.parent_id = parent_id
        self.children = children or []
        self.references = references or []
        self.state = state or {}
        self.visibility = visibility
    
    def add_child(self, child: 'SceneNode') -> None:
        """Add a child node."""
        self.children.append(child)
        child.parent_id = self.element_id
    
    def add_reference(self, element_id: str) -> None:
        """Add a reference to another element."""
        if element_id not in self.references:
            self.references.append(element_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scene node to a dictionary."""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "parent_id": self.parent_id,
            "children": [child.to_dict() for child in self.children],
            "references": self.references,
            "state": self.state,
            "visibility": self.visibility
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SceneNode':
        """Create scene node from a dictionary."""
        node = cls(
            element_id=data["element_id"],
            element_type=data["element_type"],
            parent_id=data.get("parent_id"),
            references=data.get("references", []),
            state=data.get("state", {}),
            visibility=data.get("visibility", True)
        )
        
        for child_data in data.get("children", []):
            node.add_child(cls.from_dict(child_data))
            
        return node


class RenderingContext:
    """
    Context for rendering elements, including scene graph and options.
    """
    
    def __init__(self,
                 root_node: SceneNode,
                 options: RenderingOptions,
                 timeline_id: Optional[str] = None,
                 cache: Dict[str, RenderingResult] = None):
        """
        Initialize a rendering context.
        
        Args:
            root_node: Root node of the scene graph
            options: Rendering options
            timeline_id: Optional timeline context
            cache: Optional cache of previous renderings
        """
        self.root_node = root_node
        self.options = options
        self.timeline_id = timeline_id
        self.cache = cache or {}
    
    def add_to_cache(self, element_id: str, result: RenderingResult) -> None:
        """Add a rendering result to the cache."""
        self.cache[element_id] = result
    
    def get_from_cache(self, element_id: str) -> Optional[RenderingResult]:
        """Get a rendering result from the cache."""
        return self.cache.get(element_id)
    
    def find_node(self, element_id: str) -> Optional[SceneNode]:
        """Find a node in the scene graph by element ID."""
        def _find_recursive(node: SceneNode) -> Optional[SceneNode]:
            if node.element_id == element_id:
                return node
                
            for child in node.children:
                result = _find_recursive(child)
                if result:
                    return result
                    
            return None
        
        return _find_recursive(self.root_node)


# Type for rendering functions
RenderingFunction = Callable[[Dict[str, Any], RenderingOptions], RenderingResult]


class RenderingRegistry:
    """
    Registry for element rendering functions.
    """
    
    def __init__(self):
        """Initialize the rendering registry."""
        self.renderers: Dict[str, RenderingFunction] = {}
    
    def register_renderer(self, element_type: str, renderer: RenderingFunction) -> None:
        """
        Register a rendering function for an element type.
        
        Args:
            element_type: Type of element the renderer handles
            renderer: Function that renders the element
        """
        self.renderers[element_type] = renderer
        logger.debug(f"Registered renderer for element type: {element_type}")
    
    def get_renderer(self, element_type: str) -> Optional[RenderingFunction]:
        """
        Get the rendering function for an element type.
        
        Args:
            element_type: Type of element
            
        Returns:
            Rendering function if registered, None otherwise
        """
        return self.renderers.get(element_type)
    
    def render_element(self, element_type: str, state: Dict[str, Any], 
                       options: RenderingOptions) -> Optional[RenderingResult]:
        """
        Render an element using the registered function.
        
        Args:
            element_type: Type of element
            state: Element state
            options: Rendering options
            
        Returns:
            Rendering result if a renderer exists, None otherwise
        """
        renderer = self.get_renderer(element_type)
        if renderer:
            try:
                return renderer(state, options)
            except Exception as e:
                logger.error(f"Error rendering element of type {element_type}: {e}")
                return None
        else:
            logger.warning(f"No renderer found for element type: {element_type}")
            return None

# Global registry instance
registry = RenderingRegistry() 