"""
HUD Core Implementation

The Heads-Up Display (HUD) is the shell component responsible for rendering context.
It coordinates the collection of renderings from Element Delegates without
applying compression (which is now handled by the ContextManager).
"""

import logging
from typing import Dict, Any, List, Optional, Union, Set, Tuple
import time
from dataclasses import dataclass
import inspect

from bot_framework.rendering import (
    RenderingOptions,
    RenderingResult,
    RenderingMetadata,
    RenderingImportance,
    RenderingFormat,
    CompressionHint,
    SceneNode,
    RenderingContext
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RenderingRequest:
    """Request for rendering an element."""
    timeline_id: Optional[str] = None
    root_element_id: Optional[str] = None
    format: RenderingFormat = RenderingFormat.MARKDOWN
    include_details: bool = True
    max_length: Optional[int] = None
    custom_options: Dict[str, Any] = None


@dataclass
class RenderingResponse:
    """Response containing rendered content."""
    content: str
    element_id: str
    metadata: Dict[str, Any]
    sections: Dict[str, Any] = None
    timestamp: Optional[int] = None
    

class HUD:
    """
    Heads-Up Display for rendering agent context.
    
    The HUD is responsible for:
    1. Collecting renderings from Element Delegates
    2. Building a scene graph of the current state
    3. Providing renderings for the Context Manager
    
    It no longer handles compression directly, which is now
    the responsibility of the Context Manager.
    """
    
    def __init__(self, max_context_tokens: int = 8000):
        """
        Initialize the HUD.
        
        Args:
            max_context_tokens: Maximum tokens guideline (used for caching)
        """
        self.max_context_tokens = max_context_tokens
        self.rendering_cache: Dict[str, Dict[str, RenderingResult]] = {}  # element_id -> {timeline_id -> result}
        self.cache_invalidation: Dict[str, int] = {}  # element_id -> timestamp
        self.current_scene_graph: Optional[SceneNode] = None
        
        logger.info(f"Initialized HUD (max_tokens={max_context_tokens})")
    
    def prepare_context_rendering(self, 
                                timeline_id: str,
                                root_element_id: str,
                                space_registry = None,
                                format: RenderingFormat = RenderingFormat.MARKDOWN,
                                include_details: bool = True) -> str:
        """
        Prepare the context rendering for an agent.
        
        Args:
            timeline_id: ID of the timeline to render
            root_element_id: ID of the root element (typically a Space)
            space_registry: Optional space registry to find elements
            format: Rendering format
            include_details: Whether to include detailed information
            
        Returns:
            Simple concatenated renderings as a string
        """
        logger.info(f"Preparing context rendering for timeline {timeline_id}")
        
        # Create full timeline context, not just ID
        timeline_context = {
            "timeline_id": timeline_id,
            "is_primary": True,  # Default to primary
            "last_event_id": None
        }
        
        # Update is_primary if we can determine it
        if space_registry:
            inner_space = space_registry.get_inner_space()
            if inner_space and hasattr(inner_space, '_timeline_state'):
                timeline_context["is_primary"] = (timeline_id == inner_space._timeline_state.get("primary_timeline"))
        
        # Collect renderings with full timeline context
        renderings = self.collect_renderings(
            timeline_context=timeline_context,
            root_element_id=root_element_id,
            space_registry=space_registry,
            format=format,
            include_details=include_details
        )
        
        # Simply concatenate renderings with double newline separator
        # No compression is applied here
        rendered_parts = [r.content for r in renderings]
        final_context = "\n\n".join(rendered_parts)
        
        logger.info(f"Completed context rendering for timeline {timeline_id}")
        return final_context
    
    def collect_renderings(self,
                          timeline_context: Dict[str, Any],
                          root_element_id: str,
                          space_registry = None,
                          format: RenderingFormat = RenderingFormat.MARKDOWN,
                          include_details: bool = True) -> List[RenderingResult]:
        """
        Collect renderings from elements without applying compression.
        
        Args:
            timeline_context: Timeline context for the rendering
            root_element_id: ID of the root element (typically a Space)
            space_registry: Optional space registry to find elements
            format: Rendering format
            include_details: Whether to include detailed information
            
        Returns:
            List of rendering results from all elements
        """
        # Build scene graph
        root_element = self._get_element(root_element_id, space_registry)
        if not root_element:
            logger.error(f"Could not find root element {root_element_id}")
            error_result = RenderingResult(
                content=f"Error: Could not prepare context (root element {root_element_id} not found)",
                metadata=RenderingMetadata(
                    importance=RenderingImportance.CRITICAL,
                    element_id="error",
                    element_type="error"
                ),
                sections={},
                timestamp=int(time.time() * 1000)
            )
            return [error_result]
            
        # Create rendering options with timeline context
        options = RenderingOptions(
            format=format,
            include_details=include_details,
            timeline_id=timeline_context.get("timeline_id"),
            timeline_context=timeline_context,
            target_audience="agent"
        )
        
        # Build the scene graph starting from the root element
        self.current_scene_graph = self._build_scene_graph(root_element, timeline_context)
        
        # Create rendering context
        rendering_context = RenderingContext(
            root_node=self.current_scene_graph,
            options=options,
            timeline_id=timeline_context.get("timeline_id"),
            timeline_context=timeline_context,
            cache={},  # We'll populate this as we go
            space_registry=space_registry
        )
        
        # Collect renderings from all elements in the scene graph
        renderings = self._collect_renderings(rendering_context)
        
        return renderings
    
    def render_element(self, request: RenderingRequest, space_registry = None) -> RenderingResponse:
        """
        Render a single element.
        
        Args:
            request: Rendering request
            space_registry: Optional space registry to find the element
            
        Returns:
            Rendering response with the rendered content
        """
        logger.debug(f"Rendering element {request.element_id}")
        
        # Get the element
        element = self._get_element(request.element_id, space_registry)
        if not element:
            logger.error(f"Could not find element {request.element_id}")
            return RenderingResponse(
                content=f"Error: Element {request.element_id} not found",
                element_id=request.element_id,
                metadata={"error": "element_not_found"}
            )
        
        # Get the delegate from the element
        delegate = element.get_delegate()
        if not delegate:
            logger.error(f"Element {request.element_id} has no delegate")
            return RenderingResponse(
                content=f"Error: Element {request.element_id} has no rendering delegate",
                element_id=request.element_id,
                metadata={"error": "no_delegate"}
            )
        
        # Create rendering options
        options = RenderingOptions(
            format=request.format,
            include_details=request.include_details,
            max_length=request.max_length,
            timeline_id=request.timeline_id,
            target_audience="agent",
            custom_options=request.custom_options or {}
        )
        
        # Get element state for the specified timeline
        if request.timeline_id:
            # Get timeline-specific state if available
            try:
                if hasattr(element, "get_state_for_timeline"):
                    state = element.get_state_for_timeline(request.timeline_id)
                else:
                    # Fall back to regular state
                    state = element.get_state()
            except Exception as e:
                logger.error(f"Error getting state for element {request.element_id}: {e}")
                state = element.get_state()
        else:
            # Regular state
            state = element.get_state()
        
        # Check cache
        cache_key = f"{request.element_id}:{request.timeline_id or 'default'}"
        if cache_key in self.rendering_cache:
            cached_result = self.rendering_cache[cache_key]
            if cached_result.timestamp > self.cache_invalidation.get(request.element_id, 0):
                logger.debug(f"Using cached rendering for {request.element_id}")
                result = cached_result
            else:
                # Cache is invalidated, render again
                result = delegate.render(state, options)
                self.rendering_cache[cache_key] = result
        else:
            # Not in cache, render
            result = delegate.render(state, options)
            self.rendering_cache[cache_key] = result
        
        # Convert to response
        response = RenderingResponse(
            content=result.content,
            element_id=request.element_id,
            metadata=result.metadata.to_dict(),
            sections={name: section.content for name, section in result.sections.items()},
            timestamp=result.timestamp
        )
        
        return response
    
    def invalidate_cache(self, element_id: str) -> None:
        """
        Invalidate the rendering cache for an element.
        
        Args:
            element_id: ID of the element to invalidate
        """
        self.cache_invalidation[element_id] = int(time.time() * 1000)
        logger.debug(f"Invalidated rendering cache for element {element_id}")
    
    def _get_element(self, element_id: str, space_registry = None):
        """
        Get an element by ID.
        
        Args:
            element_id: ID of the element to retrieve
            space_registry: Optional space registry to find the element
            
        Returns:
            The element if found, None otherwise
        """
        if not space_registry:
            logger.error("No space registry provided to get element")
            return None
            
        # Try to get the element from the space registry
        return space_registry.get_space(element_id)
    
    def _build_scene_graph(self, root_element, timeline_context: Dict[str, Any]) -> SceneNode:
        """
        Build a scene graph starting from the root element.
        
        Args:
            root_element: The root element to start from
            timeline_context: The timeline context to use
            
        Returns:
            The root scene node of the scene graph
        """
        # Store all created nodes by element ID for establishing references
        element_nodes = {}
        reference_collection = {}
        
        # Inner function to recursively build the graph
        def build_node(element) -> SceneNode:
            # Create the node
            node = SceneNode(
                element_id=element.id,
                element_type=element.__class__.__name__,
                parent_id=None,
                visibility=True
            )
            
            # Store in element_nodes for reference creation
            element_nodes[element.id] = node
            
            # Get state for this timeline
            try:
                if hasattr(element, "get_state_for_timeline"):
                    state = element.get_state_for_timeline(timeline_context.get("timeline_id"))
                else:
                    state = element.get_state()
                
                node.state = state
                
                # Collect reference links if available
                reference_links = self._collect_element_references(element, state)
                if reference_links:
                    reference_collection[element.id] = reference_links
                
            except Exception as e:
                logger.error(f"Error getting state for element {element.id}: {e}")
                node.state = {}
            
            # If element is a Space with mounted elements, add them as children
            if hasattr(element, "get_mounted_elements"):
                mounted_elements = element.get_mounted_elements()
                
                for mount_id, mounted_element in mounted_elements.items():
                    # Recursively build the scene graph for this element
                    child_node = build_node(mounted_element)
                    child_node.parent_id = element.id
                    node.children.append(child_node)
            
            return node
        
        # Build the graph
        root_node = build_node(root_element)
        
        # Process collected references to add secondary links
        self._add_reference_links(element_nodes, reference_collection)
        
        return root_node
        
    def _collect_element_references(self, element, state: Dict[str, Any]) -> List[str]:
        """
        Collect references from an element state.
        
        This identifies other elements that this element has relationships with,
        beyond the simple parent-child hierarchy.
        
        Args:
            element: The element to collect references from
            state: The element's state
            
        Returns:
            List of referenced element IDs
        """
        references = []
        
        # Try various methods to collect references
        
        # 1. Check for explicit related_elements in state
        if 'related_elements' in state:
            references.extend(state['related_elements'])
            
        # 2. Check for parent_element reference if available
        if hasattr(element, '_parent_element') and element._parent_element:
            parent_id, _ = element._parent_element
            references.append(parent_id)
            
        # 3. For UplinkProxy, add remote_space_id as a reference
        if hasattr(element, 'remote_space_id') and element.remote_space_id:
            references.append(element.remote_space_id)
            
        # 4. For chat elements, collect references to mentioned users/entities
        if 'message_history' in state:
            for message in state.get('message_history', []):
                if 'mentions' in message:
                    references.extend(message['mentions'])
                    
        # 5. For elements with delegates, check metadata 
        if hasattr(element, 'get_delegate') and callable(getattr(element, 'get_delegate', None)):
            delegate = element.get_delegate()
            if delegate and hasattr(delegate, 'get_metadata'):
                metadata = delegate.get_metadata(state)
                if metadata and 'related_elements' in metadata:
                    references.extend(metadata['related_elements'])
        
        # Remove duplicates and the element's own ID
        references = [ref for ref in references if ref and ref != element.id]
        return list(set(references))
        
    def _add_reference_links(self, element_nodes: Dict[str, SceneNode], 
                          reference_collection: Dict[str, List[str]]) -> None:
        """
        Add reference links to the scene graph nodes.
        
        Args:
            element_nodes: Dictionary mapping element IDs to scene nodes
            reference_collection: Dictionary mapping element IDs to their referenced element IDs
        """
        for element_id, references in reference_collection.items():
            if element_id not in element_nodes:
                continue
                
            node = element_nodes[element_id]
            
            for ref_id in references:
                # Only add references to elements that exist in the graph
                if ref_id in element_nodes:
                    node.add_reference(ref_id)
                    logger.debug(f"Added reference link: {element_id} -> {ref_id}")
    
    def _collect_renderings(self, rendering_context: RenderingContext) -> List[RenderingResult]:
        """
        Collect renderings from all elements in the scene graph.
        
        Args:
            rendering_context: Rendering context containing the scene graph
            
        Returns:
            List of rendering results
        """
        renderings = []
        
        # Start with the root node
        root_node = rendering_context.root_node
        
        # Process the root node
        root_element_id = root_node.element_id
        root_element_type = root_node.element_type
        
        # Get the element's delegate renderer
        from bot_framework.rendering import registry
        renderer = registry.get_renderer(root_element_type)
        if renderer:
            try:
                # Get element state
                element = self._get_element(root_element_id, rendering_context.space_registry)
                if element and hasattr(element, 'is_open'):
                    # Element has state tracking, use appropriate view
                    if element.is_open():
                        # Interior view for open elements
                        result = renderer(root_node.state, rendering_context.options)
                        renderings.append(result)
                        
                        # Process children recursively
                        for child_node in root_node.children:
                            # Create a new rendering context for the child
                            child_context = RenderingContext(
                                root_node=child_node,
                                options=rendering_context.options,
                                timeline_id=rendering_context.timeline_id,
                                timeline_context=rendering_context.timeline_context,
                                cache=rendering_context.cache,
                                space_registry=rendering_context.space_registry
                            )
                            
                            # Collect renderings from this child
                            child_renderings = self._collect_renderings(child_context)
                            renderings.extend(child_renderings)
                    else:
                        # Exterior view for closed elements
                        result = renderer(root_node.state, rendering_context.options)
                        renderings.append(result)
                else:
                    # No state tracking, use default rendering
                    result = renderer(root_node.state, rendering_context.options)
                    renderings.append(result)
                    
                    # Process children if no state tracking
                    for child_node in root_node.children:
                        child_context = RenderingContext(
                            root_node=child_node,
                            options=rendering_context.options,
                            timeline_id=rendering_context.timeline_id,
                            timeline_context=rendering_context.timeline_context,
                            cache=rendering_context.cache,
                            space_registry=rendering_context.space_registry
                        )
                        child_renderings = self._collect_renderings(child_context)
                        renderings.extend(child_renderings)
                        
            except Exception as e:
                logger.error(f"Error rendering element {root_element_id}: {e}")
                
                # Add an error rendering
                error_result = RenderingResult(
                    content=f"Error rendering {root_element_type} ({root_element_id}): {e}",
                    metadata=RenderingMetadata(
                        importance=RenderingImportance.HIGH,
                        element_id=root_element_id,
                        element_type=root_element_type
                    )
                )
                renderings.append(error_result)
        else:
            # No renderer found, use default rendering
            default_result = RenderingResult(
                content=f"{root_element_type} ({root_element_id})",
                metadata=RenderingMetadata(
                    importance=RenderingImportance.MEDIUM,
                    element_id=root_element_id,
                    element_type=root_element_type
                )
            )
            renderings.append(default_result)
        
        return renderings 

    def visualize_scene_graph(self, format: str = "text") -> str:
        """
        Visualize the current scene graph for debugging purposes.
        
        Args:
            format: Output format (text, markdown, or json)
            
        Returns:
            String representation of the scene graph
        """
        if not self.current_scene_graph:
            return "No scene graph available"
            
        if format == "json":
            import json
            return json.dumps(self.current_scene_graph.to_dict(), indent=2)
            
        # For text or markdown
        output = []
        visited = set()
        
        def visit_node(node, depth=0, path=""):
            if node.element_id in visited:
                return
                
            visited.add(node.element_id)
            indent = "  " * depth
            node_type = node.element_type
            
            # Format based on output type
            if format == "markdown":
                node_repr = f"{indent}- **{node_type}** (`{node.element_id}`)"
            else:
                node_repr = f"{indent}{node_type} ({node.element_id})"
                
            # Add the path
            current_path = f"{path}/{node.element_id}" if path else node.element_id
            if format == "markdown":
                node_repr += f" [Path: `{current_path}`]"
            else:
                node_repr += f" Path: {current_path}"
                
            output.append(node_repr)
            
            # Add references
            if node.references:
                ref_indent = "  " * (depth + 1)
                if format == "markdown":
                    output.append(f"{ref_indent}- *References*:")
                else:
                    output.append(f"{ref_indent}References:")
                    
                for ref_id in node.references:
                    if format == "markdown":
                        output.append(f"{ref_indent}  - `{ref_id}`")
                    else:
                        output.append(f"{ref_indent}  {ref_id}")
            
            # Visit children
            for child in node.children:
                visit_node(child, depth + 1, current_path)
        
        # Start with the root
        visit_node(self.current_scene_graph)
        
        # Add reference edges for visualization
        output.append("")
        if format == "markdown":
            output.append("**Reference Edges:**")
        else:
            output.append("Reference Edges:")
            
        edges = set()
        
        def collect_edges(node):
            for ref_id in node.references:
                edges.add((node.element_id, ref_id))
                
            for child in node.children:
                collect_edges(child)
        
        collect_edges(self.current_scene_graph)
        
        # Sort edges for stable output
        sorted_edges = sorted(edges)
        for src, dst in sorted_edges:
            if format == "markdown":
                output.append(f"- `{src}` → `{dst}`")
            else:
                output.append(f"{src} -> {dst}")
        
        return "\n".join(output)
    
    def get_element_relationships(self, element_id: str) -> Dict[str, List[str]]:
        """
        Get all relationships for a specific element in the scene graph.
        
        Args:
            element_id: ID of the element to get relationships for
            
        Returns:
            Dictionary with 'parents', 'children', and 'references' lists
        """
        if not self.current_scene_graph:
            return {"parents": [], "children": [], "references": []}
            
        # Find the element node
        def find_node(node):
            if node.element_id == element_id:
                return node
                
            for child in node.children:
                result = find_node(child)
                if result:
                    return result
                    
            return None
            
        node = find_node(self.current_scene_graph)
        if not node:
            return {"parents": [], "children": [], "references": []}
            
        # Get direct relationships
        relationships = {
            "parents": [node.parent_id] if node.parent_id else [],
            "children": [child.element_id for child in node.children],
            "references": node.references.copy()
        }
        
        # Find elements that reference this element
        referencing_elements = []
        
        def find_referencing(node):
            if element_id in node.references:
                referencing_elements.append(node.element_id)
                
            for child in node.children:
                find_referencing(child)
                
        find_referencing(self.current_scene_graph)
        relationships["referenced_by"] = referencing_elements
        
        return relationships 

    def render_element(self, element, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render an element using its delegate.
        
        Args:
            element: The element to render
            state: The element's current state
            options: Rendering options
            
        Returns:
            Rendering result containing the rendered content
        """
        # Get the delegate for this element
        delegate = self._get_element_delegate(element)
        
        if not delegate:
            logger.warning(f"No delegate found for element {element.id}, using default renderer")
            from bot_framework.rendering.delegates import default_renderer
            return default_renderer(state, options)
        
        # Special handling for remote chat elements that need connection spans
        if hasattr(element, 'is_remote') and callable(getattr(element, 'is_remote')) and element.is_remote():
            # Check if this is a ChatElement with remote rendering
            if hasattr(element, 'get_connection_spans') and callable(getattr(element, 'get_connection_spans')):
                # Get connection spans from the element
                connection_spans = element.get_connection_spans()
                
                # Render with connection spans if the delegate supports it
                if hasattr(delegate, '__call__') and len(inspect.signature(delegate).parameters) >= 3:
                    # Function delegate with support for connection spans
                    return delegate(state, options, connection_spans)
            
        # Standard rendering for regular elements
        if hasattr(delegate, 'render') and callable(getattr(delegate, 'render')):
            # Object-style delegate
            return delegate.render(state, options)
        elif hasattr(delegate, '__call__'):
            # Function-style delegate
            return delegate(state, options)
        else:
            logger.warning(f"Invalid delegate for element {element.id}, using default renderer")
            from bot_framework.rendering.delegates import default_renderer
            return default_renderer(state, options) 