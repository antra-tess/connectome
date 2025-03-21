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
    element_id: str
    timeline_id: Optional[str] = None
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
        This is a legacy method that still returns a string for compatibility.
        New code should use collect_renderings() instead.
        
        Args:
            timeline_id: ID of the timeline to render
            root_element_id: ID of the root element (typically a Space)
            space_registry: Optional space registry to find elements
            format: Rendering format
            include_details: Whether to include detailed information
            
        Returns:
            Simple concatenated renderings as a string (no compression applied)
        """
        logger.info(f"Preparing context rendering for timeline {timeline_id}")
        
        # Collect renderings
        renderings = self.collect_renderings(
            timeline_id=timeline_id,
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
                          timeline_id: str,
                          root_element_id: str,
                          space_registry = None,
                          format: RenderingFormat = RenderingFormat.MARKDOWN,
                          include_details: bool = True) -> List[RenderingResult]:
        """
        Collect renderings from elements without applying compression.
        
        Args:
            timeline_id: ID of the timeline to render
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
            
        # Create rendering options
        options = RenderingOptions(
            format=format,
            include_details=include_details,
            timeline_id=timeline_id,
            target_audience="agent"
        )
        
        # Build the scene graph starting from the root element
        self.current_scene_graph = self._build_scene_graph(root_element, timeline_id)
        
        # Create rendering context
        rendering_context = RenderingContext(
            root_node=self.current_scene_graph,
            options=options,
            timeline_id=timeline_id,
            cache={}  # We'll populate this as we go
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
    
    def _get_element(self, element_id: str, space_registry = None) -> Any:
        """
        Get an element by ID using the space registry.
        
        Args:
            element_id: ID of the element to get
            space_registry: Space registry to find the element
            
        Returns:
            Element if found, None otherwise
        """
        if not space_registry:
            logger.error("No space registry provided to find element")
            return None
            
        # Try to get the element from the space registry
        try:
            # Check if it's a space
            space = space_registry.get_space(element_id)
            if space:
                return space
                
            # Try to find it in spaces
            for space_id, space in space_registry.get_spaces().items():
                try:
                    if hasattr(space, "get_mounted_element"):
                        element = space.get_mounted_element(element_id)
                        if element:
                            return element
                except Exception as e:
                    logger.error(f"Error checking space {space_id} for element {element_id}: {e}")
        except Exception as e:
            logger.error(f"Error finding element {element_id}: {e}")
            
        return None
    
    def _build_scene_graph(self, root_element: Any, timeline_id: str) -> SceneNode:
        """
        Build a scene graph starting from the root element.
        
        Args:
            root_element: Root element to start from
            timeline_id: Timeline ID for state retrieval
            
        Returns:
            Scene graph root node
        """
        # Create the root node
        root_node = SceneNode(
            element_id=root_element.id,
            element_type=root_element.__class__.__name__,
            visibility=True
        )
        
        # Get state for this timeline if available
        try:
            if hasattr(root_element, "get_state_for_timeline"):
                state = root_element.get_state_for_timeline(timeline_id)
            else:
                state = root_element.get_state()
                
            root_node.state = state
        except Exception as e:
            logger.error(f"Error getting state for root element: {e}")
            root_node.state = {"error": str(e)}
        
        # If this is a space, add its mounted elements as children
        if hasattr(root_element, "IS_SPACE") and root_element.IS_SPACE:
            if hasattr(root_element, "get_mounted_elements"):
                try:
                    mounted_elements = root_element.get_mounted_elements()
                    for mount_id, element in mounted_elements.items():
                        # Check if element is visible in this timeline
                        if hasattr(element, "is_visible_in_timeline"):
                            if not element.is_visible_in_timeline(timeline_id):
                                continue
                        
                        # Create child node
                        child_node = self._build_scene_graph(element, timeline_id)
                        root_node.add_child(child_node)
                except Exception as e:
                    logger.error(f"Error getting mounted elements for {root_element.id}: {e}")
        
        return root_node
    
    def _collect_renderings(self, context: RenderingContext) -> List[RenderingResult]:
        """
        Collect renderings from all elements in the scene graph.
        
        Args:
            context: Rendering context with scene graph
            
        Returns:
            List of rendering results
        """
        renderings: List[RenderingResult] = []
        
        def collect_from_node(node: SceneNode) -> None:
            """Recursively collect renderings from a node and its children."""
            # Skip invisible nodes
            if not node.visibility:
                return
                
            # Get the element
            element = self._get_element(node.element_id, None)  # We'll need to find another way to get elements
            if not element:
                logger.warning(f"Could not find element {node.element_id} for rendering")
                return
                
            # Get delegate and render
            delegate = element.get_delegate()
            if delegate:
                try:
                    result = delegate.render(node.state, context.options)
                    renderings.append(result)
                    
                    # Add to cache
                    context.add_to_cache(node.element_id, result)
                except Exception as e:
                    logger.error(f"Error rendering element {node.element_id}: {e}")
            
            # Process children
            for child in node.children:
                collect_from_node(child)
        
        # Start collection from the root node
        if context.root_node:
            collect_from_node(context.root_node)
        
        return renderings 