"""
VeilProducer Component
Component for generating VEIL representations of Elements.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
import uuid
import time

from .base_component import Component

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VeilProducer(Component):
    """
    Component for generating VEIL representations of Elements.
    
    The VeilProducer is responsible for:
    - Generating VEIL representations based on the Element's state
    - Caching VEIL deltas (ΔV) between frame transitions
    - Providing compression hints to guide rendering
    """
    
    # Component unique type identifier
    COMPONENT_TYPE: str = "veil_producer"
    
    # Event types this component handles
    HANDLED_EVENT_TYPES: List[str] = [
        "state_update",
        "veil_request"
    ]
    
    def __init__(self, element=None, renderable_id: Optional[str] = None):
        """
        Initialize the VeilProducer component.
        
        Args:
            element: The Element this component is attached to
            renderable_id: Optional unique identifier for this renderable (defaults to element.id)
        """
        super().__init__(element)
        
        # Use element ID if no renderable_id provided
        self._renderable_id = renderable_id or (element.id if element else f"veil_{uuid.uuid4().hex[:8]}")
        
        # Initialize state
        self._state = {
            "last_frame_id": None,
            "veil_cache": {},  # frame_id -> veil data
            "delta_cache": {},  # (from_frame_id, to_frame_id) -> delta data
            "compression_hints": {},
            "last_update_time": int(time.time() * 1000)
        }
    
    def generate_veil(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a VEIL representation of the element.
        
        Args:
            options: Optional generation options
            
        Returns:
            VEIL representation data
        """
        if not self._is_initialized or not self._is_enabled:
            logger.warning(f"Cannot generate VEIL: VeilProducer component {self.id} is not initialized or enabled")
            return {"error": "Component not initialized or enabled"}
            
        # Get timestamp and frame ID
        timestamp = int(time.time() * 1000)
        frame_id = options.get("frame_id") if options else None
        if not frame_id:
            frame_id = f"frame_{timestamp}"
            
        # Basic VEIL structure
        veil = {
            "renderable_id": self._renderable_id,
            "timestamp": timestamp,
            "frame_id": frame_id,
            "element_type": self.element.__class__.__name__ if self.element else "unknown",
            "content": self._generate_content(options),
            "compression_hints": self._generate_compression_hints(options),
            "attributes": self._generate_attributes(options),
            "children": self._generate_children(options),
            "references": self._generate_references(options)
        }
        
        # Cache the VEIL
        self._state["veil_cache"][frame_id] = veil
        self._state["last_frame_id"] = frame_id
        self._state["last_update_time"] = timestamp
        
        return veil
    
    def generate_delta(self, from_frame_id: Optional[str] = None, 
                     to_frame_id: Optional[str] = None,
                     options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a VEIL delta (ΔV) between two frames.
        
        Args:
            from_frame_id: ID of the starting frame (if None, use the previous frame)
            to_frame_id: ID of the ending frame (if None, generate a new frame)
            options: Optional generation options
            
        Returns:
            VEIL delta representation
        """
        if not self._is_initialized or not self._is_enabled:
            logger.warning(f"Cannot generate delta: VeilProducer component {self.id} is not initialized or enabled")
            return {"error": "Component not initialized or enabled"}
            
        # Determine frame IDs
        current_time = int(time.time() * 1000)
        from_frame = from_frame_id or self._state["last_frame_id"]
        to_frame = to_frame_id or f"frame_{current_time}"
        
        # If we don't have a previous frame, generate a full VEIL
        if not from_frame or from_frame not in self._state["veil_cache"]:
            logger.debug(f"No previous frame, generating full VEIL for {to_frame}")
            veil = self.generate_veil(options)
            return {
                "type": "full",
                "renderable_id": self._renderable_id,
                "from_frame_id": None,
                "to_frame_id": to_frame,
                "timestamp": current_time,
                "veil": veil
            }
            
        # Check if we already have this delta cached
        cache_key = (from_frame, to_frame)
        if cache_key in self._state["delta_cache"]:
            logger.debug(f"Using cached delta for {from_frame} -> {to_frame}")
            return self._state["delta_cache"][cache_key]
            
        # Generate the 'to' frame if needed
        if to_frame not in self._state["veil_cache"]:
            self.generate_veil({**(options or {}), "frame_id": to_frame})
            
        # Get the two frames
        from_veil = self._state["veil_cache"][from_frame]
        to_veil = self._state["veil_cache"][to_frame]
        
        # Calculate delta
        delta = self._calculate_delta(from_veil, to_veil)
        
        # Create delta structure
        delta_result = {
            "type": "delta",
            "renderable_id": self._renderable_id,
            "from_frame_id": from_frame,
            "to_frame_id": to_frame,
            "timestamp": current_time,
            "changes": delta
        }
        
        # Cache the delta
        self._state["delta_cache"][cache_key] = delta_result
        
        return delta_result
    
    def _calculate_delta(self, from_veil: Dict[str, Any], to_veil: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the delta between two VEIL structures.
        
        Args:
            from_veil: Starting VEIL structure
            to_veil: Ending VEIL structure
            
        Returns:
            Dictionary of changes
        """
        changes = {}
        
        # Check content changes
        if from_veil.get("content") != to_veil.get("content"):
            changes["content"] = to_veil.get("content")
            
        # Check attribute changes
        from_attrs = from_veil.get("attributes", {})
        to_attrs = to_veil.get("attributes", {})
        
        attr_changes = {}
        for key, value in to_attrs.items():
            if key not in from_attrs or from_attrs[key] != value:
                attr_changes[key] = value
                
        # Check for removed attributes
        removed_attrs = []
        for key in from_attrs:
            if key not in to_attrs:
                removed_attrs.append(key)
                
        if attr_changes or removed_attrs:
            changes["attributes"] = {
                "changed": attr_changes,
                "removed": removed_attrs
            }
            
        # Check for children changes
        from_children = from_veil.get("children", [])
        to_children = to_veil.get("children", [])
        
        # Simple implementation for now - just replace children if different
        if from_children != to_children:
            changes["children"] = to_children
            
        # Check reference changes
        from_refs = from_veil.get("references", {})
        to_refs = to_veil.get("references", {})
        
        if from_refs != to_refs:
            changes["references"] = to_refs
            
        # Check compression hint changes
        from_hints = from_veil.get("compression_hints", {})
        to_hints = to_veil.get("compression_hints", {})
        
        if from_hints != to_hints:
            changes["compression_hints"] = to_hints
            
        return changes
    
    def _generate_content(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate the content portion of the VEIL representation.
        
        Args:
            options: Optional generation options
            
        Returns:
            Content representation
        """
        # Default implementation - override in subclasses
        if not self.element:
            return {"text": "No element attached"}
            
        # Try to get a string representation
        try:
            return {"text": str(self.element.name)}
        except:
            return {"text": f"Element {self.element.id}"}
    
    def _generate_compression_hints(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate compression hints for the VEIL representation.
        
        Args:
            options: Optional generation options
            
        Returns:
            Compression hints
        """
        # Default implementation - override in subclasses
        return {
            "importance": 5,  # 1-10 scale
            "can_summarize": True,
            "min_tokens": 10,
            "preferred_tokens": 50,
            "max_tokens": 200
        }
    
    def _generate_attributes(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate attributes for the VEIL representation.
        
        Args:
            options: Optional generation options
            
        Returns:
            Attributes dictionary
        """
        # Default implementation - override in subclasses
        if not self.element:
            return {}
            
        return {
            "id": self.element.id,
            "name": self.element.name,
            "description": self.element.description
        }
    
    def _generate_children(self, options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate children for the VEIL representation.
        
        Args:
            options: Optional generation options
            
        Returns:
            List of child references
        """
        # Default implementation - override in subclasses
        return []
    
    def _generate_references(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate references for the VEIL representation.
        
        Args:
            options: Optional generation options
            
        Returns:
            References dictionary
        """
        # Default implementation - override in subclasses
        return {}
    
    def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handle events relevant to VEIL production.
        
        Args:
            event: Event data
            timeline_context: Timeline context for this event
            
        Returns:
            True if the event was handled, False otherwise
        """
        event_type = event.get("event_type")
        
        if event_type == "state_update":
            # State has changed, so regenerate VEIL
            self.generate_veil()
            return True
            
        elif event_type == "veil_request":
            # Someone is explicitly requesting a VEIL
            options = event.get("options", {})
            
            if event.get("delta_request", False):
                from_frame = event.get("from_frame_id")
                to_frame = event.get("to_frame_id")
                self.generate_delta(from_frame, to_frame, options)
            else:
                self.generate_veil(options)
                
            return True
            
        return False
    
    def get_cached_veil(self, frame_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a cached VEIL representation.
        
        Args:
            frame_id: ID of the frame to retrieve (if None, get the latest)
            
        Returns:
            Cached VEIL data or None if not found
        """
        if not frame_id:
            frame_id = self._state["last_frame_id"]
            
        if not frame_id or frame_id not in self._state["veil_cache"]:
            return None
            
        return self._state["veil_cache"][frame_id]
    
    def get_cached_delta(self, from_frame_id: str, to_frame_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached VEIL delta.
        
        Args:
            from_frame_id: ID of the starting frame
            to_frame_id: ID of the ending frame
            
        Returns:
            Cached delta data or None if not found
        """
        cache_key = (from_frame_id, to_frame_id)
        return self._state["delta_cache"].get(cache_key) 