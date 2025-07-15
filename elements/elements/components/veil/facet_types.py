"""
Concrete VEILFacet Type Implementations

Implements the three distinct facet types: EventFacet, StatusFacet, and AmbientFacet.
Each type has specific temporal behaviors and use cases.
"""

import logging
from typing import Dict, Any, Optional, List
from .veil_facet import VEILFacet, VEILFacetType
from .temporal_system import ConnectomeEpoch

logger = logging.getLogger(__name__)

class EventFacet(VEILFacet):
    """
    Event Facets: Discrete temporal occurrences with specific encounter-time positioning.
    
    Examples:
    - Chat messages added/edited/deleted
    - Notes created/modified/removed
    - Agent responses generated
    - System notifications
    
    Temporal Behavior:
    - Strict veil_timestamp ordering (encounter-time)
    - Immutable temporal position once set
    - Always rendered in chronological sequence
    """
    
    def __init__(self, 
                 facet_id: str,
                 veil_timestamp: float,
                 owner_element_id: str,
                 event_type: str,
                 content: str,
                 links_to: Optional[str] = None):
        """
        Initialize an EventFacet.
        
        Args:
            facet_id: Unique identifier for this facet
            veil_timestamp: Encounter-time when event occurred
            owner_element_id: Element that owns this event
            event_type: Type of event (e.g., "message_added", "note_created", "agent_response")
            content: Event content
            links_to: Optional link to container or other facet
        """
        super().__init__(facet_id, VEILFacetType.EVENT, veil_timestamp, owner_element_id, links_to)
        self.properties.update({
            "event_type": event_type,
            "content": content,
            "content_nature": self._infer_content_nature(event_type),
            "structural_role": "temporal_event"
        })
        
    def _infer_content_nature(self, event_type: str) -> str:
        """Infer content nature from event type."""
        if event_type in ["message_added", "message_edited"]:
            return "message_content"
        elif event_type in ["note_created", "note_modified"]:
            return "note_content"
        elif event_type == "agent_response":
            return "agent_content"
        else:
            return "event_content"
        
    def to_veil_dict(self) -> Dict[str, Any]:
        """Convert EventFacet to VEIL dictionary representation."""
        return {
            "facet_id": self.facet_id,
            "facet_type": "event",
            "veil_timestamp": self.veil_timestamp,
            "owner_element_id": self.owner_element_id,
            "links_to": self.links_to,
            "properties": self.properties.copy()
        }
        
    def get_content_summary(self) -> str:
        """Get human-readable content summary for rendering."""
        event_type = self.properties.get("event_type", "unknown")
        content = self.properties.get("content", "")
        
        # Truncate long content for summary
        if len(content) > 100:
            content = content[:97] + "..."
            
        return f"{event_type}: {content}"

class StatusFacet(VEILFacet):
    """
    Status Facets: Container and Element state representations.
    
    Examples:
    - Chat container creation/updates
    - Scratchpad container status
    - Element name changes
    - Join/leave chat events
    - Space configuration updates
    
    Temporal Behavior:
    - Positioned at veil_timestamp when status change occurred
    - Represents "current state as of this time"
    - Triggers ambient facet rendering
    """
    
    def __init__(self,
                 facet_id: str,
                 veil_timestamp: float,
                 owner_element_id: str,
                 status_type: str,
                 current_state: Dict[str, Any],
                 links_to: Optional[str] = None):
        """
        Initialize a StatusFacet.
        
        Args:
            facet_id: Unique identifier for this facet
            veil_timestamp: When status change occurred
            owner_element_id: Element that owns this status
            status_type: Type of status change (e.g., "container_created", "element_renamed")
            current_state: Current state data
            links_to: Optional link to parent container
        """
        super().__init__(facet_id, VEILFacetType.STATUS, veil_timestamp, owner_element_id, links_to)
        self.properties.update({
            "status_type": status_type,
            "current_state": current_state.copy(),
            "structural_role": "status_indicator"
        })
        
    def to_veil_dict(self) -> Dict[str, Any]:
        """Convert StatusFacet to VEIL dictionary representation."""
        return {
            "facet_id": self.facet_id,
            "facet_type": "status", 
            "veil_timestamp": self.veil_timestamp,
            "owner_element_id": self.owner_element_id,
            "links_to": self.links_to,
            "properties": self.properties.copy()
        }
        
    def get_content_summary(self) -> str:
        """Get human-readable content summary for rendering."""
        status_type = self.properties.get("status_type", "unknown")
        current_state = self.properties.get("current_state", {})
        
        # Extract key info from state for summary
        if "name" in current_state:
            return f"{status_type}: {current_state['name']}"
        elif "conversation_name" in current_state:
            return f"{status_type}: {current_state['conversation_name']}"
        else:
            return f"{status_type}: {len(current_state)} properties"

class AmbientFacet(VEILFacet):
    """
    Ambient Facets: Floating contextual information without strict temporal positioning.
    
    Examples:
    - Tool instructions and availability
    - System capabilities descriptions
    - Persistent UI affordances
    - Background context information
    
    Temporal Behavior:
    - No strict veil_timestamp positioning (uses 0.0)
    - Rendered after status changes OR after configurable text thresholds
    - "Floats" temporally to maintain relevance
    
    Rendering Rules:
    1. Always render after latest status facet
    2. Re-render after every status change
    3. Re-render after `ambient_text_threshold` symbols since last ambient rendering
    """
    
    def __init__(self,
                 facet_id: str,
                 owner_element_id: str,
                 ambient_type: str,
                 content: str,
                 trigger_threshold: int = 500,  # Symbols between re-renders
                 links_to: Optional[str] = None):
        """
        Initialize an AmbientFacet.
        
        Args:
            facet_id: Unique identifier for this facet
            owner_element_id: Element that owns this ambient context
            ambient_type: Type of ambient context (e.g., "tool_instructions", "system_capabilities")
            content: Ambient content
            trigger_threshold: Symbol threshold for re-rendering
            links_to: Optional link to related facet
        """
        # Ambient facets use veil_timestamp = 0 as they float temporally
        super().__init__(facet_id, VEILFacetType.AMBIENT, 0.0, owner_element_id, links_to)
        self.properties.update({
            "ambient_type": ambient_type,
            "content": content,
            "trigger_threshold": trigger_threshold,
            "structural_role": "ambient_context"
        })
        
    def to_veil_dict(self) -> Dict[str, Any]:
        """Convert AmbientFacet to VEIL dictionary representation."""
        return {
            "facet_id": self.facet_id,
            "facet_type": "ambient",
            "veil_timestamp": 0.0,  # Ambient facets float temporally
            "owner_element_id": self.owner_element_id,
            "links_to": self.links_to,
            "properties": self.properties.copy()
        }
        
    def get_content_summary(self) -> str:
        """Get human-readable content summary for rendering."""
        ambient_type = self.properties.get("ambient_type", "unknown")
        content = self.properties.get("content", "")
        
        # Truncate long content for summary
        if len(content) > 150:
            content = content[:147] + "..."
            
        return f"{ambient_type}: {content}"
        
    def should_trigger_render(self, symbols_since_last: int) -> bool:
        """
        Check if ambient facet should trigger re-rendering.
        
        Args:
            symbols_since_last: Number of symbols rendered since last ambient rendering
            
        Returns:
            True if threshold exceeded and should re-render
        """
        threshold = self.properties.get("trigger_threshold", 500)
        return symbols_since_last >= threshold

# Utility functions for facet creation

def create_message_event_facet(message_id: str, 
                              owner_element_id: str, 
                              content: str,
                              sender_name: str = "Unknown",
                              links_to: Optional[str] = None) -> EventFacet:
    """
    Create an EventFacet for a new message.
    
    Args:
        message_id: Unique message identifier
        owner_element_id: Element owning the message
        content: Message content
        sender_name: Name of message sender
        links_to: Container to link to
        
    Returns:
        EventFacet configured for message events
    """
    facet = EventFacet(
        facet_id=message_id,
        veil_timestamp=ConnectomeEpoch.get_veil_timestamp(),
        owner_element_id=owner_element_id,
        event_type="message_added",
        content=content,
        links_to=links_to
    )
    
    # Add message-specific properties
    facet.properties.update({
        "sender_name": sender_name,
        "message_type": "user_message"
    })
    
    return facet

def create_agent_response_facet(response_id: str,
                               owner_element_id: str,
                               content: str,
                               tool_calls: Optional[List[Dict[str, Any]]] = None,
                               links_to: Optional[str] = None) -> EventFacet:
    """
    Create an EventFacet for an agent response.
    
    Args:
        response_id: Unique response identifier
        owner_element_id: Element owning the response
        content: Agent response content
        tool_calls: Optional tool calls made by agent
        links_to: Container to link to
        
    Returns:
        EventFacet configured for agent responses
    """
    facet = EventFacet(
        facet_id=response_id,
        veil_timestamp=ConnectomeEpoch.get_veil_timestamp(),
        owner_element_id=owner_element_id,
        event_type="agent_response",
        content=content,
        links_to=links_to
    )
    
    # Add agent response-specific properties
    facet.properties.update({
        "tool_calls_count": len(tool_calls) if tool_calls else 0,
        "has_tool_calls": bool(tool_calls),
        "tool_calls": tool_calls if tool_calls else [],
        "agent_name": "Agent",  # Can be customized
        "parsing_mode": "text"  # or "tool_call"
    })
    
    return facet

def create_container_status_facet(container_id: str,
                                 owner_element_id: str,
                                 container_name: str,
                                 container_type: str = "container",
                                 links_to: Optional[str] = None) -> StatusFacet:
    """
    Create a StatusFacet for container creation/updates.
    
    Args:
        container_id: Unique container identifier
        owner_element_id: Element owning the container
        container_name: Human-readable container name
        container_type: Type of container
        links_to: Parent container to link to
        
    Returns:
        StatusFacet configured for container status
    """
    return StatusFacet(
        facet_id=container_id,
        veil_timestamp=ConnectomeEpoch.get_veil_timestamp(),
        owner_element_id=owner_element_id,
        status_type="container_created",
        current_state={
            "name": container_name,
            "container_type": container_type,
            "created_at": ConnectomeEpoch.get_veil_timestamp()
        },
        links_to=links_to
    )

def create_tool_instructions_ambient_facet(facet_id: str,
                                          owner_element_id: str,
                                          tools_content: str,
                                          threshold: int = 500) -> AmbientFacet:
    """
    Create an AmbientFacet for tool instructions.
    
    Args:
        facet_id: Unique facet identifier
        owner_element_id: Element owning the tools
        tools_content: Tool instructions content
        threshold: Symbol threshold for re-rendering
        
    Returns:
        AmbientFacet configured for tool instructions
    """
    return AmbientFacet(
        facet_id=facet_id,
        owner_element_id=owner_element_id,
        ambient_type="tool_instructions",
        content=tools_content,
        trigger_threshold=threshold
    ) 