"""
VEILFacet Core Interface

Defines the abstract base interface for all VEIL facets in the endotemporal flat architecture.
Replaces current hierarchical VEIL nodes with flat, typed facets.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class VEILFacetType(Enum):
    """
    Defines the three distinct facet types with different temporal behaviors.
    """
    EVENT = "event"      # Discrete temporal occurrences
    STATUS = "status"    # Container/Element state representations  
    AMBIENT = "ambient"  # Floating context (tools, instructions)

class VEILFacet(ABC):
    """
    Abstract base interface for all VEIL facets.
    Replaces current hierarchical VEIL nodes with flat, typed facets.
    
    Core Principles:
    - Flat temporal ordering by veil_timestamp
    - Semantic facet types (event, status, ambient)
    - Simple many-to-one linking
    - Endotemporal consistency
    """
    
    def __init__(self, 
                 facet_id: str,
                 facet_type: VEILFacetType,
                 veil_timestamp: float,
                 owner_element_id: str,
                 links_to: Optional[str] = None):
        """
        Initialize a VEILFacet.
        
        Args:
            facet_id: Unique identifier for this facet
            facet_type: Type of facet (event, status, ambient)
            veil_timestamp: Connectome system epoch time for temporal ordering
            owner_element_id: ID of the element that owns this facet
            links_to: Optional link to another facet (many-to-one relationship)
        """
        self.facet_id = facet_id
        self.facet_type = facet_type
        self.veil_timestamp = veil_timestamp  # Connectome system epoch time
        self.owner_element_id = owner_element_id
        self.links_to = links_to  # Many-to-one relationship
        self.properties: Dict[str, Any] = {}
        
        logger.debug(f"Created {facet_type.value} facet {facet_id} at timestamp {veil_timestamp}")
        
    @abstractmethod
    def to_veil_dict(self) -> Dict[str, Any]:
        """
        Convert facet to VEIL dictionary representation.
        
        Returns:
            Dictionary representation compatible with existing VEIL consumers
        """
        pass
        
    @abstractmethod
    def get_content_summary(self) -> str:
        """
        Get human-readable content summary for rendering.
        
        Returns:
            Brief summary of facet content for display
        """
        pass
        
    def get_temporal_key(self) -> tuple:
        """
        Get sort key for temporal ordering.
        
        Returns:
            Tuple for chronological sorting (timestamp, facet_id)
        """
        return (self.veil_timestamp, self.facet_id)
    
    def update_properties(self, new_properties: Dict[str, Any]) -> None:
        """
        Update facet properties.
        
        Args:
            new_properties: Dictionary of properties to merge
        """
        self.properties.update(new_properties)
        
    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Get a specific property value.
        
        Args:
            key: Property key
            default: Default value if key not found
            
        Returns:
            Property value or default
        """
        return self.properties.get(key, default)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"{self.__class__.__name__}(id={self.facet_id}, "
                f"type={self.facet_type.value}, timestamp={self.veil_timestamp}, "
                f"owner={self.owner_element_id}, links_to={self.links_to})") 