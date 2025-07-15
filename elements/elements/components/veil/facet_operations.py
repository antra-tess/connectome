"""
VEILFacet Operation Classes

Handles operations on VEIL facets: add, update, and remove.
Replaces current add_node/update_node/remove_node operations with facet-aware operations.
"""

import logging
import time
from typing import Dict, Any, Optional
from .veil_facet import VEILFacet

logger = logging.getLogger(__name__)

class VEILFacetOperation:
    """
    Represents operations on VEIL facets.
    Replaces current add_node/update_node/remove_node operations.
    
    Operation Types:
    - add_facet: Create new facet in cache
    - update_facet: Modify existing facet properties  
    - remove_facet: Delete facet from cache
    """
    
    def __init__(self, 
                 operation_type: str,  # "add_facet", "update_facet", "remove_facet"
                 facet: Optional[VEILFacet] = None,
                 facet_id: Optional[str] = None,
                 property_updates: Optional[Dict[str, Any]] = None):
        """
        Initialize a VEILFacetOperation.
        
        Args:
            operation_type: Type of operation to perform
            facet: VEILFacet instance (for add_facet operations)
            facet_id: Facet ID (for update_facet/remove_facet operations)
            property_updates: Property updates (for update_facet operations)
        """
        self.operation_type = operation_type
        self.facet = facet
        self.facet_id = facet_id or (facet.facet_id if facet else None)
        self.property_updates = property_updates or {}
        self.operation_timestamp = time.time()  # When operation was created
        
        # Validate operation parameters
        self._validate_operation()
        
        logger.debug(f"Created VEILFacetOperation: {operation_type} for facet {self.facet_id}")
        
    def _validate_operation(self) -> None:
        """Validate operation parameters."""
        if self.operation_type not in ["add_facet", "update_facet", "remove_facet"]:
            raise ValueError(f"Invalid operation type: {self.operation_type}")
            
        if self.operation_type == "add_facet" and not self.facet:
            raise ValueError("add_facet operation requires facet parameter")
            
        if self.operation_type in ["update_facet", "remove_facet"] and not self.facet_id:
            raise ValueError(f"{self.operation_type} operation requires facet_id parameter")
            
        if self.operation_type == "update_facet" and not self.property_updates:
            raise ValueError("update_facet operation requires property_updates parameter")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert operation to dictionary representation.
        
        Returns:
            Dictionary representation for serialization/logging
        """
        operation_dict = {
            "operation_type": self.operation_type,
            "facet_id": self.facet_id,
            "operation_timestamp": self.operation_timestamp
        }
        
        if self.operation_type == "add_facet" and self.facet:
            operation_dict["facet_data"] = self.facet.to_veil_dict()
            
        if self.operation_type == "update_facet":
            operation_dict["property_updates"] = self.property_updates.copy()
            
        return operation_dict
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"VEILFacetOperation(type={self.operation_type}, "
                f"facet_id={self.facet_id}, timestamp={self.operation_timestamp})")

class FacetOperationBuilder:
    """
    Builder class for creating VEILFacetOperation instances.
    Provides convenience methods for common operation patterns.
    """
    
    @staticmethod
    def add_facet(facet: VEILFacet) -> VEILFacetOperation:
        """
        Create an add_facet operation.
        
        Args:
            facet: VEILFacet to add to cache
            
        Returns:
            VEILFacetOperation for adding the facet
        """
        return VEILFacetOperation(
            operation_type="add_facet",
            facet=facet
        )
    
    @staticmethod  
    def update_facet(facet_id: str, property_updates: Dict[str, Any]) -> VEILFacetOperation:
        """
        Create an update_facet operation.
        
        Args:
            facet_id: ID of facet to update
            property_updates: Properties to update
            
        Returns:
            VEILFacetOperation for updating the facet
        """
        return VEILFacetOperation(
            operation_type="update_facet",
            facet_id=facet_id,
            property_updates=property_updates
        )
    
    @staticmethod
    def remove_facet(facet_id: str) -> VEILFacetOperation:
        """
        Create a remove_facet operation.
        
        Args:
            facet_id: ID of facet to remove
            
        Returns:
            VEILFacetOperation for removing the facet
        """
        return VEILFacetOperation(
            operation_type="remove_facet",
            facet_id=facet_id
        )
    
    @staticmethod
    def update_facet_content(facet_id: str, new_content: str) -> VEILFacetOperation:
        """
        Create an operation to update facet content.
        
        Args:
            facet_id: ID of facet to update
            new_content: New content value
            
        Returns:
            VEILFacetOperation for updating facet content
        """
        return VEILFacetOperation(
            operation_type="update_facet",
            facet_id=facet_id,
            property_updates={"content": new_content}
        )
    
    @staticmethod
    def update_facet_links(facet_id: str, new_links_to: Optional[str]) -> VEILFacetOperation:
        """
        Create an operation to update facet linking.
        
        Args:
            facet_id: ID of facet to update
            new_links_to: New link target (or None to remove link)
            
        Returns:
            VEILFacetOperation for updating facet links
        """
        return VEILFacetOperation(
            operation_type="update_facet", 
            facet_id=facet_id,
            property_updates={"links_to": new_links_to}
        )

class OperationBatch:
    """
    Container for batching multiple VEILFacetOperations together.
    Useful for atomic updates that need to be processed as a group.
    """
    
    def __init__(self, batch_id: Optional[str] = None):
        """
        Initialize an operation batch.
        
        Args:
            batch_id: Optional identifier for this batch
        """
        self.batch_id = batch_id or f"batch_{int(time.time() * 1000)}"
        self.operations: list[VEILFacetOperation] = []
        self.created_at = time.time()
        
    def add_operation(self, operation: VEILFacetOperation) -> None:
        """
        Add an operation to this batch.
        
        Args:
            operation: VEILFacetOperation to add
        """
        self.operations.append(operation)
        
    def add_facet(self, facet: VEILFacet) -> None:
        """
        Add a facet addition operation to this batch.
        
        Args:
            facet: VEILFacet to add
        """
        self.add_operation(FacetOperationBuilder.add_facet(facet))
        
    def update_facet(self, facet_id: str, property_updates: Dict[str, Any]) -> None:
        """
        Add a facet update operation to this batch.
        
        Args:
            facet_id: ID of facet to update
            property_updates: Properties to update
        """
        self.add_operation(FacetOperationBuilder.update_facet(facet_id, property_updates))
        
    def remove_facet(self, facet_id: str) -> None:
        """
        Add a facet removal operation to this batch.
        
        Args:
            facet_id: ID of facet to remove
        """
        self.add_operation(FacetOperationBuilder.remove_facet(facet_id))
        
    def get_operations(self) -> list[VEILFacetOperation]:
        """
        Get all operations in this batch.
        
        Returns:
            List of VEILFacetOperation instances
        """
        return self.operations.copy()
    
    def clear(self) -> None:
        """Clear all operations from this batch."""
        self.operations.clear()
        
    def __len__(self) -> int:
        """Get number of operations in batch."""
        return len(self.operations)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"OperationBatch(id={self.batch_id}, "
                f"operations={len(self.operations)}, created_at={self.created_at})")

# Utility functions for common operation patterns

def create_message_addition_operations(message_facet: VEILFacet, 
                                     container_facet: Optional[VEILFacet] = None) -> list[VEILFacetOperation]:
    """
    Create operations for adding a new message with optional container creation.
    
    Args:
        message_facet: EventFacet for the message
        container_facet: Optional StatusFacet for container (if new container needed)
        
    Returns:
        List of VEILFacetOperation instances
    """
    operations = []
    
    # Add container first if provided
    if container_facet:
        operations.append(FacetOperationBuilder.add_facet(container_facet))
    
    # Add message facet
    operations.append(FacetOperationBuilder.add_facet(message_facet))
    
    return operations

def create_agent_response_operations(response_facet: VEILFacet,
                                   ambient_facets: Optional[list[VEILFacet]] = None) -> list[VEILFacetOperation]:
    """
    Create operations for adding an agent response with optional ambient updates.
    
    Args:
        response_facet: EventFacet for the agent response
        ambient_facets: Optional list of AmbientFacet instances to update
        
    Returns:
        List of VEILFacetOperation instances
    """
    operations = []
    
    # Add response facet
    operations.append(FacetOperationBuilder.add_facet(response_facet))
    
    # Update ambient facets if provided
    if ambient_facets:
        for ambient_facet in ambient_facets:
            operations.append(FacetOperationBuilder.add_facet(ambient_facet))
    
    return operations

def create_memory_replacement_operations(memory_facet: VEILFacet,
                                       replaced_facet_ids: list[str]) -> list[VEILFacetOperation]:
    """
    Create operations for memory replacement (remove original facets, add memory facet).
    
    Args:
        memory_facet: EventFacet representing the compressed memory
        replaced_facet_ids: List of facet IDs being replaced by the memory
        
    Returns:
        List of VEILFacetOperation instances
    """
    operations = []
    
    # Remove original facets first
    for facet_id in replaced_facet_ids:
        operations.append(FacetOperationBuilder.remove_facet(facet_id))
    
    # Add memory facet
    operations.append(FacetOperationBuilder.add_facet(memory_facet))
    
    return operations 