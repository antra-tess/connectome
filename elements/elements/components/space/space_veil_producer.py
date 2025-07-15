"""
Space VEIL Producer Component
Generates VEILFacet representation for Space elements using the new VEILFacet architecture.
"""
import logging
from typing import Dict, Any, Optional, List, Set
import copy
import time

from ..base_component import VeilProducer
from elements.component_registry import register_component
from ..veil import (
    VEILFacetCache, VEILFacetOperation, VEILFacet, VEILFacetType,
    EventFacet, StatusFacet, AmbientFacet, ConnectomeEpoch,
    FacetOperationBuilder
)
from ..veil.facet_types import create_agent_response_facet

logger = logging.getLogger(__name__)

@register_component
class SpaceVeilProducer(VeilProducer):
    """
    NEW: Pure VEILFacet Space producer.
    
    Generates VEILFacet operations for Space elements:
    - StatusFacet for space creation/updates
    - Coordinates with child VeilProducers through VEILFacetCache
    - Provides facet-based VEIL context for HUD rendering
    """
    COMPONENT_TYPE = "SpaceVeilProducer"

    def initialize(self, **kwargs) -> None:
        """Initialize the VEILFacet-based space producer."""
        super().initialize(**kwargs)
        
        # VEILFacet state tracking
        self._state.setdefault('_last_space_properties', {})
        self._state.setdefault('_has_produced_space_root_facet', False)
        
        # VEILFacet system - primary cache
        self._facet_cache = VEILFacetCache()
        self._accumulated_facet_operations: List[VEILFacetOperation] = []
        
        # Initialize ConnectomeEpoch on first producer initialization
        ConnectomeEpoch.initialize()
        
        logger.debug(f"SpaceVeilProducer initialized for Element {self.owner.id if self.owner else 'Unknown'} with pure VEILFacet architecture")

    async def receive_facet_operations(self, operations: List[VEILFacetOperation]) -> None:
        """
        Process VEILFacet operations from child producers.
        
        This is the primary entry point for VEIL management in the new architecture.
        
        Args:
            operations: List of VEILFacetOperation instances to process
        """
        if not isinstance(operations, list) or not operations:
            logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] SpaceVeilProducer received no valid facet operations")
            return

        # Apply operations to facet cache
        success_count = self._facet_cache.apply_operations(operations)
        
        # Accumulate for frame-end dispatch to uplinks
        self._accumulated_facet_operations.extend(operations)
        
        logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] Processed {success_count}/{len(operations)} facet operations in SpaceVeilProducer")

    def get_facet_cache(self) -> VEILFacetCache:
        """
        Get the VEILFacetCache for direct access.
        
        This is the primary interface for HUD and other components to access facet data.
        
        Returns:
            VEILFacetCache instance with current facets
        """
        return self._facet_cache
    
    def get_flat_veil_cache(self) -> VEILFacetCache:
        return self.get_facet_cache()
        
    def get_facet_cache_copy(self) -> VEILFacetCache:
        """
        Get a deep copy of the VEILFacetCache.
        
        Returns:
            Deep copy of VEILFacetCache to prevent external modifications
        """
        return self._facet_cache.copy()
        
    def get_accumulated_facet_operations(self) -> List[VEILFacetOperation]:
        """
        Get and clear accumulated facet operations for frame-end dispatch.
        
        Returns:
            List of accumulated facet operations, clearing the internal accumulator
        """
        operations = list(self._accumulated_facet_operations)
        self._accumulated_facet_operations.clear()
        return operations
        
    def clear_facet_cache(self) -> None:
        """
        Clear the VEILFacetCache.
        
        Used during replay and regeneration scenarios.
        """
        self._facet_cache.clear()
        logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] VEILFacetCache cleared")
        
    def get_facet_cache_statistics(self) -> Dict[str, Any]:
        """
        Get VEILFacetCache statistics for debugging and monitoring.
        
        Returns:
            Dictionary with cache statistics
        """
        return self._facet_cache.get_cache_statistics()

    def clear_facet_cache(self) -> None:
        """
        Clear the VEILFacetCache.
        
        Used during replay and regeneration scenarios.
        """
        self._facet_cache.clear()

    def calculate_delta(self) -> Optional[List[VEILFacetOperation]]:
        """
        NEW: Calculate VEILFacet operations for Space root management.
        
        This replaces the old delta operation system with VEILFacet operations, generating:
        - StatusFacet for space creation/updates
        - Uses VEILFacet temporal positioning
        
        Returns:
            List of VEILFacetOperation instances for the space
        """
        if not self.owner:
            logger.error(f"[{self.COMPONENT_TYPE}] Owner not set, cannot calculate facet operations.")
            return None

        owner_id = self.owner.id
        facet_operations = []
        space_root_facet_id = f"{owner_id}_space_root"
        current_space_props = self._get_current_space_properties()

        # Check if space root facet exists in facet cache
        root_facet_exists = self.has_facet(space_root_facet_id)
        has_produced_flag = self._state.get('_has_produced_space_root_facet', False)

        # Generate add_facet if either: never produced before OR root missing from facet cache
        if not has_produced_flag or not root_facet_exists:
            if not root_facet_exists:
                logger.warning(f"[{owner_id}/{self.COMPONENT_TYPE}] Space root facet '{space_root_facet_id}' missing from cache, regenerating add_facet operation")
            else:
                logger.info(f"[{owner_id}/{self.COMPONENT_TYPE}] Generating initial add_facet for Space root '{space_root_facet_id}'")

            # Create space root as StatusFacet
            space_root_facet = StatusFacet(
                facet_id=space_root_facet_id,
                veil_timestamp=ConnectomeEpoch.get_veil_timestamp(),
                owner_element_id=owner_id,
                status_type="space_created",
                current_state=current_space_props.copy()
            )
            
            facet_operations.append(FacetOperationBuilder.add_facet(space_root_facet))
            logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Generated add_facet for Space root {space_root_facet_id}")
            
        else:
            # Root exists in facet cache, check for property updates
            last_space_props = self._state.get('_last_space_properties', {})
            if current_space_props != last_space_props:
                logger.info(f"[{owner_id}/{self.COMPONENT_TYPE}] Generating update_facet for Space root properties")
                
                update_operation = FacetOperationBuilder.update_facet(
                    space_root_facet_id,
                    {"current_state": current_space_props.copy()}
                )
                facet_operations.append(update_operation)

        # Update state after generating operations
        if not has_produced_flag and any(
            op.operation_type == "add_facet" and 
            op.facet and op.facet.facet_id == space_root_facet_id 
            for op in facet_operations
        ):
            self._state['_has_produced_space_root_facet'] = True

        self._state['_last_space_properties'] = copy.deepcopy(current_space_props)

        if facet_operations:
            logger.info(f"[{owner_id}/{self.COMPONENT_TYPE}] Calculated {len(facet_operations)} space facet operations")
        else:
            logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] No space facet operations calculated")

        return facet_operations if facet_operations else None

    def _get_current_space_properties(self) -> Dict[str, Any]:
        """Extract properties of the Space element itself for its VEIL root StatusFacet."""
        if not self.owner:
            logger.error(f"[{self.COMPONENT_TYPE}] Owner not set, cannot get space properties.")
            return {}
            
        props = {
            "structural_role": "root",
            "content_nature": "space_summary", 
            "element_id": self.owner.id,
            "element_name": self.owner.name,
            "element_type": self.owner.__class__.__name__,
            "is_inner_space": getattr(self.owner, 'IS_INNER_SPACE', False),
            "is_uplink_space": getattr(self.owner, 'IS_UPLINK_SPACE', False)
        }
        
        # Add agent-specific properties if available
        if hasattr(self.owner, 'agent_description'):
            props['agent_description'] = self.owner.agent_description
        if hasattr(self.owner, 'agent_name'):
            props['agent_name'] = self.owner.agent_name
        if hasattr(self.owner, 'adapter_id'):
            props['adapter_id'] = self.owner.adapter_id
        if hasattr(self.owner, 'external_conversation_id'):
            props['external_conversation_id'] = self.owner.external_conversation_id
            
        return props

    # REMOVED: Legacy flat cache compatibility methods (get_flat_veil_cache, _convert_facet_to_flat_node)
    # No longer needed after completing VEILFacet architecture transition

    def get_facet_cache_size(self) -> int:
        """
        Get the size of the facet cache.
        
        Returns:
            Number of facets in the cache
        """
        return len(self._facet_cache.facets)

    def has_multimodal_content(self, owner_id: Optional[str] = None) -> bool:
        """
        Check if the facet cache contains multimodal content.
        
        Args:
            owner_id: Optional owner ID to filter by
            
        Returns:
            True if multimodal content is found, False otherwise
        """
        # Check for EventFacets with multimodal properties
        for facet in self._facet_cache.facets.values():
            if owner_id and facet.owner_element_id != owner_id:
                continue
                
            if facet.facet_type == VEILFacetType.EVENT:
                # Check for attachment or multimodal indicators
                if facet.get_property("has_attachments") or facet.get_property("multimodal_content"):
                    return True
                    
        return False

    def get_facets_by_owner(self, owner_id: str) -> Dict[str, VEILFacet]:
        """
        Get all facets belonging to a specific owner element.
        
        Args:
            owner_id: Element ID to filter by
            
        Returns:
            Dictionary of {facet_id: VEILFacet} for facets owned by the specified element
        """
        filtered_facets = {}
        
        for facet_id, facet in self._facet_cache.facets.items():
            if facet.owner_element_id == owner_id:
                filtered_facets[facet_id] = facet
        
        logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] Filtered {len(filtered_facets)} facets for owner {owner_id}")
        return filtered_facets

    def has_facet(self, facet_id: str) -> bool:
        """
        Check if a facet exists in the cache.
        
        Args:
            facet_id: Facet ID to check
            
        Returns:
            True if the facet exists, False otherwise
        """
        return self._facet_cache.get_facet_or_none(facet_id) is not None

    # --- REMOVED: All legacy delta operation methods ---
    # - receive_delta_operations()
    # - _apply_deltas_to_flat_cache()
    # - _inject_operation_index_into_cache()
    # - get_accumulated_deltas()
    # - build_hierarchical_veil_from_flat_cache()
    # - reconstruct_veil_state_at_delta_index()
    # - render_state_with_future_edits()
    # - All flat cache management methods
    
    # The VEILFacet architecture replaces all of this complexity with clean facet operations

    def emit_agent_response(self, 
                           agent_response_text: str, 
                           tool_calls_data: Optional[List[Dict[str, Any]]] = None,
                           agent_loop_component_id: Optional[str] = None,
                           parsing_mode: str = "text",
                           links_to: Optional[str] = None) -> str:
        """
        Centralized agent response VEILFacet Event creation for reuse across agent loop implementations.
        
        This method handles the creation and emission of agent response EventFacets,
        providing a consistent interface for all agent loop types.
        
        Args:
            agent_response_text: The agent's full text response from LLM
            tool_calls_data: Optional list of tool call dictionaries
            agent_loop_component_id: ID of the agent loop component that generated the response
            parsing_mode: The parsing mode used ("text", "tool_call", etc.)
            links_to: Optional container to link the response to
            
        Returns:
            The response_id of the created agent response facet
        """
        try:
            if not self.owner:
                logger.error("SpaceVeilProducer has no owner element, cannot emit agent response")
                return ""
                
            current_time = time.time()
            
            # Create unique response ID for this agent response
            response_id = f"agent_response_{self.owner.id}_{int(current_time * 1000)}"
            
            # Auto-link to space root if no explicit links_to provided
            if links_to is None:
                space_root_facet_id = f"{self.owner.id}_space_root"
                links_to = space_root_facet_id
                logger.debug(f"Auto-linking agent response {response_id} to space root {space_root_facet_id}")
            
            # Create agent response EventFacet using utility function
            agent_response_facet = create_agent_response_facet(
                response_id=response_id,
                owner_element_id=self.owner.id,
                content=agent_response_text,
                tool_calls=tool_calls_data or [],
                links_to=links_to
            )
            
            # Add additional properties specific to this agent loop
            agent_response_facet.properties.update({
                "parsing_mode": parsing_mode,
                "operation_index": int(current_time * 1000),  # For chronological ordering
                "timestamp": current_time
            })
            
            # Add agent loop component ID if provided
            if agent_loop_component_id:
                agent_response_facet.properties["agent_loop_component_id"] = agent_loop_component_id
            
            # Create VEILFacetOperation
            facet_operation = FacetOperationBuilder.add_facet(agent_response_facet)
            
            # Apply operation to our facet cache
            self._facet_cache.apply_operations([facet_operation])
            
            # Accumulate for frame-end dispatch to uplinks
            self._accumulated_facet_operations.append(facet_operation)
            
            logger.debug(f"Emitted agent response VEILFacet Event {response_id} with {len(tool_calls_data or [])} tool calls")
            
            return response_id
                
        except Exception as e:
            logger.error(f"Error emitting agent response VEILFacet Event: {e}", exc_info=True)
            return ""
