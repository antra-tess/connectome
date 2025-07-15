"""
Scratchpad VEIL Producer Component
Generates VEIL representation for notes stored in a ScratchpadElement.
"""
import logging
from typing import Dict, Any, Optional, List, Set
import time

# Assuming owner element has a get_notes() method or _notes attribute
from ..base_component import VeilProducer 
# Import the registry decorator
from elements.component_registry import register_component
# NEW: Import VEILFacet system
from ..veil import (
    VEILFacetOperation, VEILFacet, VEILFacetType,
    EventFacet, StatusFacet, AmbientFacet, ConnectomeEpoch,
    FacetOperationBuilder
)

logger = logging.getLogger(__name__)

# NEW: Removed centralized tool families - each VeilProducer sets arbitrary tool_family string

# VEIL Node Structure Constants (Example)
VEIL_SCRATCHPAD_ROOT_TYPE = "scratchpad_root"
VEIL_NOTE_ITEM_TYPE = "note_item"
VEIL_NOTE_CONTENT_PROP = "note_content"
VEIL_NOTE_TIMESTAMP_PROP = "note_timestamp" # From event? Or element state?

@register_component
class ScratchpadVeilProducer(VeilProducer):
    """
    Generates VEIL representation for notes managed by the owning ScratchpadElement.
    
    NEW: Uses VEILFacet architecture generating:
    - StatusFacet for scratchpad container creation/updates
    - EventFacet for note additions and removals
    - Maintains content-based note identification via hash IDs
    """
    COMPONENT_TYPE = "ScratchpadVeilProducer"

    # No specific sibling component dependencies, relies on owner state/methods.

    def initialize(self, **kwargs) -> None:
        """Initializes the component state for delta tracking."""
        super().initialize(**kwargs)
        self._state.setdefault('_last_generated_notes', []) 
        self._state.setdefault('_has_produced_scratchpad_root_add_before', False)
        self._state.setdefault('_last_scratchpad_root_properties', {})
        logger.debug(f"ScratchpadVeilProducer initialized for Element {self.owner.id}")

    def _get_current_notes_from_owner(self) -> List[Any]:
        """Helper to retrieve notes from the owning ScratchpadElement."""
        if not self.owner:
            # Should not happen if component is properly attached
            logger.error(f"[{self.COMPONENT_TYPE}] Owner not set, cannot get notes.")
            return ([], True) # Indicate error
        note_storage = self.get_sibling_component("NoteStorageComponent")
        if not note_storage:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] NoteStorageComponent not found on owner.")
            return ([], True) # Indicate error
        return (note_storage.get_notes_from_state(), False)

    def get_full_veil(self) -> Optional[Dict[str, Any]]:
        """
        Generates the complete VEIL structure for the scratchpad notes.
        """
        current_notes_list, is_error = self._get_current_notes_from_owner()
        if is_error:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Error getting notes from owner. Cannot generate full VEIL.")
            return None # Return None if notes cannot be fetched

        child_veil_nodes = []
        # processed_notes_for_delta = [] # This state is updated in signal_delta_produced_this_frame

        for index, note_content in enumerate(current_notes_list):
            import hashlib
            import time
            from datetime import datetime
            note_hash_id = hashlib.md5(str(note_content).encode()).hexdigest()[:8] 
            
            # Add timestamp for time markers (operation_index handles chronological placement)
            current_timestamp = time.time()
            
            child_node = {
                "veil_id": f"{self.owner.id}_scratchpad_note_{note_hash_id}",
                "node_type": VEIL_NOTE_ITEM_TYPE,
                "properties": {
                    "structural_role": "list_item",
                    "content_nature": "text_note",
                    VEIL_NOTE_CONTENT_PROP: note_content,
                    # NEW: Add timestamp fields for time markers (not chronological placement)
                    "timestamp": current_timestamp,
                    "timestamp_iso": datetime.fromtimestamp(current_timestamp).isoformat() + "Z",
                    "note_timestamp": current_timestamp,
                    "created_at": datetime.fromtimestamp(current_timestamp).isoformat() + "Z"
                    # NOTE: operation_index will be added by SpaceVeilProducer for chronological placement
                },
                "children": [] 
            }
            child_veil_nodes.append(child_node)
            # processed_notes_for_delta.append(note_content) # Don't update here

        if not child_veil_nodes and not current_notes_list: # Ensure placeholder only if truly empty
            placeholder_node = {
                "veil_id": f"{self.owner.id}_scratchpad_empty_placeholder",
                "node_type": "scratchpad_placeholder",
                "properties": {
                    "structural_role": "placeholder",
                    "content_nature": "status_message",
                    "text": "Scratchpad is empty. You can add notes here."
                },
                "children": []
            }
            child_veil_nodes.append(placeholder_node)

        # Create the root container node for the scratchpad
        root_veil_node = {
            "veil_id": f"{self.owner.id}_scratchpad_root", # Consistent root ID
            "node_type": VEIL_SCRATCHPAD_ROOT_TYPE,
            "properties": {
                "structural_role": "container",
                "content_nature": "scratchpad_summary",
                "element_id": self.owner.id,
                "element_name": self.owner.name,
                "note_count": len(current_notes_list) # Count actual notes, not placeholder
            },
            "children": child_veil_nodes
        }
        # Baselines are updated in signal_delta_produced_this_frame
        return root_veil_node

    def calculate_delta(self) -> Optional[List[VEILFacetOperation]]:
        """
        NEW: Calculate VEILFacet operations for scratchpad note management.
        
        This replaces the old delta operation system with VEILFacet operations, generating:
        - StatusFacet for scratchpad container creation/updates
        - EventFacet for note additions and removals
        - Uses content-based hash identification for note tracking
        
        Returns:
            List of VEILFacetOperation instances for the scratchpad
        """
        current_notes_list, is_error = self._get_current_notes_from_owner()
        if is_error:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Error getting notes from owner, cannot calculate facet operations.")
            return None

        if not self.owner:
            logger.error(f"[{self.COMPONENT_TYPE}] Owner not set, cannot calculate facet operations.")
            return None

        facet_operations = []
        owner_id = self.owner.id
        scratchpad_root_facet_id = f"{owner_id}_scratchpad_container"

        # 1. Handle scratchpad container (StatusFacet)
        container_facet_exists = self._state.get('_has_produced_scratchpad_root_add_before', False)
        
        # FIXED: Get enhanced tools for StatusFacet
        enhanced_tools = self._get_enhanced_tools_for_element()
        
        # Get current container state
        current_container_state = {
            "element_id": owner_id,
            "element_name": self.owner.name,
            "note_count": len(current_notes_list),
            "content_nature": "scratchpad_summary",
            # FIXED: Include enhanced tools in StatusFacet current_state
            "available_tools": enhanced_tools
        }
        
        if not container_facet_exists:
            # Create container StatusFacet
            container_facet = StatusFacet(
                facet_id=scratchpad_root_facet_id,
                veil_timestamp=ConnectomeEpoch.get_veil_timestamp(),
                owner_element_id=owner_id,
                status_type="container_created",
                current_state=current_container_state,
                links_to=f"{self.owner.get_parent_info()['parent_id']}_space_root" if hasattr(self.owner, 'get_parent_info') else None
            )
            
            facet_operations.append(FacetOperationBuilder.add_facet(container_facet))
            logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Generated add_facet for scratchpad container {scratchpad_root_facet_id}")
            
        else:
            # Check for container state updates
            last_container_state = self._state.get('_last_scratchpad_root_properties', {})
            if current_container_state != last_container_state:
                facet_operations.append(
                    FacetOperationBuilder.update_facet(
                        scratchpad_root_facet_id,
                        {"current_state": current_container_state}
                    )
                )
                logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Generated update_facet for scratchpad container {scratchpad_root_facet_id}")

        # 2. Handle note additions and removals (EventFacets)
        last_notes_list = self._state.get('_last_generated_notes', [])
        current_notes_str_set = {str(n) for n in current_notes_list}
        last_notes_str_set = {str(n) for n in last_notes_list}

        added_notes_str_content = current_notes_str_set - last_notes_str_set
        removed_notes_str_content = last_notes_str_set - current_notes_str_set
        
        notes_to_add_map = {str(n): n for n in current_notes_list}
        notes_to_remove_map = {str(n): n for n in last_notes_list}

        # Handle note additions
        for note_str_content in added_notes_str_content:
            original_note_content = notes_to_add_map.get(note_str_content)
            if original_note_content is None: 
                continue
                
            # Generate hash-based ID for content tracking
            import hashlib
            note_hash_id = hashlib.md5(str(original_note_content).encode()).hexdigest()[:8]
            note_facet_id = f"{owner_id}_scratchpad_note_{note_hash_id}"
            
            # Create EventFacet for new note
            note_facet = EventFacet(
                facet_id=note_facet_id,
                veil_timestamp=ConnectomeEpoch.get_veil_timestamp(),
                owner_element_id=owner_id,
                event_type="note_created",
                content=str(original_note_content),
                links_to=scratchpad_root_facet_id
            )
            
            # Add note-specific properties
            current_timestamp = time.time()
            note_facet.properties.update({
                "note_hash_id": note_hash_id,
                "content_nature": "text_note",
                "structural_role": "list_item",
                "timestamp": current_timestamp,
                "timestamp_iso": time.strftime('%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime(current_timestamp)),
                "note_timestamp": current_timestamp,
                "created_at": time.strftime('%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime(current_timestamp))
            })
            
            facet_operations.append(FacetOperationBuilder.add_facet(note_facet))
            logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Generated add_facet for note {note_facet_id}")

        # Handle note removals
        for note_str_content in removed_notes_str_content:
            original_note_content = notes_to_remove_map.get(note_str_content)
            if original_note_content is None:
                continue
                
            # Generate hash-based ID for removal
            import hashlib
            note_hash_id = hashlib.md5(str(original_note_content).encode()).hexdigest()[:8]
            note_facet_id = f"{owner_id}_scratchpad_note_{note_hash_id}"
            
            facet_operations.append(FacetOperationBuilder.remove_facet(note_facet_id))
            logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Generated remove_facet for note {note_facet_id}")

        # 3. Generate tool availability ambient facets (when appropriate)
        if self._should_emit_tools_ambient_facet():
            tools_ambient_facet = self._create_tools_ambient_facet()
            if tools_ambient_facet:
                facet_operations.append(FacetOperationBuilder.add_facet(tools_ambient_facet))
                logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Generated structured ambient facet for scratchpad tools")

        # Update state after generating operations
        if not container_facet_exists and any(
            op.operation_type == "add_facet" and 
            op.facet and op.facet.facet_id == scratchpad_root_facet_id 
            for op in facet_operations
        ):
            self._state['_has_produced_scratchpad_root_add_before'] = True

        self._state['_last_scratchpad_root_properties'] = current_container_state
        self._state['_last_generated_notes'] = current_notes_list

        if facet_operations:
            logger.info(f"[{owner_id}/{self.COMPONENT_TYPE}] Calculated {len(facet_operations)} scratchpad facet operations")
        else:
            logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] No scratchpad facet operations calculated")

        return facet_operations if facet_operations else None

    # --- NEW: Enhanced Structured Ambient Facet Methods ---
    
    def _get_available_tools_for_element(self) -> List[str]:
        """Get list of available tool names for this scratchpad element."""
        from ..tool_provider import ToolProviderComponent

        tool_provider = self.get_sibling_component(ToolProviderComponent)
        if tool_provider:
            return tool_provider.list_tools()
        return []

    def _get_enhanced_tools_for_element(self) -> List[Dict[str, Any]]:
        """
        Get enhanced tool definitions with complete metadata for scratchpad element.
        
        Returns rich tool information needed for tool aggregation and rendering.
        """
        from ..tool_provider import ToolProviderComponent

        tool_provider = self.get_sibling_component(ToolProviderComponent)
        if tool_provider:
            return tool_provider.get_enhanced_tool_definitions()
        return []
    
    def _should_emit_tools_ambient_facet(self) -> bool:
        """
        Determine whether to emit tools ambient facet for this scratchpad element.
        
        Returns:
            True if tools ambient facet should be emitted
        """
        enhanced_tools = self._get_enhanced_tools_for_element()
        return bool(enhanced_tools)
    
    def _create_tools_ambient_facet(self) -> Optional[AmbientFacet]:
        """
        Create enhanced AmbientFacet for available scratchpad tools with structured data.
        
        This creates structured data that HUD can consolidate and render appropriately,
        rather than pre-rendered strings.
        
        Returns:
            AmbientFacet with structured tool data for HUD consolidation
        """
        enhanced_tools = self._get_enhanced_tools_for_element()
        if not enhanced_tools:
            return None
        
        # Determine tool family for this element
        tool_family = self._classify_tool_family(enhanced_tools)
        
        # Create structured content instead of pre-rendered strings
        structured_content = {
            "tools": enhanced_tools,
            "element_context": self._get_element_context_metadata(),
            "tool_family": tool_family
        }
        
        ambient_facet = AmbientFacet(
            facet_id=f"{self.owner.id}_tools_ambient",
            owner_element_id=self.owner.id,
            ambient_type=tool_family,  # Tool family classification for HUD grouping
            content=structured_content,  # Structured data instead of string
            trigger_threshold=1500  # Element-specific threshold
        )
        
        # Add additional properties for HUD processing
        ambient_facet.properties.update({
            "data_format": "structured",
            "tools_count": len(enhanced_tools),
            "element_type": "scratchpad"
        })
        
        return ambient_facet
    
    def _classify_tool_family(self, enhanced_tools: List[Dict[str, Any]]) -> str:
        """
        Get tool family for this scratchpad element.
        
        Args:
            enhanced_tools: List of enhanced tool definitions (unused - family is element-based)
            
        Returns:
            Tool family string for this element type
        """
        # Each element type sets its own arbitrary tool_family string
        # Scratchpad elements use "scratchpad_tools" by default
        return "scratchpad_tools"
    
    def _get_element_context_metadata(self) -> Dict[str, Any]:
        """
        Get element context metadata for HUD consolidation.
        
        Returns:
            Dictionary with element context information
        """
        return {
            "element_id": self.owner.id,
            "element_name": self.owner.name,
            "element_type": "scratchpad",
            "note_count": len(self._state.get('_last_generated_notes', [])),
            "content_nature": "scratchpad_summary"
        }
