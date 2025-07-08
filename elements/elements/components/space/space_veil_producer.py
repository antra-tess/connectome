import logging
from typing import Dict, Any, Optional, List, Set
import copy
import time

from ..base_component import VeilProducer
from elements.component_registry import register_component
# Assuming your Space class is in elements.elements.space
# Adjust the import if your Space class is located elsewhere.
# from ...space import Space # This was in the prior context, might need adjustment

logger = logging.getLogger(__name__)

VEIL_SPACE_ROOT_TYPE = "space_root"

@register_component
class SpaceVeilProducer(VeilProducer):
    """
    ENHANCED: Complete owner of all VEIL operations including:
    - Flat VEIL cache management
    - Delta history tracking
    - Temporal state reconstruction
    - VEIL rendering and context generation
    """
    COMPONENT_TYPE = "SpaceVeilProducer"

    def initialize(self, **kwargs) -> None:
        """Initializes the component state for delta tracking for the Space's root node."""
        super().initialize(**kwargs)
        self._state.setdefault('_last_space_properties', {})
        self._state.setdefault('_has_produced_root_add_before', False)
        
        # MOVED FROM SPACE: Complete VEIL state ownership
        self._flat_veil_cache: Dict[str, Any] = {}
        self._delta_history: List[Dict[str, Any]] = []  # Chronological delta storage
        self._next_delta_index = 0
        self._accumulated_deltas = []  # For frame-end dispatch
        
        logger.debug(f"SpaceVeilProducer initialized for Element {self.owner.id if self.owner else 'Unknown'}")

    async def receive_delta_operations(self, delta_operations: List[Dict[str, Any]]) -> None:
        """
        ENHANCED: Complete delta processing pipeline.
        
        This replaces both Space.receive_delta() and Space._apply_deltas_to_internal_cache()
        """
        if not isinstance(delta_operations, list) or not delta_operations:
            logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] SpaceVeilProducer received no valid delta operations")
            return

        # Add operation timestamps and indices
        timestamped_deltas = []
        current_time = time.time()
        current_iso = time.strftime('%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime(current_time))
        
        for delta in delta_operations:
            # CRITICAL FIX: Deep copy to prevent data contamination
            timestamped_delta = copy.deepcopy(delta)
            timestamped_delta["operation_timestamp"] = current_time
            timestamped_delta["operation_timestamp_iso"] = current_iso  # NEW: ISO format for system messages
            timestamped_delta["operation_index"] = self._next_delta_index
            timestamped_deltas.append(timestamped_delta)
            self._next_delta_index += 1
        
        # CRITICAL FIX: Store deep copies in delta history for temporal reconstruction
        self._delta_history.extend(copy.deepcopy(timestamped_deltas))
        
        # Apply to flat VEIL cache (moved from Space)
        self._apply_deltas_to_flat_cache(timestamped_deltas)
        
        # CRITICAL: Inject operation_index into all nodes for chronological rendering
        self._inject_operation_index_into_cache(timestamped_deltas)
        
        # Accumulate for frame-end dispatch
        self._accumulated_deltas.extend(timestamped_deltas)
        
        logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] Processed {len(timestamped_deltas)} delta operations in SpaceVeilProducer")
    
    def _apply_deltas_to_flat_cache(self, deltas: List[Dict[str, Any]]) -> None:
        """
        MOVED FROM SPACE: Apply VEIL deltas to flat cache.
        
        This centralizes all flat cache management in SpaceVeilProducer.
        """
        try:
            for delta in deltas:
                op = delta.get("op")
                
                if op == "add_node":
                    node = delta.get("node", {})
                    veil_id = node.get("veil_id")
                    if veil_id:
                        # Ensure 'properties' dictionary exists in the node data
                        if "properties" not in node or not isinstance(node.get("properties"), dict):
                            node["properties"] = {}
                        
                        # If parent_id is provided at the top level of the delta op, inject it into node's properties
                        top_level_parent_id = delta.get("parent_id")
                        if top_level_parent_id:
                            node["properties"]["parent_id"] = top_level_parent_id
                            logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] Cache: Injecting parent_id '{top_level_parent_id}' into properties of node '{veil_id}' during add_node.")
                        
                        if "children" not in node or not isinstance(node.get("children"), list):
                            node["children"] = []
                        
                        # CRITICAL FIX: Deep copy to prevent reference sharing and data contamination
                        self._flat_veil_cache[veil_id] = copy.deepcopy(node)
                        
                elif op == "update_node":
                    veil_id = delta.get("veil_id")
                    if veil_id and veil_id in self._flat_veil_cache:
                        # Ensure 'properties' dictionary exists in the cached node data
                        if "properties" not in self._flat_veil_cache[veil_id] or \
                           not isinstance(self._flat_veil_cache[veil_id].get("properties"), dict):
                             self._flat_veil_cache[veil_id]["properties"] = {}

                        # Apply property updates from the delta
                        properties_to_update = delta.get("properties")
                        if properties_to_update is not None:
                            # CRITICAL FIX: Deep copy properties to prevent reference sharing
                            self._flat_veil_cache[veil_id]["properties"].update(copy.deepcopy(properties_to_update))
                        
                        # If parent_id is provided at the top level of the update_node op, update it in node's properties
                        top_level_parent_id = delta.get("parent_id")
                        if top_level_parent_id:
                            self._flat_veil_cache[veil_id]["properties"]["parent_id"] = top_level_parent_id
                            logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] Cache: Injecting/updating parent_id '{top_level_parent_id}' into properties of node '{veil_id}' during update_node.")
                        
                elif op == "remove_node":
                    veil_id = delta.get("veil_id")
                    if veil_id and veil_id in self._flat_veil_cache:
                        logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] Cache: Removing node {veil_id} from _flat_veil_cache.")
                        del self._flat_veil_cache[veil_id]
                        
            logger.debug(f"Applied {len(deltas)} deltas to flat cache: {len(self._flat_veil_cache)} total nodes")
            
        except Exception as e:
            logger.error(f"Error applying deltas to flat cache: {e}", exc_info=True)
    
    def _inject_operation_index_into_cache(self, deltas: List[Dict[str, Any]]) -> None:
        """
        CRITICAL: Inject operation_index into flat cache nodes for chronological rendering.
        
        This ensures all VEIL nodes have operation_index for proper operational chronology
        as required by the HUD rendering refactor (uses operation_index, not timestamps).
        
        Args:
            deltas: Timestamped deltas with operation_index
        """
        try:
            for delta in deltas:
                op = delta.get("op")
                operation_index = delta.get("operation_index")
                
                if operation_index is None:
                    continue
                
                if op == "add_node":
                    node = delta.get("node", {})
                    veil_id = node.get("veil_id")
                    if veil_id and veil_id in self._flat_veil_cache:
                        # Inject operation_index into node properties
                        if "properties" not in self._flat_veil_cache[veil_id]:
                            self._flat_veil_cache[veil_id]["properties"] = {}
                        self._flat_veil_cache[veil_id]["properties"]["operation_index"] = operation_index
                        
                elif op == "update_node":
                    veil_id = delta.get("veil_id")
                    if veil_id and veil_id in self._flat_veil_cache:
                        # Update operation_index to reflect latest operation
                        if "properties" not in self._flat_veil_cache[veil_id]:
                            self._flat_veil_cache[veil_id]["properties"] = {}
                        self._flat_veil_cache[veil_id]["properties"]["operation_index"] = operation_index
                
                # Note: remove_node doesn't need operation_index injection since node is deleted
                        
            logger.debug(f"Injected operation_index into {len(deltas)} cache nodes for chronological rendering")
            
        except Exception as e:
            logger.error(f"Error injecting operation_index into cache: {e}", exc_info=True)

    def get_flat_veil_cache(self) -> Dict[str, Any]:
        """
        NEW: Expose flat VEIL cache with proper encapsulation.
        
        Returns deep copy to prevent external modifications.
        """
        return copy.deepcopy(self._flat_veil_cache)

    def clear_flat_veil_cache(self) -> None:
        """
        NEW: Clear the flat VEIL cache.
        
        Used during replay and regeneration scenarios.
        """
        self._flat_veil_cache.clear()
        logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] Flat VEIL cache cleared")

    def update_flat_veil_cache(self, cache_data: Dict[str, Any]) -> None:
        """
        NEW: Update the flat VEIL cache with new data.
        
        Used during snapshot restoration.
        
        Args:
            cache_data: Dictionary of veil_id -> node_data to update the cache with
        """
        self._flat_veil_cache.clear()
        self._flat_veil_cache.update(copy.deepcopy(cache_data))
        logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] Flat VEIL cache updated with {len(cache_data)} nodes")

    def get_flat_veil_cache_size(self) -> int:
        """
        NEW: Get the size of the flat VEIL cache.
        
        Returns:
            Number of nodes in the flat VEIL cache
        """
        return len(self._flat_veil_cache)

    def get_veil_nodes_by_owner(self, owner_id: str) -> Dict[str, Any]:
        """
        NEW: Get all VEIL nodes belonging to a specific owner element.
        
        Args:
            owner_id: Element ID to filter by
            
        Returns:
            Dictionary of {veil_id: veil_node} for nodes owned by the specified element
        """
        filtered_nodes = {}
        
        for veil_id, node_data in self._flat_veil_cache.items():
            if isinstance(node_data, dict):
                props = node_data.get("properties", {})
                if props.get("owner_id") == owner_id:
                    filtered_nodes[veil_id] = copy.deepcopy(node_data)
        
        logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] Filtered {len(filtered_nodes)} VEIL nodes for owner {owner_id}")
        return filtered_nodes

    def get_veil_nodes_by_type(self, node_type: str, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """
        NEW: Get all VEIL nodes of a specific type, optionally filtered by owner.
        
        Args:
            node_type: VEIL node type to filter by
            owner_id: Optional owner ID to further filter by
            
        Returns:
            Dictionary of {veil_id: veil_node} matching the criteria
        """
        filtered_nodes = {}
        
        for veil_id, node_data in self._flat_veil_cache.items():
            if isinstance(node_data, dict):
                if node_data.get("node_type") == node_type:
                    # If owner_id is specified, also filter by owner
                    if owner_id is None:
                        filtered_nodes[veil_id] = copy.deepcopy(node_data)
                    else:
                        props = node_data.get("properties", {})
                        if props.get("owner_id") == owner_id:
                            filtered_nodes[veil_id] = copy.deepcopy(node_data)
        
        logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] Filtered {len(filtered_nodes)} VEIL nodes of type '{node_type}'" + 
                    (f" for owner {owner_id}" if owner_id else ""))
        return filtered_nodes

    def get_veil_nodes_by_content_nature(self, content_nature: str, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """
        NEW: Get all VEIL nodes with specific content_nature, optionally filtered by owner.
        
        Args:
            content_nature: Content nature to filter by (e.g., "chat_message", "attachment_content")
            owner_id: Optional owner ID to further filter by
            
        Returns:
            Dictionary of {veil_id: veil_node} matching the criteria
        """
        filtered_nodes = {}
        
        for veil_id, node_data in self._flat_veil_cache.items():
            if isinstance(node_data, dict):
                props = node_data.get("properties", {})
                if props.get("content_nature") == content_nature:
                    # If owner_id is specified, also filter by owner
                    if owner_id is None:
                        filtered_nodes[veil_id] = copy.deepcopy(node_data)
                    else:
                        if props.get("owner_id") == owner_id:
                            filtered_nodes[veil_id] = copy.deepcopy(node_data)
        
        logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] Filtered {len(filtered_nodes)} VEIL nodes with content_nature '{content_nature}'" + 
                    (f" for owner {owner_id}" if owner_id else ""))
        return filtered_nodes

    def has_multimodal_content(self, owner_id: Optional[str] = None) -> bool:
        """
        NEW: Check if the cache contains multimodal content (attachment nodes with content).
        
        Args:
            owner_id: Optional owner ID to filter by
            
        Returns:
            True if multimodal content is found, False otherwise
        """
        for veil_id, node_data in self._flat_veil_cache.items():
            if isinstance(node_data, dict):
                props = node_data.get("properties", {})
                
                # Check if it's an attachment content node
                if (props.get("content_nature", "").startswith("image") or 
                    props.get("structural_role") == "attachment_content" or
                    node_data.get("node_type") == "attachment_content_item"):
                    
                    # If owner_id filter is specified, check ownership
                    if owner_id is None or props.get("owner_id") == owner_id:
                        logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] Found multimodal content in node {veil_id}")
                        return True
        
        return False

    def check_node_exists(self, veil_id: str) -> bool:
        """
        NEW: Check if a specific VEIL node exists in the cache.
        
        Args:
            veil_id: The VEIL ID to check for
            
        Returns:
            True if the node exists, False otherwise
        """
        return veil_id in self._flat_veil_cache

    async def reconstruct_veil_state_at_delta_index(self, target_delta_index: int) -> Dict[str, Any]:
        """
        Reconstruct flat VEIL cache at specific point in timeline.
        
        Uses internal delta history for temporal reconstruction.
        """
        try:
            # Get deltas up to target index from internal history
            historical_deltas = [
                delta for delta in self._delta_history 
                if delta.get("operation_index", 0) <= target_delta_index
            ]
            
            # Reconstruct state by applying deltas chronologically
            reconstructed_cache = {}
            for delta in historical_deltas:
                reconstructed_cache = self._apply_delta_to_cache(reconstructed_cache, delta)
            
            logger.debug(f"Reconstructed VEIL state at delta {target_delta_index}: {len(reconstructed_cache)} nodes")
            return reconstructed_cache
            
        except Exception as e:
            logger.error(f"Error reconstructing VEIL state at delta {target_delta_index}: {e}", exc_info=True)
            return {}

    async def render_state_with_future_edits(self, 
                                           base_state: Dict[str, Any], 
                                           base_delta_index: int) -> Dict[str, Any]:
        """
        Apply edit/delete deltas that occurred after base_delta_index.
        
        Shows "how this moment in history looked after all edits".
        """
        try:
            # Get all deltas after the base index from internal history
            future_deltas = [
                delta for delta in self._delta_history 
                if delta.get("operation_index", 0) > base_delta_index
            ]
            
            # Filter to only edit/delete operations affecting base state content
            relevant_edits = self._filter_relevant_edit_deltas(future_deltas, base_state)
            
            # CRITICAL FIX: Deep copy base state to prevent contamination of original data
            edited_state = copy.deepcopy(base_state)
            for edit_delta in relevant_edits:
                if edit_delta["op"] in ["update_node", "remove_node"]:
                    edited_state = self._apply_delta_to_cache(edited_state, edit_delta)
            
            logger.debug(f"Applied {len(relevant_edits)} future edits to base state")
            return edited_state
            
        except Exception as e:
            logger.error(f"Error applying future edits: {e}", exc_info=True)
            return base_state

    def _apply_delta_to_cache(self, cache: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a single delta operation to a cache.
        
        Used for both current state updates and temporal reconstruction.
        """
        try:
            op = delta.get("op")
            
            if op == "add_node":
                node = delta.get("node", {})
                veil_id = node.get("veil_id")
                if veil_id:
                    # CRITICAL FIX: Deep copy to prevent reference sharing and data contamination
                    cache[veil_id] = copy.deepcopy(node)
                    
            elif op == "update_node":
                veil_id = delta.get("veil_id")
                if veil_id and veil_id in cache:
                    properties = delta.get("properties", {})
                    if properties:
                        existing_props = cache[veil_id].get("properties", {})
                        # CRITICAL FIX: Deep copy properties to prevent reference sharing
                        existing_props.update(copy.deepcopy(properties))
                        
            elif op == "remove_node":
                veil_id = delta.get("veil_id")
                if veil_id and veil_id in cache:
                    del cache[veil_id]
                    
            return cache
            
        except Exception as e:
            logger.error(f"Error applying delta to cache: {e}", exc_info=True)
            return cache

    def _filter_relevant_edit_deltas(self, deltas: List[Dict[str, Any]], base_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter deltas to only those that affect nodes in the base state.
        
        This ensures we only apply edits that are relevant to the content being reconstructed.
        """
        relevant_deltas = []
        base_veil_ids = set(base_state.keys())
        
        for delta in deltas:
            if delta.get("op") in ["update_node", "remove_node"]:
                veil_id = delta.get("veil_id")
                if veil_id in base_veil_ids:
                    relevant_deltas.append(delta)
        
        return relevant_deltas

    async def render_temporal_context_for_compression(self, 
                                                    element_id: str, 
                                                    memory_formation_index: int) -> str:
        """
        Render temporally consistent context for memory compression.
        
        This centralizes the complex temporal rendering logic in SpaceVeilProducer.
        """
        try:
            # Step 1: Reconstruct historical state
            historical_state = await self.reconstruct_veil_state_at_delta_index(memory_formation_index)
            
            # Step 2: Apply future edits for final appearance
            final_appearance_state = await self.render_state_with_future_edits(
                historical_state, memory_formation_index
            )
            
            # Step 3: Build hierarchical VEIL from temporal state
            temporal_veil = self.build_hierarchical_veil_from_flat_cache(
                flat_cache=final_appearance_state,
                root_node_id=f"{self.owner.id}_space_root",
                owner_id_for_logging=self.owner.id,
                processed_nodes_this_call=set()
            )
            
            # Step 4: Use HUD to render temporal context
            hud_component = self._get_hud_component()
            if hud_component:
                return await hud_component.render_temporal_veil_for_compression(
                    temporal_veil=temporal_veil,
                    exclude_element_id=element_id,
                    memory_formation_index=memory_formation_index
                )
            else:
                # Fallback rendering if HUD not available
                return self._render_temporal_veil_fallback(temporal_veil, element_id)
                
        except Exception as e:
            logger.error(f"Error rendering temporal context: {e}", exc_info=True)
            return f"Error rendering temporal context: {e}"

    def _get_hud_component(self):
        """Get HUD component from owner Space for temporal rendering."""
        if self.owner:
            return self.owner.get_component('HUDComponent')
        return None

    def _render_temporal_veil_fallback(self, temporal_veil: Dict[str, Any], exclude_element_id: str) -> str:
        """
        Fallback rendering when HUD component is not available.
        
        Provides basic text representation of temporal VEIL state.
        """
        try:
            if not temporal_veil:
                return "No temporal context available"
            
            def render_node(node: Dict[str, Any], depth: int = 0) -> str:
                indent = "  " * depth
                props = node.get("properties", {})
                node_type = node.get("node_type", "unknown")
                veil_id = node.get("veil_id", "unknown")
                
                # Skip the excluded element
                if props.get("element_id") == exclude_element_id:
                    return ""
                
                # Basic node representation
                result = f"{indent}{node_type}: {veil_id}\n"
                
                # Add key properties
                if props.get("element_name"):
                    result += f"{indent}  name: {props['element_name']}\n"
                if props.get("text_content"):
                    content = props['text_content'][:100] + "..." if len(props['text_content']) > 100 else props['text_content']
                    result += f"{indent}  content: {content}\n"
                
                # Recursively render children
                children = node.get("children", [])
                for child in children:
                    result += render_node(child, depth + 1)
                
                return result
            
            return render_node(temporal_veil)
            
        except Exception as e:
            logger.error(f"Error in fallback temporal rendering: {e}", exc_info=True)
            return f"Error in temporal rendering: {e}"

    def get_accumulated_deltas(self) -> List[Dict[str, Any]]:
        """
        Get and clear accumulated deltas for frame-end dispatch.
        
        Returns:
            List of accumulated deltas, clearing the internal accumulator
        """
        deltas = list(self._accumulated_deltas)
        self._accumulated_deltas.clear()
        return deltas

    def _get_current_space_properties(self) -> Dict[str, Any]:
        """Extracts properties of the Space element itself for its VEIL root node."""
        if not self.owner:
            logger.error(f"[{self.COMPONENT_TYPE}] Owner not set, cannot get space properties.")
            return {}
        props = {
            "structural_role": "root",
            "content_nature": "space_summary",
            "element_id": self.owner.id,
            "element_name": self.owner.name,
            "element_type": self.owner.__class__.__name__,
            "is_inner_space": getattr(self.owner, 'IS_INNER_SPACE', False)
        }
        if hasattr(self.owner, 'agent_description'):
            props['agent_description'] = self.owner.agent_description
            props['agent_name'] = self.owner.agent_name
        return props

    def build_hierarchical_veil_from_flat_cache(self,
                                             flat_cache: Dict[str, Any],
                                             root_node_id: str,
                                             owner_id_for_logging: str,
                                             processed_nodes_this_call: Set[str]) -> Optional[Dict[str, Any]]:
        """
        Recursively builds a hierarchical VEIL node structure from a flat cache.
        It discovers children by looking for nodes in the flat_cache whose 'parent_id'
        (in properties or directly) matches the current root_node_id.
        'processed_nodes_this_call' is a set passed by the caller to detect cycles during this specific build.
        """
        if root_node_id in processed_nodes_this_call:
            logger.warning(f"[{owner_id_for_logging}/{self.COMPONENT_TYPE}] Cycle detected for node_id {root_node_id} during VEIL reconstruction.")
            return None

        original_node_data = flat_cache.get(root_node_id)
        if not original_node_data:
            logger.warning(f"[{owner_id_for_logging}/{self.COMPONENT_TYPE}] Node {root_node_id} referenced but not found in flat_cache for reconstruction.")
            return None

        processed_nodes_this_call.add(root_node_id)

        reconstructed_node = copy.deepcopy(original_node_data)

        # Initialize children list for the reconstructed node.
        # This list will be populated by discovering children from the flat_cache.
        current_node_built_children = []

        # Iterate through the entire flat_cache to find nodes that are children of root_node_id
        for potential_child_veil_id, potential_child_node_data_from_cache in flat_cache.items():
            if potential_child_veil_id == root_node_id: # A node cannot be its own child
                continue

            parent_id_of_potential_child = None
            if isinstance(potential_child_node_data_from_cache, dict):
                # Prioritize parent_id from properties
                parent_id_in_props = potential_child_node_data_from_cache.get("properties", {}).get("parent_id")
                # Fallback to parent_id as a direct key in the node data
                parent_id_direct = potential_child_node_data_from_cache.get("parent_id")

                if parent_id_in_props:
                    parent_id_of_potential_child = parent_id_in_props
                elif parent_id_direct:
                    parent_id_of_potential_child = parent_id_direct

            if parent_id_of_potential_child == root_node_id:
                # This potential_child_veil_id is a child of the current root_node_id.
                # Recursively build this child.
                # Pass the same 'processed_nodes_this_call' set for cycle detection within this build operation.
                fully_built_child_node = self.build_hierarchical_veil_from_flat_cache(
                    flat_cache=flat_cache,
                    root_node_id=potential_child_veil_id, # Use the child's veil_id as the new root for recursion
                    owner_id_for_logging=owner_id_for_logging,
                    processed_nodes_this_call=processed_nodes_this_call
                )

                if fully_built_child_node:
                    current_node_built_children.append(fully_built_child_node)

        # Assign the discovered and built children to the reconstructed node.
        # Ensure 'children' is always a list, even if empty.
        reconstructed_node["children"] = current_node_built_children

        processed_nodes_this_call.remove(root_node_id) # Backtrack for multiple branches from the same parent
        return reconstructed_node

    def get_full_veil(self) -> Optional[Dict[str, Any]]:
        """
        Constructs and returns a full hierarchical VEIL representation of its owner Space,
        by using the Space's internal flat cache (`_flat_veil_cache`).

        This method is primarily intended to be called BY THE OWNING SPACE ITSELF
        (e.g., from Space.get_full_veil_snapshot()) to encapsulate the building logic.
        External consumers should typically call Space.get_full_veil_snapshot().
        """
        if not self.owner:
            logger.error(f"[{self.COMPONENT_TYPE}] Owner (Space) not set, cannot generate its full VEIL.")
            return None

        owner_id = self.owner.id
        # Use SpaceVeilProducer's own flat cache instead of owner's
        space_flat_cache = self._flat_veil_cache
        space_root_veil_id = f"{owner_id}_space_root"

        if not space_flat_cache or space_root_veil_id not in space_flat_cache:
            logger.warning(f"[{owner_id}/{self.COMPONENT_TYPE}] SpaceVeilProducer's flat cache is empty or root node '{space_root_veil_id}' missing. Cannot build full VEIL via get_full_veil.")
            # Return a minimal representation of the root if it's missing, or an error structure
            return {
                "veil_id": space_root_veil_id,
                "node_type": VEIL_SPACE_ROOT_TYPE,
                "properties": self._get_current_space_properties(), # Provide basic props even on error
                "children": [],
                "status": "Error: Cache unavailable or root missing for producer's get_full_veil."
            }

        logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] get_full_veil: Building hierarchical VEIL from SpaceVeilProducer's flat cache (root: '{space_root_veil_id}').")
        full_veil = self.build_hierarchical_veil_from_flat_cache(
            flat_cache=space_flat_cache,
            root_node_id=space_root_veil_id,
            owner_id_for_logging=owner_id,
            processed_nodes_this_call=set() # Important: fresh set for each top-level call
        )
        return full_veil

    def calculate_delta(self) -> Optional[List[Dict[str, Any]]]:
        """
        Calculates deltas ONLY for the Space element's own root node.
        It no longer aggregates deltas from children; Space.on_frame_end handles that.
        State updates for this producer's baseline are handled within this method.
        """
        if not self.owner:
            logger.error(f"[{self.COMPONENT_TYPE}] Owner not set, cannot calculate delta.")
            return None

        owner_id = self.owner.id
        delta_operations = []
        space_root_veil_id = f"{owner_id}_space_root"
        current_space_props = self._get_current_space_properties()

        # NEW: Check actual cache state, not just the flag
        # This prevents race conditions where cache gets cleared but flag says we already produced root
        root_exists_in_cache = (self._flat_veil_cache and space_root_veil_id in self._flat_veil_cache)

        has_produced_flag = self._state.get('_has_produced_root_add_before', False)

        # Generate add_node if either: never produced before OR root missing from cache
        if not has_produced_flag or not root_exists_in_cache:
            if not root_exists_in_cache:
                logger.warning(f"[{owner_id}/{self.COMPONENT_TYPE}] Space root '{space_root_veil_id}' missing from cache, regenerating add_node operation (flag was: {has_produced_flag})")
            else:
                logger.info(f"[{owner_id}/{self.COMPONENT_TYPE}] Generating initial 'add_node' for Space root '{space_root_veil_id}'.")

            delta_operations.append({
                "op": "add_node",
                "node": {
                    "veil_id": space_root_veil_id,
                    "node_type": VEIL_SPACE_ROOT_TYPE,
                    "properties": current_space_props,
                    "children": [] # Root node initially has no children declared by this delta
                }
            })
        else:
            # Root exists in cache and we've produced it before, check for property updates
            last_space_props = self._state.get('_last_space_properties', {})
            if current_space_props != last_space_props:
                logger.info(f"[{owner_id}/{self.COMPONENT_TYPE}] Generating 'update_node' for Space root properties on '{space_root_veil_id}'.")
                delta_operations.append({
                    "op": "update_node",
                    "veil_id": space_root_veil_id,
                    "properties": current_space_props
                })

        # NEW: Add owner tracking to all delta operations
        if delta_operations:
            delta_operations = self._add_owner_tracking_to_delta_ops(delta_operations)
            logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Calculated delta for Space root: {delta_operations}")
        else:
            logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] No delta operations for Space root this frame.")

        self._state['_last_space_properties'] = copy.deepcopy(current_space_props)

        # Update the flag based on what we actually generated
        if not has_produced_flag:
            # Check if an add_node op for the root was actually generated
            for delta_op in delta_operations:
                if delta_op.get("op") == "add_node" and \
                   isinstance(delta_op.get("node"), dict) and \
                   delta_op["node"].get("veil_id") == space_root_veil_id:
                    self._state['_has_produced_root_add_before'] = True
                    logger.info(f"[{owner_id}/{self.COMPONENT_TYPE}] Confirmed 'add_node' for Space root '{space_root_veil_id}' produced. Flag set.")
                    break

        logger.debug(
            f"[{owner_id}/{self.COMPONENT_TYPE}] calculate_delta completed. "
            f"Root add produced flag: {self._state.get('_has_produced_root_add_before', False)}. "
            f"Root exists in cache: {root_exists_in_cache}"
        )
        return delta_operations if delta_operations else None
