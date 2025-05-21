import logging
from typing import Dict, Any, Optional, List, Set
import copy

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
    Generates VEIL deltas ONLY for the Space element's own root node.
    It can also construct a full hierarchical VEIL from a flat cache if requested
    (e.g., by its owner Space calling its get_full_veil method).
    The Space element itself manages the canonical flat cache.
    """
    COMPONENT_TYPE = "SpaceVeilProducer"

    def initialize(self, **kwargs) -> None:
        """Initializes the component state for delta tracking for the Space's root node."""
        super().initialize(**kwargs)
        self._state.setdefault('_last_space_properties', {})
        self._state.setdefault('_has_produced_root_add_before', False)
        logger.debug(f"SpaceVeilProducer initialized for Element {self.owner.id if self.owner else 'Unknown'}")

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
        # Ensure owner has the _flat_veil_cache attribute
        if not hasattr(self.owner, '_flat_veil_cache'): # Corrected attribute name check
            logger.error(f"[{owner_id}/{self.COMPONENT_TYPE}] Owner Space is missing '_flat_veil_cache'. Cannot build VEIL.")
            return None

        space_flat_cache = self.owner._flat_veil_cache # Corrected attribute name usage
        space_root_veil_id = f"{owner_id}_space_root"

        if not space_flat_cache or space_root_veil_id not in space_flat_cache:
            logger.warning(f"[{owner_id}/{self.COMPONENT_TYPE}] Space's flat cache is empty or root node '{space_root_veil_id}' missing. Cannot build full VEIL via get_full_veil.")
            # Return a minimal representation of the root if it's missing, or an error structure
            return {
                "veil_id": space_root_veil_id,
                "node_type": VEIL_SPACE_ROOT_TYPE,
                "properties": self._get_current_space_properties(), # Provide basic props even on error
                "children": [],
                "status": "Error: Cache unavailable or root missing for producer's get_full_veil."
            }

        logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] get_full_veil: Building hierarchical VEIL from owner's flat cache (root: '{space_root_veil_id}').")
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

        # Optional: Check if owner is of type Space, if Space class is available for import
        # from ...space import Space # Adjust import as needed
        # if not isinstance(self.owner, Space):
        #     logger.error(f"[{self.owner.id if self.owner else 'UnknownOwner'}/{self.COMPONENT_TYPE}] SpaceVeilProducer attached to non-Space element: {type(self.owner)}. Cannot calculate delta.")
        #     return None

        owner_id = self.owner.id
        delta_operations = []
        space_root_veil_id = f"{owner_id}_space_root"
        current_space_props = self._get_current_space_properties()

        if not self._state.get('_has_produced_root_add_before', False):
            logger.info(f"[{owner_id}/{self.COMPONENT_TYPE}] Generating 'add_node' for Space root '{space_root_veil_id}'.")
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
            last_space_props = self._state.get('_last_space_properties', {})
            if current_space_props != last_space_props:
                logger.info(f"[{owner_id}/{self.COMPONENT_TYPE}] Generating 'update_node' for Space root properties on '{space_root_veil_id}'.")
                delta_operations.append({
                    "op": "update_node",
                    "veil_id": space_root_veil_id,
                    "properties": current_space_props
                })
        
        if delta_operations:
            logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Calculated delta for Space root: {delta_operations}")
        else:
            logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] No delta operations for Space root this frame.")

        self._state['_last_space_properties'] = copy.deepcopy(current_space_props)
        if not self._state.get('_has_produced_root_add_before', False):
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
        )
        return delta_operations if delta_operations else None
