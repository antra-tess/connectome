import logging
from typing import Dict, Any, Optional, List, Set
import copy # Added for deepcopy

from ..base_component import VeilProducer
from .cache_component import RemoteStateCacheComponent # Sibling component
# Import the registry decorator
from elements.component_registry import register_component

logger = logging.getLogger(__name__)

# VEIL Node Structure Constants (Example)
VEIL_UPLINK_ROOT_TYPE = "uplinked_content_container"
VEIL_REMOTE_SPACE_ID_PROP = "remote_space_id"
VEIL_REMOTE_SPACE_NAME_PROP = "remote_space_name"
VEIL_LAST_SYNC_PROP = "last_sync_timestamp"
# VEIL_CACHED_NODES_CONTAINER_TYPE = "cached_remote_nodes" # Optional intermediate node, removed for now to simplify. Children of remote root go directly under uplink root.

@register_component
class UplinkVeilProducer(VeilProducer):
    """
    Generates VEIL representation. For its own root node, it calculates deltas.
    For remote content, it processes and re-parents deltas from RemoteStateCacheComponent.
    """
    COMPONENT_TYPE = "UplinkVeilProducer"

    # Dependencies: Requires a RemoteStateCacheComponent on the same Element
    REQUIRED_SIBLING_COMPONENTS = [RemoteStateCacheComponent]

    def initialize(self, **kwargs) -> None:
        """Initializes the component state for delta tracking."""
        super().initialize(**kwargs)
        self._state.setdefault('_has_produced_uplink_root_add_before', False)
        self._state.setdefault('_last_uplink_root_properties', {})
        # _last_presented_remote_structure and _last_identified_remote_root_id removed
        logger.debug(f"UplinkVeilProducer initialized for Element {self.owner.id if self.owner else 'Unknown'}")

    def _get_cache_component(self) -> Optional[RemoteStateCacheComponent]:
        """Helper to get the sibling cache component."""
        cache_comp = self.get_sibling_component(RemoteStateCacheComponent)
        if not cache_comp:
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}/{self.COMPONENT_TYPE}] Critical: RemoteStateCacheComponent not found.")
        return cache_comp
    
    def _get_current_uplink_root_properties(self) -> Dict[str, Any]:
        """Helper to get the current properties for this producer's root node."""
        cache_comp = self._get_cache_component()
        # Get count from the main remote_state_cache, not pending_remote_deltas
        cached_nodes_count = len(cache_comp._state.get("remote_state_cache", {})) if cache_comp else 0
        
        return {
            "structural_role": "container",
            "content_nature": "uplink_summary",
            "element_id": self.owner.id,
            "element_name": self.owner.name,
            VEIL_REMOTE_SPACE_ID_PROP: getattr(self.owner, 'remote_space_id', 'unknown'),
            VEIL_REMOTE_SPACE_NAME_PROP: getattr(self.owner, 'remote_space_info', {}).get('name', 'unknown'),
            VEIL_LAST_SYNC_PROP: cache_comp._state.get("last_successful_sync") if cache_comp else None,
            "cached_node_count": cached_nodes_count
        }

    def _recursively_build_hierarchy(self, node_id_to_build: str,
                                     all_nodes_map: Dict[str, Dict[str, Any]],
                                     processed_during_this_build: Set[str]) -> Optional[Dict[str, Any]]:
        """
        Helper to recursively build a full hierarchical node structure from a flat map
        of cached nodes. Used by get_full_veil.
        """
        if node_id_to_build in processed_during_this_build:
            logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Cycle detected for node_id {node_id_to_build} during VEIL reconstruction.")
            return None 

        original_node_data = all_nodes_map.get(node_id_to_build)
        if not original_node_data:
            logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Node {node_id_to_build} referenced but not found in cache for VEIL reconstruction.")
            return None

        processed_during_this_build.add(node_id_to_build)
        
        reconstructed_node = copy.deepcopy(original_node_data)
        
        reconstructed_children_list = []
        if 'children' in reconstructed_node and isinstance(reconstructed_node['children'], list):
            original_child_node_stubs_in_parent = reconstructed_node['children']
            reconstructed_node['children'] = [] # Start fresh

            for child_node_stub in original_child_node_stubs_in_parent:
                if not isinstance(child_node_stub, dict) or 'veil_id' not in child_node_stub:
                    logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Child item in node {node_id_to_build} is not a valid node stub: {child_node_stub}")
                    continue
                
                child_veil_id = child_node_stub['veil_id']
                fully_built_child = self._recursively_build_hierarchy(child_veil_id, all_nodes_map, processed_during_this_build)
                
                if fully_built_child:
                    reconstructed_children_list.append(fully_built_child)
            
            reconstructed_node['children'] = reconstructed_children_list
        
        processed_during_this_build.remove(node_id_to_build)
        return reconstructed_node
        
    def _identify_remote_root_in_cache_for_full_veil(self, cached_nodes_dict: Dict[str, Any]) -> Optional[str]:
        """Identifies the VEIL ID of the remote space's root node within the cache."""
        remote_space_id_attr = getattr(self.owner, 'remote_space_id', None)
        if not remote_space_id_attr:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] UplinkProxy owner does not have remote_space_id. Cannot determine remote root for full VEIL.")
            return None

        potential_root_id_by_convention = f"{remote_space_id_attr}_space_root"
        if potential_root_id_by_convention in cached_nodes_dict:
            return potential_root_id_by_convention
        
        for node_id, node_data in cached_nodes_dict.items():
            props = node_data.get("properties", {})
            # Check common properties that might indicate a space's root node
            if props.get("element_id") == remote_space_id_attr and \
               node_data.get("node_type") == "space_root" and \
               props.get("structural_role") == "root" and \
               props.get("content_nature") == "space_summary":
                return node_id
        
        logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Could not identify a clear remote root for full VEIL for remote_space_id '{remote_space_id_attr}'.")
        return None

    def get_full_veil(self) -> Optional[Dict[str, Any]]:
        """
        Generates the complete VEIL structure for the cached remote space state,
        reconstructing the hierarchy.
        """
        cache_comp = self._get_cache_component()
        if not cache_comp: return None

        cached_nodes_dict = cache_comp.get_synced_remote_state(force_sync=False)
        uplink_root_properties = self._get_current_uplink_root_properties()
        
        hierarchical_children_of_uplink_root = []
        identified_remote_root_id = self._identify_remote_root_in_cache_for_full_veil(cached_nodes_dict)

        if identified_remote_root_id:
            logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] get_full_veil: Identified remote root node: {identified_remote_root_id}. Reconstructing hierarchy.")
            reconstructed_remote_root_tree = self._recursively_build_hierarchy(identified_remote_root_id, cached_nodes_dict, set())
            if reconstructed_remote_root_tree:
                hierarchical_children_of_uplink_root.append(reconstructed_remote_root_tree)
            else:
                logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] get_full_veil: Failed to reconstruct hierarchy from remote root {identified_remote_root_id}. Uplink VEIL will not show remote content tree.")
        elif cached_nodes_dict: # If no clear root but cache has content, log it.
             logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] get_full_veil: Remote state cache has {len(cached_nodes_dict)} items but no clear remote root identified. Uplink VEIL will not show remote content tree.")
        
        root_veil_node = {
            "veil_id": f"{self.owner.id}_uplink_root",
            "node_type": VEIL_UPLINK_ROOT_TYPE,
            "properties": uplink_root_properties,
            "children": hierarchical_children_of_uplink_root
        }
        return root_veil_node

    def calculate_delta(self) -> Optional[List[Dict[str, Any]]]:
        uplink_root_deltas = []
        uplink_root_veil_id = f"{self.owner.id}_uplink_root"
        
        parent_info = self.owner.get_parent_info()
        parent_space_id = parent_info.get("parent_id") if parent_info else None
        parent_space_root_veil_id = f"{parent_space_id}_space_root" if parent_space_id else None

        if not parent_space_root_veil_id:
            logger.warning(f"[{self.owner.id if self.owner else 'Unknown'}/{self.COMPONENT_TYPE}] Cannot determine parent_space_root_veil_id. Uplink root delta will not be parented.")

        current_uplink_root_props = self._get_current_uplink_root_properties()
        if not self._state.get('_has_produced_uplink_root_add_before', False):
            uplink_root_deltas.append({
                "op": "add_node",
                "parent_id": parent_space_root_veil_id,
                "node": {
                    "veil_id": uplink_root_veil_id,
                    "node_type": VEIL_UPLINK_ROOT_TYPE,
                    "properties": current_uplink_root_props,
                    "children": [] 
                }
            })
        else:
            last_props = self._state.get('_last_uplink_root_properties', {})
            if last_props != current_uplink_root_props:
                uplink_root_deltas.append({
                    "op": "update_node",
                    "veil_id": uplink_root_veil_id,
                    "properties": current_uplink_root_props
                })

        # Process pending remote deltas
        processed_remote_deltas = []
        cache_comp = self._get_cache_component()
        if not cache_comp:
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}/{self.COMPONENT_TYPE}] RemoteStateCacheComponent not found. Cannot process remote deltas.")
        else:
            pending_deltas_from_cache = cache_comp.get_pending_remote_deltas_and_clear()
            if pending_deltas_from_cache:
                logger.info(f"[{self.owner.id if self.owner else 'Unknown'}/{self.COMPONENT_TYPE}] Processing {len(pending_deltas_from_cache)} pending remote deltas.")
                remote_space_actual_id = getattr(self.owner, 'remote_space_id', None)
                expected_remote_space_root_veil_id = f"{remote_space_actual_id}_space_root" if remote_space_actual_id else None

                for remote_op in pending_deltas_from_cache:
                    op_to_emit = copy.deepcopy(remote_op) # Process a copy
                    
                    # Re-parent the remote space's root node if it's being added
                    if expected_remote_space_root_veil_id and \
                       op_to_emit.get("op") == "add_node" and \
                       isinstance(op_to_emit.get("node"), dict) and \
                       op_to_emit["node"].get("veil_id") == expected_remote_space_root_veil_id:
                        
                        logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}/{self.COMPONENT_TYPE}] Re-parenting remote space root delta '{expected_remote_space_root_veil_id}' to '{uplink_root_veil_id}'.")
                        op_to_emit["parent_id"] = uplink_root_veil_id
                    
                    processed_remote_deltas.append(op_to_emit)
            else:
                logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}/{self.COMPONENT_TYPE}] No pending remote deltas to process.")

        final_deltas = uplink_root_deltas + processed_remote_deltas

        if final_deltas:
            logger.info(f"[{self.owner.id if self.owner else 'Unknown'}/{self.COMPONENT_TYPE}] Calculated {len(final_deltas)} total delta operations ({len(uplink_root_deltas)} for uplink root, {len(processed_remote_deltas)} for remote content).")
        
        # Update state for THIS producer's root node only
        self.signal_delta_produced_this_frame(uplink_root_deltas)
        return final_deltas if final_deltas else None

    def signal_delta_produced_this_frame(self, produced_uplink_root_deltas: List[Dict[str, Any]]):
        """
        Updates baseline state for THIS UplinkVeilProducer's OWN root node,
        based on the deltas *it* generated for that root node.
        """
        current_uplink_root_props = self._get_current_uplink_root_properties()
        self._state['_last_uplink_root_properties'] = current_uplink_root_props
        
        uplink_root_veil_id = f"{self.owner.id}_uplink_root"
        if not self._state.get('_has_produced_uplink_root_add_before', False):
            for delta_op in produced_uplink_root_deltas: 
                if delta_op.get("op") == "add_node" and \
                   isinstance(delta_op.get("node"), dict) and \
                   delta_op["node"].get("veil_id") == uplink_root_veil_id:
                    self._state['_has_produced_uplink_root_add_before'] = True
                    logger.info(f"[{self.owner.id if self.owner else 'Unknown'}/{self.COMPONENT_TYPE}] Confirmed uplink root '{uplink_root_veil_id}' add_node produced.")
                    break
        
        # No longer updates _last_presented_remote_structure or _last_identified_remote_root_id here
        logger.debug(
            f"[{self.owner.id if self.owner else 'Unknown'}/{self.COMPONENT_TYPE}] Baseline updated for UplinkVeilProducer's own root node. "
            f"Uplink root added flag: {self._state.get('_has_produced_uplink_root_add_before', False)}."
        )
