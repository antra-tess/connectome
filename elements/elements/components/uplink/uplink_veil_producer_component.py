import logging
from typing import Dict, Any, Optional, List, Set

from ..base_component import VeilProducer
from .cache_component import RemoteStateCacheComponent # Sibling component
# Import the registry decorator
from elements.component_registry import register_component

logger = logging.getLogger(__name__)

# VEIL Node Structure Constants (Example)
VEIL_UPLINK_ROOT_TYPE = "uplink_proxy_root"
VEIL_REMOTE_SPACE_ID_PROP = "remote_space_id"
VEIL_REMOTE_SPACE_NAME_PROP = "remote_space_name"
VEIL_LAST_SYNC_PROP = "last_sync_timestamp"
VEIL_CACHED_NODES_CONTAINER_TYPE = "cached_remote_nodes" # Optional intermediate node

@register_component
class UplinkVeilProducer(VeilProducer):
    """
    Generates VEIL representation based on the cached state of a remote space
    managed by a sibling RemoteStateCacheComponent.
    """
    COMPONENT_TYPE = "UplinkVeilProducer"

    # Dependencies: Requires a RemoteStateCacheComponent on the same Element
    REQUIRED_SIBLING_COMPONENTS = [RemoteStateCacheComponent]

    def initialize(self, **kwargs) -> None:
        """Initializes the component state for delta tracking."""
        super().initialize(**kwargs)
        # Track the veil_ids of the nodes present in the last generated VEIL
        self._state.setdefault('_last_generated_cache_veil_ids', set())
        # Optional: Store a shallow copy of last node properties for update detection
        self._state.setdefault('_last_node_properties', {})
        # Store the latest full snapshot received, if needed for direct access
        self._state.setdefault('_current_full_snapshot', None)
        logger.debug(f"UplinkVeilProducer initialized for Element {self.owner.id}")

    def _get_cache_component(self) -> Optional[RemoteStateCacheComponent]:
        """Helper to get the sibling cache component."""
        return self.get_sibling_component(RemoteStateCacheComponent)

    def get_full_veil(self) -> Optional[Dict[str, Any]]:
        """
        Generates the complete VEIL structure for the cached remote space state.
        """
        cache_comp = self._get_cache_component()
        if not cache_comp:
            logger.error(f"[{self.owner.id}] Cannot generate VEIL: RemoteStateCacheComponent not found.")
            return None

        # Get the current cached state (don't force sync here, VEIL reflects cache)
        # The cache format is assumed to be { veil_id: node_data }
        cached_nodes_dict = cache_comp.get_synced_remote_state(force_sync=False)
        current_cache_veil_ids = set(cached_nodes_dict.keys())

        # Extract nodes directly from the cache dictionary values
        child_veil_nodes = list(cached_nodes_dict.values())

        # Create the root container node for the uplink representation
        root_veil_node = {
            "veil_id": f"{self.owner.id}_uplink_root", # Unique ID for this uplink's VEIL root
            "node_type": VEIL_UPLINK_ROOT_TYPE,
            "properties": {
                "structural_role": "container",
                "content_nature": "uplink_summary",
                "element_id": self.owner.id,
                "element_name": self.owner.name,
                VEIL_REMOTE_SPACE_ID_PROP: getattr(self.owner, 'remote_space_id', 'unknown'),
                VEIL_REMOTE_SPACE_NAME_PROP: getattr(self.owner, 'remote_space_info', {}).get('name', 'unknown'),
                VEIL_LAST_SYNC_PROP: cache_comp._state.get("last_successful_sync"),
                "cached_node_count": len(child_veil_nodes)
                # Add other relevant uplink properties (e.g., connection status?)
            },
            "children": child_veil_nodes # Embed the cached nodes directly
        }

        # Update state for delta calculation
        self._state['_last_generated_cache_veil_ids'] = current_cache_veil_ids
        # Store properties for detailed delta checks later
        self._state['_last_node_properties'] = {
            nid: node.get("properties", {}) for nid, node in cached_nodes_dict.items()
        }
        self._state['_current_full_snapshot'] = root_veil_node # Cache the generated full VEIL

        return root_veil_node

    def calculate_delta(self) -> Optional[List[Dict[str, Any]]]:
        """
        Calculates the changes (delta) in the cached state since the last VEIL generation.
        Detects added, removed, and potentially updated nodes.
        """
        cache_comp = self._get_cache_component()
        if not cache_comp:
            logger.error(f"[{self.owner.id}] Cannot calculate VEIL delta: RemoteStateCacheComponent not found.")
            return None

        current_cache_dict = cache_comp.get_synced_remote_state(force_sync=False)
        current_ids = set(current_cache_dict.keys())
        last_ids = self._state.get('_last_generated_cache_veil_ids', set())
        last_props = self._state.get('_last_node_properties', {})

        delta_operations = []
        parent_veil_id = f"{self.owner.id}_uplink_root" # ID of the container node

        # 1. Detect removed nodes
        removed_ids = last_ids - current_ids
        for removed_id in removed_ids:
            delta_operations.append({
                "op": "remove_node",
                "veil_id": removed_id
            })

        # 2. Detect added and potentially modified nodes
        current_node_properties = {}
        for node_id, current_node_data in current_cache_dict.items():
            current_props = current_node_data.get("properties", {})
            current_node_properties[node_id] = current_props # Store for next delta

            if node_id not in last_ids:
                # Added node
                delta_operations.append({
                    "op": "add_node",
                    "parent_id": parent_veil_id,
                    "node": current_node_data,
                    # "position": ? # Position might be irrelevant if children are unordered dict
                })
            else:
                # Existing node: Check for modifications
                last_node_props = last_props.get(node_id)
                if last_node_props != current_props: # Simple dict comparison
                    logger.debug(f"[{self.owner.id}] Detected property change for cached node {node_id}")
                    # Send an update operation with only the changed properties?
                    # Or send the full node update? Let's send updated properties for now.
                    # More sophisticated diffing could be done here.
                    delta_operations.append({
                        "op": "update_node",
                        "veil_id": node_id,
                        "properties": current_props # Send all current properties
                    })

        # Update state for next delta calculation
        self._state['_last_generated_cache_veil_ids'] = current_ids
        self._state['_last_node_properties'] = current_node_properties
        # After calculating delta, the full snapshot is implicitly based on these current_ids/props
        # No need to update _current_full_snapshot here as it reflects the last get_full_veil() call

        if delta_operations:
            logger.info(f"[{self.owner.id}] Calculated Uplink VEIL delta with {len(delta_operations)} operations.")
        return delta_operations

    def on_cache_updated(self, new_full_snapshot_data: Dict[str, Any]) -> None:
        """
        Called by RemoteStateCacheComponent when its cache has been fully updated 
        (e.g., after a successful sync_remote_state).
        This signals that the producer should reset its delta tracking baseline.

        Args:
            new_full_snapshot_data: The complete new state from the remote cache.
                                   While passed, this producer typically re-fetches from
                                   the cache component directly when needed.
        """
        logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Cache updated notification received. Resetting delta baseline.")
        # Reset the baseline for delta calculation.
        # The next call to calculate_delta will treat everything in the updated 
        # RemoteStateCacheComponent as new.
        self._state['_last_generated_cache_veil_ids'] = set()
        self._state['_last_node_properties'] = {}
        # Optionally, we could store the new_full_snapshot_data here if we wanted to avoid
        # an immediate re-fetch in get_full_veil, but current design re-fetches.
        self._state['_current_full_snapshot'] = None # Invalidate any previously stored full snapshot
        
        # Important: After this, the next on_frame_end will cause calculate_delta
        # to generate a potentially large delta representing the full new state.
