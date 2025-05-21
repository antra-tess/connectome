"""
Remote State Cache Component
Component for caching and retrieving state from a remote space via an Uplink.
"""

import logging
from typing import Dict, Any, Optional, List, Set, Callable, TYPE_CHECKING
import uuid
import time
from datetime import datetime, timedelta
import copy

from ..base_component import Component
from .connection_component import UplinkConnectionComponent # Needed to check connection status
# Import the registry decorator
from elements.component_registry import register_component

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...base import BaseElement # For type hinting element
    from ...space import Space # For type hinting remote_space
    from ....space_registry import SpaceRegistry # For type hinting space_registry
    from ...uplink import UplinkProxy # Specifically for self.element type hint

@register_component
class RemoteStateCacheComponent(Component):
    """
    Manages caching of remote state, history bundles, and synchronization for an Uplink.
    """
    
    COMPONENT_TYPE: str = "remote_state_cache"
    DEPENDENCIES: List[str] = ["uplink_connection"] # Needs connection status
    
    # Events this component handles
    HANDLED_EVENT_TYPES: List[str] = [
        "uplink_state_synced",
        "sync_request" # External trigger to sync
    ]
    
    def __init__(self, cache_ttl: int = 300, remote_space_id: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.remote_space_id = remote_space_id or "unknown_remote"
        # State stores cached data and metadata
        self._state = {
            "remote_state_cache": {}, # e.g., { element_id: element_data }
            "history_bundles": {}, # e.g., { span_id: bundle_data }
            "processed_remote_events": set(), # Track IDs to avoid duplicates
            "last_successful_sync": None,
            "last_sync_attempt": None,
            "cache_expiry": {}, # e.g., { cache_key: expiry_timestamp }
            "cache_ttl": cache_ttl, # Default time-to-live in seconds
            "pending_remote_deltas": [] # NEW: For storing raw deltas from remote
        }
        # Timer for auto-sync (implementation detail)
        self._auto_sync_enabled = False
        self._auto_sync_timer = None
        self._auto_sync_task = None
        self._auto_sync_stop_event = None
        self._space_registry_ref: Optional["SpaceRegistry"] = None # Will be set in initialize
        
        # REMOVED: Try to get space_registry from the owner UplinkProxy
        # if hasattr(self.element, '_space_registry') and self.element._space_registry:
        #     self._space_registry_ref = self.element._space_registry
        # else:
        #     logger.warning(f"[{self.element.id}/{self.COMPONENT_TYPE}] SpaceRegistry not found on owner element. Initial sync might be limited.")

    def initialize(self, **kwargs) -> None:
        """Initializes the component after it's been added to an element."""
        super().initialize(**kwargs)
        # Try to get space_registry from the owner UplinkProxy
        if self.owner and hasattr(self.owner, '_space_registry') and self.owner._space_registry:
            self._space_registry_ref = self.owner._space_registry
        else:
            owner_id_for_log = self.owner.id if self.owner else "UnknownOwner"
            logger.warning(f"[{owner_id_for_log}/{self.COMPONENT_TYPE}] SpaceRegistry not found on owner element. Initial sync might be limited.")
        logger.debug(f"RemoteStateCacheComponent initialized for {self.owner.id if self.owner else 'UnknownOwner'}")

    def _get_connection_comp(self) -> Optional[UplinkConnectionComponent]:
        """Helper to get the associated UplinkConnectionComponent."""
        if not self.owner:
            return None
        return self.owner.get_component_by_type("uplink_connection")

    def sync_remote_state(self, force: bool = False) -> bool:
        from ...uplink import UplinkProxy
        """
        Synchronizes the cache with the remote space's current state.
        Fetches metadata and a flat VEIL cache snapshot.

        Args:
            force: If True, forces a sync even if one was recently done.

        Returns:
            True if synchronization was successful (or attempted and no new data),
            False if a significant error occurred (e.g., remote space not found).
        """
        logger.info(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Attempting to sync remote state for space ID: {self.remote_space_id}") 
        self._state["last_sync_attempt"] = time.time()

        if not self.owner or not isinstance(self.owner, UplinkProxy):
            logger.error(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Owner element is not an UplinkProxy. Cannot sync.")
            return False
            
        uplink_proxy_element: "UplinkProxy" = self.owner 

        if not self._space_registry_ref:
            logger.error(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] SpaceRegistry reference not available. Cannot find remote space.")
            return False

        remote_space_id = uplink_proxy_element.remote_space_id
        remote_space: Optional["Space"] = self._space_registry_ref.get_space(remote_space_id)

        if not remote_space:
            logger.error(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Remote space '{remote_space_id}' not found in SpaceRegistry.")
            # Update remote_space_info to reflect that it's currently unreachable
            uplink_proxy_element.remote_space_info.update({
                "status": "unreachable",
                "last_reachability_check": time.time()
            })
            return False

        # 1. Fetch and update remote_space_info on the UplinkProxy element
        try:
            logger.debug(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Fetching metadata from remote space '{remote_space_id}'.")
            metadata = remote_space.get_space_metadata_for_uplink()
            if metadata:
                uplink_proxy_element.remote_space_info.update(metadata)
                uplink_proxy_element.remote_space_info["status"] = "reachable"
                uplink_proxy_element.remote_space_info["last_metadata_sync"] = time.time()
                logger.info(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Updated remote_space_info: {uplink_proxy_element.remote_space_info}")
            else:
                logger.warning(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Remote space '{remote_space_id}' returned no metadata.")
                # Keep existing info but mark as potentially stale or issue fetching
                uplink_proxy_element.remote_space_info["status"] = "metadata_fetch_failed"

        except Exception as e:
            logger.error(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Error fetching metadata from '{remote_space_id}': {e}", exc_info=True)
            uplink_proxy_element.remote_space_info["status"] = "metadata_fetch_error"
            # Do not necessarily return False here, could still attempt VEIL sync if desired

        # 2. Fetch flat VEIL cache snapshot from the remote space
        try:
            logger.debug(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Fetching flat VEIL cache snapshot from remote space '{remote_space_id}'.")
            flat_veil_cache_snapshot = remote_space.get_flat_veil_snapshot() 
            
            if flat_veil_cache_snapshot is not None: 
                self._state["remote_state_cache"].clear()
                self._state["remote_state_cache"] = flat_veil_cache_snapshot # It's already a deepcopy
                
                logger.info(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Successfully stored flat VEIL cache snapshot. Cache size: {len(self._state['remote_state_cache'])}")

                self._state["last_successful_sync"] = time.time()
                self._state["last_data_hash"] = hash(str(self._state["remote_state_cache"])) 
                logger.info(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Successfully synced flat VEIL snapshot. New flat cache hash: {self._state['last_data_hash']}")
                
                self._state["pending_remote_deltas"].clear()

                uplink_veil_producer = self.owner.get_component_by_type("UplinkVeilProducer")
                if uplink_veil_producer:
                    logger.debug(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Triggering emit_delta on UplinkVeilProducer after full sync.")
                    uplink_veil_producer.emit_delta()
                return True
            else:
                logger.warning(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Remote space '{remote_space_id}' returned None or empty flat VEIL snapshot.")
                # Don't clear existing cache if snapshot fetch fails, keep potentially stale data.
                return False

        except Exception as e:
            logger.error(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Error fetching VEIL snapshot from '{remote_space_id}': {e}", exc_info=True)
            return False

    def get_synced_remote_state(self, force_sync: bool = False) -> Dict[str, Any]:
        """Returns the cached remote state, optionally forcing a sync first."""
        if force_sync or self._is_cache_stale():
             logger.debug(f"Cache stale or sync forced for {self.remote_space_id}, attempting sync.")
             self.sync_remote_state() # Attempt sync, ignore result for getter
             
        # The remote_state_cache now directly holds the deepcopy from get_flat_veil_snapshot.
        # Returning another deepcopy ensures consumers can't modify this component's internal state.
        return copy.deepcopy(self._state.get("remote_state_cache", {}))

    def get_history_bundles(self, span_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Retrieves cached history bundles, potentially fetching missing ones.
        (Fetching logic TBD - would likely happen during sync_remote_state)
        """
        if span_ids:
            return {sid: self._state["history_bundles"].get(sid) for sid in span_ids if sid in self._state["history_bundles"]}
        return copy.deepcopy(self._state.get("history_bundles", {})) # Return copy

    def _is_cache_stale(self) -> bool:
        """Checks if the main cache is considered stale based on TTL."""
        last_sync = self._state.get("last_successful_sync")
        if not last_sync: # No successful sync yet, assume stale
            return True
        
        ttl = self._state.get("cache_ttl", 300) # Default to 300 seconds if not set
        if time.time() > last_sync + ttl:
            return True
        return False
         
    def enable_auto_sync(self, interval: Optional[int] = None) -> None:
        """
        Enables automatic background synchronization.
        (Note: Requires an async event loop or threading setup in the host)
        """
        if interval is not None:
            # Use the provided interval for this session's auto-sync logic
            # It could also update self._state["cache_ttl"] if they are meant to be linked
            sync_interval_seconds = interval
            self._state["cache_ttl"] = interval # Link them if auto-sync interval should also be the TTL
        else:
            sync_interval_seconds = self._state.get("cache_ttl", 300)

        if not self._auto_sync_enabled:
            self._auto_sync_enabled = True
            logger.info(f"Enabling auto-sync for {self.remote_space_id} every {sync_interval_seconds} seconds.")
            # Placeholder: Actual timer/task scheduling would integrate with host's event loop
            logger.warning("Auto-sync timer implementation required (e.g., asyncio).")
              
    def disable_auto_sync(self) -> None:
        """Disables automatic background synchronization."""
        if self._auto_sync_enabled:
            self._auto_sync_enabled = False
            logger.warning("Auto-sync timer cancellation required.")
            logger.info(f"Disabled auto-sync for {self.remote_space_id}.")

    def apply_deltas(self, deltas: List[Dict[str, Any]]) -> None:
        """
        Applies a list of VEIL delta operations to the cached state.
        Also queues these deltas to be passed on by the UplinkVeilProducer.
        """
        if not isinstance(deltas, list):
            logger.warning(f"[{self.owner.id if self.owner else 'cache'}] Invalid delta format received: not a list.")
            return

        logger.debug(f"[{self.owner.id if self.owner else 'cache'}] Applying {len(deltas)} deltas to remote_state_cache for {self.remote_space_id}.")
        cache_modified_by_these_deltas = False
        
        if "remote_state_cache" not in self._state:
             self._state["remote_state_cache"] = {}

        nodes_referenced_in_this_batch = set()

        for delta_op in deltas:
            # _process_delta_operation modifies self._state["remote_state_cache"] in place.
            # We will assume if _process_delta_operation is called, a modification might have occurred.
            # A more precise check would involve comparing cache state or having _process_delta_operation return a status.
            self._process_delta_operation(delta_op, self._state["remote_state_cache"], nodes_referenced_in_this_batch)
            cache_modified_by_these_deltas = True 

        if cache_modified_by_these_deltas: 
             # Add a deepcopy of original deltas, as _process_delta_operation might have modified them if they were complex objects.
             # For standard VEIL deltas (dicts of primitives), this is less of a concern, but good practice.
             self._state["pending_remote_deltas"].extend(copy.deepcopy(deltas))
             logger.debug(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Added {len(deltas)} to pending_remote_deltas. Total pending: {len(self._state['pending_remote_deltas'])}")

             self._state["last_successful_sync"] = time.time() # Treat delta application as a content update
             # Optionally update a separate "last_delta_update_time" if needed
             logger.info(f"Cache for {self.remote_space_id} updated via {len(deltas)} deltas.")

             uplink_veil_producer = self.owner.get_component_by_type("UplinkVeilProducer")
             if uplink_veil_producer:
                 logger.debug(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Triggering emit_delta on UplinkVeilProducer after applying deltas. Cache size: {len(self._state.get('remote_state_cache', {}))}")
                 uplink_veil_producer.emit_delta()

    def get_pending_remote_deltas_and_clear(self) -> List[Dict[str, Any]]:
        """
        Returns the list of remote deltas accumulated since the last call
        and clears the internal list. Returns a deepcopy.
        """
        # Return a deepcopy to prevent external modification of queued deltas if they are complex
        pending_deltas = copy.deepcopy(self._state.get("pending_remote_deltas", [])) 
        if pending_deltas: 
            self._state["pending_remote_deltas"] = [] 
            logger.debug(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Returning and clearing {len(pending_deltas)} pending remote deltas.")
        return pending_deltas

    def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """Handle events related to synchronization."""
        event_type = event.get("event_type")
        if event_type == "sync_request":
             force = event.get("data", {}).get("force", False)
             logger.info(f"Received sync request for {self.remote_space_id}, force={force}")
             return self.sync_remote_state(force=force) # Pass force along
        return False

    def _on_cleanup(self) -> bool:
         """Disables auto-sync on cleanup."""
         self.disable_auto_sync()
         return True 

    def _process_delta_operation(self, operation: Dict[str, Any],
                                 target_cache: Dict[str, Any], 
                                 nodes_referenced_this_op_batch: Set[str]):
        """
        Helper to process a single delta operation against the target_cache.
        target_cache is modified in place.
        nodes_referenced_this_op_batch is used to track nodes processed within a single call to apply_deltas,
        primarily for _add_or_update_node_recursive if it were still used for complex hierarchical adds.
        With flat deltas, its role is diminished but kept for structure.
        """
        op_type = operation.get("op")
        
        # Make a copy of the operation node data if it's 'add_node' or 'update_node'
        # to ensure the cache stores its own version, not a reference to the input delta's node.
        # This is crucial if the input delta might be reused or if its node part is complex.
        # For update_node, properties are copied. For add_node, the whole node is copied.

        if op_type == "add_node":
            node_to_add_original = operation.get("node")
            parent_id = operation.get("parent_id")

            if node_to_add_original and isinstance(node_to_add_original, dict) and "veil_id" in node_to_add_original:
                node_id = node_to_add_original["veil_id"]
                
                # Store a deepcopy of the node data in the cache
                node_copy_for_cache = copy.deepcopy(node_to_add_original)
                
                # Ensure children list is present and is a list of stubs if children were provided
                if "children" in node_copy_for_cache:
                    if not isinstance(node_copy_for_cache["children"], list):
                        logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Node {node_id} in add_node delta had non-list children. Replacing with empty list for cache.")
                        node_copy_for_cache["children"] = []
                    else:
                        # Convert full children to stubs if they aren't already
                        stub_children = []
                        for child in node_copy_for_cache["children"]:
                            if isinstance(child, dict) and "veil_id" in child:
                                stub_children.append({"veil_id": child["veil_id"]})
                            # else: (log warning or ignore malformed child stub)
                        node_copy_for_cache["children"] = stub_children
                else: # If no children field, ensure one exists (empty list) for consistency
                    node_copy_for_cache["children"] = []

                target_cache[node_id] = node_copy_for_cache
                
                if parent_id:
                    if parent_id in target_cache:
                        parent_node_in_cache = target_cache[parent_id]
                        if "children" not in parent_node_in_cache or not isinstance(parent_node_in_cache.get("children"), list):
                            parent_node_in_cache["children"] = []
                        
                        child_stub = {"veil_id": node_id} 
                        if not any(c.get("veil_id") == node_id for c in parent_node_in_cache["children"]):
                            parent_node_in_cache["children"].append(child_stub)
                    else:
                        logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] 'add_node' for {node_id} specified parent_id {parent_id}, but parent not in cache. Node added as potential orphan.")
                
                logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Applied 'add_node' for {node_id}. Parent hint: {parent_id}")
                nodes_referenced_this_op_batch.add(node_id)
            else:
                logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Invalid 'add_node' operation: {operation}")

        elif op_type == "update_node":
            node_id_to_update = operation.get("veil_id")
            properties_to_update = operation.get("properties")
            if node_id_to_update and properties_to_update is not None: # Allow empty properties dict
                if node_id_to_update in target_cache:
                    cached_node_data = target_cache[node_id_to_update]
                    if "properties" not in cached_node_data or not isinstance(cached_node_data.get("properties"), dict):
                         cached_node_data["properties"] = {} # Ensure properties dict exists
                    
                    # Update with a copy of the new properties
                    cached_node_data["properties"].update(copy.deepcopy(properties_to_update))
                    logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Updated properties for node {node_id_to_update} in cache.")
                    nodes_referenced_this_op_batch.add(node_id_to_update)
                else:
                    logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] 'update_node' for {node_id_to_update} but node not found in cache.")
            else:
                logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Invalid 'update_node' (missing veil_id or properties): {operation}")

        elif op_type == "remove_node":
            node_id_to_remove = operation.get("veil_id")
            if node_id_to_remove:
                if node_id_to_remove in target_cache:
                    del target_cache[node_id_to_remove]
                    nodes_referenced_this_op_batch.add(node_id_to_remove)
                    logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Removed {node_id_to_remove} from cache.")

                    for existing_node_id, existing_node_data in target_cache.items():
                        if isinstance(existing_node_data, dict):
                            children_list = existing_node_data.get("children")
                            if isinstance(children_list, list):
                                original_len = len(children_list)
                                new_children_list = [
                                    child_stub for child_stub in children_list 
                                    if not (isinstance(child_stub, dict) and child_stub.get("veil_id") == node_id_to_remove)
                                ]
                                if len(new_children_list) < original_len:
                                    existing_node_data["children"] = new_children_list
                else:
                    logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] 'remove_node' for {node_id_to_remove} but node not found in cache.")
            else:
                logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Invalid 'remove_node' (missing veil_id): {operation}")
        else:
            logger.warning(f"Unsupported delta operation received by _process_delta_operation: '{op_type}'. Skipping.") 