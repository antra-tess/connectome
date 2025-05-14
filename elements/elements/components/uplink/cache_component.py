"""
Remote State Cache Component
Component for caching and retrieving state from a remote space via an Uplink.
"""

import logging
from typing import Dict, Any, Optional, List, Set, Callable, TYPE_CHECKING
import uuid
import time
from datetime import datetime, timedelta

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
            "cache_ttl": cache_ttl # Default time-to-live in seconds
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
        """
        Synchronizes the cache with the remote space's current state.
        Fetches metadata and a full VEIL snapshot.

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

        # 2. Fetch full VEIL snapshot from the remote space
        try:
            logger.debug(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Fetching full VEIL snapshot from remote space '{remote_space_id}'.")
            full_veil_snapshot = remote_space.get_full_veil_snapshot()
            if full_veil_snapshot:
                self._cached_state = full_veil_snapshot # Replace current cache with the full snapshot
                self._state["last_successful_sync"] = time.time()
                self._state["last_data_hash"] = hash(str(full_veil_snapshot)) # Basic change detection
                logger.info(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Successfully synced full VEIL snapshot. New hash: {self._state['last_data_hash']}")
                
                # Notify the UplinkVeilProducer that the cache has been updated
                uplink_veil_producer = self.owner.get_component_by_type("UplinkVeilProducerComponent")
                if uplink_veil_producer and hasattr(uplink_veil_producer, 'on_cache_updated'):
                    uplink_veil_producer.on_cache_updated(self._cached_state)
                return True
            else:
                logger.warning(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Remote space '{remote_space_id}' returned no VEIL snapshot.")
                # Don't clear existing cache if snapshot fetch fails, keep potentially stale data.
                return False # Indicate sync was not fully successful

        except Exception as e:
            logger.error(f"[{self.owner.id if self.owner else 'cache'}/{self.COMPONENT_TYPE}] Error fetching VEIL snapshot from '{remote_space_id}': {e}", exc_info=True)
            return False

    def get_synced_remote_state(self, force_sync: bool = False) -> Dict[str, Any]:
        """Returns the cached remote state, optionally forcing a sync first."""
        if force_sync or self._is_cache_stale():
             logger.debug(f"Cache stale or sync forced for {self.remote_space_id}, attempting sync.")
             self.sync_remote_state() # Attempt sync, ignore result for getter
             
        return self._state["remote_state_cache"].copy()

    def get_history_bundles(self, span_ids: Optional[List[str]] = None) -> Dict[str, Any]:
         """
         Retrieves cached history bundles, potentially fetching missing ones.
         (Fetching logic TBD - would likely happen during sync_remote_state)
         """
         if span_ids:
              return {sid: self._state["history_bundles"].get(sid) for sid in span_ids if sid in self._state["history_bundles"]}
         return self._state["history_bundles"].copy()

    def _is_cache_stale(self) -> bool:
         """Checks if the main cache is considered stale based on TTL."""
         expiry = self._state["cache_expiry"].get("full_state")
         if not expiry: # No expiry set, assume stale
              return True
         return int(time.time() * 1000) > expiry
         
    def enable_auto_sync(self, interval: Optional[int] = None) -> None:
         """
         Enables automatic background synchronization.
         (Note: Requires an async event loop or threading setup in the host)
         """
         if interval is not None:
              self._state["cache_ttl"] = interval # Assuming sync interval relates to TTL
              # Update sync_interval in connection component? Or keep separate? Let's keep separate for now.
              
         sync_interval_seconds = self._state["cache_ttl"] 
         if not self._auto_sync_enabled:
              self._auto_sync_enabled = True
              logger.info(f"Enabling auto-sync for {self.remote_space_id} every {sync_interval_seconds} seconds.")
              # --- Timer Implementation Placeholder --- 
              # This needs integration with the main application loop (e.g., asyncio.create_task)
              # async def _sync_loop():
              #      while self._auto_sync_enabled:
              #           await asyncio.sleep(sync_interval_seconds)
              #           if self._auto_sync_enabled: # Check again in case disabled during sleep
              #                self.sync_remote_state()
              # self._auto_sync_timer = asyncio.create_task(_sync_loop())
              logger.warning("Auto-sync timer implementation required (e.g., asyncio).")
              # -------------------------------------
              
    def disable_auto_sync(self) -> None:
         """Disables automatic background synchronization."""
         if self._auto_sync_enabled:
              self._auto_sync_enabled = False
              # --- Cancel Timer Placeholder --- 
              # if self._auto_sync_timer:
              #      self._auto_sync_timer.cancel()
              #      self._auto_sync_timer = None
              logger.warning("Auto-sync timer cancellation required.")
              # ------------------------------
              logger.info(f"Disabled auto-sync for {self.remote_space_id}.")

    # --- NEW: Apply Deltas --- 
    def apply_deltas(self, deltas: List[Dict[str, Any]]) -> None:
        """
        Applies a list of VEIL delta operations to the cached state.
        Currently handles 'add_node' and 'remove_node'.
        Structure of cache assumed to be { veil_id: node_data }
        """
        if not isinstance(deltas, list):
            logger.warning(f"[{self.owner.id if self.owner else 'cache'}] Invalid delta format received: not a list.")
            return

        logger.debug(f"[{self.owner.id if self.owner else 'cache'}] Applying {len(deltas)} deltas to cache for {self.remote_space_id}.")
        cache_modified = False
        # Ensure cache exists
        if "remote_state_cache" not in self._state:
             self._state["remote_state_cache"] = {}

        for delta_op in deltas:
            op = delta_op.get("op")
            veil_id = delta_op.get("veil_id")

            if op == "add_node":
                node_data = delta_op.get("node")
                parent_id = delta_op.get("parent_id") # Optional: used if cache is hierarchical
                if node_data and "veil_id" in node_data:
                    # Simple flat cache implementation: veil_id -> node_data
                    node_veil_id = node_data["veil_id"]
                    if node_veil_id in self._state["remote_state_cache"]:
                         logger.debug(f"Updating existing node {node_veil_id} in cache via add_node delta.")
                    else:
                         logger.debug(f"Adding new node {node_veil_id} to cache via add_node delta.")
                    self._state["remote_state_cache"][node_veil_id] = node_data
                    cache_modified = True
                    # TODO: Handle hierarchical cache using parent_id if needed
                else:
                     logger.warning(f"Skipping add_node delta: missing node data or veil_id. Op: {delta_op}")

            elif op == "remove_node":
                if veil_id:
                    if veil_id in self._state["remote_state_cache"]:
                        logger.debug(f"Removing node {veil_id} from cache via remove_node delta.")
                        del self._state["remote_state_cache"][veil_id]
                        cache_modified = True
                    else:
                        logger.debug(f"Skipping remove_node delta: veil_id {veil_id} not found in cache.")
                    # TODO: Handle removal in hierarchical cache (remove children?)
                else:
                    logger.warning(f"Skipping remove_node delta: missing veil_id. Op: {delta_op}")
                    
            elif op == "update_node":
                # Placeholder for future implementation
                properties = delta_op.get("properties")
                if veil_id and properties:
                     if veil_id in self._state["remote_state_cache"]:
                          logger.debug(f"Updating properties for node {veil_id} in cache via update_node delta.")
                          # Naive update: merge properties dictionary
                          # A more robust implementation might handle specific property changes
                          self._state["remote_state_cache"][veil_id]["properties"].update(properties)
                          cache_modified = True
                     else:
                          logger.debug(f"Skipping update_node delta: veil_id {veil_id} not found in cache.")
                else:
                     logger.warning(f"Skipping update_node delta: missing veil_id or properties. Op: {delta_op}")

            else:
                logger.warning(f"Unsupported delta operation received: '{op}'. Skipping.")
                
        if cache_modified:
             # Update timestamp to reflect cache freshness from deltas
             self._state["last_successful_sync"] = int(time.time() * 1000) # Treat delta application as a successful sync
             self._state["cache_expiry"]["full_state"] = self._state["last_successful_sync"] + (self._state["cache_ttl"] * 1000)
             logger.info(f"Cache for {self.remote_space_id} updated via deltas.")
             # Optionally trigger element event? uplink_state_synced might be too coarse.
             # Maybe a more specific event like "uplink_cache_updated_via_delta"?

             # --- Notify the UplinkVeilProducer --- 
             uplink_veil_producer = self.owner.get_component_by_type("UplinkVeilProducerComponent")
             if uplink_veil_producer and hasattr(uplink_veil_producer, 'on_cache_updated'):
                 # Pass the current full state of the cache after deltas applied
                 # The on_cache_updated method in UplinkVeilProducer expects the full snapshot data
                 # which in this case is self._state["remote_state_cache"]
                 logger.debug(f"Notifying UplinkVeilProducer after applying deltas. Cache size: {len(self._state.get('remote_state_cache', {}))}")
                 uplink_veil_producer.on_cache_updated(self._state.get("remote_state_cache", {}))
             # -------------------------------------

    def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handle events related to synchronization.
        """
        event_type = event.get("event_type")
        if event_type == "sync_request":
             force = event.get("data", {}).get("force", False)
             logger.info(f"Received sync request for {self.remote_space_id}, force={force}")
             return self.sync_remote_state()
        # uplink_state_synced is handled internally by sync_remote_state
        return False

    def _on_cleanup(self) -> bool:
         """Disables auto-sync on cleanup."""
         self.disable_auto_sync()
         return True 