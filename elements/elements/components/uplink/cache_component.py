"""
Remote State Cache Component
Component for caching and retrieving state from a remote space via an Uplink.
"""

import logging
from typing import Dict, Any, Optional, List, Set
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
    
    def __init__(self, element=None, cache_ttl: int = 300, remote_space_id: Optional[str] = None):
        super().__init__(element)
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

    def _get_connection_comp(self) -> Optional[UplinkConnectionComponent]:
        """Helper to get the associated UplinkConnectionComponent."""
        if not self.element:
            return None
        return self.element.get_component_by_type("uplink_connection")

    def sync_remote_state(self) -> bool:
        """
        Attempts to synchronize state with the remote space.
        (Simulated for now - real implementation involves network calls)
        
        Returns:
            True if sync was successful, False otherwise.
        """
        if not self._is_initialized or not self._is_enabled:
            logger.warning(f"{self.COMPONENT_TYPE}: Cannot sync, component not ready.")
            return False
            
        conn_comp = self._get_connection_comp()
        if not conn_comp or not conn_comp.get_connection_state().get("connected"):
             logger.warning(f"Cannot sync {self.remote_space_id}: Uplink not connected.")
             return False
             
        timestamp = int(time.time() * 1000)
        self._state["last_sync_attempt"] = timestamp
        logger.info(f"Attempting to sync remote state for: {self.remote_space_id}")
        
        # --- Simulate Fetching Remote State --- 
        # Real implementation: network call to remote space API
        # This would fetch element states, maybe recent history based on spans, etc.
        sync_successful = True
        fetched_state = {
             f"remote_element_{uuid.uuid4().hex[:4]}": {"name": "Remote Item", "value": time.time()},
             # ... other simulated remote data ...
        }
        fetched_history_bundles = {
             f"span_{timestamp-10000}_{timestamp}": [{"event": "remote_event_1"}, {"event": "remote_event_2"}]
        }
        error_details = None
        # -------------------------------------
        
        if sync_successful:
            self._state["last_successful_sync"] = timestamp
            
            # Update cache (simple replacement for now, could be more granular)
            self._state["remote_state_cache"] = fetched_state
            self._state["history_bundles"].update(fetched_history_bundles) # Merge history
            
            # Update expiry for the whole cache (or individual items)
            self._state["cache_expiry"]["full_state"] = timestamp + (self._state["cache_ttl"] * 1000)
            
            logger.info(f"Successfully synced state for {self.remote_space_id}")
            # Notify element that sync occurred (can trigger VEIL updates etc.)
            if self.element:
                 self.element.handle_event({
                      "event_type": "uplink_state_synced",
                      "data": {"remote_space_id": self.remote_space_id, "cache_size": len(self._state["remote_state_cache"]) }
                 }, {"timeline_id": "primary"}) # Assuming primary timeline for element event
            return True
        else:
            logger.error(f"Failed to sync state for {self.remote_space_id}: {error_details}")
            # Optionally record sync error event in element's timeline?
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
            logger.warning(f"[{self.element.id if self.element else 'cache'}] Invalid delta format received: not a list.")
            return

        logger.debug(f"[{self.element.id if self.element else 'cache'}] Applying {len(deltas)} deltas to cache for {self.remote_space_id}.")
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