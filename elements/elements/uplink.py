"""
Uplink Proxy Element
Implementation of the uplink proxy using components.
"""

import logging
from typing import Dict, Any, Optional, List
import uuid
import time

from .space import Space # Inherits Space functionality (Container, Timeline)
from .components import ToolProviderComponent, VeilProducer
from .components.uplink import UplinkConnectionComponent, RemoteStateCacheComponent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UplinkProxy(Space):
    """
    Proxy element for connecting to remote spaces, using component architecture.
    
    Inherits Space functionality (ContainerComponent, TimelineComponent).
    Adds components for:
    - Uplink connection management (`UplinkConnectionComponent`)
    - Remote state caching (`RemoteStateCacheComponent`)
    - Uplink-specific tools (`ToolProviderComponent`)
    - Representation (`VeilProducer`)
    """
    
    # Uplink has an exterior representation (this might be inferred from VeilProducer later)
    HAS_EXTERIOR = True
    
    # Events specific to Uplink lifecycle are handled by components, 
    # but the element might observe them.
    EVENT_TYPES = Space.EVENT_TYPES + [
        "uplink_connected", 
        "uplink_disconnected",
        "uplink_state_synced",
        "uplink_error"
    ]
    
    def __init__(self, element_id: str, name: str, description: str, 
                 remote_space_id: str, remote_space_info: Optional[Dict[str, Any]] = None,
                 sync_interval: int = 60, cache_ttl: int = 300):
        """
        Initialize the uplink proxy.
        
        Args:
            element_id: Unique identifier for this uplink
            name: Human-readable name for this uplink
            description: Description of this uplink's purpose
            remote_space_id: ID of the remote space to connect to
            remote_space_info: Optional info about the remote space (e.g., name, type)
            sync_interval: Default interval for connection component (may be obsolete)
            cache_ttl: Default TTL for the remote state cache component
        """
        super().__init__(element_id, name, description)
        
        # Store basic remote info accessible to tools/VEIL
        self.remote_space_id = remote_space_id
        self.remote_space_info = remote_space_info or {}
        
        # --- Add Uplink Components --- 
        self._connection_comp = self.add_component(UplinkConnectionComponent, 
                                                   remote_space_id=remote_space_id,
                                                   sync_interval=sync_interval)
        if not self._connection_comp:
            logger.error(f"Failed to add UplinkConnectionComponent to UplinkProxy {element_id}")
            
        self._cache_comp = self.add_component(RemoteStateCacheComponent, 
                                              remote_space_id=remote_space_id, 
                                              cache_ttl=cache_ttl)
        if not self._cache_comp:
             logger.error(f"Failed to add RemoteStateCacheComponent to UplinkProxy {element_id}")

        self._tool_provider = self.add_component(ToolProviderComponent)
        if not self._tool_provider:
             logger.error(f"Failed to add ToolProviderComponent component to UplinkProxy {element_id}")
        else:
             self._register_uplink_tools() # Register tools if provider exists
             
        # Basic Veil Producer - A specific UplinkVeilProducer could override methods
        self._veil_producer = self.add_component(VeilProducer, renderable_id=f"uplink_{element_id}")
        if not self._veil_producer:
             logger.error(f"Failed to add VeilProducer component to UplinkProxy {element_id}")

        # Note: ContainerComponent and TimelineComponent are added by the parent Space class
        
        logger.info(f"Created uplink proxy: {name} ({element_id}) -> {remote_space_id}")
    
    # --- Tool Registration --- 
    def _register_uplink_tools(self) -> None:
        """Register tools specific to uplink proxies using ToolProviderComponent."""
        if not self._tool_provider:
             logger.warning(f"Cannot register uplink tools for {self.id}, ToolProviderComponent missing.")
             return
        
        @self._tool_provider.register_tool(
            name="connect_to_remote",
            description="Connect to the remote space associated with this uplink.",
            parameter_descriptions={}
        )
        def connect_to_remote_tool() -> Dict[str, Any]:
            """Tool to connect via UplinkConnectionComponent."""
            if not self._connection_comp:
                return {"success": False, "error": "Connection component unavailable"}
            success = self._connection_comp.connect()
            # Return current state after attempting connection
            return {
                "success": success,
                "status": "Connected." if success else "Connection failed.",
                "connection_state": self._connection_comp.get_connection_state()
            }
        
        @self._tool_provider.register_tool(
            name="disconnect_from_remote",
            description="Disconnect from the remote space.",
            parameter_descriptions={}
        )
        def disconnect_from_remote_tool() -> Dict[str, Any]:
            """Tool to disconnect via UplinkConnectionComponent."""
            if not self._connection_comp:
                return {"success": False, "error": "Connection component unavailable"}
            success = self._connection_comp.disconnect()
            return {
                "success": success,
                "status": "Disconnected." if success else "Disconnection failed.",
                "connection_state": self._connection_comp.get_connection_state() # State after disconnect
            }
        
        @self._tool_provider.register_tool(
            name="sync_remote_state",
            description="Manually trigger synchronization with the remote space.",
            parameter_descriptions={"force": "Boolean, set true to ignore cache TTL (optional, default false)"}
        )
        def sync_remote_state_tool(force: bool = False) -> Dict[str, Any]:
            """Tool to sync state via RemoteStateCacheComponent."""
            if not self._cache_comp:
                 return {"success": False, "error": "Cache component unavailable"}
            # Sync is triggered here. The component handles connection checks.
            success = self._cache_comp.sync_remote_state()
            status = "Sync successful." if success else "Sync failed (maybe not connected?)."
            return {
                "success": success,
                "status": status,
                "last_sync_attempt": self._cache_comp._state.get("last_sync_attempt"),
                "last_successful_sync": self._cache_comp._state.get("last_successful_sync")
            }
        
        @self._tool_provider.register_tool(
            name="get_connection_state",
            description="Get the current connection status of the uplink.",
            parameter_descriptions={}
        )
        def get_connection_state_tool() -> Dict[str, Any]:
            """Tool to get connection state from UplinkConnectionComponent."""
            if not self._connection_comp:
                 return {"success": False, "error": "Connection component unavailable"}
            return {
                "success": True,
                "connection_state": self._connection_comp.get_connection_state()
            }
            
        @self._tool_provider.register_tool(
            name="get_connection_spans",
            description="Get recent periods when the agent was connected via this uplink.",
            parameter_descriptions={"limit": "Max number of spans to return (optional)"}
        )
        def get_connection_spans_tool(limit: Optional[int] = None) -> Dict[str, Any]:
            """Tool to get connection spans from UplinkConnectionComponent."""
            if not self._connection_comp:
                 return {"success": False, "error": "Connection component unavailable"}
            spans = self._connection_comp.get_connection_spans(limit=limit)
            return {"success": True, "connection_spans": spans}
        
        @self._tool_provider.register_tool(
            name="enable_auto_sync",
            description="Enable automatic background synchronization of remote state.",
            parameter_descriptions={
                "interval": "Sync interval in seconds (optional, defaults to cache TTL)"
            }
        )
        def enable_auto_sync_tool(interval: Optional[int] = None) -> Dict[str, Any]:
            """Tool to enable auto-sync via RemoteStateCacheComponent."""
            if not self._cache_comp:
                 return {"success": False, "error": "Cache component unavailable"}
            self._cache_comp.enable_auto_sync(interval)
            return {
                "success": True,
                "status": "Auto-sync enabled.",
                "auto_sync_is_active": self._cache_comp._auto_sync_enabled, # Check state after enabling
                "check_interval_approx_seconds": self._cache_comp._state.get("cache_ttl")
            }
        
        @self._tool_provider.register_tool(
            name="disable_auto_sync",
            description="Disable automatic background synchronization.",
            parameter_descriptions={}
        )
        def disable_auto_sync_tool() -> Dict[str, Any]:
            """Tool to disable auto-sync via RemoteStateCacheComponent."""
            if not self._cache_comp:
                 return {"success": False, "error": "Cache component unavailable"}
            self._cache_comp.disable_auto_sync()
            return {"success": True, "status": "Auto-sync disabled.", "auto_sync_is_active": self._cache_comp._auto_sync_enabled}
            
        # Tool to get cached history bundles (useful for rendering/VEIL)
        @self._tool_provider.register_tool(
            name="get_history_bundles",
            description="Get cached history bundles from the remote space.",
            parameter_descriptions={ "span_ids": "List of span IDs to retrieve (optional, gets all cached if omitted)" }
        )
        def get_history_bundles_tool(span_ids: Optional[List[str]] = None) -> Dict[str, Any]:
            """Tool to get history bundles from RemoteStateCacheComponent."""
            if not self._cache_comp:
                 return {"success": False, "error": "Cache component unavailable", "history_bundles": {}}
            bundles = self._cache_comp.get_history_bundles(span_ids=span_ids)
            return {"success": True, "history_bundles": bundles}
            
    # --- State Representation (Could be customized in a dedicated VeilProducer subclass) ---
    def get_interior_state(self) -> Dict[str, Any]:
        """
        Aggregates state from components for a detailed view.
        (This might be superseded by VEIL generation in the future)
        """
        interior = super().get_interior_state() # Gets state from Space components (Timeline, Container)
        
        conn_state = self._connection_comp.get_connection_state() if self._connection_comp else {}
        # Get potentially synced remote state from cache
        # Pass force=False to avoid triggering sync just for viewing state
        cache_content = self._cache_comp.get_synced_remote_state(force_sync=False) if self._cache_comp else {}
        history_bundles = self._cache_comp.get_history_bundles() if self._cache_comp else {}
        conn_spans = self._connection_comp.get_connection_spans() if self._connection_comp else []
        
        interior.update({
            "element_type": "UplinkProxy",
            "remote_space_id": self.remote_space_id,
            "remote_space_info": self.remote_space_info,
            "connection": conn_state,
            "cache": {
                 "content": cache_content,
                 "history_bundles": history_bundles,
                 "last_successful_sync": self._cache_comp._state.get("last_successful_sync") if self._cache_comp else None,
                 "auto_sync_enabled": self._cache_comp._auto_sync_enabled if self._cache_comp else False
            },
            "connection_spans": conn_spans
        })
        return interior
    
    def get_exterior_state(self) -> Optional[Dict[str, Any]]:
        """
        Provides a compact summary, useful when closed or for lists.
        (This should ideally be generated by the VeilProducer)
        """
        if not self.HAS_EXTERIOR:
            return None
            
        exterior = super().get_exterior_state()
        if exterior is None:
             exterior = {
                  "id": self.id,
                  "name": self.name,
                  "type": self.__class__.__name__
             }
             
        conn_state = self._connection_comp.get_connection_state() if self._connection_comp else {}
        last_sync = self._cache_comp._state.get("last_successful_sync") if self._cache_comp else None

        exterior.update({
            "element_type": "UplinkProxy",
            "remote_space_id": self.remote_space_id,
            "remote_name": self.remote_space_info.get("name", self.remote_space_id),
            "connected": conn_state.get("connected", False),
            "last_sync": last_sync,
            "error": conn_state.get("error")
        })
        return exterior
    
    # --- Convenience Getters --- 
    def get_connection_component(self) -> Optional[UplinkConnectionComponent]:
        return self._connection_comp
        
    def get_cache_component(self) -> Optional[RemoteStateCacheComponent]:
        return self._cache_comp

    # --- Event Handling --- 
    # Default delegation via super().handle_event is likely sufficient unless 
    # specific coordination between UplinkConnection and RemoteStateCache is needed.
    # For example, maybe a successful connect should trigger an immediate sync request?
    # def handle_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
    #     handled = super().handle_event(event, timeline_context)
    #     event_type = event.get("event_type")
    #     if event_type == "uplink_connected":
    #          logger.info(f"Uplink {self.id} connected, requesting initial sync.")
    #          if self._cache_comp:
    #               # Trigger sync request event
    #               self.handle_event({"event_type": "sync_request", "data": {"force": True}}, timeline_context)
    #               handled = True # Consider it handled
    #     return handled

    # --- Obsolete Methods --- 
    # Direct methods for connect, disconnect, sync, state management are removed.
    # Functionality is now in components, accessed via tools or internal logic.

    def mount_element(self, element: BaseElement, mount_id: Optional[str] = None, 
                     mount_type: MountType = MountType.INCLUSION) -> bool:
        """
        Mount an element in this space.
        
        Args:
            element: Element to mount
            mount_id: Optional identifier for the mount point
            mount_type: Type of mounting
            
        Returns:
            True if the element was successfully mounted, False otherwise
        """
        # Use the parent implementation to mount the element
        success = super().mount_element(element, mount_id, mount_type)
        
        if success:
            # If this is a chat element being mounted with uplink, mark it as remote
            if hasattr(element, 'set_as_remote') and callable(getattr(element, 'set_as_remote')):
                element.set_as_remote(True)
            
            logger.info(f"Mounted element {element.id} in uplink proxy {self.id}")
        
        return success
    
    def get_connection_spans(self, options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Get connection spans for remote rendering.
        
        Args:
            options: Options for retrieving spans
                - limit: Maximum number of spans to retrieve (default 5)
                - include_active: Whether to include the active span (default True)
                
        Returns:
            List of connection spans
        """
        options = options or {}
        limit = options.get('limit', 5)
        include_active = options.get('include_active', True)
        
        # Start with completed spans
        spans = self._connection_comp.get_connection_spans(limit=limit)
        
        # Include active span if requested
        if include_active and self._connection_comp.get_connection_spans(limit=1):
            spans.append(self._connection_comp.get_connection_spans(limit=1)[0])
            
        # Sort by start time (newest first)
        spans.sort(key=lambda span: span.get('start_time', 0), reverse=True)
        
        # Limit the number of spans
        if limit > 0:
            spans = spans[:limit]
            
        return spans 