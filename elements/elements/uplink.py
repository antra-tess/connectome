"""
Uplink Proxy Element
Implementation of the uplink proxy using components.
"""

import logging
from typing import Dict, Any, Optional, List, Callable

# Type hinting imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..space_registry import SpaceRegistry

from .base import BaseElement, MountType

from .space import Space # Inherits Space functionality (Container, Timeline)
from .components.tool_provider import ToolProviderComponent, ToolParameter
from .components.uplink import UplinkConnectionComponent, RemoteStateCacheComponent, UplinkVeilProducer
from .components.uplink.remote_tool_provider import UplinkRemoteToolProviderComponent
from elements.elements.components.uplink.connection_component import UplinkConnectionComponent
from elements.elements.components.uplink.remote_tool_provider import UplinkRemoteToolProviderComponent
from elements.elements.components.uplink.cache_component import RemoteStateCacheComponent

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
    - Representation (`UplinkVeilProducer`)
    """
    
    # Uplink has an exterior representation (this might be inferred from VeilProducer later)
    HAS_EXTERIOR = True
    IS_UPLINK_SPACE = True
    
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
                 sync_interval: int = 60, cache_ttl: int = 300,
                 space_registry: Optional['SpaceRegistry'] = None,
                 outgoing_action_callback: Optional['OutgoingActionCallback'] = None,
                 notify_owner_of_new_deltas_callback: Optional[Callable[[str, List[Dict[str, Any]]], None]] = None):
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
            space_registry: Reference to the SpaceRegistry for finding remote space
            outgoing_action_callback: Callback for local management tools if any
            notify_owner_of_new_deltas_callback: Callback for this UplinkProxy to notify its owner (InnerSpace) of new deltas, passing self.id and deltas.
        """
        super().__init__(element_id, name, description)
        
        self._space_registry = space_registry 
        self.remote_space_id = remote_space_id # Made public for easier access by components
        
        # This callback is for THIS UplinkProxy to notify ITS owner (InnerSpace)
        self._notify_owner_of_new_deltas_callback = notify_owner_of_new_deltas_callback

        # Get the ToolProviderComponent added by the parent Space class
        self._local_tool_provider: Optional[ToolProviderComponent] = self.get_component_by_type(ToolProviderComponent)
        if not self._local_tool_provider:
            # This should ideally not happen if Space.__init__ guarantees it.
            # If it can happen, we might need to add it here, but that contradicts Space's role.
            # For now, assume Space adds it and log an error if not found.
            logger.error(f"CRITICAL: ToolProviderComponent not found on UplinkProxy {self.id} after Space initialization. Local tools cannot be registered.")
            # As a fallback, try adding it, though this might indicate a deeper issue.
            self._local_tool_provider = self.add_component(ToolProviderComponent, component_id=f"{self.id}_local_tool_provider")

        # Initialize core components for Uplink functionality
        self._connection_component: UplinkConnectionComponent = self.add_component(UplinkConnectionComponent, remote_space_id=remote_space_id, space_registry=space_registry)
        self._cache_component: RemoteStateCacheComponent = self.add_component(RemoteStateCacheComponent) # Will sync using remote_space_id and space_registry
        self._veil_producer_component: UplinkVeilProducer = self.add_component(UplinkVeilProducer) # Produces VEIL from cached state
        
        self._register_uplink_tools() # Register local tools on the self._local_tool_provider

        # NEW: Add RemoteToolProvider for tools from the remote space
        self._remote_tool_provider_component: UplinkRemoteToolProviderComponent = self.add_component(UplinkRemoteToolProviderComponent)
        
        # This will hold information about the remote space, populated by RemoteStateCacheComponent
        self.remote_space_info: Dict[str, Any] = {
            "name": remote_space_info.get("name", remote_space_id) if remote_space_info else remote_space_id,
            "type": remote_space_info.get("type", "Unknown") if remote_space_info else "Unknown",
            "info": remote_space_info or {}
        }
        
        logger.info(f"Created uplink proxy: {name} ({element_id}) -> {remote_space_id}")
    
    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        # The UplinkConnectionComponent is added during __init__ or by prefab.
        # We no longer need to set its _delta_callback here as it will directly call
        # self.process_incoming_deltas_from_remote_space which is on this UplinkProxy instance.
        conn_comp = self.get_connection_component()
        if not conn_comp:
            logger.error(f"[{self.id}] UplinkConnectionComponent not found during UplinkProxy initialize. Critical for remote delta listening.")

    # --- Tool Registration --- 
    def _register_uplink_tools(self) -> None:
        """Register tools specific to uplink proxies using ToolProviderComponent."""
        if not self._local_tool_provider: # CHANGED: Use _local_tool_provider
             logger.warning(f"Cannot register uplink tools for {self.id}, local ToolProviderComponent missing.")
             return
        
        # Define parameter schemas
        sync_remote_state_params: List[ToolParameter] = [
            {"name": "force", "type": "boolean", "description": "Set true to ignore cache TTL.", "required": False}
        ]
        get_connection_spans_params: List[ToolParameter] = [
            {"name": "limit", "type": "integer", "description": "Max number of spans to return.", "required": False}
        ]
        enable_auto_sync_params: List[ToolParameter] = [
            {"name": "interval", "type": "integer", "description": "Sync interval in seconds.", "required": False}
        ]
        get_history_bundles_params: List[ToolParameter] = [
            {"name": "span_ids", "type": "array", "description": "List of span IDs to retrieve. Gets all cached if omitted.", "required": False, "items": {"type": "string"}}
        ]

        @self._local_tool_provider.register_tool( # CHANGED: Use _local_tool_provider
            name="connect_to_remote",
            description="Connect to the remote space associated with this uplink.",
            parameters_schema=[] # No parameters
        )
        def connect_to_remote_tool() -> Dict[str, Any]:
            """Tool to connect via UplinkConnectionComponent."""
            if not self._connection_component:
                return {"success": False, "error": "Connection component unavailable"}
            success = self._connection_component.connect()
            # Return current state after attempting connection
            return {
                "success": success,
                "status": "Connected." if success else "Connection failed.",
                "connection_state": self._connection_component.get_connection_state()
            }
        
        @self._local_tool_provider.register_tool( # CHANGED: Use _local_tool_provider
            name="disconnect_from_remote",
            description="Disconnect from the remote space.",
            parameters_schema=[] # No parameters
        )
        def disconnect_from_remote_tool() -> Dict[str, Any]:
            """Tool to disconnect via UplinkConnectionComponent."""
            if not self._connection_component:
                return {"success": False, "error": "Connection component unavailable"}
            success = self._connection_component.disconnect()
            return {
                "success": success,
                "status": "Disconnected." if success else "Disconnection failed.",
                "connection_state": self._connection_component.get_connection_state() # State after disconnect
            }
        
        @self._local_tool_provider.register_tool( # CHANGED: Use _local_tool_provider
            name="sync_remote_state",
            description="Manually trigger synchronization with the remote space.",
            parameters_schema=sync_remote_state_params
        )
        def sync_remote_state_tool(force: bool = False) -> Dict[str, Any]:
            """Tool to sync state via RemoteStateCacheComponent."""
            if not self._cache_component:
                 return {"success": False, "error": "Cache component unavailable"}
            # Sync is triggered here. The component handles connection checks.
            success = self._cache_component.sync_remote_state()
            status = "Sync successful." if success else "Sync failed (maybe not connected?)."
            return {
                "success": success,
                "status": status,
                "last_sync_attempt": self._cache_component._state.get("last_sync_attempt"),
                "last_successful_sync": self._cache_component._state.get("last_successful_sync")
            }
        
        @self._local_tool_provider.register_tool( # CHANGED: Use _local_tool_provider
            name="get_connection_state",
            description="Get the current connection status of the uplink.",
            parameters_schema=[] # No parameters
        )
        def get_connection_state_tool() -> Dict[str, Any]:
            """Tool to get connection state from UplinkConnectionComponent."""
            if not self._connection_component:
                 return {"success": False, "error": "Connection component unavailable"}
            return {
                "success": True,
                "connection_state": self._connection_component.get_connection_state()
            }
            
        @self._local_tool_provider.register_tool( # CHANGED: Use _local_tool_provider
            name="get_connection_spans",
            description="Get recent periods when the agent was connected via this uplink.",
            parameters_schema=get_connection_spans_params
        )
        def get_connection_spans_tool(limit: Optional[int] = None) -> Dict[str, Any]:
            """Tool to get connection spans from UplinkConnectionComponent."""
            if not self._connection_component:
                 return {"success": False, "error": "Connection component unavailable"}
            spans = self._connection_component.get_connection_spans(limit=limit)
            return {"success": True, "connection_spans": spans}
        
        @self._local_tool_provider.register_tool( # CHANGED: Use _local_tool_provider
            name="enable_auto_sync",
            description="Enable automatic background synchronization of remote state.",
            parameters_schema=enable_auto_sync_params
        )
        def enable_auto_sync_tool(interval: Optional[int] = None) -> Dict[str, Any]:
            """Tool to enable auto-sync via RemoteStateCacheComponent."""
            if not self._cache_component:
                 return {"success": False, "error": "Cache component unavailable"}
            self._cache_component.enable_auto_sync(interval)
            return {
                "success": True,
                "status": "Auto-sync enabled.",
                "auto_sync_is_active": self._cache_component._auto_sync_enabled, # Check state after enabling
                "check_interval_approx_seconds": self._cache_component._state.get("cache_ttl")
            }
        
        @self._local_tool_provider.register_tool( # CHANGED: Use _local_tool_provider
            name="disable_auto_sync",
            description="Disable automatic background synchronization.",
            parameters_schema=[] # No parameters
        )
        def disable_auto_sync_tool() -> Dict[str, Any]:
            """Tool to disable auto-sync via RemoteStateCacheComponent."""
            if not self._cache_component:
                 return {"success": False, "error": "Cache component unavailable"}
            self._cache_component.disable_auto_sync()
            return {"success": True, "status": "Auto-sync disabled.", "auto_sync_is_active": self._cache_component._auto_sync_enabled}
            
        # Tool to get cached history bundles (useful for rendering/VEIL)
        @self._local_tool_provider.register_tool( # CHANGED: Use _local_tool_provider
            name="get_history_bundles",
            description="Get cached history bundles from the remote space.",
            parameters_schema=get_history_bundles_params
        )
        def get_history_bundles_tool(span_ids: Optional[List[str]] = None) -> Dict[str, Any]:
            """Tool to get history bundles from RemoteStateCacheComponent."""
            if not self._cache_component:
                 return {"success": False, "error": "Cache component unavailable", "history_bundles": {}}
            bundles = self._cache_component.get_history_bundles(span_ids=span_ids)
            return {"success": True, "history_bundles": bundles}
            
    # --- State Representation (Could be customized in a dedicated VeilProducer subclass) ---
    def get_interior_state(self) -> Dict[str, Any]:
        """
        Aggregates state from components for a detailed view.
        (This might be superseded by VEIL generation in the future)
        """
        interior = super().get_interior_state() # Gets state from Space components (Timeline, Container)
        
        conn_state = self._connection_component.get_connection_state() if self._connection_component else {}
        # Get potentially synced remote state from cache
        # Pass force=False to avoid triggering sync just for viewing state
        cache_content = self._cache_component.get_synced_remote_state(force_sync=False) if self._cache_component else {}
        history_bundles = self._cache_component.get_history_bundles() if self._cache_component else {}
        conn_spans = self._connection_component.get_connection_spans() if self._connection_component else []
        
        interior.update({
            "element_type": "UplinkProxy",
            "remote_space_id": self.remote_space_id,
            "remote_space_info": self.remote_space_info,
            "connection": conn_state,
            "cache": {
                 "content": cache_content,
                 "history_bundles": history_bundles,
                 "last_successful_sync": self._cache_component._state.get("last_successful_sync") if self._cache_component else None,
                 "auto_sync_enabled": self._cache_component._auto_sync_enabled if self._cache_component else False
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
             
        conn_state = self._connection_component.get_connection_state() if self._connection_component else {}
        last_sync = self._cache_component._state.get("last_successful_sync") if self._cache_component else None

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
        return self._connection_component
        
    def get_cache_component(self) -> Optional[RemoteStateCacheComponent]:
        return self._cache_component

    def process_incoming_deltas_from_remote_space(self, deltas: List[Dict[str, Any]]):
        """
        Called by the remote SharedSpace (via UplinkConnectionComponent's registration)
        when new VEIL deltas are available from the remote space.
        """
        logger.info(f"[{self.id}] UplinkProxy processing {len(deltas)} incoming deltas from remote space '{self.remote_space_id}'.")
        
        # 1. Apply deltas to local cache
        if self._cache_component:
            try:
                self._cache_component.apply_deltas(deltas)
                logger.debug(f"[{self.id}] Applied deltas to RemoteStateCacheComponent.")
            except Exception as e:
                logger.error(f"[{self.id}] Error applying deltas in cache component: {e}", exc_info=True)
        else:
            logger.warning(f"[{self.id}] Cannot apply remote deltas: RemoteStateCacheComponent not found.")

        # 2. Notify the owner of this UplinkProxy (the InnerSpace) about these deltas
        if self._notify_owner_of_new_deltas_callback:
            try:
                logger.debug(f"[{self.id}] Notifying owner (InnerSpace) of new deltas from remote space '{self.remote_space_id}'.")
                self._notify_owner_of_new_deltas_callback(self.id, deltas) # Pass uplink_id and deltas
            except Exception as e:
                logger.error(f"[{self.id}] Error calling _notify_owner_of_new_deltas_callback: {e}", exc_info=True)
        else:
            logger.warning(f"[{self.id}] No owner notification callback configured (_notify_owner_of_new_deltas_callback is None). InnerSpace will not be directly informed of these deltas.")

    # --- Event Handling --- 
    # Default delegation via super().handle_event is likely sufficient unless 
    # specific coordination between UplinkConnection and RemoteStateCache is needed.
    # For example, maybe a successful connect should trigger an immediate sync request?
    # def handle_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
    #     handled = super().handle_event(event, timeline_context)
    #     event_type = event.get("event_type")
    #     if event_type == "uplink_connected":
    #          logger.info(f"Uplink {self.id} connected, requesting initial sync.")
    #          if self._cache_component:
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
        spans = self._connection_component.get_connection_spans(limit=limit)
        
        # Include active span if requested
        if include_active and self._connection_component.get_connection_spans(limit=1):
            spans.append(self._connection_component.get_connection_spans(limit=1)[0])
            
        # Sort by start time (newest first)
        spans.sort(key=lambda span: span.get('start_time', 0), reverse=True)
        
        # Limit the number of spans
        if limit > 0:
            spans = spans[:limit]
            
        return spans 