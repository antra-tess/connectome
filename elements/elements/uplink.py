"""
Uplink Proxy Element
Implementation of the uplink proxy for connecting to remote spaces.
"""

import logging
from typing import Dict, Any, Optional, List, Set, Callable
import uuid
import time
from datetime import datetime, timedelta

from .base import BaseElement, MountType, ElementState
from .space import Space

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UplinkProxy(Space):
    """
    Proxy element for connecting to remote spaces.
    
    The UplinkProxy maintains a local representation of a remote space,
    providing access to its state and elements. It handles:
    
    - Maintaining connection state with the remote space
    - Syncing state changes between local and remote
    - Providing access to remote elements through proxies
    - Managing caching and invalidation of remote state
    
    Unlike normal Inclusions, Uplinks don't manage the lifecycle of the
    remote space but simply provide a view into it.
    """
    
    # Uplink has an exterior representation
    HAS_EXTERIOR = True
    
    # Events that uplinks handle
    EVENT_TYPES = Space.EVENT_TYPES + [
        "uplink_connected", 
        "uplink_disconnected",
        "uplink_state_synced",
        "uplink_error"
    ]
    
    def __init__(self, element_id: str, name: str, description: str, 
                 remote_space_id: str, remote_space_info: Optional[Dict[str, Any]] = None):
        """
        Initialize the uplink proxy.
        
        Args:
            element_id: Unique identifier for this uplink
            name: Human-readable name for this uplink
            description: Description of this uplink's purpose
            remote_space_id: ID of the remote space to connect to
            remote_space_info: Optional information about the remote space
        """
        super().__init__(element_id, name, description)
        
        # Remote space information
        self.remote_space_id = remote_space_id
        self.remote_space_info = remote_space_info or {}
        
        # Connection state
        self._connection_state = {
            "connected": False,
            "last_sync": None,
            "sync_interval": 60,  # seconds
            "error": None,
            "connection_history": []
        }
        
        # Cache of remote state
        self._remote_state_cache = {}
        
        # Cache expiration timestamps
        self._cache_expiry = {}
        
        # Automatic sync timer
        self._auto_sync_enabled = False
        
        # Initialize uplink-specific tools
        self._register_uplink_tools()
        
        logger.info(f"Created uplink proxy: {name} ({element_id}) -> {remote_space_id}")
    
    def _register_uplink_tools(self) -> None:
        """Register tools specific to uplink proxies."""
        
        @self.register_tool(
            name="connect_to_remote",
            description="Connect to the remote space",
            parameter_descriptions={}
        )
        def connect_to_remote_tool() -> Dict[str, Any]:
            """
            Tool to connect to the remote space.
            
            Returns:
                Result of the connection operation
            """
            success = self.connect_to_remote()
            
            return {
                "success": success,
                "remote_space_id": self.remote_space_id,
                "connection_state": self._connection_state
            }
        
        @self.register_tool(
            name="disconnect_from_remote",
            description="Disconnect from the remote space",
            parameter_descriptions={}
        )
        def disconnect_from_remote_tool() -> Dict[str, Any]:
            """
            Tool to disconnect from the remote space.
            
            Returns:
                Result of the disconnection operation
            """
            success = self.disconnect_from_remote()
            
            return {
                "success": success,
                "remote_space_id": self.remote_space_id,
                "connection_state": self._connection_state
            }
        
        @self.register_tool(
            name="sync_remote_state",
            description="Manually sync the remote space state",
            parameter_descriptions={}
        )
        def sync_remote_state_tool() -> Dict[str, Any]:
            """
            Tool to manually sync the remote space state.
            
            Returns:
                Result of the sync operation
            """
            success = self.sync_remote_state()
            
            return {
                "success": success,
                "remote_space_id": self.remote_space_id,
                "last_sync": self._connection_state["last_sync"],
                "cache_size": len(self._remote_state_cache)
            }
        
        @self.register_tool(
            name="get_connection_state",
            description="Get the current connection state",
            parameter_descriptions={}
        )
        def get_connection_state_tool() -> Dict[str, Any]:
            """
            Tool to get the current connection state.
            
            Returns:
                Current connection state
            """
            return {
                "success": True,
                "remote_space_id": self.remote_space_id,
                "connection_state": self._connection_state
            }
        
        @self.register_tool(
            name="enable_auto_sync",
            description="Enable automatic sync of remote state",
            parameter_descriptions={
                "interval": "Sync interval in seconds"
            }
        )
        def enable_auto_sync_tool(interval: int = 60) -> Dict[str, Any]:
            """
            Tool to enable automatic sync of remote state.
            
            Args:
                interval: Sync interval in seconds
                
            Returns:
                Result of the operation
            """
            self._connection_state["sync_interval"] = interval
            self._auto_sync_enabled = True
            
            return {
                "success": True,
                "auto_sync": True,
                "interval": interval
            }
        
        @self.register_tool(
            name="disable_auto_sync",
            description="Disable automatic sync of remote state",
            parameter_descriptions={}
        )
        def disable_auto_sync_tool() -> Dict[str, Any]:
            """
            Tool to disable automatic sync of remote state.
            
            Returns:
                Result of the operation
            """
            self._auto_sync_enabled = False
            
            return {
                "success": True,
                "auto_sync": False
            }
    
    def get_interior_state(self) -> Dict[str, Any]:
        """
        Get the interior state of the uplink.
        
        Returns:
            Dictionary representation of the interior state
        """
        # Get basic interior state from Space
        interior = super().get_interior_state()
        
        # Add uplink-specific state
        interior.update({
            "remote_space_id": self.remote_space_id,
            "remote_space_info": self.remote_space_info,
            "connection_state": self._connection_state,
            "remote_state": self._get_synced_remote_state()
        })
        
        return interior
    
    def get_exterior_state(self) -> Dict[str, Any]:
        """
        Get the exterior state of the uplink.
        
        Returns:
            Dictionary representation of the exterior state
        """
        # Get basic exterior state from Space
        exterior = super().get_exterior_state()
        
        # Add uplink-specific state
        exterior.update({
            "remote_space_id": self.remote_space_id,
            "connected": self._connection_state["connected"],
            "last_sync": self._connection_state["last_sync"],
            "has_error": self._connection_state["error"] is not None
        })
        
        return exterior
    
    def connect_to_remote(self) -> bool:
        """
        Connect to the remote space.
        
        Returns:
            True if connection was successful, False otherwise
        """
        # In a real implementation, this would establish a connection
        # with the remote space and retrieve its initial state
        try:
            # Mock successful connection
            self._connection_state["connected"] = True
            self._connection_state["last_sync"] = int(time.time() * 1000)
            self._connection_state["error"] = None
            self._connection_state["connection_history"].append({
                "action": "connect",
                "timestamp": int(time.time() * 1000),
                "success": True
            })
            
            # Record the connection event
            event_data = {
                "event_type": "uplink_connected",
                "uplink_id": self.id,
                "remote_space_id": self.remote_space_id,
                "timestamp": int(time.time() * 1000)
            }
            
            # Get primary timeline
            timeline_id = self._timeline_state["primary_timeline"]
            if timeline_id is not None:
                self.update_state(event_data, {"timeline_id": timeline_id})
            
            logger.info(f"Connected uplink {self.id} to remote space {self.remote_space_id}")
            
            # Sync remote state
            self.sync_remote_state()
            
            # Notify observers
            self.notify_observers({
                "type": "uplink_connected",
                "uplink_id": self.id,
                "remote_space_id": self.remote_space_id
            })
            
            return True
        except Exception as e:
            self._connection_state["connected"] = False
            self._connection_state["error"] = str(e)
            self._connection_state["connection_history"].append({
                "action": "connect",
                "timestamp": int(time.time() * 1000),
                "success": False,
                "error": str(e)
            })
            
            logger.error(f"Error connecting uplink {self.id} to remote space {self.remote_space_id}: {e}")
            
            # Notify observers
            self.notify_observers({
                "type": "uplink_error",
                "uplink_id": self.id,
                "remote_space_id": self.remote_space_id,
                "error": str(e)
            })
            
            return False
    
    def disconnect_from_remote(self) -> bool:
        """
        Disconnect from the remote space.
        
        Returns:
            True if disconnection was successful, False otherwise
        """
        # In a real implementation, this would close the connection
        # with the remote space
        try:
            # Mock successful disconnection
            self._connection_state["connected"] = False
            self._connection_state["connection_history"].append({
                "action": "disconnect",
                "timestamp": int(time.time() * 1000),
                "success": True
            })
            
            # Record the disconnection event
            event_data = {
                "event_type": "uplink_disconnected",
                "uplink_id": self.id,
                "remote_space_id": self.remote_space_id,
                "timestamp": int(time.time() * 1000)
            }
            
            # Get primary timeline
            timeline_id = self._timeline_state["primary_timeline"]
            if timeline_id is not None:
                self.update_state(event_data, {"timeline_id": timeline_id})
            
            logger.info(f"Disconnected uplink {self.id} from remote space {self.remote_space_id}")
            
            # Notify observers
            self.notify_observers({
                "type": "uplink_disconnected",
                "uplink_id": self.id,
                "remote_space_id": self.remote_space_id
            })
            
            return True
        except Exception as e:
            self._connection_state["error"] = str(e)
            self._connection_state["connection_history"].append({
                "action": "disconnect",
                "timestamp": int(time.time() * 1000),
                "success": False,
                "error": str(e)
            })
            
            logger.error(f"Error disconnecting uplink {self.id} from remote space {self.remote_space_id}: {e}")
            
            # Notify observers
            self.notify_observers({
                "type": "uplink_error",
                "uplink_id": self.id,
                "remote_space_id": self.remote_space_id,
                "error": str(e)
            })
            
            return False
    
    def sync_remote_state(self) -> bool:
        """
        Sync the remote space state.
        
        Returns:
            True if sync was successful, False otherwise
        """
        # In a real implementation, this would retrieve the current state
        # of the remote space and update the local cache
        try:
            if not self._connection_state["connected"]:
                logger.warning(f"Cannot sync uplink {self.id} - not connected to remote space {self.remote_space_id}")
                return False
                
            # Mock successful sync
            self._connection_state["last_sync"] = int(time.time() * 1000)
            
            # In a real implementation, this would update the cache with
            # actual remote state data
            self._remote_state_cache = {
                "mock_remote_state": {
                    "timestamp": int(time.time() * 1000),
                    "data": {
                        "name": self.remote_space_info.get("name", "Unknown Space"),
                        "description": self.remote_space_info.get("description", "No description"),
                        "element_count": len(self.get_mounted_elements())
                    }
                }
            }
            
            # Set cache expiry
            now = datetime.now()
            self._cache_expiry = {
                "mock_remote_state": now + timedelta(minutes=5)
            }
            
            # Record the sync event
            event_data = {
                "event_type": "uplink_state_synced",
                "uplink_id": self.id,
                "remote_space_id": self.remote_space_id,
                "timestamp": int(time.time() * 1000)
            }
            
            # Get primary timeline
            timeline_id = self._timeline_state["primary_timeline"]
            if timeline_id is not None:
                self.update_state(event_data, {"timeline_id": timeline_id})
            
            logger.info(f"Synced uplink {self.id} with remote space {self.remote_space_id}")
            
            # Notify observers
            self.notify_observers({
                "type": "uplink_state_synced",
                "uplink_id": self.id,
                "remote_space_id": self.remote_space_id
            })
            
            return True
        except Exception as e:
            self._connection_state["error"] = str(e)
            
            logger.error(f"Error syncing uplink {self.id} with remote space {self.remote_space_id}: {e}")
            
            # Notify observers
            self.notify_observers({
                "type": "uplink_error",
                "uplink_id": self.id,
                "remote_space_id": self.remote_space_id,
                "error": str(e)
            })
            
            return False
    
    def _get_synced_remote_state(self) -> Dict[str, Any]:
        """
        Get the synced remote state, refreshing if needed.
        
        Returns:
            Dictionary of remote state
        """
        # Check if we need to sync
        if self._auto_sync_enabled and self._connection_state["connected"]:
            now = int(time.time() * 1000)
            last_sync = self._connection_state["last_sync"] or 0
            sync_interval = self._connection_state["sync_interval"] * 1000  # Convert to ms
            
            if now - last_sync > sync_interval:
                self.sync_remote_state()
        
        # Check cache expiry
        now = datetime.now()
        expired_keys = []
        for key, expiry in self._cache_expiry.items():
            if now > expiry:
                expired_keys.append(key)
                
        # Remove expired items
        for key in expired_keys:
            if key in self._remote_state_cache:
                del self._remote_state_cache[key]
            if key in self._cache_expiry:
                del self._cache_expiry[key]
                
        return self._remote_state_cache
    
    def is_connected(self) -> bool:
        """
        Check if the uplink is connected to the remote space.
        
        Returns:
            True if connected, False otherwise
        """
        return self._connection_state["connected"]
    
    def _on_open(self) -> None:
        """
        Hook called when the uplink is opened.
        
        Opens all elements mounted as inclusions and connects to the remote space.
        """
        # Call parent implementation to open inclusions
        super()._on_open()
        
        # Connect to remote space if not already connected
        if not self._connection_state["connected"]:
            self.connect_to_remote()
    
    def _on_close(self) -> None:
        """
        Hook called when the uplink is closed.
        
        Closes all elements mounted as inclusions but keeps the remote connection.
        """
        # Call parent implementation to close inclusions
        super()._on_close()
        
        # We don't disconnect from the remote space when closing,
        # only when explicitly told to disconnect or when unmounted 