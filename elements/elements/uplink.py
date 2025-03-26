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
    - Tracking connection spans when an agent was connected to a remote space
    - Encapsulating remote space history within that space's context
    
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
        
        # Connection spans tracking 
        # Each span represents a period when the agent was connected to the remote space
        # This aligns with architecture which requires tracking "when an agent was connected"
        self._connection_spans = []
        self._current_span = None
        
        # History bundles from remote space
        # According to architecture: "History from a remote space is retrieved as bundles"
        self._history_bundles = {}
        
        # Track which remote histories we've seen (to avoid duplicating events)
        self._processed_remote_events = set()
        
        # Automatic sync timer
        self._auto_sync_enabled = False
        
        # Initialize uplink-specific tools
        self._register_uplink_tools()
        
        # Update the delegate
        self.update_delegate()
        
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
            "remote_state": self._get_synced_remote_state(),
            "history_bundles": self.get_history_bundles(),
            "connection_spans": self._connection_spans.copy(),
            "current_span": self._current_span.copy() if self._current_span else None
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
        
        # Add uplink-specific state (compact representation)
        exterior.update({
            "remote_space_id": self.remote_space_id,
            "remote_name": self.remote_space_info.get("name", "Unknown Space"),
            "connected": self._connection_state["connected"],
            "last_sync": self._connection_state["last_sync"],
            "has_error": self._connection_state["error"] is not None,
            "span_count": len(self._connection_spans) + (1 if self._current_span else 0),
            "current_span_id": self._current_span["span_id"] if self._current_span else None,
            "event_count": sum(len(span["events"]) for span in self._connection_spans) + 
                           (len(self._current_span["events"]) if self._current_span else 0)
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
            
            # Start a new connection span
            current_time = int(time.time() * 1000)
            self._current_span = {
                "span_id": f"span_{uuid.uuid4().hex[:8]}",
                "start_time": current_time,
                "end_time": None,
                "remote_space_id": self.remote_space_id,
                "events": []
            }
            
            # Record the connection event
            event_data = {
                "event_type": "uplink_connected",
                "uplink_id": self.id,
                "remote_space_id": self.remote_space_id,
                "timestamp": current_time,
                "span_id": self._current_span["span_id"]
            }
            
            # Get primary timeline
            timeline_id = self._timeline_state["primary_timeline"]
            if timeline_id is not None:
                # Store just the connection event in the agent's timeline
                # According to architecture: "Connecting to a remote space is recorded as a single event in the agent's Inner Space DAG"
                self.update_state(event_data, {"timeline_id": timeline_id})
            
            logger.info(f"Connected uplink {self.id} to remote space {self.remote_space_id}")
            
            # Sync remote state
            self.sync_remote_state()
            
            # Notify observers
            self.notify_observers({
                "type": "uplink_connected",
                "uplink_id": self.id,
                "remote_space_id": self.remote_space_id,
                "span_id": self._current_span["span_id"]
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
            
            # End the current connection span if exists
            if self._current_span:
                self._current_span["end_time"] = int(time.time() * 1000)
                self._connection_spans.append(self._current_span)
                
                # Create a history bundle for this span
                span_id = self._current_span["span_id"]
                self._history_bundles[span_id] = {
                    "span": self._current_span.copy(),
                    "events": self._current_span["events"].copy(),
                    "summary": f"Connected to {self.remote_space_info.get('name', self.remote_space_id)} " 
                              f"for {(self._current_span['end_time'] - self._current_span['start_time']) / 1000:.1f} seconds"
                }
                
                self._current_span = None
            
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
    
    def get_history_bundles(self, timeline_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get history bundles for all connection spans.
        
        According to the architecture, during context building, join events serve as pointers 
        that enable on-demand reconstruction of history bundles.
        
        Args:
            timeline_id: Optional timeline ID to filter by
            
        Returns:
            List of history bundles
        """
        bundles = []
        
        # Convert history bundles to list
        for span_id, bundle in self._history_bundles.items():
            bundles.append(bundle)
            
        # Add current span if active
        if self._current_span:
            # Create a temporary bundle for the current span
            current_bundle = {
                "span": self._current_span.copy(),
                "events": self._current_span["events"].copy(),
                "summary": f"Currently connected to {self.remote_space_info.get('name', self.remote_space_id)} " 
                          f"for {(int(time.time() * 1000) - self._current_span['start_time']) / 1000:.1f} seconds"
            }
            bundles.append(current_bundle)
            
        # Sort by start_time
        bundles.sort(key=lambda b: b["span"]["start_time"])
        
        return bundles
        
    def record_remote_event(self, event_data: Dict[str, Any]) -> bool:
        """
        Record an event from the remote space.
        
        Instead of directly adding to the agent's timeline, this event is stored
        in the current connection span, maintaining the separation between
        the agent's subjective timeline and remote space history.
        
        Args:
            event_data: Event data from the remote space
            
        Returns:
            True if the event was recorded, False otherwise
        """
        if not self._current_span:
            logger.warning(f"Cannot record remote event - no active connection span for uplink {self.id}")
            return False
            
        # Generate a unique event ID if not provided
        event_id = event_data.get("event_id", f"event_{uuid.uuid4().hex[:8]}")
        
        # Check if we've already processed this event
        if event_id in self._processed_remote_events:
            logger.debug(f"Skipping duplicate remote event {event_id} in uplink {self.id}")
            return False
            
        # Add to the current span's events
        self._current_span["events"].append({
            "event_id": event_id,
            "data": event_data,
            "timestamp": event_data.get("timestamp", int(time.time() * 1000))
        })
        
        # Mark as processed
        self._processed_remote_events.add(event_id)
        
        logger.debug(f"Recorded remote event {event_id} in uplink {self.id}")
        return True

    def get_span_events(self, span_id: str) -> List[Dict[str, Any]]:
        """
        Get events for a specific connection span.
        
        This is used during context building to retrieve history bundles
        for specific spans when needed.
        
        Args:
            span_id: ID of the span to get events for
            
        Returns:
            List of events in the span
        """
        # Check if this is the current span
        if self._current_span and self._current_span["span_id"] == span_id:
            return self._current_span["events"].copy()
            
        # Check history bundles
        if span_id in self._history_bundles:
            return self._history_bundles[span_id]["events"].copy()
            
        # Check connection spans
        for span in self._connection_spans:
            if span["span_id"] == span_id:
                return span["events"].copy()
                
        return []

    def update_delegate(self):
        """Update the delegate for this uplink to properly render history bundles."""
        from bot_framework.rendering.delegates import ElementDelegate, RenderingResult, RenderingMetadata
        from bot_framework.rendering import RenderingImportance, RenderingFormat, CompressionHint, RenderingOptions
        from datetime import datetime
        
        class UplinkDelegate(ElementDelegate):
            def __init__(self, element):
                super().__init__(element)
                self.uplink = element
            
            def render(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
                """Render the uplink state, including history bundles."""
                element_id = self.get_element_id()
                remote_id = state.get("remote_space_id", "unknown")
                remote_name = state.get("remote_space_info", {}).get("name", remote_id)
                connection_state = state.get("connection_state", {})
                
                # Check if the element is open
                is_open = True
                if self.uplink and hasattr(self.uplink, 'is_open'):
                    is_open = self.uplink.is_open()
                    
                if is_open:
                    # Full interior rendering with history bundles
                    history_bundles = state.get("history_bundles", [])
                    
                    if options.format == RenderingFormat.MARKDOWN:
                        content = f"## Uplink: {remote_name}\n\n"
                        content += f"**Status**: {'Connected' if connection_state.get('connected') else 'Disconnected'}\n"
                        
                        if history_bundles:
                            content += "\n### History Bundles\n\n"
                            for bundle in history_bundles:
                                span = bundle["span"]
                                start_time = self._format_timestamp(span["start_time"])
                                end_time = self._format_timestamp(span["end_time"]) if span["end_time"] else "Now"
                                event_count = len(bundle["events"])
                                
                                content += f"- **Span {span['span_id']}**: {start_time} to {end_time}\n"
                                content += f"  - {bundle['summary']}\n"
                                content += f"  - {event_count} events recorded\n"
                                
                                # Include a few recent events if available
                                if event_count > 0:
                                    content += "  - Recent events:\n"
                                    for event in bundle["events"][-3:]:  # Last 3 events
                                        event_type = event["data"].get("event_type", "unknown")
                                        content += f"    - {event_type} ({self._format_timestamp(event['timestamp'])})\n"
                        else:
                            content += "\n*No history bundles available*\n"
                    else:
                        # Text format
                        content = f"Uplink: {remote_name}\n"
                        content += f"Status: {'Connected' if connection_state.get('connected') else 'Disconnected'}\n"
                        
                        if history_bundles:
                            content += "\nHistory Bundles:\n"
                            for bundle in history_bundles:
                                span = bundle["span"]
                                event_count = len(bundle["events"])
                                content += f"- Span {span['span_id']}: {event_count} events\n"
                                content += f"  {bundle['summary']}\n"
                else:
                    # Compact exterior rendering
                    connected = connection_state.get("connected", False)
                    span_count = len(state.get("connection_spans", [])) + (1 if state.get("current_span") else 0)
                    event_count = 0
                    
                    # Count events across all spans
                    for span in state.get("connection_spans", []):
                        event_count += len(span.get("events", []))
                    
                    # Add current span events
                    if state.get("current_span"):
                        event_count += len(state.get("current_span", {}).get("events", []))
                    
                    if options.format == RenderingFormat.MARKDOWN:
                        content = f"**Uplink**: {remote_name} - "
                        content += f"{'Connected' if connected else 'Disconnected'}, "
                        content += f"{span_count} spans, {event_count} events"
                    else:
                        content = f"Uplink: {remote_name} - "
                        content += f"{'Connected' if connected else 'Disconnected'}, "
                        content += f"{span_count} spans, {event_count} events"
                
                # Create metadata
                importance = RenderingImportance.MEDIUM if connected else RenderingImportance.LOW
                metadata = self.create_metadata(
                    importance=importance,
                    format=options.format,
                    compression_hint=CompressionHint.SUMMARIZE
                )
                
                return RenderingResult(
                    content=content,
                    metadata=metadata
                )
                
            def _format_timestamp(self, timestamp):
                """Format a timestamp into a readable string."""
                if not timestamp:
                    return "unknown"
                    
                try:
                    dt = datetime.fromtimestamp(timestamp / 1000)
                    return dt.strftime("%H:%M:%S")
                except Exception:
                    return str(timestamp)
                
        # Set this delegate
        self.set_delegate(UplinkDelegate(self))
        
        # Register the delegate
        self.get_delegate().register()

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
        spans = self._connection_spans.copy()
        
        # Include active span if requested
        if include_active and self._current_span:
            spans.append(self._current_span.copy())
            
        # Sort by start time (newest first)
        spans.sort(key=lambda span: span.get('start_time', 0), reverse=True)
        
        # Limit the number of spans
        if limit > 0:
            spans = spans[:limit]
            
        return spans 