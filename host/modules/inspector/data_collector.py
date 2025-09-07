"""
Inspector Data Collector

Collects detailed information about Connectome host state including spaces,
elements, components, agents, and adapters.
"""

import logging
import time
import asyncio
import json
from typing import Dict, Any, List, Optional, Union
import traceback
import psutil
import os
import sys

from elements.space_registry import SpaceRegistry
from elements.elements.base import BaseElement
from elements.elements.space import Space
from elements.elements.inner_space import InnerSpace
from elements.elements.components.base_component import Component
from .repl_context import REPLContextManager

logger = logging.getLogger(__name__)


class InspectorDataCollector:
    """
    Collects and formats data about the current state of the Connectome host.
    
    Provides structured inspection of:
    - All spaces (Inner, Shared, Uplink)
    - Elements within each space
    - Components attached to elements
    - Agent configurations and status
    - Activity adapter connections
    - System performance metrics
    """
    
    def __init__(self, host_instance):
        """
        Initialize data collector.
        
        Args:
            host_instance: Reference to the main Host instance
        """
        self.host_instance = host_instance
        self.space_registry = SpaceRegistry.get_instance()
        self.repl_manager = REPLContextManager(host_instance)
        
    async def collect_system_status(self) -> Dict[str, Any]:
        """
        Collect overall system status and health information.
        
        Returns:
            Dictionary containing system-wide status
        """
        try:
            status = {
                "timestamp": time.time(),
                "system": {
                    "process_id": os.getpid(),
                    "python_version": f"{sys.version}",
                    "uptime_seconds": time.time() - psutil.Process().create_time(),
                },
                "memory": {
                    "rss_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                    "vms_mb": psutil.Process().memory_info().vms / 1024 / 1024,
                    "percent": psutil.Process().memory_percent(),
                },
                "cpu": {
                    "percent": psutil.Process().cpu_percent(),
                    "num_threads": psutil.Process().num_threads(),
                },
                "spaces": await self._get_spaces_summary(),
                "registry_status": await self._get_registry_status()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error collecting system status: {e}", exc_info=True)
            return {
                "error": "Failed to collect system status",
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def collect_spaces_data(self) -> Dict[str, Any]:
        """
        Collect detailed information about all spaces and their elements.
        
        Returns:
            Dictionary containing comprehensive space data
        """
        try:
            spaces_data = {
                "timestamp": time.time(),
                "summary": await self._get_spaces_summary(),
                "details": {}
            }
            
            if self.space_registry:
                spaces_dict = self.space_registry.get_spaces()
                
                for space_id, space in spaces_dict.items():
                    try:
                        space_info = await self._collect_space_details(space_id, space)
                        spaces_data["details"][space_id] = space_info
                    except Exception as e:
                        logger.error(f"Error collecting data for space {space_id}: {e}", exc_info=True)
                        spaces_data["details"][space_id] = {
                            "error": f"Failed to collect space data: {str(e)}",
                            "space_type": type(space).__name__ if space else "Unknown"
                        }
            
            return spaces_data
            
        except Exception as e:
            logger.error(f"Error collecting spaces data: {e}", exc_info=True)
            return {
                "error": "Failed to collect spaces data",
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def collect_agents_data(self) -> Dict[str, Any]:
        """
        Collect information about all agents and their configurations.
        
        Returns:
            Dictionary containing agent data
        """
        try:
            agents_data = {
                "timestamp": time.time(),
                "agents": {}
            }
            
            if self.space_registry:
                spaces_dict = self.space_registry.get_spaces()
                
                for space_id, space in spaces_dict.items():
                    if isinstance(space, InnerSpace):
                        try:
                            agent_info = await self._collect_agent_details(space_id, space)
                            agents_data["agents"][space_id] = agent_info
                        except Exception as e:
                            logger.error(f"Error collecting agent data for {space_id}: {e}", exc_info=True)
                            agents_data["agents"][space_id] = {
                                "error": f"Failed to collect agent data: {str(e)}"
                            }
            
            return agents_data
            
        except Exception as e:
            logger.error(f"Error collecting agents data: {e}", exc_info=True)
            return {
                "error": "Failed to collect agents data",
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def collect_adapters_data(self) -> Dict[str, Any]:
        """
        Collect information about activity adapters and their connections.
        
        Returns:
            Dictionary containing adapter data
        """
        try:
            adapters_data = {
                "timestamp": time.time(),
                "adapters": {}
            }
            
            # Try to get activity client from host instance
            activity_client = getattr(self.host_instance, 'activity_client', None)
            if not activity_client:
                # Try to get from event loop if available
                event_loop = getattr(self.host_instance, 'event_loop', None)
                if event_loop:
                    activity_client = getattr(event_loop, 'activity_client', None)
            
            if activity_client:
                adapters_data["adapters"] = await self._collect_activity_client_details(activity_client)
            else:
                adapters_data["message"] = "No activity client found"
            
            return adapters_data
            
        except Exception as e:
            logger.error(f"Error collecting adapters data: {e}", exc_info=True)
            return {
                "error": "Failed to collect adapters data",
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def collect_metrics_data(self) -> Dict[str, Any]:
        """
        Collect system performance metrics and statistics.
        
        Returns:
            Dictionary containing metrics data
        """
        try:
            process = psutil.Process()
            
            metrics_data = {
                "timestamp": time.time(),
                "process": {
                    "pid": process.pid,
                    "name": process.name(),
                    "status": process.status(),
                    "create_time": process.create_time(),
                    "uptime_seconds": time.time() - process.create_time(),
                    "num_threads": process.num_threads(),
                    "num_fds": process.num_fds() if hasattr(process, 'num_fds') else None,
                },
                "memory": {
                    "rss_bytes": process.memory_info().rss,
                    "vms_bytes": process.memory_info().vms,
                    "rss_mb": process.memory_info().rss / 1024 / 1024,
                    "vms_mb": process.memory_info().vms / 1024 / 1024,
                    "percent": process.memory_percent(),
                },
                "cpu": {
                    "percent": process.cpu_percent(interval=0.1),
                    "times": process.cpu_times()._asdict() if hasattr(process.cpu_times(), '_asdict') else str(process.cpu_times()),
                },
                "io": process.io_counters()._asdict() if hasattr(process, 'io_counters') and process.io_counters() else None,
                "connections": len(process.connections()) if hasattr(process, 'connections') else None,
                "spaces_count": len(self.space_registry.get_spaces()) if self.space_registry else 0,
            }
            
            return metrics_data
            
        except Exception as e:
            logger.error(f"Error collecting metrics data: {e}", exc_info=True)
            return {
                "error": "Failed to collect metrics data",
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def collect_timeline_data(self) -> Dict[str, Any]:
        """
        Collect information about all Loom DAGs (timelines) in the system.
        
        Returns:
            Dictionary containing timeline data from all spaces
        """
        try:
            timeline_data = {
                "timestamp": time.time(),
                "summary": {
                    "total_spaces_with_timelines": 0,
                    "total_timelines": 0,
                    "total_events": 0,
                    "primary_timelines": 0
                },
                "spaces": {}
            }
            
            if self.space_registry:
                spaces_dict = self.space_registry.get_spaces()
                
                for space_id, space in spaces_dict.items():
                    try:
                        space_timeline_info = await self._collect_space_timeline_details(space_id, space)
                        if space_timeline_info and space_timeline_info.get("timelines"):
                            timeline_data["spaces"][space_id] = space_timeline_info
                            timeline_data["summary"]["total_spaces_with_timelines"] += 1
                            timeline_data["summary"]["total_timelines"] += len(space_timeline_info["timelines"])
                            timeline_data["summary"]["total_events"] += space_timeline_info["summary"]["total_events"]
                            if space_timeline_info["summary"]["primary_timeline"]:
                                timeline_data["summary"]["primary_timelines"] += 1
                    except Exception as e:
                        logger.error(f"Error collecting timeline data for space {space_id}: {e}", exc_info=True)
                        timeline_data["spaces"][space_id] = {
                            "error": f"Failed to collect timeline data: {str(e)}",
                            "space_type": type(space).__name__ if space else "Unknown"
                        }
            
            return timeline_data
            
        except Exception as e:
            logger.error(f"Error collecting timeline data: {e}", exc_info=True)
            return {
                "error": "Failed to collect timeline data",
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def collect_timeline_details(self, space_id: str, timeline_id: str = None, limit: int = 100, offset: float = None) -> Dict[str, Any]:
        """
        Collect timeline information for a space or specific timeline.
        
        Args:
            space_id: The ID of the space containing the timeline(s)
            timeline_id: The ID of the timeline (if None, returns timeline list for space)
            limit: Maximum number of events to return (for specific timeline)
            offset: Timestamp offset for pagination (for specific timeline)
        
        Returns:
            Dictionary containing timeline information or event data
        """
        try:
            if not self.space_registry:
                return {"error": "Space registry not available"}
            
            spaces_dict = self.space_registry.get_spaces()
            space = spaces_dict.get(space_id)
            
            if not space:
                return {"error": f"Space '{space_id}' not found"}
            
            # Find the timeline component in the space
            timeline_component = None
            space_components = space.get_components() if hasattr(space, 'get_components') else {}
            if space_components:
                for component_id, component in space_components.items():
                    if type(component).__name__ == "TimelineComponent":
                        timeline_component = component
                        break
            
            if not timeline_component:
                return {"error": f"No timeline component found in space '{space_id}'"}
            
            # If no timeline_id specified, return timeline list for the space
            if timeline_id is None:
                return await self._collect_space_timeline_list(space_id, timeline_component)
            
            # Otherwise, return events for the specific timeline
            return await self._collect_specific_timeline_events(space_id, timeline_id, timeline_component, limit, offset)
            
        except Exception as e:
            logger.error(f"Error collecting timeline details for {space_id}/{timeline_id}: {e}", exc_info=True)
            return {
                "error": "Failed to collect timeline details",
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def _get_spaces_summary(self) -> Dict[str, Any]:
        """Get summary statistics about spaces."""
        if not self.space_registry:
            return {"error": "Space registry not available"}
        
        spaces_dict = self.space_registry.get_spaces()
        summary = {
            "total_spaces": len(spaces_dict),
            "inner_spaces": 0,
            "shared_spaces": 0,
            "other_spaces": 0,
            "space_types": {}
        }
        
        for space_id, space in spaces_dict.items():
            space_type = type(space).__name__
            summary["space_types"][space_type] = summary["space_types"].get(space_type, 0) + 1
            
            if isinstance(space, InnerSpace):
                summary["inner_spaces"] += 1
            elif space_type == "SharedSpace":  # Check by type name since SharedSpace class doesn't exist yet
                summary["shared_spaces"] += 1
            else:
                summary["other_spaces"] += 1
        
        return summary
    
    async def _get_registry_status(self) -> Dict[str, Any]:
        """Get space registry status information."""
        if not self.space_registry:
            return {"status": "not_available"}
        
        return {
            "status": "available",
            "instance_id": id(self.space_registry),
            "has_storage": hasattr(self.space_registry, 'storage') and self.space_registry.storage is not None,
        }
    
    async def _collect_space_details(self, space_id: str, space: Space) -> Dict[str, Any]:
        """Collect detailed information about a specific space."""
        space_info = {
            "id": space_id,
            "name": getattr(space, 'name', space_id),
            "type": type(space).__name__,
            "class_path": f"{type(space).__module__}.{type(space).__name__}",
            "created_time": getattr(space, 'created_time', None),
            "elements": {},
            "components": [],
            "properties": {}
        }
        
        # Collect basic properties
        for attr_name in ['description', 'status', 'is_active', 'agent_id']:
            if hasattr(space, attr_name):
                space_info["properties"][attr_name] = getattr(space, attr_name)
        
        # Collect elements through ContainerComponent
        mounted_elements = {}
        if hasattr(space, '_container') and space._container:
            try:
                mounted_elements = space._container.get_mounted_elements()
            except Exception as e:
                logger.debug(f"Error accessing mounted elements for space {space_id}: {e}")
        
        if mounted_elements:
            for mount_id, element in mounted_elements.items():
                try:
                    element_info = await self._collect_element_details(element.id, element)
                    # Store by element ID, but include mount_id info
                    element_info["mount_id"] = mount_id
                    space_info["elements"][element.id] = element_info
                except Exception as e:
                    logger.error(f"Error collecting element {element.id} (mount_id: {mount_id}): {e}", exc_info=True)
                    space_info["elements"][element.id if hasattr(element, 'id') else mount_id] = {
                        "error": f"Failed to collect element data: {str(e)}",
                        "type": type(element).__name__ if element else "Unknown",
                        "mount_id": mount_id
                    }
        
        # Collect components directly on the space
        space_components = space.get_components() if hasattr(space, 'get_components') else {}
        if space_components:
            for component_id, component in space_components.items():
                try:
                    component_info = self._collect_component_details(component_id, component)
                    space_info["components"].append(component_info)
                except Exception as e:
                    logger.error(f"Error collecting component {component_id}: {e}", exc_info=True)
                    space_info["components"].append({
                        "id": component_id,
                        "error": f"Failed to collect component data: {str(e)}",
                        "type": type(component).__name__ if component else "Unknown"
                    })
        
        return space_info
    
    async def _collect_element_details(self, element_id: str, element: BaseElement) -> Dict[str, Any]:
        """Collect detailed information about a specific element."""
        element_info = {
            "id": element_id,
            "type": type(element).__name__,
            "class_path": f"{type(element).__module__}.{type(element).__name__}",
            "components": [],
            "properties": {},
            "parent_space_id": getattr(element, 'parent_space_id', None)
        }
        
        # Collect basic properties
        for attr_name in ['name', 'description', 'status', 'is_active', 'created_time']:
            if hasattr(element, attr_name):
                element_info["properties"][attr_name] = getattr(element, attr_name)
        
        # Collect components
        element_components = element.get_components() if hasattr(element, 'get_components') else {}
        if element_components:
            for component_id, component in element_components.items():
                try:
                    component_info = self._collect_component_details(component_id, component)
                    element_info["components"].append(component_info)
                except Exception as e:
                    logger.error(f"Error collecting component {component_id}: {e}", exc_info=True)
                    element_info["components"].append({
                        "id": component_id,
                        "error": f"Failed to collect component data: {str(e)}",
                        "type": type(component).__name__ if component else "Unknown"
                    })
        
        return element_info
    
    def _collect_component_details(self, component_id: str, component: Component) -> Dict[str, Any]:
        """Collect detailed information about a specific component."""
        component_info = {
            "id": component_id,
            "type": type(component).__name__,
            "class_path": f"{type(component).__module__}.{type(component).__name__}",
            "component_type": getattr(component, 'COMPONENT_TYPE', None),
            "properties": {},
            "methods": [],
            "tools": []
        }
        
        # Collect basic properties
        for attr_name in ['name', 'description', 'status', 'is_active', 'version']:
            if hasattr(component, attr_name):
                try:
                    value = getattr(component, attr_name)
                    # Only include serializable values
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        component_info["properties"][attr_name] = value
                except Exception:
                    pass
        
        # Collect component-specific attributes
        for attr_name in ['DEPENDENCIES', 'HANDLED_EVENT_TYPES', 'PROVIDES_TOOLS']:
            if hasattr(component, attr_name):
                try:
                    value = getattr(component, attr_name)
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        component_info["properties"][attr_name] = value
                except Exception:
                    pass
        
        # Collect tool methods (handle_* methods)
        try:
            for method_name in dir(component):
                if method_name.startswith('handle_') and callable(getattr(component, method_name)):
                    component_info["tools"].append(method_name)
        except Exception as e:
            logger.debug(f"Error collecting tools for component {component_id}: {e}")
        
        # Collect public methods
        try:
            for method_name in dir(component):
                if not method_name.startswith('_') and callable(getattr(component, method_name)):
                    component_info["methods"].append(method_name)
        except Exception as e:
            logger.debug(f"Error collecting methods for component {component_id}: {e}")
        
        return component_info
    
    async def _collect_agent_details(self, agent_id: str, inner_space: InnerSpace) -> Dict[str, Any]:
        """Collect detailed information about an agent."""
        agent_info = {
            "agent_id": agent_id,
            "name": getattr(inner_space, 'name', agent_id),
            "description": getattr(inner_space, 'description', None),
            "space_info": {
                "type": type(inner_space).__name__,
                "elements_count": len(inner_space._container.get_mounted_elements()) if hasattr(inner_space, '_container') and inner_space._container else 0,
                "components_count": len(inner_space.get_components()) if hasattr(inner_space, 'get_components') else 0,
            },
            "agent_loop": None,
            "llm_provider": None,
            "hud": None,
            "tools": [],
            "properties": {}
        }
        
        # Collect basic properties
        for attr_name in ['agent_id', 'platform_aliases', 'handles_direct_messages_from_adapter_ids']:
            if hasattr(inner_space, attr_name):
                agent_info["properties"][attr_name] = getattr(inner_space, attr_name)
        
        # Find agent loop component
        space_components = inner_space.get_components() if hasattr(inner_space, 'get_components') else {}
        if space_components:
            for component_id, component in space_components.items():
                component_type = type(component).__name__
                if 'AgentLoop' in component_type or 'Loop' in component_type:
                    agent_info["agent_loop"] = {
                        "id": component_id,
                        "type": component_type,
                        "component_type": getattr(component, 'COMPONENT_TYPE', None)
                    }
                elif 'HUD' in component_type:
                    agent_info["hud"] = {
                        "id": component_id,
                        "type": component_type,
                        "component_type": getattr(component, 'COMPONENT_TYPE', None)
                    }
                elif 'LLM' in component_type or 'Provider' in component_type:
                    agent_info["llm_provider"] = {
                        "id": component_id,
                        "type": component_type,
                        "component_type": getattr(component, 'COMPONENT_TYPE', None)
                    }
                
                # Collect tools from any component
                try:
                    for method_name in dir(component):
                        if method_name.startswith('handle_') and callable(getattr(component, method_name)):
                            agent_info["tools"].append({
                                "name": method_name,
                                "component": component_id,
                                "component_type": component_type
                            })
                except Exception as e:
                    logger.debug(f"Error collecting tools from component {component_id}: {e}")
        
        return agent_info
    
    async def _collect_activity_client_details(self, activity_client) -> Dict[str, Any]:
        """Collect details about the activity client and its adapters."""
        client_info = {
            "type": type(activity_client).__name__,
            "class_path": f"{type(activity_client).__module__}.{type(activity_client).__name__}",
            "adapters": {},
            "properties": {}
        }
        
        # Collect basic properties
        for attr_name in ['adapter_api_configs', 'adapter_clients', 'connection_status']:
            if hasattr(activity_client, attr_name):
                try:
                    value = getattr(activity_client, attr_name)
                    if attr_name == 'adapter_clients' and isinstance(value, dict):
                        # Collect adapter client details
                        for adapter_id, adapter_client in value.items():
                            try:
                                adapter_info = {
                                    "id": adapter_id,
                                    "type": type(adapter_client).__name__,
                                    "status": "connected" if hasattr(adapter_client, 'connected') and adapter_client.connected else "unknown",
                                    "properties": {}
                                }
                                
                                # Collect adapter properties
                                for prop_name in ['url', 'auth_token', 'socket_id', 'last_ping_time']:
                                    if hasattr(adapter_client, prop_name):
                                        prop_value = getattr(adapter_client, prop_name)
                                        if isinstance(prop_value, (str, int, float, bool, type(None))):
                                            if prop_name != 'auth_token':  # Don't expose auth tokens
                                                adapter_info["properties"][prop_name] = prop_value
                                            else:
                                                adapter_info["properties"][prop_name] = "***" if prop_value else None
                                
                                client_info["adapters"][adapter_id] = adapter_info
                            except Exception as e:
                                logger.debug(f"Error collecting adapter {adapter_id} details: {e}")
                                client_info["adapters"][adapter_id] = {
                                    "id": adapter_id,
                                    "error": str(e)
                                }
                    elif isinstance(value, (str, int, float, bool, list, type(None))):
                        client_info["properties"][attr_name] = value
                    elif isinstance(value, dict):
                        client_info["properties"][attr_name] = len(value)
                except Exception as e:
                    logger.debug(f"Error collecting activity client property {attr_name}: {e}")
        
        return client_info
    
    async def _collect_space_timeline_details(self, space_id: str, space: Space) -> Dict[str, Any]:
        """Collect timeline information for a specific space."""
        timeline_info = {
            "space_id": space_id,
            "space_type": type(space).__name__,
            "summary": {
                "has_timeline_component": False,
                "total_events": 0,
                "primary_timeline": None
            },
            "timelines": {},
            "timeline_component_info": None
        }
        
        # Find timeline component in the space
        timeline_component = None
        space_components = space.get_components() if hasattr(space, 'get_components') else {}
        if space_components:
            for component_id, component in space_components.items():
                if type(component).__name__ == "TimelineComponent":
                    timeline_component = component
                    timeline_info["summary"]["has_timeline_component"] = True
                    timeline_info["timeline_component_info"] = {
                        "component_id": component_id,
                        "component_type": type(component).__name__,
                        "storage_initialized": getattr(component, '_storage_initialized', False),
                        "space_id": getattr(component, '_space_id', None)
                    }
                    break
        
        if not timeline_component:
            return timeline_info
        
        try:
            # Get primary timeline ID
            primary_timeline_id = timeline_component.get_primary_timeline()
            timeline_info["summary"]["primary_timeline"] = primary_timeline_id
            
            # Get all timelines information
            timelines_state = timeline_component._state.get('_timelines', {})
            all_events = timeline_component._state.get('_all_events', {})
            timeline_info["summary"]["total_events"] = len(all_events)
            
            for timeline_id, timeline_data in timelines_state.items():
                head_event_ids = timeline_data.get('head_event_ids', set())
                if isinstance(head_event_ids, set):
                    head_event_ids = list(head_event_ids)
                
                # Get timeline events for summary info
                timeline_events = timeline_component.get_timeline_events(timeline_id, limit=10)
                
                timeline_info["timelines"][timeline_id] = {
                    "timeline_id": timeline_id,
                    "is_primary": timeline_data.get('is_primary', False),
                    "head_event_ids": head_event_ids,
                    "recent_events_count": len(timeline_events),
                    "latest_event": timeline_events[0] if timeline_events else None,
                    "event_types": self._get_event_types_summary(timeline_events[:5])
                }
                
        except Exception as e:
            logger.error(f"Error collecting timeline details for space {space_id}: {e}", exc_info=True)
            timeline_info["error"] = f"Failed to collect timeline details: {str(e)}"
        
        return timeline_info
    
    async def _collect_space_timeline_list(self, space_id: str, timeline_component) -> Dict[str, Any]:
        """
        Collect timeline list for a space without event data.

        Args:
            space_id: The ID of the space
            timeline_component: The timeline component instance
            
        Returns:
            Dictionary containing timeline list with statistics
        """
        try:
            # Get primary timeline ID
            primary_timeline_id = timeline_component.get_primary_timeline()
            
            # Get all timelines information
            timelines_state = timeline_component._state.get('_timelines', {})
            all_events = timeline_component._state.get('_all_events', {})
            
            result = {
                "timestamp": time.time(),
                "space_id": space_id,
                "summary": {
                    "total_timelines": len(timelines_state),
                    "total_events": len(all_events),
                    "primary_timeline": primary_timeline_id
                },
                "timelines": {}
            }
            
            for timeline_id, timeline_data in timelines_state.items():
                head_event_ids = timeline_data.get('head_event_ids', set())
                if isinstance(head_event_ids, set):
                    head_event_ids = list(head_event_ids)
                
                # Count events in this timeline
                timeline_events_count = 0
                for event_id, event in all_events.items():
                    if event.get('timeline_id') == timeline_id:
                        timeline_events_count += 1
                
                result["timelines"][timeline_id] = {
                    "timeline_id": timeline_id,
                    "is_primary": timeline_data.get('is_primary', False),
                    "head_event_ids": head_event_ids,
                    "total_events": timeline_events_count
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error collecting timeline list for space {space_id}: {e}", exc_info=True)
            return {
                "error": "Failed to collect timeline list",
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def _collect_specific_timeline_events(self, space_id: str, timeline_id: str, timeline_component, limit: int, offset: float = None) -> Dict[str, Any]:
        """
        Collect events for a specific timeline with timestamp-based pagination.

        Args:
            space_id: The ID of the space
            timeline_id: The ID of the timeline
            timeline_component: The timeline component instance
            limit: Maximum number of events to return (negative for reverse direction)
            offset: Timestamp offset for pagination
            
        Returns:
            Dictionary containing timeline events with pagination info
        """
        try:
            # Validate timeline exists
            timelines_state = timeline_component._state.get('_timelines', {})
            if timeline_id not in timelines_state:
                return {"error": f"Timeline '{timeline_id}' not found in space '{space_id}'"}
            
            # Get all events for this timeline
            all_events = timeline_component._state.get('_all_events', {})
            timeline_events = []
            
            for event_id, event in all_events.items():
                if event.get('timeline_id') == timeline_id:
                    timeline_events.append(event)
            
            # Sort by timestamp (newest first by default)
            timeline_events.sort(key=lambda e: e.get('timestamp', 0), reverse=True)
            
            # Apply timestamp filtering if offset is provided
            filtered_events = timeline_events
            if offset is not None:
                if limit >= 0:
                    # Forward pagination: events after the offset timestamp
                    filtered_events = [e for e in timeline_events if e.get('timestamp', 0) > offset]
                else:
                    # Reverse pagination: events before the offset timestamp (older events)
                    filtered_events = [e for e in timeline_events if e.get('timestamp', 0) < offset]
            
            # Apply limit
            abs_limit = abs(limit) if limit != 0 else 100
            limited_events = filtered_events[:abs_limit]
            
            # Get timeline metadata
            timeline_info = timelines_state.get(timeline_id, {})
            head_event_ids = timeline_info.get('head_event_ids', set())
            if isinstance(head_event_ids, set):
                head_event_ids = list(head_event_ids)
            
            # Determine pagination info
            has_more = len(filtered_events) > abs_limit
            next_offset = None
            prev_offset = None
            
            if limited_events:
                if limit >= 0:
                    # Forward pagination
                    next_offset = limited_events[-1].get('timestamp') if has_more else None
                    prev_offset = limited_events[0].get('timestamp') if offset is not None else None
                else:
                    # Reverse pagination
                    next_offset = limited_events[0].get('timestamp') if offset is not None else None
                    prev_offset = limited_events[-1].get('timestamp') if has_more else None
            
            result = {
                "timestamp": time.time(),
                "space_id": space_id,
                "timeline_id": timeline_id,
                "timeline_info": {
                    "is_primary": timeline_info.get('is_primary', False),
                    "head_event_ids": head_event_ids,
                    "total_events": len(timeline_events)
                },
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "events_returned": len(limited_events),
                    "has_more": has_more,
                    "next_offset": next_offset,
                    "prev_offset": prev_offset
                },
                "events": limited_events
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error collecting events for timeline {space_id}/{timeline_id}: {e}", exc_info=True)
            return {
                "error": "Failed to collect timeline events",
                "details": str(e),
                "timestamp": time.time()
            }

    def _get_event_types_summary(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get a summary of event types from a list of events."""
        event_types = {}
        for event in events:
            payload = event.get('payload', {})
            event_type = payload.get('event_type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        return event_types

    async def collect_veil_overview(self) -> Dict[str, Any]:
        """
        Collect VEIL system overview and statistics.
        
        Returns:
            Dictionary containing VEIL system statistics and complete VEIL trees
        """
        try:
            veil_overview = {
                "timestamp": time.time(),
                "summary": {
                    "total_spaces_with_veil": 0,
                    "total_facets": 0,
                    "facet_types": {
                        "event": 0,
                        "status": 0,
                        "ambient": 0
                    },
                    "veil_producers_found": 0
                },
                "spaces": {}
            }
            
            if self.space_registry:
                spaces_dict = self.space_registry.get_spaces()
                
                for space_id, space in spaces_dict.items():
                    try:
                        space_veil_info = await self._collect_space_veil_summary(space_id, space)
                        if space_veil_info and space_veil_info.get("has_veil_producer"):
                            veil_overview["spaces"][space_id] = space_veil_info
                            veil_overview["summary"]["total_spaces_with_veil"] += 1
                            veil_overview["summary"]["total_facets"] += space_veil_info["summary"]["total_facets"]
                            veil_overview["summary"]["veil_producers_found"] += space_veil_info["summary"]["producers_count"]
                            
                            # Add to facet type counts
                            for facet_type, count in space_veil_info["summary"]["facet_types"].items():
                                veil_overview["summary"]["facet_types"][facet_type] += count
                                
                    except Exception as e:
                        logger.error(f"Error collecting VEIL overview for space {space_id}: {e}", exc_info=True)
                        veil_overview["spaces"][space_id] = {
                            "error": f"Failed to collect VEIL data: {str(e)}",
                            "space_type": type(space).__name__ if space else "Unknown"
                        }
            
            return veil_overview
            
        except Exception as e:
            logger.error(f"Error collecting VEIL overview: {e}", exc_info=True)
            return {
                "error": "Failed to collect VEIL overview",
                "details": str(e),
                "timestamp": time.time()
            }

    async def collect_veil_space_data(self, space_id: str) -> Dict[str, Any]:
        """
        Collect VEIL cache state for specific space.
        
        Args:
            space_id: The ID of the space to inspect
            
        Returns:
            Dictionary containing VEIL cache data for the space
        """
        try:
            if not self.space_registry:
                return {"error": "Space registry not available"}
            
            spaces_dict = self.space_registry.get_spaces()
            space = spaces_dict.get(space_id)
            
            if not space:
                return {"error": f"Space '{space_id}' not found"}
            
            veil_space_data = {
                "timestamp": time.time(),
                "space_id": space_id,
                "space_type": type(space).__name__,
                "veil_producers": [],
                "combined_cache_stats": {
                    "total_facets": 0,
                    "facet_types": {"event": 0, "status": 0, "ambient": 0},
                    "cache_operations": 0,
                    "recent_facets": []
                },
                "facet_cache": None
            }
            
            # Find VEIL producers in the space
            veil_producers = self._find_veil_producers_in_space(space)
            veil_space_data["veil_producers"] = [
                {
                    "component_id": comp_id,
                    "component_type": type(comp).__name__,
                    "cache_size": comp.get_facet_cache_size() if hasattr(comp, 'get_facet_cache_size') else 0,
                    "multimodal_content": comp.has_multimodal_content() if hasattr(comp, 'has_multimodal_content') else False
                }
                for comp_id, comp in veil_producers
            ]
            
            # Capture VEIL trees from all producers
            veil_space_data["veil_trees"] = {}
            for comp_id, comp in veil_producers:
                if hasattr(comp, 'get_facet_cache'):
                    try:
                        facet_cache = comp.get_facet_cache()
                        veil_tree = self._capture_veil_tree_from_producer(comp_id, comp, facet_cache)
                        if veil_tree:
                            veil_space_data["veil_trees"][comp_id] = veil_tree
                    except Exception as e:
                        logger.debug(f"Error capturing VEIL tree from producer {comp_id}: {e}")
                        veil_space_data["veil_trees"][comp_id] = {
                            "producer_id": comp_id,
                            "error": str(e)
                        }
            
            # Get combined facet cache from primary VEIL producer
            if veil_producers:
                primary_producer = veil_producers[0][1]  # Use first producer found
                if hasattr(primary_producer, 'get_facet_cache'):
                    facet_cache = primary_producer.get_facet_cache()
                    veil_space_data["facet_cache"] = self._serialize_facet_cache(facet_cache)
                    veil_space_data["combined_cache_stats"] = self._get_facet_cache_stats(facet_cache)
            
            return veil_space_data
            
        except Exception as e:
            logger.error(f"Error collecting VEIL space data for {space_id}: {e}", exc_info=True)
            return {
                "error": "Failed to collect VEIL space data",
                "details": str(e),
                "timestamp": time.time()
            }

    async def collect_veil_facets_data(self, space_id: str, facet_type: str = None, 
                                     owner_id: str = None, limit: int = 100, after_facet_id: str = None) -> Dict[str, Any]:
        """
        Collect all VEIL facets in space with filtering.
        
        Args:
            space_id: The ID of the space to inspect
            facet_type: Optional filter by facet type (event, status, ambient)
            owner_id: Optional filter by owner element ID
            limit: Maximum number of facets to return (negative for reverse direction)
            after_facet_id: Optional cursor for pagination - return facets after/before this ID based on limit sign
            
        Returns:
            Dictionary containing filtered facet data
        """
        try:
            if not self.space_registry:
                return {"error": "Space registry not available"}
            
            spaces_dict = self.space_registry.get_spaces()
            space = spaces_dict.get(space_id)
            
            if not space:
                return {"error": f"Space '{space_id}' not found"}
            
            facets_data = {
                "timestamp": time.time(),
                "space_id": space_id,
                "filters": {
                    "facet_type": facet_type,
                    "owner_id": owner_id,
                    "limit": limit,
                    "after_facet_id": after_facet_id
                },
                "facets": [],
                "summary": {
                    "total_matching": 0,
                    "returned": 0,
                    "limited": False,
                    "next_cursor": None
                }
            }
            
            # Find VEIL producer and get facet cache
            veil_producers = self._find_veil_producers_in_space(space)
            if not veil_producers:
                return {**facets_data, "error": "No VEIL producers found in space"}
            
            primary_producer = veil_producers[0][1]
            if not hasattr(primary_producer, 'get_facet_cache'):
                return {**facets_data, "error": "VEIL producer does not support facet cache access"}
            
            facet_cache = primary_producer.get_facet_cache()
            all_facets = list(facet_cache.facets.values())
            
            # Sort by timestamp (newest first) first, before filtering
            all_facets.sort(key=lambda f: f.veil_timestamp, reverse=True)
            
            # Apply filters
            filtered_facets = []
            
            # Debug pagination
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Veil facets pagination: after_facet_id={after_facet_id}, limit={limit}, total_facets={len(all_facets)}")
            
            # Handle cursor-based filtering based on limit direction
            if after_facet_id:
                # Find the cursor facet position
                cursor_index = None
                for i, facet in enumerate(all_facets):
                    if facet.facet_id == after_facet_id:
                        cursor_index = i
                        break
                
                if cursor_index is not None:
                    if limit >= 0:
                        # Positive limit: get facets AFTER the cursor in pagination order (older facets)
                        # Since facets are sorted newest first, "after" in pagination means higher indices
                        all_facets = all_facets[cursor_index + 1:] if cursor_index < len(all_facets) - 1 else []
                    else:
                        # Negative limit: get facets BEFORE the cursor in pagination order (newer facets)
                        # Since facets are sorted newest first, "before" in pagination means lower indices
                        all_facets = all_facets[:cursor_index] if cursor_index > 0 else []
                    logger.debug(f"After cursor filtering: {len(all_facets)} facets remaining")
                else:
                    # Cursor not found, return empty result
                    logger.debug(f"Cursor facet {after_facet_id} not found")
                    all_facets = []
            
            # Apply type and owner filters
            for facet in all_facets:
                # Filter by facet type
                if facet_type and facet.facet_type.value != facet_type:
                    continue
                    
                # Filter by owner
                if owner_id and facet.owner_element_id != owner_id:
                    continue
                    
                filtered_facets.append(facet)
            
            # Debug: log first few facets added
            if len(filtered_facets) <= 3:
                logger.debug(f"Added facet {len(filtered_facets)}: {facet.facet_id}")
            
            # Apply limit and set pagination info
            abs_limit = abs(limit) if limit != 0 else 100
            facets_data["summary"]["total_matching"] = len(filtered_facets)
            limited_facets = filtered_facets[:abs_limit]
            facets_data["summary"]["returned"] = len(limited_facets)
            facets_data["summary"]["limited"] = len(filtered_facets) > abs_limit
            
            # Set next cursor if there are more results
            if len(filtered_facets) > abs_limit:
                facets_data["summary"]["next_cursor"] = limited_facets[-1].facet_id
                logger.debug(f"Set next_cursor to: {facets_data['summary']['next_cursor']}")
            else:
                logger.debug("No more results, next_cursor not set")
            
            # Serialize facets
            facets_data["facets"] = [
                self._serialize_facet(facet) for facet in limited_facets
            ]
            
            return facets_data
            
        except Exception as e:
            logger.error(f"Error collecting VEIL facets data for {space_id}: {e}", exc_info=True)
            return {
                "error": "Failed to collect VEIL facets data",
                "details": str(e),
                "timestamp": time.time()
            }

    async def collect_veil_facet_details(self, space_id: str, facet_id: str) -> Dict[str, Any]:
        """
        Collect detailed information about specific facet.
        
        Args:
            space_id: The ID of the space containing the facet
            facet_id: The ID of the facet to inspect
            
        Returns:
            Dictionary containing detailed facet information
        """
        try:
            if not self.space_registry:
                return {"error": "Space registry not available"}
            
            spaces_dict = self.space_registry.get_spaces()
            space = spaces_dict.get(space_id)
            
            if not space:
                return {"error": f"Space '{space_id}' not found"}
            
            # Find VEIL producer and get facet cache
            veil_producers = self._find_veil_producers_in_space(space)
            if not veil_producers:
                return {"error": "No VEIL producers found in space"}
            
            primary_producer = veil_producers[0][1]
            if not hasattr(primary_producer, 'get_facet_cache'):
                return {"error": "VEIL producer does not support facet cache access"}
            
            facet_cache = primary_producer.get_facet_cache()
            facet = facet_cache.facets.get(facet_id)
            
            if not facet:
                return {"error": f"Facet '{facet_id}' not found in space '{space_id}'"}
            
            facet_details = {
                "timestamp": time.time(),
                "space_id": space_id,
                "facet": self._serialize_facet(facet, include_detailed=True),
                "relationships": {
                    "links_to": facet.links_to,
                    "linked_from": self._find_facets_linking_to(facet_cache, facet_id)
                },
                "context": {
                    "owner_element_info": await self._get_element_info(space, facet.owner_element_id),
                    "temporal_neighbors": self._get_temporal_neighbors(facet_cache, facet)
                }
            }
            
            return facet_details
            
        except Exception as e:
            logger.error(f"Error collecting VEIL facet details for {space_id}/{facet_id}: {e}", exc_info=True)
            return {
                "error": "Failed to collect VEIL facet details",
                "details": str(e),
                "timestamp": time.time()
            }

    async def _collect_space_veil_summary(self, space_id: str, space: Space) -> Dict[str, Any]:
        """Collect VEIL summary information for a specific space."""
        veil_info = {
            "space_id": space_id,
            "space_type": type(space).__name__,
            "has_veil_producer": False,
            "summary": {
                "producers_count": 0,
                "total_facets": 0,
                "facet_types": {"event": 0, "status": 0, "ambient": 0}
            },
            "producers": []
        }
        
        # Find VEIL producers in the space
        veil_producers = self._find_veil_producers_in_space(space)
        veil_info["has_veil_producer"] = len(veil_producers) > 0
        veil_info["summary"]["producers_count"] = len(veil_producers)
        
        for comp_id, comp in veil_producers:
            producer_info = {
                "component_id": comp_id,
                "component_type": type(comp).__name__,
                "facet_cache_size": 0
            }
            
            # Get facet cache statistics
            if hasattr(comp, 'get_facet_cache'):
                try:
                    facet_cache = comp.get_facet_cache()
                    cache_stats = self._get_facet_cache_stats(facet_cache)
                    producer_info["facet_cache_size"] = cache_stats["total_facets"]
                    veil_info["summary"]["total_facets"] += cache_stats["total_facets"]
                    
                    # Add to facet type counts
                    for facet_type, count in cache_stats["facet_types"].items():
                        veil_info["summary"]["facet_types"][facet_type] += count
                        
                except Exception as e:
                    logger.debug(f"Error collecting facet cache stats from {comp_id}: {e}")
            
            veil_info["producers"].append(producer_info)
        
        return veil_info

    async def _collect_space_veil_summary_with_trees(self, space_id: str, space: Space) -> Dict[str, Any]:
        """Collect VEIL summary information for a specific space with complete VEIL trees."""
        veil_info = {
            "space_id": space_id,
            "space_type": type(space).__name__,
            "has_veil_producer": False,
            "summary": {
                "producers_count": 0,
                "total_facets": 0,
                "facet_types": {"event": 0, "status": 0, "ambient": 0}
            },
            "producers": [],
            "veil_trees": {}
        }
        
        # Find VEIL producers in the space
        veil_producers = self._find_veil_producers_in_space(space)
        veil_info["has_veil_producer"] = len(veil_producers) > 0
        veil_info["summary"]["producers_count"] = len(veil_producers)
        
        for comp_id, comp in veil_producers:
            producer_info = {
                "component_id": comp_id,
                "component_type": type(comp).__name__,
                "facet_cache_size": 0
            }
            
            # Get facet cache statistics and complete VEIL tree
            if hasattr(comp, 'get_facet_cache'):
                try:
                    facet_cache = comp.get_facet_cache()
                    cache_stats = self._get_facet_cache_stats(facet_cache)
                    producer_info["facet_cache_size"] = cache_stats["total_facets"]
                    veil_info["summary"]["total_facets"] += cache_stats["total_facets"]
                    
                    # Add to facet type counts
                    for facet_type, count in cache_stats["facet_types"].items():
                        veil_info["summary"]["facet_types"][facet_type] += count
                    
                    # Capture complete VEIL tree at render-time
                    veil_tree = self._capture_veil_tree_from_producer(comp_id, comp, facet_cache)
                    if veil_tree:
                        veil_info["veil_trees"][comp_id] = veil_tree
                        
                except Exception as e:
                    logger.debug(f"Error collecting facet cache stats from {comp_id}: {e}")
                    producer_info["error"] = str(e)
            
            veil_info["producers"].append(producer_info)
        
        return veil_info

    def _find_veil_producers_in_space(self, space: Space) -> List[tuple]:
        """Find all VEIL producer components in a space."""
        veil_producers = []
        
        space_components = space.get_components() if hasattr(space, 'get_components') else {}
        if space_components:
            for component_id, component in space_components.items():
                # Check if component is a VEIL producer
                if hasattr(component, 'get_facet_cache') or 'VeilProducer' in str(type(component)):
                    veil_producers.append((component_id, component))
        
        return veil_producers

    def _serialize_facet_cache(self, facet_cache) -> Dict[str, Any]:
        """Serialize a VEILFacetCache for JSON output."""
        if not facet_cache or not hasattr(facet_cache, 'facets'):
            return {"facets": {}, "statistics": {}}
        
        serialized_facets = {}
        for facet_id, facet in facet_cache.facets.items():
            serialized_facets[facet_id] = self._serialize_facet(facet)
        
        return {
            "facets": serialized_facets,
            "statistics": self._get_facet_cache_stats(facet_cache)
        }

    def _serialize_facet(self, facet, include_detailed: bool = False) -> Dict[str, Any]:
        """Serialize a VEILFacet for JSON output."""
        serialized = {
            "facet_id": facet.facet_id,
            "facet_type": facet.facet_type.value,
            "veil_timestamp": facet.veil_timestamp,
            "owner_element_id": facet.owner_element_id,
            "links_to": facet.links_to,
            "properties": dict(facet.properties),
            "content_summary": facet.get_content_summary() if hasattr(facet, 'get_content_summary') else "No summary available"
        }
        
        if include_detailed:
            # Add detailed information
            serialized["class_name"] = type(facet).__name__
            serialized["temporal_key"] = facet.get_temporal_key()
            
            # Try to get VEIL dict representation
            try:
                if hasattr(facet, 'to_veil_dict'):
                    serialized["veil_dict"] = facet.to_veil_dict()
            except Exception as e:
                serialized["veil_dict_error"] = str(e)
        
        return serialized

    def _get_facet_cache_stats(self, facet_cache) -> Dict[str, Any]:
        """Get statistics about a facet cache."""
        stats = {
            "total_facets": 0,
            "facet_types": {"event": 0, "status": 0, "ambient": 0},
            "recent_facets": []
        }
        
        if not facet_cache or not hasattr(facet_cache, 'facets'):
            return stats
        
        all_facets = list(facet_cache.facets.values())
        stats["total_facets"] = len(all_facets)
        
        # Count by type
        for facet in all_facets:
            facet_type = facet.facet_type.value
            if facet_type in stats["facet_types"]:
                stats["facet_types"][facet_type] += 1
        
        # Get recent facets (last 5)
        recent_facets = sorted(all_facets, key=lambda f: f.veil_timestamp, reverse=True)[:5]
        stats["recent_facets"] = [
            {
                "facet_id": f.facet_id,
                "facet_type": f.facet_type.value,
                "timestamp": f.veil_timestamp,
                "summary": f.get_content_summary() if hasattr(f, 'get_content_summary') else "No summary"
            }
            for f in recent_facets
        ]
        
        return stats

    def _find_facets_linking_to(self, facet_cache, target_facet_id: str) -> List[str]:
        """Find all facets that link to the target facet."""
        linking_facets = []
        
        if not facet_cache or not hasattr(facet_cache, 'facets'):
            return linking_facets
        
        for facet_id, facet in facet_cache.facets.items():
            if facet.links_to == target_facet_id:
                linking_facets.append(facet_id)
        
        return linking_facets

    async def _get_element_info(self, space: Space, element_id: str) -> Dict[str, Any]:
        """Get basic information about an element."""
        element_info = {
            "element_id": element_id,
            "found": False
        }
        
        # Check if element_id refers to the space itself
        if hasattr(space, 'id') and space.id == element_id:
            element_info.update({
                "found": True,
                "type": "space",
                "name": getattr(space, 'name', element_id),
                "class_name": type(space).__name__
            })
            return element_info
        
        # Check elements within the space through ContainerComponent
        if hasattr(space, '_container') and space._container:
            try:
                mounted_elements = space._container.get_mounted_elements()
                for mount_id, element in mounted_elements.items():
                    if element.id == element_id:
                        element_info.update({
                            "found": True,
                            "type": "element",
                            "name": getattr(element, 'name', element_id),
                            "class_name": type(element).__name__,
                            "mount_id": mount_id
                        })
                        break
            except Exception as e:
                logger.debug(f"Error checking elements for element_id {element_id}: {e}")
        
        return element_info

    def _get_temporal_neighbors(self, facet_cache, target_facet) -> Dict[str, Any]:
        """Get temporal neighbors (before/after) of a facet."""
        neighbors = {
            "before": None,
            "after": None
        }
        
        if not facet_cache or not hasattr(facet_cache, 'facets'):
            return neighbors
        
        all_facets = sorted(facet_cache.facets.values(), key=lambda f: f.veil_timestamp)
        target_index = None
        
        for i, facet in enumerate(all_facets):
            if facet.facet_id == target_facet.facet_id:
                target_index = i
                break
        
        if target_index is not None:
            if target_index > 0:
                before_facet = all_facets[target_index - 1]
                neighbors["before"] = {
                    "facet_id": before_facet.facet_id,
                    "facet_type": before_facet.facet_type.value,
                    "timestamp": before_facet.veil_timestamp
                }
            
            if target_index < len(all_facets) - 1:
                after_facet = all_facets[target_index + 1]
                neighbors["after"] = {
                    "facet_id": after_facet.facet_id,
                    "facet_type": after_facet.facet_type.value,
                    "timestamp": after_facet.veil_timestamp
                }
        
        return neighbors

    def _capture_veil_tree_from_producer(self, comp_id: str, producer_component, facet_cache) -> Dict[str, Any]:
        """
        Capture the complete VEIL tree from a producer at render-time.
        
        Args:
            comp_id: Component ID of the producer
            producer_component: The VeilProducer component instance
            facet_cache: The VEILFacetCache from the producer
            
        Returns:
            Dictionary containing the complete VEIL tree with facets converted to veil_dict format
        """
        try:
            veil_tree = {
                "producer_id": comp_id,
                "producer_type": type(producer_component).__name__,
                "capture_timestamp": time.time(),
                "tree_structure": {
                    "root_facets": [],
                    "linked_facets": {},
                    "temporal_sequence": []
                },
                "facets": {},
                "metadata": {
                    "total_facets": 0,
                    "capture_method": "render_time_inspection"
                }
            }
            
            if not facet_cache or not hasattr(facet_cache, 'facets'):
                return veil_tree
            
            all_facets = list(facet_cache.facets.values())
            veil_tree["metadata"]["total_facets"] = len(all_facets)
            
            # Create temporal sequence (sorted by veil_timestamp)
            temporal_facets = sorted(all_facets, key=lambda f: f.veil_timestamp)
            veil_tree["tree_structure"]["temporal_sequence"] = [f.facet_id for f in temporal_facets]
            
            # Process each facet and convert to veil_dict format
            root_facets = []
            linked_facets = {}
            
            for facet in all_facets:
                # Convert facet to veil_dict representation
                try:
                    if hasattr(facet, 'to_veil_dict'):
                        facet_veil_dict = facet.to_veil_dict()
                    else:
                        # Fallback: create basic veil_dict manually
                        facet_veil_dict = {
                            "facet_id": getattr(facet, 'facet_id', str(facet)),
                            "facet_type": getattr(facet, 'facet_type', {}).get('value', 'unknown') if hasattr(getattr(facet, 'facet_type', None), 'value') else str(getattr(facet, 'facet_type', 'unknown')),
                            "veil_timestamp": getattr(facet, 'veil_timestamp', 0),
                            "owner_element_id": getattr(facet, 'owner_element_id', 'unknown'),
                            "links_to": getattr(facet, 'links_to', None),
                            "properties": getattr(facet, 'properties', {}),
                            "content_summary": facet.get_content_summary() if hasattr(facet, 'get_content_summary') else "No summary available"
                        }
                    
                    veil_tree["facets"][facet.facet_id] = facet_veil_dict
                    
                    # Organize by linking structure
                    if facet.links_to:
                        if facet.links_to not in linked_facets:
                            linked_facets[facet.links_to] = []
                        linked_facets[facet.links_to].append(facet.facet_id)
                    else:
                        root_facets.append(facet.facet_id)
                        
                except Exception as e:
                    logger.debug(f"Error converting facet {facet.facet_id} to veil_dict: {e}")
                    # Add error information to tree
                    veil_tree["facets"][getattr(facet, 'facet_id', f'error_{id(facet)}')] = {
                        "error": f"Failed to convert facet to veil_dict: {str(e)}",
                        "facet_type": str(type(facet).__name__),
                        "facet_id": getattr(facet, 'facet_id', f'error_{id(facet)}')
                    }
            
            veil_tree["tree_structure"]["root_facets"] = root_facets
            veil_tree["tree_structure"]["linked_facets"] = linked_facets
            
            return veil_tree
            
        except Exception as e:
            logger.error(f"Error capturing VEIL tree from producer {comp_id}: {e}", exc_info=True)
            return {
                "producer_id": comp_id,
                "error": f"Failed to capture VEIL tree: {str(e)}",
                "capture_timestamp": time.time()
            }
    
    async def update_timeline_event(self, event_id: str, update_data: Dict[str, Any], 
                                  space_id: Optional[str] = None, timeline_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Update a specific timeline event by its globally unique ID.
        
        Args:
            event_id: The globally unique ID of the event to update
            update_data: Dictionary containing the fields to update
            space_id: Optional space ID for validation (if provided, will validate event is in this space)
            timeline_id: Optional timeline ID for validation (if provided, will validate event is in this timeline)
            
        Returns:
            Dictionary containing the result of the update operation
        """
        try:
            if not self.space_registry:
                return {"error": "Space registry not available", "success": False}
            
            # Find the event across all spaces since event IDs are globally unique
            target_space = None
            target_timeline_component = None
            event = None
            
            spaces_dict = self.space_registry.get_spaces()
            for sid, space in spaces_dict.items():
                # Skip if space_id is specified and doesn't match
                if space_id and sid != space_id:
                    continue
                    
                # Find timeline component in this space
                timeline_component = None
                space_components = space.get_components() if hasattr(space, 'get_components') else {}
                if space_components:
                    for component_id, component in space_components.items():
                        if type(component).__name__ == "TimelineComponent":
                            timeline_component = component
                            break
                
                if not timeline_component:
                    continue
                
                # Check if event exists in this space
                all_events = timeline_component._state.get('_all_events', {})
                if event_id in all_events:
                    event = all_events[event_id]
                    target_space = space
                    target_timeline_component = timeline_component
                    
                    # Validate timeline_id if specified
                    if timeline_id and event.get('timeline_id') != timeline_id:
                        return {
                            "error": f"Event '{event_id}' found but is in timeline '{event.get('timeline_id')}', not '{timeline_id}'",
                            "success": False
                        }
                    break
            
            if not event:
                if space_id:
                    return {"error": f"Event '{event_id}' not found in space '{space_id}'", "success": False}
                else:
                    return {"error": f"Event '{event_id}' not found in any space", "success": False}
            
            # Apply updates to the event
            updated_fields = []
            
            for field, new_value in update_data.items():
                if field in event:
                    old_value = event[field]
                    event[field] = new_value
                    updated_fields.append({
                        "field": field,
                        "old_value": old_value,
                        "new_value": new_value
                    })
                elif field == "payload" and isinstance(event.get("payload"), dict) and isinstance(new_value, dict):
                    # Handle nested payload updates
                    for payload_field, payload_value in new_value.items():
                        if payload_field in event["payload"]:
                            old_value = event["payload"][payload_field]
                            event["payload"][payload_field] = payload_value
                            updated_fields.append({
                                "field": f"payload.{payload_field}",
                                "old_value": old_value,
                                "new_value": payload_value
                            })
                        else:
                            # Add new field to payload
                            event["payload"][payload_field] = payload_value
                            updated_fields.append({
                                "field": f"payload.{payload_field}",
                                "old_value": None,
                                "new_value": payload_value
                            })
                else:
                    # Add new field to event
                    event[field] = new_value
                    updated_fields.append({
                        "field": field,
                        "old_value": None,
                        "new_value": new_value
                    })
            
            # Trigger async persistence if available
            if hasattr(target_timeline_component, '_persist_timeline_state'):
                try:
                    import asyncio
                    loop = asyncio.get_running_loop()
                    loop.create_task(target_timeline_component._persist_timeline_state())
                except RuntimeError:
                    pass  # No event loop running
            
            return {
                "success": True,
                "timestamp": time.time(),
                "space_id": target_space.id if hasattr(target_space, 'id') else "unknown",
                "timeline_id": event.get('timeline_id'),
                "event_id": event_id,
                "updated_fields": updated_fields,
                "updated_event": dict(event)
            }
            
        except Exception as e:
            logger.error(f"Error updating timeline event {event_id}: {e}", exc_info=True)
            return {
                "error": "Failed to update timeline event",
                "details": str(e),
                "success": False,
                "timestamp": time.time()
            }
    
    async def update_veil_facet(self, space_id: str, facet_id: str, 
                              update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a specific VEIL facet.
        
        Args:
            space_id: The ID of the space containing the facet
            facet_id: The ID of the facet to update
            update_data: Dictionary containing the fields to update
            
        Returns:
            Dictionary containing the result of the update operation
        """
        try:
            logger.info(f"Starting VEIL facet update - space_id: {space_id}, facet_id: {facet_id}, update_data: {update_data}")
            if not self.space_registry:
                return {"error": "Space registry not available", "success": False}
            
            spaces_dict = self.space_registry.get_spaces()
            space = spaces_dict.get(space_id)
            
            if not space:
                return {"error": f"Space '{space_id}' not found", "success": False}
            
            # Find VEIL producer and get facet cache
            veil_producers = self._find_veil_producers_in_space(space)
            if not veil_producers:
                return {"error": "No VEIL producers found in space", "success": False}
            
            primary_producer = veil_producers[0][1]
            if not hasattr(primary_producer, 'get_facet_cache'):
                return {"error": "VEIL producer does not support facet cache access", "success": False}
            
            facet_cache = primary_producer.get_facet_cache()
            facet = facet_cache.facets.get(facet_id)
            
            if not facet:
                return {"error": f"Facet '{facet_id}' not found in space '{space_id}'", "success": False}
            
            # Apply updates to the facet
            updated_fields = []
            
            for field, new_value in update_data.items():
                if hasattr(facet, field):
                    old_value = getattr(facet, field)
                    
                    # Handle special cases for certain fields
                    if field == "properties" and isinstance(new_value, dict):
                        # Update properties dictionary
                        for prop_key, prop_value in new_value.items():
                            old_prop_value = facet.properties.get(prop_key)
                            facet.properties[prop_key] = prop_value
                            updated_fields.append({
                                "field": f"properties.{prop_key}",
                                "old_value": old_prop_value,
                                "new_value": prop_value
                            })
                    elif field == "links_to":
                        # Validate that the target facet exists if linking
                        if new_value and new_value not in facet_cache.facets:
                            return {"error": f"Target facet '{new_value}' not found for linking", "success": False}
                        setattr(facet, field, new_value)
                        updated_fields.append({
                            "field": field,
                            "old_value": old_value,
                            "new_value": new_value
                        })
                    elif field in ["veil_timestamp", "owner_element_id"]:
                        # Allow updates to these fields with validation
                        setattr(facet, field, new_value)
                        updated_fields.append({
                            "field": field,
                            "old_value": old_value,
                            "new_value": new_value
                        })
                    else:
                        # For other fields, set directly if they exist
                        setattr(facet, field, new_value)
                        updated_fields.append({
                            "field": field,
                            "old_value": old_value,
                            "new_value": new_value
                        })
                else:
                    # For fields that don't exist, try to set them anyway (for extensibility)
                    try:
                        setattr(facet, field, new_value)
                        updated_fields.append({
                            "field": field,
                            "old_value": None,
                            "new_value": new_value
                        })
                    except Exception as e:
                        logger.warning(f"Could not set field '{field}' on facet: {e}")
            
            # Note: No need to mark as "modified" since this is direct inspector editing,
            # not external modification events from adapters
            
            return {
                "success": True,
                "timestamp": time.time(),
                "space_id": space_id,
                "facet_id": facet_id,
                "updated_fields": updated_fields,
                "updated_facet": self._serialize_facet(facet, include_detailed=True)
            }
            
        except Exception as e:
            logger.error(f"Error updating VEIL facet {space_id}/{facet_id}: {e}", exc_info=True)
            return {
                "error": "Failed to update VEIL facet",
                "details": str(e),
                "success": False,
                "timestamp": time.time()
            }

    async def collect_event_details(self, event_id: str) -> Dict[str, Any]:
        """
        Collect detailed information about a specific event by its ID.
        
        Args:
            event_id: The globally unique ID of the event to inspect
            
        Returns:
            Dictionary containing detailed event information
        """
        try:
            if not self.space_registry:
                return {
                    "error": "Space registry not available",
                    "timestamp": time.time()
                }

            # Search for the event across all spaces and timelines
            spaces_dict = self.space_registry.get_spaces()
            
            for space_id, space in spaces_dict.items():
                try:
                    # Check if space has timeline storage component
                    timeline_storage = None
                    for component in space.get_components().values():
                        if hasattr(component, 'get_all_timelines'):
                            timeline_storage = component
                            break
                    
                    if not timeline_storage:
                        continue
                        
                    # Search all timelines in this space for the event
                    timelines = timeline_storage.get_all_timelines()
                    for timeline_id, timeline in timelines.items():
                        try:
                            # Try to get the event from this timeline
                            events = timeline.get_all_events()
                            if event_id in events:
                                event = events[event_id]
                                
                                # Found the event! Return detailed information
                                return {
                                    "event": self._serialize_event(event),
                                    "context": {
                                        "space_id": space_id,
                                        "space_type": type(space).__name__,
                                        "timeline_id": timeline_id,
                                        "timeline_type": type(timeline).__name__
                                    },
                                    "relationships": {
                                        "parent_events": event.parent_ids if hasattr(event, 'parent_ids') else [],
                                        "children_count": len([e for e in events.values() 
                                                            if hasattr(e, 'parent_ids') and event_id in e.parent_ids])
                                    },
                                    "timestamp": time.time()
                                }
                                
                        except Exception as e:
                            logger.debug(f"Error searching timeline {timeline_id} in space {space_id}: {e}")
                            continue
                            
                except Exception as e:
                    logger.debug(f"Error searching space {space_id} for event {event_id}: {e}")
                    continue
            
            # Event not found in any timeline
            return {
                "error": "Event not found",
                "event_id": event_id,
                "searched_spaces": len(spaces_dict),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error collecting event details for {event_id}: {e}", exc_info=True)
            return {
                "error": "Failed to collect event details",
                "details": str(e),
                "timestamp": time.time()
            }

    async def render_space_as_text(self, space_id: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Render a space (Hub) in its textual representation as it would appear to an LLM or human.
        
        Args:
            space_id: The ID of the space to render
            options: Rendering options including:
                - format: "text" (default), "markdown", "json_messages"
                - include_tools: Include tool/ambient facets (default: True)
                - include_system: Include system messages (default: True)
                - max_turns: Limit number of turns to render
                - from_timestamp: Start rendering from specific timestamp
                
        Returns:
            Dictionary containing the rendered text and metadata
        """
        try:
            if not self.space_registry:
                return {
                    "error": "Space registry not available",
                    "timestamp": time.time()
                }
            
            # Get the space
            spaces_dict = self.space_registry.get_spaces()
            space = spaces_dict.get(space_id)
            
            if not space:
                return {
                    "error": f"Space '{space_id}' not found",
                    "timestamp": time.time()
                }
            
            # Get the SpaceVeilProducer component
            space_veil_producer = None
            for component in space.get_components().values():
                if component.__class__.__name__ == "SpaceVeilProducer":
                    space_veil_producer = component
                    break
            
            if not space_veil_producer:
                return {
                    "error": f"Space '{space_id}' does not have a SpaceVeilProducer component",
                    "timestamp": time.time()
                }
            
            # Get the VEILFacetCache
            facet_cache = space_veil_producer.get_facet_cache()
            if not facet_cache:
                return {
                    "error": f"Could not retrieve facet cache for space '{space_id}'",
                    "timestamp": time.time()
                }
            
            # Get the FacetAwareHUDComponent
            hud_component = None
            for component in space.get_components().values():
                if component.__class__.__name__ == "FacetAwareHUDComponent":
                    hud_component = component
                    break
            
            if not hud_component:
                return {
                    "error": f"Space '{space_id}' does not have a FacetAwareHUDComponent",
                    "timestamp": time.time()
                }
            
            # Parse options
            render_format = options.get("format", "text") if options else "text"
            include_tools = options.get("include_tools", True) if options else True
            include_system = options.get("include_system", True) if options else True
            max_turns = options.get("max_turns") if options else None
            from_timestamp = options.get("from_timestamp") if options else None
            
            # Prepare rendering options for HUD component
            hud_options = {
                "exclude_ambient": not include_tools,
                "include_system_messages": include_system,
                "render_mode": "normal"
            }
            
            # Process facets into turns using the HUD component
            turn_messages = await hud_component._process_facets_into_turns(
                facet_cache,
                hud_options,
                tools=None
            )
            
            # Filter by timestamp if specified
            if from_timestamp:
                turn_messages = [
                    msg for msg in turn_messages 
                    if msg.get("turn_metadata", {}).get("timestamp", 0) >= from_timestamp
                ]
            
            # Limit turns if specified
            if max_turns and len(turn_messages) > max_turns:
                turn_messages = turn_messages[-max_turns:]
            
            # Format the output based on requested format
            if render_format == "json_messages":
                # Return the raw message list
                rendered_content = turn_messages
                content_type = "application/json"
            elif render_format == "markdown":
                # Format as markdown
                lines = []
                space_name = getattr(space, 'name', space_id)
                agent_name = getattr(space, 'agent_name', 'Agent')
                
                lines.append(f"# {space_name} - {agent_name}\n")
                
                for msg in turn_messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    
                    if role == "user":
                        lines.append(f"## User\n\n{content}\n")
                    elif role == "assistant":
                        lines.append(f"## {agent_name}\n\n{content}\n")
                    elif role == "system":
                        if include_system:
                            lines.append(f"## System\n\n{content}\n")
                
                rendered_content = "\n".join(lines)
                content_type = "text/markdown"
            else:
                # Default plain text format
                lines = []
                space_name = getattr(space, 'name', space_id)
                agent_name = getattr(space, 'agent_name', 'Agent')
                
                lines.append(f"[{space_name} - {agent_name}]")
                lines.append("=" * 50)
                
                for msg in turn_messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    
                    if role == "user":
                        lines.append(f"\nUser: {content}")
                    elif role == "assistant":
                        lines.append(f"\n{agent_name}: {content}")
                    elif role == "system" and include_system:
                        lines.append(f"\n[System]: {content}")
                
                rendered_content = "\n".join(lines)
                content_type = "text/plain"
            
            # Return the rendered content with metadata
            return {
                "content": rendered_content,
                "metadata": {
                    "space_id": space_id,
                    "space_name": getattr(space, 'name', space_id),
                    "space_type": type(space).__name__,
                    "agent_name": getattr(space, 'agent_name', 'Unknown'),
                    "turn_count": len(turn_messages),
                    "facet_count": len(facet_cache.facets) if hasattr(facet_cache, 'facets') else 0,
                    "format": render_format,
                    "content_type": content_type,
                    "options": options or {}
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error rendering space {space_id} as text: {e}", exc_info=True)
            return {
                "error": "Failed to render space as text",
                "details": str(e),
                "timestamp": time.time()
            }
    
    def get_object_by_path(self, path: str) -> Optional[Any]:
        """
        Resolve an object by its path identifier.
        
        Args:
            path: Path in format "type:id" (e.g., "space:demo_space", "element:elem_id")
            
        Returns:
            The resolved object or None if not found
        """
        try:
            if ':' not in path:
                logger.warning(f"Invalid path format: {path}. Expected 'type:id'")
                return None
                
            path_type, path_id = path.split(':', 1)
            
            if path_type == 'space':
                # Resolve space by ID
                return self.space_registry.get_space(path_id)
            
            elif path_type == 'element':
                # Search for element across all spaces
                spaces_dict = self.space_registry.get_all_spaces()
                for space in spaces_dict.values():
                    if hasattr(space, 'elements') and path_id in space.elements:
                        return space.elements[path_id]
                return None
                
            elif path_type == 'component':
                # Search for component across all elements in all spaces
                spaces_dict = self.space_registry.get_all_spaces()
                for space in spaces_dict.values():
                    if hasattr(space, 'elements'):
                        for element in space.elements.values():
                            if hasattr(element, 'components'):
                                for component in element.components:
                                    if hasattr(component, 'component_id') and component.component_id == path_id:
                                        return component
                return None
                
            else:
                logger.warning(f"Unknown path type: {path_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error resolving path '{path}': {e}", exc_info=True)
            return None
    
    def collect_repl_sessions(self) -> Dict[str, Any]:
        """
        Collect information about active REPL sessions.
        
        Returns:
            Dictionary containing REPL sessions info
        """
        try:
            sessions_info = self.repl_manager.list_sessions()
            
            # Return simplified session info (without full namespaces)
            return {
                "sessions": sessions_info,
                "total_sessions": len(sessions_info),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error collecting REPL sessions: {e}", exc_info=True)
            return {
                "error": "Failed to collect REPL sessions",
                "details": str(e),
                "timestamp": time.time()
            }

    def _serialize_event(self, event) -> Dict[str, Any]:
        """Serialize a timeline event for JSON output."""
        try:
            return {
                "id": getattr(event, 'id', None),
                "timestamp": getattr(event, 'timestamp', None),
                "timeline_id": getattr(event, 'timeline_id', None),
                "parent_ids": getattr(event, 'parent_ids', []),
                "payload": getattr(event, 'payload', None),
                "event_type": getattr(event, 'payload', {}).get('event_type', None) if hasattr(event, 'payload') else None
            }
        except Exception as e:
            logger.warning(f"Error serializing event: {e}")
            return {"error": "Failed to serialize event", "details": str(e)}