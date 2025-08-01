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

from elements.space_registry import SpaceRegistry
from elements.elements.base import BaseElement
from elements.elements.space import Space
from elements.elements.inner_space import InnerSpace
from elements.elements.components.base_component import Component

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
                    "python_version": f"{psutil.PYTHON}",
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
    
    async def collect_timeline_details(self, space_id: str, timeline_id: str = None, limit: int = 100) -> Dict[str, Any]:
        """
        Collect detailed information about a specific timeline in a space.
        
        Args:
            space_id: The ID of the space containing the timeline
            timeline_id: The ID of the timeline (optional, defaults to primary)
            limit: Maximum number of events to return (default: 100)
        
        Returns:
            Dictionary containing detailed timeline information
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
            if hasattr(space, 'components') and space.components:
                for component_id, component in space.components.items():
                    if type(component).__name__ == "TimelineComponent":
                        timeline_component = component
                        break
            
            if not timeline_component:
                return {"error": f"No timeline component found in space '{space_id}'"}
            
            # Get timeline details
            target_timeline_id = timeline_id
            if not target_timeline_id:
                target_timeline_id = timeline_component.get_primary_timeline()
                if not target_timeline_id:
                    return {"error": "No timeline specified and no primary timeline found"}
            
            timeline_events = timeline_component.get_timeline_events(target_timeline_id, limit=limit)
            
            # Get timeline metadata
            timeline_info = timeline_component._state.get('_timelines', {}).get(target_timeline_id, {})
            
            result = {
                "timestamp": time.time(),
                "space_id": space_id,
                "timeline_id": target_timeline_id,
                "timeline_info": {
                    "is_primary": timeline_info.get('is_primary', False),
                    "head_event_ids": list(timeline_info.get('head_event_ids', [])),
                    "total_events": len(timeline_component._state.get('_all_events', {}))
                },
                "events": timeline_events[:limit] if timeline_events else [],
                "events_returned": len(timeline_events) if timeline_events else 0,
                "events_limited": len(timeline_events) > limit if timeline_events else False
            }
            
            return result
            
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
        
        # Collect elements
        if hasattr(space, 'elements') and space.elements:
            for element_id, element in space.elements.items():
                try:
                    element_info = await self._collect_element_details(element_id, element)
                    space_info["elements"][element_id] = element_info
                except Exception as e:
                    logger.error(f"Error collecting element {element_id}: {e}", exc_info=True)
                    space_info["elements"][element_id] = {
                        "error": f"Failed to collect element data: {str(e)}",
                        "type": type(element).__name__ if element else "Unknown"
                    }
        
        # Collect components directly on the space
        if hasattr(space, 'components') and space.components:
            for component_id, component in space.components.items():
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
        if hasattr(element, 'components') and element.components:
            for component_id, component in element.components.items():
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
                "elements_count": len(inner_space.elements) if hasattr(inner_space, 'elements') and inner_space.elements else 0,
                "components_count": len(inner_space.components) if hasattr(inner_space, 'components') and inner_space.components else 0,
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
        if hasattr(inner_space, 'components') and inner_space.components:
            for component_id, component in inner_space.components.items():
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
        if hasattr(space, 'components') and space.components:
            for component_id, component in space.components.items():
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
    
    def _get_event_types_summary(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get a summary of event types from a list of events."""
        event_types = {}
        for event in events:
            payload = event.get('payload', {})
            event_type = payload.get('event_type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        return event_types