"""
Inspector Endpoint Handlers

Provides the core logic for inspector endpoints, decoupled from web server implementation.
This allows the same endpoint logic to be used by both web and CLI interfaces.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, Union

from .data_collector import InspectorDataCollector

logger = logging.getLogger(__name__)


class InspectorEndpointHandlers:
    """
    Core handler logic for inspector endpoints.
    
    Provides methods for each endpoint that return structured data, 
    independent of the delivery mechanism (HTTP, CLI, etc.).
    """
    
    def __init__(self, host_instance):
        """
        Initialize endpoint handlers.
        
        Args:
            host_instance: Reference to the main Host instance
        """
        self.host_instance = host_instance
        self.data_collector = InspectorDataCollector(host_instance)
        
        # Handler metrics
        self.start_time = time.time()
        self.request_count = 0
    
    async def handle_root(self) -> Dict[str, Any]:
        """Handle root endpoint - API overview."""
        self.request_count += 1
        
        api_info = {
            "service": "Connectome Inspector",
            "version": "1.0.0",
            "uptime_seconds": time.time() - self.start_time,
            "request_count": self.request_count,
            "endpoints": {
                "/": "API overview (this page)",
                "/status": "Overall system status and health",
                "/spaces": "Detailed information about all spaces and elements",
                "/agents": "Agent configurations and current status",
                "/adapters": "Activity adapter connection status",
                "/metrics": "System performance metrics",
                "/timelines": "Timeline DAG overview for all spaces",
                "/timelines/{space_id}": "Timeline details for specific space",
                "/timelines/{space_id}/{timeline_id}": "Specific timeline events and details",
                "/veil": "VEIL system overview and statistics",
                "/veil/{space_id}": "VEIL cache state for specific space",
                "/veil/{space_id}/facets": "All VEIL facets in space with filtering and pagination",
                "/veil/{space_id}/facets/{facet_id}": "Detailed information about specific facet",
                "/events/{event_id}": "Detailed information about specific event by globally unique ID",
                "/health": "Simple health check",
                "PUT/PATCH /events/{event_id}": "Update timeline event by globally unique event ID",
                "PUT/PATCH /veil/{space_id}/facets/{facet_id}": "Update VEIL facet",
                "/repl/sessions": "List active REPL sessions",
                "POST /repl/sessions": "Create new REPL session",
                "POST /repl/execute": "Execute code in REPL session",
                "POST /repl/complete": "Get code completions for REPL session",
                "POST /repl/inspect": "Inspect object in REPL session"
            }
        }
        return api_info
    
    async def handle_status(self) -> Dict[str, Any]:
        """Handle system status endpoint."""
        self.request_count += 1
        
        try:
            status_data = await self.data_collector.collect_system_status()
            return status_data
        except Exception as e:
            logger.error(f"Error collecting system status: {e}", exc_info=True)
            return {
                "error": "Failed to collect system status", 
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def handle_spaces(self) -> Dict[str, Any]:
        """Handle spaces inspection endpoint."""
        self.request_count += 1
        
        try:
            spaces_data = await self.data_collector.collect_spaces_data()
            return spaces_data
        except Exception as e:
            logger.error(f"Error collecting spaces data: {e}", exc_info=True)
            return {
                "error": "Failed to collect spaces data", 
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def handle_agents(self) -> Dict[str, Any]:
        """Handle agents inspection endpoint."""
        self.request_count += 1
        
        try:
            agents_data = await self.data_collector.collect_agents_data()
            return agents_data
        except Exception as e:
            logger.error(f"Error collecting agents data: {e}", exc_info=True)
            return {
                "error": "Failed to collect agents data", 
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def handle_adapters(self) -> Dict[str, Any]:
        """Handle adapters inspection endpoint."""
        self.request_count += 1
        
        try:
            adapters_data = await self.data_collector.collect_adapters_data()
            return adapters_data
        except Exception as e:
            logger.error(f"Error collecting adapters data: {e}", exc_info=True)
            return {
                "error": "Failed to collect adapters data", 
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def handle_metrics(self) -> Dict[str, Any]:
        """Handle system metrics endpoint."""
        self.request_count += 1
        
        try:
            metrics_data = await self.data_collector.collect_metrics_data()
            return metrics_data
        except Exception as e:
            logger.error(f"Error collecting metrics data: {e}", exc_info=True)
            return {
                "error": "Failed to collect metrics data", 
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def handle_timelines(self) -> Dict[str, Any]:
        """Handle timeline overview endpoint."""
        self.request_count += 1
        
        try:
            timeline_data = await self.data_collector.collect_timeline_data()
            return timeline_data
        except Exception as e:
            logger.error(f"Error collecting timeline data: {e}", exc_info=True)
            return {
                "error": "Failed to collect timeline data", 
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def handle_timeline_details(self, space_id: str, timeline_id: str = None, limit: int = 100, offset: float = None) -> Dict[str, Any]:
        """Handle timeline details endpoint for specific space/timeline."""
        self.request_count += 1
        
        try:
            if not space_id:
                return {
                    "error": "space_id is required",
                    "timestamp": time.time()
                }
            
            # Clamp limit between -1000 and 1000 (allow negative for reverse direction)
            if limit > 1000:
                limit = 1000
            elif limit < -1000:
                limit = -1000
            elif limit == 0:
                limit = 100
            
            timeline_details = await self.data_collector.collect_timeline_details(
                space_id, timeline_id, limit, offset
            )
            return timeline_details
        except Exception as e:
            logger.error(f"Error collecting timeline details: {e}", exc_info=True)
            return {
                "error": "Failed to collect timeline details", 
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def handle_veil(self) -> Dict[str, Any]:
        """Handle VEIL system overview endpoint."""
        self.request_count += 1
        
        try:
            veil_data = await self.data_collector.collect_veil_overview()
            return veil_data
        except Exception as e:
            logger.error(f"Error collecting VEIL overview: {e}", exc_info=True)
            return {
                "error": "Failed to collect VEIL overview", 
                "details": str(e),
                "timestamp": time.time()
            }

    async def handle_veil_space(self, space_id: str) -> Dict[str, Any]:
        """Handle VEIL cache state for specific space endpoint."""
        self.request_count += 1
        
        try:
            if not space_id:
                return {
                    "error": "space_id is required",
                    "timestamp": time.time()
                }
            
            veil_space_data = await self.data_collector.collect_veil_space_data(space_id)
            return veil_space_data
        except Exception as e:
            logger.error(f"Error collecting VEIL space data: {e}", exc_info=True)
            return {
                "error": "Failed to collect VEIL space data", 
                "details": str(e),
                "timestamp": time.time()
            }

    async def handle_veil_facets(self, space_id: str, facet_type: str = None, 
                               owner_id: str = None, limit: int = 100, after_facet_id: str = None) -> Dict[str, Any]:
        """Handle all VEIL facets in space with filtering endpoint."""
        self.request_count += 1
        
        try:
            if not space_id:
                return {
                    "error": "space_id is required",
                    "timestamp": time.time()
                }
            
            # Clamp limit between -1000 and 1000 (allow negative for reverse direction)
            if limit > 1000:
                limit = 1000
            elif limit < -1000:
                limit = -1000
            elif limit == 0:
                limit = 100
            
            veil_facets_data = await self.data_collector.collect_veil_facets_data(
                space_id, facet_type, owner_id, limit, after_facet_id
            )
            return veil_facets_data
        except Exception as e:
            logger.error(f"Error collecting VEIL facets data: {e}", exc_info=True)
            return {
                "error": "Failed to collect VEIL facets data", 
                "details": str(e),
                "timestamp": time.time()
            }

    async def handle_veil_facet_details(self, space_id: str, facet_id: str) -> Dict[str, Any]:
        """Handle detailed information about specific facet endpoint."""
        self.request_count += 1
        
        try:
            if not space_id or not facet_id:
                return {
                    "error": "Both space_id and facet_id are required",
                    "timestamp": time.time()
                }
            
            facet_details = await self.data_collector.collect_veil_facet_details(space_id, facet_id)
            return facet_details
        except Exception as e:
            logger.error(f"Error collecting VEIL facet details: {e}", exc_info=True)
            return {
                "error": "Failed to collect VEIL facet details", 
                "details": str(e),
                "timestamp": time.time()
            }

    async def handle_health(self) -> Dict[str, Any]:
        """Handle health check endpoint."""
        self.request_count += 1
        
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.start_time,
            "request_count": self.request_count
        }
        return health_data
    
    async def handle_update_timeline_event(self, event_id: str, update_data: Dict[str, Any],
                                         space_id: Optional[str] = None, timeline_id: Optional[str] = None) -> Dict[str, Any]:
        """Handle timeline event update endpoint."""
        self.request_count += 1
        
        try:
            if not event_id:
                return {
                    "error": "event_id is required",
                    "success": False,
                    "timestamp": time.time()
                }
            
            if not update_data:
                return {
                    "error": "update_data is required",
                    "success": False,
                    "timestamp": time.time()
                }
            
            result = await self.data_collector.update_timeline_event(
                event_id, update_data, space_id, timeline_id
            )
            return result
            
        except Exception as e:
            logger.error(f"Error handling timeline event update: {e}", exc_info=True)
            return {
                "error": "Failed to handle timeline event update", 
                "details": str(e),
                "success": False,
                "timestamp": time.time()
            }
    
    async def handle_update_veil_facet(self, space_id: str, facet_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle VEIL facet update endpoint."""
        self.request_count += 1
        
        try:
            if not space_id or not facet_id:
                return {
                    "error": "Both space_id and facet_id are required",
                    "success": False,
                    "timestamp": time.time()
                }
            
            if not update_data:
                return {
                    "error": "update_data is required",
                    "success": False,
                    "timestamp": time.time()
                }
            
            result = await self.data_collector.update_veil_facet(space_id, facet_id, update_data)
            return result
            
        except Exception as e:
            logger.error(f"Error handling VEIL facet update: {e}", exc_info=True)
            return {
                "error": "Failed to handle VEIL facet update", 
                "details": str(e),
                "success": False,
                "timestamp": time.time()
            }

    async def handle_event_details(self, event_id: str) -> Dict[str, Any]:
        """Handle detailed information about specific event endpoint."""
        self.request_count += 1
        
        try:
            if not event_id:
                return {
                    "error": "event_id is required",
                    "timestamp": time.time()
                }
            
            event_details = await self.data_collector.collect_event_details(event_id)
            return event_details
        except Exception as e:
            logger.error(f"Error collecting event details: {e}", exc_info=True)
            return {
                "error": "Failed to collect event details", 
                "details": str(e),
                "timestamp": time.time()
            }

    async def handle_repl_sessions(self) -> Dict[str, Any]:
        """Handle REPL sessions list endpoint."""
        self.request_count += 1
        
        try:
            sessions_data = self.data_collector.collect_repl_sessions()
            return sessions_data
        except Exception as e:
            logger.error(f"Error collecting REPL sessions: {e}", exc_info=True)
            return {
                "error": "Failed to collect REPL sessions",
                "details": str(e),
                "timestamp": time.time()
            }

    async def handle_repl_create(self, context_type: str, context_id: str, target_path: str = None) -> Dict[str, Any]:
        """Handle REPL session creation endpoint."""
        self.request_count += 1
        
        try:
            if not context_type or not context_id:
                return {
                    "error": "context_type and context_id are required",
                    "timestamp": time.time()
                }
            
            # Resolve target object if path provided
            target_object = None
            if target_path:
                target_object = self.data_collector.get_object_by_path(target_path)
                if target_object is None:
                    return {
                        "error": f"Failed to resolve target path: {target_path}",
                        "timestamp": time.time()
                    }
            
            # Create REPL context
            session = self.data_collector.repl_manager.create_context(
                context_type, context_id, target_object
            )
            
            # Return session info (without full namespace for brevity)
            return {
                "success": True,
                "session": {
                    "id": session["id"],
                    "type": session["type"],
                    "context_id": session["context_id"],
                    "created_at": session["created_at"],
                    "last_accessed": session["last_accessed"],
                    "history_length": len(session["history"]),
                    "namespace_keys": list(session["namespace"].keys())
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error creating REPL session: {e}", exc_info=True)
            return {
                "error": "Failed to create REPL session",
                "details": str(e),
                "timestamp": time.time()
            }

    async def handle_repl_session_history(self, session_id: str, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Handle REPL session history endpoint."""
        self.request_count += 1
        
        try:
            if not session_id:
                return {
                    "error": "session_id is required",
                    "timestamp": time.time()
                }
            
            # Get session history
            history_data = self.data_collector.repl_manager.get_session_history(
                session_id, limit, offset
            )
            
            if history_data is None:
                return {
                    "error": f"Session not found: {session_id}",
                    "timestamp": time.time()
                }
            
            return {
                "success": True,
                **history_data,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting REPL session history: {e}", exc_info=True)
            return {
                "error": "Failed to get REPL session history",
                "details": str(e),
                "timestamp": time.time()
            }

    async def handle_repl_execute(self, session_id: str, code: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Handle REPL code execution endpoint."""
        self.request_count += 1
        
        try:
            if not session_id or not code:
                return {
                    "error": "session_id and code are required",
                    "timestamp": time.time()
                }
            
            # Execute code in the specified session
            result = self.data_collector.repl_manager.execute_in_context(
                session_id, code, timeout
            )
            
            return {
                "success": result["success"],
                "output": result["output"],
                "error": result["error"],
                "execution_time_ms": result["execution_time_ms"],
                "execution_count": result.get("execution_count", 0),
                "rich_output": result.get("rich_output", []),
                "namespace_changes": result["namespace_changes"],
                "output_metadata": result.get("output_metadata", {}),
                "session_id": session_id,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error executing REPL code: {e}", exc_info=True)
            return {
                "error": "Failed to execute REPL code",
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def handle_repl_complete(self, session_id: str, code: str, cursor_pos: int) -> Dict[str, Any]:
        """Handle REPL code completion endpoint."""
        self.request_count += 1
        
        try:
            if not session_id or code is None:
                return {
                    "error": "session_id and code are required",
                    "timestamp": time.time()
                }
            
            # Get completions for the specified session
            result = self.data_collector.repl_manager.get_completions(
                session_id, code, cursor_pos
            )
            
            return {
                "success": "error" not in result,
                "completions": result.get("completions", []),
                "cursor_start": result.get("cursor_start", cursor_pos),
                "cursor_end": result.get("cursor_end", cursor_pos),
                "metadata": result.get("metadata", {}),
                "error": result.get("error", ""),
                "session_id": session_id,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting REPL completions: {e}", exc_info=True)
            return {
                "error": "Failed to get REPL completions",
                "details": str(e),
                "timestamp": time.time()
            }
    
    async def handle_repl_inspect(self, session_id: str, obj_name: str) -> Dict[str, Any]:
        """Handle REPL object inspection endpoint."""
        self.request_count += 1
        
        try:
            if not session_id or not obj_name:
                return {
                    "error": "session_id and obj_name are required",
                    "timestamp": time.time()
                }
            
            # Inspect object in the specified session
            result = self.data_collector.repl_manager.inspect_object(
                session_id, obj_name
            )
            
            return {
                "success": "error" not in result,
                "inspection": result,
                "session_id": session_id,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error inspecting REPL object: {e}", exc_info=True)
            return {
                "error": "Failed to inspect REPL object",
                "details": str(e),
                "timestamp": time.time()
            }