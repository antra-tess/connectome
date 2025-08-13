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
                "/veil/{space_id}/facets": "All VEIL facets in space with filtering",
                "/veil/{space_id}/facets/{facet_id}": "Detailed information about specific facet",
                "/health": "Simple health check"
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
    
    async def handle_timeline_details(self, space_id: str, timeline_id: str = None, limit: int = 100) -> Dict[str, Any]:
        """Handle timeline details endpoint for specific space/timeline."""
        self.request_count += 1
        
        try:
            if not space_id:
                return {
                    "error": "space_id is required",
                    "timestamp": time.time()
                }
            
            # Clamp limit between 1 and 1000
            limit = min(max(limit, 1), 1000)
            
            timeline_details = await self.data_collector.collect_timeline_details(
                space_id, timeline_id, limit
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
                               owner_id: str = None, limit: int = 100) -> Dict[str, Any]:
        """Handle all VEIL facets in space with filtering endpoint."""
        self.request_count += 1
        
        try:
            if not space_id:
                return {
                    "error": "space_id is required",
                    "timestamp": time.time()
                }
            
            # Clamp limit between 1 and 1000
            limit = min(max(limit, 1), 1000)
            
            veil_facets_data = await self.data_collector.collect_veil_facets_data(
                space_id, facet_type, owner_id, limit
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