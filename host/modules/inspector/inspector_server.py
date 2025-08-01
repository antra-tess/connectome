"""
Inspector Web Server

Simple HTTP server that provides JSON endpoints for inspecting Connectome host state.
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional
from aiohttp import web, ClientSession
from aiohttp.web import Application, Request, Response
import time

from .data_collector import InspectorDataCollector

logger = logging.getLogger(__name__)


class InspectorServer:
    """
    Web server for inspecting Connectome host state.
    
    Provides REST endpoints:
    - GET /status - Overall system status
    - GET /spaces - All spaces and their elements
    - GET /agents - Agent configurations and status
    - GET /adapters - Activity adapter connections
    - GET /metrics - System metrics and performance data
    """
    
    def __init__(self, host_instance, port: int = 8080):
        """
        Initialize inspector server.
        
        Args:
            host_instance: Reference to the main Host instance
            port: Port to run the web server on
        """
        self.host_instance = host_instance
        self.port = port
        self.app: Optional[Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.data_collector = InspectorDataCollector(host_instance)
        
        # Server metrics
        self.start_time = time.time()
        self.request_count = 0
        
    async def create_app(self) -> Application:
        """Create the aiohttp application with routes."""
        app = web.Application()
        
        # Add routes
        app.router.add_get('/', self.handle_root)
        app.router.add_get('/status', self.handle_status)
        app.router.add_get('/spaces', self.handle_spaces)
        app.router.add_get('/agents', self.handle_agents)
        app.router.add_get('/adapters', self.handle_adapters)
        app.router.add_get('/metrics', self.handle_metrics)
        app.router.add_get('/timelines', self.handle_timelines)
        app.router.add_get('/timelines/{space_id}', self.handle_timeline_details)
        app.router.add_get('/timelines/{space_id}/{timeline_id}', self.handle_timeline_details)
        app.router.add_get('/health', self.handle_health)
        
        # Add middleware for request counting
        app.middlewares.append(self.request_counter_middleware)
        
        return app
    
    async def start(self):
        """Start the inspector web server."""
        try:
            self.app = await self.create_app()
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            self.site = web.TCPSite(self.runner, 'localhost', self.port)
            await self.site.start()
            
            logger.info(f"Inspector server started on http://localhost:{self.port}")
            logger.info(f"Available endpoints:")
            logger.info(f"  GET http://localhost:{self.port}/ - API overview")
            logger.info(f"  GET http://localhost:{self.port}/status - System status")
            logger.info(f"  GET http://localhost:{self.port}/spaces - Space details")
            logger.info(f"  GET http://localhost:{self.port}/agents - Agent information")
            logger.info(f"  GET http://localhost:{self.port}/adapters - Adapter status")
            logger.info(f"  GET http://localhost:{self.port}/metrics - System metrics")
            logger.info(f"  GET http://localhost:{self.port}/timelines - Timeline DAG overview")
            logger.info(f"  GET http://localhost:{self.port}/timelines/{{space_id}} - Timeline details for space")
            logger.info(f"  GET http://localhost:{self.port}/timelines/{{space_id}}/{{timeline_id}} - Specific timeline")
            
        except Exception as e:
            logger.error(f"Failed to start inspector server: {e}", exc_info=True)
            raise
    
    async def stop(self):
        """Stop the inspector web server."""
        try:
            if self.site:
                await self.site.stop()
            if self.runner:
                await self.runner.cleanup()
            logger.info("Inspector server stopped")
        except Exception as e:
            logger.error(f"Error stopping inspector server: {e}", exc_info=True)
    
    @web.middleware
    async def request_counter_middleware(self, request: Request, handler):
        """Middleware to count requests."""
        self.request_count += 1
        start_time = time.time()
        
        try:
            response = await handler(request)
            response.headers['X-Request-ID'] = str(self.request_count)
            response.headers['X-Response-Time'] = f"{(time.time() - start_time):.3f}ms"
            return response
        except Exception as e:
            logger.error(f"Request failed: {e}", exc_info=True)
            raise
    
    def _json_response(self, data: Any, status: int = 200) -> Response:
        """Create a JSON response with proper headers."""
        return web.json_response(
            data,
            status=status,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
    
    async def handle_root(self, request: Request) -> Response:
        """Handle root endpoint - API overview."""
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
                "/health": "Simple health check"
            }
        }
        return self._json_response(api_info)
    
    async def handle_status(self, request: Request) -> Response:
        """Handle system status endpoint."""
        try:
            status_data = await self.data_collector.collect_system_status()
            return self._json_response(status_data)
        except Exception as e:
            logger.error(f"Error collecting system status: {e}", exc_info=True)
            return self._json_response(
                {"error": "Failed to collect system status", "details": str(e)},
                status=500
            )
    
    async def handle_spaces(self, request: Request) -> Response:
        """Handle spaces inspection endpoint."""
        try:
            spaces_data = await self.data_collector.collect_spaces_data()
            return self._json_response(spaces_data)
        except Exception as e:
            logger.error(f"Error collecting spaces data: {e}", exc_info=True)
            return self._json_response(
                {"error": "Failed to collect spaces data", "details": str(e)},
                status=500
            )
    
    async def handle_agents(self, request: Request) -> Response:
        """Handle agents inspection endpoint."""
        try:
            agents_data = await self.data_collector.collect_agents_data()
            return self._json_response(agents_data)
        except Exception as e:
            logger.error(f"Error collecting agents data: {e}", exc_info=True)
            return self._json_response(
                {"error": "Failed to collect agents data", "details": str(e)},
                status=500
            )
    
    async def handle_adapters(self, request: Request) -> Response:
        """Handle adapters inspection endpoint."""
        try:
            adapters_data = await self.data_collector.collect_adapters_data()
            return self._json_response(adapters_data)
        except Exception as e:
            logger.error(f"Error collecting adapters data: {e}", exc_info=True)
            return self._json_response(
                {"error": "Failed to collect adapters data", "details": str(e)},
                status=500
            )
    
    async def handle_metrics(self, request: Request) -> Response:
        """Handle system metrics endpoint."""
        try:
            metrics_data = await self.data_collector.collect_metrics_data()
            return self._json_response(metrics_data)
        except Exception as e:
            logger.error(f"Error collecting metrics data: {e}", exc_info=True)
            return self._json_response(
                {"error": "Failed to collect metrics data", "details": str(e)},
                status=500
            )
    
    async def handle_timelines(self, request: Request) -> Response:
        """Handle timeline overview endpoint."""
        try:
            timeline_data = await self.data_collector.collect_timeline_data()
            return self._json_response(timeline_data)
        except Exception as e:
            logger.error(f"Error collecting timeline data: {e}", exc_info=True)
            return self._json_response(
                {"error": "Failed to collect timeline data", "details": str(e)},
                status=500
            )
    
    async def handle_timeline_details(self, request: Request) -> Response:
        """Handle timeline details endpoint for specific space/timeline."""
        try:
            space_id = request.match_info.get('space_id')
            timeline_id = request.match_info.get('timeline_id')  # Optional
            
            # Get optional query parameters
            limit = int(request.query.get('limit', 100))
            limit = min(max(limit, 1), 1000)  # Clamp between 1 and 1000
            
            if not space_id:
                return self._json_response(
                    {"error": "space_id is required"},
                    status=400
                )
            
            timeline_details = await self.data_collector.collect_timeline_details(
                space_id, timeline_id, limit
            )
            return self._json_response(timeline_details)
        except Exception as e:
            logger.error(f"Error collecting timeline details: {e}", exc_info=True)
            return self._json_response(
                {"error": "Failed to collect timeline details", "details": str(e)},
                status=500
            )
    
    async def handle_health(self, request: Request) -> Response:
        """Handle health check endpoint."""
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.start_time,
            "request_count": self.request_count
        }
        return self._json_response(health_data)