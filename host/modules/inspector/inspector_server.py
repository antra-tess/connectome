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

from .endpoint_handlers import InspectorEndpointHandlers

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
        self.handlers = InspectorEndpointHandlers(host_instance)
        
        # Server metrics
        self.start_time = time.time()
        self.request_count = 0
        
    async def create_app(self) -> Application:
        """Create the aiohttp application with routes."""
        app = web.Application()
        
        # Add routes
        app.router.add_get('/', self.handle_root)
        app.router.add_get('/ui/', self.handle_ui)
        app.router.add_get('/favicon.png', self.handle_favicon)
        app.router.add_get('/status', self.handle_status)
        app.router.add_get('/spaces', self.handle_spaces)
        app.router.add_get('/spaces/{space_id}/render', self.handle_space_render)
        app.router.add_get('/agents', self.handle_agents)
        app.router.add_get('/adapters', self.handle_adapters)
        app.router.add_get('/metrics', self.handle_metrics)
        app.router.add_get('/timelines', self.handle_timelines)
        app.router.add_get('/timelines/{space_id}', self.handle_timeline_details)
        app.router.add_get('/timelines/{space_id}/{timeline_id}', self.handle_timeline_details)
        app.router.add_get('/veil', self.handle_veil)
        app.router.add_get('/veil/{space_id}', self.handle_veil_space)
        app.router.add_get('/veil/{space_id}/facets', self.handle_veil_facets)
        app.router.add_get('/veil/{space_id}/facets/{facet_id}', self.handle_veil_facet_details)
        app.router.add_get('/events/{event_id}', self.handle_event_details)
        app.router.add_get('/health', self.handle_health)
        
        # Write endpoints
        app.router.add_put('/events/{event_id}', self.handle_update_timeline_event)
        app.router.add_patch('/events/{event_id}', self.handle_update_timeline_event)
        app.router.add_put('/veil/{space_id}/facets/{facet_id}', self.handle_update_veil_facet)
        app.router.add_patch('/veil/{space_id}/facets/{facet_id}', self.handle_update_veil_facet)
        
        # REPL endpoints
        app.router.add_post('/repl/create', self.handle_repl_create)
        app.router.add_post('/repl/execute', self.handle_repl_execute)
        app.router.add_post('/repl/complete', self.handle_repl_complete)
        app.router.add_post('/repl/inspect', self.handle_repl_inspect)
        app.router.add_get('/repl/sessions', self.handle_repl_sessions)
        app.router.add_get('/repl/session/{session_id}/history', self.handle_repl_session_history)
        
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
            logger.info(f"  GET http://localhost:{self.port}/ui/ - Web UI with JSON visualization")
            logger.info(f"  GET http://localhost:{self.port}/status - System status")
            logger.info(f"  GET http://localhost:{self.port}/spaces - Space details")
            logger.info(f"  GET http://localhost:{self.port}/spaces/{{space_id}}/render - Render space as text")
            logger.info(f"  GET http://localhost:{self.port}/agents - Agent information")
            logger.info(f"  GET http://localhost:{self.port}/adapters - Adapter status")
            logger.info(f"  GET http://localhost:{self.port}/metrics - System metrics")
            logger.info(f"  GET http://localhost:{self.port}/timelines - Timeline DAG overview")
            logger.info(f"  GET http://localhost:{self.port}/timelines/{{space_id}} - Timeline details for space")
            logger.info(f"  GET http://localhost:{self.port}/timelines/{{space_id}}/{{timeline_id}} - Specific timeline")
            logger.info(f"  GET http://localhost:{self.port}/veil - VEIL system overview")
            logger.info(f"  GET http://localhost:{self.port}/veil/{{space_id}} - VEIL cache for specific space")
            logger.info(f"  GET http://localhost:{self.port}/veil/{{space_id}}/facets - All facets in space")
            logger.info(f"  GET http://localhost:{self.port}/veil/{{space_id}}/facets/{{facet_id}} - Specific facet details")
            logger.info(f"  PUT/PATCH http://localhost:{self.port}/events/{{event_id}} - Update timeline event")
            logger.info(f"  PUT/PATCH http://localhost:{self.port}/veil/{{space_id}}/facets/{{facet_id}} - Update VEIL facet")
            logger.info(f"  POST http://localhost:{self.port}/repl/create - Create new REPL session")
            logger.info(f"  POST http://localhost:{self.port}/repl/execute - Execute code in REPL session")
            logger.info(f"  POST http://localhost:{self.port}/repl/complete - Get code completions for REPL session")
            logger.info(f"  POST http://localhost:{self.port}/repl/inspect - Inspect object in REPL session")
            logger.info(f"  GET http://localhost:{self.port}/repl/sessions - List active REPL sessions")
            logger.info(f"  GET http://localhost:{self.port}/repl/session/{{session_id}}/history - Get REPL session history")
            
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
                'Access-Control-Allow-Methods': 'GET, POST, PUT, PATCH, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
    
    async def handle_root(self, request: Request) -> Response:
        """Handle root endpoint - API overview."""
        data = await self.handlers.handle_root()
        return self._json_response(data)
    
    async def handle_ui(self, request: Request) -> Response:
        """Handle UI visualization endpoint."""
        import os
        try:
            # Get the path to the HTML file in the same directory as this module
            current_dir = os.path.dirname(os.path.abspath(__file__))
            html_file_path = os.path.join(current_dir, 'ui.html')
            
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            return web.Response(
                text=html_content,
                content_type='text/html',
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                }
            )
        except Exception as e:
            logger.error(f"Error serving UI: {e}", exc_info=True)
            return web.Response(
                text=f"<html><body><h1>Error loading UI</h1><p>{str(e)}</p></body></html>",
                content_type='text/html',
                status=500
            )
    
    async def handle_favicon(self, request: Request) -> Response:
        """Handle favicon.png endpoint."""
        import os
        try:
            # Get the path to the favicon file in the same directory as this module
            current_dir = os.path.dirname(os.path.abspath(__file__))
            favicon_path = os.path.join(current_dir, 'favicon.png')
            
            with open(favicon_path, 'rb') as f:
                favicon_content = f.read()
            
            return web.Response(
                body=favicon_content,
                content_type='image/png',
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Cache-Control': 'public, max-age=86400'  # Cache for 1 day
                }
            )
        except Exception as e:
            logger.error(f"Error serving favicon: {e}", exc_info=True)
            return web.Response(
                text="Favicon not found",
                status=404
            )
    
    async def handle_status(self, request: Request) -> Response:
        """Handle system status endpoint."""
        data = await self.handlers.handle_status()
        status_code = 500 if "error" in data else 200
        return self._json_response(data, status=status_code)
    
    async def handle_spaces(self, request: Request) -> Response:
        """Handle spaces inspection endpoint."""
        data = await self.handlers.handle_spaces()
        status_code = 500 if "error" in data else 200
        return self._json_response(data, status=status_code)
    
    async def handle_agents(self, request: Request) -> Response:
        """Handle agents inspection endpoint."""
        data = await self.handlers.handle_agents()
        status_code = 500 if "error" in data else 200
        return self._json_response(data, status=status_code)
    
    async def handle_adapters(self, request: Request) -> Response:
        """Handle adapters inspection endpoint."""
        data = await self.handlers.handle_adapters()
        status_code = 500 if "error" in data else 200
        return self._json_response(data, status=status_code)
    
    async def handle_metrics(self, request: Request) -> Response:
        """Handle system metrics endpoint."""
        data = await self.handlers.handle_metrics()
        status_code = 500 if "error" in data else 200
        return self._json_response(data, status=status_code)
    
    async def handle_timelines(self, request: Request) -> Response:
        """Handle timeline overview endpoint."""
        data = await self.handlers.handle_timelines()
        status_code = 500 if "error" in data else 200
        return self._json_response(data, status=status_code)
    
    async def handle_timeline_details(self, request: Request) -> Response:
        """Handle timeline details endpoint for specific space/timeline."""
        space_id = request.match_info.get('space_id')
        timeline_id = request.match_info.get('timeline_id')  # Optional
        
        # Get optional query parameters
        try:
            limit = int(request.query.get('limit', 100))
        except (ValueError, TypeError):
            limit = 100
        
        # Get offset for pagination
        offset_str = request.query.get('offset')
        offset = None
        if offset_str:
            try:
                offset = float(offset_str)
            except (ValueError, TypeError):
                offset = None
        
        data = await self.handlers.handle_timeline_details(space_id, timeline_id, limit, offset)
        status_code = 400 if "error" in data and "required" in data.get("error", "") else (500 if "error" in data else 200)
        return self._json_response(data, status=status_code)
    
    async def handle_veil(self, request: Request) -> Response:
        """Handle VEIL system overview endpoint."""
        _ = request  # Request parameter required by aiohttp interface
        data = await self.handlers.handle_veil()
        status_code = 500 if "error" in data else 200
        return self._json_response(data, status=status_code)

    async def handle_veil_space(self, request: Request) -> Response:
        """Handle VEIL cache state for specific space endpoint."""
        space_id = request.match_info.get('space_id')
        data = await self.handlers.handle_veil_space(space_id)
        status_code = 400 if "error" in data and "required" in data.get("error", "") else (500 if "error" in data else 200)
        return self._json_response(data, status=status_code)

    async def handle_veil_facets(self, request: Request) -> Response:
        """Handle all VEIL facets in space with filtering endpoint."""
        space_id = request.match_info.get('space_id')
        
        # Get optional query parameters for filtering
        facet_type = request.query.get('type')  # event, status, ambient
        owner_id = request.query.get('owner')
        after_facet_id = request.query.get('after_facet_id')  # For pagination
        try:
            limit = int(request.query.get('limit', 100))
        except (ValueError, TypeError):
            limit = 100
        
        data = await self.handlers.handle_veil_facets(space_id, facet_type, owner_id, limit, after_facet_id)
        status_code = 400 if "error" in data and "required" in data.get("error", "") else (500 if "error" in data else 200)
        return self._json_response(data, status=status_code)

    async def handle_veil_facet_details(self, request: Request) -> Response:
        """Handle detailed information about specific facet endpoint."""
        space_id = request.match_info.get('space_id')
        facet_id = request.match_info.get('facet_id')
        
        data = await self.handlers.handle_veil_facet_details(space_id, facet_id)
        status_code = 400 if "error" in data and "required" in data.get("error", "") else (500 if "error" in data else 200)
        return self._json_response(data, status=status_code)

    async def handle_event_details(self, request: Request) -> Response:
        """Handle detailed information about specific event endpoint."""
        event_id = request.match_info.get('event_id')
        
        data = await self.handlers.handle_event_details(event_id)
        status_code = 400 if "error" in data and "required" in data.get("error", "") else (500 if "error" in data else 200)
        return self._json_response(data, status=status_code)

    async def handle_health(self, request: Request) -> Response:
        """Handle health check endpoint."""
        _ = request  # Request parameter required by aiohttp interface
        data = await self.handlers.handle_health()
        return self._json_response(data)

    async def handle_update_timeline_event(self, request: Request) -> Response:
        """Handle timeline event update endpoint."""
        event_id = request.match_info.get('event_id')
        
        # Parse JSON body
        try:
            body = await request.json()
        except Exception as e:
            return self._json_response({
                "error": "Invalid JSON in request body",
                "details": str(e),
                "success": False
            }, status=400)
        
        # Extract update data and optional parameters
        update_data = body.get('update_data', body)  # Support both wrapped and direct format
        space_id = body.get('space_id') or request.query.get('space_id')
        timeline_id = body.get('timeline_id') or request.query.get('timeline_id')
        
        data = await self.handlers.handle_update_timeline_event(
            event_id, update_data, space_id, timeline_id
        )
        status_code = 400 if not data.get("success", False) and "required" in data.get("error", "") else (500 if not data.get("success", False) else 200)
        return self._json_response(data, status=status_code)

    async def handle_update_veil_facet(self, request: Request) -> Response:
        """Handle VEIL facet update endpoint."""
        space_id = request.match_info.get('space_id')
        facet_id = request.match_info.get('facet_id')
        
        # Parse JSON body
        try:
            body = await request.json()
        except Exception as e:
            return self._json_response({
                "error": "Invalid JSON in request body",
                "details": str(e),
                "success": False
            }, status=400)
        
        # Extract update data
        update_data = body.get('update_data', body)  # Support both wrapped and direct format
        
        data = await self.handlers.handle_update_veil_facet(space_id, facet_id, update_data)
        status_code = 400 if not data.get("success", False) and "required" in data.get("error", "") else (500 if not data.get("success", False) else 200)
        return self._json_response(data, status=status_code)

    async def handle_repl_create(self, request: Request) -> Response:
        """Handle REPL session creation endpoint."""
        # Parse JSON body
        try:
            body = await request.json()
        except Exception as e:
            return self._json_response({
                "error": "Invalid JSON in request body",
                "details": str(e),
                "success": False
            }, status=400)
        
        # Extract required parameters
        context_type = body.get('context_type')
        context_id = body.get('context_id')
        target_path = body.get('target_path')
        
        data = await self.handlers.handle_repl_create(context_type, context_id, target_path)
        status_code = 400 if "error" in data and "required" in data.get("error", "") else (500 if "error" in data else 200)
        return self._json_response(data, status=status_code)

    async def handle_repl_execute(self, request: Request) -> Response:
        """Handle REPL code execution endpoint."""
        # Parse JSON body
        try:
            body = await request.json()
        except Exception as e:
            return self._json_response({
                "error": "Invalid JSON in request body",
                "details": str(e),
                "success": False
            }, status=400)
        
        # Extract required parameters
        session_id = body.get('session_id')
        code = body.get('code')
        timeout = body.get('timeout', 5.0)
        
        data = await self.handlers.handle_repl_execute(session_id, code, timeout)
        status_code = 400 if "error" in data and "required" in data.get("error", "") else (500 if not data.get("success", False) and "error" in data else 200)
        return self._json_response(data, status=status_code)

    async def handle_repl_sessions(self, request: Request) -> Response:
        """Handle REPL sessions list endpoint."""
        _ = request  # Request parameter required by aiohttp interface
        data = await self.handlers.handle_repl_sessions()
        status_code = 500 if "error" in data else 200
        return self._json_response(data, status=status_code)

    async def handle_repl_session_history(self, request: Request) -> Response:
        """Handle REPL session history endpoint."""
        session_id = request.match_info.get('session_id')
        
        # Get optional query parameters
        try:
            limit = int(request.query.get('limit', 50))
        except (ValueError, TypeError):
            limit = 50
        
        try:
            offset = int(request.query.get('offset', 0))
        except (ValueError, TypeError):
            offset = 0
        
        data = await self.handlers.handle_repl_session_history(session_id, limit, offset)
        status_code = 400 if "error" in data and "required" in data.get("error", "") else (500 if "error" in data else 200)
        return self._json_response(data, status=status_code)
    
    async def handle_repl_complete(self, request: Request) -> Response:
        """Handle REPL code completion endpoint."""
        try:
            request_data = await request.json()
            
            session_id = request_data.get('session_id', '')
            code = request_data.get('code', '')
            cursor_pos = request_data.get('cursor_pos', len(code))  # Default to end of code
            
        except (ValueError, TypeError) as e:
            return self._json_response({
                "error": "Invalid JSON in request body",
                "details": str(e),
                "timestamp": time.time()
            }, status=400)
        
        data = await self.handlers.handle_repl_complete(session_id, code, cursor_pos)
        status_code = 400 if "error" in data and "required" in data.get("error", "") else (500 if "error" in data and data.get("error") else 200)
        return self._json_response(data, status=status_code)
    
    async def handle_repl_inspect(self, request: Request) -> Response:
        """Handle REPL object inspection endpoint."""
        try:
            request_data = await request.json()
            
            session_id = request_data.get('session_id', '')
            obj_name = request_data.get('obj_name', '')
            
        except (ValueError, TypeError) as e:
            return self._json_response({
                "error": "Invalid JSON in request body",
                "details": str(e),
                "timestamp": time.time()
            }, status=400)
        
        data = await self.handlers.handle_repl_inspect(session_id, obj_name)
        status_code = 400 if "error" in data and "required" in data.get("error", "") else (500 if "error" in data else 200)
        return self._json_response(data, status=status_code)
    
    async def handle_space_render(self, request: Request) -> Response:
        """Handle space text rendering endpoint."""
        space_id = request.match_info.get('space_id')
        
        # Parse query parameters for rendering options
        options = {}
        if request.query.get('format'):
            options['format'] = request.query.get('format')
        if request.query.get('include_tools'):
            options['include_tools'] = request.query.get('include_tools').lower() == 'true'
        if request.query.get('include_system'):
            options['include_system'] = request.query.get('include_system').lower() == 'true'
        if request.query.get('max_turns'):
            try:
                options['max_turns'] = int(request.query.get('max_turns'))
            except ValueError:
                pass
        if request.query.get('from_timestamp'):
            try:
                options['from_timestamp'] = float(request.query.get('from_timestamp'))
            except ValueError:
                pass
        
        data = await self.handlers.handle_space_render(space_id, options if options else None)
        
        # Handle different response formats
        if not data.get("error"):
            content = data.get("content", "")
            metadata = data.get("metadata", {})
            content_type = metadata.get("content_type", "text/plain")
            
            # For JSON format, return as JSON response
            if metadata.get("format") in ["json_messages", "agent_context"]:
                return self._json_response(content)
            
            # For text/markdown formats, return as plain text response
            return web.Response(
                text=content if isinstance(content, str) else json.dumps(content),
                content_type=content_type,
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'X-Space-ID': space_id,
                    'X-Turn-Count': str(metadata.get("turn_count", 0)),
                    'X-Facet-Count': str(metadata.get("facet_count", 0))
                }
            )
        else:
            # Error response
            status_code = 400 if "not found" in data.get("error", "") else 500
            return self._json_response(data, status=status_code)