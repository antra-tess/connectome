"""
CLI Command Handler Plugin

In-process CLI command handler that executes within the host process.
Uses the same InspectorEndpointHandlers as the web server for consistency.
"""

import asyncio
import base64
import json
import logging
import time
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from queue import Queue, Empty

from .endpoint_handlers import InspectorEndpointHandlers

logger = logging.getLogger(__name__)


@dataclass
class CommandRequest:
    """Represents a CLI command request."""
    command: str
    args: Dict[str, Any]
    request_id: str
    response_callback: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass
class CommandResponse:
    """Represents a CLI command response."""
    result: Dict[str, Any]
    error: Optional[str]
    request_id: str
    execution_time_ms: float


class CLICommandHandler:
    """
    In-process CLI command handler that executes inspector commands
    within the host process using the same endpoint handlers as the web server.
    """
    
    def __init__(self, host_instance):
        """
        Initialize the CLI command handler.
        
        Args:
            host_instance: Reference to the main Host instance
        """
        self.host_instance = host_instance
        self.handlers = InspectorEndpointHandlers(host_instance)
        self.is_running = False
        self.command_queue = Queue()
        self.worker_thread = None
        self.loop = None
        
        # Statistics
        self.start_time = time.time()
        self.commands_processed = 0
        self.last_command_time = None
        
        # Command mapping
        self.command_map = {
            'overview': self._handle_overview,
            'root': self._handle_overview,  # Alias
            'status': self._handle_status,
            'spaces': self._handle_spaces,
            'agents': self._handle_agents,
            'adapters': self._handle_adapters,
            'metrics': self._handle_metrics,
            'timelines': self._handle_timelines,
            'timeline-details': self._handle_timeline_details,
            'veil': self._handle_veil,
            'veil-space': self._handle_veil_space,
            'veil-facets': self._handle_veil_facets,
            'veil-facet-details': self._handle_veil_facet_details,
            'health': self._handle_health,
            'space-render': self._handle_space_render,
            # Write commands
            'update-timeline-event': self._handle_update_timeline_event,
            'update-veil-facet': self._handle_update_veil_facet,
            # REPL commands
            'repl-create': self._handle_repl_create,
            'repl-exec': self._handle_repl_exec,
            'repl-sessions': self._handle_repl_sessions,
            'repl-history': self._handle_repl_history,
        }
    
    def start(self):
        """Start the CLI command handler in a background thread."""
        if self.is_running:
            logger.warning("CLI command handler is already running")
            return
        
        logger.info("Starting CLI command handler plugin...")
        self.is_running = True
        
        # Start worker thread with its own event loop
        self.worker_thread = threading.Thread(target=self._worker_thread_main, daemon=True)
        self.worker_thread.start()
        
        logger.info("CLI command handler plugin started successfully")
    
    def stop(self):
        """Stop the CLI command handler."""
        if not self.is_running:
            return
        
        logger.info("Stopping CLI command handler plugin...")
        self.is_running = False
        
        # Signal worker thread to stop
        try:
            self.command_queue.put(None, timeout=1.0)  # Sentinel value
        except:
            pass
        
        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        
        logger.info("CLI command handler plugin stopped")
    
    def _worker_thread_main(self):
        """Main function for the worker thread."""
        try:
            # Create event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            logger.debug("CLI command handler worker thread started")
            
            # Process commands until stopped
            while self.is_running:
                try:
                    # Get command from queue (blocking with timeout)
                    command_request = self.command_queue.get(timeout=1.0)
                    
                    # Check for sentinel value (stop signal)
                    if command_request is None:
                        break
                    
                    # Process the command
                    self.loop.run_until_complete(self._process_command(command_request))
                    
                except Empty:
                    # Timeout - continue checking if we should stop
                    continue
                except Exception as e:
                    logger.error(f"Error in CLI command handler worker thread: {e}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Fatal error in CLI command handler worker thread: {e}", exc_info=True)
        finally:
            if self.loop:
                self.loop.close()
            logger.debug("CLI command handler worker thread stopped")
    
    async def _process_command(self, request: CommandRequest):
        """
        Process a single command request.
        
        Args:
            request: The command request to process
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Processing CLI command: {request.command} (id: {request.request_id})")
            
            # Execute the command
            if request.command in self.command_map:
                handler_func = self.command_map[request.command]
                result = await handler_func(request.args)
                error = None
            else:
                result = {
                    "error": f"Unknown command: {request.command}",
                    "available_commands": list(self.command_map.keys())
                }
                error = f"Unknown command: {request.command}"
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Update statistics
            self.commands_processed += 1
            self.last_command_time = time.time()
            
            # Create response
            response = CommandResponse(
                result=result,
                error=error,
                request_id=request.request_id,
                execution_time_ms=execution_time_ms
            )
            
            # Send response via callback if provided
            if request.response_callback:
                try:
                    request.response_callback(response)
                except Exception as e:
                    logger.error(f"Error calling response callback: {e}", exc_info=True)
            
            logger.debug(f"CLI command completed: {request.command} ({execution_time_ms:.1f}ms)")
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Error processing CLI command {request.command}: {e}", exc_info=True)
            
            # Create error response
            response = CommandResponse(
                result={"error": f"Internal error: {str(e)}"},
                error=str(e),
                request_id=request.request_id,
                execution_time_ms=execution_time_ms
            )
            
            # Send error response via callback if provided
            if request.response_callback:
                try:
                    request.response_callback(response)
                except Exception as callback_error:
                    logger.error(f"Error calling response callback for error: {callback_error}", exc_info=True)
    
    def submit_command(self, command: str, args: Dict[str, Any] = None, 
                      request_id: str = None, response_callback: Callable = None) -> bool:
        """
        Submit a command for processing.
        
        Args:
            command: Command name to execute
            args: Command arguments dictionary
            request_id: Unique request identifier
            response_callback: Function to call with the response
            
        Returns:
            True if command was submitted successfully, False otherwise
        """
        if not self.is_running:
            logger.warning("CLI command handler is not running - cannot submit command")
            return False
        
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"cmd_{int(time.time() * 1000000)}"  # Microsecond timestamp
        
        # Create command request
        request = CommandRequest(
            command=command,
            args=args or {},
            request_id=request_id,
            response_callback=response_callback
        )
        
        try:
            # Submit to queue
            self.command_queue.put(request, timeout=1.0)
            logger.debug(f"Command submitted: {command} (id: {request_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to submit command {command}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            "is_running": self.is_running,
            "start_time": self.start_time,
            "uptime_seconds": time.time() - self.start_time,
            "commands_processed": self.commands_processed,
            "last_command_time": self.last_command_time,
            "queue_size": self.command_queue.qsize(),
            "available_commands": list(self.command_map.keys())
        }
    
    # Command handler methods (delegates to InspectorEndpointHandlers)
    
    async def _handle_overview(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle overview/root command."""
        return await self.handlers.handle_root()
    
    async def _handle_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status command."""
        return await self.handlers.handle_status()
    
    async def _handle_spaces(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle spaces command."""
        return await self.handlers.handle_spaces()
    
    async def _handle_agents(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agents command."""
        return await self.handlers.handle_agents()
    
    async def _handle_adapters(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle adapters command."""
        return await self.handlers.handle_adapters()
    
    async def _handle_metrics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle metrics command."""
        return await self.handlers.handle_metrics()
    
    async def _handle_timelines(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle timelines command."""
        return await self.handlers.handle_timelines()
    
    async def _handle_timeline_details(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle timeline-details command."""
        space_id = args.get('space_id')
        timeline_id = args.get('timeline_id')
        limit = args.get('limit', 100)
        offset = args.get('offset')
        return await self.handlers.handle_timeline_details(space_id, timeline_id, limit, offset)
    
    async def _handle_veil(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle veil command."""
        return await self.handlers.handle_veil()
    
    async def _handle_veil_space(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle veil-space command."""
        space_id = args.get('space_id')
        return await self.handlers.handle_veil_space(space_id)
    
    async def _handle_veil_facets(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle veil-facets command."""
        space_id = args.get('space_id')
        facet_type = args.get('facet_type')
        owner_id = args.get('owner_id')
        limit = args.get('limit', 100)
        after_facet_id = args.get('after_facet_id')
        return await self.handlers.handle_veil_facets(space_id, facet_type, owner_id, limit, after_facet_id)
    
    async def _handle_veil_facet_details(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle veil-facet-details command."""
        space_id = args.get('space_id')
        facet_id = args.get('facet_id')
        return await self.handlers.handle_veil_facet_details(space_id, facet_id)
    
    async def _handle_health(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health command."""
        return await self.handlers.handle_health()
    
    async def _handle_space_render(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle space-render command."""
        space_id = args.get('space_id')
        options = {
            'format': args.get('format', 'markdown'),
            'include_tools': args.get('include_tools', False),
            'max_messages': args.get('max_messages', 50)
        }
        return await self.handlers.handle_space_render(space_id, options)
    
    async def _handle_update_timeline_event(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update-timeline-event command."""
        event_id = args.get('event_id')
        update_data = args.get('update_data')
        space_id = args.get('space_id')
        timeline_id = args.get('timeline_id')
        return await self.handlers.handle_update_timeline_event(event_id, update_data, space_id, timeline_id)
    
    async def _handle_update_veil_facet(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update-veil-facet command."""
        space_id = args.get('space_id')
        facet_id = args.get('facet_id')
        update_data = args.get('update_data')
        return await self.handlers.handle_update_veil_facet(space_id, facet_id, update_data)
    
    async def _handle_repl_create(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle repl-create command."""
        context_type = args.get('context_type')
        context_id = args.get('context_id')
        target_path = args.get('target_path')
        
        result = await self.handlers.handle_repl_create(context_type, context_id, target_path)
        
        # Format output for terminal display
        if result.get('success'):
            session_info = result.get('session', {})
            formatted_result = {
                **result,
                'formatted_output': self._format_repl_session_created(session_info)
            }
            return formatted_result
        else:
            return result
    
    async def _handle_repl_exec(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle repl-exec command."""
        session_id = args.get('session_id')
        code = args.get('code')
        timeout = args.get('timeout', 5.0)
        
        # Handle base64 encoded code for multi-line support
        if args.get('code_base64'):
            try:
                code = base64.b64decode(args['code_base64']).decode('utf-8')
            except Exception as e:
                return {
                    'error': f'Failed to decode base64 code: {e}',
                    'timestamp': time.time()
                }
        
        result = await self.handlers.handle_repl_execute(session_id, code, timeout)
        
        # Format output for terminal display - always format, even on failure
        formatted_result = {
            **result,
            'formatted_output': self._format_repl_execution_result(result)
        }
        return formatted_result
    
    async def _handle_repl_sessions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle repl-sessions command."""
        result = await self.handlers.handle_repl_sessions()
        
        # Format output for terminal display
        if not result.get('error'):
            formatted_result = {
                **result,
                'formatted_output': self._format_repl_sessions_list(result)
            }
            return formatted_result
        else:
            return result
    
    async def _handle_repl_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle repl-history command."""
        session_id = args.get('session_id')
        limit = args.get('limit', 50)
        offset = args.get('offset', 0)
        
        result = await self.handlers.handle_repl_session_history(session_id, limit, offset)
        
        # Format output for terminal display
        if result.get('success'):
            formatted_result = {
                **result,
                'formatted_output': self._format_repl_session_history(result)
            }
            return formatted_result
        else:
            return result
    
    def _format_repl_session_created(self, session_info: Dict[str, Any]) -> str:
        """Format REPL session creation output for terminal."""
        lines = [
            f"✓ REPL session created successfully",
            f"  Session ID: {session_info.get('id')}",
            f"  Context Type: {session_info.get('type')}",
            f"  Context ID: {session_info.get('context_id')}",
            f"  Created: {session_info.get('created_at')}",
            f"  Namespace Keys: {len(session_info.get('namespace_keys', []))} available",
            "",
            "Available variables:"
        ]
        
        # Show first few namespace keys
        namespace_keys = session_info.get('namespace_keys', [])
        for key in namespace_keys[:10]:  # Show first 10 keys
            lines.append(f"  - {key}")
        
        if len(namespace_keys) > 10:
            lines.append(f"  ... and {len(namespace_keys) - 10} more")
        
        return "\n".join(lines)
    
    def _format_repl_execution_result(self, result: Dict[str, Any]) -> str:
        """Format REPL execution result for terminal."""
        lines = []
        
        # Show execution status
        if result.get('success'):
            lines.append("✓ Code executed successfully")
        else:
            lines.append("✗ Code execution failed")
        
        # Show execution time
        exec_time = result.get('execution_time_ms', 0)
        lines.append(f"  Execution time: {exec_time:.2f}ms")
        
        # Show output if present
        output = result.get('output', '').strip()
        if output:
            lines.append("")
            lines.append("Output:")
            for line in output.split('\n'):
                lines.append(f"  {line}")
        
        # Show error if present
        error = result.get('error', '').strip()
        if error:
            lines.append("")
            lines.append("Error:")
            for line in error.split('\n'):
                lines.append(f"  {line}")
        
        # Show namespace changes
        namespace_changes = result.get('namespace_changes', [])
        if namespace_changes:
            lines.append("")
            lines.append("New variables created:")
            for var in namespace_changes:
                lines.append(f"  - {var}")
        
        return "\n".join(lines)
    
    def _format_repl_sessions_list(self, result: Dict[str, Any]) -> str:
        """Format REPL sessions list for terminal."""
        sessions = result.get('sessions', [])
        total = result.get('total_sessions', 0)
        
        if total == 0:
            return "No active REPL sessions found."
        
        lines = [f"Found {total} active REPL session(s):", ""]
        
        for session in sessions:
            lines.extend([
                f"Session: {session.get('id')}",
                f"  Type: {session.get('type')}",
                f"  Context: {session.get('context_id')}",
                f"  Created: {session.get('created_at')}",
                f"  Last accessed: {session.get('last_accessed')}",
                f"  History entries: {session.get('history_length', 0)}",
                f"  Target alive: {session.get('target_alive', True)}",
                ""
            ])
        
        return "\n".join(lines)
    
    def _format_repl_session_history(self, result: Dict[str, Any]) -> str:
        """Format REPL session history for terminal."""
        session_id = result.get('session_id')
        history = result.get('history', [])
        total_count = result.get('total_history_count', 0)
        returned_count = result.get('returned_count', 0)
        offset = result.get('offset', 0)
        has_more = result.get('has_more', False)
        
        lines = [
            f"History for session: {session_id}",
            f"Showing {returned_count} of {total_count} entries (offset: {offset})",
            ""
        ]
        
        if not history:
            lines.append("No history entries found.")
            return "\n".join(lines)
        
        for i, entry in enumerate(history):
            entry_num = total_count - offset - i  # Calculate actual entry number
            lines.extend([
                f"Entry #{entry_num} - {entry.get('timestamp')}",
                f"  Code: {entry.get('code')}",
            ])
            
            # Show output if present
            output = entry.get('output', '').strip()
            if output:
                lines.append("  Output:")
                for line in output.split('\n'):
                    lines.append(f"    {line}")
            
            # Show error if present
            error = entry.get('error', '').strip()
            if error:
                lines.append("  Error:")
                for line in error.split('\n'):
                    lines.append(f"    {line}")
            
            lines.append(f"  Success: {entry.get('success')}")
            lines.append(f"  Execution time: {entry.get('execution_time_ms', 0):.2f}ms")
            lines.append("")
        
        if has_more:
            lines.append(f"... {total_count - offset - returned_count} more entries available")
        
        return "\n".join(lines)


def register_cli_commands(host_instance) -> CLICommandHandler:
    """
    Register CLI command handling within the running host process.
    
    Args:
        host_instance: The running host instance
        
    Returns:
        The CLI command handler instance
    """
    logger.info("Registering CLI command handler plugin...")
    
    # Create and start the CLI command handler
    cli_handler = CLICommandHandler(host_instance)
    cli_handler.start()
    
    logger.info("CLI command handler plugin registered successfully")
    return cli_handler