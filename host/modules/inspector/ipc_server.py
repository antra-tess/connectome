"""
IPC Server for Inspector Commands

Provides inter-process communication for CLI inspector commands.
External processes can connect via Unix socket to send commands to the running host.
"""

import asyncio
import json
import logging
import os
import socket
import tempfile
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
from concurrent.futures import Future

from .cli_handler import CLICommandHandler, CommandResponse

logger = logging.getLogger(__name__)


class IPCServer:
    """
    IPC Server for handling inspector commands from external processes.
    
    Uses Unix domain sockets for communication with external CLI clients.
    """
    
    def __init__(self, cli_handler: CLICommandHandler, socket_path: str = None):
        """
        Initialize the IPC server.
        
        Args:
            cli_handler: CLI command handler to process commands
            socket_path: Path to Unix socket (auto-generated if None)
        """
        self.cli_handler = cli_handler
        self.socket_path = socket_path or self._generate_socket_path()
        self.server = None
        self.is_running = False
        self.connections: List[asyncio.StreamWriter] = []
        
        # Statistics
        self.start_time = None
        self.connections_served = 0
        self.commands_processed = 0
        
    def _generate_socket_path(self) -> str:
        """Generate a socket path for this host instance."""
        # Use process ID to allow multiple instances
        pid = os.getpid()
        temp_dir = tempfile.gettempdir()
        socket_path = os.path.join(temp_dir, f"connectome_inspector_{pid}.sock")
        return socket_path
    
    async def start(self):
        """Start the IPC server."""
        if self.is_running:
            logger.warning("IPC server is already running")
            return
        
        try:
            # Clean up any existing socket file
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            
            # Create Unix socket server
            self.server = await asyncio.start_unix_server(
                self._handle_client,
                path=self.socket_path
            )
            
            self.is_running = True
            self.start_time = time.time()
            
            # Make socket accessible
            os.chmod(self.socket_path, 0o666)
            
            logger.info(f"IPC server started on socket: {self.socket_path}")
            
        except Exception as e:
            logger.error(f"Failed to start IPC server: {e}", exc_info=True)
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the IPC server."""
        if not self.is_running:
            return
        
        logger.info("Stopping IPC server...")
        self.is_running = False
        
        # Close all active connections
        for writer in self.connections.copy():
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as e:
                logger.debug(f"Error closing connection: {e}")
        
        self.connections.clear()
        
        # Stop the server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Clean up socket file
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        except Exception as e:
            logger.debug(f"Error removing socket file: {e}")
        
        logger.info("IPC server stopped")
    
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """
        Handle a client connection.
        
        Args:
            reader: Stream reader for incoming data
            writer: Stream writer for outgoing data
        """
        client_addr = writer.get_extra_info('peername', 'unknown')
        logger.debug(f"Client connected: {client_addr}")
        
        self.connections.append(writer)
        self.connections_served += 1
        
        try:
            while self.is_running:
                # Read command from client
                data = await reader.readline()
                if not data:
                    break  # Client disconnected
                
                try:
                    # Parse JSON command
                    command_data = json.loads(data.decode().strip())
                    logger.debug(f"Received command: {command_data}")
                    
                    # Process command
                    response = await self._process_command(command_data)
                    
                    # Send response
                    response_json = json.dumps(response) + '\n'
                    writer.write(response_json.encode())
                    await writer.drain()
                    
                    self.commands_processed += 1
                    
                except json.JSONDecodeError as e:
                    # Invalid JSON
                    error_response = {
                        "result": None,
                        "error": f"Invalid JSON: {str(e)}",
                        "request_id": None
                    }
                    response_json = json.dumps(error_response) + '\n'
                    writer.write(response_json.encode())
                    await writer.drain()
                
                except Exception as e:
                    logger.error(f"Error processing command from {client_addr}: {e}", exc_info=True)
                    error_response = {
                        "result": None,
                        "error": f"Server error: {str(e)}",
                        "request_id": command_data.get('request_id') if 'command_data' in locals() else None
                    }
                    response_json = json.dumps(error_response) + '\n'
                    writer.write(response_json.encode())
                    await writer.drain()
        
        except ConnectionResetError:
            logger.debug(f"Client {client_addr} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_addr}: {e}", exc_info=True)
        finally:
            # Clean up connection
            try:
                if writer in self.connections:
                    self.connections.remove(writer)
                writer.close()
                await writer.wait_closed()
            except Exception as e:
                logger.debug(f"Error closing connection to {client_addr}: {e}")
            
            logger.debug(f"Client {client_addr} connection closed")
    
    async def _process_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a command request from a client.
        
        Args:
            command_data: Command data from client
            
        Returns:
            Response dictionary
        """
        try:
            # Extract command info
            command = command_data.get('command')
            args = command_data.get('args', {})
            request_id = command_data.get('request_id', f"ipc_{int(time.time() * 1000000)}")
            
            if not command:
                return {
                    "result": None,
                    "error": "No command specified",
                    "request_id": request_id
                }
            
            # Create future for response
            response_future = Future()
            
            def response_callback(response: CommandResponse):
                """Callback to receive response from CLI handler."""
                try:
                    response_future.set_result({
                        "result": response.result,
                        "error": response.error,
                        "request_id": response.request_id,
                        "execution_time_ms": response.execution_time_ms
                    })
                except Exception as e:
                    if not response_future.done():
                        response_future.set_exception(e)
            
            # Submit command to CLI handler
            success = self.cli_handler.submit_command(
                command=command,
                args=args,
                request_id=request_id,
                response_callback=response_callback
            )
            
            if not success:
                return {
                    "result": None,
                    "error": "Failed to submit command to handler",
                    "request_id": request_id
                }
            
            # Wait for response (with timeout)
            try:
                # Run in thread pool to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(None, response_future.result, 30.0),  # 30 second timeout
                    timeout=35.0  # Overall timeout slightly longer
                )
                return response
            except asyncio.TimeoutError:
                return {
                    "result": None,
                    "error": "Command execution timed out",
                    "request_id": request_id
                }
            
        except Exception as e:
            logger.error(f"Error processing command: {e}", exc_info=True)
            return {
                "result": None,
                "error": f"Internal error: {str(e)}",
                "request_id": command_data.get('request_id', 'unknown')
            }
    
    def get_socket_path(self) -> str:
        """Get the socket path for this server."""
        return self.socket_path
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "is_running": self.is_running,
            "socket_path": self.socket_path,
            "start_time": self.start_time,
            "uptime_seconds": time.time() - self.start_time if self.start_time else 0,
            "active_connections": len(self.connections),
            "connections_served": self.connections_served,
            "commands_processed": self.commands_processed
        }


def create_socket_info_file(socket_path: str, host_info: Dict[str, Any] = None):
    """
    Create an info file that helps clients discover the socket.
    
    Args:
        socket_path: Path to the Unix socket
        host_info: Additional host information
    """
    try:
        info = {
            "socket_path": socket_path,
            "pid": os.getpid(),
            "created_at": time.time(),
            "host_info": host_info or {}
        }
        
        # Write to temp directory with PID in name
        temp_dir = tempfile.gettempdir()
        info_file = os.path.join(temp_dir, f"connectome_inspector_info_{os.getpid()}.json")
        
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.debug(f"Created socket info file: {info_file}")
        return info_file
        
    except Exception as e:
        logger.debug(f"Failed to create socket info file: {e}")
        return None


def discover_inspector_sockets() -> List[Dict[str, Any]]:
    """
    Discover available inspector sockets on the system.
    
    Returns:
        List of socket information dictionaries
    """
    sockets = []
    temp_dir = tempfile.gettempdir()
    
    try:
        # Look for info files
        for filename in os.listdir(temp_dir):
            if filename.startswith("connectome_inspector_info_") and filename.endswith(".json"):
                try:
                    info_path = os.path.join(temp_dir, filename)
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    
                    # Check if socket still exists
                    socket_path = info.get("socket_path")
                    if socket_path and os.path.exists(socket_path):
                        # Check if process is still running
                        pid = info.get("pid")
                        if pid and _is_process_running(pid):
                            sockets.append(info)
                        else:
                            # Clean up stale info file
                            try:
                                os.unlink(info_path)
                            except:
                                pass
                    else:
                        # Clean up stale info file
                        try:
                            os.unlink(info_path)
                        except:
                            pass
                            
                except Exception as e:
                    logger.debug(f"Error processing info file {filename}: {e}")
    
    except Exception as e:
        logger.debug(f"Error discovering inspector sockets: {e}")
    
    return sockets


def _is_process_running(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False