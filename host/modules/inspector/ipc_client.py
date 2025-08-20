"""
IPC Client for Inspector Commands

Client-side communication for sending inspector commands to running Connectome host.
External CLI processes use this to communicate with the in-process command handler.
"""

import asyncio
import json
import logging
import os
import socket
import time
from typing import Dict, Any, Optional, List
import uuid

from .ipc_server import discover_inspector_sockets

logger = logging.getLogger(__name__)


class IPCClient:
    """
    IPC Client for sending inspector commands to a running Connectome host.
    
    Uses Unix domain sockets to communicate with the host's IPC server.
    """
    
    def __init__(self, socket_path: str = None, timeout: float = 30.0):
        """
        Initialize the IPC client.
        
        Args:
            socket_path: Path to Unix socket (auto-discovered if None)
            timeout: Command timeout in seconds
        """
        self.socket_path = socket_path
        self.timeout = timeout
        self.reader = None
        self.writer = None
        self.is_connected = False
    
    async def connect(self) -> bool:
        """
        Connect to the IPC server.
        
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            # Auto-discover socket if not specified
            if not self.socket_path:
                socket_info = self._auto_discover_socket()
                if not socket_info:
                    logger.error("No running Connectome inspector found")
                    return False
                self.socket_path = socket_info["socket_path"]
                logger.debug(f"Auto-discovered socket: {self.socket_path}")
            
            # Check if socket exists
            if not os.path.exists(self.socket_path):
                logger.error(f"Socket does not exist: {self.socket_path}")
                return False
            
            # Connect to Unix socket
            self.reader, self.writer = await asyncio.open_unix_connection(self.socket_path)
            self.is_connected = True
            
            logger.debug(f"Connected to inspector IPC server: {self.socket_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to IPC server: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the IPC server."""
        if not self.is_connected:
            return
        
        try:
            if self.writer:
                self.writer.close()
                await self.writer.wait_closed()
        except Exception as e:
            logger.debug(f"Error during disconnect: {e}")
        finally:
            self.reader = None
            self.writer = None
            self.is_connected = False
            logger.debug("Disconnected from IPC server")
    
    async def send_command(self, command: str, args: Dict[str, Any] = None, 
                          request_id: str = None) -> Dict[str, Any]:
        """
        Send a command to the IPC server.
        
        Args:
            command: Command name to execute
            args: Command arguments
            request_id: Unique request ID (generated if None)
            
        Returns:
            Response dictionary from server
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to IPC server")
        
        # Generate request ID if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # Prepare command data
        command_data = {
            "command": command,
            "args": args or {},
            "request_id": request_id
        }
        
        try:
            # Send command
            command_json = json.dumps(command_data) + '\n'
            self.writer.write(command_json.encode())
            await self.writer.drain()
            
            # Read response with timeout
            response_data = await asyncio.wait_for(
                self.reader.readline(),
                timeout=self.timeout
            )
            
            if not response_data:
                raise RuntimeError("Server closed connection")
            
            # Parse response
            response = json.loads(response_data.decode().strip())
            return response
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"Command timed out after {self.timeout} seconds")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid response from server: {e}")
        except Exception as e:
            raise RuntimeError(f"Error sending command: {e}")
    
    def _auto_discover_socket(self) -> Optional[Dict[str, Any]]:
        """
        Auto-discover a running inspector socket.
        
        Returns:
            Socket info dictionary or None if none found
        """
        try:
            sockets = discover_inspector_sockets()
            
            if not sockets:
                return None
            
            # Return the most recent one (by PID or creation time)
            if len(sockets) == 1:
                return sockets[0]
            
            # Multiple sockets - choose the most recently created
            latest = max(sockets, key=lambda s: s.get("created_at", 0))
            return latest
            
        except Exception as e:
            logger.debug(f"Error during socket auto-discovery: {e}")
            return None
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        return {
            "socket_path": self.socket_path,
            "is_connected": self.is_connected,
            "timeout": self.timeout
        }


class IPCCommandExecutor:
    """
    High-level executor for IPC commands with automatic connection management.
    
    Provides a simple interface for executing commands with connection handling.
    """
    
    def __init__(self, socket_path: str = None, timeout: float = 30.0):
        """
        Initialize the command executor.
        
        Args:
            socket_path: Path to Unix socket (auto-discovered if None)
            timeout: Command timeout in seconds
        """
        self.socket_path = socket_path
        self.timeout = timeout
    
    async def execute_command(self, command: str, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a single command with automatic connection management.
        
        Args:
            command: Command name to execute
            args: Command arguments
            
        Returns:
            Command result dictionary
        """
        client = IPCClient(self.socket_path, self.timeout)
        
        try:
            # Connect
            if not await client.connect():
                return {
                    "error": "Failed to connect to Connectome inspector",
                    "details": "Make sure Connectome is running with --enable-inspector"
                }
            
            # Execute command
            response = await client.send_command(command, args)
            return response
            
        except Exception as e:
            return {
                "error": f"Command execution failed: {str(e)}",
                "command": command
            }
        finally:
            await client.disconnect()
    
    async def list_available_hosts(self) -> List[Dict[str, Any]]:
        """
        List all available Connectome hosts with inspectors.
        
        Returns:
            List of host information dictionaries
        """
        try:
            sockets = discover_inspector_sockets()
            
            # Enhance with connection test
            available_hosts = []
            for socket_info in sockets:
                # Test connection
                test_client = IPCClient(socket_info["socket_path"], timeout=5.0)
                try:
                    if await test_client.connect():
                        # Try a simple command to verify it's working
                        response = await test_client.send_command("health")
                        if response.get("error"):
                            socket_info["status"] = "error"
                            socket_info["error"] = response["error"]
                        else:
                            socket_info["status"] = "available"
                            socket_info["health"] = response.get("result", {})
                    else:
                        socket_info["status"] = "unreachable"
                except Exception as e:
                    socket_info["status"] = "error"
                    socket_info["error"] = str(e)
                finally:
                    await test_client.disconnect()
                
                available_hosts.append(socket_info)
            
            return available_hosts
            
        except Exception as e:
            logger.error(f"Error listing available hosts: {e}")
            return []


# Convenience functions for common use cases

async def execute_inspector_command(command: str, args: Dict[str, Any] = None, 
                                  socket_path: str = None, timeout: float = 30.0) -> Dict[str, Any]:
    """
    Execute an inspector command with automatic connection handling.
    
    Args:
        command: Command name to execute
        args: Command arguments
        socket_path: Socket path (auto-discovered if None)
        timeout: Timeout in seconds
        
    Returns:
        Command result dictionary
    """
    executor = IPCCommandExecutor(socket_path, timeout)
    return await executor.execute_command(command, args)


async def get_connectome_status(socket_path: str = None) -> Dict[str, Any]:
    """Get Connectome system status via IPC."""
    return await execute_inspector_command("status", socket_path=socket_path)


async def get_connectome_health(socket_path: str = None) -> Dict[str, Any]:
    """Get Connectome health check via IPC."""
    return await execute_inspector_command("health", socket_path=socket_path)


async def list_connectome_spaces(socket_path: str = None) -> Dict[str, Any]:
    """Get Connectome spaces information via IPC."""
    return await execute_inspector_command("spaces", socket_path=socket_path)


async def list_connectome_agents(socket_path: str = None) -> Dict[str, Any]:
    """Get Connectome agents information via IPC."""
    return await execute_inspector_command("agents", socket_path=socket_path)


def discover_connectome_hosts() -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for discovering Connectome hosts.
    
    Returns:
        List of host information dictionaries
    """
    try:
        # Just use the socket discovery from ipc_server module directly
        from .ipc_server import discover_inspector_sockets
        return discover_inspector_sockets()
    except Exception as e:
        logger.debug(f"Error discovering hosts: {e}")
        return []