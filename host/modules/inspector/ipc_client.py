"""
IPC Client for Inspector Commands

Client-side communication for sending inspector commands to running Connectome host.
External CLI processes use this to communicate with the in-process command handler.
"""

import asyncio
import argparse
import json
import logging
import os
import socket
import sys
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
            
            # Connect to Unix socket with increased buffer limit for large timeline events
            # Must match the server's buffer limit to handle large responses
            buffer_limit = 10 * 1024 * 1024  # 10MB
            self.reader, self.writer = await asyncio.open_unix_connection(
                self.socket_path, 
                limit=buffer_limit
            )
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


class InteractiveREPL:
    """
    Interactive REPL for Connectome Inspector via IPC.
    
    Provides a Python-like interactive console that sends commands to a REPL session
    in the running Connectome host.
    """
    
    # ANSI color codes
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def __init__(self, context: str, socket_path: str = None, timeout: float = 30.0):
        """
        Initialize the interactive REPL.
        
        Args:
            context: REPL context in format 'context_type:context_id' (e.g., 'global:test')
            socket_path: Path to Unix socket (auto-discovered if None)
            timeout: Command timeout in seconds
        """
        self.context = context
        self.socket_path = socket_path
        self.timeout = timeout
        self.session_id = None
        self.executor = IPCCommandExecutor(socket_path, timeout)
        self.code_buffer = []
        
        # Parse context
        if ':' in context:
            self.context_type, self.context_id = context.split(':', 1)
        else:
            self.context_type = 'global'
            self.context_id = context
    
    async def start(self):
        """Start the interactive REPL session."""
        try:
            print(f"{self.CYAN}Connectome Interactive REPL{self.RESET}")
            print(f"Context: {self.BOLD}{self.context}{self.RESET}")
            print("Type 'exit()' or press Ctrl+D to exit")
            print("=" * 50)
            
            # Create REPL session
            await self._create_session()
            
            # Start interactive loop
            await self._interactive_loop()
            
        except KeyboardInterrupt:
            print(f"\n{self.YELLOW}Interrupted by user{self.RESET}")
        except EOFError:
            print(f"\n{self.CYAN}Goodbye!{self.RESET}")
        except Exception as e:
            print(f"\n{self.RED}Error: {e}{self.RESET}")
        finally:
            print()
    
    async def _create_session(self):
        """Create a new REPL session."""
        print(f"Creating REPL session...")
        
        result = await self.executor.execute_command("repl-create", {
            "context_type": self.context_type,
            "context_id": self.context_id
        })
        
        if result.get("error"):
            raise RuntimeError(f"Failed to create REPL session: {result['error']}")
        
        session_data = result.get("result", {})
        if not session_data.get("success"):
            raise RuntimeError(f"Failed to create REPL session: {session_data.get('error', 'Unknown error')}")
        
        self.session_id = session_data["session"]["id"]
        session_info = session_data["session"]
        
        print(f"{self.GREEN}✓ REPL session created: {self.session_id}{self.RESET}")
        print(f"Available variables: {len(session_info.get('namespace_keys', []))} items")
        print()
    
    async def _interactive_loop(self):
        """Main interactive loop."""
        while True:
            try:
                # Show appropriate prompt
                if self.code_buffer:
                    prompt = "... "
                else:
                    prompt = ">>> "
                
                # Read input with proper handling of EOF
                try:
                    line = input(prompt).rstrip('\r\n')
                except EOFError:
                    # Ctrl+D pressed
                    break
                
                # Handle special commands
                if line.strip() in ('exit()', 'quit()', 'exit', 'quit'):
                    break
                
                # Add to code buffer
                self.code_buffer.append(line)
                
                # Check if code is complete
                code = '\n'.join(self.code_buffer)
                
                if self._is_code_complete(code):
                    # Execute complete code block
                    if code.strip():  # Only execute if there's actual code
                        await self._execute_code(code)
                    self.code_buffer = []
                
            except KeyboardInterrupt:
                # Ctrl+C pressed - clear current input and start fresh
                print()
                self.code_buffer = []
                continue
    
    def _is_code_complete(self, code: str) -> bool:
        """
        Check if the code is complete and can be executed.
        
        Args:
            code: Code string to check
            
        Returns:
            True if code is complete, False if more input is needed
        """
        if not code.strip():
            return True
        
        try:
            # Try to compile the code
            compile(code, '<stdin>', 'single')
            return True
        except SyntaxError as e:
            # Check if this is an incomplete statement
            if 'unexpected EOF' in str(e) or 'expected an indented block' in str(e):
                return False
            # If it's a different syntax error, the code is complete but invalid
            return True
        except Exception:
            # Other compilation errors - consider it complete
            return True
    
    async def _execute_code(self, code: str):
        """
        Execute code in the REPL session.
        
        Args:
            code: Code to execute
        """
        try:
            result = await self.executor.execute_command("repl-exec", {
                "session_id": self.session_id,
                "code": code
            })
            
            if result.get("error"):
                print(f"{self.RED}Communication error: {result['error']}{self.RESET}")
                return
            
            exec_result = result.get("result", {})
            
            # Show output if present
            output = exec_result.get("output", "").strip()
            if output:
                print(output)
            
            # Show error if present
            error = exec_result.get("error", "").strip()
            if error:
                print(f"{self.RED}{error}{self.RESET}")
            
            # Show namespace changes
            namespace_changes = exec_result.get("namespace_changes", [])
            if namespace_changes and len(namespace_changes) > 0:
                vars_str = ", ".join(namespace_changes)
                print(f"{self.BLUE}# New variables: {vars_str}{self.RESET}")
        
        except Exception as e:
            print(f"{self.RED}Execution error: {e}{self.RESET}")


async def run_interactive_repl(context: str, socket_path: str = None, timeout: float = 30.0):
    """
    Run an interactive REPL session.
    
    Args:
        context: REPL context in format 'context_type:context_id'
        socket_path: Path to Unix socket
        timeout: Command timeout in seconds
    """
    repl = InteractiveREPL(context, socket_path, timeout)
    await repl.start()


def main():
    """Main entry point for IPC client CLI."""
    parser = argparse.ArgumentParser(
        description="Connectome Inspector IPC Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start interactive REPL in global context
  python -m host.modules.inspector.ipc_client --repl global:test
  
  # Start REPL with auto-generated session ID
  python -m host.modules.inspector.ipc_client --repl global
  
  # Execute a single command
  python -m host.modules.inspector.ipc_client --command status
        """
    )
    
    parser.add_argument("--repl", metavar="CONTEXT", 
                        help="Start interactive REPL mode with given context (format: context_type:context_id or just context_id)")
    parser.add_argument("--command", metavar="CMD",
                        help="Execute a single inspector command")
    parser.add_argument("--socket", metavar="PATH",
                        help="Path to Unix socket (auto-discovered if not specified)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Command timeout in seconds (default: 30)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    async def run_async():
        if args.repl:
            # Auto-generate context_id if not provided
            context = args.repl
            if ':' not in context:
                import time
                context = f"global:{context}_{int(time.time())}"
            
            await run_interactive_repl(context, args.socket, args.timeout)
        
        elif args.command:
            # Execute single command
            result = await execute_inspector_command(args.command, socket_path=args.socket, timeout=args.timeout)
            
            # Pretty print result
            if result.get("error"):
                print(f"Error: {result['error']}", file=sys.stderr)
                sys.exit(1)
            else:
                import json
                print(json.dumps(result, indent=2))
        
        else:
            parser.print_help()
    
    # Run async main
    try:
        asyncio.run(run_async())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()