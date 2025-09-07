#!/usr/bin/env python3
"""
Standalone TUI Inspector

Standalone launcher for the TUI inspector that connects to a running Connectome instance.
Similar to standalone_cli.py but for the interactive TUI interface.
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path

# Add the connectome root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from .tui_inspector import main_tui
from .ipc_tui import main_ipc_tui


async def main():
    """Main entry point for standalone TUI inspector."""
    parser = argparse.ArgumentParser(
        description="Connectome Inspector TUI - Interactive terminal interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interactive Terminal UI for inspecting running Connectome instances.

Connection Modes:
  - IPC: Direct connection via Unix socket (recommended, faster)
  - HTTP: Connection via web inspector (legacy mode)

Navigation:
  - Use arrow keys to navigate menus and trees
  - Enter: Select/expand items
  - E: Edit leaf values (where supported)
  - B: Go back to previous view
  - R: Refresh current data
  - Q: Quit

The TUI provides:
  - Host discovery and selection
  - Main menu similar to the web interface
  - Tree-like navigation of complex data structures
  - Real-time editing of configuration values
  - Syntax-highlighted JSON display
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['ipc', 'http', 'auto'],
        default='auto',
        help='Connection mode: ipc (Unix socket), http (web API), auto (try IPC first, fallback to HTTP)'
    )
    
    parser.add_argument(
        '--socket', '-s',
        type=str,
        help='Unix socket path for IPC mode (auto-discovered if not specified)'
    )
    
    parser.add_argument(
        '--host', '-H',
        default='localhost',
        help='Host to connect to for HTTP mode (default: localhost)'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        help='Port to connect to for HTTP mode (if not specified, auto-discover)'
    )
    
    parser.add_argument(
        '--timeout', '-t',
        type=float,
        default=30.0,
        help='Command timeout in seconds (default: 30.0)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
    else:
        # Minimal logging for TUI - don't interfere with display
        logging.basicConfig(level=logging.WARNING)
    
    # Determine connection mode
    mode = args.mode
    
    # Try IPC mode first (if auto or explicitly selected)
    if mode in ['ipc', 'auto']:
        try:
            print("🔍 Trying IPC connection...")
            
            # Test IPC connection
            from .ipc_client import IPCCommandExecutor
            test_executor = IPCCommandExecutor(args.socket, timeout=5.0)
            test_response = await test_executor.execute_command("health")
            
            if not test_response.get("error"):
                print("✅ IPC connection successful!")
                print("\n🚀 Starting IPC TUI Inspector...")
                print("💡 Use Q to quit, arrow keys to navigate")
                print("─" * 50)
                
                # Start the IPC TUI
                await main_ipc_tui(args.socket, args.timeout)
                return 0
            else:
                if mode == 'ipc':
                    print(f"❌ IPC connection failed: {test_response['error']}")
                    return 1
                else:
                    print(f"⚠️  IPC connection failed: {test_response['error']}")
                    print("🔄 Falling back to HTTP mode...")
                    
        except Exception as e:
            if mode == 'ipc':
                print(f"❌ IPC connection failed: {e}")
                print("💡 Make sure Connectome is running with inspector enabled")
                return 1
            else:
                print(f"⚠️  IPC connection failed: {e}")
                print("🔄 Falling back to HTTP mode...")
    
    # HTTP mode (if explicitly selected or fallback from auto)
    if mode in ['http', 'auto']:
        host = args.host
        port = args.port
        
        if not port:
            try:
                # Simple auto-discovery by trying common ports
                print("🔍 Auto-discovering HTTP Connectome instance...")
                common_ports = [8080, 8000, 8081, 3000]
                
                import aiohttp
                discovered_port = None
                
                async with aiohttp.ClientSession() as session:
                    for test_port in common_ports:
                        try:
                            test_url = f"http://{host}:{test_port}/"
                            async with session.get(test_url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    if data.get('service') == 'Connectome Inspector':
                                        discovered_port = test_port
                                        break
                        except Exception:
                            continue
                
                if discovered_port:
                    port = discovered_port
                    print(f"📡 Auto-discovered Connectome instance on port {port}")
                else:
                    print("❌ No running Connectome instance found on common ports")
                    print("💡 Make sure Connectome is running with inspector enabled")
                    print(f"   Tried ports: {', '.join(map(str, common_ports))}")
                    return 1
            except Exception as e:
                print(f"❌ Auto-discovery failed: {e}")
                return 1
        
        if not port:
            print("❌ No port specified and auto-discovery failed")
            print("💡 Use --port to specify manually, or ensure Connectome is running")
            return 1
        
        # Create HTTP client URL
        base_url = f"http://{host}:{port}"
        
        try:
            # Test connection
            print(f"🔗 Connecting to {base_url}...")
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url + "/") as response:
                    if response.status == 200:
                        test_result = await response.json()
                        if test_result.get('service') == 'Connectome Inspector':
                            print(f"✅ Connected to {test_result.get('service', 'Unknown')} v{test_result.get('version', 'unknown')}")
                        else:
                            print(f"⚠️  Connected, but unexpected response: {test_result}")
                    else:
                        error_text = await response.text()
                        print(f"❌ HTTP Error {response.status}: {error_text}")
                        return 1
            
            print("\n🚀 Starting HTTP TUI Inspector...")
            print("💡 Use Q to quit, arrow keys to navigate")
            print("─" * 50)
            
            # Start the TUI with the HTTP base URL
            await main_tui(base_url)
            
            return 0
            
        except aiohttp.ClientConnectorError as e:
            print(f"❌ Connection failed: {e}")
            print(f"💡 Ensure Connectome is running on {host}:{port} with inspector enabled")
            return 1
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    print("❌ No valid connection mode could be established")
    return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)