#!/usr/bin/env python3
"""
CLI Inspector Integration

Provides a way to run inspector commands directly within the host process.
Usage: python -m host.cli_inspect <command> [options]
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Ensure we can import from the project root
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from host.modules.inspector.cli_inspector import main_cli, CLIInspector
from host.modules.inspector.ipc_client import IPCCommandExecutor, discover_connectome_hosts
from elements.space_registry import SpaceRegistry


class HostInspectorRunner:
    """
    Runs inspector commands within the context of a host process.
    """
    
    def __init__(self):
        self.host_instance = None
        self.space_registry = None
    
    def initialize_mock_host(self):
        """
        Initialize a minimal host-like object for inspection.
        
        Note: This creates a mock host instance. For full functionality,
        the CLI should be run from within an actual running host process.
        """
        class MockHost:
            def __init__(self):
                self.space_registry = SpaceRegistry.get_instance()
                self.activity_client = None
                self.event_loop = None
        
        self.host_instance = MockHost()
        return self.host_instance
    
    async def run_with_mock_host(self, args=None):
        """
        Run CLI inspector with a mock host instance.
        
        This is useful for testing or when the host is not fully initialized.
        """
        host = self.initialize_mock_host()
        return await main_cli(host, args)


async def main():
    """Main entry point for CLI inspector."""
    
    # Parse arguments to check for special modes
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            # Let the CLI parser handle help in mock mode
            runner = HostInspectorRunner()
            exit_code = await runner.run_with_mock_host()
            sys.exit(exit_code)
        
        # Check for mock mode flag
        if '--mock' in sys.argv:
            sys.argv.remove('--mock')
            logging.basicConfig(level=logging.WARNING)
            runner = HostInspectorRunner()
            exit_code = await runner.run_with_mock_host()
            sys.exit(exit_code)
    
    # Try IPC mode first (preferred)
    try:
        logging.basicConfig(level=logging.WARNING)
        
        # Discover available Connectome hosts
        hosts = discover_connectome_hosts()
        
        if not hosts:
            print("⚠️  No running Connectome host found")
            print("")
            print("📖 Usage options:")
            print("  1. Start Connectome with inspector:")
            print("     python -m host.main --enable-inspector")
            print("     # Then run CLI commands in another terminal:")
            print("     python -m host.cli_inspect status")
            print("")
            print("  2. Use mock mode (limited functionality):")
            print("     python -m host.cli_inspect --mock status")
            print("")
            print("  3. Use HTTP client mode:")
            print("     python -m host.modules.inspector.standalone_cli status")
            sys.exit(1)
        
        # Use the first available host (or most recent one)
        host_info = hosts[0]
        socket_path = host_info["socket_path"]
        
        if len(hosts) > 1:
            logger.debug(f"Multiple hosts found ({len(hosts)}), using: {socket_path}")
        
        # Create IPC executor and run command
        executor = IPCCommandExecutor(socket_path)
        
        # Use the CLI inspector for argument parsing, but execute via IPC
        from host.modules.inspector.cli_inspector import create_cli_parser
        parser = create_cli_parser()
        
        try:
            args = parser.parse_args()
        except SystemExit as e:
            # Help or argument error
            sys.exit(e.code)
        
        # Build command kwargs
        kwargs = {}
        if hasattr(args, 'space_id') and args.space_id:
            kwargs['space_id'] = args.space_id
        if hasattr(args, 'timeline_id') and args.timeline_id:
            kwargs['timeline_id'] = args.timeline_id
        if hasattr(args, 'facet_id') and args.facet_id:
            kwargs['facet_id'] = args.facet_id
        if hasattr(args, 'facet_type') and args.facet_type:
            kwargs['facet_type'] = args.facet_type
        if hasattr(args, 'owner_id') and args.owner_id:
            kwargs['owner_id'] = args.owner_id
        if hasattr(args, 'limit') and args.limit:
            kwargs['limit'] = args.limit
        
        # Execute command via IPC
        response = await executor.execute_command(args.command, kwargs)
        
        # Check for errors
        if response.get("error"):
            print(f"❌ Error: {response['error']}")
            if response.get("details") and args.verbose:
                print(f"Details: {response['details']}")
            sys.exit(1)
        
        # Format and display result
        result_data = response.get("result", response)
        
        # Create CLI inspector just for formatting (no host needed)
        cli = CLIInspector(None)
        output = cli.format_output(result_data, args.format)
        print(output)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Failed to execute command via IPC: {e}")
        if '--verbose' in sys.argv or '-v' in sys.argv:
            import traceback
            traceback.print_exc()
        
        print("\n💡 Try mock mode: python -m host.cli_inspect --mock <command>")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())