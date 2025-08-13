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

from host.modules.inspector.cli_inspector import main_cli
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
    
    # Check if we're running within a host process
    try:
        # Try to import and get existing host instance
        from host.main import get_running_host_instance
        host_instance = get_running_host_instance()
        
        if host_instance:
            # We have a running host, use it directly
            logging.basicConfig(level=logging.WARNING)
            exit_code = await main_cli(host_instance)
            sys.exit(exit_code)
    except (ImportError, AttributeError):
        # No running host instance available
        pass
    
    # Fall back to mock host or standalone mode
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        # Let the CLI parser handle help
        runner = HostInspectorRunner()
        exit_code = await runner.run_with_mock_host()
        sys.exit(exit_code)
    
    print("⚠️  No running host instance detected")
    print("")
    print("📖 Usage options:")
    print("  1. Run from within host process:")
    print("     python -m host.main --enable-inspector")
    print("     # Then in another terminal:")
    print("     python -m host.cli_inspect status")
    print("")
    print("  2. Use the web inspector:")
    print("     # Host running with inspector enabled")
    print("     curl http://localhost:8080/status")
    print("")
    print("  3. Use the standalone HTTP client:")
    print("     python -m host.modules.inspector.standalone_cli status")
    print("")
    print("  4. Mock mode (limited functionality):")
    print("     python -m host.cli_inspect --mock status")
    
    # Check for mock mode
    if '--mock' in sys.argv:
        sys.argv.remove('--mock')
        runner = HostInspectorRunner()
        exit_code = await runner.run_with_mock_host()
        sys.exit(exit_code)
    
    sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())