#!/usr/bin/env python3
"""
Standalone CLI Inspector Script

Connects to a running Connectome host and provides CLI inspection capabilities.
This script can be run independently of the host process.
"""

import asyncio
import sys
import logging
import argparse
from pathlib import Path

# Add the connectome root to path so we can import modules
connectome_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(connectome_root))

from host.modules.inspector.cli_inspector import CLIInspector, create_cli_parser


class StandaloneCLIInspector:
    """
    Standalone CLI inspector that can connect to a running Connectome host.
    
    For now, this is a placeholder that demonstrates the structure.
    Full implementation would need a way to connect to the running host
    (e.g., via IPC, socket, or shared memory).
    """
    
    def __init__(self):
        self.host_instance = None
    
    async def connect_to_host(self) -> bool:
        """
        Connect to a running Connectome host.
        
        Returns:
            True if successfully connected, False otherwise
        """
        # TODO: Implement connection mechanism
        # This could be:
        # 1. IPC socket connection
        # 2. Shared memory mapping
        # 3. HTTP client to inspector server
        # 4. Direct process inspection
        
        print("❌ Standalone host connection not yet implemented")
        print("💡 Suggestion: Use the web inspector at http://localhost:8080")
        print("   or run inspector commands from within the host process")
        return False
    
    async def run_cli(self, args=None):
        """Run the CLI inspector."""
        parser = create_cli_parser()
        parser.add_argument(
            '--host-port', 
            type=int, 
            default=8080,
            help='Port of the running inspector web server (for HTTP client mode)'
        )
        parser.add_argument(
            '--mode',
            choices=['direct', 'http'],
            default='http',
            help='Connection mode: direct (IPC) or http (web client)'
        )
        
        parsed_args = parser.parse_args(args)
        
        if parsed_args.mode == 'http':
            return await self._run_http_client_mode(parsed_args)
        else:
            return await self._run_direct_mode(parsed_args)
    
    async def _run_direct_mode(self, args):
        """Run in direct connection mode (not yet implemented)."""
        success = await self.connect_to_host()
        if not success:
            return 1
        
        # Would use CLIInspector with connected host instance
        # cli_inspector = CLIInspector(self.host_instance)
        # ... rest of CLI logic
        
        return 1
    
    async def _run_http_client_mode(self, args):
        """Run as HTTP client to the web inspector."""
        import aiohttp
        import json
        
        base_url = f"http://localhost:{args.host_port}"
        
        # Map CLI commands to HTTP endpoints
        endpoint_map = {
            'overview': '/',
            'status': '/status',
            'spaces': '/spaces',
            'agents': '/agents',
            'adapters': '/adapters',
            'metrics': '/metrics',
            'timelines': '/timelines',
            'veil': '/veil',
            'health': '/health'
        }
        
        # Handle parameterized endpoints
        if args.command == 'timeline-details':
            if not args.space_id:
                print("❌ --space-id is required for timeline-details command")
                return 1
            endpoint = f"/timelines/{args.space_id}"
            if args.timeline_id:
                endpoint += f"/{args.timeline_id}"
            if args.limit != 100:
                endpoint += f"?limit={args.limit}"
        elif args.command == 'veil-space':
            if not args.space_id:
                print("❌ --space-id is required for veil-space command")
                return 1
            endpoint = f"/veil/{args.space_id}"
        elif args.command == 'veil-facets':
            if not args.space_id:
                print("❌ --space-id is required for veil-facets command")
                return 1
            endpoint = f"/veil/{args.space_id}/facets"
            params = []
            if args.facet_type:
                params.append(f"type={args.facet_type}")
            if args.owner_id:
                params.append(f"owner={args.owner_id}")
            if args.limit != 100:
                params.append(f"limit={args.limit}")
            if params:
                endpoint += "?" + "&".join(params)
        elif args.command == 'veil-facet-details':
            if not args.space_id or not args.facet_id:
                print("❌ Both --space-id and --facet-id are required for veil-facet-details command")
                return 1
            endpoint = f"/veil/{args.space_id}/facets/{args.facet_id}"
        else:
            endpoint = endpoint_map.get(args.command)
            if not endpoint:
                print(f"❌ Unknown command: {args.command}")
                return 1
        
        # Make HTTP request
        try:
            async with aiohttp.ClientSession() as session:
                url = base_url + endpoint
                if args.verbose:
                    print(f"🔗 Requesting: {url}")
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Format output using the same logic as CLI inspector
                        from host.modules.inspector.cli_inspector import CLIInspector
                        dummy_inspector = CLIInspector(None)  # Just for formatting
                        output = dummy_inspector.format_output(data, args.format)
                        print(output)
                        return 0
                    else:
                        error_text = await response.text()
                        print(f"❌ HTTP Error {response.status}: {error_text}")
                        return 1
        
        except aiohttp.ClientConnectorError:
            print(f"❌ Could not connect to inspector server at {base_url}")
            print("💡 Make sure the Connectome host is running with inspector enabled")
            return 1
        except Exception as e:
            print(f"❌ Request failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1


async def main():
    """Main entry point for standalone CLI."""
    standalone = StandaloneCLIInspector()
    exit_code = await standalone.run_cli()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())