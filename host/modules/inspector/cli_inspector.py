#!/usr/bin/env python3
"""
CLI Inspector Tool

Command-line interface for inspecting Connectome host state.
Uses the same endpoint handlers as the web inspector for consistency.
"""

import argparse
import asyncio
import json
import sys
import logging
from typing import Dict, Any, Optional

from .endpoint_handlers import InspectorEndpointHandlers

logger = logging.getLogger(__name__)


class CLIInspector:
    """
    Command-line inspector tool for Connectome host state.
    
    Provides the same endpoints as the web inspector but through CLI interface.
    """
    
    def __init__(self, host_instance):
        """
        Initialize CLI inspector.
        
        Args:
            host_instance: Reference to the main Host instance
        """
        self.host_instance = host_instance
        self.handlers = InspectorEndpointHandlers(host_instance)
    
    async def run_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a specific inspector command.
        
        Args:
            command: The command to execute
            **kwargs: Additional arguments for the command
            
        Returns:
            Dictionary containing the command results
        """
        try:
            if command == "root" or command == "overview":
                return await self.handlers.handle_root()
            elif command == "status":
                return await self.handlers.handle_status()
            elif command == "spaces":
                return await self.handlers.handle_spaces()
            elif command == "agents":
                return await self.handlers.handle_agents()
            elif command == "adapters":
                return await self.handlers.handle_adapters()
            elif command == "metrics":
                return await self.handlers.handle_metrics()
            elif command == "timelines":
                return await self.handlers.handle_timelines()
            elif command == "timeline-details":
                space_id = kwargs.get('space_id')
                timeline_id = kwargs.get('timeline_id')
                limit = kwargs.get('limit', 100)
                return await self.handlers.handle_timeline_details(space_id, timeline_id, limit)
            elif command == "veil":
                return await self.handlers.handle_veil()
            elif command == "veil-space":
                space_id = kwargs.get('space_id')
                return await self.handlers.handle_veil_space(space_id)
            elif command == "veil-facets":
                space_id = kwargs.get('space_id')
                facet_type = kwargs.get('facet_type')
                owner_id = kwargs.get('owner_id')
                limit = kwargs.get('limit', 100)
                return await self.handlers.handle_veil_facets(space_id, facet_type, owner_id, limit)
            elif command == "veil-facet-details":
                space_id = kwargs.get('space_id')
                facet_id = kwargs.get('facet_id')
                return await self.handlers.handle_veil_facet_details(space_id, facet_id)
            elif command == "health":
                return await self.handlers.handle_health()
            else:
                return {
                    "error": f"Unknown command: {command}",
                    "available_commands": [
                        "overview", "status", "spaces", "agents", "adapters", 
                        "metrics", "timelines", "timeline-details", "veil", 
                        "veil-space", "veil-facets", "veil-facet-details", "health"
                    ]
                }
        except Exception as e:
            logger.error(f"Error executing CLI command '{command}': {e}", exc_info=True)
            return {
                "error": f"Failed to execute command '{command}'",
                "details": str(e)
            }
    
    def format_output(self, data: Dict[str, Any], format_type: str = "json") -> str:
        """
        Format output data for display.
        
        Args:
            data: Data to format
            format_type: Output format ("json", "compact", "summary")
            
        Returns:
            Formatted string
        """
        if format_type == "json":
            return json.dumps(data, indent=2, default=str)
        elif format_type == "compact":
            return json.dumps(data, separators=(',', ':'), default=str)
        elif format_type == "summary":
            return self._format_summary(data)
        else:
            return json.dumps(data, indent=2, default=str)
    
    def _format_summary(self, data: Dict[str, Any]) -> str:
        """Create a human-readable summary of the data."""
        if "error" in data:
            return f"❌ Error: {data['error']}\n"
        
        summary_lines = []
        
        # Handle different data types
        if "service" in data and data.get("service") == "Connectome Inspector":
            # Overview/root data
            summary_lines.append(f"🔍 {data['service']} v{data.get('version', 'unknown')}")
            summary_lines.append(f"⏱️  Uptime: {data.get('uptime_seconds', 0):.1f}s")
            summary_lines.append(f"📊 Requests: {data.get('request_count', 0)}")
            summary_lines.append("\n📋 Available Endpoints:")
            for endpoint, desc in data.get('endpoints', {}).items():
                summary_lines.append(f"  {endpoint} - {desc}")
        
        elif "summary" in data and "total_spaces" in data.get("summary", {}):
            # Spaces data (identified by total_spaces field in summary)
            summary = data["summary"]
            summary_lines.append(f"🏢 Spaces Overview:")
            summary_lines.append(f"  Total: {summary.get('total_spaces', 0)}")
            summary_lines.append(f"  Inner: {summary.get('inner_spaces', 0)}")
            summary_lines.append(f"  Shared: {summary.get('shared_spaces', 0)}")
            summary_lines.append(f"  Other: {summary.get('other_spaces', 0)}")
            
        elif "agents" in data:
            # Agents data
            agents = data["agents"]
            summary_lines.append(f"🤖 Agents Overview:")
            summary_lines.append(f"  Total: {len(agents)}")
            for agent_id, agent_info in agents.items():
                if not agent_info.get("error"):
                    summary_lines.append(f"  {agent_id}: {agent_info.get('name', 'Unknown')}")
        
        elif "adapters" in data:
            # Adapters data
            adapters = data["adapters"]
            if isinstance(adapters, dict):
                summary_lines.append(f"🔌 Adapters Overview:")
                summary_lines.append(f"  Total: {len(adapters)}")
                for adapter_id, adapter_info in adapters.items():
                    if not adapter_info.get("error"):
                        status = adapter_info.get("status", "unknown")
                        summary_lines.append(f"  {adapter_id}: {status}")
        
        elif "process" in data and "memory" in data:
            # Metrics data
            process = data["process"]
            memory = data["memory"]
            cpu = data["cpu"]
            summary_lines.append(f"📊 System Metrics:")
            summary_lines.append(f"  PID: {process.get('pid', 'unknown')}")
            summary_lines.append(f"  Memory: {memory.get('rss_mb', 0):.1f} MB ({memory.get('percent', 0):.1f}%)")
            summary_lines.append(f"  CPU: {cpu.get('percent', 0):.1f}%")
            summary_lines.append(f"  Threads: {process.get('num_threads', 0)}")
            summary_lines.append(f"  Spaces: {data.get('spaces_count', 0)}")
        
        elif "summary" in data and "spaces" in data:
            # Timeline or VEIL data
            summary = data["summary"]
            if "total_timelines" in summary:
                # Timeline data
                summary_lines.append(f"⏰ Timeline Overview:")
                summary_lines.append(f"  Spaces with timelines: {summary.get('total_spaces_with_timelines', 0)}")
                summary_lines.append(f"  Total timelines: {summary.get('total_timelines', 0)}")
                summary_lines.append(f"  Total events: {summary.get('total_events', 0)}")
            elif "total_facets" in summary:
                # VEIL data
                summary_lines.append(f"👁️  VEIL Overview:")
                summary_lines.append(f"  Spaces with VEIL: {summary.get('total_spaces_with_veil', 0)}")
                summary_lines.append(f"  Total facets: {summary.get('total_facets', 0)}")
                facet_types = summary.get('facet_types', {})
                for facet_type, count in facet_types.items():
                    summary_lines.append(f"  {facet_type}: {count}")
        
        elif "status" in data and data.get("status") == "healthy":
            # Health data
            summary_lines.append(f"✅ System Health: {data['status']}")
            summary_lines.append(f"⏱️  Uptime: {data.get('uptime_seconds', 0):.1f}s")
            summary_lines.append(f"📊 Requests: {data.get('request_count', 0)}")
        
        else:
            # Generic data - just show key metrics
            summary_lines.append("📄 Data Summary:")
            if "timestamp" in data:
                import datetime
                ts = datetime.datetime.fromtimestamp(data["timestamp"])
                summary_lines.append(f"  Timestamp: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Show top-level keys and their types
            for key, value in data.items():
                if key != "timestamp":
                    if isinstance(value, dict):
                        summary_lines.append(f"  {key}: {len(value)} items")
                    elif isinstance(value, list):
                        summary_lines.append(f"  {key}: {len(value)} items")
                    else:
                        summary_lines.append(f"  {key}: {value}")
        
        return "\n".join(summary_lines) + "\n"


def create_cli_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Connectome Inspector CLI - Inspect host state through command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s overview                              # Show API overview
  %(prog)s status                                # System status
  %(prog)s spaces                                # All spaces
  %(prog)s agents                                # All agents
  %(prog)s timeline-details --space-id inner_1  # Timeline for space
  %(prog)s veil-facets --space-id inner_1       # VEIL facets for space
  %(prog)s health                                # Health check
        """
    )
    
    parser.add_argument(
        'command',
        choices=[
            'overview', 'status', 'spaces', 'agents', 'adapters', 
            'metrics', 'timelines', 'timeline-details', 'veil', 
            'veil-space', 'veil-facets', 'veil-facet-details', 'health'
        ],
        help='Inspector command to execute'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'compact', 'summary'],
        default='summary',
        help='Output format (default: summary)'
    )
    
    # Command-specific arguments
    parser.add_argument('--space-id', help='Space ID for space-specific commands')
    parser.add_argument('--timeline-id', help='Timeline ID for timeline commands')
    parser.add_argument('--facet-id', help='Facet ID for facet detail commands')
    parser.add_argument('--facet-type', help='Filter facets by type (event, status, ambient)')
    parser.add_argument('--owner-id', help='Filter facets by owner element ID')
    parser.add_argument('--limit', type=int, default=100, help='Limit number of results (default: 100)')
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


async def main_cli(host_instance, args=None):
    """
    Main CLI entry point.
    
    Args:
        host_instance: The Host instance to inspect
        args: CLI arguments (if None, will parse from sys.argv)
    """
    parser = create_cli_parser()
    parsed_args = parser.parse_args(args)
    
    # Set up logging
    if parsed_args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # Create CLI inspector
    cli_inspector = CLIInspector(host_instance)
    
    # Build command kwargs
    kwargs = {}
    if parsed_args.space_id:
        kwargs['space_id'] = parsed_args.space_id
    if parsed_args.timeline_id:
        kwargs['timeline_id'] = parsed_args.timeline_id
    if parsed_args.facet_id:
        kwargs['facet_id'] = parsed_args.facet_id
    if parsed_args.facet_type:
        kwargs['facet_type'] = parsed_args.facet_type
    if parsed_args.owner_id:
        kwargs['owner_id'] = parsed_args.owner_id
    if parsed_args.limit:
        kwargs['limit'] = parsed_args.limit
    
    # Execute command
    try:
        result = await cli_inspector.run_command(parsed_args.command, **kwargs)
        
        # Format and print output
        output = cli_inspector.format_output(result, parsed_args.format)
        print(output)
        
        # Return appropriate exit code
        if "error" in result:
            return 1
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # This would be used for standalone execution
    print("❌ CLI Inspector requires a Host instance to inspect")
    print("Use: python -m host.modules.inspector.cli_inspector <host_instance>")
    sys.exit(1)