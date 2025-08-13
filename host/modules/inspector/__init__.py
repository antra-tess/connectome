"""
Inspector Module

Provides web-based inspection of Connectome host state including:
- Spaces (Inner, Shared, Uplink)
- Elements and Components
- Agent status and configuration  
- Activity adapter connections
- System metrics and health
"""

from .inspector_server import InspectorServer
from .data_collector import InspectorDataCollector
from .endpoint_handlers import InspectorEndpointHandlers
from .cli_inspector import CLIInspector
from .cli_handler import CLICommandHandler, register_cli_commands
from .ipc_server import IPCServer
from .ipc_client import IPCClient, IPCCommandExecutor

__all__ = [
    'InspectorServer', 
    'InspectorDataCollector', 
    'InspectorEndpointHandlers', 
    'CLIInspector',
    'CLICommandHandler',
    'register_cli_commands',
    'IPCServer',
    'IPCClient',
    'IPCCommandExecutor'
]