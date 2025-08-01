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

__all__ = ['InspectorServer', 'InspectorDataCollector']