"""
Context Manager

The Context Manager is responsible for context assembly for agent interactions.
It works with the HUD to build coherent contexts based on current state and
relevant information, with simple hooks for memory formation.
"""

from .core import ContextManager, ContextRequest, ContextResponse
from .compression import CompressorBase, PriorityCompressor, CategoryCompressor, SimpleCompressor

__all__ = [
    'ContextManager',
    'ContextRequest',
    'ContextResponse',
    'CompressorBase',
    'PriorityCompressor',
    'CategoryCompressor',
    'SimpleCompressor'
] 