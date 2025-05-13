"""
Elements Module
Defines the hierarchical element system for the Bot Framework.
"""

from elements.elements.base import BaseElement
from elements.elements.space import Space
from elements.elements.inner_space import InnerSpace
from elements.space_registry import SpaceRegistry
from .elements.uplink import UplinkProxy

__all__ = [
    'BaseElement',
    'Space',
    'InnerSpace',
    'SpaceRegistry'
]
