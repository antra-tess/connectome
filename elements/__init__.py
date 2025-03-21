"""
Elements Module
Defines the hierarchical element system for the Bot Framework.
"""

from bot_framework.elements.elements.base import BaseElement
from bot_framework.elements.elements.space import Space
from bot_framework.elements.elements.inner_space import InnerSpace
from bot_framework.elements.space_registry import SpaceRegistry

__all__ = [
    'BaseElement',
    'Space',
    'InnerSpace',
    'SpaceRegistry'
]