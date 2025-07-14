"""
VEIL Facet System - Core Interfaces and Types

This module implements the VEILFacet interface system for endotemporal flat architecture,
replacing the hierarchical VEIL node system with temporal facets.
"""

from .veil_facet import VEILFacet, VEILFacetType
from .facet_types import EventFacet, StatusFacet, AmbientFacet
from .temporal_system import ConnectomeEpoch
from .facet_cache import VEILFacetCache, TemporalConsistencyValidator
from .facet_operations import VEILFacetOperation, FacetOperationBuilder

__all__ = [
    'VEILFacet',
    'VEILFacetType', 
    'EventFacet',
    'StatusFacet',
    'AmbientFacet',
    'ConnectomeEpoch',
    'VEILFacetCache',
    'VEILFacetOperation',
    'FacetOperationBuilder',
    'TemporalConsistencyValidator'
] 