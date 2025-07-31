"""
VEILFacet Cache System

Manages flat VEIL facet storage with link resolution and chronological ordering.
Replaces hierarchical VEIL node caches with flat, temporal-first architecture.
"""

import logging
import copy
from typing import Dict, List, Optional, Set
from collections import defaultdict
from .veil_facet import VEILFacet, VEILFacetType
from .facet_operations import VEILFacetOperation

logger = logging.getLogger(__name__)

class VEILFacetCache:
    """
    Manages flat VEIL facet storage with link resolution.

    Core Features:
    - Flat temporal stream storage
    - Chronological ordering by veil_timestamp
    - Many-to-one link resolution
    - Facet type filtering
    - Efficient owner-based queries
    - Ambient facet management

    Key Principles:
    - All facets exist in single flat collection
    - Temporal ordering determines presentation sequence
    - Links provide structure without hierarchy
    - Facet types enable semantic behaviors
    """

    def __init__(self):
        """Initialize the VEILFacetCache."""
        self.facets: Dict[str, VEILFacet] = {}  # facet_id -> VEILFacet
        self._owner_index: Dict[str, Set[str]] = defaultdict(set)  # owner_id -> {facet_ids}
        self._link_index: Dict[str, Set[str]] = defaultdict(set)  # target_id -> {facet_ids linking to it}
        self._type_index: Dict[VEILFacetType, Set[str]] = defaultdict(set)  # facet_type -> {facet_ids}

        logger.debug("VEILFacetCache initialized")

    def add_facet(self, facet: VEILFacet) -> None:
        """
        Add facet to cache with automatic indexing.

        Args:
            facet: VEILFacet to add to cache
        """
        facet_id = facet.facet_id

        # Remove existing facet if present (for updates)
        if facet_id in self.facets:
            self.remove_facet(facet_id)

        # Add to primary storage
        self.facets[facet_id] = facet

        # Update indexes
        self._owner_index[facet.owner_element_id].add(facet_id)
        self._type_index[facet.facet_type].add(facet_id)

        if facet.links_to:
            self._link_index[facet.links_to].add(facet_id)

        logger.debug(f"Added {facet.facet_type.value} facet {facet_id} to cache")

    def remove_facet(self, facet_id: str) -> bool:
        """
        Remove facet from cache and all indexes.

        Args:
            facet_id: ID of facet to remove

        Returns:
            True if facet was removed, False if not found
        """
        if facet_id not in self.facets:
            return False

        facet = self.facets[facet_id]

        # Remove from indexes
        self._owner_index[facet.owner_element_id].discard(facet_id)
        self._type_index[facet.facet_type].discard(facet_id)

        if facet.links_to:
            self._link_index[facet.links_to].discard(facet_id)

        # Remove from primary storage
        del self.facets[facet_id]

        logger.debug(f"Removed facet {facet_id} from cache")
        return True

    def update_facet(self, facet_id: str, property_updates: Dict[str, any]) -> bool:
        """
        Update facet properties.

        Args:
            facet_id: ID of facet to update
            property_updates: Properties to update

        Returns:
            True if facet was updated, False if not found
        """
        if facet_id not in self.facets:
            return False

        facet = self.facets[facet_id]

        # Handle links_to changes specially to maintain indexes
        if "links_to" in property_updates:
            old_links_to = facet.links_to
            new_links_to = property_updates["links_to"]

            # Update link indexes
            if old_links_to:
                self._link_index[old_links_to].discard(facet_id)
            if new_links_to:
                self._link_index[new_links_to].add(facet_id)

            # Update facet
            facet.links_to = new_links_to

        # Update other properties
        remaining_updates = {k: v for k, v in property_updates.items() if k != "links_to"}
        if remaining_updates:
            facet.update_properties(remaining_updates)

        logger.debug(f"Updated facet {facet_id} with {len(property_updates)} properties")
        return True

    def get_facet(self, facet_id: str) -> Optional[VEILFacet]:
        """
        Get facet by ID.

        Args:
            facet_id: ID of facet to retrieve

        Returns:
            VEILFacet instance or None if not found
        """
        facet = self.facets.get(facet_id)
        if not facet:
            logger.error(f"Facet {facet_id} not found in cache")
            raise ValueError(f"Facet {facet_id} not found in cache")
        return facet

    def get_facet_or_none(self, facet_id: str) -> Optional[VEILFacet]:
        return self.facets.get(facet_id)

    def get_linked_facets(self, facet_id: str) -> List[VEILFacet]:
        """
        Get all facets that link to the given facet_id.

        Args:
            facet_id: Target facet ID to find links to

        Returns:
            List of VEILFacet instances that link to the target
        """
        linked_ids = self._link_index.get(facet_id, set())
        return [self.facets[fid] for fid in linked_ids if fid in self.facets]

    def get_facets_by_owner(self, owner_element_id: str) -> List[VEILFacet]:
        """
        Get all facets owned by a specific element.

        Args:
            owner_element_id: Element ID to filter by

        Returns:
            List of VEILFacet instances owned by the element
        """
        facet_ids = self._owner_index.get(owner_element_id, set())
        return [self.facets[fid] for fid in facet_ids if fid in self.facets]

    def get_facets_by_type(self, facet_type: VEILFacetType) -> List[VEILFacet]:
        """
        Get all facets of a specific type.

        Args:
            facet_type: VEILFacetType to filter by

        Returns:
            List of VEILFacet instances of the specified type
        """
        facet_ids = self._type_index.get(facet_type, set())
        return [self.facets[fid] for fid in facet_ids if fid in self.facets]

    def get_chronological_stream(self,
                                 include_ambient: bool = False,
                                 only_synthetic_agent_responses_in_history: bool = False) -> List[VEILFacet]:
        """
        Get all facets in chronological order.

        Args:
            include_ambient: Whether to include ambient facets (default: False)
            only_synthetic_agent_responses_in_history:
                Whether to include only agent synthetic responses in message history
                and exclude corresponding external messages (default: False)
        Returns:
            List of VEILFacet instances sorted by temporal order
        """
        if include_ambient:
            candidates = list(self.facets.values())
        else:
            # Exclude ambient facets for normal temporal streams
            candidates = [f for f in self.facets.values() if f.facet_type != VEILFacetType.AMBIENT]

        sorted_candidates = sorted(candidates, key=lambda f: f.get_temporal_key())

        # NEW: Filter out external messages that have corresponding synthetic responses in the cache
        # as their existence causes issues when turn structure is applied
        if only_synthetic_agent_responses_in_history:
            return self._filter_out_facets_with_synthetic_responses(sorted_candidates)

        return sorted_candidates

    def get_ambient_facets(self) -> List[VEILFacet]:
        """
        Get all ambient facets for floating context rendering.

        Returns:
            List of AmbientFacet instances
        """
        return self.get_facets_by_type(VEILFacetType.AMBIENT)

    def get_latest_status_facet(self) -> Optional[VEILFacet]:
        """
        Get the most recent status facet for ambient triggering.

        Returns:
            Latest StatusFacet by veil_timestamp, or None if no status facets
        """
        status_facets = self.get_facets_by_type(VEILFacetType.STATUS)
        if not status_facets:
            return None

        return max(status_facets, key=lambda f: f.veil_timestamp)

    def apply_operation(self, operation: VEILFacetOperation) -> bool:
        """
        Apply a VEILFacetOperation to the cache.

        Args:
            operation: VEILFacetOperation to apply

        Returns:
            True if operation was applied successfully
        """
        try:
            if operation.operation_type == "add_facet":
                if operation.facet:
                    self.add_facet(operation.facet)
                    return True

            elif operation.operation_type == "update_facet":
                if operation.facet_id and operation.property_updates:
                    return self.update_facet(operation.facet_id, operation.property_updates)

            elif operation.operation_type == "remove_facet":
                if operation.facet_id:
                    return self.remove_facet(operation.facet_id)

            logger.warning(f"Invalid operation parameters: {operation}")
            return False

        except Exception as e:
            logger.error(f"Error applying operation {operation}: {e}", exc_info=True)
            return False

    def apply_operations(self, operations: List[VEILFacetOperation]) -> int:
        """
        Apply multiple VEILFacetOperations to the cache.

        Args:
            operations: List of VEILFacetOperation instances to apply

        Returns:
            Number of operations successfully applied
        """
        success_count = 0
        for operation in operations:
            if self.apply_operation(operation):
                success_count += 1

        logger.debug(f"Applied {success_count}/{len(operations)} operations to cache")
        return success_count

    def get_facet_hierarchy(self, root_facet_id: str, max_depth: int = 10) -> Dict[str, List[VEILFacet]]:
        """
        Build facet hierarchy starting from a root facet.

        Args:
            root_facet_id: ID of root facet to start from
            max_depth: Maximum depth to traverse (prevents infinite loops)

        Returns:
            Dictionary mapping facet_id -> list of child facets
        """
        hierarchy = {}
        visited = set()

        def build_level(facet_id: str, depth: int) -> None:
            if depth >= max_depth or facet_id in visited:
                return

            visited.add(facet_id)
            children = self.get_linked_facets(facet_id)
            hierarchy[facet_id] = children

            for child in children:
                build_level(child.facet_id, depth + 1)

        build_level(root_facet_id, 0)
        return hierarchy

    def get_cache_statistics(self) -> Dict[str, any]:
        """
        Get cache statistics for debugging and monitoring.

        Returns:
            Dictionary with cache statistics
        """
        total_facets = len(self.facets)
        type_counts = {ftype.value: len(facet_ids) for ftype, facet_ids in self._type_index.items()}

        return {
            "total_facets": total_facets,
            "facet_types": type_counts,
            "owners": len(self._owner_index),
            "linked_targets": len(self._link_index),
            "chronological_range": self._get_timestamp_range()
        }

    def _get_timestamp_range(self) -> Optional[Dict[str, float]]:
        """Get timestamp range for temporal facets."""
        temporal_facets = [f for f in self.facets.values() if f.facet_type != VEILFacetType.AMBIENT]
        if not temporal_facets:
            return None

        timestamps = [f.veil_timestamp for f in temporal_facets]
        return {
            "earliest": min(timestamps),
            "latest": max(timestamps),
            "span": max(timestamps) - min(timestamps)
        }

    def _filter_out_facets_with_synthetic_responses(self, facets: List[VEILFacet]) -> List[VEILFacet]:
        """
        Fliter out facets with external messages that have corresponding synthetic responses in the cache.

        Args:
            facets: List of VEILFacet instances to filter

        Returns:
            List of filtered VEILFacet instances
        """
        if not facets:
            return []

        index_of_last_facet = len(facets) - 1
        filtered_facets = [
            facet for i, facet in enumerate(facets)
            if i == index_of_last_facet or f"synthetic_response_{facet.facet_id}" not in facets[i + 1].facet_id
        ]

        return filtered_facets

    def clear(self) -> None:
        """Clear all facets and indexes."""
        self.facets.clear()
        self._owner_index.clear()
        self._link_index.clear()
        self._type_index.clear()
        logger.debug("VEILFacetCache cleared")

    def copy(self) -> 'VEILFacetCache':
        """
        Create a deep copy of the cache.

        Returns:
            New VEILFacetCache instance with copied facets
        """
        new_cache = VEILFacetCache()
        for facet in self.facets.values():
            # Create a copy of the facet with deep-copied properties
            facet_copy = type(facet)(
                facet_id=facet.facet_id,
                veil_timestamp=facet.veil_timestamp,
                owner_element_id=facet.owner_element_id,
                **{k: v for k, v in facet.properties.items() if k not in ['facet_id', 'veil_timestamp', 'owner_element_id']}
            )
            facet_copy.links_to = facet.links_to
            facet_copy.properties = copy.deepcopy(facet.properties)
            new_cache.add_facet(facet_copy)

        return new_cache

    def __len__(self) -> int:
        """Get number of facets in cache."""
        return len(self.facets)

    def __contains__(self, facet_id: str) -> bool:
        """Check if facet exists in cache."""
        return facet_id in self.facets

    def __repr__(self) -> str:
        """String representation for debugging."""
        stats = self.get_cache_statistics()
        return f"VEILFacetCache(facets={stats['total_facets']}, types={stats['facet_types']})"

class TemporalConsistencyValidator:
    """
    Validates temporal consistency in VEILFacet streams.
    Helps detect and diagnose temporal ordering issues.
    """

    @staticmethod
    def validate_temporal_stream(facets: List[VEILFacet]) -> List[str]:
        """
        Validate temporal ordering and detect violations.

        Args:
            facets: List of VEILFacet instances to validate

        Returns:
            List of violation messages (empty if valid)
        """
        violations = []

        # Check chronological ordering for non-ambient facets
        temporal_facets = [f for f in facets if f.facet_type != VEILFacetType.AMBIENT]
        for i in range(1, len(temporal_facets)):
            current = temporal_facets[i]
            previous = temporal_facets[i-1]

            if current.veil_timestamp < previous.veil_timestamp:
                violations.append(
                    f"Temporal violation: {current.facet_id} (t={current.veil_timestamp}) "
                    f"before {previous.facet_id} (t={previous.veil_timestamp})"
                )

        return violations

    @staticmethod
    def validate_link_consistency(cache: VEILFacetCache) -> List[str]:
        """
        Validate link consistency in facet cache.

        Args:
            cache: VEILFacetCache to validate

        Returns:
            List of violation messages (empty if valid)
        """
        violations = []
        facet_ids = set(cache.facets.keys())

        for facet in cache.facets.values():
            if facet.links_to and facet.links_to not in facet_ids:
                violations.append(
                    f"Broken link: {facet.facet_id} links to missing {facet.links_to}"
                )

        return violations

    @staticmethod
    def validate_cache_indexes(cache: VEILFacetCache) -> List[str]:
        """
        Validate cache index consistency.

        Args:
            cache: VEILFacetCache to validate

        Returns:
            List of violation messages (empty if valid)
        """
        violations = []

        # Validate owner index
        for owner_id, facet_ids in cache._owner_index.items():
            for facet_id in facet_ids:
                if facet_id not in cache.facets:
                    violations.append(f"Owner index contains missing facet {facet_id}")
                elif cache.facets[facet_id].owner_element_id != owner_id:
                    violations.append(f"Owner index mismatch for facet {facet_id}")

        # Validate type index
        for facet_type, facet_ids in cache._type_index.items():
            for facet_id in facet_ids:
                if facet_id not in cache.facets:
                    violations.append(f"Type index contains missing facet {facet_id}")
                elif cache.facets[facet_id].facet_type != facet_type:
                    violations.append(f"Type index mismatch for facet {facet_id}")

        return violations