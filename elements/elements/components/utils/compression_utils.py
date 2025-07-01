"""
Compression Utilities and Data Structures

Data structures and utility functions for the cascade invalidation system
in the CompressionEngineComponent. These support cross-container memory
dependency tracking and cascade invalidation processing.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import asyncio


@dataclass
class MemoryFormationRecord:
    """
    Lightweight record of memory formation for cascade invalidation.
    
    Tracks when a memory was formed and what context existed at that time,
    enabling cascade invalidation when dependency memories are invalidated.
    """
    memory_id: str                    # Unique memory identifier (VEIL ID)
    element_id: str                   # Container element ID where memory lives
    global_formation_index: int       # Global sequential formation order
    formation_timestamp: float        # Unix timestamp of formation
    formation_delta_index: int        # SpaceVeilProducer delta index at formation
    context_memory_ids: List[str]     # Memory IDs that existed during formation
    invalidation_state: 'InvalidationState' = field(default_factory=lambda: InvalidationState.VALID)
    
    # Lightweight metadata for performance
    context_fingerprint: str = ""     # Hash of context_memory_ids for quick comparison
    formation_generation: int = 0     # Generation counter for GC optimization
    
    # Background processing state
    cascade_processed: bool = False   # Whether cascade invalidation was processed
    recompression_scheduled: bool = False  # Whether recompression was scheduled
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "memory_id": self.memory_id,
            "element_id": self.element_id,
            "global_formation_index": self.global_formation_index,
            "formation_timestamp": self.formation_timestamp,
            "formation_delta_index": self.formation_delta_index,
            "context_memory_ids": self.context_memory_ids,
            "invalidation_state": self.invalidation_state.value,
            "context_fingerprint": self.context_fingerprint,
            "formation_generation": self.formation_generation,
            "cascade_processed": self.cascade_processed,
            "recompression_scheduled": self.recompression_scheduled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryFormationRecord':
        """Create from dictionary (for loading from storage)."""
        return cls(
            memory_id=data["memory_id"],
            element_id=data["element_id"],
            global_formation_index=data["global_formation_index"],
            formation_timestamp=data["formation_timestamp"],
            formation_delta_index=data["formation_delta_index"],
            context_memory_ids=data["context_memory_ids"],
            invalidation_state=InvalidationState(data.get("invalidation_state", "valid")),
            context_fingerprint=data.get("context_fingerprint", ""),
            formation_generation=data.get("formation_generation", 0),
            cascade_processed=data.get("cascade_processed", False),
            recompression_scheduled=data.get("recompression_scheduled", False)
        )


class InvalidationState(Enum):
    """State of memory invalidation for cascade processing."""
    VALID = "valid"
    INVALIDATED_LOCAL = "invalidated_local"      # Invalidated due to local content change
    INVALIDATED_CASCADE = "invalidated_cascade"  # Invalidated due to dependency cascade
    RECOMPRESSING = "recompressing"             # Currently being recompressed
    RECOMPRESSED = "recompressed"               # Successfully recompressed
    FAILED = "failed"                           # Recompression failed


@dataclass
class CascadeInvalidationTask:
    """Task for background cascade invalidation processing."""
    trigger_element_id: str           # Element where invalidation originated
    trigger_memory_ids: List[str]     # Memory IDs that triggered the cascade
    trigger_reason: str               # Reason for original invalidation
    cascade_start_index: int          # Global formation index to start cascade from
    priority: int = 0                 # Task priority (0 = highest)
    created_at: float = field(default_factory=time.time)
    max_cascade_depth: int = 50       # Safety limit for cascade depth
    retry_count: int = 0              # Number of retry attempts


@dataclass
class CascadeStatistics:
    """Statistics for cascade invalidation monitoring and optimization."""
    total_invalidations_triggered: int = 0
    total_memories_cascaded: int = 0
    total_containers_affected: int = 0
    max_cascade_depth_seen: int = 0
    avg_cascade_processing_time: float = 0.0
    cascade_queue_max_size: int = 0
    
    # Performance metrics
    timeline_query_times: List[float] = field(default_factory=list)
    cascade_processing_times: List[float] = field(default_factory=list)
    recompression_success_rate: float = 1.0
    
    # Error tracking
    cascade_errors: int = 0
    circular_dependency_detections: int = 0
    cascade_timeouts: int = 0
    
    def update_processing_time(self, processing_time: float) -> None:
        """Update average cascade processing time with new measurement."""
        self.cascade_processing_times.append(processing_time)
        # Keep only last 100 measurements for memory efficiency
        if len(self.cascade_processing_times) > 100:
            self.cascade_processing_times.pop(0)
        
        # Update average
        if self.cascade_processing_times:
            self.avg_cascade_processing_time = sum(self.cascade_processing_times) / len(self.cascade_processing_times)
    
    def update_queue_size(self, queue_size: int) -> None:
        """Update maximum queue size seen."""
        if queue_size > self.cascade_queue_max_size:
            self.cascade_queue_max_size = queue_size
    
    def record_cascade_depth(self, depth: int) -> None:
        """Record cascade depth for statistics."""
        if depth > self.max_cascade_depth_seen:
            self.max_cascade_depth_seen = depth
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for monitoring/logging."""
        return {
            "total_invalidations_triggered": self.total_invalidations_triggered,
            "total_memories_cascaded": self.total_memories_cascaded,
            "total_containers_affected": self.total_containers_affected,
            "max_cascade_depth_seen": self.max_cascade_depth_seen,
            "avg_cascade_processing_time": self.avg_cascade_processing_time,
            "cascade_queue_max_size": self.cascade_queue_max_size,
            "current_timeline_query_times": len(self.timeline_query_times),
            "current_cascade_processing_times": len(self.cascade_processing_times),
            "recompression_success_rate": self.recompression_success_rate,
            "cascade_errors": self.cascade_errors,
            "circular_dependency_detections": self.circular_dependency_detections,
            "cascade_timeouts": self.cascade_timeouts
        }


def calculate_dependency_fingerprint(memory_ids: List[str]) -> str:
    """
    Calculate lightweight fingerprint of dependency set for quick comparison.
    
    Args:
        memory_ids: List of memory IDs to fingerprint
        
    Returns:
        16-character deterministic fingerprint
    """
    try:
        import hashlib
        # Sort for deterministic fingerprint
        sorted_ids = sorted(memory_ids)
        fingerprint_input = "|".join(sorted_ids)
        return hashlib.sha256(fingerprint_input.encode()).hexdigest()[:16]
    except Exception:
        # Fallback fingerprint on error
        return f"error_{len(memory_ids):04x}"


def find_earliest_formation_index(memory_ids: List[str], 
                                memory_id_to_formation_index: Dict[str, int]) -> Optional[int]:
    """
    Find the earliest global formation index for a set of memory IDs.
    
    Args:
        memory_ids: List of memory IDs to check
        memory_id_to_formation_index: Lookup cache for formation indices
        
    Returns:
        Earliest formation index or None if no valid indices found
    """
    earliest_index = None
    
    for memory_id in memory_ids:
        formation_index = memory_id_to_formation_index.get(memory_id)
        if formation_index is not None:
            if earliest_index is None or formation_index < earliest_index:
                earliest_index = formation_index
    
    return earliest_index


def group_memories_by_container(dependent_memories: List[MemoryFormationRecord]) -> Dict[str, List[str]]:
    """
    Group dependent memories by their container element for efficient processing.
    
    Args:
        dependent_memories: List of memory formation records
        
    Returns:
        Dictionary mapping element_id to list of memory_ids
    """
    container_groups = {}
    
    for formation_record in dependent_memories:
        element_id = formation_record.element_id
        memory_id = formation_record.memory_id
        
        if element_id not in container_groups:
            container_groups[element_id] = []
        container_groups[element_id].append(memory_id)
    
    return container_groups


def detect_circular_dependencies(new_formation_record: MemoryFormationRecord,
                               global_memory_timeline: List[MemoryFormationRecord]) -> bool:
    """
    Detect potential circular dependencies before adding to timeline.
    
    Args:
        new_formation_record: New memory formation record to check
        global_memory_timeline: Current global timeline
        
    Returns:
        True if circular dependency detected, False otherwise
    """
    try:
        new_memory_id = new_formation_record.memory_id
        context_memory_ids = set(new_formation_record.context_memory_ids)
        
        # Build dependency graph for cycle detection
        memory_dependencies = {}
        for record in global_memory_timeline:
            memory_dependencies[record.memory_id] = set(record.context_memory_ids)
        
        # Add new record to dependencies
        memory_dependencies[new_memory_id] = context_memory_ids
        
        # Use DFS to detect cycles
        visited = set()
        visiting = set()
        
        def has_cycle(memory_id: str) -> bool:
            if memory_id in visiting:
                return True  # Cycle detected
            if memory_id in visited:
                return False
            
            visiting.add(memory_id)
            
            # Check all dependencies of this memory
            dependencies = memory_dependencies.get(memory_id, set())
            for dep_memory_id in dependencies:
                if dep_memory_id in memory_dependencies:  # Only check memories we know about
                    if has_cycle(dep_memory_id):
                        return True
            
            visiting.remove(memory_id)
            visited.add(memory_id)
            return False
        
        # Check if adding this memory would create a cycle
        return has_cycle(new_memory_id)
        
    except Exception:
        # Conservative: assume circular on error
        return True


class CascadeTimelineManager:
    """
    Utility class for managing the global memory formation timeline.
    
    Provides efficient operations for timeline maintenance, lookups,
    and persistence coordination.
    """
    
    def __init__(self):
        self.timeline: List[MemoryFormationRecord] = []
        self.memory_id_to_formation_index: Dict[str, int] = {}
        self.formation_index_to_record: Dict[int, MemoryFormationRecord] = {}
        self.next_global_formation_index: int = 0
    
    def add_memory_formation(self, record: MemoryFormationRecord) -> bool:
        """
        Add a new memory formation record to the timeline.
        
        Args:
            record: Memory formation record to add
            
        Returns:
            True if successfully added, False if circular dependency detected
        """
        try:
            # Check for circular dependencies
            if detect_circular_dependencies(record, self.timeline):
                return False
            
            # Assign formation index if not set
            if record.global_formation_index < 0:
                record.global_formation_index = self.next_global_formation_index
                self.next_global_formation_index += 1
            
            # Add to timeline and update caches
            self.timeline.append(record)
            self.memory_id_to_formation_index[record.memory_id] = record.global_formation_index
            self.formation_index_to_record[record.global_formation_index] = record
            
            # Update next index if needed
            if record.global_formation_index >= self.next_global_formation_index:
                self.next_global_formation_index = record.global_formation_index + 1
            
            return True
            
        except Exception:
            return False
    
    def find_dependent_memories(self, trigger_memory_ids: List[str], 
                              cascade_start_index: int,
                              max_cascade_depth: int = 50) -> List[MemoryFormationRecord]:
        """
        Find all memories that transitively depend on the triggered memories.
        
        Args:
            trigger_memory_ids: Memory IDs that were invalidated
            cascade_start_index: Formation index to start cascade from
            max_cascade_depth: Maximum cascade depth for safety
            
        Returns:
            List of dependent memory formation records
        """
        try:
            dependent_memories = []
            trigger_memory_set = set(trigger_memory_ids)
            
            # Get candidates after cascade start index
            candidates = [
                record for record in self.timeline 
                if (record.global_formation_index > cascade_start_index and 
                    record.invalidation_state == InvalidationState.VALID)
            ]
            
            # Process in formation order for transitive dependencies
            cascade_depth = 0
            for record in candidates:
                if cascade_depth >= max_cascade_depth:
                    break
                
                # Check if this memory depends on any triggered/cascaded memory
                context_memory_set = set(record.context_memory_ids)
                if context_memory_set.intersection(trigger_memory_set):
                    dependent_memories.append(record)
                    trigger_memory_set.add(record.memory_id)  # Add for transitive dependencies
                    cascade_depth += 1
            
            return dependent_memories
            
        except Exception:
            return []
    
    def mark_memories_as_cascaded(self, dependent_memories: List[MemoryFormationRecord],
                                trigger_memory_ids: List[str]) -> None:
        """Mark memories in formation timeline as cascade-processed."""
        try:
            # Mark dependent memories as cascade invalidated
            for formation_record in dependent_memories:
                formation_record.invalidation_state = InvalidationState.INVALIDATED_CASCADE
                formation_record.cascade_processed = True
            
            # Mark trigger memories as locally invalidated
            for record in self.timeline:
                if record.memory_id in trigger_memory_ids:
                    record.invalidation_state = InvalidationState.INVALIDATED_LOCAL
                    record.cascade_processed = True
                    
        except Exception:
            pass  # Non-critical operation
    
    def rebuild_lookup_caches(self) -> None:
        """Rebuild fast lookup caches from timeline data."""
        try:
            self.memory_id_to_formation_index.clear()
            self.formation_index_to_record.clear()
            
            for record in self.timeline:
                self.memory_id_to_formation_index[record.memory_id] = record.global_formation_index
                self.formation_index_to_record[record.global_formation_index] = record
                
        except Exception:
            pass  # Non-critical operation
    
    def get_timeline_data_for_persistence(self) -> List[Dict[str, Any]]:
        """Get timeline data in format suitable for persistence."""
        return [record.to_dict() for record in self.timeline]
    
    def load_timeline_from_persistence(self, timeline_data: List[Dict[str, Any]]) -> None:
        """Load timeline from persisted data."""
        try:
            self.timeline.clear()
            
            for record_data in timeline_data:
                record = MemoryFormationRecord.from_dict(record_data)
                self.timeline.append(record)
            
            # Rebuild caches and update next index
            self.rebuild_lookup_caches()
            
            if self.timeline:
                max_index = max(record.global_formation_index for record in self.timeline)
                self.next_global_formation_index = max_index + 1
            else:
                self.next_global_formation_index = 0
                
        except Exception:
            # Reset on error
            self.timeline.clear()
            self.memory_id_to_formation_index.clear()
            self.formation_index_to_record.clear()
            self.next_global_formation_index = 0
    
    def optimize_timeline(self, retention_hours: int = 24) -> int:
        """
        Remove old processed records to optimize memory usage.
        
        Args:
            retention_hours: Hours of formation history to retain
            
        Returns:
            Number of records removed
        """
        try:
            cutoff_time = time.time() - (retention_hours * 3600)
            original_size = len(self.timeline)
            
            # Keep records that are recent or still valid
            self.timeline = [
                record for record in self.timeline
                if (record.formation_timestamp > cutoff_time or 
                    record.invalidation_state == InvalidationState.VALID)
            ]
            
            # Rebuild caches after optimization
            self.rebuild_lookup_caches()
            
            removed_count = original_size - len(self.timeline)
            return removed_count
            
        except Exception:
            return 0
