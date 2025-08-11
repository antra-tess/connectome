"""
VEILFacet Compression Engine Component

Replaces CompressionEngineComponent with VEILFacet architecture support.
Implements per-container memorization, focus-dependent rendering, and memory EventFacet integration.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from pathlib import Path

from opentelemetry import trace
from host.observability import get_tracer

from .base_component import Component
from elements.component_registry import register_component
from elements.elements.components.veil.veil_facet import VEILFacet
from elements.elements.components.veil.facet_types import VEILFacetType, EventFacet, StatusFacet, AmbientFacet
from elements.elements.components.veil.facet_cache import VEILFacetCache
from elements.elements.components.veil.temporal_system import ConnectomeEpoch
from .memory_compressor_interface import MemoryCompressor, estimate_veil_tokens

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

class IncrementalRollingWindow:
    """
    Manages rolling window with incremental chunk building for VEILFacet compression.
    
    This class determines which chunks are within the active window and which chunks
    are outside the window and need memories for efficient rendering.
    
    Key insight: Last chunk is always incomplete, so we need X-1 complete 
    chunks + 1 incomplete chunk in window.
    """
    
    def __init__(self, container_id: str):
        self.container_id = container_id
        self.window_size_focused = 8      # 8 complete chunks when focused
        self.window_size_unfocused = 2    # 2 complete chunks when unfocused  
        self.compression_chunk_size = 4000 # Trigger memory at 4k tokens
        
    def get_compression_boundary(self, n_chunks: List[Dict], is_focused: bool) -> int:
        """
        Determine which chunks are outside the window and need memories.
        
        Args:
            n_chunks: List of N-chunk structures
            is_focused: Whether container is currently focused
            
        Returns:
            Compression boundary index. Chunks before this index are outside 
            window and need memories for rendering.
        """
        total_chunks = len(n_chunks)
        if total_chunks == 0:
            return -1
            
        window_size = self.window_size_focused if is_focused else self.window_size_unfocused
        
        # Chunks before this boundary are outside window and need memories
        # We keep window_size complete chunks + 1 incomplete chunk in window
        compression_boundary = max(0, total_chunks - window_size - 1)
        
        logger.debug(f"Container {self.container_id}: {total_chunks} chunks, "
                    f"{'focused' if is_focused else 'unfocused'} window size {window_size}, "
                    f"compression boundary at index {compression_boundary}")
        
        return compression_boundary
        
    def get_chunks_needing_memories(self, n_chunks: List[Dict], is_focused: bool) -> List[int]:
        """
        Get list of chunk indices that are outside window and need memories.
        
        Args:
            n_chunks: List of N-chunk structures
            is_focused: Whether container is currently focused
            
        Returns:
            List of chunk indices that need memories for rendering
        """
        compression_boundary = self.get_compression_boundary(n_chunks, is_focused)
        
        if compression_boundary <= 0:
            return []
            
        # Return indices of chunks that are outside window
        chunks_needing_memories = list(range(compression_boundary))
        
        logger.debug(f"Container {self.container_id}: chunks {chunks_needing_memories} need memories")
        return chunks_needing_memories
        
    def is_chunk_in_window(self, chunk_index: int, n_chunks: List[Dict], is_focused: bool) -> bool:
        """
        Check if a specific chunk is within the active window.
        
        Args:
            chunk_index: Index of chunk to check
            n_chunks: List of N-chunk structures  
            is_focused: Whether container is currently focused
            
        Returns:
            True if chunk is within active window, False if it needs memory
        """
        compression_boundary = self.get_compression_boundary(n_chunks, is_focused)
        return chunk_index >= compression_boundary


@register_component
class VEILFacetCompressionEngine(Component):
    """
    VEILFacet-native compression engine with incremental per-chunk memorization.
    
    NEW ARCHITECTURE: Incremental chunk-based compression
    - Always assigns EventFacets to chunks immediately upon arrival
    - Compresses individual chunks when they exceed 4k tokens  
    - Tracks N-chunk → M-chunk mappings independently per chunk
    - No container-level compression decisions - pure incremental processing
    
    Core Features:
    - StatusFacet-based container identification
    - EventFacet grouping by links_to relationships  
    - Incremental chunking with dynamic N-chunk creation
    - Per-chunk compression when chunks exceed 4k tokens
    - Focus-dependent rendering (raw-first vs compressed-first)
    - Memory EventFacet creation with mean timestamps
    - AgentMemoryCompressor integration for LLM-based memory formation
    
    Architecture:
    - Containers = StatusFacets with EventFacets linking to them
    - Incremental N-chunk creation as EventFacets arrive
    - Individual chunk compression when chunk exceeds 4k tokens
    - Per-chunk N-chunk → M-chunk mapping tracking
    - Focus rules: focused=raw-first, unfocused=compressed-first
    - Memory EventFacets positioned at mean timestamp of replaced content
    """
    
    COMPONENT_TYPE = "VEILFacetCompressionEngine"
    HANDLED_EVENT_TYPES = ["compression_requested", "memory_formation_completed"]
    
    # Incremental compression architecture - chunk-based processing
    COMPRESSION_CHUNK_SIZE = 4000        # 4k tokens per N-chunk (individual chunk compression threshold)
    FOCUSED_CHUNK_LIMIT = 8              # 8 completed N-chunks + 1 incomplete for focused containers (9 total)
    UNFOCUSED_RECENT_CHUNKS = 2          # 2 recent N-chunks for unfocused containers (last completed + incomplete)
    MIN_COMPRESSION_THRESHOLD = COMPRESSION_CHUNK_SIZE     # Chunk-level compression threshold (4k tokens per chunk)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Per-container N-chunk/M-chunk storage
        self._container_chunks: Dict[str, Dict[str, Any]] = {}
        # Structure: {container_id: {"n_chunks": [...], "m_chunks": [...], "last_update": datetime, "total_tokens": int}}
        
        # VEILFacet-aware memory compressor (will be AgentMemoryCompressor)
        self._memory_compressor = None
        
        # Agent tracking for memory formation
        self._agent_name: Optional[str] = None
        
        logger.info(f"VEILFacetCompressionEngine initialized ({self.id}) - ready for per-container memorization")
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the VEILFacet compression engine."""
        if not super().initialize(**kwargs):
            return False
        
        # Set agent name from owner space
        if self.owner and hasattr(self.owner, 'agent_name'):
            self._agent_name = self.owner.agent_name
        elif self.owner and hasattr(self.owner, 'id'):
            self._agent_name = f"agent_{self.owner.id}"
        else:
            self._agent_name = f"agent_{self.id}"
        
        # Mark that we need async initialization
        self._needs_async_init = True
        
        logger.info(f"VEILFacetCompressionEngine initialized for agent {self._agent_name}")
        return True
    
    async def _ensure_async_initialized(self) -> None:
        """Ensure async initialization has been completed."""
        if not hasattr(self, '_needs_async_init') or not self._needs_async_init:
            return
            
        # Initialize AgentMemoryCompressor for memory formation
        await self._initialize_memory_compressor()
        
        self._needs_async_init = False
        logger.debug(f"Async initialization completed for {self._agent_name}")
    
    async def _initialize_memory_compressor(self) -> bool:
        """Initialize AgentMemoryCompressor for VEILFacet memory formation."""
        try:
            from .agent_memory_compressor import AgentMemoryCompressor
            
            # Get LLM provider for agent reflection
            llm_provider = self._get_llm_provider()
            if not llm_provider:
                logger.warning(f"LLM provider not available for AgentMemoryCompressor - will use fallback compression")
                return False
            
            # Create agent-scoped memory compressor
            agent_id = self._agent_name or self.id
            self._memory_compressor = AgentMemoryCompressor(
                agent_id=agent_id,
                token_limit=self.COMPRESSION_CHUNK_SIZE,  # 4k token limit per memory
                storage_base_path="storage_data/memory_storage",
                llm_provider=llm_provider
            )
            
            logger.info(f"AgentMemoryCompressor initialized for VEILFacet compression with agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing AgentMemoryCompressor: {e}", exc_info=True)
            return False
    
    def _get_llm_provider(self):
        """Get LLM provider from InnerSpace for memory formation."""
        try:
            if self.owner and hasattr(self.owner, '_llm_provider'):
                return self.owner._llm_provider
            elif self.owner and hasattr(self.owner, 'get_llm_provider'):
                return self.owner.get_llm_provider()
            else:
                logger.warning(f"Cannot access LLM provider from InnerSpace for memory compression")
                return None
        except Exception as e:
            logger.warning(f"Error accessing LLM provider: {e}")
            return None
    
    # --- MAIN INTERFACE METHODS ---
    
    async def process_facet_cache_with_compression(self,
                                                 facet_cache: VEILFacetCache,
                                                 focus_info: Optional[Dict[str, Any]] = None,
        ) -> VEILFacetCache:
        """
        Main interface: Process VEILFacetCache with per-container memorization.
        
        This is called by HUD to get compressed VEILFacetCache for rendering.
        
        UPDATED: Now accepts focus_info extracted by HUD instead of detecting focus internally.
        
        Args:
            facet_cache: VEILFacetCache from SpaceVeilProducer
            focus_info: Focus information extracted by HUD from latest focus_changed StatusFacet
            
        Returns:
            Processed VEILFacetCache with memory EventFacets replacing compressed content
        """
        try:
            # Ensure async initialization is completed
            await self._ensure_async_initialized()
            with tracer.start_as_current_span("veil_facet_compression.process_cache") as span:
                focus_element_id = focus_info.get('focus_element_id') if focus_info else None
                span.set_attribute("compression.focus_element_id", focus_element_id or "none")
                span.set_attribute("compression.focus_source", focus_info.get('focus_source', 'none') if focus_info else 'none')
                
                logger.debug(f"Processing VEILFacetCache with {len(facet_cache.facets)} facets, focus: {focus_element_id} (source: {focus_info.get('focus_source', 'none') if focus_info else 'none'})")
                
                # Analyze containers from VEILFacetCache (now filters out root/scratchpad)  
                containers = self._identify_containers_from_cache(facet_cache)
                span.set_attribute("compression.containers_found", len(containers))
                
                # Process each container according to compression rules
                processed_cache = VEILFacetCache()
                
                # Add all non-EventFacets first (StatusFacets, AmbientFacets)
                for facet in facet_cache.facets.values():
                    if facet.facet_type != VEILFacetType.EVENT:
                        processed_cache.add_facet(facet)
                
                # Process each container with compression logic
                for container_id, event_facets in containers.items():
                    span.add_event(f"Processing container {container_id}", {
                        "event_facets_count": len(event_facets)
                    })
                    
                    # Determine if this container is focused (gets 8-chunk treatment)
                    is_focused_container = (focus_element_id == container_id) if focus_element_id else False
                    
                    # Process with N-chunk/M-chunk logic
                    processed_facets = await self._process_container_with_memorization(
                        container_id, event_facets, is_focused_container, facet_cache
                    )
                    
                    # Add processed facets to output cache
                    for facet in processed_facets:
                        processed_cache.add_facet(facet)
                
                # FIXED: Add EventFacets that don't belong to any container (like agent responses)
                containerized_facet_ids = set()
                for event_facets in containers.values():
                    for facet in event_facets:
                        containerized_facet_ids.add(facet.facet_id)
                
                # Add uncontainerized EventFacets (agent responses, etc.)
                for facet in facet_cache.facets.values():
                    if (facet.facet_type == VEILFacetType.EVENT and 
                        facet.facet_id not in containerized_facet_ids):
                        processed_cache.add_facet(facet)
                        logger.debug(f"Added uncontainerized EventFacet: {facet.facet_id} (type: {facet.get_property('event_type', 'unknown')})")
                
                logger.info(f"VEILFacet compression complete: {len(facet_cache.facets)} → {len(processed_cache.facets)} facets")
                span.set_attribute("compression.output_facets", len(processed_cache.facets))
                return processed_cache
                
        except Exception as e:
            logger.error(f"Error processing VEILFacetCache with compression: {e}", exc_info=True)
            return facet_cache  # Fallback to unprocessed cache
    
    # --- CONTAINER ANALYSIS ---
    
    def _identify_containers_from_cache(self, facet_cache: VEILFacetCache) -> Dict[str, List[EventFacet]]:
        """
        Identify containers from VEILFacetCache using StatusFacet analysis.
        
        Containers = StatusFacets that have EventFacets linking to them.
        
        UPDATED: Only returns chat-related containers, filters out root/scratchpad containers
        which should not be compressed.
        
        Returns:
            Dictionary mapping container_id to list of EventFacets linked to it
        """
        try:
            # Find StatusFacets that serve as containers
            status_facets = [f for f in facet_cache.facets.values() 
                           if f.facet_type == VEILFacetType.STATUS]
            
            containers = {}
            
            for status_facet in status_facets:
                container_id = status_facet.facet_id
                
                # FILTER: Skip root and scratchpad containers (they should not be compressed)
                if self._is_root_container(container_id):
                    logger.debug(f"Skipping root container {container_id} from compression")
                    continue
                    
                if self._is_scratchpad_container(container_id):
                    logger.debug(f"Skipping scratchpad container {container_id} from compression")
                    continue
                
                # Find EventFacets linking to this StatusFacet
                linked_events = [f for f in facet_cache.facets.values() 
                               if f.facet_type == VEILFacetType.EVENT 
                               and f.links_to == status_facet.facet_id]
                
                if linked_events:  # Only containers with EventFacets
                    # ADDITIONAL FILTER: Skip containers that only have agent responses
                    # (agent responses should not be compressed)
                    non_agent_events = [f for f in linked_events 
                                      if f.get_property("event_type") != "agent_response"]
                    
                    if non_agent_events:  # Only include if there are non-agent events to compress
                        # Sort chronologically by veil_timestamp
                        linked_events.sort(key=lambda f: f.get_temporal_key())
                        containers[container_id] = linked_events
                        
                        logger.debug(f"Found compressible container {container_id} with {len(linked_events)} EventFacets ({len(non_agent_events)} non-agent)")
                    else:
                        logger.debug(f"Skipping container {container_id} - only contains agent responses")
            
            logger.info(f"Identified {len(containers)} compressible containers from {len(status_facets)} StatusFacets (filtered out root/scratchpad)")
            return containers
            
        except Exception as e:
            logger.error(f"Error identifying containers from VEILFacetCache: {e}", exc_info=True)
            return {}

    def _determine_container_rules(self, container_id: str) -> Dict[str, Any]:
        """
        Determine compression and rendering rules for a container.
        
        Args:
            container_id: ID of the container to analyze
            
        Returns:
            Dictionary with compression rules and limits
        """
        try:
            # Check container type patterns
            is_scratchpad = self._is_scratchpad_container(container_id)
            is_root = self._is_root_container(container_id)
            
            if is_scratchpad:
                return {
                    "enable_memorization": False,
                    "reason": "scratchpad_preservation",
                    "n_chunk_limit": float('inf'),  # Preserve all
                    "m_chunk_limit": 0              # No memories
                }
            elif is_root:
                return {
                    "enable_memorization": False,
                    "reason": "root_context_preservation", 
                    "n_chunk_limit": float('inf'),  # Preserve all
                    "m_chunk_limit": 0              # No memories
                }
            else:
                # Content containers (chat, messaging, etc.)
                return {
                    "enable_memorization": True,
                    "reason": "content_container",
                    "n_chunk_limit": self.FOCUSED_CHUNK_LIMIT,  # Will be adjusted by focus
                    "m_chunk_limit": float('inf')               # All memories
                }
                
        except Exception as e:
            logger.error(f"Error determining container rules: {e}", exc_info=True)
            return {
                "enable_memorization": False,
                "reason": "error_fallback",
                "n_chunk_limit": float('inf'),
                "m_chunk_limit": 0
            }
    
    def _is_scratchpad_container(self, container_id: str) -> bool:
        """Check if container is a scratchpad (preserve completely)."""
        scratchpad_patterns = ["scratchpad", "scratch", "notes", "working"]
        container_lower = container_id.lower()
        return any(pattern in container_lower for pattern in scratchpad_patterns)
    
    def _is_root_container(self, container_id: str) -> bool:
        """Check if container is root/space level (preserve completely).""" 
        root_patterns = ["root", "space", "main", "_base"]
        container_lower = container_id.lower()
        return any(pattern in container_lower for pattern in root_patterns)
    
    # --- PER-CONTAINER COMPRESSION PROCESSING ---
    
    async def _process_container_with_memorization(self,
                                                 container_id: str,
                                                 event_facets: List[EventFacet],
                                                 is_focused_container: bool,
                                                 full_facet_cache: VEILFacetCache) -> List[VEILFacet]:
        """
        Process a container with rolling window memorization logic.
        
        ROLLING WINDOW ARCHITECTURE: 
        - Triggers memory formation immediately when chunks complete (4k tokens)
        - Uses compressed context for memory formation (same view agent will see)
        - Memory creation happens in background via AgentMemoryCompressor
        - Rendering checks memory states and falls back to raw chunks when needed
        - Ensures consistent context between memory formation and runtime
        
        Args:
            container_id: ID of container being processed
            event_facets: EventFacets linked to this container
            is_focused_container: Whether this container is focused (affects window size)
            full_facet_cache: Complete VEILFacetCache for compressed context building
            
        Returns:
            List of processed VEILFacets using rolling window memory management
        """
        try:
            logger.debug(f"Processing container {container_id} with {len(event_facets)} EventFacets (rolling window)")
            
            # ROLLING WINDOW: Get or create chunk structure  
            chunk_structure = await self._get_or_create_container_chunks(container_id, event_facets)
            
            # Log token distribution for debugging
            memory_facets, regular_facets = self._separate_memory_and_regular_facets(event_facets)
            regular_tokens = self._calculate_event_facets_tokens(regular_facets)
            total_tokens = self._calculate_event_facets_tokens(event_facets)
            
            logger.debug(f"Container {container_id}: {regular_tokens} regular tokens, {total_tokens - regular_tokens} memory tokens, {total_tokens} total (rolling window)")
            
            # ROLLING WINDOW: Detect new facets and update with immediate memory formation
            new_facets = await self._detect_new_event_facets(chunk_structure, event_facets)
            if new_facets:
                await self._update_chunks_with_new_facets(chunk_structure, new_facets, full_facet_cache)
            
            # Render container according to focus rules
            rendered_facets = await self._render_container_with_focus_rules(
                container_id, chunk_structure, is_focused_container
            )
            
            logger.debug(f"Container {container_id} processed: {len(event_facets)} → {len(rendered_facets)} facets (rolling window)")
            return rendered_facets
            
        except Exception as e:
            logger.error(f"Error processing container {container_id} with memorization: {e}", exc_info=True)
            return event_facets  # Fallback to original facets
    
    async def _get_or_create_container_chunks(self, container_id: str, event_facets: List[EventFacet]) -> Dict[str, Any]:
        """
        Get or create N-chunk/M-chunk structure for a container.
        
        Args:
            container_id: ID of the container
            event_facets: Current EventFacets in container
            
        Returns:
            Chunk structure with N-chunks and M-chunks
        """
        try:
            if container_id not in self._container_chunks:
                # Initialize new chunk structure
                logger.debug(f"Initializing chunk structure for container {container_id}")
                chunk_structure = await self._initialize_chunks_from_event_facets(container_id, event_facets)
                self._container_chunks[container_id] = chunk_structure
            else:
                # Update existing structure with new EventFacets
                chunk_structure = self._container_chunks[container_id]
                # Note: full_facet_cache is not available here, will be passed when needed
            
            return self._container_chunks[container_id]
            
        except Exception as e:
            logger.error(f"Error getting container chunks for {container_id}: {e}", exc_info=True)
            # Fallback: create minimal structure
            return {
                "n_chunks": [],
                "m_chunks": [],
                "last_update": datetime.now(),
                "total_tokens": 0
            }
    
    async def _initialize_chunks_from_event_facets(self, container_id: str, event_facets: List[EventFacet]) -> Dict[str, Any]:
        """
        Initialize N-stream and M-stream from current EventFacets.
        
        Separates existing memory EventFacets from regular EventFacets,
        then organizes everything into the dual-stream structure.
        
        Args:
            container_id: ID of the container
            event_facets: Current EventFacets in container
            
        Returns:
            Initialized chunk structure
        """
        try:
            logger.debug(f"Initializing chunks for container {container_id} with {len(event_facets)} EventFacets")
            
            # Separate memory EventFacets from regular EventFacets
            memory_facets, regular_facets = self._separate_memory_and_regular_facets(event_facets)
            
            logger.debug(f"Container {container_id}: {len(memory_facets)} memory facets, {len(regular_facets)} regular facets")
            
            # Convert regular EventFacets into N-chunks (4k token boundaries)
            n_chunks = await self._event_facets_to_n_chunks(regular_facets)
            
            # Convert memory EventFacets into M-chunks
            m_chunks = self._memory_facets_to_m_chunks(memory_facets)
            
            # Calculate total tokens
            total_tokens = sum(self._calculate_chunk_tokens(chunk) for chunk in n_chunks)
            total_tokens += sum(self._calculate_chunk_tokens(chunk) for chunk in m_chunks)
            
            chunk_structure = {
                "n_chunks": n_chunks,
                "m_chunks": m_chunks,
                "last_update": datetime.now(),
                "total_tokens": total_tokens,
                "container_id": container_id,
                "content_fingerprint": self._calculate_content_fingerprint(regular_facets),  # Container-level for overall tracking
                "invalidation_log": []  # Track invalidation history
            }
            
            logger.debug(f"Initialized {len(n_chunks)} N-chunks, {len(m_chunks)} M-chunks for {container_id}")
            return chunk_structure
            
        except Exception as e:
            logger.error(f"Error initializing chunks for {container_id}: {e}", exc_info=True)
            return {
                "n_chunks": [],
                "m_chunks": [],
                "last_update": datetime.now(),
                "total_tokens": 0,
                "container_id": container_id
            }
    
    def _separate_memory_and_regular_facets(self, event_facets: List[EventFacet]) -> tuple[List[EventFacet], List[EventFacet]]:
        """
        Separate memory EventFacets from regular EventFacets.
        
        Memory EventFacets have event_type="compressed_memory" and is_compressed_memory=True.
        
        Args:
            event_facets: List of EventFacets to separate
            
        Returns:
            Tuple of (memory_facets, regular_facets)
        """
        memory_facets = []
        regular_facets = []
        
        for facet in event_facets:
            if (facet.properties.get("event_type") == "compressed_memory" or 
                facet.properties.get("is_compressed_memory", False)):
                memory_facets.append(facet)
            else:
                regular_facets.append(facet)
        
        return memory_facets, regular_facets
    
    async def _event_facets_to_n_chunks(self, event_facets: List[EventFacet]) -> List[Dict[str, Any]]:
        """
        Convert EventFacets into N-chunks with overflow-friendly chunking algorithm.
        
        UNIFIED ALGORITHM: Uses same overflow logic as incremental chunking to ensure
        consistent chunk boundaries regardless of real-time vs restart processing.
        
        ENHANCED: Includes per-chunk fingerprinting and temporal range tracking
        for surgical invalidation support.
        
        Args:
            event_facets: List of regular EventFacets to chunk
            
        Returns:
            List of N-chunk structures with enhanced metadata and consistent boundaries
        """
        try:
            if not event_facets:
                return []
            
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for facet in event_facets:
                # Calculate tokens for this facet
                facet_tokens = self._calculate_event_facet_tokens(facet)
                
                # UNIFIED LOGIC: Use same overflow-friendly algorithm as incremental chunking
                # Check if current chunk is already at/over limit (not would-exceed)
                if current_tokens >= self.COMPRESSION_CHUNK_SIZE and current_chunk:
                    # Current chunk is already at/over limit, complete it and start new one
                    chunk_fingerprint = self._calculate_content_fingerprint(current_chunk)
                    temporal_range = self._calculate_temporal_range(current_chunk)
                    
                    chunks.append({
                        "chunk_type": "n_chunk",
                        "event_facets": current_chunk,
                        "content_fingerprint": chunk_fingerprint,  # NEW: Per-chunk fingerprint
                        "token_count": current_tokens,
                        "chunk_index": len(chunks),
                        "created_at": datetime.now(),
                        "is_complete": True,
                        "has_memory_chunk": False,  # Will be preserved across surgical invalidations
                        "temporal_range": temporal_range  # NEW: Track chunk boundaries
                    })
                    
                    # Start new chunk
                    current_chunk = [facet]
                    current_tokens = facet_tokens
                else:
                    # Add to current chunk (allow overflow - compression will handle it)
                    current_chunk.append(facet)
                    current_tokens += facet_tokens
                    
                    # Debug: Show unified overflow behavior
                    if current_tokens >= self.COMPRESSION_CHUNK_SIZE:
                        logger.debug(f"Unified chunking: Chunk {len(chunks)} now has {current_tokens} tokens (overflow allowed for compression)")
            
            # Handle remaining facets (possibly incomplete chunk)
            if current_chunk:
                chunk_fingerprint = self._calculate_content_fingerprint(current_chunk)
                temporal_range = self._calculate_temporal_range(current_chunk)
                
                chunks.append({
                    "chunk_type": "n_chunk",
                    "event_facets": current_chunk,
                    "content_fingerprint": chunk_fingerprint,  # NEW: Per-chunk fingerprint
                    "token_count": current_tokens,
                    "chunk_index": len(chunks),
                    "created_at": datetime.now(),
                    "is_complete": current_tokens >= self.COMPRESSION_CHUNK_SIZE,
                    "has_memory_chunk": False,  # Will be preserved across surgical invalidations
                    "temporal_range": temporal_range  # NEW: Track chunk boundaries
                })
            
            logger.debug(f"Created {len(chunks)} N-chunks from {len(event_facets)} EventFacets with enhanced metadata")
            return chunks
            
        except Exception as e:
            logger.error(f"Error converting EventFacets to N-chunks: {e}", exc_info=True)
            return []

    def _calculate_temporal_range(self, facets: List[EventFacet]) -> Dict[str, float]:
        """
        Calculate temporal boundaries for a chunk.
        
        Args:
            facets: List of EventFacets to calculate range for
            
        Returns:
            Temporal range information
        """
        try:
            if not facets:
                return {"start_timestamp": 0, "end_timestamp": 0, "span_seconds": 0, "facet_count": 0}
            
            timestamps = [f.veil_timestamp for f in facets]
            return {
                "start_timestamp": min(timestamps),
                "end_timestamp": max(timestamps),
                "span_seconds": max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0,
                "facet_count": len(facets)
            }
        except Exception as e:
            logger.error(f"Error calculating temporal range: {e}", exc_info=True)
            return {"start_timestamp": 0, "end_timestamp": 0, "span_seconds": 0, "facet_count": 0}

    async def _detect_chunk_level_changes(self, container_id: str, 
                                         new_facets: List[EventFacet], 
                                         chunk_structure: Dict[str, Any]) -> List[int]:
        """
        Detect which specific chunks are affected by new/changed facets.
        
        Uses temporal range checking and facet ID cross-reference to identify
        only the chunks that actually need invalidation and rebuilding.
        
        Args:
            container_id: ID of container being analyzed
            new_facets: New/changed EventFacets
            chunk_structure: Current chunk structure
            
        Returns:
            List of chunk indices that need invalidation
        """
        try:
            changed_chunk_indices = []
            n_chunks = chunk_structure.get("n_chunks", [])
            
            if not n_chunks or not new_facets:
                logger.debug(f"No chunks or facets to analyze for {container_id}")
                return []
            
            logger.debug(f"Analyzing {len(n_chunks)} chunks for changes from {len(new_facets)} new facets")
            
            for chunk_index, chunk in enumerate(n_chunks):
                # Method 1: Temporal Range Check
                chunk_temporal_range = chunk.get("temporal_range", {})
                start_time = chunk_temporal_range.get("start_timestamp", 0)
                end_time = chunk_temporal_range.get("end_timestamp", float('inf'))
                
                # Check if any new facets fall within this chunk's time range
                affects_this_chunk = any(
                    start_time <= facet.veil_timestamp <= end_time 
                    for facet in new_facets
                )
                
                # Method 2: Facet ID Cross-Reference
                if not affects_this_chunk:
                    chunk_facet_ids = {f.facet_id for f in chunk.get("event_facets", [])}
                    new_facet_ids = {f.facet_id for f in new_facets}
                    affects_this_chunk = bool(chunk_facet_ids.intersection(new_facet_ids))
                
                if affects_this_chunk:
                    # Recalculate fingerprint for this specific chunk
                    chunk_facets = chunk.get("event_facets", [])
                    current_fingerprint = self._calculate_content_fingerprint(chunk_facets)
                    stored_fingerprint = chunk.get("content_fingerprint", "")
                    
                    if current_fingerprint != stored_fingerprint:
                        changed_chunk_indices.append(chunk_index)
                        logger.info(f"Chunk {chunk_index} fingerprint changed: {stored_fingerprint[:16]}... → {current_fingerprint[:16]}...")
                    else:
                        logger.debug(f"Chunk {chunk_index} affected but fingerprint unchanged")
            
            logger.info(f"Detected {len(changed_chunk_indices)} changed chunks for {container_id}: {changed_chunk_indices}")
            return changed_chunk_indices
            
        except Exception as e:
            logger.error(f"Error detecting chunk-level changes for {container_id}: {e}", exc_info=True)
            # Fallback: assume all chunks changed to be safe
            return list(range(len(chunk_structure.get("n_chunks", []))))

    async def _invalidate_specific_chunks(self, container_id: str, 
                                        changed_chunk_indices: List[int],
                                        chunk_structure: Dict[str, Any]) -> None:
        """
        Invalidate only specific chunks, preserving others completely.
        
        This is the core surgical invalidation method that maintains compression
        state for unaffected chunks while marking only changed chunks for rebuilding.
        
        Args:
            container_id: ID of container being processed
            changed_chunk_indices: List of chunk indices to invalidate
            chunk_structure: Current chunk structure
        """
        try:
            n_chunks = chunk_structure.get("n_chunks", [])
            m_chunks = chunk_structure.get("m_chunks", [])
            
            logger.info(f"Surgically invalidating {len(changed_chunk_indices)} chunks for {container_id}")
            
            for chunk_index in changed_chunk_indices:
                if chunk_index >= len(n_chunks):
                    logger.warning(f"Chunk index {chunk_index} out of range for {container_id}")
                    continue
                    
                chunk = n_chunks[chunk_index]
                
                # If chunk has a memory, invalidate the corresponding M-chunk
                if chunk.get("has_memory_chunk", False):
                    memory_chunk_index = chunk.get("memory_chunk_index")
                    if memory_chunk_index is not None and memory_chunk_index < len(m_chunks):
                        m_chunks[memory_chunk_index]["is_invalid"] = True
                        m_chunks[memory_chunk_index]["invalidated_at"] = datetime.now().isoformat()
                        m_chunks[memory_chunk_index]["invalidation_reason"] = "content_change_surgical"
                        
                        logger.info(f"Invalidated M-chunk {memory_chunk_index} for N-chunk {chunk_index} in {container_id}")
                
                # Clear compression flag for this chunk only (will be recompressed if needed)
                chunk["has_memory_chunk"] = False
                chunk["invalidated_at"] = datetime.now().isoformat()
                
                # Log invalidation for debugging
                invalidation_log = chunk_structure.setdefault("invalidation_log", [])
                invalidation_log.append({
                    "chunk_index": chunk_index,
                    "timestamp": datetime.now().isoformat(),
                    "reason": "content_change",
                    "fingerprint_changed": True
                })
                
                logger.debug(f"Invalidated N-chunk {chunk_index} in {container_id} (surgical)")
            
            # Preserve all other chunks' compression state
            preserved_count = len(n_chunks) - len(changed_chunk_indices)
            logger.info(f"Preserved compression state for {preserved_count} unaffected chunks in {container_id}")
            
        except Exception as e:
            logger.error(f"Error in surgical chunk invalidation for {container_id}: {e}", exc_info=True)

    async def _rebuild_affected_chunks(self, container_id: str,
                                     current_regular_facets: List[EventFacet],
                                     changed_chunk_indices: List[int], 
                                     chunk_structure: Dict[str, Any]) -> None:
        """
        Rebuild only the chunks that were affected, preserve others.
        
        This method performs precision rebuilding by only reconstructing chunks
        that actually changed, while preserving the structure and state of all
        unaffected chunks.
        
        Args:
            container_id: ID of container being processed
            current_regular_facets: Current regular EventFacets in container
            changed_chunk_indices: List of chunk indices to rebuild
            chunk_structure: Current chunk structure
        """
        try:
            n_chunks = chunk_structure.get("n_chunks", [])
            
            logger.info(f"Precision rebuilding {len(changed_chunk_indices)} affected chunks for {container_id}")
            
            # Group facets by which chunk they should belong to
            chunk_facet_mapping = self._map_facets_to_chunk_indices(
                current_regular_facets, chunk_structure
            )
            
            for chunk_index in changed_chunk_indices:
                if chunk_index >= len(n_chunks):
                    logger.warning(f"Chunk index {chunk_index} out of range for {container_id}")
                    continue
                    
                # Get facets that belong to this chunk
                chunk_facets = chunk_facet_mapping.get(chunk_index, [])
                
                # Rebuild this specific chunk with enhanced metadata
                chunk_fingerprint = self._calculate_content_fingerprint(chunk_facets)
                temporal_range = self._calculate_temporal_range(chunk_facets)
                
                rebuilt_chunk = {
                    "chunk_type": "n_chunk",
                    "event_facets": chunk_facets,
                    "content_fingerprint": chunk_fingerprint,  # NEW: Per-chunk fingerprint
                    "token_count": self._calculate_event_facets_tokens(chunk_facets),
                    "chunk_index": chunk_index,
                    "has_memory_chunk": False,  # Reset for potential recompression
                    "created_at": datetime.now(),
                    "temporal_range": temporal_range,  # NEW: Track chunk boundaries
                    "rebuild_reason": "content_change_surgical"
                }
                
                # Replace ONLY this chunk, preserve all others
                n_chunks[chunk_index] = rebuilt_chunk
                
                logger.info(f"Rebuilt chunk {chunk_index} in {container_id} with {len(chunk_facets)} facets")
            
            # Update chunk structure with modified chunks
            chunk_structure["n_chunks"] = n_chunks
            
        except Exception as e:
            logger.error(f"Error in precision chunk rebuilding for {container_id}: {e}", exc_info=True)

    def _map_facets_to_chunk_indices(self, facets: List[EventFacet], 
                                    chunk_structure: Dict[str, Any]) -> Dict[int, List[EventFacet]]:
        """
        Map facets to chunks based on temporal boundaries.
        
        Uses existing chunk temporal ranges to determine which facets belong
        to which chunks, enabling surgical rebuilding of only affected chunks.
        
        Args:
            facets: List of EventFacets to map
            chunk_structure: Current chunk structure with temporal ranges
            
        Returns:
            Dictionary mapping chunk indices to their facets
        """
        try:
            n_chunks = chunk_structure.get("n_chunks", [])
            mapping = {}
            
            # Initialize all chunk indices
            for chunk_index in range(len(n_chunks)):
                mapping[chunk_index] = []
            
            for chunk_index, chunk in enumerate(n_chunks):
                temporal_range = chunk.get("temporal_range", {})
                start_time = temporal_range.get("start_timestamp", 0)
                end_time = temporal_range.get("end_timestamp", float('inf'))
                
                chunk_facets = [
                    facet for facet in facets 
                    if start_time <= facet.veil_timestamp <= end_time
                ]
                mapping[chunk_index] = chunk_facets
            
            # Handle facets that don't fit in existing temporal ranges (new content)
            unmapped_facets = []
            for facet in facets:
                mapped = False
                for chunk_facets in mapping.values():
                    if facet in chunk_facets:
                        mapped = True
                        break
                if not mapped:
                    unmapped_facets.append(facet)
            
            # Assign unmapped facets to the last chunk or create new chunk logic
            if unmapped_facets and mapping:
                last_chunk_index = max(mapping.keys())
                mapping[last_chunk_index].extend(unmapped_facets)
                logger.debug(f"Assigned {len(unmapped_facets)} unmapped facets to chunk {last_chunk_index}")
            
            return mapping
            
        except Exception as e:
            logger.error(f"Error mapping facets to chunk indices: {e}", exc_info=True)
            # Fallback: assign all facets to chunk 0
            return {0: facets} if facets else {}
    
    def _memory_facets_to_m_chunks(self, memory_facets: List[EventFacet]) -> List[Dict[str, Any]]:
        """
        Convert memory EventFacets into M-chunks.
        
        Args:
            memory_facets: List of memory EventFacets
            
        Returns:
            List of M-chunk structures
        """
        m_chunks = []
        
        for i, memory_facet in enumerate(memory_facets):
            m_chunk = {
                "chunk_type": "m_chunk",
                "memory_facet": memory_facet,
                "token_count": memory_facet.properties.get("token_count", 0),
                "chunk_index": i,
                "created_at": datetime.now(),
                "is_complete": True
            }
            m_chunks.append(m_chunk)
        
        logger.debug(f"Created {len(m_chunks)} M-chunks from {len(memory_facets)} memory EventFacets")
        return m_chunks
    
    # --- TOKEN CALCULATION ---
    
    def _calculate_event_facets_tokens(self, event_facets: List[EventFacet]) -> int:
        """
        Calculate total tokens in a list of EventFacets using direct tiktoken approach.
        
        SIMPLIFIED: No more VEIL node conversions - direct EventFacet token calculation.
        """
        try:
            import tiktoken
            
            # Use GPT-4 encoding as default
            encoding = tiktoken.encoding_for_model("gpt-4")
            
            # Extract text content directly from EventFacets
            text_content = []
            
            for facet in event_facets:
                event_type = facet.get_property("event_type", "")
                content = facet.get_property("content", "")
                
                if event_type == "message_added":
                    # For messages, include sender and content
                    sender = facet.get_property("sender_name", "")
                    conversation = facet.get_property("conversation_name", "")
                    text_content.append(f"[{conversation}] {sender}: {content}")
                    
                elif event_type == "compressed_memory":
                    # For memories, include the memory summary
                    text_content.append(f"[Memory] {content}")
                    
                elif event_type == "agent_response":
                    # For agent responses, include the content
                    text_content.append(f"[Agent] {content}")
                    
                else:
                    # For other events, just include content
                    text_content.append(f"[{event_type}] {content}")
            
            # Combine all text and count tokens
            combined_text = "\n".join(text_content)
            token_count = len(encoding.encode(combined_text))
            
            # Add minimal overhead for structure (much less than VEIL nodes)
            overhead_factor = 1.1
            return int(token_count * overhead_factor)
            
        except ImportError:
            logger.warning("tiktoken not available, using character-based fallback")
            # Fallback: character-based estimation
            total_chars = sum(len(facet.get_property("content", "")) for facet in event_facets)
            return total_chars // 4  # 4 chars per token fallback
            
        except Exception as e:
            logger.warning(f"Error calculating EventFacet tokens: {e}")
            # Fallback: character-based estimation
            total_chars = sum(len(facet.get_property("content", "")) for facet in event_facets)
            return total_chars // 4  # 4 chars per token fallback
    
    def _calculate_event_facet_tokens(self, event_facet: EventFacet) -> int:
        """
        Calculate tokens for a single EventFacet.
        
        OPTIMIZED: Uses direct tiktoken calculation without VEIL node conversion.
        """
        return self._calculate_event_facets_tokens([event_facet])
    
    def _calculate_chunk_tokens(self, chunk: Dict[str, Any]) -> int:
        """Calculate tokens in a chunk structure."""
        try:
            # Use stored token count if available
            if "token_count" in chunk:
                return chunk["token_count"]
            
            # Calculate based on chunk type
            if chunk.get("chunk_type") == "n_chunk":
                event_facets = chunk.get("event_facets", [])
                return self._calculate_event_facets_tokens(event_facets)
            elif chunk.get("chunk_type") == "m_chunk":
                memory_facet = chunk.get("memory_facet")
                if memory_facet:
                    return memory_facet.properties.get("token_count", 0)
            
            return 0
            
        except Exception as e:
            logger.warning(f"Error calculating chunk tokens: {e}")
            return 0

    async def _update_chunks_with_new_facets(self, chunk_structure: Dict[str, Any], event_facets: List[EventFacet], full_facet_cache: VEILFacetCache = None):
        """
        Update chunk structure with incremental chunking architecture.
        
        NEW ARCHITECTURE: Always assign new EventFacets to chunks incrementally.
        Creates new chunks when current chunk would exceed 4k tokens.
        No container-level decisions - pure incremental chunk management.
        
        Args:
            chunk_structure: Existing chunk structure to update
            event_facets: Current EventFacets (may include new content)
        """
        try:
            container_id = chunk_structure.get("container_id", "unknown")
            
            logger.debug(f"Incremental chunking update for {container_id}")
            
            # Detect new EventFacets since last update
            new_facets = await self._detect_new_event_facets(chunk_structure, event_facets)
            
            # FIXED: Only check for content changes if we're NOT just adding new facets
            # Surgical invalidation should only be used for actual edits/changes to existing content,
            # not for normal incremental addition of new facets
            content_changed = False
            if not new_facets:
                # No new facets, so check if existing content was modified (edits, deletions, etc.)
                content_changed = await self._detect_content_changes(container_id, event_facets, chunk_structure)
                if content_changed:
                    logger.info(f"Content changes detected for {container_id} without new facets - surgical invalidation performed")
            else:
                logger.debug(f"New facets detected for {container_id} - using incremental addition, skipping surgical invalidation")
            
            # Separate new memory facets from new regular facets
            new_memory_facets, new_regular_facets = self._separate_memory_and_regular_facets(new_facets)
            
            # Add new memory facets to M-stream if any
            if new_memory_facets:
                new_m_chunks = self._memory_facets_to_m_chunks(new_memory_facets)
                chunk_structure["m_chunks"].extend(new_m_chunks)
                logger.info(f"Added {len(new_memory_facets)} new memory EventFacets to M-stream for {container_id}")
            
            # ROLLING WINDOW: Add new regular facets to N-stream with immediate memory formation
            if new_regular_facets:
                # Pass full_facet_cache for compressed context building
                await self._add_facets_to_n_stream_incrementally(
                    chunk_structure, new_regular_facets, container_id, full_facet_cache
                )
                logger.info(f"Added {len(new_regular_facets)} new EventFacets to N-stream for {container_id} (rolling window)")
            else:
                logger.debug(f"No new regular EventFacets detected for {container_id}")
            
            # Update metadata
            chunk_structure["last_update"] = datetime.now()
            chunk_structure["total_tokens"] = self._calculate_total_chunk_tokens(chunk_structure)
            
            final_n_count = len(chunk_structure.get('n_chunks', []))
            final_m_count = len(chunk_structure.get('m_chunks', []))
            logger.debug(f"Updated chunks for {container_id}: {final_n_count} N-chunks, {final_m_count} M-chunks (incremental)")
            
        except Exception as e:
            logger.error(f"Error in incremental chunk update: {e}", exc_info=True)
    
    async def _add_facets_to_n_stream_incrementally(self, chunk_structure: Dict[str, Any], 
                                                  new_facets: List[EventFacet], 
                                                  container_id: str,
                                                  full_facet_cache: VEILFacetCache = None):
        """
        Add new EventFacets to N-stream with rolling window memory formation.
        
        ENHANCED FOR ROLLING WINDOW:
        - Triggers immediate memory formation when chunks complete (4k tokens)
        - Uses compressed context for memory formation
        - Background memory creation starts before chunks age out of window
        
        Args:
            chunk_structure: Chunk structure to update
            new_facets: New EventFacets to add
            container_id: Container ID for logging
            full_facet_cache: Complete VEILFacetCache for compressed context building
        """
        try:
            n_chunks = chunk_structure.get("n_chunks", [])
            
            # Get or create current (chronologically latest) chunk
            if not n_chunks:
                # Create first chunk
                current_chunk = {
                    "chunk_type": "n_chunk",
                    "event_facets": [],
                    "token_count": 0,
                    "chunk_index": 0,
                    "created_at": datetime.now(),
                    "is_complete": False,
                    "has_memory_chunk": False,
                    "container_id": container_id
                }
                n_chunks.append(current_chunk)
                logger.debug(f"Created first N-chunk for {container_id}")
            else:
                # Get the chronologically latest chunk (should be last, but let's be explicit)
                current_chunk = n_chunks[-1]
                
                # CRITICAL FIX: Never add to chunks that have been compressed
                if current_chunk.get("has_memory_chunk", False):
                    current_chunk = {
                        "chunk_type": "n_chunk",
                        "event_facets": [],
                        "token_count": 0,
                        "chunk_index": len(n_chunks),
                        "created_at": datetime.now(),
                        "is_complete": False,
                        "has_memory_chunk": False
                    }
                    n_chunks.append(current_chunk)
                else:
                    logger.debug(f"Using existing latest chunk {current_chunk['chunk_index']} for new facets")
            
            # Add each new facet, creating new chunks when current is already full
            for facet in new_facets:
                facet_tokens = self._calculate_event_facet_tokens(facet)
                current_tokens = current_chunk.get("token_count", 0)
                
                content_preview = facet.get_property("content", "")[:30] + "..." if len(facet.get_property("content", "")) > 30 else facet.get_property("content", "")
                
                # CRITICAL FIX: Never add to compressed chunks
                if current_chunk.get("has_memory_chunk", False):
                    new_chunk = {
                        "chunk_type": "n_chunk",
                        "event_facets": [facet],
                        "token_count": facet_tokens,
                        "chunk_index": len(n_chunks),
                        "created_at": datetime.now(),
                        "is_complete": False,
                        "has_memory_chunk": False,
                        "container_id": container_id
                    }
                    n_chunks.append(new_chunk)
                    current_chunk = new_chunk
                
                # ROLLING WINDOW: Check if current chunk should be completed before adding
                elif current_tokens >= self.COMPRESSION_CHUNK_SIZE and current_chunk.get("event_facets"):
                    # Current chunk is full - complete it and trigger memory formation
                    current_chunk["is_complete"] = True
                    
                    # ROLLING WINDOW: Trigger immediate memory formation for completed chunk
                    if not current_chunk.get("has_memory_chunk", False) and full_facet_cache:
                        chunk_index = current_chunk.get("chunk_index", len(n_chunks) - 1)
                        logger.info(f"Rolling window: triggering memory formation for completed chunk {chunk_index} "
                                   f"({current_tokens} tokens >= {self.COMPRESSION_CHUNK_SIZE})")
                        
                        await self._trigger_chunk_memory_formation(
                            container_id, current_chunk, chunk_index, 
                            chunk_structure, full_facet_cache
                        )
                    
                    # Create new chunk for this facet
                    new_chunk = {
                        "chunk_type": "n_chunk",
                        "event_facets": [facet],
                        "token_count": facet_tokens,
                        "chunk_index": len(n_chunks),
                        "created_at": datetime.now(),
                        "is_complete": False,
                        "has_memory_chunk": False,
                        "container_id": container_id
                    }
                    n_chunks.append(new_chunk)
                    current_chunk = new_chunk
                    
                else:
                    # Add to current chunk (allow overflow - rolling window will handle completion)
                    current_chunk["event_facets"].append(facet)
                    current_chunk["token_count"] = current_tokens + facet_tokens
                    
                    new_total = current_tokens + facet_tokens
                    
                    # ROLLING WINDOW: Check if chunk just completed with this facet
                    if (new_total >= self.COMPRESSION_CHUNK_SIZE and 
                        not current_chunk.get("is_complete", False) and 
                        not current_chunk.get("has_memory_chunk", False) and 
                        full_facet_cache):
                        
                        current_chunk["is_complete"] = True
                        chunk_index = current_chunk.get("chunk_index", len(n_chunks) - 1)
                        
                        logger.info(f"Rolling window: chunk {chunk_index} completed with new facet "
                                   f"({new_total} tokens >= {self.COMPRESSION_CHUNK_SIZE}), triggering memory formation")
                        
                        await self._trigger_chunk_memory_formation(
                            container_id, current_chunk, chunk_index,
                            chunk_structure, full_facet_cache
                        )            
            # Update chunk structure
            chunk_structure["n_chunks"] = n_chunks
            
            # Log final state
            total_chunks = len(n_chunks)
            last_chunk_tokens = n_chunks[-1]["token_count"] if n_chunks else 0
            logger.debug(f"Incremental chunking complete for {container_id}: {total_chunks} N-chunks, latest chunk has {last_chunk_tokens} tokens")
            
            # DEBUGGING: Show final chunk structure
            for i, chunk in enumerate(n_chunks):
                tokens = chunk.get("token_count", 0)
                facet_count = len(chunk.get("event_facets", []))
                has_memory = chunk.get("has_memory_chunk", False)
            
        except Exception as e:
            logger.error(f"Error adding facets to N-stream incrementally: {e}", exc_info=True)

    async def _detect_new_event_facets(self, chunk_structure: Dict[str, Any], current_facets: List[EventFacet]) -> List[EventFacet]:
        """
        Detect new EventFacets since last update.
        
        Args:
            chunk_structure: Current chunk structure with existing facets
            current_facets: Current list of EventFacets
            
        Returns:
            List of new EventFacets not seen before
        """
        try:
            # Get existing facet IDs from N-chunks and M-chunks
            existing_facet_ids = set()
            
            # Collect from N-chunks
            for n_chunk in chunk_structure.get("n_chunks", []):
                for facet in n_chunk.get("event_facets", []):
                    existing_facet_ids.add(facet.facet_id)
            
            # Collect from M-chunks
            for m_chunk in chunk_structure.get("m_chunks", []):
                memory_facet = m_chunk.get("memory_facet")
                if memory_facet:
                    existing_facet_ids.add(memory_facet.facet_id)
                    # Also check replaced facet IDs
                    replaced_ids = memory_facet.properties.get("replaced_facet_ids", [])
                    existing_facet_ids.update(replaced_ids)
            
            # Find new facets
            new_facets = [facet for facet in current_facets if facet.facet_id not in existing_facet_ids]
            
            if new_facets:
                for facet in new_facets:
                    content_preview = facet.get_property("content", "")[:50] + "..." if len(facet.get_property("content", "")) > 50 else facet.get_property("content", "")
            else:
                logger.debug(f"No new EventFacets detected (checked {len(current_facets)} facets against {len(existing_facet_ids)} existing)")
            
            return new_facets
            
        except Exception as e:
            logger.error(f"Error detecting new EventFacets: {e}", exc_info=True)
            # Fallback: assume all facets are new on error
            return current_facets

    def _calculate_total_chunk_tokens(self, chunk_structure: Dict[str, Any]) -> int:
        """
        Calculate total tokens across all chunks in structure.
        
        Args:
            chunk_structure: Chunk structure to calculate tokens for
            
        Returns:
            Total token count
        """
        try:
            total_tokens = 0
            
            # Add N-chunk tokens
            for n_chunk in chunk_structure.get("n_chunks", []):
                total_tokens += n_chunk.get("token_count", 0)
            
            # Add M-chunk tokens
            for m_chunk in chunk_structure.get("m_chunks", []):
                total_tokens += m_chunk.get("token_count", 0)
            
            return total_tokens
            
        except Exception as e:
            logger.error(f"Error calculating total chunk tokens: {e}", exc_info=True)
            return 0

    def _calculate_content_fingerprint(self, event_facets: List[EventFacet]) -> str:
        """
        Calculate a content fingerprint for EventFacets to detect changes.
        
        Adapted from old compression engine to work with EventFacets instead of VEIL children.
        
        Args:
            event_facets: List of EventFacets to fingerprint
            
        Returns:
            Content fingerprint string (SHA-256 hash)
        """
        try:
            import hashlib
            import json
            
            # Extract key content for fingerprinting
            content_items = []
            
            for facet in event_facets:
                event_type = facet.get_property("event_type", "")
                
                if event_type == "message_added":
                    # For messages, include sender, content, timestamp, and metadata
                    message_content = {
                        "event_type": event_type,
                        "facet_id": facet.facet_id,
                        "sender": facet.get_property("sender_name", ""),
                        "content": facet.get_property("content", ""),
                        "timestamp": facet.veil_timestamp,
                        "external_id": facet.get_property("external_id", ""),
                        "reactions": facet.get_property("reactions", {}),
                        "message_status": facet.get_property("message_status", "received"),
                        "is_edited": facet.get_property("is_edited", False),
                        "conversation_name": facet.get_property("conversation_name", "")
                    }
                    content_items.append(message_content)
                    
                elif event_type == "compressed_memory":
                    # For memory facets, include memory-specific properties
                    memory_content = {
                        "event_type": event_type,
                        "facet_id": facet.facet_id,
                        "content": facet.get_property("content", ""),
                        "memory_id": facet.get_property("memory_id", ""),
                        "timestamp": facet.veil_timestamp,
                        "original_facet_count": facet.get_property("original_facet_count", 0),
                        "replaced_facet_ids": sorted(facet.get_property("replaced_facet_ids", []))  # Sort for deterministic hash
                    }
                    content_items.append(memory_content)
                    
                elif event_type == "agent_response":
                    # For agent responses, include content and timestamp
                    response_content = {
                        "event_type": event_type,
                        "facet_id": facet.facet_id,
                        "content": facet.get_property("content", ""),
                        "timestamp": facet.veil_timestamp
                    }
                    content_items.append(response_content)
                    
                else:
                    # For other event types, include basic properties
                    generic_content = {
                        "event_type": event_type,
                        "facet_id": facet.facet_id,
                        "timestamp": facet.veil_timestamp,
                        "content": facet.get_property("content", "")
                    }
                    content_items.append(generic_content)
            
            # Create deterministic JSON representation
            content_json = json.dumps(content_items, sort_keys=True, separators=(',', ':'))
            
            # Generate SHA-256 hash
            fingerprint = hashlib.sha256(content_json.encode('utf-8')).hexdigest()
            
            return fingerprint
            
        except Exception as e:
            logger.error(f"Error calculating content fingerprint: {e}", exc_info=True)
            # Return timestamp-based fallback
            return f"error_fallback_{datetime.now().isoformat()}"

    async def _detect_content_changes(self, container_id: str, current_facets: List[EventFacet], 
                                    chunk_structure: Dict[str, Any]) -> bool:
        """
        Detect if EventFacet content has changed since last compression using surgical chunk-level detection.
        
        IMPORTANT: This method should ONLY be used for detecting actual content changes to existing facets
        (edits, deletions, reactions, etc.), NOT for detecting new facets being added incrementally.
        
        For new facets, use _add_facets_to_n_stream_incrementally() instead.
        
        UPDATED: Now uses chunk-level change detection instead of container-level fingerprinting.
        This prevents unnecessary invalidation of unaffected chunks.
        
        Args:
            container_id: ID of the container being checked
            current_facets: Current EventFacets in container
            chunk_structure: Current chunk structure with N-chunks and M-chunks
            
        Returns:
            True if content has changed and surgical invalidation was performed
        """
        try:
            n_chunks = chunk_structure.get("n_chunks", [])
            
            if not n_chunks:
                # No existing chunks to compare against
                logger.debug(f"No existing N-chunks for {container_id}, no changes to detect")
                return False
            
            # Separate memory and regular facets
            memory_facets, regular_facets = self._separate_memory_and_regular_facets(current_facets)
            
            # Detect new facets since last update
            new_facets = await self._detect_new_event_facets(chunk_structure, current_facets)
            
            # Use surgical chunk-level change detection
            changed_chunk_indices = await self._detect_chunk_level_changes(
                container_id, new_facets, chunk_structure
            )
            
            if changed_chunk_indices:
                logger.info(f"Surgical content change detection for {container_id}: {len(changed_chunk_indices)} chunks affected")
                
                # Perform surgical invalidation and rebuilding
                await self._invalidate_specific_chunks(container_id, changed_chunk_indices, chunk_structure)
                await self._rebuild_affected_chunks(container_id, regular_facets, changed_chunk_indices, chunk_structure)
                
                # Update container-level fingerprint for overall tracking
                current_fingerprint = self._calculate_content_fingerprint(regular_facets)
                chunk_structure["content_fingerprint"] = current_fingerprint
                
                return True
            else:
                logger.debug(f"No chunk-level changes detected for {container_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error detecting content changes for {container_id}: {e}", exc_info=True)
            return True  # Assume changed on error to ensure fresh compression





    async def _recompress_m_chunks(self, container_id: str, m_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-compress M-chunks when they exceed 1-chunk limit for unfocused rendering.
        
        Creates a "memory-of-memories" when too many M-chunks exist, ensuring
        unfocused containers still get compressed memory representation.
        
        Args:
            container_id: ID of the container
            m_chunks: List of M-chunks to re-compress
            
        Returns:
            List with single re-compressed memory chunk
        """
        try:
            if not m_chunks:
                return []
            
            # Extract memory EventFacets from M-chunks
            memory_facets = []
            for m_chunk in m_chunks:
                memory_facet = m_chunk.get("memory_facet")
                if memory_facet and not m_chunk.get("is_invalid", False):
                    memory_facets.append(memory_facet)
            
            if not memory_facets:
                return []
            
            # Use AgentMemoryCompressor to create memory-of-memories
            if self._memory_compressor:
                # Create summary memory EventFacet of all individual memory EventFacets
                summary_memory_facet = await self._create_memory_summary_facet(memory_facets, container_id)
                
                if summary_memory_facet:
                    # Convert back to M-chunk format
                    recompressed_chunk = {
                        "chunk_type": "m_chunk",
                        "memory_facet": summary_memory_facet,
                        "token_count": summary_memory_facet.properties.get("token_count", 0),
                        "chunk_index": 0,
                        "created_at": datetime.now(),
                        "is_complete": True,
                        "is_recompressed": True,
                        "original_chunk_count": len(m_chunks),
                        "is_memory_of_memories": True
                    }
                    
                    logger.info(f"Re-compressed {len(m_chunks)} M-chunks into 1 memory-of-memories for {container_id}")
                    return [recompressed_chunk]
                else:
                    logger.warning(f"Failed to create memory summary for {container_id}")
                    return m_chunks[:1]  # Keep only first memory as fallback
            else:
                logger.warning(f"No memory compressor available for re-compression of {container_id}")
                return m_chunks[:1]  # Keep only first memory as fallback
                
        except Exception as e:
            logger.error(f"Error re-compressing M-chunks for {container_id}: {e}", exc_info=True)
            return m_chunks[:1]  # Fallback to first memory

    async def _create_memory_summary_facet(self, memory_facets: List[EventFacet], container_id: str) -> Optional[EventFacet]:
        """
        Create a summary memory EventFacet from multiple existing memory EventFacets.
        
        This creates a "memory-of-memories" by using AgentMemoryCompressor to 
        intelligently summarize multiple memories into a single cohesive memory.
        
        Args:
            memory_facets: List of memory EventFacets to combine
            container_id: Container ID for context
            
        Returns:
            Single summary memory EventFacet or None if failed
        """
        try:
            # Extract summaries from individual memory facets
            individual_summaries = []
            total_original_count = 0
            earliest_timestamp = float('inf')
            latest_timestamp = 0
            
            for memory_facet in memory_facets:
                summary = memory_facet.get_property("content", "Unknown memory")
                original_count = memory_facet.get_property("original_facet_count", 0)
                timestamp = memory_facet.veil_timestamp
                
                individual_summaries.append(summary)
                total_original_count += original_count
                earliest_timestamp = min(earliest_timestamp, timestamp)
                latest_timestamp = max(latest_timestamp, timestamp)
            
            # Create compression context for memory summarization
            compression_context = {
                "element_id": container_id,
                "compression_reason": "memory_recompression",
                "is_focused": True,
                "is_memory_summary": True,
                "summarized_memory_count": len(memory_facets),
                "individual_summaries": individual_summaries,  # Direct data, no pseudo-nodes!
                "full_veil_context": f"Recompressing {len(memory_facets)} memories for {container_id}"
            }
            
            # Use AgentMemoryCompressor directly with memory EventFacets (no pseudo-nodes!)
            logger.debug(f"Creating memory-of-memories for {container_id} from {len(memory_facets)} memory EventFacets (no conversions)")
            memory_result = await self._memory_compressor.compress_event_facets(
                event_facets=memory_facets,  # Use memory EventFacets directly!
                element_ids=[f"{container_id}_memory_summary"],
                compression_context=compression_context
            )
            
            if not memory_result:
                logger.warning(f"AgentMemoryCompressor failed to create memory summary for {container_id}")
                return None
            
            # Extract memory content from result
            memory_props = memory_result.get("properties", {})
            summary_text = memory_props.get("memory_summary", "")
            memory_id = memory_props.get("memory_id", f"summary_{container_id}")
            token_count = memory_props.get("token_count", 0)
            own_token_count = memory_props.get("own_token_count", 0)
            
            # Calculate mean timestamp for chronological placement
            mean_timestamp = (earliest_timestamp + latest_timestamp) / 2 if memory_facets else ConnectomeEpoch.get_veil_timestamp()
            
            # Create summary memory EventFacet
            summary_memory_facet = EventFacet(
                facet_id=f"memory_summary_{container_id}_{memory_id}",
                veil_timestamp=mean_timestamp,
                owner_element_id=memory_facets[0].owner_element_id if memory_facets else container_id,
                event_type="compressed_memory",
                content=summary_text,
                links_to=container_id
            )
            
            # Add memory-specific properties
            summary_memory_facet.properties.update({
                "is_compressed_memory": True,
                "is_memory_of_memories": True,
                "memory_id": memory_id,
                "original_facet_count": total_original_count,
                "summarized_memory_count": len(memory_facets),
                "compression_timestamp": datetime.now().isoformat(),
                "compression_approach": "veil_facet_memory_summary",
                "token_count": token_count,
                "own_token_count": own_token_count,
                "temporal_info": {
                    "earliest_content_timestamp": earliest_timestamp,
                    "latest_content_timestamp": latest_timestamp,
                    "mean_timestamp": mean_timestamp,
                    "content_timespan_seconds": latest_timestamp - earliest_timestamp,
                    "uses_mean_timestamp": True
                }
            })
            
            logger.info(f"Created memory-of-memories EventFacet {summary_memory_facet.facet_id} for {container_id}")
            return summary_memory_facet
            
        except Exception as e:
            logger.error(f"Error creating memory summary facet for {container_id}: {e}", exc_info=True)
            return None

    async def _process_container_chunk_boundaries(self, container_id: str, chunk_structure: Dict[str, Any], full_facet_cache: VEILFacetCache):
        """
        Process chunk boundaries with incremental per-chunk compression.
        
        NEW ARCHITECTURE: Compress individual chunks when they exceed 4k tokens.
        Each chunk is independently evaluated and compressed, with N-chunk → M-chunk mapping tracked.
        
        Args:
            container_id: ID of container being processed
            chunk_structure: N-chunk/M-chunk structure for container
            full_facet_cache: Complete VEILFacetCache for compression context
        """
        try:
            n_chunks = chunk_structure.get("n_chunks", [])
            
            if not n_chunks:
                logger.debug(f"No N-chunks to process for {container_id}")
                return
            
            logger.debug(f"Processing {len(n_chunks)} N-chunks for incremental compression in {container_id}")
            
            # Process each N-chunk individually
            chunks_compressed = 0
            for chunk_index, n_chunk in enumerate(n_chunks):
                try:
                    chunk_tokens = n_chunk.get("token_count", 0)
                    event_facets = n_chunk.get("event_facets", [])
                    has_memory = n_chunk.get("has_memory_chunk", False)
                    
                    # Skip chunks that are already compressed (have corresponding M-chunk)
                    if has_memory:
                        continue
                    
                    if chunk_tokens >= self.COMPRESSION_CHUNK_SIZE and event_facets:
                        
                        # Create memory EventFacet from this specific N-chunk
                        memory_facet = await self._create_memory_event_facet_from_chunk(
                            container_id, event_facets, chunk_index, full_facet_cache
                        )
                        
                        if memory_facet:
                            # Create M-chunk for this N-chunk
                            m_chunk = {
                                "chunk_type": "m_chunk",
                                "memory_facet": memory_facet,
                                "token_count": memory_facet.properties.get("token_count", 0),
                                "own_token_count": memory_facet.properties.get("own_token_count", 0),
                                "chunk_index": len(chunk_structure.get("m_chunks", [])),
                                "created_at": datetime.now(),
                                "is_complete": True,
                                "source_n_chunk_index": chunk_index,
                                "replaced_facet_count": len(event_facets)
                            }
                            
                            # Add to M-chunks
                            if "m_chunks" not in chunk_structure:
                                chunk_structure["m_chunks"] = []
                            chunk_structure["m_chunks"].append(m_chunk)
                            
                            # Mark N-chunk as having a corresponding M-chunk
                            n_chunk["has_memory_chunk"] = True
                            n_chunk["memory_chunk_index"] = m_chunk["chunk_index"]
                            
                            # Track replaced EventFacet IDs
                            replaced_facet_ids = [facet.facet_id for facet in event_facets]
                            memory_facet.properties["replaced_facet_ids"] = replaced_facet_ids
                            
                            chunks_compressed += 1
                        else:
                            logger.warning(f"Could not create memory for chunk {chunk_index}")
                    else:
                        logger.debug(f"Not compressing chunk {chunk_index}: {chunk_tokens} tokens < {self.COMPRESSION_CHUNK_SIZE} or no facets ({len(event_facets)})")
                
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_index} in {container_id}: {e}", exc_info=True)
            
            if chunks_compressed > 0:
                logger.info(f"Compressed {chunks_compressed} chunks in {container_id} (incremental per-chunk compression)")
            
            # Check if M-chunks need re-compression (memory-of-memories)
            m_chunks = chunk_structure.get("m_chunks", [])
            valid_m_chunks = [m for m in m_chunks if not m.get("is_invalid", False)]
            total_memory_tokens = sum(chunk.get("own_token_count", 0) for chunk in valid_m_chunks)
            
            # FIXED: Only recompress when we have MULTIPLE M-chunks, not a single oversized one
            if len(valid_m_chunks) > 1 and total_memory_tokens > self.COMPRESSION_CHUNK_SIZE:
                logger.info(f"Multiple M-chunks in {container_id} ({len(valid_m_chunks)} chunks, {total_memory_tokens} total tokens) exceed {self.COMPRESSION_CHUNK_SIZE} tokens, triggering re-compression")
                recompressed_m_chunks = await self._recompress_m_chunks(container_id, valid_m_chunks)
                chunk_structure["m_chunks"] = recompressed_m_chunks
                logger.info(f"Re-compressed {len(valid_m_chunks)} M-chunks into {len(recompressed_m_chunks)} memory-of-memories for {container_id}")
            elif len(valid_m_chunks) == 1:
                logger.debug(f"Single M-chunk in {container_id} has {total_memory_tokens} tokens (not recompressing single chunks)")
            else:
                logger.debug(f"M-chunks in {container_id} don't need recompression: {len(valid_m_chunks)} chunks, {total_memory_tokens} tokens")
            
            # Update chunk structure metadata
            chunk_structure["last_update"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in incremental chunk boundary processing for {container_id}: {e}", exc_info=True)
    
    async def _render_container_with_focus_rules(self, container_id: str, chunk_structure: Dict[str, Any], is_focused_container: bool) -> List[VEILFacet]:
        """
        Render container using rolling window approach with memory state checking.
        
        ROLLING WINDOW APPROACH:
        - Uses IncrementalRollingWindow to determine which chunks need memories
        - Checks AgentMemoryCompressor for memory states before using memories
        - Falls back to raw chunks when memories are not ready
        - Ensures consistent view between memory formation and runtime
        
        Args:
            container_id: ID of container being rendered
            chunk_structure: N-chunk/M-chunk structure
            is_focused_container: Whether this container is focused
            
        Returns:
            List of VEILFacets using rolling window memory management
        """
        try:
            rendered_facets = []
            n_chunks = chunk_structure.get("n_chunks", [])
            
            if not n_chunks:
                logger.debug(f"No N-chunks for {container_id}, returning empty")
                return []
            
            # ROLLING WINDOW: Use IncrementalRollingWindow to determine boundaries
            window = IncrementalRollingWindow(container_id)
            compression_boundary = window.get_compression_boundary(n_chunks, is_focused_container)
            
            logger.debug(f"Rolling window {container_id}: {len(n_chunks)} chunks, "
                        f"{'focused' if is_focused_container else 'unfocused'}, "
                        f"compression boundary at {compression_boundary}")
            
            memories_used = 0
            raw_chunks_used = 0
            
            # Process each chunk based on its position relative to the window
            for i, chunk in enumerate(n_chunks):
                if i < compression_boundary:
                    # ROLLING WINDOW: Chunk is outside window - try to use memory
                    memory_used = await self._try_use_memory_for_chunk(
                        container_id, chunk, i, rendered_facets
                    )
                    
                    if memory_used:
                        memories_used += 1
                    else:
                        # Memory not ready - fall back to raw facets
                        event_facets = chunk.get("event_facets", [])
                        rendered_facets.extend(event_facets)
                        raw_chunks_used += 1
                        logger.debug(f"Rolling window: chunk {i} outside window but memory not ready, using raw facets")
                else:
                    # ROLLING WINDOW: Chunk is within window - use raw facets
                    event_facets = chunk.get("event_facets", [])
                    rendered_facets.extend(event_facets)
                    raw_chunks_used += 1
            
            # Sort by temporal order for consistent rendering
            rendered_facets.sort(key=lambda f: f.get_temporal_key())
            
            total_memory_facets = len([f for f in rendered_facets if f.get_property("event_type") == "compressed_memory"])
            total_regular_facets = len(rendered_facets) - total_memory_facets
            
            logger.info(f"Rolling window rendered {container_id} ({'focused' if is_focused_container else 'unfocused'}): "
                       f"{len(rendered_facets)} facets ({memories_used} memories used, {raw_chunks_used} raw chunks, "
                       f"boundary at {compression_boundary})")
            
            return rendered_facets
            
        except Exception as e:
            logger.error(f"Error rendering container {container_id} with focus rules: {e}", exc_info=True)
            # Fallback: return all available facets
            all_facets = []
            for n_chunk in chunk_structure.get("n_chunks", []):
                all_facets.extend(n_chunk.get("event_facets", []))
            for m_chunk in chunk_structure.get("m_chunks", []):
                memory_facet = m_chunk.get("memory_facet")
                if memory_facet and not m_chunk.get("is_invalid", False):
                    all_facets.append(memory_facet)
            return all_facets

    async def _try_use_memory_for_chunk(self,
                                      container_id: str,
                                      chunk: Dict[str, Any],
                                      chunk_index: int,
                                      rendered_facets: List[VEILFacet]) -> bool:
        """
        Try to use memory for a chunk that's outside the rolling window.
        
        ROLLING WINDOW: Checks AgentMemoryCompressor for memory state and loads
        memory if available, otherwise indicates fallback to raw facets needed.
        
        Args:
            container_id: ID of container being rendered
            chunk: The chunk that needs memory
            chunk_index: Index of the chunk
            rendered_facets: List to append memory facet to if successful
            
        Returns:
            True if memory was used, False if raw facets should be used
        """
        try:
            if not self._memory_compressor:
                logger.debug(f"No memory compressor available for chunk {chunk_index}")
                return False
            
            # Generate memory_id using actual facet IDs (content-based, not chunk-based)
            # This ensures memory ID is deterministic based on actual content
            facet_ids = [facet.facet_id for facet in chunk.get("event_facets", [])]
            if not facet_ids:
                logger.warning(f"No facet IDs found for chunk {chunk_index}")
                return False
            memory_id = self._memory_compressor.generate_memory_id(facet_ids)
            
            logger.debug(f"Rolling window: checking memory for chunk {chunk_index} with {len(facet_ids)} facets, memory_id: {memory_id}")
            
            # Check memory state from AgentMemoryCompressor using correct memory_id
            memory_state = await self._memory_compressor.get_memory_state(memory_id)
            
            if memory_state == "existing":
                # Memory is ready - load and use it
                logger.debug(f"Rolling window: loading existing memory for chunk {chunk_index} (memory_id: {memory_id})")
                
                memory_data = await self._memory_compressor.load_memory(memory_id)
                if memory_data and memory_data.get("memorized_node"):
                    # Create memory EventFacet from loaded data
                    memory_facet = await self._create_memory_facet_from_loaded_data(
                        memory_data, container_id, chunk_index
                    )
                    
                    if memory_facet:
                        rendered_facets.append(memory_facet)
                        logger.debug(f"Rolling window: used existing memory for chunk {chunk_index}")
                        return True
                    else:
                        logger.warning(f"Failed to create memory facet from loaded data for chunk {chunk_index}")
                        return False
                else:
                    logger.warning(f"Memory data incomplete for chunk {chunk_index}, falling back to raw")
                    return False
                    
            elif memory_state == "forming":
                # Memory is being created in background - use raw facets for now
                logger.debug(f"Rolling window: memory forming for chunk {chunk_index}, using raw facets")
                return False
                
            elif memory_state == "needs_formation":
                # Memory hasn't been created yet - trigger formation and use raw facets
                logger.debug(f"Rolling window: triggering memory formation for chunk {chunk_index}")
                
                # This can happen if chunk completion didn't trigger memory formation
                # (e.g., system restart, missed trigger, etc.)
                # We don't have full_facet_cache here, so mark for future formation
                chunk["needs_memory_formation"] = True
                chunk["memory_formation_requested_at"] = datetime.now().isoformat()
                
                return False
            else:
                logger.warning(f"Unknown memory state {memory_state} for chunk {chunk_index}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking memory for chunk {chunk_index}: {e}", exc_info=True)
            return False

    async def _create_memory_facet_from_loaded_data(self,
                                                  memory_data: Dict[str, Any],
                                                  container_id: str,
                                                  chunk_index: int) -> Optional[EventFacet]:
        """
        Create memory EventFacet from loaded AgentMemoryCompressor data.
        
        Args:
            memory_data: Memory data from AgentMemoryCompressor.load_memory()
            container_id: ID of container
            chunk_index: Index of the chunk
            
        Returns:
            Memory EventFacet or None if failed
        """
        try:
            memorized_node = memory_data.get("memorized_node", {})
            properties = memorized_node.get("properties", {})
            
            memory_summary = properties.get("memory_summary", "")
            memory_id = properties.get("memory_id", f"mem_{chunk_index}")
            token_count = properties.get("token_count", 0)
            own_token_count = properties.get("own_token_count", 0)
            compression_timestamp = properties.get("compression_timestamp", "")
            
            # Parse timestamp for veil_timestamp
            try:
                from datetime import datetime
                if compression_timestamp:
                    # Remove 'Z' suffix if present and parse
                    clean_timestamp = compression_timestamp.rstrip('Z')
                    parsed_time = datetime.fromisoformat(clean_timestamp)
                    veil_timestamp = parsed_time.timestamp()
                else:
                    veil_timestamp = ConnectomeEpoch.get_veil_timestamp()
            except Exception as e:
                logger.warning(f"Error parsing compression timestamp {compression_timestamp}: {e}")
                veil_timestamp = ConnectomeEpoch.get_veil_timestamp()
            
            # Create memory EventFacet
            memory_facet = EventFacet(
                facet_id=f"loaded_memory_{container_id}_chunk_{chunk_index}_{memory_id}",
                veil_timestamp=veil_timestamp,
                owner_element_id=container_id,
                event_type="compressed_memory",
                content=memory_summary,
                links_to=container_id
            )
            
            # Add memory properties
            memory_facet.properties.update({
                "is_compressed_memory": True,
                "memory_id": memory_id,
                "token_count": token_count,
                "own_token_count": own_token_count,
                "source_chunk_index": chunk_index,
                "compression_approach": "rolling_window_loaded",
                "loaded_from_storage": True
            })
            
            logger.debug(f"Created memory facet from loaded data: {memory_summary[:100]}...")
            return memory_facet
            
        except Exception as e:
            logger.error(f"Error creating memory facet from loaded data: {e}", exc_info=True)
            return None

    # --- MEMORY EVENTFACET CREATION ---
    
    async def _create_memory_event_facet_from_chunk(self,
                                                  container_id: str,
                                                  event_facets: List[EventFacet],
                                                  chunk_index: int,
                                                  full_facet_cache: VEILFacetCache) -> Optional[EventFacet]:
        """
        Create memory EventFacet from a complete N-chunk using AgentMemoryCompressor.
        
        UPDATED: Now uses turn-based compression context for consistency with normal agent experience.
        
        This is the key orchestration method that:
        1. Gets turn-based compression context by calling HUD (per user's clarification)
        2. Calls AgentMemoryCompressor with EventFacets and turn-based context
        3. Converts result back to memory EventFacet with mean timestamp
        
        Args:
            container_id: ID of container being compressed
            event_facets: List of EventFacets to compress into memory
            chunk_index: Index of the N-chunk being compressed
            full_facet_cache: Complete VEILFacetCache for context generation
            
        Returns:
            Memory EventFacet with mean timestamp positioning, or None if failed
        """
        try:
            if not self._memory_compressor:
                logger.warning(f"No memory compressor available for {container_id}")
                return None
            
            # Get turn-based compression context by calling HUD (NEW APPROACH)
            compression_context = await self._get_compression_context_from_hud_with_content(
                container_id, full_facet_cache, event_facets
            )
            
            # Call AgentMemoryCompressor with EventFacets directly using facet IDs
            # FIXED: Use actual facet IDs for content-based memory identification
            facet_ids = [facet.facet_id for facet in event_facets]
            logger.debug(f"Calling AgentMemoryCompressor for {len(facet_ids)} EventFacets with facet IDs (turn-based context)")
            memory_result = await self._memory_compressor.compress_event_facets(
                event_facets=event_facets,
                element_ids=facet_ids,
                compression_context=compression_context
            )
            
            if not memory_result:
                logger.warning(f"AgentMemoryCompressor returned no result for {container_id}")
                return None
            
            # Extract memory content from result
            memory_props = memory_result.get("properties", {})
            memory_summary = memory_props.get("memory_summary", "")
            memory_id = memory_props.get("memory_id", f"mem_{container_id}_{chunk_index}")
            token_count = memory_props.get("token_count", 0)
            own_token_count = memory_props.get("own_token_count", 0)
            
            # Calculate mean timestamp for chronological placement
            timestamps = [facet.veil_timestamp for facet in event_facets]
            mean_timestamp = sum(timestamps) / len(timestamps) if timestamps else ConnectomeEpoch.get_veil_timestamp()
            
            # Create memory EventFacet
            memory_facet = EventFacet(
                facet_id=f"memory_{container_id}_chunk_{chunk_index}_{memory_id}",
                veil_timestamp=mean_timestamp,  # Mean of replaced content for chronological placement
                owner_element_id=event_facets[0].owner_element_id if event_facets else container_id,
                event_type="compressed_memory",  # Special event type for memories
                content=memory_summary,
                links_to=container_id  # Links to same container as replaced EventFacets
            )
            
            # Add memory-specific properties
            memory_facet.properties.update({
                "is_compressed_memory": True,
                "memory_id": memory_id,
                "original_facet_count": len(event_facets),
                "compression_timestamp": datetime.now().isoformat(),
                "replaced_facet_ids": [facet.facet_id for facet in event_facets],
                "token_count": token_count,
                "own_token_count": own_token_count,
                "source_chunk_index": chunk_index,
                "compression_approach": "veil_facet_agent_memory_compressor_turn_based",  # Updated to reflect new approach
                "temporal_info": {
                    "earliest_content_timestamp": min(timestamps) if timestamps else mean_timestamp,
                    "latest_content_timestamp": max(timestamps) if timestamps else mean_timestamp,
                    "mean_timestamp": mean_timestamp,
                    "content_timespan_seconds": (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0,
                    "uses_mean_timestamp": True
                }
            })
            
            # NEW: Detect timeline divergence caused by mean timestamp positioning
            divergence_info = self._detect_timeline_divergence(memory_facet, event_facets, full_facet_cache)
            if divergence_info:
                memory_facet.properties["timeline_divergence"] = divergence_info
                logger.debug(f"Timeline divergence detected for memory {memory_facet.facet_id}: {divergence_info['divergence_type']}")
            
            logger.info(f"Created turn-based memory EventFacet {memory_facet.facet_id} for {container_id} at timestamp {mean_timestamp}")
            return memory_facet
            
        except Exception as e:
            logger.error(f"Error creating turn-based memory EventFacet for {container_id}: {e}", exc_info=True)
            return None

    async def _build_compressed_context_for_chunk(self,
                                                 container_id: str,
                                                 chunk_structure: Dict[str, Any],
                                                 target_chunk_index: int) -> VEILFacetCache:
        """
        Build compressed context for memory creation - rolling window approach.
        
        The memory should see the same compressed view the agent will see,
        which means including existing memories for earlier chunks and raw
        content for chunks within the window.
        
        Args:
            container_id: ID of container being processed
            chunk_structure: Current chunk structure with N-chunks and M-chunks
            target_chunk_index: Index of chunk we're creating memory for
            
        Returns:
            VEILFacetCache with compressed context for memory formation
        """
        try:
            n_chunks = chunk_structure.get("n_chunks", [])
            m_chunks = chunk_structure.get("m_chunks", [])
            
            # Create synthetic facet cache for compressed context
            compressed_cache = VEILFacetCache()
            
            logger.debug(f"Building compressed context for {container_id} chunk {target_chunk_index}")
            
            # Add existing memories for chunks before target
            memories_added = 0
            for i in range(target_chunk_index):
                chunk = n_chunks[i]
                
                if chunk.get("has_memory_chunk", False):
                    # Chunk has memory - use it instead of raw content
                    memory_index = chunk.get("memory_chunk_index")
                    if memory_index is not None and memory_index < len(m_chunks):
                        memory_facet = m_chunks[memory_index].get("memory_facet")
                        if memory_facet and not m_chunks[memory_index].get("is_invalid", False):
                            compressed_cache.add_facet(memory_facet)
                            memories_added += 1
                            continue
                
                # Chunk doesn't have memory yet - include raw facets
                raw_facets_added = 0
                for facet in chunk.get("event_facets", []):
                    compressed_cache.add_facet(facet)
                    raw_facets_added += 1
                
                if raw_facets_added > 0:
                    logger.debug(f"Added {raw_facets_added} raw facets from chunk {i} (no memory available)")
            
            # Add the target chunk's raw facets (what we're memorizing)
            target_chunk = n_chunks[target_chunk_index]
            target_facets_added = 0
            for facet in target_chunk.get("event_facets", []):
                compressed_cache.add_facet(facet)
                target_facets_added += 1
            
            # Add any chunks after target (recent content within window)
            future_facets_added = 0
            for i in range(target_chunk_index + 1, len(n_chunks)):
                for facet in n_chunks[i].get("event_facets", []):
                    compressed_cache.add_facet(facet)
                    future_facets_added += 1
            
            logger.debug(f"Compressed context for {container_id} chunk {target_chunk_index}: "
                        f"{memories_added} memories + {target_facets_added} target facets + "
                        f"{future_facets_added} future facets = {len(compressed_cache.facets)} total")
            
            return compressed_cache
            
        except Exception as e:
            logger.error(f"Error building compressed context for {container_id} chunk {target_chunk_index}: {e}", exc_info=True)
            # Fallback to empty cache
            return VEILFacetCache()

    async def _trigger_chunk_memory_formation(self,
                                            container_id: str,
                                            chunk: Dict[str, Any],
                                            chunk_index: int,
                                            chunk_structure: Dict[str, Any],
                                            full_facet_cache: VEILFacetCache) -> None:
        """
        Trigger memory formation for a completed chunk - rolling window approach.
        
        This is called immediately when a chunk completes (reaches 4k tokens),
        starting background memory formation using compressed context.
        
        Args:
            container_id: ID of container being processed
            chunk: The chunk that just completed
            chunk_index: Index of the completed chunk
            chunk_structure: Current chunk structure
            full_facet_cache: Complete VEILFacetCache for context
        """
        try:
            event_facets = chunk.get("event_facets", [])
            
            if not event_facets:
                logger.warning(f"No event facets to memorize in chunk {chunk_index} for {container_id}")
                return
            
            logger.info(f"Triggering memory formation for {container_id} chunk {chunk_index} "
                       f"with {len(event_facets)} facets (rolling window, content-based ID)")
            
            # Build compressed context for this specific chunk
            compressed_cache = await self._build_compressed_context_for_chunk(
                container_id, chunk_structure, chunk_index
            )
            
            # Get turn-based compression context using compressed cache
            compression_context = await self._get_compression_context_from_hud_with_content(
                container_id, compressed_cache, event_facets
            )
            
            # Enhanced context for rolling window approach
            compression_context.update({
                "compression_reason": "chunk_completion_rolling_window",
                "chunk_index": chunk_index,
                "is_rolling_window": True,
                "uses_compressed_context": True,
                "compressed_cache_facets": len(compressed_cache.facets)
            })
            
            # Call AgentMemoryCompressor with compressed context using facet IDs
            facet_ids = [facet.facet_id for facet in event_facets]
            # Generate expected memory ID for debugging (uses actual facet IDs)
            expected_memory_id = self._memory_compressor.generate_memory_id(facet_ids) if self._memory_compressor else "unknown"
            
            logger.debug(f"Calling AgentMemoryCompressor for {len(facet_ids)} facets with compressed context "
                        f"({len(compressed_cache.facets)} facets), expected memory_id: {expected_memory_id}")
            
            memory_result = await self._memory_compressor.compress_event_facets(
                event_facets=event_facets,
                element_ids=facet_ids,
                compression_context=compression_context
            )
            
            # AgentMemoryCompressor returns:
            # - Existing memory dict if ready (shouldn't happen for new chunks)
            # - None if forming (background processing) 
            # - None if needs formation (queued)
            
            if memory_result:
                # Memory was already ready (unusual for new chunks)
                logger.info(f"Memory was immediately ready for {container_id} chunk {chunk_index}")
                await self._apply_memory_to_chunk(chunk, chunk_index, memory_result, chunk_structure)
            else:
                # Memory is forming in background - this is expected
                logger.debug(f"Memory formation queued for {container_id} chunk {chunk_index}")
                # Mark chunk as having background memory formation in progress
                chunk["memory_formation_triggered"] = True
                chunk["memory_formation_timestamp"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error triggering memory formation for {container_id} chunk {chunk_index}: {e}", exc_info=True)

    async def _apply_memory_to_chunk(self,
                                   chunk: Dict[str, Any],
                                   chunk_index: int,
                                   memory_result: Dict[str, Any],
                                   chunk_structure: Dict[str, Any]) -> None:
        """
        Apply a ready memory result to a chunk structure.
        
        Args:
            chunk: The N-chunk to update
            chunk_index: Index of the chunk
            memory_result: Memory result from AgentMemoryCompressor
            chunk_structure: Chunk structure to update
        """
        try:
            # Create memory EventFacet from the result
            memory_props = memory_result.get("properties", {})
            memory_summary = memory_props.get("memory_summary", "")
            memory_id = memory_props.get("memory_id", f"mem_{chunk_index}")
            token_count = memory_props.get("token_count", 0)
            own_token_count = memory_props.get("own_token_count", 0)
            
            # Calculate mean timestamp from original facets
            event_facets = chunk.get("event_facets", [])
            timestamps = [facet.veil_timestamp for facet in event_facets]
            mean_timestamp = sum(timestamps) / len(timestamps) if timestamps else ConnectomeEpoch.get_veil_timestamp()
            
            # Create memory EventFacet
            container_id = chunk.get("container_id", "unknown")
            memory_facet = EventFacet(
                facet_id=f"memory_{memory_id}",
                veil_timestamp=mean_timestamp,
                owner_element_id=event_facets[0].owner_element_id if event_facets else container_id,
                event_type="compressed_memory",
                content=memory_summary,
                links_to=container_id
            )
            
            # Add memory properties
            memory_facet.properties.update({
                "is_compressed_memory": True,
                "memory_id": memory_id,
                "original_facet_count": len(event_facets),
                "token_count": token_count,
                "own_token_count": own_token_count,
                "source_chunk_index": chunk_index,
                "compression_approach": "rolling_window_background",
                "replaced_facet_ids": [facet.facet_id for facet in event_facets]
            })
            
            # Create M-chunk
            m_chunk = {
                "chunk_type": "m_chunk",
                "memory_facet": memory_facet,
                "token_count": token_count,
                "own_token_count": own_token_count,
                "chunk_index": len(chunk_structure.get("m_chunks", [])),
                "created_at": datetime.now(),
                "is_complete": True,
                "source_n_chunk_index": chunk_index,
                "replaced_facet_count": len(event_facets)
            }
            
            # Add to M-chunks
            if "m_chunks" not in chunk_structure:
                chunk_structure["m_chunks"] = []
            chunk_structure["m_chunks"].append(m_chunk)
            
            # Mark N-chunk as having memory
            chunk["has_memory_chunk"] = True
            chunk["memory_chunk_index"] = m_chunk["chunk_index"]
            
            logger.info(f"Applied ready memory to chunk {chunk_index}: {memory_summary[:100]}...")
            
        except Exception as e:
            logger.error(f"Error applying memory to chunk {chunk_index}: {e}", exc_info=True)

    def _detect_timeline_divergence(self, 
                                   memory_facet: EventFacet,
                                   original_event_facets: List[EventFacet],
                                   full_facet_cache: VEILFacetCache) -> Optional[Dict[str, Any]]:
        """
        Detect if a memory EventFacet causes timeline divergence.
        
        Timeline divergence occurs when:
        1. Memory's mean timestamp falls between non-compressed content that should remain sequential
        2. The memory covers a time range that overlaps with other visible content
        3. The memory appears out of chronological order relative to surrounding context
        
        Args:
            memory_facet: The memory EventFacet being created
            original_event_facets: The original EventFacets that were compressed
            full_facet_cache: Complete VEILFacetCache for temporal context
            
        Returns:
            Divergence information dict if divergence detected, None otherwise
        """
        try:
            mean_timestamp = memory_facet.veil_timestamp
            temporal_info = memory_facet.properties.get("temporal_info", {})
            earliest_content = temporal_info.get("earliest_content_timestamp", mean_timestamp)
            latest_content = temporal_info.get("latest_content_timestamp", mean_timestamp)
            
            # Get chronological stream of all non-compressed EventFacets
            temporal_stream = full_facet_cache.get_chronological_stream(include_ambient=False)
            
            # Filter to just EventFacets (not StatusFacets) that aren't being compressed
            compressed_facet_ids = {facet.facet_id for facet in original_event_facets}
            event_stream = [
                facet for facet in temporal_stream
                if (facet.facet_type == VEILFacetType.EVENT and 
                    facet.facet_id not in compressed_facet_ids and
                    facet.properties.get("event_type") not in ["agent_response", "compressed_memory"])
            ]
            
            if len(event_stream) < 2:
                # Not enough context to detect divergence
                return None
            
            # Find events immediately before and after the memory's time range
            events_before = [f for f in event_stream if f.veil_timestamp < earliest_content]
            events_after = [f for f in event_stream if f.veil_timestamp > latest_content]
            events_overlapping = [f for f in event_stream 
                                if earliest_content <= f.veil_timestamp <= latest_content]
            
            divergence_info = {
                "memory_mean_timestamp": mean_timestamp,
                "content_time_range": [earliest_content, latest_content],
                "surrounding_context": {
                    "events_before_count": len(events_before),
                    "events_after_count": len(events_after), 
                    "overlapping_events_count": len(events_overlapping)
                }
            }
            
            # Detect specific divergence types
            if events_overlapping:
                # Type 1: Memory content overlaps with non-compressed events
                divergence_info.update({
                    "divergence_type": "temporal_overlap",
                    "description": f"Memory covers time range with {len(events_overlapping)} other events",
                    "severity": "high",
                    "overlapping_event_ids": [f.facet_id for f in events_overlapping]
                })
                return divergence_info
            
            # Check for insertion point divergence
            if events_before and events_after:
                latest_before = max(events_before, key=lambda f: f.veil_timestamp)
                earliest_after = min(events_after, key=lambda f: f.veil_timestamp)
                
                # Check if memory appears out of order relative to surrounding events
                if mean_timestamp < latest_before.veil_timestamp or mean_timestamp > earliest_after.veil_timestamp:
                    divergence_info.update({
                        "divergence_type": "insertion_order",
                        "description": f"Memory positioned between events from different time periods",
                        "severity": "medium",
                        "context_events": {
                            "before": latest_before.facet_id,
                            "after": earliest_after.facet_id
                        }
                    })
                    return divergence_info
            
            # Check for large time gaps (potential retroactive insertion)
            time_gap_threshold = 300  # 5 minutes in seconds
            
            if events_before:
                latest_before = max(events_before, key=lambda f: f.veil_timestamp)
                gap_before = earliest_content - latest_before.veil_timestamp
                
                if gap_before > time_gap_threshold:
                    divergence_info.update({
                        "divergence_type": "retroactive_insertion",
                        "description": f"Memory covers content from {gap_before:.0f}s before surrounding context",
                        "severity": "low",
                        "time_gap_seconds": gap_before
                    })
                    return divergence_info
            
            # No significant divergence detected
            return None
            
        except Exception as e:
            logger.error(f"Error detecting timeline divergence: {e}", exc_info=True)
            return None
    
    async def _get_compression_context_from_hud(self,
                                              target_container_id: str,
                                              full_facet_cache: VEILFacetCache) -> Dict[str, Any]:
        """
        LEGACY: Get compression context by calling HUD to render flat temporal context.
        
        This method is being phased out in favor of _get_compression_context_from_hud_with_content()
        which provides turn-based consistency with normal agent interaction.
        
        ORIGINAL APPROACH: AgentMemoryCompressor calls HUD to get flat temporal context
        NEW APPROACH: AgentMemoryCompressor gets same turn-based format as normal interaction
        
        Args:
            target_container_id: Container being compressed (gets focused treatment)
            full_facet_cache: Complete VEILFacetCache for context rendering
            
        Returns:
            Compression context dictionary with flat text context
        """
        try:
            # Get HUD component for context rendering
            hud_component = self._get_hud_component()
            if not hud_component:
                logger.warning(f"HUD component not available for compression context")
                return {"compression_reason": "hud_unavailable"}
            
            logger.debug(f"Requesting LEGACY flat text compression context from HUD for {target_container_id}")
            
            # Call legacy flat text HUD method
            compression_context_string = await hud_component.render_memorization_context_with_facet_cache(
                facet_cache=full_facet_cache,
                exclude_element_id=target_container_id,
                focus_element_id=target_container_id  # Target container gets focused treatment
            )
            
            compression_context = {
                "focus_element_id": target_container_id,
                "compression_reason": "veil_facet_chunk_boundary",
                "full_veil_context": compression_context_string,  # Flat text approach
                "compression_timestamp": datetime.now().isoformat(),
                "is_focused": True,  # Target container always gets focused treatment
                "memorization_approach": "legacy_flat_text"
            }
            
            logger.debug(f"Generated LEGACY compression context for {target_container_id}: {len(compression_context_string)} chars")
            return compression_context
            
        except Exception as e:
            logger.error(f"Error getting legacy compression context from HUD: {e}", exc_info=True)
            # Fallback context
            return {
                "focus_element_id": target_container_id,
                "compression_reason": "veil_facet_chunk_boundary_fallback",
                "error": str(e)
            }
    
    async def _get_compression_context_from_hud_with_content(self,
                                                           target_container_id: str,
                                                           full_facet_cache: VEILFacetCache,
                                                           content_to_memorize: List[EventFacet]) -> Dict[str, Any]:
        """
        UPDATED: Get turn-based compression context matching normal agent experience.
        
        This is the new approach that provides the agent with the same turn-based format
        it normally sees, plus a synthetic final user turn requesting memorization.
        
        Args:
            target_container_id: Container being compressed (gets focused treatment)
            full_facet_cache: Complete VEILFacetCache for context rendering
            content_to_memorize: Specific EventFacets being memorized (for synthetic turn)
            
        Returns:
            Compression context dictionary with turn-based messages
        """
        try:
            # Get HUD component for context rendering
            hud_component = self._get_hud_component()
            if not hud_component:
                logger.warning(f"HUD component not available for turn-based compression context")
                return {"compression_reason": "hud_unavailable"}
            
            logger.debug(f"Requesting turn-based compression context from HUD for {target_container_id} with {len(content_to_memorize)} facets to memorize")
            
            # Call new turn-based HUD method
            turn_messages = await hud_component.render_memorization_context_as_turns(
                facet_cache=full_facet_cache,
                exclude_element_id=target_container_id,
                focus_element_id=target_container_id,  # Target container gets focused treatment
                content_to_memorize=content_to_memorize  # NEW: Pass specific content being memorized
            )
            
            compression_context = {
                "focus_element_id": target_container_id,
                "compression_reason": "veil_facet_chunk_boundary",
                "turn_messages": turn_messages,  # NEW: Turn-based instead of flat text
                "compression_timestamp": datetime.now().isoformat(),
                "is_focused": True,  # Target container always gets focused treatment
                "content_facet_count": len(content_to_memorize),
                "memorization_approach": "turn_based"
            }
            
            logger.info(f"Generated turn-based compression context for {target_container_id}: {len(turn_messages)} turns ({len(content_to_memorize)} facets to memorize)")
            return compression_context
            
        except Exception as e:
            logger.error(f"Error getting turn-based compression context from HUD: {e}", exc_info=True)
            # Fallback to legacy method
            logger.info(f"Falling back to legacy flat text compression context for {target_container_id}")
            return await self._get_compression_context_from_hud(target_container_id, full_facet_cache)
    
    def _get_hud_component(self):
        """Get HUD component for compression context rendering."""
        try:
            return self.owner.get_hud()
        except Exception as e:
            logger.warning(f"Error getting HUD component: {e}")
            return None

