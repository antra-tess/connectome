"""
VEILFacet Compression Engine Component

Replaces CompressionEngineComponent with VEILFacet architecture support.
Implements per-container memorization, focus-dependent rendering, and memory EventFacet integration.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from opentelemetry import trace
from host.observability import get_tracer

from .base_component import Component
from elements.component_registry import register_component
from elements.elements.components.veil.veil_facet import VEILFacet
from elements.elements.components.veil.facet_types import VEILFacetType, EventFacet, StatusFacet, AmbientFacet
from elements.elements.components.veil.facet_cache import VEILFacetCache
from elements.elements.components.veil.temporal_system import ConnectomeEpoch

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


@register_component
class VEILFacetCompressionEngine(Component):
    """
    VEILFacet-native compression engine with per-container memorization.
    
    Core Features:
    - StatusFacet-based container identification
    - EventFacet grouping by links_to relationships  
    - Focus-dependent rendering (8-chunk vs 1-chunk+memories)
    - Memory EventFacet creation with mean timestamps
    - AgentMemoryCompressor integration for LLM-based memory formation
    
    Architecture:
    - Containers = StatusFacets with EventFacets linking to them
    - Per-container N-chunk/M-chunk streams (4k token boundaries)
    - Focus rules: compression target gets 8 chunks, context gets 1 chunk
    - Memory EventFacets positioned at mean timestamp of replaced content
    """
    
    COMPONENT_TYPE = "VEILFacetCompressionEngine"
    HANDLED_EVENT_TYPES = ["compression_requested", "memory_formation_completed"]
    
    # Preserve existing token limits and processing logic
    COMPRESSION_CHUNK_SIZE = 4000        # 4k tokens per N-chunk
    FOCUSED_CHUNK_LIMIT = 8              # 8 N-chunks for focused containers
    UNFOCUSED_CHUNK_LIMIT = 1            # 1 N-chunk for unfocused containers
    MIN_COMPRESSION_THRESHOLD = 1000     # 1k tokens minimum before compression
    
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
    
    async def initialize(self, **kwargs) -> None:
        """Initialize the VEILFacet compression engine."""
        await super().initialize(**kwargs)
        
        # Initialize AgentMemoryCompressor for memory formation
        await self._initialize_memory_compressor()
        
        # Set agent name from owner space
        if self.owner and hasattr(self.owner, 'agent_name'):
            self._agent_name = self.owner.agent_name
        elif self.owner and hasattr(self.owner, 'id'):
            self._agent_name = f"agent_{self.owner.id}"
        else:
            self._agent_name = f"agent_{self.id}"
        
        logger.info(f"VEILFacetCompressionEngine initialized for agent {self._agent_name}")
    
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
                                                 focus_context: Optional[Dict[str, Any]] = None,
        ) -> VEILFacetCache:
        """
        Main interface: Process VEILFacetCache with per-container memorization.
        
        This is called by HUD to get compressed VEILFacetCache for rendering.
        
        Args:
            facet_cache: VEILFacetCache from SpaceVeilProducer
            focus_context: Focus context with focus_element_id
            
        Returns:
            Processed VEILFacetCache with memory EventFacets replacing compressed content
        """
        try:
            with tracer.start_as_current_span("veil_facet_compression.process_cache") as span:
                focus_element_id = focus_context.get('focus_element_id') if focus_context else None
                span.set_attribute("compression.focus_element_id", focus_element_id or "none")
                
                logger.debug(f"Processing VEILFacetCache with {len(facet_cache.facets)} facets, focus: {focus_element_id}")
                
                # Analyze containers from VEILFacetCache  
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
                    
                    # Determine if this container needs compression
                    container_rules = self._determine_container_rules(container_id)
                    
                    if container_rules["enable_memorization"]:
                        # Process with N-chunk/M-chunk logic
                        processed_facets = await self._process_container_with_memorization(
                            container_id, event_facets, container_rules, facet_cache
                        )
                    else:
                        # Preserve all facets (scratchpad/root exclusion)
                        processed_facets = event_facets
                    
                    # Add processed facets to output cache
                    for facet in processed_facets:
                        processed_cache.add_facet(facet)
                
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
        
        Returns:
            Dictionary mapping container_id to list of EventFacets linked to it
        """
        try:
            # Find StatusFacets that serve as containers
            status_facets = [f for f in facet_cache.facets.values() 
                           if f.facet_type == VEILFacetType.STATUS]
            
            containers = {}
            
            for status_facet in status_facets:
                # Find EventFacets linking to this StatusFacet
                linked_events = [f for f in facet_cache.facets.values() 
                               if f.facet_type == VEILFacetType.EVENT 
                               and f.links_to == status_facet.facet_id]
                
                if linked_events:  # Only containers with EventFacets
                    # Sort chronologically by veil_timestamp
                    linked_events.sort(key=lambda f: f.get_temporal_key())
                    containers[status_facet.facet_id] = linked_events
                    
                    logger.debug(f"Found container {status_facet.facet_id} with {len(linked_events)} EventFacets")
            
            logger.info(f"Identified {len(containers)} containers from {len(status_facets)} StatusFacets")
            return containers
            
        except Exception as e:
            logger.error(f"Error identifying containers from cache: {e}", exc_info=True)
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
                                                 container_rules: Dict[str, Any],
                                                 full_facet_cache: VEILFacetCache) -> List[VEILFacet]:
        """
        Process a container with N-chunk/M-chunk memorization logic.
        
        Args:
            container_id: ID of container being processed
            event_facets: EventFacets linked to this container
            container_rules: Compression rules for this container
            full_facet_cache: Complete VEILFacetCache for context
            
        Returns:
            List of processed VEILFacets (mix of EventFacets and memory EventFacets)
        """
        try:
            logger.debug(f"Processing container {container_id} with {len(event_facets)} EventFacets")
            
            # Get or create chunk structure for this container
            chunk_structure = await self._get_or_create_container_chunks(container_id, event_facets)
            
            # Determine token counts
            total_tokens = self._calculate_event_facets_tokens(event_facets)
            
            # Check minimum compression threshold
            if total_tokens < self.MIN_COMPRESSION_THRESHOLD:
                logger.debug(f"Container {container_id} has {total_tokens} tokens (< {self.MIN_COMPRESSION_THRESHOLD} threshold), preserving all")
                return event_facets
            
            # Process chunk boundaries and trigger compression as needed
            await self._process_container_chunk_boundaries(container_id, chunk_structure, full_facet_cache)
            
            # Render container according to focus rules (target container gets focused treatment)
            rendered_facets = await self._render_container_with_focus_rules(
                container_id, chunk_structure, is_compression_target=True
            )
            
            logger.debug(f"Container {container_id} processed: {len(event_facets)} → {len(rendered_facets)} facets")
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
                await self._update_chunks_with_new_facets(chunk_structure, event_facets)
            
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
                "container_id": container_id
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
        Convert EventFacets into N-chunks with 4k token boundaries.
        
        Args:
            event_facets: List of regular EventFacets to chunk
            
        Returns:
            List of N-chunk structures
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
                
                # Check if adding this facet would exceed chunk limit
                if current_tokens + facet_tokens > self.COMPRESSION_CHUNK_SIZE and current_chunk:
                    # Complete current chunk
                    chunks.append({
                        "chunk_type": "n_chunk",
                        "event_facets": current_chunk,
                        "token_count": current_tokens,
                        "chunk_index": len(chunks),
                        "created_at": datetime.now(),
                        "is_complete": True
                    })
                    
                    # Start new chunk
                    current_chunk = [facet]
                    current_tokens = facet_tokens
                else:
                    # Add to current chunk
                    current_chunk.append(facet)
                    current_tokens += facet_tokens
            
            # Handle remaining facets (possibly incomplete chunk)
            if current_chunk:
                chunks.append({
                    "chunk_type": "n_chunk",
                    "event_facets": current_chunk,
                    "token_count": current_tokens,
                    "chunk_index": len(chunks),
                    "created_at": datetime.now(),
                    "is_complete": current_tokens >= self.COMPRESSION_CHUNK_SIZE
                })
            
            logger.debug(f"Created {len(chunks)} N-chunks from {len(event_facets)} EventFacets")
            return chunks
            
        except Exception as e:
            logger.error(f"Error converting EventFacets to N-chunks: {e}", exc_info=True)
            return []
    
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
        """Calculate total tokens in a list of EventFacets."""
        try:
            from .memory_compressor_interface import estimate_veil_tokens
            
            # Convert EventFacets to VEIL node format for token estimation
            veil_nodes = []
            for facet in event_facets:
                veil_node = self._convert_event_facet_to_veil_node(facet)
                veil_nodes.append(veil_node)
            
            return estimate_veil_tokens(veil_nodes)
            
        except Exception as e:
            logger.warning(f"Error calculating EventFacet tokens with tiktoken: {e}")
            # Fallback: estimate based on content length
            total_chars = sum(len(facet.properties.get("content", "")) for facet in event_facets)
            return total_chars // 4  # 4 chars per token fallback
    
    def _calculate_event_facet_tokens(self, event_facet: EventFacet) -> int:
        """Calculate tokens for a single EventFacet."""
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
    
    # --- CONVERSION UTILITIES ---
    
    def _convert_event_facet_to_veil_node(self, event_facet: EventFacet) -> Dict[str, Any]:
        """Convert EventFacet to VEIL node format for AgentMemoryCompressor."""
        try:
            # Convert EventFacet properties to VEIL node structure
            veil_node = {
                "veil_id": event_facet.facet_id,
                "node_type": "content_item",  # Generic content type
                "properties": {
                    "text_content": event_facet.properties.get("content", ""),
                    "content_nature": event_facet.properties.get("event_type", "event"),
                    "timestamp": event_facet.veil_timestamp,
                    "timestamp_iso": datetime.fromtimestamp(event_facet.veil_timestamp).isoformat() + "Z",
                    "owner_element_id": event_facet.owner_element_id,
                    "links_to": event_facet.links_to,
                    # Preserve all original properties
                    **event_facet.properties
                },
                "children": []
            }
            
            return veil_node
            
        except Exception as e:
            logger.error(f"Error converting EventFacet to VEIL node: {e}", exc_info=True)
            # Minimal fallback
            return {
                "veil_id": event_facet.facet_id,
                "node_type": "content_item",
                "properties": {
                    "text_content": str(event_facet.properties.get("content", "")),
                    "content_nature": "event"
                },
                "children": []
            }
    
    # STUB METHODS - To be implemented in next phase
    
    async def _update_chunks_with_new_facets(self, chunk_structure: Dict[str, Any], event_facets: List[EventFacet]):
        """Update existing chunk structure with new EventFacets.""" 
        # TODO: Implement chunk updating logic
        logger.debug("Chunk updating not yet implemented, using existing structure")
        pass
    
    async def _process_container_chunk_boundaries(self, container_id: str, chunk_structure: Dict[str, Any], full_facet_cache: VEILFacetCache):
        """
        Process chunk boundaries and trigger memory formation for complete N-chunks.
        
        This identifies N-chunks that are complete (>=4k tokens) and ready for compression,
        calls AgentMemoryCompressor to form memories, and converts results to memory EventFacets.
        
        Args:
            container_id: ID of container being processed
            chunk_structure: N-chunk/M-chunk structure for container
            full_facet_cache: Complete VEILFacetCache for compression context
        """
        try:
            n_chunks = chunk_structure.get("n_chunks", [])
            
            # Find complete N-chunks ready for compression
            complete_chunks = [chunk for chunk in n_chunks if chunk.get("is_complete", False)]
            
            if not complete_chunks:
                logger.debug(f"No complete N-chunks found for {container_id}")
                return
            
            logger.info(f"Processing {len(complete_chunks)} complete N-chunks for {container_id}")
            
            # Process each complete chunk
            for chunk in complete_chunks:
                try:
                    # Get EventFacets from chunk
                    event_facets = chunk.get("event_facets", [])
                    if not event_facets:
                        continue
                    
                    chunk_index = chunk.get("chunk_index", 0)
                    logger.debug(f"Forming memory for {container_id} chunk {chunk_index} with {len(event_facets)} EventFacets")
                    
                    # Create memory EventFacet from complete N-chunk
                    memory_facet = await self._create_memory_event_facet_from_chunk(
                        container_id, event_facets, chunk_index, full_facet_cache
                    )
                    
                    if memory_facet:
                        # Convert memory EventFacet to M-chunk
                        m_chunk = {
                            "chunk_type": "m_chunk",
                            "memory_facet": memory_facet,
                            "token_count": memory_facet.properties.get("token_count", 0),
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
                        
                        logger.info(f"Created memory EventFacet for {container_id} chunk {chunk_index}: {memory_facet.facet_id}")
                    else:
                        logger.warning(f"Failed to create memory EventFacet for {container_id} chunk {chunk_index}")
                
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk.get('chunk_index', 'unknown')} for {container_id}: {e}", exc_info=True)
            
            # Update chunk structure metadata
            chunk_structure["last_update"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing chunk boundaries for {container_id}: {e}", exc_info=True)
    
    async def _render_container_with_focus_rules(self, container_id: str, chunk_structure: Dict[str, Any], is_compression_target: bool) -> List[VEILFacet]:
        """
        Render container according to focus rules from memory_refactor.md.
        
        Focus Rules:
        - Compression target (focused): 8 latest N-chunks + all M-chunks
        - Context providers (unfocused): Latest incomplete N-chunk + all M-chunks
        
        Args:
            container_id: ID of container being rendered
            chunk_structure: N-chunk/M-chunk structure
            is_compression_target: Whether this container is being compressed (focused)
            
        Returns:
            List of VEILFacets according to focus rules
        """
        try:
            rendered_facets = []
            n_chunks = chunk_structure.get("n_chunks", [])
            m_chunks = chunk_structure.get("m_chunks", [])
            
            # Always include all M-chunks (memory EventFacets)
            for m_chunk in m_chunks:
                memory_facet = m_chunk.get("memory_facet")
                if memory_facet:
                    rendered_facets.append(memory_facet)
            
            # Apply focus rules for N-chunks
            if is_compression_target:
                # FOCUSED: 8 latest N-chunks for rich context during compression
                n_chunk_limit = self.FOCUSED_CHUNK_LIMIT  # 8 chunks
                selected_n_chunks = n_chunks[-n_chunk_limit:] if len(n_chunks) > n_chunk_limit else n_chunks
                
                for n_chunk in selected_n_chunks:
                    event_facets = n_chunk.get("event_facets", [])
                    rendered_facets.extend(event_facets)
                
                logger.debug(f"Focused rendering for {container_id}: {len(selected_n_chunks)} N-chunks (limit: {n_chunk_limit})")
            else:
                # UNFOCUSED: Latest incomplete N-chunk only (compressed context)
                incomplete_chunks = [chunk for chunk in n_chunks if not chunk.get("is_complete", False)]
                
                if incomplete_chunks:
                    # Use latest incomplete chunk
                    latest_incomplete = incomplete_chunks[-1]
                    event_facets = latest_incomplete.get("event_facets", [])
                    rendered_facets.extend(event_facets)
                    logger.debug(f"Unfocused rendering for {container_id}: 1 incomplete N-chunk with {len(event_facets)} facets")
                elif n_chunks:
                    # If no incomplete chunks, use latest complete chunk
                    latest_chunk = n_chunks[-1]
                    event_facets = latest_chunk.get("event_facets", [])
                    rendered_facets.extend(event_facets)
                    logger.debug(f"Unfocused rendering for {container_id}: 1 latest complete N-chunk with {len(event_facets)} facets")
                else:
                    logger.debug(f"Unfocused rendering for {container_id}: no N-chunks available")
            
            # Sort by temporal order for consistent rendering
            rendered_facets.sort(key=lambda f: f.get_temporal_key())
            
            logger.debug(f"Rendered {container_id} ({'focused' if is_compression_target else 'unfocused'}): "
                        f"{len(rendered_facets)} total facets ({len(m_chunks)} memories)")
            
            return rendered_facets
            
        except Exception as e:
            logger.error(f"Error rendering container {container_id} with focus rules: {e}", exc_info=True)
            # Fallback: return all available facets
            all_facets = []
            for n_chunk in chunk_structure.get("n_chunks", []):
                all_facets.extend(n_chunk.get("event_facets", []))
            for m_chunk in chunk_structure.get("m_chunks", []):
                memory_facet = m_chunk.get("memory_facet")
                if memory_facet:
                    all_facets.append(memory_facet)
            return all_facets
    
    # --- MEMORY EVENTFACET CREATION ---
    
    async def _create_memory_event_facet_from_chunk(self,
                                                  container_id: str,
                                                  event_facets: List[EventFacet],
                                                  chunk_index: int,
                                                  full_facet_cache: VEILFacetCache) -> Optional[EventFacet]:
        """
        Create memory EventFacet from a complete N-chunk using AgentMemoryCompressor.
        
        This is the key orchestration method that:
        1. Converts EventFacets to VEIL nodes for AgentMemoryCompressor
        2. Gets compression context by calling HUD (per user's clarification)
        3. Calls AgentMemoryCompressor to form memory
        4. Converts result back to memory EventFacet with mean timestamp
        
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
            
            # Convert EventFacets to VEIL nodes for AgentMemoryCompressor
            veil_nodes = []
            for facet in event_facets:
                veil_node = self._convert_event_facet_to_veil_node(facet)
                veil_nodes.append(veil_node)
            
            # Get compression context by calling HUD (per user's flow clarification)
            compression_context = await self._get_compression_context_from_hud(
                container_id, full_facet_cache
            )
            
            # Call AgentMemoryCompressor to form memory
            logger.debug(f"Calling AgentMemoryCompressor for {container_id} with {len(veil_nodes)} nodes")
            memory_result = await self._memory_compressor.compress_nodes(
                raw_veil_nodes=veil_nodes,
                element_ids=[container_id],
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
                "source_chunk_index": chunk_index,
                "compression_approach": "veil_facet_agent_memory_compressor",
                "temporal_info": {
                    "earliest_content_timestamp": min(timestamps) if timestamps else mean_timestamp,
                    "latest_content_timestamp": max(timestamps) if timestamps else mean_timestamp,
                    "mean_timestamp": mean_timestamp,
                    "content_timespan_seconds": (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0,
                    "uses_mean_timestamp": True
                }
            })
            
            logger.info(f"Created memory EventFacet {memory_facet.facet_id} for {container_id} at timestamp {mean_timestamp}")
            return memory_facet
            
        except Exception as e:
            logger.error(f"Error creating memory EventFacet for {container_id}: {e}", exc_info=True)
            return None
    
    async def _get_compression_context_from_hud(self,
                                              target_container_id: str,
                                              full_facet_cache: VEILFacetCache) -> Dict[str, Any]:
        """
        Get compression context by calling HUD to render "compression at this moment".
        
        This implements the user's clarification: AgentMemoryCompressor calls HUD
        to get full temporal context during memory formation.
        
        Args:
            target_container_id: Container being compressed (gets focused treatment)
            full_facet_cache: Complete VEILFacetCache for context rendering
            
        Returns:
            Compression context dictionary with rendered temporal context
        """
        try:
            # Get HUD component for context rendering
            hud_component = self._get_hud_component()
            if not hud_component:
                logger.warning(f"HUD component not available for compression context")
                return {"compression_reason": "hud_unavailable"}
            
            # Prepare focus context - target container gets focused treatment
            focus_context = {
                "focus_element_id": target_container_id,
                "compression_context": True,
                "render_mode": "compression_context"
            }
            
            # Get compression context from HUD using native VEILFacet method (no conversions!)
            # This renders: target container (8-chunk focused) + other containers (1-chunk unfocused) + StatusFacets + AmbientFacets
            logger.debug(f"Requesting native VEILFacet compression context from HUD for {target_container_id}")
            
            # Call native VEILFacet HUD method for compression context (eliminates round-trip conversions)
            compression_context_string = await hud_component.render_memorization_context_with_facet_cache(
                facet_cache=full_facet_cache,
                exclude_element_id=target_container_id,
                focus_element_id=target_container_id  # Target container gets focused treatment
            )
            
            compression_context = {
                "focus_element_id": target_container_id,
                "compression_reason": "veil_facet_chunk_boundary",
                "full_veil_context": compression_context_string,
                "compression_timestamp": datetime.now().isoformat(),
                "is_focused": True  # Target container always gets focused treatment
            }
            
            logger.debug(f"Generated compression context for {target_container_id}: {len(compression_context_string)} chars")
            return compression_context
            
        except Exception as e:
            logger.error(f"Error getting compression context from HUD: {e}", exc_info=True)
            # Fallback context
            return {
                "focus_element_id": target_container_id,
                "compression_reason": "veil_facet_chunk_boundary_fallback",
                "error": str(e)
            }
    
    def _get_hud_component(self):
        """Get HUD component for compression context rendering."""
        try:
            return self.owner.get_hud()
        except Exception as e:
            logger.warning(f"Error getting HUD component: {e}")
            return None
    
    def _convert_facet_cache_to_flat(self, facet_cache: VEILFacetCache) -> Dict[str, Any]:
        """
        Convert VEILFacetCache to flat VEIL cache format for existing HUD methods.
        
        Args:
            facet_cache: VEILFacetCache to convert
            
        Returns:
            Dictionary in flat VEIL cache format
        """
        try:
            flat_cache = {}
            
            for facet in facet_cache.facets.values():
                # Convert facet back to VEIL node format
                if facet.facet_type == VEILFacetType.EVENT:
                    veil_node = self._convert_event_facet_to_veil_node(facet)
                elif facet.facet_type == VEILFacetType.STATUS:
                    veil_node = self._convert_status_facet_to_veil_node(facet)
                elif facet.facet_type == VEILFacetType.AMBIENT:
                    veil_node = self._convert_ambient_facet_to_veil_node(facet)
                else:
                    # Generic conversion
                    veil_node = {
                        "veil_id": facet.facet_id,
                        "node_type": "unknown_facet",
                        "properties": facet.properties.copy(),
                        "children": []
                    }
                
                flat_cache[facet.facet_id] = veil_node
            
            logger.debug(f"Converted VEILFacetCache to flat cache: {len(facet_cache.facets)} facets")
            return flat_cache
            
        except Exception as e:
            logger.error(f"Error converting VEILFacetCache to flat format: {e}", exc_info=True)
            return {}
    
    def _convert_status_facet_to_veil_node(self, status_facet: StatusFacet) -> Dict[str, Any]:
        """Convert StatusFacet to VEIL node format."""
        try:
            status_type = status_facet.properties.get("status_type", "unknown")
            
            # Determine node type based on status type
            if "container" in status_type:
                node_type = "message_list"  # Container nodes
            elif "space" in status_type:
                node_type = "space_root"
            else:
                node_type = "status_update"
            
            veil_node = {
                "veil_id": status_facet.facet_id,
                "node_type": node_type,
                "properties": {
                    "operation_index": status_facet.veil_timestamp,  # Convert back to operation_index
                    "element_id": status_facet.owner_element_id,
                    "links_to": status_facet.links_to,
                    **status_facet.properties
                },
                "children": []
            }
            
            return veil_node
            
        except Exception as e:
            logger.error(f"Error converting StatusFacet to VEIL node: {e}", exc_info=True)
            return {
                "veil_id": status_facet.facet_id,
                "node_type": "status_update",
                "properties": status_facet.properties.copy(),
                "children": []
            }
    
    def _convert_ambient_facet_to_veil_node(self, ambient_facet: AmbientFacet) -> Dict[str, Any]:
        """Convert AmbientFacet to VEIL node format."""
        try:
            ambient_type = ambient_facet.properties.get("ambient_type", "unknown")
            
            veil_node = {
                "veil_id": ambient_facet.facet_id,
                "node_type": "ambient_context",
                "properties": {
                    "content_nature": f"ambient_{ambient_type}",
                    "text_content": ambient_facet.properties.get("content", ""),
                    "element_id": ambient_facet.owner_element_id,
                    **ambient_facet.properties
                },
                "children": []
            }
            
            return veil_node
            
        except Exception as e:
            logger.error(f"Error converting AmbientFacet to VEIL node: {e}", exc_info=True)
            return {
                "veil_id": ambient_facet.facet_id,
                "node_type": "ambient_context",
                "properties": ambient_facet.properties.copy(),
                "children": []
            }
    
    def _get_target_container_content_for_exclusion(self, facet_cache: VEILFacetCache, target_container_id: str) -> List[Dict[str, Any]]:
        """
        Get content from target container for exclusion from compression context.
        
        This prevents the compression context from including the content that's about to be compressed.
        
        Args:
            facet_cache: Complete VEILFacetCache
            target_container_id: Container being compressed
            
        Returns:
            List of VEIL nodes to exclude from context rendering
        """
        try:
            exclude_content = []
            
            # Find EventFacets linked to target container
            for facet in facet_cache.facets.values():
                if (facet.facet_type == VEILFacetType.EVENT and 
                    facet.links_to == target_container_id):
                    # Convert to VEIL node for exclusion
                    veil_node = self._convert_event_facet_to_veil_node(facet)
                    exclude_content.append(veil_node)
            
            logger.debug(f"Excluding {len(exclude_content)} EventFacets from compression context for {target_container_id}")
            return exclude_content
            
        except Exception as e:
            logger.error(f"Error getting target container content for exclusion: {e}", exc_info=True)
            return [] 