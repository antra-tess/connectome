"""
Compression Engine Component
Handles agent memory storage and provides compressed/full memory context to AgentLoop.
Combines memory management with future compression capabilities.
"""

import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime
import json
import asyncio
import time

from opentelemetry import trace
from host.observability import get_tracer

from .base_component import Component
from elements.component_registry import register_component
from .memory_compressor_interface import estimate_veil_tokens

# Import LLM interfaces
from llm.provider_interface import LLMMessage

# NEW: Import storage system
from storage import create_storage_from_env, StorageInterface

# NEW: Import cascade invalidation utilities
from .utils.compression_utils import (
    MemoryFormationRecord,
    CascadeInvalidationTask, 
    CascadeStatistics,
    InvalidationState,
    CascadeTimelineManager,
    calculate_dependency_fingerprint,
    find_earliest_formation_index,
    group_memories_by_container
)

if TYPE_CHECKING:
    from elements.elements.inner_space import InnerSpace

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


@register_component
class CompressionEngineComponent(Component):
    """
    Component responsible for:
    1. Storing agent's reasoning chains, responses, and tool interactions
    2. Providing memory context to AgentLoop (no compression for now - unlimited context)
    3. Future: Implementing context compression strategies
    4. NEW: Persisting all data to pluggable storage backends (file, SQLite, etc.)

    This component centralizes memory management and will evolve to include
    sophisticated compression algorithms for maintaining agent continuity
    within context limits.
    """

    COMPONENT_TYPE = "CompressionEngineComponent"
    HANDLED_EVENT_TYPES = ["reasoning_chain_stored", "memory_requested"]

    # NEW: Rolling compression token limits - UPDATED for continuous rolling
    COMPRESSION_CHUNK_SIZE = 4000    # 4k tokens per compression chunk (instead of 10 items)
    UNFOCUSED_TOTAL_LIMIT = 4000     # 4k tokens total for unfocused elements (unchanged)
    MIN_COMPRESSION_THRESHOLD = 1000 # 1k tokens minimum before triggering any compression (avoid compressing small content)
    
    # PHASE 2: NEW - Updated Constants for 8-Chunk Architecture
    FOCUSED_CHUNK_LIMIT = 8                  # 8 chunks = 32k fresh content 
    FOCUSED_MEMORY_LIMIT = 4000              # 1 chunk of memories (keep existing)
    UNFOCUSED_FRESH_LIMIT = 4000             # 1 chunk fresh content
    UNFOCUSED_MEMORY_LIMIT = 4000            # 1 chunk memories (keep existing)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Track agent identity for consistent messaging
        self._agent_name: Optional[str] = None

        # NEW: Phase 3 - Integrated MemoryCompressor for agent-driven memory formation
        self._memory_compressor = None  # Will be AgentMemoryCompressor instance

        # NEW: Conversation tracking for proper storage
        self._conversation_id: Optional[str] = None

        # Storage tracking
        self._storage = None
        self._storage_initialized = False
        
        # PHASE 2A: NEW - Dual-stream chunk tracking infrastructure
        self._element_chunks: Dict[str, Dict[str, Any]] = {}
        # Structure: {element_id: {"n_chunks": [...], "m_chunks": [...], "last_update": datetime, "total_tokens": int}}
        
        # NEW: CASCADE INVALIDATION INFRASTRUCTURE
        self._cascade_timeline_manager = CascadeTimelineManager()
        self._cascade_invalidation_queue: asyncio.Queue[CascadeInvalidationTask] = asyncio.Queue()
        self._cascade_stats = CascadeStatistics()
        
        # Background cascade processing
        self._cascade_processor_task: Optional[asyncio.Task] = None
        self._cascade_processing_enabled: bool = True
        self._max_concurrent_recompressions: int = 3
        
        # Memory persistence with cascade data
        self._cascade_timeline_file: str = ""  # Will be set during initialization
        
        logger.info(f"CompressionEngineComponent initialized ({self.id}) - ready for AgentMemoryCompressor integration, dual-stream processing, and cascade invalidation")
    
    def _on_initialize(self) -> bool:
        """Initialize the compression engine after being attached to InnerSpace."""
        try:
            # Try to get agent name from parent InnerSpace if available
            if hasattr(self.owner, 'agent_name'):
                self._agent_name = self.owner.agent_name
                # Use agent name as conversation ID for now
                self._conversation_id = f"agent_{self._agent_name}"
                logger.info(f"CompressionEngine initialized for agent: {self._agent_name}")
            else:
                # Fallback to component ID
                self._conversation_id = f"agent_{self.id}"

            # Schedule async initialization to run later
            # We can't run async code directly in _on_initialize

            async def async_init():
                try:
                    # Initialize storage backend
                    await self._initialize_storage()

                    # Load existing data from storage
                    await self._load_from_storage()

                    # NEW: Phase 3 - Initialize AgentMemoryCompressor with LLM access
                    await self._initialize_memory_compressor()

                    # NEW: Initialize cascade invalidation system
                    await self._initialize_cascade_system()

                    logger.info(f"CompressionEngine async initialization complete for {self._agent_name}")
                except Exception as e:
                    logger.error(f"Failed during async initialization: {e}", exc_info=True)

            # Try to schedule the async initialization
            try:
                # Check if we're in an async context
                loop = asyncio.get_running_loop()
                # Schedule the task
                loop.create_task(async_init())
                logger.info(f"Scheduled async initialization for CompressionEngine {self.id}")
            except RuntimeError:
                # No event loop running, we'll initialize later when needed
                logger.info(f"No event loop running, will initialize storage on first use for CompressionEngine {self.id}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize CompressionEngineComponent: {e}", exc_info=True)
            return False

    async def _ensure_storage_ready(self) -> bool:
        """Ensure storage backend is initialized before use."""
        if self._storage_initialized:
            return True

        if self._storage is None:
            logger.info(f"Storage not yet initialized for CompressionEngine {self.id}, initializing now...")
            success = await self._initialize_storage()
            if not success:
                return False

            # Also load existing data
            await self._load_from_storage()

        return self._storage_initialized

    async def _initialize_storage(self) -> bool:
        """Initialize the pluggable storage backend."""
        try:
            logger.info(f"Initializing storage backend for CompressionEngine {self.id}")

            # Create storage from environment configuration
            self._storage = create_storage_from_env()

            # Initialize the storage backend
            success = await self._storage.initialize()
            if not success:
                logger.error(f"Failed to initialize storage backend")
                return False

            self._storage_initialized = True
            logger.info(f"Storage backend successfully initialized for CompressionEngine {self.id}")

            # Test storage health
            health = await self._storage.health_check()
            logger.info(f"Storage health: {health.get('status', 'unknown')}")

            return True

        except Exception as e:
            logger.error(f"Error initializing storage backend: {e}", exc_info=True)
            return False

    async def _load_from_storage(self) -> bool:
        """Load existing conversation data and memories from storage."""
        if not self._storage_initialized or not self._conversation_id:
            return False

        try:
            logger.info(f"Loading stored data for conversation {self._conversation_id}")

            # Load conversation data (Typingcloud format with compressed memories)
            conversation_data = await self._storage.load_conversation(self._conversation_id)
            if conversation_data:
                logger.info(f"Loaded existing conversation data for {self._conversation_id}")
                # TODO: Process conversation data if needed

            # Load raw messages (uncompressed message history)
            raw_messages = await self._storage.load_raw_messages(self._conversation_id)
            if raw_messages:
                logger.info(f"Loaded {len(raw_messages)} raw messages for {self._conversation_id}")
                # TODO: Process raw messages if needed for recompression

            # Load memories (compressed memory formation sequences)
            memories = await self._storage.load_memories(self._conversation_id)
            if memories:
                logger.info(f"Loaded {len(memories)} memories for {self._conversation_id}")
                # TODO: Process memories into context

            # Load reasoning chains (agent reasoning and tool interaction history)
            agent_id = self._agent_name or self.id
            reasoning_chains = await self._storage.load_reasoning_chains(agent_id)
            if reasoning_chains:
                # Restore in-memory interactions from stored reasoning chains
                logger.info(f"Loaded {len(reasoning_chains)} reasoning chains for agent {agent_id}")

            return True

        except Exception as e:
            logger.error(f"Error loading data from storage: {e}", exc_info=True)
            return False

    async def _initialize_memory_compressor(self) -> bool:
        """
        Initialize the AgentMemoryCompressor for agent-driven memory formation.

        Phase 3: This replaces simple content summaries with agent reflection.
        """
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
                token_limit=4000,  # 4k token limit per memory as specified
                storage_base_path="storage_data/memory_storage",
                llm_provider=llm_provider
            )

            logger.info(f"AgentMemoryCompressor initialized for agent {agent_id} with LLM-based reflection")
            return True

        except Exception as e:
            logger.error(f"Error initializing AgentMemoryCompressor: {e}", exc_info=True)
            return False

    def _get_llm_provider(self):
        """Get LLM provider from InnerSpace for agent memory reflection."""
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

    async def _initialize_cascade_system(self) -> bool:
        """Initialize the cascade invalidation system."""
        try:
            # Set up cascade timeline persistence file
            if self._storage_initialized and self._conversation_id:
                self._cascade_timeline_file = f"{self._conversation_id}_cascade_timeline.json"
                
                # Load existing cascade timeline if available
                await self._load_cascade_timeline_from_storage()
                
                # Start background cascade processor
                await self._start_cascade_processor()
                
                logger.info(f"Cascade invalidation system initialized for {self._agent_name}")
                return True
            else:
                logger.warning(f"Cannot initialize cascade system - storage not ready")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing cascade system: {e}", exc_info=True)
            return False

    async def _start_cascade_processor(self) -> None:
        """Start the background cascade invalidation processor."""
        try:
            if self._cascade_processor_task is not None:
                logger.warning("Cascade processor already running")
                return
            
            loop = asyncio.get_running_loop()
            self._cascade_processor_task = loop.create_task(self._cascade_invalidation_processor())
            logger.info("Started cascade invalidation background processor")
            
        except RuntimeError:
            logger.info("No event loop running, cascade processor will start when needed")
        except Exception as e:
            logger.error(f"Error starting cascade processor: {e}", exc_info=True)

    async def _cascade_invalidation_processor(self) -> None:
        """
        Background processor for cascade invalidation tasks.
        
        This runs continuously to process cascade invalidation in the background.
        """
        logger.info("Cascade invalidation processor started")
        
        try:
            while self._cascade_processing_enabled:
                try:
                    # Wait for cascade tasks with timeout
                    task = await asyncio.wait_for(
                        self._cascade_invalidation_queue.get(), 
                        timeout=5.0
                    )
                    
                    # Process the cascade task
                    start_time = time.time()
                    await self._process_cascade_task(task)
                    processing_time = time.time() - start_time
                    
                    # Update statistics
                    self._cascade_stats.update_processing_time(processing_time)
                    
                    # Mark task as done
                    self._cascade_invalidation_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # Normal timeout, continue processing
                    continue
                except Exception as e:
                    logger.error(f"Error in cascade processor: {e}", exc_info=True)
                    self._cascade_stats.cascade_errors += 1
                    
                    # Add delay to prevent rapid error loops
                    await asyncio.sleep(1.0)
        
        except asyncio.CancelledError:
            logger.info("Cascade invalidation processor cancelled")
        except Exception as e:
            logger.error(f"Fatal error in cascade processor: {e}", exc_info=True)
        finally:
            logger.info("Cascade invalidation processor stopped")

    async def _process_cascade_task(self, task: CascadeInvalidationTask) -> None:
        """
        Process a single cascade invalidation task.
        
        This finds all dependent memories and invalidates them across containers.
        """
        try:
            logger.info(f"Processing cascade task: {len(task.trigger_memory_ids)} trigger memories, "
                       f"starting from index {task.cascade_start_index}")
            
            # STEP 1: Find all memories that depend on the invalidated ones
            dependent_memories = self._cascade_timeline_manager.find_dependent_memories(
                task.trigger_memory_ids, 
                task.cascade_start_index,
                task.max_cascade_depth
            )
            
            if not dependent_memories:
                logger.debug("No dependent memories found for cascade")
                return
            
            # STEP 2: Group dependent memories by container for efficient processing
            container_groups = group_memories_by_container(dependent_memories)
            
            # STEP 3: Apply cascade invalidation to each container
            total_cascaded = 0
            for element_id, memory_ids in container_groups.items():
                cascaded_count = await self._apply_cascade_to_container(
                    element_id, memory_ids, task.trigger_reason, task.trigger_element_id
                )
                total_cascaded += cascaded_count
            
            # STEP 4: Update formation timeline to mark cascaded memories
            self._cascade_timeline_manager.mark_memories_as_cascaded(dependent_memories, task.trigger_memory_ids)
            
            # STEP 5: Update statistics
            self._cascade_stats.total_memories_cascaded += total_cascaded
            self._cascade_stats.total_containers_affected += len(container_groups)
            self._cascade_stats.record_cascade_depth(len(dependent_memories))
            
            logger.info(f"Cascade processing complete: {total_cascaded} memories invalidated "
                       f"across {len(container_groups)} containers")
            
        except Exception as e:
            logger.error(f"Error processing cascade task: {e}", exc_info=True)
            self._cascade_stats.cascade_errors += 1

    async def _apply_cascade_to_container(self, 
                                        element_id: str, 
                                        memory_ids: List[str], 
                                        trigger_reason: str,
                                        trigger_element_id: str) -> int:
        """
        Apply cascade invalidation to memories in a specific container.
        
        This marks the M-chunks as invalid but preserves them for recompression.
        """
        try:
            cascaded_count = 0
            chunk_structure = self._element_chunks.get(element_id)
            
            if not chunk_structure:
                logger.warning(f"No chunk structure found for container {element_id}")
                return 0
            
            m_chunks = chunk_structure.get("m_chunks", [])
            for m_chunk in m_chunks:
                memory_node = m_chunk.get("memory_node", {})
                memory_id = memory_node.get("veil_id")
                
                if memory_id in memory_ids and not m_chunk.get("is_invalid", False):
                    # Mark M-chunk as cascade-invalidated
                    m_chunk["is_invalid"] = True
                    m_chunk["invalidated_at"] = datetime.now().isoformat()
                    m_chunk["invalidation_reason"] = f"cascade_from_{trigger_element_id}"
                    
                    # Update cascade metadata
                    cascade_metadata = m_chunk.get("cascade_metadata", {})
                    cascade_metadata.update({
                        "invalidation_reason": "cascade_dependency",
                        "cascade_source_element_id": trigger_element_id,
                        "cascade_trigger_reason": trigger_reason,
                        "cascaded_at": datetime.now().isoformat()
                    })
                    m_chunk["cascade_metadata"] = cascade_metadata
                    
                    cascaded_count += 1
                    logger.debug(f"Cascade-invalidated memory {memory_id} in container {element_id}")
            
            logger.info(f"Applied cascade invalidation to {cascaded_count} memories in container {element_id}")
            return cascaded_count
            
        except Exception as e:
            logger.error(f"Error applying cascade to container {element_id}: {e}", exc_info=True)
            return 0

    async def _trigger_cascade_invalidation(self, 
                                          trigger_element_id: str, 
                                          invalidated_memory_ids: List[str], 
                                          trigger_reason: str) -> None:
        """
        Trigger cascade invalidation processing for memory dependencies.
        
        This is the main entry point for cascade invalidation.
        """
        try:
            if not invalidated_memory_ids:
                return
            
            # Find the earliest formation index among invalidated memories
            earliest_formation_index = find_earliest_formation_index(
                invalidated_memory_ids, 
                self._cascade_timeline_manager.memory_id_to_formation_index
            )
            if earliest_formation_index is None:
                logger.warning(f"Could not find formation index for invalidated memories: {invalidated_memory_ids}")
                return
            
            # Create cascade task for background processing
            cascade_task = CascadeInvalidationTask(
                trigger_element_id=trigger_element_id,
                trigger_memory_ids=invalidated_memory_ids.copy(),
                trigger_reason=trigger_reason,
                cascade_start_index=earliest_formation_index,
                priority=0,  # High priority for immediate processing
                max_cascade_depth=50
            )
            
            # Queue for background processing
            await self._cascade_invalidation_queue.put(cascade_task)
            self._cascade_stats.total_invalidations_triggered += 1
            self._cascade_stats.update_queue_size(self._cascade_invalidation_queue.qsize())
            
            logger.info(f"Triggered cascade invalidation from {trigger_element_id}: "
                       f"{len(invalidated_memory_ids)} memories, starting from formation index {earliest_formation_index}")
            
        except Exception as e:
            logger.error(f"Error triggering cascade invalidation: {e}", exc_info=True)

    async def _load_cascade_timeline_from_storage(self) -> None:
        """Load cascade timeline from storage."""
        try:
            if not self._storage or not self._cascade_timeline_file:
                return
                
            # Try to load timeline data from storage
            timeline_data = await self._storage.load_raw_messages(f"cascade_timeline_{self._conversation_id}")
            if timeline_data:
                self._cascade_timeline_manager.load_timeline_from_persistence(timeline_data)
                logger.info(f"Loaded cascade timeline: {len(timeline_data)} formation records")
            else:
                logger.debug(f"No existing cascade timeline found for {self._conversation_id}")
                
        except Exception as e:
            logger.error(f"Error loading cascade timeline: {e}", exc_info=True)

    async def _save_cascade_timeline_to_storage(self) -> None:
        """Save cascade timeline to storage."""
        try:
            if not self._storage or not self._cascade_timeline_file:
                return
                
            timeline_data = self._cascade_timeline_manager.get_timeline_data_for_persistence()
            if timeline_data:
                await self._storage.store_raw_messages(f"cascade_timeline_{self._conversation_id}", timeline_data)
                logger.debug(f"Saved cascade timeline: {len(timeline_data)} formation records")
                
        except Exception as e:
            logger.error(f"Error saving cascade timeline: {e}", exc_info=True)

    async def optimize_cascade_timeline(self, retention_hours: int = 24) -> Dict[str, Any]:
        """
        Manually optimize the cascade timeline to free memory.
        
        Args:
            retention_hours: Hours of formation history to retain
            
        Returns:
            Optimization results
        """
        try:
            original_size = len(self._cascade_timeline_manager.timeline)
            removed_count = self._cascade_timeline_manager.optimize_timeline(retention_hours)
            
            # Save optimized timeline
            await self._save_cascade_timeline_to_storage()
            
            return {
                "success": True,
                "original_records": original_size,
                "removed_records": removed_count,
                "remaining_records": len(self._cascade_timeline_manager.timeline),
                "retention_hours": retention_hours
            }
            
        except Exception as e:
            logger.error(f"Error optimizing cascade timeline: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    async def get_cascade_health_status(self) -> Dict[str, Any]:
        """Get health status of cascade invalidation system for monitoring."""
        try:
            queue_size = self._cascade_invalidation_queue.qsize()
            processor_running = bool(self._cascade_processor_task and not self._cascade_processor_task.done())
            
            # Calculate error rate
            total_tasks = self._cascade_stats.total_invalidations_triggered
            error_rate = (self._cascade_stats.cascade_errors / max(1, total_tasks)) if total_tasks > 0 else 0.0
            
            # Determine health status
            health_status = "healthy"
            warnings = []
            
            if queue_size > 100:
                health_status = "warning"
                warnings.append(f"High queue size: {queue_size}")
            
            if error_rate > 0.1:
                health_status = "warning" if health_status == "healthy" else "critical"
                warnings.append(f"High error rate: {error_rate:.2%}")
            
            if not processor_running and self._cascade_processing_enabled:
                health_status = "critical"
                warnings.append("Cascade processor not running")
            
            return {
                "status": health_status,
                "cascade_enabled": self._cascade_processing_enabled,
                "processor_running": processor_running,
                "queue_size": queue_size,
                "error_rate": error_rate,
                "total_cascaded": self._cascade_stats.total_memories_cascaded,
                "total_containers_affected": self._cascade_stats.total_containers_affected,
                "max_cascade_depth": self._cascade_stats.max_cascade_depth_seen,
                "warnings": warnings,
                "timeline_size": len(self._cascade_timeline_manager.timeline)
            }
            
        except Exception as e:
            logger.error(f"Error getting cascade health status: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

    async def _create_memory_with_dependency_tracking(self, 
                                                    element_id: str, 
                                                    chunk_content: List[Dict[str, Any]], 
                                                    chunk_index: int,
                                                    chunk_element_id: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced memory creation with global dependency tracking for cascade invalidation.
        
        This creates memories using AgentMemoryCompressor while tracking formation context
        for cascade invalidation across containers.
        """
        try:
            # STEP 1: Capture current global memory context for dependency tracking
            context_memory_ids = self._get_all_valid_memory_ids()
            context_fingerprint = calculate_dependency_fingerprint(context_memory_ids)
            
            # STEP 2: Get formation metadata
            formation_timestamp = time.time()
            formation_delta_index = self._get_current_delta_index()
            global_formation_index = self._cascade_timeline_manager.next_global_formation_index
            
            # STEP 3: Create memory using AgentMemoryCompressor with enhanced context
            compression_context = {
                "element_id": element_id,
                "compression_reason": "chunk_boundary_compression",
                "is_focused": True,  # Chunk boundary compression is focused
                "use_background_compression": True,
                "global_formation_index": global_formation_index,
                "formation_context": context_memory_ids,
                "existing_memory_context": self._get_existing_memory_context_for_agent(element_id),
                "full_veil_context": await self._get_hud_rendered_context_for_memory_formation(element_id, chunk_content)
            }
            
            if self._memory_compressor:
                logger.debug(f"Creating memory with dependency tracking for N-chunk {chunk_index} of {element_id}")
                
                # Use AgentMemoryCompressor's compress_nodes for background compression
                result_node = await self._memory_compressor.compress_nodes(
                    raw_veil_nodes=chunk_content,
                    element_ids=[chunk_element_id],
                    compression_context=compression_context
                )
                
                if result_node:
                    memory_id = result_node.get("veil_id", f"memory_{chunk_element_id}")
                    
                    # STEP 4: Add cascade metadata to the memory node
                    memory_props = result_node.get("properties", {})
                    memory_props["cascade_metadata"] = {
                        "global_formation_index": global_formation_index,
                        "formation_dependencies": context_memory_ids.copy(),
                        "dependency_fingerprint": context_fingerprint,
                        "formation_timestamp": formation_timestamp,
                        "formation_delta_index": formation_delta_index,
                        "invalidation_state": "valid",
                        "cascade_generation": 0
                    }
                    
                    # STEP 5: Record in global formation timeline
                    formation_record = MemoryFormationRecord(
                        memory_id=memory_id,
                        element_id=element_id,
                        global_formation_index=global_formation_index,
                        formation_timestamp=formation_timestamp,
                        formation_delta_index=formation_delta_index,
                        context_memory_ids=context_memory_ids.copy(),
                        context_fingerprint=context_fingerprint,
                        formation_generation=0
                    )
                    
                    # Add to timeline (includes circular dependency check)
                    if self._cascade_timeline_manager.add_memory_formation(formation_record):
                        logger.info(f"Created memory {memory_id} at formation index {global_formation_index} "
                                   f"with {len(context_memory_ids)} dependencies")
                        return result_node
                    else:
                        logger.warning(f"Failed to add memory {memory_id} to timeline - circular dependency detected")
                        return None
                else:
                    logger.warning(f"AgentMemoryCompressor returned None for {chunk_element_id}")
                    return None
            else:
                logger.warning(f"No AgentMemoryCompressor available for {chunk_element_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating memory with dependency tracking: {e}", exc_info=True)
            return None

    def _get_all_valid_memory_ids(self) -> List[str]:
        """Get all currently valid memory IDs from all containers for dependency tracking."""
        valid_memory_ids = []
        
        try:
            for element_id, chunk_structure in self._element_chunks.items():
                m_chunks = chunk_structure.get("m_chunks", [])
                for m_chunk in m_chunks:
                    # Only include valid (non-invalidated) memories
                    if not m_chunk.get("is_invalid", False):
                        memory_node = m_chunk.get("memory_node", {})
                        memory_id = memory_node.get("veil_id")
                        if memory_id:
                            valid_memory_ids.append(memory_id)
        except Exception as e:
            logger.error(f"Error getting valid memory IDs: {e}", exc_info=True)
        
        return valid_memory_ids

    def _get_current_delta_index(self) -> int:
        """Get current delta index from SpaceVeilProducer for temporal consistency."""
        try:
            veil_producer = self._get_space_veil_producer()
            if veil_producer and hasattr(veil_producer, '_next_delta_index'):
                return veil_producer._next_delta_index
            return -1  # Fallback for missing producer
        except Exception as e:
            logger.warning(f"Error getting current delta index: {e}")
            return -1

    def _get_existing_memory_context_for_agent(self, element_id: str) -> List[str]:
        """Get existing memory summaries for agent reflection context."""
        try:
            memory_summaries = []
            chunk_structure = self._element_chunks.get(element_id, {})
            m_chunks = chunk_structure.get("m_chunks", [])
            
            for m_chunk in m_chunks:
                if not m_chunk.get("is_invalid", False):
                    memory_node = m_chunk.get("memory_node", {})
                    memory_summary = memory_node.get("properties", {}).get("memory_summary")
                    if memory_summary:
                        memory_summaries.append(memory_summary)
            
            return memory_summaries[-5:]  # Last 5 memories for context
        except Exception as e:
            logger.error(f"Error getting existing memory context: {e}", exc_info=True)
            return []

    async def _get_hud_rendered_context_for_memory_formation(self, element_id: str, 
                                                           chunk_content: List[Dict[str, Any]]) -> Optional[str]:
        """
        Get HUD-rendered context for memory formation with temporal consistency.
        
        This is the complex integration point mentioned in the requirements - it gets
        context that shows how the system looked at memory formation time with future edits applied.
        """
        try:
            # Get HUD component for rendering
            hud_component = self._get_hud_component()
            if not hud_component:
                logger.debug(f"HUD component not available for memory formation context")
                return None
            
            # Get current flat VEIL cache from SpaceVeilProducer
            flat_veil_cache = self._get_flat_veil_cache_from_space()
            if not flat_veil_cache:
                logger.debug(f"No flat VEIL cache available for memory formation context")
                return None
            
            # Apply any pending edit deltas for temporal consistency
            temporal_flat_cache = await self._apply_edit_deltas_to_flat_cache(
                flat_veil_cache, element_id, chunk_content
            )
            
            # Use HUD's specialized context rendering for agent memory formation
            if hasattr(hud_component, 'render_memorization_context_with_flat_cache'):
                memorization_context = await hud_component.render_memorization_context_with_flat_cache(
                    flat_veil_cache=temporal_flat_cache,
                    exclude_element_id=element_id,
                    exclude_content=chunk_content,  # Exclude the content being memorized
                    focus_element_id=element_id  # Focus on the element being memorized
                )
                
                if memorization_context:
                    logger.debug(f"Generated temporal memorization context for {element_id}: {len(memorization_context)} characters")
                    return memorization_context
            
            logger.debug(f"HUD temporal context rendering not available")
            return None
                
        except Exception as e:
            logger.warning(f"Error getting HUD-rendered context for memory formation: {e}")
            return None

    async def shutdown(self) -> bool:
        """Gracefully shutdown the compression engine and storage."""
        try:
            # NEW: Shutdown cascade invalidation system
            self._cascade_processing_enabled = False
            if self._cascade_processor_task:
                logger.info(f"Shutting down cascade processor for agent {self._agent_name}")
                self._cascade_processor_task.cancel()
                try:
                    await self._cascade_processor_task
                except asyncio.CancelledError:
                    pass
                self._cascade_processor_task = None
                
                # Save cascade timeline before shutdown
                await self._save_cascade_timeline_to_storage()

            # NEW: Phase 3 - Shutdown AgentMemoryCompressor
            if self._memory_compressor:
                logger.info(f"Shutting down AgentMemoryCompressor for agent {self._agent_name}")
                await self._memory_compressor.cleanup()
                self._memory_compressor = None

            if self._storage:
                logger.info(f"Shutting down storage backend for CompressionEngine {self.id}")
                await self._storage.shutdown()
            return True
        except Exception as e:
            logger.error(f"Error during CompressionEngine shutdown: {e}", exc_info=True)
            return False

    async def get_memory_context(self) -> List[LLMMessage]:
        """
        Placeholder for memory context retrieval.

        This will be replaced with MemoryCompressor integration in Phase 1.
        For now, returns empty to ensure no errors during Phase 2 cleanup.

        Returns:
            Empty list of LLMMessage objects (no memory context available)
        """
        logger.debug(f"CompressionEngine get_memory_context called - returning empty (MemoryCompressor not yet integrated)")
        return []

    def _extract_context_key_info(self, context: str) -> str:
        """
        Extract key information from HUD context without storing the full redundant content.

        This helps preserve what's important about each frame without storing repetitive message lists.

        Args:
            context: Full HUD context string

        Returns:
            A summary of key information from the context
        """
        if not context:
            return "Empty context"

        try:
            # Look for new messages or changes (this is a simple heuristic)
            lines = context.split('\n')

            # Count messages
            message_lines = [line for line in lines if '<sender>' in line and '<message>' in line]
            message_count = len(message_lines)

            # Look for recent activity (last few messages)
            recent_messages = message_lines[-3:] if message_lines else []

            # Extract sender names
            senders = set()
            for line in recent_messages:
                if '<sender>' in line and '</sender>' in line:
                    sender_start = line.find('<sender>') + 8
                    sender_end = line.find('</sender>')
                    if sender_end > sender_start:
                        senders.add(line[sender_start:sender_end])

            # Create summary
            if recent_messages:
                sender_list = ', '.join(sorted(senders)) if senders else "unknown senders"
                return f"Context with {message_count} messages, recent activity from: {sender_list}"
            else:
                return f"Context with {message_count} messages"

        except Exception as e:
            logger.warning(f"Error extracting context key info: {e}")
            return f"Context ({len(context)} chars)"

    async def _store_interaction_messages(self, chain_data: Dict[str, Any], context_summary: str) -> None:
        """
        Store the interaction as messages for potential recompression.

        NEW: Store interaction-focused messages instead of full context repetition.
        """
        try:
            messages = []

            # User message (context summary instead of full context)
            if context_summary:
                messages.append({
                    "role": "user",
                    "content": context_summary,
                    "timestamp": datetime.now().isoformat(),
                    "source": "interaction_summary"
                })

            # Assistant response
            agent_response = chain_data.get("agent_response")
            if agent_response:
                messages.append({
                    "role": "assistant",
                    "content": agent_response,
                    "timestamp": datetime.now().isoformat(),
                    "tool_calls": chain_data.get("tool_calls", []),
                    "tool_results": chain_data.get("tool_results", [])
                })

            # Load existing raw messages and append new ones
            existing_messages = await self._storage.load_raw_messages(self._conversation_id)
            all_messages = existing_messages + messages

            # Store updated message list
            await self._storage.store_raw_messages(self._conversation_id, all_messages)
            logger.debug(f"Stored interaction messages (total: {len(all_messages)})")

        except Exception as e:
            logger.error(f"Error storing interaction messages: {e}", exc_info=True)

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about stored memory and cascade invalidation."""
        try:
            # NEW: Phase 3 - Get stats from AgentMemoryCompressor if available
            memory_compressor_stats = {}
            if self._memory_compressor:
                try:
                    memory_compressor_stats = await self._memory_compressor.get_memory_statistics()
                except Exception as stats_err:
                    logger.warning(f"Error getting AgentMemoryCompressor stats: {stats_err}")
                    memory_compressor_stats = {"error": str(stats_err)}

            # NEW: Get cascade invalidation statistics
            cascade_stats = self._cascade_stats.to_dict()
            
            # Get timeline statistics
            timeline_stats = {
                "total_formation_records": len(self._cascade_timeline_manager.timeline),
                "next_formation_index": self._cascade_timeline_manager.next_global_formation_index,
                "timeline_cache_size": len(self._cascade_timeline_manager.memory_id_to_formation_index),
                "valid_memories": len([r for r in self._cascade_timeline_manager.timeline 
                                     if r.invalidation_state == InvalidationState.VALID]),
                "invalidated_memories": len([r for r in self._cascade_timeline_manager.timeline 
                                           if r.invalidation_state != InvalidationState.VALID])
            }
            
            # Get dual-stream chunk statistics
            chunk_stats = {}
            for element_id, chunk_structure in self._element_chunks.items():
                chunk_stats[element_id] = {
                    "n_chunks": len(chunk_structure.get("n_chunks", [])),
                    "m_chunks": len(chunk_structure.get("m_chunks", [])),
                    "total_tokens": chunk_structure.get("total_tokens", 0),
                    "last_update": chunk_structure.get("last_update", "unknown")
                }

            return {
                "agent_name": self._agent_name,
                "conversation_id": self._conversation_id,
                "memory_compressor_available": bool(self._memory_compressor),
                "memory_compressor_stats": memory_compressor_stats,
                
                # NEW: Cascade invalidation statistics
                "cascade_invalidation": {
                    "enabled": self._cascade_processing_enabled,
                    "processor_running": bool(self._cascade_processor_task and not self._cascade_processor_task.done()),
                    "queue_size": self._cascade_invalidation_queue.qsize(),
                    "statistics": cascade_stats
                },
                
                # NEW: Timeline statistics
                "global_memory_timeline": timeline_stats,
                
                # NEW: Dual-stream chunk statistics
                "dual_stream_chunks": chunk_stats,
                
                "phase": "cascade_invalidation_integrated",
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}", exc_info=True)
            return {
                "error": str(e),
                "agent_name": self._agent_name,
                "phase": "cascade_integration_error"
            }

    async def clear_memory(self) -> bool:
        """Clear all stored memory. Use with caution."""
        # Phase 2.2 cleanup: No memory interactions to clear
        logger.info(f"CompressionEngine clear_memory called - no memory to clear during Phase 2 cleanup (MemoryCompressor not yet integrated)")
        return True

    # Future compression methods (placeholder for now)
    async def enable_compression(self, strategy: str = "simple") -> bool:
        """Enable context compression (future feature)."""
        logger.info(f"Compression not yet implemented. Strategy '{strategy}' noted for future implementation.")
        self._compression_enabled = False  # Will be True when implemented
        return False

    async def compress_older_interactions(self, older_than_hours: int = 24) -> int:
        """Compress interactions older than specified hours (future feature)."""
        logger.info(f"Compression not yet implemented. Would compress interactions older than {older_than_hours} hours.")
        return 0

    # NEW: Conversation management methods

    async def store_conversation_snapshot(self, conversation_data: Dict[str, Any]) -> bool:
        """Store a complete conversation snapshot in Typingcloud format."""
        if not await self._ensure_storage_ready() or not self._conversation_id:
            return False

        try:
            success = await self._storage.store_conversation(self._conversation_id, conversation_data)
            if success:
                logger.info(f"Stored conversation snapshot for {self._conversation_id}")
            return success
        except Exception as e:
            logger.error(f"Error storing conversation snapshot: {e}", exc_info=True)
            return False

    async def load_conversation_snapshot(self) -> Optional[Dict[str, Any]]:
        """Load the stored conversation snapshot."""
        if not await self._ensure_storage_ready() or not self._conversation_id:
            return None

        try:
            return await self._storage.load_conversation(self._conversation_id)
        except Exception as e:
            logger.error(f"Error loading conversation snapshot: {e}", exc_info=True)
            return None

    async def store_memory_formation(self, memory_id: str, memory_data: Dict[str, Any]) -> bool:
        """Store a compressed memory formation sequence."""
        if not await self._ensure_storage_ready() or not self._conversation_id:
            return False

        try:
            success = await self._storage.store_memory(self._conversation_id, memory_id, memory_data)
            if success:
                logger.info(f"Stored memory formation {memory_id}")
            return success
        except Exception as e:
            logger.error(f"Error storing memory formation: {e}", exc_info=True)
            return False

    async def get_stored_memories(self) -> List[Dict[str, Any]]:
        """Get all stored memory formations for this conversation."""
        if not await self._ensure_storage_ready() or not self._conversation_id:
            return []

        try:
            return await self._storage.load_memories(self._conversation_id)
        except Exception as e:
            logger.error(f"Error loading memories: {e}", exc_info=True)
            return []

    async def get_memory_data(self) -> Dict[str, Any]:
        """
        Get structured memory data in VEIL-like format for HUD rendering.

        Returns:
            Clean VEIL-like structure for HUD rendering
        """
        try:
            memory_data = {
                "memory_type": "agent_memory_context",
                "agent_info": await self._get_agent_workspace_data(),
                "scratchpad": None,  # No scratchpad data yet
                "compressed_context": {
                    "interaction_count": 0,
                    "summary": "AgentMemoryCompressor integrated - unfocused content will be agent-reflected during rendering."
                },
                "metadata": {
                    "total_interactions": 0,  # No stored interactions during cleanup phase
                    "conversation_id": self._conversation_id,
                    "agent_name": self._agent_name,
                    "last_updated": datetime.now().isoformat(),
                    "phase": "cleanup_complete",
                    "memory_compressor_available": bool(self._memory_compressor)
                }
            }

            return memory_data

        except Exception as e:
            logger.error(f"Error generating memory data: {e}", exc_info=True)
            return {
                "memory_type": "agent_memory_context",
                "error": str(e),
                "metadata": {"agent_name": self._agent_name, "phase": "cleanup_error"}
            }

    async def _get_agent_workspace_data(self) -> Dict[str, Any]:
        """Get agent workspace metadata."""
        return {
            "agent_name": self._agent_name or "Unknown Agent",
            "conversation_id": self._conversation_id,
            "workspace_description": f"Agent workspace for {self._agent_name}" if self._agent_name else "Agent workspace",
            "storage_initialized": self._storage_initialized,
            "total_stored_interactions": 0  # No interaction storage in current implementation
        }

    async def _get_scratchpad_data(self) -> Optional[Dict[str, Any]]:
        """Get scratchpad data (placeholder for future implementation)."""
        # TODO: Implement actual scratchpad functionality
        # For now, return None to indicate no scratchpad data
        return None

    async def _get_compressed_context_summary(self) -> Dict[str, Any]:
        """Get compressed summaries of non-focused interactions."""
        if not self._memory_interactions:
            return {
                "interaction_count": 0,
                "summary": "No previous interactions to summarize."
            }

        # Group interactions by type/context
        interaction_summaries = []
        total_interactions = len(self._memory_interactions)

        # For now, create a simple summary
        # TODO: More sophisticated grouping and summarization
        recent_interactions = self._memory_interactions[-3:] if len(self._memory_interactions) > 3 else self._memory_interactions

        for i, interaction in enumerate(recent_interactions):
            context_summary = interaction.get("context_summary", "Unknown context")
            tool_count = len(interaction.get("tool_calls", []))

            interaction_summaries.append({
                "interaction_index": total_interactions - len(recent_interactions) + i + 1,
                "summary": context_summary,
                "tool_calls_made": tool_count,
                "timestamp": interaction.get("timestamp", "Unknown")
            })

        return {
            "interaction_count": total_interactions,
            "showing_recent": len(recent_interactions),
            "summary": f"I have {total_interactions} previous interactions in my memory.",
            "recent_interactions": interaction_summaries
        }

    async def _get_latest_reasoning_summary(self) -> Optional[Dict[str, Any]]:
        """Get the latest reasoning chain for the current conversation."""
        if not self._memory_interactions:
            return None

        latest_interaction = self._memory_interactions[-1]

        return {
            "has_reasoning": True,
            "timestamp": latest_interaction.get("timestamp"),
            "context_summary": latest_interaction.get("context_summary", "Unknown context"),
            "agent_response_preview": (latest_interaction.get("agent_response", "")[:200] + "...") if latest_interaction.get("agent_response") else "No response recorded",
            "tool_calls_made": len(latest_interaction.get("tool_calls", [])),
            "reasoning_notes": latest_interaction.get("reasoning_notes", "")
        }

    # NEW: VEIL Processing Methods for Unified Rendering Pipeline

    async def process_veil_with_compression(self,
                                          full_veil: Dict[str, Any],
                                          focus_context: Optional[Dict[str, Any]] = None,
                                          memory_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        NEW: Process full VEIL by compressing unfocused content while preserving focused content.

        This creates a unified rendering pipeline where:
        - Focused element content is preserved completely
        - Unfocused container children are compressed to <content_memory> summaries
        - Tool availability and metadata are maintained on compressed elements
        - Memory context is integrated as needed

        Args:
            full_veil: The complete VEIL structure from SpaceVeilProducer
            focus_context: Optional focus context with focus_element_id
            memory_data: Optional memory data for integration

        Returns:
            Processed VEIL ready for unified HUD rendering
        """
        focus_element_id = focus_context.get('focus_element_id') if focus_context else None

        logger.warning(f"🚀 COMPRESSION ENGINE DEBUG: process_veil_with_compression called - focus_element_id={focus_element_id}, full_veil_nodes={len(full_veil) if full_veil else 0}")

        with tracer.start_as_current_span("compression_engine.process_veil", attributes={
            "veil.focus_element_id": focus_element_id or "none",
            "veil.has_memory_data": bool(memory_data)
        }) as span:
            try:
                # Add the full VEIL snapshot as a span event
                # We use an event because the VEIL can be very large
                span.add_event(
                    "VEIL Snapshot",
                    attributes={"veil.snapshot.json": json.dumps(full_veil, default=str)}
                )

                if not full_veil:
                    logger.warning(f"CompressionEngine received empty VEIL for processing")
                    span.set_attribute("veil.status", "empty_veil")
                    return full_veil

                if focus_element_id:
                    logger.info(f"Processing VEIL with compression, focusing on element: {focus_element_id}")
                    processed_veil = await self._process_focused_veil(full_veil, focus_element_id, memory_data)
                else:
                    logger.info(f"Processing VEIL with compression, no focus (full context)")
                    processed_veil = await self._process_full_veil_with_memory(full_veil, memory_data)

                logger.debug(f"VEIL processing complete: preserved structure with compression applied")
                span.set_status(trace.Status(trace.StatusCode.OK))
                return processed_veil

            except Exception as e:
                logger.error(f"Error processing VEIL with compression: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, "VEIL processing failed"))
                # Fallback to original VEIL
                return full_veil

    async def _process_focused_veil(self,
                                  full_veil: Dict[str, Any],
                                  focus_element_id: str,
                                  memory_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process VEIL for focused rendering: preserve focused element, compress others.

        Args:
            full_veil: Complete VEIL structure
            focus_element_id: ID of element to focus on
            memory_data: Memory context data

        Returns:
            Processed VEIL with focused content preserved and unfocused content compressed
        """
        try:
            # Create a deep copy to avoid modifying the original
            import copy
            processed_veil = copy.deepcopy(full_veil)

            # Apply compression to unfocused containers
            await self._compress_unfocused_containers(processed_veil, focus_element_id)

            return processed_veil

        except Exception as e:
            logger.error(f"Error processing focused VEIL: {e}", exc_info=True)
            return full_veil

    async def _process_full_veil_with_memory(self,
                                           full_veil: Dict[str, Any],
                                           memory_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process VEIL for full rendering with memory integration but no compression.

        Args:
            full_veil: Complete VEIL structure
            memory_data: Memory context data

        Returns:
            Processed VEIL with memory integrated but minimal compression
        """
        try:
            if not memory_data:
                # No memory to integrate, return original VEIL
                return full_veil

            # Create a deep copy to avoid modifying the original
            import copy
            processed_veil = copy.deepcopy(full_veil)

            return processed_veil

        except Exception as e:
            logger.error(f"Error processing full VEIL with memory: {e}", exc_info=True)
            return full_veil

    async def _compress_unfocused_containers(self,
                                           veil_node: Dict[str, Any],
                                           focus_element_id: str) -> None:
        """
        Recursively compress unfocused containers while preserving focused content and tool metadata.

        ENHANCED: Now passes focus context to enable focused vs unfocused compression logic.
        This modifies the VEIL in-place by replacing unfocused container children with
        <content_memory> summaries while preserving tool availability information.

        Args:
            veil_node: Current VEIL node to process
            focus_element_id: ID of the element to preserve (not compress)
        """
        try:
            node_props = veil_node.get('properties', {})
            element_id = node_props.get('element_id')
            structural_role = node_props.get('structural_role')

            # If this is a container for the focused element, preserve it completely
            if element_id == focus_element_id:
                logger.debug(f"Preserving focused element container: {element_id}")
                return

            # Check if this container contains the focused element as a child
            # If so, recurse to children instead of compressing this container
            children = veil_node.get('children', [])
            contains_focused_element = any(
                child.get('properties', {}).get('element_id') == focus_element_id
                for child in children
            )

            if contains_focused_element:
                logger.debug(f"Container {element_id} contains focused element {focus_element_id}, recursing to children")
                # Recurse to children - some will be compressed, focused one will be preserved
                for child in children:
                    await self._compress_unfocused_containers(child, focus_element_id)
                return

            # If this is a container but NOT focused and does NOT contain focused element, compress its children
            if structural_role == "container" and element_id and element_id != focus_element_id:
                logger.debug(f"Compressing unfocused container: {element_id}")
                # NEW: Pass focus context to compression logic
                await self._compress_container_children(veil_node, element_id, focus_element_id)
                return

            # For non-container nodes, still recurse to children
            for child in children:
                await self._compress_unfocused_containers(child, focus_element_id)

        except Exception as e:
            logger.error(f"Error compressing unfocused containers: {e}", exc_info=True)

    async def _compress_container_children(self, container_node: Dict[str, Any], element_id: str, focus_element_id: str) -> None:
        """
        Compress a container's children using AgentMemoryCompressor for agent-driven memory formation.
        
        PHASE 2: ENHANCED with dual-stream architecture:
        - Uses continuous chunking instead of reactive compression
        - Maintains N-stream (messages) and M-stream (memories) in parallel
        - Renders based on focus state: 8-chunk window for focused, memory + current for unfocused
        
        NEW: Scratchpad exclusion, focus-aware stream selection, and 8-chunk boundaries.
        
        Args:
            container_node: The container node to compress
            element_id: ID of the element this container represents
            focus_element_id: ID of the element to preserve (not compress)
        """
        try:
            props = container_node.get('properties', {})
            element_name = props.get('element_name', element_id)
            available_tools = props.get('available_tools', [])
            children = container_node.get('children', [])
            
            # PHASE 1: NEW - Scratchpad exclusion check BEFORE any compression
            if self._is_scratchpad_container(container_node):
                logger.debug(f"Skipping compression for scratchpad container: {element_id}")
                # Preserve scratchpad completely, just ensure tool metadata is preserved
                if available_tools:
                    container_node['properties']['available_tools'] = available_tools
                    container_node['properties']['tools_available_despite_scratchpad_exclusion'] = True
                return  # Early exit - no compression for scratchpads
            
            # NEW: 1k token threshold check - don't compress small content
            total_tokens = self._calculate_children_tokens(children)
            if total_tokens < self.MIN_COMPRESSION_THRESHOLD:
                logger.debug(f"Container {element_id} has {total_tokens} tokens (<{self.MIN_COMPRESSION_THRESHOLD} threshold), preserving full content")
                # Keep children as-is, just ensure tool metadata is preserved
                if available_tools:
                    container_node['properties']['available_tools'] = available_tools
                    container_node['properties']['tools_available_despite_preservation'] = True
                return
            
            # PHASE 2: NEW - Use dual-stream architecture instead of old compression logic
            logger.debug(f"Using dual-stream processing for element {element_id}")
            
            # Get or create chunk structure for this element
            chunks = await self._get_or_create_element_chunks(element_id, children)
            
            # Determine if this element is focused
            is_focused = (element_id == focus_element_id)
            
            # Select which stream to render based on focus state
            selected_nodes = await self._select_rendering_stream(element_id, is_focused, chunks)
            
            # Update container children with selected stream
            container_node['children'] = selected_nodes
            logger.info(f"Selected nodes: {selected_nodes}")
            
            # Ensure tool information is preserved
            if available_tools:
                container_node['properties']['available_tools'] = available_tools
                container_node['properties']['tools_available_despite_dual_stream'] = True
                container_node['properties']['dual_stream_focus_state'] = "focused" if is_focused else "unfocused"
                container_node['properties']['dual_stream_info'] = {
                    "n_chunks": len(chunks.get("n_chunks", [])),
                    "m_chunks": len(chunks.get("m_chunks", [])),
                    "total_tokens": chunks.get("total_tokens", 0),
                    "rendered_nodes": len(selected_nodes)
                }
            
            logger.info(f"Dual-stream processing complete for {element_id}: {is_focused and 'focused' or 'unfocused'} rendering with {len(selected_nodes)} nodes")
                
        except Exception as e:
            logger.error(f"Error in dual-stream container processing: {e}", exc_info=True)
            # Fallback to original children to ensure system keeps working
            # This maintains backward compatibility during the transition
            logger.warning(f"Falling back to original children for {element_id}")
            
            # Ensure tool information is preserved even in fallback
            if available_tools:
                container_node['properties']['available_tools'] = available_tools
                container_node['properties']['tools_available_despite_fallback'] = True

    def _is_scratchpad_container(self, container_node: Dict[str, Any]) -> bool:
        """
        PHASE 1: Detect scratchpad containers to exclude from compression.
        
        Uses multiple detection methods for reliability:
        - Node type contains 'scratchpad'
        - Content nature is 'scratchpad_summary'
        - Element name contains 'scratchpad'
        - Node type is 'scratchpad_summary_container'
        
        Args:
            container_node: The container VEIL node to check
            
        Returns:
            True if this is a scratchpad container that should be excluded from compression
        """
        try:
            props = container_node.get('properties', {})
            node_type = container_node.get('node_type', '')
            content_nature = props.get('content_nature', '')
            element_name = props.get('element_name', '')
            structural_role = props.get('structural_role', '')
            
            # Multiple detection methods for reliability
            
            # Check node type for scratchpad indicators
            if 'scratchpad' in node_type.lower():
                logger.debug(f"Scratchpad detected via node_type: {node_type}")
                return True
            
            # Check content nature
            if content_nature == 'scratchpad_summary':
                logger.debug(f"Scratchpad detected via content_nature: {content_nature}")
                return True
            
            # Check element name
            if element_name and 'scratchpad' in element_name.lower():
                logger.debug(f"Scratchpad detected via element_name: {element_name}")
                return True
            
            # Check for specific scratchpad container types
            if node_type in ['scratchpad_summary_container', 'scratchpad_container']:
                logger.debug(f"Scratchpad detected via specific node_type: {node_type}")
                return True
            
            # Check structural role
            if structural_role == 'scratchpad_container':
                logger.debug(f"Scratchpad detected via structural_role: {structural_role}")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error detecting scratchpad container: {e}")
            return False  # When in doubt, allow compression rather than break the system

    async def _has_content_changed_since_compression(self,
                                                   element_id: str,
                                                   memory_id: str,
                                                   current_children: List[Dict[str, Any]]) -> bool:
        """
        Check if content has changed since the last compression by comparing content fingerprints.

        Args:
            element_id: ID of the element being checked
            memory_id: ID of the existing memory
            current_children: Current VEIL children to compare

        Returns:
            True if content has changed and recompression is needed, False otherwise
        """
        try:
            # Load existing memory to get stored content fingerprint
            existing_memory = await self._memory_compressor.load_memory(memory_id)
            if not existing_memory:
                logger.warning(f"Could not load memory {memory_id} for content comparison")
                return True  # Assume changed if we can't load the memory

            # Get stored content fingerprint
            memory_metadata = existing_memory.get("metadata", {})
            stored_fingerprint = memory_metadata.get("content_fingerprint")

            if not stored_fingerprint:
                logger.debug(f"No content fingerprint stored for memory {memory_id}, assuming content changed")
                return True

            # Calculate current content fingerprint
            current_fingerprint = self._calculate_content_fingerprint(current_children)

            # Compare fingerprints
            content_changed = stored_fingerprint != current_fingerprint

            if content_changed:
                logger.debug(f"Content fingerprint changed for {element_id}: {stored_fingerprint[:16]}... → {current_fingerprint[:16]}...")
            else:
                logger.debug(f"Content fingerprint unchanged for {element_id}: {current_fingerprint[:16]}...")

            return content_changed

        except Exception as e:
            logger.error(f"Error checking content changes for {element_id}: {e}", exc_info=True)
            return True  # Assume changed on error to ensure fresh compression

    def _calculate_content_fingerprint(self, children: List[Dict[str, Any]]) -> str:
        """
        Calculate a content fingerprint for VEIL children to detect changes.

        ENHANCED: Now includes attachment content hashes for multimodal change detection.

        Args:
            children: List of VEIL child nodes

        Returns:
            Content fingerprint string
        """
        try:
            import hashlib
            import json

            # Extract key content for fingerprinting
            content_items = []

            for child in children:
                props = child.get("properties", {})
                content_nature = props.get("content_nature", "")

                if content_nature == "chat_message":
                    # For chat messages, include sender, text, and timestamp
                    message_content = {
                        "sender": props.get("sender_name", ""),
                        "text": props.get("text_content", ""),
                        "timestamp": props.get("timestamp_iso", ""),
                        "external_id": props.get("external_id", ""),  # Include ID for edit detection
                        "reactions": props.get("reactions", {}),  # Include reactions for change detection
                        "message_status": props.get("message_status", "received")  # Include status changes
                    }

                    # NEW: Include attachment metadata in fingerprint
                    attachments = props.get("attachment_metadata", [])
                    if attachments:
                        message_content["attachments"] = [
                            {
                                "filename": att.get("filename", ""),
                                "attachment_type": att.get("attachment_type", att.get("content_type", "")),
                                "attachment_id": att.get("attachment_id", "")
                            }
                            for att in attachments
                        ]

                    # NEW: Include attachment content hashes from child nodes
                    attachment_content_hashes = self._get_attachment_content_hashes(child)
                    if attachment_content_hashes:
                        message_content["attachment_content_hashes"] = attachment_content_hashes

                    content_items.append(message_content)
                else:
                    # For other content types, include basic properties
                    generic_content = {
                        "veil_id": child.get("veil_id", ""),
                        "node_type": child.get("node_type", ""),
                        "content_nature": content_nature
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

    def _get_attachment_content_hashes(self, message_node: Dict[str, Any]) -> Dict[str, str]:
        """
        NEW: Get content hashes for attachment children to detect content changes.

        Args:
            message_node: VEIL message node with potential attachment children

        Returns:
            Dictionary mapping attachment_id to content hash
        """
        try:
            import hashlib

            attachment_hashes = {}
            children = message_node.get("children", [])

            for child in children:
                if child.get("node_type") == "attachment_content_item":
                    child_props = child.get("properties", {})
                    attachment_id = child_props.get("attachment_id")

                    if attachment_id:
                        # Create a content hash based on available properties
                        # This is a placeholder - in a real implementation, we might
                        # hash the actual file content from storage
                        content_indicators = {
                            "filename": child_props.get("filename", ""),
                            "content_nature": child_props.get("content_nature", ""),
                            "content_available": child_props.get("content_available", False)
                        }

                        # Generate hash of content indicators
                        content_string = json.dumps(content_indicators, sort_keys=True)
                        content_hash = hashlib.sha256(content_string.encode('utf-8')).hexdigest()[:16]  # Short hash

                        attachment_hashes[attachment_id] = content_hash

            return attachment_hashes

        except Exception as e:
            logger.error(f"Error getting attachment content hashes: {e}", exc_info=True)
            return {}

    async def _generate_content_summary(self, children: List[Dict[str, Any]], element_id: str) -> str:
        """
        Generate a human-readable summary of compressed content.

        Args:
            children: List of child VEIL nodes to summarize
            element_id: ID of the element being summarized

        Returns:
            Human-readable content summary
        """
        try:
            if not children:
                return "Empty conversation"

            # Count different types of content
            message_count = 0
            last_message_preview = ""
            senders = set()
            has_attachments = False

            for child in children:
                child_props = child.get('properties', {})
                content_nature = child_props.get('content_nature', '')

                if content_nature == "chat_message":
                    message_count += 1
                    sender = child_props.get('sender_name')
                    if sender:
                        senders.add(sender)

                    # Get preview of last message
                    text_content = child_props.get('text_content', '')
                    if text_content:
                        last_message_preview = text_content[:50] + ("..." if len(text_content) > 50 else "")

                    # Check for attachments
                    attachments = child_props.get('attachment_metadata', [])
                    if attachments:
                        has_attachments = True

            # Build summary
            if message_count == 0:
                return "No recent messages"

            participants = f"{len(senders)} participants" if len(senders) > 1 else list(senders)[0] if senders else "unknown"

            summary = f"Chat with {participants}: {message_count} messages"

            if has_attachments:
                summary += " (with attachments)"

            if last_message_preview:
                summary += f". Recent: \"{last_message_preview}\""

            return summary

        except Exception as e:
            logger.error(f"Error generating content summary: {e}", exc_info=True)
            return f"Content from {element_id}"

    # NEW: Phase 2 utility methods for rolling compression

    def _separate_memories_and_content(self, children: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Separate existing memories from fresh content in a list of children.

        Args:
            children: List of VEIL child nodes

        Returns:
            Tuple of (memories, fresh_content)
        """
        memories = []
        content = []

        for child in children:
            node_type = child.get("node_type", "")
            if node_type in ["content_memory", "memorized_content"]:
                memories.append(child)
            else:
                content.append(child)

        return memories, content

    def _sort_memories_by_age(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort memories by age (oldest first) using compression timestamp."""
        try:
            def get_timestamp(memory):
                props = memory.get("properties", {})
                timestamp_str = props.get("compression_timestamp", "")
                if timestamp_str:
                    try:
                        from datetime import datetime
                        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except:
                        pass
                return datetime.min  # Fallback for unparseable timestamps

            return sorted(memories, key=get_timestamp)

        except Exception as e:
            logger.error(f"Error sorting memories by age: {e}", exc_info=True)
            return memories  # Return unsorted if sorting fails

    async def _create_memory_summary(self, memories_to_summarize: List[Dict[str, Any]],
                                   element_id: str, element_name: str) -> Dict[str, Any]:
        """
        Create a higher-level summary memory from multiple existing memories.

        Args:
            memories_to_summarize: List of memory nodes to combine
            element_id: Element ID for context
            element_name: Element name for context

        Returns:
            Single summary memory node
        """
        try:
            # Extract summaries from individual memories
            individual_summaries = []
            total_original_count = 0

            for memory in memories_to_summarize:
                props = memory.get("properties", {})
                summary = props.get("memory_summary", "Unknown memory")
                original_count = props.get("original_child_count", 0)

                individual_summaries.append(summary)
                total_original_count += original_count

            # Create combined summary
            if self._memory_compressor:
                # Use LLM to create intelligent summary of summaries
                combined_content = "\n".join([f"Memory {i+1}: {summary}" for i, summary in enumerate(individual_summaries)])

                # Create compression context for memory summarization
                compression_context = {
                    "element_id": element_id,
                    "element_name": element_name,
                    "compression_reason": "memory_recompression",
                    "is_focused": True,
                    "is_memory_summary": True,
                    "summarized_memory_count": len(memories_to_summarize)
                }

                # Create a pseudo-VEIL structure for the memory content
                pseudo_content = [{
                    "veil_id": f"memory_summary_content_{i}",
                    "node_type": "memory_content",
                    "properties": {
                        "content_nature": "memory_summary",
                        "text_content": summary
                    }
                } for i, summary in enumerate(individual_summaries)]

                # Use memory compressor to create intelligent summary
                memorized_node = await self._memory_compressor.compress(
                    raw_veil_nodes=pseudo_content,
                    element_ids=[f"{element_id}_summary"],
                    compression_context=compression_context
                )

                if memorized_node:
                    summary_text = memorized_node.get("properties", {}).get("memory_summary", "Summary creation failed")
                    compression_approach = "agent_memory_recompressor_summary"
                else:
                    raise Exception("Memory compressor failed for summary")

            else:
                # Fallback: simple concatenation
                summary_text = f"Combined memory: {len(memories_to_summarize)} older memories from {element_name}"
                compression_approach = "simple_memory_summary"

            # Create summary memory node
            summary_memory = {
                "veil_id": f"summary_memory_{element_id}_{self.id}",
                "node_type": "content_memory",
                "properties": {
                    "structural_role": "compressed_content",
                    "content_nature": "content_memory",
                    "original_element_id": element_id,
                    "memory_summary": summary_text,
                    "original_child_count": total_original_count,
                    "compression_timestamp": datetime.now().isoformat(),
                    "compression_approach": compression_approach,
                    "is_focused": True,
                    "is_memory_summary": True,
                    "summarized_memory_count": len(memories_to_summarize)
                },
                "children": []
            }

            logger.info(f"Created memory summary: {len(memories_to_summarize)} memories → 1 summary")
            return summary_memory

        except Exception as e:
            logger.error(f"Error creating memory summary: {e}", exc_info=True)
            # Fallback summary
            return {
                "veil_id": f"fallback_summary_{element_id}_{self.id}",
                "node_type": "content_memory",
                "properties": {
                    "structural_role": "compressed_content",
                    "content_nature": "content_memory",
                    "original_element_id": element_id,
                    "memory_summary": f"Summary of {len(memories_to_summarize)} older memories from {element_name}",
                    "original_child_count": total_original_count,
                    "compression_approach": "fallback_memory_summary",
                    "is_focused": True,
                    "is_memory_summary": True
                },
                "children": []
            }

    def _calculate_memory_tokens(self, memory_nodes: List[Dict[str, Any]]) -> int:
        """
        Calculate total tokens in memory nodes using tiktoken-based estimation.

        Args:
            memory_nodes: List of memory VEIL nodes

        Returns:
            Accurate token count using tiktoken
        """
        try:
            return estimate_veil_tokens(memory_nodes)
        except Exception as e:
            logger.warning(f"Error calculating memory tokens with tiktoken: {e}")
            # Fallback to stored token counts or rough estimation
            total_tokens = 0
            for memory_node in memory_nodes:
                node_props = memory_node.get("properties", {})
                # Check for stored token count first
                stored_tokens = node_props.get("token_count")
                if stored_tokens:
                    total_tokens += stored_tokens
                else:
                    # Rough estimate based on memory summary length
                    memory_summary = node_props.get("memory_summary", "")
                    estimated_tokens = len(memory_summary) // 4  # 4 chars per token fallback
                    total_tokens += estimated_tokens
            return total_tokens

    def _calculate_children_tokens(self, children: List[Dict[str, Any]]) -> int:
        """
        Calculate total tokens in VEIL children using tiktoken-based estimation.

        Args:
            children: List of VEIL child nodes

        Returns:
            Accurate token count using tiktoken
        """
        try:
            return estimate_veil_tokens(children)
        except Exception as e:
            logger.warning(f"Error calculating children tokens with tiktoken: {e}")
            # Fallback estimation
            total_chars = 0
            for child in children:
                # Estimate based on text content
                props = child.get("properties", {})
                text_content = props.get("text_content", "")
                total_chars += len(text_content)
            # Rough estimate: 4 characters per token
            return total_chars // 4
    # PHASE 2A: NEW - Dual-Stream Chunk Management Methods

    async def _get_or_create_element_chunks(self, element_id: str, children: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get existing chunk structure or create from current children.
        
        This is the core method that maintains N-stream/M-stream separation for each element.
        
        Args:
            element_id: ID of the element to track
            children: Current VEIL children for this element
            
        Returns:
            Dictionary with n_chunks, m_chunks, metadata
        """
        try:
            if element_id not in self._element_chunks:
                # Initialize chunk structure from current children
                logger.debug(f"Initializing chunk structure for element {element_id}")
                self._element_chunks[element_id] = await self._initialize_chunks_from_children(element_id, children)
            else:
                # Update existing chunks with any new content
                await self._update_chunks_with_new_content(element_id, children)
            
            return self._element_chunks[element_id]
            
        except Exception as e:
            logger.error(f"Error managing chunks for element {element_id}: {e}", exc_info=True)
            # Fallback: create minimal structure
            return {
                "n_chunks": [],
                "m_chunks": [],
                "last_update": datetime.now(),
                "total_tokens": 0,
                "error": str(e)
            }

    async def _initialize_chunks_from_children(self, element_id: str, children: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Initialize N-stream and M-stream from current VEIL children.
        
        This analyzes current children to separate existing memories from fresh content,
        then organizes everything into the dual-stream structure.
        
        Args:
            element_id: ID of the element
            children: Current VEIL children
            
        Returns:
            Initialized chunk structure
        """
        try:
            logger.info(f"🟢 COMPRESSION INIT DEBUG: Starting chunk initialization for {element_id} with {len(children)} children")
            
            # Separate existing memories from fresh content
            existing_memories, fresh_content = self._separate_memories_and_content(children)
            logger.info(f"🟢 COMPRESSION INIT DEBUG: {element_id} - Found {len(existing_memories)} existing memories, {len(fresh_content)} fresh content items")
            
            # Convert fresh content into N-chunks (4k token boundaries)
            n_chunks = await self._content_to_n_chunks(fresh_content)
            logger.info(f"🟢 COMPRESSION INIT DEBUG: {element_id} - Created {len(n_chunks)} N-chunks from fresh content")
            
            # Convert existing memories into M-chunks
            m_chunks = self._memories_to_m_chunks(existing_memories)
            logger.info(f"🟢 COMPRESSION INIT DEBUG: {element_id} - Converted {len(existing_memories)} existing memories to {len(m_chunks)} M-chunks")
            
            # Calculate total tokens
            total_tokens = sum(self._calculate_chunk_tokens(chunk) for chunk in n_chunks)
            total_tokens += sum(self._calculate_chunk_tokens(chunk) for chunk in m_chunks)
            
            chunk_structure = {
                "n_chunks": n_chunks,
                "m_chunks": m_chunks,
                "last_update": datetime.now(),
                "total_tokens": total_tokens,
                "initialization_source": "veil_children"
            }
            
            # DEBUG: Check how many N-chunks are complete and ready for M-chunk creation
            complete_n_chunks = [chunk for chunk in n_chunks if chunk.get("is_complete", False)]
            incomplete_n_chunks = [chunk for chunk in n_chunks if not chunk.get("is_complete", False)]
            
            logger.info(f"🟢 COMPRESSION INIT DEBUG: {element_id} - N-chunk analysis: {len(complete_n_chunks)} complete, {len(incomplete_n_chunks)} incomplete")
            logger.info(f"🟢 COMPRESSION INIT DEBUG: {element_id} - INITIALIZATION COMPLETE - Will need {len(complete_n_chunks)} M-chunks for complete N-chunks")
            
            logger.info(f"Initialized chunks for {element_id}: {len(n_chunks)} N-chunks, {len(m_chunks)} M-chunks, {total_tokens} tokens")
            return chunk_structure
            
        except Exception as e:
            logger.error(f"Error initializing chunks for {element_id}: {e}", exc_info=True)
            return {
                "n_chunks": [],
                "m_chunks": [],
                "last_update": datetime.now(),
                "total_tokens": 0,
                "initialization_error": str(e)
            }

    async def _update_chunks_with_new_content(self, element_id: str, children: List[Dict[str, Any]]) -> None:
        """
        Update existing chunk structure with new content.
        
        This detects new content and adds it to the appropriate stream,
        triggering chunking and compression as needed.
        
        Args:
            element_id: ID of the element
            children: Current VEIL children (may include new content)
        """
        try:
            current_structure = self._element_chunks[element_id]
            
            # FIXED: Check if we need initial processing BEFORE calling _detect_new_content
            # This prevents the flag from being set before we check it
            initialization_source = current_structure.get("initialization_source")
            needs_initial_processing = (initialization_source == "veil_children" and 
                                      not current_structure.get("initial_processing_complete", False))
            
            logger.info(f"🟠 UPDATE CHUNKS DEBUG: {element_id} - initialization_source={initialization_source}, needs_initial_processing={needs_initial_processing}")
            
            # Detect new content since last update
            new_content = await self._detect_new_content(element_id, children)
            
            # Add new content to N-stream if any
            if new_content:
                current_structure["n_chunks"].extend(await self._content_to_n_chunks(new_content))
                logger.info(f"🟠 UPDATE CHUNKS DEBUG: Added {len(new_content)} new content items to N-stream for {element_id}")
            else:
                logger.info(f"🟠 UPDATE CHUNKS DEBUG: No new content detected for element {element_id}")
            
            # FIXED: Always check chunk boundaries, even if no new content
            # This is crucial for initial M-chunk creation after first initialization
            if new_content or needs_initial_processing:
                logger.info(f"🟠 UPDATE CHUNKS DEBUG: {element_id} - Triggering chunk boundary processing (new_content={len(new_content) if new_content else 0}, initial_processing={needs_initial_processing})")
                await self._process_chunk_boundaries(element_id)
                
                # Mark initial processing as complete
                if needs_initial_processing:
                    current_structure["initial_processing_complete"] = True
                    logger.info(f"🟠 UPDATE CHUNKS DEBUG: {element_id} - Marked initial processing as complete")
            else:
                logger.info(f"🟠 UPDATE CHUNKS DEBUG: {element_id} - Skipping chunk boundary processing (no new content, no initial processing needed)")
            
            # Update metadata
            current_structure["last_update"] = datetime.now()
            current_structure["total_tokens"] = self._calculate_total_chunk_tokens(current_structure)
            
            final_n_count = len(current_structure['n_chunks'])
            final_m_count = len(current_structure['m_chunks'])
            logger.info(f"🟠 UPDATE CHUNKS COMPLETE DEBUG: Updated chunks for {element_id}: {final_n_count} N-chunks, {final_m_count} M-chunks")
            
        except Exception as e:
            logger.error(f"Error updating chunks for {element_id}: {e}", exc_info=True)

    async def _content_to_n_chunks(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert VEIL content into N-chunks (4k token boundaries).
        
        Args:
            content: List of VEIL content nodes
            
        Returns:
            List of N-chunk structures
        """
        try:
            if not content:
                return []
            
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for content_item in content:
                item_tokens = self._calculate_children_tokens([content_item])
                
                # Check if adding this item would exceed chunk limit
                if current_tokens + item_tokens > self.COMPRESSION_CHUNK_SIZE and current_chunk:
                    # Complete current chunk
                    chunks.append({
                        "chunk_type": "n_chunk",
                        "content": current_chunk,
                        "token_count": current_tokens,
                        "chunk_index": len(chunks),
                        "created_at": datetime.now(),
                        "is_complete": True
                    })
                    
                    # Start new chunk
                    current_chunk = [content_item]
                    current_tokens = item_tokens
                else:
                    # Add to current chunk
                    current_chunk.append(content_item)
                    current_tokens += item_tokens
            
            # Handle remaining content (possibly incomplete chunk)
            if current_chunk:
                chunks.append({
                    "chunk_type": "n_chunk",
                    "content": current_chunk,
                    "token_count": current_tokens,
                    "chunk_index": len(chunks),
                    "created_at": datetime.now(),
                    "is_complete": current_tokens >= self.COMPRESSION_CHUNK_SIZE
                })
            
            logger.debug(f"Created {len(chunks)} N-chunks from {len(content)} content items")
            return chunks
            
        except Exception as e:
            logger.error(f"Error converting content to N-chunks: {e}", exc_info=True)
            return []

    def _memories_to_m_chunks(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert existing memory nodes into M-chunks.
        
        Args:
            memories: List of existing memory VEIL nodes
            
        Returns:
            List of M-chunk structures
        """
        try:
            m_chunks = []
            
            for i, memory in enumerate(memories):
                m_chunk = {
                    "chunk_type": "m_chunk",
                    "memory_node": memory,
                    "token_count": self._calculate_memory_tokens([memory]),
                    "chunk_index": i,
                    "created_at": datetime.now(),
                    "is_complete": True  # Memories are always complete
                }
                m_chunks.append(m_chunk)
            
            logger.debug(f"Created {len(m_chunks)} M-chunks from {len(memories)} memory nodes")
            return m_chunks
            
        except Exception as e:
            logger.error(f"Error converting memories to M-chunks: {e}", exc_info=True)
            return []

    async def _process_chunk_boundaries(self, element_id: str) -> None:
        """
        Process chunk boundaries and trigger compression as needed.
        
        ENHANCED: Now handles M-chunk invalidation and recompression for content changes.
        - When N-chunks complete (reach 4k), they are compressed to M-chunks for backup using background compression
        - N-chunks are KEPT in N-stream for focused window rendering
        - When content changes (message edits), invalidated M-chunks are recompressed
        - Only when N-stream exceeds storage limits do we physically remove old N-chunks
        - Rendering logic uses window size to decide N-chunks vs M-chunks, not storage availability
        
        Args:
            element_id: ID of the element to process
        """
        try:
            chunk_structure = self._element_chunks[element_id]
            n_chunks = chunk_structure["n_chunks"]
            m_chunks = chunk_structure["m_chunks"]
            
            logger.info(f"🟡 CHUNK BOUNDARY DEBUG: Starting boundary processing for {element_id}: {len(n_chunks)} N-chunks, {len(m_chunks)} M-chunks")
            
            # NEW: Handle invalidated M-chunks that need recompression
            invalidated_m_chunks = [m for m in m_chunks if m.get("is_invalid", False)]
            if invalidated_m_chunks:
                logger.info(f"🟡 CHUNK BOUNDARY DEBUG: Found {len(invalidated_m_chunks)} invalidated M-chunks for {element_id}, recompressing...")
                await self._recompress_invalidated_m_chunks(element_id, invalidated_m_chunks, n_chunks)
            
            # Create M-chunks for complete N-chunks (but KEEP the N-chunks!)
            complete_n_chunks = [chunk for chunk in n_chunks if chunk.get("is_complete", False)]
            
            # Find which complete N-chunks don't have corresponding M-chunks yet
            existing_m_chunk_sources = {m.get("source_n_chunk_index", -1) for m in m_chunks if not m.get("is_invalid", False)}
            new_complete_chunks = [chunk for chunk in complete_n_chunks 
                                 if chunk.get("chunk_index", -1) not in existing_m_chunk_sources]
            
            logger.info(f"🟡 CHUNK BOUNDARY DEBUG: {element_id} - Complete N-chunks: {len(complete_n_chunks)}, Existing M-chunk sources: {existing_m_chunk_sources}, New complete chunks: {len(new_complete_chunks)}")
            
            if new_complete_chunks:
                logger.info(f"🔴 M-CHUNK CREATION DEBUG: Starting M-chunk creation for {element_id} - {len(new_complete_chunks)} new complete N-chunks - BACKGROUND MODE")
                
                # Create M-chunks for new complete N-chunks using TRUE BACKGROUND COMPRESSION
                for chunk in new_complete_chunks:
                    try:
                        # Extract content from N-chunk
                        chunk_content = chunk.get("content", [])
                        if not chunk_content:
                            continue
                        
                        chunk_index = chunk.get('chunk_index', 0)
                        chunk_element_id = f"{element_id}_chunk_{chunk_index}"
                        
                        logger.info(f"🔴 M-CHUNK CREATION DEBUG: Creating M-chunk for {element_id} N-chunk {chunk_index} with {len(chunk_content)} content items")
                        
                        # NEW: Create memory with dependency tracking for cascade invalidation
                        result_node = await self._create_memory_with_dependency_tracking(
                            element_id, chunk_content, chunk_index, chunk_element_id
                        )
                        
                        if result_node:
                            # Convert result to M-chunk format
                            result_props = result_node.get("properties", {})
                            node_type = result_node.get("node_type", "")
                            
                            logger.info(f"🔴 M-CHUNK CREATION DEBUG: Memory creation result for {element_id} N-chunk {chunk_index}: node_type={node_type}")
                            
                            if node_type == "memorized_content":
                                # Completed memory
                                memory_summary = result_props.get("memory_summary", f"Background memory of N-chunk {chunk_index}")
                                compression_approach = "agent_memory_compressor_background_complete"
                                logger.info(f"🟢 M-CHUNK COMPLETE DEBUG: {element_id} N-chunk {chunk_index} - COMPLETED immediately: {memory_summary[:50]}...")
                                
                            elif node_type in ["fresh_content", "trimmed_fresh_content"]:
                                # Background compression in progress, using fresh content fallback
                                memory_summary = f"⏳ Background compression in progress for N-chunk {chunk_index} (using fresh content)"
                                compression_approach = "agent_memory_compressor_background_fallback"
                                logger.info(f"🟡 M-CHUNK PROGRESS DEBUG: {element_id} N-chunk {chunk_index} - Background compression in progress, using fallback")
                                
                            elif node_type == "compression_placeholder":
                                # Background compression starting
                                memory_summary = result_props.get("memory_summary", f"⏳ Starting background compression for N-chunk {chunk_index}")
                                compression_approach = "agent_memory_compressor_background_placeholder"
                                logger.info(f"🟡 M-CHUNK PROGRESS DEBUG: {element_id} N-chunk {chunk_index} - Background compression placeholder created")
                                
                            else:
                                # Unknown result type
                                memory_summary = f"Background compression result for N-chunk {chunk_index}"
                                compression_approach = f"agent_memory_compressor_background_{node_type}"
                                logger.info(f"🟠 M-CHUNK UNKNOWN DEBUG: {element_id} N-chunk {chunk_index} - Unknown result type: {node_type}")
                            
                            # Create memory node from background compression result
                            memory_node = {
                                "veil_id": f"memory_{element_id}_{chunk_index}_{self.id}",
                                "node_type": "content_memory",
                                "properties": {
                                    "structural_role": "compressed_content",
                                    "content_nature": "content_memory",
                                    "original_element_id": element_id,
                                    "memory_summary": memory_summary,
                                    "original_child_count": len(chunk_content),
                                    "compression_timestamp": datetime.now().isoformat(),
                                    "compression_approach": compression_approach,
                                    "is_focused": False,
                                    "is_background_compression": True,  # Now actually true!
                                    "background_compression_type": node_type,
                                    "source_chunk_index": chunk_index,
                                    "source_chunk_tokens": chunk.get('token_count', 0),
                                    # NEW: Store content fingerprint for change detection
                                    "content_fingerprint": self._calculate_content_fingerprint(chunk_content)
                                },
                                "children": []
                            }
                            
                            # CRITICAL FIX: Use content timestamp for proper chronological placement
                            memory_node = self._create_memory_with_content_timestamp(memory_node, chunk_content)
                            
                            logger.debug(f"Background compression result for N-chunk {chunk_index}: {compression_approach}")
                            
                        else:
                            # No result from background compression, create placeholder
                            logger.info(f"🔴 M-CHUNK FAILED DEBUG: Background compression returned None for {element_id} N-chunk {chunk_index}, creating placeholder")
                            memory_summary = f"⏳ Background compression failed to start for N-chunk {chunk_index}"
                            compression_approach = "agent_memory_compressor_background_failed"
                            
                            memory_node = {
                                "veil_id": f"memory_{element_id}_{chunk_index}_{self.id}",
                                "node_type": "content_memory",
                                "properties": {
                                    "structural_role": "compressed_content",
                                    "content_nature": "content_memory",
                                    "original_element_id": element_id,
                                    "memory_summary": memory_summary,
                                    "original_child_count": len(chunk_content),
                                    "compression_timestamp": datetime.now().isoformat(),
                                    "compression_approach": compression_approach,
                                    "is_focused": False,
                                    "is_background_compression": True,
                                    "background_compression_type": "failed",
                                    "source_chunk_index": chunk_index,
                                    "source_chunk_tokens": chunk.get('token_count', 0),
                                    # NEW: Store content fingerprint even for failed compression
                                    "content_fingerprint": self._calculate_content_fingerprint(chunk_content)
                                },
                                "children": []
                            }
                            
                            # CRITICAL FIX: Use content timestamp for proper chronological placement
                            memory_node = self._create_memory_with_content_timestamp(memory_node, chunk_content)
                            
                        if self._memory_compressor:
                            # Add to M-chunks
                            new_m_chunk = {
                                "chunk_type": "m_chunk",
                                "memory_node": memory_node,
                                "token_count": self._calculate_memory_tokens([memory_node]),
                                "chunk_index": len(m_chunks),
                                "created_at": datetime.now(),
                                "is_complete": True,
                                "source_n_chunk_index": chunk_index,
                                "is_background_generated": True
                            }
                            
                            m_chunks.append(new_m_chunk)
                            logger.info(f"🟢 M-CHUNK ADDED DEBUG: Added M-chunk backup for {element_id} N-chunk {chunk_index} - Total M-chunks now: {len(m_chunks)}")
                        
                    except Exception as chunk_error:
                        logger.error(f"🔴 M-CHUNK ERROR DEBUG: Error creating background M-chunk backup for {element_id}: {chunk_error}", exc_info=True)
                        continue
                
                logger.info(f"🟢 M-CHUNK CREATION COMPLETE DEBUG: Background M-chunk creation complete for {element_id}: {len(new_complete_chunks)} new backups created, N-chunks preserved")
            else:
                logger.info(f"🟡 CHUNK BOUNDARY DEBUG: No new complete N-chunks found for {element_id} - no M-chunk creation needed")
            
            # FIXED: Only remove old N-chunks when we exceed storage limits (not window limits!)
            # For now, keep all N-chunks since we have dual-stream storage separation
            # TODO: Implement N-chunk storage limits (separate from rendering window limits)
            
            # Check if M-chunks need re-compression (exceed 1-chunk limit for unfocused rendering)
            valid_m_chunks = [m for m in m_chunks if not m.get("is_invalid", False)]
            total_memory_tokens = sum(chunk.get("token_count", 0) for chunk in valid_m_chunks)
            
            if total_memory_tokens > self.COMPRESSION_CHUNK_SIZE:
                logger.info(f"Valid M-chunks exceed {self.COMPRESSION_CHUNK_SIZE} tokens for {element_id}, triggering re-compression")
                # Re-compress M-chunks into single memory-of-memories
                recompressed_memories = await self._recompress_m_chunks(element_id, valid_m_chunks)
                # Replace valid M-chunks with recompressed ones
                chunk_structure["m_chunks"] = recompressed_memories
            
            # NEW: Update content fingerprint in chunk structure for future change detection
            if n_chunks:
                # Calculate fingerprint from all N-chunk content
                all_n_content = []
                for chunk in n_chunks:
                    all_n_content.extend(chunk.get("content", []))
                chunk_structure["content_fingerprint"] = self._calculate_content_fingerprint(all_n_content)
            
            final_m_count = len(chunk_structure["m_chunks"])
            final_n_count = len(chunk_structure["n_chunks"])
            logger.info(f"🟢 CHUNK BOUNDARY COMPLETE DEBUG: Processed chunk boundaries for {element_id}: {final_n_count} N-chunks (preserved), {final_m_count} M-chunks (background)")
            
        except Exception as e:
            logger.error(f"Error processing chunk boundaries for {element_id}: {e}", exc_info=True)

    async def _recompress_invalidated_m_chunks(self, element_id: str, invalidated_m_chunks: List[Dict[str, Any]], 
                                             n_chunks: List[Dict[str, Any]]) -> None:
        """
        NEW: Recompress M-chunks that were invalidated due to content changes.
        
        When messages are edited, the corresponding M-chunks become stale and need
        to be recompressed with the updated N-chunk content.
        
        Args:
            element_id: ID of the element
            invalidated_m_chunks: List of M-chunks marked as invalid
            n_chunks: Current N-chunks with updated content
        """
        try:
            logger.info(f"Recompressing {len(invalidated_m_chunks)} invalidated M-chunks for {element_id}")
            
            for m_chunk in invalidated_m_chunks:
                try:
                    source_chunk_index = m_chunk.get("source_n_chunk_index", -1)
                    
                    if source_chunk_index == -1 or source_chunk_index >= len(n_chunks):
                        logger.warning(f"Invalid source chunk index {source_chunk_index} for M-chunk, skipping")
                        continue
                    
                    # Get the corresponding N-chunk with updated content
                    source_n_chunk = n_chunks[source_chunk_index]
                    updated_content = source_n_chunk.get("content", [])
                    
                    if not updated_content:
                        logger.warning(f"No content in source N-chunk {source_chunk_index}, skipping M-chunk recompression")
                        continue
                    
                    # NEW: Recompress using AgentMemoryCompressor with temporal context
                    chunk_element_id = f"{element_id}_chunk_{source_chunk_index}_recompressed"
                    
                    if self._memory_compressor:
                        logger.debug(f"Recompressing M-chunk for updated N-chunk {source_chunk_index} with temporal context")
                        
                        # Create enhanced compression context for recompression
                        recompression_context = {
                            "element_id": element_id,
                            "compression_reason": "cascade_recompression",
                            "is_focused": True,
                            "use_background_compression": False,  # Urgent recompression
                            "is_recompression": True,
                            "original_formation_index": m_chunk.get("cascade_metadata", {}).get("global_formation_index", -1),
                            "existing_memory_context": self._get_existing_memory_context_for_agent(element_id),
                            "full_veil_context": await self._get_hud_rendered_context_for_memory_formation(element_id, updated_content)
                        }
                        
                        # Use AgentMemoryCompressor with enhanced context
                        result_node = await self._memory_compressor.compress_nodes(
                            raw_veil_nodes=updated_content,
                            element_ids=[chunk_element_id],
                            compression_context=recompression_context
                        )
                        
                        if result_node:
                            result_props = result_node.get("properties", {})
                            node_type = result_node.get("node_type", "")
                            
                            # Update the memory node in the M-chunk
                            old_memory_node = m_chunk.get("memory_node", {})
                            
                            if node_type == "memorized_content":
                                memory_summary = result_props.get("memory_summary", f"Recompressed memory of N-chunk {source_chunk_index}")
                                compression_approach = "agent_memory_compressor_recompressed_complete"
                                
                            elif node_type in ["fresh_content", "trimmed_fresh_content"]:
                                memory_summary = f"⏳ Recompression in progress for N-chunk {source_chunk_index} (using fresh content)"
                                compression_approach = "agent_memory_compressor_recompressed_fallback"
                                
                            else:
                                memory_summary = f"Recompressed background result for N-chunk {source_chunk_index}"
                                compression_approach = f"agent_memory_compressor_recompressed_{node_type}"
                            
                            # Update the memory node properties
                            updated_memory_node = {
                                "veil_id": f"recompressed_memory_{element_id}_{source_chunk_index}_{self.id}",
                                "node_type": "content_memory",
                                "properties": {
                                    "structural_role": "compressed_content",
                                    "content_nature": "content_memory",
                                    "original_element_id": element_id,
                                    "memory_summary": memory_summary,
                                    "original_child_count": len(updated_content),
                                    "compression_timestamp": datetime.now().isoformat(),
                                    "compression_approach": compression_approach,
                                    "is_focused": False,
                                    "is_background_compression": True,
                                    "is_recompressed": True,  # NEW: Mark as recompressed
                                    "recompression_reason": "content_change_detected",  # NEW: Track why recompressed
                                    "background_compression_type": node_type,
                                    "source_chunk_index": source_chunk_index,
                                    "source_chunk_tokens": source_n_chunk.get('token_count', 0),
                                    # NEW: Update content fingerprint
                                    "content_fingerprint": self._calculate_content_fingerprint(updated_content)
                                },
                                "children": []
                            }
                            
                            # CRITICAL FIX: Use content timestamp for proper chronological placement
                            updated_memory_node = self._create_memory_with_content_timestamp(updated_memory_node, updated_content)
                            
                            # Update the M-chunk
                            m_chunk["memory_node"] = updated_memory_node
                            m_chunk["token_count"] = self._calculate_memory_tokens([updated_memory_node])
                            m_chunk["is_invalid"] = False  # Mark as valid again
                            m_chunk["recompressed_at"] = datetime.now().isoformat()
                            m_chunk.pop("invalidated_at", None)  # Remove invalidation timestamp
                            
                            logger.debug(f"Successfully recompressed M-chunk for N-chunk {source_chunk_index}: {compression_approach}")
                            
                        else:
                            logger.warning(f"Recompression failed for N-chunk {source_chunk_index}, M-chunk remains invalid")
                    else:
                        logger.warning(f"No AgentMemoryCompressor available for recompression, creating simple fallback")
                        
                        # Simple fallback recompression
                        fallback_summary = f"Recompressed simple summary of N-chunk {source_chunk_index}: {len(updated_content)} items"
                        
                        updated_memory_node = {
                            "veil_id": f"fallback_recompressed_{element_id}_{source_chunk_index}_{self.id}",
                            "node_type": "content_memory",
                            "properties": {
                                "structural_role": "compressed_content",
                                "content_nature": "content_memory",
                                "original_element_id": element_id,
                                "memory_summary": fallback_summary,
                                "original_child_count": len(updated_content),
                                "compression_timestamp": datetime.now().isoformat(),
                                "compression_approach": "simple_fallback_recompressed",
                                "is_focused": False,
                                "is_recompressed": True,
                                "recompression_reason": "content_change_detected",
                                "source_chunk_index": source_chunk_index,
                                "content_fingerprint": self._calculate_content_fingerprint(updated_content)
                            },
                            "children": []
                        }
                        
                        # CRITICAL FIX: Use content timestamp for proper chronological placement
                        updated_memory_node = self._create_memory_with_content_timestamp(updated_memory_node, updated_content)
                        
                        m_chunk["memory_node"] = updated_memory_node
                        m_chunk["token_count"] = self._calculate_memory_tokens([updated_memory_node])
                        m_chunk["is_invalid"] = False
                        m_chunk["recompressed_at"] = datetime.now().isoformat()
                        m_chunk.pop("invalidated_at", None)
                        
                except Exception as recompression_error:
                    logger.error(f"Error recompressing M-chunk for N-chunk {source_chunk_index}: {recompression_error}", exc_info=True)
                    # Keep M-chunk marked as invalid if recompression fails
                    continue
            
            # Count successful recompressions
            recompressed_count = len([m for m in invalidated_m_chunks if not m.get("is_invalid", False)])
            logger.info(f"Successfully recompressed {recompressed_count}/{len(invalidated_m_chunks)} M-chunks for {element_id}")
            
        except Exception as e:
            logger.error(f"Error recompressing invalidated M-chunks for {element_id}: {e}", exc_info=True)

    async def _recompress_m_chunks(self, element_id: str, m_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-compress M-chunks when they exceed 1-chunk limit.
        
        Args:
            element_id: ID of the element
            m_chunks: List of M-chunks to re-compress
            
        Returns:
            List with single re-compressed memory chunk
        """
        try:
            if not m_chunks:
                return []
            
            # Extract memory nodes from M-chunks
            memory_nodes = [chunk.get("memory_node", {}) for chunk in m_chunks if chunk.get("memory_node")]
            
            if not memory_nodes:
                return []
            
            # Use existing memory recompression logic
            if self._memory_compressor:
                # Create a summary memory of all the individual memories
                summary_memory = await self._create_memory_summary(memory_nodes, element_id, f"element_{element_id}")
                
                # Convert back to M-chunk format
                recompressed_chunk = {
                    "chunk_type": "m_chunk",
                    "memory_node": summary_memory,
                    "token_count": self._calculate_memory_tokens([summary_memory]),
                    "chunk_index": 0,
                    "created_at": datetime.now(),
                    "is_complete": True,
                    "is_recompressed": True,
                    "original_chunk_count": len(m_chunks)
                }
                
                logger.info(f"Re-compressed {len(m_chunks)} M-chunks into 1 memory-of-memories for {element_id}")
                return [recompressed_chunk]
            else:
                logger.warning(f"No memory compressor available for re-compression of {element_id}")
                return m_chunks[:1]  # Keep only first memory as fallback
                
        except Exception as e:
            logger.error(f"Error re-compressing M-chunks for {element_id}: {e}", exc_info=True)
            return m_chunks[:1]  # Fallback to first memory

    # PHASE 2B: NEW - Stream Selection Logic

    async def _select_rendering_stream(self, element_id: str, is_focused: bool, chunks: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Select which stream to render based on focus state.
        
        FIXED: This implements the correct dual-stream rendering logic:
        - Focused: Show all N-chunks if ≤8, OR show M-chunks for oldest + last 8 N-chunks if >8  
        - Unfocused: Show all M-chunks + current incomplete N-chunk only
        
        Args:
            element_id: ID of the element
            is_focused: Whether this element is currently focused
            chunks: Chunk structure with n_chunks and m_chunks
            
        Returns:
            Flattened list of VEIL nodes ready for rendering
        """
        try:
            n_chunks = chunks.get("n_chunks", [])
            m_chunks = chunks.get("m_chunks", [])
            
            logger.info(f"🔵 STREAM SELECTION DEBUG: {element_id} - is_focused={is_focused}, n_chunks={len(n_chunks)}, m_chunks={len(m_chunks)}")
            
            # Debug chunk states
            complete_n_chunks = [chunk for chunk in n_chunks if chunk.get("is_complete", False)]
            incomplete_n_chunks = [chunk for chunk in n_chunks if not chunk.get("is_complete", False)]
            valid_m_chunks = [m for m in m_chunks if not m.get("is_invalid", False)]
            invalid_m_chunks = [m for m in m_chunks if m.get("is_invalid", False)]
            
            logger.info(f"🔵 STREAM SELECTION DEBUG: {element_id} - N-chunks: {len(complete_n_chunks)} complete, {len(incomplete_n_chunks)} incomplete")
            logger.info(f"🔵 STREAM SELECTION DEBUG: {element_id} - M-chunks: {len(valid_m_chunks)} valid, {len(invalid_m_chunks)} invalid")
            
            if is_focused:
                logger.info(f"🔵 FOCUSED STREAM DEBUG: {element_id} - Processing focused rendering logic")
                
                # FIXED: Focused logic - use window size properly
                if len(n_chunks) <= self.FOCUSED_CHUNK_LIMIT:  # ≤8 N-chunks
                    # All N-chunks fit in window, show all fresh content (no M-chunks needed)
                    fresh_content = n_chunks
                    memory_content = []
                    logger.info(f"🔵 FOCUSED STREAM DEBUG: {element_id} - All {len(n_chunks)} N-chunks fit in {self.FOCUSED_CHUNK_LIMIT}-chunk window, showing all fresh content, NO M-chunks")
                else:
                    # >8 N-chunks: show M-chunks for oldest + last 8 N-chunks for recent
                    # Show last 8 N-chunks as fresh content
                    fresh_content = n_chunks[-self.FOCUSED_CHUNK_LIMIT:]
                    
                    # Determine how many old N-chunks we need M-chunks for
                    num_old_chunks = len(n_chunks) - self.FOCUSED_CHUNK_LIMIT
                    
                    logger.info(f"🔵 FOCUSED STREAM DEBUG: {element_id} - Window overflow: {len(n_chunks)} > {self.FOCUSED_CHUNK_LIMIT}, need M-chunks for {num_old_chunks} old N-chunks")
                    
                    # Get M-chunks for the oldest N-chunks (chronological order)
                    # M-chunks should correspond to the first num_old_chunks N-chunks
                    memory_content = []
                    for i in range(min(num_old_chunks, len(m_chunks))):
                        # Find M-chunk that corresponds to N-chunk index i
                        for m_chunk in m_chunks:
                            if m_chunk.get("source_n_chunk_index", -1) == i and not m_chunk.get("is_invalid", False):
                                memory_content.append(m_chunk)
                                logger.info(f"🔵 FOCUSED STREAM DEBUG: {element_id} - Selected M-chunk for old N-chunk {i}")
                                break
                        else:
                            logger.info(f"🔴 FOCUSED STREAM DEBUG: {element_id} - MISSING M-chunk for old N-chunk {i} - this may cause rendering issues!")
                    
                    logger.info(f"🔵 FOCUSED STREAM DEBUG: {element_id} - Window overflow result: {len(memory_content)} M-chunks (for oldest) + {len(fresh_content)} recent N-chunks")
                
                # Return: memories first, then fresh content (chronological order)
                selected_chunks = memory_content + fresh_content
                logger.info(f"🔵 FOCUSED STREAM DEBUG: {element_id} - Final selection: {len(memory_content)} M-chunks + {len(fresh_content)} N-chunks = {len(selected_chunks)} total chunks")
                
            else:
                logger.info(f"🔵 UNFOCUSED STREAM DEBUG: {element_id} - Processing unfocused rendering logic")
                
                # Unfocused: Show all M-chunks + current incomplete N-chunk only
                incomplete_chunks = [chunk for chunk in n_chunks if not chunk.get("is_complete", False)]
                current_chunk = incomplete_chunks[-1:] if incomplete_chunks else []
                selected_chunks = valid_m_chunks + current_chunk
                
                logger.info(f"🔵 UNFOCUSED STREAM DEBUG: {element_id} - Selected {len(valid_m_chunks)} valid M-chunks + {len(current_chunk)} current incomplete N-chunk = {len(selected_chunks)} total chunks")
            
            # Flatten chunks to VEIL nodes
            results = await self._flatten_chunks_for_rendering(selected_chunks)
            logger.info(f"🔵 STREAM SELECTION COMPLETE DEBUG: {element_id} - Flattened {len(selected_chunks)} chunks to {len(results)} VEIL nodes for {is_focused and 'FOCUSED' or 'UNFOCUSED'} rendering")
            
            return results
        
        except Exception as e:
            logger.info(f"🔴 STREAM SELECTION ERROR: Error selecting rendering stream for {element_id}: {e}", exc_info=True)
            # Fallback: use original children unchanged
            return []

    async def _flatten_chunks_for_rendering(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Flatten chunk structures back into VEIL nodes for rendering.
        
        This converts N-chunks and M-chunks back into the VEIL node format
        that the HUD rendering system expects.
        
      Args:
            chunks: List of chunk structures (mix of N-chunks and M-chunks)
            
        Returns:
            List of VEIL nodes ready for HUD rendering
        """
        try:
            flattened_nodes = []
            
            for chunk in chunks:
                chunk_type = chunk.get("chunk_type", "")
                
                if chunk_type == "n_chunk":
                    # N-chunk: extract content nodes
                    content = chunk.get("content", [])
                    flattened_nodes.extend(content)
                    
                elif chunk_type == "m_chunk":
                    # M-chunk: extract memory node
                    memory_node = chunk.get("memory_node", {})
                    if memory_node:
                        flattened_nodes.append(memory_node)
                        
                else:
                    logger.warning(f"Unknown chunk type: {chunk_type}")
            
            logger.debug(f"Flattened {len(chunks)} chunks into {len(flattened_nodes)} VEIL nodes")
            return flattened_nodes
            
        except Exception as e:
            logger.error(f"Error flattening chunks for rendering: {e}", exc_info=True)
            return []

    # PHASE 3: NEW - HUD-Deferred Context Generation for Enhanced Memory Creation

    async def _get_hud_rendered_context(self, element_id: str, children: List[Dict[str, Any]]) -> Optional[str]:
        """
        PHASE 3: Get memorization context by calling HUD's specialized rendering method.
        
        ENHANCED: Now uses Space's flat VEIL cache directly instead of redundant delta tracking.
        This leverages the existing Space infrastructure and allows for temporal reconstruction
        by applying edit deltas to the flat cache before rendering.
        
        Args:
            element_id: ID of the element being compressed
            children: VEIL children being memorized (to exclude from context)
            
        Returns:
            Rendered memorization context string from HUD, or None if HUD not available
        """
        try:
            # Get HUD component
            hud_component = self._get_hud_component()
            if not hud_component:
                logger.debug(f"HUD component not available for memorization context generation for {element_id}")
                return None
            
            if not self.owner:
                logger.debug(f"No owner InnerSpace available for VEIL access")
                return None
            
            # NEW: Get flat VEIL cache directly from Space instead of using SpaceVeilProducer
            flat_veil_cache = self._get_flat_veil_cache_from_space()
            if not flat_veil_cache:
                logger.debug(f"No flat VEIL cache available from Space")
                return None
            
            # NEW: For temporal consistency, apply any pending edit deltas to the cache copy
            # This allows us to render the "current state" including edits for memorization
            temporal_flat_cache = await self._apply_edit_deltas_to_flat_cache(flat_veil_cache, element_id, children)
            
            # Use HUD's specialized memorization context rendering with flat cache
            memorization_context = await hud_component.render_memorization_context_with_flat_cache(
                flat_veil_cache=temporal_flat_cache,
                exclude_element_id=element_id,
                exclude_content=children,  # Exclude the content being memorized
                focus_element_id=None  # Could pass focus context if available
            )
            
            if memorization_context:
                logger.debug(f"Generated memorization context for {element_id}: {len(memorization_context)} characters")
                return memorization_context
            else:
                logger.debug(f"HUD rendered empty memorization context for {element_id}")
                return None
                
        except Exception as e:
            logger.warning(f"Error getting memorization context for {element_id}: {e}")
            return None

    def _get_space_veil_producer(self):
        """
        NEW: Get SpaceVeilProducer from owner Space for temporal VEIL operations.
        
        This is the proper way to access VEIL data according to the new centralized architecture.
        """
        try:
            if not self.owner:
                return None
            
            # Get SpaceVeilProducer component from owner Space
            return self.owner.get_component('SpaceVeilProducer')
        except Exception as e:
            logger.debug(f"Error accessing SpaceVeilProducer: {e}")
            return None

    def _get_flat_veil_cache_from_space(self) -> Optional[Dict[str, Any]]:
        """
        FIXED: Get flat VEIL cache via SpaceVeilProducer instead of direct Space access.
        
        This properly follows the centralized VEIL architecture where SpaceVeilProducer
        owns all VEIL state management.
        
        Returns:
            Deep copy of SpaceVeilProducer's flat VEIL cache, or None if not available
        """
        try:
            # NEW: Use SpaceVeilProducer instead of direct Space access
            veil_producer = self._get_space_veil_producer()
            if veil_producer:
                flat_cache = veil_producer.get_flat_veil_cache()
                if flat_cache:
                    logger.debug(f"Retrieved flat VEIL cache from SpaceVeilProducer: {len(flat_cache)} nodes")
                    return flat_cache
                else:
                    logger.debug(f"SpaceVeilProducer flat VEIL cache is empty")
                    return None
            else:
                logger.debug(f"SpaceVeilProducer not available for VEIL access")
                return None
                
        except Exception as e:
            logger.warning(f"Error getting flat VEIL cache from SpaceVeilProducer: {e}")
            return None

    async def _apply_edit_deltas_to_flat_cache(self, flat_cache: Dict[str, Any], 
                                             element_id: str, children: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        NEW: Apply edit deltas to flat VEIL cache for temporal consistency during memorization.
        
        When memorizing content that includes message edits, we need to apply those edits
        to the flat cache so the agent sees the current state of the message during memorization.
        
        Args:
            flat_cache: Original flat VEIL cache from Space
            element_id: ID of element being memorized
            children: VEIL children being memorized (may contain edit information)
            
        Returns:
            Modified flat cache with edit deltas applied
        """
        try:
            # Create a working copy of the flat cache
            import copy
            temporal_cache = copy.deepcopy(flat_cache)
            
            # Look for edit information in the children being memorized
            edits_applied = 0
            for child in children:
                props = child.get("properties", {})
                external_id = props.get("external_id")
                
                # If this is a message with edit information, apply the edit to the cache
                if external_id and props.get("content_nature") == "chat_message":
                    # Find the corresponding node in the flat cache
                    for veil_id, cached_node in temporal_cache.items():
                        cached_props = cached_node.get("properties", {})
                        if cached_props.get("external_id") == external_id:
                            # Apply edit by updating the cached node properties
                            # This ensures the memorization context shows the edited message state
                            cached_props.update({
                                "text_content": props.get("text_content", cached_props.get("text_content", "")),
                                "message_status": props.get("message_status", cached_props.get("message_status", "received")),
                                "edit_timestamp": props.get("edit_timestamp"),
                                "edit_count": props.get("edit_count", 0)
                            })
                            edits_applied += 1
                            logger.debug(f"Applied edit delta for message {external_id} in temporal cache")
                            break
            
            if edits_applied > 0:
                logger.debug(f"Applied {edits_applied} edit deltas to temporal flat cache for {element_id}")
            
            return temporal_cache
            
        except Exception as e:
            logger.warning(f"Error applying edit deltas to flat cache: {e}")
            # Return original cache if edit application fails
            return flat_cache

    def _get_hud_component(self):
        """
        Get HUD component from owner InnerSpace to reuse its rendering infrastructure.
        
        Returns:
            HUD component instance or None if not available
        """
        try:
            if not self.owner:
                return None
                
            # Look for HUD component in owner's components
            if hasattr(self.owner, '_components'):
                for component in self.owner._components:
                    if hasattr(component, 'COMPONENT_TYPE') and component.COMPONENT_TYPE == "HUDComponent":
                        return component
            
            # Alternative: check if owner has direct HUD access
            if hasattr(self.owner, '_hud_component'):
                return self.owner._hud_component
            elif hasattr(self.owner, 'get_hud_component'):
                return self.owner.get_hud_component()
                
            return None
            
        except Exception as e:
            logger.debug(f"Error accessing HUD component: {e}")
            return None

    def _calculate_chunk_tokens(self, chunk: Dict[str, Any]) -> int:
        """Calculate tokens in a chunk structure."""
        try:
            # Use stored token count if available
            if "token_count" in chunk:
                return chunk["token_count"]
            
            # Calculate based on chunk type
            if chunk.get("chunk_type") == "n_chunk":
                content = chunk.get("content", [])
                return self._calculate_children_tokens(content)
            elif chunk.get("chunk_type") == "m_chunk":
                memory_node = chunk.get("memory_node", {})
                return self._calculate_memory_tokens([memory_node])
            
            return 0
            
        except Exception as e:
            logger.warning(f"Error calculating chunk tokens: {e}")
            return 0

    def _calculate_total_chunk_tokens(self, chunk_structure: Dict[str, Any]) -> int:
        """Calculate total tokens across all chunks in a structure."""
        try:
            n_tokens = sum(self._calculate_chunk_tokens(chunk) for chunk in chunk_structure.get("n_chunks", []))
            m_tokens = sum(self._calculate_chunk_tokens(chunk) for chunk in chunk_structure.get("m_chunks", []))
            return n_tokens + m_tokens
        except Exception as e:
            logger.warning(f"Error calculating total chunk tokens: {e}")
            return 0

    async def _detect_new_content(self, element_id: str, children: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect new content since last update.
        
        ENHANCED: Now implements proper content change detection for message edits.
        When messages are edited, this detects the content change and triggers recompression.
        
        Args:
            element_id: ID of the element
            children: Current VEIL children
            
        Returns:
            List of new content items (or all content if changes detected)
        """
        try:
            current_structure = self._element_chunks.get(element_id, {})
            last_update = current_structure.get("last_update")
            initialization_source = current_structure.get("initialization_source")
            
            if not last_update:
                # First time, all content is new
                logger.debug(f"First time processing element {element_id}, all content is new")
                return children
            
            # FIXED: After first initialization from VEIL children, we need to trigger
            # chunk boundary processing once to create M-chunks for complete N-chunks
            if initialization_source == "veil_children" and not current_structure.get("initial_processing_complete", False):
                logger.debug(f"Triggering initial chunk boundary processing for {element_id}")
                # Mark that we've done initial processing to avoid infinite loops
                current_structure["initial_processing_complete"] = True
                # Return empty list but ensure _process_chunk_boundaries gets called by the caller
                # The caller will process boundaries regardless of new content when this flag is set
                return []
            
            # NEW: Content change detection for message edits
            content_changed = await self._detect_content_changes(element_id, children, current_structure)
            
            if content_changed:
                logger.info(f"Content changes detected for element {element_id}, triggering recompression")
                # Return all content to force reprocessing
                return children
            else:
                logger.debug(f"No content changes detected for element {element_id}")
                return []
            
        except Exception as e:
            logger.error(f"Error detecting new content for {element_id}: {e}", exc_info=True)
            return []

    async def _detect_content_changes(self, element_id: str, current_children: List[Dict[str, Any]], 
                                    current_structure: Dict[str, Any]) -> bool:
        """
        NEW: Detect if content has changed since last compression by comparing fingerprints.
        
        This is critical for message edit scenarios where N-chunk content changes and 
        corresponding M-chunks need to be recompressed to maintain data consistency.
        
        Args:
            element_id: ID of the element being checked
            current_children: Current VEIL children
            current_structure: Current chunk structure with N-chunks and M-chunks
            
        Returns:
            True if content has changed and needs recompression
        """
        try:
            n_chunks = current_structure.get("n_chunks", [])
            
            if not n_chunks:
                # No existing chunks to compare against
                logger.debug(f"No existing N-chunks for {element_id}, no changes to detect")
                return False
            
            # Calculate current content fingerprint
            current_fingerprint = self._calculate_content_fingerprint(current_children)
            
            # Get stored fingerprint from chunk structure
            stored_fingerprint = current_structure.get("content_fingerprint")
            
            if not stored_fingerprint:
                logger.debug(f"No stored fingerprint for {element_id}, assuming content changed")
                # Store current fingerprint for future comparisons
                current_structure["content_fingerprint"] = current_fingerprint
                return True
            
            # Compare fingerprints
            content_changed = stored_fingerprint != current_fingerprint
            
            if content_changed:
                logger.info(f"Content fingerprint changed for {element_id}: {stored_fingerprint[:16]}... → {current_fingerprint[:16]}...")
                # Update stored fingerprint
                current_structure["content_fingerprint"] = current_fingerprint
                
                # NEW: Invalidate affected M-chunks for recompression
                await self._invalidate_affected_m_chunks(element_id, current_children, current_structure)
                
                return True
            else:
                logger.debug(f"Content fingerprint unchanged for {element_id}: {current_fingerprint[:16]}...")
                return False
                
        except Exception as e:
            logger.error(f"Error detecting content changes for {element_id}: {e}", exc_info=True)
            return True  # Assume changed on error to ensure fresh compression

    async def _invalidate_affected_m_chunks(self, element_id: str, current_children: List[Dict[str, Any]], 
                                          current_structure: Dict[str, Any]) -> None:
        """
        ENHANCED: Invalidate M-chunks and trigger cascade invalidation for cross-container dependencies.
        
        When content changes (like message edits), we need to:
        1. Rebuild N-chunks from current content
        2. Mark corresponding M-chunks as invalid/stale 
        3. Extract invalidated memory IDs and trigger cascade invalidation
        4. Schedule background recompression of affected M-chunks
        
        Args:
            element_id: ID of the element
            current_children: Current VEIL children
            current_structure: Current chunk structure
        """
        try:
            logger.info(f"Invalidating affected M-chunks for {element_id} due to content changes")
            
            # Separate current content into memories and fresh content
            existing_memories, fresh_content = self._separate_memories_and_content(current_children)
            
            # Rebuild N-chunks from current fresh content
            new_n_chunks = await self._content_to_n_chunks(fresh_content)
            
            # Get existing chunk structures
            old_n_chunks = current_structure.get("n_chunks", [])
            old_m_chunks = current_structure.get("m_chunks", [])
            
            # Compare old vs new N-chunks to find which ones changed
            changed_chunk_indices = self._find_changed_chunk_indices(old_n_chunks, new_n_chunks)
            
            if changed_chunk_indices:
                logger.info(f"Found {len(changed_chunk_indices)} changed N-chunks for {element_id}: {changed_chunk_indices}")
                
                # Mark corresponding M-chunks as invalid and collect invalidated memory IDs
                invalidated_m_chunks = []
                invalidated_memory_ids = []
                
                for m_chunk in old_m_chunks:
                    source_chunk_index = m_chunk.get("source_n_chunk_index", -1)
                    if source_chunk_index in changed_chunk_indices:
                        # Mark this M-chunk as invalid
                        m_chunk["is_invalid"] = True
                        m_chunk["invalidated_at"] = datetime.now().isoformat()
                        m_chunk["invalidation_reason"] = "content_change"
                        invalidated_m_chunks.append(m_chunk)
                        
                        # Extract memory ID for cascade processing
                        memory_node = m_chunk.get("memory_node", {})
                        memory_id = memory_node.get("veil_id")
                        if memory_id:
                            invalidated_memory_ids.append(memory_id)
                        
                        logger.debug(f"Invalidated M-chunk for N-chunk {source_chunk_index}")
                
                # Update the chunk structure with new N-chunks
                current_structure["n_chunks"] = new_n_chunks
                
                # NEW: Trigger cascade invalidation for cross-container dependencies
                if invalidated_memory_ids:
                    await self._trigger_cascade_invalidation(
                        trigger_element_id=element_id,
                        invalidated_memory_ids=invalidated_memory_ids,
                        trigger_reason="content_change"
                    )
                    logger.info(f"Triggered cascade invalidation for {len(invalidated_memory_ids)} invalidated memories")
                
                logger.info(f"Invalidated {len(invalidated_m_chunks)} M-chunks for recompression and triggered cascade processing")
            else:
                logger.debug(f"No specific N-chunk changes detected, updating N-chunks anyway")
                # Update N-chunks even if no specific changes detected
                current_structure["n_chunks"] = new_n_chunks
                
        except Exception as e:
            logger.error(f"Error invalidating affected M-chunks for {element_id}: {e}", exc_info=True)

    def _find_changed_chunk_indices(self, old_n_chunks: List[Dict[str, Any]], 
                                   new_n_chunks: List[Dict[str, Any]]) -> List[int]:
        """
        NEW: Find which N-chunk indices have changed content.
        
        Compares old vs new N-chunks to identify which specific chunks were affected
        by content changes (like message edits).
        
        Args:
            old_n_chunks: Previous N-chunk structures
            new_n_chunks: New N-chunk structures
            
        Returns:
            List of chunk indices that have changed
        """
        try:
            changed_indices = []
            
            # Compare chunks up to the minimum length
            min_length = min(len(old_n_chunks), len(new_n_chunks))
            
            for i in range(min_length):
                old_chunk = old_n_chunks[i]
                new_chunk = new_n_chunks[i]
                
                # Compare content fingerprints of individual chunks
                old_content = old_chunk.get("content", [])
                new_content = new_chunk.get("content", [])
                
                old_fingerprint = self._calculate_content_fingerprint(old_content)
                new_fingerprint = self._calculate_content_fingerprint(new_content)
                
                if old_fingerprint != new_fingerprint:
                    changed_indices.append(i)
                    logger.debug(f"N-chunk {i} changed: {old_fingerprint[:16]}... → {new_fingerprint[:16]}...")
            
            # If new chunks were added or removed, mark the affected indices
            if len(new_n_chunks) != len(old_n_chunks):
                logger.debug(f"N-chunk count changed: {len(old_n_chunks)} → {len(new_n_chunks)}")
                # Mark all chunks from the divergence point onwards as changed
                for i in range(min_length, max(len(old_n_chunks), len(new_n_chunks))):
                    if i < len(old_n_chunks) or i < len(new_n_chunks):
                        changed_indices.append(i)
            
            return list(set(changed_indices))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error finding changed chunk indices: {e}", exc_info=True)
            # Return all indices as changed on error to be safe
            return list(range(max(len(old_n_chunks), len(new_n_chunks))))

    async def process_flat_veil_with_compression(self,
                                               flat_veil_cache: Dict[str, Any],
                                               focus_context: Optional[Dict[str, Any]] = None,
                                               memory_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        NEW: Process flat VEIL cache directly with compression while avoiding hierarchy reconstruction.
        
        This is more efficient than the hierarchical approach since it works directly with
        Space's flat cache and only reconstructs hierarchy when needed for rendering.
        
        Args:
            flat_veil_cache: Flat VEIL cache from Space 
            focus_context: Optional focus context with focus_element_id
            memory_data: Optional memory data for integration
            
        Returns:
            Processed flat VEIL cache ready for HUD rendering
        """
        focus_element_id = focus_context.get('focus_element_id') if focus_context else None

        with tracer.start_as_current_span("compression_engine.process_flat_veil", attributes={
            "veil.focus_element_id": focus_element_id or "none",
            "veil.has_memory_data": bool(memory_data),
            "veil.cache_size": len(flat_veil_cache)
        }) as span:
            try:
                if not flat_veil_cache:
                    logger.warning(f"CompressionEngine received empty flat VEIL cache for processing")
                    span.set_attribute("veil.status", "empty_cache")
                    return flat_veil_cache

                logger.info(f"Processing flat VEIL cache with compression: {len(flat_veil_cache)} nodes, focus={focus_element_id or 'none'}")
                
                # Process the flat cache with dual-stream logic
                processed_flat_cache = await self._process_flat_cache_with_dual_streams(
                    flat_veil_cache, focus_element_id, memory_data
                )

                logger.debug(f"Flat VEIL processing complete: {len(processed_flat_cache)} nodes processed")
                span.set_status(trace.Status(trace.StatusCode.OK))
                return processed_flat_cache

            except Exception as e:
                logger.error(f"Error processing flat VEIL with compression: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Flat VEIL processing failed"))
                # Fallback to original flat cache
                return flat_veil_cache

    async def _process_flat_cache_with_dual_streams(self, 
                                                   flat_cache: Dict[str, Any],
                                                   focus_element_id: Optional[str],
                                                   memory_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        NEW: Process flat VEIL cache using dual-stream logic without full hierarchy reconstruction.
        
        This applies compression and stream selection directly to the flat cache,
        which is more efficient than reconstructing the full hierarchy.
        
        Args:
            flat_cache: Original flat VEIL cache from Space
            focus_element_id: Optional focused element ID
            memory_data: Optional memory context data
            
        Returns:
            Processed flat cache with compression applied
        """
        try:
            import copy
            processed_cache = copy.deepcopy(flat_cache)
            
            # Find container nodes that need dual-stream processing
            container_nodes = {}
            for veil_id, node_data in processed_cache.items():
                if isinstance(node_data, dict):
                    props = node_data.get("properties", {})
                    structural_role = props.get("structural_role", "")
                    element_id = props.get("element_id")
                    
                    if structural_role == "container" and element_id:
                        container_nodes[element_id] = veil_id
            
            # Process each container with dual-stream logic
            for element_id, container_veil_id in container_nodes.items():
                try:
                    container_node = processed_cache.get(container_veil_id)
                    if not container_node:
                        continue
                    
                    # Check if this is a scratchpad container
                    if self._is_scratchpad_container(container_node):
                        logger.debug(f"Skipping dual-stream processing for scratchpad container: {element_id}")
                        continue
                    
                    # Get children of this container from flat cache
                    children = self._get_container_children_from_flat_cache(processed_cache, container_veil_id)
                    
                    # Apply dual-stream processing
                    is_focused = (element_id == focus_element_id)
                    selected_children = await self._apply_dual_stream_selection_to_flat_cache(
                        element_id, children, is_focused, processed_cache
                    )
                    
                    # CRITICAL FIX: Update the flat cache with selected children!
                    # This was missing and caused M-chunks to be lost
                    await self._update_flat_cache_with_selected_children(
                        processed_cache, container_veil_id, selected_children
                    )
                    
                    # Update container's tool metadata
                    available_tools = container_node.get("properties", {}).get("available_tools", [])
                    if available_tools:
                        container_node["properties"]["tools_available_despite_dual_stream"] = True
                        container_node["properties"]["dual_stream_focus_state"] = "focused" if is_focused else "unfocused"
                    
                    logger.debug(f"Applied dual-stream processing to container {element_id}: {len(selected_children)} selected nodes ({is_focused and 'focused' or 'unfocused'})")
                    
                except Exception as container_error:
                    logger.error(f"Error processing container {element_id}: {container_error}", exc_info=True)
                    continue
            
            return processed_cache
            
        except Exception as e:
            logger.error(f"Error in flat cache dual-stream processing: {e}", exc_info=True)
            return flat_cache

    def _get_container_children_from_flat_cache(self, flat_cache: Dict[str, Any], container_veil_id: str) -> List[Dict[str, Any]]:
        """
        NEW: Get children of a container from flat cache without reconstructing full hierarchy.
        
        Args:
            flat_cache: Flat VEIL cache
            container_veil_id: VEIL ID of the container
            
        Returns:
            List of child VEIL nodes
        """
        try:
            children = []
            for veil_id, node_data in flat_cache.items():
                if isinstance(node_data, dict):
                    # Check if this node is a child of the container
                    props = node_data.get("properties", {})
                    parent_id = props.get("parent_id") or node_data.get("parent_id")
                    
                    if parent_id == container_veil_id:
                        children.append(node_data)
            
            logger.debug(f"Found {len(children)} children for container {container_veil_id}")
            return children
            
        except Exception as e:
            logger.error(f"Error getting container children from flat cache: {e}", exc_info=True)
            return []

    async def _apply_dual_stream_selection_to_flat_cache(self, element_id: str, children: List[Dict[str, Any]], 
                                                       is_focused: bool, flat_cache: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        NEW: Apply dual-stream selection logic to flat cache children.
        
        This uses the same logic as the hierarchical approach but works directly
        with flat cache data structures.
        
        Args:
            element_id: ID of the element being processed
            children: Child nodes from flat cache
            is_focused: Whether this element is focused
            flat_cache: Full flat cache for context
            
        Returns:
            Selected children according to dual-stream logic
        """
        try:
            # Get or create chunk structure for this element
            chunks = await self._get_or_create_element_chunks(element_id, children)
            
            # Select which stream to render based on focus state
            selected_nodes = await self._select_rendering_stream(element_id, is_focused, chunks)
            
            # DEBUG: Check if M-chunks are in the selected nodes
            memory_nodes_count = sum(1 for node in selected_nodes if node.get("node_type") == "content_memory")
            n_chunk_nodes_count = len(selected_nodes) - memory_nodes_count
            
            logger.info(f"🔵 FLAT CACHE DUAL-STREAM DEBUG: {element_id} - Selected {len(selected_nodes)} nodes: {memory_nodes_count} memory nodes, {n_chunk_nodes_count} N-chunk nodes ({is_focused and 'focused' or 'unfocused'})")
            return selected_nodes
            
        except Exception as e:
            logger.error(f"Error applying dual-stream selection to flat cache: {e}", exc_info=True)
            return children  # Fallback to original children

    def _find_space_root_in_flat_cache(self, flat_cache: Dict[str, Any]) -> Optional[str]:
        """
        NEW: Find the space root node ID in flat cache.
        
        Args:
            flat_cache: Flat VEIL cache to search
            
        Returns:
            Space root VEIL ID or None if not found
        """
        try:
            for veil_id, node_data in flat_cache.items():
                if isinstance(node_data, dict):
                    node_type = node_data.get("node_type", "")
                    if node_type == "space_root":
                        return veil_id
            
            # Fallback: look for root structural role
            for veil_id, node_data in flat_cache.items():
                if isinstance(node_data, dict):
                    props = node_data.get("properties", {})
                    if props.get("structural_role") == "root":
                        return veil_id
            
            logger.warning(f"Could not find space root in flat cache")
            return None
            
        except Exception as e:
            logger.error(f"Error finding space root in flat cache: {e}", exc_info=True)
            return None

    async def compress_with_temporal_context(self, 
                                           element_id: str, 
                                           memory_formation_index: int) -> Optional[Dict[str, Any]]:
        """
        NEW: Create memory using temporally consistent context via SpaceVeilProducer.
        
        This implements the temporal memory recompression strategy where:
        1. Historical VEIL state is reconstructed at memory formation time
        2. Future edits are applied for final appearance
        3. Temporal context is rendered excluding the content being compressed
        4. Memory is compressed using historical timestamp for consistency
        
        All temporal complexity is handled by SpaceVeilProducer.
        
        Args:
            element_id: ID of element being compressed
            memory_formation_index: Delta index when memory was originally formed
            
        Returns:
            Compressed memory node with temporal consistency, or None if failed
        """
        try:
            # Get SpaceVeilProducer for temporal operations
            veil_producer = self._get_space_veil_producer()
            if not veil_producer:
                logger.warning(f"SpaceVeilProducer not available for temporal compression of {element_id}")
                # Fallback to current state compression
                return await self._compress_with_current_state(element_id)
            
            # Use SpaceVeilProducer for temporal context rendering
            logger.info(f"Creating temporal memory for {element_id} at formation index {memory_formation_index}")
            temporal_context = await veil_producer.render_temporal_context_for_compression(
                element_id=element_id,
                memory_formation_index=memory_formation_index
            )
            
            if not temporal_context:
                logger.warning(f"No temporal context generated for {element_id}, falling back to current state")
                return await self._compress_with_current_state(element_id)
            
            # Compress with temporally consistent context using memory compressor
            if self._memory_compressor:
                # Create compression context for temporal compression
                compression_context = {
                    "element_id": element_id,
                    "compression_reason": "temporal_memory_formation",
                    "memory_formation_index": memory_formation_index,
                    "temporal_context": temporal_context,
                    "is_temporal_compression": True
                }
                
                # Use temporal context to create memory with proper timestamp
                memory_node = await self._memory_compressor.compress_with_context(
                    content_context=temporal_context,
                    compression_context=compression_context,
                    compression_timestamp=memory_formation_index  # Use historical timestamp!
                )
                
                if memory_node:
                    # Add temporal consistency metadata
                    memory_props = memory_node.get("properties", {})
                    memory_props.update({
                        "temporal_consistency": {
                            "uses_historical_state": True,
                            "applies_future_edits": True,
                            "formation_context_preserved": True,
                            "processed_via_space_veil_producer": True,
                            "memory_formation_delta_index": memory_formation_index
                        },
                        "compression_approach": "temporal_space_veil_producer",
                        "compression_timestamp": memory_formation_index  # Historical timestamp
                    })
                    
                    logger.info(f"Successfully created temporal memory for {element_id} with formation index {memory_formation_index}")
                    return memory_node
                else:
                    logger.error(f"Memory compressor failed for temporal compression of {element_id}")
                    return None
            else:
                logger.warning(f"No memory compressor available for temporal compression of {element_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error in temporal memory compression for {element_id}: {e}", exc_info=True)
            # Fallback to current state compression
            return await self._compress_with_current_state(element_id)

    async def _compress_with_current_state(self, element_id: str) -> Optional[Dict[str, Any]]:
        """
        Fallback compression using current state when temporal compression fails.
        
        Args:
            element_id: ID of element to compress
            
        Returns:
            Compressed memory node using current state
        """
        try:
            logger.debug(f"Using fallback current state compression for {element_id}")
            
            # Get current VEIL state
            flat_cache = self._get_flat_veil_cache_from_space()
            if not flat_cache:
                logger.warning(f"No VEIL cache available for fallback compression of {element_id}")
                return None
            
            # Find element's current children
            element_children = []
            for veil_id, node_data in flat_cache.items():
                if isinstance(node_data, dict):
                    props = node_data.get("properties", {})
                    if props.get("element_id") == element_id:
                        # This is a container for our element, get its children
                        element_children.extend(self._get_container_children_from_flat_cache(flat_cache, veil_id))
                        break
            
            if not element_children:
                logger.warning(f"No children found for element {element_id} in current state")
                return None
            
            # Use memory compressor for fallback compression
            if self._memory_compressor:
                compression_context = {
                    "element_id": element_id,
                    "compression_reason": "fallback_current_state",
                    "is_temporal_compression": False
                }
                
                memory_node = await self._memory_compressor.compress(
                    raw_veil_nodes=element_children,
                    element_ids=[element_id],
                    compression_context=compression_context
                )
                
                if memory_node:
                    # Mark as fallback compression
                    memory_props = memory_node.get("properties", {})
                    memory_props.update({
                        "compression_approach": "fallback_current_state",
                        "temporal_consistency": {
                            "uses_historical_state": False,
                            "applies_future_edits": False,
                            "formation_context_preserved": False,
                            "is_fallback_compression": True
                        }
                    })
                    
                    logger.debug(f"Created fallback memory for {element_id}")
                    return memory_node
                else:
                    logger.error(f"Memory compressor failed for fallback compression of {element_id}")
                    return None
            else:
                logger.warning(f"No memory compressor available for fallback compression of {element_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error in fallback compression for {element_id}: {e}", exc_info=True)
            return None

    async def get_temporal_veil_at_delta_index(self, delta_index: int) -> Optional[Dict[str, Any]]:
        """
        NEW: Get temporal VEIL state at specific delta index via SpaceVeilProducer.
        
        This provides access to historical VEIL states for temporal operations.
        
        Args:
            delta_index: Delta index to reconstruct VEIL state at
            
        Returns:
            Reconstructed flat VEIL cache at specified delta index
        """
        try:
            veil_producer = self._get_space_veil_producer()
            if veil_producer:
                return await veil_producer.reconstruct_veil_state_at_delta_index(delta_index)
            else:
                logger.warning(f"SpaceVeilProducer not available for temporal VEIL reconstruction")
                return None
        except Exception as e:
            logger.error(f"Error getting temporal VEIL at delta index {delta_index}: {e}", exc_info=True)
            return None

    async def _update_flat_cache_with_selected_children(self, 
                                                       flat_cache: Dict[str, Any], 
                                                       container_veil_id: str, 
                                                       selected_children: List[Dict[str, Any]]) -> None:
        """
        CRITICAL FIX: Update flat cache with selected children from dual-stream processing.
        
        This was the missing piece that caused M-chunks to be lost in flat cache approach.
        When dual-stream selection picks M-chunks, we need to update the flat cache so that
        HUD's hierarchy reconstruction uses the compressed nodes instead of original N-chunks.
        
        Args:
            flat_cache: The processed flat cache to update
            container_veil_id: VEIL ID of the container whose children we're updating
            selected_children: Selected children from dual-stream processing (may include M-chunks)
        """
        try:
            # First, remove all existing children of this container from flat cache
            children_to_remove = []
            for veil_id, node_data in flat_cache.items():
                if isinstance(node_data, dict):
                    props = node_data.get("properties", {})
                    parent_id = props.get("parent_id") or node_data.get("parent_id")
                    
                    if parent_id == container_veil_id:
                        children_to_remove.append(veil_id)
            
            # Remove old children
            for veil_id in children_to_remove:
                del flat_cache[veil_id]
                logger.debug(f"Removed old child {veil_id} from flat cache")
            
            # Add selected children to flat cache
            memory_nodes_added = 0
            for child in selected_children:
                child_veil_id = child.get("veil_id")
                if child_veil_id:
                    # Ensure parent_id is set correctly
                    child_props = child.get("properties", {})
                    child_props["parent_id"] = container_veil_id
                    
                    # Add to flat cache
                    flat_cache[child_veil_id] = child
                    
                    # Track memory nodes
                    if child.get("node_type") == "content_memory":
                        memory_nodes_added += 1
                        logger.info(f"🟢 FLAT CACHE UPDATE DEBUG: Added M-chunk memory node {child_veil_id} to flat cache (parent: {container_veil_id})")
                    else:
                        logger.debug(f"Added selected child {child_veil_id} to flat cache (parent: {container_veil_id})")
                else:
                    logger.warning(f"Selected child missing veil_id, cannot add to flat cache: {child}")
            
            logger.info(f"🟢 FLAT CACHE UPDATE DEBUG: Updated flat cache for container {container_veil_id}: removed {len(children_to_remove)} old children, added {len(selected_children)} selected children ({memory_nodes_added} memory nodes)")
            
        except Exception as e:
            logger.error(f"Error updating flat cache with selected children for container {container_veil_id}: {e}", exc_info=True)

    # NEW: Content Timestamp Extraction for Proper Chronological Placement

    def _extract_content_timestamp_range(self, content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract timestamp range from content being compressed.
        
        For proper chronological rendering, memories should be placed at the time
        of their most recent content, not the compression time.
        
        Args:
            content: List of VEIL content nodes
            
        Returns:
            Dictionary with 'earliest_timestamp', 'latest_timestamp', 'has_timestamps'
        """
        try:
            timestamps = []
            
            for item in content:
                props = item.get("properties", {})
                
                # Extract timestamp from various possible fields
                timestamp_value = None
                timestamp_fields = ["timestamp_iso", "timestamp", "created_at", "modified_at"]
                
                for field in timestamp_fields:
                    if field in props:
                        timestamp_value = props[field]
                        break
                
                if timestamp_value:
                    # Convert to unix timestamp for comparison
                    unix_timestamp = self._convert_to_unix_timestamp(timestamp_value)
                    if unix_timestamp is not None:
                        timestamps.append(unix_timestamp)
            
            if not timestamps:
                return {
                    "earliest_timestamp": None,
                    "latest_timestamp": None,
                    "has_timestamps": False,
                    "timestamp_count": 0
                }
            
            return {
                "earliest_timestamp": min(timestamps),
                "latest_timestamp": max(timestamps),
                "has_timestamps": True,
                "timestamp_count": len(timestamps),
                "content_timespan_seconds": max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Error extracting content timestamp range: {e}", exc_info=True)
            return {
                "earliest_timestamp": None,
                "latest_timestamp": None,
                "has_timestamps": False,
                "timestamp_count": 0,
                "error": str(e)
            }

    def _convert_to_unix_timestamp(self, timestamp_value: Any) -> Optional[float]:
        """
        Convert various timestamp formats to unix timestamp.
        
        Args:
            timestamp_value: Timestamp in various formats (ISO string, unix float, etc.)
            
        Returns:
            Unix timestamp as float, or None if conversion fails
        """
        try:
            if isinstance(timestamp_value, (int, float)):
                # Already a unix timestamp
                return float(timestamp_value)
            
            elif isinstance(timestamp_value, str) and timestamp_value:
                # Try to parse ISO format string
                from datetime import datetime
                
                # Handle various ISO format variations
                timestamp_str = timestamp_value.replace('Z', '+00:00')
                
                # Try different parsing approaches
                try:
                    dt = datetime.fromisoformat(timestamp_str)
                    return dt.timestamp()
                except ValueError:
                    # Try parsing without microseconds
                    try:
                        dt = datetime.strptime(timestamp_value, "%Y-%m-%dT%H:%M:%S")
                        return dt.timestamp()
                    except ValueError:
                        # Try parsing with Z suffix
                        try:
                            dt = datetime.strptime(timestamp_value, "%Y-%m-%dT%H:%M:%SZ")
                            return dt.timestamp()
                        except ValueError:
                            logger.warning(f"Could not parse timestamp string: {timestamp_value}")
                            return None
            
            return None
            
        except Exception as e:
            logger.warning(f"Error converting timestamp {timestamp_value}: {e}")
            return None

    # NEW: Operation Index Extraction for Chronological Placement

    def _extract_latest_operation_index(self, content: List[Dict[str, Any]]) -> int:
        """
        Extract the latest operation_index from content being compressed.
        
        NOTE: This method is now deprecated in favor of timestamp-based chronological placement.
        The AgentMemoryCompressor now extracts content timestamps for more accurate positioning.
        
        Args:
            content: List of VEIL content nodes
            
        Returns:
            Latest operation_index from content, or 0 if none found
        """
        try:
            latest_index = 0
            
            for item in content:
                props = item.get("properties", {})
                op_index = props.get("operation_index")
                
                if op_index is not None and op_index > latest_index:
                    latest_index = op_index
            
            logger.debug(f"Extracted latest operation_index from content: {latest_index}")
            return latest_index
            
        except Exception as e:
            logger.error(f"Error extracting operation_index from content: {e}")
            return 0

    def _assign_memory_operation_index(self, memory_node: Dict[str, Any], content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assign operation_index to memory based on latest content operation_index.
        
        This ensures memories appear at the chronological position of their content.
        
        Args:
            memory_node: Memory node to update
            content: Original content that was compressed
            
        Returns:
            Updated memory node with proper operation_index
        """
        try:
            # Get latest operation_index from compressed content
            latest_op_index = self._extract_latest_operation_index(content)
            
            if latest_op_index > 0:
                # Assign memory to take the position of the latest content
                memory_props = memory_node.get("properties", {})
                memory_props["operation_index"] = latest_op_index
                memory_props["replaces_content_through_index"] = latest_op_index
                memory_props["is_chronologically_positioned"] = True
                
                logger.debug(f"Memory assigned operation_index {latest_op_index} from compressed content")
            else:
                # Fallback: use current operation_index from SpaceVeilProducer
                veil_producer = self._get_space_veil_producer()
                if veil_producer:
                    memory_props = memory_node.get("properties", {})
                    fallback_index = getattr(veil_producer, '_next_delta_index', 0)
                    memory_props["operation_index"] = fallback_index
                    memory_props["is_chronologically_positioned"] = False
                    memory_props["fallback_reason"] = "no_operation_index_in_content"
                    logger.debug(f"Memory assigned fallback operation_index {fallback_index}")
        
            return memory_node
            
        except Exception as e:
            logger.error(f"Error assigning memory operation_index: {e}")
            return memory_node

    def _get_space_veil_producer(self):
        """Get SpaceVeilProducer from owner InnerSpace for operation_index access."""
        try:
            if self.owner:
                return self.owner.get_sibling_component("SpaceVeilProducer")
            return None
        except Exception as e:
            logger.warning(f"Error getting SpaceVeilProducer: {e}")
            return None

    def _create_memory_with_content_timestamp(self, 
                                             memory_node: Dict[str, Any], 
                                             content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update memory node to use content timestamp AND operation_index for proper chronological placement.
        
        UPDATED: Now handles both timestamp and operation_index for chronological positioning.
        
        Args:
            memory_node: The memory node to update
            content: Original content that was compressed
            
        Returns:
            Updated memory node with proper timestamp and operation_index
        """
        try:
            # Extract timestamp range from content
            timestamp_info = self._extract_content_timestamp_range(content)
            
            memory_props = memory_node.get("properties", {})
            
            if timestamp_info["has_timestamps"]:
                # Use the latest content timestamp for chronological placement
                latest_timestamp = timestamp_info["latest_timestamp"]
                earliest_timestamp = timestamp_info["earliest_timestamp"]
                
                # Set the main timestamp field for chronological ordering
                memory_props["timestamp"] = latest_timestamp
                memory_props["timestamp_iso"] = datetime.fromtimestamp(latest_timestamp).isoformat() + "Z"
                
                # Keep compression timestamp as metadata
                if "compression_timestamp" in memory_props:
                    memory_props["compression_metadata"] = memory_props.get("compression_metadata", {})
                    memory_props["compression_metadata"]["compression_timestamp"] = memory_props["compression_timestamp"]
                
                # Replace compression_timestamp with content timestamp for chronological placement
                memory_props["compression_timestamp"] = datetime.fromtimestamp(latest_timestamp).isoformat() + "Z"
                
                # Add temporal metadata
                memory_props["temporal_info"] = {
                    "content_earliest_timestamp": earliest_timestamp,
                    "content_latest_timestamp": latest_timestamp,
                    "content_timespan_seconds": timestamp_info["content_timespan_seconds"],
                    "content_timestamp_count": timestamp_info["timestamp_count"],
                    "uses_content_timestamp_for_placement": True
                }
                
                logger.debug(f"Updated memory timestamp to content timestamp: {latest_timestamp} ({memory_props['timestamp_iso']})")
                
            else:
                # No content timestamps available, keep compression timestamp
                memory_props["temporal_info"] = {
                    "content_earliest_timestamp": None,
                    "content_latest_timestamp": None,
                    "content_timespan_seconds": 0,
                    "content_timestamp_count": 0,
                    "uses_content_timestamp_for_placement": False,
                    "fallback_reason": "no_content_timestamps_found"
                }
                
                logger.debug(f"No content timestamps found, keeping compression timestamp for memory")

            # CRITICAL FIX: Assign operation_index for proper chronological placement
            memory_node = self._assign_memory_operation_index(memory_node, content)
            
            return memory_node
            
        except Exception as e:
            logger.error(f"Error updating memory with content timestamp and operation_index: {e}", exc_info=True)
            # Return original memory node on error
            return memory_node
