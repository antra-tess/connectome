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

from opentelemetry import trace
from host.observability import get_tracer

from .base_component import Component
from elements.component_registry import register_component

# Import LLM interfaces
from llm.provider_interface import LLMMessage

# NEW: Import storage system
from storage import create_storage_from_env, StorageInterface

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
    FOCUSED_MEMORY_LIMIT = 4000      # 4k tokens of memories for focused elements (down from 20k)
    FOCUSED_FRESH_LIMIT = 30000      # 30k tokens of fresh content for focused elements (down from 50k)
    COMPRESSION_CHUNK_SIZE = 4000    # 4k tokens per compression chunk (instead of 10 items)
    MIN_COMPRESSION_BATCH = COMPRESSION_CHUNK_SIZE     # Minimum excess before triggering compression (avoid micro-compressions)
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
        
        logger.info(f"CompressionEngineComponent initialized ({self.id}) - ready for AgentMemoryCompressor integration and dual-stream processing")
    
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
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown the compression engine and storage."""
        try:
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
        """Get statistics about stored memory."""
        try:
            # NEW: Phase 3 - Get stats from AgentMemoryCompressor if available
            memory_compressor_stats = {}
            if self._memory_compressor:
                try:
                    memory_compressor_stats = await self._memory_compressor.get_memory_statistics()
                except Exception as stats_err:
                    logger.warning(f"Error getting AgentMemoryCompressor stats: {stats_err}")
                    memory_compressor_stats = {"error": str(stats_err)}
            
            return {
                "agent_name": self._agent_name,
                "conversation_id": self._conversation_id,
                "memory_compressor_available": bool(self._memory_compressor),
                "memory_compressor_stats": memory_compressor_stats,
                "phase": "3_agent_memory_compressor_integrated",
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}", exc_info=True)
            return {
                "error": str(e),
                "agent_name": self._agent_name,
                "phase": "3_integration_error"
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
            
            # Inject memory context at the top level if provided
            if memory_data:
                processed_veil = await self._inject_memory_context(processed_veil, memory_data)
            
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
            
            # Inject memory context
            processed_veil = await self._inject_memory_context(processed_veil, memory_data)
            
            return processed_veil
            
        except Exception as e:
            logger.error(f"Error processing full VEIL with memory: {e}", exc_info=True)
            return full_veil

    async def _inject_memory_context(self, 
                                   veil_root: Dict[str, Any], 
                                   memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject memory context into VEIL structure as additional nodes.
        
        SIMPLIFIED: Only inject agent workspace info at top level.
        Element-specific memories are now handled within containers by _compress_container_children().
        
        Args:
            veil_root: Root VEIL node to modify
            memory_data: Memory data to inject
            
        Returns:
            Modified VEIL with memory context injected
        """
        try:
            # Only inject agent workspace info (not element-specific memories)
            agent_info = memory_data.get('agent_info', {})
            if agent_info:
                workspace_node = {
                    "veil_id": f"memory_workspace_{self.id}",
                    "node_type": "memory_context", 
                    "properties": {
                        "structural_role": "memory_section",
                        "content_nature": "agent_workspace",
                        "memory_type": "workspace_info",
                        "agent_name": agent_info.get('agent_name', 'Unknown'),
                        "workspace_description": agent_info.get('workspace_description', ''),
                        "total_interactions": agent_info.get('total_stored_interactions', 0)
                    },
                    "children": []
                }
                
                # Inject only workspace info at top level
                children = veil_root.get('children', [])
                veil_root['children'] = [workspace_node] + children
                logger.debug(f"Injected agent workspace memory context into VEIL")
            
            return veil_root
            
        except Exception as e:
            logger.error(f"Error injecting memory context: {e}", exc_info=True)
            return veil_root

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

    async def _apply_focused_compression(self, container_node: Dict[str, Any], element_id: str, children: List[Dict[str, Any]], element_name: str, available_tools: List[str]) -> None:
        """
        Apply focused compression with lax limits for focused elements.
        
        NEW: Enhanced with background compression support for better performance.
        For focused elements: 50k fresh content + 20k memories = 70k total
        """
        try:
            if not children:
                # Handle empty containers
                memory_summary = "Empty conversation - no content to remember"
                compression_approach = "empty_container_focused"
                logger.debug(f"Empty focused container {element_id}: no children to compress")
                
                memory_node = {
                    "veil_id": f"compressed_{element_id}_{self.id}",
                    "node_type": "content_memory",
                    "properties": {
                        "structural_role": "compressed_content",
                        "content_nature": "content_memory", 
                        "original_element_id": element_id,
                        "memory_summary": memory_summary,
                        "original_child_count": 0,
                        "compression_approach": compression_approach,
                        "is_focused": True
                    },
                    "children": []
                }
                container_node['children'] = [memory_node]
                return
            
            # NEW Phase 2: Separate memories from fresh content for rolling compression
            existing_memories, fresh_content = self._separate_memories_and_content(children)
            
            # Calculate token counts
            memory_tokens = self._calculate_memory_tokens(existing_memories)
            fresh_tokens = self._calculate_children_tokens(fresh_content)
            
            logger.debug(f"Focused element {element_id}: {fresh_tokens} fresh tokens, {memory_tokens} memory tokens (limits: {self.FOCUSED_FRESH_LIMIT} fresh, {self.FOCUSED_MEMORY_LIMIT} memory)")
            
            # Check if we're within limits
            needs_fresh_compression = fresh_tokens > self.FOCUSED_FRESH_LIMIT
            needs_memory_compression = memory_tokens > self.FOCUSED_MEMORY_LIMIT

            if not needs_fresh_compression and not needs_memory_compression:
                # Within limits, preserve everything as-is
                logger.debug(f"Focused element {element_id} within limits, preserving all content")
                container_node['children'] = existing_memories + fresh_content
                
                # Ensure tool information is preserved
                if available_tools:
                    container_node['properties']['available_tools'] = available_tools
                    container_node['properties']['tools_available_despite_preservation'] = True
                return
            
            # ENHANCED: Continuous rolling compression approach with background support
            final_memories = existing_memories.copy()
            final_content = fresh_content.copy()

            # Step 1: Apply continuous fresh content compression (with background support)
            if needs_fresh_compression:
                excess_fresh = fresh_tokens - self.FOCUSED_FRESH_LIMIT
                if excess_fresh >= self.MIN_COMPRESSION_BATCH:
                    logger.info(f"Fresh content exceeds limit by {excess_fresh} tokens, applying continuous rolling compression with background support")
                    new_memories, preserved_content = await self._compress_excess_fresh_content_async(
                        element_id, element_name, available_tools, fresh_content
                    )
                    final_memories.extend(new_memories)
                    final_content = preserved_content
                else:
                    logger.debug(f"Fresh content excess ({excess_fresh}) below minimum batch, preserving")

            # Step 2: Apply continuous memory recompression (can be background for non-critical)
            current_memory_tokens = self._calculate_memory_tokens(final_memories)
            if current_memory_tokens > self.FOCUSED_MEMORY_LIMIT:
                logger.info(f"Memory content exceeds limit by {current_memory_tokens - self.FOCUSED_MEMORY_LIMIT} tokens, applying continuous recompression")
                final_memories = await self._recompress_excess_memories(
                    element_id, element_name, final_memories
                )
            
            # Set final children: memories + preserved fresh content
            container_node['children'] = final_memories + final_content
            
            # Ensure tool information is preserved
            if available_tools:
                container_node['properties']['available_tools'] = available_tools
                container_node['properties']['tools_available_despite_compression'] = True
            
            final_memory_tokens = self._calculate_memory_tokens(final_memories)
            final_fresh_tokens = self._calculate_children_tokens(final_content)
            
            logger.info(f"Rolling compression complete for {element_id}: {final_fresh_tokens} fresh + {final_memory_tokens} memory = {final_fresh_tokens + final_memory_tokens} total tokens")
            
        except Exception as e:
            logger.error(f"Error applying focused compression: {e}", exc_info=True)
            # Fallback to simple compression with background support
            if self._memory_compressor:
                try:
                    # Use asynchronous compression for fallback too
                    result_node = await self._memory_compressor.get_memory_or_fallback(
                        element_ids=[element_id],
                        raw_veil_nodes=children,
                        token_limit=self.FOCUSED_FRESH_LIMIT + self.FOCUSED_MEMORY_LIMIT  # Full focused limit
                    )
                    
                    if result_node:
                        result_props = result_node.get("properties", {})
                        memory_summary = result_props.get("memory_summary", f"Rolling compression failed: {len(children)} items from {element_name}")
                        compression_approach = "rolling_compression_fallback_async"
                    else:
                        memory_summary = f"Rolling compression failed: {len(children)} items from {element_name}"
                        compression_approach = "rolling_compression_fallback"
                except Exception as fallback_error:
                    logger.error(f"Async fallback also failed: {fallback_error}", exc_info=True)
                    memory_summary = f"Rolling compression failed: {len(children)} items from {element_name}"
                    compression_approach = "rolling_compression_fallback"
            else:
                memory_summary = f"Rolling compression failed: {len(children)} items from {element_name}"
                compression_approach = "rolling_compression_fallback"
            
            memory_node = {
                "veil_id": f"compressed_{element_id}_{self.id}",
                "node_type": "content_memory",
                "properties": {
                    "structural_role": "compressed_content",
                    "content_nature": "content_memory", 
                    "original_element_id": element_id,
                    "memory_summary": memory_summary,
                    "original_child_count": len(children),
                    "compression_approach": compression_approach,
                    "is_focused": True
                },
                "children": []
            }
            container_node['children'] = [memory_node]

    async def _apply_unfocused_compression(self, container_node: Dict[str, Any], element_id: str, children: List[Dict[str, Any]], element_name: str, available_tools: List[str]) -> None:
        """
        Apply unfocused compression with strict limits for unfocused elements.
        
        NEW: Now uses background compression with intelligent fallbacks to prevent main thread blocking.
        For unfocused elements: 4k total (everything compressed or trimmed fresh content)
        """
        try:
            if not children:
                # Handle empty containers
                memory_summary = "Empty conversation - no content to remember"
                compression_approach = "empty_container_unfocused"
                logger.debug(f"Empty unfocused container {element_id}: no children to compress")
            elif self._memory_compressor:
                # NEW: Use get_memory_or_fallback for asynchronous compression strategy
                logger.debug(f"Using asynchronous compression strategy for unfocused {element_id} with {len(children)} children")
                
                # Get memory or trimmed fresh content (non-blocking)
                result_node = await self._memory_compressor.get_memory_or_fallback(
                    element_ids=[element_id],
                    raw_veil_nodes=children,
                    token_limit=self.UNFOCUSED_TOTAL_LIMIT  # 4k tokens for unfocused
                )
                
                if result_node:
                    # Check what type of result we got
                    node_type = result_node.get("node_type", "")
                    result_props = result_node.get("properties", {})
                    
                    if node_type == "memorized_content":
                        # We got a completed memory
                        memory_summary = result_props.get("memory_summary", "Memory content")
                        compression_approach = "agent_memory_compressor_unfocused_async"
                        logger.debug(f"Using completed memory for unfocused {element_id}")
                    
                    elif node_type in ["fresh_content", "trimmed_fresh_content"]:
                        # We got fresh content fallback (compression running in background)
                        is_trimmed = result_props.get("is_trimmed", False)
                        trimmed_count = result_props.get("trimmed_node_count", len(children))
                        original_count = result_props.get("original_node_count", len(children))
                        
                        if is_trimmed:
                            memory_summary = f"Recent {trimmed_count} of {original_count} items (compression in progress)"
                            compression_approach = "trimmed_fresh_content_unfocused_async"
                            logger.debug(f"Using trimmed fresh content for unfocused {element_id}: {trimmed_count}/{original_count} items")
                        else:
                            memory_summary = f"All {len(children)} recent items (compression in progress)"
                            compression_approach = "fresh_content_unfocused_async"
                            logger.debug(f"Using full fresh content for unfocused {element_id}")
                        
                        # For fresh content, we need to replace the container children with the result
                        container_node['children'] = result_node.get("children", children)
                        
                        # Ensure tool information is preserved
                        if available_tools:
                            container_node['properties']['available_tools'] = available_tools
                            container_node['properties']['tools_available_despite_compression'] = True
                        
                        logger.debug(f"Applied fresh content fallback to unfocused container {element_id}")
                        return  # Early return - content already set
                    
                    elif node_type == "compression_placeholder":
                        # We got a placeholder (compression starting)
                        memory_summary = result_props.get("memory_summary", "⏳ Processing conversation memory...")
                        compression_approach = "compression_placeholder_unfocused_async"
                        logger.debug(f"Using compression placeholder for unfocused {element_id}")
                    
                    else:
                        # Unknown result type, extract summary
                        memory_summary = result_props.get("memory_summary", f"Content from {element_id}")
                        compression_approach = f"unknown_result_type_{node_type}_unfocused"
                        logger.warning(f"Unknown result type {node_type} for unfocused {element_id}")
                else:
                    # No result from memory compressor
                    memory_summary = f"Content from {element_id} (compression failed)"
                    compression_approach = "memory_compressor_failed_unfocused"
            else:
                # Fallback to simple summary if AgentMemoryCompressor not available
                logger.debug(f"AgentMemoryCompressor not available, using simple summary for unfocused {element_id}")
                memory_summary = await self._generate_content_summary(children, element_id)
                compression_approach = "simple_summary_fallback_unfocused"
            
            # Create content memory node (for memory results and placeholders)
            memory_node = {
                "veil_id": f"compressed_{element_id}_{self.id}",
                "node_type": "content_memory",
                "properties": {
                    "structural_role": "compressed_content",
                    "content_nature": "content_memory", 
                    "original_element_id": element_id,
                    "memory_summary": memory_summary,
                    "original_child_count": len(children),
                    "compression_timestamp": datetime.now().isoformat(),
                    "compression_approach": compression_approach,
                    "agent_reflected": compression_approach.startswith("agent_memory"),
                    "is_focused": False,  # Mark as unfocused compression
                    "is_async_compression": "async" in compression_approach  # NEW: Track async compression
                },
                "children": []
            }
            
            # Replace children with the memory summary but preserve container metadata
            container_node['children'] = [memory_node]
            
            # Ensure tool information is preserved on the container
            if available_tools:
                container_node['properties']['available_tools'] = available_tools
                container_node['properties']['tools_available_despite_compression'] = True
            
            logger.debug(f"Applied unfocused compression to container {element_id}: {len(children)} children → {compression_approach}")
            
        except Exception as e:
            logger.error(f"Error applying unfocused compression: {e}", exc_info=True)

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
    
    async def _create_new_agent_memory(self, 
                                     element_id: str, 
                                     element_name: str, 
                                     available_tools: List[str], 
                                     children: List[Dict[str, Any]],
                                     is_recompression: bool = False,
                                     is_focused: bool = False) -> tuple[str, str]:
        """
        Create a new agent memory using AgentMemoryCompressor.
        
        ENHANCED: Now supports focused vs unfocused compression context.
        NEW: Passes existing memory context directly from compression pipeline.
        PHASE 3: Enhanced with full VEIL context for contextually aware memories.
        
        Returns:
            Tuple of (memory_summary, compression_approach)
        """
        try:
            element_ids = [element_id]
            
            # NEW: Extract existing memories from children for context
            existing_memories, fresh_content = self._separate_memories_and_content(children)
            existing_memory_summaries = []
            
            for memory_node in existing_memories:
                memory_props = memory_node.get("properties", {})
                memory_summary = memory_props.get("memory_summary", "")
                if memory_summary:
                    existing_memory_summaries.append(memory_summary)
            
            # NEW: Add focus context to compression metadata
            focus_type = "focused" if is_focused else "unfocused"
            action_type = "recompression" if is_recompression else "compression"
            
            # PHASE 3: NEW - Get full VEIL context for enhanced memory creation using HUD rendering
            full_veil_context = await self._get_hud_rendered_context(element_id, children)
            
            compression_context = {
                "element_id": element_id,
                "element_name": element_name,
                "element_type": "container",
                "compression_reason": f"{focus_type}_content_{action_type}",
                "available_tools": available_tools,
                "content_fingerprint": self._calculate_content_fingerprint(children),  # Store fingerprint
                "is_focused": is_focused,  # NEW: Focus context
                "focus_type": focus_type,  # NEW: Focus type label
                # NEW: Direct memory context pass-through from compression pipeline
                "existing_memory_context": existing_memory_summaries,
                "existing_memory_count": len(existing_memories),
                # PHASE 3: NEW - Full VEIL context for enhanced memories
                "full_veil_context": full_veil_context
            }
            
            # Pass only fresh content for compression, with enhanced context
            memorized_veil_node = await self._memory_compressor.compress(
                raw_veil_nodes=fresh_content,  # Only fresh content to compress
                element_ids=element_ids,
                compression_context=compression_context  # Enhanced with full VEIL context
            )
            
            if memorized_veil_node:
                memory_summary = memorized_veil_node.get("properties", {}).get("memory_summary", "Agent memory formation failed")
                
                # NEW: Generate focus-aware compression approach labels
                if is_recompression:
                    compression_approach = f"agent_memory_recompressor_{focus_type}_enhanced_context"
                else:
                    compression_approach = f"agent_memory_compressor_{focus_type}_enhanced_context"
                
                action = "Recompressed" if is_recompression else "Agent reflected on"
                context_info = " (with full VEIL context)" if full_veil_context else ""
                memory_context_info = f" (with {len(existing_memory_summaries)} memory context)" if existing_memory_summaries else ""
                logger.info(f"{action} {focus_type} container {element_id}: {memory_summary[:100]}...{context_info}{memory_context_info}")
                return memory_summary, compression_approach
            else:
                raise Exception("AgentMemoryCompressor returned None")
                
        except Exception as compression_error:
            logger.warning(f"AgentMemoryCompressor failed for {focus_type} {element_id}: {compression_error}")
            memory_summary = await self._generate_content_summary(children, element_id)
            compression_approach = f"fallback_simple_summary_{focus_type}"
            return memory_summary, compression_approach

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

    async def _compress_excess_fresh_content_async(self, element_id: str, element_name: str, available_tools: List[str], 
                                                  fresh_content: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        NEW: Enhanced version of fresh content compression with background processing support.
        
        This version can use background compression for non-critical chunks while maintaining
        responsive performance for focused elements.
        """
        try:
            if not fresh_content:
                return [], []
            
            total_fresh_tokens = self._calculate_children_tokens(fresh_content)
            excess_tokens = total_fresh_tokens - self.FOCUSED_FRESH_LIMIT
            
            if excess_tokens <= self.MIN_COMPRESSION_BATCH:
                # Not enough excess to warrant compression (avoid micro-compressions)
                logger.debug(f"Excess tokens ({excess_tokens}) below minimum batch size ({self.MIN_COMPRESSION_BATCH})")
                return [], fresh_content
            
            logger.info(f"Continuous rolling compression with async support: {excess_tokens} excess tokens from {element_id}")
            
            new_memories = []
            remaining_content = fresh_content.copy()
            
            # Continuously compress 4k chunks until within limits
            while True:
                current_tokens = self._calculate_children_tokens(remaining_content)
                current_excess = current_tokens - self.FOCUSED_FRESH_LIMIT
                
                if current_excess <= 0:
                    break  # Within limits now
                
                # Find 4k token boundary for next compression chunk
                chunk_tokens = min(self.COMPRESSION_CHUNK_SIZE, current_excess + 1000)  # Add buffer
                chunk_boundary = self._find_token_based_boundary(remaining_content, chunk_tokens)
                
                if chunk_boundary <= 0:
                    logger.warning(f"Cannot find valid token boundary for {element_id}")
                    break
                
                # Compress this chunk with background support
                chunk_to_compress = remaining_content[:chunk_boundary]
                remaining_content = remaining_content[chunk_boundary:]
                
                # Create memory from this chunk (can be async for non-critical chunks)
                chunk_memory = await self._create_token_based_memory_chunk_async(
                    chunk_to_compress, element_id, element_name, available_tools, len(new_memories)
                )
                
                if chunk_memory:
                    new_memories.append(chunk_memory)
                
                logger.debug(f"Compressed chunk {len(new_memories)}: {len(chunk_to_compress)} items → memory")
                
                # Safety check to avoid infinite loops
                if len(new_memories) > 20:  # Max 20 compression cycles
                    logger.warning(f"Too many compression cycles for {element_id}, stopping")
                    break
            
            final_fresh_tokens = self._calculate_children_tokens(remaining_content)
            total_memory_tokens = sum(self._calculate_memory_tokens([mem]) for mem in new_memories)
            
            logger.info(f"Continuous rolling complete: {total_fresh_tokens} → {final_fresh_tokens} fresh tokens, created {len(new_memories)} memories ({total_memory_tokens} tokens)")
            
            return new_memories, remaining_content
            
        except Exception as e:
            logger.error(f"Error in async continuous rolling compression: {e}", exc_info=True)
            # Fallback: preserve everything
            return [], fresh_content

    async def _create_token_based_memory_chunk_async(self, content_to_compress: List[Dict[str, Any]], element_id: str, 
                                                    element_name: str, available_tools: List[str], chunk_index: int) -> Dict[str, Any]:
        """
        NEW: Enhanced version of memory chunk creation with background compression support.
        
        For focused elements, we can use background compression for older chunks
        while keeping recent chunks responsive.
        """
        try:
            if not content_to_compress:
                return None
            
            # Determine if this chunk should use background compression
            # For focused elements, we can afford some background processing for older chunks
            use_background = chunk_index > 0  # First chunk synchronous, others can be background
            
            if self._memory_compressor and use_background:
                # Use the new async compression strategy
                chunk_element_id = f"{element_id}_chunk_{chunk_index}"
                
                result_node = await self._memory_compressor.get_memory_or_fallback(
                    element_ids=[chunk_element_id],
                    raw_veil_nodes=content_to_compress,
                    token_limit=self.COMPRESSION_CHUNK_SIZE
                )
                
                if result_node:
                    # Convert result to memory node format
                    result_props = result_node.get("properties", {})
                    memory_summary = result_props.get("memory_summary", f"Memory chunk {chunk_index + 1}: {len(content_to_compress)} items from {element_name}")
                    
                    memory_node = {
                        "veil_id": f"rolling_memory_{element_id}_{chunk_index}_{self.id}",
                        "node_type": "content_memory",
                        "properties": {
                            "structural_role": "compressed_content",
                            "content_nature": "content_memory",
                            "original_element_id": element_id,
                            "memory_summary": memory_summary,
                            "original_child_count": len(content_to_compress),
                            "compression_timestamp": datetime.now().isoformat(),
                            "compression_approach": "rolling_compression_async",
                            "is_focused": True,
                            "is_rolling_memory": True,
                            "chunk_index": chunk_index,
                            "is_async_compression": True
                        },
                        "children": []
                    }
                    return memory_node
            
            # Fallback to synchronous compression or simple summary
            if self._memory_compressor:
                memory_summary, compression_approach = await self._create_new_agent_memory(
                    f"{element_id}_chunk_{chunk_index}", element_name, available_tools, content_to_compress, 
                    is_recompression=False, is_focused=True
                )
            else:
                # Fallback summary
                memory_summary = f"Memory chunk {chunk_index + 1}: {len(content_to_compress)} items from {element_name}"
                compression_approach = "chunk_fallback_summary"
            
            # Create memory node
            memory_node = {
                "veil_id": f"rolling_memory_{element_id}_{chunk_index}_{self.id}",
                "node_type": "content_memory",
                "properties": {
                    "structural_role": "compressed_content",
                    "content_nature": "content_memory",
                    "original_element_id": element_id,
                    "memory_summary": memory_summary,
                    "original_child_count": len(content_to_compress),
                    "compression_timestamp": datetime.now().isoformat(),
                    "compression_approach": compression_approach,
                    "is_focused": True,
                    "is_rolling_memory": True,
                    "chunk_index": chunk_index
                },
                "children": []
            }
            
            return memory_node
            
        except Exception as e:
            logger.error(f"Error creating async token-based memory chunk: {e}", exc_info=True)
            return None

    async def _recompress_excess_memories(self, element_id: str, element_name: str, 
                                        memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        ENHANCED Phase 2: Continuous memory recompression with tighter limits.
        
        NEW APPROACH: Instead of waiting for 20k memory tokens, we recompress
        continuously when memory exceeds 4k tokens. This maintains tighter
        control over memory usage and better compression quality.
        
        Args:
            element_id: ID of element for context
            element_name: Name of element for context
            memories: List of memory VEIL nodes
            
        Returns:
            Recompressed memory list within token limits
        """
        try:
            if not memories:
                return []
            
            current_tokens = self._calculate_memory_tokens(memories)
            
            if current_tokens <= self.FOCUSED_MEMORY_LIMIT:
                return memories  # Already within limits
            
            logger.info(f"Continuous memory recompression: {len(memories)} memories ({current_tokens} tokens) for {element_id}")
            
            # Sort memories by timestamp (oldest first)
            sorted_memories = self._sort_memories_by_age(memories)
            final_memories = []
            
            # Continuously recompress until within 4k limit
            while True:
                current_memory_tokens = self._calculate_memory_tokens(sorted_memories)
                
                if current_memory_tokens <= self.FOCUSED_MEMORY_LIMIT:
                    final_memories = sorted_memories
                    break
                
                # Take 2 oldest memories and combine them
                if len(sorted_memories) < 2:
                    # Can't compress further, keep what we have
                    final_memories = sorted_memories
                    break
                
                oldest_memories = sorted_memories[:2]
                remaining_memories = sorted_memories[2:]
                
                # Create summary of the 2 oldest memories
                summary_memory = await self._create_memory_summary(oldest_memories, element_id, element_name)
                
                # Replace the 2 oldest with 1 summary
                sorted_memories = [summary_memory] + remaining_memories
                
                logger.debug(f"Combined 2 oldest memories into 1 summary, now {len(sorted_memories)} memories")
                
                # Safety check to avoid infinite loops
                if len(sorted_memories) <= 1:
                    final_memories = sorted_memories
                    break
            
            final_tokens = self._calculate_memory_tokens(final_memories)
            logger.info(f"Continuous memory recompression complete: {len(memories)} → {len(final_memories)} memories, {current_tokens} → {final_tokens} tokens")
            
            return final_memories
            
        except Exception as e:
            logger.error(f"Error in continuous memory recompression: {e}", exc_info=True)
            # Fallback: keep newest memories up to limit
            return memories[-2:] if len(memories) > 2 else memories  # Keep last 2 memories

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

    def _find_existing_total_memory(self, children: List[Dict[str, Any]], element_id: str) -> Optional[Dict[str, Any]]:
        """
        Find existing total memory for an element (for unfocused element reuse).
        
        Args:
            children: List of children to check
            element_id: ID of the element being checked
            
        Returns:
            Existing memory node if found, None otherwise
        """
        if not self._memory_compressor:
            return None
        
        memory_id = self._memory_compressor.get_memory_for_elements([element_id])
        if memory_id:
            # Look for the memory in the children
            for child in children:
                node_props = child.get("properties", {})
                if node_props.get("memory_id") == memory_id:
                    return child
                # Also check for content_memory nodes that might reference this memory
                if child.get("node_type") == "content_memory" and node_props.get("original_element_id") == element_id:
                    return child
        
        return None

    def _calculate_memory_tokens(self, memory_nodes: List[Dict[str, Any]]) -> int:
        """
        Calculate total tokens in memory nodes.
        
        Args:
            memory_nodes: List of memory VEIL nodes
            
        Returns:
            Estimated token count
        """
        total_tokens = 0
        
        for memory_node in memory_nodes:
            node_props = memory_node.get("properties", {})
            # Check for stored token count
            stored_tokens = node_props.get("token_count")
            if stored_tokens:
                total_tokens += stored_tokens
            else:
                # Estimate based on memory summary length
                memory_summary = node_props.get("memory_summary", "")
                # Rough estimate: 4 characters per token
                estimated_tokens = len(memory_summary) // 4
                total_tokens += estimated_tokens
        
        return total_tokens

    def _calculate_children_tokens(self, children: List[Dict[str, Any]]) -> int:
        """
        Calculate total tokens in VEIL children.
        
        Args:
            children: List of VEIL child nodes
            
        Returns:
            Estimated token count
        """
        try:
            from .memory_compressor_interface import estimate_veil_tokens
            return estimate_veil_tokens(children)
        except Exception as e:
            logger.warning(f"Error calculating children tokens: {e}")
            # Fallback estimation
            total_chars = 0
            for child in children:
                # Estimate based on text content
                props = child.get("properties", {})
                text_content = props.get("text_content", "")
                total_chars += len(text_content)
            
            # Rough estimate: 4 characters per token
            return total_chars // 4 

    def _create_trimmed_fresh_content(self, 
                                    raw_veil_nodes: List[Dict[str, Any]], 
                                    element_ids: List[str], 
                                    token_limit: int) -> Dict[str, Any]:
        """
        NEW: Create a trimmed version of fresh content that fits within token limits.
        
        This provides immediate context while compression runs in background.
        """
        try:
            if not raw_veil_nodes:
                return self._create_compression_placeholder("empty", element_ids, [])
            
            # Calculate total tokens
            total_tokens = estimate_veil_tokens(raw_veil_nodes)
            
            if total_tokens <= token_limit:
                # Content fits within limit, return as-is but mark as fresh
                return {
                    "veil_id": f"fresh_content_{element_ids[0] if element_ids else 'unknown'}",
                    "node_type": "fresh_content",
                    "properties": {
                        "structural_role": "container",
                        "content_nature": "fresh_content",
                        "element_id": element_ids[0] if element_ids else "unknown",
                        "is_fresh_content": True,
                        "total_tokens": total_tokens,
                        "token_limit": token_limit
                    },
                    "children": raw_veil_nodes
                }
            
            # Content exceeds limit, trim to most recent content
            trimmed_nodes = []
            current_tokens = 0
            
            # Process nodes in reverse order (newest first)
            for node in reversed(raw_veil_nodes):
                node_tokens = estimate_veil_tokens([node])
                if current_tokens + node_tokens <= token_limit:
                    trimmed_nodes.insert(0, node)  # Insert at beginning to maintain order
                    current_tokens += node_tokens
                else:
                    break
            
            if not trimmed_nodes:
                # Even single node exceeds limit, take the newest one anyway
                trimmed_nodes = [raw_veil_nodes[-1]]
                current_tokens = estimate_veil_tokens(trimmed_nodes)
            
            logger.debug(f"Trimmed content: {len(raw_veil_nodes)} → {len(trimmed_nodes)} nodes, {total_tokens} → {current_tokens} tokens")
            
            return {
                "veil_id": f"trimmed_fresh_{element_ids[0] if element_ids else 'unknown'}",
                "node_type": "trimmed_fresh_content",
                "properties": {
                    "structural_role": "container",
                    "content_nature": "trimmed_fresh_content",
                    "element_id": element_ids[0] if element_ids else "unknown",
                    "is_fresh_content": True,
                    "is_trimmed": True,
                    "total_tokens": current_tokens,
                    "token_limit": token_limit,
                    "original_node_count": len(raw_veil_nodes),
                    "trimmed_node_count": len(trimmed_nodes),
                    "trimming_note": f"Showing {len(trimmed_nodes)} most recent items (compression in progress)"
                },
                "children": trimmed_nodes
            }
            
        except Exception as e:
            logger.error(f"Error creating trimmed fresh content: {e}", exc_info=True)
            return self._create_compression_placeholder("trimming_error", element_ids, raw_veil_nodes)

    def _find_token_based_boundary(self, children: List[Dict[str, Any]], chunk_tokens: int) -> int:
        """
        Find where to split content for rolling compression, respecting conversation boundaries.
        
        ENHANCED: Now respects message boundaries and conversation flow for better compression quality.
        
        Strategy:
        1. Never split individual messages
        2. Prefer natural conversation breaks (time gaps, topic changes)
        3. Respect message groupings by same sender
        4. Fall back to message boundaries if needed
        
        Args:
            children: List of fresh content nodes (should be in chronological order)
            chunk_tokens: Number of tokens we need to remove
            
        Returns:
            Index where to split (content[0:boundary] gets compressed, content[boundary:] preserved)
        """
        try:
            if not children or chunk_tokens <= 0:
                return 0
            
            # If chunk_tokens is larger than all content, compress most but leave some
            total_tokens = self._calculate_children_tokens(children)
            if chunk_tokens >= total_tokens:
                # Leave at least 25% of content
                min_preserve = max(1, len(children) // 4)
                max_boundary = len(children) - min_preserve
                logger.debug(f"Chunk size ({chunk_tokens}) >= total ({total_tokens}), using max boundary {max_boundary}")
                return max_boundary
            
            tokens_accumulated = 0
            best_boundary = 0
            last_conversation_break = 0
            
            # Analyze message flow to find natural boundaries
            for i, content_item in enumerate(children):
                item_tokens = self._calculate_children_tokens([content_item])
                tokens_accumulated += item_tokens
                
                # Always update boundary to complete messages (never split mid-message)
                boundary_candidate = i + 1
                
                # Check if this is a natural conversation break
                is_conversation_break = self._is_natural_conversation_break(
                    children, i, content_item
                )
                
                if is_conversation_break:
                    last_conversation_break = boundary_candidate
                    logger.debug(f"Found conversation break at index {i} ({tokens_accumulated} tokens)")
                
                # If we've accumulated enough tokens, find the best stopping point
                if tokens_accumulated >= chunk_tokens:
                    # Prefer conversation breaks if we found one recently
                    conversation_break_distance = boundary_candidate - last_conversation_break
                    
                    if last_conversation_break > 0 and conversation_break_distance <= 5:
                        # Use conversation break if it's within 5 messages of target (reduced from 10)
                        best_boundary = last_conversation_break
                        logger.debug(f"Using conversation break boundary at {best_boundary} (distance: {conversation_break_distance})")
                    else:
                        # Use current message boundary (complete the current message)
                        best_boundary = boundary_candidate
                        logger.debug(f"Using message boundary at {best_boundary}")
                    
                    break
            
            # If we didn't find enough tokens, use what we have but respect limits
            if best_boundary == 0 and len(children) > 0:
                # Ensure we don't compress everything (leave at least 25% of content)
                min_preserve = max(1, len(children) // 4)
                best_boundary = min(len(children) - min_preserve, len(children) - 1)
                logger.debug(f"Insufficient tokens for target, using fallback boundary {best_boundary}")
            
            # Final safety check - ensure we don't compress everything
            min_preserve = max(1, len(children) // 4)
            max_compress = len(children) - min_preserve
            best_boundary = min(best_boundary, max_compress)
            
            final_tokens = self._calculate_children_tokens(children[:best_boundary])
            logger.debug(f"Smart boundary: compress {best_boundary}/{len(children)} messages ({final_tokens} tokens)")
            
            return best_boundary
            
        except Exception as e:
            logger.error(f"Error finding smart token-based boundary: {e}", exc_info=True)
            # Fallback: compress half, ensuring message boundaries
            return len(children) // 2

    def _is_natural_conversation_break(self, children: List[Dict[str, Any]], current_index: int, 
                                     current_item: Dict[str, Any]) -> bool:
        """
        Determine if there's a natural conversation break at this point.
        
        Natural breaks include:
        - Time gaps between messages (>1 hour)
        - Topic changes (different keywords/context)
        - Sender changes after long messages
        - Attachment boundaries (before/after attachments)
        
        Args:
            children: Full list of content items
            current_index: Index of current item being evaluated
            current_item: The current content item
            
        Returns:
            True if this is a good place for a conversation break
        """
        try:
            props = current_item.get("properties", {})
            content_nature = props.get("content_nature", "")
            
            # Only analyze chat messages for conversation breaks
            if content_nature != "chat_message":
                return False
            
            # Time-based breaks: look for gaps >1 hour
            time_break = self._check_time_gap_break(children, current_index)
            if time_break:
                return True
            
            # Sender-based breaks: new speaker after substantial content
            sender_break = self._check_sender_change_break(children, current_index)
            if sender_break:
                return True
            
            # Attachment boundaries: before/after attachments
            attachment_break = self._check_attachment_boundary_break(children, current_index)
            if attachment_break:
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking conversation break: {e}")
            return False

    def _check_time_gap_break(self, children: List[Dict[str, Any]], current_index: int) -> bool:
        """Check for significant time gaps between messages."""
        try:
            if current_index == 0:
                return True  # Beginning is always a good break
            
            current_props = children[current_index].get("properties", {})
            prev_props = children[current_index - 1].get("properties", {})
            
            current_time = current_props.get("timestamp_iso")
            prev_time = prev_props.get("timestamp_iso")
            
            if not current_time or not prev_time:
                return False
            
            # Parse timestamps and check for gaps
            from datetime import datetime, timedelta
            
            try:
                if isinstance(current_time, str):
                    current_dt = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
                else:
                    current_dt = datetime.fromtimestamp(current_time)
                
                if isinstance(prev_time, str):
                    prev_dt = datetime.fromisoformat(prev_time.replace('Z', '+00:00'))
                else:
                    prev_dt = datetime.fromtimestamp(prev_time)
                
                time_gap = current_dt - prev_dt
                
                # Consider >1 hour as a conversation break
                if time_gap > timedelta(hours=1):
                    logger.debug(f"Time gap break: {time_gap} between messages")
                    return True
                    
            except (ValueError, TypeError):
                pass  # Invalid timestamp format
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking time gap: {e}")
            return False

    def _check_sender_change_break(self, children: List[Dict[str, Any]], current_index: int) -> bool:
        """Check for sender changes after substantial content blocks."""
        try:
            if current_index == 0:
                return True  # First message is always a break
            
            current_props = children[current_index].get("properties", {})
            current_sender = current_props.get("sender_name", "")
            
            # Check if sender actually changed
            prev_props = children[current_index - 1].get("properties", {})
            prev_sender = prev_props.get("sender_name", "")
            
            if current_sender == prev_sender:
                return False  # Same sender, not a break
            
            # Look back to see how much content the previous sender had
            consecutive_messages = 0
            total_chars = 0
            
            for i in range(current_index - 1, -1, -1):
                check_props = children[i].get("properties", {})
                check_sender = check_props.get("sender_name", "")
                
                if check_sender != prev_sender:
                    break
                
                consecutive_messages += 1
                total_chars += len(check_props.get("text_content", ""))
            
            # Consider it a break only if the previous sender had substantial content:
            # - 3+ consecutive messages OR
            # - >500 characters of content
            if consecutive_messages >= 3 or total_chars > 500:
                logger.debug(f"Sender change break: {prev_sender} → {current_sender} after {consecutive_messages} messages ({total_chars} chars)")
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking sender change: {e}")
            return False

    def _check_attachment_boundary_break(self, children: List[Dict[str, Any]], current_index: int) -> bool:
        """Check for attachment boundaries (before/after attachments)."""
        try:
            current_props = children[current_index].get("properties", {})
            current_attachments = current_props.get("attachment_metadata", [])
            
            # Current message has attachments - check if previous didn't
            if current_attachments and current_index > 0:
                prev_props = children[current_index - 1].get("properties", {})
                prev_attachments = prev_props.get("attachment_metadata", [])
                
                if not prev_attachments:
                    logger.debug(f"Attachment boundary break: message {current_index} introduces attachments")
                    return True
            
            # Previous message had attachments - check if current doesn't
            if not current_attachments and current_index > 0:
                prev_props = children[current_index - 1].get("properties", {})
                prev_attachments = prev_props.get("attachment_metadata", [])
                
                if prev_attachments:
                    logger.debug(f"Attachment boundary break: message {current_index} ends attachment sequence")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking attachment boundary: {e}")
            return False 

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
            # Separate existing memories from fresh content
            existing_memories, fresh_content = self._separate_memories_and_content(children)
            
            # Convert fresh content into N-chunks (4k token boundaries)
            n_chunks = await self._content_to_n_chunks(fresh_content)
            
            # Convert existing memories into M-chunks
            m_chunks = self._memories_to_m_chunks(existing_memories)
            
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
            
            # Detect new content since last update
            new_content = await self._detect_new_content(element_id, children)
            
            if not new_content:
                logger.debug(f"No new content detected for element {element_id}")
                return
            
            # Add new content to N-stream
            current_structure["n_chunks"].extend(await self._content_to_n_chunks(new_content))
            
            # Check if we need to compress oldest N-chunks into M-chunks
            await self._process_chunk_boundaries(element_id)
            
            # Update metadata
            current_structure["last_update"] = datetime.now()
            current_structure["total_tokens"] = self._calculate_total_chunk_tokens(current_structure)
            
            logger.debug(f"Updated chunks for {element_id}: {len(current_structure['n_chunks'])} N-chunks, {len(current_structure['m_chunks'])} M-chunks")
            
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
        
        This implements the core dual-stream logic:
        - When N-chunks complete (reach 4k), they are immediately compressed to M-chunks
        - M-chunks are re-compressed when they exceed limits
        - This ensures continuous background compression rather than reactive compression
        
        Args:
            element_id: ID of the element to process
        """
        try:
            chunk_structure = self._element_chunks[element_id]
            n_chunks = chunk_structure["n_chunks"]
            m_chunks = chunk_structure["m_chunks"]
            
            # CRITICAL: Immediate compression of complete N-chunks
            complete_n_chunks = [chunk for chunk in n_chunks if chunk.get("is_complete", False)]
            
            if complete_n_chunks:
                logger.info(f"Found {len(complete_n_chunks)} complete N-chunks for {element_id}, triggering immediate compression")
                
                # Compress complete N-chunks into M-chunks
                for chunk in complete_n_chunks:
                    try:
                        # Extract content from N-chunk
                        chunk_content = chunk.get("content", [])
                        if not chunk_content:
                            continue
                        
                        # Create memory from this complete chunk
                        memory_summary, compression_approach = await self._create_new_agent_memory(
                            element_id=f"{element_id}_chunk_{chunk.get('chunk_index', 0)}",
                            element_name=f"element_{element_id}",
                            available_tools=[],  # Will be preserved at container level
                            children=chunk_content,
                            is_recompression=False,
                            is_focused=False  # Background compression, treat as unfocused
                        )
                        
                        # Create new M-chunk from the compressed memory
                        memory_node = {
                            "veil_id": f"memory_{element_id}_{chunk.get('chunk_index', 0)}_{self.id}",
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
                                "source_chunk_index": chunk.get('chunk_index', 0),
                                "source_chunk_tokens": chunk.get('token_count', 0)
                            },
                            "children": []
                        }
                        
                        # Add to M-chunks
                        new_m_chunk = {
                            "chunk_type": "m_chunk",
                            "memory_node": memory_node,
                            "token_count": self._calculate_memory_tokens([memory_node]),
                            "chunk_index": len(m_chunks),
                            "created_at": datetime.now(),
                            "is_complete": True,
                            "source_n_chunk_index": chunk.get('chunk_index', 0)
                        }
                        
                        m_chunks.append(new_m_chunk)
                        logger.debug(f"Compressed N-chunk {chunk.get('chunk_index', 0)} into M-chunk for {element_id}")
                        
                    except Exception as chunk_error:
                        logger.error(f"Error compressing individual N-chunk for {element_id}: {chunk_error}", exc_info=True)
                        continue
                
                # Remove compressed N-chunks from the list
                # Keep incomplete N-chunks (current chunk being filled)
                incomplete_n_chunks = [chunk for chunk in n_chunks if not chunk.get("is_complete", False)]
                chunk_structure["n_chunks"] = incomplete_n_chunks
                
                logger.info(f"Immediate compression complete for {element_id}: compressed {len(complete_n_chunks)} N-chunks, {len(incomplete_n_chunks)} incomplete N-chunks remaining")
            
            # Check if M-chunks need re-compression (exceed 1-chunk limit)
            total_memory_tokens = sum(chunk.get("token_count", 0) for chunk in m_chunks)
            
            if total_memory_tokens > self.COMPRESSION_CHUNK_SIZE:
                logger.info(f"M-chunks exceed {self.COMPRESSION_CHUNK_SIZE} tokens for {element_id}, triggering re-compression")
                # Re-compress M-chunks into single memory-of-memories
                recompressed_memories = await self._recompress_m_chunks(element_id, m_chunks)
                chunk_structure["m_chunks"] = recompressed_memories
            
            logger.debug(f"Processed chunk boundaries for {element_id}: {len(chunk_structure['n_chunks'])} N-chunks, {len(chunk_structure['m_chunks'])} M-chunks")
            
        except Exception as e:
            logger.error(f"Error processing chunk boundaries for {element_id}: {e}", exc_info=True)

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
        
        This is a placeholder for more sophisticated content change detection.
        For now, we'll use simple content fingerprinting.
        
        Args:
            element_id: ID of the element
            children: Current VEIL children
            
        Returns:
            List of new content items
        """
        try:
            # For Phase 2A, we'll implement simple detection
            # TODO: In later phases, use content fingerprinting for efficient detection
            
            current_structure = self._element_chunks.get(element_id, {})
            last_update = current_structure.get("last_update")
            
            if not last_update:
                # First time, all content is new
                return children
            
            # For now, return empty (no new content detection yet)
            # This will be enhanced in Phase 2B
            return []
            
        except Exception as e:
            logger.error(f"Error detecting new content for {element_id}: {e}", exc_info=True)
            return []

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
        
        This implements the core dual-stream rendering logic:
        - Focused: Show last 8 N-chunks + memories for ALL non-rendered N-chunks  
        - Unfocused: Show all memories + current N-chunk only
        
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
            
            if is_focused:
                logger.debug(f"Focused rendering for {element_id}: {len(n_chunks)} N-chunks, {len(m_chunks)} M-chunks")
                
                # Focused: Show last 8 N-chunks + memories for missing chunks
                if len(n_chunks) <= 8:
                    # All N-chunks fit, show all fresh content + no additional memories needed
                    fresh_content = n_chunks
                    memory_content = []
                    logger.debug(f"All {len(n_chunks)} N-chunks fit in 8-chunk window")
                else:
                    # Show last 8 N-chunks + memories for earlier chunks
                    fresh_content = n_chunks[-8:]  # Last 8 chunks as fresh
                    non_rendered_count = len(n_chunks) - 8  # How many chunks not shown fresh
                    memory_content = m_chunks[:non_rendered_count]  # Memories for missing chunks
                    logger.debug(f"8-chunk window: showing last 8 of {len(n_chunks)} N-chunks + {len(memory_content)} memories for missing chunks")
                
                # Return: memories first, then fresh content (chronological order)
                return await self._flatten_chunks_for_rendering(memory_content + fresh_content)
                
            else:
                logger.debug(f"Unfocused rendering for {element_id}: {len(m_chunks)} M-chunks + current N-chunk")
                
                # Unfocused: Show all memories + current N-chunk only
                current_chunk = n_chunks[-1:] if n_chunks else []
                return await self._flatten_chunks_for_rendering(m_chunks + current_chunk)
                
        except Exception as e:
            logger.error(f"Error selecting rendering stream for {element_id}: {e}", exc_info=True)
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
        
        This uses HUD's new render_memorization_context_veil method to get the full VEIL context
        while excluding the specific content being memorized to avoid duplication.
        
        Args:
            element_id: ID of the element being compressed
            children: VEIL children being memorized (to exclude from context)
            
        Returns:
            Rendered memorization context string from HUD, or None if HUD not available
        """
        try:
            # Get HUD component and SpaceVeilProducer
            hud_component = self._get_hud_component()
            if not hud_component:
                logger.debug(f"HUD component not available for memorization context generation for {element_id}")
                return None
            
            # Get SpaceVeilProducer to obtain full VEIL
            if not self.owner:
                logger.debug(f"No owner InnerSpace available for VEIL access")
                return None
                
            veil_producer = None
            if hasattr(self.owner, '_components'):
                for component in self.owner._components:
                    if hasattr(component, 'COMPONENT_TYPE') and component.COMPONENT_TYPE == "SpaceVeilProducer":
                        veil_producer = component
                        break
            
            if not veil_producer:
                logger.debug(f"SpaceVeilProducer not available for memorization context")
                return None
            
            # Get full VEIL from SpaceVeilProducer
            full_veil = veil_producer.get_full_veil()
            if not full_veil:
                logger.debug(f"Empty VEIL from SpaceVeilProducer")
                return None
            
            # Use HUD's specialized memorization context rendering
            memorization_context = await hud_component.render_memorization_context_veil(
                full_veil=full_veil,
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
