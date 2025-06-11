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

from .base_component import Component
from elements.component_registry import register_component

# Import LLM interfaces
from llm.provider_interface import LLMMessage

# NEW: Import storage system
from storage import create_storage_from_env, StorageInterface

if TYPE_CHECKING:
    from elements.elements.inner_space import InnerSpace

logger = logging.getLogger(__name__)


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
    
    # NEW: Rolling compression token limits
    FOCUSED_MEMORY_LIMIT = 20000      # 20k tokens of memories for focused elements
    FOCUSED_FRESH_LIMIT = 50000       # 50k tokens of fresh content for focused elements  
    UNFOCUSED_TOTAL_LIMIT = 4000      # 4k tokens total for unfocused elements
    
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
        
        logger.info(f"CompressionEngineComponent initialized ({self.id}) - ready for AgentMemoryCompressor integration")
    
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
        try:
            if not full_veil:
                logger.warning(f"CompressionEngine received empty VEIL for processing")
                return full_veil
            
            focus_element_id = focus_context.get('focus_element_id') if focus_context else None
            
            if focus_element_id:
                logger.info(f"Processing VEIL with compression, focusing on element: {focus_element_id}")
                processed_veil = await self._process_focused_veil(full_veil, focus_element_id, memory_data)
            else:
                logger.info(f"Processing VEIL with compression, no focus (full context)")
                processed_veil = await self._process_full_veil_with_memory(full_veil, memory_data)
            
            logger.debug(f"VEIL processing complete: preserved structure with compression applied")
            return processed_veil
            
        except Exception as e:
            logger.error(f"Error processing VEIL with compression: {e}", exc_info=True)
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
        
        ENHANCED: Now implements focus-aware compression with different limits:
        - Focused elements: 50k fresh + 20k memories (70k total)
        - Unfocused elements: 4k total (strict compression)
        
        Phase 3: This uses the agent's LLM-based reflection instead of simple content counting.
        NEW: Uses just-in-time content change detection via flat VEIL cache comparison.
        
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
            
            # NEW: Determine if this element is focused
            is_focused = (element_id == focus_element_id)
            
            if is_focused:
                logger.debug(f"Applying focused compression for element {element_id}")
                await self._apply_focused_compression(container_node, element_id, children, element_name, available_tools)
            else:
                logger.debug(f"Applying unfocused compression for element {element_id}")
                await self._apply_unfocused_compression(container_node, element_id, children, element_name, available_tools)
                
        except Exception as e:
            logger.error(f"Error compressing container children: {e}", exc_info=True)

    async def _apply_focused_compression(self, container_node: Dict[str, Any], element_id: str, children: List[Dict[str, Any]], element_name: str, available_tools: List[str]) -> None:
        """
        Apply focused compression with lax limits for focused elements.
        
        ENHANCED Phase 2: Now implements true rolling compression!
        For focused elements: 50k fresh content + 20k memories = 70k total
        
        Rolling compression strategy:
        1. Separate existing memories from fresh content
        2. Check if we exceed token limits 
        3. If fresh content > 50k: compress oldest fresh content into memories
        4. If memories > 20k: recompress oldest memories into higher-level summaries
        5. This creates a rolling window preserving recent content
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
            
            logger.info(f"Focused element {element_id}: {fresh_tokens} fresh tokens, {memory_tokens} memory tokens (limits: {self.FOCUSED_FRESH_LIMIT} fresh, {self.FOCUSED_MEMORY_LIMIT} memory)")
            
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
            
            # NEW Phase 2: Apply rolling compression as needed
            final_memories = existing_memories.copy()
            final_content = fresh_content.copy()
            
            # Step 1: Compress excess fresh content if needed
            if needs_fresh_compression:
                logger.info(f"Fresh content exceeds limit by {fresh_tokens - self.FOCUSED_FRESH_LIMIT} tokens, applying rolling compression")
                new_memories, preserved_content = await self._compress_excess_fresh_content(
                    element_id, element_name, available_tools, fresh_content
                )
                final_memories.extend(new_memories)
                final_content = preserved_content
                
                # Recalculate memory tokens after adding new memories
                memory_tokens = self._calculate_memory_tokens(final_memories)
            
            # Step 2: Recompress excess memories if needed
            if memory_tokens > self.FOCUSED_MEMORY_LIMIT:
                logger.info(f"Memory content exceeds limit by {memory_tokens - self.FOCUSED_MEMORY_LIMIT} tokens, recompressing memories")
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
            # Fallback to simple compression
            if self._memory_compressor:
                memory_summary, compression_approach = await self._create_new_agent_memory(
                    element_id, element_name, available_tools, children, is_focused=True
                )
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
        
        For unfocused elements: 4k total (everything compressed)
        """
        try:
            if not children:
                # Handle empty containers
                memory_summary = "Empty conversation - no content to remember"
                compression_approach = "empty_container_unfocused"
                logger.debug(f"Empty unfocused container {element_id}: no children to compress")
            elif self._memory_compressor:
                # NEW: Just-in-time content change detection for unfocused elements
                existing_memory_id = self._memory_compressor.get_memory_for_elements([element_id])
                
                if existing_memory_id:
                    # Check if content has changed since last compression
                    content_changed = await self._has_content_changed_since_compression(
                        element_id, existing_memory_id, children
                    )
                    
                    if content_changed:
                        logger.info(f"Content changed for unfocused {element_id}, recompressing memory {existing_memory_id}")
                        # Invalidate and recompress
                        self._memory_compressor.invalidate_memory(existing_memory_id)
                        memory_summary, compression_approach = await self._create_new_agent_memory(
                            element_id, element_name, available_tools, children, is_recompression=True, is_focused=False
                        )
                    else:
                        # Use existing valid memory
                        logger.debug(f"Content unchanged for unfocused {element_id}, using existing memory")
                        existing_memory = await self._memory_compressor.load_memory(existing_memory_id)
                        if existing_memory and existing_memory.get("memorized_node"):
                            memory_summary = existing_memory["memorized_node"]["properties"].get("memory_summary", "Memory load failed")
                            compression_approach = "existing_agent_memory_unfocused"
                        else:
                            # Fallback if memory load failed
                            memory_summary = await self._generate_content_summary(children, element_id)
                            compression_approach = "memory_load_fallback_unfocused"
                else:
                    # Create new memory - first time compressing this content
                    logger.info(f"Creating new AgentMemoryCompressor memory for unfocused container {element_id} with {len(children)} children")
                    memory_summary, compression_approach = await self._create_new_agent_memory(
                        element_id, element_name, available_tools, children, is_recompression=False, is_focused=False
                    )
            else:
                # Fallback to simple summary if AgentMemoryCompressor not available
                logger.debug(f"AgentMemoryCompressor not available, using simple summary for unfocused {element_id}")
                memory_summary = await self._generate_content_summary(children, element_id)
                compression_approach = "simple_summary_fallback_unfocused"
            
            # Create content memory node (same structure as before but marked as unfocused)
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
                    "agent_reflected": compression_approach in ["agent_memory_compressor_unfocused", "agent_memory_recompressor_unfocused", "existing_agent_memory_unfocused"],
                    "is_focused": False  # NEW: Mark as unfocused compression
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
        
        Returns:
            Tuple of (memory_summary, compression_approach)
        """
        try:
            element_ids = [element_id]
            
            # NEW: Add focus context to compression metadata
            focus_type = "focused" if is_focused else "unfocused"
            action_type = "recompression" if is_recompression else "compression"
            
            compression_context = {
                "element_id": element_id,
                "element_name": element_name,
                "element_type": "container",
                "compression_reason": f"{focus_type}_content_{action_type}",
                "available_tools": available_tools,
                "content_fingerprint": self._calculate_content_fingerprint(children),  # Store fingerprint
                "is_focused": is_focused,  # NEW: Focus context
                "focus_type": focus_type  # NEW: Focus type label
            }
            
            memorized_veil_node = await self._memory_compressor.compress(
                raw_veil_nodes=children,
                element_ids=element_ids,
                compression_context=compression_context
            )
            
            if memorized_veil_node:
                memory_summary = memorized_veil_node.get("properties", {}).get("memory_summary", "Agent memory formation failed")
                
                # NEW: Generate focus-aware compression approach labels
                if is_recompression:
                    compression_approach = f"agent_memory_recompressor_{focus_type}"
                else:
                    compression_approach = f"agent_memory_compressor_{focus_type}"
                
                action = "Recompressed" if is_recompression else "Agent reflected on"
                logger.info(f"{action} {focus_type} container {element_id}: {memory_summary[:100]}...")
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

    async def _compress_excess_fresh_content(self, element_id: str, element_name: str, available_tools: List[str], 
                                           fresh_content: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        NEW Phase 2: Compress fresh content that exceeds focused element limits.
        
        This is the core of rolling compression - we compress the OLDEST fresh content
        into memories while preserving the NEWEST content for immediate context.
        
        Args:
            element_id: ID of element being compressed
            element_name: Name of element for context
            available_tools: Available tools for this element
            fresh_content: List of fresh (non-memory) VEIL nodes
            
        Returns:
            Tuple of (new_memories, preserved_fresh_content)
        """
        try:
            if not fresh_content:
                return [], []
            
            # Calculate how much content to compress
            total_fresh_tokens = self._calculate_children_tokens(fresh_content)
            excess_tokens = total_fresh_tokens - self.FOCUSED_FRESH_LIMIT
            
            if excess_tokens <= 0:
                # Within limits, preserve everything
                return [], fresh_content
            
            logger.info(f"Compressing {excess_tokens} excess tokens from {element_id}")
            
            # Find compression boundary (oldest content to compress)
            compression_boundary = self._find_compression_boundary(fresh_content, excess_tokens)
            
            if compression_boundary <= 0:
                # Can't compress anything meaningful
                logger.warning(f"Cannot find valid compression boundary for {element_id}")
                return [], fresh_content
            
            # Split content: oldest to compress, newest to preserve
            content_to_compress = fresh_content[:compression_boundary]
            content_to_preserve = fresh_content[compression_boundary:]
            
            logger.info(f"Rolling compression: compressing {len(content_to_compress)} items, preserving {len(content_to_preserve)} items")
            
            # Create memory chunks from compressed content
            new_memories = await self._create_memory_chunks(content_to_compress, element_id, element_name, available_tools)
            
            return new_memories, content_to_preserve
            
        except Exception as e:
            logger.error(f"Error compressing excess fresh content: {e}", exc_info=True)
            # Fallback: preserve everything
            return [], fresh_content

    def _find_compression_boundary(self, fresh_content: List[Dict[str, Any]], excess_tokens: int) -> int:
        """
        NEW Phase 2: Find where to split content for rolling compression.
        
        Strategy: Compress from oldest (beginning) until we remove enough tokens.
        This preserves the most recent content for immediate context.
        
        Args:
            fresh_content: List of fresh content nodes (should be in chronological order)
            excess_tokens: Number of tokens we need to remove
            
        Returns:
            Index where to split (content[0:boundary] gets compressed, content[boundary:] preserved)
        """
        try:
            if not fresh_content or excess_tokens <= 0:
                return 0
            
            tokens_to_remove = 0
            boundary = 0
            
            # Go through content from oldest to newest
            for i, content_item in enumerate(fresh_content):
                item_tokens = self._calculate_children_tokens([content_item])
                tokens_to_remove += item_tokens
                boundary = i + 1
                
                # Stop when we've removed enough tokens
                if tokens_to_remove >= excess_tokens:
                    break
            
            # Ensure we don't compress everything (leave at least 25% of content)
            min_preserve = max(1, len(fresh_content) // 4)
            max_compress = len(fresh_content) - min_preserve
            
            boundary = min(boundary, max_compress)
            
            logger.debug(f"Compression boundary: compress {boundary}/{len(fresh_content)} items ({tokens_to_remove} tokens)")
            return boundary
            
        except Exception as e:
            logger.error(f"Error finding compression boundary: {e}", exc_info=True)
            # Fallback: compress half
            return len(fresh_content) // 2

    async def _create_memory_chunks(self, content_to_compress: List[Dict[str, Any]], element_id: str, 
                                  element_name: str, available_tools: List[str]) -> List[Dict[str, Any]]:
        """
        NEW Phase 2: Create memory chunks from content that needs to be compressed.
        
        Strategy: Break large content into optimal-sized chunks for better memory formation.
        Each chunk should be meaningful and not exceed our memory token limits.
        
        Args:
            content_to_compress: Content nodes to compress into memories
            element_id: ID of element for context
            element_name: Name of element for context
            available_tools: Available tools for context
            
        Returns:
            List of memory VEIL nodes
        """
        try:
            if not content_to_compress:
                return []
            
            memories = []
            
            # For now, create one memory per reasonable chunk
            # Future enhancement: intelligently group related content
            chunk_size = min(10, len(content_to_compress))  # Max 10 items per memory
            
            for i in range(0, len(content_to_compress), chunk_size):
                chunk = content_to_compress[i:i + chunk_size]
                
                # Create memory for this chunk using AgentMemoryCompressor
                if self._memory_compressor:
                    memory_summary, compression_approach = await self._create_new_agent_memory(
                        f"{element_id}_chunk_{i//chunk_size}", element_name, available_tools, chunk, 
                        is_recompression=False, is_focused=True
                    )
                else:
                    # Fallback summary
                    memory_summary = f"Memory chunk {i//chunk_size + 1}: {len(chunk)} items from {element_name}"
                    compression_approach = "chunk_fallback_summary"
                
                # Create memory node
                memory_node = {
                    "veil_id": f"rolling_memory_{element_id}_{i//chunk_size}_{self.id}",
                    "node_type": "content_memory",
                    "properties": {
                        "structural_role": "compressed_content",
                        "content_nature": "content_memory",
                        "original_element_id": element_id,
                        "memory_summary": memory_summary,
                        "original_child_count": len(chunk),
                        "compression_timestamp": datetime.now().isoformat(),
                        "compression_approach": compression_approach,
                        "is_focused": True,
                        "is_rolling_memory": True,  # Mark as rolling compression result
                        "chunk_index": i // chunk_size
                    },
                    "children": []
                }
                
                memories.append(memory_node)
                logger.debug(f"Created rolling memory chunk {i//chunk_size}: {memory_summary[:60]}...")
            
            logger.info(f"Created {len(memories)} memory chunks from {len(content_to_compress)} content items")
            return memories
            
        except Exception as e:
            logger.error(f"Error creating memory chunks: {e}", exc_info=True)
            return []

    async def _recompress_excess_memories(self, element_id: str, element_name: str, 
                                        memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        NEW Phase 2: Recompress memories when they exceed the memory token limit.
        
        Strategy: Merge oldest memories into higher-level summaries while preserving
        the most recent memories for better context continuity.
        
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
            
            logger.info(f"Recompressing {len(memories)} memories ({current_tokens} tokens) for {element_id}")
            
            # Strategy: Keep newest memories, merge oldest ones
            # Sort memories by timestamp (oldest first)
            sorted_memories = self._sort_memories_by_age(memories)
            
            final_memories = []
            current_memory_tokens = 0
            tokens_to_remove = current_tokens - self.FOCUSED_MEMORY_LIMIT
            
            # Start from newest memories, keep as many as possible
            for memory in reversed(sorted_memories):
                memory_tokens = self._calculate_memory_tokens([memory])
                
                if current_memory_tokens + memory_tokens <= self.FOCUSED_MEMORY_LIMIT:
                    final_memories.insert(0, memory)  # Insert at beginning to maintain order
                    current_memory_tokens += memory_tokens
                else:
                    break
            
            # If we removed memories, create a higher-level summary of the removed ones
            if len(final_memories) < len(sorted_memories):
                removed_memories = sorted_memories[:len(sorted_memories) - len(final_memories)]
                
                if removed_memories:
                    # Create summary of removed memories
                    summary_memory = await self._create_memory_summary(removed_memories, element_id, element_name)
                    
                    # Check if we have room for the summary
                    summary_tokens = self._calculate_memory_tokens([summary_memory])
                    
                    if current_memory_tokens + summary_tokens <= self.FOCUSED_MEMORY_LIMIT:
                        final_memories.insert(0, summary_memory)  # Insert at beginning (oldest)
                    else:
                        # If summary doesn't fit, remove one more recent memory to make room
                        if final_memories:
                            removed_memory = final_memories.pop()
                            logger.info(f"Removed one more memory to make room for summary")
                        final_memories.insert(0, summary_memory)
            
            final_tokens = self._calculate_memory_tokens(final_memories)
            logger.info(f"Memory recompression complete: {len(memories)} → {len(final_memories)} memories, {current_tokens} → {final_tokens} tokens")
            
            return final_memories
            
        except Exception as e:
            logger.error(f"Error recompressing memories: {e}", exc_info=True)
            # Fallback: keep newest memories up to limit
            return memories[-3:] if len(memories) > 3 else memories  # Keep last 3 memories

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