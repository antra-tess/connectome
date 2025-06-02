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
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Memory storage - list of interaction records (in-memory cache)
        self._memory_interactions: List[Dict[str, Any]] = []
        
        # Orientation conversation - prerecorded messages that establish agent context
        self._orientation_messages: List[LLMMessage] = []
        self._orientation_initialized = False
        
        # Track agent identity for consistent messaging
        self._agent_name: Optional[str] = None
        
        # Configuration
        self._max_memory_interactions = None  # Unlimited for now
        self._compression_enabled = False     # Future feature
        
        # NEW: Pluggable storage backend
        self._storage: Optional[StorageInterface] = None
        self._storage_initialized = False
        
        # NEW: Conversation tracking for proper storage
        self._conversation_id: Optional[str] = None
        
        logger.info(f"CompressionEngineComponent initialized ({self.id})")
    
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
                self._memory_interactions = reasoning_chains
                logger.info(f"Loaded {len(reasoning_chains)} reasoning chains for agent {agent_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data from storage: {e}", exc_info=True)
            return False
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown the compression engine and storage."""
        try:
            if self._storage:
                logger.info(f"Shutting down storage backend for CompressionEngine {self.id}")
                await self._storage.shutdown()
            return True
        except Exception as e:
            logger.error(f"Error during CompressionEngine shutdown: {e}", exc_info=True)
            return False
    
    async def get_memory_context(self) -> List[LLMMessage]:
        """
        Retrieve stored interaction history as LLM message format for AgentLoop.
        
        NEW APPROACH: Instead of storing full HUD contexts (which create redundancy),
        we store interaction summaries focusing on what the agent learned/decided/did.
        
        The current HUD context provides fresh state; memory provides interaction history.
        
        Returns:
            List of LLMMessage objects representing interaction history (not redundant contexts)
        """
        memory_messages: List[LLMMessage] = []
        
        try:
            # Start with orientation conversation (if available)
            if self._orientation_messages:
                memory_messages.extend(self._orientation_messages)
                logger.debug(f"Added {len(self._orientation_messages)} orientation messages")
            
            # Add stored interaction history (focused on what happened, not full contexts)
            for interaction in self._memory_interactions:
                # NEW: Instead of storing full context_received, store interaction summary
                interaction_summary = self._create_interaction_summary(interaction)
                if interaction_summary:
                    memory_messages.append(
                        LLMMessage(role="user", content=interaction_summary)
                    )
                
                # Add the agent's response/reasoning
                agent_response = interaction.get("agent_response")
                if agent_response:
                    memory_messages.append(
                        LLMMessage(role="assistant", content=agent_response)
                    )
                
                # Add tool interaction summary if present
                tool_summary = self._format_tool_interaction_summary(interaction)
                if tool_summary:
                    memory_messages.append(
                        LLMMessage(role="user", content=f"[Tool Results]: {tool_summary}")
                    )
            
            total_interactions = len(self._memory_interactions)
            total_messages = len(memory_messages)
            logger.info(f"CompressionEngine provided {total_messages} total messages ({len(self._orientation_messages)} orientation + {total_interactions} interaction summaries)")
            return memory_messages
            
        except Exception as e:
            logger.error(f"Error retrieving memory context: {e}", exc_info=True)
            return []
    
    def _create_interaction_summary(self, interaction: Dict[str, Any]) -> Optional[str]:
        """
        Create a focused summary of what happened in this interaction.
        
        Instead of storing the full HUD context (which contains redundant message history),
        we extract the key information about what the user was asking or what changed.
        
        Args:
            interaction: The stored interaction record
            
        Returns:
            A concise summary of the interaction, or None if no meaningful summary can be created
        """
        try:
            # Try to extract meaningful information from the interaction
            tool_calls = interaction.get("tool_calls", [])
            tool_results = interaction.get("tool_results", [])
            reasoning_notes = interaction.get("reasoning_notes", "")
            metadata = interaction.get("metadata", {})
            context_summary = interaction.get("context_summary", "")
            
            # If there were tool calls, focus on what the agent did
            if tool_calls:
                tool_actions = []
                for i, tool_call in enumerate(tool_calls):
                    tool_name = tool_call.get("tool_name", "unknown_tool")
                    # Clean up prefixed tool names for readability
                    clean_tool_name = tool_name.split("__")[-1] if "__" in tool_name else tool_name
                    
                    # Get result status if available
                    if i < len(tool_results) and tool_results[i]:
                        result = tool_results[i]
                        if isinstance(result, dict):
                            if result.get("success"):
                                tool_actions.append(f"Successfully used {clean_tool_name}")
                            elif "error" in result:
                                tool_actions.append(f"Attempted {clean_tool_name} (encountered error)")
                            else:
                                tool_actions.append(f"Used {clean_tool_name}")
                        else:
                            tool_actions.append(f"Used {clean_tool_name}")
                    else:
                        tool_actions.append(f"Used {clean_tool_name}")
                
                if tool_actions:
                    # Include context info if available
                    context_info = f" (Context: {context_summary})" if context_summary else ""
                    return f"[Previous Interaction]: Agent {', '.join(tool_actions)}{context_info}"
            
            # If no tool calls but there's reasoning, use that
            if reasoning_notes and "cycle" in reasoning_notes:
                cycle_type = "multi-step" if "multi-step" in reasoning_notes else "simple"
                context_info = f" in context of: {context_summary}" if context_summary else ""
                return f"[Previous Interaction]: Agent completed {cycle_type} reasoning cycle{context_info}"
            
            # If we have context summary but no tool calls, use that
            if context_summary:
                return f"[Previous Interaction]: Agent processed: {context_summary}"
            
            # Fallback: very generic summary
            return f"[Previous Interaction]: Agent processed request and responded"
            
        except Exception as e:
            logger.warning(f"Error creating interaction summary: {e}")
            return "[Previous Interaction]: Agent activity occurred"
    
    def _format_tool_interaction_summary(self, interaction: Dict[str, Any]) -> Optional[str]:
        """Format tool calls and results into a readable summary."""
        tool_calls = interaction.get("tool_calls", [])
        tool_results = interaction.get("tool_results", [])
        
        if not tool_calls and not tool_results:
            return None
        
        summary_parts = []
        
        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call.get("tool_name", "unknown_tool")
            summary_parts.append(f"Called {tool_name}")
            
            # Add result if available
            if i < len(tool_results) and tool_results[i]:
                result = tool_results[i]
                if isinstance(result, dict) and "success" in result:
                    status = "✓" if result.get("success") else "✗"
                    summary_parts.append(f"{status} {result.get('message', 'completed')}")
        
        return "; ".join(summary_parts) if summary_parts else None
    
    async def store_reasoning_chain(self, chain_data: Dict[str, Any]) -> None:
        """
        Store a complete reasoning chain from the AgentLoop.
        
        NEW APPROACH: Store interaction metadata and reasoning, not full HUD contexts.
        This eliminates redundancy while preserving essential interaction history.
        
        Args:
            chain_data: Dictionary containing:
                - context_received: The input context from HUD (NEW: we'll extract key info instead of storing full context)
                - agent_response: The agent's response/reasoning
                - tool_calls: List of tool calls made (if any)
                - tool_results: List of tool results received (if any)
                - reasoning_notes: Additional reasoning context
                - metadata: Any additional metadata
        """
        try:
            # NEW: Extract key information instead of storing full context
            context_received = chain_data.get("context_received", "")
            context_summary = self._extract_context_key_info(context_received)
            
            interaction_record = {
                "timestamp": datetime.now().isoformat(),
                # NEW: Store context summary instead of full context to eliminate redundancy
                "context_summary": context_summary,
                "context_length": len(context_received) if context_received else 0,
                "agent_response": chain_data.get("agent_response"),
                "tool_calls": chain_data.get("tool_calls", []),
                "tool_results": chain_data.get("tool_results", []),
                "reasoning_notes": chain_data.get("reasoning_notes"),
                "metadata": chain_data.get("metadata", {}),
                "agent_name": self._agent_name
            }
            
            # Store in memory for immediate access
            self._memory_interactions.append(interaction_record)
            
            # Persist to storage backend (ensure storage is ready first)
            if await self._ensure_storage_ready():
                agent_id = self._agent_name or self.id
                success = await self._storage.store_reasoning_chain(agent_id, interaction_record)
                if not success:
                    logger.error(f"Failed to persist reasoning chain to storage")
            else:
                logger.warning(f"Storage not ready, reasoning chain not persisted")
            
            # Log storage (with improved logging for new approach)
            response_preview = str(chain_data.get("agent_response", ""))[:100]
            tool_count = len(chain_data.get("tool_calls", []))
            
            logger.info(f"Stored interaction: {response_preview}{'...' if len(response_preview) == 100 else ''} (tools: {tool_count}, context: {len(context_received) if context_received else 0} chars)")
            
            # Store as raw messages for potential recompression (but with summary approach)
            if await self._ensure_storage_ready() and self._conversation_id:
                await self._store_interaction_messages(chain_data, context_summary)
            
        except Exception as e:
            logger.error(f"Error storing reasoning chain: {e}", exc_info=True)
    
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
        total_interactions = len(self._memory_interactions)
        total_tool_calls = sum(len(interaction.get("tool_calls", [])) for interaction in self._memory_interactions)
        
        oldest_timestamp = None
        newest_timestamp = None
        if self._memory_interactions:
            oldest_interaction = self._memory_interactions[0]
            newest_interaction = self._memory_interactions[-1]
            
            # Handle both datetime objects and string timestamps
            oldest_ts = oldest_interaction["timestamp"]
            newest_ts = newest_interaction["timestamp"]
            
            if isinstance(oldest_ts, str):
                oldest_timestamp = oldest_ts
            else:
                oldest_timestamp = oldest_ts.isoformat() if hasattr(oldest_ts, 'isoformat') else str(oldest_ts)
            
            if isinstance(newest_ts, str):
                newest_timestamp = newest_ts
            else:
                newest_timestamp = newest_ts.isoformat() if hasattr(newest_ts, 'isoformat') else str(newest_ts)
        
        # NEW: Get storage statistics (but don't force initialization)
        storage_stats = {}
        if self._storage_initialized:
            try:
                storage_stats = await self._storage.get_statistics()
            except Exception as e:
                logger.warning(f"Could not get storage statistics: {e}")
        
        return {
            "total_interactions": total_interactions,
            "total_tool_calls": total_tool_calls,
            "oldest_interaction": oldest_timestamp,
            "newest_interaction": newest_timestamp,
            "agent_name": self._agent_name,
            "conversation_id": self._conversation_id,
            "orientation_initialized": self._orientation_initialized,
            "orientation_message_count": len(self._orientation_messages),
            "storage_initialized": self._storage_initialized,
            "storage_stats": storage_stats
        }
    
    async def clear_memory(self) -> bool:
        """Clear all stored memory. Use with caution."""
        try:
            interaction_count = len(self._memory_interactions)
            self._memory_interactions.clear()
            
            # NEW: Also clear from persistent storage
            if await self._ensure_storage_ready() and self._conversation_id:
                await self._storage.delete_conversation(self._conversation_id)
                logger.info(f"Cleared persistent storage for conversation {self._conversation_id}")
            
            logger.warning(f"Cleared {interaction_count} memory interactions for agent {self._agent_name}")
            
            # Emit event
            if hasattr(self.owner, 'add_event_to_primary_timeline'):
                self.owner.add_event_to_primary_timeline({
                    "event_type": "memory_cleared",
                    "data": {
                        "compression_engine_id": self.id,
                        "cleared_interactions": interaction_count,
                        "timestamp": datetime.now().isoformat()
                    }
                })
            
            return True
        except Exception as e:
            logger.error(f"Error clearing memory: {e}", exc_info=True)
            return False
    
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
    
    async def set_orientation_conversation(self, orientation_messages: List[LLMMessage]) -> bool:
        """
        Set the orientation conversation that establishes agent context.
        This replaces traditional system prompts with natural conversation examples.
        
        Args:
            orientation_messages: List of LLMMessage objects representing the orientation conversation
            
        Returns:
            True if orientation was set successfully, False otherwise
        """
        try:
            self._orientation_messages = orientation_messages.copy() if orientation_messages else []
            self._orientation_initialized = True
            
            # NEW: Store orientation in persistent storage as system state
            if await self._ensure_storage_ready():
                orientation_data = {
                    "messages": [{"role": msg.role, "content": msg.content} for msg in self._orientation_messages],
                    "agent_name": self._agent_name,
                    "set_at": datetime.now().isoformat()
                }
                state_key = f"orientation_{self._agent_name or self.id}"
                await self._storage.store_system_state(state_key, orientation_data)
                logger.info(f"Stored orientation conversation in persistent storage")
            
            logger.info(f"Orientation conversation set with {len(self._orientation_messages)} messages")
            
            # Emit event to timeline for tracking
            if hasattr(self.owner, 'add_event_to_primary_timeline'):
                self.owner.add_event_to_primary_timeline({
                    "event_type": "orientation_conversation_set",
                    "data": {
                        "compression_engine_id": self.id,
                        "orientation_message_count": len(self._orientation_messages),
                        "timestamp": datetime.now().isoformat()
                    }
                })
            
            return True
        except Exception as e:
            logger.error(f"Error setting orientation conversation: {e}", exc_info=True)
            return False
    
    def get_orientation_conversation(self) -> List[LLMMessage]:
        """Get the current orientation conversation."""
        return self._orientation_messages.copy()
    
    def has_orientation(self) -> bool:
        """Check if orientation conversation has been set."""
        return self._orientation_initialized and bool(self._orientation_messages)
    
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