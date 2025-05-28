"""
Compression Engine Component
Handles agent memory storage and provides compressed/full memory context to AgentLoop.
Combines memory management with future compression capabilities.
"""

import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime
import json

from .base_component import Component
from elements.component_registry import register_component

# Import LLM interfaces
from llm.provider_interface import LLMMessage

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
    
    This component centralizes memory management and will evolve to include
    sophisticated compression algorithms for maintaining agent continuity
    within context limits.
    """
    
    COMPONENT_TYPE = "CompressionEngineComponent"
    HANDLED_EVENT_TYPES = ["reasoning_chain_stored", "memory_requested"]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Memory storage - list of interaction records
        self._memory_interactions: List[Dict[str, Any]] = []
        
        # Orientation conversation - prerecorded messages that establish agent context
        self._orientation_messages: List[LLMMessage] = []
        self._orientation_initialized = False
        
        # Track agent identity for consistent messaging
        self._agent_name: Optional[str] = None
        
        # Configuration
        self._max_memory_interactions = None  # Unlimited for now
        self._compression_enabled = False     # Future feature
        
        logger.info(f"CompressionEngineComponent initialized ({self.id})")
    
    def _on_initialize(self) -> bool:
        """Initialize the compression engine after being attached to InnerSpace."""
        try:
            # Try to get agent name from parent InnerSpace if available
            if hasattr(self.owner, 'agent_name'):
                self._agent_name = self.owner.agent_name
                logger.info(f"CompressionEngine initialized for agent: {self._agent_name}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize CompressionEngineComponent: {e}", exc_info=True)
            return False
    
    async def get_memory_context(self) -> List[LLMMessage]:
        """
        Retrieve all stored memory as LLM message format for AgentLoop.
        Includes orientation conversation (if set) plus all stored interactions.
        
        For now, returns full uncompressed history. Future versions will
        implement intelligent compression strategies.
        
        Returns:
            List of LLMMessage objects representing the agent's complete memory
        """
        memory_messages: List[LLMMessage] = []
        
        try:
            # Start with orientation conversation (if available)
            if self._orientation_messages:
                memory_messages.extend(self._orientation_messages)
                logger.debug(f"Added {len(self._orientation_messages)} orientation messages")
            
            # Add stored interaction history
            for interaction in self._memory_interactions:
                # Add the context that was received
                context_received = interaction.get("context_received")
                if context_received:
                    memory_messages.append(
                        LLMMessage(role="user", content=context_received)
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
            logger.debug(f"CompressionEngine provided {total_messages} total messages ({len(self._orientation_messages)} orientation + {total_interactions} interactions)")
            return memory_messages
            
        except Exception as e:
            logger.error(f"Error retrieving memory context: {e}", exc_info=True)
            return []
    
    def _format_tool_interaction_summary(self, interaction: Dict[str, Any]) -> Optional[str]:
        """Format tool calls and results into a readable summary."""
        tool_calls = interaction.get("tool_calls", [])
        tool_results = interaction.get("tool_results", [])
        
        if not tool_calls and not tool_results:
            return None
        
        summary_parts = []
        
        if tool_calls:
            summary_parts.append(f"Called {len(tool_calls)} tool(s):")
            for i, call in enumerate(tool_calls):
                tool_name = call.get("tool_name", "unknown")
                parameters = call.get("parameters", {})
                # Truncate parameters for readability
                params_str = str(parameters)[:100] + "..." if len(str(parameters)) > 100 else str(parameters)
                summary_parts.append(f"  {i+1}. {tool_name}({params_str})")
        
        if tool_results:
            summary_parts.append(f"Results:")
            for i, result in enumerate(tool_results):
                result_str = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                summary_parts.append(f"  {i+1}. {result_str}")
        
        return "\n".join(summary_parts)
    
    async def store_reasoning_chain(self, chain_data: Dict[str, Any]) -> None:
        """
        Store a complete reasoning chain from the AgentLoop.
        
        Args:
            chain_data: Dictionary containing:
                - context_received: The input context from HUD
                - agent_response: The agent's response/reasoning
                - tool_calls: List of tool calls made (if any)
                - tool_results: List of tool results received (if any)
                - reasoning_notes: Additional reasoning context
                - metadata: Any additional metadata
        """
        try:
            interaction_record = {
                "timestamp": datetime.now(),
                "context_received": chain_data.get("context_received"),
                "agent_response": chain_data.get("agent_response"),
                "tool_calls": chain_data.get("tool_calls", []),
                "tool_results": chain_data.get("tool_results", []),
                "reasoning_notes": chain_data.get("reasoning_notes"),
                "metadata": chain_data.get("metadata", {}),
                "agent_name": self._agent_name
            }
            
            self._memory_interactions.append(interaction_record)
            
            # Log storage (with truncation for readability)
            response_preview = str(chain_data.get("agent_response", ""))[:100]
            tool_count = len(chain_data.get("tool_calls", []))
            
            logger.info(f"Stored reasoning chain: {response_preview}{'...' if len(response_preview) == 100 else ''} (tools: {tool_count})")
            
            # Emit event to timeline for tracking
            if hasattr(self.owner, 'add_event_to_primary_timeline'):
                self.owner.add_event_to_primary_timeline({
                    "event_type": "reasoning_chain_stored",
                    "data": {
                        "compression_engine_id": self.id,
                        "interaction_count": len(self._memory_interactions),
                        "tool_calls_count": tool_count,
                        "timestamp": interaction_record["timestamp"].isoformat()
                    }
                })
                
        except Exception as e:
            logger.error(f"Error storing reasoning chain: {e}", exc_info=True)
    
    async def get_recent_interactions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent stored interactions for debugging or inspection.
        
        Args:
            limit: Maximum number of interactions to return (None for all)
            
        Returns:
            List of interaction records
        """
        interactions = self._memory_interactions
        if limit:
            interactions = interactions[-limit:]
        
        # Return copies to prevent external modification
        return [interaction.copy() for interaction in interactions]
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memory."""
        total_interactions = len(self._memory_interactions)
        total_tool_calls = sum(len(interaction.get("tool_calls", [])) for interaction in self._memory_interactions)
        
        oldest_timestamp = None
        newest_timestamp = None
        if self._memory_interactions:
            oldest_timestamp = self._memory_interactions[0]["timestamp"]
            newest_timestamp = self._memory_interactions[-1]["timestamp"]
        
        return {
            "total_interactions": total_interactions,
            "total_tool_calls": total_tool_calls,
            "agent_name": self._agent_name,
            "oldest_interaction": oldest_timestamp.isoformat() if oldest_timestamp else None,
            "newest_interaction": newest_timestamp.isoformat() if newest_timestamp else None,
            "compression_enabled": self._compression_enabled
        }
    
    async def clear_memory(self) -> bool:
        """Clear all stored memory. Use with caution."""
        try:
            interaction_count = len(self._memory_interactions)
            self._memory_interactions.clear()
            
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