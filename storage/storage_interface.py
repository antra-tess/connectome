"""
Abstract storage interface for memory persistence.

Defines the contract that all storage backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StorageInterface(ABC):
    """
    Abstract base class for all storage backends.
    
    Provides a unified interface for storing and retrieving conversation data,
    memories, reasoning chains, and system state across different storage systems.
    """
    
    def __init__(self, storage_config: Dict[str, Any]):
        """
        Initialize the storage backend.
        
        Args:
            storage_config: Configuration dictionary specific to the storage type
        """
        self.config = storage_config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the storage backend.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Properly shutdown the storage backend.
        
        Returns:
            True if shutdown succeeded, False otherwise
        """
        pass
    
    # ===== Conversation Storage =====
    
    @abstractmethod
    async def store_conversation(self, conversation_id: str, data: Dict[str, Any]) -> bool:
        """
        Store conversation data.
        
        Args:
            conversation_id: Unique identifier for the conversation
            data: Complete conversation data in Typingcloud format
            
        Returns:
            True if storage succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    async def load_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load conversation data.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            Conversation data dict or None if not found
        """
        pass
    
    @abstractmethod
    async def list_conversations(self) -> List[str]:
        """
        List all stored conversation IDs.
        
        Returns:
            List of conversation IDs
        """
        pass
    
    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all associated data.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        pass
    
    # ===== Raw Messages Storage =====
    
    @abstractmethod
    async def store_raw_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> bool:
        """
        Store raw uncompressed messages for a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            messages: List of raw message objects
            
        Returns:
            True if storage succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    async def load_raw_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Load raw uncompressed messages for a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            List of raw message objects
        """
        pass
    
    # ===== Memory Storage =====
    
    @abstractmethod
    async def store_memory(self, conversation_id: str, memory_id: str, memory_data: Dict[str, Any]) -> bool:
        """
        Store a compressed memory.
        
        Args:
            conversation_id: Unique identifier for the conversation
            memory_id: Unique identifier for the memory
            memory_data: Complete memory formation data (6-message sequence)
            
        Returns:
            True if storage succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    async def load_memories(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Load all memories for a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            List of memory objects in chronological order
        """
        pass
    
    # ===== Reasoning Chain Storage =====
    
    @abstractmethod
    async def store_reasoning_chain(self, agent_id: str, chain_data: Dict[str, Any]) -> bool:
        """
        Store an agent's reasoning chain.
        
        Args:
            agent_id: Unique identifier for the agent
            chain_data: Complete reasoning chain data
            
        Returns:
            True if storage succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    async def load_reasoning_chains(self, agent_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load reasoning chains for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            limit: Maximum number of chains to return (None for all)
            
        Returns:
            List of reasoning chain objects in chronological order
        """
        pass
    
    # ===== Chunk and Cache Storage =====
    
    @abstractmethod
    async def store_chunk_info(self, conversation_id: str, chunk_data: Dict[str, Any]) -> bool:
        """
        Store chunking information for a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            chunk_data: Chunk boundaries, token counts, and references
            
        Returns:
            True if storage succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    async def load_chunk_info(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load chunking information for a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            Chunk information dict or None if not found
        """
        pass
    
    @abstractmethod
    async def store_cache_entry(self, cache_key: str, cache_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Store a cache entry (e.g., LLM generation cache).
        
        Args:
            cache_key: Unique cache key (e.g., chunk hash)
            cache_data: Data to cache
            ttl: Time to live in seconds (None for no expiration)
            
        Returns:
            True if storage succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    async def load_cache_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Load a cache entry.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            Cached data dict or None if not found/expired
        """
        pass
    
    # ===== System State Storage =====
    
    @abstractmethod
    async def store_system_state(self, state_key: str, state_data: Dict[str, Any]) -> bool:
        """
        Store system state information.
        
        Args:
            state_key: Unique identifier for the state
            state_data: System state data
            
        Returns:
            True if storage succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    async def load_system_state(self, state_key: str) -> Optional[Dict[str, Any]]:
        """
        Load system state information.
        
        Args:
            state_key: Unique identifier for the state
            
        Returns:
            System state data dict or None if not found
        """
        pass
    
    # ===== Utility Methods =====
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health and status of the storage backend.
        
        Returns:
            Health status information
        """
        pass
    
    @abstractmethod
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage usage statistics.
        
        Returns:
            Storage statistics and usage information
        """
        pass 