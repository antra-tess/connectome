"""
MemoryCompressor Interface

This module defines the abstract interface for memory compression strategies.
MemoryCompressors take raw VEIL nodes as input and produce Memorized VEIL nodes as output,
with correlation tracking and configurable compression rules.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryCompressor(ABC):
    """
    Abstract base class for memory compression strategies.
    
    A MemoryCompressor takes raw VEIL nodes and produces compressed memory representations
    while maintaining correlations and supporting re-compression.
    
    Key Features:
    - Token-aware compression (configurable limits)
    - Correlation tracking (element_ids -> memory_id mapping)
    - File storage for persistent memories
    - Re-compression capability for existing memories
    - Configurable compression rules per implementation
    """
    
    def __init__(self, 
                 token_limit: int = 4000,
                 storage_path: Optional[str] = None,
                 compressor_id: Optional[str] = None):
        """
        Initialize the memory compressor.
        
        Args:
            token_limit: Maximum tokens per memory chunk (default: 4000)
            storage_path: Path for memory storage (default: auto-generated)
            compressor_id: Unique identifier for this compressor instance
        """
        self.token_limit = token_limit
        self.storage_path = storage_path or self._default_storage_path()
        self.compressor_id = compressor_id or self._generate_compressor_id()
        
        # Correlation tracking: element_ids -> memory_id
        self._correlations: Dict[frozenset, str] = {}
        
        # Memory metadata cache: memory_id -> metadata
        self._memory_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Invalidation tracking
        self._invalidated_memories: Set[str] = set()
        
        logger.info(f"MemoryCompressor initialized: {self.__class__.__name__} "
                   f"(token_limit={token_limit}, id={self.compressor_id})")
    
    def _default_storage_path(self) -> str:
        """Generate default storage path for memories."""
        import os
        return os.path.join(os.getcwd(), "storage_data/memory_storage", self.__class__.__name__)
    
    def _generate_compressor_id(self) -> str:
        """Generate unique compressor ID."""
        import uuid
        return f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
    
    # --- Core Compression Interface ---
    
    @abstractmethod
    async def compress(self, 
                      raw_veil_nodes: List[Dict[str, Any]], 
                      element_ids: List[str],
                      compression_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compress raw VEIL nodes into a memorized VEIL node.
        
        Args:
            raw_veil_nodes: List of raw VEIL node dictionaries to compress
            element_ids: List of element IDs that this memory represents
            compression_context: Optional context for compression (agent context, etc.)
            
        Returns:
            Memorized VEIL node dictionary with:
            {
                "veil_id": "memory_<memory_id>",
                "node_type": "memorized_content",
                "properties": {
                    "structural_role": "compressed_memory",
                    "content_nature": "memorized_content",
                    "memory_id": "<unique_memory_id>",
                    "memory_summary": "<human_readable_summary>",
                    "original_element_ids": ["<element_id_1>", ...],
                    "original_node_count": <int>,
                    "compression_timestamp": "<iso_timestamp>",
                    "token_count": <int>,
                    "compressor_type": "<compressor_class_name>",
                    "compressor_version": "<version_string>",
                    "compression_metadata": <compressor_specific_data>
                },
                "children": []
            }
        """
        pass
    
    @abstractmethod
    async def estimate_tokens(self, raw_veil_nodes: List[Dict[str, Any]]) -> int:
        """
        Estimate token count for raw VEIL nodes.
        
        Args:
            raw_veil_nodes: List of raw VEIL node dictionaries
            
        Returns:
            Estimated token count using tiktoken or similar
        """
        pass
    
    # --- Correlation Management ---
    
    def get_memory_for_elements(self, element_ids: List[str]) -> Optional[str]:
        """
        Get memory ID for a specific set of element IDs.
        
        Args:
            element_ids: List of element IDs to check
            
        Returns:
            Memory ID if correlation exists, None otherwise
        """
        element_set = frozenset(element_ids)
        return self._correlations.get(element_set)
    
    def add_correlation(self, element_ids: List[str], memory_id: str) -> None:
        """
        Add correlation between element IDs and memory ID.
        
        Args:
            element_ids: List of element IDs
            memory_id: Memory ID to correlate with
        """
        element_set = frozenset(element_ids)
        self._correlations[element_set] = memory_id
        logger.debug(f"Added correlation: {element_ids} -> {memory_id}")
    
    def remove_correlation(self, element_ids: List[str]) -> Optional[str]:
        """
        Remove correlation for element IDs.
        
        Args:
            element_ids: List of element IDs
            
        Returns:
            Removed memory ID if it existed, None otherwise
        """
        element_set = frozenset(element_ids)
        return self._correlations.pop(element_set, None)
    
    def get_all_correlations(self) -> Dict[frozenset, str]:
        """Get all correlations as a copy."""
        return self._correlations.copy()
    
    # --- Memory Invalidation ---
    
    def invalidate_memory(self, memory_id: str) -> None:
        """
        Mark a memory as invalid for recompression.
        
        Args:
            memory_id: Memory ID to invalidate
        """
        self._invalidated_memories.add(memory_id)
        logger.info(f"Memory {memory_id} marked for recompression")
    
    def invalidate_memories_for_elements(self, changed_element_ids: List[str]) -> List[str]:
        """
        Invalidate all memories that include any of the changed elements.
        
        Args:
            changed_element_ids: List of element IDs that have changed
            
        Returns:
            List of invalidated memory IDs
        """
        invalidated = []
        changed_set = set(changed_element_ids)
        
        for element_set, memory_id in self._correlations.items():
            if changed_set.intersection(element_set):
                self.invalidate_memory(memory_id)
                invalidated.append(memory_id)
        
        logger.info(f"Invalidated {len(invalidated)} memories due to changes in {changed_element_ids}")
        return invalidated
    
    def is_memory_invalidated(self, memory_id: str) -> bool:
        """Check if a memory is marked for recompression."""
        return memory_id in self._invalidated_memories
    
    def clear_invalidation(self, memory_id: str) -> None:
        """Clear invalidation flag for a memory (after recompression)."""
        self._invalidated_memories.discard(memory_id)
    
    # --- Storage Interface ---
    
    @abstractmethod
    async def store_memory(self, memory_id: str, memorized_veil_node: Dict[str, Any]) -> bool:
        """
        Store a memorized VEIL node to persistent storage.
        
        Args:
            memory_id: Unique memory identifier
            memorized_veil_node: The memorized VEIL node to store
            
        Returns:
            True if storage succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    async def load_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a memorized VEIL node from persistent storage.
        
        Args:
            memory_id: Memory identifier to load
            
        Returns:
            Memorized VEIL node dictionary or None if not found
        """
        pass
    
    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory from persistent storage.
        
        Args:
            memory_id: Memory identifier to delete
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        pass
    
    # --- Recompression Support ---
    
    async def recompress_if_needed(self, 
                                  memory_id: str, 
                                  current_raw_nodes: List[Dict[str, Any]], 
                                  element_ids: List[str],
                                  compression_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Recompress a memory if it has been invalidated.
        
        Args:
            memory_id: Memory ID to check for recompression
            current_raw_nodes: Current raw VEIL nodes for the elements
            element_ids: Element IDs associated with this memory
            compression_context: Optional compression context
            
        Returns:
            New memorized VEIL node if recompressed, None if no recompression needed
        """
        if not self.is_memory_invalidated(memory_id):
            return None
        
        logger.info(f"Recompressing invalidated memory {memory_id}")
        
        # Delete old memory
        await self.delete_memory(memory_id)
        
        # Create new memory
        new_memorized_node = await self.compress(
            raw_veil_nodes=current_raw_nodes,
            element_ids=element_ids,
            compression_context=compression_context
        )
        
        # Store new memory
        new_memory_id = new_memorized_node["properties"]["memory_id"]
        await self.store_memory(new_memory_id, new_memorized_node)
        
        # Update correlations
        self.remove_correlation(element_ids)
        self.add_correlation(element_ids, new_memory_id)
        
        # Clear invalidation
        self.clear_invalidation(memory_id)
        
        logger.info(f"Recompressed memory {memory_id} -> {new_memory_id}")
        return new_memorized_node
    
    # --- Utility Methods ---
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about this memory compressor."""
        return {
            "compressor_type": self.__class__.__name__,
            "compressor_id": self.compressor_id,
            "token_limit": self.token_limit,
            "storage_path": self.storage_path,
            "total_correlations": len(self._correlations),
            "invalidated_memories": len(self._invalidated_memories),
            "memory_metadata_cached": len(self._memory_metadata)
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources used by this compressor."""
        logger.info(f"Cleaning up MemoryCompressor {self.compressor_id}")
        # Subclasses can override for specific cleanup
        pass
    
    # --- Protected Helper Methods ---
    
    def _generate_memory_id(self, element_ids: List[str]) -> str:
        """Generate unique memory ID for element combination."""
        import hashlib
        import time
        
        # Create deterministic but unique ID
        elements_str = "_".join(sorted(element_ids))
        timestamp = str(int(time.time() * 1000))  # millisecond precision
        
        # Hash for brevity but include timestamp for uniqueness
        hash_obj = hashlib.md5(f"{elements_str}_{timestamp}".encode())
        return f"mem_{hash_obj.hexdigest()[:12]}"
    
    def _create_base_memorized_node(self, 
                                   memory_id: str, 
                                   element_ids: List[str], 
                                   memory_summary: str,
                                   original_node_count: int,
                                   token_count: int,
                                   compression_metadata: Optional[Dict[str, Any]] = None,
                                   content_timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Create base structure for memorized VEIL node."""
        
        # Use content timestamp if provided, otherwise use compression time
        if content_timestamp is not None:
            timestamp_iso = datetime.fromtimestamp(content_timestamp).isoformat() + "Z"
            actual_compression_timestamp = datetime.now().isoformat()
        else:
            timestamp_iso = datetime.now().isoformat()
            actual_compression_timestamp = timestamp_iso
        
        properties = {
            "structural_role": "compressed_memory",
            "content_nature": "memorized_content",
            "memory_id": memory_id,
            "memory_summary": memory_summary,
            "original_element_ids": element_ids.copy(),
            "original_node_count": original_node_count,
            "compression_timestamp": timestamp_iso,  # Use content timestamp for chronological placement
            "token_count": token_count,
            "compressor_type": self.__class__.__name__,
            "compressor_version": getattr(self, 'VERSION', '1.0.0'),
            "compression_metadata": compression_metadata or {}
        }
        
        # Add temporal metadata when using content timestamp
        if content_timestamp is not None:
            properties["timestamp"] = content_timestamp
            properties["timestamp_iso"] = timestamp_iso
            properties["compression_metadata"]["actual_compression_timestamp"] = actual_compression_timestamp
            properties["compression_metadata"]["uses_content_timestamp_for_placement"] = True
        else:
            properties["compression_metadata"]["uses_content_timestamp_for_placement"] = False
        
        return {
            "veil_id": f"memory_{memory_id}",
            "node_type": "memorized_content",
            "properties": properties,
            "children": []
        }

# --- Token Counting Utilities ---

def estimate_veil_tokens(veil_nodes: List[Dict[str, Any]]) -> int:
    """
    Estimate token count for VEIL nodes using tiktoken.
    
    Args:
        veil_nodes: List of VEIL node dictionaries
        
    Returns:
        Estimated token count
    """
    try:
        import tiktoken
        
        # Use GPT-4 encoding as default
        encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Extract text content from VEIL nodes
        text_content = []
        
        for node in veil_nodes:
            props = node.get("properties", {})
            
            # Extract various text fields
            if "text_content" in props:
                text_content.append(str(props["text_content"]))
            
            if "content_nature" in props and props["content_nature"] == "chat_message":
                # Chat message specific extraction
                sender = props.get("sender_name", "")
                message = props.get("text_content", "")
                timestamp = props.get("timestamp_iso", "")
                
                text_content.append(f"{sender}: {message} [{timestamp}]")
            
            # Add other relevant text properties
            for key in ["memory_summary", "summary", "description", "workspace_description"]:
                if key in props:
                    text_content.append(str(props[key]))
        
        # Combine all text and estimate tokens
        combined_text = "\n".join(text_content)
        token_count = len(encoding.encode(combined_text))
        
        # Add some overhead for VEIL structure
        overhead_factor = 1.2
        return int(token_count * overhead_factor)
        
    except ImportError:
        logger.warning("tiktoken not available, using rough estimation")
        # Fallback: rough estimation (4 chars per token)
        total_chars = sum(len(str(node)) for node in veil_nodes)
        return total_chars // 4
    except Exception as e:
        logger.error(f"Error estimating tokens: {e}")
        # Very rough fallback
        return len(veil_nodes) * 100 