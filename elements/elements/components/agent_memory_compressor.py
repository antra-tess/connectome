"""
AgentMemoryCompressor - Concrete Implementation

A concrete implementation of MemoryCompressor designed for agent memory management.
Features agent self-memorization via LLM reflection, file-based storage, and intelligent summarization.

ENHANCED: Now supports asynchronous background compression to prevent main thread blocking.
"""

import asyncio
import logging
import json
import os
import aiofiles
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from opentelemetry import trace
from host.observability import get_tracer

from .memory_compressor_interface import MemoryCompressor, estimate_veil_tokens

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

class AgentMemoryCompressor(MemoryCompressor):
    """
    Agent-focused memory compressor with LLM-based self-memorization.
    
    Features:
    - Agent self-reflection via LLM calls
    - Agent decides what to remember from its perspective
    - File-based persistent storage with agent scoping
    - Intelligent, adaptive memory formation
    - NEW: Asynchronous background compression to prevent main thread blocking
    - NEW: Intelligent fallbacks when compression is still in progress
    """
    
    def __init__(self, agent_id: str, token_limit: int = 4000, storage_base_path: str = "storage_data/memory_storage", llm_provider=None):
        # Agent-scoped storage path
        agent_storage_path = os.path.join(storage_base_path, "agents", agent_id)
        
        # Generate unique compressor ID using agent_id
        compressor_id = f"AgentMemoryCompressor_{agent_id}"
        
        super().__init__(
            token_limit=token_limit,
            storage_path=agent_storage_path, 
            compressor_id=compressor_id
        )
        
        self.agent_id = agent_id
        self.llm_provider = llm_provider
        
        # Ensure storage directory exists
        os.makedirs(agent_storage_path, exist_ok=True)
        
        # NEW: Background compression tracking
        self._compression_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"MemComp_{agent_id}")
        self._active_compressions: Set[str] = set()  # Memory IDs currently being compressed
        self._compression_futures: Dict[str, asyncio.Future] = {}  # Future objects for tracking
        self._compression_lock = threading.Lock()  # Thread-safe access to tracking structures
        
        logger.info(f"AgentMemoryCompressor initialized for agent {agent_id} with background compression")
    
    def set_llm_provider(self, llm_provider):
        """Set the LLM provider for agent reflection."""
        self.llm_provider = llm_provider
        logger.debug(f"LLM provider set for AgentMemoryCompressor {self.agent_id}")
    
    async def compress_nodes(self, 
                           raw_veil_nodes: List[Dict[str, Any]], 
                           element_ids: List[str],
                           compression_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compress VEIL nodes using agent self-memorization.
        
        NEW: This now supports both synchronous and asynchronous compression modes.
        If immediate compression is needed, it runs synchronously.
        If background compression is preferred, it starts the process and returns a placeholder.
        """
        try:
            # Generate memory ID first to check for existing memories
            memory_id = self._generate_memory_id(element_ids)
            
            # Check if we already have a memory for these elements
            existing_memory = await self.load_memory(memory_id)
            if existing_memory and existing_memory.get("memorized_node"):
                logger.debug(f"Using existing memory {memory_id} for elements {element_ids}")
                return existing_memory["memorized_node"]
            
            # Check if compression is already in progress
            with self._compression_lock:
                if memory_id in self._active_compressions:
                    logger.debug(f"Compression already in progress for {memory_id}, returning placeholder")
                    return self._create_compression_placeholder(memory_id, element_ids, raw_veil_nodes)
            
            # Determine compression mode based on context
            use_background = compression_context.get("use_background_compression", True) if compression_context else True
            is_urgent = compression_context.get("is_urgent", False) if compression_context else False
            
            if use_background and not is_urgent:
                # Start background compression
                return await self._start_background_compression(memory_id, raw_veil_nodes, element_ids, compression_context)
            else:
                # Run synchronous compression
                logger.info(f"Running synchronous compression for {memory_id} (urgent={is_urgent})")
                return await self._compress_synchronously(raw_veil_nodes, element_ids, compression_context)
                
        except Exception as e:
            logger.error(f"Error in agent memory compression: {e}", exc_info=True)
            # Fallback to basic compression
            return await self._fallback_compression(raw_veil_nodes, element_ids, compression_context)
    
    async def _start_background_compression(self, 
                                          memory_id: str,
                                          raw_veil_nodes: List[Dict[str, Any]], 
                                          element_ids: List[str],
                                          compression_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        NEW: Start background compression and return a placeholder immediately.
        
        This prevents main thread blocking during compression while ensuring
        the rendering system has something to work with.
        """
        try:
            # Mark compression as active
            with self._compression_lock:
                self._active_compressions.add(memory_id)
            
            # Create placeholder for immediate use
            placeholder = self._create_compression_placeholder(memory_id, element_ids, raw_veil_nodes)
            
            # Start background compression task
            compression_task = asyncio.create_task(
                self._compress_in_background(memory_id, raw_veil_nodes, element_ids, compression_context)
            )
            
            # Track the future
            with self._compression_lock:
                self._compression_futures[memory_id] = compression_task
            
            logger.info(f"Started background compression for {memory_id}, returning placeholder")
            return placeholder
            
        except Exception as e:
            # Clean up tracking on error
            with self._compression_lock:
                self._active_compressions.discard(memory_id)
                self._compression_futures.pop(memory_id, None)
            logger.error(f"Error starting background compression: {e}", exc_info=True)
            # Fallback to synchronous
            return await self._compress_synchronously(raw_veil_nodes, element_ids, compression_context)
    
    async def _compress_in_background(self, 
                                    memory_id: str,
                                    raw_veil_nodes: List[Dict[str, Any]], 
                                    element_ids: List[str],
                                    compression_context: Optional[Dict[str, Any]]) -> None:
        """
        NEW: Execute compression in a background thread to avoid blocking main thread.
        
        This runs the actual LLM-based compression asynchronously and stores the result.
        """
        with tracer.start_as_current_span("agent_memory.compress_background", attributes={
            "memory.id": memory_id,
            "memory.element_ids": ",".join(element_ids),
            "agent.id": self.agent_id
        }) as span:
            try:
                start_time = time.time()
                logger.info(f"Background compression started for {memory_id}")
                span.add_event("Background compression started", attributes={
                    "compression.context": json.dumps(compression_context, default=str)
                })

                # Run compression in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                memorized_node = await loop.run_in_executor(
                    self._compression_executor,
                    self._run_compression_in_thread,
                    raw_veil_nodes, element_ids, compression_context
                )
                
                if memorized_node:
                    # Store the completed memory
                    memory_data = {
                        "memory_summary": memorized_node["properties"]["memory_summary"],
                        "metadata": memorized_node["properties"].get("compression_metadata", {}),
                        "memorized_node": memorized_node
                    }
                    
                    await self._store_memory_to_file(memory_id, memory_data)
                    
                    # Update correlations
                    self.add_correlation(element_ids, memory_id)
                    
                    elapsed = time.time() - start_time
                    logger.info(f"Background compression completed for {memory_id} in {elapsed:.1f}s")
                    span.set_attribute("compression.duration_sec", elapsed)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                else:
                    logger.warning(f"Background compression failed for {memory_id}")
                    span.set_status(trace.Status(trace.StatusCode.ERROR, "Compression returned no node"))
                
            except Exception as e:
                logger.error(f"Error in background compression for {memory_id}: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Background compression failed"))
            finally:
                # Clean up tracking
                with self._compression_lock:
                    self._active_compressions.discard(memory_id)
                    self._compression_futures.pop(memory_id, None)
    
    def _run_compression_in_thread(self, 
                                 raw_veil_nodes: List[Dict[str, Any]], 
                                 element_ids: List[str],
                                 compression_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        NEW: Thread-safe wrapper for running compression synchronously in background thread.
        
        This isolates the LLM calls and heavy processing from the main async event loop.
        """
        try:
            # Run synchronous version of compression logic
            return asyncio.run(self._compress_synchronously(raw_veil_nodes, element_ids, compression_context))
        except Exception as e:
            logger.error(f"Error in thread compression: {e}", exc_info=True)
            return None
    
    async def _compress_synchronously(self, 
                                    raw_veil_nodes: List[Dict[str, Any]], 
                                    element_ids: List[str],
                                    compression_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        NEW: Synchronous compression method (extracted from original compress_nodes).
        
        This contains the original compression logic for cases where immediate results are needed.
        """
        try:
            if not self.llm_provider:
                logger.warning(f"No LLM provider available for agent {self.agent_id}, falling back to basic analysis")
                return await self._fallback_compression(raw_veil_nodes, element_ids, compression_context)
            
            logger.info(f"AgentMemoryCompressor compressing {len(raw_veil_nodes)} nodes for elements {element_ids}")
            
            # OPTIMIZED: Single-pass parsing of VEIL content and attachments
            text_content, attachments = await self._parse_veil_content_and_attachments(raw_veil_nodes)
            
            # Get agent's self-reflection on the experience
            agent_memory_summary = await self._agent_reflect_on_experience(
                text_content,
                attachments,
                element_ids, 
                compression_context
            )
            
            # Count tokens in original content
            total_tokens = estimate_veil_tokens(raw_veil_nodes)
            
            # Create memory metadata
            memory_metadata = {
                "agent_id": self.agent_id,
                "compression_approach": "agent_self_reflection",
                "original_node_count": len(raw_veil_nodes),
                "original_element_ids": element_ids,
                "token_count": total_tokens,
                "compression_timestamp": datetime.now().isoformat(),
                "compressor_type": self.__class__.__name__,
                "compression_context": compression_context or {},
                "content_fingerprint": compression_context.get("content_fingerprint") if compression_context else None,
                "has_multimodal_content": len(attachments) > 0,
                "attachment_count": len(attachments)
            }
            
            # Generate memory ID
            memory_id = self._generate_memory_id(element_ids)
            
            # Create memorized VEIL node
            memorized_node = {
                "veil_id": f"memorized_{memory_id}",
                "node_type": "memorized_content",
                "properties": {
                    "structural_role": "compressed_content",
                    "content_nature": "agent_memory",
                    "memory_id": memory_id,
                    "memory_summary": agent_memory_summary,
                    "original_element_ids": element_ids,
                    "original_node_count": len(raw_veil_nodes),
                    "token_count": total_tokens,
                    "compression_timestamp": datetime.now().isoformat(),
                    "compressor_type": self.__class__.__name__,
                    "compression_metadata": memory_metadata
                },
                "children": []
            }
            
            # Store memory persistently
            await self._store_memory_to_file(memory_id, {
                "memory_summary": agent_memory_summary,
                "metadata": memory_metadata,
                "memorized_node": memorized_node
            })
            
            # Update correlations
            self.add_correlation(element_ids, memory_id)
            
            # NEW: Better logging with multimodal context
            multimodal_info = f" with {len(attachments)} attachments" if attachments else ""
            logger.info(f"Created agent memory {memory_id}: {agent_memory_summary[:50]}... ({total_tokens} tokens{multimodal_info})")
            
            return memorized_node
            
        except Exception as e:
            logger.error(f"Error in synchronous agent memory compression: {e}", exc_info=True)
            # Fallback to basic compression
            return await self._fallback_compression(raw_veil_nodes, element_ids, compression_context)
    
    def _create_compression_placeholder(self, 
                                      memory_id: str, 
                                      element_ids: List[str], 
                                      raw_veil_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        NEW: Create a placeholder memory node while compression is in progress.
        
        This provides a fallback representation that can be used immediately
        while the actual compression runs in the background.
        """
        try:
            # Count tokens for placeholder metadata
            total_tokens = estimate_veil_tokens(raw_veil_nodes)
            
            # Create intelligent fallback summary
            placeholder_summary = self._create_quick_fallback_summary(raw_veil_nodes, element_ids)
            
            # Create placeholder node
            placeholder_node = {
                "veil_id": f"placeholder_{memory_id}",
                "node_type": "compression_placeholder",
                "properties": {
                    "structural_role": "compressed_content",
                    "content_nature": "compression_placeholder",
                    "memory_id": memory_id,
                    "memory_summary": f"⏳ {placeholder_summary}",
                    "original_element_ids": element_ids,
                    "original_node_count": len(raw_veil_nodes),
                    "token_count": total_tokens,
                    "compression_timestamp": datetime.now().isoformat(),
                    "compressor_type": f"{self.__class__.__name__}_placeholder",
                    "is_placeholder": True,
                    "compression_status": "in_progress"
                },
                "children": []
            }
            
            logger.debug(f"Created compression placeholder for {memory_id}: {placeholder_summary}")
            return placeholder_node
            
        except Exception as e:
            logger.error(f"Error creating compression placeholder: {e}", exc_info=True)
            # Ultra-simple fallback
            return {
                "veil_id": f"error_placeholder_{memory_id}",
                "node_type": "compression_placeholder",
                "properties": {
                    "memory_summary": "⏳ Processing conversation memory...",
                    "is_placeholder": True,
                    "compression_status": "in_progress"
                },
                "children": []
            }
    
    def _create_quick_fallback_summary(self, raw_veil_nodes: List[Dict[str, Any]], element_ids: List[str]) -> str:
        """
        NEW: Create a fast, lightweight summary for placeholder use.
        
        This provides immediate context without expensive LLM calls.
        """
        try:
            message_count = 0
            participants = set()
            has_attachments = False
            
            for node in raw_veil_nodes:
                props = node.get("properties", {})
                if props.get("content_nature") == "chat_message":
                    message_count += 1
                    sender = props.get("sender_name")
                    if sender:
                        participants.add(sender)
                    
                    # Check for attachments
                    if props.get("attachment_metadata"):
                        has_attachments = True
                
                # Also check for attachment content nodes
                if node.get("node_type") == "attachment_content_item":
                    has_attachments = True
            
            # Build quick summary
            if message_count > 0:
                if len(participants) <= 2:
                    summary = f"Conversation with {', '.join(list(participants)[:2])}: {message_count} messages"
                else:
                    summary = f"Group conversation ({len(participants)} participants): {message_count} messages"
                
                if has_attachments:
                    summary += " (with attachments)"
            else:
                summary = f"Content from {', '.join(element_ids)}"
                if has_attachments:
                    summary += " with attachments"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating quick fallback summary: {e}", exc_info=True)
            return f"Content from {', '.join(element_ids)}"
    
    async def check_compression_status(self, memory_id: str) -> Dict[str, Any]:
        """
        NEW: Check the status of a background compression.
        
        Returns status information including whether compression is complete,
        in progress, or failed.
        """
        try:
            # Check if memory file exists (compression complete)
            existing_memory = await self.load_memory(memory_id)
            if existing_memory:
                return {
                    "status": "complete",
                    "memory_available": True,
                    "memory_id": memory_id
                }
            
            # Check if compression is in progress
            with self._compression_lock:
                if memory_id in self._active_compressions:
                    future = self._compression_futures.get(memory_id)
                    if future and not future.done():
                        return {
                            "status": "in_progress",
                            "memory_available": False,
                            "memory_id": memory_id
                        }
                    elif future and future.done():
                        # Compression finished, check result
                        try:
                            future.result()  # This will raise if there was an exception
                            return {
                                "status": "complete",
                                "memory_available": True,
                                "memory_id": memory_id
                            }
                        except Exception as e:
                            logger.error(f"Background compression failed for {memory_id}: {e}")
                            return {
                                "status": "failed",
                                "memory_available": False,
                                "memory_id": memory_id,
                                "error": str(e)
                            }
            
            # No compression in progress and no memory file
            return {
                "status": "not_started",
                "memory_available": False,
                "memory_id": memory_id
            }
            
        except Exception as e:
            logger.error(f"Error checking compression status for {memory_id}: {e}", exc_info=True)
            return {
                "status": "error",
                "memory_available": False,
                "memory_id": memory_id,
                "error": str(e)
            }
    
    async def get_memory_or_fallback(self, 
                                   element_ids: List[str], 
                                   raw_veil_nodes: Optional[List[Dict[str, Any]]] = None,
                                   token_limit: int = 4000) -> Dict[str, Any]:
        """
        NEW: Get compressed memory if available, otherwise return intelligently trimmed content.
        
        This is the key method for the new compression strategy:
        1. Check for existing memory
        2. If no memory and compression not started, start background compression
        3. Return either memory or trimmed fresh content based on availability
        
        Args:
            element_ids: List of element IDs to get memory for
            raw_veil_nodes: Fresh content to use as fallback (if provided)
            token_limit: Token limit for fresh content fallback
            
        Returns:
            Either compressed memory node or trimmed fresh content node
        """
        try:
            memory_id = self._generate_memory_id(element_ids)
            
            # First priority: Check for existing complete memory
            existing_memory = await self.load_memory(memory_id)
            if existing_memory and existing_memory.get("memorized_node"):
                logger.debug(f"Using existing memory for elements {element_ids}")
                return existing_memory["memorized_node"]
            
            # Second priority: Check compression status
            status = await self.check_compression_status(memory_id)
            
            if status["status"] == "complete" and status["memory_available"]:
                # Memory just became available
                memory = await self.load_memory(memory_id)
                if memory and memory.get("memorized_node"):
                    logger.debug(f"Using newly available memory for elements {element_ids}")
                    return memory["memorized_node"]
            
            # Third priority: Use fresh content fallback
            if raw_veil_nodes:
                logger.debug(f"Using fresh content fallback for elements {element_ids} (status: {status['status']})")
                
                # Start background compression if not already started
                if status["status"] == "not_started":
                    # Start background compression for next time
                    asyncio.create_task(self._start_background_compression(
                        memory_id, raw_veil_nodes, element_ids, 
                        {"use_background_compression": True, "is_urgent": False}
                    ))
                    logger.debug(f"Started background compression for future use: {memory_id}")
                
                # Return trimmed fresh content
                return self._create_trimmed_fresh_content(raw_veil_nodes, element_ids, token_limit)
            
            # Last resort: Create placeholder
            logger.warning(f"No memory or fresh content available for elements {element_ids}")
            return self._create_compression_placeholder(memory_id, element_ids, [])
            
        except Exception as e:
            logger.error(f"Error in get_memory_or_fallback: {e}", exc_info=True)
            # Ultra-fallback
            return {
                "veil_id": f"error_fallback_{element_ids[0] if element_ids else 'unknown'}",
                "node_type": "error_fallback",
                "properties": {
                    "memory_summary": f"Error retrieving memory for {', '.join(element_ids)}",
                    "is_error": True
                },
                "children": []
            }
    
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

    async def _parse_veil_content_and_attachments(self, raw_veil_nodes: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        """
        NEW: Single-pass parsing of VEIL nodes to extract both text content and attachments.
        
        This replaces the inefficient dual-traversal approach with a clean single-pass operation
        that extracts text content for reflection and attachment data for multimodal LLM calls.
        
        Args:
            raw_veil_nodes: List of VEIL nodes to process
            
        Returns:
            Tuple of (formatted_text_content, llm_compatible_attachments)
        """
        try:
            content_parts = []
            attachments = []
            attachment_ids_seen = set()  # Simple deduplication
            
            for node in raw_veil_nodes:
                # Extract text content for reflection
                text_part = await self._format_node_for_reflection(node)
                if text_part:
                    content_parts.append(text_part)
                
                # Extract attachments in single pass (both direct and children)
                node_attachments = await self._extract_node_attachments_single_pass(node, attachment_ids_seen)
                attachments.extend(node_attachments)
            
            text_content = "\n".join(content_parts)
            
            logger.debug(f"Single-pass extraction: {len(content_parts)} content parts, {len(attachments)} unique attachments")
            return text_content, attachments
            
        except Exception as e:
            logger.error(f"Error in single-pass VEIL parsing: {e}", exc_info=True)
            # Fallback to text-only
            return str(raw_veil_nodes), []
    
    async def _format_node_for_reflection(self, node: Dict[str, Any]) -> str:
        """
        NEW: Format a single VEIL node for agent reflection text content.
        
        Args:
            node: VEIL node to format
            
        Returns:
            Formatted text line for reflection, or empty string if not relevant
        """
        try:
            props = node.get("properties", {})
            content_nature = props.get("content_nature", "")
            
            if content_nature == "chat_message":
                # Format chat messages with attachment info
                sender = props.get("sender_name", "Unknown")
                text = props.get("text_content", "")
                timestamp = props.get("timestamp_iso", props.get("timestamp", ""))
                
                message_line = f"[{timestamp}] {sender}: {text}"
                
                # Add attachment metadata (but not content - that's handled separately)
                attachments = props.get("attachment_metadata", [])
                if attachments:
                    attachment_descriptions = []
                    for att in attachments:
                        filename = att.get("filename", "attachment")
                        content_type = att.get("attachment_type", att.get("content_type", "unknown"))
                        attachment_descriptions.append(f"{filename} ({content_type})")
                    
                    message_line += f" (attachments: {', '.join(attachment_descriptions)})"
                
                return message_line
                
            elif content_nature == "uplink_summary":
                # Include uplink information
                remote_name = props.get("remote_space_name", "Remote Space")
                return f"[Uplink] Connected to {remote_name}"
                
            elif props.get("structural_role") == "container":
                # Include container context
                element_name = props.get("element_name", "Element")
                available_tools = props.get("available_tools", [])
                
                container_line = f"[Context] {element_name}"
                if available_tools:
                    container_line += f" (tools available: {', '.join(available_tools)})"
                return container_line
            
            elif props.get("text_content"):
                # Generic text content
                return f"[Content] {props['text_content']}"
            
            # Not a text-relevant node
            return ""
            
        except Exception as e:
            logger.error(f"Error formatting node for reflection: {e}", exc_info=True)
            return ""
    
    async def _extract_node_attachments_single_pass(self, node: Dict[str, Any], seen_ids: set) -> List[Dict[str, Any]]:
        """
        NEW: Extract all attachments from a node and its children in a single efficient pass.
        
        Args:
            node: VEIL node to extract attachments from
            seen_ids: Set of attachment IDs already processed (for deduplication)
            
        Returns:
            List of LiteLLM-compatible attachment dictionaries
        """
        try:
            attachments = []
            
            # Check if this node itself is an attachment
            if node.get("node_type") == "attachment_content_item":
                attachment = await self._process_attachment_for_llm(node)
                if attachment:
                    attachment_id = node.get("properties", {}).get("attachment_id")
                    if attachment_id and attachment_id not in seen_ids:
                        seen_ids.add(attachment_id)
                        attachments.append(attachment)
                        logger.debug(f"Found direct attachment: {node.get('properties', {}).get('filename', 'unknown')}")
            
            # Recursively check children
            children = node.get("children", [])
            for child in children:
                child_attachments = await self._extract_node_attachments_single_pass(child, seen_ids)
                attachments.extend(child_attachments)
            
            return attachments
            
        except Exception as e:
            logger.error(f"Error extracting node attachments: {e}", exc_info=True)
            return []
    
    async def _agent_reflect_on_experience(self, 
                                         text_content: str,
                                         attachments: List[Dict[str, Any]],
                                         element_ids: List[str],
                                         compression_context: Optional[Dict[str, Any]]) -> str:
        """
        Have the agent reflect on its experience and create its own memory summary.
        
        ENHANCED: Now handles multimodal content - agent is aware of images and documents.
        NEW: Uses existing memory context for better continuity in memory formation.
        PHASE 3: Enhanced with full VEIL context for contextually aware memories.
        This is the core of agent self-memorization - the agent decides what to remember.
        """
        try:
            # Prepare context information
            element_context = ", ".join(element_ids) if len(element_ids) > 1 else element_ids[0]
            
            # Focus context if available
            focus_info = ""
            if compression_context and compression_context.get("focus_element_id"):
                focus_info = f"\nNote: You were focusing on {compression_context['focus_element_id']} during this interaction."
            
            # Compression reason
            compression_reason = compression_context.get("compression_reason", "memory management") if compression_context else "memory management"
            
            # NEW: Extract existing memory context for continuity
            existing_memories = compression_context.get("existing_memory_context", []) if compression_context else []
            memory_context_section = ""
            
            if existing_memories:
                memory_context_section = f"""
EXISTING MEMORIES FOR CONTEXT:
Here are your previous memories about this conversation/element to help you understand the ongoing context:
{chr(10).join([f"- {memory}" for memory in existing_memories])}

"""
                logger.debug(f"Agent {self.agent_id} reflecting with {len(existing_memories)} existing memory context")

            # PHASE 3: NEW - Get full VEIL context from HUD for enhanced contextual awareness
            veil_context_section = ""
            full_veil_context = compression_context.get("full_veil_context") if compression_context else None
            
            if full_veil_context:
                veil_context_section = f"""
FULL CONTEXT FOR MEMORIZATION:
Here's your complete context to understand what's happening around this conversation:
<veil>
{full_veil_context}
</veil>

"""
                logger.debug(f"Agent {self.agent_id} reflecting with full VEIL context ({len(full_veil_context)} chars)")

            # NEW: Enhanced reflection prompt for multimodal content with VEIL context
            if attachments:
                reflection_prompt = f"""You are an AI agent reflecting on your recent experience that included visual and document content. You have access to your complete conversation context to understand what you should remember.

{memory_context_section}{veil_context_section}SPECIFIC CONTENT TO MEMORIZE:
{text_content}

CONTEXT:
- This experience occurred in: {element_context}
- Compression reason: {compression_reason}{focus_info}

MULTIMODAL REFLECTION TASK:
Create a concise memory summary that captures what was important about this SPECIFIC CONTENT from YOUR perspective as an AI agent. Since this interaction included attachments (images, documents, files), pay special attention to:

CONTENT ANALYSIS:
- What happened in this specific content that you should remember
- How this content relates to the broader conversation context shown above
- What images/documents were shared and their relevance
- How the visual/document content related to the conversation
- Any important outcomes or decisions in this specific content
- Context that would be useful for future interactions

MULTIMODAL CONTEXT:
- Describe what the attachments represented (diagrams, documents, images, etc.)
- Explain how the attachments connected to the conversation topic
- Note any important visual or document information discussed

{"MEMORY CONTINUITY:" + chr(10) + "Consider how this specific content relates to your existing memories and the broader context shown above. Build upon that context rather than repeating it." + chr(10) if existing_memories or veil_context_section else ""}
Keep the summary brief but informative - this will be your memory of this specific multimodal experience.

MEMORY SUMMARY:"""
            else:
                # Standard text-only reflection prompt with VEIL context
                reflection_prompt = f"""You are an AI agent reflecting on your recent experience. You have access to your complete conversation context to understand what you should remember.

{memory_context_section}{veil_context_section}SPECIFIC CONTENT TO MEMORIZE:
{text_content}

CONTEXT:
- This experience occurred in: {element_context}
- Compression reason: {compression_reason}{focus_info}

REFLECTION TASK:
Create a concise memory summary that captures what was important about this SPECIFIC CONTENT from YOUR perspective as an AI agent. Focus on:
- What happened in this specific content that you should remember
- How this content relates to the broader conversation context shown above
- Who was involved and their roles
- Any important outcomes or decisions in this specific content
- Context that would be useful for future interactions

{"MEMORY CONTINUITY:" + chr(10) + "Consider how this specific content relates to your existing memories and the broader context shown above. Build upon that context rather than repeating it." + chr(10) if existing_memories or veil_context_section else ""}
Keep the summary brief but informative - this will be your memory of this specific experience.

MEMORY SUMMARY:"""

            # NEW: Create multimodal LLM message if attachments detected
            from llm.provider_interface import LLMMessage
            
            if attachments:
                # Create multimodal message structure
                reflection_message = LLMMessage("user", {
                    "text": reflection_prompt,
                    "attachments": attachments
                })
                context_info = "multimodal content with full VEIL context" if veil_context_section else "multimodal content"
                logger.info(f"Agent {self.agent_id} reflecting on {context_info}: {len(attachments)} attachments")
            else:
                # Text-only reflection
                reflection_message = LLMMessage("user", reflection_prompt)
                context_info = "content with full VEIL context" if veil_context_section else "content"
            
            # Get agent's reflection via LLM
            llm_response = self.llm_provider.complete(messages=[reflection_message], tools=None)
            
            if llm_response and llm_response.content:
                agent_summary = llm_response.content.strip()
                
                # NEW: Log enhanced memory formation with VEIL context awareness
                context_enhancement = ""
                if veil_context_section:
                    context_enhancement = " (enhanced with full VEIL context)"
                if existing_memories:
                    context_enhancement += f" with {len(existing_memories)} memory context"
                
                if attachments:
                    logger.info(f"Agent {self.agent_id} created contextually aware multimodal memory{context_enhancement}: {agent_summary[:100]}...")
                else:
                    logger.debug(f"Agent {self.agent_id} reflected{context_enhancement}: {agent_summary[:100]}...")
                
                return agent_summary
            else:
                logger.warning(f"LLM reflection failed for agent {self.agent_id}, using fallback")
                return await self._create_fallback_summary(text_content, element_ids, len(attachments) > 0)
                
        except Exception as e:
            logger.error(f"Error during agent reflection: {e}", exc_info=True)
            return await self._create_fallback_summary(text_content, element_ids, False)
    
    async def _create_fallback_summary(self, text_content: str, element_ids: List[str], has_attachments: bool = False) -> str:
        """
        Create a basic fallback summary when LLM reflection fails.
        
        ENHANCED: Now aware of multimodal content.
        """
        lines = text_content.split('\n')
        message_lines = [line for line in lines if '] ' in line and ': ' in line]
        attachment_lines = [line for line in lines if 'Attachment content:' in line or 'attachments:' in line]
        
        if message_lines:
            base_summary = f"Conversation with {len(message_lines)} messages in {', '.join(element_ids)}"
            
            # NEW: Add attachment context to fallback
            if has_attachments or attachment_lines:
                base_summary += " (with attachments)"
                
            return base_summary
        else:
            return f"Content from {', '.join(element_ids)}"
    
    async def _fallback_compression(self, 
                                  raw_veil_nodes: List[Dict[str, Any]], 
                                  element_ids: List[str],
                                  compression_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback compression when LLM is not available."""
        logger.info(f"Using fallback compression for {len(raw_veil_nodes)} nodes")
        
        # OPTIMIZED: Use same single-pass approach for consistency
        text_content, attachments = await self._parse_veil_content_and_attachments(raw_veil_nodes)
        
        # Simple analysis
        message_count = 0
        participants = set()
        
        for node in raw_veil_nodes:
            props = node.get("properties", {})
            if props.get("content_nature") == "chat_message":
                message_count += 1
                sender = props.get("sender_name")
                if sender:
                    participants.add(sender)
        
        # Create basic summary with multimodal awareness
        if message_count > 0:
            if len(participants) <= 2:
                summary = f"Conversation with {', '.join(participants)}: {message_count} messages"
            else:
                summary = f"Group conversation: {message_count} messages"
            
            # Add attachment context to fallback
            if attachments:
                summary += f" (with {len(attachments)} attachments)"
        else:
            summary = f"Content from {', '.join(element_ids)}"
            if attachments:
                summary += f" with {len(attachments)} attachments"
        
        # Count tokens
        total_tokens = estimate_veil_tokens(raw_veil_nodes)
        memory_id = self._generate_memory_id(element_ids)
        
        # Create memorized node with multimodal metadata
        memorized_node = {
            "veil_id": f"memorized_{memory_id}",
            "node_type": "memorized_content",
            "properties": {
                "structural_role": "compressed_content",
                "content_nature": "agent_memory",
                "memory_id": memory_id,
                "memory_summary": summary,
                "original_element_ids": element_ids,
                "original_node_count": len(raw_veil_nodes),
                "token_count": total_tokens,
                "compression_timestamp": datetime.now().isoformat(),
                "compressor_type": f"{self.__class__.__name__}_fallback",
                "has_multimodal_content": len(attachments) > 0,  # NEW: Track multimodal in fallback too
                "attachment_count": len(attachments)
            },
            "children": []
        }
        
        # Store and track
        await self._store_memory_to_file(memory_id, {
            "memory_summary": summary,
            "metadata": {
                "fallback": True, 
                "agent_id": self.agent_id,
                "has_multimodal_content": len(attachments) > 0,
                "attachment_count": len(attachments)
            },
            "memorized_node": memorized_node
        })
        
        self.add_correlation(element_ids, memory_id)
        
        multimodal_info = f" with {len(attachments)} attachments" if attachments else ""
        logger.info(f"Created fallback memory {memory_id}: {summary}{multimodal_info}")
        return memorized_node
    
    # Storage methods remain the same
    async def _store_memory_to_file(self, memory_id: str, memory_data: Dict[str, Any]) -> bool:
        """Store memory data to agent-scoped file."""
        try:
            file_path = os.path.join(self.storage_path, f"{memory_id}.json")
            
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(memory_data, indent=2))
            
            logger.debug(f"Stored memory {memory_id} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing memory to file: {e}", exc_info=True)
            return False
    
    async def load_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Load memory data from agent-scoped file."""
        try:
            file_path = os.path.join(self.storage_path, f"{memory_id}.json")
            
            if not os.path.exists(file_path):
                return None
            
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                return json.loads(content)
                
        except Exception as e:
            logger.error(f"Error loading memory from file: {e}", exc_info=True)
            return None
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete memory from agent-scoped file storage."""
        try:
            file_path = os.path.join(self.storage_path, f"{memory_id}.json")
            
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Deleted memory file {file_path}")
            
            # Remove from correlations - find element_ids for this memory_id
            for element_set, corr_memory_id in list(self._correlations.items()):
                if corr_memory_id == memory_id:
                    self.remove_correlation(list(element_set))
                    break
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory file: {e}", exc_info=True)
            return False
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about agent memory storage."""
        try:
            memory_files = [f for f in os.listdir(self.storage_path) if f.endswith('.json')]
            total_size = sum(os.path.getsize(os.path.join(self.storage_path, f)) for f in memory_files)
            
            # Get base statistics and add agent-specific info
            base_stats = super().get_memory_stats()
            
            return {
                "agent_id": self.agent_id,
                "total_memories": len(memory_files),
                "storage_size_mb": total_size / (1024 * 1024),
                "storage_path": self.storage_path,
                "correlations_tracked": len(self._correlations),
                **base_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}", exc_info=True)
            return {"error": str(e)}
    
    def cleanup(self):
        """Clean up background compression resources."""
        try:
            logger.info(f"Cleaning up AgentMemoryCompressor for agent {self.agent_id}")
            
            # Stop accepting new tasks
            if self._compression_executor and not self._compression_executor._shutdown:
                logger.debug("Shutting down compression executor")
                # FIXED: Remove timeout parameter for Python version compatibility
                self._compression_executor.shutdown(wait=True)
                self._compression_executor = None
            
            # Cancel any pending futures (they should complete or be cancelled)
            with self._compression_lock:
                for memory_id, future in list(self._compression_futures.items()):
                    if not future.done():
                        logger.debug(f"Cancelling pending compression for {memory_id}")
                        future.cancel()
                
                self._compression_futures.clear()
                self._active_compressions.clear()
            
            logger.debug("AgentMemoryCompressor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during background compression cleanup: {e}", exc_info=True)
        
        # Call parent cleanup
        try:
            # FIXED: Parent cleanup should be called synchronously
            import asyncio
            if hasattr(super(), 'cleanup'):
                parent_cleanup = super().cleanup()
                # Only await if it's a coroutine
                if asyncio.iscoroutine(parent_cleanup):
                    # Can't await in sync context, so just log a warning
                    logger.warning("Parent cleanup is async but called from sync context")
                    # The coroutine will be garbage collected with a warning
                else:
                    # Parent cleanup is sync, call it normally
                    pass  # Already called above
        except Exception as e:
            logger.error(f"Error during parent cleanup: {e}", exc_info=True)

    async def compress(self, 
                      raw_veil_nodes: List[Dict[str, Any]], 
                      element_ids: List[str],
                      compression_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compress VEIL nodes using agent self-memorization.
        
        The agent reflects on the experience and creates its own memory summary.
        """
        return await self.compress_nodes(raw_veil_nodes, element_ids, compression_context)
    
    async def estimate_tokens(self, raw_veil_nodes: List[Dict[str, Any]]) -> int:
        """Estimate tokens using the interface utility."""
        return estimate_veil_tokens(raw_veil_nodes)
    
    async def store_memory(self, memory_id: str, memorized_veil_node: Dict[str, Any]) -> bool:
        """Store memory to agent-scoped file (interface compliance)."""
        try:
            memory_data = {
                "memory_summary": memorized_veil_node["properties"]["memory_summary"],
                "metadata": memorized_veil_node["properties"].get("compression_metadata", {}),
                "memorized_node": memorized_veil_node
            }
            return await self._store_memory_to_file(memory_id, memory_data)
        except Exception as e:
            logger.error(f"Error in store_memory interface: {e}", exc_info=True)
            return False

    async def _process_attachment_for_llm(self, attachment_node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        NEW: Process an attachment VEIL node into LiteLLM-compatible format for agent analysis.
        
        This converts VEIL attachment nodes into the format needed for multimodal LLM calls
        so the agent can actually see images and analyze document content.
        
        Args:
            attachment_node: VEIL node representing attachment content
            
        Returns:
            LiteLLM-compatible content part or None if not processable
        """
        try:
            props = attachment_node.get("properties", {})
            content_nature = props.get("content_nature", "unknown")
            filename = props.get("filename", "unknown_file")
            attachment_id = props.get("attachment_id")
            
            if "image" in content_nature.lower():
                # For images, try to get base64 content
                logger.debug(f"Processing image attachment for agent: {filename}")
                
                content = await self._get_attachment_content_for_agent(attachment_id, attachment_node)
                if content and isinstance(content, str):
                    # Format for LiteLLM multimodal
                    return {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{content_nature.split('/')[-1] if '/' in content_nature else 'png'};base64,{content}"
                        }
                    }
                else:
                    logger.warning(f"Could not extract image content for {attachment_id}")
                    return None
                    
            elif "text" in content_nature.lower() or content_nature in ["application/pdf", "text/plain"]:
                # For text/document files, include content directly  
                logger.debug(f"Processing text/document attachment for agent: {filename}")
                
                content = await self._get_attachment_content_for_agent(attachment_id, attachment_node)
                if content:
                    return {
                        "type": "text",
                        "text": f"[Document: {filename}]\n{content}"
                    }
                else:
                    logger.warning(f"Could not extract document content for {attachment_id}")
                    return None
            else:
                # For other file types, provide descriptive text
                logger.debug(f"Unsupported attachment type for agent analysis: {content_nature}")
                return {
                    "type": "text", 
                    "text": f"[Attachment: {filename} (Type: {content_nature}) - Content not directly analyzable by agent]"
                }
                
        except Exception as e:
            logger.error(f"Error processing attachment for LLM: {e}", exc_info=True)
            return None
    
    async def _get_attachment_content_for_agent(self, attachment_id: str, attachment_node: Dict[str, Any]) -> Optional[str]:
        """
        NEW: Get the actual attachment content for agent analysis.
        
        SIMPLIFIED: With VEIL's clean architecture, content is directly available in node properties.
        
        Args:
            attachment_id: ID of the attachment to retrieve
            attachment_node: VEIL attachment_content_item node with attachment data
            
        Returns:
            Attachment content as string (base64 for images, text for documents), or None
        """
        try:
            props = attachment_node.get("properties", {})
            filename = props.get("filename", "unknown_file")
            content_nature = props.get("content_nature", "unknown")
            
            # NEW: Get content directly from VEIL node properties (matches HUD approach)
            content = props.get("content")
            content_available = props.get("content_available", False)
            
            if content:
                logger.debug(f"Found direct content in VEIL node for agent analysis: {filename}")
                return content
            
            # Log when content should be available but isn't found
            if content_available:
                logger.warning(f"Content marked as available for {filename} but not found in VEIL node properties")
            else:
                logger.debug(f"Content not marked as available for {filename}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving attachment content for agent: {e}", exc_info=True)
            return None
    
    def _generate_memory_id(self, element_ids: List[str]) -> str:
        """
        Generate a deterministic memory ID for a set of element IDs.
        
        FIXED: Now generates truly deterministic IDs based on element content,
        ensuring the same elements always get the same memory ID.
        
        Args:
            element_ids: List of element IDs to generate memory for
            
        Returns:
            Deterministic memory ID string
        """
        try:
            import hashlib
            
            # Sort element IDs for deterministic ordering
            sorted_elements = sorted(element_ids)
            
            # Create deterministic content string including agent context
            content_parts = [
                self.agent_id,  # Include agent ID for agent-scoped memories
                "memory_for",
                *sorted_elements  # Include sorted element IDs
            ]
            
            # Create deterministic hash
            content_string = "_".join(content_parts)
            hash_object = hashlib.md5(content_string.encode('utf-8'))
            memory_id = f"mem_{hash_object.hexdigest()[:12]}"
            
            logger.debug(f"Generated deterministic memory ID {memory_id} for elements {sorted_elements}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error generating memory ID: {e}", exc_info=True)
            # Fallback to timestamp-based ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"mem_fallback_{timestamp}" 