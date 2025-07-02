"""
AgentMemoryCompressor - Concrete Implementation

A concrete implementation of MemoryCompressor designed for agent memory management.
Features agent self-memorization via LLM reflection, file-based storage, and intelligent summarization.

REFACTORED: Now uses pure async/await architecture for reliable background memory formation.
"""

import asyncio
import logging
import json
import os
import aiofiles
import time
import random
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from pathlib import Path

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
    - NEW: Pure async/await architecture for reliable background memory formation
    - NEW: Memory state tracking (existing, forming, needs-formation)
    - NEW: Non-blocking fallbacks to prevent data loss
    - NEW: Sequential memory formation per agent via async queue
    
    PURE ASYNC ARCHITECTURE:
    The compressor now uses pure async/await throughout with:
    - Async queue for sequential memory formation
    - Memory state management (existing/forming/needs-formation)
    - Background task processing with proper cleanup
    - Rate limiting and retry logic in pure async
    - No threading - eliminates GIL and event loop conflicts
    
    USAGE EXAMPLES:
    
    # Basic usage with default settings
    compressor = AgentMemoryCompressor("agent_123")
    compressor.set_llm_provider(your_llm_provider)
    
    # Configure retry behavior for different LLM providers
    compressor.configure_retry_behavior(
        max_attempts=5,           # More retries for unreliable providers
        base_delay=2.0,          # Longer initial delay
        max_delay=60.0,          # Higher max delay
        min_call_interval=1.0    # Slower calls
    )
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
        
        # NEW: Pure async infrastructure
        self._memory_formation_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._memory_states: Dict[str, str] = {}  # memory_id -> state
        self._active_formations: Dict[str, asyncio.Task] = {}  # memory_id -> task
        self._formation_semaphore = asyncio.Semaphore(1)  # Sequential processing per agent
        self._memory_cache: Dict[str, Dict[str, Any]] = {}  # memory_id -> memory_data
        self._background_processor_task: Optional[asyncio.Task] = None
        self._processor_running: bool = False
        
        # LLM Rate limiting and retry configuration (async version)
        self._max_retry_attempts: int = 4
        self._base_retry_delay: float = 1.0     # Start with 1 second
        self._max_retry_delay: float = 30.0     # Cap at 30 seconds
        self._backoff_multiplier: float = 2.0   # Exponential backoff
        self._jitter_factor: float = 0.1        # 10% randomization
        self._min_call_interval: float = 0.5    # 500ms between LLM calls
        self._last_llm_call_time: float = 0.0
        
        logger.info(f"AgentMemoryCompressor initialized for agent {agent_id} with pure async architecture")
        
        # Schedule background processor startup when event loop is available
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._start_background_processor())
        except RuntimeError:
            # No event loop running yet, will be started when needed
            logger.debug(f"Agent {agent_id}: No event loop running, background processor will start later")

    async def ensure_background_processor_running(self):
        """Ensure background processor is running - call this if needed."""
        if not self._background_processor_task or self._background_processor_task.done():
            await self._start_background_processor()
    
    def set_llm_provider(self, llm_provider):
        """Set the LLM provider for agent reflection."""
        self.llm_provider = llm_provider
        logger.debug(f"LLM provider set for AgentMemoryCompressor {self.agent_id}")
    
    def configure_retry_behavior(self, 
                                max_attempts: int = None,
                                base_delay: float = None,
                                max_delay: float = None,
                                backoff_multiplier: float = None,
                                jitter_factor: float = None,
                                min_call_interval: float = None):
        """
        Configure retry behavior for LLM calls with pure async processing.
        
        Args:
            max_attempts: Maximum retry attempts (default: 4)
            base_delay: Base delay in seconds (default: 1.0)
            max_delay: Maximum delay in seconds (default: 30.0)
            backoff_multiplier: Exponential backoff multiplier (default: 2.0)
            jitter_factor: Jitter randomization factor 0-1 (default: 0.1)
            min_call_interval: Minimum seconds between LLM calls (default: 0.5)
        """
        # Validate bounds
        if max_attempts is not None:
            self._max_retry_attempts = max(1, max_attempts)
            
        if base_delay is not None:
            self._base_retry_delay = max(0.1, base_delay)
            
        if max_delay is not None:
            self._max_retry_delay = max(1.0, max_delay)
            
        if backoff_multiplier is not None:
            self._backoff_multiplier = max(1.0, backoff_multiplier)
            
        if jitter_factor is not None:
            self._jitter_factor = max(0.0, min(1.0, jitter_factor))
            
        if min_call_interval is not None:
            self._min_call_interval = max(0.0, min_call_interval)
        
        logger.info(f"Agent {self.agent_id}: Updated retry config - max_attempts={self._max_retry_attempts}, "
                   f"base_delay={self._base_retry_delay}s, max_delay={self._max_retry_delay}s, "
                   f"backoff_multiplier={self._backoff_multiplier}, jitter_factor={self._jitter_factor}, "
                   f"min_call_interval={self._min_call_interval}s")

    async def get_memory_state(self, memory_id: str) -> str:
        """Get current memory state: 'existing', 'forming', 'needs_formation'."""
        if memory_id in self._memory_cache:
            return "existing"
        elif memory_id in self._active_formations:
            return "forming"
        else:
            # Check if file exists
            if await self._memory_file_exists(memory_id):
                return "existing"
            else:
                return "needs_formation"

    def _set_memory_state(self, memory_id: str, state: str):
        """Update memory state tracking."""
        self._memory_states[memory_id] = state
        logger.debug(f"Agent {self.agent_id}: Memory {memory_id} state → {state}")

    async def _memory_file_exists(self, memory_id: str) -> bool:
        """Check if memory file exists on disk."""
        try:
            memory_file = Path(self.storage_path) / f"{memory_id}.json"
            return memory_file.exists()
        except Exception:
            return False

    async def _start_background_processor(self):
        """Start the background memory formation processor."""
        if self._background_processor_task is not None:
            logger.warning(f"Agent {self.agent_id}: Background processor already running")
            return
        
        self._processor_running = True
        self._background_processor_task = asyncio.create_task(
            self._memory_formation_processor()
        )
        logger.info(f"Agent {self.agent_id}: Started background memory formation processor")

    async def _memory_formation_processor(self):
        """Process memory formation requests sequentially."""
        logger.info(f"Agent {self.agent_id}: Memory formation processor started")
        
        try:
            while self._processor_running:
                try:
                    # Wait for formation request with timeout
                    formation_request = await asyncio.wait_for(
                        self._memory_formation_queue.get(),
                        timeout=5.0
                    )
                    
                    memory_id = formation_request["memory_id"]
                    
                    # Process with semaphore to ensure sequential execution
                    async with self._formation_semaphore:
                        await self._process_memory_formation(formation_request)
                    
                    # Mark queue task as done
                    self._memory_formation_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue  # Normal timeout, check for shutdown
                except Exception as e:
                    logger.error(f"Agent {self.agent_id}: Error in memory formation processor: {e}", exc_info=True)
                    await asyncio.sleep(1.0)  # Prevent rapid error loops
                    
        except asyncio.CancelledError:
            logger.info(f"Agent {self.agent_id}: Memory formation processor cancelled")
        finally:
            logger.info(f"Agent {self.agent_id}: Memory formation processor stopped")
            self._processor_running = False

    async def _process_memory_formation(self, formation_request: Dict[str, Any]):
        """Process a single memory formation request."""
        try:
            memory_id = formation_request["memory_id"]
            raw_veil_nodes = formation_request["raw_veil_nodes"]
            element_ids = formation_request["element_ids"]
            compression_context = formation_request.get("compression_context", {})
            
            logger.info(f"Agent {self.agent_id}: Processing memory formation for {memory_id}")
            
            # Update state to forming
            self._set_memory_state(memory_id, "forming")
            
            # Perform actual memory formation
            memorized_node = await self._form_memory_async(
                raw_veil_nodes, element_ids, compression_context
            )
            
            if memorized_node:
                # Store memory to file and cache
                memory_data = {
                    "memory_id": memory_id,
                    "memorized_node": memorized_node,
                    "created_at": datetime.now().isoformat(),
                    "element_ids": element_ids
                }
                
                await self._store_memory_to_file(memory_id, memory_data)
                self._memory_cache[memory_id] = memory_data
                self._set_memory_state(memory_id, "existing")
                
                logger.info(f"Agent {self.agent_id}: Memory formation completed for {memory_id}")
            else:
                logger.error(f"Agent {self.agent_id}: Memory formation failed for {memory_id}")
                self._set_memory_state(memory_id, "needs_formation")  # Allow retry
                
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Error forming memory {memory_id}: {e}", exc_info=True)
            self._set_memory_state(memory_id, "needs_formation")  # Allow retry
        finally:
            # Clean up active formation tracking
            self._active_formations.pop(memory_id, None)

    async def compress_nodes(self, 
                           raw_veil_nodes: List[Dict[str, Any]], 
                           element_ids: List[str],
                           compression_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        NEW: Pure async memory formation with state-aware fallbacks.
        
        Returns:
            - Existing memory if available
            - N-chunk fallback if memory is forming or needs formation
        """
        try:
            # Ensure background processor is running
            await self.ensure_background_processor_running()
            
            # Generate memory ID
            memory_id = self._generate_memory_id(element_ids)
            
            # Check memory state
            memory_state = await self.get_memory_state(memory_id)
            
            if memory_state == "existing":
                # Load and return existing memory
                logger.debug(f"Agent {self.agent_id}: Using existing memory {memory_id}")
                if memory_id in self._memory_cache:
                    return self._memory_cache[memory_id]["memorized_node"]
                else:
                    # Load from file
                    memory_data = await self.load_memory(memory_id)
                    if memory_data and memory_data.get("memorized_node"):
                        self._memory_cache[memory_id] = memory_data
                        return memory_data["memorized_node"]
            
            elif memory_state == "forming":
                # Memory is currently being formed, return fallback
                logger.debug(f"Agent {self.agent_id}: Memory {memory_id} is forming, returning N-chunk fallback")
                return self._create_forming_fallback(memory_id, element_ids, raw_veil_nodes)
            
            elif memory_state == "needs_formation":
                # Start background formation and return fallback
                logger.debug(f"Agent {self.agent_id}: Starting background formation for {memory_id}")
                await self._queue_memory_formation(memory_id, raw_veil_nodes, element_ids, compression_context)
                return self._create_forming_fallback(memory_id, element_ids, raw_veil_nodes)
            
            # Fallback for unexpected states
            logger.warning(f"Agent {self.agent_id}: Unexpected memory state {memory_state} for {memory_id}")
            return self._create_error_fallback(element_ids, raw_veil_nodes)
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Error in memory formation: {e}", exc_info=True)
            return self._create_error_fallback(element_ids, raw_veil_nodes)

    async def _queue_memory_formation(self, memory_id: str, raw_veil_nodes: List[Dict[str, Any]], 
                                     element_ids: List[str], compression_context: Optional[Dict[str, Any]]):
        """Queue memory formation request for background processing."""
        try:
            # Mark as forming to prevent duplicate requests
            self._set_memory_state(memory_id, "forming")
            
            formation_request = {
                "memory_id": memory_id,
                "raw_veil_nodes": raw_veil_nodes,
                "element_ids": element_ids,
                "compression_context": compression_context,
                "queued_at": datetime.now().isoformat()
            }
            
            # Add to formation queue (non-blocking)
            try:
                self._memory_formation_queue.put_nowait(formation_request)
                logger.debug(f"Agent {self.agent_id}: Queued memory formation for {memory_id}")
            except asyncio.QueueFull:
                logger.warning(f"Agent {self.agent_id}: Memory formation queue full, skipping {memory_id}")
                self._set_memory_state(memory_id, "needs_formation")  # Reset for retry
                
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Error queuing memory formation: {e}", exc_info=True)
            self._set_memory_state(memory_id, "needs_formation")  # Reset for retry

    async def _form_memory_async(self, raw_veil_nodes: List[Dict[str, Any]], 
                                element_ids: List[str], 
                                compression_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Form memory using pure async LLM calls."""
        try:
            # Parse content for agent reflection
            text_content, attachments = await self._parse_veil_content_and_attachments(raw_veil_nodes)
            
            if not text_content.strip():
                logger.warning(f"Agent {self.agent_id}: No text content to memorize")
                return None
            
            # Generate memory using agent reflection
            memory_summary = await self._agent_reflect_on_experience(
                text_content, attachments, element_ids, compression_context
            )
            
            if not memory_summary:
                logger.warning(f"Agent {self.agent_id}: Agent reflection returned empty memory")
                return None
            
            # Create memorized VEIL node
            memorized_node = {
                "veil_id": f"memory_{self.agent_id}_{int(time.time())}",
                "node_type": "memorized_content",
                "properties": {
                    "structural_role": "compressed_content",
                    "content_nature": "content_memory",
                    "memory_summary": memory_summary,
                    "original_child_count": len(raw_veil_nodes),
                    "compression_timestamp": datetime.now().isoformat(),
                    "compression_approach": "agent_memory_compressor_async",
                    "is_focused": compression_context.get("is_focused", True) if compression_context else True,
                    "agent_id": self.agent_id,
                    "element_ids": element_ids
                },
                "children": []
            }
            
            return memorized_node
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Error forming memory: {e}", exc_info=True)
            return None

    def _create_forming_fallback(self, memory_id: str, element_ids: List[str], 
                                raw_veil_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create fallback node when memory is forming."""
        return {
            "veil_id": f"forming_{memory_id}",
            "node_type": "fresh_content",  # Indicates fallback
            "properties": {
                "structural_role": "compressed_content",
                "content_nature": "content_memory",
                "memory_summary": f"⏳ Memory formation in progress for {', '.join(element_ids)}",
                "original_child_count": len(raw_veil_nodes),
                "compression_approach": "forming_fallback",
                "is_forming": True,
                "memory_id": memory_id,
                "element_ids": element_ids
            },
            "children": raw_veil_nodes  # Include original content to prevent data loss
        }

    def _create_error_fallback(self, element_ids: List[str], 
                              raw_veil_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create fallback node when memory formation fails."""
        return {
            "veil_id": f"error_fallback_{int(time.time())}",
            "node_type": "fresh_content",
            "properties": {
                "structural_role": "compressed_content", 
                "content_nature": "content_memory",
                "memory_summary": f"Memory formation error for {', '.join(element_ids)}",
                "original_child_count": len(raw_veil_nodes),
                "compression_approach": "error_fallback",
                "is_error": True
            },
            "children": raw_veil_nodes  # Include original content to prevent data loss
        }

    async def _call_llm_with_retry(self, message, context_info: str = ""):
        """Pure async LLM call with retry logic and rate limiting."""
        if not self.llm_provider:
            raise Exception(f"No LLM provider available for agent {self.agent_id}")
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self._last_llm_call_time
        if time_since_last < self._min_call_interval:
            await asyncio.sleep(self._min_call_interval - time_since_last)
        
        # Retry loop with exponential backoff
        for attempt in range(self._max_retry_attempts):
            try:
                logger.debug(f"Agent {self.agent_id}: LLM call attempt {attempt + 1}/{self._max_retry_attempts} for {context_info}")
                
                # Make async LLM call
                llm_response = self.llm_provider.complete(messages=[message], tools=None)
                self._last_llm_call_time = time.time()
                
                logger.debug(f"Agent {self.agent_id}: LLM call successful for {context_info}")
                return llm_response
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if this is a retryable error
                is_retryable = any(keyword in error_msg for keyword in [
                    "overloaded", "rate limit", "timeout", "503", "502", "504", "429"
                ])
                
                if not is_retryable or attempt == self._max_retry_attempts - 1:
                    logger.error(f"Agent {self.agent_id}: LLM call failed (attempt {attempt + 1}): {e}")
                    raise e
                
                # Calculate delay with exponential backoff and jitter
                base_delay = self._base_retry_delay * (self._backoff_multiplier ** attempt)
                max_delay = min(base_delay, self._max_retry_delay)
                
                # Add jitter to prevent thundering herd
                jitter = max_delay * self._jitter_factor * random.random()
                delay = max_delay + jitter
                
                logger.warning(f"Agent {self.agent_id}: LLM call failed (attempt {attempt + 1}), retrying in {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
        
        raise Exception(f"All {self._max_retry_attempts} LLM call attempts failed")

    async def cleanup(self):
        """Clean up the async memory compressor."""
        try:
            logger.info(f"AgentMemoryCompressor cleanup started for agent {self.agent_id}")
            
            # Stop background processor
            if self._processor_running:
                logger.info(f"Agent {self.agent_id}: Stopping background memory processor")
                self._processor_running = False
                
                if self._background_processor_task:
                    self._background_processor_task.cancel()
                    try:
                        await self._background_processor_task
                    except asyncio.CancelledError:
                        pass
                    self._background_processor_task = None
            
            # Cancel active formations
            active_tasks = list(self._active_formations.values())
            for task in active_tasks:
                if not task.done():
                    task.cancel()
            
            if active_tasks:
                logger.info(f"Agent {self.agent_id}: Cancelled {len(active_tasks)} active formation tasks")
                await asyncio.gather(*active_tasks, return_exceptions=True)
            
            # Clear queues and caches
            while not self._memory_formation_queue.empty():
                try:
                    self._memory_formation_queue.get_nowait()
                    self._memory_formation_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            
            self._memory_states.clear()
            self._memory_cache.clear()
            self._active_formations.clear()
            
            logger.info(f"AgentMemoryCompressor cleanup completed for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error during AgentMemoryCompressor cleanup: {e}", exc_info=True)

    # --- Interface Compatibility Methods ---
    
    async def compress(self, 
                      raw_veil_nodes: List[Dict[str, Any]], 
                      element_ids: List[str],
                      compression_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Interface compatibility method - delegates to compress_nodes.
        
        Maintains interface compatibility while using the new async architecture.
        """
        return await self.compress_nodes(raw_veil_nodes, element_ids, compression_context)

    async def estimate_tokens(self, raw_veil_nodes: List[Dict[str, Any]]) -> int:
        """Estimate token count for raw VEIL nodes using tiktoken."""
        try:
            return estimate_veil_tokens(raw_veil_nodes)
        except Exception as e:
            logger.warning(f"Agent {self.agent_id}: Error estimating tokens with tiktoken: {e}")
            # Fallback to character-based estimation
            total_chars = 0
            for node in raw_veil_nodes:
                props = node.get("properties", {})
                text_content = props.get("text_content", "")
                total_chars += len(text_content)
            return total_chars // 4  # Rough estimate: 4 chars per token

    async def store_memory(self, memory_id: str, memorized_veil_node: Dict[str, Any]) -> bool:
        """Store a memorized VEIL node to persistent storage."""
        try:
            memory_data = {
                "memory_id": memory_id,
                "memorized_node": memorized_veil_node,
                "created_at": datetime.now().isoformat(),
                "element_ids": memorized_veil_node.get("properties", {}).get("element_ids", [])
            }
            return await self._store_memory_to_file(memory_id, memory_data)
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Error storing memory {memory_id}: {e}", exc_info=True)
            return False

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
            
            # NEW: Get agent's reflection via LLM with retry mechanism
            llm_response = await self._call_llm_with_retry(reflection_message, context_info)
            
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
                logger.warning(f"LLM reflection failed for agent {self.agent_id} after retries, using fallback")
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