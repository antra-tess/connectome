"""
Context Manager Component
Component responsible for assembling the agent's context.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import time
import json # For formatting

try:
    import tiktoken
    tiktoken_found = True
except ImportError:
    tiktoken_found = False
    # logger is not defined yet at module level, handle below

from ..base_component import Component
from .global_attention_component import GlobalAttentionComponent
from .space.container_component import ContainerComponent
from .veil_producer_component import VeilProducer # Needed for interacting with element representations
# Need BaseElement for type hinting elements found in container
from ..base import BaseElement
# Need UplinkProxy and its components to interact with them
from ..uplink import UplinkProxy
from .uplink.connection_component import UplinkConnectionComponent
from .uplink.cache_component import RemoteStateCacheComponent
# Import LLM Message structure for history typing
from ...llm.provider_interface import LLMMessage, LLMToolCall
from .memory.structured_memory_component import StructuredMemoryComponent # Add dependency
from .simple_representation_component import SimpleRepresentationComponent # Import the new component
from .history_component import HistoryComponent # Need for target element
from ..elements.chat_element import ChatElement # Need for type check
from .messaging.conversation_info_component import ConversationInfoComponent # Import new component

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not tiktoken_found:
    logger.warning("tiktoken library not found. Using rough token estimation.")

# Simple token estimation (very rough)
def estimate_tokens_fallback(text: str) -> int:
    # TODO: Replace with actual tokenizer logic (e.g., tiktoken)
    return len(text) // 4

class ContextManagerComponent(Component):
    """
    Gathers, filters, compresses, and formats information (including recent history,
    structured memories, and element VEILs) to create the final context 
    presented to the agent's core reasoning model.
    
    Lives within the InnerSpace.
    """
    
    COMPONENT_TYPE: str = "context_manager"
    # Declare dependencies
    DEPENDENCIES: List[str] = [
        ContainerComponent.COMPONENT_TYPE, 
        GlobalAttentionComponent.COMPONENT_TYPE,
        StructuredMemoryComponent.COMPONENT_TYPE # Added memory store dependency
    ]
    
    # Events this component might handle (e.g., requests to rebuild context)
    HANDLED_EVENT_TYPES: List[str] = [
        "rebuild_context_request", # Explicit trigger
        "element_state_changed",   # Implicit trigger (needs careful handling)
        "attention_focus_changed",  # From GlobalAttentionComponent
        "external_message_received", # Event type for incoming messages
        "memory_processed_up_to", # Event to update marker
        "agent_message_sent" # <<< Added new event type
    ]
    
    def __init__(self, element: Optional[BaseElement] = None,
                 token_budget: int = 4000, # Example budget
                 compression_strategy: str = "truncate_recent", # More specific strategy
                 max_items_per_element: int = 10, # Limit items shown per element
                 history_max_turns: int = 20, # Max history turns to consider
                 max_memories_in_context: int = 5, # New config for memories
                 tokenizer_model: str = "cl100k_base", # Default for tiktoken
                 memory_threshold_tokens: int = 10000, # Threshold to trigger warning
                 **kwargs):
        """
        Initialize the context manager component.
        
        Args:
            element: The Element this component is attached to (InnerSpace).
            token_budget: Maximum token budget for the generated context.
            compression_strategy: Identifier for the compression method.
            max_items_per_element: Maximum items to show per element.
            history_max_turns: Maximum history turns to consider for *recent* context.
            max_memories_in_context: Maximum number of recent memories to include.
            tokenizer_model: The model name for tiktoken tokenization.
            memory_threshold_tokens: Token count of unprocessed history before warning.
            **kwargs: Passthrough for BaseComponent.
        """
        super().__init__(element, **kwargs)
        self._state = {
            "token_budget": token_budget,
            "compression_strategy": compression_strategy,
            "max_items_per_element": max_items_per_element,
            "history_max_turns": history_max_turns,
            "max_memories_in_context": max_memories_in_context, # Store new config
            "last_build_time": None,
            "last_context_hash": None,
            # Store history as list of dicts {role: str, content: any, name: Optional[str], tool_call_id: Optional[str]}
            "conversation_history": [],
            "memory_threshold_tokens": memory_threshold_tokens,
            "processed_history_marker_timestamp": None # Tracks history consolidated into memory
        }
        self._last_built_context: Optional[str] = None
        self._tokenizer = None
        if tiktoken_found:
            try:
                self._tokenizer = tiktoken.get_encoding(tokenizer_model)
                logger.info(f"Initialized tiktoken tokenizer with model: {tokenizer_model}")
            except Exception as e:
                logger.error(f"Failed to initialize tiktoken tokenizer '{tokenizer_model}': {e}. Falling back to estimation.")
                self._tokenizer = None
        else:
             self._tokenizer = None # Explicitly None if tiktoken not found

    def _get_token_count(self, text: str) -> int:
        """Estimates token count using tiktoken if available, otherwise falls back."""
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text))
            except Exception as e:
                 logger.warning(f"tiktoken encoding failed: {e}. Falling back to estimation.")
                 # Fall through to estimation
        return estimate_tokens_fallback(text)

    # --- History Management Methods --- 

    def add_history_turn(self, role: str, content: Any, 
                         name: Optional[str] = None, 
                         tool_call_id: Optional[str] = None, # For tool results
                         timestamp: Optional[int] = None):
        """Adds a turn to the conversation history."""
        if not self._is_initialized or not self._is_enabled:
            return
            
        turn = {
            "role": role,
            "content": content, # Could be string or structured data (e.g., tool result)
            "name": name,
            "tool_call_id": tool_call_id,
            "timestamp": timestamp or int(time.time() * 1000)
        }
        self._state["conversation_history"].append(turn)
        logger.debug(f"Added history turn: Role={role}, Content Type={type(content).__name__}")
        
        # Optional: Prune history if it exceeds a certain length/token count immediately
        # self._prune_history()

    def clear_history(self):
        """Clears the entire conversation history."""
        self._state["conversation_history"] = []
        self._state["processed_history_marker_timestamp"] = None # Reset marker too
        logger.info("Conversation history cleared.")
        
    def _get_recent_history(self) -> List[Dict[str, Any]]:
         """Gets the most recent history turns up to the configured limit."""
         max_turns = self._state.get("history_max_turns", 20)
         return self._state["conversation_history"][-max_turns:]
         
    def get_history_tail(self, before_timestamp: Optional[int], token_limit: int) -> List[Dict[str, Any]]:
        """
        Retrieves a chronological sequence of history messages ending just before
        a given timestamp, up to a specified token limit.

        Args:
            before_timestamp: The timestamp (exclusive) before which to retrieve messages.
                              If None, retrieves from the end of the history.
            token_limit: The maximum number of tokens the tail should contain.

        Returns:
            A list of history turn dictionaries, in chronological order.
        """
        history = self._state["conversation_history"]
        if not history:
            return []

        end_index = len(history)
        if before_timestamp is not None:
            # Find the index of the first message AT or AFTER the timestamp
            found_index = -1
            for i, turn in enumerate(history):
                if turn.get("timestamp", 0) >= before_timestamp:
                    found_index = i
                    break
            # If found, the end index is the index of that message (exclusive)
            # If not found (all messages are before), the end index is the length
            if found_index != -1:
                end_index = found_index

        tail_messages = []
        current_tokens = 0
        # Iterate backwards from the end_index
        for i in range(end_index - 1, -1, -1):
            turn = history[i]
            turn_content_str = "" # String representation for token counting

            # Attempt to get a string representation of the content for tokenization
            content = turn.get("content")
            if isinstance(content, str):
                turn_content_str = content
            elif isinstance(content, dict):
                 # Handle common structures like LLM responses or tool calls
                 if "content" in content and isinstance(content["content"], str):
                      turn_content_str = content["content"]
                 elif "text" in content and isinstance(content["text"], str):
                      turn_content_str = content["text"]
                 elif "tool_calls" in content and isinstance(content["tool_calls"], list):
                      # Simple representation for tool calls
                      turn_content_str = json.dumps(content["tool_calls"])
                 else:
                      try: # Fallback: try to JSON dump it
                          turn_content_str = json.dumps(content)
                      except TypeError:
                          turn_content_str = str(content) # Last resort
            elif content is not None:
                turn_content_str = str(content)

            # Estimate tokens for this turn's content
            turn_tokens = self._get_token_count(turn_content_str)
            # Rough estimate for role/name/etc overhead - adjust as needed
            turn_tokens += 10

            if current_tokens + turn_tokens <= token_limit:
                tail_messages.append(turn)
                current_tokens += turn_tokens
            else:
                # Stop if adding this turn exceeds the limit
                logger.debug(f"Token limit ({token_limit}) reached while building tail.")
                break

        # Return the collected messages in chronological order
        return tail_messages[::-1]

    def update_processed_marker(self, timestamp: int):
        """Updates the marker indicating history before this time is in memory."""
        current_marker = self._state.get("processed_history_marker_timestamp")
        if current_marker is None or timestamp > current_marker:
            self._state["processed_history_marker_timestamp"] = timestamp
            logger.info(f"Updated processed history marker to timestamp: {timestamp}")
        else:
             logger.debug(f"Skipping update of processed marker ({timestamp}) - current marker is newer ({current_marker})")

    def _calculate_unprocessed_history_info(self) -> Tuple[int, int]:
        """
        Calculates the token and message count of history not yet processed into memory.
        
        Returns:
            A tuple: (unprocessed_token_count, unprocessed_message_count)
        """
        marker_ts = self._state.get("processed_history_marker_timestamp")
        history = self._state.get("conversation_history", [])
        unprocessed_tokens = 0
        unprocessed_messages = 0
        start_index = 0

        if marker_ts is not None:
            # Find the index *after* the marker timestamp
            for i, turn in enumerate(history):
                 if turn.get("timestamp", 0) > marker_ts:
                      start_index = i
                      break
            else: # If loop completes without finding a newer message
                 start_index = len(history)

        # Calculate tokens for messages from start_index onwards
        for i in range(start_index, len(history)):
            turn = history[i]
            unprocessed_messages += 1
            # Use same token counting logic as get_history_tail
            content = turn.get("content")
            turn_content_str = ""
            if isinstance(content, str):
                turn_content_str = content
            elif isinstance(content, dict):
                 # Handle common structures
                 if "content" in content and isinstance(content["content"], str):
                      turn_content_str = content["content"]
                 elif "text" in content and isinstance(content["text"], str):
                      turn_content_str = content["text"]
                 elif "tool_calls" in content and isinstance(content["tool_calls"], list):
                      turn_content_str = json.dumps(content["tool_calls"])
                 else:
                      try: turn_content_str = json.dumps(content)
                      except TypeError: turn_content_str = str(content)
            elif content is not None:
                turn_content_str = str(content)
            unprocessed_tokens += self._get_token_count(turn_content_str) + 10 # Add overhead
            
        logger.debug(f"Unprocessed history info: Tokens={unprocessed_tokens}, Messages={unprocessed_messages} (Marker={marker_ts})")
        return unprocessed_tokens, unprocessed_messages

    # --- Method to get unprocessed history chunks --- 
    async def get_unprocessed_history_chunks(self, limit: Optional[int] = 1, max_chunk_tokens: int = 2000) -> List[List[Dict[str, Any]]]:
        """
        Retrieves unprocessed history messages, grouped into chunks.

        Args:
            limit: Maximum number of chunks to return (None for all).
            max_chunk_tokens: Target maximum token count per chunk.

        Returns:
            A list of chunks, where each chunk is a list of message dictionaries.
        """
        logger.debug(f"Getting unprocessed history chunks (limit={limit}, max_tokens={max_chunk_tokens})")
        history = self._state.get("conversation_history", [])
        marker_ts = self._state.get("processed_history_marker_timestamp")
        start_index = 0

        if marker_ts is not None:
            # Find the index *after* the marker timestamp
            for i, turn in enumerate(history):
                 if turn.get("timestamp", 0) > marker_ts:
                      start_index = i
                      break
            else: # If loop completes without finding a newer message
                 start_index = len(history)
                
        if start_index >= len(history):
            logger.debug("No unprocessed history messages found after marker.")
            return [] # No unprocessed messages
           
        all_chunks = []
        current_chunk = []
        current_chunk_tokens = 0
        
        for i in range(start_index, len(history)):
            turn = history[i]
            # Use same token counting logic as elsewhere
            content = turn.get("content")
            turn_content_str = ""
            if isinstance(content, str):
                turn_content_str = content
            elif isinstance(content, dict):
                 if "content" in content and isinstance(content["content"], str):
                      turn_content_str = content["content"]
                 elif "text" in content and isinstance(content["text"], str):
                      turn_content_str = content["text"]
                 elif "tool_calls" in content and isinstance(content["tool_calls"], list):
                      turn_content_str = json.dumps(content["tool_calls"])
                 else:
                      try: turn_content_str = json.dumps(content)
                      except TypeError: turn_content_str = str(content)
            elif content is not None:
                turn_content_str = str(content)
            
            turn_tokens = self._get_token_count(turn_content_str) + 10 # Add overhead

            # If adding this turn exceeds the limit AND the current chunk is not empty, finalize the current chunk
            if current_chunk and (current_chunk_tokens + turn_tokens > max_chunk_tokens):
                all_chunks.append(current_chunk)
                # Check if we've reached the chunk limit
                if limit is not None and len(all_chunks) >= limit:
                     break
                # Start a new chunk
                current_chunk = [turn]
                current_chunk_tokens = turn_tokens
            else:
                # Add to the current chunk
                current_chunk.append(turn)
                current_chunk_tokens += turn_tokens
               
        # Add the last chunk if it's not empty and limit not reached
        if current_chunk and (limit is None or len(all_chunks) < limit):
             all_chunks.append(current_chunk)
            
        logger.debug(f"Returning {len(all_chunks)} unprocessed history chunks.")
        return all_chunks

    # --- Helper Methods --- 

    def _get_container_comp(self) -> Optional[ContainerComponent]:
        """Helper to get the ContainerComponent from the attached element."""
        if not self.element: return None
        return self.element.get_component(ContainerComponent)

    def _get_attention_comp(self) -> Optional[GlobalAttentionComponent]:
        """Helper to get the GlobalAttentionComponent from the attached element."""
        if not self.element: return None
        return self.element.get_component(GlobalAttentionComponent)

    def _get_memory_store_comp(self) -> Optional[StructuredMemoryComponent]:
        """Helper to get the StructuredMemoryComponent from the attached element."""
        if not self.element: return None
        return self.element.get_component(StructuredMemoryComponent)
        
    # --- Core Context Building Logic --- 

    def build_context(self) -> str:
        """
        Builds the agent context string.
        """
        logger.info(f"Building context (budget: {self._state['token_budget']}, strategy: {self._state['compression_strategy']})...")
        start_time = time.time()
        
        # Check memory threshold
        unprocessed_tokens, _ = self._calculate_unprocessed_history_info()
        threshold = self._state.get("memory_threshold_tokens", 10000)
        memory_alert = unprocessed_tokens > threshold
        if memory_alert:
            logger.info(f"Memory threshold exceeded: {unprocessed_tokens} unprocessed tokens (threshold: {threshold})")
            
        # 1. Gather Raw Information (now includes history)
        raw_info = self._gather_information()
        
        # 2. Filter & Prioritize Information (now includes history)
        prioritized_items = self._filter_and_prioritize(raw_info)
        
        # 3. Format intermediate representation (structured text)
        formatted_items = self._format_intermediate(prioritized_items, memory_alert=memory_alert)
        
        # 4. Compress the formatted representation
        final_context_str = self._compress_information(formatted_items)
        
        # Update state
        self._state["last_build_time"] = int(time.time() * 1000)
        self._last_built_context = final_context_str
        
        end_time = time.time()
        logger.info(f"Context build complete in {end_time - start_time:.3f} seconds. Length: {len(final_context_str)} chars.")
        
        return final_context_str

    def _gather_information(self) -> Dict[str, Any]:
        """Gather raw data including history, memories, elements (with simple representation), and uplinks."""
        logger.debug("Gathering raw context information...")
        gathered_data = {"elements": {}, "history": [], "memories": []} # Added memories key
        
        # Get attention info
        attention_comp = self._get_attention_comp()
        gathered_data["attention"] = attention_comp.get_state() if attention_comp else {}
        
        # Get recent history
        gathered_data["history"] = self._get_recent_history()
        
        # Get recent memories
        memory_store = self._get_memory_store_comp()
        if memory_store:
             max_memories = self._state.get("max_memories_in_context", 5)
             if max_memories > 0:
                  try:
                       gathered_data["memories"] = memory_store.get_memories(limit=max_memories)
                       logger.debug(f"Gathered {len(gathered_data['memories'])} recent memories.")
                  except Exception as e:
                       logger.error(f"Failed to retrieve memories from StructuredMemoryComponent: {e}")
        
        # Get representation of InnerSpace itself (if it has the component)
        if self.element:
            inner_space_data = {
                "id": self.element.id, 
                "name": self.element.name, 
                "type": self.element.__class__.__name__, 
                "simple_representation": None # Placeholder
            }
            rep_comp = self.element.get_component(SimpleRepresentationComponent)
            if rep_comp:
                 try:
                      inner_space_data["simple_representation"] = rep_comp.produce_representation()
                 except Exception as e:
                      logger.warning(f"Failed to get simple representation from InnerSpace {self.element.id}: {e}")
            gathered_data["elements"][self.element.id] = inner_space_data
        
        # Get representations of mounted elements
        container = self._get_container_comp()
        if container:
             mounted_elements = container.get_mounted_elements()
             for mount_id, element in mounted_elements.items():
                  element_data = {
                       "id": element.id,
                       "name": element.name,
                       "type": element.__class__.__name__,
                       "mount_id": mount_id,
                       "simple_representation": None, # Placeholder
                       "uplink_cache": None # Keep uplink separate for now
                  }
                  # Get Simple Representation if available
                  rep_comp = element.get_component(SimpleRepresentationComponent)
                  if rep_comp:
                       try:
                           element_data["simple_representation"] = rep_comp.produce_representation()
                       except Exception as e:
                            logger.warning(f"Failed to get simple representation from mounted element {element.id} ({mount_id}): {e}")
                  
                  # If it's an uplink, try to get cached state
                  if isinstance(element, UplinkProxy):
                       cache_comp = element.get_component(RemoteStateCacheComponent)
                       if cache_comp:
                            element_data["uplink_cache"] = cache_comp.get_synced_remote_state(force_sync=False)

                  gathered_data["elements"][element.id] = element_data
        
        return gathered_data

    def _filter_and_prioritize(self, raw_info: Dict[str, Any]) -> List[Tuple[int, str, Dict[str, Any]]]:
        """Prioritize memories, history, and elements. Returns list of (priority, type, data)."""
        logger.debug("Filtering and prioritizing context information...")
        prioritized_list = []
        
        # --- Prioritize Memories ---
        memory_items = raw_info.get("memories", [])
        for i, mem_data in enumerate(memory_items):
            # Assign high priority (negative), ensuring they come before history/elements
            # Less negative means more recent memory (higher priority within memories)
            priority = -1000 - len(memory_items) + i 
            prioritized_list.append((priority, "memory", mem_data))
            
        # --- Prioritize History --- 
        history_items = raw_info.get("history", [])
        for i, turn in enumerate(history_items):
            # Assign priority lower than memories but higher than elements
            # More recent turns have lower (better) priority value
            priority = -100 - len(history_items) + i 
            prioritized_list.append((priority, "history", turn))
            
        # --- Prioritize Elements --- 
        elements_data = raw_info.get("elements", {})
        attention_data = raw_info.get("attention", {})
        attention_focus = attention_data.get("current_focus")

        for element_id, data in elements_data.items():
            priority = 0 # Base priority for elements (comes after history)
            if data.get("type") == "InnerSpace": priority = 1 # InnerSpace important
            elif element_id == attention_focus: priority = 5 # Focused element
            elif data.get("type") == "ChatElement": priority = 10
            elif data.get("type") == "UplinkProxy": priority = 15
            else: priority = 100 # Default lower priority
            prioritized_list.append((priority, "element", data))

        prioritized_list.sort(key=lambda item: item[0])
        logger.debug(f"Prioritized context items ({len(prioritized_list)} total): {[item[1] for item in prioritized_list]}")
        return prioritized_list

    def _format_intermediate(self, prioritized_items: List[Tuple[int, str, Dict[str, Any]]], memory_alert: bool = False) -> str:
        """Formats prioritized items (memories, history, elements using simple XML representation) into a string."""
        logger.debug("Formatting intermediate context representation...")
        formatted_context = ""

        if memory_alert:
            # Use XML comment for system notes?
            formatted_context += "<!-- System Note: Unprocessed history is large. Consider running memory processing. -->\n\n"

        # Format items based on type
        for _, item_type, item_data in prioritized_items:
            if item_type == "memory":
                mem_id = item_data.get('memory_id', 'unknown')
                mem_ts = item_data.get('timestamp', 'unknown')
                mem_content = item_data.get('content', [])
                
                formatted_context += f"<memory id='{mem_id}' timestamp='{mem_ts}'>\n"
                if isinstance(mem_content, list):
                    for turn in mem_content:
                        role = turn.get('role', 'entry')
                        content = turn.get('content', '')
                        # Escape content for basic XML safety
                        safe_content = str(content).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        formatted_context += f"  <{role}>{safe_content}</{role}>\n"
                elif isinstance(mem_content, str):
                     safe_content = mem_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                     formatted_context += f"  <content>{safe_content}</content>\n"
                else:
                     safe_content = str(mem_content).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                     formatted_context += f"  <content>{safe_content}</content>\n"
                formatted_context += "</memory>\n\n"

            elif item_type == "history":
                role = item_data.get('role')
                content = item_data.get('content')
                name = item_data.get('name')
                tool_call_id = item_data.get('tool_call_id')
                
                # Format history as XML-like turns
                tag = f"turn role='{role}'"
                if name: tag += f" name='{name}'"
                if tool_call_id: tag += f" tool_call_id='{tool_call_id}'"
                
                # Basic content formatting/escaping
                safe_content = str(content).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                if isinstance(content, dict) and 'tool_calls' in content:
                     safe_content = json.dumps(content['tool_calls']).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                elif isinstance(content, dict):
                     # Attempt JSON dump for other dicts
                     try: safe_content = json.dumps(content).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                     except TypeError: pass # Keep string representation
                     
                formatted_context += f"<{tag}>{safe_content}</turn>\n"

            elif item_type == "element":
                # Get the representation produced by SimpleRepresentationComponent
                representation = item_data.get("simple_representation")
                
                if representation:
                    # Directly include the XML generated by the component
                    formatted_context += f"{representation}\n"
                else:
                    # Fallback: Create a basic info tag if no representation component
                    element_id = item_data.get("id", "unknown")
                    element_type = item_data.get("type", "UnknownType")
                    element_name = item_data.get("name", "")
                    safe_name = element_name.replace('"', '&quot;')
                    formatted_context += f'<element_info type="{element_type}" name="{safe_name}" id="{element_id}" representation="none"/>\n'
                
                # Add uplink cache separately if it exists? Or expect representation component to handle it?
                # Let's keep it separate for now as extra info.
                uplink_cache = item_data.get("uplink_cache")
                if uplink_cache:
                     try: 
                          cache_str = json.dumps(uplink_cache).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                          formatted_context += f"  <uplink_cache>{cache_str}</uplink_cache>\n"
                     except TypeError: 
                          formatted_context += "  <uplink_cache>(Cannot serialize cache)</uplink_cache>\n"
                formatted_context += "\n" # Add blank line after element section

        return formatted_context.strip()
        
    def _compress_information(self, formatted_context: str) -> str:
         """Applies compression/truncation based on token budget and strategy."""
         logger.debug(f"Compressing context using strategy: {self._state['compression_strategy']}")
         budget = self._state['token_budget']
         current_tokens = self._get_token_count(formatted_context)

         if current_tokens <= budget:
              return formatted_context # No compression needed
              
         # --- Implement Compression Strategies --- 
         strategy = self._state['compression_strategy']
         if strategy == "truncate_recent":
             # Simple truncation from the beginning (keeping the most recent)
             # This needs a more sophisticated approach that respects message boundaries
             # and uses the actual token count per line/message.
             # Placeholder: Rough character-based truncation
             estimated_chars_per_token = 4
             max_chars = budget * estimated_chars_per_token
             truncated_context = formatted_context[-max_chars:]
             logger.warning(f"Context exceeded budget ({current_tokens}/{budget} tokens). Truncating (basic). Final tokens ~{self._get_token_count(truncated_context)}")
             return truncated_context
         elif strategy == "summarize_oldest":
             # TODO: Implement summarization logic (e.g., call LLM to summarize oldest parts)
             logger.warning("'summarize_oldest' compression strategy not yet implemented. Using basic truncation.")
             estimated_chars_per_token = 4
             max_chars = budget * estimated_chars_per_token
             truncated_context = formatted_context[-max_chars:]
             return truncated_context
         else:
             logger.error(f"Unknown compression strategy: {strategy}. No compression applied.")
             return formatted_context

    # Placeholder - Needs implementation based on attention/history
    def get_current_routing_context(self) -> Dict[str, Optional[str]]:
        """
        Determines the most likely target for a response based on recent history or attention.

        Returns:
            A dictionary containing 'adapter_id' and 'conversation_id' or None if unknown.
        """
        logger.debug("Determining current routing context...")

        # Priority 1: Explicit attention focus (if it provides routing)
        attention_comp = self._get_attention_comp()
        if attention_comp:
            requests = attention_comp.get_attention_requests()
            if requests:
                # Simplistic: Use the first request found. Better logic might be needed.
                first_req_key = list(requests.keys())[0]
                first_req = requests[first_req_key]
                # Check if the element in focus provides routing info (e.g., via VEIL or state)
                # This is highly dependent on element implementation.
                # Placeholder: Assume focus doesn't directly give routing yet.
                pass 

        # Priority 2: Look backwards in history for the most recent user message
        # with adapter/conversation context (if available)
        # This requires history turns to potentially store this metadata.
        # Placeholder: Assume history turns don't have this metadata yet.
        history = self._state["conversation_history"]
        for turn in reversed(history):
             # Example: Check if turn metadata has routing info
             # if turn.get("metadata") and turn["metadata"].get("adapter_id"):
             #     return {
             #         "adapter_id": turn["metadata"].get("adapter_id"),
             #         "conversation_id": turn["metadata"].get("conversation_id")
             #     }
             pass # Placeholder
             
        # Priority 3: Default routing? Configuration? 
        # For now, return None if no context found
        logger.warning("Could not determine routing context from history or attention.")
        return {"adapter_id": None, "conversation_id": None}

    # --- Event Handling --- 

    def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handle events that might trigger context rebuilds or history updates.
        """
        event_type = event.get("event_type")
        payload = event.get("payload") # Changed from data to payload based on ActivityClient

        # Handle incoming external messages 
        if event_type == "external_message_received":
            # Assuming payload *is* the message turn structure
            # Needs refinement based on actual event structure from ActivityClient/HostEventLoop
            role = payload.get("role", "user") 
            content = payload.get("content")
            name = payload.get("name") # <<< Ensure adapter provides this
            timestamp = payload.get("timestamp") # <<< Ensure adapter provides this
            # TODO: Extract routing context (adapter_id, conversation_id) if available
            # and store it perhaps as metadata within the history turn, or use it
            # to find the CORRECT ChatElement history to add to.
            # For now, adds to *this* component's history which might be wrong for ChatElements
            if content:
                # !!! This adds to the ContextManager's history, NOT a specific ChatElement's !!!
                # !!! This needs routing logic based on conversation_id/adapter_id in payload !!!
                logger.warning("Handling external_message_received by adding to ContextManager history - NEEDS ROUTING to ChatElement!")
                self.add_history_turn(role=role, content=content, name=name, timestamp=timestamp)
                return True
                
        # Handle internal event confirming agent message was sent externally
        elif event_type == "agent_message_sent":
            if not isinstance(payload, dict):
                logger.warning(f"Received {event_type} event without valid payload dict.")
                return False
                
            # Extract details needed to add to history
            adapter_id = payload.get("adapter_id")
            conversation_id = payload.get("conversation_id")
            text = payload.get("text")
            agent_name = payload.get("agent_name")
            timestamp = payload.get("timestamp")
            
            if not all([adapter_id, conversation_id, text, agent_name, timestamp]):
                 logger.warning(f"Received {event_type} event with missing data in payload: {payload}")
                 return False
                 
            # Find the target ChatElement and add to its history
            container = self._get_container_comp()
            if not container:
                 logger.error(f"Cannot handle {event_type}: ContainerComponent not found on {self.element.id}")
                 return False
                 
            target_element = self._find_chat_element_by_conversation(container, adapter_id, conversation_id)
            
            if target_element:
                history_comp = target_element.get_component(HistoryComponent)
                if history_comp:
                     try:
                          history_comp.add_history_turn(
                               role="assistant", 
                               content=text,
                               name=agent_name,
                               timestamp=timestamp
                          )
                          logger.info(f"Recorded sent message to history of {target_element.id}")
                          return True # Event handled
                     except Exception as e:
                          logger.error(f"Error adding turn to HistoryComponent of {target_element.id}: {e}", exc_info=True)
                else:
                     logger.warning(f"Target ChatElement {target_element.id} for {event_type} is missing HistoryComponent.")
            else:
                 logger.warning(f"Could not find target ChatElement for {event_type} (adapter={adapter_id}, conv={conversation_id}) in {self.element.id}")
                 
            return False # Indicate event not fully handled if target not found/updated

        elif event_type == "rebuild_context_request":
            # Triggered externally to force a rebuild (e.g., by agent loop)
            self.build_context()
            return True
        elif event_type == "memory_processed_up_to": # Handle marker update event
            timestamp = event.get("data", {}).get("timestamp")
            if isinstance(timestamp, int):
                 self.update_processed_marker(timestamp)
                 return True # Consumed event
            else:
                 logger.warning("Received memory_processed_up_to event without valid timestamp.")
                 return False
        elif event_type in ["element_state_changed", "attention_focus_changed"]:
            # These might invalidate the last context, but rebuilding immediately
            # on every change could be inefficient. The agent loop might poll
            # build_context before its next cycle instead.
            logger.debug(f"Received event '{event_type}', context may need rebuilding.")
            # Mark context as potentially stale if needed? 
            self._last_context_hash = None # Invalidate hash
            return False # Don't consume the event, others might need it

        return False 
        
    # --- Helper to find ChatElement (Refined) ---
    def _find_chat_element_by_conversation(self, container: ContainerComponent, adapter_id: str, conversation_id: str) -> Optional[BaseElement]:
        """
        Finds a mounted element that has a ConversationInfoComponent matching
        the provided adapter_id and conversation_id.
        """
        if not container:
             return None
             
        mounted_elements = container.get_mounted_elements()
        for element in mounted_elements.values():
            # Check if the element has the ConversationInfoComponent
            info_comp = element.get_component(ConversationInfoComponent)
            if info_comp:
                 # Check if the IDs match
                 try:
                      if info_comp.get_adapter_id() == adapter_id and \
                         info_comp.get_conversation_id() == conversation_id:
                          logger.debug(f"Found target ChatElement {element.id} via ConversationInfoComponent for conv {conversation_id}")
                          return element
                 except Exception as e:
                      # Should not happen with simple getters, but good practice
                      logger.warning(f"Error accessing ConversationInfoComponent on {element.id}: {e}")
                          
        logger.debug(f"Could not find ChatElement via ConversationInfoComponent for conv {conversation_id}")
        return None # Not found 