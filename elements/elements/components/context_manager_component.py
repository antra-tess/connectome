"""
Context Manager Component
Component responsible for assembling the agent's context.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import time
import json # For formatting

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple token estimation (very rough)
def estimate_tokens(text: str) -> int:
    # TODO: Replace with actual tokenizer logic (e.g., tiktoken)
    return len(text) // 4

class ContextManagerComponent(Component):
    """
    Gathers, filters, compresses, and formats information to create the final context 
    presented to the agent's core reasoning model.
    
    Lives within the InnerSpace.
    """
    
    COMPONENT_TYPE: str = "context_manager"
    # Declare dependencies on other components expected to be on the same element (InnerSpace)
    DEPENDENCIES: List[str] = [ContainerComponent.COMPONENT_TYPE, GlobalAttentionComponent.COMPONENT_TYPE]
    
    # Events this component might handle (e.g., requests to rebuild context)
    HANDLED_EVENT_TYPES: List[str] = [
        "rebuild_context_request", # Explicit trigger
        "element_state_changed",   # Implicit trigger (needs careful handling)
        "attention_focus_changed",  # From GlobalAttentionComponent
        "external_message_received" # Event type for incoming messages
    ]
    
    def __init__(self, element: Optional[BaseElement] = None,
                 token_budget: int = 4000, # Example budget
                 compression_strategy: str = "truncate_recent", # More specific strategy
                 max_items_per_element: int = 10, # Limit items shown per element
                 history_max_turns: int = 20, # Max history turns to consider
                 **kwargs):
        """
        Initialize the context manager component.
        
        Args:
            element: The Element this component is attached to (InnerSpace).
            token_budget: Maximum token budget for the generated context.
            compression_strategy: Identifier for the compression method.
            max_items_per_element: Maximum items to show per element.
            history_max_turns: Maximum history turns to consider.
            **kwargs: Passthrough for BaseComponent.
        """
        super().__init__(element, **kwargs)
        self._state = {
            "token_budget": token_budget,
            "compression_strategy": compression_strategy,
            "max_items_per_element": max_items_per_element,
            "history_max_turns": history_max_turns,
            "last_build_time": None,
            "last_context_hash": None,
            # Store history as list of dicts {role: str, content: any, name: Optional[str], tool_call_id: Optional[str]}
            "conversation_history": []
        }
        self._last_built_context: Optional[str] = None

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
        logger.info("Conversation history cleared.")
        
    def _get_recent_history(self) -> List[Dict[str, Any]]:
         """Gets the most recent history turns up to the configured limit."""
         max_turns = self._state.get("history_max_turns", 20)
         return self._state["conversation_history"][-max_turns:]
         
    # --- Helper Methods --- 

    def _get_container_comp(self) -> Optional[ContainerComponent]:
        """Helper to get the ContainerComponent from the attached element."""
        if not self.element: return None
        return self.element.get_component(ContainerComponent)

    def _get_attention_comp(self) -> Optional[GlobalAttentionComponent]:
        """Helper to get the GlobalAttentionComponent from the attached element."""
        if not self.element: return None
        return self.element.get_component(GlobalAttentionComponent)
        
    # --- Core Context Building Logic --- 

    def build_context(self) -> str:
        """
        Builds the agent context string.
        """
        logger.info(f"Building context (budget: {self._state['token_budget']}, strategy: {self._state['compression_strategy']})...")
        start_time = time.time()
        
        # 1. Gather Raw Information (now includes history)
        raw_info = self._gather_information()
        
        # 2. Filter & Prioritize Information (now includes history)
        prioritized_items = self._filter_and_prioritize(raw_info)
        
        # 3. Format intermediate representation (structured text)
        formatted_items = self._format_intermediate(prioritized_items)
        
        # 4. Compress the formatted representation
        final_context_str = self._compress_information(formatted_items)
        
        # Update state
        self._state["last_build_time"] = int(time.time() * 1000)
        self._last_built_context = final_context_str
        
        end_time = time.time()
        logger.info(f"Context build complete in {end_time - start_time:.3f} seconds. Length: {len(final_context_str)} chars.")
        
        return final_context_str

    def _gather_information(self) -> Dict[str, Any]:
        """Gather raw data including history, elements, and uplinks."""
        logger.debug("Gathering raw context information...")
        gathered_data = {"elements": {}, "history": []}
        
        # Get attention info
        attention_comp = self._get_attention_comp()
        gathered_data["attention"] = attention_comp.get_state() if attention_comp else {}
        
        # Get recent history
        gathered_data["history"] = self._get_recent_history()
        
        # Get representation of InnerSpace itself
        if self.element:
            inner_space_data = {"id": self.element.id, "name": self.element.name, "type": "InnerSpace", "veil": None}
            if self.element.has_component(VeilProducer):
                try:
                    inner_space_data["veil"] = self.element.get_component(VeilProducer).produce_veil()
                except Exception as e:
                    logger.warning(f"Failed to get VEIL from InnerSpace {self.element.id}: {e}")
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
                       "veil": None,
                       "uplink_cache": None
                  }
                  # Get VEIL if available
                  if element.has_component(VeilProducer):
                       try:
                           veil_producer = element.get_component(VeilProducer)
                           element_data["veil"] = veil_producer.produce_veil()
                       except Exception as e:
                            logger.warning(f"Failed to get VEIL from mounted element {element.id} ({mount_id}): {e}")
                  # If it's an uplink, try to get cached state
                  if isinstance(element, UplinkProxy):
                       cache_comp = element.get_component(RemoteStateCacheComponent)
                       if cache_comp:
                            # Get cache without forcing sync for context building
                            element_data["uplink_cache"] = cache_comp.get_synced_remote_state(force_sync=False)
                            # Could also get history bundles if needed: cache_comp.get_history_bundles()

                  gathered_data["elements"][element.id] = element_data
        
        return gathered_data

    def _filter_and_prioritize(self, raw_info: Dict[str, Any]) -> List[Tuple[int, str, Dict[str, Any]]]:
        """Prioritize history and elements. Returns list of (priority, type, data)."""
        logger.debug("Filtering and prioritizing context information...")
        prioritized_list = []
        
        # --- Prioritize History --- 
        # Add history items first, with highest priority (maybe weighted by recency?)
        history_items = raw_info.get("history", [])
        for i, turn in enumerate(history_items):
            # Simple recency priority: newer turns have lower (better) priority value
            # Assign negative priorities to ensure they come before elements
            priority = -len(history_items) + i 
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
        return prioritized_list

    def _format_intermediate(self, prioritized_items: List[Tuple[int, str, Dict[str, Any]]]) -> str:
        """(Basic Implementation) Formats prioritized history into a simple string."""
        logger.debug("Formatting intermediate context representation (basic history only)...")
        formatted_context = "" # Start with empty string

        # Filter only history items for this basic version
        history_items = [item for item in prioritized_items if item[1] == 'history']
        
        for priority, item_type, data in history_items:
            role = data.get("role")
            content = data.get("content")
            name = data.get("name") # For tool calls/results
            
            block = ""
            if role == "user":
                # Extract text if content is the structured dict
                text_content = content.get("text", "(User message content missing)") if isinstance(content, dict) else str(content)
                block = f"User: {text_content}\n"
            elif role == "assistant":
                # Extract text content and simplified tool call info
                text_content = None
                tool_calls_repr = ""
                if isinstance(content, dict):
                     text_content = content.get("content")
                     tool_calls = content.get("tool_calls")
                     if tool_calls:
                          tool_calls_repr = ", ".join([f"call:{tc.get('name')}" for tc in tool_calls])
                          tool_calls_repr = f" (Tools: {tool_calls_repr})"
                elif isinstance(content, str):
                     text_content = content # Older format? 
                     
                block = f"Assistant:{tool_calls_repr} {text_content or ''}\n"
            elif role == "system":
                block = f"System: {str(content)}\n"
            elif role == "tool":
                 # Represent tool result/error
                 tool_content = str(content.get("error")) if isinstance(content, dict) and "error" in content else str(content)
                 block = f"Tool Result (for {name}): {tool_content}\n"
            elif role == "internal_monologue":
                 block = f"Internal Thought: {str(content)}\n"
            else:
                 block = f"{role.capitalize()}: {str(content)}\n"
                 
            formatted_context += block

        # TODO: Add basic element info (e.g., names of mounted elements) later
        # TODO: Add attention focus info later
        
        logger.debug(f"Formatted intermediate context length: {len(formatted_context)} chars")
        return formatted_context
        
    def _compress_information(self, formatted_context: str) -> str:
         """(Basic Implementation) Simple truncation based on token budget."""
         token_budget = self._state.get("token_budget", 4000)
         estimated_tokens = estimate_tokens(formatted_context)
         logger.debug(f"Compressing context: Estimated tokens={estimated_tokens}, Budget={token_budget}")
         
         if estimated_tokens <= token_budget:
              return formatted_context # No compression needed
              
         # Simple truncation from the beginning (very basic strategy)
         # TODO: Implement smarter strategies (summarization, keep recent, etc.)
         chars_to_keep = token_budget * 4 # Rough estimate back to chars
         truncated_context = formatted_context[-chars_to_keep:]
         # Add an indicator that truncation occurred
         truncated_context = "[...]\n" + truncated_context 
         
         logger.warning(f"Context truncated from ~{estimated_tokens} tokens to ~{token_budget} tokens.")
         return truncated_context

    # Placeholder - Needs implementation based on attention/history
    def get_current_routing_context(self) -> Dict[str, Optional[str]]:
        """
        Determines the default target for outgoing messages based on the
        most recent incoming user message's metadata.
        Returns: Dict with 'adapter_id' and 'conversation_id'.
        """
        logger.debug("Determining current routing context...")
        adapter_id = None
        conv_id = None
        fallback_adapter = "default_adapter" # Define fallbacks
        fallback_conv = "default_conv"

        history = self._state.get("conversation_history", [])
        # Iterate backwards to find the latest user message
        for turn in reversed(history):
            if turn.get('role') == 'user':
                content = turn.get('content')
                # Check if content itself is a dict containing metadata (e.g., from ActivityListener)
                if isinstance(content, dict):
                     metadata = content.get('metadata') # Look for metadata within content
                     if metadata and isinstance(metadata, dict):
                          adapter_id = metadata.get('adapter_id')
                          conv_id = metadata.get('conversation_id')
                          logger.debug(f"Found routing context in last user message metadata: adapter={adapter_id}, conv={conv_id}")
                          break # Found the latest user message with metadata

                # Optional: Check if the turn *itself* has top-level metadata (alternative storage)
                # elif 'metadata' in turn and isinstance(turn['metadata'], dict):
                #      metadata = turn['metadata']
                #      adapter_id = metadata.get('adapter_id')
                #      conv_id = metadata.get('conversation_id')
                #      logger.debug(f"Found routing context in last user message top-level metadata: adapter={adapter_id}, conv={conv_id}")
                #      break

        if not adapter_id or not conv_id:
            logger.warning(f"Could not find routing metadata in recent history. Using fallbacks: adapter={fallback_adapter}, conv={fallback_conv}")

        return {
            "adapter_id": adapter_id or fallback_adapter,
            "conversation_id": conv_id or fallback_conv
        }

    # --- Event Handling --- 

    def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """Handle events delegated to this component."""
        event_type = event.get("event_type")
        
        # Handle incoming messages from external adapters
        # Based on the structure: {"event_type": "message_received", "adapter_type": "...", "data": {...}}
        # Where the overall event passed by ActivityClient/Host might be 
        # {"event_type": "external_message_received", "payload": <original_data_from_adapter>}
        # OR ActivityClient might pass the original event structure directly.
        # Assuming ActivityClient passes the original structure received on "normalized_event"
        # Let's look for the inner event_type first.
        
        inner_event_type = event.get("event_type") # Check top level first
        event_data = event.get("data", {}) if inner_event_type else event.get("payload", {}) # Get the data/payload block
        if not inner_event_type:
             inner_event_type = event_data.get("event_type") # Check inside payload if not top level

        # Let's standardize on handling the specific incoming message type
        if inner_event_type == "message_received":
            logger.debug(f"Handling incoming message_received event data")
            
            text_content = event_data.get("text")
            if not text_content:
                 logger.warning("Ignoring message_received event: Missing 'text' in data.")
                 return False
                 
            # Extract relevant info for history metadata
            adapter_id = event.get("source_adapter_id") # Get ID used by ActivityClient connection
            if not adapter_id:
                 logger.warning("Incoming message missing source_adapter_id needed for routing.")
                 # We might try to get it from the data block if adapters include it?
                 # adapter_id = event_data.get("adapter_id") or event_data.get("adapter_name")
            
            conv_id = event_data.get("conversation_id")
            sender_info = event_data.get("sender")
            message_id = event_data.get("message_id")
            thread_id = event_data.get("thread_id")
            external_timestamp = event_data.get("timestamp")

            if not conv_id:
                 logger.warning("Incoming message missing conversation_id.")
                 # Decide: Skip message or add with fallback?
                 # Adding with fallback might break routing.
                 return False # Skip if no conversation ID
                 
            history_metadata = {
                 "adapter_id": adapter_id,
                 "conversation_id": conv_id,
                 "message_id": message_id,
                 "sender": sender_info,
                 "thread_id": thread_id,
                 "external_timestamp": external_timestamp
            }
            # Filter out None values from metadata
            history_metadata = {k: v for k, v in history_metadata.items() if v is not None}
            
            # Structure the content for history
            history_content = {
                 "text": text_content,
                 "metadata": history_metadata 
            }
            
            # Add to history with role "user"
            self.add_history_turn(
                 role="user", 
                 content=history_content, 
                 timestamp=event.get("timestamp") # Use internal host timestamp
            )
            return True # Event handled
            
        # --- Handle other control events --- 
        elif event_type == "rebuild_context_request":
            logger.info("Rebuilding context due to explicit request.")
            self.build_context() 
            return True
            
        elif event_type == "attention_focus_changed":
            logger.debug("Attention focus changed, potentially rebuild context.")
            return False 

        return False 