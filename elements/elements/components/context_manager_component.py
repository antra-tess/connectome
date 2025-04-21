"""
Context Manager Component
Component responsible for assembling the agent's context.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Set
import time
import json # For formatting

try:
    import tiktoken
    tiktoken_found = True
except ImportError:
    tiktoken_found = False
    # logger is not defined yet at module level, handle below

from ..base_component import Component
from .space.container_component import ContainerComponent
from .veil_producer_component import VeilProducer # Needed for interacting with element representations
# Need BaseElement for type hinting elements found in container
from ..base import BaseElement
# Need UplinkProxy and its components to interact with them
from ..uplink import UplinkProxy
from .uplink.connection_component import UplinkConnectionComponent
from .uplink.cache_component import RemoteStateCacheComponent
# Import LLM Message structure for history typing
# from ...llm.provider_interface import LLMMessage, LLMToolCall
from .memory.structured_memory_component import StructuredMemoryComponent # Add dependency
from .simple_representation_component import SimpleRepresentationComponent # Import the new component
from .history_component import HistoryComponent # Need for target element
from ..elements.chat_element import ChatElement # Need for type check
# ConversationInfoComponent might be redundant if info is in representation/history
# from .messaging.conversation_info_component import ConversationInfoComponent 
# Import representation components
from .base_representation_component import BaseRepresentationComponent
from .chat.chat_representation_component import ChatElementRepresentationComponent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not tiktoken_found:
    logger.warning("tiktoken library not found. Using rough token estimation.")

# Removed estimate_tokens_fallback - token counting likely moves to HUD
# def estimate_tokens_fallback(text: str) -> int: ...

class ContextManagerComponent(Component):
    """
    Orchestrates the gathering of context from various sources (history, representations, 
    memory) within the InnerSpace, providing a structured context object for the HUD.
    """
    
    COMPONENT_TYPE: str = "context_manager"
    # Update dependencies
    DEPENDENCIES: Set[str] = { # Use Set for consistency
        ContainerComponent.COMPONENT_TYPE, 
        StructuredMemoryComponent.COMPONENT_TYPE,
        HistoryComponent.COMPONENT_TYPE, # Added
        # Doesn't strictly depend on Representation, but needs *some* source
        # Let's assume it finds representation components dynamically
    }
    
    # Events this component might handle 
    HANDLED_EVENT_TYPES: List[str] = [
        "rebuild_context_request", # Explicit trigger
        # Less implicit triggering now, HUD calls when needed by AgentLoop
        # "element_state_changed",   
        # "attention_focus_changed",  
        # "external_message_received",
        # "agent_message_sent"
    ]
    
    def __init__(self, element: Optional[BaseElement] = None,
                 # Removed token budget/compression
                 # token_budget: int = 4000,
                 # compression_strategy: str = "truncate_recent",
                 max_items_per_element: int = 10, # Keep for gathering mounted elements
                 history_max_turns: int = 20, # Limit history fetched
                 max_memories_to_gather: int = 5, # Limit memories fetched
                 # Removed tokenizer
                 # tokenizer_model: str = "cl100k_base",
                 # memory_threshold_tokens: int = 10000, # Removed
                 **kwargs):
        """
        Initialize the context manager component.
        """
        super().__init__(element, **kwargs)
        self._state = {
            # Removed token/compression related state
            # "token_budget": token_budget,
            # "compression_strategy": compression_strategy,
            "max_items_per_element": max_items_per_element,
            "history_max_turns": history_max_turns,
            "max_memories_to_gather": max_memories_to_gather,
            "last_build_time": None,
            # "last_context_hash": None, # Removed
            # "conversation_history": [], # Removed
            # "memory_threshold_tokens": memory_threshold_tokens, # Removed
            # "processed_history_marker_timestamp": None # Removed
        }
        # Removed tokenizer init
        # self._last_built_context: Optional[str] = None # Removed
        # if tiktoken_found: ... # Removed tokenizer block
        # else: ... # Removed tokenizer block

    # Removed _get_token_count
    # def _get_token_count(self, text: str) -> int: ...

    # --- REMOVE HISTORY MANAGEMENT METHODS --- 
    # def add_history_turn(...): ...
    # def clear_history(): ...
    # def _get_recent_history(): ...
    # def get_history_tail(...): ...
    # def update_processed_marker(...): ...
    # def get_unprocessed_history_chunks(...): ...
    # --- END REMOVED HISTORY METHODS --- 

    # --- Component Getters (Helper Methods) ---
    # Added getters for new dependencies
    def _get_history_comp(self) -> Optional[HistoryComponent]:
         if not self.element: return None
         return self.element.get_component_by_type(HistoryComponent.COMPONENT_TYPE)
         
    def _get_representation_comp(self, target_element: Optional[BaseElement] = None) -> Optional[BaseRepresentationComponent]:
         """Finds the primary representation component (e.g., for chat)."""
         # Prioritize Chat representation if present
         if target_element:
             chat_rep = target_element.get_component_by_type(ChatElementRepresentationComponent.COMPONENT_TYPE)
             if chat_rep: return chat_rep
         # Fallback to simple or base
         simple_rep = self.element.get_component_by_type(SimpleRepresentationComponent.COMPONENT_TYPE)
         if simple_rep: return simple_rep
         base_rep = self.element.get_component_by_type(BaseRepresentationComponent.COMPONENT_TYPE)
         if base_rep: return base_rep
         logger.warning(f"[{self.element.id if self.element else '?'}] ContextManager could not find a Representation Component.")
         return None

    def _get_memory_store_comp(self) -> Optional[StructuredMemoryComponent]:
        return self.element.get_component_by_type(StructuredMemoryComponent.COMPONENT_TYPE)
        
    def _get_container_comp(self) -> Optional[ContainerComponent]: # Keep for element traversal
        return self.element.get_component_by_type(ContainerComponent.COMPONENT_TYPE)

    # --- Core Context Orchestration Method --- 

    # Renamed from build_context, returns structured dict
    def get_orchestrated_context(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Gathers, filters/prioritizes, and assembles context from various sources 
        into a structured dictionary for the HUD.
        """
        logger.debug(f"[{self.element.id if self.element else '?'}] Orchestrating context...")
        self._state["last_build_time"] = time.time()
        options = options or {}

        # 1. Gather Information
        gathered_info = self._gather_information(options)

        # 2. Filter & Prioritize (Now implemented)
        prioritized_info = self._filter_and_prioritize(gathered_info)

        # 3. Assemble the final structured context
        orchestrated_context = self._assemble_structured_context(prioritized_info, options)

        logger.debug(f"[{self.element.id if self.element else '?'}] Context orchestration complete.")
        return orchestrated_context

    def _gather_information(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Gathers structured context from History, Representation, Memory, etc."""
        info = {
            "active_conversation_id": None,
            "active_history_turns": [],
            "self_representation": None,
            "memories": [],
            "mounted_elements_summary": [] # Simplified summary
        }
        options = options or {}

        # a) Get History for Active Conversation
        history_comp = self._get_history_comp()
        if history_comp:
            active_conv_id = history_comp.get_active_conversation_id()
            info["active_conversation_id"] = active_conv_id
            if active_conv_id:
                 max_turns = self._state.get("history_max_turns", 20)
                 history_entries = history_comp.get_history(
                     conversation_id=active_conv_id,
                     include_deleted=False, # Exclude deleted by default
                     sort_by_timestamp=True
                 )
                 # Apply limit and format turns
                 limited_entries = history_entries[-max_turns:]
                 formatted_turns = []
                 for entry in limited_entries:
                      entry_data = entry.get('data', {})
                      turn = {
                          "role": entry_data.get('role', 'unknown'),
                          "content": entry_data.get('text', ''),
                          "name": entry_data.get('sender_id') or entry_data.get('user_name'),
                          "timestamp": entry.get('timestamp'),
                          "message_id": entry.get('message_id'),
                          "tool_call_id": entry_data.get('tool_call_id'),
                          "edited_timestamp": entry.get('edited_timestamp'),
                          "reactions": entry.get('reactions', {})
                      }
                      turn = {k: v for k, v in turn.items() if v is not None and v != {}} # Clean Nones and empty reactions
                      formatted_turns.append(turn)
                 info["active_history_turns"] = formatted_turns

        # b) Get Representation of this Element (InnerSpace)
        rep_comp = self._get_representation_comp() # Get own representation
        if rep_comp:
             try:
                  info["self_representation"] = rep_comp.generate_representation(options)
                  # Remove redundant history turns from representation content if present
                  if isinstance(info["self_representation"], dict) and isinstance(info["self_representation"].get("content"), dict):
                      info["self_representation"]["content"].pop("turns", None)
                      info["self_representation"]["content"].pop("active_conversation_id", None)
             except Exception as e:
                  logger.exception(f"Error generating self-representation: {e}")
                  info["self_representation"] = {"error": "Failed to generate self-representation"}

        # c) Get Memories
        memory_comp = self._get_memory_store_comp()
        if memory_comp:
             try:
                  limit = self._state.get("max_memories_to_gather", 5)
                  # TODO: Add smarter query based on context (e.g., active_conv_id)
                  info["memories"] = memory_comp.retrieve_recent_memories(limit=limit)
             except Exception as e:
                  logger.exception(f"Error retrieving memories: {e}")
                  info["memories"] = [{"error": "Failed to retrieve memories"}]

        # d) Get Summary Representations of Mounted Elements
        container_comp = self._get_container_comp()
        if container_comp:
             mounted = container_comp.get_mounted_elements()
             for mount_id, element in mounted.items():
                  element_rep_comp = self._get_representation_comp(target_element=element)
                  summary = {"mount_id": mount_id, "element_id": element.id, "name": element.name}
                  if element_rep_comp:
                       try:
                            # Ask for a summarized representation?
                            opts = options.copy()
                            opts['detail_level'] = 'summary' # Hypothetical option
                            rep = element_rep_comp.generate_representation(opts)
                            # Extract key info for summary
                            summary["type"] = rep.get("element_type")
                            summary["content_summary"] = rep.get("content", {}).get("description") # Example
                            summary["attributes"] = rep.get("attributes")
                       except Exception as e:
                            logger.warning(f"Failed to generate representation for mounted element {element.id}: {e}")
                            summary["error"] = "Representation failed"
                  else:
                       summary["error"] = "No representation component"
                  info["mounted_elements_summary"].append(summary)

        return info

    # Implement filtering/prioritization logic
    def _filter_and_prioritize(self, raw_info: Dict[str, Any]) -> Dict[str, Any]:
        """Filters and potentially summarizes gathered information, especially older tool results."""
        prioritized_info = raw_info.copy() # Start with a copy
        history_turns = prioritized_info.get("active_history_turns", [])
        if not history_turns:
            return prioritized_info # No history to process

        processed_turns = []
        num_turns = len(history_turns)
        
        # Find the index of the last tool result, if any
        last_tool_result_index = -1
        for i in range(num_turns - 1, -1, -1):
             if history_turns[i].get("role") == "tool":
                  last_tool_result_index = i
                  break
                  
        for i, turn in enumerate(history_turns):
            if turn.get("role") == "tool":
                # Check if this is an older tool result (not the most recent one)
                is_older_tool_result = (i != last_tool_result_index)

                if is_older_tool_result:
                    # Apply placeholder summarization to older tool results
                    summarized_turn = turn.copy() # Modify a copy
                    tool_name = summarized_turn.get("tool_name", "unknown_tool")
                    tool_call_id = summarized_turn.get("tool_call_id", "unknown_id")
                    # Placeholder summary indicating it was processed
                    summary_text = f"[Previous result from tool '{tool_name}' (ID: {tool_call_id}): Processed]"
                    summarized_turn["content"] = summary_text
                    summarized_turn["_summarized"] = True # Add flag indicating change
                    processed_turns.append(summarized_turn)
                    logger.debug(f"Summarized older tool result: {tool_call_id}")
                else:
                    # Keep the most recent tool result as is
                    processed_turns.append(turn)
            else:
                # Keep non-tool turns as is
                processed_turns.append(turn)

        prioritized_info["active_history_turns"] = processed_turns
        return prioritized_info

    def _assemble_structured_context(self, context_info: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
         """Assembles the final structured dictionary for the HUD."""
         # For now, mainly pass through the gathered info
         # Could add system prompt here based on state?
         final_context = {
             "history_turns": context_info.get("active_history_turns", []),
             "active_conversation_id": context_info.get("active_conversation_id"),
             "self_summary": context_info.get("self_representation"), # Renamed for clarity?
             "memories": context_info.get("memories", []),
             "environment_summary": context_info.get("mounted_elements_summary", [])
             # Could add: current_time, agent_status, pending_actions etc.
         }
         return final_context

    # --- REMOVE OLD CONTEXT METHODS --- 
    # def build_context(...): ... (and its helpers _filter_and_prioritize, _format_intermediate, _compress_information)
    # def get_current_routing_context(...): ...
    # --- END REMOVED OLD METHODS --- 

    # Event handling (simplified)
    def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        event_type = event.get("event_type")
        if event_type == "rebuild_context_request":
             logger.info(f"[{self.element.id if self.element else '?'}] Received rebuild_context_request. Context will be rebuilt on next get.")
             return True
        return False

    # --- REMOVE find_chat_element_by_conversation --- 
    # def _find_chat_element_by_conversation(...): ...
    # --- END REMOVED METHOD --- 