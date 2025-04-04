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
        "attention_focus_changed"  # From GlobalAttentionComponent
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

    def _format_intermediate(self, prioritized_items: List[Tuple[int, str, Dict[str, Any]]]) -> List[str]:
        """Formats prioritized history and element items into structured text blocks."""
        logger.debug("Formatting intermediate context representation...")
        formatted_blocks = []
        max_items_per_element = self._state.get("max_items_per_element", 10)

        for priority, item_type, data in prioritized_items:
            block = ""
            # --- Format History Turn --- 
            if item_type == "history":
                 role = data.get("role", "unknown")
                 content = data.get("content", "")
                 name = data.get("name")
                 tool_call_id = data.get("tool_call_id")
                 
                 block += f"[{role.upper()}]
" 
                 if name: block += f"(Name: {name})
" 
                 if tool_call_id: block += f"(Tool Call ID: {tool_call_id})
" 
                 
                 # Simple stringification for now
                 if isinstance(content, dict) or isinstance(content, list):
                      try:
                           block += json.dumps(content, indent=1) + "\n"
                      except TypeError:
                           block += str(content) + "\n"
                 else:
                      block += str(content) + "\n"
                      
            # --- Format Element Data --- 
            elif item_type == "element":
                 element_data = data
                 block += f"### Element: {element_data.get('name', 'Unknown')} (ID: {element_data.get('id')}, Type: {element_data.get('type')}) ###\n"
                 # Add VEIL representation 
                 veil_data = element_data.get("veil")
                 if veil_data:
                      try:
                           block += f"Representation (VEIL):
{json.dumps(veil_data, indent=1, default=str)[:1000]}...\n"
                      except Exception:
                           block += f"Representation (VEIL): {str(veil_data)[:1000]}...\n"
                 # Add Uplink Cache
                 uplink_cache = element_data.get("uplink_cache")
                 if uplink_cache:
                      block += f"Remote Cache:
" 
                      items_shown = 0
                      for key, value in uplink_cache.items():
                           if items_shown >= max_items_per_element: block += "  ... (more truncated)\n"; break
                           block += f"  - {key}: {str(value)[:100]}...\n"
                           items_shown += 1
            
            if block:
                formatted_blocks.append(block)
            
        return formatted_blocks

    def _compress_information(self, formatted_items: List[str]) -> str:
        """Applies compression strategy to fit token budget (processes history and elements)."""
        strategy = self._state.get("compression_strategy", "truncate_recent")
        budget = self._state.get("token_budget", 4000)
        logger.debug(f"Compressing {len(formatted_items)} formatted blocks using strategy: {strategy} (Budget: {budget} tokens)...")

        final_context = ""
        current_tokens = 0
        
        # Simple truncate_recent: Add blocks (history+elements) in priority order
        if strategy == "truncate_recent":
             final_context = self._compress_information_truncate(formatted_items, budget)
        elif strategy == "summarize_old":
             # TODO: Implement summarization - needs to identify history vs element blocks
             logger.warning("'summarize_old' compression strategy not yet implemented.")
             final_context = self._compress_information_truncate(formatted_items, budget)
        else:
             logger.warning(f"Unknown compression strategy '{strategy}'. Falling back to truncate.")
             final_context = self._compress_information_truncate(formatted_items, budget)

        return final_context
        
    def _compress_information_truncate(self, formatted_items: List[str], budget: int) -> str:
         final_context = ""
         current_tokens = 0
         for item_str in formatted_items:
             item_tokens = estimate_tokens(item_str)
             if current_tokens + item_tokens <= budget:
                 final_context += item_str + "\n"
                 current_tokens += item_tokens
             else:
                 remaining_budget = budget - current_tokens
                 if remaining_budget > 20:
                     cutoff = remaining_budget * 4
                     final_context += item_str[:cutoff] + "... (truncated)\n"
                 break
         return final_context

    # --- Event Handling --- 

    def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handle events that might trigger a context rebuild.
        """
        event_type = event.get("event_type")
        
        # Explicit request
        if event_type == "rebuild_context_request":
            logger.info("Received explicit request to rebuild context.")
            # In a real system, might queue this or handle async
            self.build_context()
            return True # Event was handled

        # Implicit triggers (more complex)
        # TODO: Decide if/how state changes or attention shifts automatically trigger rebuilds.
        # This can be inefficient if not managed carefully (e.g., debouncing).
        # if event_type == "element_state_changed" or event_type == "attention_focus_changed":
        #    logger.debug(f"Potential context trigger event received: {event_type}")
        #    # Trigger a rebuild, perhaps debounced?
        #    pass 
            
        return False # Event not fully handled for context rebuild yet 