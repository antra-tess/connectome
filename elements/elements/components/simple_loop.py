"""
Simple Request-Response Agent Loop Component
"""

import logging
import asyncio
import json # For parsing tool args
import time # For timestamp
from typing import Dict, Any, Optional, Callable, Tuple, List

# Base class and core imports
from .base_component import Component
from .agent_loop import BaseAgentLoopComponent # Import base class

# Type Hinting for dependencies
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..inner_space import InnerSpace
    from ....llm.provider_interface import LLMProviderInterface
    from .tool_provider_component import ToolProviderComponent
    from .hud_component import HUDComponent
    from .context_manager_component import ContextManagerComponent
    from ....host.event_loop import OutgoingActionCallback
    from ....llm.response import LLMResponse, LLMToolCall

logger = logging.getLogger(__name__)

class SimpleRequestResponseLoopComponent(BaseAgentLoopComponent):
    """
    A basic loop: Prepare context, call LLM once, parse actions/message, execute.
    Inherits action request handling (like memory processing) from BaseAgentLoopComponent.
    """
    COMPONENT_TYPE = "agent_loop.simple" # Specific type

    # __init__ is inherited from BaseAgentLoopComponent
    # set_outgoing_action_callback is inherited
    # _get_dependency is inherited
    # _call_llm is inherited
    # handle_action_request is inherited
    # _handle_memory_processing_request is inherited
    # _execute_tool is inherited
    # _publish_final_message is inherited

    async def run_cycle(self) -> None:
        logger.info(f"[{self.element.id}] Starting SimpleRequestResponse cycle...")
        
        # Get dependencies safely within the cycle
        # hud: Optional['HUDComponent'] = self._get_dependency("hud") 
        # context_manager: Optional['ContextManagerComponent'] = self._get_dependency("context_manager") # Removed
        # tool_provider: Optional['ToolProviderComponent'] = self._get_dependency("tool_provider")
        # Let's use the direct element accessors assuming they exist
        hud = self.element.get_component_by_type("hud")
        tool_provider = self.element.get_component_by_type("tool_provider") # ToolProviderComponent.COMPONENT_TYPE
        history_comp = self.element.get_component_by_type("history") # HistoryComponent.COMPONENT_TYPE - Needed for context
        
        if not hud: 
             logger.error(f"[{self.element.id}] Cycle aborted: HUDComponent missing.")
             return
        # Tool provider is needed for tool execution
        # History component is needed for routing context
        if not tool_provider:
             logger.warning(f"[{self.element.id}] ToolProviderComponent missing. Tool execution will fail.")
        if not history_comp:
             logger.warning(f"[{self.element.id}] HistoryComponent missing. Routing context for actions might be unavailable.")
             
        # --- 1. Prepare Context and Call LLM (using HUD) --- 
        llm_response: Optional[LLMResponse] = await hud.prepare_and_call_llm()
        
        if not llm_response:
             logger.warning(f"[{self.element.id}] Cycle aborted: LLM call failed or returned no response.")
             return
             
        # --- 2. Add Assistant Response to History --- 
        # REMOVED: Direct call to context_manager.add_history_turn
        # HistoryComponent will handle this based on outgoing action requests/events.
        # if context_manager and (llm_response.tool_calls or llm_response.content): ...
        # elif not context_manager: ...
             
        # --- 3. Parse LLM Response for Tool Calls --- 
        tool_calls_to_execute: List[LLMToolCall] = llm_response.tool_calls or []
        final_message_content: Optional[str] = llm_response.content
        
        # --- 4. Execute Tool Calls or Publish Final Message --- 
        
        # Get context for actions (needed for both tools and messages)
        adapter_id: Optional[str] = None
        conversation_id: Optional[str] = None
        if history_comp:
             conversation_id = history_comp.get_active_conversation_id()
             # How to get adapter_id? Assume it's stored in history entry data? Or element state?
             # For now, we might need to fetch it from the last message in the active conversation.
             if conversation_id:
                  active_history = history_comp.get_history(conversation_id)
                  if active_history:
                       last_entry_data = active_history[-1].get('data', {})
                       adapter_id = last_entry_data.get('adapter_id') 
             
        if not adapter_id or not conversation_id:
             logger.warning(f"[{self.element.id}] Could not determine adapter/conversation ID for outgoing actions from HistoryComponent.")
             # Decide if we should abort or proceed without full context?
             # Let's proceed but actions might fail.
             
        action_context = {"adapter_id": adapter_id, "conversation_id": conversation_id}
        
        if tool_calls_to_execute:
             logger.info(f"[{self.element.id}] Executing {len(tool_calls_to_execute)} tool call(s)...")
             for tool_call in tool_calls_to_execute:
                 # Pass action_context to _execute_tool
                 self._execute_tool(tool_provider, tool_call.tool_name, tool_call.parameters, action_context)
        elif final_message_content:
             logger.info(f"[{self.element.id}] Publishing final message...")
             # Pass action_context instead of context_manager
             self._publish_final_message(action_context, final_message_content)
        else:
             logger.warning(f"[{self.element.id}] LLM response had no tool calls and no content.")
             
        logger.info(f"[{self.element.id}] SimpleRequestResponse cycle finished.")

    # Removed handle_action_request
    # Removed _handle_memory_processing_request

    # _execute_tool is now inherited from BaseAgentLoopComponent
    # def _execute_tool(self, tool_provider: Optional['ToolProviderComponent'], tool_name: str, tool_args: Dict[str, Any]):
    #     ...

    # _publish_final_message is now inherited from BaseAgentLoopComponent
    # def _publish_final_message(self, context_manager: Optional['ContextManagerComponent'], message_content: str):
    #    ... 