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
        hud: Optional['HUDComponent'] = self._get_dependency("hud")
        context_manager: Optional['ContextManagerComponent'] = self._get_dependency("context_manager")
        tool_provider: Optional['ToolProviderComponent'] = self._get_dependency("tool_provider")
        
        if not hud: 
             logger.error(f"[{self.element.id}] Cycle aborted: HUDComponent missing.")
             return
             
        # --- 1. Prepare Context and Call LLM (using HUD) --- 
        llm_response: Optional[LLMResponse] = await hud.prepare_and_call_llm()
        
        if not llm_response:
             logger.warning(f"[{self.element.id}] Cycle aborted: LLM call failed or returned no response.")
             return
             
        # --- 2. Add Assistant Response to History --- 
        # Do this early, regardless of whether it's a tool call or final message
        if context_manager and (llm_response.tool_calls or llm_response.content):
             # Storing the raw response object might be too verbose; extract key info.
             history_content = {}
             if llm_response.content: history_content["content"] = llm_response.content
             if llm_response.tool_calls: history_content["tool_calls"] = [tc.to_dict() for tc in llm_response.tool_calls]
             context_manager.add_history_turn(role="assistant", content=history_content)
        elif not context_manager:
             logger.warning(f"[{self.element.id}] ContextManager missing, cannot add assistant response to history.")
             
        # --- 3. Parse LLM Response for Tool Calls --- 
        tool_calls_to_execute: List[LLMToolCall] = llm_response.tool_calls or []
        final_message_content: Optional[str] = llm_response.content
        
        # --- 4. Execute Tool Calls or Publish Final Message --- 
        if tool_calls_to_execute:
             logger.info(f"[{self.element.id}] Executing {len(tool_calls_to_execute)} tool call(s)...")
             # TODO: Handle multiple tool calls concurrently? For simple loop, maybe one by one.
             for tool_call in tool_calls_to_execute:
                 # Note: LLMResponse uses LLMToolCall which has name and parameters
                 self._execute_tool(tool_provider, tool_call.tool_name, tool_call.parameters)
                 # TODO: Add tool results back to history? Requires tool_call_id from LLM.
        elif final_message_content:
             logger.info(f"[{self.element.id}] Publishing final message...")
             # Pass ContextManager to get routing info
             self._publish_final_message(context_manager, final_message_content)
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