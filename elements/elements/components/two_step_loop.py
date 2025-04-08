"""
Two-Step Agent Loop Component
Implements a Contemplation -> Action cycle.
"""

import logging
import time
from typing import Dict, Any, Optional, List

# Assuming base class is in .base_agent_loop
from .base_agent_loop import BaseAgentLoopComponent 
# Import necessary components for interaction
from .hud_component import HUDComponent
from .context_manager_component import ContextManagerComponent
from .tool_provider_component import ToolProviderComponent
# Import LLM response structure (adjust path as needed)
from ....llm.response import LLMResponse, LLMToolCall 
# Need element base for type hint
from ...base import BaseElement 

logger = logging.getLogger(__name__)

class TwoStepLoopComponent(BaseAgentLoopComponent):
    """
    Agent loop that performs a two-step process within each cycle:
    1. Contemplation: Internal thought, planning, memory update based on context.
    2. Action: Generating external tool calls or final messages based on contemplation.
    """
    COMPONENT_TYPE = "agent_loop.two_step"

    def __init__(self, element: BaseElement):
        super().__init__(element)
        # Configuration specific to the two-step loop could go here
        # e.g., specific prompts for each step
        # Note: Using \n directly in f-strings or multi-line strings is fine.
        self._contemplation_prompt_suffix = ("\n\nInternal Thought Process: Based on the current context, reflect on the situation, "
                                             "update your internal state or plan, and decide on the *type* of external action needed next (if any). "
                                             "Do NOT generate the final external action text or tool call yet. "
                                             "Example: 'Plan: Need to ask user for clarification.' or 'State Update: Added item X to memory.'")
        self._action_prompt_suffix = ("\n\nAction Generation: Based on the context and your internal thought process, "
                                      "generate the specific external tool call (using <tool_call> tags if applicable to the LLM) or the final message text to send.")

    async def run_cycle(self) -> None:
        """Executes the two-step contemplation and action cycle."""
        logger.info(f"[{self.element.id}] Starting TwoStep cycle...")

        # --- Get Dependencies ---
        hud: Optional[HUDComponent] = self._get_dependency("hud")
        context_manager: Optional[ContextManagerComponent] = self._get_dependency("context_manager")
        tool_provider: Optional[ToolProviderComponent] = self._get_dependency("tool_provider")

        if not all([hud, context_manager, tool_provider]):
            logger.error(f"[{self.element.id}] Cycle aborted: Missing dependencies (HUD, ContextManager, or ToolProvider).")
            return
        
        # Ensure LLM provider is available via HUD
        if not hud._llm_provider:
             logger.error(f"[{self.element.id}] Cycle aborted: HUDComponent has no LLMProvider.")
             return

        # --- Step 1: Contemplation ---
        logger.debug(f"[{self.element.id}] Entering Contemplation Step...")
        contemplation_response = await self._run_contemplation_step(hud, context_manager, tool_provider)
        
        internal_thought = "(No internal thought generated)" # Default value
        if contemplation_response and contemplation_response.content:
             internal_thought = contemplation_response.content.strip()
             # Add contemplation result to history
             context_manager.add_history_turn(role="internal_monologue", content=internal_thought)
             logger.info(f"[{self.element.id}] Internal Thought: {internal_thought[:150]}...")
             # TODO: Optionally parse contemplation_response for internal tool calls?
        else:
             logger.warning(f"[{self.element.id}] Contemplation step did not produce content. Recording default thought.")
             context_manager.add_history_turn(role="internal_monologue", content=internal_thought)

        # --- Step 2: Action Generation ---
        # The internal_thought is now part of the history used by build_context
        logger.debug(f"[{self.element.id}] Entering Action Step...")
        action_response = await self._run_action_step(hud, context_manager, tool_provider)

        if not action_response:
            logger.warning(f"[{self.element.id}] Action step failed or returned no response.")
            context_manager.add_history_turn(role="system", content="(Action generation step failed)")
        else:
            # Process the action response (tool calls or final message)
            # This reuses the processing logic which should also handle history addition
            self._process_action_response(action_response, context_manager, tool_provider)

        logger.info(f"[{self.element.id}] TwoStep cycle finished.")

    async def _run_contemplation_step(self, hud: HUDComponent, context_manager: ContextManagerComponent, tool_provider: ToolProviderComponent) -> Optional[LLMResponse]:
        """Performs the internal contemplation LLM call."""
        logger.debug(f"[{self.element.id}] Preparing context for contemplation...")
        
        # Build context *without* the contemplation suffix initially (it will be added)
        processed_context_str = context_manager.build_context()
        
        # Append the contemplation prompt suffix
        contemplation_context = processed_context_str + self._contemplation_prompt_suffix
        
        # Prepare messages using HUD's helper
        messages = hud._prepare_llm_messages_from_string(contemplation_context)
        if not messages: return None

        logger.info(f"[{self.element.id}] Calling LLM for Contemplation...")
        try:
            response = await hud._llm_provider.complete(
                 messages=messages,
                 tools=None, # No external tools for contemplation? Or maybe specific internal ones?
                 model=hud._state.get("model"),
                 temperature=hud._state.get("temperature"),
                 max_tokens=hud._state.get("max_tokens", 500) # Limit thought tokens?
            )
            if response: logger.debug(f"[{self.element.id}] Raw Contemplation Response: {response}")
            return response
        except Exception as e:
             logger.error(f"[{self.element.id}] Error during Contemplation LLM call: {e}", exc_info=True)
             return None

    async def _run_action_step(self, hud: HUDComponent, context_manager: ContextManagerComponent, tool_provider: ToolProviderComponent) -> Optional[LLMResponse]:
        """Performs the external action generation LLM call."""
        logger.debug(f"[{self.element.id}] Preparing context for action generation...")
        
        # Build context - this now includes the internal monologue from the previous step's history
        processed_context_str = context_manager.build_context()
        
        # Append the action prompt suffix
        action_context = processed_context_str + self._action_prompt_suffix

        # Prepare messages
        messages = hud._prepare_llm_messages_from_string(action_context)
        if not messages: return None

        # Get ALL available tools for the action step
        tool_schemas = tool_provider.get_llm_tool_schemas()

        logger.info(f"[{self.element.id}] Calling LLM for Action Generation...")
        try:
            response = await hud._llm_provider.complete(
                 messages=messages,
                 tools=tool_schemas, # Provide all external/aggregated tools
                 model=hud._state.get("model"),
                 temperature=hud._state.get("temperature"),
                 max_tokens=hud._state.get("max_tokens")
            )
            if response: logger.debug(f"[{self.element.id}] Raw Action Response: {response}")
            return response
        except Exception as e:
             logger.error(f"[{self.element.id}] Error during Action LLM call: {e}", exc_info=True)
             return None

    def _process_action_response(self, llm_response: LLMResponse, context_manager: ContextManagerComponent, tool_provider: ToolProviderComponent):
        """Processes the response from the action generation step."""
        logger.debug(f"[{self.element.id}] Processing action response...")

        # Add assistant action response (tool calls or content) to history
        history_content = {}
        if llm_response.content: history_content["content"] = llm_response.content
        if llm_response.tool_calls: history_content["tool_calls"] = [tc.to_dict() for tc in llm_response.tool_calls]
        if history_content: 
            context_manager.add_history_turn(role="assistant", content=history_content)

        # --- Execute Tool Calls or Publish Final Message ---
        tool_calls_to_execute: List[LLMToolCall] = llm_response.tool_calls or []
        final_message_content: Optional[str] = llm_response.content

        if tool_calls_to_execute:
            logger.info(f"[{self.element.id}] Executing {len(tool_calls_to_execute)} tool call(s) from action step...")
            for tool_call in tool_calls_to_execute:
                # Reuse the execution logic from base or simple loop (needs implementation access)
                self._execute_tool(tool_provider, tool_call.tool_name, tool_call.parameters)
        elif final_message_content:
            logger.info(f"[{self.element.id}] Publishing final message from action step...")
            # Reuse the publishing logic from base or simple loop (needs implementation access)
            self._publish_final_message(context_manager, final_message_content)
        else:
            logger.warning(f"[{self.element.id}] Action LLM response had no tool calls and no content.")
            context_manager.add_history_turn(role="system", content="(Action step produced no actionable output)")

    # --- Methods to be inherited or implemented --- 
    # We need _execute_tool and _publish_final_message. 
    # Let's assume they will be in the base class for now.

    # def _execute_tool(self, tool_provider: Optional[ToolProviderComponent], tool_name: str, tool_args: Dict[str, Any]):
    #     # Logic from SimpleRequestResponseLoopComponent or BaseAgentLoopComponent
    #     pass 

    # def _publish_final_message(self, context_manager: Optional[ContextManagerComponent], message_content: str):
    #     # Logic from SimpleRequestResponseLoopComponent or BaseAgentLoopComponent
    #     pass 