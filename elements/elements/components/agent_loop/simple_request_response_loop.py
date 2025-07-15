"""
Simple Request Response Loop Component
Implements a basic agent loop with memory integration.
"""
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

from elements.component_registry import register_component
from .base_agent_loop_component import BaseAgentLoopComponent
from .utils import create_multimodal_llm_message

# NEW: Import LLMToolCall for tool processing
from llm.provider_interface import LLMToolCall

if TYPE_CHECKING:
    from ...inner_space import InnerSpace

logger = logging.getLogger(__name__)


@register_component
class SimpleRequestResponseLoopComponent(BaseAgentLoopComponent):
    """
    A basic agent loop with memory integration that:
    1. Gets memory context from CompressionEngine (conversation history)
    2. Gets fresh context from HUD (current frame)
    3. Combines and sends to LLM with available tools
    4. Processes LLM response (tool calls + text)
    5. Stores complete reasoning chain back to CompressionEngine

    This provides simple request-response behavior while maintaining full memory continuity.
    """
    COMPONENT_TYPE = "SimpleRequestResponseLoopComponent"

    def __init__(self, parent_inner_space: 'InnerSpace', agent_loop_name: Optional[str] = None, **kwargs):
        super().__init__(parent_inner_space=parent_inner_space, agent_loop_name=agent_loop_name, **kwargs)
        logger.info(f"SimpleRequestResponseLoopComponent with memory initialized for '{self.parent_inner_space.name}'")

    async def trigger_cycle(self, focus_context: Optional[Dict[str, Any]] = None):
        logger.info(f"{self.agent_loop_name} ({self.id}): Simple cycle with memory triggered in InnerSpace '{self.parent_inner_space.name}'.")

        # Get required components
        hud = self._get_hud()
        llm_provider = self._get_llm_provider()
        compression_engine = self._get_compression_engine()

        if not hud or not llm_provider:
            logger.error(f"{self.agent_loop_name} ({self.id}): Missing critical components (HUD, LLM). Aborting cycle.")
            return

        if not compression_engine:
            logger.warning(f"{self.agent_loop_name} ({self.id}): CompressionEngine not available. Proceeding without memory.")

        try:
            # NEW: Get enhanced tool definitions from VEIL for Phase 2 integration
            enhanced_tools_from_veil = self._extract_enhanced_tools_from_veil()
            
            pipeline_options = {
                'focus_context': focus_context,
                'include_memory': True,
                'render_style': 'chronological_flat',
                'tool_rendering_mode': 'simple'  # Use simple mode for existing tool_call API
            }
            context_data = await hud.get_agent_context_via_compression_engine(
                options=pipeline_options,
                tools=enhanced_tools_from_veil
            )
            logger.info(f"Using unified CompressionEngine pipeline for context generation")

            if not context_data:
                logger.warning(f"{self.agent_loop_name} ({self.id}): No context data received. Aborting cycle.")
                return

            # --- NEW: Process Turn-Based Context from HUD ---
            self._log_context_format(context_data)
            messages = self._process_context_to_messages(context_data)
            
            # Log message details
            if self._is_turn_based_context(context_data):
                logger.info(f"Built {len(messages)} turn-based messages for LLM")
            elif self._is_multimodal_turn_based_context(context_data):
                multimodal_info = context_data.get("multimodal_content", {})
                attachment_count = multimodal_info.get("attachment_count", 0)
                logger.info(f"Built {len(messages)} messages with multimodal turn-based content: {attachment_count} attachments")
            else:
                logger.warning(f"Processing legacy context format - this should not happen with new HUD")

            # Log final message details for debugging
            for i, msg in enumerate(messages):
                if msg.is_multimodal():
                    attachment_count = msg.get_attachment_count()
                    text_length = len(msg.get_text_content())
                    logger.debug(f"Message {i} ({msg.role}): {text_length} chars text + {attachment_count} attachments")
                else:
                    logger.debug(f"Message {i} ({msg.role}): {len(msg.get_text_content())} chars")

            # Get aggregated tools and send to LLM
            aggregated_tools = await self.aggregate_tools()
            # Pass original context data for scaffolding provider to preserve turn metadata
            llm_response_obj = llm_provider.complete(messages=messages, tools=aggregated_tools, original_context_data=context_data)

            if not llm_response_obj:
                logger.warning(f"{self.agent_loop_name} ({self.id}): LLM returned no response. Aborting cycle.")
                return

            agent_response_text = llm_response_obj.content
            agent_tool_calls = llm_response_obj.tool_calls or []
            logger.critical(f"LLM RESPONSE: {agent_response_text}")
            logger.critical(f"LLM TOOL CALLS: {agent_tool_calls}")

            logger.info(f"LLM response: {len(agent_response_text or '')} chars, {len(agent_tool_calls)} tool calls")

            # --- NEW: Emit Agent Response Delta for VEIL ---
            await self._emit_agent_response_delta(agent_response_text, agent_tool_calls)

            # --- Process Tool Calls (unchanged - still essential) ---
            tool_results = []
            if agent_tool_calls:
                logger.info(f"Processing {len(agent_tool_calls)} tool calls...")
                for tool_call in agent_tool_calls:
                    if not isinstance(tool_call, LLMToolCall):
                        continue

                    # Parse tool target and name
                    raw_tool_name = tool_call.tool_name
                    target_element_id = None
                    actual_tool_name = raw_tool_name

                    if "__" in raw_tool_name:
                        # Handle prefixed tools (e.g., "smith_a1b2__send_message")
                        parts = raw_tool_name.split("__", 1)
                        prefix = parts[0]
                        actual_tool_name = parts[1]

                        # Resolve the prefix to the full element ID
                        target_element_id = self._resolve_prefix_to_element_id(prefix)
                        if not target_element_id:
                            logger.error(f"Could not resolve tool prefix '{prefix}' to element ID. Skipping.")
                            continue

                        # Validate the target element exists
                        target_element = self.parent_inner_space.get_element_by_id(target_element_id)
                        if not target_element:
                            logger.error(f"Tool target element '{target_element_id}' not found. Skipping.")
                            continue
                    else:
                        # For non-prefixed tools, try to find the appropriate target element
                        # Look through mounted elements to find one that has this tool
                        target_element_id = self._find_element_with_tool(actual_tool_name)
                        if not target_element_id:
                            # Fallback to InnerSpace itself
                            target_element_id = self.parent_inner_space.id
                            logger.debug(f"No specific element found for tool '{actual_tool_name}', using InnerSpace")

                    # Execute tool
                    try:
                        calling_context = {"loop_component_id": self.id}
                        tool_result = await self.parent_inner_space.execute_action_on_element(
                            element_id=target_element_id,
                            action_name=actual_tool_name,
                            parameters=tool_call.parameters,
                            calling_context=calling_context
                        )
                        tool_results.append({
                            "tool_name": raw_tool_name,
                            "parameters": tool_call.parameters,
                            "result": tool_result
                        })
                        logger.debug(f"Tool '{actual_tool_name}' executed successfully on element '{target_element_id}'")
                    except Exception as e:
                        logger.error(f"Error executing tool '{actual_tool_name}': {e}", exc_info=True)
                        tool_results.append({
                            "tool_name": raw_tool_name,
                            "parameters": tool_call.parameters,
                            "result": {"error": str(e)}
                        })

            # --- Process Text Response (always, regardless of tool calls) ---
            if agent_response_text:
                logger.debug(f"Processing text response ({len(agent_tool_calls)} tool calls also present)...")

                # NEW: Smart chat fallback - if agent has text, try to send to activating chat
                # This keeps conversations flowing even when agent uses other tools
                fallback_sent = await self._try_smart_chat_fallback(agent_response_text, focus_context)

                if not fallback_sent:
                    # Original HUD processing as fallback to the fallback
                    processed_actions = await hud.process_llm_response(agent_response_text)
                    final_action_requests = processed_actions
                    logger.info(f"Dispatching {len(final_action_requests)} final action(s) from text response.")
                    for action_request in final_action_requests:
                         target_module = action_request.get("target_module")
                         if target_module and self._get_outgoing_action_callback():
                              try:
                                  await self._get_outgoing_action_callback()(action_request)
                              except Exception as e:
                                  logger.error(f"Error dispatching final external action: {e}")

        except Exception as e:
            logger.error(f"{self.agent_loop_name} ({self.id}): Error during simple cycle: {e}", exc_info=True)
        finally:
            logger.info(f"{self.agent_loop_name} ({self.id}): Simple cycle with memory completed.") 