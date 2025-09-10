"""
Tool Text Parsing Agent Loop Component

This module implements a text-based tool parsing approach for agent loops,
moving away from the LLM tool_call API to enable greater control and broader
LLM compatibility.

Phase 3 of the Tool Use Refactor.
"""

import logging
import json
import re
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import time

from elements.component_registry import register_component
from .base_agent_loop_component import BaseAgentLoopComponent
from .utils import create_multimodal_llm_message

logger = logging.getLogger(__name__)

@dataclass
class ParsedToolCall:
    """Represents a parsed tool call from LLM text response."""
    tool_name: str
    parameters: Dict[str, Any]
    target_element_name: Optional[str] = None
    raw_text: str = ""

@register_component
class ToolTextParsingLoopComponent(BaseAgentLoopComponent):
    """
    NEW: Agent loop component that uses text-based tool call parsing.

    This component moves away from the LLM tool_call API to a text-parsing approach,
    enabling:
    - Multiple tool calls per response
    - Broader LLM compatibility
    - Greater control over tool call format
    - Aggregated tool definitions with target parameters

    The component:
    1. Aggregates tools and builds element name mapping
    2. Sends aggregated tools + context to HUD for full rendering
    3. Parses tool calls from LLM text response
    4. Resolves element names to element_ids
    5. Executes tool calls sequentially or in parallel
    6. Emits agent response delta for chronological rendering
    """

    COMPONENT_TYPE = "ToolTextParsingLoopComponent"

    def __init__(self, parent_inner_space: 'InnerSpace', agent_loop_name: Optional[str] = None, **kwargs):
        super().__init__(parent_inner_space=parent_inner_space, agent_loop_name=agent_loop_name, **kwargs)
        # Cache for element name to element_id mapping
        self._element_name_to_id_mapping: Dict[str, str] = {}
        logger.info(f"ToolTextParsingLoopComponent initialized for '{self.parent_inner_space.name}' (ultra-concise XML tool call format)")

    async def trigger_cycle(self, focus_context: Optional[Dict[str, Any]] = None):
        """
        Main cycle for text-based tool parsing agent loop.

        Args:
            focus_context: Optional context about the focused element
        """
        logger.info(f"{self.agent_loop_name} ({self.id}): Text-parsing cycle triggered in InnerSpace '{self.parent_inner_space.name}'.")

        # Get required components
        hud = self._get_hud()
        llm_provider = self._get_llm_provider()
        compression_engine = self._get_compression_engine()

        if not hud or not llm_provider:
            logger.error(f"{self.agent_loop_name} ({self.id}): Missing critical components (HUD, LLM). Aborting cycle.")
            return

        if not compression_engine:
            logger.warning(f"{self.agent_loop_name} ({self.id}): CompressionEngine not available. Proceeding without memory.")

        # Streaming lifecycle: mark start
        try:
            await self._emit_stream_lifecycle_event("agent_response_started", {
                "response_id": f"resp_{int(time.time()*1000)}",
                "loop_component_id": self.id,
                "timestamp": time.time()
            })
        except Exception:
            pass

        try:
            # 1. Aggregate tools and build element name mapping
            aggregated_tools = await self.aggregate_tools()
            enhanced_tools_from_veil = self._extract_enhanced_tools_from_veil()
            logger.debug(f"Enhanced tools from VEIL: {len(enhanced_tools_from_veil)} tools")
            for i, tool in enumerate(enhanced_tools_from_veil[:3]):  # Log first 3 tools for debugging
                logger.debug(f"Enhanced tool {i}: keys={list(tool.keys()) if isinstance(tool, dict) else 'not_dict'}")
            self._build_element_name_mapping(enhanced_tools_from_veil)

            # 2. Send tools + context to HUD for rendering with aggregation
            pipeline_options = {
                'focus_context': focus_context,
                'include_memory': True,
                'render_style': 'chronological_flat',
                'tool_rendering_mode': 'full'  # Use full mode for text parsing
            }
            context_data = None
            # Prefer HUD snapshot if focus provided
            try:
                focus_element_id = (focus_context or {}).get('focus_element_id') if isinstance(focus_context, dict) else None
                if focus_element_id and hasattr(hud, 'get_snapshot'):
                    snap = hud.get_snapshot(focus_element_id, max_age_ms=5000, generate_if_missing=False)
                    if not snap and hasattr(hud, 'refresh_snapshot'):
                        try:
                            await hud.refresh_snapshot(focus_element_id)
                            snap = hud.get_snapshot(focus_element_id, max_age_ms=5000, generate_if_missing=False)
                        except Exception:
                            pass
                    # Unwrap snapshot wrapper to messages if present
                    if isinstance(snap, dict) and 'messages' in snap:
                        context_data = snap.get('messages')
                    elif snap is not None:
                        context_data = snap
            except Exception:
                pass

            # Fallback to live rendering if no snapshot available
            if context_data is None:
                context_data = await hud.get_agent_context_via_compression_engine(
                    options=pipeline_options,
                    tools=enhanced_tools_from_veil
                )
            logger.info(f"Using full tool rendering mode for ultra-concise XML text parsing")

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

            # 3. Cache context for inspector before sending to LLM
            try:
                cache_key = hud.cache_llm_context(messages, {
                    'agent_loop_type': 'ToolTextParsingLoop',
                    'message_count': len(messages),
                    'has_tools': len(enhanced_tools_from_veil) > 0
                })
                logger.debug(f"Cached LLM context for inspector with key: {cache_key}")
            except Exception as cache_error:
                logger.warning(f"Failed to cache LLM context: {cache_error}")

            # 4. Send rendered context to LLM (no separate tool definitions - they're in the context)
            # NOTE: We don't pass tools parameter to LLM since they're already rendered in the context
            # Metadata now travels with LLMMessage objects, no need for original_context_data
            if self._is_cancelled():
                raise Exception("Agent loop preempted before LLM call")
            llm_response_obj = llm_provider.complete(messages=messages, tools=[])
            if self._is_cancelled():
                raise Exception("Agent loop preempted after LLM call")

            if not llm_response_obj:
                logger.warning(f"{self.agent_loop_name} ({self.id}): LLM returned no response. Aborting cycle.")
                return

            agent_response_text = llm_response_obj.content or ""
            logger.info(f"LLM response: {len(agent_response_text)} chars")

            # Emit finalization lifecycle event
            try:
                await self._emit_stream_lifecycle_event("agent_response_finalized", {
                    "response_id": f"resp_{int(time.time()*1000)}",
                    "final_text": agent_response_text,
                    "total_chars": len(agent_response_text),
                    "loop_component_id": self.id,
                    "timestamp": time.time()
                })
            except Exception:
                pass

            # 4. Parse tool calls from text response
            parsed_tool_calls = self._parse_tool_calls_from_response(agent_response_text)
            logger.info(f"Parsed {len(parsed_tool_calls)} tool calls from response")

            # 5. Resolve element names to element_ids
            resolved_tool_calls = self._resolve_target_element_names(parsed_tool_calls)

            # --- Emit Agent Response Delta for VEIL ---
            await self._emit_agent_response_delta(agent_response_text, resolved_tool_calls)

            # 6. Execute tool calls sequentially
            tool_results = []
            if resolved_tool_calls:
                logger.info(f"Executing {len(resolved_tool_calls)} resolved tool calls...")
                for tool_call in resolved_tool_calls:
                    try:
                        tool_result = await self._execute_parsed_tool_call(tool_call)
                        tool_results.append(tool_result)
                        logger.debug(f"Tool '{tool_call.tool_name}' executed successfully")
                    except Exception as e:
                        logger.error(f"Error executing tool '{tool_call.tool_name}': {e}", exc_info=True)
                        tool_results.append({
                            "tool_name": tool_call.tool_name,
                            "parameters": tool_call.parameters,
                            "result": {"error": str(e)}
                        })

            # 7. Handle non-tool conversational text as fallback response
            non_tool_text = self._extract_non_tool_text(agent_response_text)
            if non_tool_text.strip() and focus_context:
                await self._send_conversational_response(non_tool_text, focus_context)
        except Exception as e:
            # Emit interruption lifecycle if cancellation was requested
            if self._is_cancelled():
                try:
                    await self._emit_stream_lifecycle_event("agent_response_interrupted", {
                        "response_id": f"resp_{int(time.time()*1000)}",
                        "final_text": "",
                        "total_chars": 0,
                        "interrupted_by_event_id": None,
                        "loop_component_id": self.id,
                        "timestamp": time.time()
                    })
                except Exception:
                    pass
                self._reset_cancel()
            logger.error(f"{self.agent_loop_name} ({self.id}): Error during text-parsing cycle: {e}", exc_info=True)
        finally:
            logger.info(f"{self.agent_loop_name} ({self.id}): Text-parsing cycle completed.")

    def _build_element_name_mapping(self, enhanced_tools: List[Dict[str, Any]]):
        """
        Build mapping from element names to element_ids for tool call resolution.

        This enables the agent to use human-readable element names in tool calls
        which are then resolved back to element_ids for execution.

        Args:
            enhanced_tools: Enhanced tool definitions from VEIL with element metadata
        """
        try:
            self._element_name_to_id_mapping.clear()
            logger.debug(f"Building element name mapping from {len(enhanced_tools)} enhanced tools")

            for i, tool_def in enumerate(enhanced_tools):
                element_id = tool_def.get("target_element_id")
                element_name = tool_def.get("element_name")
                tool_name = tool_def.get("name", "unknown")

                logger.debug(f"Tool {i} ({tool_name}): element_id='{element_id}', element_name='{element_name}', keys={list(tool_def.keys())}")

                if element_id and element_name:
                    # Avoid overwriting if multiple tools from same element
                    if element_name not in self._element_name_to_id_mapping:
                        self._element_name_to_id_mapping[element_name] = element_id
                        logger.debug(f"Mapped element name '{element_name}' -> '{element_id}'")
                    else:
                        logger.debug(f"Element name '{element_name}' already mapped to '{self._element_name_to_id_mapping[element_name]}'")
                else:
                    logger.warning(f"Tool {tool_name} missing required metadata: element_id='{element_id}', element_name='{element_name}'")

            logger.debug(f"Built element name mapping with {len(self._element_name_to_id_mapping)} entries: {self._element_name_to_id_mapping}")

        except Exception as e:
            logger.error(f"Error building element name mapping: {e}", exc_info=True)

    def _parse_tool_calls_from_response(self, response_text: str) -> List[ParsedToolCall]:
        """
        Parse tool calls from LLM response text.

        Primary format is ultra-concise XML (tool names as elements), with JSON as fallback.

        Args:
            response_text: Raw text response from LLM

        Returns:
            List of ParsedToolCall objects
        """
        try:
            parsed_calls = []

            # First try XML format (Primary format - as instructed by HUD)
            xml_calls = self._parse_xml_tool_calls(response_text)
            parsed_calls.extend(xml_calls)

            # Fallback to JSON blocks format for backward compatibility
            if not xml_calls:
                json_calls = self._parse_json_tool_calls(response_text)
                parsed_calls.extend(json_calls)

            logger.debug(f"Parsed {len(parsed_calls)} tool calls from response (XML: {len(xml_calls)}, JSON: {len(parsed_calls) - len(xml_calls)})")
            return parsed_calls

        except Exception as e:
            logger.error(f"Error parsing tool calls from response: {e}", exc_info=True)
            return []

    def _parse_json_tool_calls(self, response_text: str) -> List[ParsedToolCall]:
        """
        Parse JSON-format tool calls from response (FALLBACK FORMAT).

        Looks for <tool_calls> JSON blocks with format:
        [{"tool": "name", "parameters": {...}}]
        Only used when XML parsing yields no results.
        """
        try:
            parsed_calls = []

            # Find <tool_calls> blocks
            tool_calls_pattern = r'<tool_calls>\s*(\[.*?\])\s*</tool_calls>'
            matches = re.findall(tool_calls_pattern, response_text, re.DOTALL)

            for match in matches:
                try:
                    tool_calls_data = json.loads(match)
                    if isinstance(tool_calls_data, list):
                        for tool_call_data in tool_calls_data:
                            if isinstance(tool_call_data, dict):
                                # Support both "tool" and "name" for compatibility
                                tool_name = tool_call_data.get("tool") or tool_call_data.get("name", "")
                                parameters = tool_call_data.get("parameters", {})

                                if tool_name:
                                    parsed_call = ParsedToolCall(
                                        tool_name=tool_name,
                                        parameters=parameters,
                                        target_element_name=parameters.get("source"),
                                        raw_text=match
                                    )
                                    parsed_calls.append(parsed_call)
                                    logger.info(f"Parsed JSON tool call: {tool_name} with params: {parameters}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON tool call block: {e}")
                    continue

            return parsed_calls

        except Exception as e:
            logger.error(f"Error parsing JSON tool calls: {e}", exc_info=True)
            return []

    def _parse_xml_tool_calls(self, response_text: str) -> List[ParsedToolCall]:
        """
        Parse XML-format tool calls from response (PRIMARY FORMAT).

        Looks for <tool_calls> blocks with ultra-concise format:
        <tool_calls>
        <tool_name param1="value1" source="element_name">
        </tool_calls>
        """
        try:
            parsed_calls = []

            element_pattern = r'<([a-zA-Z_][a-zA-Z0-9_]*)\s*([^>]*)>(.*?)</\1>'
            element_matches = re.findall(element_pattern, response_text, re.DOTALL)

            for tool_name, attributes_str, inner_content in element_matches:
                try:
                    # Parse attributes from the attributes string
                    parameters = {}

                    # Add the inner content to the parameters (if not empty)
                    inner_content = inner_content.strip()
                    if inner_content:
                        parameters["inner_content"] = inner_content

                    # Extract all attribute="value" pairs
                    attr_pattern = r'(\w+)="([^"]*)"'
                    attr_matches = re.findall(attr_pattern, attributes_str)

                    for attr_name, attr_value in attr_matches:
                        # Try to parse as JSON/number, fall back to string
                        try:
                            # Try parsing as number first
                            if attr_value.isdigit():
                                parameters[attr_name] = int(attr_value)
                            elif attr_value.replace('.', '').replace('-', '').isdigit():
                                parameters[attr_name] = float(attr_value)
                            # Try parsing as JSON for complex types
                            elif attr_value.startswith(('[', '{')):
                                parameters[attr_name] = json.loads(attr_value)
                            else:
                                parameters[attr_name] = attr_value
                        except:
                            parameters[attr_name] = attr_value

                    if tool_name:
                        parsed_call = ParsedToolCall(
                            tool_name=tool_name,
                            parameters=parameters,
                            target_element_name=parameters.get("source"),
                            raw_text=f'<{tool_name} {attributes_str}>'
                        )
                        parsed_calls.append(parsed_call)
                        logger.info(f"Parsed ultra-concise XML tool call: {tool_name} with params: {list(parameters.keys())}")

                except Exception as e:
                    logger.warning(f"Failed to parse ultra-concise XML tool call '{tool_name}': {e}")
                    continue

            return parsed_calls

        except Exception as e:
            logger.error(f"Error parsing ultra-concise XML tool calls: {e}", exc_info=True)
            return []

    def _resolve_target_element_names(self, tool_calls: List[ParsedToolCall]) -> List[ParsedToolCall]:
        """
        Convert target_element names back to element_ids for execution.

        Args:
            tool_calls: List of parsed tool calls with element names

        Returns:
            List of tool calls with resolved element_ids
        """
        try:
            resolved_calls = []

            for tool_call in tool_calls:
                resolved_call = ParsedToolCall(
                    tool_name=tool_call.tool_name,
                    parameters=tool_call.parameters.copy(),
                    target_element_name=tool_call.target_element_name,
                    raw_text=tool_call.raw_text
                )
                # Resolve target_element name to element_id
                if tool_call.target_element_name:
                    # FIXED: Always remove target_element parameter to prevent it from being passed to tools
                    if "source" in resolved_call.parameters:
                        logger.debug(f"Removing 'source' parameter from {tool_call.tool_name} tool call. Before: {resolved_call.parameters}")
                        del resolved_call.parameters["source"]
                        logger.debug(f"After removal: {resolved_call.parameters}")

                    element_id = self._element_name_to_id_mapping.get(tool_call.target_element_name)
                    if element_id:
                        resolved_call.target_element_id = element_id
                        logger.debug(f"Resolved '{tool_call.target_element_name}' -> '{element_id}'")
                    else:
                        logger.warning(f"Could not resolve target element name '{tool_call.target_element_name}' - will use fallback routing")
                        # Mark as unresolved but still remove the target_element parameter
                        resolved_call.target_element_id = None
                else:
                    # No target element specified, will use default resolution
                    resolved_call.target_element_id = None

                resolved_calls.append(resolved_call)

            return resolved_calls

        except Exception as e:
            logger.error(f"Error resolving target element names: {e}", exc_info=True)
            return tool_calls

    async def _execute_parsed_tool_call(self, tool_call: ParsedToolCall) -> Dict[str, Any]:
        """
        Execute a parsed and resolved tool call.

        Args:
            tool_call: ParsedToolCall with resolved element_id

        Returns:
            Tool execution result
        """
        try:
            # Determine target element
            target_element_id = getattr(tool_call, 'target_element_id', None)

            if not target_element_id:
                # Try to find element with this tool
                target_element_id = self._find_element_with_tool(tool_call.tool_name)
                if not target_element_id:
                    # Fallback to InnerSpace itself
                    target_element_id = self.parent_inner_space.id
                    logger.debug(f"No specific element found for tool '{tool_call.tool_name}', using InnerSpace")

            # Execute tool
            calling_context = {"loop_component_id": self.id, "parsing_mode": "text"}
            logger.debug(f"Executing tool '{tool_call.tool_name}' on element '{target_element_id}' with parameters: {tool_call.parameters}")
            tool_result = await self.parent_inner_space.execute_action_on_element(
                element_id=target_element_id,
                action_name=tool_call.tool_name,
                parameters=tool_call.parameters,
                calling_context=calling_context
            )

            return {
                "tool_name": tool_call.tool_name,
                "parameters": tool_call.parameters,
                "result": tool_result,
                "target_element_id": target_element_id
            }

        except Exception as e:
            logger.error(f"Error executing parsed tool call '{tool_call.tool_name}': {e}", exc_info=True)
            raise

    def _extract_non_tool_text(self, response_text: str) -> str:
        """
        Extract non-tool text from response for conversational processing.

        Removes tool call blocks and returns the remaining text.

        Args:
            response_text: Full LLM response text

        Returns:
            Text with tool calls removed
        """
        try:
            cleaned_text = response_text

            # Remove individual tool XML elements using the same pattern as parsing
            # This matches the same format as _parse_xml_tool_calls: <tool_name attributes>content</tool_name>
            element_pattern = r'<([a-zA-Z_][a-zA-Z0-9_]*)\s*([^>]*)>(.*?)</\1>'
            cleaned_text = re.sub(element_pattern, '', cleaned_text, flags=re.DOTALL)

            # Clean up extra whitespace
            cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
            cleaned_text = cleaned_text.strip()

            return cleaned_text

        except Exception as e:
            logger.error(f"Error extracting non-tool text: {e}", exc_info=True)
            return response_text

    async def _send_conversational_response(self, conversational_text: str, focus_context: Dict[str, Any]) -> None:
        """
        Send non-tool conversational text as a message to the focused conversation.

        This provides fallback behavior so the agent can respond normally without tools.

        Args:
            conversational_text: The non-tool text to send as a response
            focus_context: Focus context containing target element information
        """
        try:
            focus_element_id = focus_context.get('focus_element_id')
            if not focus_element_id:
                logger.warning("No focus element ID in context for conversational response")
                return

            # Send the conversational text as a message to the focused element
            calling_context = {"loop_component_id": self.id, "parsing_mode": "text", "response_type": "conversational"}
            result = await self.parent_inner_space.execute_action_on_element(
                element_id=focus_element_id,
                action_name="msg",
                parameters={"inner_content": conversational_text},
                calling_context=calling_context
            )

            if result and result.get("success"):
                logger.info(f"Sent conversational response ({len(conversational_text)} chars) to element {focus_element_id}")
            else:
                logger.warning(f"Failed to send conversational response: {result}")

        except Exception as e:
            logger.error(f"Error sending conversational response: {e}", exc_info=True)
