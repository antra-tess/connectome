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
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

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
        logger.info(f"ToolTextParsingLoopComponent initialized for '{self.parent_inner_space.name}'")

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

        try:
            # 1. Aggregate tools and build element name mapping
            aggregated_tools = await self.aggregate_tools()
            enhanced_tools_from_veil = self._extract_enhanced_tools_from_veil()
            self._build_element_name_mapping(enhanced_tools_from_veil)

            # 2. Send tools + context to HUD for rendering with aggregation
            pipeline_options = {
                'focus_context': focus_context,
                'include_memory': True,
                'render_style': 'chronological_flat',
                'tool_rendering_mode': 'full'  # Use full mode for text parsing
            }
            context_data = await hud.get_agent_context_via_compression_engine(
                options=pipeline_options,
                tools=enhanced_tools_from_veil
            )
            logger.info(f"Using full tool rendering mode for text parsing")

            if not context_data:
                logger.warning(f"{self.agent_loop_name} ({self.id}): No context data received. Aborting cycle.")
                return

            # --- HUD automatically detects and returns appropriate format ---
            has_multimodal_content = isinstance(context_data, dict) and 'attachments' in context_data
            if has_multimodal_content:
                attachment_count = len(context_data.get('attachments', []))
                text_length = len(context_data.get('text', ''))
                logger.info(f"HUD returned multimodal content: {text_length} chars text + {attachment_count} attachments")
            else:
                # Context is text-only string
                logger.debug(f"HUD returned text-only context: {len(str(context_data))} chars")

            # --- Build Message for LLM (with multimodal support) ---
            user_message = create_multimodal_llm_message("user", context_data)
            logger.critical(f"USER MESSAGE: {user_message}")
            messages = [user_message]

            # Log message details
            if user_message.is_multimodal():
                attachment_count = user_message.get_attachment_count()
                text_length = len(user_message.get_text_content())
                logger.info(f"Built multimodal message: {text_length} chars text + {attachment_count} attachments")
            else:
                logger.debug(f"Built text-only message: {len(user_message.get_text_content())} chars")

            # 3. Send rendered context to LLM (no separate tool definitions - they're in the context)
            # NOTE: We don't pass tools parameter to LLM since they're already rendered in the context
            llm_response_obj = llm_provider.complete(messages=messages, tools=[])

            if not llm_response_obj:
                logger.warning(f"{self.agent_loop_name} ({self.id}): LLM returned no response. Aborting cycle.")
                return

            agent_response_text = llm_response_obj.content or ""
            logger.info(f"LLM response: {len(agent_response_text)} chars")
            logger.critical(f"LLM RESPONSE: {agent_response_text}")

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
        except Exception as e:
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
            
            for tool_def in enhanced_tools:
                element_id = tool_def.get("target_element_id")
                element_name = tool_def.get("element_name")
                
                if element_id and element_name:
                    # Avoid overwriting if multiple tools from same element
                    if element_name not in self._element_name_to_id_mapping:
                        self._element_name_to_id_mapping[element_name] = element_id
                        logger.debug(f"Mapped element name '{element_name}' -> '{element_id}'")
            
            logger.debug(f"Built element name mapping with {len(self._element_name_to_id_mapping)} entries")
            
        except Exception as e:
            logger.error(f"Error building element name mapping: {e}", exc_info=True)

    def _parse_tool_calls_from_response(self, response_text: str) -> List[ParsedToolCall]:
        """
        Parse tool calls from LLM response text.
        
        Supports both JSON blocks and XML format as described in the refactor document.
        
        Args:
            response_text: Raw text response from LLM
            
        Returns:
            List of ParsedToolCall objects
        """
        try:
            parsed_calls = []
            
            # First try JSON blocks format (Option A from refactor doc)
            json_calls = self._parse_json_tool_calls(response_text)
            parsed_calls.extend(json_calls)
            
            # Then try XML format (Option B from refactor doc)
            xml_calls = self._parse_xml_tool_calls(response_text)
            parsed_calls.extend(xml_calls)
            
            logger.debug(f"Parsed {len(parsed_calls)} tool calls from response")
            return parsed_calls
            
        except Exception as e:
            logger.error(f"Error parsing tool calls from response: {e}", exc_info=True)
            return []

    def _parse_json_tool_calls(self, response_text: str) -> List[ParsedToolCall]:
        """
        Parse JSON-format tool calls from response.
        
        Looks for <tool_calls> JSON blocks as described in the refactor document.
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
                                tool_name = tool_call_data.get("tool", "")
                                parameters = tool_call_data.get("parameters", {})
                                
                                if tool_name:
                                    parsed_call = ParsedToolCall(
                                        tool_name=tool_name,
                                        parameters=parameters,
                                        target_element_name=parameters.get("target_element"),
                                        raw_text=match
                                    )
                                    parsed_calls.append(parsed_call)
                                    logger.debug(f"Parsed JSON tool call: {tool_name}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON tool call block: {e}")
                    continue
            
            return parsed_calls
            
        except Exception as e:
            logger.error(f"Error parsing JSON tool calls: {e}", exc_info=True)
            return []

    def _parse_xml_tool_calls(self, response_text: str) -> List[ParsedToolCall]:
        """
        Parse XML-format tool calls from response.
        
        Looks for <tool_call> XML blocks as described in the refactor document.
        """
        try:
            parsed_calls = []
            
            # Find <tool_call> blocks
            tool_call_pattern = r'<tool_call\s+tool="([^"]+)"[^>]*>(.*?)</tool_call>'
            matches = re.findall(tool_call_pattern, response_text, re.DOTALL)
            
            for tool_name, parameters_block in matches:
                try:
                    # Parse parameters from the block
                    parameters = {}
                    param_pattern = r'<parameter\s+name="([^"]+)"[^>]*>(.*?)</parameter>'
                    param_matches = re.findall(param_pattern, parameters_block, re.DOTALL)
                    
                    for param_name, param_value in param_matches:
                        # Try to parse as JSON, fall back to string
                        try:
                            parameters[param_name] = json.loads(param_value.strip())
                        except:
                            parameters[param_name] = param_value.strip()
                    
                    if tool_name:
                        parsed_call = ParsedToolCall(
                            tool_name=tool_name,
                            parameters=parameters,
                            target_element_name=parameters.get("target_element"),
                            raw_text=f'<tool_call tool="{tool_name}">{parameters_block}</tool_call>'
                        )
                        parsed_calls.append(parsed_call)
                        logger.debug(f"Parsed XML tool call: {tool_name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to parse XML tool call parameters: {e}")
                    continue
            
            return parsed_calls
            
        except Exception as e:
            logger.error(f"Error parsing XML tool calls: {e}", exc_info=True)
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
                    element_id = self._element_name_to_id_mapping.get(tool_call.target_element_name)
                    if element_id:
                        # Remove target_element from parameters and store element_id separately
                        if "target_element" in resolved_call.parameters:
                            del resolved_call.parameters["target_element"]
                        resolved_call.target_element_id = element_id
                        logger.debug(f"Resolved '{tool_call.target_element_name}' -> '{element_id}'")
                    else:
                        logger.warning(f"Could not resolve target element name '{tool_call.target_element_name}'")
                        # Keep the original call but mark as unresolved
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
            
            # Remove <tool_calls> JSON blocks
            cleaned_text = re.sub(r'<tool_calls>.*?</tool_calls>', '', cleaned_text, flags=re.DOTALL)
            
            # Remove <tool_call> XML blocks
            cleaned_text = re.sub(r'<tool_call\s+[^>]*>.*?</tool_call>', '', cleaned_text, flags=re.DOTALL)
            
            # Clean up extra whitespace
            cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
            cleaned_text = cleaned_text.strip()
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error extracting non-tool text: {e}", exc_info=True)
            return response_text

    async def _emit_agent_response_delta(self, agent_response_text: str, tool_calls: List[ParsedToolCall]) -> None:
        """
        Emit a VEIL delta containing the agent's response for chronological rendering.
        
        Args:
            agent_response_text: The agent's full text response from LLM
            tool_calls: List of parsed tool calls
        """
        try:
            current_time = time.time()
            
            # Create unique VEIL ID for this agent response
            response_veil_id = f"agent_response_{self.parent_inner_space.id}_{int(current_time * 1000)}"
            
            # Build agent response delta
            agent_response_delta = {
                "op": "add_node",
                "node": {
                    "veil_id": response_veil_id,
                    "node_type": "agent_response",
                    "properties": {
                        "structural_role": "list_item",
                        "content_nature": "agent_response",
                        "agent_response_text": agent_response_text,
                        "tool_calls_count": len(tool_calls),
                        "has_tool_calls": len(tool_calls) > 0,
                        "agent_name": getattr(self.parent_inner_space, 'agent_name', 'Agent'),
                        "agent_loop_component_id": self.id,
                        "timestamp": current_time,
                        "parsing_mode": "text",  # Distinguish from tool_call API mode
                        # Add operation index for chronological ordering
                        "operation_index": int(current_time * 1000)
                    }
                }
            }
            
            # Add owner tracking to the delta
            if self.parent_inner_space:
                agent_response_delta["node"]["properties"]["owner_element_id"] = self.parent_inner_space.id
            
            # Submit delta to VEIL system
            space_veil_producer = self.parent_inner_space.get_component_by_type("SpaceVeilProducer")
            if space_veil_producer:
                space_veil_producer.signal_delta_produced_externally([agent_response_delta])
                logger.debug(f"Emitted agent response delta with {len(tool_calls)} tool calls")
            else:
                logger.warning(f"No SpaceVeilProducer found to emit agent response delta")
                
        except Exception as e:
            logger.error(f"Error emitting agent response delta: {e}", exc_info=True) 