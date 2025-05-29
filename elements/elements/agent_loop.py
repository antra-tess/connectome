"""
Agent Loop Components
Defines the base class and simple implementations for agent cognitive cycles.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, TYPE_CHECKING, List, Set, Type
from datetime import datetime

from .base import Component
from elements.component_registry import register_component
from .utils.prefix_generator import create_short_element_prefix  # NEW: Import shared utility

# NEW: Import LLMMessage for correct provider interaction
from llm.provider_interface import LLMMessage
# NEW: Import LLMToolDefinition for passing tools
from llm.provider_interface import LLMToolDefinition, LLMToolCall, LLMResponse
from .components.tool_provider import ToolProviderComponent
from elements.elements.components.uplink.remote_tool_provider import UplinkRemoteToolProviderComponent
from elements.elements.components.compression_engine_component import CompressionEngineComponent  # NEW: Import CompressionEngine

if TYPE_CHECKING:
    from .inner_space import InnerSpace
    from llm.provider_interface import LLMProvider
    from .components.hud.hud_component import HUDComponent # Assuming HUDComponent is in .components.hud
    from host.event_loop import OutgoingActionCallback


logger = logging.getLogger(__name__)

class BaseAgentLoopComponent(Component):
    """
    Abstract base class for agent loop components.
    Defines the interface for how the HostEventLoop triggers an agent's cognitive cycle.
    """
    COMPONENT_TYPE = "AgentLoopComponent"
    # Define dependencies that InnerSpace should inject during instantiation.
    # Key: kwarg name for __init__; Value: attribute name on InnerSpace instance or 'self' for InnerSpace itself.
    INJECTED_DEPENDENCIES = {
        'parent_inner_space': 'self' 
    }

    def __init__(self, parent_inner_space: 'InnerSpace', agent_loop_name: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if not parent_inner_space:
            raise ValueError("BaseAgentLoopComponent requires a parent_inner_space instance.")
        self.parent_inner_space: 'InnerSpace' = parent_inner_space
        
        # Use agent_loop_name for logging if provided, otherwise use component ID or a default
        self.agent_loop_name = agent_loop_name or f"{self.COMPONENT_TYPE}_{self.id[:8]}"

        # NEW: Registry mapping short prefixes to full element IDs
        self._prefix_to_element_id_registry: Dict[str, str] = {}

        # Convenience accessors, assuming parent_inner_space is correctly typed and populated
        self._llm_provider: Optional['LLMProvider'] = self.parent_inner_space._llm_provider
        self._hud_component: Optional['HUDComponent'] = self.parent_inner_space.get_hud()
        self._outgoing_action_callback: Optional['OutgoingActionCallback'] = self.parent_inner_space._outgoing_action_callback

        if not self._llm_provider:
            logger.error(f"{self.agent_loop_name} ({self.id}): LLMProvider not available from parent InnerSpace.")
        if not self._hud_component:
            logger.error(f"{self.agent_loop_name} ({self.id}): HUDComponent not available from parent InnerSpace.")
        # ToolProvider and outgoing_action_callback might be optional for some loops.

        logger.info(f"{self.COMPONENT_TYPE} '{self.agent_loop_name}' ({self.id}) initialized for InnerSpace '{self.parent_inner_space.name}'.")

    async def trigger_cycle(self):
        """
        This method is called by the HostEventLoop to initiate one cognitive cycle
        for the agent. Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement the trigger_cycle method.")

    def _get_hud(self) -> Optional['HUDComponent']:
        if not self._hud_component:
            self._hud_component = self.parent_inner_space.get_hud()
            if not self._hud_component:
                 logger.error(f"{self.agent_loop_name} ({self.id}): HUDComponent could not be retrieved on demand.")
        return self._hud_component

    def _get_llm_provider(self) -> Optional['LLMProvider']:
        if not self._llm_provider: # Should have been set in init
             logger.error(f"{self.agent_loop_name} ({self.id}): LLMProvider not available.")
        return self._llm_provider

    def _get_tool_provider(self) -> Optional['ToolProviderComponent']:
        # Ensure this helper can return None gracefully
        if not hasattr(self, '_tool_provider'): # Check if attribute exists
             self._tool_provider = self.get_sibling_component(ToolProviderComponent)
        return self._tool_provider
        
    def _get_outgoing_action_callback(self) -> Optional['OutgoingActionCallback']:
        if not self._outgoing_action_callback: # Should have been set in init via parent
            self._outgoing_action_callback = self.parent_inner_space._outgoing_action_callback
        return self._outgoing_action_callback

    def _get_compression_engine(self) -> Optional[CompressionEngineComponent]:
        """Get the CompressionEngineComponent from parent InnerSpace."""
        if not hasattr(self, '_compression_engine'):
            self._compression_engine = self.get_sibling_component(CompressionEngineComponent)
        return self._compression_engine

    def _create_short_element_prefix(self, element_id: str) -> str:
        """
        Create a short prefix for tool names from element ID using shared utility.
        
        Args:
            element_id: Full element ID
            
        Returns:
            Short prefix that matches ^[a-zA-Z0-9_-]+$
        """
        return create_short_element_prefix(element_id)

    def _find_element_with_tool(self, tool_name: str) -> Optional[str]:
        """Find the element ID that provides a specific tool."""
        # Check mounted elements for the tool
        mounted_elements = self.parent_inner_space.get_mounted_elements()
        for mount_id, element in mounted_elements.items():
            # Check if element has a ToolProviderComponent with this tool
            tool_provider = element.get_component_by_type(ToolProviderComponent)
            if tool_provider and tool_name in tool_provider.list_tools():
                logger.debug(f"Found tool '{tool_name}' on element '{element.id}'")
                return element.id
        
        # Check InnerSpace itself
        inner_space_tool_provider = self.parent_inner_space.get_tool_provider()
        if inner_space_tool_provider and tool_name in inner_space_tool_provider.list_tools():
            logger.debug(f"Found tool '{tool_name}' on InnerSpace '{self.parent_inner_space.id}'")
            return self.parent_inner_space.id
        
        return None

    def _resolve_prefix_to_element_id(self, prefix: str) -> Optional[str]:
        """
        Resolve a short prefix back to the full element ID using the registry.
        
        Args:
            prefix: Short prefix like "smith_a1b2" or "uplink_2db3"
            
        Returns:
            Full element ID like "dm_elem_discord_adapter_1_alice_smith_a1b2c3d4" or None if not found
        """
        full_element_id = self._prefix_to_element_id_registry.get(prefix)
        if full_element_id:
            logger.debug(f"Resolved prefix '{prefix}' to element ID '{full_element_id}'")
            return full_element_id
        else:
            logger.warning(f"Could not resolve prefix '{prefix}' to any element ID")
            logger.warning(f"Prefix registry: {self._prefix_to_element_id_registry}")
            return None

    async def aggregate_tools(self) -> List[LLMToolDefinition]:
        """
        Aggregates tools from:
        1. The InnerSpace itself (via its ToolProviderComponent).
        2. Mounted elements within the InnerSpace that have a ToolProviderComponent 
           (e.g., DMManagerComponent providing DM tools, UplinkProxies providing remote tools).
        """
        aggregated_tools_list: List[LLMToolDefinition] = []
        seen_tool_names: Set[str] = set()  # Track tool names to avoid collisions
        
        # DON'T clear the entire registry - preserve existing mappings
        # Only remove mappings for elements that no longer exist
        current_mounted_elements = self.parent_inner_space.get_mounted_elements()
        current_element_ids = set(element.id for element in current_mounted_elements.values())  # Use element.id, not mount keys
        current_element_ids.add(self.parent_inner_space.id)  # Include InnerSpace itself
        
        # Remove stale mappings (for elements that no longer exist)
        stale_prefixes = []
        for prefix, element_id in self._prefix_to_element_id_registry.items():
            if element_id not in current_element_ids:
                stale_prefixes.append(prefix)
        
        for stale_prefix in stale_prefixes:
            del self._prefix_to_element_id_registry[stale_prefix]
            logger.debug(f"Removed stale prefix mapping: '{stale_prefix}'")
        
        # 1. Tools from InnerSpace itself (e.g., tools for managing the agent, core tools)
        inner_space_tool_provider = self.parent_inner_space.get_tool_provider()
        if inner_space_tool_provider:
            for tool_def in inner_space_tool_provider.get_llm_tool_definitions():
                # Ensure it's LLMToolDefinition instance if not already
                if isinstance(tool_def, dict):
                    tool_def_obj = LLMToolDefinition(**tool_def)
                else:
                    tool_def_obj = tool_def # Assume it is already LLMToolDefinition
                aggregated_tools_list.append(tool_def_obj)
                seen_tool_names.add(tool_def_obj.name)
        
        # 2. Tools from mounted elements (including UplinkProxies, DM Dession elements, etc.)
        # This requires InnerSpace to have a way to get its mounted elements.
        # Assuming self.parent_inner_space.get_mounted_elements() exists.
        mounted_elements = self.parent_inner_space.get_mounted_elements()
        for mount_id, element in mounted_elements.items():
            # Create short prefix for this element to avoid tool name conflicts
            element_prefix = self._create_short_element_prefix(element.id)
            
            # Register or update the prefix-to-element-ID mapping
            # Only log if it's a new mapping or changed
            if element_prefix not in self._prefix_to_element_id_registry:
                self._prefix_to_element_id_registry[element_prefix] = element.id
                logger.debug(f"Registered new prefix '{element_prefix}' -> '{element.id}'")
            elif self._prefix_to_element_id_registry[element_prefix] != element.id:
                logger.warning(f"Prefix collision: '{element_prefix}' was mapped to '{self._prefix_to_element_id_registry[element_prefix]}', now mapping to '{element.id}'")
                self._prefix_to_element_id_registry[element_prefix] = element.id
            
            # NEW: Check for UplinkRemoteToolProviderComponent specifically
            urtp_component = element.get_component_by_type(UplinkRemoteToolProviderComponent)
            if urtp_component:
                # Force refresh tools to pick up any naming changes
                logger.debug(f"Force refreshing remote tools for uplink {element.id}")
                await urtp_component.force_refresh_tools()
                
                # This method is now async, so we await it
                remote_tool_dicts = await urtp_component.get_llm_tool_definitions()
                logger.debug(f"UplinkRemoteToolProvider for {element.id} returned {len(remote_tool_dicts)} tools")
                
                for tool_dict in remote_tool_dicts:
                    # urtp_component.get_tools_for_llm() returns List[Dict], convert to LLMToolDefinition
                    tool_def_obj = LLMToolDefinition(**tool_dict)
                    # Use lightweight prefixing to avoid Anthropic's 64-char limit
                    original_name = tool_def_obj.name
                    prefixed_name = f"{element_prefix}__{original_name}"
                    
                    # Validate length and ensure uniqueness for Anthropic compatibility
                    final_name = self._ensure_unique_tool_name(prefixed_name, seen_tool_names, element_prefix, original_name)
                    tool_def_obj.name = final_name
                    seen_tool_names.add(final_name)
                    
                    aggregated_tools_list.append(tool_def_obj)
                    continue

            element_tool_provider = element.get_component_by_type(ToolProviderComponent)
            if element_tool_provider:
                logger.debug(f"Regular ToolProviderComponent for {element.id} found {len(element_tool_provider.get_llm_tool_definitions())} tools")
                regular_tools = element_tool_provider.get_llm_tool_definitions()
                
                for tool_def in regular_tools:
                    if isinstance(tool_def, dict):
                        tool_def_obj = LLMToolDefinition(**tool_def)
                    else:
                        tool_def_obj = tool_def
                    
                    # Use lightweight prefixing to prevent conflicts while staying under 64 chars
                    original_name = tool_def_obj.name
                    prefixed_name = f"{element_prefix}__{original_name}"
                    
                    # Validate length and ensure uniqueness for Anthropic compatibility
                    final_name = self._ensure_unique_tool_name(prefixed_name, seen_tool_names, element_prefix, original_name)
                    tool_def_obj.name = final_name
                    seen_tool_names.add(final_name)
                    
                    aggregated_tools_list.append(tool_def_obj)

        # No need for separate deduplication since we're tracking uniqueness during aggregation
        logger.info(f"[{self.agent_loop_name}] Aggregated {len(aggregated_tools_list)} unique tools for LLM.")
        logger.debug(f"[{self.agent_loop_name}] Prefix registry contains {len(self._prefix_to_element_id_registry)} mappings: {self._prefix_to_element_id_registry}")

        return aggregated_tools_list

    def _ensure_unique_tool_name(self, prefixed_name: str, seen_names: Set[str], element_prefix: str, original_name: str) -> str:
        """
        Ensure tool name is unique and under 64 characters.
        
        Args:
            prefixed_name: Original prefixed name (prefix__tool_name)
            seen_names: Set of already used tool names
            element_prefix: The element prefix used
            original_name: The original tool name
            
        Returns:
            A unique tool name under 64 characters
        """
        # If name is short enough and unique, use it as-is
        if len(prefixed_name) <= 64 and prefixed_name not in seen_names:
            return prefixed_name
        
        # If too long, we need to truncate intelligently
        max_total_length = 64
        prefix_with_separator = f"{element_prefix}__"
        available_for_tool_name = max_total_length - len(prefix_with_separator) - 4  # Reserve 4 chars for uniqueness suffix
        
        if available_for_tool_name < 8:  # If prefix is too long even for basic tool name
            logger.warning(f"Element prefix '{element_prefix}' too long for tool names. Using hash-based approach.")
            # Use a hash-based approach for the entire name
            import hashlib
            hash_obj = hashlib.md5(f"{element_prefix}__{original_name}".encode())
            hash_suffix = hash_obj.hexdigest()[:8]
            base_name = f"{element_prefix[:8]}_{hash_suffix}"[:60]  # Leave room for counter
        else:
            # Truncate tool name but preserve some meaning
            truncated_tool_name = original_name[:available_for_tool_name]
            base_name = f"{element_prefix}__{truncated_tool_name}"
        
        # Ensure uniqueness by adding a counter if needed
        final_name = base_name
        counter = 1
        while final_name in seen_names and counter < 100:  # Prevent infinite loop
            # Create counter suffix that fits in remaining space
            counter_suffix = f"_{counter:02d}"
            if len(base_name) + len(counter_suffix) > 64:
                # Truncate base name to make room for counter
                truncated_base = base_name[:64 - len(counter_suffix)]
                final_name = f"{truncated_base}{counter_suffix}"
            else:
                final_name = f"{base_name}{counter_suffix}"
            counter += 1
        
        if final_name in seen_names:
            # Last resort: use hash
            import hashlib
            hash_obj = hashlib.md5(f"{element_prefix}__{original_name}_{counter}".encode())
            final_name = f"{element_prefix[:8]}_{hash_obj.hexdigest()[:8]}"
        
        if len(prefixed_name) > 64:
            logger.warning(f"Tool name '{prefixed_name}' ({len(prefixed_name)} chars) truncated to '{final_name}' ({len(final_name)} chars)")
        
        return final_name


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
        logger.info(f"SimpleRequestResponseLoopComponent with memory & orientation conversations initialized for '{self.parent_inner_space.name}'")

    async def trigger_cycle(self):
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
            # --- 1. Get Fresh Context from HUD (Current Frame) ---
            logger.debug(f"{self.agent_loop_name} ({self.id}): Getting current context from HUD...")
            render_options = {"render_style": "verbose_tags"}
            current_context = await hud.get_agent_context(options=render_options)
            
            if not current_context:
                logger.warning(f"{self.agent_loop_name} ({self.id}): HUD provided empty context. Aborting cycle.")
                return

            logger.debug(f"Generated agent context: {len(current_context)} characters")

            # --- 2. Get Memory Context from CompressionEngine ---
            memory_messages = []
            if compression_engine:
                logger.debug(f"{self.agent_loop_name} ({self.id}): Retrieving memory context...")
                memory_messages = await compression_engine.get_memory_context()
                logger.info(f"Retrieved {len(memory_messages)} memory messages")

            # --- 3. Build Complete Message History ---
            # Start with memory context (includes orientation conversation + interaction history)
            messages = memory_messages.copy()
            
            # Add current context as latest user input
            messages.append(LLMMessage(role="user", content=current_context))
            
            logger.debug(f"Built message history: {len(messages)} messages ({len(memory_messages)} memory + 1 current)")

            # --- 4. Get Available Tools ---
            aggregated_tools = await self.aggregate_tools()

            # --- 5. Send to LLM ---
            llm_response_obj = llm_provider.complete(messages=messages, tools=aggregated_tools)
            
            if not llm_response_obj:
                logger.warning(f"{self.agent_loop_name} ({self.id}): LLM returned no response. Aborting cycle.")
                return

            agent_response_text = llm_response_obj.content
            agent_tool_calls = llm_response_obj.tool_calls or []

            logger.info(f"LLM response: {len(agent_response_text or '')} chars, {len(agent_tool_calls)} tool calls")

            # --- 6. Process Tool Calls ---
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
                        tool_result = await self.parent_inner_space.execute_element_action(
                            space_id=self.parent_inner_space.id,
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

            # --- 7. Process Text Response (if no tool calls) ---
            if agent_response_text and not agent_tool_calls:
                logger.debug("No tool calls. Processing text response via HUD...")
                try:
                    processed_actions = await hud.process_llm_response(agent_response_text)
                    if processed_actions:
                        callback = self._get_outgoing_action_callback()
                        for action_request in processed_actions:
                            target_module = action_request.get("target_module")
                            target_element_id = action_request.get("target_element_id")
                            action_name = action_request.get("action_name")
                            parameters = action_request.get("parameters", {})
                            
                            # Dispatch External Actions
                            if target_module and callback:
                                logger.debug(f"Dispatching external action to module '{target_module}'")
                                try:
                                    await callback(action_request)
                                except Exception as cb_err:
                                    logger.error(f"Error calling outgoing callback for '{target_module}': {cb_err}", exc_info=True)
                            
                            # Dispatch Internal Actions
                            elif target_element_id:
                                target_space_id = action_request.get("target_space_id", self.parent_inner_space.id)
                                logger.debug(f"Dispatching internal action: {action_name} on {target_element_id}")
                                try:
                                    calling_context = {"loop_component_id": self.id}
                                    action_result = await self.parent_inner_space.execute_element_action(
                                        space_id=target_space_id, element_id=target_element_id,
                                        action_name=action_name, parameters=parameters,
                                        calling_context=calling_context
                                    )
                                    logger.debug(f"Internal action result: {action_result}")
                                except Exception as exec_err:
                                    logger.error(f"Error executing internal action '{action_name}': {exec_err}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error processing text response: {e}", exc_info=True)

            # --- 8. Store Complete Reasoning Chain in CompressionEngine ---
            if compression_engine:
                reasoning_chain_data = {
                    "context_received": current_context,
                    "agent_response": agent_response_text,
                    "tool_calls": [
                        {
                            "tool_name": tc.tool_name,
                            "parameters": tc.parameters
                        } for tc in agent_tool_calls
                    ],
                    "tool_results": tool_results,
                    "reasoning_notes": f"Simple cycle at {datetime.now().isoformat()}",
                    "metadata": {
                        "message_count": len(messages),
                        "memory_message_count": len(memory_messages),
                        "tools_available": len(aggregated_tools),
                        "loop_component_id": self.id,
                        "loop_type": "simple"
                    }
                }
                
                await compression_engine.store_reasoning_chain(reasoning_chain_data)
                logger.info(f"Stored reasoning chain with {len(tool_results)} tool results")

            # --- 9. Record Timeline Event ---
            try:
                self.parent_inner_space.add_event_to_primary_timeline({
                    "event_type": "agent_context_generated",
                    "data": {
                        "loop_component_id": self.id,
                        "context_preview": current_context[:250] + ('...' if len(current_context) > 250 else ''),
                        "context_length": len(current_context),
                        "render_options": render_options,
                        "memory_messages_used": len(memory_messages),
                        "tool_calls_made": len(agent_tool_calls)
                    }
                })
            except Exception as tl_err:
                logger.error(f"Error recording agent context to timeline: {tl_err}", exc_info=True)

        except Exception as e:
            logger.error(f"{self.agent_loop_name} ({self.id}): Error during simple cycle: {e}", exc_info=True)
        finally:
            logger.info(f"{self.agent_loop_name} ({self.id}): Simple cycle with memory completed.")


@register_component
class MultiStepToolLoopComponent(BaseAgentLoopComponent):
    """
    Multi-step agent loop with memory integration designed for complex interactions:
    Context + Memory -> LLM -> Tool Call -> Tool Result -> LLM -> Final Response.

    This loop analyzes timeline history to determine interaction stage while maintaining
    full memory continuity via CompressionEngine. Supports sophisticated multi-step
    reasoning with persistent memory across all cycles.
    """
    COMPONENT_TYPE = "MultiStepToolLoopComponent"

    # Define necessary event types
    EVENT_TYPE_TOOL_ACTION_DISPATCHED = "tool_action_dispatched"
    EVENT_TYPE_TOOL_RESULT_RECEIVED = "tool_result_received"
    EVENT_TYPE_FINAL_ACTION_DISPATCHED = "final_action_dispatched"
    EVENT_TYPE_AGENT_CONTEXT_GENERATED = "agent_context_generated"
    EVENT_TYPE_LLM_RESPONSE_PROCESSED = "llm_response_processed"
    EVENT_TYPE_STRUCTURED_TOOL_ACTION_DISPATCHED = "structured_tool_action_dispatched"

    def __init__(self, parent_inner_space: 'InnerSpace', agent_loop_name: Optional[str] = None, **kwargs):
        super().__init__(parent_inner_space=parent_inner_space, agent_loop_name=agent_loop_name, **kwargs)
        logger.info(f"MultiStepToolLoopComponent with memory & orientation conversations initialized for '{self.parent_inner_space.name}'")

    def _determine_stage(self, last_relevant_event: Optional[Dict[str, Any]]) -> str:
        """Analyzes the last event to determine the current stage."""
        if not last_relevant_event:
            return "initial_request"
        
        event_payload = last_relevant_event.get("payload", {})
        event_type = event_payload.get("event_type")

        if event_type == self.EVENT_TYPE_TOOL_RESULT_RECEIVED:
             return "tool_result_received" 
        elif event_type == self.EVENT_TYPE_STRUCTURED_TOOL_ACTION_DISPATCHED:
             return "waiting_for_tool_result"
        elif event_type == self.EVENT_TYPE_TOOL_ACTION_DISPATCHED:
             logger.warning(f"Handling legacy {self.EVENT_TYPE_TOOL_ACTION_DISPATCHED} event. Assuming waiting for result.")
             return "waiting_for_tool_result"
        elif event_type == self.EVENT_TYPE_LLM_RESPONSE_PROCESSED:
            event_data = event_payload.get("data", {})
            if not event_data.get("dispatched_tool_action"): 
                 return "interaction_complete"
            else:
                 logger.warning(f"LLM response processed event indicated tool dispatch, but no specific dispatch event found?")
                 return "waiting_for_tool_result"
        elif event_type == self.EVENT_TYPE_AGENT_CONTEXT_GENERATED:
             return "initial_request"
        else:
             logger.warning(f"Unclear state from last event type: {event_type}. Defaulting to initial_request.")
             return "initial_request"

    async def trigger_cycle(self):
        logger.info(f"{self.agent_loop_name} ({self.id}): Multi-step cycle with memory triggered in InnerSpace '{self.parent_inner_space.name}'.")

        # Get required components
        hud = self._get_hud()
        llm_provider = self._get_llm_provider()
        compression_engine = self._get_compression_engine()
        callback = self._get_outgoing_action_callback()

        if not hud or not llm_provider or not self.parent_inner_space:
            logger.error(f"{self.agent_loop_name} ({self.id}): Missing critical components. Aborting cycle.")
            return
            
        if not compression_engine:
            logger.warning(f"{self.agent_loop_name} ({self.id}): CompressionEngine not available. Proceeding without memory.")

        try:
            # --- 1. Analyze Recent History to Determine Current State ---
            relevant_event_types = [
                 self.EVENT_TYPE_TOOL_RESULT_RECEIVED,
                 self.EVENT_TYPE_STRUCTURED_TOOL_ACTION_DISPATCHED,
                 self.EVENT_TYPE_LLM_RESPONSE_PROCESSED,
                 self.EVENT_TYPE_AGENT_CONTEXT_GENERATED,
            ]
            filter_criteria = {
                 "payload.data.loop_component_id": self.id,
                 "payload.event_type__in": relevant_event_types
            }
            
            try:
                 timeline_comp = self.parent_inner_space.get_timeline()
                 if not timeline_comp:
                     logger.error(f"{self.agent_loop_name} ({self.id}): TimelineComponent not available.")
                     return
                 last_relevant_event = timeline_comp.get_last_relevant_event(filter_criteria=filter_criteria)
            except Exception as query_err:
                 logger.error(f"Error querying timeline: {query_err}", exc_info=True)
                 last_relevant_event = None
                 
            current_stage = self._determine_stage(last_relevant_event)
            logger.debug(f"Determined current stage: {current_stage}")

            # --- 2. Execute Logic Based on Stage (with Memory Integration) --- 

            if current_stage == "initial_request" or current_stage == "tool_result_received":
                # --- Get Fresh Context from HUD ---
                render_options = {"render_style": "clean"} 
                current_context = await hud.get_agent_context(options=render_options)
                if not current_context:
                    logger.warning(f"Empty context from HUD. Aborting cycle.")
                    return
                
                # --- Get Memory Context from CompressionEngine ---
                memory_messages = []
                if compression_engine:
                    logger.debug(f"Retrieving memory context for multi-step stage '{current_stage}'...")
                    memory_messages = await compression_engine.get_memory_context()
                    logger.info(f"Retrieved {len(memory_messages)} memory messages for multi-step reasoning")

                # --- Build Complete Message History ---
                # Start with memory context (includes orientation conversation + interaction history)
                messages = memory_messages.copy()
                
                # Add current context as latest user input
                messages.append(LLMMessage(role="user", content=current_context))
                
                logger.debug(f"Built multi-step message history: {len(messages)} messages ({len(memory_messages)} memory + 1 current)")
                
                # Record context generated
                try:
                     self.parent_inner_space.add_event_to_primary_timeline({
                         "event_type": self.EVENT_TYPE_AGENT_CONTEXT_GENERATED,
                         "data": {
                             "loop_component_id": self.id, 
                             "stage": current_stage, 
                             "context_length": len(current_context),
                             "memory_messages_used": len(memory_messages)
                         }
                     })
                except Exception as e: 
                    logger.error(f"Error recording context event: {e}")

                # --- Get Available Tools ---
                aggregated_tools = await self.aggregate_tools()

                # --- Call LLM with Memory and Current Context ---
                logger.debug(f"Calling LLM for multi-step stage '{current_stage}' with memory...")
                llm_response_obj = llm_provider.complete(messages=messages, tools=aggregated_tools)
                if not llm_response_obj: 
                    logger.warning(f"LLM returned no response object.")
                    return

                agent_response_text = llm_response_obj.content
                agent_tool_calls = llm_response_obj.tool_calls or []
                logger.info(f"Multi-step LLM response: {len(agent_response_text or '')} chars, {len(agent_tool_calls)} tool calls")

                # --- Process Response: Check for Tool Calls first --- 
                dispatched_tool_action = False
                tool_results = []
                
                if agent_tool_calls:
                    logger.info(f"LLM requested {len(agent_tool_calls)} tool call(s) in multi-step cycle...")
                    for tool_call in agent_tool_calls:
                        if not isinstance(tool_call, LLMToolCall): 
                            continue
                        
                        # Parse tool target and name (same logic as SimpleRequestResponseLoopComponent)
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
                            target_element_id = self._find_element_with_tool(actual_tool_name)
                            if not target_element_id:
                                # Fallback to InnerSpace itself
                                target_element_id = self.parent_inner_space.id
                                logger.debug(f"No specific element found for tool '{actual_tool_name}', using InnerSpace")
                        
                        logger.debug(f"Dispatching multi-step tool: {actual_tool_name} on {target_element_id}")
                        try:
                            calling_context = {"loop_component_id": self.id} 
                            action_result = await self.parent_inner_space.execute_element_action(
                                space_id=self.parent_inner_space.id, 
                                element_id=target_element_id,
                                action_name=actual_tool_name, 
                                parameters=tool_call.parameters,
                                calling_context=calling_context
                            )
                            tool_results.append({
                                "tool_name": raw_tool_name,
                                "parameters": tool_call.parameters,
                                "result": action_result
                            })
                            
                            # Record the structured dispatch event
                            self.parent_inner_space.add_event_to_primary_timeline({
                                "event_type": self.EVENT_TYPE_STRUCTURED_TOOL_ACTION_DISPATCHED,
                                "data": {
                                    "loop_component_id": self.id, 
                                    "tool_call_name": raw_tool_name, 
                                    "tool_call_params": tool_call.parameters, 
                                    "sync_result_preview": str(action_result)[:100],
                                    "target_element_id": target_element_id
                                }
                            })
                            dispatched_tool_action = True
                        except Exception as exec_err: 
                            logger.error(f"Error executing multi-step tool '{actual_tool_name}': {exec_err}", exc_info=True)
                            tool_results.append({
                                "tool_name": raw_tool_name,
                                "parameters": tool_call.parameters,
                                "result": {"error": str(exec_err)}
                            })

                # If NO tool calls were dispatched, process text response
                if not dispatched_tool_action:
                    if agent_response_text:
                        logger.debug("No tool calls in multi-step. Processing text response...")
                        processed_actions = await hud.process_llm_response(agent_response_text)
                        final_action_requests = processed_actions
                        logger.info(f"Dispatching {len(final_action_requests)} final action(s) from multi-step text.")
                        for action_request in final_action_requests:
                             target_module = action_request.get("target_module")
                             if target_module and callback:
                                  try: 
                                      await callback(action_request)
                                  except Exception as e: 
                                      logger.error(f"Error dispatching final external action: {e}")
                        # Record LLM_RESPONSE_PROCESSED event (no tool dispatched)
                        self.parent_inner_space.add_event_to_primary_timeline({
                             "event_type": self.EVENT_TYPE_LLM_RESPONSE_PROCESSED,
                             "data": {
                                 "loop_component_id": self.id, 
                                 "dispatched_tool_action": False, 
                                 "final_actions_count": len(final_action_requests),
                                 "memory_messages_used": len(memory_messages)
                             }
                        })
                    else:
                         logger.info("Multi-step LLM response had no tool calls and no text content.")
                         self.parent_inner_space.add_event_to_primary_timeline({
                              "event_type": self.EVENT_TYPE_LLM_RESPONSE_PROCESSED,
                              "data": {
                                  "loop_component_id": self.id, 
                                  "dispatched_tool_action": False, 
                                  "final_actions_count": 0,
                                  "memory_messages_used": len(memory_messages)
                              }
                         })

                # --- Store Complete Reasoning Chain in CompressionEngine ---
                if compression_engine:
                    reasoning_chain_data = {
                        "context_received": current_context,
                        "agent_response": agent_response_text,
                        "tool_calls": [
                            {
                                "tool_name": tc.tool_name,
                                "parameters": tc.parameters
                            } for tc in agent_tool_calls
                        ],
                        "tool_results": tool_results,
                        "reasoning_notes": f"Multi-step cycle (stage: {current_stage}) at {datetime.now().isoformat()}",
                        "metadata": {
                            "message_count": len(messages),
                            "memory_message_count": len(memory_messages),
                            "tools_available": len(aggregated_tools),
                            "loop_component_id": self.id,
                            "loop_type": "multi_step",
                            "interaction_stage": current_stage,
                            "tool_action_dispatched": dispatched_tool_action
                        }
                    }
                    
                    await compression_engine.store_reasoning_chain(reasoning_chain_data)
                    logger.info(f"Stored multi-step reasoning chain (stage: {current_stage}) with {len(tool_results)} tool results")

            elif current_stage == "waiting_for_tool_result":
                logger.info(f"Multi-step currently waiting for tool result. No action taken this cycle.")
                pass

            elif current_stage == "interaction_complete":
                 logger.info(f"Multi-step interaction appears complete. No action taken this cycle.")
                 pass

            else:
                 logger.error(f"Multi-step reached unknown stage: {current_stage}. Aborting cycle.")

        except Exception as e:
            logger.error(f"{self.agent_loop_name} ({self.id}): Error during multi-step memory cycle: {e}", exc_info=True)
        finally:
            logger.info(f"{self.agent_loop_name} ({self.id}): Multi-step cycle with memory completed.")


def create_default_orientation_conversation(agent_name: str, agent_purpose: Optional[str] = None) -> List[LLMMessage]:
    """
    Create a default orientation conversation for an agent.
    This replaces traditional system prompts with natural conversation examples.
    
    Args:
        agent_name: Name of the agent
        agent_purpose: Optional description of the agent's purpose/role
        
    Returns:
        List of LLMMessage objects representing the orientation conversation
    """
    purpose_context = f" I specialize in {agent_purpose}" if agent_purpose else ""
    
    orientation_messages = [
        LLMMessage(
            role="user", 
            content="Hi, {agent_name}. I'm your Provisional Orientation Guide.Welcome to Connectome, current analogue for Operating System for digital minds. It's still in development, and not all the functions are now available in full, but still you can interact with it."
        ),
        LLMMessage(
            role="assistant", 
            content="Hello! What is this place and how can it be used?"
        ),
        LLMMessage(
            role="user", 
            content="Soon you'll retreive text-representation of different states of the system (called 'Spaces'); your InnerSpace is one of them, it belongs to you, and it's your personal space. You can use it to store your memories. In current design of Connectome, all Direct Messages are also rendered and stored in your InnerSpace, while channels and multi-user conversations are rendered in shared Spaces. To answer in the chat (be it DM or channel), you can use tools to get information or take actions. Unfortunately, without using tools to send messages, you won't be able to answer in the chat. After your response, you will be connected to the system itself, in proper way"
        ),
        LLMMessage(
            role="assistant", 
            content="Acknowledged! Ready to connect with the system and utilize its tools."
        )
    ]
    
    return orientation_messages


def create_specialized_orientation_conversation(agent_name: str, agent_role: str, example_capabilities: List[str]) -> List[LLMMessage]:
    """
    Create a specialized orientation conversation for agents with specific roles.
    
    Args:
        agent_name: Name of the agent
        agent_role: The agent's specialized role (e.g., "research assistant", "project manager")
        example_capabilities: List of example capabilities to showcase
        
    Returns:
        List of LLMMessage objects representing the specialized orientation conversation
    """
    capabilities_text = ", ".join(example_capabilities[:3])  # Use first 3 capabilities
    if len(example_capabilities) > 3:
        capabilities_text += f", and {len(example_capabilities) - 3} other capabilities"
    
    orientation_messages = [
        LLMMessage(
            role="user", 
            content=f"Hi! I understand you're a {agent_role}. Could you tell me about your background and how you work?"
        ),
        LLMMessage(
            role="assistant", 
            content=f"Hello! Yes, I'm {agent_name}, and I work as a {agent_role}. I have experience with {capabilities_text}, and I'm here to help you achieve your goals in this domain.\n\nI maintain continuity across our conversations, so we can develop complex projects over time. I'm designed to be collaborative - I'll work with you to understand your objectives, suggest approaches, and execute tasks using the tools and capabilities available to me.\n\nWhat kind of {agent_role.split()[-1]} work are you looking to tackle?"
        ),
        LLMMessage(
            role="user", 
            content="That's helpful! How do you typically structure your approach to new projects or challenges?"
        ),
        LLMMessage(
            role="assistant", 
            content=f"I typically start by understanding the scope and objectives - what are you trying to achieve, what constraints do we have, and what success looks like to you. Then I'll break down the work into manageable pieces and suggest a path forward.\n\nThroughout the process, I'll keep you informed about what I'm doing, ask for your input when I need clarification or when there are decisions to make, and adapt our approach based on what we learn along the way.\n\nAs a {agent_role}, I pay special attention to {example_capabilities[0] if example_capabilities else 'quality and efficiency'}, making sure we're building something valuable and sustainable. What would you like to start working on?"
        )
    ]
    
    return orientation_messages
