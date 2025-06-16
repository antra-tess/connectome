"""
Agent Loop Components
Defines the base class and simple implementations for agent cognitive cycles.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, TYPE_CHECKING, List, Set, Type, Union
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
    
    # Events this component reacts to
    HANDLED_EVENT_TYPES = [
        "activation_call",  # Signals that the agent should consider running a cycle
    ]
    
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

    def handle_event(self, event_node: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handles events for the AgentLoop component.
        Currently processes "activation_call" events to trigger agent cycles.
        
        Args:
            event_node: The event node from the timeline
            timeline_context: The timeline context for the event
            
        Returns:
            True if event was handled, False otherwise
        """
        event_payload = event_node.get('payload', {})
        event_type = event_payload.get('event_type')
        
        if event_type not in self.HANDLED_EVENT_TYPES:
            return False
        
        if event_type == "activation_call":
            activation_reason = event_payload.get('activation_reason', 'unknown')
            source_element_id = event_payload.get('source_element_id', 'unknown')
            
            logger.info(f"[{self.agent_loop_name}] Received activation_call: reason='{activation_reason}', source='{source_element_id}'")
            
            # Check if we should actually trigger a cycle based on our own logic
            should_activate = self._should_activate_for_reason(activation_reason, event_payload)
            
            if should_activate:
                logger.info(f"[{self.agent_loop_name}] Activating agent cycle due to: {activation_reason}")
                
                # Run the cycle asynchronously
                asyncio.create_task(self._run_activation_cycle(activation_reason, event_payload))
            else:
                logger.debug(f"[{self.agent_loop_name}] Skipping activation for reason: {activation_reason}")
            
            return True
        
        return False
    
    def _should_activate_for_reason(self, activation_reason: str, event_payload: Dict[str, Any]) -> bool:
        """
        Decides whether to activate the agent loop for a given activation reason.
        Subclasses can override this for more sophisticated activation logic.
        
        Args:
            activation_reason: The reason for activation (e.g., "direct_message_received")
            event_payload: The full event payload
            
        Returns:
            True if the agent should activate, False otherwise
        """
        # For now, activate for all activation calls
        # Future enhancement: could check agent state, recent activity, specific conditions, etc.
        return True
    
    async def _run_activation_cycle(self, activation_reason: str, event_payload: Dict[str, Any]) -> None:
        """
        Runs the agent cycle and calls on_frame_end afterwards.
        
        Args:
            activation_reason: The reason for activation
            event_payload: The full event payload
        """
        try:
            logger.debug(f"[{self.agent_loop_name}] Starting activation cycle for reason: {activation_reason}")
            
            # NEW: Extract focus context for targeted rendering
            focus_context = event_payload.get('focus_context', {})
            focus_element_id = focus_context.get('focus_element_id')
            
            if focus_element_id:
                logger.info(f"[{self.agent_loop_name}] Activation with focused context on element: {focus_element_id}")
            
            # Run the actual agent cycle with focus context
            await self.trigger_cycle(focus_context=focus_context)
            
            # Call on_frame_end to complete the frame
            if hasattr(self.parent_inner_space, 'on_frame_end') and callable(self.parent_inner_space.on_frame_end):
                logger.debug(f"[{self.agent_loop_name}] Calling on_frame_end after cycle completion")
                self.parent_inner_space.on_frame_end()
            else:
                logger.warning(f"[{self.agent_loop_name}] parent_inner_space does not have callable on_frame_end method")
                
        except Exception as e:
            logger.error(f"[{self.agent_loop_name}] Error during activation cycle: {e}", exc_info=True)

    async def trigger_cycle(self, focus_context: Optional[Dict[str, Any]] = None):
        """
        This method is called by the HostEventLoop to initiate one cognitive cycle
        for the agent. Subclasses must implement this.
        
        Args:
            focus_context: Optional context for focused rendering on specific elements
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

    async def _try_smart_chat_fallback(self, agent_text: str, focus_context: Optional[Dict[str, Any]]) -> bool:
        """
        Smart fallback: if agent has text but no tool calls, try to send it to the activating chat.
        
        This keeps conversations flowing when the agent forgets to use the send_message tool.
        
        Args:
            agent_text: The agent's text response
            focus_context: Focus context from activation event
            
        Returns:
            True if fallback message was sent successfully, False otherwise
        """
        try:
            # Get the source element that activated this loop
            target_element_id = None
            
            if focus_context and focus_context.get('focus_element_id'):
                target_element_id = focus_context.get('focus_element_id')
                logger.debug(f"Smart chat fallback: Using focused element {target_element_id}")
            else:
                # Try to get from recent activation events if no focus context
                try:
                    timeline_comp = self.parent_inner_space.get_timeline()
                    if timeline_comp:
                        # Look for recent activation_call events
                        filter_criteria = {
                            "payload.event_type": "activation_call"
                        }
                        last_activation = timeline_comp.get_last_relevant_event(filter_criteria=filter_criteria, limit=5)
                        if last_activation:
                            last_activation_payload = last_activation.get('payload', {})
                            target_element_id = last_activation_payload.get('source_element_id')
                            logger.debug(f"Smart chat fallback: Found activation source {target_element_id}")
                except Exception as e:
                    logger.debug(f"Could not get activation source from timeline: {e}")
            
            if not target_element_id:
                logger.debug(f"Smart chat fallback: No target element identified")
                return False
            
            # Check if target element exists and has send_message tool
            target_element = self.parent_inner_space.get_element_by_id(target_element_id)
            if not target_element:
                logger.debug(f"Smart chat fallback: Target element {target_element_id} not found")
                return False
            
            # Look for send_message tool on the target element
            tool_provider = target_element.get_component_by_type(ToolProviderComponent)
            if not tool_provider:
                logger.debug(f"Smart chat fallback: No ToolProvider on element {target_element_id}")
                return False
            
            available_tools = tool_provider.list_tools()
            send_message_tool = None
            
            # Look for variations of send message tool
            for tool_name in available_tools:
                if tool_name.lower() in ['send_message', 'send_msg', 'reply', 'send', 'message']:
                    send_message_tool = tool_name
                    break
            
            if not send_message_tool:
                logger.debug(f"Smart chat fallback: No send_message tool found on element {target_element_id}. Available: {available_tools}")
                return False
            
            # Execute the send_message tool with agent's text
            logger.info(f"Smart chat fallback: Sending agent response via {send_message_tool} to {target_element_id}")
            
            calling_context = {"loop_component_id": self.id, "fallback_send": True}
            tool_result = await self.parent_inner_space.execute_action_on_element(
                element_id=target_element_id,
                action_name=send_message_tool,
                parameters={"text": agent_text},
                calling_context=calling_context
            )
            
            logger.info(f"Smart chat fallback: Successfully sent message to {target_element_id} via {send_message_tool}")
            return True
            
        except Exception as e:
            logger.warning(f"Smart chat fallback failed: {e}", exc_info=True)
            return False


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
            # --- Get Context via Unified Pipeline or Fallback ---
            
            if compression_engine and hasattr(hud, 'get_agent_context_via_compression_engine'):
                # Use unified CompressionEngine pipeline
                pipeline_options = {
                    'focus_context': focus_context,
                    'include_memory': True,
                    'render_style': 'verbose_tags'
                }
                context_data = await hud.get_agent_context_via_compression_engine(options=pipeline_options)
                logger.info(f"Using unified CompressionEngine pipeline for context generation")
            else:
                # Simple fallback: get context directly from HUD
                render_options = {'render_style': 'verbose_tags'}
                if focus_context and focus_context.get('focus_element_id'):
                    render_options['render_type'] = 'focused'
                    render_options['focus_element_id'] = focus_context.get('focus_element_id')
                else:
                    render_options['render_type'] = 'full'
                
                context_data = await hud.get_agent_context(options=render_options)
                logger.info(f"Using direct HUD rendering (compression engine not available)")
                
            if not context_data:
                logger.warning(f"{self.agent_loop_name} ({self.id}): No context data received. Aborting cycle.")
                return

            # --- HUD automatically detects and returns appropriate format ---
            has_multimodal_content = isinstance(context_data, dict) and 'attachments' in context_data
            if has_multimodal_content:
                logger.critical(f"CONTEXT DATA text {context_data.get('text')}")
                attachment_count = len(context_data.get('attachments', []))
                text_length = len(context_data.get('text', ''))
                logger.info(f"HUD returned multimodal content: {text_length} chars text + {attachment_count} attachments")
            else:
                logger.critical(f"CONTEXT DATA text {context_data}")
                # Context is text-only string
                logger.debug(f"HUD returned text-only context: {len(str(context_data))} chars")

            
            # --- Build Message for LLM (with multimodal support) ---
            user_message = create_multimodal_llm_message("user", context_data)
            messages = [user_message]
            
            # Log message details
            if user_message.is_multimodal():
                attachment_count = user_message.get_attachment_count()
                text_length = len(user_message.get_text_content())
                logger.info(f"Built multimodal message: {text_length} chars text + {attachment_count} attachments")
            else:
                logger.debug(f"Built text-only message: {len(user_message.get_text_content())} chars")

            # --- Get Available Tools ---
            aggregated_tools = await self.aggregate_tools()

            # --- Send to LLM ---
            llm_response_obj = llm_provider.complete(messages=messages, tools=aggregated_tools)
            
            if not llm_response_obj:
                logger.warning(f"{self.agent_loop_name} ({self.id}): LLM returned no response. Aborting cycle.")
                return

            agent_response_text = llm_response_obj.content
            agent_tool_calls = llm_response_obj.tool_calls or []

            logger.info(f"LLM response: {len(agent_response_text or '')} chars, {len(agent_tool_calls)} tool calls")

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
                    # REMOVED: Excessive timeline event recording - action dispatching creates its own events

            # --- Store Complete Reasoning Chain in CompressionEngine ---
            if compression_engine:
                try:
                    # NEW: Determine which rendering approach was used
                    if hasattr(hud, 'get_agent_context_via_compression_engine') and compression_engine:
                        render_approach = "unified_compression_pipeline"
                    else:
                        render_approach = "legacy_combined_rendering"
                    
                    # Store reasoning chain with context summary instead of full redundant context
                    reasoning_chain_data = {
                        "render_approach_used": render_approach,  # NEW: Track which rendering approach was used
                        "had_multimodal_content": has_multimodal_content,  # NEW: Track multimodal usage
                        "agent_response": agent_response_text,
                        "tool_calls": [
                            {
                                "tool_name": tc.tool_name if hasattr(tc, 'tool_name') else str(tc), 
                                "parameters": tc.parameters if hasattr(tc, 'parameters') else {}
                            } 
                            for tc in agent_tool_calls
                        ],
                        "tool_results": tool_results,
                        "reasoning_notes": f"simple_cycle with {render_approach}" + (" (multimodal)" if has_multimodal_content else ""),
                        "metadata": {
                            "cycle_type": "simple_request_response",
                            "had_focus_context": bool(focus_context),
                            "focus_element_id": focus_context.get('focus_element_id') if focus_context else None,
                            "multimodal_attachments": len(context_data.get('attachments', [])) if isinstance(context_data, dict) else 0,
                            "used_unified_pipeline": render_approach == "unified_compression_pipeline"
                        }
                    }
                    
                    # Instead of storing the full context, create a context summary
                    context_summary = f"{render_approach.replace('_', ' ').title()}"
                    if focus_context and focus_context.get('focus_element_id'):
                        context_summary += f" focused on {focus_context.get('focus_element_id')}"
                    if render_approach == "unified_compression_pipeline":
                        context_summary += " with integrated memory and compression"
                    if has_multimodal_content:
                        attachment_count = len(context_data.get('attachments', [])) if isinstance(context_data, dict) else 0
                        context_summary += f" (multimodal: {attachment_count} attachments)"
                    
                    text_length = len(context_data.get('text', '')) if isinstance(context_data, dict) else len(str(context_data))
                    reasoning_chain_data["context_received"] = f"Agent processed {context_summary} ({text_length} chars)"
                    
                    logger.info(f"Stored reasoning chain with {render_approach}" + (" (multimodal)" if has_multimodal_content else ""))
                except Exception as store_err:
                    logger.error(f"Error storing reasoning chain: {store_err}", exc_info=True)

        except Exception as e:
            logger.error(f"{self.agent_loop_name} ({self.id}): Error during simple cycle: {e}", exc_info=True)
        finally:
            logger.info(f"{self.agent_loop_name} ({self.id}): Simple cycle with memory completed.")

def create_multimodal_llm_message(role: str, context_data: Union[str, Dict[str, Any]], name: Optional[str] = None) -> LLMMessage:
    """
    Create an LLMMessage that supports multimodal content.
    
    Args:
        role: Message role ("user", "assistant", "system")
        context_data: Either a string (text-only) or dict with 'text' and 'attachments' keys
                     (format automatically determined by HUD based on content)
        name: Optional name for the message
        
    Returns:
        LLMMessage with appropriate content format
    """
    if isinstance(context_data, str):
        # Simple text message
        return LLMMessage(role=role, content=context_data, name=name)
    
    elif isinstance(context_data, dict) and 'text' in context_data:
        # Multimodal message (HUD detected attachments and returned structured format)
        text_content = context_data['text']
        attachments = context_data.get('attachments', [])
        
        if not attachments:
            # No attachments, just return text
            return LLMMessage(role=role, content=text_content, name=name)
        
        # Build multimodal content list
        content_parts = []
        
        # Add text part first
        if text_content.strip():
            content_parts.append({
                "type": "text",
                "text": text_content
            })
        
        # Add attachment parts
        for attachment in attachments:
            if isinstance(attachment, dict):
                content_parts.append(attachment)
        return LLMMessage(role=role, content=content_parts, name=name)
    
    else:
        # Fallback to string representation
        return LLMMessage(role=role, content=str(context_data), name=name)
