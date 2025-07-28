"""
Base Agent Loop Component
Defines the abstract base class for agent cognitive cycles.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, TYPE_CHECKING, List, Set, Type, Union
from datetime import datetime

from ...base import Component
from elements.component_registry import register_component
from ...utils.prefix_generator import create_short_element_prefix  # NEW: Import shared utility

# NEW: Import LLMMessage for correct provider interaction
from llm.provider_interface import LLMMessage
# NEW: Import LLMToolDefinition for passing tools
from llm.provider_interface import LLMToolDefinition, LLMToolCall, LLMResponse
from ...components.tool_provider import ToolProviderComponent
from elements.elements.components.uplink.remote_tool_provider import UplinkRemoteToolProviderComponent
from elements.elements.components.veil_facet_compression_engine import VEILFacetCompressionEngine  # NEW: Import VEILFacet CompressionEngine
from elements.elements.components.veil import VEILFacetType  # NEW: Import VEILFacetType for facet filtering

# NEW: Import turn-based message utilities
from .utils import create_multimodal_llm_message

# NEW: Turn-based messaging configuration
TURN_BASED_CONFIG = {
    "max_turn_length": 4000,  # Max characters per turn
    "default_ambient_threshold_chars": 2000,  # Default re-render ambient every N characters
    "status_repetition_enabled": True,
    "focused_element_status_every_turn": True,
    "include_turn_metadata": True,
    # Per-facet ambient thresholds (can override default)
    "ambient_facet_thresholds": {
        "tool_instructions": 1500,  # Tools need frequent updates
        "general_instructions": 3000,  # General instructions less frequent
    }
}

if TYPE_CHECKING:
    from ...inner_space import InnerSpace
    from llm.provider_interface import LLMProvider
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

    def _get_compression_engine(self) -> Optional[VEILFacetCompressionEngine]:
        """Get the VEILFacetCompressionEngine from parent InnerSpace."""
        if not hasattr(self, '_compression_engine'):
            self._compression_engine = self.get_sibling_component("VEILFacetCompressionEngine")
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

    async def _emit_agent_response_delta(self, agent_response_text: str, agent_tool_calls: List) -> None:
        """
        Emit agent response via SpaceVeilProducer for centralized, reusable VEILFacet Event creation.
        Also persists to timeline for replay capability.
        
        Args:
            agent_response_text: The agent's text response from LLM
            agent_tool_calls: List of tool calls the agent made
        """
        try:
            # Get the SpaceVeilProducer for centralized agent response emission
            space_veil_producer = self._get_space_veil_producer()
            if not space_veil_producer:
                logger.error(f"{self.agent_loop_name} ({self.id}): No SpaceVeilProducer available")
                return
            
            # Convert tool calls to serializable format
            tool_calls_data = self._convert_tool_calls_to_data(agent_tool_calls)
            
            # Use centralized agent response emission
            response_id = space_veil_producer.emit_agent_response(
                agent_response_text=agent_response_text,
                tool_calls_data=tool_calls_data,
                agent_loop_component_id=self.id,
                parsing_mode=self._get_parsing_mode(),
                links_to=None
            )
            
            if response_id:
                logger.debug(f"Successfully emitted agent response {response_id}")
                
                # Persist to timeline for replay
                await self._persist_agent_response_to_timeline(
                    response_id=response_id,
                    agent_response_text=agent_response_text,
                    tool_calls_data=tool_calls_data
                )
            else:
                logger.warning(f"Failed to emit agent response via SpaceVeilProducer")
                
        except Exception as e:
            logger.error(f"Error emitting agent response: {e}", exc_info=True)

    def _extract_enhanced_tools_from_veil(self) -> List[Dict[str, Any]]:
        """
        NEW: Extract enhanced tool definitions from VEILFacetCache for Phase 2 integration.
        
        This method gets enhanced tool metadata from StatusFacets that represent
        container creation events with available_tools metadata.
        
        Returns:
            List of enhanced tool definitions with complete metadata
        """
        try:
            enhanced_tools = []
            
            hud = self._get_hud()
            if not hud:
                logger.debug(f"No HUD available for extracting enhanced tools from VEIL")
                return []
            
            # Get VEILFacetCache directly from SpaceVeilProducer via HUD
            veil_producer = hud._get_space_veil_producer()
            if not veil_producer:
                logger.debug(f"No SpaceVeilProducer available for enhanced tool extraction")
                return []
                
            facet_cache = veil_producer.get_facet_cache()
            if not facet_cache:
                logger.debug(f"No VEILFacetCache available for enhanced tool extraction")
                return []
            
            # Extract enhanced tools from StatusFacets representing container creation
            for facet in facet_cache.facets.values():
                if facet.facet_type == VEILFacetType.STATUS:  # StatusFacet
                    # Look for container creation status facets
                    if facet.get_property("status_type") == "container_created":
                        current_state = facet.get_property("current_state")
                        if isinstance(current_state, dict):
                            # Extract element metadata from the status facet
                            element_id = current_state.get("element_id")
                            element_name = current_state.get("element_name") or current_state.get("conversation_name")
                            
                            # Extract available_tools from current_state
                            available_tools = current_state.get("available_tools", [])
                            if isinstance(available_tools, list) and available_tools:
                                # Check if these are enhanced tool definitions (not just names)
                                for tool in available_tools:
                                    if isinstance(tool, dict) and "name" in tool and "parameters" in tool:
                                        # FIXED: Add element metadata to each tool
                                        enhanced_tool = dict(tool)  # Copy the tool
                                        enhanced_tool["target_element_id"] = element_id
                                        enhanced_tool["element_name"] = element_name
                                        enhanced_tools.append(enhanced_tool)
                                        
                    # Also check general status facets with available_tools in properties
                    elif facet.properties:
                        properties = facet.properties
                        if isinstance(properties, dict):
                            # Extract element metadata from properties
                            element_id = properties.get("element_id")
                            element_name = properties.get("element_name") or properties.get("conversation_name")
                            
                            available_tools = properties.get("available_tools", [])
                            if isinstance(available_tools, list) and available_tools:
                                for tool in available_tools:
                                    if isinstance(tool, dict) and "name" in tool and "parameters" in tool:
                                        # FIXED: Add element metadata to each tool
                                        enhanced_tool = dict(tool)  # Copy the tool
                                        enhanced_tool["target_element_id"] = element_id
                                        enhanced_tool["element_name"] = element_name
                                        enhanced_tools.append(enhanced_tool)
            
            logger.debug(f"Extracted {len(enhanced_tools)} enhanced tools from VEILFacetCache")
            
            # Debug logging to help troubleshoot
            if not enhanced_tools:
                logger.warning(f"No enhanced tools extracted from VEILFacetCache. Checking facet count...")
                total_facets = len(facet_cache.facets) if facet_cache and facet_cache.facets else 0
                status_facets = sum(1 for f in facet_cache.facets.values() if f.facet_type == VEILFacetType.STATUS) if facet_cache and facet_cache.facets else 0
                logger.warning(f"Total facets: {total_facets}, Status facets: {status_facets}")
            else:
                logger.debug(f"Sample enhanced tool: {enhanced_tools[0] if enhanced_tools else 'none'}")
            
            return enhanced_tools
            
        except Exception as e:
            logger.error(f"Error extracting enhanced tools from VEILFacetCache: {e}", exc_info=True)
            return []

    # --- NEW: Turn-Based Message Processing Utilities ---
    
    def _is_turn_based_context(self, context_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        """
        Check if context data is in turn-based format.
        
        Args:
            context_data: Context from HUD
            
        Returns:
            True if context is turn-based message array, False otherwise
        """
        return isinstance(context_data, list) and all(
            isinstance(item, dict) and "role" in item and "content" in item 
            for item in context_data
        )
    
    def _is_multimodal_turn_based_context(self, context_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        """
        Check if context data contains multimodal turn-based content.
        
        Args:
            context_data: Context from HUD
            
        Returns:
            True if context contains multimodal turn-based data, False otherwise
        """
        if isinstance(context_data, dict):
            return 'messages' in context_data and 'multimodal_content' in context_data
        return False
    
    def _build_messages_from_turn_based_context(self, context_data: List[Dict[str, Any]]) -> List[LLMMessage]:
        """
        Convert turn-based context to LLM messages.
        
        Args:
            context_data: Turn-based message array from HUD
            
        Returns:
            List of LLMMessage objects for LLM provider
        """
        messages = []
        
        for turn_data in context_data:
            role = turn_data.get("role", "user")
            content = turn_data.get("content", "")
            turn_metadata = turn_data.get("turn_metadata")  # Extract turn metadata if present
            
            # Create LLM message using existing utility with metadata
            message = create_multimodal_llm_message(role, content, turn_metadata=turn_metadata)
            messages.append(message)
            
        logger.debug(f"Built {len(messages)} messages from turn-based context")
        return messages
    
    def _build_messages_from_multimodal_turn_based_context(self, context_data: Dict[str, Any]) -> List[LLMMessage]:
        """
        Convert multimodal context with turn-based messages to LLM messages.
        
        Args:
            context_data: Multimodal context dict with 'messages' array
            
        Returns:
            List of LLMMessage objects for LLM provider
        """
        turn_messages = context_data.get("messages", [])
        messages = []
        
        for turn_data in turn_messages:
            role = turn_data.get("role", "user")
            content = turn_data.get("content", "")
            turn_metadata = turn_data.get("turn_metadata")  # Extract turn metadata if present
            
            # For multimodal content, we may need to handle attachments
            # The create_multimodal_llm_message should handle this
            message = create_multimodal_llm_message(role, content, turn_metadata=turn_metadata)
            messages.append(message)
            
        logger.debug(f"Built {len(messages)} messages from multimodal turn-based context")
        return messages
    
    def _process_context_to_messages(self, context_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> List[LLMMessage]:
        """
        Universal context processor that handles all context formats.
        
        This method processes the new turn-based format from HUD and converts it to LLM messages.
        
        Args:
            context_data: Context from HUD in turn-based format
            
        Returns:
            List of LLMMessage objects for LLM provider
        """
        try:
            if self._is_turn_based_context(context_data):
                # New turn-based format from HUD
                return self._build_messages_from_turn_based_context(context_data)
            
            elif self._is_multimodal_turn_based_context(context_data):
                # Multimodal format with turn-based messages
                return self._build_messages_from_multimodal_turn_based_context(context_data)
                
        except Exception as e:
            logger.error(f"Error processing context to messages: {e}", exc_info=True)
            # Fallback: treat as legacy
            return self._build_messages_from_legacy_context(context_data)
    
    def _log_context_format(self, context_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """
        Log detailed information about the context format received.
        
        Args:
            context_data: Context from HUD
        """
        if self._is_turn_based_context(context_data):
            turn_count = len(context_data)
            turn_roles = [turn.get("role", "unknown") for turn in context_data]
            logger.info(f"Received turn-based context: {turn_count} turns with roles {turn_roles}")
            
            # Log turn metadata if available
            for i, turn in enumerate(context_data):
                metadata = turn.get("turn_metadata", {})
                if metadata:
                    logger.debug(f"Turn {i}: {metadata}")
            
        elif self._is_multimodal_turn_based_context(context_data):
            turn_count = len(context_data.get("messages", []))
            multimodal_info = context_data.get("multimodal_content", {})
            attachment_count = multimodal_info.get("attachment_count", 0)
            logger.info(f"Received multimodal turn-based context: {turn_count} turns, {attachment_count} attachments")
            
        else:
            # Legacy format (should be rare with new HUD)
            if isinstance(context_data, dict) and 'attachments' in context_data:
                # Legacy multimodal format
                attachment_count = len(context_data.get('attachments', []))
                text_length = len(context_data.get('text', ''))
                logger.warning(f"Received legacy multimodal context: {text_length} chars text + {attachment_count} attachments")
            else:
                # Legacy string format
                logger.warning(f"Received legacy string context: {len(str(context_data))} chars")

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

    def _get_space_veil_producer(self):
        """
        Get the SpaceVeilProducer from the parent InnerSpace.
        
        Returns:
            SpaceVeilProducer instance or None if not available
        """
        producer = self.get_sibling_component("SpaceVeilProducer")
        if not producer:
            logger.error(f"{self.agent_loop_name} ({self.id}): No SpaceVeilProducer available for agent response emission")
            return None
        return producer
    
    def _get_parsing_mode(self) -> str:
        """Get parsing mode for this loop type. Override in subclasses if needed."""
        # Default implementation based on class type
        if self.__class__.__name__ == "ToolTextParsingLoopComponent":
            return "text"
        elif self.__class__.__name__ == "SimpleRequestResponseLoopComponent":
            return "tool_call"
        else:
            return "unknown"
    
    def _convert_tool_calls_to_data(self, agent_tool_calls: List) -> List[Dict[str, Any]]:
        """Convert tool calls to serializable format. Override if needed."""
        tool_calls_data = []
        
        for tool_call in agent_tool_calls:
            if hasattr(tool_call, '__dict__'):  # ParsedToolCall objects
                tool_call_dict = {
                    "tool_name": getattr(tool_call, 'tool_name', ''),
                    "parameters": getattr(tool_call, 'parameters', {}),
                    "target_element_name": getattr(tool_call, 'target_element_name', None),
                    "raw_text": getattr(tool_call, 'raw_text', '')
                }
                if hasattr(tool_call, 'target_element_id'):
                    tool_call_dict["target_element_id"] = tool_call.target_element_id
                tool_calls_data.append(tool_call_dict)
            elif isinstance(tool_call, dict):  # Already a dict
                tool_calls_data.append(tool_call)
            else:  # LLMToolCall objects
                tool_call_dict = {
                    "tool_name": getattr(tool_call, 'tool_name', ''),
                    "parameters": getattr(tool_call, 'parameters', {})
                }
                tool_calls_data.append(tool_call_dict)
        
        return tool_calls_data
    
    async def _persist_agent_response_to_timeline(
        self, 
        response_id: str,
        agent_response_text: str, 
        tool_calls_data: List[Dict[str, Any]]
    ) -> None:
        """Persist agent response to timeline for replay capability."""
        try:
            import time
            
            # Create timeline event
            agent_response_event = {
                "event_type": "agent_response_generated",
                "target_element_id": self.parent_inner_space.id,
                "is_replayable": True,
                "payload": {
                    "response_id": response_id,
                    "agent_response_text": agent_response_text,
                    "tool_calls_data": tool_calls_data,
                    "agent_loop_component_id": self.id,
                    "agent_loop_type": self.__class__.__name__,
                    "parsing_mode": self._get_parsing_mode(),
                    "timestamp": time.time(),
                    "agent_name": getattr(self.parent_inner_space, 'agent_name', 'Unknown Agent')
                }
            }
            
            # Add to timeline
            timeline_context = {"timeline_id": self.parent_inner_space.get_primary_timeline()}
            event_id = self.parent_inner_space.add_event_to_timeline(
                agent_response_event, 
                timeline_context
            )
            
            if event_id:
                logger.debug(f"Persisted agent response to timeline: {event_id}")
            else:
                logger.warning(f"Failed to persist agent response to timeline")
                
        except Exception as e:
            logger.error(f"Error persisting agent response to timeline: {e}", exc_info=True)
