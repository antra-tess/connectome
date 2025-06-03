import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime, timezone

from ..base_component import Component
# Needs access to the SpaceVeilProducer on the owner (InnerSpace)
from ..space.space_veil_producer import SpaceVeilProducer
# Import the registry decorator
from elements.component_registry import register_component
# Import shared prefix generator for consistency with AgentLoop
from ...utils.prefix_generator import create_short_element_prefix
# May need access to GlobalAttentionComponent for filtering
# from ..attention.global_attention_component import GlobalAttentionComponent

if TYPE_CHECKING:
    # May need LLM provider for summarization/rendering assistance
    from llm.provider_interface import LLMProviderInterface

logger = logging.getLogger(__name__)

@register_component
class HUDComponent(Component):
    """
    Head-Up Display Component.

    Responsible for generating a contextual representation (e.g., a prompt)
    of the agent's current state based on the InnerSpace's aggregated VEIL.
    """
    COMPONENT_TYPE = "HUDComponent"

    # Dependencies that InnerSpace should inject
    # Optional: LLMProvider for advanced rendering/summarization
    INJECTED_DEPENDENCIES = {
        'llm_provider': '_llm_provider'
    }

    def __init__(self, llm_provider: Optional['LLMProviderInterface'] = None, **kwargs):
        super().__init__(**kwargs)
        self._llm_provider = llm_provider # Optional LLM for advanced processing

    def initialize(self, **kwargs) -> None:
        """Initializes the HUD component."""
        super().initialize(**kwargs)
        logger.debug(f"HUDComponent initialized for Element {self.owner.id}")
        if self._llm_provider:
            logger.debug(f"HUDComponent using LLM provider: {self._llm_provider.__class__.__name__}")

    def _get_space_veil_producer(self) -> Optional[SpaceVeilProducer]:
        """Helper to get the SpaceVeilProducer from the owning InnerSpace."""
        if not self.owner:
            return None
        # Assuming SpaceVeilProducer is the primary/only VEIL producer on InnerSpace
        return self.get_sibling_component("SpaceVeilProducer")

    # def _get_global_attention(self) -> Optional[GlobalAttentionComponent]:
    #     """Helper to get the GlobalAttentionComponent from the owning InnerSpace."""
    #     if not self.owner:
    #         return None
    #     return self.owner.get_component(GlobalAttentionComponent)

    async def get_agent_context(self, options: Optional[Dict[str, Any]] = None) -> str:
        """
        (Async) Generates the agent's context, suitable for an LLM prompt.

        Retrieves the aggregated VEIL from the InnerSpace, filters/renders it,
        and returns a structured string representation.

        Args:
            options: Optional dictionary controlling context generation
                     (e.g., verbosity, max_tokens, focus_element_id,
                      render_style: 'clean' (default) or 'verbose_tags',
                      focused_rendering: bool - if True, only render focus_element_id).

        Returns:
            A string representing the agent's current context.
        """
        logger.debug(f"Generating agent context for {self.owner.id}...")
        options = options or {}

        # NEW: Check for focused rendering mode
        focused_rendering = options.get('focused_rendering', False)
        focus_element_id = options.get('focus_element_id')
        conversation_context = options.get('conversation_context', {})
        
        if focused_rendering and focus_element_id:
            logger.info(f"[{self.owner.id}] FOCUSED RENDERING: Generating context only for element {focus_element_id}")
            return await self._get_focused_element_context(focus_element_id, conversation_context, options)
        else:
            logger.debug(f"[{self.owner.id}] FULL RENDERING: Generating complete space context")
            return await self._get_full_space_context(options)

    async def _get_focused_element_context(self, focus_element_id: str, conversation_context: Dict[str, Any], options: Dict[str, Any]) -> str:
        """
        Generate context for a specific focused element with essential InnerSpace context.
        
        This ensures the agent maintains their identity and workspace context while
        focusing on the specific conversation that triggered activation.
        
        Args:
            focus_element_id: The ID of the element to focus on
            conversation_context: Additional context about the conversation
            options: Rendering options
            
        Returns:
            A focused context string containing InnerSpace essentials + the focused element
        """
        try:
            # Find the focused element
            focused_element = None
            if focus_element_id == self.owner.id:
                # If focusing on InnerSpace itself, use full context
                logger.info(f"[{self.owner.id}] Focus target is InnerSpace itself, using full context")
                return await self._get_full_space_context(options)
            else:
                # Check mounted elements
                mounted_elements = self.owner.get_mounted_elements() if hasattr(self.owner, 'get_mounted_elements') else {}
                for mount_id, element in mounted_elements.items():
                    if element.id == focus_element_id or mount_id == focus_element_id:
                        focused_element = element
                        break
            
            if not focused_element:
                logger.warning(f"[{self.owner.id}] Focused element {focus_element_id} not found, falling back to full context")
                return await self._get_full_space_context(options)
            
            # Build contextual focused rendering: InnerSpace essentials + focused element
            context_parts = []
            
            # 1. InnerSpace Identity & Context
            innerspace_header = self._build_innerspace_identity_context()
            context_parts.append(innerspace_header)
            
            # 2. Essential InnerSpace Components (scratchpad, etc.)
            essential_components = await self._get_essential_innerspace_components(options)
            if essential_components:
                context_parts.append(essential_components)
            
            # 3. Focused Element Header
            focus_header = self._build_focused_context_header(focused_element, conversation_context)
            context_parts.append(focus_header)
            
            # 4. Focused Element Content
            focused_veil = self._get_element_veil(focused_element)
            if focused_veil:
                attention_requests = {}  # No global attention in focused mode
                focused_content = self._render_veil_node_to_string(focused_veil, attention_requests, options, indent=0)
                context_parts.append(focused_content)
            else:
                context_parts.append(f"[No content available for {focus_element_id}]")
            
            # Combine all parts
            full_focused_context = "\n\n".join(context_parts)
            
            logger.info(f"[{self.owner.id}] Generated CONTEXTUAL FOCUSED context: InnerSpace + {focus_element_id} ({len(full_focused_context)} chars)")
            return full_focused_context
            
        except Exception as e:
            logger.error(f"[{self.owner.id}] Error generating focused context for {focus_element_id}: {e}", exc_info=True)
            # Fallback to full context on error
            return await self._get_full_space_context(options)

    def _build_innerspace_identity_context(self) -> str:
        """Build essential InnerSpace identity and context."""
        # Get basic InnerSpace info
        space_name = getattr(self.owner, 'name', 'Unknown InnerSpace')
        space_id = self.owner.id
        
        # Try to get agent name from the space
        agent_name = getattr(self.owner, 'agent_name', None)
        if not agent_name and hasattr(self.owner, 'name'):
            agent_name = self.owner.name
        
        # Build identity context
        identity_parts = [
            f"[AGENT WORKSPACE] {space_name}",
        ]
        
        if agent_name:
            identity_parts.append(f"Agent: {agent_name}")
        
        identity_parts.append(f"Space Type: {self.owner.__class__.__name__}")
        
        # Add any InnerSpace-specific properties
        if hasattr(self.owner, 'description') and self.owner.description:
            identity_parts.append(f"Purpose: {self.owner.description}")
        
        return " | ".join(identity_parts)

    async def _get_essential_innerspace_components(self, options: Dict[str, Any]) -> Optional[str]:
        """
        Get essential InnerSpace components that should always be visible.
        
        Args:
            options: Rendering options
            
        Returns:
            Rendered essential components or None if none found
        """
        try:
            essential_parts = []
            
            # Look for scratchpad or similar essential components
            if hasattr(self.owner, '_flat_veil_cache'):
                flat_cache = self.owner._flat_veil_cache
                
                for veil_id, veil_node in flat_cache.items():
                    node_props = veil_node.get('properties', {})
                    node_type = veil_node.get('node_type', '')
                    content_nature = node_props.get('content_nature', '')
                    
                    # Include scratchpad content
                    if 'scratchpad' in node_type.lower() or 'scratchpad' in content_nature.lower():
                        scratchpad_content = self._render_veil_node_to_string(veil_node, {}, options, indent=0)
                        essential_parts.append(f"[SCRATCHPAD]\n{scratchpad_content}")
                    
                    # Include other essential components as needed
                    # Could add agent status, important state, etc.
            
            return "\n\n".join(essential_parts) if essential_parts else None
            
        except Exception as e:
            logger.warning(f"[{self.owner.id}] Error getting essential components: {e}")
            return None

    def _build_focused_context_header(self, focused_element, conversation_context: Dict[str, Any]) -> str:
        """Build a descriptive header for the focused conversation element."""
        element_name = getattr(focused_element, 'name', 'Unknown Element')
        element_type = focused_element.__class__.__name__
        
        header = f"[ACTIVE CONVERSATION] {element_name} ({element_type})"
        
        # Add conversation details if available
        if conversation_context:
            is_dm = conversation_context.get('is_dm', False)
            recent_sender = conversation_context.get('recent_sender')
            adapter_id = conversation_context.get('adapter_id')
            recent_preview = conversation_context.get('recent_message_preview', '')
            
            context_details = []
            if is_dm:
                context_details.append("Direct Message")
            else:
                context_details.append("Channel/Group")
                
            if recent_sender:
                context_details.append(f"Recent activity from: {recent_sender}")
                
            if adapter_id:
                context_details.append(f"Platform: {adapter_id}")
            
            if context_details:
                header += f" - {' | '.join(context_details)}"
                
            # Add preview of recent activity
            if recent_preview:
                preview = recent_preview[:80] + "..." if len(recent_preview) > 80 else recent_preview
                header += f"\nRecent: \"{preview}\""
        
        return header

    def _get_element_veil(self, element) -> Optional[Dict[str, Any]]:
        """
        Get the VEIL representation for a specific element.
        
        Args:
            element: The element to get VEIL data for
            
        Returns:
            VEIL node data for the element, or None if not available
        """
        try:
            # For elements with their own VEIL producer, get their VEIL directly
            if hasattr(element, 'get_components'):
                components = element.get_components()
                for component in components.values():
                    if hasattr(component, 'get_full_veil') and callable(component.get_full_veil):
                        return component.get_full_veil()
            
            # Fallback: try to get from space's flat cache
            if hasattr(self.owner, '_flat_veil_cache'):
                flat_cache = self.owner._flat_veil_cache
                # Look for VEIL nodes that belong to this element
                for veil_id, veil_node in flat_cache.items():
                    node_props = veil_node.get('properties', {})
                    if node_props.get('element_id') == element.id:
                        return veil_node
            
            # Last resort: create a basic VEIL representation
            return {
                "veil_id": f"{element.id}_basic",
                "node_type": "basic_element",
                "properties": {
                    "element_id": element.id,
                    "element_name": getattr(element, 'name', 'Unknown'),
                    "element_type": element.__class__.__name__,
                    "veil_status": "basic_representation"
                },
                "children": []
            }
            
        except Exception as e:
            logger.error(f"Error getting VEIL for element {element.id}: {e}", exc_info=True)
            return None

    async def _get_full_space_context(self, options: Dict[str, Any]) -> str:
        """
        Generate the full space context (original behavior).
        
        Args:
            options: Rendering options
            
        Returns:
            Full space context string
        """
        # 1. Get the full aggregated VEIL from InnerSpace's producer
        veil_producer = self._get_space_veil_producer()
        if not veil_producer:
            logger.error(f"[{self.owner.id}] Cannot generate context: SpaceVeilProducer not found.")
            return "Error: Could not retrieve internal state."

        try:
            full_veil = veil_producer.get_full_veil()
            if not full_veil:
                 logger.warning(f"[{self.owner.id}] SpaceVeilProducer returned empty VEIL.")
                 return "Current context is empty."
        except Exception as e:
            logger.error(f"[{self.owner.id}] Error getting full VEIL from SpaceVeilProducer: {e}", exc_info=True)
            return f"Error: Failed to retrieve internal state - {e}"

        # 2. Get attention signals (optional filtering)
        attention_requests = {} 

        # 3. Render the VEIL structure into a string
        try:
            # Render synchronously
            context_string = self._render_veil_node_to_string(full_veil, attention_requests, options, indent=0)
        except Exception as e:
            logger.error(f"[{self.owner.id}] Error rendering VEIL to string: {e}", exc_info=True)
            # Maybe return partial context or just the error?
            import json
            return f"Error rendering context: {e}\nRaw VEIL:\n{json.dumps(full_veil, indent=2)}"

        logger.info(f"Generated FULL agent context for {self.owner.id} (approx length: {len(context_string)}).")
        return context_string

    # --- Rendering Helpers --- 

    def _render_space_root(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent_str: str, node_info: str) -> str:
        """Renders the root node of a space VEIL."""
        props = node.get("properties", {})
        element_name = props.get("element_name", "Unknown Space")
        element_type = props.get("element_type", "Space")
        is_inner_space = props.get("is_inner_space", False)

        header = f"{indent_str}[Space: {element_name} ({element_type})]"
        if is_inner_space:
            header += " (Agent's Inner Space)"
        
        output = f"{header}{indent_str}" # Dynamic separator length
        # Children are handled by the main loop, so this renderer just provides the header.
        return output

    def _render_scratchpad_placeholder(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent_str: str, node_info: str) -> str:
        """Renders the placeholder for an empty scratchpad."""
        props = node.get("properties", {})
        text = props.get("text", "Scratchpad is empty.")
        return f"{indent_str}<em>{text}</em>"

    def _render_default(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent_str: str, node_info: str) -> str:
        """Default fallback renderer - shows type and dumps properties."""
        props = node.get("properties", {})
        output = f"{indent_str}>> Default Render for {node_info}:\n"
        for key, value in props.items():
            # Skip annotations we already used for dispatch/display
            if key in ["structural_role", "content_nature", "rendering_hint"]:
                 continue
            # Avoid rendering huge child lists embedded in properties
            if isinstance(value, (list, dict)) and len(str(value)) > 100:
                 output += f"{indent_str}  {key}: [complex data omitted]\n"
            else:
                 output += f"{indent_str}  {key}: {value}\n"
        return output

    def _render_container(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent_str: str, node_info: str) -> str:
        """Renders a container node with element information and tool availability."""
        props = node.get("properties", {})
        element_name = props.get("element_name", "Unknown Element")
        element_id = props.get("element_id", "unknown_id")
        content_nature = props.get("content_nature", "container")
        
        # NEW: Include tool targeting information
        available_tools = props.get("available_tools", [])
        tool_target_element_id = props.get("tool_target_element_id")
        
        # Build header with element information
        header = f"{indent_str}{element_name}:"
        
        # Add tool information if available
        if available_tools:
            # NEW: Check if this is a remote element accessed via uplink proxy
            prefix_element_id = self._determine_tool_prefix_element_id(element_id)
            
            # Create short prefix for display (same logic as agent loop)
            short_prefix = self._create_short_element_prefix(prefix_element_id)
            
            # NEW: Get chat prefix mappings if this is an uplink proxy
            chat_prefix_info = ""
            if prefix_element_id != element_id:  # This is a remote element accessed via uplink
                uplink_element = self._get_uplink_element_for_remote(element_id)
                if uplink_element and hasattr(uplink_element, '_remote_tool_provider_component'):
                    urtp = uplink_element._remote_tool_provider_component
                    if urtp and hasattr(urtp, 'get_chat_prefix_mappings'):
                        chat_mappings = urtp.get_chat_prefix_mappings()
                        if chat_mappings:
                            chat_prefix_info = f" (Chat prefixes: {', '.join(f'{p}={n}' for p, n in chat_mappings.items())})"
            
            prefixed_tools = [f"{short_prefix}__{tool}" for tool in available_tools]
            
            if tool_target_element_id and tool_target_element_id != element_id:
                header += f" [Tools: {', '.join(prefixed_tools)} → {tool_target_element_id}]"
            else:
                header += f" [Tools: {', '.join(prefixed_tools)}]"
                
            # If using uplink proxy prefix, add clarification with chat prefix info
            if prefix_element_id != element_id:
                header += f" (via uplink{chat_prefix_info})"
        
        output = f"{header}\n"
        
        # Add element ID for verbose mode
        use_verbose_tags = options.get("render_style") == "verbose_tags"
        if use_verbose_tags and element_id:
            output += f"{indent_str}  (Element ID: {element_id})\n"
        
        return output

    def _get_uplink_element_for_remote(self, remote_element_id: str):
        """Helper to find the uplink element that provides access to a remote element."""
        if not self.owner:
            return None
            
        mounted_elements = self.owner.get_mounted_elements()
        
        for mount_id, element in mounted_elements.items():
            # Check if this is an uplink proxy
            if hasattr(element, 'remote_space_id') and hasattr(element, '_remote_tool_provider_component'):
                # Check if this uplink proxy has remote tools from the remote_element_id
                remote_tool_provider = getattr(element, '_remote_tool_provider_component', None)
                if remote_tool_provider and hasattr(remote_tool_provider, '_raw_remote_tool_definitions'):
                    # Check if any remote tool definitions come from this remote_element_id
                    for tool_def in remote_tool_provider._raw_remote_tool_definitions:
                        if tool_def.get('provider_element_id') == remote_element_id:
                            return element
        return None

    def _determine_tool_prefix_element_id(self, element_id: str) -> str:
        """
        Determine which element ID should be used for tool prefixing.
        
        For remote elements accessed via uplink proxies, returns the uplink proxy ID.
        For local elements, returns the element ID itself.
        
        Args:
            element_id: The element ID from the VEIL node
            
        Returns:
            Element ID to use for generating tool prefixes
        """
        # Check if this element might be a remote element by looking for uplink proxies
        # that could be proxying tools from this element
        
        if not self.owner:
            return element_id
            
        # Get mounted elements from the InnerSpace
        mounted_elements = self.owner.get_mounted_elements()
        
        for mount_id, element in mounted_elements.items():
            # Check if this is an uplink proxy
            if hasattr(element, 'remote_space_id') and hasattr(element, '_remote_tool_provider_component'):
                # Check if this uplink proxy has remote tools from the element_id
                remote_tool_provider = getattr(element, '_remote_tool_provider_component', None)
                if remote_tool_provider and hasattr(remote_tool_provider, '_raw_remote_tool_definitions'):
                    # Check if any remote tool definitions come from this element_id
                    for tool_def in remote_tool_provider._raw_remote_tool_definitions:
                        if tool_def.get('provider_element_id') == element_id:
                            # This element's tools are proxied by this uplink
                            return element.id
        
        # If no uplink proxy found, use the element's own ID
        return element_id

    def _create_short_element_prefix(self, element_id: str) -> str:
        """
        Create a short prefix for tool names using shared utility for consistency with AgentLoop.
        
        Args:
            element_id: Full element ID
            
        Returns:
            Short prefix for display consistency that matches ^[a-zA-Z0-9_-]+$
        """
        return create_short_element_prefix(element_id)

    def _render_chat_message(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent_str: str, node_info: str) -> str:
        """Renders a chat message node cleanly."""
        props = node.get("properties", {})
        sender = props.get("sender_name", "Unknown")
        text = props.get("text_content", "")
        timestamp_iso_val = props.get("timestamp_iso", "") 
        is_edited = props.get("is_edited", False)
        original_external_id = props.get("external_id")
        message_status = props.get("message_status", "received")  # NEW: Get message status
        reactions = props.get("reactions", {})  # NEW: Get reactions

        formatted_timestamp = "[timestamp N/A]"
        if isinstance(timestamp_iso_val, (int, float)):
            try:
                dt_object = datetime.fromtimestamp(timestamp_iso_val, tz=timezone.utc)
                formatted_timestamp = dt_object.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                formatted_timestamp = "[invalid timestamp]"
        elif isinstance(timestamp_iso_val, str) and timestamp_iso_val:
            formatted_timestamp = timestamp_iso_val # If it's already a string, use as is

        # Simple chat format
        output = f"{indent_str}<timestamp>{formatted_timestamp}</timestamp> <sender>{sender}</sender>: <message>{text}</message>"
        if is_edited:
            output += " (edited)"
        
        # NEW: Add message status indicators for pending states
        if message_status == "pending_send":
            output += " ⏳"
        elif message_status == "pending_edit":
            output += " ✏️"
        elif message_status == "pending_delete":
            output += " 🗑️"
        elif message_status == "failed_to_send":
            output += " ❌"
        
        if original_external_id:
            output += f" [ext_id: {original_external_id}]"

        # NEW: Render reactions if present
        if reactions:
            reaction_parts = []
            for emoji, user_list in reactions.items():
                if user_list:  # Only show emojis that have users
                    # Filter out pending markers for display
                    actual_users = [u for u in user_list if not u.startswith("pending_")]
                    pending_users = [u for u in user_list if u.startswith("pending_")]
                    
                    user_count = len(actual_users)
                    if user_count > 0:
                        reaction_display = f"{emoji}:{user_count}"
                        if pending_users:
                            reaction_display += "⏳"  # Indicate pending reactions
                        reaction_parts.append(reaction_display)
            
            if reaction_parts:
                output += f" [{', '.join(reaction_parts)}]"

        # --- NEW: Render Attachment Metadata ---
        attachment_metadata = props.get("attachment_metadata", [])
        if attachment_metadata:
            output += "\n" # Newline before attachments section if text exists
            for idx, att_meta in enumerate(attachment_metadata):
                filename = att_meta.get('filename', att_meta.get('attachment_id', f'attachment_{idx+1}'))
                att_type = att_meta.get('attachment_type', 'unknown')
                # Check if content is available (from a child VEIL_ATTACHMENT_CONTENT_NODE_TYPE)
                # This requires the main renderer to pass child info or MessageListVeilProducer to flatten it.
                # For now, we just display metadata. The main renderer will handle child content nodes.
                output += f"{indent_str}  [Attachment: {filename} (Type: {att_type})]"
                # If we knew content was available here, we could add: (Content Available)
                output += "\n"
        else:
             output += "\n" # Ensure newline after message text even if no attachments

        # output += f" [@ {timestamp}]" # Optional timestamp
        # output += "\n" # Removed this as newlines are handled by attachments or final message newline
        return output

    def _render_uplink_summary(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent_str: str, node_info: str) -> str:
        """Renders a cleaner summary for an uplink node."""
        props = node.get("properties", {})
        remote_name = props.get("remote_space_name", "Unknown Space")
        remote_id = props.get("remote_space_id", "unknown_id")
        node_count = props.get("cached_node_count", 0)
        # Add specific title for uplink section
        output = f"{indent_str}Uplink to {remote_name} ({remote_id}):\n"
        output += f"{indent_str}  (Contains {node_count} cached items)\n"
        # Children (the cached items) will be rendered by the main loop
        return output

    def _render_attachment_content(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent_str: str, node_info: str) -> str:
        """Renders a placeholder for fetched attachment content."""
        props = node.get("properties", {})
        filename = props.get("filename", "unknown_file")
        content_type = props.get("content_nature", "unknown_type")
        attachment_id = props.get("attachment_id", "unknown_id")
        
        output = f"{indent_str}  >> Fetched Attachment: {filename} (Type: {content_type}, ID: {attachment_id}) - Content ready for processing by agent.\n"
        # For text-based content, a preview could be rendered here if available in props.
        # if "content_preview" in props:
        #    output += f"{indent_str}     Preview: {props['content_preview'][:100]}...\n"
        return output

    # --- Main Recursive Renderer (Dispatcher) --- 

    def _render_veil_node_to_string(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent: int = 0) -> str:
        """
        Recursively renders a VEIL node and its children into a string.
        Dispatches rendering based on node properties/annotations.
        Supports different rendering styles via options['render_style'].
        """
        indent_str = "  " * indent
        node_type = node.get("node_type", "unknown")
        props = node.get("properties", {})
        children = node.get("children", [])
        node_id = node.get("veil_id")
        external_id = props.get("external_id")

        # --- Determine Rendering Strategy --- 
        structural_role = props.get("structural_role")
        content_nature = props.get("content_nature")
        rendering_hint = props.get("rendering_hint")

        # Determine rendering style
        render_style = options.get("render_style", "verbose_tags") # Default to clean
        use_verbose_tags = (render_style == "verbose_tags")


        node_info = f"Type='{node_type}', Role='{structural_role}', Nature='{content_nature}', ID='{node_id}'"
        if external_id:
            node_info += f", ExternalID='{external_id}'"

        # Decide which renderer to use based on hints/type (Order matters)
        render_func = self._render_default # Default fallback
        
        if content_nature == "space_summary":
            render_func = self._render_space_root
        elif content_nature == "chat_message":
            render_func = self._render_chat_message
        elif node_type == "scratchpad_placeholder": # NEW DISPATCH RULE
            render_func = self._render_scratchpad_placeholder
        elif content_nature == "uplink_summary":
             render_func = self._render_uplink_summary
        elif node_type == "attachment_content_item": # From VEIL_ATTACHMENT_CONTENT_NODE_TYPE
             render_func = self._render_attachment_content
        elif structural_role == "container" or node_type == "message_list_container": # Treat message list container like other containers
             render_func = self._render_container
        # Add more dispatch rules here...
        # elif structural_role == "list_item": # Generic list item renderer?
             # render_func = self._render_list_item 
        # elif content_nature == "space_summary": # Maybe same as container?
             # render_func = self._render_container

        # --- Construct Output --- 
        output = "" 
        node_content_str = "" # Content generated by the specific renderer

        # Optional: Add opening tag if verbose style is enabled
        if use_verbose_tags:
             output += f"{indent_str}<{node_type}>\n"
             # Increase indent for content within tags
             content_indent_str = indent_str + "  "
        else:
             content_indent_str = indent_str # Use same indent for clean style

        # Call the chosen rendering function for the node's specific content
        try:
             node_content_str = render_func(node, attention, options, content_indent_str, node_info)
             # Prepend the content string to the main output
             output += node_content_str
        except Exception as render_err:
             logger.error(f"Error calling renderer {render_func.__name__} for node {node_id}: {render_err}", exc_info=True)
             # Add error message, respecting indent
             output += f"{content_indent_str}>> Error rendering content for {node_info}: {render_err}\n"

        # Check attention (append after content for clarity)
        if node_id in attention:
             # Respect indent based on style
             output += f"{content_indent_str}  *ATTENTION: {attention[node_id].get('reason', '')}*\n"

        # Render children recursively (always use the main dispatcher)
        # Only render children if the current node isn't a specific content type 
        # that shouldn't have its VEIL children rendered (like a chat message itself)
        if children and render_func not in []: # No longer excluding _render_chat_message
            rendered_children_output = ""
            children_to_render = children # TODO: Apply filtering/limiting here
            # TODO: Sort children based on timestamp or other properties?
            # Example Sort (if timestamp exists): 
            # children_to_render.sort(key=lambda c: c.get('properties', {}).get('timestamp', 0))
            
            for child_node in children_to_render:
                 # Pass options down for consistent rendering style
                 rendered_children_output += self._render_veil_node_to_string(child_node, attention, options, indent + 1)
            
            # Only add if children produced output (avoid empty <Children> sections)
            if rendered_children_output: 
                 # Add children output respecting style
                 # In verbose mode, children are naturally indented inside the parent tag
                 # In clean mode, they follow the parent's rendered content
                 output += rendered_children_output 
                 
        # Optional: Add closing tag if verbose style is enabled
        if use_verbose_tags:
             output += f"{indent_str}</{node_type}>\n"
                 
        return output # Return accumulated string

    async def process_llm_response(self, llm_response_text: str) -> List[Dict[str, Any]]:
        """
        Parses the LLM's response text to extract structured actions and content.

        Args:
            llm_response_text: The raw text output from the LLM.

        Returns:
            A list of action dictionaries. Each dictionary should conform to a 
            structure that AgentLoopComponent can dispatch.
            Example action structure:
            {
                "action_name": "some_tool_or_verb",
                "target_element_id": "element_id_to_act_on", (optional, for element-specific actions)
                "target_space_id": "space_id_of_target_element", (optional, defaults to InnerSpace)
                "target_module": "module_name_for_external_action", (optional, for adapter actions)
                "parameters": { "param1": "value1", ... } 
            }
        """
        logger.debug(f"[{self.owner.id}] HUD processing LLM response: {llm_response_text[:100]}...")
        extracted_actions = []

        # Attempt to parse the response as JSON
        # This is a basic first pass. More sophisticated parsing (e.g., for VEIL-structured actions)
        # or natural language parsing could be added later.
        try:
            import json
            parsed_response = json.loads(llm_response_text)
            
            if isinstance(parsed_response, dict) and "actions" in parsed_response:
                actions_from_llm = parsed_response.get("actions")
                if isinstance(actions_from_llm, list):
                    for llm_action in actions_from_llm:
                        if isinstance(llm_action, dict):
                            # Basic transformation: assume llm_action structure matches our desired format.
                            # More mapping/validation might be needed here.
                            # e.g., mapping "action_type" from LLM to "action_name"
                            action_to_dispatch = {}
                            action_to_dispatch["action_name"] = llm_action.get("action_type") # or get("action_name")
                            action_to_dispatch["parameters"] = llm_action.get("parameters", {})
                            
                            # Add target_element_id, target_space_id, or target_module if present
                            if "target_element_id" in llm_action:
                                action_to_dispatch["target_element_id"] = llm_action["target_element_id"]
                            if "target_space_id" in llm_action:
                                action_to_dispatch["target_space_id"] = llm_action["target_space_id"]
                            if "target_module" in llm_action:
                                action_to_dispatch["target_module"] = llm_action["target_module"]

                            if action_to_dispatch.get("action_name"): # Must have an action name
                                extracted_actions.append(action_to_dispatch)
                                logger.info(f"Extracted action: {action_to_dispatch}")
                            else:
                                logger.warning(f"Skipping LLM action due to missing action_name: {llm_action}")
                        else:
                            logger.warning(f"Skipping non-dict item in LLM actions list: {llm_action}")
                else:
                    logger.warning("LLM response JSON had 'actions' key, but it was not a list.")
            else:
                # No 'actions' key found, or not a dict. Try other parsing? For now, assume no actions.
                logger.debug("LLM response not a dict or no 'actions' key found. No structured actions extracted.")

            # TODO: Handle other parts of the LLM response, e.g., "response_to_user" or free text
            # This content might need to be placed into the DAG as a new message/event.

        except json.JSONDecodeError:
            logger.warning(f"LLM response was not valid JSON: {llm_response_text[:200]}...")
            # Fallback: Treat the whole response as a potential natural language command?
            # For now, we don't parse actions from non-JSON.
        except Exception as e:
            logger.error(f"Error processing LLM response in HUD: {e}", exc_info=True)

        if not extracted_actions:
            logger.info(f"[{self.owner.id}] No actions extracted from LLM response.")
        
        return extracted_actions

    # Optional summarization helper
    # def _summarize_context(self, context: str) -> str: ...

    # Other potential methods:
    # - get_focused_context(element_id)
    # - get_context_summary()
