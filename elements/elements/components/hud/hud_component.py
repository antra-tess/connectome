import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union
from datetime import datetime, timezone
import copy
import json

from ..base_component import Component
# Needs access to the SpaceVeilProducer on the owner (InnerSpace)
from ..space.space_veil_producer import SpaceVeilProducer
# Import the registry decorator
from elements.component_registry import register_component
# Import shared prefix generator for consistency with AgentLoop
from ...utils.prefix_generator import create_short_element_prefix
# May need access to GlobalAttentionComponent for filtering
# from ..attention.global_attention_component import GlobalAttentionComponent

from opentelemetry import trace
from host.observability import get_tracer

if TYPE_CHECKING:
    # May need LLM provider for summarization/rendering assistance
    from llm.provider_interface import LLMProviderInterface

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

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

    async def get_agent_context(self, options: Optional[Dict[str, Any]] = None) -> Union[str, Dict[str, Any]]:
        """
        Generate agent context using modular rendering functions.

        Args:
            options: Optional dictionary controlling which rendering function to use:
                     - render_type: 'memory', 'focused', 'full', or 'combined' (default: 'combined')
                     - focus_element_id: Element ID for focused rendering
                     - memory_context: Memory data for memory rendering
                     - Other rendering-specific options

        Returns:
            A string representing the agent's context (if no live multimodal content)
            OR a dictionary with 'text' and 'attachments' keys (if live multimodal content detected)
        """
        logger.debug(f"Generating agent context for {self.owner.id}...")
        options = options or {}

        render_type = options.get('render_type', 'combined')  # Changed default from 'full' to 'combined'

        # Set default render_style to use rich tags for better structure
        if 'render_style' not in options:
            options['render_style'] = 'verbose_tags'

        try:
            if render_type == 'memory':
                context = await self.render_memory_frame_part(options)
            elif render_type == 'focused':
                context = await self.render_focused_frame_part(options)
            elif render_type == 'full':
                context = await self.render_full_frame(options)
            elif render_type == 'combined':
                context = await self.render_combined_frame(options)
            else:
                logger.warning(f"Unknown render_type '{render_type}', falling back to full rendering")
                context = await self.render_full_frame(options)

            # FIXED: Auto-detect multimodal content for all render types
            # (render_combined_frame already handles this internally, so check if it's a dict)
            if isinstance(context, dict):
                # Combined frame already detected and structured multimodal content
                return context
            elif isinstance(context, str):
                # Other render types return strings, check for live multimodal content
                focus_element_id = options.get('focus_element_id')
                if focus_element_id and self.owner:
                    # OPTIMIZED: Single-pass detection and extraction instead of separate operations
                    return await self._detect_and_extract_multimodal_content(context, options, focus_element_id)

                # No multimodal content detected, return text
                return context
            else:
                # Fallback for unexpected types
                return context

        except Exception as e:
            logger.error(f"Error in get_agent_context with render_type '{render_type}': {e}", exc_info=True)
            # Fallback to basic full rendering
            fallback_context = await self.render_full_frame({})
            return fallback_context

    async def _extract_multimodal_content(self, text_context: str, options: Dict[str, Any], focus_element_id: str) -> Dict[str, Any]:
        """
        Extract multimodal content from VEIL and return structured data.

        OPTIMIZED: Uses targeted element filtering when focus_element_id is provided,
        then recursively searches through those nodes for maximum efficiency.

        Args:
            text_context: The rendered text context
            options: Rendering options
            focus_element_id: Element ID to search for attachments

        Returns:
            Dictionary with 'text' and 'attachments' keys
        """
        try:
            if not self.owner:
                logger.warning(f"Cannot extract multimodal content: No owner element.")
                return {"text": text_context, "attachments": []}

            # OPTIMIZED: Use provided focus element for targeted search
            if focus_element_id:
                # Use targeted search for the focused element
                attachment_nodes = self._find_attachment_nodes_for_element(focus_element_id)
                logger.info(f"[{self.owner.id}] Using targeted search for element {focus_element_id}")
            else:
                # Fall back to full VEIL search if no focus element
                attachment_nodes = self._find_attachment_nodes_recursively()
                logger.info(f"[{self.owner.id}] Using full VEIL search (no focus element)")

            # Process attachment nodes into LiteLLM-compatible format
            processed_attachments = []
            for attachment_node in attachment_nodes:
                processed_attachment = await self._process_attachment_node_for_llm(attachment_node)
                if processed_attachment:
                    processed_attachments.append(processed_attachment)

            logger.info(f"[{self.owner.id}] Extracted {len(processed_attachments)} multimodal attachments using optimized search")

            return {
                "text": text_context,
                "attachments": processed_attachments
            }

        except Exception as e:
            logger.error(f"[{self.owner.id}] Error extracting multimodal content: {e}", exc_info=True)
            return {"text": text_context, "attachments": []}

    def _find_attachment_nodes_for_element(self, element_id: str) -> List[Dict[str, Any]]:
        """
        NEW: Efficiently find attachment nodes for a specific element using granular filtering.

        This is much more efficient than searching the entire VEIL when we know which element
        we're interested in.

        Args:
            element_id: The element ID to search for attachments

        Returns:
            List of attachment content node dictionaries
        """
        try:
            attachment_nodes = []

            # Get only the VEIL nodes for this specific element
            element_nodes = self.owner.get_veil_nodes_by_owner(element_id)
            if not element_nodes:
                logger.debug(f"No VEIL nodes found for element {element_id}")
                return []

            # Recursively search through this element's nodes
            def search_node_for_attachments(node: Dict[str, Any]):
                if not isinstance(node, dict):
                    return

                # Check if this node is an attachment content item
                node_type = node.get("node_type", "")
                props = node.get("properties", {})
                structural_role = props.get("structural_role", "")

                if (node_type == "attachment_content_item" or
                    structural_role == "attachment_content"):
                    # Found an attachment node!
                    attachment_nodes.append(node)
                    logger.debug(f"Found attachment node {node.get('veil_id')} for element {element_id}")

                # Recursively search children
                children = node.get("children", [])
                for child in children:
                    search_node_for_attachments(child)

            # Search through all nodes owned by this element
            for veil_id, node_data in element_nodes.items():
                search_node_for_attachments(node_data)

            logger.debug(f"Found {len(attachment_nodes)} attachment nodes for element {element_id}")
            return attachment_nodes

        except Exception as e:
            logger.error(f"Error finding attachment nodes for element {element_id}: {e}", exc_info=True)
            return []

    def _find_attachment_nodes_recursively(self) -> List[Dict[str, Any]]:
        """
        NEW: Recursively search through the entire VEIL hierarchy to find attachment content nodes.

        This handles the case where attachment nodes are children of message nodes rather than
        top-level nodes in the flat VEIL cache.

        Returns:
            List of attachment content node dictionaries
        """
        try:
            attachment_nodes = []

            # Get the full VEIL from SpaceVeilProducer to search through hierarchy
            veil_producer = self._get_space_veil_producer()
            if not veil_producer:
                logger.warning(f"Cannot find attachments: SpaceVeilProducer not available")
                return []

            full_veil = veil_producer.get_full_veil()
            if not full_veil:
                logger.debug(f"No VEIL available for attachment search")
                return []

            # Recursively search through VEIL hierarchy
            def search_node_for_attachments(node: Dict[str, Any]):
                if not isinstance(node, dict):
                    return

                # Check if this node is an attachment content item
                node_type = node.get("node_type", "")
                props = node.get("properties", {})
                structural_role = props.get("structural_role", "")

                if (node_type == "attachment_content_item" or
                    structural_role == "attachment_content"):
                    # Found an attachment node!
                    attachment_nodes.append(node)
                    logger.debug(f"Found attachment node {node.get('veil_id')} at depth")

                # Recursively search children
                children = node.get("children", [])
                for child in children:
                    search_node_for_attachments(child)

            # Start recursive search from root
            search_node_for_attachments(full_veil)

            logger.debug(f"Found {len(attachment_nodes)} attachment nodes through recursive search")
            return attachment_nodes

        except Exception as e:
            logger.error(f"Error in recursive attachment search: {e}", exc_info=True)
            return []

    async def _process_attachment_node_for_llm(self, attachment_node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process an attachment VEIL node into LiteLLM-compatible format.

        Args:
            attachment_node: VEIL node representing attachment content

        Returns:
            LiteLLM-compatible content part or None if not processable
        """
        try:
            props = attachment_node.get("properties", {})
            content_type = props.get("content_nature", "unknown")
            filename = props.get("filename", "unknown_file")
            attachment_id = props.get("attachment_id")

            # NEW: Get content directly from VEIL node properties (much simpler!)
            content = props.get("content")
            content_available = props.get("content_available", False)

            if "image" in content_type.lower():
                # For images, we need base64 data
                logger.debug(f"Processing image attachment: {filename}")

                if content and isinstance(content, str):
                    # Content is available directly in VEIL node
                    return {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{content_type.split('/')[-1] if '/' in content_type else 'png'};base64,{content}"
                        }
                    }
                else:
                    logger.warning(f"No content available for image attachment {filename} (content_available: {content_available})")
                    return None

            elif "text" in content_type.lower():
                # For text files, include the content directly
                logger.debug(f"Processing text attachment: {filename}")

                if content:
                    return {
                        "type": "text",
                        "text": f"[Attachment: {filename}]\n{content}"
                    }
                else:
                    logger.warning(f"No content available for text attachment {filename} (content_available: {content_available})")
                    return None
            else:
                # For other file types, just mention them
                logger.debug(f"Unsupported attachment type for LLM: {content_type}")
                return {
                    "type": "text",
                    "text": f"[Attachment: {filename} (Type: {content_type}) - Content not directly viewable]"
                }

        except Exception as e:
            logger.error(f"Error processing attachment node for LLM: {e}", exc_info=True)
            return None

    async def render_memory_frame_part(self, options: Dict[str, Any]) -> str:
        """
        Render memory context part of the frame.


        Args:
            options: Rendering options, should include 'memory_context' with memory data

        Returns:
            Rendered memory frame part
        """
        try:
            memory_context = options.get('memory_context')
            if not memory_context:
                return "[MEMORY]\n(No memory context provided)"

            frame_parts = []

            # Agent Workspace Header
            workspace_section = self._render_workspace_section(memory_context.get('agent_info', {}))
            if workspace_section:
                frame_parts.append(workspace_section)

            # Scratchpad Section
            scratchpad_section = self._render_scratchpad_section(memory_context.get('scratchpad'))
            if scratchpad_section:
                frame_parts.append(scratchpad_section)

            # Compressed Context Section
            compressed_section = self._render_compressed_context_section(memory_context.get('compressed_context', {}))
            if compressed_section:
                frame_parts.append(compressed_section)

            # No more latest reasoning section - removed in Phase 2 cleanup

            memory_frame = "\n\n".join(frame_parts)
            logger.info(f"[{self.owner.id}] Rendered memory frame: {len(memory_frame)} chars, {len(frame_parts)} sections")
            return memory_frame

        except Exception as e:
            logger.error(f"[{self.owner.id}] Error rendering memory frame: {e}", exc_info=True)
            return "[MEMORY]\n(Error rendering memory context)"

    async def render_focused_frame_part(self, options: Dict[str, Any]) -> str:
        """
        Render focused element with essential InnerSpace context.

        Args:
            options: Rendering options, should include 'focus_element_id' and optionally 'conversation_context'

        Returns:
            Rendered focused frame part
        """
        try:
            focus_element_id = options.get('focus_element_id')
            if not focus_element_id:
                logger.warning(f"[{self.owner.id}] No focus_element_id provided for focused rendering")
                return await self.render_full_frame(options)

            conversation_context = options.get('conversation_context', {})

            # Find the focused element
            focused_element = None
            if focus_element_id == self.owner.id:
                logger.info(f"[{self.owner.id}] Focus target is InnerSpace itself, using full context")
                return await self.render_full_frame(options)
            else:
                mounted_elements = self.owner.get_mounted_elements() if hasattr(self.owner, 'get_mounted_elements') else {}
                for mount_id, element in mounted_elements.items():
                    if element.id == focus_element_id or mount_id == focus_element_id:
                        focused_element = element
                        break

            if not focused_element:
                logger.warning(f"[{self.owner.id}] Focused element {focus_element_id} not found, falling back to full context")
                return await self.render_full_frame(options)

            # Build focused rendering: InnerSpace essentials + focused element
            context_parts = []

            # InnerSpace Identity & Context
            innerspace_header = self._build_innerspace_identity_context()
            context_parts.append(innerspace_header)

            # Essential InnerSpace Components
            essential_components = await self._get_essential_innerspace_components(options)
            if essential_components:
                context_parts.append(essential_components)

            # Focused Element Header
            focus_header = self._build_focused_context_header(focused_element, conversation_context)
            context_parts.append(focus_header)

            # Focused Element Content
            focused_veil = self._get_element_veil(focused_element)
            if focused_veil:
                attention_requests = {}  # No global attention in focused mode
                focused_content = self._render_veil_node_to_string(focused_veil, attention_requests, options, indent=0)
                context_parts.append(focused_content)
            else:
                context_parts.append(f"[No content available for {focus_element_id}]")

            focused_frame = "\n\n".join(context_parts)
            logger.info(f"[{self.owner.id}] Rendered focused frame: InnerSpace + {focus_element_id} ({len(focused_frame)} chars)")
            return focused_frame

        except Exception as e:
            logger.error(f"[{self.owner.id}] Error rendering focused frame: {e}", exc_info=True)
            return await self.render_full_frame(options)

    async def render_full_frame(self, options: Dict[str, Any]) -> str:
        """
        Render complete space context with all VEIL objects.

        Args:
            options: Rendering options (render_style, etc.)

        Returns:
            Rendered full frame
        """
        try:
            # Get the full aggregated VEIL from InnerSpace's producer
            veil_producer = self._get_space_veil_producer()
            if not veil_producer:
                logger.error(f"[{self.owner.id}] Cannot generate context: SpaceVeilProducer not found.")
                return "Error: Could not retrieve internal state."

            full_veil = veil_producer.get_full_veil()
            if not full_veil:
                logger.warning(f"[{self.owner.id}] SpaceVeilProducer returned empty VEIL.")
                return "Current context is empty."

            # Get attention signals
            attention_requests = {}

            # Render the VEIL structure into a string
            context_string = self._render_veil_node_to_string(full_veil, attention_requests, options, indent=0)

            logger.info(f"[{self.owner.id}] Rendered full frame: {len(context_string)} chars")
            return context_string

        except Exception as e:
            logger.error(f"[{self.owner.id}] Error rendering full frame: {e}", exc_info=True)
            import json
            return f"Error rendering context: {e}\nRaw VEIL:\n{json.dumps(full_veil, indent=2) if 'full_veil' in locals() else 'N/A'}"

    async def render_combined_frame(self, options: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
        """
        Render combined frame with memory + focused element.

        Args:
            options: Rendering options, should include 'memory_context' and 'focus_element_id'

        Returns:
            Rendered combined frame (string or dict with multimodal content)
        """
        try:
            frame_parts = []

            # Memory part
            memory_frame = await self.render_memory_frame_part(options)
            if memory_frame:
                frame_parts.append(memory_frame)

            # Focused part
            focused_frame = await self.render_focused_frame_part(options)
            if focused_frame:
                frame_parts.append(focused_frame)

            combined_text = "\n\n".join(frame_parts)

            # OPTIMIZED: Single pass detection + extraction instead of two separate searches
            focus_element_id = options.get('focus_element_id')
            if focus_element_id and self.owner:
                # Single efficient operation: detect AND extract if found
                return await self._detect_and_extract_multimodal_content(combined_text, options, focus_element_id)
            else:
                # No focus element - return text only
                logger.info(f"[{self.owner.id}] Rendered combined frame: {len(combined_text)} chars, {len(frame_parts)} parts")
                return combined_text

        except Exception as e:
            logger.error(f"[{self.owner.id}] Error rendering combined frame: {e}", exc_info=True)
            # Fallback to full frame
            return await self.render_full_frame(options)

    async def _detect_and_extract_multimodal_content(self, text_context: str, options: Dict[str, Any], focus_element_id: str) -> Union[str, Dict[str, Any]]:
        """
        OPTIMIZED: Single-pass multimodal detection and extraction.

        Combines detection and extraction into one efficient operation, eliminating
        the need for separate detection + extraction searches.

        Args:
            text_context: The rendered text context
            options: Rendering options
            focus_element_id: Element ID to check for live attachments

        Returns:
            Text string if no multimodal content, or dict with 'text' and 'attachments' if multimodal
        """
        try:
            # Single search operation - find attachments and process them if found
            attachment_nodes = self._find_attachment_nodes_for_element(focus_element_id)

            if not attachment_nodes:
                # No multimodal content found - return text only
                logger.debug(f"No live attachment content found for element {focus_element_id}")
                logger.info(f"[{self.owner.id}] Rendered combined frame: {len(text_context)} chars (text-only)")
                return text_context

            # Multimodal content found - process attachments immediately
            logger.info(f"[{self.owner.id}] Live multimodal content detected, processing {len(attachment_nodes)} attachments")

            # Process attachment nodes into LiteLLM-compatible format
            processed_attachments = []
            for attachment_node in attachment_nodes:
                processed_attachment = await self._process_attachment_node_for_llm(attachment_node)
                if processed_attachment:
                    processed_attachments.append(processed_attachment)

            logger.info(f"[{self.owner.id}] Extracted {len(processed_attachments)} multimodal attachments in single pass")

            return {
                "text": text_context,
                "attachments": processed_attachments
            }

        except Exception as e:
            logger.error(f"[{self.owner.id}] Error in combined multimodal detection/extraction: {e}", exc_info=True)
            # Fallback to text-only on error
            return text_context

    def _has_live_multimodal_content_for_element(self, element_id: str) -> bool:
        """
        OPTIMIZED: Check if element has LIVE multimodal content using granular filtering + recursive search.

        This only looks for fresh attachment content, not compressed memory descriptions.
        Compressed memories containing attachment descriptions are rendered as text.

        OPTIMIZED: First filters VEIL nodes by element_id, then recursively searches only
        through those nodes for efficiency.

        Args:
            element_id: Element ID to check for live attachments

        Returns:
            True if live attachment nodes are found
        """
        try:
            if not self.owner:
                return False

            # OPTIMIZED: First get only the VEIL nodes for this specific element
            element_nodes = self.owner.get_veil_nodes_by_owner(element_id)
            if not element_nodes:
                logger.debug(f"No VEIL nodes found for element {element_id}")
                return False

            # OPTIMIZED: Recursively search only through this element's nodes
            def search_element_nodes_for_attachments(nodes_dict: Dict[str, Any]) -> bool:
                for veil_id, node_data in nodes_dict.items():
                    if not isinstance(node_data, dict):
                        continue

                    # Check if this node is an attachment content item
                    node_type = node_data.get("node_type", "")
                    props = node_data.get("properties", {})
                    structural_role = props.get("structural_role", "")
                    content_nature = props.get("content_nature", "")

                    # If this is an attachment node
                    if (node_type == "attachment_content_item" or
                        structural_role == "attachment_content" or
                        content_nature.startswith("image") or
                        content_nature == "attachment_content"):

                        # Make sure this is NOT a compressed memory describing attachments
                        if node_type not in ["content_memory", "memorized_content"]:
                            logger.debug(f"Found live attachment node {veil_id} for element {element_id}")
                            return True

                    # OPTIMIZED: Also check if this node has children with attachments
                    if self._node_has_attachment_children(node_data):
                        return True

                return False

            # Search through the element's nodes
            has_attachments = search_element_nodes_for_attachments(element_nodes)

            if not has_attachments:
                logger.debug(f"No live attachment content found for element {element_id}")

            return has_attachments

        except Exception as e:
            logger.error(f"Error checking live multimodal content for element {element_id}: {e}", exc_info=True)
            return False

    def _node_has_attachment_children(self, node: Dict[str, Any]) -> bool:
        """
        NEW: Recursively check if a node has attachment children.

        This efficiently searches through a node's children hierarchy to find attachment nodes.

        Args:
            node: VEIL node to search through

        Returns:
            True if attachment children are found
        """
        try:
            children = node.get("children", [])

            for child in children:
                if not isinstance(child, dict):
                    continue

                # Check if this child is an attachment
                child_node_type = child.get("node_type", "")
                child_props = child.get("properties", {})
                child_structural_role = child_props.get("structural_role", "")
                child_content_nature = child_props.get("content_nature", "")

                if (child_node_type == "attachment_content_item" or
                    child_structural_role == "attachment_content" or
                    child_content_nature.startswith("image") or
                    child_content_nature == "attachment_content"):

                    # Make sure this is NOT a compressed memory
                    if child_node_type not in ["content_memory", "memorized_content"]:
                        logger.debug(f"Found attachment child: {child.get('veil_id')}")
                        return True

                # Recursively check this child's children
                if self._node_has_attachment_children(child):
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking node children for attachments: {e}", exc_info=True)
            return False

    def _render_workspace_section(self, agent_info: Dict[str, Any]) -> str:
        """Render the agent workspace section."""
        agent_name = agent_info.get('agent_name', 'Unknown Agent')
        workspace_desc = agent_info.get('workspace_description', 'Agent workspace')
        total_interactions = agent_info.get('total_stored_interactions', 0)

        return f"[AGENT WORKSPACE]\n{workspace_desc}\nAgent: {agent_name} | Total interactions in memory: {total_interactions}"

    def _render_scratchpad_section(self, scratchpad_data: Optional[Dict[str, Any]]) -> Optional[str]:
        """Render the scratchpad section if data exists."""
        if not scratchpad_data:
            return "[SCRATCHPAD]\n(Empty - no scratchpad data available)"

        # TODO: Implement actual scratchpad rendering when scratchpad functionality is added
        return "[SCRATCHPAD]\n(Scratchpad functionality not yet implemented)"

    def _render_compressed_context_section(self, compressed_data: Dict[str, Any]) -> Optional[str]:
        """Render compressed context of other conversations/interactions."""
        interaction_count = compressed_data.get('interaction_count', 0)

        if interaction_count == 0:
            return "[COMPRESSED CONTEXT]\n(No previous interactions to reference)"

        summary = compressed_data.get('summary', f"I have {interaction_count} previous interactions.")
        recent_interactions = compressed_data.get('recent_interactions', [])

        context_parts = [f"[COMPRESSED CONTEXT]\n{summary}"]

        if recent_interactions:
            context_parts.append("Recent interactions:")
            for interaction in recent_interactions:
                interaction_summary = interaction.get('summary', 'Unknown context')
                tool_count = interaction.get('tool_calls_made', 0)
                interaction_index = interaction.get('interaction_index', '?')

                tools_text = f" (used {tool_count} tools)" if tool_count > 0 else ""
                context_parts.append(f"  {interaction_index}. {interaction_summary}{tools_text}")

        return "\n".join(context_parts)

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
        Get the VEIL representation for a specific element using granular filtering.

        Args:
            element: The element to get VEIL data for

        Returns:
            VEIL node data for the element with properly reconstructed hierarchy, or None if not available
        """
        try:
            # NEW: Use granular filtering by owner_id for much better performance
            if not self.owner:
                logger.warning(f"Cannot get element VEIL: No owner InnerSpace")
                return None

            # Get all VEIL nodes belonging to this element
            element_nodes = self.owner.get_veil_nodes_by_owner(element.id)

            if not element_nodes:
                logger.debug(f"No VEIL nodes found for element {element.id}")
                return None

            # Find the root/container node for this element
            container_nodes = {
                veil_id: node_data
                for veil_id, node_data in element_nodes.items()
                if isinstance(node_data, dict) and
                node_data.get("properties", {}).get("structural_role") == "container"
            }

            if container_nodes:
                # Return the first container node found (should typically be only one)
                container_veil_id, container_node = next(iter(container_nodes.items()))

                # NEW: Properly reconstruct hierarchy using parent-child relationships
                # Instead of just flattening children, reconstruct the proper hierarchy
                reconstructed_container = copy.deepcopy(container_node)

                # Build hierarchy by finding child nodes and organizing them by parent_id
                child_nodes_by_parent = {}
                root_children = []

                for veil_id, node_data in element_nodes.items():
                    if veil_id == container_veil_id:
                        continue  # Skip the container itself

                    if isinstance(node_data, dict):
                        props = node_data.get("properties", {})
                        parent_id = props.get("parent_id")

                        # If parent_id points to the container, it's a direct child
                        if parent_id == container_veil_id:
                            root_children.append(copy.deepcopy(node_data))
                        elif parent_id:
                            # Group by parent for sub-hierarchy
                            if parent_id not in child_nodes_by_parent:
                                child_nodes_by_parent[parent_id] = []
                            child_nodes_by_parent[parent_id].append(copy.deepcopy(node_data))
                        else:
                            # No parent_id, treat as direct child
                            root_children.append(copy.deepcopy(node_data))

                # NEW: Recursively attach children to their parents to maintain hierarchy
                def attach_children_recursively(node):
                    node_veil_id = node.get("veil_id")
                    if node_veil_id in child_nodes_by_parent:
                        node["children"] = child_nodes_by_parent[node_veil_id]
                        # Recursively process the children
                        for child in node["children"]:
                            attach_children_recursively(child)

                # Attach children to root-level nodes
                for child in root_children:
                    attach_children_recursively(child)

                # Set the properly organized children on the container
                reconstructed_container["children"] = root_children

                logger.debug(f"Retrieved VEIL for element {element.id}: container + {len(root_children)} root children (hierarchy preserved)")
                return reconstructed_container
            else:
                # No container found, create a synthetic one from all nodes but still preserve hierarchy
                logger.debug(f"No container node found for element {element.id}, creating synthetic structure with hierarchy")

                # Try to build hierarchy from all available nodes
                nodes_by_veil_id = {veil_id: copy.deepcopy(node_data) for veil_id, node_data in element_nodes.items()}
                root_nodes = []

                # Find root nodes (no parent or parent not in this element's nodes)
                for veil_id, node_data in nodes_by_veil_id.items():
                    props = node_data.get("properties", {})
                    parent_id = props.get("parent_id")

                    if not parent_id or parent_id not in nodes_by_veil_id:
                        root_nodes.append(node_data)

                # Build hierarchy recursively
                def build_synthetic_hierarchy(node):
                    node_veil_id = node.get("veil_id")
                    children = []

                    for other_veil_id, other_node in nodes_by_veil_id.items():
                        if other_veil_id != node_veil_id:
                            other_props = other_node.get("properties", {})
                            if other_props.get("parent_id") == node_veil_id:
                                child_with_hierarchy = build_synthetic_hierarchy(other_node)
                                children.append(child_with_hierarchy)

                    node["children"] = children
                    return node

                # Build hierarchy for all root nodes
                hierarchical_roots = [build_synthetic_hierarchy(root) for root in root_nodes]

                return {
                    "veil_id": f"{element.id}_synthetic_root",
                    "node_type": "synthetic_container",
                    "properties": {
                        "element_id": element.id,
                        "element_name": getattr(element, 'name', 'Unknown'),
                        "element_type": element.__class__.__name__,
                        "veil_status": "synthetic_from_granular_filtering_with_hierarchy"
                    },
                    "children": hierarchical_roots
                }

        except Exception as e:
            logger.error(f"Error getting VEIL for element {element.id}: {e}", exc_info=True)
            return None

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
        element_type = props.get("element_type", "Element")

        # NEW: Include tool targeting information
        available_tools = props.get("available_tools", [])
        tool_target_element_id = props.get("tool_target_element_id")

        # Build header with element information - always include element type and ID for future needs
        header = f"{indent_str}{element_name}"

        # NEW: Always show element ID and type (important for targeting and debugging)
        header += f" (ID: {element_id})"
        if element_type != "Element":  # Only show type if it's not generic
            header += f" [{element_type}]"

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
                header += f" - Tools: {', '.join(prefixed_tools)} → {tool_target_element_id}"
            else:
                header += f" - Tools: {', '.join(prefixed_tools)}"

            # If using uplink proxy prefix, add clarification with chat prefix info
            if prefix_element_id != element_id:
                header += f" (via uplink{chat_prefix_info})"

        header += ":"  # End with colon for readability
        output = f"{header}\n"

        # Add additional details in verbose mode
        use_verbose_tags = options.get("render_style") == "verbose_tags"
        if use_verbose_tags:
            if content_nature and content_nature != "container":
                output += f"{indent_str}  (Content Nature: {content_nature})\n"
            if tool_target_element_id and tool_target_element_id != element_id:
                output += f"{indent_str}  (Tool Target: {tool_target_element_id})\n"

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
        error_details = props.get("error_details", None)

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

        if error_details:
            output += f" [error: {error_details}]"

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
                # For now, we'll just display metadata. The main renderer will handle child content nodes.
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
        """Renders attachment content with preview when available."""
        props = node.get("properties", {})
        filename = props.get("filename", "unknown_file")
        content_type = props.get("content_nature", "unknown_type")
        attachment_id = props.get("attachment_id", "unknown_id")

        output = f"{indent_str}  >> Attachment: {filename} (Type: {content_type}, ID: {attachment_id})\n"

        # Try to get actual content for preview
        try:
            # This is a synchronous preview - for async operations, we'd need a different approach
            # For now, we'll just indicate that content is available
            if props.get("content_available", False):
                if "image" in content_type.lower():
                    output += f"{indent_str}     [Image content available for multimodal processing]\n"
                elif "text" in content_type.lower():
                    # For text, we could show a preview if content is small enough
                    output += f"{indent_str}     [Text content available for LLM processing]\n"
                else:
                    output += f"{indent_str}     [Content available but not directly viewable]\n"
            else:
                output += f"{indent_str}     [Content not yet fetched or unavailable]\n"
        except Exception as e:
            logger.warning(f"Error checking attachment content availability: {e}")
            output += f"{indent_str}     [Content status unknown]\n"

        return output

    def _render_memory_context(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent_str: str, node_info: str) -> str:
        """Renders memory context nodes injected by CompressionEngine."""
        props = node.get("properties", {})
        memory_type = props.get("memory_type", "unknown")

        if memory_type == "workspace_info":
            agent_name = props.get("agent_name", "Unknown")
            workspace_desc = props.get("workspace_description", "")
            total_interactions = props.get("total_interactions", 0)

            output = f"{indent_str}[AGENT WORKSPACE]\n"
            output += f"{indent_str}{workspace_desc}\n"
            output += f"{indent_str}Agent: {agent_name} | Total interactions in memory: {total_interactions}\n"

        elif memory_type == "compressed_context":
            interaction_count = props.get("interaction_count", 0)
            summary = props.get("summary", "")
            recent_interactions = props.get("recent_interactions", [])

            output = f"{indent_str}[COMPRESSED CONTEXT]\n"
            output += f"{indent_str}{summary}\n"

            if recent_interactions:
                output += f"{indent_str}Recent interactions:\n"
                for interaction in recent_interactions:
                    interaction_summary = interaction.get('summary', 'Unknown context')
                    tool_count = interaction.get('tool_calls_made', 0)
                    interaction_index = interaction.get('interaction_index', '?')

                    tools_text = f" (used {tool_count} tools)" if tool_count > 0 else ""
                    output += f"{indent_str}  {interaction_index}. {interaction_summary}{tools_text}\n"
        else:
            # Generic memory context (no more reasoning chain support)
            output = f"{indent_str}[MEMORY: {memory_type.upper()}]\n"
            output += f"{indent_str}(Memory context of type {memory_type})\n"

        return output

    def _render_content_memory(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent_str: str, node_info: str) -> str:
        """Renders compressed content memory summaries created by CompressionEngine."""
        props = node.get("properties", {})
        memory_summary = props.get("memory_summary", "Content summary unavailable")
        original_element_id = props.get("original_element_id", "unknown")
        original_child_count = props.get("original_child_count", 0)

        output = f"{indent_str}<content_memory>{memory_summary}</content_memory>\n"

        # Add optional compression details in verbose mode
        use_verbose_tags = options.get("render_style") == "verbose_tags"
        if use_verbose_tags:
            output += f"{indent_str}  (Compressed {original_child_count} items from {original_element_id})\n"

        return output

    def _render_memorized_content(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent_str: str, node_info: str) -> str:
        """Renders memorized content nodes created by MemoryCompressor."""
        props = node.get("properties", {})
        memory_summary = props.get("memory_summary", "Memory content unavailable")
        memory_id = props.get("memory_id", "unknown")
        original_element_ids = props.get("original_element_ids", [])
        original_node_count = props.get("original_node_count", 0)
        token_count = props.get("token_count", 0)
        compression_timestamp = props.get("compression_timestamp", "")
        compressor_type = props.get("compressor_type", "unknown")

        output = f"{indent_str}<memorized_content>{memory_summary}</memorized_content>\n"

        # Add optional compression details in verbose mode
        use_verbose_tags = options.get("render_style") == "verbose_tags"
        if use_verbose_tags:
            element_info = ", ".join(original_element_ids) if original_element_ids else "unknown elements"
            output += f"{indent_str}  (Memory {memory_id}: {original_node_count} nodes from {element_info}, {token_count} tokens, {compressor_type})\n"
            if compression_timestamp:
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(compression_timestamp.replace('Z', '+00:00'))
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    output += f"{indent_str}  (Compressed: {formatted_time})\n"
                except:
                    output += f"{indent_str}  (Compressed: {compression_timestamp})\n"

        return output

    def _render_fresh_content(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent_str: str, node_info: str) -> str:
        """NEW: Renders fresh content nodes from async compression fallback."""
        props = node.get("properties", {})
        element_id = props.get("element_id", "unknown")
        total_tokens = props.get("total_tokens", 0)
        token_limit = props.get("token_limit", 0)
        is_trimmed = props.get("is_trimmed", False)

        if is_trimmed:
            trimmed_count = props.get("trimmed_node_count", 0)
            original_count = props.get("original_node_count", 0)
            trimming_note = props.get("trimming_note", "Content trimmed")

            output = f"{indent_str}[{element_id}] {trimming_note}\n"
            output += f"{indent_str}  (Showing {trimmed_count} of {original_count} items, {total_tokens}/{token_limit} tokens)\n"
        else:
            output = f"{indent_str}[{element_id}] Fresh content (compression in progress)\n"
            output += f"{indent_str}  ({total_tokens} tokens, within {token_limit} limit)\n"

        # Children will be rendered normally by the main dispatcher
        return output

    def _render_compression_placeholder(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent_str: str, node_info: str) -> str:
        """NEW: Renders compression placeholder nodes while compression is in progress."""
        props = node.get("properties", {})
        memory_summary = props.get("memory_summary", "⏳ Processing conversation memory...")
        compression_status = props.get("compression_status", "in_progress")
        original_count = props.get("original_node_count", 0)

        output = f"{indent_str}<compression_placeholder>{memory_summary}</compression_placeholder>\n"

        # Add status information in verbose mode
        use_verbose_tags = options.get("render_style") == "verbose_tags"
        if use_verbose_tags and original_count > 0:
            output += f"{indent_str}  (Compressing {original_count} items in background, status: {compression_status})\n"

        return output

    # --- Main Recursive Renderer (Dispatcher) ---

    def _render_veil_node_to_string(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent: int = 0) -> str:
        """
        Recursively renders a VEIL node and its children into a string.
        Dispatches rendering based on node properties/annotations.
        Supports different rendering styles via options['render_style'].
        """
        #print(f"\n\nnode: {node}\n\n")
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
        elif node_type == "memory_context": # NEW: Memory context nodes from CompressionEngine
             render_func = self._render_memory_context
        elif node_type == "content_memory": # NEW: Compressed content memory from CompressionEngine
             render_func = self._render_content_memory
        elif node_type == "memorized_content": # NEW: Memorized content from MemoryCompressor
             render_func = self._render_memorized_content
        elif node_type in ["fresh_content", "trimmed_fresh_content"]: # NEW: Fresh content fallback from async compression
             render_func = self._render_fresh_content
        elif node_type == "compression_placeholder": # NEW: Compression placeholder from async compression
             render_func = self._render_compression_placeholder
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

    # NEW: Unified VEIL Processing Pipeline

    async def get_agent_context_via_compression_engine(self, options: Optional[Dict[str, Any]] = None) -> Union[str, Dict[str, Any]]:
        """
        NEW: Generate agent context using CompressionEngine as VEIL processor for unified rendering.

        This creates the unified rendering pipeline where:
        1. CompressionEngine processes full VEIL (compression + memory integration)
        2. HUD renders the processed VEIL using standard rendering methods
        3. All rendering logic is centralized in HUD, CompressionEngine handles preprocessing

        Args:
            options: Rendering options including focus_context, memory integration, etc.

        Returns:
            Rendered context (string or dict with multimodal content)
        """
        with tracer.start_as_current_span("hud.render_context_pipeline") as span:
            try:
                logger.debug(f"Generating agent context via CompressionEngine unified pipeline...")
                options = options or {}
                span.set_attribute("hud.render.options", json.dumps(options, default=str))

                # Get required components
                compression_engine = self.get_sibling_component("CompressionEngineComponent")
                veil_producer = self._get_space_veil_producer()

                if not compression_engine:
                    logger.warning(f"CompressionEngine not available, falling back to standard rendering")
                    span.set_attribute("hud.render.fallback", "no_compression_engine")
                    return await self.get_agent_context(options)

                if not veil_producer:
                    logger.error(f"SpaceVeilProducer not available, cannot generate context")
                    span.set_status(trace.Status(trace.StatusCode.ERROR, "SpaceVeilProducer not found"))
                    return "Error: Could not retrieve internal state."

                # Get full VEIL from SpaceVeilProducer
                full_veil = veil_producer.get_full_veil()
                if not full_veil:
                    logger.warning(f"Empty VEIL from SpaceVeilProducer")
                    span.set_attribute("hud.render.status", "empty_veil")
                    return "Current context is empty."

                # Get memory data if needed
                memory_data = None
                focus_context = options.get('focus_context')
                include_memory = options.get('include_memory', True)

                if include_memory:
                    memory_data = await compression_engine.get_memory_data()
                    logger.debug(f"Retrieved memory data for VEIL processing")

                # Process VEIL through CompressionEngine
                processed_veil = await compression_engine.process_veil_with_compression(
                    full_veil=full_veil,
                    focus_context=focus_context,
                    memory_data=memory_data
                )
                span.add_event("Processed VEIL", attributes={"veil.processed.json": json.dumps(processed_veil, default=str)})

                # Set render style
                if 'render_style' not in options:
                    options['render_style'] = 'verbose_tags'

                # Render the processed VEIL using standard HUD rendering
                attention_requests = {}  # Could integrate attention system here if needed
                context_string = self._render_veil_node_to_string(processed_veil, attention_requests, options, indent=0)
                span.set_attribute("hud.render.output_char_length", len(context_string))

                # FIXED: Auto-detect live multimodal content (consistent with get_agent_context)
                focus_element_id = focus_context.get('focus_element_id') if focus_context else None
                if focus_element_id:
                    # OPTIMIZED: Single-pass detection and extraction instead of separate operations
                    final_context = await self._detect_and_extract_multimodal_content(context_string, options, focus_element_id)
                    span.set_attribute("hud.render.multimodal", True)
                    return final_context
                else:
                    logger.info(f"Generated context via CompressionEngine pipeline: {len(context_string)} chars (text-only)")
                    span.set_attribute("hud.render.multimodal", False)
                    return context_string

            except Exception as e:
                logger.error(f"Error in unified CompressionEngine pipeline: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, "HUD rendering failed"))
                # Fallback to standard rendering
                return await self.get_agent_context(options)

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
