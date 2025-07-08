import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union
from datetime import datetime, timezone
import time
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
        """
        Helper to get the SpaceVeilProducer from the owning InnerSpace.
        
        This is the primary way HUD should access VEIL data, ensuring consistency
        with the centralized VEIL management architecture.
        """
        if not self.owner:
            return None
        # Assuming SpaceVeilProducer is the primary/only VEIL producer on InnerSpace
        return self.get_sibling_component("SpaceVeilProducer")

    def get_flat_veil_cache_via_producer(self) -> Dict[str, Any]:
        """
        NEW: Centralized method to get flat VEIL cache via SpaceVeilProducer.
        
        This ensures all HUD VEIL access goes through the centralized architecture
        and provides a consistent access pattern.
        
        Returns:
            Flat VEIL cache from SpaceVeilProducer, or empty dict if not available
        """
        try:
            veil_producer = self._get_space_veil_producer()
            if veil_producer:
                return veil_producer.get_flat_veil_cache()
            else:
                logger.warning(f"[{self.owner.id if self.owner else 'Unknown'}] SpaceVeilProducer not available for flat cache access")
                return {}
        except Exception as e:
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Error getting flat VEIL cache via producer: {e}", exc_info=True)
            return {}

    async def get_temporal_veil_context(self, element_id: str, memory_formation_index: int) -> str:
        """
        NEW: Get temporal context for memory compression via SpaceVeilProducer.
        
        This provides a direct interface for HUD to request temporal context,
        integrating seamlessly with the temporal memory recompression system.
        
        Args:
            element_id: Element ID to exclude from temporal context
            memory_formation_index: Delta index when memory was formed
            
        Returns:
            Temporal context string for memory compression
        """
        try:
            veil_producer = self._get_space_veil_producer()
            if veil_producer:
                return await veil_producer.render_temporal_context_for_compression(
                    element_id, memory_formation_index
                )
            else:
                logger.warning(f"[{self.owner.id if self.owner else 'Unknown'}] SpaceVeilProducer not available for temporal context")
                return "Error: SpaceVeilProducer not available for temporal context"
        except Exception as e:
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Error getting temporal context: {e}", exc_info=True)
            return f"Error getting temporal context: {e}"

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

            # FIXED: Get flat cache via SpaceVeilProducer instead of direct access
            veil_producer = self._get_space_veil_producer()
            if not veil_producer:
                logger.warning(f"[{self.owner.id}] Cannot get essential components: SpaceVeilProducer not available")
                return None

            flat_cache = veil_producer.get_flat_veil_cache()

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

        # TEMPORAL EXCLUSION: Skip rendering this container if it's the excluded element
        exclude_element_id = options.get("exclude_element_id")
        if exclude_element_id and element_id == exclude_element_id:
            return f"{indent_str}[EXCLUDED: {element_name} - content being memorized separately]\n"

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
            
            # TEMPORAL CONTEXT: Add temporal metadata if available
            if options.get("temporal_context") and options.get("memory_formation_index") is not None:
                formation_index = options.get("memory_formation_index")
                output += f"{indent_str}  (Temporal: Formation Index {formation_index})\n"

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
        MODIFIED: Always use chronological flat rendering by default.
        
        Args:
            options: Rendering options including:
                - render_style: 'chronological_flat' (default), 'hierarchical' (deprecated)
                - include_time_markers: bool (default: True)
                - include_system_messages: bool (default: True)
                - focus_context: dict for focus element
                - include_memory: bool (default: True)
        """
        
        with tracer.start_as_current_span("hud.render_context_pipeline") as span:
            try:
                logger.debug(f"Generating agent context via CompressionEngine unified pipeline...")
                options = options or {}
                
                # Set default rendering options for chronological flat
                options.setdefault('render_style', 'chronological_flat')
                options.setdefault('include_time_markers', True)
                options.setdefault('include_system_messages', True)
                
                span.set_attribute("hud.render.options", json.dumps(options, default=str))

                # Get required components
                compression_engine = self.get_sibling_component("CompressionEngineComponent")
                veil_producer = self._get_space_veil_producer()                    
                flat_veil_cache = veil_producer.get_flat_veil_cache()                    
                logger.debug(f"Using flat VEIL cache approach for CompressionEngine processing")
                span.set_attribute("hud.render.veil_source", "flat_cache")
                
                # Get memory data if needed
                memory_data = None
                include_memory = options.get('include_memory', True)
                if include_memory:
                    memory_data = await compression_engine.get_memory_data()
                    logger.debug(f"Retrieved memory data for flat cache VEIL processing")
                
                # Process flat VEIL cache through CompressionEngine
                focus_context = options.get('focus_context')
                processed_veil = await compression_engine.process_flat_veil_with_compression(
                    flat_veil_cache=flat_veil_cache,
                    focus_context=focus_context,
                    memory_data=memory_data
                )
                
                if not processed_veil:
                    logger.warning(f"CompressionEngine returned empty result from flat cache processing")
                    span.set_attribute("hud.render.status", "empty_processed_veil")
                    return "Current context is empty."
                
                # Render the processed flat cache
                context_string = await self._render_processed_flat_veil(processed_veil, options)

                logger.info(f"Agent context generated via CompressionEngine: {len(context_string)} chars")
                span.set_attribute("hud.render.output_length", len(context_string))
                span.set_status(trace.Status(trace.StatusCode.OK))
                return context_string

            except Exception as e:
                logger.error(f"Error in CompressionEngine context pipeline: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, "Context pipeline failed"))
                
                # NEW: Fallback to direct chronological flat rendering
                try:
                    logger.info(f"Falling back to direct chronological flat rendering")
                    span.set_attribute("hud.render.fallback", "direct_flat_cache")
                    
                    # Get flat cache directly
                    flat_cache = self.get_flat_veil_cache_via_producer()
                    if flat_cache:
                        fallback_context = self._render_chronological_flat_content(flat_cache, options)
                        logger.info(f"Fallback chronological rendering successful: {len(fallback_context)} chars")
                        return fallback_context
                    else:
                        logger.warning(f"No flat cache available for fallback")
                        return "Error: No context available"
                        
                except Exception as fallback_error:
                    logger.error(f"Fallback rendering also failed: {fallback_error}", exc_info=True)
                    return "Error: Both primary and fallback rendering failed"

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

    # PHASE 3: NEW - Specialized Memorization Context Rendering

    async def render_temporal_veil_for_compression(self, 
                                                 temporal_veil: Dict[str, Any],
                                                 exclude_element_id: str,
                                                 memory_formation_index: int) -> str:
        """
        NEW: Render temporal VEIL context for memory compression.
        
        This method is called by SpaceVeilProducer.render_temporal_context_for_compression()
        to generate temporally consistent context for memory compression.
        
        The temporal_veil passed in has been reconstructed at the memory formation point
        and then had future edits applied, providing the historically accurate context
        with current appearance.
        
        Args:
            temporal_veil: VEIL reconstructed at memory formation time with future edits applied
            exclude_element_id: Element ID to exclude from rendering (avoid duplication)
            memory_formation_index: Delta index when memory was formed
            
        Returns:
            Rendered temporal context string for memory compression
        """
        try:
            logger.debug(f"Rendering temporal VEIL for compression: excluding {exclude_element_id}, formation index {memory_formation_index}")
            
            # Use specialized rendering options for temporal context
            options = {
                "render_style": "verbose_tags",
                "temporal_context": True,
                "memory_formation_index": memory_formation_index,
                "exclude_element_id": exclude_element_id
            }
            
            # Add temporal context header
            context_lines = []
            context_lines.append("[TEMPORAL CONTEXT FOR MEMORY COMPRESSION]")
            context_lines.append(f"Memory formation at delta index: {memory_formation_index}")
            context_lines.append(f"Content reconstructed with historical accuracy + future edits applied")
            context_lines.append(f"Excluding element: {exclude_element_id}")
            context_lines.append("")
            
            # Render the temporal VEIL structure
            attention_requests = {}
            temporal_content = self._render_veil_node_to_string(temporal_veil, attention_requests, options, indent=0)
            
            context_lines.append(temporal_content)
            context_lines.append("")
            context_lines.append("[END TEMPORAL CONTEXT]")
            
            result = "\n".join(context_lines)
            
            logger.debug(f"Rendered temporal context: {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error rendering temporal VEIL for compression: {e}", exc_info=True)
            # Fallback to basic rendering
            return f"Error rendering temporal context: {e}"

    async def render_memorization_context_veil(self, 
                                             full_veil: Dict[str, Any],
                                             exclude_element_id: str,
                                             exclude_content: List[Dict[str, Any]],
                                             focus_element_id: Optional[str] = None) -> str:
        """
        PHASE 3: Render VEIL context for memorization with selective content exclusion.
        
        This creates the specialized context that shows the agent the full conversation state
        while excluding the specific content being memorized to avoid duplication.
        
        The rendered output provides complete context so the agent understands:
        - What other conversations/elements are active
        - What has already been memorized (M-chunks)
        - What the current conversation state is
        - Available tools and context
        
        But EXCLUDES the specific N-chunk being memorized to avoid seeing it twice.
        
        Args:
            full_veil: Complete VEIL structure from SpaceVeilProducer
            exclude_element_id: ID of element where content is being memorized
            exclude_content: Specific VEIL nodes to exclude from rendering
            focus_element_id: Optional focus element for normal focused/unfocused logic
            
        Returns:
            Rendered memorization context string
        """
        try:
            logger.debug(f"Rendering memorization context: excluding content from {exclude_element_id}")
            
            # Create a deep copy of the VEIL for modification
            import copy
            context_veil = copy.deepcopy(full_veil)
            
            # Recursively find and modify the target element to exclude specific content
            self._exclude_content_from_veil_element(context_veil, exclude_element_id, exclude_content)
            
            # Use standard rendering with special options
            options = {
                "render_style": "verbose_tags",
                "memorization_context": True,
                "excluded_element_id": exclude_element_id
            }
            
            attention_requests = {}
            
            # Render the modified VEIL
            context_string = self._render_veil_node_to_string(context_veil, attention_requests, options, indent=0)
            
            logger.debug(f"Rendered memorization context: {len(context_string)} characters")
            return context_string
            
        except Exception as e:
            logger.error(f"Error rendering memorization context: {e}", exc_info=True)
            # Fallback: render full VEIL without exclusions
            options = {"render_style": "verbose_tags"}
            return self._render_veil_node_to_string(full_veil, {}, options, indent=0)

    async def render_memorization_context_with_flat_cache(self, 
                                                        flat_veil_cache: Dict[str, Any],
                                                        exclude_element_id: str,
                                                        exclude_content: List[Dict[str, Any]],
                                                        focus_element_id: Optional[str] = None) -> str:
        """
        NEW: Render memorization context using flat VEIL cache directly from Space.
        
        This eliminates the need for redundant VEIL hierarchy reconstruction and leverages
        the existing Space flat cache infrastructure. It reconstructs the hierarchy as needed
        and then applies exclusion logic for memorization context.
        
        Args:
            flat_veil_cache: Flat VEIL cache from Space (already includes any edit deltas)
            exclude_element_id: ID of element where content is being memorized
            exclude_content: Specific VEIL nodes to exclude from rendering
            focus_element_id: Optional focus element for normal focused/unfocused logic
            
        Returns:
            Rendered memorization context string
        """
        try:
            logger.debug(f"Rendering memorization context from flat cache: excluding content from {exclude_element_id}")
            
            # Reconstruct hierarchical VEIL from flat cache
            hierarchical_veil = self._reconstruct_veil_hierarchy_from_flat_cache(flat_veil_cache)
            if not hierarchical_veil:
                logger.warning(f"Failed to reconstruct VEIL hierarchy from flat cache for memorization context")
                return "Error: Could not reconstruct VEIL hierarchy for memorization context"
            
            # Use existing memorization context logic
            return await self.render_memorization_context_veil(
                full_veil=hierarchical_veil,
                exclude_element_id=exclude_element_id,
                exclude_content=exclude_content,
                focus_element_id=focus_element_id
            )
            
        except Exception as e:
            logger.error(f"Error rendering memorization context from flat cache: {e}", exc_info=True)
            return f"Error rendering memorization context: {e}"

    def _reconstruct_veil_hierarchy_from_flat_cache(self, flat_cache: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        NEW: Reconstruct hierarchical VEIL structure from flat cache.
        
        This reuses the same hierarchy reconstruction logic that SpaceVeilProducer uses,
        but can work directly with the flat cache without going through the producer.
        
        Args:
            flat_cache: Flat VEIL cache from Space
            
        Returns:
            Hierarchical VEIL structure, or None if reconstruction fails
        """
        try:
            if not flat_cache:
                logger.debug(f"Empty flat cache provided for hierarchy reconstruction")
                return None
            
            # Find the space root node to start reconstruction
            space_root_id = None
            for veil_id, node_data in flat_cache.items():
                if isinstance(node_data, dict):
                    node_type = node_data.get("node_type", "")
                    if node_type == "space_root":
                        space_root_id = veil_id
                        break
            
            if not space_root_id:
                logger.warning(f"No space root node found in flat cache for hierarchy reconstruction")
                # Try to find any root-like node
                for veil_id, node_data in flat_cache.items():
                    if isinstance(node_data, dict):
                        props = node_data.get("properties", {})
                        if props.get("structural_role") == "root":
                            space_root_id = veil_id
                            logger.debug(f"Using alternative root node: {space_root_id}")
                            break
                
                if not space_root_id:
                    logger.error(f"Cannot find any root node in flat cache for hierarchy reconstruction")
                    return None
            
            # Use the same hierarchy reconstruction logic as SpaceVeilProducer
            # We'll implement a simplified version here
            hierarchical_veil = self._build_veil_hierarchy_from_flat_cache(
                flat_cache=flat_cache,
                root_node_id=space_root_id,
                processed_nodes=set()
            )
            
            if hierarchical_veil:
                logger.debug(f"Successfully reconstructed VEIL hierarchy from flat cache starting at {space_root_id}")
                return hierarchical_veil
            else:
                logger.warning(f"Failed to reconstruct VEIL hierarchy from root {space_root_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error reconstructing VEIL hierarchy from flat cache: {e}", exc_info=True)
            return None

    def _build_veil_hierarchy_from_flat_cache(self, 
                                            flat_cache: Dict[str, Any],
                                            root_node_id: str, 
                                            processed_nodes: set) -> Optional[Dict[str, Any]]:
        """
        NEW: Recursively build hierarchical VEIL structure from flat cache.
        
        This mirrors the logic in SpaceVeilProducer.build_hierarchical_veil_from_flat_cache
        but is implemented in HUD for direct flat cache processing.
        
        Args:
            flat_cache: Flat VEIL cache
            root_node_id: Current node to build hierarchy for
            processed_nodes: Set of already processed nodes (cycle detection)
            
        Returns:
            Hierarchical node structure or None if failed
        """
        try:
            # Cycle detection
            if root_node_id in processed_nodes:
                logger.warning(f"Cycle detected in VEIL hierarchy reconstruction for node {root_node_id}")
                return None
            
            # Get node data from flat cache
            original_node_data = flat_cache.get(root_node_id)
            if not original_node_data:
                logger.warning(f"Node {root_node_id} not found in flat cache during hierarchy reconstruction")
                return None
            
            processed_nodes.add(root_node_id)
            
            # Create deep copy of node data
            import copy
            reconstructed_node = copy.deepcopy(original_node_data)
            
            # Find and build children
            children = []
            for potential_child_id, child_node_data in flat_cache.items():
                if potential_child_id == root_node_id:
                    continue  # Node can't be its own child
                
                if not isinstance(child_node_data, dict):
                    continue
                
                # Check if this node is a child of the current root
                parent_id = None
                child_props = child_node_data.get("properties", {})
                if "parent_id" in child_props:
                    parent_id = child_props["parent_id"]
                elif "parent_id" in child_node_data:
                    parent_id = child_node_data["parent_id"]
                
                if parent_id == root_node_id:
                    # This is a child - recursively build its hierarchy
                    child_hierarchy = self._build_veil_hierarchy_from_flat_cache(
                        flat_cache=flat_cache,
                        root_node_id=potential_child_id,
                        processed_nodes=processed_nodes
                    )
                    
                    if child_hierarchy:
                        children.append(child_hierarchy)
            
            # Set children in reconstructed node
            reconstructed_node["children"] = children
            
            # Remove from processed set for backtracking
            processed_nodes.remove(root_node_id)
            
            return reconstructed_node
            
        except Exception as e:
            logger.error(f"Error building hierarchy for node {root_node_id}: {e}", exc_info=True)
            # Remove from processed set on error
            processed_nodes.discard(root_node_id)
            return None

    def _exclude_content_from_veil_element(self, 
                                         veil_node: Dict[str, Any], 
                                         target_element_id: str, 
                                         exclude_content: List[Dict[str, Any]]) -> None:
        """
        Recursively find target element and exclude specific content from it.
        
        This modifies the VEIL in-place to remove the content being memorized
        while preserving all other context.
        
        Args:
            veil_node: Current VEIL node (modified in-place)
            target_element_id: Element ID to find and modify
            exclude_content: Content to exclude from that element
        """
        try:
            # Check if this is the target element container
            props = veil_node.get("properties", {})
            element_id = props.get("element_id")
            
            if element_id == target_element_id:
                logger.debug(f"Found target element {target_element_id}, excluding memorization content")
                
                # Remove the specific content being memorized from children
                children = veil_node.get("children", [])
                exclude_veil_ids = {node.get("veil_id") for node in exclude_content if node.get("veil_id")}
                
                # Filter out the excluded content
                filtered_children = [
                    child for child in children 
                    if child.get("veil_id") not in exclude_veil_ids
                ]
                
                # Update children
                veil_node["children"] = filtered_children
                
                # Add a note about excluded content for context
                if len(filtered_children) < len(children):
                    excluded_count = len(children) - len(filtered_children)
                    props["memorization_note"] = f"Excluding {excluded_count} items being memorized"
                
                logger.debug(f"Excluded {len(children) - len(filtered_children)} items from element {target_element_id}")
                return
            
            # Recursively search children
            children = veil_node.get("children", [])
            for child in children:
                self._exclude_content_from_veil_element(child, target_element_id, exclude_content)
                
        except Exception as e:
            logger.error(f"Error excluding content from VEIL element: {e}", exc_info=True)

    async def _render_processed_flat_veil(self, processed_flat_cache: Dict[str, Any], options: Dict[str, Any]) -> str:
        """
        MODIFIED: Render processed flat VEIL cache using chronological flat rendering.
        
        This now bypasses hierarchy reconstruction and renders directly from flat cache.
        """
        try:
            # Check if we should use chronological flat rendering (default)
            render_style = options.get('render_style', 'chronological_flat')
            if render_style == 'chronological_flat':
                # NEW: Use chronological flat rendering
                context_string = self._render_chronological_flat_content(processed_flat_cache, options)
                logger.debug(f"Rendered flat VEIL chronologically: {len(context_string)} characters")
                return context_string
            else:
                # DEPRECATED: Fall back to hierarchical rendering for compatibility
                logger.debug(f"Using deprecated hierarchical rendering (render_style: {render_style})")
                hierarchical_veil = self._reconstruct_veil_hierarchy_from_flat_cache(processed_flat_cache)
                if not hierarchical_veil:
                    logger.warning(f"Failed to reconstruct hierarchy from processed flat cache")
                    return "Error: Could not reconstruct VEIL hierarchy for rendering"
                
                # Set default render style if not specified
                if 'render_style' not in options:
                    options['render_style'] = 'verbose_tags'
                
                # Render using standard hierarchical rendering
                attention_requests = {}
                context_string = self._render_veil_node_to_string(hierarchical_veil, attention_requests, options, indent=0)
                
                logger.debug(f"Rendered processed flat VEIL hierarchically: {len(context_string)} characters")
                return context_string
                
        except Exception as e:
            logger.error(f"Error rendering processed flat VEIL: {e}", exc_info=True)
            return f"Error rendering processed VEIL: {e}"

    def _count_memory_nodes_recursively(self, veil_node: Dict[str, Any]) -> int:
        """
        DEBUG: Recursively count memory nodes in a VEIL hierarchy.
        
        Args:
            veil_node: Root VEIL node to start counting from
            
        Returns:
            Total count of memory nodes in the hierarchy
        """
        try:
            count = 0
            
            # Check if current node is a memory node
            if isinstance(veil_node, dict):
                node_type = veil_node.get("node_type", "")
                if node_type == "content_memory":
                    count += 1
                
                # Recursively count in children
                children = veil_node.get("children", [])
                for child in children:
                    count += self._count_memory_nodes_recursively(child)
            
            return count
            
        except Exception as e:
            logger.error(f"Error counting memory nodes: {e}", exc_info=True)
            return 0

    # NEW: Chronological Flat Rendering Implementation

    def _get_chronological_timestamp(self, node_data: Dict[str, Any]) -> float:
        """
        Extract unified chronological timestamp from VEIL node for sorting.
        
        Handles various timestamp fields and formats to provide consistent chronological ordering.
        This solves the historical message problem where old messages get fresh operation indices.
        
        Args:
            node_data: VEIL node to extract timestamp from
            
        Returns:
            Unix timestamp as float for sorting (fallback to operation_index if no content timestamps)
        """
        try:
            if not isinstance(node_data, dict):
                return 0.0
                
            props = node_data.get("properties", {})
            
            # Check unified timestamp field first (new standard)
            timestamp = props.get("timestamp")
            if timestamp:
                if isinstance(timestamp, (int, float)):
                    return float(timestamp)
                elif isinstance(timestamp, str):
                    numeric_ts = self._convert_timestamp_to_numeric(timestamp)
                    if numeric_ts:
                        return numeric_ts
            
            # Check legacy message timestamp_iso field
            timestamp_iso = props.get("timestamp_iso")
            if timestamp_iso:
                numeric_ts = self._convert_timestamp_to_numeric(timestamp_iso)
                if numeric_ts:
                    return numeric_ts
            
            # Check compression_timestamp for old memories
            compression_timestamp = props.get("compression_timestamp")
            if compression_timestamp:
                numeric_ts = self._convert_timestamp_to_numeric(compression_timestamp)
                if numeric_ts:
                    return numeric_ts
            
            # Check other timestamp fields
            for field in ["created_at", "message_timestamp", "note_timestamp"]:
                value = props.get(field)
                if value:
                    numeric_ts = self._convert_timestamp_to_numeric(value)
                    if numeric_ts:
                        return numeric_ts
            
            # Fallback to operation_index for content without timestamps
            operation_index = props.get("operation_index", 0)
            if operation_index:
                # Convert operation_index to timestamp-like value for sorting
                # Use a base time + operation_index seconds for relative ordering
                return float(operation_index)
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error extracting chronological timestamp: {e}")
            return 0.0

    def _convert_timestamp_to_numeric(self, timestamp_value: Any) -> Optional[float]:
        """
        Convert various timestamp formats to numeric Unix timestamp.
        
        Args:
            timestamp_value: Timestamp in various formats
            
        Returns:
            Unix timestamp as float, or None if conversion fails
        """
        try:
            if isinstance(timestamp_value, (int, float)):
                return float(timestamp_value)
            
            elif isinstance(timestamp_value, str):
                from datetime import datetime
                try:
                    # Handle ISO formats
                    if timestamp_value.endswith('Z'):
                        dt = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                    elif '+' in timestamp_value or timestamp_value.endswith('UTC'):
                        dt = datetime.fromisoformat(timestamp_value.replace('UTC', ''))
                    else:
                        dt = datetime.fromisoformat(timestamp_value)
                    
                    return dt.timestamp()
                    
                except ValueError:
                    # Try parsing as numeric string
                    try:
                        return float(timestamp_value)
                    except ValueError:
                        return None
            
            return None
            
        except Exception:
            return None

    def _extract_chronological_content_from_flat_cache(self, flat_cache: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all renderable content from flat cache in natural chronological order.
        
        ENHANCED: Now extracts both content nodes AND container nodes to provide
        complete structural information including space metadata and chat containers.
        
        Returns:
            List of content items with metadata for chronological rendering
        """
        try:
            content_items = []
            container_items = []  # NEW: Separate containers for structural rendering
            
            # NEW: Sort by unified timestamp field for true chronological order
            veil_nodes = [(veil_id, node_data) for veil_id, node_data in flat_cache.items()]
            veil_nodes.sort(key=lambda x: self._get_chronological_timestamp(x[1]))
            
            for veil_id, node_data in veil_nodes:
                if not isinstance(node_data, dict):
                    continue
                    
                node_type = node_data.get("node_type", "")
                props = node_data.get("properties", {})
                content_nature = props.get("content_nature", "")
                structural_role = props.get("structural_role", "")

                # NEW: Extract container items (structural)
                container_item = None
                if node_type == "space_root" or structural_role == "root":
                    container_item = self._extract_space_root_content(node_data)
                elif structural_role == "container" and content_nature == "message_list":
                    container_item = self._extract_message_list_container_content(node_data)
                elif structural_role == "container" and content_nature == "scratchpad_summary":
                    container_item = self._extract_scratchpad_container_content(node_data)
                    
                if container_item:
                    container_items.append(container_item)

                # Extract content items (chronological)
                content_item = None
                if content_nature == "chat_message":
                    content_item = self._extract_chat_message_content(node_data)
                elif content_nature == "scratchpad_summary" or "scratchpad" in node_type.lower():
                    content_item = self._extract_scratchpad_content(node_data)
                elif node_type in ["content_memory", "memorized_content"]:
                    content_item = self._extract_memory_content(node_data)
                elif structural_role == "attachment_content":
                    # Attachments are handled as children of messages
                    continue
                    
                if content_item:
                    content_items.append(content_item)
            
            logger.debug(f"Extracted {len(content_items)} content items and {len(container_items)} container items")
            
            # Return combined structure: containers provide metadata, content provides chronological flow
            return {
                "content_items": content_items,
                "container_items": container_items
            }
            
        except Exception as e:
            logger.error(f"Error extracting chronological content: {e}", exc_info=True)
            return {"content_items": [], "container_items": []}

    def _extract_chat_message_content(self, node_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract chat message content with metadata."""
        try:
            props = node_data.get("properties", {})
            timestamp = props.get("timestamp_iso", 0)
            
            # Convert timestamp to numeric if it's a string
            if isinstance(timestamp, str):
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamp = dt.timestamp()
                except:
                    timestamp = 0
            
            # DEBUG: Log extracted conversation context
            logger.debug(f"Extracting chat message - adapter_type: {props.get('adapter_type')}, server_name: {props.get('server_name')}, conversation_name: {props.get('conversation_name')}")
            
            return {
                "content_type": "chat_message",
                "timestamp": timestamp,
                "veil_id": node_data.get("veil_id"),
                "node_data": node_data,
                "conversation_name": props.get("conversation_name"),
                "element_id": props.get("element_id"),
                "sender_name": props.get("sender_name", "Unknown"),
                "text_content": props.get("text_content", ""),
                "is_edited": props.get("is_edited", False),
                "is_agent": props.get("is_agent", False),  # SIMPLIFIED: Use direct is_agent field from VEIL
                # NEW: Extract conversation context directly for message rendering
                "adapter_type": props.get("adapter_type"),
                "server_name": props.get("server_name"),
            }
        except Exception as e:
            logger.warning(f"Error extracting chat message content: {e}")
            return None

    def _extract_scratchpad_content(self, node_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract scratchpad content with metadata."""
        try:
            props = node_data.get("properties", {})
            # Use current time for scratchpad content if no timestamp
            timestamp = props.get("timestamp", time.time())
            
            return {
                "content_type": "scratchpad",
                "timestamp": timestamp,
                "veil_id": node_data.get("veil_id"),
                "node_data": node_data,
                "element_id": props.get("element_id"),
                "note_content": props.get("note_content", "")
            }
        except Exception as e:
            logger.warning(f"Error extracting scratchpad content: {e}")
            return None

    def _extract_memory_content(self, node_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract memory content with metadata."""
        try:
            props = node_data.get("properties", {})
            
            # NEW: Use unified timestamp field for chronological placement
            # This ensures memories appear at content time, not compression time
            timestamp = props.get("timestamp")  # Content timestamp from AgentMemoryCompressor
            
            if not timestamp:
                # Fallback to compression_timestamp for legacy memories
                timestamp = props.get("compression_timestamp", time.time())
            
            # Convert to numeric if needed
            if isinstance(timestamp, str):
                numeric_ts = self._convert_timestamp_to_numeric(timestamp)
                timestamp = numeric_ts if numeric_ts else time.time()
            elif not isinstance(timestamp, (int, float)):
                timestamp = time.time()
            
            return {
                "content_type": "memory",
                "timestamp": float(timestamp),
                "veil_id": node_data.get("veil_id"),
                "node_data": node_data,
                "memory_summary": props.get("memory_summary", ""),
                "memory_type": props.get("memory_type", "unknown"),
                # NEW: Include conversation context for memories
                "conversation_name": props.get("conversation_name"),
                "adapter_type": props.get("adapter_type"),
                "server_name": props.get("server_name")
            }
        except Exception as e:
            logger.warning(f"Error extracting memory content: {e}")
            return None

    def _extract_space_root_content(self, node_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """NEW: Extract space root content for inner_space wrapper."""
        try:
            props = node_data.get("properties", {})
            
            return {
                "container_type": "space_root",
                "veil_id": node_data.get("veil_id"),
                "node_data": node_data,
                "element_id": props.get("element_id"),
                "element_name": props.get("element_name", "Unknown Space"),
                "agent_name": props.get("agent_name"),
                "agent_description": props.get("agent_description"),
                "is_inner_space": props.get("is_inner_space", False)
            }
        except Exception as e:
            logger.warning(f"Error extracting space root content: {e}")
            return None

    def _extract_message_list_container_content(self, node_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """NEW: Extract message list container for chat_info sections."""
        try:
            props = node_data.get("properties", {})
            
            # DEBUG: Log extracted metadata for debugging
            logger.debug(f"Extracting message list container - adapter_type: {props.get('adapter_type')}, server_name: {props.get('server_name')}, conversation_name: {props.get('conversation_name')}")
            
            return {
                "container_type": "message_list",
                "veil_id": node_data.get("veil_id"),
                "node_data": node_data,
                "element_id": props.get("element_id"),
                "element_name": props.get("element_name", "Unknown Chat"),
                "conversation_name": props.get("conversation_name"),
                "adapter_type": props.get("adapter_type"),
                "server_name": props.get("server_name"),
                "adapter_id": props.get("adapter_id"),
                "external_conversation_id": props.get("external_conversation_id"),
                "available_tools": props.get("available_tools", []),
                "tool_target_element_id": props.get("tool_target_element_id"),
                "message_count": props.get("message_count", 0),
                "alias": props.get("alias"),
                "is_focused": False  # Will be determined later by focus context
            }
        except Exception as e:
            logger.warning(f"Error extracting message list container: {e}")
            return None

    def _extract_scratchpad_container_content(self, node_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """NEW: Extract scratchpad container for scratchpad info."""
        try:
            props = node_data.get("properties", {})
            
            return {
                "container_type": "scratchpad",
                "veil_id": node_data.get("veil_id"),
                "node_data": node_data,
                "element_id": props.get("element_id"),
                "element_name": props.get("element_name", "Scratchpad"),
                "note_count": props.get("note_count", 0)
            }
        except Exception as e:
            logger.warning(f"Error extracting scratchpad container: {e}")
            return None

    def _group_content_by_type(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENHANCED: Group content items and organize container items.
        
        Args:
            extraction_result: Dictionary with 'content_items' and 'container_items' keys
            
        Returns:
            Dictionary with grouped content and organized containers
        """
        try:
            content_items = extraction_result.get("content_items", [])
            container_items = extraction_result.get("container_items", [])
            
            # Group content items by type (as before)
            content_groups = []
            current_group = None
            
            for item in content_items:
                content_type = item.get("content_type")
                
                # Start new group if type changes or no current group
                if not current_group or current_group["group_type"] != content_type:
                    if current_group:
                        content_groups.append(current_group)
                    
                    current_group = {
                        "group_type": content_type,
                        "items": [item],
                        "start_timestamp": item.get("timestamp", 0),
                        "end_timestamp": item.get("timestamp", 0)
                    }
                else:
                    # Add to current group
                    current_group["items"].append(item)
                    current_group["end_timestamp"] = item.get("timestamp", 0)
            
            # Add final group
            if current_group:
                content_groups.append(current_group)
            
            # NEW: Organize containers by type for structural rendering
            containers = {
                "space_root": None,
                "message_lists": [],
                "scratchpad": None
            }
            
            for container in container_items:
                container_type = container.get("container_type")
                if container_type == "space_root":
                    containers["space_root"] = container
                elif container_type == "message_list":
                    containers["message_lists"].append(container)
                elif container_type == "scratchpad":
                    containers["scratchpad"] = container
            
            logger.debug(f"Grouped {len(content_groups)} content groups and {len(container_items)} containers")
            
            return {
                "content_groups": content_groups,
                "containers": containers
            }
            
        except Exception as e:
            logger.error(f"Error grouping content by type: {e}", exc_info=True)
            return {"content_groups": [], "containers": {"space_root": None, "message_lists": [], "scratchpad": None}}

    def _apply_time_markers(self, content_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply relative time markers to content groups.
        
        Time intervals (relative to most recent message):
        - 1 minute ago
        - 1 hour ago  
        - 8 hours ago
        - 1 day ago
        - 2 days ago
        - more than 2 days ago
        - more than 7 days ago
        """
        try:
            if not content_groups:
                return content_groups
            
            # Find most recent timestamp
            most_recent_time = 0
            for group in content_groups:
                group_time = group.get("end_timestamp", 0)
                if group_time > most_recent_time:
                    most_recent_time = group_time
            
            if most_recent_time == 0:
                return content_groups
            
            # Define time intervals (in seconds)
            intervals = [
                (60, "1 minute ago"),           # 1 minute
                (3600, "1 hour ago"),           # 1 hour  
                (8 * 3600, "8 hours ago"),      # 8 hours
                (24 * 3600, "1 day ago"),       # 1 day
                (2 * 24 * 3600, "2 days ago"),  # 2 days
                (7 * 24 * 3600, "more than 2 days ago"),  # 7 days
                (float('inf'), "more than 7 days ago")     # > 7 days
            ]
            
            # Track which intervals we've already marked
            marked_intervals = set()
            
            # Process groups from newest to oldest
            for group in reversed(content_groups):
                group_time = group.get("end_timestamp", 0)
                time_diff = most_recent_time - group_time
                
                # Find appropriate interval
                for threshold, label in intervals:
                    if time_diff <= threshold and threshold not in marked_intervals:
                        # Mark this interval on the last message of the group
                        if group["items"]:
                            last_item = group["items"][-1]
                            last_item["time_marker"] = label
                            marked_intervals.add(threshold)
                        break
            
            return content_groups
            
        except Exception as e:
            logger.error(f"Error applying time markers: {e}", exc_info=True)
            return content_groups

    def _render_chronological_flat_content(self, flat_cache: Dict[str, Any], options: Dict[str, Any]) -> str:
        """
        ENHANCED: Main chronological flat rendering with complete structural information.
        
        Now renders both container metadata and chronological content to match expected output.
        """
        try:
            logger.debug(f"Starting enhanced chronological flat rendering of {len(flat_cache)} VEIL nodes")
            
            # Extract both content and containers
            extraction_result = self._extract_chronological_content_from_flat_cache(flat_cache)
            if not extraction_result.get("content_items") and not extraction_result.get("container_items"):
                return "Current context is empty."
            
            # Group content and organize containers
            grouped_result = self._group_content_by_type(extraction_result)
            content_groups = grouped_result.get("content_groups", [])
            containers = grouped_result.get("containers", {})
            
            # NEW: Determine focus context from options
            focus_context = options.get("focus_context", {})
            focused_element_id = focus_context.get("focus_element_id")  # FIXED: Use correct field name
            
            # NEW: Mark focused message list containers
            for msg_list in containers.get("message_lists", []):
                if msg_list.get("element_id") == focused_element_id:
                    msg_list["is_focused"] = True
            
            # Apply time markers to content groups
            if content_groups:
                content_groups = self._apply_time_markers(content_groups)
            
            # Generate system messages for edits (if enabled)
            if options.get("include_system_messages", True) and content_groups:
                content_groups = self._generate_system_messages(content_groups, flat_cache)
            
            # NEW: Build complete structure with containers
            rendered_sections = []
            
            # 1. Render space root (inner_space wrapper)
            space_root = containers.get("space_root")
            if space_root:
                space_section = self._render_space_root_wrapper(space_root, containers, options)
                if space_section:
                    rendered_sections.append(space_section)
            
            # 2. Render early scratchpad sections (if any exist early in timeline)
            early_scratchpad_groups = [g for g in content_groups if g.get("group_type") == "scratchpad" and g.get("start_timestamp", 0) < (content_groups[0].get("start_timestamp", 0) + 3600) if content_groups]
            for group in early_scratchpad_groups:
                rendered_section = self._render_flat_scratchpad_group(group, options)
                if rendered_section:
                    rendered_sections.append(rendered_section)
            
            # 3. Render chat info containers 
            message_lists = containers.get("message_lists", [])
            if message_lists:
                chat_info_section = self._render_chat_info_section(message_lists, options)
                if chat_info_section:
                    rendered_sections.append(chat_info_section)
            
            # 4. Render chronological content groups (all types interleaved)
            for group in content_groups:
                group_type = group.get("group_type")
                
                if group_type == "chat_message":
                    rendered_section = self._render_flat_chat_group(group, options)
                elif group_type == "memory":
                    rendered_section = self._render_flat_memory_group(group, options)
                elif group_type == "scratchpad":
                    # Skip early scratchpad groups already rendered
                    if group not in early_scratchpad_groups:
                        rendered_section = self._render_flat_scratchpad_group(group, options)
                    else:
                        continue
                elif group_type == "system":
                    rendered_section = self._render_flat_system_group(group, options)
                else:
                    logger.warning(f"Unknown content group type: {group_type}")
                    continue
                
                if rendered_section:
                    rendered_sections.append(rendered_section)
            
            # 5. Close space root wrapper if we opened it
            if space_root:
                rendered_sections.append("</inner_space>")
            
            result = "\n\n".join(rendered_sections)
            logger.info(f"Rendered enhanced chronological flat content: {len(result)} chars, {len(rendered_sections)} sections")
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced chronological flat rendering: {e}", exc_info=True)
            return f"Error rendering enhanced chronological content: {e}"

    def _render_space_root_wrapper(self, space_root: Dict[str, Any], containers: Dict[str, Any], options: Dict[str, Any]) -> str:
        """NEW: Render the space root as inner_space wrapper with agent metadata."""
        try:
            agent_name = space_root.get("agent_name")
            agent_description = space_root.get("agent_description")
            
            # Build opening tag with agent metadata
            if agent_name:
                opening_tag = f'<inner_space agent_name="{agent_name}"'
                if agent_description:
                    opening_tag += f' agent_description="{agent_description}"'
                opening_tag += '>'
            else:
                opening_tag = '<inner_space>'
            
            return opening_tag
            
        except Exception as e:
            logger.error(f"Error rendering space root wrapper: {e}", exc_info=True)
            return '<inner_space>'

    def _render_chat_info_section(self, message_lists: List[Dict[str, Any]], options: Dict[str, Any]) -> str:
        """NEW: Render chat_info section with channel information."""
        try:
            if not message_lists:
                return ""
            
            # Group by adapter type and server
            adapter_groups = {}
            for msg_list in message_lists:
                adapter_type = msg_list.get("adapter_type") or "unknown"  # FIXED: Handle None values
                server_name = msg_list.get("server_name") or "default"    # FIXED: Handle None values
                alias = msg_list.get("alias") or "unknown"
                key = f"{adapter_type}:{server_name}"
                
                if key not in adapter_groups:
                    adapter_groups[key] = {
                        "adapter_type": adapter_type,
                        "server_name": server_name,
                        "channels": []
                    }
                
                adapter_groups[key]["channels"].append(msg_list)
            
            # Render chat info sections
            info_sections = []
            for group_key, group_data in adapter_groups.items():
                adapter_type = group_data["adapter_type"]
                server_name = group_data["server_name"]
                channels = group_data["channels"]
                
                # Build chat_info opening tag
                info_section = f'<chat_info type="{adapter_type}" server="{server_name}" alias="{alias}">\n'
                
                # Add channels
                for channel in channels:
                    conversation_name = channel.get("conversation_name", "unknown")
                    is_focused = channel.get("is_focused", False)
                    focused_attr = f' focused={is_focused}'
                    info_section += f'<channel conversation_name="{conversation_name}"{focused_attr}>\n'
                
                info_section += '</chat_info>'
                info_sections.append(info_section)
            
            return "\n\n".join(info_sections)
            
        except Exception as e:
            logger.error(f"Error rendering chat info section: {e}", exc_info=True)
            return ""

    def _render_flat_chat_group(self, group: Dict[str, Any], options: Dict[str, Any]) -> str:
        """Render a group of consecutive chat messages."""
        try:
            items = group.get("items", [])
            if not items:
                return ""
            
            # Render messages
            message_lines = []
            for item in items:
                message_line = self._render_flat_chat_message(item, options)
                if message_line:
                    message_lines.append(message_line)
            
            if not message_lines:
                return ""
            
            # Wrap in chat_contents tags
            content = "\n".join(message_lines)
            return f"<chat_contents>\n{content}\n</chat_contents>"
            
        except Exception as e:
            logger.error(f"Error rendering flat chat group: {e}", exc_info=True)
            return ""

    def _render_flat_chat_message(self, item: Dict[str, Any], options: Dict[str, Any]) -> str:
        """ENHANCED: Render individual chat message with full conversation context."""
        try:
            is_agent = item.get("is_agent", False)
            sender_name = item.get("sender_name", "Unknown")
            text_content = item.get("text_content", "")
            is_edited = item.get("is_edited", False)
            time_marker = item.get("time_marker")
            
            # FIXED: Use conversation context that was already extracted during content extraction
            conversation_name = item.get("conversation_name", "")
            adapter_type = item.get("adapter_id") or ""      # FIXED: Handle None values
            server_name = item.get("server_name") or ""        # FIXED: Handle None values
            
            # Build opening chat_message tag with conversation context
            message_attrs = []
            if conversation_name:
                message_attrs.append(f'conversation_name="{conversation_name}"')
            if adapter_type:
                message_attrs.append(f'messenger="{adapter_type}"')
            if server_name:
                message_attrs.append(f'server="{server_name}"')
            # logger.critical("MESSAGE ATTRS: " + str(message_attrs) + "Item: " + str(item))
            
            if message_attrs:
                opening_tag = f'<chat_message {" ".join(message_attrs)}>'
            else:
                opening_tag = '<chat_message>'
            
            # Build message content
            if is_agent:
                # Agent messages: no sender tag
                message_content = text_content
            else:
                # Human messages: include sender tag
                message_content = f"<sender>{sender_name}</sender>{text_content}"
            
            # Add edited marker
            if is_edited:
                message_content += " (edited)"
            
            # Handle attachments
            node_data = item.get("node_data", {})
            attachment_text = self._render_flat_message_attachments(node_data)
            if attachment_text:
                message_content += f"\n{attachment_text}"
            
            # Build complete message with tags
            message_line = f"{opening_tag}{message_content}</chat_message>"
            
            # Add time marker if present
            if time_marker:
                message_line = f"<time_marker>{time_marker}</time_marker>\n{message_line}"
            
            return message_line
            
        except Exception as e:
            logger.error(f"Error rendering enhanced flat chat message: {e}", exc_info=True)
            return ""

    def _render_flat_message_attachments(self, node_data: Dict[str, Any]) -> str:
        """Render attachments for a message in flat view."""
        try:
            props = node_data.get("properties", {})
            attachment_metadata = props.get("attachment_metadata", [])
            
            if not attachment_metadata:
                return ""
            
            attachment_lines = []
            for att_meta in attachment_metadata:
                filename = att_meta.get('filename', 'unknown_file')
                att_type = att_meta.get('attachment_type', 'unknown')
                attachment_lines.append(f"[Attachment: {filename} (Type: {att_type})]")
            
            return "\n".join(attachment_lines)
            
        except Exception as e:
            logger.warning(f"Error rendering message attachments: {e}")
            return ""

    def _render_flat_scratchpad_group(self, group: Dict[str, Any], options: Dict[str, Any]) -> str:
        """Render a group of scratchpad content."""
        try:
            items = group.get("items", [])
            if not items:
                return ""
            
            # Check if there's any actual content before proceeding
            has_content = any(item.get("note_content", "").strip() for item in items)
            if not has_content:
                # FIXED: Don't render empty scratchpad with just time markers - it's confusing
                return ""
            
            # Render scratchpad items
            scratchpad_lines = []
            for item in items:
                note_content = item.get("note_content", "").strip()
                time_marker = item.get("time_marker")
                
                # Only add time marker if there's actual content
                if time_marker and note_content:
                    scratchpad_lines.append(f"<time_marker>{time_marker}</time_marker>")
                
                if note_content:
                    scratchpad_lines.append(note_content)
            
            if not scratchpad_lines:
                return ""
            
            # Wrap in scratchpad tags
            content = "\n".join(scratchpad_lines)
            return f"<scratchpad>\n{content}\n</scratchpad>"
            
        except Exception as e:
            logger.error(f"Error rendering flat scratchpad group: {e}", exc_info=True)
            return ""

    def _render_flat_memory_group(self, group: Dict[str, Any], options: Dict[str, Any]) -> str:
        """Render a group of memory content."""
        try:
            items = group.get("items", [])
            if not items:
                return ""
            
            # Render memory items
            memory_lines = []
            for item in items:
                memory_summary = item.get("memory_summary", "")
                memory_type = item.get("memory_type", "unknown")
                time_marker = item.get("time_marker")
                
                # NEW: Extract conversation context for memory
                conversation_name = item.get("conversation_name") or ""    # FIXED: Handle None values
                adapter_type = item.get("adapter_type") or ""              # FIXED: Handle None values
                server_name = item.get("server_name") or ""                # FIXED: Handle None values
                
                if time_marker:
                    memory_lines.append(f"<time_marker>{time_marker}</time_marker>")
                
                if memory_summary:
                    # Build memory tag with conversation context
                    memory_attrs = []
                    if conversation_name:
                        memory_attrs.append(f'conversation_name="{conversation_name}"')
                    if time_marker:
                        memory_attrs.append(f'timestamp="{time_marker}"')
                    
                    if memory_attrs:
                        memory_tag = f'<memory {" ".join(memory_attrs)}>'
                        closing_tag = '</memory>'
                    else:
                        memory_tag = f'<{memory_type}_memory>'
                        closing_tag = f'</{memory_type}_memory>'
                    
                    memory_lines.append(f"{memory_tag}{memory_summary}{closing_tag}")
            
            if not memory_lines:
                return ""
            
            content = "\n".join(memory_lines)
            return content  # Memory content doesn't need wrapper tags
            
        except Exception as e:
            logger.error(f"Error rendering flat memory group: {e}", exc_info=True)
            return ""

    def _render_flat_system_group(self, group: Dict[str, Any], options: Dict[str, Any]) -> str:
        """Render a group of system messages (edit notifications, etc.)."""
        try:
            items = group.get("items", [])
            if not items:
                return ""
            
            # Render system messages
            system_lines = []
            for item in items:
                system_text = item.get("system_text", "")
                time_marker = item.get("time_marker")
                
                if time_marker:
                    system_lines.append(f"<time_marker>{time_marker}</time_marker>")
                
                if system_text:
                    system_lines.append(f"[SYSTEM] {system_text}")
            
            if not system_lines:
                return ""
            
            content = "\n".join(system_lines)
            return content  # System messages don't need wrapper tags
            
        except Exception as e:
            logger.error(f"Error rendering flat system group: {e}", exc_info=True)
            return ""

    def _generate_system_messages(self, content_groups: List[Dict[str, Any]], flat_cache: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate system messages for edit/delete operations."""
        try:
            # For now, detect edits from message properties
            # Future: could track actual edit deltas
            
            for group in content_groups:
                if group.get("group_type") != "chat_message":
                    continue
                
                for item in group.get("items", []):
                    if item.get("is_edited", False):
                        # Create system message for this edit
                        system_message = self._create_edit_system_message(item)
                        if system_message:
                            # Insert system message at appropriate chronological position
                            # For now, add to same group
                            pass  # Implementation depends on exact requirements
            
            return content_groups
            
        except Exception as e:
            logger.error(f"Error generating system messages: {e}", exc_info=True)
            return content_groups

    def _create_edit_system_message(self, edited_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create system message for an edited message."""
        try:
            sender_name = edited_item.get("sender_name", "Unknown")
            text_content = edited_item.get("text_content", "")
            
            # Create preview (first 50 chars)
            preview = text_content[:50]
            truncated = ""
            if len(text_content) > 50:
                truncated = f" ({len(text_content) - 50} symbols truncated)"
            
            system_text = f'Message <sender>{sender_name}</sender> with content "{preview}"{truncated} was edited'
            
            return {
                "content_type": "system",
                "timestamp": edited_item.get("timestamp", 0),
                "system_text": system_text
            }
            
        except Exception as e:
            logger.warning(f"Error creating edit system message: {e}")
            return None
