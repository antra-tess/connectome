"""
FacetAware HUD Component

New HUD component that renders VEIL facets in temporal streams.
Replaces current hierarchical tree rendering with facet-aware logic while
maintaining interface compatibility with existing systems.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

from ..base_component import Component
from elements.component_registry import register_component
from ..veil import (
    VEILFacetCache, VEILFacet, VEILFacetType,
    EventFacet, StatusFacet, AmbientFacet, 
    TemporalConsistencyValidator
)
from ..space.space_veil_producer import SpaceVeilProducer

logger = logging.getLogger(__name__)

@dataclass
class RenderingOptions:
    """Configuration for facet stream rendering."""
    exclude_ambient: bool = False
    include_memory: bool = True
    focus_element_id: Optional[str] = None
    render_mode: str = "normal"  # "normal", "memorization"
    ambient_text_threshold: int = 500
    include_time_markers: bool = True
    include_system_messages: bool = True

@register_component
class FacetAwareHUDComponent(Component):
    """
    New HUD component that renders VEIL facets in temporal streams.
    
    Key Features:
    - Temporal stream rendering with facet-aware logic
    - Interface compatibility with existing HUD methods
    - Ambient facet positioning and triggering
    - CompressionEngine integration for memory operations
    - Legacy flat cache conversion utilities
    
    Core Principles:
    - Facets rendered in chronological order by veil_timestamp
    - Ambient facets float temporally based on status changes/thresholds  
    - EventFacets and StatusFacets maintain strict temporal positioning
    - Memory facets integrate seamlessly into temporal stream
    """
    
    COMPONENT_TYPE = "FacetAwareHUDComponent"
    
    def initialize(self, **kwargs) -> None:
        """Initialize the facet-aware HUD component."""
        super().initialize(**kwargs)
        self._rendering_stats = {
            "total_renders": 0,
            "facets_processed": 0,
            "ambient_renders": 0,
            "memory_context_renders": 0
        }
        logger.debug(f"FacetAwareHUDComponent initialized for Element {self.owner.id}")

    def _get_space_veil_producer(self) -> Optional[SpaceVeilProducer]:
        """Get the SpaceVeilProducer from the owning InnerSpace."""
        if not self.owner:
            return None
        return self.get_sibling_component("SpaceVeilProducer")

    async def get_agent_context_via_compression_engine(self, 
                                                 options: Optional[Dict[str, Any]] = None,
                                                 tools: Optional[List[Dict[str, Any]]] = None) -> Union[str, Dict[str, Any]]:
        """
        INTERFACE PRESERVED: Main entry point for agent loops to get context.
        
        NEW IMPLEMENTATION: Pass VEILFacetCache to CompressionEngine instead of flat VEIL cache.
        
        Args:
            options: Rendering options (unchanged interface)
            tools: Enhanced tool definitions from VEIL (unchanged interface)
            
        Returns:
            Rendered context string or multimodal dict (unchanged interface)
        """
        try:
            self._rendering_stats["total_renders"] += 1
            options = options or {}
            
            # Get VEILFacetCache instead of flat cache
            veil_producer = self._get_space_veil_producer()
            if not veil_producer:
                logger.error(f"[{self.owner.id}] Cannot generate context: SpaceVeilProducer not available")
                return "Error: VEILFacetCache not available"
            
            facet_cache = veil_producer.get_facet_cache()  # NEW: Get VEILFacetCache
            
            # Get VEILFacetCompressionEngine
            compression_engine = self.get_sibling_component("VEILFacetCompressionEngine") 
            if not compression_engine:
                logger.warning(f"[{self.owner.id}] VEILFacetCompressionEngine not available, using direct rendering")
                return await self._render_temporal_facet_stream_direct(facet_cache, options, tools)
            
            # NEW: Pass VEILFacetCache to CompressionEngine (Option A from clarification)
            processed_facets = await compression_engine.process_facet_cache_with_compression(
                facet_cache=facet_cache,
                focus_context=options.get('focus_context')
            )
            
            # NEW: Render processed facets using temporal stream rendering
            context_string = await self._render_temporal_facet_stream(processed_facets, options, tools)
            
            # Check for multimodal content if focus element provided
            focus_element_id = options.get('focus_context', {}).get('focus_element_id')
            if focus_element_id:
                return await self._detect_and_extract_multimodal_content(context_string, options, focus_element_id)
            
            return context_string
            
        except Exception as e:
            logger.error(f"Error in facet-aware context generation: {e}", exc_info=True)
            return "Error: Facet context generation failed"

    # REMOVED: render_memorization_context_with_flat_cache() - replaced by render_memorization_context_with_facet_cache()
    # The flat cache interface is no longer used after VEILFacet architecture completion

    async def render_memorization_context_with_facet_cache(self, 
                                                         facet_cache: VEILFacetCache,
                                                         exclude_element_id: str,
                                                         focus_element_id: Optional[str] = None) -> str:
        """
        NEW: Native VEILFacet compression context rendering (eliminates flat cache conversions).
        
        This replaces the round-trip conversion pattern:
        VEILFacetCache → flat_cache → VEILFacetCache
        
        MEMORIZATION RULES:
        - Skip ambient facets (they float temporally and aren't memorizable)  
        - Focus on element_id context if provided
        - Exclude EventFacets from target container to avoid duplication
        - Use temporal ordering for chronological context
        
        Args:
            facet_cache: Native VEILFacetCache to render
            exclude_element_id: Element ID where content is being memorized  
            focus_element_id: Optional focus element for context
            
        Returns:
            Rendered memorization context string
        """
        try:
            self._rendering_stats["memory_context_renders"] += 1
            
            # Apply memorization exclusion rules directly to VEILFacetCache
            filtered_facets = self._apply_memorization_exclusions_native(
                facet_cache, exclude_element_id
            )
            
            # Render temporal stream without ambient facets
            memorization_options = RenderingOptions(
                exclude_ambient=True,  # Skip ambient facets for memorization
                focus_element_id=focus_element_id,
                render_mode="memorization"
            )
            
            context = await self._render_temporal_facet_stream(filtered_facets, memorization_options.__dict__)
            
            logger.debug(f"Generated native VEILFacet memorization context: {len(context)} chars")
            return context
            
        except Exception as e:
            logger.error(f"Error in native VEILFacet memorization context: {e}", exc_info=True)
            return f"Error rendering memorization context: {e}"

    def _apply_memorization_exclusions_native(self, 
                                            facet_cache: VEILFacetCache, 
                                            exclude_element_id: str) -> VEILFacetCache:
        """
        Apply memorization exclusion rules directly to VEILFacetCache (no conversions).
        
        Args:
            facet_cache: Source VEILFacetCache
            exclude_element_id: Element ID where content is being memorized
            
        Returns:
            Filtered VEILFacetCache with exclusions applied
        """
        try:
            filtered_cache = VEILFacetCache()
            
            for facet in facet_cache.facets.values():
                # Skip ambient facets (they float temporally and aren't memorizable)
                if facet.facet_type == VEILFacetType.AMBIENT:
                    continue
                    
                # Skip EventFacets linked to the excluded element (the content being compressed)
                if (facet.facet_type == VEILFacetType.EVENT and 
                    facet.links_to == exclude_element_id):
                    continue
                    
                # Include all other facets
                filtered_cache.add_facet(facet)
                    
            logger.debug(f"Applied native memorization exclusions: {len(facet_cache)} → {len(filtered_cache)} facets")
            return filtered_cache
            
        except Exception as e:
            logger.error(f"Error applying native memorization exclusions: {e}", exc_info=True)
            return facet_cache  # Return unfiltered cache on error

    async def _render_temporal_facet_stream(self, 
                                          facet_cache: VEILFacetCache, 
                                          options: Dict[str, Any],
                                          tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Render complete temporal stream with stateful consolidated structural logic.
        
        NEW APPROACH: Instead of facet-by-facet rendering, we maintain system state
        and render consolidated structural sections when status changes occur.
        
        Key Features:
        - Agent workspace wrapper with metadata
        - Focus state tracking and updates after agent turns
        - Consolidated chat_info sections showing all active channels
        - Proper ambient tool positioning (status changes + content thresholds)
        - System message generation for key events
        - Status facet metadata tracking
        
        Args:
            facet_cache: VEILFacetCache with facets to render
            options: Rendering options and configuration
            tools: Optional enhanced tool definitions
            
        Returns:
            Rendered temporal stream as consolidated structural format
        """
        try:
            render_opts = RenderingOptions(**{k: v for k, v in options.items() if hasattr(RenderingOptions, k)})
            
            # Get chronologically ordered facets (all types)
            temporal_stream = facet_cache.get_chronological_stream(include_ambient=True)
            
            if not temporal_stream:
                return self._render_empty_context(render_opts)
            
            # Validate temporal consistency
            violations = TemporalConsistencyValidator.validate_temporal_stream(temporal_stream)
            if violations:
                logger.warning(f"Temporal violations detected: {violations}")
            
            # Initialize system state tracking
            system_state = self._initialize_system_state(temporal_stream, options)
            rendered_sections = []
            content_since_last_status = []
            content_symbols_since_ambient = 0
            
            # Track facets processed for stats
            self._rendering_stats["facets_processed"] += len(temporal_stream)
            
            # 1. Render initial agent workspace wrapper
            workspace_section = self._render_agent_workspace_wrapper(system_state)
            if workspace_section:
                rendered_sections.append(workspace_section)
            
            # 2. Process temporal stream with state tracking
            for i, facet in enumerate(temporal_stream):
                
                # Update system state based on facet
                state_changed = self._update_system_state(system_state, facet, options)
                
                if facet.facet_type == VEILFacetType.STATUS:
                    # Status change: render consolidated system state
                    if state_changed:
                        # First render any accumulated content
                        if content_since_last_status:
                            content_section = "\n".join(content_since_last_status)
                            rendered_sections.append(content_section)
                            content_since_last_status = []
                        
                        # Render consolidated chat_info with current system state
                        chat_info_section = self._render_consolidated_chat_info(system_state, tools)
                        if chat_info_section:
                            rendered_sections.append(chat_info_section)
                        
                        # Check if we should render ambient tools after status change
                        if not render_opts.exclude_ambient:
                            ambient_section = self._render_ambient_tools_section(facet_cache, system_state)
                            if ambient_section:
                                rendered_sections.append(ambient_section)
                                content_symbols_since_ambient = 0
                                self._rendering_stats["ambient_renders"] += 1
                        
                        # Generate system message for status change
                        system_message = self._generate_system_message_for_status(facet, system_state)
                        if system_message:
                            content_since_last_status.append(system_message)
                
                elif facet.facet_type == VEILFacetType.EVENT:
                    # Event facet: render as content and track focus changes
                    event_content = self._render_event_facet_with_system_context(facet, system_state, render_opts)
                    if event_content:
                        content_since_last_status.append(event_content)
                        content_symbols_since_ambient += len(event_content)
                        
                        # Check for focus changes after agent responses
                        if facet.get_property("event_type") == "agent_response":
                            focus_changed = self._update_focus_after_agent_turn(system_state, facet)
                            if focus_changed:
                                # Re-render consolidated state after focus change
                                if content_since_last_status:
                                    content_section = "\n".join(content_since_last_status)
                                    rendered_sections.append(content_section)
                                    content_since_last_status = []
                                
                                chat_info_section = self._render_consolidated_chat_info(system_state, tools)
                                if chat_info_section:
                                    rendered_sections.append(chat_info_section)
                        
                        # Check ambient threshold trigger
                        if (content_symbols_since_ambient >= render_opts.ambient_text_threshold and 
                            not render_opts.exclude_ambient):
                            ambient_section = self._render_ambient_tools_section(facet_cache, system_state)
                            if ambient_section:
                                content_since_last_status.append(ambient_section)
                                content_symbols_since_ambient = 0
                                self._rendering_stats["ambient_renders"] += 1
                
                elif facet.facet_type == VEILFacetType.AMBIENT:
                    # Ambient facets are rendered via _render_ambient_tools_section, skip individual rendering
                    continue
            
            # 3. Render any remaining content
            if content_since_last_status:
                content_section = "\n".join(content_since_last_status)
                rendered_sections.append(content_section)
            
            # 4. Final ambient render if needed
            if not render_opts.exclude_ambient and content_symbols_since_ambient > 0:
                final_ambient = self._render_ambient_tools_section(facet_cache, system_state)
                if final_ambient:
                    rendered_sections.append(final_ambient)
                    self._rendering_stats["ambient_renders"] += 1
            
            # 5. Close agent workspace wrapper
            rendered_sections.append("</inner_space>")
            
            # Combine all sections
            full_context = "\n\n".join(filter(None, rendered_sections))
            
            logger.debug(f"Rendered structural temporal stream: {len(full_context)} chars, {len(rendered_sections)} sections")
            return full_context
            
        except Exception as e:
            logger.error(f"Error rendering temporal facet stream: {e}", exc_info=True)
            return "Error: Failed to render temporal facet stream"

    def _initialize_system_state(self, temporal_stream: List[VEILFacet], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize system state tracking from temporal stream and options.
        
        Extracts agent info, active channels, adapters, and initial focus state.
        """
        system_state = {
            # Agent workspace info
            "agent_name": "Unknown Agent",
            "agent_description": "Agent workspace",
            "workspace_element_id": None,
            
            # Active channels by element_id
            "active_channels": {},  # element_id -> channel_info
            
            # Adapter groups for chat_info rendering
            "adapter_groups": {},  # "adapter_type:server_name" -> group_info
            
            # Focus tracking
            "focused_element_id": options.get('focus_context', {}).get('focus_element_id'),
            "last_agent_response_container": None,
            
            # Timing and state
            "last_status_timestamp": 0,
            "system_initialized": False
        }
        
        # Extract agent and workspace info from space_created status facets
        for facet in temporal_stream:
            if (facet.facet_type == VEILFacetType.STATUS and 
                facet.get_property("status_type") == "space_created"):
                current_state = facet.get_property("current_state", {})
                system_state["agent_name"] = current_state.get("agent_name", system_state["agent_name"])
                system_state["agent_description"] = current_state.get("agent_description", system_state["agent_description"])
                system_state["workspace_element_id"] = current_state.get("element_id")
                break
        
        return system_state

    def _update_system_state(self, system_state: Dict[str, Any], facet: VEILFacet, options: Dict[str, Any]) -> bool:
        """
        Update system state based on facet. Returns True if structural state changed.
        """
        state_changed = False
        
        if facet.facet_type == VEILFacetType.STATUS:
            status_type = facet.get_property("status_type")
            current_state = facet.get_property("current_state", {})
            
            if status_type == "container_created" and current_state.get("conversation_name"):
                # New chat container created
                element_id = current_state.get("element_id")
                conversation_name = current_state.get("conversation_name")
                
                channel_info = {
                    "element_id": element_id,
                    "conversation_name": conversation_name,
                    "adapter_type": current_state.get("adapter_type", "unknown"),
                    "server_name": current_state.get("server_name", "default"),
                    "alias": current_state.get("alias", "unknown"),
                    "message_count": current_state.get("message_count", 0),
                    "is_focused": element_id == system_state["focused_element_id"],
                    "available_tools": current_state.get("available_tools", []),
                    "tool_target_element_id": current_state.get("tool_target_element_id")
                }
                
                system_state["active_channels"][element_id] = channel_info
                
                # Update adapter groups
                adapter_key = f"{channel_info['adapter_type']}:{channel_info['server_name']}"
                if adapter_key not in system_state["adapter_groups"]:
                    system_state["adapter_groups"][adapter_key] = {
                        "adapter_type": channel_info["adapter_type"],
                        "server_name": channel_info["server_name"],
                        "alias": channel_info["alias"],
                        "channels": []
                    }
                
                system_state["adapter_groups"][adapter_key]["channels"].append(channel_info)
                state_changed = True
                
            system_state["last_status_timestamp"] = facet.veil_timestamp
        
        return state_changed

    def _update_focus_after_agent_turn(self, system_state: Dict[str, Any], facet: EventFacet) -> bool:
        """
        Update focus state after agent response. Returns True if focus changed.
        """
        if facet.get_property("event_type") != "agent_response":
            return False
        
        # Determine which container this response was sent to
        links_to = getattr(facet, 'links_to', None)
        target_container = None
        
        if links_to:
            # Find the container this response links to
            for element_id, channel_info in system_state["active_channels"].items():
                if element_id in links_to:
                    target_container = element_id
                    break
        
        if not target_container:
            # Fallback: use owner_element_id
            target_container = facet.owner_element_id
        
        # Update focus if it changed
        previous_focus = system_state["focused_element_id"]
        if target_container != previous_focus and target_container in system_state["active_channels"]:
            # Update focus state
            if previous_focus and previous_focus in system_state["active_channels"]:
                system_state["active_channels"][previous_focus]["is_focused"] = False
            
            system_state["active_channels"][target_container]["is_focused"] = True
            system_state["focused_element_id"] = target_container
            system_state["last_agent_response_container"] = target_container
            
            # Update adapter groups
            for adapter_group in system_state["adapter_groups"].values():
                for channel in adapter_group["channels"]:
                    channel["is_focused"] = (channel["element_id"] == target_container)
            
            logger.debug(f"Focus changed to {target_container} after agent response")
            return True
        
        return False

    def _render_agent_workspace_wrapper(self, system_state: Dict[str, Any]) -> str:
        """Render opening agent workspace wrapper with metadata."""
        agent_name = system_state["agent_name"]
        agent_description = system_state["agent_description"]
        
        if agent_name and agent_name != "Unknown Agent":
            opening_tag = f'<inner_space agent_name="{agent_name}"'
            if agent_description and agent_description != "Agent workspace":
                opening_tag += f' agent_description="{agent_description}"'
            opening_tag += '>'
        else:
            opening_tag = '<inner_space>'
        
        return opening_tag

    def _render_consolidated_chat_info(self, system_state: Dict[str, Any], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Render consolidated chat_info sections with all active channels and focus states.
        
        Groups channels by adapter and renders complete system state at this moment.
        """
        if not system_state["adapter_groups"]:
            return ""
        
        info_sections = []
        
        for adapter_group in system_state["adapter_groups"].values():
            adapter_type = adapter_group["adapter_type"]
            server_name = adapter_group["server_name"]
            alias = adapter_group["alias"]
            channels = adapter_group["channels"]
            
            if not channels:
                continue
            
            # Build chat_info opening tag
            info_section = f'<chat_info type="{adapter_type}" server="{server_name}" alias="{alias}">'
            channel_lines = []
            
            # Add all channels in this adapter group
            for channel in channels:
                conversation_name = channel["conversation_name"]
                is_focused = channel["is_focused"]
                available_tools = channel.get("available_tools", [])
                
                # Build tool list for this channel
                if available_tools:
                    tool_names = [tool.get("name", tool.get("tool_name", "unknown")) for tool in available_tools]
                    tools_attr = f' tools="({", ".join(tool_names)})"'
                else:
                    tools_attr = ''
                
                # Build channel line
                focused_attr = ' focused="True"' if is_focused else ' focused="False"'
                channel_line = f'<channel conversation_name="{conversation_name}"{focused_attr}{tools_attr}>'
                channel_lines.append(channel_line)
            
            info_section += "\n" + "\n".join(channel_lines) + "\n</chat_info>"
            info_sections.append(info_section)
        
        return "\n\n".join(info_sections)

    def _render_ambient_tools_section(self, facet_cache: VEILFacetCache, system_state: Dict[str, Any]) -> str:
        """
        Render ambient tools section when triggered by status changes or content thresholds.
        """
        ambient_facets = facet_cache.get_ambient_facets()
        
        if not ambient_facets:
            return ""
        
        tool_sections = []
        for ambient_facet in ambient_facets:
            ambient_type = ambient_facet.get_property("ambient_type")
            content = ambient_facet.get_property("content", "")
            
            if ambient_type == "tool_instructions" and content:
                tool_sections.append(content)
        
        if tool_sections:
            return "<tool_use_instructions>\n" + "\n".join(tool_sections) + "\n</tool_use_instructions>"
        
        return ""

    def _generate_system_message_for_status(self, status_facet: StatusFacet, system_state: Dict[str, Any]) -> str:
        """Generate system messages for status changes."""
        status_type = status_facet.get_property("status_type")
        current_state = status_facet.get_property("current_state", {})
        
        if status_type == "container_created" and current_state.get("conversation_name"):
            conversation_name = current_state.get("conversation_name")
            return f"<system>Connected to conversation: {conversation_name}</system>"
        elif status_type == "space_created":
            agent_name = current_state.get("agent_name", "Agent")
            return f"<system>Agent workspace initialized: {agent_name}</system>"
        elif status_type == "chat_joined":
            chat_name = current_state.get("conversation_name", "Unknown")
            return f"<system>Joined conversation: {chat_name}</system>"
        
        return ""

    def _render_event_facet_with_system_context(self, facet: EventFacet, system_state: Dict[str, Any], options: RenderingOptions) -> str:
        """Render event facets with system context and focus awareness."""
        try:
            event_type = facet.get_property("event_type")
            content = facet.get_property("content", "")
            
            if event_type == "message_added":
                sender = facet.get_property("sender_name", "Unknown")
                
                # Determine conversation from links or owner
                conversation_name = self._determine_conversation_for_event(facet, system_state)
                
                # Build message with conversation context
                msg_attrs = []
                if conversation_name:
                    msg_attrs.append(f'source="{conversation_name}"')
                if sender:
                    msg_attrs.append(f'sender="{sender}"')
                
                msg_tag = f'<msg {" ".join(msg_attrs)}>' if msg_attrs else '<msg>'
                
                # Handle reactions and status
                reactions = facet.get_property("reactions", {})
                message_status = facet.get_property("message_status", "received")
                
                message_content = content
                
                # Add reactions if present
                if reactions:
                    reaction_text = ", ".join([f"{emoji}: {len(users)}" for emoji, users in reactions.items()])
                    message_content += f" [Reactions: {reaction_text}]"
                
                # Add status indicators
                if message_status == "pending_send":
                    message_content += " [PENDING CONFIRMATION]"
                elif message_status == "failed_to_send":
                    message_content += " [SEND FAILED]"
                
                return f"{msg_tag}{message_content}</msg>"
                
            elif event_type == "agent_response":
                agent_name = facet.get_property("agent_name", "Agent")
                tool_calls_count = facet.get_property("tool_calls_count", 0)
                
                tool_info = f' tool_calls="{tool_calls_count}"' if tool_calls_count > 0 else ""
                response_content = f'<agent_response agent="{agent_name}"{tool_info}>{content}</agent_response>'
                
                # Add system message for successful delivery if needed
                message_status = facet.get_property("message_status", "sent")
                if message_status == "sent":
                    conversation_name = self._determine_conversation_for_event(facet, system_state)
                    delivery_message = f"<system>Message delivered to {conversation_name}</system>"
                    return f"{response_content}\n{delivery_message}"
                
                return response_content
                
            elif event_type == "note_created":
                return f"<note>{content}</note>"
                
            elif event_type == "message_edited":
                original_msg_id = facet.get_property("original_message_id", "unknown")
                sender = facet.get_property("sender_name", "Unknown")
                return f"<system>Message from {sender} (ID: {original_msg_id}) was edited: {content}</system>"
                
            else:
                return f"<event type=\"{event_type}\">{content}</event>"
                
        except Exception as e:
            logger.error(f"Error rendering event facet: {e}", exc_info=True)
            return ""

    def _determine_conversation_for_event(self, facet: EventFacet, system_state: Dict[str, Any]) -> str:
        """Determine conversation name for an event facet."""
        # Try to get from facet properties first
        conversation_name = facet.get_property("conversation_name")
        if conversation_name:
            return conversation_name
        
        # Try to determine from links_to relationships
        links_to = getattr(facet, 'links_to', None)
        if links_to:
            for element_id in links_to:
                if element_id in system_state["active_channels"]:
                    return system_state["active_channels"][element_id]["conversation_name"]
        
        # Fallback to owner element
        owner_element_id = facet.owner_element_id
        if owner_element_id in system_state["active_channels"]:
            return system_state["active_channels"][owner_element_id]["conversation_name"]
        
        return "Unknown"

    def _render_single_facet(self, facet: VEILFacet, options: RenderingOptions) -> str:
        """Render individual facet based on type."""
        
        try:
            if facet.facet_type == VEILFacetType.EVENT:
                return self._render_event_facet(facet, options)
            elif facet.facet_type == VEILFacetType.STATUS:
                return self._render_status_facet(facet, options)
            elif facet.facet_type == VEILFacetType.AMBIENT:
                return self._render_ambient_facet(facet, options)
            else:
                logger.warning(f"Unknown facet type: {facet.facet_type}")
                return ""
                
        except Exception as e:
            logger.error(f"Error rendering facet {facet.facet_id}: {e}", exc_info=True)
            return ""

    def _render_event_facet(self, facet: EventFacet, options: RenderingOptions) -> str:
        """Render event facets (messages, notes, responses)."""
        try:
            event_type = facet.get_property("event_type")
            content = facet.get_property("content", "")
            
            if event_type == "message_added":
                sender = facet.get_property("sender_name", "Unknown")
                conversation = facet.get_property("conversation_name", "")
                
                # Build message with conversation context
                msg_attrs = []
                if conversation:
                    msg_attrs.append(f'source="{conversation}"')
                if sender:
                    msg_attrs.append(f'sender="{sender}"')
                
                msg_tag = f'<msg {" ".join(msg_attrs)}>' if msg_attrs else '<msg>'
                
                # Handle reactions and status
                reactions = facet.get_property("reactions", {})
                message_status = facet.get_property("message_status", "received")
                
                message_content = content
                
                # Add reactions if present
                if reactions:
                    reaction_text = ", ".join([f"{emoji}: {len(users)}" for emoji, users in reactions.items()])
                    message_content += f" [Reactions: {reaction_text}]"
                
                # Add status indicators
                if message_status == "pending_send":
                    message_content += " [PENDING CONFIRMATION]"
                elif message_status == "failed_to_send":
                    message_content += " [SEND FAILED]"
                
                return f"{msg_tag}{message_content}</msg>"
                
            elif event_type == "note_created":
                return f"<note>{content}</note>"
                
            elif event_type == "agent_response":
                agent_name = facet.get_property("agent_name", "Agent")
                tool_calls_count = facet.get_property("tool_calls_count", 0)
                
                tool_info = f' tool_calls="{tool_calls_count}"' if tool_calls_count > 0 else ""
                return f'<agent_response agent="{agent_name}"{tool_info}>{content}</agent_response>'
                
            elif event_type == "message_edited":
                original_msg_id = facet.get_property("original_message_id", "unknown")
                sender = facet.get_property("sender_name", "Unknown")
                return f"<system>Message from {sender} (ID: {original_msg_id}) was edited: {content}</system>"
                
            else:
                return f"<event type=\"{event_type}\">{content}</event>"
                
        except Exception as e:
            logger.error(f"Error rendering event facet: {e}", exc_info=True)
            return ""

    def _render_status_facet(self, facet: StatusFacet, options: RenderingOptions) -> str:
        """Render status facets (containers, system state)."""
        try:
            status_type = facet.get_property("status_type")
            current_state = facet.get_property("current_state", {})
            
            if status_type == "container_created":
                container_name = current_state.get("element_name", "Unknown")
                conversation_name = current_state.get("conversation_name", "")
                message_count = current_state.get("message_count", 0)
                
                if conversation_name:
                    return f"<chat_info conversation=\"{conversation_name}\" messages=\"{message_count}\">Connected to {container_name}</chat_info>"
                else:
                    return f"<status>Container {container_name} created with {message_count} messages</status>"
                    
            elif status_type == "space_created":
                space_name = current_state.get("element_name", "Unknown")
                return f"<status>Agent workspace: {space_name}</status>"
                
            elif status_type == "chat_joined":
                chat_name = current_state.get("conversation_name", "Unknown")
                return f"<status>Joined conversation: {chat_name}</status>"
                
            else:
                return f"<status type=\"{status_type}\">{current_state}</status>"
                
        except Exception as e:
            logger.error(f"Error rendering status facet: {e}", exc_info=True)
            return ""

    def _render_ambient_facet(self, facet: AmbientFacet, options: RenderingOptions) -> str:
        """Render ambient facets (tools, instructions)."""
        try:
            ambient_type = facet.get_property("ambient_type")
            content = facet.get_property("content", "")
            
            if ambient_type == "tool_instructions":
                return f"<ambient_tools>{content}</ambient_tools>"
            else:
                return f"<ambient type=\"{ambient_type}\">{content}</ambient>"
                
        except Exception as e:
            logger.error(f"Error rendering ambient facet: {e}", exc_info=True)
            return ""

    def _render_ambient_facets(self, facet_cache: VEILFacetCache, options: RenderingOptions) -> str:
        """Render all ambient facets as floating context."""
        try:
            ambient_facets = facet_cache.get_ambient_facets()
            
            if not ambient_facets:
                return ""
                
            ambient_sections = []
            for ambient_facet in ambient_facets:
                ambient_content = self._render_ambient_facet(ambient_facet, options)
                if ambient_content:
                    ambient_sections.append(ambient_content)
                    
            return "\n".join(ambient_sections)
            
        except Exception as e:
            logger.error(f"Error rendering ambient facets: {e}", exc_info=True)
            return ""

    # REMOVED: _convert_flat_cache_to_facets(), _convert_node_to_facet(), _apply_memorization_exclusions()
    # These flat cache conversion methods are no longer needed after VEILFacet architecture completion

    async def _render_temporal_facet_stream_direct(self, 
                                                 facet_cache: VEILFacetCache,
                                                 options: Dict[str, Any],
                                                 tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Direct rendering without CompressionEngine (fallback).
        
        Used when CompressionEngine is not available.
        """
        try:
            return await self._render_temporal_facet_stream(facet_cache, options, tools)
            
        except Exception as e:
            logger.error(f"Error in direct temporal rendering: {e}", exc_info=True)
            return "Error: Direct temporal rendering failed"

    def _render_empty_context(self, options: RenderingOptions) -> str:
        """Render context when no facets are available."""
        if options.render_mode == "memorization":
            return "No temporal context available for memorization"
        else:
            return "[AGENT WORKSPACE]\nNo content available"

    async def _detect_and_extract_multimodal_content(self, 
                                                   text_context: str, 
                                                   options: Dict[str, Any],
                                                   focus_element_id: str) -> Union[str, Dict[str, Any]]:
        """
        Check for and extract multimodal content from facets.
        
        This maintains compatibility with existing multimodal handling.
        """
        try:
            # For now, return text-only (multimodal support can be added later)
            # This maintains interface compatibility
            return text_context
            
        except Exception as e:
            logger.error(f"Error in multimodal detection: {e}", exc_info=True)
            return text_context

    def get_rendering_statistics(self) -> Dict[str, Any]:
        """
        Get rendering statistics for debugging and monitoring.
        
        Returns:
            Dictionary with rendering statistics
        """
        return self._rendering_stats.copy() 