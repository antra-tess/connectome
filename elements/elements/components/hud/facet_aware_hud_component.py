"""
FacetAware HUD Component

New HUD component that renders VEIL facets in temporal streams.
Replaces current hierarchical tree rendering with facet-aware logic while
maintaining interface compatibility with existing systems.
"""

import logging
import json
import time
import copy
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import asyncio
import os

from ..base_component import Component
from elements.component_registry import register_component
from ..veil import (
    VEILFacetCache, VEILFacet, VEILFacetType, ConnectomeEpoch,
    EventFacet, StatusFacet, AmbientFacet,
)
from ..veil.facet_cache import TemporalConsistencyValidator
from ..space.space_veil_producer import SpaceVeilProducer

logger = logging.getLogger(__name__)

# NEW: Define minor status types that don't trigger ambient facets
MINOR_STATUS_TYPES = {
    "message_count_updated",   # Count changes, same capabilities
    "connection_status_updated", # Network state, same functionality
    "read_status_updated",     # Read/unread markers, same context
    "activity_status_updated", # Typing indicators, transient
    "user_presence_updated",   # Online/offline status, not structural
    "notification_received"    # Notifications, not capabilities
}

@dataclass
class SnapshotEntry:
    data: Dict[str, Any]
    timestamp_ms: int
    access_count: int = 0


class SnapshotCache:
    def __init__(self, ttl_ms: int = 5000):
        self._cache: Dict[str, SnapshotEntry] = {}
        self._ttl_ms = ttl_ms
        self._metrics = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "refreshes": 0
        }
        
    def get(self, key: str, max_age_ms: Optional[int] = None) -> Optional[Dict[str, Any]]:
        now = int(time.time() * 1000)
        max_age = max_age_ms or self._ttl_ms
        
        if key in self._cache:
            entry = self._cache[key]
            age = now - entry.timestamp_ms
            if age <= max_age:
                entry.access_count += 1
                self._metrics["hits"] += 1
                logger.debug(f"Snapshot cache hit: {key}, age={age}ms")
                return entry.data
            else:
                del self._cache[key]
                self._metrics["evictions"] += 1
                
        self._metrics["misses"] += 1
        return None
        
    def put(self, key: str, data: Dict[str, Any]) -> None:
        self._cache[key] = SnapshotEntry(
            data=data,
            timestamp_ms=int(time.time() * 1000)
        )
        self._metrics["refreshes"] += 1
        
    def get_metrics(self) -> Dict[str, Any]:
        """Return cache metrics with hit rate calculation."""
        total = self._metrics["hits"] + self._metrics["misses"]
        hit_rate = self._metrics["hits"] / total if total > 0 else 0
        return {
            **self._metrics,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache)
        }


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
    render_agent_messages: bool = False

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

    NEW: Three-Phase Ambient Facet Approach (Fixed)
    ==============================================

    PHASE 1: Natural Chronological Placement
    - Ambient facets assigned to turns based on their temporal occurrence
    - Simple chronological placement, no logic applied yet

    PHASE 2: Additive Triggered Placement
    - Identify turns with triggers (status changes, content thresholds, first turn)
    - ADD all available ambient facets to triggered turns (no filtering!)
    - Ambient facets are never lost, only added to additional locations

    PHASE 3: Simple Reverse Deduplication
    - Iterate backwards through turns (latest to earliest)
    - Keep ambient facets only in their LATEST occurrence
    - Remove earlier occurrences of the same ambient facet

    Benefits of New Approach:
    - No ambient facets are ever lost (only moved)
    - Simpler logic: place → trigger → deduplicate
    - Consistent behavior regardless of trigger type
    - Each ambient facet appears exactly once in optimal location

    Example:
    Turn 0: [tool_A, tool_B] + status change → triggers → [tool_A, tool_B, tool_C]
    Turn 2: [tool_A, tool_C] + status change → triggers → [tool_A, tool_C, tool_B]

    After deduplication:
    Turn 0: [] (all moved to later turn)
    Turn 2: [tool_A, tool_C, tool_B] (latest occurrence)
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
        self._agent_name = "Agent"
        self._focus_change_history = []
        self._current_focus = None
        # NEW: Enhanced snapshot cache with TTL and metrics
        ttl_ms = kwargs.get('snapshot_ttl_ms', 5000)
        self._snapshot_cache = SnapshotCache(ttl_ms=ttl_ms)
        # Start metrics logging task
        self._metrics_task = asyncio.create_task(self._log_metrics_periodically())
        logger.debug(f"FacetAwareHUDComponent initialized for Element {self.owner.id}")
        logger.info("Turn-based rendering with sophisticated status repetition and ambient threshold rules is active")

    async def _log_metrics_periodically(self):
        """Log snapshot cache metrics periodically."""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                metrics = self._snapshot_cache.get_metrics()
                logger.info(f"[HUD Snapshot Cache] Metrics: {metrics}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Metrics logging error: {e}")

    def cleanup(self) -> bool:
        """Clean up the component including stopping metrics task."""
        try:
            if hasattr(self, '_metrics_task'):
                self._metrics_task.cancel()
        except Exception as e:
            logger.debug(f"Error cancelling metrics task: {e}")
        return super().cleanup()

    def _extract_element_type_from_id(self, element_id: str) -> str:
        """Extract element type from element ID."""
        try:
            # Element IDs typically contain adapter type information
            # Format examples: "discord_conversation_...", "terminal_...", etc.
            if not element_id:
                return "unknown"
                
            # Try to extract from common patterns
            if "conversation" in element_id.lower():
                return "conversation"
            elif "terminal" in element_id.lower():
                return "terminal"
            elif "scratchpad" in element_id.lower():
                return "scratchpad"
            elif "file" in element_id.lower():
                return "file"
            else:
                # Extract first part before underscore as type
                parts = element_id.split('_')
                return parts[0] if parts else "unknown"
        except Exception as e:
            logger.debug(f"Error extracting element type from ID {element_id}: {e}")
            return "unknown"

    def _extract_element_name_from_id(self, element_id: str) -> str:
        """Extract element name from element ID."""
        try:
            if not element_id:
                return "Unknown Element"
                
            # For conversation elements, try to extract conversation name
            if "conversation" in element_id.lower():
                # Element IDs might contain conversation names after certain patterns
                # This is a simplified extraction - could be enhanced based on actual ID format
                parts = element_id.split('_')
                if len(parts) > 2:
                    # Try to find a meaningful name part
                    for part in parts:
                        if part and part not in ['conversation', 'discord', 'slack', 'terminal']:
                            return part.replace('-', ' ').title()
                return "Conversation"
            else:
                # For non-conversation elements, use a generic name
                element_type = self._extract_element_type_from_id(element_id)
                return f"{element_type.title()} Element"
        except Exception as e:
            logger.debug(f"Error extracting element name from ID {element_id}: {e}")
            return "Unknown Element"

    def record_focus_change(self, focus_info: Dict[str, Any]) -> None:
        """Record focus change history for debugging."""
        focus_element_id = focus_info.get('focus_element_id')
        if not focus_element_id:
            logger.debug("Cannot record focus change: no focus_element_id provided")
            return
            
        # Extract element type and name from element_id if not provided
        focus_element_type = focus_info.get('focus_element_type', self._extract_element_type_from_id(focus_element_id))
        focus_element_name = focus_info.get('focus_element_name', self._extract_element_name_from_id(focus_element_id))
        
        self._focus_change_history.append({
            "veil_timestamp": ConnectomeEpoch.get_veil_timestamp(),
            "owner_element_id": focus_element_id,
            "current_state": {
                "focused_element_id": focus_element_id,
                "focused_element_type": focus_element_type,
                "focused_element_name": focus_element_name
            }
        })

    def _get_space_veil_producer(self) -> Optional[SpaceVeilProducer]:
        """Get the SpaceVeilProducer from the owning InnerSpace."""
        if not self.owner:
            return None
        return self.get_sibling_component("SpaceVeilProducer")

    async def get_agent_context_via_compression_engine(self,
                                                 options: Optional[Dict[str, Any]] = None,
                                                 tools: Optional[List[Dict[str, Any]]] = None) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        INTERFACE UPDATED: Main entry point for agent loops to get turn-based context.

        NEW IMPLEMENTATION: Returns turn-based message format by splitting temporal stream at agent responses.

        Args:
            options: Rendering options (unchanged interface)
            tools: Enhanced tool definitions from VEIL (unchanged interface)

        Returns:
            Turn-based message list or multimodal dict with turn-based messages
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
            
            self._extract_current_focus_info(options) # NEW: Extract current focus at HUD level

            # Get VEILFacetCompressionEngine
            compression_engine = self.get_sibling_component("VEILFacetCompressionEngine")
            if not compression_engine:
                logger.warning(f"[{self.owner.id}] VEILFacetCompressionEngine not available, using direct rendering")
                return await self._process_facets_into_turns_direct(facet_cache, options, tools)

            # NEW: Pass extracted focus info to CompressionEngine
            processed_facets = await compression_engine.process_facet_cache_with_compression(
                facet_cache=facet_cache,
                focus_info=self._current_focus  # Pass structured focus info instead of focus_context
            )
            # NEW: Process facets into turn-based messages
            turn_based_messages = await self._process_facets_into_turns(processed_facets, options, tools)

            # Check for multimodal content if focus element provided
            focus_element_id = self._current_focus.get('focus_element_id') if self._current_focus else None
            if focus_element_id and self._has_multimodal_content(processed_facets, focus_element_id):
                return {
                    "messages": turn_based_messages,
                    "multimodal_content": {
                        "has_attachments": True,
                        "attachment_count": self._count_multimodal_content(processed_facets, focus_element_id),
                        "supported_types": ["image", "document"]
                    }
                }

            return turn_based_messages

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
        LEGACY: Native VEILFacet compression context rendering (eliminates flat cache conversions).

        This method is being phased out in favor of render_memorization_context_as_turns()
        which provides turn-based consistency with normal agent interaction.

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

    async def render_memorization_context_as_turns(self,
                                                  facet_cache: VEILFacetCache,
                                                  exclude_element_id: str,
                                                  focus_element_id: Optional[str] = None,
                                                  content_to_memorize: Optional[List] = None) -> List[Dict[str, Any]]:
        """
        NEW: Render memorization context using turn-based format like normal agent interaction.

        SIMPLIFIED: Just provides clean conversation turns. AgentMemoryCompressor handles memorization prompting.

        Args:
            facet_cache: Complete VEILFacetCache for context
            exclude_element_id: Element where memorization is happening
            focus_element_id: Focus element for context (defaults to exclude_element_id)
            content_to_memorize: Specific EventFacets being memorized (metadata only)

        Returns:
            List of clean conversation turns (no synthetic memorization request)
        """
        try:
            self._rendering_stats["memory_context_renders"] += 1

            # Use focus_element_id for focused treatment, default to exclude_element_id
            effective_focus_id = focus_element_id or exclude_element_id

            # Apply memorization exclusion rules directly to VEILFacetCache
            filtered_facets = self._apply_memorization_exclusions_native(
                facet_cache, exclude_element_id
            )

            # Prepare focus info for turn processing
            focus_info = {
                "focus_element_id": effective_focus_id,
                "focus_source": "memorization_context",
                "focus_context": {
                    "memorization_context": True,
                    "exclude_element_id": exclude_element_id
                }
            }

            # Process facets into turns using existing turn-based pipeline
            memorization_options = {
                "exclude_ambient": True,  # Skip ambient facets for memorization
                "focus_context": focus_info["focus_context"],
                "render_mode": "memorization",
                "include_time_markers": False,  # Cleaner for memorization
                "include_system_messages": True
            }

            # Use existing turn processing pipeline - just return clean conversation turns
            turn_messages = await self._process_facets_into_turns(
                filtered_facets, memorization_options, tools=None
            )

            # Add metadata about content being memorized (for AgentMemoryCompressor)
            if content_to_memorize and turn_messages:
                # Add metadata to the last turn or create metadata container
                content_summary = self._create_content_summary_for_memorization(content_to_memorize)
                metadata = {
                    "content_to_memorize_count": len(content_to_memorize),
                    "content_to_memorize_summary": content_summary,
                    "target_element_id": exclude_element_id
                }

                # Attach to context for AgentMemoryCompressor to use
                turn_messages.append({
                    "role": "system",
                    "content": "",  # Empty content
                    "turn_metadata": {
                        "turn_type": "memorization_metadata",
                        "is_internal": True,
                        "memorization_info": metadata
                    }
                })

            logger.info(f"Generated clean turn-based memorization context: {len(turn_messages)} turns")
            return turn_messages

        except Exception as e:
            logger.error(f"Error in turn-based memorization context: {e}", exc_info=True)
            # Fallback to empty context with error metadata
            return [{
                "role": "system",
                "content": "",
                "turn_metadata": {
                    "turn_type": "memorization_metadata",
                    "is_internal": True,
                    "error_fallback": True,
                    "error": str(e),
                    "target_element_id": exclude_element_id
                }
            }]

    def _create_content_summary_for_memorization(self, content_to_memorize: Optional[List]) -> str:
        """
        Create a summary of the content being memorized for the synthetic turn.

        UPDATED: Now renders the actual content being memorized in full, not previews/heuristics.

        Args:
            content_to_memorize: List of EventFacets being memorized

        Returns:
            Full rendered content of the facets being memorized
        """
        try:
            if not content_to_memorize:
                return "recent conversation content"

            # Render the actual content being memorized in full
            rendered_parts = []

            # Create minimal system_state and options for rendering
            minimal_system_state = {
                "active_channels": {},
                "agent_name": self._agent_name,
                "content_length_total": 0
            }

            minimal_options = RenderingOptions(
                exclude_ambient=True,
                render_mode="memorization",
                include_time_markers=False,
                include_system_messages=False,
                render_agent_messages=True
            )

            for facet in content_to_memorize:
                try:
                    # Reuse existing rendering logic
                    rendered_content = self._render_event_facet_with_system_context(
                        facet, minimal_system_state, minimal_options
                    )
                    if rendered_content:
                        rendered_parts.append(rendered_content)
                except Exception as e:
                    logger.warning(f"Error rendering facet {getattr(facet, 'facet_id', 'unknown')} for memorization: {e}")
                    # Fallback to basic content extraction
                    content = getattr(facet, 'content', '') or facet.get_property('content', '')
                    if content:
                        rendered_parts.append(content)

            if rendered_parts:
                return "\n".join(rendered_parts)
            else:
                return "recent conversation content"

        except Exception as e:
            logger.error(f"Error creating content summary: {e}", exc_info=True)
            return "recent conversation content"

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

                # FIXED: Include ALL EventFacets in context for full conversational awareness
                # The LLM needs to see the complete conversation flow to form good memories
                # Content duplication is acceptable for better memory formation quality

                # Include all other facets (EventFacets and StatusFacets)
                filtered_cache.add_facet(facet)

            logger.debug(f"Applied native memorization exclusions: {len(facet_cache)} → {len(filtered_cache)} facets")
            return filtered_cache

        except Exception as e:
            logger.error(f"Error applying native memorization exclusions: {e}", exc_info=True)
            return facet_cache  # Return unfiltered cache on error

    async def _render_temporal_facet_stream(self,
                                          facet_cache: VEILFacetCache,
                                          options: Dict[str, Any]) -> str:
        """
        Render temporal facet stream for memorization context.

        This method renders facets in chronological order, suitable for
        memorization contexts where temporal consistency is important.

        Args:
            facet_cache: VEILFacetCache to render
            options: Rendering options dictionary

        Returns:
            Rendered temporal stream as string
        """
        try:
            # Create RenderingOptions from dict
            render_opts = RenderingOptions(**{k: v for k, v in options.items() if hasattr(RenderingOptions, k)})

            # Get chronological temporal stream
            temporal_stream = facet_cache.get_chronological_stream(include_ambient=not render_opts.exclude_ambient)

            if not temporal_stream:
                return self._render_empty_context(render_opts)

            # Initialize system state for rendering
            system_state = self._initialize_system_state(temporal_stream, options)

            # Render each facet in temporal order
            rendered_parts = []

            for facet in temporal_stream:
                rendered_content = ""

                if facet.facet_type == VEILFacetType.EVENT:
                    rendered_content = self._render_event_facet_with_system_context(facet, system_state, render_opts)
                elif facet.facet_type == VEILFacetType.STATUS:
                    if render_opts.include_system_messages:
                        rendered_content = self._generate_system_message_for_status(facet)
                elif facet.facet_type == VEILFacetType.AMBIENT:
                    if not render_opts.exclude_ambient:
                        rendered_content = self._render_ambient_facet_individual(facet)

                if rendered_content:
                    rendered_parts.append(rendered_content)

            # Join with appropriate separators
            if render_opts.render_mode == "memorization":
                # Compact format for memorization
                return "\n".join(rendered_parts)
            else:
                # Regular format with more spacing
                return "\n\n".join(rendered_parts)

        except Exception as e:
            logger.error(f"Error rendering temporal facet stream: {e}", exc_info=True)
            return f"Error rendering temporal stream: {e}"

    # --- NEW: Turn-Based Message Processing ---

    async def _process_facets_into_turns(self,
                                       facet_cache: VEILFacetCache,
                                       options: Dict[str, Any],
                                       tools: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        ENHANCED: Clean three-phase approach for turn-based rendering with historical focus tracking.

        Phase 1: Collect turn structure with facets as objects (no rendering yet)
        Phase 2: Apply retroactive de-rendering and other processing logic
        Phase 3: Render turns to final content

        Args:
            facet_cache: Processed VEILFacetCache from compression engine
            options: Rendering options
            tools: Enhanced tool definitions

        Returns:
            List of turn-based message dictionaries
        """
        try:
            # Get chronological temporal stream
            temporal_stream = facet_cache.get_chronological_stream(
                include_ambient=True,
                only_synthetic_agent_responses_in_history=True
            )
            if not temporal_stream:
                return [{
                    "role": "user",
                    "content": "[AGENT WORKSPACE]\nNo content available",
                    "turn_metadata": {"turn_index": 0, "facet_count": 0}
                }]

            # NEW: Build historical focus map from StatusFacets
            historical_focus_map = self._build_historical_focus_map()

            # Initialize system state for processing
            system_state = self._initialize_system_state(temporal_stream, options)

            # PHASE 1: Collect clean turn structure with facets as objects
            turn_structure = self._collect_turn_structure_with_facets(temporal_stream)

            # PHASE 2: Apply processing logic (retroactive de-rendering, etc.)
            processed_turns = self._process_turn_structure(turn_structure, system_state, options, tools)

            # NEW: Add historical focus to each turn
            for turn_data in processed_turns:
                # Calculate turn timestamp (using first facet's timestamp as approximation)
                all_facets = turn_data["status"] + turn_data["ambient"] + turn_data["events"]
                turn_timestamp = min(f.veil_timestamp for f in all_facets) if all_facets else 0.0

                # Get historical focus for this turn
                historical_focus_id = self._get_historical_focus_for_turn(turn_timestamp, historical_focus_map)
                turn_data["historical_focus_id"] = historical_focus_id

            # PHASE 3: Render processed turns to final content
            turn_messages = await self._render_turn_structure_to_messages(processed_turns, system_state, options, tools)

            logger.debug(f"Processed {len(temporal_stream)} facets into {len(turn_messages)} turns using clean turn structure with historical focus")
            return turn_messages

        except Exception as e:
            logger.error(f"Error processing facets into turns: {e}", exc_info=True)
            return [{"role": "user", "content": f"Error: {e}", "turn_metadata": {"turn_index": 0, "facet_count": 0, "error": True}}]

    async def _process_facets_into_turns_direct(self,
                                              facet_cache: VEILFacetCache,
                                              options: Dict[str, Any],
                                              tools: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Direct turn processing without compression engine (fallback).
        """
        try:
            return await self._process_facets_into_turns(facet_cache, options, tools)
        except Exception as e:
            logger.error(f"Error in direct turn processing: {e}", exc_info=True)
            return [{"role": "user", "content": "Error: Direct turn processing failed", "turn_metadata": {"turn_index": 0, "facet_count": 0, "error": True}}]

    def _collect_turn_structure_with_facets(self, temporal_stream: List[VEILFacet]) -> List[Dict[str, Any]]:
        """
        PHASE 1: Collect clean turn structure with facets as objects.

        Creates the clean structure suggested:
        [
          {"type": "user_turn", "status": [StatusFacet_1], "ambient": [AmbientFacet_1], "events": [EventFacet_1, EventFacet_2]},
          {"type": "agent_turn", "status": [], "ambient": [], "events": [EventFacet_3]},
          ...
        ]

        ENHANCED: Handles consecutive agent_response events by grouping them into a single agent turn.

        Args:
            temporal_stream: Chronologically ordered list of VEILFacet instances

        Returns:
            List of turn dictionaries with facets as objects
        """
        if not temporal_stream:
            return []

        turn_structure = []
        current_turn = {
            "type": "user_turn",
            "status": [],
            "ambient": [],
            "events": [],
            "turn_index": 0
        }

        for facet in temporal_stream:
            # Check if this facet triggers agent turn logic
            is_agent_response = (
                facet.facet_type == VEILFacetType.EVENT and
                facet.get_property("event_type") == "agent_response"
            )

            if is_agent_response:
                # End current user turn if it has content
                if current_turn["status"] or current_turn["ambient"] or current_turn["events"]:
                    turn_structure.append(current_turn)

                # Start new agent turn for this agent response
                current_turn = {
                    "type": "agent_turn",
                    "status": [],
                    "ambient": [],
                    "events": [facet],  # Add the agent response event
                    "turn_index": len(turn_structure)
                }
                logger.debug(f"Started new agent turn {current_turn['turn_index']} for agent_response")
            else:
                # Non-agent-response facet
                if current_turn["type"] == "agent_turn":
                    # Switch from agent turn to user turn

                    # End current agent turn
                    turn_structure.append(current_turn)

                    # Start fresh user turn for this content
                    current_turn = {
                        "type": "user_turn",
                        "status": [],
                        "ambient": [],
                        "events": [],
                        "turn_index": len(turn_structure)
                    }
                    logger.debug(f"Started new user turn {current_turn['turn_index']} after agent turn")

                # Add facet to current turn based on type
                if facet.facet_type == VEILFacetType.STATUS:
                    current_turn["status"].append(facet)
                elif facet.facet_type == VEILFacetType.AMBIENT:
                    current_turn["ambient"].append(facet)
                elif facet.facet_type == VEILFacetType.EVENT:
                    current_turn["events"].append(facet)

        # Add final turn if it has content
        if current_turn["status"] or current_turn["ambient"] or current_turn["events"]:
            turn_structure.append(current_turn)

        logger.debug(f"Collected turn structure: {len(turn_structure)} turns from {len(temporal_stream)} facets")
        return turn_structure

    def _process_turn_structure(self,
                              turn_structure: List[Dict[str, Any]],
                              system_state: Dict[str, Any],
                              options: Dict[str, Any],
                              tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        PHASE 2: Apply processing logic to turn structure.

        NEW APPROACH: Additive ambient placement instead of filtering.
        - Collect all available ambient facets
        - Add ambient facets to turns that have triggers (status changes, thresholds)
        - Never filter out ambient facets (they can only be moved, not lost)

        Args:
            turn_structure: List of turn dictionaries with facets
            system_state: Current system state tracking
            options: Rendering options
            tools: Enhanced tool definitions

        Returns:
            Processed turn structure ready for rendering
        """
        try:
            processed_turns = []

            # PHASE 2: Initialize scene-context collection for historical replay
            scene_context = {
                "scene_status_messages": [],  # Store scene-related status messages
                "scene_status_by_type": {},   # Group by status type for deduplication
                "last_scene_change_turn": -1  # Track when last scene change occurred
            }

            # Collect all ambient facets from all turns for triggered placement
            all_ambient_facets = set()
            for turn_data in turn_structure:
                for ambient_facet in turn_data["ambient"]:
                    all_ambient_facets.add(ambient_facet.facet_id)

            all_ambient_facets_dict = {}
            for turn_data in turn_structure:
                for ambient_facet in turn_data["ambient"]:
                    all_ambient_facets_dict[ambient_facet.facet_id] = ambient_facet

            # Process each turn to update system state and apply logic
            for turn_data in turn_structure:
                processed_turn = dict(turn_data)  # Copy turn data

                # FIXED: Create turn-specific system state to avoid retroactive changes
                turn_specific_system_state = copy.deepcopy(system_state)  # Deep copy to avoid shared references

                # Update turn-specific state based on status facets in this turn
                for status_facet in turn_data["status"]:
                    self._update_system_state(turn_specific_system_state, status_facet, options)
                    self._update_status_message_tracking(turn_specific_system_state, status_facet)

                # PHASE 2: Collect scene-related status messages for historical replay
                scene_changing_status_facets = [
                    facet for facet in turn_data["status"]
                    if self._is_scene_changing_status(facet)
                ]

                if scene_changing_status_facets:
                    # This turn has scene changes - collect all scene-related status messages
                    for status_facet in scene_changing_status_facets:
                        self._collect_scene_status_message(status_facet, scene_context, turn_data["turn_index"])

                    # Mark this as a scene change turn
                    scene_context["last_scene_change_turn"] = turn_data["turn_index"]

                # Store turn-specific state for rendering (don't modify global state)
                turn_data["system_state"] = turn_specific_system_state

                # Update global state for next turn's baseline
                for status_facet in turn_data["status"]:
                    self._update_system_state(system_state, status_facet, options)
                    self._update_status_message_tracking(system_state, status_facet)

                # NEW APPROACH: Additive ambient placement based on triggers
                triggered_ambient_facets = self._get_triggered_ambient_facets_for_turn(
                    turn_data,
                    all_ambient_facets_dict,
                    turn_specific_system_state,
                    options
                )

                # Combine natural ambient facets with triggered ambient facets
                combined_ambient = list(turn_data["ambient"])  # Start with natural placement
                for triggered_facet in triggered_ambient_facets:
                    if triggered_facet not in combined_ambient:
                        combined_ambient.append(triggered_facet)

                processed_turn["ambient"] = combined_ambient

                # PHASE 2: Attach scene context for historical replay
                processed_turn["scene_context"] = scene_context

                processed_turns.append(processed_turn)

            # Apply retroactive ambient de-rendering (backwards pass) - keep latest only
            self._apply_simple_retroactive_deduplication(processed_turns)

            logger.debug(f"Processed {len(processed_turns)} turns with new additive ambient approach")
            return processed_turns

        except Exception as e:
            logger.error(f"Error processing turn structure: {e}", exc_info=True)
            return turn_structure  # Return unprocessed on error

    def _collect_scene_status_message(self, status_facet: VEILFacet, scene_context: Dict[str, Any], turn_index: int) -> None:
        """
        PHASE 2: Collect scene-related status message for historical replay.

        Args:
            status_facet: Scene-changing status facet to collect
            scene_context: Scene context collection dictionary
            turn_index: Current turn index
        """
        try:
            status_type = status_facet.get_property("status_type")
            current_state = status_facet.get_property("current_state", {})

            # Create scene status entry
            scene_status_entry = {
                "facet": status_facet,
                "status_type": status_type,
                "current_state": current_state,
                "collected_at_turn": turn_index,
                "timestamp": status_facet.veil_timestamp
            }

            # Add to scene status messages list
            scene_context["scene_status_messages"].append(scene_status_entry)

            # Group by type for deduplication (keep latest of each type)
            if status_type not in scene_context["scene_status_by_type"]:
                scene_context["scene_status_by_type"][status_type] = []
            scene_context["scene_status_by_type"][status_type].append(scene_status_entry)

            logger.debug(f"Collected scene status message: {status_type} at turn {turn_index}")

        except Exception as e:
            logger.error(f"Error collecting scene status message: {e}", exc_info=True)

    async def _render_turn_structure_to_messages(self,
                                                 processed_turns: List[Dict[str, Any]],
                                                 system_state: Dict[str, Any],
                                                 options: Dict[str, Any],
                                                 tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        PHASE 3: Render processed turn structure to final messages.

        Args:
            processed_turns: Processed turn structure with facets
            system_state: Current system state tracking
            options: Rendering options
            tools: Enhanced tool definitions

        Returns:
            List of final turn message dictionaries
        """
        try:
            turn_messages = []

            for turn_data in processed_turns:
                # Render each section: Status → Ambient → Events
                content_sections = []

                # REFINED: Render full status section when there are status changes, minimal when not
                # Get turn-specific system state for historically accurate rendering
                turn_system_state = turn_data.get("system_state", system_state)

                if turn_data["status"] or turn_data["turn_index"] == 0:
                    # Full status section for status changes or first turn
                    status_content = await self._render_status_section_from_facets(
                        turn_data["status"],
                        turn_data["turn_index"],
                        system_state,       # Global state (for focus fallback)
                        turn_system_state,  # Turn-specific historical state
                        tools,
                        bool(turn_data["status"]),
                        turn_data.get("scene_context")  # PHASE 4: Pass scene context for historical replay
                    )
                    if status_content:
                        content_sections.append(status_content)
                else:
                    # No status changes this turn - render each status message from this turn (if any)
                    if turn_data["status"] and turn_data["type"] != "agent_turn":
                        for status_facet in turn_data["status"]:
                            system_message = self._generate_system_message_for_status(status_facet)
                            if system_message:
                                content_sections.append(system_message)

                # Render Ambient section
                if turn_data["ambient"]:
                    ambient_content = await self._render_ambient_section_from_facets(
                        turn_data["ambient"],
                        turn_system_state,  # Use turn-specific state
                        tools
                    )
                    if ambient_content:
                        content_sections.append(ambient_content)

                # Render Events section
                if turn_data["events"]:
                    events_content = await self._render_events_section_from_facets(
                        turn_data["events"],
                        system_state,
                        options
                    )
                    if events_content:
                        content_sections.append(events_content)

                # Build final turn message
                turn_content = "\n\n".join(filter(None, content_sections))

                # Calculate turn metadata
                all_facets = turn_data["status"] + turn_data["ambient"] + turn_data["events"]
                turn_metadata = {
                    "turn_index": turn_data["turn_index"],
                    "facet_count": len(all_facets),
                    "has_status_changes": bool(turn_data["status"]),
                    "has_agent_response": turn_data["type"] == "agent_turn",
                    "timestamp_range": [
                        min(f.veil_timestamp for f in all_facets),
                        max(f.veil_timestamp for f in all_facets)
                    ] if all_facets else [0.0, 0.0]
                }

                # Determine role
                role = "assistant" if turn_data["type"] == "agent_turn" else "user"

                turn_message = {
                    "role": role,
                    "content": turn_content,
                    "turn_metadata": turn_metadata
                }

                turn_messages.append(turn_message)

            return turn_messages

        except Exception as e:
            logger.error(f"Error rendering turn structure to messages: {e}", exc_info=True)
            return []

    def _update_system_state_for_turn(self, turn_container: Dict[str, Any], system_state: Dict[str, Any]) -> None:
        """
        Update system state after processing a turn container.

        Args:
            turn_container: The turn container that was processed
            system_state: System state to update
        """
        try:
            # Process status facets to update system state
            for status_facet in turn_container.get("status_facets", []):
                self._update_system_state(system_state, status_facet, {})
                self._update_status_message_tracking(system_state, status_facet)

            # Update content length tracking
            events_content_length = len(turn_container.get("events_section", ""))
            self._update_content_length_tracking(system_state, events_content_length)

        except Exception as e:
            logger.error(f"Error updating system state for turn {turn_container.get('turn_index', 'unknown')}: {e}", exc_info=True)

    def _apply_retroactive_ambient_derendering(self, turn_containers: List[Dict[str, Any]]) -> None:
        """
        PASS 2: Apply retroactive ambient de-rendering by going backwards and removing
        ambient content that appeared in later turns.

        RULE: Each ambient message/group should appear only ONCE in the entire context.
        When rendered in a later turn, remove it from all earlier turns.

        Args:
            turn_containers: List of turn containers to process
        """
        try:
            rendered_ambient_ids = set()

            # Go backwards through turns (from latest to earliest)
            for turn_container in reversed(turn_containers):
                current_ambient_content = turn_container.get("ambient_section", "")

                if current_ambient_content:
                    # Track which ambient facets are rendered in this turn
                    ambient_facets = turn_container.get("ambient_facets", [])
                    current_turn_ambient_ids = {f.facet_id for f in ambient_facets}

                    # If any of these ambient facets were already rendered in later turns, remove them
                    conflicting_ids = current_turn_ambient_ids.intersection(rendered_ambient_ids)
                    if conflicting_ids:
                        logger.debug(f"Retroactive de-rendering: Removing {len(conflicting_ids)} ambient facets from turn {turn_container['turn_index']} "
                                   f"(already rendered in later turns): {conflicting_ids}")

                        # Remove conflicting ambient content
                        turn_container["ambient_section"] = self._remove_conflicting_ambient_content(
                            current_ambient_content, conflicting_ids, ambient_facets
                        )

                    # Add current turn's ambient IDs to the rendered set
                    rendered_ambient_ids.update(current_turn_ambient_ids)

            logger.debug(f"Retroactive ambient de-rendering complete: {len(rendered_ambient_ids)} unique ambient facets rendered")

        except Exception as e:
            logger.error(f"Error in retroactive ambient de-rendering: {e}", exc_info=True)

    def _remove_conflicting_ambient_content(self,
                                          ambient_content: str,
                                          conflicting_ids: set,
                                          ambient_facets: List[VEILFacet]) -> str:
        """
        Remove ambient content that conflicts with later turns.

        For now, this is a simple implementation that removes all ambient content if any conflicts.
        Could be enhanced to remove only specific conflicting sections.

        Args:
            ambient_content: Original ambient content string
            conflicting_ids: Set of ambient facet IDs that conflict
            ambient_facets: List of ambient facets in this turn

        Returns:
            Modified ambient content with conflicts removed
        """
        try:
            # Simple approach: if any ambient facets conflict, remove entire ambient section
            # This ensures retroactive de-rendering while being safe

            non_conflicting_facets = [f for f in ambient_facets if f.facet_id not in conflicting_ids]

            if not non_conflicting_facets:
                # All ambient content conflicts - remove everything
                return ""

            # Some ambient content is safe - would need more sophisticated parsing to preserve only non-conflicting parts
            # For now, use conservative approach and remove all if any conflicts
            logger.debug(f"Removing entire ambient section due to conflicts (conservative approach)")
            return ""

        except Exception as e:
            logger.error(f"Error removing conflicting ambient content: {e}", exc_info=True)
            return ambient_content  # Return original on error

    def _render_turn_container_to_content(self, turn_container: Dict[str, Any]) -> str:
        """
        Convert a turn container to final rendered content string.

        Args:
            turn_container: Turn container with rendered sections

        Returns:
            Final turn content string
        """
        try:
            content_sections = []

            # Add sections in proper order: Status → Ambient → Events
            status_section = turn_container.get("status_section", "")
            if status_section:
                content_sections.append(status_section)

            ambient_section = turn_container.get("ambient_section", "")
            if ambient_section:
                content_sections.append(ambient_section)

            events_section = turn_container.get("events_section", "")
            if events_section:
                content_sections.append(events_section)

            return "\n\n".join(filter(None, content_sections))

        except Exception as e:
            logger.error(f"Error rendering turn container to content: {e}", exc_info=True)
            return f"Error rendering turn {turn_container.get('turn_index', 'unknown')}: {e}"

    async def _render_status_section_from_facets(self,
                                               status_facets: List[VEILFacet],
                                               turn_index: int,
                                               system_state: Dict[str, Any],
                                               turn_system_state: Dict[str, Any],
                                               tools: Optional[List[Dict[str, Any]]],
                                               has_status_changes: bool,
                                               scene_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Render Status section from status facets with historical accuracy.

        Args:
            status_facets: List of status facets to render
            turn_index: Current turn index
            system_state: Global system state (for focus fallback)
            turn_system_state: Turn-specific historical system state
            tools: Enhanced tool definitions
            has_status_changes: Whether this turn has status changes
            scene_context: Scene context for historical replay (PHASE 4)

        Returns:
            Rendered status section string
        """
        try:
            status_parts = []

            # SIMPLIFIED: Expand status list with previous scene messages if this turn has scene changes
            expanded_status_facets = list(status_facets)  # Start with current turn's status messages

            # If this turn has scene changes, add all previous scene-related status messages
            has_scene_changes = any(self._is_scene_changing_status(facet) for facet in status_facets)
            if has_scene_changes and scene_context:
                previous_scene_messages = [
                    msg["facet"] for msg in scene_context.get("scene_status_messages", [])
                    if msg["collected_at_turn"] < turn_index
                ]
                expanded_status_facets = previous_scene_messages + expanded_status_facets
            # Extract specific facet types for consolidated rendering
            workspace_facets = [facet for facet in expanded_status_facets if facet.get_property("status_type") == "space_created"]
            chat_facets = [facet for facet in expanded_status_facets if facet.get_property("status_type") in ["container_created", "chat_joined"]]

            # Render consolidated sections based on actual facets collected
            if workspace_facets:
                workspace_section = self._render_agent_workspace_wrapper_from_facets(workspace_facets)
                if workspace_section:
                    status_parts.append(workspace_section)

            if chat_facets:
                chat_info_section = self._render_consolidated_chat_info_from_facets(chat_facets, tools)
                if chat_info_section:
                    status_parts.append(chat_info_section)

            # Render remaining individual status messages (excluding those already rendered in consolidated sections)
            already_rendered_facets = set(workspace_facets + chat_facets)

            for status_facet in expanded_status_facets:
                if status_facet not in already_rendered_facets:
                    system_message = self._generate_system_message_for_status(status_facet)
                    if system_message:
                        status_parts.append(system_message)

            return "\n".join(status_parts) if status_parts else ""

        except Exception as e:
            logger.error(f"Error rendering status section from facets: {e}", exc_info=True)
            return ""

    def _render_agent_workspace_wrapper_from_facets(self, workspace_facets: List[VEILFacet]) -> str:
        """Render opening agent workspace wrapper from space_created facets."""
        if not workspace_facets:
            return ""

        # Use the latest workspace facet (should only be one, but just in case)
        workspace_facet = workspace_facets[-1]
        current_state = workspace_facet.get_property("current_state", {})

        agent_name = current_state.get("agent_name", "Unknown Agent")
        agent_description = current_state.get("agent_description", "Agent workspace")

        if agent_name and agent_name != "Unknown Agent":
            opening_tag = f'<inner_space agent_name="{agent_name}"'
            if agent_description and agent_description != "Agent workspace":
                opening_tag += f' agent_description="{agent_description}"'
            opening_tag += '>'
        else:
            opening_tag = '<inner_space>'

        return opening_tag

    def _render_consolidated_chat_info_from_facets(self, chat_facets: List[VEILFacet], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Render consolidated chat_info sections from container_created/chat_joined facets."""
        if not chat_facets:
            return ""

        # Group facets by adapter
        adapter_groups = {}

        for facet in chat_facets:
            current_state = facet.get_property("current_state", {})
            adapter_type = current_state.get("adapter_type")
            server_name = current_state.get("server_name")

            if adapter_type and server_name:
                group_key = f"{adapter_type}_{server_name}"
                if group_key not in adapter_groups:
                    adapter_groups[group_key] = {
                        "adapter_type": adapter_type,
                        "server_name": server_name,
                        "alias": current_state.get("alias", ""),
                        "channels": []
                    }

                # Avoid duplicates - check if this conversation already exists
                conversation_name = current_state.get("conversation_name")
                existing_channels = adapter_groups[group_key]["channels"]
                channel_exists = any(ch.get("conversation_name") == conversation_name for ch in existing_channels)

                if not channel_exists:
                    channel_info = {
                        "conversation_name": conversation_name,
                        "available_tools": current_state.get("available_tools", [])
                    }
                    adapter_groups[group_key]["channels"].append(channel_info)

        # Render each adapter group
        info_sections = []
        for adapter_group in adapter_groups.values():
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
                available_tools = channel.get("available_tools", [])

                # Build tool list for this channel
                if available_tools:
                    tool_names = [tool.get("name", tool.get("tool_name", "unknown")) for tool in available_tools]
                    tools_attr = f' tools="({", ".join(tool_names)})"'
                else:
                    tools_attr = ''

                # Build channel line (no focused attribute - handled separately)
                channel_line = f'<channel conversation_name="{conversation_name}"{tools_attr}>'
                channel_lines.append(channel_line)

            info_section += "\n" + "\n".join(channel_lines) + "\n</chat_info>"
            info_sections.append(info_section)

        return "\n\n".join(info_sections)

    async def _render_ambient_section_from_facets(self,
                                                ambient_facets: List[VEILFacet],
                                                system_state: Dict[str, Any],
                                                tools: Optional[List[Dict[str, Any]]]) -> str:
        """
        Render Ambient section from ambient facets.

        Args:
            ambient_facets: List of ambient facets to render
            system_state: Current system state
            tools: Enhanced tool definitions

        Returns:
            Rendered ambient section string
        """
        try:
            if not ambient_facets:
                return ""

            # Group and render ambient facets
            ambient_sections = []

            # Separate tool-type ambient facets for family grouping
            tool_ambient_facets = [f for f in ambient_facets if f.get_property("ambient_type") in ["tool_instructions", "messaging_tools", "terminal_tools", "file_tools", "scratchpad_tools"]]
            other_ambient_facets = [f for f in ambient_facets if f not in tool_ambient_facets]

            # Render tool ambient facets with consolidation
            if tool_ambient_facets:
                tool_families = self._group_ambient_facets_by_family(tool_ambient_facets)

                tool_sections = []
                for family, facets in tool_families.items():
                    if len(facets) == 1:
                        consolidated = self._render_single_family_tools(family, facets)
                        if consolidated:
                            tool_sections.append(consolidated)
                    else:
                        consolidated = self._render_tools_consolidated_by_family(family, facets)
                        if consolidated:
                            tool_sections.append(consolidated)

                if tool_sections:
                    # Add XML format instructions at the beginning
                    xml_instructions = self._generate_xml_tool_call_format_instructions()
                    tool_content = xml_instructions + "\n\n" + "\n\n".join(tool_sections)
                    ambient_sections.append("<tool_instructions>\n" + tool_content + "\n</tool_instructions>")

            # Render other ambient facets individually
            for ambient_facet in other_ambient_facets:
                ambient_content = self._render_ambient_facet_individual(ambient_facet)
                if ambient_content:
                    ambient_sections.append(ambient_content)

            return "\n\n".join(ambient_sections)

        except Exception as e:
            logger.error(f"Error rendering ambient section from facets: {e}", exc_info=True)
            return ""

    async def _render_events_section_from_facets(self,
                                               event_facets: List[VEILFacet],
                                               system_state: Dict[str, Any],
                                               options: Dict[str, Any]) -> str:
        """
        Render Events section from event facets.

        Args:
            event_facets: List of event facets to render
            system_state: Current system state
            options: Rendering options

        Returns:
            Rendered events section string
        """
        try:
            if not event_facets:
                return ""

            render_opts = RenderingOptions(**{k: v for k, v in options.items() if hasattr(RenderingOptions, k)})
            event_parts = []

            for event_facet in event_facets:
                event_content = self._render_event_facet_with_system_context(event_facet, system_state, render_opts)
                if event_content:
                    event_parts.append(event_content)

                # Update content length tracking for ambient threshold logic
                if event_content:
                    self._update_content_length_tracking(system_state, len(event_content))

            return "\n".join(event_parts) if event_parts else ""

        except Exception as e:
            logger.error(f"Error rendering events section from facets: {e}", exc_info=True)
            return ""

    def _render_ambient_tools_section(self, facet_cache: VEILFacetCache, system_state: Dict[str, Any]) -> str:
        """
        NEW: Render ambient tools section with intelligent consolidation by tool family.

        Processes structured ambient facets and groups them by tool family for
        consolidated rendering with target parameter injection.
        """
        ambient_facets = facet_cache.get_ambient_facets()

        if not ambient_facets:
            return ""

        # Group ambient facets by tool family
        tool_families = self._group_ambient_facets_by_family(ambient_facets)

        # Render each family with appropriate consolidation
        sections = []
        for family, facets in tool_families.items():
            if len(facets) == 1:
                # Single element - render normally but check for structured data
                consolidated = self._render_single_family_tools(family, facets)
                if consolidated:
                    sections.append(consolidated)
            else:
                # Multiple elements - consolidate by family type
                consolidated = self._render_tools_consolidated_by_family(family, facets)
                if consolidated:
                    sections.append(consolidated)

        if sections:
            # Add XML format instructions at the beginning
            xml_instructions = self._generate_xml_tool_call_format_instructions()
            tool_content = xml_instructions + "\n\n" + "\n\n".join(sections)
            return "<tool_instructions>\n" + tool_content + "\n</tool_instructions>"

        return ""

    def _group_ambient_facets_by_family(self, ambient_facets: List[VEILFacet]) -> Dict[str, List[VEILFacet]]:
        """Group ambient facets by their tool family type."""
        families = {}
        for facet in ambient_facets:
            # Check if this is a structured ambient facet
            if facet.get_property("data_format") == "structured":
                content = facet.get_property("content", {})
                if isinstance(content, dict):
                    family = content.get("tool_family", facet.get_property("ambient_type", "unknown_tools"))
                else:
                    family = facet.get_property("ambient_type", "unknown_tools")
            else:
                # Fallback for non-structured facets
                family = facet.get_property("ambient_type", "unknown_tools")

            if family not in families:
                families[family] = []
            families[family].append(facet)

        return families

    def _build_historical_focus_map(self) -> Dict[float, str]:
        """NEW: Build map of timestamp -> focused_element_id from stored history of focus states"""

        focus_map = {}

        for focus_state in self._focus_change_history:
            timestamp = focus_state["veil_timestamp"]
            focused_element_id = focus_state["current_state"]["focused_element_id"]

            if focused_element_id:
                focus_map[timestamp] = focused_element_id

        logger.debug(f"Built historical focus map with {len(focus_map)} focus changes")
        return focus_map

    def _get_historical_focus_for_turn(self, turn_timestamp: float, focus_map: Dict[float, str]) -> Optional[str]:
        """NEW: Get what was focused at a specific timestamp"""

        # Find the latest focus change before or at this turn's timestamp
        relevant_focus_changes = [
            (ts, element_id) for ts, element_id in focus_map.items()
            if ts <= turn_timestamp
        ]

        if relevant_focus_changes:
            # Return the most recent focus change
            latest_change = max(relevant_focus_changes, key=lambda x: x[0])
            return latest_change[1]

        return None

    def _is_minor_status_change(self, status_facet: StatusFacet) -> bool:
        """Determine if this is a minor status change that shouldn't trigger ambient facets"""
        status_type = status_facet.get_property("status_type")
        return status_type in MINOR_STATUS_TYPES

    def _is_scene_changing_status(self, status_facet: StatusFacet) -> bool:
        """Determine if this is a major scene change that should trigger ambient facets"""
        return not self._is_minor_status_change(status_facet)

    def _render_single_family_tools(self, family: str, facets: List[VEILFacet]) -> str:
        """Render tools for a single element (no consolidation needed)."""
        try:
            facet = facets[0]

            # Check if it's structured data
            if facet.get_property("data_format") == "structured":
                content = facet.get_property("content", {})
                if isinstance(content, dict):
                    tools = content.get("tools", [])
                    element_context = content.get("element_context", {})

                    return self._render_structured_tools_for_element(tools, element_context, family)

            # Fallback for string content
            return facet.get_property("content", "")

        except Exception as e:
            logger.error(f"Error rendering single family tools: {e}", exc_info=True)
            return ""

    def _render_tools_consolidated_by_family(self, family: str, facets: List[VEILFacet]) -> str:
        """Render consolidated tools for a specific family."""
        try:
            if family == "messaging_tools":
                return self._render_messaging_tools_consolidated(facets)
            elif family == "terminal_tools":
                return self._render_terminal_tools_consolidated(facets)
            elif family == "file_tools":
                return self._render_file_tools_consolidated(facets)
            elif family == "scratchpad_tools":
                return self._render_scratchpad_tools_consolidated(facets)
            else:
                return self._render_generic_tools_consolidated(family, facets)

        except Exception as e:
            logger.error(f"Error rendering consolidated tools for family {family}: {e}", exc_info=True)
            return ""

    def _render_messaging_tools_consolidated(self, facets: List[VEILFacet]) -> str:
        """Render messaging tools with conversation target selection."""
        try:
            conversations = []
            all_tools = {}

            for facet in facets:
                content = facet.get_property("content", {})
                if isinstance(content, dict):
                    tools = content.get("tools", [])
                    element_context = content.get("element_context", {})

                    # Collect conversation info
                    conv_name = element_context.get("conversation_name") or element_context.get("element_name", "Unknown")
                    conversations.append({
                        "name": conv_name,
                        "element_id": element_context.get("element_id"),
                        "adapter_type": element_context.get("adapter_type"),
                        "server_name": element_context.get("server_name")
                    })

                    # Collect unique tools
                    for tool in tools:
                        tool_name = tool.get("name", "unknown")
                        if tool_name not in all_tools:
                            all_tools[tool_name] = tool

            # Build conversation list for target parameter
            conv_list = ", ".join([f'"{c["name"]}"' for c in conversations])

            # FIXED: Render tools with XML examples instead of plain descriptions
            tool_lines = []
            for tool_name, tool_def in all_tools.items():
                description = tool_def.get("description", "No description")

                # Create XML example for messaging tools
                if tool_name in ["send_message", "send_msg", "reply"]:
                    example_target = conversations[0]["name"] if conversations else "target"
                    xml_example = f'<{tool_name} text="message text" source="{example_target}">'
                    tool_lines.append(f"{xml_example} - {description}")
                elif tool_name in ["msg"]:
                    example_target = conversations[0]["name"] if conversations else "target"
                    xml_example = f'<{tool_name} source="{example_target}">message text</{tool_name}>'
                    tool_lines.append(f"{xml_example} - {description}")
                else:
                    tool_lines.append(f"<{tool_name} ...> - {description}")

            return f"Messaging Tools (targets: {conv_list})\n" + "\n".join(tool_lines)

        except Exception as e:
            logger.error(f"Error rendering messaging tools: {e}", exc_info=True)
            return ""

    def _render_terminal_tools_consolidated(self, facets: List[VEILFacet]) -> str:
        """Render terminal tools with terminal target selection."""
        try:
            terminals = []
            all_tools = {}

            for facet in facets:
                content = facet.get_property("content", {})
                if isinstance(content, dict):
                    tools = content.get("tools", [])
                    element_context = content.get("element_context", {})

                    # Collect terminal info
                    term_name = element_context.get("element_name", "Unknown Terminal")
                    terminals.append({
                        "name": term_name,
                        "element_id": element_context.get("element_id")
                    })

                    # Collect unique tools
                    for tool in tools:
                        tool_name = tool.get("name", "unknown")
                        if tool_name not in all_tools:
                            all_tools[tool_name] = tool

            term_list = ", ".join([f'"{t["name"]}"' for t in terminals])

            # FIXED: Render tools with XML examples instead of plain descriptions
            tool_lines = []
            for tool_name, tool_def in all_tools.items():
                description = tool_def.get("description", "No description")

                # Create XML example for terminal tools
                if tool_name in ["execute_command", "change_directory", "kill_process"]:
                    example_target = terminals[0]["name"] if terminals else "Terminal"
                    if tool_name == "execute_command":
                        xml_example = f'<{tool_name} command="ls -la" source="{example_target}">'
                    elif tool_name == "change_directory":
                        xml_example = f'<{tool_name} path="/home/user" source="{example_target}">'
                    else:
                        xml_example = f'<{tool_name} process_id="1234" source="{example_target}">'
                    tool_lines.append(f"{xml_example} - {description}")
                else:
                    tool_lines.append(f"<{tool_name} ...> - {description}")

            return f"Terminal Tools (targets: {term_list})\n" + "\n".join(tool_lines)

        except Exception as e:
            logger.error(f"Error rendering terminal tools: {e}", exc_info=True)
            return ""

    def _render_file_tools_consolidated(self, facets: List[VEILFacet]) -> str:
        """Render file tools consolidated."""
        try:
            all_tools = {}

            for facet in facets:
                content = facet.get_property("content", {})
                if isinstance(content, dict):
                    tools = content.get("tools", [])

                    # Collect unique tools
                    for tool in tools:
                        tool_name = tool.get("name", "unknown")
                        if tool_name not in all_tools:
                            all_tools[tool_name] = tool

            # FIXED: Render tools with XML examples instead of plain descriptions
            tool_lines = []
            for tool_name, tool_def in all_tools.items():
                description = tool_def.get("description", "No description")

                # Create XML examples for common file tools
                if tool_name == "read_file":
                    xml_example = f'<{tool_name} file_path="path/to/file.txt">'
                elif tool_name == "write_file":
                    xml_example = f'<{tool_name} file_path="path/to/file.txt" content="file content">'
                elif tool_name == "list_directory":
                    xml_example = f'<{tool_name} directory_path="path/to/directory">'
                else:
                    xml_example = f'<{tool_name} ...>'

                tool_lines.append(f"{xml_example} - {description}")

            return f"File Tools\n" + "\n".join(tool_lines)

        except Exception as e:
            logger.error(f"Error rendering file tools: {e}", exc_info=True)
            return ""

    def _render_scratchpad_tools_consolidated(self, facets: List[VEILFacet]) -> str:
        """Render scratchpad tools consolidated."""
        try:
            scratchpads = []
            all_tools = {}

            for facet in facets:
                content = facet.get_property("content", {})
                if isinstance(content, dict):
                    tools = content.get("tools", [])
                    element_context = content.get("element_context", {})

                    # Collect scratchpad info
                    scratchpad_name = element_context.get("element_name", "Unknown Scratchpad")
                    scratchpads.append({
                        "name": scratchpad_name,
                        "element_id": element_context.get("element_id")
                    })

                    # Collect unique tools
                    for tool in tools:
                        tool_name = tool.get("name", "unknown")
                        if tool_name not in all_tools:
                            all_tools[tool_name] = tool

            scratchpad_list = ", ".join([f'"{s["name"]}"' for s in scratchpads])

            # FIXED: Render tools with XML examples instead of plain descriptions
            tool_lines = []
            for tool_name, tool_def in all_tools.items():
                description = tool_def.get("description", "No description")

                # Create XML examples for scratchpad tools
                if tool_name == "add_note_to_scratchpad":
                    xml_example = f'<{tool_name} note_text="Note content">'
                elif tool_name == "get_notes_from_scratchpad":
                    xml_example = f'<{tool_name}>'
                elif tool_name == "clear_all_scratchpad_notes":
                    xml_example = f'<{tool_name}>'
                else:
                    xml_example = f'<{tool_name} ...>'

                tool_lines.append(f"{xml_example} - {description}")

            return f"Scratchpad Tools (targets: {scratchpad_list})\n" + "\n".join(tool_lines)

        except Exception as e:
            logger.error(f"Error rendering scratchpad tools: {e}", exc_info=True)
            return ""

    def _render_generic_tools_consolidated(self, tool_family: str, facets: List[VEILFacet]) -> str:
        """Render generic tools consolidated."""
        try:
            all_tools = {}

            for facet in facets:
                content = facet.get_property("content", {})
                if isinstance(content, dict):
                    tools = content.get("tools", [])

                    # Collect unique tools
                    for tool in tools:
                        tool_name = tool.get("name", "unknown")
                        if tool_name not in all_tools:
                            all_tools[tool_name] = tool

            # FIXED: Render tools with XML examples instead of plain descriptions
            tool_lines = []
            for tool_name, tool_def in all_tools.items():
                description = tool_def.get("description", "No description")
                xml_example = f'<{tool_name} ...>'
                tool_lines.append(f"{xml_example} - {description}")

            return f"{tool_family.replace('_', ' ').title()}\n" + "\n".join(tool_lines)

        except Exception as e:
            logger.error(f"Error rendering generic tools: {e}", exc_info=True)
            return ""

    def _render_structured_tools_for_element(self, tools: List[Dict[str, Any]], element_context: Dict[str, Any], family: str) -> str:
        """Render structured tools for a single element."""
        try:
            element_name = element_context.get("conversation_name") or element_context.get("element_name", "Unknown")

            tool_lines = []
            for tool in tools:
                tool_name = tool.get("name", "unknown")
                description = tool.get("description", "No description")
                xml_example = f'<{tool_name} ...>'
                tool_lines.append(f"{xml_example} - {description}")

            return f"Tools for {element_name}\n" + "\n".join(tool_lines)

        except Exception as e:
            logger.error(f"Error rendering structured tools: {e}", exc_info=True)
            return ""

    def _generate_system_message_for_status(self, status_facet: StatusFacet) -> str:
        """Generate individual system messages for status changes from facet data only."""
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
                # ENHANCED: Intelligent agent message rendering based on timestamp context
                is_from_current_agent = facet.get_property("is_from_current_agent", False)

                if is_from_current_agent and not options.render_agent_messages:
                    # Check if this agent message should be rendered or skipped
                    if not self._should_render_agent_message(facet, system_state):
                        logger.debug(f"Skipping agent message from {facet.get_property('timestamp_iso')}: tool call context available")
                        return ""  # Skip rendering - tool call context is available

                sender = facet.get_property("sender_name", "Unknown")
                is_edited = facet.get_property("is_edited", False)

                # Determine conversation from links or owner
                conversation_name = self._determine_conversation_for_event(facet, system_state)
                focus_content = f'<system>Currently focused on: {conversation_name}</system>'

                # Build message with conversation context
                msg_attrs = []
                if conversation_name:
                    msg_attrs.append(f'source="{conversation_name}"')
                if sender:
                    msg_attrs.append(f'sender="{sender}"')
                if is_edited:
                    msg_attrs.append('edited')
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
                elif message_status == "retrying":
                    retry_count = facet.get_property("retry_count", 0)
                    message_content += f" [RETRYING (attempt {retry_count})]"
                elif message_status == "failed_to_send":
                    retry_count = facet.get_property("retry_count", 0)
                    if retry_count > 0:
                        message_content += f" [SEND FAILED - {retry_count} retries]"
                    else:
                        message_content += " [SEND FAILED]"

                if self._focus_required(facet.facet_id):
                    return f"{focus_content}\n\n{msg_tag}{message_content}</msg>"

                return f"{msg_tag}{message_content}</msg>"

            elif event_type == "agent_response":
                return content

            elif event_type == "note_created":
                return f"<note>{content}</note>"

            elif event_type == "compressed_memory":
                # NEW: Render compressed memory EventFacets (M-chunks)
                return self._render_memory_event_facet(facet, system_state, options)

            else:
                return ""
                # return f"<event type=\"{event_type}\">{content}</event>"

        except Exception as e:
            logger.error(f"Error rendering event facet: {e}", exc_info=True)
            return ""

    def _render_memory_event_facet(self, facet: EventFacet, system_state: Dict[str, Any], options: RenderingOptions) -> str:
        """
        Render compressed memory EventFacets (M-chunks) with proper formatting.

        Memory EventFacets should be rendered with special formatting that:
        - Clearly indicates this is compressed content
        - Shows the temporal range covered
        - Displays the memory summary
        - Optionally shows metadata (token count, original message count)

        Args:
            facet: The memory EventFacet to render
            system_state: Current system state
            options: Rendering options

        Returns:
            Rendered memory content string
        """
        try:
            # Extract memory properties
            memory_summary = facet.get_property("content", "")

            memory_content = f'<memory>\n{memory_summary}\n</memory>'

            # NEW: Add timeline divergence indicators if present
            divergence_info = facet.get_property("timeline_divergence")
            if divergence_info:
                divergence_indicator = self._render_timeline_divergence_indicator(divergence_info)
                if divergence_indicator:
                    memory_content = divergence_indicator + "\n" + memory_content

                return memory_content

        except Exception as e:
            logger.error(f"Error rendering memory EventFacet {facet.facet_id}: {e}", exc_info=True)
            # Fallback rendering to ensure memory is not invisible
            content = facet.get_property("content", "")
            return f"<memory>[Compressed memory: {content}]</memory>"

    def _render_timeline_divergence_indicator(self, divergence_info: Dict[str, Any]) -> str:
        """
        Render timeline divergence indicators based on divergence information.

        Creates visual indicators to help users understand when memories appear
        out of chronological order or cause temporal conflicts.

        Args:
            divergence_info: Timeline divergence information from memory EventFacet

        Returns:
            Rendered divergence indicator string
        """
        try:
            divergence_type = divergence_info.get("divergence_type", "unknown")
            description = divergence_info.get("description", "Timeline irregularity detected")
            severity = divergence_info.get("severity", "low")

            # Choose indicator style based on severity
            if severity == "high":
                indicator_style = "⚠️  TIMELINE DIVERGENCE"
                wrapper = "timeline_divergence_high"
            elif severity == "medium":
                indicator_style = "⏰ Timeline Note"
                wrapper = "timeline_divergence_medium"
            else:
                indicator_style = "ℹ️  Timeline Info"
                wrapper = "timeline_divergence_low"

            # Create specific indicators based on divergence type
            if divergence_type == "temporal_overlap":
                overlapping_count = divergence_info.get("surrounding_context", {}).get("overlapping_events_count", 0)
                detail = f"This memory covers a time period that overlaps with {overlapping_count} other visible events."

            elif divergence_type == "insertion_order":
                detail = "This memory appears between events from different time periods due to mean timestamp positioning."

            elif divergence_type == "retroactive_insertion":
                time_gap = divergence_info.get("time_gap_seconds", 0)
                minutes = int(time_gap // 60)
                detail = f"This memory covers content from {minutes} minutes before the surrounding conversation context."

            else:
                detail = description

            # Format the complete indicator
            return f'<{wrapper}>\n{indicator_style}: {detail}\n</{wrapper}>'

        except Exception as e:
            logger.error(f"Error rendering timeline divergence indicator: {e}", exc_info=True)
            return "<timeline_divergence_low>\nℹ️  Timeline Note: Memory positioning may affect chronological flow.\n</timeline_divergence_low>"

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
                elif message_status == "retrying":
                    # Note: retry_count not available in this method context, using generic message
                    message_content += " [RETRYING]"
                elif message_status == "failed_to_send":
                    message_content += " [SEND FAILED]"

                return f"{msg_tag}{message_content}</msg>"

            elif event_type == "note_created":
                return f"<note>{content}</note>"

            elif event_type == "agent_response":
                return f'{content}'

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

    # --- MISSING HELPER METHODS IMPLEMENTATION ---

    def _initialize_system_state(self, temporal_stream: List[VEILFacet], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize system state for turn-based processing.

        Args:
            temporal_stream: Chronologically ordered list of VEILFacet instances
            options: Rendering options

        Returns:
            Initial system state dictionary
        """
        # Get agent and space info from the owner
        agent_name = "Unknown Agent"
        agent_description = "Agent workspace"
        if self.owner and hasattr(self.owner, 'get_parent_object'):
            parent_space = self.owner.get_parent_object()
            if parent_space:
                agent_name = getattr(parent_space, 'agent_name', agent_name)
                agent_description = getattr(parent_space, 'agent_description', agent_description)

        # Get focus context to identify focused element
        focus_context = options.get('focus_context', {})
        focus_element_id = focus_context.get('focus_element_id')

        return {
            "agent_name": agent_name,
            "agent_description": agent_description,
            "adapter_groups": {},  # Will be populated by status facets
            "active_channels": {},  # Channel tracking
            "focus_element_id": focus_element_id,  # Track focused element
            "last_status_timestamp": 0,
            "content_length_total": 0,
            "ambient_threshold": options.get("ambient_text_threshold", 500),
            "rendered_ambient_ids": set(),
            "status_message_history": []
        }

    def _has_multimodal_content(self, facet_cache: VEILFacetCache, focus_element_id: str) -> bool:
        """
        Check if facets contain multimodal content.

        Args:
            facet_cache: VEILFacetCache to check
            focus_element_id: Element ID to focus on

        Returns:
            True if multimodal content is present
        """
        # For now, return False as multimodal support will be added later
        # This can be enhanced to scan facets for attachment content
        return False

    def _count_multimodal_content(self, facet_cache: VEILFacetCache, focus_element_id: str) -> int:
        """
        Count multimodal attachments in facets.

        Args:
            facet_cache: VEILFacetCache to check
            focus_element_id: Element ID to focus on

        Returns:
            Number of multimodal attachments
        """
        # For now, return 0 as multimodal support will be added later
        return 0

    def _update_system_state(self, system_state: Dict[str, Any], status_facet: VEILFacet, options: Dict[str, Any]) -> None:
        """
        Update system state based on status facets.

        Args:
            system_state: System state to update
            status_facet: Status facet to process
            options: Rendering options
        """
        status_type = status_facet.get_property("status_type")
        current_state = status_facet.get_property("current_state", {})

        if status_type == "container_created":
            # Update adapter groups and channels
            adapter_type = current_state.get("adapter_type")
            server_name = current_state.get("server_name")
            alias = current_state.get("alias", "")
            conversation_name = current_state.get("conversation_name")
            element_id = current_state.get("element_id")

            if adapter_type and server_name:
                group_key = f"{adapter_type}_{server_name}"
                if group_key not in system_state["adapter_groups"]:
                    system_state["adapter_groups"][group_key] = {
                        "adapter_type": adapter_type,
                        "server_name": server_name,
                        "alias": alias,
                        "channels": []
                    }

                # Add channel info with deduplication
                channel_info = {
                    "conversation_name": conversation_name,
                    "element_id": element_id,
                    "adapter_type": adapter_type,
                    "server_name": server_name,
                    "is_focused": element_id == system_state.get("focus_element_id"),  # Set based on focus context
                    "available_tools": []
                }

                # FIXED: Deduplicate channels based on conversation_name
                existing_channels = system_state["adapter_groups"][group_key]["channels"]
                channel_exists = any(
                    ch.get("conversation_name") == conversation_name
                    for ch in existing_channels
                )

                if not channel_exists:
                    system_state["adapter_groups"][group_key]["channels"].append(channel_info)
                    logger.debug(f"Added new channel '{conversation_name}' to adapter group '{group_key}'")
                else:
                    logger.debug(f"Channel '{conversation_name}' already exists in adapter group '{group_key}' - skipping duplicate")

                system_state["active_channels"][element_id] = channel_info

        # Update timestamp tracking
        system_state["last_status_timestamp"] = max(
            system_state["last_status_timestamp"],
            status_facet.veil_timestamp
        )

    def _update_status_message_tracking(self, system_state: Dict[str, Any], status_facet: VEILFacet) -> None:
        """
        Track status messages for repetition logic.

        Args:
            system_state: System state to update
            status_facet: Status facet to track
        """
        status_entry = {
            "facet_id": status_facet.facet_id,
            "status_type": status_facet.get_property("status_type"),
            "timestamp": status_facet.veil_timestamp,
            "current_state": status_facet.get_property("current_state", {})
        }
        system_state["status_message_history"].append(status_entry)

    def _update_content_length_tracking(self, system_state: Dict[str, Any], content_length: int) -> None:
        """
        Update content length tracking for ambient threshold logic.

        Args:
            system_state: System state to update
            content_length: Length of content to add
        """
        system_state["content_length_total"] += content_length

    def _should_render_ambient_facet(self, ambient_facet: VEILFacet, system_state: Dict[str, Any],
                                   turn_index: int, has_status_changes: bool) -> bool:
        """
        DEPRECATED: This method was part of the old filtering approach.
        The new additive approach doesn't filter ambient facets.
        """
        logger.warning("_should_render_ambient_facet() is deprecated - new approach doesn't filter")
        return True  # Always return True for compatibility

    def _mark_ambient_as_rendered(self, ambient_facet: VEILFacet, system_state: Dict[str, Any], turn_index: int) -> None:
        """
        DEPRECATED: This method was part of the old filtering approach.
        The new approach doesn't need render tracking.
        """
        logger.warning("_mark_ambient_as_rendered() is deprecated - new approach doesn't need render tracking")
        # Do nothing

    def _render_repeated_status_messages(self, system_state: Dict[str, Any], turn_index: int) -> str:
        """
        Render repeated status messages when status changes occur.

        Args:
            system_state: Current system state
            turn_index: Current turn index

        Returns:
            Rendered status messages string
        """
        # For now, return empty - this can be enhanced to show repeated status
        return ""

    def _render_ambient_facet_individual(self, ambient_facet: VEILFacet) -> str:
        """
        Render individual ambient facet.

        Args:
            ambient_facet: Ambient facet to render

        Returns:
            Rendered ambient facet string
        """
        ambient_type = ambient_facet.get_property("ambient_type")
        content = ambient_facet.get_property("content", "")

        # Handle structured content
        if ambient_facet.get_property("data_format") == "structured" and isinstance(content, dict):
            # This would be handled by the consolidated rendering
            return ""

        # Handle string content
        if ambient_type == "tool_instructions":
            return f"<ambient_tools>{content}</ambient_tools>"
        else:
            return f"<ambient type=\"{ambient_type}\">{content}</ambient>"

    def _generate_xml_tool_call_format_instructions(self) -> str:
        """
        Generate XML tool call format instructions.

        Returns:
            XML format instruction string
        """
        return """TOOL CALL FORMAT: Use XML format for all tool calls. Do NOT use JSON format.

To call tools, use this ultra-concise XML format to send messages:

<msg source="element_name">message</msg>

and this format to use any other tool:
<tool_name param1="value1" param2="value2" source="element_name">

Examples:
<msg source="discord_chat">Hello, world!</msg>

<execute_command command="ls -la" source="Terminal">
<msg source="zulip_chat">Command executed!</msg>

Use the actual tool name as the XML element name. You can make multiple tool calls in one response. The source parameter specifies which conversation or element to use - choose from the sources listed for each tool type."""

    def _should_render_agent_message(self, message_facet: VEILFacet, system_state: Dict[str, Any]) -> bool:
        """
        ENHANCED: Intelligent decision on whether to render agent messages.

        Agent messages are rendered when:
        1. The message timestamp is older than the connection event (historical data, tool calls lost)
        2. The message is in a failed/pending state (tool call didn't complete)
        3. Explicit override flag is set

        Agent messages are skipped when:
        1. Recent agent messages where tool call context is available
        2. Successfully sent messages where agent_response facets exist

        Args:
            message_facet: The message_added EventFacet from current agent
            system_state: Current system state tracking

        Returns:
            True if agent message should be rendered, False to skip
        """
        try:
            # Get message timestamp
            message_timestamp = message_facet.get_property("timestamp_iso")
            if not message_timestamp:
                # No timestamp - default to rendering for safety
                logger.debug("Agent message has no timestamp - rendering for safety")
                return True

            # Check message status - always render failed/pending messages
            message_status = message_facet.get_property("message_status", "received")
            if message_status in ["failed_to_send", "pending_send", "pending_edit", "pending_delete"]:
                logger.debug(f"Agent message has status '{message_status}' - rendering")
                return True

            # Get connection timestamp for this conversation
            conversation_name = message_facet.get_property("conversation_name")
            connection_timestamp = self._get_connection_timestamp_for_conversation(conversation_name, system_state)

            if connection_timestamp is None:
                # No connection timestamp found - this might be historical data, render it
                logger.debug(f"No connection timestamp found for '{conversation_name}' - rendering agent message (likely historical)")
                return True

            # Compare timestamps - if message is older than connection, tool call context is likely lost
            if message_timestamp < connection_timestamp:
                logger.debug(f"Agent message ({message_timestamp}) is older than connection ({connection_timestamp}) - rendering (tool calls lost)")
                return True

            # Message is newer than connection - tool call context should be available, skip rendering
            logger.debug(f"Agent message ({message_timestamp}) is newer than connection ({connection_timestamp}) - skipping (tool call available)")
            return False

        except Exception as e:
            logger.error(f"Error determining agent message rendering: {e}", exc_info=True)
            # Default to rendering on error for safety
            return True

    def _get_connection_timestamp_for_conversation(self, conversation_name: str, system_state: Dict[str, Any]) -> Optional[float]:
        """
        Get the connection timestamp for a specific conversation.

        This helps determine when the agent connected to the conversation,
        which is used to decide if tool call context is available.

        Args:
            conversation_name: Name of the conversation
            system_state: Current system state tracking

        Returns:
            Connection timestamp or None if not found
        """
        try:
            # Look for connection timestamp in active channels
            active_channels = system_state.get("active_channels", {})

            for channel_info in active_channels.values():
                if channel_info.get("conversation_name") == conversation_name:
                    # Return the earliest status timestamp as connection time
                    # Could be enhanced to track specific connection events
                    return system_state.get("last_status_timestamp", 0)

            # If not found in active channels, check if this conversation was just connected
            # Could be enhanced to maintain per-conversation connection timestamps
            return system_state.get("last_status_timestamp", 0)

        except Exception as e:
            logger.error(f"Error getting connection timestamp for '{conversation_name}': {e}", exc_info=True)
            return None

    def _get_triggered_ambient_facets_for_turn(self,
                                             turn_data: Dict[str, Any],
                                             all_ambient_facets_dict: Dict[str, VEILFacet],
                                             system_state: Dict[str, Any],
                                             options: Dict[str, Any]) -> List[VEILFacet]:
        """
        NEW: Determine which ambient facets should be added to this turn based on triggers.

        Triggers:
        1. First turn (turn 0) - add all ambient facets
        2. Status changes - add all ambient facets (scene changes re-show tools)
        3. Content threshold exceeded - add all ambient facets

        Args:
            turn_data: Current turn data
            all_ambient_facets_dict: Dictionary of all available ambient facets
            system_state: Current system state
            options: Rendering options

        Returns:
            List of ambient facets to add to this turn (additive, no filtering)
        """
        try:
            triggered_facets = []
            turn_index = turn_data.get("turn_index", 0)
            has_status_changes = bool(turn_data.get("status", []))

            # Trigger 1: Always add all ambient facets on first turn
            if turn_index == 0:
                triggered_facets = list(all_ambient_facets_dict.values())
                logger.debug(f"Turn {turn_index}: Adding all {len(triggered_facets)} ambient facets (first turn)")
                return triggered_facets

            # NEW: Check for scene-changing status facets (ignore minor ones)
            scene_changing_status_facets = [
                facet for facet in turn_data.get("status", [])
                if self._is_scene_changing_status(facet)
            ]

            # Trigger 2: Major scene changes trigger ambient facets
            if scene_changing_status_facets:
                scene_types = [f.get_property("status_type") for f in scene_changing_status_facets]
                logger.debug(f"Turn {turn_index}: Scene changes detected: {scene_types}")
                triggered_facets = list(all_ambient_facets_dict.values())
                logger.debug(f"Turn {turn_index}: Adding all {len(triggered_facets)} ambient facets (scene changes)")
                return triggered_facets

            # Trigger 3: Content threshold exceeded - add all ambient facets
            content_threshold = system_state.get("ambient_threshold", 500)
            if system_state.get("content_length_total", 0) >= content_threshold:
                triggered_facets = list(all_ambient_facets_dict.values())
                logger.debug(f"Turn {turn_index}: Adding all {len(triggered_facets)} ambient facets (content threshold {system_state['content_length_total']} >= {content_threshold})")
                return triggered_facets

            # No significant triggers - minor status updates won't trigger ambient facets
            return []

        except Exception as e:
            logger.error(f"Error getting triggered ambient facets: {e}", exc_info=True)
            return []

    def _apply_simple_retroactive_deduplication(self, processed_turns: List[Dict[str, Any]]) -> None:
        """
        NEW: Simple reverse-iteration deduplication to keep only latest occurrence of each ambient facet.

        Algorithm:
        1. Iterate backwards through turns (latest to earliest)
        2. Track seen ambient facet IDs
        3. Keep ambient facet only if not seen before (i.e., this is its latest occurrence)
        4. Remove ambient facet if seen before (i.e., later occurrence exists)

        This ensures each ambient facet appears exactly once in its latest qualifying turn.

        Args:
            processed_turns: List of processed turn dictionaries with facets
        """
        try:
            seen_ambient_ids = set()
            removed_count = 0

            # Iterate backwards through turns (latest to earliest)
            for turn_data in reversed(processed_turns):
                turn_index = turn_data.get("turn_index", 0)
                current_ambient_facets = turn_data.get("ambient", [])

                if current_ambient_facets:
                    # Keep only ambient facets not seen in later turns
                    filtered_ambient = []
                    for facet in current_ambient_facets:
                        facet_id = facet.facet_id

                        if facet_id not in seen_ambient_ids:
                            # First time seeing this facet (latest occurrence) - keep it
                            filtered_ambient.append(facet)
                            seen_ambient_ids.add(facet_id)
                            logger.debug(f"Keeping ambient facet {facet_id} in turn {turn_index} (latest occurrence)")
                        else:
                            # Already seen in later turn - remove this earlier occurrence
                            removed_count += 1
                            logger.debug(f"Removing ambient facet {facet_id} from turn {turn_index} (later occurrence exists)")

                    turn_data["ambient"] = filtered_ambient

            logger.debug(f"Simple retroactive deduplication complete: {removed_count} earlier occurrences removed, each ambient facet appears once in latest qualifying turn")

        except Exception as e:
            logger.error(f"Error in simple retroactive deduplication: {e}", exc_info=True)

    def _extract_current_focus_info(self, options: Dict[str, Any]) -> None:
        """
        Extract current focus information.

        This implements the new focus flow where HUD extracts focus info and passes it
        to CompressionEngine for decision-making.

        Args:
            options: Rendering options (may contain explicit focus override)
        """
        # Check for explicit focus override in options first
        focus_context = options.get('focus_context', {})
        explicit_focus = focus_context.get('focus_element_id')

        if explicit_focus:
            # Check if this is actually a focus change
            current_focus_id = self._current_focus.get('focus_element_id') if self._current_focus else None
            if current_focus_id != explicit_focus:
                # This is a focus change - record it
                self.record_focus_change(focus_context)
            
            self._current_focus = {
                "focus_element_id": explicit_focus,
                "focus_source": "explicit_override",
                "focus_timestamp": ConnectomeEpoch.get_veil_timestamp(),
                "focus_context": focus_context
            }
            logger.debug(f"Focus set to: {explicit_focus}")
        else:
            # No explicit focus provided - keep current focus if any
            if not self._current_focus:
                logger.debug("No focus information available - no current focus set")
            else:
                logger.debug(f"Using existing focus: {self._current_focus.get('focus_element_id')}")

    def _focus_required(self, facet_id: str) -> bool:
        """
        Determine if focus is required for the current turn.

        Args:
            facet_id: The ID of the facet to check

        Returns:
            True if focus is required, False otherwise
        """
        if not self._current_focus or not facet_id:
            return False

        return self._current_focus.get('focus_context', {}).get('focus_element_id', None) in facet_id

    # --- Snapshot Cache APIs ---
    def get_snapshot(self, focus_element_id: str, max_age_ms: Optional[int] = None, generate_if_missing: bool = False) -> Optional[Dict[str, Any]]:
        try:
            entry = self._snapshot_cache.get(focus_element_id, max_age_ms)
            if entry:
                return entry
            if generate_if_missing:
                # Caller should await refresh_snapshot then re-fetch
                return None
            return None
        except Exception as e:
            logger.debug(f"HUD get_snapshot error: {e}")
            return None

    async def refresh_snapshot(self, focus_element_id: str) -> Optional[Dict[str, Any]]:
        try:
            options = {
                'focus_context': { 'focus_element_id': focus_element_id },
                'include_memory': True,
                'render_mode': 'normal',
                'include_time_markers': True,
                'include_system_messages': True
            }
            # Use VEIL-enhanced tools if needed; pass None to keep interface simple here
            snapshot = await self.get_agent_context_via_compression_engine(options=options, tools=None)
            # Store with timestamp and metadata
            wrapped = {
                "messages": snapshot if isinstance(snapshot, list) or isinstance(snapshot, dict) else snapshot,
                "_snapshot_timestamp": time.time(),
                "_focus_element_id": focus_element_id
            }
            self._snapshot_cache.put(focus_element_id, wrapped)
            # Optional: emit internal event here (skipped)
            return wrapped
        except Exception as e:
            logger.error(f"HUD refresh_snapshot error: {e}", exc_info=True)
            return None

    async def refresh_snapshots(self, focus_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for fid in focus_ids or []:
            snap = await self.refresh_snapshot(fid)
            if snap:
                results[fid] = snap
        return results
