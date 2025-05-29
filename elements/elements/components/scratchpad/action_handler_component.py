"""
Scratchpad Action Handler Component
Provides the logic for actions (tools) related to the scratchpad, 
interacting with NoteStorageComponent and the parent Space's timeline.
"""
import logging
from typing import Dict, Any, Optional, List

from ..base_component import Component
from .note_storage_component import NoteStorageComponent
from ..tool_provider import ToolParameter # Import the new ToolParameter type

# Import the registry decorator
from elements.component_registry import register_component

# Need access to parent Space's timeline and ToolProvider on the owner element
from ...base import BaseElement # For type hinting owner
from ...space import Space # To check parent type
from ..space.timeline_component import TimelineComponent
from ..tool_provider import ToolProviderComponent # To register tools

logger = logging.getLogger(__name__)

@register_component
class ScratchpadActionHandler(Component):
    """
    Implements the logic for scratchpad tools like add, get, clear.
    Relies on a sibling NoteStorageComponent for state and records actions
    on the parent Space's timeline. Registers its tools with the owner's
    ToolProviderComponent.
    """
    COMPONENT_TYPE = "ScratchpadActionHandler"
    REQUIRED_SIBLING_COMPONENTS = [NoteStorageComponent, ToolProviderComponent] # ToolProvider is now required

    def initialize(self, **kwargs) -> None:
        """Initializes the component and registers its tools."""
        super().initialize(**kwargs)
        self._register_scratchpad_tools()
        logger.debug(f"ScratchpadActionHandler initialized and tools registered for Element {self.owner.id if self.owner else 'Unknown'}")

    # --- Helper Methods --- 

    def _get_note_storage_comp(self) -> Optional[NoteStorageComponent]:
        """Helper to get the required sibling NoteStorageComponent."""
        return self.get_sibling_component(NoteStorageComponent)

    def _get_tool_provider_comp(self) -> Optional[ToolProviderComponent]:
        """Helper to get the required sibling ToolProviderComponent."""
        return self.get_sibling_component(ToolProviderComponent)

    def _get_parent_space_timeline(self) -> Optional[TimelineComponent]:
        """Traverse up to find the parent Space and its TimelineComponent."""
        if not self.owner:
            logger.warning(f"[{self.owner.id if self.owner else 'Handler'}] Cannot find parent space to record timeline event.")
            return None
        
        parent_space = self.owner.get_parent_object()
        # Ensure parent_space is an instance of Space before trying to get component
        # This check might be redundant if self.owner.parent_space is guaranteed to be a Space
        if isinstance(parent_space, BaseElement) and hasattr(parent_space, 'get_component_by_type'):
            timeline_comp = parent_space.get_component_by_type(TimelineComponent.COMPONENT_TYPE) # Use string type
            if not timeline_comp:
                 logger.warning(f"[{self.owner.id if self.owner else 'Handler'}] Parent space {parent_space.id} does not have a TimelineComponent.")
            return timeline_comp
        else:
            logger.warning(f"[{self.owner.id if self.owner else 'Handler'}] Owner's parent ({type(parent_space)}) is not a Space or lacks get_component_by_type. Cannot get timeline.")
            return None

    def _register_scratchpad_tools(self) -> None:
        """Registers scratchpad-related tools with the sibling ToolProviderComponent."""
        tool_provider = self._get_tool_provider_comp()
        if not tool_provider:
            logger.error(f"[{self.owner.id if self.owner else 'Handler'}] ToolProviderComponent not found. Cannot register scratchpad tools.")
            return

        add_note_params: List[ToolParameter] = [
            {"name": "note_content", "type": "string", "description": "The text content of the note to add.", "required": True}
        ]
        get_notes_params: List[ToolParameter] = [] # No parameters
        clear_notes_params: List[ToolParameter] = [] # No parameters

        tool_provider.register_tool_function(
            name="add_note_to_scratchpad",
            description="Adds a new text note to this scratchpad element.",
            parameters_schema=add_note_params,
            tool_func=self.handle_add_note
        )

        tool_provider.register_tool_function(
            name="get_notes_from_scratchpad",
            description="Retrieves all notes currently stored in this scratchpad element.",
            parameters_schema=get_notes_params,
            tool_func=self.handle_get_notes
        )

        tool_provider.register_tool_function(
            name="clear_all_scratchpad_notes",
            description="Clears all notes from this scratchpad element.",
            parameters_schema=clear_notes_params,
            tool_func=self.handle_clear_notes
        )
        logger.info(f"Scratchpad tools registered for Element {self.owner.id if self.owner else 'Unknown'}")

    # --- Action/Tool Logic Methods --- 

    def handle_add_note(self, note_content: str) -> Dict[str, Any]:
        """
        Logic for the 'add_note_to_scratchpad' tool. Adds note to state and records event.

        Args:
            note_content: The string content of the note.

        Returns:
            A dictionary result suitable for ToolProviderComponent execution:
            { "success": bool, "result": Any, "error": Optional[str] }
        """
        storage_comp = self._get_note_storage_comp()
        if not storage_comp:
            return { "success": False, "result": None, "error": "NoteStorageComponent not found." }
            
        success = storage_comp.add_note_to_state(note_content)
        
        if success:
            # Record event on parent space timeline
            timeline_comp = self._get_parent_space_timeline()
            if timeline_comp:
                event_payload = {
                    "event_type": "scratchpad_note_added",
                    "data": {
                        "element_id": self.owner.id if self.owner else None,
                        "note_content_preview": note_content[:50] + ('...' if len(note_content) > 50 else ''),
                        # Add any other relevant context
                    }
                }
                timeline_comp.add_event_to_primary_timeline(event_payload)
            else: 
                logger.warning("Could not record scratchpad_note_added event: Parent timeline not found.")
                
            return { "success": True, "result": "Note added successfully.", "error": None }
        else:
            return { "success": False, "result": None, "error": "Failed to add note to state (check logs for details)." }

    def handle_get_notes(self) -> Dict[str, Any]:
        """
        Logic for the 'get_notes_from_scratchpad' tool. Retrieves notes from state.

        Returns:
            A dictionary result: { "success": bool, "result": List[str], "error": Optional[str] }
        """
        storage_comp = self._get_note_storage_comp()
        if not storage_comp:
            return { "success": False, "result": [], "error": "NoteStorageComponent not found." }
            
        try:
            notes = storage_comp.get_notes_from_state()
            return { "success": True, "result": notes, "error": None }
        except Exception as e:
            error_msg = f"Error retrieving notes from state: {e}"
            logger.error(f"[{self.owner.id if self.owner else 'Handler'}] {error_msg}", exc_info=True)
            return { "success": False, "result": [], "error": error_msg }

    def handle_clear_notes(self) -> Dict[str, Any]:
        """
        Logic for the 'clear_all_scratchpad_notes' tool. Clears notes from state and records event.

        Returns:
            A dictionary result: { "success": bool, "result": str, "error": Optional[str] }
        """
        storage_comp = self._get_note_storage_comp()
        if not storage_comp:
            return { "success": False, "result": None, "error": "NoteStorageComponent not found." }
            
        # Get count before clearing for the event
        original_note_count = len(storage_comp.get_notes_from_state())
        
        success = storage_comp.clear_notes_in_state()
        
        if success:
            # Record event on parent space timeline
            timeline_comp = self._get_parent_space_timeline()
            if timeline_comp:
                event_payload = {
                    "event_type": "scratchpad_notes_cleared",
                    "data": {
                        "element_id": self.owner.id if self.owner else None,
                        "cleared_note_count": original_note_count
                    }
                }
                timeline_comp.add_event_to_primary_timeline(event_payload)
            else:
                logger.warning("Could not record scratchpad_notes_cleared event: Parent timeline not found.")
                
            result_msg = f"All notes cleared ({original_note_count} notes removed)."
            return { "success": True, "result": result_msg, "error": None }
        else:
            return { "success": False, "result": None, "error": "Failed to clear notes from state (check logs for details)." } 