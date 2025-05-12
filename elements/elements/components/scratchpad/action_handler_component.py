"""
Scratchpad Action Handler Component
Provides the logic for actions (tools) related to the scratchpad, 
interacting with NoteStorageComponent and the parent Space's timeline.
"""
import logging
from typing import Dict, Any, Optional, List

from ..base_component import Component
from .note_storage_component import NoteStorageComponent

# Import the registry decorator
from elements.component_registry import register_component

# Need access to parent Space's timeline
from ...space import Space # To check parent type
from ..space.timeline_component import TimelineComponent

logger = logging.getLogger(__name__)

@register_component
class ScratchpadActionHandler(Component):
    """
    Implements the logic for scratchpad tools like add, get, clear.
    Relies on a sibling NoteStorageComponent for state and records actions
    on the parent Space's timeline.
    """
    COMPONENT_TYPE = "ScratchpadActionHandler"
    REQUIRED_SIBLING_COMPONENTS = [NoteStorageComponent]

    def initialize(self, **kwargs) -> None:
        """Initializes the component."""
        super().initialize(**kwargs)
        logger.debug(f"ScratchpadActionHandler initialized for Element {self.owner.id if self.owner else 'Unknown'}")

    # --- Helper Methods --- 

    def _get_note_storage_comp(self) -> Optional[NoteStorageComponent]:
        """Helper to get the required sibling NoteStorageComponent."""
        # Use the base class helper for required siblings
        return self.get_sibling_component(NoteStorageComponent)

    def _get_parent_space_timeline(self) -> Optional[TimelineComponent]:
        """Traverse up to find the parent Space and its TimelineComponent."""
        if not self.owner or not self.owner.parent_space:
            logger.warning(f"[{self.owner.id if self.owner else 'Handler'}] Cannot find parent space to record timeline event.")
            return None
        
        parent_space = self.owner.parent_space
        if isinstance(parent_space, Space):
            timeline_comp = parent_space.get_component_by_type(TimelineComponent.COMPONENT_TYPE)
            if not timeline_comp:
                 logger.warning(f"[{self.owner.id if self.owner else 'Handler'}] Parent space {parent_space.id} does not have a TimelineComponent.")
            return timeline_comp
        else:
            logger.warning(f"[{self.owner.id if self.owner else 'Handler'}] Owner's parent is not a Space ({type(parent_space)}). Cannot get timeline.")
            return None

    # --- Action/Tool Logic Methods --- 

    def handle_add_note(self, note_content: str) -> Dict[str, Any]:
        """
        Logic for the 'add_note' tool. Adds note to state and records event.

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
        Logic for the 'get_notes' tool. Retrieves notes from state.

        Returns:
            A dictionary result: { "success": bool, "result": List[str], "error": Optional[str] }
        """
        storage_comp = self._get_note_storage_comp()
        if not storage_comp:
            return { "success": False, "result": [], "error": "NoteStorageComponent not found." }
            
        try:
            notes = storage_comp.get_notes_from_state()
            # Optional: Record retrieval event?
            # timeline_comp = self._get_parent_space_timeline()
            # if timeline_comp:
            #     event_payload = { ... "event_type": "scratchpad_notes_retrieved" ...}
            #     timeline_comp.add_event_to_primary_timeline(event_payload)
                
            return { "success": True, "result": notes, "error": None }
        except Exception as e:
            error_msg = f"Error retrieving notes from state: {e}"
            logger.error(f"[{self.owner.id if self.owner else 'Handler'}] {error_msg}", exc_info=True)
            return { "success": False, "result": [], "error": error_msg }

    def handle_clear_notes(self) -> Dict[str, Any]:
        """
        Logic for the 'clear_notes' tool. Clears notes from state and records event.

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