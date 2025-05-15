import logging
from typing import Dict, Any, Optional, List

from ..base_component import Component
# Import the registry decorator
from elements.component_registry import register_component

logger = logging.getLogger(__name__)

@register_component
class NoteStorageComponent(Component):
    """
    Manages the list of notes (state) for its owner element.
    Provides basic methods to add, retrieve, and clear notes from its internal state.
    """
    COMPONENT_TYPE = "NoteStorageComponent"

    def initialize(self, **kwargs) -> None:
        """Initializes the component state."""
        super().initialize(**kwargs)
        # _notes: List of simple strings for now.
        self._state.setdefault('_notes', []) 
        logger.debug(f"NoteStorageComponent initialized for Element {self.owner.id if self.owner else 'Unknown'}")

    def add_note_to_state(self, note_content: str) -> bool:
        """
        Adds a new note to the internal state.

        Args:
            note_content: The string content of the note.

        Returns:
            True if the note was added successfully, False otherwise.
        """
        is_note_written = False
        if not isinstance(note_content, str): # Basic validation
            logger.warning(f"[{self.owner.id if self.owner else 'Unknown'}] Attempted to add invalid note content type: {type(note_content)}")
            is_note_written = False
        if not note_content: # Disallow empty notes
            logger.warning(f"[{self.owner.id if self.owner else 'Unknown'}] Attempted to add empty note content.")
            is_note_written = False
            
        try:
            self._state['_notes'].append(note_content)
            logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] Note added to state. Current count: {len(self._state['_notes'])}")
            is_note_written = True
        except Exception as e:
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Error adding note to state: {e}", exc_info=True)
            is_note_written = False
        
        if is_note_written:
            veil_producer = self.get_sibling_component("ScratchpadVeilProducer")
            veil_producer.emit_delta()
        return is_note_written

    def get_notes_from_state(self) -> List[str]:
        """
        Retrieves the current list of notes from state.

        Returns:
            A list containing the note strings.
        """
        # Return a copy to prevent external modification of the internal list
        return list(self._state.get('_notes', []))

    def clear_notes_in_state(self) -> bool:
        """
        Removes all notes from the internal state.

        Returns:
            True if notes were cleared successfully, False otherwise.
        """
        is_notes_cleared = False
        try:
            original_note_count = len(self._state.get('_notes', []))
            if original_note_count == 0:
                logger.debug(f"[{self.owner.id if self.owner else 'Unknown'}] clear_notes_in_state called, but no notes exist in state.")
                # No action needed, considered success
                is_notes_cleared = True
                
            self._state['_notes'] = []
            logger.info(f"[{self.owner.id if self.owner else 'Unknown'}] All notes cleared from state (removed {original_note_count} notes).")
            is_notes_cleared = True
        except Exception as e:
            logger.error(f"[{self.owner.id if self.owner else 'Unknown'}] Error clearing notes from state: {e}", exc_info=True)
            is_notes_cleared = False 
        
        if is_notes_cleared:
            veil_producer = self.get_sibling_component("ScratchpadVeilProducer")
            veil_producer.emit_delta()
        return is_notes_cleared