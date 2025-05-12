"""
Scratchpad VEIL Producer Component
Generates VEIL representation for notes stored in a ScratchpadElement.
"""
import logging
from typing import Dict, Any, Optional, List, Set

# Assuming owner element has a get_notes() method or _notes attribute
from ..base_component import VeilProducer 
# Import the registry decorator
from elements.component_registry import register_component

logger = logging.getLogger(__name__)

# VEIL Node Structure Constants (Example)
VEIL_SCRATCHPAD_ROOT_TYPE = "scratchpad_root"
VEIL_NOTE_ITEM_TYPE = "note_item"
VEIL_NOTE_CONTENT_PROP = "note_content"
VEIL_NOTE_TIMESTAMP_PROP = "note_timestamp" # From event? Or element state?

@register_component
class ScratchpadVeilProducer(VeilProducer):
    """
    Generates VEIL representation for notes managed by the owning ScratchpadElement.
    Assumes the owner element provides a way to get the current list of notes 
    (e.g., a `get_notes()` method returning a list of strings or dicts).
    """
    COMPONENT_TYPE = "ScratchpadVeilProducer"

    # No specific sibling component dependencies, relies on owner state/methods.

    def initialize(self, **kwargs) -> None:
        """Initializes the component state for delta tracking."""
        super().initialize(**kwargs)
        # Track the notes represented in the last generated VEIL
        # Store as a list of note content/ids for simple comparison
        self._state.setdefault('_last_generated_notes', []) 
        logger.debug(f"ScratchpadVeilProducer initialized for Element {self.owner.id}")

    def _get_current_notes_from_owner(self) -> List[Any]:
        """Helper to retrieve notes from the owning ScratchpadElement."""
        if not self.owner:
            return []
        # Option 1: Assume a get_notes() method exists
        if hasattr(self.owner, 'get_notes') and callable(self.owner.get_notes):
            try:
                return self.owner.get_notes() # Expected: List[str] or List[Dict]
            except Exception as e:
                logger.error(f"[{self.owner.id}] Error calling get_notes() on owner: {e}")
                return []
        # Option 2: Fallback to accessing a protected attribute (less ideal)
        elif hasattr(self.owner, '_notes'):
            notes = getattr(self.owner, '_notes')
            return notes if isinstance(notes, list) else []
        else:
            logger.warning(f"[{self.owner.id}] ScratchpadVeilProducer cannot find notes on owner. Owner type: {type(self.owner)}")
            return []

    def get_full_veil(self) -> Optional[Dict[str, Any]]:
        """
        Generates the complete VEIL structure for the scratchpad notes.
        """
        current_notes = self._get_current_notes_from_owner()
        
        # Create child nodes for each note
        # Assuming notes are simple strings for now
        child_veil_nodes = []
        processed_notes_for_delta = [] # Track notes included in VEIL
        for index, note_content in enumerate(current_notes):
            # Need a stable ID for each note item in VEIL
            # Option A: Use index (brittle if notes reorder/delete)
            # Option B: Hash content (better, but collisions?)
            # Option C: Assume notes are dicts with IDs from owner? Let's hash for now.
            import hashlib
            note_id = hashlib.md5(str(note_content).encode()).hexdigest()[:8] 
            
            child_node = {
                "veil_id": f"{self.owner.id}_note_{note_id}",
                "node_type": VEIL_NOTE_ITEM_TYPE,
                "properties": {
                    "structural_role": "list_item",
                    "content_nature": "text_note",
                    VEIL_NOTE_CONTENT_PROP: note_content,
                    # Add timestamp if available, e.g., from note dict or event
                    # VEIL_NOTE_TIMESTAMP_PROP: note.get('timestamp') 
                },
                "children": [] # Notes are leaf nodes
            }
            child_veil_nodes.append(child_node)
            processed_notes_for_delta.append(note_content) # Use content for simple delta

        # Create the root container node for the scratchpad
        root_veil_node = {
            "veil_id": f"{self.owner.id}_scratchpad_root", 
            "node_type": VEIL_SCRATCHPAD_ROOT_TYPE,
            "properties": {
                "structural_role": "container",
                "content_nature": "scratchpad_summary",
                "element_id": self.owner.id,
                "element_name": self.owner.name,
                "note_count": len(child_veil_nodes)
            },
            "children": child_veil_nodes
        }

        # Update state for delta calculation
        self._state['_last_generated_notes'] = processed_notes_for_delta

        return root_veil_node

    def calculate_delta(self) -> Optional[List[Dict[str, Any]]]:
        """
        Calculates the changes (delta) in the notes since the last VEIL generation.
        Detects added and removed notes based on content comparison.
        NOTE: This simple version doesn't detect modifications to existing notes.
        """
        current_notes = self._get_current_notes_from_owner()
        last_notes = self._state.get('_last_generated_notes', [])

        # Using sets for efficient diff (assumes note contents are hashable)
        try:
             current_set = set(current_notes)
             last_set = set(last_notes)
        except TypeError:
             # Handle unhashable notes (e.g., dicts) - fallback to list comparison
             logger.warning(f"[{self.owner.id}] Notes contain unhashable types. Using slower list comparison for delta.")
             # TODO: Implement list-based diff if needed
             current_set = set(str(n) for n in current_notes) # Simple string conversion
             last_set = set(str(n) for n in last_notes)

        added_notes = current_set - last_set
        removed_notes = last_set - current_set

        delta_operations = []
        parent_veil_id = f"{self.owner.id}_scratchpad_root" # ID of the container node

        # 1. Detect removed notes
        for note_content in removed_notes:
            # Need to reconstruct the veil_id used previously
            import hashlib
            note_id = hashlib.md5(str(note_content).encode()).hexdigest()[:8]
            removed_veil_id = f"{self.owner.id}_note_{note_id}"
            delta_operations.append({
                "op": "remove_node",
                "veil_id": removed_veil_id
            })

        # 2. Detect added notes
        processed_notes_for_delta = [] # Store notes included in VEIL
        for note_content in current_notes: # Iterate current notes to preserve order for add
             processed_notes_for_delta.append(note_content)
             if note_content in added_notes:
                 import hashlib
                 note_id = hashlib.md5(str(note_content).encode()).hexdigest()[:8]
                 added_node = {
                     "veil_id": f"{self.owner.id}_note_{note_id}",
                     "node_type": VEIL_NOTE_ITEM_TYPE,
                     "properties": {
                         "structural_role": "list_item",
                         "content_nature": "text_note",
                         VEIL_NOTE_CONTENT_PROP: note_content,
                     },
                     "children": []
                 }
                 delta_operations.append({
                     "op": "add_node",
                     "parent_id": parent_veil_id,
                     "node": added_node,
                     # TODO: Add position hint?
                 })
                 added_notes.remove(note_content) # Avoid adding duplicates if content appears multiple times

        # Update state for next delta calculation
        self._state['_last_generated_notes'] = processed_notes_for_delta

        if delta_operations:
            logger.info(f"[{self.owner.id}] Calculated Scratchpad VEIL delta with {len(delta_operations)} operations.")
        return delta_operations
