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
        self._state.setdefault('_last_generated_notes', []) 
        self._state.setdefault('_has_produced_scratchpad_root_add_before', False)
        self._state.setdefault('_last_scratchpad_root_properties', {})
        logger.debug(f"ScratchpadVeilProducer initialized for Element {self.owner.id}")

    def _get_current_notes_from_owner(self) -> List[Any]:
        """Helper to retrieve notes from the owning ScratchpadElement."""
        if not self.owner:
            # Should not happen if component is properly attached
            logger.error(f"[{self.COMPONENT_TYPE}] Owner not set, cannot get notes.")
            return ([], True) # Indicate error
        note_storage = self.get_sibling_component("NoteStorageComponent")
        if not note_storage:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] NoteStorageComponent not found on owner.")
            return ([], True) # Indicate error
        return (note_storage.get_notes_from_state(), False)

    def get_full_veil(self) -> Optional[Dict[str, Any]]:
        """
        Generates the complete VEIL structure for the scratchpad notes.
        """
        current_notes_list, is_error = self._get_current_notes_from_owner()
        if is_error:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Error getting notes from owner. Cannot generate full VEIL.")
            return None # Return None if notes cannot be fetched

        child_veil_nodes = []
        # processed_notes_for_delta = [] # This state is updated in signal_delta_produced_this_frame

        for index, note_content in enumerate(current_notes_list):
            import hashlib
            import time
            from datetime import datetime
            note_hash_id = hashlib.md5(str(note_content).encode()).hexdigest()[:8] 
            
            # Add timestamp for time markers (operation_index handles chronological placement)
            current_timestamp = time.time()
            
            child_node = {
                "veil_id": f"{self.owner.id}_scratchpad_note_{note_hash_id}",
                "node_type": VEIL_NOTE_ITEM_TYPE,
                "properties": {
                    "structural_role": "list_item",
                    "content_nature": "text_note",
                    VEIL_NOTE_CONTENT_PROP: note_content,
                    # NEW: Add timestamp fields for time markers (not chronological placement)
                    "timestamp": current_timestamp,
                    "timestamp_iso": datetime.fromtimestamp(current_timestamp).isoformat() + "Z",
                    "note_timestamp": current_timestamp,
                    "created_at": datetime.fromtimestamp(current_timestamp).isoformat() + "Z"
                    # NOTE: operation_index will be added by SpaceVeilProducer for chronological placement
                },
                "children": [] 
            }
            child_veil_nodes.append(child_node)
            # processed_notes_for_delta.append(note_content) # Don't update here

        if not child_veil_nodes and not current_notes_list: # Ensure placeholder only if truly empty
            placeholder_node = {
                "veil_id": f"{self.owner.id}_scratchpad_empty_placeholder",
                "node_type": "scratchpad_placeholder",
                "properties": {
                    "structural_role": "placeholder",
                    "content_nature": "status_message",
                    "text": "Scratchpad is empty. You can add notes here."
                },
                "children": []
            }
            child_veil_nodes.append(placeholder_node)

        # Create the root container node for the scratchpad
        root_veil_node = {
            "veil_id": f"{self.owner.id}_scratchpad_root", # Consistent root ID
            "node_type": VEIL_SCRATCHPAD_ROOT_TYPE,
            "properties": {
                "structural_role": "container",
                "content_nature": "scratchpad_summary",
                "element_id": self.owner.id,
                "element_name": self.owner.name,
                "note_count": len(current_notes_list) # Count actual notes, not placeholder
            },
            "children": child_veil_nodes
        }
        # Baselines are updated in signal_delta_produced_this_frame
        return root_veil_node

    def calculate_delta(self) -> Optional[List[Dict[str, Any]]]:
        """
        Calculates the changes (delta) in the notes since the last VEIL generation.
        Detects added and removed notes and handles the scratchpad root node.
        """
        current_notes_list, is_error = self._get_current_notes_from_owner()
        if is_error:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Error getting notes from owner, cannot calculate delta.")
            return None

        delta_operations = []
        scratchpad_root_veil_id = f"{self.owner.id}_scratchpad_root"

        current_scratchpad_root_properties = {
            "structural_role": "container",
            "content_nature": "scratchpad_summary",
            "element_id": self.owner.id,
            "element_name": self.owner.name,
            "note_count": len(current_notes_list) 
        }
        
        parent_veil_id_for_scratchpad_root = None
        if self.owner and hasattr(self.owner, 'get_parent_info') and callable(getattr(self.owner, 'get_parent_info')):
            parent_info = self.owner.get_parent_info()
            if parent_info and parent_info.get('parent_id'):
                parent_space_id = parent_info['parent_id']
                parent_veil_id_for_scratchpad_root = f"{parent_space_id}_space_root"
                logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Scratchpad root '{scratchpad_root_veil_id}' will be parented to space root: {parent_veil_id_for_scratchpad_root}")
            else:
                logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Could not determine parent_space_id for scratchpad root from get_parent_info(). Parent info: {parent_info}")
        else:
            logger.warning(f"[{self.owner.id if self.owner else 'UnknownOwner'}/{self.COMPONENT_TYPE}] Owner does not have get_parent_info or it's not callable. Scratchpad root will not be parented.")

        if not self._state.get('_has_produced_scratchpad_root_add_before', False):
            logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Generating 'add_node' for scratchpad root '{scratchpad_root_veil_id}'.")
            add_node_op_for_root = {
                "op": "add_node",
                "node": {
                    "veil_id": scratchpad_root_veil_id,
                    "node_type": VEIL_SCRATCHPAD_ROOT_TYPE,
                    "properties": current_scratchpad_root_properties,
                    "children": [] 
                }
            }
            if parent_veil_id_for_scratchpad_root:
                add_node_op_for_root["parent_id"] = parent_veil_id_for_scratchpad_root
            delta_operations.append(add_node_op_for_root)
        else:
            last_root_props = self._state.get('_last_scratchpad_root_properties', {})
            if current_scratchpad_root_properties != last_root_props:
                logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Generating 'update_node' for scratchpad root '{scratchpad_root_veil_id}' properties.")
                delta_operations.append({
                    "op": "update_node",
                    "veil_id": scratchpad_root_veil_id,
                    "properties": current_scratchpad_root_properties
                })

        last_notes_list = self._state.get('_last_generated_notes', [])
        current_notes_str_set = {str(n) for n in current_notes_list}
        last_notes_str_set = {str(n) for n in last_notes_list}

        added_notes_str_content = current_notes_str_set - last_notes_str_set
        removed_notes_str_content = last_notes_str_set - current_notes_str_set
        
        notes_to_add_map = {str(n): n for n in current_notes_list}
        notes_to_remove_map = {str(n): n for n in last_notes_list}

        for note_str_content in removed_notes_str_content:
            original_note_content = notes_to_remove_map.get(note_str_content)
            if original_note_content is None: continue
            import hashlib
            note_hash_id = hashlib.md5(str(original_note_content).encode()).hexdigest()[:8]
            removed_veil_id = f"{self.owner.id}_scratchpad_note_{note_hash_id}"
            delta_operations.append({"op": "remove_node", "veil_id": removed_veil_id})

        for note_str_content in added_notes_str_content:
            original_note_content = notes_to_add_map.get(note_str_content)
            if original_note_content is None: continue
            import hashlib
            import time
            from datetime import datetime
            note_hash_id = hashlib.md5(str(original_note_content).encode()).hexdigest()[:8]
            
            # Add timestamp for time markers (operation_index handles chronological placement)
            current_timestamp = time.time()
            
            added_node = {
                "veil_id": f"{self.owner.id}_scratchpad_note_{note_hash_id}",
                "node_type": VEIL_NOTE_ITEM_TYPE,
                "properties": {
                    "structural_role": "list_item",
                    "content_nature": "text_note",
                    VEIL_NOTE_CONTENT_PROP: original_note_content,
                    # NEW: Add timestamp fields for time markers (not chronological placement)
                    "timestamp": current_timestamp,
                    "timestamp_iso": datetime.fromtimestamp(current_timestamp).isoformat() + "Z",
                    "note_timestamp": current_timestamp,
                    "created_at": datetime.fromtimestamp(current_timestamp).isoformat() + "Z"
                    # NOTE: operation_index will be added by SpaceVeilProducer for chronological placement
                },
                "children": []
            }
            delta_operations.append({
                "op": "add_node",
                "parent_id": scratchpad_root_veil_id,
                "node": added_node
            })

        # --- Update State After Deltas are Determined ---
        if not self._state.get('_has_produced_scratchpad_root_add_before', False):
            for delta_op in delta_operations:
                if delta_op.get("op") == "add_node" and isinstance(delta_op.get("node"), dict) and delta_op["node"].get("veil_id") == scratchpad_root_veil_id:
                    self._state['_has_produced_scratchpad_root_add_before'] = True
                    break
        
        self._state['_last_scratchpad_root_properties'] = current_scratchpad_root_properties
        self._state['_last_generated_notes'] = current_notes_list # Corrected key
        
        if delta_operations:
            logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Calculated Scratchpad VEIL delta with {len(delta_operations)} operations.")
        else:
            logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] No Scratchpad VEIL delta operations calculated.")
        
        logger.debug(
            f"[{self.owner.id}/{self.COMPONENT_TYPE}] calculate_delta finished. Baseline updated. "
            f"Scratchpad root props tracked. Notes count: {len(current_notes_list)}. "
            f"Root add produced: {self._state.get('_has_produced_scratchpad_root_add_before', False)}"
        )
        # --- End State Update ---
        return delta_operations
