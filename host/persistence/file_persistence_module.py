import os
import json
import logging
import shutil
from typing import Dict, Any, Optional

from .base_persistence_module import BasePersistenceModule

logger = logging.getLogger(__name__)

class FilePersistenceModule(BasePersistenceModule):
    """
    A simple persistence module that saves component state to JSON files.
    
    Structure: {base_dir}/elements/{element_id}/components/{component_type}.json
    """

    def __init__(self, base_dir: str = "./saved_state"):
        """
        Initializes the file persistence module.
        
        Args:
            base_dir: The root directory to store saved state files.
        """
        self.base_dir = os.path.abspath(base_dir)
        self._element_dir_base = os.path.join(self.base_dir, "elements")
        logger.info(f"Initializing FilePersistenceModule with base directory: {self.base_dir}")
        # Ensure base directory exists
        os.makedirs(self._element_dir_base, exist_ok=True)

    def _get_component_file_path(self, element_id: str, component_type: str, create_dirs: bool = False) -> str:
        """Constructs the full path for a component's state file."""
        # Sanitize component_type to be safe for filenames
        safe_component_type = component_type.replace('.', '_').replace('/', '_').replace('\\', '_')
        element_path = os.path.join(self._element_dir_base, element_id)
        component_dir = os.path.join(element_path, "components")
        if create_dirs:
            os.makedirs(component_dir, exist_ok=True)
        return os.path.join(component_dir, f"{safe_component_type}.json")

    def _get_element_path(self, element_id: str) -> str:
        """Constructs the full path for an element's directory."""
        return os.path.join(self._element_dir_base, element_id)

    def save_component_state(self, element_id: str, component_type: str, state: Dict[str, Any]) -> bool:
        """Saves component state to a JSON file."""
        file_path = self._get_component_file_path(element_id, component_type, create_dirs=True)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Use indent for readability during debugging
                json.dump(state, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved state for {component_type} of element {element_id} to {file_path}")
            return True
        except (IOError, TypeError, json.JSONDecodeError) as e:
            logger.error(f"Failed to save state for {component_type} of element {element_id} to {file_path}: {e}", exc_info=True)
            return False

    def load_component_state(self, element_id: str, component_type: str) -> Optional[Dict[str, Any]]:
        """Loads component state from a JSON file."""
        file_path = self._get_component_file_path(element_id, component_type)
        if not os.path.exists(file_path):
            logger.debug(f"No saved state file found for {component_type} of element {element_id} at {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            logger.debug(f"Loaded state for {component_type} of element {element_id} from {file_path}")
            return state
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load state for {component_type} of element {element_id} from {file_path}: {e}", exc_info=True)
            return None

    def delete_component_state(self, element_id: str, component_type: str) -> bool:
        """Deletes the state file for a specific component."""
        file_path = self._get_component_file_path(element_id, component_type)
        if not os.path.exists(file_path):
            logger.debug(f"State file for {component_type} of element {element_id} not found for deletion.")
            return True # Considered success if it doesn't exist
            
        try:
            os.remove(file_path)
            logger.info(f"Deleted state file for {component_type} of element {element_id}: {file_path}")
            # Optional: Clean up empty directories if desired, but might be complex/risky
            return True
        except OSError as e:
            logger.error(f"Failed to delete state file {file_path}: {e}", exc_info=True)
            return False

    def delete_element_state(self, element_id: str) -> bool:
        """Deletes the entire directory containing state for an element."""
        element_path = self._get_element_path(element_id)
        if not os.path.isdir(element_path):
             logger.debug(f"Element state directory for {element_id} not found for deletion.")
             return True # Considered success if it doesn't exist
             
        try:
            # Use shutil.rmtree to remove the directory and all its contents
            shutil.rmtree(element_path)
            logger.info(f"Deleted element state directory for {element_id}: {element_path}")
            return True
        except OSError as e:
            logger.error(f"Failed to delete element state directory {element_path}: {e}", exc_info=True)
            return False

    # Snapshot methods remain unimplemented in this basic version 