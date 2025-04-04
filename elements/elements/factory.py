import yaml
import importlib
import inspect # Added for class inspection
from pathlib import Path
from typing import Any, Dict, Type, Optional, List
import logging
import sys # To manage module path

from .base import BaseElement
from .components.base_component import BaseComponent

# Configure logging
logger = logging.getLogger(__name__)

# --- Component Registry ---
# Populated by scanning the components directory.
COMPONENT_REGISTRY: Dict[str, Type[BaseComponent]] = {}
_COMPONENT_SCAN_DONE = False # Flag to ensure scanning happens only once

def register_component(name: str, component_class: Type[BaseComponent]):
    """Registers a component class with the factory."""
    if name in COMPONENT_REGISTRY and COMPONENT_REGISTRY[name] != component_class:
        logger.warning(f"Component type '{name}' is already registered with a different class ({COMPONENT_REGISTRY[name].__name__}). Overwriting with {component_class.__name__}.")
    elif name not in COMPONENT_REGISTRY:
         logger.debug(f"Registered component: {name} -> {component_class.__name__}")
    COMPONENT_REGISTRY[name] = component_class


def _scan_and_register_components(base_dir_str: str = "elements/elements/components"):
    """
    Scans the specified directory recursively for component modules (*_component.py)
    and registers the BaseComponent subclasses found within them.
    """
    global _COMPONENT_SCAN_DONE
    if _COMPONENT_SCAN_DONE:
        return
    _COMPONENT_SCAN_DONE = True # Mark as done even if errors occur below

    logger.info(f"Scanning for components in '{base_dir_str}'...")
    base_dir = Path(base_dir_str)
    if not base_dir.is_dir():
        logger.error(f"Component base directory '{base_dir}' not found. Cannot scan for components.")
        return

    # Ensure the root 'elements' directory is in the path for relative imports
    # This might need adjustment based on how the project is run.
    project_root = Path(__file__).parent.parent # Assumes factory.py is in elements/elements
    if str(project_root.parent) not in sys.path:
        # Add the parent directory of 'elements' (likely the project root) to sys.path
        sys.path.insert(0, str(project_root.parent))

    # Use pathlib's rglob to find all potential component files
    for component_file in base_dir.rglob("*_component.py"):
        # Construct the module path relative to the project structure
        # e.g., elements/elements/components/messaging/history_component.py
        # needs to become elements.elements.components.messaging.history_component
        try:
            relative_path = component_file.relative_to(project_root.parent) # Path relative to project root added to sys.path
            module_dotted_path = str(relative_path.with_suffix('')).replace('/', '.').replace('\\', '.')

            logger.debug(f"Attempting to import module: {module_dotted_path}")
            module = importlib.import_module(module_dotted_path)

            # Inspect the imported module for BaseComponent subclasses
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BaseComponent) and obj is not BaseComponent:
                    # Register the class using its actual name (e.g., "HistoryComponent")
                    register_component(obj.__name__, obj)

        except ImportError as e:
            logger.warning(f"Could not import module {module_dotted_path}: {e}")
        except Exception as e:
            logger.error(f"Error processing component file {component_file}: {e}", exc_info=True) # Log traceback

    logger.info(f"Component scan complete. Registered components: {list(COMPONENT_REGISTRY.keys())}")


def _find_component_class(type_name: str) -> Optional[Type[BaseComponent]]:
    """
    Finds a component class by its type name.
    First checks the registry (populated by the scan), then attempts dynamic import as a fallback.
    """
    # Ensure the scan has run at least once
    if not _COMPONENT_SCAN_DONE:
         _scan_and_register_components() # Run scan on first lookup if not already done

    component_class = COMPONENT_REGISTRY.get(type_name)
    if component_class:
        return component_class

    # Fallback: Attempt dynamic import (simple heuristic based on convention)
    # This might catch components added after initial scan if they follow the pattern.
    logger.warning(f"Component type '{type_name}' not found in registry after scan. Attempting fallback dynamic import.")
    try:
        # Assuming components live directly under elements.elements.components (adjust if needed)
        # Example: HistoryComponent -> .components.history
        module_suffix = type_name.lower().replace('component', '')
        # Try common locations - this heuristic is less reliable than the initial scan.
        potential_modules = [
            f".components.{module_suffix}", # e.g., .components.history
             # Add more patterns here if components are nested deeper by convention
             # e.g., f".components.core.{module_suffix}",
        ]
        for module_name_rel in potential_modules:
            try:
                # Perform relative import within the 'elements.elements' package
                module = importlib.import_module(module_name_rel, package="elements.elements")
                component_class = getattr(module, type_name, None)
                if component_class and issubclass(component_class, BaseComponent):
                    logger.info(f"Dynamically imported and registered fallback component: {type_name}")
                    register_component(type_name, component_class) # Register after successful import
                    return component_class
            except ImportError:
                logger.debug(f"Fallback import failed for {module_name_rel}")
                continue # Try next potential location
            except AttributeError:
                 logger.warning(f"AttributeError while trying fallback load of {type_name} from {module_name_rel}.")
                 continue

    except Exception as e:
         logger.error(f"Unexpected error during fallback dynamic import for {type_name}: {e}", exc_info=True)

    logger.error(f"Component type '{type_name}' not found in registry and fallback dynamic import failed.")
    return None


# --- Element Factory ---

class ElementFactory:
    """
    Factory responsible for creating Element instances from prefab configurations.
    Scans for components automatically on initialization.
    """
    def __init__(self, prefab_directory: str = "prefabs"):
        self.prefab_dir = Path(prefab_directory)
        if not self.prefab_dir.is_dir():
            logger.warning(f"Prefab directory '{self.prefab_dir}' does not exist. Creating it.")
            try:
                self.prefab_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                 logger.error(f"Failed to create prefab directory '{self.prefab_dir}': {e}")
                 # Decide how critical this is - raise error?
                 # raise FileNotFoundError(f"Prefab directory '{self.prefab_dir}' not found and could not be created.")

        # Scan for components upon factory initialization
        _scan_and_register_components()

    def load_prefab_config(self, prefab_name: str) -> Optional[Dict[str, Any]]:
        """Loads the configuration data from a prefab YAML file."""
        config_path = self.prefab_dir / f"{prefab_name}.yaml"
        if not config_path.is_file():
            logger.error(f"Prefab configuration file not found: {config_path}")
            return None
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            if not isinstance(config_data, dict):
                logger.error(f"Invalid YAML format in {config_path}. Expected a dictionary.")
                return None
            # Basic validation
            if config_data.get("type") != "element_prefab":
                 logger.warning(f"Prefab {config_path} missing 'type: element_prefab'.")
                 # Allow proceeding but warn
            if "components" not in config_data or not isinstance(config_data["components"], list):
                 logger.error(f"Prefab {config_path} missing or invalid 'components' list.")
                 return None
            return config_data
        except yaml.YAMLError as e:
            logger.error(f"Error parsing prefab YAML file {config_path}: {e}")
            return None
        except IOError as e:
            logger.error(f"Error reading prefab file {config_path}: {e}")
            return None

    def get_available_prefabs(self) -> List[str]:
        """Scans the prefab directory and returns a list of available prefab names (without .yaml extension)."""
        if not self.prefab_dir.is_dir():
            logger.warning(f"Prefab directory '{self.prefab_dir}' not found when listing prefabs.")
            return []

        prefab_names = []
        try:
            for prefab_file in self.prefab_dir.glob("*.yaml"):
                # Optionally load and validate prefab type here? For now, just list files.
                prefab_names.append(prefab_file.stem) # .stem gets filename without extension
        except Exception as e:
            logger.error(f"Error scanning prefab directory '{self.prefab_dir}': {e}", exc_info=True)

        return sorted(prefab_names)

    def create_element(self, element_id: str, prefab_name: str, initial_state: Optional[Dict[str, Any]] = None) -> Optional[BaseElement]:
        """
        Creates a BaseElement instance based on a prefab configuration.

        Args:
            element_id: The unique ID for the new element instance.
            prefab_name: The name of the prefab configuration file (without .yaml extension).
            initial_state: Optional dictionary to override component configurations.
                           Structure: { "ComponentName": { "config_key": value, ... }, ... }

        Returns:
            A configured BaseElement instance or None if creation fails.
        """
        config_data = self.load_prefab_config(prefab_name)
        if not config_data:
            return None

        # Prepare the initial state override dictionary
        overrides = initial_state or {}

        # --- TODO: Incorporate initial_state overrides for *element* properties (e.g., name)? ---
        # Currently only handling component config overrides.

        element_name = config_data.get("name", prefab_name) # Use prefab name as default element name
        # Potential override for element name itself could go here if needed
        # element_name = overrides.get("_element", {}).get("name", element_name)

        element = BaseElement(element_id=element_id, name=element_name)
        logger.info(f"Creating element '{element_id}' ({element_name}) from prefab '{prefab_name}'")

        component_configs = config_data.get("components", [])
        if not component_configs:
             logger.warning(f"Prefab '{prefab_name}' has no components defined.")

        for comp_config in component_configs:
            if not isinstance(comp_config, dict) or "type" not in comp_config:
                logger.warning(f"Skipping invalid component definition in '{prefab_name}': {comp_config}")
                continue

            comp_type_name = comp_config["type"]
            # Start with config from prefab
            comp_args = comp_config.get("config", {}).copy() # Use copy to avoid modifying original config_data

            if not isinstance(comp_args, dict):
                 logger.warning(f"Invalid base 'config' for component '{comp_type_name}' in '{prefab_name}'. Expected dict, got {type(comp_args)}. Skipping component.")
                 continue

            # Apply overrides from initial_state if present
            if comp_type_name in overrides:
                component_overrides = overrides[comp_type_name]
                if isinstance(component_overrides, dict):
                    logger.debug(f"Applying overrides for {comp_type_name}: {component_overrides}")
                    comp_args.update(component_overrides) # Merge overrides, taking precedence
                else:
                    logger.warning(f"Invalid override format for component '{comp_type_name}' in initial_state. Expected dict, got {type(component_overrides)}. Skipping overrides.")

            component_class = _find_component_class(comp_type_name)
            if component_class:
                try:
                    # Instantiate component, passing the merged config dictionary
                    component_instance = component_class(element=element, **comp_args)
                    element.add_component(component_instance)
                    logger.debug(f"Added component '{comp_type_name}' to element {element_id} with final config: {comp_args}")
                except TypeError as e:
                     logger.error(f"Error instantiating component {comp_type_name} for element {element_id}. "
                                  f"Check __init__ signature and final config args ({comp_args}). Error: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error creating component {comp_type_name} for element {element_id}: {e}", exc_info=True)
            else:
                logger.error(f"Could not add component '{comp_type_name}' to element {element_id} as class was not found.")

        return element

# Example Usage (could be in a main script or test)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    factory = ElementFactory()

    print(f"\nComponents found by scan: {list(COMPONENT_REGISTRY.keys())}")

    # --- Create elements with and without overrides --- 

    # 1. Create using only prefab defaults
    print("\n--- Creating element chat_1 (default prefab) ---")
    chat_element_default = factory.create_element(element_id="chat_1", prefab_name="chat_element")

    if chat_element_default:
        print(f"Successfully created element: {chat_element_default.id} ({chat_element_default.name})")
        # Example check for HistoryComponent
        try:
             from .components.history import HistoryComponent # Import needed for type hint/lookup
             history_comp = chat_element_default.get_component(HistoryComponent)
             if history_comp:
                 max_entries_val = getattr(history_comp, 'max_entries', '{Attribute missing}')
                 print(f"  (History component max_entries check: {max_entries_val})") # Expect 1000 from prefab
             else:
                  print("  (History component not found on chat_1)")
        except ImportError: print("Could not import HistoryComponent for demo.")
        except Exception as e: print(f"Error during HistoryComponent check: {e}")
    else:
        print("Failed to create chat_1 element.")

    # 2. Create with initial_state override for HistoryComponent
    print("\n--- Creating element chat_2 (with override) ---")
    override_state = {
        "HistoryComponent": {
            "max_entries": 50 # Override value
        }
        # Can add overrides for other components here, e.g.:
        # "PublisherComponent": { "batch_size": 5 }
    }
    chat_element_override = factory.create_element(
        element_id="chat_2",
        prefab_name="chat_element",
        initial_state=override_state
    )

    if chat_element_override:
        print(f"Successfully created element: {chat_element_override.id} ({chat_element_override.name})")
        # Example check for HistoryComponent
        try:
             # Re-import not strictly necessary but clearer for example
             from .components.history import HistoryComponent
             history_comp = chat_element_override.get_component(HistoryComponent)
             if history_comp:
                 max_entries_val = getattr(history_comp, 'max_entries', '{Attribute missing}')
                 print(f"  (History component max_entries check: {max_entries_val})") # Expect 50 from override
             else:
                  print("  (History component not found on chat_2)")
        except ImportError: print("Could not import HistoryComponent for demo.")
        except Exception as e: print(f"Error during HistoryComponent check: {e}")
    else:
        print("Failed to create chat_2 element.") 