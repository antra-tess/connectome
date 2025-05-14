"""
Component Registry
Handles discovery and registration of Component classes.
"""
import logging
import pkgutil
import importlib
import inspect
from typing import Dict, Type, Optional

# Assuming components inherit from a common base
from .elements.components.base_component import Component 

logger = logging.getLogger(__name__)

# Global registry mapping COMPONENT_TYPE name to class
COMPONENT_REGISTRY: Dict[str, Type[Component]] = {}

def register_component(cls: Type[Component]):
    """
    Decorator to register a Component class in the global registry.
    Uses the class's COMPONENT_TYPE attribute as the key.
    """
    if not issubclass(cls, Component):
        raise TypeError(f"Class {cls.__name__} must inherit from BaseComponent to be registered.")
        
    component_type = getattr(cls, 'COMPONENT_TYPE', None)
    if not component_type or not isinstance(component_type, str):
        raise ValueError(f"Component class {cls.__name__} must have a valid string COMPONENT_TYPE attribute.")
        
    if component_type in COMPONENT_REGISTRY:
        # Allow re-registration during development/reloading?
        logger.warning(f"Component type '{component_type}' is already registered. Overwriting with {cls.__name__}.")
        
    COMPONENT_REGISTRY[component_type] = cls
    logger.debug(f"Registered component: '{component_type}' -> {cls.__name__}")
    return cls

def find_component_class(name: str) -> Optional[Type[Component]]:
    """
    Finds a registered component class by its COMPONENT_TYPE name.

    Args:
        name: The COMPONENT_TYPE name of the component.

    Returns:
        The component class if found, otherwise None.
    """
    return COMPONENT_REGISTRY.get(name)

def scan_and_load_components(package_path: str = 'elements.elements.components'):
    """
    Scans a package directory recursively and imports modules to trigger
    registration via the @register_component decorator.

    Args:
        package_path: The dotted path to the package containing components (e.g., 'elements.elements.components').
    """
    logger.info(f"Scanning for components in package: {package_path}")
    try:
        package = importlib.import_module(package_path)
        prefix = package.__name__ + '.'
        
        count = 0
        # pkgutil.walk_packages iterates through all submodules
        for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, prefix):
            try:
                # Import the module - this triggers @register_component decorators within
                importlib.import_module(modname)
                count += 1
                # logger.debug(f"Imported module: {modname}")
            except ImportError as e:
                logger.error(f"Failed to import module {modname} during component scan: {e}")
            except Exception as e:
                 logger.error(f"Error loading module {modname} during component scan: {e}", exc_info=True)
        
        logger.info(f"Component scan complete. Imported {count} potential component modules. Registry size: {len(COMPONENT_REGISTRY)}. Registry: {COMPONENT_REGISTRY.keys()}")
        
    except ImportError:
        logger.error(f"Could not import the base component package: {package_path}")
    except Exception as e:
        logger.error(f"Error during component scan of {package_path}: {e}", exc_info=True)

# --- Example Usage (Optional - for testing or explicit loading) ---
# def load_core_components():
#     # Explicitly import modules containing core components
#     # This ensures they are registered if scanning is not used or fails
#     try:
#         from .elements.components.space import timeline_component, container_component, space_veil_producer
#         from .elements.components.tool_provider import ToolProviderComponent
#         from .elements.components.scratchpad import note_storage_component, action_handler_component, scratchpad_veil_producer
#         # ... import others ...
#         logger.info("Explicitly loaded core components.")
#     except ImportError as e:
#         logger.error(f"Failed to explicitly load core components: {e}")

# Consider calling scan_and_load_components() early in the application startup,
# for instance, in host/main.py before initializing elements/spaces. 