"""
Element Factory Component
Provides tools for an agent to dynamically create and configure new elements
within its InnerSpace.
"""
import logging
from typing import Dict, Any, Optional, List, Type

from ..base import BaseElement
from .base_component import Component, BaseComponent # Import BaseComponent for type check

# Use the new component registry
from elements.component_registry import COMPONENT_REGISTRY, find_component_class, register_component

# Need access to parent Space to mount elements
from ..space import Space 
from ..inner_space import InnerSpace # Specifically expect InnerSpace

# Import PREFABS from the dedicated module
from elements.prefabs import PREFABS

logger = logging.getLogger(__name__)

# --- Prefab Definitions ---
# MOVED to elements/prefabs.py

@register_component
class ElementFactoryComponent(Component):
    """
    Provides tools for creating and managing elements dynamically.
    Should be attached to an InnerSpace element.
    """
    COMPONENT_TYPE = "ElementFactoryComponent"
    # No specific sibling dependencies, relies on owner being InnerSpace
    # and access to the Component Registry.

    def initialize(self, **kwargs) -> None:
        """Initializes the component."""
        super().initialize(**kwargs)
        # Ensure owner is an InnerSpace for mounting capabilities
        if not isinstance(self.owner, InnerSpace):
            logger.error(f"ElementFactoryComponent must be owned by an InnerSpace, not {type(self.owner)}. Functionality will be limited.")
        # Ensure component registry is populated (assuming scan was called at startup)
        if not COMPONENT_REGISTRY:
             logger.error("Component Registry is empty! ElementFactory cannot function. Was scan_and_load_components called?")
        logger.debug(f"ElementFactoryComponent initialized for Element {self.owner.id if self.owner else 'Unknown'}")

    def _get_inner_space(self) -> Optional[InnerSpace]:
        """Helper to get the owner cast as InnerSpace."""
        if isinstance(self.owner, InnerSpace):
            return self.owner
        return None

    # --- Tool Logic Methods --- 

    def handle_list_available_prefabs(self) -> Dict[str, Any]:
        """
        Logic for the 'list_available_prefabs' tool.
        Returns a dictionary of available prefab names and their descriptions.
        """
        try:
            prefab_info = { name: data.get("description", "No description.") 
                            for name, data in PREFABS.items() }
            return { "success": True, "result": prefab_info, "error": None }
        except Exception as e:
            logger.error(f"Error listing prefabs: {e}", exc_info=True)
            return { "success": False, "result": {}, "error": f"Failed to list prefabs: {e}" }

    def handle_list_available_components(self) -> Dict[str, Any]:
        """
        Logic for the 'list_available_components' tool. 
        Returns a list of registered component type names.
        """
        try:
            # TODO: Replace with proper registry access
            component_names = list(COMPONENT_REGISTRY.keys())
            component_names.sort()
            return { "success": True, "result": component_names, "error": None }
        except Exception as e:
            logger.error(f"Error listing components: {e}", exc_info=True)
            return { "success": False, "result": [], "error": f"Failed to list components: {e}" }

    def handle_create_element_from_prefab(
        self,
        prefab_name: str,
        element_id: str, 
        name: str, 
        config_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Logic for the 'create_element_from_prefab' tool.
        Creates an element using a predefined prefab template.

        Args:
            prefab_name: The name of the prefab to use (see list_available_prefabs).
            element_id: The desired unique ID for the new element.
            name: The human-readable name for the new element.
            config_overrides: Optional. A dictionary where keys are component type names
                              and values are dictionaries of config settings to 
                              override or provide for that component.
                              E.g., {"MessageListComponent": {"channel_id": "12345"}}

        Returns:
            A dictionary result: { "success": bool, "result": str, "error": Optional[str] }
        """
        config_overrides = config_overrides or {}
        
        # 1. Find prefab
        prefab_data = PREFABS.get(prefab_name)
        if not prefab_data:
            return { "success": False, "result": None, "error": f"Prefab '{prefab_name}' not found." }

        # 2. Construct final component configurations
        final_component_configs = []
        prefab_components = prefab_data.get("components", [])
        required_configs = prefab_data.get("required_configs", {})
        missing_requirements = []

        for base_comp_config in prefab_components:
            comp_type = base_comp_config.get("type")
            if not comp_type:
                 logger.warning(f"Prefab '{prefab_name}' has invalid component entry: {base_comp_config}. Skipping.")
                 continue
            
            # Start with base config from prefab
            final_config = base_comp_config.get("config", {}).copy()
            
            # Apply overrides provided by the agent
            if comp_type in config_overrides:
                override = config_overrides[comp_type]
                if isinstance(override, dict):
                    final_config.update(override)
                else:
                    logger.warning(f"Invalid override format for {comp_type} in prefab '{prefab_name}'. Expected dict, got {type(override)}. Ignoring override.")
            
            # Check for required configurations
            if comp_type in required_configs:
                for req_key in required_configs[comp_type]:
                    if req_key not in final_config or final_config[req_key] is None:
                        missing_requirements.append(f"Component '{comp_type}' requires config key '{req_key}'.")
                        
            final_component_configs.append({"type": comp_type, "config": final_config})
            
        # Check if all requirements were met
        if missing_requirements:
            error_msg = f"Missing required configurations for prefab '{prefab_name}': {'; '.join(missing_requirements)}"
            logger.error(error_msg)
            return { "success": False, "result": None, "error": error_msg }

        # 3. Call the base element creation logic
        # Pass the fully constructed list of component configs
        return self.handle_create_element(
            element_id=element_id, 
            name=name, 
            component_configs=final_component_configs
        )

    def handle_create_element(self, element_id: str, name: str, component_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Logic for the 'create_element' tool.
        Creates a new BaseElement with the specified components and mounts it 
        in the owning InnerSpace.

        Args:
            element_id: The desired unique ID for the new element.
            name: The human-readable name for the new element.
            component_configs: A list of component configurations, where each dict is:
                               { "type": "ComponentName", "config": { ... } }

        Returns:
            A dictionary result: { "success": bool, "result": str, "error": Optional[str] }
        """
        inner_space = self._get_inner_space()
        if not inner_space:
            return { "success": False, "result": None, "error": "Cannot create element: Owner is not an InnerSpace." }

        # Validate element_id uniqueness within the InnerSpace?
        if inner_space.get_element_by_id(element_id) or inner_space.id == element_id:
            return { "success": False, "result": None, "error": f"Element ID '{element_id}' already exists in this InnerSpace." }

        logger.info(f"Attempting to create dynamic element: ID='{element_id}', Name='{name}'")
        try:
            # 1. Create the base element instance
            new_element = BaseElement(element_id=element_id, name=name)
            # Set parent space reference immediately for components needing it during init
            new_element.set_parent_space(inner_space)

            # 2. Add components based on config
            if not isinstance(component_configs, list):
                 raise ValueError("component_configs must be a list.")
                 
            for comp_config in component_configs:
                if not isinstance(comp_config, dict) or "type" not in comp_config:
                    logger.warning(f"Skipping invalid component definition: {comp_config}")
                    continue
                    
                comp_type_name = comp_config["type"]
                comp_args = comp_config.get("config", {}).copy()
                if not isinstance(comp_args, dict):
                    logger.warning(f"Invalid 'config' for {comp_type_name}, expected dict. Skipping.")
                    continue

                # Find component class using registry/lookup
                component_class = find_component_class(comp_type_name)
                if component_class: # find_component_class returns None if not found
                    try:
                        # Instantiate component, passing element and config
                        component_instance = component_class(element=new_element, **comp_args)
                        new_element.add_component(component_instance)
                        logger.debug(f"Added component '{comp_type_name}' to element {element_id}")
                    except Exception as comp_init_err:
                         # Clean up partially created element? Maybe not necessary.
                         raise ValueError(f"Error initializing component {comp_type_name}: {comp_init_err}") from comp_init_err
                else:
                    raise ValueError(f"Component type '{comp_type_name}' not found or invalid.")

            # 3. Mount the fully configured element in the InnerSpace
            # Using default MountType.INCLUSION
            mount_success = inner_space.mount_element(new_element)
            if not mount_success:
                 # This ideally shouldn't fail if ID was checked, but handle just in case.
                 raise RuntimeError(f"Failed to mount newly created element {element_id} into InnerSpace {inner_space.id}.")

            # 3.5 Finalize element setup (e.g., register local tools)
            new_element.finalize_setup()

            # 4. Record creation event on InnerSpace timeline
            timeline_comp = inner_space.get_timeline()
            if timeline_comp:
                 event_payload = {
                     "event_type": "element_created_dynamically",
                     "data": {
                         "factory_component_id": self.id,
                         "new_element_id": element_id,
                         "new_element_name": name,
                         "component_types": [c.get('type') for c in component_configs if c.get('type')]
                     }
                 }
                 timeline_comp.add_event_to_primary_timeline(event_payload)
            
            result_msg = f"Element '{name}' ({element_id}) created successfully with {len(new_element.get_components())} components."
            logger.info(result_msg)
            return { "success": True, "result": result_msg, "error": None }

        except Exception as e:
            error_msg = f"Failed to create element '{element_id}': {e}"
            logger.error(error_msg, exc_info=True)
            # TODO: Cleanup? Should the partially created element be removed if mounting failed?
            return { "success": False, "result": None, "error": error_msg }
            
    # TODO: Add handle_delete_element ? (Requires careful implementation) 