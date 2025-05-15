"""
Element Factory Component
Provides tools for an agent to dynamically create and configure new elements
within its InnerSpace.
"""
import logging
from typing import Dict, Any, Optional, List, Type
import inspect

from ..base import BaseElement, MountType
from .base_component import Component # Import BaseComponent for type check

# Use the new component registry
from elements.component_registry import COMPONENT_REGISTRY, find_component_class, register_component

# Need access to parent Space to mount elements
from ..space import Space 

# Import PREFABS from the dedicated module
from elements.prefabs import PREFABS

# --- For dynamic element class instantiation ---
from ..uplink import UplinkProxy # Will be used by name lookup
# Add other element types here if they can be created by prefab

logger = logging.getLogger(__name__)

# A simple lookup for element classes by name
# This can be expanded or made more sophisticated (e.g., using component_registry pattern for elements)
ELEMENT_CLASS_LOOKUP = {
    "BaseElement": BaseElement,
    "Space": Space,
    "UplinkProxy": UplinkProxy,
    # Add InnerSpace if it can be created this way (less common for agent to create its own InnerSpace)
}

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
        from ..inner_space import InnerSpace # Specifically expect InnerSpace
        """Initializes the component."""
        super().initialize(**kwargs)
        # Ensure owner is an InnerSpace for mounting capabilities
        if not isinstance(self.owner, InnerSpace):
            logger.error(f"ElementFactoryComponent must be owned by an InnerSpace, not {type(self.owner)}. Functionality will be limited.")
        # Ensure component registry is populated (assuming scan was called at startup)
        if not COMPONENT_REGISTRY:
             logger.error("Component Registry is empty! ElementFactory cannot function. Was scan_and_load_components called?")
        logger.debug(f"ElementFactoryComponent initialized for Element {self.owner.id if self.owner else 'Unknown'}")

    def _get_inner_space(self) -> Optional[Any]:
        from ..inner_space import InnerSpace # Specifically expect InnerSpace
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
        # 'name' will be part of element_config now
        # name: str,
        element_config: Optional[Dict[str, Any]] = None, # New arg for element-level constructor/config
        component_config_overrides: Optional[Dict[str, Dict[str, Any]]] = None, # Renamed for clarity
        mount_id_override: Optional[str] = None # NEW: For specifying a custom mount_id
    ) -> Dict[str, Any]:
        """
        Logic for the 'create_element_from_prefab' tool.
        Creates an element using a predefined prefab template.

        Args:
            prefab_name: The name of the prefab to use (see list_available_prefabs).
            element_id: The desired unique ID for the new element.
            element_config: Optional. Dictionary of configuration values for the element itself,
                            used for its constructor arguments (e.g., {"name": "My Uplink", "remote_space_id": "xyz"}).
            component_config_overrides: Optional. A dictionary where keys are component type names
                                      and values are dictionaries of config settings to 
                                      override or provide for that component.
                                      E.g., {"MessageListComponent": {"channel_id": "12345"}}
            mount_id_override: Optional. If provided, this ID will be used to mount the element
                               in the parent InnerSpace. Defaults to element_id.

        Returns:
            A dictionary result: { "success": bool, "result": str, "error": Optional[str] }
        """
        element_config = element_config or {}
        component_config_overrides = component_config_overrides or {}
        
        # 1. Find prefab
        prefab_data = PREFABS.get(prefab_name)
        if not prefab_data:
            return { "success": False, "result": None, "error": f"Prefab '{prefab_name}' not found." }

        # --- Element Class and Constructor Args --- 
        element_class_name = prefab_data.get("element_class_name", "BaseElement")
        element_class = ELEMENT_CLASS_LOOKUP.get(element_class_name)
        if not element_class:
            return { "success": False, "result": None, "error": f"Element class '{element_class_name}' specified in prefab not found."}

        # Check for required element_config keys
        required_element_keys = prefab_data.get("required_element_config_keys", [])
        missing_element_keys = []
        for req_key in required_element_keys:
            if req_key not in element_config or element_config[req_key] is None:
                missing_element_keys.append(req_key)
        if missing_element_keys:
            error_msg = f"Missing required keys in element_config for prefab '{prefab_name}': {', '.join(missing_element_keys)}"
            return { "success": False, "result": None, "error": error_msg }

        # Prepare element constructor arguments
        element_constructor_args = {"element_id": element_id} # element_id is always passed
        # Add other args specified by prefab's element_constructor_arg_keys from element_config
        constructor_arg_keys_from_prefab = prefab_data.get("element_constructor_arg_keys", [])
        for key in constructor_arg_keys_from_prefab:
            if key in element_config:
                element_constructor_args[key] = element_config[key]
            # else: # Do not error if not present, constructor might have defaults
            #    logger.warning(f"Constructor key '{key}' for {element_class_name} not found in element_config, using default if any.")
        
        # --- Component Configurations (similar to before) ---
        final_component_configs = []
        prefab_components = prefab_data.get("components", [])
        required_component_configs = prefab_data.get("required_configs", {}) # For component-specific required keys
        missing_requirements = []

        for base_comp_config in prefab_components:
            comp_type_name = base_comp_config.get("type")
            if not comp_type_name:
                 logger.warning(f"Prefab '{prefab_name}' has invalid component entry: {base_comp_config}. Skipping.")
                 continue
            
            final_config = base_comp_config.get("config", {}).copy()
            
            if comp_type_name in component_config_overrides:
                override = component_config_overrides[comp_type_name]
                if isinstance(override, dict):
                    final_config.update(override)
                else:
                    logger.warning(f"Invalid override format for {comp_type_name} in prefab '{prefab_name}'. Expected dict, got {type(override)}. Ignoring override.")
            
            if comp_type_name in required_component_configs:
                for req_key in required_component_configs[comp_type_name]:
                    if req_key not in final_config or final_config[req_key] is None:
                        missing_requirements.append(f"Component '{comp_type_name}' requires config key '{req_key}'.")
                        
            final_component_configs.append({"type": comp_type_name, "config": final_config})
            
        if missing_requirements:
            error_msg = f"Missing required configurations for prefab '{prefab_name}' components: {'; '.join(missing_requirements)}"
            logger.error(error_msg)
            return { "success": False, "result": None, "error": error_msg }

        # --- Create and Mount Element (using modified logic) ---
        inner_space = self._get_inner_space()
        if not inner_space:
            return { "success": False, "result": None, "error": "Cannot create element: Owner is not an InnerSpace." }

        if inner_space.get_element_by_id(element_id) or inner_space.id == element_id:
            return { "success": False, "result": None, "error": f"Element ID '{element_id}' already exists in this InnerSpace." }

        logger.info(f"Attempting to create element '{element_constructor_args.get('name', element_id)}' (ID: {element_id}) from prefab '{prefab_name}' of type {element_class_name}.")
        
        try:
            # 1. Instantiate the specific element class
            # Pass SpaceRegistry if the element class constructor accepts it (like UplinkProxy)
            if 'space_registry' in inspect.signature(element_class.__init__).parameters and hasattr(inner_space, '_space_registry'):
                element_constructor_args['space_registry'] = inner_space._space_registry
            
            new_element = element_class(**element_constructor_args)
            new_element._set_parent(inner_space.id, MountType.INCLUSION) # Corrected: Use _set_parent and provide MountType
            new_element.set_registry(inner_space._space_registry)

            # 2. Add components (from prefab, with overrides applied)
            if not isinstance(final_component_configs, list):
                 raise ValueError("final_component_configs must be a list.")
                 
            for comp_spec in final_component_configs:
                comp_type_name = comp_spec["type"]
                comp_args = comp_spec.get("config", {}).copy()
                
                # If component already added by element constructor (e.g. UplinkProxy adds its own core ones),
                # this would either re-add (if add_component allows) or could be skipped.
                # For now, assume add_component handles this or we only list non-default components in prefab.
                existing_comp = new_element.get_component_by_type(comp_type_name)
                if existing_comp:
                    logger.debug(f"Component of type '{comp_type_name}' already exists on element {element_id}. Skipping addition from prefab.")
                    # Optionally, could try to reconfigure it: if hasattr(existing_comp, 'configure'): existing_comp.configure(**comp_args)
                    continue

                component_class_to_add = find_component_class(comp_type_name)
                if component_class_to_add:
                    # Inject dependencies if MessageActionHandler (or others in future)
                    if hasattr(component_class_to_add, 'INJECTED_DEPENDENCIES') and hasattr(inner_space, '_outgoing_action_callback'):
                        if 'outgoing_action_callback' in component_class_to_add.INJECTED_DEPENDENCIES:
                            comp_args['outgoing_action_callback'] = inner_space._outgoing_action_callback
                    
                    # Instantiate component, passing element and config
                    # The component's __init__ should accept 'element' as first arg if it needs it.
                    # Most of our components take element_id, name in constructor, but add_component in BaseElement
                    # handles wrapping and setting the owner. Here, we directly instantiate.
                    # Let's assume component constructors are flexible or primarily use kwargs from config.
                    # A safer way is new_element.add_component(component_class_to_add, **comp_args) if add_component takes type and kwargs
                    
                    # Simplification: Assume add_component on BaseElement can take type and config args
                    added_comp_instance = new_element.add_component(component_class_to_add, **comp_args)
                    if added_comp_instance:
                        logger.debug(f"Added component '{comp_type_name}' to element {element_id} via prefab.")
                    else:
                        # add_component should log its own error if it failed
                        raise ValueError(f"Failed to add component '{comp_type_name}' to element {element_id} via prefab.")
                else:
                    raise ValueError(f"Component type '{comp_type_name}' for prefab not found or invalid.")

            # 3. Mount the fully configured element in the InnerSpace
            actual_mount_id_to_use = mount_id_override if mount_id_override else new_element.id
            mount_success, final_mount_id = inner_space.mount_element(new_element, mount_id=actual_mount_id_to_use) # Pass actual_mount_id_to_use
            if not mount_success:
                 # This ideally shouldn't fail if ID was checked, but handle just in case.
                 # mount_element in ContainerComponent logs its own errors.
                 raise RuntimeError(f"Failed to mount newly created element {element_id} (attempted mount_id: {actual_mount_id_to_use}) into InnerSpace {inner_space.id}. Mount ID reported by mount: {final_mount_id}")

            # 3.5 Finalize element setup (e.g., register local tools)
            new_element.finalize_setup() # Ensure this method exists on BaseElement/subclasses

            # 4. Record creation event on InnerSpace timeline (similar to handle_create_element)
            event_payload = {
                "event_type": "element_created_from_prefab",
                "data": {
                    "factory_component_id": self.id,
                    "prefab_name": prefab_name,
                    "new_element_id": element_id,
                    "new_element_name": new_element.name,
                    "mount_id": final_mount_id, # Report the actual mount_id used
                    "element_class": new_element.__class__.__name__,
                    "component_types": [c.COMPONENT_TYPE for c in new_element.get_components().values()]
                }
            }
            inner_space.add_event_to_primary_timeline(event_payload)
            
            # --- NEW: Set element attributes from element_config based on prefab hint ---
            attributes_to_set_map = prefab_data.get("element_attributes_from_config", {})
            if attributes_to_set_map:
                logger.debug(f"Setting attributes on element {new_element.id} from element_config based on prefab hints: {attributes_to_set_map}")
                for config_key, attribute_name in attributes_to_set_map.items():
                    if config_key in element_config:
                        value_to_set = element_config[config_key]
                        try:
                            setattr(new_element, attribute_name, value_to_set)
                            logger.debug(f"Set attribute '{attribute_name}' on element {new_element.id} to: {value_to_set}")
                        except Exception as e:
                            logger.error(f"Error setting attribute '{attribute_name}' on element {new_element.id} from config key '{config_key}': {e}", exc_info=True)
                    else:
                        logger.warning(f"Prefab '{prefab_name}' specified setting attribute '{attribute_name}' from config key '{config_key}', but key was not found in element_config.")
            # --- END NEW --- 

            result_msg = f"Element '{new_element.name}' ({element_id}) created successfully from prefab '{prefab_name}' with {len(new_element.get_components())} components."
            logger.info(result_msg)
            return { "success": True, "result": result_msg, "error": None, "element": new_element}

        except Exception as e:
            error_msg = f"Failed to create element '{element_id}' from prefab '{prefab_name}': {e}"
            logger.error(error_msg, exc_info=True)
            return { "success": False, "result": None, "error": error_msg }

    def handle_create_element(self, element_id: str, name: str, component_configs: List[Dict[str, Any]], description: str = "") -> Dict[str, Any]:
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
            new_element = BaseElement(element_id=element_id, name=name, description=description) # Added description
            # Set parent space reference immediately for components needing it during init
            new_element._set_parent(inner_space.id, MountType.INCLUSION) # Corrected: Use _set_parent and provide MountType

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
            mount_success, _ = inner_space.mount_element(new_element)
            if not mount_success:
                 # This ideally shouldn't fail if ID was checked, but handle just in case.
                 raise RuntimeError(f"Failed to mount newly created element {element_id} into InnerSpace {inner_space.id}.")

            # 3.5 Finalize element setup (e.g., register local tools)
            new_element.finalize_setup()

            # 4. Record creation event on InnerSpace timeline
            event_payload = {
                "event_type": "element_created_dynamically",
                "data": {
                         "factory_component_id": self.id,
                         "new_element_id": element_id,
                         "new_element_name": name,
                         "component_types": [c.get('type') for c in component_configs if c.get('type')]
                     }
                 }
            inner_space.add_event_to_primary_timeline(event_payload)
            
            result_msg = f"Element '{name}' ({element_id}) created successfully with {len(new_element.get_components())} components."
            logger.info(result_msg)
            return { "success": True, "result": result_msg, "error": None }

        except Exception as e:
            error_msg = f"Failed to create element '{element_id}': {e}"
            logger.error(error_msg, exc_info=True)
            # TODO: Cleanup? Should the partially created element be removed if mounting failed?
            return { "success": False, "result": None, "error": error_msg }
            
    # TODO: Add handle_delete_element ? (Requires careful implementation) 