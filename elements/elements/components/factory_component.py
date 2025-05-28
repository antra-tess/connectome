"""
Element Factory Component
Provides tools for an agent or a Space to dynamically create and configure new elements.
"""
import logging
import inspect
from typing import Dict, Any, Optional, List, Type, Callable

from ..base import BaseElement, MountType
from .base_component import Component
from elements.component_registry import COMPONENT_REGISTRY, find_component_class, register_component
from elements.prefabs import PREFABS
from elements.space_registry import SpaceRegistry # For get_instance()

logger = logging.getLogger(__name__)

# --- Element Class Lookup ---
ELEMENT_CLASS_LOOKUP: Dict[str, Type[BaseElement]] = {
    "BaseElement": BaseElement,
    # Other classes will be added by _populate_element_class_lookup
}

_ELEMENT_CLASSES_POPULATED = False


def _populate_element_class_lookup():
    """
    Dynamically imports and adds element classes to ELEMENT_CLASS_LOOKUP
    to avoid circular dependencies at module load time.
    """
    global _ELEMENT_CLASSES_POPULATED
    if _ELEMENT_CLASSES_POPULATED:
        return

    from ..space import Space # Dynamically import Space
    from ..uplink import UplinkProxy # Dynamically import UplinkProxy
    ELEMENT_CLASS_LOOKUP["Space"] = Space
    ELEMENT_CLASS_LOOKUP["UplinkProxy"] = UplinkProxy
    _ELEMENT_CLASSES_POPULATED = True

@register_component
class ElementFactoryComponent(Component):
    """
    Provides tools for creating and managing elements dynamically.
    Should be attached to a Space element (e.g., InnerSpace or SharedSpace).
    """
    COMPONENT_TYPE = "ElementFactoryComponent"

    def __init__(self, element: Optional[BaseElement] = None, **config: Any):
        super().__init__(element=element, **config)
        self._owner_space: Optional["Space"] = None # Will be cast from self.owner
        self._outgoing_action_callback_for_created: Optional[Callable] = None

    def initialize(self, **kwargs) -> None:
        """Initializes the component."""
        from ..space import Space # Import here for type checking
        super().initialize(**kwargs)

        if not isinstance(self.owner, Space):
            logger.error(
                f"ElementFactoryComponent must be owned by a Space derivative (e.g., InnerSpace, SharedSpace), "
                f"not {type(self.owner)}. Functionality will be limited."
            )
            return # Cannot proceed without a Space owner

        self._owner_space = self.owner # Owner is confirmed to be a Space

        # Retrieve outgoing_action_callback from the owner Space
        if hasattr(self._owner_space, 'get_outgoing_action_callback'):
            self._outgoing_action_callback_for_created = self._owner_space.get_outgoing_action_callback()
            if not self._outgoing_action_callback_for_created:
                 logger.warning(f"ElementFactoryComponent on {self.owner.id} did not get an outgoing_action_callback from its owner.")
        elif hasattr(self._owner_space, '_outgoing_action_callback'): # Fallback for direct attribute access
             self._outgoing_action_callback_for_created = self._owner_space._outgoing_action_callback
             if not self._outgoing_action_callback_for_created:
                 logger.warning(f"ElementFactoryComponent on {self.owner.id} did not get an outgoing_action_callback from its owner (via _outgoing_action_callback).")
        else:
            logger.warning(
                f"Owner Space {self.owner.id} does not have 'get_outgoing_action_callback' or '_outgoing_action_callback'. "
                f"Created elements/components requiring it may not function correctly."
            )
        
        # Ensure element class lookup is populated
        _populate_element_class_lookup()
        if not COMPONENT_REGISTRY:
             logger.error("Component Registry is empty! ElementFactory cannot function. Was scan_and_load_components called?")
        
        logger.debug(f"ElementFactoryComponent initialized for Element {self.owner.id if self.owner else 'Unknown'}. Callback acquired: {'Yes' if self._outgoing_action_callback_for_created else 'No'}")

    def _get_owner_space(self) -> Optional["Space"]:
        """Helper to get the owner, ensuring it's a Space instance."""
        # self._owner_space is set during initialize after type check
        return self._owner_space

    def _prepare_constructor_args(self, target_class: Type, base_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares constructor arguments for a target class (Element or Component),
        injecting space_registry and outgoing_action_callback if accepted by the constructor.
        """
        final_args = base_args.copy()
        try:
            sig = inspect.signature(target_class.__init__)
            params = sig.parameters

            if 'space_registry' in params:
                final_args['space_registry'] = SpaceRegistry.get_instance()
            
            if 'outgoing_action_callback' in params and self._outgoing_action_callback_for_created:
                final_args['outgoing_action_callback'] = self._outgoing_action_callback_for_created
            
        except Exception as e:
            logger.warning(f"Could not inspect constructor for {target_class.__name__} to inject dependencies: {e}", exc_info=False)
        return final_args

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
        element_config: Optional[Dict[str, Any]] = None,
        component_config_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        mount_id_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Logic for the 'create_element_from_prefab' tool.
        Creates an element using a predefined prefab template.
        """
        element_config = element_config or {}
        component_config_overrides = component_config_overrides or {}
        
        owner_space = self._get_owner_space()
        if not owner_space:
            return { "success": False, "result": None, "error": "Cannot create element: Owner is not a valid Space or not initialized." }

        # 1. Find prefab
        prefab_data = PREFABS.get(prefab_name)
        if not prefab_data:
            return { "success": False, "result": None, "error": f"Prefab '{prefab_name}' not found." }

        # Element Class and Constructor Args
        element_class_name = prefab_data.get("element_class_name", "BaseElement")
        _populate_element_class_lookup() # Ensure it's populated if not already
        element_class = ELEMENT_CLASS_LOOKUP.get(element_class_name)
        if not element_class:
            return { "success": False, "result": None, "error": f"Element class '{element_class_name}' specified in prefab not found in lookup."}

        # Check for required element_config keys
        required_element_keys = prefab_data.get("required_element_config_keys", [])
        missing_element_keys = [req_key for req_key in required_element_keys if req_key not in element_config or element_config[req_key] is None]
        if missing_element_keys:
            error_msg = f"Missing required keys in element_config for prefab '{prefab_name}': {', '.join(missing_element_keys)}"
            return { "success": False, "result": None, "error": error_msg }

        # Prepare element constructor arguments
        base_element_constructor_args = {"element_id": element_id}
        constructor_arg_keys_from_prefab = prefab_data.get("element_constructor_arg_keys", [])
        for key in constructor_arg_keys_from_prefab:
            if key in element_config:
                base_element_constructor_args[key] = element_config[key]
        
        # Component Configurations
        final_component_configs = []
        prefab_components = prefab_data.get("components", [])
        required_component_configs = prefab_data.get("required_configs", {})
        missing_requirements = []

        for base_comp_config in prefab_components:
            comp_type_name = base_comp_config.get("type")
            if not comp_type_name:
                 logger.warning(f"Prefab '{prefab_name}' has invalid component entry: {base_comp_config}. Skipping.")
                 continue
            
            final_config = base_comp_config.get("config", {}).copy()
            if comp_type_name in component_config_overrides:
                override = component_config_overrides[comp_type_name]
                if isinstance(override, dict): final_config.update(override)
                else: logger.warning(f"Invalid override format for {comp_type_name} in prefab '{prefab_name}'. Ignoring.")
            
            if comp_type_name in required_component_configs:
                for req_key in required_component_configs[comp_type_name]:
                    if req_key not in final_config or final_config[req_key] is None:
                        missing_requirements.append(f"Component '{comp_type_name}' requires config key '{req_key}'.")
                        
            final_component_configs.append({"type": comp_type_name, "config": final_config})
            
        if missing_requirements:
            error_msg = f"Missing required configurations for prefab '{prefab_name}' components: {'; '.join(missing_requirements)}"
            return { "success": False, "result": None, "error": error_msg }

        # Create and Mount Element
        if owner_space.get_element_by_id(element_id) or owner_space.id == element_id: # Check owner_space
            return { "success": False, "result": None, "error": f"Element ID '{element_id}' already exists in owner space '{owner_space.id}'." }

        logger.info(f"Attempting to create element '{element_config.get('name', element_id)}' (ID: {element_id}) from prefab '{prefab_name}' of type {element_class_name} in space {owner_space.id}.")
        
        try:
            # 1. Instantiate the specific element class with injected dependencies
            actual_element_constructor_args = self._prepare_constructor_args(element_class, base_element_constructor_args)
            new_element = element_class(**actual_element_constructor_args)
            new_element._set_parent(owner_space.id, MountType.INCLUSION)

            # 2. Add components
            for comp_spec in final_component_configs:
                comp_type_name = comp_spec["type"]
                comp_args = comp_spec.get("config", {}).copy()
                
                existing_comp = new_element.get_component_by_type(comp_type_name)
                if existing_comp:
                    logger.debug(f"Component '{comp_type_name}' already on element {element_id}. Skipping from prefab.")
                    continue

                component_class_to_add = find_component_class(comp_type_name)
                if component_class_to_add:
                    actual_comp_args = self._prepare_constructor_args(component_class_to_add, comp_args)
                    # BaseElement.add_component takes component type and kwargs for its constructor
                    added_comp_instance = new_element.add_component(component_class_to_add, **actual_comp_args)
                    if added_comp_instance:
                        logger.debug(f"Added component '{comp_type_name}' to element {element_id} via prefab.")
                    else:
                        raise ValueError(f"Failed to add component '{comp_type_name}' to element {element_id} via prefab (add_component returned None).")
                else:
                    raise ValueError(f"Component type '{comp_type_name}' for prefab not found.")

            # 3. Mount the element in the owner_space
            actual_mount_id_to_use = mount_id_override if mount_id_override else new_element.id
            
            # Prepare creation data for replay
            creation_data = {
                'prefab_name': prefab_name,
                'element_config': element_config,
                'component_config_overrides': component_config_overrides,
                'mount_id_override': mount_id_override
            }
            
            mount_success, final_mount_id = owner_space.mount_element(new_element, mount_id=actual_mount_id_to_use, creation_data=creation_data)
            if not mount_success:
                 raise RuntimeError(f"Failed to mount element {element_id} (mount_id: {actual_mount_id_to_use}) into {owner_space.id}. Mount ID from call: {final_mount_id}")

            new_element.finalize_setup()

            # 4. Record creation event on owner_space timeline
            event_payload = {
                "event_type": "element_created_from_prefab",
                "data": {
                    "factory_component_id": self.id,
                    "prefab_name": prefab_name,
                    "new_element_id": element_id,
                    "new_element_name": new_element.name,
                    "mount_id": final_mount_id,
                    "element_class": new_element.__class__.__name__,
                    "component_types": [c.COMPONENT_TYPE for c in new_element.get_components().values()]
                }
            }
            owner_space.add_event_to_primary_timeline(event_payload)
            
            # Set element attributes from element_config based on prefab hint
            attributes_to_set_map = prefab_data.get("element_attributes_from_config", {})
            if attributes_to_set_map:
                logger.debug(f"Setting attributes on {new_element.id} from element_config: {attributes_to_set_map}")
                for config_key, attribute_name in attributes_to_set_map.items():
                    if config_key in element_config:
                        try:
                            setattr(new_element, attribute_name, element_config[config_key])
                            logger.debug(f"Set attribute '{attribute_name}' on {new_element.id} to: {element_config[config_key]}")
                        except Exception as e:
                            logger.error(f"Error setting attribute '{attribute_name}' on {new_element.id} from config '{config_key}': {e}")
                    # else: (No warning if key not present, attribute might be optional)
            
            result_msg = f"Element '{new_element.name}' ({element_id}) created from prefab '{prefab_name}' in space '{owner_space.id}'."
            logger.info(result_msg)
            return { "success": True, "result": result_msg, "error": None, "element_id": new_element.id, "element": new_element }

        except Exception as e:
            error_msg = f"Failed to create element '{element_id}' from prefab '{prefab_name}' in space '{owner_space.id}': {e}"
            logger.error(error_msg, exc_info=True)
            return { "success": False, "result": None, "error": error_msg }

    def handle_create_element(
        self, 
        element_id: str, 
        name: str, 
        component_configs: List[Dict[str, Any]], 
        description: str = "",
        element_class_name: str = "BaseElement" # Allow specifying class, defaults to BaseElement
    ) -> Dict[str, Any]:
        """
        Logic for the 'create_element' tool.
        Creates a new element with specified components and mounts it in the owning Space.
        """
        owner_space = self._get_owner_space()
        if not owner_space:
            return { "success": False, "result": None, "error": "Cannot create element: Owner is not a valid Space or not initialized." }

        if owner_space.get_element_by_id(element_id) or owner_space.id == element_id:
            return { "success": False, "result": None, "error": f"Element ID '{element_id}' already exists in space '{owner_space.id}'." }

        _populate_element_class_lookup() # Ensure it's populated
        element_class_to_create = ELEMENT_CLASS_LOOKUP.get(element_class_name)
        if not element_class_to_create:
            return { "success": False, "result": None, "error": f"Element class '{element_class_name}' not found in lookup."}

        logger.info(f"Attempting to create dynamic element: ID='{element_id}', Name='{name}', Class='{element_class_name}' in space '{owner_space.id}'.")
        try:
            # 1. Create the element instance with injected dependencies
            base_element_args = {"element_id": element_id, "name": name, "description": description}
            actual_element_args = self._prepare_constructor_args(element_class_to_create, base_element_args)
            new_element = element_class_to_create(**actual_element_args)
            
            new_element._set_parent(owner_space.id, MountType.INCLUSION)

            # 2. Add components
            if not isinstance(component_configs, list):
                 raise ValueError("component_configs must be a list.")
                 
            for comp_config in component_configs:
                if not isinstance(comp_config, dict) or "type" not in comp_config:
                    logger.warning(f"Skipping invalid component definition: {comp_config}")
                    continue
                    
                comp_type_name = comp_config["type"]
                comp_args = comp_config.get("config", {}).copy() # Ensure it's a dict

                component_class = find_component_class(comp_type_name)
                if component_class:
                    actual_comp_args = self._prepare_constructor_args(component_class, comp_args)
                    # add_component on BaseElement can take type and constructor kwargs
                    added_comp = new_element.add_component(component_class, **actual_comp_args)
                    if not added_comp:
                         raise ValueError(f"Failed to add component '{comp_type_name}' to element {element_id} (add_component returned None).")
                    logger.debug(f"Added component '{comp_type_name}' to element {element_id}")
                else:
                    raise ValueError(f"Component type '{comp_type_name}' not found.")

            # 3. Mount in the owner_space
            creation_data = {
                'element_class_name': element_class_name,
                'component_configs': component_configs,
                'dynamic_creation': True
            }
            
            mount_success, final_mount_id = owner_space.mount_element(new_element, creation_data=creation_data) # Uses element_id as mount_id by default
            if not mount_success:
                 raise RuntimeError(f"Failed to mount element {element_id} into {owner_space.id}. Mount ID: {final_mount_id}")

            new_element.finalize_setup()

            # 4. Record creation event on owner_space timeline
            event_payload = {
                "event_type": "element_created_dynamically",
                "data": {
                    "factory_component_id": self.id,
                    "new_element_id": element_id,
                    "new_element_name": name,
                    "element_class": new_element.__class__.__name__,
                    "mount_id": final_mount_id,
                    "component_types": [c.get('type') for c in component_configs if isinstance(c, dict) and c.get('type')]
                }
            }
            owner_space.add_event_to_primary_timeline(event_payload)
            
            result_msg = f"Element '{name}' ({element_id}) created dynamically in space '{owner_space.id}' with {len(new_element.get_components())} components."
            logger.info(result_msg)
            return { "success": True, "result": result_msg, "error": None, "element_id": new_element.id, "element": new_element}

        except Exception as e:
            error_msg = f"Failed to create element '{element_id}' in space '{owner_space.id}': {e}"
            logger.error(error_msg, exc_info=True)
            return { "success": False, "result": None, "error": error_msg }

    # TODO: Add handle_delete_element ? (Requires careful implementation regarding ownership and cleanup)
    # def handle_delete_element(self, element_id_to_delete: str) -> Dict[str, Any]: ...
