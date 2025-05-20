"""
Uplink Manager Component
Manages UplinkProxy elements within an InnerSpace, allowing the agent to
discover, create, and manage connections to remote SharedSpaces.
"""
import logging
import re
from typing import Dict, Any, Optional, List

from ..base import Component, BaseElement
from elements.component_registry import register_component
from .tool_provider import ToolProviderComponent, ToolParameter # For tool registration
from .factory_component import ElementFactoryComponent # For creating uplinks

# Type checking imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..inner_space import InnerSpace

# NEW: Import SpaceRegistry directly
from elements.space_registry import SpaceRegistry

logger = logging.getLogger(__name__)

@register_component
class UplinkManagerComponent(Component):
    COMPONENT_TYPE = "UplinkManagerComponent"
    owner: Optional['InnerSpace'] # Type hint for owner

    def __init__(self, **kwargs): # Removed element: Optional[BaseElement] = None
        super().__init__(**kwargs) # Pass only **kwargs
        self._active_uplinks: Dict[str, str] = {} # remote_space_id -> uplink_element_id (mount_id)
        self._factory_component: Optional[ElementFactoryComponent] = None
        self._tool_provider_component: Optional[ToolProviderComponent] = None
        # Ensure owner is InnerSpace for type hinting and specific owner access
        # self.owner: Optional['InnerSpace'] = element if isinstance(element, BaseElement) else None # REMOVED

    def initialize(self, **kwargs) -> None:
        super().initialize(**kwargs)
        if self.owner:
            self._factory_component = self.owner.get_component_by_type(ElementFactoryComponent)
            if not self._factory_component:
                logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] ElementFactoryComponent not found on owner. Cannot create uplinks.")
            
            self._tool_provider_component = self.owner.get_component_by_type(ToolProviderComponent)
            if not self._tool_provider_component:
                logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] ToolProviderComponent not found on owner. Cannot register uplink management tools.")
            else:
                self._register_uplink_management_tools()
        else:
            logger.error(f"[{self.COMPONENT_TYPE}] Owner not set or not BaseElement. Cannot initialize properly.")

    def _generate_safe_id_string(self, base_string: str) -> str:
        """Generates a string suitable for use as an ID component from a base string."""
        s = re.sub(r'[^a-zA-Z0-9_\\-]', '', base_string.replace(' ', '_'))
        return s[:50] # Limit length

    def get_uplink_for_space(self, remote_space_id: str) -> Optional[BaseElement]:
        """
        Retrieves an active UplinkProxy element mounted on the owner InnerSpace
        that connects to the specified remote_space_id.
        """
        if not self.owner:
            logger.error(f"[{self.COMPONENT_TYPE}] Cannot get uplink: component has no owner.")
            return None

        uplink_element_id = self._active_uplinks.get(remote_space_id)
        if uplink_element_id:
            element = self.owner.get_mounted_element(uplink_element_id)
            if element and hasattr(element, 'remote_space_id') and element.remote_space_id == remote_space_id:
                return element
            else:
                logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Stale uplink entry for {remote_space_id}. Removing.")
                del self._active_uplinks[remote_space_id]

        # Fallback: Scan mounted elements if not in cache (e.g., if added manually or cache missed)
        for mount_id, element in self.owner.get_mounted_elements().items():
            if hasattr(element, 'remote_space_id') and element.remote_space_id == remote_space_id and hasattr(element, 'get_connection_component'):
                self._active_uplinks[remote_space_id] = mount_id # Cache it by mount_id
                logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Discovered and cached active uplink '{mount_id}' to {remote_space_id}.")
                return element
        return None

    def ensure_uplink_to_shared_space(
        self, 
        shared_space_id: str, 
        shared_space_name: Optional[str] = None, 
        shared_space_description: Optional[str] = None
    ) -> Optional[BaseElement]:
        """
        Ensures an UplinkProxy to the specified SharedSpace exists and is active on the owner InnerSpace.
        If not, creates one using the ElementFactoryComponent and "standard_uplink_proxy" prefab.
        """
        if not self.owner or not self._factory_component:
            logger.error(f"[{self.COMPONENT_TYPE}] Owner or ElementFactoryComponent not available. Cannot ensure uplink to {shared_space_id}.")
            return None

        existing_uplink = self.get_uplink_for_space(shared_space_id)
        if existing_uplink:
            logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Uplink to {shared_space_id} (Element: {existing_uplink.id}) already exists.")
            return existing_uplink

        logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] No active uplink for {shared_space_id}. Creating one.")

        # --- Get remote space info for the UplinkProxy constructor ---
        remote_space_metadata: Optional[Dict[str, Any]] = None
        
        # Get SpaceRegistry instance directly
        registry_to_use = SpaceRegistry.get_instance()
        
        if registry_to_use:
            remote_space_instance = registry_to_use.get_space(shared_space_id)
            if remote_space_instance and hasattr(remote_space_instance, 'get_space_metadata_for_uplink'):
                remote_space_metadata = remote_space_instance.get_space_metadata_for_uplink()
            elif remote_space_instance:
                remote_space_metadata = {
                    "space_id": remote_space_instance.id,
                    "name": remote_space_instance.name,
                    "description": remote_space_instance.description,
                    "element_type": remote_space_instance.__class__.__name__,
                    "adapter_id": getattr(remote_space_instance, 'adapter_id', None),
                    "external_conversation_id": getattr(remote_space_instance, 'external_conversation_id', None)
                }
                logger.warning(f"Remote space {shared_space_id} did not have get_space_metadata_for_uplink, using basic attributes.")
            else:
                logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Could not retrieve instance of remote space {shared_space_id} from registry.")
        else:
            # This case should be rare if SpaceRegistry.get_instance() is robust
            logger.error(f"[{self.owner.id if self.owner else 'NoOwner'}/{self.COMPONENT_TYPE}] SpaceRegistry.get_instance() returned None. Cannot get remote_space_info.")
        # --- End get remote space info ---

        safe_remote_id_part = self._generate_safe_id_string(shared_space_id)
        agent_id_part = self._generate_safe_id_string(self.owner.agent_id) if hasattr(self.owner, 'agent_id') else "unknown_agent"
        new_uplink_element_id = f"uplink_{agent_id_part}_to_{safe_remote_id_part}"

        uplink_name = f"Uplink: {shared_space_name or shared_space_id}"
        uplink_description = f"Connects to SharedSpace: {shared_space_description or shared_space_id}"
        
        element_config = {
            "remote_space_id": shared_space_id,
            "name": uplink_name,
            "description": uplink_description,
            "remote_space_info": remote_space_metadata,
            "notify_owner_of_new_deltas_callback": self.owner.handle_deltas_from_uplink
        }

        try:
            creation_result = self._factory_component.handle_create_element_from_prefab(
                element_id=new_uplink_element_id,
                prefab_name="standard_uplink_proxy",
                element_config=element_config
            )

            if creation_result and creation_result.get("success"):
                created_uplink_id = creation_result.get("element_id", new_uplink_element_id)
                new_uplink_element = self.owner.get_mounted_element(created_uplink_id)
                new_uplink_element._connection_component.connect()
                if new_uplink_element:
                    self._active_uplinks[shared_space_id] = created_uplink_id # Cache by mount_id
                    logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Successfully created UplinkProxy '{new_uplink_element.name}' ({created_uplink_id}) to {shared_space_id}.")
                    return new_uplink_element
                else:
                    logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Factory created uplink to {shared_space_id}, but could not retrieve mounted element '{created_uplink_id}'.")
            else:
                error_msg = creation_result.get("error", "Unknown error") if creation_result else "Factory returned None"
                logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Failed to create UplinkProxy to {shared_space_id}: {error_msg}")
            return None
        except Exception as e:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Exception creating UplinkProxy for {shared_space_id}: {e}", exc_info=True)
            return None

    def _register_uplink_management_tools(self):
        if not self._tool_provider_component:
            return

        list_uplinks_params: List[ToolParameter] = []
        ensure_uplink_params: List[ToolParameter] = [
            {"name": "shared_space_id", "type": "string", "description": "The unique ID of the SharedSpace to connect to.", "required": True},
            {"name": "shared_space_name", "type": "string", "description": "Optional name for the SharedSpace (used if creating the uplink).", "required": False},
            {"name": "shared_space_description", "type": "string", "description": "Optional description for the SharedSpace.", "required": False}
        ]

        self._tool_provider_component.register_tool_function(
            name="list_active_uplinks",
            description="Lists all currently active uplinks from this InnerSpace to remote SharedSpaces.",
            parameters_schema=list_uplinks_params,
            tool_func=self.list_active_uplinks_tool
        )
        self._tool_provider_component.register_tool_function(
            name="ensure_uplink_to_space",
            description="Ensures an uplink connection exists to a specified SharedSpace. Creates one if necessary.",
            parameters_schema=ensure_uplink_params,
            tool_func=self.ensure_uplink_to_space_tool
        )

    def list_active_uplinks_tool(self) -> Dict[str, Any]:
        """Tool function to list active uplinks."""
        if not self.owner:
             return {"success": False, "error": "Component has no owner."}

        uplinks_info = []
        # Refresh active_uplinks from currently mounted elements
        current_mounted_uplinks = {}
        for mount_id, element in self.owner.get_mounted_elements().items():
            if hasattr(element, 'remote_space_id') and hasattr(element, 'get_connection_component'): # Basic check for UplinkProxy
                remote_id = getattr(element, 'remote_space_id')
                current_mounted_uplinks[remote_id] = mount_id
                uplinks_info.append({
                    "uplink_element_id": mount_id,
                    "remote_space_id": remote_id,
                    "remote_space_name": getattr(element, 'name', 'Unknown'), # UplinkProxy name
                    # Could add connection status if UplinkProxy provides a simple getter
                })
        self._active_uplinks = current_mounted_uplinks # Update cache
        return {"success": True, "active_uplinks": uplinks_info}

    def ensure_uplink_to_space_tool(self, shared_space_id: str, shared_space_name: Optional[str] = None, shared_space_description: Optional[str] = None) -> Dict[str, Any]:
        """Tool function to ensure an uplink exists."""
        uplink_element = self.ensure_uplink_to_shared_space(shared_space_id, shared_space_name, shared_space_description)
        if uplink_element:
            return {"success": True, "uplink_element_id": uplink_element.id, "remote_space_id": shared_space_id, "status": f"Uplink to {shared_space_id} ensured."}
        else:
            return {"success": False, "error": f"Failed to ensure uplink to {shared_space_id}."}
