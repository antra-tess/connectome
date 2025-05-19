import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import asyncio
import time # For cache timestamp

from elements.elements.components.base_component import Component
from elements.elements.components.tool_provider import LLMToolDefinition, ToolProviderComponent
from elements.component_registry import register_component

if TYPE_CHECKING:
    from elements.elements.uplink import UplinkProxy
    from elements.elements.components.uplink.connection_component import UplinkConnectionComponent
    from elements.elements.inner_space import InnerSpace # For type hinting agent_id source
    from elements.elements.components.tool_provider import ToolProviderComponent

logger = logging.getLogger(__name__)

@register_component
class UplinkRemoteToolProviderComponent(Component):
    """
    Provides tools to an InnerSpace that are actually hosted on a remote SharedSpace,
    accessed via an UplinkProxy. It fetches tool definitions from the remote space,
    prefixes their names, and handles dispatching their execution back to the remote space.
    """
    COMPONENT_TYPE = "UplinkRemoteToolProviderComponent"

    def __init__(self, element: Optional['UplinkProxy'] = None, **kwargs):
        super().__init__(element, **kwargs)
        self._raw_remote_tool_definitions: List[Dict[str, Any]] = []
        self._cache_timestamp: Optional[float] = None
        self._sync_in_progress = False
        # TODO: Consider making cache_duration configurable
        self._cache_duration_seconds = 300 # Cache tools for 5 minutes
        self._local_tool_provider_on_owner: Optional[ToolProviderComponent] = None
        self._default_remote_chat_tool_name: Optional[str] = None
        self._tools_registered_with_local_provider = False

    def initialize(self, **kwargs):
        # Synchronous part of initialization
        if self.owner and hasattr(self.owner, 'get_component_by_type'):
            self._local_tool_provider_on_owner = self.owner.get_component_by_type(ToolProviderComponent)
            if not self._local_tool_provider_on_owner:
                logger.warning(f"[{self.owner.id if self.owner else 'UnknownOwner'}-{self.id}] Could not find ToolProviderComponent on owner. Cannot register remote tools.")
        else:
            logger.warning(f"[{self.id}] Has no owner or owner lacks get_component_by_type. Cannot get local tool provider.")
        
        # Call super().initialize() if it might be async or does important things
        # If Component.initialize is not async, this await is not strictly needed for super, but harmless.
        super().initialize(**kwargs)
        logger.debug(f"[{self.owner.id if self.owner else 'UnknownOwner'}-{self.id}] UplinkRemoteToolProviderComponent initialized.")
        # Do not fetch tools here; will be fetched on demand by get_llm_tool_definitions

    async def _ensure_tools_fetched_and_registered(self, force_refresh: bool = False) -> None:
        """Ensures tools are fetched from remote and registered with the local provider."""
        now = time.monotonic()
        cache_expired = self._cache_timestamp is None or (now - self._cache_timestamp > self._cache_duration_seconds)

        if not self._raw_remote_tool_definitions or force_refresh or cache_expired:
            if self._sync_in_progress:
                logger.info(f"[{self.owner.id}-{self.id}] Sync already in progress. Waiting for it to complete before re-fetching.")
                # Basic re-entrancy guard; could use a Lock for more robust async protection
                while self._sync_in_progress:
                    await asyncio.sleep(0.1)
                # After sync completes, re-check cache status as it might have been updated
                now = time.monotonic()
                cache_expired = self._cache_timestamp is None or (now - self._cache_timestamp > self._cache_duration_seconds)
                if not force_refresh and not cache_expired and self._raw_remote_tool_definitions:
                    logger.info(f"[{self.owner.id}-{self.id}] Tools were fetched by another task. Using fresh cache.")
                    if not self._tools_registered_with_local_provider:
                        self._register_tools_with_local_provider()
                    return
            
            self._sync_in_progress = True
            try:
                logger.info(f"[{self.owner.id}-{self.id}] Fetching/refreshing remote tool definitions. Force: {force_refresh}, Cache Expired: {cache_expired}")
                remote_tools = await self._fetch_remote_tool_definitions_from_connection()
                if remote_tools is not None: # Explicit check for None, as empty list is valid
                    self._raw_remote_tool_definitions = remote_tools
                    self._cache_timestamp = now
                    self._tools_registered_with_local_provider = False # Mark for re-registration
                    logger.info(f"[{self.owner.id}-{self.id}] Remote tool definitions updated. Found {len(self._raw_remote_tool_definitions)} tools.")
                else:
                    logger.warning(f"[{self.owner.id}-{self.id}] Fetching remote tools returned None. Retaining potentially stale cache.")
            except Exception as e:
                logger.error(f"[{self.owner.id}-{self.id}] Error during tool definition fetch: {e}", exc_info=True)
            finally:
                self._sync_in_progress = False
        
        if self._raw_remote_tool_definitions and not self._tools_registered_with_local_provider:
            self._register_tools_with_local_provider()

    async def _fetch_remote_tool_definitions_from_connection(self) -> Optional[List[Dict[str, Any]]]:
        if not self.owner:
            logger.error(f"[{self.id}] Cannot fetch: No owner.")
            return None
        connection_comp: Optional['UplinkConnectionComponent'] = self.get_sibling_component("UplinkConnectionComponent")
        if not connection_comp:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] ConnectionComponent not found.")
            return None
        if not connection_comp.is_connected:
            logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Not connected to {self.owner.remote_space_id}. Cannot fetch tools.")
            return None # Return None to indicate fetch was attempted but failed due to no connection
        try:
            remote_tools = await connection_comp.fetch_remote_public_tool_definitions()
            if remote_tools is None:
                 logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Fetch returned None.")
                 return None
            logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Fetched {len(remote_tools)} tools from {self.owner.remote_space_id}.")
            return remote_tools
        except Exception as e:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Error fetching: {e}", exc_info=True)
            return None # Indicate error

    def _register_tools_with_local_provider(self):
        if not self._local_tool_provider_on_owner or not self.owner:
            logger.warning(f"[{self.owner.id if self.owner else 'NoOwner'}-{self.id}] Local tool provider or owner missing. Skipping registration.")
            return
        if not self._raw_remote_tool_definitions:
            logger.info(f"[{self.owner.id}-{self.id}] No raw remote tool definitions to register.")
            return

        logger.debug(f"[{self.owner.id}-{self.id}] Registering {len(self._raw_remote_tool_definitions)} remote tools with local provider.")
        self._default_remote_chat_tool_name = None
        registered_count = 0
        for tool_def in self._raw_remote_tool_definitions:
            provider_element_id = tool_def.get("provider_element_id")
            tool_name_on_remote = tool_def.get("tool_name")
            description = tool_def.get("description")
            parameters_schema = tool_def.get("parameters_schema")
            if not (provider_element_id and tool_name_on_remote and description is not None and parameters_schema is not None):
                logger.warning(f"Skipping malformed tool_def: {tool_def}")
                continue
            prefixed_tool_name = f"{self.owner.id}::{provider_element_id}::{tool_name_on_remote}"
            async def tool_executor(calling_context: Optional[Dict[str, Any]] = None, **kwargs):
                return await self.execute_tool(prefixed_tool_name=prefixed_tool_name, calling_context=calling_context, **kwargs)
            self._local_tool_provider_on_owner.register_tool_function(
                name=prefixed_tool_name, description=description,
                parameters_schema=parameters_schema, tool_func=tool_executor,
                # is_async=True
            )
            registered_count += 1
            if tool_name_on_remote == "send_message" and "chat" in provider_element_id.lower() and not self._default_remote_chat_tool_name:
                self._default_remote_chat_tool_name = prefixed_tool_name
                logger.info(f"Default remote chat tool: '{prefixed_tool_name}'.")
        
        if registered_count > 0:
            self._tools_registered_with_local_provider = True
            logger.info(f"[{self.owner.id}-{self.id}] Successfully registered {registered_count} remote tools locally.")
        elif self._raw_remote_tool_definitions: # Had definitions but none were valid for registration
            self._tools_registered_with_local_provider = False # Explicitly false
            logger.warning(f"[{self.owner.id}-{self.id}] Had raw tool definitions but none were registered.")
        else: # No raw definitions to begin with
            self._tools_registered_with_local_provider = False

    async def get_llm_tool_definitions(self) -> List[Dict[str, Any]]:
        await self._ensure_tools_fetched_and_registered()
        llm_tools: List[Dict[str, Any]] = []
        if not self._raw_remote_tool_definitions:
            logger.warning(f"[{self.owner.id if self.owner else 'NoOwner'}/{self.COMPONENT_TYPE}] No raw remote tools to convert to LLM definitions.")
            return []

        for tool_info in self._raw_remote_tool_definitions:
            provider_element_id = tool_info.get("provider_element_id")
            tool_name = tool_info.get("tool_name")
            description = tool_info.get("description")
            parameters_schema = tool_info.get("parameters_schema")
            if not (provider_element_id and tool_name and description is not None and parameters_schema is not None):
                logger.warning(f"Skipping malformed tool_info for LLM: {tool_info}")
                continue
            prefixed_name = f"{self.owner.id}::{provider_element_id}::{tool_name}"
            llm_tools.append({
                "name": prefixed_name, "description": description,
                "parameters_schema": parameters_schema
            })
        logger.debug(f"[{self.owner.id if self.owner else 'NoOwner'}/{self.COMPONENT_TYPE}] Providing {len(llm_tools)} prefixed remote tools to LLM.")
        return llm_tools

    async def execute_tool(self, prefixed_tool_name: str, calling_context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        if not self.owner or not hasattr(self.owner, 'get_connection_component') or not hasattr(self.owner, 'remote_space_id') or not hasattr(self.owner, 'id'):
            return {"success": False, "error": "Owner or connection missing attributes."}
        logger.info(f"[{self.owner.id}-{self.id}] Executing tool: '{prefixed_tool_name}', params: {kwargs}")
        parsed_remote_target_element_id, parsed_action_name_on_remote = None, None
        if "::" in prefixed_tool_name:
            parts = prefixed_tool_name.split("::")
            if len(parts) == 3:
                owner_uplink_id, remote_target_element_id, action_name_on_remote = parts
                if owner_uplink_id != self.owner.id:
                    return {"success": False, "error": f"Mismatched uplink ID: expected {self.owner.id}, got {owner_uplink_id}"}
                parsed_remote_target_element_id = remote_target_element_id
                parsed_action_name_on_remote = action_name_on_remote
            else:
                return {"success": False, "error": f"Invalid prefixed tool name format: {prefixed_tool_name}"}
        else:
            return {"success": False, "error": f"Tool name '{prefixed_tool_name}' not remote format."}
        if not parsed_remote_target_element_id or not parsed_action_name_on_remote:
            return {"success": False, "error": "Failed to parse remote target/action from tool name."}
        connection_component = self.owner.get_connection_component()
        if not connection_component:
            return {"success": False, "error": "UplinkConnectionComponent not found."}
        if not connection_component.is_connected:
            return {"success": False, "error": "Not connected to remote space."}
        
        # Extract source_agent_id and source_agent_name from calling_context if available
        source_agent_id_from_context = None
        source_agent_name_from_context = None
        if calling_context:
            source_agent_id_from_context = calling_context.get('source_agent_id') # Check InnerSpace.execute_element_action
            source_agent_name_from_context = calling_context.get('source_agent_name')
        
        # Fallback to owner InnerSpace's agent_id if not in context (though it should be for agent actions)
        if not source_agent_id_from_context and isinstance(self.owner, TYPE_CHECKING.InnerSpace):
             source_agent_id_from_context = self.owner.agent_id
        if not source_agent_name_from_context and isinstance(self.owner, TYPE_CHECKING.InnerSpace):
             source_agent_name_from_context = self.owner.agent_name

        action_payload_for_remote = {
            "event_type": "action_request_for_remote",
            "source_uplink_id": self.owner.id,
            "source_agent_id": source_agent_id_from_context, 
            "source_agent_name": source_agent_name_from_context,
            "remote_target_element_id": parsed_remote_target_element_id,
            "action_name": parsed_action_name_on_remote,
            "action_parameters": kwargs 
        }
        logger.debug(f"Dispatching remote action: {action_payload_for_remote}")
        try:
            dispatch_success = await connection_component.send_event_to_remote_space(action_payload_for_remote)
            if dispatch_success:
                logger.info(f"Remote action '{parsed_action_name_on_remote}' dispatched to '{parsed_remote_target_element_id}'.")
                return {"success": True, "status": "pending_remote_execution"}
            else:
                return {"success": False, "error": "Failed to dispatch action to remote."}
        except Exception as e:
            logger.error(f"Exception during dispatch: {e}", exc_info=True)
            return {"success": False, "error": f"Exception during dispatch: {e}"}

    def list_tools(self) -> List[str]:
        if self._raw_remote_tool_definitions:
            return [f"{self.owner.id}::{td.get('provider_element_id')}::{td.get('tool_name')}" for td in self._raw_remote_tool_definitions if td.get('provider_element_id') and td.get('tool_name')]
        return []

    async def force_refresh_tools(self) -> bool:
        """Public method to explicitly refresh the tool cache and re-register."""
        logger.info(f"[{self.owner.id}-{self.id}] Force refreshing remote tools.")
        await self._ensure_tools_fetched_and_registered(force_refresh=True)
        return True # Indicate completion of attempt


