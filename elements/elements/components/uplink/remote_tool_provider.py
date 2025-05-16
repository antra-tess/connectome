import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from elements.elements.components.base_component import Component
from elements.elements.components.tool_provider import LLMToolDefinition # For type hinting
from elements.component_registry import register_component

if TYPE_CHECKING:
    from elements.elements.uplink import UplinkProxy
    from elements.elements.components.uplink.connection_component import UplinkConnectionComponent
    from elements.elements.inner_space import InnerSpace # For type hinting agent_id source

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
        self._cached_remote_tools: Optional[List[Dict[str, Any]]] = None
        self._cache_timestamp: Optional[float] = None
        # TODO: Add cache TTL configuration

    async def _fetch_remote_tool_definitions_from_connection(self) -> List[Dict[str, Any]]:
        """
        Fetches the raw tool definitions from the remote space via the UplinkConnectionComponent.
        This method will be responsible for caching later.
        """
        if not self.owner:
            logger.error(f"[{self.id}] Cannot fetch remote tools: No owner (UplinkProxy).")
            return []

        connection_comp: Optional['UplinkConnectionComponent'] = self.owner.get_component_by_type("UplinkConnectionComponent")
        if not connection_comp:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] UplinkConnectionComponent not found on owner. Cannot fetch remote tools.")
            return []

        if not connection_comp.is_connected():
            logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Not connected to remote space {self.owner.remote_space_id}. Cannot fetch tools.")
            return []

        try:
            # This method needs to be added to UplinkConnectionComponent
            remote_tools = await connection_comp.fetch_remote_public_tool_definitions()
            if remote_tools is None: # Explicit None check if fetch can fail that way
                 logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Fetch remote public tool definitions returned None.")
                 return []
            logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Fetched {len(remote_tools)} tool definitions from remote space {self.owner.remote_space_id}.")
            return remote_tools
        except Exception as e:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Error fetching remote tool definitions: {e}", exc_info=True)
            return []

    async def get_tools_for_llm(self) -> List[LLMToolDefinition]:
        """
        Gets tool definitions from the remote space, prefixes their names,
        and formats them for the LLM.
        """
        # TODO: Implement caching for remote_raw_tools
        remote_raw_tools = await self._fetch_remote_tool_definitions_from_connection()
        if not remote_raw_tools:
            return []

        llm_tools: List[LLMToolDefinition] = []
        for tool_info in remote_raw_tools:
            provider_element_id = tool_info.get("provider_element_id")
            tool_name = tool_info.get("tool_name")
            description = tool_info.get("description")
            parameters_schema = tool_info.get("parameters_schema")

            if not (provider_element_id and tool_name and description is not None and parameters_schema is not None):
                logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Skipping malformed remote tool definition: {tool_info}")
                continue

            prefixed_name = f"{self.owner.id}::{provider_element_id}::{tool_name}"
            
            llm_tools.append(LLMToolDefinition(
                name=prefixed_name,
                description=description,
                parameters_schema=parameters_schema
            ))
        logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Providing {len(llm_tools)} prefixed remote tools to LLM.")
        return llm_tools

    async def execute_tool(self, prefixed_tool_name: str, calling_context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes a tool request by dispatching it to the remote space.
        The `prefixed_tool_name` is expected to be in the format:
        'uplink_proxy_id::remote_provider_element_id::actual_tool_name'
        
        `calling_context` is expected to contain 'agent_id' if available.
        """
        calling_context = calling_context or {}
        agent_id = calling_context.get("agent_id", "unknown_agent_via_uplink") # Get agent_id from context
        agent_name = calling_context.get("agent_name", "Unknown Agent via Uplink") # NEW: Get agent_name

        logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Attempting to execute remote tool: {prefixed_tool_name} with args {kwargs} for agent {agent_id} ({agent_name})")

        try:
            parts = prefixed_tool_name.split('::', 2)
            if len(parts) != 3:
                raise ValueError("Prefixed tool name is not in the expected format 'uplink_id::remote_element_id::tool_name'")
            
            uplink_id_from_name, remote_target_element_id, actual_tool_name = parts

            if uplink_id_from_name != self.owner.id:
                # This should ideally not happen if routing by InnerSpace is correct
                logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Mismatched uplink ID in tool name! Expected '{self.owner.id}', got '{uplink_id_from_name}'. Aborting.")
                return {"success": False, "error": "Mismatched uplink ID in tool name."}

        except ValueError as e:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Error parsing prefixed tool name '{prefixed_tool_name}': {e}")
            return {"success": False, "error": f"Invalid prefixed tool name format: {e}"}

        connection_comp: Optional['UplinkConnectionComponent'] = self.owner.get_component_by_type("UplinkConnectionComponent")
        if not connection_comp:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] UplinkConnectionComponent not found. Cannot dispatch remote tool execution.")
            return {"success": False, "error": "UplinkConnectionComponent not found."}

        if not connection_comp.is_connected():
            logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Not connected to remote space {self.owner.remote_space_id}. Cannot execute remote tool.")
            return {"success": False, "error": "Not connected to remote space."}

        action_payload_for_remote = {
            "event_type": "action_request_for_remote",
            "source_uplink_id": self.owner.id,
            "source_agent_id": agent_id, 
            "source_agent_name": agent_name, # NEW: Add agent_name
            "remote_target_element_id": remote_target_element_id,
            "action_name": actual_tool_name,
            "action_parameters": kwargs 
        }

        logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Dispatching remote action: {action_payload_for_remote}")
        
        try:
            dispatch_success = await connection_comp.send_event_to_remote_space(action_payload_for_remote)
            if dispatch_success:
                logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Remote action '{actual_tool_name}' successfully dispatched to element '{remote_target_element_id}'.")
                return {"success": True, "status": "pending_remote_execution", "detail": "Action dispatched to remote space."}
            else:
                logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Failed to dispatch remote action '{actual_tool_name}' via UplinkConnectionComponent.")
                return {"success": False, "error": "Failed to dispatch action to remote space."}
        except Exception as e:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Exception during remote action dispatch: {e}", exc_info=True)
            return {"success": False, "error": f"Exception during remote action dispatch: {e}"}

    def list_tools(self) -> List[str]:
        """
        Returns a list of names of the tools this provider offers.
        This is for compatibility if something iterates tool providers and calls list_tools.
        The actual tool definitions for LLM are via get_tools_for_llm.
        """
        # This might need to be async if it has to fetch.
        # For now, assume it can use cached or previously fetched names.
        # This is a simplified version. A robust one would call/await get_tools_for_llm
        # and extract names, but that might be too heavy for a simple list_tools.
        if self._cached_remote_tools: # Assuming _cached_remote_tools stores the LLMToolDefinition like structures
            # Ensure _cached_remote_tools actually contains dicts with 'name' key from LLMToolDefinition
            # or adapt if it stores the raw dicts from _fetch_remote_tool_definitions_from_connection
            return [tool_def.get('name') for tool_def in self._cached_remote_tools if isinstance(tool_def, dict) and 'name' in tool_def]
        
        logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] list_tools called, but remote tools are not cached. May return empty or outdated list.")
        return []

    # TODO: Add methods for refreshing the cache (_cached_remote_tools)
    # async def refresh_remote_tools_cache(self):
    #     # Fetch raw definitions
    #     raw_tools = await self._fetch_remote_tool_definitions_from_connection()
    #     # Convert to LLMToolDefinition format for consistency if _cached_remote_tools stores that
    #     processed_tools_for_cache = []
    #     for tool_info in raw_tools:
    #         provider_element_id = tool_info.get("provider_element_id")
    #         tool_name = tool_info.get("tool_name")
    #         # ... (rest of the parsing from get_tools_for_llm)
    #         prefixed_name = f"{self.owner.id}::{provider_element_id}::{tool_name}"
    #         processed_tools_for_cache.append(LLMToolDefinition(
    #             name=prefixed_name,
    #             description=tool_info.get("description"),
    #             parameters_schema=tool_info.get("parameters_schema")
    #         ))
    #     self._cached_remote_tools = processed_tools_for_cache
    #     self._cache_timestamp = time.monotonic()


