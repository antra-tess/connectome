import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import asyncio
import time # For cache timestamp

from elements.elements.components.base_component import Component
from elements.elements.components.tool_provider import LLMToolDefinition, ToolParameter, ToolProviderComponent
from elements.component_registry import register_component

if TYPE_CHECKING:
    from elements.elements.uplink import UplinkProxy
    from elements.elements.components.uplink.connection_component import UplinkConnectionComponent # For type hinting agent_id source
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
        self._tools_registered_with_local_provider = False
        self._local_tool_provider_on_owner: Optional[ToolProviderComponent] = None
        
        # NEW: Ultra-short chat prefix mapping for agent understanding
        self._chat_prefix_registry: Dict[str, Dict[str, str]] = {}  # {short_prefix: {element_id, display_name}}
        self._next_chat_prefix_index = 1

    def initialize(self, **kwargs):
        # Synchronous part of initialization - need local ToolProvider for proxying
        if self.owner and hasattr(self.owner, 'get_component_by_type'):
            self._local_tool_provider_on_owner = self.owner.get_component_by_type(ToolProviderComponent)
            if not self._local_tool_provider_on_owner:
                logger.warning(f"[{self.owner.id if self.owner else 'UnknownOwner'}-{self.id}] Could not find ToolProviderComponent on owner. Cannot register remote tools locally.")
        else:
            logger.warning(f"[{self.id}] Has no owner or owner lacks get_component_by_type. Cannot get local tool provider.")
        
        logger.debug(f"[{self.owner.id if self.owner else 'UnknownOwner'}-{self.id}] UplinkRemoteToolProviderComponent initialized.")
        super().initialize(**kwargs)
        # Tools will be fetched and registered on demand

    async def _ensure_tools_fetched_and_registered(self, force_refresh: bool = False) -> None:
        """Ensures tools are fetched from remote and registered locally with original names for proxying."""
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
                        self._register_tools_locally_with_original_names()
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

        # Register tools locally with ORIGINAL names for proxying
        if self._raw_remote_tool_definitions and not self._tools_registered_with_local_provider:
            self._register_tools_locally_with_original_names()

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

    def _convert_json_schema_to_tool_parameters(self, json_schema: Dict[str, Any]) -> List[ToolParameter]:
        """Converts a JSON schema (from LLMToolDefinition.parameters) to List[ToolParameter]."""
        tool_params: List[ToolParameter] = []
        properties = json_schema.get("properties", {})
        required_list = json_schema.get("required", [])
        for prop_name, prop_details in properties.items():
            param = ToolParameter(
                name=prop_name,
                description=prop_details.get("description", ""),
                type=prop_details.get("type", "string"), # Default to string if type missing
                required=prop_name in required_list
            )
            # Add optional fields like 'items', 'properties' (for nested objects), 'enum' if present
            if "items" in prop_details:
                param["items"] = prop_details["items"]
            if "properties" in prop_details: # If the parameter itself is an object
                param["properties"] = prop_details["properties"]
            if "enum" in prop_details:
                param["enum"] = prop_details["enum"]
            tool_params.append(param)
        return tool_params

    async def get_llm_tool_definitions(self) -> List[Dict[str, Any]]:
        await self._ensure_tools_fetched_and_registered()
        llm_tools: List[Dict[str, Any]] = []
        if not self._raw_remote_tool_definitions:
            logger.warning(f"[{self.owner.id if self.owner else 'NoOwner'}/{self.COMPONENT_TYPE}] No raw remote tools to convert to LLM definitions.")
            return []

        # Group tools by provider element to create chat prefixes
        tools_by_element: Dict[str, List[Dict[str, Any]]] = {}
        for tool_info in self._raw_remote_tool_definitions:
            provider_element_id = tool_info.get("provider_element_id")
            if provider_element_id:
                if provider_element_id not in tools_by_element:
                    tools_by_element[provider_element_id] = []
                tools_by_element[provider_element_id].append(tool_info)

        for provider_element_id, tools_list in tools_by_element.items():
            # Create ultra-short chat prefix for this element
            first_tool = tools_list[0]  # Get info from first tool
            chat_prefix = self._create_ultra_short_chat_prefix(
                element_id=provider_element_id,
                element_info={"name": provider_element_id}  # Could be enhanced with more info
            )
            
            # Add tools with two-level prefixing: chat_prefix__tool_name
            for tool_info in tools_list:
                tool_name = tool_info.get("tool_name")
                description = tool_info.get("description")
                json_parameters_schema = tool_info.get("parameters_schema")
                
                if not (tool_name and description is not None and json_parameters_schema is not None):
                    logger.warning(f"Skipping malformed tool_info for LLM: {tool_info}")
                    continue
                
                # Create two-level tool name: chat_prefix__tool_name
                prefixed_tool_name = f"{chat_prefix}__{tool_name}"
                
                llm_tools.append({
                    "name": prefixed_tool_name,  # e.g., "c1__send_message"
                    "description": f"[{self._chat_prefix_registry[chat_prefix]['display_name']}] {description}",
                    "parameters": json_parameters_schema
                })
        
        logger.debug(f"[{self.owner.id if self.owner else 'NoOwner'}/{self.COMPONENT_TYPE}] Providing {len(llm_tools)} tools with ultra-short chat prefixes.")
        if self._chat_prefix_registry:
            logger.debug(f"Chat prefix mappings: {', '.join(f'{p}={info[\"display_name\"]}' for p, info in self._chat_prefix_registry.items())}")
        
        return llm_tools

    async def execute_tool_by_elements(self, remote_target_element_id: str, action_name: str, calling_context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by specifying the remote target element and action directly.
        This is the new simpler execution path that doesn't require parsing complex tool names.
        """
        from elements.elements.inner_space import InnerSpace
        if not self.owner or not hasattr(self.owner, 'get_connection_component') or not hasattr(self.owner, 'remote_space_id') or not hasattr(self.owner, 'id'):
            return {"success": False, "error": "Owner or connection missing attributes."}
        
        logger.info(f"[{self.owner.id}-{self.id}] Executing remote tool: '{action_name}' on '{remote_target_element_id}', params: {kwargs}")
        
        connection_component = self.owner.get_connection_component()
        if not connection_component:
            return {"success": False, "error": "UplinkConnectionComponent not found."}
        if not connection_component.is_connected:
            return {"success": False, "error": "Not connected to remote space."}

        # Extract source_agent_id and source_agent_name from calling_context if available
        source_agent_id_from_context = None
        source_agent_name_from_context = None
        if calling_context:
            source_agent_id_from_context = calling_context.get('source_agent_id')
            source_agent_name_from_context = calling_context.get('source_agent_name')

        action_payload_for_remote = {
            "event_type": "action_request_for_remote",
            "source_uplink_id": self.owner.id,
            "source_agent_id": source_agent_id_from_context, 
            "source_agent_name": source_agent_name_from_context,
            "remote_target_element_id": remote_target_element_id,
            "action_name": action_name,
            "action_parameters": kwargs 
        }
        try:
            dispatch_success = await connection_component.send_event_to_remote_space(action_payload_for_remote)
            if dispatch_success:
                logger.info(f"Remote action '{action_name}' dispatched to '{remote_target_element_id}'.")
                return {"success": True, "status": "pending_remote_execution"}
            else:
                return {"success": False, "error": "Failed to dispatch action to remote."}
        except Exception as e:
            logger.error(f"Exception during dispatch: {e}", exc_info=True)
            return {"success": False, "error": f"Exception during dispatch: {e}"}

    async def execute_tool(self, prefixed_tool_name: str, calling_context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute tool method that handles two-level prefixing: chat_prefix__tool_name.
        AgentLoop passes the prefixed tool name after stripping the uplink prefix.
        """
        logger.info(f"[{self.owner.id}-{self.id}] execute_tool called with: '{prefixed_tool_name}', params: {kwargs}")
        
        # Parse two-level format: chat_prefix__tool_name
        if "__" not in prefixed_tool_name:
            return {"success": False, "error": f"Tool name '{prefixed_tool_name}' missing chat prefix. Expected format: 'chat_prefix__tool_name'"}
        
        parts = prefixed_tool_name.split("__", 1)
        chat_prefix = parts[0]
        actual_tool_name = parts[1]
        
        # Resolve chat prefix to element ID
        target_element_id = self._resolve_chat_prefix_to_element_id(chat_prefix)
        if not target_element_id:
            return {"success": False, "error": f"Unknown chat prefix '{chat_prefix}'. Available: {list(self._chat_prefix_registry.keys())}"}
        
        # Verify the tool exists for this element
        tool_found = False
        for tool_def in self._raw_remote_tool_definitions:
            if (tool_def.get("provider_element_id") == target_element_id and 
                tool_def.get("tool_name") == actual_tool_name):
                tool_found = True
                break
        
        if not tool_found:
            return {"success": False, "error": f"Tool '{actual_tool_name}' not found for chat prefix '{chat_prefix}' (element: {target_element_id})"}
        
        # Execute using the new method
        return await self.execute_tool_by_elements(
            remote_target_element_id=target_element_id,
            action_name=actual_tool_name,
            calling_context=calling_context,
            **kwargs
        )

    def list_tools(self) -> List[str]:
        if self._raw_remote_tool_definitions:
            # Return two-level prefixed tool names: chat_prefix__tool_name
            tools = []
            for tool_def in self._raw_remote_tool_definitions:
                provider_element_id = tool_def.get("provider_element_id")
                tool_name = tool_def.get("tool_name")
                if provider_element_id and tool_name:
                    # Find the chat prefix for this element
                    chat_prefix = None
                    for prefix, info in self._chat_prefix_registry.items():
                        if info["element_id"] == provider_element_id:
                            chat_prefix = prefix
                            break
                    
                    if chat_prefix:
                        tools.append(f"{chat_prefix}__{tool_name}")
                    else:
                        # Fallback if prefix not found (shouldn't happen)
                        tools.append(tool_name)
            return tools
        return []

    async def force_refresh_tools(self) -> bool:
        """Public method to explicitly refresh the tool cache and re-register."""
        logger.info(f"[{self.owner.id}-{self.id}] Force refreshing remote tools.")
        await self._ensure_tools_fetched_and_registered(force_refresh=True)
        return True # Indicate completion of attempt

    def _register_tools_locally_with_original_names(self):
        """Register remote tools locally with two-level naming: chat_prefix__tool_name."""
        if not self._local_tool_provider_on_owner or not self.owner:
            logger.warning(f"[{self.owner.id if self.owner else 'NoOwner'}-{self.id}] Local tool provider or owner missing. Skipping local registration.")
            return
        if not self._raw_remote_tool_definitions:
            logger.info(f"[{self.owner.id}-{self.id}] No raw remote tool definitions to register locally.")
            return

        logger.debug(f"[{self.owner.id}-{self.id}] Registering {len(self._raw_remote_tool_definitions)} remote tools locally with two-level naming.")
        registered_count = 0
        
        # Group tools by element to use consistent chat prefixes
        tools_by_element: Dict[str, List[Dict[str, Any]]] = {}
        for tool_def in self._raw_remote_tool_definitions:
            provider_element_id = tool_def.get("provider_element_id")
            if provider_element_id:
                if provider_element_id not in tools_by_element:
                    tools_by_element[provider_element_id] = []
                tools_by_element[provider_element_id].append(tool_def)
        
        for provider_element_id, tools_list in tools_by_element.items():
            # Get or create chat prefix for this element
            chat_prefix = self._create_ultra_short_chat_prefix(
                element_id=provider_element_id,
                element_info={"name": provider_element_id}
            )
            
            for tool_def in tools_list:
                tool_name_on_remote = tool_def.get("tool_name")
                description = tool_def.get("description")
                json_parameters_schema = tool_def.get("parameters_schema")
                
                if not (tool_name_on_remote and description is not None and json_parameters_schema is not None):
                    logger.warning(f"Skipping malformed tool_def for local registration: {tool_def}")
                    continue
                
                # Convert JSON schema to ToolParameter list
                list_tool_parameters = self._convert_json_schema_to_tool_parameters(json_parameters_schema)

                # Create executor that captures the correct tool info
                async def tool_executor(calling_context: Optional[Dict[str, Any]] = None, 
                                        # Capture the correct info for this specific executor
                                        captured_provider_element_id=provider_element_id,
                                        captured_tool_name=tool_name_on_remote,
                                        **kwargs):
                    # Use the captured info when calling the actual execution logic
                    return await self.execute_tool_by_elements(
                        remote_target_element_id=captured_provider_element_id,
                        action_name=captured_tool_name,
                        calling_context=calling_context, 
                        **kwargs
                    )

                try:
                    # Register with two-level naming: chat_prefix__tool_name
                    local_tool_name = f"{chat_prefix}__{tool_name_on_remote}"
                    self._local_tool_provider_on_owner.register_tool_function(
                        name=local_tool_name,
                        description=f"[{self._chat_prefix_registry[chat_prefix]['display_name']}] {description}",
                        parameters_schema=list_tool_parameters,
                        tool_func=tool_executor
                    )
                    registered_count += 1
                    logger.debug(f"Registered remote tool '{local_tool_name}' locally for proxying")
                    
                except Exception as e:
                    logger.error(f"[{self.owner.id}-{self.id}] Error during local tool registration: {e}", exc_info=True)
        
        if registered_count > 0:
            self._tools_registered_with_local_provider = True
            logger.info(f"[{self.owner.id}-{self.id}] Successfully registered {registered_count} remote tools locally with two-level naming.")
        else:
            self._tools_registered_with_local_provider = False
            logger.warning(f"[{self.owner.id}-{self.id}] Had raw tool definitions but none were registered locally.")

    def _create_ultra_short_chat_prefix(self, element_id: str, element_info: Dict[str, Any]) -> str:
        """
        Create an ultra-short prefix for a chat element and register the mapping.
        
        Args:
            element_id: Full element ID like "chat_elem_shared_zulip_adapter_1148/channel events_236c5545"
            element_info: Info about the element for display purposes
            
        Returns:
            Ultra-short prefix like "c1", "c2", etc.
        """
        # Check if we already have a prefix for this element
        for prefix, info in self._chat_prefix_registry.items():
            if info["element_id"] == element_id:
                return prefix
        
        # Create new ultra-short prefix
        short_prefix = f"c{self._next_chat_prefix_index}"
        self._next_chat_prefix_index += 1
        
        # Extract display name from element info or element_id
        display_name = element_info.get("name", element_id)
        if display_name.startswith("chat_elem_"):
            # Clean up the display name
            display_name = display_name.replace("chat_elem_", "").replace("_", " ")
        
        # Register the mapping
        self._chat_prefix_registry[short_prefix] = {
            "element_id": element_id,
            "display_name": display_name
        }
        
        logger.debug(f"[{self.owner.id}-{self.id}] Created chat prefix '{short_prefix}' for '{display_name}' ({element_id})")
        return short_prefix

    def _resolve_chat_prefix_to_element_id(self, chat_prefix: str) -> Optional[str]:
        """
        Resolve ultra-short chat prefix back to element ID.
        
        Args:
            chat_prefix: Ultra-short prefix like "c1"
            
        Returns:
            Full element ID or None if not found
        """
        info = self._chat_prefix_registry.get(chat_prefix)
        if info:
            return info["element_id"]
        return None

    def get_chat_prefix_mappings(self) -> Dict[str, str]:
        """
        Get chat prefix mappings for agent understanding.
        
        Returns:
            Dict mapping short prefixes to display names
        """
        return {prefix: info["display_name"] for prefix, info in self._chat_prefix_registry.items()}


