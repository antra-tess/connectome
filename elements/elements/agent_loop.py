"""
Agent Loop Components
Defines the base class and simple implementations for agent cognitive cycles.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, TYPE_CHECKING, List, Set, Type

from .base import Component
from elements.component_registry import register_component

# NEW: Import LLMMessage for correct provider interaction
from llm.provider_interface import LLMMessage
# NEW: Import LLMToolDefinition for passing tools
from llm.provider_interface import LLMToolDefinition, LLMToolCall, LLMResponse
from .components.tool_provider import ToolProviderComponent
from elements.elements.components.uplink.remote_tool_provider import UplinkRemoteToolProviderComponent

if TYPE_CHECKING:
    from .inner_space import InnerSpace
    from llm.provider_interface import LLMProvider
    from .components.hud.hud_component import HUDComponent # Assuming HUDComponent is in .components.hud
    from host.event_loop import OutgoingActionCallback


logger = logging.getLogger(__name__)

class BaseAgentLoopComponent(Component):
    """
    Abstract base class for agent loop components.
    Defines the interface for how the HostEventLoop triggers an agent's cognitive cycle.
    """
    COMPONENT_TYPE = "AgentLoopComponent"
    # Define dependencies that InnerSpace should inject during instantiation.
    # Key: kwarg name for __init__; Value: attribute name on InnerSpace instance or 'self' for InnerSpace itself.
    INJECTED_DEPENDENCIES = {
        'parent_inner_space': 'self' 
    }

    def __init__(self, parent_inner_space: 'InnerSpace', system_prompt_template: Optional[str] = None, agent_loop_name: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if not parent_inner_space:
            raise ValueError("BaseAgentLoopComponent requires a parent_inner_space instance.")
        self.parent_inner_space: 'InnerSpace' = parent_inner_space
        self._system_prompt_template = system_prompt_template or "You are AI assistant '{agent_name}'. Be helpful."
        
        # Use agent_loop_name for logging if provided, otherwise use component ID or a default
        self.agent_loop_name = agent_loop_name or f"{self.COMPONENT_TYPE}_{self.id[:8]}"

        # Convenience accessors, assuming parent_inner_space is correctly typed and populated
        self._llm_provider: Optional['LLMProvider'] = self.parent_inner_space._llm_provider
        self._hud_component: Optional['HUDComponent'] = self.parent_inner_space.get_hud()
        self._outgoing_action_callback: Optional['OutgoingActionCallback'] = self.parent_inner_space._outgoing_action_callback

        if not self._llm_provider:
            logger.error(f"{self.agent_loop_name} ({self.id}): LLMProvider not available from parent InnerSpace.")
        if not self._hud_component:
            logger.error(f"{self.agent_loop_name} ({self.id}): HUDComponent not available from parent InnerSpace.")
        # ToolProvider and outgoing_action_callback might be optional for some loops.

        logger.info(f"{self.COMPONENT_TYPE} '{self.agent_loop_name}' ({self.id}) initialized for InnerSpace '{self.parent_inner_space.name}'.")

    async def trigger_cycle(self):
        """
        This method is called by the HostEventLoop to initiate one cognitive cycle
        for the agent. Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement the trigger_cycle method.")

    def _get_hud(self) -> Optional['HUDComponent']:
        if not self._hud_component:
            self._hud_component = self.parent_inner_space.get_hud()
            if not self._hud_component:
                 logger.error(f"{self.agent_loop_name} ({self.id}): HUDComponent could not be retrieved on demand.")
        return self._hud_component

    def _get_llm_provider(self) -> Optional['LLMProvider']:
        if not self._llm_provider: # Should have been set in init
             logger.error(f"{self.agent_loop_name} ({self.id}): LLMProvider not available.")
        return self._llm_provider

    def _get_tool_provider(self) -> Optional['ToolProviderComponent']:
        # Ensure this helper can return None gracefully
        if not hasattr(self, '_tool_provider'): # Check if attribute exists
             self._tool_provider = self.get_sibling_component(ToolProviderComponent)
        return self._tool_provider
        
    def _get_outgoing_action_callback(self) -> Optional['OutgoingActionCallback']:
        if not self._outgoing_action_callback: # Should have been set in init via parent
            self._outgoing_action_callback = self.parent_inner_space._outgoing_action_callback
        return self._outgoing_action_callback

    async def aggregate_tools(self) -> List[LLMToolDefinition]:
        """
        Aggregates tools from:
        1. The InnerSpace itself (via its ToolProviderComponent).
        2. Mounted elements within the InnerSpace that have a ToolProviderComponent 
           (e.g., DMManagerComponent providing DM tools, UplinkProxies providing remote tools).
        """
        aggregated_tools_list: List[LLMToolDefinition] = []
        # Use a set to keep track of tool names to avoid duplicates if components offer same named tools
        # Note: This only de-duplicates if the LLMToolDefinition is hashable and considered equal
        # For now, we rely on unique naming or prefixes to differentiate.
        # A more robust de-duplication might be needed if tools from different sources have identical names/schemas.
        # For this iteration, we assume that if a tool name is the same, it's the same tool, or one overrides the other.
        # More sophisticated handling might be needed (e.g. prefixing tools from mounted elements)
        
        # 1. Tools from InnerSpace itself (e.g., tools for managing the agent, core tools)
        inner_space_tool_provider = self.parent_inner_space.get_tool_provider()
        if inner_space_tool_provider:
            for tool_def in inner_space_tool_provider.get_llm_tool_definitions():
                # Ensure it's LLMToolDefinition instance if not already
                if isinstance(tool_def, dict):
                    tool_def_obj = LLMToolDefinition(**tool_def)
                else:
                    tool_def_obj = tool_def # Assume it is already LLMToolDefinition
                aggregated_tools_list.append(tool_def_obj)
        
        # 2. Tools from mounted elements (including UplinkProxies, DM Dession elements, etc.)
        # This requires InnerSpace to have a way to get its mounted elements.
        # Assuming self.parent_inner_space.get_mounted_elements() exists.
        mounted_elements = self.parent_inner_space.get_mounted_elements()
        for mount_id, element in mounted_elements.items():
            # Check if the element has a standard ToolProviderComponent
            element_tool_provider = element.get_component_by_type(ToolProviderComponent)
            if element_tool_provider:
                for tool_def in element_tool_provider.get_llm_tool_definitions():
                    if isinstance(tool_def, dict):
                        tool_def_obj = LLMToolDefinition(**tool_def)
                    else:
                        tool_def_obj = tool_def
                    # Optional: Prefix tools from mounted elements if ambiguity is a concern
                    # tool_def_obj.name = f"{mount_id}::{tool_def_obj.name}"
                    aggregated_tools_list.append(tool_def_obj)
            
            # NEW: Check for UplinkRemoteToolProviderComponent specifically
            urtp_component = element.get_component_by_type(UplinkRemoteToolProviderComponent)
            if urtp_component:
                # This method is now async, so we await it
                remote_tool_dicts = await urtp_component.get_llm_tool_definitions()
                for tool_dict in remote_tool_dicts:
                    # urtp_component.get_tools_for_llm() returns List[Dict], convert to LLMToolDefinition
                    aggregated_tools_list.append(LLMToolDefinition(**tool_dict))

        # De-duplicate based on tool name (last one wins if names clash, consider warning)
        final_tools_dict: Dict[str, LLMToolDefinition] = {}
        for tool in aggregated_tools_list:
            if tool.name in final_tools_dict:
                logger.warning(f"Duplicate tool name '{tool.name}' found during aggregation. Overwriting.")
            final_tools_dict[tool.name] = tool
        
        final_tools_list = list(final_tools_dict.values())
        logger.info(f"[{self.agent_loop_name}] Aggregated {len(final_tools_list)} unique tools for LLM.")
        return final_tools_list

    async def _get_llm_response(self, current_context_messages: List[LLMMessage]) -> LLMResponse:
        """Gets response from LLM, including tool aggregation."""
        if not self._llm_provider:
            logger.error(f"[{self.agent_loop_name}] LLM provider not available.")
            return LLMResponse(content="Error: LLM provider not available.", finish_reason="error")

        # Aggregate tools asynchronously
        available_tools = await self.aggregate_tools()
        # ... rest of the method remains the same ...

@register_component
class SimpleRequestResponseLoopComponent(BaseAgentLoopComponent):
    """
    A basic agent loop that:
    1. Gets context from HUD.
    2. Gets available tools from InnerSpace ToolProvider.
    3. Sends context and tools to LLM.
    4. Checks LLM response for structured tool calls.
       - If tool calls exist, dispatches them.
    5. If no tool calls, processes text content via HUD and dispatches resulting actions.
    """
    COMPONENT_TYPE = "SimpleRequestResponseLoopComponent"

    def __init__(self, parent_inner_space: 'InnerSpace', system_prompt_template: Optional[str] = None, agent_loop_name: Optional[str] = None, **kwargs):
        super().__init__(parent_inner_space=parent_inner_space, system_prompt_template=system_prompt_template, agent_loop_name=agent_loop_name, **kwargs)
        # Additional initialization specific to this loop type, if any.

    async def trigger_cycle(self):
        logger.info(f"{self.agent_loop_name} ({self.id}): Cycle triggered in InnerSpace '{self.parent_inner_space.name}'.")

        hud = self._get_hud()
        llm_provider = self._get_llm_provider()

        if not hud:
            logger.error(f"{self.agent_loop_name} ({self.id}): HUDComponent not available. Aborting cycle.")
            return
        if not llm_provider:
            logger.error(f"{self.agent_loop_name} ({self.id}): LLMProvider not available. Aborting cycle.")
            return

        try:
            # 1. Prepare context using HUD
            logger.debug(f"{self.agent_loop_name} ({self.id}): Preparing LLM context via HUD...")
            render_options = {"render_style": "clean"}
            llm_context_string = await hud.get_agent_context(options=render_options)
            if not llm_context_string:
                logger.warning(f"{self.agent_loop_name} ({self.id}): HUD generated an empty context string. Aborting cycle.")
                return
            
            # Optional: Record context to timeline
            try:
                self.parent_inner_space.add_event_to_primary_timeline({
                    "event_type": "agent_context_generated",
                    "data": {
                        "loop_component_id": self.id,
                        "context_preview": llm_context_string[:250] + ('...' if len(llm_context_string) > 250 else ''),
                        "context_length": len(llm_context_string),
                        "render_options": render_options
                    }
                })
            except Exception as tl_err:
                logger.error(f"Error recording agent context to timeline: {tl_err}", exc_info=True)

            # 2. Aggregate Tools from InnerSpace and its Children
            aggregated_tools = await self.aggregate_tools()

            # 3. Send context and aggregated tools to LLM
            logger.debug(f"{self.agent_loop_name} ({self.id}): Sending context and tools to LLM...")
            
            # Format the system prompt
            formatted_system_prompt = self._system_prompt_template.format(agent_name=self.parent_inner_space.name)
            
            messages = [
                LLMMessage(role="system", content=formatted_system_prompt),
                LLMMessage(role="user", content=llm_context_string)
            ]
            llm_response_obj = llm_provider.complete(messages=messages, tools=aggregated_tools)
            
            if not llm_response_obj:
                logger.warning(f"{self.agent_loop_name} ({self.id}): LLM returned no response object. Aborting cycle.")
                return

            llm_response_text = llm_response_obj.content # May be None if tool call occurs
            logger.info(f"{self.agent_loop_name} ({self.id}): Received LLM response. Finish reason: {llm_response_obj.finish_reason}")
            if llm_response_text:
                 logger.debug(f"LLM Response Text Content: {llm_response_text[:100]}...")
            if llm_response_obj.tool_calls:
                 logger.debug(f"LLM Response Tool Calls: {llm_response_obj.tool_calls}")

            # 4. Process LLM Response: Check for Tool Calls first
            if llm_response_obj.tool_calls:
                logger.info(f"LLM requested {len(llm_response_obj.tool_calls)} tool call(s). Dispatching...")
                for tool_call in llm_response_obj.tool_calls:
                    if not isinstance(tool_call, LLMToolCall):
                        logger.warning(f"Skipping invalid item in tool_calls: {tool_call}")
                        continue
                    
                    # Parse prefixed tool name to find target element and actual tool name
                    raw_tool_name = tool_call.tool_name
                    target_element_id: str
                    actual_tool_name: str
                    
                    if "::" in raw_tool_name:
                        parts = raw_tool_name.split("::", 1)
                        target_element_id = parts[0]
                        actual_tool_name = parts[1]
                        # Validate target_element_id exists in InnerSpace? Optional.
                        if not self.parent_inner_space.get_element_by_id(target_element_id):
                             logger.error(f"LLM requested tool '{actual_tool_name}' on non-existent element '{target_element_id}'. Skipping.")
                             continue
                    else:
                        # Assume tool is on InnerSpace itself
                        target_element_id = self.parent_inner_space.id 
                        actual_tool_name = raw_tool_name
                    
                    target_space_id = self.parent_inner_space.id # Assume target space is always current InnerSpace for now
                    parameters = tool_call.parameters
                    
                    logger.debug(f"Dispatching structured tool call: Space='{target_space_id}', Element='{target_element_id}', Action='{actual_tool_name}', Params='{parameters}'")
                    try:
                        calling_context = { "loop_component_id": self.id } 
                        # AWAIT the call to execute_element_action
                        action_result = await self.parent_inner_space.execute_element_action(
                            space_id=target_space_id,
                            element_id=target_element_id, # Use parsed target_element_id
                            action_name=actual_tool_name,  # Use parsed actual_tool_name
                            parameters=parameters,
                            calling_context=calling_context
                        )
                        logger.debug(f"Structured tool call result for {actual_tool_name} on {target_element_id}: {action_result}")
                    except Exception as exec_err:
                        logger.error(f"Error executing structured tool call '{actual_tool_name}' on element '{target_element_id}': {exec_err}", exc_info=True)
            
            # 5. If NO tool calls, process text content via HUD for actions/response
            elif llm_response_text:
                logger.debug(f"{self.agent_loop_name} ({self.id}): No tool calls detected. Processing text response via HUD...")
                processed_actions = await hud.process_llm_response(llm_response_text)
                
                # --- Dispatch Actions from Text --- 
                if processed_actions and isinstance(processed_actions, list):
                    logger.info(f"{self.agent_loop_name} ({self.id}): Dispatching {len(processed_actions)} actions extracted by HUD from text.")
                    callback = self._get_outgoing_action_callback() # Get callback here
                    for action_request in processed_actions:
                        if not isinstance(action_request, dict) or not action_request.get("action_name"):
                            logger.warning(f"Invalid action format from HUD: {action_request}. Skipping.")
                            continue
                        
                        target_module = action_request.get("target_module")
                        target_element_id = action_request.get("target_element_id")
                        action_name = action_request.get("action_name")
                        parameters = action_request.get("parameters", {})
                        
                        # Dispatch External Actions
                        if target_module:
                            if callback:
                                logger.debug(f"Dispatching external action to module '{target_module}': {action_request}")
                                try: callback(action_request)
                                except Exception as cb_err: logger.error(f"Error calling outgoing callback for '{target_module}': {cb_err}", exc_info=True)
                            else: logger.error(f"Cannot dispatch external action to '{target_module}': No outgoing_action_callback. Action: {action_request}")
                        
                        # Dispatch Internal Actions from Text
                        elif target_element_id:
                            target_space_id = action_request.get("target_space_id", self.parent_inner_space.id)
                            logger.debug(f"Dispatching internal action from text: Space='{target_space_id}', Element='{target_element_id}', Action='{action_name}', Params='{parameters}'")
                            try:
                                calling_context = { "loop_component_id": self.id }
                                # AWAIT the call to execute_element_action
                                action_result = await self.parent_inner_space.execute_element_action(
                                    space_id=target_space_id, element_id=target_element_id,
                                    action_name=action_name, parameters=parameters,
                                    calling_context=calling_context
                                )
                                logger.debug(f"Internal action from text result for {action_name} on {target_element_id}: {action_result}")
                            except Exception as exec_err: logger.error(f"Error executing internal action from text '{action_name}' on '{target_element_id}': {exec_err}", exc_info=True)
                        else:
                            logger.warning(f"Could not dispatch text action: No target_module or target_element_id. Action: {action_request}")
                else:
                    logger.info(f"{self.agent_loop_name} ({self.id}): No actions extracted by HUD from text response.")
            else:
                 logger.info(f"{self.agent_loop_name} ({self.id}): LLM response had no tool calls and no text content.")

        except Exception as e:
            logger.error(f"{self.agent_loop_name} ({self.id}): Error during cognitive cycle: {e}", exc_info=True)
        finally:
            logger.info(f"{self.agent_loop_name} ({self.id}): Cycle finished.")


@register_component
class MultiStepToolLoopComponent(BaseAgentLoopComponent):
    """
    An agent loop designed for multi-step interactions, specifically handling 
    the cycle of: Context -> LLM -> Tool Call -> Tool Result -> LLM -> Final Response.

    This loop relies on analyzing the recent timeline history to determine the 
    current stage of the interaction and uses structured tool calling.
    """
    COMPONENT_TYPE = "MultiStepToolLoopComponent"

    # Define necessary event types (assuming these will be standardized)
    EVENT_TYPE_TOOL_ACTION_DISPATCHED = "tool_action_dispatched"
    EVENT_TYPE_TOOL_RESULT_RECEIVED = "tool_result_received"
    EVENT_TYPE_FINAL_ACTION_DISPATCHED = "final_action_dispatched"
    EVENT_TYPE_AGENT_CONTEXT_GENERATED = "agent_context_generated"
    EVENT_TYPE_LLM_RESPONSE_PROCESSED = "llm_response_processed"
    # New event for structured tool call dispatch
    EVENT_TYPE_STRUCTURED_TOOL_ACTION_DISPATCHED = "structured_tool_action_dispatched"

    def __init__(self, parent_inner_space: 'InnerSpace', system_prompt_template: Optional[str] = None, agent_loop_name: Optional[str] = None, **kwargs):
        super().__init__(parent_inner_space=parent_inner_space, system_prompt_template=system_prompt_template, agent_loop_name=agent_loop_name, **kwargs)
        logger.info(f"{self.COMPONENT_TYPE} '{self.agent_loop_name}' ({self.id}) initialized.")

    def _determine_stage(self, last_relevant_event: Optional[Dict[str, Any]]) -> str:
        """Analyzes the last event to determine the current stage."""
        if not last_relevant_event:
            return "initial_request" # No prior relevant event found
        
        # The event_node's "payload" key contains what the component originally logged.
        # This payload should directly have "event_type" and "data".
        event_payload = last_relevant_event.get("payload", {})
        event_type = event_payload.get("event_type")
        event_data = event_payload.get("data", {}) 

        if event_type == self.EVENT_TYPE_TOOL_RESULT_RECEIVED:
             return "tool_result_received" 
        elif event_type == self.EVENT_TYPE_STRUCTURED_TOOL_ACTION_DISPATCHED:
             return "waiting_for_tool_result"
        elif event_type == self.EVENT_TYPE_TOOL_ACTION_DISPATCHED: # Legacy/text-based dispatch
             # We might need to distinguish if this was the intended final step
             logger.warning(f"Handling legacy {self.EVENT_TYPE_TOOL_ACTION_DISPATCHED} event. Assuming waiting for result.")
             return "waiting_for_tool_result"
        elif event_type == self.EVENT_TYPE_LLM_RESPONSE_PROCESSED:
            # Check if the processed response indicated completion (no tool dispatched)
            if not event_data.get("dispatched_tool_action"): 
                 return "interaction_complete" # Final response was likely dispatched
            else:
                 # This case should ideally not be reached if tool dispatch events are used properly
                 logger.warning(f"LLM response processed event indicated tool dispatch, but no specific dispatch event found? Assuming waiting.")
                 return "waiting_for_tool_result"
        elif event_type == self.EVENT_TYPE_AGENT_CONTEXT_GENERATED:
             # If context was just generated, we're ready for the initial LLM call.
             return "initial_request"
        else:
             logger.warning(f"Unclear state from last event type: {event_type}. Defaulting to initial_request.")
             return "initial_request"

    async def trigger_cycle(self):
        logger.info(f"{self.agent_loop_name} ({self.id}): Multi-step cycle triggered in InnerSpace '{self.parent_inner_space.name}'.")

        hud = self._get_hud()
        llm_provider = self._get_llm_provider()
        callback = self._get_outgoing_action_callback()

        if not hud or not llm_provider or not self.parent_inner_space:
            logger.error(f"{self.agent_loop_name} ({self.id}): Missing critical components (HUD, LLM, Timeline). Aborting cycle.")
            return

        try:
            # --- 1. Analyze Recent History to Determine Current State ---
            relevant_event_types = [
                 self.EVENT_TYPE_TOOL_RESULT_RECEIVED,
                 self.EVENT_TYPE_STRUCTURED_TOOL_ACTION_DISPATCHED,
                 self.EVENT_TYPE_LLM_RESPONSE_PROCESSED,
                 self.EVENT_TYPE_AGENT_CONTEXT_GENERATED,
            ]
            # Corrected filter criteria to point to the actual data locations
            filter_criteria = {
                 "payload.data.loop_component_id": self.id, # Corrected: targets payload.data
                 "payload.event_type__in": relevant_event_types # Correct: targets payload.event_type
            }
            try:
                 # Ensure we get a timeline component instance before calling methods on it
                 timeline_comp = self.parent_inner_space.get_timeline()
                 if not timeline_comp:
                     logger.error(f"{self.agent_loop_name} ({self.id}): TimelineComponent not available. Cannot determine stage.")
                     return "initial_request" # Fallback or error state
                 last_relevant_event = timeline_comp.get_last_relevant_event(filter_criteria=filter_criteria)
            except Exception as query_err:
                 logger.error(f"Error querying timeline: {query_err}", exc_info=True)
                 last_relevant_event = None
                 
            current_stage = self._determine_stage(last_relevant_event)
            logger.debug(f"Determined current stage: {current_stage}")

            # --- 2. Execute Logic Based on Stage --- 

            # === STAGE: Initial Request or Processing Tool Result ===
            if current_stage == "initial_request" or current_stage == "tool_result_received":
                # Generate context
                render_options = {"render_style": "clean"} 
                context_string = await hud.get_agent_context(options=render_options)
                if not context_string: return
                
                # Record context generated
                try:
                     self.parent_inner_space.add_event_to_primary_timeline({
                         "event_type": self.EVENT_TYPE_AGENT_CONTEXT_GENERATED,
                         "data": {"loop_component_id": self.id, "stage": current_stage, "context_length": len(context_string)}
                     })
                except Exception as e: logger.error(f"Error recording context event: {e}")

                # --- Aggregate Tools (InnerSpace + Children) --- 
                aggregated_tools = await self.aggregate_tools()

                # Call LLM with aggregated tools
                logger.debug(f"Calling LLM for stage '{current_stage}'...")
                
                # Format the system prompt
                formatted_system_prompt = self._system_prompt_template.format(agent_name=self.parent_inner_space.name)
                
                messages = [
                    LLMMessage(role="system", content=formatted_system_prompt),
                    LLMMessage(role="user", content=context_string)
                ]
                llm_response_obj = llm_provider.complete(messages=messages, tools=aggregated_tools)
                if not llm_response_obj: logger.warning(f"LLM returned no response object."); return

                llm_response_text = llm_response_obj.content
                logger.info(f"LLM response received. Finish reason: {llm_response_obj.finish_reason}")

                # --- Process Response: Check for Tool Calls first --- 
                dispatched_tool_action = False
                if llm_response_obj.tool_calls:
                    logger.info(f"LLM requested {len(llm_response_obj.tool_calls)} tool call(s). Dispatching...")
                    for tool_call in llm_response_obj.tool_calls:
                        if not isinstance(tool_call, LLMToolCall): continue
                        
                        # Parse prefixed name
                        raw_tool_name = tool_call.tool_name
                        target_element_id: str
                        actual_tool_name: str
                        if "::" in raw_tool_name:
                            parts = raw_tool_name.split("::", 1)
                            target_element_id = parts[0]
                            actual_tool_name = parts[1]
                            if not self.parent_inner_space.get_element_by_id(target_element_id):
                                 logger.error(f"LLM tool target element '{target_element_id}' not found. Skipping call.")
                                 continue
                        else:
                            target_element_id = self.parent_inner_space.id
                            actual_tool_name = raw_tool_name
                        
                        target_space_id = self.parent_inner_space.id
                        parameters = tool_call.parameters
                        
                        logger.debug(f"Dispatching structured tool call: Element='{target_element_id}', Action='{actual_tool_name}'...")
                        try:
                            calling_context = { "loop_component_id": self.id } 
                            # AWAIT the call to execute_element_action
                            action_result = await self.parent_inner_space.execute_element_action(
                                space_id=target_space_id, 
                                element_id=target_element_id,
                                action_name=actual_tool_name, 
                                parameters=parameters,
                                calling_context=calling_context
                            )
                            # Record the structured dispatch event
                            self.parent_inner_space.add_event_to_primary_timeline({
                                "event_type": self.EVENT_TYPE_STRUCTURED_TOOL_ACTION_DISPATCHED,
                                "data": {"loop_component_id": self.id, "tool_call_name": raw_tool_name, "tool_call_params": parameters, "sync_result_preview": str(action_result)[:100]}
                            })
                            dispatched_tool_action = True
                        except Exception as exec_err: logger.error(f"Error executing structured tool call '{actual_tool_name}' on '{target_element_id}': {exec_err}", exc_info=True)

                # If NO tool calls were dispatched, process text response via HUD
                if not dispatched_tool_action:
                    if llm_response_text:
                        logger.debug("No tool calls. Processing text response via HUD...")
                        processed_actions = await hud.process_llm_response(llm_response_text)
                        final_action_requests = processed_actions
                        logger.info(f"Dispatching {len(final_action_requests)} final action(s) from text.")
                        for action_request in final_action_requests:
                             target_module = action_request.get("target_module")
                             if target_module and callback:
                                  try: callback(action_request)
                                  except Exception as e: logger.error(f"Error dispatching final external action: {e}")
                        # Record LLM_RESPONSE_PROCESSED event (no tool dispatched)
                        self.parent_inner_space.add_event_to_primary_timeline({
                             "event_type": self.EVENT_TYPE_LLM_RESPONSE_PROCESSED,
                             "data": {"loop_component_id": self.id, "dispatched_tool_action": False, "final_actions_count": len(final_action_requests)}
                        })
                    else:
                         # Record LLM_RESPONSE_PROCESSED event (no tool, no text)
                         logger.info("LLM response had no tool calls and no text content.")
                         self.parent_inner_space.add_event_to_primary_timeline({
                              "event_type": self.EVENT_TYPE_LLM_RESPONSE_PROCESSED,
                              "data": {"loop_component_id": self.id, "dispatched_tool_action": False, "final_actions_count": 0}
                         })

            # === STAGE: Waiting for Tool Result ===
            elif current_stage == "waiting_for_tool_result":
                logger.info(f"Currently waiting for tool result. No action taken this cycle.")
                pass

            # === STAGE: Interaction Complete ===
            elif current_stage == "interaction_complete":
                 logger.info(f"Multi-step interaction appears complete. No action taken this cycle.")
                 pass

            # === STAGE: Unknown/Error ===
            else:
                 logger.error(f"Reached unknown stage: {current_stage}. Aborting cycle.")

        except Exception as e:
            logger.error(f"{self.agent_loop_name} ({self.id}): Error during multi-step cognitive cycle: {e}", exc_info=True)
        finally:
            logger.info(f"{self.agent_loop_name} ({self.id}): Multi-step cycle finished.") 
            logger.info(f"{self.agent_loop_name} ({self.id}): Multi-step cycle finished.") 