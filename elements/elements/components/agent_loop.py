"""
Agent Loop Components

Components defining different strategies for the agent's core reasoning cycle.
"""

import logging
import abc
import asyncio # Added for async tool check
from typing import Dict, Any, Optional, Callable, List, Union # Added Union
import time

# Core element/component imports
from .base import Component # Adjusted import
# Need to handle potential circular import if InnerSpace imports this file
from typing import TYPE_CHECKING 
if TYPE_CHECKING:
    from ..inner_space import InnerSpace # Adjusted import
    from ....llm.provider_interface import LLMProviderInterface # Adjusted import
    from .tool_provider import ToolProvider # Adjusted import
    from .hud_component import HUDComponent # Adjusted import
    from .publisher import PublisherComponent # Adjusted import
    from .context_manager import ContextManagerComponent # Adjusted import
    # Callback for outgoing actions 
    from ....host.event_loop import OutgoingActionCallback 
    # Import Memory Components for handling memory requests
    from .memory.self_query_memory_generator import SelfQueryMemoryGenerationComponent
    from .memory.curated_memory_generator import CuratedMemoryGenerationComponent # Import curated generator
    from .memory.structured_memory_component import StructuredMemoryComponent # To store results

logger = logging.getLogger(__name__)

class BaseAgentLoopComponent(Component, abc.ABC):
    """
    Abstract base class for components that manage the agent's reasoning loop.
    Subclasses implement different strategies (Simple, ReAct, Two-Phase, etc.).
    """
    COMPONENT_TYPE = "agent_loop"
    # Declare components expected on the same element (optional for validation)
    # DEPENDENCIES = {"tool_provider", "hud", "context_manager"}
    
    # Declare injected dependencies from the parent Element (InnerSpace)
    INJECTED_DEPENDENCIES: Dict[str, str] = {
        'llm_provider': '_llm_provider' # Expects __init__(..., llm_provider=...)
    }

    def __init__(self, 
                 element: 'InnerSpace',
                 llm_provider: Optional['LLMProviderInterface'] = None): # Make optional if injected
                 # Removed callback from init, will be set via setter
        super().__init__(element)
        # Store injected provider if passed, otherwise expect it to be injected
        # during component setup by the parent element.
        self._llm_provider = llm_provider 
        self._outgoing_action_callback: Optional['OutgoingActionCallback'] = None
        
        # References to dependencies - get them when needed to avoid init order issues
        self._tool_provider: Optional['ToolProviderComponent'] = None # Corrected type hint
        self._hud: Optional['HUDComponent'] = None
        # self._publisher: Optional['PublisherComponent'] = None # Removed publisher dependency
        self._context_manager: Optional['ContextManagerComponent'] = None # Corrected type hint
        # Add refs for memory components (will be fetched by _get_dependency)
        self._memory_generator_self_query: Optional[SelfQueryMemoryGenerationComponent] = None
        self._memory_generator_curated: Optional[CuratedMemoryGenerationComponent] = None
        self._memory_store: Optional[StructuredMemoryComponent] = None
             
        logger.debug(f"{self.__class__.__name__} initialized for element {element.id}")

    def set_outgoing_action_callback(self, callback: 'OutgoingActionCallback'):
        """Sets the callback for enqueuing outgoing actions."""
        self._outgoing_action_callback = callback
        logger.debug(f"Outgoing action callback set for {self.__class__.__name__} on {self.element.id}")

    # Helper to safely get component references only when needed during run_cycle
    def _get_dependency(self, component_type_str: str, required: bool = True) -> Optional[Component]:
        """Safely gets a dependency component instance from the parent element."""
        attr_name = f"_{component_type_str}"
        if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
             return getattr(self, attr_name)
             
        if not hasattr(self, 'element') or self.element is None:
             logger.error(f"{self.__class__.__name__} cannot get dependency: element reference is not set.")
             return None
             
        comp = self.element.get_component_by_type(component_type_str)
        if required and not comp:
            logger.error(f"{self.__class__.__name__} on {self.element.id} missing required dependency: {component_type_str}")
        elif comp and hasattr(self, attr_name):
             setattr(self, attr_name, comp)
             
        return comp

    @abc.abstractmethod
    async def run_cycle(self) -> None:
        """
        Executes one iteration or the complete logic of the agent's reasoning cycle
        according to the specific strategy implemented by the subclass.
        """
        pass
        
    async def _call_llm(self, context: Any) -> Any:
        """Helper to call the LLM provider."""
        if not self._llm_provider:
            logger.error("Cannot call LLM: No provider available.")
            return None 
        try:
            logger.debug(f"[{self.element.id}] Calling LLM...")
            response = await self._llm_provider.generate(context)
            logger.debug(f"[{self.element.id}] LLM response received.")
            return response
        except Exception as e:
            logger.error(f"Error during LLM call: {e}", exc_info=True)
            return None
            
    # --- Common Action/Tool Handling Logic --- 
    # Moved here from SimpleRequestResponseLoopComponent
    
    def _execute_tool(self, tool_provider: Optional['ToolProviderComponent'], tool_name: str, tool_args: Dict[str, Any]):
        """Handles getting tool definition and executing based on execution_info."""
        # Get dependencies needed within this method
        context_manager = self._get_dependency("context_manager", required=False)
        
        if not tool_provider:
            logger.error(f"[{self.element.id}] Cannot execute tool '{tool_name}': ToolProvider not found.")
            # Log error to history if possible
            if context_manager: context_manager.add_history_turn(role="system", content=f"Error: ToolProvider not found, cannot execute {tool_name}.")
            return
            
        tool_def = tool_provider.get_tool_definition(tool_name)
        if not tool_def:
            logger.error(f"[{self.element.id}] Tool '{tool_name}' definition not found.")
            # Log error to history if possible
            if context_manager: context_manager.add_history_turn(role="system", content=f"Error: Tool '{tool_name}' definition not found.")
            return
            
        exec_info = tool_def.execution_info
        exec_type = exec_info.get("type")

        # TODO: Get actual tool_call_id from LLM response if available
        tool_call_id = f"temp_{tool_name}_{int(time.time()*1000)}" 

        if exec_type == "direct_call":
            tool_func = exec_info.get("function")
            if not callable(tool_func):
                 logger.error(f"[{self.element.id}] Invalid execution_info for direct_call tool '{tool_name}': 'function' is missing or not callable.")
                 if context_manager: context_manager.add_history_turn(role="system", content=f"Error: Invalid definition for direct_call tool {tool_name}.")
                 return
            try:
                 logger.info(f"[{self.element.id}] Executing direct_call tool '{tool_name}' synchronously.")
                 tool_result = tool_func(**tool_args)
                 logger.info(f"[{self.element.id}] Direct tool '{tool_name}' result: {str(tool_result)[:100]}...")
                 # Add result to history
                 if context_manager:
                     context_manager.add_history_turn(
                          role="tool", content=str(tool_result)[:500], # Truncate result string
                          name=tool_name, tool_call_id=tool_call_id
                     )
            except Exception as tool_err:
                 logger.error(f"[{self.element.id}] Error executing direct_call tool '{tool_name}': {tool_err}", exc_info=True)
                 # Add error to history
                 if context_manager:
                     context_manager.add_history_turn(
                          role="tool", content={"error": str(tool_err)}, 
                          name=tool_name, tool_call_id=tool_call_id
                     )

        elif exec_type == "action_request":
            if not self._outgoing_action_callback:
                 logger.error(f"[{self.element.id}] Cannot enqueue action_request tool '{tool_name}': Callback not set.")
                 if context_manager: context_manager.add_history_turn(role="system", content=f"Error: Cannot enqueue action request for tool {tool_name}, callback missing.")
                 return
            
            target_module = exec_info.get("target_module")
            action_type = exec_info.get("action_type")
            payload_template = exec_info.get("payload", {}) 
            
            if not target_module or not action_type:
                 logger.error(f"[{self.element.id}] Invalid execution_info for action_request tool '{tool_name}': Missing target_module or action_type.")
                 if context_manager: context_manager.add_history_turn(role="system", content=f"Error: Invalid definition for action_request tool {tool_name}.")
                 return
                 
            final_payload = payload_template.copy()
            final_payload['tool_name'] = tool_name
            final_payload['tool_args'] = tool_args
            final_payload['tool_call_id'] = tool_call_id
            # Ensure element reference exists before accessing id/name
            if self.element:
                 # Use InnerSpace ID as requesting_element_id
                 final_payload['requesting_element_id'] = self.element.id
                 # Derive agent_id if needed (e.g., strip suffix)
                 final_payload['requesting_agent_id'] = self.element.id.replace("_inner_space", "")
                 # <<< ADD AGENT NAME TO PAYLOAD >>>
                 final_payload['agent_name'] = self.element.name or self.element.id 
            else:
                 logger.error(f"[{self.__class__.__name__}] Cannot add requesting IDs/name: Element reference missing.")
                 # Optionally add placeholders or skip these fields
                 final_payload['requesting_element_id'] = "unknown_element"
                 final_payload['requesting_agent_id'] = "unknown_agent"
                 final_payload['agent_name'] = "Unknown Agent"
                 
            action_request = {
                 "target_module": target_module,
                 "action_type": action_type,
                 "payload": final_payload
            }
            logger.info(f"[{self.element.id if self.element else '?'}] Enqueuing action_request for tool: {tool_name}")
            self._outgoing_action_callback(action_request)
            # Add note to history that action was enqueued
            if context_manager:
                 context_manager.add_history_turn(role="system", content=f"Enqueued action request for tool '{tool_name}' (ID: {tool_call_id}).")
            
        else:
            logger.error(f"[{self.element.id}] Unknown execution_info type '{exec_type}' for tool '{tool_name}'.")
            if context_manager: context_manager.add_history_turn(role="system", content=f"Error: Unknown execution type for tool {tool_name}.")

    def _publish_final_message(self, context_manager: Optional['ContextManagerComponent'], message_content: str):
        """Handles publishing the final message via the ToolProvider using send_message tool."""
        logger.debug(f"[{self.element.id}] Attempting to publish final message via send_message tool.")
        
        tool_provider: Optional['ToolProviderComponent'] = self._get_dependency("tool_provider")
        # We need context_manager again here, ensure it's passed or get it again
        if context_manager is None:
             context_manager = self._get_dependency("context_manager")
             
        if not tool_provider:
             logger.error(f"[{self.element.id}] Cannot publish message: ToolProvider not found.")
             if context_manager: context_manager.add_history_turn(role="system", content="Error: Cannot publish message, ToolProvider missing.")
             return
        if not context_manager:
             logger.error(f"[{self.element.id}] Cannot publish message: ContextManager not found (required for routing).")
             # Cannot log to history without context manager
             return
             
        # --- Get Target Context using ContextManager --- 
        target_conversation_id = "fallback_conv_id"
        target_adapter_id = "fallback_adapter_id"
        # This call might fail if context_manager wasn't found
        current_routing_info = context_manager.get_current_routing_context() 
        if current_routing_info:
             target_conversation_id = current_routing_info.get("conversation_id", target_conversation_id)
             target_adapter_id = current_routing_info.get("adapter_id", target_adapter_id)
        else: 
            logger.warning(f"[{self.element.id}] ContextManager returned no routing info, using fallbacks.")
        # -------------------------------------------
        
        tool_args = {
             "text": message_content,
             "adapter_id": target_adapter_id,
             "conversation_id": target_conversation_id
        }
        
        # Use the _execute_tool method to handle calling the 'send_message' tool
        self._execute_tool(tool_provider, "send_message", tool_args)

    # --- Incoming Action Request Handling --- 

    async def handle_action_request(self, action_type: str, payload: Dict[str, Any]):
        """Handles incoming action requests targeted at the AgentLoop."""
        logger.info(f"[{self.element.id}] Received action request: type='{action_type}'")

        if action_type == "trigger_memory_processing":
            await self._handle_memory_processing_request(payload)
        # Add handlers for other action types here...
        # elif action_type == "some_other_action":
        #     await self._handle_other_action(payload)
        else:
            logger.warning(f"[{self.element.id}] Unhandled action_type '{action_type}' received by BaseAgentLoopComponent.")
            # Log to history maybe?
            ctx_mgr = self._get_dependency("context_manager", required=False)
            if ctx_mgr:
                 ctx_mgr.add_history_turn(role="system", content=f"Warning: Received unhandled action request type '{action_type}'")

    async def _handle_memory_processing_request(self, payload: Dict[str, Any]):
        """Handles the specific logic for the trigger_memory_processing action."""
        tool_args = payload.get('tool_args', {})
        process_one_chunk = tool_args.get('process_one_chunk', True)
        # Get the selected mechanism, defaulting to 'self_query'
        mechanism = tool_args.get('generation_mechanism', 'self_query').lower()
        tool_call_id = payload.get('tool_call_id', 'unknown')

        logger.info(f"[{self.element.id}] Handling trigger_memory_processing (from tool {tool_call_id}). "
                    f"Params: process_one_chunk={process_one_chunk}, mechanism='{mechanism}'")

        # Determine which generator component to use
        generator_comp_type: Optional[str] = None
        if mechanism == "curated":
            generator_comp_type = CuratedMemoryGenerationComponent.COMPONENT_TYPE
        elif mechanism == "self_query":
            generator_comp_type = SelfQueryMemoryGenerationComponent.COMPONENT_TYPE
        else:
            logger.warning(f"[{self.element.id}] Invalid generation_mechanism '{mechanism}' requested. Defaulting to self_query.")
            generator_comp_type = SelfQueryMemoryGenerationComponent.COMPONENT_TYPE

        # Get necessary components
        ctx_mgr: Optional['ContextManagerComponent'] = self._get_dependency("context_manager")
        # Fetch the selected generator component instance
        # Type hint needs to be broad or handled carefully
        mem_gen: Optional[Union[SelfQueryMemoryGenerationComponent, CuratedMemoryGenerationComponent]] = None
        if generator_comp_type:
             mem_gen = self._get_dependency(generator_comp_type)

        if not ctx_mgr:
            logger.error(f"[{self.element.id}] Cannot process memory: ContextManagerComponent not found.")
            return
        if not mem_gen:
            logger.error(f"[{self.element.id}] Cannot process memory: Memory Generation Component ('{generator_comp_type}') not found or failed to get.")
            if ctx_mgr: # Log to history if possible
                 ctx_mgr.add_history_turn(role="system", content=f"Error handling memory processing for {tool_call_id}: Requested memory generator '{generator_comp_type}' not found.")
            return

        # --- Actual Memory Processing Logic --- 
        # (The rest of the logic remains the same, as it calls mem_gen.generate_memory_for_chunk)
        result_message = f"Memory processing using '{mechanism}' failed for {tool_call_id}."
        try:
            # 1. Get unprocessed history chunk(s)
            unprocessed_chunks = await ctx_mgr.get_unprocessed_history_chunks(limit=1 if process_one_chunk else None)

            if not unprocessed_chunks:
                logger.info(f"[{self.element.id}] No unprocessed history chunks found for memory generation.")
                result_message = f"Memory processing triggered ({tool_call_id}), but no unprocessed history found."
                ctx_mgr.add_history_turn(role="system", content=result_message)
                return

            processed_ids = []
            failed_count = 0
            last_processed_ts = None

            for chunk_messages in unprocessed_chunks:
                if not chunk_messages: continue

                logger.info(f"[{self.element.id}] Processing memory chunk via '{mechanism}' generator... ({len(chunk_messages)} messages)")
                
                # 2. Call the selected memory generator for the chunk
                memory_id = await mem_gen.generate_memory_for_chunk(chunk_messages)

                if memory_id:
                    logger.info(f"[{self.element.id}] Successfully generated memory {memory_id} for chunk using '{mechanism}'.")
                    processed_ids.append(memory_id)
                    # 3. Update the marker timestamp
                    chunk_last_ts = chunk_messages[-1].get('timestamp')
                    if chunk_last_ts is not None:
                         ctx_mgr.update_processed_marker(chunk_last_ts)
                         last_processed_ts = chunk_last_ts
                else:
                    logger.error(f"[{self.element.id}] Failed to generate memory for chunk using '{mechanism}'. See generator logs.")
                    failed_count += 1
            
            # 4. Log overall result to history
            if processed_ids:
                 result_message = f"Memory processing via '{mechanism}' complete for {tool_call_id}. Generated {len(processed_ids)} memories. History processed up to timestamp {last_processed_ts}."
                 if failed_count > 0:
                      result_message += f" {failed_count} subsequent chunks failed processing."
            elif failed_count > 0:
                 result_message = f"Memory processing via '{mechanism}' failed for all {failed_count} chunks attempted for {tool_call_id}."
            
            logger.info(f"[{self.element.id}] Memory processing result: {result_message}")
            ctx_mgr.add_history_turn(role="system", content=result_message)

        except AttributeError as ae:
            # Handle cases where ContextManager is missing expected methods
            error_msg = f"Error during memory processing ({tool_call_id}): Missing required method on ContextManager ({ae})."
            logger.error(f"[{self.element.id}] {error_msg}", exc_info=True)
            if ctx_mgr: ctx_mgr.add_history_turn(role="system", content=error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during memory processing for tool call {tool_call_id}: {e}"
            logger.error(f"[{self.element.id}] {error_msg}", exc_info=True)
            if ctx_mgr: ctx_mgr.add_history_turn(role="system", content=error_msg)
        # ---------------------------------------

# --- Concrete Implementations Removed - Moved to separate files like simple_loop.py --- 