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
    
    # Updated signature to accept action_context
    def _execute_tool(self, tool_provider: Optional['ToolProviderComponent'], 
                        tool_name: str, tool_args: Dict[str, Any], 
                        action_context: Dict[str, Any]):
        """Handles getting tool definition and executing based on execution_info."""
        # Removed internal fetching of context_manager
        # context_manager = self._get_dependency("context_manager", required=False)
        
        if not tool_provider:
            logger.error(f"[{self.element.id}] Cannot execute tool '{tool_name}': ToolProvider not found.")
            # Removed logging to history via context_manager
            return
            
        tool_def = tool_provider.get_tool_definition(tool_name)
        if not tool_def:
            logger.error(f"[{self.element.id}] Tool '{tool_name}' definition not found.")
            # Removed logging to history via context_manager
            return
            
        exec_info = tool_def.execution_info
        exec_type = exec_info.get("type")

        # TODO: Get actual tool_call_id from LLM response if available
        tool_call_id = f"temp_{tool_name}_{int(time.time()*1000)}" 

        if exec_type == "direct_call":
            # Direct call execution - needs careful thought about context/permissions
            # This remains unchanged for now, but might need action_context later
            tool_func = exec_info.get("function")
            if not callable(tool_func):
                 logger.error(f"[{self.element.id}] Invalid execution_info for direct_call tool '{tool_name}': 'function' is missing or not callable.")
                 # Removed logging to history via context_manager
                 return
            try:
                 logger.info(f"[{self.element.id}] Executing direct_call tool '{tool_name}' synchronously.")
                 tool_result = tool_func(**tool_args)
                 logger.info(f"[{self.element.id}] Direct tool '{tool_name}' result: {str(tool_result)[:100]}...")
                 # TODO: How should direct tool results be handled / added to history?
                 # Maybe emit a specific event? 
            except Exception as tool_err:
                 logger.error(f"[{self.element.id}] Error executing direct_call tool '{tool_name}': {tool_err}", exc_info=True)
                 # Removed logging to history via context_manager

        elif exec_type == "action_request":
            if not self._outgoing_action_callback:
                 logger.error(f"[{self.element.id}] Cannot enqueue action_request tool '{tool_name}': Callback not set.")
                 # Removed logging to history via context_manager
                 return
            
            target_module = exec_info.get("target_module")
            action_type = exec_info.get("action_type")
            payload_template = exec_info.get("payload", {}) 
            
            if not target_module or not action_type:
                 logger.error(f"[{self.element.id}] Invalid execution_info for action_request tool '{tool_name}': Missing target_module or action_type.")
                 # Removed logging to history via context_manager
                 return
                 
            # Start with base payload from tool definition
            final_payload = payload_template.copy()
            # Merge tool arguments provided by LLM
            final_payload.update(tool_args)
            # Add tool metadata
            final_payload['tool_name'] = tool_name
            final_payload['tool_call_id'] = tool_call_id 
            
            # Add requestor info
            if self.element:
                 final_payload['requesting_element_id'] = self.element.id
                 final_payload['requesting_agent_id'] = self.element.id.replace("_inner_space", "")
                 final_payload['agent_name'] = self.element.name or self.element.id 
            else:
                 logger.error(f"[{self.__class__.__name__}] Cannot add requesting IDs/name: Element reference missing.")
                 final_payload['requesting_element_id'] = "unknown_element"
                 final_payload['requesting_agent_id'] = "unknown_agent"
                 final_payload['agent_name'] = "Unknown Agent"
                 
            # Add routing context (adapter_id, conversation_id) from action_context
            if action_context.get("adapter_id"):
                 final_payload['adapter_id'] = action_context["adapter_id"]
            if action_context.get("conversation_id"):
                 final_payload['conversation_id'] = action_context["conversation_id"]
                 
            # Construct the full action request
            action_request = {
                "type": "action_request",
                "target_module": target_module,
                "action_type": action_type,
                "payload": final_payload
            }

            logger.info(f"[{self.element.id}] Enqueuing action request for tool '{tool_name}': Target={target_module}, Action={action_type}")
            logger.debug(f"Action Request Payload: {final_payload}")
            self._outgoing_action_callback(action_request)
        else:
             logger.error(f"[{self.element.id}] Unknown execution type '{exec_type}' for tool '{tool_name}'.")
             # Removed logging to history via context_manager

    # Updated signature to accept action_context
    def _publish_final_message(self, action_context: Dict[str, Any], message_content: str):
        """Constructs and enqueues a 'send_message' action request."""
        if not self._outgoing_action_callback:
             logger.error(f"[{self.element.id}] Cannot publish final message: Callback not set.")
             return
             
        # Removed context_manager usage for getting routing info
        # context_manager = self._get_dependency("context_manager", required=False)
        # if not context_manager:
        #      logger.error(f"[{self.element.id}] Cannot publish final message: ContextManager not found for routing info.")
        #      return
             
        # Assume context_manager provided routing info; replace with action_context
        # adapter_id = context_manager.get_last_context_value('adapter_id')
        # conversation_id = context_manager.get_last_context_value('conversation_id')
        adapter_id = action_context.get("adapter_id")
        conversation_id = action_context.get("conversation_id")

        if not adapter_id or not conversation_id:
             logger.error(f"[{self.element.id}] Cannot publish final message: Missing adapter_id or conversation_id in action_context.")
             # Consider falling back to a default or logging an internal error message?
             return
             
        payload = {
             "adapter_id": adapter_id,
             "conversation_id": conversation_id,
             "text": message_content,
             # Add requestor info
             "requesting_element_id": self.element.id if self.element else "unknown",
             "requesting_agent_id": self.element.id.replace("_inner_space", "") if self.element else "unknown",
             "agent_name": self.element.name if self.element else "Unknown Agent"
        }
        
        action_request = {
            "type": "action_request",
            "target_module": "ActivityClient", # Standard target for messaging
            "action_type": "send_message",
            "payload": payload
        }
        
        logger.info(f"[{self.element.id}] Enqueuing final message action request to {adapter_id}/{conversation_id}")
        logger.debug(f"Send Message Payload: {payload}")
        self._outgoing_action_callback(action_request)

    # --- Memory Processing Logic --- 
    # (Keep handle_action_request and _handle_memory_processing_request as they are for now)
    async def handle_action_request(self, action_type: str, payload: Dict[str, Any]):
        """Handles specific action requests targeted at the agent loop itself."""
        logger.debug(f"[{self.element.id}] AgentLoop received action request: {action_type}")
        if action_type == "trigger_memory_processing":
            await self._handle_memory_processing_request(payload)
        else:
            logger.warning(f"[{self.element.id}] AgentLoop received unhandled action request type: {action_type}")

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