"""
HUD Component
Component responsible for formatting context and presenting it to the agent using LLMProvider.
"""

import logging
from typing import Dict, Any, Optional, List
import time
import json # For JSON parsing in response handling

from ..base_component import Component
# Remove ContextManagerComponent dependency for now
# from .context_manager_component import ContextManagerComponent 
from .tool_provider_component import ToolProvider
# Need BaseElement for type hinting
from ..base import BaseElement
# Import the new representation base class
from .base_representation_component import BaseRepresentationComponent 
# Import LLM provider interface and data classes
from ...llm.provider_interface import (
    LLMProvider,
    LLMMessage,
    LLMToolDefinition,
    LLMToolCall,
    LLMResponse
)
# Optional: Import factory if HUD creates its own provider
# from ...llm.provider_factory import LLMProviderFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HUDComponent(Component):
    """
    Formats context (from representation), prepares messages and tool definitions, 
    interacts with an LLMProvider, and handles the LLMResponse.
    
    Lives within the InnerSpace.
    """
    
    COMPONENT_TYPE: str = "hud"
    # Update dependencies: Remove ContextManager, ensure ToolProvider is listed if needed
    # DEPENDENCIES: List[str] = [ToolProvider.COMPONENT_TYPE]
    
    # Declare injected dependencies from the parent Element (InnerSpace)
    # Format: {kwarg_name_in_init: attribute_name_in_parent_element}
    INJECTED_DEPENDENCIES: Dict[str, str] = {
        'llm_provider': '_llm_provider' # Expects __init__(..., llm_provider=...), gets it from element._llm_provider
    }
    
    HANDLED_EVENT_TYPES: List[str] = [
        "run_agent_cycle_request" # Explicit request to run the cycle
    ]
    
    def __init__(self, element: Optional[BaseElement] = None,
                 llm_provider: Optional[LLMProvider] = None, 
                 system_prompt: str = "You are a helpful AI assistant.",
                 model: Optional[str] = None, 
                 temperature: Optional[float] = 0.7,
                 max_tokens: Optional[int] = 1000,
                 # history_representation is less relevant now, as we get structured data
                 # history_representation: str = "user_message", 
                 **kwargs):
        """
        Initialize the HUD component.
        
        Args:
            element: The Element this component is attached to (InnerSpace).
            llm_provider: An instance conforming to the LLMProvider interface.
            system_prompt: The base system prompt.
            model: Default model name to use (can be overridden in calls).
            temperature: Default sampling temperature.
            max_tokens: Default max tokens for responses.
            **kwargs: Passthrough for BaseComponent.
        """
        super().__init__(element, **kwargs)
        
        if llm_provider is None:
            # Should ideally be injected or retrieved via service locator/registry
            logger.error("HUDComponent requires an LLMProvider instance during initialization!")
            # raise ValueError("LLMProvider instance is required for HUDComponent")
            self._llm_provider = None # Set to None, methods will fail gracefully
        else:
            self._llm_provider = llm_provider
            
        self._state = {
            "system_prompt": system_prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # "history_representation": history_representation, # Removed
            # State related to last interaction
            "last_llm_response": None, # Store the LLMResponse object
            "last_presentation_time": None
        }
        # No need to store the prompt string directly anymore
        # self._last_presented_prompt: Optional[str] = None

    # --- Helper Methods --- 

    # def _get_context_manager(self) -> Optional[ContextManagerComponent]:
    #     if not self.element: return None
    #     return self.element.get_component(ContextManagerComponent)

    def _get_tool_provider(self) -> Optional[ToolProvider]:
        if not self.element: return None
        return self.element.get_component(ToolProvider)

    def _get_representation_component(self) -> Optional[BaseRepresentationComponent]:
         """Finds any component inheriting from BaseRepresentationComponent."""
         if not self.element: return None
         # Look for specific types first, then the base type
         for comp_type in ["representation.chat", "simple_representation", "representation.base"]:
              comp = self.element.get_component_by_type(comp_type)
              if comp and isinstance(comp, BaseRepresentationComponent):
                   return comp
         # Fallback check: iterate all components (less efficient)
         for comp in self.element.get_components().values():
              if isinstance(comp, BaseRepresentationComponent):
                   return comp
         logger.warning(f"No Representation Component found on element {self.element.id}")
         return None

    def _prepare_tool_definitions(self) -> Optional[List[LLMToolDefinition]]:
        """Gets tools from ToolProvider and formats them as LLMToolDefinition list."""
        tool_provider = self._get_tool_provider()
        if not tool_provider:
            return None

        # Use get_llm_tool_schemas which is already designed for this
        tool_definitions = tool_provider.get_llm_tool_schemas() 
        
        # --- TODO: Validate/Refine Schemas if needed --- 
        # The schemas returned by ToolProviderComponent.get_llm_tool_schemas 
        # *should* already be in a format suitable for LLMs like OpenAI
        # (e.g., {"type": "function", "function": {...}})
        # Add validation here if necessary, or adjustments for specific providers
        # if the LLMProvider interface doesn't handle it.
        # For now, we assume the format is correct.
        if tool_definitions:
            logger.debug(f"Prepared {len(tool_definitions)} tool definitions for LLM.")
        
        return tool_definitions # Return the list directly

    # --- Refactored Message Preparation --- 
    def _prepare_llm_messages(self) -> Optional[List[LLMMessage]]:
        """Prepares the list of messages using orchestrated context."""
        logger.debug("Preparing LLM messages from orchestrated context...")
        messages = []

        # 1. Add System Prompt
        if self._state.get("system_prompt"):
            messages.append(LLMMessage(role="system", content=self._state["system_prompt"]))

        # 2. Get Orchestrated Context from ContextManagerComponent
        # context_manager: Optional['ContextManagerComponent'] = self._get_dependency("context_manager") # Use helper if exists
        context_manager = self.element.get_component_by_type("context_manager") # Or direct access
        
        if not context_manager:
             logger.error(f"[{self.element.id}] Cannot prepare messages: ContextManagerComponent missing.")
             messages.append(LLMMessage(role="user", content="[System Error: Cannot retrieve context - ContextManager missing]"))
             return messages

        try:
            orchestrated_context = context_manager.get_orchestrated_context()
        except Exception as e:
             logger.exception(f"Error getting orchestrated context: {e}")
             messages.append(LLMMessage(role="user", content="[System Error: Failed to orchestrate context]"))
             return messages

        # 3. Extract and Format History Turns
        history_turns = orchestrated_context.get("history_turns", [])
        
        if not history_turns and orchestrated_context.get("active_conversation_id") is None:
             logger.debug("No active conversation history found in orchestrated context.")
             # Proceed with only system prompt initially?
        elif not history_turns:
             logger.warning("Orchestrated context is missing 'history_turns' list.")
             # Add a placeholder?
             messages.append(LLMMessage(role="user", content="[System Note: Current context available but contains no chat turns.]"))

        for turn in history_turns:
            # Turn data should already be formatted by ContextManager
            role = turn.get("role")
            turn_content = turn.get("content")
            # --- Handle Tool Role --- 
            if role == "tool":
                 tool_call_id = turn.get("tool_call_id")
                 tool_name = turn.get("tool_name") # ContextManager should include this
                 if tool_call_id and turn_content is not None:
                      # Format according to LLM provider expectations (e.g., OpenAI)
                      # Content should be the result string/JSON
                      content_str = json.dumps(turn_content) if isinstance(turn_content, (dict, list)) else str(turn_content)
                      # Some providers might want tool_call_id directly in the message object
                      # For OpenAI, the content is the result, role is 'tool', and tool_call_id is a parameter
                      # Let's create a standard LLMMessage first
                      messages.append(LLMMessage(role="tool", content=content_str, tool_call_id=tool_call_id, name=tool_name))
                 else:
                      logger.warning(f"Skipping tool turn in orchestrated context due to missing tool_call_id or content: {turn}")
            # --- Handle Other Roles --- 
            elif role and turn_content is not None:
                 llm_role = "assistant" if role == "agent" else role # user, system, assistant
                 # Ensure content is string
                 content_str = json.dumps(turn_content) if isinstance(turn_content, (dict, list)) else str(turn_content)
                 # Add tool_calls to assistant message if present in original turn?
                 # The history turn structure might need refinement if assistant message *also* has tool_calls key
                 messages.append(LLMMessage(role=llm_role, content=content_str))
            else:
                 logger.warning(f"Skipping non-tool turn in orchestrated context due to missing role or content: {turn}")
                 
        # 4. TODO: Optionally add other parts of the orchestrated context? 
        # E.g., append self_summary, memories, environment_summary as a final user message?
        # Example:
        # summary_parts = []
        # if orchestrated_context.get('self_summary'): summary_parts.append(json.dumps(orchestrated_context['self_summary']))
        # if orchestrated_context.get('memories'): summary_parts.append(json.dumps(orchestrated_context['memories']))
        # if orchestrated_context.get('environment_summary'): summary_parts.append(json.dumps(orchestrated_context['environment_summary']))
        # if summary_parts:
        #     messages.append(LLMMessage(role="user", content="[Current Context Summary]\n" + "\n---\n".join(summary_parts))) 
        
        if len(messages) <= 1: # Only system prompt present
             logger.warning("Prepared LLM messages contain only system prompt. No history or other context added.")
             messages.append(LLMMessage(role="user", content="What should I do next?"))
             
        # TODO: Implement final token budget check/truncation here if needed
        # This would operate on the final list of LLMMessage objects.
             
        return messages

    # Renamed from run_agent_cycle
    async def prepare_and_call_llm(self) -> Optional[LLMResponse]:
        """
        Prepares messages (from representation) and tools, calls the LLM provider.
        Returns the LLMResponse or None if the call failed.
        """
        if not self._llm_provider:
             logger.error("Cannot run agent cycle: LLMProvider is not configured.")
             return None

        # REMOVED: ContextManager usage
        # context_manager = self._get_context_manager()
        # if not context_manager: ...
        # processed_context_str = context_manager.build_context()

        # Prepare messages using the representation component
        messages = self._prepare_llm_messages()
        if not messages:
             logger.error("Failed to prepare messages for LLM from representation.")
             return None
             
        # Prepare tool definitions
        tool_definitions = self._prepare_tool_definitions()
        
        logger.info(f"[{self.element.id}] Calling LLM with {len(messages)} messages and {len(tool_definitions) if tool_definitions else 0} tools...")
        self._state["last_presentation_time"] = int(time.time() * 1000)
        
        try:
            # --- Call LLM Provider --- 
            llm_response: LLMResponse = await self._llm_provider.complete(
                messages=messages,
                tools=tool_definitions, # Pass the structured tool definitions
                model=self._state.get("model"),
                temperature=self._state.get("temperature"),
                max_tokens=self._state.get("max_tokens")
            )
            # -----------------------
            
            # Log the raw response for debugging if needed
            if llm_response: logger.debug(f"[{self.element.id}] Raw LLM Response: {llm_response}")
            
            self._state["last_llm_response"] = llm_response # Store structured response
            # Return the response for the AgentLoopComponent to handle
            return llm_response 
            
        except Exception as e:
             logger.error(f"[{self.element.id}] Error during LLM completion: {e}", exc_info=True)
             self._state["last_llm_response"] = None # Clear last response on error
             return None

    # --- Event Handling --- 
    def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """Handle events."""
        event_type = event.get("event_type")
        
        if event_type == "run_agent_cycle_request":
            logger.info("Received request to run agent cycle.")
            # We need to run this async task but the handler itself is sync
            # Use asyncio.create_task if this component is running in an async context
            # For simplicity here, assume prepare_and_call_llm is intended to be awaited elsewhere
            # or handled by the caller of _on_event if needed.
            # await self.prepare_and_call_llm() # Cannot await in sync function
            # Instead, the AgentLoopComponent should call prepare_and_call_llm
            logger.debug("HUD processing run_agent_cycle_request (action likely deferred to AgentLoop)." )
            return True # Acknowledge handling the request intent
            
        # No longer need agent_response_received event from outside
        # elif event_type == "agent_response_received": ... 
                 
        return False # Event not handled 