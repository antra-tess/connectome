"""
HUD Component
Component responsible for formatting context and presenting it to the agent using LLMProvider.
"""

import logging
from typing import Dict, Any, Optional, List
import time
import json # For JSON parsing in response handling

from ..base_component import Component
from .context_manager_component import ContextManagerComponent
from .tool_provider_component import ToolProvider
# Need BaseElement for type hinting
from ..base import BaseElement
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
    Formats context, prepares messages and tool definitions, interacts with an
    LLMProvider, and handles the LLMResponse.
    
    Lives within the InnerSpace.
    """
    
    COMPONENT_TYPE: str = "hud"
    # Declare dependencies on other components (optional, for validation later)
    # DEPENDENCIES: List[str] = [ContextManagerComponent.COMPONENT_TYPE, ToolProvider.COMPONENT_TYPE]
    
    # Declare injected dependencies from the parent Element (InnerSpace)
    # Format: {kwarg_name_in_init: attribute_name_in_parent_element}
    INJECTED_DEPENDENCIES: Dict[str, str] = {
        'llm_provider': '_llm_provider' # Expects __init__(..., llm_provider=...), gets it from element._llm_provider
    }
    
    HANDLED_EVENT_TYPES: List[str] = [
        # "agent_response_received", # Handled internally after call now
        "run_agent_cycle_request" # Explicit request to run the cycle
    ]
    
    def __init__(self, element: Optional[BaseElement] = None,
                 llm_provider: Optional[LLMProvider] = None, # Inject the provider
                 system_prompt: str = "You are a helpful AI assistant.",
                 # tool_format is less relevant now, provider handles formatting
                 # response_prefix: Optional[str] = None,
                 model: Optional[str] = None, # Default model for provider
                 temperature: Optional[float] = 0.7,
                 max_tokens: Optional[int] = 1000,
                 # Add config for how to represent history to LLM
                 history_representation: str = "user_message", # or "message_list"
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
            history_representation: How to represent history to LLM
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
            "history_representation": history_representation,
            # State related to last interaction
            "last_llm_response": None, # Store the LLMResponse object
            "last_presentation_time": None
        }
        # No need to store the prompt string directly anymore
        # self._last_presented_prompt: Optional[str] = None

    # --- Helper Methods --- 

    def _get_context_manager(self) -> Optional[ContextManagerComponent]:
        if not self.element: return None
        return self.element.get_component(ContextManagerComponent)

    def _get_tool_provider(self) -> Optional[ToolProvider]:
        if not self.element: return None
        return self.element.get_component(ToolProvider)

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

    def _prepare_llm_messages(self) -> Optional[List[LLMMessage]]:
        """Prepares the list of messages for the LLM call."""
        logger.debug("Preparing LLM messages...")
        context_manager = self._get_context_manager()
        if not context_manager:
             logger.error("Cannot prepare messages: ContextManagerComponent missing.")
             return None

        # 1. Build the context string (which now includes history implicitly)
        processed_context_str = context_manager.build_context()

        # 2. Assemble messages based on representation strategy
        messages = []
        if self._state.get("system_prompt"):            
            messages.append(LLMMessage(role="system", content=self._state["system_prompt"]))            

        history_rep = self._state.get("history_representation", "user_message")
        
        if history_rep == "user_message":
            # Simplest: Send the entire processed context as one user message
            # This relies on ContextManager formatting history correctly within the string.
            messages.append(LLMMessage(role="user", content=processed_context_str))
        elif history_rep == "message_list":
            # TODO: More complex: Get structured history from ContextManager 
            # and format it into individual LLMMessages (user, assistant, tool)
            # Requires ContextManager to provide structured history access.
            logger.warning("'message_list' history representation not yet fully implemented in HUD.")
            # Fallback to user_message style for now
            messages.append(LLMMessage(role="user", content=processed_context_str))
        else:
             logger.error(f"Unknown history_representation: {history_rep}")
             return None
        
        return messages

    # Renamed from run_agent_cycle
    async def prepare_and_call_llm(self) -> Optional[LLMResponse]:
        """
        Prepares messages and tools, calls the LLM provider.
        Returns the LLMResponse or None if the call failed.
        """
        if not self._llm_provider:
             logger.error("Cannot run agent cycle: LLMProvider is not configured.")
             return None
             
        # Get context string (includes history) using ContextManager
        context_manager = self._get_context_manager()
        if not context_manager:
             logger.error("Cannot prepare messages: ContextManagerComponent missing.")
             return None
        processed_context_str = context_manager.build_context()

        # Prepare messages using the context string
        messages = self._prepare_llm_messages_from_string(processed_context_str)
        if not messages:
             logger.error("Failed to prepare messages for LLM.")
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

    # Renamed from _prepare_llm_messages to clarify input
    def _prepare_llm_messages_from_string(self, processed_context_str: str) -> Optional[List[LLMMessage]]:
        """Prepares the list of messages using the pre-built context string."""
        logger.debug("Preparing LLM messages from context string...")
        messages = []
        if self._state.get("system_prompt"):            
            messages.append(LLMMessage(role="system", content=self._state["system_prompt"]))            

        history_rep = self._state.get("history_representation", "user_message")
        
        if history_rep == "user_message":
            # Simplest: Send the entire processed context as one user message
            messages.append(LLMMessage(role="user", content=processed_context_str))
        elif history_rep == "message_list":
            # TODO: Requires ContextManager to provide structured history access.
            logger.warning("'message_list' history representation not yet fully implemented in HUD.")
            # Fallback to user_message style for now
            messages.append(LLMMessage(role="user", content=processed_context_str))
        else:
             logger.error(f"Unknown history_representation: {history_rep}")
             return None
        
        return messages

    # --- Event Handling --- 
    def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """Handle events."""
        event_type = event.get("event_type")
        
        if event_type == "run_agent_cycle_request":
            logger.info("Received request to run agent cycle.")
            self.prepare_and_call_llm() # Trigger the cycle
            return True # Event handled
            
        # No longer need agent_response_received event from outside
        # elif event_type == "agent_response_received": ... 
                 
        return False # Event not handled 