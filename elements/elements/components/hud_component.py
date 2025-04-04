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
    DEPENDENCIES: List[str] = [ContextManagerComponent.COMPONENT_TYPE, ToolProvider.COMPONENT_TYPE]
    
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

        tools = tool_provider.list_tools()
        tool_definitions = []
        for tool_name, tool_info in tools.items():
            # --- TODO: Convert parameter_descriptions into JSON Schema --- 
            # This is a placeholder conversion. Real implementation needs robust 
            # parsing of descriptions into a valid JSON Schema object.
            parameters_schema = {"type": "object", "properties": {}, "required": []}
            param_descriptions = tool_info.get('parameter_descriptions', {})
            if isinstance(param_descriptions, dict):
                for param_name, desc in param_descriptions.items():
                     # Basic type guessing (needs improvement)
                     param_type = "string" 
                     if "int" in desc.lower() or "number" in desc.lower(): param_type = "number"
                     if "bool" in desc.lower(): param_type = "boolean"
                     if "list" in desc.lower() or "array" in desc.lower(): param_type = "array"
                     # Assume required if not explicitly optional?
                     parameters_schema["properties"][param_name] = {"type": param_type, "description": desc}
                     # Naive assumption about required params
                     if "optional" not in desc.lower():
                          parameters_schema["required"].append(param_name)
            else:
                 logger.warning(f"Invalid parameter_descriptions format for tool {tool_name}: {param_descriptions}")
            # -----------------------------------------------------------

            tool_def = LLMToolDefinition(
                name=tool_name,
                description=tool_info.get("description", "No description."),
                parameters=parameters_schema
            )
            tool_definitions.append(tool_def)

        return tool_definitions if tool_definitions else None

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

    def run_agent_cycle(self) -> Optional[LLMResponse]:
        """
        Prepares messages and tools, calls the LLM provider, and handles the response.
        Returns the LLMResponse or None if the call failed.
        """
        if not self._llm_provider:
             logger.error("Cannot run agent cycle: LLMProvider is not configured.")
             return None
             
        messages = self._prepare_llm_messages()
        if not messages:
             logger.error("Failed to prepare messages for LLM.")
             return None
             
        tool_definitions = self._prepare_tool_definitions()
        
        logger.info(f"Running agent cycle with {len(messages)} messages and {len(tool_definitions) if tool_definitions else 0} tools...")
        self._state["last_presentation_time"] = int(time.time() * 1000)
        
        try:
            # --- Call LLM Provider --- 
            llm_response: LLMResponse = self._llm_provider.complete(
                messages=messages,
                tools=tool_definitions,
                model=self._state.get("model"),
                temperature=self._state.get("temperature"),
                max_tokens=self._state.get("max_tokens")
                # Pass other kwargs if needed
            )
            # -----------------------
            
            self._state["last_llm_response"] = llm_response # Store structured response
            self.handle_agent_response(llm_response)
            return llm_response
            
        except Exception as e:
             logger.error(f"Error during LLM completion: {e}", exc_info=True)
             # Store error state? 
             self._state["last_llm_response"] = None # Clear last response on error
             return None

    def handle_agent_response(self, response: LLMResponse) -> None:
        """Processes the structured LLMResponse."""
        if not response:
             logger.warning("handle_agent_response called with None response.")
             return
             
        logger.info(f"Handling LLM response (Finish reason: {response.finish_reason})...")
        
        context_manager = self._get_context_manager()
        if not context_manager:
             logger.error("Cannot add response to history: ContextManagerComponent missing.")
             # Should we proceed with tool calls even if history can't be saved?
             # For now, let's return early to highlight the missing dependency.
             return 
             
        tool_provider = self._get_tool_provider()
        executed_tool = False
        assistant_content_added = False # Flag to avoid adding empty assistant messages

        # --- Record Assistant's Turn (Tool Calls or Content) --- 
        # Add the assistant's raw response (which might contain tool calls) to history first
        if response.tool_calls or response.content: 
             context_manager.add_history_turn(role="assistant", content=response) # Store the whole response object? Or just tool_calls/content? Let's store response for now.
             assistant_content_added = True

        # --- Handle Tool Calls --- 
        if response.tool_calls:
             if not tool_provider:
                  logger.error("LLM requested tool calls, but ToolProvider is missing!")
                  # Add error message to history
                  context_manager.add_history_turn(role="system", content="Error: ToolProvider component is missing, cannot execute tools.")
                  return
                  
             executed_tool = True
             logger.info(f"Processing {len(response.tool_calls)} tool call(s)...")
             for tool_call in response.tool_calls:
                  tool_name = tool_call.tool_name
                  tool_params = tool_call.parameters
                  # TODO: Need a way to get tool_call_id if provided by LLM API for result matching
                  tool_call_id_for_history = f"tool_{tool_name}_{int(time.time()*1000)}" # Generate temporary ID
                  
                  logger.info(f"Attempting tool call: {tool_name} with params: {tool_params}")
                  tool_result_content = None
                  try:
                       tool_result = tool_provider.execute_tool(tool_name, **tool_params)
                       logger.info(f"Tool execution result for {tool_name}: {tool_result}")
                       tool_result_content = tool_result # Store result to add to history
                       
                  except Exception as e:
                       logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
                       tool_result_content = {"error": f"Error executing tool {tool_name}: {str(e)}"} # Store error
                  
                  # Add tool result (or error) to history
                  context_manager.add_history_turn(
                      role="tool", 
                      content=tool_result_content, 
                      name=tool_name,
                      tool_call_id=tool_call_id_for_history # Reference the call
                  )
        
        # --- Handle Text Content --- 
        if response.content and not executed_tool: 
             # Assistant provided only text content, history was already added above
             logger.info(f"Agent response was plain text (already added to history): {response.content[:100]}...")
             # --- TODO: Handle Plain Text Action --- 
             # This is where you might emit an event to publish the message
             # if self.element:
             #     self.element.emit_event("publish_message_request", {"data": {"text": response.content}})
             # ----------------------------------
             pass 
        elif not executed_tool and not assistant_content_added:
             # Only log warning if assistant didn't provide content *and* didn't call tools
             logger.warning("LLM response had no tool calls and no content.")

    # --- Event Handling --- 
    def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """Handle events."""
        event_type = event.get("event_type")
        
        if event_type == "run_agent_cycle_request":
            logger.info("Received request to run agent cycle.")
            self.run_agent_cycle() # Trigger the cycle
            return True # Event handled
            
        # No longer need agent_response_received event from outside
        # elif event_type == "agent_response_received": ... 
                 
        return False # Event not handled 