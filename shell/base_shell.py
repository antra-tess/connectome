"""
Base Shell

Defines the abstract base class for Shell implementations, which serves as the 
agentic loop container enclosing the model mind, providing a runtime environment 
and coordination layer.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Callable

from shell.hud.core import HUD, RenderingRequest, RenderingResponse
from shell.context_manager import ContextManager, ContextRequest, ContextResponse
from elements.elements.space import Space
from elements.space_registry import SpaceRegistry
from elements.elements.inner_space import InnerSpace
from llm import (
    LLMProvider, 
    LLMProviderFactory,
    LLMMessage, 
    LLMToolDefinition,
    LLMToolCall,
    LLMResponse
)
from rendering import RenderingFormat


class BaseShell(ABC):
    """
    Abstract base class for Shell implementations.
    
    The Shell is responsible for:
    1. Activating in response to external events or internal timers
    2. Processing agent actions and managing their execution
    3. Managing memory formation
    4. Providing internal tools accessible to the agent
    5. Presenting the Inner Space as the primary container for the agent
    
    Different Shell implementations can provide various interaction models:
    - Two-phase models (separate contemplation and engagement phases)
    - Single-phase interactive models
    - Specialized task-oriented models
    """
    
    def __init__(self, 
                 registry: SpaceRegistry,
                 hud: Optional[HUD] = None,
                 context_manager: Optional[ContextManager] = None,
                 inner_space: Optional[Space] = None,
                 model_info: Dict[str, Any] = None,
                 llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the shell with required components.
        
        Args:
            registry: Space Registry for creating InnerSpace
            hud: HUD instance for rendering (created if None)
            context_manager: Context Manager instance (created if None)
            inner_space: Agent's Inner Space (created if None)
            model_info: Dictionary containing information about the LLM model
            llm_config: Configuration for the LLM provider
        """
        self.model_info = model_info or {}
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        
        # Create components if not provided
        self.hud = hud or self._create_hud()
        self.context_manager = context_manager or self._create_context_manager()
        
        # Save registry reference
        self.registry = registry
        
        # Create or use existing inner space
        self.inner_space = inner_space or self._create_inner_space(registry)
        
        # Initialize the agent's subjective timeline
        self.current_timeline_context = {
            "timeline_id": "primary-branch-001",
            "is_primary": True,
            "last_event_id": None
        }
        
        # Set up internal tools
        self.internal_tools = self._setup_internal_tools()
        
        # Initialize LLM provider
        self.llm_config = llm_config or {"type": "litellm", "default_model": "gpt-4"}
        self._initialize_llm_provider()
        
        # Configure tool calling
        self.max_sequential_tool_calls = self.llm_config.get("max_sequential_tool_calls", 5)
        
        # Track state
        self.last_action_time = None
        self.is_processing_event = False
    
    def _initialize_llm_provider(self):
        """Initialize the LLM provider from configuration."""
        try:
            self.llm_provider = LLMProviderFactory.create_from_config(self.llm_config)
            self.logger.info(f"Initialized LLM provider: {self.llm_provider.__class__.__name__}")
        except (ImportError, ValueError) as e:
            self.logger.error(f"Failed to initialize LLM provider: {e}")
            raise RuntimeError(f"Failed to initialize LLM provider: {e}")
        
    def _create_hud(self) -> HUD:
        """Create and initialize the HUD component."""
        self.logger.info("Creating HUD")
        return HUD()
    
    def _create_context_manager(self) -> ContextManager:
        """Create and initialize the Context Manager component."""
        self.logger.info("Creating Context Manager")
        from shell.context_manager.core import ContextManager
        return ContextManager()
    
    def _create_inner_space(self, registry: SpaceRegistry) -> Space:
        """
        Create and initialize the agent's Inner Space.
        
        Args:
            registry: The space registry to use for creating the inner space
            
        Returns:
            The created inner space
        """
        self.logger.info("Creating agent's Inner Space")
        
        try:
            # Create the InnerSpace instance
            inner_space = InnerSpace("inner_space", "Agent's Inner Space", registry)
            
            # Verify it was registered properly
            if not registry.get_inner_space():
                self.logger.error("InnerSpace created but not registered with SpaceRegistry")
                # Try manual registration if automatic registration failed
                if not registry.register_inner_space(inner_space):
                    raise RuntimeError("Failed to register InnerSpace with SpaceRegistry")
                self.logger.info("InnerSpace manually registered with SpaceRegistry")
            
            # Verify it's accessible as a space
            if not registry.get_space(inner_space.id):
                self.logger.error("InnerSpace not accessible as a Space in SpaceRegistry")
                # Try manual registration as a regular space
                if not registry.register_space(inner_space):
                    raise RuntimeError("Failed to register InnerSpace as a Space in SpaceRegistry")
                self.logger.info("InnerSpace manually registered as a Space")
            
            self.logger.info(f"InnerSpace '{inner_space.id}' successfully created and registered")
            return inner_space
            
        except Exception as e:
            self.logger.error(f"Error creating InnerSpace: {e}")
            raise RuntimeError(f"Failed to create InnerSpace: {e}")
    
    def _setup_internal_tools(self) -> Dict[str, Dict[str, Any]]:
        """Set up internal tools that are always accessible to the agent."""
        tools = {}
        # Add standard internal tools
        tools["sleep"] = {
            "name": "sleep",
            "description": "Sleep for a specified duration",
            "parameters": {
                "seconds": "Duration to sleep in seconds"
            },
            "handler": self._handle_sleep
        }
        tools["set_reminder"] = {
            "name": "set_reminder",
            "description": "Set a reminder for a future time",
            "parameters": {
                "message": "Reminder message",
                "seconds": "Seconds from now to trigger the reminder"
            },
            "handler": self._handle_set_reminder
        }
        return tools
    
    def _handle_sleep(self, seconds: int) -> str:
        """Handle the sleep internal tool."""
        self.logger.info(f"Sleep requested for {seconds} seconds")
        # In a real implementation, we would use asyncio.sleep or similar
        return f"I'll pause for {seconds} seconds."
    
    def _handle_set_reminder(self, message: str, seconds: int) -> str:
        """Handle the set_reminder internal tool."""
        self.logger.info(f"Reminder set: '{message}' in {seconds} seconds")
        # In a real implementation, this would register a callback
        return f"Reminder set: '{message}' in {seconds} seconds"
    
    def handle_external_event(self, event: Dict[str, Any], timeline_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle an external event from the Activity Layer.
        
        This is the main entry point for the shell's event loop.
        
        Args:
            event: The external event to handle
            timeline_context: Optional timeline context, defaults to primary timeline
            
        Returns:
            Result of the event handling
        """
        self.logger.info(f"Handling external event: {event.get('type', 'unknown')}")
        
        # Validate and normalize timeline context
        timeline_context = self._validate_timeline_context(timeline_context or self.current_timeline_context)
        
        try:
            # Mark that we're processing an event
            self.is_processing_event = True
            
            # Check if this is an attention event
            if event.get('type') == 'attention_event':
                return self._handle_attention_event(event, timeline_context)
            
            # Normal event processing - update inner space with event
            self.inner_space.receive_event(event, timeline_context)
            
            # Begin the interaction cycle
            result = self._execute_interaction_cycle(event, timeline_context)
            
            # Update current timeline context
            self.current_timeline_context = timeline_context
            
            # Done processing
            self.is_processing_event = False
            
            return result
        except Exception as e:
            self.logger.error(f"Error handling event: {e}", exc_info=True)
            self.is_processing_event = False
            return {"error": str(e)}
    
    def _validate_timeline_context(self, timeline_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize timeline context."""
        # Ensure required fields
        validated = timeline_context.copy()
        if "timeline_id" not in validated:
            validated["timeline_id"] = "primary-branch-001"
        if "is_primary" not in validated:
            validated["is_primary"] = True
        if "last_event_id" not in validated:
            validated["last_event_id"] = None
        
        return validated
    
    @abstractmethod
    def _execute_interaction_cycle(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single interaction cycle.
        
        This is the core method that different shell implementations will override
        to implement their specific interaction models.
        
        Args:
            event: The event that triggered this cycle
            timeline_context: The timeline context for this cycle
            
        Returns:
            Result of the interaction cycle
        """
        pass
    
    def render_context(self, timeline_context: Dict[str, Any]) -> str:
        """
        Render the current context for presentation to the agent.
        
        Args:
            timeline_context: The timeline context to render for
            
        Returns:
            Rendered context as a string
        """
        self.logger.info("Rendering context for agent")
        
        # Create rendering request with timeline context
        request = RenderingRequest(
            timeline_id=timeline_context.get("timeline_id"),
            root_element_id=self.inner_space.id,
            format=RenderingFormat.MARKDOWN,
            include_details=True
        )
        
        # Get rendering from HUD
        rendered_context = self.hud.prepare_context_rendering(
            timeline_id=request.timeline_id,
            root_element_id=request.root_element_id,
            space_registry=self.registry,
            format=request.format,
            include_details=request.include_details
        )
        
        # Pass to Context Manager for final assembly
        context_request = ContextRequest(
            rendering_response=RenderingResponse(
                content=rendered_context,
                element_id=self.inner_space.id,
                metadata={"timeline_id": timeline_context.get("timeline_id")},
                timestamp=int(time.time() * 1000)
            ),
            timeline_context=timeline_context
        )
        
        context_response = self.context_manager.assemble_context(context_request)
        
        return context_response.context
    
    def present_context_to_agent(self, context: str) -> LLMResponse:
        """
        Present the context to the agent and get a response.
        
        Args:
            context: The context to present
            
        Returns:
            The agent's response
        """
        self.logger.info("Presenting context to agent")
        
        try:
            # Using actual LLM integration
            response = self.llm_provider.generate_response(
                messages=[{"role": "system", "content": context}],
                tools=self._get_available_tools()
            )
            return response
        except Exception as e:
            self.logger.error(f"Error presenting context to agent: {e}")
            # Return an error response rather than raising exception
            return LLMResponse(
                content="Error: Unable to process context. Please try again.",
                tool_calls=[],
                message_id=f"error-{int(time.time())}"
            )
    
    def parse_agent_actions(self, agent_response: LLMResponse) -> List[Dict[str, Any]]:
        """
        Parse the agent's response into executable actions.
        
        Args:
            agent_response: The agent's response
            
        Returns:
            List of parsed actions
        """
        self.logger.info("Parsing agent actions")
        
        actions = []
        
        # Process tool calls first (structured output from LLM)
        if agent_response.tool_calls:
            for tool_call in agent_response.tool_calls:
                # Handle shell tools
                if tool_call.name in self.internal_tools:
                    actions.append({
                        "type": "shell_tool",
                        "tool_name": tool_call.name,
                        "parameters": tool_call.parameters
                    })
                # Handle element actions (formatted as element_id.action_name)
                elif "." in tool_call.name:
                    element_id, action_name = tool_call.name.split(".", 1)
                    actions.append({
                        "type": "element_action",
                        "element_id": element_id,
                        "action_name": action_name,
                        "parameters": tool_call.parameters
                    })
                else:
                    # Unknown tool
                    self.logger.warning(f"Unknown tool: {tool_call.name}")
                    actions.append({
                        "type": "unknown_tool",
                        "tool_name": tool_call.name,
                        "parameters": tool_call.parameters
                    })
        
        # If no tool calls and there's content, treat as a message
        elif agent_response.content:
            actions.append({
                "type": "message",
                "content": agent_response.content
            })
        
        return actions
    
    def _get_available_tools(self) -> List[LLMToolDefinition]:
        """Get the list of tools available to the agent."""
        # Collect internal tools
        tools = []
        for tool_id, tool_info in self.internal_tools.items():
            tools.append(LLMToolDefinition(
                name=tool_info["name"],
                description=tool_info["description"],
                parameters=tool_info["parameters"]
            ))
        
        # Collect tools from inner space
        if self.inner_space:
            inner_space_tools = self.inner_space.get_available_tools()
            tools.extend(inner_space_tools)
        
        return tools
    
    def _execute_tool(self, tool_call: LLMToolCall, timeline_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call from the agent."""
        tool_name = tool_call.name
        parameters = tool_call.parameters
        
        # Check if it's an internal tool
        if tool_name in self.internal_tools:
            handler = self.internal_tools[tool_name]["handler"]
            try:
                result = handler(**parameters)
                return {
                    "success": True,
                    "tool_name": tool_name,
                    "result": result
                }
            except Exception as e:
                self.logger.error(f"Error executing tool {tool_name}: {e}")
                return {
                    "success": False,
                    "tool_name": tool_name,
                    "error": str(e)
                }
        
        # Check if it's an element action (format: element_id.action_name)
        if "." in tool_name:
            element_id, action_name = tool_name.split(".", 1)
            try:
                # Execute action on element through inner space
                result = self.inner_space.execute_element_action(
                    None, element_id, action_name, parameters, timeline_context
                )
                return {
                    "success": True,
                    "element_id": element_id,
                    "action_name": action_name,
                    "result": result
                }
            except Exception as e:
                self.logger.error(f"Error executing element action {tool_name}: {e}")
                return {
                    "success": False,
                    "element_id": element_id,
                    "action_name": action_name,
                    "error": str(e)
                }
        
        # Unknown tool
        return {
            "success": False,
            "tool_name": tool_name,
            "error": f"Unknown tool: {tool_name}"
        }
    
    def execute_multi_turn_with_tools(self, context: str, timeline_context: Dict[str, Any]) -> Tuple[LLMResponse, List[Dict[str, Any]]]:
        """
        Execute a multi-turn interaction with the agent using tools.
        
        Args:
            context: The context to present to the agent
            timeline_context: The timeline context for this interaction
            
        Returns:
            Tuple of (final response, tool results)
        """
        messages = [{"role": "system", "content": context}]
        tool_results = []
        
        for _ in range(self.max_sequential_tool_calls):
            # Present context to agent
            response = self.llm_provider.generate_response(
                messages=messages,
                tools=self._get_available_tools(),
                timeline_context=timeline_context
            )
            
            # Check for tool calls
            if not response.tool_calls:
                return response, tool_results
                
            # Execute tools and add results
            for tool_call in response.tool_calls:
                tool_result = self._execute_tool(tool_call, timeline_context)
                tool_results.append(tool_result)
                
                # Add tool result to messages
                messages.append({
                    "role": "tool", 
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result)
                })
        
        # Final response after tool calls
        final_response = self.llm_provider.generate_response(
            messages=messages,
            timeline_context=timeline_context
        )
        
        return final_response, tool_results 
    
    def get_elements_requesting_attention(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all elements that are currently requesting attention.
        
        Returns:
            Dictionary mapping element IDs to their attention request data
        """
        # Get attention requests from inner space
        if self.inner_space:
            return self.inner_space.get_elements_requesting_attention()
        return {}
    
    def update_element_with_response(self, element_id: str, response_data: Dict[str, Any], 
                                   timeline_context: Dict[str, Any]) -> bool:
        """
        Update an element with a response from the agent.
        
        This is used to handle responses to elements that requested attention.
        
        Args:
            element_id: ID of the element to update
            response_data: Response data from the agent
            timeline_context: Timeline context for the response
            
        Returns:
            True if the element was updated successfully, False otherwise
        """
        self.logger.info(f"Updating element {element_id} with agent response")
        
        try:
            # Find the element in the inner space
            element = self._find_element_by_id(element_id)
            if not element:
                self.logger.error(f"Could not find element {element_id} to update with response")
                return False
                
            # Check if the element has a handle_response method
            if hasattr(element, 'handle_response') and callable(element.handle_response):
                success = element.handle_response(response_data, timeline_context)
                return success
            else:
                self.logger.warning(f"Element {element_id} does not have a handle_response method")
                return False
        except Exception as e:
            self.logger.error(f"Error updating element with response: {e}")
            return False
            
    def _find_element_by_id(self, element_id: str) -> Optional[Any]:
        """
        Find an element by ID anywhere in the inner space hierarchy.
        
        Args:
            element_id: ID of the element to find
            
        Returns:
            The element if found, None otherwise
        """
        # Check inner space mounted elements first
        if self.inner_space:
            # Try direct mounted elements
            if hasattr(self.inner_space, '_mounted_elements') and element_id in self.inner_space._mounted_elements:
                return self.inner_space._mounted_elements[element_id]["element"]
                
            # Try elements requesting attention
            attention_elements = self.inner_space.get_elements_requesting_attention()
            for element_data in attention_elements.values():
                if "source_element_id" in element_data and element_data["source_element_id"] == element_id:
                    return element_data.get("element")
                
            # Ask spaces in inner_space
            spaces = self.registry.get_spaces()
            for space_id, space in spaces.items():
                if space_id != self.inner_space.id:
                    try:
                        # Try to find in mounted elements
                        element = space._find_element_by_id(element_id) if hasattr(space, '_find_element_by_id') else None
                        if element:
                            return element
                    except Exception as e:
                        self.logger.debug(f"Error searching for element in space {space_id}: {e}")
        
        return None
    
    def _handle_attention_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an attention event from the SpaceRegistry.
        
        Args:
            event: The attention event to handle
            timeline_context: Timeline context for the event
            
        Returns:
            Result of the attention handling
        """
        event_type = event.get('event_type')
        event_data = event.get('event_data', {})
        source_element_id = event.get('source_element_id')
        space_id = event.get('space_id')
        
        # Get event-specific timeline context if available, otherwise use the provided one
        event_timeline_context = event.get('timeline_context') or timeline_context
        validated_timeline_context = self._validate_timeline_context(event_timeline_context)
        
        self.logger.info(f"Handling attention event: {event_type} from element {source_element_id} "
                         f"in space {space_id}, timeline {validated_timeline_context.get('timeline_id', 'unknown')}")
        
        if event_type in ['attention_requested', 'inner_space_attention_requested']:
            # Mark that interaction is needed with this element
            event["needs_response"] = True
            event["response_target"] = source_element_id
            event["original_event_data"] = event_data
            event["timeline_context"] = validated_timeline_context
            
            # Begin the interaction cycle
            result = self._execute_interaction_cycle(event, validated_timeline_context)
            return result
        elif event_type in ['attention_cleared', 'inner_space_attention_cleared']:
            # No action needed, just acknowledge
            return {
                "status": "acknowledged",
                "event_type": event_type,
                "element_id": source_element_id,
                "space_id": space_id,
                "timeline_id": validated_timeline_context.get('timeline_id')
            }
        else:
            self.logger.warning(f"Unknown attention event type: {event_type}")
            return {
                "status": "error",
                "error": f"Unknown attention event type: {event_type}"
            }
            
    def _execute_interaction_cycle(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default implementation of the interaction cycle.
        
        Subclasses should override this for their specific interaction model.
        
        Args:
            event: Event that triggered this cycle
            timeline_context: Timeline context for this cycle
            
        Returns:
            Result of the interaction cycle
        """
        self.logger.info("Executing default interaction cycle")
        
        # Check if this event needs a response to a specific element
        response_target = event.get("response_target")
        
        # Render current context
        context = self.render_context(timeline_context)
        
        # Present context to agent
        agent_response = self.present_context_to_agent(context)
        
        # Parse agent actions
        actions = self.parse_agent_actions(agent_response)
        
        # If this event needs a response to a specific element, update that element
        result = {"actions_executed": []}
        if response_target:
            # Prepare response data
            response_data = {
                "content": agent_response.content,
                "original_event": event.get("original_event_data", event),
                "timestamp": int(time.time() * 1000)
            }
            
            # Update the element
            success = self.update_element_with_response(response_target, response_data, timeline_context)
            result["element_updated"] = success
            result["response_target"] = response_target
            
        # For all other actions, execute them
        for action in actions:
            action_result = self.execute_action(action, timeline_context)
            result["actions_executed"].append(action_result)
            
        return result

    def execute_action(self, action: Dict[str, Any], timeline_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action parsed from the agent's response.
        
        This method ensures that all element actions are routed through the Space layer
        to maintain proper timeline recording and state management.
        
        Args:
            action: Action to execute
            timeline_context: Timeline context for this action
            
        Returns:
            Result of the action execution
        """
        action_type = action.get("type")
        
        # Validate timeline context
        validated_timeline_context = self._validate_timeline_context(timeline_context)
        
        if action_type == "shell_tool":
            # Execute a shell tool
            tool_name = action.get("tool_name")
            parameters = action.get("parameters", {})
            
            if tool_name in self.internal_tools:
                try:
                    # Add timeline context to parameters if not already there
                    if "timeline_context" not in parameters:
                        parameters["timeline_context"] = validated_timeline_context
                        
                    handler = self.internal_tools[tool_name]["handler"]
                    result = handler(**parameters)
                    
                    # Record shell tool execution in inner space timeline
                    self._record_shell_tool_execution(tool_name, parameters, validated_timeline_context)
                    
                    return {
                        "type": "shell_tool",
                        "tool_name": tool_name,
                        "success": True,
                        "result": result
                    }
                except Exception as e:
                    self.logger.error(f"Error executing shell tool {tool_name}: {e}")
                    return {
                        "type": "shell_tool",
                        "tool_name": tool_name,
                        "success": False,
                        "error": str(e)
                    }
            else:
                return {
                    "type": "shell_tool",
                    "tool_name": tool_name,
                    "success": False,
                    "error": f"Unknown shell tool: {tool_name}"
                }
        
        elif action_type == "element_action":
            # Execute an action on an element
            element_id = action.get("element_id")
            action_name = action.get("action_name")
            parameters = action.get("parameters", {})
            space_id = action.get("space_id")
            
            # Add timeline context to parameters
            if "timeline_context" not in parameters:
                parameters["timeline_context"] = validated_timeline_context
            
            # First check if a space_id is specified
            if space_id:
                # Find the space
                space = self.registry.get_space(space_id)
                if space:
                    # Execute the action through the space
                    result = space.execute_action_on_element(element_id, action_name, parameters)
                    return {
                        "type": "element_action",
                        "element_id": element_id,
                        "space_id": space_id,
                        "action_name": action_name,
                        "success": "error" not in result,
                        "result": result
                    }
                else:
                    return {
                        "type": "element_action",
                        "element_id": element_id,
                        "action_name": action_name,
                        "success": False,
                        "error": f"Space not found: {space_id}"
                    }
            
            # If no space_id specified, find the element and its containing space
            element_info = self._find_element_with_space(element_id)
            if element_info:
                element, containing_space = element_info
                # Execute the action through the containing space
                result = containing_space.execute_action_on_element(element_id, action_name, parameters)
                return {
                    "type": "element_action",
                    "element_id": element_id,
                    "space_id": containing_space.id,
                    "action_name": action_name,
                    "success": "error" not in result,
                    "result": result
                }
            else:
                # Not found in any space, try inner space for elements without a container
                try:
                    result = self.inner_space.execute_action_on_element(element_id, action_name, parameters)
                    return {
                        "type": "element_action",
                        "element_id": element_id,
                        "space_id": self.inner_space.id,
                        "action_name": action_name,
                        "success": "error" not in result,
                        "result": result
                    }
                except Exception as e:
                    return {
                        "type": "element_action",
                        "element_id": element_id,
                        "action_name": action_name,
                        "success": False,
                        "error": f"Element not found: {element_id}"
                    }
        
        elif action_type == "message":
            # Handle a simple message (no tool calls)
            content = action.get("content")
            
            # If we have a response target, update it with this message
            response_target = action.get("response_target")
            if response_target:
                response_data = {
                    "content": content,
                    "event_type": "agent_message",
                    "timestamp": int(time.time() * 1000)
                }
                success = self.update_element_with_response(response_target, response_data, validated_timeline_context)
                return {
                    "type": "message",
                    "success": success,
                    "content_length": len(content) if content else 0
                }
            
            # Otherwise record the message in the inner space
            message_event = {
                "event_type": "agent_message",
                "content": content,
                "timestamp": int(time.time() * 1000)
            }
            self.inner_space.update_state(message_event, validated_timeline_context)
            
            self.logger.info(f"Agent message: {content}")
            return {
                "type": "message",
                "success": True,
                "content_length": len(content) if content else 0
            }
        
        else:
            return {
                "type": "unknown",
                "action_type": action_type,
                "success": False,
                "error": f"Unknown action type: {action_type}"
            }
            
    def _find_element_with_space(self, element_id: str) -> Optional[Tuple[Any, Any]]:
        """
        Find an element and its containing space.
        
        Args:
            element_id: ID of the element to find
            
        Returns:
            Tuple of (element, containing_space) if found, None otherwise
        """
        # Check all known spaces in registry
        for space in self.registry.get_all_spaces():
            element = space.get_mounted_element(element_id)
            if element:
                return (element, space)
                
        return None
        
    def _record_shell_tool_execution(self, tool_name: str, parameters: Dict[str, Any], 
                                   timeline_context: Dict[str, Any]) -> None:
        """
        Record shell tool execution in the inner space timeline.
        
        Args:
            tool_name: Name of the shell tool
            parameters: Tool parameters
            timeline_context: Timeline context
        """
        tool_event = {
            "event_type": "shell_tool_execution",
            "tool_name": tool_name,
            "parameters": {k: v for k, v in parameters.items() if k != "timeline_context"},
            "timestamp": int(time.time() * 1000)
        }
        
        self.inner_space.update_state(tool_event, timeline_context)

    def get_scene_graph(self, format: str = "text") -> str:
        """
        Get a visualization of the current scene graph.
        
        Args:
            format: Output format (text, markdown, or json)
            
        Returns:
            String representation of the scene graph
        """
        # Ensure we have a current scene graph
        if not hasattr(self.hud, 'current_scene_graph') or not self.hud.current_scene_graph:
            # Trigger a rendering to build the scene graph
            self.render_context(self.current_timeline_context)
            
        # Now visualize it
        return self.hud.visualize_scene_graph(format)
    
    def get_element_relationships(self, element_id: str) -> Dict[str, List[str]]:
        """
        Get all relationships for a specific element in the scene graph.
        
        Args:
            element_id: ID of the element to get relationships for
            
        Returns:
            Dictionary with 'parents', 'children', and 'references' lists
        """
        # Ensure we have a current scene graph
        if not hasattr(self.hud, 'current_scene_graph') or not self.hud.current_scene_graph:
            # Trigger a rendering to build the scene graph
            self.render_context(self.current_timeline_context)
            
        return self.hud.get_element_relationships(element_id) 