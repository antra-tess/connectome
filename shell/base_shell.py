"""
Base Shell

Defines the abstract base class for Shell implementations, which serves as the 
agentic loop container enclosing the model mind, providing a runtime environment 
and coordination layer.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Callable

from bot_framework.shell.hud.core import HUD, RenderingRequest, RenderingResponse
from bot_framework.shell.context_manager import ContextManager, ContextRequest, ContextResponse
from bot_framework.elements.elements.space import Space
from bot_framework.elements.space_registry import SpaceRegistry
from bot_framework.elements.elements.inner_space import InnerSpace
from bot_framework.llm import (
    LLMProvider, 
    LLMProviderFactory,
    LLMMessage, 
    LLMToolDefinition,
    LLMToolCall,
    LLMResponse
)


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
        
        # Create or use existing inner space
        self.inner_space = inner_space or self._create_inner_space(registry)
        
        # Initialize the agent's subjective timeline
        self.current_timeline_context = {
            "branchId": "primary-branch-001",
            "isPrimary": True,
            "lastEventId": None
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
        from bot_framework.shell.context_manager.core import ContextManager
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
        # InnerSpace will register itself with the registry
        return InnerSpace("inner_space", "Agent's Inner Space", registry)
    
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
        
        # Use provided timeline context or default to primary
        timeline_context = timeline_context or self.current_timeline_context
        
        try:
            # Mark that we're processing an event
            self.is_processing_event = True
            
            # Update inner space with event
            self.inner_space.receive_event(event, timeline_context)
            
            # Begin the interaction cycle
            result = self._execute_interaction_cycle(event, timeline_context)
            
            # Done processing
            self.is_processing_event = False
            
            return result
        except Exception as e:
            self.logger.error(f"Error handling event: {e}", exc_info=True)
            self.is_processing_event = False
            return {"error": str(e)}
    
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
        
        # Prepare rendering request
        request = RenderingRequest(timeline_context=timeline_context)
        
        # Get rendering from HUD
        rendering_response = self.hud.prepare_context_rendering(request)
        
        # Pass to Context Manager for final assembly
        context_request = ContextRequest(
            rendering_response=rendering_response,
            timeline_context=timeline_context
        )
        
        context_response = self.context_manager.assemble_context(context_request)
        
        return context_response.context
    
    def present_context_to_agent(self, context: str) -> LLMResponse:
        """
        Present the prepared context to the agent and get a response.
        
        This method now returns a standardized LLMResponse rather than a string,
        allowing for better handling of tool calls.
        
        Args:
            context: The prepared context to present
            
        Returns:
            The agent's response as an LLMResponse object
        """
        self.logger.info("Presenting context to agent")
        
        # Create a user message with the context
        messages = [LLMMessage(role="user", content=context)]
        
        # Get available tools
        tools = self.get_available_tools_as_llm_tools()
        
        # Call the LLM with the context and tools
        response = self.llm_provider.complete(
            messages=messages,
            tools=tools,
            temperature=self.model_info.get("temperature", 0.7),
            max_tokens=self.model_info.get("max_tokens", 2000)
        )
        
        self.logger.info(
            f"Agent response received. Content length: "
            f"{len(response.content or '')}. Tool calls: {len(response.tool_calls)}"
        )
        
        return response
    
    def execute_multi_turn_with_tools(self, 
                                     initial_context: str, 
                                     timeline_context: Dict[str, Any],
                                     max_turns: int = None) -> Tuple[LLMResponse, List[Dict[str, Any]]]:
        """
        Execute a multi-turn conversation with tool calls and responses.
        
        This method handles the sequential execution of tool calls:
        1. Present initial context to agent
        2. If agent calls tools, execute them and add results to conversation
        3. Present the updated conversation with tool results back to agent
        4. Continue until agent gives a final response without tool calls or max_turns is reached
        
        Args:
            initial_context: The initial context to present
            timeline_context: The timeline context
            max_turns: Maximum number of turns to allow
            
        Returns:
            Tuple of (final_response, tool_results)
        """
        max_turns = max_turns or self.max_sequential_tool_calls
        
        self.logger.info(f"Starting multi-turn conversation with max {max_turns} turns")
        
        # Initialize conversation with user context
        messages = [LLMMessage(role="user", content=initial_context)]
        
        # Get available tools
        tools = self.get_available_tools_as_llm_tools()
        
        # Track tool results
        tool_results = []
        
        # Tool calling loop
        for turn in range(max_turns):
            self.logger.info(f"Turn {turn+1}/{max_turns}")
            
            # Call LLM
            response = self.llm_provider.complete(
                messages=messages,
                tools=tools,
                temperature=self.model_info.get("temperature", 0.7),
                max_tokens=self.model_info.get("max_tokens", 2000)
            )
            
            # If no tool calls, we're done
            if not response.tool_calls:
                self.logger.info("No tool calls, ending conversation")
                return response, tool_results
            
            # Process tool calls
            for tool_call in response.tool_calls:
                self.logger.info(f"Executing tool: {tool_call.tool_name}")
                
                # Add assistant message with tool call
                messages.append(LLMMessage(
                    role="assistant",
                    content=response.content,
                ))
                
                # Execute tool
                action = self._convert_tool_call_to_action(tool_call)
                result = self.execute_action(action, timeline_context)
                tool_results.append(result)
                
                # Format result for conversation
                result_str = json.dumps(result)
                
                # Add function response to conversation
                messages.append(LLMMessage(
                    role="function",
                    name=tool_call.tool_name,
                    content=result_str
                ))
        
        # If we reached max turns, return the last response
        self.logger.info(f"Reached maximum turns ({max_turns}), ending conversation")
        return response, tool_results
    
    def _convert_tool_call_to_action(self, tool_call: LLMToolCall) -> Dict[str, Any]:
        """
        Convert a tool call to an action that can be executed.
        
        Args:
            tool_call: The tool call to convert
            
        Returns:
            Action dictionary
        """
        tool_name = tool_call.tool_name
        parameters = tool_call.parameters
        
        # Check if this is an element action (contains a dot)
        if "." in tool_name:
            element_id, action_name = tool_name.split(".", 1)
            return {
                "type": "element_action",
                "element_id": element_id,
                "action_name": action_name,
                "parameters": parameters
            }
        else:
            # Shell tool
            return {
                "type": "shell_tool",
                "tool_name": tool_name,
                "parameters": parameters
            }
    
    def get_available_tools_as_llm_tools(self) -> List[LLMToolDefinition]:
        """
        Get all available tools in LLMToolDefinition format.
        
        Returns:
            List of tool definitions
        """
        llm_tools = []
        
        # Add internal tools
        for name, tool in self.internal_tools.items():
            llm_tools.append(LLMToolDefinition(
                name=name,
                description=tool.get("description", ""),
                parameters=tool.get("parameters", {})
            ))
        
        # Add element tools
        # In a real implementation, this would collect tools from mounted elements
        # For now, this is just a placeholder
        
        return llm_tools
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available tools (both internal and from elements).
        
        Returns:
            List of tool descriptions
        """
        tools = []
        
        # Add internal tools
        for name, tool in self.internal_tools.items():
            tools.append({
                "name": name,
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {})
            })
        
        # Add element tools
        # This would collect tools from mounted elements in the inner space
        # Implementation depends on how elements expose their tools
        
        return tools
    
    def parse_agent_actions(self, agent_response: LLMResponse) -> List[Dict[str, Any]]:
        """
        Parse the agent's response into executable actions.
        
        Args:
            agent_response: The LLM response
            
        Returns:
            List of parsed actions
        """
        self.logger.info("Parsing agent actions")
        actions = []
        
        # Handle tool calls if present
        if agent_response.tool_calls:
            for tool_call in agent_response.tool_calls:
                actions.append(self._convert_tool_call_to_action(tool_call))
        
        # If no tool calls but there's content, add a message action
        elif agent_response.content:
            actions.append({
                "type": "message",
                "content": agent_response.content
            })
        
        return actions
    
    def execute_action(self, action: Dict[str, Any], timeline_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a parsed action.
        
        Args:
            action: The action to execute
            timeline_context: The timeline context
            
        Returns:
            Result of the action execution
        """
        action_type = action.get("type", "unknown")
        self.logger.info(f"Executing action of type: {action_type}")
        
        # Record agent action in timeline
        self._record_agent_action_in_timeline(action, timeline_context)
        
        if action_type == "element_action":
            # Execute an action on an element
            return self._execute_element_action(action, timeline_context)
        elif action_type == "shell_tool":
            # Execute an internal shell tool
            return self._execute_shell_tool(action, timeline_context)
        elif action_type == "message":
            # Send a message
            return self._execute_message_action(action, timeline_context)
        else:
            self.logger.warning(f"Unknown action type: {action_type}")
            return {"error": f"Unknown action type: {action_type}"}
    
    def _record_agent_action_in_timeline(self, action: Dict[str, Any], timeline_context: Dict[str, Any]):
        """
        Record an agent action as an event in the timeline DAG.
        
        Args:
            action: The action to record
            timeline_context: The timeline context
        """
        self.logger.info(f"Recording agent action in timeline: {action.get('type', 'unknown')}")
        
        # Create an event representing the agent action
        agent_action_event = {
            "type": "agent_action",
            "action": action,
            "timestamp": self._get_current_timestamp()
        }
        
        # Add to timeline in inner space
        self.inner_space.add_event_to_timeline(agent_action_event, timeline_context)
    
    def _execute_element_action(self, action: Dict[str, Any], timeline_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action on an element.
        
        Args:
            action: The element action to execute
            timeline_context: The timeline context
            
        Returns:
            Result of the action execution
        """
        space_id = action.get("space_id")
        element_id = action.get("element_id")
        action_name = action.get("action_name")
        parameters = action.get("parameters", {})
        
        self.logger.info(f"Executing element action: {action_name} on {space_id}/{element_id}")
        
        # Use inner_space to execute the action
        return self.inner_space.execute_element_action(
            space_id, element_id, action_name, parameters, timeline_context
        )
    
    def _execute_shell_tool(self, action: Dict[str, Any], timeline_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an internal shell tool.
        
        Args:
            action: The shell tool action to execute
            timeline_context: The timeline context
            
        Returns:
            Result of the tool execution
        """
        tool_name = action.get("tool_name")
        parameters = action.get("parameters", {})
        
        self.logger.info(f"Executing shell tool: {tool_name}")
        
        # Check if tool exists
        if tool_name not in self.internal_tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        # Get the tool handler
        handler = self.internal_tools[tool_name]["handler"]
        
        # Execute the tool
        try:
            result = handler(**parameters)
            return {"result": result}
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": f"Error executing tool {tool_name}: {e}"}
    
    def _execute_message_action(self, action: Dict[str, Any], timeline_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a message action by sending it through the Space.
        
        Args:
            action: The message action to execute
            timeline_context: The timeline context
            
        Returns:
            Result of the action execution
        """
        content = action.get("content", "")
        destination = action.get("destination", "default")
        
        self.logger.info(f"Sending message to {destination}: {content[:50]}...")
        
        # Only propagate externally if we're in the primary timeline
        if timeline_context.get("isPrimary", False):
            message = {
                "type": "message",
                "content": content,
                "destination": destination,
                "timestamp": self._get_current_timestamp()
            }
            
            # Send through Inner Space, which will propagate to Activity Layer
            # This follows the proper flow: Shell -> Space -> ActivityLayer -> External
            self.inner_space.send_message(message, timeline_context)
            
            return {"result": "Message sent"}
        else:
            self.logger.info("Message not propagated (non-primary timeline)")
            return {"result": "Message not propagated (non-primary timeline)"}
    
    def _get_current_timestamp(self) -> int:
        """Get the current timestamp in milliseconds."""
        import time
        return int(time.time() * 1000) 