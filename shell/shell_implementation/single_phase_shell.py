"""
Single-Phase Shell Implementation

Implements a shell that uses a single-phase interaction model.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import time
import re

from shell.base_shell import BaseShell
from elements.space_registry import SpaceRegistry
from shell.hud.core import HUD
from shell.context_manager import ContextManager
from elements.elements.space import Space
from llm import LLMResponse, LLMMessage


class SinglePhaseShell(BaseShell):
    """
    Single-Phase Shell implementation.
    
    This shell uses a single-phase interaction model where the agent
    directly responds to context with actions, without a separate
    contemplation phase.
    """
    
    def __init__(self,
                 registry: SpaceRegistry,
                 hud: Optional[HUD] = None,
                 context_manager: Optional[ContextManager] = None,
                 inner_space: Optional[Space] = None,
                 model_info: Dict[str, Any] = None,
                 llm_config: Optional[Dict[str, Any]] = None):
        """Initialize the Single-Phase Shell."""
        super().__init__(registry, hud, context_manager, inner_space, model_info, llm_config)
        
        # Additional tools specific to this shell
        self._setup_single_phase_tools()
        
        self.logger.info("Single-Phase Shell initialized")
    
    def _setup_single_phase_tools(self):
        """Set up tools specific to the single-phase shell."""
        # No additional tools needed for the basic single-phase shell
        pass
    
    def _execute_interaction_cycle(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single-phase interaction cycle.
        
        Args:
            event: The event that triggered this cycle
            timeline_context: The timeline context for this cycle
            
        Returns:
            Result of the interaction cycle
        """
        self.logger.info("Starting single-phase interaction cycle")
        
        # Render context
        context = self.render_context(timeline_context)
        
        # Prepare context for agent
        prepared_context = self._prepare_context(context)
        
        # Use multi-turn tool execution
        agent_response, tool_results = self.execute_multi_turn_with_tools(
            prepared_context, 
            timeline_context
        )
        
        # Record the agent's final response as an event if appropriate
        if agent_response.content and timeline_context.get("is_primary", False):
            # Create response event for recording in timeline
            response_event = {
                "type": "agent_response",
                "content": agent_response.content,
                "in_response_to": event.get("id"),
                "timestamp": int(time.time() * 1000)
            }
            
            # Add to timeline in inner space
            self.inner_space.add_event_to_timeline(response_event, timeline_context)
            
            # If this event needs a response, update the requesting element
            if event.get("needs_response") and event.get("response_target"):
                # Update the element that requested attention
                target_element_id = event.get("response_target")
                self.update_element_with_response(
                    target_element_id, 
                    agent_response.content, 
                    event, 
                    timeline_context
                )
        
        return {
            "agent_response": agent_response.content,
            "tool_results": tool_results
        }
    
    def _prepare_context(self, base_context: str) -> str:
        """
        Prepare the context for the agent.
        
        Adds any necessary instructions or metadata to the context.
        """
        return (
            "# SINGLE-PHASE INTERACTION\n\n"
            "Analyze the context and respond with appropriate actions.\n"
            "You can directly execute actions as needed.\n\n"
            "## Context:\n\n" + base_context
        )
    
    def present_context_to_agent(self, context: str) -> str:
        """
        Present the prepared context to the agent and get a response.
        
        This implementation would use the LLM to generate a response.
        
        Args:
            context: The prepared context to present
            
        Returns:
            The agent's response
        """
        self.logger.info("Presenting context to agent")
        
        # This is a placeholder that would be implemented with actual LLM integration
        # In a real implementation, this would call the LLM service
        
        # For development/testing, return a mock response
        return (
            "Based on the provided context, I'll help resolve this issue.\n\n"
            "First, I need to check the file permissions:\n"
            "run_command(command=\"ls -la /path/to/file\")\n\n"
            "Then I'll create a directory for the backup:\n"
            "create_directory(path=\"/backups/data\")"
        )
    
    def parse_agent_actions(self, agent_response: str) -> List[Dict[str, Any]]:
        """
        Parse the agent's response into executable actions.
        
        This implementation focuses on extracting tool calls and message content.
        
        Args:
            agent_response: The raw response from the agent
            
        Returns:
            List of parsed actions
        """
        self.logger.info("Parsing agent actions")
        
        actions = []
        
        # Extract potential tool calls
        tool_pattern = r'(\w+)\(([^)]*)\)'
        
        # Find all potential tool calls
        matches = re.findall(tool_pattern, agent_response)
        
        for match in matches:
            tool_name = match[0]
            params_str = match[1]
            
            # Check if this is a recognized tool
            if tool_name in self.internal_tools:
                # Parse parameters
                params = {}
                if params_str:
                    # Basic parameter parsing
                    for param in params_str.split(','):
                        if '=' in param:
                            key, value = param.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # Remove quotes if present
                            if (value.startswith('"') and value.endswith('"')) or \
                               (value.startswith("'") and value.endswith("'")):
                                value = value[1:-1]
                                
                            params[key] = value
                
                # Add as a shell tool action
                actions.append({
                    "type": "shell_tool",
                    "tool_name": tool_name,
                    "parameters": params
                })
            else:
                # This might be an element action
                # Format: element_id.action_name(params)
                element_match = re.match(r'(\w+)\.(\w+)', tool_name)
                if element_match:
                    element_id = element_match.group(1)
                    action_name = element_match.group(2)
                    
                    actions.append({
                        "type": "element_action",
                        "element_id": element_id,
                        "action_name": action_name,
                        "parameters": self._parse_parameters(params_str)
                    })
        
        # If no actions were parsed, treat the entire response as a message
        if not actions:
            actions.append({
                "type": "message",
                "content": agent_response
            })
        
        return actions
    
    def _parse_parameters(self, params_str: str) -> Dict[str, Any]:
        """Parse parameters from a string into a dictionary."""
        params = {}
        if not params_str:
            return params
            
        # Split by commas, but respect quotes
        in_quotes = False
        quote_char = None
        current_param = ""
        
        for char in params_str:
            if char in ['"', "'"]:
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                current_param += char
            elif char == ',' and not in_quotes:
                if current_param:
                    self._add_param_to_dict(current_param.strip(), params)
                    current_param = ""
            else:
                current_param += char
                
        # Don't forget the last parameter
        if current_param:
            self._add_param_to_dict(current_param.strip(), params)
            
        return params
    
    def _add_param_to_dict(self, param_str: str, params_dict: Dict[str, Any]):
        """Add a parameter to the parameter dictionary."""
        if '=' in param_str:
            key, value = param_str.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Remove quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
                
            # Try to parse as JSON if it looks like a complex value
            if value.startswith('{') or value.startswith('['):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass  # Keep as string if not valid JSON
                    
            params_dict[key] = value 

    def _should_send_external_message(self, event: Dict[str, Any]) -> bool:
        """
        Determine if the agent's response should be sent as an external message.
        
        Args:
            event: The event that triggered the response
            
        Returns:
            True if response should be sent externally, False otherwise
        """
        # Check if this was a message event that expects a response
        event_type = event.get("type")
        return event_type in ["message_received", "user_message", "message_text"] 