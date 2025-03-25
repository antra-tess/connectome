"""
Two-Phase Shell Implementation

Implements a shell that uses a two-phase interaction model (contemplation and engagement).
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import time

from bot_framework.shell.base_shell import BaseShell
from bot_framework.elements.space_registry import SpaceRegistry
from bot_framework.shell.hud.core import HUD
from bot_framework.shell.context_manager import ContextManager
from bot_framework.elements.elements.space import Space
from bot_framework.llm import LLMResponse, LLMMessage


class TwoPhaseShell(BaseShell):
    """
    Two-Phase Shell implementation.
    
    This shell uses a two-phase interaction model:
    1. Contemplation Phase: The agent thinks through the context without producing actions
    2. Engagement Phase: The agent produces actions based on its contemplation
    
    This model allows for more thorough reasoning before taking action.
    """
    
    def __init__(self,
                 registry: SpaceRegistry,
                 hud: Optional[HUD] = None,
                 context_manager: Optional[ContextManager] = None,
                 inner_space: Optional[Space] = None,
                 model_info: Dict[str, Any] = None,
                 llm_config: Optional[Dict[str, Any]] = None):
        """Initialize the Two-Phase Shell."""
        super().__init__(registry, hud, context_manager, inner_space, model_info, llm_config)
        
        # Phase tracking
        self.current_phase = "contemplation"  # Start in contemplation phase
        self.contemplation_result = None
        
        # Additional tools specific to this shell
        self._setup_two_phase_tools()
        
        self.logger.info("Two-Phase Shell initialized")
    
    def _setup_two_phase_tools(self):
        """Set up tools specific to the two-phase shell."""
        # Add phase transition tool
        self.internal_tools["transition_to_engagement"] = {
            "name": "transition_to_engagement",
            "description": "Transition from contemplation to engagement phase",
            "parameters": {},
            "handler": self._handle_transition_to_engagement
        }
        
        self.internal_tools["take_notes"] = {
            "name": "take_notes",
            "description": "Save notes during contemplation phase",
            "parameters": {
                "notes": "The notes to save"
            },
            "handler": self._handle_take_notes
        }
    
    def _handle_transition_to_engagement(self) -> str:
        """Handle transition from contemplation to engagement phase."""
        if self.current_phase == "contemplation":
            self.current_phase = "engagement"
            self.logger.info("Transitioning to engagement phase")
            return "Transitioned to engagement phase."
        else:
            return "Already in engagement phase."
    
    def _handle_take_notes(self, notes: str) -> str:
        """Handle taking notes during contemplation."""
        if self.current_phase == "contemplation":
            self.logger.info("Saving contemplation notes")
            self.contemplation_result = notes
            return "Notes saved."
        else:
            return "Notes can only be taken during contemplation phase."
    
    def _execute_interaction_cycle(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a two-phase interaction cycle.
        
        Args:
            event: The event that triggered this cycle
            timeline_context: The timeline context for this cycle
            
        Returns:
            Result of the interaction cycle
        """
        self.logger.info(f"Starting interaction cycle in phase: {self.current_phase}")
        
        # Reset phase to contemplation at the start of each cycle
        self.current_phase = "contemplation"
        self.contemplation_result = None
        
        # Render context
        context = self.render_context(timeline_context)
        
        # --- Contemplation Phase ---
        
        # Present context with contemplation prompt
        contemplation_context = self._prepare_contemplation_context(context)
        
        # Use multi-turn tool execution for contemplation phase
        contemplation_response, contemplation_tool_results = self.execute_multi_turn_with_tools(
            contemplation_context, 
            timeline_context
        )
        
        # Check if agent requested transition to engagement
        if not self.contemplation_result and self.current_phase == "contemplation":
            # No explicit transition, use the response as contemplation result
            self.contemplation_result = contemplation_response.content or "No specific conclusions reached during contemplation."
            self.current_phase = "engagement"
        
        # Record contemplation as an event in the timeline
        if timeline_context.get("is_primary", False):
            contemplation_event = {
                "type": "agent_contemplation", 
                "content": self.contemplation_result,
                "in_response_to": event.get("id"),
                "timestamp": int(time.time() * 1000)
            }
            self.inner_space.add_event_to_timeline(contemplation_event, timeline_context)
        
        # --- Engagement Phase ---
        
        # Present context with engagement prompt and contemplation result
        engagement_context = self._prepare_engagement_context(context, self.contemplation_result)
        
        # Use multi-turn tool execution for engagement phase
        engagement_response, engagement_tool_results = self.execute_multi_turn_with_tools(
            engagement_context, 
            timeline_context
        )
        
        # Record engagement response as an event in the timeline
        if engagement_response.content and timeline_context.get("is_primary", False):
            engagement_event = {
                "type": "agent_response",
                "content": engagement_response.content,
                "preceded_by_contemplation": True,
                "in_response_to": event.get("id"),
                "timestamp": int(time.time() * 1000)
            }
            self.inner_space.add_event_to_timeline(engagement_event, timeline_context)
            
            # If this event needs a response, update the requesting element
            if event.get("needs_response") and event.get("response_target"):
                # Update the element that requested attention
                target_element_id = event.get("response_target")
                self.update_element_with_response(
                    target_element_id, 
                    engagement_response.content, 
                    event, 
                    timeline_context
                )
        
        # Combine all tool results
        all_tool_results = contemplation_tool_results + engagement_tool_results
        
        # Reset phase for next cycle
        self.current_phase = "contemplation"
        
        return {
            "contemplation_result": self.contemplation_result,
            "engagement_response": engagement_response.content,
            "tool_results": all_tool_results
        }
    
    def _prepare_contemplation_context(self, base_context: str) -> str:
        """
        Prepare the context for the contemplation phase.
        
        Args:
            base_context: The base rendered context
            
        Returns:
            Context for contemplation phase
        """
        contemplation_prefix = (
            "# CONTEMPLATION PHASE\n\n"
            "You are in the contemplation phase. In this phase, you should:\n"
            "1. Analyze the context thoroughly\n"
            "2. Consider various approaches and their implications\n"
            "3. Develop a plan for actions you will take\n"
            "4. Use the take_notes tool to save your thoughts\n\n"
            "Once you have thoroughly analyzed the situation, you can use the "
            "transition_to_engagement tool to move to the engagement phase where "
            "you will take concrete actions.\n\n"
            "## Context:\n\n"
        )
        
        return contemplation_prefix + base_context
    
    def _prepare_engagement_context(self, base_context: str, contemplation_result: str) -> str:
        """
        Prepare the context for the engagement phase.
        
        Args:
            base_context: The base rendered context
            contemplation_result: The result of the contemplation phase
            
        Returns:
            Context for engagement phase
        """
        engagement_prefix = (
            "# ENGAGEMENT PHASE\n\n"
            "You are in the engagement phase. In this phase, you should:\n"
            "1. Take concrete actions based on your contemplation\n"
            "2. Execute your plan developed during contemplation\n"
            "3. Respond to the user with clear, actionable information\n\n"
            "## Your Contemplation:\n\n"
            f"{contemplation_result}\n\n"
            "## Context:\n\n"
        )
        
        return engagement_prefix + base_context
    
    def present_context_to_agent(self, context: str) -> str:
        """
        Present the prepared context to the agent and get a response.
        
        This implementation would use the LLM to generate a response.
        
        Args:
            context: The prepared context to present
            
        Returns:
            The agent's response
        """
        self.logger.info(f"Presenting context to agent in phase: {self.current_phase}")
        
        # This is a placeholder that would be implemented with actual LLM integration
        # In a real implementation, this would call the LLM service
        
        # For development/testing, return a mock response
        phase = self.current_phase
        if phase == "contemplation":
            return (
                "I've analyzed the situation carefully. Based on the context, I need to...\n\n"
                "take_notes(notes=\"I should respond to the user's query about file organization "
                "by first examining the current structure, then suggesting improvements.\")"
            )
        else:  # engagement phase
            return (
                "Based on my analysis, I'll help you organize your files.\n\n"
                "First, let's create a directory structure that separates concerns:"
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
        
        # Check for tool calls using simple regex pattern
        tool_pattern = r'(\w+)\(([^)]*)\)'
        
        # Find all potential tool calls
        matches = re.findall(tool_pattern, agent_response)
        
        for match in matches:
            tool_name = match[0]
            params_str = match[1]
            
            # Check if this is a recognized internal tool
            if tool_name in self.internal_tools:
                # Parse parameters
                params = {}
                if params_str:
                    # Very basic parameter parsing - in a real implementation, this would be more robust
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
        
        # If we're in engagement phase and no actions were found,
        # treat the entire response as a message action
        if self.current_phase == "engagement" and not actions:
            actions.append({
                "type": "message",
                "content": agent_response
            })
        
        return actions 

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