"""
Prompt Manager
Manages system prompts for the agent, including customization and formatting.
"""

import logging
from typing import Dict, Any, Optional

from config import AGENT_NAME, AGENT_DESCRIPTION
from interface.prompt_library.base_prompts import (
    DEFAULT_SYSTEM_PROMPT,
    CONVERSATION_GUIDELINES,
    SAFETY_GUARDRAILS
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages system prompts for the agent.
    
    Responsible for constructing the base system prompt that defines the agent's
    behavior, personality, and capabilities.
    """
    
    def __init__(self, custom_prompt: Optional[str] = None):
        """
        Initialize the prompt manager with an optional custom prompt.
        
        Args:
            custom_prompt: Optional custom system prompt to use instead of the default
        """
        self.custom_prompt = custom_prompt
        self.environment_renderer = None  # Will be set by set_environment_renderer
        logger.info("Prompt manager initialized")
    
    def set_environment_renderer(self, environment_renderer):
        """
        Set the environment renderer for adding environment capabilities to prompts.
        
        Args:
            environment_renderer: The environment renderer instance
        """
        self.environment_renderer = environment_renderer
        logger.debug("Environment renderer set in prompt manager")
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the agent.
        
        This combines the base prompt with environment capabilities if available.
        
        Returns:
            Formatted system prompt
        """
        # Start with either custom or default prompt
        base_prompt = self.custom_prompt if self.custom_prompt else self._build_default_prompt()
        
        # Enhance with environment capabilities if available
        if self.environment_renderer:
            # Use the environment renderer to add capabilities information
            enhanced_prompt = self.environment_renderer.format_prompt_with_environment(base_prompt)
            logger.debug("Enhanced system prompt with environment capabilities")
            return enhanced_prompt
        
        # Return the base prompt if no environment renderer is available
        return base_prompt
    
    def _build_default_prompt(self) -> str:
        """
        Build the default system prompt using components from the prompt library.
        
        Returns:
            Default system prompt
        """
        return DEFAULT_SYSTEM_PROMPT.format(
            agent_name=AGENT_NAME,
            agent_description=AGENT_DESCRIPTION,
            conversation_guidelines=CONVERSATION_GUIDELINES,
            safety_guardrails=SAFETY_GUARDRAILS
        )
    
    def update_custom_prompt(self, new_prompt: str) -> None:
        """
        Update the custom prompt.
        
        Args:
            new_prompt: New custom prompt to use
        """
        self.custom_prompt = new_prompt
        logger.info("Custom prompt updated")
        
    def reset_to_default_prompt(self) -> None:
        """
        Reset to the default system prompt.
        """
        self.custom_prompt = None
        logger.info("Reset to default prompt")
        
    def enhance_prompt_with_context(self, prompt: str, context_data: Dict[str, Any]) -> str:
        """
        Enhance the prompt with additional context data.
        
        Args:
            prompt: Base prompt to enhance
            context_data: Dictionary of context data to add to the prompt
            
        Returns:
            Enhanced prompt with context data
        """
        enhanced_prompt = prompt
        
        # Add any relevant context data to the prompt
        if 'conversation_summary' in context_data:
            enhanced_prompt += f"\n\nCONVERSATION SUMMARY:\n{context_data['conversation_summary']}"
            
        if 'user_preferences' in context_data:
            enhanced_prompt += f"\n\nUSER PREFERENCES:\n{context_data['user_preferences']}"
            
        if 'platform_info' in context_data:
            enhanced_prompt += f"\n\nPLATFORM INFORMATION:\n{context_data['platform_info']}"
            
        return enhanced_prompt 