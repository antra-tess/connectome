"""
Prompt Manager
Manages system prompts for the agent, including customization and formatting.
"""

import logging
from typing import Dict, Any, Optional

from config import AGENT_NAME, AGENT_DESCRIPTION
from agent.prompt_library.base_prompts import (
    DEFAULT_SYSTEM_PROMPT,
    CONVERSATION_GUIDELINES,
    SAFETY_GUARDRAILS
)
from agent.prompt_library.tool_prompts import TOOL_USAGE_GUIDELINES

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
        logger.info("Prompt manager initialized")
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the agent.
        
        Returns:
            Formatted system prompt
        """
        if self.custom_prompt:
            return self.custom_prompt
        
        return self._build_default_prompt()
    
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
            tool_usage_guidelines=TOOL_USAGE_GUIDELINES,
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