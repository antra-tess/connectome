"""
Environment Renderer Component

Handles the rendering of environments for the agent.
"""

import logging
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


class EnvironmentRenderer:
    """
    Renders environments for agent consumption.
    
    This component is responsible for:
    1. Rendering environment states for the agent
    2. Formatting environment capabilities and tools
    3. Creating a coherent view of the environment hierarchy
    """
    
    def __init__(self, environment_manager):
        """
        Initialize the environment renderer.
        
        Args:
            environment_manager: The environment manager instance to use
        """
        self.environment_manager = environment_manager
    
    def render_environment(self, env_id: str) -> Dict[str, Any]:
        """
        Render an environment for agent consumption.
        
        Args:
            env_id: ID of the environment to render
            
        Returns:
            Rendered environment
        """
        try:
            # Get the environment
            environment = self.environment_manager.get_environment(env_id)
            
            if not environment:
                return {
                    "result": "error",
                    "message": f"Environment '{env_id}' not found"
                }
            
            # Render the environment
            rendered = environment.render()
            
            return {
                "result": "success",
                "environment": rendered
            }
        except Exception as e:
            logger.error(f"Error rendering environment '{env_id}': {str(e)}")
            return {
                "result": "error",
                "message": f"Failed to render environment: {str(e)}"
            }
    
    def get_environment_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the state of all environments.
        
        Returns:
            Dictionary mapping environment IDs to their states
        """
        states = {}
        
        try:
            # Get all environments
            environments = self.environment_manager.get_all_environments()
            
            # Get state of each environment
            for env_id, environment in environments.items():
                states[env_id] = environment.get_state()
                
            return states
        except Exception as e:
            logger.error(f"Error getting environment states: {str(e)}")
            return {}
    
    def render_for_agent(self) -> Dict[str, Any]:
        """
        Render all environments for agent consumption.
        
        Returns:
            Rendered environments
        """
        try:
            # Get all environments
            environments = self.environment_manager.get_all_environments()
            
            # Render each environment
            rendered = {}
            for env_id, environment in environments.items():
                rendered[env_id] = environment.render_state_for_context()
                
            return {
                "result": "success",
                "environments": rendered
            }
        except Exception as e:
            logger.error(f"Error rendering environments for agent: {str(e)}")
            return {
                "result": "error",
                "message": f"Failed to render environments: {str(e)}"
            }
    
    def format_prompt_with_environment(self, base_prompt: str) -> str:
        """
        Format a prompt with environment information.
        
        Args:
            base_prompt: Base prompt to append environment information to
            
        Returns:
            Formatted prompt
        """
        try:
            # Get system environment (it contains all mounted environments)
            system_env = self.environment_manager.get_environment("system")
            
            if not system_env:
                logger.error("System environment not found")
                return base_prompt
            
            # Get rendered environment
            env_prompt = system_env.render_capabilities_for_context()
            
            # Combine prompts
            return f"{base_prompt}\n\n{env_prompt}"
        except Exception as e:
            logger.error(f"Error formatting prompt with environment: {str(e)}")
            return base_prompt 