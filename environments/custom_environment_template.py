"""
Custom Environment Template

This template can be copied to create new environment classes.
To create a new environment:
1. Copy this file with a descriptive name (e.g., my_environment.py) in environments/environment_classes/
2. Rename the class to match your environment name
3. Implement the required methods
4. Add any custom tools using the register_tool method
5. Update the configuration to use your new environment

Your environment will be automatically discovered by the framework.
"""

from environments.base import Environment
from typing import Dict, Any, Optional, Callable


class CustomEnvironmentTemplate(Environment):
    """
    Template for creating custom environments.
    
    This class serves as a starting point for creating new environment types.
    """
    
    def __init__(self, env_id: Optional[str] = None, name: str = "Custom Environment",
                 description: str = "A custom environment template"):
        """
        Initialize the custom environment.
        
        Args:
            env_id: Optional identifier for this environment instance
            name: Human-readable name for this environment
            description: Description of what this environment does
        """
        # Always call the parent class constructor first
        super().__init__(env_id=env_id, name=name, description=description)
        
        # Register any tools this environment provides
        self._register_tools()
    
    def _register_tools(self):
        """
        Register tools provided by this environment.
        """
        # Example tool registration:
        self.register_tool(
            tool_func=self.example_tool,
            name="example_tool",
            description="An example tool that does something useful",
            parameter_descriptions={
                "param1": "Description of parameter 1",
                "param2": "Description of parameter 2"
            }
        )
    
    def example_tool(self, param1: str, param2: int = 0) -> Dict[str, Any]:
        """
        Example tool implementation.
        
        Args:
            param1: First parameter description
            param2: Second parameter description (optional)
            
        Returns:
            Dictionary with the tool's result
        """
        # Implement your tool logic here
        return {
            "result": f"Processed {param1} with value {param2}",
            "status": "success"
        }
    
    def get_state(self) -> Dict[str, Any]:
        """
        Return the current state of this environment.
        
        This method should be implemented to provide information about
        the environment's current state for context building.
        
        Returns:
            Dictionary containing state information
        """
        # Example state - customize this for your environment
        state = super().get_state()  # Get the base state
        
        # Add custom state properties
        state.update({
            "custom_property": "value",
            # Add other state properties here
        })
        
        return state
    
    def render(self) -> Dict[str, Any]:
        """
        Render this environment for presentation.
        
        This method is used to generate a representation of the environment
        that can be shown to users or used by the agent.
        
        Returns:
            Dictionary with rendered representation
        """
        # Example rendering - customize this for your environment
        rendered = super().render()  # Get the base rendering
        
        # Add custom rendering properties
        rendered.update({
            "rendered_property": "Some human-readable value",
            # Add other rendering properties here
        })
        
        return rendered
        
    def render_state_for_context(self) -> Dict[str, Any]:
        """
        Render the current state of this environment for inclusion in the agent's context.
        
        This method focuses on the current state and content that's relevant
        for the agent's reasoning process. It should provide a human-readable
        representation of state that can be included directly in prompts.
        
        Returns:
            Dictionary with formatted state for context inclusion
        """
        # Get base implementation first
        state = super().render_state_for_context()
        
        # Update with custom environment type
        state["type"] = "custom_environment"
        
        # Create a human-readable formatted state text
        formatted_text = [
            f"# {self.name} Status",
            f"This is a custom environment for [describe purpose here].",
            "",
            "## Current State:",
            "- Example state item 1: [value]",
            "- Example state item 2: [value]",
            "",
            "## Available Actions:",
            "- Use the example_tool to perform example operations"
        ]
        
        # Update the state with our formatted text
        state["formatted_state_text"] = "\n".join(formatted_text)
        state["state_summary"] = "Custom environment with [describe key state features]"
        
        return state

# Notes:
# 1. Do not instantiate your environment here
# 2. Your environment will be discovered automatically
# 3. Configuration in config.py will determine if and how your environment is used 