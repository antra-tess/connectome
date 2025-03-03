"""
System Environment
Root environment that mounts all other environments and provides core system tools.
"""

import logging
import os
from typing import Dict, Any, Optional, List

from environments.base import Environment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SystemEnvironment(Environment):
    """
    Root environment that provides core system tools and mounts other environments.
    
    The SystemEnvironment is the root of the environment hierarchy and provides
    tools for managing other environments.
    """
    
    def __init__(self):
        """Initialize the system environment."""
        super().__init__(env_id="system", name="System Environment", 
                        description="Root environment that provides core system tools")
        self._environment_manager = None
        self._register_privileged_tools()
        logger.info("System environment initialized")
        
    def set_environment_manager(self, environment_manager):
        """
        Set the environment manager.
        
        Args:
            environment_manager: The environment manager instance
        """
        self._environment_manager = environment_manager
        
    def _register_privileged_tools(self):
        """Register privileged system tools."""
        # Environment management tools
        self.register_tool(
            self.mount_environment,
            name="mount_environment",
            description="Mount an environment to make its tools available",
            parameter_descriptions={
                "env_id": "ID of the environment to mount",
                "mount_point": "Optional name for the mount point"
            }
        )
        
        self.register_tool(
            self.unmount_environment,
            name="unmount_environment",
            description="Unmount an environment, removing access to its tools",
            parameter_descriptions={
                "env_id": "ID of the environment to unmount"
            }
        )
        
        self.register_tool(
            self.list_environments,
            name="list_environments",
            description="List all available environments and their tools",
            parameter_descriptions={}
        )
        
        self.register_tool(
            self.get_system_state,
            name="get_system_state",
            description="Get information about the current system state",
            parameter_descriptions={}
        )
        
    def mount_environment(self, env_id: str, mount_point: Optional[str] = None) -> Dict[str, Any]:
        """
        Mount an environment to the system environment.
        
        Args:
            env_id: ID of the environment to mount
            mount_point: Optional name for the mount point
            
        Returns:
            Result of the operation
        """
        if not self._environment_manager:
            return {"status": "error", "message": "Environment manager not set"}
            
        # Get the environment to mount
        environment = self._environment_manager.get_environment(env_id)
        if not environment:
            return {"status": "error", "message": f"Environment {env_id} not found"}
            
        # Check if already mounted
        if env_id in self.children:
            return {"status": "error", "message": f"Environment {env_id} already mounted"}
            
        # Mount the environment - this will handle observer notifications
        # since we've implemented it in the base class
        mount_result = self.mount(environment, mount_point)
        if not mount_result:
            return {"status": "error", "message": f"Failed to mount environment {env_id}"}
                
        return {
            "status": "success", 
            "message": f"Environment {env_id} mounted successfully",
            "mount_point": mount_point or environment.name
        }
        
    def unmount_environment(self, env_id: str) -> Dict[str, Any]:
        """
        Unmount an environment from the system environment.
        
        Args:
            env_id: ID of the environment to unmount
            
        Returns:
            Result of the operation
        """
        if not self._environment_manager:
            return {"status": "error", "message": "Environment manager not set"}
            
        # Check if mounted
        if env_id not in self.children:
            return {"status": "error", "message": f"Environment {env_id} not mounted"}
            
        # Unmount the environment - this will handle observer notifications
        # since we've implemented it in the base class
        unmount_result = self.unmount(env_id)
        if not unmount_result:
            return {"status": "error", "message": f"Failed to unmount environment {env_id}"}
                
        return {
            "status": "success", 
            "message": f"Environment {env_id} unmounted successfully"
        }
    
    def list_environments(self) -> Dict[str, Any]:
        """
        List all available environments and their tools.
        
        Returns:
            Dictionary with environment information
        """
        if not self._environment_manager:
            logger.error("Environment manager not set")
            return {
                "success": False,
                "error": "Environment manager not set"
            }
            
        # Get all environments from the manager
        environments = self._environment_manager.get_all_environments()
        
        # Build environment list
        env_list = []
        for env_id, env in environments.items():
            env_list.append({
                "id": env.id,
                "name": env.name,
                "description": env.description,
                "tools": list(env.get_all_tools().keys())
            })
            
        # Get mounted environments
        mounted_envs = []
        for child_id, child in self.children.items():
            mount_point = child.metadata.get("mount_point", child.name)
            mounted_envs.append({
                "id": child.id,
                "name": child.name,
                "mount_point": mount_point
            })
            
        return {
            "success": True,
            "all_environments": env_list,
            "mounted_environments": mounted_envs
        }
    
    def get_system_state(self) -> Dict[str, Any]:
        """
        Get information about the current system state.
        
        Returns:
            Dictionary with system state information
        """
        # Get all registered tools from mounted environments
        tools = self.get_all_tools()
        tool_list = []
        for tool_name, tool_info in tools.items():
            tool_list.append({
                "name": tool_name,
                "description": tool_info.get("description", ""),
                "environment_id": tool_info.get("environment_id", "unknown")
            })
        
        # Get environment hierarchy
        mounted_envs = []
        for child_id, child in self.children.items():
            mount_point = child.metadata.get("mount_point", child.name)
            mounted_envs.append({
                "id": child.id,
                "name": child.name,
                "mount_point": mount_point
            })
            
        return {
            "success": True,
            "environment_id": self.id,
            "name": self.name,
            "tools_available": len(tool_list),
            "tools": tool_list,
            "mounted_environments": mounted_envs
        }
    
    def render_capabilities_for_context(self) -> str:
        """
        Render a formatted text representation of available tools and environments.
        
        This method creates a user-friendly, formatted text output that describes
        all available environments and tools that can be used by the agent.
        
        Returns:
            String containing formatted capabilities information ready for inclusion in prompt
        """
        if not self._environment_manager:
            return "No environment information available."
            
        # Get all environments
        all_environments = self._environment_manager.get_all_environments()
        
        # Format environment descriptions
        env_section = "## Available Environments\n"
        for env_id, env in all_environments.items():
            # Skip self in the listing
            if env_id == self.id:
                continue
                
            env_info = env.render()
            mount_point = env_info.get("mount_point", env_info.get("name", env_id))
            description = env_info.get("description", "No description")
            
            env_section += f"\n### {env_info.get('name')} ({mount_point})\n"
            env_section += f"{description}\n"
        
        # Format tool descriptions
        tools_section = "\n## Available Tools\n"
        all_tools = self.get_all_tools()
        
        for tool_name, tool in all_tools.items():
            # Get tool details
            is_privileged = tool.get("is_privileged", False)
            tag = "[Privileged]" if is_privileged else ""
            description = tool.get("description", "No description")
            
            tools_section += f"\n### {tool_name} {tag}\n"
            tools_section += f"{description}\n"
            
            # Add parameter descriptions
            parameters = tool.get("parameters", {})
            if parameters:
                tools_section += "Parameters:\n"
                for param_name, param_info in parameters.items():
                    # Skip 'self' parameter for methods
                    if param_name == 'self':
                        continue
                        
                    required = param_info.get("required", False)
                    req_tag = "(required)" if required else "(optional)"
                    desc = param_info.get("description", "No description")
                    tools_section += f"- {param_name} {req_tag}: {desc}\n"
        
        # Combine sections
        return f"{env_section}\n{tools_section}" 