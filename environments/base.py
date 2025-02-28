"""
Base Environment
Defines the base class for all environments in the Bot Framework.
"""

import logging
import uuid
from typing import Dict, Any, Optional, Callable, List, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Environment:
    """
    Base class for all environments the agent can interact with.
    
    Environments provide a way to organize tools and represent external systems.
    They can be nested to create hierarchical structures.
    """
    
    def __init__(self, env_id: Optional[str] = None, name: str = "Unnamed Environment", 
                 description: str = ""):
        """
        Initialize a new environment.
        
        Args:
            env_id: Unique identifier for this environment (generated if not provided)
            name: Human-readable name for this environment
            description: Detailed description of this environment
        """
        self.id = env_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.parent = None
        self.children: Dict[str, 'Environment'] = {}
        self.metadata: Dict[str, Any] = {}
        logger.info(f"Created environment: {self.name} ({self.id})")
    
    def register_tool(self, tool_func: Callable, name: Optional[str] = None, 
                     description: str = "", parameter_descriptions: Optional[Dict[str, str]] = None) -> str:
        """
        Register a tool with this environment.
        
        Args:
            tool_func: Function implementing the tool
            name: Name for the tool (defaults to function name)
            description: Description of what the tool does
            parameter_descriptions: Descriptions of the tool's parameters
            
        Returns:
            The registered tool name
        """
        tool_name = name or tool_func.__name__
        
        if tool_name in self.tools:
            logger.warning(f"Tool {tool_name} already exists in environment {self.name}, overwriting")
        
        # Extract parameter info from function signature
        import inspect
        sig = inspect.signature(tool_func)
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            # Skip self parameter for methods
            if param_name == 'self':
                continue
                
            param_desc = "No description available"
            if parameter_descriptions and param_name in parameter_descriptions:
                param_desc = parameter_descriptions[param_name]
                
            # Determine if parameter is required
            required = param.default == inspect.Parameter.empty
            
            parameters[param_name] = {
                'description': param_desc,
                'required': required
            }
        
        self.tools[tool_name] = {
            'name': tool_name,
            'description': description,
            'parameters': parameters,
            'function': tool_func,
            'environment_id': self.id
        }
        
        logger.info(f"Registered tool {tool_name} in environment {self.name}")
        return tool_name
        
    def mount(self, environment: 'Environment', mount_point: Optional[str] = None) -> bool:
        """
        Mount a child environment.
        
        Args:
            environment: The environment to mount
            mount_point: Optional name for the mount point
            
        Returns:
            True if successful, False otherwise
        """
        if environment.id == self.id:
            logger.error(f"Cannot mount environment to itself: {self.id}")
            return False
            
        # Check for circular references
        if self._is_ancestor_of(environment):
            logger.error(f"Circular mount detected: {self.id} is already an ancestor of {environment.id}")
            return False
        
        # Set mount point if provided
        if mount_point:
            environment.metadata['mount_point'] = mount_point
        
        # Set parent reference
        environment.parent = self
        
        # Add to children
        self.children[environment.id] = environment
        
        logger.info(f"Mounted environment {environment.name} to {self.name}")
        return True
        
    def unmount(self, env_id: str) -> bool:
        """
        Unmount a child environment.
        
        Args:
            env_id: ID of the child environment to unmount
            
        Returns:
            True if successful, False otherwise
        """
        if env_id not in self.children:
            logger.error(f"Environment {env_id} is not mounted in {self.name}")
            return False
            
        # Clear parent reference
        self.children[env_id].parent = None
        
        # Remove from children
        del self.children[env_id]
        
        logger.info(f"Unmounted environment {env_id} from {self.name}")
        return True
    
    def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool by name from this environment only (not from children).
        
        Args:
            tool_name: Name of the tool to get
            
        Returns:
            Tool dict if found, None otherwise
        """
        return self.tools.get(tool_name)
    
    def resolve_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Resolve a tool by name, checking this environment and all mounted environments.
        
        Args:
            tool_name: Name of the tool to resolve
            
        Returns:
            Tool dict if found, None otherwise
        """
        # Check if tool exists in this environment
        if tool_name in self.tools:
            return self.tools[tool_name]
            
        # Check child environments
        for child in self.children.values():
            tool = child.resolve_tool(tool_name)
            if tool:
                return tool
                
        # Not found
        return None
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool within this environment.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Result of the tool execution
            
        Raises:
            ValueError: If the tool is not found
        """
        tool = self.resolve_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found in environment {self.name} or its children")
            
        try:
            logger.info(f"Executing tool {tool_name} in environment {self.name}")
            return tool['function'](**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of this environment.
        
        Returns:
            Dictionary representing the environment state
        """
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'metadata': self.metadata,
            'tools': list(self.tools.keys()),
            'children': [
                {
                    'id': child.id,
                    'name': child.name, 
                    'mount_point': child.metadata.get('mount_point', child.name)
                } 
                for child in self.children.values()
            ]
        }
    
    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all tools available in this environment and its children.
        
        Returns:
            Dictionary mapping tool names to tool dicts
        """
        all_tools = dict(self.tools)
        
        for child in self.children.values():
            # Add tools from child, potentially overriding parent tools
            child_tools = child.get_all_tools()
            all_tools.update(child_tools)
            
        return all_tools
    
    def render(self) -> Dict[str, Any]:
        """
        Render this environment's structure and capabilities for presentation to the agent.
        
        Returns:
            Dictionary with rendered environment information
        """
        tools_info = []
        for tool_name, tool in self.tools.items():
            tools_info.append({
                'name': tool_name,
                'description': tool['description'],
                'parameters': tool['parameters']
            })
            
        child_envs = []
        for child in self.children.values():
            mount_point = child.metadata.get('mount_point', child.name)
            child_envs.append({
                'id': child.id,
                'name': child.name,
                'description': child.description,
                'mount_point': mount_point
            })
            
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'tools': tools_info,
            'mounted_environments': child_envs,
            'metadata': self.metadata
        }
        
    def render_state_for_context(self) -> Dict[str, Any]:
        """
        Render the current state of this environment for inclusion in the agent's context.
        
        Unlike render(), which describes the environment's structure and capabilities,
        this method focuses on the current state and content that's relevant
        for the agent's reasoning process.
        
        Environment implementations should provide a 'formatted_state_text' field
        containing a human-readable representation of their state that can be
        directly included in the agent's prompt.
        
        Returns:
            Dictionary with state data for context inclusion
        """
        # Base implementation provides minimal state info
        return {
            "type": "environment",
            "id": self.id,
            "name": self.name,
            "state_summary": "Default environment with no specialized state",
            "formatted_state_text": f"Environment '{self.name}' has no specialized state information."
        }
        
    def _is_ancestor_of(self, env: 'Environment') -> bool:
        """
        Check if this environment is an ancestor of the given environment.
        
        Args:
            env: Environment to check
            
        Returns:
            True if this environment is an ancestor of env, False otherwise
        """
        if env.id in self.children:
            return True
            
        for child in self.children.values():
            if child._is_ancestor_of(env):
                return True
                
        return False
        
    def __repr__(self) -> str:
        """String representation of the environment."""
        return f"<Environment {self.name} ({self.id})>" 