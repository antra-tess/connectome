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
        self.observers: List[Any] = []  # List of observers for this environment
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
        
    def register_observer(self, observer) -> bool:
        """
        Register an observer for this environment.
        
        Args:
            observer: Observer object to notify of state changes
            
        Returns:
            True if registration was successful
        """
        if observer in self.observers:
            logger.debug(f"Observer already registered for environment {self.id}")
            return False
            
        self.observers.append(observer)
        logger.debug(f"Registered observer for environment {self.id}")
        
        # Notify the observer of the current state
        try:
            self._notify_observer_of_initial_state(observer)
        except Exception as e:
            logger.error(f"Error sending initial state to observer: {str(e)}")
            
        return True
        
    def unregister_observer(self, observer) -> bool:
        """
        Unregister an observer from this environment.
        
        Args:
            observer: Observer object to remove
            
        Returns:
            True if unregistration was successful
        """
        if observer not in self.observers:
            logger.debug(f"Observer not registered for environment {self.id}")
            return False
            
        self.observers.remove(observer)
        logger.debug(f"Unregistered observer from environment {self.id}")
        return True
        
    def notify_observers(self, update_data: Dict[str, Any]) -> None:
        """
        Notify all observers of a state change.
        
        Args:
            update_data: Data about the update
        """
        env_state = self.render_state_for_context()
        
        for observer in self.observers:
            try:
                observer.observe_environment_update(self.id, env_state, update_data)
                logger.debug(f"Notified observer about update to environment {self.id}")
            except Exception as e:
                logger.error(f"Error notifying observer about environment update: {str(e)}")
                
    def _notify_observer_of_initial_state(self, observer) -> None:
        """
        Notify a specific observer of the current state.
        
        Args:
            observer: Observer to notify
        """
        env_state = self.render_state_for_context()
        
        # Create initial state update data
        update_data = {
            "type": "initial_state",
            "env_id": self.id,
            "is_initial": True
        }
        
        # Notify the observer
        observer.observe_environment_update(self.id, env_state, update_data)
        logger.debug(f"Sent initial state to observer for environment {self.id}")
        
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
                
        # Create mount event data
        mount_event = {
            "type": "environment_mounted",
            "env_id": environment.id,
            "mount_point": mount_point or environment.name,
            "environment_state": environment.render_state_for_context()
        }
        
        # Notify observers about the mount
        self.notify_observers(mount_event)
        
        # Register all our observers with the child environment
        for observer in self.observers:
            environment.register_observer(observer)
            
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
        
        # Get the environment and its mount point before unmounting
        environment = self.children[env_id]
        mount_point = environment.metadata.get('mount_point', environment.name)
            
        # Create unmount event data
        unmount_event = {
            "type": "environment_unmounted",
            "env_id": env_id,
            "mount_point": mount_point
        }
        
        # Unregister all our observers from the child environment
        for observer in self.observers:
            environment.unregister_observer(observer)
            
        # Clear parent reference
        self.children[env_id].parent = None
        
        # Remove from children
        del self.children[env_id]
        
        # Notify observers about the unmount
        self.notify_observers(unmount_event)
        
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
    
    def update_state(self, update_data: Dict[str, Any]) -> bool:
        """
        Update the state of this environment based on activity data.
        
        This base implementation simply updates the metadata with the provided data.
        Subclasses should override this method to provide environment-specific state updates.
        
        Args:
            update_data: Data to update the environment state with
            
        Returns:
            True if the update was successful, False otherwise
        """
        try:
            # Default implementation just updates metadata
            update_type = update_data.get('type', 'generic_update')
            
            # Store the update in the environment's history
            if 'history' not in self.metadata:
                self.metadata['history'] = []
                
            # Add timestamp if not present
            if 'timestamp' not in update_data:
                import time
                update_data['timestamp'] = time.time()
                
            # Add to history
            self.metadata['history'].append(update_data)
            
            # Limit history size to prevent unbounded growth
            max_history = 100  # Configurable
            if len(self.metadata['history']) > max_history:
                self.metadata['history'] = self.metadata['history'][-max_history:]
                
            # Update current state based on update type
            if 'current_state' not in self.metadata:
                self.metadata['current_state'] = {}
                
            # Store the latest update of each type
            self.metadata['current_state'][update_type] = update_data
            
            # Notify observers about the state change
            self.notify_observers(update_data)
            
            logger.info(f"Updated state of environment {self.id} with {update_type}")
            return True
        except Exception as e:
            logger.error(f"Error updating environment state: {str(e)}")
            return False
    
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
        
    def format_update_for_agent(self, update_data: Dict[str, Any]) -> str:
        """
        Format an environment update in a way that is informative and useful for the agent.
        
        This method converts environment update data into a structured string format
        that can be passed directly to the agent as part of a message. Environments
        should override this method to provide specialized formatting for their updates.
        
        Args:
            update_data: Data about the update from notify_observers
            
        Returns:
            Formatted string describing the update in a way optimized for the agent
        """
        # Base implementation provides a simple XML-style format
        update_type = update_data.get('type', 'generic_update')
        
        # Start with environment identification
        formatted_update = f'<environment id="{self.id}" name="{self.name}" update_type="{update_type}">\n'
        
        # Add update details
        formatted_update += '  <update_details>\n'
        for key, value in update_data.items():
            if key != 'type' and not isinstance(value, (dict, list)):
                formatted_update += f'    <{key}>{value}</{key}>\n'
        formatted_update += '  </update_details>\n'
        
        # Add current state summary
        state = self.render_state_for_context()
        formatted_update += f'  <current_state>{state.get("state_summary", "No state summary available")}</current_state>\n'
        
        # End environment tag
        formatted_update += '</environment>'
        
        return formatted_update
        
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