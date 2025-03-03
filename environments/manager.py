"""
Environment Manager
Manages the lifecycle of environments in the Bot Framework.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
import traceback

from environments.base import Environment
from environments.system import SystemEnvironment
from environments.message_service import MessageService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnvironmentManager:
    """
    Manages the lifecycle of environments in the Bot Framework.
    
    The EnvironmentManager is responsible for registering, retrieving,
    and maintaining environments. It provides a central registry for
    all available environments.
    """
    
    def __init__(self, context_manager=None):
        """
        Initialize the environment manager.
        
        Args:
            context_manager: Optional context manager instance
        """
        self.environments: Dict[str, Environment] = {}
        self.system_environment = SystemEnvironment()
        self._context_manager = context_manager
        
        # Create the message service
        self.message_service = MessageService(self)
        
        # Register the system environment
        self.register_environment(self.system_environment)
        
        # Connect system environment to managers
        self.system_environment.set_environment_manager(self)
        if context_manager:
            self.system_environment.set_context_manager(context_manager)
            
        self.response_callback = None
        
        logger.info("Environment manager initialized")
    
    def register_environment(self, environment: Environment) -> bool:
        """
        Register a new environment.
        
        Args:
            environment: Environment to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        if environment.id in self.environments:
            logger.warning(f"Environment with ID {environment.id} already exists, overwriting")
            
        self.environments[environment.id] = environment
        
        logger.info(f"Registered environment: {environment.id}")
        return True
    
    def unregister_environment(self, env_id: str) -> bool:
        """
        Unregister an environment.
        
        Args:
            env_id: ID of the environment to unregister
            
        Returns:
            True if unregistration was successful, False otherwise
        """
        if env_id == "system":
            logger.error("Cannot unregister the system environment")
            return False
            
        if env_id not in self.environments:
            logger.warning(f"Environment with ID {env_id} not found")
            return False
            
        # Check if the environment is mounted anywhere and unmount it
        for env in self.environments.values():
            if env_id in env.children:
                env.unmount(env_id)
                
        # Remove from registry
        del self.environments[env_id]
        logger.info(f"Unregistered environment: {env_id}")
        return True
    
    def get_environment(self, env_id: str) -> Optional[Environment]:
        """
        Get an environment by ID.
        
        Args:
            env_id: ID of the environment to get
            
        Returns:
            Environment instance if found, None otherwise
        """
        return self.environments.get(env_id)
    
    def get_all_environments(self) -> Dict[str, Environment]:
        """
        Get all registered environments.
        
        Returns:
            Dictionary mapping environment IDs to environment instances
        """
        return self.environments
    
    def set_context_manager(self, context_manager) -> None:
        """
        Set the context manager.
        
        Args:
            context_manager: Context manager instance
        """
        self._context_manager = context_manager
        self.system_environment.set_context_manager(context_manager)
        
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available tools from the system environment and mounted environments.
        
        Returns:
            Dictionary mapping tool names to tool definitions
        """
        return self.system_environment.get_all_tools()
        
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool by name.
        
        This will search for the tool in the system environment and all
        mounted environments.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Result of the tool execution
            
        Raises:
            ValueError: If the tool is not found
        """
        return self.system_environment.execute_tool(tool_name, **kwargs)
        
    def register_message_observer(self, observer: Callable) -> None:
        """
        Register an observer for message events.
        
        Args:
            observer: Callable object to notify
        """
        self.message_service.register_observer(observer)
        logger.debug(f"Registered message observer: {observer}")
    
    # New methods for environment-specific observer pattern
    def register_environment_observer(self, env_id: str, observer) -> None:
        """
        Register an observer for a specific environment.
        
        Args:
            env_id: Environment ID to observe
            observer: Observer object (typically an Interface Layer) to notify
        """
        if env_id not in self.environments:
            logger.warning(f"Cannot register observer for non-existent environment: {env_id}")
            return
            
        # Use the environment's built-in observer management
        environment = self.environments[env_id]
        environment.register_observer(observer)
    
    def unregister_environment_observer(self, env_id: str, observer) -> None:
        """
        Remove an observer from a specific environment.
        
        Args:
            env_id: Environment ID
            observer: Observer to remove
        """
        if env_id not in self.environments:
            logger.warning(f"Cannot unregister observer from non-existent environment: {env_id}")
            return
            
        # Use the environment's built-in observer management
        environment = self.environments[env_id]
        environment.unregister_observer(observer)
    
    def update_environment_state(self, env_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update the state of a specific environment and notify observers.
        
        Args:
            env_id: ID of the environment to update
            update_data: Data to update the environment with
            
        Returns:
            True if update was successful, False otherwise
        """
        environment = self.get_environment(env_id)
        if not environment:
            logger.warning(f"Cannot update non-existent environment: {env_id}")
            return False
            
        # Have the environment update its internal state
        # The environment will notify observers itself
        try:
            return environment.update_state(update_data)
        except Exception as e:
            logger.error(f"Error updating environment {env_id}: {str(e)}")
            return False
    
    def process_message(self, user_id, message_text, message_id=None, platform=None, env_id=None):
        """Process an incoming message through the message service"""
        logger.info(f"Processing message from user {user_id}: {message_text}")
        
        # Process the message through the message service
        self.message_service.process_message(
            user_id=user_id,
            message_text=message_text,
            message_id=message_id,
            platform=platform,
            env_id=env_id
        )
        
        # Return acknowledgment
        return {"status": "processing"}
    
    def send_response(self, user_id, message_text, message_id=None, platform=None):
        """Send a response back through the response callback"""
        if self.response_callback:
            response_data = {
                "user_id": user_id,
                "message": message_text,
                "message_id": message_id,
                "platform": platform
            }
            self.response_callback(response_data)
            return True
        else:
            logger.warning("No response callback set in EnvironmentManager")
            return False
    
    def set_response_callback(self, callback: Callable) -> None:
        """
        Set the callback for sending responses back to the Activity Layer.
        
        Args:
            callback: Function to call with response data
        """
        self.response_callback = callback
        logger.debug("Response callback set in EnvironmentManager") 