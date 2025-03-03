"""
Interface Layer

Main class that coordinates all interface layer components.
"""

import logging
from typing import Dict, Any, List, Optional
import time

from interface.prompt_manager import PromptManager
from interface.protocol import get_protocol
from config import STORAGE_PATH, AGENT_NAME, DEFAULT_PROTOCOL, PROTOCOLS

from bot_framework.interface.layer.tool_manager import PrivilegedToolManager
from bot_framework.interface.layer.context_handler import ContextHandler
from bot_framework.interface.layer.llm_processor import LLMProcessor
from bot_framework.interface.layer.message_processor import MessageProcessor
from bot_framework.interface.layer.environment_renderer import EnvironmentRenderer

# Configure logging
logger = logging.getLogger(__name__)


class InterfaceLayer:
    """
    Interface Layer - presentation and cognitive layer between user and environments.
    
    The InterfaceLayer is responsible for:
    1. Rendering environments to be consumed by the agent
    2. Executing tools in the appropriate environments
    3. Managing privileged tools
    4. Processing messages from users
    5. Managing conversation flow and context
    
    This class serves as the main orchestrator, delegating responsibilities to
    specialized components.
    """
    
    def __init__(self, environment_manager, protocol_name: str = DEFAULT_PROTOCOL):
        """
        Initialize the interface layer.
        
        Args:
            environment_manager: The environment manager instance
            protocol_name: Name of the protocol to use
        """
        self.environment_manager = environment_manager
        self.protocol_name = protocol_name
        
        self.context_handler = ContextHandler()
        
        # Get protocol instance
        protocol_instance = get_protocol(protocol_name)

        # Initialize prompt manager
        prompt_manager = PromptManager()

        self.llm_processor = LLMProcessor(protocol_instance, prompt_manager)
        
        self.tool_manager = PrivilegedToolManager(self.environment_manager)
        
        # Initialize environment renderer
        self.environment_renderer = EnvironmentRenderer()
        
        # Initialize message processor
        self.message_processor = MessageProcessor(
            self.environment_manager,
            self.context_handler,
            self.llm_processor,
            self.tool_manager,
            self.environment_renderer
        )
        
        # Register as an observer for messages from the Environment Layer
        self.environment_manager.register_message_observer(self)
        
        # Track mounted environments and register as observer for each
        self.mounted_environments = set()
        self._register_for_system_environment()
        
        logger.info(f"Interface layer initialized with protocol: {protocol_name}")
    
    def _register_for_system_environment(self):
        """
        Register as an observer for the system environment and any mounted environments.
        """
        # Get the system environment
        system_env = self.environment_manager.system_environment
        if system_env:
            # Register for the system environment
            self.environment_manager.register_environment_observer(system_env.id, self)
            self.mounted_environments.add(system_env.id)
            
            # The base class observer system will automatically register us for child environments
            # No need to register for each child separately as it's now handled by the mount mechanism
            logger.info(f"Registered as observer for system environment and its children")
    
    def observe_message(self, message_data):
        """
        Handle incoming messages from the environment layer.
        
        This method is called by the MessageService when a new message is received.
        
        Args:
            message_data: Dictionary containing message data
            
        Returns:
            None
        """
        user_id = message_data.get('user_id')
        message_text = message_data.get('message_text', '')
        message_id = message_data.get('message_id')
        platform = message_data.get('platform')
        
        logger.info(f"Interface layer received message from user {user_id}")
        
        # Process the message
        self.process_message(user_id, message_text, message_id, platform)
    
    def process_message(self, user_id: str, message_text: str, message_id: Optional[str] = None, 
                        platform: Optional[str] = None, env_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an incoming message.
        
        This is the main entry point for the interface layer.
        
        Args:
            user_id: ID of the user who sent the message
            message_text: Text of the message
            message_id: Optional ID of the message
            platform: Optional platform the message was sent from
            env_id: Optional environment ID where the message is processed
            
        Returns:
            Processing result
        """
        return self.message_processor.process_message(user_id, message_text, message_id, platform, env_id)
    
    def render_environments(self) -> Dict[str, Any]:
        """
        Render all environments.
        
        Returns:
            Dictionary mapping environment IDs to rendered environments
        """
        return self.environment_renderer.render_for_agent()
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        return self.tool_manager.execute_tool(tool_name, **kwargs)
    
    def get_environment_context(self, env_id: str, max_messages: Optional[int] = None) -> Dict[str, Any]:
        """
        Get the context for an environment.
        
        Args:
            env_id: ID of the environment
            max_messages: Optional maximum number of messages to include
            
        Returns:
            Environment context dictionary
        """
        return self.context_handler.get_environment_context(env_id, max_messages)
    
    def clear_environment_context(self, env_id: str) -> Dict[str, Any]:
        """
        Clear the context for an environment.
        
        Args:
            env_id: ID of the environment
            
        Returns:
            Result of the operation
        """
        return self.context_handler.clear_environment_context(env_id)
    
    def observe_environment_update(self, env_id: str, env_state: Dict[str, Any], update_data: Dict[str, Any]):
        """
        Observe an update to an environment and determine if a notification is needed.
        
        Args:
            env_id: ID of the environment that was updated
            env_state: Current state of the environment
            update_data: Data about the update
        
        Returns:
            Result of processing the notification if one was created
        """
        logger.debug(f"Observed update for environment {env_id}: {update_data}")
        
        # Handle special administrative update types
        update_type = update_data.get('type', 'generic_update')
        
        # Handle environment mounting
        if update_type == 'environment_mounted':
            mounted_env_id = update_data.get('env_id')
            if mounted_env_id and mounted_env_id not in self.mounted_environments:
                # Register as observer for the newly mounted environment
                self.environment_manager.register_environment_observer(mounted_env_id, self)
                self.mounted_environments.add(mounted_env_id)
                logger.info(f"Registered as observer for newly mounted environment {mounted_env_id}")
            
        # Handle environment unmounting
        elif update_type == 'environment_unmounted':
            unmounted_env_id = update_data.get('env_id')
            if unmounted_env_id and unmounted_env_id in self.mounted_environments:
                # Unregister as observer for the unmounted environment
                self.mounted_environments.remove(unmounted_env_id)
                logger.info(f"Unregistered as observer for unmounted environment {unmounted_env_id}")
        
        # Handle initial state (we need to add the environment regardless)
        elif update_type == 'initial_state':
            # Make sure the environment is in our tracked set
            if env_id not in self.mounted_environments:
                self.mounted_environments.add(env_id)
                logger.info(f"Added environment {env_id} to tracked environments")
        
        environment = self.environment_manager.get_environment(env_id)
        if not environment:
            logger.error(f"Cannot find environment {env_id} to process update")
            return None
        
        # Get the environment to format its update data for the agent
        formatted_update = environment.format_update_for_agent(update_data) if hasattr(environment, 'format_update_for_agent') else str(update_data)
        
        # Create a synthetic message from the environment update
        system_prefix = f"Environment '{env_id}' has been updated. The following information represents this update:"
        synthetic_message = f"{system_prefix}\n\n{formatted_update}"
        
        # Process this synthetic message through the normal message processing pipeline
        # Using "system" as the user_id to indicate this is a system-generated message
        response = self.process_message(
            user_id="system",
            message_text=synthetic_message,
            message_id=f"env_update_{env_id}_{int(time.time())}",
            platform="environment",
            env_id=env_id  # Pass the actual environment ID
        )
        
        logger.info(f"Processed environment update for {env_id} with response: {response}")
        return response
    
    def _update_requires_attention(self, env_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Determine if an environment update requires the agent's attention.
        
        Args:
            env_id: ID of the environment
            update_data: Data about the update
            
        Returns:
            True if the agent should be notified, False otherwise
        """
        # By default, these update types always require attention
        attention_types = [
            'new_message', 
            'document_change', 
            'email_received',
            'environment_mounted',
            'environment_unmounted'
        ]
        
        update_type = update_data.get('type', '')
        if update_type in attention_types:
            return True
            
        # Check for explicit attention flag
        if update_data.get('requires_attention', False):
            return True
            
        # Otherwise, don't bother the agent
        return False 