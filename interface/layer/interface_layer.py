"""
Interface Layer

Main class that coordinates all interface layer components.
"""

import logging
from typing import Dict, Any, List, Optional
import time

from interface.prompt_manager import PromptManager
from interface.protocol import get_protocol
from config import DEFAULT_PROTOCOL

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
        
        # Initialize context handler
        self.context_handler = ContextHandler()
        self.context_handler.set_environment_manager(environment_manager)
        
        # Set up environment manager to use context handler
        self.environment_manager.set_context_manager(self.context_handler)
        
        # Get protocol instance
        protocol_instance = get_protocol(protocol_name)
        
        # Initialize environment renderer first (needed by prompt manager)
        self.environment_renderer = EnvironmentRenderer(self.environment_manager)
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager(protocol_instance.protocol_prompt_format)
        
        # Connect prompt manager with context handler and environment renderer
        self.context_handler.set_prompt_manager(self.prompt_manager)
        self.prompt_manager.set_environment_renderer(self.environment_renderer)
        
        # Initialize LLM processor with protocol
        self.llm_processor = LLMProcessor(protocol_instance, self.prompt_manager)
        
        # Initialize tool manager
        self.tool_manager = PrivilegedToolManager(self.environment_manager)
        
        # Initialize message processor
        self.message_processor = MessageProcessor(
            self.environment_manager,
            self.context_handler,
            self.llm_processor,
            self.tool_manager,
            self.environment_renderer
        )
        
        # Initialize mounted environments tracking
        self.mounted_environments = set()
        
        # Register as an observer for messages from the Environment Layer
        self.environment_manager.register_message_observer(self)
        
        # Register as observer for the system environment
        system_env = self.environment_manager.get_environment("system")
        if system_env:
            self.environment_manager.register_environment_observer("system", self)
            self.mounted_environments.add("system")
        
        # Register privileged tools with the context handler
        self._register_privileged_tools()
        
        logger.info(f"Interface layer initialized with protocol: {protocol_name}")
    
    def _register_privileged_tools(self):
        """
        Register privileged tools with the context handler.
        """
        # Implement the logic to register privileged tools with the context handler
        pass
    
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
        
        Requires standardized event format:
        {
          "event": "eventType",
          "data": {
            // Event-specific data
          }
        }
        
        Args:
            env_id: ID of the environment that was updated
            env_state: Current state of the environment
            update_data: Data about the update
        
        Returns:
            Result of processing the notification if one was created
        """
        logger.debug(f"Observed update for environment {env_id}: {update_data}")
        
        # Extract event type and data from standardized format
        if 'event' not in update_data or 'data' not in update_data:
            logger.error(f"Invalid update format: missing 'event' or 'data' fields in {update_data}")
            return None
            
        event_type = update_data['event']
        event_data = update_data['data']
            
        # Handle special administrative events
        # Handle environment mounting
        if event_type == 'environmentMounted':
            mounted_env_id = event_data.get('envId')
            if mounted_env_id and mounted_env_id not in self.mounted_environments:
                # Register as observer for the newly mounted environment
                self.environment_manager.register_environment_observer(mounted_env_id, self)
                self.mounted_environments.add(mounted_env_id)
                logger.info(f"Registered as observer for newly mounted environment {mounted_env_id}")
            
        # Handle environment unmounting
        elif event_type == 'environmentUnmounted':
            unmounted_env_id = event_data.get('envId')
            if unmounted_env_id and unmounted_env_id in self.mounted_environments:
                # Unregister as observer for the unmounted environment
                self.mounted_environments.remove(unmounted_env_id)
                logger.info(f"Unregistered as observer for unmounted environment {unmounted_env_id}")
        
        # Handle initial state (we need to add the environment regardless)
        elif event_type == 'initialState':
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
        
        # Only process the update if it requires the agent's attention
        if not self._update_requires_attention(env_id, update_data):
            logger.debug(f"Update for environment {env_id} does not require attention, skipping notification")
            return None
        
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
        
        Delegates the decision to the environment's own requires_attention method,
        allowing each environment to implement its own rules.
        
        Requires standardized format:
        {
          "event": "eventType",
          "data": {
            // Event-specific data
          }
        }
        
        Args:
            env_id: ID of the environment
            update_data: Data about the update
            
        Returns:
            True if the agent should be notified, False otherwise
        """
        # Ensure we have the required format
        if 'event' not in update_data or 'data' not in update_data:
            logger.error(f"Invalid update format: missing 'event' or 'data' fields in {update_data}")
            return False
            
        # Get the environment
        environment = self.environment_manager.get_environment(env_id)
        if not environment:
            logger.error(f"Cannot find environment {env_id} to check attention requirements")
            return False
            
        # Delegate decision to the environment
        try:
            return environment.requires_attention(update_data)
        except Exception as e:
            logger.error(f"Error checking if update requires attention: {str(e)}")
            logger.exception("Full exception details:")
            
            # Default to requiring attention if there's an error
            # This is safer than potentially missing important updates
            return True 