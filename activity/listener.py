"""
Message Listener
Handles incoming messages from normalizing adapters.
"""

import logging
from typing import Dict, Any, Callable, Optional
import time
import uuid
import traceback

from config import SOCKET_HOST, SOCKET_PORT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MessageHandler:
    """
    Handles incoming messages from normalizing adapters.
    
    Validates message data, processes messages through the environment layer,
    and handles the responses.
    """
    
    def __init__(self, environment_manager):
        """
        Initialize the message handler.
        
        Args:
            environment_manager: EnvironmentManager instance to handle messages
        """
        self.environment_manager = environment_manager
        logger.info("Message handler initialized")
    
    def handle_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle an incoming message from an adapter.
        
        Args:
            data: Message data dictionary including:
                - chat_id: Identifier for the conversation
                - user_id: Identifier for the user
                - content: Message content
                - adapter_id: Identifier for the source adapter
                - event_type: Type of event (e.g., 'chat_message', 'document_update', 'email')
                
        Returns:
            Response data if a response should be sent, None otherwise
        """
        try:
            # Validate message data
            required_fields = ['chat_id', 'user_id', 'content', 'adapter_id']
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field '{field}'")
                    return None
            env_id = data['env_id']
            
            # Handle the message for this environment
            environment = self.environment_manager.get_environment(env_id)
            if not environment:
                logger.info(f"Environment {env_id} not found, might need to be created")
                return {"status": "failure", "env_id": env_id}
            # Update the environment state
            self.environment_manager.update_environment_state(env_id, data)
            return {"status": "success", "env_id": env_id}

        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def handle_clear_context(self, data: Dict[str, Any]) -> bool:
        """
        Handle context clearing request from an adapter.
        
        Args:
            data: Request data including:
                - chat_id: Identifier for the conversation to clear
                - adapter_id: Identifier for the source adapter
                
        Returns:
            True if context was cleared successfully, False otherwise
        """
        try:
            env_id = data['env_id']
            # Use the environment manager to execute the clear_environment_context tool
            # This assumes the tool is registered in one of the environments
            try:
                success = self.environment_manager.execute_tool("clear_environment_context", env_id=env_id)
                return success.get('success', False) if isinstance(success, dict) else bool(success)
            except ValueError:
                logger.error("clear_environment_context tool not found in any environment")
                return False
                
        except Exception as e:
            logger.error(f"Error clearing context: {str(e)}")
            return False
    
    def _send_response(self, response_data: Dict[str, Any]) -> None:
        """
        Callback method for sending responses.
        
        This will be called by the Environment Layer when a response is ready.
        In a real implementation, this would send the response through Socket.IO.
        
        Args:
            response_data: Dictionary containing response data
        """
        # This is a placeholder - in the real implementation, this would send
        # the response through the Socket.IO client or other mechanism
        logger.info(f"Sending response to adapter {response_data.get('adapter_id')}")
        # In a real implementation: socket_client.send_message(response_data)
