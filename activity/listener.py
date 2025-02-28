"""
Message Listener
Handles incoming messages from normalizing adapters.
"""

import logging
from typing import Dict, Any, Callable, Optional

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
        
        # Register this handler as the response callback
        self.environment_manager.set_response_callback(self._send_response)
        
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
            
            # Ensure the message has a role field
            if 'role' not in data:
                data['role'] = 'user'
            
            # Add platform information if available
            if 'platform' not in data:
                data['platform'] = data.get('adapter_id', 'unknown')
                
            # Process the message through the environment layer
            response = self.environment_manager.process_incoming_message(data)
            
            # If the environment manager generated a response, add the adapter_id to it
            if response and 'adapter_id' not in response:
                response['adapter_id'] = data['adapter_id']
                
            return response
            
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
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
            chat_id = data.get('chat_id')
            
            if not chat_id:
                logger.error("Missing required field 'chat_id'")
                return False
            
            # Use the environment manager to execute the clear_context tool
            # This assumes the tool is registered in one of the environments
            try:
                success = self.environment_manager.execute_tool("clear_context", chat_id=chat_id)
                return success.get('success', False) if isinstance(success, dict) else bool(success)
            except ValueError:
                logger.error("clear_context tool not found in any environment")
                return False
                
        except Exception as e:
            logger.error(f"Error clearing context: {str(e)}")
            return False
    
    def handle_event(self, event_type: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle an incoming event from an adapter.
        
        Args:
            event_type: Type of event
            data: Event data
                
        Returns:
            Response data if a response should be sent, None otherwise
        """
        try:
            if event_type == 'message':
                return self.handle_message(data)
            elif event_type == 'clear_context':
                success = self.handle_clear_context(data)
                return {
                    'success': success,
                    'adapter_id': data.get('adapter_id', 'unknown')
                }
            else:
                logger.warning(f"Unsupported event type: {event_type}")
                return None
        except Exception as e:
            logger.error(f"Error handling event: {str(e)}")
            return None
    
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


def handle_message(message_data, environment_manager):
    """
    Handle an incoming message from a user
    
    Args:
        message_data: Dictionary containing the message data
        environment_manager: The EnvironmentManager instance
        
    Returns:
        Dictionary with response status
    """
    try:
        logger.info(f"Handling message from user: {message_data.get('user_id', 'unknown')}")
        
        # Extract message data
        user_id = message_data.get('user_id')
        message_text = message_data.get('content', '')
        message_id = message_data.get('message_id')
        platform = message_data.get('platform')
        
        if not user_id or not message_text:
            logger.error("Missing required fields in message data")
            return {"error": "Missing required fields"}
        
        # Process the message through the environment manager
        return environment_manager.process_message(
            user_id=user_id,
            message_text=message_text,
            message_id=message_id,
            platform=platform
        )
        
    except Exception as e:
        logger.error(f"Error handling message: {str(e)}")
        return {"error": str(e)}
        
        
def create_message_handler(environment_manager):
    """
    Creates a message handler function bound to the given environment_manager
    
    Args:
        environment_manager: The EnvironmentManager instance
        
    Returns:
        Function that handles incoming messages
    """
    def message_handler(message_data):
        return handle_message(message_data, environment_manager)
    
    return message_handler 