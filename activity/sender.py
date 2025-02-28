"""
Message Sender
Handles sending messages to normalizing adapters via Socket.IO clients.
"""

import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global reference to the SocketIO client
_socket_client = None


def initialize_sender(socket_client) -> None:
    """
    Initialize the message sender with a Socket.IO client.
    
    Args:
        socket_client: SocketIOClient instance to use for sending messages
    """
    global _socket_client
    _socket_client = socket_client
    logger.info("Message sender initialized")


def send_response(response: Dict[str, Any]) -> bool:
    """
    Send a response to an adapter.
    
    Args:
        response: Response data to send, including:
            - chat_id: Identifier for the conversation
            - content: Response content
            - adapter_id: Identifier for the target adapter
            
    Returns:
        True if response was sent successfully, False otherwise
    """
    if _socket_client is None:
        logger.error("Cannot send response - socket client not initialized")
        return False
    
    # Validate required fields
    if 'chat_id' not in response:
        logger.error("Missing required field 'chat_id' in response")
        return False
        
    if 'content' not in response:
        logger.error("Missing required field 'content' in response")
        return False
        
    if 'adapter_id' not in response:
        logger.error("Missing required field 'adapter_id' in response")
        return False
    
    # Send the response using the Socket.IO client
    return _socket_client.send_message(response)


def send_error(adapter_id: str, chat_id: str, error_message: str) -> bool:
    """
    Send an error message to an adapter.
    
    Args:
        adapter_id: Identifier for the target adapter
        chat_id: Identifier for the conversation
        error_message: Error message to send
        
    Returns:
        True if error message was sent successfully, False otherwise
    """
    response = {
        'adapter_id': adapter_id,
        'chat_id': chat_id,
        'content': f"Error: {error_message}",
        'type': 'error'
    }
    
    return send_response(response)


def send_typing_indicator(adapter_id: str, chat_id: str, is_typing: bool = True) -> bool:
    """
    Send a typing indicator to an adapter.
    
    Args:
        adapter_id: Identifier for the target adapter
        chat_id: Identifier for the conversation
        is_typing: Whether the bot is typing or has stopped typing
        
    Returns:
        True if typing indicator was sent successfully, False otherwise
    """
    if _socket_client is None:
        logger.error("Cannot send typing indicator - socket client not initialized")
        return False
    
    indicator = {
        'adapter_id': adapter_id,
        'chat_id': chat_id,
        'is_typing': is_typing,
        'type': 'typing_indicator'
    }
    
    try:
        return _socket_client.send_message(indicator)
    except Exception as e:
        logger.error(f"Failed to send typing indicator: {str(e)}")
        return False 