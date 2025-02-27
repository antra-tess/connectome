"""
Socket.IO Listener
Handles incoming Socket.IO events from normalizing layers.
"""

import logging
from typing import Dict, Any

from config import SOCKET_HOST, SOCKET_PORT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def register_socket_events(sio, agent):
    """
    Register Socket.IO event handlers.
    
    Args:
        sio: Socket.IO server instance
        agent: Agent instance to handle messages
    """
    @sio.event
    def connect(sid, environ):
        """
        Handle client connection.
        
        Args:
            sid: Session ID
            environ: WSGI environment
        """
        logger.info(f"Client connected: {sid}")
    
    @sio.event
    def disconnect(sid):
        """
        Handle client disconnection.
        
        Args:
            sid: Session ID
        """
        logger.info(f"Client disconnected: {sid}")
    
    @sio.on('chat_message')
    def handle_chat_message(sid, data):
        """
        Handle incoming chat messages from normalizing layers.
        
        Args:
            sid: Session ID
            data: Message data including:
                - chat_id: Identifier for the conversation
                - user_id: Identifier for the user
                - content: Message content
                - platform: Optional platform identifier
        """
        try:
            logger.info(f"Received message from {data.get('user_id', 'unknown')} in chat {data.get('chat_id', 'unknown')}")
            
            # Validate required fields
            if 'chat_id' not in data:
                logger.error("Missing required field 'chat_id'")
                sio.emit('error', {'message': "Missing required field 'chat_id'"}, room=sid)
                return
                
            if 'user_id' not in data:
                logger.error("Missing required field 'user_id'")
                sio.emit('error', {'message': "Missing required field 'user_id'"}, room=sid)
                return
                
            if 'content' not in data:
                logger.error("Missing required field 'content'")
                sio.emit('error', {'message': "Missing required field 'content'"}, room=sid)
                return
            
            # Set default role if not provided
            if 'role' not in data:
                data['role'] = 'user'
            
            # Process message with agent
            response = agent.handle_message(data)
            
            # Send response if there is one
            if response:
                logger.info(f"Sending response to chat {response.get('chat_id', 'unknown')}")
                sio.emit('agent_response', response, room=sid)
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            sio.emit('error', {'message': f"Error processing message: {str(e)}"}, room=sid)
    
    @sio.on('register_platform')
    def handle_platform_registration(sid, data):
        """
        Handle platform registration from normalizing layers.
        
        Args:
            sid: Session ID
            data: Registration data including:
                - platform_id: Identifier for the platform
                - platform_name: Name of the platform
                - capabilities: List of platform capabilities
        """
        try:
            platform_id = data.get('platform_id')
            platform_name = data.get('platform_name', 'Unknown Platform')
            
            if not platform_id:
                logger.error("Missing required field 'platform_id'")
                sio.emit('error', {'message': "Missing required field 'platform_id'"}, room=sid)
                return
            
            logger.info(f"Platform registered: {platform_name} ({platform_id})")
            
            # Acknowledge registration
            sio.emit('registration_success', {
                'message': f"Successfully registered platform: {platform_name}",
                'platform_id': platform_id
            }, room=sid)
            
        except Exception as e:
            logger.error(f"Error registering platform: {str(e)}")
            sio.emit('error', {'message': f"Error registering platform: {str(e)}"}, room=sid)
    
    @sio.on('clear_context')
    def handle_clear_context(sid, data):
        """
        Handle context clearing request from normalizing layers.
        
        Args:
            sid: Session ID
            data: Request data including:
                - chat_id: Identifier for the conversation to clear
        """
        try:
            chat_id = data.get('chat_id')
            
            if not chat_id:
                logger.error("Missing required field 'chat_id'")
                sio.emit('error', {'message': "Missing required field 'chat_id'"}, room=sid)
                return
            
            # Clear context
            from context.context_manager import ContextManager
            context_manager = ContextManager()
            success = context_manager.clear_context(chat_id)
            
            if success:
                logger.info(f"Context cleared for chat {chat_id}")
                sio.emit('context_cleared', {'chat_id': chat_id}, room=sid)
            else:
                logger.warning(f"Failed to clear context for chat {chat_id}")
                sio.emit('error', {'message': f"Failed to clear context for chat {chat_id}"}, room=sid)
                
        except Exception as e:
            logger.error(f"Error clearing context: {str(e)}")
            sio.emit('error', {'message': f"Error clearing context: {str(e)}"}, room=sid)
    
    logger.info("Socket.IO event handlers registered") 