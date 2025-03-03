"""
Message Service
Handles the flow of messages between the Activity Layer and the Environment Layer.
"""

import logging
from typing import Dict, Any, List, Callable, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MessageService:
    """
    MessageService serves as the central point for handling message flow 
    between the Activity Layer and other components.
    """
    
    def __init__(self):
        self.observers = []
        logger.debug("MessageService initialized")
    
    def register_observer(self, observer):
        """
        Register an object to be notified of new messages.
        The observer should have an observe_message method.
        """
        if observer not in self.observers:
            self.observers.append(observer)
            logger.debug(f"Added message observer: {observer}")
    
    def unregister_observer(self, observer):
        """Remove an observer from the notification list"""
        if observer in self.observers:
            self.observers.remove(observer)
            logger.debug(f"Removed message observer: {observer}")
    
    def process_message(self, user_id, message_text, message_id=None, platform=None, env_id=None):
        """
        Process an incoming message and notify all observers
        
        Args:
            user_id: ID of the user sending the message
            message_text: Content of the message
            message_id: Optional ID for the message
            platform: Optional platform identifier (e.g., 'telegram', 'slack')
            env_id: Optional environment ID where the message is processed
            
        Returns:
            Dict containing status information
        """
        # Create a message object
        message_data = {
            "user_id": user_id,
            "message_text": message_text,
            "message_id": message_id,
            "platform": platform,
            "env_id": env_id,
            "timestamp": time.time()
        }
        
        logger.info(f"MessageService processing message from user {user_id}")
        
        # Notify all observers about the new message
        self._notify_observers(message_data)
        
        return {"status": "message_processed"}
    
    def _notify_observers(self, message_data):
        """Notify all observers about a new message"""
        for observer in self.observers:
            try:
                # Try the standard observe_message interface
                if hasattr(observer, 'observe_message'):
                    observer.observe_message(message_data)
                    logger.debug(f"Notified observer {observer} about new message using observe_message")
                # Try the process_message interface as a fallback
                elif hasattr(observer, 'process_message'):
                    observer.process_message(
                        user_id=message_data.get("user_id"),
                        message_text=message_data.get("message_text"),
                        message_id=message_data.get("message_id"),
                        platform=message_data.get("platform"),
                        env_id=message_data.get("env_id")
                    )
                    logger.debug(f"Notified observer {observer} about new message using process_message")
                else:
                    logger.warning(f"Observer {observer} has no compatible message handling method")
            except Exception as e:
                logger.error(f"Error notifying observer {observer}: {str(e)}") 