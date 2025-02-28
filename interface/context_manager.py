"""
Context Manager
Manages conversation context, including storage and retrieval of messages.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

from config import MAX_HISTORY_MESSAGES, STORAGE_PATH
from bot_framework.interface.formatter import format_multi_user_chat

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages conversation context, including storage and retrieval of messages.
    
    This version uses a simple file-based storage system. In a production
    environment, you might want to use a database instead.
    """
    
    def __init__(self, storage_path: str = STORAGE_PATH):
        """
        Initialize the context manager.
        
        Args:
            storage_path: Path to the storage directory
        """
        self.storage_path = storage_path
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        logger.info(f"Context manager initialized with storage path: {storage_path}")
    
    def save_message(self, message_data: Dict[str, Any]) -> None:
        """
        Save a message to storage.
        
        Args:
            message_data: Dictionary containing message data:
                - chat_id: Identifier for the conversation
                - user_id: Identifier for the user
                - content: Message content
                - role: Role of the message (user, assistant, tool_result)
                - timestamp: Optional timestamp
        """
        chat_id = message_data.get('chat_id')
        if not chat_id:
            logger.error("Cannot save message without chat_id")
            return
        
        # Ensure chat_id is safe to use in a filename
        safe_chat_id = self._sanitize_chat_id(chat_id)
        
        # Add timestamp if not present
        if 'timestamp' not in message_data:
            import time
            message_data['timestamp'] = time.time()
        
        # Create chat log file path
        chat_log_path = os.path.join(self.storage_path, f"{safe_chat_id}.json")
        
        try:
            # Load existing messages
            messages = []
            if os.path.exists(chat_log_path):
                with open(chat_log_path, 'r') as f:
                    messages = json.load(f)
            
            # Append new message
            messages.append(message_data)
            
            # Save messages back to file
            with open(chat_log_path, 'w') as f:
                json.dump(messages, f, indent=2)
                
            logger.info(f"Saved message for user {message_data.get('user_id')} in chat {chat_id}")
            
        except Exception as e:
            logger.error(f"Error saving message: {str(e)}")
    
    def get_context(self, chat_id: str, max_messages: int = MAX_HISTORY_MESSAGES) -> List[Dict[str, Any]]:
        """
        Get the conversation context for a chat.
        
        Args:
            chat_id: Identifier for the conversation
            max_messages: Maximum number of messages to include in context
            
        Returns:
            List of messages representing the conversation context
        """
        # Ensure chat_id is safe to use in a filename
        safe_chat_id = self._sanitize_chat_id(chat_id)
        
        # Create chat log file path
        chat_log_path = os.path.join(self.storage_path, f"{safe_chat_id}.json")
        
        try:
            # Load messages
            messages = []
            if os.path.exists(chat_log_path):
                with open(chat_log_path, 'r') as f:
                    messages = json.load(f)
            
            # Sort by timestamp
            messages.sort(key=lambda x: x.get('timestamp', 0))
            
            # Limit to max_messages
            if len(messages) > max_messages:
                messages = messages[-max_messages:]
                
            logger.info(f"Retrieved {len(messages)} messages for chat {chat_id}")
            
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def get_formatted_context(self, chat_id: str, max_messages: int = MAX_HISTORY_MESSAGES) -> str:
        """
        Get the formatted conversation context for a chat.
        
        Args:
            chat_id: Identifier for the conversation
            max_messages: Maximum number of messages to include in context
            
        Returns:
            Formatted conversation context as a string
        """
        messages = self.get_context(chat_id, max_messages)
        return format_multi_user_chat(messages)
    
    def clear_context(self, chat_id: str) -> bool:
        """
        Clear the conversation context for a chat.
        
        Args:
            chat_id: Identifier for the conversation
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure chat_id is safe to use in a filename
        safe_chat_id = self._sanitize_chat_id(chat_id)
        
        # Create chat log file path
        chat_log_path = os.path.join(self.storage_path, f"{safe_chat_id}.json")
        
        try:
            if os.path.exists(chat_log_path):
                os.remove(chat_log_path)
                logger.info(f"Cleared context for chat {chat_id}")
                return True
            else:
                logger.warning(f"No context found for chat {chat_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error clearing context: {str(e)}")
            return False
    
    def _sanitize_chat_id(self, chat_id: str) -> str:
        """
        Sanitize a chat ID to make it safe for use in a filename.
        
        Args:
            chat_id: Chat ID to sanitize
            
        Returns:
            Sanitized chat ID
        """
        # Replace any characters that are not alphanumeric, dash, or underscore
        import re
        return re.sub(r'[^a-zA-Z0-9-_]', '_', str(chat_id)) 