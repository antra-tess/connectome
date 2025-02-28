"""
Messaging Environment
Environment for messaging functionality, providing tools for sending messages.
"""

import logging
from typing import Dict, Any, Optional, List

from environments.base import Environment
from activity.sender import send_response, send_typing_indicator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MessagingEnvironment(Environment):
    """
    Environment for messaging functionality.
    
    Provides tools for sending messages, typing indicators, and other
    messaging-related functionality.
    """
    
    def __init__(self, env_id: str = "messaging", name: str = "Messaging Environment", 
                 description: str = "Environment for sending messages and notifications"):
        """
        Initialize the messaging environment.
        
        Args:
            env_id: Unique identifier for this environment
            name: Human-readable name for this environment
            description: Description of this environment's purpose
        """
        super().__init__(env_id, name, description)
        # Initialize message tracking
        self._message_history: Dict[str, List[Dict[str, Any]]] = {}  # chat_id -> messages
        self._register_messaging_tools()
        logger.info(f"Created environment: {name} ({env_id})")
    
    def _register_messaging_tools(self):
        """Register all messaging-related tools."""
        
        @self.register_tool(
            name="send_message",
            description="Send a message to the current chat",
            parameter_descriptions={
                "chat_id": "ID of the chat to send the message to",
                "content": "Message content to send",
                "adapter_id": "ID of the adapter to send the message through"
            }
        )
        def send_message(chat_id: str, content: str, adapter_id: str) -> Dict[str, Any]:
            """
            Send a message to the specified chat.
            
            Args:
                chat_id: ID of the chat to send the message to
                content: Message content to send
                adapter_id: ID of the adapter to send the message through
                
            Returns:
                Dictionary with information about the sent message
            """
            logger.info(f"Sending message to chat {chat_id} via adapter {adapter_id}")
            
            response = {
                'chat_id': chat_id,
                'content': content,
                'adapter_id': adapter_id
            }
            
            success = send_response(response)
            
            # Record the sent message for state tracking
            self.record_chat_message(chat_id, "agent", content, adapter_id, "assistant")
            
            return {
                'success': success,
                'chat_id': chat_id,
                'adapter_id': adapter_id,
                'content_length': len(content)
            }
        
        @self.register_tool(
            name="send_typing_indicator",
            description="Send a typing indicator to show the agent is processing",
            parameter_descriptions={
                "chat_id": "ID of the chat to send the typing indicator to",
                "adapter_id": "ID of the adapter to send the indicator through",
                "is_typing": "Whether the agent is typing (True) or has stopped typing (False)"
            }
        )
        def send_typing(chat_id: str, adapter_id: str, is_typing: bool = True) -> Dict[str, Any]:
            """
            Send a typing indicator to the specified chat.
            
            Args:
                chat_id: ID of the chat to send the typing indicator to
                adapter_id: ID of the adapter to send the indicator through
                is_typing: Whether the agent is typing
                
            Returns:
                Dictionary with information about the sent indicator
            """
            logger.info(f"Sending typing indicator to chat {chat_id} via adapter {adapter_id}: {'typing' if is_typing else 'stopped typing'}")
            
            success = send_typing_indicator(adapter_id, chat_id, is_typing)
            
            return {
                'success': success,
                'chat_id': chat_id,
                'adapter_id': adapter_id,
                'is_typing': is_typing
            }

    def record_chat_message(self, chat_id: str, user_id: str, content: str, 
                       adapter_id: Optional[str] = None, role: str = "user") -> None:
        """
        Record a chat message in the messaging environment's internal history.
        
        This is specifically for messaging domain modeling, separate from
        the interface layer's context management.
        
        Args:
            chat_id: ID of the chat the message belongs to
            user_id: ID of the user who sent the message
            content: Content of the message
            adapter_id: ID of the adapter the message was sent through
            role: Role of the message sender (user, assistant, system, tool_result)
        """
        # Initialize chat history if needed
        if chat_id not in self._message_history:
            self._message_history[chat_id] = []
            
        # Create message record
        message = {
            'chat_id': chat_id,
            'user_id': user_id,
            'content': content,
            'adapter_id': adapter_id,
            'role': role,
            'timestamp': logging.Formatter.formatTime(logging.LogRecord("", 0, "", 0, None, None, None))
        }
        
        # Add to history
        self._message_history[chat_id].append(message)
        
        # Limit history size (optional)
        max_history = 100  # Configurable
        if len(self._message_history[chat_id]) > max_history:
            self._message_history[chat_id] = self._message_history[chat_id][-max_history:]
            
    def render_state_for_context(self) -> Dict[str, Any]:
        """
        Render the messaging environment's state for inclusion in the agent's context.
        
        This method provides a complete representation of the environment state
        that can be directly included in the agent's prompt without the Interface Layer
        needing to understand messaging-specific details.
        
        Returns:
            Dictionary with formatted messaging state
        """
        # Get base state info
        state = super().render_state_for_context()
        state["type"] = "messaging"
        
        # Build the fully formatted state text that can be included directly in prompts
        state["formatted_state_text"] = self._build_formatted_state_text()
        
        # Include raw data for potential specialized handling if needed
        if not self._message_history:
            state["state_summary"] = "No active conversations"
            state["active_chats"] = []
            return state
        
        # Build chat summaries
        active_chats = []
        for chat_id, messages in self._message_history.items():
            if not messages:
                continue
                
            # Get participant information
            participants = list(set(msg.get('user_id') for msg in messages 
                              if msg.get('user_id') != 'agent' and msg.get('role') != 'tool_result'))
                
            # Get message counts
            user_msgs = sum(1 for msg in messages if msg.get('role') == 'user')
            agent_msgs = sum(1 for msg in messages if msg.get('role') == 'assistant')
            
            # Create chat summary
            chat_info = {
                'chat_id': chat_id,
                'participants': participants,
                'message_count': len(messages),
                'user_messages': user_msgs,
                'agent_messages': agent_msgs,
                'last_updated': messages[-1].get('timestamp', 'unknown')
            }
            active_chats.append(chat_info)
        
        # Update state with conversation summaries
        state.update({
            "state_summary": f"Managing {len(active_chats)} active conversations",
            "active_chats": active_chats
        })
        
        return state
    
    def _build_formatted_state_text(self) -> str:
        """
        Build a formatted text representation of the messaging environment state.
        
        This creates a human-readable summary of all active chats that can be
        directly included in the agent's prompt.
        
        Returns:
            String representation ready for inclusion in prompt
        """
        if not self._message_history:
            return "No active conversations."
        
        lines = []
        lines.append(f"Managing {len(self._message_history)} active conversations:")
        
        for chat_id, messages in self._message_history.items():
            if not messages:
                continue
                
            # Get participant information
            participants = list(set(msg.get('user_id') for msg in messages 
                              if msg.get('user_id') != 'agent' and msg.get('role') != 'tool_result'))
            
            # Get message counts and timing
            message_count = len(messages)
            last_update = messages[-1].get('timestamp', 'unknown time')
            
            # Create a summary line for this chat
            participants_str = ", ".join(participants) if participants else "no identified participants"
            lines.append(f"- Chat {chat_id} with {participants_str}: {message_count} messages (last updated: {last_update})")
        
        return "\n".join(lines)
    
    def get_chat_state(self, chat_id: str, max_messages: int = 20) -> Dict[str, Any]:
        """
        Get the state of a specific chat.
        
        Args:
            chat_id: ID of the chat to get state for
            max_messages: Maximum number of messages to include
            
        Returns:
            Dictionary with chat state information
        """
        if chat_id not in self._message_history:
            return {
                "chat_id": chat_id,
                "exists": False,
                "message_count": 0,
                "formatted_conversation": ""
            }
            
        # Get recent messages
        messages = self._message_history[chat_id][-max_messages:]
        
        # Format conversation
        formatted_conversation = self._format_multi_user_chat(messages)
        
        return {
            "chat_id": chat_id,
            "exists": True,
            "message_count": len(messages),
            "formatted_conversation": formatted_conversation,
            "participants": list(set(msg.get('user_id') for msg in messages 
                               if msg.get('user_id') != 'agent'))
        }
    
    def _format_multi_user_chat(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format a list of messages from multiple users into a text format
        suitable for inclusion in the agent's context.
        
        Args:
            messages: List of message dictionaries
                
        Returns:
            Formatted multi-user chat as a string
        """
        formatted_lines = []
        
        # Group consecutive messages from the same user
        current_user = None
        current_content_lines = []
        
        for msg in messages:
            user_id = msg.get('user_id', 'unknown')
            content = msg.get('content', '')
            role = msg.get('role', 'user')
            
            # Skip empty messages
            if not content.strip():
                continue
            
            # Handle tool results differently
            if role == 'tool_result':
                formatted_lines.append(f"[Tool Result]: {content}")
                continue
                
            # If this is a new user, flush the previous user's messages
            if user_id != current_user and current_content_lines:
                if current_user == 'agent':
                    formatted_lines.append(f"[Assistant]: {' '.join(current_content_lines)}")
                else:
                    formatted_lines.append(f"[User {current_user}]: {' '.join(current_content_lines)}")
                current_content_lines = []
            
            # Set current user and add this content
            current_user = user_id
            current_content_lines.append(content)
        
        # Flush the last user's messages
        if current_content_lines:
            if current_user == 'agent':
                formatted_lines.append(f"[Assistant]: {' '.join(current_content_lines)}")
            else:
                formatted_lines.append(f"[User {current_user}]: {' '.join(current_content_lines)}")
        
        # Join all lines with newlines
        return "\n".join(formatted_lines) 