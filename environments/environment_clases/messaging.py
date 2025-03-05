"""
Messaging Environment
Environment for messaging functionality, providing tools for sending messages.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import uuid

from environments.base import Environment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MessagingEnvironment(Environment):
    """
    Environment for messaging functionality.
    
    Provides tools for sending messages, typing indicators, and other
    messaging-related functionality.
    """
    EVENT_TYPES = [
        "messageReceived", 
        "messageSent", 
        "messageUpdated", 
        "messageDeleted", 
        "reactionAdded", 
        "reactionRemoved", 
        "fetchHistory"
    ]
    
    # Define which events should trigger agent attention
    # Note: messageReceived is handled specially to check for mentions
    # before triggering agent attention in group chats
    ATTENTION_SIGNALS = [
        # Direct messages to agent always need attention
        "messageSent",
        # Messages in a direct chat with the agent
        # Note: messageReceived is conditionally handled based on mentions in group chats
    ]
    
    # Messages starting with these markers are ignored
    IGNORE_MARKERS = ["."]
    
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
        # Initialize observer callbacks
        self._message_observers: List[Callable] = []
        self._typing_observers: List[Callable] = []
        self._register_messaging_tools()
        logger.info(f"Created environment: {name} ({env_id})")
    
    def register_message_observer(self, callback: Callable[[Dict[str, Any]], bool]) -> None:
        """
        Register a callback for message events.
        
        The callback will receive a standardized format:
        {
          "action": "sendMessage",  // Or other action types
          "payload": {
            "conversationId": "C123",
            "text": "Hello World!",
            // Other action-specific fields
          },
          "adapter_id": "adapter_id"
        }
        
        Args:
            callback: Function to call when messages are published
        """
        if callback not in self._message_observers:
            self._message_observers.append(callback)
            logger.debug(f"Registered message observer for environment {self.id}")
        else:
            logger.debug(f"Message observer already registered for environment {self.id}")
    
    def register_typing_observer(self, callback: Callable[[Dict[str, Any]], bool]) -> None:
        """
        Register a callback for typing indicator events.
        
        The callback will receive a standardized format:
        {
          "action": "typingIndicator",
          "payload": {
            "conversationId": "C123",
            "isTyping": true
          },
          "adapter_id": "adapter_id"
        }
        
        Args:
            callback: Function to call when typing indicators are published
        """
        if callback not in self._typing_observers:
            self._typing_observers.append(callback)
            logger.debug(f"Registered typing observer for environment {self.id}")
        else:
            logger.debug(f"Typing observer already registered for environment {self.id}")
    
    def publish_message(self, message_data: Dict[str, Any]) -> bool:
        """
        Publish a message to registered message observers.
        
        Requires standardized message actions in the format:
        {
          "action": "sendMessage",
          "payload": {
            "conversationId": "C123",
            "text": "Hello World!",
            ...
          }
        }
        
        Args:
            message_data: Message data to publish
            
        Returns:
            True if the message was published, False otherwise
        """
        try:
            # Extract adapter ID
            adapter_id = message_data.get('adapter_id')
            if not adapter_id:
                logger.error("Missing adapter_id in published message")
                return False
                
            # Validate message format
            if 'action' not in message_data or 'payload' not in message_data:
                logger.error("Invalid message format: requires 'action' and 'payload' fields")
                return False
            
            # Ensure payload has required fields based on action type
            action = message_data['action']
            payload = message_data['payload']
            
            if action == 'sendMessage':
                if 'conversationId' not in payload or 'text' not in payload:
                    logger.error("Invalid sendMessage payload: requires 'conversationId' and 'text' fields")
                    return False
            elif action == 'editMessage':
                if 'conversationId' not in payload or 'messageId' not in payload or 'text' not in payload:
                    logger.error("Invalid editMessage payload: requires 'conversationId', 'messageId', and 'text' fields")
                    return False
            elif action == 'deleteMessage':
                if 'conversationId' not in payload or 'messageId' not in payload:
                    logger.error("Invalid deleteMessage payload: requires 'conversationId' and 'messageId' fields")
                    return False
            elif action == 'addReaction' or action == 'removeReaction':
                if 'conversationId' not in payload or 'messageId' not in payload or 'emoji' not in payload:
                    logger.error(f"Invalid {action} payload: requires 'conversationId', 'messageId', and 'emoji' fields")
                    return False
            elif action == 'typingIndicator':
                if 'conversationId' not in payload or 'isTyping' not in payload:
                    logger.error("Invalid typingIndicator payload: requires 'conversationId' and 'isTyping' fields")
                    return False
                
            # Notify all observers
            for observer in self._message_observers:
                try:
                    observer(message_data)
                except Exception as e:
                    logger.error(f"Error notifying message observer: {str(e)}")
                    
            logger.info(f"Published {message_data['action']} action to {len(self._message_observers)} observers")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing message: {str(e)}")
            return False
    
    def publish_typing_indicator(self, adapter_id: str, chat_id: str, is_typing: bool = True) -> bool:
        """
        Publish a typing indicator to registered typing observers.
        
        Uses standardized format:
        {
          "action": "typingIndicator",
          "payload": {
            "conversationId": "C123",
            "isTyping": true
          }
        }
        
        Args:
            adapter_id: ID of the adapter to send through
            chat_id: ID of the chat to send to
            is_typing: Whether the agent is typing (True) or has stopped typing (False)
            
        Returns:
            True if the indicator was published, False otherwise
        """
        try:
            # Create typing indicator data in standardized format
            typing_data = {
                "action": "typingIndicator",
                "payload": {
                    "conversationId": chat_id,
                    "isTyping": is_typing
                },
                "adapter_id": adapter_id
            }
            
            # Publish to typing observers
            for observer in self._typing_observers:
                try:
                    observer(typing_data)
                except Exception as e:
                    logger.error(f"Error notifying typing observer: {str(e)}")
                    
            logger.debug(f"Published typing indicator ({is_typing}) for chat {chat_id} to {len(self._typing_observers)} observers")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing typing indicator: {str(e)}")
            return False
    
    def _register_messaging_tools(self):
        """
        Register messaging-specific tools for the environment.
        """
        @self.register_tool(
            name="send_message",
            description="Send a message to the current chat",
            parameter_descriptions={
                "chat_id": "ID of the chat to send the message to",
                "content": "Message content to send",
                "adapter_id": "ID of the adapter to send the message through",
                "thread_id": "Optional thread ID if this message is part of a thread",
                "mentions": "Optional list of user IDs to mention",
                "parse_mode": "Optional parsing mode (e.g., 'markdown', 'plain')",
                "disable_link_previews": "Optional flag to disable link previews",
                "disable_mentions": "Optional flag to disable mentions processing"
            }
        )
        def send_message(
            chat_id: str, 
            content: str, 
            adapter_id: str,
            thread_id: Optional[str] = None,
            mentions: Optional[List[Dict[str, str]]] = None,
            parse_mode: Optional[str] = None,
            disable_link_previews: Optional[bool] = None,
            disable_mentions: Optional[bool] = None
        ) -> Dict[str, Any]:
            """
            Send a message to a chat through an adapter.
            
            Args:
                chat_id: ID of the chat to send to
                content: Message content
                adapter_id: ID of the adapter to send through
                thread_id: Optional thread ID for threaded replies
                mentions: Optional list of user mentions [{userId: "U123", displayName: "User"}]
                parse_mode: Optional parsing mode for the message
                disable_link_previews: Whether to disable link previews
                disable_mentions: Whether to disable mentions processing
                
            Returns:
                Result of the message sending operation
            """
            logger.info(f"Sending message to chat {chat_id} via adapter {adapter_id}")
            
            # Build options object if any options are provided
            options = {}
            if parse_mode:
                options["parseMode"] = parse_mode
            if disable_link_previews is not None:
                options["disableLinkPreviews"] = disable_link_previews
            if disable_mentions is not None:
                options["disableMentions"] = disable_mentions
                
            # Format in the expected structure
            message_data = {
                "action": "sendMessage",
                "payload": {
                    "conversationId": chat_id,
                    "text": content,
                    "attachments": []
                }
            }
            
            # Add optional fields
            if thread_id:
                message_data["payload"]["threadId"] = thread_id
                
            if mentions and isinstance(mentions, list):
                message_data["payload"]["mentions"] = mentions
                
            if options:
                message_data["payload"]["options"] = options
            
            # Add source adapter ID
            message_data["adapter_id"] = adapter_id
            
            # Publish the message
            result = self.publish_message(message_data)
            
            if result:
                # Record in message history as an agent message
                message_id = self.record_chat_message(
                    chat_id=chat_id,
                    user_id="agent",
                    content=content,
                    adapter_id=adapter_id,
                    role="assistant",
                    thread_id=thread_id
                )
                
                # Return success with message ID
                return {
                    "status": "success",
                    "message_id": message_id,
                    "content": content
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to send message"
                }
        
        @self.register_tool(
            name="edit_message",
            description="Edit an existing message",
            parameter_descriptions={
                "chat_id": "ID of the chat containing the message",
                "message_id": "ID of the message to edit",
                "content": "New content for the message",
                "adapter_id": "ID of the adapter to send the edit through"
            }
        )
        def edit_message(chat_id: str, message_id: str, content: str, adapter_id: str) -> Dict[str, Any]:
            """
            Edit an existing message.
            
            Args:
                chat_id: ID of the chat containing the message
                message_id: ID of the message to edit
                content: New content for the message
                adapter_id: ID of the adapter to send through
                
            Returns:
                Result of the edit operation
            """
            logger.info(f"Editing message {message_id} in chat {chat_id} via adapter {adapter_id}")
            
            # Format in the expected structure
            edit_data = {
                "action": "editMessage",
                "payload": {
                    "conversationId": chat_id,
                    "messageId": message_id,
                    "text": content,
                    "attachments": []
                },
                "adapter_id": adapter_id
            }
            
            # Publish the edit
            result = self.publish_message(edit_data)
            
            if result:
                # Update local message history
                if chat_id in self._message_history:
                    for i, msg in enumerate(self._message_history[chat_id]):
                        if msg.get('message_id') == message_id:
                            # Store original content in edit history
                            if 'edit_history' not in msg:
                                msg['edit_history'] = []
                                
                            msg['edit_history'].append(msg['content'])
                            msg['content'] = content
                            msg['edited'] = True
                            msg['edit_timestamp'] = int(datetime.now().timestamp() * 1000)
                            
                            # Update in-place
                            self._message_history[chat_id][i] = msg
                            break
                
                # Return success
                return {
                    "status": "success",
                    "message_id": message_id,
                    "content": content
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to edit message"
                }
                
        @self.register_tool(
            name="delete_message",
            description="Delete a message from the chat",
            parameter_descriptions={
                "chat_id": "ID of the chat containing the message",
                "message_id": "ID of the message to delete",
                "adapter_id": "ID of the adapter to send the deletion through"
            }
        )
        def delete_message(chat_id: str, message_id: str, adapter_id: str) -> Dict[str, Any]:
            """
            Delete a message from the chat.
            
            Args:
                chat_id: ID of the chat containing the message
                message_id: ID of the message to delete
                adapter_id: ID of the adapter to send through
                
            Returns:
                Result of the deletion operation
            """
            logger.info(f"Deleting message {message_id} in chat {chat_id} via adapter {adapter_id}")
            
            # Format in the expected structure
            delete_data = {
                "action": "deleteMessage",
                "payload": {
                    "conversationId": chat_id,
                    "messageId": message_id
                },
                "adapter_id": adapter_id
            }
            
            # Publish the deletion
            result = self.publish_message(delete_data)
            
            if result:
                # Mark as deleted in local message history, but preserve for context
                if chat_id in self._message_history:
                    for i, msg in enumerate(self._message_history[chat_id]):
                        if msg.get('message_id') == message_id:
                            # Store original content in deletion history
                            if 'deletion_history' not in msg:
                                msg['deletion_history'] = []
                                
                            msg['deletion_history'].append(msg['content'])
                            msg['content'] = "[Message deleted]"
                            msg['deleted'] = True
                            msg['deletion_timestamp'] = int(datetime.now().timestamp() * 1000)
                            
                            # Update in-place
                            self._message_history[chat_id][i] = msg
                            break
                
                # Return success
                return {
                    "status": "success",
                    "message_id": message_id
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to delete message"
                }
                
        @self.register_tool(
            name="add_reaction",
            description="Add a reaction to a message",
            parameter_descriptions={
                "chat_id": "ID of the chat containing the message",
                "message_id": "ID of the message to react to",
                "emoji": "Emoji reaction to add (e.g., ':+1:' or '👍')",
                "adapter_id": "ID of the adapter to send the reaction through"
            }
        )
        def add_reaction(chat_id: str, message_id: str, emoji: str, adapter_id: str) -> Dict[str, Any]:
            """
            Add a reaction to a message.
            
            Args:
                chat_id: ID of the chat containing the message
                message_id: ID of the message to react to
                emoji: Emoji reaction to add
                adapter_id: ID of the adapter to send through
                
            Returns:
                Result of the reaction operation
            """
            logger.info(f"Adding reaction {emoji} to message {message_id} in chat {chat_id}")
            
            # Format in the expected structure
            reaction_data = {
                "action": "addReaction",
                "payload": {
                    "conversationId": chat_id,
                    "messageId": message_id,
                    "emoji": emoji
                },
                "adapter_id": adapter_id
            }
            
            # Publish the reaction
            result = self.publish_message(reaction_data)
            
            # For local processing, track it as agent's reaction
            if result:
                # Update local message history
                if chat_id in self._message_history:
                    for i, msg in enumerate(self._message_history[chat_id]):
                        if msg.get('message_id') == message_id:
                            # Initialize reactions dict if needed
                            if 'reactions' not in msg:
                                msg['reactions'] = {}
                                
                            # Add agent to this reaction
                            if emoji not in msg['reactions']:
                                msg['reactions'][emoji] = []
                                
                            if "agent" not in msg['reactions'][emoji]:
                                msg['reactions'][emoji].append("agent")
                                
                            # Update in-place
                            self._message_history[chat_id][i] = msg
                            break
                
                # Return success
                return {
                    "status": "success",
                    "message_id": message_id,
                    "emoji": emoji
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to add reaction"
                }
                
        @self.register_tool(
            name="remove_reaction",
            description="Remove a reaction from a message",
            parameter_descriptions={
                "chat_id": "ID of the chat containing the message",
                "message_id": "ID of the message to remove reaction from",
                "emoji": "Emoji reaction to remove (e.g., ':+1:' or '👍')",
                "adapter_id": "ID of the adapter to send the removal through"
            }
        )
        def remove_reaction(chat_id: str, message_id: str, emoji: str, adapter_id: str) -> Dict[str, Any]:
            """
            Remove a reaction from a message.
            
            Args:
                chat_id: ID of the chat containing the message
                message_id: ID of the message to remove reaction from
                emoji: Emoji reaction to remove
                adapter_id: ID of the adapter to send through
                
            Returns:
                Result of the reaction removal operation
            """
            logger.info(f"Removing reaction {emoji} from message {message_id} in chat {chat_id}")
            
            # Format in the expected structure
            reaction_data = {
                "action": "removeReaction",
                "payload": {
                    "conversationId": chat_id,
                    "messageId": message_id,
                    "emoji": emoji
                },
                "adapter_id": adapter_id
            }
            
            # Publish the reaction removal
            result = self.publish_message(reaction_data)
            
            # For local processing, track the agent's reaction removal
            if result:
                # Update local message history
                if chat_id in self._message_history:
                    for i, msg in enumerate(self._message_history[chat_id]):
                        if msg.get('message_id') == message_id:
                            if 'reactions' in msg and emoji in msg['reactions']:
                                # Remove agent from reaction list
                                if "agent" in msg['reactions'][emoji]:
                                    msg['reactions'][emoji].remove("agent")
                                    
                                    # Clean up empty reaction lists
                                    if not msg['reactions'][emoji]:
                                        del msg['reactions'][emoji]
                                        
                                    # Clean up empty reactions dict
                                    if not msg['reactions']:
                                        del msg['reactions']
                                        
                                    # Update in-place
                                    self._message_history[chat_id][i] = msg
                                    break
                
                # Return success
                return {
                    "status": "success",
                    "message_id": message_id,
                    "emoji": emoji
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to remove reaction"
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
            Send a typing indicator to the chat.
            
            Args:
                chat_id: ID of the chat to send to
                adapter_id: ID of the adapter to send through
                is_typing: Whether the agent is typing or has stopped
                
            Returns:
                Result of the operation
            """
            logger.info(f"Sending typing indicator to chat {chat_id} via adapter {adapter_id}: typing={is_typing}")
            
            result = self.publish_typing_indicator(
                adapter_id=adapter_id,
                chat_id=chat_id,
                is_typing=is_typing
            )
            
            if result:
                return {
                    "status": "success",
                    "typing_indicator_sent": is_typing
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to send typing indicator"
                }

    def record_chat_message(
        self, 
        chat_id: str, 
        user_id: str, 
        content: str, 
        adapter_id: Optional[str] = None, 
        role: str = "user", 
        message_id: Optional[str] = None,
        timestamp: Optional[int] = None,
        thread_id: Optional[str] = None,
        display_name: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        is_history: bool = False
    ) -> None:
        """
        Record a chat message in the message history.
        
        Args:
            chat_id: ID of the chat/conversation
            user_id: ID of the message sender
            content: Message content
            adapter_id: ID of the adapter if applicable
            role: Role of the message sender (user or assistant)
            message_id: Unique ID for the message
            timestamp: Message timestamp in milliseconds since epoch
            thread_id: Thread ID if part of a thread
            display_name: Display name of the sender
            attachments: List of message attachments
            is_history: Whether this is a historical message being loaded
        """
        # Initialize chat history if needed
        if chat_id not in self._message_history:
            self._message_history[chat_id] = []
            
        # Format timestamp if not provided
        if not timestamp:
            timestamp = int(datetime.now().timestamp() * 1000)  # Convert to milliseconds
            
        # Generate message ID if not provided
        if not message_id:
            message_id = str(uuid.uuid4())
            
        # Create message object with all metadata
        message = {
            "message_id": message_id,
            "chat_id": chat_id,
            "user_id": user_id,
            "content": content,
            "role": role,
            "timestamp": timestamp,
        }
        
        # Add optional fields if provided
        if adapter_id:
            message["adapter_id"] = adapter_id
            
        if thread_id:
            message["thread_id"] = thread_id
            
        if display_name:
            message["display_name"] = display_name
            
        if attachments and isinstance(attachments, list):
            message["attachments"] = attachments
            
        # Add message to history
        self._message_history[chat_id].append(message)
        
        # Log the action
        if is_history:
            logger.debug(f"Loaded historical message {message_id} in chat {chat_id}")
        else:
            logger.info(f"Recorded new message {message_id} from {user_id} in chat {chat_id}")
        
        # Keep history sorted by timestamp
        self._message_history[chat_id].sort(key=lambda x: x.get('timestamp', 0))
        
        # Only notify observers for new user messages, not for historical or agent messages
        # This prevents circular notification loops when the agent sends a message
        if role == "user" and not is_history:
            # Create update data for the new message
            update_data = {
                'type': 'messageReceived',
                'messageId': message_id,
                'conversationId': chat_id,
                'sender': {
                    'userId': user_id,
                    'displayName': display_name
                },
                'text': content,
                'timestamp': timestamp,
                'adapter_id': adapter_id
            }
            
            # Add thread_id if present
            if thread_id:
                update_data['threadId'] = thread_id
                
            # Add attachments if present
            if attachments:
                update_data['attachments'] = attachments
            
            # Notify observers about the new message
            logger.debug(f"Notifying observers about new message in chat {chat_id} from user {user_id}")
            self.notify_observers(update_data)
        
        # Return the message ID for reference
        return message_id
    
    def render_state_for_context(self) -> Dict[str, Any]:
        """
        Render the current messaging environment state for inclusion in the LLM context.
        
        Returns:
            Dictionary containing formatted conversation history for each chat
        """
        context_data = {}
        
        # Format each active conversation
        for chat_id, messages in self._message_history.items():
            if messages:
                # Format conversation
                formatted_conversation = self._format_multi_user_chat(chat_id)
                
                # Add to context data
                context_data[f"conversation_{chat_id}"] = formatted_conversation
        
        return context_data
    
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
        
        formatted_parts = ["MESSAGING ENVIRONMENT STATE:"]
        
        # Format each active conversation
        for chat_id, messages in self._message_history.items():
            if not messages:
                continue
                
            # Get participant information
            participants = []
            for msg in messages:
                user_id = msg.get('user_id')
                display_name = msg.get('display_name')
                if user_id not in ['agent', 'system'] and user_id not in [p.split('(')[1][:-1] if '(' in p else p for p in participants]:
                    if display_name:
                        participants.append(f"{display_name} ({user_id})")
                    else:
                        participants.append(user_id)
            
            # Get message counts
            message_count = len(messages)
            user_msgs = sum(1 for msg in messages if msg.get('role') == 'user')
            agent_msgs = sum(1 for msg in messages if msg.get('role') == 'assistant')
            
            # Get last update time
            last_message = max(messages, key=lambda x: x.get('timestamp', 0))
            last_timestamp = last_message.get('timestamp')
            last_update = "unknown time"
            
            if last_timestamp:
                try:
                    if isinstance(last_timestamp, int):
                        if last_timestamp > 1000000000000:  # Milliseconds format
                            ts = datetime.fromtimestamp(last_timestamp / 1000)
                        else:  # Seconds format
                            ts = datetime.fromtimestamp(last_timestamp)
                        last_update = ts.strftime("%Y-%m-%d %H:%M:%S")
                    elif isinstance(last_timestamp, str):
                        last_update = last_timestamp
                except Exception:
                    pass
            
            # Add conversation header
            formatted_parts.append(f"\n===== Conversation: {chat_id} =====")
            formatted_parts.append(f"Participants: {', '.join(participants)}")
            formatted_parts.append(f"Message Count: {message_count} ({user_msgs} user, {agent_msgs} agent)")
            formatted_parts.append(f"Last Updated: {last_update}")
            
            # Add conversation format options
            formatted_parts.append("\nCONVERSATION HISTORY:")
            formatted_parts.append(self._format_multi_user_chat(chat_id))
        
        return "\n".join(formatted_parts)
    
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
        formatted_conversation = self._format_multi_user_chat(chat_id)
        
        return {
            "chat_id": chat_id,
            "exists": True,
            "message_count": len(messages),
            "formatted_conversation": formatted_conversation,
            "participants": list(set(msg.get('user_id') for msg in messages 
                               if msg.get('user_id') != 'agent'))
        }
    
    def _format_multi_user_chat(self, chat_id: str, include_timestamps: bool = True) -> str:
        """
        Format chat history for a multi-user conversation in a readable format.
        This includes usernames, timestamps, and formatting for reactions, edits, and threading.
        
        Args:
            chat_id: ID of the chat to format
            include_timestamps: Whether to include timestamps in the formatted output
            
        Returns:
            Formatted chat history as a string
        """
        if chat_id not in self._message_history or not self._message_history[chat_id]:
            return "No messages in this chat."
        
        formatted_chat = []
        current_date = None
        
        # Process messages in chronological order
        for msg in sorted(self._message_history[chat_id], key=lambda x: x.get('timestamp', 0)):
            # Extract message data
            user_id = msg.get('user_id', 'unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp')
            edited = msg.get('edited', False)
            reactions = msg.get('reactions', {})
            thread_id = msg.get('thread_id')
            display_name = msg.get('display_name')
            attachments = msg.get('attachments', [])
            deleted = msg.get('deleted', False)
            deletion_timestamp = msg.get('deletion_timestamp')
            
            # Parse and format timestamp
            time_str = ""
            date_str = ""
            if timestamp:
                try:
                    # Handle different timestamp formats
                    if isinstance(timestamp, int):
                        if timestamp > 1000000000000:  # Milliseconds format
                            ts = datetime.fromtimestamp(timestamp / 1000)
                        else:  # Seconds format
                            ts = datetime.fromtimestamp(timestamp)
                    elif isinstance(timestamp, str):
                        # Try parsing various string formats
                        try:
                            ts = datetime.fromisoformat(timestamp)
                        except ValueError:
                            try:
                                ts = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                            except ValueError:
                                # Fall back to current time if we can't parse
                                logger.warning(f"Could not parse timestamp: {timestamp}")
                                ts = datetime.now()
                    else:
                        ts = datetime.now()
                    
                    # Format time and date
                    time_str = ts.strftime("%H:%M:%S")
                    date_str = ts.strftime("%Y-%m-%d")
                    
                    # Add date divider if the date changed
                    if include_timestamps and (current_date != date_str):
                        formatted_chat.append(f"\n--- {date_str} ---\n")
                        current_date = date_str
                except Exception as e:
                    logger.error(f"Error formatting timestamp {timestamp}: {str(e)}")
                    time_str = "??:??:??"
            
            # Determine user display prefix
            if msg.get('role') == 'assistant':
                user_prefix = "🤖 Agent"
            else:
                # Use display name if available, fall back to user ID
                user_prefix = f"👤 {display_name}" if display_name else f"👤 User {user_id}"
            
            # Format message with user, timestamp, and content
            if include_timestamps:
                message_header = f"{user_prefix} [{time_str}]:"
            else:
                message_header = f"{user_prefix}:"
            
            # Handle deleted messages with special formatting
            if deleted:
                formatted_lines = [f"{message_header} [Message deleted]"]
                
                # Add deletion timestamp if available
                if deletion_timestamp:
                    try:
                        if isinstance(deletion_timestamp, int):
                            if deletion_timestamp > 1000000000000:  # Milliseconds format
                                del_ts = datetime.fromtimestamp(deletion_timestamp / 1000)
                            else:  # Seconds format
                                del_ts = datetime.fromtimestamp(deletion_timestamp)
                            del_time_str = del_ts.strftime("%Y-%m-%d %H:%M:%S")
                            formatted_lines.append(f"{' ' * len(message_header)} [Deleted on: {del_time_str}]")
                    except Exception:
                        pass
                
                # Include original content if available in deletion history
                if 'deletion_history' in msg and msg['deletion_history']:
                    original_content = msg['deletion_history'][-1]
                    if original_content:
                        content_preview = original_content[:50] + "..." if len(original_content) > 50 else original_content
                        formatted_lines.append(f"{' ' * len(message_header)} [Original content: \"{content_preview}\"]")
            else:
                # Handle multi-line messages with proper indentation
                content_lines = content.strip().split('\n')
                formatted_lines = [f"{message_header} {content_lines[0]}"]
                
                # Add indentation for subsequent lines
                if len(content_lines) > 1:
                    header_space = " " * len(message_header)
                    for line in content_lines[1:]:
                        formatted_lines.append(f"{header_space} {line}")
            
            # Add edited indicator
            if edited:
                formatted_lines[-1] += " (edited)"
            
            # Add thread indicator
            if thread_id:
                formatted_lines.append(f"{' ' * len(message_header)} [Thread: {thread_id}]")
            
            # Add attachment indicators
            if attachments:
                for i, attachment in enumerate(attachments):
                    attachment_type = attachment.get('type', 'file')
                    attachment_name = attachment.get('name', f'Attachment {i+1}')
                    formatted_lines.append(f"{' ' * len(message_header)} 📎 {attachment_type}: {attachment_name}")
            
            # Format reactions if present
            if reactions:
                reaction_strs = []
                for emoji, users in reactions.items():
                    if users:
                        if len(users) == 1:
                            # Show user for single reaction
                            user_name = next((msg.get('display_name', user) for msg in self._message_history[chat_id] 
                                           if msg.get('user_id') == users[0]), users[0])
                            reaction_strs.append(f"{emoji} {user_name}")
                        else:
                            # Show count for multiple reactions
                            reaction_strs.append(f"{emoji} {len(users)}")
                
                if reaction_strs:
                    reactions_line = " | ".join(reaction_strs)
                    formatted_lines.append(f"{' ' * len(message_header)} [{reactions_line}]")
            
            # Join all formatted parts and add to chat
            formatted_chat.append("\n".join(formatted_lines))
        
        # Join all messages with newlines
        return "\n\n".join(formatted_chat)

    def format_update_for_agent(self, update_data: Dict[str, Any]) -> str:
        """
        Format a messaging environment update in a way that's optimized for the agent.
        
        Specializes the base Environment.format_update_for_agent method to provide
        messaging-specific formatting, particularly for new messages.
        
        Requires standardized format:
        {
          "event": "messageReceived",
          "data": {
            "messageId": "xyz123",
            "conversationId": "C123",
            "sender": { "userId": "U123", "displayName": "Username" },
            "text": "Message content",
            ...
          }
        }
        
        Args:
            update_data: Data about the update from notify_observers
            
        Returns:
            Formatted string describing the update that can be sent directly to the agent
        """
        # Validate the standardized format
        if 'event' not in update_data or 'data' not in update_data:
            logger.error(f"Invalid update format: missing 'event' or 'data' fields in {update_data}")
            return f"<error>Invalid update format: {update_data}</error>"
            
        event_type = update_data['event']
        data = update_data['data']
        
        # For new messages, use a chat-optimized format
        if event_type == 'messageReceived':
            chat_id = data.get('conversationId')
            if not chat_id or chat_id not in self._message_history:
                # Fallback to base formatting if we can't find the chat
                return super().format_update_for_agent(update_data)
            
            # Get the newest message (should be the one that triggered this update)
            messages = self._message_history[chat_id]
            if not messages:
                return super().format_update_for_agent(update_data)
            
            # Find the message that matches the messageId
            message_id = data.get('messageId')
            latest_message = None
            
            if message_id:
                # Try to find the specific message
                for msg in messages:
                    if msg.get('message_id') == message_id:
                        latest_message = msg
                        break
            
            # If we couldn't find the specific message, use the latest one
            if not latest_message:
                latest_message = messages[-1]
            
            # Create a specialized format for new messages that looks like a chat interface
            formatted_update = f'<chat id="{chat_id}" event="{event_type}">\n'
            
            # Add user info section
            user_id = latest_message.get('user_id', 'unknown')
            display_name = latest_message.get('display_name')
            user_display = display_name if display_name else user_id
            formatted_update += f'  <user id="{user_id}" displayName="{user_display}">\n'
            
            # Add the message content, preserving formatting
            content = latest_message.get('content', '')
            formatted_update += f'    <message id="{latest_message.get("message_id", "unknown")}" timestamp="{latest_message.get("timestamp", "unknown")}">\n'
            
            # Format the content with proper indentation
            content_lines = content.split("\n")
            for line in content_lines:
                formatted_update += f'      {line}\n'
            
            formatted_update += '    </message>\n'
            
            # Add thread info if present
            thread_id = latest_message.get('thread_id')
            if thread_id:
                formatted_update += f'    <thread id="{thread_id}" />\n'
                
            # Add attachment info if present
            attachments = latest_message.get('attachments', [])
            if attachments:
                formatted_update += f'    <attachments count="{len(attachments)}">\n'
                for attachment in attachments[:3]:  # Limit to first 3 for brevity
                    attachment_type = attachment.get('type', 'file')
                    attachment_name = attachment.get('name', 'unnamed')
                    formatted_update += f'      <attachment type="{attachment_type}" name="{attachment_name}" />\n'
                if len(attachments) > 3:
                    formatted_update += f'      <more count="{len(attachments) - 3}" />\n'
                formatted_update += f'    </attachments>\n'
            
            formatted_update += '  </user>\n'
            
            # Add context info - how many previous messages in this chat
            previous_count = len(messages) - 1
            if previous_count > 0:
                formatted_update += f'  <context>\n'
                formatted_update += f'    <previous_messages_count>{previous_count}</previous_messages_count>\n'
                formatted_update += f'  </context>\n'
            
            formatted_update += '</chat>'
            return formatted_update
        
        # For message updates
        elif event_type == 'messageUpdated':
            chat_id = data.get('conversationId')
            message_id = data.get('messageId')
            new_text = data.get('newText') or data.get('text')
            
            if chat_id and message_id and new_text:
                return f'<chat id="{chat_id}" event="{event_type}">\n' + \
                       f'  <message id="{message_id}" action="updated">\n' + \
                       f'    <new_content>\n' + \
                       f'      {new_text}\n' + \
                       f'    </new_content>\n' + \
                       f'  </message>\n' + \
                       f'</chat>'
        
        # For message deletions
        elif event_type == 'messageDeleted':
            chat_id = data.get('conversationId')
            message_id = data.get('messageId')
            
            if chat_id and message_id:
                return f'<chat id="{chat_id}" event="{event_type}">\n' + \
                       f'  <message id="{message_id}" action="deleted" />\n' + \
                       f'</chat>'
        
        # For reactions
        elif event_type in ['reactionAdded', 'reactionRemoved']:
            chat_id = data.get('conversationId')
            message_id = data.get('messageId')
            emoji = data.get('emoji')
            
            if chat_id and message_id and emoji:
                action = "added" if event_type == 'reactionAdded' else "removed"
                
                # Get user info
                sender = data.get('sender', {})
                user_id = sender.get('userId', 'unknown')
                display_name = sender.get('displayName', user_id)
                
                return f'<chat id="{chat_id}" event="{event_type}">\n' + \
                       f'  <message id="{message_id}">\n' + \
                       f'    <reaction emoji="{emoji}" action="{action}" userId="{user_id}" displayName="{display_name}" />\n' + \
                       f'  </message>\n' + \
                       f'</chat>'
        
        # For other update types, use the base implementation
        return super().format_update_for_agent(update_data)

    def is_direct_message_chat(self, chat_id: str) -> bool:
        """
        Determine if a chat is a direct message (1:1) conversation with the agent.
        
        Args:
            chat_id: The ID of the chat to check
            
        Returns:
            True if the chat is a direct message, False otherwise
        """
        # Check if we have this chat in our history
        if chat_id not in self._message_history:
            return False
            
        # Get unique user IDs in this chat (excluding the agent)
        unique_users = set()
        for msg in self._message_history[chat_id]:
            user_id = msg.get('user_id')
            if user_id and user_id != 'agent' and user_id != 'system':
                unique_users.add(user_id)
                
        # If there's only one unique user (besides the agent), it's a DM
        return len(unique_users) == 1
        
    def update_state(self, update_data: Dict[str, Any]) -> bool:
        """
        Update the messaging environment state based on activity data.
        
        Requires standardized format:
        
        {
          "event": "messageReceived",
          "data": {
            "messageId": "xyz123",
            "conversationId": "C123",
            "sender": { "userId": "U123", "displayName": "Username" },
            "text": "Message content",
            "threadId": "T123",
            "attachments": [...],
            "timestamp": 1675703895000
          }
        }
        
        Args:
            update_data: Data to update the environment state with
            
        Returns:
            True if the update was successful, False otherwise
        """
        # Validate the standardized format
        if 'event' not in update_data or 'data' not in update_data:
            logger.error(f"Invalid update format: missing 'event' or 'data' fields in {update_data}")
            return False
            
        event_type = update_data['event']
        
        # If the event type is not recognized, use base implementation
        if event_type not in self.EVENT_TYPES:
            return super().update_state(update_data)
        
        # Extract data from nested structure
        data = update_data['data']
        
        # For messageReceived events, check if it's in a direct message chat
        # If so, add requiresAttention flag to ensure agent sees it
        if event_type == 'messageReceived':
            chat_id = data.get('conversationId')
            if chat_id and self.is_direct_message_chat(chat_id):
                # This is a direct message to the agent, it needs attention
                data['requiresAttention'] = True
                logger.debug(f"Message in direct chat {chat_id} marked as requiring attention")
                
        try:
            # Handle message received
            if event_type == 'messageReceived':
                # Extract message data
                message_id = data.get('messageId')
                chat_id = data.get('conversationId')
                text = data.get('text')
                timestamp = data.get('timestamp')
                
                # Handle sender information
                sender = data.get('sender', {})
                if isinstance(sender, dict):
                    user_id = sender.get('userId')
                    display_name = sender.get('displayName')
                else:
                    logger.error("Invalid 'sender' field format in messageReceived event")
                    return False
                
                # Get adapter info
                adapter_id = data.get('adapter_id')
                
                # Get threading info
                thread_id = data.get('threadId')
                
                # Get attachments
                attachments = data.get('attachments', [])
                
                if not all([chat_id, user_id, text]):
                    logger.warning(f"Incomplete message data received: {data}")
                    return False
                
                # Check IGNORE_MARKERS - skip recording if message starts with an ignored marker
                if any(text.startswith(marker) for marker in self.IGNORE_MARKERS):
                    logger.info(f"Ignoring message with marker: {text[:10]}... - no metadata stored")
                    # Return True to indicate we handled it but don't store anything
                    return True
                
                # Record in message history with all new metadata
                self.record_chat_message(
                    chat_id=chat_id,
                    user_id=user_id,
                    content=text,
                    adapter_id=adapter_id,
                    role="user",
                    message_id=message_id,
                    timestamp=timestamp,
                    thread_id=thread_id,
                    display_name=display_name,
                    attachments=attachments
                )
            
            # Handle message sent (our agent's messages)
            elif event_type == 'messageSent':
                # Extract message data
                message_id = data.get('messageId')
                chat_id = data.get('conversationId')
                text = data.get('text')
                timestamp = data.get('timestamp')
                adapter_id = data.get('adapter_id')
                thread_id = data.get('threadId')
                attachments = data.get('attachments', [])
                
                if not all([chat_id, text]):
                    logger.warning(f"Incomplete message data for sent message: {data}")
                    return False
                
                # Record in message history
                self.record_chat_message(
                    chat_id=chat_id,
                    user_id="agent",
                    content=text,
                    adapter_id=adapter_id,
                    role="assistant",
                    message_id=message_id,
                    timestamp=timestamp,
                    thread_id=thread_id,
                    attachments=attachments
                )
            
            # Handle message update/edit
            elif event_type == 'messageUpdated':
                # Extract message data
                message_id = data.get('messageId')
                chat_id = data.get('conversationId')
                new_text = data.get('newText') or data.get('text')
                timestamp = data.get('timestamp')
                
                if not all([chat_id, message_id, new_text]):
                    logger.warning(f"Incomplete data for message editing: {data}")
                    return False
                
                # Find and update message content
                if chat_id in self._message_history:
                    for i, msg in enumerate(self._message_history[chat_id]):
                        if msg.get('message_id') == message_id:
                            # Store original content in edit history
                            if 'edit_history' not in msg:
                                msg['edit_history'] = []
                                
                            msg['edit_history'].append(msg['content'])
                            msg['content'] = new_text
                            msg['edited'] = True
                            if timestamp:
                                msg['edit_timestamp'] = timestamp
                            
                            # Update in-place
                            self._message_history[chat_id][i] = msg
                            logger.info(f"Updated message {message_id} content in chat {chat_id}")
                            break
            
            # Handle message deletion
            elif event_type == 'messageDeleted':
                # Extract message data
                message_id = data.get('messageId')
                chat_id = data.get('conversationId')
                
                if not all([chat_id, message_id]):
                    logger.warning(f"Incomplete data for message deletion: {data}")
                    return False
                
                # Find and mark message as deleted
                if chat_id in self._message_history:
                    for i, msg in enumerate(self._message_history[chat_id]):
                        if msg.get('message_id') == message_id:
                            # Store original content in deletion history
                            if 'deletion_history' not in msg:
                                msg['deletion_history'] = []
                                
                            msg['deletion_history'].append(msg['content'])
                            
                            # Mark as deleted rather than removing to preserve context
                            self._message_history[chat_id][i]['deleted'] = True
                            self._message_history[chat_id][i]['content'] = "[Message deleted]"
                            self._message_history[chat_id][i]['deletion_timestamp'] = int(datetime.now().timestamp() * 1000)
                            logger.info(f"Marked message {message_id} as deleted in chat {chat_id}")
                            break
            
            # Handle reaction added
            elif event_type == 'reactionAdded':
                # Extract reaction data
                message_id = data.get('messageId')
                chat_id = data.get('conversationId')
                emoji = data.get('emoji')
                
                # Get user info
                sender = data.get('sender', {})
                if isinstance(sender, dict):
                    user_id = sender.get('userId')
                else:
                    logger.error("Invalid 'sender' field format in reactionAdded event")
                    return False
                
                if not all([chat_id, message_id, emoji]):
                    logger.warning(f"Incomplete data for reaction: {data}")
                    return False
                
                # Find message and add reaction
                if chat_id in self._message_history:
                    for i, msg in enumerate(self._message_history[chat_id]):
                        if msg.get('message_id') == message_id:
                            # Initialize reactions dict if needed
                            if 'reactions' not in msg:
                                msg['reactions'] = {}
                                
                            # Add user to this reaction
                            if emoji not in msg['reactions']:
                                msg['reactions'][emoji] = []
                                
                            if user_id and user_id not in msg['reactions'][emoji]:
                                msg['reactions'][emoji].append(user_id)
                                
                            # Update in-place
                            self._message_history[chat_id][i] = msg
                            logger.info(f"Added reaction {emoji} to message {message_id} in chat {chat_id}")
                            break
            
            # Handle reaction removed
            elif event_type == 'reactionRemoved':
                # Extract reaction data
                message_id = data.get('messageId')
                chat_id = data.get('conversationId')
                emoji = data.get('emoji')
                
                # Get user info
                sender = data.get('sender', {})
                if isinstance(sender, dict):
                    user_id = sender.get('userId')
                else:
                    logger.error("Invalid 'sender' field format in reactionRemoved event")
                    return False
                
                if not all([chat_id, message_id, emoji]):
                    logger.warning(f"Incomplete data for reaction removal: {data}")
                    return False
                
                # Find message and remove reaction
                if chat_id in self._message_history:
                    for i, msg in enumerate(self._message_history[chat_id]):
                        if msg.get('message_id') == message_id:
                            if 'reactions' in msg and emoji in msg['reactions']:
                                # Remove user from reaction list
                                if user_id and user_id in msg['reactions'][emoji]:
                                    msg['reactions'][emoji].remove(user_id)
                                    
                                    # Clean up empty reaction lists
                                    if not msg['reactions'][emoji]:
                                        del msg['reactions'][emoji]
                                        
                                    # Clean up empty reactions dict
                                    if not msg['reactions']:
                                        del msg['reactions']
                                        
                                    # Update in-place
                                    self._message_history[chat_id][i] = msg
                                    logger.info(f"Removed reaction {emoji} from message {message_id} in chat {chat_id}")
                                    break
            
            # Handle fetch history
            elif event_type == 'fetchHistory':
                # Extract history data
                chat_id = data.get('conversationId')
                messages = data.get('messages', [])
                
                if not chat_id:
                    logger.warning(f"Missing chat_id in fetchHistory: {data}")
                    return False
                
                # Process batch of messages
                if messages:
                    logger.info(f"Processing {len(messages)} historical messages for chat {chat_id}")
                    
                    # Initialize chat history if needed
                    if chat_id not in self._message_history:
                        self._message_history[chat_id] = []
                    
                    # Track existing message IDs to avoid duplicates
                    existing_msg_ids = {msg.get('message_id') for msg in self._message_history[chat_id] 
                                      if msg.get('message_id')}
                    
                    # Process each message in history batch
                    for msg in messages:
                        # Extract message data
                        message_id = msg.get('messageId')
                        
                        # Skip if we already have this message
                        if message_id and message_id in existing_msg_ids:
                            continue
                            
                        # Get sender info
                        sender = msg.get('sender', {})
                        if isinstance(sender, dict):
                            user_id = sender.get('userId')
                            display_name = sender.get('displayName')
                        else:
                            user_id = msg.get('user_id', 'unknown')
                            display_name = None
                        
                        # Extract content
                        text = msg.get('text', '')
                        
                        # Get other metadata
                        adapter_id = msg.get('adapter_id')
                        role = msg.get('role', 'user')
                        timestamp = msg.get('timestamp')
                        thread_id = msg.get('threadId')
                        attachments = msg.get('attachments', [])
                        
                        # Record message with all metadata
                        self.record_chat_message(
                            chat_id=chat_id,
                            user_id=user_id,
                            content=text,
                            adapter_id=adapter_id,
                            role=role,
                            message_id=message_id,
                            timestamp=timestamp,
                            thread_id=thread_id,
                            display_name=display_name,
                            attachments=attachments,
                            is_history=True
                        )
                        
                        # Add to tracking set
                        if message_id:
                            existing_msg_ids.add(message_id)
            
            # Call base implementation to handle metadata and observers
            # Create a compatible format for the base implementation
            converted_update = {
                "type": event_type,
                **{k: v for k, v in update_data.items() if k != 'event' and k != 'data'},
                **data
            }
            super().update_state(converted_update)
            return True
            
        except Exception as e:
            logger.error(f"Error updating messaging environment state: {str(e)}")
            logger.exception("Full exception details:")
            return False 