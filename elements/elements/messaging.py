"""
Chat Element
Element for handling chat/messaging functionality with timeline-aware state management.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import uuid
import time

from .object import Object

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChatElement(Object):
    """
    Element for handling chat/messaging functionality.
    
    Provides tools for sending messages, typing indicators, and other
    messaging-related functionality with timeline-aware state management.
    
    Each instance of ChatElement handles messages for a specific platform/adapter,
    identified by platform and adapter_id parameters.
    """
    EVENT_TYPES = [
        "message_received", 
        "message_sent", 
        "message_updated", 
        "message_deleted", 
        "reaction_added", 
        "reaction_removed", 
        "clear_context"
    ]
    
    # Define which events should trigger agent attention
    ATTENTION_SIGNALS = [
        "message_sent",
    ]
    
    # Messages starting with these markers are ignored
    IGNORE_MARKERS = ["."]
    
    def __init__(self, element_id: str, name: str, description: str,
                 platform: str, adapter_id: str):
        """
        Initialize the chat element.
        
        Args:
            element_id: Unique identifier for this element
            name: Human-readable name for this element
            description: Description of this element's purpose
            platform: Platform identifier (e.g., 'discord', 'zulip', 'telegram')
            adapter_id: ID of the adapter this element is associated with
        """
        super().__init__(element_id, name, description)
        
        # Store platform and adapter information
        self.platform = platform
        self.adapter_id = adapter_id
        
        # Initialize timeline-aware state
        self._timeline_state = {
            "messages": {},  # timeline_id -> message history
            "active_timelines": set(),  # Set of active timeline IDs
            "timeline_metadata": {}  # timeline_id -> metadata
        }
        
        # Register messaging tools
        self._register_messaging_tools()
        logger.info(f"Created chat element for {platform} (adapter: {adapter_id}): {name} ({element_id})")
    
    def publish_message(self, message_data: Dict[str, Any]) -> bool:
        """
        Publish a message to be sent via the appropriate adapter.
        
        This method now routes messages through the SpaceRegistry.
        
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
            if 'event_type' not in message_data or 'data' not in message_data:
                logger.error("Invalid message format: requires 'event_type' and 'data' fields")
                return False
            
            # Ensure payload has required fields based on event type
            event_type = message_data['event_type']
            data = message_data['data']
            
            if event_type == 'send_message':
                if 'conversation_id' not in data or 'text' not in data:
                    logger.error("Invalid send_message data: requires 'conversation_id' and 'text' fields")
                    return False
            elif event_type == 'edit_message':
                if 'conversation_id' not in data or 'message_id' not in data or 'text' not in data:
                    logger.error("Invalid edit_message data: requires 'conversation_id', 'message_id', and 'text' fields")
                    return False
            elif event_type == 'delete_message':
                if 'conversation_id' not in data or 'message_id' not in data:
                    logger.error("Invalid delete_message data: requires 'conversation_id' and 'message_id' fields")
                    return False
            elif event_type == 'add_reaction' or event_type == 'remove_reaction':
                if 'conversation_id' not in data or 'message_id' not in data or 'emoji' not in data:
                    logger.error(f"Invalid {event_type} data: requires 'conversation_id', 'message_id', and 'emoji' fields")
                    return False
            elif event_type == 'typing_indicator':
                if 'conversation_id' not in data or 'is_typing' not in data:
                    logger.error("Invalid typing_indicator data: requires 'conversation_id' and 'is_typing' fields")
                    return False
            
            # Check for registry
            if not self._registry:
                logger.error(f"Cannot publish message for element {self.id}: No registry reference")
                return False
                
            # Send message through the registry
            return self._registry.send_external_message(message_data)
            
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False
    
    def publish_typing_indicator(self, message_data: Dict[str, Any]) -> bool:
        """
        Publish a typing indicator through the SpaceRegistry.
        
        Args:
            message_data: Typing indicator data to publish
            
        Returns:
            True if the indicator was published, False otherwise
        """
        try:
            # Extract adapter ID
            adapter_id = message_data.get('adapter_id')
            if not adapter_id:
                logger.error("Missing adapter_id in typing indicator")
                return False
                
            # Validate message format
            if 'event_type' not in message_data or 'data' not in message_data:
                logger.error("Invalid typing indicator format: requires 'event_type' and 'data' fields")
                return False
                
            # Verify this is a typing indicator event
            if message_data['event_type'] != 'typing_indicator':
                logger.error(f"Invalid event type for typing indicator: {message_data['event_type']}")
                return False
                
            # Ensure data has required fields
            data = message_data['data']
            if 'conversation_id' not in data or 'is_typing' not in data:
                logger.error("Invalid typing_indicator data: requires 'conversation_id' and 'is_typing' fields")
                return False
                
            # Check for registry
            if not self._registry:
                logger.error(f"Cannot publish typing indicator for element {self.id}: No registry reference")
                return False
                
            # Send through registry
            return self._registry.send_typing_indicator(
                adapter_id, 
                data['conversation_id'], 
                data['is_typing']
            )
                
        except Exception as e:
            logger.error(f"Error publishing typing indicator: {e}")
            return False
    
    def _register_messaging_tools(self):
        """
        Register messaging-specific tools for the element.
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
            Send a message to a chat
            
            Args:
                chat_id: Chat ID to send to
                content: Message content
                adapter_id: ID of the adapter to use
                thread_id: Optional thread ID
                mentions: Optional list of user mentions
                parse_mode: Optional parse mode
                disable_link_previews: Whether to disable link previews
                disable_mentions: Whether to disable mention processing
                
            Returns:
                Status information
            """
            # Log and notify observers
            logger.debug(f"Sending message to chat {chat_id} via adapter {adapter_id}: {content[:50]}...")
            
            # Construct message data
            message_data = {
                "event_type": "send_message",
                "data": {
                    "conversation_id": chat_id,
                    "text": content
                },
                "adapter_id": adapter_id
            }
            
            # Add optional parameters if provided
            if thread_id:
                message_data["data"]["thread_id"] = thread_id
            if mentions:
                message_data["data"]["mentions"] = mentions
            if parse_mode:
                message_data["data"]["parse_mode"] = parse_mode
            if disable_link_previews is not None:
                message_data["data"]["disable_link_previews"] = disable_link_previews
            if disable_mentions is not None:
                message_data["data"]["disable_mentions"] = disable_mentions
                
            # Publish the message
            published = self.publish_message(message_data)
            
            # Return status
            return {
                "success": published,
                "chat_id": chat_id,
                "adapter_id": adapter_id,
                "message": "Message sent successfully" if published else "Failed to send message"
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
            Edit an existing message
            
            Args:
                chat_id: Chat ID containing the message
                message_id: ID of the message to edit
                content: New content for the message
                adapter_id: ID of the adapter to use
                
            Returns:
                Status information
            """
            logger.debug(f"Editing message {message_id} in chat {chat_id} via adapter {adapter_id}")
            
            # Construct message data
            message_data = {
                "event_type": "edit_message",
                "data": {
                    "conversation_id": chat_id,
                    "message_id": message_id,
                    "text": content
                },
                "adapter_id": adapter_id
            }
            
            # Publish the message
            published = self.publish_message(message_data)
            
            # Return status
            return {
                "success": published,
                "chat_id": chat_id,
                "message_id": message_id,
                "adapter_id": adapter_id,
                "message": "Message edited successfully" if published else "Failed to edit message"
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
            Delete a message from the chat
            
            Args:
                chat_id: Chat ID containing the message
                message_id: ID of the message to delete
                adapter_id: ID of the adapter to use
                
            Returns:
                Status information
            """
            logger.debug(f"Deleting message {message_id} in chat {chat_id} via adapter {adapter_id}")
            
            # Construct message data
            message_data = {
                "event_type": "delete_message",
                "data": {
                    "conversation_id": chat_id,
                    "message_id": message_id
                },
                "adapter_id": adapter_id
            }
            
            # Publish the message
            published = self.publish_message(message_data)
            
            # Return status
            return {
                "success": published,
                "chat_id": chat_id,
                "message_id": message_id,
                "adapter_id": adapter_id,
                "message": "Message deleted successfully" if published else "Failed to delete message"
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
            Add a reaction to a message
            
            Args:
                chat_id: Chat ID containing the message
                message_id: ID of the message to react to
                emoji: Emoji reaction to add
                adapter_id: ID of the adapter to use
                
            Returns:
                Status information
            """
            logger.debug(f"Adding reaction {emoji} to message {message_id} in chat {chat_id}")
            
            # Construct message data
            message_data = {
                "event_type": "add_reaction",
                "data": {
                    "conversation_id": chat_id,
                    "message_id": message_id,
                    "emoji": emoji
                },
                "adapter_id": adapter_id
            }
            
            # Publish the message
            published = self.publish_message(message_data)
            
            # Return status
            return {
                "success": published,
                "chat_id": chat_id,
                "message_id": message_id,
                "emoji": emoji,
                "adapter_id": adapter_id,
                "message": "Reaction added successfully" if published else "Failed to add reaction"
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
            Remove a reaction from a message
            
            Args:
                chat_id: Chat ID containing the message
                message_id: ID of the message to remove reaction from
                emoji: Emoji reaction to remove
                adapter_id: ID of the adapter to use
                
            Returns:
                Status information
            """
            logger.debug(f"Removing reaction {emoji} from message {message_id} in chat {chat_id}")
            
            # Construct message data
            message_data = {
                "event_type": "remove_reaction",
                "data": {
                    "conversation_id": chat_id,
                    "message_id": message_id,
                    "emoji": emoji
                },
                "adapter_id": adapter_id
            }
            
            # Publish the message
            published = self.publish_message(message_data)
            
            # Return status
            return {
                "success": published,
                "chat_id": chat_id,
                "message_id": message_id,
                "emoji": emoji,
                "adapter_id": adapter_id,
                "message": "Reaction removed successfully" if published else "Failed to remove reaction"
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
            Send a typing indicator
            
            Args:
                chat_id: Chat ID to send the indicator to
                adapter_id: ID of the adapter to use
                is_typing: Whether typing is active or not
                
            Returns:
                Status information
            """
            # Construct message data
            message_data = {
                "event_type": "typing_indicator",
                "data": {
                    "conversation_id": chat_id,
                    "is_typing": is_typing
                },
                "adapter_id": adapter_id
            }
            
            # Publish the message
            published = self.publish_typing_indicator(message_data)
            
            # Return status
            return {
                "success": published,
                "chat_id": chat_id,
                "adapter_id": adapter_id,
                "is_typing": is_typing,
                "message": "Typing indicator sent successfully" if published else "Failed to send typing indicator"
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
        is_history: bool = False,
        timeline_id: Optional[str] = None
    ) -> None:
        """
        Record a chat message in the timeline-aware state.
        
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
            timeline_id: ID of the timeline to record the message in
        """
        # Use provided timeline_id or chat_id as fallback
        timeline_id = timeline_id or chat_id
        
        # Initialize timeline if needed
        if timeline_id not in self._timeline_state["messages"]:
            self._timeline_state["messages"][timeline_id] = []
            
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
            "timeline_id": timeline_id
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
            
        # Add message to timeline
        self._timeline_state["messages"][timeline_id].append(message)
        
        # Log the action
        if is_history:
            logger.debug(f"Loaded historical message {message_id} in timeline {timeline_id}")
        else:
            logger.info(f"Recorded new message {message_id} from {user_id} in timeline {timeline_id}")
        
        # Keep timeline sorted by timestamp
        self._timeline_state["messages"][timeline_id].sort(key=lambda x: x.get('timestamp', 0))
        
        # Only notify observers for new user messages, not for historical or agent messages
        if role == "user" and not is_history:
            # Create update data for the new message
            update_data = {
                'type': 'message_received',
                'messageId': message_id,
                'conversationId': chat_id,
                'timelineId': timeline_id,
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
            logger.debug(f"Notifying observers about new message in timeline {timeline_id} from user {user_id}")
            self.notify_observers(update_data)
        
        # Return the message ID for reference
        return message_id
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the element.
        
        Returns:
            Dictionary representation of the current state
        """
        return {
            "timeline_state": self._timeline_state,
            "chat_ids": list(self._timeline_state["messages"].keys())
        }
    
    def is_direct_message_chat(self, chat_id: str) -> bool:
        """
        Determine if a chat is a direct message (1:1) conversation with the agent.
        
        Args:
            chat_id: The ID of the chat to check
            
        Returns:
            True if the chat is a direct message, False otherwise
        """
        # Check if we have this chat in our history
        if chat_id not in self._timeline_state["messages"]:
            return False
            
        # Get unique user IDs in this chat (excluding the agent)
        unique_users = set()
        for msg in self._timeline_state["messages"][chat_id]:
            user_id = msg.get('user_id')
            if user_id and user_id != 'agent' and user_id != 'system':
                unique_users.add(user_id)
                
        # If there's only one unique user (besides the agent), it's a DM
        return len(unique_users) == 1
        
    def update_state(self, update_data: Dict[str, Any], timeline_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the element state based on incoming event data with timeline context.
        
        Args:
            update_data: Event data with event_type and other fields
            timeline_context: Optional timeline context for state updates
            
        Returns:
            True if the state was successfully updated, False otherwise
        """
        try:
            # Extract the event type
            event_type = update_data.get('event_type')
            if not event_type:
                logger.error("Missing event_type in update data")
                return False
                
            # Get or create timeline context
            timeline_id = timeline_context.get('timeline_id') if timeline_context else None
            if not timeline_id:
                timeline_id = update_data.get('conversation_id')
                
            if timeline_id:
                if timeline_id not in self._timeline_state["active_timelines"]:
                    self._timeline_state["active_timelines"].add(timeline_id)
                    self._timeline_state["timeline_metadata"][timeline_id] = {
                        "created_at": int(time.time() * 1000),
                        "last_updated": int(time.time() * 1000)
                    }
                
                # Update timeline metadata
                self._timeline_state["timeline_metadata"][timeline_id]["last_updated"] = int(time.time() * 1000)
                
                # Handle different event types
                if event_type == 'message_received':
                    return self._handle_message_received(update_data, timeline_id)
                    
                elif event_type == 'message_updated':
                    return self._handle_message_updated(update_data, timeline_id)
                    
                elif event_type == 'message_deleted':
                    return self._handle_message_deleted(update_data, timeline_id)
                    
                elif event_type == 'reaction_added':
                    return self._handle_reaction_added(update_data, timeline_id)
                    
                elif event_type == 'reaction_removed':
                    return self._handle_reaction_removed(update_data, timeline_id)
                    
                elif event_type == 'clear_context':
                    return self._handle_clear_context(update_data, timeline_id)
                    
            else:
                logger.error("Missing timeline_id in update data")
                return False
                
        except Exception as e:
            logger.error(f"Error updating state: {e}")
            return False
            
    def _handle_message_received(self, event_data: Dict[str, Any], timeline_id: str) -> bool:
        """
        Handle a message_received event.
        
        Args:
            event_data: Message data
            timeline_id: Timeline ID
            
        Returns:
            True if handled successfully
        """
        try:
            # Extract required fields
            adapter_id = event_data.get('adapter_id')
            conversation_id = event_data.get('conversation_id')
            message_id = event_data.get('message_id')
            sender = event_data.get('sender', {})
            user_id = sender.get('user_id') if sender else None
            display_name = sender.get('display_name') if sender else None
            text = event_data.get('text', '')
            thread_id = event_data.get('thread_id')
            attachments = event_data.get('attachments', [])
            timestamp = event_data.get('timestamp')
            
            if not all([conversation_id, user_id, text]):
                logger.error(f"Missing required fields in message_received event: {event_data}")
                return False
                
            # Record the message in the element
            self.record_chat_message(
                chat_id=conversation_id, 
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
            
            # Process the message through message processing pipeline
            # This will trigger any observers or agent attention if needed
            # We can use the environment_manager if it's set
            if hasattr(self, 'environment_manager') and self.environment_manager:
                self.environment_manager.process_message(
                    user_id=user_id,
                    message_text=text,
                    message_id=message_id,
                    platform=self.platform,
                    env_id=self.id
                )
                
            return True
            
        except Exception as e:
            logger.error(f"Error handling message_received event: {e}")
            return False
            
    def _handle_message_updated(self, event_data: Dict[str, Any], timeline_id: str) -> bool:
        """
        Handle a message_updated event.
        
        Args:
            event_data: Message update data
            timeline_id: Timeline ID
            
        Returns:
            True if handled successfully
        """
        try:
            # Extract required fields
            adapter_id = event_data.get('adapter_id')
            conversation_id = event_data.get('conversation_id')
            message_id = event_data.get('message_id')
            new_text = event_data.get('new_text', '')
            
            if not all([conversation_id, message_id, new_text]):
                logger.error(f"Missing required fields in message_updated event: {event_data}")
                return False
                
            # Update in message history
            if conversation_id in self._timeline_state["messages"]:
                for i, msg in enumerate(self._timeline_state["messages"][conversation_id]):
                    if msg.get('message_id') == message_id:
                        # Store original content in edit history
                        if 'edit_history' not in msg:
                            msg['edit_history'] = []
                            
                        msg['edit_history'].append(msg['content'])
                        msg['content'] = new_text
                        msg['edited'] = True
                        msg['edit_timestamp'] = int(time.time() * 1000)
                        
                        # Update in-place
                        self._timeline_state["messages"][conversation_id][i] = msg
                        logger.debug(f"Updated message {message_id} in chat {conversation_id}")
                        return True
                        
            logger.warning(f"Message {message_id} not found in chat {conversation_id} history")
            return False
            
        except Exception as e:
            logger.error(f"Error handling message_updated event: {e}")
            return False
            
    def _handle_message_deleted(self, event_data: Dict[str, Any], timeline_id: str) -> bool:
        """
        Handle a message_deleted event.
        
        Args:
            event_data: Message deletion data
            timeline_id: Timeline ID
            
        Returns:
            True if handled successfully
        """
        try:
            # Extract required fields
            adapter_id = event_data.get('adapter_id')
            conversation_id = event_data.get('conversation_id')
            message_id = event_data.get('message_id')
            
            if not all([conversation_id, message_id]):
                logger.error(f"Missing required fields in message_deleted event: {event_data}")
                return False
                
            # Mark as deleted in message history
            if conversation_id in self._timeline_state["messages"]:
                for i, msg in enumerate(self._timeline_state["messages"][conversation_id]):
                    if msg.get('message_id') == message_id:
                        # Store original content in deletion history
                        if 'deletion_history' not in msg:
                            msg['deletion_history'] = []
                            
                        msg['deletion_history'].append(msg['content'])
                        msg['content'] = "[Message deleted]"
                        msg['deleted'] = True
                        msg['deletion_timestamp'] = int(time.time() * 1000)
                        
                        # Update in-place
                        self._timeline_state["messages"][conversation_id][i] = msg
                        logger.debug(f"Marked message {message_id} as deleted in chat {conversation_id}")
                        return True
                        
            logger.warning(f"Message {message_id} not found in chat {conversation_id} history")
            return False
            
        except Exception as e:
            logger.error(f"Error handling message_deleted event: {e}")
            return False
            
    def _handle_reaction_added(self, event_data: Dict[str, Any], timeline_id: str) -> bool:
        """
        Handle a reaction_added event.
        
        Args:
            event_data: Reaction data
            timeline_id: Timeline ID
            
        Returns:
            True if handled successfully
        """
        try:
            # Extract required fields
            adapter_id = event_data.get('adapter_id')
            conversation_id = event_data.get('conversation_id')
            message_id = event_data.get('message_id')
            emoji = event_data.get('emoji')
            
            if not all([conversation_id, message_id, emoji]):
                logger.error(f"Missing required fields in reaction_added event: {event_data}")
                return False
                
            # Update reactions in message history
            if conversation_id in self._timeline_state["messages"]:
                for i, msg in enumerate(self._timeline_state["messages"][conversation_id]):
                    if msg.get('message_id') == message_id:
                        # Initialize reactions dict if needed
                        if 'reactions' not in msg:
                            msg['reactions'] = {}
                            
                        # Add reaction
                        if emoji not in msg['reactions']:
                            msg['reactions'][emoji] = []
                            
                        # Add user to this reaction (using 'user' as placeholder since we don't have user ID)
                        if "user" not in msg['reactions'][emoji]:
                            msg['reactions'][emoji].append("user")
                            
                        # Update in-place
                        self._timeline_state["messages"][conversation_id][i] = msg
                        logger.debug(f"Added reaction {emoji} to message {message_id} in chat {conversation_id}")
                        return True
                        
            logger.warning(f"Message {message_id} not found in chat {conversation_id} history")
            return False
            
        except Exception as e:
            logger.error(f"Error handling reaction_added event: {e}")
            return False
            
    def _handle_reaction_removed(self, event_data: Dict[str, Any], timeline_id: str) -> bool:
        """
        Handle a reaction_removed event.
        
        Args:
            event_data: Reaction data
            timeline_id: Timeline ID
            
        Returns:
            True if handled successfully
        """
        try:
            # Extract required fields
            adapter_id = event_data.get('adapter_id')
            conversation_id = event_data.get('conversation_id')
            message_id = event_data.get('message_id')
            emoji = event_data.get('emoji')
            
            if not all([conversation_id, message_id, emoji]):
                logger.error(f"Missing required fields in reaction_removed event: {event_data}")
                return False
                
            # Update reactions in message history
            if conversation_id in self._timeline_state["messages"]:
                for i, msg in enumerate(self._timeline_state["messages"][conversation_id]):
                    if msg.get('message_id') == message_id:
                        if 'reactions' in msg and emoji in msg['reactions']:
                            # Remove user from reaction list
                            if "user" in msg['reactions'][emoji]:
                                msg['reactions'][emoji].remove("user")
                                
                            # Clean up empty reaction lists
                            if not msg['reactions'][emoji]:
                                del msg['reactions'][emoji]
                                
                            # Clean up empty reactions dict
                            if not msg['reactions']:
                                del msg['reactions']
                                
                            # Update in-place
                            self._timeline_state["messages"][conversation_id][i] = msg
                            logger.debug(f"Removed reaction {emoji} from message {message_id} in chat {conversation_id}")
                            return True
                        else:
                            logger.warning(f"Reaction {emoji} not found on message {message_id} in chat {conversation_id}")
                            return False
                            
            logger.warning(f"Message {message_id} not found in chat {conversation_id} history")
            return False
            
        except Exception as e:
            logger.error(f"Error handling reaction_removed event: {e}")
            return False
            
    def _handle_clear_context(self, event_data: Dict[str, Any], timeline_id: str) -> bool:
        """
        Handle a clear_context event.
        
        Args:
            event_data: Clear context data
            timeline_id: Timeline ID
            
        Returns:
            True if handled successfully
        """
        try:
            # Extract required fields
            adapter_id = event_data.get('adapter_id')
            conversation_id = event_data.get('conversation_id')
            
            if not all([adapter_id, conversation_id]):
                logger.error(f"Missing required fields in clear_context event: {event_data}")
                return False
                
            # Clear context for this chat
            if conversation_id in self._timeline_state["messages"]:
                # Optionally archive history instead of deleting it
                self._timeline_state["messages"][conversation_id] = []
                logger.info(f"Cleared message history for chat {conversation_id}")
                return True
            else:
                logger.warning(f"No history found for chat {conversation_id}")
                return True  # Return true anyway since there's nothing to clear
                
        except Exception as e:
            logger.error(f"Error handling clear_context event: {e}")
            return False 