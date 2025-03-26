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
                 platform: str = "unknown", adapter_id: str = "default"):
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
        
        # Flag to indicate if this is a remote chat element (used for delegate selection)
        self._is_remote = False
        
        # Register messaging tools
        self._register_messaging_tools()
        logger.info(f"Created chat element for {platform} (adapter: {adapter_id}): {name} ({element_id})")
    
    def set_as_remote(self, is_remote: bool = True) -> None:
        """
        Set whether this is a remote chat element.
        
        Args:
            is_remote: Whether this is a remote chat element
        """
        self._is_remote = is_remote
        # Update the delegate when remote status changes
        self.update_delegate()
        
    def is_remote(self) -> bool:
        """
        Check if this is a remote chat element.
        
        Returns:
            True if this is a remote chat element, False otherwise
        """
        return self._is_remote
    
    def update_delegate(self) -> None:
        """
        Update the delegate for this chat element based on remote status.
        """
        # Import here to avoid circular imports
        try:
            from rendering.delegates import create_chat_delegate
            
            # Create the appropriate delegate function
            delegate_func = create_chat_delegate(self._is_remote)
            
            # Register the delegate with the element
            self._delegate = delegate_func
            
            logger.debug(f"Updated delegate for chat element {self.id} (remote: {self._is_remote})")
        except ImportError:
            logger.warning(f"Could not create chat delegate for element {self.id}")
    
    def get_delegate(self):
        """
        Get the element delegate for rendering.
        
        If the delegate hasn't been set yet, update it first.
        
        Returns:
            Delegate function or object
        """
        if not self._delegate:
            self.update_delegate()
        return self._delegate
    
    def get_connection_spans(self, options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Get connection spans for remote chat rendering.
        
        This is used by the remote rendering system to retrieve connection span information
        when this chat element is accessed via an uplink.
        
        Args:
            options: Options for retrieving connection spans
                - limit: Maximum number of spans to retrieve
                - include_active: Whether to include the active span
                
        Returns:
            List of connection span information
        """
        options = options or {}
        limit = options.get('limit', 5)
        include_active = options.get('include_active', True)
        
        # This would typically be provided by the UplinkProxy, but we implement it here
        # to support the ElementDelegate interfaces
        spans = []
        
        # If this is a remote element and has a parent uplink, retrieve spans from it
        if self._is_remote and hasattr(self, '_parent_element') and self._parent_element:
            parent_id, _ = self._parent_element
            parent = self._registry.get_element(parent_id) if self._registry else None
            
            if parent and hasattr(parent, 'get_connection_spans'):
                # Delegate to parent uplink
                return parent.get_connection_spans(options)
        
        # Otherwise return empty list as this is a direct element
        return spans
    
    def publish_message(self, message_data: Dict[str, Any], timeline_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Publish a message to the Activity Layer.
        
        Args:
            message_data: Message data to publish
            timeline_context: Optional timeline context
            
        Returns:
            True if the message was published successfully, False otherwise
        """
        # Validate timeline context
        validated_timeline_context = self._validate_timeline_context(timeline_context)
        
        # Only publish if this is the primary timeline
        if not validated_timeline_context.get("is_primary", False):
            logger.info(f"Not publishing message to external systems (non-primary timeline): {validated_timeline_context.get('timeline_id')}")
            return False
        
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
        # Get the basic state
        state = super().get_state()
        
        # Add chat-specific state
        state.update({
            "platform": self.platform,
            "adapter_id": self.adapter_id,
            "timeline_state": self._timeline_state,
            "chat_ids": list(self._timeline_state.get("messages", {}).keys()),
            "is_remote": self._is_remote,
            "active_timeline": next(iter(self._timeline_state.get("active_timelines", set())), None)
        })
        
        return state
    
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
        Update the element state based on incoming event data.
        
        Args:
            update_data: Event data to update with
            timeline_context: Optional timeline context for the update
            
        Returns:
            True if update was successful, False otherwise
        """
        # Call parent implementation
        success = super().update_state(update_data, timeline_context)
        
        if not success:
            return False
        
        # Check for message events that might need agent attention
        event_type = update_data.get("event_type")
        if event_type in self.EVENT_TYPES and event_type != "needs_attention" and event_type != "attention_cleared":
            # Check if this event should trigger agent attention
            if self._needs_agent_attention(update_data):
                # Signal attention need through standard state update mechanism
                self.signal_attention_need({
                    "event_type": event_type,
                    "event_data": update_data,
                    "reason": "message_requires_attention",
                    "conversation_id": update_data.get("conversation_id"),
                    "adapter_id": self.adapter_id,
                    "platform": self.platform
                }, timeline_context)
                
                logger.info(f"Signaled attention need for {event_type} event in timeline {timeline_context.get('timeline_id', 'unknown')}")
        
        return True
    
    def _needs_agent_attention(self, event_data: Dict[str, Any]) -> bool:
        """
        Determine if an event needs agent attention.
        
        Args:
            event_data: Event data to check
            
        Returns:
            True if agent attention is needed, False otherwise
        """
        # Check event type
        event_type = event_data.get("event_type")
        
        # Some event types always need attention
        if event_type in self.ATTENTION_SIGNALS:
            return True
        
        # For message events, check content
        if event_type == "message_received":
            message_data = event_data.get("data", {})
            message_text = message_data.get("text", "")
            
            # Message starting with ignore markers should be ignored
            for marker in self.IGNORE_MARKERS:
                if message_text.startswith(marker):
                    return False
            
            # Check for mentions of the agent
            if "@agent" in message_text.lower() or "agent:" in message_text.lower():
                return True
                
            # Check if this is a direct message
            if message_data.get("is_direct", False):
                return True
        
        return False
    
    def handle_response(self, response_data: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handle agent response to a message.
        
        Args:
            response_data: Response data from the agent
            timeline_context: Timeline context
            
        Returns:
            True if response was handled successfully, False otherwise
        """
        logger.info(f"Handling agent response for chat element {self.id} in timeline {timeline_context.get('timeline_id', 'unknown')}")
        
        # Extract relevant information
        response_content = response_data.get("content", "")
        original_event = response_data.get("original_event", {})
        
        # Get conversation information from the original event
        conversation_id = original_event.get("conversation_id")
        if not conversation_id:
            # Try to get it from the event data
            event_data = original_event.get("data", {})
            conversation_id = event_data.get("conversation_id")
        
        # If still no conversation ID, use the first available one
        if not conversation_id and hasattr(self, "_timeline_state") and hasattr(self._timeline_state, "messages"):
            # Find the first available conversation
            for timeline_id, messages in self._timeline_state["messages"].items():
                if messages:
                    # Get conversation ID from the first message
                    first_message = messages[0]
                    conversation_id = first_message.get("conversation_id")
                    if conversation_id:
                        break
        
        if not conversation_id:
            logger.error("Cannot handle response: No conversation ID found")
            return False
        
        # Prepare message data
        message_data = {
            "event_type": "send_message",
            "adapter_id": self.adapter_id,
            "data": {
                "conversation_id": conversation_id,
                "text": response_content,
                "is_response": True,
                "in_response_to": original_event.get("id")
            }
        }
        
        # Thread ID if available
        thread_id = original_event.get("thread_id")
        if thread_id:
            message_data["data"]["thread_id"] = thread_id
        
        # Publish the message with timeline context
        success = self.publish_message(message_data, timeline_context)
        
        # Clear attention need
        self.clear_attention_need(timeline_context)
        
        return success 