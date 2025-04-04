"""
Chat Element
Element for handling chat/messaging functionality using a component-based approach.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
import uuid
import time

from .base import BaseElement # Inherit directly from BaseElement
from .components import ToolProvider, VeilProducer # Base components
from .components.messaging import HistoryComponent, PublisherComponent # Messaging specific
from .components.space import TimelineComponent # Needed for dependencies

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChatElement(BaseElement):
    """
    Element for handling chat/messaging functionality.
    
    Relies on components for:
    - History management (`HistoryComponent`)
    - Publishing messages/indicators (`PublisherComponent`)
    - Providing tools (`ToolProvider`)
    - Generating representation (`VeilProducer` - basic for now)
    - Timeline management (`TimelineComponent`)
    """
    # EVENT_TYPES handled by components. Element itself might not need many.
    EVENT_TYPES: List[str] = [] 
    
    # Define which events should trigger agent attention - Could move to an AttentionComponent later
    ATTENTION_SIGNALS = [
        "message_received", # Typically incoming messages need attention
    ]
    
    def __init__(self, element_id: str, name: str, description: str,
                 platform: str = "unknown", adapter_id: str = "default", 
                 is_remote: bool = False):
        """
        Initialize the chat element.
        
        Args:
            element_id: Unique identifier for this element
            name: Human-readable name for this element
            description: Description of this element's purpose
            platform: Platform identifier (e.g., 'discord', 'zulip', 'telegram')
            adapter_id: ID of the adapter this element is associated with
            is_remote: Flag if this represents a remote chat (e.g., via Uplink)
        """
        super().__init__(element_id, name, description)
        
        # Store config needed by components or the element itself
        self.platform = platform
        self.adapter_id = adapter_id
        self._is_remote = is_remote
        
        # --- Add Core Components --- 
        # Add TimelineComponent first as others depend on it
        self._timeline_comp = self.add_component(TimelineComponent)
        if not self._timeline_comp:
             logger.error(f"Failed to add TimelineComponent to ChatElement {element_id}")
             # Decide how critical this is - maybe raise an error?

        # History management
        self._history_comp = self.add_component(HistoryComponent)
        if not self._history_comp:
             logger.error(f"Failed to add HistoryComponent to ChatElement {element_id}")

        # Publishing messages
        self._publisher_comp = self.add_component(PublisherComponent, platform=self.platform, adapter_id=self.adapter_id)
        if not self._publisher_comp:
             logger.error(f"Failed to add PublisherComponent to ChatElement {element_id}")
        
        # Tool provider
        self._tool_provider = self.add_component(ToolProvider)
        if not self._tool_provider:
             logger.error(f"Failed to add ToolProvider component to ChatElement {element_id}")
        else:
             # Register tools only if ToolProvider was added successfully
             self._register_messaging_tools()

        # Basic Veil Producer
        # A more specific ChatVeilProducer could be created later
        self._veil_producer = self.add_component(VeilProducer, renderable_id=f"chat_{element_id}") 
        if not self._veil_producer:
             logger.error(f"Failed to add VeilProducer component to ChatElement {element_id}")
            
        logger.info(f"Created chat element for {platform} (adapter: {adapter_id}): {name} ({element_id})")

    # --- Tool Registration (Delegated to ToolProvider) --- 
    def _register_messaging_tools(self) -> None:
        """Register tools using the ToolProvider component."""
        
        # Ensure tool provider exists before registering
        if not self._tool_provider:
            return
            
        @self._tool_provider.register_tool(
            name="send_message",
            description="Send a message to the chat",
            parameter_descriptions={
                "conversation_id": "ID of the conversation/channel",
                "text": "Content of the message",
                "reply_to_message_id": "Optional ID of the message to reply to"
            }
        )
        def send_message(conversation_id: str, text: str, reply_to_message_id: Optional[str] = None) -> Dict[str, Any]:
            """Handles sending a message by requesting publication via an event."""
            if not self._publisher_comp:
                return {"success": False, "error": "Publisher component not available"}
                
            # Generate a temporary ID for optimistic UI/history update if needed
            temp_message_id = f"agent_msg_{uuid.uuid4().hex[:8]}"
            
            message_data = {
                "event_type": "send_message", # Type for the external system
                "adapter_id": self.adapter_id, 
                "data": {
                    "conversation_id": conversation_id,
                    "text": text,
                    "platform": self.platform, 
                    "sender": {"id": "agent", "name": "Agent"}, # TODO: Get proper agent ID/Name
                    "timestamp": int(time.time() * 1000),
                    "message_id": temp_message_id 
                }
            }
            if reply_to_message_id:
                message_data["data"]["reply_to_message_id"] = reply_to_message_id
                
            # Use handle_event to trigger the publisher on the correct timeline
            # The PublisherComponent listens for "publish_message_request"
            timeline_id = self._timeline_comp.get_primary_timeline() if self._timeline_comp else None
            if not timeline_id:
                 return {"success": False, "error": "Cannot determine primary timeline for publishing"}
                 
            # Trigger the publish request event
            event_to_publish = {
                "event_type": "publish_message_request", 
                "data": message_data # Pass the fully formed message data
            }
            success = self.handle_event(
                event=event_to_publish,
                timeline_context={"timeline_id": timeline_id}
            )
            
            # Optional: Optimistically add to history immediately 
            # if success and self._history_comp:
            #      self._history_comp.add_message(message_data['data'], timeline_id)
            
            return {
                "success": success,
                "message_id": temp_message_id if success else None,
                "status": "Message requested for sending."
            }

        @self._tool_provider.register_tool(
            name="edit_message",
            description="Edit a previously sent message",
            parameter_descriptions={
                "conversation_id": "ID of the conversation/channel",
                "message_id": "ID of the message to edit",
                "text": "New content of the message"
            }
        )
        def edit_message(conversation_id: str, message_id: str, text: str) -> Dict[str, Any]:
            """Handles editing a message by requesting publication via an event."""
            if not self._publisher_comp:
                return {"success": False, "error": "Publisher component not available"}
                
            message_data = {
                "event_type": "edit_message", # Type for the external system
                "adapter_id": self.adapter_id,
                "data": {
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                    "text": text,
                    "platform": self.platform,
                    "timestamp": int(time.time() * 1000)
                }
            }
            
            timeline_id = self._timeline_comp.get_primary_timeline() if self._timeline_comp else None
            if not timeline_id:
                 return {"success": False, "error": "Cannot determine primary timeline for publishing edit"}
                 
            # Trigger the publish request event
            event_to_publish = {
                "event_type": "publish_message_request", 
                "data": message_data
            }
            success = self.handle_event(
                event=event_to_publish,
                timeline_context={"timeline_id": timeline_id}
            )
            
            return {"success": success, "status": "Edit requested."} 

        @self._tool_provider.register_tool(
            name="delete_message",
            description="Delete a previously sent message",
            parameter_descriptions={
                "conversation_id": "ID of the conversation/channel",
                "message_id": "ID of the message to delete"
            }
        )
        def delete_message(conversation_id: str, message_id: str) -> Dict[str, Any]:
            """Handles deleting a message by requesting publication via an event."""
            if not self._publisher_comp:
                return {"success": False, "error": "Publisher component not available"}
                
            message_data = {
                "event_type": "delete_message", # Type for the external system
                "adapter_id": self.adapter_id,
                "data": {
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                    "platform": self.platform,
                    "timestamp": int(time.time() * 1000)
                }
            }
            
            timeline_id = self._timeline_comp.get_primary_timeline() if self._timeline_comp else None
            if not timeline_id:
                 return {"success": False, "error": "Cannot determine primary timeline for publishing delete"}
                 
            # Trigger the publish request event
            event_to_publish = {
                "event_type": "publish_message_request", 
                "data": message_data
            }
            success = self.handle_event(
                event=event_to_publish,
                timeline_context={"timeline_id": timeline_id}
            )
            
            return {"success": success, "status": "Delete requested."} 

        @self._tool_provider.register_tool(
            name="send_typing_indicator",
            description="Send a typing indicator to the chat",
            parameter_descriptions={
                "conversation_id": "ID of the conversation/channel",
                "is_typing": "Boolean indicating if typing (true) or stopped typing (false)"
            }
        )
        def send_typing_indicator(conversation_id: str, is_typing: bool = True) -> Dict[str, Any]:
            """Handles sending a typing indicator via an event."""
            if not self._publisher_comp:
                return {"success": False, "error": "Publisher component not available"}
                
            indicator_data = {
                "event_type": "typing_indicator", # Type for the external system
                "adapter_id": self.adapter_id,
                "data": {
                    "conversation_id": conversation_id,
                    "is_typing": is_typing,
                    "platform": self.platform,
                    "timestamp": int(time.time() * 1000)
                }
            }
            
            timeline_id = self._timeline_comp.get_primary_timeline() if self._timeline_comp else None
            if not timeline_id:
                 # Publisher enforces primary timeline check internally for indicators too (currently)
                 return {"success": False, "error": "Cannot determine primary timeline for indicator"}
                 
            # Trigger the publish indicator request event
            event_to_publish = {
                 "event_type": "publish_indicator_request", 
                 "data": indicator_data
            }
            success = self.handle_event(
                event=event_to_publish,
                timeline_context={"timeline_id": timeline_id} 
            )
            
            return {"success": success, "status": "Indicator requested."}
            
        # Tool to get history (useful for agent introspection)
        @self._tool_provider.register_tool(
            name="get_message_history",
            description="Get recent message history for a specific timeline",
            parameter_descriptions={
                 "timeline_id": "ID of the timeline (optional, defaults to primary)",
                 "limit": "Max number of messages (optional)"
            }
        )
        def get_message_history(timeline_id: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
            if not self._history_comp:
                 return {"success": False, "error": "History component not available", "messages": []}
                 
            target_timeline_id = timeline_id
            if not target_timeline_id:
                 target_timeline_id = self._timeline_comp.get_primary_timeline() if self._timeline_comp else None
                 
            if not target_timeline_id:
                 return {"success": False, "error": "Cannot determine timeline ID", "messages": []}
                 
            messages = self._history_comp.get_messages(target_timeline_id, limit=limit)
            return {"success": True, "timeline_id": target_timeline_id, "messages": messages}

    # --- Event Handling (Mostly Delegated) ---
    # Override handle_event ONLY if ChatElement needs to coordinate BETWEEN components.
    # For example, if an incoming message should trigger attention.
    def handle_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        # Let components handle the event first
        handled_by_components = super().handle_event(event, timeline_context)
        
        # Element-specific coordination:
        event_type = event.get("event_type")
        if event_type in self.ATTENTION_SIGNALS:
            # TODO: Implement Attention Request Logic (potentially via an AttentionComponent)
            logger.debug(f"ChatElement {self.id} detected attention signal: {event_type}. Attention request logic TBD.")
            # Example: 
            # attention_comp = self.get_component_by_type("attention")
            # if attention_comp:
            #     attention_comp.request_attention(event=event, timeline_context=timeline_context)
            #     handled_by_components = True # Consider it handled if attention was triggered
                
        return handled_by_components

    # --- Convenience Getters for Components (Optional) ---
    def get_history_component(self) -> Optional[HistoryComponent]:
        return self._history_comp
        
    def get_publisher_component(self) -> Optional[PublisherComponent]:
        return self._publisher_comp
        
    def get_tool_provider_component(self) -> Optional[ToolProvider]:
        return self._tool_provider

    # --- Obsolete Methods from original ChatElement ---
    # Methods like publish_message, publish_typing_indicator, add_message_to_history,
    # get_messages, update_message, delete_message are now handled by components.

    # --- Existing Methods not refactored (Review if still needed) ---
    # Keep methods not directly related to core component functions if they are still relevant
    # For example, methods related to delegate setup might need review based on VEIL strategy.
    
    # Example of a potentially kept method (if its logic isn't component-based):
    # def some_other_chat_specific_logic(self):
    #    pass

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