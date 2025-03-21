"""
Message Listener

Handles incoming messages and routes them to the appropriate space.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Callable
import uuid
import time

from ..elements.space_registry import SpaceRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MessageHandler:
    """
    Handles incoming messages and determines their timeline context.
    
    This class is responsible for:
    1. Receiving incoming messages
    2. Determining the timeline context
    3. Passing the message and context to the SpaceRegistry for routing
    
    The MessageHandler does not determine how messages are routed to spaces -
    that is the responsibility of the SpaceRegistry.
    """
    
    def __init__(self, space_registry: SpaceRegistry):
        """
        Initialize the message handler.
        
        Args:
            space_registry: Registry of available spaces
        """
        self.space_registry = space_registry
        logger.info("Initialized MessageHandler")
    
    def _determine_timeline_context(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the timeline context for an incoming message.
        
        This method extracts relevant context information from the event data
        and constructs a unique timeline identifier.
        
        Args:
            event_data: The event data received
            
        Returns:
            Dictionary containing timeline context information
        """
        # Extract relevant fields for timeline context
        adapter_type = event_data.get('adapter_type')
        conversation_id = event_data.get('conversation_id')
        thread_id = event_data.get('thread_id')
        
        # Generate a timeline ID if adapter_type and conversation_id are available
        timeline_id = None
        if adapter_type and conversation_id:
            timeline_id = f"{adapter_type}_{conversation_id}"
        else:
            # Fallback to a random timeline ID if required fields are missing
            timeline_id = f"timeline_{str(uuid.uuid4())[:8]}"
            logger.warning(f"Missing required fields for timeline context. Generated random ID: {timeline_id}")
        
        # Construct the timeline context
        timeline_context = {
            "timeline_id": timeline_id,
            "timestamp": int(time.time() * 1000)
        }
        
        # Add thread ID if available
        if thread_id:
            timeline_context["thread_id"] = thread_id
            
        logger.debug(f"Determined timeline context: {timeline_context}")
        return timeline_context
    
    def handle_message(self, message_data: Dict[str, Any]) -> bool:
        """
        Handle an incoming message.
        
        Args:
            message_data: Raw message data received from an adapter
            
        Returns:
            True if the message was handled successfully, False otherwise
        """
        try:
            # Validate message format
            if not isinstance(message_data, dict):
                logger.error(f"Invalid message format: {message_data}")
                return False
            
            # Convert to standard event format if needed
            event_data = self._standardize_message_format(message_data)
            
            # Determine timeline context
            timeline_context = self._determine_timeline_context(event_data)
            
            # Route the message through the space registry
            success = self.space_registry.route_message(event_data, timeline_context)
            
            if not success:
                logger.warning(f"Failed to route message: {event_data.get('event_type')}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            return False
    
    def handle_clear_context(self, clear_data: Dict[str, Any]) -> bool:
        """
        Handle a clear context request.
        
        Args:
            clear_data: Data specifying which context to clear
            
        Returns:
            True if context was cleared successfully, False otherwise
        """
        try:
            # Extract fields for context identification
            adapter_type = clear_data.get('adapter_type')
            conversation_id = clear_data.get('conversation_id')
            
            # Create event data for context clearing
            event_data = {
                "event_type": "clear_context",
                "adapter_type": adapter_type,
                "conversation_id": conversation_id,
                "timestamp": int(time.time() * 1000)
            }
            
            # Determine timeline context
            timeline_context = self._determine_timeline_context(event_data)
            
            # Route the clear context request
            return self.space_registry.route_message(event_data, timeline_context)
            
        except Exception as e:
            logger.error(f"Error handling clear context: {e}", exc_info=True)
            return False
    
    def _standardize_message_format(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize message format to ensure consistent processing.
        
        Args:
            message_data: Raw message data from an adapter
            
        Returns:
            Standardized event data
        """
        # If the message already has an event_type, it's likely already standardized
        if 'event_type' in message_data:
            return message_data
            
        # Extract common fields
        message_type = message_data.get('type', 'message')
        sender = message_data.get('sender', {})
        content = message_data.get('content', '')
        adapter_type = message_data.get('adapter_type', 'unknown')
        
        # Create standardized event data
        event_data = {
            "event_type": f"message_{message_type}",
            "adapter_type": adapter_type,
            "conversation_id": message_data.get('conversation_id', str(uuid.uuid4())),
            "sender": sender,
            "content": content,
            "timestamp": message_data.get('timestamp', int(time.time() * 1000)),
            "raw_data": message_data
        }
        
        # Add optional fields if present
        for field in ['thread_id', 'attachments', 'metadata']:
            if field in message_data:
                event_data[field] = message_data[field]
                
        return event_data
