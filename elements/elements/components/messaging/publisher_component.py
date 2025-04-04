"""
Messaging Publisher Component
Component for publishing messages from a ChatElement to the Activity Layer.
"""

import logging
from typing import Dict, Any, Optional, List
import uuid
import time

from ..base_component import Component
from ..space.timeline_component import TimelineComponent # Needed for checking primary timeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PublisherComponent(Component):
    """
    Handles publishing messages and indicators to external systems via the Activity Layer.
    Respects primary timeline constraints.
    """
    
    COMPONENT_TYPE: str = "messaging_publisher"
    DEPENDENCIES: List[str] = ["timeline"] # Depends on TimelineComponent for primary check
    
    # Events that might trigger publishing (e.g., a tool execution resulting in a message)
    HANDLED_EVENT_TYPES: List[str] = [
        "publish_message_request",
        "publish_indicator_request"
    ]
    
    def __init__(self, element=None, platform: str = "unknown", adapter_id: str = "default"):
        super().__init__(element)
        self.platform = platform
        self.adapter_id = adapter_id
        self._state = {
            "last_publish_time": None
        }

    def _get_timeline_comp(self) -> Optional[TimelineComponent]:
        """Helper to get the associated TimelineComponent."""
        if not self.element:
            return None
        return self.element.get_component_by_type("timeline")

    def publish_message(self, message_data: Dict[str, Any], timeline_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Publish a message to the Activity Layer if on the primary timeline.
        
        Args:
            message_data: Message data to publish.
            timeline_context: Optional timeline context.
            
        Returns:
            True if the message was published successfully, False otherwise.
        """
        if not self._is_initialized or not self._is_enabled:
            return False

        timeline_comp = self._get_timeline_comp()
        if not timeline_comp:
             logger.error(f"{self.COMPONENT_TYPE}: Cannot publish message, missing TimelineComponent.")
             return False

        # Use provided context or get primary timeline context
        current_timeline_id = timeline_context.get("timeline_id") if timeline_context else timeline_comp.get_primary_timeline()
        is_primary = timeline_comp.is_primary_timeline(current_timeline_id)
        
        # Only publish if this is the primary timeline
        if not is_primary:
            logger.info(f"Not publishing message to external systems (non-primary timeline): {current_timeline_id}")
            return False
            
        # Check for registry access
        registry = self.element.get_registry() if self.element else None
        if not registry:
            logger.error(f"Cannot publish message for element {self.element.id if self.element else 'N/A'}: No registry reference")
            return False
            
        try:
            # Ensure essential fields are present
            if 'event_type' not in message_data or 'data' not in message_data:
                logger.error("Invalid message format: requires 'event_type' and 'data' fields")
                return False
            
            # Add adapter_id if not present
            if 'adapter_id' not in message_data:
                 message_data['adapter_id'] = self.adapter_id
                 
            # Add platform if not present in data
            if 'platform' not in message_data['data']:
                 message_data['data']['platform'] = self.platform
                 
            # Add timestamp if not present
            if 'timestamp' not in message_data:
                 message_data['timestamp'] = int(time.time() * 1000)

            # Basic validation based on event type (can be expanded)
            event_type = message_data['event_type']
            data = message_data['data']
            required_fields = []
            if event_type == 'send_message':
                required_fields = ['conversation_id', 'text']
            elif event_type == 'edit_message':
                required_fields = ['conversation_id', 'message_id', 'text']
            elif event_type == 'delete_message':
                 required_fields = ['conversation_id', 'message_id']
            # Add more validations as needed...

            for field in required_fields:
                 if field not in data:
                      logger.error(f"Invalid {event_type} data: missing required field '{field}'")
                      return False
            
            # Send message through the registry
            success = registry.send_external_message(message_data)
            if success:
                self._state["last_publish_time"] = int(time.time() * 1000)
                logger.debug(f"Published {event_type} via adapter {self.adapter_id}")
            else:
                 logger.error(f"Failed to publish {event_type} via adapter {self.adapter_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False
            
    def publish_indicator(self, indicator_data: Dict[str, Any]) -> bool:
        """
        Publish an indicator (e.g., typing) to the Activity Layer.
        Indicators are often less critical and might not strictly require primary timeline.
        (Decision: For now, let's enforce primary timeline for consistency).
        
        Args:
            indicator_data: Indicator data to publish.
            
        Returns:
            True if the indicator was published, False otherwise.
        """
        if not self._is_initialized or not self._is_enabled:
            return False

        # Check primary timeline status (using default/current context)
        timeline_comp = self._get_timeline_comp()
        if not timeline_comp:
             logger.error(f"{self.COMPONENT_TYPE}: Cannot publish indicator, missing TimelineComponent.")
             return False
        current_timeline_id = timeline_comp.get_primary_timeline()
        if not timeline_comp.is_primary_timeline(current_timeline_id):
             logger.info(f"Not publishing indicator (non-primary timeline): {current_timeline_id}")
             return False # Enforce primary for now
             
        # Check for registry access
        registry = self.element.get_registry() if self.element else None
        if not registry:
            logger.error(f"Cannot publish indicator for element {self.element.id if self.element else 'N/A'}: No registry reference")
            return False

        try:
            # Ensure essential fields
            if 'event_type' not in indicator_data or 'data' not in indicator_data:
                logger.error("Invalid indicator format: requires 'event_type' and 'data' fields")
                return False
            
            # Add adapter_id if not present
            if 'adapter_id' not in indicator_data:
                 indicator_data['adapter_id'] = self.adapter_id
                 
            # Add platform if not present in data
            if 'platform' not in indicator_data['data']:
                 indicator_data['data']['platform'] = self.platform
                 
            # Add timestamp if not present
            if 'timestamp' not in indicator_data:
                 indicator_data['timestamp'] = int(time.time() * 1000)
                 
            # Example validation for typing indicator
            if indicator_data['event_type'] == 'typing_indicator':
                 if 'conversation_id' not in indicator_data['data'] or 'is_typing' not in indicator_data['data']:
                      logger.error("Invalid typing_indicator data: requires 'conversation_id' and 'is_typing'")
                      return False
                      
            # Send indicator through the registry
            success = registry.send_external_message(indicator_data)
            if success:
                 logger.debug(f"Published indicator {indicator_data['event_type']} via adapter {self.adapter_id}")
            else:
                 logger.error(f"Failed to publish indicator {indicator_data['event_type']} via adapter {self.adapter_id}")
            return success

        except Exception as e:
            logger.error(f"Error publishing indicator: {e}")
            return False

    def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handle requests to publish messages or indicators.
        """
        event_type = event.get("event_type")
        data = event.get("data", {})

        if event_type == "publish_message_request":
            # Assume 'data' contains the message_data structure needed for publish_message
            return self.publish_message(data, timeline_context)
            
        elif event_type == "publish_indicator_request":
            # Assume 'data' contains the indicator_data structure needed for publish_indicator
            return self.publish_indicator(data)
            
        return False 