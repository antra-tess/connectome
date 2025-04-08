"""
Mock Activity Adapter for Testing
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

# --- Base Class (Conceptual - would move to a base file) ---
class BaseActivityAdapter:
    """Conceptual base class for activity adapters."""
    def __init__(self, adapter_id: str, name: str):
        self.adapter_id = adapter_id
        self.name = name
        self._incoming_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        logger.info(f"Initialized Adapter: {name} ({adapter_id})")

    def set_incoming_event_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Sets the callback to use for injecting incoming events into the Host."""
        self._incoming_event_callback = callback
        logger.info(f"Incoming event callback set for adapter {self.adapter_id}")

    async def start(self):
        """Start any connections or listeners (e.g., Discord client.run)."""
        logger.info(f"Adapter {self.adapter_id} starting (mock has no action).")
        pass # Mock doesn't need to start anything

    async def stop(self):
        """Stop connections and cleanup."""
        logger.info(f"Adapter {self.adapter_id} stopping (mock has no action).")
        pass # Mock doesn't need to stop anything

    async def send_message(self, conversation_id: str, text: str, context: Optional[Dict[str, Any]] = None):
        """Send a message to the external system."""
        raise NotImplementedError
# -----------------------------------------------------------

class MockActivityAdapter(BaseActivityAdapter):
    """A simple mock adapter that logs sends and allows simulating receives."""

    async def send_message(self, conversation_id: str, text: str, context: Optional[Dict[str, Any]] = None):
        """Simulates sending a message by logging it."""
        logger.info(f"[MOCK ADAPTER {self.adapter_id}] SEND to conv '{conversation_id}': {text}")
        # In a real adapter, this would interact with the external API
        await asyncio.sleep(0.01) # Simulate small network delay
        # Simulate a successful send
        return {"success": True, "message_id": f"mock_msg_{int(time.time()*1000)}"}

    def simulate_incoming_message(
        self, 
        text: str, 
        conversation_id: str = "general", 
        user_id: str = "mock_user_123", 
        user_name: str = "Mock User"
    ):
        """
        Simulates receiving a message from the outside.
        Formats the event correctly and uses the callback to inject it.
        """
        if not self._incoming_event_callback:
            logger.error(f"[MOCK ADAPTER {self.adapter_id}] Cannot simulate incoming message: Callback not set.")
            return

        logger.info(f"[MOCK ADAPTER {self.adapter_id}] SIMULATING incoming message from '{user_name}' in '{conversation_id}': {text}")
        
        # --- Format the event payload --- 
        event_payload = {
            "text": text,
            "metadata": {
                "adapter_id": self.adapter_id,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "user_name": user_name,
                "message_id": f"mock_recv_{int(time.time()*1000)}" 
                # Add any other mock metadata needed
            }
        }
        
        # --- Create the full event structure ---
        # This structure should match what ActivityClient expects to put on the loop
        event = {
            "event_type": "external_message_received", # Standardized event type
            "timestamp": int(time.time() * 1000),
            "payload": event_payload,
            "source_adapter_id": self.adapter_id # Maybe useful for routing?
        }
        
        # Inject the event using the callback provided by ActivityClient/HostEventLoop
        try:
            self._incoming_event_callback(event)
            logger.debug(f"[MOCK ADAPTER {self.adapter_id}] Injected simulated event into host loop.")
        except Exception as e:
            logger.error(f"[MOCK ADAPTER {self.adapter_id}] Error calling incoming event callback: {e}", exc_info=True) 