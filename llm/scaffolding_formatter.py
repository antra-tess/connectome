import logging
import time
import threading
from typing import Dict, Any, List, Optional, Union

from opentelemetry import trace
from llm.provider_interface import (
    LLMMessage,
    LLMToolDefinition
)

from host.observability import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

class ScaffoldingFormatter:
    """Formatter implementation for Scaffolding LLMProvider."""

    def format_context(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Format a list of LLMMessages for display in the web interface."""
        # Metadata now travels with LLMMessage objects, no need for original_context tracking
        return [self.format_message(msg) for msg in messages]

    def format_message(self, message: LLMMessage) -> Dict[str, Any]:
        """Format an LLMMessage for display in the web interface."""
        try:
            formatted = {
                "role": message.role,
                "content": self._format_content(message.content),
                "name": getattr(message, 'name', None)
            }

            # Add metadata about multimodal content
            if message.is_multimodal():
                formatted["is_multimodal"] = True
                formatted["attachment_count"] = message.get_attachment_count()
                formatted["text_length"] = len(message.get_text_content())
            else:
                formatted["is_multimodal"] = False
                formatted["text_length"] = len(str(message.content))

            # Use turn metadata directly from message object (no more dual tracking)
            if hasattr(message, 'turn_metadata') and message.turn_metadata:
                formatted["turn_metadata"] = message.turn_metadata

            return formatted

        except Exception as e:
            logger.error(f"Error formatting message for display: {e}")
            return {
                "role": message.role,
                "content": f"Error formatting message: {str(e)}",
                "is_multimodal": False,
                "text_length": 0
            }

    def format_tool(self, tool: LLMToolDefinition) -> Dict[str, Any]:
        """Format a tool definition for display in the web interface."""
        try:
            return {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        except Exception as e:
            logger.error(f"Error formatting tool for display: {e}")
            return {
                "name": "unknown",
                "description": f"Error formatting tool: {str(e)}",
                "parameters": {}
            }

    def _format_content(self, content: Union[str, List[Dict[str, Any]]]) -> str:
        """Format message content for display in the web interface."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            # Handle multimodal content
            formatted_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        formatted_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "unknown")
                        # Truncate very long URLs for display
                        display_url = url[:100] + "..." if len(url) > 100 else url
                        formatted_parts.append(f"[IMAGE: {display_url}]")
                    elif part.get("type") == "image":
                        formatted_parts.append("[IMAGE: Base64 data]")
                    else:
                        formatted_parts.append(f"[{part.get('type', 'UNKNOWN').upper()}: {str(part)[:50]}...]")
                else:
                    formatted_parts.append(str(part))
            return "\n".join(formatted_parts)

        return str(content)
