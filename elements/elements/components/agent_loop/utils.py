"""
Agent Loop Utilities
Shared utility functions for agent loop components.
"""
from typing import Dict, Any, Optional, Union

# NEW: Import LLMMessage for correct provider interaction
from llm.provider_interface import LLMMessage


def create_multimodal_llm_message(role: str, context_data: Union[str, Dict[str, Any]], 
                                 name: Optional[str] = None, turn_metadata: Optional[Dict[str, Any]] = None) -> LLMMessage:
    """
    Create an LLMMessage that supports multimodal content.

    Args:
        role: Message role ("user", "assistant", "system")
        context_data: Either a string (text-only) or dict with 'text' and 'attachments' keys
                     (format automatically determined by HUD based on content)
        name: Optional name for the message
        turn_metadata: Optional metadata about the turn (turn index, facet count, etc.)

    Returns:
        LLMMessage with appropriate content format and metadata
    """
    if isinstance(context_data, str):
        # Simple text message
        return LLMMessage(role=role, content=context_data, name=name, turn_metadata=turn_metadata)

    elif isinstance(context_data, dict) and 'text' in context_data:
        # Multimodal message (HUD detected attachments and returned structured format)
        text_content = context_data['text']
        attachments = context_data.get('attachments', [])

        if not attachments:
            # No attachments, just return text
            return LLMMessage(role=role, content=text_content, name=name, turn_metadata=turn_metadata)

        # Build multimodal content list
        content_parts = []

        # Add text part first
        if text_content.strip():
            content_parts.append({
                "type": "text",
                "text": text_content
            })

        # Add attachment parts
        for attachment in attachments:
            if isinstance(attachment, dict):
                content_parts.append(attachment)
        return LLMMessage(role=role, content=content_parts, name=name, turn_metadata=turn_metadata)

    else:
        # Fallback to string representation
        return LLMMessage(role=role, content=str(context_data), name=name, turn_metadata=turn_metadata) 