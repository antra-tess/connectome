"""
Context Module
Manages conversation context, including storage, retrieval, and formatting.
"""

from context.context_manager import ContextManager
from context.formatter import (
    format_multi_user_chat,
    format_message_for_display,
    format_conversation_summary
) 