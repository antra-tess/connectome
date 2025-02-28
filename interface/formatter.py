"""
Chat Formatter
Formats multi-user chat messages for inclusion in the LLM context.
"""

import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_multi_user_chat(messages: List[Dict[str, Any]]) -> str:
    """
    Format a list of messages from multiple users into a text format
    that can be included in the LLM context.
    
    Args:
        messages: List of message dictionaries, each containing:
            - user_id: Identifier for the user
            - content: Message content
            - role: Role of the message (user, assistant, tool_result)
            
    Returns:
        Formatted multi-user chat as a string
    """
    formatted_lines = []
    
    # Group consecutive messages from the same user
    current_user = None
    current_content_lines = []
    
    for msg in messages:
        user_id = msg.get('user_id', 'unknown')
        content = msg.get('content', '')
        role = msg.get('role', 'user')
        
        # Skip empty messages
        if not content.strip():
            continue
        
        # Handle tool results differently - they're typically not shown to users
        if role == 'tool_result':
            formatted_lines.append(f"[Tool Result]: {content}")
            continue
            
        # If this is a new user, flush the previous user's messages
        if user_id != current_user and current_content_lines:
            if current_user == 'agent':
                formatted_lines.append(f"[Assistant]: {' '.join(current_content_lines)}")
            else:
                formatted_lines.append(f"[User {current_user}]: {' '.join(current_content_lines)}")
            current_content_lines = []
        
        # Set current user and add this content
        current_user = user_id
        current_content_lines.append(content)
    
    # Flush the last user's messages
    if current_content_lines:
        if current_user == 'agent':
            formatted_lines.append(f"[Assistant]: {' '.join(current_content_lines)}")
        else:
            formatted_lines.append(f"[User {current_user}]: {' '.join(current_content_lines)}")
    
    # Join all lines with newlines
    return "\n".join(formatted_lines)


def format_message_for_display(message: Dict[str, Any]) -> str:
    """
    Format a single message for display to users.
    
    Args:
        message: Message dictionary containing:
            - user_id: Identifier for the user
            - content: Message content
            - role: Role of the message
            
    Returns:
        Formatted message as a string
    """
    user_id = message.get('user_id', 'unknown')
    content = message.get('content', '')
    role = message.get('role', 'user')
    
    if role == 'assistant':
        return f"[Assistant]: {content}"
    elif role == 'user':
        return f"[{user_id}]: {content}"
    elif role == 'tool_result':
        return f"[System]: Tool executed with result: {content}"
    else:
        return f"[{user_id}]: {content}"


def format_conversation_summary(messages: List[Dict[str, Any]], max_length: int = 200) -> str:
    """
    Create a brief summary of the conversation.
    
    Args:
        messages: List of messages in the conversation
        max_length: Maximum length of the summary
        
    Returns:
        Brief summary of the conversation
    """
    if not messages:
        return "No messages in conversation."
    
    # Count messages by role
    user_count = sum(1 for msg in messages if msg.get('role') == 'user')
    assistant_count = sum(1 for msg in messages if msg.get('role') == 'assistant')
    
    # Get unique users
    unique_users = set(msg.get('user_id') for msg in messages if msg.get('role') == 'user')
    
    # Get time span
    if len(messages) > 1 and all('timestamp' in msg for msg in messages):
        import datetime
        timestamps = [msg.get('timestamp') for msg in messages]
        first_time = datetime.datetime.fromtimestamp(min(timestamps))
        last_time = datetime.datetime.fromtimestamp(max(timestamps))
        time_span = f"from {first_time.strftime('%Y-%m-%d %H:%M')} to {last_time.strftime('%Y-%m-%d %H:%M')}"
    else:
        time_span = ""
    
    # Create summary
    summary = (
        f"Conversation with {len(unique_users)} user(s) {time_span}. "
        f"Contains {user_count} user message(s) and {assistant_count} assistant response(s)."
    )
    
    # Add recent topics if possible
    if len(messages) > 0:
        recent_topics = messages[-min(3, len(messages)):]
        topics = "; ".join([msg.get('content', '')[:30] + "..." for msg in recent_topics if msg.get('role') == 'user'])
        if topics:
            summary += f" Recent topics: {topics}"
    
    # Truncate if needed
    if len(summary) > max_length:
        summary = summary[:max_length-3] + "..."
    
    return summary 