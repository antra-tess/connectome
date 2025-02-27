"""
Memory Management Tools
Tools for managing agent memory and context.
"""

import logging
from typing import Optional, List, Dict, Any

from tools.registry import register_tool
from context.context_manager import ContextManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@register_tool(
    name="compress_context",
    description="Compress the current conversation context to save space.",
    parameter_descriptions={
        "chat_id": "ID of the chat whose context to compress",
        "summary": "Your summary of the conversation so far",
        "keep_last_messages": "Number of recent messages to keep uncompressed (optional)"
    }
)
def compress_context(chat_id: str, summary: str, keep_last_messages: Optional[int] = 5) -> str:
    """
    Compress the conversation context by replacing older messages with a summary.
    
    This tool allows the agent to manage its own context window by summarizing
    older parts of the conversation when the context gets too large.
    
    Args:
        chat_id: ID of the chat whose context to compress
        summary: A summary of the conversation to replace older messages
        keep_last_messages: Number of recent messages to keep (not compress)
        
    Returns:
        Confirmation message
    """
    logger.info(f"Compressing context for chat {chat_id}, keeping {keep_last_messages} recent messages")
    
    try:
        # Get context manager
        context_manager = ContextManager()
        
        # Get all messages
        messages = context_manager.get_context(chat_id)
        
        if len(messages) <= keep_last_messages:
            return "Context is already small enough, no compression needed."
        
        # Keep the most recent messages
        recent_messages = messages[-keep_last_messages:]
        
        # Clear the existing context
        context_manager.clear_context(chat_id)
        
        # Add the summary as a special "memory" message
        context_manager.save_message({
            'chat_id': chat_id,
            'user_id': 'system',
            'content': summary,
            'role': 'memory',
            'memory_type': 'context_summary'
        })
        
        # Re-add the recent messages
        for msg in recent_messages:
            context_manager.save_message(msg)
        
        return f"Context compressed successfully. Replaced {len(messages) - keep_last_messages} older messages with a summary and kept {keep_last_messages} recent messages."
    
    except Exception as e:
        error_msg = f"Error compressing context: {str(e)}"
        logger.error(error_msg)
        return error_msg


@register_tool(
    name="save_memory",
    description="Save an important piece of information as a long-term memory.",
    parameter_descriptions={
        "chat_id": "ID of the chat this memory belongs to",
        "content": "The content to save as a memory",
        "memory_type": "Type of memory (e.g., 'fact', 'preference', 'instruction')",
        "importance": "Importance rating from 1-10 (optional)"
    }
)
def save_memory(chat_id: str, content: str, memory_type: str, importance: Optional[int] = 5) -> str:
    """
    Save an important piece of information as a long-term memory.
    
    This allows the agent to explicitly remember important facts, preferences,
    or instructions that should persist across conversations.
    
    Args:
        chat_id: ID of the chat this memory belongs to
        content: The content to save as a memory
        memory_type: Type of memory (fact, preference, instruction, etc.)
        importance: Importance rating from 1-10
        
    Returns:
        Confirmation message
    """
    logger.info(f"Saving memory for chat {chat_id}, type: {memory_type}, importance: {importance}")
    
    try:
        # Get context manager
        context_manager = ContextManager()
        
        # Save as a special memory message
        context_manager.save_message({
            'chat_id': chat_id,
            'user_id': 'system',
            'content': content,
            'role': 'memory',
            'memory_type': memory_type,
            'importance': importance
        })
        
        return f"Memory saved successfully: {content[:50]}..." if len(content) > 50 else f"Memory saved successfully: {content}"
    
    except Exception as e:
        error_msg = f"Error saving memory: {str(e)}"
        logger.error(error_msg)
        return error_msg


@register_tool(
    name="retrieve_memories",
    description="Retrieve memories relevant to a specific topic or query.",
    parameter_descriptions={
        "chat_id": "ID of the chat to retrieve memories from",
        "query": "Search query to find relevant memories",
        "memory_type": "Type of memories to retrieve (optional)",
        "max_results": "Maximum number of memories to retrieve (optional)"
    }
)
def retrieve_memories(chat_id: str, query: str, memory_type: Optional[str] = None, max_results: Optional[int] = 5) -> str:
    """
    Retrieve memories relevant to a specific topic or query.
    
    This allows the agent to search through saved memories to find
    information relevant to the current conversation.
    
    Args:
        chat_id: ID of the chat to retrieve memories from
        query: Search query to find relevant memories
        memory_type: Type of memories to retrieve (optional filter)
        max_results: Maximum number of memories to retrieve
        
    Returns:
        String containing the retrieved memories
    """
    logger.info(f"Retrieving memories for chat {chat_id}, query: {query}, type: {memory_type}")
    
    try:
        # Get context manager
        context_manager = ContextManager()
        
        # Get all messages
        messages = context_manager.get_context(chat_id)
        
        # Filter to memory messages
        memory_messages = [msg for msg in messages if msg.get('role') == 'memory']
        
        # Filter by memory type if specified
        if memory_type:
            memory_messages = [msg for msg in memory_messages if msg.get('memory_type') == memory_type]
        
        # Simple keyword search for now - in a real implementation, you would use
        # vector similarity search or another more sophisticated method
        relevant_memories = []
        for msg in memory_messages:
            if query.lower() in msg.get('content', '').lower():
                relevant_memories.append(msg)
        
        # Sort by importance if available
        relevant_memories.sort(key=lambda x: x.get('importance', 5), reverse=True)
        
        # Limit to max_results
        relevant_memories = relevant_memories[:max_results]
        
        if not relevant_memories:
            return f"No memories found matching query: {query}"
        
        # Format the memories
        result = f"Found {len(relevant_memories)} memories related to '{query}':\n\n"
        for i, memory in enumerate(relevant_memories, 1):
            content = memory.get('content', '')
            memory_type = memory.get('memory_type', 'unspecified')
            importance = memory.get('importance', 'unspecified')
            
            result += f"{i}. [{memory_type}] (Importance: {importance}): {content}\n\n"
        
        return result
    
    except Exception as e:
        error_msg = f"Error retrieving memories: {str(e)}"
        logger.error(error_msg)
        return error_msg


@register_tool(
    name="analyze_context_needs",
    description="Analyze the current context length and determine if compression is needed.",
    parameter_descriptions={
        "chat_id": "ID of the chat to analyze",
        "threshold_messages": "Threshold for number of messages before compression is recommended (optional)"
    }
)
def analyze_context_needs(chat_id: str, threshold_messages: Optional[int] = 20) -> Dict[str, Any]:
    """
    Analyze the current context length and determine if compression is needed.
    
    This tool helps the agent proactively manage its context window by
    providing information about the current context size.
    
    Args:
        chat_id: ID of the chat to analyze
        threshold_messages: Threshold for number of messages before compression is recommended
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Analyzing context needs for chat {chat_id}")
    
    try:
        # Get context manager
        context_manager = ContextManager()
        
        # Get all messages
        messages = context_manager.get_context(chat_id)
        
        # Calculate total content length
        total_chars = sum(len(msg.get('content', '')) for msg in messages)
        
        # Determine if compression is needed
        needs_compression = len(messages) > threshold_messages
        
        # Prepare result
        result = {
            "message_count": len(messages),
            "total_characters": total_chars,
            "threshold": threshold_messages,
            "needs_compression": needs_compression,
            "recommendation": (
                "Context compression is recommended to maintain efficient operation." 
                if needs_compression else 
                "Context size is within acceptable limits, no compression needed at this time."
            )
        }
        
        return result
    
    except Exception as e:
        error_msg = f"Error analyzing context: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg} 