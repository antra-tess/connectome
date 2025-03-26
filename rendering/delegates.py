"""
Element Delegates

Provides the interfaces and base implementations for Element Delegates, which 
transform element state into rendered text for the agent context.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
import inspect
import time
from datetime import datetime

from .api import (
    RenderingOptions, 
    RenderingResult, 
    RenderingMetadata,
    RenderingImportance,
    RenderingFormat,
    CompressionHint,
    registry
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ElementDelegate(ABC):
    """
    Base class for all Element Delegates.
    
    Element Delegates are responsible for transforming element state into
    renderable content for the agent's context. They provide the bridge
    between the element's internal state and its external representation.
    """
    
    def __init__(self, element=None):
        """
        Initialize the element delegate.
        
        Args:
            element: Optional reference to the associated element
        """
        self.element = element
    
    @abstractmethod
    def render(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the element state into a text representation.
        
        Args:
            state: Current state of the element
            options: Rendering options
            
        Returns:
            Rendering result containing the rendered content and metadata
        """
        pass
    
    def render_exterior(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the element's exterior view (when closed).
        
        Args:
            state: Current state of the element
            options: Rendering options
            
        Returns:
            Rendering result containing the exterior view
        """
        # Default implementation returns a compact representation
        element_id = self.get_element_id()
        element_type = self.get_element_type()
        
        if options.format == RenderingFormat.MARKDOWN:
            content = f"**{element_type}**: {element_id}"
        else:
            content = f"{element_type}: {element_id}"
            
        metadata = self.create_metadata(
            importance=RenderingImportance.LOW,
            format=options.format,
            compression_hint=CompressionHint.OMIT
        )
        
        return RenderingResult(
            content=content,
            metadata=metadata
        )
    
    def render_interior(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the element's interior view (when open).
        
        Args:
            state: Current state of the element
            options: Rendering options
            
        Returns:
            Rendering result containing the interior view
        """
        # Default implementation calls the main render method
        return self.render(state, options)
    
    def get_element_id(self) -> str:
        """Get the ID of the associated element."""
        if self.element and hasattr(self.element, 'id'):
            return self.element.id
        return "unknown_element"
    
    def get_element_type(self) -> str:
        """Get the type of the associated element."""
        if self.element:
            return self.element.__class__.__name__
        return "unknown_type"
    
    def create_metadata(self, 
                       importance: RenderingImportance = RenderingImportance.MEDIUM,
                       format: RenderingFormat = RenderingFormat.TEXT,
                       compression_hint: CompressionHint = CompressionHint.SUMMARIZE,
                       related_elements: List[str] = None,
                       attributes: Dict[str, Any] = None) -> RenderingMetadata:
        """
        Create metadata for the rendering result.
        
        Args:
            importance: Importance level
            format: Format of the content
            compression_hint: Hint for compression
            related_elements: Related element IDs
            attributes: Additional attributes
            
        Returns:
            Rendering metadata
        """
        return RenderingMetadata(
            element_id=self.get_element_id(),
            element_type=self.get_element_type(),
            importance=importance,
            format=format,
            compression_hint=compression_hint,
            related_elements=related_elements or [],
            attributes=attributes or {}
        )
    
    def register(self) -> None:
        """Register this delegate with the global registry."""
        element_type = self.get_element_type()
        
        def renderer_func(state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
            # Check if element is open/closed
            if self.element and hasattr(self.element, 'is_open'):
                if self.element.is_open():
                    return self.render_interior(state, options)
                else:
                    return self.render_exterior(state, options)
            # Fall back to main render method if state not available
            return self.render(state, options)
            
        registry.register_renderer(element_type, renderer_func)
        logger.debug(f"Registered delegate for element type: {element_type}")


class FunctionDelegate:
    """
    Function-based delegate that uses a function to render element state.
    """
    
    def __init__(self, element_type: str, render_func: Callable):
        """
        Initialize the function delegate.
        
        Args:
            element_type: Type of element this delegate handles
            render_func: Function to render the element
        """
        self.element_type = element_type
        self.render_func = render_func
    
    def register(self) -> None:
        """Register this delegate with the global registry."""
        registry.register_renderer(self.element_type, self.render_func)
        logger.debug(f"Registered function delegate for element type: {self.element_type}")


class StaticDelegate(ElementDelegate):
    """
    Delegate that renders static content for an element.
    """
    
    def __init__(self, element=None, static_content: str = "",
                importance: RenderingImportance = RenderingImportance.LOW,
                format: RenderingFormat = RenderingFormat.TEXT):
        """
        Initialize the static delegate.
        
        Args:
            element: Associated element
            static_content: Static content to render
            importance: Importance level for the content
            format: Format of the content
        """
        super().__init__(element)
        self.static_content = static_content
        self.importance = importance
        self.format = format
    
    def render(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the static content.
        
        Args:
            state: Current state (ignored)
            options: Rendering options
            
        Returns:
            Rendering result with static content
        """
        metadata = self.create_metadata(
            importance=self.importance,
            format=self.format,
            compression_hint=CompressionHint.OMIT
        )
        
        return RenderingResult(
            content=self.static_content,
            metadata=metadata
        )


class DefaultDelegate(ElementDelegate):
    """
    Default delegate that renders a generic representation of element state.
    """
    
    def render(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render the element state in a generic way.
        
        Args:
            state: Current state of the element
            options: Rendering options
            
        Returns:
            Rendering result with generic representation
        """
        element_id = self.get_element_id()
        element_type = self.get_element_type()
        
        if options.format == RenderingFormat.MARKDOWN:
            content = f"## {element_type}: {element_id}\n\n"
            
            # Add state summary
            content += "### State\n\n"
            for key, value in state.items():
                if isinstance(value, dict) or isinstance(value, list):
                    content += f"- **{key}**: *complex data*\n"
                else:
                    content += f"- **{key}**: {value}\n"
        else:
            content = f"{element_type}: {element_id}\n\n"
            
            # Add state summary
            content += "State:\n"
            for key, value in state.items():
                if isinstance(value, dict) or isinstance(value, list):
                    content += f"- {key}: (complex data)\n"
                else:
                    content += f"- {key}: {value}\n"
        
        metadata = self.create_metadata(
            importance=RenderingImportance.LOW,
            format=options.format,
            compression_hint=CompressionHint.SUMMARIZE
        )
        
        return RenderingResult(
            content=content,
            metadata=metadata
        )
    
    def render_exterior(self, state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
        """
        Render a compact exterior view of the element.
        
        Args:
            state: Current state of the element
            options: Rendering options
            
        Returns:
            Rendering result with compact representation
        """
        element_id = self.get_element_id()
        element_type = self.get_element_type()
        
        # Get a brief summary of state if available
        state_summary = ""
        if state:
            # Try to get a meaningful summary field
            summary_fields = ["name", "description", "status", "type"]
            for field in summary_fields:
                if field in state:
                    state_summary = f" - {state[field]}"
                    break
        
        if options.format == RenderingFormat.MARKDOWN:
            content = f"**{element_type}**: {element_id}{state_summary}"
        else:
            content = f"{element_type}: {element_id}{state_summary}"
        
        metadata = self.create_metadata(
            importance=RenderingImportance.LOW,
            format=options.format,
            compression_hint=CompressionHint.OMIT
        )
        
        return RenderingResult(
            content=content,
            metadata=metadata
        )


def render_element_decorator(element_type: str = None, **kwargs):
    """
    Decorator to register a function as an element renderer.
    
    Args:
        element_type: Type of element this function renders
        **kwargs: Additional options for the renderer
        
    Returns:
        Decorator function
    """
    def decorator(func):
        nonlocal element_type
        
        # If element_type not provided, try to derive from function name
        if not element_type:
            func_name = func.__name__
            if func_name.startswith("render_"):
                element_type = func_name[7:]  # Remove "render_" prefix
            else:
                element_type = func_name
        
        # Create a wrapper function that matches the expected signature
        @registry.register_renderer(element_type)
        def wrapper(state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
            return func(state, options, **kwargs)
            
        return func
        
    return decorator


# Register a default renderer for unknown element types
@render_element_decorator("default")
def render_default(state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
    """Default renderer for unknown element types."""
    delegate = DefaultDelegate()
    return delegate.render(state, options)


# ------ Chat Element Delegate Functions ------

def calculate_message_importance(message: Dict[str, Any], options: Dict[str, Any]) -> RenderingImportance:
    """
    Calculate the importance of a chat message for rendering decisions.
    
    Args:
        message: The message data
        options: Rendering options
        
    Returns:
        Importance level for the message
    """
    # Get current time for recency calculations
    now = datetime.now().timestamp() * 1000 if not isinstance(datetime.now().timestamp(), int) else datetime.now().timestamp()
    
    # Messages from the current user are highly important
    if options.get('focus_user_id') and message.get('user_id') == options.get('focus_user_id'):
        return RenderingImportance.HIGH
        
    # Recent messages are more important
    msg_time = message.get('timestamp', 0)
    if isinstance(msg_time, str):
        try:
            # Convert ISO format to timestamp if needed
            msg_time = datetime.fromisoformat(msg_time.replace('Z', '+00:00')).timestamp() * 1000
        except (ValueError, TypeError):
            msg_time = 0
    
    # Messages in the last hour are higher importance
    if now - msg_time < 3600000:  # 1 hour in milliseconds
        return RenderingImportance.HIGH
    
    # Messages with mentions to the agent are important
    if message.get('mentions_agent', False):
        return RenderingImportance.HIGH
        
    # Questions are generally important
    if '?' in message.get('content', '') and message.get('user_id') != 'agent':
        return RenderingImportance.MEDIUM
        
    # Default importance for older messages
    return RenderingImportance.LOW

def determine_compression_hint(message: Dict[str, Any]) -> CompressionHint:
    """
    Determine compression hint for a message.
    
    Args:
        message: The message data
        
    Returns:
        Compression hint for the message
    """
    # System messages can often be omitted
    if message.get('role') == 'system':
        return CompressionHint.OMIT
        
    # Very short messages from the agent can be omitted in compression
    if message.get('role') == 'agent' and len(message.get('content', '')) < 20:
        return CompressionHint.OMIT
        
    # Long messages can be summarized
    if len(message.get('content', '')) > 500:
        return CompressionHint.SUMMARIZE
        
    # Default for normal messages
    return CompressionHint.NONE

def format_message_content(message: Dict[str, Any], options: Dict[str, Any]) -> str:
    """
    Format the content of a message for rendering.
    
    Args:
        message: The message data
        options: Rendering options
        
    Returns:
        Formatted message content
    """
    sender = message.get('display_name') or message.get('user_id', 'unknown')
    content = message.get('content', '')
    
    # Determine how to format based on the options
    if options.get('format') == RenderingFormat.MARKDOWN:
        return f"**{sender}**: {content}"
    elif options.get('format') == RenderingFormat.HTML:
        return f"<div class='message'><span class='sender'>{sender}</span>: {content}</div>"
    else:
        return f"{sender}: {content}"

def render_chat_history(state: Dict[str, Any], options: RenderingOptions) -> RenderingResult:
    """
    Render chat history for a ChatElement.
    
    Args:
        state: The current state of the chat element
        options: Rendering options
        
    Returns:
        Rendering result with the chat history
    """
    # Extract messages from timeline-specific state
    timeline_state = state.get('timeline_state', {})
    active_timeline = state.get('active_timeline', next(iter(timeline_state.get('messages', {}).keys()), None))
    
    if not active_timeline or not timeline_state.get('messages'):
        # No messages to render
        return RenderingResult(
            content="No messages in chat history.",
            metadata=RenderingMetadata(
                element_id=state.get('id', 'unknown'),
                element_type="ChatElement",
                importance=RenderingImportance.LOW,
                format=options.format,
                compression_hint=CompressionHint.OMIT
            )
        )
    
    # Get messages for the active timeline
    messages = timeline_state.get('messages', {}).get(active_timeline, [])
    
    # Render each message with metadata
    rendered_messages = []
    for msg in messages:
        importance = calculate_message_importance(msg, options.__dict__)
        compression_hint = determine_compression_hint(msg)
        
        rendered_messages.append({
            "type": "message",
            "id": msg.get('message_id', f"msg-{len(rendered_messages)}"),
            "content": format_message_content(msg, options.__dict__),
            "raw_content": msg.get('content', ''),
            "metadata": {
                "timestamp": msg.get('timestamp'),
                "user_id": msg.get('user_id'),
                "display_name": msg.get('display_name'),
                "importance": importance.name,
                "compression_hint": compression_hint.value,
                "is_remote": False  # Direct messages in Model 1
            }
        })
    
    # Create the overall rendering result
    metadata = RenderingMetadata(
        element_id=state.get('id', 'unknown'),
        element_type="ChatElement",
        importance=RenderingImportance.MEDIUM,
        format=options.format,
        compression_hint=CompressionHint.SUMMARIZE,
        attributes={
            "total_messages": len(messages),
            "timeline_id": active_timeline,
            "platform": state.get('platform', 'unknown'),
            "adapter_id": state.get('adapter_id', 'unknown')
        }
    )
    
    # Combine all messages into a single content string
    content_lines = [msg["content"] for msg in rendered_messages]
    content = "\n".join(content_lines)
    
    # Create the result with sections for potential compression
    result = RenderingResult(
        content=content,
        metadata=metadata
    )
    
    # Add sections for individual messages to enable partial compression
    for i, msg in enumerate(rendered_messages):
        section_metadata = RenderingMetadata(
            element_id=f"{state.get('id', 'unknown')}-msg-{i}",
            element_type="ChatMessage",
            importance=RenderingImportance[msg["metadata"]["importance"]],
            format=options.format,
            compression_hint=CompressionHint(msg["metadata"]["compression_hint"]),
        )
        
        section = RenderingResult(
            content=msg["content"],
            metadata=section_metadata
        )
        
        result.with_section(f"message-{i}", section)
    
    return result

def render_remote_chat_history(state: Dict[str, Any], options: RenderingOptions, connection_spans: List[Dict[str, Any]]) -> RenderingResult:
    """
    Render remote chat history for an uplinked ChatElement.
    
    Args:
        state: The current state of the chat element
        options: Rendering options
        connection_spans: Connection spans for remote history
        
    Returns:
        Rendering result with the remote chat history
    """
    # Process each connection span
    rendered_spans = []
    
    for span in connection_spans:
        # Get messages for this span
        span_messages = state.get('history_bundles', {}).get(span.get('span_id'), {}).get('events', [])
        
        # Determine span rendering approach based on age
        span_start_time = span.get('start_time', 0)
        now = time.time() * 1000  # current time in milliseconds
        
        # Recent spans (last 24 hours): Detailed view
        if now - span_start_time < 86400000:  # 24 hours in milliseconds
            span_content = render_span_detailed(span_messages, options)
            compression_hint = CompressionHint.SUMMARIZE
        # Mid-term spans (1-7 days): Q&A pairs
        elif now - span_start_time < 604800000:  # 7 days in milliseconds
            span_content = render_span_qa_pairs(span_messages, options)
            compression_hint = CompressionHint.SUMMARIZE
        # Historical spans (older): Summary only
        else:
            span_content = render_span_summary(span_messages, options)
            compression_hint = CompressionHint.REFERENCE
        
        # Create metadata for this span
        span_metadata = RenderingMetadata(
            element_id=f"{state.get('id', 'unknown')}-span-{span.get('span_id')}",
            element_type="RemoteChatSpan",
            importance=RenderingImportance.MEDIUM,
            format=options.format,
            compression_hint=compression_hint,
            attributes={
                "span_id": span.get('span_id'),
                "start_time": span.get('start_time'),
                "end_time": span.get('end_time'),
                "is_active": span.get('is_active', False),
                "message_count": len(span_messages)
            }
        )
        
        # Add to rendered spans
        rendered_spans.append(RenderingResult(
            content=span_content,
            metadata=span_metadata
        ))
    
    # Create the overall rendering result
    metadata = RenderingMetadata(
        element_id=state.get('id', 'unknown'),
        element_type="RemoteChatElement",
        importance=RenderingImportance.MEDIUM,
        format=options.format,
        compression_hint=CompressionHint.SUMMARIZE,
        attributes={
            "remote_space_id": state.get('remote_space_id', 'unknown'),
            "connection_span_count": len(connection_spans),
            "has_active_connection": any(span.get('is_active', False) for span in connection_spans),
            "platform": state.get('platform', 'unknown'),
            "adapter_id": state.get('adapter_id', 'unknown')
        }
    )
    
    # Combine spans into a single content string
    span_lines = []
    for span in rendered_spans:
        span_start = datetime.fromtimestamp(span.metadata.attributes.get('start_time', 0)/1000).strftime('%Y-%m-%d %H:%M')
        span_status = "[ACTIVE]" if span.metadata.attributes.get('is_active') else ""
        
        span_lines.append(f"--- Connection Span: {span_start} {span_status} ---")
        span_lines.append(span.content)
    
    content = "\n".join(span_lines)
    
    # Create the result with sections for potential compression
    result = RenderingResult(
        content=content,
        metadata=metadata
    )
    
    # Add sections for individual spans to enable partial compression
    for i, span in enumerate(rendered_spans):
        result.with_section(f"span-{i}", span)
    
    return result

def render_span_detailed(messages: List[Dict[str, Any]], options: RenderingOptions) -> str:
    """
    Render a connection span with detailed message content.
    
    Args:
        messages: Messages in this span
        options: Rendering options
        
    Returns:
        Rendered span content
    """
    if not messages:
        return "No messages in this connection span."
    
    # Render each message in detail
    content_lines = []
    for msg in messages:
        sender = msg.get('display_name') or msg.get('user_id', 'unknown')
        content = msg.get('content', '')
        
        if options.format == RenderingFormat.MARKDOWN:
            content_lines.append(f"**{sender}**: {content}")
        else:
            content_lines.append(f"{sender}: {content}")
    
    return "\n".join(content_lines)

def render_span_qa_pairs(messages: List[Dict[str, Any]], options: RenderingOptions) -> str:
    """
    Render a connection span preserving question-answer pairs.
    
    Args:
        messages: Messages in this span
        options: Rendering options
        
    Returns:
        Rendered span content with Q&A pairs
    """
    if not messages:
        return "No messages in this connection span."
    
    # Find question-answer pairs
    qa_pairs = []
    current_question = None
    
    for msg in messages:
        user_id = msg.get('user_id', '')
        content = msg.get('content', '')
        
        # User messages with questions are potential question starters
        if user_id != 'agent' and ('?' in content or any(kw in content.lower() for kw in ['how', 'what', 'why', 'when', 'where', 'who'])):
            # If we have an existing question without an answer, add it solo
            if current_question:
                qa_pairs.append((current_question, None))
            
            # Set the new question
            current_question = msg
        
        # Agent responses after a question are answers
        elif user_id == 'agent' and current_question:
            qa_pairs.append((current_question, msg))
            current_question = None
    
    # Add any remaining question
    if current_question:
        qa_pairs.append((current_question, None))
    
    # If no Q&A pairs found, summarize the span
    if not qa_pairs:
        message_count = len(messages)
        user_count = len(set(msg.get('user_id', '') for msg in messages))
        
        return f"Connection span with {message_count} messages from {user_count} participants."
    
    # Render Q&A pairs
    content_lines = []
    for question, answer in qa_pairs:
        q_sender = question.get('display_name') or question.get('user_id', 'unknown')
        q_content = question.get('content', '')
        
        if options.format == RenderingFormat.MARKDOWN:
            content_lines.append(f"**Q ({q_sender})**: {q_content}")
        else:
            content_lines.append(f"Q ({q_sender}): {q_content}")
        
        if answer:
            a_content = answer.get('content', '')
            
            if options.format == RenderingFormat.MARKDOWN:
                content_lines.append(f"**A (agent)**: {a_content}")
            else:
                content_lines.append(f"A (agent): {a_content}")
    
    return "\n".join(content_lines)

def render_span_summary(messages: List[Dict[str, Any]], options: RenderingOptions) -> str:
    """
    Render a connection span as a summary.
    
    Args:
        messages: Messages in this span
        options: Rendering options
        
    Returns:
        Rendered span summary
    """
    if not messages:
        return "No messages in this connection span."
    
    # Create a summary of the span
    message_count = len(messages)
    user_count = len(set(msg.get('user_id', '') for msg in messages))
    
    # Extract key topics through basic keyword analysis
    all_content = " ".join([msg.get('content', '') for msg in messages])
    words = all_content.lower().split()
    # Filter out common stop words
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    content_words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Count word frequencies
    from collections import Counter
    word_counts = Counter(content_words)
    
    # Get top 5 keywords
    top_keywords = [keyword for keyword, _ in word_counts.most_common(5)]
    
    # Create summary
    summary = f"Historical connection with {message_count} messages from {user_count} participants."
    
    if top_keywords:
        summary += f" Topics included: {', '.join(top_keywords)}."
    
    return summary

# Register the chat rendering functions
registry.register_renderer("ChatElement", render_chat_history)
registry.register_renderer("RemoteChatElement", render_remote_chat_history)

# Create a factory function for chat delegates
def create_chat_delegate(is_remote: bool = False):
    """
    Create a chat delegate for either direct or remote chat elements.
    
    Args:
        is_remote: Whether this is a remote chat element
        
    Returns:
        The appropriate delegate function
    """
    if is_remote:
        return lambda state, options, connection_spans: render_remote_chat_history(state, options, connection_spans)
    else:
        return lambda state, options: render_chat_history(state, options) 