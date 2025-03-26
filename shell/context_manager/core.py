"""
Context Manager Core Implementation

Provides the core Context Manager functionality for assembling context
for agent interactions.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from shell.hud.core import RenderingResponse


@dataclass
class ContextRequest:
    """
    Request for context assembly.
    
    Attributes:
        rendering_response: The rendering response from the HUD
        timeline_context: The timeline context for this request
        additional_context: Optional additional context to include
    """
    rendering_response: RenderingResponse
    timeline_context: Dict[str, Any]
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextResponse:
    """
    Response from context assembly.
    
    Attributes:
        context: The assembled context as a string
        token_count: Estimated token count of the context
        has_truncations: Whether any content was truncated
    """
    context: str
    token_count: int
    has_truncations: bool = False


class ContextManager:
    """
    Context Manager for assembling context for agent interactions.
    
    Responsible for:
    1. Taking rendering results from the HUD
    2. Assembling them into a coherent context
    3. Providing a simplified interface for memory hooks
    4. Ensuring the context fits within model constraints
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Context Manager.
        
        Args:
            config: Optional configuration for the Context Manager
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Initializing Context Manager")
        
        # Default token limit (can be overridden in config)
        self.token_limit = self.config.get("token_limit", 8000)
        
        # Simple memory hooks for plugins
        self.memory_hooks = []
        
    def assemble_context(self, request: ContextRequest) -> ContextResponse:
        """
        Assemble a context for agent interaction.
        
        Args:
            request: Context assembly request
            
        Returns:
            Response containing the assembled context
        """
        self.logger.info("Assembling context for agent interaction")
        
        # Get the rendering response from the HUD
        rendering_response = request.rendering_response
        
        # Create a list to hold content sections
        context_sections = []
        total_tokens = 0
        has_truncations = False
        
        # Add important metadata from the rendering response
        if rendering_response.metadata:
            metadata_text = self._format_metadata(rendering_response.metadata)
            metadata_tokens = self._estimate_tokens(metadata_text)
            context_sections.append(metadata_text)
            total_tokens += metadata_tokens
        
        # Add the main content
        main_content = rendering_response.content
        main_content_tokens = self._estimate_tokens(main_content)
        
        # Check if adding this would exceed our token limit
        if total_tokens + main_content_tokens > self.token_limit:
            # Try to add a truncated version
            available_tokens = self.token_limit - total_tokens
            if available_tokens > 100:  # Only add if we can include something meaningful
                truncated_text = self._truncate_text(main_content, available_tokens)
                context_sections.append(truncated_text)
                context_sections.append("\n[Content truncated due to token limits]\n")
                has_truncations = True
                total_tokens += self._estimate_tokens(truncated_text) + 10  # Approximate for truncation message
            else:
                has_truncations = True
        else:
            # Add the full content
            context_sections.append(main_content)
            total_tokens += main_content_tokens
        
        # Add sections if present
        if rendering_response.sections:
            for section_name, section_content in rendering_response.sections.items():
                section_text = f"\n## {section_name}\n{section_content}"
                section_tokens = self._estimate_tokens(section_text)
                
                # Check if adding this would exceed our token limit
                if total_tokens + section_tokens > self.token_limit:
                    has_truncations = True
                    continue
                
                context_sections.append(section_text)
                total_tokens += section_tokens
        
        # Add any additional context provided in the request
        if request.additional_context:
            additional_text = self._format_additional_context(request.additional_context)
            additional_tokens = self._estimate_tokens(additional_text)
            
            if total_tokens + additional_tokens <= self.token_limit:
                context_sections.append(additional_text)
                total_tokens += additional_tokens
            else:
                has_truncations = True
        
        # Join sections with separators
        assembled_context = "\n\n".join(context_sections)
        
        return ContextResponse(
            context=assembled_context,
            token_count=total_tokens,
            has_truncations=has_truncations
        )
    
    def register_memory_hook(self, hook_function):
        """
        Register a memory hook function.
        
        Hook functions receive the rendering results and timeline context,
        and return modified rendering results.
        
        Args:
            hook_function: The hook function to register
        """
        self.logger.info(f"Registering memory hook: {hook_function.__name__}")
        self.memory_hooks.append(hook_function)
        
    def unregister_memory_hook(self, hook_function):
        """
        Unregister a memory hook function.
        
        Args:
            hook_function: The hook function to unregister
            
        Returns:
            True if the hook was removed, False if not found
        """
        if hook_function in self.memory_hooks:
            self.memory_hooks.remove(hook_function)
            self.logger.info(f"Unregistered memory hook: {hook_function.__name__}")
            return True
        return False
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata into a string representation."""
        lines = ["# Context Information"]
        
        for key, value in metadata.items():
            if key == "timestamp":
                lines.append(f"Current Time: {value}")
            elif key == "token_usage":
                lines.append(f"Context Token Usage: {value}/{self.token_limit}")
            else:
                lines.append(f"{key.replace('_', ' ').title()}: {value}")
                
        return "\n".join(lines)
    
    def _format_additional_context(self, additional_context: Dict[str, Any]) -> str:
        """Format additional context into a string representation."""
        if not additional_context:
            return ""
            
        lines = ["# Additional Information"]
        
        for key, value in additional_context.items():
            if isinstance(value, str):
                lines.append(f"## {key.replace('_', ' ').title()}")
                lines.append(value)
            elif isinstance(value, dict):
                lines.append(f"## {key.replace('_', ' ').title()}")
                for subkey, subvalue in value.items():
                    lines.append(f"- {subkey}: {subvalue}")
            elif isinstance(value, list):
                lines.append(f"## {key.replace('_', ' ').title()}")
                for item in value:
                    lines.append(f"- {item}")
            else:
                lines.append(f"## {key.replace('_', ' ').title()}")
                lines.append(str(value))
                
        return "\n".join(lines)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        
        This is a simple approximation. For production, should use
        actual tokenizer from the LLM being used.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Simple approximation: 1 token ≈ 4 characters or 0.75 words
        return max(len(text) // 4, len(text.split()) * 3 // 4)
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within max_tokens.
        
        Args:
            text: The text to truncate
            max_tokens: Maximum tokens to allow
            
        Returns:
            Truncated text
        """
        # Simple approximation: 4 chars ≈ 1 token
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text
            
        # Try to truncate at paragraph boundaries first
        paragraphs = text.split("\n\n")
        truncated = ""
        
        for para in paragraphs:
            if len(truncated) + len(para) + 2 <= max_chars:  # +2 for "\n\n"
                if truncated:
                    truncated += "\n\n"
                truncated += para
            else:
                break
                
        if truncated:
            return truncated
            
        # If we can't truncate at paragraph boundaries, just truncate at max_chars
        return text[:max_chars] + "..." 