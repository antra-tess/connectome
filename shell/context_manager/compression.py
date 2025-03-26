"""
Context Compression

Handles compression strategies for the ContextManager to reduce context size
while preserving essential information. Moved from HUD to allow for more flexible
and pluggable compression approaches.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Set
import re

from rendering import (
    RenderingResult,
    RenderingImportance,
    CompressionHint,
    RenderingContext
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CompressorBase:
    """
    Base class for compression strategies.
    
    Compressors are responsible for reducing the size of rendered content
    to fit within token limits while preserving essential information.
    """
    
    def __init__(self, max_tokens: int = 8000):
        """
        Initialize the compressor.
        
        Args:
            max_tokens: Maximum number of tokens allowed
        """
        self.max_tokens = max_tokens
    
    def compress(self, renderings: List[RenderingResult], context: Optional[RenderingContext] = None) -> str:
        """
        Compress the renderings to fit within token limits.
        
        Args:
            renderings: List of rendering results
            context: Optional rendering context
            
        Returns:
            Compressed context as a string
        """
        raise NotImplementedError("Subclasses must implement compress")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        # Simple estimate based on character count
        # More sophisticated implementations could use actual tokenizers
        return len(text) // 4


class PriorityCompressor(CompressorBase):
    """
    Compressor that prioritizes content based on importance and compression hints.
    """
    
    def compress(self, renderings: List[RenderingResult], context: Optional[RenderingContext] = None) -> str:
        """
        Compress renderings by prioritizing content based on importance.
        
        Args:
            renderings: List of rendering results
            context: Optional rendering context
            
        Returns:
            Compressed context
        """
        # Sort renderings by importance
        sorted_renderings = sorted(
            renderings, 
            key=lambda r: r.metadata.importance.value,
            reverse=True
        )
        
        # Group by compression hint
        no_compression = []
        summarizable = []
        truncatable = []
        omittable = []
        
        for rendering in sorted_renderings:
            if rendering.metadata.compression_hint == CompressionHint.NONE:
                no_compression.append(rendering)
            elif rendering.metadata.compression_hint == CompressionHint.SUMMARIZE:
                summarizable.append(rendering)
            elif rendering.metadata.compression_hint == CompressionHint.TRUNCATE:
                truncatable.append(rendering)
            elif rendering.metadata.compression_hint == CompressionHint.OMIT:
                omittable.append(rendering)
            else:
                # Default to summarizable
                summarizable.append(rendering)
        
        # Start with critical and non-compressible content
        final_parts = []
        
        # Add all critical content regardless of compression hint
        for rendering in sorted_renderings:
            if rendering.metadata.importance == RenderingImportance.CRITICAL:
                final_parts.append(rendering.content)
        
        # Add remaining non-compressible content
        for rendering in no_compression:
            if rendering.metadata.importance != RenderingImportance.CRITICAL:
                final_parts.append(rendering.content)
        
        # Calculate remaining token budget
        current_tokens = sum(self.estimate_tokens(part) for part in final_parts)
        remaining_tokens = self.max_tokens - current_tokens
        
        # Helper to add content if it fits
        def add_if_fits(content: str) -> bool:
            nonlocal remaining_tokens
            tokens = self.estimate_tokens(content)
            if tokens <= remaining_tokens:
                final_parts.append(content)
                remaining_tokens -= tokens
                return True
            return False
        
        # Add high importance content first
        high_importance = [r for r in summarizable + truncatable if r.metadata.importance == RenderingImportance.HIGH]
        for rendering in high_importance:
            add_if_fits(rendering.content)
        
        # Add medium importance content
        medium_importance = [r for r in summarizable + truncatable if r.metadata.importance == RenderingImportance.MEDIUM]
        for rendering in medium_importance:
            if rendering in summarizable:
                if not add_if_fits(rendering.content):
                    # Try to add a summarized version
                    if "summary" in rendering.sections:
                        add_if_fits(rendering.sections["summary"].content)
            elif rendering in truncatable:
                content = rendering.content
                tokens = self.estimate_tokens(content)
                
                if tokens <= remaining_tokens:
                    final_parts.append(content)
                    remaining_tokens -= tokens
                else:
                    # Truncate to fit
                    ratio = remaining_tokens / tokens
                    if ratio > 0.3:  # Only truncate if we can keep at least 30%
                        truncated_length = int(len(content) * ratio)
                        truncated = content[:truncated_length] + "\n...(truncated)..."
                        final_parts.append(truncated)
                        remaining_tokens = 0
        
        # Add low importance content if space remains
        low_importance = [r for r in summarizable + truncatable + omittable if r.metadata.importance == RenderingImportance.LOW]
        for rendering in low_importance:
            if rendering.metadata.compression_hint == CompressionHint.SUMMARIZE:
                # Try summary first
                if "summary" in rendering.sections:
                    if not add_if_fits(rendering.sections["summary"].content):
                        continue
                else:
                    add_if_fits(rendering.content)
            else:
                add_if_fits(rendering.content)
        
        # Add minimal importance content if space remains
        minimal_importance = [r for r in omittable if r.metadata.importance == RenderingImportance.MINIMAL]
        for rendering in minimal_importance:
            add_if_fits(rendering.content)
        
        # Combine parts
        return "\n\n".join(final_parts)


class CategoryCompressor(CompressorBase):
    """
    Compressor that organizes content by category and compresses categories as units.
    """
    
    def compress(self, renderings: List[RenderingResult], context: Optional[RenderingContext] = None) -> str:
        """
        Compress renderings by organizing them into categories.
        
        Args:
            renderings: List of rendering results
            context: Optional rendering context
            
        Returns:
            Compressed context
        """
        # Extract critical content first
        critical_parts = []
        for rendering in renderings:
            if rendering.metadata.importance == RenderingImportance.CRITICAL:
                critical_parts.append(rendering.content)
        
        # Group remaining content by category
        categories: Dict[str, List[RenderingResult]] = {}
        for rendering in renderings:
            if rendering.metadata.importance == RenderingImportance.CRITICAL:
                continue  # Already handled critical content
                
            category = rendering.metadata.element_type
            if category not in categories:
                categories[category] = []
            categories[category].append(rendering)
        
        # Start with critical content
        final_parts = critical_parts
        
        # Calculate remaining token budget
        current_tokens = sum(self.estimate_tokens(part) for part in final_parts)
        remaining_tokens = self.max_tokens - current_tokens
        
        # Process each category
        for category, category_renderings in categories.items():
            # Sort by importance within category
            sorted_renderings = sorted(
                category_renderings, 
                key=lambda r: r.metadata.importance.value,
                reverse=True
            )
            
            # Calculate total tokens for this category
            category_content = [r.content for r in sorted_renderings]
            category_tokens = sum(self.estimate_tokens(content) for content in category_content)
            
            if category_tokens <= remaining_tokens:
                # Add all content from this category
                final_parts.extend(category_content)
                remaining_tokens -= category_tokens
            else:
                # Add header for category
                category_header = f"--- {category.upper()} ---"
                final_parts.append(category_header)
                
                # Try to include most important content first
                for rendering in sorted_renderings:
                    content_tokens = self.estimate_tokens(rendering.content)
                    
                    if content_tokens <= remaining_tokens:
                        # Add in full
                        final_parts.append(rendering.content)
                        remaining_tokens -= content_tokens
                    elif rendering.metadata.compression_hint == CompressionHint.SUMMARIZE and "summary" in rendering.sections:
                        # Try to add summary
                        summary = rendering.sections["summary"].content
                        summary_tokens = self.estimate_tokens(summary)
                        
                        if summary_tokens <= remaining_tokens:
                            final_parts.append(summary)
                            remaining_tokens -= summary_tokens
                    elif rendering.metadata.compression_hint == CompressionHint.TRUNCATE and remaining_tokens > 0:
                        # Truncate to fit
                        ratio = remaining_tokens / content_tokens
                        if ratio > 0.3:  # Only truncate if we can keep at least 30%
                            truncated_length = int(len(rendering.content) * ratio)
                            truncated = rendering.content[:truncated_length] + "\n...(truncated)..."
                            final_parts.append(truncated)
                            remaining_tokens = 0
        
        # Combine parts
        return "\n\n".join(final_parts)


class SimpleCompressor(CompressorBase):
    """
    Simple compressor that just concatenates content until the token limit is reached.
    Useful as a fallback or for simple use cases.
    """
    
    def compress(self, renderings: List[RenderingResult], context: Optional[RenderingContext] = None) -> str:
        """
        Simply compress renderings by concatenating until the token limit is reached.
        
        Args:
            renderings: List of rendering results
            context: Optional rendering context
            
        Returns:
            Compressed context
        """
        # Sort by importance
        sorted_renderings = sorted(
            renderings, 
            key=lambda r: r.metadata.importance.value,
            reverse=True
        )
        
        # Add content until we hit the token limit
        final_parts = []
        total_tokens = 0
        
        for rendering in sorted_renderings:
            content_tokens = self.estimate_tokens(rendering.content)
            
            if total_tokens + content_tokens <= self.max_tokens:
                final_parts.append(rendering.content)
                total_tokens += content_tokens
            else:
                # We've reached the limit
                break
        
        # Combine parts
        return "\n\n".join(final_parts)


# Create some default compressor instances
default_compressor = PriorityCompressor()
category_compressor = CategoryCompressor()
simple_compressor = SimpleCompressor() 