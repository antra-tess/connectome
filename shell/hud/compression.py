"""
HUD Compression

Provides compression strategies for the HUD to reduce context size while
preserving essential information.
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
    
    def compress(self, renderings: List[RenderingResult], context: RenderingContext) -> str:
        """
        Compress the renderings to fit within token limits.
        
        Args:
            renderings: List of rendering results
            context: Rendering context
            
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
    
    def compress(self, renderings: List[RenderingResult], context: RenderingContext) -> str:
        """
        Compress renderings by prioritizing content based on importance.
        
        Args:
            renderings: List of rendering results
            context: Rendering context
            
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


class SummaryCompressor(CompressorBase):
    """
    Compressor that generates summaries for content that exceeds token limits.
    """
    
    def __init__(self, max_tokens: int = 8000, summarizer=None):
        """
        Initialize the summary compressor.
        
        Args:
            max_tokens: Maximum number of tokens allowed
            summarizer: Optional summarizer function
        """
        super().__init__(max_tokens)
        self.summarizer = summarizer
    
    def compress(self, renderings: List[RenderingResult], context: RenderingContext) -> str:
        """
        Compress renderings by summarizing content that exceeds token limits.
        
        Args:
            renderings: List of rendering results
            context: Rendering context
            
        Returns:
            Compressed context
        """
        # Start with critical content
        final_parts = []
        critical_renderings = [r for r in renderings if r.metadata.importance == RenderingImportance.CRITICAL]
        for rendering in critical_renderings:
            final_parts.append(rendering.content)
        
        # Calculate remaining token budget
        current_tokens = sum(self.estimate_tokens(part) for part in final_parts)
        remaining_tokens = self.max_tokens - current_tokens
        
        # Sort remaining renderings by importance
        remaining_renderings = [r for r in renderings if r.metadata.importance != RenderingImportance.CRITICAL]
        sorted_renderings = sorted(
            remaining_renderings, 
            key=lambda r: r.metadata.importance.value,
            reverse=True
        )
        
        # Group content by category
        categories: Dict[str, List[RenderingResult]] = {}
        for rendering in sorted_renderings:
            category = rendering.metadata.element_type
            if category not in categories:
                categories[category] = []
            categories[category].append(rendering)
        
        # Process each category
        for category, category_renderings in categories.items():
            # Calculate total tokens for this category
            category_tokens = sum(self.estimate_tokens(r.content) for r in category_renderings)
            
            if category_tokens <= remaining_tokens:
                # Add all content in this category
                for rendering in category_renderings:
                    final_parts.append(rendering.content)
                remaining_tokens -= category_tokens
            else:
                # Try to summarize this category
                if self.summarizer:
                    # Generate summary
                    category_content = "\n\n".join(r.content for r in category_renderings)
                    summary = self.summarizer(category_content, max_tokens=remaining_tokens // 2)
                    
                    # Add summary
                    final_parts.append(f"Summary of {category}:\n{summary}")
                    remaining_tokens -= self.estimate_tokens(summary)
                else:
                    # No summarizer, use truncation
                    for rendering in category_renderings:
                        if self.estimate_tokens(rendering.content) <= remaining_tokens:
                            final_parts.append(rendering.content)
                            remaining_tokens -= self.estimate_tokens(rendering.content)
                        else:
                            break
        
        # Combine parts
        return "\n\n".join(final_parts)


class SmartCompressor(CompressorBase):
    """
    Smart compressor that uses a combination of strategies.
    """
    
    def __init__(self, max_tokens: int = 8000, compression_ratio: float = 0.8, summarizer=None):
        """
        Initialize the smart compressor.
        
        Args:
            max_tokens: Maximum number of tokens allowed
            compression_ratio: Target compression ratio (0 to 1)
            summarizer: Optional summarizer function
        """
        super().__init__(max_tokens)
        self.compression_ratio = compression_ratio
        self.summarizer = summarizer
        self.priority_compressor = PriorityCompressor(max_tokens)
        self.summary_compressor = SummaryCompressor(max_tokens, summarizer)
    
    def compress(self, renderings: List[RenderingResult], context: RenderingContext) -> str:
        """
        Compress renderings using smart strategies.
        
        Args:
            renderings: List of rendering results
            context: Rendering context
            
        Returns:
            Compressed context
        """
        # Calculate total tokens
        total_tokens = sum(self.estimate_tokens(r.content) for r in renderings)
        
        # If total tokens are within limit, use priority compressor
        if total_tokens <= self.max_tokens:
            return self.priority_compressor.compress(renderings, context)
        
        # If compression ratio is high, use summary compressor
        compression_needed = total_tokens / self.max_tokens
        if compression_needed > 2.0 and self.summarizer:
            return self.summary_compressor.compress(renderings, context)
        
        # Otherwise use priority compressor
        return self.priority_compressor.compress(renderings, context)


# Create a default compressor instance
default_compressor = PriorityCompressor() 