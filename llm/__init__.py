"""
LLM Integration Package

This package provides a pluggable interface for integrating Large Language Models
with the bot framework. It abstracts away the details of specific LLM providers,
allowing the framework to switch between different backends without changing the
core architecture.
"""

from bot_framework.llm.provider_interface import (
    LLMProvider,
    LLMMessage,
    LLMToolDefinition,
    LLMToolCall,
    LLMResponse
)
from bot_framework.llm.provider_factory import LLMProviderFactory

# Define the public API
__all__ = [
    "LLMProvider",
    "LLMMessage",
    "LLMToolDefinition",
    "LLMToolCall",
    "LLMResponse",
    "LLMProviderFactory",
] 