"""
LLM Provider Factory

Factory for creating LLM provider instances based on configuration.
"""

import logging
from typing import Dict, Any, Optional

from llm.provider_interface import LLMProvider


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""

    @staticmethod
    def create_provider(provider_type: str, config: Optional[Dict[str, Any]] = None) -> LLMProvider:
        """
        Create an LLM provider instance based on the specified type.

        Args:
            provider_type: The type of provider to create ("litellm", etc.)
            config: Optional configuration for the provider

        Returns:
            An LLM provider instance

        Raises:
            ValueError: If the provider type is not supported
        """
        logger = logging.getLogger(f"{__name__}.LLMProviderFactory")
        config = config or {}

        logger.info(f"Creating LLM provider of type: {provider_type}")

        if provider_type.lower() == "litellm":
            try:
                from llm.litellm_provider import LiteLLMProvider
                return LiteLLMProvider(**config)
            except ImportError as e:
                logger.error(f"Error importing LiteLLM provider: {e}")
                raise ImportError(
                    "LiteLLM provider requires the litellm package. "
                    "Please install it with: pip install litellm"
                )
        elif provider_type.lower() == "scaffolding":
            from llm.scaffolding_provider import ScaffoldingLLMProvider
            return ScaffoldingLLMProvider(**config)
        else:
            logger.error(f"Unsupported provider type: {provider_type}")
            raise ValueError(f"Unsupported provider type: {provider_type}")

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> LLMProvider:
        """
        Create an LLM provider from a configuration dictionary.

        Args:
            config: Configuration dictionary with at least a "type" field

        Returns:
            An LLM provider instance
        """
        if "type" not in config:
            raise ValueError("Provider configuration must include a 'type' field")

        provider_type = config.pop("type")
        return cls.create_provider(provider_type, config)