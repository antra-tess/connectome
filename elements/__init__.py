"""
Elements Module
Defines the hierarchical element system for the Bot Framework.
"""

from bot_framework.elements.elements.base import BaseElement
from bot_framework.elements.elements.space import Space
from bot_framework.elements.elements.inner_space import InnerSpace
from bot_framework.elements.space_registry import SpaceRegistry
from .elements.messaging import ChatElement
from .elements.uplink import UplinkProxy

__all__ = [
    'BaseElement',
    'Space',
    'InnerSpace',
    'SpaceRegistry'
]

def create_direct_chat_element(element_id, name, description, platform, adapter_id, registry=None):
    """
    Create a ChatElement for direct inclusion in InnerSpace (Model 1).
    
    Args:
        element_id: Unique identifier for the element
        name: Human-readable name
        description: Description of the element's purpose
        platform: Platform identifier (e.g., 'telegram')
        adapter_id: ID of the adapter this element is associated with
        registry: Optional SpaceRegistry to register with
        
    Returns:
        Configured ChatElement instance
    """
    chat = ChatElement(element_id, name, description, platform, adapter_id)
    # Mark as direct (not remote)
    chat.set_as_remote(False)
    
    # Register if registry provided
    if registry and isinstance(registry, SpaceRegistry):
        registry.register_element(chat)
        
    return chat

def create_uplinked_chat_setup(inner_space, registry, platform="telegram", adapter_id="telegram_adapter"):
    """
    Create a shared space with a chat element and an uplink in the inner space (Model 2).
    
    Args:
        inner_space: The agent's inner space
        registry: SpaceRegistry to register elements with
        platform: Platform identifier (e.g., 'telegram')
        adapter_id: ID of the adapter this element is associated with
        
    Returns:
        Tuple of (shared_space, uplink, chat_element)
    """
    # Create a shared space for remote chats
    shared_space = Space("shared_space", "Shared Chat Space", "Space containing shared chat elements")
    registry.register_space(shared_space)
    
    # Create chat element in the shared space
    chat_element = ChatElement(
        f"{platform}_chat", 
        f"{platform.capitalize()} Chat", 
        f"Interface for {platform} messages",
        platform=platform,
        adapter_id=adapter_id
    )
    # Add to shared space
    shared_space.mount_element(chat_element)
    
    # Create uplink in the inner space
    uplink = UplinkProxy(
        f"{platform}_uplink", 
        f"{platform.capitalize()} Chat Uplink", 
        f"Connection to shared {platform} chat space",
        remote_space_id="shared_space"
    )
    
    # Mount the uplink in the inner space
    inner_space.mount_element(uplink)
    
    return (shared_space, uplink, chat_element)