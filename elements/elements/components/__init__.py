"""
Components package
This package contains the components used in the component-based architecture.
"""

# Core components
from .base import Component, ComponentRegistry

# Base/utility components
from .base_representation_component import BaseRepresentationComponent
from .tool_provider_component import ToolProviderComponent
from .veil_producer_component import VeilProducer
from .element_factory_component import ElementFactoryComponent
from .simple_representation_component import SimpleRepresentationComponent
from .text_state_component import TextStateComponent

# Space components
from .space.container_component import ContainerComponent
from .space.timeline_component import TimelineComponent
from .global_attention import GlobalAttentionComponent

# Agent components
from .agent_loop import BaseAgentLoopComponent, SimpleRequestResponseLoopComponent
from .hud_component import HUDComponent
from .context_manager_component import ContextManagerComponent
from .core_tools import CoreToolsComponent

# Messaging components
from .messaging.history_component import HistoryComponent
from .messaging.publisher_component import PublisherComponent
from .messaging.messaging_tools_component import MessagingToolsComponent

# Chat-specific components
from .chat.chat_representation_component import ChatElementRepresentationComponent

# Uplink components
from .uplink.uplink_connection_component import UplinkConnectionComponent
from .uplink.remote_state_cache_component import RemoteStateCacheComponent

__all__ = [
    # Core
    "Component", 
    "ComponentRegistry",
    
    # Base/utility
    "BaseRepresentationComponent", 
    "ToolProviderComponent", 
    "VeilProducer", 
    "ElementFactoryComponent",
    "SimpleRepresentationComponent",
    "TextStateComponent",
    
    # Space
    "ContainerComponent",
    "TimelineComponent",
    "GlobalAttentionComponent",
    
    # Agent
    "BaseAgentLoopComponent",
    "SimpleRequestResponseLoopComponent",
    "HUDComponent",
    "ContextManagerComponent",
    "CoreToolsComponent",
    
    # Messaging
    "HistoryComponent",
    "PublisherComponent",
    "MessagingToolsComponent",
    
    # Chat
    "ChatElementRepresentationComponent",
    
    # Uplink
    "UplinkConnectionComponent",
    "RemoteStateCacheComponent"
] 