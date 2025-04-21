"""
Components package
This package contains the components used in the component-based architecture.
"""

from .base import Component, ComponentRegistry
from .base_representation_component import BaseRepresentationComponent
from .tool_provider_component import ToolProvider
from .veil_producer_component import VeilProducer
from .element_factory_component import ElementFactoryComponent
from .global_attention import GlobalAttentionComponent
from .space import ContainerComponent, TimelineComponent
from .messaging import HistoryComponent, PublisherComponent
from .uplink import UplinkConnectionComponent, RemoteStateCacheComponent

# Import specific component types for registration or direct use
from .agent_loop import BaseAgentLoopComponent, SimpleRequestResponseLoopComponent
from .container import ContainerComponent
from .core_tools import CoreToolsComponent
from .hud import HUDComponent
from .simple_representation_component import SimpleRepresentationComponent
from .text_state_component import TextStateComponent
from .tool_provider_component import ToolProviderComponent

# Import chat-specific components
from .chat.chat_representation_component import ChatElementRepresentationComponent

# Import messaging components
from .messaging.messaging_tools_component import MessagingToolsComponent

__all__ = [
    "Component", 
    "ToolProvider", 
    "VeilProducer", 
    "ElementFactoryComponent",
    "GlobalAttentionComponent",
    "ContainerComponent",
    "TimelineComponent",
    "HistoryComponent",
    "PublisherComponent",
    "UplinkConnectionComponent",
    "RemoteStateCacheComponent",
    "BaseAgentLoopComponent",
    "SimpleRequestResponseLoopComponent",
    "CoreToolsComponent",
    "HUDComponent",
    "SimpleRepresentationComponent",
    "TextStateComponent",
    "ToolProviderComponent",
    "ChatElementRepresentationComponent",
    "MessagingToolsComponent",
    "BaseRepresentationComponent"
] 