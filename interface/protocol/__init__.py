"""
Protocol Module
Provides access to different tool-usage protocols for the agent.
"""

import importlib
import inspect
import logging
import os
import sys
from typing import Dict, Any, Optional, Type

from interface.protocol.base_protocol import BaseProtocol
from config import LLM_MODEL, LLM_PROVIDER

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Registry of available protocols
_protocol_registry = {}


def register_protocol(name: str, protocol_class: Any) -> None:
    """
    Register a protocol class with the given name.
    
    Args:
        name: Name of the protocol
        protocol_class: Protocol class to register
    """
    _protocol_registry[name] = protocol_class
    logger.info(f"Registered protocol: {name}")


def discover_protocol_classes() -> Dict[str, Type[BaseProtocol]]:
    """
    Dynamically discover all protocol classes from the protocol directory.
    
    Returns:
        dict: A dictionary mapping class names to protocol classes
    """
    protocol_classes = {}
    protocols_dir = os.path.dirname(__file__)
    
    logger.info(f"Discovering protocol classes from: {protocols_dir}")
    
    # List all Python files in the protocols directory
    try:
        for filename in os.listdir(protocols_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]  # Remove the .py extension
                
                # Skip base module as it's imported explicitly
                if module_name == 'base_protocol':
                    continue
                
                try:
                    # Import the module dynamically
                    module_path = f"interface.protocol.{module_name}"
                    module = importlib.import_module(module_path)
                    
                    # Find all classes in the module that inherit from BaseProtocol
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseProtocol) and 
                            obj is not BaseProtocol):
                            logger.info(f"Discovered protocol class: {name}")
                            protocol_classes[name] = obj
                            
                            # Register the protocol by its conventional name (without the 'Protocol' suffix)
                            if name.endswith('Protocol'):
                                protocol_name = name[:-8].lower()  # Remove 'Protocol' and lowercase
                                register_protocol(protocol_name, obj)
                            
                except Exception as e:
                    logger.error(f"Error importing module {module_name}: {str(e)}")
    except Exception as e:
        logger.error(f"Error scanning protocols directory: {str(e)}")
    
    return protocol_classes


def get_protocol(protocol_name: str) -> BaseProtocol:
    """
    Get a protocol instance by name.
    
    Args:
        protocol_name: Name of the protocol
        
    Returns:
        Protocol instance
    """
    # If the registry is empty, discover protocols first
    if not _protocol_registry:
        discover_protocol_classes()
    
    # Check if the protocol is registered
    if protocol_name not in _protocol_registry:
        # Log the error and fall back to a default protocol
        logger.error(f"Protocol '{protocol_name}' not found. Available protocols: {list(_protocol_registry.keys())}")
        
        # Check if we have any registered protocols to fall back to
        if _protocol_registry:
            default_protocol = next(iter(_protocol_registry.keys()))
            logger.warning(f"Falling back to '{default_protocol}' protocol")
            protocol_class = _protocol_registry[default_protocol]
        else:
            # If no protocols are registered, raise an error
            raise ValueError(f"No protocols available. Protocol '{protocol_name}' not found.")
    else:
        protocol_class = _protocol_registry[protocol_name]
    
    # Create model info dictionary
    model_info = {
        "model": LLM_MODEL,
        "provider": LLM_PROVIDER,
        "has_tool_calling": (LLM_MODEL),
    }
    
    # Create and return an instance of the protocol class
    return protocol_class(model_info=model_info)


def _model_supports_tool_calling(model_name: str) -> bool:
    """
    Check if the specified model supports native function/tool calling.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if the model supports tool calling, False otherwise
    """
    # Most OpenAI models with function calling support
    if any(model in model_name.lower() for model in [
        "gpt-4", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", 
        "gpt-4-0613", "gpt-4-1106-preview", "gpt-4-turbo"
    ]):
        return True
    
    # Anthropic Claude models with function calling support
    if "claude" in model_name.lower() and any(version in model_name.lower() for version in [
        "opus", "sonnet", "3", "3.5"
    ]):
        return True
    
    # Add other models as they become available with function calling
    
    return False


def list_available_protocols() -> Dict[str, str]:
    """
    List all available protocols.
    
    Returns:
        Dictionary mapping protocol names to their descriptions
    """
    protocols = {}
    
    # If the registry is empty, discover protocols first
    if not _protocol_registry:
        discover_protocol_classes()
    
    # Add registered protocols
    for name in _protocol_registry:
        protocol_class = _protocol_registry[name]
        protocols[name] = protocol_class.__doc__ or "No description available"
    
    return protocols


# Discover protocol classes at module load time
discover_protocol_classes()
