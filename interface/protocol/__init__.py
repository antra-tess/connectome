"""
Protocol Module
Provides access to different tool-usage protocols for the agent.
"""

import importlib
import logging
from typing import Dict, Any

from interface.protocol.base_protocol import BaseProtocol

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


def get_protocol(protocol_name: str) -> BaseProtocol:
    """
    Get a protocol instance by name.
    
    Args:
        protocol_name: Name of the protocol to get
        
    Returns:
        Protocol instance
        
    Raises:
        ValueError: If the protocol is not found
    """
    # If the protocol is already in the registry, return an instance
    if protocol_name in _protocol_registry:
        return _protocol_registry[protocol_name]()
    
    # Try to dynamically import the protocol module
    try:
        # Assume the protocol module is named like 'react_protocol.py'
        module_name = f"{protocol_name}_protocol"
        module_path = f"interface.protocol.{module_name}"
        
        # Import the module
        module = importlib.import_module(module_path)
        
        # Look for a class named like 'ReActProtocol'
        class_name = f"{protocol_name.capitalize()}Protocol"
        if hasattr(module, class_name):
            protocol_class = getattr(module, class_name)
            # Register the protocol for future use
            register_protocol(protocol_name, protocol_class)
            return protocol_class()
        else:
            # Try a different naming convention: Like 'ReactProtocol'
            # Make the first letter of each word after '_' uppercase
            words = protocol_name.split('_')
            capitalized_name = ''.join(word.capitalize() for word in words)
            class_name = f"{capitalized_name}Protocol"
            
            if hasattr(module, class_name):
                protocol_class = getattr(module, class_name)
                register_protocol(protocol_name, protocol_class)
                return protocol_class()
            else:
                raise ValueError(f"Could not find protocol class in module {module_path}")
    except ImportError:
        raise ValueError(f"Protocol module for '{protocol_name}' not found")
    except Exception as e:
        logger.error(f"Error loading protocol '{protocol_name}': {str(e)}")
        raise ValueError(f"Error loading protocol '{protocol_name}': {str(e)}")


def list_available_protocols() -> Dict[str, str]:
    """
    List all available protocols.
    
    Returns:
        Dictionary mapping protocol names to their descriptions
    """
    protocols = {}
    
    # Add registered protocols
    for name in _protocol_registry:
        protocol_class = _protocol_registry[name]
        protocols[name] = protocol_class.__doc__ or "No description available"
    
    # Try to import and list additional protocols
    try:
        # This is a simplified version, in a real implementation you'd scan the directory
        for protocol_name in ["react", "function_calling"]:
            if protocol_name not in protocols:
                try:
                    protocol = get_protocol(protocol_name)
                    protocols[protocol_name] = protocol.__class__.__doc__ or "No description available"
                except ValueError:
                    pass
    except Exception as e:
        logger.error(f"Error listing protocols: {str(e)}")
    
    return protocols

# Import and register built-in protocols
try:
    from interface.protocol.react_protocol import ReactProtocol
    register_protocol("react", ReactProtocol)
except ImportError:
    logger.warning("ReactProtocol not found")

try:
    from interface.protocol.function_calling import FunctionCallingProtocol
    register_protocol("function_calling", FunctionCallingProtocol)
except ImportError:
    logger.warning("FunctionCallingProtocol not found") 