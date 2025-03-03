#!/usr/bin/env python3
"""
Utility script to list all available protocol classes.
This can be used by administrators to see what protocol classes
are available for configuration.
"""

import os
import sys
import importlib
import inspect

# Add the parent directory to the path so we can import from the bot_framework package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from bot_framework.interface.protocol.base_protocol import BaseProtocol


def list_protocol_classes():
    """
    Discover and list all protocol classes from the protocols directory.
    
    Returns:
        dict: A dictionary mapping class names to protocol class info
    """
    protocol_classes = {}
    protocols_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../interface/protocol'))
    
    print(f"Scanning for protocol classes in: {protocols_dir}")
    
    # List all Python files in the protocols directory
    for filename in os.listdir(protocols_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]  # Remove the .py extension
            
            # Skip base module 
            if module_name == 'base_protocol':
                continue
                
            try:
                # Import the module dynamically
                module_path = f"bot_framework.interface.protocol.{module_name}"
                module = importlib.import_module(module_path)
                
                # Find all classes in the module that inherit from BaseProtocol
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseProtocol) and 
                        obj is not BaseProtocol):
                        # Get class docstring for description
                        description = obj.__doc__ or "No description available"
                        description = description.strip().split('\n')[0]  # First line only
                        
                        # Extract protocol name for config (remove 'Protocol' suffix and lowercase)
                        if name.endswith('Protocol'):
                            protocol_name = name[:-8].lower()
                        else:
                            protocol_name = name.lower()
                        
                        protocol_classes[name] = {
                            "module": module_name,
                            "description": description,
                            "file": filename,
                            "config_name": protocol_name
                        }
                        
            except Exception as e:
                print(f"Error importing module {module_name}: {str(e)}")
    
    return protocol_classes


if __name__ == "__main__":
    protocol_classes = list_protocol_classes()
    
    if not protocol_classes:
        print("No protocol classes found.")
    else:
        print("\nAvailable Protocol Classes:")
        print("=" * 80)
        print(f"{'Class Name':<30} {'Config Name':<15} {'Description'}")
        print("-" * 80)
        
        for class_name, info in sorted(protocol_classes.items()):
            print(f"{class_name:<30} {info['config_name']:<15} {info['description']}")
        
        print("\nUsage in config.py:")
        print("""
    Example configuration entry:
    {
        "name": "custom",           # The config_name from above
        "class": "CustomProtocol",  # The class name from above
        "enabled": true
    }
        """) 