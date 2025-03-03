#!/usr/bin/env python3
"""
Utility script to list all available environment classes.
This can be used by administrators to see what environment classes
are available for configuration.
"""

import os
import sys
import importlib
import inspect

# Add the parent directory to the path so we can import from the bot_framework package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from bot_framework.environments.base import Environment

def list_environment_classes():
    """
    Discover and list all environment classes from the environments directory.
    
    Returns:
        dict: A dictionary mapping class names to environment class info
    """
    env_classes = {}
    environments_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../environments/environment_classes/'))
    
    print(f"Scanning for environment classes in: {environments_dir}")
    
    # List all Python files in the environments directory
    for filename in os.listdir(environments_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]  # Remove the .py extension
            
            try:
                # Import the module dynamically
                module_path = f"bot_framework.environments.{module_name}"
                module = importlib.import_module(module_path)
                
                # Find all classes in the module that inherit from Environment
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, Environment) and 
                        obj is not Environment):
                        # Get class docstring for description
                        description = obj.__doc__ or "No description available"
                        description = description.strip().split('\n')[0]  # First line only
                        
                        env_classes[name] = {
                            "module": module_name,
                            "description": description,
                            "file": filename
                        }
                        
            except Exception as e:
                print(f"Error importing module {module_name}: {str(e)}")
    
    return env_classes

if __name__ == "__main__":
    env_classes = list_environment_classes()
    
    if not env_classes:
        print("No environment classes found.")
    else:
        print("\nAvailable Environment Classes:")
        print("=" * 80)
        print(f"{'Class Name':<30} {'Module':<20} {'Description'}")
        print("-" * 80)
        
        for class_name, info in sorted(env_classes.items()):
            print(f"{class_name:<30} {info['module']:<20} {info['description']}")
        
        print("\nUsage in config.py:")
        print("""
    Example configuration entry:
    {
        "id": "custom_env",
        "class": "YourEnvironmentClass",
        "enabled": true,
        "mount_point": "custom"  # Optional
    }
        """) 