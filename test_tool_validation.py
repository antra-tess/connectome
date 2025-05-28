#!/usr/bin/env python3
"""
Test script for LiteLLM tool validation and formatting.
"""

import sys
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm.provider_interface import LLMToolDefinition
from llm.litellm_provider import validate_llm_tool_definition, LiteLLMProvider


def test_valid_tool():
    """Test a valid tool definition."""
    print("=== Testing Valid Tool ===")
    
    tool = LLMToolDefinition(
        name="send_message",
        description="Send a message to a chat channel",
        parameters={
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "The ID of the channel to send to"
                },
                "message": {
                    "type": "string", 
                    "description": "The message content"
                },
                "priority": {
                    "type": "integer",
                    "description": "Message priority (1-5)",
                    "enum": [1, 2, 3, 4, 5]
                }
            },
            "required": ["channel_id", "message"]
        }
    )
    
    # Validate
    issues = validate_llm_tool_definition(tool)
    print(f"Validation issues: {issues}")
    
    # Format for LiteLLM
    try:
        provider = LiteLLMProvider(default_model="gpt-4")
        formatted = provider.format_tool_for_provider(tool)
        print(f"Formatted tool:")
        print(json.dumps(formatted, indent=2))
        return True
    except Exception as e:
        print(f"Error formatting tool: {e}")
        return False


def test_invalid_tool():
    """Test an invalid tool definition."""
    print("\n=== Testing Invalid Tool ===")
    
    # Tool with malformed parameters
    tool = LLMToolDefinition(
        name="bad_tool",
        description="A tool with bad parameters",
        parameters={
            # Missing "type": "object"
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "A parameter"
                }
            },
            "required": ["param1", "nonexistent_param"]  # nonexistent_param not in properties
        }
    )
    
    # Validate
    issues = validate_llm_tool_definition(tool)
    print(f"Validation issues: {issues}")
    
    # Try to format (should handle gracefully)
    try:
        provider = LiteLLMProvider(default_model="gpt-4")
        formatted = provider.format_tool_for_provider(tool)
        print(f"Formatted tool (with fixes):")
        print(json.dumps(formatted, indent=2))
        return True
    except Exception as e:
        print(f"Error formatting tool: {e}")
        return False


def test_tool_from_component_format():
    """Test a tool in the format that comes from ToolProviderComponent."""
    print("\n=== Testing Tool from ToolProviderComponent ===")
    
    # This simulates what ToolProviderComponent.get_llm_tool_definitions() returns
    tool = LLMToolDefinition(
        name="execute_command",
        description="Execute a system command",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute"
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds"
                }
            },
            "required": ["command"]
        }
    )
    
    # Validate
    issues = validate_llm_tool_definition(tool)
    print(f"Validation issues: {issues}")
    
    # Format for LiteLLM
    try:
        provider = LiteLLMProvider(default_model="gpt-4")
        formatted = provider.format_tool_for_provider(tool)
        print(f"Formatted tool:")
        print(json.dumps(formatted, indent=2))
        return True
    except Exception as e:
        print(f"Error formatting tool: {e}")
        return False


if __name__ == "__main__":
    print("Testing LiteLLM Tool Validation and Formatting\n")
    
    success = True
    success &= test_valid_tool()
    success &= test_invalid_tool()
    success &= test_tool_from_component_format()
    
    print(f"\n=== Results ===")
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        sys.exit(1) 