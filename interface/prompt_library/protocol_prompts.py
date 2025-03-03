"""
Protocol Prompts
Prompt templates for different tool usage protocols.
"""

from typing import Dict, Any, List

# Custom protocol format (example)
CUSTOM_PROTOCOL_FORMAT = """TOOL USAGE INSTRUCTIONS:
When you need to use a tool, follow this custom format:

<<TOOL: tool_name>>
Parameter1: value1
Parameter2: value2
<</TOOL>>

{tool_descriptions}

The system will execute the tool and provide a result in this format:

<<RESULT>>
[Tool result will appear here]
<</RESULT>>

After using tools as needed, provide your final response to the user.
"""

def format_tool_description(tool: Dict[str, Any]) -> str:
    """
    Format a single tool description for inclusion in a prompt.
    
    Args:
        tool: Tool definition dictionary
        
    Returns:
        Formatted tool description string
    """
    name = tool.get('name', 'unknown_tool')
    description = tool.get('description', 'No description available')
    parameters = tool.get('parameters', {})
    
    formatted_desc = f"Tool: {name}\nDescription: {description}\n"
    
    if parameters:
        formatted_desc += "Parameters:\n"
        for param_name, param_info in parameters.items():
            required = param_info.get('required', False)
            param_desc = param_info.get('description', 'No description')
            
            req_tag = "(required)" if required else "(optional)"
            formatted_desc += f"  - {param_name} {req_tag}: {param_desc}\n"
    
    return formatted_desc
