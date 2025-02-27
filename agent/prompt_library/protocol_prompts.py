"""
Protocol Prompts
Prompt templates for different tool usage protocols.
"""

from typing import Dict, Any, List

# ReAct pattern protocol format
REACT_PROMPT_FORMAT = """TOOL USAGE INSTRUCTIONS:
You have access to the following tools to assist users:

{tool_descriptions}

To use these tools, follow this format exactly:

Thought: Think step-by-step about how to respond to the user's request and whether any tools can help.

Action: tool_name(param1=value1, param2=value2, ...)
[Use this format to call a tool. Only call one tool at a time. Provide parameter values exactly as required.]

Observation: [Tool output will appear here]

Continue this pattern of Thought, Action, Observation until you have all the information needed.

Final Answer: [Your final response to the user's request]

IMPORTANT:
- Always start with a Thought.
- Always end with a Final Answer after you have the information you need.
- If you don't need to use a tool, provide your response directly as a Final Answer.
- Call tools exactly as specified with their correct parameters.
"""

# Function calling protocol format (for models with native function calling)
FUNCTION_CALLING_PROMPT_FORMAT = """TOOL USAGE INSTRUCTIONS:
You have access to various tools to assist users. The system will handle the formatting of tool calls.

{tool_descriptions}

When you need to use a tool:
1. Clearly indicate that you need to call a specific tool
2. The system will execute the tool and provide the result
3. Use the tool result to inform your final response

IMPORTANT:
- Only call tools when necessary to fulfill the user's request
- You can call multiple tools if needed
- Make sure to provide a final response that addresses the user's request
"""

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

def format_react_system_prompt(base_prompt: str, tools: List[Dict[str, Any]]) -> str:
    """
    Format a complete system prompt with ReAct tool instructions.
    
    Args:
        base_prompt: The base system prompt
        tools: List of available tools
        
    Returns:
        Complete formatted system prompt
    """
    tool_descriptions = "\n\n".join([format_tool_description(tool) for tool in tools])
    
    react_instructions = REACT_PROMPT_FORMAT.format(tool_descriptions=tool_descriptions)
    
    return f"{base_prompt}\n\n{react_instructions}"

def format_function_calling_system_prompt(base_prompt: str, tools: List[Dict[str, Any]]) -> str:
    """
    Format a complete system prompt with function calling tool instructions.
    
    Args:
        base_prompt: The base system prompt
        tools: List of available tools
        
    Returns:
        Complete formatted system prompt
    """
    tool_descriptions = "\n\n".join([format_tool_description(tool) for tool in tools])
    
    function_calling_instructions = FUNCTION_CALLING_PROMPT_FORMAT.format(
        tool_descriptions=tool_descriptions
    )
    
    return f"{base_prompt}\n\n{function_calling_instructions}" 