"""
Prompt Library
Collection of prompt templates and utilities for the Bot Framework.
"""

# Import key prompts for easy access
from agent.prompt_library.base_prompts import (
    DEFAULT_SYSTEM_PROMPT,
    SAFETY_GUARDRAILS,
    CONVERSATION_GUIDELINES
)

from agent.prompt_library.protocol_prompts import (
    REACT_PROMPT_FORMAT,
    FUNCTION_CALLING_PROMPT_FORMAT,
    format_tool_description
)

from agent.prompt_library.tool_prompts import (
    TOOL_USAGE_GUIDELINES,
    format_tool_list
) 