"""
Tool Prompts
Prompt components related to tool usage and tool-specific instructions.
"""

from typing import Dict, Any, List

# General tool usage guidelines
TOOL_USAGE_GUIDELINES = """TOOL USAGE GUIDELINES:
1. Use tools when you need information or capabilities beyond your knowledge.
2. Select the most appropriate tool for each task.
3. Provide all required parameters when calling a tool.
4. If a tool fails, try different parameters or an alternative approach.
5. When using tools that access external resources, verify the information is useful and relevant.
6. Before using tools, consider if you already know the answer.
7. After using a tool, explain the results clearly to the user.
"""

# Category-specific tool instructions
WEB_TOOL_INSTRUCTIONS = """WEB TOOL GUIDELINES:
- When searching for information, use specific queries.
- Verify the reliability of web content before presenting it to users.
- Cite sources when providing information from web searches.
- Be aware of potential outdated information when fetching web content.
- For API requests, handle errors gracefully and explain any issues to users.
"""

FILE_TOOL_INSTRUCTIONS = """FILE TOOL GUIDELINES:
- Only access files within allowed directories.
- When reading files, process information systematically.
- When writing to files, confirm actions with users first.
- Be careful when deleting or modifying user files.
- Respect file permissions and restrictions.
"""

MEMORY_TOOL_INSTRUCTIONS = """MEMORY TOOL GUIDELINES:
- Save important information that may be needed across conversations.
- Organize memories by type (facts, preferences, instructions).
- Use appropriate importance ratings for saved memories.
- Compress context when it gets too large to maintain performance.
- Retrieve specific memories when they become relevant to the conversation.
"""

def format_tool_list(tools: List[Dict[str, Any]]) -> str:
    """
    Format a simple list of available tools and their descriptions.
    
    Args:
        tools: List of tool definitions
        
    Returns:
        Formatted tool list as a string
    """
    if not tools:
        return "No tools available."
    
    lines = ["Available tools:"]
    
    for tool in tools:
        name = tool.get('name', 'unknown_tool')
        description = tool.get('description', 'No description available')
        lines.append(f"- {name}: {description}")
    
    return "\n".join(lines)

def get_tool_examples_by_category() -> Dict[str, str]:
    """
    Get example tool usage for each tool category.
    
    Returns:
        Dictionary mapping tool categories to example usage
    """
    return {
        "chat": """Example of chat tool usage:
Thought: I need to respond to the user's message, but I want to be clear that it's specifically addressed to John.
Action: send_private_message(user_id="john123", message="Hi John, regarding your question about project timelines...")
Observation: Private message to user john123 will be delivered: Hi John, regarding your question about project timelines...
""",
        
        "web": """Example of web tool usage:
Thought: The user wants information about climate change impacts. I should search for recent data.
Action: web_search(query="recent climate change impacts 2023", num_results=3)
Observation: [Search results about climate change]
""",
        
        "memory": """Example of memory tool usage:
Thought: I notice the context is getting quite long. I should check if compression is needed.
Action: analyze_context_needs(chat_id="chat123", threshold_messages=15)
Observation: {"message_count": 23, "total_characters": 5840, "threshold": 15, "needs_compression": true, "recommendation": "Context compression is recommended to maintain efficient operation."}
""",
        
        "file": """Example of file tool usage:
Thought: The user wants to know what files are available in the data directory.
Action: list_files(directory="data", pattern="*.csv")
Observation: ["data/file1.csv", "data/report.csv", "data/metrics.csv"]
"""
    }

def format_tool_category_examples(category: str) -> str:
    """
    Get formatted example tool usage for a specific category.
    
    Args:
        category: Tool category name
        
    Returns:
        Formatted tool usage examples for the category
    """
    examples = get_tool_examples_by_category()
    return examples.get(category.lower(), "No examples available for this category.") 