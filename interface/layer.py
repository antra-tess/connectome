"""
Interface Layer
Handles the presentation of environments and tools to the agent and provides cognitive processing capabilities.
"""

import logging
import functools
from typing import Dict, Any, Optional, List, Callable, Tuple
from utils.llm import send_to_llm

from interface.context_manager import ContextManager
from config import MAX_HISTORY_MESSAGES, DEFAULT_PROTOCOL, AGENT_NAME, STORAGE_PATH, LLM_TEMPERATURE, LLM_MAX_TOKENS
from utils.llm import send_to_llm
from interface.prompt_manager import PromptManager
from interface.protocol import get_protocol

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InterfaceLayer:
    """
    Presentation and cognitive layer between the user and environments.
    
    The InterfaceLayer is responsible for:
    1. Rendering environments for cognitive processing
    2. Executing tools and managing interactions with the environment system
    3. Providing privileged tools that operate on the cognitive process itself
    4. Processing messages and generating responses using LLM capabilities
    5. Managing conversation flow and context
    
    This merges the functionality of the former Agent class and InterfaceLayer.
    """
    
    def __init__(self, environment_manager, protocol_name: str = DEFAULT_PROTOCOL):
        """
        Initialize the interface layer.
        
        Args:
            environment_manager: Environment manager instance
            protocol_name: Name of the protocol to use (defaults to config value)
        """
        self.environment_manager = environment_manager
        self.name = AGENT_NAME
        self.prompt_manager = PromptManager()
        self.protocol = get_protocol(protocol_name)
        
        # Initialize context manager directly
        self.context_manager = ContextManager(storage_path=STORAGE_PATH)
        
        # Initialize tools registry (privileged tools are registered directly)
        self.privileged_tools: Dict[str, Dict[str, Any]] = {}
        
        # Register privileged tools
        self._register_privileged_tools()
        
        # Accessible tools for the current context (populated during message processing)
        self.tools = []
        
        # Register as an observer for messages from the Environment Layer
        self.environment_manager.register_message_observer(self)
        
        self.messages_context = {}  # Stores message context per user
        
        logger.info(f"Interface layer initialized with protocol: {protocol_name}")
    
    def _register_privileged_tools(self):
        """Register all privileged tools that operate at the interface layer."""
        
        # Context Management Tools
        # Implement these directly using the context_manager
        
        @self.register_privileged_tool(
            name="get_context",
            description="Retrieve the context for a conversation",
            parameter_descriptions={
                "chat_id": "ID of the chat to retrieve context for",
                "max_messages": "Maximum number of messages to retrieve (optional)"
            }
        )
        def get_context(chat_id: str, max_messages: Optional[int] = MAX_HISTORY_MESSAGES) -> Dict[str, Any]:
            """Get conversation context using the context manager."""
            try:
                logger.info(f"[Privileged] Getting context for chat {chat_id}")
                context = self.context_manager.get_context(chat_id, max_messages)
                return {"success": True, "context": context}
            except Exception as e:
                logger.error(f"Error getting context for chat {chat_id}: {str(e)}")
                return {"success": False, "error": str(e)}
        
        @self.register_privileged_tool(
            name="clear_context",
            description="Clear the context for a conversation",
            parameter_descriptions={
                "chat_id": "ID of the chat to clear context for"
            }
        )
        def clear_context(chat_id: str) -> Dict[str, Any]:
            """Clear conversation context using the context manager."""
            try:
                logger.info(f"[Privileged] Clearing context for chat {chat_id}")
                success = self.context_manager.clear_context(chat_id)
                return {"success": success}
            except Exception as e:
                logger.error(f"Error clearing context for chat {chat_id}: {str(e)}")
                return {"success": False, "error": str(e)}
        
        @self.register_privileged_tool(
            name="save_message",
            description="Save a message to the conversation context",
            parameter_descriptions={
                "message_data": "Message data to save"
            }
        )
        def save_message(message_data: Dict[str, Any]) -> Dict[str, Any]:
            """Save a message to context using the context manager."""
            try:
                if "chat_id" not in message_data:
                    logger.error("Missing chat_id in message data")
                    return {"success": False, "error": "Missing chat_id in message data"}
                
                logger.info(f"[Privileged] Saving message to chat {message_data.get('chat_id')}")
                self.context_manager.save_message(message_data)
                return {"success": True}
            except Exception as e:
                logger.error(f"Error saving message: {str(e)}")
                return {"success": False, "error": str(e)}
        
        @self.register_privileged_tool(
            name="summarize_context",
            description="Summarize the conversation context",
            parameter_descriptions={
                "chat_id": "ID of the chat to summarize",
                "max_length": "Maximum length of the summary in characters (optional)",
                "keep_last_messages": "Number of recent messages to keep uncompressed (optional)"
            }
        )
        def summarize_context(chat_id: str, max_length: Optional[int] = 500, 
                           keep_last_messages: Optional[int] = 5) -> Dict[str, Any]:
            """Summarize conversation context using the context manager."""
            try:
                logger.info(f"[Privileged] Summarizing context for chat {chat_id}")
                # This is a placeholder - actual summarization would need to be implemented
                # in the context manager or using the LLM
                context = self.context_manager.get_context(chat_id)
                
                # Simple placeholder implementation - in reality, you'd use more sophisticated
                # summarization techniques, potentially calling the LLM
                summary = f"Conversation with {len(context)} messages"
                if len(context) > keep_last_messages:
                    # Keep only the specified number of recent messages uncompressed
                    compressed_context = context[:-keep_last_messages]
                    recent_context = context[-keep_last_messages:]
                    
                    # In a real implementation, you would compress the older messages
                    # using the LLM or other summarization techniques
                    
                    return {
                        "success": True, 
                        "summary": summary,
                        "compressed_messages": len(compressed_context),
                        "retained_messages": len(recent_context)
                    }
                else:
                    return {
                        "success": True,
                        "summary": summary,
                        "message": "Context too short to require summarization"
                    }
            except Exception as e:
                logger.error(f"Error summarizing context for chat {chat_id}: {str(e)}")
                return {"success": False, "error": str(e)}
        
        # Additional Memory Tools
        @self.register_privileged_tool(
            name="compress_context",
            description="Compress the current conversation context to save space.",
            parameter_descriptions={
                "chat_id": "ID of the chat whose context to compress",
                "summary": "Your summary of the conversation so far",
                "keep_last_messages": "Number of recent messages to keep uncompressed (optional)"
            }
        )
        def compress_context(chat_id: str, summary: str, keep_last_messages: Optional[int] = 5) -> Dict[str, Any]:
            """
            Compress the conversation context by replacing older messages with a summary.
            
            This tool allows the agent to manage its own context window by summarizing
            older parts of the conversation when the context gets too large.
            """
            try:
                logger.info(f"[Privileged] Compressing context for chat {chat_id}")
                
                # Get the current context
                context = self.context_manager.get_context(chat_id)
                
                if len(context) <= keep_last_messages:
                    return {
                        "success": True,
                        "message": "Context too short to require compression"
                    }
                
                # Keep only the specified number of recent messages
                recent_context = context[-keep_last_messages:]
                
                # Create a summary message
                summary_message = {
                    'chat_id': chat_id,
                    'user_id': 'system',
                    'content': f"[SUMMARY: {summary}]",
                    'role': 'system'
                }
                
                # Clear the existing context
                self.context_manager.clear_context(chat_id)
                
                # Save the summary message
                self.context_manager.save_message(summary_message)
                
                # Save the recent messages
                for msg in recent_context:
                    self.context_manager.save_message(msg)
                
                return {
                    "success": True,
                    "message": f"Compressed context successfully. Kept {keep_last_messages} recent messages, replaced {len(context) - keep_last_messages} with summary."
                }
                
            except Exception as e:
                logger.error(f"Error compressing context for chat {chat_id}: {str(e)}")
                return {"success": False, "error": str(e)}
        
        @self.register_privileged_tool(
            name="save_memory",
            description="Save an important piece of information as a long-term memory.",
            parameter_descriptions={
                "chat_id": "ID of the chat this memory belongs to",
                "content": "The content to save as a memory",
                "memory_type": "Type of memory (e.g., 'fact', 'preference', 'instruction')",
                "importance": "Importance rating from 1-10 (optional)"
            }
        )
        def save_memory(chat_id: str, content: str, memory_type: str, importance: Optional[int] = 5) -> Dict[str, Any]:
            """
            Save an important piece of information as a long-term memory.
            
            This allows the agent to explicitly remember important facts, preferences,
            or instructions that should persist across conversations.
            """
            try:
                logger.info(f"[Privileged] Saving memory for chat {chat_id}, type: {memory_type}, importance: {importance}")
                
                # Save as a special memory message
                self.context_manager.save_message({
                    'chat_id': chat_id,
                    'user_id': 'system',
                    'content': content,
                    'role': 'memory',
                    'memory_type': memory_type,
                    'importance': importance
                })
                
                return {
                    "success": True,
                    "message": f"Memory saved successfully: {content[:50]}..." if len(content) > 50 else f"Memory saved successfully: {content}"
                }
                
            except Exception as e:
                logger.error(f"Error saving memory for chat {chat_id}: {str(e)}")
                return {"success": False, "error": str(e)}
        
        @self.register_privileged_tool(
            name="retrieve_memories",
            description="Retrieve memories relevant to a specific topic or query.",
            parameter_descriptions={
                "chat_id": "ID of the chat to retrieve memories from",
                "query": "Search query to find relevant memories",
                "memory_type": "Type of memories to retrieve (optional)",
                "max_results": "Maximum number of memories to retrieve (optional)"
            }
        )
        def retrieve_memories(chat_id: str, query: str, memory_type: Optional[str] = None, max_results: Optional[int] = 5) -> Dict[str, Any]:
            """
            Retrieve memories relevant to a specific topic or query.
            
            This allows the agent to search through saved memories to find
            information relevant to the current conversation.
            """
            try:
                logger.info(f"[Privileged] Retrieving memories for chat {chat_id}, query: {query}, type: {memory_type}")
                
                # Get all messages for the chat
                all_messages = self.context_manager.get_context(chat_id)
                
                # Filter for memory messages
                memories = [msg for msg in all_messages if msg.get('role') == 'memory']
                
                # Filter by memory type if specified
                if memory_type:
                    memories = [msg for msg in memories if msg.get('memory_type') == memory_type]
                
                # Simple keyword search implementation
                # In a real implementation, you might use embeddings or more sophisticated search
                relevant_memories = []
                query_terms = query.lower().split()
                
                for memory in memories:
                    content = memory.get('content', '').lower()
                    # Check if any query term is in the content
                    if any(term in content for term in query_terms):
                        relevant_memories.append(memory)
                
                # Limit results
                relevant_memories = relevant_memories[:max_results]
                
                # Format the results
                result_str = f"Found {len(relevant_memories)} relevant memories:\n\n"
                for i, memory in enumerate(relevant_memories):
                    memory_type = memory.get('memory_type', 'unknown')
                    importance = memory.get('importance', 'unknown')
                    content = memory.get('content', '')
                    result_str += f"{i+1}. [{memory_type}, importance: {importance}] {content}\n\n"
                
                if not relevant_memories:
                    result_str = f"No memories found matching query: {query}"
                
                return {
                    "success": True,
                    "memories": relevant_memories,
                    "result": result_str
                }
                
            except Exception as e:
                logger.error(f"Error retrieving memories for chat {chat_id}: {str(e)}")
                return {"success": False, "error": str(e)}
        
        @self.register_privileged_tool(
            name="analyze_context_needs",
            description="Analyze the current context length and determine if compression is needed.",
            parameter_descriptions={
                "chat_id": "ID of the chat to analyze",
                "threshold_messages": "Threshold for number of messages before compression is recommended (optional)"
            }
        )
        def analyze_context_needs(chat_id: str, threshold_messages: Optional[int] = 20) -> Dict[str, Any]:
            """
            Analyze the current context length and determine if compression is needed.
            
            This tool helps the agent proactively manage its context window by
            providing information about the current context size.
            """
            try:
                logger.info(f"[Privileged] Analyzing context needs for chat {chat_id}")
                
                # Get all messages
                messages = self.context_manager.get_context(chat_id)
                
                # Calculate total content length
                total_chars = sum(len(msg.get('content', '')) for msg in messages)
                
                # Determine if compression is needed
                needs_compression = len(messages) > threshold_messages
                
                return {
                    "success": True,
                    "message_count": len(messages),
                    "total_characters": total_chars,
                    "needs_compression": needs_compression,
                    "threshold": threshold_messages,
                    "recommendation": "Compress the context" if needs_compression else "No compression needed"
                }
                
            except Exception as e:
                logger.error(f"Error analyzing context needs for chat {chat_id}: {str(e)}")
                return {"success": False, "error": str(e)}
        
        @self.register_privileged_tool(
            name="search_conversation_history",
            description="Search through the conversation history for relevant information.",
            parameter_descriptions={
                "chat_id": "ID of the chat to search",
                "query": "The search query to find relevant messages",
                "max_results": "Maximum number of results to return (optional)"
            }
        )
        def search_conversation_history(chat_id: str, query: str, max_results: Optional[int] = 5) -> Dict[str, Any]:
            """
            Search through conversation history for relevant messages.
            
            This tool allows the agent to find specific information from earlier in the conversation.
            """
            try:
                logger.info(f"[Privileged] Searching conversation history for chat {chat_id}, query: {query}")
                
                # Get all messages for the chat
                all_messages = self.context_manager.get_context(chat_id)
                
                # Simple keyword search implementation
                # In a real implementation, you might use embeddings or more sophisticated search
                relevant_messages = []
                query_terms = query.lower().split()
                
                for message in all_messages:
                    content = message.get('content', '').lower()
                    role = message.get('role', '')
                    
                    # Skip system messages
                    if role == 'system':
                        continue
                        
                    # Check if any query term is in the content
                    if any(term in content for term in query_terms):
                        relevant_messages.append(message)
                
                # Limit results
                relevant_messages = relevant_messages[:max_results]
                
                # Format the results
                result_str = f"Found {len(relevant_messages)} relevant messages:\n\n"
                for i, message in enumerate(relevant_messages):
                    user_id = message.get('user_id', 'unknown')
                    role = message.get('role', 'unknown')
                    content = message.get('content', '')
                    
                    # Format the message based on role
                    if role == 'user':
                        result_str += f"{i+1}. [User {user_id}]: {content}\n\n"
                    elif role == 'assistant':
                        result_str += f"{i+1}. [Assistant]: {content}\n\n"
                    else:
                        result_str += f"{i+1}. [{role}]: {content}\n\n"
                
                if not relevant_messages:
                    result_str = f"No messages found matching query: {query}"
                
                return {
                    "success": True,
                    "messages": relevant_messages,
                    "result": result_str
                }
                
            except Exception as e:
                logger.error(f"Error searching conversation history for chat {chat_id}: {str(e)}")
                return {"success": False, "error": str(e)}
        
        @self.register_privileged_tool(
            name="summarize_conversation",
            description="Generate a summary of the current conversation.",
            parameter_descriptions={
                "chat_id": "ID of the chat to summarize",
                "max_length": "Maximum length of the summary in characters (optional)"
            }
        )
        def summarize_conversation(chat_id: str, max_length: Optional[int] = 500) -> Dict[str, Any]:
            """
            Generate a summary of the current conversation.
            
            This tool provides a condensed overview of the conversation so far,
            helping the agent understand the conversation history without
            reviewing all messages individually.
            """
            try:
                logger.info(f"[Privileged] Generating conversation summary for chat {chat_id}")
                
                # Get the conversation messages
                messages = self.context_manager.get_context(chat_id)
                
                if not messages:
                    return {
                        "success": True,
                        "summary": "No conversation history found."
                    }
                
                # Build conversation text for LLM summarization
                conversation_text = ""
                for msg in messages:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    
                    # Skip system messages
                    if role == 'system':
                        continue
                        
                    if role == 'user':
                        conversation_text += f"User: {content}\n"
                    elif role == 'assistant':
                        conversation_text += f"Assistant: {content}\n"
                    else:
                        conversation_text += f"{role.capitalize()}: {content}\n"
                
                # Use LLM to generate the summary
                # In a production system, this would call the actual LLM
                prompt = f"""Summarize the following conversation concisely, focusing on main topics and key information.
                Keep the summary under {max_length} characters.
                
                CONVERSATION:
                {conversation_text}
                
                SUMMARY:
                """
                
                # Call the LLM to generate a summary
                summary = self._generate_summary(prompt, max_length)
                
                return {
                    "success": True,
                    "summary": summary,
                    "message_count": len(messages)
                }
                
            except Exception as e:
                logger.error(f"Error summarizing conversation for chat {chat_id}: {str(e)}")
                return {"success": False, "error": str(e)}
    
    def register_privileged_tool(
        self, 
        name: str, 
        description: str = "", 
        parameter_descriptions: Optional[Dict[str, str]] = None
    ) -> Callable:
        """
        Register a privileged tool with the interface layer.
        
        Args:
            name: Name of the tool
            description: Description of the tool
            parameter_descriptions: Dictionary mapping parameter names to descriptions
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            # Create tool metadata
            tool_info = {
                "function": func,
                "description": description,
                "parameters": {},
                "is_privileged": True
            }
            
            # Add parameter descriptions if provided
            if parameter_descriptions:
                for param_name, param_desc in parameter_descriptions.items():
                    tool_info["parameters"][param_name] = {
                        "description": param_desc
                    }
            
            # Store the tool
            self.privileged_tools[name] = tool_info
            logger.info(f"Registered privileged tool: {name}")
            
            return func
        
        return decorator
        
    def render_for_agent(self) -> Dict[str, Any]:
        """
        Render the current system state for agent processing.
        
        This includes information about mounted environments, available tools,
        privileged tools, and other relevant context.
        
        Returns:
            Dictionary with rendered information
        """
        system_env = self.environment_manager.get_environment("system")
        if not system_env:
            logger.error("System environment not found")
            return {
                "error": "System environment not found"
            }
        
        # Get information about mounted environments
        mounted_environments = []
        for child_id, child in system_env.children.items():
            mount_point = child.metadata.get("mount_point", child.name)
            mounted_environments.append({
                "id": child.id,
                "name": child.name,
                "description": child.description,
                "mount_point": mount_point
            })
        
        # Get available environment tools
        available_tools = []
        for tool_name, tool in system_env.get_all_tools().items():
            available_tools.append({
                "name": tool_name,
                "description": tool.get("description", ""),
                "environment_id": tool.get("environment_id", "unknown"),
                "parameters": tool.get("parameters", {}),
                "is_privileged": False
            })
        
        # Add privileged tools
        tools = []
        for tool_name, tool in self.privileged_tools.items():
            tools.append({
                "name": tool_name,
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
                "is_privileged": True
            })
        
        # Combine all tools into a single list for the agent
        tools.extend(available_tools)
        
        # Build the rendered representation
        return {
            "system": {
                "id": system_env.id,
                "name": system_env.name,
                "description": system_env.description
            },
            "mounted_environments": mounted_environments,
            "tools": tools
        }
    
    def render_environment(self, env_id: str) -> Dict[str, Any]:
        """
        Render a specific environment.
        
        Args:
            env_id: ID of the environment to render
            
        Returns:
            Dictionary with rendered environment information
        """
        env = self.environment_manager.get_environment(env_id)
        if not env:
            logger.error(f"Environment not found: {env_id}")
            return {
                "error": f"Environment not found: {env_id}"
            }
            
        return env.render()
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool.
        
        This will first check if the tool is a privileged tool, and if so,
        execute it directly. Otherwise, it will delegate to the environment
        manager to execute the tool.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Result of the tool execution
            
        Raises:
            ValueError: If the tool is not found
        """
        try:
            # Check if this is a privileged tool
            if tool_name in self.privileged_tools:
                logger.info(f"Executing privileged tool: {tool_name}")
                tool = self.privileged_tools[tool_name]
                function = tool["function"]
                return function(**kwargs)
            
            # Otherwise, delegate to the environment manager
            logger.info(f"Executing environment tool: {tool_name}")
            result = self.environment_manager.execute_tool(tool_name, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            raise
    
    def format_prompt_with_environment(self, base_prompt: str) -> str:
        """
        Format a prompt with information about the current environment.
        
        This method now relies on each environment to provide its own properly
        formatted representation of both structure and state, removing environment-specific
        knowledge from the Interface Layer.
        
        Args:
            base_prompt: Base prompt to format
            
        Returns:
            Formatted prompt with environment information
        """
        # Get the system environment since it manages all environments
        system_env = self.environment_manager.get_environment("system")
        if not system_env:
            logger.error("System environment not found")
            return base_prompt
        
        # Get the formatted structure (capabilities) from the system environment
        structure_section = system_env.render_capabilities_for_context()
        
        # Get formatted state from all environments
        state_section = "\n## Current Environment States\n"
        env_states = self._get_environment_states()
        for env_id, env_state in env_states.items():
            if "formatted_state_text" in env_state:
                env_name = env_state.get("name", env_id)
                state_section += f"\n### {env_name} State\n"
                state_section += f"{env_state['formatted_state_text']}\n"   
        
        # Combine base prompt with environment information
        formatted_prompt = f"{base_prompt}\n\n"
        formatted_prompt += f"{structure_section}\n\n"
        formatted_prompt += f"{state_section}\n"
        
        return formatted_prompt
    
    def get_privileged_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        Get descriptions of all privileged tools.
        
        Returns:
            List of dictionaries with tool descriptions
        """
        tool_descriptions = []
        
        for tool_name, tool in self.privileged_tools.items():
            tool_description = {
                "name": tool_name,
                "description": tool.get("description", ""),
                "parameters": {}
            }
            
            # Add parameter descriptions
            for param_name, param_info in tool.get("parameters", {}).items():
                tool_description["parameters"][param_name] = {
                    "description": param_info.get("description", ""),
                    "required": param_info.get("required", False)
                }
            
            tool_descriptions.append(tool_description)
        
        return tool_descriptions
    
    def observe_message(self, message_data):
        """
        Handle incoming messages from the environment layer.
        
        This method is called by the MessageService when a new message is received.
        
        Args:
            message_data: Dictionary containing message data
            
        Returns:
            None
        """
        user_id = message_data.get('user_id')
        message_text = message_data.get('message_text', '')
        message_id = message_data.get('message_id')
        platform = message_data.get('platform')
        
        logger.info(f"Interface layer received message from user {user_id}")
        
        # Save to context
        if user_id not in self.messages_context:
            self.messages_context[user_id] = []
        
        self.messages_context[user_id].append({
            'role': 'user',
            'content': message_text,
            'message_id': message_id,
            'platform': platform
        })
        
        # Process the message
        self._process_message(user_id, message_text, message_id, platform)
    
    def _process_message(self, user_id, message_text, message_id=None, platform=None):
        """
        Internal method to process a message and generate a response.
        
        This updated version supports tool execution using the protocol.
        
        Args:
            user_id: ID of the user sending the message
            message_text: Content of the message
            message_id: Optional ID for the message
            platform: Optional platform identifier
            
        Returns:
            None (response is sent asynchronously)
        """
        try:
            logger.info(f"Processing message from user {user_id}: {message_text}")
            
            # Build the conversation context
            context = self._build_context(user_id)
            
            # Create the base prompt
            base_prompt = self._create_base_prompt()
            
            # Format the prompt with environment information
            enhanced_prompt = self.format_prompt_with_environment(base_prompt)
            
            # Format for LiteLLM and send
            formatted_data = self.format_for_litellm(context, enhanced_prompt)
            # Import here to avoid circular imports
            llm_response = send_to_llm(formatted_data)
            
            # Extract any tool calls using the protocol
            tool_calls = self.protocol.extract_tool_calls(llm_response)
            
            # If there are tool calls, execute them and continue the conversation
            if tool_calls:
                logger.info(f"Found {len(tool_calls)} tool calls in LLM response")
                
                # Process each tool call
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name")
                    parameters = tool_call.get("parameters", {})
                    
                    logger.info(f"Executing tool: {tool_name}")
                    
                    # Execute the tool
                    try:
                        result = self.execute_tool(tool_name, **parameters)
                        
                        # Format the result according to the protocol
                        tool_result = self.protocol.format_tool_result(tool_name, result)
                        
                        # Add the tool result to the context
                        if user_id in self.messages_context:
                            self.messages_context[user_id].append({
                                "role": "tool_result",
                                "content": tool_result
                            })
                            
                        # Continue the conversation with the tool result
                        # This is recursive to handle multiple tool calls in sequence
                        return self._process_message(user_id, message_text, message_id, platform)
                        
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {str(e)}")
                        error_message = f"Error executing tool {tool_name}: {str(e)}"
                        
                        # Format the error as a tool result
                        tool_result = self.protocol.format_tool_result(tool_name, error_message)
                        
                        # Add the error to the context
                        if user_id in self.messages_context:
                            self.messages_context[user_id].append({
                                "role": "tool_result",
                                "content": tool_result
                            })
                        
                        # Continue the conversation with the error
                        return self._process_message(user_id, message_text, message_id, platform)
            
            # Extract the final response using the protocol
            response_text = self.protocol.extract_final_response(llm_response)
            if not response_text:
                # Fallback if protocol couldn't extract a final response
                response_text = "I'm sorry, I couldn't generate a proper response. Please try again."
            
            # Send the response back through the environment manager
            self.environment_manager.send_response(
                user_id=user_id,
                message_text=response_text,
                message_id=message_id,
                platform=platform
            )
            
            # Save the response to context
            if user_id in self.messages_context:
                self.messages_context[user_id].append({
                    'role': 'assistant',
                    'content': response_text
                })
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            # Send error response
            self.environment_manager.send_response(
                user_id=user_id,
                message_text=f"Sorry, I encountered an error: {str(e)}",
                message_id=message_id,
                platform=platform
            )
    
    def _build_context(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Build the conversation context for a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of dictionaries representing the conversation context
        """
        if user_id not in self.messages_context:
            return []
            
        # Return the messages for this user in a format suitable for LLM
        return self.messages_context[user_id]
    
    def _get_environment_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current state of all environments.
        
        This method collects state information from all environments
        but delegates the actual formatting to each environment.
        
        Returns:
            Dictionary mapping environment IDs to environment state dictionaries
        """
        env_states = {}
        
        # Get states from all environments
        for env_id, env in self.environment_manager.get_all_environments().items():
            try:
                env_states[env_id] = env.render_state_for_context()
            except Exception as e:
                logger.error(f"Error getting state from environment {env_id}: {str(e)}")
                env_states[env_id] = {
                    "error": f"Failed to get state: {str(e)}",
                    "id": env_id,
                    "name": getattr(env, "name", env_id),
                    "formatted_state_text": f"Error: Could not retrieve state for {env_id}"
                }
                
        return env_states
    
    def _create_base_prompt(self) -> str:
        """
        Create the base prompt for the conversation.
        
        Returns:
            String representing the base prompt
        """
        # Use the base prompts from the prompt library instead of hardcoding
        from interface.prompt_library.base_prompts import (
            DEFAULT_SYSTEM_PROMPT,
            CONVERSATION_GUIDELINES,
            SAFETY_GUARDRAILS
        )
        
        # Format the base prompt with agent information
        base_prompt = DEFAULT_SYSTEM_PROMPT.format(
            agent_name=self.name,
            agent_description="an AI assistant that helps users by accessing various environments and tools",
            conversation_guidelines=CONVERSATION_GUIDELINES,
            tool_usage_guidelines="",  # This will be added by the protocol
            safety_guardrails=SAFETY_GUARDRAILS
        )
        
        return base_prompt
    
    def format_for_litellm(self, context: List[Dict[str, Any]], prompt: str) -> Dict[str, Any]:
        """
        Format the prompt for LiteLLM using the selected protocol.
        
        Args:
            context: List of dictionaries representing the conversation context
            prompt: String representing the base prompt
            
        Returns:
            Formatted data ready for sending to LiteLLM
        """
        # Get available tools for the agent
        tools = self.render_for_agent().get("tools", [])
        
        # Use the protocol to format the messages and get additional parameters
        formatted_messages, additional_params = self.protocol.format_for_litellm(
            prompt, context, tools
        )
        
        # Combine with default parameters
        params = {
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            **additional_params
        }
        
        # Return both the formatted messages and parameters
        return {
            "messages": formatted_messages,
            **params
        }
    
    def _send_to_llm(self, formatted_data: Dict[str, Any]) -> str:
        """
        Send the formatted prompt to the LLM and get the response.
        
        Args:
            formatted_data: Dictionary containing formatted prompt data
            
        Returns:
            String representing the response from the LLM
        """
        try:
            # Import here to avoid circular imports
            from utils.llm import send_to_llm
            
            # Send to LLM
            response = send_to_llm(formatted_data)
            
            # Use the protocol to extract the final response
            final_response = self.protocol.extract_final_response(response)
            
            if final_response:
                return final_response
                
            # Fallback if protocol couldn't extract a final response
            if isinstance(response, dict) and "content" in response:
                return response["content"]
            elif isinstance(response, str):
                return response
            else:
                logger.warning(f"Unexpected response format from LLM: {type(response)}")
                return str(response)
                
        except Exception as e:
            logger.error(f"Error sending to LLM: {str(e)}")
            raise
    
    def _generate_summary(self, prompt: str, max_length: int) -> str:
        """
        Helper method to generate a summary using the LLM.
        
        In a production system, this would call the actual LLM service.
        Here we use the prompt_manager to process the request.
        """
        try:
            # This is a simplified approach - in a real implementation,
            # you would use the prompt manager and LLM to generate the summary
            request = {
                "messages": [{"role": "system", "content": prompt}]
            }
            
            # Process with prompt manager
            response = self.prompt_manager.process_prompt(request)
            
            # Extract the summary from the response
            summary = response.get('content', '')
            
            # Ensure it doesn't exceed max_length
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
                
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary with LLM: {str(e)}")
            return f"Error generating summary: {str(e)}" 