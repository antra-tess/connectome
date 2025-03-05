"""
Context Handler Component

Handles the management of conversation context for the interface layer.
"""

import logging
from typing import Dict, Any, List, Optional
import time

from config import CONFIG

# Configure logging
logger = logging.getLogger(__name__)


class ContextHandler:
    """
    Handles conversation context and history.
    
    Responsible for storing, retrieving, and formatting conversation context.
    Acts as the single source of truth for all context-related operations.
    """
    
    def __init__(self, storage_path=None):
        """
        Initialize the context handler.
        
        Args:
            storage_path: Optional path to store context data
        """
        # Initialize message store - format: Dict[env_id, List[message_data]]
        self.message_store = {}
        
        # Initialize metadata store - format: Dict[env_id, Dict[metadata_key, value]]
        self.metadata_store = {}
        
        # Reference to environment manager (set later)
        self.environment_manager = None
        
        # Reference to prompt manager (set later)
        self.prompt_manager = None
        
        # Storage path for persistence (optional)
        self.storage_path = storage_path
        
        logger.info("Context handler initialized")
    
    def set_environment_manager(self, environment_manager):
        """Set the environment manager reference"""
        self.environment_manager = environment_manager
        logger.debug("Environment manager reference set in context handler")
    
    def set_prompt_manager(self, prompt_manager):
        """Set the prompt manager reference"""
        self.prompt_manager = prompt_manager
        logger.debug("Prompt manager reference set in context handler")
    
    def register_privileged_tools(self, tool_manager):
        """
        Register all context management methods as privileged tools.
        
        This method should be called during initialization to make context
        management tools available to the agent.
        
        Args:
            tool_manager: The privileged tool manager instance
        """
        # Register summarize_environment_context
        tool_manager.register_tool(
            name="summarize_environment_context",
            func=self.summarize_environment_context,
            description="Generate a summary of the conversation context in an environment",
            parameter_descriptions={
                "env_id": "ID of the environment to summarize context for",
                "max_length": "Maximum length of the summary in characters (default: 500)",
                "keep_last_messages": "Number of recent messages to keep intact (default: 5)"
            }
        )
        
        # Register analyze_context_needs
        tool_manager.register_tool(
            name="analyze_context_needs",
            func=self.analyze_context_needs,
            description="Analyze if an environment's context needs management operations",
            parameter_descriptions={
                "env_id": "ID of the environment to analyze",
                "threshold_messages": "Message count threshold for suggesting summarization (default: 20)"
            }
        )
        
        # Register search_environment_context
        tool_manager.register_tool(
            name="search_environment_context",
            func=self.search_environment_context,
            description="Search for relevant information across environments",
            parameter_descriptions={
                "query": "Search query to find relevant information",
                "env_id": "Optional environment ID to limit search scope (default: all environments)",
                "max_results": "Maximum number of results to return (default: 5)"
            }
        )
        
        # Register clear_environment_context
        tool_manager.register_tool(
            name="clear_environment_context",
            func=self.clear_environment_context,
            description="Clear the context for a specific environment",
            parameter_descriptions={
                "env_id": "ID of the environment to clear context for"
            }
        )
        
        logger.info("Registered context management privileged tools")
    
    def get_environment_context(self, env_id: str, max_messages: Optional[int] = None) -> Dict[str, Any]:
        """
        Get the context for a specific environment.
        
        Args:
            env_id: ID of the environment to get context for
            max_messages: Maximum number of messages to include
            
        Returns:
            Environment context dictionary
        """
        try:
            if not self.environment_manager:
                return {"error": "Environment manager not set"}
                
            # Get the environment
            environment = self.environment_manager.get_environment(env_id)
            if not environment:
                return {"error": f"Environment {env_id} not found"}
                
            # Get environment state from the environment itself
            env_state = environment.render_state_for_context()
            
            # Get any stored messages for this environment
            messages = self._get_environment_messages(env_id, max_messages)
            
            # Get metadata (may contain summaries, etc.)
            metadata = self._get_environment_metadata(env_id)
            
            # Combine into a context object
            return {
                "env_id": env_id,
                "state": env_state,
                "messages": messages,
                "metadata": metadata,
                "summarized": metadata.get("has_summary", False),
                "total_messages": len(messages)
            }
            
        except Exception as e:
            logger.error(f"Error getting environment context: {str(e)}")
            return {
                "error": f"Failed to retrieve environment context: {str(e)}",
                "env_id": env_id
            }
    
    def clear_environment_context(self, env_id: str) -> Dict[str, Any]:
        """
        Clear the context for a specific environment.
        
        Args:
            env_id: ID of the environment to clear context for
            
        Returns:
            Status dictionary
        """
        try:
            # Clear messages for this environment
            if env_id in self.message_store:
                del self.message_store[env_id]
                
            # Clear metadata for this environment
            if env_id in self.metadata_store:
                del self.metadata_store[env_id]
                
            return {
                "status": "success",
                "message": f"Context for environment {env_id} has been cleared"
            }
                
        except Exception as e:
            logger.error(f"Error clearing environment context: {str(e)}")
            return {
                "status": "error",
                "message": f"Error clearing environment context: {str(e)}"
            }
    
    def save_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save a message and associate it with an environment.
        
        This method associates messages with environments and maintains
        the proper context relationship.
        
        Args:
            message_data: Message data with env_id (or chat_id for backward compatibility)
            
        Returns:
            Status dictionary
        """
        try:
            # Handle both env_id and chat_id for backward compatibility
            env_id = message_data.get("env_id")
            if not env_id:
                # Fall back to chat_id and convert it to environment format
                chat_id = message_data.get("chat_id")
                if chat_id:
                    # For backward compatibility, treat chat_id as messaging environment
                    env_id = f"messaging_{chat_id}"
                else:
                    return {
                        "status": "error",
                        "message": "Missing required env_id or chat_id"
                    }
            
            # Ensure we have an env_id now
            message_data["env_id"] = env_id
            
            # Initialize storage for this environment if needed
            if env_id not in self.message_store:
                self.message_store[env_id] = []
                
            # Add timestamp if not present
            if "timestamp" not in message_data:
                message_data["timestamp"] = time.time()
                
            # Store the message
            self.message_store[env_id].append(message_data)
            
            # Check if we need context management operations
            self._check_summarization_needs(env_id)
            
            return {
                "status": "success",
                "message": "Message saved successfully",
                "env_id": env_id
            }
        
        except Exception as e:
            logger.error(f"Error saving message: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to save message: {str(e)}"
            }
    
    def summarize_environment_context(self, env_id: str, max_length: int = 500, 
                                     keep_last_messages: int = 5) -> Dict[str, Any]:
        """
        Summarize the context for a specific environment.
        
        This is a privileged tool that allows the agent to optimize its own context.
        
        Args:
            env_id: ID of the environment to summarize
            max_length: Maximum length of the summary in characters
            keep_last_messages: Number of recent messages to keep intact
            
        Returns:
            Status dictionary with summary information
        """
        try:
            # Get all messages for this environment
            all_messages = self._get_environment_messages(env_id, max_messages=0)
            
            # If not enough messages to summarize, return early
            if len(all_messages) <= keep_last_messages:
                return {
                    "status": "skipped",
                    "message": "Not enough messages to require summarization",
                    "total_messages": len(all_messages),
                    "env_id": env_id
                }
            
            # Split messages: ones to summarize and ones to keep
            messages_to_summarize = all_messages[:-keep_last_messages] if keep_last_messages > 0 else all_messages
            messages_to_keep = all_messages[-keep_last_messages:] if keep_last_messages > 0 else []
            
            # Format messages for summarization
            formatted_messages = self._format_messages_for_summary(messages_to_summarize)
            
            # Generate summary with LLM
            summary = self._generate_summary_with_llm(formatted_messages, max_length)
            
            # Store the summary in metadata
            metadata = self._get_environment_metadata(env_id) or {}
            metadata["summary"] = summary
            metadata["has_summary"] = True
            metadata["summarized_until"] = len(messages_to_summarize)
            metadata["last_summarized_at"] = time.time()
            self._store_environment_metadata(env_id, metadata)
            
            # Update message store to only keep recent messages
            if keep_last_messages > 0:
                self.message_store[env_id] = messages_to_keep
            
            return {
                "status": "success",
                "message": "Environment context summarized successfully",
                "summary": summary,
                "summarized_messages": len(messages_to_summarize),
                "kept_messages": len(messages_to_keep),
                "env_id": env_id
            }
            
        except Exception as e:
            logger.error(f"Error summarizing environment context: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to summarize environment context: {str(e)}",
                "env_id": env_id
            }
    
    def build_agent_context(self, env_id: Optional[str] = None, max_tokens: int = 8000) -> List[Dict[str, Any]]:
        """
        Build the complete context for the agent in a provider-agnostic format.
        
        Args:
            env_id: Optional environment ID to focus context on
            max_tokens: Maximum tokens to include in context
        
        Returns:
            Formatted context messages ready for LLM
        """
        context = []
        
        # 1. Start with system prompt if prompt manager is available
        # Note: The system prompt now includes environment capabilities via the EnvironmentRenderer
        if self.prompt_manager:
            system_prompt = self.prompt_manager.get_system_prompt()
            context.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 2. Get mounted environments
        mounted_envs = []
        if self.environment_manager:
            system_env = self.environment_manager.get_environment("system")
            if system_env and hasattr(system_env, 'children'):
                mounted_envs = list(system_env.children.keys())
        
        # 4. Add messages from environments (prioritizing specified env_id)
        if env_id and env_id != "system":
            # Focus more context on the specified environment
            env_context = self.get_environment_context(env_id)
            
            # Add recent messages from this environment
            messages = env_context.get("messages", [])
            context.extend(self._format_messages_for_llm(messages))
            
            # Include fewer messages from other environments
            for other_env_id in mounted_envs:
                if other_env_id != env_id:
                    other_env_context = self.get_environment_context(other_env_id, max_messages=5)
                    other_messages = other_env_context.get("messages", [])
                    context.extend(self._format_messages_for_llm(other_messages))
        else:
            # Add recent messages from all environments
            for env_id in mounted_envs:
                env_context = self.get_environment_context(env_id)
                messages = env_context.get("messages", [])
                context.extend(self._format_messages_for_llm(messages))
        
        # 5. Optimize token usage if needed
        return self._optimize_context_tokens(context, max_tokens)
    
    def analyze_context_needs(self, env_id: str, threshold_messages: int = 20) -> Dict[str, Any]:
        """
        Analyze if an environment's context needs management operations.
        
        This is a privileged tool that helps the agent decide when to optimize context.
        
        Args:
            env_id: ID of the environment to analyze
            threshold_messages: Message count threshold for suggesting summarization
            
        Returns:
            Analysis results
        """
        try:
            # Get context information
            env_context = self.get_environment_context(env_id)
            total_messages = env_context.get("total_messages", 0)
            has_summary = env_context.get("summarized", False)
            
            # Analyze token usage (approximate)
            messages = env_context.get("messages", [])
            approximate_tokens = sum(len(str(msg.get("content", ""))) // 4 for msg in messages)
            
            # Make recommendations
            needs_summarization = total_messages > threshold_messages and not has_summary
            approaching_limit = approximate_tokens > (CONFIG.MAX_CONTEXT_TOKENS * 0.8)
            
            return {
                "status": "success",
                "env_id": env_id,
                "total_messages": total_messages,
                "has_summary": has_summary,
                "approximate_tokens": approximate_tokens,
                "needs_summarization": needs_summarization,
                "approaching_context_limit": approaching_limit,
                "recommendations": [
                    "summarize_environment_context" if needs_summarization else None,
                    "clear_older_messages" if approaching_limit and not needs_summarization else None
                ]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing context needs: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to analyze context needs: {str(e)}",
                "env_id": env_id
            }
    
    def search_environment_context(self, query: str, env_id: Optional[str] = None, 
                                 max_results: int = 5) -> Dict[str, Any]:
        """
        Search for relevant information across environments.
        
        This is a privileged tool that allows the agent to find information in its context.
        
        Args:
            query: Search query
            env_id: Optional environment ID to limit search scope
            max_results: Maximum number of results to return
            
        Returns:
            Search results
        """
        try:
            # Determine which environments to search
            env_ids = []
            if env_id:
                # Search in specific environment
                env_ids = [env_id]
            elif self.environment_manager:
                # Search in all environments
                env_ids = list(self.environment_manager.get_environments().keys())
            
            # Collect all matching results across environments
            all_results = []
            
            for eid in env_ids:
                # Get messages from this environment
                messages = self._get_environment_messages(eid, max_messages=0)
                
                # Simple keyword matching (replace with semantic search in production)
                for msg in messages:
                    content = str(msg.get("content", "")).lower()
                    if query.lower() in content:
                        # Add environment ID to the result
                        result = msg.copy()
                        result["env_id"] = eid
                        all_results.append(result)
            
            # Sort by relevance and limit results
            # (Simple version: just sort by recency)
            all_results.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            all_results = all_results[:max_results]
            
            return {
                "status": "success",
                "query": query,
                "results": all_results,
                "total_results": len(all_results),
                "environments_searched": env_ids
            }
            
        except Exception as e:
            logger.error(f"Error searching environment context: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to search environment context: {str(e)}"
            }
    
    # Private helper methods
    def _get_environment_messages(self, env_id: str, max_messages: int = 20) -> List[Dict[str, Any]]:
        """Get messages for a specific environment"""
        messages = self.message_store.get(env_id, [])
        
        # Sort by timestamp
        messages.sort(key=lambda x: x.get("timestamp", 0))
        
        # Limit to max_messages if specified
        if max_messages > 0:
            messages = messages[-max_messages:]
            
        return messages
    
    def _get_environment_metadata(self, env_id: str) -> Dict[str, Any]:
        """Get metadata for a specific environment"""
        return self.metadata_store.get(env_id, {})
    
    def _store_environment_metadata(self, env_id: str, metadata: Dict[str, Any]) -> None:
        """Store metadata for a specific environment"""
        self.metadata_store[env_id] = metadata
    
    def _check_summarization_needs(self, env_id: str) -> None:
        """Check if an environment needs summarization and mark for future summarization if needed"""
        # Get current message count
        messages = self._get_environment_messages(env_id, max_messages=0)
        total_messages = len(messages)
        
        # If above threshold and no summary exists, mark for summarization
        metadata = self._get_environment_metadata(env_id)
        has_summary = metadata.get("has_summary", False)
        
        if total_messages > CONFIG.SUMMARIZATION_THRESHOLD and not has_summary:
            metadata["needs_summarization"] = True
            self._store_environment_metadata(env_id, metadata)
    
    def _format_messages_for_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Format a list of messages for summarization"""
        formatted = []
        for msg in messages:
            user_id = msg.get("user_id", "unknown")
            content = msg.get("content", "")
            role = msg.get("role", "user")
            env_id = msg.get("env_id", "unknown")
            formatted.append(f"[{role.upper()}:{user_id} in {env_id}] {content}")
        
        return "\n".join(formatted)
    
    def _generate_summary_with_llm(self, formatted_text: str, max_length: int) -> str:
        """Generate a summary using an LLM"""
        # This is a placeholder - implement your actual LLM call here
        # In a real implementation, this would call your LLM processor
        return f"This is a placeholder summary of {len(formatted_text)} characters of conversation."
    
    def _format_messages_for_llm(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format messages for LLM consumption in a provider-agnostic format.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted messages for LLM
        """
        formatted_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Convert to standard format
            formatted_messages.append({
                "role": role,
                "content": content
            })
        
        return formatted_messages
    
    def _optimize_context_tokens(self, context: List[Dict[str, Any]], max_tokens: int = 8000) -> List[Dict[str, Any]]:
        """
        Optimize context to fit within token limits.
        
        Args:
            context: The full context list
            max_tokens: Maximum tokens to include
            
        Returns:
            Optimized context list
        """
        # Simple token estimation function (average 4 chars per token)
        def estimate_tokens(text):
            return len(text) / 4
        
        # Always include system messages
        system_messages = [msg for msg in context if msg.get("role") == "system"]
        
        # Get conversation messages (non-system)
        conversation = [msg for msg in context if msg.get("role") != "system"]
        
        # Calculate token usage of system messages
        system_tokens = sum(estimate_tokens(msg.get("content", "")) for msg in system_messages)
        
        # Calculate available tokens for conversation
        available_tokens = max_tokens - system_tokens
        
        # Select messages to fit in available tokens, prioritizing recent messages
        selected_conversation = []
        token_count = 0
        
        for msg in reversed(conversation):
            msg_tokens = estimate_tokens(msg.get("content", ""))
            if token_count + msg_tokens <= available_tokens:
                selected_conversation.insert(0, msg)
                token_count += msg_tokens
            else:
                break
        
        # Combine system messages and selected conversation
        return system_messages + selected_conversation 