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
    Interface layer component responsible for agent's cognitive context management.
    
    This class handles higher-level context operations such as:
    1. Building composite context from multiple environments
    2. Summarizing and analyzing context needs
    3. Managing context window limits
    4. Providing privileged context management tools
    
    This component is environment-aware and builds context based on the
    environment hierarchy (system, shared, remote environments).
    """
    
    def __init__(self, storage_path=None):
        """
        Initialize the context handler.
        
        Args:
            storage_path: Optional path for persistent storage (if needed)
        """
        self.storage_path = storage_path
        self.environment_manager = None
        self.message_store = {}  # In-memory message store
        self.metadata_store = {}  # In-memory metadata store
        
    def set_environment_manager(self, environment_manager):
        """Set the environment manager reference for environment context collection"""
        self.environment_manager = environment_manager
    
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
    
    def build_agent_context(self) -> List[Dict[str, Any]]:
        """
        Build a comprehensive context for the agent across all environments.
        
        This method:
        1. Collects system environment state
        2. Gathers context from all relevant mounted environments
        3. Organizes it in a structured way for the agent
        
        Returns:
            List of context elements in priority order
        """
        try:
            if not self.environment_manager:
                return [{"type": "error", "content": "Environment manager not set"}]
                
            context_elements = []
            
            # 1. Add system instructions and agent identity
            context_elements.append({
                "type": "system",
                "content": self._get_system_instructions()
            })
            
            # 2. Get the system environment (container for all environments)
            system_env = self.environment_manager.get_environment("system")
            if not system_env:
                return [{"type": "error", "content": "System environment not found"}]
                
            # 3. Add the system environment state
            sys_env_state = system_env.render_state_for_context()
            context_elements.append({
                "type": "system_environment",
                "content": sys_env_state
            })
            
            # 4. Get all mounted environments from the system environment
            mounted_envs = system_env.get_mounted_environments()
            
            # 5. For each mounted environment, add its context
            for env_id in mounted_envs:
                env_context = self.get_environment_context(env_id)
                
                # Add summary if available
                if env_context.get("summarized"):
                    metadata = env_context.get("metadata", {})
                    if "summary" in metadata:
                        context_elements.append({
                            "type": "environment_summary",
                            "env_id": env_id,
                            "content": metadata["summary"]
                        })
                
                # Add recent messages from this environment
                messages = env_context.get("messages", [])
                if messages:
                    context_elements.append({
                        "type": "environment_messages",
                        "env_id": env_id,
                        "content": messages
                    })
                    
                # Add environment state
                state = env_context.get("state")
                if state:
                    context_elements.append({
                        "type": "environment_state",
                        "env_id": env_id,
                        "content": state
                    })
            
            return context_elements
            
        except Exception as e:
            logger.error(f"Error building agent context: {str(e)}")
            return [{
                "type": "error",
                "content": f"Failed to build agent context: {str(e)}"
            }]
    
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
    
    def _get_system_instructions(self) -> str:
        """Get the system instructions for the agent"""
        # This is a placeholder - implement your actual system instructions
        return "You are a helpful AI assistant with access to multiple environments."
    
    def _sanitize_env_id(self, env_id: str) -> str:
        """Sanitize an environment ID to ensure it's safe to use as a key"""
        return env_id.replace(" ", "_").replace("/", "_").replace("\\", "_") 