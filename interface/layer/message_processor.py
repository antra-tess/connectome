"""
Message Processor Component

Handles processing of incoming messages and generating responses.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import uuid
import traceback

# Configure logging
logger = logging.getLogger(__name__)


class MessageProcessor:
    """
    Processes messages and generates responses.
    
    This component is responsible for:
    1. Processing incoming messages
    2. Generating responses using the LLM
    3. Executing tool calls from LLM responses
    4. Formatting responses for the user
    """
    
    def __init__(self, environment_manager, context_handler, llm_processor, tool_manager, environment_renderer):
        """
        Initialize the message processor.
        
        Args:
            environment_manager: The environment manager instance
            context_handler: The context handler instance
            llm_processor: The LLM processor instance
            tool_manager: The tool manager instance
            environment_renderer: The environment renderer instance
        """
        self.environment_manager = environment_manager
        self.context_handler = context_handler
        self.llm_processor = llm_processor
        self.tool_manager = tool_manager
        self.environment_renderer = environment_renderer
    
    def process_message(self, user_id: str, message_text: str, message_id: Optional[str] = None, 
                        platform: Optional[str] = None, env_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a message and generate a response.
        
        Args:
            user_id: ID of the user sending the message
            message_text: Content of the message
            message_id: Optional message ID
            platform: Optional platform identifier
            env_id: Optional environment ID where the message is processed
            
        Returns:
            Processing result including the agent's response
        """
        try:
            # Generate a chat ID if not already set
            chat_id = f"chat_{user_id}_{platform}" if platform else f"chat_{user_id}"
            
            # If no specific environment ID is provided, default to messaging environment
            if not env_id:
                env_id = f"messaging_{chat_id}"
            
            # Save the user message to context
            self.context_handler.save_message({
                "env_id": env_id,
                "user_id": user_id,
                "content": message_text,
                "role": "user",
                "platform": platform,
                "message_id": message_id or str(uuid.uuid4())
            })
            
            # 1. Build the context from the context handler
            context = self.context_handler.build_agent_context(env_id)
            
            # 2. Collect available tools separately (tools aren't part of the context)
            available_tools = self._get_available_tools()
            
            # 3. Process with LLM (the LLMProcessor will apply protocol-specific formatting)
            processed = self.llm_processor.process_with_context(context, available_tools)
            
            # Extract the agent's response
            response_text = processed["response"]
            temp_context = []
            
            # Cache the assistant's initial response
            temp_context.append({
                "env_id": env_id,
                "user_id": "assistant",
                "content": response_text,
                "role": "assistant"
            })
            
            # Execute any tool calls
            final_response = None
            
            if processed["tool_calls"]:
                # Execute each tool call in sequence
                for tool_call in processed["tool_calls"]:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    # Check if this is a privileged tool
                    if tool_name in self.tool_manager.privileged_tools:
                        try:
                            # Execute privileged tool
                            result = self.tool_manager.execute_tool(tool_name, **tool_args)
                            
                            # Format the result according to the protocol
                            tool_result = self.llm_processor.format_tool_result(tool_name, result)
                            
                            # Cache the tool result
                            temp_context.append({
                                "env_id": env_id,
                                "content": tool_result,
                                "role": "system",
                                "user_id": "system"
                            })
                            
                        except Exception as e:
                            error_message = f"Error executing tool '{tool_name}': {str(e)}"
                            logger.error(error_message)
                            
                            # Format the error according to the protocol
                            tool_result = self.llm_processor.format_tool_result(tool_name, error_message)
                            
                            # Cache the error message
                            temp_context.append({
                                "env_id": env_id,
                                "content": tool_result,
                                "role": "system",
                                "user_id": "system"
                            })
                    else:
                        # Try to execute environment tool
                        try:
                            result = self.environment_manager.execute_tool(tool_name, **tool_args)
                            
                            # Format the result according to the protocol
                            tool_result = self.llm_processor.format_tool_result(tool_name, result)
                            
                            # Cache the tool result
                            temp_context.append({
                                "env_id": env_id,
                                "content": tool_result,
                                "role": "system",
                                "user_id": "system"
                            })
                        except Exception as e:
                            error_message = f"Error executing environment tool '{tool_name}': {str(e)}"
                            logger.error(error_message)
                            
                            # Format the error according to the protocol
                            tool_result = self.llm_processor.format_tool_result(tool_name, error_message)
                            
                            # Cache the error message
                            temp_context.append({
                                "env_id": env_id,
                                "content": tool_result,
                                "role": "system",
                                "user_id": "system"
                            })
                
                # After executing all tools, generate a final response
                # Create a new context that includes the tool results
                final_context = context + temp_context
                
                # Generate the final response (without passing tools again)
                final_response = self.llm_processor.generate_final_response(final_context)

            # Save the FINAL assistant's response to context (not the initial one)
            self.context_handler.save_message({
                "env_id": env_id,
                "user_id": "assistant",
                "content": final_response or response_text,
                "role": "assistant"
            })
            
            # Return the final response or the original one if no tools were called
            return {
                "status": "success",
                "response": final_response or response_text,
                "chat_id": chat_id,
                "tool_calls": processed["tool_calls"]
            }
            
        except Exception as e:
            error_message = f"Error processing message: {str(e)}"
            logger.error(error_message)
            traceback.print_exc()
            
            return {
                "status": "error",
                "message": error_message
            }
    
    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get descriptions of all available tools.
        
        Returns:
            List of tool descriptions
        """
        # Get privileged tool descriptions
        privileged_tools = self.tool_manager.get_tool_descriptions()
        
        # Get environment tool descriptions
        environment_tools = []
        try:
            for env_id, env in self.environment_manager.get_all_environments().items():
                for tool_name, tool in env.get_tools().items():
                    environment_tools.append({
                        "name": tool_name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameter_descriptions", {})
                    })
        except Exception as e:
            logger.error(f"Error getting environment tools: {str(e)}")
        
        # Combine all tools
        return privileged_tools + environment_tools 