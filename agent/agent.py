"""
Core Agent Implementation
Responsible for processing messages, managing conversation flow, and generating responses.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from config import DEFAULT_PROTOCOL, AGENT_NAME
from context.context_manager import ContextManager
from tools.registry import get_tool_descriptions, execute_tool
from utils.llm import send_to_llm
from agent.prompt_manager import PromptManager
from agent.protocol import get_protocol

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Agent:
    """
    Main agent class that processes messages, calls tools, and generates responses.
    
    Coordinates between the protocol, tools, context management, and the LLM.
    """
    
    def __init__(self, protocol_name: str = DEFAULT_PROTOCOL):
        """
        Initialize the agent with the specified protocol.
        
        Args:
            protocol_name: Name of the protocol to use (defaults to config value)
        """
        self.name = AGENT_NAME
        self.context_manager = ContextManager()
        self.prompt_manager = PromptManager()
        self.protocol = get_protocol(protocol_name)
        self.tools = get_tool_descriptions()
        
        logger.info(f"Agent initialized with protocol: {protocol_name}")
        logger.info(f"Available tools: {[tool['name'] for tool in self.tools]}")
    
    def handle_message(self, message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Main entry point for processing a message.
        
        Args:
            message_data: Dictionary containing message data:
                - chat_id: Identifier for the conversation
                - user_id: Identifier for the user
                - content: Message content
                - platform: Optional platform identifier
                
        Returns:
            Response data if a response should be sent, None otherwise
        """
        logger.info(f"Handling message from user {message_data.get('user_id')} in chat {message_data.get('chat_id')}")
        
        # Save incoming message to context
        self.context_manager.save_message(message_data)
        
        # Build conversation context
        chat_id = message_data.get('chat_id')
        context = self.context_manager.get_context(chat_id)
        
        # Create base prompt with the tools
        base_prompt = self.prompt_manager.get_system_prompt()
        formatted_prompt, additional_params = self.protocol.format_for_litellm(
            base_prompt, context, self.tools
        )
        
        # Send to LLM
        llm_response = send_to_llm(formatted_prompt, additional_params)
        
        # Process the response (potentially multiple rounds of tool calls)
        return self._process_llm_response(llm_response, chat_id)
    
    def _process_llm_response(self, llm_response: str, chat_id: str) -> Optional[Dict[str, Any]]:
        """
        Process the LLM response, handling any tool calls and continuing the conversation
        until a final response is generated.
        
        Args:
            llm_response: Raw LLM response
            chat_id: Identifier for the conversation
            
        Returns:
            Response data if a final response was generated, None otherwise
        """
        # Check for tool calls
        tool_calls = self.protocol.extract_tool_calls(llm_response)
        
        if tool_calls:
            logger.info(f"Found {len(tool_calls)} tool calls in LLM response")
            
            # Execute each tool and collect results
            for tool_call in tool_calls:
                tool_name = tool_call.get('name')
                params = tool_call.get('parameters', {})
                
                logger.info(f"Executing tool: {tool_name} with parameters: {params}")
                
                # Execute the tool
                try:
                    result = execute_tool(tool_name, **params)
                    logger.info(f"Tool {tool_name} execution result: {result}")
                    
                    # Format the result according to the protocol
                    formatted_result = self.protocol.format_tool_result(tool_name, result)
                    
                    # Save the tool call and result to context
                    self.context_manager.save_message({
                        'chat_id': chat_id,
                        'user_id': 'agent',
                        'content': formatted_result,
                        'role': 'tool_result'
                    })
                    
                    # Get updated context
                    context = self.context_manager.get_context(chat_id)
                    
                    # Create new prompt with the updated context
                    base_prompt = self.prompt_manager.get_system_prompt()
                    formatted_prompt, additional_params = self.protocol.format_for_litellm(
                        base_prompt, context, self.tools
                    )
                    
                    # Send to LLM again
                    new_llm_response = send_to_llm(formatted_prompt, additional_params)
                    
                    # Recursively process the new response
                    return self._process_llm_response(new_llm_response, chat_id)
                    
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {str(e)}")
                    error_result = f"Error executing tool {tool_name}: {str(e)}"
                    formatted_result = self.protocol.format_tool_result(tool_name, error_result)
                    
                    # Save the error to context
                    self.context_manager.save_message({
                        'chat_id': chat_id,
                        'user_id': 'agent',
                        'content': formatted_result,
                        'role': 'tool_result'
                    })
        
        # Check for final response
        final_response = self.protocol.extract_final_response(llm_response)
        
        if final_response:
            logger.info(f"Final response generated: {final_response[:100]}...")
            
            # Save the response to context
            self.context_manager.save_message({
                'chat_id': chat_id,
                'user_id': 'agent',
                'content': final_response,
                'role': 'assistant'
            })
            
            # Return the response
            return {
                'chat_id': chat_id,
                'content': final_response
            }
        
        logger.warning("No tool calls or final response found in LLM output")
        return None 