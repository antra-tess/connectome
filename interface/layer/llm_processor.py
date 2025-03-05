"""
LLM Processor Component

Handles interactions with the LLM, including prompt formatting and response processing.
"""

import logging
from typing import Dict, Any, List, Optional
import json
import os
import time
from pathlib import Path

# Remove import from utils.llm and add direct imports
import litellm
from litellm import completion
# Keep config imports and add new ones
from config import LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_API_KEY, LLM_MODEL, LLM_PROVIDER, LLM_BASE_URL

# Configure logging
logger = logging.getLogger(__name__)


class LLMProcessor:
    """
    Handles interactions with the LLM, including prompt formatting and response processing.
    
    This component is responsible for:
    1. Formatting prompts for the LLM
    2. Sending requests to the LLM
    3. Processing LLM responses
    4. Generating summaries and other text content
    """
    
    def __init__(self, protocol, prompt_manager):
        """
        Initialize the LLM processor.
        
        Args:
            protocol: The protocol instance to use
            prompt_manager: The prompt manager instance to use
        """
        self.protocol = protocol
        self.prompt_manager = prompt_manager
        
        # Define the logs directory path
        self.logs_dir = Path("request_logs")
        
        # Initialize LiteLLM
        self.initialize_litellm()
    
    def initialize_litellm(self) -> None:
        """
        Initialize LiteLLM with configuration from environment variables.
        """
        # Set LiteLLM API key
        litellm.api_key = LLM_API_KEY
        
        # Set custom base URL if provided
        if LLM_BASE_URL:
            litellm.api_base = LLM_BASE_URL
        
        # Set any other LiteLLM configurations here
        
        # Ensure the request logs directory exists
        try:
            self.logs_dir.mkdir(exist_ok=True)
            logger.info(f"Debug logs directory initialized at: {self.logs_dir.absolute()}")
        except Exception as e:
            logger.warning(f"Could not create debug logs directory: {str(e)}. Debug logs may not be saved.")
        
        logger.info(f"LiteLLM initialized with model: {LLM_MODEL}, provider: {LLM_PROVIDER}")
    
    def format_for_litellm(self, context: List[Dict[str, Any]], 
                          tool_descriptions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format context and tools for the specific LLM protocol.
        
        Args:
            context: Pre-formatted context from the ContextHandler
            tool_descriptions: Available tools for the LLM
            
        Returns:
            Formatted request for the LLM protocol
        """
        # Initialize parameters for LiteLLM
        params = {
            "messages": context.copy() if context else [],  # Create a copy to avoid modifying the original
            "model": LLM_MODEL,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS
        }
        
        # No tools to format
        if not tool_descriptions or not self.protocol.supports_tools():
            return params
        
        # Get a copy of messages to possibly modify
        messages = params["messages"]
        
        # Get system message if it exists
        system_message = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_message = msg
                system_idx = i
                break
        
        # Use the protocol's format_tools method to get protocol-specific tool formatting
        formatted_tools = self.protocol.format_tools(tool_descriptions)
        
        # Handle based on what format_tools returns
        if isinstance(formatted_tools, list):
            # For protocols that return structured tool data (like OpenAI function calling)
            params["tools"] = formatted_tools
            
            # If we have a system message, enhance it with protocol instructions
            if system_message:
                # Add protocol instructions to system message
                original_content = system_message["content"]
                system_message["content"] = self.protocol.format_system_prompt(original_content, tool_descriptions)
                messages[system_idx] = system_message  # Update the message in place
            
        elif isinstance(formatted_tools, str):
            # For protocols that return textual tool descriptions (like ReAct)
            if system_message:
                # Add protocol instructions and tool descriptions to system message
                original_content = system_message["content"]
                system_message["content"] = self.protocol.format_system_prompt(original_content, tool_descriptions)
                messages[system_idx] = system_message  # Update the message in place
            else:
                # Create a new system message with protocol instructions and tool descriptions
                base_prompt = "You are a helpful AI assistant."
                system_content = self.protocol.format_system_prompt(base_prompt, tool_descriptions)
                messages.insert(0, {"role": "system", "content": system_content})
        
        # Update the messages in the params
        params["messages"] = messages
        
        # Log the request for debugging
        self._log_request_to_file(params)
        
        return params
    
    def send_to_llm(self, formatted_data: Dict[str, Any]) -> str:
        """
        Send a request to the LLM.
        
        Args:
            formatted_data: Formatted data for the LLM
            
        Returns:
            LLM response
        """
        try:
            messages = formatted_data.get("messages", [])
            
            # Set model if not provided
            if "model" not in formatted_data:
                formatted_data["model"] = LLM_MODEL
            
            # Log the request (excluding sensitive data)
            logger.info(f"Sending request to LLM: model={formatted_data.get('model')}, "
                      f"message_count={len(messages)}")
            
            # Call LiteLLM
            start_time = time.time()
            response = completion(**formatted_data)
            end_time = time.time()
            
            # Extract the response content
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    content = response.choices[0].message.content
                else:
                    content = str(response.choices[0])
            else:
                content = str(response)
            
            # Log the response (truncated for brevity)
            logger.info(f"Received response from LLM: {content[:100]}...")
            
            # Use the protocol to extract the final response
            final_response = self.protocol.extract_final_response(content)
            
            # Fallback if protocol couldn't extract a final response
            if final_response is None:
                logger.warning("Protocol couldn't extract final response, using raw response")
                final_response = content
                
            # Log both request and response to file
            response_info = {
                "raw_response": content,
                "final_response": final_response,
                "response_time_seconds": round(end_time - start_time, 2)
            }
            self._log_request_to_file(formatted_data, response_info)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error sending request to LLM: {str(e)}")
            # Still log the failed request
            self._log_request_to_file(formatted_data, {"error": str(e)})
            return f"Error generating response: {str(e)}"
    
    def _log_request_to_file(self, request_data: Dict[str, Any], response_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log the request and response data to a JSON file for debugging purposes.
        
        Args:
            request_data: The formatted request data sent to the LLM
            response_data: Optional response data from the LLM
        """
        try:
            # Use the class-level logs directory path instead of creating it here
            logs_dir = self.logs_dir
            
            # Get existing log files
            existing_logs = sorted([f for f in logs_dir.glob("request_log_*.json")])
            
            # Determine the next log number
            next_num = 1
            if existing_logs:
                # Extract the highest number
                try:
                    last_file = existing_logs[-1].name
                    last_num = int(last_file.split('_')[2].split('.')[0])
                    next_num = last_num + 1
                except (ValueError, IndexError):
                    next_num = len(existing_logs) + 1
            
            # Limit total log files (keep last 100)
            if len(existing_logs) >= 100:
                # Remove the oldest log file
                oldest_file = existing_logs[0]
                oldest_file.unlink(missing_ok=True)
            
            # Get message count and token estimation
            messages = request_data.get("messages", [])
            message_count = len(messages)
            total_content_length = sum(len(m.get("content", "")) for m in messages)
            
            # Create log data with detailed information
            log_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": {
                    "model": request_data.get("model", LLM_MODEL),
                    "provider": LLM_PROVIDER,
                    "message_count": message_count,
                    "estimated_content_length": total_content_length,
                    "temperature": request_data.get("temperature", LLM_TEMPERATURE),
                    "max_tokens": request_data.get("max_tokens", LLM_MAX_TOKENS)
                },
                "request": request_data,
                # Add debug information about the protocol
                "protocol_info": {
                    "name": getattr(self.protocol, "name", str(self.protocol.__class__.__name__)),
                    "description": getattr(self.protocol, "description", "No description available")
                }
            }
            
            # Add response data if available
            if response_data:
                log_data["response"] = response_data
            
            # Write to file
            log_path = logs_dir / f"request_log_{next_num}.json"
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
                
            log_message = f"Logged LLM request to {log_path} ({message_count} messages, ~{total_content_length} chars)"
            if response_data:
                log_message += f" with response"
            logger.info(log_message)
            
        except Exception as e:
            logger.error(f"Error logging request to file: {str(e)}")
            # Don't raise - this is just debugging functionality
    
    def process_llm_output(self, llm_response: str) -> Dict[str, Any]:
        """
        Process the output from the LLM, extracting tool calls and final responses.
        
        Args:
            llm_response: Raw response from the LLM
            
        Returns:
            Processed response with tool calls and final response
        """
        # Extract any tool calls using the protocol
        tool_calls = self.protocol.extract_tool_calls(llm_response)
        
        # Extract the final response using the protocol
        response_text = self.protocol.extract_final_response(llm_response)
        
        # Fallback if protocol couldn't extract a final response
        if response_text is None and not tool_calls:
            response_text = llm_response
        
        return {
            "tool_calls": tool_calls,
            "response_text": response_text
        }
    
    def format_tool_result(self, tool_name: str, result: Any) -> str:
        """
        Format the result of a tool execution.
        
        Args:
            tool_name: Name of the tool that was executed
            result: Result of the tool execution
            
        Returns:
            Formatted tool result
        """
        return self.protocol.format_tool_result(tool_name, result)
    
    def generate_summary(self, prompt: str, max_length: int = 500) -> str:
        """
        Generate a summary using the LLM.
        
        Args:
            prompt: Prompt for the summary
            max_length: Maximum length of the summary
            
        Returns:
            Generated summary
        """
        try:
            # Create a system message instructing the LLM to summarize
            system_message = {
                "role": "system",
                "content": f"Please summarize the following conversation in {max_length} characters or less:"
            }
            
            # Create a user message with the content to summarize
            user_message = {
                "role": "user",
                "content": prompt
            }
            
            # Format for LiteLLM
            formatted_data = self.format_for_litellm([system_message, user_message])
            
            # Send to LLM
            summary = self.send_to_llm(formatted_data)
            
            # Trim if necessary
            if len(summary) > max_length:
                summary = summary[:max_length - 3] + "..."
                
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def process_with_context(self, context: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a request with pre-built context and available tools.
        
        Args:
            context: Formatted context messages from the ContextHandler
            tools: Available tools for the LLM to use
            
        Returns:
            Processed result including response text and any tool calls
        """
        try:
            # Format the context and tools for the specific LLM protocol
            formatted_data = self.format_for_litellm(context, tools)
            
            # Send to the LLM
            llm_response = self.send_to_llm(formatted_data)
            
            # Process the LLM's output
            processed = self.process_llm_output(llm_response)
            
            return {
                "response": processed.get("response_text", ""),
                "tool_calls": processed.get("tool_calls", []),
                "raw_response": llm_response
            }
            
        except Exception as e:
            logger.error(f"Error processing with context: {str(e)}")
            return {
                "response": f"Error processing your request: {str(e)}",
                "tool_calls": [],
                "error": str(e)
            }

    def generate_final_response(self, context: List[Dict[str, Any]]) -> str:
        """
        Generate a final response after tool execution.
        
        Args:
            context: Complete context including tool results
            
        Returns:
            Final response from the LLM
        """
        try:
            # Format the context for the LLM
            formatted_data = self.format_for_litellm(context)
            
            # Send to the LLM
            llm_response = self.send_to_llm(formatted_data)
            
            # Extract the final response using the protocol
            final_response = self.protocol.extract_final_response(llm_response)
            if not final_response:
                # Fallback in case response extraction fails
                processed = self.process_llm_output(llm_response)
                return processed.get("response_text", "I've processed the information, but encountered an issue generating a cohesive response.")
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating final response: {str(e)}")
            return f"Error generating final response: {str(e)}"
