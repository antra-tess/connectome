"""
Scaffolding LLM Provider

A test LLM provider that redirects LLM calls to a web interface for manual response input.
This enables cost-free testing, debugging, and validation of the complete agent loop.
"""

import logging
import time
import json
import requests
from typing import Dict, Any, List, Optional, Union

from llm.provider_interface import LLMProvider, LLMMessage, LLMResponse, LLMToolDefinition
from .scaffolding_formatter import ScaffoldingFormatter

logger = logging.getLogger(__name__)


class ScaffoldingLLMProvider(LLMProvider):
    """
    Scaffolding LLM provider that redirects to web interface for manual testing.

    This provider intercepts LLM calls and sends them to a web interface where
    a human can manually provide responses, enabling:
    - Cost-free testing without actual LLM API calls
    - Full visibility into agent context and responses
    - Debugging of agent loop behavior
    - Edge case testing with specific response scenarios
    """

    def __init__(self, web_server_url: str = "http://localhost:6200", timeout: int = 300, **kwargs):
        """
        Initialize the scaffolding provider.

        Args:
            web_server_url: URL of the web interface server
            timeout: Timeout in seconds for waiting for manual response
            **kwargs: Additional configuration (ignored)
        """
        self.web_server_url = web_server_url.rstrip('/')
        self.timeout = timeout
        self.session_id = int(time.time())
        self.formatter = ScaffoldingFormatter()

        logger.info(f"ScaffoldingLLMProvider initialized with web server: {self.web_server_url}")

    def complete(self,
                 messages: List[LLMMessage],
                 model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 tools: Optional[List[LLMToolDefinition]] = None,
                 **kwargs) -> LLMResponse:
        """
        Send messages to web interface and get manual response.

        Args:
            messages: List of conversation messages
            model: Model name (for display only)
            temperature: Temperature setting (for display only)
            max_tokens: Max tokens setting (for display only)
            tools: Available tools (for display only)
            **kwargs: Additional parameters

        Returns:
            LLMResponse with manual input as content
        """
        try:
            logger.info(f"Intercepting LLM call with {len(messages)} messages")

            context_data = {
                "messages": self.formatter.format_context(messages),
                "tools": [self.formatter.format_tool(tool) for tool in (tools or [])],
                "session_id": self.session_id,
                "model": model or "unspecified",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timestamp": time.time(),
                "call_number": getattr(self, '_call_counter', 0) + 1
            }

            # Increment call counter
            self._call_counter = getattr(self, '_call_counter', 0) + 1

            # Send context to web interface and wait for response
            logger.info(f"Sending context to web interface: {self.web_server_url}")
            response = requests.post(
                f"{self.web_server_url}/submit-context",
                json=context_data,
                timeout=self.timeout
            )

            if response.status_code != 200:
                logger.error(f"Web server returned status {response.status_code}: {response.text}")
                return LLMResponse(
                    content=f"Error: Web server returned status {response.status_code}",
                    finish_reason="error"
                )

            response_data = response.json()
            manual_response = response_data.get("manual_response", "No response provided")

            logger.info(f"Received manual response: {len(manual_response)} characters")

            return LLMResponse(
                content=manual_response,
                finish_reason="stop",
                usage={"prompt_tokens": 0, "completion_tokens": len(manual_response.split()), "total_tokens": 0}
            )

        except requests.exceptions.Timeout:
            logger.error(f"Timeout waiting for response from web interface after {self.timeout}s")
            return LLMResponse(
                content="Error: Timeout waiting for manual response",
                finish_reason="error"
            )
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to web interface at {self.web_server_url}")
            return LLMResponse(
                content=f"Error: Cannot connect to scaffolding web interface at {self.web_server_url}",
                finish_reason="error"
            )
        except Exception as e:
            logger.error(f"Error in scaffolding provider: {e}", exc_info=True)
            return LLMResponse(
                content=f"Error: {str(e)}",
                finish_reason="error"
            )

    def format_tool_for_provider(self, tool: Dict[str, Any]) -> Any:
        """
        Not needed for scaffolding - tools are displayed in web interface.

        Args:
            tool: Tool definition

        Returns:
            The tool unchanged
        """
        return tool

    def parse_response(self, raw_response: Any) -> LLMResponse:
        """
        Not needed for scaffolding - responses come from web interface.

        Args:
            raw_response: Raw response

        Returns:
            The response unchanged
        """
        if isinstance(raw_response, LLMResponse):
            return raw_response
        else:
            return LLMResponse(content=str(raw_response), finish_reason="stop")
