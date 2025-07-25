import logging
import requests
import time
import threading
from typing import Any, List, Optional

from llm.provider_interface import (
    LLMMessage,
    LLMToolDefinition
)
from llm.scaffolding_formatter import ScaffoldingFormatter

logger = logging.getLogger(__name__)

class ScaffoldingObserver:
    """Observer implementation for Scaffolding LLMProvider."""

    def __init__(self, web_server_url: str = "http://localhost:6200"):
        """
        Initialize the Scaffolding observer.

        Args:
            web_server_url: URL of the web interface server
            timeout: Timeout in seconds for waiting for LLM response
        """
        self.web_server_url = web_server_url.rstrip('/')
        self.session_id = int(time.time())
        self.formatter = ScaffoldingFormatter()

        logger.info(f"ScaffoldingObserver initialized with web server: {self.web_server_url}")

    def observe_request(self,
                        messages: List[LLMMessage],
                        model: Optional[str] = None,
                        temperature: Optional[float] = None,
                        max_tokens: Optional[int] = None,
                        tools: Optional[List[LLMToolDefinition]] = None,
                        original_context: Optional[List[Any]] = None) -> None:
        logger.info(f"Sending LLM request to web interface: {self.web_server_url}")

        context_data = {
            "messages": self.formatter.format_context(messages, original_context),
            "tools": [self.formatter.format_tool(tool) for tool in (tools or [])],
            "session_id": self.session_id,
            "model": model or "unspecified",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timestamp": time.time(),
            "call_number": getattr(self, '_call_counter', 0) + 1
        }
        self._call_counter = getattr(self, '_call_counter', 0) + 1

        threading.Thread(
            target=self._post_to_scaffolding_server,
            args=(f"{self.web_server_url}/submit-context", context_data),
            daemon=True
        ).start()

    def observe_response(self, response: str) -> None:
        logger.info(f"Sending LLM response to web interface: {self.web_server_url}")

        threading.Thread(
            target=self._post_to_scaffolding_server,
            args=(
                f"{self.web_server_url}/submit-response",
                {"response": response if response else "No response"}
            ),
            daemon=True
        ).start()

    def _post_to_scaffolding_server(self, url, data):
        try:
            requests.post(url, json=data)
        except Exception as e:
            logger.debug(f"Background POST to Scaffolding Server failed: {e}")
