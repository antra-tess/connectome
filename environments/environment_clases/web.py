"""
Web Environment
Defines an environment for web-related tools and functionality.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List
import requests
from urllib.parse import urlparse

from environments.base import Environment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WebEnvironment(Environment):
    """
    Environment for web-related tools and functionality.
    
    Provides tools for web search, fetching web pages, and making HTTP requests.
    """
    
    def __init__(self, env_id: str = "web", name: str = "Web Environment", 
                description: str = "Environment for interacting with web content"):
        """
        Initialize the web environment.
        
        Args:
            env_id: Unique identifier for this environment
            name: Human-readable name for this environment
            description: Detailed description of this environment
        """
        super().__init__(env_id=env_id, name=name, description=description)
        # Track recently accessed URLs and searches
        self._recent_searches = []
        self._recent_urls = []
        self._register_web_tools()
        
    def _register_web_tools(self):
        """Register web-related tools."""
        # Web search tool
        self.register_tool(
            self.web_search,
            name="web_search",
            description="Search the web for information on a topic",
            parameter_descriptions={
                "query": "The search query to find information about",
                "num_results": "Number of search results to return"
            }
        )
        
        # Fetch webpage tool
        self.register_tool(
            self.fetch_webpage,
            name="fetch_webpage",
            description="Fetch the content of a webpage and convert it to a readable format",
            parameter_descriptions={
                "url": "The URL of the webpage to fetch",
                "format": "Output format: 'markdown' (default), 'text', or 'html'",
                "max_length": "Maximum length of content to return in characters",
                "include_images": "Whether to include image descriptions",
                "timeout": "Request timeout in seconds"
            }
        )
        
        # HTTP request tool
        self.register_tool(
            self.make_http_request,
            name="make_http_request",
            description="Make an HTTP request to an API endpoint",
            parameter_descriptions={
                "url": "The URL of the API endpoint",
                "method": "HTTP method (GET, POST, PUT, DELETE)",
                "headers": "HTTP headers as a JSON string or dictionary",
                "data": "Request body as a JSON string or dictionary",
                "params": "URL parameters as a JSON string or dictionary"
            }
        )
        
    def render_state_for_context(self) -> Dict[str, Any]:
        """
        Render the web environment's state for inclusion in the agent's context.
        
        This provides information about recent web searches and accessed URLs
        that can help the agent maintain context about web interactions.
        
        Returns:
            Dictionary with formatted web environment state
        """
        # Get base state info
        state = super().render_state_for_context()
        state["type"] = "web"
        
        # Build a formatted state text
        formatted_text = []
        
        # Add recent searches if any
        if self._recent_searches:
            formatted_text.append("Recent searches:")
            for search in self._recent_searches[-5:]:  # Show last 5 searches
                query = search.get("query", "Unknown query")
                timestamp = search.get("timestamp", "Unknown time")
                results_count = len(search.get("results", []))
                formatted_text.append(f"- \"{query}\" ({results_count} results) at {timestamp}")
        else:
            formatted_text.append("No recent web searches.")
            
        # Add recent URLs if any
        if self._recent_urls:
            formatted_text.append("\nRecently accessed URLs:")
            for url_entry in self._recent_urls[-5:]:  # Show last 5 URLs
                url = url_entry.get("url", "Unknown URL")
                timestamp = url_entry.get("timestamp", "Unknown time")
                success = url_entry.get("success", False)
                status = "successfully" if success else "with errors"
                formatted_text.append(f"- {url} (accessed {status} at {timestamp})")
        else:
            formatted_text.append("\nNo recently accessed URLs.")
        
        # Set the formatted text
        state["formatted_state_text"] = "\n".join(formatted_text)
        
        # Include raw data for potential specialized handling
        state["recent_searches"] = self._recent_searches[-5:] if self._recent_searches else []
        state["recent_urls"] = self._recent_urls[-5:] if self._recent_urls else []
        
        return state
    
    def web_search(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            num_results: Number of search results to return
            
        Returns:
            List of search results
        """
        # Record the search in recent_searches
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        try:
            # Implementation would be here...
            # This is a placeholder
            results = [{"title": f"Result {i} for {query}", "url": f"https://example.com/result{i}", "snippet": f"This is result {i} for {query}"} for i in range(num_results)]
            
            # Record successful search
            self._recent_searches.append({
                "query": query,
                "timestamp": timestamp,
                "results": results
            })
            
            # Limit history size
            if len(self._recent_searches) > 20:
                self._recent_searches = self._recent_searches[-20:]
                
            return results
            
        except Exception as e:
            logger.error(f"Error during web search for '{query}': {str(e)}")
            
            # Record failed search
            self._recent_searches.append({
                "query": query,
                "timestamp": timestamp,
                "error": str(e),
                "results": []
            })
            
            raise
    
    def fetch_webpage(self, url: str, format: str = "markdown", 
                 max_length: int = 100000, include_images: bool = False,
                 timeout: int = 10) -> Dict[str, Any]:
        """
        Fetch and process a webpage.
        
        Args:
            url: URL to fetch
            format: Format to return the content in ("markdown", "html", "text")
            max_length: Maximum length of content to return
            include_images: Whether to include image information
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with webpage content and metadata
        """
        # Record the URL access
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        try:
            # Implementation would be here...
            # This is a placeholder
            content = f"Content for {url} in {format} format"
            
            # Record successful fetch
            self._recent_urls.append({
                "url": url,
                "timestamp": timestamp,
                "format": format,
                "success": True
            })
            
            # Limit history size
            if len(self._recent_urls) > 20:
                self._recent_urls = self._recent_urls[-20:]
                
            return {
                "url": url,
                "content": content[:max_length],
                "title": f"Title for {url}",
                "timestamp": timestamp,
                "format": format
            }
            
        except Exception as e:
            logger.error(f"Error fetching webpage {url}: {str(e)}")
            
            # Record failed fetch
            self._recent_urls.append({
                "url": url,
                "timestamp": timestamp,
                "format": format,
                "success": False,
                "error": str(e)
            })
            
            raise
    
    def make_http_request(self, url: str, method: str = "GET", 
                        headers: Optional[Any] = None, 
                        data: Optional[Any] = None,
                        params: Optional[Any] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to an API endpoint.
        
        Args:
            url: The URL of the API endpoint
            method: HTTP method (GET, POST, PUT, DELETE)
            headers: HTTP headers as a JSON string or dictionary
            data: Request body as a JSON string or dictionary
            params: URL parameters as a JSON string or dictionary
            
        Returns:
            Dictionary with response information
        """
        logger.info(f"Making HTTP {method} request to: {url}")
        
        # Normalize method
        method = method.upper()
        if method not in ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"]:
            return {
                "success": False,
                "error": f"Unsupported HTTP method: {method}",
                "content": None
            }
            
        # Parse headers, data, and params if they are JSON strings
        try:
            if headers and isinstance(headers, str):
                headers = json.loads(headers)
                
            if data and isinstance(data, str):
                data = json.loads(data)
                
            if params and isinstance(params, str):
                params = json.loads(params)
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"JSON parsing error: {str(e)}",
                "content": None
            }
            
        # Validate URL format and security (similar to fetch_webpage)
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {
                    "success": False,
                    "error": "Invalid URL format. Please provide a URL including http:// or https://",
                    "content": None
                }
            
            # Security check for non-HTTP protocols
            if parsed_url.scheme not in ['http', 'https']:
                return {
                    "success": False,
                    "error": f"Unsupported URL protocol: {parsed_url.scheme}. Only HTTP and HTTPS are allowed.",
                    "content": None
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"URL validation error: {str(e)}",
                "content": None
            }
            
        # Make the request
        try:
            # Start timing
            start_time = time.time()
            
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data if method in ["POST", "PUT", "PATCH"] else None,
                params=params,
                timeout=30  # Default timeout of 30 seconds
            )
            
            # Get response time
            response_time = time.time() - start_time
            
            # Try to parse JSON response
            try:
                content = response.json()
            except:
                content = response.text
                
            return {
                "success": True,
                "status_code": response.status_code,
                "url": url,
                "method": method,
                "headers": dict(response.headers),
                "content": content,
                "response_time": response_time
            }
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out",
                "content": None
            }
        except requests.exceptions.TooManyRedirects:
            return {
                "success": False,
                "error": "Too many redirects",
                "content": None
            }
        except requests.exceptions.HTTPError as e:
            return {
                "success": False,
                "error": f"HTTP error: {str(e)}",
                "content": None
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Connection error. Please check the URL and your internet connection.",
                "content": None
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Request error: {str(e)}",
                "content": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "content": None
            } 