"""
Web Tools
Tools for interacting with web resources and APIs.
"""

import logging
import json
from typing import Optional, Dict, Any, List
import requests

from tools.registry import register_tool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@register_tool(
    name="web_search",
    description="Search the web for information on a topic.",
    parameter_descriptions={
        "query": "The search query",
        "num_results": "Number of search results to return (optional)"
    }
)
def web_search(query: str, num_results: Optional[int] = 3) -> List[Dict[str, str]]:
    """
    Search the web for information.
    
    This is a placeholder for a real web search implementation. In production,
    this would connect to a search engine API (Google, Bing, etc.)
    
    Args:
        query: The search query
        num_results: Number of results to return
        
    Returns:
        List of search results, each containing a title, url, and snippet
    """
    logger.info(f"Web search for: {query}, requesting {num_results} results")
    
    # This is a mock response - in production, this would call a real search API
    mock_results = [
        {
            "title": f"Example result 1 for: {query}",
            "url": "https://example.com/result1",
            "snippet": f"This is a sample search result about {query}. It contains some context that might be useful for the agent."
        },
        {
            "title": f"Example result 2 for: {query}",
            "url": "https://example.com/result2",
            "snippet": f"Another example result about {query}, with slightly different information than the first result."
        },
        {
            "title": f"Example result 3 for: {query}",
            "url": "https://example.com/result3",
            "snippet": f"A third example of what a search result for {query} might look like in a real search engine."
        },
        {
            "title": f"Example result 4 for: {query}",
            "url": "https://example.com/result4",
            "snippet": f"This would be a fourth result for {query}, showing that we can return a variable number of results."
        },
        {
            "title": f"Example result 5 for: {query}",
            "url": "https://example.com/result5",
            "snippet": f"A fifth example of a search result for {query}, demonstrating the flexibility of the search tool."
        }
    ]
    
    return mock_results[:min(num_results, len(mock_results))]


@register_tool(
    name="fetch_webpage",
    description="Fetch the content of a webpage given its URL.",
    parameter_descriptions={
        "url": "The URL of the webpage to fetch",
        "extract_text_only": "Whether to return only the main text content (optional)"
    }
)
def fetch_webpage(url: str, extract_text_only: Optional[bool] = True) -> str:
    """
    Fetch the content of a webpage.
    
    This is a placeholder that would need a proper implementation using
    requests, BeautifulSoup, or a similar library to fetch and parse web content.
    
    Args:
        url: The URL of the webpage to fetch
        extract_text_only: Whether to return only the main text content
        
    Returns:
        The fetched webpage content
    """
    logger.info(f"Fetching webpage: {url}, extract_text_only: {extract_text_only}")
    
    try:
        # In a real implementation, this would use requests to fetch the page
        # and BeautifulSoup to parse and extract text if extract_text_only is True
        
        # This is a mock response
        if "example.com" in url:
            return f"This is a mock webpage content for {url}. In a real implementation, this would contain the actual HTML or extracted text from the webpage. The 'extract_text_only' parameter is set to {extract_text_only}."
        else:
            return f"Mock content for {url}. This would contain either HTML or plain text depending on the extract_text_only parameter."
    
    except Exception as e:
        error_msg = f"Error fetching webpage {url}: {str(e)}"
        logger.error(error_msg)
        return error_msg


@register_tool(
    name="make_http_request",
    description="Make an HTTP request to an API endpoint.",
    parameter_descriptions={
        "url": "The URL of the API endpoint",
        "method": "HTTP method (GET, POST, PUT, DELETE)",
        "headers": "HTTP headers as a JSON string (optional)",
        "data": "Request body as a JSON string (optional)",
        "params": "URL parameters as a JSON string (optional)"
    }
)
def make_http_request(
    url: str, 
    method: str, 
    headers: Optional[str] = None,
    data: Optional[str] = None,
    params: Optional[str] = None
) -> Dict[str, Any]:
    """
    Make an HTTP request to an API endpoint.
    
    This allows the agent to interact with external APIs.
    Security measures should be implemented to restrict which domains
    the agent can make requests to.
    
    Args:
        url: The URL of the API endpoint
        method: HTTP method (GET, POST, PUT, DELETE)
        headers: HTTP headers as a JSON string
        data: Request body as a JSON string
        params: URL parameters as a JSON string
        
    Returns:
        Dictionary containing response data
    """
    logger.info(f"Making HTTP {method} request to {url}")
    
    # Security check - in a real implementation, this would check against
    # a whitelist of allowed domains
    allowed_domains = ["api.example.com", "data.example.org"]
    domain = url.split("//")[-1].split("/")[0]
    
    if not any(domain.endswith(allowed) for allowed in allowed_domains):
        error_msg = f"Security error: Requests to {domain} are not allowed"
        logger.error(error_msg)
        return {"error": error_msg, "status_code": 403}
    
    try:
        # Parse the optional parameters
        headers_dict = json.loads(headers) if headers else {}
        data_dict = json.loads(data) if data else {}
        params_dict = json.loads(params) if params else {}
        
        # In a real implementation, this would make the actual HTTP request
        # For now, we'll just return a mock response
        if "api.example.com" in url:
            mock_response = {
                "status_code": 200,
                "success": True,
                "data": {
                    "message": f"This is a mock response for a {method} request to {url}",
                    "request_headers": headers_dict,
                    "request_data": data_dict,
                    "request_params": params_dict
                }
            }
            return mock_response
        else:
            return {
                "status_code": 404,
                "success": False,
                "error": "API endpoint not found"
            }
    
    except json.JSONDecodeError as e:
        error_msg = f"Error parsing JSON: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "status_code": 400}
        
    except Exception as e:
        error_msg = f"Error making HTTP request: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "status_code": 500} 