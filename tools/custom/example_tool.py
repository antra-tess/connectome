"""
Example Custom Tool
Shows how to create and register a custom tool for the Bot Framework.
"""

import logging
from typing import Optional, Dict, Any

from tools.registry import register_tool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@register_tool(
    name="example_weather_tool",
    description="Get the current weather for a specified location.",
    parameter_descriptions={
        "location": "Name of the city or location to get weather for",
        "units": "Temperature units: 'celsius' or 'fahrenheit' (optional)"
    }
)
def get_weather(location: str, units: Optional[str] = "celsius") -> Dict[str, Any]:
    """
    Example custom tool to fetch weather information.
    
    This is just a demonstration and returns mock data.
    In a real implementation, you would call a weather API here.
    
    Args:
        location: The city or location to get weather for
        units: Temperature units to use (celsius or fahrenheit)
        
    Returns:
        Dictionary with weather information
    """
    logger.info(f"Getting weather for {location} in {units}")
    
    # This would be replaced with a real API call
    mock_weather_data = {
        "location": location,
        "temperature": 22 if units == "celsius" else 72,
        "units": units,
        "condition": "Sunny",
        "humidity": 65,
        "wind_speed": 5
    }
    
    return mock_weather_data


# Example of how to register the tool if you want to do it after the function is defined
# (this is an alternative to the decorator approach)
# 
# from tools.registry import registry
# registry.register(
#     name="example_weather_tool",
#     description="Get the current weather for a specified location."
# )(get_weather) 