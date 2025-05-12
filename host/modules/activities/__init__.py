"""
Host Activity Modules

Modules responsible for interacting with external services/APIs via
a standardized interface.
"""

# Import the central client and listener classes
from .activity_client import ActivityClient

# Optionally, import specific adapter implementations if they were 
# implemented here instead of externally.
# from .discord_adapter import DiscordAdapter

__all__ = [
    "ActivityClient",
    "ActivityListener"
    # "DiscordAdapter", 
] 