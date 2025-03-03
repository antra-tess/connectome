"""
Configuration settings for the Bot Framework
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Adapter Connection Settings
# Default adapter settings (can be overridden or extended via environment variables)
DEFAULT_ADAPTERS = [
    {
        "id": "default",
        "name": "Default Adapter",
        "url": os.getenv('DEFAULT_ADAPTER_URL', 'http://localhost:5000'),
        "auth_token": os.getenv('DEFAULT_ADAPTER_TOKEN', ''),
        "platforms": ["default"]
    }
]

# Additional adapters can be configured via environment variables or config files
# Example format in .env: ADAPTERS='[{"id":"slack","name":"Slack Adapter","url":"http://slack-adapter:5001","auth_token":"token123","platforms":["slack"]}]'
ADDITIONAL_ADAPTERS = os.getenv('ADAPTERS', '[]')

# Socket.IO Client Settings
SOCKET_RECONNECTION_ATTEMPTS = int(os.getenv('SOCKET_RECONNECTION_ATTEMPTS', 5))
SOCKET_RECONNECTION_DELAY = int(os.getenv('SOCKET_RECONNECTION_DELAY', 1000))
SOCKET_TIMEOUT = int(os.getenv('SOCKET_TIMEOUT', 5000))

# LiteLLM Settings
LLM_API_KEY = os.getenv('LLM_API_KEY', '')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4')
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')
LLM_BASE_URL = os.getenv('LLM_BASE_URL', '')
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))
LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', 1000))

# Storage Settings
STORAGE_TYPE = os.getenv('STORAGE_TYPE', 'file')  # 'file', 'redis', etc.
STORAGE_PATH = os.getenv('STORAGE_PATH', 'data/')

# Context Settings
MAX_HISTORY_MESSAGES = int(os.getenv('MAX_HISTORY_MESSAGES', 50))
CONTEXT_WINDOW_TOKENS = int(os.getenv('CONTEXT_WINDOW_TOKENS', 8192))

# Agent Settings
DEFAULT_PROTOCOL = os.getenv('DEFAULT_PROTOCOL', 'react')  # 'react', 'function_calling', 'custom'
AGENT_NAME = os.getenv('AGENT_NAME', 'BotFramework Assistant')
AGENT_DESCRIPTION = os.getenv('AGENT_DESCRIPTION', 'A helpful assistant that can participate in multi-user conversations.') 

# Environment Configuration
# Default set of environments to initialize and mount to the system environment
# Format: List of dictionaries with 'id', 'class', 'enabled', and 'mount_point' keys
# - id: Unique identifier for the environment
# - class: The class name (must be importable)
# - enabled: Whether to initialize this environment
# - mount_point: Optional mount point in the system environment (if None, uses the environment id)
DEFAULT_ENVIRONMENTS = [
    {
        "id": "web",
        "class": "WebEnvironment",
        "enabled": True,
        "mount_point": None  # Will use the id as mount point
    },
    {
        "id": "messaging",
        "class": "MessagingEnvironment",
        "enabled": True,
        "mount_point": None
    },
    {
        "id": "file",
        "class": "FileEnvironment",
        "enabled": True,
        "mount_point": None
    }
]

# Parse environment-specific configuration from environment variables
# Example format in .env: ENVIRONMENTS='[{"id":"custom","class":"CustomEnvironment","enabled":true,"mount_point":"custom_mount"}]'
ENVIRONMENTS_CONFIG = os.getenv('ENVIRONMENTS', '[]')

# Combined environment configuration (default + any from environment variables)
try:
    import json
    additional_envs = json.loads(ENVIRONMENTS_CONFIG)
    ENVIRONMENTS = DEFAULT_ENVIRONMENTS + additional_envs
except (json.JSONDecodeError, TypeError):
    ENVIRONMENTS = DEFAULT_ENVIRONMENTS 
