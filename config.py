"""
Configuration settings for the Bot Framework
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Socket.IO Server Settings
SOCKET_HOST = os.getenv('SOCKET_HOST', '0.0.0.0')
SOCKET_PORT = int(os.getenv('SOCKET_PORT', 5000))

# LiteLLM Settings
LLM_API_KEY = os.getenv('LLM_API_KEY', '')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4')
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')
LLM_BASE_URL = os.getenv('LLM_BASE_URL', '')

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