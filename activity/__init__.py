"""
Messaging Module
Handles communication with normalizing adapters via Socket.IO clients.
"""

from activity.client import SocketIOClient
from activity.listener import create_message_handler, MessageHandler
from activity.sender import (
    initialize_sender,
    send_response,
    send_error,
    send_typing_indicator
) 