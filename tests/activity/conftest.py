"""
Test configuration and fixtures specific to activity layer tests.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_socketio_client_instance():
    """
    Fixture that provides a mock SocketIOClient instance.
    """
    mock_client = MagicMock()
    
    # Configure common methods
    mock_client.send_response = MagicMock(return_value=True)
    mock_client.send_message = MagicMock(return_value=True)
    mock_client.send_error = MagicMock(return_value=True)
    mock_client.send_typing_indicator = MagicMock(return_value=True)
    mock_client.clients = {}
    mock_client.connected_adapters = {}
    
    return mock_client


@pytest.fixture
def mock_message_handler():
    """
    Fixture that provides a mock MessageHandler instance.
    """
    mock_handler = MagicMock()
    
    # Configure common methods
    mock_handler.handle_message = MagicMock(return_value={"status": "success"})
    mock_handler.handle_clear_context = MagicMock(return_value=True)
    
    return mock_handler


@pytest.fixture
def bot_request_event_data():
    """
    Fixture that provides sample bot_request event data.
    """
    return {
        "message": "Hello, bot!",
        "user_id": "test_user",
        "conversation_id": "test_conversation",
        "timestamp": 1646870400,
        "platform": "test_platform",
        "message_id": "test_message_id"
    }


@pytest.fixture
def clear_context_event_data():
    """
    Fixture that provides sample clear_context event data.
    """
    return {
        "conversation_id": "test_conversation",
        "user_id": "test_user",
        "adapter_id": "test_adapter"
    } 