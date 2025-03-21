"""
Tests for the SocketIOClient class in the activity layer.
"""

import pytest
from unittest.mock import MagicMock, patch, call, PropertyMock
import json

from activity.client import SocketIOClient


class TestSocketIOClient:
    """Test suite for the SocketIOClient class."""

    @patch('activity.client.socketio.Client')
    def test_init(self, mock_socketio_client_class, mock_environment_manager):
        """Test the initialization of SocketIOClient."""
        # Setup
        message_handler = MagicMock()
        
        # Execute
        with patch('activity.client.SocketIOClient._load_adapter_configs'):
            client = SocketIOClient(message_handler)
        
        # Verify
        assert client.message_handler == message_handler
        assert isinstance(client.clients, dict)
        assert isinstance(client.adapters, dict)
        assert isinstance(client.connected_adapters, dict)
    
    @patch('activity.client.DEFAULT_ADAPTERS')
    @patch('activity.client.ADDITIONAL_ADAPTERS')
    @patch('activity.client.json.loads')
    def test_load_adapter_configs(self, mock_json_loads, mock_additional_adapters, mock_default_adapters):
        """Test loading adapter configurations."""
        # Setup
        message_handler = MagicMock()
        mock_default_adapters.return_value = [
            {
                "id": "default",
                "name": "Default Adapter",
                "url": "http://localhost:5000",
                "auth_token": "",
                "platforms": ["default"]
            }
        ]
        mock_additional_adapters.return_value = '[{"id":"test","name":"Test Adapter","url":"http://test:5001","auth_token":"test_token","platforms":["test"]}]'
        mock_json_loads.return_value = [
            {
                "id": "test",
                "name": "Test Adapter",
                "url": "http://test:5001",
                "auth_token": "test_token",
                "platforms": ["test"]
            }
        ]
        
        # Execute
        with patch('activity.client.SocketIOClient._load_adapter_configs'):
            client = SocketIOClient(message_handler)
        
        # Reset the mock to clear the call made during initialization
        mock_json_loads.reset_mock()
        
        # Now call the method directly
        client._load_adapter_configs()
        
        # Verify
        mock_json_loads.assert_called_once_with(mock_additional_adapters)
        # Check that adapters were loaded
        assert len(client.adapters) > 0
    
    @patch('activity.client.socketio.Client')
    def test_connect_to_adapters(self, mock_socketio_client_class, sample_adapter_config):
        """Test connecting to adapters."""
        # Setup
        message_handler = MagicMock()
        
        # Mock initialization
        with patch('activity.client.SocketIOClient._load_adapter_configs'):
            client = SocketIOClient(message_handler)
        
        # Set up the adapters dict manually
        client.adapters = {"test_adapter": sample_adapter_config}
        
        # Mock the _connect_to_adapter method
        client._connect_to_adapter = MagicMock()
        
        # Execute
        client.connect_to_adapters()
        
        # Verify
        client._connect_to_adapter.assert_called_once_with("test_adapter", sample_adapter_config)
    
    @patch('activity.client.socketio.Client')
    def test_connect_to_adapter(self, mock_socketio_client_class, sample_adapter_config):
        """Test connecting to a specific adapter."""
        # Setup
        mock_client = MagicMock()
        mock_socketio_client_class.return_value = mock_client
        message_handler = MagicMock()
        
        # Mock initialization
        with patch('activity.client.SocketIOClient._load_adapter_configs'):
            client = SocketIOClient(message_handler)
        
        # Mock the _register_event_handlers and _register_with_adapter methods
        client._register_event_handlers = MagicMock()
        client._register_with_adapter = MagicMock()
        
        # Execute
        client._connect_to_adapter("test_adapter", sample_adapter_config)
        
        # Verify
        mock_socketio_client_class.assert_called_once()
        mock_client.connect.assert_called_once()
        # Verify the arguments passed to connect contain the URL from the adapter config
        connect_args = mock_client.connect.call_args[0][0]
        assert sample_adapter_config["url"] in connect_args
        client._register_event_handlers.assert_called_once()
        client._register_with_adapter.assert_called_once()
    
    @patch('activity.client.socketio.Client')
    def test_register_event_handlers(self, mock_socketio_client_class, sample_adapter_config):
        """Test registering event handlers with a Socket.IO client."""
        # Setup
        mock_client = MagicMock()
        message_handler = MagicMock()
        
        # Mock initialization
        with patch('activity.client.SocketIOClient._load_adapter_configs'):
            client = SocketIOClient(message_handler)
        
        # Execute
        client._register_event_handlers(mock_client, "test_adapter")
        
        # Verify
        # Check that the event decorators were called
        assert mock_client.event.call_count >= 3  # connect, disconnect, connect_error
        assert mock_client.on.call_count >= 1     # bot_request
    
    @patch('activity.client.socketio.Client')
    def test_register_with_adapter(self, mock_socketio_client_class, sample_adapter_config):
        """Test registering with an adapter."""
        # Setup
        mock_client = MagicMock()
        mock_socketio_client_class.return_value = mock_client
        message_handler = MagicMock()
        
        # Mock initialization
        with patch('activity.client.SocketIOClient._load_adapter_configs'):
            client = SocketIOClient(message_handler)
        
        # Setup client for the adapter
        client.clients["test_adapter"] = mock_client
        
        # Execute
        client._register_with_adapter("test_adapter", sample_adapter_config)
        
        # Verify
        mock_client.emit.assert_called_once()
        # Check that the emission was to 'register'
        assert mock_client.emit.call_args[0][0] == 'register'
    
    @patch('activity.client.socketio.Client')
    def test_send_message(self, mock_socketio_client_class, sample_adapter_config):
        """Test sending a message to an adapter."""
        # Setup
        mock_client = MagicMock()
        message_handler = MagicMock()
        
        # Mock initialization
        with patch('activity.client.SocketIOClient._load_adapter_configs'):
            client = SocketIOClient(message_handler)
        
        # Setup client and connected status for the adapter
        client.clients["test_adapter"] = mock_client
        client.connected_adapters["test_adapter"] = True
        
        # Create a test message
        test_message = {
            "adapter_id": "test_adapter",
            "user_id": "test_user",
            "message_text": "Test message"
        }
        
        # Execute
        result = client.send_message(test_message)
        
        # Verify
        assert result is True
        mock_client.emit.assert_called_once()
        # Check that the emission was to the 'message' event
        assert mock_client.emit.call_args[0][0] == 'message'
    
    @patch('activity.client.socketio.Client')
    def test_send_message_not_connected(self, mock_socketio_client_class):
        """Test sending a message to an adapter that is not connected."""
        # Setup
        message_handler = MagicMock()
        
        # Mock initialization
        with patch('activity.client.SocketIOClient._load_adapter_configs'):
            client = SocketIOClient(message_handler)
        
        # Create a test message to an adapter that doesn't exist or isn't connected
        test_message = {
            "adapter_id": "nonexistent_adapter",
            "user_id": "test_user",
            "message_text": "Test message"
        }
        
        # Execute
        result = client.send_message(test_message)
        
        # Verify
        assert result is False
    
    @patch('activity.client.socketio.Client')
    def test_send_response(self, mock_socketio_client_class, sample_adapter_config):
        """Test sending a response to a user."""
        # Setup
        mock_client = MagicMock()
        message_handler = MagicMock()
        
        # Mock initialization
        with patch('activity.client.SocketIOClient._load_adapter_configs'):
            client = SocketIOClient(message_handler)
        
        # Setup client and connected status for the adapter
        client.clients["test_adapter"] = mock_client
        client.connected_adapters["test_adapter"] = True
        
        # Execute
        result = client.send_response(
            user_id="test_user",
            message_text="Test response",
            message_id="test_message_id",
            platform="test_platform",
            adapter_id="test_adapter"
        )
        
        # Verify
        assert result is True
        mock_client.emit.assert_called_once()
        # Check that the emission was to the 'response' event
        assert mock_client.emit.call_args[0][0] == 'response'
    
    @patch('activity.client.socketio.Client')
    def test_send_error(self, mock_socketio_client_class, sample_adapter_config):
        """Test sending an error message."""
        # Setup
        mock_client = MagicMock()
        message_handler = MagicMock()
        
        # Mock initialization
        with patch('activity.client.SocketIOClient._load_adapter_configs'):
            client = SocketIOClient(message_handler)
        
        # Setup client and connected status for the adapter
        client.clients["test_adapter"] = mock_client
        client.connected_adapters["test_adapter"] = True
        
        # Execute
        result = client.send_error(
            adapter_id="test_adapter",
            chat_id="test_chat",
            error_message="Test error message"
        )
        
        # Verify
        assert result is True
        mock_client.emit.assert_called_once()
        # Check that the emission was to the 'error' event
        assert mock_client.emit.call_args[0][0] == 'error'
    
    @patch('activity.client.socketio.Client')
    def test_send_typing_indicator(self, mock_socketio_client_class, sample_adapter_config):
        """Test sending a typing indicator."""
        # Setup
        mock_client = MagicMock()
        message_handler = MagicMock()
        
        # Mock initialization
        with patch('activity.client.SocketIOClient._load_adapter_configs'):
            client = SocketIOClient(message_handler)
        
        # Setup client and connected status for the adapter
        client.clients["test_adapter"] = mock_client
        client.connected_adapters["test_adapter"] = True
        
        # Execute
        result = client.send_typing_indicator(
            adapter_id="test_adapter",
            chat_id="test_chat",
            is_typing=True
        )
        
        # Verify
        assert result is True
        mock_client.emit.assert_called_once()
        # Check that the emission was to the 'typing' event
        assert mock_client.emit.call_args[0][0] == 'typing'
    
    @patch('activity.client.socketio.Client')
    def test_close_connections(self, mock_socketio_client_class):
        """Test closing all connections."""
        # Setup
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()
        message_handler = MagicMock()
        
        # Mock initialization
        with patch('activity.client.SocketIOClient._load_adapter_configs'):
            client = SocketIOClient(message_handler)
        
        # Setup multiple clients
        client.clients = {
            "adapter1": mock_client1,
            "adapter2": mock_client2
        }
        
        # Execute
        client.close_connections()
        
        # Verify
        mock_client1.disconnect.assert_called_once()
        mock_client2.disconnect.assert_called_once() 