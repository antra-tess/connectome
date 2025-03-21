"""
Integration tests for the activity layer components.

These tests verify that the components of the activity layer work well together.
"""

import pytest
from unittest.mock import MagicMock, patch, call
import json

# Change the imports to be relative to the current package
from activity.client import SocketIOClient
from activity.listener import MessageHandler


@pytest.fixture
def mock_socketio_module():
    """Mock the socketio module."""
    with patch('activity.client.socketio') as mock_socketio:
        # Configure the mock socketio module
        mock_client = MagicMock()
        mock_socketio.Client.return_value = mock_client
        
        yield mock_socketio


class TestActivityLayerIntegration:
    """Integration tests for the activity layer components."""

    @patch('activity.client.SocketIOClient._load_adapter_configs')
    def test_message_flow(self, mock_load_configs, mock_environment_manager, mock_socketio_module, sample_message_data):
        """
        Test the complete message flow from receiving a message to sending a response.
        
        This test verifies:
        1. A message is received by the MessageHandler
        2. The message is processed by the environment manager
        3. A response is generated
        4. The SocketIOClient sends the response back
        """
        # Setup
        # Create the message handler with a mock environment manager
        message_handler = MessageHandler(mock_environment_manager)
        
        # Configure the mock environment manager to return a valid response
        mock_response = {
            "status": "success",
            "response": "This is a test response",
            "message_id": "response_id",
            "user_id": sample_message_data["user_id"],
            "adapter_id": sample_message_data["adapter_id"]
        }
        mock_environment_manager.process_message.return_value = mock_response
        
        # Create the SocketIOClient with the message handler
        socket_client = SocketIOClient(message_handler)
        
        # Configure the socket client manually
        mock_client = mock_socketio_module.Client.return_value
        socket_client.clients[sample_message_data["adapter_id"]] = mock_client
        socket_client.connected_adapters[sample_message_data["adapter_id"]] = True
        
        # Patch the socket client's send_response method to capture calls
        with patch.object(socket_client, 'send_response') as mock_send_response:
            mock_send_response.return_value = True
            
            # Patch the MessageHandler's _send_response method to use our socket client
            def send_response_patch(response_data):
                return socket_client.send_response(
                    user_id=response_data.get("user_id"),
                    message_text=response_data.get("response"),
                    message_id=response_data.get("message_id"),
                    adapter_id=response_data.get("adapter_id")
                )
            
            with patch.object(message_handler, '_send_response', send_response_patch):
                # Execute - simulate receiving a message
                result = message_handler.handle_message(sample_message_data)
                
                # Verify
                # Check that the message was processed
                assert result["status"] == "success"
                mock_environment_manager.process_message.assert_called_once()
                
                # Check that send_response was called with the correct parameters
                mock_send_response.assert_called_once()
                call_args = mock_send_response.call_args[1]  # Get the kwargs
                assert call_args["user_id"] == sample_message_data["user_id"]
                assert call_args["message_text"] == mock_response["response"]
    
    @patch('activity.client.SocketIOClient._load_adapter_configs')
    def test_clear_context_flow(self, mock_load_configs, mock_environment_manager, mock_socketio_module):
        """
        Test the flow for clearing context.
        
        This test verifies:
        1. A clear context request is received
        2. The request is processed by the environment manager
        3. A confirmation response is sent back
        """
        # Setup
        # Create the message handler with a mock environment manager
        message_handler = MessageHandler(mock_environment_manager)
        
        # Configure the mock environment manager to return success for clear_context
        mock_environment_manager.clear_context.return_value = True
        
        # Create the SocketIOClient with the message handler
        socket_client = SocketIOClient(message_handler)
        
        # Configure the socket client manually
        mock_client = mock_socketio_module.Client.return_value
        adapter_id = "test_adapter"
        socket_client.clients[adapter_id] = mock_client
        socket_client.connected_adapters[adapter_id] = True
        
        # Create a clear context request
        clear_context_data = {
            "conversation_id": "test_conversation",
            "user_id": "test_user",
            "adapter_id": adapter_id
        }
        
        # Execute
        result = message_handler.handle_clear_context(clear_context_data)
        
        # Verify
        assert result is True
        mock_environment_manager.clear_context.assert_called_once()
    
    @patch('activity.client.SocketIOClient._load_adapter_configs')
    def test_bot_request_event_handling(self, mock_load_configs, mock_environment_manager, mock_socketio_module, sample_message_data):
        """
        Test handling of bot_request events from Socket.IO.
        
        This test verifies that when a Socket.IO client receives a bot_request event,
        it properly forwards the message to the MessageHandler.
        """
        # Setup
        # Create the message handler with a mock environment manager
        message_handler = MessageHandler(mock_environment_manager)
        message_handler.handle_message = MagicMock(return_value={"status": "success"})
        
        # Create the SocketIOClient with the message handler
        socket_client = SocketIOClient(message_handler)
        
        # Configure the socket client manually
        mock_client = mock_socketio_module.Client.return_value
        adapter_id = "test_adapter"
        socket_client.clients[adapter_id] = mock_client
        
        # Register event handlers (this is where the bot_request handler is defined)
        socket_client._register_event_handlers(mock_client, adapter_id)
        
        # Get the bot_request handler from the registered handlers
        bot_request_handler = None
        for args, kwargs in mock_client.on.call_args_list:
            if args[0] == 'bot_request':
                bot_request_handler = args[1]
                break
        
        assert bot_request_handler is not None, "bot_request handler not registered"
        
        # Execute - simulate receiving a bot_request event
        event_data = {
            "message": "Hello bot",
            "user_id": "user123",
            "conversation_id": "conv123"
        }
        
        # Call the bot_request handler with the event data
        bot_request_handler(event_data)
        
        # Verify
        message_handler.handle_message.assert_called_once()
        # Check that the adapter_id was added to the data
        call_args = message_handler.handle_message.call_args[0][0]
        assert "adapter_id" in call_args
        assert call_args["adapter_id"] == adapter_id 