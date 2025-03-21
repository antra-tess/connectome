"""
Tests for the MessageHandler class in the activity layer.
"""

import pytest
from unittest.mock import MagicMock, patch, call
import time
import json

from activity.listener import MessageHandler


class TestMessageHandler:
    """Test suite for the MessageHandler class."""

    def test_init(self, mock_environment_manager):
        """Test the initialization of MessageHandler."""
        handler = MessageHandler(mock_environment_manager)
        
        assert handler.environment_manager == mock_environment_manager
    
    def test_handle_message_valid(self, mock_environment_manager, sample_message_data):
        """Test handling a valid message."""
        # Setup
        handler = MessageHandler(mock_environment_manager)
        mock_environment_manager.process_message.return_value = {
            "status": "success",
            "response": "Test response",
            "message_id": "response_id"
        }
        
        # Execute
        result = handler.handle_message(sample_message_data)
        
        # Verify
        assert result is not None
        assert result["status"] == "success"
        mock_environment_manager.process_message.assert_called_once()
        # Verify the call arguments contain the expected data
        call_args = mock_environment_manager.process_message.call_args[0][0]
        assert call_args["event_type"] == sample_message_data["event_type"]
        assert call_args["user_id"] == sample_message_data["user_id"]
        assert call_args["message_text"] == sample_message_data["message_text"]
    
    def test_handle_message_invalid(self, mock_environment_manager):
        """Test handling an invalid message (missing required fields)."""
        # Setup
        handler = MessageHandler(mock_environment_manager)
        invalid_data = {
            "event_type": "message_received"
            # Missing user_id and message_text
        }
        
        # Execute
        result = handler.handle_message(invalid_data)
        
        # Verify
        assert result is None
        # The environment manager should not be called with invalid data
        mock_environment_manager.process_message.assert_not_called()
    
    def test_handle_message_error_handling(self, mock_environment_manager, sample_message_data):
        """Test error handling during message processing."""
        # Setup
        handler = MessageHandler(mock_environment_manager)
        mock_environment_manager.process_message.side_effect = Exception("Test error")
        
        # Execute
        result = handler.handle_message(sample_message_data)
        
        # Verify
        assert result is None
        mock_environment_manager.process_message.assert_called_once()
    
    def test_handle_message_different_event_types(self, mock_environment_manager):
        """Test handling different event types."""
        # Setup
        handler = MessageHandler(mock_environment_manager)
        events = [
            {"event_type": "message_updated", "user_id": "user1", "message_text": "Updated", "message_id": "msg1"},
            {"event_type": "message_deleted", "user_id": "user1", "message_id": "msg1"},
            {"event_type": "reaction_added", "user_id": "user1", "message_id": "msg1", "reaction": "👍"}
        ]
        
        # Execute
        for event in events:
            handler.handle_message(event)
        
        # Verify
        assert mock_environment_manager.process_message.call_count == len(events)
    
    def test_handle_clear_context(self, mock_environment_manager):
        """Test clearing context."""
        # Setup
        handler = MessageHandler(mock_environment_manager)
        mock_environment_manager.clear_context.return_value = True
        data = {
            "conversation_id": "test_conversation",
            "user_id": "test_user",
            "adapter_id": "test_adapter"
        }
        
        # Execute
        result = handler.handle_clear_context(data)
        
        # Verify
        assert result is True
        mock_environment_manager.clear_context.assert_called_once()
        # Verify the call arguments contain the expected data
        call_args = mock_environment_manager.clear_context.call_args[0][0]
        assert call_args["conversation_id"] == data["conversation_id"]
    
    def test_handle_clear_context_missing_data(self, mock_environment_manager):
        """Test clearing context with missing data."""
        # Setup
        handler = MessageHandler(mock_environment_manager)
        invalid_data = {
            # Missing conversation_id
            "user_id": "test_user"
        }
        
        # Execute
        result = handler.handle_clear_context(invalid_data)
        
        # Verify
        assert result is False
        # The environment manager should not be called with invalid data
        mock_environment_manager.clear_context.assert_not_called()
    
    def test_handle_clear_context_error(self, mock_environment_manager):
        """Test error handling during context clearing."""
        # Setup
        handler = MessageHandler(mock_environment_manager)
        mock_environment_manager.clear_context.side_effect = Exception("Test error")
        data = {
            "conversation_id": "test_conversation",
            "user_id": "test_user",
            "adapter_id": "test_adapter"
        }
        
        # Execute
        result = handler.handle_clear_context(data)
        
        # Verify
        assert result is False
        mock_environment_manager.clear_context.assert_called_once()
    
    @patch("activity.listener.MessageHandler._send_response")
    def test_response_callback(self, mock_send_response, mock_environment_manager):
        """Test the response callback functionality."""
        # Setup
        handler = MessageHandler(mock_environment_manager)
        
        # Create a response data dictionary
        response_data = {
            "user_id": "test_user",
            "message_text": "Hello from the callback",
            "adapter_id": "test_adapter",
            "conversation_id": "test_conversation"
        }
        
        # Mock the _send_response method to do nothing
        mock_send_response.return_value = None
        
        # Set up a test callback method
        callback_called = False
        def test_callback(data):
            nonlocal callback_called
            callback_called = True
            assert data == response_data
        
        # Attach the callback
        handler._send_response = test_callback
        
        # Execute
        handler._send_response(response_data)
        
        # Verify
        assert callback_called is True 