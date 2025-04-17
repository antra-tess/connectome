import pytest
from unittest.mock import MagicMock, patch, call

from elements.elements.components.messaging.messaging_tools_component import MessagingToolsComponent
from elements.elements.components.tool_provider_component import ToolProviderComponent
from elements.elements.base_element import BaseElement

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_tool_provider():
    """Fixture for a mocked ToolProviderComponent."""
    mock = MagicMock(spec=ToolProviderComponent)
    mock.register_tool = MagicMock()
    return mock

@pytest.fixture
def mock_element():
    """Fixture for a mocked BaseElement."""
    element = MagicMock(spec=BaseElement)
    element.id = "test_element_msg_tools"
    # In this component, ToolProvider isn't fetched via get_component_by_type,
    # it's passed directly to register_tools.
    return element

@pytest.fixture
def messaging_tools_component(mock_element):
    """Fixture for a MessagingToolsComponent instance."""
    comp = MessagingToolsComponent(element=mock_element)
    comp._initialize()
    comp._enable()
    return comp

# --- Expected Tool Definitions (for comparison) ---

EXPECTED_TOOLS = {
    "send_message": {
        "description": "Sends a text message to a specific conversation via a specified adapter.",
        "parameters": {
            "type": "object",
            "properties": {
                "adapter_id": {"type": "string", "description": "The ID of the adapter to send through (e.g., 'discord_1')."},
                "conversation_id": {"type": "string", "description": "The ID of the conversation/channel/DM to send to."},
                "text": {"type": "string", "description": "The text content of the message to send."}
            },
            "required": ["adapter_id", "conversation_id", "text"]
        },
        "execution_info": {
            "type": "action_request",
            "target_module": "ActivityClient",
            "action_type": "send_message",
            "payload": {}
        }
    },
    "edit_message": {
        "description": "Edits the text content of a previously sent message.",
        "parameters": {
            "type": "object",
            "properties": {
                "adapter_id": {"type": "string", "description": "The ID of the adapter the message exists on."},
                "conversation_id": {"type": "string", "description": "The ID of the conversation containing the message."},
                "message_id": {"type": "string", "description": "The ID of the specific message to edit."},
                "text": {"type": "string", "description": "The new text content for the message."}
            },
            "required": ["adapter_id", "conversation_id", "message_id", "text"]
        },
        "execution_info": {
            "type": "action_request",
            "target_module": "ActivityClient",
            "action_type": "edit_message",
            "payload": {}
        }
    },
    "delete_message": {
        "description": "Deletes a previously sent message.",
        "parameters": {
            "type": "object",
            "properties": {
                "adapter_id": {"type": "string", "description": "The ID of the adapter the message exists on."},
                "conversation_id": {"type": "string", "description": "The ID of the conversation containing the message."},
                "message_id": {"type": "string", "description": "The ID of the specific message to delete."}
            },
            "required": ["adapter_id", "conversation_id", "message_id"]
        },
        "execution_info": {
            "type": "action_request",
            "target_module": "ActivityClient",
            "action_type": "delete_message",
            "payload": {}
        }
    },
    "add_reaction": {
        "description": "Adds an emoji reaction to/from a specific message.",
         "parameters": {
            "type": "object",
            "properties": {
                "adapter_id": {"type": "string", "description": "The ID of the adapter the message exists on."},
                "conversation_id": {"type": "string", "description": "The ID of the conversation containing the message."},
                "message_id": {"type": "string", "description": "The ID of the specific message to react to."},
                "emoji": {"type": "string", "description": "The emoji to add or remove (e.g., '👍', '🎉')."}
            },
            "required": ["adapter_id", "conversation_id", "message_id", "emoji"]
        },
        "execution_info": {
            "type": "action_request",
            "target_module": "ActivityClient",
            "action_type": "add_reaction",
            "payload": {}
        }
    },
    "remove_reaction": {
        "description": "Removes an emoji reaction to/from a specific message.",
         "parameters": {
            "type": "object",
            "properties": {
                "adapter_id": {"type": "string", "description": "The ID of the adapter the message exists on."},
                "conversation_id": {"type": "string", "description": "The ID of the conversation containing the message."},
                "message_id": {"type": "string", "description": "The ID of the specific message to react to."},
                "emoji": {"type": "string", "description": "The emoji to add or remove (e.g., '👍', '🎉')."}
            },
            "required": ["adapter_id", "conversation_id", "message_id", "emoji"]
        },
        "execution_info": {
            "type": "action_request",
            "target_module": "ActivityClient",
            "action_type": "remove_reaction",
            "payload": {}
        }
    }
}


# --- Test Cases ---

def test_messaging_tools_initialization(messaging_tools_component, mock_element):
    """Test correct initialization."""
    assert messaging_tools_component.element == mock_element
    assert messaging_tools_component._is_initialized
    assert messaging_tools_component._is_enabled
    assert messaging_tools_component.COMPONENT_TYPE == "messaging_tools"
    assert not messaging_tools_component.DEPENDENCIES # Empty set


def test_register_tools_success(messaging_tools_component, mock_tool_provider):
    """Test successful registration of all messaging tools."""
    messaging_tools_component.register_tools(mock_tool_provider)

    # Verify register_tool was called 5 times
    assert mock_tool_provider.register_tool.call_count == 5

    # Get all the calls made to register_tool
    registration_calls = mock_tool_provider.register_tool.call_args_list

    # Create a dictionary of registered tools from the calls for easier lookup
    registered_tools_data = {}
    for c in registration_calls:
        args, kwargs = c
        tool_name, description, parameters, execution_info = args
        registered_tools_data[tool_name] = {
            "description": description,
            "parameters": parameters,
            "execution_info": execution_info
        }

    # Assert that all expected tools were registered and match definitions
    assert set(registered_tools_data.keys()) == set(EXPECTED_TOOLS.keys())

    for tool_name, expected_data in EXPECTED_TOOLS.items():
        assert tool_name in registered_tools_data
        assert registered_tools_data[tool_name]["description"] == expected_data["description"]
        assert registered_tools_data[tool_name]["parameters"] == expected_data["parameters"]
        assert registered_tools_data[tool_name]["execution_info"] == expected_data["execution_info"]


def test_register_tools_no_provider(messaging_tools_component):
    """Test registration attempt with a None tool provider."""
    with patch('logging.Logger.error') as mock_log_error:
        messaging_tools_component.register_tools(None)
        mock_log_error.assert_called_once_with(
            f"Cannot register messaging tools: ToolProvider component is missing on {messaging_tools_component.element.id}"
        ) 