"""
Integration Test for Scenario 1: Direct Message Ping-Pong
"""
import pytest
import asyncio
import time
from typing import Dict, Any, List, Callable, Optional
import logging # Added for logger in fixture

# Assuming your project structure allows these imports
# You might need to adjust paths based on how pytest discovers your modules
from host.event_loop import HostEventLoop, OutgoingActionCallback
from host.routing.external_event_router import ExternalEventRouter
from host.config import AgentConfig, LLMConfig
from elements.space_registry import SpaceRegistry
from elements.elements.inner_space import InnerSpace
from elements.elements.space import Space # For SharedSpace
from elements.elements.uplink import UplinkProxy # For verifying uplink creation
from elements.elements.agent_loop import BaseAgentLoopComponent # For checking its presence
from elements.elements.components.hud.hud_component import HUDComponent
from elements.elements.components.dm_manager_component import DirectMessageManagerComponent
from elements.elements.components.uplink_manager_component import UplinkManagerComponent # For uplink test
from elements.elements.components.messaging.message_action_handler import MessageActionHandler
from elements.elements.components.messaging.message_list import MessageListComponent
from elements.elements.components.messaging.message_list_veil_producer import MessageListVeilProducer
from elements.elements.components.uplink.cache_component import RemoteStateCacheComponent # For uplink VEIL
from elements.elements.components.uplink.uplink_veil_producer_component import UplinkVeilProducer # For uplink VEIL
from elements.prefabs import PREFABS # To ensure prefabs are loaded
from elements.component_registry import scan_and_load_components
from elements.elements.components.tool_provider import ToolProviderComponent
# Import LLM interface components
from llm.provider_interface import (
    LLMProvider,
    LLMMessage,
    LLMToolDefinition,
    LLMToolCall,
    LLMResponse
)
# LLMProviderFactory is no longer needed here for mock setup
# from llm.provider_factory import LLMProviderFactory

logger = logging.getLogger(__name__) # Added logger

# --- Mock LLM Provider ---
class MockLLMProvider(LLMProvider):
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig(type="mock", default_model="mock_model")
        self.responses: List[LLMResponse] = []
        self.default_response_content = "Mock LLM response."

    def queue_response(self, content: Optional[str] = None, tool_calls: Optional[List[LLMToolCall]] = None, finish_reason: str = "stop"):
        self.responses.append(LLMResponse(content=content, tool_calls=tool_calls, finish_reason=finish_reason))

    def complete(self,
                 messages: List[LLMMessage],
                 model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 tools: Optional[List[LLMToolDefinition]] = None,
                 **kwargs) -> LLMResponse:
        if self.responses:
            return self.responses.pop(0)
        
        # Default behavior if no specific response is queued
        # For the ping-pong, the agent's loop might try to generate a reply.
        # We can make it "smart" enough to use the last user message content if needed, or just a canned reply.
        last_user_message = next((m.content for m in reversed(messages) if m.role == "user"), "User")
        # A simple echo or a fixed reply. For ping-pong, a fixed reply is fine.
        # response_content = f"Mock reply to: {last_user_message}"
        response_content = self.default_response_content
        
        logger.debug(f"MockLLMProvider: complete() called. Last user message: '{last_user_message}'. Responding with: '{response_content}'")
        return LLMResponse(content=response_content, finish_reason="stop")

    def format_tool_for_provider(self, tool: Dict[str, Any]) -> Any:
        # For a mock, we might not need complex formatting.
        # Just return the tool definition or a simplified version.
        logger.debug(f"MockLLMProvider: format_tool_for_provider() called for tool: {tool.get('name')}")
        return {
            "name": tool.get("name"),
            "description": tool.get("description"),
            "parameters": tool.get("parameters_schema") # Assuming parameters_schema is what's passed
        }

    def parse_response(self, raw_response: Any) -> LLMResponse:
        # If raw_response is already an LLMResponse in mock scenarios, just return it.
        # Otherwise, adapt based on what complete() might return if it were less direct.
        logger.debug(f"MockLLMProvider: parse_response() called with: {raw_response}")
        if isinstance(raw_response, LLMResponse):
            return raw_response
        # Fallback if raw_response is something else (e.g. a dict)
        return LLMResponse(content=str(raw_response), finish_reason="stop")


# --- Mock ActivityClient ---
class MockActivityClient:
    def __init__(self): # Removed host_event_loop_enqueue_callback from init
        self.host_event_loop_enqueue_callback: Optional[Callable] = None
        self.outgoing_actions_received: List[Dict[str, Any]] = []
        self.outgoing_action_responses: List[Optional[Dict[str, Any]]] = [] # To simulate adapter responses

    def set_enqueue_callback(self, callback: Callable):
        """Sets the callback for enqueuing events into the HostEventLoop."""
        self.host_event_loop_enqueue_callback = callback

    def simulate_incoming_event(self, event_data: Dict[str, Any]):
        """Simulates an event coming from an adapter into the HostEventLoop."""
        if not self.host_event_loop_enqueue_callback:
            logger.error("MockActivityClient: enqueue_callback not set. Cannot simulate event.")
            return
        self.host_event_loop_enqueue_callback(event_data, {})
        print(f"MockActivityClient: Simulated incoming event: {event_data['payload']['event_type_from_adapter']}")

    def handle_outgoing_action(self, action_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Captures outgoing actions that would normally go to an external adapter.
        This method signature matches OutgoingActionCallback.
        """
        print(f"MockActivityClient: Received outgoing action: {action_request['action_type']} with payload {action_request.get('payload')}")
        self.outgoing_actions_received.append(action_request)
        
        if self.outgoing_action_responses:
            return self.outgoing_action_responses.pop(0)
        return {"success": True, "message_id": "mock_external_msg_id_123"} 

    def queue_outgoing_action_response(self, response: Optional[Dict[str, Any]]):
        self.outgoing_action_responses.append(response)

    def clear_received_actions(self):
        self.outgoing_actions_received = []
        self.outgoing_action_responses = []

# --- Test Agent Configuration ---
TEST_AGENT_ID = "test_agent_dm"
TEST_ADAPTER_ID = "mock_dm_adapter"
TEST_SHARED_ADAPTER_ID = "mock_shared_adapter" # For scenario 2

# Agent config now also handles the shared adapter for Scenario 2
test_agent_config = AgentConfig(
    agent_id=TEST_AGENT_ID,
    name="Test DM Agent",
    description="Agent for testing DM and SharedSpace ping-pong.",
    system_prompt_template="You are a test agent. Please reply with 'Pong'.", # Make prompt specific for test
    handles_direct_messages_from_adapter_ids=[TEST_ADAPTER_ID, TEST_SHARED_ADAPTER_ID]
)

# --- Pytest Fixtures ---
@pytest.fixture
def setup_test_environment():
    scan_and_load_components() # Renamed for broader use
    space_registry = SpaceRegistry()
    
    # Instantiate MockActivityClient first
    mock_activity_client = MockActivityClient()

    # Instantiate HostEventLoop with all required arguments
    event_loop = HostEventLoop(
        host_router=None, # Following host/main.py pattern
        activity_client=mock_activity_client,
        external_event_router=None, # Will be set later
        space_registry=space_registry,
        agent_configs=[test_agent_config]
    )

    # Set the enqueue callback for MockActivityClient
    mock_activity_client.set_enqueue_callback(event_loop.enqueue_incoming_event)

    # Create LLMConfig for mock provider
    mock_llm_config = LLMConfig(type="mock_llm", default_model="test_model")
    
    # Instantiate our new MockLLMProvider directly
    llm_provider = MockLLMProvider(config=mock_llm_config)
    # We can set a specific response for the ping-pong test if needed
    llm_provider.default_response_content = "Pong"


    external_event_router = ExternalEventRouter(
        space_registry=space_registry,
        mark_agent_for_cycle_callback=event_loop.mark_agent_for_cycle,
        agent_configs=[test_agent_config]
    )
    event_loop.external_event_router = external_event_router

    agent_inner_space = InnerSpace(
        element_id=f"inner_space_{TEST_AGENT_ID}",
        name=f"InnerSpace for {TEST_AGENT_ID}",
        description="Test InnerSpace",
        agent_id=TEST_AGENT_ID,
        llm_provider=llm_provider,
        system_prompt_template=test_agent_config.system_prompt_template,
        outgoing_action_callback=mock_activity_client.handle_outgoing_action, 
        space_registry=space_registry,
        mark_agent_for_cycle_callback=event_loop.mark_agent_for_cycle
    )
    space_registry.register_inner_space(agent_inner_space, TEST_AGENT_ID)
    assert agent_inner_space.get_dm_manager() is not None
    assert agent_inner_space.get_uplink_manager() is not None
    assert agent_inner_space.agent_id is not None

    return event_loop, mock_activity_client, space_registry, agent_inner_space

# --- Test Function (Scenario 1) ---
@pytest.mark.asyncio
async def test_dm_ping_pong(setup_test_environment):
    event_loop, mock_activity_client, space_registry, agent_inner_space = setup_test_environment

    dm_sender_external_id = "user_external_123"
    dm_sender_display_name = "DM User"
    incoming_dm_text = "Hello Agent!"
    dm_conversation_id_for_adapter = dm_sender_external_id 

    incoming_dm_payload = {
        "source_adapter_id": TEST_ADAPTER_ID,
        "payload": {
            "event_type_from_adapter": "message_received",
            "adapter_data": {
                "message_id": "external_msg_001",
                "conversation_id": dm_conversation_id_for_adapter, 
                "is_direct_message": True, 
                "sender": {"user_id": dm_sender_external_id, "display_name": dm_sender_display_name},
                "text": incoming_dm_text,
                "timestamp": time.time(),
                "recipient_connectome_agent_id": TEST_AGENT_ID 
            }
        }
    }
    mock_activity_client.simulate_incoming_event(incoming_dm_payload)
    await event_loop._process_incoming_event_queue()
    await asyncio.sleep(0.01) 

    dm_manager = agent_inner_space.get_dm_manager()
    assert dm_manager is not None
    dm_element = dm_manager.get_dm_element_for_user(TEST_ADAPTER_ID, dm_sender_external_id)
    assert dm_element is not None, f"DM Element for user {dm_sender_external_id} (adapter {TEST_ADAPTER_ID}) not found via DMManager."
    assert agent_inner_space.get_mounted_element(dm_element.id) is dm_element

    msg_list_comp = dm_element.get_component_by_type(MessageListComponent)
    assert msg_list_comp is not None
    messages = msg_list_comp.get_messages()
    assert len(messages) == 1
    assert messages[0]['text'] == incoming_dm_text
    assert messages[0]['sender_id'] == dm_sender_external_id

    reply_text = "Hello User! I am Test Agent."
    tool_provider_on_dm_element = dm_element.get_component_by_type(ToolProviderComponent)
    assert tool_provider_on_dm_element is not None

    action_result = agent_inner_space.execute_action_on_element(
        element_id=dm_element.id, 
        action_name="send_message",
        parameters={"text": reply_text}
    )
    assert action_result['success'] is True, f"send_message tool execution failed: {action_result.get('error')}"

    assert len(mock_activity_client.outgoing_actions_received) == 1
    outgoing_action = mock_activity_client.outgoing_actions_received[0]
    assert outgoing_action['action_type'] == "send_message"
    outgoing_payload = outgoing_action['payload']
    assert outgoing_payload['adapter_id'] == TEST_ADAPTER_ID
    assert outgoing_payload['conversation_id'] == dm_sender_external_id 
    assert outgoing_payload['text'] == reply_text
    assert outgoing_payload['requesting_element_id'] == dm_element.id
    assert outgoing_payload['requesting_agent_id'] == agent_inner_space.agent_id

    print(f"test_dm_ping_pong ({TEST_AGENT_ID}) PASSED!")

# --- Test Function (Scenario 2) ---
@pytest.mark.asyncio
async def test_shared_space_mention_reply(setup_test_environment):
    event_loop, mock_activity_client, space_registry, agent_inner_space = setup_test_environment
    mock_activity_client.clear_received_actions() # Clear actions from previous tests if fixture is session-scoped

    # --- Phase 1: Setup SharedSpace and Simulate Incoming Mention ---
    shared_channel_id = "channel_789"
    shared_channel_name = "Test Shared Channel"
    message_sender_external_id = "user_external_456"
    message_sender_display_name = "Channel User"
    incoming_mention_text = f"Hello @{TEST_AGENT_ID}, can you help?"

    # Simulate incoming message to a shared channel that mentions the agent
    # ExternalEventRouter should create the SharedSpace and ensure an Uplink for the agent.
    mention_payload = {
        "source_adapter_id": TEST_SHARED_ADAPTER_ID,
        "payload": {
            "event_type_from_adapter": "message_received",
            "adapter_data": {
                "message_id": "external_msg_002",
                "conversation_id": shared_channel_id,
                "channel_name": shared_channel_name,
                "is_direct_message": False,
                "sender": {"user_id": message_sender_external_id, "display_name": message_sender_display_name},
                "text": incoming_mention_text,
                "timestamp": time.time(),
                "mentions": [{"user_id": TEST_AGENT_ID, "display_name": test_agent_config.name}]
            }
        }
    }
    mock_activity_client.simulate_incoming_event(mention_payload)
    await event_loop._process_incoming_event_queue()
    await asyncio.sleep(0.01) # Allow async tasks

    # Assertions for SharedSpace and Uplink creation:
    # 1. SharedSpace should exist
    shared_space_id = f"shared_{TEST_SHARED_ADAPTER_ID}_{shared_channel_id}"
    shared_space = space_registry.get_space(shared_space_id)
    assert shared_space is not None, f"SharedSpace {shared_space_id} not created."
    assert shared_space.name == shared_channel_name
    assert shared_space.adapter_id == TEST_SHARED_ADAPTER_ID
    assert shared_space.external_conversation_id == shared_channel_id

    # 2. Message should be in SharedSpace (assuming it has a MessageListComponent, e.g., via a chat element prefab)
    #    This part is a bit more complex as SharedSpace itself doesn't directly hold MessageListComponent.
    #    It would be on a mounted element within SharedSpace. For this test, we'll assume the event reached it.
    #    A more detailed test would check the mounted chat element's MessageListComponent.
    #    For now, we'll trust receive_event on SharedSpace processed it.

    # 3. UplinkProxy should be created in InnerSpace
    uplink_manager = agent_inner_space.get_uplink_manager()
    assert uplink_manager is not None
    uplink_element = uplink_manager.get_uplink_for_space(shared_space_id)
    assert uplink_element is not None, f"UplinkProxy for SharedSpace {shared_space_id} not found."
    assert isinstance(uplink_element, UplinkProxy)
    assert uplink_element.remote_space_id == shared_space_id

    # --- Phase 1.5: Verify message in SharedSpace's MessageListComponent ---
    # Assuming "standard_shared_space_chat" prefab mounts an element with ID suffix "chat_interface"
    # and this element is configured in "standard_uplink_proxy" as the target_element_id_in_remote_space.
    mounted_chat_element_id_suffix = "chat_interface" 
    mounted_chat_element_id = f"{shared_space.id}_{mounted_chat_element_id_suffix}"
    
    # Process events again to ensure any element initialisation events from prefab are handled.
    await event_loop._process_incoming_event_queue()
    await asyncio.sleep(0.01)

    chat_element_in_shared_space = shared_space.get_mounted_element(mounted_chat_element_id)
    assert chat_element_in_shared_space is not None, (
        f"Chat element '{mounted_chat_element_id}' not found in SharedSpace '{shared_space.id}'. "
        f"Mounted elements: {list(shared_space._mounted_elements.keys())}"
    )

    ss_msg_list_comp = chat_element_in_shared_space.get_component(MessageListComponent)
    assert ss_msg_list_comp is not None, (
        f"MessageListComponent not found on chat element '{chat_element_in_shared_space.id}' in SharedSpace."
    )
    
    shared_space_messages = ss_msg_list_comp.get_messages()
    assert len(shared_space_messages) == 1, (
        f"Expected 1 message in SharedSpace's chat element, found {len(shared_space_messages)}. Messages: {shared_space_messages}"
    )
    assert shared_space_messages[0]['text'] == incoming_mention_text
    assert shared_space_messages[0]['sender_id'] == message_sender_external_id
    internal_msg_id_in_shared_space = shared_space_messages[0]['internal_id']

    # --- Phase 2: Simulate Agent "seeing" the message via Uplink and replying ---
    # This requires the UplinkProxy's VeilProducer to work, which in turn uses RemoteStateCacheComponent.
    # For an integration test, we might assume the VEIL is correctly produced or test it separately.
    # Here, we'll focus on the action of replying via the UplinkProxy.

    # Verify the VEIL produced by the UplinkProxy
    uplink_veil_producer = uplink_element.get_component(UplinkVeilProducer)
    assert uplink_veil_producer is not None, "UplinkVeilProducerComponent not found on UplinkProxy."

    # Process one cycle for the agent's InnerSpace which might trigger cache updates or component initializations.
    agent_inner_space.process_cycle() 
    await asyncio.sleep(0.01) # Allow async tasks within process_cycle

    uplink_veil = uplink_veil_producer.get_full_veil()
    assert uplink_veil is not None, "UplinkProxy failed to produce a VEIL."

    # Assert the content of the VEIL.
    # The UplinkVeilProducer wraps the remote element's VEIL.
    # Assumed VEIL_UPLINK_WRAPPER_NODE_TYPE, replace if UplinkVeilProducer defines a constant
    UPLINK_WRAPPER_NODE_TYPE_EXPECTED = "uplinked_content_container" 
    assert uplink_veil.get('node_type') == UPLINK_WRAPPER_NODE_TYPE_EXPECTED, (
        f"Uplink VEIL root node type is not '{UPLINK_WRAPPER_NODE_TYPE_EXPECTED}', got {uplink_veil.get('node_type')}"
    )
    assert 'children' in uplink_veil and len(uplink_veil['children']) == 1, (
        f"Uplink VEIL does not have the expected structure (single child for wrapped content). Children: {uplink_veil.get('children')}"
    )
    
    remote_space_root_veil = uplink_veil['children'][0]
    assert remote_space_root_veil.get('node_type') == MessageListVeilProducer.VEIL_CONTAINER_TYPE, (
        f"Wrapped VEIL node type is not '{MessageListVeilProducer.VEIL_CONTAINER_TYPE}', got {remote_space_root_veil.get('node_type')}"
    )
    
    found_message_in_veil = False
    if 'children' in remote_space_root_veil:
        for veil_msg_node in remote_space_root_veil['children']:
            if (veil_msg_node.get('node_type') == MessageListVeilProducer.VEIL_MESSAGE_NODE_TYPE and
               veil_msg_node.get('veil_id') == internal_msg_id_in_shared_space):
                assert veil_msg_node['properties'].get(MessageListVeilProducer.VEIL_CONTENT_PROP) == incoming_mention_text
                found_message_in_veil = True
                break
    
    assert found_message_in_veil, (
        f"Message '{incoming_mention_text}' (internal_id: {internal_msg_id_in_shared_space}) not found in Uplink VEIL. VEIL: {uplink_veil}"
    )
        
    reply_text_to_channel = f"Hello {message_sender_display_name}! Yes, I can help."

    tool_provider_on_uplink = uplink_element.get_component("ToolProviderComponent")
    assert tool_provider_on_uplink is not None, "ToolProviderComponent not found on UplinkProxy."
    
    # MessageActionHandler on UplinkProxy should have send_message
    action_result = agent_inner_space.execute_action_on_element(
        element_id=uplink_element.id, # Target the UplinkProxy element
        action_name="send_message",
        parameters={"text": reply_text_to_channel}
    )
    assert action_result['success'] is True, f"send_message via UplinkProxy failed: {action_result.get('error')}"

    # --- Phase 3: Assertions for outgoing message from Uplink ---
    assert len(mock_activity_client.outgoing_actions_received) == 1, "Outgoing message from Uplink not captured."
    outgoing_action = mock_activity_client.outgoing_actions_received[0]
    assert outgoing_action['action_type'] == "send_message"
    outgoing_payload = outgoing_action['payload']
    assert outgoing_payload['adapter_id'] == TEST_SHARED_ADAPTER_ID
    assert outgoing_payload['conversation_id'] == shared_channel_id # Should be the channel ID
    assert outgoing_payload['text'] == reply_text_to_channel
    assert outgoing_payload['requesting_element_id'] == uplink_element.id
    # requesting_agent_id for actions from UplinkProxy (which is in InnerSpace) should be the InnerSpace's agent_id
    assert outgoing_payload['requesting_agent_id'] == agent_inner_space.agent_id

    print(f"test_shared_space_mention_reply ({TEST_AGENT_ID} in {shared_channel_id}) PASSED!") 