"""
Integration Test for Scenario 1: Direct Message Ping-Pong
"""
import pytest
import asyncio
import time
from typing import Dict, Any, List, Callable, Optional
import logging # Added for logger in fixture
import inspect

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
from elements.elements.components.chat_manager_component import ChatManagerComponent
from elements.elements.components.uplink_manager_component import UplinkManagerComponent # For uplink test
from elements.elements.components.uplink.remote_tool_provider import UplinkRemoteToolProviderComponent
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
        self._next_send_should_fail = False

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

    async def handle_outgoing_action(self, action_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Captures outgoing actions that would normally go to an external adapter.
        This method signature matches OutgoingActionCallback.
        Simulates the new multi-stage confirmation for send_message.
        """
        print(f"MockActivityClient: Received outgoing action: {action_request['action_type']} with payload {action_request.get('payload')}")
        self.outgoing_actions_received.append(action_request)

        # Default response to caller (MessageActionHandler) - confirms receipt by ActivityClient
        response_to_caller = {"success": True, "message_id": f"mock_ac_receipt_id_{len(self.outgoing_actions_received)}"}
        
        # If it's a send_message action, simulate the adapter's confirmation/failure event flow
        if action_request['action_type'] == "send_message" and self.host_event_loop_enqueue_callback:
            outgoing_payload = action_request.get('payload', {})
            internal_request_id = outgoing_payload.get('internal_request_id')
            original_conversation_id = outgoing_payload.get('conversation_id')
            original_adapter_id = outgoing_payload.get('adapter_id')
            # The requesting_element_id is the DMElement or UplinkProxy where MessageListComponent lives
            target_element_id_for_confirmation = outgoing_payload.get('requesting_element_id')

            if not internal_request_id:
                logger.error("MockActivityClient: 'internal_request_id' missing in send_message payload. Cannot simulate ack.")
                return response_to_caller # Still return success for AC receipt
            
            # Determine if this should be a failure based on test setup
            should_fail = self._next_send_should_fail
            self._next_send_should_fail = False # Reset for next call

            # Simulate enqueuing an event back to HostEventLoop as if from an adapter
            # This event will be processed by ExternalEventRouter
            simulated_adapter_event_data: Dict[str, Any]
            if not should_fail:
                simulated_adapter_event_data = {
                    "source_adapter_id": original_adapter_id, # The adapter that "sent" it
                    "payload": { # This is what ExternalEventRouter expects as top-level payload from ActivityClient
                        "event_type_from_adapter": "adapter_send_success_ack", # Type EER will check
                        "adapter_data": { # Data specific to this ack type
                            "internal_request_id": internal_request_id,
                            "conversation_id": original_conversation_id,
                            # 'is_direct_message' would be needed if EER routing for acks relied on it here,
                            # but we are directly using target_element_id_for_confirmation
                            "target_element_id_for_confirmation": target_element_id_for_confirmation,
                            "external_message_ids": ["external_msg_001.1"],
                            "confirmed_timestamp": time.time()
                        }
                    }
                }
                logger.info(f"MockActivityClient: Simulating 'adapter_send_success_ack' for req_id: {internal_request_id}")
            else: # Simulate failure
                simulated_adapter_event_data = {
                    "source_adapter_id": original_adapter_id,
                    "payload": {
                        "event_type_from_adapter": "adapter_send_failure_ack",
                        "adapter_data": {
                            "internal_request_id": internal_request_id,
                            "conversation_id": original_conversation_id,
                            "target_element_id_for_confirmation": target_element_id_for_confirmation,
                            "error_message": "Simulated: Failed to send message due to external adapter error.",
                            "failed_timestamp": time.time()
                        }
                    }
                }
                logger.info(f"MockActivityClient: Simulating 'adapter_send_failure_ack' for req_id: {internal_request_id}")

            # The timeline_context for these adapter-sourced events is usually minimal or constructed by HEL/EER.
            # For HostEventLoop.enqueue_incoming_event, it expects (event_data, timeline_context)
            # Let's pass a simple timeline_context or None, as EER will build what it needs.
            self.host_event_loop_enqueue_callback(simulated_adapter_event_data, {"source": "mock_adapter_ack"})
            logger.debug(f"MockActivityClient: Enqueued simulated adapter ack for req_id {internal_request_id} to HostEventLoop.")
            
            # The old echo logic is now replaced by this ack simulation.
            # The MAH's call to add_pending_message + this ack flow handles agent's own messages.

        return response_to_caller # Return immediate success for AC accepting the request

    def prepare_to_fail_next_send(self):
        """Sets a flag to make the next send_message action simulate a failure."""
        self._next_send_should_fail = True

    async def start_connections(self): # Keep async for consistency
        pass

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
    from elements.component_registry import COMPONENT_REGISTRY

    space_registry = SpaceRegistry.get_instance()
    
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
        agent_name=test_agent_config.name,
        description="Test InnerSpace",
        agent_id=TEST_AGENT_ID,
        llm_provider=llm_provider,
        system_prompt_template=test_agent_config.system_prompt_template,
        outgoing_action_callback=event_loop.get_outgoing_action_callback(),
        mark_agent_for_cycle_callback=event_loop.mark_agent_for_cycle
    )
    space_registry.register_inner_space(agent_inner_space, TEST_AGENT_ID)
    space_registry.response_callback = event_loop.get_outgoing_action_callback()
    assert agent_inner_space.get_component_by_type(ChatManagerComponent) is not None
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

    chat_manager = agent_inner_space.get_component_by_type(ChatManagerComponent)
    assert chat_manager is not None
    dm_element = chat_manager.get_dm_element_for_user(TEST_ADAPTER_ID, dm_sender_external_id)
    assert dm_element is not None, f"DM Element for user {dm_sender_external_id} (adapter {TEST_ADAPTER_ID}) not found via ChatManager."
    assert agent_inner_space.get_mounted_element(dm_element.mount_id) is dm_element

    msg_list_comp = dm_element.get_component_by_type(MessageListComponent)
    assert msg_list_comp is not None
    messages = msg_list_comp.get_messages()
    assert len(messages) == 1
    assert messages[0]['text'] == incoming_dm_text
    assert messages[0]['sender_id'] == dm_sender_external_id

    # --- Agent sends a reply ---
    reply_text = "Hello User! I am Test Agent." # Original reply text
    # MessageActionHandler is on dm_element, and execute_action_on_element is on InnerSpace
    tool_provider_on_dm_element = dm_element.get_component_by_type(ToolProviderComponent)
    assert tool_provider_on_dm_element is not None

    # Agent uses the send_message tool on the DMElement
    action_result = await agent_inner_space.execute_action_on_element(
        element_id=dm_element.mount_id, 
        action_name="send_message",
        parameters={"text": reply_text}
    )
    assert action_result['success'] is True, f"send_message tool execution failed: {action_result.get('error')}"
    assert action_result['status'] == "pending_confirmation", "send_message tool should return 'pending_confirmation'"
    internal_req_id_agent_reply = action_result.get('internal_request_id')
    assert internal_req_id_agent_reply is not None, "send_message tool did not return an internal_request_id"

    # Check MessageListComponent: Agent's message should be 'pending_send'
    messages_before_ack = msg_list_comp.get_messages()
    assert len(messages_before_ack) == 2, "Expected 2 messages (incoming + pending outgoing)"
    assert messages_before_ack[1]['text'] == reply_text
    assert messages_before_ack[1]['status'] == "pending_send", "Agent's reply should be 'pending_send' before ack"
    assert messages_before_ack[1]['internal_request_id'] == internal_req_id_agent_reply
    assert messages_before_ack[1]['sender_id'] == TEST_AGENT_ID # Agent is the sender

    # Process the outgoing action queue so MockActivityClient receives the action
    await event_loop._process_outgoing_action_queue()
    await asyncio.sleep(0.01) # Allow time for async processing

    # Verify outgoing action was received by MockActivityClient
    assert len(mock_activity_client.outgoing_actions_received) == 1
    outgoing_action = mock_activity_client.outgoing_actions_received[0]
    assert outgoing_action['action_type'] == "send_message"
    outgoing_payload = outgoing_action['payload']
    assert outgoing_payload['adapter_id'] == TEST_ADAPTER_ID
    assert outgoing_payload['conversation_id'] == dm_sender_external_id 
    assert outgoing_payload['text'] == reply_text
    assert outgoing_payload['requesting_element_id'] == dm_element.id
    assert outgoing_payload['requesting_agent_id'] == agent_inner_space.agent_id
    assert outgoing_payload['internal_request_id'] == internal_req_id_agent_reply

    # Process the event queue again to handle the mocked adapter acknowledgment
    # MockActivityClient's handle_outgoing_action simulates the adapter_send_success_ack
    await event_loop._process_incoming_event_queue()
    await asyncio.sleep(0.01) # Allow time for event processing

    # Check MessageListComponent again: Agent's message should now be 'sent'
    messages_after_reply_ack = msg_list_comp.get_messages()
    assert len(messages_after_reply_ack) == 2, f"Expected 2 messages after agent reply ack, got {len(messages_after_reply_ack)}"
    
    # Original incoming message
    assert messages_after_reply_ack[0]['text'] == incoming_dm_text 
    assert messages_after_reply_ack[0]['sender_id'] == dm_sender_external_id

    # Agent's reply, now confirmed
    assert messages_after_reply_ack[1]['text'] == reply_text 
    assert messages_after_reply_ack[1]['sender_id'] == TEST_AGENT_ID # Agent is the sender
    assert messages_after_reply_ack[1]['status'] == "sent", "Agent's reply should be 'sent' after ack"
    assert messages_after_reply_ack[1]['internal_request_id'] == internal_req_id_agent_reply
    assert messages_after_reply_ack[1]['original_external_id'] == f"external_msg_001.1", "External message ID mismatch after ack"

    # --- Simulate a second incoming message from the same user ---
    second_incoming_dm_text = "Thanks for the reply!"
    second_incoming_dm_payload = {
        "source_adapter_id": TEST_ADAPTER_ID,
        "payload": {
            "event_type_from_adapter": "message_received",
            "adapter_data": {
                "message_id": "external_msg_002", # New external ID for the second message
                "conversation_id": dm_conversation_id_for_adapter, 
                "is_direct_message": True, 
                "sender": {"user_id": dm_sender_external_id, "display_name": dm_sender_display_name},
                "text": second_incoming_dm_text,
                "timestamp": time.time() + 1, # Ensure a later timestamp
                "recipient_connectome_agent_id": TEST_AGENT_ID 
            }
        }
    }
    mock_activity_client.simulate_incoming_event(second_incoming_dm_payload)
    await event_loop._process_incoming_event_queue()
    await asyncio.sleep(0.01)

    # Assert that the same DMElement is used and the new message is added
    dm_element_after_second_msg = chat_manager.get_dm_element_for_user(TEST_ADAPTER_ID, dm_sender_external_id)
    assert dm_element_after_second_msg is dm_element, "ChatManager created a new DMElement for a subsequent message instead of reusing."
    
    messages_after_second_incoming = msg_list_comp.get_messages()
    assert len(messages_after_second_incoming) == 3, f"Expected 3 messages after second incoming DM, got {len(messages_after_second_incoming)}"
    assert messages_after_second_incoming[0]['text'] == incoming_dm_text
    assert messages_after_second_incoming[1]['text'] == reply_text
    assert messages_after_second_incoming[2]['text'] == second_incoming_dm_text
    assert messages_after_second_incoming[2]['sender_id'] == dm_sender_external_id

    # --- Test VEIL production from DMElement ---
    # Call process_cycle on the InnerSpace to simulate end-of-frame processing,
    # which includes on_frame_end for its TimelineComponent.
    agent_inner_space.on_frame_end()
    await asyncio.sleep(0.1) # Allow async tasks from process_cycle to complete

    veil_producer = dm_element.get_component_by_type("MessageListVeilProducer")
    assert veil_producer is not None, "MessageListVeilProducer not found on DMElement."
    
    dm_veil = veil_producer.get_full_veil()
    assert dm_veil is not None, "DMElement's MessageListVeilProducer failed to produce a VEIL."
    assert dm_veil.get('node_type') == "message_list", f"VEIL node type mismatch: {dm_veil.get('node_type')}"
    
    veil_messages = dm_veil.get('children', [])
    assert len(veil_messages) == 3, f"Expected 3 messages in VEIL, got {len(veil_messages)}"

    assert veil_messages[0]['properties'].get('text_content') == incoming_dm_text
    assert veil_messages[0]['properties'].get('sender_name') == dm_sender_display_name
    assert veil_messages[1]['properties'].get('text_content') == reply_text
    assert veil_messages[1]['properties'].get('sender_name') == test_agent_config.name # Agent's name
    assert veil_messages[2]['properties'].get('text_content') == second_incoming_dm_text
    assert veil_messages[2]['properties'].get('sender_name') == dm_sender_display_name
    
    inner_space_veil_producer = agent_inner_space.get_component_by_type("SpaceVeilProducer")
    assert inner_space_veil_producer is not None, "SpaceVeilProducer not found on InnerSpace."
    inner_space_veil = inner_space_veil_producer.get_full_veil()
    assert inner_space_veil is not None, "InnerSpace's VEIL not found."

    hud = agent_inner_space.get_component_by_type("HUDComponent")
    assert hud is not None, "HUDComponent not found on InnerSpace."
    agent_context = await hud.get_agent_context()
    assert agent_context is not None, "Agent context not found on InnerSpace."
    agent_loop = agent_inner_space.get_component_by_type("MultiStepToolLoopComponent")
    assert agent_loop is not None, "AgentLoop not found on InnerSpace."
    tools = await agent_loop.aggregate_tools()
    assert len(tools) > 0, "No tools found on InnerSpace."

    print(f"test_dm_ping_pong ({TEST_AGENT_ID}) PASSED!")

# #--- Test Function (Scenario 2) ---
@pytest.mark.asyncio
async def test_shared_space_mention_reply(setup_test_environment):
    event_loop, mock_activity_client, space_registry, agent_inner_space = setup_test_environment
    mock_activity_client.clear_received_actions()

    # --- Phase 1: Setup SharedSpace and Simulate Incoming Mention ---
    shared_channel_id = "channel_789"
    shared_channel_name = "Test Shared Channel"
    message_sender_external_id = "user_external_456"
    message_sender_display_name = "Channel User"
    incoming_mention_text = f"Hello @{test_agent_config.name}, can you help?" # Use agent's configured name

    mention_payload = {
        "source_adapter_id": TEST_SHARED_ADAPTER_ID,
        "payload": {
            "event_type_from_adapter": "message_received",
            "adapter_data": {
                "message_id": "external_shared_msg_001", # More descriptive ID
                "conversation_id": shared_channel_id,
                "channel_name": shared_channel_name,
                "is_direct_message": False,
                "sender": {"user_id": message_sender_external_id, "display_name": message_sender_display_name},
                "text": incoming_mention_text,
                "timestamp": time.time(),
                # Use agent's name for mention text, but agent_id for structured mention if adapter provides
                "mentions": [{"user_id": TEST_AGENT_ID, "display_name": test_agent_config.name}]
            }
        }
    }
    mock_activity_client.simulate_incoming_event(mention_payload)
    await event_loop._process_incoming_event_queue()
    await asyncio.sleep(0.01)

    shared_space_id = f"shared_{TEST_SHARED_ADAPTER_ID}_{shared_channel_id}"
    shared_space = space_registry.get_space(shared_space_id)
    assert shared_space is not None, f"SharedSpace {shared_space_id} not created."
    assert shared_space.name == shared_channel_name

    uplink_manager = agent_inner_space.get_uplink_manager()
    assert uplink_manager is not None
    uplink_element = uplink_manager.get_uplink_for_space(shared_space_id)
    assert uplink_element is not None, f"UplinkProxy for SharedSpace {shared_space_id} not found."
    assert isinstance(uplink_element, UplinkProxy)

    mounted_chat_element_id_suffix = "chat_interface" 
    mounted_chat_element_id = f"{shared_space.id}_{mounted_chat_element_id_suffix}"
    
    await event_loop._process_incoming_event_queue()
    await asyncio.sleep(0.01)

    chat_element_in_shared_space = shared_space.get_mounted_element(mounted_chat_element_id)
    assert chat_element_in_shared_space is not None, (
        f"Chat element '{mounted_chat_element_id}' not found in SharedSpace '{shared_space.id}'. "
        f"Mounted elements: {list(shared_space.get_mounted_elements().keys())}"
    )
    ss_msg_list_comp = chat_element_in_shared_space.get_component_by_type(MessageListComponent)
    assert ss_msg_list_comp is not None
    
    shared_space_messages = ss_msg_list_comp.get_messages()
    assert len(shared_space_messages) == 1
    assert shared_space_messages[0]['text'] == incoming_mention_text
    internal_msg_id_in_shared_space = shared_space_messages[0]['internal_id']


    # --- Phase 2: Agent replies to the shared channel via Uplink ---
    reply_text_to_channel = f"Hello @{message_sender_display_name}! Yes, I can help with the shared channel."

    # Get the UplinkRemoteToolProviderComponent from the uplink_element
    urtp_component = uplink_element.get_component_by_type(UplinkRemoteToolProviderComponent)
    assert urtp_component is not None, "UplinkRemoteToolProviderComponent not found on uplink element."

    # Fetch the tools as the LLM would see them (these are prefixed)
    # Note: aggregate_tools in AgentLoopComponent also converts these to LLMToolDefinition objects.
    # Here, we get them raw from URTP to find the correct name.
    # Ensure URTP component is initialized if it fetches tools on init
    if hasattr(urtp_component, 'initialize_component') and inspect.iscoroutinefunction(urtp_component.initialize_component):
        await urtp_component.initialize_component()
    elif hasattr(urtp_component, 'initialize_component'):
        urtp_component.initialize_component()
        
    available_remote_tools_dicts = await urtp_component.get_llm_tool_definitions() # This returns List[Dict]
    
    # Find the prefixed name for send_message targeting the chat_element_in_shared_space
    # The provider_element_id in the tool definition from URTP is the ID of the element *in the SharedSpace*.
    # The chat_element_in_shared_space.id is this ID.
    prefixed_send_message_tool_name = None
    for tool_dict in available_remote_tools_dicts:
        # Expected name format: f"{uplink_element.id}::{chat_element_in_shared_space.id}::send_message"
        # Check if the tool_dict corresponds to the send_message tool from the chat_element_in_shared_space
        # The tool_dict['name'] will already be prefixed by URTPC's get_tools_for_llm
        # The structure of the prefixed name is owner_uplink_id::remote_provider_element_id::actual_tool_name
        name_parts = tool_dict['name'].split('::')
        if len(name_parts) == 3:
            owner_uplink_id, remote_provider_el_id, actual_tool_name = name_parts
            if owner_uplink_id == uplink_element.id and \
               remote_provider_el_id == chat_element_in_shared_space.id and \
               actual_tool_name == "send_message":
                prefixed_send_message_tool_name = tool_dict['name']
                break
    
    assert prefixed_send_message_tool_name is not None, \
        f"Could not find prefixed send_message tool for {chat_element_in_shared_space.id} via uplink {uplink_element.id}. Tools found: {[t['name'] for t in available_remote_tools_dicts]}"

    action_result = await agent_inner_space.execute_action_on_element(
        element_id=uplink_element.id, # Target is the UplinkProxy
        action_name=prefixed_send_message_tool_name, # Use the dynamically found prefixed name
        parameters={"text": reply_text_to_channel}
    )
    assert action_result['success'] is True, f"send_message via UplinkProxy failed: {action_result.get('error')}"
    # internal_req_id_agent_shared_reply = action_result.get('internal_request_id') # This will be None from UplinkRemoteToolProviderComponent
    # assert internal_req_id_agent_shared_reply is not None # This assertion would fail

    # Give a bit more time for the asyncio.create_task in SharedSpace.receive_event to run
    await asyncio.sleep(0.05) # Increased sleep time slightly

    # Process the outgoing action queue so MockActivityClient receives the action from SharedSpace
    await event_loop._process_outgoing_action_queue()
    await asyncio.sleep(0.01) # Allow time for async processing

    # Verify outgoing action to MockActivityClient for agent's reply
    assert len(mock_activity_client.outgoing_actions_received) == 1
    outgoing_action_agent_reply = mock_activity_client.outgoing_actions_received[0]
    assert outgoing_action_agent_reply['action_type'] == "send_message"
    outgoing_payload_agent_reply = outgoing_action_agent_reply['payload']
    assert outgoing_payload_agent_reply['adapter_id'] == TEST_SHARED_ADAPTER_ID
    assert outgoing_payload_agent_reply['conversation_id'] == shared_channel_id
    assert outgoing_payload_agent_reply['text'] == reply_text_to_channel
    
    # Retrieve the internal_request_id from the captured action for later verification
    internal_req_id_agent_shared_reply = outgoing_payload_agent_reply.get('internal_request_id')
    assert internal_req_id_agent_shared_reply is not None, "internal_request_id not found in outgoing action payload captured by MockActivityClient"

    # Process event queue for the adapter_send_success_ack for agent's reply
    await event_loop._process_incoming_event_queue()
    await asyncio.sleep(0.01)

    # Verify agent's message is now in SharedSpace's MessageListComponent and marked 'sent'
    shared_space_messages_after_agent_reply = ss_msg_list_comp.get_messages()
    assert len(shared_space_messages_after_agent_reply) == 2, "Expected 2 messages in shared space after agent reply"
    assert shared_space_messages_after_agent_reply[1]['text'] == reply_text_to_channel
    assert shared_space_messages_after_agent_reply[1]['status'] == "sent"
    assert shared_space_messages_after_agent_reply[1]['internal_request_id'] == internal_req_id_agent_shared_reply

    # Manually trigger on_frame_end for SharedSpace again after the second message
    logger.info(f"Manually calling on_frame_end for SharedSpace ({shared_space.id}) after second incoming message.")
    shared_space.on_frame_end()
    await asyncio.sleep(0.01) # Allow deltas to be processed by UplinkProxy

    

    # --- Phase 3: Simulate a second incoming message in the shared channel ---
    second_sender_external_id = "user_external_789"
    second_sender_display_name = "Another Channel User"
    second_incoming_shared_text = "Thanks for offering help!"

    second_mention_payload = {
        "source_adapter_id": TEST_SHARED_ADAPTER_ID,
        "payload": {
            "event_type_from_adapter": "message_received",
            "adapter_data": {
                "message_id": "external_shared_msg_002",
                "conversation_id": shared_channel_id,
                "channel_name": shared_channel_name,
                "is_direct_message": False,
                "sender": {"user_id": second_sender_external_id, "display_name": second_sender_display_name},
                "text": second_incoming_shared_text,
                "timestamp": time.time() + 5 # Ensure later timestamp
                # No explicit mention of the agent this time
            }
        }
    }
    mock_activity_client.simulate_incoming_event(second_mention_payload)
    await event_loop._process_incoming_event_queue()
    await asyncio.sleep(0.01)

    # Verify the second message is in SharedSpace's MessageListComponent
    shared_space_messages_after_second_incoming = ss_msg_list_comp.get_messages()
    assert len(shared_space_messages_after_second_incoming) == 3, "Expected 3 messages in shared space after second incoming"
    assert shared_space_messages_after_second_incoming[0]['text'] == incoming_mention_text
    assert shared_space_messages_after_second_incoming[1]['text'] == reply_text_to_channel
    assert shared_space_messages_after_second_incoming[2]['text'] == second_incoming_shared_text
    assert shared_space_messages_after_second_incoming[2]['sender_id'] == second_sender_external_id
    internal_msg_id_second_shared_incoming = shared_space_messages_after_second_incoming[2]['internal_id']

    # Manually trigger on_frame_end for SharedSpace again after the second message
    logger.info(f"Manually calling on_frame_end for SharedSpace ({shared_space.id}) after second incoming message.")
    shared_space.on_frame_end()
    await asyncio.sleep(0.01) # Allow deltas to be processed by UplinkProxy


    # --- Phase 4: Verify Agent's Perspective (VEIL, Context, Tools) ---
    # Trigger InnerSpace cycle to update its components, including Uplink's cache/VEIL
    agent_inner_space.on_frame_end() 
    await asyncio.sleep(0.1)

    uplink_veil_producer = uplink_element.get_component_by_type(UplinkVeilProducer)
    assert uplink_veil_producer is not None
    uplink_veil = uplink_veil_producer.get_full_veil()
    assert uplink_veil is not None, "UplinkProxy failed to produce VEIL after second message."

    
    uplink_children = uplink_veil.get('children', [])
    assert len(uplink_children) == 1, \
        f"Uplink VEIL root should have 1 child (the remote space's root), got {len(uplink_children)}"
    
    remote_space_root_in_uplink_veil = uplink_children[0]
    # The remote space's root node_type should be "space_root" (or what SpaceVeilProducer.VEIL_SPACE_ROOT_TYPE is)
    # Let's import SpaceVeilProducer to check its constant if available, otherwise hardcode "space_root"
    from elements.elements.components.space.space_veil_producer import VEIL_SPACE_ROOT_TYPE as RemoteSpaceRootNodeType
    assert remote_space_root_in_uplink_veil.get('node_type') == RemoteSpaceRootNodeType, \
        f"Remote space root node type in Uplink VEIL mismatch. Expected {RemoteSpaceRootNodeType}, got {remote_space_root_in_uplink_veil.get('node_type')}"
    assert remote_space_root_in_uplink_veil.get('veil_id') == f"{shared_space.id}_space_root", \
        f"Remote space root veil_id mismatch. Expected f'{shared_space.id}_space_root', got {remote_space_root_in_uplink_veil.get('veil_id')}"

    # Now find the message_list within the children of the remote_space_root_in_uplink_veil
    remote_space_root_children = remote_space_root_in_uplink_veil.get('children', [])
    
    # Find the chat element's message list VEIL node
    # The chat_element_in_shared_space.id is like "shared_mock_shared_adapter_channel_789_chat_interface"
    # Its MessageListVeilProducer would create a root like "shared_mock_shared_adapter_channel_789_chat_interface_message_list_root"
    expected_chat_message_list_veil_id = f"{chat_element_in_shared_space.id}_message_list_root"
    
    remote_chat_veil = None
    for child_node in remote_space_root_children:
        if child_node.get('veil_id') == expected_chat_message_list_veil_id:
            remote_chat_veil = child_node
            break
    
    assert remote_chat_veil is not None, \
        f"Message list VEIL node '{expected_chat_message_list_veil_id}' not found as a child of the remote space root in Uplink VEIL."

    veil_chat_messages = remote_chat_veil.get('children', [])
    # At this point in the test (after second incoming shared message, agent has replied once), we expect 3 messages total.
    assert len(veil_chat_messages) == 3, \
        f"Expected 3 messages in Uplink VEIL's remote message list after 3rd total message in shared space, got {len(veil_chat_messages)}"
    
    # Check content of messages in VEIL
    assert veil_chat_messages[0]['properties'].get('text_content') == incoming_mention_text
    assert veil_chat_messages[1]['properties'].get('text_content') == reply_text_to_channel
    assert veil_chat_messages[2]['properties'].get('text_content') == second_incoming_shared_text


    # --- Phase 5: Simulate a third incoming message (4th total) in the shared channel, WITH mention ---
    third_user_sender_external_id = "user_external_101112"
    third_user_sender_display_name = "Yet Another Channel User"
    third_incoming_shared_text_with_mention = f"Hey @{test_agent_config.name}, one more thing!"

    fourth_message_payload = {
        "source_adapter_id": TEST_SHARED_ADAPTER_ID,
        "payload": {
            "event_type_from_adapter": "message_received",
            "adapter_data": {
                "message_id": "external_shared_msg_003",
                "conversation_id": shared_channel_id,
                "channel_name": shared_channel_name,
                "is_direct_message": False,
                "sender": {"user_id": third_user_sender_external_id, "display_name": third_user_sender_display_name},
                "text": third_incoming_shared_text_with_mention,
                "timestamp": time.time() + 10, # Ensure later timestamp
                "mentions": [{"user_id": TEST_AGENT_ID, "display_name": test_agent_config.name}]
            }
        }
    }
    mock_activity_client.simulate_incoming_event(fourth_message_payload)
    await event_loop._process_incoming_event_queue()
    await asyncio.sleep(0.01)

    # Verify the fourth message is in SharedSpace's MessageListComponent
    shared_space_messages_after_fourth_incoming = ss_msg_list_comp.get_messages()
    assert len(shared_space_messages_after_fourth_incoming) == 4, "Expected 4 messages in shared space after fourth incoming"
    assert shared_space_messages_after_fourth_incoming[3]['text'] == third_incoming_shared_text_with_mention

    # Manually trigger on_frame_end for SharedSpace again after the fourth message
    logger.info(f"Manually calling on_frame_end for SharedSpace ({shared_space.id}) after fourth incoming message (with mention).")
    shared_space.on_frame_end()
    await asyncio.sleep(0.1) # Allow deltas to be processed by UplinkProxy
    
    agent_inner_space.on_frame_end()
    await asyncio.sleep(0.1)

    # Re-fetch VEIL from UplinkProxy
    uplink_veil_after_fourth = uplink_veil_producer.get_full_veil()
    assert uplink_veil_after_fourth is not None, "UplinkProxy failed to produce VEIL after fourth message."
    
    uplink_children_after_fourth = uplink_veil_after_fourth.get('children', [])
    assert len(uplink_children_after_fourth) == 1, "Uplink VEIL root (after 4th msg) should have 1 child"
    
    remote_space_root_after_fourth = uplink_children_after_fourth[0]
    assert remote_space_root_after_fourth.get('node_type') == RemoteSpaceRootNodeType
    assert remote_space_root_after_fourth.get('veil_id') == f"{shared_space.id}_space_root"

    remote_space_root_children_after_fourth = remote_space_root_after_fourth.get('children', [])
    remote_chat_veil_after_fourth = None
    for child_node in remote_space_root_children_after_fourth:
        if child_node.get('veil_id') == expected_chat_message_list_veil_id: # Same ID as before
            remote_chat_veil_after_fourth = child_node
            break
            
    assert remote_chat_veil_after_fourth is not None, \
        f"Message list VEIL node '{expected_chat_message_list_veil_id}' not found (after 4th msg)."
    assert remote_chat_veil_after_fourth.get('node_type') == 'message_list'
    
    veil_chat_messages_after_fourth = remote_chat_veil_after_fourth.get('children', [])
    assert len(veil_chat_messages_after_fourth) == 4, \
        f"Expected 4 messages in Uplink VEIL's remote message list, got {len(veil_chat_messages_after_fourth)}"

    # Check content of all messages in VEIL
    assert veil_chat_messages_after_fourth[0]['properties'].get('text_content') == incoming_mention_text
    assert veil_chat_messages_after_fourth[1]['properties'].get('text_content') == reply_text_to_channel
    assert veil_chat_messages_after_fourth[2]['properties'].get('text_content') == second_incoming_shared_text
    assert veil_chat_messages_after_fourth[3]['properties'].get('text_content') == third_incoming_shared_text_with_mention

    # Verify InnerSpace can generate its full context (which includes the Uplink's VEIL)
    hud = agent_inner_space.get_component_by_type("HUDComponent")
    assert hud is not None, "HUDComponent not found on InnerSpace for shared space test."
    agent_shared_context = await hud.get_agent_context()
    assert agent_shared_context is not None and len(agent_shared_context) > 0, "Agent context for shared space is missing or empty."
    # Verify agent loop can aggregate tools (which should include tools from the UplinkProxy)
    # Assuming MultiStepToolLoopComponent or similar is used
    agent_loop = agent_inner_space.get_component_by_type("MultiStepToolLoopComponent") 
    assert agent_loop is not None, "AgentLoopComponent not found on InnerSpace for shared space test."
    
    aggregated_tools = await agent_loop.aggregate_tools()
    assert aggregated_tools is not None and len(aggregated_tools) > 0, "No tools aggregated by agent loop for shared space scenario."
    
    # Check if the send_message tool from the UplinkProxy is present
    # Tool names are prefixed with element_id::, e.g., "uplink_proxy_shared_..._789_chat_interface::send_message"
    # Or, if UplinkProxy directly exposes tools without element_id prefix when it's the direct tool provider.
    # For now, we check if *any* send_message tool is available. A more specific check might be needed
    # depending on how ToolProviderComponent and aggregation name tools from mounted elements.
    
    # Looking for a tool that starts with the uplink_element.id and contains "send_message"
    # This depends on the naming convention from aggregate_tools when tools come from mounted elements.
    # A simpler check: ensure at least one send_message tool exists if the naming is complex.
    found_uplink_send_message = False
    for tool_def in aggregated_tools:
        # Example check: if tool name is prefixed like "uplink_element_id::tool_name"
        if tool_def.name.startswith(f"{uplink_element.id}::") and "send_message" in tool_def.name:
            found_uplink_send_message = True
            break
        # Simpler check if tools are not prefixed by default from Uplink's ToolProvider
        elif tool_def.name == "send_message" and uplink_element.get_component_by_type("ToolProviderComponent").has_tool("send_message"):
             # This check implies the tool from Uplink is not prefixed or we need to confirm its source
             # For now, let's assume a general send_message from the context of the uplink is sufficient to find
             # More robustly: check `tool_def.owner_element_id` if LLMToolDefinition stores it.
             # Given current LLMToolDefinition, we can't directly trace back.
             # The test for now will rely on the agent being able to *use* a send_message tool that works for the uplink.
             # The execute_action_on_element in previous steps already tests this.
             # So just checking for general tool availability is a good indicator.
             pass # Covered by the fact that execute_action_on_element worked earlier.

    # We already tested execute_action_on_element for the uplink.
    # The critical part is that the agent_context (HUD output) and available tools (from agent_loop) are generated.
    logger.info(f"Agent context for shared space (first 200 chars): {agent_shared_context[:200]}")
    logger.info(f"Aggregated tools for shared space (count: {len(aggregated_tools)}): {[t.name for t in aggregated_tools]}")

    print(f"test_shared_space_mention_reply ({TEST_AGENT_ID} in {shared_channel_id}) PASSED!")

# --- NEW TEST CASE for Send Failure ---
# @pytest.mark.asyncio
# async def test_dm_send_failure(setup_test_environment):
#     """
#     Tests that if an outgoing DM fails at the adapter level, MessageListComponent reflects this.
#     """
#     event_loop, mock_activity_client, space_registry, agent_inner_space = setup_test_environment
#     mock_activity_client.prepare_to_fail_next_send()

#     # 2. Agent tries to send a message (e.g. proactively, or in response to something)
#     # For simplicity, let's have it send a message to a known DM partner without prior incoming.
#     # This requires DMElement to be pre-created or creatable by send_message_tool if no prior chat.
#     # Let's assume DMManager can ensure/create the DMElement for the agent to send.
#     target_user_id_for_failed_dm = "user_failed_dm_target"
    
#     # Ensure DMManager can create the DM element
#     # First, ensure FactoryComponent is present on InnerSpace
#     factory_comp = agent_inner_space.get_component_by_type("ElementFactoryComponent")
#     if not factory_comp:
#         agent_inner_space.add_component(ElementFactoryComponent) # Add if missing
#         logger.info("Added ElementFactoryComponent to agent_inner_space for failure test")
    
#     # This call should create the DMElement
#     dm_element = dm_manager.ensure_dm_element_for_user(target_user_id_for_failed_dm, "User for Failed DM")
#     assert dm_element is not None, f"Failed to create DMElement for {target_user_id_for_failed_dm}"
#     await event_loop._process_incoming_event_queue() # Process any mount events
#     await asyncio.sleep(0.01)

#     # Get MessageActionHandler from the newly created/retrieved DMElement
#     mah = dm_element.get_component_by_type("MessageActionHandler")
#     assert mah is not None, "MessageActionHandler not found on DMElement for failure test"

#     failed_message_text = "This message is destined to fail."
#     send_tool_result = await mah.send_message_tool(text=failed_message_text)
    
#     assert send_tool_result['success'] is True # Dispatch to AC should still be true
#     assert send_tool_result['status'] == "pending_confirmation"
#     internal_failed_req_id = send_tool_result.get('internal_request_id')
#     assert internal_failed_req_id is not None

#     # 3. Check MessageListComponent on the DMElement
#     msg_list_comp = dm_element.get_component_by_type("MessageListComponent")
#     assert msg_list_comp is not None

#     messages_after_failed_dispatch = msg_list_comp.get_messages()
#     assert len(messages_after_failed_dispatch) == 1, "Expected 1 pending message after failed dispatch attempt"
#     pending_failed_msg = messages_after_failed_dispatch[0]
#     assert pending_failed_msg['text'] == failed_message_text
#     assert pending_failed_msg['status'] == "pending_send"
#     assert pending_failed_msg['internal_request_id'] == internal_failed_req_id

#     # 4. Process event queue to handle the simulated FAILURE ACK from MockActivityClient
#     await event_loop._process_incoming_event_queue()
#     await asyncio.sleep(0.01)

#     # 5. Check MessageListComponent again: message should be marked as failed
#     messages_after_failure_ack = msg_list_comp.get_messages()
#     assert len(messages_after_failure_ack) == 1, "Expected 1 message after failure ack"
#     final_failed_msg = messages_after_failure_ack[0]
#     assert final_failed_msg['text'] == failed_message_text
#     assert final_failed_msg['status'] == "failed_to_send"
#     assert final_failed_msg['internal_request_id'] == internal_failed_req_id
#     assert final_failed_msg['error_details'] == "Simulated: Failed to send message due to external adapter error."
#     logger.info(f"Successfully tested message send failure. Message status: {final_failed_msg['status']}, error: {final_failed_msg['error_details']}")
