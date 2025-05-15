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
                            "external_message_ids": [f"mock_external_msg_id_{internal_request_id}"],
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
        agent_name=test_agent_config.name,
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
    assert messages_after_reply_ack[1]['original_external_id'] == f"mock_external_msg_id_{internal_req_id_agent_reply}", "External message ID mismatch after ack"

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
    dm_element_after_second_msg = dm_manager.get_dm_element_for_user(TEST_ADAPTER_ID, dm_sender_external_id)
    assert dm_element_after_second_msg is dm_element, "DMManager created a new DMElement for a subsequent message instead of reusing."
    
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
    await asyncio.sleep(0.01) # Allow async tasks from process_cycle to complete

    veil_producer = dm_element.get_component_by_type("MessageListVeilProducer")
    assert veil_producer is not None, "MessageListVeilProducer not found on DMElement."
    
    dm_veil = veil_producer.get_full_veil()
    assert dm_veil is not None, "DMElement's MessageListVeilProducer failed to produce a VEIL."
    assert dm_veil.get('node_type') == "message_list_container", f"VEIL node type mismatch: {dm_veil.get('node_type')}"
    
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
    
    # logger.critical(f"agent_context: {agent_context}")
    agent_loop = agent_inner_space.get_component_by_type("MultiStepToolLoopComponent")
    assert agent_loop is not None, "AgentLoop not found on InnerSpace."
    tools = agent_loop.aggregate_tools()
    assert len(tools) > 0, "No tools found on InnerSpace."

    print(f"test_dm_ping_pong ({TEST_AGENT_ID}) PASSED!")

# # --- Test Function (Scenario 2) ---
# @pytest.mark.asyncio
# async def borked_test_shared_space_mention_reply(setup_test_environment):
#     event_loop, mock_activity_client, space_registry, agent_inner_space = setup_test_environment
#     mock_activity_client.clear_received_actions() # Clear actions from previous tests if fixture is session-scoped

#     # --- Phase 1: Setup SharedSpace and Simulate Incoming Mention ---
#     shared_channel_id = "channel_789"
#     shared_channel_name = "Test Shared Channel"
#     message_sender_external_id = "user_external_456"
#     message_sender_display_name = "Channel User"
#     incoming_mention_text = f"Hello @{TEST_AGENT_ID}, can you help?"

#     # Simulate incoming message to a shared channel that mentions the agent
#     # ExternalEventRouter should create the SharedSpace and ensure an Uplink for the agent.
#     mention_payload = {
#         "source_adapter_id": TEST_SHARED_ADAPTER_ID,
#         "payload": {
#             "event_type_from_adapter": "message_received",
#             "adapter_data": {
#                 "message_id": "external_msg_002",
#                 "conversation_id": shared_channel_id,
#                 "channel_name": shared_channel_name,
#                 "is_direct_message": False,
#                 "sender": {"user_id": message_sender_external_id, "display_name": message_sender_display_name},
#                 "text": incoming_mention_text,
#                 "timestamp": time.time(),
#                 "mentions": [{"user_id": TEST_AGENT_ID, "display_name": test_agent_config.name}]
#             }
#         }
#     }
#     mock_activity_client.simulate_incoming_event(mention_payload)
#     await event_loop._process_incoming_event_queue()
#     await asyncio.sleep(0.01) # Allow async tasks

#     # Assertions for SharedSpace and Uplink creation:
#     # 1. SharedSpace should exist
#     shared_space_id = f"shared_{TEST_SHARED_ADAPTER_ID}_{shared_channel_id}"
#     shared_space = space_registry.get_space(shared_space_id)
#     assert shared_space is not None, f"SharedSpace {shared_space_id} not created."
#     assert shared_space.name == shared_channel_name
#     assert shared_space.adapter_id == TEST_SHARED_ADAPTER_ID
#     assert shared_space.external_conversation_id == shared_channel_id

#     # 2. Message should be in SharedSpace (assuming it has a MessageListComponent, e.g., via a chat element prefab)
#     #    This part is a bit more complex as SharedSpace itself doesn't directly hold MessageListComponent.
#     #    It would be on a mounted element within SharedSpace. For this test, we'll assume the event reached it.
#     #    A more detailed test would check the mounted chat element's MessageListComponent.
#     #    For now, we'll trust receive_event on SharedSpace processed it.

#     # 3. UplinkProxy should be created in InnerSpace
#     uplink_manager = agent_inner_space.get_uplink_manager()
#     assert uplink_manager is not None
#     uplink_element = uplink_manager.get_uplink_for_space(shared_space_id)
#     assert uplink_element is not None, f"UplinkProxy for SharedSpace {shared_space_id} not found."
#     assert isinstance(uplink_element, UplinkProxy)
#     assert uplink_element.remote_space_id == shared_space_id

#     # --- Phase 1.5: Verify message in SharedSpace's MessageListComponent ---
#     # Assuming "standard_shared_space_chat" prefab mounts an element with ID suffix "chat_interface"
#     # and this element is configured in "standard_uplink_proxy" as the target_element_id_in_remote_space.
#     mounted_chat_element_id_suffix = "chat_interface" 
#     mounted_chat_element_id = f"{shared_space.id}_{mounted_chat_element_id_suffix}"
    
#     # Process events again to ensure any element initialisation events from prefab are handled.
#     await event_loop._process_incoming_event_queue()
#     await asyncio.sleep(0.01)

#     chat_element_in_shared_space = shared_space.get_mounted_element(mounted_chat_element_id)
#     assert chat_element_in_shared_space is not None, (
#         f"Chat element '{mounted_chat_element_id}' not found in SharedSpace '{shared_space.id}'. "
#         f"Mounted elements: {list(shared_space._mounted_elements.keys())}"
#     )

#     ss_msg_list_comp = chat_element_in_shared_space.get_component(MessageListComponent)
#     assert ss_msg_list_comp is not None, (
#         f"MessageListComponent not found on chat element '{chat_element_in_shared_space.id}' in SharedSpace."
#     )
    
#     shared_space_messages = ss_msg_list_comp.get_messages()
#     assert len(shared_space_messages) == 1, (
#         f"Expected 1 message in SharedSpace's chat element, found {len(shared_space_messages)}. Messages: {shared_space_messages}"
#     )
#     assert shared_space_messages[0]['text'] == incoming_mention_text
#     assert shared_space_messages[0]['sender_id'] == message_sender_external_id
#     internal_msg_id_in_shared_space = shared_space_messages[0]['internal_id']

#     # --- Phase 2: Simulate Agent "seeing" the message via Uplink and replying ---
#     # This requires the UplinkProxy's VeilProducer to work, which in turn uses RemoteStateCacheComponent.
#     # For an integration test, we might assume the VEIL is correctly produced or test it separately.
#     # Here, we'll focus on the action of replying via the UplinkProxy.

#     # Verify the VEIL produced by the UplinkProxy
#     uplink_veil_producer = uplink_element.get_component(UplinkVeilProducer)
#     assert uplink_veil_producer is not None, "UplinkVeilProducerComponent not found on UplinkProxy."

#     # Process one cycle for the agent's InnerSpace which might trigger cache updates or component initializations.
#     agent_inner_space.process_cycle() 
#     await asyncio.sleep(0.01) # Allow async tasks within process_cycle

#     uplink_veil = uplink_veil_producer.get_full_veil()
#     assert uplink_veil is not None, "UplinkProxy failed to produce a VEIL."

#     # Assert the content of the VEIL.
#     # The UplinkVeilProducer wraps the remote element's VEIL.
#     # Assumed VEIL_UPLINK_WRAPPER_NODE_TYPE, replace if UplinkVeilProducer defines a constant
#     UPLINK_WRAPPER_NODE_TYPE_EXPECTED = "uplinked_content_container" 
#     assert uplink_veil.get('node_type') == UPLINK_WRAPPER_NODE_TYPE_EXPECTED, (
#         f"Uplink VEIL root node type is not '{UPLINK_WRAPPER_NODE_TYPE_EXPECTED}', got {uplink_veil.get('node_type')}"
#     )
#     assert 'children' in uplink_veil and len(uplink_veil['children']) == 1, (
#         f"Uplink VEIL does not have the expected structure (single child for wrapped content). Children: {uplink_veil.get('children')}"
#     )
    
#     remote_space_root_veil = uplink_veil['children'][0]
#     assert remote_space_root_veil.get('node_type') == MessageListVeilProducer.VEIL_CONTAINER_TYPE, (
#         f"Wrapped VEIL node type is not '{MessageListVeilProducer.VEIL_CONTAINER_TYPE}', got {remote_space_root_veil.get('node_type')}"
#     )
    
#     found_message_in_veil = False
#     if 'children' in remote_space_root_veil:
#         for veil_msg_node in remote_space_root_veil['children']:
#             if (veil_msg_node.get('node_type') == MessageListVeilProducer.VEIL_MESSAGE_NODE_TYPE and
#                veil_msg_node.get('veil_id') == internal_msg_id_in_shared_space):
#                 assert veil_msg_node['properties'].get(MessageListVeilProducer.VEIL_CONTENT_PROP) == incoming_mention_text
#                 found_message_in_veil = True
#                 break
    
#     assert found_message_in_veil, (
#         f"Message '{incoming_mention_text}' (internal_id: {internal_msg_id_in_shared_space}) not found in Uplink VEIL. VEIL: {uplink_veil}"
#     )
        
#     reply_text_to_channel = f"Hello {message_sender_display_name}! Yes, I can help."

#     tool_provider_on_uplink = uplink_element.get_component("ToolProviderComponent")
#     assert tool_provider_on_uplink is not None, "ToolProviderComponent not found on UplinkProxy."
    
#     # MessageActionHandler on UplinkProxy should have send_message
#     action_result = agent_inner_space.execute_action_on_element(
#         element_id=uplink_element.id, # Target the UplinkProxy element
#         action_name="send_message",
#         parameters={"text": reply_text_to_channel}
#     )
#     assert action_result['success'] is True, f"send_message via UplinkProxy failed: {action_result.get('error')}"

#     # --- Phase 3: Assertions for outgoing message from Uplink ---
#     assert len(mock_activity_client.outgoing_actions_received) == 1, "Outgoing message from Uplink not captured."
#     outgoing_action = mock_activity_client.outgoing_actions_received[0]
#     assert outgoing_action['action_type'] == "send_message"
#     outgoing_payload = outgoing_action['payload']
#     assert outgoing_payload['adapter_id'] == TEST_SHARED_ADAPTER_ID
#     assert outgoing_payload['conversation_id'] == shared_channel_id # Should be the channel ID
#     assert outgoing_payload['text'] == reply_text_to_channel
#     assert outgoing_payload['requesting_element_id'] == uplink_element.id
#     # requesting_agent_id for actions from UplinkProxy (which is in InnerSpace) should be the InnerSpace's agent_id
#     assert outgoing_payload['requesting_agent_id'] == agent_inner_space.agent_id

#     print(f"test_shared_space_mention_reply ({TEST_AGENT_ID} in {shared_channel_id}) PASSED!") 

# # --- NEW TEST CASE for Send Failure ---
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
