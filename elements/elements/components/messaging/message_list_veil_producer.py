"""
Message List Veil Producer Component
Generates VEIL representation for a list of messages.
"""
import logging
from typing import Dict, Any, Optional, List
import time

from ..base_component import VeilProducer
# Assuming MessageListComponent is in the same directory
from .message_list import MessageListComponent, MessageType
from elements.component_registry import register_component
# NEW: Import VEILFacet system
from ..veil import (
    VEILFacetOperation, VEILFacet, VEILFacetType,
    EventFacet, StatusFacet, AmbientFacet, ConnectomeEpoch,
    FacetOperationBuilder)
from ..veil.facet_types import (create_message_event_facet,
    create_container_status_facet, create_tool_instructions_ambient_facet
)

logger = logging.getLogger(__name__)

# NEW: Removed centralized tool families - each VeilProducer sets arbitrary tool_family string

# VEIL Node Structure Constants (Example)
VEIL_CONTAINER_TYPE = "message_list"
VEIL_MESSAGE_NODE_TYPE = "message"
VEIL_SENDER_PROP = "sender_name"
VEIL_TIMESTAMP_PROP = "timestamp_iso"
VEIL_CONTENT_PROP = "text_content"
VEIL_INTERNAL_ID_PROP = "connectome_internal_msg_id"
VEIL_EXTERNAL_ID_PROP = "external_id"
VEIL_ADAPTER_ID_PROP = "adapter_id"
VEIL_EDITED_FLAG_PROP = "is_edited"
VEIL_ATTACHMENT_METADATA_PROP = "attachment_metadata"
VEIL_ATTACHMENT_CONTENT_NODE_TYPE = "attachment_content_item"

@register_component
class MessageListVeilProducer(VeilProducer):
    """
    Generates VEIL representation based on the state of a sibling MessageListComponent.
    Handles both full VEIL generation and potentially delta calculation (basic implementation).
    """
    COMPONENT_TYPE = "MessageListVeilProducer"

    # Dependencies: Requires a MessageListComponent on the same Element
    REQUIRED_SIBLING_COMPONENTS = [MessageListComponent]

    def initialize(self, **kwargs) -> None:
        """Initializes the component."""
        super().initialize(**kwargs)
        self._state.setdefault('_last_generated_veil_message_ids', set())
        self._state.setdefault('_last_generated_element_properties', set())
        self._state.setdefault('_has_produced_list_root_add_before', False)
        self._state.setdefault('_last_list_root_properties', {})
        # NEW: Track message content for edit detection
        self._state.setdefault('_last_message_content_map', {})
        logger.debug(f"MessageListVeilProducer initialized for Element {self.owner.id}")

    def _get_message_list_component(self) -> Optional[MessageListComponent]:
        """Helper to get the sibling message list component."""
        return self.get_sibling_component(MessageListComponent)

    def _get_available_tools_for_element(self) -> List[str]:
        """Get list of available tool names for this element."""
        from ..tool_provider import ToolProviderComponent

        tool_provider = self.get_sibling_component(ToolProviderComponent)
        if tool_provider:
            return tool_provider.list_tools()
        return []

    def _get_enhanced_tools_for_element(self) -> List[Dict[str, Any]]:
        """
        NEW: Get enhanced tool definitions with complete metadata for VEIL emission.

        Returns rich tool information needed for tool aggregation and rendering.
        """
        from ..tool_provider import ToolProviderComponent

        tool_provider = self.get_sibling_component(ToolProviderComponent)
        if tool_provider:
            return tool_provider.get_enhanced_tool_definitions()
        return []

    def _get_conversation_metadata(self) -> Dict[str, Any]:
        """Get conversation metadata from the owner element."""
        metadata = {}
        if self.owner:
            metadata.update({
                "adapter_type": getattr(self.owner, 'adapter_type', None),
                "server_name": getattr(self.owner, 'server_name', None),
                "conversation_name": getattr(self.owner, 'conversation_name', None),
                "adapter_id": getattr(self.owner, 'adapter_id', None),
                "external_conversation_id": getattr(self.owner, 'external_conversation_id', None),
                "alias": getattr(self.owner, 'alias', None)
            })
        return metadata

    def get_full_veil(self) -> Optional[Dict[str, Any]]:
        """
        Generates the complete VEIL structure for the current message list.
        """
        message_list_comp = self._get_message_list_component()
        if not message_list_comp:
            logger.error(f"[{self.owner.id}] Cannot generate VEIL: MessageListComponent not found.")
            return None

        # Get conversation metadata for VEIL properties
        conversation_metadata = self._get_conversation_metadata()


        messages = message_list_comp.get_messages() # Get all current messages
        message_nodes = []
        current_message_ids = set()

        for msg_data in messages:
            internal_id = msg_data.get('internal_id')
            if not internal_id: continue # Skip messages without internal ID

            current_message_ids.add(internal_id)

            # Basic VEIL node for a message
            # The VEIL_ATTACHMENT_METADATA_PROP will now carry attachments processed by MessageListComponent,
            # which may include inline 'content'.
            processed_attachments_from_mlc = msg_data.get('attachments', [])

            message_node = {
                "veil_id": internal_id, # Use internal ID for delta tracking
                "node_type": VEIL_MESSAGE_NODE_TYPE,
                "properties": {
                    "structural_role": "list_item",
                    "content_nature": "chat_message",
                    VEIL_SENDER_PROP: msg_data.get('sender_name', 'Unknown'),
                    VEIL_TIMESTAMP_PROP: msg_data.get('timestamp'), # Assuming timestamp is suitable format
                    VEIL_CONTENT_PROP: msg_data.get('text', ''),
                    VEIL_INTERNAL_ID_PROP: internal_id,
                    VEIL_EXTERNAL_ID_PROP: msg_data.get('original_external_id'),
                    VEIL_ADAPTER_ID_PROP: msg_data.get('adapter_id'),
                    VEIL_EDITED_FLAG_PROP: msg_data.get('is_edited', False),
                    # NEW: Include reaction data for HUD rendering
                    "reactions": msg_data.get('reactions', {}),  # Include full reaction dict {emoji: [user_ids]}
                    "message_status": msg_data.get('status', 'received'),  # Include message status for pending states
                    # NEW: Include retry information for HUD display
                    "retry_count": msg_data.get('retry_count', 0),
                    "retry_reason": msg_data.get('retry_reason', None),
                    # VEIL_ATTACHMENT_METADATA_PROP: processed_attachments_from_mlc # Store the rich attachment dicts
                    # Let's refine this: the VEIL_ATTACHMENT_METADATA_PROP should probably just be the metadata part,
                    # and the content part should lead to a child node if content exists.
                    "adapter_type": conversation_metadata.get("adapter_type"),
                    "server_name": conversation_metadata.get("server_name"),
                    "conversation_name": conversation_metadata.get("conversation_name"),
                    "error_details": msg_data.get('error_details', None),
                    "is_from_current_agent": msg_data.get('is_from_current_agent', False),  # For HUD rendering decisions
                    VEIL_ATTACHMENT_METADATA_PROP: [
                        {k: v for k, v in att.items() if k != 'content'}
                        for att in processed_attachments_from_mlc
                    ]
                },
                "children": [] # Initialize, may add attachment content nodes later
            }

            # --- NEW: Create child nodes for attachments that have content ---
            for att_data_from_mlc in processed_attachments_from_mlc:
                attachment_id = att_data_from_mlc.get('attachment_id')
                if not attachment_id: continue

                if att_data_from_mlc.get('content') is not None:
                    # Content is available (either was inline or fetched and stored by MessageListComponent)
                    attachment_content_node = {
                        "veil_id": f"att_{attachment_id}_content_{internal_id}",
                        "node_type": VEIL_ATTACHMENT_CONTENT_NODE_TYPE,
                        "properties": {
                            "structural_role": "attachment_content",
                            "content_nature": att_data_from_mlc.get("content_type", "unknown"),
                            "filename": att_data_from_mlc.get("filename", attachment_id),
                            "content_available": True, # Explicitly state content is here
                            "attachment_id": attachment_id,
                            "original_message_veil_id": internal_id,
                            # NEW: Include actual content for multimodal processing
                            "content": att_data_from_mlc.get("content"),  # Direct content access!
                            # "content_preview": str(att_data_from_mlc.get("content"))[:100] # Optional: Careful with large content
                        }
                    }
                    # NEW: Add owner tracking to attachment nodes
                    self._add_owner_tracking(attachment_content_node)
                    message_node["children"].append(attachment_content_node)
                # else: No content available for this attachment in MessageListComponent's state.
                # The metadata is already in VEIL_ATTACHMENT_METADATA_PROP.

            # NEW: Add owner tracking to message nodes
            self._add_owner_tracking(message_node)
            message_nodes.append(message_node)


        # NEW: Get enhanced tool definitions for Phase 1 VEIL enhancement
        enhanced_tools = self._get_enhanced_tools_for_element()
        available_tool_names = self._get_available_tools_for_element()  # Backward compatibility

        # Create the root container node for the list
        root_veil_node = {
            "veil_id": f"{self.owner.id}_message_list_root",
            "node_type": VEIL_CONTAINER_TYPE,
            "properties": {
                "structural_role": "container",
                "content_nature": "message_list",
                "element_id": self.owner.id,
                "element_name": self.owner.name,
                "message_count": len(message_nodes),
                # ENHANCED: Rich tool metadata for aggregation and rendering
                "available_tools": enhanced_tools,
                # BACKWARD COMPATIBILITY: Simple tool names for existing components
                "available_tool_names": available_tool_names,
                "tool_target_element_id": self.owner.id,  # Explicit target for tools
                # NEW: Include conversation metadata for rich VEIL context
                "adapter_type": conversation_metadata.get("adapter_type"),
                "server_name": conversation_metadata.get("server_name"),
                "conversation_name": conversation_metadata.get("conversation_name"),
                "adapter_id": conversation_metadata.get("adapter_id"),
                "external_conversation_id": conversation_metadata.get("external_conversation_id"),
                "alias": conversation_metadata.get("alias")
            },
            "children": message_nodes
        }

        # NEW: Add owner tracking to the root container
        self._add_owner_tracking(root_veil_node)

        # Update state for delta calculation - message IDs are handled in signal_delta_produced_this_frame
        # self._state['_last_generated_veil_message_ids'] = current_message_ids
        # Properties of the root for its own update detection are also handled in signal_delta_produced_this_frame

        return root_veil_node

    def calculate_delta(self) -> Optional[List[VEILFacetOperation]]:
        """
        NEW: Calculate VEILFacet operations for message list management with phase awareness.

        This replaces the old delta operation system with VEILFacet operations, generating:
        - StatusFacet for message list container creation/updates
        - EventFacet for new messages added
        - EventFacet for message edits/deletes
        - AmbientFacet for tool availability (when appropriate)

        NEW: Phase-aware processing - defers content processing during structural phase.

        Returns:
            List of VEILFacetOperation instances for the message list
        """
        
        # NEW: Check if we're in structural replay phase and should defer content processing
        should_defer = self._should_defer_content_processing()
        
        if should_defer:
            return None

        facet_operations = []

        message_list_comp = self._get_message_list_component()
        if not message_list_comp:
            logger.error(f"[{self.owner.id}] Cannot calculate facet operations: MessageListComponent not found.")
            return None

        if not self.owner:
            logger.error(f"[{self.COMPONENT_TYPE}] Owner not set, cannot calculate facet operations.")
            return None

        # Continue with existing facet operations logic
        owner_id = self.owner.id
        list_root_facet_id = f"{owner_id}_message_list_container"

        # Get current messages and metadata
        current_messages = message_list_comp.get_messages()
        current_message_ids = {msg.get('internal_id') for msg in current_messages if msg.get('internal_id')}
        conversation_metadata = self._get_conversation_metadata()

        # 1. Handle message list container (StatusFacet)
        container_facet_exists = self._state.get('_has_produced_list_root_add_before', False)

        # FIXED: Get enhanced tools for StatusFacet
        enhanced_tools = self._get_enhanced_tools_for_element()

        # Get current container state
        current_container_state = {
            "element_id": owner_id,
            "element_name": self.owner.name,
            "message_count": len(current_message_ids),
            "conversation_name": conversation_metadata.get("conversation_name"),
            "adapter_type": conversation_metadata.get("adapter_type"),
            "server_name": conversation_metadata.get("server_name"),
            "adapter_id": conversation_metadata.get("adapter_id"),
            "external_conversation_id": conversation_metadata.get("external_conversation_id"),
            "alias": conversation_metadata.get("alias"),
            # FIXED: Include enhanced tools in StatusFacet current_state
            "available_tools": enhanced_tools
        }

        if not container_facet_exists:
            # Create container StatusFacet
            container_facet = StatusFacet(
                facet_id=list_root_facet_id,
                veil_timestamp=ConnectomeEpoch.get_veil_timestamp(),
                owner_element_id=owner_id,
                status_type="container_created",
                current_state=current_container_state,
                links_to=f"{self.owner.get_parent_info()['parent_id']}_space_root" if hasattr(self.owner, 'get_parent_info') else None
            )

            facet_operations.append(FacetOperationBuilder.add_facet(container_facet))
            logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Generated add_facet for container {list_root_facet_id}")

        else:
            # Check for container state updates
            last_container_state = self._state.get('_last_list_root_properties', {})
            if current_container_state != last_container_state:
                facet_operations.append(
                    FacetOperationBuilder.update_facet(
                        list_root_facet_id,
                        {"current_state": current_container_state}
                    )
                )
                logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Generated update_facet for container {list_root_facet_id}")

        # 2. Handle message additions (EventFacets)
        last_message_ids = self._state.get('_last_generated_veil_message_ids', set())
        added_message_ids = current_message_ids - last_message_ids

        for msg_data in current_messages:
            internal_id = msg_data.get('internal_id')
            if internal_id in added_message_ids:
                # Create EventFacet for new message
                message_facet = EventFacet(
                    facet_id=internal_id,
                    veil_timestamp=ConnectomeEpoch.get_veil_timestamp(),
                    owner_element_id=owner_id,
                    event_type="message_added",
                    content=msg_data.get('text', ''),
                    links_to=list_root_facet_id
                )

                # NEW: Create synthetic agent_response facet for external agent messages
                # Only create synthetic for messages that don't have internal origin (tool calls)
                if self._should_create_synthetic_agent_response(msg_data, owner_id):
                    synthetic_response_facet = self._create_synthetic_agent_response_facet(msg_data, conversation_metadata)
                    if synthetic_response_facet:
                        facet_operations.append(FacetOperationBuilder.add_facet(synthetic_response_facet))

                        # IMMEDIATE MARKING: Mark message as internal origin to prevent future synthetics
                        self._mark_message_as_internal_origin(msg_data)
                        facet_operations.append(FacetOperationBuilder.update_facet(
                            internal_id,
                            {"is_internal_origin": True}
                        ))

                        logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Generated synthetic agent_response and marked message {internal_id} as internal origin")

                # Add message-specific properties
                message_facet.properties.update({
                    "sender_name": msg_data.get('sender_name', 'Unknown'),
                    "timestamp_iso": msg_data.get('timestamp'),
                    "external_id": msg_data.get('original_external_id'),
                    "adapter_id": msg_data.get('adapter_id'),
                    "is_edited": msg_data.get('is_edited', False),
                    "reactions": msg_data.get('reactions', {}),
                    "message_status": msg_data.get('status', 'received'),
                    "adapter_type": conversation_metadata.get("adapter_type"),
                    "server_name": conversation_metadata.get("server_name"),
                    "conversation_name": conversation_metadata.get("conversation_name"),
                    "is_agent": msg_data.get('sender_name', 'Unknown') == self.owner.get_parent_object().agent_name,
                    "is_from_current_agent": msg_data.get('is_from_current_agent', False),  # For HUD rendering decisions
                    "is_internal_origin": msg_data.get('is_internal_origin', False),  # NEW: Track message origin for synthetic response logic
                    "error_details": msg_data.get('error_details', None),
                    "attachment_metadata": [
                        {k: v for k, v in att.items() if k != 'content'}
                        for att in msg_data.get('attachments', [])
                    ]
                })

                facet_operations.append(FacetOperationBuilder.add_facet(message_facet))
                logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Generated add_facet for message {internal_id}")

        # 3. Handle message removals
        removed_message_ids = last_message_ids - current_message_ids
        for removed_id in removed_message_ids:
            facet_operations.append(FacetOperationBuilder.remove_facet(removed_id))
            logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Generated remove_facet for message {removed_id}")

        # 4. Handle message edits/deletes with dual operations (update content + add audit event)
        # Check for edit/delete operations from MessageListComponent
        if message_list_comp:
            pending_operations = message_list_comp.get_pending_veil_operations()

            for operation in pending_operations:
                if operation["operation_type"] == "edit":
                    # DUAL OPERATION: Update original message facet + add edit audit event
                    original_message_id = operation["veil_id"]
                    edit_details = operation.get("edit_details", {})
                    new_text = edit_details.get("new_text", "")
                    edit_timestamp = edit_details.get("edit_timestamp", time.time())

                    # 1. Update the original message facet with new content
                    facet_operations.append(FacetOperationBuilder.update_facet(
                        original_message_id,
                        {
                            "content": new_text,
                            "is_edited": True,
                            "last_edited_timestamp": edit_timestamp,
                            "original_content": edit_details.get("original_text", "")  # Preserve for history
                        }
                    ))

                    # 2. Add audit EventFacet for the edit operation
                    edit_facet = self._create_message_edit_facet(operation, conversation_metadata)
                    if edit_facet:
                        facet_operations.append(FacetOperationBuilder.add_facet(edit_facet))

                elif operation["operation_type"] == "reaction_added":
                    # Update the original message facet with new reaction data
                    original_message_id = operation["veil_id"]

                    # Get the updated message data to extract current reactions
                    updated_message = self._get_message_by_id(original_message_id)
                    if updated_message:
                        facet_operations.append(FacetOperationBuilder.update_facet(
                            original_message_id,
                            {
                                "reactions": updated_message.get('reactions', {}),
                                "last_reaction_timestamp": operation["reaction_details"]["timestamp"]
                            }
                        ))
                        logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Generated update_facet for reaction_added on message {original_message_id}")

                elif operation["operation_type"] == "reaction_removed":
                    # Update the original message facet with new reaction data
                    original_message_id = operation["veil_id"]

                    # Get the updated message data to extract current reactions
                    updated_message = self._get_message_by_id(original_message_id)
                    if updated_message:
                        facet_operations.append(FacetOperationBuilder.update_facet(
                            original_message_id,
                            {
                                "reactions": updated_message.get('reactions', {}),
                                "last_reaction_timestamp": operation["reaction_details"]["timestamp"]
                            }
                        ))
                        logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Generated update_facet for reaction_removed on message {original_message_id}")

        # 5. Detect content changes via fallback comparison
        if message_list_comp:
            current_messages = message_list_comp.get_messages()
            last_message_content = self._state.get('_last_message_content_map', {})

            for msg_data in current_messages:
                internal_id = msg_data.get('internal_id')
                if internal_id in last_message_content:
                    last_content = last_message_content[internal_id]
                    current_content = msg_data.get('text', '')

                    if last_content != current_content and msg_data.get('is_edited', False):
                        # Create EventFacet for detected edit
                        edit_facet = self._create_content_change_edit_facet(msg_data, last_content, conversation_metadata)
                        if edit_facet:
                            facet_operations.append(FacetOperationBuilder.add_facet(edit_facet))

            # Update content tracking for next frame
            self._update_message_content_tracking(current_messages)

        # 6. Generate tool availability ambient facets (when appropriate)
        if self._should_emit_tools_ambient_facet():
            tools_ambient_facet = self._create_tools_ambient_facet()
            if tools_ambient_facet:
                facet_operations.append(FacetOperationBuilder.add_facet(tools_ambient_facet))
                logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Generated structured ambient facet for tools")

        # Update state after generating operations
        if not container_facet_exists and any(
            op.operation_type == "add_facet" and
            op.facet and op.facet.facet_id == list_root_facet_id
            for op in facet_operations
        ):
            self._state['_has_produced_list_root_add_before'] = True

        self._state['_last_list_root_properties'] = current_container_state
        self._state['_last_generated_veil_message_ids'] = current_message_ids

        return facet_operations if facet_operations else None

    def _update_message_content_tracking(self, current_messages: List[Dict[str, Any]]) -> None:
        """Update content tracking for next frame's comparison."""
        content_map = {}
        for msg in current_messages:
            internal_id = msg.get('internal_id')
            if internal_id:
                content_map[internal_id] = msg.get('text', '')
        self._state['_last_message_content_map'] = content_map

    # --- NEW: VEILFacet Helper Methods ---
    def _create_message_edit_facet(self, operation: Dict[str, Any], conversation_metadata: Dict[str, Any]) -> Optional[EventFacet]:
        """
        Create EventFacet for message edit operations.

        Args:
            operation: Edit operation data from MessageListComponent
            conversation_metadata: Conversation context metadata

        Returns:
            EventFacet for the edit operation
        """
        try:
            edit_details = operation.get("edit_details", {})
            sender_info = operation.get("sender_info", {})

            edit_facet = EventFacet(
                facet_id=f"edit_{operation['veil_id']}_{int(time.time() * 1000)}",
                veil_timestamp=ConnectomeEpoch.get_veil_timestamp(),
                owner_element_id=self.owner.id,
                event_type="message_edited",
                content=f"Message edited: {edit_details.get('new_preview', {}).get('preview', 'Unknown')}",
                links_to=f"{self.owner.id}_message_list_container"
            )

            # Add edit-specific properties
            edit_facet.properties.update({
                "original_message_id": operation["veil_id"],
                "edit_timestamp": edit_details.get("edit_timestamp"),
                "sender_name": sender_info.get("sender_name"),
                "sender_id": sender_info.get("sender_id"),
                "conversation_name": conversation_metadata.get("conversation_name"),
                "adapter_type": conversation_metadata.get("adapter_type"),
                "server_name": conversation_metadata.get("server_name"),
                "original_preview": edit_details.get("original_preview", {}).get("preview"),
                "new_preview": edit_details.get("new_preview", {}).get("preview"),
                "edit_context": {
                    "original_truncated": edit_details.get("original_preview", {}).get("truncated", False),
                    "new_truncated": edit_details.get("new_preview", {}).get("truncated", False),
                    "edit_type": "content_change"
                }
            })

            return edit_facet

        except Exception as e:
            logger.error(f"Error creating message edit facet: {e}", exc_info=True)
            return None

    def _create_message_delete_facet(self, operation: Dict[str, Any], conversation_metadata: Dict[str, Any]) -> Optional[EventFacet]:
        """
        Create EventFacet for message delete operations.

        Args:
            operation: Delete operation data from MessageListComponent
            conversation_metadata: Conversation context metadata

        Returns:
            EventFacet for the delete operation
        """
        try:
            delete_details = operation.get("delete_details", {})
            sender_info = operation.get("sender_info", {})

            delete_facet = EventFacet(
                facet_id=f"delete_{operation['veil_id']}_{int(time.time() * 1000)}",
                veil_timestamp=ConnectomeEpoch.get_veil_timestamp(),
                owner_element_id=self.owner.id,
                event_type="message_deleted",
                content=f"Message deleted: {delete_details.get('original_preview', {}).get('preview', 'Unknown')}",
                links_to=f"{self.owner.id}_message_list_container"
            )

            # Add delete-specific properties
            delete_facet.properties.update({
                "original_message_id": operation["veil_id"],
                "delete_timestamp": delete_details.get("delete_timestamp"),
                "sender_name": sender_info.get("sender_name"),
                "sender_id": sender_info.get("sender_id"),
                "conversation_name": conversation_metadata.get("conversation_name"),
                "adapter_type": conversation_metadata.get("adapter_type"),
                "server_name": conversation_metadata.get("server_name"),
                "original_preview": delete_details.get("original_preview", {}).get("preview"),
                "deletion_source": delete_details.get("deletion_source", "external"),
                "delete_context": {
                    "original_truncated": delete_details.get("original_preview", {}).get("truncated", False),
                    "delete_type": "message_removal"
                }
            })

            return delete_facet

        except Exception as e:
            logger.error(f"Error creating message delete facet: {e}", exc_info=True)
            return None

    def _create_synthetic_agent_response_facet(self, msg_data: Dict[str, Any], conversation_metadata: Dict[str, Any]) -> Optional[EventFacet]:
        """
        Create synthetic agent_response EventFacet for historical agent messages.

        This ensures consistent turn structure when processing historical agent messages
        that don't have corresponding agent_response facets from the original sending.

        Args:
            msg_data: Message data from MessageListComponent
            conversation_metadata: Conversation context metadata

        Returns:
            EventFacet for the synthetic agent response
        """
        try:
            # Get agent info from parent space
            agent_name = "Agent"  # Default fallback
            if self.owner and hasattr(self.owner, 'get_parent_object'):
                parent_space = self.owner.get_parent_object()
                if parent_space and hasattr(parent_space, 'agent_name'):
                    agent_name = parent_space.agent_name

            # Create synthetic agent response facet with current processing timestamp
            response_facet = EventFacet(
                facet_id=f"synthetic_response_{msg_data.get('internal_id')}_{int(time.time() * 1000)}",
                veil_timestamp=ConnectomeEpoch.get_veil_timestamp(),  # Use current processing time like other facets
                owner_element_id=self.owner.id,
                event_type="agent_response",
                content=msg_data.get('text', ''),
                links_to=f"{self.owner.get_parent_object().id}_space_root" if hasattr(self.owner, 'get_parent_object') else None
            )

            # Add response-specific properties
            response_facet.properties.update({
                "agent_name": agent_name,
                "tool_calls_count": 0,  # Historical messages don't have tool call data
                "message_status": "sent",  # Historical messages are always sent
                "conversation_name": conversation_metadata.get("conversation_name"),
                "adapter_type": conversation_metadata.get("adapter_type"),
                "server_name": conversation_metadata.get("server_name"),
                "synthetic": True,  # Mark as synthetic for debugging
                "original_message_id": msg_data.get('internal_id'),
                "original_message_timestamp": msg_data.get('timestamp'),  # Preserve historical timestamp as metadata
                "historical_reconstruction": True,  # Flag for audit purposes
                "parsing_mode": "historical_text"  # Indicate this was reconstructed from historical message
            })

            return response_facet

        except Exception as e:
            logger.error(f"Error creating synthetic agent response facet: {e}", exc_info=True)
            return None

    def _should_create_synthetic_agent_response(self, msg_data: Dict[str, Any], owner_id: str) -> bool:
        """
        Determine if we should create a synthetic agent_response facet for this message.

        NEW LOGIC:
        - Only agent messages need synthetic responses
        - Messages with internal_origin=True already have tool call context - no synthetic needed
        - Messages with internal_origin=False are external and need synthetic if from our agent

        Args:
            msg_data: Message data from MessageListComponent
            owner_id: Owner element ID

        Returns:
            True if synthetic agent_response should be created
        """
        try:
            # Must be from current agent to need synthetic response
            if not msg_data.get('is_from_current_agent', False):
                logger.debug(f"[{owner_id}] Message not from current agent - skipping synthetic")
                return False

            # If already marked as internal origin, no synthetic needed
            if msg_data.get('is_internal_origin', False):
                logger.debug(f"[{owner_id}] Message {msg_data.get('internal_id')} already marked as internal origin - skipping synthetic")
                return False

            # This is an external agent message that needs synthetic response
            logger.debug(f"[{owner_id}] Message {msg_data.get('internal_id')} is external agent message - creating synthetic response")
            return True

        except Exception as e:
            logger.error(f"Error determining synthetic agent response need: {e}", exc_info=True)
            # Safe default: don't create synthetic if unsure
            return False

    def _create_content_change_edit_facet(self, msg_data: Dict[str, Any], last_content: str, conversation_metadata: Dict[str, Any]) -> Optional[EventFacet]:
        """
        Create EventFacet for content changes detected by fallback comparison.

        Args:
            msg_data: Message data
            last_content: Previous content for comparison
            conversation_metadata: Conversation context metadata

        Returns:
            EventFacet for the content change
        """
        try:
            edit_facet = EventFacet(
                facet_id=f"edit_fallback_{msg_data.get('internal_id')}_{int(time.time() * 1000)}",
                veil_timestamp=ConnectomeEpoch.get_veil_timestamp(),
                owner_element_id=self.owner.id,
                event_type="message_edited",
                content=f"Message content changed: {msg_data.get('text', '')[:50]}{'...' if len(msg_data.get('text', '')) > 50 else ''}",
                links_to=f"{self.owner.id}_message_list_container"
            )

            # Add edit-specific properties
            edit_facet.properties.update({
                "original_message_id": msg_data.get('internal_id'),
                "edit_timestamp": msg_data.get('last_edited_timestamp', time.time()),
                "sender_name": msg_data.get('sender_name'),
                "conversation_name": conversation_metadata.get("conversation_name"),
                "adapter_type": conversation_metadata.get("adapter_type"),
                "server_name": conversation_metadata.get("server_name"),
                "original_preview": last_content[:50] if last_content else "",
                "new_preview": msg_data.get('text', '')[:50],
                "edit_context": {
                    "original_truncated": len(last_content) > 50 if last_content else False,
                    "new_truncated": len(msg_data.get('text', '')) > 50,
                    "edit_type": "content_change_fallback",
                    "detection_method": "content_comparison"
                }
            })

            return edit_facet

        except Exception as e:
            logger.error(f"Error creating content change edit facet: {e}", exc_info=True)
            return None

    def _format_tools_for_ambient_facet(self, enhanced_tools: List[Dict[str, Any]]) -> str:
        """
        Format enhanced tool definitions for ambient facet content.

        Args:
            enhanced_tools: List of enhanced tool definitions

        Returns:
            Formatted string content for ambient facet
        """
        try:
            if not enhanced_tools:
                return "No tools available"

            tool_lines = []
            for tool in enhanced_tools:
                tool_name = tool.get("name", "unknown")
                description = tool.get("description", "No description")

                # Format as simple tool instruction
                tool_lines.append(f"- {tool_name}: {description}")

            return f"Available tools for {self.owner.name}:\n" + "\n".join(tool_lines)

        except Exception as e:
            logger.error(f"Error formatting tools for ambient facet: {e}", exc_info=True)
            return "Error formatting tool information"

    # --- NEW: Enhanced Structured Ambient Facet Methods ---

    def _should_emit_tools_ambient_facet(self) -> bool:
        """
        Determine whether to emit tools ambient facet for this element.

        Returns:
            True if tools ambient facet should be emitted
        """
        enhanced_tools = self._get_enhanced_tools_for_element()
        return bool(enhanced_tools)

    def _create_tools_ambient_facet(self) -> Optional[AmbientFacet]:
        """
        Create enhanced AmbientFacet for available tools with structured data.

        This creates structured data that HUD can consolidate and render appropriately,
        rather than pre-rendered strings.

        Returns:
            AmbientFacet with structured tool data for HUD consolidation
        """
        enhanced_tools = self._get_enhanced_tools_for_element()
        if not enhanced_tools:
            return None

        # Determine tool family for this element
        tool_family = self._classify_tool_family(enhanced_tools)

        # Create structured content instead of pre-rendered strings
        structured_content = {
            "tools": enhanced_tools,
            "element_context": self._get_element_context_metadata(),
            "tool_family": tool_family
        }

        ambient_facet = AmbientFacet(
            facet_id=f"{self.owner.id}_tools_ambient",
            owner_element_id=self.owner.id,
            ambient_type=tool_family,  # Tool family classification for HUD grouping
            content=structured_content,  # Structured data instead of string
            trigger_threshold=1500  # Element-specific threshold
        )

        # Add additional properties for HUD processing
        ambient_facet.properties.update({
            "data_format": "structured",
            "tools_count": len(enhanced_tools),
            "element_type": "messaging"
        })

        return ambient_facet

    def _classify_tool_family(self, enhanced_tools: List[Dict[str, Any]]) -> str:
        """
        Get tool family for this messaging element.

        Args:
            enhanced_tools: List of enhanced tool definitions (unused - family is element-based)

        Returns:
            Tool family string for this element type
        """
        # Each element type sets its own arbitrary tool_family string
        # Messaging elements use "messaging_tools" by default
        return "messaging_tools"

    def _get_element_context_metadata(self) -> Dict[str, Any]:
        """
        Get element context metadata for HUD consolidation.

        Returns:
            Dictionary with element context information
        """
        conversation_metadata = self._get_conversation_metadata()

        return {
            "element_id": self.owner.id,
            "element_name": self.owner.name,
            "conversation_name": conversation_metadata.get("conversation_name"),
            "element_type": "messaging",
            "adapter_type": conversation_metadata.get("adapter_type"),
            "server_name": conversation_metadata.get("server_name"),
            "adapter_id": conversation_metadata.get("adapter_id"),
            "external_conversation_id": conversation_metadata.get("external_conversation_id"),
            "alias": conversation_metadata.get("alias")
        }

    def _get_message_by_id(self, internal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get message data by internal ID from the MessageListComponent.

        Args:
            internal_id: Internal message ID

        Returns:
            Message data dictionary or None if not found
        """
        message_list_comp = self._get_message_list_component()
        if not message_list_comp:
            return None

        return message_list_comp.get_message_by_internal_id(internal_id)

    def _mark_message_as_internal_origin(self, msg_data: Dict[str, Any]) -> None:
        """
        Mark a message as having internal origin to prevent future synthetic generation.

        Updates the message data in-place within MessageListComponent to ensure
        immediate consistency and prevent infinite synthetic generation loops.

        Args:
            msg_data: Message data dictionary to mark
        """
        try:
            # Update the message data in-place
            msg_data['is_internal_origin'] = True
            logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Marked message {msg_data.get('internal_id')} as internal origin in MessageListComponent")
        except Exception as e:
            logger.error(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Failed to mark message as internal origin: {e}", exc_info=True)

    def _should_defer_content_processing(self) -> bool:
        """
        NEW: Determine if content processing should be deferred during structural replay phase.

        UPDATED: Only defer during structural phase. During content phase, allow VEIL emission
        for real-time chronological VEIL building.

        Returns:
            True if content processing should be deferred, False otherwise
        """
        try:
            # Check if owner space is in structural replay phase (ONLY defer during structural)
            if self.owner and hasattr(self.owner, '_replay_in_progress'):
                if self.owner._replay_in_progress:
                    current_phase = getattr(self.owner, '_current_replay_phase', None)
                    if current_phase == 'structural':
                        logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Structural phase - deferring content processing")
                        return True
                    elif current_phase == 'content':
                        logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Content phase - allowing VEIL emission for chronological order")
                        return False  # NEW: Allow processing during content phase for real-time VEIL emission

            # Check if parent space is in structural replay phase (for nested elements)
            if self.owner and hasattr(self.owner, 'get_parent_object'):
                parent_space = self.owner.get_parent_object()
                if parent_space and hasattr(parent_space, '_replay_in_progress'):
                    if parent_space._replay_in_progress:
                        current_phase = getattr(parent_space, '_current_replay_phase', None)
                        if current_phase == 'structural':
                            logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Parent in structural phase - deferring content")
                            return True
                        elif current_phase == 'content':
                            logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Parent in content phase - allowing VEIL emission")
                            return False  # NEW: Allow processing during content phase

            return False  # Normal operation - no deferral

        except Exception as e:
            logger.warning(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Error checking replay phase: {e}")
            # Safe default: don't defer if unsure
            return False
