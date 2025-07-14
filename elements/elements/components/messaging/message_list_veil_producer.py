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
    FacetOperationBuilder, create_message_event_facet, 
    create_container_status_facet, create_tool_instructions_ambient_facet
)

logger = logging.getLogger(__name__)

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
                    # VEIL_ATTACHMENT_METADATA_PROP: processed_attachments_from_mlc # Store the rich attachment dicts
                    # Let's refine this: the VEIL_ATTACHMENT_METADATA_PROP should probably just be the metadata part,
                    # and the content part should lead to a child node if content exists.
                    "adapter_type": conversation_metadata.get("adapter_type"),
                    "server_name": conversation_metadata.get("server_name"),
                    "conversation_name": conversation_metadata.get("conversation_name"),
                    "error_details": msg_data.get('error_details', None),
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
        NEW: Calculate VEILFacet operations for message list management.
        
        This replaces the old delta operation system with VEILFacet operations, generating:
        - StatusFacet for message list container creation/updates
        - EventFacet for new messages added  
        - EventFacet for message edits/deletes
        - AmbientFacet for tool availability (when appropriate)
        
        Returns:
            List of VEILFacetOperation instances for the message list
        """
        message_list_comp = self._get_message_list_component()
        if not message_list_comp:
            logger.error(f"[{self.owner.id}] Cannot calculate facet operations: MessageListComponent not found.")
            return None

        if not self.owner:
            logger.error(f"[{self.COMPONENT_TYPE}] Owner not set, cannot calculate facet operations.")
            return None

        facet_operations = []
        owner_id = self.owner.id
        list_root_facet_id = f"{owner_id}_message_list_container"
        
        # Get current messages and metadata
        current_messages = message_list_comp.get_messages()
        current_message_ids = {msg.get('internal_id') for msg in current_messages if msg.get('internal_id')}
        conversation_metadata = self._get_conversation_metadata()
        
        # 1. Handle message list container (StatusFacet)
        container_facet_exists = self._state.get('_has_produced_list_root_add_before', False)
        
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
            "alias": conversation_metadata.get("alias")
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

        # 4. Handle message edits (EventFacets for edit operations)
        # Check for edit/delete operations from MessageListComponent
        if message_list_comp:
            pending_operations = message_list_comp.get_pending_veil_operations()
            
            for operation in pending_operations:
                if operation["operation_type"] == "edit":
                    # Create EventFacet for edit operation
                    edit_facet = self._create_message_edit_facet(operation, conversation_metadata)
                    if edit_facet:
                        facet_operations.append(FacetOperationBuilder.add_facet(edit_facet))
                        
                elif operation["operation_type"] == "delete":
                    # For deletes, we use remove_facet operation
                    facet_operations.append(FacetOperationBuilder.remove_facet(operation["veil_id"]))

        # 5. Detect content changes via fallback comparison
        if message_list_comp:
            current_messages = message_list_comp.get_messages()
            last_message_content = self._state.get('_last_message_content_map', {})
            
            for msg in current_messages:
                internal_id = msg.get('internal_id')
                if internal_id in last_message_content:
                    last_content = last_message_content[internal_id]
                    current_content = msg.get('text', '')
                    
                    if last_content != current_content and msg.get('is_edited', False):
                        # Create EventFacet for detected edit
                        edit_facet = self._create_content_change_edit_facet(msg, last_content, conversation_metadata)
                        if edit_facet:
                            facet_operations.append(FacetOperationBuilder.add_facet(edit_facet))
            
            # Update content tracking for next frame
            self._update_message_content_tracking(current_messages)

        # 6. Generate tool availability ambient facets (when container state changes)
        if facet_operations and any(op.operation_type in ["add_facet", "update_facet"] for op in facet_operations):
            enhanced_tools = self._get_enhanced_tools_for_element()
            if enhanced_tools:
                tools_content = self._format_tools_for_ambient_facet(enhanced_tools)
                tools_ambient_facet = AmbientFacet(
                    facet_id=f"{owner_id}_tools_ambient",
                    owner_element_id=owner_id,
                    ambient_type="tool_instructions",
                    content=tools_content,
                    trigger_threshold=500  # Re-render after 500 symbols
                )
                
                facet_operations.append(FacetOperationBuilder.add_facet(tools_ambient_facet))
                logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] Generated ambient facet for tools")

        # Update state after generating operations
        if not container_facet_exists and any(
            op.operation_type == "add_facet" and 
            op.facet and op.facet.facet_id == list_root_facet_id 
            for op in facet_operations
        ):
            self._state['_has_produced_list_root_add_before'] = True

        self._state['_last_list_root_properties'] = current_container_state
        self._state['_last_generated_veil_message_ids'] = current_message_ids

        if facet_operations:
            logger.info(f"[{owner_id}/{self.COMPONENT_TYPE}] Calculated {len(facet_operations)} facet operations")
        else:
            logger.debug(f"[{owner_id}/{self.COMPONENT_TYPE}] No facet operations calculated")

        return facet_operations if facet_operations else None

    # --- NEW: Rich Delta Generation Methods ---
    def _create_edit_delta(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Create rich edit delta with full context for system messages."""
        edit_details = operation.get("edit_details", {})
        conversation_context = operation.get("conversation_context", {})
        sender_info = operation.get("sender_info", {})
        
        return {
            "op": "update_node",
            "veil_id": operation["veil_id"],
            "properties": {
                "text_content": edit_details.get("new_text"),
                "is_edited": True,
                "edit_timestamp": edit_details.get("edit_timestamp"),
                "last_edited_timestamp": edit_details.get("edit_timestamp"),
                # NEW: Rich context for system message generation
                "edit_context": {
                    "conversation_name": conversation_context.get("conversation_name"),
                    "adapter_type": conversation_context.get("adapter_type"), 
                    "server_name": conversation_context.get("server_name"),
                    "sender_name": sender_info.get("sender_name"),
                    "sender_id": sender_info.get("sender_id"),
                    "original_preview": edit_details.get("original_preview", {}).get("preview"),
                    "original_truncated": edit_details.get("original_preview", {}).get("truncated", False),
                    "original_truncated_count": edit_details.get("original_preview", {}).get("truncated_count", 0),
                    "new_preview": edit_details.get("new_preview", {}).get("preview"), 
                    "new_truncated": edit_details.get("new_preview", {}).get("truncated", False),
                    "new_truncated_count": edit_details.get("new_preview", {}).get("truncated_count", 0),
                    "edit_type": "content_change"
                }
            }
        }

    def _create_delete_delta(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Create rich delete delta with full context for system messages."""
        delete_details = operation.get("delete_details", {})
        conversation_context = operation.get("conversation_context", {})
        sender_info = operation.get("sender_info", {})
        
        return {
            "op": "remove_node",
            "veil_id": operation["veil_id"],
            # NEW: Rich deletion context
            "deletion_context": {
                "conversation_name": conversation_context.get("conversation_name"),
                "adapter_type": conversation_context.get("adapter_type"),
                "server_name": conversation_context.get("server_name"), 
                "sender_name": sender_info.get("sender_name"),
                "sender_id": sender_info.get("sender_id"),
                "original_preview": delete_details.get("original_preview", {}).get("preview"),
                "original_truncated": delete_details.get("original_preview", {}).get("truncated", False),
                "original_truncated_count": delete_details.get("original_preview", {}).get("truncated_count", 0),
                "delete_timestamp": delete_details.get("delete_timestamp"),
                "deletion_source": delete_details.get("deletion_source", "external"),
                "delete_type": "message_removal"
            }
        }

    def _create_content_change_delta(self, msg: Dict[str, Any], last_content: str) -> Dict[str, Any]:
        """Create delta for content changes detected by fallback comparison."""
        conversation_metadata = self._get_conversation_metadata()
        
        return {
            "op": "update_node",
            "veil_id": msg.get('internal_id'),
            "properties": {
                "text_content": msg.get('text', ''),
                "is_edited": True,
                "edit_timestamp": msg.get('last_edited_timestamp', time.time()),
                "last_edited_timestamp": msg.get('last_edited_timestamp', time.time()),
                # Rich context for system message generation
                "edit_context": {
                    "conversation_name": conversation_metadata.get("conversation_name"),
                    "adapter_type": conversation_metadata.get("adapter_type"),
                    "server_name": conversation_metadata.get("server_name"),
                    "sender_name": msg.get('sender_name'),
                    "sender_id": msg.get('sender_id'),
                    "original_preview": last_content[:50] if last_content else "",
                    "original_truncated": len(last_content) > 50 if last_content else False,
                    "original_truncated_count": max(0, len(last_content) - 50) if last_content else 0,
                    "new_preview": msg.get('text', '')[:50],
                    "new_truncated": len(msg.get('text', '')) > 50,
                    "new_truncated_count": max(0, len(msg.get('text', '')) - 50),
                    "edit_type": "content_change_fallback"
                }
            }
        }

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
                content=f"Message edited: {edit_details.get('new_preview', {}).get('preview', 'Unknown')}"
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
    
    def _create_content_change_edit_facet(self, msg: Dict[str, Any], last_content: str, conversation_metadata: Dict[str, Any]) -> Optional[EventFacet]:
        """
        Create EventFacet for content changes detected by fallback comparison.
        
        Args:
            msg: Message data
            last_content: Previous content for comparison
            conversation_metadata: Conversation context metadata
            
        Returns:
            EventFacet for the content change
        """
        try:
            edit_facet = EventFacet(
                facet_id=f"edit_fallback_{msg.get('internal_id')}_{int(time.time() * 1000)}",
                veil_timestamp=ConnectomeEpoch.get_veil_timestamp(),
                owner_element_id=self.owner.id,
                event_type="message_edited",
                content=f"Message content changed: {msg.get('text', '')[:50]}{'...' if len(msg.get('text', '')) > 50 else ''}"
            )
            
            # Add edit-specific properties
            edit_facet.properties.update({
                "original_message_id": msg.get('internal_id'),
                "edit_timestamp": msg.get('last_edited_timestamp', time.time()),
                "sender_name": msg.get('sender_name'),
                "conversation_name": conversation_metadata.get("conversation_name"),
                "adapter_type": conversation_metadata.get("adapter_type"), 
                "server_name": conversation_metadata.get("server_name"),
                "original_preview": last_content[:50] if last_content else "",
                "new_preview": msg.get('text', '')[:50],
                "edit_context": {
                    "original_truncated": len(last_content) > 50 if last_content else False,
                    "new_truncated": len(msg.get('text', '')) > 50,
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

