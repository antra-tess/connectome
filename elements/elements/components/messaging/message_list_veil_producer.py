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

    def calculate_delta(self) -> Optional[List[Dict[str, Any]]]:
        """
        Calculates the changes (delta) since the last VEIL generation.
        Detects added or removed messages and handles the list root node.
        """
        parent_obj = self.owner.get_parent_object()
        agent_name = parent_obj.agent_name if parent_obj else None
        message_list_comp = self._get_message_list_component()
        if not message_list_comp:
            logger.error(f"[{self.owner.id}] Cannot calculate VEIL delta: MessageListComponent not found.")
            return None

        delta_operations = []
        list_root_veil_id = f"{self.owner.id}_message_list_root"

        # Get current messages and their IDs
        current_messages = message_list_comp.get_messages()
        current_message_ids = {msg.get('internal_id') for msg in current_messages if msg.get('internal_id')}

        # Include metadata in current properties for delta tracking
        conversation_metadata = self._get_conversation_metadata()

        # NEW: Get enhanced tool definitions for Phase 1 VEIL enhancement
        enhanced_tools = self._get_enhanced_tools_for_element()
        available_tool_names = self._get_available_tools_for_element()  # Backward compatibility

        # Prepare current properties for the list root node
        current_list_root_properties = {
            "structural_role": "container",
            "content_nature": "message_list",
            "element_id": self.owner.id,
            "element_name": self.owner.name,
            "message_count": len(current_message_ids), # Use count of valid messages
            # ENHANCED: Rich tool metadata for aggregation and rendering
            "available_tools": enhanced_tools,
            # BACKWARD COMPATIBILITY: Simple tool names for existing components
            "available_tool_names": available_tool_names,
            "tool_target_element_id": self.owner.id,  # Explicit target for tools
            # NEW: Include metadata in delta tracking
            "adapter_type": conversation_metadata.get("adapter_type"),
            "server_name": conversation_metadata.get("server_name"), 
            "conversation_name": conversation_metadata.get("conversation_name"),
            "adapter_id": conversation_metadata.get("adapter_id"),
            "external_conversation_id": conversation_metadata.get("external_conversation_id"),
            "alias": conversation_metadata.get("alias")
        }

        # 1. Handle the list root node (add or update)
        if not self._state.get('_has_produced_list_root_add_before', False):
            logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Generating 'add_node' for list root '{list_root_veil_id}'.")

            parent_veil_id_for_list_root = None
            if self.owner and hasattr(self.owner, 'get_parent_info'):
                parent_info = self.owner.get_parent_info()
                parent_space_id = parent_info['parent_id']
                parent_veil_id_for_list_root = f"{parent_space_id}_space_root"
                logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] List root '{list_root_veil_id}' will be parented to space root: {parent_veil_id_for_list_root}")

            add_node_op_for_root = {
                "op": "add_node",
                "node": {
                    "veil_id": list_root_veil_id,
                    "node_type": VEIL_CONTAINER_TYPE,
                    "properties": current_list_root_properties,
                    "children": [] # Children (messages) will be added by subsequent deltas to this parent
                }
            }
            if parent_veil_id_for_list_root:
                add_node_op_for_root["parent_id"] = parent_veil_id_for_list_root

            delta_operations.append(add_node_op_for_root)
            # Flag will be set in signal_delta_produced_this_frame
        else:
            # Root already added, check for property updates on the list root itself
            last_root_props = self._state.get('_last_list_root_properties', {})
            if current_list_root_properties != last_root_props:
                logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Generating 'update_node' for list root '{list_root_veil_id}' properties.")
                delta_operations.append({
                    "op": "update_node",
                    "veil_id": list_root_veil_id,
                    "properties": current_list_root_properties
                })

        # 2. Detect added and removed messages
        last_message_ids = self._state.get('_last_generated_veil_message_ids', set())

        # Detect added messages
        added_message_ids = current_message_ids - last_message_ids
        for msg_data in current_messages:
            internal_id = msg_data.get('internal_id')
            if internal_id in added_message_ids:
                # Create the VEIL node for the new message (similar to get_full_veil)
                # but without extensive attachment processing for delta, keep it lean.
                # Attachment changes would ideally be part of message 'update_node' or specific attachment deltas.
                processed_attachments_from_mlc = msg_data.get('attachments', [])
                message_node_properties = {
                    "structural_role": "list_item",
                    "content_nature": "chat_message",
                    VEIL_SENDER_PROP: msg_data.get('sender_name', 'Unknown'),
                    VEIL_TIMESTAMP_PROP: msg_data.get('timestamp'),
                    VEIL_CONTENT_PROP: msg_data.get('text', ''),
                    VEIL_INTERNAL_ID_PROP: internal_id,
                    VEIL_EXTERNAL_ID_PROP: msg_data.get('original_external_id'),
                    VEIL_ADAPTER_ID_PROP: msg_data.get('adapter_id'),
                    VEIL_EDITED_FLAG_PROP: msg_data.get('is_edited', False),
                    # NEW: Include reaction data for HUD rendering in deltas too
                    "reactions": msg_data.get('reactions', {}),  # Include full reaction dict {emoji: [user_ids]}
                    "message_status": msg_data.get('status', 'received'),  # Include message status for pending states
                    "adapter_type": conversation_metadata.get("adapter_type"),
                    "server_name": conversation_metadata.get("server_name"),
                    "conversation_name": conversation_metadata.get("conversation_name"),
                    "is_agent": msg_data.get('sender_name', 'Unknown') == agent_name,
                    "error_details": msg_data.get('error_details', None),
                    VEIL_ATTACHMENT_METADATA_PROP: [
                        {k: v for k, v in att.items() if k != 'content'}
                        for att in processed_attachments_from_mlc
                    ]
                }
                # Create child nodes for attachments that have content for the delta 'add_node'
                message_children_nodes = []
                for att_data_from_mlc in processed_attachments_from_mlc:
                    attachment_id = att_data_from_mlc.get('attachment_id')
                    if not attachment_id: continue
                    if att_data_from_mlc.get('content') is not None:
                        attachment_content_node = {
                            "veil_id": f"att_{attachment_id}_content_{internal_id}",
                            "node_type": VEIL_ATTACHMENT_CONTENT_NODE_TYPE,
                            "properties": {
                                "structural_role": "attachment_content",
                                "content_nature": att_data_from_mlc.get("content_type", "unknown"),
                                "filename": att_data_from_mlc.get("filename", attachment_id),
                                "content_available": True,
                                "attachment_id": attachment_id,
                                "original_message_veil_id": internal_id,
                                # NEW: Include actual content for multimodal processing
                                "content": att_data_from_mlc.get("content"),  # Direct content access!
                            }
                        }
                        message_children_nodes.append(attachment_content_node)

                delta_operations.append({
                    "op": "add_node",
                    "parent_id": list_root_veil_id,
                    "node": {
                        "veil_id": internal_id,
                        "node_type": VEIL_MESSAGE_NODE_TYPE,
                        "properties": message_node_properties,
                        "children": message_children_nodes
                    }
                })

        # Detect removed messages
        removed_message_ids = last_message_ids - current_message_ids
        for removed_id in removed_message_ids:
            delta_operations.append({
                "op": "remove_node",
                "veil_id": removed_id # This implies removal from its parent (the list root)
            })

        # NEW: Check for edit/delete operations from MessageListComponent
        message_list_comp = self._get_message_list_component()
        if message_list_comp:
            pending_operations = message_list_comp.get_pending_veil_operations()
            
            for operation in pending_operations:
                if operation["operation_type"] == "edit":
                    delta_operations.append(self._create_edit_delta(operation))
                elif operation["operation_type"] == "delete":
                    delta_operations.append(self._create_delete_delta(operation))

        # NEW: Fallback - detect edits by comparing message content
        if message_list_comp:
            current_messages = message_list_comp.get_messages()
            last_message_content = self._state.get('_last_message_content_map', {})
            
            for msg in current_messages:
                internal_id = msg.get('internal_id')
                if internal_id in last_message_content:
                    last_content = last_message_content[internal_id]
                    current_content = msg.get('text', '')
                    
                    if last_content != current_content and msg.get('is_edited', False):
                        # Detected edit that wasn't captured by operation tracking
                        delta_operations.append(self._create_content_change_delta(msg, last_content))
            
            # Update content tracking for next frame
            self._update_message_content_tracking(current_messages)

        # NEW: Add owner tracking to all delta operations
        if delta_operations:
            delta_operations = self._add_owner_tracking_to_delta_ops(delta_operations)
            logger.info(f"[{self.owner.id}/{self.COMPONENT_TYPE}] Calculated VEIL delta with {len(delta_operations)} operations (owner-tracked).")
        else:
            logger.debug(f"[{self.owner.id}/{self.COMPONENT_TYPE}] No VEIL delta operations calculated.")

        # --- Update State After Deltas are Determined ---
        list_root_veil_id = f"{self.owner.id}_message_list_root"
        # Update _has_produced_list_root_add_before
        if not self._state.get('_has_produced_list_root_add_before', False):
            for delta_op in delta_operations: # Check the deltas *this producer* just made
                if delta_op.get("op") == "add_node" and \
                   isinstance(delta_op.get("node"), dict) and \
                   delta_op["node"].get("veil_id") == list_root_veil_id:
                    self._state['_has_produced_list_root_add_before'] = True
                    break

        # Update baseline for the list root's properties (using current_list_root_properties from earlier in this method)
        self._state['_last_list_root_properties'] = current_list_root_properties

        # Update baseline for message IDs (using current_message_ids from earlier in this method)
        self._state['_last_generated_veil_message_ids'] = current_message_ids

        logger.debug(
            f"[{self.owner.id}/{self.COMPONENT_TYPE}] calculate_delta finished. Baseline updated. "
            f"List root props tracked. Message IDs: {len(current_message_ids)}. "
            f"Root add produced: {self._state.get('_has_produced_list_root_add_before', False)}"
        )
        # --- End State Update ---

        return delta_operations

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

