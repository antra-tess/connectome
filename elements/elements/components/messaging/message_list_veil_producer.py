"""
Message List Veil Producer Component
Generates VEIL representation for a list of messages.
"""
import logging
from typing import Dict, Any, Optional, List

from ..base_component import VeilProducer
# Assuming MessageListComponent is in the same directory
from .message_list import MessageListComponent, MessageType 
from elements.component_registry import register_component

logger = logging.getLogger(__name__)

# VEIL Node Structure Constants (Example)
VEIL_CONTAINER_TYPE = "message_list_container"
VEIL_MESSAGE_NODE_TYPE = "message_item"
VEIL_SENDER_PROP = "sender_name"
VEIL_TIMESTAMP_PROP = "timestamp_iso"
VEIL_CONTENT_PROP = "text_content"
VEIL_INTERNAL_ID_PROP = "connectome_internal_msg_id"
VEIL_EXTERNAL_ID_PROP = "external_msg_id"
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
        logger.debug(f"MessageListVeilProducer initialized for Element {self.owner.id}")

    def _get_message_list_component(self) -> Optional[MessageListComponent]:
        """Helper to get the sibling message list component."""
        return self.get_sibling_component(MessageListComponent)

    def get_full_veil(self) -> Optional[Dict[str, Any]]:
        """
        Generates the complete VEIL structure for the current message list.
        """
        message_list_comp = self._get_message_list_component()
        if not message_list_comp:
            logger.error(f"[{self.owner.id}] Cannot generate VEIL: MessageListComponent not found.")
            return None

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
                    # VEIL_ATTACHMENT_METADATA_PROP: processed_attachments_from_mlc # Store the rich attachment dicts
                    # Let's refine this: the VEIL_ATTACHMENT_METADATA_PROP should probably just be the metadata part,
                    # and the content part should lead to a child node if content exists.
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
                            # "content_preview": str(att_data_from_mlc.get("content"))[:100] # Optional: Careful with large content
                        }
                    }
                    message_node["children"].append(attachment_content_node)
                # else: No content available for this attachment in MessageListComponent's state.
                # The metadata is already in VEIL_ATTACHMENT_METADATA_PROP.

            message_nodes.append(message_node)

        # Create the root container node for the list
        root_veil_node = {
            "veil_id": f"{self.owner.id}_message_list_root",
            "node_type": VEIL_CONTAINER_TYPE,
            "properties": {
                "structural_role": "container",
                "content_nature": "message_list",
                "element_id": self.owner.id,
                "element_name": self.owner.name,
                "message_count": len(message_nodes)
                # Add other container properties/annotations if needed
            },
            "children": message_nodes
        }
        
        # Update state for delta calculation
        self._state['_last_generated_veil_message_ids'] = current_message_ids

        return root_veil_node

    def calculate_delta(self) -> Optional[List[Dict[str, Any]]]:
        """
        Calculates the changes (delta) since the last VEIL generation.
        Basic implementation: detects added or removed messages.
        Does not detect property changes on existing messages yet.
        """
        message_list_comp = self._get_message_list_component()
        if not message_list_comp:
            logger.error(f"[{self.owner.id}] Cannot calculate VEIL delta: MessageListComponent not found.")
            return None

        current_messages = message_list_comp.get_messages()
        current_ids = {msg.get('internal_id') for msg in current_messages if msg.get('internal_id')}
        last_ids = self._state.get('_last_generated_veil_message_ids', set())
        
        if current_ids == last_ids:
             # TODO: Implement checking for property changes within existing messages
             # For now, if IDs match, assume no change. This is incomplete.
             logger.debug(f"[{self.owner.id}] No message additions/removals detected for VEIL delta.")
             return [] # No delta

        delta_operations = []
        parent_veil_id = f"{self.owner.id}_message_list_root" # ID of the container node

        # Detect added messages
        added_ids = current_ids - last_ids
        for msg in current_messages:
            internal_id = msg.get('internal_id')
            if internal_id in added_ids:
                # Create the VEIL node for the new message (similar to get_full_veil)
                message_node = {
                    "veil_id": internal_id,
                    "node_type": VEIL_MESSAGE_NODE_TYPE,
                    "properties": {
                        "structural_role": "list_item",
                        "content_nature": "chat_message",
                        VEIL_SENDER_PROP: msg.get('sender_name', 'Unknown'),
                        VEIL_TIMESTAMP_PROP: msg.get('timestamp'),
                        VEIL_CONTENT_PROP: msg.get('text', ''),
                        VEIL_INTERNAL_ID_PROP: internal_id,
                        VEIL_EXTERNAL_ID_PROP: msg.get('original_external_id'),
                        VEIL_ADAPTER_ID_PROP: msg.get('adapter_id'),
                        VEIL_EDITED_FLAG_PROP: msg.get('is_edited', False),
                        VEIL_ATTACHMENT_METADATA_PROP: msg.get('attachments', []) # Add attachment metadata
                    },
                    "children": []
                }
                delta_operations.append({
                    "op": "add_node",
                    "parent_id": parent_veil_id,
                    "node": message_node,
                    # "position": index # Optional: specify where to insert
                })

        # Detect removed messages
        removed_ids = last_ids - current_ids
        for removed_id in removed_ids:
            delta_operations.append({
                "op": "remove_node",
                "veil_id": removed_id
            })
            
        # TODO: Detect modified messages (property changes)
        # This would involve iterating through common IDs and comparing properties.
        # modified_ids = last_ids.intersection(current_ids)
        # for msg_id in modified_ids:
        #     # Compare properties of old and new message state... generate "update_node" ops
        
        # Update state for next delta calculation
        self._state['_last_generated_veil_message_ids'] = current_ids

        if delta_operations:
            logger.info(f"[{self.owner.id}] Calculated VEIL delta with {len(delta_operations)} operations.")
        return delta_operations

