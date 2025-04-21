import logging
from typing import Optional, List, Dict, Any
import html
import json

# Base classes and dependencies
from ..base import Component, BaseElement # Corrected base imports
from .history import HistoryComponent # Assuming history is in the same dir now?
                                     # Adjust if history is in ../history.py
from ..base_representation_component import BaseRepresentationComponent

logger = logging.getLogger(__name__)

# Change inheritance
class ChatElementRepresentationComponent(BaseRepresentationComponent):
    '''
    Provides a structured representation for chat elements, focusing on
    the history of the active conversation.
    '''
    COMPONENT_TYPE: str = "representation.chat" # More specific type
    # Update dependencies
    DEPENDENCIES = {HistoryComponent.COMPONENT_TYPE} # Depends on HistoryComponent

    def __init__(self, element: Optional[BaseElement] = None,
                 max_history_turns: int = 10, # Max turns to include in representation
                 **kwargs):
        # Initialize Component first, then BaseRepresentation specific things if needed
        super().__init__(element, **kwargs)
        self._max_history_turns = max_history_turns
        logger.debug(f"ChatElementRepresentationComponent initialized for {element.id if element else 'detached'} with max_history_turns={max_history_turns}")

    # Override generate_representation to use the base and add chat specifics
    def generate_representation(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generates structured representation including recent history turns
        from the active conversation.
        """
        # Get the base structure (id, type, name)
        representation = super().generate_representation(options)
        # The _generate_* methods will be called by the super method to fill details
        return representation

    # --- Implement abstract methods --- 

    def _generate_content(self, options: Optional[Dict[str, Any]] = None) -> Any:
        """Content for chat representation is primarily the turn history."""
        history_comp = self.element.get_component(HistoryComponent)
        if not history_comp:
            return {"error": "HistoryComponent not found"}

        active_conversation_id = history_comp.get_active_conversation_id()
        if not active_conversation_id:
            return {"state": "no_active_conversation"}

        try:
            history_entries = history_comp.get_history(
                conversation_id=active_conversation_id,
                include_deleted=False,
                sort_by_timestamp=True
            )

            if self._max_history_turns > 0:
                recent_turns_entries = history_entries[-self._max_history_turns:]
            else:
                recent_turns_entries = history_entries

            # Format turns as dictionaries
            formatted_turns = []
            for entry in recent_turns_entries:
                entry_data = entry.get('data', {})
                turn = {
                    "role": entry_data.get('role', 'unknown'),
                    "content": entry_data.get('text', ''), # Main text content
                    # Add other relevant fields directly
                    "name": entry_data.get('sender_id') or entry_data.get('user_name'),
                    "timestamp": entry.get('timestamp'),
                    "message_id": entry.get('message_id'),
                    "tool_call_id": entry_data.get('tool_call_id'), # Optional
                    "edited_timestamp": entry.get('edited_timestamp'), # Optional
                    "reactions": entry.get('reactions', {}) # Optional, defaults to empty
                }
                # Clean up None values for cleaner representation?
                turn = {k: v for k, v in turn.items() if v is not None}
                formatted_turns.append(turn)
                
            return {
                "active_conversation_id": active_conversation_id,
                "turns": formatted_turns
            }

        except Exception as e:
            logger.exception(f"Error getting history for representation on {self.element.id}: {e}")
            return {"error": "Failed to retrieve history", "details": str(e)}

    def _generate_attributes(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate chat-specific attributes."""
        attrs = {}
        history_comp = self.element.get_component(HistoryComponent)
        if history_comp:
             attrs["active_conversation_id"] = history_comp.get_active_conversation_id()
             try:
                attrs["tracked_conversations_count"] = len(history_comp.list_conversations())
             except Exception:
                 attrs["tracked_conversations_count"] = "Error"
        return attrs

    def _generate_children(self, options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chat representation typically doesn't have nested element children in this way."""
        # If ChatElement *could* contain other elements via ContainerComponent,
        # we might fetch their representations here.
        # For now, assume history turns are the primary "children" content.
        return []

    # _generate_compression_hints can use the default implementation 