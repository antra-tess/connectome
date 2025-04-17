import logging
from typing import Optional, List, Dict, Any
import html
import json

from ..base_component import Component
from ..base import BaseElement
from .simple_representation_component import SimpleRepresentationComponent
# Import dependency
from .history_component import HistoryComponent

logger = logging.getLogger(__name__)

class ChatElementRepresentationComponent(SimpleRepresentationComponent):
    '''
    Provides a simple XML representation for elements that primarily use
    a HistoryComponent, including recent turns.
    '''
    COMPONENT_TYPE: str = "simple_representation.chat"
    # Depends on HistoryComponent
    DEPENDENCIES = SimpleRepresentationComponent.DEPENDENCIES + [HistoryComponent.COMPONENT_TYPE]

    def __init__(self, element: Optional[BaseElement] = None, 
                 max_history_turns: int = 5, # Max turns to include
                 **kwargs):
        super().__init__(element, **kwargs)
        self._max_history_turns = max_history_turns

    def produce_representation(self) -> str:
        '''
        Generates XML representation including recent history turns.
        '''
        if not self.element:
            return '<element_representation type="Detached" id="unknown"/>'
        
        # Basic info attributes
        element_type = self.element.__class__.__name__
        element_name = self.element.name or ""
        element_id = self.element.id
        safe_name = html.escape(element_name, quote=True)
        
        history_str = "  (No HistoryComponent found)\n"
        history_comp = self.element.get_component(HistoryComponent)
        if history_comp:
            try:
                # 1. Get the active conversation ID
                active_conversation_id = history_comp.get_active_conversation_id()

                if active_conversation_id:
                    # 2. Fetch history for the *active* conversation
                    history_entries = history_comp.get_history(
                        conversation_id=active_conversation_id,
                        include_deleted=False,
                        sort_by_timestamp=True
                    )

                    # Apply turn limit *after* fetching
                    if self._max_history_turns > 0:
                        recent_turns = history_entries[-self._max_history_turns:]
                    else:
                        recent_turns = history_entries # Include all if limit is 0 or negative

                    if recent_turns:
                        # 3. Add active_conversation_id to history tag
                        history_str = f'  <history active_conversation_id="{html.escape(active_conversation_id, quote=True)}">\n'
                        for entry in recent_turns:
                            # 2. Access data within the 'data' field and handle potential missing keys
                            entry_data = entry.get('data', {}) # Default to empty dict if data is missing
                            role = entry_data.get('role', 'unknown') # Example: Extract role
                            content = entry_data.get('text', '') # Example: Extract text content
                            # Determine name - might be sender_id, user_name, etc.
                            name = entry_data.get('sender_id') or entry_data.get('user_name') # Adapt based on actual payload keys
                            tool_call_id = entry_data.get('tool_call_id') # If applicable

                            # Construct the tag attributes
                            tag_attrs = f"role='{html.escape(role)}'"
                            if name: tag_attrs += f" name='{html.escape(str(name), quote=True)}'" # Ensure name is string
                            if tool_call_id: tag_attrs += f" tool_call_id='{html.escape(tool_call_id, quote=True)}'"

                            # 3. Indicate edits
                            edited_timestamp = entry.get('edited_timestamp')
                            if edited_timestamp:
                                tag_attrs += f' edited_timestamp="{edited_timestamp}"'

                            # 4. Show Reactions (as a JSON string attribute for simplicity)
                            reactions = entry.get('reactions')
                            if reactions:
                                try:
                                    reactions_json = json.dumps(reactions)
                                    tag_attrs += f' reactions="{html.escape(reactions_json, quote=True)}"'
                                except TypeError:
                                    logger.warning(f"Could not serialize reactions for entry {entry.get('message_id')} in conv {active_conversation_id}")

                            # Escape content
                            safe_content = str(content)
                            if isinstance(content, (dict, list)): # Handle complex content types
                               try: safe_content = json.dumps(content)
                               except TypeError: pass # Keep string representation if fails
                            safe_content = html.escape(safe_content)

                            history_str += f"    <turn {tag_attrs}>{safe_content}</turn>\n"
                        history_str += "  </history>\n"
                    else:
                        # History exists for the conversation, but it's empty or filtered out
                        history_str = f'  <history active_conversation_id="{html.escape(active_conversation_id, quote=True)}" />\n'
                else:
                    # 4. Handle case where there is no active conversation yet
                    history_str = '  <history state="no_active_conversation" />\n'

            except AttributeError as ae:
                 logger.warning(f"HistoryComponent on {element_id} missing expected method (likely get_active_conversation_id or get_history): {ae}")
                 history_str = "  (HistoryComponent missing expected method)\n"
            except Exception as e:
                logger.exception(f"Error getting history from HistoryComponent on {element_id}: {e}")
                history_str = "  (Error getting history)\n"
        
        # Return tag with attributes and history content
        return (
            f'<element_representation type="{element_type}" name="{safe_name}" id="{element_id}">\n'
            f'{history_str}' # Includes newline
            f'</element_representation>'
        ) 