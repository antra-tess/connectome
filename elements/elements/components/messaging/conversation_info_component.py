import logging
from typing import Optional

from ..base_component import Component
from ..base import BaseElement

logger = logging.getLogger(__name__)

class ConversationInfoComponent(Component):
    '''
    A simple component to store identification information for a conversation 
    associated with an Element (typically a ChatElement), linking it to an 
    external adapter and conversation ID.
    '''
    COMPONENT_TYPE: str = "conversation_info"
    DEPENDENCIES = [] 

    def __init__(self, element: Optional[BaseElement] = None, 
                 adapter_id: Optional[str] = None, 
                 conversation_id: Optional[str] = None,
                 **kwargs):
        super().__init__(element, **kwargs)
        # Store IDs directly in state
        self._state = {
            "adapter_id": adapter_id,
            "conversation_id": conversation_id
        }
        if not adapter_id or not conversation_id:
            logger.warning(f"ConversationInfoComponent on {element.id if element else 'detached'} initialized without adapter_id or conversation_id.")

    def get_adapter_id(self) -> Optional[str]:
        """Returns the associated adapter ID."""
        return self._state.get("adapter_id")

    def get_conversation_id(self) -> Optional[str]:
        """Returns the associated conversation ID."""
        return self._state.get("conversation_id") 