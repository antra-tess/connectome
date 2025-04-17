import logging
from typing import Optional
import html # For escaping

from ..base_component import Component
from ..base import BaseElement
from .simple_representation_component import SimpleRepresentationComponent
# Import dependency
from .state.text_state_component import TextStateComponent

logger = logging.getLogger(__name__)

class TextElementRepresentationComponent(SimpleRepresentationComponent):
    '''
    Provides a simple XML representation for elements that primarily use
    a TextStateComponent, including the text content.
    '''
    COMPONENT_TYPE: str = "simple_representation.text"
    # Depends on TextStateComponent to get the content
    DEPENDENCIES = SimpleRepresentationComponent.DEPENDENCIES + [TextStateComponent.COMPONENT_TYPE]

    def __init__(self, element: Optional[BaseElement] = None, max_content_length: int = 500, **kwargs):
        super().__init__(element, **kwargs)
        self._max_content_length = max_content_length

    def produce_representation(self) -> str:
        '''
        Generates XML representation including truncated text content.
        '''
        if not self.element:
            return '<element_representation type="Detached" id="unknown"/>'
        
        # Basic info attributes
        element_type = self.element.__class__.__name__
        element_name = self.element.name or ""
        element_id = self.element.id
        safe_name = html.escape(element_name, quote=True)
        
        # Get content from TextStateComponent
        text_content = "(No TextStateComponent found)"
        text_state_comp = self.element.get_component(TextStateComponent)
        if text_state_comp:
            try:
                text_content = text_state_comp.get_text()
                # Truncate if needed
                if len(text_content) > self._max_content_length:
                     text_content = text_content[:self._max_content_length] + "... (truncated)"
            except Exception as e:
                logger.warning(f"Error getting text from TextStateComponent on {element_id}: {e}")
                text_content = "(Error getting text)"
        
        # Escape content for XML safety
        safe_content = html.escape(text_content)
        
        # Return tag with attributes and content
        return (
            f'<element_representation type="{element_type}" name="{safe_name}" id="{element_id}">\n'
            f'  <content>{safe_content}</content>\n'
            f'</element_representation>'
        ) 