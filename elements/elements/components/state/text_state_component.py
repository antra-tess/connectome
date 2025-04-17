import logging
from typing import Dict, Any, Optional

from ..base_component import Component

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextStateComponent(Component):
    """
    A simple component to hold and manage a block of text state for an Element.
    Useful for things like notebooks, scratchpads, or simple text displays.
    """

    COMPONENT_TYPE: str = "state.text"
    DEPENDENCIES = []

    def __init__(self, element: Optional["BaseElement"] = None, initial_text: str = "", **kwargs):
        """
        Initializes the text state component.

        Args:
            element: The Element this component is attached to.
            initial_text: The initial text content.
            **kwargs: Passthrough for BaseComponent.
        """
        super().__init__(element, **kwargs)
        self._state = {
            "text": initial_text
        }
        logger.debug(f"TextStateComponent initialized for element {element.id if element else 'None'}")

    def get_text(self) -> str:
        """Returns the current text content."""
        return self._state.get("text", "")

    def set_text(self, new_text: str) -> None:
        """Overwrites the current text content."""
        if not isinstance(new_text, str):
            logger.error(f"Cannot set text: new_text must be a string, got {type(new_text)}")
            return 
        if self._state.get("text") != new_text:
            self._state["text"] = new_text
            logger.debug(f"Text state updated for element {self.element.id if self.element else 'Unknown'}")
            # Optionally notify element of state change if needed
            # if self.element: self.element.notify_state_changed(self)

    def append_text(self, text_to_append: str, separator: str = "\n") -> None:
        """Appends text to the current content with an optional separator."""
        if not isinstance(text_to_append, str):
             logger.error(f"Cannot append text: text_to_append must be a string, got {type(text_to_append)}")
             return
        if not text_to_append: # Don't append empty strings
             return
             
        current_text = self._state.get("text", "")
        # Add separator only if current text is not empty
        if current_text:
             new_text = current_text + separator + text_to_append
        else:
             new_text = text_to_append
             
        self._state["text"] = new_text
        logger.debug(f"Appended text state for element {self.element.id if self.element else 'Unknown'}")
        # Optionally notify element of state change if needed
        # if self.element: self.element.notify_state_changed(self) 