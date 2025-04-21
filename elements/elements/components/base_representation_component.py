import logging
from typing import Optional, Dict, Any, List
import abc # Use Abstract Base Classes

from .base import Component # Assuming base Component is here
from ..base import BaseElement # Assuming BaseElement is here

logger = logging.getLogger(__name__)

class BaseRepresentationComponent(Component, abc.ABC):
    """
    Abstract base class for components that generate structured representations
    of their element's state. Designed to be compatible with future VEIL structures.
    """
    COMPONENT_TYPE = "representation.base" # Generic base type
    # Base representation doesn't introduce new dependencies usually
    DEPENDENCIES = set()

    @abc.abstractmethod
    def generate_representation(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generates a structured representation of the element.

        Args:
            options: Optional dictionary for context-specific generation parameters
                     (e.g., detail level, target format hints).

        Returns:
            A dictionary representing the element, typically including fields like:
            - element_id: str
            - element_type: str
            - element_name: str
            - content: Any (Structured content, depends on element type)
            - attributes: Dict[str, Any] (Metadata or specific attributes)
            - children: List[Dict[str, Any]] (Representations of child/mounted elements or items)
            - compression_hints: Dict[str, Any] (Hints for VEIL/summarization - Optional)
        """
        # Basic structure common to all representations
        if not self.element:
             # Handle detached component case
             return {
                 "element_id": "unknown",
                 "element_type": "Detached",
                 "element_name": "",
                 "content": {"error": "Component is detached from element"},
                 "attributes": {},
                 "children": [],
                 "compression_hints": {}
             }

        base_representation = {
            "element_id": self.element.id,
            "element_type": self.element.__class__.__name__,
            "element_name": self.element.name or "",
            "content": self._generate_content(options),
            "attributes": self._generate_attributes(options),
            "children": self._generate_children(options),
            "compression_hints": self._generate_compression_hints(options),
        }
        return base_representation

    # --- Abstract methods for subclasses to implement ---

    @abc.abstractmethod
    def _generate_content(self, options: Optional[Dict[str, Any]] = None) -> Any:
        """Generates the primary content representation for the element."""
        raise NotImplementedError

    @abc.abstractmethod
    def _generate_attributes(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generates additional metadata attributes for the element."""
        # Default implementation could return basic attributes
        return {}

    @abc.abstractmethod
    def _generate_children(self, options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generates representations for child/mounted elements or items."""
        # Default implementation might return empty list
        return []

    # --- Optional methods for VEIL alignment ---

    def _generate_compression_hints(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generates hints for compression or summarization (Optional)."""
        # Default implementation returns empty dict
        return {}

    # TODO: Consider adding a simple notification mechanism or caching later
    # def notify_representation_changed(self): ...
    # self._cached_representation = None ...


</rewritten_file> 