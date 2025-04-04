"""
Element Factory Component
Component responsible for facilitating Element creation via the central ElementFactory.
"""

import logging
from typing import Dict, Any, Optional, List, Type
import uuid

# Assuming the central factory lives here
from ..factory import ElementFactory
from ..base import BaseElement
from ..base_component import Component

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ElementFactoryComponent(Component):
    """
    Provides an interface for creating new Elements using prefab definitions
    managed by the central ElementFactory.
    Typically used by InnerSpace.
    """

    COMPONENT_TYPE: str = "element_factory"
    DEPENDENCIES: List[str] = []

    # Events this component might handle (e.g., requests to create elements)
    HANDLED_EVENT_TYPES: List[str] = [
        "create_element_request" # Keep this if we want event-driven creation
    ]

    def __init__(self, element: Optional[BaseElement] = None, central_factory: Optional[ElementFactory] = None):
        """
        Initializes the component.

        Args:
            element: The element this component is attached to.
            central_factory: The central ElementFactory instance. This MUST be provided.
        """
        super().__init__(element)
        if central_factory is None:
            # In a real system, might get this via dependency injection or service locator
            logger.warning("Central ElementFactory not provided to ElementFactoryComponent. Creating a default one. This might lead to inconsistencies.")
            self._central_factory = ElementFactory() # Fallback, might not find all components/prefabs
        else:
            self._central_factory = central_factory

        # No internal state needed for registry anymore
        # self._state = {}

    @property
    def central_factory(self) -> ElementFactory:
        """Provides access to the central factory."""
        # Consider if direct access is desired or if methods should wrap it.
        if not self._central_factory:
             # This case should ideally not happen if __init__ enforces it.
             raise ValueError("Central ElementFactory is not configured for this component.")
        return self._central_factory

    def create_element(
        self,
        prefab_name: str,
        element_id: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        # Retain name/description overrides if useful, factory doesn't handle these yet
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[BaseElement]:
        """
        Creates an instance of an element using its prefab definition.

        Args:
            prefab_name: The name of the prefab to instantiate (e.g., "chat_element").
            element_id: Optional unique ID for the new element. If None, factory might generate one.
            initial_state: Optional dictionary to override component configurations.
                           Structure: { "ComponentName": { "config_key": value, ... }, ... }
            name: Optional override for the element's name (takes precedence over prefab name).
            description: Optional override for the element's description.

        Returns:
            An instance of the created element, or None on failure.
        """
        logger.info(f"Request received to create element from prefab '{prefab_name}'")

        # Generate default ID if needed - Or let the central factory handle it?
        # For consistency, maybe let central factory assign ID if None is passed.
        # element_id = element_id or f"{prefab_name.lower()}_{uuid.uuid4().hex[:8]}"

        try:
            new_element = self.central_factory.create_element(
                element_id=element_id, # Pass None if not provided
                prefab_name=prefab_name,
                initial_state=initial_state
            )

            if new_element:
                # Apply name/description overrides if provided *after* creation
                if name is not None:
                    new_element.name = name
                    logger.debug(f"Overrode element {new_element.id} name to '{name}'")
                if description is not None:
                    new_element.description = description # Assuming description is a direct attribute
                    logger.debug(f"Overrode element {new_element.id} description")

                logger.info(f"Successfully created element instance: {new_element.name} ({new_element.id}) from prefab {prefab_name}")
                # Note: Mounting the element is the responsibility of the caller (e.g., InnerSpace)
                return new_element
            else:
                 # Error logged by central_factory
                 return None

        except Exception as e:
            # Catch potential errors from the factory call itself
            logger.error(f"Error calling central factory to create prefab '{prefab_name}': {e}", exc_info=True)
            return None

    # Removed register_element_type, get_element_class, get_registered_types,
    # _register_standard_element_types as they are no longer relevant.

    # Potential event handling logic (optional)
    # def _on_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
    #     """
    #     Handle events like create_element_request.
    #     """
    #     event_type = event.get("event_type")
    #     if event_type == "create_element_request":
    #         data = event.get("data", {})
    #         prefab_name = data.get("prefab_name")
    #         element_id = data.get("element_id")
    #         initial_state = data.get("initial_state")
    #         name = data.get("name")
    #         description = data.get("description")

    #         if prefab_name:
    #             created_element = self.create_element(
    #                 prefab_name=prefab_name,
    #                 element_id=element_id,
    #                 initial_state=initial_state,
    #                 name=name,
    #                 description=description
    #             )
    #             # TODO: Decide if mounting should happen here or be requested separately.
    #             # If mounting here, need access to the ContainerComponent of self.element (InnerSpace)
    #             # if created_element and self.element:
    #             #     container = self.element.get_component(ContainerComponent) # Need ContainerComponent import
    #             #     if container:
    #             #         container.mount_element(created_element)

    #             return created_element is not None
    #         else:
    #             logger.warning("Received create_element_request without prefab_name.")

    #     return False 