import abc
from typing import Dict, Any, Optional

class BasePersistenceModule(abc.ABC):
    """
    Abstract base class for modules responsible for saving and loading 
    the state of Elements and Components.
    """

    @abc.abstractmethod
    def save_component_state(self, element_id: str, component_type: str, state: Dict[str, Any]) -> bool:
        """
        Saves the state dictionary for a specific component of an element.

        Args:
            element_id: The unique ID of the element.
            component_type: The type identifier string of the component.
            state: The state dictionary to save (must be serializable).

        Returns:
            True if saving was successful, False otherwise.
        """
        pass

    @abc.abstractmethod
    def load_component_state(self, element_id: str, component_type: str) -> Optional[Dict[str, Any]]:
        """
        Loads the previously saved state dictionary for a specific component.

        Args:
            element_id: The unique ID of the element.
            component_type: The type identifier string of the component.

        Returns:
            The loaded state dictionary, or None if not found or an error occurred.
        """
        pass

    @abc.abstractmethod
    def delete_component_state(self, element_id: str, component_type: str) -> bool:
        """
        Deletes the saved state for a specific component.

        Args:
            element_id: The unique ID of the element.
            component_type: The type identifier string of the component.

        Returns:
            True if deletion was successful or state didn't exist, False on error.
        """
        pass

    @abc.abstractmethod
    def delete_element_state(self, element_id: str) -> bool:
        """
        Deletes all saved state associated with a specific element.

        Args:
            element_id: The unique ID of the element.

        Returns:
            True if deletion was successful or element state didn't exist, False on error.
        """
        pass

    # --- Optional Snapshot Methods (More complex) ---

    # @abc.abstractmethod
    # def save_space_snapshot(self, space_id: str, space_data: Dict[str, Any]) -> bool:
    #     """
    #     Saves a complete snapshot of a space, including its structure and component states.
    #     (Implementation details depend heavily on how space structure is represented).
    #     """
    #     pass

    # @abc.abstractmethod
    # def load_space_snapshot(self, space_id: str) -> Optional[Dict[str, Any]]:
    #     """
    #     Loads a complete snapshot of a space.
    #     """
    #     pass 