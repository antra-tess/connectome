"""
Container Component
Manages child elements mounted within a Space.
"""
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

from ...base import Component, MountType

if TYPE_CHECKING:
    from ...base import BaseElement

logger = logging.getLogger(__name__)

class ContainerComponent(Component):
    """
    Manages the collection of child Elements mounted within the owning Space element.
    Maintains the hierarchical structure and provides methods for managing children.
    """
    COMPONENT_TYPE = "ContainerComponent"

    def initialize(self, **kwargs) -> None:
        """Initializes the component state."""
        super().initialize(**kwargs)
        # _mounted_elements structure: { mount_id: {"element": BaseElement, "mount_type": MountType} }
        self._state.setdefault('_mounted_elements', {}) 
        logger.debug(f"ContainerComponent initialized for Element {self.owner.id}")

    def mount_element(self, element: 'BaseElement', mount_id: Optional[str] = None, mount_type: MountType = MountType.INCLUSION) -> bool:
        """
        Mounts a child element.

        Args:
            element: The BaseElement instance to mount.
            mount_id: Optional identifier for the mount point. Defaults to element.id.
            mount_type: The type of mounting (e.g., INCLUSION, UPLINK).

        Returns:
            True if mounting was successful, False otherwise.
        """
        if not hasattr(element, 'id'): # Basic check for element validity
            logger.error(f"[{self.owner.id}] Cannot mount invalid object: {element}")
            return False
            
        final_mount_id = mount_id if mount_id else element.id
        if not final_mount_id:
            logger.error(f"[{self.owner.id}] Cannot mount element: Mount ID cannot be empty (element ID: {element.id}).")
            return False

        if final_mount_id in self._state['_mounted_elements']:
            logger.error(f"[{self.owner.id}] Cannot mount element '{element.id}': Mount ID '{final_mount_id}' already exists.")
            return False

        # Optional: Check if element with same ID is already mounted
        # for mount_info in self._state['_mounted_elements'].values():
        #     if mount_info['element'].id == element.id:
        #         logger.error(f"[{self.owner.id}] Cannot mount element '{element.id}': Element instance already mounted under mount_id '{mount_info['element'].id}'.") # This logic needs mount_id retrieval
        #         return False

        # Set parent relationship
        element.parent = self.owner 

        self._state['_mounted_elements'][final_mount_id] = {
            'element': element,
            'mount_type': mount_type
        }
        logger.info(f"[{self.owner.id}] Element '{element.name}' ({element.id}) mounted as '{final_mount_id}' (Type: {mount_type.name}).")
        
        # Record the mount event in the owner Space's timeline
        if hasattr(self.owner, 'add_event_to_timeline'):
            mount_event_payload = {
                'event_type': 'element_mounted',
                'payload': {
                    'mount_id': final_mount_id,
                    'element_id': element.id,
                    'element_name': element.name,
                    'element_type': element.__class__.__name__,
                    'mount_type': mount_type.name
                }
            }
            # Assume appending to the primary timeline of the owner Space
            timeline_context = {}
            if hasattr(self.owner, 'get_primary_timeline'):
                primary_timeline = self.owner.get_primary_timeline()
                if primary_timeline:
                    timeline_context['timeline_id'] = primary_timeline
            self.owner.add_event_to_timeline(mount_event_payload, timeline_context)
        else:
            logger.warning(f"[{self.owner.id}] Owner Space does not have add_event_to_timeline method. Mount event not recorded.")
            
        return True

    def unmount_element(self, mount_id: str) -> bool:
        """
        Unmounts a child element.

        Args:
            mount_id: The identifier of the mount point to remove.

        Returns:
            True if unmounting was successful, False otherwise.
        """
        mount_info = self._state['_mounted_elements'].get(mount_id)
        if not mount_info:
            logger.warning(f"[{self.owner.id}] Cannot unmount element: Mount ID '{mount_id}' not found.")
            return False

        element_instance = mount_info.get('element')
        if element_instance:
            element_instance.parent = None # Clear parent relationship
            logger.info(f"[{self.owner.id}] Element '{element_instance.name}' ({element_instance.id}) unmounted from '{mount_id}'.")
        else:
            logger.warning(f"[{self.owner.id}] Mount point '{mount_id}' existed but had no element instance.")

        element_id_unmounted = element_instance.id if element_instance else None
        element_name_unmounted = element_instance.name if element_instance else None

        del self._state['_mounted_elements'][mount_id]
        
        # Record the unmount event in the owner Space's timeline
        if hasattr(self.owner, 'add_event_to_timeline'):
            unmount_event_payload = {
                'event_type': 'element_unmounted',
                'payload': {
                    'mount_id': mount_id,
                    'element_id': element_id_unmounted, # May be None if element was missing
                    'element_name': element_name_unmounted
                }
            }
            # Assume appending to the primary timeline
            timeline_context = {}
            if hasattr(self.owner, 'get_primary_timeline'):
                primary_timeline = self.owner.get_primary_timeline()
                if primary_timeline:
                    timeline_context['timeline_id'] = primary_timeline
            self.owner.add_event_to_timeline(unmount_event_payload, timeline_context)
        else:
            logger.warning(f"[{self.owner.id}] Owner Space does not have add_event_to_timeline method. Unmount event not recorded.")
            
        return True

    def get_mounted_element(self, mount_id: str) -> Optional['BaseElement']:
        """Gets a mounted element instance by its mount ID."""
        mount_info = self._state['_mounted_elements'].get(mount_id)
        return mount_info.get('element') if mount_info else None

    def get_mounted_elements(self) -> Dict[str, 'BaseElement']:
        """Gets a dictionary mapping mount_id to mounted element instances."""
        return {mount_id: info['element'] for mount_id, info in self._state['_mounted_elements'].items() if 'element' in info}

    def get_mounted_elements_info(self) -> Dict[str, Dict[str, Any]]:
        """Gets metadata about mounted elements (ID, name, type), keyed by mount ID."""
        info_dict = {}
        for mount_id, info in self._state['_mounted_elements'].items():
            element = info.get('element')
            if element:
                info_dict[mount_id] = {
                    'element_id': element.id,
                    'element_name': element.name,
                    'element_type': element.__class__.__name__,
                    'mount_type': info.get('mount_type', MountType.UNKNOWN).name
                }
        return info_dict
