"""
Container Component
Manages child elements mounted within a Space.
"""
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING, List, Tuple

from ...base import Component, MountType
from ..base_component import Component as BaseComponent
from ...base import BaseElement
# Import the registry decorator
from elements.component_registry import register_component

if TYPE_CHECKING:
    from ...base import BaseElement

logger = logging.getLogger(__name__)

@register_component
class ContainerComponent(BaseComponent):
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
        logger.debug(f"ContainerComponent initialized for Element {self.owner.id if self.owner else 'Unknown'}")
        return True

    def mount_element(self, element: 'BaseElement', mount_id: Optional[str] = None, mount_type: MountType = MountType.INCLUSION,) -> Tuple[bool, Optional[str]]:
        """
        Mounts a child element.

        Args:
            element: The BaseElement instance to mount.
            mount_id: Optional identifier for the mount point. Defaults to element.id.
            mount_type: The type of mounting (e.g., INCLUSION, UPLINK).

        Returns:
            True if mounting was successful, False otherwise.
        """
        if not self.owner:
            logger.error("Cannot mount element: ContainerComponent has no owner element.")
            return False, None
            
        # Use provided mount_id or generate one from element.id
        actual_mount_id = mount_id if mount_id else element.id

        if actual_mount_id in self._state['_mounted_elements']:
            logger.warning(f"Element with mount_id '{actual_mount_id}' already mounted in {self.owner.id if self.owner else 'Unknown'}.")
            return False, actual_mount_id

        # Set parent on the mounted element
        element._set_parent(self.owner.id if self.owner else 'Unknown', mount_type)
        element._set_mount_id(actual_mount_id)

        self._state['_mounted_elements'][actual_mount_id] = {
            'element': element,
            'mount_type': mount_type
        }
        logger.info(f"[{self.owner.id if self.owner else 'Unknown'}] Element '{element.name}' ({element.id}) mounted as '{actual_mount_id}' (Type: {mount_type}).")
        
        # Record event on the Space's timeline
        if self.owner and hasattr(self.owner, 'add_event_to_timeline'):
            event_payload = {
                'event_type': 'element_mounted',
                'data': {
                    'mount_id': actual_mount_id,
                    'element_id': element.id,
                    'element_name': element.name,
                    'element_type': element.__class__.__name__,
                    'mount_type': mount_type.name if isinstance(mount_type, MountType) else str(mount_type)
                },
                'context': {"component_id": self.id}
            }
            self.owner.add_event_to_timeline(event_payload, timeline_context={})
        
        return True, actual_mount_id

    def unmount_element(self, mount_id: str) -> bool:
        """
        Unmounts a child element.

        Args:
            mount_id: The identifier of the mount point to remove.

        Returns:
            True if unmounting was successful, False otherwise.
        """
        if not self.owner:
            logger.error("Cannot unmount element: ContainerComponent has no owner element.")
            return False

        if mount_id not in self._state['_mounted_elements']:
            logger.warning(f"Element with mount_id '{mount_id}' not found in {self.owner.id if self.owner else 'Unknown'}.")
            return False
        
        mounted_info = self._state['_mounted_elements'][mount_id]
        element_to_unmount = mounted_info["element"]
        
        # Cleanup the element being unmounted
        element_to_unmount.cleanup()
        element_to_unmount._clear_parent()
        
        del self._state['_mounted_elements'][mount_id]
        logger.info(f"Unmounted element '{element_to_unmount.name}' (mount_id: {mount_id}) from {self.owner.id if self.owner else 'Unknown'}.")
        
        # Record event on the Space's timeline
        if self.owner and hasattr(self.owner, 'add_event_to_timeline'):
            event_payload = {
                'event_type': "element_unmounted",
                'data': {
                    "unmounted_element_id": element_to_unmount.id,
                    "unmounted_element_name": element_to_unmount.name,
                    "mount_id": mount_id,
                    "unmounted_from_element_id": self.owner.id
                },
                'context': {"component_id": self.id}
            }
            self.owner.add_event_to_timeline(event_payload, timeline_context={})
            
        return True

    def get_mounted_element(self, mount_id: str) -> Optional['BaseElement']:
        """Gets a mounted element by its mount ID."""
        mounted_info = self._state['_mounted_elements'].get(mount_id)
        return mounted_info["element"] if mounted_info else None

    def get_mounted_elements(self) -> Dict[str, 'BaseElement']:
        """Gets all mounted elements, keyed by their mount ID."""
        return {mount_id: info["element"] for mount_id, info in self._state['_mounted_elements'].items()}

    def get_mounted_elements_with_info(self) -> Dict[str, Dict[str, Any]]:
        """Gets all mounted elements with their mount info (element and mount_type)."""
        return self._state['_mounted_elements'].copy()

    # --- VEIL Production (Optional) ---
    def produce_veil_structure(self) -> List[Dict[str, Any]]:
        """Produces a VEIL structure for the mounted elements."""
        veil_nodes = []
        if not self.owner:
            logger.warning(f"ContainerComponent ({self.id}) cannot produce VEIL: no owner element.")
            return veil_nodes
            
        for mount_id, info in self._state['_mounted_elements'].items():
            element = info.get('element')
            if element:
                veil_nodes.append({
                    'mount_id': mount_id,
                    'element_id': element.id,
                    'element_name': element.name,
                    'element_type': element.__class__.__name__,
                    'mount_type': info.get('mount_type', MountType.UNKNOWN).name
                })
        return veil_nodes
