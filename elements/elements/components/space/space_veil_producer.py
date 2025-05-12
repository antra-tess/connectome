import logging
from typing import Dict, Any, Optional, List, Set, Type

# Need Component base class and potentially Space/BaseElement for type hints/checks
from ..base_component import Component, VeilProducer
from ...base import BaseElement # For checking children
from ...space import Space # For checking owner type
# Import the registry decorator
from elements.component_registry import register_component

# Need to know the base VeilProducer type to find on children
# Assuming a base or common ancestor for VEIL producers exists or we use duck typing
# If not, we might need to import specific producers like UplinkVeilProducer etc.
# Let's assume a generic way to find them for now.
# Option 1: Assume a base VeilProducer class exists somewhere (e.g., in veil directory)
# from ..veil.veil_producer import VeilProducer # Ideal if base class exists
# Option 2: Duck typing (check for get_full_veil/calculate_delta methods) - Safer if no base class
# Option 3: If we only expect specific producers, import them.

# Let's use Option 2 (Duck Typing) for now to avoid assuming a base class structure

logger = logging.getLogger(__name__)

# VEIL Node Structure Constants (Example)
VEIL_SPACE_ROOT_TYPE = "space_root" # Generic type for space VEIL root

@register_component
class SpaceVeilProducer(VeilProducer):
    """
    Generates VEIL representation for a Space element, aggregating VEIL
    from its mounted child elements that also have VeilProducers.
    """
    COMPONENT_TYPE = "SpaceVeilProducer"

    # No specific sibling dependencies, relies on owner being a Space
    # and children potentially having VeilProducers.

    def initialize(self, **kwargs) -> None:
        """Initializes the component state for delta tracking."""
        super().initialize(**kwargs)
        # Track the veil_ids/structure of child VEIL roots from the last generation
        self._state.setdefault('_last_child_veil_roots', {}) # { child_element_id: child_root_veil_node }
        # Track properties of the space itself last time
        self._state.setdefault('_last_space_properties', self._get_current_space_properties())
        logger.debug(f"SpaceVeilProducer initialized for Element {self.owner.id}")

    def _get_current_space_properties(self) -> Dict[str, Any]:
        """Extracts properties of the Space element itself for VEIL."""
        # Basic properties - can be expanded
        props = {
            "structural_role": "root",
            "content_nature": "space_summary",
            "element_id": self.owner.id,
            "element_name": self.owner.name,
            "element_type": self.owner.__class__.__name__,
            # Add any other relevant Space-level properties from owner state/metadata
            "is_inner_space": getattr(self.owner, 'IS_INNER_SPACE', False)
        }
        return props

    def _find_child_veil_producers(self, child_element: BaseElement) -> List[Any]:
        """Finds components on a child element that look like VeilProducers (duck typing)."""
        producers = []
        for comp in child_element.get_components().values():
            # Duck typing: Check for required methods
            if hasattr(comp, 'get_full_veil') and callable(comp.get_full_veil) and \
               hasattr(comp, 'calculate_delta') and callable(comp.calculate_delta):
                 producers.append(comp)
        return producers


    def get_full_veil(self) -> Optional[Dict[str, Any]]:
        """
        Generates the complete VEIL structure for the Space, including children.
        """
        if not isinstance(self.owner, Space):
            logger.error(f"[{self.owner.id}] SpaceVeilProducer attached to non-Space element: {type(self.owner)}. Cannot generate VEIL.")
            return None

        owner_id = self.owner.id
        current_space_props = self._get_current_space_properties()

        # Aggregate VEIL from children
        aggregated_children_veils = []
        current_child_veil_roots = {} # Store for delta tracking { child_element_id: child_root_veil }
        processed_child_elements = set()

        mounted_elements = self.owner.get_mounted_elements()

        for mount_id, child_element in mounted_elements.items():
            if child_element.id in processed_child_elements:
                continue
            processed_child_elements.add(child_element.id)

            child_producers = self._find_child_veil_producers(child_element)

            for child_producer in child_producers:
                 try:
                     child_veil = child_producer.get_full_veil()
                     if child_veil:
                         aggregated_children_veils.append(child_veil)
                         current_child_veil_roots[child_element.id] = child_veil # Store the root node
                 except Exception as e:
                     logger.error(f"[{owner_id}] Error getting full VEIL from child producer {child_producer.__class__.__name__} on {child_element.id}: {e}", exc_info=True)

        # Create the root node for the Space
        root_veil_node = {
            "veil_id": f"{owner_id}_space_root",
            "node_type": VEIL_SPACE_ROOT_TYPE,
            "properties": {**current_space_props, "veil_child_count": len(aggregated_children_veils)},
            "children": aggregated_children_veils
        }

        # Update state for delta calculation
        self._state['_last_child_veil_roots'] = current_child_veil_roots
        self._state['_last_space_properties'] = current_space_props

        return root_veil_node


    def calculate_delta(self) -> Optional[List[Dict[str, Any]]]:
        """
        Calculates the changes (delta) for the Space, aggregating child deltas.
        """
        if not isinstance(self.owner, Space):
            logger.error(f"[{self.owner.id}] SpaceVeilProducer attached to non-Space element: {type(self.owner)}. Cannot calculate delta.")
            return None

        owner_id = self.owner.id
        delta_operations = []
        space_root_veil_id = f"{owner_id}_space_root"

        # 1. Calculate delta for Space properties
        last_space_props = self._state.get('_last_space_properties', {})
        current_space_props = self._get_current_space_properties()
        if current_space_props != last_space_props:
             logger.debug(f"[{owner_id}] Detected property change for Space root.")
             delta_operations.append({
                 "op": "update_node",
                 "veil_id": space_root_veil_id,
                 "properties": current_space_props # Send all current properties including annotations
             })

        # 2. Calculate and aggregate deltas from children
        last_child_roots = self._state.get('_last_child_veil_roots', {}) # { child_element_id: child_root_veil_node }
        current_child_roots = {} # { child_element_id: child_root_veil_node }
        processed_child_elements = set()
        mounted_elements = self.owner.get_mounted_elements()

        for mount_id, child_element in mounted_elements.items():
            if child_element.id in processed_child_elements:
                continue
            processed_child_elements.add(child_element.id)

            child_producers = self._find_child_veil_producers(child_element)
            child_element_produced_veil = False

            for child_producer in child_producers:
                 try:
                     # Get child deltas
                     child_deltas = child_producer.calculate_delta()
                     if child_deltas and isinstance(child_deltas, list):
                         # TODO: Adjust parent_id if needed? Assume child deltas are self-contained for now.
                         delta_operations.extend(child_deltas)

                     # Get child's current root VEIL for add/remove detection
                     # Avoid calling get_full_veil again if possible; delta calculation
                     # should ideally be sufficient. We need the child's root veil_id.
                     # This implies child producers should consistently return deltas
                     # relative to their own structure/root.
                     # We mainly need to know if a child VEIL *exists* now vs last time.
                     # Let's try getting the *current* full VEIL to check for existence/root node.
                     # This isn't ideal performance-wise if get_full_veil is expensive.
                     current_child_veil = child_producer.get_full_veil()
                     if current_child_veil:
                         current_child_roots[child_element.id] = current_child_veil
                         child_element_produced_veil = True

                 except Exception as e:
                     logger.error(f"[{owner_id}] Error calculating delta/getting VEIL from child producer {child_producer.__class__.__name__} on {child_element.id}: {e}", exc_info=True)

            # If this child existed last time but doesn't produce VEIL now (or producer removed)
            if child_element.id in last_child_roots and not child_element_produced_veil:
                 removed_node_id = last_child_roots[child_element.id].get("veil_id")
                 if removed_node_id:
                     logger.debug(f"[{owner_id}] Child element {child_element.id} VEIL root {removed_node_id} removed.")
                     delta_operations.append({"op": "remove_node", "veil_id": removed_node_id})

        # 3. Detect newly added child VEIL roots
        added_child_element_ids = set(current_child_roots.keys()) - set(last_child_roots.keys())
        for added_id in added_child_element_ids:
             added_node = current_child_roots[added_id]
             logger.debug(f"[{owner_id}] Child element {added_id} VEIL root {added_node.get('veil_id')} added.")
             delta_operations.append({
                 "op": "add_node",
                 "parent_id": space_root_veil_id, # Add it to the space's root
                 "node": added_node
             })

        # Update state for next delta calculation
        self._state['_last_child_veil_roots'] = current_child_roots
        self._state['_last_space_properties'] = current_space_props

        if delta_operations:
            logger.info(f"[{owner_id}] SpaceVeilProducer calculated delta with {len(delta_operations)} operations.")
        return delta_operations
