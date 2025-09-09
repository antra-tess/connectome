"""
Inner Space
Special space element representing the agent's subjective experience.
Uses the component architecture pattern consistently.
"""

import logging
from typing import Dict, Any, Optional, List, Type, Callable, Set
import inspect
import time
import re # For _generate_safe_id_string
import asyncio
import copy

from .space import Space  # Inherit from Space to get Container and Timeline functionality
from .base import BaseElement, MountType

# Import core components needed by InnerSpace
from .components.tool_provider import ToolProviderComponent 
from .components.factory_component import ElementFactoryComponent
from .components.base_component import VeilProducer, Component
from .components.hud.facet_aware_hud_component import FacetAwareHUDComponent # NEW: Import FacetAware HUD for VEILFacet architecture
from .components.uplink_manager_component import UplinkManagerComponent # NEWLY ADDED
from .components.veil_facet_compression_engine import VEILFacetCompressionEngine # NEW: Import VEILFacet CompressionEngine

# Import the specific Space Veil Producer
from .components.space.space_veil_producer import SpaceVeilProducer

# Import agent loop components
from .components.agent_loop import BaseAgentLoopComponent, SimpleRequestResponseLoopComponent
from .components.agent_loop.heartbeat_component import HeartbeatComponent
from .components.agent_loop.decider_component import ActivationDeciderComponent
from .components.agent_loop.interrupt_decider_component import InterruptDeciderComponent

# Import messaging components
from .components.messaging.message_list import MessageListComponent

# Type checking imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from llm.provider_interface import LLMProviderInterface
    from host.event_loop import OutgoingActionCallback
    # from elements.space_registry import SpaceRegistry # No longer needed for type hint here

# --- Agent cycle callback type removed - AgentLoop will self-trigger ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InnerSpace(Space):
    """
    Special space element representing the agent's subjective experience.
    
    Inherits from Space (which provides ContainerComponent and TimelineComponent).
    Adds agent-specific components like HUDComponent, ContextManagerComponent, 
    AgentLoopComponent, and tools for managing the agent's experience.
    """
    
    # Identify this as an InnerSpace (useful for type checking)
    IS_INNER_SPACE = True
    
    # Additional event types specific to InnerSpace
    EVENT_TYPES = Space.EVENT_TYPES + [
        "attention_requested",
        "attention_cleared",
        "agent_loop_cycle_start",
        "agent_loop_cycle_end",
        "agent_message_sent",
        "agent_message_received",
        "tool_executed"
    ]
    
    def __init__(
        self,
        element_id: str,
        name: str,
        agent_name: str,
        agent_description: str,
        description: str,
        agent_id: str,
        llm_provider: 'LLMProviderInterface',
        agent_loop_component_type: Type[BaseAgentLoopComponent] = SimpleRequestResponseLoopComponent,
        agent_purpose: Optional[str] = None,
        outgoing_action_callback: Optional['OutgoingActionCallback'] = None,
        # space_registry: Optional['SpaceRegistry'] = None, # REMOVE PARAMETER
        additional_components: Optional[List[Type[Component]]] = None,
        **kwargs
    ):
        """
        Initialize the InnerSpace with agent-specific components.
        
        Args:
            element_id: Unique identifier for this space
            name: Human-readable name for this space
            description: Description of this space's purpose
            agent_id: The unique ID of the agent this InnerSpace belongs to.
            llm_provider: Interface to the LLM powering the agent
            agent_loop_component_type: Type of AgentLoopComponent to use
            agent_purpose: Optional description of the agent's purpose
            outgoing_action_callback: Callback function for sending actions to external systems
            # space_registry: Reference to the SpaceRegistry instance # REMOVE FROM DOCSTRING
            additional_components: Optional list of additional component types to add
            **kwargs: Additional keyword arguments for Space initialization
        """
        # Initialize the base Space (adds ContainerComponent, TimelineComponent, and now ElementFactoryComponent)
        # Pass outgoing_action_callback to Space constructor so it can give it to its ElementFactoryComponent
        super().__init__(element_id=element_id, name=name, description=description, 
                         outgoing_action_callback=outgoing_action_callback, # PASS TO PARENT
                         **kwargs)
        
        self.agent_id = agent_id
        self.agent_name = agent_name
        self._llm_provider = llm_provider
        self.agent_description = agent_description
        
        # NEW: Adapter tracking for mention-based activation
        self._adapter_mappings: Dict[str, Dict[str, str]] = {}  # {adapter_type: {adapter_name: adapter_id}}
        self._agent_adapter_ids: Set[str] = set()  # Known adapter_ids for this agent
        
        # self._outgoing_action_callback = outgoing_action_callback # Already set by super().__init__ if Space stores it
        # self._space_registry = space_registry # REMOVE ATTRIBUTE
        
        self._tool_provider = None
        self._uplink_manager = None
        self._compression_engine = None  # NEW: Add CompressionEngine reference
        self._hud = None
        self._agent_loop = None
        
        logger.info(f"Adding core components to InnerSpace: {name} ({element_id}) for agent {agent_id}")
                    
        self._tool_provider = self.get_component_by_type(ToolProviderComponent)
        self._element_factory = self.get_component_by_type(ElementFactoryComponent)
        if not self._element_factory:
            logger.error(f"CRITICAL: ElementFactoryComponent was not added by Space superclass to InnerSpace {self.id}")
        else:
            # Ensure space_registry is set on the factory if it has a setter and Space.__init__ didn't pass it
            # This is no longer needed as ElementFactoryComponent will get SpaceRegistry.get_instance() itself.
            # if self._space_registry and hasattr(self._element_factory, 'set_space_registry') and callable(getattr(self._element_factory, 'set_space_registry')):
            #     # Check if it already has it (e.g. if Space init passed it via kwargs successfully)
            #     if not (hasattr(self._element_factory, '_space_registry_for_created_elements') and getattr(self._element_factory, '_space_registry_for_created_elements')):
            #         self._element_factory.set_space_registry(self._space_registry)
            #         logger.info(f"Set space_registry on ElementFactoryComponent for InnerSpace {self.id}")
            
            # outgoing_action_callback should have been handled by Space.__init__ or its setter fallback.
            # self._propagate_callback_to_component(ElementFactoryComponent, self._outgoing_action_callback) # Already done by Space or via __init__

            # --- Automatically create a default scratchpad for the agent ---
            scratchpad_id = f"scratchpad_{self.agent_id}"
            logger.info(f"[{self.id}] Attempting to create default scratchpad element '{scratchpad_id}'.")
            scratchpad_config = {
                "name": "Agent Scratchpad",
                "description": f"Default scratchpad for agent {self.agent_name}"
            }
            creation_result = self._element_factory.handle_create_element_from_prefab(
                element_id=scratchpad_id,
                prefab_name="simple_scratchpad",
                element_config=scratchpad_config 
            )
            if creation_result and creation_result.get("success"):
                logger.info(f"[{self.id}] Successfully created and mounted default scratchpad element '{scratchpad_id}'.")
            else:
                error_msg = creation_result.get("error", "Unknown error") if creation_result else "Factory returned None"
                logger.error(f"[{self.id}] Failed to create default scratchpad for agent {self.agent_id}: {error_msg}")
            
        # Add UplinkManagerComponent
        self._uplink_manager = self.add_component(UplinkManagerComponent)
        if not self._uplink_manager:
            logger.error(f"Failed to add UplinkManagerComponent to InnerSpace {self.id}")

        # NEW: Add VEILFacetCompressionEngine - Required for agent memory and VEILFacet-native context management
        self._compression_engine = self.add_component(VEILFacetCompressionEngine)
        if not self._compression_engine:
            logger.error(f"Failed to add VEILFacetCompressionEngine to InnerSpace {self.id}")
        else:
            logger.info(f"VEILFacetCompressionEngine successfully added to InnerSpace {self.id}")

        # Add FacetAware HUD component for VEILFacet temporal stream rendering
        hud_kwargs = {'llm_provider': llm_provider} if llm_provider else {}
        self._hud = self.add_component(FacetAwareHUDComponent, **hud_kwargs)
        if not self._hud:
            logger.error(f"Failed to add FacetAwareHUDComponent to InnerSpace {self.id}")
        else:
            logger.info(f"FacetAwareHUDComponent successfully added to InnerSpace {self.id}")
        
        # NEW: Add Decider and Heartbeat components
        self._activation_decider = self.add_component(ActivationDeciderComponent)
        if not self._activation_decider:
            logger.error(f"Failed to add ActivationDeciderComponent to InnerSpace {self.id}")
        else:
            logger.info(f"ActivationDeciderComponent successfully added to InnerSpace {self.id}")
        
        self._interrupt_decider = self.add_component(InterruptDeciderComponent)
        if not self._interrupt_decider:
            logger.error(f"Failed to add InterruptDeciderComponent to InnerSpace {self.id}")
        else:
            logger.info(f"InterruptDeciderComponent successfully added to InnerSpace {self.id}")
        
        self._heartbeat = self.add_component(HeartbeatComponent)
        if not self._heartbeat:
            logger.error(f"Failed to add HeartbeatComponent to InnerSpace {self.id}")
        else:
            logger.info(f"HeartbeatComponent successfully added to InnerSpace {self.id}")
        
        # Add MessageListComponent to handle message_received events
        self._message_list = self.add_component(MessageListComponent)
        if not self._message_list:
            logger.error(f"Failed to add MessageListComponent to InnerSpace {self.id}")
        else:
            logger.info(f"MessageListComponent successfully added to InnerSpace {self.id}")
        
        # Add any additional requested components
        if additional_components:
            for component_type in additional_components:
                if component_type in [ToolProviderComponent, ElementFactoryComponent, 
                                     UplinkManagerComponent, # NEWLY ADDED
                                     VEILFacetCompressionEngine, # NEW: Add VEILFacet compression engine to skip list
                                     # GlobalAttentionComponent, # REMOVED
                                     FacetAwareHUDComponent, # NEW: Add FacetAware HUD to skip list
                                     MessageListComponent, # Add MessageListComponent to skip list
                                     ActivationDeciderComponent, InterruptDeciderComponent, HeartbeatComponent # Add decider components to skip list
                                     # ContextManagerComponent # REMOVED
                                     ]:
                    # Skip components that we already added
                    logger.warning(f"Skipping duplicate component: {component_type.__name__}")
                    continue
                
                self.add_component(component_type)
        
        # --- Add Agent Loop Component Last (depends on other components) ---
        # Prepare kwargs for the agent loop component
        agent_loop_kwargs = {
            "parent_inner_space": self, # Directly pass self as parent_inner_space
            "agent_loop_name": f"{self.name}_AgentLoop" # Provide a name for the loop
        }
        # The llm_provider and outgoing_action_callback are accessed by the loop *through* parent_inner_space.
        # Additional INJECTED_DEPENDENCIES can be added here if a specific loop type requires them directly.
        
        # Add the agent loop component
        self._agent_loop = self.add_component(agent_loop_component_type, **agent_loop_kwargs)
        if not self._agent_loop:
            logger.error(f"Failed to add AgentLoopComponent to InnerSpace {self.id}")
            
        # Propagate the outgoing action callback if provided
        if outgoing_action_callback:
            self.set_outgoing_action_callback(outgoing_action_callback)
            
        # --- Register Tools from Components ---
        self._register_component_tools()
        # -------------------------------------
            
        logger.info(f"InnerSpace initialization complete: {name} ({element_id}) for agent {agent_id}")
    
    def set_outgoing_action_callback(self, callback: 'OutgoingActionCallback') -> None:
        """
        Set or update the outgoing action callback for all components that need it.
        
        Args:
            callback: The callback function for sending actions to external systems
        """
        logger.info(f"Setting outgoing action callback for InnerSpace {self.id}")
        self._outgoing_action_callback = callback
        
        # Propagate to components that might need it
        self._propagate_callback_to_component(ToolProviderComponent, callback)
        self._propagate_callback_to_component(ElementFactoryComponent, callback)
        self._propagate_callback_to_component(UplinkManagerComponent, callback) # NEWLY ADDED
        self._propagate_callback_to_component(BaseAgentLoopComponent, callback)
        # Add other components as needed
    
    def _propagate_callback_to_component(self, component_type: Type[Component], callback: 'OutgoingActionCallback') -> None:
        """
        Helper method to propagate a callback to a component if it exists and has the setter.
        
        Args:
            component_type: Type of component to find
            callback: Callback to set
        """
        # Find component by base class if needed
        component = self.get_component_by_type(component_type)
        if component and hasattr(component, 'set_outgoing_action_callback'):
            try:
                component.set_outgoing_action_callback(callback)
                logger.debug(f"Propagated callback to {component.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error setting callback on {component.__class__.__name__}: {e}")
    
    # --- Tool Registration from Components ---
    def _register_component_tools(self) -> None:
        """
        Iterates through attached components and registers methods 
        marked as tools (e.g., starting with 'handle_') with the 
        main ToolProviderComponent.
        """
        tool_provider = self.get_tool_provider()
        if not tool_provider:
            logger.error(f"[{self.id}] Cannot register component tools: ToolProviderComponent not found.")
            return

        logger.debug(f"[{self.id}] Scanning attached components for tools to register...")
        registered_count = 0
        skipped_count = 0

        for comp_id, component in self.get_components().items():
            # Skip the main tool provider itself
            if component is tool_provider:
                continue
                
            # Inspect methods of the component
            for attr_name in dir(component):
                if attr_name.startswith("handle_"):
                    try:
                        attr_value = getattr(component, attr_name)
                        if callable(attr_value) and not attr_name.startswith('_'): # Ensure it's a callable public method
                            # Derive tool name from method name
                            tool_name = attr_name.replace("handle_", "")
                            # TODO: How to get description and parameter info?
                            # Option 1: Docstrings (requires parsing)
                            # Option 2: Decorator on the handle_ methods
                            # Option 3: Separate registration dict in component
                            # Let's use docstring for now, assuming first line is description
                            docstring = inspect.getdoc(attr_value) or ""
                            description = docstring.split('\n')[0] if docstring else f"Tool '{tool_name}' from {component.COMPONENT_TYPE}"
                            
                            # --- REVERTED: Inspect parameters ---
                            # Parameter descriptions are harder from docstring reliably
                            param_descriptions = {} # Placeholder
                            # --- END REVERTED ---
                            
                            # Check if tool already exists (e.g. from another component automatically registering it)
                            if tool_name in tool_provider.list_tools():
                                 logger.debug(f"Tool '{tool_name}' already registered. Skipping registration from {component.COMPONENT_TYPE}.")
                                 skipped_count += 1
                                 continue
                                 
                            # Register the tool
                            tool_provider.register_tool_function(
                                name=tool_name,
                                description=description,
                                parameters_schema=[], # Changed from parameter_descriptions=param_descriptions
                                tool_func=attr_value
                            )
                            registered_count += 1
                            logger.debug(f"Registered tool '{tool_name}' from {component.COMPONENT_TYPE} ({component.id})")
                    except Exception as e:
                        logger.error(f"Error inspecting/registering method {attr_name} from component {component.id}: {e}", exc_info=True)
        
        logger.info(f"[{self.id}] Component tool registration complete. Registered: {registered_count}, Skipped (already exist): {skipped_count}.")

    # --- Action Execution --- 
    async def execute_element_action(
        self,
        space_id: str,
                               element_id: str, 
                               action_name: str, 
                               parameters: Dict[str, Any],
        calling_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes an action (tool) on a target element, potentially in another space.
        Records the result back to this InnerSpace's timeline.
        
        Args:
            space_id: ID of the Space containing the target element.
            element_id: ID of the target element.
            action_name: Name of the action/tool to execute.
            parameters: Dictionary of parameters for the action.
            calling_context: Dictionary containing context about the caller, expected to include:
                             - 'loop_component_id': ID of the AgentLoopComponent instance.
                             - 'initiating_event_id': (Optional) ID of the event that triggered this action.
            
        Returns:
            A dictionary containing the result of the action execution, typically:
            { "success": bool, "result": Any, "error": Optional[str] }
        """
        calling_context = calling_context or {}
        loop_component_id = calling_context.get("loop_component_id")
        if not loop_component_id:
             # Log warning but proceed - result won't be linked to a specific loop instance
             logger.warning(f"[{self.id}] execute_element_action called without loop_component_id in calling_context.")

        action_result = { "success": False, "result": None, "error": "Initialization error" }
        target_space = None
        target_element = None
        tool_provider = None
        space_registry_instance = None # For clarity

        try:
            # 0. Get SpaceRegistry instance
            from elements.space_registry import SpaceRegistry # Local import for usage
            space_registry_instance = SpaceRegistry.get_instance()
            if not space_registry_instance:
                # This should ideally not happen if get_instance() works correctly
                raise RuntimeError("Failed to get SpaceRegistry instance.")

            # 1. Find Target Space
            if space_id == self.id:
                target_space = self
            # elif self._space_registry: # Change to use space_registry_instance
            #     target_space = self._space_registry.get_space(space_id)
            else:
                target_space = space_registry_instance.get_space(space_id)
            
            if not target_space:
                raise ValueError(f"Target space '{space_id}' not found.")

            # 2. Find Target Element in Target Space
            # Use get_element_by_id if space provides it, otherwise assume component access
            if hasattr(target_space, 'get_element_by_id'):
                 target_element = target_space.get_element_by_id(element_id)
            elif hasattr(target_space, 'get_mounted_elements'): # Fallback check mounted
                 mounted = target_space.get_mounted_elements()
                 target_element = mounted.get(element_id) # Assumes element_id is mount_id?
                 # TODO: Need better way to find element within a Space if not using get_element_by_id
            
            if not target_element:
                 # Final check: is the target element the space itself?
                 if target_space.id == element_id:
                      target_element = target_space
                 else:
                      raise ValueError(f"Target element '{element_id}' not found in space '{space_id}'.")

            # 3. Find Tool Provider on Target Element
            # Assuming ToolProviderComponent exists and has the standard type name
            tool_provider = target_element.get_component_by_type("ToolProviderComponent")
            if not tool_provider:
                 raise ValueError(f"ToolProviderComponent not found on target element '{element_id}'.")
                 
            # 4. Execute Tool
            logger.info(f"Executing action '{action_name}' on element '{element_id}' in space '{space_id}'.")
            # AWAIT THE ASYNC CALL to tool_provider.execute_tool
            action_result = await tool_provider.execute_tool(tool_name=action_name, calling_context=calling_context, **parameters)
            logger.info(f"Action '{action_name}' executed. Success: {action_result.get('success')}")

        except Exception as e:
            error_msg = f"Error during action execution: {e}"
            logger.error(f"[{self.id}] {error_msg}", exc_info=True)
            action_result["success"] = False
            action_result["error"] = error_msg
            action_result["result"] = None

        # 5. Record Result Event on *this* InnerSpace's Timeline
        try:
            timeline_comp = self.get_timeline() # Get own timeline component
            if timeline_comp:
                # Ensure this payload structure matches what handle_event expects
                # and what mark_agent_for_cycle needs
                result_event_payload_data = {
                    "loop_component_id": loop_component_id, # Link back to the loop
                    "original_action_request": { # Include original request details
                         "space_id": space_id,
                         "element_id": element_id,
                         "action_name": action_name,
                         "parameters": parameters # Be careful about recording sensitive params
                    },
                    "execution_result": action_result # Include the full result
                }
                result_event_payload_outer = {
                    "event_type": "tool_result_received", # This is the key type handle_event checks
                    "data": result_event_payload_data
                }
                
                # Add event to primary timeline
                # The timeline component should wrap this in the standard event structure
                # { "event_id": ..., "timestamp": ..., "payload": result_event_payload_outer }
                event_id = timeline_comp.add_event_to_primary_timeline(result_event_payload_outer)
                if event_id:
                     logger.info(f"Recorded tool result event '{event_id}' to timeline for loop {loop_component_id}.")
                else:
                     logger.error(f"Failed to record tool result event to timeline.")
            else:
                logger.error(f"Cannot record tool result: TimelineComponent not found on InnerSpace '{self.id}'.")
        except Exception as record_err:
            logger.error(f"[{self.id}] Critical error recording tool result event to timeline: {record_err}", exc_info=True)

        return action_result

    # --- Attention Management Methods ---
    def request_attention(self, source_element_id: str, request_data: Dict[str, Any] = None) -> None:
        """
        Register an attention request from an element.
        
        Args:
            source_element_id: ID of the element requesting attention
            request_data: Optional additional data about the request
        """
        # if self._global_attention: # REMOVED
        #     self._global_attention.register_attention_request(source_element_id, request_data or {}) # REMOVED
        # else: # REMOVED
        logger.warning(f"Cannot register attention request: GlobalAttentionComponent functionality removed or unavailable.")
    
    def clear_attention(self, source_element_id: str) -> None:
        """
        Clear an attention request from an element.
        
        Args:
            source_element_id: ID of the element clearing its attention request
        """
        # if self._global_attention: # REMOVED
        #     self._global_attention.clear_attention_request(source_element_id) # REMOVED
        # else: # REMOVED
        logger.warning(f"Cannot clear attention request: GlobalAttentionComponent functionality removed or unavailable.")
            
    def get_attention_requests(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all current attention requests.
            
        Returns:
            Dictionary mapping element IDs to their attention requests
        """
        # if self._global_attention: # REMOVED
        #     return self._global_attention.get_attention_requests() # REMOVED
        logger.warning(f"Cannot get attention requests: GlobalAttentionComponent functionality removed or unavailable.")
        return {}
    
    # --- Agent Loop Control Methods ---
    def start_agent_loop(self) -> bool:
        """
        Start the agent's reasoning loop.
        
        Returns:
            True if the loop was successfully started, False otherwise
        """
        if self._agent_loop and hasattr(self._agent_loop, 'start_loop'):
            return self._agent_loop.start_loop()
        logger.error(f"Cannot start agent loop: AgentLoopComponent unavailable or lacks start_loop method")
        return False
        
    def pause_agent_loop(self) -> bool:
        """
        Pause the agent's reasoning loop.
        
        Returns:
            True if the loop was successfully paused, False otherwise
        """
        if self._agent_loop and hasattr(self._agent_loop, 'pause_loop'):
            return self._agent_loop.pause_loop()
        logger.error(f"Cannot pause agent loop: AgentLoopComponent unavailable or lacks pause_loop method")
        return False
    
    def trigger_agent_cycle(self) -> bool:
        """
        Manually trigger an agent reasoning cycle.
        
        Returns:
            True if the cycle was successfully triggered, False otherwise
        """
        if self._agent_loop and hasattr(self._agent_loop, 'run_cycle'):
            return self._agent_loop.run_cycle()
        logger.error(f"Cannot trigger agent cycle: AgentLoopComponent unavailable or lacks run_cycle method")
        return False
    
    # --- Convenience Getters ---
    def get_agent_loop_component(self) -> Optional[BaseAgentLoopComponent]:
        """Get the agent loop component."""
        return self._agent_loop

    def get_uplink_manager(self) -> Optional[UplinkManagerComponent]:
        return self._uplink_manager

    def get_hud(self) -> Optional[FacetAwareHUDComponent]:
        return self._hud

    def get_compression_engine(self) -> Optional[VEILFacetCompressionEngine]:
        """Get the VEILFacet compression engine component."""
        return self._compression_engine

    def handle_deltas_from_uplink(self, uplink_id: str, deltas: List[Dict[str, Any]]):
        """
        Receives VEIL deltas from a specific UplinkProxy that has new data from its remote space.
        This can be used to trigger HUD updates or other reactive logic within the InnerSpace.
        Args:
            uplink_id: The ID of the UplinkProxy element that sourced these deltas.
            deltas: A list of VEIL delta dictionaries.
        """
        logger.info(f"[{self.id}] Received {len(deltas)} deltas from Uplink '{uplink_id}'.")
        
        # Example: Notify HUDComponent or AgentLoopComponent
        hud_comp = self.get_hud()
        if hud_comp:
            logger.debug(f"[{self.id}] Informing HUD component about new deltas from {uplink_id}. (Placeholder)")

        if self._agent_loop and deltas: 
            event_payload_for_cycle = {
                "trigger_type": "uplink_delta_received",
                "source_uplink_id": uplink_id,
                "delta_count": len(deltas)
            }
            current_time = time.time()
            logger.info(f"[{self.id}] Marking agent '{self.agent_id}' for cycle due to deltas from uplink '{uplink_id}'.")
            self._agent_loop.run_cycle()

    # --- VEIL Rendering & Context Generation (Delegated to HUDComponent) ---
    def get_rendered_veil(self, for_round_id: Optional[str] = None) -> Optional[str]:
        if self._hud:
            return self._hud.get_rendered_veil(for_round_id)
        logger.warning(f"[{self.id}] HUDComponent not available, cannot render VEIL for InnerSpace {self.id}")
        return None

    # --- Event Handling ---
    def handle_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handle events specific to InnerSpace or delegate to Space.
        This method now also includes logic to mark the agent for a cycle
        when a 'tool_result_received' event specific to this InnerSpace occurs.
        """
        event_type = event.get("event_type")
        event_payload = event.get("payload", {})
        handled_by_self = False

        # Specific InnerSpace event handling
        if event_type == "tool_result_received":
            # Check if this result is for an action initiated by *this* InnerSpace's agent
            # The calling_context in the tool_result_received event should have loop_component_id
            # which belongs to this InnerSpace's agent loop.
            # For now, we assume any tool_result_received on the InnerSpace's timeline might be relevant.
            # A more precise check could be added if loop_component_id is available in event_payload.
            
            logger.info(f"[{self.id}] InnerSpace received 'tool_result_received'. Payload: {event_payload}")
            if self._agent_loop:
                current_time = time.time()
                # Pass the original tool_result_received payload as context for the cycle
                logger.info(f"[{self.id}] Marking agent '{self.agent_id}' for cycle due to tool_result_received.")
                self._agent_loop.run_cycle()
                handled_by_self = True # Considers this aspect handled
            else:
                logger.warning(f"[{self.id}] Received tool_result_received but _agent_loop is not set.")

        return handled_by_self # Indicates if InnerSpace-specific logic handled it.

    # --- Connection and Uplink Management ---
    def get_connected_spaces(self) -> List[str]:
        """
        Retrieves a list of remote space IDs this InnerSpace is connected to via Uplinks.
        Delegates to UplinkManagerComponent.
        """
        if self._uplink_manager:
            # Assuming UplinkManagerComponent.list_active_uplinks_tool() returns a dict
            # with an "active_uplinks" key, which is a list of dicts.
            uplinks_result = self._uplink_manager.list_active_uplinks_tool()
            if uplinks_result.get("success"):
                return [uplink_info.get("remote_space_id") for uplink_info in uplinks_result.get("active_uplinks", []) if uplink_info.get("remote_space_id")]
            else:
                logger.error(f"[{self.id}] Failed to list active uplinks from UplinkManager: {uplinks_result.get('error')}")
        return []

    def get_uplink_for_space(self, remote_space_id: str) -> Optional[BaseElement]:
        """
        Gets the UplinkProxy element for a given remote_space_id.
        Delegates to UplinkManagerComponent.
        """
        if self._uplink_manager:
            return self._uplink_manager.get_uplink_for_space(remote_space_id)
        logger.error(f"[{self.id}] get_uplink_for_space: UplinkManagerComponent not available.")
        return None

    def ensure_uplink_to_shared_space(
        self, 
        shared_space_id: str, 
        shared_space_name: Optional[str] = "Shared Space", 
        shared_space_description: Optional[str] = "Uplink to a shared space"
    ) -> Optional[BaseElement]:
        """
        Ensures an UplinkProxy to the specified SharedSpace exists and is active.
        Delegates to UplinkManagerComponent.
        """
        if self._uplink_manager:
            return self._uplink_manager.ensure_uplink_to_shared_space(shared_space_id, shared_space_name, shared_space_description)
        logger.error(f"[{self.id}] ensure_uplink_to_shared_space: UplinkManagerComponent not available. Cannot ensure uplink to {shared_space_id}.")
        return None

    async def execute_action_on_element(self, element_id: str, action_name: str, parameters: Dict[str, Any], calling_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes an action on an element, ensuring agent's identity is in the calling_context.
        """
        # Prepare the calling_context for actions initiated by this InnerSpace
        final_calling_context = calling_context.copy() if calling_context else {}
        
        # Ensure the agent's own identity is part of the context it sends out
        if 'source_agent_id' not in final_calling_context and hasattr(self, 'agent_id'):
            final_calling_context['source_agent_id'] = self.agent_id
        if 'source_agent_name' not in final_calling_context and hasattr(self, 'agent_name'):
            final_calling_context['source_agent_name'] = self.agent_name
        
        # Add any other context this InnerSpace wants to provide by default
        final_calling_context.setdefault('initiator_space_id', self.id)

        # Call the super method with the augmented context
        return await super().execute_action_on_element(element_id, action_name, parameters, calling_context=final_calling_context)
    
    def register_adapter_mapping(self, adapter_type: str, adapter_name: str, adapter_id: str) -> None:
        """
        Register an adapter mapping discovered from incoming messages.
        
        Args:
            adapter_type: Type of adapter (e.g., 'zulip', 'discord')
            adapter_name: Display name of the adapter/bot
            adapter_id: Unique ID assigned by the platform
        """
        if adapter_type not in self._adapter_mappings:
            self._adapter_mappings[adapter_type] = {}

            
        # Register adapter mapping - if message reached this InnerSpace, it's for this agent
        previous_id = self._adapter_mappings[adapter_type].get(adapter_name)
        if previous_id and previous_id != adapter_id:
            logger.warning(f"[{self.id}] Adapter ID changed for {adapter_type}/{adapter_name}: {previous_id} -> {adapter_id}")
        
        self._adapter_mappings[adapter_type][adapter_name] = adapter_id
        self._agent_adapter_ids.add(adapter_id)
        
        logger.info(f"[{self.id}] Registered adapter mapping: {adapter_type}/{adapter_name} -> {adapter_id}")

    def is_mention_for_agent(self, mentions: List[str]) -> bool:
        """
        Check if any of the mentions are for this agent.
        
        Args:
            mentions: List of adapter_ids that were mentioned
            
        Returns:
            True if this agent was mentioned, False otherwise
        """
        if not mentions:
            return False
            
        # Check if any mentioned adapter_id belongs to this agent
        for mention_id in mentions:
            if mention_id in self._agent_adapter_ids:
                logger.debug(f"[{self.id}] Agent mention detected: adapter_id {mention_id}")
                return True
                
        return False

    def get_agent_adapter_ids(self) -> Set[str]:
        """
        Get all known adapter IDs for this agent.
        
        Returns:
            Set of adapter_ids associated with this agent
        """
        return self._agent_adapter_ids.copy()

    def get_adapter_mappings(self) -> Dict[str, Dict[str, str]]:
        """Get all adapter mappings for this agent."""
        return copy.deepcopy(self._adapter_mappings)