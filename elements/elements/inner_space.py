"""
Inner Space
Special space element representing the agent's subjective experience.
Uses the component architecture pattern consistently.
"""

import logging
from typing import Dict, Any, Optional, List, Type, Callable
import inspect
import time
import re # For _generate_safe_id_string

from .space import Space  # Inherit from Space to get Container and Timeline functionality
from .base import BaseElement, MountType

# Import core components needed by InnerSpace
from .components.tool_provider import ToolProviderComponent 
from .components.factory_component import ElementFactoryComponent
from .components.base_component import VeilProducer, Component
from .components.hud.hud_component import HUDComponent # Explicitly import if kept
from .components.dm_manager_component import DirectMessageManagerComponent # NEWLY ADDED
from .components.uplink_manager_component import UplinkManagerComponent # NEWLY ADDED

# Import the specific Space Veil Producer
from .components.space.space_veil_producer import SpaceVeilProducer

# Import agent loop components
from .agent_loop import BaseAgentLoopComponent, MultiStepToolLoopComponent, SimpleRequestResponseLoopComponent

# Type checking imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from llm.provider_interface import LLMProviderInterface
    from host.event_loop import OutgoingActionCallback
    from elements.space_registry import SpaceRegistry

# Re-define type hint for callback if not easily importable from event_loop
MarkAgentForCycleCallable = Callable[[str, Dict[str, Any], float], None]

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
        description: str,
        agent_id: str,
        llm_provider: 'LLMProviderInterface',
        agent_loop_component_type: Type[BaseAgentLoopComponent] = MultiStepToolLoopComponent,
        system_prompt_template: Optional[str] = None,
        outgoing_action_callback: Optional['OutgoingActionCallback'] = None,
        space_registry: Optional['SpaceRegistry'] = None,
        mark_agent_for_cycle_callback: Optional[MarkAgentForCycleCallable] = None,
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
            system_prompt_template: Optional system prompt template for the agent
            outgoing_action_callback: Callback function for sending actions to external systems
            space_registry: Reference to the SpaceRegistry instance
            mark_agent_for_cycle_callback: Callback to HostEventLoop.mark_agent_for_cycle.
            additional_components: Optional list of additional component types to add
            **kwargs: Additional keyword arguments for Space initialization
        """
        # Initialize the base Space (adds ContainerComponent and TimelineComponent)
        super().__init__(element_id=element_id, name=name, description=description, **kwargs)
        
        # Store key dependencies
        self.agent_id = agent_id
        self.agent_name = agent_name
        self._llm_provider = llm_provider
        self._outgoing_action_callback = outgoing_action_callback
        self._space_registry = space_registry
        self._mark_agent_for_cycle = mark_agent_for_cycle_callback
        self._system_prompt_template = system_prompt_template
        
        # Initialize component references (will be populated during initialization)
        self._tool_provider = None
        self._element_factory = None
        self._dm_manager = None # NEWLY ADDED
        self._uplink_manager = None # NEWLY ADDED
        # self._global_attention = None # REMOVED
        self._hud = None
        # self._context_manager = None # REMOVED
        self._agent_loop = None
        
        # --- Add Core InnerSpace Components ---
        logger.info(f"Adding core components to InnerSpace: {name} ({element_id}) for agent {agent_id}")
        
        # Add tool provider for registering and executing tools
        self._tool_provider = self.add_component(ToolProviderComponent)
        if not self._tool_provider:
            logger.error(f"Failed to add ToolProviderComponent to InnerSpace {self.id}")
            
        # Add element factory for creating new elements
        self._element_factory = self.add_component(ElementFactoryComponent)
        if not self._element_factory:
            logger.error(f"Failed to add ElementFactoryComponent to InnerSpace {self.id}")
        else:
            # --- Automatically create a default scratchpad for the agent ---
            scratchpad_id = f"scratchpad_{self.agent_id}"
            logger.info(f"[{self.id}] Attempting to create default scratchpad element '{scratchpad_id}'.")
            scratchpad_config = {
                "name": "Agent Scratchpad",
                "description": f"Default scratchpad for agent {self.agent_id}"
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
            # ------------------------------------------------------------------
            
        # Add DirectMessageManagerComponent NEWLY ADDED
        self._dm_manager = self.add_component(DirectMessageManagerComponent)
        if not self._dm_manager:
            logger.error(f"Failed to add DirectMessageManagerComponent to InnerSpace {self.id}")

        # NEW: Add UplinkManagerComponent
        self._uplink_manager = self.add_component(UplinkManagerComponent)
        if not self._uplink_manager:
            logger.error(f"Failed to add UplinkManagerComponent to InnerSpace {self.id}")

        # Add HUD component for rendering agent context
        hud_kwargs = {'llm_provider': llm_provider} if llm_provider else {}
        self._hud = self.add_component(HUDComponent, **hud_kwargs)
        if not self._hud:
            logger.error(f"Failed to add HUDComponent to InnerSpace {self.id}")
            
        # Add VEIL producer for representing InnerSpace - Use SpaceVeilProducer
        self.add_component(SpaceVeilProducer)
        
        # Add any additional requested components
        if additional_components:
            for component_type in additional_components:
                if component_type in [ToolProviderComponent, ElementFactoryComponent, 
                                     DirectMessageManagerComponent, # NEWLY ADDED
                                     UplinkManagerComponent, # NEWLY ADDED
                                     # GlobalAttentionComponent, # REMOVED
                                     HUDComponent 
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
            "system_prompt_template": self._system_prompt_template,
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
            
        if not self._space_registry:
            logger.warning(f"InnerSpace {self.id} initialized without SpaceRegistry. Cross-space actions will not be possible.")
            
        # Check if mark_agent_for_cycle callback was provided
        if not self._mark_agent_for_cycle:
             logger.warning(f"InnerSpace {self.id} for agent {self.agent_id} initialized without mark_agent_for_cycle callback. Tool results will not trigger agent cycles via HostEventLoop.")
            
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
        self._propagate_callback_to_component(DirectMessageManagerComponent, callback) # NEWLY ADDED
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
    def execute_element_action(
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

        try:
            # 1. Find Target Space
            if space_id == self.id:
                target_space = self
            elif self._space_registry:
                target_space = self._space_registry.get_space(space_id)
            
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
            # ToolProviderComponent.execute_tool is assumed to return a dict like action_result
            action_result = tool_provider.execute_tool(action_name, parameters)
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
        # Attempt to retrieve if not already set (might have been added later)
        if not self._agent_loop:
             self._agent_loop = self.get_component_by_type(BaseAgentLoopComponent)
        return self._agent_loop
        
    def get_tool_provider(self) -> Optional[ToolProviderComponent]:
        """Get the tool provider component."""
        return self._tool_provider
        
    def get_element_factory(self) -> Optional[ElementFactoryComponent]:
        """Get the element factory component."""
        return self._element_factory
        
    def get_dm_manager(self) -> Optional[DirectMessageManagerComponent]: # NEWLY ADDED
        """Get the DirectMessageManager component.""" # NEWLY ADDED
        return self._dm_manager # NEWLY ADDED
        
    def get_uplink_manager(self) -> Optional[UplinkManagerComponent]: # NEWLY ADDED
        """Get the UplinkManagerComponent.""" # NEWLY ADDED
        return self._uplink_manager # NEWLY ADDED
        
    def get_hud(self) -> Optional[HUDComponent]:
        """Get the HUD component."""
        return self._hud
    
    # --- Event Handling ---
    def handle_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        """
        Handle InnerSpace-specific events, including triggering agent cycles on tool results
        via the HostEventLoop.
        
        Args:
            event: The event to handle (expected structure: {"event_id": ..., "timestamp": ..., "payload": {...}})
            timeline_context: Context information about the timeline
            
        Returns:
            True if the event was handled, False otherwise
        """
        # Let Space handle the event first (records to timeline, updates components, etc.)
        handled = super().handle_event(event, timeline_context)
        
        # Extract event payload robustly (this is the inner payload, e.g., what was passed to add_event_...)
        event_payload = event.get('payload', {})
        event_type = event_payload.get("event_type")
        event_data = event_payload.get("data", {})

        # --- Trigger Agent Cycle via HostEventLoop on Tool Result ---
        if event_type == "tool_result_received":
            # The loop_component_id is still useful for logging/debugging
            loop_component_id = event_data.get("loop_component_id")
            logger.debug(f"[{self.id}] Detected tool result event. Attempting to mark agent {self.agent_id} for cycle via callback.")
            
            if callable(self._mark_agent_for_cycle):
                try:
                    current_time = time.monotonic()
                    # Pass the *entire event payload* (which contains event_type: "tool_result_received")
                    # and the agent_id to the callback.
                    self._mark_agent_for_cycle(self.agent_id, event_payload, current_time)
                    logger.info(f"[{self.id}] Called mark_agent_for_cycle for agent {self.agent_id} due to tool result (loop: {loop_component_id}).")
                    # Mark handled because we successfully delegated the trigger action
                    handled = True 
                except Exception as e:
                    logger.error(f"Error calling mark_agent_for_cycle callback for agent {self.agent_id}: {e}", exc_info=True)
            else:
                logger.warning(f"[{self.id}] Tool result received, but mark_agent_for_cycle callback is not available or not callable. Cannot trigger agent cycle.")
        # --- End Trigger Agent Loop Logic ---

        # Direct attention events to the global attention component if not already handled
        elif event_type in ["attention_requested", "attention_cleared"]: # MODIFIED condition
            if not handled:
                logger.warning(f"InnerSpace received attention event '{event_type}' but GlobalAttentionComponent is not available.")
                # Event is not "handled" by a specific component here, but we acknowledge it.
                # Set handled to True if we consider logging it as handling.
                # For now, let it fall through, HostEventLoop might still see it if it bubbles up.
                pass # Or handled = True if logging is enough.
                
        # Add any other InnerSpace-specific event handling here
                
        return handled

    # --- Uplink Management ---
    def get_connected_spaces(self) -> List[str]:
        """
        Get IDs of all spaces this InnerSpace is connected to via UplinkProxy elements.
        Delegates to UplinkManagerComponent if available.
        """
        if self._uplink_manager:
            # Assuming UplinkManagerComponent's list_active_uplinks_tool returns a dict
            # with an 'active_uplinks' list of dicts, each having 'remote_space_id'
            tool_result = self._uplink_manager.list_active_uplinks_tool()
            if tool_result.get("success"):
                return [uplink.get("remote_space_id") for uplink in tool_result.get("active_uplinks", []) if uplink.get("remote_space_id")]
            else:
                logger.error(f"[{self.id}] get_connected_spaces: Error from UplinkManager: {tool_result.get('error')}")
                return [] # Return empty on error to avoid downstream issues
        
        # Fallback (original logic, less efficient if UplinkManager is present but fails to provide)
        logger.warning(f"[{self.id}] get_connected_spaces: UplinkManagerComponent not available or failed. Using fallback scan.")
        connected_spaces = []
        for element in self.get_mounted_elements().values():
            # Check if this is an UplinkProxy (through duck typing to avoid import cycles)
            if hasattr(element, 'remote_space_id') and callable(getattr(element, 'get_connection_component', None)):
                connected_spaces.append(element.remote_space_id)
        return connected_spaces
    
    def get_uplink_for_space(self, remote_space_id: str) -> Optional[BaseElement]:
        """
        Get the UplinkProxy element for a specific remote space.
        Delegates to UplinkManagerComponent.
        """
        if self._uplink_manager:
            return self._uplink_manager.get_uplink_for_space(remote_space_id)
        
        logger.warning(f"[{self.id}] get_uplink_for_space: UplinkManagerComponent not available. Cannot get uplink for {remote_space_id}.")
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
            return self._uplink_manager.ensure_uplink_to_shared_space(
                shared_space_id=shared_space_id,
                shared_space_name=shared_space_name,
                shared_space_description=shared_space_description
            )
        
        logger.error(f"[{self.id}] ensure_uplink_to_shared_space: UplinkManagerComponent not available. Cannot ensure uplink to {shared_space_id}.")
        return None