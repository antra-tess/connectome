"""
Inner Space
Special space element using component architecture, representing the agent's subjective experience.
"""

import logging
from typing import Dict, Any, Optional, List, Type, Callable
import uuid

from .space import Space # Inherits ContainerComponent, TimelineComponent, etc.
from .base import BaseElement, MountType
# Import the central factory
from .factory import ElementFactory
# Import LLM Provider for type hint
from ..llm.provider_interface import LLMProvider
from .components import ( # Import necessary components
    ToolProvider, 
    VeilProducer, 
    ElementFactoryComponent,
    GlobalAttentionComponent,
    ContainerComponent, # Needed for type hinting in tool
    ContextManagerComponent, # <-- Add ContextManager
    HUDComponent,             # <-- Add HUD
    TimelineComponent,
    HistoryComponent,
    PublisherComponent,
    CoreToolsComponent, # Added
    MessagingToolsComponent, # New
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llm.provider_interface import LLMProviderInterface
    from host.event_loop import OutgoingActionCallback
    # Import Base from agent_loop, Simple from simple_loop
    from .agent_loop import BaseAgentLoopComponent 
    from .simple_loop import SimpleRequestResponseLoopComponent

class InnerSpace(Space):
    """
    Special space element representing the agent's subjective experience.
    Uses component-based architecture.
    
    Inherits standard Space components (Container, Timeline) and adds:
    - `ElementFactoryComponent`: To create new elements via prefabs.
    - `GlobalAttentionComponent`: To track attention needs (placeholder).
    - `ToolProvider`: For InnerSpace specific tools (like mounting).
    - `VeilProducer`: For representation.
    
    Crucially, this will later host components moved from the Shell (HUD, ContextManager).
    """
    
    # Define event types InnerSpace might specifically handle or coordinate
    EVENT_TYPES = Space.EVENT_TYPES + [
        "attention_requested", # Listens for attention events for the GlobalAttentionComponent
        "attention_cleared"
        # Add others as needed
    ]
    
    def __init__(
        self,
        id: str,
        name: str,
        llm_provider: 'LLMProviderInterface',
        agent_loop_component_type: Type['BaseAgentLoopComponent'] = SimpleRequestResponseLoopComponent,
        components_to_add: Optional[List[Type[Component]]] = None, # Allow adding extra components
        **kwargs # Pass other BaseElement args like description
    ):
        # Call Space init first, which adds ContainerComponent and TimelineComponent
        super().__init__(id=id, name=name, **kwargs)
        
        self._llm_provider = llm_provider
        self._outgoing_action_callback: Optional['OutgoingActionCallback'] = None
        self._agent_loop_component_type = agent_loop_component_type
        
        logger.info(f"Initializing InnerSpace: {name} ({id}) with Agent Loop: {agent_loop_component_type.__name__}")
        
        # --- Define Core Components --- 
        # Components automatically added by BaseElement/Space: ContainerComponent, TimelineComponent
        # Additional core components specific to InnerSpace:
        inner_space_core_component_classes = [
            # HistoryComponent, # Assuming History is needed, needs creation
            ToolProvider, # Manages tool definitions
            # VeilProducer, # For representing InnerSpace state (add when ready)
            # ElementFactoryComponent, # For creating new elements via tools
            GlobalAttentionComponent, # Manages agent focus
            ContextManagerComponent, # Builds context, manages history
            HUDComponent, # Handles LLM interaction formatting
            CoreToolsComponent, # For future internal agent tools
            MessagingToolsComponent, # For external communication tools (New)
            self._agent_loop_component_type # The chosen reasoning loop
        ]

        # Combine with any extra components passed in constructor
        component_classes_to_instantiate = inner_space_core_component_classes
        if components_to_add:
             for comp_cls in components_to_add:
                  if not any(issubclass(comp_cls, existing_cls) or issubclass(existing_cls, comp_cls) 
                             for existing_cls in component_classes_to_instantiate + [ContainerComponent, TimelineComponent]):
                      component_classes_to_instantiate.append(comp_cls)

        # --- Instantiate and Add Components --- 
        available_dependencies = {
            '_llm_provider': self._llm_provider,
        }
        for CompCls in component_classes_to_instantiate:
            comp_kwargs = {}
            requirements = getattr(CompCls, 'INJECTED_DEPENDENCIES', {})
            for kwarg_name, source_attr_name in requirements.items():
                if source_attr_name in available_dependencies:
                    dependency_instance = available_dependencies[source_attr_name]
                    if dependency_instance is not None:
                        comp_kwargs[kwarg_name] = dependency_instance
                        # logger.debug(...) # Keep logging concise for now
                    else:
                        logger.error(f"{CompCls.__name__} requires injected kwarg '{kwarg_name}' but source '{source_attr_name}' is None.")
                else:
                    logger.error(f"{CompCls.__name__} requires injected kwarg '{kwarg_name}' but source '{source_attr_name}' unknown.")
            try:
                self.add_component(CompCls(**comp_kwargs)) 
            except Exception as e:
                 logger.error(f"Error instantiating component {CompCls.__name__} for InnerSpace {self.id}: {e}", exc_info=True)
                 raise RuntimeError(f"Failed to instantiate component {CompCls.__name__}") from e
                 
        # --- Register Tools (After components are added) --- 
        tool_provider_instance = self.get_component(ToolProvider)
        if tool_provider_instance:
            logger.info(f"Registering tools...")
            # Register Core Tools (if any)
            core_tools_instance = self.get_component(CoreToolsComponent)
            if core_tools_instance:
                 core_tools_instance.register_tools(tool_provider_instance)
            else: logger.warning("CoreToolsComponent not found, skipping its tool registration.")
            
            # Register Messaging Tools (New)
            messaging_tools_instance = self.get_component(MessagingToolsComponent)
            if messaging_tools_instance:
                 messaging_tools_instance.register_tools(tool_provider_instance)
            else: logger.warning("MessagingToolsComponent not found, skipping its tool registration.")
                 
        else:
             logger.error("Cannot register tools: ToolProvider component failed to initialize or is missing.")
        # ---------------------------
                 
        logger.info(f"InnerSpace {self.id} component initialization finished.")

    # Replace the entire set_outgoing_action_callback method
    def set_outgoing_action_callback(self, callback: 'OutgoingActionCallback'):
        """Sets the callback function used by components to enqueue outgoing actions."""
        logger.info(f"Setting outgoing action callback for InnerSpace {self.id}")
        self._outgoing_action_callback = callback
        
        # Propagate callback to ALL components that might need it
        components_needing_callback = [
            CoreToolsComponent, # Added
            HUDComponent, # If it needs to enqueue tool actions
            BaseAgentLoopComponent # Find any agent loop component
            # Add other component base classes or specific types here if they need the callback
        ]
        
        for comp_type_or_base in components_needing_callback:
             # Find component by type (handles specific types and base classes)
             component_instance = self.get_component_by_type(comp_type_or_base.COMPONENT_TYPE if hasattr(comp_type_or_base, 'COMPONENT_TYPE') else comp_type_or_base.__name__)
             
             # More robust way to find agent loop by base type if COMPONENT_TYPE isn't standard
             if comp_type_or_base is BaseAgentLoopComponent and not component_instance:
                  for comp in self.get_components().values():
                       if isinstance(comp, BaseAgentLoopComponent):
                            component_instance = comp
                            break 
                            
             if component_instance:
                 if hasattr(component_instance, 'set_outgoing_action_callback'):
                     # Prefer setter method if available
                     try:
                          component_instance.set_outgoing_action_callback(callback)
                          logger.debug(f"Propagated outgoing callback to {component_instance.__class__.__name__} via setter.")
                     except Exception as e:
                          logger.error(f"Error calling set_outgoing_action_callback on {component_instance.__class__.__name__}: {e}")
                 elif hasattr(component_instance, '_outgoing_action_callback'):
                     # Fallback: Set protected attribute directly if setter missing
                     component_instance._outgoing_action_callback = callback
                     logger.debug(f"Propagated outgoing callback to {component_instance.__class__.__name__} via attribute.")
                 else:
                     logger.warning(f"Component {component_instance.__class__.__name__} found, but cannot set outgoing callback (no setter or attribute).")
             else:
                 # Log if a potentially required component wasn't found (might be optional)
                 logger.debug(f"Component type {comp_type_or_base.__name__} not found on InnerSpace {self.id} during callback propagation.")

    # --- Tool Registration --- 
    def _register_inner_space_tools(self) -> None:
        """Register tools specific to inner space using ToolProvider."""
        
        @self._tool_provider.register_tool(
            name="create_and_mount_element",
            description="Creates a new element from a prefab definition and mounts it in this inner space.",
            parameter_descriptions={
                "prefab_name": "Name of the element prefab to create (e.g., 'chat_element')",
                "name": "Optional override name for the new element",
                "description": "Optional override description of the element's purpose",
                "element_id": "Optional specific ID for the new element",
                "mount_id": "Optional identifier for the mount point (defaults to element_id)",
                "mount_type": "Type of mounting ('inclusion' or 'uplink', default 'inclusion')",
                "initial_state": "Dictionary of component config overrides (e.g., {'HistoryComponent': {'max_entries': 50}})"
            }
        )
        def create_and_mount_element(prefab_name: str,
                                    name: Optional[str] = None,
                                    description: Optional[str] = None,
                                    element_id: Optional[str] = None,
                                    mount_id: Optional[str] = None,
                                    mount_type: str = "inclusion",
                                    initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Tool to create and mount a new element using the ElementFactoryComponent and prefab."""
            if not self._element_factory:
                 return {"success": False, "error": "ElementFactoryComponent unavailable"}

            container = self.get_component(ContainerComponent) # Use type-safe getter
            if not container:
                 return {"success": False, "error": "ContainerComponent unavailable"}
                 
            # Validate mount type string
            try:
                mt_enum = MountType(mount_type.lower())
            except ValueError:
                return {"success": False, "error": f"Invalid mount type: '{mount_type}'. Use 'inclusion' or 'uplink'."}
                
            # Use factory component to create the element instance
            new_element = self._element_factory.create_element(
                prefab_name=prefab_name,
                element_id=element_id, # Factory handles ID generation if None
                name=name,             # Pass overrides directly
                description=description,
                initial_state=initial_state # Pass component overrides
            )
            
            if not new_element:
                # Error logged by factory component or central factory
                return {"success": False, "error": f"Failed to create element from prefab '{prefab_name}'"}
                
            # Use container to mount the created element
            actual_element_id = new_element.id
            mount_id = mount_id or actual_element_id
            
            # Check if mount_id or element_id already exists in container before mounting
            existing_mount = container.get_mounted_element(mount_id)
            if existing_mount:
                 return {"success": False, "error": f"Mount ID '{mount_id}' is already in use."}
            # Check if element with same ID exists (should not happen with UUIDs from factory?)
            # for _, el_info in container.get_mounted_elements_info().items():
            #      if el_info['element_id'] == actual_element_id:
            #           return {"success": False, "error": f"Element ID '{actual_element_id}' already exists in container."}

            mount_success = container.mount_element(new_element, mount_id, mt_enum)
            
            if not mount_success:
                # Error logged by container
                return {"success": False, "error": f"Failed to mount created element '{actual_element_id}' at mount point '{mount_id}'"}
                
            # Success!
            return {
                "success": True,
                "element_id": actual_element_id,
                "mount_id": mount_id,
                "prefab_name": prefab_name,
                "mount_type": mount_type
            }

        @self._tool_provider.register_tool(
            name="unmount_element",
            description="Unmount an element currently mounted in this inner space.",
            parameter_descriptions={
                "mount_id": "Identifier of the mount point to unmount"
            }
        )
        def unmount_element_tool(mount_id: str) -> Dict[str, Any]:
            """Tool to unmount an element using the ContainerComponent."""
            container = self.get_component(ContainerComponent)
            if not container:
                 return {"success": False, "error": "ContainerComponent unavailable"}
                 
            # Get info *before* unmounting for the response
            element_info = container.get_mounted_elements_info().get(mount_id)
            if not element_info:
                 return {"success": False, "error": f"No element found at mount point '{mount_id}'"}
                 
            success = container.unmount_element(mount_id)
            
            return {
                 "success": success,
                 "unmounted_mount_id": mount_id,
                 "unmounted_element_id": element_info.get("element_id"),
                 "status": "Element unmounted." if success else "Failed to unmount element."
            }
            
        @self._tool_provider.register_tool(
            name="list_available_prefabs",
            description="List the names of element prefabs that can be created.",
            parameter_descriptions={}
        )
        def list_available_prefabs_tool() -> Dict[str, Any]:
            """Tool to list available element prefabs from the central factory."""
            if not self._element_factory:
                 return {"success": False, "error": "ElementFactoryComponent unavailable", "prefabs": []}
            try:
                # Access the central factory via the component property
                prefab_names = self._element_factory.central_factory.get_available_prefabs()
                return {"success": True, "available_prefabs": prefab_names}
            except Exception as e:
                 logger.error(f"Error getting available prefabs via factory component: {e}", exc_info=True)
                 return {"success": False, "error": "Failed to retrieve prefab list.", "prefabs": []}
            
        @self._tool_provider.register_tool(
            name="get_attention_requests",
            description="Get a list of elements currently requesting attention.",
            parameter_descriptions={}
        )
        def get_attention_requests_tool() -> Dict[str, Any]:
             """Tool to get attention requests from GlobalAttentionComponent."""
             if not self._global_attention:
                  return {"success": False, "error": "GlobalAttentionComponent unavailable", "requests": {}}
             requests = self._global_attention.get_attention_requests()
             formatted_requests = [
                  {
                       "request_key": key, 
                       "space_id": req.get("space_id"), 
                       "source_element_id": req.get("source_element_id"), 
                       "timestamp": req.get("timestamp"),
                       "request_data": req.get("request_data")
                  } 
                  for key, req in requests.items()
             ]
             return {"success": True, "attention_requests": formatted_requests}
             
        # Add other InnerSpace-specific tools if needed (e.g., managing focus, interacting with shell components once added)

    # --- Action Execution Routing --- 
    def execute_element_action(self, 
                               space_id: Optional[str], 
                               element_id: str, 
                               action_name: str, 
                               parameters: Dict[str, Any],
                               timeline_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes an action, routing it to the correct element/space.
        Overrides the base Space implementation if needed, or relies on it.
        
        Args:
            space_id: ID of the target space (None or self.id for InnerSpace).
            element_id: ID of the target element (can be self.id for InnerSpace itself).
            action_name: Name of the action/tool.
            parameters: Parameters for the action.
            timeline_context: Timeline context (important for actions).
            
        Returns:
            Result of the action execution.
        """
        target_space_id = space_id or self.id
        
        # Timeline context handling (ensure primary if not specified)
        if not timeline_context:
             timeline_comp = self.get_component_by_type("timeline")
             primary_timeline = timeline_comp.get_primary_timeline() if timeline_comp else None
             if primary_timeline:
                  timeline_context = {"timeline_id": primary_timeline}
             else:
                  # Cannot execute action without a timeline context
                  return {"error": f"Cannot execute action '{action_name}': Missing timeline context and no primary timeline found."} 
                  
        # --- Action on InnerSpace itself or its directly mounted elements --- 
        if target_space_id == self.id:
             # Use the inherited Space.execute_action_on_element which uses ToolProvider
             # This correctly handles actions targeted at the InnerSpace itself (element_id == self.id)
             # or elements directly mounted within it.
             return super().execute_action_on_element(element_id, action_name, parameters)
             
        # --- Action on an element in a DIFFERENT space (requires registry) --- 
        else:
            registry = self.get_registry()
            if not registry:
                return {"error": f"Cannot execute action in space '{target_space_id}': SpaceRegistry not available."} 
                
            target_space = registry.get_space(target_space_id)
            if not target_space:
                return {"error": f"Target space '{target_space_id}' not found in registry."}
                
            # Ensure the target space has the execution method
            if not hasattr(target_space, 'execute_action_on_element') or not callable(getattr(target_space, 'execute_action_on_element')):
                 return {"error": f"Target space '{target_space_id}' does not support execute_action_on_element."} 
                 
            # Delegate execution to the target space
            # We might need to inject/ensure the timeline_context is passed correctly if needed by the remote execution
            logger.debug(f"Routing action '{action_name}' on element '{element_id}' to remote space '{target_space_id}'")
            # Assuming parameters dict might already contain timeline_context if needed, 
            # or the remote space handles its own timeline based on the call.
            # For safety, let's ensure it's included if the remote method expects it.
            # This introspection is complex; assuming the parameters are passed as is for now.
            try:
                 result = target_space.execute_action_on_element(element_id, action_name, parameters)
                 # TODO: Add timeline recording for the *initiation* of the remote action in InnerSpace timeline?
                 return result
            except Exception as e:
                 logger.error(f"Error executing action on remote space '{target_space_id}': {e}", exc_info=True)
                 return {"error": f"Error during execution on remote space '{target_space_id}': {str(e)}"}
                 
    # --- Convenience Getters --- 
    def get_element_factory(self) -> Optional[ElementFactoryComponent]:
        return self._element_factory
        
    def get_global_attention(self) -> Optional[GlobalAttentionComponent]:
        return self._global_attention

    # --- Event Handling --- 
    # Override handle_event if InnerSpace needs specific coordination, 
    # e.g., routing attention events to GlobalAttentionComponent.
    def handle_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> bool:
        # Allow base class (Space) and its components (Timeline, Container) to handle first
        handled = super().handle_event(event, timeline_context)
        
        # Explicitly route attention events to the global attention manager if not already handled
        event_type = event.get("event_type")
        if event_type in ["attention_requested", "attention_cleared"] and self._global_attention:
             if not handled: # Avoid double handling if Space/BaseElement already routed it
                  handled = self._global_attention.handle_event(event, timeline_context)
             else:
                  # If already handled (e.g. by BaseElement delegation), ensure our component still sees it if needed
                  # This depends on whether handle_event in BaseElement stops propagation
                  # For safety, let's ensure GlobalAttentionComponent always sees these events if present
                  self._global_attention.handle_event(event, timeline_context)
                  
        # Add other InnerSpace specific coordination logic here...
        
        return handled
        
    # --- Obsolete Methods --- 
    # _register_standard_element_types -> Moved to ElementFactoryComponent
    # _get_element_class -> Moved to ElementFactoryComponent
    # _attention_requests handling -> Moved to GlobalAttentionComponent
    
    def get_space(self, space_id: str) -> Optional[Space]:
        """
        Get a space by ID.
        
        Args:
            space_id: ID of the space to retrieve
            
        Returns:
            The space, or None if not found
        """
        # First check spaces we've already loaded
        if space_id in self.spaces:
            return self.spaces[space_id]
            
        # If not found, try to get from registry
        if self._registry:
            space = self._registry.get_space(space_id)
            if space:
                self.spaces[space_id] = space
                return space
                
        return None
    
    def _record_timeline_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Record an event in the timeline.
        
        Args:
            event_type: Type of the event
            event_data: Data for the event
        """
        # Create the event data
        full_event_data = {
            "event_type": event_type,
            "timestamp": int(time.time() * 1000),
            **event_data
        }
        
        # Use primary timeline if available, otherwise create a new one
        timeline_id = self._timeline_state["primary_timeline"]
        if timeline_id is None:
            timeline_id = str(uuid.uuid4())
            self._timeline_state["primary_timeline"] = timeline_id
            self._timeline_state["active_timelines"].add(timeline_id)
            self._timeline_state["events"][timeline_id] = []
            self._timeline_state["timeline_metadata"][timeline_id] = {
                "created_at": int(time.time() * 1000),
                "last_updated": int(time.time() * 1000)
            }
            
        # Update the state
        self.update_state(full_event_data, {"timeline_id": timeline_id})
    
    def _get_element_by_id(self, element_id: str) -> Optional[BaseElement]:
        """
        Get an element by ID.
        
        This method is used by the mount_element tool to find elements by ID.
        If needed, it can be extended to look up elements in a registry
        outside this space.
        
        Args:
            element_id: ID of the element to get
            
        Returns:
            The element if found, None otherwise
        """
        # Check mounted elements
        for mount_info in self._mounted_elements.values():
            element = mount_info["element"]
            if element.id == element_id:
                return element
                
        return None
    
    def receive_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """
        Receive an event from the Activity Layer.
        
        Args:
            event: The event to receive
            timeline_context: The timeline context for this event
        """
        logger.info(f"Receiving event in timeline {timeline_context.get('timeline_id')}: {event.get('type')}")
        
        # Store the event in the timeline DAG
        self.add_event_to_timeline(event, timeline_context)
        
        # Process the event
        try:
            event_type = event.get("type")
            if event_type in self.EVENT_TYPES:
                self._process_event(event, timeline_context)
            
            # Route to appropriate element if specified
            if "targetElement" in event:
                target_element_id = event["targetElement"]
                
                if target_element_id in self.mounted_elements:
                    element = self.mounted_elements[target_element_id]
                    element.receive_event(event, timeline_context)
                else:
                    logger.warning(f"Event targeted non-existent element: {target_element_id}")
        except Exception as e:
            logger.error(f"Error processing event: {e}", exc_info=True)
    
    def add_event_to_timeline(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> str:
        """
        Add an event to the timeline DAG.
        
        Args:
            event: The event to add
            timeline_context: The timeline context for this event
            
        Returns:
            ID of the newly created event
        """
        logger.info(f"Adding event to timeline {timeline_context.get('timeline_id')}: {event.get('type')}")
        
        # Create event object with metadata
        event_id = event.get("id", str(uuid.uuid4()))
        event["id"] = event_id
        event["timestamp"] = event.get("timestamp", int(time.time() * 1000))
        
        # Get branch ID from context
        branch_id = timeline_context.get("timeline_id")
        if not branch_id:
            logger.warning("No branch ID in timeline context, using primary branch")
            branch_id = "primary-branch-001"
        
        # Link to parent event if available
        parent_event_id = timeline_context.get("last_event_id")
        if parent_event_id:
            event["parent_id"] = parent_event_id
        
        # Store in timeline state
        if branch_id not in self._timeline_state["events"]:
            self._timeline_state["events"][branch_id] = {}
            
        self._timeline_state["events"][branch_id][event_id] = event
        
        # Update timeline context with this event as the new last event
        timeline_context["last_event_id"] = event["id"]
        
        # Notify any observers
        self._notify_observers("timeline_updated", {
            "element_id": self.id,
            "event_id": event_id,
            "event_type": event.get("type"),
            "timeline_id": branch_id
        })
        
        return event["id"]
    
    def send_message(self, message: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """
        Send a message from the agent to external systems.
        
        This passes through the Inner Space, to the Activity Layer, and then to external adapters.
        
        Args:
            message: The message to send
            timeline_context: The timeline context for this message
        """
        logger.info(f"Sending message in timeline {timeline_context.get('timeline_id')}: {message.get('type')}")
        
        # Add the message to the timeline as an event
        message_id = self.add_event_to_timeline(message, timeline_context)
        
        # Only propagate externally if in primary timeline
        if timeline_context.get("is_primary", False):
            # In a real implementation, this would call Activity Layer to propagate externally
            logger.info(f"Propagating message externally: {message_id}")
            
            # If we have an associated registry, propagate through it
            if self._registry:
                self._registry.propagate_message(message, timeline_context)
        else:
            logger.info(f"Message not propagated externally (non-primary timeline): {message_id}")
    
    def _process_event(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """
        Process an event based on its type.
        
        Args:
            event: The event to process
            timeline_context: The timeline context for this event
        """
        event_type = event.get("type")
        
        if event_type == "user_message":
            # Process user message
            self._handle_user_message(event, timeline_context)
        elif event_type == "system_message":
            # Process system message
            self._handle_system_message(event, timeline_context)
        elif event_type == "tool_response":
            # Process tool response
            self._handle_tool_response(event, timeline_context)
        elif event_type == "agent_message":
            # Process agent message
            self._handle_agent_message(event, timeline_context)
        elif event_type == "state_update":
            # Process state update
            self._handle_state_update(event, timeline_context)
        else:
            logger.warning(f"Unknown event type: {event_type}")
            
    def _handle_user_message(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """Handle a user message event"""
        # In a complete implementation, this would process the user message
        # and potentially update the state of the inner space
        logger.info(f"Handling user message: {event.get('content', '')[:30]}...")
        
    def _handle_system_message(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """Handle a system message event"""
        # Process system message
        logger.info(f"Handling system message: {event.get('content', '')[:30]}...")
        
    def _handle_tool_response(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """Handle a tool response event"""
        # Process tool response
        logger.info(f"Handling tool response: {event.get('toolName', '')} - {event.get('status', '')}")
        
    def _handle_agent_message(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """Handle an agent message event"""
        # Process agent message
        logger.info(f"Handling agent message: {event.get('content', '')[:30]}...")
        
    def _handle_state_update(self, event: Dict[str, Any], timeline_context: Dict[str, Any]) -> None:
        """Handle a state update event"""
        # Process state update
        logger.info(f"Handling state update: {event.get('stateType', '')}")
    
    def mount_element(self, element: BaseElement, mount_type: MountType = MountType.CHILD) -> None:
        """
        Mount an element to this space.
        
        Args:
            element: The element to mount
            mount_type: The type of mount
        """
        logger.info(f"Mounting element {element.element_id} to inner space with mount type {mount_type}")
        
        # Store the element
        self.mounted_elements[element.element_id] = element
        
        # Update the element's parent reference
        element.parent = self 

    def _on_element_state_changed(self, element_id: str, state_data: Dict[str, Any]) -> None:
        """
        Handle element state change notification.
        
        Args:
            element_id: ID of the element that changed
            state_data: Data about the state change
        """
        logger.debug(f"Element state changed: {element_id}")
        
        # Check for attention request
        event_type = state_data.get("type")
        if event_type == "attention_requested":
            self._handle_attention_request(element_id, state_data)
        elif event_type == "attention_cleared":
            self._handle_attention_cleared(element_id)
    
    def _handle_attention_request(self, element_id: str, request_data: Dict[str, Any]) -> None:
        """
        Handle an attention request from a Space or Element.
        
        Args:
            element_id: ID of the Space or Element requesting attention
            request_data: Data about the attention request
        """
        # Add or update the attention request
        self._attention_requests[element_id] = {
            "timestamp": request_data.get("timestamp", int(time.time() * 1000)),
            "data": request_data,
            "source_element_id": request_data.get("source_element_id", element_id)
        }
        
        logger.info(f"Element/Space {element_id} attention request registered in InnerSpace")
        
        # Notify registry that a component needs attention
        if self._registry:
            self._registry._notify_observers("inner_space_attention_requested", {
                "inner_space_id": self.id,
                "element_id": element_id,
                "source_element_id": request_data.get("source_element_id", element_id),
                "request_data": request_data
            })
    
    def _handle_attention_cleared(self, element_id: str) -> None:
        """
        Handle attention cleared notification from a Space or Element.
        
        Args:
            element_id: ID of the Space or Element clearing attention
        """
        # Remove the attention request if it exists
        if element_id in self._attention_requests:
            source_element_id = self._attention_requests[element_id].get("source_element_id", element_id)
            del self._attention_requests[element_id]
            
            # Notify registry that this element no longer needs attention
            if self._registry:
                self._registry._notify_observers("inner_space_attention_cleared", {
                    "inner_space_id": self.id,
                    "element_id": element_id,
                    "source_element_id": source_element_id
                })
                
            logger.info(f"Element/Space {element_id} attention cleared in InnerSpace")
    
    def get_elements_requesting_attention(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all elements currently requesting attention in the InnerSpace.
        
        Returns:
            Dictionary mapping element IDs to their attention request data
        """
        return self._attention_requests.copy()
    
    def handle_element_observer_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle events from observed elements.
        
        Args:
            event_data: Data about the event
        """
        element_id = event_data.get("element_id")
        if not element_id:
            logger.warning("Received element event without element_id")
            return
            
        event_type = event_data.get("type")
        
        # Handle attention related events
        if event_type == "attention_requested":
            # This is coming from a Space that has already evaluated the need
            self._handle_attention_request(element_id, event_data)
        elif event_type == "attention_cleared":
            self._handle_attention_cleared(element_id)
        elif event_type == "element_state_changed":
            if event_data.get("state_change") == "attention_needed":
                # Handle direct attention requests from elements mounted in InnerSpace
                # For elements in sub-spaces, the Space should handle this
                if element_id in self._mounted_elements:
                    self._handle_attention_request(element_id, event_data)
            else:
                logger.debug(f"Element state changed: {element_id} - {event_data.get('state_change')}")
        else:
            logger.debug(f"Received element event: {event_type} from {element_id}") 

    # --- Add Convenience Getters for Shell Components --- 
    def get_context_manager(self) -> Optional[ContextManagerComponent]:
         """Returns the context manager component instance."""
         return self.get_component(ContextManagerComponent) # Use base class getter
         
    def get_hud(self) -> Optional[HUDComponent]:
         """Returns the HUD component instance."""
         return self.get_component(HUDComponent) # Use base class getter 

    def get_agent_loop_component(self) -> Optional['BaseAgentLoopComponent']:
         """Convenience method to get the active AgentLoop component."""
         for comp in self.get_components().values():
              if isinstance(comp, BaseAgentLoopComponent):
                   return comp
         return None

    def get_hud(self) -> Optional[HUDComponent]:
        """Convenience method to get the HUD component."""
        return self.get_component(HUDComponent)

    # Override receive_event if InnerSpace needs specific handling before components
    # def receive_event(self, event_data: Dict[str, Any], timeline_context: Dict[str, Any]):
    #     logger.debug(f"InnerSpace {self.id} received event: {event_data.get('event_type')}")
    #     # Potentially do InnerSpace-specific logic here
    #     super().receive_event(event_data, timeline_context)
    #     # Potentially do InnerSpace-specific logic after component processing 