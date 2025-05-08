"""
Agent Loop Components
Defines the base class and simple implementations for agent cognitive cycles.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, TYPE_CHECKING

from .base import Component

if TYPE_CHECKING:
    from .inner_space import InnerSpace
    from ..llm.provider_interface import LLMProviderInterface
    from .components.hud import HUDComponent # Assuming HUDComponent is in .components.hud
    from .components.tool_provider import ToolProviderComponent # Assuming ToolProviderComponent is in .components.tool_provider
    from host.event_loop import OutgoingActionCallback


logger = logging.getLogger(__name__)

class BaseAgentLoopComponent(Component):
    """
    Abstract base class for agent loop components.
    Defines the interface for how the HostEventLoop triggers an agent's cognitive cycle.
    """
    COMPONENT_TYPE = "AgentLoopComponent"
    # Define dependencies that InnerSpace should inject during instantiation.
    # Key: kwarg name for __init__; Value: attribute name on InnerSpace instance or 'self' for InnerSpace itself.
    INJECTED_DEPENDENCIES = {
        'parent_inner_space': 'self' 
    }

    def __init__(self, element_id: str, name: str, parent_inner_space: 'InnerSpace', **kwargs):
        super().__init__(element_id, name, **kwargs)
        if not parent_inner_space: # Should be guaranteed by InnerSpace's injection
            raise ValueError("BaseAgentLoopComponent requires a parent_inner_space instance.")
        self.parent_inner_space: 'InnerSpace' = parent_inner_space
        
        # Convenience accessors, assuming parent_inner_space is correctly typed and populated
        self._llm_provider: Optional['LLMProviderInterface'] = self.parent_inner_space._llm_provider
        self._hud_component: Optional['HUDComponent'] = self.parent_inner_space.get_hud()
        self._tool_provider: Optional['ToolProviderComponent'] = self.parent_inner_space.get_component(ToolProviderComponent)
        self._outgoing_action_callback: Optional['OutgoingActionCallback'] = self.parent_inner_space._outgoing_action_callback

        if not self._llm_provider:
            logger.error(f"{self.name} ({self.id}): LLMProvider not available from parent InnerSpace.")
        if not self._hud_component:
            logger.error(f"{self.name} ({self.id}): HUDComponent not available from parent InnerSpace.")
        # ToolProvider and outgoing_action_callback might be optional for some loops.

        logger.info(f"{self.COMPONENT_TYPE} '{self.name}' ({self.id}) initialized for InnerSpace '{self.parent_inner_space.name}'.")

    async def trigger_cycle(self):
        """
        This method is called by the HostEventLoop to initiate one cognitive cycle
        for the agent. Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement the trigger_cycle method.")

    def _get_hud(self) -> Optional['HUDComponent']:
        if not self._hud_component:
            self._hud_component = self.parent_inner_space.get_hud()
            if not self._hud_component:
                 logger.error(f"{self.name} ({self.id}): HUDComponent could not be retrieved on demand.")
        return self._hud_component

    def _get_llm_provider(self) -> Optional['LLMProviderInterface']:
        if not self._llm_provider: # Should have been set in init
             logger.error(f"{self.name} ({self.id}): LLMProvider not available.")
        return self._llm_provider

    def _get_tool_provider(self) -> Optional['ToolProviderComponent']:
        if not self._tool_provider:
            self._tool_provider = self.parent_inner_space.get_component(ToolProviderComponent)
        return self._tool_provider
        
    def _get_outgoing_action_callback(self) -> Optional['OutgoingActionCallback']:
        if not self._outgoing_action_callback: # Should have been set in init via parent
            self._outgoing_action_callback = self.parent_inner_space._outgoing_action_callback
        return self._outgoing_action_callback


class SimpleRequestResponseLoopComponent(BaseAgentLoopComponent):
    """
    A basic agent loop that:
    1. Gets context from HUD.
    2. Sends context to LLM.
    3. Processes LLM response via HUD (which may extract actions).
    4. Dispatches actions.
    """
    COMPONENT_TYPE = "SimpleRequestResponseLoopComponent"

    def __init__(self, element_id: str, name: str, parent_inner_space: 'InnerSpace', **kwargs):
        super().__init__(element_id, name, parent_inner_space, **kwargs)
        # Additional initialization specific to this loop type, if any.

    async def trigger_cycle(self):
        logger.info(f"{self.name} ({self.id}): Cycle triggered in InnerSpace '{self.parent_inner_space.name}'.")

        hud = self._get_hud()
        llm_provider = self._get_llm_provider()

        if not hud:
            logger.error(f"{self.name} ({self.id}): HUDComponent not available. Aborting cycle.")
            return
        if not llm_provider:
            logger.error(f"{self.name} ({self.id}): LLMProvider not available. Aborting cycle.")
            return

        try:
            # 1. Prepare context using HUD
            # Assuming HUDComponent has a method like `prepare_llm_context`
            # This method would look at InnerSpace's DAG, use VeilProducers, CompressionEngine etc.
            # For now, let's assume it returns a string or a structured dict for the LLM.
            logger.debug(f"{self.name} ({self.id}): Preparing LLM context via HUD...")
            llm_context = await hud.prepare_llm_context() # This method needs to be async
            if llm_context is None:
                logger.warning(f"{self.name} ({self.id}): HUD prepared a null context. Aborting cycle.")
                return

            # 2. Send context to LLM
            logger.debug(f"{self.name} ({self.id}): Sending context to LLM...")
            # Assuming LLMProvider has a method like `generate_text_response` or similar
            llm_response_text = await llm_provider.generate_text_response(llm_context) # This needs to be async
            if not llm_response_text:
                logger.warning(f"{self.name} ({self.id}): LLM returned no response. Aborting cycle.")
                return
            
            logger.debug(f"{self.name} ({self.id}): Received LLM response.")

            # 3. Process LLM response via HUD
            # Assuming HUDComponent has a method like `process_llm_response`
            # This method would parse the LLM's output (potentially VEIL-structured),
            # extract content to be saved, and identify actions to be taken.
            # It might return a list of actions or directly call handlers.
            # For this example, let's assume it returns a list of parsed action dictionaries.
            logger.debug(f"{self.name} ({self.id}): Processing LLM response via HUD...")
            # This method on HUD also needs to be async if it involves I/O or complex parsing
            processed_actions = await hud.process_llm_response(llm_response_text) 

            # 4. Dispatch actions
            if processed_actions and isinstance(processed_actions, list):
                logger.info(f"{self.name} ({self.id}): Dispatching {len(processed_actions)} actions.")
                for action_request in processed_actions:
                    if not isinstance(action_request, dict):
                        logger.warning(f"Invalid action format from HUD: {action_request}. Skipping.")
                        continue
                    
                    target_module = action_request.get("target_module")
                    
                    if target_module: # Assumed to be for external systems like ActivityClient
                        callback = self._get_outgoing_action_callback()
                        if callback:
                            logger.debug(f"Dispatching external action via callback: {action_request}")
                            callback(action_request) # Enqueues to HostEventLoop's outgoing queue
                        else:
                            logger.error(f"Cannot dispatch external action: No outgoing_action_callback set. Action: {action_request}")
                    elif action_request.get("target_element_id"): # Action for an element within a Space
                        tool_provider = self._get_tool_provider()
                        if tool_provider:
                            # TODO: This part needs refinement.
                            # The action_request should match what ToolProviderComponent.execute_tool expects.
                            # Or, InnerSpace.execute_element_action should be called.
                            # For now, a placeholder:
                            action_name = action_request.get("action_name")
                            action_params = action_request.get("payload", {}).get("parameters", {}) # common pattern
                            element_id_target = action_request.get("target_element_id")
                            space_id_target = action_request.get("target_space_id", self.parent_inner_space.id) # Default to own InnerSpace

                            logger.debug(f"Dispatching internal action: TargetSpace='{space_id_target}', TargetElement='{element_id_target}', Action='{action_name}'")
                            # Using parent InnerSpace's action execution method
                            # This allows actions to target elements in this InnerSpace or other Spaces via registry.
                            # Timeline context for actions also needs to be considered.
                            # For now, let's assume a default timeline context is handled by execute_element_action.
                            action_result = self.parent_inner_space.execute_element_action(
                                space_id=space_id_target,
                                element_id=element_id_target,
                                action_name=action_name,
                                parameters=action_params
                                # timeline_context might be needed here
                            )
                            logger.debug(f"Internal action result: {action_result}")
                            # TODO: Optionally record action result in DAG or handle errors
                        else:
                            logger.error(f"Cannot dispatch internal action: ToolProviderComponent not available. Action: {action_request}")
                    else:
                        logger.warning(f"Could not dispatch action: No target_module or target_element_id specified. Action: {action_request}")
            else:
                logger.info(f"{self.name} ({self.id}): No actions to dispatch from LLM response.")

        except Exception as e:
            logger.error(f"{self.name} ({self.id}): Error during cognitive cycle: {e}", exc_info=True)
        finally:
            logger.info(f"{self.name} ({self.id}): Cycle finished.") 