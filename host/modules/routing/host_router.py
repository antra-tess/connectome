"""
Host Router Module

Responsible for mapping incoming event context (e.g., conversation ID)
to the target agent ID.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class HostRouter:
    """
    Maps contextual information from incoming events to agent IDs.
    
    This allows the HostEventLoop to know which ShellModule should handle
    an event without needing direct knowledge of Spaces or Elements.
    """
    
    def __init__(self):
        # Simple routing map: context_key -> context_value -> agent_id
        # Example: {'conversation_id': {'conv123': 'agent_001'}}
        self._routing_map: Dict[str, Dict[Any, str]] = {}
        logger.info("HostRouter initialized.")
        
    def register_agent_route(
        self, 
        agent_id: str, 
        context_key: str, 
        context_value: Any
    ) -> None:
        """
        Registers a specific context value to map to an agent.
        
        Args:
            agent_id: The ID of the agent/shell to route to.
            context_key: The key in the event context to check (e.g., 'conversation_id').
            context_value: The specific value of the context key that maps to this agent.
        """
        if not isinstance(context_key, str):
             logger.error(f"Cannot register route: context_key must be a string, got {type(context_key)}")
             return
             
        # Ensure the inner dictionary exists for the key
        if context_key not in self._routing_map:
            self._routing_map[context_key] = {}
            
        # Check for conflicts
        existing_agent = self._routing_map[context_key].get(context_value)
        if existing_agent and existing_agent != agent_id:
            logger.warning(
                f"Routing conflict: Context ['{context_key}': '{context_value}'] already mapped to agent '{existing_agent}'. "
                f"Overwriting with '{agent_id}'."
            )
            
        self._routing_map[context_key][context_value] = agent_id
        logger.info(f"Registered route: ['{context_key}': '{context_value}'] -> Agent '{agent_id}'")

    def get_target_agent_id(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Determines the target agent ID based on the provided context.
        
        Args:
            context: The context dictionary associated with an incoming event 
                     (should contain keys like 'conversation_id', 'adapter_id', etc.).
                     
        Returns:
            The matched agent ID, or None if no route is found.
        """
        if not isinstance(context, dict):
            logger.warning("Cannot route: Invalid context provided (not a dict).")
            return None
            
        # --- Routing Logic --- 
        # Iterate through known context keys in our map
        for key, value_map in self._routing_map.items():
            if key in context:
                value = context[key]
                if value in value_map:
                    target_agent = value_map[value]
                    logger.debug(f"Routing context via ['{key}': '{value}'] -> Agent '{target_agent}'")
                    return target_agent
                    
        # TODO: Implement more sophisticated routing if needed (e.g., wildcards, default routes)
        # ---------------------
        
        logger.debug(f"No specific route found for context: {context}")
        return None # No specific route found

    # Optional: Method to remove routes
    def unregister_agent_route(self, context_key: str, context_value: Any) -> bool:
        """
        Removes a specific routing rule.
        """
        if context_key in self._routing_map and context_value in self._routing_map[context_key]:
            removed_agent_id = self._routing_map[context_key].pop(context_value)
            logger.info(f"Unregistered route: ['{context_key}': '{context_value}'] (was -> Agent '{removed_agent_id}')")
            # Clean up empty inner dict if necessary
            if not self._routing_map[context_key]:
                del self._routing_map[context_key]
            return True
        logger.warning(f"Could not unregister route: ['{context_key}': '{context_value}'] not found.")
        return False 