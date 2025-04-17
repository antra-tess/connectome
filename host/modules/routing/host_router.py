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
            context_key: The key in the event context to check (e.g., 'adapter_id', 'user_id').
            context_value: The specific value of the context key that maps to this agent.
        """
        if not isinstance(context_key, str):
             logger.error(f"Cannot register route: context_key must be a string, got {type(context_key)}")
             return
             
        if context_key not in self._routing_map:
            self._routing_map[context_key] = {}
            
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
        Determines the target agent ID based on the provided context (usually event payload).
        Prioritizes lookup using 'adapter_id' if present.
        
        Args:
            context: The context dictionary, typically the event payload which should 
                     contain keys like 'adapter_id', 'conversation_id', etc.
                     
        Returns:
            The matched agent ID, or None if no route is found.
        """
        if not isinstance(context, dict):
            logger.warning("Cannot route: Invalid context provided (not a dict).")
            return None
            
        # --- Routing Logic --- 
        # 1. Prioritize routing by adapter_id
        adapter_id = context.get("adapter_id")
        if adapter_id:
            adapter_routing_key = "adapter_id"
            if adapter_routing_key in self._routing_map:
                target_agent = self._routing_map[adapter_routing_key].get(adapter_id)
                if target_agent:
                    logger.debug(f"Routing context via ['{adapter_routing_key}': '{adapter_id}'] -> Agent '{target_agent}'")
                    return target_agent
            # If adapter_id was present but not found in map, log and continue to check other keys
            logger.debug(f"Context contains adapter_id '{adapter_id}', but no route found for it.")

        # 2. Fallback: Check other registered keys (e.g., user_id, timeline_id if applicable)
        # Iterate through known context keys *other than* adapter_id if we already checked it
        keys_to_check = [k for k in self._routing_map.keys() if k != "adapter_id"]
        
        for key in keys_to_check:
            if key in context:
                value = context[key]
                value_map = self._routing_map[key]
                if value in value_map:
                    target_agent = value_map[value]
                    logger.debug(f"Routing context via fallback ['{key}': '{value}'] -> Agent '{target_agent}'")
                    return target_agent
        # ---------------------
        
        logger.debug(f"No specific route found for context: {context}")
        return None # No route found

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