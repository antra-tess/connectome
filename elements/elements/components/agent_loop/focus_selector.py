"""
Focus Selection Strategies
Provides pluggable focus selection strategies for activation decisions.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time

from elements.utils.element_id_generator import ElementIdGenerator

logger = logging.getLogger(__name__)


class FocusStrategy(ABC):
    @abstractmethod
    def select_focus(self, candidates: List[Dict[str, Any]], 
                     context: Dict[str, Any]) -> Optional[str]:
        pass


class ConversationFocusStrategy(FocusStrategy):
    """Focus on conversation that triggered event."""
    def select_focus(self, candidates, context):
        triggering_conv = context.get('triggering_conversation_id')
        if triggering_conv and any(c['id'] == triggering_conv for c in candidates):
            return triggering_conv
        return candidates[0]['id'] if candidates else None


class LastActiveFocusStrategy(FocusStrategy):
    """Focus on most recently active conversation."""
    def select_focus(self, candidates, context):
        if not candidates:
            return None
        # Sort by last activity timestamp
        sorted_candidates = sorted(
            candidates, 
            key=lambda c: c.get('last_activity', 0), 
            reverse=True
        )
        return sorted_candidates[0]['id']


class PriorityQueueStrategy(FocusStrategy):
    """Focus based on conversation priorities."""
    def __init__(self, priority_rules: Dict[str, int]):
        self.priority_rules = priority_rules
        
    def select_focus(self, candidates, context):
        if not candidates:
            return None
        # Score each candidate
        scored = []
        for c in candidates:
            score = 0
            if c.get('is_dm'):
                score += self.priority_rules.get('dm_bonus', 10)
            if c.get('has_mention'):
                score += self.priority_rules.get('mention_bonus', 5)
            if c.get('adapter_type') in self.priority_rules:
                score += self.priority_rules[c['adapter_type']]
            scored.append((score, c['id']))
        # Return highest priority
        scored.sort(reverse=True)
        return scored[0][1]


class FocusSelector:
    def __init__(self, strategy: str = "conversation", config: Dict[str, Any] = None):
        self.config = config or {}
        self.strategy = self._create_strategy(strategy)
        
    def _create_strategy(self, strategy_name: str) -> FocusStrategy:
        if strategy_name == "conversation":
            return ConversationFocusStrategy()
        elif strategy_name == "last_active":
            return LastActiveFocusStrategy()
        elif strategy_name == "priority":
            return PriorityQueueStrategy(self.config.get('priorities', {}))
        else:
            logger.warning(f"Unknown focus strategy: {strategy_name}, using conversation")
            return ConversationFocusStrategy()
            
    def select_focus(self, event_contexts: List[Dict[str, Any]], 
                     current_context: Dict[str, Any]) -> Optional[str]:
        """Select focus element from event contexts."""
        candidates = []
        for ctx in event_contexts:
            candidate = {
                'id': ctx.get('element_id') or ctx.get('conversation_id'),
                'conversation_id': ctx.get('conversation_id'),
                'adapter_type': ctx.get('adapter_type'),
                'is_dm': ctx.get('is_dm', False),
                'has_mention': ctx.get('has_mention', False),
                'last_activity': ctx.get('timestamp', time.time())
            }
            candidates.append(candidate)
            
        return self.strategy.select_focus(candidates, current_context)
        
    def generate_focus_element_id(self, event_contexts: List[Dict[str, Any]], 
                                 current_context: Dict[str, Any], 
                                 owner_space_id: Optional[str] = None) -> Optional[str]:
        """Select focus and generate full element ID."""
        focus_conversation_id = self.select_focus(event_contexts, current_context)
        
        if focus_conversation_id:
            # Find the context for this conversation
            ctx = next((c for c in event_contexts 
                       if c.get('conversation_id') == focus_conversation_id or 
                          c.get('element_id') == focus_conversation_id), None)
            if ctx:
                return ElementIdGenerator.generate_target_element_id(
                    adapter_id=ctx.get('adapter_id'),
                    conversation_id=ctx.get('conversation_id'),
                    is_dm=ctx.get('is_dm', False),
                    owner_space_id=owner_space_id
                )
        return None 