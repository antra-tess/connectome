"""
Structured logging utilities for agent loop decision tracking.
Provides comprehensive decision audit trail for interrupt classification and activations.
"""

import logging
import json
import time
from enum import Enum
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    INTERRUPT_CLASSIFICATION = "interrupt_classification"
    ACTIVATION_DECISION = "activation_decision" 
    FOCUS_SELECTION = "focus_selection"


def log_decision(decision_type: DecisionType, event_id: str, 
                event_type: str, decision: Dict[str, Any], 
                reason: str, metadata: Optional[Dict[str, Any]] = None):
    """Log structured decision for analysis."""
    log_entry = {
        "timestamp": time.time(),
        "decision_type": decision_type.value,
        "event_id": event_id,
        "event_type": event_type,
        "decision": decision,
        "reason": reason,
        "metadata": metadata or {}
    }
    
    # Log as JSON for parsing
    logger.info(f"DECISION_LOG: {json.dumps(log_entry)}")
    
    # Also log human-readable summary
    logger.info(f"[{decision_type.value}] Event {event_id} ({event_type}) -> "
                f"{decision.get('interrupt_class', decision.get('activation_decision'))}, "
                f"reason: {reason}")


def log_rules_loaded(component_type: str, element_id: str, rules: Dict[str, Any]):
    """Log effective rules at INFO level for visibility."""
    logger.info(f"[{element_id}] {component_type} effective rules loaded:")
    
    if component_type == "InterruptDeciderComponent":
        interrupt_types = list(rules.get('interrupt_event_types', []))
        force_normal_types = list(rules.get('force_normal_event_types', []))
        conditions = rules.get('conditions', [])
        
        logger.info(f"  Interrupt event types: {interrupt_types}")
        logger.info(f"  Force normal types: {force_normal_types}")
        logger.info(f"  Conditions: {len(conditions)} rules")
        for i, cond in enumerate(conditions):
            logger.info(f"    Rule {i+1}: type={cond.get('type')}, interrupt={cond.get('interrupt')}")
    
    elif component_type == "ActivationDeciderComponent":
        activate_types = list(rules.get('activate_event_types', []))
        preempt_types = list(rules.get('preempt_event_types', []))
        focus_resolution = rules.get('focus_resolution', {})
        
        logger.info(f"  Activate event types: {activate_types}")
        logger.info(f"  Preempt event types: {preempt_types}")
        logger.info(f"  Focus resolution strategy: {focus_resolution.get('strategy', 'direct')}")
        if focus_resolution.get('config'):
            logger.info(f"  Focus resolution config: {focus_resolution['config']}")


def log_rules_reload_attempt(component_type: str, element_id: str, success: bool, 
                           error_msg: Optional[str] = None):
    """Log YAML rules reload attempts."""
    if success:
        logger.info(f"[{element_id}] {component_type} rules reloaded successfully")
    else:
        logger.warning(f"[{element_id}] {component_type} rules reload failed: {error_msg}")


def log_validation_error(component_type: str, element_id: str, validation_errors: list):
    """Log YAML validation errors."""
    logger.error(f"[{element_id}] {component_type} rule validation failed:")
    for error in validation_errors:
        logger.error(f"  - {error}") 