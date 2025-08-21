"""
YAML Rules Validator
Provides validation for YAML configuration files used by decider components.
"""

import yaml
from typing import Dict, Any, List, Tuple
import jsonschema

INTERRUPT_RULES_SCHEMA = {
    "type": "object",
    "properties": {
        "interrupt_event_types": {"type": "array", "items": {"type": "string"}},
        "force_normal_event_types": {"type": "array", "items": {"type": "string"}},
        "conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {"type": "string"},
                    "interrupt": {"type": "boolean"}
                }
            }
        }
    }
}

ACTIVATION_RULES_SCHEMA = {
    "type": "object", 
    "properties": {
        "activate_event_types": {"type": "array", "items": {"type": "string"}},
        "preempt_event_types": {"type": "array", "items": {"type": "string"}},
        "focus_selection": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string"},
                "priorities": {"type": "object"}
            }
        }
    }
}


def validate_rules(rules: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate rules against schema, return (is_valid, errors)."""
    try:
        jsonschema.validate(rules, schema)
        return True, []
    except jsonschema.ValidationError as e:
        return False, [str(e)]
    except Exception as e:
        return False, [f"Validation error: {str(e)}"]


def validate_interrupt_rules(rules: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate interrupt decider rules."""
    return validate_rules(rules, INTERRUPT_RULES_SCHEMA)


def validate_activation_rules(rules: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate activation decider rules."""
    return validate_rules(rules, ACTIVATION_RULES_SCHEMA) 