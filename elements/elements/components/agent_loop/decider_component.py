import logging
from typing import Dict, Any, Optional, List
import os
import asyncio
import yaml

from ..base_component import Component
from elements.component_registry import register_component
from elements.utils.element_id_generator import ElementIdGenerator
from .logging_utils import log_decision, log_rules_loaded, DecisionType, log_rules_reload_attempt, log_validation_error
from .rule_validator import validate_activation_rules
from .focus_selector import FocusSelector

logger = logging.getLogger(__name__)


class RulesFileWatcher:
    def __init__(self, component, file_path: str):
        self.component = component
        self.file_path = file_path
        self.last_modified = 0
        self._watch_task = None
        
    def start(self):
        self._watch_task = asyncio.create_task(self._watch_loop())
        
    async def _watch_loop(self):
        while True:
            try:
                if os.path.exists(self.file_path):
                    stat = os.stat(self.file_path)
                    if stat.st_mtime > self.last_modified:
                        self.last_modified = stat.st_mtime
                        await self.component.reload_rules()
            except Exception as e:
                logger.debug(f"Rules file watch error: {e}")
            await asyncio.sleep(5)  # Check every 5 seconds


@register_component
class ActivationDeciderComponent(Component):
	COMPONENT_TYPE = "ActivationDeciderComponent"

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._rules: Dict[str, Any] = {}
		self._heartbeat_ms: int = 300
		self._pending_decisions: Dict[str, Dict[str, Any]] = {}
		self._event_context_by_id: Dict[str, Dict[str, Any]] = {}
		self._recent_focus_candidates: List[str] = []
		# NEW: Track last activation context for conditional rules in InterruptDecider
		self._last_activation_context: Dict[str, Any] = {}
		self._rules_file_path: Optional[str] = None
		self._file_watcher: Optional[RulesFileWatcher] = None

	def initialize(self, **kwargs) -> None:
		super().initialize(**kwargs)
		self._load_rules()
		self._heartbeat_ms = int(kwargs.get("heartbeat_ms", 300))
		logger.info(f"ActivationDeciderComponent initialized for {self.owner.id if self.owner else 'unknown'} with rules: {list(self._rules.keys())}")
		# Log effective rules for visibility
		log_rules_loaded("ActivationDeciderComponent", self.owner.id if self.owner else 'unknown', self._rules)
		
		# Start file watcher for hot reload
		if self._rules_file_path and not self._file_watcher:
			self._file_watcher = RulesFileWatcher(self, self._rules_file_path)
			self._file_watcher.start()
			
		# Initialize focus selector
		focus_config = self._rules.get('focus_selection', {})
		strategy = focus_config.get('strategy', 'conversation')
		self._focus_selector = FocusSelector(strategy, focus_config)

	def _load_rules(self) -> None:
		try:
			agent_id = getattr(self.owner, 'agent_id', None)
			config_dir = os.environ.get('CONNECTOME_CONFIG_DIR', 'config/agents')
			candidates = []
			if agent_id:
				candidates.append(os.path.join(config_dir, agent_id, 'decider.yaml'))
			candidates.append(os.path.join('config', 'decider.yaml'))
			for path in candidates:
				if os.path.exists(path):
					try:
						with open(path, 'r') as f:
							data = yaml.safe_load(f) or {}
						rules = data.get('activation_rules', data) or {}
						
						# Validate rules
						is_valid, errors = validate_activation_rules(rules)
						if is_valid:
							self._rules = rules
							self._rules_file_path = path  # Remember successful path
							return
						else:
							log_validation_error("ActivationDeciderComponent", 
												self.owner.id if self.owner else 'unknown', 
												errors)
							logger.warning(f"Skipping invalid rules from {path}")
					except Exception as e:
						logger.warning(f"Failed to load YAML rules from {path}: {e}")
		except Exception as e:
			logger.debug(f"ActivationDecider _load_rules error: {e}")
		self._rules = {}

	async def reload_rules(self) -> None:
		"""Reload rules from file with validation."""
		if not self._rules_file_path:
			return
			
		try:
			with open(self._rules_file_path, 'r') as f:
				data = yaml.safe_load(f) or {}
			rules = data.get('activation_rules', data) or {}
			
			# Validate new rules
			is_valid, errors = validate_activation_rules(rules)
			if is_valid:
				self._rules = rules
				# Reinitialize focus selector with new rules
				focus_config = self._rules.get('focus_selection', {})
				strategy = focus_config.get('strategy', 'conversation')
				self._focus_selector = FocusSelector(strategy, focus_config)
				
				log_rules_reload_attempt("ActivationDeciderComponent", 
										self.owner.id if self.owner else 'unknown', True)
				log_rules_loaded("ActivationDeciderComponent", 
								self.owner.id if self.owner else 'unknown', self._rules)
			else:
				log_validation_error("ActivationDeciderComponent", 
									self.owner.id if self.owner else 'unknown', errors)
				log_rules_reload_attempt("ActivationDeciderComponent", 
										self.owner.id if self.owner else 'unknown', 
										False, f"Validation failed: {'; '.join(errors)}")
		except Exception as e:
			log_rules_reload_attempt("ActivationDeciderComponent", 
									self.owner.id if self.owner else 'unknown', 
									False, str(e))

	def classify_event(self, event_payload: Dict[str, Any]) -> Dict[str, Any]:
		"""Classify event for activation (rule-based)."""
		etype = event_payload.get("event_type")
		payload = event_payload.get("payload", {})
		is_dm = bool(payload.get("is_dm", False))
		mentions = payload.get("mentions", []) or []

		# Track focus candidates from event context
		try:
			external_conversation_id = event_payload.get("external_conversation_id") or payload.get("external_conversation_id")
			if external_conversation_id and external_conversation_id not in self._recent_focus_candidates:
				self._recent_focus_candidates.append(external_conversation_id)
				if len(self._recent_focus_candidates) > 50:
					self._recent_focus_candidates.pop(0)
		except Exception:
			pass

		activate_types = set(self._rules.get('activate_event_types', []))
		high_priority_types = set(self._rules.get('preempt_event_types', []))

		activate = False
		priority = 'queue'
		reason = 'none'

		if etype in activate_types:
			activate = True
			reason = 'rule_match'
		elif etype == 'message_received' and (is_dm or self._is_mention_for_agent(mentions)):
			activate = True
			reason = 'direct_message' if is_dm else 'mention'

		if etype in high_priority_types:
			priority = 'preempt'

		decision = {
			"activation_decision": bool(activate),
			"activation_priority": priority if activate else 'none',
			"activation_reason": reason if activate else 'none'
		}

		# Log structured decision for analysis
		event_id = event_payload.get('id', 'unknown')
		metadata = {
			"is_dm": is_dm,
			"mentions": mentions,
			"priority": priority,
			"activate_types_match": etype in activate_types,
			"high_priority_match": etype in high_priority_types
		}
		
		log_decision(
			DecisionType.ACTIVATION_DECISION,
			event_id=event_id,
			event_type=etype,
			decision=decision,
			reason=reason,
			metadata=metadata
		)
		logger.critical(f"Activation decision: {decision}")
		logger.critical(f"Payload: {payload}")
		return decision

	def remember_decision(self, event_id: str, decision: Dict[str, Any]) -> None:
		self._pending_decisions[event_id] = decision

	def remember_event_context(self, event_id: str, context: Dict[str, Any]) -> None:
		self._event_context_by_id[event_id] = context or {}

	def on_component_processed(self, ack_payload: Dict[str, Any]) -> None:
		try:
			original_event_id = ack_payload.get('original_event_id')
			if not original_event_id:
				return
			decision = self._pending_decisions.get(original_event_id)
			if not decision:
				return
			if decision.get('activation_decision'):
				self._emit_activation_after_processing(decision, ack_payload)
		except Exception as e:
			logger.debug(f"ActivationDecider on_component_processed error: {e}")

	def _emit_activation_after_processing(self, decision: Dict[str, Any], ack_payload: Dict[str, Any]) -> None:
		try:
			parent_space = self.owner.get_parent_object() if hasattr(self.owner, 'get_parent_object') else self.owner
			if not parent_space or not hasattr(parent_space, 'receive_event'):
				return
			focus_element_id = None
			try:
				original_event_id = ack_payload.get('original_event_id')
				ctx = self._event_context_by_id.get(original_event_id, {})
				
				# Use advanced focus selection strategy
				event_contexts = [ctx] if ctx else []
				focus_element_id = self._select_focus_element(event_contexts)
				
				# NEW: Remember last activation context for later conditional rules
				conversation_id = ctx.get('conversation_id') if ctx else None
				self._last_activation_context = {
					"sender_id": ctx.get('sender_id') if ctx else None,
					"conversation_id": conversation_id,
					"adapter_id": ctx.get('adapter_id') if ctx else None
				}
			except Exception:
				pass
			envelope = {
				"event_type": "activation_call",
				"is_replayable": False,
				"payload": {
					"event_type": "activation_call",
					"activation_reason": decision.get('activation_reason', 'rule_match'),
					"priority": decision.get('activation_priority', 'queue'),
					"timestamp": __import__('time').time(),
					"focus_context": ({ "focus_element_id": focus_element_id } if focus_element_id else {})
				}
			}
			timeline_context = {"timeline_id": parent_space.get_primary_timeline() if hasattr(parent_space, 'get_primary_timeline') else None}
			parent_space.receive_event(envelope, timeline_context)
		except Exception as e:
			logger.debug(f"ActivationDecider activation emit error: {e}")

	async def evaluate_and_maybe_activate(self, context: Dict[str, Any]) -> None:
		try:
			if not self._pending_decisions:
				return
			last_event_id, decision = next(reversed(self._pending_decisions.items()))
			if decision.get('activation_decision'):
				parent_space = self.owner.get_parent_object() if hasattr(self.owner, 'get_parent_object') else self.owner
				if not parent_space or not hasattr(parent_space, 'receive_event'):
					return
				ctx = self._event_context_by_id.get(last_event_id, {})
				
				# Use advanced focus selection strategy
				event_contexts = [ctx] if ctx else []
				focus_element_id = self._select_focus_element(event_contexts)
				
				# NEW: Remember last activation context
				conversation_id = ctx.get('conversation_id') if ctx else None
				self._last_activation_context = {
					"sender_id": ctx.get('sender_id') if ctx else None,
					"conversation_id": conversation_id,
					"adapter_id": ctx.get('adapter_id') if ctx else None
				}
				envelope = {
					"event_type": "activation_call",
					"is_replayable": False,
					"payload": {
						"event_type": "activation_call",
						"activation_reason": decision.get('activation_reason', 'rule_match'),
						"priority": decision.get('activation_priority', 'queue'),
						"timestamp": __import__('time').time(),
						"focus_context": ({ "focus_element_id": focus_element_id } if focus_element_id else {})
					}
				}
				timeline_context = {"timeline_id": parent_space.get_primary_timeline() if hasattr(parent_space, 'get_primary_timeline') else None}
				parent_space.receive_event(envelope, timeline_context)
		except Exception as e:
			logger.debug(f"ActivationDecider evaluate_and_maybe_activate error: {e}")

	def _is_mention_for_agent(self, mentions) -> bool:
		try:
			agent_name = getattr(self.owner, 'agent_name', None)
			alias = getattr(self.owner, 'alias', None)
			# NEW: Load additional handles from env (comma-separated)
			extra_handles_env = os.environ.get('CONNECTOME_AGENT_HANDLES') or os.environ.get('AGENT_HANDLES') or os.environ.get('BOT_HANDLES')
			extra_handles = []
			if extra_handles_env:
				try:
					extra_handles = [h.strip() for h in extra_handles_env.split(',') if h.strip()]
				except Exception:
					extra_handles = []
			if not mentions:
				return False
			handles = [h for h in [agent_name, alias] if h] + extra_handles
			return any(m in handles for m in mentions)
		except Exception:
			return False

	def get_focus_candidates(self) -> List[str]:
		return list(self._recent_focus_candidates)

	def _select_focus_element(self, event_contexts: List[Dict[str, Any]]) -> Optional[str]:
		"""Use configured strategy to select focus."""
		current_context = {
			'triggering_conversation_id': event_contexts[-1].get('conversation_id') if event_contexts else None
		}
		parent_space = self.owner.get_parent_object() if hasattr(self.owner, 'get_parent_object') else self.owner
		owner_space_id = getattr(parent_space, 'id', None) if parent_space else None
		
		return self._focus_selector.generate_focus_element_id(
			event_contexts, current_context, owner_space_id)

	# NEW: Expose last activation context for sibling components (e.g., InterruptDecider)
	def get_last_activation_context(self) -> Dict[str, Any]:
		try:
			return dict(self._last_activation_context)
		except Exception:
			return {} 