import logging
from typing import Dict, Any, Optional, List
import os
import asyncio
import yaml

from ..base_component import Component
from elements.component_registry import register_component
from .logging_utils import log_decision, log_rules_loaded, DecisionType, log_rules_reload_attempt, log_validation_error
from .rule_validator import validate_interrupt_rules

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
class InterruptDeciderComponent(Component):
	COMPONENT_TYPE = "InterruptDeciderComponent"

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._rules: Dict[str, Any] = {}
		self._recent_focus_candidates: List[str] = []
		self._rules_file_path: Optional[str] = None
		self._file_watcher: Optional[RulesFileWatcher] = None

	def initialize(self, **kwargs) -> None:
		super().initialize(**kwargs)
		self._load_rules()
		logger.info(f"InterruptDeciderComponent initialized for {self.owner.id if self.owner else 'unknown'} with rules: {list(self._rules.keys())}")
		# Log effective rules for visibility
		log_rules_loaded("InterruptDeciderComponent", self.owner.id if self.owner else 'unknown', self._rules)
		
		# Start file watcher for hot reload
		if self._rules_file_path and not self._file_watcher:
			self._file_watcher = RulesFileWatcher(self, self._rules_file_path)
			self._file_watcher.start()

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
						rules = data.get('interrupt_rules', data) or {}
						
						# Validate rules
						is_valid, errors = validate_interrupt_rules(rules)
						if is_valid:
							self._rules = rules
							self._rules_file_path = path  # Remember successful path
							return
						else:
							log_validation_error("InterruptDeciderComponent", 
												self.owner.id if self.owner else 'unknown', 
												errors)
							logger.warning(f"Skipping invalid rules from {path}")
					except Exception as e:
						logger.warning(f"Failed to load YAML rules from {path}: {e}")
		except Exception as e:
			logger.debug(f"InterruptDecider _load_rules error: {e}")
		self._rules = {}

	async def reload_rules(self) -> None:
		"""Reload rules from file with validation."""
		if not self._rules_file_path:
			return
			
		try:
			with open(self._rules_file_path, 'r') as f:
				data = yaml.safe_load(f) or {}
			rules = data.get('interrupt_rules', data) or {}
			
			# Validate new rules
			is_valid, errors = validate_interrupt_rules(rules)
			if is_valid:
				self._rules = rules
				log_rules_reload_attempt("InterruptDeciderComponent", 
										self.owner.id if self.owner else 'unknown', True)
				log_rules_loaded("InterruptDeciderComponent", 
								self.owner.id if self.owner else 'unknown', self._rules)
			else:
				log_validation_error("InterruptDeciderComponent", 
									self.owner.id if self.owner else 'unknown', errors)
				log_rules_reload_attempt("InterruptDeciderComponent", 
										self.owner.id if self.owner else 'unknown', 
										False, f"Validation failed: {'; '.join(errors)}")
		except Exception as e:
			log_rules_reload_attempt("InterruptDeciderComponent", 
									self.owner.id if self.owner else 'unknown', 
									False, str(e))

	def classify_interrupt(self, event_payload: Dict[str, Any]) -> Dict[str, Any]:
		"""Classify interrupt vs normal. Default: DM or mention → interrupt; else normal. YAML can override."""
		etype = event_payload.get('event_type')
		payload = event_payload.get('payload', {})
		is_dm = bool(payload.get('is_dm', False))
		mentions = payload.get('mentions', []) or []

		interrupt_event_types = set(self._rules.get('interrupt_event_types', []))
		force_normal_types = set(self._rules.get('force_normal_event_types', []))
		conditions: List[Dict[str, Any]] = self._rules.get('conditions', []) or []
		rule_interrupt = etype in interrupt_event_types
		rule_normal = etype in force_normal_types

		# Default decision
		interrupt = False
		if rule_interrupt:
			interrupt = True
		elif rule_normal:
			interrupt = False
		# else:
		# 	interrupt = bool(is_dm or self._is_mention(mentions))

		# Evaluate optional condition rules (first match wins)
		for cond in conditions:
			try:
				c_type = cond.get('type')
				if c_type == 'typing_from_last_activator':
					if etype == 'typing_notification' and self._is_from_last_activator(payload):
						interrupt = bool(cond.get('interrupt', True))
						break
				elif c_type == 'field_equals':
					# Generic matcher: event.payload[field] == value
					field = cond.get('field')
					value = cond.get('value')
					if field and self._deep_get(payload, field) == value:
						interrupt = bool(cond.get('interrupt', True))
						break
				elif c_type == 'event_type':
					if etype == cond.get('value'):
						interrupt = bool(cond.get('interrupt', True))
						break
			except Exception:
				continue

		decision = {"interrupt_class": "interrupt" if interrupt else "normal"}

		# Log structured decision for analysis
		event_id = event_payload.get('id', 'unknown')
		reason = "dm_detected" if is_dm else "mention_detected" if mentions else "rule_match" if (rule_interrupt or rule_normal) else "default"
		metadata = {
			"is_dm": is_dm, 
			"mentions": mentions,
			"rule_interrupt": rule_interrupt,
			"rule_normal": rule_normal
		}
		
		log_decision(
			DecisionType.INTERRUPT_CLASSIFICATION,
			event_id=event_id,
			event_type=etype,
			decision=decision,
			reason=reason,
			metadata=metadata
		)

		# Track focus candidates
		conv = event_payload.get('external_conversation_id') or payload.get('external_conversation_id')
		if conv and conv not in self._recent_focus_candidates:
			self._recent_focus_candidates.append(conv)
			if len(self._recent_focus_candidates) > 50:
				self._recent_focus_candidates.pop(0)
		return decision

	def get_focus_candidates(self) -> List[str]:
		return list(self._recent_focus_candidates)

	def _is_mention(self, mentions) -> bool:
		try:
			agent_name = getattr(self.owner, 'agent_name', None)
			alias = getattr(self.owner, 'alias', None)
			# Env-based extra handles (comma-separated)
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

	def _is_from_last_activator(self, payload: Dict[str, Any]) -> bool:
		"""Check if event sender matches last activation sender in same conversation."""
		try:
			# Find ActivationDecider sibling to get last activation context
			activation_decider = None
			for _name, comp in self.owner.get_components().items():
				if getattr(comp, 'COMPONENT_TYPE', '') == 'ActivationDeciderComponent':
					activation_decider = comp
					break
			if not activation_decider or not hasattr(activation_decider, 'get_last_activation_context'):
				return False
			last_ctx = activation_decider.get_last_activation_context() or {}
			if not last_ctx:
				return False
			last_sender = last_ctx.get('sender_id')
			last_conv = last_ctx.get('conversation_id')
			curr_sender = payload.get('sender_id') or payload.get('user_id') or payload.get('author_id')
			curr_conv = payload.get('external_conversation_id')
			return bool(last_sender and curr_sender and last_conv and curr_conv and last_sender == curr_sender and last_conv == curr_conv)
		except Exception:
			return False

	def _deep_get(self, data: Dict[str, Any], dotted_path: str) -> Any:
		try:
			parts = dotted_path.split('.')
			curr = data
			for p in parts:
				if isinstance(curr, dict) and p in curr:
					curr = curr[p]
				else:
					return None
			return curr
		except Exception:
			return None 