#!/usr/bin/env python3
"""
Chat Log to DAG Converter Utility

Converts arbitrary chat logs into Connectome-compatible DAG history files.
Reuses existing TimelineComponent and storage patterns to generate proper DAG structures.

Usage:
    python -m tools.chat_log_to_dag --input chat.json --output timeline.json --space-id demo_space
    python -m tools.chat_log_to_dag --input chat.csv --format csv --output timeline.json --space-id demo_space
"""

import argparse
import json
import csv
import uuid
import time
import logging
import hashlib
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Import existing Connectome components
from elements.elements.components.space.timeline_component import TimelineComponent
from elements.elements.base import BaseElement
from storage.file_storage import FileStorage
from storage import create_storage_from_env

# Import configuration system
try:
    from host.config import HostSettings, AgentConfig, ActivityAdapterConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

class ConnectomeConfigDiscovery:
    """Discovers existing Connectome configuration for automatic Space/Agent assignment."""
    
    def __init__(self):
        self.host_settings = None
        self.agents = []
        self.adapters = []
        self.config_loaded = False
        
    def load_configuration(self) -> bool:
        """Load Connectome configuration from environment variables."""
        if not CONFIG_AVAILABLE:
            logger.warning("Connectome configuration system not available - cannot auto-discover Spaces/Agents")
            return False
            
        try:
            self.host_settings = HostSettings()
            
            # Agents and adapters are already parsed in HostSettings.__init__
            self.agents = self.host_settings.agents
            self.adapters = self.host_settings.activity_client_adapter_configs
            
            logger.debug(f"Discovered {len(self.adapters)} activity adapters")
            logger.debug(f"Discovered {len(self.agents)} agents")
            
            self.config_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Connectome configuration: {e}", exc_info=True)
            return False
    
    def get_first_agent(self) -> Optional[AgentConfig]:
        """Get the first available agent from configuration."""
        if not self.config_loaded:
            self.load_configuration()
        
        return self.agents[0] if self.agents else None
    
    def get_first_adapter(self) -> Optional[ActivityAdapterConfig]:
        """Get the first available activity adapter from configuration."""
        if not self.config_loaded:
            self.load_configuration()
        
        return self.adapters[0] if self.adapters else None
    
    def get_suggested_space_id(self, agent: AgentConfig) -> str:
        """Generate a suggested space ID for the agent's InnerSpace."""
        return f"{agent.agent_id}_inner_space"
    
    def get_suggested_adapter_id(self) -> str:
        """Get the actual adapter ID from the first available adapter."""
        adapter = self.get_first_adapter()
        if adapter:
            return adapter.id  # Use the real adapter ID, not a fake one
        return "chat_log_importer"

class ChatMessage:
    """Represents a normalized chat message from various input formats.
    
    Only 'text' is mandatory - all other fields have intelligent defaults.
    """
    
    def __init__(self, 
                 text: str,
                 sender_id: str = None,
                 sender_name: str = None,
                 timestamp: Union[float, str, datetime, None] = None,
                 message_id: str = None,
                 is_dm: bool = False,
                 mentions: List[str] = None,
                 attachments: List[Dict] = None,
                 conversation_id: str = None,
                 metadata: Dict[str, Any] = None,
                 _message_index: int = None):
        
        if not text or not isinstance(text, str):
            raise ValueError("Text is required and must be a non-empty string")
        
        self.text = text.strip()
        
        # Generate intelligent defaults for missing fields
        self.sender_id = sender_id or self._generate_sender_id()
        self.sender_name = sender_name or self._generate_sender_name(self.sender_id)
        self.timestamp = self._normalize_timestamp(timestamp, _message_index)
        self.message_id = message_id or self._generate_message_id()
        self.is_dm = is_dm
        self.mentions = mentions or []
        self.attachments = attachments or []
        self.conversation_id = conversation_id or "imported_conversation" 
        self.metadata = metadata or {}

    def _normalize_timestamp(self, timestamp: Union[float, str, datetime, None], message_index: int = None) -> float:
        """Normalize various timestamp formats to Unix timestamp float."""
        if timestamp is None:
            return self._generate_timestamp(message_index)
        
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
        
        if isinstance(timestamp, str):
            try:
                # Try parsing ISO format first
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.timestamp()
            except ValueError:
                try:
                    # Try parsing as Unix timestamp string
                    return float(timestamp)
                except ValueError:
                    logger.warning(f"Could not parse timestamp '{timestamp}', generating procedural timestamp")
                    return self._generate_timestamp(message_index)
        
        if isinstance(timestamp, datetime):
            return timestamp.timestamp()
        
        logger.warning(f"Unknown timestamp type {type(timestamp)}, generating procedural timestamp")
        return self._generate_timestamp(message_index)

    def _generate_timestamp(self, message_index: int = None) -> float:
        """Generate meaningful timestamps using procgen heuristics."""
        base_time = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)  # Start of 2024, 9 AM UTC
        
        if message_index is not None:
            # Create realistic conversation flow with variable gaps
            # First message at base time, subsequent messages with realistic delays
            if message_index == 0:
                return base_time.timestamp()
            
            # Generate conversation-like timing patterns
            # Short bursts of messages, then longer pauses
            burst_size = 3  # Messages in a conversational burst
            
            if message_index % (burst_size * 2) < burst_size:
                # Within a burst - short delays (30 seconds to 3 minutes)
                delay_minutes = random.uniform(0.5, 3)
            else:
                # Between bursts - longer delays (5 minutes to 2 hours)
                delay_minutes = random.uniform(5, 120)
            
            # Add some randomness but keep it realistic
            total_delay = message_index * delay_minutes + random.uniform(-delay_minutes * 0.2, delay_minutes * 0.2)
            return (base_time + timedelta(minutes=total_delay)).timestamp()
        
        # Fallback: random time in the past year
        days_ago = random.randint(1, 365)
        hour = random.randint(8, 22)  # Realistic messaging hours
        minute = random.randint(0, 59)
        
        random_time = base_time - timedelta(days=days_ago) + timedelta(hours=hour, minutes=minute)
        return random_time.timestamp()

    def _generate_sender_id(self) -> str:
        """Generate a sender ID based on message content hash."""
        # Use first few words of message to create consistent sender ID
        words = self.text.lower().split()[:3]
        content_hash = hashlib.md5(" ".join(words).encode()).hexdigest()[:8]
        return f"user_{content_hash}"

    def _generate_sender_name(self, sender_id: str) -> str:
        """Generate a human-readable sender name from sender ID."""
        # Extract the hash part and map to names
        if sender_id.startswith("user_"):
            hash_part = sender_id[5:]
        else:
            hash_part = hashlib.md5(sender_id.encode()).hexdigest()[:8]
        
        # Simple name generation based on hash
        first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", 
                      "Iris", "Jack", "Kate", "Leo", "Maya", "Noah", "Olivia", "Paul"]
        last_names = ["Smith", "Johnson", "Brown", "Davis", "Miller", "Wilson", "Moore", 
                     "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin"]
        
        # Use hash to deterministically select names
        hash_int = int(hash_part, 16)
        first_idx = hash_int % len(first_names)
        last_idx = (hash_int // len(first_names)) % len(last_names)
        
        return f"{first_names[first_idx]} {last_names[last_idx]}"

    def _generate_message_id(self) -> str:
        """Generate a message ID based on content and timestamp."""
        content_sample = self.text[:50].lower().replace(" ", "")
        timestamp_str = str(int(self.timestamp))[-6:]  # Last 6 digits of timestamp
        content_hash = hashlib.md5(content_sample.encode()).hexdigest()[:6]
        return f"msg_{timestamp_str}_{content_hash}"

    def to_connectome_event_payload(self, adapter_id: str = "chat_log_importer", force_dm: bool = None) -> Dict[str, Any]:
        """Convert to Connectome event payload format matching real system events."""
        # Use force_dm if provided, otherwise use the message's is_dm setting
        is_dm_value = force_dm if force_dm is not None else self.is_dm
        
        # Create the nested payload data
        payload_data = {
            "source_adapter_id": adapter_id,
            "external_conversation_id": self.conversation_id,
            "is_dm": is_dm_value,
            "text": self.text,
            "sender_external_id": self.sender_id,
            "sender_display_name": self.sender_name,
            "timestamp": self.timestamp,
            "original_message_id_external": self.message_id,
            "mentions": self.mentions,
            "attachments": self.attachments,
            "adapter_type": "chat_log_importer",
            "original_adapter_data": {
                "message_id": self.message_id,
                "text": self.text,
                "sender": {
                    "user_id": self.sender_id,
                    "display_name": self.sender_name
                },
                "timestamp": self.timestamp,
                "conversation_id": self.conversation_id,
                "metadata": self.metadata
            }
        }
        
        # Return event with BOTH top-level and nested fields to match real system structure
        return {
            "event_type": "message_received",
            "source_adapter_id": adapter_id,                    # Top-level for ChatManager
            "external_conversation_id": self.conversation_id,   # Top-level for ChatManager
            "is_replayable": True,                              # Match real system events
            "payload": payload_data                             # Nested for other components
        }

class ChatLogParser:
    """Parses various chat log formats into normalized ChatMessage objects."""
    
    @staticmethod
    def parse_json_messages(data: Union[Dict, List]) -> List[ChatMessage]:
        """Parse JSON format chat logs."""
        messages = []
        
        # Handle both single dict and list of messages
        if isinstance(data, dict):
            if "messages" in data:
                message_list = data["messages"]
            else:
                # Single message object
                message_list = [data]
        else:
            message_list = data
        
        for idx, msg_data in enumerate(message_list):
            # Handle simple string messages
            if isinstance(msg_data, str):
                if not msg_data.strip():
                    logger.warning(f"Skipping empty message at index {idx}")
                    continue
                text = msg_data.strip()
                # Create a ChatMessage with just text - all other fields will be auto-generated
                message = ChatMessage(text=text, _message_index=idx)
                messages.append(message)
                continue
            
            # Handle dictionary messages  
            if not isinstance(msg_data, dict):
                logger.warning(f"Skipping message at index {idx}: unsupported type {type(msg_data)}")
                continue
                
            # Support various JSON schemas - now more flexible with defaults
            text = msg_data.get("text") or msg_data.get("content") or msg_data.get("message")
            
            # Skip empty messages
            if not text:
                logger.warning(f"Skipping message at index {idx}: no text content found")
                continue
            
            sender_id = msg_data.get("sender_id") or msg_data.get("user_id") or msg_data.get("author")
            sender_name = msg_data.get("sender_name") or msg_data.get("username") or msg_data.get("display_name")
            timestamp = msg_data.get("timestamp") or msg_data.get("time") or msg_data.get("created_at")
            message_id = msg_data.get("message_id") or msg_data.get("id")
            is_dm = msg_data.get("is_dm", msg_data.get("direct_message", False))
            mentions = msg_data.get("mentions", [])
            attachments = msg_data.get("attachments", [])
            conversation_id = msg_data.get("conversation_id") or msg_data.get("channel_id") or msg_data.get("chat_id")
            
            # Extract remaining fields as metadata
            metadata = {k: v for k, v in msg_data.items() 
                       if k not in ["text", "content", "message", "sender_id", "user_id", "author",
                                   "sender_name", "username", "display_name", "timestamp", "time", 
                                   "created_at", "message_id", "id", "is_dm", "direct_message",
                                   "mentions", "attachments", "conversation_id", "channel_id", "chat_id"]}
            
            message = ChatMessage(
                text=text,
                sender_id=sender_id,
                sender_name=sender_name,
                timestamp=timestamp,
                message_id=message_id,
                is_dm=is_dm,
                mentions=mentions,
                attachments=attachments,
                conversation_id=conversation_id,
                metadata=metadata,
                _message_index=idx
            )
            messages.append(message)
        
        return messages
    
    @staticmethod
    def parse_csv_messages(csv_path: Path) -> List[ChatMessage]:
        """Parse CSV format chat logs."""
        messages = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Map common CSV column names - now more flexible with defaults
                text = row.get("text") or row.get("message") or row.get("content")
                
                # Skip empty messages
                if not text:
                    logger.warning(f"Skipping CSV row {idx + 1}: no text content found")
                    continue
                
                sender_id = row.get("sender_id") or row.get("user_id") or row.get("username")
                sender_name = row.get("sender_name") or row.get("display_name") or row.get("name")
                timestamp = row.get("timestamp") or row.get("time") or row.get("date")
                message_id = row.get("message_id") or row.get("id")
                is_dm = row.get("is_dm", "false").lower() in ["true", "1", "yes"]
                conversation_id = row.get("conversation_id") or row.get("channel")
                
                # Parse mentions and attachments if present as JSON strings
                mentions = []
                if row.get("mentions"):
                    try:
                        mentions = json.loads(row["mentions"])
                    except json.JSONDecodeError:
                        mentions = [row["mentions"]]  # Single mention as string
                
                attachments = []
                if row.get("attachments"):
                    try:
                        attachments = json.loads(row["attachments"])
                    except json.JSONDecodeError:
                        pass  # Keep empty list
                
                # Extract remaining columns as metadata
                metadata = {k: v for k, v in row.items() 
                           if k not in ["text", "message", "content", "sender_id", "user_id", "username",
                                       "sender_name", "display_name", "name", "timestamp", "time", "date",
                                       "message_id", "id", "is_dm", "conversation_id", "channel", 
                                       "mentions", "attachments"]}
                
                message = ChatMessage(
                    text=text,
                    sender_id=sender_id,
                    sender_name=sender_name,
                    timestamp=timestamp,
                    message_id=message_id,
                    is_dm=is_dm,
                    mentions=mentions,
                    attachments=attachments,
                    conversation_id=conversation_id,
                    metadata=metadata,
                    _message_index=idx
                )
                messages.append(message)
        
        return messages

class MockElement(BaseElement):
    """Mock element for TimelineComponent testing."""
    
    def __init__(self, element_id: str):
        super().__init__(
            element_id=element_id,
            name=f"Mock Element {element_id}",
            description="Mock element for chat log to DAG conversion"
        )
        self.finalize_setup()

class ChatLogToDAGConverter:
    """Main converter class that creates DAG history files from chat logs."""
    
    def __init__(self, space_id: str = None, adapter_id: str = None, auto_assign: bool = False, 
                 conversation_mode: str = "auto", dm_interlocutor_name: str = None):
        self.space_id = space_id
        self.adapter_id = adapter_id
        self.auto_assign = auto_assign
        self.conversation_mode = conversation_mode  # "auto", "dm", or "group"
        self.dm_interlocutor_name = dm_interlocutor_name
        self.timeline_component = None
        self.mock_element = None
        self.config_discovery = None
        self.assigned_agent = None
        
        # Handle auto-assignment mode
        if auto_assign:
            self.config_discovery = ConnectomeConfigDiscovery()
            success = self.config_discovery.load_configuration()
            
            if success:
                # Get first available agent and adapter
                self.assigned_agent = self.config_discovery.get_first_agent()
                first_adapter = self.config_discovery.get_first_adapter()
                
                if self.assigned_agent:
                    # Auto-assign space ID based on agent
                    if not self.space_id:
                        self.space_id = self.config_discovery.get_suggested_space_id(self.assigned_agent)
                    logger.info(f"Auto-assigned to agent '{self.assigned_agent.agent_id}' ({self.assigned_agent.name})")
                    logger.info(f"Using space ID: {self.space_id}")
                else:
                    logger.warning("No agents found in configuration - using defaults")
                
                if first_adapter and not self.adapter_id:
                    self.adapter_id = self.config_discovery.get_suggested_adapter_id()
                    logger.info(f"Using adapter ID: {self.adapter_id}")
            else:
                logger.warning("Failed to load configuration - using defaults")
        
        # Set defaults if not assigned
        if not self.space_id:
            self.space_id = "imported_space"
        if not self.adapter_id:
            self.adapter_id = "chat_log_importer"
        
    def _determine_conversation_mode_and_interlocutor(self, messages: List[ChatMessage]) -> tuple[bool, str]:
        """Determine conversation mode and consistent interlocutor name for DM imports."""
        
        if self.conversation_mode == "group":
            return False, None
        elif self.conversation_mode == "dm":
            # Use provided interlocutor name or generate consistent one
            if self.dm_interlocutor_name:
                return True, self.dm_interlocutor_name
            
            # Generate consistent interlocutor from all unique senders
            unique_senders = set(msg.sender_name for msg in messages if msg.sender_name)
            if len(unique_senders) == 1:
                # Single sender - use their name
                interlocutor_name = list(unique_senders)[0]
            elif len(unique_senders) > 1:
                # Multiple senders - create combined name
                sorted_senders = sorted(unique_senders)
                if len(sorted_senders) <= 3:
                    interlocutor_name = ", ".join(sorted_senders)
                else:
                    interlocutor_name = f"{sorted_senders[0]} and {len(sorted_senders)-1} others"
            else:
                # No sender names - use generic
                interlocutor_name = "Imported Chat Participants"
            
            return True, interlocutor_name
        else:  # "auto" mode
            # Auto-detect based on message patterns
            # Use sender_name if available (more reliable than auto-generated sender_id)
            unique_names = set(msg.sender_name for msg in messages if msg.sender_name)
            if unique_names:
                unique_senders = unique_names
            else:
                unique_senders = set(msg.sender_id for msg in messages if msg.sender_id)
            
            # If 2 or fewer unique senders, treat as DM
            if len(unique_senders) <= 2:
                # Generate consistent interlocutor name
                unique_names = set(msg.sender_name for msg in messages if msg.sender_name)
                if len(unique_names) == 1:
                    interlocutor_name = list(unique_names)[0]
                elif len(unique_names) == 2:
                    interlocutor_name = " & ".join(sorted(unique_names))
                else:
                    interlocutor_name = "Chat Partner"
                return True, interlocutor_name
            else:
                # Multiple senders - treat as group chat
                return False, None

    async def convert_messages_to_dag(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """Convert chat messages to Connectome DAG format."""
        
        # Create mock element and timeline component
        # Use the actual space_id as element ID so TimelineComponent generates correct storage keys
        self.mock_element = MockElement(self.space_id)
        self.mock_element.add_component(TimelineComponent)
        self.timeline_component = self.mock_element.get_component_by_type("TimelineComponent")
        
        if not self.timeline_component:
            raise RuntimeError("Failed to create TimelineComponent")
        
        # Initialize timeline component  
        self.timeline_component.initialize()
        
        logger.info(f"Converting {len(messages)} messages to DAG for space '{self.space_id}'")
        
        # Sort messages by timestamp to ensure proper order
        sorted_messages = sorted(messages, key=lambda m: m.timestamp)
        
        # Determine conversation mode and interlocutor consistency
        is_dm_conversation, dm_interlocutor_name = self._determine_conversation_mode_and_interlocutor(sorted_messages)
        
        logger.info(f"Conversation mode: {'DM' if is_dm_conversation else 'Group'}")
        if is_dm_conversation and dm_interlocutor_name:
            logger.info(f"DM interlocutor: {dm_interlocutor_name}")
        
        # Add messages as timeline events
        for i, message in enumerate(sorted_messages):
            # Use determined conversation mode
            force_dm = is_dm_conversation
            
            # For DM mode, ensure consistent interlocutor name
            if is_dm_conversation and dm_interlocutor_name:
                # Override the sender name to maintain consistency
                original_sender_name = message.sender_name
                message.sender_name = dm_interlocutor_name
                
            event_payload = message.to_connectome_event_payload(self.adapter_id, force_dm=force_dm)
            
            # Restore original sender name after creating payload
            if is_dm_conversation and dm_interlocutor_name:
                message.sender_name = original_sender_name
            
            # Add to primary timeline
            event_id = self.timeline_component.add_event_to_primary_timeline(event_payload)
            
            if event_id:
                logger.debug(f"Added message {i+1}/{len(messages)}: {message.text[:50]}... (ID: {event_id})")
            else:
                logger.error(f"Failed to add message {i+1}: {message.text[:50]}...")
        
        # Let TimelineComponent persist to storage using its own logic
        await self.timeline_component._persist_timeline_state()
        
        # Extract DAG data in Connectome format for human-readable output
        timeline_state = self._extract_timeline_state()
        timeline_events = self._extract_timeline_events()
        
        # Enhanced metadata with agent information
        metadata = {
            "space_id": self.space_id,
            "adapter_id": self.adapter_id,
            "message_count": len(messages),
            "time_range": {
                "start": min(m.timestamp for m in messages) if messages else None,
                "end": max(m.timestamp for m in messages) if messages else None
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "chat_log_to_dag_converter",
            "auto_assigned": self.auto_assign
        }
        
        # Add agent information if auto-assigned
        if self.auto_assign and self.assigned_agent:
            metadata["assigned_agent"] = {
                "agent_id": self.assigned_agent.agent_id,
                "name": self.assigned_agent.name,
                "description": self.assigned_agent.description,
                "agent_loop_component_type": self.assigned_agent.agent_loop_component_type_name
            }
            
        # Add adapter information if discovered
        if self.config_discovery:
            first_adapter = self.config_discovery.get_first_adapter()
            if first_adapter:
                metadata["source_adapter"] = {
                    "adapter_id": first_adapter.id,
                    "url": first_adapter.url
                }
        
        return {
            "timeline_state": timeline_state,
            "timeline_events": timeline_events,
            "metadata": metadata
        }
    
    def _extract_timeline_state(self) -> Dict[str, Any]:
        """Extract timeline state in storage format."""
        if not self.timeline_component:
            return {}
        
        # Convert sets to lists for JSON serialization (matching TimelineComponent._persist_timeline_state)
        timelines_serializable = {}
        for timeline_id, timeline_info in self.timeline_component._state['_timelines'].items():
            timeline_copy = timeline_info.copy()
            if isinstance(timeline_copy.get('head_event_ids'), set):
                timeline_copy['head_event_ids'] = list(timeline_copy['head_event_ids'])
            timelines_serializable[timeline_id] = timeline_copy
        
        return {
            '_timelines': timelines_serializable,
            '_primary_timeline_id': self.timeline_component._state['_primary_timeline_id'],
            'last_updated': time.time()
        }
    
    def _extract_timeline_events(self) -> Dict[str, Any]:
        """Extract timeline events in storage format.""" 
        if not self.timeline_component:
            return {}
        
        return {
            '_all_events': self.timeline_component._state['_all_events'],
            'last_updated': time.time()
        }

async def save_dag_to_storage_format(dag_data: Dict[str, Any], output_path: Path, space_id: str, auto_assign: bool = False):
    """Save DAG data in Connectome storage format using the actual storage system."""
    
    # Create timeline state and events data
    state_data = dag_data["timeline_state"]
    events_data = dag_data["timeline_events"]
    
    # Determine where to save files based on auto-assignment mode
    if auto_assign and dag_data.get("metadata", {}).get("auto_assigned"):
        # TimelineComponent has already persisted to storage using correct naming
        # We just need to create the human-readable combined file
        try:
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # TimelineComponent uses "space_" prefix in storage keys
            storage_space_id = f"space_{space_id}"
            combined_data = {
                "storage_keys": {
                    "timeline_state": f"timeline_state_{storage_space_id}",
                    "timeline_events": f"timeline_events_{storage_space_id}"
                },
                "note": "Data has been stored using TimelineComponent's native storage logic",
                **dag_data
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)
            
            logger.info("DAG data stored using Connectome's native TimelineComponent:")
            logger.info(f"  🏛️  Timeline state: timeline_state_{storage_space_id} (ready for Connectome)")
            logger.info(f"  📊 Timeline events: timeline_events_{storage_space_id} (ready for Connectome)")
            logger.info(f"  📋 Combined format: {output_path} (human-readable)")
            
        except Exception as e:
            logger.error(f"Failed to create human-readable output file: {e}", exc_info=True)
            logger.info("Storage was successful, but combined file creation failed")
    
    if not auto_assign or not dag_data.get("metadata", {}).get("auto_assigned"):
        # Save as separate files for manual inspection/use
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timeline state file in FileStorage format
        state_file_data = {
            "state_key": f"timeline_state_{space_id}",
            "data": state_data,
            "stored_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Create timeline events file in FileStorage format
        events_file_data = {
            "state_key": f"timeline_events_{space_id}",
            "data": events_data,
            "stored_at": datetime.now(timezone.utc).isoformat()
        }
        
        state_file = output_dir / f"timeline_state_{space_id}.json"
        events_file = output_dir / f"timeline_events_{space_id}.json"
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_file_data, f, indent=2, ensure_ascii=False)
        
        with open(events_file, 'w', encoding='utf-8') as f:
            json.dump(events_file_data, f, indent=2, ensure_ascii=False)
        
        # Also save combined format for easier inspection
        combined_data = {
            "timeline_state_file": str(state_file),
            "timeline_events_file": str(events_file),
            **dag_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"DAG files saved:")
        logger.info(f"  Timeline state: {state_file}")
        logger.info(f"  Timeline events: {events_file}")
        logger.info(f"  Combined format: {output_path}")
    
    # Show assignment details for auto-assigned configurations
    if auto_assign and dag_data.get("metadata", {}).get("auto_assigned"):
        metadata = dag_data.get("metadata", {})
        if "assigned_agent" in metadata:
            agent_info = metadata["assigned_agent"]
            logger.info(f"  🤖 Assigned to agent: {agent_info['name']} ({agent_info['agent_id']})")
        if "source_adapter" in metadata:
            adapter_info = metadata["source_adapter"]
            logger.info(f"  🔌 Source adapter: {adapter_info['adapter_id']}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Convert chat logs to Connectome DAG format")
    parser.add_argument("--input", "-i", required=True, type=Path,
                       help="Input chat log file (JSON or CSV)")
    parser.add_argument("--output", "-o", type=Path,
                       help="Output DAG file path (JSON). Optional when using --auto-assign - will be auto-generated.")
    parser.add_argument("--space-id", 
                       help="Space ID for the DAG (auto-discovered if --auto-assign is used)")
    parser.add_argument("--format", choices=["json", "csv"], default="auto",
                       help="Input format (auto-detect by default)")
    parser.add_argument("--adapter-id", 
                       help="Adapter ID to use in events (auto-discovered if --auto-assign is used)")  
    parser.add_argument("--auto-assign", action="store_true",
                       help="Automatically assign to first found Space and Agent from Connectome configuration")
    parser.add_argument("--conversation-mode", choices=["auto", "dm", "group"], default="auto",
                       help="How to treat the conversation: auto-detect (default), force DM, or force group chat")
    parser.add_argument("--dm-interlocutor-name", 
                       help="Name to use for DM partner (auto-generated if not provided)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    async def convert():
        try:
            # Validate arguments
            if not args.auto_assign and not args.space_id:
                logger.error("Either --space-id must be provided or --auto-assign must be used")
                return 1
            
            if not args.auto_assign and not args.output:
                logger.error("--output is required when not using --auto-assign")
                return 1
                
            # Determine input format
            input_format = args.format
            if input_format == "auto":
                input_format = "json" if args.input.suffix.lower() == ".json" else "csv"
            
            logger.info(f"Reading {input_format.upper()} chat log from {args.input}")
            
            # Parse input messages
            if input_format == "json":
                with open(args.input, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                messages = ChatLogParser.parse_json_messages(data)
            else:  # csv
                messages = ChatLogParser.parse_csv_messages(args.input)
            
            logger.info(f"Parsed {len(messages)} messages")
            
            if not messages:
                logger.error("No messages found in input file")
                return 1
            
            # Convert to DAG format with auto-assignment support
            converter = ChatLogToDAGConverter(
                space_id=args.space_id,
                adapter_id=args.adapter_id,
                auto_assign=args.auto_assign,
                conversation_mode=args.conversation_mode,
                dm_interlocutor_name=args.dm_interlocutor_name
            )
            dag_data = await converter.convert_messages_to_dag(messages)
            
            # Auto-generate output path if not provided and using auto-assign
            output_path = args.output
            if not output_path and args.auto_assign:
                # Generate meaningful output filename based on input and discovered config
                input_stem = args.input.stem  # filename without extension
                space_id = converter.space_id
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{input_stem}_to_{space_id}_{timestamp}.json"
                output_path = args.input.parent / output_filename
                logger.info(f"📄 Auto-generated output path: {output_path}")
            
            # Save output with auto-assignment awareness
            await save_dag_to_storage_format(
                dag_data, 
                output_path, 
                converter.space_id,  # Use the resolved space_id
                auto_assign=args.auto_assign
            )
            
            if args.auto_assign:
                logger.info("🎉 Conversion completed! Files are ready for Connectome to load.")
                logger.info("💡 Start Connectome with CONNECTOME_EVENT_REPLAY_MODE=enabled to load the imported chat history.")
            else:
                logger.info("Conversion completed successfully!")
            return 0
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}", exc_info=True)
            return 1
    
    import asyncio
    exit_code = asyncio.run(convert())
    exit(exit_code)

if __name__ == "__main__":
    main()