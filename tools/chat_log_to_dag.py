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
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Import existing Connectome components
from elements.elements.components.space.timeline_component import TimelineComponent
from elements.elements.base import BaseElement
from storage.file_storage import FileStorage
from storage import create_storage_from_env

logger = logging.getLogger(__name__)

class ChatMessage:
    """Represents a normalized chat message from various input formats."""
    
    def __init__(self, 
                 text: str,
                 sender_id: str,
                 sender_name: str = None,
                 timestamp: Union[float, str, datetime] = None,
                 message_id: str = None,
                 is_dm: bool = False,
                 mentions: List[str] = None,
                 attachments: List[Dict] = None,
                 conversation_id: str = None,
                 metadata: Dict[str, Any] = None):
        
        self.text = text
        self.sender_id = sender_id
        self.sender_name = sender_name or sender_id
        self.timestamp = self._normalize_timestamp(timestamp)
        self.message_id = message_id or str(uuid.uuid4())
        self.is_dm = is_dm
        self.mentions = mentions or []
        self.attachments = attachments or []
        self.conversation_id = conversation_id or "default_conversation"
        self.metadata = metadata or {}

    def _normalize_timestamp(self, timestamp: Union[float, str, datetime, None]) -> float:
        """Normalize various timestamp formats to Unix timestamp float."""
        if timestamp is None:
            return time.time()
        
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
                    logger.warning(f"Could not parse timestamp '{timestamp}', using current time")
                    return time.time()
        
        if isinstance(timestamp, datetime):
            return timestamp.timestamp()
        
        logger.warning(f"Unknown timestamp type {type(timestamp)}, using current time")
        return time.time()

    def to_connectome_event_payload(self, adapter_id: str = "chat_log_importer") -> Dict[str, Any]:
        """Convert to Connectome event payload format."""
        return {
            "event_type": "message_received",
            "payload": {
                "source_adapter_id": adapter_id,
                "external_conversation_id": self.conversation_id,
                "is_dm": self.is_dm,
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
        
        for msg_data in message_list:
            # Support various JSON schemas
            text = msg_data.get("text") or msg_data.get("content") or msg_data.get("message", "")
            sender_id = msg_data.get("sender_id") or msg_data.get("user_id") or msg_data.get("author", "unknown")
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
                metadata=metadata
            )
            messages.append(message)
        
        return messages
    
    @staticmethod
    def parse_csv_messages(csv_path: Path) -> List[ChatMessage]:
        """Parse CSV format chat logs."""
        messages = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Map common CSV column names
                text = row.get("text") or row.get("message") or row.get("content", "")
                sender_id = row.get("sender_id") or row.get("user_id") or row.get("username", "unknown")
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
                    metadata=metadata
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
    
    def __init__(self, space_id: str, adapter_id: str = "chat_log_importer"):
        self.space_id = space_id
        self.adapter_id = adapter_id
        self.timeline_component = None
        self.mock_element = None
        
    async def convert_messages_to_dag(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """Convert chat messages to Connectome DAG format."""
        
        # Create mock element and timeline component
        self.mock_element = MockElement(f"mock_element_{self.space_id}")
        self.mock_element.add_component(TimelineComponent)
        self.timeline_component = self.mock_element.get_component_by_type("TimelineComponent")
        
        if not self.timeline_component:
            raise RuntimeError("Failed to create TimelineComponent")
        
        # Initialize timeline component  
        self.timeline_component.initialize()
        
        logger.info(f"Converting {len(messages)} messages to DAG for space '{self.space_id}'")
        
        # Sort messages by timestamp to ensure proper order
        sorted_messages = sorted(messages, key=lambda m: m.timestamp)
        
        # Add messages as timeline events
        for i, message in enumerate(sorted_messages):
            event_payload = message.to_connectome_event_payload(self.adapter_id)
            
            # Add to primary timeline
            event_id = self.timeline_component.add_event_to_primary_timeline(event_payload)
            
            if event_id:
                logger.debug(f"Added message {i+1}/{len(messages)}: {message.text[:50]}... (ID: {event_id})")
            else:
                logger.error(f"Failed to add message {i+1}: {message.text[:50]}...")
        
        # Extract DAG data in Connectome format
        timeline_state = self._extract_timeline_state()
        timeline_events = self._extract_timeline_events()
        
        return {
            "timeline_state": timeline_state,
            "timeline_events": timeline_events,
            "metadata": {
                "space_id": self.space_id,
                "adapter_id": self.adapter_id,
                "message_count": len(messages),
                "time_range": {
                    "start": min(m.timestamp for m in messages) if messages else None,
                    "end": max(m.timestamp for m in messages) if messages else None
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generator": "chat_log_to_dag_converter"
            }
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

async def save_dag_to_storage_format(dag_data: Dict[str, Any], output_path: Path, space_id: str):
    """Save DAG data in Connectome storage format."""
    
    # Create timeline state file
    state_data = {
        "state_key": f"timeline_state_{space_id}",
        "data": dag_data["timeline_state"],
        "stored_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Create timeline events file  
    events_data = {
        "state_key": f"timeline_events_{space_id}",
        "data": dag_data["timeline_events"],
        "stored_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Save as separate files (matching FileStorage format)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    state_file = output_dir / f"timeline_state_{space_id}.json"
    events_file = output_dir / f"timeline_events_{space_id}.json"
    
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state_data, f, indent=2, ensure_ascii=False)
    
    with open(events_file, 'w', encoding='utf-8') as f:
        json.dump(events_data, f, indent=2, ensure_ascii=False)
    
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

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Convert chat logs to Connectome DAG format")
    parser.add_argument("--input", "-i", required=True, type=Path,
                       help="Input chat log file (JSON or CSV)")
    parser.add_argument("--output", "-o", required=True, type=Path,
                       help="Output DAG file path (JSON)")
    parser.add_argument("--space-id", required=True,
                       help="Space ID for the DAG")
    parser.add_argument("--format", choices=["json", "csv"], default="auto",
                       help="Input format (auto-detect by default)")
    parser.add_argument("--adapter-id", default="chat_log_importer", 
                       help="Adapter ID to use in events")
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
            
            # Convert to DAG format
            converter = ChatLogToDAGConverter(args.space_id, args.adapter_id)
            dag_data = await converter.convert_messages_to_dag(messages)
            
            # Save output
            await save_dag_to_storage_format(dag_data, args.output, args.space_id)
            
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