"""
Protocol Persistence Layer

Provides persistent storage for protocol state to survive restarts.
Supports both file-based (simple) and database (scalable) backends.
"""

import json
import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import logging

from .fix_protocol import ProtocolMessage, ProtocolState


class ProtocolPersistence(ABC):
    """Abstract base for protocol persistence"""
    
    @abstractmethod
    async def save_state(self, node_id: str, state: Dict[str, Any]) -> None:
        """Save protocol state"""
        pass
    
    @abstractmethod
    async def load_state(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Load protocol state"""
        pass
    
    @abstractmethod
    async def save_message(self, node_id: str, sequence: int, message: ProtocolMessage) -> None:
        """Save a message for retransmission"""
        pass
    
    @abstractmethod
    async def load_messages(self, node_id: str, from_seq: int = 0) -> Dict[int, ProtocolMessage]:
        """Load messages for retransmission"""
        pass
    
    @abstractmethod
    async def delete_messages_before(self, node_id: str, sequence: int) -> None:
        """Delete old messages that are no longer needed"""
        pass


class FilePersistence(ProtocolPersistence):
    """Simple file-based persistence for development/small deployments"""
    
    def __init__(self, base_dir: str = "./protocol_state"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.logger = logging.getLogger("FilePersistence")
        
    def _get_state_path(self, node_id: str) -> str:
        return os.path.join(self.base_dir, f"{node_id}_state.json")
    
    def _get_messages_path(self, node_id: str) -> str:
        return os.path.join(self.base_dir, f"{node_id}_messages.json")
    
    async def save_state(self, node_id: str, state: Dict[str, Any]) -> None:
        """Save protocol state to file"""
        try:
            state_path = self._get_state_path(node_id)
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            self.logger.debug(f"Saved state for {node_id}")
        except Exception as e:
            self.logger.error(f"Failed to save state for {node_id}: {e}")
            
    async def load_state(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Load protocol state from file"""
        try:
            state_path = self._get_state_path(node_id)
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load state for {node_id}: {e}")
        return None
        
    async def save_message(self, node_id: str, sequence: int, message: ProtocolMessage) -> None:
        """Save a message to file"""
        try:
            messages_path = self._get_messages_path(node_id)
            
            # Load existing messages
            messages = {}
            if os.path.exists(messages_path):
                with open(messages_path, 'r') as f:
                    messages = json.load(f)
            
            self.logger.debug(f"Loaded {len(messages)} existing messages for {node_id}")
            
            # Add new message
            messages[str(sequence)] = message.to_dict()
            
            self.logger.info(f"Saving message seq={sequence} for {node_id}, total messages: {len(messages)}")
            
            # Save back
            with open(messages_path, 'w') as f:
                json.dump(messages, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save message {sequence} for {node_id}: {e}")
            
    async def load_messages(self, node_id: str, from_seq: int = 0) -> Dict[int, ProtocolMessage]:
        """Load messages from file"""
        try:
            messages_path = self._get_messages_path(node_id)
            if os.path.exists(messages_path):
                with open(messages_path, 'r') as f:
                    data = json.load(f)
                    
                result = {}
                for seq_str, msg_dict in data.items():
                    seq = int(seq_str)
                    if seq >= from_seq:
                        result[seq] = ProtocolMessage.from_dict(msg_dict)
                return result
        except Exception as e:
            self.logger.error(f"Failed to load messages for {node_id}: {e}")
        return {}
        
    async def delete_messages_before(self, node_id: str, sequence: int) -> None:
        """Delete old messages from file"""
        try:
            messages_path = self._get_messages_path(node_id)
            if os.path.exists(messages_path):
                with open(messages_path, 'r') as f:
                    messages = json.load(f)
                
                # Keep only messages >= sequence
                filtered = {seq: msg for seq, msg in messages.items() 
                           if int(seq) >= sequence}
                
                with open(messages_path, 'w') as f:
                    json.dump(filtered, f, indent=2)
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup messages for {node_id}: {e}")


class SQLitePersistence(ProtocolPersistence):
    """SQLite-based persistence for production use"""
    
    def __init__(self, db_path: str = "./protocol_state.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("SQLitePersistence")
        self._init_db()
        
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS protocol_state (
                    node_id TEXT PRIMARY KEY,
                    outbound_sequence INTEGER,
                    peer_states TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS protocol_messages (
                    node_id TEXT,
                    sequence INTEGER,
                    message TEXT,
                    timestamp REAL,
                    PRIMARY KEY (node_id, sequence)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                ON protocol_messages(timestamp)
            """)
            
    async def save_state(self, node_id: str, state: Dict[str, Any]) -> None:
        """Save protocol state to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO protocol_state 
                    (node_id, outbound_sequence, peer_states, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    node_id,
                    state.get('outbound_sequence', 0),
                    json.dumps(state.get('peer_states', {}))
                ))
        except Exception as e:
            self.logger.error(f"Failed to save state for {node_id}: {e}")
            
    async def load_state(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Load protocol state from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT outbound_sequence, peer_states 
                    FROM protocol_state 
                    WHERE node_id = ?
                """, (node_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'outbound_sequence': row[0],
                        'peer_states': json.loads(row[1])
                    }
        except Exception as e:
            self.logger.error(f"Failed to load state for {node_id}: {e}")
        return None
        
    async def save_message(self, node_id: str, sequence: int, message: ProtocolMessage) -> None:
        """Save a message to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO protocol_messages 
                    (node_id, sequence, message, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (
                    node_id,
                    sequence,
                    json.dumps(message.to_dict()),
                    message.timestamp
                ))
        except Exception as e:
            self.logger.error(f"Failed to save message {sequence} for {node_id}: {e}")
            
    async def load_messages(self, node_id: str, from_seq: int = 0) -> Dict[int, ProtocolMessage]:
        """Load messages from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT sequence, message 
                    FROM protocol_messages 
                    WHERE node_id = ? AND sequence >= ?
                    ORDER BY sequence
                """, (node_id, from_seq))
                
                result = {}
                for row in cursor:
                    sequence, message_json = row
                    result[sequence] = ProtocolMessage.from_dict(json.loads(message_json))
                return result
        except Exception as e:
            self.logger.error(f"Failed to load messages for {node_id}: {e}")
        return {}
        
    async def delete_messages_before(self, node_id: str, sequence: int) -> None:
        """Delete old messages from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    DELETE FROM protocol_messages 
                    WHERE node_id = ? AND sequence < ?
                """, (node_id, sequence))
        except Exception as e:
            self.logger.error(f"Failed to cleanup messages for {node_id}: {e}")


class PersistentFIXProtocol:
    """
    Enhanced FIX Protocol with persistence support.
    
    This wraps the original FIXProtocol and adds persistence for:
    - Sequence numbers
    - Peer states
    - Message storage
    """
    
    def __init__(self, protocol, persistence: ProtocolPersistence):
        self.protocol = protocol
        self.persistence = persistence
        self.logger = logging.getLogger(f"PersistentProtocol.{protocol.node_id}")
        
    async def initialize(self):
        """Load persisted state on startup"""
        state = await self.persistence.load_state(self.protocol.node_id)
        
        if state:
            self.logger.info(f"Loaded persisted state: outbound_seq={state['outbound_sequence']}")
            
            # Restore sequence
            self.protocol.outbound_sequence = state['outbound_sequence']
            
            # Restore peer states
            for peer_id, peer_state_dict in state['peer_states'].items():
                peer_state = ProtocolState(
                    sequence=peer_state_dict.get('sequence', 0),
                    last_received=peer_state_dict.get('last_received', 0)
                )
                self.protocol.peer_states[peer_id] = peer_state
                
            # Restore messages
            messages = await self.persistence.load_messages(self.protocol.node_id)
            for seq, message in messages.items():
                self.protocol.message_storage[seq] = message
                
            self.logger.info(f"Restored {len(messages)} messages for retransmission")
        else:
            self.logger.info("No persisted state found, starting fresh")
            
    async def save_state(self):
        """Persist current state"""
        # Convert peer states to dict
        peer_states_dict = {}
        for peer_id, peer_state in self.protocol.peer_states.items():
            peer_states_dict[peer_id] = {
                'sequence': peer_state.sequence,
                'last_received': peer_state.last_received
            }
            
        state = {
            'outbound_sequence': self.protocol.outbound_sequence,
            'peer_states': peer_states_dict
        }
        
        self.logger.info(f"Saving state: outbound_seq={self.protocol.outbound_sequence}, peers={list(peer_states_dict.keys())}")
        await self.persistence.save_state(self.protocol.node_id, state)
        
    async def send_message(self, *args, **kwargs) -> int:
        """Send message and persist"""
        sequence = await self.protocol.send_message(*args, **kwargs)
        
        # Persist the message
        message = self.protocol.message_storage[sequence]
        await self.persistence.save_message(self.protocol.node_id, sequence, message)
        
        # Update state
        await self.save_state()
        
        return sequence
        
    async def handle_incoming_message(self, peer_id: str, data: Dict[str, Any]) -> None:
        """Handle incoming message and update persistent state"""
        await self.protocol.handle_incoming_message(peer_id, data)
        
        # Save updated state after processing
        await self.save_state()
        
    # Delegate other methods to the wrapped protocol
    def __getattr__(self, name):
        return getattr(self.protocol, name) 