"""
SQLite storage implementation.

Implements the StorageInterface using SQLite database for structured storage
with better query capabilities and ACID compliance.

Note: This is an example implementation to demonstrate the pluggable architecture.
Install aiosqlite to use this backend: pip install aiosqlite
"""

import json
import aiosqlite
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import logging

from .storage_interface import StorageInterface

logger = logging.getLogger(__name__)


class SQLiteStorage(StorageInterface):
    """
    SQLite-based storage implementation.
    
    Provides structured storage with relational capabilities while maintaining
    the same interface as file-based storage. Offers better concurrent access,
    ACID transactions, and query capabilities.
    """
    
    def __init__(self, storage_config: Dict[str, Any]):
        super().__init__(storage_config)
        
        # Database file path
        self.db_path = Path(storage_config.get('db_path', './storage_data/connectome.db'))
        
        # Connection settings
        self.connection_timeout = storage_config.get('connection_timeout', 30.0)
        self.enable_wal_mode = storage_config.get('enable_wal_mode', True)
        
        # Connection pool (for future optimization)
        self._connection = None
        
        self.logger.info(f"SQLiteStorage initialized with db_path: {self.db_path}")
    
    async def initialize(self) -> bool:
        """Initialize the SQLite database and create tables."""
        try:
            # Create parent directory if it doesn't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self._connection = await aiosqlite.connect(
                str(self.db_path),
                timeout=self.connection_timeout
            )
            
            # Enable WAL mode for better concurrent access
            if self.enable_wal_mode:
                await self._connection.execute("PRAGMA journal_mode=WAL")
            
            # Create tables
            await self._create_tables()
            
            # Create indexes for better performance
            await self._create_indexes()
            
            await self._connection.commit()
            
            self.logger.info("SQLiteStorage initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLiteStorage: {e}", exc_info=True)
            return False
    
    async def shutdown(self) -> bool:
        """Close the database connection."""
        try:
            if self._connection:
                await self._connection.close()
                self._connection = None
            
            self.logger.info("SQLiteStorage shutdown completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during SQLiteStorage shutdown: {e}", exc_info=True)
            return False
    
    async def _create_tables(self) -> None:
        """Create all necessary tables."""
        
        # Conversations table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Raw messages table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS raw_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                message_data TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
            )
        """)
        
        # Memories table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                memory_data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id),
                UNIQUE (conversation_id, memory_id)
            )
        """)
        
        # Reasoning chains table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                chain_data TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Chunk info table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS chunk_info (
                conversation_id TEXT PRIMARY KEY,
                chunk_data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
            )
        """)
        
        # Cache entries table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                cache_data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT
            )
        """)
        
        # System state table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                state_key TEXT PRIMARY KEY,
                state_data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
    
    async def _create_indexes(self) -> None:
        """Create indexes for better query performance."""
        
        # Index on raw_messages for conversation lookups
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_raw_messages_conversation 
            ON raw_messages (conversation_id, message_index)
        """)
        
        # Index on memories for conversation lookups
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_conversation 
            ON memories (conversation_id, created_at)
        """)
        
        # Index on reasoning chains for agent lookups
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_reasoning_chains_agent 
            ON reasoning_chains (agent_id, created_at)
        """)
        
        # Index on cache entries for expiration cleanup
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_entries_expires 
            ON cache_entries (expires_at)
        """)
    
    def _serialize_data(self, data: Any) -> str:
        """Serialize data to JSON string."""
        return json.dumps(data, default=str)
    
    def _deserialize_data(self, json_str: str) -> Any:
        """Deserialize JSON string to data."""
        return json.loads(json_str)
    
    # ===== Conversation Storage =====
    
    async def store_conversation(self, conversation_id: str, data: Dict[str, Any]) -> bool:
        """Store conversation data."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            json_data = self._serialize_data(data)
            
            await self._connection.execute("""
                INSERT OR REPLACE INTO conversations 
                (conversation_id, data, created_at, updated_at)
                VALUES (?, ?, COALESCE((SELECT created_at FROM conversations WHERE conversation_id = ?), ?), ?)
            """, (conversation_id, json_data, conversation_id, now, now))
            
            await self._connection.commit()
            
            self.logger.info(f"Stored conversation {conversation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store conversation {conversation_id}: {e}", exc_info=True)
            return False
    
    async def load_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load conversation data."""
        try:
            cursor = await self._connection.execute("""
                SELECT data FROM conversations WHERE conversation_id = ?
            """, (conversation_id,))
            
            row = await cursor.fetchone()
            if row:
                self.logger.debug(f"Loaded conversation {conversation_id}")
                return self._deserialize_data(row[0])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load conversation {conversation_id}: {e}", exc_info=True)
            return None
    
    async def list_conversations(self) -> List[str]:
        """List all conversation IDs."""
        try:
            cursor = await self._connection.execute("""
                SELECT conversation_id FROM conversations ORDER BY created_at
            """)
            
            rows = await cursor.fetchall()
            return [row[0] for row in rows]
            
        except Exception as e:
            self.logger.error(f"Failed to list conversations: {e}", exc_info=True)
            return []
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all associated data."""
        try:
            # Delete in order due to foreign key constraints
            await self._connection.execute("DELETE FROM raw_messages WHERE conversation_id = ?", (conversation_id,))
            await self._connection.execute("DELETE FROM memories WHERE conversation_id = ?", (conversation_id,))
            await self._connection.execute("DELETE FROM chunk_info WHERE conversation_id = ?", (conversation_id,))
            await self._connection.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
            
            await self._connection.commit()
            
            self.logger.info(f"Deleted conversation {conversation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete conversation {conversation_id}: {e}", exc_info=True)
            return False
    
    # ===== Raw Messages Storage =====
    
    async def store_raw_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> bool:
        """Store raw messages for a conversation."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            # Clear existing messages for this conversation
            await self._connection.execute("DELETE FROM raw_messages WHERE conversation_id = ?", (conversation_id,))
            
            # Insert new messages
            for i, message in enumerate(messages):
                message_data = self._serialize_data(message)
                await self._connection.execute("""
                    INSERT INTO raw_messages (conversation_id, message_data, message_index, created_at)
                    VALUES (?, ?, ?, ?)
                """, (conversation_id, message_data, i, now))
            
            await self._connection.commit()
            
            self.logger.debug(f"Stored {len(messages)} raw messages for conversation {conversation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store raw messages for {conversation_id}: {e}", exc_info=True)
            return False
    
    async def load_raw_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Load raw messages for a conversation."""
        try:
            cursor = await self._connection.execute("""
                SELECT message_data FROM raw_messages 
                WHERE conversation_id = ? 
                ORDER BY message_index
            """, (conversation_id,))
            
            rows = await cursor.fetchall()
            messages = [self._deserialize_data(row[0]) for row in rows]
            
            self.logger.debug(f"Loaded {len(messages)} raw messages for conversation {conversation_id}")
            return messages
            
        except Exception as e:
            self.logger.error(f"Failed to load raw messages for {conversation_id}: {e}", exc_info=True)
            return []
    
    # ===== Memory Storage =====
    
    async def store_memory(self, conversation_id: str, memory_id: str, memory_data: Dict[str, Any]) -> bool:
        """Store a memory entry."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            json_data = self._serialize_data(memory_data)
            
            await self._connection.execute("""
                INSERT OR REPLACE INTO memories (conversation_id, memory_id, memory_data, created_at)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, memory_id, json_data, now))
            
            await self._connection.commit()
            
            self.logger.debug(f"Stored memory {memory_id} for conversation {conversation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store memory {memory_id}: {e}", exc_info=True)
            return False
    
    async def load_memories(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Load all memories for a conversation."""
        try:
            cursor = await self._connection.execute("""
                SELECT memory_id, memory_data, created_at FROM memories 
                WHERE conversation_id = ? 
                ORDER BY created_at
            """, (conversation_id,))
            
            rows = await cursor.fetchall()
            memories = []
            for row in rows:
                memory_data = self._deserialize_data(row[1])
                memory_data['memory_id'] = row[0]
                memory_data['created_at'] = row[2]
                memories.append(memory_data)
            
            self.logger.debug(f"Loaded {len(memories)} memories for conversation {conversation_id}")
            return memories
            
        except Exception as e:
            self.logger.error(f"Failed to load memories for {conversation_id}: {e}", exc_info=True)
            return []
    
    # ===== Reasoning Chain Storage =====
    
    async def store_reasoning_chain(self, agent_id: str, chain_data: Dict[str, Any]) -> bool:
        """Store an agent's reasoning chain."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            json_data = self._serialize_data(chain_data)
            
            await self._connection.execute("""
                INSERT INTO reasoning_chains (agent_id, chain_data, created_at)
                VALUES (?, ?, ?)
            """, (agent_id, json_data, now))
            
            await self._connection.commit()
            
            self.logger.debug(f"Stored reasoning chain for agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store reasoning chain for {agent_id}: {e}", exc_info=True)
            return False
    
    async def load_reasoning_chains(self, agent_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load reasoning chains for an agent."""
        try:
            query = """
                SELECT chain_data, created_at FROM reasoning_chains 
                WHERE agent_id = ? 
                ORDER BY created_at DESC
            """
            params = (agent_id,)
            
            if limit:
                query += " LIMIT ?"
                params = (*params, limit)
            
            cursor = await self._connection.execute(query, params)
            rows = await cursor.fetchall()
            
            chains = []
            for row in rows:
                chain_data = self._deserialize_data(row[0])
                chain_data['stored_at'] = row[1]
                chains.append(chain_data)
            
            # Reverse to get chronological order (oldest first)
            chains.reverse()
            
            self.logger.debug(f"Loaded {len(chains)} reasoning chains for agent {agent_id}")
            return chains
            
        except Exception as e:
            self.logger.error(f"Failed to load reasoning chains for {agent_id}: {e}", exc_info=True)
            return []
    
    # ===== Chunk and Cache Storage =====
    
    async def store_chunk_info(self, conversation_id: str, chunk_data: Dict[str, Any]) -> bool:
        """Store chunking information."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            json_data = self._serialize_data(chunk_data)
            
            await self._connection.execute("""
                INSERT OR REPLACE INTO chunk_info 
                (conversation_id, chunk_data, created_at, updated_at)
                VALUES (?, ?, COALESCE((SELECT created_at FROM chunk_info WHERE conversation_id = ?), ?), ?)
            """, (conversation_id, json_data, conversation_id, now, now))
            
            await self._connection.commit()
            
            self.logger.debug(f"Stored chunk info for conversation {conversation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store chunk info for {conversation_id}: {e}", exc_info=True)
            return False
    
    async def load_chunk_info(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load chunking information."""
        try:
            cursor = await self._connection.execute("""
                SELECT chunk_data FROM chunk_info WHERE conversation_id = ?
            """, (conversation_id,))
            
            row = await cursor.fetchone()
            if row:
                self.logger.debug(f"Loaded chunk info for conversation {conversation_id}")
                return self._deserialize_data(row[0])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load chunk info for {conversation_id}: {e}", exc_info=True)
            return None
    
    async def store_cache_entry(self, cache_key: str, cache_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Store a cache entry with optional TTL."""
        try:
            now = datetime.now(timezone.utc)
            created_at = now.isoformat()
            expires_at = None
            
            if ttl:
                expires_at = (now.timestamp() + ttl)
                expires_at = datetime.fromtimestamp(expires_at, timezone.utc).isoformat()
            
            json_data = self._serialize_data(cache_data)
            
            await self._connection.execute("""
                INSERT OR REPLACE INTO cache_entries (cache_key, cache_data, created_at, expires_at)
                VALUES (?, ?, ?, ?)
            """, (cache_key, json_data, created_at, expires_at))
            
            await self._connection.commit()
            
            self.logger.debug(f"Stored cache entry {cache_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store cache entry {cache_key}: {e}", exc_info=True)
            return False
    
    async def load_cache_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load a cache entry, checking expiration."""
        try:
            cursor = await self._connection.execute("""
                SELECT cache_data, expires_at FROM cache_entries WHERE cache_key = ?
            """, (cache_key,))
            
            row = await cursor.fetchone()
            if not row:
                return None
            
            # Check expiration
            if row[1]:  # expires_at is not None
                expiry_time = datetime.fromisoformat(row[1])
                if datetime.now(timezone.utc) > expiry_time:
                    # Cache expired, remove entry
                    await self._connection.execute("DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,))
                    await self._connection.commit()
                    self.logger.debug(f"Removed expired cache entry {cache_key}")
                    return None
            
            self.logger.debug(f"Loaded cache entry {cache_key}")
            return self._deserialize_data(row[0])
            
        except Exception as e:
            self.logger.error(f"Failed to load cache entry {cache_key}: {e}", exc_info=True)
            return None
    
    # ===== System State Storage =====
    
    async def store_system_state(self, state_key: str, state_data: Dict[str, Any]) -> bool:
        """Store system state information."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            json_data = self._serialize_data(state_data)
            
            await self._connection.execute("""
                INSERT OR REPLACE INTO system_state 
                (state_key, state_data, created_at, updated_at)
                VALUES (?, ?, COALESCE((SELECT created_at FROM system_state WHERE state_key = ?), ?), ?)
            """, (state_key, json_data, state_key, now, now))
            
            await self._connection.commit()
            
            self.logger.debug(f"Stored system state {state_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store system state {state_key}: {e}", exc_info=True)
            return False
    
    async def load_system_state(self, state_key: str) -> Optional[Dict[str, Any]]:
        """Load system state information."""
        try:
            cursor = await self._connection.execute("""
                SELECT state_data FROM system_state WHERE state_key = ?
            """, (state_key,))
            
            row = await cursor.fetchone()
            if row:
                self.logger.debug(f"Loaded system state {state_key}")
                return self._deserialize_data(row[0])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load system state {state_key}: {e}", exc_info=True)
            return None
    
    # ===== Utility Methods =====
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health and status of the SQLite storage."""
        try:
            # Test database connection
            cursor = await self._connection.execute("SELECT 1")
            await cursor.fetchone()
            
            # Get database file size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            # Check WAL mode
            cursor = await self._connection.execute("PRAGMA journal_mode")
            journal_mode = (await cursor.fetchone())[0]
            
            return {
                'status': 'healthy',
                'database_path': str(self.db_path),
                'database_size_bytes': db_size,
                'journal_mode': journal_mode,
                'connection_active': self._connection is not None,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        try:
            stats = {
                'storage_type': 'sqlite',
                'database_path': str(self.db_path),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Count records in each table
            tables = ['conversations', 'raw_messages', 'memories', 'reasoning_chains', 
                     'chunk_info', 'cache_entries', 'system_state']
            
            for table in tables:
                cursor = await self._connection.execute(f"SELECT COUNT(*) FROM {table}")
                count = (await cursor.fetchone())[0]
                stats[f'{table}_count'] = count
            
            # Get database size
            if self.db_path.exists():
                stats['database_size_bytes'] = self.db_path.stat().st_size
            
            # Get database page info
            cursor = await self._connection.execute("PRAGMA page_count")
            page_count = (await cursor.fetchone())[0]
            
            cursor = await self._connection.execute("PRAGMA page_size")
            page_size = (await cursor.fetchone())[0]
            
            stats['page_count'] = page_count
            stats['page_size_bytes'] = page_size
            stats['total_pages_bytes'] = page_count * page_size
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}", exc_info=True)
            return {
                'storage_type': 'sqlite',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            } 