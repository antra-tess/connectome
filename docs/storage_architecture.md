# Pluggable Storage Architecture

## Overview

The Connectome memory persistence system features a pluggable storage architecture that allows you to start with simple file-based storage and easily upgrade to more sophisticated solutions like SQLite, Redis, or PostgreSQL as your needs evolve.

## Key Benefits

- **Start Simple**: Begin with file-based storage for easy debugging and inspection
- **Scale Gradually**: Upgrade to SQLite for better concurrency and query capabilities
- **Future-Proof**: Add new storage backends without changing application code
- **Unified Interface**: All storage backends implement the same `StorageInterface`
- **Easy Migration**: Move data between storage backends with minimal effort

## Architecture

### Storage Interface

All storage backends implement the `StorageInterface` abstract base class, which defines methods for:

- **Conversation Storage**: Store/load Typingcloud format conversations with compressed memories
- **Raw Messages**: Store/load uncompressed message history for recompression
- **Memory Storage**: Store/load compressed memory formation sequences
- **Reasoning Chains**: Store/load agent reasoning and tool interaction history
- **Chunk Information**: Store/load chunking decisions and metadata
- **Cache Management**: Store/load LLM generation cache with TTL support
- **System State**: Store/load system configuration and runtime state
- **Utilities**: Health checks, statistics, and administrative functions

### Available Backends

#### File Storage (`file`)

**Best for**: Development, debugging, small deployments

- Stores data in organized JSON files following the documented directory structure
- Human-readable format for easy inspection and debugging
- Automatic backup creation with configurable retention
- No external dependencies

**Directory Structure**:
```
storage_data/
├── conversations/
│   └── {conversation_id}/
│       ├── current.json         # Typingcloud format with memories
│       ├── raw_messages.json    # Uncompressed message history
│       ├── chunks.json         # Chunking decisions
│       └── memories.json       # Memory formation records
├── cache/
│   └── {cache_key}.json        # LLM generation cache
├── agents/
│   └── {agent_id}/
│       └── reasoning_chains.json  # Agent reasoning history
└── system/
    └── {state_key}.json        # System state data
```

#### SQLite Storage (`sqlite`)

**Best for**: Production deployments, concurrent access, complex queries

- Structured relational storage with ACID compliance
- Better concurrent access with WAL mode
- Efficient indexing for fast lookups
- Built-in data integrity and consistency
- Requires `aiosqlite` package

**Features**:
- Foreign key constraints maintain data relationships
- Indexes optimized for common query patterns
- Automatic expiration handling for cache entries
- Transaction support for data consistency

## Usage Examples

### Basic Usage

```python
from storage import create_file_storage, create_sqlite_storage

# Create file storage
storage = create_file_storage('./my_storage')
await storage.initialize()

# Store a conversation
conversation_data = {...}  # Typingcloud format
await storage.store_conversation("conv_001", conversation_data)

# Load the conversation
loaded = await storage.load_conversation("conv_001")

await storage.shutdown()
```

### Using the Factory

```python
from storage import StorageFactory

# Create file storage with custom config
storage = StorageFactory.create_storage('file', {
    'base_dir': './my_storage',
    'create_backups': True,
    'pretty_print_json': True
})

# Create SQLite storage
storage = StorageFactory.create_storage('sqlite', {
    'db_path': './my_database.db',
    'enable_wal_mode': True
})
```

### Environment Configuration

Set environment variables and create storage:

```bash
export CONNECTOME_STORAGE_TYPE=sqlite
export CONNECTOME_STORAGE_DB_PATH=./production.db
export CONNECTOME_STORAGE_WAL_MODE=true
```

```python
from storage import create_storage_from_env

storage = create_storage_from_env()
await storage.initialize()
```

### Data Migration

```python
from storage import create_file_storage, create_sqlite_storage

# Load from file storage
file_storage = create_file_storage('./old_storage')
await file_storage.initialize()

# Create SQLite storage
sqlite_storage = create_sqlite_storage('./new_database.db')
await sqlite_storage.initialize()

# Migrate conversations
conversations = await file_storage.list_conversations()
for conv_id in conversations:
    data = await file_storage.load_conversation(conv_id)
    await sqlite_storage.store_conversation(conv_id, data)
    
    # Migrate associated data
    raw_messages = await file_storage.load_raw_messages(conv_id)
    await sqlite_storage.store_raw_messages(conv_id, raw_messages)
    
    memories = await file_storage.load_memories(conv_id)
    for memory in memories:
        await sqlite_storage.store_memory(conv_id, memory['memory_id'], memory)
```

## Configuration Options

### File Storage Configuration

```python
file_config = {
    'base_dir': './storage_data',           # Storage root directory
    'create_backups': True,                 # Create timestamped backups
    'max_backup_files': 5,                  # Keep 5 most recent backups
    'pretty_print_json': True               # Human-readable JSON formatting
}
```

### SQLite Storage Configuration

```python
sqlite_config = {
    'db_path': './connectome.db',           # Database file path
    'connection_timeout': 30.0,             # Connection timeout in seconds
    'enable_wal_mode': True                 # Enable WAL mode for concurrency
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CONNECTOME_STORAGE_TYPE` | Storage backend type | `file` |
| `CONNECTOME_STORAGE_BASE_DIR` | File storage directory | `./storage_data` |
| `CONNECTOME_STORAGE_DB_PATH` | SQLite database path | `./storage_data/connectome.db` |
| `CONNECTOME_STORAGE_CREATE_BACKUPS` | Enable file backups | `true` |
| `CONNECTOME_STORAGE_PRETTY_JSON` | Pretty-print JSON | `true` |
| `CONNECTOME_STORAGE_WAL_MODE` | Enable SQLite WAL mode | `true` |
| `CONNECTOME_STORAGE_TIMEOUT` | SQLite connection timeout | `30.0` |
| `CONNECTOME_STORAGE_MAX_BACKUPS` | Max backup files to keep | `5` |

## Adding New Storage Backends

To add a new storage backend:

1. **Implement the Interface**: Create a class that inherits from `StorageInterface`
2. **Implement All Methods**: Provide implementations for all abstract methods
3. **Register the Backend**: Add it to the storage factory
4. **Add Dependencies**: Include any required packages in setup

### Example: Redis Storage Backend

```python
from storage import StorageInterface, StorageFactory
import aioredis
import json

class RedisStorage(StorageInterface):
    def __init__(self, storage_config):
        super().__init__(storage_config)
        self.redis_url = storage_config.get('redis_url', 'redis://localhost:6379')
        self.redis = None
    
    async def initialize(self):
        self.redis = await aioredis.from_url(self.redis_url)
        return True
    
    async def store_conversation(self, conversation_id, data):
        key = f"conversation:{conversation_id}"
        await self.redis.set(key, json.dumps(data))
        return True
    
    # ... implement other methods
    
# Register the new backend
StorageFactory.register_backend('redis', RedisStorage)
```

## Performance Considerations

### File Storage

- **Pros**: Simple, debuggable, no dependencies
- **Cons**: Slower for large datasets, limited concurrency
- **Best for**: Development, small deployments, debugging

### SQLite Storage

- **Pros**: Fast queries, ACID compliance, better concurrency
- **Cons**: Single-file limitation, requires dependency
- **Best for**: Production deployments, moderate scale

### Future Backends

- **Redis**: Fast cache-like storage with TTL and pub/sub
- **PostgreSQL**: Full-featured relational database with JSON support
- **MongoDB**: Document storage for flexible schema evolution

## Monitoring and Maintenance

### Health Checks

```python
health = await storage.health_check()
print(f"Storage status: {health['status']}")
```

### Storage Statistics

```python
stats = await storage.get_storage_stats()
print(f"Conversations: {stats['conversation_count']}")
print(f"Total size: {stats.get('total_bytes', 'N/A')} bytes")
```

### Backup and Recovery

For file storage, backups are created automatically. For SQLite:

```bash
# Backup SQLite database
cp connectome.db connectome.db.backup

# Or use SQLite backup command
sqlite3 connectome.db ".backup backup.db"
```

## Integration with Connectome Components

### CompressionEngineComponent Integration

```python
# Update CompressionEngineComponent to use pluggable storage
from storage import create_storage_from_env

class CompressionEngineComponent(Component):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._storage = create_storage_from_env()
    
    async def _on_initialize(self):
        await self._storage.initialize()
        return True
    
    async def store_reasoning_chain(self, chain_data):
        agent_id = self._agent_name or self.id
        await self._storage.store_reasoning_chain(agent_id, chain_data)
    
    async def get_memory_context(self):
        # Load from persistent storage instead of memory
        agent_id = self._agent_name or self.id
        chains = await self._storage.load_reasoning_chains(agent_id)
        # Convert to LLM messages...
```

This pluggable storage architecture provides a solid foundation for memory persistence that can grow with your needs, from simple file-based storage for development to sophisticated database solutions for production deployments. 