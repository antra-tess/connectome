"""
File-based storage implementation.

Implements the StorageInterface using a file-based approach optimized
for debugging, performance, and maintainability as described in the
storage architecture documentation.
"""

import os
import json
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import logging
import hashlib
import time

from .storage_interface import StorageInterface

logger = logging.getLogger(__name__)


class FileStorage(StorageInterface):
    """
    File-based storage implementation following the directory structure:
    
    conversations/
        {conversation_id}/
            current.json         # Typingcloud format with compressed memories
            raw_messages.json    # Complete history of uncompressed messages
            chunks.json         # Record of chunking decisions
            memories.json       # Record of memory formation
    cache/
        {cache_key}.json        # LLM generation cache and other cached data
    agents/
        {agent_id}/
            reasoning_chains.json  # Agent reasoning chains
    system/
        {state_key}.json        # System state data
    """
    
    def __init__(self, storage_config: Dict[str, Any]):
        super().__init__(storage_config)
        
        # Get base directory from config, default to './storage_data'
        self.base_dir = Path(storage_config.get('base_dir', './storage_data'))
        
        # Define subdirectories
        self.conversations_dir = self.base_dir / 'conversations'
        self.cache_dir = self.base_dir / 'cache'
        self.agents_dir = self.base_dir / 'agents'
        self.system_dir = self.base_dir / 'system'
        
        # Configuration options
        self.create_backups = storage_config.get('create_backups', True)
        self.max_backup_files = storage_config.get('max_backup_files', 5)
        self.pretty_print_json = storage_config.get('pretty_print_json', True)
        
        self.logger.info(f"FileStorage initialized with base_dir: {self.base_dir}")
    
    async def initialize(self) -> bool:
        """Create necessary directories and verify write permissions."""
        try:
            # Create all required directories
            directories = [
                self.conversations_dir,
                self.cache_dir,
                self.agents_dir,
                self.system_dir
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created/verified directory: {directory}")
            
            # Test write permissions by creating a test file
            test_file = self.base_dir / '.storage_test'
            async with aiofiles.open(test_file, 'w') as f:
                await f.write('{"test": true}')
            
            # Clean up test file
            if test_file.exists():
                test_file.unlink()
            
            self.logger.info("FileStorage initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FileStorage: {e}", exc_info=True)
            return False
    
    async def shutdown(self) -> bool:
        """Graceful shutdown - no cleanup needed for file storage."""
        self.logger.info("FileStorage shutdown completed")
        return True
    
    # ===== Helper Methods =====
    
    def _get_conversation_dir(self, conversation_id: str) -> Path:
        """Get the directory path for a conversation."""
        return self.conversations_dir / conversation_id
    
    def _get_agent_dir(self, agent_id: str) -> Path:
        """Get the directory path for an agent."""
        return self.agents_dir / agent_id
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize a string to be safe as a filename."""
        # Replace problematic characters with underscores
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        return ''.join(c if c in safe_chars else '_' for c in filename)
    
    async def _write_json_file(self, file_path: Path, data: Dict[str, Any], create_backup: bool = True) -> bool:
        """
        Write JSON data to a file with optional backup creation.
        
        Args:
            file_path: Path to the file
            data: Data to write
            create_backup: Whether to create a backup of existing file
            
        Returns:
            True if write succeeded, False otherwise
        """
        try:
            # Create parent directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if file exists and backup is enabled
            if create_backup and self.create_backups and file_path.exists():
                await self._create_backup(file_path)
            
            # Write the file
            json_str = json.dumps(data, indent=2 if self.pretty_print_json else None, default=str)
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json_str)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write JSON file {file_path}: {e}", exc_info=True)
            return False
    
    async def _read_json_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Read JSON data from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Parsed JSON data or None if file doesn't exist or can't be read
        """
        try:
            if not file_path.exists():
                return None
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                try:
                    return json.loads(content)
                except Exception as e:
                    self.logger.error(f"Failed to parse JSON file {file_path}: {e}", exc_info=True)
                    return None
        except Exception as e:
            self.logger.error(f"Failed to read JSON file {file_path}: {e}", exc_info=True)
            return None
    
    async def _create_backup(self, file_path: Path) -> bool:
        """Create a backup of an existing file with timestamp."""
        try:
            if not file_path.exists():
                return True
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_suffix(f'.{timestamp}.backup{file_path.suffix}')
            
            # Copy the file
            async with aiofiles.open(file_path, 'rb') as src:
                content = await src.read()
            async with aiofiles.open(backup_path, 'wb') as dst:
                await dst.write(content)
            
            # Clean up old backups
            await self._cleanup_old_backups(file_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup for {file_path}: {e}", exc_info=True)
            return False
    
    async def _cleanup_old_backups(self, original_file: Path) -> None:
        """Remove old backup files, keeping only the most recent ones."""
        try:
            backup_pattern = f"{original_file.stem}.*.backup{original_file.suffix}"
            backup_files = list(original_file.parent.glob(backup_pattern))
            
            if len(backup_files) > self.max_backup_files:
                # Sort by modification time (oldest first)
                backup_files.sort(key=lambda f: f.stat().st_mtime)
                
                # Remove oldest files
                files_to_remove = backup_files[:-self.max_backup_files]
                for backup_file in files_to_remove:
                    backup_file.unlink()
                    self.logger.debug(f"Removed old backup: {backup_file}")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup old backups: {e}", exc_info=True)
    
    # ===== Conversation Storage =====
    
    async def store_conversation(self, conversation_id: str, data: Dict[str, Any]) -> bool:
        """Store conversation data in current.json."""
        conv_dir = self._get_conversation_dir(conversation_id)
        current_file = conv_dir / 'current.json'
        
        # Add metadata
        enhanced_data = {
            **data,
            '_storage_metadata': {
                'stored_at': datetime.now(timezone.utc).isoformat(),
                'storage_backend': 'file',
                'conversation_id': conversation_id
            }
        }
        
        success = await self._write_json_file(current_file, enhanced_data)
        if success:
            self.logger.info(f"Stored conversation {conversation_id}")
        return success
    
    async def load_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load conversation data from current.json."""
        conv_dir = self._get_conversation_dir(conversation_id)
        current_file = conv_dir / 'current.json'
        
        data = await self._read_json_file(current_file)
        if data:
            self.logger.debug(f"Loaded conversation {conversation_id}")
        return data
    
    async def list_conversations(self) -> List[str]:
        """List all conversation directories."""
        try:
            if not self.conversations_dir.exists():
                return []
            
            conversations = []
            for item in self.conversations_dir.iterdir():
                if item.is_dir():
                    # Verify it has a current.json file
                    current_file = item / 'current.json'
                    if current_file.exists():
                        conversations.append(item.name)
            
            return sorted(conversations)
            
        except Exception as e:
            self.logger.error(f"Failed to list conversations: {e}", exc_info=True)
            return []
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation directory and all its contents."""
        try:
            conv_dir = self._get_conversation_dir(conversation_id)
            if not conv_dir.exists():
                return True  # Already deleted
            
            # Remove all files in the directory
            for file_path in conv_dir.rglob('*'):
                if file_path.is_file():
                    file_path.unlink()
            
            # Remove the directory
            conv_dir.rmdir()
            
            self.logger.info(f"Deleted conversation {conversation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete conversation {conversation_id}: {e}", exc_info=True)
            return False
    
    # ===== Raw Messages Storage =====
    
    async def store_raw_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> bool:
        """Store raw messages in raw_messages.json."""
        conv_dir = self._get_conversation_dir(conversation_id)
        raw_file = conv_dir / 'raw_messages.json'
        
        data = {
            'messages': messages,
            '_metadata': {
                'stored_at': datetime.now(timezone.utc).isoformat(),
                'message_count': len(messages),
                'conversation_id': conversation_id
            }
        }
        
        success = await self._write_json_file(raw_file, data)
        if success:
            self.logger.debug(f"Stored {len(messages)} raw messages for conversation {conversation_id}")
        return success
    
    async def load_raw_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Load raw messages from raw_messages.json."""
        conv_dir = self._get_conversation_dir(conversation_id)
        raw_file = conv_dir / 'raw_messages.json'
        
        data = await self._read_json_file(raw_file)
        if data and 'messages' in data:
            self.logger.debug(f"Loaded {len(data['messages'])} raw messages for conversation {conversation_id}")
            return data['messages']
        
        return []
    
    # ===== Memory Storage =====
    
    async def store_memory(self, conversation_id: str, memory_id: str, memory_data: Dict[str, Any]) -> bool:
        """Store a memory entry in memories.json."""
        conv_dir = self._get_conversation_dir(conversation_id)
        memories_file = conv_dir / 'memories.json'
        
        # Load existing memories
        existing_data = await self._read_json_file(memories_file) or {'memories': [], '_metadata': {}}
        
        # Add new memory
        memory_entry = {
            'memory_id': memory_id,
            'created_at': datetime.now(timezone.utc).isoformat(),
            **memory_data
        }
        
        existing_data['memories'].append(memory_entry)
        existing_data['_metadata'].update({
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'memory_count': len(existing_data['memories']),
            'conversation_id': conversation_id
        })
        
        success = await self._write_json_file(memories_file, existing_data)
        if success:
            self.logger.debug(f"Stored memory {memory_id} for conversation {conversation_id}")
        return success
    
    async def load_memories(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Load all memories from memories.json."""
        conv_dir = self._get_conversation_dir(conversation_id)
        memories_file = conv_dir / 'memories.json'
        
        data = await self._read_json_file(memories_file)
        if data and 'memories' in data:
            memories = data['memories']
            self.logger.debug(f"Loaded {len(memories)} memories for conversation {conversation_id}")
            return memories
        
        return []
    
    # ===== Reasoning Chain Storage =====
    
    async def store_reasoning_chain(self, agent_id: str, chain_data: Dict[str, Any]) -> bool:
        """Store an agent's reasoning chain."""
        agent_dir = self._get_agent_dir(agent_id)
        chains_file = agent_dir / 'reasoning_chains.json'
        
        # Load existing chains
        existing_data = await self._read_json_file(chains_file) or {'chains': [], '_metadata': {}}
        
        # Add new reasoning chain
        chain_entry = {
            'stored_at': datetime.now(timezone.utc).isoformat(),
            **chain_data
        }
        
        existing_data['chains'].append(chain_entry)
        existing_data['_metadata'].update({
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'chain_count': len(existing_data['chains']),
            'agent_id': agent_id
        })
        
        success = await self._write_json_file(chains_file, existing_data)
        if success:
            self.logger.debug(f"Stored reasoning chain for agent {agent_id}")
        return success
    
    async def load_reasoning_chains(self, agent_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load reasoning chains for an agent."""
        agent_dir = self._get_agent_dir(agent_id)
        chains_file = agent_dir / 'reasoning_chains.json'
        
        data = await self._read_json_file(chains_file)
        if data and 'chains' in data:
            chains = data['chains']
            if limit:
                chains = chains[-limit:]  # Get most recent chains
            
            self.logger.debug(f"Loaded {len(chains)} reasoning chains for agent {agent_id}")
            return chains
        
        return []
    
    # ===== Chunk and Cache Storage =====
    
    async def store_chunk_info(self, conversation_id: str, chunk_data: Dict[str, Any]) -> bool:
        """Store chunking information in chunks.json."""
        conv_dir = self._get_conversation_dir(conversation_id)
        chunks_file = conv_dir / 'chunks.json'
        
        enhanced_data = {
            **chunk_data,
            '_metadata': {
                'stored_at': datetime.now(timezone.utc).isoformat(),
                'conversation_id': conversation_id
            }
        }
        
        success = await self._write_json_file(chunks_file, enhanced_data)
        if success:
            self.logger.debug(f"Stored chunk info for conversation {conversation_id}")
        return success
    
    async def load_chunk_info(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load chunking information from chunks.json."""
        conv_dir = self._get_conversation_dir(conversation_id)
        chunks_file = conv_dir / 'chunks.json'
        
        data = await self._read_json_file(chunks_file)
        if data:
            self.logger.debug(f"Loaded chunk info for conversation {conversation_id}")
        return data
    
    async def store_cache_entry(self, cache_key: str, cache_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Store a cache entry with optional TTL."""
        cache_file = self.cache_dir / f"{self._sanitize_filename(cache_key)}.json"
        
        entry_data = {
            'cache_key': cache_key,
            'data': cache_data,
            'stored_at': datetime.now(timezone.utc).isoformat(),
            'expires_at': None
        }
        
        if ttl:
            expiry_time = datetime.now(timezone.utc).timestamp() + ttl
            entry_data['expires_at'] = datetime.fromtimestamp(expiry_time, timezone.utc).isoformat()
        
        success = await self._write_json_file(cache_file, entry_data, create_backup=False)
        if success:
            self.logger.debug(f"Stored cache entry {cache_key}")
        return success
    
    async def load_cache_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load a cache entry, checking expiration."""
        cache_file = self.cache_dir / f"{self._sanitize_filename(cache_key)}.json"
        
        data = await self._read_json_file(cache_file)
        if not data:
            return None
        
        # Check expiration
        if data.get('expires_at'):
            expiry_time = datetime.fromisoformat(data['expires_at'])
            if datetime.now(timezone.utc) > expiry_time:
                # Cache expired, remove file
                try:
                    cache_file.unlink()
                    self.logger.debug(f"Removed expired cache entry {cache_key}")
                except Exception:
                    pass
                return None
        
        self.logger.debug(f"Loaded cache entry {cache_key}")
        return data.get('data')
    
    # ===== System State Storage =====
    
    async def store_system_state(self, state_key: str, state_data: Dict[str, Any]) -> bool:
        """Store system state information."""
        state_file = self.system_dir / f"{self._sanitize_filename(state_key)}.json"
        
        enhanced_data = {
            'state_key': state_key,
            'data': state_data,
            'stored_at': datetime.now(timezone.utc).isoformat()
        }
        
        success = await self._write_json_file(state_file, enhanced_data)
        if success:
            self.logger.debug(f"Stored system state {state_key}")
        return success
    
    async def load_system_state(self, state_key: str) -> Optional[Dict[str, Any]]:
        """Load system state information."""
        state_file = self.system_dir / f"{self._sanitize_filename(state_key)}.json"
        
        data = await self._read_json_file(state_file)
        if data and 'data' in data:
            self.logger.debug(f"Loaded system state {state_key}")
            return data['data']
        
        return None
    
    # ===== Utility Methods =====
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health and status of the file storage."""
        try:
            # Check directory accessibility
            directories_ok = all(
                d.exists() and d.is_dir()
                for d in [self.conversations_dir, self.cache_dir, self.agents_dir, self.system_dir]
            )
            
            # Check write permissions
            test_file = self.base_dir / '.health_check'
            write_ok = False
            try:
                async with aiofiles.open(test_file, 'w') as f:
                    await f.write('test')
                test_file.unlink()
                write_ok = True
            except Exception:
                pass
            
            # Get disk usage
            disk_usage = None
            try:
                import shutil
                total, used, free = shutil.disk_usage(self.base_dir)
                disk_usage = {
                    'total_bytes': total,
                    'used_bytes': used,
                    'free_bytes': free,
                    'free_percent': (free / total) * 100
                }
            except Exception:
                pass
            
            return {
                'status': 'healthy' if directories_ok and write_ok else 'unhealthy',
                'directories_accessible': directories_ok,
                'write_permissions': write_ok,
                'base_directory': str(self.base_dir),
                'disk_usage': disk_usage,
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
                'storage_type': 'file',
                'base_directory': str(self.base_dir),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Count conversations
            conversation_count = len(await self.list_conversations())
            stats['conversation_count'] = conversation_count
            
            # Count cache files
            cache_files = list(self.cache_dir.glob('*.json')) if self.cache_dir.exists() else []
            stats['cache_entries'] = len(cache_files)
            
            # Count agent directories
            agent_dirs = list(self.agents_dir.iterdir()) if self.agents_dir.exists() else []
            agent_dirs = [d for d in agent_dirs if d.is_dir()]
            stats['agent_count'] = len(agent_dirs)
            
            # Count system state files
            system_files = list(self.system_dir.glob('*.json')) if self.system_dir.exists() else []
            stats['system_state_count'] = len(system_files)
            
            # Calculate directory sizes
            def get_dir_size(directory: Path) -> int:
                if not directory.exists():
                    return 0
                return sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
            
            stats['directory_sizes'] = {
                'conversations_bytes': get_dir_size(self.conversations_dir),
                'cache_bytes': get_dir_size(self.cache_dir),
                'agents_bytes': get_dir_size(self.agents_dir),
                'system_bytes': get_dir_size(self.system_dir),
                'total_bytes': get_dir_size(self.base_dir)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}", exc_info=True)
            return {
                'storage_type': 'file',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            } 