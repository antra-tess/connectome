"""
Storage factory for creating storage backend instances.

Provides a centralized way to create and configure different storage backends
based on configuration parameters.
"""

from typing import Dict, Any, Optional
import logging

from .storage_interface import StorageInterface
from .file_storage import FileStorage

logger = logging.getLogger(__name__)


class StorageFactory:
    """
    Factory class for creating storage backend instances.
    
    Supports pluggable storage backends that can be easily swapped
    based on configuration without changing client code.
    """
    
    # Registry of available storage backends
    _STORAGE_BACKENDS = {
        'file': FileStorage,
    }
    
    @classmethod
    def _initialize_backends(cls):
        """Initialize additional backends if their dependencies are available."""
        # Try to register SQLite backend if aiosqlite is available
        if 'sqlite' not in cls._STORAGE_BACKENDS:
            try:
                import aiosqlite
                from .sqlite_storage import SQLiteStorage
                cls._STORAGE_BACKENDS['sqlite'] = SQLiteStorage
                logger.debug("SQLite storage backend registered (aiosqlite available)")
            except ImportError:
                logger.debug("SQLite storage backend not available (aiosqlite not installed)")
        
        # Future backends can be added here:
        # try:
        #     import redis
        #     from .redis_storage import RedisStorage
        #     cls._STORAGE_BACKENDS['redis'] = RedisStorage
        # except ImportError:
        #     pass
    
    @classmethod
    def create_storage(cls, storage_type: str, storage_config: Dict[str, Any]) -> StorageInterface:
        """
        Create a storage backend instance.
        
        Args:
            storage_type: Type of storage backend ('file', 'sqlite', etc.)
            storage_config: Configuration dictionary for the storage backend
            
        Returns:
            Storage backend instance
            
        Raises:
            ValueError: If storage_type is not supported
            Exception: If storage backend initialization fails
        """
        # Initialize backends lazily
        cls._initialize_backends()
        
        if storage_type not in cls._STORAGE_BACKENDS:
            available_types = ', '.join(cls._STORAGE_BACKENDS.keys())
            raise ValueError(f"Unsupported storage type '{storage_type}'. Available types: {available_types}")
        
        storage_class = cls._STORAGE_BACKENDS[storage_type]
        
        try:
            logger.info(f"Creating {storage_type} storage backend")
            storage_instance = storage_class(storage_config)
            logger.info(f"Successfully created {storage_type} storage backend")
            return storage_instance
            
        except Exception as e:
            logger.error(f"Failed to create {storage_type} storage backend: {e}", exc_info=True)
            raise
    
    @classmethod
    def get_available_backends(cls) -> Dict[str, str]:
        """
        Get information about available storage backends.
        
        Returns:
            Dictionary mapping backend names to their class descriptions
        """
        # Initialize backends to get full list
        cls._initialize_backends()
        
        backends = {}
        for name, storage_class in cls._STORAGE_BACKENDS.items():
            backends[name] = storage_class.__doc__.split('\n')[0] if storage_class.__doc__ else "No description"
        
        return backends
    
    @classmethod
    def register_backend(cls, name: str, storage_class: type) -> None:
        """
        Register a new storage backend.
        
        Args:
            name: Name of the storage backend
            storage_class: Class implementing StorageInterface
            
        Raises:
            ValueError: If the class doesn't implement StorageInterface
        """
        if not issubclass(storage_class, StorageInterface):
            raise ValueError(f"Storage class {storage_class.__name__} must implement StorageInterface")
        
        cls._STORAGE_BACKENDS[name] = storage_class
        logger.info(f"Registered storage backend: {name}")
    
    @classmethod
    def create_default_file_storage(cls, base_dir: Optional[str] = None) -> StorageInterface:
        """
        Create a file storage backend with default configuration.
        
        Args:
            base_dir: Base directory for storage (defaults to './storage_data')
            
        Returns:
            FileStorage instance with default configuration
        """
        config = {
            'base_dir': base_dir or './storage_data',
            'create_backups': True,
            'max_backup_files': 5,
            'pretty_print_json': True
        }
        
        return cls.create_storage('file', config)
    
    @classmethod
    def create_default_sqlite_storage(cls, db_path: Optional[str] = None) -> StorageInterface:
        """
        Create a SQLite storage backend with default configuration.
        
        Args:
            db_path: Database file path (defaults to './storage_data/connectome.db')
            
        Returns:
            SQLiteStorage instance with default configuration
            
        Raises:
            ValueError: If SQLite backend is not available
        """
        cls._initialize_backends()
        
        if 'sqlite' not in cls._STORAGE_BACKENDS:
            raise ValueError("SQLite storage backend not available. Install aiosqlite: pip install aiosqlite")
        
        config = {
            'db_path': db_path or './storage_data/connectome.db',
            'connection_timeout': 30.0,
            'enable_wal_mode': True
        }
        
        return cls.create_storage('sqlite', config)
    
    @classmethod
    def create_from_env_config(cls) -> StorageInterface:
        """
        Create a storage backend from environment variables.
        
        Environment variables:
        - CONNECTOME_STORAGE_TYPE: Type of storage backend (default: 'file')
        - CONNECTOME_STORAGE_BASE_DIR: Base directory for file storage
        - CONNECTOME_STORAGE_DB_PATH: Database path for SQLite storage
        - CONNECTOME_STORAGE_CREATE_BACKUPS: Whether to create backups (default: 'true')
        - CONNECTOME_STORAGE_PRETTY_JSON: Whether to pretty-print JSON (default: 'true')
        - CONNECTOME_STORAGE_WAL_MODE: Enable WAL mode for SQLite (default: 'true')
        
        Returns:
            Storage backend instance based on environment configuration
        """
        import os
        
        storage_type = os.getenv('CONNECTOME_STORAGE_TYPE', 'file')
        
        if storage_type == 'file':
            config = {
                'base_dir': os.getenv('CONNECTOME_STORAGE_BASE_DIR', './storage_data'),
                'create_backups': os.getenv('CONNECTOME_STORAGE_CREATE_BACKUPS', 'true').lower() == 'true',
                'pretty_print_json': os.getenv('CONNECTOME_STORAGE_PRETTY_JSON', 'true').lower() == 'true',
                'max_backup_files': int(os.getenv('CONNECTOME_STORAGE_MAX_BACKUPS', '5'))
            }
        elif storage_type == 'sqlite':
            config = {
                'db_path': os.getenv('CONNECTOME_STORAGE_DB_PATH', './storage_data/connectome.db'),
                'connection_timeout': float(os.getenv('CONNECTOME_STORAGE_TIMEOUT', '30.0')),
                'enable_wal_mode': os.getenv('CONNECTOME_STORAGE_WAL_MODE', 'true').lower() == 'true'
            }
        else:
            # For future storage backends, add their environment config here
            config = {}
        
        logger.info(f"Creating storage from environment config: type={storage_type}")
        return cls.create_storage(storage_type, config)


# Convenience function for easy storage creation
def create_storage(storage_type: str = 'file', **kwargs) -> StorageInterface:
    """
    Convenience function to create a storage backend.
    
    Args:
        storage_type: Type of storage backend
        **kwargs: Configuration parameters for the storage backend
        
    Returns:
        Storage backend instance
    """
    return StorageFactory.create_storage(storage_type, kwargs) 