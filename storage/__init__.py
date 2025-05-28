"""
Storage module for Connectome memory persistence.

Provides pluggable storage backends starting with file-based storage,
with the ability to upgrade to SQLite, Redis, or other solutions.
"""

from .storage_interface import StorageInterface
from .file_storage import FileStorage
from .storage_factory import StorageFactory, create_storage

# Conditionally import SQLite storage if available
try:
    import aiosqlite
    from .sqlite_storage import SQLiteStorage
    _SQLITE_AVAILABLE = True
except ImportError:
    SQLiteStorage = None
    _SQLITE_AVAILABLE = False

__all__ = [
    'StorageInterface', 
    'FileStorage', 
    'StorageFactory',
    'create_storage'
]

# Add SQLite to exports if available
if _SQLITE_AVAILABLE:
    __all__.append('SQLiteStorage')


def get_available_backends():
    """Get list of available storage backends."""
    return StorageFactory.get_available_backends()


def create_file_storage(base_dir: str = './storage_data', **kwargs):
    """Create a file storage backend with custom configuration."""
    return StorageFactory.create_default_file_storage(base_dir)


def create_sqlite_storage(db_path: str = './storage_data/connectome.db', **kwargs):
    """Create a SQLite storage backend with custom configuration."""
    if not _SQLITE_AVAILABLE:
        raise ImportError("SQLite storage requires aiosqlite. Install it with: pip install aiosqlite")
    return StorageFactory.create_default_sqlite_storage(db_path)


def create_storage_from_env():
    """Create storage backend from environment variables."""
    return StorageFactory.create_from_env_config() 