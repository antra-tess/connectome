"""
ConnectomeEpoch Temporal System

Manages global temporal reference for all VEIL facets within the connectome system.
Provides veil_timestamp generation for endotemporal ordering.
"""

import logging
import time
from typing import Optional
from threading import Lock

logger = logging.getLogger(__name__)

class ConnectomeEpoch:
    """
    Manages global temporal reference for VEIL facets.
    
    Provides consistent veil_timestamp generation across the entire connectome system
    for endotemporal ordering. All VEIL facets use timestamps relative to this epoch.
    
    Key Features:
    - Single system-wide temporal reference
    - Thread-safe timestamp generation
    - Encounter-time semantics (when agent encountered information)
    - Monotonic timestamp ordering
    """
    
    _system_start_time: Optional[float] = None
    _initialization_lock = Lock()
    _last_timestamp: float = 0.0
    _timestamp_lock = Lock()
    
    @classmethod
    def initialize(cls) -> None:
        """
        Initialize system epoch on first call.
        Thread-safe initialization ensures consistent system start time.
        """
        if cls._system_start_time is None:
            with cls._initialization_lock:
                # Double-check pattern to prevent race conditions
                if cls._system_start_time is None:
                    cls._system_start_time = time.time()
                    logger.info(f"ConnectomeEpoch initialized at system time {cls._system_start_time}")
                    
    @classmethod
    def get_veil_timestamp(cls) -> float:
        """
        Get current veil_timestamp since system epoch.
        
        Returns monotonically increasing timestamps to ensure proper
        chronological ordering even with rapid successive calls.
        
        Returns:
            veil_timestamp as float (seconds since system epoch)
        """
        if cls._system_start_time is None:
            cls.initialize()
        
        with cls._timestamp_lock:
            current_time = time.time()
            epoch_time = current_time - cls._system_start_time
            
            # Ensure monotonic ordering - prevent identical timestamps
            if epoch_time <= cls._last_timestamp:
                epoch_time = cls._last_timestamp + 0.000001  # 1 microsecond increment
            
            cls._last_timestamp = epoch_time
            return epoch_time
        
    @classmethod
    def get_epoch_start(cls) -> float:
        """
        Get system epoch start time.
        
        Returns:
            Absolute timestamp when the connectome system epoch began
        """
        if cls._system_start_time is None:
            cls.initialize()
        return cls._system_start_time
    
    @classmethod
    def veil_timestamp_to_absolute(cls, veil_timestamp: float) -> float:
        """
        Convert veil_timestamp to absolute system time.
        
        Args:
            veil_timestamp: Timestamp relative to connectome epoch
            
        Returns:
            Absolute system timestamp
        """
        if cls._system_start_time is None:
            cls.initialize()
        return cls._system_start_time + veil_timestamp
    
    @classmethod
    def absolute_to_veil_timestamp(cls, absolute_time: float) -> float:
        """
        Convert absolute system time to veil_timestamp.
        
        Args:
            absolute_time: Absolute system timestamp
            
        Returns:
            veil_timestamp relative to connectome epoch
        """
        if cls._system_start_time is None:
            cls.initialize()
        return absolute_time - cls._system_start_time
    
    @classmethod
    def reset_epoch(cls) -> None:
        """
        Reset the epoch (primarily for testing).
        
        WARNING: This should only be used in test environments.
        Resetting during normal operation will break temporal consistency.
        """
        with cls._initialization_lock:
            with cls._timestamp_lock:
                cls._system_start_time = None
                cls._last_timestamp = 0.0
                logger.warning("ConnectomeEpoch reset - temporal consistency may be affected")
    
    @classmethod
    def get_current_status(cls) -> dict:
        """
        Get current epoch status for debugging.
        
        Returns:
            Dictionary with epoch status information
        """
        return {
            "initialized": cls._system_start_time is not None,
            "epoch_start": cls._system_start_time,
            "last_timestamp": cls._last_timestamp,
            "current_veil_timestamp": cls.get_veil_timestamp() if cls._system_start_time else None
        } 