#!/usr/bin/env python3
"""
PID File Support for Connectome

This module provides utilities for creating and managing PID files
to help the shutdown command reliably track Connectome processes.
"""

import os
import sys
import atexit
import signal
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ConnectomePIDManager:
    """Manages PID files for Connectome processes."""
    
    def __init__(self, storage_dir: str = "./storage_data", process_name: str = "main"):
        self.storage_dir = Path(storage_dir)
        self.pid_dir = self.storage_dir / "pids"
        self.process_name = process_name
        self.pid_file_path = self.pid_dir / f"{process_name}.pid"
        self._pid_created = False
        
        # Ensure PID directory exists
        self.pid_dir.mkdir(parents=True, exist_ok=True)
    
    def create_pid_file(self) -> bool:
        """Create a PID file for the current process."""
        try:
            pid = os.getpid()
            
            # Check if PID file already exists
            if self.pid_file_path.exists():
                try:
                    with open(self.pid_file_path, 'r') as f:
                        existing_pid = int(f.read().strip())
                    
                    # Check if the existing process is still running
                    if self._is_process_running(existing_pid):
                        logger.warning(f"PID file {self.pid_file_path} already exists for running process {existing_pid}")
                        return False
                    else:
                        logger.info(f"Removing stale PID file for process {existing_pid}")
                        self.pid_file_path.unlink()
                except (ValueError, OSError):
                    # Invalid or unreadable PID file, remove it
                    self.pid_file_path.unlink()
            
            # Create new PID file
            with open(self.pid_file_path, 'w') as f:
                f.write(str(pid))
            
            self._pid_created = True
            logger.info(f"Created PID file: {self.pid_file_path} (PID: {pid})")
            
            # Register cleanup handlers
            atexit.register(self.cleanup_pid_file)
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create PID file: {e}")
            return False
    
    def cleanup_pid_file(self):
        """Remove the PID file."""
        if self._pid_created and self.pid_file_path.exists():
            try:
                self.pid_file_path.unlink()
                logger.info(f"Cleaned up PID file: {self.pid_file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup PID file: {e}")
            finally:
                self._pid_created = False
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, cleaning up...")
        self.cleanup_pid_file()
        # Re-raise the signal to allow normal shutdown
        if signum == signal.SIGTERM:
            sys.exit(0)
        elif signum == signal.SIGINT:
            sys.exit(1)
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is running."""
        try:
            os.kill(pid, 0)  # Send null signal to test if process exists
            return True
        except OSError:
            return False
    
    @classmethod
    def get_process_name_from_args(cls, argv: Optional[list] = None) -> str:
        """Generate a process name based on command line arguments."""
        if argv is None:
            argv = sys.argv
        
        # Extract meaningful identifiers from command line
        process_parts = ["connectome"]
        
        for arg in argv:
            if "--adapter" in arg or "-a" in arg:
                # Extract adapter type
                try:
                    adapter_idx = argv.index(arg)
                    if adapter_idx + 1 < len(argv):
                        adapter_type = argv[adapter_idx + 1]
                        process_parts.append(f"adapter_{adapter_type}")
                except (ValueError, IndexError):
                    process_parts.append("adapter")
            elif "discord" in arg.lower():
                process_parts.append("discord")
            elif "telegram" in arg.lower():
                process_parts.append("telegram")
            elif "slack" in arg.lower():
                process_parts.append("slack")
            elif "--test" in arg or "test" in arg.lower():
                process_parts.append("test")
            elif "--dev" in arg or "debug" in arg.lower():
                process_parts.append("dev")
        
        return "_".join(process_parts)

# Convenience functions for easy integration
def create_connectome_pid_file(storage_dir: str = "./storage_data", process_name: Optional[str] = None) -> ConnectomePIDManager:
    """Create a PID file for the current Connectome process."""
    if process_name is None:
        process_name = ConnectomePIDManager.get_process_name_from_args()
    
    pid_manager = ConnectomePIDManager(storage_dir, process_name)
    pid_manager.create_pid_file()
    return pid_manager

def cleanup_all_pid_files(storage_dir: str = "./storage_data"):
    """Clean up all PID files (for testing/reset purposes)."""
    pid_dir = Path(storage_dir) / "pids"
    if pid_dir.exists():
        for pid_file in pid_dir.glob("*.pid"):
            try:
                pid_file.unlink()
                logger.info(f"Removed PID file: {pid_file}")
            except Exception as e:
                logger.warning(f"Failed to remove PID file {pid_file}: {e}")

if __name__ == "__main__":
    # Test the PID manager
    logging.basicConfig(level=logging.INFO)
    
    print("Testing PID file management...")
    
    # Create a test PID file
    manager = create_connectome_pid_file(process_name="test_process")
    
    print(f"PID file created: {manager.pid_file_path}")
    print(f"Process PID: {os.getpid()}")
    
    input("Press Enter to cleanup and exit...")
    
    manager.cleanup_pid_file()
    print("PID file cleaned up") 