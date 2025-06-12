#!/usr/bin/env python3
"""
Graceful shutdown command for Connectome background processes.

This module provides a command-line interface to gracefully shutdown
running Connectome instances, ensuring proper persistence and cleanup.
"""

import os
import sys
import signal
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Optional, List
import psutil
import json
import time

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_connectome_processes() -> List[dict]:
    """Find running Connectome processes."""
    connectome_processes = []
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('connectome' in arg.lower() or 'host.main' in arg for arg in cmdline):
                    # Filter out this shutdown command itself
                    if 'shutdown.py' not in ' '.join(cmdline) and 'connectome-shutdown' not in ' '.join(cmdline):
                        connectome_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': ' '.join(cmdline),
                            'create_time': proc.info['create_time'],
                            'age_seconds': time.time() - proc.info['create_time']
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.error(f"Error scanning for Connectome processes: {e}")
    
    return connectome_processes

def send_graceful_shutdown_signal(pid: int, timeout: int = 30) -> bool:
    """Send graceful shutdown signal to a process."""
    try:
        process = psutil.Process(pid)
        logger.info(f"Sending SIGTERM to Connectome process {pid} ({process.name()})")
        
        # Send SIGTERM for graceful shutdown
        process.send_signal(signal.SIGTERM)
        
        # Wait for graceful shutdown
        try:
            process.wait(timeout=timeout)
            logger.info(f"Process {pid} shut down gracefully")
            return True
        except psutil.TimeoutExpired:
            logger.warning(f"Process {pid} did not respond to SIGTERM within {timeout}s, sending SIGKILL")
            process.send_signal(signal.SIGKILL)
            try:
                process.wait(timeout=5)
                logger.info(f"Process {pid} forcefully terminated")
                return True
            except psutil.TimeoutExpired:
                logger.error(f"Process {pid} could not be terminated")
                return False
    except psutil.NoSuchProcess:
        logger.info(f"Process {pid} has already exited")
        return True
    except psutil.AccessDenied:
        logger.error(f"Permission denied to terminate process {pid}")
        return False
    except Exception as e:
        logger.error(f"Error terminating process {pid}: {e}")
        return False

def save_shutdown_state(storage_dir: str = "./storage_data") -> bool:
    """Save shutdown timestamp and state information."""
    try:
        storage_path = Path(storage_dir)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        shutdown_state = {
            'shutdown_timestamp': time.time(),
            'shutdown_command': 'connectome-shutdown',
            'shutdown_method': 'graceful',
            'storage_dir': str(storage_path.absolute())
        }
        
        shutdown_file = storage_path / '.shutdown_state'
        with open(shutdown_file, 'w') as f:
            json.dump(shutdown_state, f, indent=2)
        
        logger.info(f"Shutdown state saved to {shutdown_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save shutdown state: {e}")
        return False

async def shutdown_connectome_async(args) -> bool:
    """Async shutdown logic for comprehensive cleanup."""
    logger.info("Starting comprehensive Connectome shutdown...")
    
    # Find running processes
    processes = find_connectome_processes()
    
    if not processes:
        logger.info("No running Connectome processes found")
        if args.save_state:
            save_shutdown_state(args.storage_dir)
        return True
    
    logger.info(f"Found {len(processes)} Connectome process(es):")
    for proc in processes:
        age_mins = proc['age_seconds'] / 60
        logger.info(f"  PID {proc['pid']}: {proc['name']} (running {age_mins:.1f} minutes)")
        if args.verbose:
            logger.info(f"    Command: {proc['cmdline']}")
    
    if args.list_only:
        return True
    
    # Confirm shutdown unless forced
    if not args.force:
        response = input(f"\nShutdown {len(processes)} Connectome process(es)? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            logger.info("Shutdown cancelled by user")
            return False
    
    # Save shutdown state before terminating processes
    if args.save_state:
        save_shutdown_state(args.storage_dir)
    
    # Shutdown each process gracefully
    success_count = 0
    for proc in processes:
        if send_graceful_shutdown_signal(proc['pid'], args.timeout):
            success_count += 1
    
    if success_count == len(processes):
        logger.info(f"Successfully shut down all {len(processes)} Connectome processes")
        return True
    else:
        logger.warning(f"Successfully shut down {success_count}/{len(processes)} processes")
        return False

def main():
    """Main entry point for the shutdown command."""
    parser = argparse.ArgumentParser(
        description="Gracefully shutdown Connectome background processes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  connectome-shutdown                    # Interactive shutdown of all processes
  connectome-shutdown --force            # Force shutdown without confirmation
  connectome-shutdown --list             # List running processes only
  connectome-shutdown --timeout 60       # Wait up to 60 seconds for graceful shutdown
  connectome-shutdown --storage-dir /path # Specify storage directory for state
        """
    )
    
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force shutdown without confirmation')
    parser.add_argument('--list', '-l', action='store_true', dest='list_only',
                       help='List running Connectome processes without shutting down')
    parser.add_argument('--timeout', '-t', type=int, default=30,
                       help='Timeout in seconds for graceful shutdown (default: 30)')
    parser.add_argument('--storage-dir', '-s', default='./storage_data',
                       help='Storage directory path (default: ./storage_data)')
    parser.add_argument('--save-state', action='store_true', default=True,
                       help='Save shutdown state information (default: True)')
    parser.add_argument('--no-save-state', action='store_false', dest='save_state',
                       help='Skip saving shutdown state information')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output including command lines')
    
    args = parser.parse_args()
    
    try:
        success = asyncio.run(shutdown_connectome_async(args))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Shutdown command interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Shutdown command failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 