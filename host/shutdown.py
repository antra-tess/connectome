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
    """
    Find running Connectome agent processes with precise targeting.
    
    IMPROVED: Only targets actual Connectome agent processes, not infrastructure.
    """
    connectome_processes = []
    
    # Define precise patterns for actual Connectome agent processes
    CONNECTOME_ENTRY_POINTS = [
        'host.main',           # python -m host.main
        'host/main.py',        # python host/main.py
        'main.py',             # python main.py (from host dir)
        'connectome_agent',    # If packaged as executable
        'run_connectome',      # Custom runner scripts
    ]
    
    # Define patterns to EXCLUDE (infrastructure that should not be killed)
    EXCLUDE_PATTERNS = [
        'shutdown.py',
        'connectome-shutdown', 
        'connectome-adapters',
        'connectome-infra',
        'connectome-monitor',
        'connectome-proxy',
        'connectome-gateway',
        'connectome-build',
        'connectome-deploy',
        'connectome-test',
        'pytest',
        'nose',
        'unittest'
    ]
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cwd']):
            try:
                cmdline = proc.info['cmdline']
                if not cmdline:
                    continue
                
                cmdline_str = ' '.join(cmdline)
                
                # First, check exclusion patterns
                if any(exclude_pattern in cmdline_str.lower() for exclude_pattern in EXCLUDE_PATTERNS):
                    continue
                
                # Check if this matches a known Connectome entry point
                is_connectome_agent = False
                matched_entry_point = None
                
                for entry_point in CONNECTOME_ENTRY_POINTS:
                    if entry_point in cmdline_str:
                        is_connectome_agent = True
                        matched_entry_point = entry_point
                        break
                
                # Additional validation: check working directory for Connectome project
                if is_connectome_agent:
                    try:
                        cwd = proc.info['cwd']
                        # Look for Connectome project indicators in working directory
                        if cwd:
                            cwd_path = Path(cwd)
                            connectome_indicators = [
                                cwd_path / 'elements',
                                cwd_path / 'host',
                                cwd_path / 'pyproject.toml',
                                cwd_path / 'connectome.toml'
                            ]
                            
                            # Only consider it a Connectome agent if we find project indicators
                            if any(indicator.exists() for indicator in connectome_indicators):
                                # Determine process type for better classification
                                process_type = _classify_connectome_process(cmdline_str, cwd)
                                
                                connectome_processes.append({
                                    'pid': proc.info['pid'],
                                    'name': proc.info['name'],
                                    'cmdline': cmdline_str,
                                    'create_time': proc.info['create_time'],
                                    'age_seconds': time.time() - proc.info['create_time'],
                                    'cwd': cwd,
                                    'entry_point': matched_entry_point,
                                    'process_type': process_type
                                })
                    except (psutil.AccessDenied, OSError):
                        # If we can't access CWD, be more conservative
                        logger.warning(f"Cannot verify working directory for PID {proc.info['pid']}, skipping")
                        continue
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.error(f"Error scanning for Connectome processes: {e}")
    
    return connectome_processes

def _classify_connectome_process(cmdline: str, cwd: str) -> str:
    """Classify the type of Connectome process for better reporting."""
    if 'discord' in cmdline.lower():
        return 'discord_agent'
    elif 'telegram' in cmdline.lower():
        return 'telegram_agent'
    elif 'slack' in cmdline.lower():
        return 'slack_agent'
    elif '--adapter' in cmdline:
        return 'adapter_agent'
    elif 'test' in cmdline.lower():
        return 'test_instance'
    elif 'dev' in cmdline.lower() or 'debug' in cmdline.lower():
        return 'development_instance'
    else:
        return 'main_agent'

def find_connectome_by_pidfile(storage_dir: str = "./storage_data") -> List[dict]:
    """
    NEW: Find Connectome processes using PID files for more reliable tracking.
    """
    processes = []
    pid_dir = Path(storage_dir) / "pids"
    
    if not pid_dir.exists():
        return processes
    
    try:
        for pid_file in pid_dir.glob("*.pid"):
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # Check if process is still running
                if psutil.pid_exists(pid):
                    proc = psutil.Process(pid)
                    cmdline = ' '.join(proc.cmdline())
                    
                    # Verify it's actually a Connectome process
                    if any(entry in cmdline for entry in ['host.main', 'host/main.py']):
                        processes.append({
                            'pid': pid,
                            'name': proc.name(),
                            'cmdline': cmdline,
                            'create_time': proc.create_time(),
                            'age_seconds': time.time() - proc.create_time(),
                            'pid_file': str(pid_file),
                            'process_type': _classify_connectome_process(cmdline, proc.cwd())
                        })
                else:
                    # Clean up stale PID file
                    logger.info(f"Cleaning up stale PID file: {pid_file}")
                    pid_file.unlink()
                    
            except (ValueError, psutil.NoSuchProcess, psutil.AccessDenied, OSError) as e:
                logger.warning(f"Error processing PID file {pid_file}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error scanning PID files: {e}")
    
    return processes

def get_all_connectome_processes(storage_dir: str = "./storage_data") -> List[dict]:
    """
    Get all Connectome processes using both process scanning and PID files.
    
    NEW: Combines both methods for comprehensive but safe process discovery.
    """
    # Get processes from both methods
    scanned_processes = find_connectome_processes()
    pidfile_processes = find_connectome_by_pidfile(storage_dir)
    
    # Merge and deduplicate by PID
    all_processes = {}
    
    # Add scanned processes
    for proc in scanned_processes:
        all_processes[proc['pid']] = proc
    
    # Add PID file processes (may override with more accurate info)
    for proc in pidfile_processes:
        if proc['pid'] in all_processes:
            # Merge information, preferring PID file data for reliability
            all_processes[proc['pid']]['pid_file'] = proc.get('pid_file')
        else:
            all_processes[proc['pid']] = proc
    
    return list(all_processes.values())

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
    all_processes = get_all_connectome_processes(args.storage_dir)
    
    # Apply filtering based on command-line arguments
    processes = filter_processes(all_processes, args)
    
    if not processes:
        if all_processes:
            logger.info("No Connectome processes match the specified criteria")
            if args.list_only:
                logger.info("Available processes (not matching criteria):")
                _show_process_summary(all_processes, args)
        else:
            logger.info("No running Connectome processes found")
        
        if args.save_state and not args.dry_run:
            save_shutdown_state(args.storage_dir)
        return True
    
    # Show process summary
    _show_process_summary(processes, args)
    
    if args.list_only or args.dry_run:
        if args.dry_run:
            logger.info("DRY RUN: Would shutdown the above processes")
        return True
    
    # Special handling for different process types
    critical_processes = [p for p in processes if p.get('process_type') in ['main_agent', 'discord_agent', 'telegram_agent', 'slack_agent']]
    test_processes = [p for p in processes if p.get('process_type') == 'test_instance']
    dev_processes = [p for p in processes if p.get('process_type') == 'development_instance']
    
    # Show warnings for critical processes
    if critical_processes and not args.force:
        logger.warning(f"Found {len(critical_processes)} critical agent process(es) that will be shut down:")
        for proc in critical_processes:
            logger.warning(f"  - {proc.get('process_type', 'unknown')} (PID {proc['pid']})")
    
    # Confirm shutdown unless forced
    if not args.force:
        shutdown_summary = []
        if critical_processes:
            shutdown_summary.append(f"{len(critical_processes)} agent(s)")
        if test_processes:
            shutdown_summary.append(f"{len(test_processes)} test instance(s)")
        if dev_processes:
            shutdown_summary.append(f"{len(dev_processes)} dev instance(s)")
        
        summary = ", ".join(shutdown_summary)
        response = input(f"\nShutdown {summary}? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            logger.info("Shutdown cancelled by user")
            return False
    
    # Save shutdown state before terminating processes
    if args.save_state:
        save_shutdown_state(args.storage_dir)
    
    # Shutdown processes with different timeouts based on type
    success_count = 0
    for proc in processes:
        proc_type = proc.get('process_type', 'unknown')
        
        # Use longer timeout for critical processes
        if proc_type in ['main_agent', 'discord_agent', 'telegram_agent', 'slack_agent']:
            timeout = max(args.timeout, 45)  # At least 45 seconds for agents
        elif proc_type == 'test_instance':
            timeout = 15  # Shorter timeout for test instances
        else:
            timeout = args.timeout
        
        logger.info(f"Shutting down {proc_type} (PID {proc['pid']}) with {timeout}s timeout...")
        
        if send_graceful_shutdown_signal(proc['pid'], timeout):
            success_count += 1
            
            # Clean up PID file if it exists
            if proc.get('pid_file'):
                try:
                    Path(proc['pid_file']).unlink(missing_ok=True)
                    logger.debug(f"Cleaned up PID file: {proc['pid_file']}")
                except Exception as e:
                    logger.warning(f"Could not remove PID file {proc['pid_file']}: {e}")
    
    if success_count == len(processes):
        logger.info(f"Successfully shut down all {len(processes)} Connectome processes")
        return True
    else:
        logger.warning(f"Successfully shut down {success_count}/{len(processes)} processes")
        return False

def filter_processes(processes: List[dict], args) -> List[dict]:
    """Filter processes based on command-line arguments."""
    filtered = processes
    
    # Filter by specific PIDs if provided
    if args.pid:
        filtered = [p for p in filtered if p['pid'] in args.pid]
        logger.info(f"Filtering by PIDs: {args.pid}")
    
    # Filter by process type if specified
    if args.type:
        filtered = [p for p in filtered if p.get('process_type') == args.type]
        logger.info(f"Filtering by type: {args.type}")
    
    # Exclude specific process type if specified
    if args.exclude:
        filtered = [p for p in filtered if p.get('process_type') != args.exclude]
        logger.info(f"Excluding type: {args.exclude}")
    
    return filtered

def _show_process_summary(processes: List[dict], args):
    """Show a summary of processes in a consistent format."""
    # Categorize processes for better reporting
    process_categories = {}
    for proc in processes:
        proc_type = proc.get('process_type', 'unknown')
        if proc_type not in process_categories:
            process_categories[proc_type] = []
        process_categories[proc_type].append(proc)
    
    logger.info(f"Found {len(processes)} Connectome process(es):")
    for category, procs in process_categories.items():
        logger.info(f"  {category.replace('_', ' ').title()}: {len(procs)} process(es)")
        for proc in procs:
            age_mins = proc['age_seconds'] / 60
            entry_point = proc.get('entry_point', 'unknown')
            pid_file_info = f" (PID file: {Path(proc['pid_file']).name})" if proc.get('pid_file') else ""
            
            logger.info(f"    PID {proc['pid']}: {proc['name']} via {entry_point} (running {age_mins:.1f} minutes){pid_file_info}")
            if args.verbose:
                logger.info(f"      Command: {proc['cmdline']}")
                logger.info(f"      Working Dir: {proc.get('cwd', 'unknown')}")

def main():
    """Main entry point for the shutdown command."""
    parser = argparse.ArgumentParser(
        description="Gracefully shutdown Connectome agent processes with precise targeting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Process Types:
  main_agent         - Core Connectome agent instances
  discord_agent      - Discord-connected agents  
  telegram_agent     - Telegram-connected agents
  slack_agent        - Slack-connected agents
  adapter_agent      - Generic adapter agents
  test_instance      - Test/development instances
  development_instance - Development instances

Examples:
  connectome-shutdown                           # Interactive shutdown of all processes
  connectome-shutdown --force                   # Force shutdown without confirmation
  connectome-shutdown --list                    # List running processes only
  connectome-shutdown --type main_agent         # Shutdown only main agents
  connectome-shutdown --type test_instance     # Shutdown only test instances
  connectome-shutdown --exclude test_instance  # Shutdown all except test instances
  connectome-shutdown --timeout 60             # Wait up to 60 seconds for graceful shutdown
  connectome-shutdown --storage-dir /path      # Specify storage directory for state

Safety Features:
  - Only targets actual Connectome agent processes
  - Excludes infrastructure processes (connectome-*, build tools, etc.)
  - Verifies working directory contains Connectome project files
  - Uses PID files when available for reliable tracking
  - Different timeouts for different process types
  - Detailed process classification and reporting
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
                       help='Verbose output including command lines and working directories')
    
    # NEW: Process type filtering options
    parser.add_argument('--type', choices=[
        'main_agent', 'discord_agent', 'telegram_agent', 'slack_agent', 
        'adapter_agent', 'test_instance', 'development_instance'
    ], help='Shutdown only processes of this type')
    
    parser.add_argument('--exclude', choices=[
        'main_agent', 'discord_agent', 'telegram_agent', 'slack_agent',
        'adapter_agent', 'test_instance', 'development_instance'  
    ], help='Exclude processes of this type from shutdown')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be shut down without actually doing it')
    
    parser.add_argument('--pid', type=int, action='append',
                       help='Shutdown specific process by PID (can be used multiple times)')
    
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