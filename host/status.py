#!/usr/bin/env python3
"""
Status command for Connectome processes.

This module provides a command-line interface to check the status
of running Connectome instances and their health.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import psutil
import json
import time
from datetime import datetime, timedelta

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_connectome_processes() -> List[dict]:
    """Find running Connectome processes with detailed information."""
    connectome_processes = []
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info', 'cpu_percent']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('connectome' in arg.lower() or 'host.main' in arg for arg in cmdline):
                    # Filter out status/shutdown commands
                    if not any(x in ' '.join(cmdline) for x in ['status.py', 'shutdown.py', 'connectome-status', 'connectome-shutdown']):
                        # Get additional process info
                        process = psutil.Process(proc.info['pid'])
                        
                        connectome_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': ' '.join(cmdline),
                            'create_time': proc.info['create_time'],
                            'age_seconds': time.time() - proc.info['create_time'],
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                            'cpu_percent': process.cpu_percent(),
                            'status': process.status(),
                            'num_threads': process.num_threads(),
                            'connections': len(process.connections()) if hasattr(process, 'connections') else 0
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        logger.error(f"Error scanning for Connectome processes: {e}")
    
    return connectome_processes

def check_storage_health(storage_dir: str = "./storage_data") -> Dict[str, Any]:
    """Check storage system health and provide statistics."""
    storage_path = Path(storage_dir)
    health_info = {
        'storage_exists': storage_path.exists(),
        'storage_path': str(storage_path.absolute()),
        'is_writable': False,
        'total_size_mb': 0,
        'file_count': 0,
        'directories': {},
        'recent_activity': None,
        'shutdown_state': None
    }
    
    try:
        if storage_path.exists():
            # Check write permissions
            test_file = storage_path / '.status_test'
            try:
                test_file.write_text('test')
                test_file.unlink()
                health_info['is_writable'] = True
            except Exception:
                health_info['is_writable'] = False
            
            # Calculate storage statistics
            total_size = 0
            file_count = 0
            most_recent = 0
            
            for item in storage_path.rglob('*'):
                if item.is_file():
                    try:
                        size = item.stat().st_size
                        total_size += size
                        file_count += 1
                        mtime = item.stat().st_mtime
                        if mtime > most_recent:
                            most_recent = mtime
                    except Exception:
                        continue
            
            health_info['total_size_mb'] = total_size / 1024 / 1024
            health_info['file_count'] = file_count
            
            if most_recent > 0:
                health_info['recent_activity'] = {
                    'last_modified': datetime.fromtimestamp(most_recent).isoformat(),
                    'age_minutes': (time.time() - most_recent) / 60
                }
            
            # Check for specific directories
            for subdir in ['conversations', 'cache', 'agents', 'system']:
                subdir_path = storage_path / subdir
                if subdir_path.exists():
                    file_count = len(list(subdir_path.rglob('*.json')))
                    health_info['directories'][subdir] = {
                        'exists': True,
                        'file_count': file_count
                    }
                else:
                    health_info['directories'][subdir] = {'exists': False}
            
            # Check for shutdown state
            shutdown_file = storage_path / '.shutdown_state'
            if shutdown_file.exists():
                try:
                    with open(shutdown_file, 'r') as f:
                        health_info['shutdown_state'] = json.load(f)
                except Exception:
                    health_info['shutdown_state'] = {'error': 'Could not read shutdown state'}
    
    except Exception as e:
        logger.error(f"Error checking storage health: {e}")
        health_info['error'] = str(e)
    
    return health_info

def format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"

def display_process_status(processes: List[dict], verbose: bool = False):
    """Display process status information."""
    if not processes:
        print("🚫 No Connectome processes are currently running")
        return
    
    print(f"🚀 Found {len(processes)} running Connectome process(es):")
    print()
    
    for i, proc in enumerate(processes, 1):
        uptime = format_uptime(proc['age_seconds'])
        memory_mb = proc['memory_mb']
        
        print(f"📊 Process {i}:")
        print(f"   PID: {proc['pid']}")
        print(f"   Status: {proc['status']}")
        print(f"   Uptime: {uptime}")
        print(f"   Memory: {memory_mb:.1f} MB")
        print(f"   CPU: {proc['cpu_percent']:.1f}%")
        print(f"   Threads: {proc['num_threads']}")
        print(f"   Connections: {proc['connections']}")
        
        if verbose:
            print(f"   Command: {proc['cmdline']}")
        
        print()

def display_storage_status(storage_info: Dict[str, Any], verbose: bool = False):
    """Display storage system status."""
    print("💾 Storage System Status:")
    
    if not storage_info['storage_exists']:
        print(f"   ❌ Storage directory not found: {storage_info['storage_path']}")
        return
    
    print(f"   📁 Path: {storage_info['storage_path']}")
    print(f"   ✅ Exists: {storage_info['storage_exists']}")
    print(f"   ✏️  Writable: {'✅' if storage_info['is_writable'] else '❌'}")
    print(f"   📊 Total Size: {storage_info['total_size_mb']:.1f} MB")
    print(f"   📄 File Count: {storage_info['file_count']}")
    
    if storage_info['recent_activity']:
        age_mins = storage_info['recent_activity']['age_minutes']
        if age_mins < 5:
            activity_status = "🟢 Very recent"
        elif age_mins < 30:
            activity_status = "🟡 Recent"
        elif age_mins < 1440:  # 24 hours
            activity_status = "🟠 Some time ago"
        else:
            activity_status = "🔴 Long time ago"
        
        print(f"   ⏰ Last Activity: {activity_status} ({age_mins:.1f} minutes ago)")
    else:
        print(f"   ⏰ Last Activity: 🔴 Unknown")
    
    # Directory status
    print("   📂 Directories:")
    for dir_name, dir_info in storage_info['directories'].items():
        if dir_info['exists']:
            print(f"      {dir_name}: ✅ ({dir_info['file_count']} files)")
        else:
            print(f"      {dir_name}: ❌ Missing")
    
    # Shutdown state
    if storage_info['shutdown_state']:
        shutdown_time = storage_info['shutdown_state'].get('shutdown_timestamp')
        if shutdown_time:
            shutdown_age = (time.time() - shutdown_time) / 60
            print(f"   🛑 Last Shutdown: {shutdown_age:.1f} minutes ago ({storage_info['shutdown_state'].get('shutdown_method', 'unknown')})")
    
    print()

def main():
    """Main entry point for the status command."""
    parser = argparse.ArgumentParser(
        description="Check status of Connectome processes and storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  connectome-status                      # Basic status information
  connectome-status --verbose           # Detailed status with command lines
  connectome-status --storage-dir /path  # Check specific storage directory
  connectome-status --json              # Output in JSON format
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed information including command lines')
    parser.add_argument('--storage-dir', '-s', default='./storage_data',
                       help='Storage directory path to check (default: ./storage_data)')
    parser.add_argument('--json', action='store_true',
                       help='Output status information in JSON format')
    parser.add_argument('--processes-only', '-p', action='store_true',
                       help='Show only process information, skip storage')
    parser.add_argument('--storage-only', action='store_true',
                       help='Show only storage information, skip processes')
    
    args = parser.parse_args()
    
    try:
        # Gather information
        processes = [] if args.storage_only else find_connectome_processes()
        storage_info = {} if args.processes_only else check_storage_health(args.storage_dir)
        
        if args.json:
            # JSON output
            output = {
                'timestamp': datetime.now().isoformat(),
                'processes': processes,
                'storage': storage_info
            }
            print(json.dumps(output, indent=2))
        else:
            # Human-readable output
            print("🔍 Connectome Status Report")
            print("=" * 50)
            print()
            
            if not args.storage_only:
                display_process_status(processes, args.verbose)
            
            if not args.processes_only:
                display_storage_status(storage_info, args.verbose)
            
            # Summary
            running_count = len(processes) if processes else 0
            storage_ok = storage_info.get('storage_exists', False) and storage_info.get('is_writable', False)
            
            if running_count > 0 and storage_ok:
                status_emoji = "🟢"
                status_text = "Healthy"
            elif running_count > 0:
                status_emoji = "🟡"
                status_text = "Running (storage issues)"
            elif storage_ok:
                status_emoji = "🟡"
                status_text = "Storage OK (not running)"
            else:
                status_emoji = "🔴"
                status_text = "Offline"
            
            print(f"📋 Overall Status: {status_emoji} {status_text}")
            
            if not args.processes_only and not args.storage_only:
                print(f"   Processes: {running_count} running")
                print(f"   Storage: {'✅' if storage_ok else '❌'}")
        
        # Exit code: 0 if healthy, 1 if issues
        if args.processes_only:
            sys.exit(0 if len(processes) > 0 else 1)
        elif args.storage_only:
            sys.exit(0 if storage_info.get('storage_exists') and storage_info.get('is_writable') else 1)
        else:
            healthy = len(processes) > 0 and storage_info.get('storage_exists', False) and storage_info.get('is_writable', False)
            sys.exit(0 if healthy else 1)
            
    except Exception as e:
        logger.error(f"Status command failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 