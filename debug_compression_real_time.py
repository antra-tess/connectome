#!/usr/bin/env python3
"""
Real-time Compression Debug Monitor

This script helps debug compression behavior while actually using Connectome.
Run this alongside your Connectome session to see what's happening.
"""

import os
import json
import time
import glob
from datetime import datetime
from pathlib import Path

def check_memory_storage():
    """Check the current state of memory storage."""
    print(f"\n🔍 Memory Storage Check - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    # Check main storage directory
    storage_base = "storage_data/memory_storage"
    
    if not os.path.exists(storage_base):
        print(f"❌ Storage directory doesn't exist: {storage_base}")
        return
    
    print(f"📁 Storage base: {storage_base}")
    
    # Find agent directories
    agents_dir = os.path.join(storage_base, "agents")
    if os.path.exists(agents_dir):
        agent_dirs = [d for d in os.listdir(agents_dir) if os.path.isdir(os.path.join(agents_dir, d))]
        print(f"🤖 Found {len(agent_dirs)} agent directories: {agent_dirs}")
        
        for agent_id in agent_dirs:
            agent_path = os.path.join(agents_dir, agent_id)
            print(f"\n  Agent: {agent_id}")
            
            # Check for memory files
            memory_files = glob.glob(os.path.join(agent_path, "*.json"))
            print(f"    💾 Memory files: {len(memory_files)}")
            
            for memory_file in memory_files:
                file_size = os.path.getsize(memory_file)
                mod_time = os.path.getmtime(memory_file)
                mod_time_str = datetime.fromtimestamp(mod_time).strftime('%H:%M:%S')
                filename = os.path.basename(memory_file)
                
                print(f"      📄 {filename} ({file_size} bytes, modified: {mod_time_str})")
                
                # Try to read the memory file
                try:
                    with open(memory_file, 'r') as f:
                        memory_data = json.load(f)
                    
                    # Extract key info
                    memorized_node = memory_data.get("memorized_node", {})
                    props = memorized_node.get("properties", {})
                    summary = props.get("memory_summary", "No summary")
                    element_ids = memory_data.get("metadata", {}).get("element_ids", [])
                    
                    print(f"         🧠 Elements: {element_ids}")
                    print(f"         📝 Summary: {summary[:60]}...")
                    
                except Exception as e:
                    print(f"         ❌ Error reading file: {e}")
    else:
        print(f"❌ Agents directory doesn't exist: {agents_dir}")

def check_background_compression_status():
    """Check for signs of background compression activity."""
    print(f"\n⚙️  Background Compression Activity Check")
    print("=" * 60)
    
    # Look for temp files or other signs of compression activity
    temp_patterns = [
        "storage_data/memory_storage/**/*temp*",
        "storage_data/memory_storage/**/*tmp*",
        "storage_data/memory_storage/**/*progress*"
    ]
    
    temp_files = []
    for pattern in temp_patterns:
        temp_files.extend(glob.glob(pattern, recursive=True))
    
    if temp_files:
        print(f"🔄 Found {len(temp_files)} temporary files (compression in progress?):")
        for temp_file in temp_files:
            print(f"    📄 {temp_file}")
    else:
        print("📊 No temporary files found (no active compression detected)")

def analyze_memory_file_changes():
    """Analyze recent changes to memory files."""
    print(f"\n📈 Recent Memory File Changes")
    print("=" * 60)
    
    storage_base = "storage_data/memory_storage"
    if not os.path.exists(storage_base):
        print("❌ Storage directory doesn't exist")
        return
    
    # Find all JSON files
    memory_files = glob.glob(os.path.join(storage_base, "**", "*.json"), recursive=True)
    
    # Sort by modification time
    recent_files = []
    current_time = time.time()
    
    for file_path in memory_files:
        mod_time = os.path.getmtime(file_path)
        age_minutes = (current_time - mod_time) / 60
        
        if age_minutes < 30:  # Files modified in last 30 minutes
            recent_files.append((file_path, mod_time, age_minutes))
    
    recent_files.sort(key=lambda x: x[1], reverse=True)  # Sort by modification time, newest first
    
    if recent_files:
        print(f"📁 {len(recent_files)} memory files modified in last 30 minutes:")
        for file_path, mod_time, age_minutes in recent_files:
            filename = os.path.basename(file_path)
            agent_dir = os.path.basename(os.path.dirname(file_path))
            mod_time_str = datetime.fromtimestamp(mod_time).strftime('%H:%M:%S')
            
            print(f"    🕒 {mod_time_str} ({age_minutes:.1f}m ago) - {agent_dir}/{filename}")
    else:
        print("📊 No recently modified memory files found")

def suggest_debugging_steps():
    """Suggest debugging steps based on findings."""
    print(f"\n🛠️  Debugging Suggestions")
    print("=" * 60)
    
    suggestions = [
        "1. Check Connectome logs for AgentMemoryCompressor activity",
        "2. Look for 'get_memory_or_fallback' calls in the logs",
        "3. Check if ThreadPoolExecutor is starting background tasks",
        "4. Verify that compression_context is being passed correctly",
        "5. Monitor storage_data/memory_storage for new files during refocus",
        "6. Check if LLM provider is available and working",
        "7. Look for any asyncio or threading-related errors in logs"
    ]
    
    for suggestion in suggestions:
        print(f"    💡 {suggestion}")

def watch_mode():
    """Run in watch mode to continuously monitor changes."""
    print("👁️  Starting watch mode (Ctrl+C to stop)")
    print("🔄 Monitoring every 10 seconds...")
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
            
            print("🔍 CONNECTOME COMPRESSION MONITOR")
            print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            check_memory_storage()
            check_background_compression_status()
            analyze_memory_file_changes()
            
            print(f"\n⏳ Next check in 10 seconds... (Ctrl+C to stop)")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")

def main():
    """Main function to run diagnostics."""
    import sys
    
    print("🚀 Connectome Compression Debug Monitor")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        watch_mode()
    else:
        print("📊 Single-shot analysis:")
        check_memory_storage()
        check_background_compression_status()
        analyze_memory_file_changes()
        suggest_debugging_steps()
        
        print(f"\n💡 Tip: Run 'python {sys.argv[0]} watch' for continuous monitoring")

if __name__ == "__main__":
    main() 