#!/usr/bin/env python3
"""
IPC-based TUI Inspector

Interactive text-based user interface for inspecting Connectome host state via IPC.
Provides tree-like navigation, editing capabilities, and real-time data updates.
"""

import asyncio
import json
import sys
import os
import logging
import termios
import tty
import signal
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Add the connectome root to path so we can import modules
connectome_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(connectome_root))

from host.modules.inspector.ipc_client import IPCClient, IPCCommandExecutor

logger = logging.getLogger(__name__)


class NavigationMode(Enum):
    MAIN_MENU = "main_menu"
    TREE_VIEW = "tree_view"
    DETAIL_VIEW = "detail_view"
    EDIT_MODE = "edit_mode"
    HOST_SELECTION = "host_selection"


@dataclass
class TreeNode:
    """Represents a node in the tree navigation."""
    id: str
    label: str
    data: Any = None
    children: List['TreeNode'] = field(default_factory=list)
    parent: Optional['TreeNode'] = None
    is_expanded: bool = False
    is_editable: bool = False
    node_type: str = "data"  # data, command, action
    command_path: Optional[str] = None  # For executable commands
    write_endpoint: Optional[str] = None  # For editable nodes


class TerminalController:
    """Handles low-level terminal operations."""
    
    def __init__(self):
        self.original_settings = None
        
    def setup_terminal(self):
        """Setup terminal for raw input mode."""
        if not sys.stdin.isatty():
            return
        
        try:
            self.original_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin)
        except (termios.error, AttributeError):
            pass
    
    def restore_terminal(self):
        """Restore original terminal settings."""
        if self.original_settings and sys.stdin.isatty():
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_settings)
            except (termios.error, AttributeError):
                pass
    
    def get_terminal_size(self) -> Tuple[int, int]:
        """Get terminal dimensions."""
        try:
            size = os.get_terminal_size()
            return size.lines, size.columns
        except OSError:
            return 24, 80
    
    def clear_screen(self):
        """Clear the terminal screen."""
        print('\033[2J\033[H', end='', flush=True)
    
    def move_cursor(self, row: int, col: int):
        """Move cursor to specific position."""
        print(f'\033[{row};{col}H', end='', flush=True)
    
    def hide_cursor(self):
        """Hide the cursor."""
        print('\033[?25l', end='', flush=True)
    
    def show_cursor(self):
        """Show the cursor."""
        print('\033[?25h', end='', flush=True)
    
    def set_color(self, fg: Optional[str] = None, bg: Optional[str] = None, bold: bool = False):
        """Set text color and style."""
        codes = []
        
        if bold:
            codes.append('1')
        
        color_map = {
            'black': '30', 'red': '31', 'green': '32', 'yellow': '33',
            'blue': '34', 'magenta': '35', 'cyan': '36', 'white': '37',
            'bright_black': '90', 'bright_red': '91', 'bright_green': '92',
            'bright_yellow': '93', 'bright_blue': '94', 'bright_magenta': '95',
            'bright_cyan': '96', 'bright_white': '97'
        }
        
        if fg and fg in color_map:
            codes.append(color_map[fg])
        
        if bg and bg in color_map:
            codes.append(str(int(color_map[bg]) + 10))
        
        if codes:
            print(f'\033[{";".join(codes)}m', end='', flush=True)
    
    def reset_colors(self):
        """Reset all colors and styles."""
        print('\033[0m', end='', flush=True)
    
    def get_key(self) -> str:
        """Get a single keypress."""
        if not sys.stdin.isatty():
            return input()
        
        try:
            key = sys.stdin.read(1)
            
            # Handle escape sequences (arrow keys, etc.)
            if ord(key) == 27:  # ESC
                try:
                    next_char = sys.stdin.read(1)
                    if next_char == '[':
                        third_char = sys.stdin.read(1)
                        return f'ESC[{third_char}'
                except:
                    return 'ESC'
            
            return key
        except (KeyboardInterrupt, EOFError):
            return 'CTRL_C'


class IPCTUIInspector:
    """
    IPC-based Text User Interface for Connectome Inspector.
    
    Provides an interactive terminal interface that connects via IPC to 
    running Connectome instances with:
    - Host discovery and selection
    - Tree-like exploration of data structures  
    - Editing capabilities for writable nodes
    - Real-time data updates
    - Multi-host support
    """
    
    def __init__(self, socket_path: str = None, timeout: float = 30.0):
        """
        Initialize IPC TUI inspector.
        
        Args:
            socket_path: Path to Unix socket (auto-discovered if None)
            timeout: Command timeout in seconds
        """
        self.socket_path = socket_path
        self.timeout = timeout
        self.ipc_client = IPCClient(socket_path, timeout)
        self.executor = IPCCommandExecutor(socket_path, timeout)
        
        self.terminal = TerminalController()
        
        # UI State
        self.mode = NavigationMode.HOST_SELECTION if not socket_path else NavigationMode.MAIN_MENU
        self.current_menu_index = 0
        self.current_tree_index = 0
        self.current_tree_node = None
        self.tree_root = None
        self.scroll_offset = 0
        self.status_message = ""
        self.edit_buffer = ""
        self.search_query = ""
        
        # Available hosts
        self.available_hosts = []
        self.current_host_index = 0
        self.current_host = None
        
        # Main menu items
        self.main_menu_items = [
            ("System Status", "status", "📊 Overall system status and health"),
            ("Spaces", "spaces", "🏢 Detailed information about all spaces"),
            ("Agents", "agents", "🤖 Agent configurations and status"),
            ("Adapters", "adapters", "🔌 Activity adapter connections"),
            ("Metrics", "metrics", "📈 System performance metrics"),
            ("Timelines", "timelines", "⏰ Timeline DAG overview"),
            ("VEIL Overview", "veil", "👁️ VEIL system overview"),
            ("Health Check", "health", "✅ Simple health check"),
            ("Switch Host", "switch_host", "🔄 Connect to a different host"),
            ("Quit", "quit", "❌ Exit the inspector")
        ]
        
        # Track if we're running
        self.running = True
        
    async def start(self):
        """Start the IPC TUI inspector."""
        try:
            # Setup signal handling
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Setup terminal
            self.terminal.setup_terminal()
            self.terminal.hide_cursor()
            
            # Discover hosts if needed
            if self.mode == NavigationMode.HOST_SELECTION:
                await self._discover_hosts()
            
            # Main loop
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"IPC TUI Inspector error: {e}", exc_info=True)
            self.status_message = f"Error: {str(e)}"
        finally:
            await self._cleanup()
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        self.running = False
    
    async def _cleanup(self):
        """Cleanup terminal state and connections."""
        self.terminal.show_cursor()
        self.terminal.reset_colors()
        self.terminal.restore_terminal()
        
        # Disconnect IPC client
        if self.ipc_client.is_connected:
            await self.ipc_client.disconnect()
    
    async def _discover_hosts(self):
        """Discover available Connectome hosts."""
        self.status_message = "Discovering Connectome hosts..."
        try:
            self.available_hosts = await self.executor.list_available_hosts()
            if self.available_hosts:
                self.status_message = f"Found {len(self.available_hosts)} host(s)"
            else:
                self.status_message = "No running Connectome hosts found"
        except Exception as e:
            self.status_message = f"Host discovery failed: {str(e)}"
            self.available_hosts = []
    
    async def _connect_to_host(self, host_info: Dict[str, Any]) -> bool:
        """Connect to a specific host."""
        try:
            self.status_message = f"Connecting to host PID {host_info.get('pid', 'unknown')}..."
            
            # Update socket path
            self.socket_path = host_info["socket_path"]
            self.ipc_client = IPCClient(self.socket_path, self.timeout)
            self.executor = IPCCommandExecutor(self.socket_path, self.timeout)
            
            # Test connection
            response = await self.executor.execute_command("health")
            if response.get("error"):
                self.status_message = f"Connection failed: {response['error']}"
                return False
            
            self.current_host = host_info
            self.status_message = f"Connected to host PID {host_info.get('pid', 'unknown')}"
            return True
            
        except Exception as e:
            self.status_message = f"Connection error: {str(e)}"
            return False
    
    async def _main_loop(self):
        """Main UI event loop."""
        while self.running:
            try:
                # Render current screen
                await self._render_screen()
                
                # Get user input
                key = self.terminal.get_key()
                
                # Handle input based on current mode
                await self._handle_input(key)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self.status_message = f"Internal error: {str(e)}"
                await asyncio.sleep(0.1)
    
    async def _render_screen(self):
        """Render the current screen based on mode."""
        self.terminal.clear_screen()
        
        # Render header
        await self._render_header()
        
        # Render content based on mode
        if self.mode == NavigationMode.HOST_SELECTION:
            await self._render_host_selection()
        elif self.mode == NavigationMode.MAIN_MENU:
            await self._render_main_menu()
        elif self.mode == NavigationMode.TREE_VIEW:
            await self._render_tree_view()
        elif self.mode == NavigationMode.DETAIL_VIEW:
            await self._render_detail_view()
        elif self.mode == NavigationMode.EDIT_MODE:
            await self._render_edit_mode()
        
        # Render footer
        await self._render_footer()
    
    async def _render_header(self):
        """Render the header with title and breadcrumbs."""
        rows, cols = self.terminal.get_terminal_size()
        
        # Title bar
        self.terminal.move_cursor(1, 1)
        self.terminal.set_color('bright_cyan', bold=True)
        title = "🔍 Connectome Inspector IPC TUI"
        
        # Add host info if connected
        if self.current_host:
            host_info = f" [PID {self.current_host.get('pid', '?')}]"
            title += host_info
        
        # Add breadcrumbs based on mode
        breadcrumb = ""
        if self.mode == NavigationMode.TREE_VIEW and self.current_tree_node:
            breadcrumb = f" > {self.current_tree_node.label}"
        elif self.mode == NavigationMode.DETAIL_VIEW and self.current_tree_node:
            breadcrumb = f" > {self.current_tree_node.label} > Details"
        elif self.mode == NavigationMode.EDIT_MODE and self.current_tree_node:
            breadcrumb = f" > {self.current_tree_node.label} > Edit"
        elif self.mode == NavigationMode.HOST_SELECTION:
            breadcrumb = " > Select Host"
        
        full_title = title + breadcrumb
        if len(full_title) > cols - 2:
            full_title = full_title[:cols-5] + "..."
        
        print(full_title.ljust(cols), end='')
        
        # Separator line
        self.terminal.move_cursor(2, 1)
        self.terminal.set_color('cyan')
        print('─' * cols, end='')
        
        self.terminal.reset_colors()
    
    async def _render_host_selection(self):
        """Render the host selection screen."""
        rows, cols = self.terminal.get_terminal_size()
        
        self.terminal.move_cursor(4, 2)
        self.terminal.set_color('bright_green', bold=True)
        print("Available Connectome Hosts:")
        
        if not self.available_hosts:
            self.terminal.move_cursor(6, 4)
            self.terminal.set_color('yellow')
            print("No running hosts found. Make sure Connectome is running with inspector enabled.")
            self.terminal.move_cursor(8, 4)
            self.terminal.set_color('bright_cyan')
            print("Press 'r' to refresh, 'q' to quit")
            self.terminal.reset_colors()
            return
        
        for i, host in enumerate(self.available_hosts):
            row = 6 + i
            if row >= rows - 3:
                break
                
            self.terminal.move_cursor(row, 4)
            
            # Highlight current selection
            if i == self.current_host_index:
                self.terminal.set_color('black', 'bright_cyan', bold=True)
                marker = "► "
            else:
                self.terminal.set_color('white')
                marker = "  "
            
            # Host info
            pid = host.get('pid', '?')
            status = host.get('status', 'unknown')
            socket_path = host.get('socket_path', 'unknown')
            created_at = host.get('created_at', 0)
            created_str = datetime.fromtimestamp(created_at).strftime('%H:%M:%S') if created_at else '?'
            
            # Status colors
            if status == 'available':
                status_color = 'bright_green'
            elif status == 'error':
                status_color = 'bright_red'
            else:
                status_color = 'yellow'
            
            print(f"{marker}PID {pid} - Created {created_str}")
            
            if i == self.current_host_index:
                self.terminal.move_cursor(row + 1, 6)
                self.terminal.set_color(status_color)
                print(f"Status: {status}")
                if status == 'error' and 'error' in host:
                    self.terminal.move_cursor(row + 2, 6)
                    self.terminal.set_color('red')
                    error_msg = host['error'][:cols-10] + "..." if len(host['error']) > cols-10 else host['error']
                    print(f"Error: {error_msg}")
            
            self.terminal.reset_colors()
    
    async def _render_main_menu(self):
        """Render the main menu."""
        rows, cols = self.terminal.get_terminal_size()
        
        # Connection status
        self.terminal.move_cursor(4, 2)
        if self.current_host:
            self.terminal.set_color('bright_green')
            print(f"Connected to PID {self.current_host.get('pid', '?')}")
        else:
            self.terminal.set_color('yellow')
            print("Not connected to any host")
        
        # Menu title
        self.terminal.move_cursor(6, 2)
        self.terminal.set_color('bright_green', bold=True)
        print("Select an option:")
        
        # Calculate available space for menu items
        # Reserve space for header (6 rows), scroll indicators (1 row each), and footer (3 rows)
        content_rows = rows - 12  # More conservative calculation
        
        # Adjust scroll offset to ensure current selection is visible
        if self.current_menu_index < self.scroll_offset:
            self.scroll_offset = self.current_menu_index
        elif self.current_menu_index >= self.scroll_offset + content_rows:
            self.scroll_offset = self.current_menu_index - content_rows + 1
        
        # Ensure scroll offset doesn't go beyond bounds
        max_scroll = max(0, len(self.main_menu_items) - content_rows)
        self.scroll_offset = min(self.scroll_offset, max_scroll)
        
        # Show up arrow if there are items above
        if self.scroll_offset > 0:
            self.terminal.move_cursor(7, 2)
            self.terminal.set_color('bright_yellow', bold=True)
            print("↑ ↑ ↑  More items above  ↑ ↑ ↑")
            self.terminal.reset_colors()
        else:
            # Clear the line if no scroll indicator needed
            self.terminal.move_cursor(7, 1)
            print(' ' * (cols - 1))
        
        # Clear the menu content area
        menu_start_row = 8
        for i in range(content_rows):
            self.terminal.move_cursor(menu_start_row + i, 1)
            print(' ' * (cols - 1))  # Clear entire line
        
        # Menu items with scrolling
        visible_items = self.main_menu_items[self.scroll_offset:self.scroll_offset + content_rows]
        for display_idx, (menu_idx, (name, cmd, description)) in enumerate(zip(
            range(self.scroll_offset, min(len(self.main_menu_items), self.scroll_offset + content_rows)),
            visible_items
        )):
            row = menu_start_row + display_idx
            self.terminal.move_cursor(row, 4)
            
            # Highlight current selection
            if menu_idx == self.current_menu_index:
                self.terminal.set_color('black', 'bright_cyan', bold=True)
                print(f"► {name}")
                self.terminal.move_cursor(row, 6 + len(name) + 2)
                self.terminal.set_color('cyan')
                print(f"- {description}")
            else:
                self.terminal.set_color('white')
                print(f"  {name}")
                self.terminal.move_cursor(row, 6 + len(name) + 2)
                self.terminal.set_color('bright_black')
                print(f"- {description}")
            
            self.terminal.reset_colors()
        
        # Show down arrow if there are more items below
        down_arrow_row = menu_start_row + content_rows
        if self.scroll_offset + content_rows < len(self.main_menu_items):
            self.terminal.move_cursor(down_arrow_row, 2)
            self.terminal.set_color('bright_yellow', bold=True)
            print("↓ ↓ ↓  More items below  ↓ ↓ ↓")
            self.terminal.reset_colors()
        else:
            # Clear the line if no scroll indicator needed
            self.terminal.move_cursor(down_arrow_row, 1)
            print(' ' * (cols - 1))
    
    async def _render_tree_view(self):
        """Render the tree navigation view."""
        if not self.tree_root:
            self.terminal.move_cursor(4, 2)
            self.terminal.set_color('yellow')
            print("Loading data...")
            self.terminal.reset_colors()
            return
        
        rows, cols = self.terminal.get_terminal_size()
        content_rows = rows - 6
        
        # Render tree
        visible_nodes = self._get_visible_tree_nodes()
        
        for i, (node, depth, is_current) in enumerate(visible_nodes[self.scroll_offset:]):
            if i >= content_rows:
                break
            
            row = 4 + i
            col = 2 + (depth * 2)
            
            self.terminal.move_cursor(row, col)
            
            # Selection highlighting
            if is_current:
                self.terminal.set_color('black', 'bright_yellow', bold=True)
            else:
                self.terminal.reset_colors()
            
            # Tree structure symbols
            if node.children:
                expand_symbol = "▼" if node.is_expanded else "▶"
                tree_line = f"{expand_symbol} "
            else:
                tree_line = "  "
            
            # Node icon and label
            if node.is_editable:
                icon = "📝"
            elif node.node_type == "command":
                icon = "⚡"
            elif node.node_type == "action":
                icon = "🔧"
            elif node.children:
                icon = "📁"
            else:
                icon = "📄"
            
            display_text = f"{tree_line}{icon} {node.label}"
            
            # Truncate if too long
            max_width = cols - col - 2
            if len(display_text) > max_width:
                display_text = display_text[:max_width-3] + "..."
            
            print(display_text)
            self.terminal.reset_colors()
    
    def _get_visible_tree_nodes(self) -> List[Tuple[TreeNode, int, bool]]:
        """Get list of visible tree nodes with depth and selection info."""
        visible = []
        
        def traverse(node: TreeNode, depth: int):
            is_current = node == self.current_tree_node
            visible.append((node, depth, is_current))
            
            if node.is_expanded:
                for child in node.children:
                    traverse(child, depth + 1)
        
        if self.tree_root:
            for child in self.tree_root.children:
                traverse(child, 0)
        
        return visible
    
    async def _render_detail_view(self):
        """Render detailed view of selected node."""
        if not self.current_tree_node or not self.current_tree_node.data:
            self.terminal.move_cursor(4, 2)
            self.terminal.set_color('yellow')
            print("No data to display")
            self.terminal.reset_colors()
            return
        
        rows, cols = self.terminal.get_terminal_size()
        
        # Node info
        self.terminal.move_cursor(4, 2)
        self.terminal.set_color('bright_green', bold=True)
        print(f"Details: {self.current_tree_node.label}")
        
        # Data display
        data_str = json.dumps(self.current_tree_node.data, indent=2, default=str)
        data_lines = data_str.split('\n')
        
        content_rows = rows - 8
        
        for i, line in enumerate(data_lines[self.scroll_offset:]):
            if i >= content_rows:
                break
            
            row = 6 + i
            self.terminal.move_cursor(row, 2)
            
            # Syntax highlighting for JSON
            if line.strip().startswith('"') and ':' in line:
                self.terminal.set_color('cyan')
            elif line.strip() in ['{', '}', '[', ']']:
                self.terminal.set_color('bright_white', bold=True)
            else:
                self.terminal.set_color('white')
            
            # Truncate long lines
            if len(line) > cols - 4:
                line = line[:cols-7] + "..."
            
            print(line)
        
        self.terminal.reset_colors()
    
    async def _render_edit_mode(self):
        """Render edit mode interface."""
        rows, cols = self.terminal.get_terminal_size()
        
        # Edit header
        self.terminal.move_cursor(4, 2)
        self.terminal.set_color('bright_yellow', bold=True)
        print(f"Editing: {self.current_tree_node.label if self.current_tree_node else 'Unknown'}")
        
        # Current value
        if self.current_tree_node and self.current_tree_node.data is not None:
            self.terminal.move_cursor(6, 2)
            self.terminal.set_color('white')
            print("Original value:")
            self.terminal.move_cursor(7, 4)
            self.terminal.set_color('bright_black')
            original = json.dumps(self.current_tree_node.data, default=str)
            if len(original) > cols - 8:
                original = original[:cols-11] + "..."
            print(original)
        
        # Edit buffer
        self.terminal.move_cursor(9, 2)
        self.terminal.set_color('white')
        print("New value:")
        
        self.terminal.move_cursor(10, 2)
        self.terminal.set_color('black', 'white')
        edit_display = self.edit_buffer
        if len(edit_display) > cols - 6:
            edit_display = edit_display[:cols-9] + "..."
        print(f" {edit_display} ")
        
        # Instructions
        self.terminal.move_cursor(12, 2)
        self.terminal.set_color('bright_cyan')
        print("Type new value, then:")
        self.terminal.move_cursor(13, 4)
        print("• Ctrl+S: Save changes")
        self.terminal.move_cursor(14, 4)
        print("• Esc: Cancel editing")
        
        self.terminal.reset_colors()
    
    async def _render_footer(self):
        """Render footer with controls and status."""
        rows, cols = self.terminal.get_terminal_size()
        
        # Status message
        if self.status_message:
            self.terminal.move_cursor(rows - 2, 2)
            color = 'bright_red' if any(word in self.status_message.lower() for word in ['error', 'failed', 'not']) else 'bright_green'
            self.terminal.set_color(color)
            status_display = self.status_message
            if len(status_display) > cols - 4:
                status_display = status_display[:cols-7] + "..."
            print(status_display)
        
        # Controls
        self.terminal.move_cursor(rows - 1, 2)
        self.terminal.set_color('bright_black')
        
        if self.mode == NavigationMode.HOST_SELECTION:
            controls = "↑↓: Navigate • Enter: Connect • R: Refresh • Q: Quit"
        elif self.mode == NavigationMode.MAIN_MENU:
            controls = "↑↓: Navigate • Enter: Select • Q: Quit"
        elif self.mode == NavigationMode.TREE_VIEW:
            controls = "↑↓: Navigate • →: Expand • ←: Collapse • Enter: Details • E: Edit • B: Back • R: Refresh • Q: Quit"
        elif self.mode == NavigationMode.DETAIL_VIEW:
            controls = "↑↓: Scroll • E: Edit • B: Back • Q: Quit"
        elif self.mode == NavigationMode.EDIT_MODE:
            controls = "Type to edit • Ctrl+S: Save • Esc: Cancel"
        else:
            controls = "Q: Quit"
        
        if len(controls) > cols - 4:
            controls = controls[:cols-7] + "..."
        
        print(controls)
        self.terminal.reset_colors()
    
    async def _handle_input(self, key: str):
        """Handle keyboard input based on current mode."""
        if key in ['q', 'Q'] and self.mode != NavigationMode.EDIT_MODE:
            self.running = False
            return
        
        if key == 'CTRL_C':
            self.running = False
            return
        
        if self.mode == NavigationMode.HOST_SELECTION:
            await self._handle_host_selection_input(key)
        elif self.mode == NavigationMode.MAIN_MENU:
            await self._handle_main_menu_input(key)
        elif self.mode == NavigationMode.TREE_VIEW:
            await self._handle_tree_view_input(key)
        elif self.mode == NavigationMode.DETAIL_VIEW:
            await self._handle_detail_view_input(key)
        elif self.mode == NavigationMode.EDIT_MODE:
            await self._handle_edit_mode_input(key)
    
    async def _handle_host_selection_input(self, key: str):
        """Handle input in host selection mode."""
        if key == 'ESC[A':  # Up arrow
            self.current_host_index = max(0, self.current_host_index - 1)
        elif key == 'ESC[B':  # Down arrow
            self.current_host_index = min(len(self.available_hosts) - 1, self.current_host_index + 1)
        elif key == '\r' or key == '\n':  # Enter
            await self._select_host()
        elif key in ['r', 'R']:
            await self._discover_hosts()
    
    async def _handle_main_menu_input(self, key: str):
        """Handle input in main menu mode."""
        if key == 'ESC[A':  # Up arrow
            self.current_menu_index = max(0, self.current_menu_index - 1)
        elif key == 'ESC[B':  # Down arrow
            self.current_menu_index = min(len(self.main_menu_items) - 1, self.current_menu_index + 1)
        elif key == '\r' or key == '\n':  # Enter
            await self._execute_menu_selection()
    
    async def _handle_tree_view_input(self, key: str):
        """Handle input in tree view mode."""
        if key == 'ESC[A':  # Up arrow
            await self._navigate_tree_up()
        elif key == 'ESC[B':  # Down arrow
            await self._navigate_tree_down()
        elif key == 'ESC[C':  # Right arrow
            await self._expand_current_node()
        elif key == 'ESC[D':  # Left arrow
            await self._collapse_current_node()
        elif key == '\r' or key == '\n':  # Enter
            await self._view_node_details()
        elif key in ['e', 'E']:
            await self._edit_current_node()
        elif key in ['b', 'B']:
            self.mode = NavigationMode.MAIN_MENU
            self.status_message = ""
        elif key in ['r', 'R']:
            await self._refresh_current_data()
    
    async def _handle_detail_view_input(self, key: str):
        """Handle input in detail view mode."""
        if key == 'ESC[A':  # Up arrow
            self.scroll_offset = max(0, self.scroll_offset - 1)
        elif key == 'ESC[B':  # Down arrow
            self.scroll_offset += 1
        elif key in ['e', 'E']:
            await self._edit_current_node()
        elif key in ['b', 'B']:
            self.mode = NavigationMode.TREE_VIEW
    
    async def _handle_edit_mode_input(self, key: str):
        """Handle input in edit mode."""
        if key == '\x1b':  # Escape
            self.mode = NavigationMode.DETAIL_VIEW
            self.edit_buffer = ""
        elif key == '\x13':  # Ctrl+S
            await self._save_edit()
        elif key == '\x08' or key == '\x7f':  # Backspace
            if self.edit_buffer:
                self.edit_buffer = self.edit_buffer[:-1]
        elif len(key) == 1 and ord(key) >= 32:  # Printable characters
            self.edit_buffer += key
    
    async def _select_host(self):
        """Select and connect to a host."""
        if not self.available_hosts or self.current_host_index >= len(self.available_hosts):
            return
        
        host_info = self.available_hosts[self.current_host_index]
        success = await self._connect_to_host(host_info)
        
        if success:
            self.mode = NavigationMode.MAIN_MENU
            self.current_menu_index = 0
    
    async def _execute_menu_selection(self):
        """Execute the currently selected main menu item."""
        if self.current_menu_index >= len(self.main_menu_items):
            return
        
        name, cmd, description = self.main_menu_items[self.current_menu_index]
        
        if cmd == "quit":
            self.running = False
            return
        elif cmd == "switch_host":
            await self._discover_hosts()
            self.mode = NavigationMode.HOST_SELECTION
            self.current_host_index = 0
            return
        
        # Load data for the selected endpoint
        self.status_message = f"Loading {name}..."
        try:
            data = await self._fetch_endpoint_data(cmd)
            self.tree_root = await self._build_tree_from_data(data, name, cmd)
            self.current_tree_node = self.tree_root.children[0] if self.tree_root.children else None
            self.mode = NavigationMode.TREE_VIEW
            self.scroll_offset = 0
            self.status_message = f"Loaded {name}"
            self._current_endpoint = cmd  # Track for refresh
        except Exception as e:
            self.status_message = f"Error loading {name}: {str(e)}"
    
    async def _fetch_endpoint_data(self, endpoint: str) -> Dict[str, Any]:
        """Fetch data from the specified endpoint via IPC."""
        if not self.current_host:
            raise ConnectionError("Not connected to any host")
        
        response = await self.executor.execute_command(endpoint)
        
        if response.get("error"):
            raise RuntimeError(response["error"])
        
        return response.get("result", {})
    
    async def _build_tree_from_data(self, data: Any, root_label: str, endpoint: str) -> TreeNode:
        """Build a tree structure from the data."""
        root = TreeNode("root", root_label)
        
        def build_node(obj: Any, label: str, parent: TreeNode, path: str = "", depth: int = 0) -> TreeNode:
            node = TreeNode(path or label, label, obj, parent=parent)
            
            # Determine if this is a writable endpoint based on the data structure and endpoint type
            if isinstance(obj, dict):
                node.children = []
                for key, value in obj.items():
                    child_path = f"{path}.{key}" if path else key
                    child = build_node(value, key, node, child_path, depth + 1)
                    node.children.append(child)
                    
                # Special handling for different endpoints
                if endpoint in ["agents", "spaces", "adapters"] and depth <= 2:
                    # Make certain configuration fields editable
                    for child in node.children:
                        if child.label in ["name", "description", "enabled", "timeout", "max_retries"]:
                            if isinstance(child.data, (str, int, float, bool)):
                                child.is_editable = True
                                child.write_endpoint = f"write_{endpoint}"
                                child.command_path = f"{endpoint}.{child.id}"
                            
            elif isinstance(obj, list) and obj:
                node.children = []
                for i, item in enumerate(obj):
                    child_path = f"{path}[{i}]" if path else f"[{i}]"
                    child = build_node(item, f"Item {i}", node, child_path, depth + 1)
                    node.children.append(child)
            else:
                # Leaf node - make some editable based on context
                if isinstance(obj, (str, int, float, bool)) or obj is None:
                    # Only make certain fields editable for safety
                    if (parent and parent.label in ["configuration", "settings", "metadata"] or
                        label in ["name", "description", "enabled", "timeout", "max_retries", "log_level"]):
                        node.is_editable = True
                        node.write_endpoint = f"write_{endpoint}"
                        node.command_path = path
            
            return node
        
        if isinstance(data, dict):
            for key, value in data.items():
                child = build_node(value, key, root, key, 0)
                root.children.append(child)
        else:
            child = build_node(data, "Data", root, "data", 0)
            root.children.append(child)
        
        # Auto-expand first level
        for child in root.children:
            child.is_expanded = True
        
        return root
    
    async def _navigate_tree_up(self):
        """Navigate up in the tree view."""
        visible_nodes = self._get_visible_tree_nodes()
        if not visible_nodes:
            return
        
        current_index = -1
        for i, (node, _, is_current) in enumerate(visible_nodes):
            if is_current:
                current_index = i
                break
        
        if current_index > 0:
            self.current_tree_node = visible_nodes[current_index - 1][0]
            
            # Adjust scroll offset if needed
            if current_index - 1 < self.scroll_offset:
                self.scroll_offset = max(0, current_index - 1)
    
    async def _navigate_tree_down(self):
        """Navigate down in the tree view."""
        visible_nodes = self._get_visible_tree_nodes()
        if not visible_nodes:
            return
        
        current_index = -1
        for i, (node, _, is_current) in enumerate(visible_nodes):
            if is_current:
                current_index = i
                break
        
        if current_index < len(visible_nodes) - 1:
            self.current_tree_node = visible_nodes[current_index + 1][0]
            
            # Adjust scroll offset if needed
            rows, _ = self.terminal.get_terminal_size()
            content_rows = rows - 6
            if current_index + 1 >= self.scroll_offset + content_rows:
                self.scroll_offset = current_index + 1 - content_rows + 1
    
    async def _expand_current_node(self):
        """Expand the current tree node."""
        if self.current_tree_node and self.current_tree_node.children:
            self.current_tree_node.is_expanded = True
    
    async def _collapse_current_node(self):
        """Collapse the current tree node."""
        if self.current_tree_node and self.current_tree_node.children:
            self.current_tree_node.is_expanded = False
    
    async def _view_node_details(self):
        """Switch to detail view for the current node."""
        if self.current_tree_node:
            if self.current_tree_node.node_type == "command":
                # Execute command node
                await self._execute_command_node()
            else:
                self.mode = NavigationMode.DETAIL_VIEW
                self.scroll_offset = 0
    
    async def _execute_command_node(self):
        """Execute a command node."""
        if not self.current_tree_node or not self.current_tree_node.command_path:
            return
        
        try:
            self.status_message = f"Executing {self.current_tree_node.label}..."
            
            response = await self.executor.execute_command(
                self.current_tree_node.command_path,
                self.current_tree_node.data if isinstance(self.current_tree_node.data, dict) else {}
            )
            
            if response.get("error"):
                self.status_message = f"Command failed: {response['error']}"
            else:
                self.status_message = f"Command executed: {self.current_tree_node.label}"
                # Update the node with response data
                self.current_tree_node.data = response.get("result", "Success")
                
        except Exception as e:
            self.status_message = f"Execution error: {str(e)}"
    
    async def _edit_current_node(self):
        """Enter edit mode for the current node."""
        if self.current_tree_node and self.current_tree_node.is_editable:
            self.mode = NavigationMode.EDIT_MODE
            self.edit_buffer = str(self.current_tree_node.data) if self.current_tree_node.data is not None else ""
        else:
            self.status_message = "This item is not editable"
    
    async def _save_edit(self):
        """Save the current edit via IPC."""
        if not self.current_tree_node:
            return
        
        try:
            # Parse the new value
            try:
                new_value = json.loads(self.edit_buffer)
            except json.JSONDecodeError:
                # If not JSON, try to infer the type based on the original value
                original_type = type(self.current_tree_node.data)
                if original_type == bool:
                    new_value = self.edit_buffer.lower() in ('true', '1', 'yes', 'on')
                elif original_type == int:
                    new_value = int(self.edit_buffer)
                elif original_type == float:
                    new_value = float(self.edit_buffer)
                else:
                    new_value = self.edit_buffer
            
            # Send write command via IPC
            if self.current_tree_node.write_endpoint and self.current_tree_node.command_path:
                self.status_message = "Saving changes..."
                
                write_args = {
                    "path": self.current_tree_node.command_path,
                    "value": new_value
                }
                
                response = await self.executor.execute_command(
                    self.current_tree_node.write_endpoint,
                    write_args
                )
                
                if response.get("error"):
                    self.status_message = f"Save failed: {response['error']}"
                else:
                    # Update the node with new value
                    self.current_tree_node.data = new_value
                    self.status_message = f"Saved {self.current_tree_node.label}"
                    self.mode = NavigationMode.DETAIL_VIEW
                    self.edit_buffer = ""
            else:
                # Local update only (for read-only data)
                self.current_tree_node.data = new_value
                self.status_message = f"Updated locally: {self.current_tree_node.label}"
                self.mode = NavigationMode.DETAIL_VIEW
                self.edit_buffer = ""
                
        except ValueError as e:
            self.status_message = f"Invalid value: {str(e)}"
        except Exception as e:
            self.status_message = f"Save error: {str(e)}"
    
    async def _refresh_current_data(self):
        """Refresh the current data view."""
        if self.mode == NavigationMode.TREE_VIEW and hasattr(self, '_current_endpoint'):
            try:
                self.status_message = "Refreshing..."
                data = await self._fetch_endpoint_data(self._current_endpoint)
                self.tree_root = await self._build_tree_from_data(data, self.tree_root.label, self._current_endpoint)
                
                # Try to maintain current selection
                if self.current_tree_node:
                    # Find node with same path
                    self.current_tree_node = self._find_node_by_id(self.tree_root, self.current_tree_node.id)
                
                if not self.current_tree_node and self.tree_root.children:
                    self.current_tree_node = self.tree_root.children[0]
                
                self.status_message = "Data refreshed"
            except Exception as e:
                self.status_message = f"Refresh failed: {str(e)}"
    
    def _find_node_by_id(self, root: TreeNode, node_id: str) -> Optional[TreeNode]:
        """Find a node by its ID in the tree."""
        if root.id == node_id:
            return root
        
        for child in root.children:
            found = self._find_node_by_id(child, node_id)
            if found:
                return found
        
        return None


async def main_ipc_tui(socket_path: str = None, timeout: float = 30.0):
    """
    Main IPC TUI entry point.
    
    Args:
        socket_path: Path to Unix socket (auto-discovered if None)
        timeout: Command timeout in seconds
    """
    tui = IPCTUIInspector(socket_path, timeout)
    try:
        await tui.start()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="IPC-based TUI Inspector for Connectome",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover and connect to running Connectome instance
  python -m host.modules.inspector.ipc_tui
  
  # Connect to specific socket
  python -m host.modules.inspector.ipc_tui --socket /tmp/connectome_inspector_1234.sock
  
  # Set timeout
  python -m host.modules.inspector.ipc_tui --timeout 60
        """.strip()
    )
    
    parser.add_argument(
        '--socket', '-s',
        type=str,
        help='Path to Unix socket (auto-discovered if not specified)'
    )
    
    parser.add_argument(
        '--timeout', '-t',
        type=float,
        default=30.0,
        help='Command timeout in seconds (default: 30.0)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
    else:
        # Minimal logging to avoid interfering with TUI
        logging.basicConfig(level=logging.WARNING)
    
    try:
        asyncio.run(main_ipc_tui(args.socket, args.timeout))
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)