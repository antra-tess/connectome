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
import subprocess
import tempfile
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
    drill_down_endpoint: Optional[str] = None  # For nodes that support detail drilling


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
        self.search_query = ""
        
        # Detail view state
        self._detail_tree_root = None
        self._detail_tree_node_id = None
        self._detail_current_node = None
        
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
        
        # Pagination state
        self._pagination_cursor = None  # Cursor for next batch (after_facet_id or offset)
        self._pagination_limit = 20  # Default limit
        self._pagination_has_more = False
        self._pagination_loading = False  # Prevent recursive loading
        self._last_scroll_direction = None  # Track last scroll direction for pagination
        self._rendering = False  # Prevent tree modifications during rendering
        
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
                
                # Check for pagination after input handling (safe from recursion)
                if self._last_scroll_direction == "down":
                    await self._check_pagination_on_scroll(self._last_scroll_direction)
                self._last_scroll_direction = None  # Reset after check
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self.status_message = f"Internal error: {str(e)}"
                await asyncio.sleep(0.1)
    
    async def _render_screen(self):
        """Render the current screen based on mode."""
        self._rendering = True
        try:
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
            
            # Render footer
            await self._render_footer()
        finally:
            self._rendering = False
    
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
        self._render_tree_generic(is_detail_mode=False)
    
    def _get_visible_tree_nodes(self) -> List[Tuple[TreeNode, int, bool]]:
        """Get list of visible tree nodes with depth and selection info."""
        return self._get_visible_nodes(is_detail_mode=False)
    
    def _get_tree_state(self, is_detail_mode: bool) -> Tuple[TreeNode, TreeNode, int]:
        """Get the current tree state (root, current_node, start_row) for the given mode."""
        if is_detail_mode:
            return (self._detail_tree_root, self._detail_current_node, 6)
        else:
            return (self.tree_root, self.current_tree_node, 4)
    
    def _set_current_node(self, node: TreeNode, is_detail_mode: bool):
        """Set the current node for the given mode."""
        if is_detail_mode:
            self._detail_current_node = node
        else:
            self.current_tree_node = node
    
    def _get_visible_nodes(self, is_detail_mode: bool) -> List[Tuple[TreeNode, int, bool]]:
        """Get list of visible tree nodes with depth and selection info for either mode."""
        visible = []
        tree_root, current_node, _ = self._get_tree_state(is_detail_mode)
        visited = set()  # Prevent infinite recursion from circular references
        
        def traverse(node: TreeNode, depth: int):
            # Prevent infinite recursion
            if node.id in visited or depth > 50:  # Max depth safety limit
                return
            visited.add(node.id)
            
            is_current = node == current_node
            visible.append((node, depth, is_current))
            
            if node.is_expanded:
                for child in node.children:
                    traverse(child, depth + 1)
            
            visited.remove(node.id)  # Allow same node at different levels
        
        if tree_root:
            for child in tree_root.children:
                traverse(child, 0)
        
        return visible
    
    def _render_tree_generic(self, is_detail_mode: bool):
        """Generic tree rendering for both tree view and detail view modes."""
        tree_root, current_node, start_row = self._get_tree_state(is_detail_mode)
        
        if not tree_root:
            self.terminal.move_cursor(start_row, 2)
            self.terminal.set_color('yellow')
            print("Loading data..." if not is_detail_mode else "No data to display")
            self.terminal.reset_colors()
            return
        
        rows, cols = self.terminal.get_terminal_size()
        content_rows = rows - (8 if is_detail_mode else 6)
        
        # Render tree
        visible_nodes = self._get_visible_nodes(is_detail_mode)
        
        for i, (node, depth, is_current) in enumerate(visible_nodes[self.scroll_offset:]):
            if i >= content_rows:
                break
            
            row = start_row + i
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
            elif not is_detail_mode and node.node_type == "command":
                icon = "⚡"
            elif not is_detail_mode and node.node_type == "action":
                icon = "🔧"
            elif node.children:
                icon = "📁"
            else:
                icon = "📄"
            
            # For leaf nodes, show the value alongside the key
            if not node.children:
                # Format the value for display
                value_str = self._format_value_for_display(node.data)
                display_text = f"{tree_line}{icon} {node.label}: {value_str}"
            else:
                display_text = f"{tree_line}{icon} {node.label}"
            
            # Add detail inspection indicator for nodes that support meaningful detail views (only in tree view)
            # Exclude editable nodes since they should open the editor directly, not inspection view
            if not is_detail_mode and (node.node_type in ["command", "action"] or node.drill_down_endpoint) and not node.is_editable:
                display_text += " [Enter - 🔎]"
            
            # Truncate if too long
            max_width = cols - col - 2
            if len(display_text) > max_width:
                display_text = display_text[:max_width-3] + "..."
            
            print(display_text)
            self.terminal.reset_colors()
    
    def _navigate_tree_generic(self, direction: str, is_detail_mode: bool):
        """Generic tree navigation for both modes."""
        visible_nodes = self._get_visible_nodes(is_detail_mode)
        if not visible_nodes:
            return
        
        current_index = -1
        for i, (node, _, is_current) in enumerate(visible_nodes):
            if is_current:
                current_index = i
                break
        
        # Handle case where no node is currently selected
        if current_index == -1:
            if visible_nodes:
                self._set_current_node(visible_nodes[0][0], is_detail_mode)
            return
        
        if direction == "up" and current_index > 0:
            self._set_current_node(visible_nodes[current_index - 1][0], is_detail_mode)
            # Adjust scroll offset if needed
            if current_index - 1 < self.scroll_offset:
                self.scroll_offset = max(0, current_index - 1)
        elif direction == "down" and current_index < len(visible_nodes) - 1:
            self._set_current_node(visible_nodes[current_index + 1][0], is_detail_mode)
            # Adjust scroll offset if needed
            rows, _ = self.terminal.get_terminal_size()
            content_rows = rows - (8 if is_detail_mode else 6)
            if current_index + 1 >= self.scroll_offset + content_rows:
                self.scroll_offset = current_index + 1 - content_rows + 1
    
    def _expand_collapse_node_generic(self, expand: bool, is_detail_mode: bool):
        """Generic expand/collapse for both modes."""
        _, current_node, _ = self._get_tree_state(is_detail_mode)
        if current_node and current_node.children:
            if expand:
                # Expand the current node
                current_node.is_expanded = expand
            else:
                # Collapse behavior: if expanded, collapse; if already collapsed, go to parent
                if current_node.is_expanded:
                    current_node.is_expanded = False
                elif current_node.parent and current_node.parent.id not in ["root", "detail_root", "facets_root", "events_root"]:
                    # Node is already collapsed, snap to parent
                    self._set_current_node(current_node.parent, is_detail_mode)
                    self._adjust_scroll_to_current_node(is_detail_mode)
                elif current_node.parent and current_node.parent.id in ["root", "detail_root", "facets_root", "events_root"]:
                    # Top-level node that would snap to root - instead collapse all siblings
                    self._collapse_siblings_of_node(current_node, is_detail_mode)
        elif current_node and not current_node.children:
            # Leaf node: snap to parent if collapse is requested
            if not expand:
                if current_node.parent and current_node.parent.id not in ["root", "detail_root", "facets_root", "events_root"]:
                    self._set_current_node(current_node.parent, is_detail_mode)
                    self._adjust_scroll_to_current_node(is_detail_mode)
                elif current_node.parent and current_node.parent.id in ["root", "detail_root", "facets_root", "events_root"]:
                    # Top-level leaf node - collapse all siblings
                    self._collapse_siblings_of_node(current_node, is_detail_mode)
    
    def _adjust_scroll_to_current_node(self, is_detail_mode: bool):
        """Adjust scroll offset to ensure current node is visible."""
        visible_nodes = self._get_visible_nodes(is_detail_mode)
        if not visible_nodes:
            return
        
        tree_root, current_node, _ = self._get_tree_state(is_detail_mode)
        if not current_node:
            return
        
        # Find the index of the current node in visible nodes
        current_index = -1
        for i, (node, _, is_current) in enumerate(visible_nodes):
            if node == current_node:
                current_index = i
                break
        
        if current_index == -1:
            return
        
        # Adjust scroll offset to ensure current node is visible
        rows, _ = self.terminal.get_terminal_size()
        content_rows = rows - (8 if is_detail_mode else 6)
        
        # If current node is above visible area, scroll up
        if current_index < self.scroll_offset:
            self.scroll_offset = max(0, current_index)
        # If current node is below visible area, scroll down
        elif current_index >= self.scroll_offset + content_rows:
            self.scroll_offset = current_index - content_rows + 1
        
        # Ensure scroll offset doesn't exceed bounds
        max_scroll = max(0, len(visible_nodes) - content_rows)
        self.scroll_offset = min(self.scroll_offset, max_scroll)
    
    def _collapse_siblings_of_node(self, current_node: TreeNode, is_detail_mode: bool):
        """Collapse all siblings of the current node, creating a 'focus' effect."""
        if not current_node or not current_node.parent:
            return
        
        # Collapse all siblings (other children of the same parent)
        for sibling in current_node.parent.children:
            if sibling != current_node and sibling.children:
                sibling.is_expanded = False
        
        # Do not expand the current node - back should never expand, only collapse siblings
        
        # Adjust scroll to ensure current node remains visible after sibling collapse
        self._adjust_scroll_to_current_node(is_detail_mode)
    
    def _format_value_for_display(self, value: Any) -> str:
        """Format a value for display in the tree view."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, str):
            # Escape and limit string length
            escaped = value.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            if len(escaped) > 50:
                return f'"{escaped[:47]}..."'
            return f'"{escaped}"'
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                return "[]"
            elif len(value) == 1:
                return f"[1 item]"
            else:
                return f"[{len(value)} items]"
        elif isinstance(value, dict):
            if len(value) == 0:
                return "{}"
            elif len(value) == 1:
                return "{1 key}"
            else:
                return f"{{{len(value)} keys}}"
        else:
            # For other types, convert to string and limit length
            str_val = str(value)
            if len(str_val) > 50:
                return f"{str_val[:47]}..."
            return str_val
    
    async def _render_detail_view(self):
        """Render detailed view of selected node as a foldable tree."""
        if not self.current_tree_node or not self.current_tree_node.data:
            self.terminal.move_cursor(4, 2)
            self.terminal.set_color('yellow')
            print("No data to display")
            self.terminal.reset_colors()
            return
        
        # Node info header
        self.terminal.move_cursor(4, 2)
        self.terminal.set_color('bright_green', bold=True)
        print(f"Details: {self.current_tree_node.label}")
        self.terminal.reset_colors()
        
        # Initialize detail tree if needed
        if not hasattr(self, '_detail_tree_root') or self._detail_tree_node_id != self.current_tree_node.id:
            await self._build_detail_tree()
        
        # Render using unified tree rendering
        self._render_tree_generic(is_detail_mode=True)
    
    
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
            controls = "↑↓: Navigate • →: Expand • ←: Collapse • E: Edit • B: Back • Q: Quit"
        else:
            controls = "Q: Quit"
        
        if len(controls) > cols - 4:
            controls = controls[:cols-7] + "..."
        
        print(controls)
        self.terminal.reset_colors()
    
    async def _handle_input(self, key: str):
        """Handle keyboard input based on current mode."""
        if key in ['q', 'Q']:
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
            self._reset_pagination_state()
        elif key in ['r', 'R']:
            await self._refresh_current_data()
    
    async def _handle_detail_view_input(self, key: str):
        """Handle input in detail view mode."""
        if key == 'ESC[A':  # Up arrow
            await self._navigate_detail_tree_up()
        elif key == 'ESC[B':  # Down arrow
            await self._navigate_detail_tree_down()
        elif key == 'ESC[C':  # Right arrow
            await self._expand_detail_node()
        elif key == 'ESC[D':  # Left arrow
            await self._collapse_detail_node()
        elif key in ['e', 'E']:
            await self._edit_detail_node()
        elif key in ['b', 'B']:
            self.mode = NavigationMode.TREE_VIEW
            # Reset scroll offset when going back
            self.scroll_offset = 0
            self._reset_pagination_state()
    
    
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
            self.tree_root = await self._build_tree_from_data(data, name, cmd, {})
            self.current_tree_node = self.tree_root.children[0] if self.tree_root.children else None
            self.mode = NavigationMode.TREE_VIEW
            self.scroll_offset = 0
            self.status_message = f"Loaded {name}"
            self._reset_pagination_state()
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
    
    async def _build_tree_from_data(self, data: Any, root_label: str, endpoint: str, context: Dict[str, Any] = None) -> TreeNode:
        """Build a tree structure from the data."""
        root = TreeNode("root", root_label)
        context = context or {}
        
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
                
                # Special drill-down endpoints for VEIL overview
                if endpoint == "veil" and depth == 1:
                    # Spaces in VEIL overview can drill down to detailed space VEIL
                    if label in ["spaces"] and isinstance(obj, dict):
                        for child in node.children:
                            # Each space can drill down to /veil/{space_id}
                            child.drill_down_endpoint = f"veil_space_{child.label}"
                
                # Add drill-down for individual spaces in VEIL overview to go directly to facets
                if endpoint == "veil" and depth == 1 and parent and parent.label == "spaces":
                    # This is an individual space under veil/spaces/  
                    # Add drill-down to see the facets list directly
                    space_id = label  # The space ID is the label at this level
                    node.drill_down_endpoint = f"veil_facets_{space_id}"
                
                # For VEIL space details, add facets drill-down
                if endpoint.startswith("veil_space") and depth == 1:
                    if label in ["facet_cache", "combined_cache_stats"] and isinstance(obj, dict):
                        # Extract space_id from the context
                        space_id = context.get('space_id', 'unknown')
                        node.drill_down_endpoint = f"veil_facets_{space_id}"
                
                # Add drill-down for individual spaces in timeline overview to go directly to events
                if endpoint == "timelines" and depth == 1 and parent and parent.label == "spaces":
                    # This is an individual space under timelines/spaces/
                    # Add drill-down to see the timeline events directly
                    space_id = label  # The space ID is the label at this level  
                    node.drill_down_endpoint = f"timeline_details_{space_id}"
                
                # Add drill-down for individual timelines: any node that is a direct child of a "timelines" container
                # This should work in both "timelines" (overview) and "timeline_details" (specific space timeline) views
                if endpoint in ["timelines", "timeline_details"] and parent and parent.label == "timelines":
                    # This is an individual timeline under any "timelines" container
                    timeline_id = label  # The timeline ID is the current label
                    
                    # For timeline_details endpoint, use the space_id from context instead of tree walking
                    if endpoint == "timeline_details" and context and "space_id" in context:
                        space_id = context["space_id"]
                        node.drill_down_endpoint = f"timeline_events_{space_id}_{timeline_id}"
                    else:
                        # Walk up the tree to find the space_id (for timelines overview)
                        space_node = parent.parent  # Parent is "timelines", parent.parent should be the space
                        if space_node:
                            space_id = space_node.label
                            node.drill_down_endpoint = f"timeline_events_{space_id}_{timeline_id}"
                
                # For other endpoints, add drill-down logic based on patterns
                if endpoint == "spaces" and depth == 1 and parent and parent.label == "details":
                    # Individual spaces under details can drill down to their timelines, veil, etc.
                    # But InnerSpaces (agent spaces) should drill down to agent details instead
                    if label.endswith("_inner_space"):
                        # This is an InnerSpace (agent space) - drill down to agent details
                        agent_id = label.replace("_inner_space", "")
                        node.drill_down_endpoint = f"agent_details_{agent_id}"
                    else:
                        # Regular Space - drill down to VEIL space data
                        node.drill_down_endpoint = f"veil_space_{label}"
                elif endpoint == "agents" and depth == 1 and parent and parent.label == "agents":
                    # Individual agents under the agents node can drill down to their inner space details
                    node.drill_down_endpoint = f"agent_details_{label}"
                            
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
    
    async def _build_facets_tree(self, facets_data: Dict[str, Any], root_label: str) -> TreeNode:
        """Build a simplified tree structure for facets drill-down without container metadata."""
        root = TreeNode("facets_root", root_label)
        
        # Get the facets array from the response
        facets = facets_data.get("facets", [])
        
        # Create nodes directly for each facet without any container structure
        for i, facet in enumerate(facets):
            # Use facet_id as the node ID if available, otherwise use index
            facet_id = facet.get("facet_id", f"facet_{i}")
            facet_type = facet.get("facet_type", "unknown")
            owner_id = facet.get("owner_element_id", "unknown")
            
            # Create a readable label for the facet
            label = f"{facet_id} ({facet_type})"
            
            # Build the facet node with expandable structure
            facet_node = TreeNode(facet_id, label, facet, parent=root)
            
            # Make facet data expandable by creating child nodes for facet properties
            if isinstance(facet, dict):
                facet_node.children = []
                for key, value in facet.items():
                    child_path = f"{facet_id}.{key}"
                    child = self._build_node_from_value(value, key, facet_node, child_path, 1)
                    facet_node.children.append(child)
                    
                # Auto-expand the facet to show its properties  
                facet_node.is_expanded = True
            
            root.children.append(facet_node)
        
        return root
    
    async def _build_timeline_events_tree(self, events_data: Dict[str, Any], root_label: str) -> TreeNode:
        """Build a simplified tree structure for timeline events drill-down without container metadata."""
        root = TreeNode("events_root", root_label)
        
        # Get the events array from the response
        events = events_data.get("events", [])
        
        # Create nodes directly for each event without any container structure
        for i, event in enumerate(events):
            # Use id as the node ID if available, otherwise use index
            event_id = event.get("id", f"event_{i}")
            payload = event.get("payload", {})
            event_type = payload.get("event_type", "unknown")
            timestamp = event.get("timestamp", 0)
            
            # Create a readable label for the event
            # Format timestamp for display
            import datetime
            try:
                dt = datetime.datetime.fromtimestamp(timestamp)
                time_str = dt.strftime("%H:%M:%S")
            except (ValueError, OSError):
                time_str = str(timestamp)[:10]  # Fallback to truncated timestamp
            
            label = f"{event_id} ({event_type}) at {time_str}"
            
            # Build the event node with expandable structure
            event_node = TreeNode(event_id, label, event, parent=root)
            
            # Make event data expandable by creating child nodes for event properties
            if isinstance(event, dict):
                event_node.children = []
                for key, value in event.items():
                    child_path = f"{event_id}.{key}"
                    child = self._build_node_from_value(value, key, event_node, child_path, 1)
                    event_node.children.append(child)
                    
                # Auto-expand the event to show its properties  
                event_node.is_expanded = True
            
            root.children.append(event_node)
        
        return root
    
    def _build_node_from_value(self, value: Any, label: str, parent: TreeNode, path: str, depth: int) -> TreeNode:
        """Build a tree node from a value, similar to the main build_node function."""
        node = TreeNode(path, label, value, parent=parent)
        
        if isinstance(value, dict):
            node.children = []
            for key, child_value in value.items():
                child_path = f"{path}.{key}"
                child = self._build_node_from_value(child_value, key, node, child_path, depth + 1)
                node.children.append(child)
        elif isinstance(value, list) and value:
            node.children = []
            for i, item in enumerate(value):
                child_path = f"{path}[{i}]"
                child = self._build_node_from_value(item, f"Item {i}", node, child_path, depth + 1)
                node.children.append(child)
        else:
            # Leaf node - make certain fields editable
            if isinstance(value, (str, int, float, bool)) or value is None:
                # Use same logic as main tree building method
                if (parent and parent.label in ["configuration", "settings", "metadata", "properties", "data", "payload"] or
                    label in ["name", "description", "enabled", "timeout", "max_retries", "log_level", 
                             "facet_type", "facet_id", "owner_element_id", "event_type", "timestamp",
                             "message", "content", "status", "priority", "category", "tags"]):
                    node.is_editable = True
                    # Use appropriate write endpoint based on the context
                    # Walk up the tree to find the root to determine the context
                    root_node = parent
                    while root_node and root_node.parent:
                        root_node = root_node.parent
                    
                    if root_node and root_node.id == "events_root":
                        node.write_endpoint = "update-timeline-event"
                    elif root_node and root_node.id == "facets_root":
                        node.write_endpoint = "update-veil-facet"
                    else:
                        # Fallback logic for other contexts
                        node.write_endpoint = "update-veil-facet"
                    node.command_path = path
        
        return node
    
    async def _navigate_tree_up(self):
        """Navigate up in the tree view."""
        self._last_scroll_direction = "up"
        self._navigate_tree_generic("up", is_detail_mode=False)
    
    async def _navigate_tree_down(self):
        """Navigate down in the tree view."""
        self._last_scroll_direction = "down"
        self._navigate_tree_generic("down", is_detail_mode=False)
    
    async def _expand_current_node(self):
        """Expand the current tree node."""
        self._expand_collapse_node_generic(expand=True, is_detail_mode=False)
    
    async def _collapse_current_node(self):
        """Collapse the current tree node."""
        self._expand_collapse_node_generic(expand=False, is_detail_mode=False)
    
    async def _view_node_details(self):
        """Switch to detail view for the current node or expand if it's just a regular tree node."""
        if self.current_tree_node:
            if self.current_tree_node.node_type == "command":
                # Execute command node
                await self._execute_command_node()
            elif self.current_tree_node.drill_down_endpoint:
                # Drill down to detailed endpoint
                await self._drill_down_to_endpoint()
            elif self.current_tree_node.is_editable:
                # For editable nodes, open the editor directly instead of detail view
                await self._edit_current_node()
            elif (self.current_tree_node.node_type in ["action"] or
                  not self.current_tree_node.children):
                # Only switch to detail view for action nodes or leaf nodes
                # that don't have drill-down endpoints
                self.mode = NavigationMode.DETAIL_VIEW
                self.scroll_offset = 0
                self._reset_pagination_state()
                # Reset detail tree state to force rebuild
                self._detail_tree_root = None
                self._detail_tree_node_id = None
            else:
                # For regular tree nodes with children, just expand them
                await self._expand_current_node()
    
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
    
    async def _drill_down_to_endpoint(self):
        """Drill down to a detailed endpoint for the current node."""
        if not self.current_tree_node or not self.current_tree_node.drill_down_endpoint:
            return
        
        try:
            drill_endpoint = self.current_tree_node.drill_down_endpoint
            
            # Parse the drill-down endpoint to get the actual endpoint name
            if drill_endpoint.startswith("veil_space_"):
                space_id = drill_endpoint.replace("veil_space_", "")
                endpoint_name = f"veil_space"
                endpoint_args = {"space_id": space_id}
                display_name = f"VEIL for Space {space_id}"
            elif drill_endpoint.startswith("veil_facets_"):
                space_id = drill_endpoint.replace("veil_facets_", "")
                endpoint_name = f"veil_facets"
                endpoint_args = {"space_id": space_id}
                display_name = f"VEIL Facets for Space {space_id}"
            elif drill_endpoint.startswith("timeline_details_"):
                space_id = drill_endpoint.replace("timeline_details_", "")
                endpoint_name = f"timeline_details"
                endpoint_args = {"space_id": space_id}
                display_name = f"Timeline Events for Space {space_id}"
            elif drill_endpoint.startswith("timeline_events_"):
                # Parse timeline_events_{space_id}_{timeline_id}
                # This is tricky because both space_id and timeline_id can contain underscores
                # We need to split from the end to get the timeline_id, then the rest is space_id
                remaining = drill_endpoint.replace("timeline_events_", "")
                parts = remaining.rsplit("_", 1)  # Split from the right to get the last part as timeline_id
                if len(parts) >= 2:
                    space_id = parts[0]
                    timeline_id = parts[1]
                    endpoint_name = f"timeline_details"
                    endpoint_args = {"space_id": space_id, "timeline_id": timeline_id}
                    display_name = f"Events for Timeline {timeline_id} in Space {space_id}"
                else:
                    # Fallback if parsing fails - treat the whole string as space_id
                    space_id = remaining
                    endpoint_name = f"timeline_details"
                    endpoint_args = {"space_id": space_id}
                    display_name = f"Timeline Events for Space {space_id}"
            elif drill_endpoint.startswith("agent_details_"):
                agent_id = drill_endpoint.replace("agent_details_", "")
                endpoint_name = f"agent_details" 
                endpoint_args = {"agent_id": agent_id}
                display_name = f"Agent Details for {agent_id}"
            else:
                # Generic case - use the drill_endpoint as-is
                endpoint_name = drill_endpoint
                endpoint_args = {}
                display_name = f"Details for {self.current_tree_node.label}"
            
            self.status_message = f"Loading {display_name}..."
            
            # Fetch data from the drill-down endpoint
            data = await self._fetch_drill_down_data(endpoint_name, endpoint_args)
            
            # Add pagination info to display name if available
            if endpoint_name in ["veil_facets", "timeline_details"]:
                # Handle different pagination response formats
                if endpoint_name == "veil_facets":
                    # VEIL facets uses 'summary' format
                    summary = data.get("summary", {})
                    if summary:
                        returned = summary.get("returned", 0)
                        total = summary.get("total_matching", 0)
                        limited = summary.get("limited", False)
                        if limited:
                            display_name += f" ({returned} of {total}, paginated)"
                        else:
                            display_name += f" ({returned} items)"
                elif endpoint_name == "timeline_details":
                    # Timeline details uses 'pagination' format
                    pagination = data.get("pagination", {})
                    if pagination:
                        returned = pagination.get("events_returned", 0)
                        has_more = pagination.get("has_more", False)
                        total_events = data.get("timeline_info", {}).get("total_events", 0)
                        if has_more:
                            display_name += f" ({returned} of {total_events}, paginated)"
                        else:
                            display_name += f" ({returned} items)"
            
            # Build new tree with the detailed data
            if endpoint_name == "veil_facets":
                # Special handling for facets - show only facets without container structure
                self.tree_root = await self._build_facets_tree(data, display_name)
                # Store space_id for refresh functionality
                self._current_space_id = endpoint_args.get("space_id", "unknown")
                # Initialize pagination state
                await self._initialize_pagination_state(data, endpoint_name)
            elif endpoint_name == "timeline_details" and endpoint_args.get("timeline_id"):
                # Special handling for timeline events - show events in flat list similar to facets
                self.tree_root = await self._build_timeline_events_tree(data, display_name)
                # Store space_id and timeline_id for refresh functionality
                self._current_space_id = endpoint_args.get("space_id", "unknown")
                self._current_timeline_id = endpoint_args.get("timeline_id", "unknown")
                # Initialize pagination state
                await self._initialize_pagination_state(data, endpoint_name)
            else:
                self.tree_root = await self._build_tree_from_data(data, display_name, endpoint_name, endpoint_args)
            
            self.current_tree_node = self.tree_root.children[0] if self.tree_root.children else None
            self.mode = NavigationMode.TREE_VIEW
            self.scroll_offset = 0
            self.status_message = f"Loaded {display_name}"
            self._current_endpoint = endpoint_name  # Track for refresh
            
        except Exception as e:
            # Check if it's an IPC buffer overflow and provide helpful message
            error_str = str(e)
            if "chunk is longer than limit" in error_str:
                self.status_message = f"Error: Too much data to display. Try reducing pagination limit."
            else:
                self.status_message = f"Error drilling down: {error_str}"
    
    async def _fetch_drill_down_data(self, endpoint_name: str, endpoint_args: Dict[str, str]) -> Dict[str, Any]:
        """Fetch data from a drill-down endpoint."""
        if not self.current_host:
            raise ConnectionError("Not connected to any host")
        
        # Map endpoint names to actual IPC command names
        if endpoint_name == "veil_space":
            command = "veil-space"
            command_args = endpoint_args
        elif endpoint_name == "veil_facets":
            command = "veil-facets"
            command_args = endpoint_args.copy()
            # Add conservative limit to avoid IPC overflow (tested safe threshold is 44, using 20 for safety)
            if 'limit' not in command_args:
                command_args['limit'] = 20
        elif endpoint_name == "timeline_details":
            command = "timeline-details"
            command_args = endpoint_args.copy()
            # Add conservative limit for timeline events to avoid IPC overflow (using 20 to match veil facets)
            # Use negative limit to get older events (reverse chronological order)
            if 'limit' not in command_args:
                command_args['limit'] = -20
        elif endpoint_name == "agent_details":
            command = "agent_details"
            command_args = endpoint_args
        else:
            # Fallback to using endpoint_name directly
            command = endpoint_name
            command_args = endpoint_args
        
        response = await self.executor.execute_command(command, command_args)
        
        if response.get("error"):
            raise RuntimeError(response["error"])
        
        return response.get("result", {})
    
    async def _edit_current_node(self):
        """Edit the current node using external editor."""
        if self.current_tree_node and self.current_tree_node.is_editable:
            await self._edit_with_external_editor(self.current_tree_node)
        else:
            self.status_message = "This item is not editable"
    
    async def _edit_with_external_editor(self, node: 'TreeNode'):
        """Edit a node using external editor."""
        if not node:
            return
            
        try:
            # Get the editor from environment variable, default to vi
            editor = os.environ.get('EDITOR', 'vi')
            
            # Prepare the content to edit
            if node.data is not None:
                # Pretty-format JSON if possible, otherwise use string representation
                try:
                    content = json.dumps(node.data, indent=2, default=str)
                except (TypeError, ValueError):
                    content = str(node.data)
            else:
                content = ""
            
            # Create a temporary file with appropriate extension
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            # Store terminal state and restore to normal mode
            self.terminal.restore_terminal()
            self.terminal.show_cursor()
            self.terminal.reset_colors()
            
            # Clear screen and show editing message
            print("\033[2J\033[H", end='')
            print(f"Opening {node.label} in {editor}...")
            print(f"Temporary file: {tmp_file_path}")
            print("Save and exit the editor to apply changes, or exit without saving to cancel.")
            print()
            
            # Launch the external editor
            result = subprocess.run([editor, tmp_file_path], cwd=os.getcwd())
            
            if result.returncode == 0:
                # Read the modified content
                try:
                    with open(tmp_file_path, 'r') as tmp_file:
                        modified_content = tmp_file.read().strip()
                    
                    if modified_content != content.strip():
                        # Content was changed, parse and save
                        await self._save_external_edit(node, modified_content)
                    else:
                        self.status_message = f"No changes made to {node.label}"
                except Exception as e:
                    self.status_message = f"Error reading edited file: {str(e)}"
            else:
                self.status_message = f"Editor exited with code {result.returncode}"
            
        except Exception as e:
            self.status_message = f"Editor error: {str(e)}"
        finally:
            # Clean up temporary file
            try:
                if 'tmp_file_path' in locals():
                    os.unlink(tmp_file_path)
            except OSError:
                pass
                
            # Restore terminal state
            self.terminal.setup_terminal()
            self.terminal.hide_cursor()
    
    async def _save_external_edit(self, node: 'TreeNode', content: str):
        """Save the content from external editor."""
        try:
            # Parse the new value
            try:
                new_value = json.loads(content)
            except json.JSONDecodeError:
                # If not JSON, try to infer the type based on the original value
                if node.data is not None:
                    original_type = type(node.data)
                    if original_type == bool:
                        new_value = content.lower() in ('true', '1', 'yes', 'on')
                    elif original_type == int:
                        new_value = int(content)
                    elif original_type == float:
                        new_value = float(content)
                    else:
                        new_value = content
                else:
                    new_value = content
            
            # Send write command via IPC
            if node.write_endpoint and node.command_path:
                self.status_message = "Saving changes..."
                
                # Construct arguments based on the write endpoint type
                if node.write_endpoint == "update-veil-facet":
                    write_args = await self._build_veil_facet_update_args(node, new_value)
                elif node.write_endpoint == "update-timeline-event":
                    write_args = await self._build_timeline_event_update_args(node, new_value)
                else:
                    # Fallback for unknown endpoints
                    write_args = {
                        "path": node.command_path,
                        "value": new_value
                    }
                
                if write_args:
                    response = await self.executor.execute_command(
                        node.write_endpoint,
                        write_args
                    )
                    
                    if response.get("error"):
                        error_msg = response['error']
                        if "Unknown command" in error_msg:
                            self.status_message = f"Command not available: {node.write_endpoint}. Host may need restart."
                        else:
                            self.status_message = f"Save failed: {error_msg}"
                    else:
                        # Update the node with new value
                        node.data = new_value
                        self.status_message = f"Saved {node.label}"
                else:
                    # Provide more detailed error information
                    tree_path = []
                    current = node
                    while current:
                        tree_path.append(f"{current.id}({current.label})")
                        current = current.parent
                    path_str = ' -> '.join(reversed(tree_path))
                    self.status_message = f"Could not determine update parameters. Path: {path_str[:100]}..."
            else:
                # Local update only (for read-only data)
                node.data = new_value
                self.status_message = f"Updated locally: {node.label}"
                
        except ValueError as e:
            self.status_message = f"Invalid value: {str(e)}"
        except Exception as e:
            self.status_message = f"Save error: {str(e)}"
    
    async def _build_veil_facet_update_args(self, node: 'TreeNode', new_value: Any) -> Optional[Dict[str, Any]]:
        """Build arguments for update-veil-facet command."""
        try:
            logger.debug(f"Building VEIL facet update args for node: {node.id} (label: {node.label})")
            
            # Extract space_id and facet_id from the tree structure
            # Walk up the tree to find the facet root and space info
            facet_node = node
            
            # Debug: log the tree path
            path_debug = []
            current = node
            while current:
                path_debug.append(f"{current.id}({current.label})")
                current = current.parent
            logger.debug(f"Tree path: {' -> '.join(reversed(path_debug))}")
            
            # Look for a facet node - could be direct parent of the field being edited
            # In VEIL facets, the structure is usually: facets_root -> facet_node -> properties -> field
            while facet_node and facet_node.parent:
                # Check if this node represents a facet (not the root and has facet-like data)
                if (facet_node.parent and 
                    facet_node.parent.id == "facets_root" and 
                    facet_node.id != "facets_root"):
                    # This should be the actual facet node
                    break
                facet_node = facet_node.parent
            
            if not facet_node or facet_node.id == "facets_root":
                logger.error(f"Could not find facet node in tree path")
                return None
            
            facet_id = facet_node.id
            logger.debug(f"Extracted facet_id: {facet_id}")
            
            # Extract space_id from the current context
            space_id = getattr(self, '_current_space_id', 'unknown')
            logger.debug(f"Space ID from context: {space_id}")
            
            if space_id == 'unknown':
                # Try to extract from the tree root label or other context
                tree_root = node
                while tree_root and tree_root.parent:
                    tree_root = tree_root.parent
                # Extract space_id from label if available
                if hasattr(tree_root, 'label'):
                    label = tree_root.label
                    logger.debug(f"Tree root label: {label}")
                    
                    # Try different patterns for space ID extraction
                    if 'Space' in label:
                        parts = label.split()
                        for i, part in enumerate(parts):
                            if part == 'Space' and i + 1 < len(parts):
                                space_id = parts[i + 1]
                                logger.debug(f"Extracted space_id from 'Space X' pattern: {space_id}")
                                break
                    elif 'for Space' in label:
                        # Pattern: "VEIL Facets for Space space_id"
                        match = label.split('for Space ')
                        if len(match) > 1:
                            space_id = match[1].strip()
                            logger.debug(f"Extracted space_id from 'for Space X' pattern: {space_id}")
                    elif '_inner_space' in facet_id:
                        # Try to extract from facet_id pattern like "conversation_facet_AdapterOfPlatform_inner_space_12345"
                        space_id = facet_id.split('_inner_space')[0].replace('conversation_facet_', '').replace('_facet_', '_')
                        if space_id:
                            space_id = space_id + '_inner_space'
                            logger.debug(f"Extracted space_id from facet_id pattern: {space_id}")
            
            if space_id == 'unknown' or not space_id:
                logger.error(f"Could not determine space_id from context or labels")
                return None
                
            # Extract the field path relative to the facet
            # The node.id contains the full path, we need to remove the facet_id prefix
            field_path = node.id
            if field_path.startswith(facet_id + "."):
                field_path = field_path[len(facet_id + "."):]
            else:
                # Fallback to using the label if path parsing fails
                field_path = node.label
            
            logger.debug(f"Field path being updated: {field_path}")
            
            # Handle nested properties specially
            if field_path.startswith("properties."):
                # Extract the property name
                prop_name = field_path[len("properties."):]
                update_data = {
                    "properties": {prop_name: new_value}
                }
                logger.debug(f"Updating nested property: {prop_name}")
            else:
                # Direct field update
                update_data = {field_path: new_value}
                logger.debug(f"Updating direct field: {field_path}")
            
            result = {
                "space_id": space_id,
                "facet_id": facet_id,
                "update_data": update_data
            }
            logger.debug(f"Built VEIL facet update args: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error building veil facet update args: {e}", exc_info=True)
            return None
    
    async def _build_timeline_event_update_args(self, node: 'TreeNode', new_value: Any) -> Optional[Dict[str, Any]]:
        """Build arguments for update-timeline-event command."""
        try:
            logger.debug(f"Building timeline event update args for node: {node.id} (label: {node.label})")
            
            # Extract event_id from the tree structure
            event_node = node
            
            # Debug: log the tree path
            path_debug = []
            current = node
            while current:
                path_debug.append(f"{current.id}({current.label})")
                current = current.parent
            logger.debug(f"Tree path: {' -> '.join(reversed(path_debug))}")
            
            # Look for an event node - structure is usually: events_root -> event_node -> payload/properties -> field
            while event_node and event_node.parent:
                # Check if this node represents an event (not the root and has event-like data)
                if (event_node.parent and 
                    event_node.parent.id == "events_root" and 
                    event_node.id != "events_root"):
                    # This should be the actual event node
                    break
                event_node = event_node.parent
            
            if not event_node or event_node.id == "events_root":
                logger.error(f"Could not find event node in tree path")
                return None
                
            event_id = event_node.id
            logger.debug(f"Extracted event_id: {event_id}")
            
            # Extract space_id and timeline_id from current context
            space_id = getattr(self, '_current_space_id', None)
            timeline_id = getattr(self, '_current_timeline_id', None)
            logger.debug(f"Space ID from context: {space_id}, Timeline ID: {timeline_id}")
            
            result = {
                "event_id": event_id,
                "update_data": new_value,
                "space_id": space_id,
                "timeline_id": timeline_id
            }
            logger.debug(f"Built timeline event update args: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error building timeline event update args: {e}", exc_info=True)
            return None
    
    async def _refresh_current_data(self):
        """Refresh the current data view."""
        if self.mode == NavigationMode.TREE_VIEW and hasattr(self, '_current_endpoint'):
            try:
                self.status_message = "Refreshing..."
                
                if self._current_endpoint == "veil_facets":
                    # Special handling for facets refresh using drill-down data fetching
                    # Extract space_id from the current tree root label or stored context
                    space_id = getattr(self, '_current_space_id', 'unknown')
                    endpoint_args = {"space_id": space_id}
                    data = await self._fetch_drill_down_data(self._current_endpoint, endpoint_args)
                    self.tree_root = await self._build_facets_tree(data, self.tree_root.label)
                elif self._current_endpoint == "timeline_details" and hasattr(self, '_current_timeline_id'):
                    # Special handling for timeline events refresh
                    space_id = getattr(self, '_current_space_id', 'unknown')
                    timeline_id = getattr(self, '_current_timeline_id', 'unknown')
                    endpoint_args = {"space_id": space_id, "timeline_id": timeline_id}
                    data = await self._fetch_drill_down_data(self._current_endpoint, endpoint_args)
                    self.tree_root = await self._build_timeline_events_tree(data, self.tree_root.label)
                else:
                    data = await self._fetch_endpoint_data(self._current_endpoint)
                    # For refresh, we may not have the original context, so pass empty dict
                    self.tree_root = await self._build_tree_from_data(data, self.tree_root.label, self._current_endpoint, {})
                
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
    
    async def _build_detail_tree(self):
        """Build a tree structure for the detail view of the current node."""
        if not self.current_tree_node:
            return
        
        self._detail_tree_node_id = self.current_tree_node.id
        self._detail_tree_root = TreeNode("detail_root", f"Details: {self.current_tree_node.label}")
        
        # Build tree from the node's data
        def build_detail_node(obj: Any, label: str, parent: TreeNode, path: str = "", depth: int = 0) -> TreeNode:
            node = TreeNode(path or label, label, obj, parent=parent)
            
            if isinstance(obj, dict):
                node.children = []
                for key, value in obj.items():
                    child_path = f"{path}.{key}" if path else key
                    child = build_detail_node(value, key, node, child_path, depth + 1)
                    node.children.append(child)
                    
                # Make leaf string/number fields editable in detail view
                for child in node.children:
                    if not child.children and isinstance(child.data, (str, int, float, bool)) or child.data is None:
                        child.is_editable = True
                        # Set appropriate write endpoint for detail view
                        if hasattr(self, '_current_endpoint'):
                            if self._current_endpoint == "veil_facets":
                                child.write_endpoint = "update-veil-facet"
                            elif self._current_endpoint == "timeline_details":
                                child.write_endpoint = "update-timeline-event"
                            else:
                                child.write_endpoint = "update-veil-facet"  # Default fallback
                        else:
                            child.write_endpoint = "update-veil-facet"  # Default fallback
                        child.command_path = child.id
                        
            elif isinstance(obj, list) and obj:
                node.children = []
                for i, item in enumerate(obj):
                    child_path = f"{path}[{i}]" if path else f"[{i}]"
                    child = build_detail_node(item, f"Item {i}", node, child_path, depth + 1)
                    node.children.append(child)
            
            return node
        
        # Build the detail tree from current node's data
        if self.current_tree_node.data is not None:
            detail_child = build_detail_node(
                self.current_tree_node.data, 
                "Data", 
                self._detail_tree_root, 
                "data", 
                0
            )
            self._detail_tree_root.children.append(detail_child)
            
            # Auto-expand first few levels for better navigation
            detail_child.is_expanded = True
            for child in detail_child.children[:5]:  # Expand first 5 children
                child.is_expanded = True
        
        # Set initial selection to first expandable node or first child
        self._detail_current_node = None
        if self._detail_tree_root.children:
            first_child = self._detail_tree_root.children[0]
            if first_child.children:
                self._detail_current_node = first_child
            else:
                self._detail_current_node = first_child
    
    def _get_visible_detail_tree_nodes(self) -> List[Tuple[TreeNode, int, bool]]:
        """Get list of visible detail tree nodes with depth and selection info."""
        return self._get_visible_nodes(is_detail_mode=True)
    
    async def _navigate_detail_tree_up(self):
        """Navigate up in the detail tree view."""
        self._last_scroll_direction = "up"
        self._navigate_tree_generic("up", is_detail_mode=True)
    
    async def _navigate_detail_tree_down(self):
        """Navigate down in the detail tree view."""
        self._last_scroll_direction = "down"
        self._navigate_tree_generic("down", is_detail_mode=True)
    
    async def _expand_detail_node(self):
        """Expand the current detail tree node."""
        self._expand_collapse_node_generic(expand=True, is_detail_mode=True)
    
    async def _collapse_detail_node(self):
        """Collapse the current detail tree node."""
        self._expand_collapse_node_generic(expand=False, is_detail_mode=True)
    
    async def _edit_detail_node(self):
        """Edit the current detail node using external editor."""
        if self._detail_current_node and self._detail_current_node.is_editable:
            await self._edit_with_external_editor(self._detail_current_node)
        else:
            self.status_message = "This item is not editable"
    
    async def _check_pagination_on_scroll(self, direction: str):
        """Check if we need to load more data when scrolling near the end."""
        if self._pagination_loading or not self._pagination_has_more or self._rendering:
            return
        
        # Only trigger pagination for supported endpoints
        if not hasattr(self, '_current_endpoint') or self._current_endpoint not in ["veil_facets", "timeline_details"]:
            return
        
        # Only check when scrolling down
        if direction != "down":
            return
        
        # Get current tree state based on mode
        is_detail_mode = self.mode == NavigationMode.DETAIL_VIEW
        visible_nodes = self._get_visible_nodes(is_detail_mode)
        
        if not visible_nodes:
            return
        
        # Find current node index
        current_index = -1
        tree_root, current_node, _ = self._get_tree_state(is_detail_mode)
        
        for i, (node, _, is_current) in enumerate(visible_nodes):
            if is_current:
                current_index = i
                break
        
        # Check if we're near the end (within 5 items)
        if current_index >= len(visible_nodes) - 5:
            await self._load_more_paginated_data()
    
    async def _load_more_paginated_data(self):
        """Load additional paginated data and append to current tree."""
        if self._pagination_loading:
            return  # Prevent concurrent loading
        
        try:
            self._pagination_loading = True
            self.status_message = "Loading more..."
            
            # Determine current endpoint and args
            if self._current_endpoint == "veil_facets":
                space_id = getattr(self, '_current_space_id', 'unknown')
                endpoint_args = {
                    "space_id": space_id,
                    "limit": self._pagination_limit
                }
                # Add cursor if we have one
                if self._pagination_cursor:
                    endpoint_args["after_facet_id"] = self._pagination_cursor
                
                # Debug: log what we're sending
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Veil facets pagination request: args={endpoint_args}")
                    
                data = await self._fetch_drill_down_data(self._current_endpoint, endpoint_args)
                await self._append_facets_to_tree(data)
                
            elif self._current_endpoint == "timeline_details":
                space_id = getattr(self, '_current_space_id', 'unknown')
                timeline_id = getattr(self, '_current_timeline_id', 'unknown')
                endpoint_args = {
                    "space_id": space_id,
                    "timeline_id": timeline_id,
                    "limit": -self._pagination_limit  # Negative limit for older events
                }
                # Add cursor if we have one
                if self._pagination_cursor:
                    endpoint_args["offset"] = self._pagination_cursor
                
                # Debug: log what we're sending
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Timeline pagination request: args={endpoint_args}")
                    
                data = await self._fetch_drill_down_data(self._current_endpoint, endpoint_args)
                await self._append_timeline_events_to_tree(data)
            
            self.status_message = "Loaded more items"
            
        except Exception as e:
            self.status_message = f"Error loading more data: {str(e)}"
        finally:
            self._pagination_loading = False
    
    async def _append_facets_to_tree(self, data: Dict[str, Any]):
        """Append new facets to the existing tree."""
        if not self.tree_root or self.tree_root.id != "facets_root":
            return
        
        # Get the facets array from the response
        facets = data.get("facets", [])
        
        # Debug: log facet data structure
        import logging
        logger = logging.getLogger(__name__)
        if facets:
            first_facet = facets[0]
            last_facet = facets[-1]
            logger.debug(f"Facets structure - first: {first_facet.keys() if isinstance(first_facet, dict) else type(first_facet)}")
            logger.debug(f"First facet ID: {first_facet.get('facet_id') if isinstance(first_facet, dict) else 'N/A'}")
            logger.debug(f"Last facet ID: {last_facet.get('facet_id') if isinstance(last_facet, dict) else 'N/A'}")
        
        # Update pagination state
        summary = data.get("summary", {})
        self._pagination_has_more = summary.get("limited", False)
        
        # Set cursor to the next_cursor provided by the API (more reliable than manual extraction)
        if self._pagination_has_more:
            self._pagination_cursor = summary.get("next_cursor", None)
            # Debug: log cursor info
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Veil facets pagination: cursor={self._pagination_cursor}, has_more={self._pagination_has_more}, facets_count={len(facets)}")
            logger.debug(f"Summary: {summary}")
            if facets:
                logger.debug(f"Last facet ID: {facets[-1].get('facet_id') if facets else 'None'}")
        
        # Append new facets to existing tree
        for i, facet in enumerate(facets):
            facet_id = facet.get("facet_id", f"facet_{len(self.tree_root.children) + i}")
            facet_type = facet.get("facet_type", "unknown")
            
            # Ensure unique ID by including current tree size
            unique_facet_id = f"{facet_id}_batch_{len(self.tree_root.children)}_{i}"
            
            label = f"{facet_id} ({facet_type})"
            facet_node = TreeNode(unique_facet_id, label, facet, parent=self.tree_root)
            
            # Make facet data expandable
            if isinstance(facet, dict):
                facet_node.children = []
                for key, value in facet.items():
                    child_path = f"{unique_facet_id}.{key}"
                    child = self._build_node_from_value(value, key, facet_node, child_path, 1)
                    facet_node.children.append(child)
                facet_node.is_expanded = True
            
            self.tree_root.children.append(facet_node)
    
    async def _append_timeline_events_to_tree(self, data: Dict[str, Any]):
        """Append new timeline events to the existing tree."""
        if not self.tree_root or self.tree_root.id != "events_root":
            return
        
        # Get the events array from the response
        events = data.get("events", [])
        
        # Debug: log event data structure
        import logging
        logger = logging.getLogger(__name__)
        if events:
            first_event = events[0]
            last_event = events[-1]
            logger.debug(f"Events structure - first: {first_event.keys() if isinstance(first_event, dict) else type(first_event)}")
            logger.debug(f"First event timestamp: {first_event.get('timestamp') if isinstance(first_event, dict) else 'N/A'}")
            logger.debug(f"Last event timestamp: {last_event.get('timestamp') if isinstance(last_event, dict) else 'N/A'}")
        
        # Update pagination state
        pagination = data.get("pagination", {})
        self._pagination_has_more = pagination.get("has_more", False)
        
        # Set cursor to the last event timestamp for next batch
        # Since we're using negative limit (reverse chronological), the last event is the oldest
        if events and self._pagination_has_more:
            last_event = events[-1]  # This should be the oldest event
            self._pagination_cursor = last_event.get("timestamp", None)
            # Debug: log cursor info
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Timeline pagination: cursor={self._pagination_cursor}, has_more={self._pagination_has_more}, events_count={len(events)}")
            logger.debug(f"First event timestamp: {events[0].get('timestamp') if events else 'None'}")
            logger.debug(f"Last event timestamp: {last_event.get('timestamp') if isinstance(last_event, dict) else 'None'}")
        
        # Append new events to existing tree
        for i, event in enumerate(events):
            event_id = event.get("id", f"event_{len(self.tree_root.children) + i}")
            payload = event.get("payload", {})
            event_type = payload.get("event_type", "unknown")
            timestamp = event.get("timestamp", 0)
            
            # Ensure unique ID by including current tree size
            unique_event_id = f"{event_id}_batch_{len(self.tree_root.children)}_{i}"
            
            # Create readable label
            import datetime
            try:
                dt = datetime.datetime.fromtimestamp(timestamp)
                time_str = dt.strftime("%H:%M:%S")
            except (ValueError, OSError):
                time_str = str(timestamp)[:10]
            
            label = f"{event_id} ({event_type}) at {time_str}"
            event_node = TreeNode(unique_event_id, label, event, parent=self.tree_root)
            
            # Make event data expandable
            if isinstance(event, dict):
                event_node.children = []
                for key, value in event.items():
                    child_path = f"{unique_event_id}.{key}"
                    child = self._build_node_from_value(value, key, event_node, child_path, 1)
                    event_node.children.append(child)
                event_node.is_expanded = True
            
            self.tree_root.children.append(event_node)
    
    async def _initialize_pagination_state(self, data: Dict[str, Any], endpoint_name: str):
        """Initialize pagination state based on the response data."""
        if endpoint_name == "veil_facets":
            summary = data.get("summary", {})
            facets = data.get("facets", [])
            self._pagination_limit = 20  # Keep same limit as initial fetch
            self._pagination_has_more = summary.get("limited", False)
            # Set cursor to next_cursor provided by API if there are more items
            if self._pagination_has_more:
                self._pagination_cursor = summary.get("next_cursor", None)
            else:
                self._pagination_cursor = None
                
        elif endpoint_name == "timeline_details":
            pagination = data.get("pagination", {})
            events = data.get("events", [])
            self._pagination_limit = 20  # Keep same limit as initial fetch (will be made negative when used)
            self._pagination_has_more = pagination.get("has_more", False)
            # Set cursor to last event timestamp if there are more items
            if events and self._pagination_has_more:
                last_event = events[-1]
                self._pagination_cursor = last_event.get("timestamp", None)
            else:
                self._pagination_cursor = None
        else:
            # Reset pagination for non-paginated endpoints
            self._pagination_cursor = None
            self._pagination_limit = 20
            self._pagination_has_more = False
        
        self._pagination_loading = False
    
    def _reset_pagination_state(self):
        """Reset pagination state when switching views."""
        self._pagination_cursor = None
        self._pagination_limit = 20
        self._pagination_has_more = False
        self._pagination_loading = False
        self._last_scroll_direction = None


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