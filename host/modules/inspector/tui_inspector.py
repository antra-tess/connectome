#!/usr/bin/env python3
"""
TUI Inspector Tool

Interactive text-based user interface for inspecting Connectome host state.
Provides tree-like navigation and editing capabilities using terminal controls.
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
from dataclasses import dataclass
from enum import Enum

from .endpoint_handlers import InspectorEndpointHandlers
import aiohttp

logger = logging.getLogger(__name__)


class NavigationMode(Enum):
    MAIN_MENU = "main_menu"
    TREE_VIEW = "tree_view"
    DETAIL_VIEW = "detail_view"
    EDIT_MODE = "edit_mode"
    HUD_VIEW = "hud_view"


@dataclass
class TreeNode:
    """Represents a node in the tree navigation."""
    id: str
    label: str
    data: Any = None
    children: List['TreeNode'] = None
    parent: Optional['TreeNode'] = None
    is_expanded: bool = False
    is_editable: bool = False
    is_hud_viewable: bool = False
    hud_space_id: Optional[str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


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
            # Handle case where termios is not available
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
            return 24, 80  # Default fallback
    
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


class TUIInspector:
    """
    Text User Interface for Connectome Inspector.
    
    Provides an interactive terminal interface with:
    - Main menu navigation similar to HTML UI
    - Tree-like exploration of data structures
    - Editing capabilities for leaf nodes
    - Real-time data updates
    """
    
    def __init__(self, host_instance_or_url):
        """
        Initialize TUI inspector.
        
        Args:
            host_instance_or_url: Either a Host instance or HTTP base URL
        """
        self.host_instance = host_instance_or_url
        
        # Check if we're using HTTP client or direct host
        if isinstance(host_instance_or_url, str):
            # HTTP URL
            self.base_url = host_instance_or_url
            self.handlers = None
            self.http_session = None
        else:
            # Direct host instance
            self.base_url = None
            self.handlers = InspectorEndpointHandlers(host_instance_or_url)
            self.http_session = None
        
        self.terminal = TerminalController()
        
        # UI State
        self.mode = NavigationMode.MAIN_MENU
        self.current_menu_index = 0
        self.current_tree_index = 0
        self.current_tree_node = None
        self.tree_root = None
        self.scroll_offset = 0
        self.status_message = ""
        self.edit_buffer = ""
        self.search_query = ""
        
        # HUD view state
        self.hud_content = ""
        self.current_hud_space_id = None
        
        # Main menu items (similar to HTML UI sidebar)
        self.main_menu_items = [
            ("System Status", "status", "📊 Overall system status and health"),
            ("Spaces", "spaces", "🏢 Detailed information about all spaces"),
            ("Agents", "agents", "🤖 Agent configurations and status"),
            ("Adapters", "adapters", "🔌 Activity adapter connections"),
            ("Metrics", "metrics", "📈 System performance metrics"),
            ("Timelines", "timelines", "⏰ Timeline DAG overview"),
            ("VEIL Overview", "veil", "👁️  VEIL system overview"),
            ("Health Check", "health", "✅ Simple health check"),
            ("Quit", "quit", "❌ Exit the inspector")
        ]
        
        # Track if we're running
        self.running = True
        
    async def start(self):
        """Start the TUI inspector."""
        try:
            # Setup signal handling
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Setup HTTP session if needed
            if self.base_url:
                self.http_session = aiohttp.ClientSession()
            
            # Setup terminal
            self.terminal.setup_terminal()
            self.terminal.hide_cursor()
            
            # Main loop
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"TUI Inspector error: {e}", exc_info=True)
            self.status_message = f"Error: {str(e)}"
        finally:
            await self._cleanup()
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        self.running = False
    
    async def _cleanup(self):
        """Cleanup terminal state."""
        self.terminal.show_cursor()
        self.terminal.reset_colors()
        self.terminal.restore_terminal()
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
    
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
                await asyncio.sleep(0.1)  # Prevent rapid error loops
    
    async def _render_screen(self):
        """Render the current screen based on mode."""
        self.terminal.clear_screen()
        
        # Render header
        await self._render_header()
        
        # Render content based on mode
        if self.mode == NavigationMode.MAIN_MENU:
            await self._render_main_menu()
        elif self.mode == NavigationMode.TREE_VIEW:
            await self._render_tree_view()
        elif self.mode == NavigationMode.DETAIL_VIEW:
            await self._render_detail_view()
        elif self.mode == NavigationMode.EDIT_MODE:
            await self._render_edit_mode()
        elif self.mode == NavigationMode.HUD_VIEW:
            await self._render_hud_view()
        
        # Render footer
        await self._render_footer()
        
        # Ensure cursor stays hidden after each complete render
        self.terminal.hide_cursor()
    
    async def _render_header(self):
        """Render the header with title and breadcrumbs."""
        rows, cols = self.terminal.get_terminal_size()
        
        # Title bar
        self.terminal.move_cursor(1, 1)
        self.terminal.set_color('bright_cyan', bold=True)
        title = "🔍 Connectome Inspector TUI"
        
        # Add breadcrumbs based on mode
        breadcrumb = ""
        if self.mode == NavigationMode.TREE_VIEW and self.current_tree_node:
            breadcrumb = f" > {self.current_tree_node.label}"
        elif self.mode == NavigationMode.DETAIL_VIEW and self.current_tree_node:
            breadcrumb = f" > {self.current_tree_node.label} > Details"
        elif self.mode == NavigationMode.EDIT_MODE and self.current_tree_node:
            breadcrumb = f" > {self.current_tree_node.label} > Edit"
        elif self.mode == NavigationMode.HUD_VIEW:
            breadcrumb = f" > HUD View > {self.current_hud_space_id or 'Unknown'}"
        
        full_title = title + breadcrumb
        if len(full_title) > cols - 2:
            full_title = full_title[:cols-5] + "..."
        
        print(full_title.ljust(cols), end='')
        
        # Separator line
        self.terminal.move_cursor(2, 1)
        self.terminal.set_color('cyan')
        print('─' * cols, end='')
        
        self.terminal.reset_colors()
    
    async def _render_main_menu(self):
        """Render the main menu."""
        rows, cols = self.terminal.get_terminal_size()
        
        # Menu title
        self.terminal.move_cursor(4, 2)
        self.terminal.set_color('bright_green', bold=True)
        print("Select an option:")
        
        # Calculate available space for menu items
        # Reserve space for header (4 rows), title (1 row), scroll indicators (1 row each), and footer (3 rows)
        content_rows = rows - 10  # More conservative calculation
        
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
            self.terminal.move_cursor(5, 2)
            self.terminal.set_color('bright_yellow', bold=True)
            print("↑ ↑ ↑  More items above  ↑ ↑ ↑")
            self.terminal.reset_colors()
        else:
            # Clear the line if no scroll indicator needed
            self.terminal.move_cursor(5, 1)
            print(' ' * (cols - 1))
        
        # Clear the menu content area
        menu_start_row = 6
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
        
        # Ensure cursor stays hidden
        self.terminal.hide_cursor()
    
    async def _render_tree_view(self):
        """Render the tree navigation view."""
        if not self.tree_root:
            self.terminal.move_cursor(4, 2)
            self.terminal.set_color('yellow')
            print("Loading data...")
            self.terminal.reset_colors()
            return
        
        rows, cols = self.terminal.get_terminal_size()
        content_rows = rows - 6  # Header + footer space
        
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
            elif node.children:
                icon = "📁"
            else:
                icon = "📄"
            
            # Add HUD view indicator if available
            hud_indicator = ""
            if node.is_hud_viewable:
                hud_indicator = " [Enter - 👁️]"
            
            display_text = f"{tree_line}{icon} {node.label}{hud_indicator}"
            
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
        
        content_rows = rows - 8  # Leave space for header/footer/controls
        
        for i, line in enumerate(data_lines[self.scroll_offset:]):
            if i >= content_rows:
                break
            
            row = 6 + i
            self.terminal.move_cursor(row, 2)
            
            # Syntax highlighting for JSON
            if line.strip().startswith('"') and ':' in line:
                # Key-value pair
                self.terminal.set_color('cyan')
            elif line.strip() in ['{', '}', '[', ']']:
                # Brackets
                self.terminal.set_color('bright_white', bold=True)
            else:
                # Values
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
        
        # Edit buffer
        self.terminal.move_cursor(6, 2)
        self.terminal.set_color('white')
        print("Current value:")
        
        self.terminal.move_cursor(7, 2)
        self.terminal.set_color('black', 'white')
        edit_display = self.edit_buffer
        if len(edit_display) > cols - 6:
            edit_display = edit_display[:cols-9] + "..."
        print(f" {edit_display} ")
        
        # Instructions
        self.terminal.move_cursor(9, 2)
        self.terminal.set_color('bright_cyan')
        print("Enter new value and press Ctrl+S to save, Esc to cancel")
        
        self.terminal.reset_colors()
    
    async def _render_hud_view(self):
        """Render HUD view interface."""
        rows, cols = self.terminal.get_terminal_size()
        
        # HUD header
        self.terminal.move_cursor(4, 2)
        self.terminal.set_color('bright_green', bold=True)
        print(f"👁️ HUD View: {self.current_hud_space_id or 'Unknown'}")
        
        # Status line if loading
        if not self.hud_content:
            self.terminal.move_cursor(6, 2)
            self.terminal.set_color('yellow')
            print("Loading HUD content...")
            self.terminal.reset_colors()
            return
        
        # HUD content
        content_rows = rows - 8  # Leave space for header/footer/controls
        hud_lines = self.hud_content.split('\n')
        
        for i, line in enumerate(hud_lines[self.scroll_offset:]):
            if i >= content_rows:
                break
            
            row = 6 + i
            self.terminal.move_cursor(row, 2)
            
            # Basic syntax highlighting for markdown content
            if line.strip().startswith('# '):
                self.terminal.set_color('bright_cyan', bold=True)
            elif line.strip().startswith('## '):
                self.terminal.set_color('cyan', bold=True)
            elif line.strip().startswith('### '):
                self.terminal.set_color('bright_blue', bold=True)
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                self.terminal.set_color('bright_green')
            elif line.strip().startswith('**') and line.strip().endswith('**'):
                self.terminal.set_color('bright_white', bold=True)
            else:
                self.terminal.set_color('white')
            
            # Truncate long lines
            if len(line) > cols - 4:
                line = line[:cols-7] + "..."
            
            print(line)
        
        self.terminal.reset_colors()
    
    async def _render_footer(self):
        """Render footer with controls and status."""
        rows, cols = self.terminal.get_terminal_size()
        
        # Status message
        if self.status_message:
            self.terminal.move_cursor(rows - 2, 2)
            self.terminal.set_color('bright_red' if 'error' in self.status_message.lower() else 'bright_green')
            status_display = self.status_message
            if len(status_display) > cols - 4:
                status_display = status_display[:cols-7] + "..."
            print(status_display)
        
        # Controls
        self.terminal.move_cursor(rows - 1, 2)
        self.terminal.set_color('bright_black')
        
        if self.mode == NavigationMode.MAIN_MENU:
            controls = "↑↓: Navigate • Enter: Select • Q: Quit"
        elif self.mode == NavigationMode.TREE_VIEW:
            controls = "↑↓: Navigate • →: Expand • ←: Collapse • Enter: Details • E: Edit • B: Back • Q: Quit"
        elif self.mode == NavigationMode.DETAIL_VIEW:
            controls = "↑↓: Scroll • E: Edit • B: Back • Q: Quit"
        elif self.mode == NavigationMode.EDIT_MODE:
            controls = "Type to edit • Ctrl+S: Save • Esc: Cancel"
        elif self.mode == NavigationMode.HUD_VIEW:
            controls = "↑↓: Scroll • R: Refresh • B: Back • Q: Quit"
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
        
        if self.mode == NavigationMode.MAIN_MENU:
            await self._handle_main_menu_input(key)
        elif self.mode == NavigationMode.TREE_VIEW:
            await self._handle_tree_view_input(key)
        elif self.mode == NavigationMode.DETAIL_VIEW:
            await self._handle_detail_view_input(key)
        elif self.mode == NavigationMode.EDIT_MODE:
            await self._handle_edit_mode_input(key)
        elif self.mode == NavigationMode.HUD_VIEW:
            await self._handle_hud_view_input(key)
    
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
            # Check if current node is HUD viewable first
            if self.current_tree_node and self.current_tree_node.is_hud_viewable:
                await self._view_hud()
            else:
                await self._view_node_details()
        elif key in ['e', 'E']:
            await self._edit_current_node()
        elif key in ['b', 'B']:
            self.mode = NavigationMode.MAIN_MENU
            self.status_message = ""
            self.scroll_offset = 0  # Reset scroll when returning to main menu
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
            self.scroll_offset = 0  # Reset scroll when returning to tree view
    
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
    
    async def _handle_hud_view_input(self, key: str):
        """Handle input in HUD view mode."""
        if key == 'ESC[A':  # Up arrow
            self.scroll_offset = max(0, self.scroll_offset - 1)
        elif key == 'ESC[B':  # Down arrow
            self.scroll_offset += 1
        elif key in ['r', 'R']:
            await self._refresh_hud_content()
        elif key in ['b', 'B']:
            self.mode = NavigationMode.TREE_VIEW
            self.scroll_offset = 0  # Reset scroll when returning to tree view
    
    async def _execute_menu_selection(self):
        """Execute the currently selected main menu item."""
        if self.current_menu_index >= len(self.main_menu_items):
            return
        
        name, cmd, description = self.main_menu_items[self.current_menu_index]
        
        if cmd == "quit":
            self.running = False
            return
        
        # Load data for the selected endpoint
        self.status_message = f"Loading {name}..."
        try:
            data = await self._fetch_endpoint_data(cmd)
            
            # DEBUG: Save actual data structure to file for inspection
            import json
            debug_file = f"/tmp/inspector_data_{cmd}.json"
            try:
                with open(debug_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                logger.info(f"DEBUG: Saved actual data structure to {debug_file}")
            except Exception as e:
                logger.error(f"Failed to save debug data: {e}")
            
            self.tree_root = await self._build_tree_from_data(data, name)
            self.current_tree_node = self.tree_root.children[0] if self.tree_root.children else None
            self.mode = NavigationMode.TREE_VIEW
            self.scroll_offset = 0
            self.status_message = f"Loaded {name}"
        except Exception as e:
            self.status_message = f"Error loading {name}: {str(e)}"
    
    async def _fetch_endpoint_data(self, endpoint: str) -> Dict[str, Any]:
        """Fetch data from the specified endpoint."""
        if self.base_url:
            # Use HTTP client
            endpoint_map = {
                "status": "/status",
                "spaces": "/spaces", 
                "agents": "/agents",
                "adapters": "/adapters",
                "metrics": "/metrics",
                "timelines": "/timelines",
                "veil": "/veil",
                "health": "/health"
            }
            
            if endpoint in endpoint_map:
                url = self.base_url + endpoint_map[endpoint]
                try:
                    async with self.http_session.get(url) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            raise ValueError(f"HTTP {response.status}: {error_text}")
                except aiohttp.ClientConnectorError as e:
                    raise ConnectionError(f"Could not connect to {url}: {e}")
            else:
                raise ValueError(f"Unknown endpoint: {endpoint}")
        else:
            # Use direct handlers
            if endpoint == "status":
                return await self.handlers.handle_status()
            elif endpoint == "spaces":
                return await self.handlers.handle_spaces()
            elif endpoint == "agents":
                return await self.handlers.handle_agents()
            elif endpoint == "adapters":
                return await self.handlers.handle_adapters()
            elif endpoint == "metrics":
                return await self.handlers.handle_metrics()
            elif endpoint == "timelines":
                return await self.handlers.handle_timelines()
            elif endpoint == "veil":
                return await self.handlers.handle_veil()
            elif endpoint == "health":
                return await self.handlers.handle_health()
            else:
                raise ValueError(f"Unknown endpoint: {endpoint}")
    
    async def _build_tree_from_data(self, data: Any, root_label: str) -> TreeNode:
        """Build a tree structure from the data."""
        root = TreeNode("root", root_label)
        
        def build_node(obj: Any, label: str, parent: TreeNode, path: str = "") -> TreeNode:
            node = TreeNode(path or label, label, obj, parent=parent)
            
            # Debug logging for all dict nodes
            if isinstance(obj, dict):
                logger.debug(f"Processing node: label='{label}', path='{path}', keys={list(obj.keys())[:5]}...")
                
                # Check if this is a space-like node
                is_space_like = self._is_space_like_node(obj, label, path)
                logger.debug(f"  -> is_space_like: {is_space_like}")
                
                if is_space_like:
                    # Check if it's HUD viewable
                    hud_info = self._check_hud_viewable(obj, label, path)
                    logger.debug(f"  -> hud_info: {hud_info}")
                    
                    if hud_info:
                        node.is_hud_viewable = True
                        node.hud_space_id = hud_info
                        logger.info(f"✓ Found HUD-viewable node: {label} at {path} -> space_id: {hud_info}")
                    else:
                        logger.debug(f"  -> No HUD capability found for {label}")
                else:
                    logger.debug(f"  -> Not space-like: {label}")
            
            if isinstance(obj, dict):
                node.children = []
                for key, value in obj.items():
                    child_path = f"{path}.{key}" if path else key
                    child = build_node(value, key, node, child_path)
                    node.children.append(child)
            elif isinstance(obj, list) and obj:
                node.children = []
                for i, item in enumerate(obj):
                    child_path = f"{path}[{i}]" if path else f"[{i}]"
                    child = build_node(item, f"Item {i}", node, child_path)
                    node.children.append(child)
            else:
                # Leaf node - make it editable if it's a simple value
                if isinstance(obj, (str, int, float, bool)) or obj is None:
                    node.is_editable = True
            
            return node
        
        if isinstance(data, dict):
            for key, value in data.items():
                child = build_node(value, key, root, key)
                root.children.append(child)
        else:
            child = build_node(data, "Data", root, "data")
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
            self.mode = NavigationMode.DETAIL_VIEW
            self.scroll_offset = 0
    
    async def _edit_current_node(self):
        """Enter edit mode for the current node."""
        if self.current_tree_node and self.current_tree_node.is_editable:
            self.mode = NavigationMode.EDIT_MODE
            self.edit_buffer = str(self.current_tree_node.data) if self.current_tree_node.data is not None else ""
        else:
            self.status_message = "This item is not editable"
    
    async def _save_edit(self):
        """Save the current edit."""
        if not self.current_tree_node:
            return
        
        try:
            # Try to parse as JSON first
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
            
            # Update the node
            self.current_tree_node.data = new_value
            
            # Here you would typically call an API to persist the change
            # For now, we'll just update the local data
            self.status_message = f"Updated {self.current_tree_node.label} to: {new_value}"
            self.mode = NavigationMode.DETAIL_VIEW
            self.edit_buffer = ""
            
        except ValueError as e:
            self.status_message = f"Invalid value: {str(e)}"
    
    async def _refresh_current_data(self):
        """Refresh the current data view."""
        if self.mode == NavigationMode.TREE_VIEW and hasattr(self, '_current_endpoint'):
            try:
                self.status_message = "Refreshing..."
                data = await self._fetch_endpoint_data(self._current_endpoint)
                self.tree_root = await self._build_tree_from_data(data, self.tree_root.label)
                self.status_message = "Data refreshed"
            except Exception as e:
                self.status_message = f"Refresh failed: {str(e)}"
    
    async def _view_hud(self):
        """Switch to HUD view for the current node."""
        if not self.current_tree_node or not self.current_tree_node.is_hud_viewable:
            self.status_message = "This node does not have HUD view capability"
            return
        
        # Set HUD view state
        self.current_hud_space_id = self.current_tree_node.hud_space_id
        self.mode = NavigationMode.HUD_VIEW
        self.scroll_offset = 0
        self.hud_content = ""  # Clear old content
        
        # Load HUD content
        await self._load_hud_content()
    
    async def _load_hud_content(self):
        """Load HUD content from the server."""
        if not self.current_hud_space_id:
            self.status_message = "No space ID available for HUD view"
            return
        
        try:
            self.status_message = "Loading HUD content..."
            
            if self.base_url:
                # Use HTTP client
                url = f"{self.base_url}/spaces/{self.current_hud_space_id}/render?format=markdown"
                async with self.http_session.get(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.hud_content = result.get('content', 'No content available')
                    else:
                        error_text = await response.text()
                        self.hud_content = f"Error loading HUD: HTTP {response.status}: {error_text}"
            else:
                # Use direct handlers
                options = {'format': 'markdown'}
                result = await self.handlers.handle_space_render(self.current_hud_space_id, options)
                self.hud_content = result.get('content', 'No content available')
                
            self.status_message = f"Loaded HUD for {self.current_hud_space_id}"
            
        except Exception as e:
            self.hud_content = f"Error loading HUD: {str(e)}"
            self.status_message = f"Failed to load HUD: {str(e)}"
    
    async def _refresh_hud_content(self):
        """Refresh the current HUD content."""
        await self._load_hud_content()
    
    def _is_space_like_node(self, obj: Dict[str, Any], label: str, path: str) -> bool:
        """Check if a node is space-like (matches web UI logic)."""
        if not isinstance(obj, dict):
            return False
        
        path_parts = path.split('.')
        logger.debug(f"    _is_space_like_node: path_parts={path_parts}, obj_keys={list(obj.keys())[:3]}")
        
        # Check for space structure (from spaces endpoint)
        if len(path_parts) >= 2 and path_parts[-2] == "details":
            has_type = 'type' in obj
            has_elements_or_components = 'elements' in obj or 'components' in obj
            result = has_type and has_elements_or_components
            logger.debug(f"    -> Space check: has_type={has_type}, has_elements_or_components={has_elements_or_components} -> {result}")
            return result
        
        # Check for agent structure (from agents endpoint)  
        if len(path_parts) >= 2 and path_parts[-2] == "agents":
            has_agent_id = 'agent_id' in obj
            is_inner_space = 'type' in obj and 'InnerSpace' in str(obj.get('type', ''))
            result = has_agent_id or is_inner_space
            logger.debug(f"    -> Agent check: has_agent_id={has_agent_id}, is_inner_space={is_inner_space} -> {result}")
            return result
            
        logger.debug(f"    -> Neither space nor agent structure")
        return False
    
    def _check_hud_viewable(self, obj: Dict[str, Any], label: str, path: str) -> Optional[str]:
        """Check if a node has HUD view capability and return space_id if available."""
        # Check if it has a direct 'hud' child
        if 'hud' in obj:
            return self._extract_space_id(obj, label, path)
        
        # Check if it has a 'components' with facet-aware HUD component  
        if 'components' in obj:
            if self._has_facet_aware_hud_component(obj['components']):
                return self._extract_space_id(obj, label, path)
        
        # Check if any of its elements have HUD components
        if 'elements' in obj and isinstance(obj['elements'], dict):
            for element_id, element_data in obj['elements'].items():
                if isinstance(element_data, dict) and 'components' in element_data:
                    if self._has_facet_aware_hud_component(element_data['components']):
                        return self._extract_space_id(obj, label, path)
        
        return None
    
    def _has_facet_aware_hud_component(self, components) -> bool:
        """Check if components contain a facet-aware HUD component."""
        # Components can be either a dict (old format) or a list (new format)
        if isinstance(components, list):
            # New format: list of component info dicts
            for comp in components:
                if isinstance(comp, dict):
                    # Check component ID patterns
                    comp_id = comp.get('id', '').lower()
                    if ('facetawarehudcomponent' in comp_id or
                        'facet_aware_hud' in comp_id or
                        'hud' in comp_id):
                        return True
                        
                    # Check component type and class_path
                    comp_type = comp.get('type', '').lower()
                    class_path = comp.get('class_path', '').lower()
                    if ('facetawarehudcomponent' in comp_type or
                        'facet_aware_hud' in comp_type or
                        'facetawarehudcomponent' in class_path or
                        'facet_aware_hud' in class_path):
                        return True
        elif isinstance(components, dict):
            # Old format: dict of component_id -> component_info
            for comp_id, comp in components.items():
                if isinstance(comp, dict):
                    # Check component ID patterns
                    comp_id_lower = comp_id.lower()
                    if ('facetawarehudcomponent' in comp_id_lower or
                        'facet_aware_hud' in comp_id_lower or
                        'hud' in comp_id_lower):
                        return True
                        
                    # Check component type
                    comp_type = comp.get('type', '')
                    if isinstance(comp_type, str):
                        comp_type_lower = comp_type.lower()
                        if ('facetawarehudcomponent' in comp_type_lower or
                            'facet_aware_hud' in comp_type_lower):
                            return True
        
        return False
    
    def _extract_space_id(self, obj: Dict[str, Any], label: str, path: str) -> str:
        """Extract space ID from object, label, or path."""
        # For spaces/agents endpoints, the space ID is typically the key in the path
        path_parts = path.split('.')
        
        # For spaces endpoint: path like "details.space_id"
        if len(path_parts) >= 2 and path_parts[-2] == "details":
            return path_parts[-1]  # space_id
            
        # For agents endpoint: path like "agents.agent_id"
        if len(path_parts) >= 2 and path_parts[-2] == "agents":
            return path_parts[-1]  # agent_id (which is the space_id)
        
        # Try direct space ID field
        for id_field in ['id', 'agent_id']:
            if id_field in obj:
                return str(obj[id_field])
        
        # Use label as space ID (fallback case)
        if label and label != "root":
            return label
        
        return "unknown"


async def main_tui(host_instance):
    """
    Main TUI entry point.
    
    Args:
        host_instance: The Host instance to inspect
    """
    tui = TUIInspector(host_instance)
    try:
        await tui.start()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    print("❌ TUI Inspector requires a Host instance to inspect")
    print("Use: python -m host.modules.inspector.tui_inspector <host_instance>")
    sys.exit(1)