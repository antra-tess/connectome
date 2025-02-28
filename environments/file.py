"""
File Environment
Defines an environment for file system operations.
"""

import logging
import os
import json
import fnmatch
from typing import Dict, Any, Optional, List

from environments.base import Environment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FileEnvironment(Environment):
    """
    Environment for file system operations.
    
    Provides tools for reading, writing, listing, and deleting files.
    """
    
    def __init__(self, env_id: str = "file", name: str = "File Environment", 
                description: str = "Environment for file system operations"):
        """
        Initialize the file environment.
        
        Args:
            env_id: Unique identifier for this environment
            name: Human-readable name for this environment
            description: Description of this environment's purpose
        """
        super().__init__(env_id, name, description)
        self._register_file_tools()
        
        # Define allowed directories and file extensions for security
        self.allowed_directories = ["data", "templates", "user_files"]
        self.allowed_extensions = [".txt", ".md", ".json", ".csv", ".yaml", ".yml"]
        
        # Track recently accessed and modified files
        self._recent_files = []
        
        logger.info(f"Created environment: {name} ({env_id})")
    
    def render_state_for_context(self) -> Dict[str, Any]:
        """
        Render the file environment's state for inclusion in the agent's context.
        
        This provides information about recent file operations that can help the
        agent maintain context about file interactions.
        
        Returns:
            Dictionary with formatted file environment state
        """
        # Get base state info
        state = super().render_state_for_context()
        state["type"] = "file"
        
        # Build formatted state text
        formatted_text = []
        
        # Add allowed directories and extensions
        formatted_text.append("Allowed directories:")
        for directory in self.allowed_directories:
            formatted_text.append(f"- {directory}")
        
        formatted_text.append("\nAllowed file extensions:")
        for ext in self.allowed_extensions:
            formatted_text.append(f"- {ext}")
        
        # Add recent file operations if any
        if self._recent_files:
            formatted_text.append("\nRecent file operations:")
            for file_op in self._recent_files[-5:]:  # Show last 5 operations
                op_type = file_op.get("operation", "Unknown operation")
                path = file_op.get("path", "Unknown path")
                timestamp = file_op.get("timestamp", "Unknown time")
                success = file_op.get("success", False)
                status = "successfully" if success else "with errors"
                
                formatted_text.append(f"- {op_type} on '{path}' ({status} at {timestamp})")
        else:
            formatted_text.append("\nNo recent file operations.")
        
        # Set the formatted text
        state["formatted_state_text"] = "\n".join(formatted_text)
        
        # Include raw data for potential specialized handling
        state["allowed_directories"] = self.allowed_directories
        state["allowed_extensions"] = self.allowed_extensions
        state["recent_operations"] = self._recent_files[-5:] if self._recent_files else []
        
        return state
        
    def _register_file_tools(self):
        """Register all file-related tools."""
        
        @self.register_tool(
            name="read_file",
            description="Read the contents of a file.",
            parameter_descriptions={
                "file_path": "Path to the file to read",
                "start_line": "Starting line number (1-indexed, optional)",
                "end_line": "Ending line number (1-indexed, optional)",
            }
        )
        def read_file(file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> Dict[str, Any]:
            """
            Read the contents of a file.
            
            Security measures are implemented to restrict access to only
            allowed directories and file types.
            
            Args:
                file_path: Path to the file to read
                start_line: Starting line number (1-indexed, optional)
                end_line: Ending line number (1-indexed, optional)
                
            Returns:
                Dictionary with file content and metadata
            """
            # Record the operation
            import time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            logger.info(f"Reading file: {file_path}, lines {start_line}-{end_line}")
            
            # Security checks
            try:
                # Check for directory traversal attacks
                file_path = os.path.normpath(file_path)
                
                # Check if file is in an allowed directory
                file_dir = os.path.dirname(file_path)
                
                if not any(file_dir.startswith(allowed_dir) for allowed_dir in self.allowed_directories):
                    error_msg = f"Security error: Access to directory '{file_dir}' is not allowed"
                    logger.error(error_msg)
                    return {
                        "content": error_msg,
                        "file_path": file_path,
                        "success": False,
                        "error": error_msg
                    }
                    
                # Check file extension
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext not in self.allowed_extensions:
                    error_msg = f"Security error: Access to file type '{file_ext}' is not allowed"
                    logger.error(error_msg)
                    return {
                        "content": error_msg,
                        "file_path": file_path,
                        "success": False,
                        "error": error_msg
                    }
                
                # Check if file exists
                if not os.path.isfile(file_path):
                    return {
                        "content": f"Error: File '{file_path}' does not exist.",
                        "file_path": file_path,
                        "success": False,
                        "error": f"Error: File '{file_path}' does not exist."
                    }
                
                # Read the file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Apply line number filters if provided
                if start_line is not None:
                    start_idx = max(0, start_line - 1)  # Convert to 0-indexed
                else:
                    start_idx = 0
                    
                if end_line is not None:
                    end_idx = min(len(lines), end_line)  # End line is inclusive
                else:
                    end_idx = len(lines)
                    
                filtered_lines = lines[start_idx:end_idx]
                content = "".join(filtered_lines)
                
                # Record successful operation
                self._recent_files.append({
                    "operation": "read",
                    "path": file_path,
                    "timestamp": timestamp,
                    "success": True
                })
                
                # Limit history size
                if len(self._recent_files) > 20:
                    self._recent_files = self._recent_files[-20:]
                
                return {
                    "content": content,
                    "file_path": file_path,
                    "success": True
                }
                
            except Exception as e:
                error_msg = f"Error reading file {file_path}: {str(e)}"
                logger.error(error_msg)
                
                # Record failed operation
                self._recent_files.append({
                    "operation": "read",
                    "path": file_path,
                    "timestamp": timestamp,
                    "success": False,
                    "error": str(e)
                })
                
                raise
        
        @self.register_tool(
            name="write_file",
            description="Write content to a file.",
            parameter_descriptions={
                "file_path": "Path to the file to write",
                "content": "Content to write to the file",
                "append": "Whether to append to the file instead of overwriting (optional)"
            }
        )
        def write_file(file_path: str, content: str, append: Optional[bool] = False) -> Dict[str, Any]:
            """
            Write content to a file.
            
            Args:
                file_path: Path to the file to write
                content: Content to write to the file
                append: Whether to append to the file instead of overwriting
                
            Returns:
                Dictionary with information about the file write operation
            """
            # Record the operation
            import time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            try:
                # Implementation would be here...
                # This is a placeholder
                
                # Record successful operation
                self._recent_files.append({
                    "operation": "write",
                    "path": file_path,
                    "timestamp": timestamp,
                    "success": True,
                    "append": append
                })
                
                # Limit history size
                if len(self._recent_files) > 20:
                    self._recent_files = self._recent_files[-20:]
                
                return {
                    "file_path": file_path,
                    "success": True,
                    "bytes_written": len(content),
                    "mode": "append" if append else "write"
                }
                
            except Exception as e:
                logger.error(f"Error writing to file {file_path}: {str(e)}")
                
                # Record failed operation
                self._recent_files.append({
                    "operation": "write",
                    "path": file_path,
                    "timestamp": timestamp,
                    "success": False,
                    "append": append,
                    "error": str(e)
                })
                
                raise
        
        @self.register_tool(
            name="list_files",
            description="List files in a directory.",
            parameter_descriptions={
                "directory": "Path to the directory to list",
                "pattern": "File pattern to match (optional)",
            }
        )
        def list_files(directory: str, pattern: Optional[str] = None) -> Dict[str, Any]:
            """
            List files in a directory.
            
            Args:
                directory: Path to the directory to list
                pattern: File pattern to match
                
            Returns:
                Dictionary with list of files and directory information
            """
            # Record the operation
            import time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            try:
                # Implementation would be here...
                # This is a placeholder
                files = [f"file1.txt", f"file2.md", f"file3.json"]
                
                # Record successful operation
                self._recent_files.append({
                    "operation": "list",
                    "path": directory,
                    "timestamp": timestamp,
                    "success": True,
                    "pattern": pattern
                })
                
                # Limit history size
                if len(self._recent_files) > 20:
                    self._recent_files = self._recent_files[-20:]
                
                return {
                    "directory": directory,
                    "pattern": pattern,
                    "files": files,
                    "count": len(files),
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"Error listing directory {directory}: {str(e)}")
                
                # Record failed operation
                self._recent_files.append({
                    "operation": "list",
                    "path": directory,
                    "timestamp": timestamp,
                    "success": False,
                    "pattern": pattern,
                    "error": str(e)
                })
                
                raise
        
        @self.register_tool(
            name="delete_file",
            description="Delete a file.",
            parameter_descriptions={
                "file_path": "Path to the file to delete"
            }
        )
        def delete_file(file_path: str) -> Dict[str, Any]:
            """
            Delete a file.
            
            Args:
                file_path: Path to the file to delete
                
            Returns:
                Dictionary with information about the delete operation
            """
            # Record the operation
            import time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            try:
                # Implementation would be here...
                # This is a placeholder
                
                # Record successful operation
                self._recent_files.append({
                    "operation": "delete",
                    "path": file_path,
                    "timestamp": timestamp,
                    "success": True
                })
                
                # Limit history size
                if len(self._recent_files) > 20:
                    self._recent_files = self._recent_files[-20:]
                
                return {
                    "file_path": file_path,
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")
                
                # Record failed operation
                self._recent_files.append({
                    "operation": "delete",
                    "path": file_path,
                    "timestamp": timestamp,
                    "success": False,
                    "error": str(e)
                })
                
                raise 