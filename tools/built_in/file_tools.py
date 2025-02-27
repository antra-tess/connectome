"""
File Tools
Tools for working with files and file systems.
"""

import logging
import os
import json
from typing import Optional, Dict, Any, List

from tools.registry import register_tool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@register_tool(
    name="read_file",
    description="Read the contents of a file.",
    parameter_descriptions={
        "file_path": "Path to the file to read",
        "start_line": "Starting line number (1-indexed, optional)",
        "end_line": "Ending line number (1-indexed, optional)",
    }
)
def read_file(file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """
    Read the contents of a file.
    
    Security measures should be implemented to restrict access to only
    allowed directories and file types.
    
    Args:
        file_path: Path to the file to read
        start_line: Starting line number (1-indexed, optional)
        end_line: Ending line number (1-indexed, optional)
        
    Returns:
        File contents as a string
    """
    logger.info(f"Reading file: {file_path}, lines {start_line}-{end_line}")
    
    # Security checks
    try:
        # Check for directory traversal attacks
        file_path = os.path.normpath(file_path)
        
        # Check if file is in an allowed directory
        allowed_directories = ["data", "templates", "user_files"]
        file_dir = os.path.dirname(file_path)
        
        if not any(file_dir.startswith(allowed_dir) for allowed_dir in allowed_directories):
            error_msg = f"Security error: Access to directory '{file_dir}' is not allowed"
            logger.error(error_msg)
            return error_msg
            
        # Check file extension
        allowed_extensions = [".txt", ".md", ".json", ".csv", ".yaml", ".yml"]
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in allowed_extensions:
            error_msg = f"Security error: Access to file type '{file_ext}' is not allowed"
            logger.error(error_msg)
            return error_msg
            
        # In a real implementation, this would check if the file exists and read it
        # For now, we'll just return a mock response
        
        # Mock file content
        mock_content = f"This is mock content for file {file_path}.\n"
        mock_content += "Line 2 of the mock file.\n"
        mock_content += "Line 3 of the mock file.\n"
        mock_content += "Line 4 of the mock file.\n"
        mock_content += "Line 5 of the mock file.\n"
        
        lines = mock_content.splitlines()
        
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
        return "\n".join(filtered_lines)
        
    except Exception as e:
        error_msg = f"Error reading file {file_path}: {str(e)}"
        logger.error(error_msg)
        return error_msg


@register_tool(
    name="write_file",
    description="Write content to a file.",
    parameter_descriptions={
        "file_path": "Path to the file to write",
        "content": "Content to write to the file",
        "append": "Whether to append to the file instead of overwriting (optional)"
    }
)
def write_file(file_path: str, content: str, append: Optional[bool] = False) -> str:
    """
    Write content to a file.
    
    Security measures should be implemented to restrict access to only
    allowed directories and file types.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        append: Whether to append to the file instead of overwriting
        
    Returns:
        Confirmation message
    """
    logger.info(f"Writing to file: {file_path}, append: {append}")
    
    # Security checks
    try:
        # Check for directory traversal attacks
        file_path = os.path.normpath(file_path)
        
        # Check if file is in an allowed directory
        allowed_directories = ["data", "templates", "user_files"]
        file_dir = os.path.dirname(file_path)
        
        if not any(file_dir.startswith(allowed_dir) for allowed_dir in allowed_directories):
            error_msg = f"Security error: Access to directory '{file_dir}' is not allowed"
            logger.error(error_msg)
            return error_msg
            
        # Check file extension
        allowed_extensions = [".txt", ".md", ".json", ".csv", ".yaml", ".yml"]
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in allowed_extensions:
            error_msg = f"Security error: Access to file type '{file_ext}' is not allowed"
            logger.error(error_msg)
            return error_msg
            
        # In a real implementation, this would write to the file
        # For now, we'll just return a mock response
        mode = "append" if append else "write"
        return f"Successfully {mode}ed {len(content)} characters to {file_path}"
        
    except Exception as e:
        error_msg = f"Error writing to file {file_path}: {str(e)}"
        logger.error(error_msg)
        return error_msg


@register_tool(
    name="list_files",
    description="List files in a directory.",
    parameter_descriptions={
        "directory": "Path to the directory to list",
        "pattern": "File pattern to match (optional)",
    }
)
def list_files(directory: str, pattern: Optional[str] = None) -> List[str]:
    """
    List files in a directory.
    
    Security measures should be implemented to restrict access to only
    allowed directories.
    
    Args:
        directory: Path to the directory to list
        pattern: File pattern to match (e.g., "*.txt")
        
    Returns:
        List of file paths
    """
    logger.info(f"Listing files in directory: {directory}, pattern: {pattern}")
    
    # Security checks
    try:
        # Check for directory traversal attacks
        directory = os.path.normpath(directory)
        
        # Check if directory is allowed
        allowed_directories = ["data", "templates", "user_files"]
        
        if not any(directory.startswith(allowed_dir) for allowed_dir in allowed_directories):
            error_msg = f"Security error: Access to directory '{directory}' is not allowed"
            logger.error(error_msg)
            return [error_msg]
            
        # In a real implementation, this would list files in the directory
        # For now, we'll just return a mock response
        mock_files = [
            f"{directory}/file1.txt",
            f"{directory}/file2.md",
            f"{directory}/data.json",
            f"{directory}/config.yaml",
            f"{directory}/report.csv"
        ]
        
        # Apply pattern filtering if provided
        if pattern:
            import fnmatch
            mock_files = [f for f in mock_files if fnmatch.fnmatch(os.path.basename(f), pattern)]
            
        return mock_files
        
    except Exception as e:
        error_msg = f"Error listing files in directory {directory}: {str(e)}"
        logger.error(error_msg)
        return [error_msg]


@register_tool(
    name="delete_file",
    description="Delete a file.",
    parameter_descriptions={
        "file_path": "Path to the file to delete"
    }
)
def delete_file(file_path: str) -> str:
    """
    Delete a file.
    
    Security measures should be implemented to restrict access to only
    allowed directories and file types.
    
    Args:
        file_path: Path to the file to delete
        
    Returns:
        Confirmation message
    """
    logger.info(f"Deleting file: {file_path}")
    
    # Security checks
    try:
        # Check for directory traversal attacks
        file_path = os.path.normpath(file_path)
        
        # Check if file is in an allowed directory
        allowed_directories = ["data", "templates", "user_files"]
        file_dir = os.path.dirname(file_path)
        
        if not any(file_dir.startswith(allowed_dir) for allowed_dir in allowed_directories):
            error_msg = f"Security error: Access to directory '{file_dir}' is not allowed"
            logger.error(error_msg)
            return error_msg
            
        # Check file extension
        allowed_extensions = [".txt", ".md", ".json", ".csv", ".yaml", ".yml"]
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in allowed_extensions:
            error_msg = f"Security error: Access to file type '{file_ext}' is not allowed"
            logger.error(error_msg)
            return error_msg
            
        # In a real implementation, this would delete the file
        # For now, we'll just return a mock response
        return f"Successfully deleted file: {file_path}"
        
    except Exception as e:
        error_msg = f"Error deleting file {file_path}: {str(e)}"
        logger.error(error_msg)
        return error_msg 