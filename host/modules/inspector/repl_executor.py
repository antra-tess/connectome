"""
REPL Executor for Connectome Inspector

Provides a safe Python REPL executor for debugging and introspection.
This is an external debugging tool that runs in the Inspector's process space,
similar to pdb or Chrome DevTools - completely isolated from agent code.
"""

import io
import sys
import time
import traceback
import contextlib
from typing import Dict, Any, Optional


class SafeREPLExecutor:
    """
    Safe Python REPL executor for the Connectome Inspector.
    
    Executes Python code strings with output capture and exception handling.
    Designed for debugging and introspection of Connectome system state.
    """
    
    def __init__(self):
        """Initialize the REPL executor."""
        self.default_namespace = {
            '__builtins__': __builtins__,
            'time': time,
            'traceback': traceback,
        }
    
    def execute(self, code: str, namespace: Optional[Dict[str, Any]] = None, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Execute Python code and capture output.
        
        Args:
            code: Python code string to execute
            namespace: Variable namespace for execution (uses default if None)
            timeout: Execution timeout in seconds (not enforced yet - basic implementation)
            
        Returns:
            Dict containing:
                - output: Captured stdout
                - error: Captured stderr or exception message
                - success: Boolean indicating successful execution
                - execution_time_ms: Execution time in milliseconds
        """
        start_time = time.time()
        
        # Use provided namespace or create a copy of default
        if namespace is not None:
            exec_namespace = namespace
            # Ensure builtins are available
            if '__builtins__' not in exec_namespace:
                exec_namespace['__builtins__'] = __builtins__
        else:
            exec_namespace = self.default_namespace.copy()
        
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result = {
            'output': '',
            'error': '',
            'success': False,
            'execution_time_ms': 0
        }
        
        try:
            # Redirect stdout and stderr to capture output
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):
                
                # Execute the code
                exec(code, exec_namespace)
            
            # Successful execution
            result['output'] = stdout_capture.getvalue()
            result['error'] = stderr_capture.getvalue()
            result['success'] = True
            
        except Exception as e:
            # Capture any exceptions
            result['output'] = stdout_capture.getvalue()
            result['error'] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result['success'] = False
            
        finally:
            # Calculate execution time
            end_time = time.time()
            result['execution_time_ms'] = round((end_time - start_time) * 1000, 2)
        
        return result
    
    def get_inspector_namespace(self, host_instance=None) -> Dict[str, Any]:
        """
        Create a namespace with useful objects for inspector debugging.
        
        Args:
            host_instance: Reference to the main Host instance
            
        Returns:
            Dict with useful debugging objects
        """
        namespace = self.default_namespace.copy()
        
        if host_instance:
            namespace.update({
                'host': host_instance,
                'space_registry': getattr(host_instance, 'space_registry', None),
                'event_loop': getattr(host_instance, 'event_loop', None),
                'activity_client': getattr(host_instance, 'activity_client', None),
            })
            
            # Add space registry if available
            try:
                from elements.space_registry import SpaceRegistry
                namespace['SpaceRegistry'] = SpaceRegistry
                namespace['spaces'] = SpaceRegistry.get_instance()
            except ImportError:
                pass
        
        return namespace