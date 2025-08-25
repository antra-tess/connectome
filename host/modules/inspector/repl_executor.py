"""
REPL Executor for Connectome Inspector

Provides an IPython-based REPL executor for debugging and introspection.
This is an external debugging tool that runs in the Inspector's process space,
similar to pdb or Chrome DevTools - completely isolated from agent code.

Features:
- IPython shell with rich output, syntax highlighting, and tab completion
- Object introspection and attribute exploration
- Session-based namespace management
- Backward compatibility with existing REPL API
"""

import io
import sys
import time
import traceback
import contextlib
import json
import html
import inspect
from typing import Dict, Any, Optional, List, Tuple

try:
    from IPython.utils.capture import capture_output
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    # Fallback to basic implementation if IPython not available
    TerminalInteractiveShell = None


class IPythonREPLExecutor:
    """
    IPython-based REPL executor for the Connectome Inspector.
    
    Provides rich interactive Python environment with:
    - Tab completion and object introspection
    - Rich output display (HTML, images, etc.)
    - Session-based namespace management
    - Command history and enhanced error reporting
    - Backward compatibility with existing REPL API
    """
    
    def __init__(self):
        """Initialize the IPython REPL executor."""
        self.default_namespace = {
            '__builtins__': __builtins__,
            'time': time,
            'traceback': traceback,
        }
        
        # Initialize IPython shell if available
        if IPYTHON_AVAILABLE:
            self.shell = None  # Will be created per-session
        else:
            self.shell = None
            
        # Fallback for when IPython is not available
        self._fallback_executor = SafeREPLExecutor() if not IPYTHON_AVAILABLE else None
    
    def create_shell(self, namespace: Optional[Dict[str, Any]] = None) -> Any:
        """Create a new IPython shell instance with the given namespace."""
        if not IPYTHON_AVAILABLE:
            return None
            
        # Create isolated IPython shell (not singleton) with custom configuration
        shell = TerminalInteractiveShell()
        
        # Configure shell for better completion
        shell.readline_use = False  # Disable readline for programmatic use
        shell.autoindent = True
        shell.colors = 'NoColor'  # Avoid terminal color codes
        
        # Disable Jedi to avoid index errors in programmatic usage
        shell.Completer.use_jedi = False  # Use basic completer for reliability
        shell.Completer.greedy = True   # More aggressive completion
        
        # Initialize IPython internals properly
        shell.init_history()
        shell.init_display_formatter()
        
        # Update namespace AND sync completer namespaces
        if namespace:
            shell.user_ns.update(namespace)
            # CRITICAL: Sync completer namespaces with user namespace
            shell.Completer.namespace = shell.user_ns
            shell.Completer.global_namespace = shell.user_global_ns
            
        # Shell is ready for use
        
        return shell
    
    def execute(self, code: str, namespace: Optional[Dict[str, Any]] = None, timeout: float = 5.0, shell_instance: Any = None) -> Dict[str, Any]:
        """
        Execute Python code using IPython and capture rich output.
        
        Args:
            code: Python code string to execute
            namespace: Variable namespace for execution (uses default if None)
            timeout: Execution timeout in seconds (not enforced yet)
            shell_instance: Existing IPython shell instance to use
            
        Returns:
            Dict containing:
                - output: Captured stdout/display output
                - error: Captured stderr or exception message
                - success: Boolean indicating successful execution
                - execution_time_ms: Execution time in milliseconds
                - execution_count: IPython execution counter
                - rich_output: Rich display data (HTML, images, etc.)
                - completions: Available completions (if requested)
        """
        start_time = time.time()
        
        # Fallback to basic executor if IPython not available
        if not IPYTHON_AVAILABLE or shell_instance is None:
            return self._fallback_execute(code, namespace, timeout)
            
        # Use provided shell or create temporary one
        shell = shell_instance
        if namespace:
            shell.user_ns.update(namespace)
            
        result = {
            'output': '',
            'error': '',
            'success': False,
            'execution_time_ms': 0,
            'execution_count': 0,
            'rich_output': [],
            'completions': []
        }
        
        try:
            # Use IPython's capture_output for rich display capture
            with capture_output() as captured:
                exec_result = shell.run_cell(code, store_history=True, silent=False)
                
            # Process execution result
            result['success'] = exec_result.success
            result['execution_count'] = shell.execution_count
            
            # Capture output
            result['output'] = captured.stdout
            if captured.stderr:
                result['error'] = captured.stderr
                
            # Handle execution errors
            if exec_result.error_in_exec:
                result['error'] = str(exec_result.error_in_exec)
                result['success'] = False
                
            # Capture rich output (HTML, images, etc.)
            if hasattr(captured, 'outputs'):
                result['rich_output'] = self._format_rich_outputs(captured.outputs)
                
        except Exception as e:
            result['error'] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result['success'] = False
            
        finally:
            # Calculate execution time
            end_time = time.time()
            result['execution_time_ms'] = round((end_time - start_time) * 1000, 2)
            
        return result
    
    def _fallback_execute(self, code: str, namespace: Optional[Dict[str, Any]] = None, timeout: float = 5.0) -> Dict[str, Any]:
        """Fallback execution using basic exec() when IPython is not available."""
        if self._fallback_executor:
            return self._fallback_executor.execute(code, namespace, timeout)
            
        # Basic implementation as fallback
        start_time = time.time()
        
        exec_namespace = namespace or self.default_namespace.copy()
        if '__builtins__' not in exec_namespace:
            exec_namespace['__builtins__'] = __builtins__
            
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result = {
            'output': '',
            'error': '',
            'success': False,
            'execution_time_ms': 0,
            'execution_count': 0,
            'rich_output': [],
            'completions': []
        }
        
        try:
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):
                exec(code, exec_namespace)
            
            result['output'] = stdout_capture.getvalue()
            result['error'] = stderr_capture.getvalue()
            result['success'] = True
            
        except Exception as e:
            result['output'] = stdout_capture.getvalue()
            result['error'] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result['success'] = False
            
        finally:
            end_time = time.time()
            result['execution_time_ms'] = round((end_time - start_time) * 1000, 2)
            
        return result
    
    
    def _format_rich_outputs(self, outputs) -> List[Dict[str, Any]]:
        """Format rich outputs from IPython display system."""
        formatted_outputs = []
        
        for output in outputs:
            if hasattr(output, 'data'):
                output_data = {
                    'output_type': getattr(output, 'output_type', 'display_data'),
                    'data': {}
                }
                
                # Handle different MIME types
                for mime_type, content in output.data.items():
                    if mime_type == 'text/html':
                        output_data['data']['html'] = content
                    elif mime_type == 'text/plain':
                        output_data['data']['text'] = content
                    elif mime_type == 'image/png':
                        output_data['data']['image_png'] = content
                    elif mime_type == 'application/json':
                        output_data['data']['json'] = content
                    else:
                        output_data['data'][mime_type] = str(content)
                
                formatted_outputs.append(output_data)
        
        return formatted_outputs
    
    def get_completions(self, code: str, cursor_pos: int, shell_instance: Any = None) -> List[str]:
        """Get tab completions for the given code at cursor position."""
        if not IPYTHON_AVAILABLE or shell_instance is None:
            return []
            
        try:
            # Extract current line from cursor position
            current_line = code[:cursor_pos].split('\n')[-1]
            
            # Get completions using IPython's stable API
            _, matches = shell_instance.complete(current_line, current_line, len(current_line))
            
            # Clean up completions: remove leading dots and deduplicate
            return list(dict.fromkeys(  # Preserves order while deduplicating
                comp.lstrip('.') for comp in matches 
                if comp and comp.lstrip('.')  # Filter out empty/dot-only completions
            ))
            
        except Exception as e:
            # Log the error for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Completion error for code '{code[:cursor_pos]}': {e}")
            return []
    
    def inspect_object(self, obj_name: str, shell_instance: Any = None) -> Dict[str, Any]:
        """Get detailed information about an object."""
        if not IPYTHON_AVAILABLE or shell_instance is None:
            return {'error': 'IPython not available'}
            
        try:
            obj = shell_instance.user_ns.get(obj_name)
            if obj is None:
                return {'error': f'Object "{obj_name}" not found'}
                
            # Use IPython's inspector
            inspector_result = shell_instance.inspector.info(obj, obj_name, detail_level=1)
            
            return {
                'name': obj_name,
                'type': str(type(obj).__name__),
                'docstring': getattr(obj, '__doc__', '') or '',
                'source': inspector_result.get('source', ''),
                'file': inspector_result.get('file', ''),
                'definition': inspector_result.get('definition', ''),
            }
            
        except Exception as e:
            return {'error': f'Error inspecting object: {e}'}
    
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


# Backward compatibility alias and fallback class
class SafeREPLExecutor:
    """
    Legacy SafeREPLExecutor for backward compatibility.
    
    This is now a thin wrapper around the basic execution functionality
    when IPython is not available, maintaining the original API.
    """
    
    def __init__(self):
        self.default_namespace = {
            '__builtins__': __builtins__,
            'time': time,
            'traceback': traceback,
        }
    
    def execute(self, code: str, namespace: Optional[Dict[str, Any]] = None, timeout: float = 5.0) -> Dict[str, Any]:
        """Execute code using basic exec() - legacy implementation."""
        start_time = time.time()
        
        exec_namespace = namespace or self.default_namespace.copy()
        if '__builtins__' not in exec_namespace:
            exec_namespace['__builtins__'] = __builtins__
            
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result = {
            'output': '',
            'error': '',
            'success': False,
            'execution_time_ms': 0
        }
        
        try:
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):
                exec(code, exec_namespace)
            
            result['output'] = stdout_capture.getvalue()
            result['error'] = stderr_capture.getvalue()
            result['success'] = True
            
        except Exception as e:
            result['output'] = stdout_capture.getvalue()
            result['error'] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result['success'] = False
            
        finally:
            end_time = time.time()
            result['execution_time_ms'] = round((end_time - start_time) * 1000, 2)
            
        return result
    
    def get_inspector_namespace(self, host_instance=None) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        namespace = self.default_namespace.copy()
        
        if host_instance:
            namespace.update({
                'host': host_instance,
                'space_registry': getattr(host_instance, 'space_registry', None),
                'event_loop': getattr(host_instance, 'event_loop', None),
                'activity_client': getattr(host_instance, 'activity_client', None),
            })
            
            try:
                from elements.space_registry import SpaceRegistry
                namespace['SpaceRegistry'] = SpaceRegistry
                namespace['spaces'] = SpaceRegistry.get_instance()
            except ImportError:
                pass
        
        return namespace