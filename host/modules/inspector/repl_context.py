"""
REPL Context Manager for Connectome Inspector

Manages REPL sessions with different scopes (global, space-level, component-level, etc).
Tracks session state, history, and provides appropriate namespaces for each context.
"""

import json
import time
import pprint
import weakref
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from .repl_executor import IPythonREPLExecutor


class REPLContextManager:
    """
    Manages REPL contexts and sessions for the Connectome Inspector.
    
    Provides scoped REPL environments for debugging different parts of the system:
    - Global: Full system access with host instance
    - Space: Scoped to a specific space and its elements
    - Component: Scoped to a specific component instance
    - Element: Scoped to a specific element and its components
    """
    
    def __init__(self, host_instance):
        """
        Initialize the REPL context manager.
        
        Args:
            host_instance: Reference to the main Host instance
        """
        self.host_instance = host_instance
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Keep weak reference to host to avoid circular references
        self._host_ref = weakref.ref(host_instance) if host_instance else None
        
        # Initialize IPython REPL executor for code execution
        self.executor = IPythonREPLExecutor()
    
    def create_context(self, context_type: str, context_id: str, target_object: Any = None) -> Dict[str, Any]:
        """
        Create a new REPL context/session.
        
        Args:
            context_type: Type of context ('global', 'space', 'component', 'element')
            context_id: Unique identifier for this context instance
            target_object: Optional target object for scoped contexts
            
        Returns:
            Session dictionary with id, type, namespace, history, and metadata
        """
        session_id = f"{context_type}:{context_id}"
        
        # Create base session structure
        session = {
            'id': session_id,
            'type': context_type,
            'context_id': context_id,
            'namespace': {},
            'ipython_shell': None,  # New: IPython shell instance
            'history': [],
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'target_ref': None
        }
        
        # Store weak reference to target object if provided
        if target_object is not None:
            session['target_ref'] = weakref.ref(target_object)
        
        # Build namespace based on context type
        session['namespace'] = self._build_namespace(context_type, context_id, target_object)
        
        # Create IPython shell instance for this session
        session['ipython_shell'] = self.executor.create_shell(session['namespace'])
        
        # Store session
        self.sessions[session_id] = session
        
        return session
    
    def _build_namespace(self, context_type: str, context_id: str, target_object: Any = None) -> Dict[str, Any]:
        """
        Build the appropriate namespace for a context type.
        
        Args:
            context_type: Type of context
            context_id: Context identifier
            target_object: Target object for scoped contexts
            
        Returns:
            Namespace dictionary with appropriate objects and utilities
        """
        # Base namespace with useful utilities
        namespace = {
            '__builtins__': __builtins__,
            'json': json,
            'time': time,
            'pprint': pprint,
            'datetime': datetime,
            'weakref': weakref,
        }
        
        if context_type == 'global':
            # Global context: full system access
            host = self._host_ref() if self._host_ref else None
            if host:
                namespace.update({
                    'host': host,
                    'space_registry': getattr(host, 'space_registry', None),
                    'event_loop': getattr(host, 'event_loop', None),
                    'activity_client': getattr(host, 'activity_client', None),
                })
                
                # Add space registry if available
                try:
                    from elements.space_registry import SpaceRegistry
                    namespace['SpaceRegistry'] = SpaceRegistry
                    namespace['spaces'] = SpaceRegistry.get_instance()
                except ImportError:
                    pass
        
        elif context_type == 'space':
            # Space context: scoped to a specific space
            # TODO: Implement space-specific namespace
            namespace['space_id'] = context_id
            if target_object:
                namespace['space'] = target_object
        
        elif context_type == 'component':
            # Component context: scoped to a specific component
            # TODO: Implement component-specific namespace
            namespace['component_id'] = context_id
            if target_object:
                namespace['component'] = target_object
        
        elif context_type == 'element':
            # Element context: scoped to a specific element
            # TODO: Implement element-specific namespace
            namespace['element_id'] = context_id
            if target_object:
                namespace['element'] = target_object
        
        return namespace
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an existing REPL session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session dictionary or None if not found
        """
        session = self.sessions.get(session_id)
        if session:
            # Update last accessed time
            session['last_accessed'] = datetime.now().isoformat()
        return session
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active REPL sessions.
        
        Returns:
            List of session metadata (without full namespaces for brevity)
        """
        sessions_info = []
        for session_id, session in self.sessions.items():
            # Check if target object is still alive (for weak references)
            target_alive = True
            if session.get('target_ref'):
                target_alive = session['target_ref']() is not None
            
            session_info = {
                'id': session['id'],
                'type': session['type'],
                'context_id': session['context_id'],
                'created_at': session['created_at'],
                'last_accessed': session['last_accessed'],
                'history_length': len(session['history']),
                'target_alive': target_alive
            }
            sessions_info.append(session_info)
        
        return sessions_info
    
    def get_session_history(self, session_id: str, limit: int = 50, offset: int = 0) -> Optional[Dict[str, Any]]:
        """
        Get the execution history for a specific REPL session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of history entries to return (default 50)
            offset: Number of entries to skip from the start (default 0)
            
        Returns:
            Dictionary with session info and paginated history, or None if session not found
        """
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        # Get paginated history (newest first)
        history = session['history']
        total_count = len(history)
        
        # Reverse slice to get newest first, then apply pagination
        paginated_history = history[::-1][offset:offset + limit]
        
        return {
            'session_id': session_id,
            'context_type': session['type'],
            'context_id': session['context_id'],
            'created_at': session['created_at'],
            'last_accessed': session['last_accessed'],
            'history': paginated_history,
            'total_history_count': total_count,
            'returned_count': len(paginated_history),
            'offset': offset,
            'has_more': offset + len(paginated_history) < total_count
        }
    
    def destroy_session(self, session_id: str) -> bool:
        """
        Destroy a REPL session and clean up resources.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was destroyed, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def execute_in_context(self, session_id: str, code: str, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Execute Python code in a specific REPL session context.
        
        Args:
            session_id: Session identifier
            code: Python code to execute
            timeout: Execution timeout in seconds
            
        Returns:
            Execution result with additional namespace_changes field
        """
        # Check if session exists
        session = self.sessions.get(session_id)
        if not session:
            return {
                'output': '',
                'error': f'Session "{session_id}" not found',
                'success': False,
                'execution_time_ms': 0,
                'execution_count': 0,
                'rich_output': [],
                'namespace_changes': []
            }
        
        # Get IPython shell and namespace before execution
        shell = session['ipython_shell']
        namespace = session['namespace']
        keys_before = set(namespace.keys())
        
        # Execute code using IPython executor with session shell
        result = self.executor.execute(code, namespace, timeout, shell_instance=shell)
        
        # Detect namespace changes (IPython shell user_ns)
        if shell:
            keys_after = set(shell.user_ns.keys())
            new_variables = keys_after - keys_before
            # Update session namespace with new variables
            session['namespace'].update({k: shell.user_ns[k] for k in new_variables})
        else:
            keys_after = set(namespace.keys())
            new_variables = keys_after - keys_before
        
        # Add namespace changes to result
        result['namespace_changes'] = sorted(list(new_variables))
        
        # Track execution in session history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'code': code,
            'output': result['output'],
            'error': result['error'],
            'success': result['success'],
            'execution_time_ms': result['execution_time_ms'],
            'execution_count': result.get('execution_count', 0),
            'rich_output': result.get('rich_output', []),
            'namespace_changes': result['namespace_changes']
        }
        session['history'].append(history_entry)
        
        # Update last accessed time
        session['last_accessed'] = datetime.now().isoformat()
        
        return result
    
    def get_completions(self, session_id: str, code: str, cursor_pos: int) -> Dict[str, Any]:
        """
        Get tab completions for code at cursor position in a specific session.
        
        Args:
            session_id: Session identifier
            code: Code string to get completions for
            cursor_pos: Cursor position in the code
            
        Returns:
            Dict with completions list and metadata
        """
        session = self.sessions.get(session_id)
        if not session:
            return {
                'completions': [],
                'error': f'Session "{session_id}" not found'
            }
            
        try:
            shell = session['ipython_shell']
            completions = self.executor.get_completions(code, cursor_pos, shell)
            
            return {
                'completions': completions,
                'cursor_start': cursor_pos,
                'cursor_end': cursor_pos,
                'metadata': {
                    'session_id': session_id,
                    'code_length': len(code)
                }
            }
            
        except Exception as e:
            return {
                'completions': [],
                'error': f'Error getting completions: {str(e)}'
            }
    
    def inspect_object(self, session_id: str, obj_name: str) -> Dict[str, Any]:
        """
        Get detailed information about an object in a specific session.
        
        Args:
            session_id: Session identifier
            obj_name: Name of object to inspect
            
        Returns:
            Dict with object information
        """
        session = self.sessions.get(session_id)
        if not session:
            return {'error': f'Session "{session_id}" not found'}
            
        try:
            shell = session['ipython_shell']
            inspection_result = self.executor.inspect_object(obj_name, shell)
            
            # Add session context
            inspection_result['session_id'] = session_id
            inspection_result['session_type'] = session['type']
            
            return inspection_result
            
        except Exception as e:
            return {'error': f'Error inspecting object: {str(e)}'}
    
    def cleanup_dead_sessions(self) -> int:
        """
        Clean up sessions where target objects have been garbage collected.
        
        Returns:
            Number of sessions cleaned up
        """
        dead_sessions = []
        
        for session_id, session in self.sessions.items():
            if session.get('target_ref'):
                if session['target_ref']() is None:
                    dead_sessions.append(session_id)
        
        for session_id in dead_sessions:
            del self.sessions[session_id]
        
        return len(dead_sessions)