"""
Shell Module

Provides the Shell component, which serves as the agentic loop container enclosing 
the model mind, providing a runtime environment and coordination layer.

The Shell is responsible for:
1. Activating in response to external events or internal timers
2. Processing agent actions and managing their execution
3. Managing memory formation
4. Providing internal tools accessible to the agent
5. Presenting the Inner Space as the primary container for the agent
"""

# Base Shell
from shell.base_shell import BaseShell

# Shell Implementations
from shell.shell_implementation.two_phase_shell import TwoPhaseShell
from shell.shell_implementation.single_phase_shell import SinglePhaseShell

# HUD
from shell.hud import HUD, RenderingRequest, RenderingResponse

# Context Manager
from shell.context_manager import ContextManager, ContextRequest, ContextResponse

__all__ = [
    'BaseShell',
    'TwoPhaseShell',
    'SinglePhaseShell',
    'HUD',
    'RenderingRequest',
    'RenderingResponse',
    'ContextManager',
    'ContextRequest',
    'ContextResponse'
]
