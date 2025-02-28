"""
Environments Module
Defines the hierarchical environment system for the Bot Framework.
"""

from environments.base import Environment
from environments.system import SystemEnvironment
from environments.web import WebEnvironment
from environments.messaging import MessagingEnvironment
from environments.file import FileEnvironment
from environments.manager import EnvironmentManager