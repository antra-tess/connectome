#!/usr/bin/env python3
"""
Connectome Inspector

Convenient entry point for inspecting running Connectome host instances.
Provides the same functionality as host.cli_inspect but with a shorter command.

Usage:
    python -m host.inspect status
    python -m host.inspect spaces
    python -m host.inspect agents
    python -m host.inspect --mock health
"""

from host.cli_inspect import main
import asyncio
import sys

if __name__ == "__main__":
    # Simply delegate to the main CLI inspector
    asyncio.run(main())