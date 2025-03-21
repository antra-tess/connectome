#!/usr/bin/env python
"""
Helper script to run tests with the proper Python path configuration.
"""

import os
import sys
import pytest

# Add the parent directory to sys.path to allow imports to work properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    # Get command line arguments
    args = sys.argv[1:] if len(sys.argv) > 1 else ["activity/"]
    
    # Run pytest with the specified arguments
    sys.exit(pytest.main(args)) 