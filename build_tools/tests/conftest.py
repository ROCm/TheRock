"""Pytest configuration for build_tools tests."""

import sys
from pathlib import Path

# Add build_tools to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
