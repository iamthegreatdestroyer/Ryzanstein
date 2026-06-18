"""Pytest configuration for Ryot tests.

Adds the sigmalang source directory to sys.path so tests can import
sigmalang without it being installed as a package.
"""
import sys
from pathlib import Path

# sigmalang lives one level up in the same Layer-4-Storage directory
SIGMALANG_ROOT = Path(__file__).parent.parent.parent / "sigmalang"
if SIGMALANG_ROOT.exists() and str(SIGMALANG_ROOT) not in sys.path:
    sys.path.insert(0, str(SIGMALANG_ROOT))
