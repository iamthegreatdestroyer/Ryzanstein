"""Pytest configuration for Ryot tests.

Adds the sigmalang source directory to sys.path so tests can import
sigmalang without it being installed as a package.
"""
import sys
from pathlib import Path

# sigmalang lives in the sibling Layer-4-Storage-sigmalang repo
SIGMALANG_ROOT = Path(__file__).parent.parent.parent / "Layer-4-Storage-sigmalang"
if SIGMALANG_ROOT.exists() and str(SIGMALANG_ROOT) not in sys.path:
    sys.path.insert(0, str(SIGMALANG_ROOT))
