"""Pytest configuration for imaris_3d_nodes tests.

Sets up Synapse sys.modules aliases so that plugins can use
`from nodes.base import ...` import style during testing.
"""
import sys
from pathlib import Path

# Set up IMMEDIATELY at module import time (before pytest collection)
# so parent package __init__.py can import its modules
_PYSIDE_NODE = Path('/Users/s/Desktop/demo/PySide_Node')
if str(_PYSIDE_NODE) not in sys.path:
    sys.path.insert(0, str(_PYSIDE_NODE))

# Register sys.modules aliases just like app.py does, so that
# `from nodes.base import ...` resolves to synapse.nodes.base
import synapse
from synapse import nodes as _nodes_pkg
from synapse.nodes import base as _nodes_base_pkg
sys.modules['nodes'] = _nodes_pkg
sys.modules['nodes.base'] = _nodes_base_pkg
