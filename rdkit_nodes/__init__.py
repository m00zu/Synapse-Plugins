"""
rdkit_nodes — Cheminformatics plugin for Synapse
=================================================
Provides node classes for chemical structure handling using RDKit.

Vendor setup
------------
Before using these nodes, populate the vendor directory by running::

    python plugins/rdkit_nodes/setup_vendor.py /path/to/rdkit-*.whl

This extracts the RDKit wheel into plugins/rdkit_nodes/vendor/rdkit/ so the
plugin works inside the frozen .app without a system-level rdkit installation.
"""
from __future__ import annotations

import pathlib
import sys

# ── Vendor injection ──────────────────────────────────────────────────────────
_vendor = pathlib.Path(__file__).parent / 'vendor'
if _vendor.is_dir() and str(_vendor) not in sys.path:
    sys.path.insert(0, str(_vendor))

try:
    from rdkit import Chem  # noqa: F401 — validate rdkit is importable
except ImportError as _e:
    raise ImportError(
        "RDKit not found in vendor/. Run:\n"
        "  python plugins/rdkit_nodes/setup_vendor.py /path/to/rdkit-*.whl\n"
        f"Original error: {_e}"
    ) from _e

# ── Import node classes (the plugin loader discovers them here) ───────────────
from .chem_nodes import *  # noqa: F401,F403
from .docking_nodes import *  # noqa: F401,F403
from .viewer_nodes import *  # noqa: F401,F403
