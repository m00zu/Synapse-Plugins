"""
sam2_nodes — SAM2 interactive segmentation plugin for Synapse
=============================================================
Click on objects in images to generate precise segmentation masks
using Meta's Segment Anything Model 2 (SAM2) via ONNX Runtime.

Models are downloaded automatically on first use from HuggingFace.
"""
from __future__ import annotations

import pathlib
import sys

# ── Vendor injection (lightweight — no heavy imports here) ────────────────────
_vendor = pathlib.Path(__file__).parent / 'vendor'
if _vendor.is_dir() and str(_vendor) not in sys.path:
    sys.path.insert(0, str(_vendor))

from .sam2_segment import *       # noqa: F401,F403
from .tracking import *           # noqa: F401,F403
from .video_utils import *        # noqa: F401,F403
from .grounding import *          # noqa: F401,F403
from .video_analyze import *      # noqa: F401,F403
from .cellpose import *           # noqa: F401,F403
from .particle_tracking import *  # noqa: F401,F403
