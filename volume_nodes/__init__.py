"""
volume_nodes -- 3D volume processing plugin for Synapse
======================================================
Provides nodes for loading, processing, segmenting, and visualizing
3D volumetric data (Z-stack TIFFs).
"""
from __future__ import annotations

from nodes.base import PORT_COLORS
try:
    # Available in Synapse with the port-type registry.  Older
    # Synapse versions get a no-op fallback so this plugin still loads.
    from nodes.base import register_port_type
except ImportError:
    def register_port_type(name, cls):  # noqa: ARG001
        pass
from .data_model import (
    VolumeData, VolumeMaskData, VolumeLabelData, VolumeColorData,
)

# Register custom port colours for volume data types
PORT_COLORS['volume']       = (220, 120,  50)  # Orange-amber
PORT_COLORS['volume_mask']  = (180,  90,  30)  # Darker amber
PORT_COLORS['volume_label'] = (240, 180,  60)  # Gold
PORT_COLORS['volume_color'] = (200,  80, 150)  # Pink-magenta

# Register port-type -> data class for connection-time type checking.
register_port_type('volume',       VolumeData)
register_port_type('volume_mask',  VolumeMaskData)
register_port_type('volume_label', VolumeLabelData)
register_port_type('volume_color', VolumeColorData)

# ── Export node classes (plugin loader discovers them here) ──────────────────
from .io_nodes import *        # noqa: F401,F403
from .process_nodes import *   # noqa: F401,F403
from .segment_nodes import *   # noqa: F401,F403
from .image_ops_nodes import * # noqa: F401,F403
from .viewer_nodes import *    # noqa: F401,F403
