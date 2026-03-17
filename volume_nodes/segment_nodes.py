"""
segment_nodes.py — 3D labeling and watershed segmentation nodes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from nodes.base import BaseExecutionNode, PORT_COLORS
from nodes.base import BaseImageProcessNode
from data_models import TableData
from .data_model import VolumeMaskData, VolumeLabelData

_MC = PORT_COLORS.get('volume_mask', (180, 90, 30))
_LC = PORT_COLORS.get('volume_label', (240, 180, 60))


def _mid_slice_label_preview(label_vol: np.ndarray) -> Image.Image:
    """Render the middle Z-slice of a label volume as a colored PIL image."""
    from nodes.vision_nodes import _label_palette
    mid = label_vol[label_vol.shape[0] // 2]
    labels = np.unique(mid)
    labels = labels[labels != 0]
    n = int(labels.max()) if len(labels) else 0
    palette = _label_palette(max(n, 1))
    rgb = np.zeros((*mid.shape, 3), dtype=np.uint8)
    for lbl in labels:
        rgb[mid == lbl] = palette[(int(lbl) - 1) % len(palette)]
    return Image.fromarray(rgb, 'RGB')


# ══════════════════════════════════════════════════════════════════════════════
#  Label3DNode
# ══════════════════════════════════════════════════════════════════════════════

class Label3DNode(BaseImageProcessNode):
    """Label connected components in a 3D binary volume.

    Outputs a label volume (integer per region) and a properties table
    with volume, centroid, bounding box, and equivalent diameter.

    Keywords: label, connected, components, 3D, regions, 標記, 連通, 體積
    """
    __identifier__ = 'nodes.Volume.Morphology'
    NODE_NAME      = '3D Label'
    PORT_SPEC      = {'inputs': ['volume_mask'],
                      'outputs': ['volume_label', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('volume_mask', color=_MC)
        self.add_output('volume_label', color=_LC)
        self.add_output('table', color=PORT_COLORS['table'])
        self.add_combo_menu('connectivity', 'Connectivity',
                            items=['6 (faces)', '18 (faces+edges)',
                                   '26 (faces+edges+corners)'])
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from scipy.ndimage import label as nd_label, generate_binary_structure
        from skimage.measure import regionprops_table

        port = self.inputs().get('volume_mask')
        if not port or not port.connected_ports():
            return False, "No volume mask connected"
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if not isinstance(data, VolumeMaskData):
            return False, "Input must be VolumeMaskData"

        mask = np.asarray(data.payload, dtype=bool)
        conn_str = str(self.get_property('connectivity'))
        conn = 1
        if '18' in conn_str:
            conn = 2
        elif '26' in conn_str:
            conn = 3

        struct = generate_binary_structure(3, conn)
        self.set_progress(30)
        labeled, n_labels = nd_label(mask, structure=struct)
        self.set_progress(60)

        # Region properties
        props = regionprops_table(labeled, properties=[
            'label', 'area', 'centroid', 'bbox',
            'equivalent_diameter_area',
        ])
        df = pd.DataFrame(props)
        # Rename for clarity
        rename = {
            'area': 'volume_voxels',
            'equivalent_diameter_area': 'equivalent_diameter',
        }
        # Handle centroid columns (centroid-0, centroid-1, centroid-2)
        for c in df.columns:
            if c == 'centroid-0':
                rename[c] = 'centroid_z'
            elif c == 'centroid-1':
                rename[c] = 'centroid_y'
            elif c == 'centroid-2':
                rename[c] = 'centroid_x'
        df = df.rename(columns=rename)
        # Round float columns
        for c in df.select_dtypes(include='float').columns:
            df[c] = df[c].round(2)

        self.set_progress(80)
        self.output_values['volume_label'] = VolumeLabelData(
            payload=labeled.astype(np.int32), spacing=data.spacing)
        self.output_values['table'] = TableData(payload=df)
        self.set_display(_mid_slice_label_preview(labeled))
        self.set_progress(100)
        return True, None


# ══════════════════════════════════════════════════════════════════════════════
#  Watershed3DNode
# ══════════════════════════════════════════════════════════════════════════════

class Watershed3DNode(BaseImageProcessNode):
    """3D marker-based watershed to separate touching objects.

    Pipeline: distance transform → peak detection → watershed.

    Keywords: watershed, split, separate, touching, 3D, 分水嶺, 分割, 體積
    """
    __identifier__ = 'nodes.Volume.Morphology'
    NODE_NAME      = '3D Watershed'
    PORT_SPEC      = {'inputs': ['volume_mask'],
                      'outputs': ['volume_label', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('volume_mask', color=_MC)
        self.add_output('volume_label', color=_LC)
        self.add_output('table', color=PORT_COLORS['table'])
        self._add_int_spinbox('min_distance', 'Min Object Sep. (px)',
                              value=10, min_val=1, max_val=500)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from scipy.ndimage import (distance_transform_edt,
                                   label as nd_label)
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
        from skimage.measure import regionprops_table

        port = self.inputs().get('volume_mask')
        if not port or not port.connected_ports():
            return False, "No volume mask connected"
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if not isinstance(data, VolumeMaskData):
            return False, "Input must be VolumeMaskData"

        mask = np.asarray(data.payload, dtype=bool)
        min_dist = int(self.get_property('min_distance'))

        self.set_progress(20)
        dist = distance_transform_edt(mask)
        self.set_progress(40)

        coords = peak_local_max(dist, min_distance=min_dist, labels=mask)
        markers_bool = np.zeros(mask.shape, dtype=bool)
        markers_bool[tuple(coords.T)] = True
        markers, _ = nd_label(markers_bool)
        self.set_progress(60)

        labeled = watershed(-dist, markers, mask=mask)
        n_labels = labeled.max()
        self.set_progress(75)

        # Properties table
        props = regionprops_table(labeled, properties=[
            'label', 'area', 'centroid', 'equivalent_diameter_area',
        ])
        df = pd.DataFrame(props)
        rename = {
            'area': 'volume_voxels',
            'equivalent_diameter_area': 'equivalent_diameter',
        }
        for c in df.columns:
            if c == 'centroid-0':
                rename[c] = 'centroid_z'
            elif c == 'centroid-1':
                rename[c] = 'centroid_y'
            elif c == 'centroid-2':
                rename[c] = 'centroid_x'
        df = df.rename(columns=rename)
        for c in df.select_dtypes(include='float').columns:
            df[c] = df[c].round(2)

        self.set_progress(85)
        self.output_values['volume_label'] = VolumeLabelData(
            payload=labeled.astype(np.int32), spacing=data.spacing)
        self.output_values['table'] = TableData(payload=df)
        self.set_display(_mid_slice_label_preview(labeled))
        self.set_progress(100)
        return True, None
