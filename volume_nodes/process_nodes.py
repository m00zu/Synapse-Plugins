"""
process_nodes.py — 3D volume morphology and mask processing nodes.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from nodes.base import BaseExecutionNode, PORT_COLORS
from nodes.base import BaseImageProcessNode
from .data_model import VolumeData, VolumeMaskData

_VC = PORT_COLORS.get('volume', (220, 120, 50))
_MC = PORT_COLORS.get('volume_mask', (180, 90, 30))


# ── helpers ──────────────────────────────────────────────────────────────────

def _mid_slice_preview(vol: np.ndarray) -> Image.Image:
    """Return a PIL Image of the middle Z-slice, normalized to 0-255."""
    mid = vol[vol.shape[0] // 2]
    if mid.dtype == bool:
        return Image.fromarray((mid.astype(np.uint8) * 255), 'L')
    mn, mx = float(mid.min()), float(mid.max())
    if mx > mn:
        mid = ((mid.astype(np.float64) - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        mid = np.zeros(mid.shape, dtype=np.uint8)
    return Image.fromarray(mid, 'L')


def _get_volume_mask(node) -> VolumeMaskData | None:
    port = node.inputs().get('volume_mask')
    if not port or not port.connected_ports():
        return None
    cp = port.connected_ports()[0]
    data = cp.node().output_values.get(cp.name())
    if isinstance(data, VolumeMaskData):
        return data
    return None


def _get_volume(node) -> VolumeData | None:
    port = node.inputs().get('volume')
    if not port or not port.connected_ports():
        return None
    cp = port.connected_ports()[0]
    data = cp.node().output_values.get(cp.name())
    if isinstance(data, VolumeData):
        return data
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Threshold3DNode
# ══════════════════════════════════════════════════════════════════════════════

class Threshold3DNode(BaseImageProcessNode):
    """Threshold a 3D volume to produce a binary volume mask.

    Methods: manual value, Otsu auto-threshold, Li auto-threshold.

    Keywords: threshold, binarize, 3D, volume, segment, 閾值, 二值化, 體積
    """
    __identifier__ = 'nodes.Volume.Filters'
    NODE_NAME      = '3D Threshold'
    PORT_SPEC      = {'inputs': ['volume'], 'outputs': ['volume_mask']}

    def __init__(self):
        super().__init__()
        self.add_input('volume', color=_VC)
        self.add_output('volume_mask', color=_MC)
        self.add_combo_menu('method', 'Method', items=['manual', 'otsu', 'li'])
        self._add_int_spinbox('threshold', 'Threshold', value=128,
                              min_val=0, max_val=65535)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        data = _get_volume(self)
        if data is None:
            return False, "No volume connected"
        vol = data.payload
        method = str(self.get_property('method'))

        self.set_progress(30)
        if method == 'otsu':
            from skimage.filters import threshold_otsu
            t = threshold_otsu(vol)
        elif method == 'li':
            from skimage.filters import threshold_li
            t = threshold_li(vol)
        else:
            t = int(self.get_property('threshold'))

        mask = vol > t
        self.set_progress(80)

        self.output_values['volume_mask'] = VolumeMaskData(
            payload=mask, spacing=data.spacing)
        self.set_display(_mid_slice_preview(mask))
        self.set_progress(100)
        return True, None


# ══════════════════════════════════════════════════════════════════════════════
#  DistanceRingMask3DNode  ★ HIGH PRIORITY
# ══════════════════════════════════════════════════════════════════════════════

class DistanceRingMask3DNode(BaseImageProcessNode):
    """Expand a 3D mask outward by a given distance (ring / shell mask).

    Uses the Euclidean distance transform.  The *spacing-aware* option
    accounts for anisotropic voxel dimensions (e.g. Z ≠ XY).

    Keywords: distance, ring, expand, shell, 3D, dilate zone, 距離, 環狀, 膨脹, 體積
    """
    __identifier__ = 'nodes.Volume.Morphology'
    NODE_NAME      = '3D Distance Ring Mask'
    PORT_SPEC      = {'inputs': ['volume_mask'], 'outputs': ['volume_mask']}

    def __init__(self):
        super().__init__()
        self.add_input('volume_mask', color=_MC)
        self.add_output('volume_mask', color=_MC)
        self._add_int_spinbox('distance', 'Distance (px)', value=10,
                              min_val=1, max_val=50000)
        self.add_checkbox('include_original', '', text='Include Original Mask',
                          state=False)
        self.add_checkbox('spacing_aware', '', text='Spacing-Aware (anisotropic)',
                          state=True)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from scipy.ndimage import distance_transform_edt

        data = _get_volume_mask(self)
        if data is None:
            return False, "No volume mask connected"

        mask = np.asarray(data.payload, dtype=bool)
        dist = int(self.get_property('distance'))
        spacing = data.spacing if self.get_property('spacing_aware') else None

        self.set_progress(30)
        edt = distance_transform_edt(~mask, sampling=spacing)
        ring = (edt > 0) & (edt <= dist)

        if self.get_property('include_original'):
            ring = ring | mask

        self.set_progress(80)
        self.output_values['volume_mask'] = VolumeMaskData(
            payload=ring, spacing=data.spacing)
        self.set_display(_mid_slice_preview(ring))
        self.set_progress(100)
        return True, None


# ══════════════════════════════════════════════════════════════════════════════
#  RemoveSmallObjects3DNode
# ══════════════════════════════════════════════════════════════════════════════

class RemoveSmallObjects3DNode(BaseImageProcessNode):
    """Remove small 3D connected components from a volume mask.

    Keywords: remove, small, objects, 3D, clean, debris, 移除, 去噪, 體積
    """
    __identifier__ = 'nodes.Volume.Morphology'
    NODE_NAME      = '3D Remove Small Obj'
    PORT_SPEC      = {'inputs': ['volume_mask'], 'outputs': ['volume_mask']}

    def __init__(self):
        super().__init__()
        self.add_input('volume_mask', color=_MC)
        self.add_output('volume_mask', color=_MC)
        self._add_int_spinbox('min_size', 'Min Size (voxels)', value=500,
                              min_val=1, max_val=9999999)
        self.add_combo_menu('connectivity', 'Connectivity',
                            items=['6 (faces)', '18 (faces+edges)',
                                   '26 (faces+edges+corners)'])
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.morphology import remove_small_objects

        data = _get_volume_mask(self)
        if data is None:
            return False, "No volume mask connected"

        mask = np.asarray(data.payload, dtype=bool)
        min_size = int(self.get_property('min_size'))
        conn_str = str(self.get_property('connectivity'))
        conn = 1
        if '18' in conn_str:
            conn = 2
        elif '26' in conn_str:
            conn = 3

        self.set_progress(30)
        clean = remove_small_objects(mask, max_size=max(1, min_size - 1), connectivity=conn)
        self.set_progress(80)

        self.output_values['volume_mask'] = VolumeMaskData(
            payload=clean, spacing=data.spacing)
        self.set_display(_mid_slice_preview(clean))
        self.set_progress(100)
        return True, None


# ══════════════════════════════════════════════════════════════════════════════
#  FillHoles3DNode
# ══════════════════════════════════════════════════════════════════════════════

class FillHoles3DNode(BaseImageProcessNode):
    """Fill small holes / voids inside a 3D volume mask.

    Keywords: fill, holes, 3D, close, interior, 填孔, 體積, 閉合
    """
    __identifier__ = 'nodes.Volume.Morphology'
    NODE_NAME      = '3D Fill Holes'
    PORT_SPEC      = {'inputs': ['volume_mask'], 'outputs': ['volume_mask']}

    def __init__(self):
        super().__init__()
        self.add_input('volume_mask', color=_MC)
        self.add_output('volume_mask', color=_MC)
        self._add_int_spinbox('max_hole_size', 'Max Hole Size (voxels)',
                              value=500, min_val=1, max_val=9999999)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.morphology import remove_small_holes

        data = _get_volume_mask(self)
        if data is None:
            return False, "No volume mask connected"

        mask = np.asarray(data.payload, dtype=bool)
        max_size = int(self.get_property('max_hole_size'))

        self.set_progress(30)
        filled = remove_small_holes(mask, area_threshold=max_size)
        self.set_progress(80)

        self.output_values['volume_mask'] = VolumeMaskData(
            payload=filled, spacing=data.spacing)
        self.set_display(_mid_slice_preview(filled))
        self.set_progress(100)
        return True, None


# ══════════════════════════════════════════════════════════════════════════════
#  Erode3DNode / Dilate3DNode
# ══════════════════════════════════════════════════════════════════════════════

class Erode3DNode(BaseImageProcessNode):
    """3D morphological erosion with ball / cube / octahedron kernel.

    Keywords: erode, shrink, morphology, 3D, 侵蝕, 形態學, 體積
    """
    __identifier__ = 'nodes.Volume.Morphology'
    NODE_NAME      = '3D Erode'
    PORT_SPEC      = {'inputs': ['volume_mask'], 'outputs': ['volume_mask']}

    def __init__(self):
        super().__init__()
        self.add_input('volume_mask', color=_MC)
        self.add_output('volume_mask', color=_MC)
        self._add_int_spinbox('radius', 'Radius (voxels)', value=2,
                              min_val=1, max_val=50)
        self.add_combo_menu('kernel', 'Kernel', items=['ball', 'cube', 'octahedron'])
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.morphology import erosion, ball, cube, octahedron

        data = _get_volume_mask(self)
        if data is None:
            return False, "No volume mask connected"

        mask = np.asarray(data.payload, dtype=np.uint8)
        r = int(self.get_property('radius'))
        k = str(self.get_property('kernel'))

        fp = ball(r) if k == 'ball' else (cube(2 * r + 1) if k == 'cube'
                                          else octahedron(r))
        self.set_progress(30)
        result = erosion(mask, footprint=fp) > 0
        self.set_progress(80)

        self.output_values['volume_mask'] = VolumeMaskData(
            payload=result, spacing=data.spacing)
        self.set_display(_mid_slice_preview(result))
        self.set_progress(100)
        return True, None


class Dilate3DNode(BaseImageProcessNode):
    """3D morphological dilation with ball / cube / octahedron kernel.

    Keywords: dilate, expand, morphology, 3D, 膨脹, 形態學, 體積
    """
    __identifier__ = 'nodes.Volume.Morphology'
    NODE_NAME      = '3D Dilate'
    PORT_SPEC      = {'inputs': ['volume_mask'], 'outputs': ['volume_mask']}

    def __init__(self):
        super().__init__()
        self.add_input('volume_mask', color=_MC)
        self.add_output('volume_mask', color=_MC)
        self._add_int_spinbox('radius', 'Radius (voxels)', value=2,
                              min_val=1, max_val=50)
        self.add_combo_menu('kernel', 'Kernel', items=['ball', 'cube', 'octahedron'])
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.morphology import dilation, ball, cube, octahedron

        data = _get_volume_mask(self)
        if data is None:
            return False, "No volume mask connected"

        mask = np.asarray(data.payload, dtype=np.uint8)
        r = int(self.get_property('radius'))
        k = str(self.get_property('kernel'))

        fp = ball(r) if k == 'ball' else (cube(2 * r + 1) if k == 'cube'
                                          else octahedron(r))
        self.set_progress(30)
        result = dilation(mask, footprint=fp) > 0
        self.set_progress(80)

        self.output_values['volume_mask'] = VolumeMaskData(
            payload=result, spacing=data.spacing)
        self.set_display(_mid_slice_preview(result))
        self.set_progress(100)
        return True, None


# ══════════════════════════════════════════════════════════════════════════════
#  Open3DNode / Close3DNode
# ══════════════════════════════════════════════════════════════════════════════

class Open3DNode(BaseImageProcessNode):
    """3D morphological opening (erosion → dilation).  Removes small protrusions.

    Keywords: open, morphology, 3D, smooth, 開運算, 形態學, 體積
    """
    __identifier__ = 'nodes.Volume.Morphology'
    NODE_NAME      = '3D Open'
    PORT_SPEC      = {'inputs': ['volume_mask'], 'outputs': ['volume_mask']}

    def __init__(self):
        super().__init__()
        self.add_input('volume_mask', color=_MC)
        self.add_output('volume_mask', color=_MC)
        self._add_int_spinbox('radius', 'Radius (voxels)', value=2,
                              min_val=1, max_val=50)
        self.add_combo_menu('kernel', 'Kernel', items=['ball', 'cube', 'octahedron'])
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.morphology import opening, ball, cube, octahedron

        data = _get_volume_mask(self)
        if data is None:
            return False, "No volume mask connected"

        mask = np.asarray(data.payload, dtype=np.uint8)
        r = int(self.get_property('radius'))
        k = str(self.get_property('kernel'))

        fp = ball(r) if k == 'ball' else (cube(2 * r + 1) if k == 'cube'
                                          else octahedron(r))
        self.set_progress(30)
        result = opening(mask, footprint=fp) > 0
        self.set_progress(80)

        self.output_values['volume_mask'] = VolumeMaskData(
            payload=result, spacing=data.spacing)
        self.set_display(_mid_slice_preview(result))
        self.set_progress(100)
        return True, None


class Close3DNode(BaseImageProcessNode):
    """3D morphological closing (dilation → erosion).  Fills small gaps.

    Keywords: close, morphology, 3D, fill, 閉運算, 形態學, 體積
    """
    __identifier__ = 'nodes.Volume.Morphology'
    NODE_NAME      = '3D Close'
    PORT_SPEC      = {'inputs': ['volume_mask'], 'outputs': ['volume_mask']}

    def __init__(self):
        super().__init__()
        self.add_input('volume_mask', color=_MC)
        self.add_output('volume_mask', color=_MC)
        self._add_int_spinbox('radius', 'Radius (voxels)', value=2,
                              min_val=1, max_val=50)
        self.add_combo_menu('kernel', 'Kernel', items=['ball', 'cube', 'octahedron'])
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.morphology import closing, ball, cube, octahedron

        data = _get_volume_mask(self)
        if data is None:
            return False, "No volume mask connected"

        mask = np.asarray(data.payload, dtype=np.uint8)
        r = int(self.get_property('radius'))
        k = str(self.get_property('kernel'))

        fp = ball(r) if k == 'ball' else (cube(2 * r + 1) if k == 'cube'
                                          else octahedron(r))
        self.set_progress(30)
        result = closing(mask, footprint=fp) > 0
        self.set_progress(80)

        self.output_values['volume_mask'] = VolumeMaskData(
            payload=result, spacing=data.spacing)
        self.set_display(_mid_slice_preview(result))
        self.set_progress(100)
        return True, None
