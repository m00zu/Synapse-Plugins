"""
nodes/mask_nodes.py
===================
Mask-domain utility nodes: morphological operations and region measurement.

Nodes:
  ErosionNode     — binary erosion    (mask → mask)
  DilationNode    — binary dilation   (mask → mask)
  MorphOpenNode   — binary opening    (mask → mask)  erode then dilate
  MorphCloseNode  — binary closing    (mask → mask)  dilate then erode
  FillHolesNode   — fill ALL enclosed holes (mask → mask)
                    (contrast: RemoveSmallHolesNode only fills holes ≤ N px²)
  SkeletonizeNode — thin to 1-px centerline (mask → mask)
  RegionPropsNode — label components + measure (mask + opt. image → table + label_image)

The label_image output of RegionPropsNode paints each region a distinct color
and draws its label number at the centroid, so the user can match every table
row (via the `label` column) to a region in the visualization.
"""
from __future__ import annotations

import json as _json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from scipy.ndimage import (
    binary_erosion  as _nd_erosion,
    binary_dilation as _nd_dilation,
    binary_opening  as _nd_opening,
    binary_closing  as _nd_closing,
    binary_fill_holes,
)

import threading

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QPainter, QPixmap, QImage
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene,
                                QGraphicsPixmapItem)
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget

from data_models import ImageData, MaskData, SkeletonData, TableData, LabelData
from nodes.base import PORT_COLORS
from nodes.base import BaseImageProcessNode, _arr_to_pil


# ---------------------------------------------------------------------------
# Cycling color palette for region labels (bright, distinct, dark-background)
# ---------------------------------------------------------------------------

_LABEL_COLORS: list[tuple[int, int, int]] = [
    (255,  85,  85),  # red
    ( 85, 170, 255),  # blue
    ( 85, 255, 170),  # mint
    (255, 200,  85),  # amber
    (200,  85, 255),  # violet
    (255, 170,  85),  # orange
    ( 85, 255, 255),  # cyan
    (255,  85, 200),  # pink
    (170, 255,  85),  # lime
    (255, 255,  85),  # yellow
    (140, 100, 200),  # lavender
    (100, 200, 140),  # sage
    (200, 140, 100),  # tan
    ( 85, 100, 255),  # indigo
    (255, 100,  85),  # coral
    (100, 255, 100),  # light green
    (255,  85, 130),  # rose
    ( 85, 200, 200),  # teal
    (200, 200,  85),  # gold
    (150, 150, 255),  # periwinkle
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mask_to_bool(mask_data: MaskData) -> np.ndarray:
    arr = mask_data.payload
    if arr.ndim == 3:
        arr = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(arr.dtype)
    return arr > (127 if arr.dtype == np.uint8 else arr.max() / 2)


def _bool_to_mask(arr: np.ndarray) -> MaskData:
    u8 = arr.astype(np.uint8) * 255
    return MaskData(payload=u8)


def _get_mask_input(node, port_name: str = 'mask'):
    """Fetch MaskData from a connected input port; return (data, error_str)."""
    port = node.inputs().get(port_name)
    if not port or not port.connected_ports():
        return None, "No input connected"
    cp   = port.connected_ports()[0]
    data = cp.node().output_values.get(cp.name())
    if not isinstance(data, MaskData):
        return None, "Input must be MaskData"
    return data, None


# ===========================================================================
# Morphological operation nodes
# ===========================================================================

class ErosionNode(BaseImageProcessNode):
    """
    Applies binary erosion to shrink foreground regions of a mask.

    Each iteration removes one layer of boundary pixels. Useful for separating touching objects or removing thin protrusions.

    **iterations** — number of erosion passes (default: 1).

    Keywords: shrink, erode, erosion, binary, morphology, 侵蝕, 形態學, 二值化, 收縮, 遮罩
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME      = 'Erode Mask'
    PORT_SPEC      = {'inputs': ['mask'], 'outputs': ['mask']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('mask',  color=PORT_COLORS['mask'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        self._add_int_spinbox('iterations', 'Iterations', value=1, min_val=1, max_val=100)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        data, err = _get_mask_input(self)
        if err:
            return False, err

        iters  = int(self.get_property('iterations'))
        binary = _mask_to_bool(data)
        self.set_progress(40)
        result = _nd_erosion(binary, iterations=iters)
        self.set_progress(90)
        out = _bool_to_mask(result)
        self.output_values['mask'] = out
        self.set_display(out.payload)
        self.set_progress(100)
        return True, None


class DilationNode(BaseImageProcessNode):
    """
    Applies binary dilation to expand foreground regions of a mask.

    Each iteration adds one layer of boundary pixels. Useful for bridging small gaps or growing regions uniformly.

    **iterations** — number of dilation passes (default: 1).

    Keywords: grow, expand, dilate, dilation, binary, 膨脹, 擴張, 形態學, 二值化, 遮罩
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME      = 'Dilate Mask'
    PORT_SPEC      = {'inputs': ['mask'], 'outputs': ['mask']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('mask',  color=PORT_COLORS['mask'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        self._add_int_spinbox('iterations', 'Iterations', value=1, min_val=1, max_val=100)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        data, err = _get_mask_input(self)
        if err:
            return False, err

        iters  = int(self.get_property('iterations'))
        binary = _mask_to_bool(data)
        self.set_progress(40)
        result = _nd_dilation(binary, iterations=iters)
        self.set_progress(90)
        out = _bool_to_mask(result)
        self.output_values['mask'] = out
        self.set_display(out.payload)
        self.set_progress(100)
        return True, None


class MorphOpenNode(BaseImageProcessNode):
    """
    Applies binary opening (erosion then dilation) to a mask.

    Removes small foreground objects and thin protrusions while preserving the overall shape of larger regions.

    **iterations** — number of opening passes (default: 1).

    Keywords: open, opening, morphology, denoise, clean, 開運算, 形態學, 去噪, 侵蝕, 膨脹
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME      = 'Morph Open'
    PORT_SPEC      = {'inputs': ['mask'], 'outputs': ['mask']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('mask',  color=PORT_COLORS['mask'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        self._add_int_spinbox('iterations', 'Iterations', value=1, min_val=1, max_val=50)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        data, err = _get_mask_input(self)
        if err:
            return False, err

        iters  = int(self.get_property('iterations'))
        binary = _mask_to_bool(data)
        self.set_progress(40)
        result = _nd_opening(binary, iterations=iters)
        self.set_progress(90)
        out = _bool_to_mask(result)
        self.output_values['mask'] = out
        self.set_display(out.payload)
        self.set_progress(100)
        return True, None


class MorphCloseNode(BaseImageProcessNode):
    """
    Applies binary closing (dilation then erosion) to a mask.

    Fills small holes and connects nearby fragments without significantly changing the overall region size.

    **iterations** — number of closing passes (default: 1).

    Keywords: close, closing, morphology, fill gaps, connect, 閉運算, 形態學, 填孔, 膨脹, 侵蝕
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME      = 'Morph Close'
    PORT_SPEC      = {'inputs': ['mask'], 'outputs': ['mask']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('mask',  color=PORT_COLORS['mask'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        self._add_int_spinbox('iterations', 'Iterations', value=1, min_val=1, max_val=50)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        data, err = _get_mask_input(self)
        if err:
            return False, err

        iters  = int(self.get_property('iterations'))
        binary = _mask_to_bool(data)
        self.set_progress(40)
        result = _nd_closing(binary, iterations=iters)
        self.set_progress(90)
        out = _bool_to_mask(result)
        self.output_values['mask'] = out
        self.set_display(out.payload)
        self.set_progress(100)
        return True, None


class FillHolesNode(BaseImageProcessNode):
    """
    Fills all enclosed background holes in a binary mask regardless of size.

    Uses `scipy.ndimage.binary_fill_holes`. Contrast with RemoveSmallHolesNode, which only fills holes smaller than a user-defined area threshold and leaves larger holes intact.

    Keywords: holes, fill, interior, enclosed, background, 填孔, 內部, 遮罩, 二值化, 閉合
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME      = 'Fill Holes'
    PORT_SPEC      = {'inputs': ['mask'], 'outputs': ['mask']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('mask',  color=PORT_COLORS['mask'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        data, err = _get_mask_input(self)
        if err:
            return False, err

        binary = _mask_to_bool(data)
        self.set_progress(50)
        result = binary_fill_holes(binary)
        self.set_progress(90)
        out = _bool_to_mask(result)
        self.output_values['mask'] = out
        self.set_display(out.payload)
        self.set_progress(100)
        return True, None


class SkeletonizeNode(BaseImageProcessNode):
    """
    Reduces foreground regions to 1-pixel-wide centrelines (skeleton).

    Uses `skimage.morphology.skeletonize`. Outputs SkeletonData, which can feed SkeletonAnalysisNode or any node that accepts a mask.

    **Method** — *zhang* (default) or *lee*. Lee tends to produce cleaner skeletons on thick blob-like shapes.

    **Prune Spurs** — iteratively removes endpoint pixels (dead-end tips). Setting N removes all branches shorter than N pixels (0 = off).

    Keywords: skeleton, centerline, thinning, medial axis, filopodia, 骨架化, 中軸線, 細化, 形態學, 遮罩
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME      = 'Skeletonize'
    PORT_SPEC      = {'inputs': ['mask'], 'outputs': ['skeleton']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('mask',     color=PORT_COLORS['mask'])
        self.add_output('skeleton', multi_output=True, color=PORT_COLORS['skeleton'])

        self.add_combo_menu('method', 'Method', items=['zhang', 'lee'], tab='Parameters')
        self._add_int_spinbox('prune_iters', 'Prune Spurs (iterations)', value=0, min_val=0, max_val=500, tab='Parameters')

        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.morphology import skeletonize
        from scipy.ndimage import convolve

        data, err = _get_mask_input(self)
        if err:
            return False, err

        method      = self.get_property('method') or 'zhang'
        prune_iters = max(0, int(self.get_property('prune_iters')))

        binary = _mask_to_bool(data)
        self.set_progress(20)
        skel = skeletonize(binary, method=method)
        self.set_progress(60)

        # Spur pruning: iteratively remove endpoint pixels (1 neighbour)
        if prune_iters > 0:
            all_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
            for _ in range(prune_iters):
                nb = convolve(skel.astype(np.uint8), all_kernel, mode='constant')
                endpoints = skel & (nb == 1)
                if not endpoints.any():
                    break
                skel = skel & ~endpoints
        self.set_progress(85)

        skel_u8 = skel.astype(np.uint8) * 255
        out = SkeletonData(payload=skel_u8)
        self.output_values['skeleton'] = out
        self.set_display(skel_u8)
        self.set_progress(100)
        return True, None


def _get_skeleton_input(node):
    """Helper to extract a boolean numpy array from a skeleton input port."""
    port = node.inputs().get('skeleton')
    if not port or not port.connected_ports():
        node.mark_error()
        return None, "No skeleton input connected"
    cp = port.connected_ports()[0]
    val = cp.node().output_values.get(cp.name())
    if not isinstance(val, MaskData):  # SkeletonData is a MaskData subclass
        node.mark_error()
        return None, "Input must be SkeletonData (connect from Skeletonize node)"
    return val, None


class SkeletonAnalysisNode(BaseImageProcessNode):
    """
    Analyses a skeleton produced by SkeletonizeNode.

    Outputs:
    - **skeleton** — filtered SkeletonData (short isolated segments removed)
    - **stats** — total `skeleton_length_px` and `junction_count` (after filtering)
    - **junction_image** — RGB visualisation: skeleton in grey, junction pixels in red on black
    - **junction_mask** — binary MaskData of junction pixels only (pass to DrawShapeNode)

    **Min Segment Length** — remove isolated skeleton segments shorter than this many pixels (0 = keep all).

    **Junction Radius** — dilate each junction point into a filled circle of this radius in pixels for the junction_mask output (default: 4).

    Keywords: skeleton, analysis, junction, length, branch, filter, 骨架, 分析, 節點, 長度
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME      = 'Skeleton Analysis'
    PORT_SPEC      = {'inputs': ['skeleton'], 'outputs': ['skeleton', 'table', 'image', 'mask', 'label_image']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('skeleton',        color=PORT_COLORS['skeleton'])
        self.add_output('skeleton',       multi_output=True, color=PORT_COLORS['skeleton'])
        self.add_output('stats',          multi_output=True, color=PORT_COLORS['table'])
        self.add_output('junction_image', multi_output=True, color=PORT_COLORS['image'])
        self.add_output('junction_mask',  multi_output=True, color=PORT_COLORS['mask'])
        self.add_output('label_image',    multi_output=True, color=PORT_COLORS['label_image'])

        self._add_int_spinbox('min_segment_px',  'Min Segment Length (px)', value=0, min_val=0, max_val=100000, tab='Parameters')
        self._add_int_spinbox('skeleton_width',  'Skeleton Width (px)',     value=1, min_val=0, max_val=20,     tab='Parameters')
        self._add_int_spinbox('junction_radius', 'Junction Radius (px)',    value=4, min_val=0, max_val=50,     tab='Parameters')

        self.create_preview_widgets()

    @staticmethod
    def _numpy_stats(skel: np.ndarray, labeled: np.ndarray, num: int):
        """Vectorised numpy fallback: (length, cy, cx, junction_count) per component."""
        from scipy.ndimage import convolve, center_of_mass
        skel_u8 = skel.astype(np.uint8)
        all_kernel = np.ones((3, 3), dtype=np.uint8); all_kernel[1, 1] = 0
        nb_count = convolve(skel_u8, all_kernel, mode='constant') * skel_u8
        # Vectorised edge bincount — no Python loop over components
        oh = skel[:, :-1] & skel[:, 1:]
        ov = skel[:-1, :] & skel[1:, :]
        dd = skel[:-1, :-1] & skel[1:, 1:]
        dl = skel[:-1, 1:]  & skel[1:, :-1]
        ortho_lbl = np.concatenate([labeled[:, :-1][oh], labeled[:-1, :][ov]])
        diag_lbl  = np.concatenate([labeled[:-1, :-1][dd], labeled[:-1, 1:][dl]])
        ortho_c = np.bincount(ortho_lbl, minlength=num + 1)
        diag_c  = np.bincount(diag_lbl,  minlength=num + 1)
        lengths = ortho_c[1:] + diag_c[1:] * np.sqrt(2)
        lengths[lengths == 0] = 1.0
        junc_c = np.bincount(labeled[nb_count >= 3], minlength=num + 1)[1:]
        centroids = center_of_mass(skel, labels=labeled, index=np.arange(1, num + 1))
        return [(float(lengths[i]), float(centroids[i][0]), float(centroids[i][1]), int(junc_c[i]))
                for i in range(num)], nb_count

    def evaluate(self):
        self.reset_progress()
        from scipy.ndimage import label as nd_label, binary_dilation, convolve
        from skimage.morphology import disk

        data, err = _get_skeleton_input(self)
        if err:
            return False, err

        skel = _mask_to_bool(data)
        min_seg = max(0, int(self.get_property('min_segment_px')))
        junc_r  = max(0, int(self.get_property('junction_radius')))
        self.set_progress(20)

        _8conn = np.ones((3, 3), dtype=np.int32)

        # Remove short isolated segments using vectorised bincount
        if min_seg > 0:
            labeled_pre, num_pre = nd_label(skel, structure=_8conn)
            sizes = np.bincount(labeled_pre.ravel())
            mask_keep = np.zeros(num_pre + 1, dtype=bool)
            mask_keep[1:] = sizes[1:] >= min_seg
            skel = mask_keep[labeled_pre]
        self.set_progress(35)

        labeled, num = nd_label(skel, structure=_8conn)

        # Try Rust backend first
        results = None
        nb_count = None
        try:
            import image_process_rs as _rs
            fn = getattr(_rs, 'skeleton_stats_full', None)
            if fn is not None:
                rs_rows = fn(np.ascontiguousarray(skel.astype(np.uint8)))
                results = [(float(l), float(cy), float(cx), int(j)) for l, cy, cx, j in rs_rows]
        except Exception:
            results = None

        if results is None:
            results, nb_count = self._numpy_stats(skel, labeled, num)

        # Build nb_count for junction mask if Rust was used
        if nb_count is None:
            skel_u8 = skel.astype(np.uint8)
            all_kernel = np.ones((3, 3), dtype=np.uint8); all_kernel[1, 1] = 0
            nb_count = convolve(skel_u8, all_kernel, mode='constant') * skel_u8
        self.set_progress(60)

        rows = [
            {
                'label':          i + 1,
                'centroid_y':     round(r[1], 1),
                'centroid_x':     round(r[2], 1),
                'length_px':      round(r[0], 2),
                'junction_count': r[3],
            }
            for i, r in enumerate(results)
        ]

        junction_mask = skel & (nb_count >= 3)
        if junc_r > 0:
            junction_mask_out = binary_dilation(junction_mask, structure=disk(junc_r))
        else:
            junction_mask_out = junction_mask
        self.set_progress(75)

        # Colored label image via palette lookup (no Python loop)
        _PAL = np.array([
            [0,   0,   0  ],
            [255, 100, 100], [100, 200, 100], [100, 140, 255],
            [255, 210, 60 ], [200, 100, 255], [60,  210, 210],
            [255, 140, 50 ], [160, 255, 100], [255, 100, 180],
        ], dtype=np.uint8)
        if num > 0:
            pal = np.tile(_PAL[1:], ((num // (len(_PAL) - 1) + 1), 1))[:num]
            pal = np.vstack([_PAL[:1], pal])  # index 0 = black background
        else:
            pal = _PAL
        label_rgb = pal[labeled % len(pal)]

        skel_u8 = skel.astype(np.uint8)
        # Junction visualisation: black bg, skeleton=grey, junctions=red (synthetic, 8-bit)
        junc_rgb = np.zeros((*skel.shape, 3), dtype=np.float32)
        junc_rgb[skel]          = (200 / 255.0, 200 / 255.0, 200 / 255.0)
        junc_rgb[junction_mask] = (1.0, 80 / 255.0, 80 / 255.0)

        skel_arr   = skel_u8 * 255
        jmask_arr  = junction_mask_out.astype(np.uint8) * 255

        self.output_values['skeleton']       = SkeletonData(payload=skel_arr)
        self.output_values['stats']          = TableData(payload=pd.DataFrame(rows))
        self.output_values['junction_image'] = ImageData(payload=junc_rgb, bit_depth=8)
        self.output_values['junction_mask']  = MaskData(payload=jmask_arr)
        self.output_values['label_image']    = LabelData(payload=labeled.astype(np.int32), image=label_rgb)
        self.set_display(label_rgb)
        self.set_progress(100)
        return True, None


class MedialAxisNode(BaseImageProcessNode):
    """
    Computes the medial axis (distance-based skeleton) of a binary mask.

    Uses `skimage.morphology.medial_axis`, which finds the set of pixels equidistant from the nearest background pixel. Unlike skeletonize, it also produces a distance transform that encodes the local radius at each skeleton point.

    Outputs:
    - **skeleton** — the 1-pixel-wide medial-axis skeleton (SkeletonData)
    - **distance** — skeleton coloured by local radius using the plasma colourmap (purple = thin, yellow = thick); background is black

    Keywords: medial axis, skeleton, distance transform, thinning, centerline, 中軸線, 骨架化, 距離轉換, 細化, 形態學
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME      = 'Medial Axis'
    PORT_SPEC      = {'inputs': ['mask'], 'outputs': ['skeleton', 'image']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('mask',      color=PORT_COLORS['mask'])
        self.add_output('skeleton', multi_output=True, color=PORT_COLORS['skeleton'])
        self.add_output('distance', multi_output=True, color=PORT_COLORS['image'])
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        import numpy as np
        from skimage.morphology import medial_axis

        data, err = _get_mask_input(self)
        if err:
            return False, err

        binary = _mask_to_bool(data)
        self.set_progress(30)
        skel, distance = medial_axis(binary, return_distance=True)
        self.set_progress(80)

        # Skeleton output as SkeletonData
        skel_arr = skel.astype(np.uint8) * 255
        out_skel = SkeletonData(payload=skel_arr)
        self.output_values['skeleton'] = out_skel
        self.set_display(skel_arr)

        # Skeleton colored by local radius using plasma colormap, background black
        import matplotlib.cm as cm
        skel_dist = (distance * skel).astype(float)
        d_max = skel_dist.max()
        if d_max > 0:
            normed = skel_dist / d_max          # 0.0–1.0 on skeleton, 0 on background
            rgba = cm.plasma(normed)            # H×W×4 float [0,1]
            rgb  = rgba[..., :3].astype(np.float32)
            rgb[~skel] = 0                      # ensure background is pure black
        else:
            rgb = np.zeros((*skel.shape, 3), dtype=np.float32)
        self.output_values['distance'] = ImageData(payload=rgb, bit_depth=8)

        self.set_progress(100)
        return True, None


# ===========================================================================
# RegionPropsNode
# ===========================================================================

class ParticlePropsNode(BaseImageProcessNode):
    """
    Labels connected components in a mask and measures each region.

    Shape columns (always present):
    - `label` — integer region ID (matches label_image pixel values)
    - `area` — number of pixels in the region
    - `equivalent_diameter` — diameter of a circle with the same area
    - `centroid_y` / `centroid_x` — pixel coordinates of the region centre
    - `bbox_top` / `bbox_left` / `bbox_bottom` / `bbox_right` — tight bounding box corners
    - `perimeter` — outer boundary length in pixels
    - `circularity` — `4*pi*area/perimeter^2`; 1.0 = perfect circle, lower = more irregular
    - `eccentricity` — 0 = circle, 1 = line; measures elongation
    - `orientation` — angle of major axis in degrees
    - `major_axis` / `minor_axis` — lengths of the fitted ellipse axes
    - `solidity` — area / convex_hull_area; 1 = perfectly convex
    - `extent` — area / bounding_box_area; fraction of bbox filled
    - `euler_number` — 1 = no holes; decreases by 1 for each enclosed hole

    Intensity columns (present only when intensity_image is connected):
    - `mean_intensity` — average pixel value inside the region
    - `sum_intensity` — total pixel intensity (mean x area)
    - `max_intensity` / `min_intensity` — brightest and darkest pixels
    - `weighted_centroid_y` / `weighted_centroid_x` — intensity-weighted centre

    Outputs:
    - **table** — TableData with all columns above
    - **label_image** — LabelData with integer label array and coloured RGB visualisation

    Keywords: regionprops, measure, area, perimeter, eccentricity, 粒子, 區域, 量測, 面積, 遮罩
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME      = 'Particle Props'
    PORT_SPEC      = {'inputs': ['mask', 'label_image', 'image'], 'outputs': ['table', 'label_image']}
    _collection_aware = True
    OUTPUT_COLUMNS = {
        'table': [
            'label', 'area', 'equivalent_diameter',
            'centroid_y', 'centroid_x',
            'bbox_top', 'bbox_left', 'bbox_bottom', 'bbox_right',
            'perimeter', 'circularity', 'eccentricity', 'orientation',
            'major_axis', 'minor_axis',
            'solidity', 'extent', 'euler_number',
            # intensity columns present when intensity_image port is connected:
            # 'mean_intensity', 'sum_intensity', 'max_intensity', 'min_intensity',
            # 'weighted_centroid_y', 'weighted_centroid_x'
        ]
    }

    def __init__(self):
        super().__init__()
        self.add_input('mask',            color=PORT_COLORS['mask'])
        self.add_input('label_image',     color=PORT_COLORS['label'])
        self.add_input('intensity_image', color=PORT_COLORS['image'])
        self.add_output('table',          color=PORT_COLORS['table'])
        self.add_output('label_image',    color=PORT_COLORS['label'])
        self._add_int_spinbox('min_area', 'Min Region Area (px²)',
                              value=0, min_val=0, max_val=999_999)
        self.create_preview_widgets()

    # ------------------------------------------------------------------
    def evaluate(self):
        self.reset_progress()
        from skimage.measure import label as sk_label, regionprops_table

        # --- label_image (pre-labeled) or mask (binary → sk_label) ---
        labeled = None
        lbl_port = self.inputs().get('label_image')
        if lbl_port and lbl_port.connected_ports():
            cp = lbl_port.connected_ports()[0]
            lbl_data = cp.node().output_values.get(cp.name())
            if isinstance(lbl_data, LabelData) and lbl_data.payload is not None:
                labeled = np.asarray(lbl_data.payload, dtype=np.int32)

        if labeled is None:
            data, err = _get_mask_input(self)
            if err:
                return False, err
            binary = _mask_to_bool(data)
            labeled = sk_label(binary)
        self.set_progress(15)

        # --- optional intensity image ---
        intensity_arr = None
        img_port = self.inputs().get('intensity_image')
        if img_port and img_port.connected_ports():
            cp2   = img_port.connected_ports()[0]
            idata = cp2.node().output_values.get(cp2.name())
            if isinstance(idata, ImageData):
                _iarr = idata.payload
                if _iarr.ndim == 3:
                    _iarr = np.dot(_iarr[..., :3], [0.299, 0.587, 0.114])
                intensity_arr = _iarr.astype(np.float64)
        self.set_progress(30)

        # --- measure ---
        props = [
            'label', 'area', 'centroid', 'bbox',
            'perimeter', 'eccentricity',
            'axis_major_length', 'axis_minor_length',
            'orientation', 'solidity', 'extent', 'euler_number',
            'equivalent_diameter_area',
        ]
        if intensity_arr is not None:
            props += ['mean_intensity', 'max_intensity', 'min_intensity', 'weighted_centroid']

        table = regionprops_table(
            labeled,
            intensity_image=intensity_arr,
            properties=props,
        )
        df = pd.DataFrame(table)

        # Friendly column names
        df.rename(columns={
            'centroid-0':              'centroid_y',
            'centroid-1':              'centroid_x',
            'bbox-0':                  'bbox_top',
            'bbox-1':                  'bbox_left',
            'bbox-2':                  'bbox_bottom',
            'bbox-3':                  'bbox_right',
            'axis_major_length':       'major_axis',
            'axis_minor_length':       'minor_axis',
            'equivalent_diameter_area': 'equivalent_diameter',
            'weighted_centroid-0':     'weighted_centroid_y',
            'weighted_centroid-1':     'weighted_centroid_x',
        }, inplace=True)

        # Convert orientation from radians to degrees (more intuitive)
        if 'orientation' in df.columns:
            df['orientation'] = np.degrees(df['orientation']).round(2)

        # Derived column: circularity = 4π·area / perimeter²  (1.0 = perfect circle)
        if 'area' in df.columns and 'perimeter' in df.columns:
            p = df['perimeter'].replace(0, np.nan)
            df['circularity'] = (4 * np.pi * df['area'] / (p ** 2)).round(4).clip(upper=1.0)

        # Derived intensity column: sum = mean × area
        if intensity_arr is not None and 'mean_intensity' in df.columns:
            df['sum_intensity'] = (df['mean_intensity'] * df['area']).round(2)

        # Reorder columns for readability
        _base_cols = [
            'label', 'area', 'equivalent_diameter',
            'centroid_y', 'centroid_x',
            'bbox_top', 'bbox_left', 'bbox_bottom', 'bbox_right',
            'perimeter', 'circularity', 'eccentricity', 'orientation',
            'major_axis', 'minor_axis',
            'solidity', 'extent', 'euler_number',
        ]
        _intensity_cols = [
            'mean_intensity', 'sum_intensity',
            'max_intensity', 'min_intensity',
            'weighted_centroid_y', 'weighted_centroid_x',
        ]
        ordered = [c for c in _base_cols if c in df.columns]
        if intensity_arr is not None:
            ordered += [c for c in _intensity_cols if c in df.columns]
        df = df[ordered]

        # Filter by minimum area
        min_area = int(self.get_property('min_area'))
        if min_area > 0:
            df = df[df['area'] >= min_area].reset_index(drop=True)

        self.set_progress(70)

        # --- visualization ---
        vis = self._make_label_image(labeled, df)
        self.set_progress(90)

        self.output_values['table']       = TableData(payload=df)
        self.output_values['label_image'] = LabelData(payload=labeled.astype(np.int32),
                                                       image=vis)
        self.set_display(vis)
        self.set_progress(100)
        return True, None

    # ------------------------------------------------------------------
    def _make_label_image(
        self,
        labeled:  np.ndarray,
        df:       pd.DataFrame,
    ) -> np.ndarray:
        """
        Paint each region a unique cycling color and overlay its label number
        at the centroid (white text with black shadow for contrast).
        Returns an RGB numpy uint8 array.
        """
        H, W = labeled.shape
        rgb  = np.zeros((H, W, 3), dtype=np.uint8)

        # Build a fast label -> color LUT (vectorised, no per-label loop)
        n_colors = len(_LABEL_COLORS)
        max_lbl  = int(labeled.max()) if labeled.size else 0
        if max_lbl > 0:
            lut = np.zeros((max_lbl + 1, 3), dtype=np.uint8)
            labels_in_df: set[int] = (
                set(df['label'].astype(int).tolist())
                if 'label' in df.columns else set()
            )
            for lbl in labels_in_df:
                if 0 < lbl <= max_lbl:
                    lut[lbl] = _LABEL_COLORS[(lbl - 1) % n_colors]
            rgb = lut[labeled]

        # Draw label number at centroid with black shadow + white text
        # Use PIL temporarily for text rendering, then convert back to numpy
        if 'centroid_x' in df.columns and 'centroid_y' in df.columns:
            img  = _arr_to_pil(rgb, 'RGB')
            draw = ImageDraw.Draw(img)
            for row in df[['label', 'centroid_x', 'centroid_y']].itertuples(index=False):
                cx = int(round(float(row.centroid_x)))
                cy = int(round(float(row.centroid_y)))
                text = str(int(row.label))
                draw.text((cx + 1, cy + 1), text, fill=(0,   0,   0))   # shadow
                draw.text((cx,     cy    ), text, fill=(255, 255, 255))  # label
            rgb = np.array(img)

        return rgb


# Backward-compatibility alias (imports of RegionPropsNode still resolve)
RegionPropsNode = ParticlePropsNode


# ===========================================================================
# ParticleSelectNode  —  interactive per-particle mask filter
# ===========================================================================

class _FilterRow(QtWidgets.QWidget):
    """One filter rule: [column] [op] [value] [×]"""
    changed = QtCore.Signal()
    removed = QtCore.Signal(object)

    _SS = ("QComboBox, QDoubleSpinBox { background: #222; color: #eee;"
           " border: 1px solid #444; padding: 2px; border-radius: 2px; }"
           "QComboBox:focus, QDoubleSpinBox:focus { border: 1px solid #2e7d32; }")

    def __init__(self, columns=None, parent=None):
        super().__init__(parent)
        self.setStyleSheet(self._SS)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self._col_combo = QtWidgets.QComboBox()
        self._col_combo.setMinimumWidth(70)
        if columns:
            self._col_combo.addItems(columns)
        self._col_combo.currentTextChanged.connect(lambda: self.changed.emit())
        layout.addWidget(self._col_combo)

        self._op_combo = QtWidgets.QComboBox()
        self._op_combo.addItems(['\u2265', '\u2264', '>', '<', '=', '\u2260'])
        self._op_combo.setFixedWidth(45)
        self._op_combo.currentTextChanged.connect(lambda: self.changed.emit())
        layout.addWidget(self._op_combo)

        self._val_spin = QtWidgets.QDoubleSpinBox()
        self._val_spin.setRange(-9_999_999, 9_999_999)
        self._val_spin.setDecimals(3)
        self._val_spin.setSingleStep(1.0)
        self._val_spin.setMinimumWidth(60)
        self._val_spin.valueChanged.connect(lambda: self.changed.emit())
        layout.addWidget(self._val_spin)

        rm_btn = QtWidgets.QToolButton()
        rm_btn.setText('X')
        rm_btn.setFixedSize(22, 22)
        rm_btn.setStyleSheet(
            'QToolButton { font-weight:bold; font-size:11px; color:#ff5555; background:#333;'
            ' border:1px solid #555; border-radius:2px; }'
            'QToolButton:hover { background:#555; color:#ff3333; }')
        rm_btn.clicked.connect(lambda: self.removed.emit(self))
        layout.addWidget(rm_btn)

    def set_columns(self, columns):
        prev = self._col_combo.currentText()
        self._col_combo.blockSignals(True)
        self._col_combo.clear()
        self._col_combo.addItems(columns)
        idx = self._col_combo.findText(prev)
        if idx >= 0:
            self._col_combo.setCurrentIndex(idx)
        self._col_combo.blockSignals(False)

    def get_filter(self):
        return {
            'column': self._col_combo.currentText(),
            'op': self._op_combo.currentText(),
            'value': self._val_spin.value(),
        }

    def set_filter(self, f):
        self._col_combo.blockSignals(True)
        self._op_combo.blockSignals(True)
        self._val_spin.blockSignals(True)
        idx = self._col_combo.findText(f.get('column', ''))
        if idx >= 0:
            self._col_combo.setCurrentIndex(idx)
        op_idx = self._op_combo.findText(f.get('op', '\u2265'))
        if op_idx >= 0:
            self._op_combo.setCurrentIndex(op_idx)
        self._val_spin.setValue(f.get('value', 0))
        self._col_combo.blockSignals(False)
        self._op_combo.blockSignals(False)
        self._val_spin.blockSignals(False)


class _ParticleSelectWidget(NodeBaseWidget):
    """Dynamic particle filter widget. Filter rules added/removed by user."""

    def __init__(self, parent=None, name='', label=''):
        super().__init__(parent, name, label)

        container = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(container)
        vlay.setContentsMargins(4, 2, 4, 2)
        vlay.setSpacing(3)

        self._scroll_content = QtWidgets.QWidget()
        self._filter_layout = QtWidgets.QVBoxLayout(self._scroll_content)
        self._filter_layout.setSpacing(2)
        self._filter_layout.setContentsMargins(0, 0, 0, 0)
        self._filter_layout.addStretch()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self._scroll_content)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(100)
        scroll.viewport().setStyleSheet("background: transparent;")
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        vlay.addWidget(scroll)

        self._filter_rows: list[_FilterRow] = []
        self._columns: list[str] = []

        bottom_row = QtWidgets.QHBoxLayout()
        self._add_btn = QtWidgets.QPushButton("+ Filter")
        self._add_btn.setFixedHeight(20)
        self._add_btn.setStyleSheet("font-size:9px;")
        self._add_btn.clicked.connect(self._add_filter)
        bottom_row.addWidget(self._add_btn)
        self._status = QtWidgets.QLabel("No filters")
        self._status.setStyleSheet("color:#aaa; font-size:9px;")
        bottom_row.addStretch()
        bottom_row.addWidget(self._status)
        vlay.addLayout(bottom_row)

        container.setMinimumWidth(240)
        self.set_custom_widget(container)
        self._data: dict[int, dict] = {}
        self._blocking = False

    def _add_filter(self):
        row = _FilterRow(self._columns)
        row.changed.connect(self._on_filter_changed)
        row.removed.connect(self._remove_filter)
        self._filter_layout.insertWidget(self._filter_layout.count() - 1, row)
        self._filter_rows.append(row)
        self._update_status()

    def _remove_filter(self, row):
        if row in self._filter_rows:
            self._filter_rows.remove(row)
        self._filter_layout.removeWidget(row)
        row.deleteLater()
        self._on_filter_changed()

    def _get_filters(self):
        return [r.get_filter() for r in self._filter_rows]

    def passing_labels(self) -> set[int]:
        filters = self._get_filters()
        if not filters:
            return set(self._data.keys())
        import operator
        ops = {'\u2265': operator.ge, '\u2264': operator.le,
               '>': operator.gt, '<': operator.lt,
               '=': operator.eq, '\u2260': operator.ne,
               '>=': operator.ge, '<=': operator.le,
               '==': operator.eq, '!=': operator.ne}
        result = set()
        for lbl, props in self._data.items():
            ok = True
            for f in filters:
                col = f['column']
                if col not in props:
                    continue
                op_fn = ops.get(f['op'], operator.ge)
                if not op_fn(float(props[col]), f['value']):
                    ok = False
                    break
            if ok:
                result.add(lbl)
        return result

    def _update_status(self):
        n_total = len(self._data)
        if not n_total:
            self._status.setText("Run upstream first")
            return
        n_pass = len(self.passing_labels())
        n_filt = len(self._filter_rows)
        self._status.setText(
            f"{n_pass}/{n_total} pass" +
            (f"  ({n_filt} filter{'s' if n_filt != 1 else ''})" if n_filt else ""))

    def _on_filter_changed(self):
        self._update_status()
        self._emit()

    def _emit(self):
        self.value_changed.emit(self.get_name(), self.get_value())

    def populate(self, data: dict[int, dict], columns: list[str] = None):
        self._data = data
        if columns:
            self._columns = columns
            for row in self._filter_rows:
                row.set_columns(columns)
        self._update_status()

    def get_value(self) -> str:
        return _json.dumps({"filters": self._get_filters()})

    def set_value(self, value: str):
        if not value:
            return
        try:
            state = _json.loads(value)
        except (ValueError, TypeError):
            return
        saved_filters = state.get("filters", [])
        for row in list(self._filter_rows):
            self._remove_filter(row)
        self._blocking = True
        for f in saved_filters:
            if f.get('column'):
                self._add_filter()
                self._filter_rows[-1].set_filter(f)
        self._blocking = False
        self._update_status()


class ParticleSelectNode(BaseImageProcessNode):
    """
    Filters particles from a LabelData input using dynamic filter rules.

    Add filter rules (e.g. area >= 100, circularity >= 0.5) to keep only
    particles matching all conditions. Available properties are auto-detected
    from the upstream data.

    Keywords: filter, select, particles, interactive, 粒子, 篩選, 區域, 遮罩, 互動
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME      = 'Particle Select'
    PORT_SPEC      = {'inputs': ['label_image', 'image'],
                      'outputs': ['mask', 'label_image']}

    def __init__(self):
        super().__init__()
        self.add_input('label_image', color=PORT_COLORS['label'])
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        self.add_output('label_image', color=PORT_COLORS['label'])
        self._sel_widget = _ParticleSelectWidget(self.view, name='selection_state', label='')
        self.add_custom_widget(self._sel_widget)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.measure import regionprops_table

        port = self.inputs().get('label_image')
        if not port or not port.connected_ports():
            return False, "No label_image connected"
        cp   = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if not isinstance(data, LabelData):
            return False, "Input must be LabelData (connect from Particle Props or Watershed)"

        labeled = data.payload
        self.set_progress(20)

        prop_names = ['label', 'area', 'perimeter', 'eccentricity',
                      'solidity', 'extent', 'major_axis_length', 'minor_axis_length']
        props = regionprops_table(labeled, properties=prop_names)

        areas = props['area']
        perimeters = props['perimeter']
        safe_perim = np.where(perimeters > 0, perimeters, 1.0)
        circularity = np.where(
            perimeters > 0,
            4 * np.pi * areas / (safe_perim ** 2),
            0.0
        )

        particle_data = {}
        col_names = [c for c in prop_names if c != 'label'] + ['circularity']
        for i, lbl in enumerate(props['label']):
            d = {c: float(props[c][i]) for c in prop_names if c != 'label'}
            d['circularity'] = float(np.clip(circularity[i], 0, 1))
            particle_data[int(lbl)] = d
        self.set_progress(40)

        self._sel_widget.populate(particle_data, columns=col_names)
        keep = self._sel_widget.passing_labels()

        self.set_progress(70)
        out_arr = np.isin(labeled, list(keep))
        out     = _bool_to_mask(out_arr)
        self.output_values['mask'] = out

        filtered_labels = np.where(out_arr, labeled, 0).astype(np.int32)
        keep_sorted = sorted(keep)
        try:
            from .vision_nodes import _label_palette
        except ImportError:
            from vision_nodes import _label_palette
        palette = _label_palette(len(keep_sorted))

        bg_rgb = None
        img_port = self.inputs().get('image')
        if img_port and img_port.connected_ports():
            icp = img_port.connected_ports()[0]
            idata = icp.node().output_values.get(icp.name())
            if isinstance(idata, ImageData):
                bg = idata.payload
                if bg.ndim == 2:
                    bg = np.stack([bg]*3, axis=-1)
                elif bg.ndim == 3 and bg.shape[2] == 4:
                    bg = bg[..., :3]
                # Ensure uint8 for blending
                if bg.dtype != np.uint8:
                    from data_models import array_to_pil
                    bg = np.array(array_to_pil(bg, getattr(idata, 'bit_depth', 8),
                                               getattr(idata, 'display_min', None),
                                               getattr(idata, 'display_max', None)).convert('RGB'))
                if bg.shape[:2] == labeled.shape:
                    bg_rgb = bg

        if bg_rgb is not None:
            rgb = bg_rgb.copy()
            max_lbl = int(filtered_labels.max()) if filtered_labels.size else 0
            if max_lbl > 0 and keep_sorted:
                color_lut = np.zeros((max_lbl + 1, 3), dtype=np.float32)
                for idx, lbl in enumerate(keep_sorted):
                    if 0 < lbl <= max_lbl:
                        color_lut[lbl] = palette[idx % len(palette)]
                sel_mask = filtered_labels > 0
                overlay_colors = color_lut[filtered_labels]
                rgb[sel_mask] = (rgb[sel_mask].astype(np.float32) * 0.55
                                 + overlay_colors[sel_mask] * 0.45).astype(np.uint8)
            desel = (labeled > 0) & ~out_arr
            rgb[desel] = (rgb[desel].astype(np.float32) * 0.25).astype(np.uint8)
        else:
            max_lbl = int(filtered_labels.max()) if filtered_labels.size else 0
            if max_lbl > 0 and keep_sorted:
                lut = np.zeros((max_lbl + 1, 3), dtype=np.uint8)
                for idx, lbl in enumerate(keep_sorted):
                    if 0 < lbl <= max_lbl:
                        lut[lbl] = palette[idx % len(palette)]
                rgb = lut[filtered_labels]
            else:
                rgb = np.zeros((*filtered_labels.shape, 3), dtype=np.uint8)

        self.output_values['label_image'] = LabelData(
            payload=filtered_labels, image=rgb)

        self.set_display(rgb)
        self.set_progress(100)
        return True, None


# ═══════════════════════════════════════════════════════════════════════════════
#  Visual Particle Select — click-to-toggle interactive particle selection
# ═══════════════════════════════════════════════════════════════════════════════

def _mouse_pos_qpoint(event):
    """Qt5/Qt6 compat helper for mouse position."""
    pos_fn = getattr(event, "position", None)
    if callable(pos_fn):
        return pos_fn().toPoint()
    return event.pos()


class _ParticleSelectGraphicsView(QGraphicsView):
    """Displays a label image where clicking a particle toggles its selection state."""

    selection_changed = Signal()

    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._label_arr: np.ndarray | None = None
        self._bg_rgb: np.ndarray | None = None   # optional background image
        self._selected: set[int] = set()
        self._all_labels: set[int] = set()
        self._palette: dict[int, tuple] = {}
        self._pixmap_item: QGraphicsPixmapItem | None = None

        self._scale: float = 1.0
        self._pan_start: QtCore.QPoint | None = None
        self._overlay_alpha: float = 0.45

    # ── data loading ─────────────────────────────────────────────────────

    def load_label_data(self, label_arr: np.ndarray,
                        palette: dict[int, tuple],
                        bg_rgb: np.ndarray | None = None):
        self._label_arr = label_arr
        self._palette = palette
        self._bg_rgb = bg_rgb
        self._all_labels = set(palette.keys())
        # Prune stale selections; if first load select all
        self._selected &= self._all_labels
        if not self._selected:
            self._selected = set(self._all_labels)
        self._update_display()

    # ── rendering ────────────────────────────────────────────────────────

    def _update_display(self):
        if self._label_arr is None:
            return
        h, w = self._label_arr.shape
        max_label = int(self._label_arr.max())

        has_bg = self._bg_rgb is not None
        if has_bg:
            bg = self._bg_rgb
            # Selected: blend 50% label color over image
            # Deselected: dim the image to 30%
            # Background (label 0): show original image
            lut_color = np.zeros((max_label + 1, 3), dtype=np.float32)
            lut_alpha = np.zeros(max_label + 1, dtype=np.float32)
            for lbl, color in self._palette.items():
                if lbl in self._selected:
                    lut_color[lbl] = color
                    lut_alpha[lbl] = self._overlay_alpha
                else:
                    lut_alpha[lbl] = -1.0       # sentinel: dim
            overlay = lut_color[self._label_arr]    # (H,W,3)
            alpha = lut_alpha[self._label_arr]      # (H,W)

            rgb = bg.astype(np.float32).copy()
            # Dim deselected regions
            dim_mask = alpha < 0
            rgb[dim_mask] = rgb[dim_mask] * 0.25
            # Blend selected regions
            sel_mask = alpha > 0
            a = alpha[sel_mask, np.newaxis]
            rgb[sel_mask] = rgb[sel_mask] * (1 - a) + overlay[sel_mask] * a
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            # No background: solid color / dimmed as before
            lut = np.zeros((max_label + 1, 3), dtype=np.uint8)
            for lbl, color in self._palette.items():
                if lbl in self._selected:
                    lut[lbl] = color
                else:
                    lut[lbl] = [int(c * 0.25) for c in color]
            rgb = lut[self._label_arr]

        qimg = QImage(rgb.data.tobytes(), w, h, 3 * w,
                      QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        if self._pixmap_item is None:
            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self._pixmap_item.setZValue(-1)
            self.scene().addItem(self._pixmap_item)
            self.scene().setSceneRect(QRectF(0, 0, w, h))
            self.fitInView(self._pixmap_item,
                           Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self._pixmap_item.setPixmap(pixmap)

    # ── mouse events ─────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        # Middle-button pan
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = _mouse_pos_qpoint(event)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        # Map click to pixel coords
        scene_pos = self.mapToScene(_mouse_pos_qpoint(event))
        x, y = int(scene_pos.x()), int(scene_pos.y())
        if self._label_arr is None:
            return
        h, w = self._label_arr.shape
        if not (0 <= x < w and 0 <= y < h):
            return

        label_id = int(self._label_arr[y, x])
        if label_id == 0:
            return  # background click — ignore

        # Toggle
        if label_id in self._selected:
            self._selected.discard(label_id)
        else:
            self._selected.add(label_id)

        self._update_display()
        self.selection_changed.emit()
        event.accept()

    def mouseMoveEvent(self, event):
        if self._pan_start is not None:
            delta = _mouse_pos_qpoint(event) - self._pan_start
            self._pan_start = _mouse_pos_qpoint(event)
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and \
                self._pan_start is not None:
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)
        self._scale *= factor
        event.accept()


class _InteractiveParticleSelectWidget(NodeBaseWidget):
    """Embeds a visual particle selector with click-to-toggle on the node surface."""

    _img_signal = Signal(object, object, object)  # (label_arr, palette, bg_rgb)

    _VIEW_MAX = 400   # max dimension (width or height)
    _VIEW_MIN = 200   # min dimension
    _CHROME_H = 80    # approx height of toolbar + filter row + status + margins

    def __init__(self, parent=None, name='', label=''):
        super().__init__(parent, name, label)

        container = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(container)
        vlay.setContentsMargins(4, 2, 4, 2)
        vlay.setSpacing(3)

        # ── toolbar row ──────────────────────────────────────────────────
        tb = QtWidgets.QHBoxLayout()
        tb.setSpacing(4)
        self._btn_all = QtWidgets.QPushButton("All")
        self._btn_all.setFixedHeight(22)
        self._btn_none = QtWidgets.QPushButton("None")
        self._btn_none.setFixedHeight(22)
        self._btn_fit = QtWidgets.QPushButton("Fit")
        self._btn_fit.setFixedHeight(22)
        tb.addWidget(self._btn_all)
        tb.addWidget(self._btn_none)
        tb.addWidget(self._btn_fit)
        tb.addSpacing(8)
        lbl_op = QtWidgets.QLabel("Opacity:")
        lbl_op.setStyleSheet("color:#ccc; font-size:9px;")
        tb.addWidget(lbl_op)
        self._opacity_spin = QtWidgets.QSpinBox()
        self._opacity_spin.setRange(0, 100)
        self._opacity_spin.setValue(45)
        self._opacity_spin.setSuffix("%")
        self._opacity_spin.setFixedWidth(58)
        self._opacity_spin.setFixedHeight(22)
        tb.addWidget(self._opacity_spin)
        tb.addStretch()
        vlay.addLayout(tb)

        # ── size filter row ──────────────────────────────────────────────
        sz = QtWidgets.QHBoxLayout()
        sz.setSpacing(4)
        sz.addWidget(QtWidgets.QLabel("Min:"))
        self._min_spin = QtWidgets.QSpinBox()
        self._min_spin.setRange(0, 9_999_999)
        self._min_spin.setSuffix(" px²")
        self._min_spin.setToolTip("Minimum area (0 = no limit)")
        self._min_spin.setFixedWidth(90)
        sz.addWidget(self._min_spin)
        sz.addWidget(QtWidgets.QLabel("Max:"))
        self._max_spin = QtWidgets.QSpinBox()
        self._max_spin.setRange(0, 9_999_999)
        self._max_spin.setSuffix(" px²")
        self._max_spin.setToolTip("Maximum area (0 = no limit)")
        self._max_spin.setFixedWidth(90)
        sz.addWidget(self._max_spin)
        sz.addStretch()
        vlay.addLayout(sz)

        # ── status ───────────────────────────────────────────────────────
        self._status = QtWidgets.QLabel("Run upstream first")
        self._status.setStyleSheet("color:#aaa;font-size:10px;")
        vlay.addWidget(self._status)

        # ── graphics view ────────────────────────────────────────────────
        self._scene = QGraphicsScene()          # prevent GC
        self._view = _ParticleSelectGraphicsView(self._scene)
        self._view.setMinimumSize(self._VIEW_MIN, self._VIEW_MIN)
        self._view.setFixedSize(self._VIEW_MAX, self._VIEW_MAX)
        self._view.setStyleSheet("background:#1a1a1a;")
        vlay.addWidget(self._view)

        self._container = container
        self.set_custom_widget(self._container)

        # ── internal state ───────────────────────────────────────────────
        self._particle_data: dict[int, int] = {}

        # ── connections ──────────────────────────────────────────────────
        self._view.selection_changed.connect(self._on_selection_changed)
        self._btn_all.clicked.connect(self._select_all)
        self._btn_none.clicked.connect(self._clear_all)
        self._btn_fit.clicked.connect(self._fit_view)
        self._min_spin.valueChanged.connect(self._on_filter_changed)
        self._max_spin.valueChanged.connect(self._on_filter_changed)
        self._opacity_spin.valueChanged.connect(self._on_opacity_changed)
        self._img_signal.connect(self._apply_label_data,
                                 Qt.ConnectionType.QueuedConnection)

    # ── thread-safe data loading ─────────────────────────────────────────

    def populate(self, label_arr: np.ndarray,
                 particle_data: dict[int, int],
                 palette: dict[int, tuple],
                 bg_rgb: np.ndarray | None = None):
        self._particle_data = particle_data
        if threading.current_thread() is threading.main_thread():
            self._apply_label_data(label_arr, palette, bg_rgb)
        else:
            self._img_signal.emit(label_arr, palette, bg_rgb)

    def _apply_label_data(self, label_arr, palette, bg_rgb=None):
        # Resize view to match image aspect ratio
        h, w = label_arr.shape[:2]
        if w >= h:
            vw = self._VIEW_MAX
            vh = max(self._VIEW_MIN, int(self._VIEW_MAX * h / w))
        else:
            vh = self._VIEW_MAX
            vw = max(self._VIEW_MIN, int(self._VIEW_MAX * w / h))
        self._view.setFixedSize(vw, vh)
        # Resize the container (the widget the proxy wraps) to match
        self._container.setFixedSize(vw + 8, vh + self._CHROME_H)

        self._view.load_label_data(label_arr, palette, bg_rgb)
        self._update_status()

        # Force proxy widget + node to recalculate geometry
        if self.widget():
            self.widget().adjustSize()
        if self.node and hasattr(self.node, 'view') and \
                hasattr(self.node.view, 'draw_node'):
            self.node.view.draw_node()

    # ── toolbar actions ──────────────────────────────────────────────────

    def _select_all(self):
        self._view._selected = set(self._view._all_labels)
        self._view._update_display()
        self._update_status()
        self._emit()

    def _clear_all(self):
        self._view._selected.clear()
        self._view._update_display()
        self._update_status()
        self._emit()

    def _fit_view(self):
        if self._view._pixmap_item:
            self._view.fitInView(self._view._pixmap_item,
                                 Qt.AspectRatioMode.KeepAspectRatio)
            self._view._scale = 1.0

    def _on_filter_changed(self):
        """Deselect particles outside the size range."""
        mn = self._min_spin.value()
        mx = self._max_spin.value()
        to_remove = set()
        for lbl, area in self._particle_data.items():
            if mn > 0 and area < mn:
                to_remove.add(lbl)
            if mx > 0 and area > mx:
                to_remove.add(lbl)
        self._view._selected -= to_remove
        self._view._update_display()
        self._update_status()
        self._emit()

    def _on_opacity_changed(self, val):
        self._view._overlay_alpha = val / 100.0
        self._view._update_display()

    # ── selection feedback ───────────────────────────────────────────────

    def _on_selection_changed(self):
        self._update_status()
        self._emit()

    def _update_status(self):
        n_sel = len(self._view._selected)
        n_total = len(self._view._all_labels)
        self._status.setText(f"{n_sel} / {n_total} selected")

    def _emit(self):
        self.value_changed.emit(self.get_name(), self.get_value())

    # ── NodeBaseWidget interface ─────────────────────────────────────────

    def get_value(self) -> str:
        return _json.dumps({
            "min_size": self._min_spin.value(),
            "max_size": self._max_spin.value(),
            "selected": sorted(self._view._selected),
        })

    def set_value(self, value: str):
        if not value:
            return
        try:
            state = _json.loads(value)
        except (ValueError, TypeError):
            return
        self._min_spin.blockSignals(True)
        self._max_spin.blockSignals(True)
        self._min_spin.setValue(state.get("min_size", 0))
        self._max_spin.setValue(state.get("max_size", 0))
        self._min_spin.blockSignals(False)
        self._max_spin.blockSignals(False)
        self._view._selected = set(state.get("selected", []))


class InteractiveParticleSelectNode(BaseImageProcessNode):
    """
    Selects particles visually by clicking them directly in the label image.

    Displays the label image with coloured regions. Click a particle to deselect it (dims); click again to re-select. All / None buttons for bulk operations. Min/Max area filters to exclude tiny debris.

    Same inputs and outputs as Particle Select but with visual, spatial interaction instead of a checkbox list.

    Keywords: visual, click, select, particle, interactive, picker, 視覺, 點選, 粒子, 互動
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME      = 'Visual Particle Select'
    PORT_SPEC      = {'inputs': ['label_image', 'image'],
                      'outputs': ['mask', 'label_image']}

    def __init__(self):
        super().__init__()
        self.add_input('label_image', color=PORT_COLORS['label'])
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('mask',        color=PORT_COLORS['mask'])
        self.add_output('label_image', color=PORT_COLORS['label'])
        self._sel_widget = _InteractiveParticleSelectWidget(
            self.view, name='vis_selection_state', label='')
        self.add_custom_widget(self._sel_widget)

    def evaluate(self):
        self.reset_progress()
        from skimage.measure import regionprops_table

        # 1. Get LabelData
        port = self.inputs().get('label_image')
        if not port or not port.connected_ports():
            return False, "No label_image connected"
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if not isinstance(data, LabelData):
            return False, ("Input must be LabelData "
                           "(connect from Particle Props or Watershed)")

        labeled = data.payload
        self.set_progress(20)

        # 2. Per-label areas
        props = regionprops_table(labeled, properties=['label', 'area'])
        particle_data = {
            int(l): int(a)
            for l, a in zip(props['label'], props['area'])
        }
        self.set_progress(30)

        # 3. Build palette
        try:
            from .vision_nodes import _label_palette
        except ImportError:
            from vision_nodes import _label_palette
        all_labels = sorted(particle_data.keys())
        pal = _label_palette(len(all_labels))
        palette = {lbl: pal[i % len(pal)]
                   for i, lbl in enumerate(all_labels)}

        # 4. Optional image input for overlay
        bg_rgb = None
        img_port = self.inputs().get('image')
        if img_port and img_port.connected_ports():
            icp = img_port.connected_ports()[0]
            idata = icp.node().output_values.get(icp.name())
            if isinstance(idata, ImageData):
                bg = idata.payload
                if bg.ndim == 2:
                    bg = np.stack([bg]*3, axis=-1)
                elif bg.ndim == 3 and bg.shape[2] == 4:
                    bg = bg[..., :3]
                if bg.dtype != np.uint8:
                    from data_models import array_to_pil
                    bg = np.array(array_to_pil(bg, getattr(idata, 'bit_depth', 8),
                                               getattr(idata, 'display_min', None),
                                               getattr(idata, 'display_max', None)).convert('RGB'))
                if bg.shape[:2] == labeled.shape:
                    bg_rgb = bg

        # 5. Populate widget (thread-safe, no emission)
        self._sel_widget.populate(labeled, particle_data, palette, bg_rgb)
        self.set_progress(40)

        # 6. Read selection state
        state_str = self.get_property('vis_selection_state')
        try:
            state = _json.loads(state_str) if state_str else {}
        except (ValueError, TypeError):
            state = {}

        mn = state.get('min_size', 0)
        mx = state.get('max_size', 0)
        sel_ids = set(state.get('selected', []))
        all_keep = not sel_ids  # empty → keep all in size range

        keep = set()
        for lbl, area in particle_data.items():
            if mn > 0 and area < mn:
                continue
            if mx > 0 and area > mx:
                continue
            if all_keep or lbl in sel_ids:
                keep.add(lbl)

        self.set_progress(70)

        # 7. Outputs
        out_arr = np.isin(labeled, list(keep))
        out = _bool_to_mask(out_arr)
        self.output_values['mask'] = out

        filtered_labels = np.where(out_arr, labeled, 0).astype(np.int32)
        keep_sorted = sorted(keep)

        if bg_rgb is not None:
            rgb = bg_rgb.copy()
            for idx, lbl in enumerate(keep_sorted):
                mask_lbl = filtered_labels == lbl
                color = np.array(palette.get(lbl, pal[idx % len(pal)]),
                                 dtype=np.float32)
                rgb[mask_lbl] = (rgb[mask_lbl].astype(np.float32) * 0.55
                                 + color * 0.45).astype(np.uint8)
            desel = (labeled > 0) & ~out_arr
            rgb[desel] = (rgb[desel].astype(np.float32) * 0.25).astype(np.uint8)
        else:
            rgb = np.zeros((*filtered_labels.shape, 3), dtype=np.uint8)
            for idx, lbl in enumerate(keep_sorted):
                rgb[filtered_labels == lbl] = palette.get(
                    lbl, pal[idx % len(pal)])

        self.output_values['label_image'] = LabelData(
            payload=filtered_labels, image=rgb)

        self.set_progress(100)
        return True, None
