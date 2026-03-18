"""
nodes/vision_nodes.py
=====================
Advanced image-analysis nodes: segmentation, feature detection, colocalization,
texture analysis, and intensity profiling.

Nodes
-----
WhiteTopHatNode     — morphological white top-hat (removes broad background)
BandpassFilterNode  — FFT bandpass filter (keeps objects between size_min–size_max)
WatershedNode       — marker-watershed to separate touching objects
BlobDetectNode      — Laplacian-of-Gaussian blob/spot detection → table + overlay
FrangiNode          — Frangi multi-scale vesselness (tubular structure enhancement)
ColocalizationNode  — Pearson PCC + Manders M1/M2/MOC + Li ICQ (masked & unmasked)
GLCMTextureNode     — GLCM texture features (contrast, homogeneity, energy, …)
IntensityProfileNode — pixel intensity along a line segment → line plot figure
"""
from __future__ import annotations

import json
import threading
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw  # still used by FindContoursNode for drawing

import NodeGraphQt
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, QRectF, QPointF, QLineF, Signal
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QPixmap, QImage
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsLineItem, QGraphicsEllipseItem,
)
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget

from data_models import ImageData, MaskData, TableData, FigureData, LabelData
from nodes.base import PORT_COLORS, BaseExecutionNode
from nodes.base import BaseImageProcessNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_image_arr(node, port_name: str):
    """Return (arr, arr, cls) from an image/mask input port, or (None, None, None).

    With the numpy-payload pipeline the second element is the same array
    (kept for call-site compatibility).
    """
    port = node.inputs().get(port_name)
    if not port or not port.connected_ports():
        return None, None, None
    cp   = port.connected_ports()[0]
    data = cp.node().output_values.get(cp.name())
    if isinstance(data, ImageData):
        return data.payload, data.payload, ImageData
    if isinstance(data, MaskData):
        return data.payload, data.payload, MaskData
    return None, None, None


def _get_mask_arr(node, port_name: str):
    """Return binary bool array from a mask port, or None."""
    port = node.inputs().get(port_name)
    if not port or not port.connected_ports():
        return None
    cp   = port.connected_ports()[0]
    data = cp.node().output_values.get(cp.name())
    if isinstance(data, MaskData):
        arr = data.payload  # numpy array
        if arr.ndim == 3:
            arr = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(arr.dtype)
        return arr > 0
    return None


def _to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert numpy image array to 2-D grayscale, preserving dtype."""
    if arr.ndim == 2:
        return arr
    return np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(arr.dtype)


def _to_rgb(arr: np.ndarray) -> np.ndarray:
    """Ensure numpy image array is 3-channel RGB uint8."""
    if arr.ndim == 2:
        rgb = np.stack([arr] * 3, axis=-1)
    else:
        rgb = arr[..., :3].copy()
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)
    return rgb


# ===========================================================================
# 1 — WhiteTopHatNode
# ===========================================================================

class WhiteTopHatNode(BaseImageProcessNode):
    """
    Applies a morphological white top-hat filter to extract small bright features.

    Subtracts the morphological opening (background estimate) from the original image,
    leaving only bright structures that fit inside the disk of the given radius. Useful
    for equalising uneven illumination or removing broad bright background before
    thresholding. Works on grayscale and each channel of RGB independently.

    **Disk Radius** — radius of the structuring element in pixels.

    Keywords: white top-hat, morphology, background removal, uneven illumination, bright spots, 頂帽濾波, 去背景, 形態學, 亮點, 均勻化
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME      = 'White Top-Hat'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}

    def __init__(self):
        super().__init__()
        self.add_input('image',  color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'])
        self._add_int_spinbox('radius', 'Disk Radius (px)', value=15, min_val=1, max_val=500)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        arr, pil_in, out_cls = _get_image_arr(self, 'image')
        if arr is None:
            return False, "No input connected"

        radius = int(self.get_property('radius'))
        self.set_progress(20)

        rs_ok = False
        try:
            import image_process_rs as _rs
            arr_f = arr.astype(np.float32)
            if arr_f.ndim == 2:
                result = np.asarray(_rs.white_tophat(np.ascontiguousarray(arr_f), radius))
                rs_ok = True
            elif arr_f.ndim == 3:
                channels = [np.asarray(_rs.white_tophat(np.ascontiguousarray(arr_f[:, :, c]), radius))
                            for c in range(arr.shape[2])]
                result = np.stack(channels, axis=2)
                rs_ok = True
        except Exception:
            pass

        if not rs_ok:
            from skimage.morphology import white_tophat, disk
            selem = disk(radius)
            if arr.ndim == 2:
                result = white_tophat(arr, selem)
            else:
                channels = [white_tophat(arr[:, :, c], selem) for c in range(arr.shape[2])]
                result = np.stack(channels, axis=2)

        self.set_progress(80)
        self.output_values['image'] = out_cls(payload=result)
        self.set_display(result)
        self.set_progress(100)
        return True, None


# ===========================================================================
# 1b — BlackTopHatNode
# ===========================================================================

class BlackTopHatNode(BaseImageProcessNode):
    """
    Applies a morphological black top-hat filter to extract small dark features.

    Subtracts the original image from its morphological closing, revealing small dark
    structures (holes, valleys, cracks) on a bright background. The complement of
    White Top-Hat -- use White Top-Hat for bright features on dark backgrounds. Works
    on grayscale and each channel of RGB independently.

    **Disk Radius** — radius of the structuring element in pixels.

    Keywords: black top-hat, bottom-hat, morphology, dark spots, backgrounds, 頂帽濾波, 去背景, 形態學, 暗點, 均勻化
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME      = 'Black Top-Hat'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}

    def __init__(self):
        super().__init__()
        self.add_input('image',  color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'])
        self._add_int_spinbox('radius', 'Disk Radius (px)', value=15, min_val=1, max_val=500)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        arr, pil_in, out_cls = _get_image_arr(self, 'image')
        if arr is None:
            return False, "No input connected"

        radius = int(self.get_property('radius'))
        self.set_progress(20)

        rs_ok = False
        try:
            import image_process_rs as _rs
            arr_f = arr.astype(np.float32)
            if arr_f.ndim == 2:
                result = np.asarray(_rs.black_tophat(np.ascontiguousarray(arr_f), radius))
                rs_ok = True
            elif arr_f.ndim == 3:
                channels = [np.asarray(_rs.black_tophat(np.ascontiguousarray(arr_f[:, :, c]), radius))
                            for c in range(arr.shape[2])]
                result = np.stack(channels, axis=2)
                rs_ok = True
        except Exception:
            pass

        if not rs_ok:
            from skimage.morphology import black_tophat, disk
            selem = disk(radius)
            if arr.ndim == 2:
                result = black_tophat(arr, selem)
            else:
                channels = [black_tophat(arr[:, :, c], selem) for c in range(arr.shape[2])]
                result = np.stack(channels, axis=2)

        self.set_progress(80)
        self.output_values['image'] = out_cls(payload=result)
        self.set_display(result)
        self.set_progress(100)
        return True, None


# ===========================================================================
# 2 — BandpassFilterNode
# ===========================================================================

class BandpassFilterNode(BaseImageProcessNode):
    """
    Applies an FFT-based bandpass filter to a grayscale image.

    Keeps spatial frequencies corresponding to object sizes between the two cutoffs,
    analogous to ImageJ's Process > FFT > Bandpass Filter.

    **Remove < (px)** — suppress structures smaller than this value (high-pass cutoff).
    **Remove > (px)** — suppress structures larger than this value (low-pass cutoff). Set to `0` to disable low-pass (keep all large features).

    Keywords: bandpass, fft filter, frequency filter, remove noise, remove background, 帶通, 頻率濾波, 去噪, 去背景, 影像處理
    """
    __identifier__ = 'nodes.image_process.filter'
    NODE_NAME      = 'Bandpass Filter'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}

    def __init__(self):
        super().__init__()
        self.add_input('image',  color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'])
        self._add_int_spinbox('small_px', 'Remove < (px)', value=3,   min_val=1,   max_val=9999)
        self._add_int_spinbox('large_px', 'Remove > (px)', value=100, min_val=0,   max_val=9999)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        arr, pil_in, out_cls = _get_image_arr(self, 'image')
        if arr is None:
            return False, "No input connected"

        # Convert to grayscale float for FFT
        gray = _to_gray(arr).astype(np.float64)

        H, W     = gray.shape
        small_px = int(self.get_property('small_px'))
        large_px = int(self.get_property('large_px'))
        self.set_progress(20)

        fft_shift = np.fft.fftshift(np.fft.fft2(gray))

        cy, cx = H // 2, W // 2
        Y, X   = np.ogrid[:H, :W]
        dist   = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        # Spatial frequency ↔ object size: freq = W / size_px
        high_freq = W / float(small_px) if small_px > 0 else np.inf
        low_freq  = W / float(large_px) if large_px > 0 else 0.0

        bandpass  = (dist >= low_freq) & (dist <= high_freq)
        self.set_progress(50)

        filtered  = np.fft.ifft2(np.fft.ifftshift(fft_shift * bandpass)).real
        filtered  = np.clip(filtered, 0.0, 1.0).astype(np.float32)
        self.set_progress(80)

        self.output_values['image'] = out_cls(payload=filtered)
        self.set_display(filtered)
        self.set_progress(100)
        return True, None


# ===========================================================================
# 3 — WatershedNode
# ===========================================================================

class WatershedNode(BaseImageProcessNode):
    """
    Separates touching or overlapping objects using marker-controlled watershed segmentation.

    Pipeline:
    - Compute Euclidean distance transform of the binary mask.
    - Find local maxima (object centres) in the distance map; **Min Object Sep.** controls the minimum allowed gap between two peaks.
    - Run watershed on the inverted distance map to delineate each object.

    Table columns:
    - `label` — integer region ID (matches label_image pixel values)
    - `area` — number of pixels in the region
    - `equivalent_diameter` — diameter of a circle with the same area (`sqrt(4*area/pi)`)
    - `centroid_y`, `centroid_x` — pixel coordinates of the region centre
    - `perimeter` — outer boundary length in pixels
    - `circularity` — `4*pi*area/perimeter^2`; 1.0 = perfect circle, lower = more irregular
    - `eccentricity` — 0 = circle, 1 = line; measures elongation
    - `orientation` — angle of major axis in degrees; 0 = right, +90 = up, -90 = down
    - `major_axis` — length of longest axis of the fitted ellipse
    - `minor_axis` — length of shortest axis of the fitted ellipse
    - `solidity` — area / convex_hull_area; 1 = perfectly convex, <1 = concave
    - `extent` — area / bounding_box_area; fraction of bounding box filled
    - `euler_number` — 1 = no holes; decreases by 1 for each enclosed hole

    Keywords: watershed, split touching objects, segmentation, distance transform, marker-based, 分水嶺, 分割, 距離轉換, 分離, 影像處理
    """
    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME      = 'Watershed'
    PORT_SPEC      = {'inputs': ['mask'], 'outputs': ['label_image', 'table']}
    OUTPUT_COLUMNS = {
        'table': [
            'label', 'area', 'equivalent_diameter',
            'centroid_y', 'centroid_x',
            'perimeter', 'circularity', 'eccentricity', 'orientation',
            'major_axis', 'minor_axis',
            'solidity', 'extent', 'euler_number',
        ]
    }

    def __init__(self):
        super().__init__()
        self.add_input('mask',         color=PORT_COLORS['mask'])
        self.add_output('label_image', color=PORT_COLORS['label'])
        self.add_output('table',       color=PORT_COLORS['table'])
        self._add_int_spinbox('min_distance', 'Min Object Sep. (px)', value=10, min_val=1, max_val=500)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.feature import peak_local_max
        from skimage.measure import regionprops_table

        binary = _get_mask_arr(self, 'mask')
        if binary is None:
            return False, "No mask connected"

        # Try Rust for EDT, label, watershed; fall back to scipy/skimage
        try:
            import image_process_rs as _rs
            _has_rs = True
        except Exception:
            _has_rs = False

        self.set_progress(15)
        if _has_rs:
            try:
                dist = np.asarray(_rs.distance_transform_edt(binary.astype(np.uint8)),
                                  dtype=np.float64)
            except Exception:
                from scipy.ndimage import distance_transform_edt
                dist = distance_transform_edt(binary)
        else:
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(binary)

        min_dist  = int(self.get_property('min_distance'))
        self.set_progress(30)

        coords   = peak_local_max(dist, min_distance=min_dist, labels=binary)
        markers_bool = np.zeros(binary.shape, dtype=bool)
        if len(coords):
            markers_bool[tuple(coords.T)] = True

        if _has_rs:
            try:
                markers_arr, _ = _rs.label_2d(markers_bool.astype(np.uint8), 8)
                markers = np.asarray(markers_arr, dtype=np.int32)
            except Exception:
                from scipy.ndimage import label as nd_label
                markers, _ = nd_label(markers_bool)
        else:
            from scipy.ndimage import label as nd_label
            markers, _ = nd_label(markers_bool)
        self.set_progress(50)

        if _has_rs:
            try:
                neg_dist_f = (1.0 - np.clip(dist / max(dist.max(), 1e-9), 0, 1)).astype(np.float32)
                labeled = np.asarray(
                    _rs.watershed(neg_dist_f, markers.astype(np.uint32),
                                  binary.astype(np.uint8)),
                    dtype=np.int32)
            except Exception:
                from skimage.segmentation import watershed
                labeled = watershed(-dist, markers, mask=binary)
        else:
            from skimage.segmentation import watershed
            labeled = watershed(-dist, markers, mask=binary)
        self.set_progress(65)

        # Build coloured label image (LUT-based, no per-label loop)
        n_labels  = labeled.max()
        palette   = _label_palette(n_labels)
        lut       = np.zeros((n_labels + 1, 3), dtype=np.uint8)
        for lbl in range(1, n_labels + 1):
            lut[lbl] = palette[(lbl - 1) % len(palette)]
        rgb = lut[labeled]

        # Region-props table
        rp = regionprops_table(
            labeled,
            properties=[
                'label', 'area', 'centroid',
                'axis_major_length', 'axis_minor_length', 'eccentricity',
                'equivalent_diameter_area', 'perimeter', 'solidity',
                'orientation', 'extent', 'euler_number',
            ],
        )
        df = pd.DataFrame(rp)
        df.rename(columns={
            'centroid-0':              'centroid_y',
            'centroid-1':              'centroid_x',
            'axis_major_length':       'major_axis',
            'axis_minor_length':       'minor_axis',
            'equivalent_diameter_area': 'equivalent_diameter',
        }, inplace=True)
        # Convert orientation from radians to degrees
        if 'orientation' in df.columns:
            df['orientation'] = np.degrees(df['orientation']).round(2)
        # Derived column: circularity = 4π·area / perimeter²  (1.0 = perfect circle)
        if 'area' in df.columns and 'perimeter' in df.columns:
            p = df['perimeter'].replace(0, np.nan)
            df['circularity'] = (4 * np.pi * df['area'] / (p ** 2)).round(4).clip(upper=1.0)
        # Reorder for readability
        _col_order = [
            'label', 'area', 'equivalent_diameter',
            'centroid_y', 'centroid_x',
            'perimeter', 'circularity', 'eccentricity', 'orientation',
            'major_axis', 'minor_axis',
            'solidity', 'extent', 'euler_number',
        ]
        df = df[[c for c in _col_order if c in df.columns]]
        self.set_progress(85)

        self.output_values['label_image'] = LabelData(payload=labeled.astype(np.int32),
                                                       image=rgb)
        self.output_values['table']       = TableData(payload=df)
        self.set_display(rgb)
        self.set_progress(100)
        return True, None


def _label_palette(n: int) -> list[tuple[int, int, int]]:
    """Return a cycling list of bright, distinct colours for label overlays."""
    BASE = [
        (255, 85, 85), (85, 170, 255), (85, 255, 170), (255, 200, 85),
        (200, 85, 255), (255, 170, 85), (85, 255, 255), (255, 85, 200),
        (170, 255, 85), (85, 85, 255), (255, 130, 130), (130, 255, 130),
    ]
    if n <= len(BASE):
        return BASE
    import colorsys
    extra = [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / n, 0.85, 0.95))
        for i in range(n)
    ]
    return extra


# ===========================================================================
# 4 — BlobDetectNode
# ===========================================================================

class BlobDetectNode(BaseImageProcessNode):
    """
    Detects bright blobs using Laplacian-of-Gaussian (LoG) filtering.

    Finds roughly circular bright spots such as cells, nuclei, vesicles, or puncta
    using `skimage.feature.blob_log`.

    Outputs:
    - *table* — one row per blob with columns `y`, `x`, `radius_px`
    - *overlay* — original image with detected blobs circled in red

    **Min Radius** — smallest blob radius to detect (pixels).
    **Max Radius** — largest blob radius to detect (pixels).
    **Threshold** — detection sensitivity; lower values find more blobs.

    Keywords: blob detect, log, spot detection, puncta, nuclei, 斑點偵測, 亮點, 分割, 強度, 影像處理
    """
    __identifier__ = 'nodes.image_process.measure'
    NODE_NAME      = 'Blob Detect'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['table', 'image']}
    OUTPUT_COLUMNS = {'table': ['y', 'x', 'radius_px']}

    def __init__(self):
        super().__init__()
        self.add_input('image',     color=PORT_COLORS['image'])
        self.add_output('table',    color=PORT_COLORS['table'])
        self.add_output('overlay',  color=PORT_COLORS['image'])
        self._add_int_spinbox('min_radius', 'Min Radius (px)', value=3,  min_val=1, max_val=500)
        self._add_int_spinbox('max_radius', 'Max Radius (px)', value=20, min_val=1, max_val=500)
        self._add_float_spinbox('threshold', 'Threshold', value=0.1, min_val=0.001, max_val=10.0, step=0.01, decimals=3)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.feature import blob_log

        arr, pil_in, _ = _get_image_arr(self, 'image')
        if arr is None:
            return False, "No input connected"

        # Normalise to [0,1] float grayscale
        gray = _to_gray(arr).astype(np.float64) / 255.0
        min_r = int(self.get_property('min_radius'))
        max_r = int(self.get_property('max_radius'))
        thr   = float(self.get_property('threshold'))
        min_s = max(1.0, min_r / np.sqrt(2))
        max_s = max(min_s + 1, max_r / np.sqrt(2))
        self.set_progress(20)

        blobs = blob_log(gray, min_sigma=min_s, max_sigma=max_s,
                         num_sigma=10, threshold=thr)
        self.set_progress(70)

        rows = []
        for b in blobs:
            y, x, sigma = b
            rows.append({'y': float(y), 'x': float(x),
                         'radius_px': float(sigma * np.sqrt(2))})

        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['y', 'x', 'radius_px'])
        self.output_values['table'] = TableData(payload=df)

        # Draw overlay using PIL for ellipse drawing, then convert back to numpy
        overlay_rgb = _to_rgb(arr)
        pil_tmp = Image.fromarray(overlay_rgb, 'RGB')
        draw    = ImageDraw.Draw(pil_tmp)
        for r in rows:
            cx, cy, rad = r['x'], r['y'], r['radius_px']
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                         outline=(255, 60, 60), width=2)
        overlay_arr = np.array(pil_tmp)

        self._make_image_output(overlay_arr, 'overlay')
        self.set_display(overlay_arr)
        self.set_progress(100)
        return True, None


# ===========================================================================
# 5 — FrangiNode
# ===========================================================================

class FrangiNode(BaseImageProcessNode):
    """
    Enhances tubular structures using the Frangi multi-scale vesselness filter.

    Detects curvilinear features (blood vessels, filopodia, collagen fibres) across a
    range of scales using `skimage.filters.frangi`. Output is a response map normalised
    to 0-255 uint8 for downstream thresholding.

    **Sigma Min** — smallest vessel width scale to detect.
    **Sigma Max** — largest vessel width scale to detect.

    Keywords: frangi, vesselness, tubeness, ridge enhancement, fibers, 紋理, 邊緣增強, 影像處理, 管狀, 輪廓
    """
    __identifier__ = 'nodes.image_process.filter'
    NODE_NAME      = 'Frangi Tubeness'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}

    def __init__(self):
        super().__init__()
        self.add_input('image',  color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'])
        self._add_int_spinbox('sigma_min', 'Sigma Min', value=1,  min_val=1, max_val=100)
        self._add_int_spinbox('sigma_max', 'Sigma Max', value=8,  min_val=1, max_val=100)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.filters import frangi

        arr, pil_in, _ = _get_image_arr(self, 'image')
        if arr is None:
            return False, "No input connected"

        gray    = _to_gray(arr).astype(np.float64) / 255.0
        s_min   = int(self.get_property('sigma_min'))
        s_max   = int(self.get_property('sigma_max'))
        sigmas  = range(s_min, s_max + 1)
        self.set_progress(20)

        response = frangi(gray, sigmas=sigmas, black_ridges=False)
        self.set_progress(70)

        vmax = response.max()
        if vmax > 0:
            response = response / vmax
        result_f = response.astype(np.float32)

        self._make_image_output(result_f)
        self.set_display(result_f)
        self.set_progress(100)
        return True, None


# ===========================================================================
# 6 — ColocalizationNode
# ===========================================================================

class ColocalizationNode(BaseExecutionNode):
    """
    Computes colocalization metrics between two channels.

    All metrics respect the mask input when connected. Without a mask, all pixels are used.

    Metrics:

    - *Pearson r* — linear correlation (-1 to 1)
    - *Spearman r* — rank correlation, robust to non-linear relationships
    - *Kendall tau* — rank correlation, more robust for small samples
    - *MOC* — Manders' Overlap Coefficient (0 to 1)
    - *M1* — fraction of ch1 intensity where ch2 is above its Otsu threshold
    - *M2* — fraction of ch2 intensity where ch1 is above its Otsu threshold
    - *ICQ* — Li's Intensity Correlation Quotient (-0.5 to 0.5)

    Outputs a 1-row table and a scatter plot of ch1 vs ch2 intensities.

    Keywords: colocalization, coloc, pearson, spearman, kendall, manders, icq, 共定位, 強度, 相關性, 影像處理, 分析
    """
    __identifier__ = 'nodes.image_process.measure'
    NODE_NAME      = 'Colocalization'
    PORT_SPEC      = {'inputs': ['image', 'image', 'mask'], 'outputs': ['table', 'figure']}
    OUTPUT_COLUMNS = {
        'table': ['Pearson_r', 'Pearson_p', 'Spearman_r', 'Spearman_p',
                  'Kendall_tau', 'Kendall_p', 'MOC', 'M1', 'M2', 'ICQ',
                  'Otsu_ch1', 'Otsu_ch2', 'n_pixels']
    }

    def __init__(self):
        super().__init__()
        self.add_input('ch1',  color=PORT_COLORS['image'])
        self.add_input('ch2',  color=PORT_COLORS['image'])
        self.add_input('mask', color=PORT_COLORS['mask'])   # optional
        self.add_output('table',  color=PORT_COLORS['table'])
        self.add_output('figure', color=PORT_COLORS['figure'])

    def _read_gray(self, port_name: str):
        port = self.inputs().get(port_name)
        if not port or not port.connected_ports():
            return None
        cp   = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if isinstance(data, (ImageData, MaskData)):
            return _to_gray(data.payload).astype(np.float64)
        return None

    def evaluate(self):
        self.reset_progress()

        a = self._read_gray('ch1')
        b = self._read_gray('ch2')
        if a is None or b is None:
            return False, "Both ch1 and ch2 must be connected"
        if a.shape != b.shape:
            return False, f"Channel shapes differ: {a.shape} vs {b.shape}"

        mask = _get_mask_arr(self, 'mask')  # may be None
        self.set_progress(15)

        # Apply mask — use only masked pixels for all metrics
        if mask is not None and mask.any():
            a_m = a[mask].ravel()
            b_m = b[mask].ravel()
        else:
            a_m = a.ravel()
            b_m = b.ravel()

        if a_m.size < 2:
            self.mark_error()
            return False, "Not enough pixels for correlation (need at least 2)"

        self.set_progress(25)

        # ── Pearson (PCC) ─────────────────────────────────────────────────
        from scipy.stats import pearsonr, spearmanr, kendalltau
        pcc, pcc_p = pearsonr(a_m, b_m)

        # ── Spearman ─────────────────────────────────────────────────────
        spearman_r, spearman_p = spearmanr(a_m, b_m)

        # ── Kendall ──────────────────────────────────────────────────────
        # Subsample for large images (Kendall is O(n²))
        if a_m.size > 50000:
            idx = np.random.choice(a_m.size, 50000, replace=False)
            kendall_r, kendall_p = kendalltau(a_m[idx], b_m[idx])
        else:
            kendall_r, kendall_p = kendalltau(a_m, b_m)

        self.set_progress(35)

        # ── Manders Overlap Coefficient ────────────────────────────────────
        denom = np.sqrt(np.sum(a_m ** 2) * np.sum(b_m ** 2))
        moc   = float(np.sum(a_m * b_m) / denom) if denom > 0 else float('nan')

        # ── Manders M1 / M2 (Otsu thresholds) ─────────────────────────────
        from skimage.filters import threshold_otsu
        t1 = float(threshold_otsu(a_m)) if a_m.max() > a_m.min() else 0.0
        t2 = float(threshold_otsu(b_m)) if b_m.max() > b_m.min() else 0.0
        s_a = np.sum(a_m)
        s_b = np.sum(b_m)
        m1 = float(np.sum(a_m[b_m > t2]) / s_a) if s_a > 0 else float('nan')
        m2 = float(np.sum(b_m[a_m > t1]) / s_b) if s_b > 0 else float('nan')
        self.set_progress(50)

        # ── Li ICQ ─────────────────────────────────────────────────────────
        product = (a_m - a_m.mean()) * (b_m - b_m.mean())
        icq     = float(np.sum(product > 0) / product.size - 0.5)

        row = {
            'Pearson_r':    [round(float(pcc), 4)],
            'Pearson_p':    [round(float(pcc_p), 6)],
            'Spearman_r':   [round(float(spearman_r), 4)],
            'Spearman_p':   [round(float(spearman_p), 6)],
            'Kendall_tau':  [round(float(kendall_r), 4)],
            'Kendall_p':    [round(float(kendall_p), 6)],
            'MOC':          [round(moc, 4)],
            'M1':           [round(m1, 4)],
            'M2':           [round(m2, 4)],
            'ICQ':          [round(icq, 4)],
            'Otsu_ch1':     [round(float(t1), 4)],
            'Otsu_ch2':     [round(float(t2), 4)],
            'n_pixels':     [int(a_m.size)],
        }
        self.output_values['table'] = TableData(payload=pd.DataFrame(row))
        self.set_progress(65)

        # ── Scatter plot (ch1 vs ch2, subsampled) ──────────────────────────
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        MAX_PTS = 20_000
        flat_a  = a_m
        flat_b  = b_m

        if len(flat_a) > MAX_PTS:
            idx    = np.random.choice(len(flat_a), MAX_PTS, replace=False)
            flat_a = flat_a[idx]
            flat_b = flat_b[idx]

        fig    = Figure(figsize=(5, 5))
        canvas = FigureCanvasAgg(fig)
        ax     = fig.add_subplot(111)
        ax.scatter(flat_a, flat_b, s=1, alpha=0.3, c='steelblue', linewidths=0)
        ax.set_xlabel('Channel 1 intensity')
        ax.set_ylabel('Channel 2 intensity')
        ax.set_title(f'PCC = {pcc:.3f}  |  M1={m1:.3f}  M2={m2:.3f}')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        fig.tight_layout()

        self.output_values['figure'] = FigureData(payload=fig)
        self.set_progress(100)
        self.mark_clean()
        return True, None


# ===========================================================================
# 7 — GLCMTextureNode
# ===========================================================================

class GLCMTextureNode(BaseImageProcessNode):
    """
    Computes Haralick texture features from a Grey-Level Co-occurrence Matrix (GLCM).

    Averages texture features over four orientations (0, 45, 90, 135 degrees) at the
    given pixel distance. Outputs a single-row table with columns: `contrast`,
    `dissimilarity`, `homogeneity`, `energy`, `correlation`, `ASM`.

    **Distance** — pixel offset for co-occurrence pairs.
    **Grey Levels** — number of quantisation levels (fewer = faster, coarser).

    Keywords: glcm, haralick, texture analysis, co-occurrence, contrast, 紋理, 對比, 共現矩陣, 影像分析, 強度
    """
    __identifier__ = 'nodes.image_process.measure'
    NODE_NAME      = 'GLCM Texture'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['table']}
    OUTPUT_COLUMNS = {
        'table': ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    }

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('table', color=PORT_COLORS['table'])
        self._add_int_spinbox('distance', 'Distance (px)', value=1, min_val=1, max_val=50)
        self._add_int_spinbox('levels', 'Grey Levels', value=64, min_val=4, max_val=256)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.feature import graycomatrix, graycoprops

        arr, pil_in, _ = _get_image_arr(self, 'image')
        if arr is None:
            return False, "No input connected"

        gray   = _to_gray(arr)
        dist   = int(self.get_property('distance'))
        levels = int(self.get_property('levels'))
        self.set_progress(20)

        # Quantise to [0, levels-1]
        # gray is float [0,1] — quantize directly to [0, levels-1]
        arr_q  = (np.clip(gray.astype(np.float64), 0, 1) * (levels - 1)).astype(np.uint8)
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm   = graycomatrix(arr_q, distances=[dist], angles=angles,
                              levels=levels, normed=True)
        self.set_progress(60)

        row = {}
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            row[prop] = [float(graycoprops(glcm, prop).mean())]

        self.output_values['table'] = TableData(payload=pd.DataFrame(row))
        self.set_display(arr)
        self.set_progress(100)
        return True, None


# ===========================================================================
# ImageStatsNode
# ===========================================================================

class ImageStatsNode(BaseImageProcessNode):
    """
    Measures whole-mask region properties and pixel intensity statistics in a single table row.

    Connect an image, a mask, or both -- at least one must be connected.

    Mask columns (present when mask is connected):
    - `image_size_px` — total pixels in the image (H x W); denominator for area_fraction
    - `area_px` — number of foreground pixels
    - `area_fraction` — area_px / image_size_px (0-1); multiply by 100 for %
    - `perimeter_px` — boundary length in pixels
    - `solidity` — area / convex_hull_area (1 = convex)
    - `eccentricity` — shape elongation (0 = circle, 1 = line)
    - `major_axis_px` — major axis of the fitted ellipse
    - `minor_axis_px` — minor axis of the fitted ellipse
    - `extent` — area / bounding_box_area
    - `euler_number` — 1 = no holes; decreases by 1 per enclosed hole
    - `centroid_y`, `centroid_x` — pixel coordinates of the mask centroid

    Intensity columns (present when image is connected, pixel values 0-255):
    - `mean`, `std`, `min`, `max`, `median` — overall grayscale or luminance; restricted to masked region when mask is also connected

    Per-channel columns (RGB image with *Per Channel* checked):
    - `mean_r/g/b`, `std_r/g/b`, `min_r/g/b`, `max_r/g/b`

    **Column Prefix** — optional string prepended to all column names.

    Keywords: mean intensity, area fraction, image size, statistics, coverage, perimeter, solidity, eccentricity, DAB, stain quantification, 平均強度, 面積比, 影像大小, 統計, 周長, 溶實度, 離心率, 影像分析
    """
    __identifier__ = 'nodes.image_process.measure'
    NODE_NAME      = 'Image Stats'
    PORT_SPEC      = {'inputs': ['image', 'mask'], 'outputs': ['table']}
    OUTPUT_COLUMNS = {
        'table': [
            'image_size_px', 'area_px', 'area_fraction',
            'perimeter_px', 'solidity', 'eccentricity',
            'major_axis_px', 'minor_axis_px', 'extent', 'euler_number',
            'centroid_y', 'centroid_x',
            'mean', 'std', 'min', 'max', 'median',
        ]
    }

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_input('mask',  color=PORT_COLORS['mask'])
        self.add_output('table', color=PORT_COLORS['table'])
        self.add_text_input('col_prefix', 'Column Prefix', text='', tab='Parameters')
        self.add_checkbox('per_channel', '', text='Per Channel (RGB)', state=True)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        arr, pil_in, _ = _get_image_arr(self, 'image')
        mask            = _get_mask_arr(self, 'mask')

        if arr is None and mask is None:
            return False, "Connect an image, a mask, or both"

        prefix = str(self.get_property('col_prefix') or '')
        row: dict = {}

        # ── Mask area + region shape stats ───────────────────────────────────
        if mask is not None:
            from skimage.measure import regionprops
            total_px = int(mask.size)
            area_px  = int(mask.sum())
            row[f'{prefix}image_size_px'] = total_px
            row[f'{prefix}area_px']       = area_px
            row[f'{prefix}area_fraction'] = float(mask.mean())   # area_px / total_px

            if mask.any():
                labeled    = mask.astype(np.int32)
                props_list = regionprops(labeled)
                p          = props_list[0]
                row[f'{prefix}perimeter_px']  = float(p.perimeter)
                row[f'{prefix}solidity']      = float(p.solidity)
                row[f'{prefix}eccentricity']  = float(p.eccentricity)
                row[f'{prefix}major_axis_px'] = float(p.axis_major_length)
                row[f'{prefix}minor_axis_px'] = float(p.axis_minor_length)
                row[f'{prefix}extent']        = float(p.extent)
                row[f'{prefix}euler_number']  = int(p.euler_number)
                row[f'{prefix}centroid_y']    = float(p.centroid[0])
                row[f'{prefix}centroid_x']    = float(p.centroid[1])
            else:
                for col in ('perimeter_px', 'solidity', 'eccentricity',
                            'major_axis_px', 'minor_axis_px', 'extent'):
                    row[f'{prefix}{col}'] = 0.0
                row[f'{prefix}euler_number'] = 0
                row[f'{prefix}centroid_y']   = 0.0
                row[f'{prefix}centroid_x']   = 0.0

        self.set_progress(40)

        # ── Intensity stats ───────────────────────────────────────────────────
        if arr is not None:
            per_ch = bool(self.get_property('per_channel'))
            is_rgb = arr.ndim == 3 and arr.shape[2] >= 3
            gray   = _to_gray(arr).astype(np.float64)

            def _stats(pixels: np.ndarray) -> dict:
                p = pixels.ravel()
                return {
                    'mean':   float(np.mean(p)),
                    'std':    float(np.std(p)),
                    'min':    float(np.min(p)),
                    'max':    float(np.max(p)),
                    'median': float(np.median(p)),
                }

            if mask is not None and mask.shape == gray.shape:
                if not mask.any():
                    return False, "Mask is empty — no pixels to measure"
                overall_px = gray[mask]
            else:
                overall_px = gray

            for k, v in _stats(overall_px).items():
                row[f'{prefix}{k}'] = v
            self.set_progress(70)

            if per_ch and is_rgb:
                for i, ch_name in enumerate(('r', 'g', 'b')):
                    ch    = arr[:, :, i].astype(np.float64)
                    ch_px = ch[mask] if (mask is not None and mask.shape == ch.shape) else ch
                    s     = _stats(ch_px)
                    for k in ('mean', 'std', 'min', 'max'):
                        row[f'{prefix}{k}_{ch_name}'] = s[k]

        self.set_progress(88)
        self.output_values['table'] = TableData(
            payload=pd.DataFrame({k: [v] for k, v in row.items()})
        )
        if arr is not None:
            preview = arr
        else:
            preview = mask.astype(np.uint8) * 255
        self.set_display(preview)
        self.set_progress(100)
        return True, None


# ===========================================================================
# 8 — IntensityProfileNode  (interactive line-drawing widget)
# ===========================================================================

class _LineGraphicsView(QGraphicsView):
    """
    Embedded view that lets the user draw and adjust a single line segment on the loaded image.

    Interactions:
    - Click + drag on empty area — draw a new line
    - Click + drag near endpoint — move that endpoint
    - Click + drag on line body — move the whole line
    - Delete key — clear the line
    - Middle-drag / scroll wheel — pan / zoom
    """
    line_committed = Signal(float, float, float, float)   # x1, y1, x2, y2

    _HANDLE_R = 6    # endpoint handle radius in scene pixels
    _HIT_TOL  = 12   # click-detection tolerance in scene pixels

    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._line_item: QGraphicsLineItem | None = None
        self._h1: QGraphicsEllipseItem | None = None   # start-point handle
        self._h2: QGraphicsEllipseItem | None = None   # end-point handle
        self._dragging: str | None = None              # 'new'|'h1'|'h2'|'body'
        self._drag_start   = QPointF()
        self._line_at_drag: tuple | None = None        # snapshot for body drag
        self._pan_start    = None
        self._scale        = 1.0

        self._line_pen   = QPen(QColor(255, 60,  60),  2.0, Qt.PenStyle.SolidLine,
                                Qt.PenCapStyle.RoundCap)
        self._hpen       = QPen(QColor(255, 220, 50),  1.5)
        self._hbrush     = QBrush(QColor(255, 220, 50, 200))

    # ── public API ────────────────────────────────────────────────────────────

    def clear_line(self):
        for item in (self._line_item, self._h1, self._h2):
            if item is not None:
                self.scene().removeItem(item)
        self._line_item = self._h1 = self._h2 = None
        self._dragging = None

    def get_line(self) -> tuple | None:
        """Returns (x1, y1, x2, y2) in image/scene coordinates, or None."""
        if self._line_item is None:
            return None
        ln = self._line_item.line()
        return ln.x1(), ln.y1(), ln.x2(), ln.y2()

    def set_line(self, x1: float, y1: float, x2: float, y2: float):
        """Restore a saved line (must be called on the main thread)."""
        self.clear_line()
        self._create_items(QPointF(x1, y1), QPointF(x2, y2))

    def load_image(self, img):
        """Replace the background image. Accepts numpy array. Existing line is kept in place."""
        for item in list(self.scene().items()):
            if isinstance(item, QGraphicsPixmapItem):
                self.scene().removeItem(item)
        arr = _to_rgb(img) if isinstance(img, np.ndarray) else np.array(img.convert('RGB'))
        h, w = arr.shape[:2]
        q    = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pm   = QPixmap.fromImage(q)
        px   = QGraphicsPixmapItem(pm)
        px.setZValue(-1)
        self.scene().addItem(px)
        self.scene().setSceneRect(QRectF(pm.rect()))
        self.fitInView(px, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_in(self):    self._zoom(1.2)
    def zoom_out(self):   self._zoom(1 / 1.2)
    def zoom_reset(self):
        for item in self.scene().items():
            if isinstance(item, QGraphicsPixmapItem):
                self.fitInView(item, Qt.AspectRatioMode.KeepAspectRatio)
                return

    # ── mouse events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        sp = self.mapToScene(event.pos())

        if self._line_item is not None:
            ln = self._line_item.line()
            # Hit-test endpoints first
            if self._dist(sp, QPointF(ln.x1(), ln.y1())) <= self._HIT_TOL:
                self._dragging = 'h1'
                event.accept(); return
            if self._dist(sp, QPointF(ln.x2(), ln.y2())) <= self._HIT_TOL:
                self._dragging = 'h2'
                event.accept(); return
            # Hit-test line body
            if self._near_line(sp, ln):
                self._dragging    = 'body'
                self._drag_start  = sp
                self._line_at_drag = (ln.x1(), ln.y1(), ln.x2(), ln.y2())
                event.accept(); return

        # Start drawing a new line
        self.clear_line()
        self._dragging   = 'new'
        self._drag_start = sp
        event.accept()

    def mouseMoveEvent(self, event):
        if self._pan_start is not None:
            d = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            event.accept(); return

        if self._dragging is None:
            return super().mouseMoveEvent(event)

        sp = self.mapToScene(event.pos())

        if self._dragging == 'new':
            if self._line_item is None:
                self._create_items(self._drag_start, sp)
            else:
                ln = self._line_item.line()
                self._move_line(ln.x1(), ln.y1(), sp.x(), sp.y())

        elif self._dragging == 'h1':
            ln = self._line_item.line()
            self._move_line(sp.x(), sp.y(), ln.x2(), ln.y2())

        elif self._dragging == 'h2':
            ln = self._line_item.line()
            self._move_line(ln.x1(), ln.y1(), sp.x(), sp.y())

        elif self._dragging == 'body':
            dx = sp.x() - self._drag_start.x()
            dy = sp.y() - self._drag_start.y()
            x1, y1, x2, y2 = self._line_at_drag
            self._move_line(x1 + dx, y1 + dy, x2 + dx, y2 + dy)

        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept(); return

        if event.button() == Qt.MouseButton.LeftButton and self._dragging is not None:
            if self._line_item is not None:
                ln = self._line_item.line()
                if abs(ln.x2() - ln.x1()) + abs(ln.y2() - ln.y1()) < 3:
                    self.clear_line()
                else:
                    self.line_committed.emit(ln.x1(), ln.y1(), ln.x2(), ln.y2())
            self._dragging = None
            event.accept(); return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.clear_line()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        self._zoom(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)
        event.accept()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _create_items(self, p1: QPointF, p2: QPointF):
        ln = QGraphicsLineItem(QLineF(p1, p2))
        ln.setPen(self._line_pen)
        ln.setZValue(1)
        self.scene().addItem(ln)
        self._line_item = ln

        r = self._HANDLE_R
        for attr, pos in (('_h1', p1), ('_h2', p2)):
            h = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            h.setPos(pos)
            h.setPen(self._hpen)
            h.setBrush(self._hbrush)
            h.setZValue(2)
            self.scene().addItem(h)
            setattr(self, attr, h)

    def _move_line(self, x1, y1, x2, y2):
        self._line_item.setLine(QLineF(x1, y1, x2, y2))
        self._h1.setPos(x1, y1)
        self._h2.setPos(x2, y2)

    @staticmethod
    def _dist(a: QPointF, b: QPointF) -> float:
        return abs(a.x() - b.x()) + abs(a.y() - b.y())

    def _near_line(self, pt: QPointF, ln: QLineF) -> bool:
        dx, dy = ln.x2() - ln.x1(), ln.y2() - ln.y1()
        L2 = dx * dx + dy * dy
        if L2 < 1:
            return False
        t  = max(0.0, min(1.0, ((pt.x() - ln.x1()) * dx + (pt.y() - ln.y1()) * dy) / L2))
        return self._dist(pt, QPointF(ln.x1() + t * dx, ln.y1() + t * dy)) <= self._HIT_TOL

    def _zoom(self, factor: float):
        self.scale(factor, factor)
        self._scale *= factor


class _NodeLineViewWidget(NodeBaseWidget):
    """Embeds a _LineGraphicsView on the node card for interactive line drawing."""

    line_committed   = Signal(float, float, float, float)
    _img_signal      = Signal(object)             # numpy array → main thread
    _set_line_signal = Signal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent, name='_line_view', label='')

        container = QtWidgets.QWidget()
        container.setMinimumWidth(340)
        root = QtWidgets.QVBoxLayout(container)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(3)

        # Toolbar
        tb = QtWidgets.QHBoxLayout()
        tb.addWidget(QtWidgets.QLabel('Draw line on image:'))
        tb.addStretch()
        for icon, slot, tip in (('+', 'zoom_in', 'Zoom in'),
                                ('-', 'zoom_out', 'Zoom out'),
                                ('⊙', 'zoom_reset', 'Fit to view')):
            b = QtWidgets.QPushButton(icon)
            b.setFixedSize(42, 24)
            b.setProperty('compact', True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _, s=slot: getattr(self._view, s)())
            tb.addWidget(b)
        clear_btn = QtWidgets.QPushButton('Clear')
        clear_btn.setFixedHeight(22)
        clear_btn.clicked.connect(self._view_clear)
        tb.addWidget(clear_btn)
        root.addLayout(tb)

        # Drawing view
        self._scene = QGraphicsScene()
        self._view  = _LineGraphicsView(self._scene)
        self._view.setMinimumSize(420, 380)
        self._view.line_committed.connect(self.line_committed)
        root.addWidget(self._view)

        tip = QtWidgets.QLabel('Drag to draw  ·  Drag endpoints or line to adjust  ·  Delete to clear')
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tip.setStyleSheet('color:#888; font-size:9px; padding:1px;')
        root.addWidget(tip)

        self.set_custom_widget(container)

        # Thread-safe signals
        self._img_signal.connect(self._view.load_image, Qt.ConnectionType.QueuedConnection)
        self._set_line_signal.connect(self._view.set_line, Qt.ConnectionType.QueuedConnection)

    def _view_clear(self):
        self._view.clear_line()

    # ── public API ────────────────────────────────────────────────────────────

    def load_image(self, img):
        """Thread-safe: load a numpy array into the view."""
        if threading.current_thread() is threading.main_thread():
            self._view.load_image(img)
        else:
            self._img_signal.emit(img)

    def set_line(self, x1: float, y1: float, x2: float, y2: float):
        """Thread-safe: restore a saved line."""
        if threading.current_thread() is threading.main_thread():
            self._view.set_line(x1, y1, x2, y2)
        else:
            self._set_line_signal.emit(x1, y1, x2, y2)

    def get_line(self) -> tuple | None:
        return self._view.get_line()

    # required by NodeBaseWidget — unused but must return something
    def get_value(self):  return self._view.get_line()
    def set_value(self, v): pass


class IntensityProfileNode(BaseImageProcessNode):
    """
    Plots pixel intensity along an interactively drawn line segment.

    Draw the line directly on the image preview. The plot shows intensity (or per-channel
    R/G/B) vs distance in pixels. Useful for measuring gradients, checking membrane
    sharpness, or verifying stain distribution across tissue layers.

    Keywords: intensity profile, line scan, plot along line, rgb profile, gradient measurement, 強度, 線掃描, 輪廓, 影像分析, 梯度
    """
    __identifier__ = 'nodes.image_process.measure'
    NODE_NAME      = 'Intensity Profile'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__()
        self.add_input('image',  color=PORT_COLORS['image'])
        self.add_output('figure', color=PORT_COLORS['figure'])

        # Persist the drawn line across workflow save/load
        self.create_property(
            'line_data', '',
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value
        )

        # Interactive line-drawing widget (replaces spinboxes + separate preview)
        self._line_widget = _NodeLineViewWidget(self.view)
        self.add_custom_widget(self._line_widget)
        self._line_widget.line_committed.connect(self._on_line_committed)

        # Standard live-preview checkbox (auto-evaluate on property change)
        self.add_checkbox('live_preview', '', text='Live Update', state=True)

    def _on_line_committed(self, x1: float, y1: float, x2: float, y2: float):
        """Called on the main thread when the user finishes drawing/adjusting."""
        self.set_property('line_data', json.dumps([x1, y1, x2, y2]))

    def on_input_connected(self, in_port, out_port):
        super().on_input_connected(in_port, out_port)
        # Refresh the image in the view when a new upstream node connects
        self._load_image_into_widget()

    def _load_image_into_widget(self):
        img_port = self.inputs().get('image')
        if not img_port or not img_port.connected_ports():
            return
        cp   = img_port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if isinstance(data, ImageData):
            self._line_widget.load_image(data.payload)  # numpy array

    def evaluate(self):
        self.reset_progress()
        from scipy.ndimage import map_coordinates
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        arr, pil_in, _ = _get_image_arr(self, 'image')
        if arr is None:
            return False, "No input connected"

        # Always update the image preview in the widget (thread-safe)
        self._line_widget.load_image(arr)
        self.set_progress(10)

        # Prefer the widget's live state; fall back to saved property on reload
        line = self._line_widget.get_line()
        if line is None:
            saved = str(self.get_property('line_data') or '').strip()
            if saved:
                try:
                    coords = json.loads(saved)
                    if len(coords) == 4:
                        x1, y1, x2, y2 = [float(c) for c in coords]
                        self._line_widget.set_line(x1, y1, x2, y2)
                        line = (x1, y1, x2, y2)
                except Exception:
                    pass

        if line is None:
            self.mark_error()
            return False, "Draw a line on the image to measure intensity."

        H, W = arr.shape[:2]
        x1 = float(np.clip(line[0], 0, W - 1))
        y1 = float(np.clip(line[1], 0, H - 1))
        x2 = float(np.clip(line[2], 0, W - 1))
        y2 = float(np.clip(line[3], 0, H - 1))

        length  = max(1, int(np.hypot(x2 - x1, y2 - y1)))
        xs      = np.linspace(x1, x2, length)
        ys      = np.linspace(y1, y2, length)
        dist_ax = np.linspace(0.0, float(length), length)
        self.set_progress(30)

        fig    = Figure(figsize=(6, 3))
        _      = FigureCanvasAgg(fig)
        ax     = fig.add_subplot(111)

        if arr.ndim == 2:
            profile = map_coordinates(arr.astype(float), [ys, xs], order=1)
            ax.plot(dist_ax, profile, color='gray', lw=1.5)
        else:
            for c, col in enumerate(['red', 'green', 'blue']):
                if c >= arr.shape[2]:
                    break
                profile = map_coordinates(arr[:, :, c].astype(float), [ys, xs], order=1)
                ax.plot(dist_ax, profile, color=col, lw=1.5, label=col.upper())
            ax.legend(fontsize=8)

        ax.set_xlabel('Distance (px)')
        ax.set_ylabel('Intensity')
        ax.set_title(f'Profile  ({int(x1)},{int(y1)}) → ({int(x2)},{int(y2)})')
        ax.set_ylim(0, 255)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        self.set_progress(90)

        self.output_values['figure'] = FigureData(payload=fig)
        self.mark_clean()
        self.set_progress(100)
        return True, None


# ===========================================================================
# Edge Detection Nodes
# ===========================================================================

class CannyEdgeNode(BaseImageProcessNode):
    """
    Detects edges using the Canny algorithm, producing a thin binary edge mask.

    Converts input to grayscale, applies optional Gaussian blur, then runs the Canny
    algorithm with hysteresis thresholding. Leave both thresholds at `0` to use
    automatic Otsu-based values.

    **Sigma** — scale of detected edges; larger values produce coarser edges.
    **Low Threshold** — lower bound for hysteresis thresholding.
    **High Threshold** — upper bound for hysteresis thresholding.

    Keywords: canny, edge detection, binary edges, hysteresis threshold, contour prep, 邊緣, 輪廓, 二值化, 影像處理, 閾值
    """
    __identifier__ = 'nodes.image_process.filter'
    NODE_NAME      = 'Canny Edge'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['mask']}

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('mask', color=PORT_COLORS['mask'], multi_output=True)
        self._add_float_spinbox('sigma',           'Sigma',           value=1.0, min_val=0.1, max_val=20.0, step=0.1, decimals=1)
        self._add_float_spinbox('low_threshold',   'Low Threshold',   value=0.0, min_val=0.0, max_val=65535.0, step=1.0, decimals=1)
        self._add_float_spinbox('high_threshold',  'High Threshold',  value=0.0, min_val=0.0, max_val=65535.0, step=1.0, decimals=1)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.feature import canny as skimage_canny

        arr, _, _ = _get_image_arr(self, 'image')
        if arr is None:
            return False, "No image connected"

        self.set_progress(20)
        gray = arr.astype(np.float32) if arr.ndim == 2 else arr.mean(axis=2).astype(np.float32)

        sigma = float(self.get_property('sigma'))
        lo    = float(self.get_property('low_threshold'))  or None  # 0 → None (auto)
        hi    = float(self.get_property('high_threshold')) or None  # 0 → None (auto)

        self.set_progress(40)
        edges = skimage_canny(gray, sigma=sigma, low_threshold=lo, high_threshold=hi)
        self.set_progress(80)

        out = (edges.astype(np.uint8) * 255)
        self.output_values['mask'] = MaskData(payload=out)
        self.set_display(out)
        self.set_progress(100)
        return True, None


class SobelEdgeNode(BaseImageProcessNode):
    """
    Computes edge strength using the Sobel gradient-magnitude filter.

    Calculates the Sobel gradient in X and Y, combines them as `sqrt(Gx^2 + Gy^2)`,
    and scales to 0-255. Good for visualising edge strength. Connect to
    BinaryThresholdNode to convert the gradient into a mask.

    Keywords: sobel, gradient magnitude, edge strength, derivative filter, boundary enhancement, 邊緣, 梯度, 強度, 影像處理, 輪廓
    """
    __identifier__ = 'nodes.image_process.filter'
    NODE_NAME      = 'Sobel Edge'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}

    def __init__(self):
        super().__init__()
        self.add_input('image',   color=PORT_COLORS['image'])
        self.add_output('image',  color=PORT_COLORS['image'], multi_output=True)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.filters import sobel

        arr, _, _ = _get_image_arr(self, 'image')
        if arr is None:
            return False, "No image connected"

        self.set_progress(30)
        gray = arr.astype(np.float32) if arr.ndim == 2 else arr.mean(axis=2).astype(np.float32)
        mag  = sobel(gray)
        self.set_progress(70)

        mn, mx = mag.min(), mag.max()
        scaled = ((mag - mn) / (mx - mn)).astype(np.float32) if mx > mn else np.zeros_like(mag, dtype=np.float32)
        self._make_image_output(scaled)
        self.set_display(scaled)
        self.set_progress(100)
        return True, None


class PrewittEdgeNode(BaseImageProcessNode):
    """
    Computes edge strength using the Prewitt gradient-magnitude filter.

    Similar to Sobel but uses equal-weight kernels. Slightly more sensitive to noise,
    but sometimes picks up finer detail at diagonal edges. Connect to
    BinaryThresholdNode to convert the gradient into a mask.

    Keywords: prewitt, gradient filter, edge map, derivative operator, diagonal edges, 邊緣, 梯度, 影像處理, 輪廓, 方向
    """
    __identifier__ = 'nodes.image_process.filter'
    NODE_NAME      = 'Prewitt Edge'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}

    def __init__(self):
        super().__init__()
        self.add_input('image',   color=PORT_COLORS['image'])
        self.add_output('image',  color=PORT_COLORS['image'], multi_output=True)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.filters import prewitt

        arr, _, _ = _get_image_arr(self, 'image')
        if arr is None:
            return False, "No image connected"

        self.set_progress(30)
        gray = arr.astype(np.float32) if arr.ndim == 2 else arr.mean(axis=2).astype(np.float32)
        mag  = prewitt(gray)
        self.set_progress(70)

        mn, mx = mag.min(), mag.max()
        scaled = ((mag - mn) / (mx - mn)).astype(np.float32) if mx > mn else np.zeros_like(mag, dtype=np.float32)
        self._make_image_output(scaled)
        self.set_display(scaled)
        self.set_progress(100)
        return True, None


class LaplacianEdgeNode(BaseImageProcessNode):
    """
    Highlights regions of rapid intensity change using the Laplacian of Gaussian (LoG) filter.

    Responds to blob-like features as well as sharp edges at the scale set by sigma.
    Output is a signed response normalised to 0-255 where `128` represents the
    zero-crossing. Connect to BinaryThresholdNode for a binary mask.

    **Sigma** — spatial scale of the Gaussian smoothing before the Laplacian.

    Keywords: laplacian, log edge, zero-crossing, second derivative, blob response, 邊緣, 斑點偵測, 影像處理, 輪廓, 二階導數
    """
    __identifier__ = 'nodes.image_process.filter'
    NODE_NAME      = 'Laplacian Edge'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}

    def __init__(self):
        super().__init__()
        self.add_input('image',   color=PORT_COLORS['image'])
        self.add_output('image',  color=PORT_COLORS['image'], multi_output=True)
        self._add_float_spinbox('sigma', 'Sigma', value=1.0, min_val=0.1, max_val=20.0, step=0.1, decimals=1)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.filters import gaussian, laplace

        arr, _, _ = _get_image_arr(self, 'image')
        if arr is None:
            return False, "No image connected"

        self.set_progress(20)
        sigma = float(self.get_property('sigma'))
        gray  = arr.astype(np.float32) if arr.ndim == 2 else arr.mean(axis=2).astype(np.float32)
        blurred = gaussian(gray, sigma=sigma, preserve_range=True)
        self.set_progress(50)
        log   = laplace(blurred)
        self.set_progress(75)

        mn, mx = log.min(), log.max()
        ext = max(abs(mn), abs(mx))
        if ext > 0:
            scaled = ((log / ext * 0.5) + 0.5).clip(0, 1).astype(np.float32)
        else:
            scaled = np.full(log.shape, 0.5, dtype=np.float32)
        self._make_image_output(scaled)
        self.set_display(scaled)
        self.set_progress(100)
        return True, None


# ===========================================================================
# FindContoursNode
# ===========================================================================

class FindContoursNode(BaseImageProcessNode):
    """
    Finds all contours in a binary mask or edge image at a given intensity level.

    Outputs:
    - *mask* — binary image with all selected contours drawn as lines
    - *table* — coordinate table with columns `contour_id`, `x`, `y`

    Modes:
    - *All contours* — return every contour found, sorted largest first
    - *Largest only* — return only the contour with the greatest enclosed area
    - *Filter by min area* — discard contours with enclosed area below **Min Area**

    **Contour Level** — intensity threshold for contour detection (normalised 0-1).
    **Line Width** — stroke width in pixels for the output mask drawing.

    Keywords: find contours, outline, boundary tracing, polygon points, outer contour, 輪廓, 邊界, 多邊形, 影像處理, 分割
    """
    __identifier__ = 'nodes.image_process.measure'
    NODE_NAME      = 'Find Contours'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['mask', 'table']}
    OUTPUT_COLUMNS = {'table': ['contour_id', 'x', 'y']}

    def __init__(self):
        super().__init__()
        self.add_input('image/mask',  color=PORT_COLORS['image'])
        self.add_output('mask',       color=PORT_COLORS['mask'],  multi_output=True)
        self.add_output('table',      color=PORT_COLORS['table'], multi_output=True)
        self._add_float_spinbox('level',      'Contour Level (0–1)', value=0.5, min_val=0.0, max_val=1.0, step=0.05, decimals=2)
        self._add_int_spinbox('line_width',   'Line Width (px)',      value=2,   min_val=1,   max_val=50)
        self.add_combo_menu('mode', 'Mode',
                            items=['All contours', 'Largest only',
                                   'Filter by min area'])
        self._add_int_spinbox('min_area', 'Min Area (px²)', value=100,
                              min_val=0, max_val=9999999)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.measure import find_contours

        arr, _, _ = _get_image_arr(self, 'image/mask')
        if arr is None:
            return False, "No input connected"

        self.set_progress(20)

        # Normalise to 0–1 float for find_contours
        gray = arr.astype(np.float32) if arr.ndim == 2 else arr.mean(axis=2).astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0

        level    = float(self.get_property('level'))
        contours = find_contours(gray, level=level)

        if not contours:
            return False, f"No contours found at level {level:.2f}"

        self.set_progress(50)

        # Compute enclosed area for each contour (shoelace formula)
        def _area(c):
            x, y = c[:, 1], c[:, 0]
            return abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) * 0.5

        mode = str(self.get_property('mode'))
        if 'Largest' in mode:
            contours = [max(contours, key=_area)]
        elif 'Filter' in mode:
            min_a = int(self.get_property('min_area'))
            contours = [c for c in contours if _area(c) >= min_a]
            if not contours:
                return False, f"No contours with area >= {min_a} px²"
            # Sort largest first
            contours.sort(key=_area, reverse=True)
        else:
            # All contours — sort largest first
            contours.sort(key=_area, reverse=True)

        self.set_progress(65)

        H, W = gray.shape
        lw   = int(self.get_property('line_width'))

        # Output 1: draw all selected contours as a binary mask
        mask_pil = Image.new('L', (W, H), 0)
        draw = ImageDraw.Draw(mask_pil)
        for contour in contours:
            pts = [(float(c[1]), float(c[0])) for c in contour]
            if len(pts) >= 2:
                draw.line(pts + [pts[0]], fill=255, width=lw)
        mask_arr = np.array(mask_pil)
        self.output_values['mask'] = MaskData(payload=mask_arr)
        self.set_progress(80)

        # Output 2: (contour_id, x, y) coordinate table
        dfs = []
        for i, contour in enumerate(contours):
            cdf = pd.DataFrame({
                'contour_id': i,
                'x': contour[:, 1].round(2),
                'y': contour[:, 0].round(2),
            })
            dfs.append(cdf)
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
            columns=['contour_id', 'x', 'y'])
        self.output_values['table'] = TableData(payload=df)

        self.set_display(mask_arr)
        self.set_progress(100)
        return True, None


# ===========================================================================
# Hough Transform Nodes
# ===========================================================================

def _edge_to_binary(arr: np.ndarray) -> np.ndarray:
    """Convert an image/mask array to a bool edge map (H×W)."""
    gray = arr if arr.ndim == 2 else arr.mean(axis=2)
    return gray > 0


def _arr_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Return a uint8 (H, W, 3) array from any input array."""
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3].astype(np.uint8)
    gray = arr if arr.ndim == 2 else arr.mean(axis=2)
    gray8 = ((gray / max(gray.max(), 1)) * 255).astype(np.uint8)
    return np.stack([gray8, gray8, gray8], axis=2)


class HoughCirclesNode(BaseImageProcessNode):
    """
    Detects circles in a Canny edge image using the Hough circle transform.

    Sweeps a range of radii and votes for circle centres; peaks in the accumulator
    become detections. Connect a CannyEdgeNode output to the input.

    Outputs:
    - *overlay* — RGB image with detected circles drawn in green
    - *table* — columns `cx`, `cy`, `radius` for every detected circle

    **Min Radius** — smallest circle radius to search for (pixels).
    **Max Radius** — largest circle radius to search for (pixels).
    **Threshold** — fraction of the peak accumulator value required for a detection.

    Keywords: hough circle, circle detection, round objects, radius fit, edge voting, 霍夫, 圓形偵測, 邊緣, 影像處理, 輪廓
    """
    __identifier__ = 'nodes.image_process.measure'
    NODE_NAME      = 'Hough Circles'
    PORT_SPEC      = {'inputs': ['mask'], 'outputs': ['image', 'table']}
    OUTPUT_COLUMNS = {'table': ['cx', 'cy', 'radius']}

    def __init__(self):
        super().__init__()
        self.add_input('mask',     color=PORT_COLORS['mask'])
        self.add_output('overlay', color=PORT_COLORS['image'], multi_output=True)
        self.add_output('table',   color=PORT_COLORS['table'], multi_output=True)
        self._add_int_spinbox('min_radius',  'Min Radius (px)',  value=10,  min_val=1,    max_val=1000)
        self._add_int_spinbox('max_radius',  'Max Radius (px)',  value=60,  min_val=1,    max_val=1000)
        self._add_int_spinbox('num_peaks',   'Max Circles',      value=10,  min_val=1,    max_val=1000)
        self._add_float_spinbox('threshold', 'Threshold (0–1)',  value=0.5, min_val=0.01, max_val=1.0, step=0.05, decimals=2)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.transform import hough_circle, hough_circle_peaks

        arr, _, _ = _get_image_arr(self, 'mask')
        if arr is None:
            return False, "No input connected"

        self.set_progress(10)
        binary  = _edge_to_binary(arr)
        min_r   = int(self.get_property('min_radius'))
        max_r   = int(self.get_property('max_radius'))
        n_peaks = int(self.get_property('num_peaks'))
        thresh  = float(self.get_property('threshold'))

        radii     = np.arange(min_r, max_r + 1)
        hough_res = hough_circle(binary, radii)
        self.set_progress(65)

        accums, cx, cy, rad = hough_circle_peaks(
            hough_res, radii,
            num_peaks=n_peaks,
            threshold=thresh * hough_res.max(),
        )
        self.set_progress(80)

        df = pd.DataFrame({'cx': cx.astype(float), 'cy': cy.astype(float),
                           'radius': rad.astype(float)})
        self.output_values['table'] = TableData(payload=df)

        rgb = _arr_to_rgb(arr)
        pil_tmp = Image.fromarray(rgb, 'RGB')
        draw = ImageDraw.Draw(pil_tmp)
        for x, y, r in zip(cx, cy, rad):
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(0, 220, 0), width=2)
        overlay_arr = np.array(pil_tmp)
        self._make_image_output(overlay_arr, 'overlay')
        self.set_display(overlay_arr)
        self.set_progress(100)
        return True, None


class HoughLinesNode(BaseImageProcessNode):
    """
    Detects straight lines in a Canny edge image using the Hough line transform.

    Each detected line is described by (`theta`, `rho`): the perpendicular angle and
    distance from the image origin. Lines are extended to the full image boundary for
    the overlay. Connect a CannyEdgeNode output to the input.

    Outputs:
    - *overlay* — RGB image with detected lines drawn in red
    - *table* — columns `theta` (rad), `rho` (px), and endpoint coordinates `x0`, `y0`, `x1`, `y1`

    **Threshold** — fraction of the peak accumulator value required for a detection.
    **Min Distance** — minimum pixel separation between detected lines.
    **Min Angle** — minimum angular separation in degrees between detected lines.

    Keywords: hough line, line detection, straight edges, rho theta, line fit, 霍夫, 直線偵測, 邊緣, 影像處理, 輪廓
    """
    __identifier__ = 'nodes.image_process.measure'
    NODE_NAME      = 'Hough Lines'
    PORT_SPEC      = {'inputs': ['mask'], 'outputs': ['image', 'table']}
    OUTPUT_COLUMNS = {'table': ['theta', 'rho', 'x0', 'y0', 'x1', 'y1']}

    def __init__(self):
        super().__init__()
        self.add_input('mask',     color=PORT_COLORS['mask'])
        self.add_output('overlay', color=PORT_COLORS['image'], multi_output=True)
        self.add_output('table',   color=PORT_COLORS['table'], multi_output=True)
        self._add_int_spinbox('num_peaks',    'Max Lines',           value=10,  min_val=1, max_val=500)
        self._add_float_spinbox('threshold',  'Threshold (0–1)',     value=0.5, min_val=0.01, max_val=1.0, step=0.05, decimals=2)
        self._add_int_spinbox('min_distance', 'Min Distance (px)',   value=9,   min_val=1, max_val=500)
        self._add_int_spinbox('min_angle',    'Min Angle (deg)',     value=10,  min_val=1, max_val=90)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.transform import hough_line, hough_line_peaks

        arr, _, _ = _get_image_arr(self, 'mask')
        if arr is None:
            return False, "No input connected"

        self.set_progress(10)
        binary   = _edge_to_binary(arr)
        H, W     = binary.shape
        n_peaks  = int(self.get_property('num_peaks'))
        thresh   = float(self.get_property('threshold'))
        min_dist = int(self.get_property('min_distance'))
        min_ang  = int(self.get_property('min_angle'))

        h, theta, d = hough_line(binary)
        self.set_progress(55)

        accums, angles, dists = hough_line_peaks(
            h, theta, d,
            num_peaks=n_peaks,
            threshold=thresh * h.max(),
            min_distance=min_dist,
            min_angle=min_ang,
        )
        self.set_progress(75)

        rows = []
        rgb  = _arr_to_rgb(arr)
        pil_tmp = Image.fromarray(rgb, 'RGB')
        draw = ImageDraw.Draw(pil_tmp)

        for angle, rho in zip(angles, dists):
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            if abs(sin_a) > 1e-6:
                # compute y at x=0 and x=W-1
                y0 = int(round((rho - 0     * cos_a) / sin_a))
                y1 = int(round((rho - (W-1) * cos_a) / sin_a))
                x0, x1 = 0, W - 1
            else:
                # nearly vertical line — compute x at y=0 and y=H-1
                x0 = int(round((rho - 0     * sin_a) / cos_a))
                x1 = int(round((rho - (H-1) * sin_a) / cos_a))
                y0, y1 = 0, H - 1

            rows.append({'theta': round(float(angle), 5), 'rho': round(float(rho), 2),
                         'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})
            draw.line([(x0, y0), (x1, y1)], fill=(220, 0, 0), width=2)

        overlay_arr = np.array(pil_tmp)
        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=['theta', 'rho', 'x0', 'y0', 'x1', 'y1'])
        self.output_values['table']   = TableData(payload=df)
        self._make_image_output(overlay_arr, 'overlay')
        self.set_display(overlay_arr)
        self.set_progress(100)
        return True, None


class HoughEllipseNode(BaseImageProcessNode):
    """
    Detects ellipses in a Canny edge image using the Hough ellipse transform.

    Slow on large images -- resize input to under 300x300 px for best speed. Uses
    `skimage.transform.hough_ellipse`.

    Outputs:
    - *overlay* — RGB image with detected ellipses drawn in cyan
    - *table* — columns `cx`, `cy`, `a` (semi-major), `b` (semi-minor), `orientation` (rad)

    **Min Semi-Major** — smallest semi-major axis to search for (pixels).
    **Max Semi-Major** — largest semi-major axis to search for (pixels).
    **Accuracy** — step size in pixels for the accumulator; larger = faster but coarser.
    **Threshold** — fraction of the peak accumulator value required for a detection.

    Keywords: hough ellipse, ellipse detection, oval fit, orientation, semi-major, 霍夫, 橢圓偵測, 輪廓, 影像處理, 方向
    """
    __identifier__ = 'nodes.image_process.measure'
    NODE_NAME      = 'Hough Ellipse'
    PORT_SPEC      = {'inputs': ['mask'], 'outputs': ['image', 'table']}
    OUTPUT_COLUMNS = {'table': ['cx', 'cy', 'a', 'b', 'orientation']}

    def __init__(self):
        super().__init__()
        self.add_input('mask',     color=PORT_COLORS['mask'])
        self.add_output('overlay', color=PORT_COLORS['image'], multi_output=True)
        self.add_output('table',   color=PORT_COLORS['table'], multi_output=True)
        self._add_int_spinbox('min_size',    'Min Semi-Major (px)', value=10,  min_val=1, max_val=500)
        self._add_int_spinbox('max_size',    'Max Semi-Major (px)', value=60,  min_val=1, max_val=500)
        self._add_int_spinbox('accuracy',    'Accuracy (px step)',  value=10,  min_val=1, max_val=50)
        self._add_int_spinbox('num_peaks',   'Max Ellipses',        value=5,   min_val=1, max_val=50)
        self._add_float_spinbox('threshold', 'Threshold (0–1)',     value=0.5, min_val=0.01, max_val=1.0, step=0.05, decimals=2)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()
        from skimage.transform import hough_ellipse

        arr, _, _ = _get_image_arr(self, 'mask')
        if arr is None:
            return False, "No input connected"

        self.set_progress(5)
        binary   = _edge_to_binary(arr).astype(np.uint8)
        min_size = int(self.get_property('min_size'))
        max_size = int(self.get_property('max_size'))
        accuracy = int(self.get_property('accuracy'))
        n_peaks  = int(self.get_property('num_peaks'))
        thresh   = float(self.get_property('threshold'))

        result = hough_ellipse(binary, accuracy=accuracy,
                               min_size=min_size, max_size=max_size)
        self.set_progress(80)

        if result.size == 0:
            return False, "No ellipses found"

        result.sort(order='accumulator')
        cutoff  = int(thresh * float(result['accumulator'].max()))
        top     = result[result['accumulator'] >= cutoff][-n_peaks:][::-1]

        rows = []
        rgb  = _arr_to_rgb(arr)
        pil_tmp = Image.fromarray(rgb, 'RGB')
        draw = ImageDraw.Draw(pil_tmp)

        for row in top:
            yc, xc  = float(row['yc']), float(row['xc'])
            a, b    = float(row['a']), float(row['b'])
            orient  = float(row['orientation'])
            rows.append({'cx': round(xc, 2), 'cy': round(yc, 2),
                         'a': round(a, 2), 'b': round(b, 2),
                         'orientation': round(orient, 4)})
            # Draw approximated rotated ellipse
            t   = np.linspace(0, 2 * np.pi, 180)
            cos_o, sin_o = np.cos(orient), np.sin(orient)
            xs  = xc + a * np.cos(t) * cos_o - b * np.sin(t) * sin_o
            ys  = yc + a * np.cos(t) * sin_o + b * np.sin(t) * cos_o
            pts = list(zip(xs.tolist(), ys.tolist()))
            if len(pts) >= 2:
                draw.line(pts + [pts[0]], fill=(0, 220, 220), width=2)

        overlay_arr = np.array(pil_tmp)
        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=['cx', 'cy', 'a', 'b', 'orientation'])
        self.output_values['table']   = TableData(payload=df)
        self._make_image_output(overlay_arr, 'overlay')
        self.set_display(overlay_arr)
        self.set_progress(100)
        return True, None


class ImageHistogramNode(BaseExecutionNode):
    """
    Plots the pixel intensity histogram of an image.

    Outputs a figure showing intensity distribution per channel (R/G/B for colour images,
    a single Intensity curve for grayscale). Optionally accepts a mask to restrict the
    histogram to the masked region only. Also outputs a table with columns `Pixel_Value`
    and one column per channel.

    **Bins** — number of histogram bins (default 256; auto-capped at max pixel value + 1).
    **Log Y-axis** — show frequency on a log scale.
    **Fill Alpha** — line / fill opacity (0.0-1.0).

    Keywords: histogram, intensity distribution, pixel histogram, channel plot, 直方圖, 強度分佈, 像素, 通道, 影像
    """
    __identifier__ = 'nodes.image_process.measure'
    NODE_NAME      = 'Image Histogram'
    PORT_SPEC      = {'inputs': ['image', 'mask'], 'outputs': ['figure', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_input('mask',  color=PORT_COLORS['mask'])
        self.add_output('figure', color=PORT_COLORS['figure'])
        self.add_output('table',  color=PORT_COLORS['table'])
        self._add_int_spinbox('bins', 'Bins', value=256, min_val=2, max_val=65536)
        self.add_checkbox('log_y',   '', text='Log Y-axis',  state=False)
        self._add_float_spinbox('alpha', 'Fill Alpha', value=0.3,
                                min_val=0.0, max_val=1.0, step=0.05, decimals=2)

    def evaluate(self):
        img_port = self.inputs().get('image')
        if not img_port or not img_port.connected_ports():
            self.mark_error()
            return False, "No image connected"

        cp  = img_port.connected_ports()[0]
        val = cp.node().output_values.get(cp.name())
        if not isinstance(val, ImageData):
            self.mark_error()
            return False, "Expected ImageData on 'image' port"

        img_np  = val.payload  # numpy array
        self.set_progress(10)

        # Optional mask
        bool_mask = None
        mask_port = self.inputs().get('mask')
        if mask_port and mask_port.connected_ports():
            mcp  = mask_port.connected_ports()[0]
            mval = mcp.node().output_values.get(mcp.name())
            if mval is not None:
                from data_models import MaskData
                m_np = mval.payload if isinstance(mval, MaskData) else np.asarray(mval)
                if m_np.ndim == 3:
                    m_np = _to_gray(m_np)
                if m_np.shape[:2] != img_np.shape[:2]:
                    from skimage.transform import resize
                    m_np = resize(m_np.astype(np.uint8),
                                  img_np.shape[:2], order=0,
                                  preserve_range=True).astype(np.uint8)
                bool_mask = m_np > 0

        bins  = max(2, int(self.get_property('bins') or 256))
        log_y = bool(self.get_property('log_y'))
        alpha = float(self.get_property('alpha') or 0.3)

        # Determine bit-depth range for x-axis display
        bit_depth = getattr(val, 'bit_depth', 8) or 8
        range_max = (1 << bit_depth)  # e.g. 256, 4096, 65536
        bins = min(bins, range_max)

        is_rgb = (img_np.ndim == 3 and img_np.shape[2] >= 3)
        self.set_progress(20)

        # Scale float [0,1] to bit-depth range for histogram
        scale = float(range_max - 1)  # e.g. 255, 4095

        # ── Build histogram data ────────────────────────────────────────────
        bin_edges  = np.linspace(0, range_max, bins + 1)
        pixel_vals = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # for plotting
        bin_lower  = bin_edges[:-1]
        bin_upper  = bin_edges[1:]
        hist_dict  = {'Bin_Lower': bin_lower, 'Bin_Upper': bin_upper}

        if is_rgb:
            ch_names  = ['Red', 'Green', 'Blue']
            ch_colors = ['#e74c3c', '#2ecc71', '#3498db']
            ch_data   = []
            for i, name in enumerate(ch_names):
                px = img_np[:, :, i][bool_mask] if bool_mask is not None \
                     else img_np[:, :, i].ravel()
                px_scaled = px.astype(np.float32) * scale
                h, _ = np.histogram(px_scaled, bins=bins, range=(0, range_max))
                hist_dict[name] = h
                ch_data.append((name, ch_colors[i], h))
        else:
            px = img_np[bool_mask] if bool_mask is not None else img_np.ravel()
            px_scaled = px.astype(np.float32) * scale
            h, _ = np.histogram(px_scaled, bins=bins, range=(0, range_max))
            hist_dict['Intensity'] = h
            ch_data = [('Intensity', '#95a5a6', h)]

        self.set_progress(60)

        # ── Plot ────────────────────────────────────────────────────────────
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig    = Figure(figsize=(6, 3.5), tight_layout=True)
        canvas = FigureCanvasAgg(fig)
        ax     = fig.add_subplot(111)

        for name, color, h in ch_data:
            if log_y:
                h_plot = np.where(h > 0, h, np.nan)
                ax.plot(pixel_vals, h_plot, color=color, linewidth=1.2, label=name)
                h_fill = np.maximum(h, 0.8)
                ax.fill_between(pixel_vals, h_fill, 0.8, alpha=alpha, color=color)
            else:
                ax.plot(pixel_vals, h, color=color, linewidth=1.2, label=name)
                ax.fill_between(pixel_vals, h, alpha=alpha, color=color)

        if log_y:
            ax.set_yscale('log')
            ax.set_ylim(bottom=0.8)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Count (log)' if log_y else 'Count')
        ax.set_title('Pixel Intensity Histogram'
                     + (' (masked)' if bool_mask is not None else ''))
        ax.set_xlim(0, range_max)
        if is_rgb:
            ax.legend(fontsize=8, framealpha=0.6)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

        self.output_values['figure'] = FigureData(payload=fig)
        self.output_values['table']  = TableData(payload=pd.DataFrame(hist_dict))
        self.set_progress(100)
        self.mark_clean()
        return True, None
