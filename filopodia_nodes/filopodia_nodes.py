"""
nodes/filopodia_nodes.py
========================
FiloQuant-style filopodia detection and measurement nodes.

Typical pipeline:

  ImageReadNode
       │ image
       ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  (optional) ROIMaskNode / BinaryThresholdNode → cell_mask input     │
  └─────────────────────────────────────────────────────────────────────┘
       │ image                      │ mask (cell body)
       ▼                            ▼
  CellEdgeMaskNode ─── mask ──►─────┬──► FilopodiaDetectNode ──► mask
                                    │                                │
                                    └────────────────────────────────┼──► FilopodiaAnalyzeNode
                                                   filopodia_mask ──►┘
                                                                      │
                                                          table ──────┤──► DataTableCellNode
                                                   visualization ─────┘──► ImageCellNode

References:
  Jacquemet et al. (2017) FiloQuant reveals increased filopodia density
  during breast cancer progression. J Cell Biol. doi:10.1083/jcb.201704045
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from scipy.ndimage import (
    binary_erosion   as _nd_erosion,
    binary_dilation  as _nd_dilation,
    binary_fill_holes,
    binary_closing   as _nd_closing,
    convolve         as _nd_convolve,
    label            as _nd_label,
    median_filter,
)

from PySide6 import QtWidgets, QtCore
from data_models import ImageData, MaskData, TableData
from nodes.base import BaseExecutionNode, PORT_COLORS
from nodes.base import BaseImageProcessNode
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget


class _ThresholdSliderWidget(NodeBaseWidget):
    """Compact slider + spinbox for threshold, adapts range to bit depth."""

    def __init__(self, parent=None, name='threshold', label='Threshold',
                 value=6, max_val=255):
        super().__init__(parent, name, label)
        self._max_val = max_val

        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setRange(0, max_val)
        self._slider.setValue(value)
        self._slider.setMinimumWidth(100)
        layout.addWidget(self._slider)

        self._spin = QtWidgets.QSpinBox()
        self._spin.setRange(0, max_val)
        self._spin.setValue(value)
        self._spin.setFixedWidth(65)
        self._spin.setStyleSheet(
            "QSpinBox { background: #222; color: #eee; border: 1px solid #444;"
            " padding: 2px; border-radius: 2px; }")
        layout.addWidget(self._spin)

        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)

        self.set_custom_widget(container)

    def _on_slider(self, val):
        self._spin.blockSignals(True)
        self._spin.setValue(val)
        self._spin.blockSignals(False)
        self.value_changed.emit(self.get_name(), val)

    def _on_spin(self, val):
        self._slider.blockSignals(True)
        self._slider.setValue(val)
        self._slider.blockSignals(False)
        self.value_changed.emit(self.get_name(), val)

    def set_range(self, max_val):
        """Update range when bit depth changes."""
        self._max_val = max_val
        self._slider.setRange(0, max_val)
        self._spin.setRange(0, max_val)

    def get_value(self):
        return self._spin.value()

    def set_value(self, value):
        self._slider.blockSignals(True)
        self._spin.blockSignals(True)
        try:
            v = int(value)
        except (ValueError, TypeError):
            v = 0
        self._slider.setValue(v)
        self._spin.setValue(v)
        self._slider.blockSignals(False)
        self._spin.blockSignals(False)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 5×5 center-surround sharpening kernel used by FiloQuant.
# Sum = 0 (DC-neutral); centre = +24 emphasises thin bright structures.
_SHARPEN_KERNEL = np.array([
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, 24, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
], dtype=np.float64)

# ImageJ's morphology (Erode/Dilate/Open/Close/Fill Holes) uses a 3×3 square
# (8-connectivity). scipy.ndimage defaults to a 3×3 cross (4-connectivity),
# so every morphology call here must pass this explicitly to match FiloQuant.
_SQUARE_3x3 = np.ones((3, 3), dtype=bool)

# FiloQuant's Process > Binary > Options has "Pad edges when eroding" checked,
# so pixels outside the image are treated as foreground during erosion (the
# cell isn't artificially eroded at the image border). scipy.ndimage defaults
# to border_value=0 (background outside) for erosion, which differs — pass
# these explicitly to match.
_PAD_ERODE  = {'border_value': 1}  # erode: outside = foreground
_PAD_DILATE = {'border_value': 0}  # dilate: outside = background (scipy default)

# 9×9 "bell" kernel from FiloQuant's macro — accentuates the cell boundary
# before skeletonising the contour. Sum is non-zero so the convolution is not
# DC-neutral; ImageJ then clips to [0, 255] and skeletonises any positive pixel.
_CONTOUR_KERNEL = np.array([
    [ 0,  0,  0, -1, -1, -1,  0,  0,  0],
    [ 0, -1, -1, -3, -3, -3, -1, -1,  0],
    [ 0, -1, -3, -3, -1, -3, -3, -1,  0],
    [-1, -3, -3,  6, 13,  6, -3, -3, -1],
    [-1, -3, -1, 13, 24, 13, -1, -3, -1],
    [-1, -3, -3,  6, 13,  6, -3, -3, -1],
    [ 0, -1, -3, -3, -1, -3, -3, -1,  0],
    [ 0, -1, -1, -3, -3, -3, -1, -1,  0],
    [ 0,  0,  0, -1, -1, -1,  0,  0,  0],
], dtype=np.float32)


def _filoquant_contour_length(cell_arr: np.ndarray) -> float:
    """Total cell-edge length in pixels, computed via FiloQuant's contour
    skeleton pipeline so the result is numerically comparable to the
    macro's ``contour.tif`` skeleton-pixel sum rather than a generic
    perimeter estimator.

    Pipeline mirror of FiloQuant macro's ``contour`` block:
        Close iter=4 → Erode iter=4 → Dilate iter=4
        → Convolve 9×9 bell → clip to [0, 255] uint8
        → Skeletonize (Zhang-Suen)
        → sum edge weights (orthogonal=1, diagonal=√2)
    """
    from skimage.morphology import skeletonize

    m = _nd_closing(cell_arr, iterations=4, structure=_SQUARE_3x3)
    m = _nd_erosion(m, iterations=4, structure=_SQUARE_3x3, **_PAD_ERODE)
    m = _nd_dilation(m, iterations=4, structure=_SQUARE_3x3, **_PAD_DILATE)

    convolved = _nd_convolve(
        (m.astype(np.uint8) * 255).astype(np.float32),
        _CONTOUR_KERNEL, mode='reflect',
    )
    convolved = np.clip(convolved, 0.0, 255.0).astype(np.uint8)
    skel = skeletonize(convolved > 0)

    if not skel.any():
        return 0.0
    ortho = (int(np.count_nonzero(skel[:, :-1] & skel[:, 1:]))
           + int(np.count_nonzero(skel[:-1, :] & skel[1:, :])))
    diag  = (int(np.count_nonzero(skel[:-1, :-1] & skel[1:, 1:]))
           + int(np.count_nonzero(skel[:-1, 1:]  & skel[1:, :-1])))
    return ortho + diag * 2 ** 0.5


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_port(node, port_name: str):
    """Return the data value from a single connected input port, or None."""
    port = node.inputs().get(port_name)
    if not port or not port.connected_ports():
        return None
    cp = port.connected_ports()[0]
    return cp.node().output_values.get(cp.name())


def _mask_to_bool(mask_data: MaskData) -> np.ndarray:
    """MaskData → boolean numpy array (True = foreground)."""
    arr = mask_data.payload
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr > (0.5 if arr.dtype in (np.float32, np.float64) else 127)


def _bool_to_mask(arr: np.ndarray, input_img: ImageData | None=None) -> MaskData:
    """Boolean numpy array → MaskData (white = foreground)."""
    u8 = arr.astype(np.uint8) * 255
    kwargs = {}
    if input_img is not None:
        kwargs = {f: getattr(input_img, f, None)
                    for f in type(input_img).model_fields if f != 'payload'}
    return MaskData(payload=u8, **kwargs)


def _skeleton_branch_stats(skel_binary: np.ndarray) -> list[dict]:
    """
    Analyse a skeletonized binary image and return one dict per connected
    branch: {length_px, y, x}.

    Length is computed with vectorized numpy edge counting so diagonal edges
    contribute √2 ≈ 1.4142 px and orthogonal edges contribute 1.0 px.
    This replaces the old per-pixel BFS and is orders of magnitude faster
    on large skeletons.

    The reported (y, x) is one endpoint (a pixel with exactly 1 neighbour);
    falls back to the first foreground pixel if the component has no endpoint
    (closed loop).
    """
    # Use 8-connectivity so diagonally-touching skeleton pixels belong to the
    # same branch component; default 4-connectivity splits them into separate
    # components and eliminates all diagonal edges → integer-only lengths.
    labeled, n = _nd_label(skel_binary, structure=np.ones((3, 3), dtype=np.int32))
    if n == 0:
        return []

    # Precompute neighbour counts once for the entire image (not per component).
    kernel_nb = np.ones((3, 3), dtype=np.int32)
    kernel_nb[1, 1] = 0
    nb_count     = _nd_convolve(skel_binary.astype(np.int32), kernel_nb, mode='constant')
    nb_count    *= skel_binary                        # zero out background
    endpoint_map = skel_binary & (nb_count == 1)     # True where pixel has exactly 1 neighbour

    results: list[dict] = []

    for i in range(1, n + 1):
        comp = (labeled == i)
        ys, xs = np.where(comp)
        if len(ys) == 0:
            continue

        # --- Vectorized edge counting (no per-pixel Python loop) ----------
        # Orthogonal edges: right-neighbour + down-neighbour
        ortho = (int(np.count_nonzero(comp[:, :-1] & comp[:, 1:]))
               + int(np.count_nonzero(comp[:-1, :] & comp[1:, :])))
        # Diagonal edges: down-right + down-left
        diag  = (int(np.count_nonzero(comp[:-1, :-1] & comp[1:, 1:]))
               + int(np.count_nonzero(comp[:-1, 1:]  & comp[1:, :-1])))

        # Edge count = n_connections, not n_pixels. A single isolated pixel has
        # 0 edges → length 0, but it physically spans 1 pixel, so clamp to 1.0.
        total_len = ortho + diag * 2 ** 0.5
        if total_len == 0:
            continue

        # Representative endpoint (precomputed above -- no extra convolve)
        ep_ys, ep_xs = np.where(endpoint_map & comp)
        sy = int(ep_ys[0]) if len(ep_ys) else int(ys[0])
        sx = int(ep_xs[0]) if len(ep_ys) else int(xs[0])

        results.append({'length_px': total_len, 'y': sy, 'x': sx})

    return results


# ---------------------------------------------------------------------------
# Node 1 -- CellEdgeMaskNode
# ---------------------------------------------------------------------------

class CellEdgeMaskNode(BaseImageProcessNode):
    """
    Generates a binary cell-body mask from a fluorescence image.

    Step 1 of the FiloQuant pipeline. Converts the input to grayscale,
    applies a lower-bound intensity threshold, optionally fills interior
    holes, then smooths the mask with morphological opening
    (**n_open** erosions followed by **n_open** dilations). Extra
    erode+dilate cycles can be added to further refine the boundary.

    Parameters:
    - **threshold** -- intensity cutoff (0--255)
    - **n_open** -- number of opening iterations for smoothing
    - **n_erode_dilate** -- additional erode+dilate cycles
    - **fill_holes** -- fill interior holes before opening

    Output port `mask` is a MaskData (white = cell body).

    Keywords: cell edge, cell mask, filopodia prep, segmentation, threshold cell body, 細胞邊緣, 遮罩, 分割, 絲足, 閾值
    """
    __identifier__ = 'plugins.Filopodia'
    NODE_NAME      = 'Cell Edge Mask'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['mask']}

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        import NodeGraphQt
        H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value
        self._thresh_slider = _ThresholdSliderWidget(
            self.view, name='threshold', label='Threshold', value=6, max_val=255)
        self.add_custom_widget(self._thresh_slider, widget_type=H, tab='Properties')
        self._add_int_spinbox('n_open',         'Open Iterations',      value=5,  min_val=0,  max_val=100)
        self._add_int_spinbox('n_erode_dilate', 'Extra Erode+Dilate',   value=0,  min_val=0,  max_val=80)
        self.add_checkbox('fill_holes', '', text='Fill Interior Holes', state=True)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        data = _read_port(self, 'image')
        if not isinstance(data, ImageData):
            self.mark_error()
            return False, 'No image connected'

        arr = data.payload
        gray = arr if arr.ndim == 2 else arr[..., :3].mean(axis=2)
        gray = gray.astype(np.float64)
        bit_depth = getattr(data, 'bit_depth', 8) or 8
        max_val = float((1 << bit_depth) - 1)
        self._thresh_slider.set_range(int(max_val))
        self.set_progress(10)

        # Convert threshold from bit-depth scale to [0,1]
        thresh_raw = float(self.get_property('threshold'))
        thresh = thresh_raw / max_val

        binary = gray >= thresh
        self.set_progress(25)

        if bool(self.get_property('fill_holes')):
            # scipy's binary_fill_holes structure controls BACKGROUND flood-fill
            # connectivity. ImageJ's "Fill Holes" treats background as 4-conn
            # (does not pass through diagonal gaps), so leave the default 3×3
            # cross here rather than the 8-conn square used for foreground ops.
            binary = binary_fill_holes(binary)
        self.set_progress(45)

        n_open = int(self.get_property('n_open'))
        if n_open > 0:
            binary = _nd_erosion(binary, iterations=n_open,
                                 structure=_SQUARE_3x3, **_PAD_ERODE)
            binary = _nd_dilation(binary, iterations=n_open,
                                  structure=_SQUARE_3x3, **_PAD_DILATE)
        self.set_progress(65)

        n_ed = int(self.get_property('n_erode_dilate'))
        if n_ed > 0:
            binary = _nd_erosion(binary, iterations=n_ed,
                                 structure=_SQUARE_3x3, **_PAD_ERODE)
            binary = _nd_dilation(binary, iterations=n_ed,
                                  structure=_SQUARE_3x3, **_PAD_DILATE)
        self.set_progress(85)

        mask_data = _bool_to_mask(binary, data)
        self.output_values['mask'] = mask_data

        # Preview: full brightness inside mask, dimmed outside
        preview_arr = np.where(binary, gray, gray * 0.15).astype(np.float32)
        self.set_display(preview_arr)
        self.set_progress(100)
        self.mark_clean()
        return True, None


# ---------------------------------------------------------------------------
# Node 2 -- FilopodiaDetectNode
# ---------------------------------------------------------------------------

class FilopodiaDetectNode(BaseImageProcessNode):
    """
    Detects filopodia candidates as a binary mask.

    Step 2 of the FiloQuant pipeline. Optionally applies CLAHE for local
    contrast enhancement and a 5x5 centre-surround sharpening convolution
    to accentuate thin bright structures, followed by a 3x3 median
    despeckle (x2) and intensity thresholding. Small isolated blobs
    (<8 px) and near-circular blobs (circularity > 0.80) are discarded —
    matches FiloQuant's
    ``Analyze Particles size=8-Infinity circularity=0.00-0.80``.
    If a `cell_mask` is connected and **n_distance_from_edge** > 0, only
    candidates within that band of the cell edge are kept (matches
    FiloQuant's "Maximum distance from cell edges" semantics).

    Parameters:
    - **threshold** -- intensity cutoff (0--255)
    - **n_distance_from_edge** -- maximum band width (px) from cell edge to
      keep candidates within. 0 = disabled (keep all).
    - **use_convolve** -- enable 5x5 sharpening kernel
    - **use_clahe** -- enable CLAHE local contrast pre-enhancement

    Output port `mask` is a MaskData of filopodia candidate regions.
    Connect to FilopodiaAnalyzeNode together with the `cell_mask`.

    Keywords: filopodia detect, protrusion mask, clahe, sharpen, median filter, 絲足, 細胞邊緣, 偵測, 遮罩, 分析
    """
    __identifier__ = 'plugins.Filopodia'
    NODE_NAME      = 'Filopodia Detect'
    PORT_SPEC      = {'inputs': ['image', 'mask'], 'outputs': ['mask']}

    def __init__(self):
        super().__init__()
        self.add_input('image',     color=PORT_COLORS['image'])
        self.add_input('cell_mask', color=PORT_COLORS['mask'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        import NodeGraphQt
        H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value
        self._thresh_slider = _ThresholdSliderWidget(
            self.view, name='threshold', label='Threshold', value=25, max_val=255)
        self.add_custom_widget(self._thresh_slider, widget_type=H, tab='Properties')
        self._add_int_spinbox('n_distance_from_edge','Distance from Edge (px)', value=0,  min_val=0,   max_val=200)
        self.add_checkbox('use_convolve', '', text='Sharpen (5×5 kernel)', state=True)
        self.add_checkbox('use_clahe',    '', text='CLAHE pre-enhance',    state=False)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        data = _read_port(self, 'image')
        if not isinstance(data, ImageData):
            self.mark_error()
            return False, 'No image connected'

        arr = data.payload
        gray = arr if arr.ndim == 2 else arr[..., :3].mean(axis=2)
        gray = gray.astype(np.float64)
        bit_depth = getattr(data, 'bit_depth', 8) or 8
        max_val = float((1 << bit_depth) - 1)
        self._thresh_slider.set_range(int(max_val))
        self.set_progress(10)

        if bool(self.get_property('use_clahe')):
            from skimage.exposure import equalize_adapthist
            # equalize_adapthist expects [0,1] float -- data already in that range
            gray = equalize_adapthist(gray, clip_limit=0.03).astype(np.float64)
        self.set_progress(25)

        if bool(self.get_property('use_convolve')):
            # ImageJ's Convolve uses edge replication ('nearest' in scipy) and
            # clips the result back to [0, 255] on an 8-bit image. Mirror both:
            # mode='nearest' for the boundary, np.clip to [0, 1] for the range.
            gray = _nd_convolve(gray, _SHARPEN_KERNEL, mode='nearest')
            gray = np.clip(gray, 0.0, 1.0)
        self.set_progress(40)

        # Despeckle × 2 (3×3 median, matches FiloQuant's "Despeckle" macro
        # call). ImageJ replicates edge pixels for the median; scipy defaults
        # to 'reflect' which differs subtly at the boundary. mode='nearest'
        # is scipy's edge-replication equivalent.
        gray = median_filter(gray, size=3, mode='nearest')
        gray = median_filter(gray, size=3, mode='nearest')
        self.set_progress(55)

        # Convert threshold from bit-depth scale to [0,1]
        bit_depth = getattr(data, 'bit_depth', 8) or 8
        thresh_raw = float(self.get_property('threshold'))
        max_val = float((1 << bit_depth) - 1)
        binary = gray >= (thresh_raw / max_val)
        self.set_progress(65)

        # Discard tiny blobs (FiloQuant uses size >= 8 px in Analyze Particles
        # with 8-connectivity). max_size=7 removes objects with area <= 7,
        # keeping objects >= 8 px.
        from skimage.morphology import remove_small_objects
        from skimage.measure import (
            regionprops, label as _sk_label, perimeter_crofton,
        )
        binary = remove_small_objects(binary, max_size=7, connectivity=2)
        self.set_progress(72)

        # Circularity filter: FiloQuant's Analyze Particles uses
        # circularity=0.00-0.80, dropping near-circular blobs (which tend to
        # be cell-body remnants rather than thin protrusions).
        # circularity = 4π·area / perimeter² ; > 0.80 → discard.
        #
        # IMPORTANT: skimage's default ``regionprops.perimeter`` systematically
        # underestimates the perimeter of small blobs (it can even yield
        # circularity > 1, which is geometrically impossible). ImageJ's Analyze
        # Particles uses a Crofton-style perimeter that matches small blobs
        # accurately. Use ``perimeter_crofton`` to stay numerically faithful
        # to ImageJ — otherwise ~50% of the blobs Fiji keeps get dropped.
        if binary.any():
            lbl = _sk_label(binary, connectivity=2)
            keep_mask = np.zeros_like(binary)
            for rp in regionprops(lbl):
                p = perimeter_crofton(rp.image, directions=4)
                if p <= 0:
                    # Single-pixel / degenerate region — circularity undefined;
                    # ImageJ's Analyze Particles drops these too.
                    continue
                circ = 4.0 * np.pi * rp.area / (p * p)
                if circ <= 0.80:
                    keep_mask[lbl == rp.label] = True
            binary = keep_mask
        self.set_progress(82)

        # Match FiloQuant: when n_distance_from_edge > 0, keep only candidates
        # within a band of that width OUTSIDE the cell edge. The cell body
        # itself is also kept here — FilopodiaAnalyze subtracts it later.
        cell_md = _read_port(self, 'cell_mask')
        n_dist = int(self.get_property('n_distance_from_edge'))
        if isinstance(cell_md, MaskData) and n_dist > 0:
            zone = _nd_dilation(_mask_to_bool(cell_md),
                                iterations=n_dist, structure=_SQUARE_3x3)
            binary = binary & zone
        self.set_progress(90)

        mask_data = _bool_to_mask(binary, data)
        self.output_values['mask'] = mask_data

        # Preview: dim the filopodia candidates that fall inside the cell body
        # (those get subtracted by FilopodiaAnalyze anyway) so the user sees
        # mainly the "real" protrusion candidates outside the cell. Mirrors the
        # CellEdgeMaskNode "bright inside / dimmed outside" preview convention.
        if isinstance(cell_md, MaskData):
            cell_bool = _mask_to_bool(cell_md)
            display = np.zeros(binary.shape, dtype=np.float32)
            display[binary & ~cell_bool] = 1.0    # outside cell  → bright
            display[binary & cell_bool]  = 0.15   # inside cell   → dimmed
        else:
            display = binary.astype(np.float32)
        self.set_display(display)
        self.set_progress(100)
        self.mark_clean()
        return True, None


# ---------------------------------------------------------------------------
# Node 3 -- FilopodiaAnalyzeNode
# ---------------------------------------------------------------------------

class FilopodiaAnalyzeNode(BaseImageProcessNode):
    """
    Skeletonizes the filopodia mask and measures each branch.

    Step 3 (final step) of the FiloQuant pipeline.

    Processing steps:
    - Subtract the cell body (`cell_mask`) from the filopodia candidate mask to isolate protrusions only
    - Remove objects smaller than **min_size_px**
    - Optionally close small gaps with **repair_cycles** morphological close iterations (FiloQuant's "Filopodia repair")
    - Skeletonize via `skimage.morphology.skeletonize`
    - Measure each connected skeleton branch with diagonal-aware edge counting
    - Measure total cell edge length via FiloQuant's contour skeleton pipeline
      (close 4 → erode 4 → dilate 4 → 9×9 bell convolve → skeletonize → sum
      diagonal-aware edge weights), so `edge_length_px` is numerically
      comparable to the FiloQuant macro's contour.tif skeleton-pixel count.

    Outputs:
    - `table` -- TableData with columns: `x`, `y`, `filopodia_length_px`,
      `edge_length_px`, `filopodia_length_um`, `edge_length_um` (one row per
      detected filopodium skeleton branch).  The `_um` columns are populated
      when the input mask carries ``scale_um`` (micrometers per pixel,
      typically set by image readers from microscope metadata); otherwise
      they're NaN.
    - `visualization` -- colour composite (dark background, green = cell body, cyan = isolated filopodia mask, magenta = skeleton)

    Keywords: filopodia analysis, skeleton length, branch measurement, protrusion quantification, edge length, micrometers, 絲足, 分析, 骨架, 細胞邊緣, 量測
    """
    __identifier__ = 'plugins.Filopodia'
    NODE_NAME      = 'Filopodia Analyze'
    PORT_SPEC      = {'inputs': ['mask', 'mask'], 'outputs': ['table', 'image']}
    OUTPUT_COLUMNS = {
        'table': ['x', 'y', 'filopodia_length_px', 'edge_length_px',
                  'filopodia_length_um', 'edge_length_um']
    }

    def __init__(self):
        super().__init__()
        self.add_input('filopodia_mask', color=PORT_COLORS['mask'])
        self.add_input('cell_mask',      color=PORT_COLORS['mask'])
        self.add_output('table',         color=PORT_COLORS['table'])
        self.add_output('visualization', color=PORT_COLORS['image'])
        self._add_int_spinbox('min_size_px',   'Min Size (px)',       value=3, min_val=1, max_val=100000)
        self._add_int_spinbox('repair_cycles', 'Repair Cycles (close)', value=2, min_val=0, max_val=30)
        self.create_preview_widgets()
        self.set_property('live_preview', True)

    def evaluate(self):
        self.reset_progress()

        rs_skeleton_branch_stats = None
        try:
            import image_process_rs as _image_process_rs
            rs_skeleton_branch_stats = getattr(_image_process_rs, "skeleton_branch_stats", None)
            # print('Rust backend available.')
        except Exception:
            rs_skeleton_branch_stats = None
        
        filo_md = _read_port(self, 'filopodia_mask')
        cell_md = _read_port(self, 'cell_mask')

        if not isinstance(filo_md, MaskData):
            self.mark_error()
            return False, 'filopodia_mask not connected'
        if not isinstance(cell_md, MaskData):
            self.mark_error()
            return False, 'cell_mask not connected'

        filo_arr = _mask_to_bool(filo_md)
        cell_arr = _mask_to_bool(cell_md)
        self.set_progress(10)

        # Remove cell body from filopodia candidates
        isolated = filo_arr & ~cell_arr
        self.set_progress(20)

        from skimage.morphology import remove_small_objects, skeletonize

        min_size = max(1, int(self.get_property('min_size_px')))
        # FiloQuant: Analyze Particles size=min_size-Infinity keeps area >=
        # min_size. skimage.remove_small_objects(max_size=N) removes area <= N
        # (keeps > N), so to keep area >= min_size we pass max_size=min_size-1.
        # 8-connectivity to match ImageJ.
        isolated = remove_small_objects(isolated, max_size=min_size - 1,
                                        connectivity=2)
        self.set_progress(35)

        repair = int(self.get_property('repair_cycles'))
        if repair > 0:
            isolated = _nd_closing(isolated, iterations=repair,
                                   structure=_SQUARE_3x3)
        self.set_progress(50)

        # Skeletonize
        skeleton = skeletonize(isolated)
        self.set_progress(65)

        # Measure each branch
        branches = None
        if rs_skeleton_branch_stats is not None:
            try:
                # Rust binding expects a contiguous uint8 numpy array.
                skeleton_u8 = np.ascontiguousarray(skeleton.astype(np.uint8))
                rs_rows = rs_skeleton_branch_stats(skeleton_u8, include_singletons=False)
                branches = [
                    {'length_px': float(length), 'y': int(y), 'x': int(x)}
                    for length, y, x in rs_rows
                ]
            except Exception:
                branches = None
        if branches is None:
            branches = _skeleton_branch_stats(skeleton)
        self.set_progress(80)

        # Total cell edge length via FiloQuant's contour skeleton pipeline
        # (matches the macro's contour.tif output rather than a generic
        # perimeter estimator).
        edge_length_px = _filoquant_contour_length(cell_arr)

        # Pull microns-per-pixel from input metadata (filopodia mask first;
        # fall back to cell mask).  None / 0 / negative → no µm conversion.
        scale_um = getattr(filo_md, 'scale_um', None)
        if not scale_um or scale_um <= 0:
            scale_um = getattr(cell_md, 'scale_um', None)
        if not scale_um or scale_um <= 0:
            scale_um = None

        edge_length_um = (round(edge_length_px * scale_um, 2)
                           if scale_um is not None else float('nan'))

        # Build results table
        rows = [
            {
                'x':                   b['x'],
                'y':                   b['y'],
                'filopodia_length_px': round(b['length_px'], 2),
                'edge_length_px':      round(edge_length_px, 2),
                'filopodia_length_um': (round(b['length_px'] * scale_um, 2)
                                         if scale_um is not None
                                         else float('nan')),
                'edge_length_um':      edge_length_um,
            }
            for b in branches
        ]
        df = (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(columns=['x', 'y', 'filopodia_length_px',
                                        'edge_length_px',
                                        'filopodia_length_um',
                                        'edge_length_um'])
        )
        self.output_values['table'] = TableData(payload=df)
        self.set_progress(90)

        # Build colour composite visualization (synthetic, always 8-bit)
        vis = self._make_visualization(cell_arr, isolated, skeleton)
        self.output_values['visualization'] = ImageData(payload=vis, bit_depth=8)
        self.set_display(vis)
        self.set_progress(100)
        self.mark_clean()
        return True, None

    # ------------------------------------------------------------------
    def _make_visualization(
        self,
        cell_arr:  np.ndarray,   # boolean -- cell body
        filo_arr:  np.ndarray,   # boolean -- isolated filopodia regions
        skeleton:  np.ndarray,   # boolean -- thinned filopodia skeleton
    ) -> np.ndarray:
        """
        Colour composite on a dark background (float32 [0, 1]):
          • Dark green  -- cell body
          • Cyan        -- isolated filopodia mask (before skeletonization)
          • Magenta     -- skeleton (1-pixel centerlines)
        """
        H, W = cell_arr.shape
        rgb = np.zeros((H, W, 3), dtype=np.float32)

        # Cell body: dark green fill so the context is visible but not distracting
        rgb[cell_arr, 1] = 60 / 255.0
        rgb[cell_arr, 2] = 60 / 255.0

        # Filopodia mask: dim cyan
        rgb[filo_arr, 1] = 100 / 255.0
        rgb[filo_arr, 2] = 100 / 255.0

        # Skeleton: bright magenta on top of everything
        rgb[skeleton, 0] = 1.0
        rgb[skeleton, 2] = 1.0

        return rgb
