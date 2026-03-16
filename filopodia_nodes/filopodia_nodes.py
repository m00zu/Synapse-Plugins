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

from data_models import ImageData, MaskData, TableData
from nodes.base import BaseExecutionNode, PORT_COLORS
from nodes.base import BaseImageProcessNode, _arr_to_pil


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
    return np.array(mask_data.payload.convert('L')) > 127


def _bool_to_mask(arr: np.ndarray) -> MaskData:
    """Boolean numpy array → MaskData (white = foreground)."""
    u8 = arr.astype(np.uint8) * 255
    return MaskData(payload=_arr_to_pil(u8, 'L'))


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

        # Representative endpoint (precomputed above — no extra convolve)
        ep_ys, ep_xs = np.where(endpoint_map & comp)
        sy = int(ep_ys[0]) if len(ep_ys) else int(ys[0])
        sx = int(ep_xs[0]) if len(ep_ys) else int(xs[0])

        results.append({'length_px': total_len, 'y': sy, 'x': sx})

    return results


# ---------------------------------------------------------------------------
# Node 1 — CellEdgeMaskNode
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
    - **threshold** — intensity cutoff (0--255)
    - **n_open** — number of opening iterations for smoothing
    - **n_erode_dilate** — additional erode+dilate cycles
    - **fill_holes** — fill interior holes before opening

    Output port `mask` is a MaskData (white = cell body).

    Keywords: cell edge, cell mask, filopodia prep, segmentation, threshold cell body, 細胞邊緣, 遮罩, 分割, 絲足, 閾值
    """
    __identifier__ = 'plugins.Plugins.filopodia'
    NODE_NAME      = 'Cell Edge Mask'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['mask']}

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        self._add_int_spinbox('threshold',      'Threshold (0–255)',    value=6,  min_val=0,  max_val=255)
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

        gray = np.array(data.payload.convert('L'))
        self.set_progress(10)

        binary = gray >= int(self.get_property('threshold'))
        self.set_progress(25)

        if bool(self.get_property('fill_holes')):
            binary = binary_fill_holes(binary)
        self.set_progress(45)

        n_open = int(self.get_property('n_open'))
        if n_open > 0:
            binary = _nd_erosion(binary,  iterations=n_open)
            binary = _nd_dilation(binary, iterations=n_open)
        self.set_progress(65)

        n_ed = int(self.get_property('n_erode_dilate'))
        if n_ed > 0:
            binary = _nd_erosion(binary,  iterations=n_ed)
            binary = _nd_dilation(binary, iterations=n_ed)
        self.set_progress(85)

        mask_data = _bool_to_mask(binary)
        self.output_values['mask'] = mask_data

        # Preview: original image at full brightness inside mask,
        # dimmed to 15 % outside so the cell boundary is immediately visible.
        preview_arr = np.where(binary, gray, (gray * 0.15).astype(np.uint8)).astype(np.uint8)
        self.set_display(_arr_to_pil(preview_arr, 'L'))
        self.set_progress(100)
        self.mark_clean()
        return True, None


# ---------------------------------------------------------------------------
# Node 2 — FilopodiaDetectNode
# ---------------------------------------------------------------------------

class FilopodiaDetectNode(BaseImageProcessNode):
    """
    Detects filopodia candidates as a binary mask.

    Step 2 of the FiloQuant pipeline. Optionally applies CLAHE for local
    contrast enhancement and a 5x5 centre-surround sharpening convolution
    to accentuate thin bright structures, followed by a 3x3 median
    despeckle (x2) and intensity thresholding. Small isolated blobs
    (<8 px) are discarded. If a `cell_mask` is connected, an exclusion
    zone is dilated around the cell body so candidates too close to the
    cell edge are removed.

    Parameters:
    - **threshold** — intensity cutoff (0--255)
    - **n_distance_from_edge** — exclusion zone width in pixels around the cell body
    - **use_convolve** — enable 5x5 sharpening kernel
    - **use_clahe** — enable CLAHE local contrast pre-enhancement

    Output port `mask` is a MaskData of filopodia candidate regions.
    Connect to FilopodiaAnalyzeNode together with the `cell_mask`.

    Keywords: filopodia detect, protrusion mask, clahe, sharpen, median filter, 絲足, 細胞邊緣, 偵測, 遮罩, 分析
    """
    __identifier__ = 'plugins.Plugins.filopodia'
    NODE_NAME      = 'Filopodia Detect'
    PORT_SPEC      = {'inputs': ['image', 'mask'], 'outputs': ['mask']}

    def __init__(self):
        super().__init__()
        self.add_input('image',     color=PORT_COLORS['image'])
        self.add_input('cell_mask', color=PORT_COLORS['mask'])
        self.add_output('mask', color=PORT_COLORS['mask'])
        self._add_int_spinbox('threshold',           'Threshold (0–255)',       value=25, min_val=0,   max_val=255)
        self._add_int_spinbox('n_distance_from_edge','Distance from Edge (px)', value=1,  min_val=0,   max_val=200)
        self.add_checkbox('use_convolve', '', text='Sharpen (5×5 kernel)', state=True)
        self.add_checkbox('use_clahe',    '', text='CLAHE pre-enhance',    state=False)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        data = _read_port(self, 'image')
        if not isinstance(data, ImageData):
            self.mark_error()
            return False, 'No image connected'

        gray = np.array(data.payload.convert('L')).astype(np.float64)
        self.set_progress(10)

        if bool(self.get_property('use_clahe')):
            from skimage.exposure import equalize_adapthist
            gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
            # clip_limit=0.03 matches FiloQuant's slope=3 / 256 bins default
            gray = equalize_adapthist(gray_u8, clip_limit=0.03) * 255.0
        self.set_progress(25)

        if bool(self.get_property('use_convolve')):
            gray = _nd_convolve(gray, _SHARPEN_KERNEL, mode='reflect')
        self.set_progress(40)

        # Despeckle × 2 (3×3 median, matches FiloQuant's "Despeckle" macro call)
        gray = median_filter(gray, size=3)
        gray = median_filter(gray, size=3)
        self.set_progress(55)

        binary = gray >= int(self.get_property('threshold'))
        self.set_progress(65)

        # Discard tiny blobs (FiloQuant uses size >= 8 px in Analyze Particles)
        # max_size=7 removes objects with area <= 7, keeping objects >= 8 px
        from skimage.morphology import remove_small_objects
        binary = remove_small_objects(binary, max_size=7)
        self.set_progress(75)

        # Optional: exclude candidates within n px of the cell body edge
        cell_md = _read_port(self, 'cell_mask')
        n_dist = int(self.get_property('n_distance_from_edge'))
        if isinstance(cell_md, MaskData) and n_dist > 0:
            zone = _nd_dilation(_mask_to_bool(cell_md), iterations=n_dist)
            binary = binary & ~zone
        self.set_progress(90)

        mask_data = _bool_to_mask(binary)
        self.output_values['mask'] = mask_data
        self.set_display(mask_data.payload)
        self.set_progress(100)
        self.mark_clean()
        return True, None


# ---------------------------------------------------------------------------
# Node 3 — FilopodiaAnalyzeNode
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
    - Measure total cell edge length via `skimage.measure.perimeter`

    Outputs:
    - `table` — TableData with columns: `x`, `y`, `filopodia_length_px`, `edge_length_px` (one row per detected filopodium skeleton branch)
    - `visualization` — colour composite (dark background, green = cell body, cyan = isolated filopodia mask, magenta = skeleton)

    Keywords: filopodia analysis, skeleton length, branch measurement, protrusion quantification, edge length, 絲足, 分析, 骨架, 細胞邊緣, 量測
    """
    __identifier__ = 'plugins.Plugins.filopodia'
    NODE_NAME      = 'Filopodia Analyze'
    PORT_SPEC      = {'inputs': ['mask', 'mask'], 'outputs': ['table', 'image']}
    OUTPUT_COLUMNS = {
        'table': ['x', 'y', 'filopodia_length_px', 'edge_length_px']
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

        min_size = int(self.get_property('min_size_px'))
        isolated = remove_small_objects(isolated, max_size=max(1, min_size))
        self.set_progress(35)

        repair = int(self.get_property('repair_cycles'))
        if repair > 0:
            isolated = _nd_closing(isolated, iterations=repair)
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

        # Total cell edge perimeter
        from skimage.measure import perimeter as _perimeter
        edge_length_px = float(_perimeter(cell_arr, neighborhood=8))

        # Build results table
        rows = [
            {
                'x':                   b['x'],
                'y':                   b['y'],
                'filopodia_length_px': round(b['length_px'], 2),
                'edge_length_px':      round(edge_length_px, 2),
            }
            for b in branches
        ]
        df = (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(columns=['x', 'y', 'filopodia_length_px', 'edge_length_px'])
        )
        self.output_values['table'] = TableData(payload=df)
        self.set_progress(90)

        # Build colour composite visualization
        vis = self._make_visualization(cell_arr, isolated, skeleton)
        self.output_values['visualization'] = ImageData(payload=vis)
        self.set_display(vis)
        self.set_progress(100)
        self.mark_clean()
        return True, None

    # ------------------------------------------------------------------
    def _make_visualization(
        self,
        cell_arr:  np.ndarray,   # boolean — cell body
        filo_arr:  np.ndarray,   # boolean — isolated filopodia regions
        skeleton:  np.ndarray,   # boolean — thinned filopodia skeleton
    ) -> Image.Image:
        """
        Colour composite on a dark background:
          • Dark green  — cell body
          • Cyan        — isolated filopodia mask (before skeletonization)
          • Magenta     — skeleton (1-pixel centerlines)
        """
        H, W = cell_arr.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)

        # Cell body: dark green fill so the context is visible but not distracting
        rgb[cell_arr, 0] = 0
        rgb[cell_arr, 1] = 60
        rgb[cell_arr, 2] = 60

        # Filopodia mask: dim cyan
        rgb[filo_arr, 0] = 0
        rgb[filo_arr, 1] = 100
        rgb[filo_arr, 2] = 100

        # Skeleton: bright magenta on top of everything
        rgb[skeleton, 0] = 255
        rgb[skeleton, 1] = 0
        rgb[skeleton, 2] = 255

        return _arr_to_pil(rgb, 'RGB')
