"""
video_analyze_node.py — SAM2 Video Analyze: integrated video analysis workflow.

Replaces the 6-node chain (FolderIterator → ImageReader → SAM2 Segment →
SAM2 Track → ParticleProps → BatchAccumulator) with a single node.

Features:
  - Folder-based image loading with frame slider
  - Interactive SAM2 annotation (click/auto) on any frame (multi-keyframe)
  - Optional GroundingDINO text-prompted detection
  - Keyframe-segmented centroid tracking across all frames
  - Per-frame correction: add/remove points post-analysis to fix mistakes
  - regionprops measurement with frame + filename columns
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import warnings
from pathlib import Path

# Suppress NumPy 2.0 __array_wrap__ deprecation from skimage/numpy internals
warnings.filterwarnings('ignore', message='__array_wrap__.*', category=DeprecationWarning)
# Suppress onnxruntime shape merge warnings
warnings.filterwarnings('ignore', message='.*MergeShapeInfo.*')

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget

from data_models import ImageData, TableData
from nodes.base import PORT_COLORS
from nodes.base import BaseImageProcessNode

from .engine import SAM2ImageSession
from .model_manager import SAM2ModelManager
from .tracking import CentroidTrackingStrategy, SAM2FrameTracker
from .viewer import SAM2ClickGraphicsView, _obj_color

logger = logging.getLogger(__name__)

__all__ = ['SAM2VideoAnalyzeNode']

_model_manager = SAM2ModelManager()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nat_key(p: Path):
    """Natural sort key for filenames."""
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r'(\d+)', p.name)]


def _load_image_array(path: str) -> np.ndarray:
    """Load an image file to RGB uint8 array. PIL-first with tifffile fallback."""
    if path.lower().endswith(('.tif', '.tiff')):
        try:
            pil_img = Image.open(path)
            arr = np.asarray(pil_img)
        except Exception:
            import tifffile
            arr = tifffile.imread(path)
        # Normalize 16-bit to 8-bit
        if arr.dtype in (np.uint16, np.int16, np.float32, np.float64):
            if arr.dtype in (np.float32, np.float64):
                arr = arr.astype(np.float64)
                mn, mx = arr.min(), arr.max()
                if mx > mn:
                    arr = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
            else:
                arr = arr.astype(np.float64)
                mn, mx = arr.min(), arr.max()
                if mx > mn:
                    arr = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
        # Handle single channel → RGB
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.concatenate([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr.astype(np.uint8)
    else:
        pil_img = Image.open(path).convert('RGB')
        return np.asarray(pil_img, dtype=np.uint8)


# Simple frame cache (keyed by path string)
_frame_cache: dict[str, np.ndarray] = {}
_CACHE_MAX = 20


def _cached_load(path: str) -> np.ndarray:
    if path in _frame_cache:
        return _frame_cache[path]
    arr = _load_image_array(path)
    if len(_frame_cache) >= _CACHE_MAX:
        # Evict oldest entry
        oldest = next(iter(_frame_cache))
        del _frame_cache[oldest]
    _frame_cache[path] = arr
    return arr


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _measure(label_arr: np.ndarray, rgb_arr: np.ndarray) -> pd.DataFrame:
    """Measure regionprops on a label array. Returns DataFrame with standard columns."""
    from skimage.measure import regionprops_table

    # Convert RGB to grayscale for intensity measurements
    gray = np.dot(rgb_arr[:, :, :3].astype(np.float64),
                  [0.2989, 0.5870, 0.1140])

    props = [
        'label', 'area', 'centroid', 'bbox',
        'perimeter', 'eccentricity',
        'axis_major_length', 'axis_minor_length',
        'orientation', 'solidity', 'extent',
        'equivalent_diameter_area',
        'mean_intensity', 'max_intensity', 'min_intensity',
    ]
    table = regionprops_table(
        label_arr, intensity_image=gray, properties=props)
    df = pd.DataFrame(table)
    if df.empty:
        return df

    df.rename(columns={
        'centroid-0': 'centroid_y',
        'centroid-1': 'centroid_x',
        'bbox-0': 'bbox_top',
        'bbox-1': 'bbox_left',
        'bbox-2': 'bbox_bottom',
        'bbox-3': 'bbox_right',
        'axis_major_length': 'major_axis',
        'axis_minor_length': 'minor_axis',
        'equivalent_diameter_area': 'equivalent_diameter',
    }, inplace=True)

    if 'orientation' in df.columns:
        df['orientation'] = np.degrees(df['orientation']).round(2)
    if 'area' in df.columns and 'perimeter' in df.columns:
        p = df['perimeter'].replace(0, np.nan)
        circ = (4 * np.pi * df['area'] / (p ** 2)).round(4)
        df['circularity'] = circ.where(circ <= 1.0, 1.0)
    if 'mean_intensity' in df.columns:
        df['sum_intensity'] = (df['mean_intensity'] * df['area']).round(2)

    ordered = [c for c in [
        'label', 'area', 'equivalent_diameter',
        'centroid_y', 'centroid_x',
        'bbox_top', 'bbox_left', 'bbox_bottom', 'bbox_right',
        'perimeter', 'circularity', 'eccentricity', 'orientation',
        'major_axis', 'minor_axis', 'solidity', 'extent',
        'mean_intensity', 'sum_intensity', 'max_intensity', 'min_intensity',
    ] if c in df.columns]
    return df[ordered]


def _build_label_arr(masks: dict[int, np.ndarray], h: int, w: int) -> np.ndarray:
    """Build int32 label array from per-object masks."""
    label_arr = np.zeros((h, w), dtype=np.int32)
    for obj_id, mask in sorted(masks.items()):
        while mask.ndim > 2:
            mask = mask[0]
        if mask.shape[0] != h or mask.shape[1] != w:
            continue
        label_arr[mask > 0] = obj_id
    return label_arr


def _render_trajectory(df: pd.DataFrame, bg_image: np.ndarray | None,
                       img_size: tuple[int, int] | None = None) -> Image.Image:
    """Draw centroid trajectories on a background image (or blank canvas).

    df must have columns: frame, label, centroid_x, centroid_y.
    Returns an RGB PIL Image.
    """
    if bg_image is not None:
        h, w = bg_image.shape[:2]
        # Dim the background so trajectories stand out
        dimmed = (bg_image.astype(np.float32) * 0.3)
        canvas = Image.fromarray(np.clip(dimmed, 0, 255).astype(np.uint8))
    elif img_size:
        w, h = img_size
        canvas = Image.new('RGB', (w, h), (20, 20, 20))
    else:
        w, h = 800, 600
        canvas = Image.new('RGB', (w, h), (20, 20, 20))

    draw = ImageDraw.Draw(canvas)

    required = {'frame', 'label', 'centroid_x', 'centroid_y'}
    if not required.issubset(set(df.columns)):
        return canvas

    labels = sorted(df['label'].unique())
    for obj_id in labels:
        obj_df = df[df['label'] == obj_id].sort_values('frame')
        if len(obj_df) < 2:
            continue

        r, g, b = _obj_color(int(obj_id))
        color = (r, g, b)
        pts = list(zip(obj_df['centroid_x'].values,
                       obj_df['centroid_y'].values))

        # Draw trajectory line
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            draw.line([(x0, y0), (x1, y1)], fill=color, width=2)

        # Draw dots at each centroid
        for i, (x, y) in enumerate(pts):
            radius = 4 if i == 0 or i == len(pts) - 1 else 2
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=color)

        # Label at start point
        x0, y0 = pts[0]
        draw.text((x0 + 6, y0 - 6), str(obj_id), fill=color)

    return canvas


# ---------------------------------------------------------------------------
# _VideoAnalyzeWidget — two-column interactive widget
# ---------------------------------------------------------------------------

class _VideoAnalyzeWidget(NodeBaseWidget):
    """Integrated widget: object list on left, canvas + slider on right."""

    _progress_signal = Signal(int, int)       # (current, total)
    _done_signal = Signal(object)             # result DataFrame or None
    _frame_display_signal = Signal(object)    # rgb_arr to display
    _auto_done_signal = Signal(object)        # dict[int, np.ndarray]
    _preview_signal = Signal(int, object, object)  # (frame_idx, masks_dict, rgb_arr)

    _VIEW_MAX = 600
    _VIEW_MIN = 200

    def __init__(self, parent=None, name='', label=''):
        super().__init__(parent, name, label)

        self._session: SAM2ImageSession | None = None
        self._session_lock = threading.Lock()

        # ── state ──────────────────────────────────────────────────────
        self._files: list[Path] = []          # image-folder mode
        self._video_reader = None             # video mode (imageio reader)
        self._video_n_frames: int = 0
        self._source_mode: str = 'folder'     # 'folder' or 'video'
        self._current_frame_idx: int = 0
        self._rgb_arr: np.ndarray | None = None
        self._encoded_frame_idx: int = -1  # which frame is SAM2-encoded

        # Keyframes: frame_idx → {obj_id: mask}
        self._keyframes: dict[int, dict[int, np.ndarray]] = {}
        # Keyframe point prompts: frame_idx → {obj_id: (coords, labels)}
        self._keyframe_points: dict[int, dict[int, tuple]] = {}

        # Analysis results
        self._frame_masks: dict[int, dict[int, np.ndarray]] = {}
        self._result_df: pd.DataFrame | None = None
        self._corrections_pending: bool = False
        self._cancel_analysis: bool = False
        self._analyzing: bool = False

        # ── build UI ───────────────────────────────────────────────────
        container = QtWidgets.QWidget()
        main_vlay = QtWidgets.QVBoxLayout(container)
        main_vlay.setContentsMargins(4, 2, 4, 2)
        main_vlay.setSpacing(3)

        # ── top compact controls (folder, video, settings) ────────────
        _ss = "font-size:10px;"
        _combo_ss = ("QComboBox { font-size:10px; padding:1px 2px; }"
                     "QComboBox::drop-down { width:14px; }")
        _le_ss = "font-size:10px; padding:1px 2px;"

        # Row 1: Folder
        r1 = QtWidgets.QHBoxLayout(); r1.setSpacing(3); r1.setContentsMargins(0,0,0,0)
        r1.addWidget(self._make_label("Folder", 38, _ss))
        self._le_folder = QtWidgets.QLineEdit()
        self._le_folder.setStyleSheet(_le_ss)
        self._le_folder.setPlaceholderText("Image folder…")
        r1.addWidget(self._le_folder, 1)
        self._btn_folder = QtWidgets.QPushButton("…")
        self._btn_folder.setFixedSize(35, 45)
        r1.addWidget(self._btn_folder)
        main_vlay.addLayout(r1)

        # Row 2: Video
        r2 = QtWidgets.QHBoxLayout(); r2.setSpacing(3); r2.setContentsMargins(0,0,0,0)
        r2.addWidget(self._make_label("Video", 38, _ss))
        self._le_video = QtWidgets.QLineEdit()
        self._le_video.setStyleSheet(_le_ss)
        self._le_video.setPlaceholderText("Video file…")
        r2.addWidget(self._le_video, 1)
        self._btn_video = QtWidgets.QPushButton("…")
        self._btn_video.setFixedSize(35, 45)
        r2.addWidget(self._btn_video)
        main_vlay.addLayout(r2)

        # Row 3: Pattern | Model | Track method
        r3 = QtWidgets.QHBoxLayout(); r3.setSpacing(4); r3.setContentsMargins(0,0,0,0)
        r3.addWidget(self._make_label("Pat", 22, _ss))
        self._le_pattern = QtWidgets.QLineEdit("*.tif")
        self._le_pattern.setStyleSheet(_le_ss)
        self._le_pattern.setFixedWidth(50)
        r3.addWidget(self._le_pattern)
        r3.addWidget(self._make_label("Model", 32, _ss))
        self._combo_model = QtWidgets.QComboBox()
        self._combo_model.addItems(['tiny', 'small', 'base_plus', 'large'])
        self._combo_model.setStyleSheet(_combo_ss)
        self._combo_model.setFixedWidth(72)
        r3.addWidget(self._combo_model)
        r3.addWidget(self._make_label("Track", 30, _ss))
        self._combo_track = QtWidgets.QComboBox()
        self._combo_track.addItems(['Centroid', 'Memory', 'Cellpose'])
        self._combo_track.setStyleSheet(_combo_ss)
        self._combo_track.setFixedWidth(78)
        r3.addWidget(self._combo_track)
        r3.addStretch()
        main_vlay.addLayout(r3)

        # Row 4: Mode | Text prompt | DINO score
        r4 = QtWidgets.QHBoxLayout(); r4.setSpacing(4); r4.setContentsMargins(0,0,0,0)
        r4.addWidget(self._make_label("Mode", 30, _ss))
        self._combo_mode = QtWidgets.QComboBox()
        self._combo_mode.addItems(['Manual', 'Text'])
        self._combo_mode.setStyleSheet(_combo_ss)
        self._combo_mode.setFixedWidth(65)
        r4.addWidget(self._combo_mode)
        self._lbl_text = self._make_label("Text", 24, _ss)
        r4.addWidget(self._lbl_text)
        self._le_text_prompt = QtWidgets.QLineEdit()
        self._le_text_prompt.setStyleSheet(_le_ss)
        self._le_text_prompt.setPlaceholderText("e.g. nucleus, membrane")
        r4.addWidget(self._le_text_prompt, 1)
        self._lbl_dino = self._make_label("DINO", 28, _ss)
        r4.addWidget(self._lbl_dino)
        self._spin_gdino = QtWidgets.QDoubleSpinBox()
        self._spin_gdino.setRange(0.01, 1.0)
        self._spin_gdino.setSingleStep(0.05)
        self._spin_gdino.setDecimals(2)
        self._spin_gdino.setValue(0.3)
        self._spin_gdino.setFixedWidth(52)
        self._spin_gdino.setStyleSheet(_ss)
        r4.addWidget(self._spin_gdino)
        main_vlay.addLayout(r4)

        # Text-mode widgets visibility
        self._text_mode_widgets = [
            self._lbl_text, self._le_text_prompt,
            self._lbl_dino, self._spin_gdino]
        self._update_mode_visibility()

        main_vlay.addWidget(self._make_separator())

        # ── two-column section ─────────────────────────────────────────
        two_col = QtWidgets.QHBoxLayout()
        two_col.setSpacing(6)

        # LEFT column — objects + collapsible params
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(2)

        lbl_objects = QtWidgets.QLabel("Objects")
        lbl_objects.setStyleSheet("font-weight:bold; font-size:11px;")
        left.addWidget(lbl_objects)

        self._obj_list = QtWidgets.QListWidget()
        self._obj_list.setMaximumHeight(150)
        self._obj_list.setMinimumHeight(36)
        self._obj_list.setFixedWidth(140)
        self._obj_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        left.addWidget(self._obj_list)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(2)
        self._btn_add_obj = QtWidgets.QPushButton("+")
        self._btn_add_obj.setFixedSize(36, 24)
        self._btn_add_obj.setToolTip("Add object")
        self._btn_del_obj = QtWidgets.QPushButton("-")
        self._btn_del_obj.setFixedSize(36, 24)
        self._btn_del_obj.setToolTip("Delete object")
        self._btn_clear_all = QtWidgets.QPushButton("X")
        self._btn_clear_all.setFixedSize(36, 24)
        self._btn_clear_all.setToolTip("Clear all")
        self._btn_clear_all.setStyleSheet(
            "QPushButton { color:#c55; font-weight:bold; }"
            "QPushButton:hover { background:#4a2020; }")
        btn_row.addWidget(self._btn_add_obj)
        btn_row.addWidget(self._btn_del_obj)
        btn_row.addWidget(self._btn_clear_all)
        btn_row.addStretch()
        left.addLayout(btn_row)

        # Keyframe indicator
        self._kf_label = QtWidgets.QLabel("Keyframes: -")
        self._kf_label.setStyleSheet("color:#aaa; font-size:9px;")
        self._kf_label.setWordWrap(True)
        left.addWidget(self._kf_label)

        # ── Collapsible: Tracking params ──
        self._track_toggle = self._make_collapse_btn("Tracking")
        left.addWidget(self._track_toggle)
        self._track_params = QtWidgets.QWidget()
        tp_lay = QtWidgets.QVBoxLayout(self._track_params)
        tp_lay.setContentsMargins(0, 0, 0, 0); tp_lay.setSpacing(1)
        self._spin_score = self._make_spin("Score", 0.5, 0.01, 1.0, 0.05, 2)
        tp_lay.addLayout(self._spin_score[0])
        self._spin_iou = self._make_spin("IoU", 0.2, 0.0, 1.0, 0.05, 2)
        tp_lay.addLayout(self._spin_iou[0])
        self._spin_dormant = self._make_spin("Dormant", 10, 0, 999, 1, 0)
        tp_lay.addLayout(self._spin_dormant[0])
        self._spin_appear = self._make_spin("Appear", 0.3, 0.0, 1.0, 0.05, 2)
        tp_lay.addLayout(self._spin_appear[0])
        self._track_params.setVisible(False)
        left.addWidget(self._track_params)

        # ── Collapsible: Auto-segment params ──
        self._auto_toggle = self._make_collapse_btn("Auto Segment")
        left.addWidget(self._auto_toggle)
        self._auto_params = QtWidgets.QWidget()
        ap_lay = QtWidgets.QVBoxLayout(self._auto_params)
        ap_lay.setContentsMargins(0, 0, 0, 0); ap_lay.setSpacing(1)
        self._spin_auto_grid = self._make_spin("Grid", 16, 2, 64, 2, 0)
        ap_lay.addLayout(self._spin_auto_grid[0])
        self._spin_auto_score = self._make_spin("Score", 0.85, 0.01, 1.0, 0.05, 2)
        ap_lay.addLayout(self._spin_auto_score[0])
        self._spin_auto_min = self._make_spin("Min%", 0.1, 0.0, 100.0, 0.1, 1)
        ap_lay.addLayout(self._spin_auto_min[0])
        self._spin_auto_max = self._make_spin("Max%", 50.0, 0.0, 100.0, 1.0, 1)
        ap_lay.addLayout(self._spin_auto_max[0])
        self._auto_params.setVisible(False)
        left.addWidget(self._auto_params)

        left.addStretch()

        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(160)
        two_col.addWidget(left_widget)

        # RIGHT column
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(3)

        # Toolbar
        tb = QtWidgets.QHBoxLayout()
        tb.setSpacing(4)

        self._btn_mode = QtWidgets.QPushButton("+Include")
        self._btn_mode.setCheckable(True)
        self._btn_mode.setChecked(True)
        self._btn_mode.setFixedHeight(24)
        self._btn_mode.setToolTip("Toggle: Include / Exclude")
        self._update_mode_style(True)
        tb.addWidget(self._btn_mode)

        self._btn_undo = QtWidgets.QPushButton("Undo")
        self._btn_undo.setFixedHeight(22)
        tb.addWidget(self._btn_undo)

        self._btn_clear = QtWidgets.QPushButton("Clear")
        self._btn_clear.setFixedHeight(22)
        self._btn_clear.setToolTip("Clear current object's points")
        tb.addWidget(self._btn_clear)

        self._btn_fit = QtWidgets.QPushButton("Fit")
        self._btn_fit.setFixedHeight(22)
        tb.addWidget(self._btn_fit)

        self._btn_auto = QtWidgets.QPushButton("Auto")
        self._btn_auto.setFixedHeight(22)
        self._btn_auto.setToolTip("Auto-segment all objects on this frame")
        self._btn_auto.setStyleSheet(
            "QPushButton { background:#1a3a6a; color:white; "
            "border:1px solid #2a5a9a; border-radius:3px; padding:2px 6px; }"
            "QPushButton:hover { background:#254a80; }"
            "QPushButton:disabled { background:#333; color:#777; }")
        tb.addWidget(self._btn_auto)

        tb.addStretch()
        self._score_label = QtWidgets.QLabel("")
        self._score_label.setStyleSheet("color:#aaa;font-size:10px;")
        tb.addWidget(self._score_label)
        right.addLayout(tb)

        # Canvas
        self._scene = QtWidgets.QGraphicsScene()
        self._view = SAM2ClickGraphicsView(self._scene)
        self._view.setMinimumSize(self._VIEW_MIN, self._VIEW_MIN)
        self._view.setFixedSize(self._VIEW_MAX, self._VIEW_MAX)
        self._view.setStyleSheet("background:#1a1a1a;")
        right.addWidget(self._view)

        # Frame slider
        slider_row = QtWidgets.QHBoxLayout()
        slider_row.setContentsMargins(0, 2, 0, 0)

        lbl_frame = QtWidgets.QLabel("Frame")
        lbl_frame.setFixedWidth(40)
        slider_row.addWidget(lbl_frame)

        self._slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(1)
        self._slider.setMaximum(1)
        self._slider.setValue(1)
        slider_row.addWidget(self._slider, 1)

        self._spin_frame = QtWidgets.QSpinBox()
        self._spin_frame.setMinimum(1)
        self._spin_frame.setMaximum(1)
        self._spin_frame.setValue(1)
        self._spin_frame.setFixedWidth(70)
        slider_row.addWidget(self._spin_frame)

        self._total_label = QtWidgets.QLabel("/ 0")
        self._total_label.setFixedWidth(50)
        slider_row.addWidget(self._total_label)

        self._ref_label = QtWidgets.QLabel("")
        self._ref_label.setStyleSheet("color:#5a5;font-size:10px;")
        self._ref_label.setFixedWidth(60)
        slider_row.addWidget(self._ref_label)

        right.addLayout(slider_row)

        # Tip
        tip = QtWidgets.QLabel(
            "Click to annotate \u00b7 Ctrl+Z = undo \u00b7 "
            "Middle-click = pan \u00b7 Scroll = zoom")
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tip.setStyleSheet("color:#999; font-size:9px; padding:1px;")
        tip.setWordWrap(True)
        right.addWidget(tip)

        two_col.addLayout(right, 1)
        main_vlay.addLayout(two_col)

        # ── bottom section ─────────────────────────────────────────────
        bottom = QtWidgets.QHBoxLayout()
        bottom.setSpacing(6)

        self._btn_analyze = QtWidgets.QPushButton("\u25b6  Analyze All")
        self._btn_analyze.setFixedHeight(30)
        self._btn_analyze.setStyleSheet(
            "QPushButton { background:#1a5c1a; color:white; font-weight:bold; "
            "border:1px solid #2a8a2a; border-radius:4px; padding:4px 16px; }"
            "QPushButton:hover { background:#258025; }"
            "QPushButton:disabled { background:#333; color:#777; }")
        bottom.addWidget(self._btn_analyze)

        self._btn_cancel = QtWidgets.QPushButton("Cancel")
        self._btn_cancel.setFixedHeight(30)
        self._btn_cancel.setEnabled(False)
        bottom.addWidget(self._btn_cancel)

        self._status_label = QtWidgets.QLabel("")
        self._status_label.setStyleSheet("color:#aaa; font-size:10px;")
        bottom.addWidget(self._status_label, 1)

        main_vlay.addLayout(bottom)

        self._container = container
        self.set_custom_widget(self._container)

        # ── connections ────────────────────────────────────────────────
        # Top controls
        self._btn_folder.clicked.connect(self._on_browse_folder)
        self._btn_video.clicked.connect(self._on_browse_video)
        self._le_folder.editingFinished.connect(self._on_folder_edited)
        self._le_video.editingFinished.connect(self._on_video_edited)
        self._le_pattern.editingFinished.connect(self._on_pattern_edited)
        self._combo_model.currentTextChanged.connect(self._on_model_changed)
        self._combo_mode.currentTextChanged.connect(self._on_analysis_mode_changed)
        # Collapse buttons
        self._track_toggle.clicked.connect(
            lambda: self._toggle_section(self._track_toggle, self._track_params))
        self._auto_toggle.clicked.connect(
            lambda: self._toggle_section(self._auto_toggle, self._auto_params))
        # Canvas / interaction
        self._btn_mode.toggled.connect(self._on_mode_toggled)
        self._btn_add_obj.clicked.connect(self._on_add_object)
        self._btn_del_obj.clicked.connect(self._on_delete_object)
        self._btn_clear_all.clicked.connect(self._on_clear_all)
        self._btn_undo.clicked.connect(self._on_undo)
        self._btn_clear.clicked.connect(self._on_clear)
        self._btn_fit.clicked.connect(self._on_fit)
        self._btn_auto.clicked.connect(self._on_auto_segment)
        self._btn_analyze.clicked.connect(self._on_analyze)
        self._btn_cancel.clicked.connect(self._on_cancel)
        self._view.points_changed.connect(self._on_points_changed)
        self._obj_list.currentItemChanged.connect(self._on_list_selection)
        self._slider.valueChanged.connect(self._on_slider)
        self._spin_frame.valueChanged.connect(self._on_spin)
        self._progress_signal.connect(self._on_progress,
                                      Qt.ConnectionType.QueuedConnection)
        self._done_signal.connect(self._on_done,
                                  Qt.ConnectionType.QueuedConnection)
        self._frame_display_signal.connect(self._apply_frame,
                                           Qt.ConnectionType.QueuedConnection)
        self._auto_done_signal.connect(self._on_auto_done,
                                       Qt.ConnectionType.QueuedConnection)
        self._preview_signal.connect(self._on_preview,
                                     Qt.ConnectionType.QueuedConnection)

        self._rebuild_obj_list()

    # ── helper: create labeled spinbox ─────────────────────────────────
    @staticmethod
    def _make_label(text: str, width: int, ss: str = "") -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setFixedWidth(width)
        if ss:
            lbl.setStyleSheet(ss)
        return lbl

    @staticmethod
    def _make_collapse_btn(text: str) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton()
        btn.setText(f"\u25b6 {text}")
        btn.setCheckable(True)
        btn.setChecked(False)
        btn.setStyleSheet(
            "QToolButton { border:none; font-size:10px; font-weight:bold;"
            " color:#aaa; padding:1px 0; text-align:left; }"
            "QToolButton:hover { color:#ddd; }")
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        return btn

    @staticmethod
    def _toggle_section(btn: QtWidgets.QToolButton, widget: QtWidgets.QWidget):
        vis = not widget.isVisible()
        widget.setVisible(vis)
        label = btn.text()[2:]  # strip arrow
        btn.setText(f"\u25bc {label}" if vis else f"\u25b6 {label}")

    def _update_mode_visibility(self):
        is_text = (self._combo_mode.currentText() == 'Text')
        for w in self._text_mode_widgets:
            w.setVisible(is_text)

    # ── top controls handlers ──────────────────────────────────────────
    def _on_browse_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self._container, "Select Image Folder")
        if d:
            self._le_folder.setText(d)
            self._on_folder_edited()

    def _on_browse_video(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._container, "Select Video",
            filter="Video (*.mp4 *.avi *.mov *.mkv *.webm *.flv);;All (*)")
        if f:
            self._le_video.setText(f)
            self._on_video_edited()

    def _on_folder_edited(self):
        folder = self._le_folder.text().strip()
        pattern = self._le_pattern.text().strip() or '*.tif'
        if self.node:
            variant = self._combo_model.currentText()
            self.node._ensure_session(variant)
        self.scan_folder(folder, pattern)

    def _on_video_edited(self):
        path = self._le_video.text().strip()
        if self.node:
            variant = self._combo_model.currentText()
            self.node._ensure_session(variant)
        self.load_video(path)

    def _on_pattern_edited(self):
        folder = self._le_folder.text().strip()
        pattern = self._le_pattern.text().strip() or '*.tif'
        if folder:
            self.scan_folder(folder, pattern)

    def _on_model_changed(self, text):
        if self.node:
            self.node._ensure_session(text)

    def _on_analysis_mode_changed(self, _text):
        self._update_mode_visibility()

    @staticmethod
    def _make_spin(label: str, value: float, min_v: float, max_v: float,
                   step: float, decimals: int):
        lay = QtWidgets.QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lbl = QtWidgets.QLabel(label)
        lbl.setFixedWidth(50)
        lbl.setStyleSheet("font-size:10px;")
        spin = QtWidgets.QDoubleSpinBox()
        spin.setMinimum(min_v)
        spin.setMaximum(max_v)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setFixedWidth(70)
        lay.addWidget(lbl)
        lay.addWidget(spin)
        lay.addStretch()
        return lay, spin

    @staticmethod
    def _make_separator():
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("color:#444;")
        return line

    # ── mode toggle ────────────────────────────────────────────────────
    def _on_mode_toggled(self, checked: bool):
        self._update_mode_style(checked)
        self._view.set_mode(1 if checked else 0)

    def _update_mode_style(self, include: bool):
        if include:
            self._btn_mode.setText("+Include")
            self._btn_mode.setStyleSheet(
                "QPushButton { background:#1a5c1a; color:white; "
                "border:1px solid #2a8a2a; border-radius:3px; padding:2px 8px; }"
                "QPushButton:hover { background:#258025; }")
        else:
            self._btn_mode.setText("\u2212Exclude")
            self._btn_mode.setStyleSheet(
                "QPushButton { background:#6a1a1a; color:white; "
                "border:1px solid #9a2a2a; border-radius:3px; padding:2px 8px; }"
                "QPushButton:hover { background:#802525; }")

    # ── object list ────────────────────────────────────────────────────
    def _rebuild_obj_list(self):
        self._obj_list.blockSignals(True)
        cur = self._view.current_obj
        self._obj_list.clear()
        for oid in self._view.object_ids:
            pts, labs = self._view.get_points(oid)
            n_fg = labs.count(1)
            n_bg = labs.count(0)
            desc = f"Object {oid}"
            if n_fg or n_bg:
                parts = []
                if n_fg:
                    parts.append(f"{n_fg} fg")
                if n_bg:
                    parts.append(f"{n_bg} bg")
                desc += f"  ({', '.join(parts)})"
            if self._session and oid in self._session.scores:
                desc += f"  [{self._session.scores[oid]:.2f}]"

            li = QtWidgets.QListWidgetItem()
            li.setData(Qt.ItemDataRole.UserRole, oid)
            r, g, b = _obj_color(oid)
            pix = QPixmap(12, 12)
            pix.fill(QColor(r, g, b))
            li.setIcon(QIcon(pix))
            li.setText(desc)
            self._obj_list.addItem(li)
            if oid == cur:
                li.setSelected(True)
                self._obj_list.setCurrentItem(li)
        self._obj_list.blockSignals(False)

    def _on_list_selection(self, current, _previous):
        if current is None:
            return
        oid = current.data(Qt.ItemDataRole.UserRole)
        if oid is not None:
            self._view.set_current_object(oid)

    def _on_add_object(self):
        self._view.add_object()
        self._rebuild_obj_list()

    def _on_delete_object(self):
        oid = self._view.current_obj
        self._view.delete_object(oid)
        if self._session:
            self._session._masks.pop(oid, None)
            self._session._scores.pop(oid, None)
        # Remove from current keyframe
        fi = self._current_frame_idx
        if fi in self._keyframes:
            self._keyframes[fi].pop(oid, None)
            if not self._keyframes[fi]:
                del self._keyframes[fi]
        if fi in self._keyframe_points:
            self._keyframe_points[fi].pop(oid, None)
            if not self._keyframe_points[fi]:
                del self._keyframe_points[fi]
        self._rebuild_obj_list()
        self._refresh_mask_overlay()
        self._update_kf_label()

    def _on_clear_all(self):
        self._view.clear_all()
        if self._session:
            self._session._masks.clear()
            self._session._scores.clear()
        self._keyframes.clear()
        self._keyframe_points.clear()
        self._frame_masks.clear()
        self._result_df = None
        self._corrections_pending = False
        self._rebuild_obj_list()
        self._refresh_mask_overlay()
        self._score_label.setText("")
        self._status_label.setText("")
        self._update_kf_label()

    # ── toolbar ────────────────────────────────────────────────────────
    def _on_undo(self):
        self._view.undo_last_point()

    def _on_clear(self):
        self._view.clear_current()
        self._rebuild_obj_list()
        self._run_all_predicts()

    def _on_fit(self):
        if self._view._bg_pixmap_item:
            self._view.fitInView(self._view._bg_pixmap_item,
                                 Qt.AspectRatioMode.KeepAspectRatio)
            self._view._scale = 1.0

    # ── auto-segment ───────────────────────────────────────────────────
    def _on_auto_segment(self):
        if self._session is None:
            return
        self._ensure_encoded()
        if not self._session.is_image_set:
            return
        grid = int(self._spin_auto_grid[1].value())
        score_thr = self._spin_auto_score[1].value()
        min_pct = self._spin_auto_min[1].value()
        max_pct = self._spin_auto_max[1].value()
        self._auto_params = (grid, score_thr, min_pct / 100.0, max_pct / 100.0)
        self._btn_auto.setEnabled(False)
        self._btn_auto.setText("Running...")
        self._score_label.setText(f"Auto-segmenting ({grid}x{grid})...")
        t = threading.Thread(target=self._auto_worker, daemon=True)
        t.start()

    def _auto_worker(self):
        try:
            grid, score_thr, min_frac, max_frac = self._auto_params
            self._session.auto_segment(
                points_per_side=grid, score_threshold=score_thr,
                nms_iou_threshold=0.5,
                min_area_frac=min_frac, max_area_frac=max_frac)
            self._auto_done_signal.emit(self._session.masks)
        except Exception:
            logger.exception("Auto-segment failed")
            self._auto_done_signal.emit({})

    def _on_auto_done(self, masks: dict):
        self._btn_auto.setEnabled(True)
        self._btn_auto.setText("Auto")
        if not masks:
            self._score_label.setText("No objects found")
            return
        # Rebuild viewer objects from auto-generated masks
        self._view.clear_all()
        max_id = 0
        for oid in sorted(masks.keys()):
            max_id = max(max_id, oid)
            self._view._objects[oid] = {"points": [], "labels": [], "items": []}
        self._view._current_obj = sorted(masks.keys())[0]
        self._view._next_obj = max_id + 1
        self._refresh_mask_overlay()
        self._rebuild_obj_list()
        self._score_label.setText(f"{len(masks)} objects found")
        # Store as keyframe
        fi = self._current_frame_idx
        self._keyframes[fi] = {oid: m.copy() for oid, m in masks.items()}
        self._keyframe_points[fi] = {}  # no explicit points for auto
        self._update_kf_label()
        if self._frame_masks:
            self._corrections_pending = True
            self._status_label.setText("Corrections pending \u2014 Analyze All to re-track")

    # ── frame slider ───────────────────────────────────────────────────
    def _on_slider(self, val):
        self._spin_frame.blockSignals(True)
        self._spin_frame.setValue(val)
        self._spin_frame.blockSignals(False)
        self._on_frame_changed(val - 1)

    def _on_spin(self, val):
        self._slider.blockSignals(True)
        self._slider.setValue(val)
        self._slider.blockSignals(False)
        self._on_frame_changed(val - 1)

    def set_frame_range(self, n_frames: int):
        last = max(1, n_frames)
        self._slider.setMaximum(last)
        self._spin_frame.setMaximum(last)
        self._total_label.setText(f"/ {n_frames}")

    @property
    def _total_frames(self) -> int:
        if self._source_mode == 'video':
            return self._video_n_frames
        return len(self._files)

    def _load_frame(self, frame_idx: int) -> np.ndarray | None:
        """Load a frame from either folder or video source."""
        if self._source_mode == 'video' and self._video_reader is not None:
            try:
                arr = self._video_reader.get_data(frame_idx)
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                elif arr.ndim == 3 and arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                return arr.astype(np.uint8)
            except Exception as e:
                logger.error("Failed to read video frame %d: %s", frame_idx, e)
                return None
        elif self._source_mode == 'folder' and frame_idx < len(self._files):
            try:
                return _cached_load(str(self._files[frame_idx]))
            except Exception as e:
                logger.error("Failed to load %s: %s", self._files[frame_idx], e)
                return None
        return None

    def _frame_name(self, frame_idx: int) -> str:
        """Return a display name for the frame."""
        if self._source_mode == 'folder' and frame_idx < len(self._files):
            return self._files[frame_idx].name
        return f"frame_{frame_idx + 1:06d}"

    def _ensure_encoded(self):
        """Lazily encode the current frame in SAM2 (only when needed)."""
        if self._session is None or self._rgb_arr is None:
            return
        if self._encoded_frame_idx == self._current_frame_idx:
            return
        self._session.set_image(self._rgb_arr)
        self._encoded_frame_idx = self._current_frame_idx

    def _on_frame_changed(self, frame_idx: int):
        """Load and display a frame."""
        if frame_idx < 0 or frame_idx >= self._total_frames:
            return
        # Save old frame state BEFORE switching (uses _current_frame_idx)
        self._save_current_as_keyframe()
        self._current_frame_idx = frame_idx
        rgb_arr = self._load_frame(frame_idx)
        if rgb_arr is None:
            return

        self._rgb_arr = rgb_arr

        # Load image into canvas (no SAM2 encoding — deferred to click)
        self._apply_frame(rgb_arr)

        # Restore points/masks for this frame if it's a keyframe
        self._restore_frame_state(frame_idx)

        # Update ref label
        if frame_idx in self._keyframes:
            self._ref_label.setText("Keyframe")
            self._ref_label.setStyleSheet("color:#5a5;font-size:10px;font-weight:bold;")
        else:
            self._ref_label.setText("")
            self._ref_label.setStyleSheet("color:#5a5;font-size:10px;")

    def _apply_frame(self, rgb_arr: np.ndarray, skip_masks: bool = False):
        """Display an RGB array in the canvas."""
        h, w = rgb_arr.shape[:2]
        if w >= h:
            vw = self._VIEW_MAX
            vh = max(self._VIEW_MIN, int(self._VIEW_MAX * h / w))
        else:
            vh = self._VIEW_MAX
            vw = max(self._VIEW_MIN, int(self._VIEW_MAX * w / h))
        self._view.setFixedSize(vw, vh)
        self._view.load_image(rgb_arr)
        # Always clear stale mask overlay first
        self._view.clear_mask()

        if skip_masks:
            return

        # Show tracked masks overlay if available (post-analysis)
        fi = self._current_frame_idx
        if fi in self._frame_masks:
            self._view.update_masks(self._frame_masks[fi])
        elif fi in self._keyframes:
            # Show keyframe masks
            masks = {oid: m for oid, m in self._keyframes[fi].items()
                     if np.any(m > 0)}
            if masks:
                self._view.update_masks(masks)

    def _restore_frame_state(self, frame_idx: int):
        """Restore point prompts and masks when navigating to a keyframe."""
        if frame_idx in self._keyframe_points:
            # Restore points visually — show stored masks without re-encoding
            kf_pts = self._keyframe_points[frame_idx]
            objects_data = {}
            for oid, (coords, labels) in kf_pts.items():
                pts = [(int(c[0]), int(c[1])) for c in coords]
                labs = [int(l) for l in labels]
                objects_data[str(oid)] = {"points": pts, "labels": labs}
            if objects_data:
                self._view.restore_objects(objects_data)
                self._rebuild_obj_list()
                # Show stored masks if available (no SAM2 encoding needed)
                if frame_idx in self._keyframes and self._session:
                    self._session._masks = {
                        oid: m.copy()
                        for oid, m in self._keyframes[frame_idx].items()}
                    self._refresh_mask_overlay()
                return

        if frame_idx in self._keyframes:
            # Keyframe with masks but no explicit points (auto-generated)
            masks = self._keyframes[frame_idx]
            self._view.clear_all()
            max_id = 0
            for oid in sorted(masks.keys()):
                max_id = max(max_id, oid)
                self._view._objects[oid] = {"points": [], "labels": [], "items": []}
            if masks:
                self._view._current_obj = sorted(masks.keys())[0]
                self._view._next_obj = max_id + 1
            if self._session:
                self._session._masks = {oid: m.copy() for oid, m in masks.items()}
            self._refresh_mask_overlay()
            self._rebuild_obj_list()
            return

        # Not a keyframe — clear points but show tracked masks if available
        self._view.clear_all()
        if self._session:
            self._session._masks.clear()
            self._session._scores.clear()
        self._rebuild_obj_list()

    def _save_current_as_keyframe(self):
        """If current frame has any points/masks, save as keyframe."""
        if self._session is None:
            return
        has_masks = any(np.any(m > 0) for m in self._session.masks.values())
        has_points = any(len(o["points"]) > 0 for o in self._view._objects.values())
        if not has_masks and not has_points:
            return

        fi = self._current_frame_idx
        if has_masks:
            self._keyframes[fi] = {oid: m.copy()
                                   for oid, m in self._session.masks.items()
                                   if np.any(m > 0)}
        if has_points:
            kf_pts = {}
            for oid in self._view.object_ids:
                pts, labs = self._view.get_points(oid)
                if pts:
                    kf_pts[oid] = (
                        np.array(pts, dtype=np.int32),
                        np.array(labs, dtype=np.int32),
                    )
            if kf_pts:
                self._keyframe_points[fi] = kf_pts

    # ── point interaction ──────────────────────────────────────────────
    def _on_points_changed(self, obj_id: int):
        if self._session is None:
            return
        self._ensure_encoded()
        if not self._session.is_image_set:
            return

        points, labels = self._view.get_points(obj_id)
        if points:
            coords = np.array(points, dtype=np.int32)
            labs = np.array(labels, dtype=np.int32)
            mask, scores = self._session.predict(coords, labs, label_id=obj_id)
            best = float(np.max(scores)) if np.asarray(scores).size > 0 else 0.0
            self._score_label.setText(f"Score: {best:.2f}")
        else:
            h, w = self._session.orig_im_size
            self._session._masks[obj_id] = np.zeros((h, w), dtype=np.uint8)
            self._score_label.setText("")

        self._refresh_mask_overlay()
        self._rebuild_obj_list()

        # Auto-save keyframe
        fi = self._current_frame_idx
        self._keyframes[fi] = {oid: m.copy()
                               for oid, m in self._session.masks.items()
                               if np.any(m > 0)}
        kf_pts = {}
        for oid in self._view.object_ids:
            pts, labs = self._view.get_points(oid)
            if pts:
                kf_pts[oid] = (
                    np.array(pts, dtype=np.int32),
                    np.array(labs, dtype=np.int32),
                )
        if kf_pts:
            self._keyframe_points[fi] = kf_pts
        self._update_kf_label()

        # Mark corrections pending if post-analysis
        if self._frame_masks:
            self._corrections_pending = True
            self._status_label.setText(
                "Corrections pending \u2014 Analyze All to re-track")

    def _run_all_predicts(self):
        if self._session is None:
            return
        self._ensure_encoded()
        if not self._session.is_image_set:
            return
        for oid in self._view.object_ids:
            points, labels = self._view.get_points(oid)
            if points:
                coords = np.array(points, dtype=np.int32)
                labs = np.array(labels, dtype=np.int32)
                self._session.predict(coords, labs, label_id=oid)
            elif oid in self._session._masks:
                h, w = self._session.orig_im_size
                self._session._masks[oid] = np.zeros((h, w), dtype=np.uint8)
        valid = set(self._view.object_ids)
        for k in list(self._session._masks.keys()):
            if k not in valid:
                del self._session._masks[k]
        self._refresh_mask_overlay()

    def _refresh_mask_overlay(self):
        if self._session is None:
            return
        masks = {k: v for k, v in self._session.masks.items()
                 if np.any(v > 0)}
        if masks:
            self._view.update_masks(masks)
        else:
            self._view.clear_mask()

    def _update_kf_label(self):
        if self._keyframes:
            kfs = sorted(self._keyframes.keys())
            kf_str = ", ".join(str(k + 1) for k in kfs)  # 1-based
            self._kf_label.setText(f"Keyframes: {kf_str}")
        else:
            self._kf_label.setText("Keyframes: -")

    # ── folder scanning ────────────────────────────────────────────────
    def scan_folder(self, folder_path: str, pattern: str):
        """Scan folder for images, update slider."""
        self._close_video()
        self._source_mode = 'folder'
        self._files = []
        self._frame_masks.clear()
        self._result_df = None
        self._corrections_pending = False

        if not folder_path or not os.path.isdir(folder_path):
            self.set_frame_range(0)
            return

        matches = sorted(Path(folder_path).glob(pattern), key=_nat_key)
        self._files = [p for p in matches if p.is_file()]

        if not self._files:
            self.set_frame_range(0)
            return

        self.set_frame_range(len(self._files))
        self._slider.blockSignals(True)
        self._spin_frame.blockSignals(True)
        self._slider.setValue(1)
        self._spin_frame.setValue(1)
        self._slider.blockSignals(False)
        self._spin_frame.blockSignals(False)

        # Show first frame
        self._on_frame_changed(0)

    # ── video loading ───────────────────────────────────────────────────
    def load_video(self, video_path: str):
        """Load a video file using imageio + ffmpeg."""
        self._close_video()
        self._files = []
        self._frame_masks.clear()
        self._result_df = None
        self._corrections_pending = False

        if not video_path or not os.path.isfile(video_path):
            self._source_mode = 'folder'
            self.set_frame_range(0)
            return

        try:
            import imageio
            self._video_reader = imageio.get_reader(video_path, 'ffmpeg')
            self._video_n_frames = self._video_reader.count_frames()
            self._source_mode = 'video'
        except Exception as e:
            logger.error("Cannot open video '%s': %s", video_path, e)
            self._source_mode = 'folder'
            self.set_frame_range(0)
            return

        self.set_frame_range(self._video_n_frames)
        self._slider.blockSignals(True)
        self._spin_frame.blockSignals(True)
        self._slider.setValue(1)
        self._spin_frame.setValue(1)
        self._slider.blockSignals(False)
        self._spin_frame.blockSignals(False)

        # Show first frame
        QtCore.QTimer.singleShot(0, lambda: self._on_frame_changed(0))

    def _close_video(self):
        if self._video_reader is not None:
            try:
                self._video_reader.close()
            except Exception:
                pass
            self._video_reader = None
            self._video_n_frames = 0

    # ── analyze ────────────────────────────────────────────────────────
    def _on_analyze(self):
        if self._total_frames == 0:
            self._status_label.setText("No images loaded")
            return
        if not self._keyframes:
            # Check if text mode
            mode = self._combo_mode.currentText() or 'Manual'
            if mode == 'Manual':
                self._status_label.setText("No keyframes \u2014 annotate objects first")
                return

        self._cancel_analysis = False
        self._analyzing = True
        self._btn_analyze.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._status_label.setText("Starting analysis...")

        t = threading.Thread(target=self._analyze_worker, daemon=True)
        t.start()

    def _on_cancel(self):
        self._cancel_analysis = True
        self._status_label.setText("Cancelling...")

    def _analyze_worker(self):
        """Background thread: run analysis on all frames."""
        try:
            mode = self._combo_mode.currentText() or 'Manual'

            if mode == 'Text':
                result = self._analyze_text_mode()
            else:
                track_method = self._combo_track.currentText() or 'Centroid'
                if track_method == 'Memory':
                    result = self._analyze_manual_mode_memory()
                elif track_method == 'Cellpose':
                    result = self._analyze_cellpose_mode()
                else:
                    result = self._analyze_manual_mode()

            self._done_signal.emit(result)
        except Exception:
            logger.exception("Analysis failed")
            self._done_signal.emit(None)

    def _analyze_manual_mode(self) -> pd.DataFrame | None:
        """Keyframe-segmented centroid tracking."""
        if not self._keyframes:
            return None

        score_thr = self._spin_score[1].value()
        iou_thr = self._spin_iou[1].value()
        max_lost = int(self._spin_dormant[1].value())
        app_wt = self._spin_appear[1].value()
        variant = self._combo_model.currentText() or 'tiny'

        enc_path, dec_path = _model_manager.get_model_paths(variant)

        # Sort keyframes
        sorted_kfs = sorted(self._keyframes.keys())
        total_frames = self._total_frames
        all_rows = []
        frame_masks_new: dict[int, dict[int, np.ndarray]] = {}

        for kf_idx, kf_start in enumerate(sorted_kfs):
            if self._cancel_analysis:
                return None

            kf_end = sorted_kfs[kf_idx + 1] if kf_idx + 1 < len(sorted_kfs) else total_frames
            kf_masks = self._keyframes[kf_start]

            if not kf_masks:
                continue

            # Create fresh session + tracker for this segment
            session = SAM2ImageSession(str(enc_path), str(dec_path))
            strategy = CentroidTrackingStrategy(
                score_threshold=score_thr,
                iou_threshold=iou_thr,
                appearance_weight=app_wt)
            tracker = SAM2FrameTracker(
                session, strategy, max_lost_frames=max_lost)

            # Process the keyframe itself
            rgb_kf = self._load_frame(kf_start)
            if rgb_kf is None:
                continue
            session.set_image(rgb_kf)
            tracker.set_reference_masks(
                _build_label_arr(kf_masks, rgb_kf.shape[0], rgb_kf.shape[1]),
                rgb_arr=rgb_kf)

            frame_masks_new[kf_start] = kf_masks
            label_arr = _build_label_arr(kf_masks, rgb_kf.shape[0], rgb_kf.shape[1])
            df_row = _measure(label_arr, rgb_kf)
            if not df_row.empty:
                df_row.insert(0, 'frame', kf_start + 1)
                df_row.insert(1, 'file', self._frame_name(kf_start))
                all_rows.append(df_row)

            self._preview_signal.emit(kf_start, kf_masks, rgb_kf)
            self._progress_signal.emit(kf_start + 1, total_frames)

            # Track forward through segment
            for fi in range(kf_start + 1, kf_end):
                if self._cancel_analysis:
                    return None

                # If this frame is also a keyframe, use its masks directly
                if fi in self._keyframes:
                    masks = self._keyframes[fi]
                    frame_masks_new[fi] = masks
                    rgb = self._load_frame(fi)
                    if rgb is None:
                        self._progress_signal.emit(fi + 1, total_frames)
                        continue
                    h, w = rgb.shape[:2]
                    label_arr = _build_label_arr(masks, h, w)
                    df_row = _measure(label_arr, rgb)
                    if not df_row.empty:
                        df_row.insert(0, 'frame', fi + 1)
                        df_row.insert(1, 'file', self._frame_name(fi))
                        all_rows.append(df_row)
                    self._preview_signal.emit(fi, masks, rgb)
                    self._progress_signal.emit(fi + 1, total_frames)
                    continue

                rgb = self._load_frame(fi)
                if rgb is None:
                    self._progress_signal.emit(fi + 1, total_frames)
                    continue

                masks, _ = tracker.track_frame(rgb)
                frame_masks_new[fi] = masks

                h, w = rgb.shape[:2]
                label_arr = _build_label_arr(masks, h, w)
                df_row = _measure(label_arr, rgb)
                if not df_row.empty:
                    df_row.insert(0, 'frame', fi + 1)
                    df_row.insert(1, 'file', self._frame_name(fi))
                    all_rows.append(df_row)

                self._preview_signal.emit(fi, masks, rgb)
                self._progress_signal.emit(fi + 1, total_frames)

        self._frame_masks = frame_masks_new

        if all_rows:
            return pd.concat(all_rows, ignore_index=True)
        return pd.DataFrame()

    def _analyze_manual_mode_memory(self) -> pd.DataFrame | None:
        """Memory-attention tracking using SAM2VideoSession."""
        from .video_session import SAM2VideoSession

        if not self._keyframes:
            return None

        variant = self._combo_model.currentText() or 'tiny'
        model_paths = _model_manager.get_video_model_paths(variant)
        # Convert Path → str
        model_paths = {k: str(v) for k, v in model_paths.items()}

        video_session = SAM2VideoSession(
            model_paths, cancel_check=lambda: self._cancel_analysis)

        total_frames = self._total_frames
        all_rows = []
        frame_masks_new: dict[int, dict[int, np.ndarray]] = {}

        # Find the first keyframe to initialize objects
        sorted_kfs = sorted(self._keyframes.keys())
        first_kf = sorted_kfs[0]

        # Process frames sequentially — initialize on keyframes, propagate otherwise
        for fi in range(total_frames):
            if self._cancel_analysis:
                return None

            rgb = self._load_frame(fi)
            if rgb is None:
                self._progress_signal.emit(fi + 1, total_frames)
                continue

            video_session.set_image(rgb)

            if fi in self._keyframes:
                # Initialize (or re-initialize) objects from keyframe masks
                kf_masks = self._keyframes[fi]
                masks: dict[int, np.ndarray] = {}
                for oid, mask in kf_masks.items():
                    if not np.any(mask > 0):
                        continue
                    # Use box + centroid for initialization (preserves scale)
                    ys, xs = np.where(mask > 0)
                    x1, y1 = int(xs.min()), int(ys.min())
                    x2, y2 = int(xs.max()), int(ys.max())
                    cx, cy = int(xs.mean()), int(ys.mean())
                    coords = np.array([[x1, y1], [x2, y2], [cx, cy]],
                                      dtype=np.int32)
                    labels = np.array([2, 3, 1], dtype=np.int32)
                    pred_mask = video_session.initialize_object(
                        oid, coords, labels)
                    masks[oid] = pred_mask
                frame_masks_new[fi] = masks
            elif fi >= first_kf:
                # Propagate using memory attention
                masks, _scores = video_session.propagate()
                frame_masks_new[fi] = masks
            else:
                # Before first keyframe — skip
                self._progress_signal.emit(fi + 1, total_frames)
                continue

            h, w = rgb.shape[:2]
            label_arr = _build_label_arr(masks, h, w)
            df_row = _measure(label_arr, rgb)
            if not df_row.empty:
                df_row.insert(0, 'frame', fi + 1)
                df_row.insert(1, 'file', self._frame_name(fi))
                all_rows.append(df_row)

            self._preview_signal.emit(fi, masks, rgb)
            self._progress_signal.emit(fi + 1, total_frames)

        self._frame_masks = frame_masks_new

        if all_rows:
            return pd.concat(all_rows, ignore_index=True)
        return pd.DataFrame()

    def _analyze_cellpose_mode(self) -> pd.DataFrame | None:
        """Automatic segmentation using Cellpose ONNX (no manual annotation needed)."""
        try:
            from .cellpose import CellposeONNX
        except ImportError:
            logger.error("cellpose_onnx module not found")
            return None

        if not self._total_frames:
            return None

        model_name = 'nuclei'  # Default to nuclei segmentation

        self._progress_signal.emit(f"Downloading {model_name} model (first time only)…")
        try:
            model_path = CellposeONNX.download_model(model_name)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return None

        self._progress_signal.emit(f"Loading Cellpose {model_name}…")
        try:
            model = CellposeONNX(model_path, gpu=True)
        except Exception as e:
            logger.warning(f"GPU unavailable, using CPU: {e}")
            try:
                model = CellposeONNX(model_path, gpu=False)
            except Exception:
                return None

        total_frames = self._total_frames
        all_rows = []
        frame_masks_new: dict[int, dict[int, np.ndarray]] = {}
        diameter = 30  # Default nuclei diameter

        for fi in range(total_frames):
            if self._cancel_analysis:
                return None

            rgb = self._load_frame(fi)
            if rgb is None:
                self._progress_signal.emit(fi + 1, total_frames)
                continue

            # Run Cellpose
            try:
                if rgb.ndim == 3 and rgb.shape[2] == 3:
                    gray = rgb.mean(axis=2).astype(np.uint8)
                else:
                    gray = rgb if rgb.ndim == 2 else rgb[:, :, 0]

                masks, _flows = model.predict(gray, diameter=diameter)

                # Convert label image to dict of masks
                masks_dict = {}
                for label_id in np.unique(masks):
                    if label_id == 0:  # Skip background
                        continue
                    masks_dict[label_id] = (masks == label_id).astype(np.uint8) * 255

                frame_masks_new[fi] = masks_dict

                # Measure each object
                from skimage.measure import regionprops
                props = regionprops(masks, intensity_image=rgb if rgb.ndim == 2 else gray)

                for prop in props:
                    row = {
                        'label': prop.label,
                        'frame': fi + 1,
                        'file': self._frame_name(fi),
                        'area': prop.area,
                        'perimeter': prop.perimeter,
                        'eccentricity': prop.eccentricity,
                        'solidity': prop.solidity,
                        'mean_intensity': prop.mean_intensity,
                        'centroid_y': prop.centroid[0],
                        'centroid_x': prop.centroid[1],
                    }
                    all_rows.append(row)

                self._preview_signal.emit(fi, masks_dict, rgb)
                self._progress_signal.emit(fi + 1, total_frames)
            except Exception as e:
                logger.error(f"Cellpose error on frame {fi}: {e}")
                self._progress_signal.emit(f"Frame {fi}: error")

        self._frame_masks = frame_masks_new

        if all_rows:
            return pd.concat([pd.DataFrame(all_rows)], ignore_index=True)
        return pd.DataFrame()

    def _analyze_text_mode(self) -> pd.DataFrame | None:
        """Per-frame GroundingDINO + SAM2 detection."""
        from .grounding import GroundingDINOSession

        text_prompt = self._le_text_prompt.text().strip()
        score_threshold = self._spin_gdino.value()
        variant = self._combo_model.currentText() or 'tiny'

        if not text_prompt.strip():
            return None

        # Load models
        gdino_model = _model_manager.get_gdino_model_path()
        gdino_tokenizer = _model_manager.get_gdino_tokenizer_path()
        gdino = GroundingDINOSession(str(gdino_model), str(gdino_tokenizer))

        enc_path, dec_path = _model_manager.get_model_paths(variant)
        session = SAM2ImageSession(str(enc_path), str(dec_path))

        total_frames = self._total_frames
        all_rows = []
        frame_masks_new: dict[int, dict[int, np.ndarray]] = {}

        for fi in range(total_frames):
            if self._cancel_analysis:
                return None

            # Use correction keyframe if available
            if fi in self._keyframes:
                masks = self._keyframes[fi]
                frame_masks_new[fi] = masks
                rgb = self._load_frame(fi)
                if rgb is None:
                    self._progress_signal.emit(fi + 1, total_frames)
                    continue
                h, w = rgb.shape[:2]
                label_arr = _build_label_arr(masks, h, w)
                df_row = _measure(label_arr, rgb)
                if not df_row.empty:
                    df_row.insert(0, 'frame', fi + 1)
                    df_row.insert(1, 'file', self._frame_name(fi))
                    all_rows.append(df_row)
                self._preview_signal.emit(fi, masks, rgb)
                self._progress_signal.emit(fi + 1, total_frames)
                continue

            rgb = self._load_frame(fi)
            if rgb is None:
                self._progress_signal.emit(fi + 1, total_frames)
                continue

            # GroundingDINO detect
            detections = gdino.detect(rgb, text_prompt,
                                      score_threshold=score_threshold)
            if not detections:
                self._preview_signal.emit(fi, {}, rgb)
                self._progress_signal.emit(fi + 1, total_frames)
                continue

            # Category-grouped sorting
            categories = []
            for d in detections:
                if d.label not in categories:
                    categories.append(d.label)
            detections = sorted(
                detections,
                key=lambda d: (categories.index(d.label), -d.score))

            # SAM2 refine each detection
            session.set_image(rgb)
            masks: dict[int, np.ndarray] = {}
            det_labels: dict[int, str] = {}
            for idx, det in enumerate(detections, start=1):
                mask, _ = session.predict_box(det.box_xyxy, label_id=idx)
                masks[idx] = mask
                det_labels[idx] = det.label

            frame_masks_new[fi] = masks

            h, w = rgb.shape[:2]
            label_arr = _build_label_arr(masks, h, w)
            df_row = _measure(label_arr, rgb)
            if not df_row.empty:
                df_row.insert(0, 'frame', fi + 1)
                df_row.insert(1, 'file', self._frame_name(fi))
                # Add category column
                df_row['category'] = df_row['label'].map(
                    lambda lid: det_labels.get(lid, ''))
                all_rows.append(df_row)

            self._preview_signal.emit(fi, masks, rgb)
            self._progress_signal.emit(fi + 1, total_frames)

        self._frame_masks = frame_masks_new

        if all_rows:
            return pd.concat(all_rows, ignore_index=True)
        return pd.DataFrame()

    def _on_preview(self, frame_idx: int, masks: dict, rgb_arr: np.ndarray):
        """Live preview: update slider + show masks on canvas during analysis."""
        if not self._analyzing:
            return
        # Update slider position without triggering _on_frame_changed
        self._slider.blockSignals(True)
        self._spin_frame.blockSignals(True)
        self._slider.setValue(frame_idx + 1)
        self._spin_frame.setValue(frame_idx + 1)
        self._slider.blockSignals(False)
        self._spin_frame.blockSignals(False)
        self._current_frame_idx = frame_idx

        # Display the frame image (skip_masks=True: _frame_masks not yet updated)
        self._apply_frame(rgb_arr, skip_masks=True)
        self._rgb_arr = rgb_arr

        # Show tracked masks overlay from the worker directly
        if masks:
            self._view.update_masks(masks)

    def _on_progress(self, current: int, total: int):
        self._status_label.setText(f"Processing frame {current}/{total}")
        if self.node:
            pct = int(current / max(1, total) * 100)
            self.node.set_progress(pct)

    def _on_done(self, result):
        self._analyzing = False
        self._btn_analyze.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._corrections_pending = False

        if result is None:
            if self._cancel_analysis:
                self._status_label.setText("Analysis cancelled")
            else:
                self._status_label.setText("Analysis failed")
            return

        self._result_df = result
        n_rows = len(result) if result is not None else 0
        n_frames = len(self._frame_masks)
        self._status_label.setText(
            f"Done: {n_rows} measurements across {n_frames} frames")

        if self.node:
            self.node.set_progress(100)

        # Refresh current frame to show tracked overlay
        fi = self._current_frame_idx
        if fi in self._frame_masks:
            self._view.clear_mask()
            self._view.update_masks(self._frame_masks[fi])

        # Trigger node evaluate to push outputs
        if self.node:
            self.node.mark_dirty()

    # ── NodeBaseWidget interface ───────────────────────────────────────
    def get_value(self) -> str:
        return json.dumps({
            "folder_path": self._le_folder.text(),
            "video_path": self._le_video.text(),
            "pattern": self._le_pattern.text(),
            "model_variant": self._combo_model.currentText(),
            "track_method": self._combo_track.currentText(),
            "analysis_mode": self._combo_mode.currentText(),
            "text_prompt": self._le_text_prompt.text(),
            "gdino_score": self._spin_gdino.value(),
            "current_frame": self._current_frame_idx,
        })

    def set_value(self, value):
        if not value:
            return
        try:
            d = json.loads(value) if isinstance(value, str) else value
        except (json.JSONDecodeError, TypeError):
            return
        if d.get('folder_path'):
            self._le_folder.setText(d['folder_path'])
        if d.get('video_path'):
            self._le_video.setText(d['video_path'])
        if d.get('pattern'):
            self._le_pattern.setText(d['pattern'])
        if d.get('model_variant'):
            idx = self._combo_model.findText(d['model_variant'])
            if idx >= 0:
                self._combo_model.setCurrentIndex(idx)
        if d.get('track_method'):
            idx = self._combo_track.findText(d['track_method'])
            if idx >= 0:
                self._combo_track.setCurrentIndex(idx)
        if d.get('analysis_mode'):
            idx = self._combo_mode.findText(d['analysis_mode'])
            if idx >= 0:
                self._combo_mode.setCurrentIndex(idx)
        if d.get('text_prompt'):
            self._le_text_prompt.setText(d['text_prompt'])
        if 'gdino_score' in d:
            self._spin_gdino.setValue(float(d['gdino_score']))


# ---------------------------------------------------------------------------
# SAM2VideoAnalyzeNode
# ---------------------------------------------------------------------------

class SAM2VideoAnalyzeNode(BaseImageProcessNode):
    """Integrated video analysis: annotate + track + measure in one node.

    1. Select an image folder and browse frames with the slider
    2. Annotate objects on any frame (click, auto, or text)
    3. Click "Analyze All" to track and measure across all frames
    4. Correct mistakes on any frame, then re-analyze

    Keywords: SAM2, video, analyze, track, timelapse, measure, regionprops,
              影片, 分析, 追蹤, 量測, 時間序列
    """

    __identifier__ = 'plugins.Plugins.VideoAnalysis'
    NODE_NAME      = 'SAM2 Video Analyze'
    PORT_SPEC      = {'inputs': [], 'outputs': ['table', 'image', 'image']}

    _UI_PROPS = frozenset({
        'color', 'pos', 'selected', 'name', 'progress',
        'show_preview', 'live_preview',
    })

    def __init__(self):
        super().__init__()

        # ── ports ──────────────────────────────────────────────────────
        self.add_output('table', color=PORT_COLORS['table'])
        self.add_output('overlay', color=PORT_COLORS['image'])
        self.add_output('trajectory', color=PORT_COLORS['image'])

        # ── main widget (contains all controls) ──────────────────────
        self._widget = _VideoAnalyzeWidget(self.view)
        self._widget._node_ref = self
        self.add_custom_widget(self._widget)

        # ── SAM2 session ──────────────────────────────────────────────
        self._session: SAM2ImageSession | None = None
        self._current_variant: str | None = None
        self._session_lock = threading.Lock()

        # Pre-warm
        threading.Thread(target=self._prewarm, daemon=True).start()

    @property
    def _analyze_widget(self) -> _VideoAnalyzeWidget:
        return self._widget

    def _prewarm(self):
        try:
            self._ensure_session('tiny')
        except Exception:
            logger.debug("SAM2 Video Analyze pre-warm failed", exc_info=True)

    def _ensure_session(self, variant: str) -> SAM2ImageSession:
        with self._session_lock:
            if self._session is not None and self._current_variant == variant:
                return self._session
            logger.info("Loading SAM2 model '%s' for video analyze ...", variant)
            # Release old session to free GPU/CPU memory
            old = self._session
            self._session = None
            del old
            enc_path, dec_path = _model_manager.get_model_paths(variant)
            self._session = SAM2ImageSession(str(enc_path), str(dec_path))
            self._current_variant = variant
            self._analyze_widget._session = self._session
            self._analyze_widget._encoded_frame_idx = -1
            return self._session

    def evaluate(self):
        self.reset_progress()

        # Push cached results if available
        w = self._analyze_widget
        if w._result_df is not None and not w._result_df.empty:
            self.output_values['table'] = TableData(payload=w._result_df)

        # Build overlay for current frame
        fi = w._current_frame_idx
        masks = w._frame_masks.get(fi) or w._keyframes.get(fi)
        if masks and w._rgb_arr is not None:
            rgb = w._rgb_arr
            vis = rgb.astype(np.float32).copy()
            for obj_id in sorted(masks.keys()):
                m = masks[obj_id]
                while m.ndim > 2:
                    m = m[0]
                if m.shape[0] != rgb.shape[0] or m.shape[1] != rgb.shape[1]:
                    continue
                mask_bool = m > 0
                if not np.any(mask_bool):
                    continue
                color = np.array(_obj_color(obj_id), dtype=np.float32)
                vis[mask_bool] = vis[mask_bool] * 0.5 + color * 0.5
            vis = np.clip(vis, 0, 255).astype(np.uint8)
            self.output_values['overlay'] = ImageData(
                payload=Image.fromarray(vis, mode='RGB'))

        # Build trajectory graph
        if w._result_df is not None and not w._result_df.empty:
            bg = w._rgb_arr  # Use current frame as background
            img_sz = None
            if bg is not None:
                img_sz = (bg.shape[1], bg.shape[0])
            traj_img = _render_trajectory(w._result_df, bg, img_sz)
            self.output_values['trajectory'] = ImageData(payload=traj_img)

        self.set_progress(100)
        self.mark_clean()
        return True, None
