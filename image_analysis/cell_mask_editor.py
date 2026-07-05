"""
image_analysis/cell_mask_editor.py
==================================
Cell Mask Editor -- draw one boolean mask per cell as independent, possibly
OVERLAPPING layers, and emit them as a CollectionData of per-cell MaskData.

Reuses the drawing canvas + tools from roi_nodes (_LabelEditorView). Unlike
LabelEditor (a single int32 partition where a pixel has one label), each cell
here is its own boolean layer, so two cells may share pixels.
"""
from __future__ import annotations

import threading
import numpy as np

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PIL import Image as _PIL, ImageDraw

from NodeGraphQt.widgets.node_widgets import NodeBaseWidget
from data_models import ImageData, MaskData, CollectionData
from nodes.base import BaseExecutionNode, PORT_COLORS

try:
    from .roi_nodes import _LabelEditorView
except ImportError:  # pragma: no cover - flat-load fallback
    from roi_nodes import _LabelEditorView
try:
    from .vision_nodes import _label_palette
except ImportError:  # pragma: no cover
    from vision_nodes import _label_palette


def _to_display_u8(arr: np.ndarray) -> np.ndarray:
    """Normalise an image array to uint8 for on-canvas display, PRESERVING
    colour channels (RGB stays RGB, grayscale stays 2-D)."""
    a = arr.astype(np.float32)
    mx = float(a.max()) if a.size else 1.0
    if mx <= 1.0:
        a = a * 255.0
    elif mx > 255.0:
        a = a / mx * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


class CellMaskEditorWidget(NodeBaseWidget):
    """Layered boolean-mask editor. One layer per cell; layers may overlap."""

    masks_changed = Signal()
    _ui_refresh_signal = Signal()   # -> main thread (no payload; data already applied)

    def __init__(self, parent=None):
        super().__init__(parent, 'cell_mask_editor')

        root = QtWidgets.QVBoxLayout()
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(3)
        container = QtWidgets.QWidget()
        container.setLayout(root)

        # Tool bar: 5 tools + boolean op.
        tb = QtWidgets.QHBoxLayout(); tb.setSpacing(3)
        self._tool_group = QtWidgets.QButtonGroup(container)
        self._tool_names = ['rect', 'ellipse', 'polygon', 'lasso', 'brush']
        for i, label in enumerate(['Rect', 'Ellipse', 'Polygon', 'Lasso', 'Brush']):
            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True); btn.setFixedHeight(24); btn.setMinimumWidth(45)
            self._tool_group.addButton(btn, i); tb.addWidget(btn)
            if i == 0:
                btn.setChecked(True)
        tb.addSpacing(6); tb.addWidget(QtWidgets.QLabel('Op:'))
        self._op_combo = QtWidgets.QComboBox()
        self._op_combo.addItems(['Union', 'Subtract'])
        tb.addWidget(self._op_combo); tb.addStretch()
        root.addLayout(tb)

        # Brush size.
        br = QtWidgets.QHBoxLayout(); br.setSpacing(3)
        br.addWidget(QtWidgets.QLabel('Brush:'))
        self._brush_spin = QtWidgets.QSpinBox(); self._brush_spin.setRange(1, 50)
        self._brush_spin.setValue(5); self._brush_spin.setFixedWidth(50)
        self._brush_spin.valueChanged.connect(lambda v: self._view.set_brush_size(v))
        br.addWidget(self._brush_spin); br.addStretch()
        root.addLayout(br)

        # Canvas (reused from roi_nodes).
        self._view = _LabelEditorView(); self._view.setMinimumSize(640, 512)
        root.addWidget(self._view)

        # Cell (layer) list + controls.
        lrow = QtWidgets.QHBoxLayout(); lrow.setSpacing(3)
        self._cell_list = QtWidgets.QListWidget(); self._cell_list.setFixedHeight(80)
        self._cell_list.currentRowChanged.connect(self._on_cell_selected)
        lrow.addWidget(self._cell_list)
        col = QtWidgets.QVBoxLayout(); col.setSpacing(3)
        add_btn = QtWidgets.QPushButton('+ New Cell'); add_btn.setFixedHeight(22)
        add_btn.clicked.connect(self._add_cell); col.addWidget(add_btn)
        del_btn = QtWidgets.QPushButton('Delete'); del_btn.setFixedHeight(22)
        del_btn.clicked.connect(self._delete_cell); col.addWidget(del_btn)
        clr_btn = QtWidgets.QPushButton('Clear All'); clr_btn.setFixedHeight(22)
        clr_btn.clicked.connect(self._clear); col.addWidget(clr_btn)
        col.addStretch(); lrow.addLayout(col)
        root.addLayout(lrow)

        self.set_custom_widget(container)

        # State.
        self._layers: dict[int, np.ndarray] = {}   # cell_id -> bool[H,W]
        self._current = 1
        self._img_h, self._img_w = 1, 1
        self._bg = None
        self._opacity = 0.5

        self._tool_group.idClicked.connect(
            lambda i: self._view.set_tool(self._tool_names[i]))
        self._view.shape_committed.connect(self._on_shape_committed)
        self._ui_refresh_signal.connect(self._refresh_ui,
                                        Qt.ConnectionType.QueuedConnection)

    # NodeBaseWidget contract (no serialised scalar value).
    def get_value(self):        return ''
    def set_value(self, _v):    pass

    # ── public API ────────────────────────────────────────────────────────
    def set_input(self, bg_arr):
        """Thread-safe: evaluate() may run on a worker thread.

        The data state (self._bg / self._img_h / self._img_w / self._layers /
        self._current) is always updated SYNCHRONOUSLY, on whatever thread this
        is called from, so that callers reading get_masks() right after this
        call (e.g. evaluate()'s subsequent _emit()) see the post-reset state.
        Only the Qt UI refresh is marshaled to the main thread.
        """
        self._update_state(bg_arr)
        if threading.current_thread() is not threading.main_thread():
            self._ui_refresh_signal.emit()
        else:
            self._refresh_ui()

    def _update_state(self, bg_arr):
        """Pure-data update: safe to call from any thread (no Qt calls here)."""
        if bg_arr is None:
            return
        h, w = bg_arr.shape[:2]
        shape_changed = (h != self._img_h or w != self._img_w)
        content_changed = (self._bg is None or self._bg.shape != bg_arr.shape
                           or not np.array_equal(self._bg, bg_arr))
        self._bg = bg_arr
        self._img_h, self._img_w = h, w
        if shape_changed:
            # Different image geometry -> old layers are meaningless; reset.
            self._layers = {}
            self._current = 1
        elif content_changed and self._layers:
            # Same size, different image -> keep no stale drawings.
            self._layers = {}
            self._current = 1

    def _refresh_ui(self):
        """Qt-only refresh; must run on the main thread."""
        self._refresh_cell_list()
        self._render()

    def set_layers(self, layers: dict):
        """Replace all layers (used programmatically / in tests)."""
        self._layers = {int(k): np.array(v, dtype=bool, copy=True) for k, v in layers.items()}
        self._current = min(self._layers) if self._layers else 1
        self._refresh_cell_list()
        self._render()
        self.masks_changed.emit()

    def get_masks(self) -> dict:
        """Return {cell_id: bool[H,W]} for every non-empty layer."""
        return {cid: m for cid, m in sorted(self._layers.items()) if m.any()}

    # ── drawing ───────────────────────────────────────────────────────────
    def _blank(self):
        return np.zeros((self._img_h, self._img_w), dtype=bool)

    def _on_shape_committed(self, tool: str, pts: list):
        if self._bg is None:
            return
        self._layers.setdefault(self._current, self._blank())
        if tool == 'brush':
            shape = self._brush_mask(pts)
        else:
            if len(pts) < 2:
                return
            shape = self._rasterize(tool, pts)
        if shape is None:
            return
        op = self._op_combo.currentText()
        if op == 'Union':
            self._layers[self._current] |= shape
        else:  # Subtract
            self._layers[self._current] &= ~shape
        self._refresh_cell_list()
        self._render()
        self.masks_changed.emit()

    def _rasterize(self, tool: str, pts: list):
        img = _PIL.new('L', (self._img_w, self._img_h), 0)
        draw = ImageDraw.Draw(img)
        if tool in ('rect', 'ellipse'):
            x0 = min(pts[0][0], pts[1][0]); y0 = min(pts[0][1], pts[1][1])
            x1 = max(pts[0][0], pts[1][0]); y1 = max(pts[0][1], pts[1][1])
            if x1 <= x0 or y1 <= y0:
                return None
            (draw.ellipse if tool == 'ellipse' else draw.rectangle)(
                [x0, y0, x1, y1], fill=255)
        elif tool in ('polygon', 'lasso'):
            if len(pts) < 3:
                return None
            draw.polygon([(float(x), float(y)) for x, y in pts], fill=255)
        return np.array(img) > 0

    def _brush_mask(self, pts: list):
        m = self._blank()
        r = max(1, self._brush_spin.value() // 2)
        h, w = self._img_h, self._img_w
        for (px, py) in pts:
            y0 = max(0, py - r); y1 = min(h, py + r + 1)
            x0 = max(0, px - r); x1 = min(w, px + r + 1)
            m[y0:y1, x0:x1] = True
        return m

    # ── cell list ─────────────────────────────────────────────────────────
    def _add_cell(self):
        nxt = (max(self._layers) + 1) if self._layers else 1
        self._layers[nxt] = self._blank()
        self._current = nxt
        self._refresh_cell_list()

    def _delete_cell(self):
        row = self._cell_list.currentRow()
        if row < 0:
            return
        cid = self._cell_list.item(row).data(Qt.ItemDataRole.UserRole)
        self._layers.pop(int(cid), None)
        self._current = min(self._layers) if self._layers else 1
        self._refresh_cell_list(); self._render(); self.masks_changed.emit()

    def _clear(self):
        self._layers = {}; self._current = 1
        self._refresh_cell_list(); self._render(); self.masks_changed.emit()

    def _on_cell_selected(self, row):
        if row < 0:
            return
        self._current = int(self._cell_list.item(row).data(Qt.ItemDataRole.UserRole))

    def _refresh_cell_list(self):
        self._cell_list.blockSignals(True)
        self._cell_list.clear()
        for cid in sorted(self._layers):
            px = int(self._layers[cid].sum())
            item = QtWidgets.QListWidgetItem(f'  Cell {cid}  ({px:,} px)')
            item.setData(Qt.ItemDataRole.UserRole, int(cid))
            self._cell_list.addItem(item)
            if cid == self._current:
                self._cell_list.setCurrentRow(self._cell_list.count() - 1)
        self._cell_list.blockSignals(False)

    # ── render ────────────────────────────────────────────────────────────
    def _color(self, cid: int):
        pal = _label_palette(max(cid, 1))
        return np.array(pal[(cid - 1) % len(pal)], dtype=np.float32)

    def _compose_canvas(self) -> np.ndarray:
        """Build the (H, W, 3) uint8 RGB canvas: colour-preserving background
        with each cell layer composited as a semi-transparent colour overlay."""
        disp = _to_display_u8(self._bg)
        if disp.ndim == 2:
            canvas = np.stack([disp] * 3, axis=-1).astype(np.float32)
        else:
            canvas = disp[..., :3].astype(np.float32)
        a = self._opacity
        for cid, m in sorted(self._layers.items()):
            if not m.any():
                continue
            canvas[m] = (1 - a) * canvas[m] + a * self._color(cid)
        return np.ascontiguousarray(np.clip(canvas, 0, 255).astype(np.uint8))

    def _render(self):
        if self._bg is None:
            return
        canvas = self._compose_canvas()
        h, w = canvas.shape[:2]
        qimg = QImage(canvas.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._view.set_pixmap(QPixmap.fromImage(qimg.copy()))


class CellMaskEditorNode(BaseExecutionNode):
    """
    Draw one boolean mask per cell as independent, possibly OVERLAPPING layers.
    Outputs a collection of per-cell masks (`cells`) plus the pass-through
    `image`. Feed both into a Crop Cells node.

    Unlike Label Editor (a single-label partition), a pixel here can belong to
    several cells -- required when cell bodies or filopodia overlap.

    Keywords: cell mask, overlap, per cell, draw, filopodia, 細胞遮罩, 重疊, 手動, 絲足
    """
    __identifier__ = 'nodes.image_process.Visualize'
    NODE_NAME      = 'Cell Mask Editor'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['cells', 'image']}

    _COLLECTION_COLOR = PORT_COLORS.get('collection', (218, 165, 32))

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('cells', color=self._COLLECTION_COLOR, multi_output=True)
        self.add_output('image', color=PORT_COLORS['image'], multi_output=True)
        self._editor = CellMaskEditorWidget(self.view)
        self._editor.masks_changed.connect(self._on_masks_changed)
        self.add_custom_widget(self._editor)

    def _read_image(self):
        port = self.inputs().get('image')
        if port and port.connected_ports():
            cp = port.connected_ports()[0]
            data = cp.node().output_values.get(cp.name())
            if isinstance(data, ImageData):
                return data
        return None

    def _on_masks_changed(self):
        self._emit(self._read_image())
        self.mark_dirty()

    def _emit(self, img):
        scale = getattr(img, 'scale_um', None) if img else None
        payload = {}
        for cid, m in self._editor.get_masks().items():
            payload[f'cell{cid}'] = MaskData(
                payload=(m.astype(np.uint8) * 255), bit_depth=8, scale_um=scale)
        self.output_values['cells'] = CollectionData(payload=payload)

    def evaluate(self):
        img = self._read_image()
        if not isinstance(img, ImageData):
            self.mark_error()
            return False, 'No image connected'
        self._editor.set_input(img.payload)
        self.output_values['image'] = img
        self._emit(img)
        self.mark_clean()
        return True, None
