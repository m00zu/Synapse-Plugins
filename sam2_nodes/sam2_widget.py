"""
widget.py — SAM2SegmentWidget (NodeBaseWidget) for interactive click-to-segment.

Supports multiple objects via a list below the canvas (like DrawShapeNode).
Toolbar: Include/Exclude toggle, Add Object, Undo, Clear, Fit.
"""
from __future__ import annotations

import json
import logging
import threading

import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap, QIcon
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget

from .viewer import SAM2ClickGraphicsView, _obj_color
from .engine import SAM2ImageSession

logger = logging.getLogger(__name__)


class SAM2SegmentWidget(NodeBaseWidget):
    """Embedded click-to-segment widget for SAM2SegmentNode."""

    _img_signal = Signal(object)
    _setup_signal = Signal(object, object, str)  # (rgb_arr, session, saved_state)
    _auto_done_signal = Signal(object)  # dict[int, np.ndarray] from auto-segment
    masks_updated = Signal(object)  # dict[int, np.ndarray]

    _VIEW_MAX = 560
    _VIEW_MIN = 200
    _CHROME_H = 250  # toolbar + list + tips

    def __init__(self, parent=None, name='', label=''):
        super().__init__(parent, name, label)

        container = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(container)
        vlay.setContentsMargins(4, 2, 4, 2)
        vlay.setSpacing(3)

        # ── toolbar ──────────────────────────────────────────────────
        tb = QtWidgets.QHBoxLayout()
        tb.setSpacing(4)

        self._btn_mode = QtWidgets.QPushButton("+Include")
        self._btn_mode.setCheckable(True)
        self._btn_mode.setChecked(True)
        self._btn_mode.setFixedHeight(24)
        self._btn_mode.setToolTip("Toggle: Include (foreground) / Exclude (background)")
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
        self._btn_auto.setToolTip(
            "Auto-segment: find all objects via a grid of point prompts")
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
        vlay.addLayout(tb)

        # ── graphics view ────────────────────────────────────────────
        self._scene = QtWidgets.QGraphicsScene()
        self._view = SAM2ClickGraphicsView(self._scene)
        self._view.setMinimumSize(self._VIEW_MIN, self._VIEW_MIN)
        self._view.setFixedSize(self._VIEW_MAX, self._VIEW_MAX)
        self._view.setStyleSheet("background:#1a1a1a;")
        vlay.addWidget(self._view)

        # ── object list ──────────────────────────────────────────────
        list_row = QtWidgets.QHBoxLayout()
        list_row.setSpacing(4)

        self._obj_list = QtWidgets.QListWidget()
        self._obj_list.setMaximumHeight(120)
        self._obj_list.setMinimumHeight(36)
        self._obj_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        list_row.addWidget(self._obj_list)

        btn_col = QtWidgets.QVBoxLayout()
        btn_col.setSpacing(3)
        self._btn_add_obj = QtWidgets.QPushButton("+")
        self._btn_add_obj.setFixedSize(40, 40)
        self._btn_add_obj.setToolTip("Add a new object")
        btn_col.addWidget(self._btn_add_obj)

        self._btn_del_obj = QtWidgets.QPushButton("-")
        self._btn_del_obj.setFixedSize(40, 40)
        self._btn_del_obj.setToolTip("Delete selected object")
        btn_col.addWidget(self._btn_del_obj)

        self._btn_clear_all = QtWidgets.QPushButton("X")
        self._btn_clear_all.setFixedSize(40, 24)
        self._btn_clear_all.setToolTip("Clear all objects and masks")
        self._btn_clear_all.setStyleSheet(
            "QPushButton { color:#c55; font-weight:bold; }"
            "QPushButton:hover { background:#4a2020; }")
        btn_col.addWidget(self._btn_clear_all)
        btn_col.addStretch()
        list_row.addLayout(btn_col)
        vlay.addLayout(list_row)

        tip = QtWidgets.QLabel(
            "Click to place points \u00b7 Ctrl+Z = undo \u00b7 "
            "Middle-click = pan \u00b7 Scroll = zoom")
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tip.setStyleSheet("color:#999; font-size:9px; padding:1px;")
        tip.setWordWrap(True)
        vlay.addWidget(tip)

        self._container = container
        self.set_custom_widget(self._container)

        # ── state ────────────────────────────────────────────────────
        self._session: SAM2ImageSession | None = None
        self._rgb_arr: np.ndarray | None = None

        # ── connections ──────────────────────────────────────────────
        self._btn_mode.toggled.connect(self._on_mode_toggled)
        self._btn_add_obj.clicked.connect(self._on_add_object)
        self._btn_del_obj.clicked.connect(self._on_delete_object)
        self._btn_clear_all.clicked.connect(self._on_clear_all)
        self._btn_undo.clicked.connect(self._on_undo)
        self._btn_clear.clicked.connect(self._on_clear)
        self._btn_fit.clicked.connect(self._on_fit)
        self._btn_auto.clicked.connect(self._on_auto_segment)
        self._view.points_changed.connect(self._on_points_changed)
        self._obj_list.currentItemChanged.connect(self._on_list_selection)
        self._img_signal.connect(self._apply_image,
                                 Qt.ConnectionType.QueuedConnection)
        self._setup_signal.connect(self._deferred_setup,
                                   Qt.ConnectionType.QueuedConnection)
        self._auto_done_signal.connect(self._on_auto_done,
                                       Qt.ConnectionType.QueuedConnection)

        # Build initial list
        self._rebuild_obj_list()

    # ── mode toggle ──────────────────────────────────────────────────

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

    # ── object list ──────────────────────────────────────────────────

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
            # Append score if available
            if self._session is not None and oid in self._session.scores:
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
            if self._session is not None and oid in self._session.scores:
                self._score_label.setText(f"Score: {self._session.scores[oid]:.2f}")
            else:
                self._score_label.setText("")

    def _on_add_object(self):
        self._view.add_object()
        self._rebuild_obj_list()

    def _on_delete_object(self):
        oid = self._view.current_obj
        self._view.delete_object(oid)
        # Remove just this object's mask and score (don't re-run all predictions,
        # which would wipe auto-generated masks that have no points).
        if self._session is not None:
            self._session._masks.pop(oid, None)
            self._session._scores.pop(oid, None)
        self._rebuild_obj_list()
        self._refresh_mask_overlay()
        self._emit()

    def _on_clear_all(self):
        """Remove all objects, points, and masks."""
        self._view.clear_all()
        if self._session is not None:
            self._session._masks.clear()
            self._session._scores.clear()
        self._rebuild_obj_list()
        self._refresh_mask_overlay()
        self._score_label.setText("")
        self._emit()

    # ── toolbar actions ──────────────────────────────────────────────

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

    # ── auto-segment ──────────────────────────────────────────────────

    def _on_auto_segment(self):
        if self._session is None or not self._session.is_image_set:
            return
        # Read settings from node properties
        grid = 16
        score_thr = 0.85
        min_pct = 0.1
        max_pct = 50.0
        if self.node:
            grid = int(self.node.get_property('auto_grid') or 16)
            score_thr = float(self.node.get_property('auto_score') or 0.85)
            min_pct = float(self.node.get_property('auto_min_area') or 0.1)
            max_pct = float(self.node.get_property('auto_max_area') or 50.0)
        self._auto_params = (grid, score_thr, min_pct / 100.0, max_pct / 100.0)
        self._btn_auto.setEnabled(False)
        self._btn_auto.setText("Running...")
        self._score_label.setText(f"Auto-segmenting ({grid}x{grid})...")
        if self.node:
            self.node.reset_progress()
            self.node.set_progress(1)
        t = threading.Thread(target=self._auto_worker, daemon=True)
        t.start()

    def _auto_progress(self, frac: float):
        """Called from auto_segment worker thread with progress 0..1."""
        if self.node:
            self.node.set_progress(int(frac * 95) + 1)

    def _auto_worker(self):
        try:
            grid, score_thr, min_frac, max_frac = self._auto_params
            self._session.auto_segment(
                points_per_side=grid, score_threshold=score_thr,
                nms_iou_threshold=0.5,
                min_area_frac=min_frac, max_area_frac=max_frac,
                progress_cb=self._auto_progress)
            self._auto_done_signal.emit(self._session.masks)
        except Exception:
            logger.exception("Auto-segment failed")
            self._auto_done_signal.emit({})

    def _on_auto_done(self, masks: dict):
        """Runs on main thread after auto-segment completes."""
        if self.node:
            self.node.set_progress(100)
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
        self._emit()

    # ── image + session ──────────────────────────────────────────────

    def set_session(self, session: SAM2ImageSession):
        self._session = session

    def load_image(self, rgb_arr: np.ndarray):
        self._rgb_arr = rgb_arr
        if threading.current_thread() is threading.main_thread():
            self._apply_image(rgb_arr)
        else:
            self._img_signal.emit(rgb_arr)

    def _apply_image(self, rgb_arr: np.ndarray):
        h, w = rgb_arr.shape[:2]
        if w >= h:
            vw = self._VIEW_MAX
            vh = max(self._VIEW_MIN, int(self._VIEW_MAX * h / w))
        else:
            vh = self._VIEW_MAX
            vw = max(self._VIEW_MIN, int(self._VIEW_MAX * w / h))
        self._view.setFixedSize(vw, vh)
        self._container.setFixedSize(vw + 8, vh + self._CHROME_H)

        self._view.load_image(rgb_arr)
        self._rebuild_obj_list()

        if self.widget():
            self.widget().adjustSize()
        if self.node and hasattr(self.node, 'view') and \
                hasattr(self.node.view, 'draw_node'):
            self.node.view.draw_node()

    def setup_from_worker(self, rgb_arr: np.ndarray,
                          session: SAM2ImageSession, saved: str):
        """Thread-safe: called from worker thread, defers to main thread."""
        self._session = session
        self._rgb_arr = rgb_arr
        self._setup_signal.emit(rgb_arr, session, saved)

    def _deferred_setup(self, rgb_arr: np.ndarray,
                        session: SAM2ImageSession, saved: str):
        """Runs on main thread — safe for all Qt widget operations."""
        self._session = session
        self._apply_image(rgb_arr)
        if saved:
            self.set_value(saved)
            self.run_predict()

    # ── decode on point change ───────────────────────────────────────

    def _on_points_changed(self, obj_id: int):
        if self._session is None or not self._session.is_image_set:
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
        self._emit()

    def _run_all_predicts(self):
        if self._session is None or not self._session.is_image_set:
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
        self._emit()

    def _refresh_mask_overlay(self):
        if self._session is None:
            return
        masks = {k: v for k, v in self._session.masks.items()
                 if np.any(v > 0)}
        if masks:
            self._view.update_masks(masks)
        else:
            self._view.clear_mask()
        self.masks_updated.emit(self._session.masks)

    def run_predict(self):
        self._run_all_predicts()

    # ── NodeBaseWidget interface ─────────────────────────────────────

    def _emit(self):
        self.value_changed.emit(self.get_name(), self.get_value())

    def get_value(self) -> str:
        return json.dumps({"objects": {
            str(k): v for k, v in self._view.get_all_objects().items()
        }})

    def set_value(self, value: str):
        if not value:
            return
        try:
            d = json.loads(value)
            if "objects" in d:
                self._view.restore_objects(d["objects"])
                self._rebuild_obj_list()
            elif "points" in d:
                pts = d.get("points", [])
                labs = d.get("labels", [])
                if pts:
                    self._view.restore_objects({"1": {"points": pts, "labels": labs}})
                    self._rebuild_obj_list()
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
