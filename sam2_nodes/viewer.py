"""
viewer.py — QGraphicsView for SAM2 click-to-segment interaction.

Supports multiple objects, each with its own set of points and mask.
Left-click places a point (foreground or background depending on mode).
Middle-click + drag to pan.  Scroll to zoom.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsPixmapItem,
    QGraphicsScene, QGraphicsView,
)

# Distinct colours per object (R, G, B)
OBJ_COLORS = [
    (0, 200, 255),    # cyan
    (255, 140, 0),    # orange
    (0, 255, 120),    # green
    (255, 60, 200),   # magenta
    (255, 255, 0),    # yellow
    (120, 120, 255),  # blue
    (255, 160, 160),  # pink
    (160, 255, 160),  # light green
]


def _mouse_pos_qpoint(event):
    """Qt5/Qt6 compat helper for mouse position."""
    pos_fn = getattr(event, "position", None)
    if callable(pos_fn):
        return pos_fn().toPoint()
    return event.pos()


def _obj_color(obj_id: int) -> tuple[int, int, int]:
    return OBJ_COLORS[(obj_id - 1) % len(OBJ_COLORS)]


class SAM2ClickGraphicsView(QGraphicsView):
    """Image viewer where clicks place foreground/background points."""

    points_changed = Signal(int)  # emits current obj_id

    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._image_arr: np.ndarray | None = None
        self._bg_pixmap_item: QGraphicsPixmapItem | None = None
        self._mask_pixmap_item: QGraphicsPixmapItem | None = None

        # Per-object state: {obj_id: {"points": [(x,y),...], "labels": [0|1,...], "items": [QGraphicsEllipseItem,...]}}
        self._objects: dict[int, dict] = {}
        self._current_obj: int = 1
        self._next_obj: int = 2  # next available ID
        self._objects[1] = {"points": [], "labels": [], "items": []}

        # 1 = foreground (include), 0 = background (exclude)
        self._mode: int = 1

        self._scale: float = 1.0
        self._pan_start: QtCore.QPoint | None = None

    # ── public API ──────────────────────────────────────────────────────

    @property
    def current_obj(self) -> int:
        return self._current_obj

    @property
    def num_objects(self) -> int:
        return len(self._objects)

    @property
    def object_ids(self) -> list[int]:
        return sorted(self._objects.keys())

    def add_object(self) -> int:
        """Create a new object and make it current. Returns its ID."""
        oid = self._next_obj
        self._next_obj += 1
        self._objects[oid] = {"points": [], "labels": [], "items": []}
        self._current_obj = oid
        self._dim_non_current_points()
        return oid

    def set_current_object(self, obj_id: int):
        if obj_id in self._objects:
            self._current_obj = obj_id
            self._dim_non_current_points()

    def delete_object(self, obj_id: int):
        """Remove an object and all its points."""
        if obj_id not in self._objects:
            return
        for item in self._objects[obj_id]["items"]:
            self.scene().removeItem(item)
        del self._objects[obj_id]
        if not self._objects:
            self._objects[1] = {"points": [], "labels": [], "items": []}
            self._current_obj = 1
            self._next_obj = 2
        elif self._current_obj == obj_id:
            self._current_obj = sorted(self._objects.keys())[0]
        self._dim_non_current_points()

    def load_image(self, rgb_arr: np.ndarray):
        """Display an RGB uint8 array (H, W, 3) as background."""
        self._image_arr = rgb_arr
        h, w = rgb_arr.shape[:2]
        qimg = QImage(rgb_arr.data.tobytes(), w, h, 3 * w,
                      QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        if self._bg_pixmap_item is None:
            self._bg_pixmap_item = QGraphicsPixmapItem(pixmap)
            self._bg_pixmap_item.setZValue(-2)
            self.scene().addItem(self._bg_pixmap_item)
        else:
            self._bg_pixmap_item.setPixmap(pixmap)
        self.scene().setSceneRect(0, 0, w, h)
        self.fitInView(self._bg_pixmap_item,
                       Qt.AspectRatioMode.KeepAspectRatio)
        self.clear_all()

    def update_masks(self, masks: dict[int, np.ndarray]):
        """Overlay all object masks, each in its own colour."""
        if self._image_arr is None:
            return
        h, w = self._image_arr.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for obj_id, mask in masks.items():
            while mask.ndim > 2:
                mask = mask[0]
            if mask.shape[0] != h or mask.shape[1] != w:
                continue
            r, g, b = _obj_color(obj_id)
            m = mask > 0
            rgba[m, 0] = r
            rgba[m, 1] = g
            rgba[m, 2] = b
            rgba[m, 3] = 100
        qimg = QImage(rgba.data.tobytes(), w, h, 4 * w,
                      QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        if self._mask_pixmap_item is None:
            self._mask_pixmap_item = QGraphicsPixmapItem(pixmap)
            self._mask_pixmap_item.setZValue(-1)
            self.scene().addItem(self._mask_pixmap_item)
        else:
            self._mask_pixmap_item.setPixmap(pixmap)

    def clear_mask(self):
        if self._mask_pixmap_item is not None:
            self.scene().removeItem(self._mask_pixmap_item)
            self._mask_pixmap_item = None

    def get_points(self, obj_id: int | None = None) -> tuple[list[tuple[int, int]], list[int]]:
        """Get points for a specific object (or current if None)."""
        oid = obj_id if obj_id is not None else self._current_obj
        obj = self._objects.get(oid, {"points": [], "labels": []})
        return list(obj["points"]), list(obj["labels"])

    def get_all_objects(self) -> dict[int, dict]:
        """Return {obj_id: {"points": [...], "labels": [...]}} for serialization."""
        return {oid: {"points": list(o["points"]), "labels": list(o["labels"])}
                for oid, o in self._objects.items()}

    def set_mode(self, label: int):
        self._mode = label

    def clear_current(self):
        """Clear points for the current object only."""
        obj = self._objects.get(self._current_obj)
        if not obj:
            return
        for item in obj["items"]:
            self.scene().removeItem(item)
        obj["points"].clear()
        obj["labels"].clear()
        obj["items"].clear()

    def clear_all(self):
        """Remove all objects' points and masks."""
        for obj in self._objects.values():
            for item in obj["items"]:
                self.scene().removeItem(item)
        self._objects.clear()
        self._objects[1] = {"points": [], "labels": [], "items": []}
        self._current_obj = 1
        self._next_obj = 2
        self.clear_mask()

    def undo_last_point(self):
        obj = self._objects.get(self._current_obj)
        if not obj or not obj["points"]:
            return
        obj["points"].pop()
        obj["labels"].pop()
        item = obj["items"].pop()
        self.scene().removeItem(item)
        self.points_changed.emit(self._current_obj)

    def restore_objects(self, data: dict[int, dict]):
        """Restore multi-object state from serialized data."""
        self.clear_all()
        max_id = 0
        for oid_str, obj_data in data.items():
            oid = int(oid_str)
            max_id = max(max_id, oid)
            pts = obj_data.get("points", [])
            labs = obj_data.get("labels", [])
            items = []
            for (x, y), lbl in zip(pts, labs):
                items.append(self._make_point_item(x, y, lbl, oid))
            self._objects[oid] = {"points": list(pts), "labels": list(labs), "items": items}
        if not self._objects:
            self._objects[1] = {"points": [], "labels": [], "items": []}
        self._current_obj = sorted(self._objects.keys())[0]
        self._next_obj = max_id + 1 if max_id > 0 else 2
        self._dim_non_current_points()

    # ── internal helpers ────────────────────────────────────────────────

    def _make_point_item(self, x: int, y: int, label: int, obj_id: int) -> QGraphicsEllipseItem:
        # Scale radius to ~1% of image short side, clamp to [2, 8]
        if self._image_arr is not None:
            short = min(self._image_arr.shape[0], self._image_arr.shape[1])
            r = max(2, min(8, int(short * 0.01 + 0.5)))
        else:
            r = 5
        cr, cg, cb = _obj_color(obj_id)
        if label == 1:
            color = QColor(cr, cg, cb, 220)
        else:
            color = QColor(220, 40, 40, 220)
        item = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        pen_w = max(0.5, r / 4)
        item.setPen(QPen(Qt.GlobalColor.white, pen_w))
        item.setBrush(color)
        item.setZValue(10)
        self.scene().addItem(item)
        return item

    def _dim_non_current_points(self):
        """Make non-current objects' points semi-transparent."""
        for oid, obj in self._objects.items():
            opacity = 1.0 if oid == self._current_obj else 0.35
            for item in obj["items"]:
                item.setOpacity(opacity)

    # ── mouse events ────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = _mouse_pos_qpoint(event)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        if self._image_arr is None:
            return

        scene_pos = self.mapToScene(_mouse_pos_qpoint(event))
        x, y = int(scene_pos.x()), int(scene_pos.y())
        h, w = self._image_arr.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        obj = self._objects[self._current_obj]
        obj["points"].append((x, y))
        obj["labels"].append(self._mode)
        item = self._make_point_item(x, y, self._mode, self._current_obj)
        obj["items"].append(item)
        self.points_changed.emit(self._current_obj)
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

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Z and \
                event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.undo_last_point()
            event.accept()
            return
        super().keyPressEvent(event)
