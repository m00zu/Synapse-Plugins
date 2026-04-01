"""
nodes/roi_nodes.py
==================
ROIMaskNode – draw an ROI (ellipse, rectangle, or polygon) directly on the
node surface and output it as a binary MaskData mask.

Adapted from PyQt5 FRAP tool (FRAPy) and ported to PySide6 /
NodeGraphQt conventions.
"""

from __future__ import annotations

import json
import threading
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import (QPen, QBrush, QColor, QPainter, QFont,
                            QPolygonF, QPixmap, QImage,
                            QPainterPath, QPainterPathStroker)
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsPolygonItem,
    QGraphicsLineItem, QGraphicsSimpleTextItem, QGraphicsPathItem,
    QGraphicsItem,
)
from PySide6.QtCore import QLineF
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget
from nodes.base import BaseExecutionNode, BaseImageProcessNode, PORT_COLORS
from data_models import ImageData, MaskData, LabelData, TableData


def _image_hw(image) -> tuple[int, int]:
    """Return (width, height) from a numpy array or PIL Image."""
    if isinstance(image, np.ndarray):
        return image.shape[1], image.shape[0]
    return image.width, image.height


def _ensure_display_rgb(image) -> np.ndarray:
    """Convert *image* (numpy array or PIL Image) to an RGB uint8 numpy array for display."""
    from PIL import Image as _PIL
    if isinstance(image, _PIL.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    arr = image
    if arr.dtype != np.uint8:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip((arr.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


# Qt6: QMouseEvent.pos() is deprecated; use position().toPoint() when available.
def _mouse_pos_qpoint(event):
    pos_fn = getattr(event, "position", None)
    if callable(pos_fn):
        return pos_fn().toPoint()
    return event.pos()


# ── Mask rasterisation (no cv2) ──────────────────────────────────────────────

def _roi_dict_to_mask_arr(roi_data: dict, img_w: int, img_h: int) -> np.ndarray:
    """
    Rasterise an ROI data dict → uint8 numpy array (H×W, values 0 or 255).

    roi_data keys
    -------------
    shape       : 'ellipse' | 'rectangle' | 'polygon'
    center      : [cx, cy]          (ellipse / rectangle)
    axes        : [half_w, half_h]  (ellipse / rectangle)
    angle       : rotation degrees  (ellipse / rectangle)
    polypoints  : [[x,y], ...]      (polygon)
    """
    from PIL import Image, ImageDraw

    mask_img = Image.new('L', (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask_img)
    shape = roi_data.get('shape')

    if shape in ('ellipse', 'rectangle'):
        cx   = float(roi_data['center'][0])
        cy   = float(roi_data['center'][1])
        ax   = float(roi_data['axes'][0])
        ay   = float(roi_data['axes'][1])
        arad = np.deg2rad(float(roi_data.get('angle', 0)))
        ca, sa = np.cos(arad), np.sin(arad)

        if shape == 'ellipse':
            t = np.linspace(0, 2 * np.pi, 256, endpoint=False)
            lx = ax * np.cos(t)
            ly = ay * np.sin(t)
        else:                                      # rectangle – 4 corners
            lx = np.array([-ax, -ax,  ax,  ax])
            ly = np.array([ ay, -ay, -ay,  ay])

        xr = lx * ca - ly * sa + cx
        yr = lx * sa + ly * ca + cy
        draw.polygon([(float(xr[i]), float(yr[i])) for i in range(len(xr))],
                     fill=255)

    elif shape == 'polygon':
        pts = roi_data.get('polypoints', [])
        if len(pts) >= 3:
            draw.polygon([(float(p[0]), float(p[1])) for p in pts], fill=255)

    return np.array(mask_img)


def _roi_dict_to_outline_points(roi_data: dict) -> list[tuple[float, float]]:
    """Convert ROI dict to ordered outline points in image coordinates."""
    shape = roi_data.get('shape')
    if shape in ('ellipse', 'rectangle'):
        cx = float(roi_data['center'][0])
        cy = float(roi_data['center'][1])
        ax = float(roi_data['axes'][0])
        ay = float(roi_data['axes'][1])
        arad = np.deg2rad(float(roi_data.get('angle', 0)))
        ca, sa = np.cos(arad), np.sin(arad)

        if shape == 'ellipse':
            t = np.linspace(0, 2 * np.pi, 240, endpoint=False)
            lx = ax * np.cos(t)
            ly = ay * np.sin(t)
        else:
            lx = np.array([-ax, -ax,  ax,  ax], dtype=float)
            ly = np.array([ ay, -ay, -ay,  ay], dtype=float)

        xr = lx * ca - ly * sa + cx
        yr = lx * sa + ly * ca + cy
        return [(float(xr[i]), float(yr[i])) for i in range(len(xr))]

    if shape == 'polygon':
        pts = roi_data.get('polypoints', [])
        return [(float(p[0]), float(p[1])) for p in pts] if len(pts) >= 2 else []
    return []


def _draw_styled_polyline(
    draw,
    pts: list[tuple[float, float]],
    color: tuple[int, int, int],
    width: int,
    style: str,
    closed: bool = False,
):
    """Draw polyline/polygon with simple dash patterns via PIL.ImageDraw."""
    if len(pts) < 2:
        return
    path = list(pts)
    if closed:
        path.append(path[0])

    if style == 'solid':
        draw.line(path, fill=color, width=round(width), joint='curve')
        # PIL butt-caps leave gaps at sharp corners — fill with circles
        r = max(1, round(width) // 2)
        for px, py in path:
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color)
        return

    base = max(2, int(width))
    patterns = {
        'dashed':  [6 * base, 3 * base],
        'dotted':  [1 * base, 2 * base],
        'dashdot': [6 * base, 3 * base, 1 * base, 3 * base],
    }
    pattern = patterns.get(style, [6 * base, 3 * base])
    pat_i = 0
    pat_used = 0.0
    draw_on = True

    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        dx, dy = (x2 - x1), (y2 - y1)
        seg_len = float(np.hypot(dx, dy))
        if seg_len <= 1e-9:
            continue
        ux, uy = dx / seg_len, dy / seg_len
        d = 0.0
        while d < seg_len:
            remaining_seg = seg_len - d
            remaining_pat = pattern[pat_i] - pat_used
            step = min(remaining_seg, remaining_pat)
            if draw_on and step > 0:
                sx = x1 + ux * d
                sy = y1 + uy * d
                ex = x1 + ux * (d + step)
                ey = y1 + uy * (d + step)
                draw.line([(sx, sy), (ex, ey)], fill=color, width=int(width))
                # round caps at dash endpoints to avoid corner gaps
                r = max(1, int(width) // 2)
                draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=color)
                draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=color)
            d += step
            pat_used += step
            if pat_used >= pattern[pat_i] - 1e-9:
                pat_used = 0.0
                pat_i = (pat_i + 1) % len(pattern)
                draw_on = not draw_on


def _draw_shape_overlay(
    image,
    roi_data: dict,
    line_color: tuple[int, int, int],
    line_width: float,
    line_style: str,
    label_text: str = '',
    label_x_offset: float = 8.0,
    label_y_offset: float = -8.0,
    label_font_size: float = 12.0,
):
    """Return RGB numpy array with ROI outline drawn above the input image.

    *image* may be a numpy array or PIL Image.
    """
    from PIL import Image as _PIL, ImageDraw

    if isinstance(image, np.ndarray):
        arr = image if image.ndim == 3 else np.stack([image]*3, axis=-1)
        if arr.dtype != np.uint8:
            arr = (arr / arr.max() * 255).astype(np.uint8) if arr.max() > 0 else arr.astype(np.uint8)
        pil_image = _PIL.fromarray(arr, 'RGB')
    else:
        pil_image = image

    if pil_image.mode != 'RGB':
        out = pil_image.convert('RGB')
    else:
        out = pil_image.copy()
    draw = ImageDraw.Draw(out)
    shape = roi_data.get('shape')

    if shape == 'arrow':
        pts = roi_data.get('points', [])
        if isinstance(pts, list) and len(pts) == 2:
            x1, y1 = float(pts[0][0]), float(pts[0][1])
            x2, y2 = float(pts[1][0]), float(pts[1][1])
            dx, dy = (x1 - x2), (y1 - y2)
            L = float(np.hypot(dx, dy))
            head_len = max(8.0, 4.0 * float(line_width))
            if L > 1e-6:
                ux, uy = dx / L, dy / L
                # Shaft ends at arrowhead base so it does not poke through tip
                base_x = x2 + ux * head_len * 0.9
                base_y = y2 + uy * head_len * 0.9
                _draw_styled_polyline(
                    draw, [(x1, y1), (base_x, base_y)],
                    color=line_color,
                    width=max(1, int(line_width)),
                    style=str(line_style or 'solid').lower(),
                    closed=False,
                )
                ang = np.deg2rad(28.0)
                c, s = np.cos(ang), np.sin(ang)
                lx = x2 + (ux * c - uy * s) * head_len
                ly = y2 + (ux * s + uy * c) * head_len
                rx = x2 + (ux * c + uy * s) * head_len
                ry = y2 + (-ux * s + uy * c) * head_len
                draw.polygon([(x2, y2), (lx, ly), (rx, ry)], fill=line_color)
            else:
                _draw_styled_polyline(
                    draw, [(x1, y1), (x2, y2)],
                    color=line_color,
                    width=max(1, int(line_width)),
                    style=str(line_style or 'solid').lower(),
                    closed=False,
                )
            txt = str(label_text or '').strip()
            if txt:
                from PIL import ImageFont
                try:
                    font = ImageFont.load_default(size=round(max(4.0, label_font_size)))
                except TypeError:
                    font = None
                tx, ty = x2 + label_x_offset, y2 + label_y_offset
                draw.text((tx + 1, ty + 1), txt, fill=(0, 0, 0), font=font)
                draw.text((tx, ty), txt, fill=line_color, font=font)
        return out

    pts = _roi_dict_to_outline_points(roi_data)
    if pts:
        closed = shape in ('ellipse', 'rectangle', 'polygon')
        fill_mode = bool(roi_data.get('fill_mode', False))

        if fill_mode and closed and len(pts) >= 3:
            # Fill with user-controlled opacity
            alpha = int(roi_data.get('fill_opacity', 180))
            fill_rgba = (line_color[0], line_color[1], line_color[2], alpha)
            # Draw filled polygon on RGBA overlay
            img_w, img_h = out.size
            ov = _PIL.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
            ov_draw = ImageDraw.Draw(ov)
            int_pts = [(int(round(x)), int(round(y))) for x, y in pts]
            ov_draw.polygon(int_pts, fill=fill_rgba, outline=line_color,
                            width=max(1, int(line_width)))
            out = _PIL.alpha_composite(out.convert('RGBA'), ov).convert('RGB')
        else:
            _draw_styled_polyline(
                draw, pts,
                color=line_color,
                width=max(1, int(line_width)),
                style=str(line_style or 'solid').lower(),
                closed=closed,
            )
    return out


def _mask_to_outline_paths(mask_bool: np.ndarray) -> list[list[tuple[float, float]]]:
    """Extract contour paths from a binary mask as [(x,y), ...] lists."""
    from skimage.measure import find_contours
    contours = find_contours(mask_bool.astype(np.uint8), level=0.5)
    paths: list[list[tuple[float, float]]] = []
    for c in contours:
        if len(c) < 2:
            continue
        # skimage gives (row, col) => (y, x)
        pts = [(float(p[1]), float(p[0])) for p in c]
        paths.append(pts)
    return paths


# ── Custom resizable / movable graphics items ─────────────────────────────────

class _ResizableMixin:
    """
    Provides shared resize and move logic for CustomEllipseItem and CustomRectangleItem.

    Subclasses must also inherit from a QGraphicsItem subclass.
    """

    _EDGE_TOL_FRAC = 0.12   # fraction of width/height used as edge tolerance
    _EDGE_TOL_MIN  = 5.0    # minimum tolerance in px

    _CURSORS = {
        'top':    Qt.CursorShape.SizeVerCursor,
        'bottom': Qt.CursorShape.SizeVerCursor,
        'left':   Qt.CursorShape.SizeHorCursor,
        'right':  Qt.CursorShape.SizeHorCursor,
        None:     Qt.CursorShape.SizeAllCursor,
    }

    def _init_resize(self):
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
        self.setAcceptHoverEvents(True)
        self.selected_edge = None
        self.click_pos     = None
        self.click_rect    = None
        self.curr_origin   = QPointF()    # transform origin in local coords
        self.drag_offset   = QPointF()

    def _edge_at(self, pos: QPointF) -> str | None:
        rect  = self.rect()
        tol_w = max(self._EDGE_TOL_MIN, rect.width()  * self._EDGE_TOL_FRAC)
        tol_h = max(self._EDGE_TOL_MIN, rect.height() * self._EDGE_TOL_FRAC)
        if abs(rect.top()    - pos.y()) < tol_h: return 'top'
        if abs(rect.bottom() - pos.y()) < tol_h: return 'bottom'
        if abs(rect.left()   - pos.x()) < tol_w: return 'left'
        if abs(rect.right()  - pos.x()) < tol_w: return 'right'
        return None

    def hoverMoveEvent(self, event):
        if self.isSelected():
            self.setCursor(self._CURSORS[self._edge_at(event.pos())])
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.click_pos  = event.pos()
            self.click_rect = QRectF(self.rect())
            self.selected_edge = self._edge_at(self.click_pos)
            if self.selected_edge is None:
                self.drag_offset = (event.scenePos()
                                    - self.mapToScene(self.curr_origin))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.click_pos is None:
            return super().mouseMoveEvent(event)
        pos = event.pos()
        dx  = pos.x() - self.click_pos.x()
        dy  = pos.y() - self.click_pos.y()
        rect = QRectF(self.click_rect)

        if self.selected_edge is None:
            self.setPos(event.scenePos() - self.drag_offset - self.curr_origin)
            return

        if   self.selected_edge == 'top':    rect.setTop(rect.top() + dy)
        elif self.selected_edge == 'bottom': rect.setBottom(rect.bottom() + dy)
        elif self.selected_edge == 'left':   rect.setLeft(rect.left() + dx)
        elif self.selected_edge == 'right':  rect.setRight(rect.right() + dx)

        # Shift-constrain: force square (equal width/height)
        from PySide6.QtWidgets import QApplication
        if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
            side = max(rect.width(), rect.height())
            if self.selected_edge in ('top', 'bottom'):
                mid_x = rect.center().x()
                rect.setLeft(mid_x - side / 2)
                rect.setRight(mid_x + side / 2)
            else:
                mid_y = rect.center().y()
                rect.setTop(mid_y - side / 2)
                rect.setBottom(mid_y + side / 2)

        MIN = 6.0
        if rect.width()  < MIN:
            if self.selected_edge == 'left':  rect.setLeft(rect.right()  - MIN)
            else:                             rect.setRight(rect.left()   + MIN)
        if rect.height() < MIN:
            if self.selected_edge == 'top':   rect.setTop(rect.bottom()  - MIN)
            else:                             rect.setBottom(rect.top()   + MIN)
        self.setRect(rect)

    def mouseReleaseEvent(self, event):
        if self.selected_edge is not None:
            center = self.rect().center()
            self.setTransformOriginPoint(center)
            self.curr_origin   = center
            self.selected_edge = None
        self.click_pos = None
        super().mouseReleaseEvent(event)


class CustomArrowItem(QGraphicsLineItem):
    """Draws a line with a filled arrowhead at the tip and an optional annotation label that follows the arrow."""

    _HEAD_HALF_ANGLE = 0.42   # radians ≈ 24°

    def __init__(self, x1=0.0, y1=0.0, x2=0.0, y2=0.0, parent=None):
        super().__init__(x1, y1, x2, y2, parent)
        self._label_item = QGraphicsSimpleTextItem('', self)   # child → auto-follows
        self._label_item.setZValue(1)
        self._label_x_offset = 8.0
        self._label_y_offset = -8.0
        self._font_size = 12.0
        self._update_label_pos()

    # ── label helpers ──────────────────────────────────────────────────────

    def set_label(self, text: str):
        self._label_item.setText(str(text or ''))
        self._label_item.setVisible(bool(text and text.strip()))
        self._update_label_pos()

    def set_label_offset(self, x: float, y: float):
        self._label_x_offset = float(x)
        self._label_y_offset = float(y)
        self._update_label_pos()

    def set_font_size(self, size: float):
        self._font_size = max(4.0, float(size))
        f = self._label_item.font()
        f.setPointSizeF(self._font_size)
        f.setWeight(QFont.Weight.Light)
        self._label_item.setFont(f)

    def _update_label_pos(self):
        ln = self.line()
        self._label_item.setPos(ln.x2() + self._label_x_offset,
                                ln.y2() + self._label_y_offset)

    def setLine(self, x1, y1, x2, y2):   # override to keep label in sync
        super().setLine(x1, y1, x2, y2)
        self._update_label_pos()

    def setPen(self, pen):                # override to keep label colour in sync
        super().setPen(pen)
        self._label_item.setBrush(QBrush(pen.color()))

    def shape(self):
        """Wider hit-test area so the arrow line is easy to click/select."""
        path = QPainterPath()
        path.moveTo(self.line().p1())
        path.lineTo(self.line().p2())
        stroker = QPainterPathStroker()
        stroker.setWidth(max(12.0, self.pen().widthF() + 10.0))
        return stroker.createStroke(path)

    def _head_size(self) -> float:
        """Arrowhead length proportional to pen width (min 10 px)."""
        return max(10.0, self.pen().widthF() * 4.0)

    def boundingRect(self):
        extra = self._head_size() + self.pen().widthF()
        r = super().boundingRect()
        return r.adjusted(-extra, -extra, extra, extra)

    def paint(self, painter, _option=None, _widget=None):
        import math
        ln  = self.line()
        pen = self.pen()
        dx, dy = ln.x1() - ln.x2(), ln.y1() - ln.y2()
        L = math.hypot(dx, dy)
        if L < 1e-6:
            painter.setPen(pen)
            painter.drawLine(ln)
            return
        ux, uy = dx / L, dy / L
        x2, y2 = ln.x2(), ln.y2()
        head = self._head_size()
        # Draw shaft stopping just inside arrowhead base (FlatCap, no overshoot at tip)
        shaft_pen = QPen(pen)
        shaft_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(shaft_pen)
        painter.drawLine(QPointF(ln.x1(), ln.y1()),
                         QPointF(x2 + ux * head * 0.9, y2 + uy * head * 0.9))
        # Draw filled arrowhead triangle
        ang = self._HEAD_HALF_ANGLE
        c, s = math.cos(ang), math.sin(ang)
        poly = QPolygonF([
            QPointF(x2, y2),
            QPointF(x2 + (ux * c - uy * s) * head, y2 + (ux * s + uy * c) * head),
            QPointF(x2 + (ux * c + uy * s) * head, y2 + (-ux * s + uy * c) * head),
        ])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(pen.color()))
        painter.drawPolygon(poly)


# ── Cubic Bézier curve item ──────────────────────────────────────────────────

class _CurveHandle(QGraphicsEllipseItem):
    """Provides a small draggable circle handle for editing a Bezier control point."""

    _RADIUS = 5.0

    def __init__(self, parent_curve: 'CustomCurveItem', index: int):
        r = self._RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r, parent_curve)
        self._index = index
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setPen(QPen(QColor(255, 255, 100), 1))
        self.setBrush(QBrush(QColor(255, 255, 100, 180)))
        self.setZValue(10)
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def itemChange(self, change, value):
        if (change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
                and self.parentItem() is not None):
            parent = self.parentItem()
            if getattr(parent, '_syncing', False):
                return super().itemChange(change, value)
            # Handle pos is in parent coords — use directly as control point
            pos = self.pos()
            if self._index == 1:
                parent._cp1 = pos
            else:
                parent._cp2 = pos
            parent._rebuild_path()
        return super().itemChange(change, value)


class CustomCurveItem(QGraphicsPathItem):
    """Draws a cubic Bezier curve with two draggable control-point handles, an optional arrowhead at the endpoint, and an annotation label."""

    _HEAD_HALF_ANGLE = 0.42  # same as CustomArrowItem

    def __init__(self, x1=0.0, y1=0.0, x2=0.0, y2=0.0, parent=None):
        super().__init__(parent)
        self._p1 = QPointF(x1, y1)
        self._p2 = QPointF(x2, y2)
        self._syncing = False

        # Label must exist before _rebuild_path (which calls _update_label_pos)
        self._label_item = QGraphicsSimpleTextItem('', self)
        self._label_item.setZValue(1)
        self._label_x_offset = 8.0
        self._label_y_offset = -8.0
        self._font_size = 12.0

        self._compute_default_control_points()
        self._rebuild_path()

        # Control-point handles
        self._h1 = _CurveHandle(self, 1)
        self._h2 = _CurveHandle(self, 2)
        self._h1.setVisible(False)
        self._h2.setVisible(False)
        self._sync_handles()

    # ── control point defaults ────────────────────────────────────────────

    def _compute_default_control_points(self):
        """Place control points at 1/3 and 2/3 along the line,
        offset perpendicular for a gentle S-curve."""
        import math
        dx = self._p2.x() - self._p1.x()
        dy = self._p2.y() - self._p1.y()
        L = math.hypot(dx, dy)
        offset = max(30.0, L * 0.25)
        # Perpendicular unit vector
        if L < 1e-6:
            nx, ny = 0, -1
        else:
            nx, ny = -dy / L, dx / L
        third = QPointF(self._p1.x() + dx / 3, self._p1.y() + dy / 3)
        two_third = QPointF(self._p1.x() + 2 * dx / 3,
                            self._p1.y() + 2 * dy / 3)
        self._cp1 = QPointF(third.x() + nx * offset,
                             third.y() + ny * offset)
        self._cp2 = QPointF(two_third.x() - nx * offset,
                             two_third.y() - ny * offset)

    # ── path rebuild ──────────────────────────────────────────────────────

    def _rebuild_path(self):
        path = QPainterPath()
        path.moveTo(self._p1)
        path.cubicTo(self._cp1, self._cp2, self._p2)
        self.setPath(path)
        self._update_label_pos()

    # ── public API ────────────────────────────────────────────────────────

    def setEndpoints(self, p1: QPointF, p2: QPointF,
                     recompute_cp: bool = True):
        self._p1 = QPointF(p1)
        self._p2 = QPointF(p2)
        if recompute_cp:
            self._compute_default_control_points()
        self._rebuild_path()
        self._sync_handles()

    def setControlPoints(self, cp1: QPointF, cp2: QPointF):
        self._cp1 = QPointF(cp1)
        self._cp2 = QPointF(cp2)
        self._rebuild_path()
        self._sync_handles()

    def endpoints(self) -> tuple[QPointF, QPointF]:
        return QPointF(self._p1), QPointF(self._p2)

    def controlPoints(self) -> tuple[QPointF, QPointF]:
        return QPointF(self._cp1), QPointF(self._cp2)

    # ── label helpers (same API as CustomArrowItem) ───────────────────────

    def set_label(self, text: str):
        self._label_item.setText(str(text or ''))
        self._label_item.setVisible(bool(text and text.strip()))
        self._update_label_pos()

    def set_label_offset(self, x: float, y: float):
        self._label_x_offset = float(x)
        self._label_y_offset = float(y)
        self._update_label_pos()

    def set_font_size(self, size: float):
        self._font_size = max(4.0, float(size))
        f = self._label_item.font()
        f.setPointSizeF(self._font_size)
        f.setWeight(QFont.Weight.Light)
        self._label_item.setFont(f)

    def _update_label_pos(self):
        self._label_item.setPos(self._p2.x() + self._label_x_offset,
                                self._p2.y() + self._label_y_offset)

    # ── handle sync ───────────────────────────────────────────────────────

    def _sync_handles(self):
        """Move handle widgets to match current control points."""
        self._syncing = True
        self._h1.setPos(self._cp1)
        self._h2.setPos(self._cp2)
        self._syncing = False

    # ── selection → show/hide handles ─────────────────────────────────────

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            visible = bool(value)
            self._h1.setVisible(visible)
            self._h2.setVisible(visible)
        return super().itemChange(change, value)

    # ── hit-testing ───────────────────────────────────────────────────────

    def shape(self):
        stroker = QPainterPathStroker()
        stroker.setWidth(max(12.0, self.pen().widthF() + 10.0))
        return stroker.createStroke(self.path())

    # ── painting ──────────────────────────────────────────────────────────

    def _head_size(self) -> float:
        return max(10.0, self.pen().widthF() * 4.0)

    def paint(self, painter, option=None, widget=None):
        import math
        pen = self.pen()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw the Bézier curve
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self.path())

        # Arrowhead at p2
        p2 = self._p2
        # Tangent at t=1 is (p2 - cp2)
        dx = p2.x() - self._cp2.x()
        dy = p2.y() - self._cp2.y()
        L = math.hypot(dx, dy)
        if L > 1e-6:
            ux, uy = dx / L, dy / L
            head = self._head_size()
            ang = self._HEAD_HALF_ANGLE
            c, s = math.cos(ang), math.sin(ang)
            poly = QPolygonF([
                QPointF(p2.x(), p2.y()),
                QPointF(p2.x() - (ux * c - uy * s) * head,
                        p2.y() - (ux * s + uy * c) * head),
                QPointF(p2.x() - (ux * c + uy * s) * head,
                        p2.y() - (-ux * s + uy * c) * head),
            ])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(pen.color()))
            painter.drawPolygon(poly)

        # When selected: draw tangent construction lines
        if self.isSelected():
            dash = QPen(QColor(255, 255, 100, 120), 1,
                        Qt.PenStyle.DashLine)
            painter.setPen(dash)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(self._p1, self._cp1)
            painter.drawLine(self._p2, self._cp2)

    def setPen(self, pen):
        super().setPen(pen)
        self._label_item.setBrush(QBrush(pen.color()))


class CustomEllipseItem(_ResizableMixin, QGraphicsEllipseItem):
    def __init__(self, *args, **kwargs):
        QGraphicsEllipseItem.__init__(self, *args, **kwargs)
        self._init_resize()


class CustomRectangleItem(_ResizableMixin, QGraphicsRectItem):
    def __init__(self, *args, **kwargs):
        QGraphicsRectItem.__init__(self, *args, **kwargs)
        self._init_resize()


# ── Interactive drawing view ──────────────────────────────────────────────────

class ROIGraphicsView(QGraphicsView):
    """
    Embedded QGraphicsView for drawing a single ROI shape.

    Supported shapes:
    - *ellipse* / *rectangle* — click-drag to create, edge-drag to resize, body-drag to move
    - *polygon* — left-click to add vertices, Enter to close, Backspace to undo last vertex, Delete to clear

    Emits `shape_committed` once a shape is finalised (mouse-release for ellipse/rect, Enter for polygon).
    """

    shape_committed = Signal()

    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._curr_shape      : str                   = 'ellipse'
        self._shape_item                              = None
        self._drawing         : bool                  = False
        self._drawing_polygon : bool                  = False
        self._start_pt        : QPointF               = QPointF()
        self._poly_pts        : QPolygonF             = QPolygonF()
        self._scale           : float                 = 1.0
        self._pan_start       : QtCore.QPoint | None  = None
        # Cyan default: visible on both dark and bright/grayscale images
        self._pen_color  = QColor(0, 220, 220)
        self._fill_color = QColor(0, 220, 220, 50)
        self._pen_width  = 2.0
        self._pen_style  = Qt.PenStyle.SolidLine
        self._arrow_label          = ''
        self._arrow_label_x_off    = 8.0
        self._arrow_label_y_off    = -8.0
        self._arrow_label_font_size = 12.0

    # ── public API ────────────────────────────────────────────────────────────

    def set_shape_type(self, shape: str):
        self._curr_shape = shape

    def clear_roi(self):
        if self._shape_item is not None:
            self.scene().removeItem(self._shape_item)
            self._shape_item = None
        self._drawing = self._drawing_polygon = False
        self._poly_pts = QPolygonF()

    def get_roi_data(self) -> dict | None:
        if self._shape_item is None:
            return None
        return self._build_roi_dict(self._shape_item)

    def load_roi_data(self, roi_data: dict):
        """Reconstruct scene items from a stored ROI dict."""
        if self._shape_item is not None:
            self.scene().removeItem(self._shape_item)
            self._shape_item = None
        shape = roi_data.get('shape')
        if shape in ('ellipse', 'rectangle'):
            cx = float(roi_data['center'][0])
            cy = float(roi_data['center'][1])
            ax = float(roi_data['axes'][0])
            ay = float(roi_data['axes'][1])
            rect = QRectF(cx - ax, cy - ay, 2 * ax, 2 * ay)
            item = (CustomEllipseItem if shape == 'ellipse'
                    else CustomRectangleItem)(rect)
            center = rect.center()
            item.setTransformOriginPoint(center)
            item.curr_origin = center
            item.setRotation(float(roi_data.get('angle', 0)))
            self._style(item)
            self.scene().addItem(item)
            self._shape_item = item
        elif shape == 'polygon':
            pts = roi_data.get('polypoints', [])
            if len(pts) >= 3:
                poly = QPolygonF([QPointF(p[0], p[1]) for p in pts])
                item = QGraphicsPolygonItem(poly)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
                self._style(item)
                self.scene().addItem(item)
                self._shape_item = item
        elif shape == 'arrow':
            pts = roi_data.get('points', [])
            if isinstance(pts, list) and len(pts) == 2:
                p1 = QPointF(float(pts[0][0]), float(pts[0][1]))
                p2 = QPointF(float(pts[1][0]), float(pts[1][1]))
                item = CustomArrowItem(p1.x(), p1.y(), p2.x(), p2.y())
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
                self._style(item)
                item.set_label(self._arrow_label)
                item.set_label_offset(self._arrow_label_x_off, self._arrow_label_y_off)
                item.set_font_size(self._arrow_label_font_size)
                self.scene().addItem(item)
                self._shape_item = item

    def load_image(self, image):
        """Replace the background pixmap. Existing ROI is kept in place.

        *image* may be a numpy array (H, W) or (H, W, 3) or a PIL Image.
        """
        # Remove old pixmap (keep ROI items)
        for item in list(self.scene().items()):
            if isinstance(item, QGraphicsPixmapItem):
                self.scene().removeItem(item)

        arr = _ensure_display_rgb(image)
        h, w, _ = arr.shape
        q_img   = QImage(arr.data, w, h, 3 * w,
                         QImage.Format.Format_RGB888).copy()  # .copy() detaches from arr
        pixmap  = QPixmap.fromImage(q_img)
        px_item = QGraphicsPixmapItem(pixmap)
        px_item.setZValue(-1)   # always behind ROI items
        self.scene().addItem(px_item)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(px_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_rotation(self, angle: float):
        if isinstance(self._shape_item, (CustomEllipseItem, CustomRectangleItem)):
            center = self._shape_item.rect().center()
            self._shape_item.setTransformOriginPoint(center)
            self._shape_item.curr_origin = center
            self._shape_item.setRotation(angle)

    def zoom_in(self):
        self._apply_zoom(1.2)

    def zoom_out(self):
        self._apply_zoom(1 / 1.2)

    def zoom_reset(self):
        for item in self.scene().items():
            if isinstance(item, QGraphicsPixmapItem):
                self.fitInView(item, Qt.AspectRatioMode.KeepAspectRatio)
                return

    # ── drawing events ────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = _mouse_pos_qpoint(event)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        scene_pos = self.mapToScene(_mouse_pos_qpoint(event))

        # Only draw if no shape exists yet OR existing shape is being edited
        if self._curr_shape in ('ellipse', 'rectangle', 'arrow'):
            if self._shape_item is None:
                # Start new shape
                self._drawing  = True
                self._start_pt = scene_pos
                if self._curr_shape == 'ellipse':
                    item = CustomEllipseItem(QRectF(scene_pos, scene_pos))
                elif self._curr_shape == 'rectangle':
                    item = CustomRectangleItem(QRectF(scene_pos, scene_pos))
                else:
                    item = CustomArrowItem(scene_pos.x(), scene_pos.y(),
                                          scene_pos.x(), scene_pos.y())
                    item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
                    item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
                    item.set_label(self._arrow_label)
                    item.set_label_offset(self._arrow_label_x_off, self._arrow_label_y_off)
                    item.set_font_size(self._arrow_label_font_size)
                self._style(item)
                self.scene().addItem(item)
                self._shape_item = item
                return          # don't pass event (avoids node-graph selection)
            # else: let super handle (forwards to item's mousePressEvent)

        elif self._curr_shape == 'lasso':
            if self._shape_item is None:
                self._poly_pts = QPolygonF()
                self._poly_pts.append(scene_pos)
                item = QGraphicsPolygonItem(QPolygonF(self._poly_pts))
                self._style(item)
                self.scene().addItem(item)
                self._shape_item = item
                self._drawing = True
                return

        elif self._curr_shape == 'polygon':
            if self._shape_item is None:
                # First vertex
                self._poly_pts = QPolygonF()
                self._poly_pts.append(scene_pos)
                item = QGraphicsPolygonItem(QPolygonF(self._poly_pts))
                self._style(item)
                self.scene().addItem(item)
                self._shape_item     = item
                self._drawing_polygon = True
                return
            elif self._drawing_polygon:
                self._poly_pts.append(scene_pos)
                self._shape_item.setPolygon(QPolygonF(self._poly_pts))
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_start is not None:
            cur_pos = _mouse_pos_qpoint(event)
            delta = cur_pos - self._pan_start
            self._pan_start = cur_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        scene_pos = self.mapToScene(_mouse_pos_qpoint(event))
        if self._drawing and self._curr_shape == 'lasso':
            self._poly_pts.append(scene_pos)
            self._shape_item.setPolygon(self._poly_pts)
            return
        if self._drawing:
            if self._curr_shape in ('ellipse', 'rectangle'):
                rect = QRectF(self._start_pt, scene_pos).normalized()
                # Shift-constrain: force square / circle
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    side = max(rect.width(), rect.height())
                    if scene_pos.x() < self._start_pt.x():
                        rect.setLeft(rect.right() - side)
                    else:
                        rect.setRight(rect.left() + side)
                    if scene_pos.y() < self._start_pt.y():
                        rect.setTop(rect.bottom() - side)
                    else:
                        rect.setBottom(rect.top() + side)
                self._shape_item.setRect(rect)
            elif self._curr_shape == 'arrow' and isinstance(self._shape_item, QGraphicsLineItem):
                self._shape_item.setLine(self._start_pt.x(), self._start_pt.y(),
                                         scene_pos.x(), scene_pos.y())
        elif self._drawing_polygon and self._shape_item is not None:
            tmp = QPolygonF(self._poly_pts)
            tmp.append(scene_pos)
            self._shape_item.setPolygon(tmp)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drawing and self._curr_shape == 'lasso':
                self._drawing = False
                if len(self._poly_pts) >= 3:
                    self._shape_item.setPolygon(self._poly_pts)
                    self._shape_item.setFlag(
                        QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
                    self._shape_item.setFlag(
                        QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
                    self.shape_committed.emit()
                else:
                    self.scene().removeItem(self._shape_item)
                    self._shape_item = None
                self._poly_pts = QPolygonF()
                return
            if self._drawing:
                # Initial shape creation by drag
                self._drawing = False
                item = self._shape_item
                if item is not None:
                    if self._curr_shape in ('ellipse', 'rectangle'):
                        r = item.rect()
                        if r.width() < 6 or r.height() < 6:
                            self.scene().removeItem(item)
                            self._shape_item = None
                        else:
                            center = r.center()
                            item.setTransformOriginPoint(center)
                            item.curr_origin = center
                            self.shape_committed.emit()
                    elif self._curr_shape == 'arrow' and isinstance(item, QGraphicsLineItem):
                        ln = item.line()
                        if ln.length() < 6:
                            self.scene().removeItem(item)
                            self._shape_item = None
                        else:
                            self.shape_committed.emit()
                    else:
                        self.shape_committed.emit()
                return
            elif not self._drawing_polygon and self._shape_item is not None:
                # Move or resize of an existing shape — forward to items first,
                # then re-emit so the node re-evaluates with updated geometry.
                super().mouseReleaseEvent(event)
                self.shape_committed.emit()
                return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._drawing_polygon:
            self._finish_polygon()
            return
        if key == Qt.Key.Key_Backspace and self._drawing_polygon:
            pts = list(self._poly_pts)
            if pts:
                pts.pop()
                self._poly_pts = QPolygonF(pts)
                cursor_scene = self.mapToScene(
                    self.mapFromGlobal(QtGui.QCursor.pos()))
                tmp = QPolygonF(self._poly_pts)
                tmp.append(cursor_scene)
                self._shape_item.setPolygon(tmp)
            return
        if key == Qt.Key.Key_Delete:
            self.clear_roi()
            return
        super().keyPressEvent(event)

    # Wheel zooms the view; consuming the event prevents node-graph scroll
    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._apply_zoom(factor)
        event.accept()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _finish_polygon(self):
        self._drawing_polygon = False
        if self._shape_item is not None and len(self._poly_pts) >= 3:
            self._shape_item.setPolygon(self._poly_pts)
            self._shape_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            self._shape_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            self.shape_committed.emit()
        elif self._shape_item is not None:
            self.scene().removeItem(self._shape_item)
            self._shape_item = None
        self._poly_pts = QPolygonF()

    def set_pen_color(self, color: QColor):
        """Change the outline color and re-style the current shape if any."""
        self._pen_color = color
        fill = QColor(color)
        fill.setAlpha(50)
        self._fill_color = fill
        if self._shape_item is not None:
            self._style(self._shape_item)

    def set_pen_width(self, width: float):
        self._pen_width = float(max(1.0, width))
        if self._shape_item is not None:
            self._style(self._shape_item)

    def set_pen_style(self, style_name: str):
        mapping = {
            'solid': Qt.PenStyle.SolidLine,
            'dashed': Qt.PenStyle.DashLine,
            'dotted': Qt.PenStyle.DotLine,
            'dashdot': Qt.PenStyle.DashDotLine,
        }
        self._pen_style = mapping.get(str(style_name).lower(), Qt.PenStyle.SolidLine)
        if self._shape_item is not None:
            self._style(self._shape_item)

    def set_arrow_label(self, text: str):
        self._arrow_label = str(text or '')
        if isinstance(self._shape_item, CustomArrowItem):
            self._shape_item.set_label(self._arrow_label)

    def set_arrow_label_offset(self, x: float, y: float):
        self._arrow_label_x_off = float(x)
        self._arrow_label_y_off = float(y)
        if isinstance(self._shape_item, CustomArrowItem):
            self._shape_item.set_label_offset(x, y)

    def set_arrow_label_font_size(self, size: float):
        self._arrow_label_font_size = max(4.0, float(size))
        if isinstance(self._shape_item, CustomArrowItem):
            self._shape_item.set_font_size(self._arrow_label_font_size)

    def _style(self, item):
        # pen_width is in scene (image-pixel) units so it scales proportionally
        # with the view — matching the PIL output at any zoom level.
        item.setPen(QPen(self._pen_color, self._pen_width,
                         self._pen_style,
                         Qt.PenCapStyle.RoundCap,
                         Qt.PenJoinStyle.RoundJoin))
        if hasattr(item, 'setBrush'):
            item.setBrush(QBrush(self._fill_color))

    def _apply_zoom(self, factor: float):
        self.scale(factor, factor)
        self._scale *= factor

    def _build_roi_dict(self, item) -> dict | None:
        if isinstance(item, (CustomEllipseItem, CustomRectangleItem)):
            rect   = item.rect()
            center = item.mapToScene(rect.center())
            return {
                'shape':      'ellipse' if isinstance(item, CustomEllipseItem)
                              else 'rectangle',
                'center':     [round(center.x(), 2), round(center.y(), 2)],
                'axes':       [round(rect.width()  / 2, 2),
                               round(rect.height() / 2, 2)],
                'angle':      round(float(item.rotation()), 4),
                'polypoints': [],
            }
        if isinstance(item, QGraphicsPolygonItem):
            poly = item.polygon()
            pos  = item.pos()
            return {
                'shape':      'polygon',
                'center':     [],
                'axes':       [],
                'angle':      0.0,
                'polypoints': [[round(p.x() + pos.x(), 2),
                                round(p.y() + pos.y(), 2)]
                               for p in poly],
            }
        if isinstance(item, QGraphicsLineItem):
            ln = item.line()
            pos = item.pos()
            return {
                'shape': 'arrow',
                'center': [],
                'axes': [],
                'angle': 0.0,
                'polypoints': [],
                'points': [
                    [round(ln.x1() + pos.x(), 2), round(ln.y1() + pos.y(), 2)],
                    [round(ln.x2() + pos.x(), 2), round(ln.y2() + pos.y(), 2)],
                ],
            }
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  MultiShapeGraphicsView — multi-shape drawing canvas
# ══════════════════════════════════════════════════════════════════════════════

class MultiShapeGraphicsView(QGraphicsView):
    """
    Drawing canvas that supports multiple independent shapes, each with its own style.

    Key differences from ROIGraphicsView:
    - Stores N shapes (dict keyed by ID) instead of one `_shape_item`
    - Per-shape style dict (colour, width, dash pattern, label, etc.)
    - Shift-constrain for square/circle during creation
    - Live dimension and coordinate overlays
    - Text shape support
    """

    shape_committed = Signal(str)           # shape_id
    shape_selected  = Signal(str)           # shape_id ('' = deselect)
    shapes_changed  = Signal()              # structural add/delete
    mouse_moved     = Signal(float, float)  # cursor in image-space

    _PEN_STYLE_MAP = {
        'solid':   Qt.PenStyle.SolidLine,
        'dashed':  Qt.PenStyle.DashLine,
        'dotted':  Qt.PenStyle.DotLine,
        'dashdot': Qt.PenStyle.DashDotLine,
    }

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

        # --- multi-shape storage ---
        self._shape_items:  dict[str, QGraphicsItem] = {}
        self._shape_styles: dict[str, dict] = {}
        self._selected_id:  str | None = None
        self._next_id: int = 0

        # --- drawing state ---
        self._curr_shape:      str       = 'ellipse'
        self._drawing:         bool      = False
        self._drawing_polygon: bool      = False
        self._start_pt:        QPointF   = QPointF()
        self._poly_pts:        QPolygonF = QPolygonF()
        self._active_draw_id:  str | None = None

        # --- defaults for next new shape ---
        self._default_color  = QColor(0, 220, 220)
        self._default_width  = 2.0
        self._default_style  = 'solid'
        self._default_font_size = 12.0

        # --- zoom / pan / handle drag ---
        self._scale:     float = 1.0
        self._pan_start: QtCore.QPoint | None = None
        self._dragging_handle: _CurveHandle | None = None

        # --- mask contour scene items (locked, non-selectable) ---
        self._mask_contour_items: list[QGraphicsPathItem] = []
        self._mask_bbox_items: dict[str, QGraphicsPathItem] = {}  # mask_id → bbox

        # --- dimension / coordinate overlays ---
        self._dim_overlay = QGraphicsSimpleTextItem()
        self._dim_overlay.setZValue(1000)
        self._dim_overlay.setBrush(QBrush(QColor(255, 255, 0)))
        self._dim_overlay.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self._dim_overlay.setVisible(False)
        scene.addItem(self._dim_overlay)

        self._coord_overlay = QGraphicsSimpleTextItem()
        self._coord_overlay.setZValue(1000)
        self._coord_overlay.setBrush(QBrush(QColor(200, 200, 200)))
        self._coord_overlay.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self._coord_overlay.setVisible(False)
        scene.addItem(self._coord_overlay)

    # ── ID helpers ─────────────────────────────────────────────────────────

    def _new_id(self) -> str:
        sid = f'shape_{self._next_id}'
        self._next_id += 1
        return sid

    # ── public API ─────────────────────────────────────────────────────────

    def set_shape_type(self, shape: str):
        self._curr_shape = shape

    def set_defaults(self, *, color: QColor | None = None,
                     width: float | None = None,
                     style: str | None = None,
                     font_size: float | None = None):
        if color is not None:
            self._default_color = color
        if width is not None:
            self._default_width = max(0.5, width)
        if style is not None:
            self._default_style = style
        if font_size is not None:
            self._default_font_size = max(4.0, font_size)

    def _current_default_style(self) -> dict:
        c = self._default_color
        return {
            'line_width':      self._default_width,
            'line_style':      self._default_style,
            'line_color':      [c.red(), c.green(), c.blue(), c.alpha()],
            'label_text':      '',
            'label_x_offset':  8.0,
            'label_y_offset':  -8.0,
            'label_font_size': self._default_font_size,
        }

    # ── selection ──────────────────────────────────────────────────────────

    def select_shape(self, shape_id: str | None):
        for sid, item in self._shape_items.items():
            item.setSelected(sid == shape_id)
        self._selected_id = shape_id
        self.shape_selected.emit(shape_id or '')
        self._update_overlays()

    def selected_id(self) -> str | None:
        return self._selected_id

    # ── delete ─────────────────────────────────────────────────────────────

    def delete_shape(self, shape_id: str):
        if shape_id in self._shape_items:
            item = self._shape_items.pop(shape_id)
            self._shape_styles.pop(shape_id, None)
            self.scene().removeItem(item)
            if self._selected_id == shape_id:
                self._selected_id = None
                self.shape_selected.emit('')
            self.shapes_changed.emit()
            self._update_overlays()

    def clear_all(self):
        for sid in list(self._shape_items):
            item = self._shape_items.pop(sid)
            self.scene().removeItem(item)
        self._shape_styles.clear()
        self._selected_id = None
        self._drawing = self._drawing_polygon = False
        self._poly_pts = QPolygonF()
        self._active_draw_id = None
        self.shapes_changed.emit()
        self.shape_selected.emit('')
        self._update_overlays()

    # ── style helpers ──────────────────────────────────────────────────────

    def update_shape_style(self, shape_id: str, key: str, value):
        """Update one style key for a specific shape and re-apply visuals."""
        if shape_id in self._shape_styles:
            self._shape_styles[shape_id][key] = value
            self._apply_style(shape_id)

    def get_shape_style(self, shape_id: str) -> dict:
        return dict(self._shape_styles.get(shape_id, {}))

    def get_shape_geometry(self, shape_id: str) -> dict:
        """Return geometry dict: x, y (center/start), w, h, font_size."""
        item = self._shape_items.get(shape_id)
        if item is None:
            return {}
        if isinstance(item, (CustomEllipseItem, CustomRectangleItem)):
            r = item.rect()
            c = item.mapToScene(r.center())
            return {'x': c.x(), 'y': c.y(),
                    'w': r.width(), 'h': r.height(), 'font_size': 0}
        if isinstance(item, QGraphicsPolygonItem):
            br = item.boundingRect()
            c = item.mapToScene(br.center())
            return {'x': c.x(), 'y': c.y(),
                    'w': br.width(), 'h': br.height(), 'font_size': 0}
        if isinstance(item, CustomCurveItem):
            import math
            pos = item.pos()
            p1, p2 = item.endpoints()
            chord = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
            return {'x': p1.x() + pos.x(), 'y': p1.y() + pos.y(),
                    'w': chord, 'h': 0, 'font_size': 0}
        if isinstance(item, CustomArrowItem):
            ln = item.line()
            pos = item.pos()
            return {'x': ln.x1() + pos.x(), 'y': ln.y1() + pos.y(),
                    'w': ln.length(), 'h': 0, 'font_size': 0}
        if isinstance(item, QGraphicsSimpleTextItem):
            pos = item.pos()
            st = self._shape_styles.get(shape_id, {})
            return {'x': pos.x(), 'y': pos.y(), 'w': 0, 'h': 0,
                    'font_size': float(st.get('label_font_size', 12.0))}
        return {}

    def set_shape_geometry(self, shape_id: str, x: float, y: float,
                           w: float, h: float, *, notify: bool = True):
        """Reposition / resize a shape on the canvas."""
        item = self._shape_items.get(shape_id)
        if item is None:
            return
        if isinstance(item, (CustomEllipseItem, CustomRectangleItem)):
            new_rect = QRectF(x - w / 2, y - h / 2, w, h)
            item.setRect(new_rect)
            item.setTransformOriginPoint(new_rect.center())
            item.curr_origin = new_rect.center()
        elif isinstance(item, QGraphicsPolygonItem):
            # Scale polygon proportionally around centroid
            old_br = item.boundingRect()
            if old_br.width() < 1 or old_br.height() < 1:
                return
            old_c = item.mapToScene(old_br.center())
            sx = w / old_br.width() if old_br.width() > 0 else 1
            sy = h / old_br.height() if old_br.height() > 0 else 1
            poly = item.polygon()
            new_poly = QPolygonF()
            for p in poly:
                sp = item.mapToScene(p)
                np_ = QPointF(x + (sp.x() - old_c.x()) * sx,
                              y + (sp.y() - old_c.y()) * sy)
                new_poly.append(item.mapFromScene(np_))
            item.setPolygon(new_poly)
        elif isinstance(item, CustomCurveItem):
            import math
            pos = item.pos()
            p1, p2 = item.endpoints()
            old_chord = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
            new_chord = max(1.0, w)
            if old_chord < 0.01:
                dx, dy = new_chord, 0
            else:
                dx = (p2.x() - p1.x()) / old_chord * new_chord
                dy = (p2.y() - p1.y()) / old_chord * new_chord
            item.setPos(0, 0)
            new_p1 = QPointF(x, y)
            new_p2 = QPointF(x + dx, y + dy)
            item.setEndpoints(new_p1, new_p2, recompute_cp=False)
            # Scale control points proportionally
            if old_chord > 0.01:
                scale = new_chord / old_chord
                cp1, cp2 = item.controlPoints()
                offset1 = QPointF(cp1.x() - p1.x(), cp1.y() - p1.y())
                offset2 = QPointF(cp2.x() - p1.x(), cp2.y() - p1.y())
                item.setControlPoints(
                    QPointF(x + offset1.x() * scale, y + offset1.y() * scale),
                    QPointF(x + offset2.x() * scale, y + offset2.y() * scale))
        elif isinstance(item, CustomArrowItem):
            # Preserve direction, set new start and length
            ln = item.line()
            pos = item.pos()
            length = max(1.0, w)
            old_len = ln.length()
            if old_len < 0.01:
                dx, dy = length, 0
            else:
                dx = (ln.x2() - ln.x1()) / old_len * length
                dy = (ln.y2() - ln.y1()) / old_len * length
            item.setPos(0, 0)
            item.setLine(x, y, x + dx, y + dy)
        elif isinstance(item, QGraphicsSimpleTextItem):
            item.setPos(x, y)
        self._update_overlays(shape_id)
        if notify:
            self.shape_committed.emit(shape_id)
            self.shapes_changed.emit()

    def _apply_style(self, shape_id: str):
        item = self._shape_items.get(shape_id)
        st = self._shape_styles.get(shape_id, {})
        if item is None:
            return
        c = st.get('line_color', [0, 220, 220, 255])
        color = QColor(int(c[0]), int(c[1]), int(c[2]),
                        int(c[3]) if len(c) > 3 else 255)
        width = float(st.get('line_width', 2.0))
        ps = self._PEN_STYLE_MAP.get(
            st.get('line_style', 'solid'), Qt.PenStyle.SolidLine)
        pen = QPen(color, width, ps,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        item.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(50)
        if (hasattr(item, 'setBrush')
                and not isinstance(item,
                                   (QGraphicsSimpleTextItem, CustomCurveItem))):
            item.setBrush(QBrush(fill))
        # Text items
        if isinstance(item, QGraphicsSimpleTextItem):
            item.setBrush(QBrush(color))
            font = item.font()
            font.setPointSizeF(max(4.0, float(st.get('label_font_size', 12.0))))
            font.setWeight(QFont.Weight.Light)
            item.setFont(font)
        # Arrow / curve label
        if isinstance(item, (CustomArrowItem, CustomCurveItem)):
            item.set_label(str(st.get('label_text', '')))
            item.set_label_offset(
                float(st.get('label_x_offset', 8.0)),
                float(st.get('label_y_offset', -8.0)))
            item.set_font_size(float(st.get('label_font_size', 12.0)))

    # ── image ──────────────────────────────────────────────────────────────

    def load_image(self, image):
        """*image* may be a numpy array or PIL Image."""
        for item in list(self.scene().items()):
            if isinstance(item, QGraphicsPixmapItem):
                self.scene().removeItem(item)
        arr = _ensure_display_rgb(image)
        h, w, _ = arr.shape
        q_img = QImage(arr.data, w, h, 3 * w,
                       QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_img)
        px_item = QGraphicsPixmapItem(pixmap)
        px_item.setZValue(-1)
        self.scene().addItem(px_item)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(px_item, Qt.AspectRatioMode.KeepAspectRatio)
        # Deferred re-fit: after the parent proxy widget finishes layout
        # (draw_node → setTitleAlign → stylesheet change → re-layout),
        # the viewport may have changed.  Re-fit once Qt has processed
        # pending events.
        self._px_item_ref = px_item
        QtCore.QTimer.singleShot(0, self._deferred_fit)

    def _deferred_fit(self):
        px = getattr(self, '_px_item_ref', None)
        if px is not None and px.scene() is not None:
            self.fitInView(px, Qt.AspectRatioMode.KeepAspectRatio)

    # ── zoom ───────────────────────────────────────────────────────────────

    def zoom_in(self):
        self._apply_zoom(1.2)

    def zoom_out(self):
        self._apply_zoom(1 / 1.2)

    def zoom_reset(self):
        for item in self.scene().items():
            if isinstance(item, QGraphicsPixmapItem):
                self.fitInView(item, Qt.AspectRatioMode.KeepAspectRatio)
                return

    def _apply_zoom(self, factor: float):
        self.scale(factor, factor)
        self._scale *= factor

    # ── mouse events ───────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        # --- middle-button pan ---
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = _mouse_pos_qpoint(event)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        scene_pos = self.mapToScene(_mouse_pos_qpoint(event))

        # --- hit-test existing items (skip during polygon drawing) ---
        if not self._drawing_polygon:
            hit = self.scene().itemAt(scene_pos, self.transform())
            # Let curve control-point handles drag independently
            # We handle this manually because Qt's default drag moves
            # all selected items (including the parent curve).
            if isinstance(hit, _CurveHandle):
                self._dragging_handle = hit
                event.accept()
                return
            hit_id = self._id_of(hit)
            if hit_id is not None:
                self.select_shape(hit_id)
                super().mousePressEvent(event)
                return

        # --- text shape: click-to-place ---
        if self._curr_shape == 'text':
            text, ok = QtWidgets.QInputDialog.getText(
                self, 'Text Annotation', 'Enter text:')
            if ok and text.strip():
                sid = self._new_id()
                item = QGraphicsSimpleTextItem(text.strip())
                item.setPos(scene_pos)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
                item.setData(0, sid)
                item.setZValue(0)
                self.scene().addItem(item)
                st = self._current_default_style()
                self._shape_items[sid] = item
                self._shape_styles[sid] = st
                self._apply_style(sid)
                self.select_shape(sid)
                self.shape_committed.emit(sid)
                self.shapes_changed.emit()
            return

        # --- ellipse / rectangle / arrow / curve: start drawing ---
        if self._curr_shape in ('ellipse', 'rectangle', 'arrow', 'curve'):
            sid = self._new_id()
            self._drawing = True
            self._start_pt = scene_pos
            self._active_draw_id = sid
            if self._curr_shape == 'ellipse':
                item = CustomEllipseItem(QRectF(scene_pos, scene_pos))
            elif self._curr_shape == 'rectangle':
                item = CustomRectangleItem(QRectF(scene_pos, scene_pos))
            elif self._curr_shape == 'curve':
                item = CustomCurveItem(scene_pos.x(), scene_pos.y(),
                                       scene_pos.x(), scene_pos.y())
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
                item.setFlag(
                    QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
            else:
                item = CustomArrowItem(scene_pos.x(), scene_pos.y(),
                                       scene_pos.x(), scene_pos.y())
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            item.setData(0, sid)
            st = self._current_default_style()
            self._shape_items[sid] = item
            self._shape_styles[sid] = st
            self.scene().addItem(item)
            self._apply_style(sid)
            if isinstance(item, (CustomArrowItem, CustomCurveItem)):
                item.set_label(st.get('label_text', ''))
                item.set_label_offset(
                    st.get('label_x_offset', 8.0),
                    st.get('label_y_offset', -8.0))
                item.set_font_size(st.get('label_font_size', 12.0))
            return

        # --- polygon: vertex by vertex ---
        if self._curr_shape == 'polygon':
            if not self._drawing_polygon:
                sid = self._new_id()
                self._active_draw_id = sid
                self._poly_pts = QPolygonF()
                self._poly_pts.append(scene_pos)
                item = QGraphicsPolygonItem(QPolygonF(self._poly_pts))
                item.setData(0, sid)
                st = self._current_default_style()
                self._shape_items[sid] = item
                self._shape_styles[sid] = st
                self.scene().addItem(item)
                self._apply_style(sid)
                self._drawing_polygon = True
            else:
                self._poly_pts.append(scene_pos)
                self._shape_items[self._active_draw_id].setPolygon(
                    QPolygonF(self._poly_pts))
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Curve control-point handle drag
        if self._dragging_handle is not None:
            scene_pos = self.mapToScene(_mouse_pos_qpoint(event))
            parent = self._dragging_handle.parentItem()
            local_pos = parent.mapFromScene(scene_pos)
            self._dragging_handle.setPos(local_pos)
            event.accept()
            return

        # Pan
        if self._pan_start is not None:
            cur = _mouse_pos_qpoint(event)
            delta = cur - self._pan_start
            self._pan_start = cur
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        scene_pos = self.mapToScene(_mouse_pos_qpoint(event))
        self.mouse_moved.emit(scene_pos.x(), scene_pos.y())

        if self._drawing and self._active_draw_id:
            item = self._shape_items.get(self._active_draw_id)
            if item is None:
                pass
            elif self._curr_shape in ('ellipse', 'rectangle'):
                rect = QRectF(self._start_pt, scene_pos).normalized()
                # Shift-constrain → square / circle
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    side = max(rect.width(), rect.height())
                    # preserve the start corner
                    x0 = self._start_pt.x()
                    y0 = self._start_pt.y()
                    dx = 1 if scene_pos.x() >= x0 else -1
                    dy = 1 if scene_pos.y() >= y0 else -1
                    rect = QRectF(x0, y0, dx * side, dy * side).normalized()
                item.setRect(rect)
                self._update_overlays(self._active_draw_id)
            elif self._curr_shape == 'arrow' and isinstance(item, QGraphicsLineItem):
                item.setLine(self._start_pt.x(), self._start_pt.y(),
                             scene_pos.x(), scene_pos.y())
                self._update_overlays(self._active_draw_id)
            elif self._curr_shape == 'curve' and isinstance(item, CustomCurveItem):
                item.setEndpoints(self._start_pt, scene_pos,
                                  recompute_cp=True)
                self._update_overlays(self._active_draw_id)
        elif self._drawing_polygon and self._active_draw_id:
            item = self._shape_items.get(self._active_draw_id)
            if item is not None:
                tmp = QPolygonF(self._poly_pts)
                tmp.append(scene_pos)
                item.setPolygon(tmp)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # End curve control-point handle drag
        if self._dragging_handle is not None:
            self._dragging_handle = None
            self.shapes_changed.emit()
            if self._selected_id:
                self.shape_committed.emit(self._selected_id)
            event.accept()
            return

        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._drawing and self._active_draw_id:
                self._drawing = False
                sid = self._active_draw_id
                self._active_draw_id = None
                item = self._shape_items.get(sid)
                if item is None:
                    return

                if self._curr_shape in ('ellipse', 'rectangle'):
                    r = item.rect()
                    if r.width() < 6 or r.height() < 6:
                        self._remove_item(sid)
                        return
                    center = r.center()
                    item.setTransformOriginPoint(center)
                    item.curr_origin = center
                elif self._curr_shape == 'arrow' and isinstance(item, QGraphicsLineItem):
                    if item.line().length() < 6:
                        self._remove_item(sid)
                        return
                elif self._curr_shape == 'curve' and isinstance(item, CustomCurveItem):
                    import math
                    p1, p2 = item.endpoints()
                    if math.hypot(p2.x() - p1.x(), p2.y() - p1.y()) < 6:
                        self._remove_item(sid)
                        return

                self.select_shape(sid)
                self.shape_committed.emit(sid)
                self.shapes_changed.emit()
                return

            # Not drawing — could be move/resize of existing item
            if not self._drawing_polygon and self._selected_id:
                super().mouseReleaseEvent(event)
                self.shape_committed.emit(self._selected_id)
                self.shapes_changed.emit()
                self._update_overlays()
                return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Double-click on a text item to re-edit it."""
        scene_pos = self.mapToScene(_mouse_pos_qpoint(event))
        hit = self.scene().itemAt(scene_pos, self.transform())
        hit_id = self._id_of(hit)
        if hit_id and isinstance(self._shape_items.get(hit_id),
                                 QGraphicsSimpleTextItem):
            item = self._shape_items[hit_id]
            text, ok = QtWidgets.QInputDialog.getText(
                self, 'Edit Text', 'Text:', text=item.text())
            if ok and text.strip():
                item.setText(text.strip())
                self.shape_committed.emit(hit_id)
                self.shapes_changed.emit()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._drawing_polygon:
            self._finish_polygon()
            return
        if key == Qt.Key.Key_Backspace and self._drawing_polygon:
            pts = list(self._poly_pts)
            if pts:
                pts.pop()
                self._poly_pts = QPolygonF(pts)
                item = self._shape_items.get(self._active_draw_id)
                if item is not None:
                    cursor_scene = self.mapToScene(
                        self.mapFromGlobal(QtGui.QCursor.pos()))
                    tmp = QPolygonF(self._poly_pts)
                    tmp.append(cursor_scene)
                    item.setPolygon(tmp)
            return
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self._selected_id:
                self.delete_shape(self._selected_id)
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._apply_zoom(factor)
        event.accept()

    # ── polygon finish ─────────────────────────────────────────────────────

    def _finish_polygon(self):
        self._drawing_polygon = False
        sid = self._active_draw_id
        self._active_draw_id = None
        item = self._shape_items.get(sid)
        if item is not None and len(self._poly_pts) >= 3:
            item.setPolygon(self._poly_pts)
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            self.select_shape(sid)
            self.shape_committed.emit(sid)
            self.shapes_changed.emit()
        elif sid and sid in self._shape_items:
            self._remove_item(sid)
        self._poly_pts = QPolygonF()

    # ── internal helpers ───────────────────────────────────────────────────

    def _remove_item(self, sid: str):
        item = self._shape_items.pop(sid, None)
        self._shape_styles.pop(sid, None)
        if item is not None:
            self.scene().removeItem(item)
        if self._selected_id == sid:
            self._selected_id = None
            self.shape_selected.emit('')
        self._update_overlays()

    def _id_of(self, item: QGraphicsItem | None) -> str | None:
        """Walk up to find a tracked shape item and return its ID."""
        while item is not None:
            sid = item.data(0)
            if isinstance(sid, str) and sid in self._shape_items:
                return sid
            item = item.parentItem()
        return None

    def _update_overlays(self, shape_id: str | None = None):
        sid = shape_id or self._selected_id
        if sid is None or sid not in self._shape_items:
            self._dim_overlay.setVisible(False)
            self._coord_overlay.setVisible(False)
            return

        item = self._shape_items[sid]
        self._dim_overlay.setVisible(True)
        self._coord_overlay.setVisible(True)

        if isinstance(item, (CustomEllipseItem, CustomRectangleItem)):
            r = item.rect()
            w, h = round(r.width(), 1), round(r.height(), 1)
            center = item.mapToScene(r.center())
            self._dim_overlay.setText(f'{w:.1f} \u00d7 {h:.1f}')
            self._coord_overlay.setText(
                f'({center.x():.1f}, {center.y():.1f})')
            br = item.mapToScene(r.bottomRight())
            self._dim_overlay.setPos(br.x() + 4, br.y() + 2)
            self._coord_overlay.setPos(br.x() + 4, br.y() + 18)
        elif isinstance(item, QGraphicsPolygonItem):
            br = item.boundingRect()
            w, h = round(br.width(), 1), round(br.height(), 1)
            center = item.mapToScene(br.center())
            self._dim_overlay.setText(f'{w:.1f} \u00d7 {h:.1f} (bbox)')
            self._coord_overlay.setText(
                f'({center.x():.1f}, {center.y():.1f})')
            brs = item.mapToScene(br.bottomRight())
            self._dim_overlay.setPos(brs.x() + 4, brs.y() + 2)
            self._coord_overlay.setPos(brs.x() + 4, brs.y() + 18)
        elif isinstance(item, CustomCurveItem):
            import math
            pos = item.pos()
            p1, p2 = item.endpoints()
            chord = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
            self._dim_overlay.setText(f'Curve L={chord:.1f}')
            self._coord_overlay.setText(
                f'({p1.x()+pos.x():.1f},{p1.y()+pos.y():.1f})'
                f'\u2192({p2.x()+pos.x():.1f},{p2.y()+pos.y():.1f})')
            self._dim_overlay.setPos(
                p2.x() + pos.x() + 4, p2.y() + pos.y() + 2)
            self._coord_overlay.setPos(
                p2.x() + pos.x() + 4, p2.y() + pos.y() + 18)
        elif isinstance(item, CustomArrowItem):
            ln = item.line()
            pos = item.pos()
            length = ln.length()
            self._dim_overlay.setText(f'L={length:.1f}')
            self._coord_overlay.setText(
                f'({ln.x1()+pos.x():.1f},{ln.y1()+pos.y():.1f})'
                f'\u2192({ln.x2()+pos.x():.1f},{ln.y2()+pos.y():.1f})')
            self._dim_overlay.setPos(
                ln.x2() + pos.x() + 4, ln.y2() + pos.y() + 2)
            self._coord_overlay.setPos(
                ln.x2() + pos.x() + 4, ln.y2() + pos.y() + 18)
        elif isinstance(item, QGraphicsSimpleTextItem):
            pos = item.pos()
            br = item.boundingRect()
            self._dim_overlay.setText(f'Text')
            self._coord_overlay.setText(f'({pos.x():.1f}, {pos.y():.1f})')
            self._dim_overlay.setPos(pos.x() + br.width() + 4, pos.y())
            self._coord_overlay.setPos(
                pos.x() + br.width() + 4, pos.y() + 16)
        else:
            self._dim_overlay.setVisible(False)
            self._coord_overlay.setVisible(False)

    # ── serialisation ──────────────────────────────────────────────────────

    def get_all_shapes_data(self) -> list:
        result = []
        for sid, item in self._shape_items.items():
            d = self._build_roi_dict(item)
            if d is not None:
                d['id'] = sid
                d.update(self._shape_styles.get(sid, {}))
                result.append(d)
        return result

    def load_all_shapes_data(self, shapes: list):
        self.clear_all()
        for sd in shapes:
            sid = sd.get('id', self._new_id())
            # update next_id counter
            if sid.startswith('shape_'):
                try:
                    num = int(sid.split('_', 1)[1])
                    if num >= self._next_id:
                        self._next_id = num + 1
                except ValueError:
                    pass
            item = self._create_item_from_dict(sd)
            if item is None:
                continue
            item.setData(0, sid)
            self.scene().addItem(item)
            self._shape_items[sid] = item
            self._shape_styles[sid] = {
                'line_width':      sd.get('line_width', 2.0),
                'line_style':      sd.get('line_style', 'solid'),
                'line_color':      sd.get('line_color', [0, 220, 220, 255]),
                'label_text':      sd.get('label_text', ''),
                'label_x_offset':  sd.get('label_x_offset', 8.0),
                'label_y_offset':  sd.get('label_y_offset', -8.0),
                'label_font_size': sd.get('label_font_size', 12.0),
            }
            self._apply_style(sid)

    # ── mask contour rendering (locked, non-interactive) ──────────────

    def set_mask_contours(self, contour_data: list[list[list[tuple[float, float]]]],
                          styles: list[dict]):
        """Display mask contours in the scene as locked path items.

        *contour_data* is a list of masks, each being a list of contour paths.
        *styles* is a parallel list of style dicts (line_color, line_width, line_style).
        """
        # Remove old contour items
        for item in self._mask_contour_items:
            self.scene().removeItem(item)
        self._mask_contour_items.clear()
        for item in self._mask_bbox_items.values():
            self.scene().removeItem(item)
        self._mask_bbox_items.clear()

        palette = [(255, 80, 80), (80, 200, 80), (80, 120, 255),
                   (255, 200, 50), (200, 80, 255)]
        pen_style_map = self._PEN_STYLE_MAP

        for i, paths in enumerate(contour_data):
            st = styles[i] if i < len(styles) else {}
            c = st.get('line_color', list(palette[i % len(palette)]))
            alpha = int(c[3]) if len(c) > 3 else 255
            color = QColor(int(c[0]), int(c[1]), int(c[2]))
            width = float(st.get('line_width', 2.0))
            style = pen_style_map.get(st.get('line_style', 'solid'),
                                      Qt.PenStyle.SolidLine)
            fill_mode = bool(st.get('fill_mode', False))

            all_x, all_y = [], []
            for pts in paths:
                if len(pts) < 2:
                    continue
                path = QPainterPath()
                path.moveTo(pts[0][0], pts[0][1])
                for x, y in pts[1:]:
                    path.lineTo(x, y)
                if fill_mode:
                    path.closeSubpath()
                item = QGraphicsPathItem(path)
                pen = QPen(color, width, style)
                pen.setCosmetic(True)
                item.setPen(pen)
                if fill_mode:
                    fill_color = QColor(int(c[0]), int(c[1]), int(c[2]), alpha)
                    item.setBrush(QBrush(fill_color))
                else:
                    item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                item.setZValue(-1)   # behind user shapes
                self.scene().addItem(item)
                self._mask_contour_items.append(item)
                for x, y in pts:
                    all_x.append(x)
                    all_y.append(y)

            # Bounding box per mask — hidden until selected in list
            mid = f'mask_{i}'
            if all_x:
                x0, x1 = min(all_x), max(all_x)
                y0, y1 = min(all_y), max(all_y)
                bbox_path = QPainterPath()
                bbox_path.addRect(QRectF(x0, y0, x1 - x0, y1 - y0))
                bbox_item = QGraphicsPathItem(bbox_path)
                bbox_color = QColor(color)
                bbox_color.setAlpha(140)
                bbox_pen = QPen(bbox_color, 1.0, Qt.PenStyle.DashLine)
                bbox_pen.setCosmetic(True)
                bbox_item.setPen(bbox_pen)
                bbox_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                bbox_item.setZValue(-1)
                bbox_item.setVisible(False)
                self.scene().addItem(bbox_item)
                self._mask_bbox_items[mid] = bbox_item

    def update_mask_contour_style(self, index: int, style: dict):
        """Re-apply style to all contour items belonging to mask *index*."""
        # We need to know which items belong to which mask.
        # Since we rebuild all contours at once, we store per-mask ranges.
        # For simplicity, just call set_mask_contours again from the widget.
        pass

    def show_mask_bbox(self, mask_id: str | None):
        """Show bounding box for *mask_id*, hide all others."""
        for mid, item in self._mask_bbox_items.items():
            item.setVisible(mid == mask_id)

    def clear_mask_contours(self):
        for item in self._mask_contour_items:
            self.scene().removeItem(item)
        self._mask_contour_items.clear()
        for item in self._mask_bbox_items.values():
            self.scene().removeItem(item)
        self._mask_bbox_items.clear()

    def _build_roi_dict(self, item) -> dict | None:
        if isinstance(item, (CustomEllipseItem, CustomRectangleItem)):
            rect = item.rect()
            center = item.mapToScene(rect.center())
            return {
                'shape': ('ellipse' if isinstance(item, CustomEllipseItem)
                          else 'rectangle'),
                'center': [round(center.x(), 2), round(center.y(), 2)],
                'axes':   [round(rect.width() / 2, 2),
                           round(rect.height() / 2, 2)],
                'angle':  round(float(item.rotation()), 4),
                'polypoints': [],
            }
        if isinstance(item, QGraphicsPolygonItem):
            poly = item.polygon()
            pos = item.pos()
            return {
                'shape': 'polygon',
                'center': [], 'axes': [], 'angle': 0.0,
                'polypoints': [[round(p.x() + pos.x(), 2),
                                round(p.y() + pos.y(), 2)] for p in poly],
            }
        if isinstance(item, CustomCurveItem):
            pos = item.pos()
            p1, p2 = item.endpoints()
            cp1, cp2 = item.controlPoints()
            return {
                'shape': 'curve',
                'center': [], 'axes': [], 'angle': 0.0,
                'polypoints': [],
                'points': [
                    [round(p1.x() + pos.x(), 2),
                     round(p1.y() + pos.y(), 2)],
                    [round(p2.x() + pos.x(), 2),
                     round(p2.y() + pos.y(), 2)],
                ],
                'control_points': [
                    [round(cp1.x() + pos.x(), 2),
                     round(cp1.y() + pos.y(), 2)],
                    [round(cp2.x() + pos.x(), 2),
                     round(cp2.y() + pos.y(), 2)],
                ],
            }
        if isinstance(item, CustomArrowItem):
            ln = item.line()
            pos = item.pos()
            return {
                'shape': 'arrow',
                'center': [], 'axes': [], 'angle': 0.0,
                'polypoints': [],
                'points': [
                    [round(ln.x1() + pos.x(), 2),
                     round(ln.y1() + pos.y(), 2)],
                    [round(ln.x2() + pos.x(), 2),
                     round(ln.y2() + pos.y(), 2)],
                ],
            }
        if isinstance(item, QGraphicsSimpleTextItem):
            pos = item.pos()
            return {
                'shape': 'text',
                'center': [], 'axes': [], 'angle': 0.0,
                'polypoints': [],
                'text_pos': [round(pos.x(), 2), round(pos.y(), 2)],
                'text_content': item.text(),
            }
        return None

    def _create_item_from_dict(self, sd: dict) -> QGraphicsItem | None:
        shape = sd.get('shape')
        if shape in ('ellipse', 'rectangle'):
            cx = float(sd['center'][0])
            cy = float(sd['center'][1])
            ax = float(sd['axes'][0])
            ay = float(sd['axes'][1])
            rect = QRectF(cx - ax, cy - ay, 2 * ax, 2 * ay)
            cls = CustomEllipseItem if shape == 'ellipse' else CustomRectangleItem
            item = cls(rect)
            center = rect.center()
            item.setTransformOriginPoint(center)
            item.curr_origin = center
            item.setRotation(float(sd.get('angle', 0)))
            return item
        if shape == 'polygon':
            pts = sd.get('polypoints', [])
            if len(pts) >= 3:
                poly = QPolygonF([QPointF(p[0], p[1]) for p in pts])
                item = QGraphicsPolygonItem(poly)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
                return item
        if shape == 'curve':
            pts = sd.get('points', [])
            cps = sd.get('control_points', [])
            if isinstance(pts, list) and len(pts) == 2:
                p1, p2 = pts
                item = CustomCurveItem(
                    float(p1[0]), float(p1[1]),
                    float(p2[0]), float(p2[1]))
                if isinstance(cps, list) and len(cps) == 2:
                    item.setControlPoints(
                        QPointF(float(cps[0][0]), float(cps[0][1])),
                        QPointF(float(cps[1][0]), float(cps[1][1])))
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
                item.setFlag(
                    QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
                return item
        if shape == 'arrow':
            pts = sd.get('points', [])
            if isinstance(pts, list) and len(pts) == 2:
                p1, p2 = pts
                item = CustomArrowItem(
                    float(p1[0]), float(p1[1]),
                    float(p2[0]), float(p2[1]))
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
                return item
        if shape == 'text':
            text = sd.get('text_content', '')
            pos = sd.get('text_pos', [0, 0])
            item = QGraphicsSimpleTextItem(text)
            item.setPos(float(pos[0]), float(pos[1]))
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            return item
        return None


# ── Node widget (toolbar + embedded view + rotation row) ─────────────────────

class NodeROIViewWidget(NodeBaseWidget):
    """
    Embeds the ROI drawing view directly on the node surface.

    Emits `roi_committed(dict)` whenever the user finalises a shape. Thread-safe image and ROI loading via cross-thread signals.
    """

    roi_committed = Signal(dict)
    _img_signal   = Signal(object)   # PIL Image -> main thread
    _roi_signal   = Signal(object)   # roi dict  -> main thread
    _VIEW_MAX_W = 640
    _VIEW_MAX_H = 560
    _VIEW_MIN_W = 260
    _VIEW_MIN_H = 220
    _VIEW_DEFAULT_W = 500
    _VIEW_DEFAULT_H = 500

    def __init__(self, parent=None):
        super().__init__(parent, name='_roi_view', label='')

        container = QtWidgets.QWidget()
        container.setMinimumWidth(320)
        root = QtWidgets.QVBoxLayout(container)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(3)

        # ── toolbar ──────────────────────────────────────────────────────────
        tb = QtWidgets.QHBoxLayout()
        self._shape_group = QtWidgets.QButtonGroup(container)
        for i, (label, key) in enumerate([
                ('Ellipse',    'ellipse'),
                ('Rectangle',  'rectangle'),
                ('Polygon',    'polygon'),
                ('Lasso',      'lasso'),
        ]):
            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True)
            btn.setProperty('shape_key', key)
            btn.setFixedHeight(22)
            if i == 0:
                btn.setChecked(True)
            self._shape_group.addButton(btn, i)
            tb.addWidget(btn)

        tb.addStretch()

        # Zoom buttons
        for icon, slot, tip in (('+', '_zoom_in', 'Zoom in'),
                                ('-', '_zoom_out', 'Zoom out'),
                                ('⊙', '_zoom_reset', 'Fit to view')):
            b = QtWidgets.QPushButton(icon)
            b.setFixedSize(52, 52)
            b.setProperty('compact', True)
            b.setStyleSheet('font-size: 18px; font-weight: 600; padding: 0px;')
            b.setToolTip(tip)
            b.clicked.connect(getattr(self, slot))
            tb.addWidget(b)

        # Color swatch button — shows current pen color, opens picker on click
        self._color_btn = QtWidgets.QPushButton()
        self._color_btn.setFixedSize(22, 22)
        self._color_btn.setToolTip('Shape color')
        tb.addWidget(self._color_btn)

        clear_btn = QtWidgets.QPushButton('Clear')
        clear_btn.setFixedHeight(22)
        tb.addWidget(clear_btn)
        root.addLayout(tb)

        # ── drawing view ─────────────────────────────────────────────────────
        self._scene = QGraphicsScene()
        self._view  = ROIGraphicsView(self._scene)
        self._view.setFixedSize(self._VIEW_DEFAULT_W, self._VIEW_DEFAULT_H)
        root.addWidget(self._view)

        # ── rotation row (hidden for polygon) ────────────────────────────────
        rot_row = QtWidgets.QHBoxLayout()
        rot_row.addWidget(QtWidgets.QLabel('Rotation:'))
        self._rot_slider  = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._rot_slider.setRange(-180, 180)
        self._rot_slider.setValue(0)
        self._rot_spinbox = QtWidgets.QSpinBox()
        self._rot_spinbox.setRange(-180, 180)
        self._rot_spinbox.setFixedWidth(58)
        rot_row.addWidget(self._rot_slider, stretch=1)
        rot_row.addWidget(self._rot_spinbox)
        self._rot_row_widget = QtWidgets.QWidget()
        self._rot_row_widget.setLayout(rot_row)
        root.addWidget(self._rot_row_widget)

        # ── tip label ─────────────────────────────────────────────────────────
        self._tip = QtWidgets.QLabel()
        self._tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tip.setStyleSheet('color:#999; font-size:9px; padding:1px;')
        root.addWidget(self._tip)

        self._container = container
        self.set_custom_widget(container)

        # ── wire up ───────────────────────────────────────────────────────────
        self._shape_group.idClicked.connect(self._on_shape_btn)
        clear_btn.clicked.connect(self._view.clear_roi)
        self._color_btn.clicked.connect(self._on_color_btn)
        self._rot_slider.valueChanged.connect(self._on_rot_slider)
        self._rot_spinbox.valueChanged.connect(self._on_rot_spinbox)
        self._view.shape_committed.connect(self._on_shape_committed)

        self._update_tip()
        self._update_rot_visibility()
        self._update_color_btn()

        # Thread-safe UI bridges
        self._img_signal.connect(self._apply_image, Qt.ConnectionType.QueuedConnection)
        self._roi_signal.connect(self._apply_roi_data, Qt.ConnectionType.QueuedConnection)

    # ── public helpers ────────────────────────────────────────────────────────

    def _compute_view_size(self, w: int, h: int) -> tuple[int, int]:
        if w <= 0 or h <= 0:
            return self._VIEW_DEFAULT_W, self._VIEW_DEFAULT_H
        scale = min(self._VIEW_MAX_W / float(w), self._VIEW_MAX_H / float(h))
        dw = int(round(w * scale))
        dh = int(round(h * scale))
        dw = max(self._VIEW_MIN_W, min(self._VIEW_MAX_W, dw))
        dh = max(self._VIEW_MIN_H, min(self._VIEW_MAX_H, dh))
        return dw, dh

    def _apply_view_size(self, w: int, h: int):
        dw, dh = self._compute_view_size(w, h)
        self._view.setFixedSize(dw, dh)
        # Resize top-level custom widget so node frame follows the internal view.
        self._container.adjustSize()
        hint = self._container.sizeHint()
        self._container.setFixedSize(hint)
        group = self.widget()
        if group:
            group.setMinimumSize(hint)
            group.resize(hint)
            group.adjustSize()
        self.resize(hint.width(), hint.height())
        self.updateGeometry()
        # Force NodeGraphQt to recompute node bounds from widget geometry.
        if self.node and self.node.view:
            self.node.view.draw_node()

    def _apply_image(self, image):
        w, h = _image_hw(image)
        self._apply_view_size(w, h)
        self._view.load_image(image)

    def load_image(self, image):
        if threading.current_thread() is threading.main_thread():
            self._apply_image(image)
        else:
            self._img_signal.emit(image)

    def _apply_roi_data(self, roi_data: dict):
        self._view.load_roi_data(roi_data)
        angle = int(roi_data.get('angle', 0))
        self._rot_slider.blockSignals(True)
        self._rot_spinbox.blockSignals(True)
        self._rot_slider.setValue(angle)
        self._rot_spinbox.setValue(angle)
        self._rot_slider.blockSignals(False)
        self._rot_spinbox.blockSignals(False)
        self._update_rot_visibility()

    def load_roi_data(self, roi_data: dict):
        if threading.current_thread() is threading.main_thread():
            self._apply_roi_data(roi_data)
        else:
            self._roi_signal.emit(roi_data)

    def clear_roi(self):
        self._view.clear_roi()

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_shape_btn(self, btn_id):
        key = self._shape_group.button(btn_id).property('shape_key')
        self._view.set_shape_type(key)
        self._update_tip()
        self._update_rot_visibility()

    def _on_rot_slider(self, value):
        self._rot_spinbox.blockSignals(True)
        self._rot_spinbox.setValue(value)
        self._rot_spinbox.blockSignals(False)
        self._view.set_rotation(float(value))
        roi = self._view.get_roi_data()
        if roi is not None:
            self.roi_committed.emit(roi)

    def _on_rot_spinbox(self, value):
        self._rot_slider.blockSignals(True)
        self._rot_slider.setValue(value)
        self._rot_slider.blockSignals(False)
        self._view.set_rotation(float(value))
        roi = self._view.get_roi_data()
        if roi is not None:
            self.roi_committed.emit(roi)

    def _on_shape_committed(self):
        # Reset rotation display to match actual item rotation
        roi = self._view.get_roi_data()
        if roi is not None:
            angle = int(roi.get('angle', 0))
            self._rot_slider.blockSignals(True)
            self._rot_spinbox.blockSignals(True)
            self._rot_slider.setValue(angle)
            self._rot_spinbox.setValue(angle)
            self._rot_slider.blockSignals(False)
            self._rot_spinbox.blockSignals(False)
            self._update_rot_visibility()
            self.roi_committed.emit(roi)

    def _on_color_btn(self):
        parent = QtWidgets.QApplication.activeWindow()
        color = QtWidgets.QColorDialog.getColor(
            self._view._pen_color, parent, 'Shape Color')
        if color.isValid():
            self._view.set_pen_color(color)
            self._update_color_btn()

    def _update_color_btn(self):
        c = self._view._pen_color
        self._color_btn.setStyleSheet(
            f'background-color: rgb({c.red()},{c.green()},{c.blue()});'
            f'border: 1px solid #555; border-radius: 3px;')

    def _zoom_in(self):    self._view.zoom_in()
    def _zoom_out(self):   self._view.zoom_out()
    def _zoom_reset(self): self._view.zoom_reset()

    def _update_tip(self):
        checked = self._shape_group.checkedButton()
        key = checked.property('shape_key') if checked else 'ellipse'
        tips = {
            'ellipse':    'Drag to draw  ·  drag edges to resize  ·  drag body to move',
            'rectangle':  'Drag to draw  ·  drag edges to resize  ·  drag body to move',
            'polygon':    'Click to add vertices  ·  Enter = close  ·  Backspace = undo  ·  Del = clear',
            'lasso':      'Click and drag to draw freehand  ·  Del = clear',
        }
        self._tip.setText(tips.get(key, ''))

    def _update_rot_visibility(self):
        checked = self._shape_group.checkedButton()
        key = checked.property('shape_key') if checked else 'ellipse'
        roi = self._view.get_roi_data()
        # Show rotation only for ellipse/rect shape types AND only when an
        # ellipse/rect item is actually drawn
        show = (key not in ('polygon', 'lasso') and
                roi is not None and
                roi.get('shape') in ('ellipse', 'rectangle'))
        self._rot_row_widget.setVisible(show)

    # ── NodeBaseWidget required interface ─────────────────────────────────────

    def get_value(self):  return ''
    def set_value(self, value): pass


# ── The node ──────────────────────────────────────────────────────────────────

class ROIMaskNode(BaseExecutionNode):
    """
    Draws an ROI (ellipse, rectangle, polygon, or lasso) directly on the node surface and outputs a binary mask plus a cropped image.

    Inputs:
    - **image** — the image to draw on (sets the background)

    Outputs:
    - **mask** — binary L-mode PIL image (0 / 255)
    - **cropped_image** — input image with non-ROI pixels set to black

    Keywords: roi, region of interest, polygon, ellipse, rectangle, 感興趣區域, 遮罩, 多邊形, 橢圓, 裁切
    """

    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME       = 'ROI Mask'
    PORT_SPEC       = {'inputs': ['image'], 'outputs': ['mask', 'image']}

    # Properties that must not trigger mark_dirty / re-eval
    _UI_PROPS = frozenset({'color', 'pos', 'selected', 'name', 'progress',
                           'image_view', 'roi_data'})

    def __init__(self):
        super().__init__()
        self.add_input('image',         color=PORT_COLORS['image'])
        self.add_output('mask',         color=PORT_COLORS['mask'])
        self.add_output('cropped_image', color=PORT_COLORS['image'])

        # Serialised ROI storage
        self.create_property('roi_data', '')

        # Inline drawing widget
        self._roi_widget = NodeROIViewWidget(self.view)
        self._roi_widget.roi_committed.connect(self._on_roi_committed)
        self.add_custom_widget(self._roi_widget)

        self._last_img_id: int | None = None   # track input image changes

    # ── ROI committed callback ────────────────────────────────────────────────

    def _on_roi_committed(self, roi: dict):
        """Called when the user finalises a shape in the embedded view."""
        # set_property → mark_dirty (cascades downstream) then evaluate this node.
        self.set_property('roi_data', json.dumps(roi))
        success, _ = self.evaluate()
        if success:
            self.mark_clean()
        else:
            self.mark_error()

    # ── evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self):
        self.reset_progress()

        # ── get input image ──────────────────────────────────────────────────
        port = self.inputs().get('image')
        if not port or not port.connected_ports():
            return False, "No input connected"
        up_node = port.connected_ports()[0].node()
        data    = up_node.output_values.get(port.connected_ports()[0].name())
        if not isinstance(data, ImageData):
            return False, "Input must be ImageData"

        img_arr  = data.payload          # numpy array (H, W) or (H, W, 3)
        img_id   = id(img_arr)

        # Load image into the embedded view if it has changed
        if img_id != self._last_img_id:
            self._roi_widget.load_image(img_arr)
            self._last_img_id = img_id
            # Restore any previously stored ROI overlay
            roi_raw = self.get_property('roi_data')
            if roi_raw:
                try:
                    self._roi_widget.load_roi_data(json.loads(roi_raw))
                except Exception:
                    pass

        self.set_progress(20)

        # ── generate mask ────────────────────────────────────────────────────
        roi_raw = self.get_property('roi_data')
        if not roi_raw:
            # No ROI drawn yet — not an error, just nothing to output
            self.set_progress(100)
            return True, None

        try:
            roi = json.loads(roi_raw)
        except (json.JSONDecodeError, TypeError):
            return False, "ROI data is corrupt"

        img_h, img_w = img_arr.shape[:2]
        self.set_progress(40)

        mask_arr = _roi_dict_to_mask_arr(roi, img_w, img_h)
        self.set_progress(70)

        self.output_values['mask'] = MaskData(payload=mask_arr)

        # ── cropped image (black outside ROI) ────────────────────────────────
        if img_arr.ndim == 2:
            rgb_arr = np.stack([img_arr]*3, axis=-1)
        else:
            rgb_arr = img_arr.copy()
        mask_bool  = mask_arr > 0
        cropped    = rgb_arr.copy()
        cropped[~mask_bool] = 0
        self._make_image_output(cropped, 'cropped_image')

        self.set_progress(100)
        return True, None


# ===========================================================================
# CropNode — interactive rectangle crop
# ===========================================================================

class NodeCropViewWidget(NodeBaseWidget):
    """
    Embeds a rectangle-only ROI view for cropping on the node surface.

    Emits `roi_committed(dict)` whenever the user finalises or moves the crop rectangle. Thread-safe: `load_image` and `load_roi_data` may be called from worker threads.
    """

    roi_committed = Signal(dict)
    _img_signal   = Signal(object)   # PIL Image  → main thread
    _roi_signal   = Signal(object)   # roi dict   → main thread
    _VIEW_MAX_W = 640
    _VIEW_MAX_H = 560
    _VIEW_MIN_W = 260
    _VIEW_MIN_H = 220
    _VIEW_DEFAULT_W = 500
    _VIEW_DEFAULT_H = 400

    def __init__(self, parent=None):
        super().__init__(parent, name='_crop_view', label='')

        container = QtWidgets.QWidget()
        # container.setMinimumWidth(320)
        root = QtWidgets.QVBoxLayout(container)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(3)

        # ── toolbar ──────────────────────────────────────────────────────────
        tb = QtWidgets.QHBoxLayout()
        tb.addWidget(QtWidgets.QLabel('Crop region:'))
        tb.addStretch()
        for icon, slot, tip in (('+', '_zoom_in', 'Zoom in'),
                                ('-', '_zoom_out', 'Zoom out'),
                                ('⊙', '_zoom_reset', 'Fit to view')):
            b = QtWidgets.QPushButton(icon)
            b.setFixedSize(52, 52)
            b.setProperty('compact', True)
            b.setStyleSheet('font-size: 18px; font-weight: 600; padding: 0px;')
            b.setToolTip(tip)
            b.clicked.connect(getattr(self, slot))
            tb.addWidget(b)
        clear_btn = QtWidgets.QPushButton('Clear')
        clear_btn.setFixedHeight(22)
        clear_btn.clicked.connect(self._on_clear)
        tb.addWidget(clear_btn)
        root.addLayout(tb)

        # ── drawing view (rectangle mode only) ───────────────────────────────
        self._scene = QGraphicsScene()
        self._view  = ROIGraphicsView(self._scene)
        self._view.set_shape_type('rectangle')
        self._view.setFixedSize(self._VIEW_DEFAULT_W, self._VIEW_DEFAULT_H)
        root.addWidget(self._view)

        tip = QtWidgets.QLabel(
            'Drag to draw crop rectangle · Drag edges to resize · Drag body to move · Delete to clear')
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tip.setStyleSheet('color:#999; font-size:9px; padding:1px;')
        tip.setWordWrap(True)
        root.addWidget(tip)

        self._container = container
        self.set_custom_widget(container)
        self._view.shape_committed.connect(self._on_shape_committed)

        # Thread-safe bridges: worker thread emits signal → main thread updates view
        self._img_signal.connect(self._apply_image, Qt.ConnectionType.QueuedConnection)
        self._roi_signal.connect(self._view.load_roi_data, Qt.ConnectionType.QueuedConnection)

    # ── public API ────────────────────────────────────────────────────────────

    def _compute_view_size(self, w: int, h: int) -> tuple[int, int]:
        if w <= 0 or h <= 0:
            return self._VIEW_DEFAULT_W, self._VIEW_DEFAULT_H
        scale = min(self._VIEW_MAX_W / float(w), self._VIEW_MAX_H / float(h))
        dw = int(round(w * scale))
        dh = int(round(h * scale))
        dw = max(self._VIEW_MIN_W, min(self._VIEW_MAX_W, dw))
        dh = max(self._VIEW_MIN_H, min(self._VIEW_MAX_H, dh))
        return dw, dh

    def _apply_view_size(self, w: int, h: int):
        dw, dh = self._compute_view_size(w, h)
        self._view.setFixedSize(dw, dh)
        # Resize top-level custom widget so node frame follows the internal view.
        self._container.adjustSize()
        hint = self._container.sizeHint()
        self._container.setFixedSize(hint)
        group = self.widget()
        if group:
            group.setMinimumSize(hint)
            group.resize(hint)
            group.adjustSize()
        self.resize(hint.width(), hint.height())
        self.updateGeometry()
        # Force NodeGraphQt to recompute node bounds from widget geometry.
        if self.node and self.node.view:
            self.node.view.draw_node()

    def _apply_image(self, image):
        w, h = _image_hw(image)
        self._apply_view_size(w, h)
        self._view.load_image(image)

    def load_image(self, image):
        if threading.current_thread() is threading.main_thread():
            self._apply_image(image)
        else:
            self._img_signal.emit(image)

    def load_roi_data(self, roi_data: dict):
        if threading.current_thread() is threading.main_thread():
            self._view.load_roi_data(roi_data)
        else:
            self._roi_signal.emit(roi_data)

    def clear_roi(self):
        self._view.clear_roi()

    # ── NodeBaseWidget required interface ─────────────────────────────────────

    def get_value(self):         return ''
    def set_value(self, _value): pass

    # ── zoom helpers ──────────────────────────────────────────────────────────

    def _zoom_in(self):    self._view.zoom_in()
    def _zoom_out(self):   self._view.zoom_out()
    def _zoom_reset(self): self._view.zoom_reset()

    # ── internal slots ────────────────────────────────────────────────────────

    def _on_clear(self):
        self._view.clear_roi()
        self.roi_committed.emit({})

    def _on_shape_committed(self):
        roi_data = self._view.get_roi_data()
        if roi_data is not None:
            self.roi_committed.emit(roi_data)


class CropNode(BaseExecutionNode):
    """
    Crops an image or mask to a rectangle drawn directly on the node.

    Click and drag on the node surface to draw the crop rectangle. Drag the edges to resize it, or drag the body to move it. Press Delete to clear the selection (outputs the full image when no rectangle is drawn). Supports both ImageData and MaskData inputs.

    Keywords: crop, trim image, rectangular roi, bounding box, cutout, 裁切, 感興趣區域, 遮罩, 邊界框, 剪裁
    """
    __identifier__ = 'nodes.image_process.Transform'
    NODE_NAME      = 'Crop'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}

    _UI_PROPS = frozenset({'color', 'pos', 'selected', 'name', 'progress',
                           'image_view', 'show_preview', 'live_preview', 'crop_rect'})

    def __init__(self):
        super().__init__()
        self.add_input('image',  color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'], multi_output=True)

        self.create_property('crop_rect', '')   # JSON string of roi_data dict

        self._crop_widget = NodeCropViewWidget(self.view)
        self._crop_widget.roi_committed.connect(self._on_roi_committed)
        self.add_custom_widget(self._crop_widget)

        self._last_img_id: int | None = None

    # ── ROI callback ──────────────────────────────────────────────────────────

    def _on_roi_committed(self, roi: dict):
        """Store rect and trigger re-evaluation without adding to undo stack."""
        super(BaseExecutionNode, self).set_property(
            'crop_rect', json.dumps(roi), push_undo=False)
        self.mark_dirty()

    # ── evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self):
        self.reset_progress()

        port = self.inputs().get('image')
        if not port or not port.connected_ports():
            return False, "No input connected"
        cp   = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())

        if isinstance(data, ImageData):
            arr_in  = data.payload      # numpy array
            out_cls = ImageData
        elif isinstance(data, MaskData):
            arr_in  = data.payload      # numpy array
            out_cls = MaskData
        else:
            return False, "Input must be ImageData or MaskData"

        self.set_progress(20)

        # Load image into widget when the upstream image changes
        img_id = id(arr_in)
        if img_id != self._last_img_id:
            self._crop_widget.load_image(arr_in)
            self._last_img_id = img_id
            # Restore saved rectangle overlay
            rect_raw = self.get_property('crop_rect')
            if rect_raw:
                try:
                    self._crop_widget.load_roi_data(json.loads(rect_raw))
                except Exception:
                    pass

        self.set_progress(40)

        # Get crop coordinates from stored roi_data
        rect_raw = self.get_property('crop_rect')
        if not rect_raw:
            self.output_values['image'] = data
            self.set_progress(100)
            return True, None

        try:
            roi = json.loads(rect_raw)
        except (json.JSONDecodeError, TypeError):
            return False, "Crop rect data is corrupt"

        if not roi:
            # Cleared — pass through
            self.output_values['image'] = data
            self.set_progress(100)
            return True, None

        H, W   = arr_in.shape[:2]
        cx, cy = float(roi['center'][0]), float(roi['center'][1])
        ax, ay = float(roi['axes'][0]),   float(roi['axes'][1])
        left   = max(0, int(round(cx - ax)))
        top    = max(0, int(round(cy - ay)))
        right  = min(W, int(round(cx + ax)))
        bottom = min(H, int(round(cy + ay)))
        right  = max(left + 1, right)
        bottom = max(top  + 1, bottom)

        self.set_progress(70)
        cropped = arr_in[top:bottom, left:right].copy()
        self.output_values['image'] = out_cls(payload=cropped)
        self.set_progress(100)
        return True, None


class MaskCropNode(BaseImageProcessNode):
    """
    Crops an image to a mask's bounding box.

    When **Black outside** is checked, pixels outside the mask are set to zero.
    When unchecked, all pixels within the bounding box are kept.
    **Padding** adds extra pixels around the bounding box.

    Keywords: mask crop, bounding box, extract region, cut out, trim, 遮罩裁切, 邊界框
    """
    __identifier__ = 'nodes.image_process.Transform'
    NODE_NAME      = 'Mask Crop'
    PORT_SPEC      = {'inputs': ['image', 'mask'], 'outputs': ['image']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_input('mask',  color=PORT_COLORS['mask'])
        self.add_output('image', color=PORT_COLORS['image'])

        self.create_property('black_outside', True)
        self.create_property('padding', 0)

        from NodeGraphQt.widgets.node_widgets import NodeBaseWidget as _NBW

        class _MaskCropWidget(_NBW):
            def __init__(self, parent):
                super().__init__(parent, name='_mask_crop_opts', label='')
                w = QtWidgets.QWidget()
                lay = QtWidgets.QHBoxLayout(w)
                lay.setContentsMargins(4, 2, 4, 2)
                self._chk = QtWidgets.QCheckBox('Black outside mask')
                self._chk.setChecked(True)
                self._chk.setStyleSheet('color:#ccc; font-size:9px;')
                lbl = QtWidgets.QLabel('Padding:')
                lbl.setStyleSheet('color:#ccc; font-size:9px;')
                self._spin = QtWidgets.QSpinBox()
                self._spin.setRange(0, 500)
                self._spin.setValue(0)
                self._spin.setSuffix(' px')
                self._spin.setFixedWidth(70)
                lay.addWidget(self._chk)
                lay.addStretch()
                lay.addWidget(lbl)
                lay.addWidget(self._spin)
                self.set_custom_widget(w)
            def get_value(self): return None
            def set_value(self, v): pass

        self._mc_w = _MaskCropWidget(self.view)
        self._mc_w._chk.toggled.connect(lambda v: self.set_property('black_outside', v))
        self._mc_w._spin.valueChanged.connect(lambda v: self.set_property('padding', v))
        self.add_custom_widget(self._mc_w)
        self.create_preview_widgets()

    def evaluate(self):
        self.reset_progress()

        img_data = self._get_input_image_data()
        if img_data is None:
            return False, "No image input"
        arr_in = img_data.payload

        mask_port = self.inputs().get('mask')
        if not mask_port or not mask_port.connected_ports():
            self._make_image_output(arr_in)
            self.set_display(arr_in)
            self.set_progress(100)
            return True, None

        cp = mask_port.connected_ports()[0]
        mdata = cp.node().output_values.get(cp.name())
        if not isinstance(mdata, MaskData) or mdata.payload is None:
            return False, "Mask input required"

        mask_arr = mdata.payload
        binary = mask_arr > 0

        if not binary.any():
            return False, "Mask is empty"

        self.set_progress(30)

        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        top, bottom = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1]) + 1
        left, right = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1]) + 1

        pad = int(self.get_property('padding') or 0)
        H, W = arr_in.shape[:2]
        top    = max(0, top - pad)
        left   = max(0, left - pad)
        bottom = min(H, bottom + pad)
        right  = min(W, right + pad)

        self.set_progress(60)

        cropped = arr_in[top:bottom, left:right].copy()

        if self.get_property('black_outside'):
            mask_crop = binary[top:bottom, left:right]
            cropped[~mask_crop] = 0

        self.set_progress(90)
        self._make_image_output(cropped)
        self.set_display(cropped)
        self.set_progress(100)
        return True, None


class NodeMultiShapeWidget(NodeBaseWidget):
    """
    Embeds a multi-shape drawing canvas over an image on the node surface.

    Each shape has its own colour, width, and style. Emits `shapes_changed(list)` whenever the shape collection changes.
    """

    shapes_changed       = Signal(list)          # full shapes_data list
    _img_signal          = Signal(object)        # cross-thread image load
    _shapes_signal       = Signal(object)        # cross-thread shapes load
    _contour_signal      = Signal(object, object)  # cross-thread mask contour update
    _mask_count_signal   = Signal(int)           # cross-thread mask count update

    _VIEW_MAX_W = 1200;  _VIEW_MAX_H = 900
    _VIEW_MIN_W = 400;   _VIEW_MIN_H = 300
    _VIEW_DEF_W = 900;   _VIEW_DEF_H = 700

    def __init__(self, parent=None):
        super().__init__(parent, name='_draw_shape_view', label='')

        container = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(container)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(3)

        # ── toolbar ────────────────────────────────────────────────────────
        tb = QtWidgets.QHBoxLayout()
        self._shape_group = QtWidgets.QButtonGroup(container)
        for i, (label, key) in enumerate([
                ('Ellipse', 'ellipse'),
                ('Rect', 'rectangle'),
                ('Polygon', 'polygon'),
                ('Arrow', 'arrow'),
                ('Curve', 'curve'),
                ('Text', 'text'),
        ]):
            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True)
            btn.setProperty('shape_key', key)
            btn.setFixedHeight(22)
            if i == 0:
                btn.setChecked(True)
            self._shape_group.addButton(btn, i)
            tb.addWidget(btn)

        tb.addStretch()
        for icon, slot, tip in (('+', '_zoom_in', 'Zoom in'),
                                ('-', '_zoom_out', 'Zoom out'),
                                ('\u2299', '_zoom_reset', 'Fit to view')):
            b = QtWidgets.QPushButton(icon)
            b.setFixedSize(34, 24)
            b.setProperty('compact', True)
            b.setToolTip(tip)
            b.clicked.connect(getattr(self, slot))
            tb.addWidget(b)
        del_btn = QtWidgets.QPushButton('Del')
        del_btn.setFixedHeight(22)
        del_btn.setToolTip('Delete selected shape (Delete / Backspace)')
        del_btn.clicked.connect(self._on_delete_selected)
        tb.addWidget(del_btn)
        clear_btn = QtWidgets.QPushButton('Clear All')
        clear_btn.setFixedHeight(22)
        tb.addWidget(clear_btn)
        root.addLayout(tb)

        # ── drawing canvas ─────────────────────────────────────────────────
        self._scene = QGraphicsScene()
        self._scene.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self._view = MultiShapeGraphicsView(self._scene)
        self._view.setFixedSize(self._VIEW_DEF_W, self._VIEW_DEF_H)
        root.addWidget(self._view)

        # ── status bar ─────────────────────────────────────────────────────
        self._status_label = QtWidgets.QLabel('Ready')
        self._status_label.setStyleSheet(
            'color:#ccc; font-size:10px; font-family:monospace; padding:2px;')
        root.addWidget(self._status_label)

        # ── shape list ─────────────────────────────────────────────────────
        self._shape_list = QtWidgets.QListWidget()
        self._shape_list.setMaximumHeight(150)
        self._shape_list.setMinimumHeight(40)
        self._shape_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._shape_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        root.addWidget(self._shape_list)

        # ── per-shape style row ────────────────────────────────────────────
        style_row = QtWidgets.QHBoxLayout()
        style_row.addWidget(QtWidgets.QLabel('W:'))
        self._width_spin = QtWidgets.QDoubleSpinBox()
        self._width_spin.setRange(0.5, 40.0)
        self._width_spin.setSingleStep(0.5)
        self._width_spin.setDecimals(1)
        self._width_spin.setValue(2.0)
        self._width_spin.setFixedWidth(68)
        style_row.addWidget(self._width_spin)

        self._style_combo = QtWidgets.QComboBox()
        self._style_combo.addItems(['solid', 'dashed', 'dotted', 'dashdot'])
        self._style_combo.setFixedWidth(88)
        style_row.addWidget(self._style_combo)

        self._color_btn = QtWidgets.QPushButton()
        self._color_btn.setFixedSize(22, 22)
        self._color_btn.setToolTip('Shape color')
        style_row.addWidget(self._color_btn)

        style_row.addSpacing(6)

        self._fill_btn = QtWidgets.QPushButton('Fill')
        self._fill_btn.setCheckable(True)
        self._fill_btn.setFixedHeight(22)
        self._fill_btn.setToolTip('Fill shape with semi-transparent color')
        self._fill_btn.toggled.connect(self._on_style_control_changed)
        style_row.addWidget(self._fill_btn)

        self._fill_opacity_spin = QtWidgets.QSpinBox()
        self._fill_opacity_spin.setRange(0, 255)
        self._fill_opacity_spin.setValue(180)
        self._fill_opacity_spin.setSuffix('α')
        self._fill_opacity_spin.setFixedWidth(60)
        self._fill_opacity_spin.setFixedHeight(22)
        self._fill_opacity_spin.setToolTip('Fill opacity (0 = transparent, 255 = solid)')
        self._fill_opacity_spin.valueChanged.connect(self._on_style_control_changed)
        style_row.addWidget(self._fill_opacity_spin)

        self._fill_all_btn = QtWidgets.QPushButton('Fill All')
        self._fill_all_btn.setFixedHeight(22)
        self._fill_all_btn.setToolTip('Set fill mode on all mask inputs at once')
        self._fill_all_btn.clicked.connect(self._on_fill_all)
        style_row.addWidget(self._fill_all_btn)

        style_row.addSpacing(10)

        self._auto_fill_cb = QtWidgets.QCheckBox('Auto Fill')
        self._auto_fill_cb.setToolTip('Automatically fill all incoming mask inputs')
        self._auto_fill_cb.setChecked(False)
        style_row.addWidget(self._auto_fill_cb)

        style_row.addSpacing(6)

        self._no_preview_cb = QtWidgets.QCheckBox('No Preview')
        self._no_preview_cb.setToolTip('Skip loading image into the canvas (faster)')
        self._no_preview_cb.setChecked(False)
        style_row.addWidget(self._no_preview_cb)

        style_row.addSpacing(10)

        style_row.addWidget(QtWidgets.QLabel('Lbl Opacity:'))
        self._label_opacity_spin = QtWidgets.QSpinBox()
        self._label_opacity_spin.setRange(0, 100)
        self._label_opacity_spin.setValue(70)
        self._label_opacity_spin.setSuffix('%')
        self._label_opacity_spin.setFixedWidth(62)
        self._label_opacity_spin.setToolTip('Opacity of the label image overlay')
        style_row.addWidget(self._label_opacity_spin)

        style_row.addSpacing(10)

        # Hidden label edit — kept for backward compat with saved workflows
        self._label_edit = QtWidgets.QLineEdit()
        self._label_edit.hide()
        self._label_edit.setFixedWidth(90)
        style_row.addWidget(self._label_edit)
        style_row.addStretch()
        root.addLayout(style_row)

        # ── geometry row (position / size / font-size) ────────────────────
        geo_row = QtWidgets.QHBoxLayout()
        for lbl, attr, lo, hi, dec, step, w in (
            ('X:', '_geo_x_spin', -9999, 9999, 1, 1.0, 72),
            ('Y:', '_geo_y_spin', -9999, 9999, 1, 1.0, 72),
            ('W:', '_geo_w_spin', 1, 9999, 1, 1.0, 72),
            ('H:', '_geo_h_spin', 1, 9999, 1, 1.0, 72),
            ('Font:', '_geo_sz_spin', 4, 120, 1, 1.0, 58),
        ):
            geo_row.addWidget(QtWidgets.QLabel(lbl))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(dec)
            spin.setSingleStep(step)
            spin.setFixedWidth(w)
            setattr(self, attr, spin)
            geo_row.addWidget(spin)
        geo_row.addStretch()
        root.addLayout(geo_row)

        tip = QtWidgets.QLabel(
            'Shift = square/circle \u00b7 Del/\u232b = delete shape \u00b7 '
            'Enter = close polygon \u00b7 Dbl-click = edit text')
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tip.setStyleSheet('color:#999; font-size:9px; padding:1px;')
        tip.setWordWrap(True)
        root.addWidget(tip)

        self._container = container
        self.set_custom_widget(container)

        # ── internal state ─────────────────────────────────────────────────
        self._line_color = QColor(0, 220, 220)
        self._updating_controls = False
        self._mask_contour_styles: dict[str, dict] = {}
        self._mask_count = 0
        self._update_color_btn()
        self._active_shape_id = None  # last selected shape, survives focus changes

        # ── connections ────────────────────────────────────────────────────
        self._shape_group.idClicked.connect(self._on_shape_btn)
        self._view.shape_committed.connect(self._on_shape_committed)
        self._view.shape_selected.connect(self._on_canvas_selection)
        self._view.shapes_changed.connect(self._rebuild_shape_list)
        self._view.mouse_moved.connect(self._on_mouse_moved)
        self._shape_list.currentItemChanged.connect(self._on_list_selection)
        self._shape_list.customContextMenuRequested.connect(
            self._on_list_context_menu)
        self._width_spin.valueChanged.connect(self._on_style_control_changed)
        self._style_combo.currentTextChanged.connect(
            self._on_style_control_changed)
        self._color_btn.clicked.connect(self._on_pick_color)
        self._label_edit.textChanged.connect(self._on_style_control_changed)
        self._geo_x_spin.valueChanged.connect(self._on_geometry_changed)
        self._geo_y_spin.valueChanged.connect(self._on_geometry_changed)
        self._geo_w_spin.valueChanged.connect(self._on_geometry_changed)
        self._geo_h_spin.valueChanged.connect(self._on_geometry_changed)
        self._geo_sz_spin.valueChanged.connect(self._on_style_control_changed)
        clear_btn.clicked.connect(self._on_clear_all)

        self._img_signal.connect(
            self._apply_image, Qt.ConnectionType.QueuedConnection)
        self._shapes_signal.connect(
            self._apply_shapes_data, Qt.ConnectionType.QueuedConnection)
        self._contour_signal.connect(
            self._apply_mask_contours, Qt.ConnectionType.QueuedConnection)
        self._mask_count_signal.connect(
            self._apply_mask_count, Qt.ConnectionType.QueuedConnection)

    # ── colour swatch ──────────────────────────────────────────────────────

    def _update_color_btn(self):
        c = self._line_color
        self._color_btn.setStyleSheet(
            f'background-color: rgb({c.red()},{c.green()},{c.blue()});'
            f'border: 1px solid #555; border-radius: 3px;')

    # ── shape toolbar ──────────────────────────────────────────────────────

    def _on_shape_btn(self, btn_id):
        btn = self._shape_group.button(btn_id)
        key = btn.property('shape_key') if btn else 'ellipse'
        self._view.set_shape_type(key)

    # ── canvas → widget ────────────────────────────────────────────────────

    def _on_shape_committed(self, shape_id: str):
        self._rebuild_shape_list()
        # Refresh geometry spinboxes if this shape is selected
        if shape_id == self._view.selected_id():
            self._updating_controls = True
            self._load_geometry_for_shape(shape_id)
            self._updating_controls = False
        self._emit_shapes()

    def _on_canvas_selection(self, shape_id: str):
        """Canvas click selected a shape — sync list + style controls."""
        if shape_id:
            self._active_shape_id = shape_id
        self._sync_list_to_canvas(shape_id)
        if shape_id:
            self._load_style_for_shape(shape_id)

    def _on_mouse_moved(self, x: float, y: float):
        parts = [f'Cursor({x:.0f}, {y:.0f})']
        sid = self._view.selected_id()
        if sid and sid in self._view._shape_items:
            item = self._view._shape_items[sid]
            if isinstance(item, (CustomEllipseItem, CustomRectangleItem)):
                r = item.rect()
                c = item.mapToScene(r.center())
                parts.append(f'{r.width():.1f}\u00d7{r.height():.1f}')
                parts.append(f'Center({c.x():.1f}, {c.y():.1f})')
            elif isinstance(item, CustomCurveItem):
                import math
                p1, p2 = item.endpoints()
                parts.append(f'Curve L={math.hypot(p2.x()-p1.x(), p2.y()-p1.y()):.1f}')
            elif isinstance(item, CustomArrowItem):
                parts.append(f'L={item.line().length():.1f}')
            elif isinstance(item, QGraphicsPolygonItem):
                br = item.boundingRect()
                parts.append(f'{br.width():.1f}\u00d7{br.height():.1f} bbox')
        self._status_label.setText('  \u2502  '.join(parts))

    # ── list ↔ canvas sync ─────────────────────────────────────────────────

    def _rebuild_shape_list(self):
        self._shape_list.blockSignals(True)
        old_sel = self._view.selected_id()
        # Also check if a mask contour was selected
        old_cur = self._shape_list.currentItem()
        old_list_sid = (old_cur.data(Qt.ItemDataRole.UserRole)
                        if old_cur else None)
        self._shape_list.clear()
        # Canvas shapes
        for sid, item in self._view._shape_items.items():
            st = self._view._shape_styles.get(sid, {})
            desc = self._shape_description(sid, item)
            li = QtWidgets.QListWidgetItem()
            li.setData(Qt.ItemDataRole.UserRole, sid)
            c = st.get('line_color', [0, 220, 220, 255])
            pix = QPixmap(12, 12)
            pix.fill(QColor(int(c[0]), int(c[1]), int(c[2])))
            li.setIcon(QtGui.QIcon(pix))
            li.setText(desc)
            self._shape_list.addItem(li)
            if sid == old_sel:
                li.setSelected(True)
                self._shape_list.setCurrentItem(li)
        # Mask contour virtual entries
        for i in range(self._mask_count):
            mid = f'mask_{i}'
            st = self._mask_contour_styles.get(mid, {})
            c = st.get('line_color', [0, 220, 220, 255])
            li = QtWidgets.QListWidgetItem()
            li.setData(Qt.ItemDataRole.UserRole, mid)
            pix = QPixmap(12, 12)
            pix.fill(QColor(int(c[0]), int(c[1]), int(c[2])))
            li.setIcon(QtGui.QIcon(pix))
            li.setText(f'Mask #{i}')
            self._shape_list.addItem(li)
            if mid == old_list_sid:
                li.setSelected(True)
                self._shape_list.setCurrentItem(li)
        self._shape_list.blockSignals(False)

    def _shape_description(self, sid: str, item) -> str:
        if isinstance(item, CustomEllipseItem):
            r = item.rect()
            return f'Ellipse  ({r.width():.0f}\u00d7{r.height():.0f})'
        if isinstance(item, CustomRectangleItem):
            r = item.rect()
            return f'Rect  ({r.width():.0f}\u00d7{r.height():.0f})'
        if isinstance(item, QGraphicsPolygonItem):
            n = len(item.polygon())
            return f'Polygon  ({n} pts)'
        if isinstance(item, CustomCurveItem):
            import math
            p1, p2 = item.endpoints()
            chord = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
            return f'Curve  (L={chord:.0f})'
        if isinstance(item, CustomArrowItem):
            return f'Arrow  (L={item.line().length():.0f})'
        if isinstance(item, QGraphicsSimpleTextItem):
            txt = item.text()[:20]
            return f'Text  "{txt}"'
        return sid

    def _sync_list_to_canvas(self, shape_id: str):
        self._shape_list.blockSignals(True)
        for i in range(self._shape_list.count()):
            li = self._shape_list.item(i)
            if li.data(Qt.ItemDataRole.UserRole) == shape_id:
                self._shape_list.setCurrentItem(li)
                break
        else:
            self._shape_list.clearSelection()
        self._shape_list.blockSignals(False)

    def _on_list_selection(self, current, _previous):
        if current is not None:
            sid = current.data(Qt.ItemDataRole.UserRole)
            self._active_shape_id = sid
            if self._is_mask_contour(sid):
                # Mask contour: deselect canvas, show bbox, load mask style
                self._view.select_shape(None)
                self._view.show_mask_bbox(sid)
                self._load_style_for_shape(sid)
            else:
                self._view.select_shape(sid)
                self._view.show_mask_bbox(None)  # hide all mask bboxes
                self._load_style_for_shape(sid)
        else:
            self._view.show_mask_bbox(None)

    def _on_list_context_menu(self, pos):
        item = self._shape_list.itemAt(pos)
        if item is None:
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        if self._is_mask_contour(sid):
            return  # mask contours can't be deleted
        menu = QtWidgets.QMenu(self._shape_list)
        act_del = menu.addAction('Delete shape')
        chosen = menu.exec(self._shape_list.mapToGlobal(pos))
        if chosen == act_del:
            self._view.delete_shape(sid)
            self._rebuild_shape_list()
            self._emit_shapes()

    # ── per-shape style ────────────────────────────────────────────────────

    def _is_mask_contour(self, shape_id: str) -> bool:
        return shape_id is not None and shape_id.startswith('mask_')

    def _load_style_for_shape(self, shape_id: str):
        # Mask contour entries: style from _mask_contour_styles
        if self._is_mask_contour(shape_id):
            st = self._mask_contour_styles.get(shape_id, {})
        else:
            st = self._view.get_shape_style(shape_id)
        if not st:
            return
        self._updating_controls = True
        self._width_spin.setValue(float(st.get('line_width', 2.0)))
        self._style_combo.setCurrentText(str(st.get('line_style', 'solid')))
        c = st.get('line_color', [0, 220, 220, 255])
        self._line_color = QColor(int(c[0]), int(c[1]), int(c[2]),
                                   int(c[3]) if len(c) > 3 else 255)
        self._update_color_btn()
        self._label_edit.setText(str(st.get('label_text', '')))
        self._fill_btn.setChecked(bool(st.get('fill_mode', False)))
        self._fill_opacity_spin.setValue(int(st.get('fill_opacity', 180)))
        self._load_geometry_for_shape(shape_id)
        self._updating_controls = False

    def _load_geometry_for_shape(self, shape_id: str):
        """Populate X/Y/W/H/Sz spinboxes from shape geometry."""
        if self._is_mask_contour(shape_id):
            # Mask contours: all geometry controls disabled
            for spin in (self._geo_x_spin, self._geo_y_spin,
                         self._geo_w_spin, self._geo_h_spin,
                         self._geo_sz_spin):
                spin.setEnabled(False)
                spin.setValue(0)
            return
        geo = self._view.get_shape_geometry(shape_id)
        if not geo:
            return
        item = self._view._shape_items.get(shape_id)
        is_text = isinstance(item, QGraphicsSimpleTextItem)
        is_arrow = isinstance(item, CustomArrowItem)
        is_curve = isinstance(item, CustomCurveItem)
        is_poly = isinstance(item, QGraphicsPolygonItem) and not isinstance(
            item, (CustomEllipseItem, CustomRectangleItem))

        self._geo_x_spin.setEnabled(True)
        self._geo_y_spin.setEnabled(True)
        self._geo_w_spin.setEnabled(not is_text)
        self._geo_h_spin.setEnabled(not is_text and not is_arrow and not is_curve)
        # Sz: always enabled — controls label font for arrows, text font for
        # text shapes, label font for all other shapes
        self._geo_sz_spin.setEnabled(True)

        self._geo_x_spin.setValue(geo.get('x', 0))
        self._geo_y_spin.setValue(geo.get('y', 0))
        self._geo_w_spin.setValue(geo.get('w', 0))
        self._geo_h_spin.setValue(geo.get('h', 0))
        # Load font size from style (applies to all shape types)
        st = self._view.get_shape_style(shape_id)
        self._geo_sz_spin.setValue(
            float(st.get('label_font_size', 12.0)) if st else 12.0)

    def _on_geometry_changed(self, *_args):
        if self._updating_controls:
            return
        sid = self._active_shape_id
        if not sid or self._is_mask_contour(sid):
            return
        x = self._geo_x_spin.value()
        y = self._geo_y_spin.value()
        w = self._geo_w_spin.value()
        h = self._geo_h_spin.value()
        self._view.set_shape_geometry(sid, x, y, w, h, notify=False)
        self._rebuild_shape_list()
        self._emit_shapes_no_reload()

    def _on_style_control_changed(self, *_args):
        if self._updating_controls:
            return
        sid = self._active_shape_id
        if sid and self._is_mask_contour(sid):
            # Update mask contour style
            self._mask_contour_styles[sid] = {
                'line_width': self._width_spin.value(),
                'line_style': self._style_combo.currentText(),
                'line_color': [self._line_color.red(),
                               self._line_color.green(),
                               self._line_color.blue(),
                               self._line_color.alpha()],
                'label_text': self._label_edit.text(),
                'fill_mode':  self._fill_btn.isChecked(),
                'fill_opacity': self._fill_opacity_spin.value(),
            }
            self._emit_shapes_no_reload()
        elif sid:
            # apply to selected canvas shape (immediate visual update)
            self._view.update_shape_style(sid, 'line_width',
                                          self._width_spin.value())
            self._view.update_shape_style(sid, 'line_style',
                                          self._style_combo.currentText())
            self._view.update_shape_style(sid, 'line_color', [
                self._line_color.red(), self._line_color.green(),
                self._line_color.blue(), self._line_color.alpha()])
            self._view.update_shape_style(sid, 'label_text',
                                          self._label_edit.text())
            self._view.update_shape_style(sid, 'label_font_size',
                                          self._geo_sz_spin.value())
            self._view.update_shape_style(sid, 'fill_mode',
                                          self._fill_btn.isChecked())
            self._view.update_shape_style(sid, 'fill_opacity',
                                          self._fill_opacity_spin.value())
            self._emit_shapes_no_reload()
        else:
            # no selection → update defaults for next new shape
            self._view.set_defaults(
                color=self._line_color,
                width=self._width_spin.value(),
                style=self._style_combo.currentText(),
                font_size=self._geo_sz_spin.value())

    def _on_fill_all(self):
        """Set fill_mode=True on every mask contour at once."""
        for i in range(self._mask_count):
            mid = f'mask_{i}'
            st = self._mask_contour_styles.setdefault(mid, {})
            st['fill_mode'] = True
        self._rebuild_shape_list()
        self._emit_shapes()
        # Refresh the fill button state if a mask is currently selected
        cur = self._shape_list.currentItem()
        if cur and self._is_mask_contour(cur.data(Qt.ItemDataRole.UserRole)):
            self._updating_controls = True
            self._fill_btn.setChecked(True)
            self._updating_controls = False

    def _on_pick_color(self):
        parent = QtWidgets.QApplication.activeWindow()
        color = QtWidgets.QColorDialog.getColor(
            self._line_color, parent, 'Shape Color')
        if color.isValid():
            self._line_color = color
            self._update_color_btn()
            self._on_style_control_changed()

    def _on_delete_selected(self):
        # Check list selection for mask contour first
        cur = self._shape_list.currentItem()
        if cur:
            list_sid = cur.data(Qt.ItemDataRole.UserRole)
            if self._is_mask_contour(list_sid):
                # Can't delete mask contours — they come from connections
                return
        sid = self._view.selected_id()
        if sid:
            self._view.delete_shape(sid)
            self._rebuild_shape_list()
            self._emit_shapes()

    def _on_clear_all(self):
        self._view.clear_all()
        self._rebuild_shape_list()
        self._emit_shapes()

    # ── mask contour management ────────────────────────────────────────────

    _MASK_PALETTE = [
        [0, 220, 220, 255],    # cyan
        [220, 0, 220, 255],    # magenta
        [220, 220, 0, 255],    # yellow
        [0, 220, 80, 255],     # lime-green
        [255, 140, 0, 255],    # orange
        [100, 140, 255, 255],  # light blue
        [255, 80, 80, 255],    # coral
        [180, 100, 255, 255],  # purple
    ]

    def set_mask_count(self, n: int):
        """Ensure exactly *n* mask contour entries exist with styles."""
        import threading
        if threading.current_thread() is not threading.main_thread():
            self._mask_count_signal.emit(n)
            return
        old = self._mask_count
        self._mask_count = n
        # Create styles for new masks, keep existing
        for i in range(n):
            mid = f'mask_{i}'
            if mid not in self._mask_contour_styles:
                pal = self._MASK_PALETTE[i % len(self._MASK_PALETTE)]
                self._mask_contour_styles[mid] = {
                    'line_width': 2.0,
                    'line_style': 'solid',
                    'line_color': list(pal),
                    'label_text': '',
                }
        # Remove excess mask entries
        for i in range(n, old):
            self._mask_contour_styles.pop(f'mask_{i}', None)
        if old != n:
            self._rebuild_shape_list()

    def get_mask_contour_styles(self) -> list[dict]:
        """Return mask contour style dicts in order."""
        result = []
        for i in range(self._mask_count):
            mid = f'mask_{i}'
            st = dict(self._mask_contour_styles.get(mid, {}))
            st['id'] = mid
            st['shape'] = 'mask_contour'
            result.append(st)
        return result

    def set_mask_contours(self, contour_data, contour_styles):
        """Pass mask contour paths + styles to the scene view (thread-safe)."""
        import threading
        styles = []
        for i in range(len(contour_data)):
            mid = f'mask_{i}'
            st = self._mask_contour_styles.get(mid, {})
            if i < len(contour_styles) and contour_styles[i]:
                st = contour_styles[i]
            styles.append(st)
        if threading.current_thread() is threading.main_thread():
            self._view.set_mask_contours(contour_data, styles)
        else:
            self._contour_signal.emit(contour_data, styles)

    def _apply_mask_contours(self, contour_data, styles):
        """Main-thread slot for set_mask_contours."""
        self._view.set_mask_contours(contour_data, styles)

    def _apply_mask_count(self, n: int):
        """Main-thread slot for set_mask_count (cross-thread routing)."""
        self.set_mask_count(n)

    # ── emit to node ───────────────────────────────────────────────────────

    def _debounce_fire(self):
        """Called after debounce timer expires.
        Persist shape data and mark node dirty — output updates on next
        Run Graph. Avoids running evaluate() which causes focus/selection loss."""
        data = self._view.get_all_shapes_data()
        data.extend(self.get_mask_contour_styles())
        # Directly update the node model without triggering shapes_changed signal
        # (which would call evaluate and steal focus)
        node = self.parent()
        while node and not hasattr(node, 'model'):
            node = node.parent() if hasattr(node, 'parent') else None
        if hasattr(self, '_owner_node') and self._owner_node is not None:
            self._owner_node.model.set_property('shapes_data', json.dumps(data))
            self._owner_node.mark_dirty()

    def _emit_shapes(self):
        data = self._view.get_all_shapes_data()
        data.extend(self.get_mask_contour_styles())
        self.shapes_changed.emit(data)

    def _emit_shapes_no_reload(self):
        """Emit shapes for output re-evaluation without reloading the widget.
        Used during spinbox adjustments to prevent focus loss."""
        data = self._view.get_all_shapes_data()
        data.extend(self.get_mask_contour_styles())
        self._no_reload = True
        self.shapes_changed.emit(data)
        self._no_reload = False

    # ── view sizing ────────────────────────────────────────────────────────

    def _compute_view_size(self, w: int, h: int) -> tuple[int, int]:
        if w <= 0 or h <= 0:
            return self._VIEW_DEF_W, self._VIEW_DEF_H
        scale = min(self._VIEW_MAX_W / float(w), self._VIEW_MAX_H / float(h))
        dw = max(self._VIEW_MIN_W, min(self._VIEW_MAX_W, int(round(w * scale))))
        dh = max(self._VIEW_MIN_H, min(self._VIEW_MAX_H, int(round(h * scale))))
        return dw, dh

    def _apply_view_size(self, w: int, h: int):
        dw, dh = self._compute_view_size(w, h)
        self._view.setFixedSize(dw, dh)
        self._container.adjustSize()
        hint = self._container.sizeHint()
        self._container.setFixedSize(hint)
        group = self.widget()
        if group:
            group.setMinimumSize(hint)
            group.resize(hint)
            group.adjustSize()
        self.resize(hint.width(), hint.height())
        self.updateGeometry()
        if self.node and self.node.view:
            self.node.view.draw_node()

    def _apply_image(self, image):
        w, h = _image_hw(image)
        self._apply_view_size(w, h)
        self._view.load_image(image)

    # ── public loaders (thread-safe) ───────────────────────────────────────

    def load_image(self, image):
        if threading.current_thread() is threading.main_thread():
            self._apply_image(image)
        else:
            self._img_signal.emit(image)

    def load_shapes_data(self, shapes: list):
        if threading.current_thread() is threading.main_thread():
            self._apply_shapes_data(shapes)
        else:
            self._shapes_signal.emit(shapes)

    def _apply_shapes_data(self, shapes):
        # Filter out mask_contour entries — those are virtual, not canvas items
        canvas_shapes = [s for s in shapes if s.get('shape') != 'mask_contour']
        # Restore mask contour styles if present
        for s in shapes:
            if s.get('shape') == 'mask_contour' and 'id' in s:
                self._mask_contour_styles[s['id']] = {
                    'line_width': s.get('line_width', 2.0),
                    'line_style': s.get('line_style', 'solid'),
                    'line_color': s.get('line_color', [0, 220, 220, 255]),
                    'label_text': s.get('label_text', ''),
                }
        self._view.load_all_shapes_data(canvas_shapes)
        self._rebuild_shape_list()

    def _zoom_in(self):    self._view.zoom_in()
    def _zoom_out(self):   self._view.zoom_out()
    def _zoom_reset(self): self._view.zoom_reset()

    def get_value(self):         return ''
    def set_value(self, _value): pass


# ══════════════════════════════════════════════════════════════════════════════
#  Text overlay helper (PIL-based, for evaluate output)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_text_overlay(image, shape_data: dict):
    """Draw a standalone text annotation. Accepts numpy or PIL, returns PIL."""
    from PIL import Image as _PIL, ImageDraw, ImageFont
    text = shape_data.get('text_content', '')
    if not text.strip():
        return image if not isinstance(image, np.ndarray) else _PIL.fromarray(_ensure_display_rgb(image), 'RGB')
    if isinstance(image, np.ndarray):
        pil_image = _PIL.fromarray(_ensure_display_rgb(image), 'RGB')
    else:
        pil_image = image
    out = pil_image.copy() if pil_image.mode == 'RGB' else pil_image.convert('RGB')
    draw = ImageDraw.Draw(out)
    pos = shape_data.get('text_pos', [0, 0])
    font_size = max(4.0, float(shape_data.get('label_font_size', 12.0)))
    c = shape_data.get('line_color', [255, 255, 255, 255])
    color = (int(c[0]), int(c[1]), int(c[2]))
    try:
        font = ImageFont.load_default(size=round(font_size))
    except TypeError:
        font = None
    x, y = float(pos[0]), float(pos[1])
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=color, font=font)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Cubic Bézier curve overlay helper (PIL-based, for evaluate output)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_curve_overlay(image, shape_data: dict, *,
                        line_color=(0, 220, 220), line_width=2.0,
                        line_style='solid',
                        label_text='', label_x_offset=8.0,
                        label_y_offset=-8.0, label_font_size=12.0):
    """Draw a cubic Bezier curve with arrowhead. Accepts numpy or PIL, returns PIL."""
    from PIL import Image as _PIL, ImageDraw
    pts = shape_data.get('points', [])
    cps = shape_data.get('control_points', [])
    if isinstance(image, np.ndarray):
        pil_image = _PIL.fromarray(_ensure_display_rgb(image), 'RGB')
    else:
        pil_image = image
    if len(pts) != 2 or len(cps) != 2:
        return pil_image
    out = pil_image.copy() if pil_image.mode == 'RGB' else pil_image.convert('RGB')
    draw = ImageDraw.Draw(out)
    p1, p2 = pts
    cp1, cp2 = cps
    # Sample cubic bezier at N points
    N = 64
    curve_pts = []
    for i in range(N + 1):
        t = i / N
        mt = 1 - t
        x = (mt**3 * p1[0] + 3 * mt**2 * t * cp1[0]
             + 3 * mt * t**2 * cp2[0] + t**3 * p2[0])
        y = (mt**3 * p1[1] + 3 * mt**2 * t * cp1[1]
             + 3 * mt * t**2 * cp2[1] + t**3 * p2[1])
        curve_pts.append((x, y))
    _draw_styled_polyline(draw, curve_pts,
                          color=line_color,
                          width=max(1, int(line_width)),
                          style=str(line_style or 'solid').lower(),
                          closed=False)
    # Arrowhead at p2 using tangent (p2 - cp2)
    dx = float(p2[0] - cp2[0])
    dy = float(p2[1] - cp2[1])
    L = float(np.hypot(dx, dy))
    head_len = max(8.0, 4.0 * float(line_width))
    if L > 1e-6:
        ux, uy = dx / L, dy / L
        ang = np.deg2rad(28.0)
        c, s = np.cos(ang), np.sin(ang)
        lx = p2[0] - (ux * c - uy * s) * head_len
        ly = p2[1] - (ux * s + uy * c) * head_len
        rx = p2[0] - (ux * c + uy * s) * head_len
        ry = p2[1] - (-ux * s + uy * c) * head_len
        draw.polygon([(p2[0], p2[1]), (lx, ly), (rx, ry)], fill=line_color)
    # Label
    txt = str(label_text or '').strip()
    if txt:
        from PIL import ImageFont
        try:
            font = ImageFont.load_default(size=round(max(4.0, label_font_size)))
        except TypeError:
            font = None
        tx, ty = float(p2[0]) + label_x_offset, float(p2[1]) + label_y_offset
        draw.text((tx + 1, ty + 1), txt, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), txt, fill=line_color, font=font)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  DrawShapeNode (rewritten for multi-shape)
# ══════════════════════════════════════════════════════════════════════════════

class DrawShapeNode(BaseExecutionNode):
    """
    Draw shapes, text, and annotations on an image.

    Shapes: rectangle, ellipse, polygon, arrow, bezier curve, and free text.
    Each shape has its own color, line width, line style (solid/dashed/dotted),
    and optional fill with adjustable opacity. Shapes can be moved, resized,
    and edited interactively on the canvas.

    Inputs:
    - **image** — background image (optional)
    - **mask** (multi-input) — binary masks shown as colored contours
    - **label_image** — segmentation labels shown as colored overlay

    Controls:
    - Line width, style, and color per shape
    - Fill toggle + opacity for closed shapes (rectangle, ellipse, polygon)
    - Auto Fill — fill all mask contours at once
    - Fill All — apply fill to every mask input
    - Label overlay opacity — transparency of segmentation label colors
    - Geometry spinboxes (X, Y, W, H) for precise positioning
    - Font size for texts
    - No Preview — skip interactive canvas rendering for speed

    Hold Shift while drawing to constrain to square/circle.

    Keywords: annotate, draw, shape, rectangle, ellipse, polygon, arrow, text,
    curve, mask outline, fill, overlay, 標註, 繪圖, 形狀
    """
    __identifier__ = 'nodes.image_process.Visualize'
    NODE_NAME      = 'Draw Shape'
    PORT_SPEC      = {'inputs': ['image', 'mask', 'label_image'],
                       'outputs': ['image']}

    _UI_PROPS = frozenset({
        'color', 'pos', 'selected', 'name', 'progress', 'image_view',
        'shapes_data', 'auto_fill', 'no_preview',
    })

    def __init__(self):
        super().__init__()
        self.add_input('image',  color=PORT_COLORS['image'])
        self.add_input('mask', multi_input=True, color=PORT_COLORS['mask'])
        self.add_input('label_image', color=PORT_COLORS['label_image'])
        self.add_output('image', color=PORT_COLORS['image'], multi_output=True)

        self.create_property('shapes_data', '[]')
        self.create_property('auto_fill', False)
        self.create_property('no_preview', False)

        self._draw_widget = NodeMultiShapeWidget(self.view)
        self._draw_widget._owner_node = self  # back-reference for debounce
        self._draw_widget.shapes_changed.connect(self._on_shapes_changed)
        self.add_custom_widget(self._draw_widget)
        self._last_img_id: int | None = None
        self._label_opacity_value: int = 70   # shadow: safe to read from any thread
        self._auto_fill: bool = False         # shadow
        self._no_preview: bool = False        # shadow
        self._draw_widget._label_opacity_spin.valueChanged.connect(
            lambda v: setattr(self, '_label_opacity_value', v))
        self._draw_widget._auto_fill_cb.toggled.connect(self._on_auto_fill_changed)
        self._draw_widget._no_preview_cb.toggled.connect(self._on_no_preview_changed)

    def _on_auto_fill_changed(self, v: bool):
        self._auto_fill = v
        self.model.set_property('auto_fill', v)
        self.mark_dirty()

    def _on_no_preview_changed(self, v: bool):
        self._no_preview = v
        self.model.set_property('no_preview', v)

    def set_property(self, name, value, push_undo=True):
        super().set_property(name, value, push_undo)
        if name == 'auto_fill':
            self._auto_fill = bool(value)
            self._draw_widget._auto_fill_cb.blockSignals(True)
            self._draw_widget._auto_fill_cb.setChecked(bool(value))
            self._draw_widget._auto_fill_cb.blockSignals(False)
        elif name == 'no_preview':
            self._no_preview = bool(value)
            self._draw_widget._no_preview_cb.blockSignals(True)
            self._draw_widget._no_preview_cb.setChecked(bool(value))
            self._draw_widget._no_preview_cb.blockSignals(False)

    def _on_shapes_changed(self, shapes: list):
        # Update model directly to persist shapes_data for save/load,
        # but avoid NodeGraphQt set_property → draw_node() which forces
        # a group-box stylesheet reset + full re-layout that can disrupt
        # the embedded QGraphicsView's viewport transform.
        self.model.set_property('shapes_data', json.dumps(shapes))
        # Save focused widget before evaluate (draw_node steals focus)
        focused = QtWidgets.QApplication.focusWidget()
        success, _ = self.evaluate()
        if success:
            self.mark_clean()
        else:
            self.mark_error()
        # Cascade dirty state to all downstream nodes so they re-evaluate
        for out_port in self.outputs().values():
            for in_port in out_port.connected_ports():
                dn = in_port.node()
                if hasattr(dn, 'mark_dirty'):
                    dn.mark_dirty()
        # Restore focus after all UI updates are done
        if focused is not None and focused.isVisible():
            QtCore.QTimer.singleShot(0, focused.setFocus)

    def evaluate(self):
        self.reset_progress()

        # ── sync settings from model (handles JSON load where node.set_property
        #    is bypassed — NodeGraphQt calls model.set_property directly) ──
        af = bool(self.get_property('auto_fill'))
        np_ = bool(self.get_property('no_preview'))
        if af != self._auto_fill:
            self._auto_fill = af
            self._draw_widget._auto_fill_cb.blockSignals(True)
            self._draw_widget._auto_fill_cb.setChecked(af)
            self._draw_widget._auto_fill_cb.blockSignals(False)
        if np_ != self._no_preview:
            self._no_preview = np_
            self._draw_widget._no_preview_cb.blockSignals(True)
            self._draw_widget._no_preview_cb.setChecked(np_)
            self._draw_widget._no_preview_cb.blockSignals(False)

        # ── image input (optional if mask connected) ───────────────────────
        arr_in = None
        img_port = self.inputs().get('image')
        if img_port and img_port.connected_ports():
            cp_img = img_port.connected_ports()[0]
            data = cp_img.node().output_values.get(cp_img.name())
            if not isinstance(data, ImageData):
                return False, "Image input must be ImageData."
            arr_in = data.payload       # numpy array

        # ── mask inputs (multi-input, optional) ───────────────────────────
        mask_arrays = []
        mask_port = self.inputs().get('mask')
        if mask_port:
            for cp in mask_port.connected_ports():
                mdata = cp.node().output_values.get(cp.name())
                if isinstance(mdata, MaskData):
                    m = mdata.payload
                    mask_bool = m.astype(bool) if m.dtype != bool else m
                    mask_arrays.append(mask_bool)

        # ── label_image input (watershed / segmentation labels) ──────────
        _label_overlay_pil = None
        label_port = self.inputs().get('label_image')
        if label_port and label_port.connected_ports():
            cp_lbl = label_port.connected_ports()[0]
            ldata = cp_lbl.node().output_values.get(cp_lbl.name())
            if isinstance(ldata, LabelData):
                if ldata.image is not None:
                    _label_overlay_pil = ldata.image    # PIL Image
                label_arr = np.asarray(ldata.payload)
                for lbl in np.unique(label_arr):
                    if lbl == 0:
                        continue
                    mask_arrays.append(label_arr == lbl)

        if arr_in is None and not mask_arrays:
            return False, "Connect an image, a mask, or both"

        if arr_in is None and mask_arrays:
            h, w = mask_arrays[0].shape
            arr_in = np.zeros((h, w, 3), dtype=np.uint8)

        self.set_progress(20)

        # ── load shapes ────────────────────────────────────────────────────
        raw = str(self.get_property('shapes_data') or '[]')
        try:
            shapes = json.loads(raw)
            if not isinstance(shapes, list):
                shapes = []
        except Exception:
            shapes = []

        # Separate mask_contour style dicts from drawable shapes
        draw_shapes = [s for s in shapes if s.get('shape') != 'mask_contour']
        mask_styles = {s['id']: s for s in shapes
                       if s.get('shape') == 'mask_contour'}

        self.set_progress(40)

        # Work on the original float32 image; draw annotations on a
        # separate RGBA overlay in PIL, then composite only drawn pixels
        # back so untouched pixels keep full bit-depth precision.
        from PIL import Image as _PIL, ImageDraw
        h_img, w_img = arr_in.shape[:2]
        result = arr_in.copy()  # float32 [0,1] — preserved

        # Build a uint8 RGB canvas for PIL drawing (display quality only)
        canvas_rgb = _ensure_display_rgb(arr_in)
        canvas = _PIL.fromarray(canvas_rgb, 'RGB')

        # ── label_image overlay (alpha-composite on float) ────────────────
        opacity = self._label_opacity_value
        if _label_overlay_pil is not None:
            alpha_f = opacity / 100.0
            if isinstance(_label_overlay_pil, np.ndarray):
                lbl_arr_raw = _label_overlay_pil
                if lbl_arr_raw.dtype in (np.float32, np.float64):
                    lbl_f = np.clip(lbl_arr_raw, 0, 1)
                else:
                    lbl_f = lbl_arr_raw.astype(np.float32) / 255.0
                if lbl_f.ndim == 2:
                    lbl_f = np.stack([lbl_f]*3, axis=-1)
                else:
                    lbl_f = lbl_f[:, :, :3]
            else:
                lbl_rgb = _label_overlay_pil.convert('RGB')
                if lbl_rgb.size != (w_img, h_img):
                    lbl_rgb = lbl_rgb.resize((w_img, h_img), _PIL.NEAREST)
                lbl_f = np.array(lbl_rgb).astype(np.float32) / 255.0
            # Blend only where label != background (non-black)
            fg_mask = lbl_f.any(axis=2)
            if result.ndim == 2:
                result = np.stack([result]*3, axis=-1)
            result[fg_mask] = (1 - alpha_f) * result[fg_mask] + alpha_f * lbl_f[fg_mask]
            # Also update the PIL canvas for drawing overlays on top
            canvas_rgb = _ensure_display_rgb(result)
            canvas = _PIL.fromarray(canvas_rgb, 'RGB')

        # ── update widget (skip when No Preview or when adjusting controls) ─
        no_preview = self._no_preview
        no_reload = getattr(self._draw_widget, '_no_reload', False)
        if not no_preview and not no_reload:
            self._draw_widget.set_mask_count(len(mask_arrays))
            if _label_overlay_pil is not None and (img_port is None or not img_port.connected_ports()):
                if isinstance(_label_overlay_pil, np.ndarray):
                    disp = _label_overlay_pil
                    if disp.dtype in (np.float32, np.float64):
                        disp = np.clip(disp * 255, 0, 255).astype(np.uint8)
                    if disp.ndim == 2:
                        disp = np.stack([disp]*3, axis=-1)
                    self._draw_widget.load_image(disp[:, :, :3])
                else:
                    display = _label_overlay_pil.convert('RGB')
                    self._draw_widget.load_image(np.array(display))
            else:
                self._draw_widget.load_image(arr_in)
            if draw_shapes:
                self._draw_widget.load_shapes_data(draw_shapes)

        # ── draw mask outlines on a transparent RGBA overlay ──────────────
        auto_fill = self._auto_fill
        contour_data_for_scene: list[list[list[tuple[float, float]]]] = []
        contour_styles_for_scene: list[dict] = []
        overlay = _PIL.new('RGBA', (w_img, h_img), (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        if mask_arrays:
            palette = NodeMultiShapeWidget._MASK_PALETTE
            for i, mask_bool in enumerate(mask_arrays):
                mid = f'mask_{i}'
                ms = mask_styles.get(mid, {})
                c = ms.get('line_color', palette[i % len(palette)])
                mask_color = (int(c[0]), int(c[1]), int(c[2]))
                mask_width = max(1, int(ms.get('line_width', 2)))
                mask_style = ms.get('line_style', 'solid')
                fill_mode  = auto_fill or bool(ms.get('fill_mode', False))
                paths = _mask_to_outline_paths(mask_bool) if mask_bool.any() else []
                contour_data_for_scene.append(paths)
                scene_style = ms if not auto_fill else {**ms, 'fill_mode': True}
                contour_styles_for_scene.append(scene_style)
                if fill_mode and mask_bool.any():
                    c_full = ms.get('line_color', [0, 220, 220, 255])
                    alpha_val = int(c_full[3]) if len(c_full) > 3 else 180
                    fill_color = (mask_color[0], mask_color[1], mask_color[2], alpha_val)
                    fill_arr = np.zeros((h_img, w_img, 4), dtype=np.uint8)
                    fill_arr[mask_bool] = fill_color
                    fill_ov = _PIL.fromarray(fill_arr, 'RGBA')
                    overlay = _PIL.alpha_composite(overlay, fill_ov)
                    ov_draw = ImageDraw.Draw(overlay)
                else:
                    for pts in paths:
                        _draw_styled_polyline(
                            ov_draw, pts,
                            color=mask_color, width=mask_width,
                            style=mask_style, closed=False)
        if not no_preview:
            self._draw_widget.set_mask_contours(contour_data_for_scene,
                                                contour_styles_for_scene)

        self.set_progress(60)

        # ── draw all shapes on the same RGBA overlay ──────────────────────
        # Shapes use helper functions that expect a PIL RGB image input,
        # so we composite onto the canvas, then diff to find drawn pixels.
        canvas_before = canvas.copy()
        for sd in draw_shapes:
            c = sd.get('line_color', [0, 220, 220, 255])
            color = (int(c[0]), int(c[1]), int(c[2]))
            lw = float(sd.get('line_width', 2.0))
            ls = str(sd.get('line_style', 'solid'))

            if sd.get('shape') == 'text':
                canvas = _draw_text_overlay(canvas, sd)
            elif sd.get('shape') == 'curve':
                canvas = _draw_curve_overlay(
                    canvas, sd,
                    line_color=color, line_width=lw, line_style=ls,
                    label_text=str(sd.get('label_text', '')),
                    label_x_offset=float(sd.get('label_x_offset', 8.0)),
                    label_y_offset=float(sd.get('label_y_offset', -8.0)),
                    label_font_size=float(sd.get('label_font_size', 12.0)),
                )
            else:
                canvas = _draw_shape_overlay(
                    canvas, sd,
                    line_color=color, line_width=lw, line_style=ls,
                    label_text=str(sd.get('label_text', '')),
                    label_x_offset=float(sd.get('label_x_offset', 8.0)),
                    label_y_offset=float(sd.get('label_y_offset', -8.0)),
                    label_font_size=float(sd.get('label_font_size', 12.0)),
                )

        # ── composite annotations onto float32 result ─────────────────────
        # 1. Mask overlay (RGBA → blend onto float)
        ov_arr = np.array(overlay)
        ov_alpha = ov_arr[:, :, 3:4].astype(np.float32) / 255.0
        ov_rgb = ov_arr[:, :, :3].astype(np.float32) / 255.0
        drawn_mask = ov_alpha > 0
        if result.ndim == 2:
            result = np.stack([result]*3, axis=-1)
        if drawn_mask.any():
            alpha3 = np.broadcast_to(ov_alpha, result.shape)
            result = np.where(np.broadcast_to(drawn_mask, result.shape),
                              (1 - alpha3) * result + alpha3 * ov_rgb,
                              result)

        # 2. Shape drawings (diff canvas_before vs canvas → stamp changed pixels)
        before_arr = np.array(canvas_before)
        after_arr = np.array(canvas) if not isinstance(canvas, np.ndarray) else canvas
        shape_diff = (before_arr != after_arr).any(axis=2)
        if shape_diff.any():
            shape_rgb_f = after_arr.astype(np.float32) / 255.0
            result[shape_diff] = shape_rgb_f[shape_diff]

        self._make_image_output(result)
        self.set_progress(100)
        return True, None


# ─────────────────────────────────────────────────────────────────────────────
# MaskEditorNode
# ─────────────────────────────────────────────────────────────────────────────

class _MaskEditorView(QGraphicsView):
    """Provides a QGraphicsView canvas where the user draws shapes to edit a mask."""
    shape_committed = Signal(str, list)   # (tool, [(x,y), ...] in scene coords)

    _TOOLS = ('rect', 'ellipse', 'polygon', 'lasso')

    def __init__(self, parent=None):
        super().__init__(parent)
        scene = QGraphicsScene(self)
        self.setScene(scene)
        scene.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._tool = 'rect'
        self._drawing = False
        self._start = QPointF()
        self._preview = None        # live QGraphicsItem for rect/ellipse
        self._poly_item = None      # live QGraphicsPathItem for polygon/lasso
        self._poly_pts: list[QPointF] = []
        self._bg_item: QGraphicsPixmapItem | None = None
        self._img_w = 1
        self._img_h = 1

    # ── public ───────────────────────────────────────────────────────────────

    def set_pixmap(self, qpixmap: QPixmap):
        scene = self.scene()
        if self._bg_item:
            scene.removeItem(self._bg_item)
        self._bg_item = scene.addPixmap(qpixmap)
        self._bg_item.setZValue(-1)
        self._img_w = qpixmap.width()
        self._img_h = qpixmap.height()
        scene.setSceneRect(0, 0, self._img_w, self._img_h)
        self._fit()

    def set_tool(self, tool: str):
        self._tool = tool
        self._cancel()

    # ── layout ───────────────────────────────────────────────────────────────

    def _fit(self):
        if self._img_w > 0 and self._img_h > 0:
            self.fitInView(QRectF(0, 0, self._img_w, self._img_h),
                           Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _scene_pt(self, event) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def _make_pen(self) -> QPen:
        pen = QPen(QColor(255, 220, 0))
        pen.setWidthF(1.0)
        pen.setCosmetic(True)
        return pen

    def _cancel(self):
        scene = self.scene()
        for item in (self._preview, self._poly_item):
            if item:
                scene.removeItem(item)
        self._preview = self._poly_item = None
        self._poly_pts = []
        self._drawing = False

    # ── mouse events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pt = self._scene_pt(event)
        if self._tool in ('polygon', 'lasso'):
            if not self._drawing:
                self._drawing = True
                self._poly_pts = [pt]
            else:
                self._poly_pts.append(pt)
            self._update_poly_preview()
        else:
            self._drawing = True
            self._start = pt
            self._update_box_preview(pt)

    def mouseMoveEvent(self, event):
        if not self._drawing:
            return
        pt = self._scene_pt(event)
        if self._tool == 'lasso':
            self._poly_pts.append(pt)
            self._update_poly_preview()
        elif self._tool == 'polygon':
            self._update_poly_preview(tentative=pt)
        else:
            self._update_box_preview(pt)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pt = self._scene_pt(event)
        if self._tool == 'lasso':
            if len(self._poly_pts) >= 3:
                self._commit_polygon()
        elif self._tool != 'polygon':
            # rect / ellipse: commit on release
            pts = [(self._start.x(), self._start.y()), (pt.x(), pt.y())]
            self._cancel()
            self.shape_committed.emit(self._tool, pts)

    def mouseDoubleClickEvent(self, event):
        if self._tool == 'polygon' and self._drawing:
            if len(self._poly_pts) >= 3:
                self._commit_polygon()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._tool == 'polygon' and self._drawing and len(self._poly_pts) >= 3:
                self._commit_polygon()
        elif event.key() == Qt.Key.Key_Escape:
            self._cancel()
        else:
            super().keyPressEvent(event)

    # ── preview drawing ───────────────────────────────────────────────────────

    def _update_box_preview(self, cur: QPointF):
        scene = self.scene()
        if self._preview:
            scene.removeItem(self._preview)
        pen = self._make_pen()
        x0 = min(self._start.x(), cur.x())
        y0 = min(self._start.y(), cur.y())
        x1 = max(self._start.x(), cur.x())
        y1 = max(self._start.y(), cur.y())
        r = QRectF(x0, y0, x1 - x0, y1 - y0)
        if self._tool == 'ellipse':
            self._preview = scene.addEllipse(r, pen)
        else:
            self._preview = scene.addRect(r, pen)
        self._preview.setZValue(10)

    def _update_poly_preview(self, tentative: QPointF | None = None):
        scene = self.scene()
        if self._poly_item:
            scene.removeItem(self._poly_item)
        pts = list(self._poly_pts)
        if tentative:
            pts.append(tentative)
        if len(pts) < 2:
            self._poly_item = None
            return
        path = QPainterPath(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        pen = self._make_pen()
        self._poly_item = scene.addPath(path, pen)
        self._poly_item.setZValue(10)

    def _commit_polygon(self):
        pts = [(p.x(), p.y()) for p in self._poly_pts]
        self._cancel()
        self.shape_committed.emit('polygon', pts)


class MaskEditorWidget(NodeBaseWidget):
    """
    Provides an interactive mask editing widget with shape drawing tools and boolean operations.

    Supports rect, ellipse, polygon, and lasso tools with undo history.
    """

    mask_changed = Signal()

    _OVERLAY_COLOR = (0, 210, 180)   # teal
    _OVERLAY_ALPHA = 0.55            # 0–1
    _MAX_UNDO = 30

    _set_input_signal = Signal(object, object)   # (mask_arr, bg_pil) → main thread

    def __init__(self, parent=None):
        super().__init__(parent, 'mask_editor')

        root = QtWidgets.QVBoxLayout()
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(3)
        container = QtWidgets.QWidget()
        container.setLayout(root)

        # ── tool bar ─────────────────────────────────────────────────────────
        tb = QtWidgets.QHBoxLayout()
        tb.setSpacing(3)
        self._tool_group = QtWidgets.QButtonGroup(container)
        self._tool_names = ['rect', 'ellipse', 'polygon', 'lasso']
        for i, label in enumerate(['Rect', 'Ellipse', 'Polygon', 'Lasso']):
            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(24)
            btn.setMinimumWidth(55)
            self._tool_group.addButton(btn, i)
            tb.addWidget(btn)
            if i == 0:
                btn.setChecked(True)
        tb.addSpacing(6)
        tb.addWidget(QtWidgets.QLabel('Op:'))
        self._op_combo = QtWidgets.QComboBox()
        self._op_combo.addItems(['Union', 'Subtract', 'Intersect', 'XOR', 'Replace'])
        self._op_combo.setFixedHeight(24)
        self._op_combo.setMinimumWidth(55)
        tb.addWidget(self._op_combo)
        tb.addStretch()
        root.addLayout(tb)

        # ── canvas ───────────────────────────────────────────────────────────
        self._view = _MaskEditorView()
        self._view.setMinimumSize(400, 300)
        root.addWidget(self._view)

        # ── bottom bar ───────────────────────────────────────────────────────
        bb = QtWidgets.QHBoxLayout()
        bb.setSpacing(4)
        for label, slot in [('Undo', '_undo'), ('Reset', '_reset'), ('Clear', '_clear')]:
            btn = QtWidgets.QPushButton(label)
            btn.setFixedHeight(22)
            btn.setMinimumWidth(55)
            btn.clicked.connect(getattr(self, slot))
            bb.addWidget(btn)

        bb.addSpacing(8)
        _LS = 'color:#ccc; font-size:9px;'
        lbl_c = QtWidgets.QLabel('Color:')
        lbl_c.setStyleSheet(_LS)
        self._mask_color_btn = QtWidgets.QPushButton()
        self._mask_color_btn.setFixedSize(24, 18)
        self._mask_color = QtGui.QColor(*self._OVERLAY_COLOR)
        self._update_mask_color_swatch()
        self._mask_color_btn.clicked.connect(self._pick_mask_color)
        bb.addWidget(lbl_c)
        bb.addWidget(self._mask_color_btn)

        bb.addSpacing(4)
        lbl_o = QtWidgets.QLabel('Opacity:')
        lbl_o.setStyleSheet(_LS)
        self._mask_opacity_spin = QtWidgets.QSpinBox()
        self._mask_opacity_spin.setRange(10, 100)
        self._mask_opacity_spin.setValue(int(self._OVERLAY_ALPHA * 100))
        self._mask_opacity_spin.setSuffix('%')
        self._mask_opacity_spin.setFixedWidth(58)
        self._mask_opacity_spin.valueChanged.connect(lambda _: self._render())
        bb.addWidget(lbl_o)
        bb.addWidget(self._mask_opacity_spin)

        bb.addStretch()
        self._info_lbl = QtWidgets.QLabel('No mask')
        self._info_lbl.setStyleSheet('color:#aaa; font-size:10px;')
        bb.addWidget(self._info_lbl)
        root.addLayout(bb)

        self.set_custom_widget(container)

        # ── state ────────────────────────────────────────────────────────────
        self._edit_mask: np.ndarray | None = None    # bool H×W
        self._input_mask: np.ndarray | None = None   # original input (for Reset)
        self._history: list[np.ndarray] = []         # undo stack
        self._img_w = 1
        self._img_h = 1
        self._bg_pil = None

        # ── connections ──────────────────────────────────────────────────────
        self._tool_group.idClicked.connect(
            lambda i: self._view.set_tool(self._tool_names[i]))
        self._view.shape_committed.connect(self._on_shape_committed)
        self._set_input_signal.connect(
            self._apply_set_input, Qt.ConnectionType.QueuedConnection)

    def get_value(self):         return ''
    def set_value(self, _value): pass

    # ── public API ───────────────────────────────────────────────────────────

    def set_input(self, mask_arr: np.ndarray | None, bg_pil=None):
        """Thread-safe: called from evaluate() which may run on worker thread."""
        if threading.current_thread() is not threading.main_thread():
            self._set_input_signal.emit(mask_arr, bg_pil)
        else:
            self._apply_set_input(mask_arr, bg_pil)

    def _apply_set_input(self, mask_arr, bg_pil):
        mask_changed = (
            mask_arr is not None and (
                self._input_mask is None
                or mask_arr.shape != self._input_mask.shape
                or not np.array_equal(mask_arr, self._input_mask)
            )
        )
        # Also detect background image change (new image connected)
        bg_changed = False
        if bg_pil is not None:
            if self._bg_pil is None:
                bg_changed = True
            elif isinstance(bg_pil, np.ndarray) and isinstance(self._bg_pil, np.ndarray):
                bg_changed = bg_pil.shape != self._bg_pil.shape
            else:
                bg_changed = True

        if mask_changed or bg_changed:
            if mask_arr is not None:
                self._input_mask = mask_arr.astype(bool)
                self._edit_mask = self._input_mask.copy()
                self._img_h, self._img_w = mask_arr.shape[:2]
            elif bg_pil is not None:
                # New image but no mask — reset to blank
                h = bg_pil.shape[0] if isinstance(bg_pil, np.ndarray) else 512
                w = bg_pil.shape[1] if isinstance(bg_pil, np.ndarray) else 512
                self._input_mask = np.zeros((h, w), dtype=bool)
                self._edit_mask = self._input_mask.copy()
                self._img_h, self._img_w = h, w
            self._history.clear()
        elif mask_arr is None and self._edit_mask is None:
            return
        self._bg_pil = bg_pil
        self._render()

    def get_mask(self) -> np.ndarray | None:
        return self._edit_mask

    # ── shape operations ─────────────────────────────────────────────────────

    def _on_shape_committed(self, tool: str, pts: list):
        if self._edit_mask is None or len(pts) < 2:
            return
        shape_mask = self._rasterize(tool, pts)
        if shape_mask is None:
            return
        self._push_history()
        op = self._op_combo.currentText()
        if op == 'Union':
            self._edit_mask = self._edit_mask | shape_mask
        elif op == 'Subtract':
            self._edit_mask = self._edit_mask & ~shape_mask
        elif op == 'Intersect':
            self._edit_mask = self._edit_mask & shape_mask
        elif op == 'XOR':
            self._edit_mask = self._edit_mask ^ shape_mask
        elif op == 'Replace':
            self._edit_mask = shape_mask.copy()
        self._render()
        self.mask_changed.emit()

    def _rasterize(self, tool: str, pts: list) -> np.ndarray | None:
        from PIL import Image as _PIL, ImageDraw
        img = _PIL.new('L', (self._img_w, self._img_h), 0)
        draw = ImageDraw.Draw(img)
        if tool in ('rect', 'ellipse'):
            x0 = min(pts[0][0], pts[1][0])
            y0 = min(pts[0][1], pts[1][1])
            x1 = max(pts[0][0], pts[1][0])
            y1 = max(pts[0][1], pts[1][1])
            if x1 <= x0 or y1 <= y0:
                return None
            if tool == 'ellipse':
                draw.ellipse([x0, y0, x1, y1], fill=255)
            else:
                draw.rectangle([x0, y0, x1, y1], fill=255)
        elif tool in ('polygon', 'lasso'):
            if len(pts) < 3:
                return None
            draw.polygon([(float(x), float(y)) for x, y in pts], fill=255)
        return np.array(img) > 0

    # ── undo / reset / clear ─────────────────────────────────────────────────

    def _push_history(self):
        if self._edit_mask is not None:
            self._history.append(self._edit_mask.copy())
            if len(self._history) > self._MAX_UNDO:
                self._history.pop(0)

    def _undo(self):
        if self._history:
            self._edit_mask = self._history.pop()
            self._render()
            self.mask_changed.emit()

    def _reset(self):
        if self._input_mask is not None:
            self._push_history()
            self._edit_mask = self._input_mask.copy()
            self._render()
            self.mask_changed.emit()

    def _clear(self):
        if self._edit_mask is not None:
            self._push_history()
            self._edit_mask = np.zeros((self._img_h, self._img_w), dtype=bool)
            self._render()
            self.mask_changed.emit()

    # ── rendering ────────────────────────────────────────────────────────────

    def _update_mask_color_swatch(self):
        c = self._mask_color
        self._mask_color_btn.setStyleSheet(
            f'background-color: rgb({c.red()},{c.green()},{c.blue()});'
            f'border: 2px solid #555; border-radius: 3px;')

    def _pick_mask_color(self):
        c = QtWidgets.QColorDialog.getColor(
            self._mask_color, QtWidgets.QApplication.activeWindow(), "Mask Color")
        if c.isValid():
            self._mask_color = c
            self._update_mask_color_swatch()
            self._render()

    def _render(self):
        if self._edit_mask is None:
            return
        h, w = self._edit_mask.shape
        # Background: reference image or dark gray
        if self._bg_pil is not None:
            bg = _ensure_display_rgb(self._bg_pil)
            bg_h, bg_w = bg.shape[:2]
            if bg_w != w or bg_h != h:
                from PIL import Image as _PIL
                bg = np.array(_PIL.fromarray(bg).resize((w, h)))
        else:
            bg = np.full((h, w, 3), 30, dtype=np.uint8)

        # Color overlay on mask pixels
        result = bg.astype(np.float32)
        fg = self._edit_mask
        r = self._mask_color.red()
        g = self._mask_color.green()
        b = self._mask_color.blue()
        a = self._mask_opacity_spin.value() / 100.0
        result[fg, 0] = result[fg, 0] * (1 - a) + r * a
        result[fg, 1] = result[fg, 1] * (1 - a) + g * a
        result[fg, 2] = result[fg, 2] * (1 - a) + b * a
        result = result.clip(0, 255).astype(np.uint8)

        qimg = QImage(result.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        self._view.set_pixmap(QPixmap.fromImage(qimg))
        self._info_lbl.setText(f'Pixels: {int(fg.sum()):,}')


class MaskEditorNode(BaseExecutionNode):
    """
    Interactively edits a mask by drawing shapes and applying boolean operations.

    Supports add, subtract, and intersect modes with rect, ellipse, polygon, and lasso tools. Accepts an optional background image for visual reference and an optional mask input as the starting state.
    """

    __identifier__ = 'nodes.image_process.morphology'
    NODE_NAME = 'Mask Editor'
    PORT_SPEC = {'inputs': ['image', 'mask'], 'outputs': ['mask']}

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_input('mask',  color=PORT_COLORS['mask'])
        self.add_output('mask', color=PORT_COLORS['mask'], multi_output=True)

        self._editor = MaskEditorWidget(self.view)
        self._editor.mask_changed.connect(self._on_mask_changed)
        self.add_custom_widget(self._editor)

    def _on_mask_changed(self):
        """Called when user draws/undoes in the editor; push result downstream."""
        result = self._editor.get_mask()
        if result is not None:
            self.output_values['mask'] = MaskData(payload=(result.astype(np.uint8) * 255))
        else:
            self.output_values['mask'] = MaskData(payload=np.zeros((1, 1), dtype=np.uint8))
        # Mark this node and all downstream dirty so Run Graph picks up edits
        self.mark_dirty()

    def evaluate(self):
        self.reset_progress()

        # ── optional background image ─────────────────────────────────────
        bg_arr = None
        img_port = self.inputs().get('image')
        if img_port and img_port.connected_ports():
            cp = img_port.connected_ports()[0]
            data = cp.node().output_values.get(cp.name())
            if isinstance(data, ImageData):
                bg_arr = data.payload       # numpy array

        # ── mask input ───────────────────────────────────────────────────
        mask_arr = None
        mask_port = self.inputs().get('mask')
        if mask_port and mask_port.connected_ports():
            cp = mask_port.connected_ports()[0]
            data = cp.node().output_values.get(cp.name())
            if isinstance(data, MaskData):
                m = data.payload
                mask_arr = m.astype(bool) if m.dtype != bool else m

        if mask_arr is None and bg_arr is None:
            return False, 'Connect a mask or image'

        # If only image is connected, start with a blank mask
        if mask_arr is None:
            h, w = bg_arr.shape[:2]
            mask_arr = np.zeros((h, w), dtype=bool)

        # Before calling set_input (which queues a signal to reset _edit_mask),
        # check if the user has edited the mask and the input is the same.
        # We compare against _input_mask which was set by the previous set_input.
        prev_input = getattr(self._editor, '_input_mask', None)
        edited = self._editor._edit_mask  # direct access, safe from bg thread (just a numpy read)
        input_unchanged = (
            prev_input is not None
            and mask_arr.shape == prev_input.shape
            and np.array_equal(mask_arr.astype(bool), prev_input)
        )
        has_edits = (
            input_unchanged
            and edited is not None
            and not np.array_equal(edited, prev_input)
        )

        self._editor.set_input(mask_arr, bg_arr)
        self.set_progress(50)

        # ── output ───────────────────────────────────────────────────────
        if has_edits:
            # Input hasn't changed and user has edits — keep user's version
            out_mask = edited.astype(np.uint8) * 255
        else:
            # New input or no edits — output the input mask
            out_mask = mask_arr.astype(np.uint8) * 255
        self.output_values['mask'] = MaskData(payload=out_mask)

        self.set_progress(100)
        return True, None


# ═══════════════════════════════════════════════════════════════════════
# ScaleBarNode — draw a calibrated scale bar on a microscopy image
# ═══════════════════════════════════════════════════════════════════════

class _DualIntWidget(NodeBaseWidget):
    """Two labeled integer spinboxes in a single row."""

    def __init__(self, parent=None, name='', label='',
                 labels=('A', 'B'), values=(0, 0),
                 mins=(0, 0), maxs=(999, 999), steps=(1, 1)):
        super().__init__(parent, name, label)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self._spins = []
        for i, lbl_text in enumerate(labels):
            lbl = QtWidgets.QLabel(lbl_text)
            lbl.setStyleSheet('color: #999; font-size: 9px;')
            layout.addWidget(lbl)
            spin = QtWidgets.QSpinBox()
            spin.setRange(mins[i], maxs[i])
            spin.setValue(values[i])
            spin.setSingleStep(steps[i])
            spin.setMinimumWidth(45)
            spin.valueChanged.connect(
                lambda _v: self.value_changed.emit(self.get_name(), self.get_value()))
            layout.addWidget(spin)
            self._spins.append(spin)

        self.set_custom_widget(container)

    def get_value(self):
        return [s.value() for s in self._spins]

    def set_value(self, value):
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            for i, s in enumerate(self._spins):
                s.blockSignals(True)
                try:
                    s.setValue(int(value[i]))
                except (ValueError, TypeError):
                    pass
                s.blockSignals(False)


class ScaleBarNode(BaseImageProcessNode):
    """
    Draws a calibrated scale bar on a microscopy image.

    Reads the `scale_um` metadata from the upstream ImageData to calculate
    the correct pixel length for the bar. If no scale info is available,
    the node reports an error.

    Options:
    - **bar_length_um** — desired bar length in micrometers
    - **position** — corner placement
    - **bar_color** — color of the bar and label
    - **bar_height** — thickness in pixels
    - **show_label** — display "100 µm" text
    - **font_size** — label size
    - **padding_x / padding_y** — margin from image edge

    Keywords: scale bar, calibration, micrometer, microscopy, annotation, 比例尺, 校正, 微米, 標註
    """
    __identifier__ = 'nodes.image_process.Visualize'
    NODE_NAME      = 'Scale Bar'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}

    _UI_PROPS = frozenset({
        'color', 'pos', 'selected', 'name', 'progress', 'image_view',
        'show_preview', 'live_preview', 'bar_color',
    })
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'], multi_output=True)

        from nodes.base import NodeColorPickerWidget
        import NodeGraphQt
        H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value

        # Row 1: bar length + position
        self._add_float_spinbox('bar_length_um', 'Bar Length (µm)',
                                value=50.0, min_val=0.1, max_val=10000.0, step=0.5, decimals=2)
        self.add_combo_menu('position', 'Position',
                            items=['bottom-right', 'bottom-left', 'top-right', 'top-left'])

        # Row 2: color
        color_w = NodeColorPickerWidget(self.view, name='bar_color', label='Bar Color')
        color_w.set_value([255, 255, 255, 255])
        self.add_custom_widget(color_w, widget_type=H, tab='Properties')

        # Row 3: bar height + font size + text gap (compact)
        w = _DualIntWidget(self.view, name='bar_style', label='Height / Font',
                           labels=('H', 'Font'), values=(6, 14),
                           mins=(1, 6), maxs=(50, 72))
        self.add_custom_widget(w, widget_type=H, tab='Properties')

        # Row 4: show label + text-bar gap
        self.add_combo_menu('show_label', 'Show Label', items=['True', 'False'])
        self._add_int_spinbox('text_gap', 'Text-Bar Gap (px)', value=4, min_val=0, max_val=50)

        # Row 5: padding X + Y (compact)
        w = _DualIntWidget(self.view, name='padding', label='Padding',
                           labels=('X', 'Y'), values=(20, 20),
                           mins=(0, 0), maxs=(500, 500))
        self.add_custom_widget(w, widget_type=H, tab='Properties')

        self.create_preview_widgets()
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        from PIL import Image as _PIL, ImageDraw, ImageFont

        in_port = self.inputs().get('image')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No image connected"
        connected = in_port.connected_ports()[0]
        img_data = connected.node().output_values.get(connected.name())

        if not isinstance(img_data, ImageData):
            self.mark_error()
            return False, "Input is not an ImageData"

        scale_um = img_data.scale_um
        if not scale_um or scale_um <= 0:
            self.mark_error()
            return False, "No scale info. Connect to an image with calibration data (e.g. OIR, calibrated TIFF)."

        self.set_progress(20)

        bar_um   = float(self.get_property('bar_length_um') or 5)
        position = str(self.get_property('position') or 'bottom-right')
        color_raw = self.get_property('bar_color') or [255, 255, 255, 255]
        bar_color = tuple(int(c) for c in color_raw[:4]) if isinstance(color_raw, (list, tuple)) else (255, 255, 255, 255)
        show_lbl = str(self.get_property('show_label') or 'True') == 'True'
        text_gap = int(self.get_property('text_gap') or 4)

        # bar_style = [height, font_size]
        bar_style = self.get_property('bar_style') or [6, 14]
        if isinstance(bar_style, (list, tuple)) and len(bar_style) >= 2:
            bar_h, fsize = int(bar_style[0]), int(bar_style[1])
        else:
            bar_h, fsize = 6, 14

        # padding = [x, y]
        padding = self.get_property('padding') or [20, 20]
        if isinstance(padding, (list, tuple)) and len(padding) >= 2:
            pad_x, pad_y = int(padding[0]), int(padding[1])
        else:
            pad_x, pad_y = 20, 20

        bar_px = int(round(bar_um / scale_um))
        if bar_px < 1:
            self.mark_error()
            return False, f"Bar too short: {bar_um} µm = {bar_px} px at {scale_um:.4f} µm/px"

        self.set_progress(40)

        arr = img_data.payload.copy()  # float32 [0, 1]
        h, w = arr.shape[:2]

        # Normalize bar color to [0, 1]
        bar_color_f = tuple(c / 255.0 for c in bar_color[:3])

        # Font — render text label as a small PIL mask, then stamp onto float array
        font = None
        for font_name in ("Arial", "arial.ttf",
                          "C:/Windows/Fonts/arial.ttf",
                          "/System/Library/Fonts/Helvetica.ttc",
                          "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            try:
                font = ImageFont.truetype(font_name, fsize)
                break
            except (OSError, IOError):
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size=fsize)
            except TypeError:
                font = ImageFont.load_default()

        # Label text
        if bar_um >= 1000:
            label = f"{bar_um / 1000:.1f} mm"
        elif bar_um >= 1:
            label = f"{int(bar_um)} \u00b5m" if bar_um == int(bar_um) else f"{bar_um:.1f} \u00b5m"
        else:
            label = f"{bar_um * 1000:.0f} nm"

        if show_lbl:
            tmp = _PIL.new('L', (w, h), 0)
            draw = ImageDraw.Draw(tmp)
            bbox = draw.textbbox((0, 0), label, font=font)
            lbl_w, lbl_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            lbl_w, lbl_h = 0, 0

        total_w = max(bar_px, lbl_w)
        total_h = bar_h + (lbl_h + text_gap if show_lbl else 0)

        x0 = (w - pad_x - total_w) if 'right' in position else pad_x
        y0 = (h - pad_y - total_h) if 'bottom' in position else pad_y

        self.set_progress(60)

        # Draw text label as a mask, then blend into float array
        if show_lbl:
            txt_x = x0 + (bar_px - lbl_w) // 2
            txt_y = y0
            mask_img = _PIL.new('L', (w, h), 0)
            mask_draw = ImageDraw.Draw(mask_img)
            mask_draw.text((txt_x, txt_y), label, fill=255, font=font)
            mask = np.array(mask_img).astype(np.float32) / 255.0
            bar_y = y0 + lbl_h + text_gap
        else:
            mask = None
            bar_y = y0

        # Stamp text onto image
        if mask is not None:
            if arr.ndim == 2:
                arr = np.where(mask > 0, mask * bar_color_f[0], arr)
            else:
                for c in range(min(3, arr.shape[2])):
                    arr[:, :, c] = np.where(mask > 0, mask * bar_color_f[c], arr[:, :, c])

        # Draw bar rectangle directly on float array
        by0 = max(0, bar_y)
        by1 = min(h, bar_y + bar_h)
        bx0 = max(0, x0)
        bx1 = min(w, x0 + bar_px)
        if arr.ndim == 2:
            arr[by0:by1, bx0:bx1] = bar_color_f[0]
        else:
            for c in range(min(3, arr.shape[2])):
                arr[by0:by1, bx0:bx1, c] = bar_color_f[c]

        self.set_progress(90)
        self._make_image_output(arr)
        self.set_display(arr)
        self.set_progress(100)
        return True, None


# ===========================================================================
# Mask Overlay — lightweight contour/fill overlay on an image
# ===========================================================================

class MaskOverlayNode(BaseImageProcessNode):
    """
    Draw a mask contour (or fill) on an image.

    A lightweight alternative to Draw Shape for simple mask visualization.
    Connect an image and a mask, and the mask boundary is drawn as a colored
    contour on the output image. Optionally fill the masked region with a
    semi-transparent color.

    Controls:
    - Line width, style (solid/dashed/dotted), and color
    - Fill toggle with adjustable opacity

    Keywords: mask, contour, outline, overlay, boundary, fill, 遮罩, 輪廓, 疊加
    """
    __identifier__ = 'nodes.image_process.Visualize'
    NODE_NAME      = 'Mask Overlay'
    PORT_SPEC      = {'inputs': ['image', 'mask'], 'outputs': ['image']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_input('mask', color=PORT_COLORS['mask'])
        self.add_output('image', color=PORT_COLORS['image'], multi_output=True)

        import NodeGraphQt
        H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value

        # Row 1: Color + Line Width + Style
        row1 = QtWidgets.QWidget()
        lay1 = QtWidgets.QHBoxLayout(row1)
        lay1.setContentsMargins(0, 0, 0, 0)
        lay1.setSpacing(4)

        _LS = 'color:#ccc; font-size:9px;'

        lbl_c = QtWidgets.QLabel('Color:')
        lbl_c.setStyleSheet(_LS)
        self._color_btn = QtWidgets.QPushButton()
        self._color_btn.setFixedSize(28, 32)
        self._line_color = QtGui.QColor(255, 255, 255)
        self._update_color_swatch()
        self._color_btn.clicked.connect(self._pick_color)
        self.create_property('line_color', [255, 255, 255, 255])

        lbl_w = QtWidgets.QLabel('Width:')
        lbl_w.setStyleSheet(_LS)
        self._width_spin = QtWidgets.QDoubleSpinBox()
        self._width_spin.setRange(0.5, 20.0)
        self._width_spin.setValue(2.0)
        self._width_spin.setSingleStep(0.5)
        self._width_spin.setDecimals(1)
        self._width_spin.setFixedWidth(48)
        self.create_property('line_width', 2.0)
        self._width_spin.valueChanged.connect(
            lambda v: self.set_property('line_width', v))

        lbl_s = QtWidgets.QLabel('Style:')
        lbl_s.setStyleSheet(_LS)
        self._style_combo = QtWidgets.QComboBox()
        self._style_combo.addItems(['solid', 'dashed', 'dotted', 'dashdot'])
        self._style_combo.setFixedWidth(72)
        self.create_property('line_style', 'solid')
        self._style_combo.currentTextChanged.connect(
            lambda v: self.set_property('line_style', v))

        lay1.addWidget(lbl_c)
        lay1.addWidget(self._color_btn)
        lay1.addSpacing(6)
        lay1.addWidget(lbl_w)
        lay1.addWidget(self._width_spin)
        lay1.addSpacing(6)
        lay1.addWidget(lbl_s)
        lay1.addWidget(self._style_combo)
        lay1.addStretch()

        class _RowWidget(NodeBaseWidget):
            def __init__(self, parent, name, widget):
                super().__init__(parent, name, '')
                self.set_custom_widget(widget)
            def get_value(self): return None
            def set_value(self, v): pass

        self.add_custom_widget(_RowWidget(self.view, '_overlay_row1', row1),
                               widget_type=H, tab='Properties')

        # Row 2: Fill toggle + Opacity
        row2 = QtWidgets.QWidget()
        lay2 = QtWidgets.QHBoxLayout(row2)
        lay2.setContentsMargins(0, 0, 0, 0)
        lay2.setSpacing(4)

        lbl_f = QtWidgets.QLabel('Fill:')
        lbl_f.setStyleSheet(_LS)
        self._fill_cb = QtWidgets.QCheckBox()
        self.create_property('fill_mode', 'Off')
        self._fill_cb.toggled.connect(
            lambda v: self.set_property('fill_mode', 'On' if v else 'Off'))

        lbl_a = QtWidgets.QLabel('Opacity:')
        lbl_a.setStyleSheet(_LS)
        self._opacity_spin = QtWidgets.QSpinBox()
        self._opacity_spin.setRange(0, 100)
        self._opacity_spin.setValue(30)
        self._opacity_spin.setSuffix('%')
        self._opacity_spin.setFixedWidth(58)
        self.create_property('fill_opacity', 30)
        self._opacity_spin.valueChanged.connect(
            lambda v: self.set_property('fill_opacity', v))

        lay2.addWidget(lbl_f)
        lay2.addWidget(self._fill_cb)
        lay2.addSpacing(6)
        lay2.addWidget(lbl_a)
        lay2.addWidget(self._opacity_spin)
        lay2.addStretch()

        self.add_custom_widget(_RowWidget(self.view, '_overlay_row2', row2),
                               widget_type=H, tab='Properties')

        self.create_preview_widgets()

    def _update_color_swatch(self):
        c = self._line_color
        self._color_btn.setStyleSheet(
            f'background-color: rgb({c.red()},{c.green()},{c.blue()});'
            f'border: 2px solid #555; border-radius: 3px;')

    def _pick_color(self):
        c = QtWidgets.QColorDialog.getColor(
            self._line_color, QtWidgets.QApplication.activeWindow(), "Line Color")
        if c.isValid():
            self._line_color = c
            self._update_color_swatch()
            self.set_property('line_color', [c.red(), c.green(), c.blue(), c.alpha()])

    def evaluate(self):
        self.reset_progress()
        from PIL import Image as _PIL, ImageDraw

        # Read image
        data = self._get_input_image_data()
        if data is None:
            return False, "No image connected"
        arr = data.payload.copy()
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        h, w = arr.shape[:2]

        # Read mask
        mask_port = self.inputs().get('mask')
        if not mask_port or not mask_port.connected_ports():
            # No mask — pass through image
            self._make_image_output(arr)
            self.set_display(arr)
            self.set_progress(100)
            self.mark_clean()
            return True, None

        cp = mask_port.connected_ports()[0]
        mdata = cp.node().output_values.get(cp.name())
        if not isinstance(mdata, MaskData):
            return False, "Mask input must be MaskData"
        mask_bool = mdata.payload.astype(bool) if mdata.payload.dtype != bool else mdata.payload

        self.set_progress(30)

        # Read style properties
        lw = max(1, int(self.get_property('line_width')))
        ls = str(self.get_property('line_style') or 'solid')
        color_raw = self.get_property('line_color') or [0, 220, 220, 255]
        color = (int(color_raw[0]), int(color_raw[1]), int(color_raw[2]))
        color_f = tuple(c / 255.0 for c in color)
        fill_on = str(self.get_property('fill_mode')) == 'On'
        fill_pct = int(self.get_property('fill_opacity') or 30)

        self.set_progress(50)

        # Fill: blend directly on float array
        if fill_on and mask_bool.any():
            alpha_f = fill_pct / 100.0
            for c in range(3):
                arr[:, :, c] = np.where(mask_bool,
                    (1 - alpha_f) * arr[:, :, c] + alpha_f * color_f[c],
                    arr[:, :, c])

        self.set_progress(70)

        # Contour: draw on RGBA overlay, stamp onto float
        paths = _mask_to_outline_paths(mask_bool)
        if paths:
            ov = _PIL.new('RGBA', (w, h), (0, 0, 0, 0))
            ov_draw = ImageDraw.Draw(ov)
            for pts in paths:
                _draw_styled_polyline(ov_draw, pts,
                                      color=color, width=lw,
                                      style=ls, closed=False)
            ov_arr = np.array(ov)
            ov_alpha = ov_arr[:, :, 3:4].astype(np.float32) / 255.0
            ov_rgb = ov_arr[:, :, :3].astype(np.float32) / 255.0
            mask_drawn = ov_alpha > 0
            if mask_drawn.any():
                alpha3 = np.broadcast_to(ov_alpha, arr.shape)
                mask3 = np.broadcast_to(mask_drawn, arr.shape)
                arr = np.where(mask3,
                    (1 - alpha3) * arr + alpha3 * ov_rgb, arr)

        self.set_progress(90)
        self._make_image_output(arr)
        self.set_display(arr)
        self.set_progress(100)
        self.mark_clean()
        return True, None


class LabelOverlayNode(BaseImageProcessNode):
    """
    Overlay a label image on top of a base image with automatic coloring.

    Each unique label gets a distinct color from the selected colormap.
    Background (label 0) is always transparent.

    Controls:
    - Opacity of the label overlay
    - Colormap (tab10, tab20, Set1, Set3, etc.)
    - Line width and style for contour-only mode
    - Fill toggle: filled regions or contour outlines only

    Keywords: label, overlay, segmentation, colormap, contour, fill, 標籤, 疊加, 分割
    """
    __identifier__ = 'nodes.image_process.Visualize'
    NODE_NAME      = 'Label Overlay'
    PORT_SPEC      = {'inputs': ['image', 'label'], 'outputs': ['image']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_input('label_image', color=PORT_COLORS.get('label', (160, 220, 40)))
        self.add_output('image', color=PORT_COLORS['image'], multi_output=True)

        import NodeGraphQt
        H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value

        _LS = 'color:#ccc; font-size:9px;'

        # Row 1: Colormap + Opacity
        row1 = QtWidgets.QWidget()
        lay1 = QtWidgets.QHBoxLayout(row1)
        lay1.setContentsMargins(0, 0, 0, 0)
        lay1.setSpacing(4)

        lbl_cm = QtWidgets.QLabel('Cmap:')
        lbl_cm.setStyleSheet(_LS)
        self._cmap_combo = QtWidgets.QComboBox()
        self._cmap_combo.addItems(['default', 'tab10', 'tab20', 'Set1', 'Set2', 'Set3',
                                    'Paired', 'Accent', 'Pastel1', 'Dark2'])
        self._cmap_combo.setMinimumWidth(72)
        self._cmap_combo.setMaxVisibleItems(6)
        self._cmap_combo.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.create_property('colormap', 'default')
        self._cmap_combo.currentTextChanged.connect(
            lambda v: self.set_property('colormap', v))

        lbl_a = QtWidgets.QLabel('Opacity:')
        lbl_a.setStyleSheet(_LS)
        self._opacity_spin = QtWidgets.QSpinBox()
        self._opacity_spin.setRange(0, 100)
        self._opacity_spin.setValue(50)
        self._opacity_spin.setSuffix('%')
        self._opacity_spin.setFixedWidth(58)
        self.create_property('opacity', 50)
        self._opacity_spin.valueChanged.connect(
            lambda v: self.set_property('opacity', v))

        lay1.addWidget(lbl_cm)
        lay1.addWidget(self._cmap_combo)
        lay1.addSpacing(6)
        lay1.addWidget(lbl_a)
        lay1.addWidget(self._opacity_spin)
        lay1.addStretch()

        class _RowWidget(NodeBaseWidget):
            def __init__(self, parent, name, widget):
                super().__init__(parent, name, '')
                self.set_custom_widget(widget)
            def get_value(self): return None
            def set_value(self, v): pass

        self.add_custom_widget(_RowWidget(self.view, '_lbl_ov_row1', row1),
                               widget_type=H, tab='Properties')

        # Row 2: Fill + Line Width + Style
        row2 = QtWidgets.QWidget()
        lay2 = QtWidgets.QHBoxLayout(row2)
        lay2.setContentsMargins(0, 0, 0, 0)
        lay2.setSpacing(4)

        lbl_f = QtWidgets.QLabel('Fill:')
        lbl_f.setStyleSheet(_LS)
        self._fill_cb = QtWidgets.QCheckBox()
        self._fill_cb.setChecked(True)
        self.create_property('fill_mode', 'On')
        self._fill_cb.toggled.connect(
            lambda v: self.set_property('fill_mode', 'On' if v else 'Off'))

        lbl_w = QtWidgets.QLabel('Width:')
        lbl_w.setStyleSheet(_LS)
        self._width_spin = QtWidgets.QDoubleSpinBox()
        self._width_spin.setRange(0.5, 20.0)
        self._width_spin.setValue(2.0)
        self._width_spin.setSingleStep(0.5)
        self._width_spin.setDecimals(1)
        self._width_spin.setMinimumWidth(48)
        self.create_property('line_width', 2.0)
        self._width_spin.valueChanged.connect(
            lambda v: self.set_property('line_width', v))

        lbl_s = QtWidgets.QLabel('Style:')
        lbl_s.setStyleSheet(_LS)
        self._style_combo = QtWidgets.QComboBox()
        self._style_combo.addItems(['solid', 'dashed', 'dotted', 'dashdot'])
        self._style_combo.setMinimumWidth(72)
        self.create_property('line_style', 'solid')
        self._style_combo.currentTextChanged.connect(
            lambda v: self.set_property('line_style', v))

        lay2.addWidget(lbl_f)
        lay2.addWidget(self._fill_cb)
        lay2.addSpacing(6)
        lay2.addWidget(lbl_w)
        lay2.addWidget(self._width_spin)
        lay2.addSpacing(6)
        lay2.addWidget(lbl_s)
        lay2.addWidget(self._style_combo)
        lay2.addStretch()

        self.add_custom_widget(_RowWidget(self.view, '_lbl_ov_row2', row2),
                               widget_type=H, tab='Properties')

        self.create_preview_widgets()
        self._fix_widget_z_order()

    def evaluate(self):
        self.reset_progress()
        from PIL import Image as _PIL, ImageDraw
        from skimage.segmentation import find_boundaries
        import matplotlib.cm as cm

        # Read image
        data = self._get_input_image_data()
        if data is None:
            return False, "No image connected"
        arr = data.payload.copy()
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        h, w = arr.shape[:2]

        # Read label image
        label_port = self.inputs().get('label_image')
        if not label_port or not label_port.connected_ports():
            self._make_image_output(arr)
            self.set_display(arr)
            self.set_progress(100)
            self.mark_clean()
            return True, None

        cp = label_port.connected_ports()[0]
        ldata = cp.node().output_values.get(cp.name())
        if ldata is None:
            self._make_image_output(arr)
            self.set_display(arr)
            self.mark_clean()
            return True, None

        labels = ldata.payload if hasattr(ldata, 'payload') else ldata
        if hasattr(labels, 'numpy'):
            labels = labels.numpy()
        labels = np.asarray(labels)
        if labels.ndim == 3:
            labels = labels[:, :, 0]
        labels = labels.astype(np.int32)

        self.set_progress(30)

        # Get parameters
        cmap_name = str(self.get_property('colormap') or 'default')
        alpha_pct = int(self.get_property('opacity') or 50)
        alpha_f = alpha_pct / 100.0
        fill_on = str(self.get_property('fill_mode')) == 'On'
        lw = max(1, int(self.get_property('line_width')))
        ls = str(self.get_property('line_style') or 'solid')

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]  # skip background
        n_labels = int(labels.max()) if labels.max() > 0 else 1

        # Build color lookup: label → (r, g, b) in 0-255
        if cmap_name == 'default':
            try:
                from .vision_nodes import _label_palette
            except ImportError:
                from vision_nodes import _label_palette
            palette = _label_palette(n_labels)
            def _get_color(lbl):
                c = palette[(int(lbl) - 1) % len(palette)]
                return c  # (r, g, b) uint8
        else:
            try:
                cmap = cm.get_cmap(cmap_name)
            except ValueError:
                cmap = cm.get_cmap('tab20')
            n_colors = cmap.N if hasattr(cmap, 'N') else 20
            def _get_color(lbl):
                rgba = cmap((int(lbl) - 1) % n_colors / max(n_colors - 1, 1))
                return (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))

        self.set_progress(50)

        # Fill: blend each label's color onto the base image
        if fill_on:
            for lbl in unique_labels:
                color = _get_color(lbl)
                color_f = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
                mask = labels == lbl
                for c in range(3):
                    arr[:, :, c] = np.where(mask,
                        (1 - alpha_f) * arr[:, :, c] + alpha_f * color_f[c],
                        arr[:, :, c])

        self.set_progress(70)

        # Contours: draw boundaries for each label
        boundaries = find_boundaries(labels, mode='thick')
        if boundaries.any():
            # Draw colored contours per label
            ov = _PIL.new('RGBA', (w, h), (0, 0, 0, 0))
            ov_draw = ImageDraw.Draw(ov)

            for lbl in unique_labels:
                color = _get_color(lbl)
                lbl_boundary = boundaries & (labels == lbl)
                if not lbl_boundary.any():
                    continue
                # Convert boundary mask to contour paths
                paths = _mask_to_outline_paths(lbl_boundary)
                for pts in paths:
                    _draw_styled_polyline(ov_draw, pts,
                                          color=color, width=lw,
                                          style=ls, closed=False)

            ov_arr = np.array(ov)
            ov_alpha = ov_arr[:, :, 3:4].astype(np.float32) / 255.0
            ov_rgb = ov_arr[:, :, :3].astype(np.float32) / 255.0
            mask_drawn = ov_alpha > 0
            if mask_drawn.any():
                alpha3 = np.broadcast_to(ov_alpha, arr.shape)
                mask3 = np.broadcast_to(mask_drawn, arr.shape)
                arr = np.where(mask3,
                    (1 - alpha3) * arr + alpha3 * ov_rgb, arr)

        self.set_progress(90)
        self._make_image_output(arr)
        self.set_display(arr)
        self.set_progress(100)
        self.mark_clean()
        return True, None


# ─────────────────────────────────────────────────────────────────────────────
# LabelEditorNode
# ─────────────────────────────────────────────────────────────────────────────

class _LabelEditorView(QGraphicsView):
    """QGraphicsView canvas for multi-label editing with shape, brush, and fill tools."""
    shape_committed = Signal(str, list)   # (tool, data)  — data depends on tool

    _TOOLS = ('rect', 'ellipse', 'polygon', 'lasso', 'brush', 'flood_fill')

    def __init__(self, parent=None):
        super().__init__(parent)
        scene = QGraphicsScene(self)
        self.setScene(scene)
        scene.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._tool = 'rect'
        self._drawing = False
        self._start = QPointF()
        self._preview = None
        self._poly_item = None
        self._poly_pts: list[QPointF] = []
        self._bg_item: QGraphicsPixmapItem | None = None
        self._img_w = 1
        self._img_h = 1

        # brush tool state
        self._brush_size = 5
        self._brush_pts: list[tuple[int, int]] = []
        self._brush_preview_items: list = []

    # ── public ───────────────────────────────────────────────────────────────

    def set_pixmap(self, qpixmap: QPixmap):
        scene = self.scene()
        if self._bg_item:
            scene.removeItem(self._bg_item)
        self._bg_item = scene.addPixmap(qpixmap)
        self._bg_item.setZValue(-1)
        self._img_w = qpixmap.width()
        self._img_h = qpixmap.height()
        scene.setSceneRect(0, 0, self._img_w, self._img_h)
        self._fit()

    def set_tool(self, tool: str):
        self._tool = tool
        self._cancel()

    def set_brush_size(self, size: int):
        self._brush_size = max(1, min(size, 50))

    def brush_size(self) -> int:
        return self._brush_size

    # ── layout ───────────────────────────────────────────────────────────────

    def _fit(self):
        if self._img_w > 0 and self._img_h > 0:
            self.fitInView(QRectF(0, 0, self._img_w, self._img_h),
                           Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _scene_pt(self, event) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def _make_pen(self) -> QPen:
        pen = QPen(QColor(255, 220, 0))
        pen.setWidthF(1.0)
        pen.setCosmetic(True)
        return pen

    def _cancel(self):
        scene = self.scene()
        for item in (self._preview, self._poly_item):
            if item:
                scene.removeItem(item)
        self._preview = self._poly_item = None
        self._poly_pts = []
        self._drawing = False
        self._clear_brush_preview()
        self._brush_pts = []

    def _clear_brush_preview(self):
        scene = self.scene()
        for item in self._brush_preview_items:
            scene.removeItem(item)
        self._brush_preview_items = []

    # ── mouse events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pt = self._scene_pt(event)
        if self._tool == 'flood_fill':
            ix = int(pt.x())
            iy = int(pt.y())
            if 0 <= ix < self._img_w and 0 <= iy < self._img_h:
                self.shape_committed.emit('flood_fill', [(ix, iy)])
            return
        if self._tool == 'brush':
            self._drawing = True
            self._brush_pts = []
            self._stamp_brush(pt)
            return
        if self._tool in ('polygon', 'lasso'):
            if not self._drawing:
                self._drawing = True
                self._poly_pts = [pt]
            else:
                self._poly_pts.append(pt)
            self._update_poly_preview()
        else:
            self._drawing = True
            self._start = pt
            self._update_box_preview(pt)

    def mouseMoveEvent(self, event):
        if not self._drawing:
            return
        pt = self._scene_pt(event)
        if self._tool == 'brush':
            self._stamp_brush(pt)
            return
        if self._tool == 'lasso':
            self._poly_pts.append(pt)
            self._update_poly_preview()
        elif self._tool == 'polygon':
            self._update_poly_preview(tentative=pt)
        else:
            self._update_box_preview(pt)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pt = self._scene_pt(event)
        if self._tool == 'brush':
            if self._brush_pts:
                pts = list(self._brush_pts)
                self._brush_pts = []
                self._clear_brush_preview()
                self._drawing = False
                self.shape_committed.emit('brush', pts)
            return
        if self._tool == 'lasso':
            if len(self._poly_pts) >= 3:
                self._commit_polygon()
        elif self._tool != 'polygon':
            pts = [(self._start.x(), self._start.y()), (pt.x(), pt.y())]
            self._cancel()
            self.shape_committed.emit(self._tool, pts)

    def mouseDoubleClickEvent(self, event):
        if self._tool == 'polygon' and self._drawing:
            if len(self._poly_pts) >= 3:
                self._commit_polygon()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._tool == 'polygon' and self._drawing and len(self._poly_pts) >= 3:
                self._commit_polygon()
        elif event.key() == Qt.Key.Key_Escape:
            self._cancel()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        if self._tool == 'brush':
            delta = event.angleDelta().y()
            step = 1 if delta > 0 else -1
            self._brush_size = max(1, min(50, self._brush_size + step))
            # Notify widget to update spinbox
            parent = self.parent()
            while parent and not isinstance(parent, LabelEditorWidget):
                parent = parent.parent()
            if parent and hasattr(parent, '_brush_spin'):
                parent._brush_spin.setValue(self._brush_size)
            event.accept()
        else:
            super().wheelEvent(event)

    # ── brush stamping ────────────────────────────────────────────────────────

    def _stamp_brush(self, pt: QPointF):
        ix = int(pt.x())
        iy = int(pt.y())
        r = self._brush_size
        # Collect all pixels in the circle
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    px = ix + dx
                    py = iy + dy
                    if 0 <= px < self._img_w and 0 <= py < self._img_h:
                        coord = (px, py)
                        if coord not in self._brush_pts:
                            self._brush_pts.append(coord)
        # Draw preview circle
        pen = self._make_pen()
        scene = self.scene()
        r_scene = float(r)
        item = scene.addEllipse(
            ix - r_scene, iy - r_scene, r_scene * 2, r_scene * 2, pen)
        item.setZValue(10)
        self._brush_preview_items.append(item)
        # Limit preview items to avoid slowdown
        if len(self._brush_preview_items) > 100:
            old = self._brush_preview_items.pop(0)
            scene.removeItem(old)

    # ── preview drawing ───────────────────────────────────────────────────────

    def _update_box_preview(self, cur: QPointF):
        scene = self.scene()
        if self._preview:
            scene.removeItem(self._preview)
        pen = self._make_pen()
        x0 = min(self._start.x(), cur.x())
        y0 = min(self._start.y(), cur.y())
        x1 = max(self._start.x(), cur.x())
        y1 = max(self._start.y(), cur.y())
        r = QRectF(x0, y0, x1 - x0, y1 - y0)
        if self._tool == 'ellipse':
            self._preview = scene.addEllipse(r, pen)
        else:
            self._preview = scene.addRect(r, pen)
        self._preview.setZValue(10)

    def _update_poly_preview(self, tentative: QPointF | None = None):
        scene = self.scene()
        if self._poly_item:
            scene.removeItem(self._poly_item)
        pts = list(self._poly_pts)
        if tentative:
            pts.append(tentative)
        if len(pts) < 2:
            self._poly_item = None
            return
        path = QPainterPath(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        pen = self._make_pen()
        self._poly_item = scene.addPath(path, pen)
        self._poly_item.setZValue(10)

    def _commit_polygon(self):
        pts = [(p.x(), p.y()) for p in self._poly_pts]
        self._cancel()
        self.shape_committed.emit('polygon', pts)


class LabelEditorWidget(NodeBaseWidget):
    """
    Interactive multi-label editing widget with shape, brush, and flood-fill tools.

    Maintains an int32 label array where 0=background and 1+ are distinct labels.
    """

    label_changed = Signal()

    _OVERLAY_ALPHA = 0.50
    _MAX_UNDO = 30

    _set_input_signal = Signal(object, object)   # (label_arr, bg_pil) -> main thread

    def __init__(self, parent=None):
        super().__init__(parent, 'label_editor')

        root = QtWidgets.QVBoxLayout()
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(3)
        container = QtWidgets.QWidget()
        container.setLayout(root)

        # ── tool bar ─────────────────────────────────────────────────────────
        tb = QtWidgets.QHBoxLayout()
        tb.setSpacing(3)
        self._tool_group = QtWidgets.QButtonGroup(container)
        self._tool_names = ['rect', 'ellipse', 'polygon', 'lasso', 'brush', 'flood_fill']
        tool_labels = ['Rect', 'Ellipse', 'Polygon', 'Lasso', 'Brush', 'Fill']
        for i, label in enumerate(tool_labels):
            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(24)
            btn.setMinimumWidth(45)
            self._tool_group.addButton(btn, i)
            tb.addWidget(btn)
            if i == 0:
                btn.setChecked(True)
        tb.addSpacing(6)
        tb.addWidget(QtWidgets.QLabel('Op:'))
        self._op_combo = QtWidgets.QComboBox()
        self._op_combo.addItems(['Union', 'Subtract', 'Replace'])
        self._op_combo.setFixedHeight(24)
        self._op_combo.setMinimumWidth(55)
        tb.addWidget(self._op_combo)
        tb.addStretch()
        root.addLayout(tb)

        # ── brush size row ───────────────────────────────────────────────────
        br = QtWidgets.QHBoxLayout()
        br.setSpacing(3)
        _LS = 'color:#ccc; font-size:9px;'
        lbl_bs = QtWidgets.QLabel('Brush:')
        lbl_bs.setStyleSheet(_LS)
        self._brush_spin = QtWidgets.QSpinBox()
        self._brush_spin.setRange(1, 50)
        self._brush_spin.setValue(5)
        self._brush_spin.setFixedWidth(50)
        self._brush_spin.valueChanged.connect(
            lambda v: self._view.set_brush_size(v))
        br.addWidget(lbl_bs)
        br.addWidget(self._brush_spin)

        br.addSpacing(8)
        lbl_o = QtWidgets.QLabel('Opacity:')
        lbl_o.setStyleSheet(_LS)
        self._opacity_spin = QtWidgets.QSpinBox()
        self._opacity_spin.setRange(10, 100)
        self._opacity_spin.setValue(int(self._OVERLAY_ALPHA * 100))
        self._opacity_spin.setSuffix('%')
        self._opacity_spin.setFixedWidth(58)
        self._opacity_spin.valueChanged.connect(lambda _: self._render())
        br.addWidget(lbl_o)
        br.addWidget(self._opacity_spin)
        br.addStretch()
        root.addLayout(br)

        # ── canvas ───────────────────────────────────────────────────────────
        self._view = _LabelEditorView()
        self._view.setMinimumSize(400, 300)
        root.addWidget(self._view)

        # ── label list + controls ────────────────────────────────────────────
        label_row = QtWidgets.QHBoxLayout()
        label_row.setSpacing(3)

        self._label_list = QtWidgets.QListWidget()
        self._label_list.setFixedHeight(90)
        self._label_list.setMinimumWidth(200)
        self._label_list.setStyleSheet(
            'QListWidget { font-size: 11px; }'
            'QListWidget::item { padding: 3px 4px; }'
            'QListWidget::item:selected { background-color: #555; border: 2px solid #fff; }')
        self._label_list.currentRowChanged.connect(self._on_label_selected)
        label_row.addWidget(self._label_list)

        label_btn_col = QtWidgets.QVBoxLayout()
        label_btn_col.setSpacing(3)
        self._add_label_btn = QtWidgets.QPushButton('+ New Label')
        self._add_label_btn.setFixedHeight(22)
        self._add_label_btn.clicked.connect(self._add_label)
        label_btn_col.addWidget(self._add_label_btn)

        self._del_label_btn = QtWidgets.QPushButton('Delete')
        self._del_label_btn.setFixedHeight(22)
        self._del_label_btn.clicked.connect(self._delete_label)
        label_btn_col.addWidget(self._del_label_btn)

        self._cur_label_lbl = QtWidgets.QLabel('Label: 1')
        self._cur_label_lbl.setStyleSheet('color:#fff; font-size:11px; font-weight:bold;')
        label_btn_col.addWidget(self._cur_label_lbl)
        label_btn_col.addStretch()
        label_row.addLayout(label_btn_col)
        root.addLayout(label_row)

        # ── bottom bar ───────────────────────────────────────────────────────
        bb = QtWidgets.QHBoxLayout()
        bb.setSpacing(4)
        for label, slot in [('Undo', '_undo'), ('Reset', '_reset'), ('Clear', '_clear')]:
            btn = QtWidgets.QPushButton(label)
            btn.setFixedHeight(22)
            btn.setMinimumWidth(55)
            btn.clicked.connect(getattr(self, slot))
            bb.addWidget(btn)
        bb.addStretch()
        self._info_lbl = QtWidgets.QLabel('No labels')
        self._info_lbl.setStyleSheet('color:#aaa; font-size:10px;')
        bb.addWidget(self._info_lbl)
        root.addLayout(bb)

        self.set_custom_widget(container)

        # ── state ────────────────────────────────────────────────────────────
        self._label_arr: np.ndarray | None = None     # int32 H x W
        self._input_label_arr: np.ndarray | None = None  # original (for Reset)
        self._history: list[np.ndarray] = []
        self._img_w = 1
        self._img_h = 1
        self._bg_pil = None
        self._current_label = 1
        self._palette: list[tuple[int, int, int]] = []
        self._has_user_edits = False
        self._refresh_palette(1)

        # ── connections ──────────────────────────────────────────────────────
        self._tool_group.idClicked.connect(
            lambda i: self._view.set_tool(self._tool_names[i]))
        self._view.shape_committed.connect(self._on_shape_committed)
        self._set_input_signal.connect(
            self._apply_set_input, Qt.ConnectionType.QueuedConnection)

    def get_value(self):         return ''
    def set_value(self, _value): pass

    # ── palette ──────────────────────────────────────────────────────────────

    def _refresh_palette(self, n_labels: int):
        n = max(n_labels, 1)
        try:
            from .vision_nodes import _label_palette
        except ImportError:
            from vision_nodes import _label_palette
        self._palette = _label_palette(n)

    def _label_color(self, label: int) -> tuple[int, int, int]:
        if not self._palette:
            self._refresh_palette(label)
        return self._palette[(label - 1) % len(self._palette)]

    # ── public API ───────────────────────────────────────────────────────────

    def set_input(self, label_arr: np.ndarray | None, bg_pil=None):
        """Thread-safe: called from evaluate() which may run on worker thread."""
        if threading.current_thread() is not threading.main_thread():
            self._set_input_signal.emit(label_arr, bg_pil)
        else:
            self._apply_set_input(label_arr, bg_pil)

    def _apply_set_input(self, label_arr, bg_pil):
        label_changed = (
            label_arr is not None and (
                self._input_label_arr is None
                or label_arr.shape != self._input_label_arr.shape
                or not np.array_equal(label_arr, self._input_label_arr)
            )
        )
        bg_changed = False
        if bg_pil is not None:
            if self._bg_pil is None:
                bg_changed = True
            elif isinstance(bg_pil, np.ndarray) and isinstance(self._bg_pil, np.ndarray):
                bg_changed = bg_pil.shape != self._bg_pil.shape
            else:
                bg_changed = True

        if label_changed or bg_changed:
            if label_arr is not None:
                self._input_label_arr = label_arr.astype(np.int32)
                self._label_arr = self._input_label_arr.copy()
                self._img_h, self._img_w = label_arr.shape[:2]
            elif bg_pil is not None:
                h = bg_pil.shape[0] if isinstance(bg_pil, np.ndarray) else 512
                w = bg_pil.shape[1] if isinstance(bg_pil, np.ndarray) else 512
                self._input_label_arr = np.zeros((h, w), dtype=np.int32)
                self._label_arr = self._input_label_arr.copy()
                self._img_h, self._img_w = h, w
            self._history.clear()
            self._has_user_edits = False
        elif label_arr is None and self._label_arr is None:
            return
        self._bg_pil = bg_pil
        max_lbl = int(self._label_arr.max()) if self._label_arr is not None and self._label_arr.max() > 0 else 1
        self._refresh_palette(max_lbl)
        self._update_label_list()
        self._render()

    def get_labels(self) -> np.ndarray | None:
        return self._label_arr

    # ── label list management ────────────────────────────────────────────────

    def _update_label_list(self):
        self._label_list.blockSignals(True)
        self._label_list.clear()
        if self._label_arr is None:
            self._label_list.blockSignals(False)
            return
        unique = np.unique(self._label_arr)
        unique = unique[unique > 0]
        for lbl in unique:
            count = int(np.sum(self._label_arr == lbl))
            r, g, b = self._label_color(int(lbl))
            swatch = QPixmap(14, 14)
            swatch.fill(QColor(r, g, b))
            item = QtWidgets.QListWidgetItem(
                f'  Label {lbl}  ({count:,} px)')
            item.setIcon(QtGui.QIcon(swatch))
            item.setData(Qt.ItemDataRole.UserRole, int(lbl))
            self._label_list.addItem(item)
        # Select the current label row
        for i in range(self._label_list.count()):
            item = self._label_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == self._current_label:
                self._label_list.setCurrentRow(i)
                break
        self._label_list.blockSignals(False)

    def _on_label_selected(self, row):
        if row < 0:
            self._clear_highlight()
            return
        item = self._label_list.item(row)
        if item:
            self._current_label = item.data(Qt.ItemDataRole.UserRole)
            if self._palette:
                color = self._palette[(self._current_label - 1) % len(self._palette)]
                self._cur_label_lbl.setText(f'Label: {self._current_label}')
                self._cur_label_lbl.setStyleSheet(
                    f'color: rgb({color[0]},{color[1]},{color[2]}); '
                    f'font-size: 11px; font-weight: bold;')
            else:
                self._cur_label_lbl.setText(f'Label: {self._current_label}')
            self._highlight_label(self._current_label)

    def _clear_highlight(self):
        """Remove any existing highlight items from the canvas."""
        scene = self._view.scene()
        for item in getattr(self, '_highlight_items', []):
            try:
                scene.removeItem(item)
            except Exception:
                pass
        self._highlight_items = []

    def _highlight_label(self, label_id):
        """Draw a bounding box + brighter overlay around the selected label."""
        self._clear_highlight()
        if self._label_arr is None:
            return

        mask = self._label_arr == label_id
        if not mask.any():
            return

        # Find bounding box
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

        # Get label color
        if self._palette:
            color = self._palette[(label_id - 1) % len(self._palette)]
        else:
            color = (255, 255, 0)

        scene = self._view.scene()

        # Draw bounding box
        pen = QPen(QColor(*color))
        pen.setWidthF(2.0)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        pad = 3
        rect_item = scene.addRect(
            QRectF(x0 - pad, y0 - pad, (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad),
            pen)
        rect_item.setZValue(20)
        self._highlight_items.append(rect_item)

        # Draw label number near top-left of bbox
        text_item = scene.addSimpleText(str(label_id))
        text_item.setBrush(QBrush(QColor(*color)))
        font = text_item.font()
        font.setPixelSize(max(14, min(30, (y1 - y0) // 4)))
        font.setBold(True)
        text_item.setFont(font)
        text_item.setPos(x0 - pad, y0 - pad - font.pixelSize() - 2)
        text_item.setZValue(20)
        self._highlight_items.append(text_item)

    def _add_label(self):
        if self._label_arr is None:
            return
        max_label = int(self._label_arr.max()) if self._label_arr.max() > 0 else 0
        self._current_label = max_label + 1
        self._cur_label_lbl.setText(f'Label: {self._current_label}')
        self._refresh_palette(self._current_label)
        self._update_label_list()

    def _delete_label(self):
        if self._label_arr is None:
            return
        row = self._label_list.currentRow()
        if row < 0:
            return
        item = self._label_list.item(row)
        lbl = item.data(Qt.ItemDataRole.UserRole)
        self._push_history()
        self._label_arr[self._label_arr == lbl] = 0
        self._has_user_edits = True
        self._update_label_list()
        self._render()
        self.label_changed.emit()

    # ── shape operations ─────────────────────────────────────────────────────

    def _on_shape_committed(self, tool: str, pts: list):
        if self._label_arr is None:
            return
        if tool == 'flood_fill':
            self._do_flood_fill(pts[0])
            return
        if tool == 'brush':
            self._do_brush(pts)
            return
        if len(pts) < 2:
            return
        shape_mask = self._rasterize(tool, pts)
        if shape_mask is None:
            return
        self._push_history()
        op = self._op_combo.currentText()
        lbl = self._current_label
        if op == 'Union':
            self._label_arr[shape_mask] = lbl
        elif op == 'Subtract':
            self._label_arr[shape_mask & (self._label_arr == lbl)] = 0
        elif op == 'Replace':
            self._label_arr[shape_mask] = lbl
        self._has_user_edits = True
        max_lbl = int(self._label_arr.max()) if self._label_arr.max() > 0 else 1
        self._refresh_palette(max_lbl)
        self._update_label_list()
        self._render()
        self.label_changed.emit()

    def _do_brush(self, pts: list):
        if self._label_arr is None or not pts:
            return
        self._push_history()
        op = self._op_combo.currentText()
        lbl = self._current_label
        h, w = self._label_arr.shape
        for (px, py) in pts:
            if 0 <= px < w and 0 <= py < h:
                if op == 'Union' or op == 'Replace':
                    self._label_arr[py, px] = lbl
                elif op == 'Subtract':
                    if self._label_arr[py, px] == lbl:
                        self._label_arr[py, px] = 0
        self._has_user_edits = True
        max_lbl = int(self._label_arr.max()) if self._label_arr.max() > 0 else 1
        self._refresh_palette(max_lbl)
        self._update_label_list()
        self._render()
        self.label_changed.emit()

    def _do_flood_fill(self, seed: tuple):
        if self._label_arr is None:
            return
        x, y = seed
        h, w = self._label_arr.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        target_val = int(self._label_arr[y, x])
        fill_val = self._current_label
        if target_val == fill_val:
            return
        self._push_history()
        # BFS flood fill
        visited = np.zeros((h, w), dtype=bool)
        stack = [(x, y)]
        visited[y, x] = True
        filled = []
        while stack:
            cx, cy = stack.pop()
            filled.append((cx, cy))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    if int(self._label_arr[ny, nx]) == target_val:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
        op = self._op_combo.currentText()
        for (px, py) in filled:
            if op == 'Union' or op == 'Replace':
                self._label_arr[py, px] = fill_val
            elif op == 'Subtract':
                if self._label_arr[py, px] == fill_val:
                    self._label_arr[py, px] = 0
        self._has_user_edits = True
        max_lbl = int(self._label_arr.max()) if self._label_arr.max() > 0 else 1
        self._refresh_palette(max_lbl)
        self._update_label_list()
        self._render()
        self.label_changed.emit()

    def _rasterize(self, tool: str, pts: list) -> np.ndarray | None:
        from PIL import Image as _PIL, ImageDraw
        img = _PIL.new('L', (self._img_w, self._img_h), 0)
        draw = ImageDraw.Draw(img)
        if tool in ('rect', 'ellipse'):
            x0 = min(pts[0][0], pts[1][0])
            y0 = min(pts[0][1], pts[1][1])
            x1 = max(pts[0][0], pts[1][0])
            y1 = max(pts[0][1], pts[1][1])
            if x1 <= x0 or y1 <= y0:
                return None
            if tool == 'ellipse':
                draw.ellipse([x0, y0, x1, y1], fill=255)
            else:
                draw.rectangle([x0, y0, x1, y1], fill=255)
        elif tool in ('polygon', 'lasso'):
            if len(pts) < 3:
                return None
            draw.polygon([(float(x), float(y)) for x, y in pts], fill=255)
        return np.array(img) > 0

    # ── undo / reset / clear ─────────────────────────────────────────────────

    def _push_history(self):
        if self._label_arr is not None:
            self._history.append(self._label_arr.copy())
            if len(self._history) > self._MAX_UNDO:
                self._history.pop(0)

    def _undo(self):
        if self._history:
            self._label_arr = self._history.pop()
            max_lbl = int(self._label_arr.max()) if self._label_arr.max() > 0 else 1
            self._refresh_palette(max_lbl)
            self._update_label_list()
            self._render()
            self.label_changed.emit()

    def _reset(self):
        if self._input_label_arr is not None:
            self._push_history()
            self._label_arr = self._input_label_arr.copy()
            self._has_user_edits = False
            max_lbl = int(self._label_arr.max()) if self._label_arr.max() > 0 else 1
            self._refresh_palette(max_lbl)
            self._update_label_list()
            self._render()
            self.label_changed.emit()

    def _clear(self):
        if self._label_arr is not None:
            self._push_history()
            self._label_arr = np.zeros((self._img_h, self._img_w), dtype=np.int32)
            self._has_user_edits = True
            self._update_label_list()
            self._render()
            self.label_changed.emit()

    # ── rendering ────────────────────────────────────────────────────────────

    def _render(self):
        if self._label_arr is None:
            return
        h, w = self._label_arr.shape
        # Background
        if self._bg_pil is not None:
            bg = _ensure_display_rgb(self._bg_pil)
            bg_h, bg_w = bg.shape[:2]
            if bg_w != w or bg_h != h:
                from PIL import Image as _PIL
                bg = np.array(_PIL.fromarray(bg).resize((w, h)))
        else:
            bg = np.full((h, w, 3), 30, dtype=np.uint8)

        result = bg.astype(np.float32)
        alpha = self._opacity_spin.value() / 100.0

        unique = np.unique(self._label_arr)
        unique = unique[unique > 0]
        for lbl in unique:
            r, g, b = self._label_color(int(lbl))
            mask = self._label_arr == lbl
            result[mask, 0] = result[mask, 0] * (1 - alpha) + r * alpha
            result[mask, 1] = result[mask, 1] * (1 - alpha) + g * alpha
            result[mask, 2] = result[mask, 2] * (1 - alpha) + b * alpha

        result = result.clip(0, 255).astype(np.uint8)
        qimg = QImage(result.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        self._view.set_pixmap(QPixmap.fromImage(qimg))

        # Info
        total = int(np.sum(self._label_arr > 0))
        n_labels = len(unique)
        self._info_lbl.setText(f'{n_labels} label(s), {total:,} px')


class LabelEditorNode(BaseExecutionNode):
    """
    Interactively edits a multi-label image by drawing shapes and applying operations.

    Supports rect, ellipse, polygon, lasso, brush, and flood-fill tools.
    Each label is assigned a distinct color. Outputs the label array and a summary table.
    """

    __identifier__ = 'nodes.image_process.Visualize'
    NODE_NAME = 'Label Editor'
    PORT_SPEC = {'inputs': ['image', 'label_image'], 'outputs': ['label_image', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('image', color=PORT_COLORS['image'])
        self.add_input('label_image', color=PORT_COLORS.get('label', (160, 220, 40)))
        self.add_output('label_image', color=PORT_COLORS.get('label', (160, 220, 40)),
                        multi_output=True)
        self.add_output('table', color=PORT_COLORS['table'], multi_output=True)

        self._editor = LabelEditorWidget(self.view)
        self._editor.label_changed.connect(self._on_label_changed)
        self.add_custom_widget(self._editor)

    def _on_label_changed(self):
        """Called when user edits labels; push result downstream."""
        import pandas as pd
        result = self._editor.get_labels()
        if result is not None:
            self.output_values['label_image'] = LabelData(payload=result.astype(np.int32))
            # Build summary table
            unique = np.unique(result)
            unique = unique[unique > 0]
            rows = []
            for lbl in unique:
                rows.append({'label': int(lbl), 'area': int(np.sum(result == lbl))})
            if rows:
                self.output_values['table'] = TableData(payload=pd.DataFrame(rows))
            else:
                self.output_values['table'] = TableData(
                    payload=pd.DataFrame(columns=['label', 'area']))
        else:
            self.output_values['label_image'] = LabelData(
                payload=np.zeros((1, 1), dtype=np.int32))
            self.output_values['table'] = TableData(
                payload=pd.DataFrame(columns=['label', 'area']))
        self.mark_dirty()

    def evaluate(self):
        import pandas as pd
        self.reset_progress()

        # ── optional background image ─────────────────────────────────────
        bg_arr = None
        img_port = self.inputs().get('image')
        if img_port and img_port.connected_ports():
            cp = img_port.connected_ports()[0]
            data = cp.node().output_values.get(cp.name())
            if isinstance(data, ImageData):
                bg_arr = data.payload

        # ── label_image input ───────────────────────────────────────────────
        label_arr = None
        label_port = self.inputs().get('label_image')
        if label_port and label_port.connected_ports():
            cp = label_port.connected_ports()[0]
            data = cp.node().output_values.get(cp.name())
            if isinstance(data, LabelData):
                label_arr = data.payload.astype(np.int32)

        if label_arr is None and bg_arr is None:
            return False, 'Connect a label image or background image'

        # If only image is connected, start with a blank label array
        if label_arr is None:
            h, w = bg_arr.shape[:2]
            label_arr = np.zeros((h, w), dtype=np.int32)

        # Preserve user edits if input hasn't changed
        prev_input = getattr(self._editor, '_input_label_arr', None)
        edited = self._editor._label_arr
        input_unchanged = (
            prev_input is not None
            and label_arr.shape == prev_input.shape
            and np.array_equal(label_arr, prev_input)
        )
        has_edits = (
            input_unchanged
            and edited is not None
            and not np.array_equal(edited, prev_input)
        )

        self._editor.set_input(label_arr, bg_arr)
        self.set_progress(50)

        # ── output ───────────────────────────────────────────────────────
        if has_edits:
            out_labels = edited.astype(np.int32)
        else:
            out_labels = label_arr.astype(np.int32)

        self.output_values['label_image'] = LabelData(payload=out_labels)

        # Build summary table
        unique = np.unique(out_labels)
        unique = unique[unique > 0]
        rows = []
        for lbl in unique:
            rows.append({'label': int(lbl), 'area': int(np.sum(out_labels == lbl))})
        if rows:
            self.output_values['table'] = TableData(payload=pd.DataFrame(rows))
        else:
            self.output_values['table'] = TableData(
                payload=pd.DataFrame(columns=['label', 'area']))

        self.set_progress(100)
        return True, None
