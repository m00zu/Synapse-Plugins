"""
plugins/figure_plotting/svg_editor_node.py
==========================================
Interactive SVG editor node — extracted from nodes/display_nodes.py.
Allows element-level editing of matplotlib figures rendered as SVG.
"""
import re
import xml.etree.ElementTree as _ET
import io as _io

from data_models import FigureData
from nodes.base import BaseExecutionNode, PORT_COLORS

# ─── Inline SVG Editor ────────────────────────────────────────────────────────
# All SVG editor classes are prefixed with _ to indicate they are internal helpers
# used only by SvgEditorNode.

_SVG_NS  = 'http://www.w3.org/2000/svg'
_XLINK_NS = 'http://www.w3.org/1999/xlink'


def _ltag(elem):
    """Return local tag name without namespace prefix."""
    t = elem.tag
    return t.split('}', 1)[1] if '}' in t else t


def _css_parse(s: str) -> dict:
    """Parse 'fill: #abc; stroke: none' into a dict."""
    d = {}
    for part in s.split(';'):
        if ':' in part:
            k, v = part.split(':', 1)
            d[k.strip()] = v.strip()
    return d


def _css_build(d: dict) -> str:
    """Rebuild a CSS style string from a dict."""
    return '; '.join(f'{k}: {v}' for k, v in d.items() if v.strip())


def _find_id(root, eid: str):
    """Find first element in tree with the given id attribute."""
    for e in root.iter():
        if e.get('id') == eid:
            return e
    return None


def _elem_texts(elem) -> str:
    """Collect visible text from an element and its descendants."""
    parts = [s for e in elem.iter() if (s := (e.text or '').strip())]
    return ' '.join(parts)


_TEXT_ID_RE = re.compile(r'^text_\d+$')

def _has_text_descendant(elem) -> bool:
    """True if elem is/contains a <text> element, or is a matplotlib text group (id=text_N)."""
    if _ltag(elem) == 'text':
        return True
    if _TEXT_ID_RE.match(elem.get('id', '')):
        return True
    return any(_ltag(e) == 'text' for e in elem.iter())


def _get_mpl_text_scale(elem):
    """Extract the font scale factor from a matplotlib text group.

    Matplotlib encodes font size as ``scale(S -S)`` on an inner ``<g>``
    transform — NOT via CSS ``font-size``.  Returns ``(scale_float, inner_g_elem)``
    or ``(None, None)`` if the pattern is not found.
    """
    for ch in elem.iter():
        if ch is elem:
            continue
        t = ch.get('transform', '')
        m = re.search(r'scale\(([\d.]+)\s+(-?[\d.]+)\)', t)
        if m:
            return float(m.group(1)), ch
    return None, None


def _is_draggable(elem, tick_text_ids=frozenset()) -> bool:
    """Determine if an SVG overlay should be movable (draggable).

    Returns True for:
      - text groups  (text_N, <text>) — EXCEPT tick labels
      - stat-annotation text  (stat_text:*)
      - user-added annotations  (annotation_*)

    Tick labels, data elements (line2d, PathCollection, etc.) are NOT
    draggable.
    """
    eid = elem.get('id', '')
    # Tick labels must stay locked to the axes
    if eid in tick_text_ids:
        return False
    # User-added annotations are always draggable
    if any(eid.startswith(p) for p in
           ('annotation_', 'rect_ann_', 'ellipse_ann_',
            'line_ann_', 'arrow_ann_')):
        return True
    if _has_text_descendant(elem):
        return True
    if eid.startswith('stat_text:'):
        return True
    return False


# ── Marker shape SVG path data (normalised to ~±3.5 unit coords) ──────────
# These match matplotlib's default marker size coordinate system.
_MARKER_PATHS = {
    'circle': ('M 0 -3.5 C 1.933 -3.5 3.5 -1.933 3.5 0 '
               'C 3.5 1.933 1.933 3.5 0 3.5 '
               'C -1.933 3.5 -3.5 1.933 -3.5 0 '
               'C -3.5 -1.933 -1.933 -3.5 0 -3.5 Z'),
    'square': 'M -3 -3 L 3 -3 L 3 3 L -3 3 Z',
    'triangle up': 'M 0 -3.5 L 3.031 1.75 L -3.031 1.75 Z',
    'triangle down': 'M 0 3.5 L 3.031 -1.75 L -3.031 -1.75 Z',
    'diamond': 'M 0 -3.5 L 3.5 0 L 0 3.5 L -3.5 0 Z',
    'star': ('M 0 -3.5 L 0.815 -1.082 L 3.329 -1.082 '
             'L 1.257 0.413 L 2.072 2.832 L 0 1.337 '
             'L -2.072 2.832 L -1.257 0.413 '
             'L -3.329 -1.082 L -0.815 -1.082 Z'),
    'plus': ('M -1 -3 L 1 -3 L 1 -1 L 3 -1 L 3 1 '
             'L 1 1 L 1 3 L -1 3 L -1 1 L -3 1 '
             'L -3 -1 L -1 -1 Z'),
    'x': ('M -2.475 -3.182 L 0 -0.707 L 2.475 -3.182 '
           'L 3.182 -2.475 L 0.707 0 L 3.182 2.475 '
           'L 2.475 3.182 L 0 0.707 L -2.475 3.182 '
           'L -3.182 2.475 L -0.707 0 L -3.182 -2.475 Z'),
}

_MARKER_NAMES = list(_MARKER_PATHS.keys())


def _change_marker_shape_for_group(group_elem, new_d, svg_root):
    """Change marker shape for one scatter group without affecting others.

    If the <defs> path is referenced by other groups (shared),
    clone it with a unique ID and redirect this group's <use>
    elements to the clone.
    """
    if svg_root is None:
        return

    # 1. Find what path ID this group's <use> elements reference
    use_elems = [ue for ue in group_elem.iter() if _ltag(ue) == 'use']
    if not use_elems:
        # Fallback: try inline <defs> within the group
        for de in group_elem.iter():
            if _ltag(de) == 'defs':
                for pe in de:
                    if _ltag(pe) == 'path':
                        pe.set('d', new_d)
        return

    # Get the href from the first <use>
    href = None
    for attr_name in [f'{{{_XLINK_NS}}}href', 'href']:
        href = use_elems[0].get(attr_name)
        if href:
            break
    if not href or not href.startswith('#'):
        return
    old_path_id = href[1:]

    # 2. Find the referenced <path> element
    path_elem = None
    for el in svg_root.iter():
        if _ltag(el) == 'path' and el.get('id') == old_path_id:
            path_elem = el
            break
    if path_elem is None:
        return

    # 3. Check if other groups also reference this path
    all_uses = [ue for ue in svg_root.iter() if _ltag(ue) == 'use']
    other_refs = False
    for ue in all_uses:
        if ue in use_elems:
            continue
        for attr_name in [f'{{{_XLINK_NS}}}href', 'href']:
            if ue.get(attr_name) == f'#{old_path_id}':
                other_refs = True
                break
        if other_refs:
            break

    if not other_refs:
        # No other group uses this path — safe to modify in place
        path_elem.set('d', new_d)
    else:
        # Shared path — clone with a new ID
        import copy as _copy
        new_path = _copy.deepcopy(path_elem)
        # Generate unique ID
        counter = 1
        while True:
            new_id = f'{old_path_id}_{counter}'
            if svg_root.find(f'.//*[@id="{new_id}"]') is None:
                break
            counter += 1
        new_path.set('id', new_id)
        new_path.set('d', new_d)

        # Insert clone next to original
        parent = None
        for candidate in svg_root.iter():
            if path_elem in list(candidate):
                parent = candidate
                break
        if parent is not None:
            idx = list(parent).index(path_elem)
            parent.insert(idx + 1, new_path)
        else:
            defs_el = svg_root.find(f'{{{_SVG_NS}}}defs')
            if defs_el is None:
                import xml.etree.ElementTree as _ET_local
                defs_el = _ET_local.SubElement(svg_root, f'{{{_SVG_NS}}}defs')
                svg_root.insert(0, defs_el)
            defs_el.append(new_path)

        # Redirect this group's <use> elements to the clone
        for ue in use_elems:
            for attr_name in [f'{{{_XLINK_NS}}}href', 'href']:
                if ue.get(attr_name):
                    ue.set(attr_name, f'#{new_id}')


def _detect_marker_shape(elem):
    """Detect current marker shape from a PathCollection group's <defs>."""
    for ch in elem.iter():
        if _ltag(ch) == 'defs':
            for p in ch:
                if _ltag(p) == 'path':
                    d = p.get('d', '')
                    # Try matching against known shapes
                    d_norm = ' '.join(d.split())
                    for name, path_d in _MARKER_PATHS.items():
                        if ' '.join(path_d.split()) == d_norm:
                            return name
                    return 'circle'  # unknown shape, default label
    return 'circle'


def _collect_rendered_positions(elem):
    """
    Collect (x, y) positions from rendered descendants, skipping <defs>.
    Used to compute manual bounds when QSvgRenderer.boundsOnElement()
    is unreliable (groups with inline <defs> whose paths sit near the
    SVG origin, inflating the bounding box).
    """
    positions = []
    def _walk(el):
        for child in el:
            if _ltag(child) == 'defs':
                continue
            tag = _ltag(child)
            if tag == 'use':
                try:
                    positions.append((float(child.get('x', '0')),
                                      float(child.get('y', '0'))))
                except (ValueError, TypeError):
                    pass
            elif tag == 'path':
                d = child.get('d', '')
                for m in re.finditer(
                        r'[ML]\s*([-\d.e+]+)\s+([-\d.e+]+)', d):
                    try:
                        positions.append((float(m.group(1)),
                                          float(m.group(2))))
                    except ValueError:
                        pass
            elif tag == 'line':
                try:
                    positions.append((float(child.get('x1', '0')),
                                      float(child.get('y1', '0'))))
                    positions.append((float(child.get('x2', '0')),
                                      float(child.get('y2', '0'))))
                except (ValueError, TypeError):
                    pass
            _walk(child)
    _walk(elem)
    return positions


# ── Selectable overlay item ───────────────────────────────────────────────────

class _SvgOverlay(object):
    """
    Mixed into QGraphicsRectItem to make an SVG element selectable.
    Transparent normally; highlights on hover/selection.
    Optionally movable (for text groups).
    """
    pass  # see _make_svg_overlay() factory below


def _make_svg_overlay(eid: str, tag: str, rect, sx: float, sy: float,
                       movable: bool = False):
    """
    Factory that returns a configured QGraphicsRectItem overlay for one SVG element.
    Uses a local import to defer heavy PySide6 imports until first use.
    """
    from PySide6 import QtWidgets, QtCore, QtGui

    NO_PEN  = QtGui.QPen(QtCore.Qt.PenStyle.NoPen)
    NO_BR   = QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush)
    HV_PEN  = QtGui.QPen(QtGui.QColor(255, 210, 60, 180), 1.0,
                          QtCore.Qt.PenStyle.DashLine)
    SEL_PEN = QtGui.QPen(QtGui.QColor(40, 200, 255, 230), 1.0,
                          QtCore.Qt.PenStyle.DashLine)

    class _Overlay(QtWidgets.QGraphicsRectItem):
        def __init__(self):
            super().__init__(rect)
            self.eid     = eid
            self.etag    = tag
            self._sx     = sx
            self._sy     = sy
            self._dbl_cb = None   # callable(eid)
            self._mov_cb = None   # callable(eid, dx_svg, dy_svg)

            self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            self.setAcceptHoverEvents(True)
            self.setZValue(10)
            self.setToolTip(f'<{tag}>\nid: "{eid}"')
            self.setPen(NO_PEN)
            self.setBrush(NO_BR)

            if movable:
                self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
                self.setFlag(
                    QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
                self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        # ── hover ─────────────────────────────────────────────────────────
        def hoverEnterEvent(self, e):
            if not self.isSelected():
                self.setPen(HV_PEN)
            super().hoverEnterEvent(e)

        def hoverLeaveEvent(self, e):
            if not self.isSelected():
                self.setPen(NO_PEN)
            super().hoverLeaveEvent(e)

        # ── selection ─────────────────────────────────────────────────────
        def itemChange(self, change, value):
            if change == (QtWidgets.QGraphicsItem
                          .GraphicsItemChange.ItemSelectedChange):
                if value:
                    self.setPen(SEL_PEN)
                else:
                    self.setPen(NO_PEN)
            return super().itemChange(change, value)

        # ── drag (movable items only) ──────────────────────────────────────
        def mouseReleaseEvent(self, e):
            pos_before = QtCore.QPointF(self.pos())
            super().mouseReleaseEvent(e)
            delta = self.pos()   # pos() starts at (0,0); delta = drag amount
            if (self._mov_cb and
                    (abs(delta.x()) > 0.5 or abs(delta.y()) > 0.5)):
                dx_svg = delta.x() / self._sx
                dy_svg = delta.y() / self._sy
                self.setPos(0.0, 0.0)   # reset – rerender will reposition
                self._mov_cb(self.eid, dx_svg, dy_svg)

        # ── double-click → properties panel ───────────────────────────────
        def mouseDoubleClickEvent(self, e):
            if self._dbl_cb:
                self._dbl_cb(self.eid)
            super().mouseDoubleClickEvent(e)

    return _Overlay()


def _build_qpath_from_elem(elem, sx, sy, vx, vy):
    """Build a QPainterPath in scene coordinates from an SVG element's
    rendered <path> children (skipping <defs>)."""
    from PySide6 import QtGui
    qp = QtGui.QPainterPath()
    _skip_defs = set()
    for de in elem.iter():
        if _ltag(de) == 'defs':
            for dd in de.iter():
                _skip_defs.add(dd)
    for ch in elem.iter():
        if ch in _skip_defs:
            continue
        d = ch.get('d', '') if _ltag(ch) == 'path' else ''
        if not d:
            continue
        tokens = re.findall(
            r'[a-zA-Z]+|-?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?', d)
        idx, cmd = 0, ''
        try:
            while idx < len(tokens):
                t = tokens[idx]
                if t.isalpha():
                    cmd = t
                    idx += 1
                if cmd in ('M', 'L'):
                    x = (float(tokens[idx]) - vx) * sx
                    y = (float(tokens[idx + 1]) - vy) * sy
                    (qp.moveTo if cmd == 'M' else qp.lineTo)(x, y)
                    idx += 2
                elif cmd == 'C' and idx + 5 < len(tokens):
                    coords = [float(tokens[idx + j]) for j in range(6)]
                    qp.cubicTo(
                        (coords[0] - vx) * sx, (coords[1] - vy) * sy,
                        (coords[2] - vx) * sx, (coords[3] - vy) * sy,
                        (coords[4] - vx) * sx, (coords[5] - vy) * sy)
                    idx += 6
                elif cmd in ('Z', 'z'):
                    qp.closeSubpath()
                else:
                    idx += 1
        except (ValueError, IndexError):
            pass
    return qp


def _make_svg_path_overlay(eid: str, tag: str, qpath, sx: float, sy: float,
                            movable: bool = False):
    """Factory: QGraphicsPathItem overlay that follows the actual SVG path
    outline — used for line-like elements instead of a bounding rect."""
    from PySide6 import QtWidgets, QtCore, QtGui

    NO_PEN  = QtGui.QPen(QtCore.Qt.PenStyle.NoPen)
    HV_PEN  = QtGui.QPen(QtGui.QColor(255, 210, 60, 200), 1.5,
                          QtCore.Qt.PenStyle.DashLine)
    SEL_PEN = QtGui.QPen(QtGui.QColor(40, 200, 255, 230), 1.5,
                          QtCore.Qt.PenStyle.DashLine)
    HIT_W   = 10.0   # wide transparent stroke for click hit-testing

    class _PathOv(QtWidgets.QGraphicsPathItem):
        def __init__(self):
            super().__init__(qpath)
            self.eid     = eid
            self.etag    = tag
            self._sx     = sx
            self._sy     = sy
            self._dbl_cb = None
            self._mov_cb = None

            self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            self.setAcceptHoverEvents(True)
            self.setZValue(10)
            self.setToolTip(f'<{tag}>\nid: "{eid}"')
            self.setPen(NO_PEN)
            self.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))

            if movable:
                self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
                self.setFlag(
                    QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
                self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        def shape(self):
            """Widen the clickable area around the thin path."""
            stroker = QtGui.QPainterPathStroker()
            stroker.setWidth(HIT_W)
            return stroker.createStroke(self.path())

        def boundingRect(self):
            return self.shape().boundingRect()

        # ── hover ─────────────────────────────────────────────────────────
        def hoverEnterEvent(self, e):
            if not self.isSelected():
                self.setPen(HV_PEN)
            super().hoverEnterEvent(e)

        def hoverLeaveEvent(self, e):
            if not self.isSelected():
                self.setPen(NO_PEN)
            super().hoverLeaveEvent(e)

        # ── selection ─────────────────────────────────────────────────────
        def itemChange(self, change, value):
            if change == (QtWidgets.QGraphicsItem
                          .GraphicsItemChange.ItemSelectedChange):
                self.setPen(SEL_PEN if value else NO_PEN)
            return super().itemChange(change, value)

        # ── drag ──────────────────────────────────────────────────────────
        def mouseReleaseEvent(self, e):
            super().mouseReleaseEvent(e)
            delta = self.pos()
            if (self._mov_cb and
                    (abs(delta.x()) > 0.5 or abs(delta.y()) > 0.5)):
                dx_svg = delta.x() / self._sx
                dy_svg = delta.y() / self._sy
                self.setPos(0.0, 0.0)
                self._mov_cb(self.eid, dx_svg, dy_svg)

        def mouseDoubleClickEvent(self, e):
            if self._dbl_cb:
                self._dbl_cb(self.eid)
            super().mouseDoubleClickEvent(e)

    return _PathOv()


def _is_line_like(elem):
    """True if the SVG element contains only stroke-only paths (fill: none).
    Skips <defs> subtrees since those are shared definitions, not rendered content.
    """
    found_path = False
    def _walk(el):
        nonlocal found_path
        for ch in el:
            if _ltag(ch) == 'defs':
                continue
            if _ltag(ch) == 'path':
                st = _css_parse(ch.get('style', ''))
                fill = st.get('fill', ch.get('fill', ''))
                if fill not in ('none', ''):
                    return False
                found_path = True
            if _ltag(ch) == 'use':
                # <use> references are rendered content — not line-like
                return False
            result = _walk(ch)
            if result is False:
                return False
        return None
    _walk(elem)
    return found_path


# ── Graphics view ─────────────────────────────────────────────────────────────

class _SvgEditorView(object):
    """
    QGraphicsView subclass that renders an SVG as a background pixmap and
    overlays transparent selectable items for each identified element.

    Signals
    -------
    element_selected(str)           – emitted when the user clicks an element
    element_double_clicked(str)     – emitted on double-click
    element_moved(str, float, float)– emitted when a text group is dragged
                                      args: (element_id, dx_svg, dy_svg)
    """

    # Skip these SVG tag types when building overlays
    _SKIP_TAGS = frozenset({
        'defs', 'metadata', 'style', 'clipPath', 'symbol',
        'linearGradient', 'radialGradient', 'pattern', 'filter',
        'mask', 'title', 'desc', 'script',
    })

    def __new__(cls):
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtSvg import QSvgRenderer

        class _View(QtWidgets.QGraphicsView):
            element_selected       = QtCore.Signal(str)
            element_double_clicked = QtCore.Signal(str)
            element_moved          = QtCore.Signal(str, float, float)
            delete_requested       = QtCore.Signal(str)   # eid

            _SKIP   = _SvgEditorView._SKIP_TAGS

            def __init__(self):
                super().__init__()
                from PySide6.QtSvgWidgets import QGraphicsSvgItem
                self._sc = QtWidgets.QGraphicsScene(self)
                self.setScene(self._sc)
                self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
                self.setTransformationAnchor(
                    QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
                self.setHorizontalScrollBarPolicy(
                    QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
                self.setVerticalScrollBarPolicy(
                    QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
                self.setStyleSheet(
                    "background: #1c1c1c; border: 1px solid #333;")
                self.setMinimumSize(540, 440)

                self._renderer = None
                self._bg       = None      # QGraphicsSvgItem (vector)
                self._overlays = {}        # eid → overlay item
                self.svg_root  = None      # ET.Element
                self._tree     = None      # ET.ElementTree
                self._sx = self._sy = 1.0
                self._vbox    = QtCore.QRectF()
                self._pan_pos = None       # middle-button pan

                self._sc.selectionChanged.connect(self._on_sel_changed)

            # ── public ────────────────────────────────────────────────────
            def load_svg(self, svg_bytes: bytes):
                from PySide6.QtSvgWidgets import QGraphicsSvgItem
                self._sc.clear()
                self._overlays.clear()
                self._renderer = self._bg = None
                self.svg_root  = self._tree = None

                _ET.register_namespace('', _SVG_NS)
                _ET.register_namespace('xlink', _XLINK_NS)
                try:
                    self._tree    = _ET.ElementTree(_ET.fromstring(svg_bytes))
                    self.svg_root = self._tree.getroot()
                except Exception as ex:
                    print(f'SvgEditorView: XML parse error: {ex}')
                    return

                from PySide6.QtSvg import QSvgRenderer
                self._renderer = QSvgRenderer(svg_bytes)
                if not self._renderer.isValid():
                    print('SvgEditorView: renderer invalid')
                    return

                self._vbox = self._renderer.viewBoxF()
                # Scene coordinates = viewBox coordinates (1:1 mapping)
                self._sx = 1.0
                self._sy = 1.0

                self._bg = QGraphicsSvgItem()
                self._bg.setSharedRenderer(self._renderer)
                self._bg.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.NoCache)
                self._bg.setZValue(0)
                self._sc.addItem(self._bg)
                self._create_overlays()
                self.fitInView(
                    self._sc.sceneRect(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio)

            def rerender(self, svg_bytes: bytes, keep_eid: str = None):
                """Re-render background and recreate overlays from updated SVG."""
                from PySide6.QtSvg import QSvgRenderer
                _ET.register_namespace('', _SVG_NS)
                _ET.register_namespace('xlink', _XLINK_NS)
                r = QSvgRenderer(svg_bytes)
                if not r.isValid():
                    return
                self._renderer = r
                self._vbox = r.viewBoxF()
                self._sx = 1.0
                self._sy = 1.0

                # Update the vector background
                if self._bg:
                    self._bg.setSharedRenderer(r)

                # Recreate overlays
                for ov in list(self._overlays.values()):
                    self._sc.removeItem(ov)
                self._overlays.clear()
                self._create_overlays()

                # Restore selection
                if keep_eid and keep_eid in self._overlays:
                    self._overlays[keep_eid].setSelected(True)

            def overlay_count(self):
                return len(self._overlays)

            def zoom_in(self):   self.scale(1.25, 1.25)
            def zoom_out(self):  self.scale(0.80, 0.80)
            def zoom_fit(self):
                if self._sc.sceneRect().isValid():
                    self.fitInView(
                        self._sc.sceneRect(),
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio)

            # ── private ───────────────────────────────────────────────────
            def _create_overlays(self):
                if not self.svg_root or not self._renderer:
                    return
                vx, vy = self._vbox.x(), self._vbox.y()
                vw, vh = self._vbox.width(), self._vbox.height()
                sx, sy = self._sx, self._sy
                MIN_A   = 9.0    # min area in scene px² — skip invisible/tiny items
                MAX_COV = 0.90   # skip elements covering >70% of SVG area (containers)
                total_svg_area = vw * vh if (vw > 0 and vh > 0) else 1.0

                # Build set of IDs that are inside <defs>/<clipPath>/<symbol>.
                # These are definition elements, not rendered directly.
                _DEF_CONTAINERS = {'defs', 'clipPath', 'symbol'}
                defs_ids: set = set()
                for container in self.svg_root.iter():
                    if _ltag(container) in _DEF_CONTAINERS:
                        for desc in container.iter():
                            eid_d = desc.get('id')
                            if eid_d:
                                defs_ids.add(eid_d)

                # Build set of text IDs inside tick groups — these must
                # NOT be draggable (axis tick labels stay locked).
                _TICK_RE = re.compile(r'^(xtick|ytick)_\d+$')
                tick_text_ids: set = set()
                for _te in self.svg_root.iter():
                    _tid = _te.get('id', '')
                    if _TICK_RE.match(_tid):
                        for _td in _te.iter():
                            _tdid = _td.get('id', '')
                            if _tdid and _has_text_descendant(_td):
                                tick_text_ids.add(_tdid)

                # Structural containers — no overlay, children handled individually
                _STRUCTURAL_RE = re.compile(
                    r'^(figure|axes|subplot|inset_axes|polar_axes'
                    r'|xtick|ytick|matplotlib\.axis)_\d+$')

                # Helper: check if a <g> wraps a purely white-fill (#ffffff)
                # background path. These are figure/axes bg rects and should
                # not be interactive overlays.
                def _is_white_bg_patch(el):
                    if _ltag(el) != 'g':
                        return False
                    for ch in el:
                        if _ltag(ch) == 'path':
                            st = _css_parse(ch.get('style', ''))
                            f = st.get('fill', ch.get('fill', ''))
                            if f in ('#ffffff', '#fff', 'white'):
                                return True
                    return False

                # Semantic groups — get ONE overlay; non-semantic children
                # are suppressed to avoid overlapping "blocks".
                _SEMANTIC_RE = re.compile(
                    r'^(line2d|PathCollection|QuadMesh|PolyCollection'
                    r'|FillBetweenPolyCollection'
                    r'|mcoll|AxesImage|FancyBboxPatch|legend'
                    r'|text)_\d+$')

                # Identify semantic-group roots and their non-root children
                semantic_roots = set()
                for elem in self.svg_root.iter():
                    eid = elem.get('id', '')
                    if _SEMANTIC_RE.match(eid):
                        semantic_roots.add(eid)

                semantic_child_ids = set()
                for elem in self.svg_root.iter():
                    eid = elem.get('id', '')
                    if eid in semantic_roots:
                        for desc in elem.iter():
                            did = desc.get('id')
                            if did and did != eid and did not in semantic_roots:
                                semantic_child_ids.add(did)

                # Collect overlay candidates
                candidates = []
                for elem in self.svg_root.iter():
                    tag = _ltag(elem)
                    eid = elem.get('id')
                    if not eid or tag in self._SKIP:
                        continue
                    if eid in defs_ids:
                        continue
                    if _STRUCTURAL_RE.match(eid):
                        continue
                    if eid in semantic_child_ids:
                        continue  # parent semantic group handles this
                    if _is_white_bg_patch(elem):
                        continue  # figure/axes white background rect

                    # Groups with inline <defs> confuse QSvgRenderer
                    # — the defs paths near SVG origin inflate the box.
                    # Compute bounds manually from rendered children.
                    _has_defs = any(
                        _ltag(c) == 'defs' for c in elem)
                    if _has_defs:
                        pts = _collect_rendered_positions(elem)
                        if pts:
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            _P = 4.0
                            b = QtCore.QRectF(
                                min(xs) - _P, min(ys) - _P,
                                max(xs) - min(xs) + 2 * _P,
                                max(ys) - min(ys) + 2 * _P)
                        else:
                            b = self._renderer.boundsOnElement(eid)
                    else:
                        b = self._renderer.boundsOnElement(eid)
                    if b.isEmpty():
                        continue

                    # Detect line-like elements BEFORE coverage filter so
                    # they are not rejected — a regression line spanning
                    # the full axes has a large bounding rect but uses
                    # QPainterPathStroker for hit-testing (thin clickable
                    # strip), so it won't block smaller elements.
                    line_like = _is_line_like(elem)

                    coverage = (b.width() * b.height()) / total_svg_area
                    if coverage > MAX_COV and not line_like:
                        continue

                    sr = QtCore.QRectF(
                        (b.x() - vx) * sx, (b.y() - vy) * sy,
                        b.width() * sx,     b.height() * sy)
                    area = sr.width() * sr.height()
                    if area < MIN_A:
                        continue

                    movable = _is_draggable(elem, tick_text_ids)
                    candidates.append(
                        (eid, tag, sr, area, movable, line_like, elem))

                # Sort by area DESCENDING so larger items get lower z-values.
                # Smaller elements (text, markers) end up on top and are always
                # clickable — this prevents large overlays from blocking them.
                candidates.sort(key=lambda c: c[3], reverse=True)

                for i, (eid, tag, sr, area, movable, line_like, elem) \
                        in enumerate(candidates):
                    if line_like:
                        qp = _build_qpath_from_elem(
                            elem, sx, sy, vx, vy)
                        if qp.isEmpty():
                            ov = _make_svg_overlay(
                                eid, tag, sr, sx, sy,
                                movable=movable)
                        else:
                            ov = _make_svg_path_overlay(
                                eid, tag, qp, sx, sy,
                                movable=movable)
                    else:
                        ov = _make_svg_overlay(
                            eid, tag, sr, sx, sy,
                            movable=movable)
                    ov.setZValue(10 + i)
                    ov._dbl_cb = lambda e=eid: \
                        self.element_double_clicked.emit(e)
                    ov._mov_cb = lambda e, dx, dy: \
                        self.element_moved.emit(e, dx, dy)
                    self._sc.addItem(ov)
                    self._overlays[eid] = ov

            def _on_sel_changed(self):
                for it in self._sc.selectedItems():
                    if hasattr(it, 'eid'):
                        self.element_selected.emit(it.eid)
                        return

            # ── events ────────────────────────────────────────────────────
            def wheelEvent(self, e):
                f = 1.15 if e.angleDelta().y() > 0 else 1.0 / 1.15
                self.scale(f, f)
                e.accept()

            def mousePressEvent(self, e):
                from PySide6 import QtCore
                if e.button() == QtCore.Qt.MouseButton.MiddleButton:
                    self._pan_pos = e.position().toPoint()
                    e.accept()
                    return
                super().mousePressEvent(e)

            def mouseMoveEvent(self, e):
                if self._pan_pos is not None:
                    d = e.position().toPoint() - self._pan_pos
                    self._pan_pos = e.position().toPoint()
                    self.horizontalScrollBar().setValue(
                        self.horizontalScrollBar().value() - d.x())
                    self.verticalScrollBar().setValue(
                        self.verticalScrollBar().value() - d.y())
                    e.accept()
                    return
                super().mouseMoveEvent(e)

            def keyPressEvent(self, e):
                if e.key() in (QtCore.Qt.Key.Key_Delete,
                               QtCore.Qt.Key.Key_Backspace):
                    for it in self._sc.selectedItems():
                        if hasattr(it, 'eid'):
                            self.delete_requested.emit(it.eid)
                            e.accept()
                            return
                    # No overlay selected — let the event propagate
                    # so NodeGraphQt can delete the node itself
                    return
                # Ctrl+Z / Ctrl+Shift+Z handled by widget
                super().keyPressEvent(e)

            def mouseReleaseEvent(self, e):
                from PySide6 import QtCore
                if e.button() == QtCore.Qt.MouseButton.MiddleButton:
                    self._pan_pos = None
                    e.accept()
                    return
                super().mouseReleaseEvent(e)

        instance = _View()
        return instance


# ── Properties panel ─────────────────────────────────────────────────────────

class _SvgPropsPanel(object):
    """
    QWidget that shows editable properties for the selected SVG element.
    Shown in a splitter pane; double-click any overlay item to open it.

    Signal
    ------
    apply_requested(str)  – element id; panel has already written changes to
                            the xml_elem in-place; caller should rerender.
    """

    _PANEL_SS = """
        QWidget          { background: #2b2b2b; color: #ddd; }
        QLabel           { color: #bbb; font-size: 9pt; }
        QDoubleSpinBox, QLineEdit {
            background: #1e1e1e; border: 1px solid #444; color: #eee;
            padding: 2px; border-radius: 2px;
        }
        QPushButton {
            background: #3a3a3a; border: 1px solid #555; color: #ddd;
            padding: 3px 8px; border-radius: 3px;
        }
        QPushButton:hover          { background: #4a4a4a; }
        QPushButton#apply_btn      { background: #1a5c1a; border-color: #2e7d32; }
        QPushButton#apply_btn:hover{ background: #2a7c2a; }
        QPushButton#reset_btn      { background: #5c1a1a; border-color: #7d2e2e; }
        QPushButton#reset_btn:hover{ background: #7c2a2a; }
    """

    _CSS_FIELDS = [
        ('fill',            'color'),
        ('stroke',          'color'),
        ('stroke-width',    'float'),
        ('opacity',         'float01'),
        ('font-size',       'float'),
        ('stroke-dasharray','text'),
    ]

    def __new__(cls, parent=None):
        from PySide6 import QtWidgets, QtCore, QtGui

        class _Panel(QtWidgets.QWidget):
            apply_requested = QtCore.Signal(str)   # element_id
            reset_requested = QtCore.Signal()

            def __init__(self, parent=None):
                super().__init__(parent)
                self.setStyleSheet(_SvgPropsPanel._PANEL_SS)
                self.setMinimumWidth(300)
                self._eid      = None
                self._xml_elem = None
                self._svg_root = None
                self._wmap     = {}   # key → widget with ._gv() value getter

                outer = QtWidgets.QVBoxLayout(self)
                outer.setContentsMargins(6, 6, 6, 6)
                outer.setSpacing(3)

                # title
                self._title = QtWidgets.QLabel("No element selected")
                self._title.setWordWrap(True)
                f = self._title.font(); f.setBold(True)
                self._title.setFont(f)
                outer.addWidget(self._title)

                sep = QtWidgets.QFrame()
                sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
                sep.setStyleSheet("color: #444;")
                outer.addWidget(sep)

                # scroll area for form
                scroll = QtWidgets.QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setStyleSheet("QScrollArea { border: none; }")
                self._fc = QtWidgets.QWidget()
                self._fl = QtWidgets.QFormLayout(self._fc)
                self._fl.setContentsMargins(4, 4, 4, 4)
                self._fl.setSpacing(6)
                self._fl.setLabelAlignment(
                    QtCore.Qt.AlignmentFlag.AlignRight)
                scroll.setWidget(self._fc)
                outer.addWidget(scroll, stretch=1)

                # buttons
                btn_row = QtWidgets.QHBoxLayout()
                self._apply_btn = QtWidgets.QPushButton("Apply")
                self._apply_btn.setObjectName("apply_btn")
                self._apply_btn.setEnabled(False)
                self._apply_btn.clicked.connect(self._on_apply)

                self._reset_btn = QtWidgets.QPushButton("Reset SVG")
                self._reset_btn.setObjectName("reset_btn")
                self._reset_btn.setToolTip(
                    "Reload SVG from the upstream figure (clears all edits)")
                self._reset_btn.clicked.connect(
                    lambda: self.reset_requested.emit())

                btn_row.addWidget(self._apply_btn, stretch=1)
                btn_row.addWidget(self._reset_btn)
                outer.addLayout(btn_row)

                # Auto-apply with debounce — changes fire after a short
                # pause so the user sees live updates without excessive
                # rerenders during rapid spinbox clicks.
                self._debounce = QtCore.QTimer(self)
                self._debounce.setSingleShot(True)
                self._debounce.setInterval(350)
                self._debounce.timeout.connect(self._on_apply)

            # ── public ────────────────────────────────────────────────────
            def set_svg_root(self, root):
                """Store reference to SVG root for marker shape operations."""
                self._svg_root = root

            def load(self, eid: str, xml_elem):
                self._debounce.stop()   # cancel pending auto-apply
                self._eid      = eid
                self._xml_elem = xml_elem
                self._wmap.clear()
                while self._fl.rowCount():
                    self._fl.removeRow(0)

                if xml_elem is None:
                    self._title.setText(f'Element not found: {eid}')
                    self._apply_btn.setEnabled(False)
                    return

                tag   = _ltag(xml_elem)
                short = eid if len(eid) <= 30 else eid[:27] + '…'
                self._title.setText(f'<{tag}>  {short}')
                self._apply_btn.setEnabled(True)

                # Collect CSS style dict
                css = _css_parse(xml_elem.get('style', ''))
                # Fallback: direct attributes
                for a in ('fill', 'stroke', 'opacity'):
                    if a not in css and xml_elem.get(a):
                        css[a] = xml_elem.get(a)
                # For groups: search ALL descendants for the first
                # element with a style (direct children may be nested
                # <g clip-path> wrappers without their own styles).
                if tag == 'g':
                    for ch in xml_elem.iter():
                        if ch is xml_elem:
                            continue
                        ch_css = _css_parse(ch.get('style', ''))
                        if ch_css:
                            for k in ('fill', 'stroke', 'stroke-width',
                                      'opacity', 'font-size'):
                                if k not in css and k in ch_css:
                                    css[k] = ch_css[k]
                            break

                # ── Build type-aware field list with sensible defaults ──
                # Determine element flavour
                is_text = (tag in ('text', 'tspan')
                           or _has_text_descendant(xml_elem)
                           or eid.startswith('stat_text:')
                           or eid.startswith('annotation_'))
                is_scatter = eid.startswith('PathCollection_')
                is_line = eid.startswith('line2d_')

                # Core fields for every element
                # Text elements default fill to black (fill:none makes text invisible)
                fill_default = '#000000' if is_text else 'none'
                fields = [
                    ('fill',         'color',   fill_default),
                    ('stroke',       'color',   'none'),
                    ('stroke-width', 'float',   '1'),
                    ('opacity',      'float01', '1'),
                ]
                if is_text:
                    # Matplotlib SVG text encodes font size via
                    # transform="…scale(S -S)" on an inner <g>, NOT
                    # CSS font-size.  Read the scale factor and convert
                    # to an approximate pt value (scale × 100 ≈ pt).
                    scale_val, _inner = _get_mpl_text_scale(xml_elem)
                    if scale_val:
                        default_fs = str(round(scale_val * 100, 1))
                    else:
                        # Fallback: try CSS / attribute
                        default_fs = ''
                        for el in xml_elem.iter():
                            fs = (el.get('font-size', '')
                                  or _css_parse(el.get('style', '')).get(
                                      'font-size', ''))
                            if fs:
                                default_fs = fs.replace('px', '').replace(
                                    'pt', '').strip()
                                break
                        if not default_fs:
                            default_fs = '10'
                    fields.append(('font-size', 'float', default_fs))
                elif is_line:
                    # Named line-style presets instead of raw dasharray
                    pass  # handled below as a combo
                else:
                    fields.append(
                        ('stroke-dasharray', 'text', ''))

                # Render fields — use CSS value when present, else default
                for prop, kind, default in fields:
                    val = css.get(prop, '') or default
                    # Never show font-size as 0 (Qt warning)
                    if prop == 'font-size' and val:
                        try:
                            if float(val.replace('px', '').replace(
                                    'pt', '')) <= 0:
                                val = default
                        except ValueError:
                            pass
                    w = self._make_editor(kind, val)
                    if w is not None:
                        self._wmap[f'css:{prop}'] = w
                        lbl = QtWidgets.QLabel(prop)
                        lbl.setStyleSheet("color: #aaa; font-size: 9pt;")
                        self._fl.addRow(lbl, w)

                # ── Scatter marker shape & size ───────────────────────
                if is_scatter:
                    sep_m = QtWidgets.QFrame()
                    sep_m.setFrameShape(QtWidgets.QFrame.Shape.HLine)
                    sep_m.setStyleSheet("color: #444;")
                    self._fl.addRow(sep_m)

                    cur_shape = _detect_marker_shape(xml_elem)
                    combo = QtWidgets.QComboBox()
                    combo.addItems(_MARKER_NAMES)
                    idx = _MARKER_NAMES.index(cur_shape) \
                        if cur_shape in _MARKER_NAMES else 0
                    combo.setCurrentIndex(idx)
                    combo.setStyleSheet(
                        "QComboBox { background: #1e1e1e;"
                        " border: 1px solid #444; color: #eee;"
                        " padding: 2px; }")
                    combo._gv = combo.currentText
                    combo.currentIndexChanged.connect(
                        lambda _v: self._debounce.start())
                    self._wmap['marker:shape'] = combo
                    lbl_ms = QtWidgets.QLabel("marker")
                    lbl_ms.setStyleSheet("color: #aaa; font-size: 9pt;")
                    self._fl.addRow(lbl_ms, combo)

                    # Marker size — scale factor on <use> transforms
                    cur_size = 1.0
                    for _u in xml_elem.iter():
                        _ut = _u.get('transform', '')
                        m_s = re.search(
                            r'scale\(([\d.]+)', _ut)
                        if m_s and _ltag(_u) == 'use':
                            try:
                                cur_size = float(m_s.group(1))
                            except ValueError:
                                pass
                            break
                    sp_sz = QtWidgets.QDoubleSpinBox()
                    sp_sz.setRange(0.1, 20.0)
                    sp_sz.setDecimals(2)
                    sp_sz.setSingleStep(0.1)
                    sp_sz.setValue(cur_size)
                    sp_sz.setStyleSheet(
                        "QDoubleSpinBox { background: #1e1e1e;"
                        " border: 1px solid #444; color: #eee;"
                        " padding: 2px; }")
                    sp_sz._gv = lambda s=sp_sz: str(s.value())
                    sp_sz.valueChanged.connect(
                        lambda _v: self._debounce.start())
                    self._wmap['marker:size'] = sp_sz
                    lbl_msz = QtWidgets.QLabel("marker size")
                    lbl_msz.setStyleSheet("color: #aaa; font-size: 9pt;")
                    self._fl.addRow(lbl_msz, sp_sz)

                # ── Line style preset ─────────────────────────────────
                if is_line:
                    sep_l = QtWidgets.QFrame()
                    sep_l.setFrameShape(QtWidgets.QFrame.Shape.HLine)
                    sep_l.setStyleSheet("color: #444;")
                    self._fl.addRow(sep_l)

                    _LINE_STYLES = [
                        ('Solid',    ''),
                        ('Dashed',   '6,4'),
                        ('Dotted',   '2,2'),
                        ('Dash-Dot', '6,2,2,2'),
                    ]
                    cur_da = css.get('stroke-dasharray', '') or ''
                    combo_ls = QtWidgets.QComboBox()
                    combo_ls.setStyleSheet(
                        "QComboBox { background: #1e1e1e;"
                        " border: 1px solid #444; color: #eee;"
                        " padding: 2px; }")
                    match_idx = 0
                    for li, (lname, lval) in enumerate(_LINE_STYLES):
                        combo_ls.addItem(lname, lval)
                        if cur_da.replace(' ', '') == lval.replace(' ', ''):
                            match_idx = li
                    combo_ls.setCurrentIndex(match_idx)
                    combo_ls._gv = lambda c=combo_ls: c.currentData()
                    combo_ls.currentIndexChanged.connect(
                        lambda _v: self._debounce.start())
                    self._wmap['css:stroke-dasharray'] = combo_ls
                    lbl_ls = QtWidgets.QLabel("line style")
                    lbl_ls.setStyleSheet("color: #aaa; font-size: 9pt;")
                    self._fl.addRow(lbl_ls, combo_ls)

                # ── Font family & weight (text elements) ──────────────
                if is_text:
                    sep_f = QtWidgets.QFrame()
                    sep_f.setFrameShape(QtWidgets.QFrame.Shape.HLine)
                    sep_f.setStyleSheet("color: #444;")
                    self._fl.addRow(sep_f)

                    # Font family
                    _FONTS = ['sans-serif', 'serif', 'monospace',
                              'Arial', 'Helvetica', 'Times New Roman',
                              'Courier New', 'Georgia', 'Verdana']
                    cur_ff = css.get('font-family', '')
                    if not cur_ff:
                        for el in xml_elem.iter():
                            cur_ff = (el.get('font-family', '')
                                      or _css_parse(el.get('style', '')).get(
                                          'font-family', ''))
                            if cur_ff:
                                break
                    # Extract first font from comma-separated CSS font-family list
                    # e.g. "Arial', 'Helvetica', sans-serif" → "Arial"
                    if ',' in cur_ff:
                        cur_ff = cur_ff.split(',')[0]
                    cur_ff = cur_ff.strip().strip("'\"")
                    combo_ff = QtWidgets.QComboBox()
                    combo_ff.setEditable(True)
                    combo_ff.setStyleSheet(
                        "QComboBox { background: #1e1e1e;"
                        " border: 1px solid #444; color: #eee;"
                        " padding: 2px; }")
                    combo_ff.addItems(_FONTS)
                    ff_idx = -1
                    for fi, fn in enumerate(_FONTS):
                        if fn.lower() == cur_ff.lower():
                            ff_idx = fi
                            break
                    if ff_idx >= 0:
                        combo_ff.setCurrentIndex(ff_idx)
                    elif cur_ff:
                        combo_ff.setCurrentText(cur_ff)
                    combo_ff._gv = combo_ff.currentText
                    combo_ff.currentTextChanged.connect(
                        lambda _v: self._debounce.start())
                    self._wmap['css:font-family'] = combo_ff
                    lbl_ff = QtWidgets.QLabel("font")
                    lbl_ff.setStyleSheet("color: #aaa; font-size: 9pt;")
                    self._fl.addRow(lbl_ff, combo_ff)

                    # Font weight
                    cur_fw = css.get('font-weight', '')
                    if not cur_fw:
                        for el in xml_elem.iter():
                            cur_fw = (el.get('font-weight', '')
                                      or _css_parse(el.get('style', '')).get(
                                          'font-weight', ''))
                            if cur_fw:
                                break
                    combo_fw = QtWidgets.QComboBox()
                    combo_fw.setStyleSheet(
                        "QComboBox { background: #1e1e1e;"
                        " border: 1px solid #444; color: #eee;"
                        " padding: 2px; }")
                    combo_fw.addItems(['normal', 'bold'])
                    combo_fw.setCurrentIndex(
                        1 if cur_fw in ('bold', '700') else 0)
                    combo_fw._gv = combo_fw.currentText
                    combo_fw.currentIndexChanged.connect(
                        lambda _v: self._debounce.start())
                    self._wmap['css:font-weight'] = combo_fw
                    lbl_fw = QtWidgets.QLabel("weight")
                    lbl_fw.setStyleSheet("color: #aaa; font-size: 9pt;")
                    self._fl.addRow(lbl_fw, combo_fw)

                # ── Text content row ──────────────────────────────────
                txt = _elem_texts(xml_elem)
                if txt:
                    sep2 = QtWidgets.QFrame()
                    sep2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
                    sep2.setStyleSheet("color: #444;")
                    self._fl.addRow(sep2)
                    te = QtWidgets.QLineEdit(txt)
                    te._gv = te.text
                    te.textChanged.connect(
                        lambda _v: self._debounce.start())
                    self._wmap['text:content'] = te
                    lbl2 = QtWidgets.QLabel("text content")
                    lbl2.setStyleSheet("color: #aaa; font-size: 9pt;")
                    self._fl.addRow(lbl2, te)

                # ── Geometry controls for annotation shapes ──────────────
                is_ann_shape = any(eid.startswith(p) for p in
                                   ('rect_ann_', 'ellipse_ann_',
                                    'line_ann_', 'arrow_ann_'))
                if is_ann_shape:
                    sep_g = QtWidgets.QFrame()
                    sep_g.setFrameShape(QtWidgets.QFrame.Shape.HLine)
                    sep_g.setStyleSheet("color: #444;")
                    self._fl.addRow(sep_g)

                    _SB_SS = ("QDoubleSpinBox { background: #1e1e1e;"
                              " border: 1px solid #444; color: #eee;"
                              " padding: 2px; }")

                    if tag == 'rect':
                        for attr, label in [('x', 'X'), ('y', 'Y'),
                                            ('width', 'W'), ('height', 'H')]:
                            val = float(xml_elem.get(attr, 0))
                            sb = QtWidgets.QDoubleSpinBox()
                            sb.setRange(-9999, 9999)
                            sb.setDecimals(1)
                            sb.setSingleStep(1.0)
                            sb.setValue(val)
                            sb.setStyleSheet(_SB_SS)
                            sb._gv = lambda s=sb: str(s.value())
                            sb.valueChanged.connect(
                                lambda _v: self._debounce.start())
                            self._wmap[f'attr:{attr}'] = sb
                            lbl_g = QtWidgets.QLabel(label)
                            lbl_g.setStyleSheet("color: #aaa; font-size: 9pt;")
                            self._fl.addRow(lbl_g, sb)

                    elif tag == 'ellipse':
                        for attr, label in [('cx', 'CX'), ('cy', 'CY'),
                                            ('rx', 'RX'), ('ry', 'RY')]:
                            val = float(xml_elem.get(attr, 0))
                            sb = QtWidgets.QDoubleSpinBox()
                            sb.setRange(-9999, 9999)
                            sb.setDecimals(1)
                            sb.setSingleStep(1.0)
                            sb.setValue(val)
                            sb.setStyleSheet(_SB_SS)
                            sb._gv = lambda s=sb: str(s.value())
                            sb.valueChanged.connect(
                                lambda _v: self._debounce.start())
                            self._wmap[f'attr:{attr}'] = sb
                            lbl_g = QtWidgets.QLabel(label)
                            lbl_g.setStyleSheet("color: #aaa; font-size: 9pt;")
                            self._fl.addRow(lbl_g, sb)

                    elif tag == 'line':
                        for attr, label in [('x1', 'X1'), ('y1', 'Y1'),
                                            ('x2', 'X2'), ('y2', 'Y2')]:
                            val = float(xml_elem.get(attr, 0))
                            sb = QtWidgets.QDoubleSpinBox()
                            sb.setRange(-9999, 9999)
                            sb.setDecimals(1)
                            sb.setSingleStep(1.0)
                            sb.setValue(val)
                            sb.setStyleSheet(_SB_SS)
                            sb._gv = lambda s=sb: str(s.value())
                            sb.valueChanged.connect(
                                lambda _v: self._debounce.start())
                            self._wmap[f'attr:{attr}'] = sb
                            lbl_g = QtWidgets.QLabel(label)
                            lbl_g.setStyleSheet("color: #aaa; font-size: 9pt;")
                            self._fl.addRow(lbl_g, sb)

                self._fc.adjustSize()

            # ── private ───────────────────────────────────────────────────
            def _make_editor(self, kind: str, current: str):
                """Return a configured editor widget with a ._gv() value getter."""
                current = str(current).strip()

                if kind == 'color':
                    ctr = QtWidgets.QWidget()
                    hl  = QtWidgets.QHBoxLayout(ctr)
                    hl.setContentsMargins(0, 0, 0, 0)
                    hl.setSpacing(3)
                    sw = QtWidgets.QPushButton()
                    sw.setFixedSize(22, 22)
                    tx = QtWidgets.QLineEdit()
                    tx.setMaximumWidth(90)
                    tx.setText(current)
                    sw.setStyleSheet(self._swatch_ss(current))

                    def _pick(_checked=False, s=sw, t=tx):
                        col = QtWidgets.QColorDialog.getColor(
                            QtGui.QColor(t.text())
                            if QtGui.QColor(t.text()).isValid()
                            else QtGui.QColor(),
                            QtWidgets.QApplication.activeWindow())
                        if col.isValid():
                            h = col.name()
                            t.setText(h)
                            s.setStyleSheet(self._swatch_ss(h))

                    sw.clicked.connect(_pick)
                    tx.textChanged.connect(
                        lambda v, s=sw: s.setStyleSheet(self._swatch_ss(v)))
                    tx.textChanged.connect(
                        lambda _v: self._debounce.start())
                    hl.addWidget(sw)
                    hl.addWidget(tx)
                    hl.addStretch()
                    ctr._gv = tx.text
                    return ctr

                if kind == 'float':
                    sp = QtWidgets.QDoubleSpinBox()
                    sp.setRange(-9999, 9999)
                    sp.setDecimals(2)
                    sp.setSingleStep(0.5)
                    try:
                        num = ''.join(
                            c for c in current if c in '0123456789.-+')
                        sp.setValue(float(num) if num else 0.0)
                    except ValueError:
                        sp.setValue(0.0)
                    sp._gv = lambda s=sp: str(s.value())
                    sp.valueChanged.connect(
                        lambda _v: self._debounce.start())
                    return sp

                if kind == 'float01':
                    sp = QtWidgets.QDoubleSpinBox()
                    sp.setRange(0.0, 1.0)
                    sp.setDecimals(3)
                    sp.setSingleStep(0.05)
                    try:
                        sp.setValue(float(current))
                    except ValueError:
                        sp.setValue(1.0)
                    sp._gv = lambda s=sp: str(s.value())
                    sp.valueChanged.connect(
                        lambda _v: self._debounce.start())
                    return sp

                if kind == 'text':
                    ed = QtWidgets.QLineEdit(current)
                    ed._gv = ed.text
                    ed.textChanged.connect(
                        lambda _v: self._debounce.start())
                    return ed

                return None

            def _swatch_ss(self, v: str) -> str:
                c  = QtGui.QColor(v)
                bg = c.name() if c.isValid() else '#666'
                return (f"background: {bg}; border: 1px solid #555;"
                        f" border-radius: 2px;")

            def _on_apply(self):
                self._debounce.stop()
                if not self._eid or self._xml_elem is None:
                    return
                elem = self._xml_elem
                tag  = _ltag(elem)
                css  = _css_parse(elem.get('style', ''))

                # Track which CSS properties the user actually changed
                user_changed = {}
                for key, w in self._wmap.items():
                    val = w._gv() if hasattr(w, '_gv') else None
                    if val is None:
                        continue

                    if key.startswith('css:'):
                        prop = key[4:]
                        user_changed[prop] = val
                        css[prop] = val

                    elif key.startswith('attr:'):
                        # Direct SVG attribute (x, y, width, height, etc.)
                        attr_name = key[5:]
                        elem.set(attr_name, val)

                    elif key == 'text:content':
                        for sub in elem.iter():
                            lt = _ltag(sub)
                            if lt == 'tspan' and sub.text:
                                sub.text = val
                                break
                            if lt == 'text' and sub.text:
                                sub.text = val
                                break

                    elif key == 'marker:shape':
                        # Replace marker path for THIS group only.
                        new_d = _MARKER_PATHS.get(val)
                        if new_d:
                            _change_marker_shape_for_group(
                                elem, new_d, self._svg_root)

                    elif key == 'marker:size':
                        # Scale <use> elements around their own center.
                        # Matplotlib positions <use> with x/y attrs, so a bare
                        # scale() would also scale the x/y offset from origin.
                        # Use translate(cx,cy) scale(s) translate(-cx,-cy) to
                        # scale in place.
                        try:
                            new_scale = float(val)
                            if new_scale > 0:
                                for ue in elem.iter():
                                    if _ltag(ue) == 'use':
                                        cx = float(ue.get('x', 0))
                                        cy = float(ue.get('y', 0))
                                        ue.set('transform',
                                               f'translate({cx:.4g},{cy:.4g}) '
                                               f'scale({new_scale:.4g}) '
                                               f'translate({-cx:.4g},{-cy:.4g})')
                        except (ValueError, TypeError):
                            pass

                # ── Handle font-size ──
                # With svg.fonttype='none', text is real <text> elements —
                # CSS font-size works directly. For old glyph-path text,
                # try scale transform as fallback.
                if 'font-size' in user_changed:
                    try:
                        new_pt = float(user_changed['font-size'])
                        if new_pt > 0:
                            # Try scale-based approach for old glyph-path text
                            _, inner_g = _get_mpl_text_scale(elem)
                            if inner_g is not None:
                                new_s = new_pt / 100.0
                                t = inner_g.get('transform', '')
                                t = re.sub(
                                    r'scale\([\d.]+\s+(-?)[\d.]+\)',
                                    lambda m: f'scale({new_s:.6g} '
                                              f'{m.group(1)}{new_s:.6g})',
                                    t)
                                inner_g.set('transform', t)
                                # Scale text: don't use CSS font-size
                                css.pop('font-size', None)
                                user_changed.pop('font-size', None)
                            # else: real <text> element — CSS font-size will
                            # be applied below via the normal CSS write path
                    except (ValueError, TypeError):
                        pass

                # Write CSS back to element
                if css:
                    elem.set('style', _css_build(css))

                # Propagate user-changed properties to ALL leaf descendants.
                # Critical for groups: children have their own style attrs
                # that override the parent — we must force-set on leaves.
                # Skip <defs> descendants — those are shared marker/clip
                # definitions that must not be mutated.
                if user_changed and tag == 'g':
                    _LEAF = {'path', 'line', 'circle', 'ellipse',
                             'rect', 'polygon', 'polyline',
                             'text', 'tspan', 'use'}
                    # Collect all elements inside <defs> so we skip them
                    _defs_elems: set = set()
                    for _de in elem.iter():
                        if _ltag(_de) == 'defs':
                            for _dd in _de.iter():
                                _defs_elems.add(_dd)
                    _TEXT_TAGS = {'text', 'tspan'}
                    for ch in elem.iter():
                        if ch is elem:
                            continue
                        if ch in _defs_elems:
                            continue
                        lt = _ltag(ch)
                        if lt in _LEAF:
                            c2 = _css_parse(ch.get('style', ''))
                            _FONT_PROPS = {
                                'font-size', 'font-family', 'font-weight'}
                            for k, v in user_changed.items():
                                # Skip font props on non-text leaves
                                if k in _FONT_PROPS and lt not in _TEXT_TAGS:
                                    continue
                                # Skip stroke-dasharray on text leaves
                                if k == 'stroke-dasharray' and lt in _TEXT_TAGS:
                                    continue
                                c2[k] = v
                            ch.set('style', _css_build(c2))

                self.apply_requested.emit(self._eid)

        return _Panel(parent)


# ── NodeBaseWidget wrapper ────────────────────────────────────────────────────

class NodeSvgEditorWidget(object):
    """
    NodeBaseWidget that embeds the SVG editor (view + properties panel)
    directly on the node surface.

    Signal
    ------
    svg_modified(str) – emitted after each user edit with the new SVG string
    reset_requested() – emitted when the user clicks "Reset SVG"
    """

    def __new__(cls, parent=None):
        from PySide6 import QtWidgets, QtCore, QtGui
        from NodeGraphQt.widgets.node_widgets import NodeBaseWidget

        class _Widget(NodeBaseWidget):
            svg_modified  = QtCore.Signal(str)
            reset_requested = QtCore.Signal()

            def __init__(self, parent=None):
                super().__init__(parent, name='_svg_editor', label='')
                self._svg_bytes = None
                self._undo_stack = []  # list of bytes (XML snapshots)
                self._redo_stack = []
                _MAX_UNDO = 50

                ctr = QtWidgets.QWidget()
                ctr.setMinimumWidth(1100)
                ctr.setMinimumHeight(800)
                root = QtWidgets.QVBoxLayout(ctr)
                root.setContentsMargins(2, 2, 2, 2)
                root.setSpacing(2)

                _TB_SS = ("QPushButton { background: #3a3a3a;"
                          " border: 1px solid #555; color: #ddd;"
                          " border-radius: 3px; }"
                          " QPushButton:hover { background: #4a4a4a; }"
                          " QPushButton:disabled { color: #666; }")

                # ── toolbar row 1: zoom + undo/redo ──────────────────────
                tb = QtWidgets.QHBoxLayout()
                for label, slot in (('+', 'zoom_in'),
                                    ('−', 'zoom_out'),
                                    ('Fit', 'zoom_fit')):
                    b = QtWidgets.QPushButton(label)
                    b.setFixedHeight(22)
                    b.setFixedWidth(36 if len(label) == 1 else 40)
                    b.setStyleSheet(_TB_SS)
                    b.clicked.connect(
                        lambda _=None, s=slot: getattr(self._view, s)())
                    tb.addWidget(b)

                tb.addWidget(self._vsep())

                # Undo / Redo
                self._undo_btn = QtWidgets.QPushButton('Undo')
                self._undo_btn.setFixedHeight(22)
                self._undo_btn.setMinimumWidth(48)
                self._undo_btn.setStyleSheet(_TB_SS)
                self._undo_btn.setEnabled(False)
                self._undo_btn.setToolTip('Undo (Ctrl+Z)')
                self._undo_btn.clicked.connect(self._on_undo)
                tb.addWidget(self._undo_btn)

                self._redo_btn = QtWidgets.QPushButton('Redo')
                self._redo_btn.setFixedHeight(22)
                self._redo_btn.setMinimumWidth(48)
                self._redo_btn.setStyleSheet(_TB_SS)
                self._redo_btn.setEnabled(False)
                self._redo_btn.setToolTip('Redo (Ctrl+Shift+Z)')
                self._redo_btn.clicked.connect(self._on_redo)
                tb.addWidget(self._redo_btn)

                tb.addWidget(self._vsep())

                # Delete
                del_btn = QtWidgets.QPushButton('Del')
                del_btn.setFixedHeight(22)
                del_btn.setMinimumWidth(36)
                del_btn.setStyleSheet(_TB_SS)
                del_btn.setToolTip('Delete selected element')
                del_btn.clicked.connect(self._on_delete_selected)
                tb.addWidget(del_btn)

                tb.addWidget(self._vsep())

                # Add text
                add_text_btn = QtWidgets.QPushButton('Txt')
                add_text_btn.setToolTip('Add text annotation')
                add_text_btn.setFixedHeight(22)
                add_text_btn.setMinimumWidth(36)
                add_text_btn.setStyleSheet(_TB_SS)
                add_text_btn.clicked.connect(self._on_add_text)
                tb.addWidget(add_text_btn)

                # Add shapes
                for label, tip, slot in (
                    ('Rect',  'Add rectangle', '_on_add_rect'),
                    ('Oval',  'Add ellipse',   '_on_add_ellipse'),
                    ('Line',  'Add line',      '_on_add_line'),
                    ('Arrow', 'Add arrow',     '_on_add_arrow'),
                ):
                    sb = QtWidgets.QPushButton(label)
                    sb.setToolTip(tip)
                    sb.setFixedHeight(22)
                    sb.setMinimumWidth(42)
                    sb.setStyleSheet(_TB_SS)
                    sb.clicked.connect(
                        lambda _=None, s=slot: getattr(self, s)())
                    tb.addWidget(sb)

                tb.addStretch()
                self._info = QtWidgets.QLabel("No SVG loaded")
                self._info.setStyleSheet(
                    "color: #777; font-size: 9pt; padding-right: 4px;")
                tb.addWidget(self._info)
                root.addLayout(tb)

                # ── toolbar row 2: color palettes ──────────────────────
                tb2 = QtWidgets.QHBoxLayout()
                pal_lbl = QtWidgets.QLabel("Palette:")
                pal_lbl.setStyleSheet("color: #aaa; font-size: 8pt;")
                tb2.addWidget(pal_lbl)

                _PALETTES = {
                    'Nature': ['#E64B35','#4DBBD5','#00A087',
                               '#3C5488','#F39B7F','#8491B4','#91D1C2'],
                    'Science': ['#3B4992','#EE0000','#008B45',
                                '#631879','#008280','#BB0021','#5F559B'],
                    'ColorBrewer Set1': ['#E41A1C','#377EB8','#4DAF4A',
                                         '#984EA3','#FF7F00','#A65628'],
                    'Pastel': ['#AEC6CF','#FFD1DC','#B5EAD7',
                               '#FFDAC1','#C7CEEA','#F7DC6F','#A8D8A8'],
                    'Grayscale': ['#000000','#444444','#888888',
                                  '#AAAAAA','#CCCCCC','#EEEEEE'],
                }
                self._pal_combo = QtWidgets.QComboBox()
                self._pal_combo.setFixedHeight(22)
                self._pal_combo.setFixedWidth(160)
                self._pal_combo.setStyleSheet(
                    "QComboBox { background:#3a3a3a; color:#ddd;"
                    " border:1px solid #555; border-radius:3px; }"
                    " QComboBox QAbstractItemView { background:#2d2d2d;"
                    " color:#ddd; }")
                for name in _PALETTES:
                    self._pal_combo.addItem(name)
                tb2.addWidget(self._pal_combo)

                apply_pal_btn = QtWidgets.QPushButton('Apply')
                apply_pal_btn.setFixedHeight(22)
                apply_pal_btn.setFixedWidth(70)
                apply_pal_btn.setStyleSheet(_TB_SS)
                apply_pal_btn.setToolTip(
                    'Recolor all data series with selected palette')
                apply_pal_btn.clicked.connect(
                    lambda: self._on_apply_palette(
                        _PALETTES[self._pal_combo.currentText()]))
                tb2.addWidget(apply_pal_btn)

                tb2.addStretch()
                root.addLayout(tb2)

                # ── toolbar row 3: canvas padding ─────────────────────────
                tb3 = QtWidgets.QHBoxLayout()
                pad_lbl = QtWidgets.QLabel("Padding:")
                pad_lbl.setStyleSheet("color: #aaa; font-size: 8pt;")
                tb3.addWidget(pad_lbl)
                for side, label in [('top', 'T'), ('bottom', 'B'),
                                    ('left', 'L'), ('right', 'R')]:
                    sl = QtWidgets.QLabel(label)
                    sl.setStyleSheet("color:#888; font-size:8pt;")
                    sb = QtWidgets.QSpinBox()
                    sb.setRange(0, 500)
                    sb.setValue(0)
                    sb.setSuffix('px')
                    sb.setFixedWidth(65)
                    sb.setFixedHeight(22)
                    sb.setStyleSheet(
                        "QSpinBox { background:#3a3a3a; color:#ddd;"
                        " border:1px solid #555; border-radius:3px; }")
                    setattr(self, f'_pad_{side}', sb)
                    tb3.addWidget(sl)
                    tb3.addWidget(sb)
                apply_pad_btn = QtWidgets.QPushButton('Apply Padding')
                apply_pad_btn.setFixedHeight(22)
                apply_pad_btn.setStyleSheet(_TB_SS)
                apply_pad_btn.clicked.connect(self._on_apply_padding)
                tb3.addWidget(apply_pad_btn)
                tb3.addStretch()
                root.addLayout(tb3)

                # ── splitter: view  |  properties panel ───────────────────
                self._sp = QtWidgets.QSplitter(
                    QtCore.Qt.Orientation.Horizontal)

                self._view = _SvgEditorView()
                self._sp.addWidget(self._view)

                self._props = _SvgPropsPanel()
                self._sp.addWidget(self._props)
                self._sp.setSizes([860, 0])   # props hidden until dbl-click

                root.addWidget(self._sp, stretch=1)

                # ── tip label ─────────────────────────────────────────────
                self._tip = QtWidgets.QLabel(
                    "Click to select & edit \u00b7 Del to remove \u00b7 "
                    "Ctrl+Z undo \u00b7 Ctrl+Shift+Z redo")
                self._tip.setAlignment(
                    QtCore.Qt.AlignmentFlag.AlignCenter)
                self._tip.setStyleSheet(
                    "color: #555; font-size: 8pt; padding: 2px;")
                root.addWidget(self._tip)

                self.set_custom_widget(ctr)

                # ── connections ───────────────────────────────────────────
                self._view.element_selected.connect(self._on_selected)
                self._view.element_double_clicked.connect(self._on_dbl)
                self._view.element_moved.connect(self._on_moved)
                self._view.delete_requested.connect(self._on_delete)
                self._props.apply_requested.connect(self._on_apply)
                self._props.reset_requested.connect(
                    lambda: self.reset_requested.emit())

                # Keyboard shortcuts
                undo_sc = QtGui.QShortcut(
                    QtGui.QKeySequence.StandardKey.Undo, ctr)
                undo_sc.activated.connect(self._on_undo)
                redo_sc = QtGui.QShortcut(
                    QtGui.QKeySequence.StandardKey.Redo, ctr)
                redo_sc.activated.connect(self._on_redo)

            # ── public ────────────────────────────────────────────────────
            def load_svg(self, svg_bytes: bytes):
                self._svg_bytes = svg_bytes
                self._undo_stack.clear()
                self._redo_stack.clear()
                self._undo_btn.setEnabled(False)
                self._redo_btn.setEnabled(False)
                self._view.load_svg(svg_bytes)
                n = self._view.overlay_count()
                self._info.setText(f"{n} element(s)")

            def get_svg(self) -> bytes:
                return self._svg_bytes

            # ── helpers ───────────────────────────────────────────────────
            @staticmethod
            def _vsep():
                sep = QtWidgets.QFrame()
                sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
                sep.setStyleSheet("color: #555;")
                sep.setFixedHeight(18)
                return sep

            def _snapshot(self):
                """Push current SVG bytes onto undo stack before a change."""
                if self._svg_bytes:
                    self._undo_stack.append(self._svg_bytes)
                    if len(self._undo_stack) > 50:
                        self._undo_stack.pop(0)
                    self._redo_stack.clear()
                    self._undo_btn.setEnabled(True)
                    self._redo_btn.setEnabled(False)

            def _center_vb(self):
                vb = self._view._vbox
                return vb.x() + vb.width() / 2, vb.y() + vb.height() / 2

            def _ns(self, local):
                return f'{{{_SVG_NS}}}{local}' if _SVG_NS else local

            def _unique_id(self, prefix):
                n = sum(1 for e in self._view.svg_root.iter()
                        if (e.get('id') or '').startswith(prefix))
                return f'{prefix}{n}'

            # ── undo / redo ───────────────────────────────────────────────
            def _on_undo(self):
                if not self._undo_stack:
                    return
                self._redo_stack.append(self._svg_bytes)
                prev = self._undo_stack.pop()
                self._svg_bytes = prev
                self._view.load_svg(prev)
                self._undo_btn.setEnabled(bool(self._undo_stack))
                self._redo_btn.setEnabled(True)
                self.svg_modified.emit(prev.decode('utf-8', errors='replace'))

            def _on_redo(self):
                if not self._redo_stack:
                    return
                self._undo_stack.append(self._svg_bytes)
                nxt = self._redo_stack.pop()
                self._svg_bytes = nxt
                self._view.load_svg(nxt)
                self._undo_btn.setEnabled(True)
                self._redo_btn.setEnabled(bool(self._redo_stack))
                self.svg_modified.emit(nxt.decode('utf-8', errors='replace'))

            # ── delete ────────────────────────────────────────────────────
            def _on_delete(self, eid: str):
                if self._view.svg_root is None:
                    return
                elem = _find_id(self._view.svg_root, eid)
                if elem is None:
                    return
                parent_map = {c: p for p in self._view.svg_root.iter()
                              for c in p}
                parent = parent_map.get(elem)
                if parent is None:
                    return  # root — don't delete
                self._snapshot()
                parent.remove(elem)
                self._tip.setText(f"Deleted: {eid}")
                self._reserialize()

            def _on_delete_selected(self):
                for it in self._view._sc.selectedItems():
                    if hasattr(it, 'eid'):
                        self._on_delete(it.eid)
                        return

            # ── slots ─────────────────────────────────────────────────────
            def _on_selected(self, eid: str):
                short = eid if len(eid) <= 34 else eid[:31] + '…'
                self._tip.setText(f"Selected: {short}")
                if self._sp.sizes()[1] < 50:
                    total = sum(self._sp.sizes())
                    self._sp.setSizes([int(total * 0.78), int(total * 0.22)])
                if self._view.svg_root is not None:
                    elem = _find_id(self._view.svg_root, eid)
                    self._props.set_svg_root(self._view.svg_root)
                    self._props.load(eid, elem)

            def _on_dbl(self, eid: str):
                self._on_selected(eid)

            def _on_moved(self, eid: str, dx: float, dy: float):
                if self._view.svg_root is None:
                    return
                elem = _find_id(self._view.svg_root, eid)
                if elem is None:
                    return
                self._snapshot()
                tag = _ltag(elem)
                is_ann = any(eid.startswith(p) for p in
                             ('annotation_', 'rect_ann_', 'ellipse_ann_',
                              'line_ann_', 'arrow_ann_'))

                if is_ann and tag in ('rect', 'ellipse', 'line', 'text'):
                    # Move annotation shapes by updating their coordinates
                    # directly instead of stacking translate transforms.
                    if tag == 'rect' or tag == 'text':
                        for attr in ('x', 'y'):
                            old = float(elem.get(attr, 0))
                            delta = dx if attr == 'x' else dy
                            elem.set(attr, f'{old + delta:.2f}')
                    elif tag == 'ellipse':
                        for attr in ('cx', 'cy'):
                            old = float(elem.get(attr, 0))
                            delta = dx if attr == 'cx' else dy
                            elem.set(attr, f'{old + delta:.2f}')
                    elif tag == 'line':
                        for attr in ('x1', 'x2'):
                            old = float(elem.get(attr, 0))
                            elem.set(attr, f'{old + dx:.2f}')
                        for attr in ('y1', 'y2'):
                            old = float(elem.get(attr, 0))
                            elem.set(attr, f'{old + dy:.2f}')
                else:
                    # For other elements (matplotlib text groups etc.),
                    # accumulate into a single translate.
                    existing = elem.get('transform', '')
                    m = re.search(
                        r'translate\(([\d.e+-]+)[,\s]+([\d.e+-]+)\)', existing)
                    if m:
                        old_dx = float(m.group(1))
                        old_dy = float(m.group(2))
                        new_t = f'translate({old_dx + dx:.4f} {old_dy + dy:.4f})'
                        existing = re.sub(
                            r'translate\([\d.e+-]+[,\s]+[\d.e+-]+\)',
                            new_t, existing, count=1)
                    else:
                        existing = f'translate({dx:.4f} {dy:.4f}) {existing}'
                    elem.set('transform', existing.strip())
                self._reserialize(eid)

            def _on_apply(self, eid: str):
                self._snapshot()
                self._reserialize(eid)

            # ── shape insertion ───────────────────────────────────────────
            def _on_add_text(self):
                if self._view.svg_root is None:
                    return
                self._snapshot()
                new_id = self._unique_id('annotation_')
                cx, cy = self._center_vb()
                el = _ET.SubElement(self._view.svg_root, self._ns('text'))
                el.set('id', new_id)
                el.set('x', f'{cx:.2f}')
                el.set('y', f'{cy:.2f}')
                el.set('style',
                       'fill:#000000;font-size:14px;font-family:sans-serif')
                el.text = 'Annotation'
                self._reserialize(keep_eid=new_id)

            def _on_add_rect(self):
                if self._view.svg_root is None:
                    return
                self._snapshot()
                new_id = self._unique_id('rect_ann_')
                cx, cy = self._center_vb()
                vb = self._view._vbox
                w, h = vb.width() * 0.15, vb.height() * 0.1
                el = _ET.SubElement(self._view.svg_root, self._ns('rect'))
                el.set('id', new_id)
                el.set('x', f'{cx - w/2:.2f}')
                el.set('y', f'{cy - h/2:.2f}')
                el.set('width',  f'{w:.2f}')
                el.set('height', f'{h:.2f}')
                el.set('style', 'fill:none;stroke:#000000;stroke-width:1.5px')
                self._reserialize(keep_eid=new_id)

            def _on_add_ellipse(self):
                if self._view.svg_root is None:
                    return
                self._snapshot()
                new_id = self._unique_id('ellipse_ann_')
                cx, cy = self._center_vb()
                vb = self._view._vbox
                rx, ry = vb.width() * 0.08, vb.height() * 0.06
                el = _ET.SubElement(self._view.svg_root, self._ns('ellipse'))
                el.set('id', new_id)
                el.set('cx', f'{cx:.2f}')
                el.set('cy', f'{cy:.2f}')
                el.set('rx', f'{rx:.2f}')
                el.set('ry', f'{ry:.2f}')
                el.set('style', 'fill:none;stroke:#000000;stroke-width:1.5px')
                self._reserialize(keep_eid=new_id)

            def _on_add_line(self):
                if self._view.svg_root is None:
                    return
                self._snapshot()
                new_id = self._unique_id('line_ann_')
                cx, cy = self._center_vb()
                half = self._view._vbox.width() * 0.1
                el = _ET.SubElement(self._view.svg_root, self._ns('line'))
                el.set('id', new_id)
                el.set('x1', f'{cx - half:.2f}')
                el.set('y1', f'{cy:.2f}')
                el.set('x2', f'{cx + half:.2f}')
                el.set('y2', f'{cy:.2f}')
                el.set('style', 'stroke:#000000;stroke-width:1.5px')
                self._reserialize(keep_eid=new_id)

            def _on_add_arrow(self):
                if self._view.svg_root is None:
                    return
                self._snapshot()
                # Ensure defs + arrowhead marker exist
                defs_el = self._view.svg_root.find(self._ns('defs'))
                if defs_el is None:
                    defs_el = _ET.SubElement(
                        self._view.svg_root, self._ns('defs'))
                    self._view.svg_root.insert(0, defs_el)
                marker_id = 'ann_arrowhead'
                if defs_el.find(
                    f'.//{self._ns("marker")}[@id="{marker_id}"]') is None:
                    mk = _ET.SubElement(defs_el, self._ns('marker'))
                    mk.set('id', marker_id)
                    mk.set('markerWidth', '8')
                    mk.set('markerHeight', '6')
                    mk.set('refX', '8')
                    mk.set('refY', '3')
                    mk.set('orient', 'auto')
                    poly = _ET.SubElement(mk, self._ns('polygon'))
                    poly.set('points', '0 0, 8 3, 0 6')
                    poly.set('fill', '#000000')

                new_id = self._unique_id('arrow_ann_')
                cx, cy = self._center_vb()
                half = self._view._vbox.width() * 0.1
                el = _ET.SubElement(self._view.svg_root, self._ns('line'))
                el.set('id', new_id)
                el.set('x1', f'{cx - half:.2f}')
                el.set('y1', f'{cy:.2f}')
                el.set('x2', f'{cx + half:.2f}')
                el.set('y2', f'{cy:.2f}')
                el.set('style',
                       f'stroke:#000000;stroke-width:1.5px;'
                       f'marker-end:url(#{marker_id})')
                self._reserialize(keep_eid=new_id)

            # ── palette ───────────────────────────────────────────────────
            def _on_apply_palette(self, colors):
                """Recolor fill/stroke of path-collection groups with palette."""
                if self._view.svg_root is None:
                    return
                import re as _re

                def _set_color(style_str, prop, color):
                    """Replace or insert a CSS property in a style string."""
                    pat = _re.compile(
                        rf'(?<![a-z-]){_re.escape(prop)}\s*:\s*[^;]+')
                    if pat.search(style_str):
                        return pat.sub(f'{prop}:{color}', style_str)
                    return style_str.rstrip(';') + f';{prop}:{color}'

                # Collect candidate elements: <g> groups that contain <path>
                # or <use> children — these are matplotlib series groups
                root = self._view.svg_root
                series_groups = []
                for g in root.iter(self._ns('g')):
                    gid = g.get('id', '')
                    # matplotlib PathCollection groups are named
                    # "PathCollection_N" or contain them as immediate children
                    children_tags = {c.tag for c in g}
                    if (any(t in children_tags for t in
                            (self._ns('path'), self._ns('use')))
                            and 'PathCollection' in gid):
                        series_groups.append(g)

                if not series_groups:
                    # Fallback: any top-level <g> with path/use children
                    for g in root:
                        if g.tag != self._ns('g'):
                            continue
                        children_tags = {c.tag for c in g}
                        if any(t in children_tags for t in
                               (self._ns('path'), self._ns('use'))):
                            series_groups.append(g)

                if not series_groups:
                    self._tip.setText("No data series found to recolor.")
                    return

                self._snapshot()
                for i, grp in enumerate(series_groups):
                    color = colors[i % len(colors)]
                    for child in grp:
                        style = child.get('style', '')
                        if not style:
                            continue
                        if 'fill' in style:
                            style = _set_color(style, 'fill', color)
                        stroke_m = _re.search(r'stroke\s*:\s*([^;]+)', style)
                        if stroke_m and 'none' not in stroke_m.group(1):
                            style = _set_color(style, 'stroke', color)
                        child.set('style', style)

                self._reserialize()
                self._tip.setText(
                    f"Palette applied to {len(series_groups)} series.")

            # ── reserialize ───────────────────────────────────────────────
            def _expand_canvas(self):
                """Expand the SVG canvas (viewBox, width/height, bg rect)
                to encompass all annotation elements."""
                root = self._view.svg_root
                if root is None:
                    return
                vb_str = root.get('viewBox', '')
                if not vb_str:
                    return
                try:
                    parts = vb_str.split()
                    vx, vy, vw, vh = (float(parts[0]), float(parts[1]),
                                      float(parts[2]), float(parts[3]))
                except (ValueError, IndexError):
                    return

                margin = 10
                min_x, min_y = vx, vy
                max_x, max_y = vx + vw, vy + vh

                for elem in root.iter():
                    eid = elem.get('id', '')
                    if not any(eid.startswith(p) for p in
                               ('annotation_', 'rect_ann_', 'ellipse_ann_',
                                'line_ann_', 'arrow_ann_')):
                        continue
                    tag = _ltag(elem)
                    # Also account for translate transforms on the element
                    tx_off, ty_off = 0.0, 0.0
                    t_str = elem.get('transform', '')
                    t_match = re.search(
                        r'translate\(([\d.e+-]+)[,\s]+([\d.e+-]+)\)', t_str)
                    if t_match:
                        tx_off = float(t_match.group(1))
                        ty_off = float(t_match.group(2))

                    if tag == 'rect':
                        ex = float(elem.get('x', 0)) + tx_off
                        ey = float(elem.get('y', 0)) + ty_off
                        ew = float(elem.get('width', 0))
                        eh = float(elem.get('height', 0))
                        min_x = min(min_x, ex - margin)
                        min_y = min(min_y, ey - margin)
                        max_x = max(max_x, ex + ew + margin)
                        max_y = max(max_y, ey + eh + margin)
                    elif tag == 'ellipse':
                        cx = float(elem.get('cx', 0)) + tx_off
                        cy = float(elem.get('cy', 0)) + ty_off
                        rx = float(elem.get('rx', 0))
                        ry = float(elem.get('ry', 0))
                        min_x = min(min_x, cx - rx - margin)
                        min_y = min(min_y, cy - ry - margin)
                        max_x = max(max_x, cx + rx + margin)
                        max_y = max(max_y, cy + ry + margin)
                    elif tag == 'line':
                        x1 = float(elem.get('x1', 0)) + tx_off
                        y1 = float(elem.get('y1', 0)) + ty_off
                        x2 = float(elem.get('x2', 0)) + tx_off
                        y2 = float(elem.get('y2', 0)) + ty_off
                        min_x = min(min_x, x1 - margin, x2 - margin)
                        min_y = min(min_y, y1 - margin, y2 - margin)
                        max_x = max(max_x, x1 + margin, x2 + margin)
                        max_y = max(max_y, y1 + margin, y2 + margin)
                    elif tag == 'text':
                        tx = float(elem.get('x', 0)) + tx_off
                        ty = float(elem.get('y', 0)) + ty_off
                        min_x = min(min_x, tx - margin)
                        min_y = min(min_y, ty - 20 - margin)
                        max_x = max(max_x, tx + 100 + margin)
                        max_y = max(max_y, ty + margin)

                new_vw = max_x - min_x
                new_vh = max_y - min_y
                if min_x < vx or min_y < vy or new_vw > vw or new_vh > vh:
                    root.set('viewBox',
                             f'{min_x:.2f} {min_y:.2f} {new_vw:.2f} {new_vh:.2f}')
                    # Update SVG width/height to match
                    root.set('width', f'{new_vw:.2f}pt')
                    root.set('height', f'{new_vh:.2f}pt')
                    # Expand the background rect (first <rect> in root, usually
                    # matplotlib's figure patch)
                    for child in root:
                        if _ltag(child) == 'rect':
                            child.set('x', f'{min_x:.2f}')
                            child.set('y', f'{min_y:.2f}')
                            child.set('width', f'{new_vw:.2f}')
                            child.set('height', f'{new_vh:.2f}')
                            break

            def _on_apply_padding(self):
                """Add user-specified padding around the SVG canvas."""
                root = self._view.svg_root
                if root is None:
                    return
                vb_str = root.get('viewBox', '')
                if not vb_str:
                    return
                try:
                    parts = vb_str.split()
                    vx, vy, vw, vh = (float(parts[0]), float(parts[1]),
                                      float(parts[2]), float(parts[3]))
                except (ValueError, IndexError):
                    return

                pt = self._pad_top.value()
                pb = self._pad_bottom.value()
                pl = self._pad_left.value()
                pr = self._pad_right.value()

                if pt == 0 and pb == 0 and pl == 0 and pr == 0:
                    return

                self._snapshot()
                new_x = vx - pl
                new_y = vy - pt
                new_w = vw + pl + pr
                new_h = vh + pt + pb

                root.set('viewBox',
                         f'{new_x:.2f} {new_y:.2f} {new_w:.2f} {new_h:.2f}')
                root.set('width', f'{new_w:.2f}pt')
                root.set('height', f'{new_h:.2f}pt')
                # Expand background rect
                for child in root:
                    if _ltag(child) == 'rect':
                        child.set('x', f'{new_x:.2f}')
                        child.set('y', f'{new_y:.2f}')
                        child.set('width', f'{new_w:.2f}')
                        child.set('height', f'{new_h:.2f}')
                        break

                # Reset spinboxes after applying
                self._pad_top.setValue(0)
                self._pad_bottom.setValue(0)
                self._pad_left.setValue(0)
                self._pad_right.setValue(0)

                self._reserialize()

            def _reserialize(self, keep_eid: str = None):
                if self._view._tree is None:
                    return
                self._expand_canvas()
                _ET.register_namespace('', _SVG_NS)
                _ET.register_namespace('xlink', _XLINK_NS)
                try:
                    buf = _io.BytesIO()
                    self._view._tree.write(
                        buf, encoding='utf-8', xml_declaration=True)
                    new_bytes = buf.getvalue()
                    self._svg_bytes = new_bytes
                    self._view.rerender(new_bytes, keep_eid=keep_eid)
                    n = self._view.overlay_count()
                    self._info.setText(f"{n} element(s)")
                    self.svg_modified.emit(
                        new_bytes.decode('utf-8', errors='replace'))
                except Exception as ex:
                    print(f'NodeSvgEditorWidget: rerender error: {ex}')
                    import traceback; traceback.print_exc()

            # ── NodeBaseWidget interface ───────────────────────────────────
            def get_value(self):  return ''
            def set_value(self, v): pass

        return _Widget(parent)


# ── Node ──────────────────────────────────────────────────────────────────────

class SvgEditorNode(BaseExecutionNode):
    """
    Converts an upstream matplotlib Figure to SVG for interactive element editing.

    Usage:
    - Click any highlighted element to select it.
    - Double-click to open the properties panel (fill, stroke, opacity, etc.).
    - Drag text labels (orange cursor) to reposition them.
    - Click "Apply" in the properties panel to commit changes.
    - Click "Reset SVG" to discard edits and reload from the figure.

    Edits are stored in the `_svg_data` node property and survive
    re-evaluation as long as the upstream figure is unchanged. Reset SVG
    clears them.

    Keywords: svg, vector edit, annotate figure, tweak labels, style paths, 向量圖, 圖形編輯, 標注, 顯示, 樣式
    """

    __identifier__ = 'nodes.display'
    NODE_NAME      = 'SVG Editor'
    PORT_SPEC      = {'inputs': ['figure'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('in',  color=PORT_COLORS['figure'])
        self.add_output('out', multi_output=True, color=PORT_COLORS['figure'])

        import NodeGraphQt
        self.create_property(
            '_svg_data', '',
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value)

        self._svg_widget = NodeSvgEditorWidget(self.view)
        self._svg_widget.svg_modified.connect(self._on_svg_modified)
        self._svg_widget.reset_requested.connect(self._on_reset)
        self.add_custom_widget(self._svg_widget)

    # ── property override: _svg_data must not trigger mark_dirty ──────────────
    def set_property(self, name, value, push_undo=True):
        if name == '_svg_data':
            # Skip BaseExecutionNode.set_property (which calls mark_dirty).
            # Go directly to NodeGraphQt.BaseNode.set_property.
            from NodeGraphQt.nodes.base_node import BaseNode
            BaseNode.set_property(self, name, value, push_undo)
            return
        super().set_property(name, value, push_undo)

    # ── private callbacks ─────────────────────────────────────────────────────
    def _on_svg_modified(self, new_svg: str):
        self.set_property('_svg_data', new_svg, push_undo=False)
        # Push edited SVG to downstream nodes
        svg_bytes = (new_svg.encode('utf-8')
                     if isinstance(new_svg, str) else new_svg)
        fig = self._get_upstream_fig()
        self.output_values['out'] = FigureData(
            payload=fig, svg_override=svg_bytes)
        self.mark_dirty()

    def _on_reset(self):
        """Clear stored edits and reload fresh SVG from the figure."""
        self.set_property('_svg_data', '', push_undo=False)
        fig = self._get_upstream_fig()
        if fig is not None:
            import matplotlib
            old_fonttype = matplotlib.rcParams.get('svg.fonttype', 'path')
            old_font = matplotlib.rcParams.get('font.family', ['sans-serif'])
            old_sans = matplotlib.rcParams.get('font.sans-serif', [])
            matplotlib.rcParams['svg.fonttype'] = 'none'
            matplotlib.rcParams['font.family'] = ['sans-serif']
            matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
            buf = _io.BytesIO()
            fig.savefig(buf, format='svg', bbox_inches='tight')
            matplotlib.rcParams['svg.fonttype'] = old_fonttype
            matplotlib.rcParams['font.family'] = old_font
            matplotlib.rcParams['font.sans-serif'] = old_sans
            svg_bytes = buf.getvalue()
            self.set_property('_svg_data', svg_bytes.decode('utf-8'),
                               push_undo=False)
            self.set_display(svg_bytes)

    def _get_upstream_fig(self):
        import matplotlib.figure
        p = self.inputs().get('in')
        if not p or not p.connected_ports():
            return None
        up  = p.connected_ports()[0]
        val = up.node().output_values.get(up.name())
        if val is None:
            return None
        if isinstance(val, FigureData):
            return val.payload
        if hasattr(val, 'payload') and isinstance(
                val.payload, matplotlib.figure.Figure):
            return val.payload
        if isinstance(val, matplotlib.figure.Figure):
            return val
        return None

    # ── evaluate ──────────────────────────────────────────────────────────────
    def evaluate(self):
        self.reset_progress()

        fig = self._get_upstream_fig()
        if fig is None:
            self.mark_error()
            return False, "No input figure connected"

        existing = self.get_property('_svg_data')
        if not existing:
            # First run or after reset: generate fresh SVG
            # Use 'none' fonttype so text is real <text> elements, not paths
            import matplotlib
            old_fonttype = matplotlib.rcParams.get('svg.fonttype', 'path')
            old_font = matplotlib.rcParams.get('font.family', ['sans-serif'])
            old_sans = matplotlib.rcParams.get('font.sans-serif', [])
            matplotlib.rcParams['svg.fonttype'] = 'none'
            matplotlib.rcParams['font.family'] = ['sans-serif']
            matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
            buf = _io.BytesIO()
            fig.savefig(buf, format='svg', bbox_inches='tight')
            matplotlib.rcParams['svg.fonttype'] = old_fonttype
            matplotlib.rcParams['font.family'] = old_font
            matplotlib.rcParams['font.sans-serif'] = old_sans
            svg_bytes = buf.getvalue()
            self.set_property('_svg_data',
                               svg_bytes.decode('utf-8'), push_undo=False)
            self.set_display(svg_bytes)
        else:
            # Subsequent runs: preserve edits
            raw = existing
            self.set_display(
                raw.encode('utf-8') if isinstance(raw, str) else raw)

        svg_data = self.get_property('_svg_data')
        if svg_data:
            svg_bytes = (svg_data.encode('utf-8')
                         if isinstance(svg_data, str) else svg_data)
            self.output_values['out'] = FigureData(
                payload=fig, svg_override=svg_bytes)
        else:
            self.output_values['out'] = FigureData(payload=fig)
        self.mark_clean()
        return True, None

    def _display_ui(self, data):
        """Load SVG bytes into the inline editor (Main Thread only)."""
        if isinstance(data, (bytes, bytearray)) and data.strip().startswith(b'<'):
            self._svg_widget.load_svg(bytes(data))
            self.view.draw_node()
