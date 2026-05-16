"""Inline widget for CellQCFilterNode.

Layout (3 regions):
  Files tree (left)  --  Composite preview (right, scrollable)
  Cells table (bottom row, full width)

The widget DOES NOT own the exclusion state -- it mutates the node's
excluded_cells_json property dict.  The node's evaluate() applies that
property to incoming dataset entries.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from NodeGraphQt import NodeBaseWidget

from .data import ImarisDatasetData, ImarisDatasetEntry


def _common_prefix_strip(names: list[str]) -> dict[str, str]:
    """Return {full: short} stripping the longest common prefix (word boundary aware)."""
    if not names:
        return {}
    prefix = os.path.commonprefix(names)
    if len(prefix) < 4:
        return {n: n for n in names}
    for stop in '_- ':
        cut = prefix.rfind(stop)
        if cut >= 4:
            prefix = prefix[: cut + 1]
            break
    return {n: (n[len(prefix):] or n) for n in names}


class _CompositeView(QtWidgets.QGraphicsView):
    """Pannable, zoomable composite view with clickable bbox overlays."""
    cell_clicked = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.setTransformationAnchor(self.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(self.DragMode.ScrollHandDrag)
        self._pixmap_item: QtWidgets.QGraphicsPixmapItem | None = None
        self._bbox_items: dict[int, QtWidgets.QGraphicsRectItem] = {}
        self._excluded: set[int] = set()

    def set_composite(self, pixmap: QtGui.QPixmap | None):
        self._scene.clear()
        self._bbox_items = {}
        if pixmap is None or pixmap.isNull():
            text = self._scene.addText('(composite not found)')
            text.setDefaultTextColor(QtGui.QColor('gray'))
            self._pixmap_item = None
            return
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect().toRectF())
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def set_cells(self, cells: pd.DataFrame, excluded: Iterable[int]):
        for it in self._bbox_items.values():
            self._scene.removeItem(it)
        self._bbox_items = {}
        self._excluded = set(int(c) for c in excluded)

        if self._pixmap_item is None or cells.empty:
            return

        for _, r in cells.iterrows():
            try:
                cid = int(r['cell'])
                x0, x1 = int(r['bbox_min_x_px']), int(r['bbox_max_x_px'])
                y0, y1 = int(r['bbox_min_y_px']), int(r['bbox_max_y_px'])
            except (KeyError, ValueError, TypeError):
                continue
            rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
            item = QtWidgets.QGraphicsRectItem(rect)
            item.setData(0, cid)
            item.setAcceptHoverEvents(True)
            item.setToolTip(f'cell {cid}')
            self._update_bbox_pen(item, cid in self._excluded)
            self._scene.addItem(item)
            self._bbox_items[cid] = item

    def set_excluded(self, excluded: Iterable[int]):
        new_excl = set(int(c) for c in excluded)
        for cid, item in self._bbox_items.items():
            self._update_bbox_pen(item, cid in new_excl)
        self._excluded = new_excl

    def _update_bbox_pen(self, item, is_excluded: bool):
        if is_excluded:
            pen = QtGui.QPen(QtGui.QColor(220, 30, 30), 3)
        else:
            pen = QtGui.QPen(QtGui.QColor(0, 255, 0, 128), 2)
        pen.setStyle(QtCore.Qt.PenStyle.SolidLine)
        item.setPen(pen)
        item.setBrush(QtCore.Qt.GlobalColor.transparent)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())
            hits = [it for it in self._scene.items(pos)
                    if isinstance(it, QtWidgets.QGraphicsRectItem)]
            if hits:
                hits.sort(key=lambda i: i.rect().width() * i.rect().height())
                cid = hits[0].data(0)
                if isinstance(cid, int):
                    self.cell_clicked.emit(cid)
                    return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        zoom = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        self.scale(zoom, zoom)


class _PixmapCache:
    """Tiny LRU cache for decoded composite QPixmaps."""

    def __init__(self, maxsize: int = 5):
        self._max = maxsize
        self._order: list[str] = []
        self._cache: dict[str, QtGui.QPixmap] = {}

    def get(self, path: Path) -> QtGui.QPixmap:
        key = str(path)
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        pm = QtGui.QPixmap(key)
        if pm.isNull():
            return pm
        self._cache[key] = pm
        self._order.append(key)
        while len(self._order) > self._max:
            evict = self._order.pop(0)
            self._cache.pop(evict, None)
        return pm


class CellQCWidget(NodeBaseWidget):
    """Inline NodeGraphQt widget that the QC node embeds."""

    def __init__(self, parent=None, name='cell_qc', label=''):
        super().__init__(parent, name, label)

        self._dataset: ImarisDatasetData | None = None
        self._current_entry: ImarisDatasetEntry | None = None
        self._excluded_map: dict[str, set[int]] = {}
        self._stem_to_display: dict[str, str] = {}
        self._pixmaps = _PixmapCache(maxsize=5)

        container = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(container)
        main_layout.setContentsMargins(2, 2, 2, 2)

        # Top row: file tree + composite
        top = QtWidgets.QHBoxLayout()

        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._search = QtWidgets.QLineEdit()
        self._search.setPlaceholderText('search files...')
        left_layout.addWidget(self._search)
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels(['File', 'Excluded'])
        self._tree.setRootIsDecorated(True)
        self._tree.setSelectionMode(self._tree.SelectionMode.SingleSelection)
        left_layout.addWidget(self._tree)
        self._total_label = QtWidgets.QLabel('Total: 0 / 0 excluded')
        left_layout.addWidget(self._total_label)
        left_widget.setMaximumWidth(280)
        top.addWidget(left_widget)

        self._composite = _CompositeView()
        top.addWidget(self._composite, stretch=1)

        main_layout.addLayout(top, stretch=2)

        # Bottom row
        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(['ID', 'pct_above', 'centroid', 'status'])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(self._table.SelectionBehavior.SelectRows)
        main_layout.addWidget(self._table, stretch=1)

        controls = QtWidgets.QHBoxLayout()
        self._clear_current_btn = QtWidgets.QPushButton('Clear current')
        self._clear_all_btn = QtWidgets.QPushButton('Clear all')
        self._auto_thr_spin = QtWidgets.QDoubleSpinBox()
        self._auto_thr_spin.setRange(-100.0, 100.0)
        self._auto_thr_spin.setValue(0.95)
        self._auto_apply_btn = QtWidgets.QPushButton('Auto-exclude > value')
        controls.addWidget(self._clear_current_btn)
        controls.addWidget(self._clear_all_btn)
        controls.addStretch(1)
        controls.addWidget(QtWidgets.QLabel('threshold:'))
        controls.addWidget(self._auto_thr_spin)
        controls.addWidget(self._auto_apply_btn)
        main_layout.addLayout(controls)

        self.set_custom_widget(container)

        # Wiring
        self._tree.currentItemChanged.connect(self._on_tree_selection_changed)
        self._composite.cell_clicked.connect(self._toggle_cell)
        self._table.cellClicked.connect(self._on_table_clicked)
        self._search.textChanged.connect(self._on_search)
        self._clear_current_btn.clicked.connect(self._clear_current)
        self._clear_all_btn.clicked.connect(self._clear_all)
        self._auto_apply_btn.clicked.connect(self._auto_exclude)

    # ── Public API ──────────────────────────────────────────────
    def load_dataset(self, dataset: ImarisDatasetData, excluded_map: dict):
        self._dataset = dataset
        self._excluded_map = {k: set(v) for k, v in (excluded_map or {}).items()}
        self._rebuild_tree()
        self._refresh_totals()

    def get_excluded_map(self) -> dict[str, list[int]]:
        return {k: sorted(v) for k, v in self._excluded_map.items() if v}

    # ── Tree population ─────────────────────────────────────────
    def _rebuild_tree(self):
        self._tree.clear()
        if self._dataset is None:
            return
        all_stems = [e.file_stem for e in self._dataset.entries]
        self._stem_to_display = _common_prefix_strip(all_stems)

        groups: dict[str, list[ImarisDatasetEntry]] = {}
        for e in self._dataset.entries:
            groups.setdefault(e.group, []).append(e)

        for group_name, entries in groups.items():
            group_item = QtWidgets.QTreeWidgetItem(
                self._tree, [f'{group_name} ({len(entries)})', ''])
            group_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, ('group', group_name))
            group_item.setFlags(group_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
            for entry in entries:
                short = self._stem_to_display.get(entry.file_stem, entry.file_stem)
                excl_count = len(self._excluded_map.get(entry.file_stem, set()))
                child = QtWidgets.QTreeWidgetItem(
                    group_item, [short, f'{excl_count}/{entry.n_cells}'])
                child.setToolTip(0, entry.file_stem)
                child.setData(0, QtCore.Qt.ItemDataRole.UserRole, ('file', entry.file_stem))
            group_item.setExpanded(True)

    def _refresh_totals(self):
        if self._dataset is None:
            self._total_label.setText('Total: 0 / 0 excluded')
            return
        total = self._dataset.total_cells()
        excl = sum(len(s) for s in self._excluded_map.values())
        self._total_label.setText(f'Total: {excl} / {total} excluded')

    # ── Selection / display ─────────────────────────────────────
    def _on_tree_selection_changed(self, current, _previous):
        if current is None:
            return
        kind_stem = current.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not kind_stem or kind_stem[0] != 'file':
            return
        stem = kind_stem[1]
        if self._dataset is None:
            return
        entry = self._dataset.by_stem(stem)
        if entry is None:
            return
        self._current_entry = entry
        pixmap = self._pixmaps.get(entry.composite_path)
        self._composite.set_composite(pixmap if not pixmap.isNull() else None)
        excluded = self._excluded_map.get(stem, set())
        self._composite.set_cells(entry.per_cell_table, excluded)
        self._populate_table(entry, excluded)

    def _populate_table(self, entry: ImarisDatasetEntry, excluded: set[int]):
        df = entry.per_cell_table
        self._table.setRowCount(len(df))
        pct_cols = [c for c in df.columns if c.startswith('pct_above_')]
        pct_col = pct_cols[0] if pct_cols else None
        for i, (_, r) in enumerate(df.iterrows()):
            cid = int(r['cell'])
            cx = float(r.get('centroid_x_px', 0))
            cy = float(r.get('centroid_y_px', 0))
            pct = float(r[pct_col]) if pct_col else float('nan')
            self._table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(cid)))
            self._table.setItem(i, 1, QtWidgets.QTableWidgetItem(f'{pct:.3f}'))
            self._table.setItem(i, 2, QtWidgets.QTableWidgetItem(f'({cx:.0f},{cy:.0f})'))
            status = 'EXCLUDED' if cid in excluded else 'kept'
            item = QtWidgets.QTableWidgetItem(status)
            if cid in excluded:
                item.setForeground(QtGui.QColor(220, 30, 30))
            self._table.setItem(i, 3, item)

    # ── Click handlers ──────────────────────────────────────────
    def _toggle_cell(self, cell_id: int):
        if self._current_entry is None:
            return
        stem = self._current_entry.file_stem
        excl = self._excluded_map.setdefault(stem, set())
        if cell_id in excl:
            excl.remove(cell_id)
        else:
            excl.add(cell_id)
        self._notify_node()
        self._composite.set_excluded(excl)
        self._populate_table(self._current_entry, excl)
        self._refresh_tree_counts()
        self._refresh_totals()

    def _on_table_clicked(self, row: int, _col: int):
        item = self._table.item(row, 0)
        if item is None:
            return
        try:
            cid = int(item.text())
        except ValueError:
            return
        self._toggle_cell(cid)

    def _clear_current(self):
        if self._current_entry is None:
            return
        self._excluded_map[self._current_entry.file_stem] = set()
        self._notify_node()
        self._composite.set_excluded(set())
        self._populate_table(self._current_entry, set())
        self._refresh_tree_counts()
        self._refresh_totals()

    def _clear_all(self):
        self._excluded_map = {}
        self._notify_node()
        if self._current_entry is not None:
            self._composite.set_excluded(set())
            self._populate_table(self._current_entry, set())
        self._refresh_tree_counts()
        self._refresh_totals()

    def _auto_exclude(self):
        if self._current_entry is None:
            return
        thr = float(self._auto_thr_spin.value())
        df = self._current_entry.per_cell_table
        pct_cols = [c for c in df.columns if c.startswith('pct_above_')]
        if not pct_cols:
            return
        col = pct_cols[0]
        new_ids = set(int(c) for c in df[df[col] > thr]['cell'])
        excl = self._excluded_map.setdefault(self._current_entry.file_stem, set())
        excl |= new_ids
        self._notify_node()
        self._composite.set_excluded(excl)
        self._populate_table(self._current_entry, excl)
        self._refresh_tree_counts()
        self._refresh_totals()

    def _on_search(self, text: str):
        text = text.strip().lower()
        for i in range(self._tree.topLevelItemCount()):
            group_item = self._tree.topLevelItem(i)
            any_visible = False
            for j in range(group_item.childCount()):
                child = group_item.child(j)
                stem = child.data(0, QtCore.Qt.ItemDataRole.UserRole)[1]
                vis = (text == '') or (text in stem.lower())
                child.setHidden(not vis)
                any_visible = any_visible or vis
            group_item.setHidden(not any_visible)

    def _refresh_tree_counts(self):
        for i in range(self._tree.topLevelItemCount()):
            group_item = self._tree.topLevelItem(i)
            for j in range(group_item.childCount()):
                child = group_item.child(j)
                stem = child.data(0, QtCore.Qt.ItemDataRole.UserRole)[1]
                entry = self._dataset.by_stem(stem) if self._dataset else None
                if entry is None:
                    continue
                excl = len(self._excluded_map.get(stem, set()))
                child.setText(1, f'{excl}/{entry.n_cells}')

    def _notify_node(self):
        node = self.node
        if node is None:
            return
        node.set_property('excluded_cells_json',
                          json.dumps(self.get_excluded_map()))
        node.mark_dirty()

    # ── NodeBaseWidget contract ────────────────────────────────
    def get_value(self):
        return json.dumps(self.get_excluded_map())

    def set_value(self, value):
        try:
            data = json.loads(value or '{}')
            self._excluded_map = {k: set(int(c) for c in v) for k, v in data.items()}
        except (TypeError, ValueError):
            self._excluded_map = {}
        self._rebuild_tree()
        self._refresh_totals()
