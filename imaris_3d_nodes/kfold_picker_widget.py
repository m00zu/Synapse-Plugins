"""Inline heatmap picker for the K-Fold node.

Displays a (threshold x step_um) heatmap colored by mean_neglog10p, with
non-passing cells marked. Click a cell to override the auto-pick; the click
writes the chosen combo back to the node's override_threshold / override_step_um
properties.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.patches
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from NodeGraphQt import NodeBaseWidget


class _HeatmapCanvas(FigureCanvasQTAgg):
    cell_clicked = QtCore.Signal(int, int)  # threshold, step_um

    def __init__(self, parent=None):
        self._fig = Figure(figsize=(4.5, 3.5), tight_layout=True)
        super().__init__(self._fig)
        self.setParent(parent)
        self._ax = self._fig.add_subplot(111)
        self._thresholds: list[int] = []
        self._steps: list[int] = []
        self.mpl_connect('button_press_event', self._on_click)

    def set_ranking(self, df: pd.DataFrame, chosen: tuple[int, int] | None = None):
        self._ax.clear()
        if df is None or df.empty:
            self._ax.set_title('No data')
            self.draw_idle()
            return

        self._thresholds = sorted(df['threshold'].unique().tolist())
        self._steps = sorted(df['step_um'].unique().tolist())
        grid = np.full((len(self._steps), len(self._thresholds)), np.nan)
        for _, r in df.iterrows():
            si = self._steps.index(int(r['step_um']))
            ti = self._thresholds.index(int(r['threshold']))
            grid[si, ti] = r['mean_neglog10p']

        im = self._ax.imshow(grid, aspect='auto', origin='lower',
                             cmap='viridis', interpolation='nearest')
        self._ax.set_xticks(range(len(self._thresholds)))
        self._ax.set_xticklabels(self._thresholds)
        self._ax.set_yticks(range(len(self._steps)))
        self._ax.set_yticklabels(self._steps)
        self._ax.set_xlabel('threshold')
        self._ax.set_ylabel('step_um')
        self._fig.colorbar(im, ax=self._ax, label='mean -log10 p')

        # Overlay X for non-passing
        for _, r in df.iterrows():
            if not bool(r['passes_filter']):
                si = self._steps.index(int(r['step_um']))
                ti = self._thresholds.index(int(r['threshold']))
                self._ax.plot(ti, si, marker='x', color='red', ms=8, mew=2)

        # Highlight chosen
        if chosen is not None:
            ct, cs = chosen
            if ct in self._thresholds and cs in self._steps:
                ti = self._thresholds.index(ct)
                si = self._steps.index(cs)
                self._ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (ti - 0.5, si - 0.5), 1, 1,
                        edgecolor='white', linewidth=3, facecolor='none',
                    )
                )
        self.draw_idle()

    def _on_click(self, event):
        if event.inaxes is not self._ax:
            return
        if not self._thresholds or not self._steps:
            return
        ti = int(round(event.xdata or 0))
        si = int(round(event.ydata or 0))
        if 0 <= ti < len(self._thresholds) and 0 <= si < len(self._steps):
            self.cell_clicked.emit(self._thresholds[ti], self._steps[si])


class KFoldPickerWidget(NodeBaseWidget):
    """Wrapped NodeGraphQt widget that the node embeds."""

    def __init__(self, parent=None, name='kfold_picker', label=''):
        super().__init__(parent, name=name, label=label)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self._chosen_label = QtWidgets.QLabel('No combo picked yet')
        self._chosen_label.setStyleSheet('font-weight: bold; padding: 4px;')
        layout.addWidget(self._chosen_label)

        self._canvas = _HeatmapCanvas()
        layout.addWidget(self._canvas)

        self._reset_btn = QtWidgets.QPushButton('Reset to auto-pick')
        layout.addWidget(self._reset_btn)

        self.set_custom_widget(container)

        self._canvas.cell_clicked.connect(self._on_cell_clicked)
        self._reset_btn.clicked.connect(self._on_reset)

    def set_ranking(self, df: pd.DataFrame | None, chosen: tuple[int, int] | None):
        self._canvas.set_ranking(df, chosen)
        if chosen is not None:
            self._chosen_label.setText(
                f'Chosen: threshold={chosen[0]}  step_um={chosen[1]}'
            )

    def _on_cell_clicked(self, threshold: int, step_um: int):
        node = self.node
        if node is None:
            return
        node.set_property('override_threshold', threshold)
        node.set_property('override_step_um', step_um)
        node.mark_dirty()
        self._chosen_label.setText(
            f'Chosen (override): threshold={threshold}  step_um={step_um}'
        )

    def _on_reset(self):
        node = self.node
        if node is None:
            return
        node.set_property('override_threshold', -1)
        node.set_property('override_step_um', -1)
        node.mark_dirty()
        self._chosen_label.setText('Reset to auto-pick (re-run graph)')

    def get_value(self):
        return None

    def set_value(self, value):
        pass
