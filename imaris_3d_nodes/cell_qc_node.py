"""CellQCFilterNode -- inline widget to click cells out of a dataset.

Persistent state lives in the `excluded_cells_json` property
({file_stem: [cell_ids]}); the widget mutates it, evaluate() applies it.
"""
from __future__ import annotations

import json

from nodes.base import BaseExecutionNode, PORT_COLORS

from .cell_qc_widget import CellQCWidget
from .data import PORT_TYPE_NAME, IMARIS_DATASET_COLOR

PORT_COLORS.setdefault(PORT_TYPE_NAME, IMARIS_DATASET_COLOR)


class CellQCFilterNode(BaseExecutionNode):
    """Lets the user exclude per-file outlier cells before downstream stats."""

    __identifier__ = 'plugins.Imaris3D.qc'
    NODE_NAME = 'Cell QC Filter'

    PORT_SPEC = {'inputs': ['dataset'], 'outputs': ['dataset']}

    _UI_PROPS = frozenset({'excluded_cells_json'})

    def __init__(self):
        super().__init__()
        self.add_input('dataset', color=PORT_COLORS.get(PORT_TYPE_NAME))
        self.add_output('dataset', color=PORT_COLORS.get(PORT_TYPE_NAME))

        # Persistent state: JSON-encoded {file_stem: [cell_ids]}
        self.add_text_input('excluded_cells_json', 'Excluded cells (JSON)',
                            text='{}', tab='State')

        # Inline viewer
        self._qc_widget = CellQCWidget(self.view, name='cell_qc')
        self.add_custom_widget(self._qc_widget, tab='Viewer')

    def _get_input(self, name: str):
        in_port = self.inputs().get(name)
        if not in_port or not in_port.connected_ports():
            return None
        upstream = in_port.connected_ports()[0]
        return upstream.node().output_values.get(upstream.name())

    def evaluate(self) -> tuple[bool, str | None]:
        ds = self._get_input('dataset')
        if ds is None or not ds.entries:
            return False, 'No dataset on input'

        # Parse the persistent exclusion map
        raw = self.get_property('excluded_cells_json') or '{}'
        try:
            excluded_map = json.loads(raw)
            if not isinstance(excluded_map, dict):
                excluded_map = {}
        except (TypeError, ValueError):
            excluded_map = {}

        # Apply to each entry
        for entry in ds.entries:
            ids = excluded_map.get(entry.file_stem, [])
            try:
                entry.excluded_cells = set(int(c) for c in ids)
            except (TypeError, ValueError):
                entry.excluded_cells = set()

        # Push to widget for display (may not be attached in headless tests)
        try:
            self._qc_widget.load_dataset(ds, excluded_map)
        except Exception:
            pass

        self.output_values['dataset'] = ds
        self.mark_clean()
        return True, f'Excluded {ds.total_excluded()} / {ds.total_cells()} cells'
