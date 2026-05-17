"""BlankNormalizeNode -- blank cells, optionally normalize to a reference group."""
from __future__ import annotations

from nodes.base import BaseExecutionNode, PORT_COLORS
from data_models import TableData

from . import blank_core
from .data import PORT_TYPE_NAME, IMARIS_DATASET_COLOR

PORT_COLORS.setdefault(PORT_TYPE_NAME, IMARIS_DATASET_COLOR)


class BlankNormalizeNode(BaseExecutionNode):
    __identifier__ = 'plugins.Imaris3D.apply'
    NODE_NAME = 'Blank + Normalize'

    # Use TYPE names in PORT_SPEC (matches PORT_COLORS keys) so the
    # Node Explorer tree icon renders the correct colors.
    PORT_SPEC = {
        'inputs':  ['imaris_dataset', 'table'],
        'outputs': ['table'],
    }

    _UI_PROPS = frozenset({'reference_group', 'normalize'})

    def __init__(self):
        super().__init__()
        self.add_input('imaris_dataset', color=PORT_COLORS.get(PORT_TYPE_NAME))
        self.add_input('chosen_combo', color=PORT_COLORS.get('table'))
        self.add_output('wide_table', color=PORT_COLORS.get('table'))

        self.add_text_input('reference_group', 'Reference group', text='', tab='Settings')
        self.add_checkbox('normalize', 'Normalize',
                          text='Divide by reference group mean',
                          state=True, tab='Settings')

    def _get_input(self, name: str):
        in_port = self.inputs().get(name)
        if not in_port or not in_port.connected_ports():
            return None
        upstream = in_port.connected_ports()[0]
        return upstream.node().output_values.get(upstream.name())

    def evaluate(self) -> tuple[bool, str | None]:
        ds = self._get_input('imaris_dataset')
        if ds is None or not ds.entries:
            return False, 'No dataset on input'

        chosen = self._get_input('chosen_combo')
        if chosen is None or chosen.payload.empty:
            return False, 'No chosen_combo on input (run K-Fold first)'

        threshold = int(chosen.payload['threshold'].iloc[0])
        step_um   = int(chosen.payload['step_um'].iloc[0])

        wide = ds.to_wide_blanked(threshold=threshold, step_um=step_um)
        if wide.empty:
            return False, f'No data for pct_above_{threshold}_at_{step_um}um'

        if self.get_property('normalize'):
            ref = self.get_property('reference_group')
            if ref and ref in wide.columns:
                wide = blank_core.normalize_by_reference(wide, ref_group=ref)

        self.output_values['wide_table'] = TableData(payload=wide)
        self.mark_clean()
        return True, f'{len(wide.columns)} groups, n_max={len(wide)}'
