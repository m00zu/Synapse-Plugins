"""KFoldComboPickerNode -- K-fold CV ranking + auto-pick best (threshold, step_um).

Widget for manual override is added in Task 10. For now, auto-picks the top-ranked combo.
"""
from __future__ import annotations

import re

import pandas as pd
from nodes.base import BaseExecutionNode, PORT_COLORS
from data_models import TableData

from . import screen_core
from .data import PORT_TYPE_NAME, IMARIS_DATASET_COLOR
from .kfold_picker_widget import KFoldPickerWidget

PCT_RE = re.compile(r'^pct_above_(\d+)_at_(\d+)um$')

# Defensive: ensure port color is registered
PORT_COLORS.setdefault(PORT_TYPE_NAME, IMARIS_DATASET_COLOR)


class KFoldComboPickerNode(BaseExecutionNode):
    __identifier__ = 'plugins.Imaris3D.screen'
    NODE_NAME = 'K-Fold Combo Picker'

    PORT_SPEC = {
        'inputs':  ['dataset'],
        'outputs': ['ranking_table', 'chosen_combo'],
    }

    _UI_PROPS = frozenset({
        'ref_group', 'cmp_group', 'n_folds', 'n_seeds',
        'primary_test', 'primary_fold', 'fold_change_min',
        'override_threshold', 'override_step_um',
    })

    def __init__(self):
        super().__init__()
        self.add_input('dataset', color=PORT_COLORS.get(PORT_TYPE_NAME))
        self.add_output('ranking_table', color=PORT_COLORS.get('table'))
        self.add_output('chosen_combo', color=PORT_COLORS.get('table'))

        self.add_text_input('ref_group', 'Reference group', text='neg', tab='CV')
        self.add_text_input('cmp_group', 'Comparison group', text='pos', tab='CV')
        self._add_int_spinbox('n_folds', 'K folds', value=5, min_val=2, max_val=20, step=1, tab='CV')
        self._add_int_spinbox('n_seeds', 'Seeds', value=5, min_val=1, max_val=50, step=1, tab='CV')
        self.add_combo_menu('primary_test', 'Primary test',
                            items=['student', 'welch', 'mw'], tab='CV')
        self.add_combo_menu('primary_fold', 'Primary fold',
                            items=['median', 'mean'], tab='CV')
        self._add_float_spinbox('fold_change_min', 'Min |fold change|',
                                value=1.2, min_val=1.0, max_val=10.0, step=0.1, decimals=2,
                                tab='CV')
        self._add_int_spinbox('override_threshold', 'Override threshold',
                              value=-1, min_val=-1, max_val=255, step=1, tab='Override')
        self._add_int_spinbox('override_step_um', 'Override step_um',
                              value=-1, min_val=-1, max_val=100, step=1, tab='Override')

        # Inline heatmap widget
        self._picker_widget = KFoldPickerWidget(self.view, name='kfold_picker')
        self.add_custom_widget(self._picker_widget, tab='Heatmap')

    def _get_input(self, name: str):
        """Helper: fetch upstream output_values for a connected input port."""
        in_port = self.inputs().get(name)
        if not in_port or not in_port.connected_ports():
            return None
        upstream = in_port.connected_ports()[0]
        return upstream.node().output_values.get(upstream.name())

    def evaluate(self) -> tuple[bool, str | None]:
        ds = self._get_input('dataset')
        if ds is None or not ds.entries:
            return False, 'No dataset on input'

        long_df = ds.to_long_per_cell()
        if long_df.empty:
            return False, 'Dataset has no cells (all excluded?)'

        thresholds, steps = set(), set()
        for col in long_df.columns:
            m = PCT_RE.match(col)
            if m:
                thresholds.add(int(m.group(1)))
                steps.add(int(m.group(2)))
        if not thresholds or not steps:
            return False, 'Dataset contains no pct_above_* columns'

        ranking = screen_core.run_kfold_ranking(
            long_df,
            thresholds=sorted(thresholds),
            steps=sorted(steps),
            ref_group=self.get_property('ref_group'),
            cmp_group=self.get_property('cmp_group'),
            k=int(self.get_property('n_folds')),
            seeds=range(int(self.get_property('n_seeds'))),
            primary_test=self.get_property('primary_test'),
            primary_fold=self.get_property('primary_fold'),
            fold_change_min=float(self.get_property('fold_change_min')),
        )
        if ranking.empty:
            return False, 'K-fold produced no valid combos'

        ov_t = int(self.get_property('override_threshold'))
        ov_s = int(self.get_property('override_step_um'))
        if ov_t >= 0 and ov_s >= 0:
            chosen = pd.DataFrame([{'threshold': ov_t, 'step_um': ov_s}])
        else:
            top = ranking.iloc[0]
            chosen = pd.DataFrame([{'threshold': int(top['threshold']),
                                    'step_um': int(top['step_um'])}])

        # Push ranking + chosen combo to the inline widget
        chosen_tuple = (int(chosen.iloc[0]['threshold']), int(chosen.iloc[0]['step_um']))
        try:
            self._picker_widget.set_ranking(ranking, chosen_tuple)
        except Exception:
            pass

        self.output_values['ranking_table'] = TableData(payload=ranking)
        self.output_values['chosen_combo'] = TableData(payload=chosen)
        self.mark_clean()
        return True, (
            f"chosen: threshold={chosen.iloc[0]['threshold']}, "
            f"step_um={chosen.iloc[0]['step_um']}"
        )
