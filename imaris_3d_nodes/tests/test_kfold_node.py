"""Tests for KFoldComboPickerNode."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SYNAPSE = Path('/Users/s/Desktop/demo/PySide_Node/synapse')
if str(_SYNAPSE) not in sys.path:
    sys.path.insert(0, str(_SYNAPSE))


def _make_corrected_csv(path: Path, group: str, file_idx: int, hi_mean: float, rng):
    rows = []
    for c in range(1, 21):
        rows.append({
            'cell': c,
            'pct_above_12_at_3um': float(rng.normal(hi_mean, 0.05)),
            'pct_above_24_at_3um': float(rng.normal(0.05, 0.05)),
        })
    rows.append({'cell': 'BG', 'pct_above_12_at_3um': 0.0, 'pct_above_24_at_3um': 0.0})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_kfold_node_picks_correct_combo(tmp_path, qapp):
    """End-to-end: LoadImarisDatasetNode -> KFoldComboPickerNode in a graph."""
    import NodeGraphQt
    from imaris_3d_nodes.load_dataset_node import LoadImarisDatasetNode
    from imaris_3d_nodes.kfold_picker_node import KFoldComboPickerNode

    # Set up synthetic dataset: neg group low values, pos group high at (12, 3)
    rng = np.random.default_rng(7)
    (tmp_path / 'neg').mkdir()
    (tmp_path / 'pos').mkdir()
    for i in range(5):
        _make_corrected_csv(tmp_path / 'neg' / f'F{i}_corrected.csv', 'neg', i, 0.10, rng)
        _make_corrected_csv(tmp_path / 'pos' / f'F{i}_corrected.csv', 'pos', i, 0.50, rng)

    g = NodeGraphQt.NodeGraph()
    g.register_node(LoadImarisDatasetNode)
    g.register_node(KFoldComboPickerNode)

    loader = g.create_node('plugins.Imaris3D.io.LoadImarisDatasetNode')
    loader.set_property('dataset_dir', str(tmp_path))
    loader.set_property('layout', 'auto')
    ok, msg = loader.evaluate()
    assert ok, msg

    kfold = g.create_node('plugins.Imaris3D.screen.KFoldComboPickerNode')
    # Connect loader.dataset -> kfold.dataset
    loader.set_output(0, kfold.input(0))
    kfold.set_property('ref_group', 'neg')
    kfold.set_property('cmp_group', 'pos')
    kfold.set_property('n_folds', 2)
    kfold.set_property('n_seeds', 3)
    kfold.set_property('primary_test', 'student')
    kfold.set_property('primary_fold', 'median')

    ok, msg = kfold.evaluate()
    assert ok, msg

    chosen = kfold.output_values.get('chosen_combo')
    assert chosen is not None
    assert (int(chosen.payload['threshold'].iloc[0]),
            int(chosen.payload['step_um'].iloc[0])) == (12, 3)

    ranking = kfold.output_values.get('ranking_table')
    assert ranking is not None
    assert len(ranking.payload) == 2
