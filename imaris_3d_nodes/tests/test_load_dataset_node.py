"""Tests for LoadImarisDatasetNode."""
import sys
from pathlib import Path

import pandas as pd

_SYNAPSE = Path('/Users/s/Desktop/demo/PySide_Node/synapse')
if str(_SYNAPSE) not in sys.path:
    sys.path.insert(0, str(_SYNAPSE))


def _make_corrected_csv(path: Path, n_cells: int = 3):
    rows = []
    for i in range(1, n_cells + 1):
        rows.append({'cell': i, 'pct_above_12_at_3um': 0.1 * i})
    rows.append({'cell': 'BG', 'pct_above_12_at_3um': 0.05})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_dataset_node_subfolder_layout(tmp_path, qapp):
    from imaris_3d_nodes.load_dataset_node import LoadImarisDatasetNode

    (tmp_path / 'neg').mkdir()
    (tmp_path / 'pos').mkdir()
    _make_corrected_csv(tmp_path / 'neg' / 'F001_corrected.csv', 3)
    _make_corrected_csv(tmp_path / 'neg' / 'F002_corrected.csv', 2)
    _make_corrected_csv(tmp_path / 'pos' / 'F003_corrected.csv', 4)

    node = LoadImarisDatasetNode()
    node.set_property('dataset_dir', str(tmp_path))
    node.set_property('layout', 'auto')

    ok, msg = node.evaluate()
    assert ok, msg
    ds = node.output_values.get('imaris_dataset')
    assert ds is not None
    assert len(ds.entries) == 3
    assert {e.group for e in ds.entries} == {'neg', 'pos'}
    assert {e.file_stem for e in ds.entries} == {'F001', 'F002', 'F003'}


def test_load_dataset_node_errors_on_empty_dir(tmp_path, qapp):
    from imaris_3d_nodes.load_dataset_node import LoadImarisDatasetNode
    node = LoadImarisDatasetNode()
    node.set_property('dataset_dir', str(tmp_path))
    ok, msg = node.evaluate()
    assert not ok
    assert 'No *_corrected.csv' in msg
