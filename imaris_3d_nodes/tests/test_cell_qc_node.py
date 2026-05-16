"""Tests for CellQCFilterNode (logic only -- widget tested manually)."""
import json
import sys
from pathlib import Path

import pandas as pd

_SYNAPSE = Path('/Users/s/Desktop/demo/PySide_Node/synapse')
if str(_SYNAPSE) not in sys.path:
    sys.path.insert(0, str(_SYNAPSE))


def _make_corrected_csv(path: Path, vals: list[float]):
    rows = [{'cell': i + 1, 'pct_above_12_at_3um': v} for i, v in enumerate(vals)]
    rows.append({'cell': 'BG', 'pct_above_12_at_3um': 0.05})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_qc_node_applies_excluded_cells_json(tmp_path, qapp):
    import NodeGraphQt
    from imaris_3d_nodes.load_dataset_node import LoadImarisDatasetNode
    from imaris_3d_nodes.cell_qc_node import CellQCFilterNode

    _make_corrected_csv(tmp_path / 'F1_corrected.csv', [0.1, 0.2, 0.3])

    g = NodeGraphQt.NodeGraph()
    g.register_node(LoadImarisDatasetNode)
    g.register_node(CellQCFilterNode)

    loader = g.create_node('plugins.Imaris3D.io.LoadImarisDatasetNode')
    loader.set_property('dataset_dir', str(tmp_path))
    loader.set_property('layout', 'auto')
    loader.set_property('default_group', 'neg')
    ok, _ = loader.evaluate()
    assert ok

    qc = g.create_node('plugins.Imaris3D.qc.CellQCFilterNode')
    loader.set_output(0, qc.input(0))
    qc.set_property('excluded_cells_json', json.dumps({'F1': [2]}))

    ok, msg = qc.evaluate()
    assert ok, msg

    out_ds = qc.output_values.get('dataset')
    assert out_ds is not None
    assert out_ds.entries[0].excluded_cells == {2}
    assert out_ds.total_excluded() == 1


def test_qc_node_passes_through_when_no_exclusions(tmp_path, qapp):
    import NodeGraphQt
    from imaris_3d_nodes.load_dataset_node import LoadImarisDatasetNode
    from imaris_3d_nodes.cell_qc_node import CellQCFilterNode

    _make_corrected_csv(tmp_path / 'F1_corrected.csv', [0.1, 0.2, 0.3])

    g = NodeGraphQt.NodeGraph()
    g.register_node(LoadImarisDatasetNode)
    g.register_node(CellQCFilterNode)

    loader = g.create_node('plugins.Imaris3D.io.LoadImarisDatasetNode')
    loader.set_property('dataset_dir', str(tmp_path))
    loader.set_property('layout', 'auto')
    loader.set_property('default_group', 'neg')
    loader.evaluate()

    qc = g.create_node('plugins.Imaris3D.qc.CellQCFilterNode')
    loader.set_output(0, qc.input(0))
    qc.set_property('excluded_cells_json', '{}')

    ok, _ = qc.evaluate()
    assert ok
    out_ds = qc.output_values.get('dataset')
    assert out_ds.total_excluded() == 0


def test_qc_node_handles_invalid_json(tmp_path, qapp):
    import NodeGraphQt
    from imaris_3d_nodes.load_dataset_node import LoadImarisDatasetNode
    from imaris_3d_nodes.cell_qc_node import CellQCFilterNode

    _make_corrected_csv(tmp_path / 'F1_corrected.csv', [0.1, 0.2, 0.3])

    g = NodeGraphQt.NodeGraph()
    g.register_node(LoadImarisDatasetNode)
    g.register_node(CellQCFilterNode)

    loader = g.create_node('plugins.Imaris3D.io.LoadImarisDatasetNode')
    loader.set_property('dataset_dir', str(tmp_path))
    loader.set_property('layout', 'auto')
    loader.set_property('default_group', 'neg')
    loader.evaluate()

    qc = g.create_node('plugins.Imaris3D.qc.CellQCFilterNode')
    loader.set_output(0, qc.input(0))
    qc.set_property('excluded_cells_json', 'not valid json')

    ok, _ = qc.evaluate()
    assert ok
    out_ds = qc.output_values.get('dataset')
    assert out_ds.total_excluded() == 0
