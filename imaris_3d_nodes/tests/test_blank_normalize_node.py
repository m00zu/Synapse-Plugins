"""Tests for BlankNormalizeNode (using real graph + fake upstream nodes)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

_SYNAPSE = Path('/Users/s/Desktop/demo/PySide_Node/synapse')
if str(_SYNAPSE) not in sys.path:
    sys.path.insert(0, str(_SYNAPSE))


def _make_corrected_csv(path: Path, vals: list[float]):
    rows = [{'cell': i + 1, 'pct_above_12_at_3um': v} for i, v in enumerate(vals)]
    rows.append({'cell': 'BG', 'pct_above_12_at_3um': 0.05})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_blank_normalize_node_outputs_wide_table(tmp_path, qapp):
    import NodeGraphQt
    from imaris_3d_nodes.load_dataset_node import LoadImarisDatasetNode
    from imaris_3d_nodes.blank_normalize_node import BlankNormalizeNode

    # Build a tiny dataset via Load node
    (tmp_path / 'neg').mkdir()
    (tmp_path / 'pos').mkdir()
    _make_corrected_csv(tmp_path / 'neg' / 'F1_corrected.csv', [0.10, 0.20])
    _make_corrected_csv(tmp_path / 'pos' / 'F2_corrected.csv', [0.30, 0.40])

    g = NodeGraphQt.NodeGraph()
    g.register_node(LoadImarisDatasetNode)
    g.register_node(BlankNormalizeNode)

    loader = g.create_node('plugins.Imaris3D.io.LoadImarisDatasetNode')
    loader.set_property('dataset_dir', str(tmp_path))
    loader.set_property('layout', 'auto')
    ok, _ = loader.evaluate()
    assert ok

    # Need a fake chosen_combo TableData — create it manually as a parameter
    from data_models import TableData
    chosen = TableData(payload=pd.DataFrame([{'threshold': 12, 'step_um': 3}]))

    blank = g.create_node('plugins.Imaris3D.apply.BlankNormalizeNode')
    blank.set_property('reference_group', 'neg')
    blank.set_property('normalize', True)
    # Connect dataset
    loader.set_output(0, blank.input(0))
    # Inject chosen_combo via a faux node — simplest: directly set blank.input_values
    # No -- there's no input_values dict. We'll use a tiny fake upstream by adding
    # a 'parameters' node. The simplest is to construct another Load node? No.
    # Use direct: stash the chosen on a fake upstream-style mock by creating a
    # second node and manually populating output_values.

    # We'll use the loader as a hack: it has only one output, but we can create
    # a SECOND loader and overwrite its output_values with our chosen TableData
    # then connect to blank.input(1).
    fake = g.create_node('plugins.Imaris3D.io.LoadImarisDatasetNode')
    fake.output_values['imaris_dataset'] = chosen  # cheat: store TableData here
    # Wire fake.dataset -> blank.chosen_combo  (port 1)
    fake.set_output(0, blank.input(1))

    ok, msg = blank.evaluate()
    assert ok, msg

    out = blank.output_values.get('wide_table')
    assert out is not None
    assert set(out.payload.columns) == {'neg', 'pos'}
    # neg blanked: [0.05, 0.15], mean=0.10. Normalize divides by 0.10.
    assert out.payload['neg'].tolist() == pytest.approx([0.5, 1.5])
    assert out.payload['pos'].tolist() == pytest.approx([2.5, 3.5])


def test_blank_normalize_no_normalize_keeps_blanked(tmp_path, qapp):
    import NodeGraphQt
    from imaris_3d_nodes.load_dataset_node import LoadImarisDatasetNode
    from imaris_3d_nodes.blank_normalize_node import BlankNormalizeNode

    (tmp_path / 'neg').mkdir()
    (tmp_path / 'pos').mkdir()
    _make_corrected_csv(tmp_path / 'neg' / 'F1_corrected.csv', [0.10, 0.20])
    _make_corrected_csv(tmp_path / 'pos' / 'F2_corrected.csv', [0.30, 0.40])

    g = NodeGraphQt.NodeGraph()
    g.register_node(LoadImarisDatasetNode)
    g.register_node(BlankNormalizeNode)

    loader = g.create_node('plugins.Imaris3D.io.LoadImarisDatasetNode')
    loader.set_property('dataset_dir', str(tmp_path))
    loader.set_property('layout', 'auto')
    loader.evaluate()

    from data_models import TableData
    chosen = TableData(payload=pd.DataFrame([{'threshold': 12, 'step_um': 3}]))

    blank = g.create_node('plugins.Imaris3D.apply.BlankNormalizeNode')
    blank.set_property('reference_group', 'neg')
    blank.set_property('normalize', False)
    loader.set_output(0, blank.input(0))
    fake = g.create_node('plugins.Imaris3D.io.LoadImarisDatasetNode')
    fake.output_values['imaris_dataset'] = chosen
    fake.set_output(0, blank.input(1))

    ok, _ = blank.evaluate()
    assert ok
    out = blank.output_values.get('wide_table')
    assert out.payload['neg'].tolist() == pytest.approx([0.05, 0.15])
    assert out.payload['pos'].tolist() == pytest.approx([0.25, 0.35])
