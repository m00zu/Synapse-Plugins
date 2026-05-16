"""Tests for io.py."""
from pathlib import Path

import pandas as pd
import pytest

from imaris_3d_nodes import io as imaris_io
from imaris_3d_nodes.data import ImarisDatasetEntry

FIX = Path(__file__).parent / 'fixtures'


def test_load_entry_from_csv_splits_bg_row():
    csv = FIX / 'neg_F001_corrected.csv'
    entry = imaris_io.load_entry_from_csv(csv, group='neg_control')
    assert entry.file_stem == 'neg_F001'
    assert entry.group == 'neg_control'
    assert len(entry.per_cell_table) == 3   # BG row excluded
    assert set(entry.per_cell_table['cell'].astype(int)) == {1, 2, 3}
    # BG row preserved separately
    assert float(entry.bg_row['pct_above_12_at_3um']) == pytest.approx(0.05)


def test_load_entry_from_csv_finds_composite_sibling(tmp_path):
    csv = tmp_path / 'X_corrected.csv'
    df = pd.DataFrame({'cell': [1, 'BG'], 'pct_above_12_at_3um': [0.5, 0.1]})
    df.to_csv(csv, index=False)
    (tmp_path / 'X_composite.png').write_bytes(b'\x89PNG\r\n')
    entry = imaris_io.load_entry_from_csv(csv, group='g')
    assert entry.composite_path == tmp_path / 'X_composite.png'


def test_load_entry_returns_none_when_no_bg_row(tmp_path):
    csv = tmp_path / 'X_corrected.csv'
    df = pd.DataFrame({'cell': [1, 2, 3], 'pct_above_12_at_3um': [0.5, 0.1, 0.2]})
    df.to_csv(csv, index=False)
    entry = imaris_io.load_entry_from_csv(csv, group='g')
    assert entry is None


def test_load_entry_returns_none_when_no_cell_column(tmp_path):
    csv = tmp_path / 'X_corrected.csv'
    pd.DataFrame({'foo': [1]}).to_csv(csv, index=False)
    assert imaris_io.load_entry_from_csv(csv, group='g') is None


def test_detect_thresholds_and_steps():
    csv = FIX / 'neg_F001_corrected.csv'
    thrs, steps = imaris_io.detect_thresholds_and_steps(csv)
    assert thrs == [12, 24]
    assert steps == [3, 6]


def test_scan_dataset_dir_subfolders_layout(tmp_path):
    (tmp_path / 'neg').mkdir()
    (tmp_path / 'pos').mkdir()
    (tmp_path / 'neg' / 'F001_corrected.csv').write_text('cell\nBG\n')
    (tmp_path / 'neg' / 'F002_corrected.csv').write_text('cell\nBG\n')
    (tmp_path / 'pos' / 'F003_corrected.csv').write_text('cell\nBG\n')

    pairs = imaris_io.scan_dataset_dir(tmp_path)
    assert len(pairs) == 3
    groups = {g for _, g in pairs}
    assert groups == {'neg', 'pos'}


def test_scan_dataset_dir_flat_layout(tmp_path):
    (tmp_path / 'F001_corrected.csv').write_text('cell\nBG\n')
    (tmp_path / 'F002_corrected.csv').write_text('cell\nBG\n')

    pairs = imaris_io.scan_dataset_dir(tmp_path, default_group='my_group')
    assert len(pairs) == 2
    assert all(g == 'my_group' for _, g in pairs)


def test_scan_dataset_dir_flat_layout_default_group_uses_root_name(tmp_path):
    sub = tmp_path / 'experiment_42'
    sub.mkdir()
    (sub / 'F001_corrected.csv').write_text('cell\nBG\n')
    pairs = imaris_io.scan_dataset_dir(sub)
    assert pairs[0][1] == 'experiment_42'
