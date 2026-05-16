"""Tests for the ImarisDatasetData / ImarisDatasetEntry types."""
from pathlib import Path

import pandas as pd
import pytest

from imaris_3d_nodes.data import (
    ImarisDatasetData, ImarisDatasetEntry, PORT_TYPE_NAME,
)


@pytest.fixture
def cells_df():
    return pd.DataFrame({
        'cell': [1, 2, 3, 4],
        'pct_above_12_at_3um': [0.10, 0.20, 0.30, 0.40],
        'bbox_min_x_px': [10, 50, 100, 200],
        'bbox_max_x_px': [30, 70, 120, 220],
        'bbox_min_y_px': [10, 50, 100, 200],
        'bbox_max_y_px': [30, 70, 120, 220],
        'centroid_x_px': [20, 60, 110, 210],
        'centroid_y_px': [20, 60, 110, 210],
    })


@pytest.fixture
def bg_row():
    return pd.Series({'pct_above_12_at_3um': 0.05})


@pytest.fixture
def entry(cells_df, bg_row, tmp_path):
    return ImarisDatasetEntry(
        file_stem='F001',
        group='neg_control',
        csv_path=tmp_path / 'F001_corrected.csv',
        composite_path=tmp_path / 'F001_composite.png',
        per_cell_table=cells_df,
        bg_row=bg_row,
    )


def test_entry_kept_table_no_exclusions(entry):
    assert len(entry.kept_table) == 4


def test_entry_kept_table_with_exclusions(entry):
    entry.excluded_cells = {2, 4}
    kept = entry.kept_table
    assert len(kept) == 2
    assert set(kept['cell']) == {1, 3}


def test_entry_n_cells_and_n_excluded(entry):
    entry.excluded_cells = {2}
    assert entry.n_cells == 4
    assert entry.n_excluded == 1


def test_to_long_per_cell_two_entries(cells_df, bg_row, tmp_path):
    e1 = ImarisDatasetEntry('F001', 'neg', tmp_path / 'a.csv', tmp_path / 'a.png',
                            cells_df, bg_row)
    e2 = ImarisDatasetEntry('F002', 'pos', tmp_path / 'b.csv', tmp_path / 'b.png',
                            cells_df, bg_row)
    ds = ImarisDatasetData(entries=[e1, e2])
    long = ds.to_long_per_cell()
    assert len(long) == 8
    assert set(long['group']) == {'neg', 'pos'}
    assert {'group', 'file_stem', 'cell'} <= set(long.columns)


def test_to_wide_blanked_subtracts_bg(cells_df, bg_row, tmp_path):
    e1 = ImarisDatasetEntry('F001', 'neg', tmp_path / 'a.csv', tmp_path / 'a.png',
                            cells_df, bg_row)
    ds = ImarisDatasetData(entries=[e1])
    wide = ds.to_wide_blanked(threshold=12, step_um=3)
    # BG was 0.05; original values were 0.10, 0.20, 0.30, 0.40
    assert wide['neg'].tolist() == pytest.approx([0.05, 0.15, 0.25, 0.35])


def test_to_wide_blanked_excludes_filtered_cells(cells_df, bg_row, tmp_path):
    e1 = ImarisDatasetEntry('F001', 'neg', tmp_path / 'a.csv', tmp_path / 'a.png',
                            cells_df, bg_row)
    e1.excluded_cells = {2, 4}
    ds = ImarisDatasetData(entries=[e1])
    wide = ds.to_wide_blanked(threshold=12, step_um=3)
    # Cells 1, 3 survive: 0.10-0.05=0.05, 0.30-0.05=0.25
    assert wide['neg'].dropna().tolist() == pytest.approx([0.05, 0.25])


def test_groups_preserves_encounter_order(cells_df, bg_row, tmp_path):
    e1 = ImarisDatasetEntry('F1', 'pos', tmp_path/'a.csv', tmp_path/'a.png',
                            cells_df, bg_row)
    e2 = ImarisDatasetEntry('F2', 'neg', tmp_path/'b.csv', tmp_path/'b.png',
                            cells_df, bg_row)
    e3 = ImarisDatasetEntry('F3', 'pos', tmp_path/'c.csv', tmp_path/'c.png',
                            cells_df, bg_row)
    ds = ImarisDatasetData(entries=[e1, e2, e3])
    assert ds.groups == ['pos', 'neg']


def test_total_cells_and_excluded(cells_df, bg_row, tmp_path):
    e1 = ImarisDatasetEntry('F1', 'neg', tmp_path/'a.csv', tmp_path/'a.png',
                            cells_df, bg_row)
    e1.excluded_cells = {2}
    e2 = ImarisDatasetEntry('F2', 'neg', tmp_path/'b.csv', tmp_path/'b.png',
                            cells_df, bg_row)
    ds = ImarisDatasetData(entries=[e1, e2])
    assert ds.total_cells() == 8
    assert ds.total_excluded() == 1
