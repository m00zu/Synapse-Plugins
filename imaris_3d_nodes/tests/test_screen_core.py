"""Tests for screen_core.py."""
import numpy as np
import pandas as pd
import pytest

from imaris_3d_nodes import screen_core


@pytest.fixture
def long_df():
    """Two groups, two (threshold,step) combos, with a clear winner at (12,3)."""
    rng = np.random.default_rng(42)
    rows = []
    # neg group: low values at all combos
    for f in range(5):
        for c in range(20):
            rows.append({
                'group': 'neg', 'file_stem': f'neg_{f}', 'cell': c,
                'pct_above_12_at_3um': rng.normal(0.1, 0.05),
                'pct_above_24_at_3um': rng.normal(0.05, 0.05),
            })
    # pos group: high at (12,3), low at (24,3)
    for f in range(5):
        for c in range(20):
            rows.append({
                'group': 'pos', 'file_stem': f'pos_{f}', 'cell': c,
                'pct_above_12_at_3um': rng.normal(0.5, 0.05),
                'pct_above_24_at_3um': rng.normal(0.05, 0.05),
            })
    return pd.DataFrame(rows)


def test_run_kfold_ranking_picks_correct_winner(long_df):
    """The (12,3) combo should rank above (24,3)."""
    out = screen_core.run_kfold_ranking(
        long_df,
        thresholds=[12, 24],
        steps=[3],
        ref_group='neg',
        cmp_group='pos',
        k=2,
        seeds=[0, 1, 2],
        primary_test='student',
        primary_fold='median',
    )
    assert len(out) == 2
    winner = out.iloc[0]
    assert (int(winner['threshold']), int(winner['step_um'])) == (12, 3)


def test_run_kfold_ranking_required_columns(long_df):
    out = screen_core.run_kfold_ranking(
        long_df, thresholds=[12], steps=[3],
        ref_group='neg', cmp_group='pos', k=2, seeds=[0],
    )
    assert {'threshold', 'step_um', 'worst_p', 'mean_neglog10p',
            'fold_change', 'passes_filter'} <= set(out.columns)
