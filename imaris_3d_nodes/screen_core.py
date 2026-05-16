"""K-fold cross-validated ranking of (threshold, step_um) combos.

Adapted from `/Users/s/Desktop/demo/Imaris_process/app/pipeline/screen.py`.
Input is the long per-cell DataFrame produced by ImarisDatasetData.to_long_per_cell();
output is one row per combo with worst_p, mean_neglog10p, fold_change, passes_filter.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold


def _group_stats(ref: pd.Series, cmp: pd.Series,
                 primary_test: str, primary_fold: str) -> dict:
    """One-shot stats on two arrays. Adapted from upstream `_group_stats`."""
    _, welch_t_p   = stats.ttest_ind(ref, cmp, equal_var=False)
    _, student_t_p = stats.ttest_ind(ref, cmp, equal_var=True)
    try:
        _, mw_p = stats.mannwhitneyu(ref, cmp, alternative='two-sided')
    except ValueError:
        mw_p = float('nan')

    r_m, c_m   = float(ref.mean()), float(cmp.mean())
    r_md, c_md = float(ref.median()), float(cmp.median())
    mean_fold = c_m / r_m if r_m != 0 and not np.isnan(r_m) else float('nan')
    med_fold  = c_md / r_md if r_md != 0 and not np.isnan(r_md) else float('nan')

    if primary_test == 'mw':
        p_value = float(mw_p)
    elif primary_test == 'welch':
        p_value = float(welch_t_p)
    else:  # student
        p_value = float(student_t_p)
    fold_change = med_fold if primary_fold == 'median' else mean_fold

    return {
        'welch_t_p': float(welch_t_p), 'student_t_p': float(student_t_p),
        'mw_p': float(mw_p),
        'mean_fold': mean_fold, 'median_fold': med_fold,
        'ref_mean': r_m, 'cmp_mean': c_m,
        'ref_median': r_md, 'cmp_median': c_md,
        'n_ref': int(len(ref)), 'n_cmp': int(len(cmp)),
        'p_value': float(p_value), 'fold_change': float(fold_change),
    }


def run_kfold_ranking(
    long_df: pd.DataFrame,
    *,
    thresholds: Iterable[int],
    steps: Iterable[int],
    ref_group: str,
    cmp_group: str,
    k: int = 2,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    primary_test: str = 'student',
    primary_fold: str = 'median',
    fold_change_min: float = 1.2,
) -> pd.DataFrame:
    """K-fold stratified CV ranking across (threshold, step_um) combos.

    For each combo, run K stratified folds across seeds, run `_group_stats`
    on each fold, then aggregate (worst_p = max over folds; mean_neglog10p =
    mean of -log10(p_value)). Rows are sorted so the most discriminating
    combo appears first.
    """
    seeds = list(seeds)
    thresholds = list(thresholds)
    steps = list(steps)

    pool = long_df[long_df['group'].isin([ref_group, cmp_group])].copy()
    if pool.empty:
        return pd.DataFrame()
    labels = (pool['group'] == cmp_group).astype(int).values

    rows = []
    for thr in thresholds:
        for step in steps:
            col = f'pct_above_{thr}_at_{step}um'
            if col not in pool.columns:
                continue

            per_fold = []
            for seed in seeds:
                skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
                for train_idx, _ in skf.split(pool, labels):
                    sub = pool.iloc[train_idx]
                    ref = sub[sub['group'] == ref_group][col].dropna()
                    cmp = sub[sub['group'] == cmp_group][col].dropna()
                    if len(ref) < 2 or len(cmp) < 2:
                        continue
                    s = _group_stats(ref, cmp, primary_test, primary_fold)
                    per_fold.append(s)

            if not per_fold:
                continue

            pf = pd.DataFrame(per_fold)
            worst_p = float(pf['p_value'].max())
            mean_neglog10p = float((-np.log10(pf['p_value'].clip(lower=1e-300))).mean())
            fold_change = float(pf['fold_change'].median())
            passes = (worst_p < 0.05) and (abs(fold_change) >= fold_change_min)

            rows.append({
                'threshold': int(thr),
                'step_um': int(step),
                'worst_p': worst_p,
                'mean_neglog10p': mean_neglog10p,
                'fold_change': fold_change,
                'passes_filter': bool(passes),
                'n_folds': int(len(per_fold)),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        by=['passes_filter', 'worst_p', 'mean_neglog10p'],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    return out
