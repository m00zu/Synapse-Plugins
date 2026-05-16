"""Pure-pandas blanking + reference normalization."""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_by_reference(wide: pd.DataFrame, ref_group: str) -> pd.DataFrame:
    """Divide every column by the mean of `ref_group`. No-op if ref missing or zero."""
    if ref_group not in wide.columns:
        return wide
    ref_mean = float(wide[ref_group].mean())
    if ref_mean == 0 or np.isnan(ref_mean):
        return wide
    return wide / ref_mean
