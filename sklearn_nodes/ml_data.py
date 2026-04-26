"""
ml_data.py
==========
Data classes and constants for the sklearn plugin.
"""
from pydantic import BaseModel, ConfigDict
from typing import Any, Optional
from data_models import NodeData, TableData
import pandas as pd

# Port color for sklearn model connections (gold)
SKLEARN_PORT_COLOR = (241, 196, 15)


class SklearnModelData(NodeData):
    """Wraps a trained scikit-learn estimator.

    Attributes:
        payload: the fitted sklearn estimator object
        model_type: string name, e.g. 'RandomForestClassifier'
        feature_names: list of EXPANDED feature names (one per matrix column).
                       For scalar source columns this matches the column name;
                       for 1-D ndarray columns (e.g. fingerprints) entries are
                       like ``'fp[0]', 'fp[1]', …, 'fp[2047]'``.
        feature_columns: list of ORIGINAL source column names from the input
                         DataFrame.  Used by PredictNode to rebuild X with the
                         same scalar-or-ndarray expansion.  Empty means
                         "all numeric scalar columns except target" (legacy).
        target_name: the target column name
        score: training accuracy or R² score (optional)
        task: 'classification' or 'regression'
    """
    payload: Any
    model_type: str = ""
    feature_names: list[str] = []
    feature_columns: list[str] = []
    target_name: str = ""
    score: Optional[float] = None
    task: str = "classification"


# ── Helper: build (X, y, names) from a DataFrame ─────────────────────────────

def _parse_column_list(text: str) -> list[str]:
    """Parse a comma-separated column-selector string into a list of names."""
    if not text:
        return []
    return [c.strip() for c in str(text).split(',') if c.strip()]


def build_xy(
    df: pd.DataFrame,
    target: str,
    feature_columns: list[str] | str | None = None,
    *,
    require_target: bool = True,
):
    """Build a model-ready ``(X, y, feature_names, used_columns)`` from a table.

    Behavior:
      - If ``feature_columns`` is empty/None, all numeric scalar columns
        except ``target`` are used (legacy behavior).
      - Otherwise each listed column is used as-is.  If a column's first row
        holds a 1-D ``np.ndarray``, the column expands into multiple feature
        columns (``'col[0]', 'col[1]', …``); a scalar column contributes one.
      - ``y`` is ``None`` when ``require_target`` is False or when the target
        column is missing.

    Returns:
        X: 2-D float64 array of shape ``(N, total_dim)``.
        y: 1-D array or ``None``.
        feature_names: list of expanded feature names (length == X.shape[1]).
        used_columns: list of original source column names.
    """
    import numpy as np

    if isinstance(feature_columns, str):
        feature_columns = _parse_column_list(feature_columns)

    if feature_columns:
        used_columns = list(feature_columns)
        missing = [c for c in used_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {', '.join(missing)}")
    else:
        used_columns = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target
        ]
        if not used_columns:
            raise ValueError(
                "No numeric feature columns found. "
                "Use the Feature Columns selector to specify columns "
                "(including 1-D ndarray columns like fingerprints)."
            )

    blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    n = len(df)
    for col in used_columns:
        series = df[col]
        sample = series.iloc[0] if n > 0 else None
        if isinstance(sample, np.ndarray) and sample.ndim == 1:
            mat = np.stack(series.tolist()).astype(np.float64, copy=False)
            blocks.append(mat)
            feature_names.extend([f"{col}[{i}]" for i in range(mat.shape[1])])
        else:
            arr = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64)
            blocks.append(arr.reshape(-1, 1))
            feature_names.append(col)

    X = np.hstack(blocks) if blocks else np.zeros((n, 0), dtype=np.float64)

    y = None
    if target and target in df.columns:
        y = df[target].to_numpy()
    elif require_target:
        raise ValueError(f"Target column '{target}' not found")

    return X, y, feature_names, used_columns
