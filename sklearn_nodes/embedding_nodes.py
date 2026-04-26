"""
embedding_nodes.py
==================
Dimensionality-reduction nodes (UMAP for now; t-SNE / PaCMAP could land here
later).  Backed by the vendored ``umap_rs_py`` Rust extension.

The UMAP node consumes any TableData (including MolTableData with array-valued
columns like fingerprints) — feature columns are flattened by ``build_xy``
exactly the way classifier / regressor nodes do it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nodes.base import BaseExecutionNode, PORT_COLORS
from data_models import TableData

from .ml_data import build_xy


# Metric option exposed in the combo.  'auto' means "pick a sensible default
# based on the input dtype" (see _resolve_metric).
_METRICS = [
    'auto',
    # Continuous (f32 / f64):
    'euclidean', 'manhattan', 'chebyshev', 'cosine', 'correlation',
    'minkowski', 'canberra', 'braycurtis', 'tanimoto',
    # Boolean bit-vector:
    'jaccard', 'dice', 'hamming',
    # MinHash signature (uint32):
    'hash_jaccard',
]

# Which metrics are valid for which dtype.  Used to give a clean error if a
# user picks a bool-only metric on a float input, etc.
_BOOL_METRICS = {'jaccard', 'dice', 'hamming'}
_HASH_METRICS = {'hash_jaccard'}
_CONT_METRICS = {
    'euclidean', 'sqeuclidean', 'manhattan', 'chebyshev', 'cosine',
    'correlation', 'tanimoto', 'minkowski', 'canberra', 'braycurtis',
}


def _resolve_metric(dtype, requested: str) -> str:
    """Pick a default metric for the input dtype if 'auto', else validate."""
    if requested == 'auto':
        if dtype == np.bool_:
            return 'jaccard'
        if dtype == np.uint32:
            return 'hash_jaccard'
        return 'euclidean'

    # Validate compatibility.
    if dtype == np.bool_ and requested not in _BOOL_METRICS:
        raise ValueError(
            f"Metric '{requested}' is not defined for boolean inputs. "
            f"Use one of: {sorted(_BOOL_METRICS)}"
        )
    if dtype == np.uint32 and requested not in _HASH_METRICS:
        raise ValueError(
            f"Metric '{requested}' is not defined for uint32 hash inputs. "
            f"Use 'hash_jaccard'."
        )
    if dtype.kind == 'f' and requested not in _CONT_METRICS:
        raise ValueError(
            f"Metric '{requested}' is not defined for continuous inputs. "
            f"Use one of: {sorted(_CONT_METRICS)}"
        )
    return requested


def _knn(matrix: np.ndarray, k: int, metric: str, p: float = 2.0):
    """Dispatch to the right umap_rs_py.knn_<dtype> kernel."""
    import umap_rs_py
    dt = matrix.dtype
    if dt == np.bool_:
        return umap_rs_py.knn_bool(matrix, k, metric, p)
    if dt == np.uint32:
        return umap_rs_py.knn_u32(matrix, k, metric, p)
    if dt == np.float32:
        return umap_rs_py.knn_f32(matrix, k, metric, p)
    # Cast everything else to float64.
    return umap_rs_py.knn_f64(matrix.astype(np.float64), k, metric, p)


def _build_init(X: np.ndarray, n_components: int, method: str, seed: int) -> np.ndarray:
    """Build initial embedding scaled to [-10, 10] (umap-rs convention)."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    if method == 'random':
        init = rng.uniform(-10.0, 10.0, size=(n, n_components)).astype(np.float32)
        return init

    # PCA: cast to float for sklearn, then scale to [-10, 10].
    from sklearn.decomposition import PCA
    X_float = X if X.dtype.kind == 'f' else X.astype(np.float32)
    init = PCA(n_components=n_components, random_state=seed).fit_transform(X_float)
    init = init.astype(np.float32)
    for c in range(n_components):
        col = init[:, c]
        lo, hi = float(col.min()), float(col.max())
        if hi > lo:
            init[:, c] = -10.0 + (col - lo) * 20.0 / (hi - lo)
    return init


class UMAPEmbeddingNode(BaseExecutionNode):
    """UMAP dimensionality reduction (Rust backend).

    Takes a table, picks the chosen feature columns (scalar or 1-D ndarray
    columns both supported via ``build_xy``), computes KNN + PCA-or-random
    init, runs UMAP, and appends ``umap_0``, ``umap_1`` (... up to
    ``n_components - 1``) columns to the output table.

    Backed by the vendored ``umap_rs_py`` Rust crate — fast brute-force KNN
    with rayon parallelism and the patched UMAP optimizer.

    Keywords: UMAP, dimensionality reduction, embedding, visualization, ML
    """
    __identifier__ = 'plugins.ML.Embedding'
    NODE_NAME = 'UMAP'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector(
            'feature_columns',
            label='Feature Columns (blank=all numeric)',
            mode='multi',
        )
        self._add_int_spinbox('n_components', 'Components', value=2,
                              min_val=2, max_val=64)
        self._add_int_spinbox('n_neighbors', 'Neighbours', value=15,
                              min_val=2, max_val=200)
        self._add_float_spinbox('min_dist', 'Min Dist', value=0.1,
                                min_val=0.0, max_val=1.0, step=0.05, decimals=3)
        self._add_float_spinbox('spread', 'Spread', value=1.0,
                                min_val=0.1, max_val=10.0, step=0.1, decimals=2)
        self.add_combo_menu('metric', 'Metric', items=_METRICS)
        self._add_float_spinbox('p', 'p (Minkowski only)', value=2.0,
                                min_val=0.1, max_val=20.0, step=0.5, decimals=2)
        self.add_combo_menu('init', 'Init', items=['pca', 'random'])
        self._add_int_spinbox('n_epochs', 'Epochs (0=auto)', value=0,
                              min_val=0, max_val=10000)
        self._add_int_spinbox('random_seed', 'Random Seed', value=42,
                              min_val=0, max_val=99999)

        self.output_values = {}

    def _get_input_df(self):
        port = self.inputs().get('table')
        if not port or not port.connected_ports():
            return None
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if isinstance(data, TableData):
            return data.df
        return None

    def evaluate(self):
        self.reset_progress()

        df = self._get_input_df()
        if df is None:
            self.mark_error()
            return False, "No table connected"

        n_components = int(self.get_property('n_components') or 2)
        n_neighbors  = int(self.get_property('n_neighbors')  or 15)
        min_dist     = float(self.get_property('min_dist')   or 0.1)
        spread       = float(self.get_property('spread')     or 1.0)
        metric_in    = str(self.get_property('metric') or 'auto').strip()
        p_minkowski  = float(self.get_property('p') or 2.0)
        init_method  = str(self.get_property('init') or 'pca').strip().lower()
        n_epochs_in  = int(self.get_property('n_epochs') or 0)
        seed         = int(self.get_property('random_seed') or 42)

        feat_text = str(self.get_property('feature_columns') or '').strip()

        self.set_progress(10)

        # Build the feature matrix (ndarray-aware).
        try:
            X, _, feature_names, _ = build_xy(
                df, target='', feature_columns=feat_text,
                require_target=False,
            )
        except ValueError as e:
            self.mark_error()
            return False, str(e)

        n = X.shape[0]
        if n < n_neighbors + 1:
            self.mark_error()
            return False, (f"n_neighbors={n_neighbors} requires at least "
                            f"{n_neighbors + 1} samples; got {n}.")

        # Resolve metric (auto-pick or validate against dtype).
        try:
            metric = _resolve_metric(X.dtype, metric_in)
        except ValueError as e:
            self.mark_error()
            return False, str(e)

        self.set_progress(25)

        # KNN via umap-rs (dtype-dispatched).
        try:
            knn_idx, knn_dist = _knn(X, n_neighbors, metric, p_minkowski)
        except Exception as e:
            self.mark_error()
            return False, f"KNN failed ({metric}): {e}"

        self.set_progress(55)

        # Initialization (PCA scaled to [-10, 10] or uniform random).
        try:
            init = _build_init(X, n_components, init_method, seed)
        except Exception as e:
            self.mark_error()
            return False, f"Init ({init_method}) failed: {e}"

        self.set_progress(70)

        # UMAP fit.
        try:
            import umap_rs_py
            X_for_umap = X.astype(np.float32) if X.dtype != np.float32 else X
            embedding = np.asarray(umap_rs_py.fit(
                X_for_umap, knn_idx, knn_dist, init,
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                spread=spread,
                n_epochs=(n_epochs_in or None),
                learning_rate=1.0,
                negative_sample_rate=5,
                repulsion_strength=1.0,
            ))
        except Exception as e:
            self.mark_error()
            return False, f"UMAP fit failed: {e}"

        self.set_progress(95)

        # Assemble output: original df + umap_0 / umap_1 / ... columns.
        out_df = df.copy()
        for c in range(n_components):
            out_df[f'umap_{c}'] = embedding[:, c]

        self.output_values['result'] = TableData(payload=out_df)
        self.mark_clean()
        self.set_progress(100)
        return True, (f"{n} × {len(feature_names)} → {n_components}D "
                      f"({metric}, {init_method} init)")
