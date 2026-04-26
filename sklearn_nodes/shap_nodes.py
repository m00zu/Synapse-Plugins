"""
shap_nodes.py
=============
SHAP analysis nodes for trained sklearn / xgboost models.

Provides four nodes — global summary, feature dependence, single-sample
waterfall, and raw SHAP values — all sharing a common explainer-selection
helper that auto-picks ``TreeExplainer`` / ``LinearExplainer`` /
``KernelExplainer`` based on the wrapped model class.

Requires the ``shap`` package — install with ``pip install shap``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from nodes.base import BaseExecutionNode, PORT_COLORS
from data_models import TableData, FigureData

from .ml_data import SklearnModelData, SKLEARN_PORT_COLOR, build_xy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _import_shap():
    try:
        import shap  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "shap is not installed. Run `pip install shap`."
        ) from e
    return __import__('shap')


def _model_kind(model) -> str:
    """Classify the model into 'tree' | 'linear' | 'other' for explainer selection."""
    name = type(model).__name__
    tree_classes = {
        'RandomForestClassifier', 'RandomForestRegressor',
        'ExtraTreesClassifier', 'ExtraTreesRegressor',
        'GradientBoostingClassifier', 'GradientBoostingRegressor',
        'HistGradientBoostingClassifier', 'HistGradientBoostingRegressor',
        'AdaBoostClassifier', 'AdaBoostRegressor',
        'DecisionTreeClassifier', 'DecisionTreeRegressor',
        'XGBClassifier', 'XGBRegressor',
        'LGBMClassifier', 'LGBMRegressor',
        'CatBoostClassifier', 'CatBoostRegressor',
    }
    linear_classes = {
        'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'BayesianRidge',
        'LogisticRegression', 'RidgeClassifier',
        'SGDClassifier', 'SGDRegressor',
    }
    if name in tree_classes:
        return 'tree'
    if name in linear_classes:
        return 'linear'
    return 'other'


def _build_explainer(shap, model, X_background):
    """Pick the fastest valid SHAP explainer for the given model."""
    kind = _model_kind(model)
    if kind == 'tree':
        try:
            return shap.TreeExplainer(model)
        except Exception:
            pass
    if kind == 'linear':
        try:
            return shap.LinearExplainer(model, X_background)
        except Exception:
            pass
    # Fallback: model-agnostic Kernel.  Slow but always works.
    predict_fn = (model.predict_proba if hasattr(model, 'predict_proba')
                  else model.predict)
    return shap.KernelExplainer(predict_fn, X_background)


def _reduce_to_predicted_class(sv, model, X):
    """For multi-class output (3-D SHAP values), select predicted class per sample.

    Binary classification and regression already return 2-D arrays — passes
    those through unchanged.
    """
    import shap as _shap

    arr = np.asarray(sv.values)
    # Some explainers return list-of-arrays for multi-class; normalize.
    if isinstance(sv.values, list):
        arrs = [np.asarray(v) for v in sv.values]
        # Stack along last axis -> (n, f, c)
        arr = np.stack(arrs, axis=-1)

    if arr.ndim == 2:
        return sv

    if arr.ndim == 3:
        try:
            preds = model.predict(X)
        except Exception:
            preds = np.argmax(np.abs(arr).sum(axis=1), axis=-1)
        # Map class labels to column indices in the SHAP output.
        if hasattr(model, 'classes_'):
            classes = list(model.classes_)
            class_idx = np.array(
                [classes.index(p) if p in classes else 0 for p in preds],
                dtype=int,
            )
        else:
            class_idx = np.asarray(preds, dtype=int)

        n = arr.shape[0]
        new_values = arr[np.arange(n), :, class_idx]

        base = np.asarray(sv.base_values)
        if base.ndim == 2:  # (n, c)
            base = base[np.arange(n), class_idx]
        elif base.ndim == 1 and base.size == arr.shape[2]:
            # one base per class — broadcast back to per-sample
            base = base[class_idx]

        return _shap.Explanation(
            values=new_values,
            base_values=base,
            data=sv.data,
            feature_names=sv.feature_names,
        )
    return sv


def _compute_shap(model_data, df, max_samples, background_samples):
    """Compute SHAP values for the model on (a sample of) df.

    Returns ``(shap_explanation, X_used, feature_names)``.  For multi-class
    classification the SHAP values are reduced to the predicted class.
    """
    shap = _import_shap()

    target = model_data.target_name
    source_columns = list(model_data.feature_columns or model_data.feature_names)
    X, _, names, _ = build_xy(
        df, target='', feature_columns=source_columns, require_target=False,
    )
    if X.shape[0] == 0:
        raise ValueError("Input table is empty.")

    n_samples = min(int(max_samples), X.shape[0])
    X_sample = X[:n_samples]
    bg = min(int(background_samples), n_samples)
    X_background = X_sample[:bg]

    # Convert to DataFrame so SHAP picks up nice feature names.
    X_sample_df = pd.DataFrame(X_sample, columns=names)
    X_bg_df = pd.DataFrame(X_background, columns=names)

    explainer = _build_explainer(shap, model_data.payload, X_bg_df)
    sv = explainer(X_sample_df)

    # Multi-class → predicted-class reduction.
    sv = _reduce_to_predicted_class(sv, model_data.payload, X_sample_df)
    return sv, X_sample_df, names


# ── Common port-reading helpers (mirrors eval_nodes.py) ───────────────────────

class _SHAPInputMixin:
    def _get_input(self, port_name, expected_type):
        port = self.inputs().get(port_name)
        if not port or not port.connected_ports():
            return None
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        return data if isinstance(data, expected_type) else None

    def _get_input_df(self, port_name):
        port = self.inputs().get(port_name)
        if not port or not port.connected_ports():
            return None
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if isinstance(data, TableData):
            return data.df
        return None


# ── Nodes ─────────────────────────────────────────────────────────────────────

class SHAPSummaryNode(_SHAPInputMixin, BaseExecutionNode):
    """Global SHAP summary — which features matter most, with sign and spread.

    Outputs:
      - ``figure`` — matplotlib bar plot of mean |SHAP| per feature.
      - ``table``  — one row per feature (mean_abs_shap / mean_shap / std_shap),
                     sorted by mean |SHAP| descending.

    For multi-class classification, SHAP values for the predicted class of
    each sample are used (matches what the model actually decided).

    Keywords: SHAP, feature importance, explanation, ML, interpretability
    """
    __identifier__ = 'plugins.ML.Evaluation'
    NODE_NAME = 'SHAP Summary'
    PORT_SPEC = {'inputs': ['sklearn_model', 'table'], 'outputs': ['figure', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('model', color=SKLEARN_PORT_COLOR)
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('figure', color=PORT_COLORS['figure'])
        self.add_output('summary', color=PORT_COLORS['table'])

        self._add_int_spinbox('max_samples', 'Max Samples', value=1000,
                              min_val=10, max_val=100000)
        self._add_int_spinbox('background_samples', 'Background Samples',
                              value=100, min_val=10, max_val=10000)
        self._add_int_spinbox('max_display', 'Top N Features', value=20,
                              min_val=1, max_val=200)
        self.add_combo_menu('plot_type', 'Plot Type', items=['bar', 'beeswarm'])
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()

        model_data = self._get_input('model', SklearnModelData)
        if model_data is None:
            self.mark_error()
            return False, "No model connected"

        df = self._get_input_df('table')
        if df is None:
            self.mark_error()
            return False, "No table connected"

        self.set_progress(20)

        try:
            sv, X_used, names = _compute_shap(
                model_data, df,
                max_samples=int(self.get_property('max_samples') or 1000),
                background_samples=int(self.get_property('background_samples') or 100),
            )
        except (RuntimeError, ValueError) as e:
            self.mark_error()
            return False, str(e)
        except Exception as e:
            self.mark_error()
            return False, f"SHAP computation failed: {e}"

        self.set_progress(70)

        import shap
        max_display = int(self.get_property('max_display') or 20)
        plot_type = str(self.get_property('plot_type') or 'bar')

        plt.close('all')
        fig = plt.figure(figsize=(10, max(4, max_display * 0.35)))
        try:
            if plot_type == 'beeswarm':
                shap.plots.beeswarm(sv, max_display=max_display, show=False)
            else:
                shap.plots.bar(sv, max_display=max_display, show=False)
            fig = plt.gcf()
            fig.tight_layout()
        except Exception as e:
            plt.close(fig)
            self.mark_error()
            return False, f"Plot failed: {e}"

        # Build summary table sorted by mean |SHAP|
        vals = np.asarray(sv.values)
        summary_df = pd.DataFrame({
            'feature':       names,
            'mean_abs_shap': np.abs(vals).mean(axis=0),
            'mean_shap':     vals.mean(axis=0),
            'std_shap':      vals.std(axis=0),
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

        self.output_values['figure'] = FigureData(payload=fig)
        self.output_values['summary'] = TableData(payload=summary_df)
        self.mark_clean()
        self.set_progress(100)

        top = summary_df.iloc[0]
        return True, (f"Top feature: {top['feature']} "
                      f"(mean|SHAP|={top['mean_abs_shap']:.4f})")


class SHAPDependenceNode(_SHAPInputMixin, BaseExecutionNode):
    """SHAP dependence plot — how a single feature drives the prediction,
    coloured by an interacting feature.

    Inputs:
      - ``feature`` — the feature to plot on the x-axis.  Use the expanded
                      name from a fingerprint / vector column (e.g. ``fp[42]``)
                      or just the column name for scalar features.  An integer
                      is interpreted as the column index.
      - ``interaction_feature`` — feature used for colouring (or ``auto`` for
                                  the strongest interaction).

    Keywords: SHAP, dependence, interaction, partial dependence, ML
    """
    __identifier__ = 'plugins.ML.Evaluation'
    NODE_NAME = 'SHAP Dependence'
    PORT_SPEC = {'inputs': ['sklearn_model', 'table'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__()
        self.add_input('model', color=SKLEARN_PORT_COLOR)
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('figure', color=PORT_COLORS['figure'])

        self.add_text_input('feature', 'Feature', text='')
        self.add_text_input('interaction_feature', 'Interaction', text='auto')
        self._add_int_spinbox('max_samples', 'Max Samples', value=1000,
                              min_val=10, max_val=100000)
        self._add_int_spinbox('background_samples', 'Background Samples',
                              value=100, min_val=10, max_val=10000)
        self.output_values = {}

    def _resolve_feature(self, token: str, names: list[str]) -> int:
        token = (token or '').strip()
        if not token:
            raise ValueError("Feature name is required")
        if token.lstrip('-').isdigit():
            idx = int(token)
            if 0 <= idx < len(names):
                return idx
            raise ValueError(
                f"Feature index {idx} out of range (0–{len(names) - 1})"
            )
        if token in names:
            return names.index(token)
        # Fuzzy hint
        sample = ', '.join(names[:8]) + ('…' if len(names) > 8 else '')
        raise ValueError(f"Feature '{token}' not found. Available: {sample}")

    def evaluate(self):
        self.reset_progress()

        model_data = self._get_input('model', SklearnModelData)
        if model_data is None:
            self.mark_error()
            return False, "No model connected"
        df = self._get_input_df('table')
        if df is None:
            self.mark_error()
            return False, "No table connected"

        self.set_progress(20)

        try:
            sv, X_used, names = _compute_shap(
                model_data, df,
                max_samples=int(self.get_property('max_samples') or 1000),
                background_samples=int(self.get_property('background_samples') or 100),
            )
        except (RuntimeError, ValueError) as e:
            self.mark_error()
            return False, str(e)
        except Exception as e:
            self.mark_error()
            return False, f"SHAP computation failed: {e}"

        feat_token = str(self.get_property('feature') or '').strip()
        try:
            feat_idx = self._resolve_feature(feat_token, names)
        except ValueError as e:
            self.mark_error()
            return False, str(e)

        inter_token = str(self.get_property('interaction_feature') or 'auto').strip()
        if inter_token.lower() == 'auto':
            inter_idx = 'auto'
        else:
            try:
                inter_idx = self._resolve_feature(inter_token, names)
            except ValueError as e:
                self.mark_error()
                return False, str(e)

        self.set_progress(70)

        import shap
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        try:
            color = sv[:, inter_idx] if inter_idx != 'auto' else sv
            shap.plots.scatter(sv[:, feat_idx], color=color, show=False)
            fig = plt.gcf()
            fig.tight_layout()
        except Exception as e:
            plt.close(fig)
            self.mark_error()
            return False, f"Plot failed: {e}"

        self.output_values['figure'] = FigureData(payload=fig)
        self.mark_clean()
        self.set_progress(100)
        return True, f"Dependence for {names[feat_idx]}"


class SHAPSampleNode(_SHAPInputMixin, BaseExecutionNode):
    """Per-sample SHAP waterfall — why did the model predict X for THIS row?

    Outputs:
      - ``figure`` — waterfall plot showing each feature's push from the
                     base value to the final prediction for the chosen row.
      - ``contributions`` — table of (feature, value, shap_contribution)
                             rows, sorted by |contribution| descending.

    Keywords: SHAP, local explanation, waterfall, ML, interpretability
    """
    __identifier__ = 'plugins.ML.Evaluation'
    NODE_NAME = 'SHAP Sample'
    PORT_SPEC = {'inputs': ['sklearn_model', 'table'], 'outputs': ['figure', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('model', color=SKLEARN_PORT_COLOR)
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('figure', color=PORT_COLORS['figure'])
        self.add_output('contributions', color=PORT_COLORS['table'])

        self._add_int_spinbox('sample_index', 'Sample Index', value=0,
                              min_val=0, max_val=1_000_000)
        self._add_int_spinbox('max_samples', 'Max Samples', value=1000,
                              min_val=10, max_val=100000)
        self._add_int_spinbox('background_samples', 'Background Samples',
                              value=100, min_val=10, max_val=10000)
        self._add_int_spinbox('max_display', 'Top N Features', value=15,
                              min_val=1, max_val=200)
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()

        model_data = self._get_input('model', SklearnModelData)
        if model_data is None:
            self.mark_error()
            return False, "No model connected"
        df = self._get_input_df('table')
        if df is None:
            self.mark_error()
            return False, "No table connected"

        sample_idx = int(self.get_property('sample_index') or 0)

        self.set_progress(20)

        try:
            sv, X_used, names = _compute_shap(
                model_data, df,
                max_samples=int(self.get_property('max_samples') or 1000),
                background_samples=int(self.get_property('background_samples') or 100),
            )
        except (RuntimeError, ValueError) as e:
            self.mark_error()
            return False, str(e)
        except Exception as e:
            self.mark_error()
            return False, f"SHAP computation failed: {e}"

        if sample_idx < 0 or sample_idx >= len(X_used):
            self.mark_error()
            return False, (f"Sample index {sample_idx} out of range "
                            f"(0–{len(X_used) - 1})")

        self.set_progress(70)

        import shap
        max_display = int(self.get_property('max_display') or 15)
        plt.close('all')
        fig = plt.figure(figsize=(10, max(4, max_display * 0.35)))
        try:
            shap.plots.waterfall(sv[sample_idx], max_display=max_display, show=False)
            fig = plt.gcf()
            fig.tight_layout()
        except Exception as e:
            plt.close(fig)
            self.mark_error()
            return False, f"Plot failed: {e}"

        # Per-feature contribution table for this sample.
        vals = np.asarray(sv.values)[sample_idx]
        feat_vals = X_used.iloc[sample_idx].to_numpy()
        contrib_df = pd.DataFrame({
            'feature':           names,
            'feature_value':     feat_vals,
            'shap_contribution': vals,
        })
        contrib_df['abs_contribution'] = np.abs(contrib_df['shap_contribution'])
        contrib_df = (contrib_df
                      .sort_values('abs_contribution', ascending=False)
                      .drop(columns='abs_contribution')
                      .reset_index(drop=True))

        self.output_values['figure'] = FigureData(payload=fig)
        self.output_values['contributions'] = TableData(payload=contrib_df)
        self.mark_clean()
        self.set_progress(100)

        base = float(np.mean(np.asarray(sv.base_values)))
        pred = float(base + np.sum(vals))
        return True, f"Sample {sample_idx}: base={base:.3f} → pred={pred:.3f}"


class SHAPValuesNode(_SHAPInputMixin, BaseExecutionNode):
    """Raw SHAP value matrix as a table — one row per sample, one column per
    feature, plus ``base_value`` and ``prediction`` columns.

    Useful as input to downstream nodes (Heatmap, dimensionality reduction
    on SHAP, custom analysis).

    Keywords: SHAP, raw values, attribution, ML, interpretability
    """
    __identifier__ = 'plugins.ML.Evaluation'
    NODE_NAME = 'SHAP Values'
    PORT_SPEC = {'inputs': ['sklearn_model', 'table'], 'outputs': ['table']}

    def __init__(self):
        super().__init__()
        self.add_input('model', color=SKLEARN_PORT_COLOR)
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('shap_values', color=PORT_COLORS['table'])

        self._add_int_spinbox('max_samples', 'Max Samples', value=1000,
                              min_val=10, max_val=100000)
        self._add_int_spinbox('background_samples', 'Background Samples',
                              value=100, min_val=10, max_val=10000)
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()

        model_data = self._get_input('model', SklearnModelData)
        if model_data is None:
            self.mark_error()
            return False, "No model connected"
        df = self._get_input_df('table')
        if df is None:
            self.mark_error()
            return False, "No table connected"

        self.set_progress(20)

        try:
            sv, X_used, names = _compute_shap(
                model_data, df,
                max_samples=int(self.get_property('max_samples') or 1000),
                background_samples=int(self.get_property('background_samples') or 100),
            )
        except (RuntimeError, ValueError) as e:
            self.mark_error()
            return False, str(e)
        except Exception as e:
            self.mark_error()
            return False, f"SHAP computation failed: {e}"

        self.set_progress(70)

        vals = np.asarray(sv.values)
        base = np.asarray(sv.base_values)
        if base.ndim == 0:
            base = np.full(vals.shape[0], float(base))
        elif base.ndim > 1:
            # Flatten via mean if shape is unexpected (e.g. (n, c)).
            base = base.mean(axis=tuple(range(1, base.ndim)))

        out_df = pd.DataFrame(vals, columns=names)
        out_df.insert(0, 'base_value', base)
        out_df.insert(1, 'prediction', base + vals.sum(axis=1))

        self.output_values['shap_values'] = TableData(payload=out_df)
        self.mark_clean()
        self.set_progress(100)
        return True, f"{vals.shape[0]} × {vals.shape[1]} SHAP matrix"
