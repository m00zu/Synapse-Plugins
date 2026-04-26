"""
base_model_node.py
==================
Shared base classes for sklearn training nodes (classifiers + regressors).

Pulls the previously-duplicated ``evaluate`` / ``_get_input_df`` plumbing
out of ``classifier_nodes.py`` so a brand-new regressor doesn't need to
re-implement table reading, X/y assembly, fitting, or model-data wrapping.
"""
from __future__ import annotations

from nodes.base import BaseExecutionNode
from data_models import TableData

from .ml_data import SklearnModelData, build_xy


class _BaseModelNode(BaseExecutionNode):
    """Common training-node plumbing.

    Subclasses must:
      - declare ``_TASK = 'classification' | 'regression'``
      - implement ``_build_model() -> sklearn estimator``
      - call ``self.add_input('train', ...)`` and ``add_output('model', ...)``,
        ``add_output('result', ...)`` in their own ``__init__``.
      - call ``self._add_column_selector('target_column', ...)`` and
        ``self._add_column_selector('feature_columns', ...)``.
    """

    _TASK = 'classification'

    def _get_input_df(self):
        port = self.inputs().get('train')
        if not port or not port.connected_ports():
            return None
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if isinstance(data, TableData):
            return data.df
        return None

    def _build_model(self):
        """Override in subclasses to return a fresh sklearn estimator."""
        raise NotImplementedError

    def evaluate(self):
        self.reset_progress()

        df = self._get_input_df()
        if df is None:
            self.mark_error()
            return False, "No training data connected"

        target = str(self.get_property('target_column') or '').strip()
        if not target or target not in df.columns:
            self.mark_error()
            return False, f"Target column '{target}' not found"

        self.set_progress(20)

        feat_text = str(self.get_property('feature_columns') or '').strip()
        try:
            X, y, feature_names, used_columns = build_xy(df, target, feat_text)
        except ValueError as e:
            self.mark_error()
            return False, str(e)

        self.set_progress(30)

        model = self._build_model()
        try:
            model.fit(X, y)
        except Exception as e:
            self.mark_error()
            return False, f"Training failed: {e}"

        self.set_progress(70)

        score = float(model.score(X, y))
        predictions = model.predict(X)
        result_df = df.copy()
        result_df['prediction'] = predictions

        self.set_progress(90)

        self.output_values['model'] = SklearnModelData(
            payload=model,
            model_type=type(model).__name__,
            feature_names=feature_names,
            feature_columns=used_columns,
            target_name=target,
            score=score,
            task=self._TASK,
        )
        self.output_values['result'] = TableData(payload=result_df)
        self.mark_clean()
        self.set_progress(100)
        return True, None


class _BaseClassifierNode(_BaseModelNode):
    """Base class for classifier training nodes."""
    _TASK = 'classification'


class _BaseRegressorNode(_BaseModelNode):
    """Base class for regressor training nodes."""
    _TASK = 'regression'
