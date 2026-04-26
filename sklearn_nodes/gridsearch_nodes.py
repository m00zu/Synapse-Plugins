"""
gridsearch_nodes.py
===================
K-fold grid search for sklearn classifiers and regressors.

Provides two nodes (``GridSearchClassifierNode``, ``GridSearchRegressorNode``)
sharing an inline ``ParamGridWidget``.  The widget introspects the chosen
estimator's ``__init__`` signature and shows one row per parameter; the user
types comma-separated values to sweep (one value = fixed; multiple = swept).
Empty rows fall back to the sklearn default.

The actual search is delegated to ``sklearn.model_selection.GridSearchCV``,
which handles K-fold CV, refitting the best estimator, and parallel jobs.
"""
from __future__ import annotations

import ast
import importlib
import inspect
from typing import Any

import pandas as pd
from PySide6 import QtCore, QtWidgets
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget

from nodes.base import BaseExecutionNode, PORT_COLORS
from data_models import TableData

from .ml_data import SklearnModelData, SKLEARN_PORT_COLOR, build_xy


# ── Model registries ──────────────────────────────────────────────────────────
# Display name → fully-qualified sklearn class.  Lazy-imported on demand.

CLASSIFIER_MODELS: dict[str, str] = {
    'Random Forest':            'sklearn.ensemble.RandomForestClassifier',
    'Extra Trees':              'sklearn.ensemble.ExtraTreesClassifier',
    'Gradient Boosting':        'sklearn.ensemble.GradientBoostingClassifier',
    'Hist Gradient Boosting':   'sklearn.ensemble.HistGradientBoostingClassifier',
    'AdaBoost':                 'sklearn.ensemble.AdaBoostClassifier',
    'SVM':                      'sklearn.svm.SVC',
    'KNN':                      'sklearn.neighbors.KNeighborsClassifier',
    'Logistic Regression':      'sklearn.linear_model.LogisticRegression',
    'Ridge Classifier':         'sklearn.linear_model.RidgeClassifier',
    'Decision Tree':            'sklearn.tree.DecisionTreeClassifier',
    'MLP':                      'sklearn.neural_network.MLPClassifier',
    'Naive Bayes':              'sklearn.naive_bayes.GaussianNB',
    'LDA':                      'sklearn.discriminant_analysis.LinearDiscriminantAnalysis',
    'QDA':                      'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis',
    'XGBoost':                  'xgboost.XGBClassifier',
}

REGRESSOR_MODELS: dict[str, str] = {
    'Linear Regression':        'sklearn.linear_model.LinearRegression',
    'Ridge':                    'sklearn.linear_model.Ridge',
    'Lasso':                    'sklearn.linear_model.Lasso',
    'Elastic Net':              'sklearn.linear_model.ElasticNet',
    'Bayesian Ridge':           'sklearn.linear_model.BayesianRidge',
    'Random Forest':            'sklearn.ensemble.RandomForestRegressor',
    'Extra Trees':              'sklearn.ensemble.ExtraTreesRegressor',
    'Gradient Boosting':        'sklearn.ensemble.GradientBoostingRegressor',
    'Hist Gradient Boosting':   'sklearn.ensemble.HistGradientBoostingRegressor',
    'SVR':                      'sklearn.svm.SVR',
    'KNN':                      'sklearn.neighbors.KNeighborsRegressor',
    'MLP':                      'sklearn.neural_network.MLPRegressor',
    'Decision Tree':            'sklearn.tree.DecisionTreeRegressor',
    'XGBoost':                  'xgboost.XGBRegressor',
}


# Curated list of "commonly-tuned" parameters per model.  If a model class is
# in this dict, only the listed params show up in the table.  Models NOT in
# this dict show every parameter from their __init__ signature.
CURATED_PARAMS: dict[str, list[str]] = {
    'sklearn.ensemble.RandomForestClassifier': [
        'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
        'max_features', 'random_state',
    ],
    'sklearn.ensemble.RandomForestRegressor': [
        'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
        'max_features', 'random_state',
    ],
    'sklearn.ensemble.ExtraTreesClassifier': [
        'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
        'max_features', 'random_state',
    ],
    'sklearn.ensemble.ExtraTreesRegressor': [
        'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
        'max_features', 'random_state',
    ],
    'sklearn.ensemble.GradientBoostingClassifier': [
        'n_estimators', 'learning_rate', 'max_depth', 'min_samples_split',
        'subsample', 'random_state',
    ],
    'sklearn.ensemble.GradientBoostingRegressor': [
        'n_estimators', 'learning_rate', 'max_depth', 'min_samples_split',
        'subsample', 'random_state',
    ],
    'sklearn.ensemble.HistGradientBoostingClassifier': [
        'learning_rate', 'max_iter', 'max_depth', 'min_samples_leaf',
        'l2_regularization', 'random_state',
    ],
    'sklearn.ensemble.HistGradientBoostingRegressor': [
        'learning_rate', 'max_iter', 'max_depth', 'min_samples_leaf',
        'l2_regularization', 'random_state',
    ],
    'sklearn.ensemble.AdaBoostClassifier': [
        'n_estimators', 'learning_rate', 'random_state',
    ],
    'sklearn.svm.SVC':  ['C', 'kernel', 'gamma', 'degree', 'random_state'],
    'sklearn.svm.SVR':  ['C', 'kernel', 'gamma', 'degree', 'epsilon'],
    'sklearn.neighbors.KNeighborsClassifier': [
        'n_neighbors', 'weights', 'metric', 'leaf_size',
    ],
    'sklearn.neighbors.KNeighborsRegressor': [
        'n_neighbors', 'weights', 'metric', 'leaf_size',
    ],
    'sklearn.linear_model.LogisticRegression': [
        'C', 'penalty', 'solver', 'max_iter', 'random_state',
    ],
    'sklearn.linear_model.RidgeClassifier': ['alpha', 'random_state'],
    'sklearn.linear_model.Ridge':           ['alpha', 'random_state'],
    'sklearn.linear_model.Lasso':           ['alpha', 'max_iter', 'random_state'],
    'sklearn.linear_model.ElasticNet': [
        'alpha', 'l1_ratio', 'max_iter', 'random_state',
    ],
    'sklearn.linear_model.BayesianRidge': [
        'alpha_1', 'alpha_2', 'lambda_1', 'lambda_2', 'max_iter',
    ],
    'sklearn.tree.DecisionTreeClassifier': [
        'max_depth', 'min_samples_split', 'min_samples_leaf', 'criterion',
        'random_state',
    ],
    'sklearn.tree.DecisionTreeRegressor': [
        'max_depth', 'min_samples_split', 'min_samples_leaf', 'criterion',
        'random_state',
    ],
    'sklearn.neural_network.MLPClassifier': [
        'hidden_layer_sizes', 'activation', 'alpha', 'learning_rate_init',
        'max_iter', 'random_state',
    ],
    'sklearn.neural_network.MLPRegressor': [
        'hidden_layer_sizes', 'activation', 'alpha', 'learning_rate_init',
        'max_iter', 'random_state',
    ],
    'xgboost.XGBClassifier': [
        'n_estimators', 'max_depth', 'learning_rate', 'subsample',
        'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state',
    ],
    'xgboost.XGBRegressor': [
        'n_estimators', 'max_depth', 'learning_rate', 'subsample',
        'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state',
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _import_class(dotted: str):
    module, _, cls = dotted.rpartition('.')
    return getattr(importlib.import_module(module), cls)


def _format_default(default) -> str:
    if default is inspect.Parameter.empty:
        return ''
    if default is None:
        return 'None'
    if isinstance(default, str):
        return repr(default)
    return str(default)


def _infer_type_hint(default) -> str:
    if default is inspect.Parameter.empty:
        return 'any'
    if default is None:
        return 'None'
    return type(default).__name__


def _parse_token(token: str) -> Any:
    """Parse one comma-separated token. Tries Python literal first
    (handles None, True/False, ints, floats, tuples, dicts), falls back
    to plain string for unquoted strings like 'rbf'.
    """
    s = token.strip()
    if not s:
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


# ── Inline param-grid widget ──────────────────────────────────────────────────

class ParamGridWidget(NodeBaseWidget):
    """Inline (param, type, values) table.  Auto-rebuilds when the chosen
    estimator changes; persists the user's typed values across model switches
    so flipping models doesn't lose work.
    """

    _MIN_TABLE_HEIGHT = 280

    def __init__(self, parent=None, name: str = 'param_grid', label: str = ''):
        super().__init__(parent, name, label)
        self._estimator_class = None
        self._curated_subset: list[str] | None = None
        self._suppress_emit = False
        # Param values keyed by param name; persists across model switches.
        self._values_text: dict[str, str] = {}

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._table = QtWidgets.QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(['Parameter', 'Type', 'Values'])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QtWidgets.QTableWidget.EditTrigger.NoEditTriggers)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self._table.setMinimumHeight(self._MIN_TABLE_HEIGHT)
        layout.addWidget(self._table)

        hint = QtWidgets.QLabel(
            '<span style="color:#888; font-size:10pt;">'
            'Comma-separated values: <b>1 value = fixed</b>, '
            '<b>2+ values = swept</b>. Empty row = sklearn default. '
            'Use Python literals: <i>None, True, False, 0.1, "rbf", (50, 50)</i>.'
            '</span>'
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.set_custom_widget(container)

    # ── Public API ─────────────────────────────────────────────────────────
    def set_estimator(self, cls, curated: list[str] | None = None) -> None:
        """Introspect ``cls.__init__`` and rebuild the table rows."""
        self._estimator_class = cls
        self._curated_subset = curated
        self._build_rows()

    def get_value(self) -> dict[str, str]:
        """Return ``{param_name: 'comma, separated, text'}`` for any row with
        non-empty content. Persists across model switches."""
        self._sync_state()
        return {k: v for k, v in self._values_text.items() if v}

    def set_value(self, value):
        if not isinstance(value, dict):
            return
        self._suppress_emit = True
        try:
            self._values_text = {str(k): str(v) for k, v in value.items() if v}
            self._build_rows()
        finally:
            self._suppress_emit = False

    def build_param_grid(self) -> dict[str, list]:
        """Parse the current state into a sklearn-compatible ``param_grid``."""
        self._sync_state()
        grid: dict[str, list] = {}
        for name, text in self._values_text.items():
            if not text:
                continue
            tokens = [t for t in text.split(',') if t.strip()]
            if not tokens:
                continue
            grid[name] = [_parse_token(t) for t in tokens]
        return grid

    # ── Internal ───────────────────────────────────────────────────────────
    def _build_rows(self) -> None:
        self._table.clearContents()
        self._table.setRowCount(0)
        if self._estimator_class is None:
            return
        try:
            sig = inspect.signature(self._estimator_class.__init__)
        except (TypeError, ValueError):
            return

        rows: list[tuple[str, Any]] = []
        for pname, param in sig.parameters.items():
            if pname in ('self', 'args', 'kwargs'):
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD):
                continue
            if self._curated_subset and pname not in self._curated_subset:
                continue
            rows.append((pname, param.default))

        self._table.setRowCount(len(rows))
        for r, (pname, default) in enumerate(rows):
            type_hint = _infer_type_hint(default)
            default_str = _format_default(default)

            name_item = QtWidgets.QTableWidgetItem(pname)
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(r, 0, name_item)

            type_label = QtWidgets.QLabel(
                f'<span style="color:#aaa; font-size:9pt;">'
                f'{type_hint}<br><i>={default_str}</i></span>'
            )
            type_label.setContentsMargins(4, 0, 4, 0)
            self._table.setCellWidget(r, 1, type_label)

            line = QtWidgets.QLineEdit()
            line.setPlaceholderText('use default')
            saved = self._values_text.get(pname, '')
            if saved:
                line.setText(saved)
            line.setProperty('_param_name', pname)
            line.editingFinished.connect(self._on_value_changed)
            self._table.setCellWidget(r, 2, line)

        self._table.resizeRowsToContents()

    def _on_value_changed(self) -> None:
        self._sync_state()
        if not self._suppress_emit:
            self.value_changed.emit(self.get_name(), self.get_value())

    def _sync_state(self) -> None:
        for r in range(self._table.rowCount()):
            name_item = self._table.item(r, 0)
            line = self._table.cellWidget(r, 2)
            if name_item is None or not isinstance(line, QtWidgets.QLineEdit):
                continue
            self._values_text[name_item.text()] = line.text().strip()


# ── Grid Search nodes ─────────────────────────────────────────────────────────

class _GridSearchBase(BaseExecutionNode):
    """Shared evaluate logic for classifier/regressor grid search."""

    _MODELS: dict[str, str] = {}
    _SCORINGS: list[str] = []
    _DEFAULT_SCORING: str = ''
    _TASK: str = ''

    _UI_PROPS = frozenset({'model', 'target_column', 'feature_columns',
                            'cv_folds', 'scoring', 'n_jobs', 'param_grid'})

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('best_model', color=SKLEARN_PORT_COLOR)
        self.add_output('cv_results', color=PORT_COLORS['table'])

        self.add_combo_menu('model', 'Model', items=list(self._MODELS))
        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns',
                                   label='Feature Columns (blank=all numeric)',
                                   mode='multi')
        self._param_widget = ParamGridWidget(self.view, name='param_grid', label='')
        self.add_custom_widget(self._param_widget)
        self._add_int_spinbox('cv_folds', 'CV Folds', value=5, min_val=2, max_val=20)
        self.add_combo_menu('scoring', 'Scoring', items=self._SCORINGS)
        self._add_int_spinbox('n_jobs', 'Parallel Jobs',
                              value=-1, min_val=-1, max_val=64)

        self.output_values = {}

        # Initial widget rows for the default model.
        first_model = next(iter(self._MODELS), None)
        if first_model:
            self._update_estimator(first_model)

    def set_property(self, name, value, push_undo=True):
        super().set_property(name, value, push_undo)
        if name == 'model' and value in self._MODELS:
            self._update_estimator(value)

    def _update_estimator(self, model_label: str) -> None:
        try:
            cls = _import_class(self._MODELS[model_label])
        except Exception as e:
            print(f"[gridsearch] Could not import {self._MODELS[model_label]}: {e}")
            return
        curated = CURATED_PARAMS.get(self._MODELS[model_label])
        self._param_widget.set_estimator(cls, curated)

    def _get_input_df(self):
        port = self.inputs().get('train')
        if not port or not port.connected_ports():
            return None
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if isinstance(data, TableData):
            return data.df
        return None

    def evaluate(self):
        from sklearn.model_selection import GridSearchCV

        self.reset_progress()
        df = self._get_input_df()
        if df is None:
            self.mark_error()
            return False, "No training data connected"

        target = str(self.get_property('target_column') or '').strip()
        if not target or target not in df.columns:
            self.mark_error()
            return False, f"Target column '{target}' not found"

        model_label = str(self.get_property('model') or '').strip()
        if model_label not in self._MODELS:
            self.mark_error()
            return False, f"Unknown model: {model_label}"

        self.set_progress(15)

        feat_text = str(self.get_property('feature_columns') or '').strip()
        try:
            X, y, feature_names, used_columns = build_xy(df, target, feat_text)
        except ValueError as e:
            self.mark_error()
            return False, str(e)

        self.set_progress(25)

        try:
            cls = _import_class(self._MODELS[model_label])
            estimator = cls()
        except Exception as e:
            self.mark_error()
            return False, f"Could not construct {model_label}: {e}"

        try:
            param_grid = self._param_widget.build_param_grid()
        except Exception as e:
            self.mark_error()
            return False, f"Could not parse param grid: {e}"

        if not param_grid:
            self.mark_error()
            return False, ("No params to sweep — fill in at least one Values "
                            "cell (e.g. n_estimators: 50, 100, 200).")

        cv = int(self.get_property('cv_folds') or 5)
        scoring = str(self.get_property('scoring') or self._DEFAULT_SCORING)
        n_jobs = int(self.get_property('n_jobs') or -1)

        self.set_progress(40)

        try:
            gs = GridSearchCV(
                estimator, param_grid,
                cv=cv, scoring=scoring,
                n_jobs=n_jobs, refit=True,
                error_score='raise',
            )
            gs.fit(X, y)
        except Exception as e:
            self.mark_error()
            return False, f"Grid search failed: {e}"

        self.set_progress(85)

        # Build results table — one row per combo, with separate columns.
        cv_results = gs.cv_results_
        rows: dict[str, Any] = {
            'rank':            cv_results['rank_test_score'],
            'mean_test_score': cv_results['mean_test_score'],
            'std_test_score':  cv_results['std_test_score'],
            'mean_fit_time':   cv_results['mean_fit_time'],
        }
        for pname in param_grid:
            key = f'param_{pname}'
            if key in cv_results:
                rows[pname] = cv_results[key]
        results_df = pd.DataFrame(rows).sort_values('rank').reset_index(drop=True)

        self.output_values['best_model'] = SklearnModelData(
            payload=gs.best_estimator_,
            model_type=type(gs.best_estimator_).__name__,
            feature_names=feature_names,
            feature_columns=used_columns,
            target_name=target,
            score=float(gs.best_score_),
            task=self._TASK,
        )
        self.output_values['cv_results'] = TableData(payload=results_df)
        self.mark_clean()
        self.set_progress(100)
        return True, (f"Best {scoring}: {gs.best_score_:.4f} "
                      f"({len(results_df)} combos)")


class GridSearchClassifierNode(_GridSearchBase):
    """K-fold grid search across hyperparameters for a classifier.

    Pick a model, fill in values to sweep (one value = fixed; multiple =
    swept), run K-fold cross-validation.  Outputs the refit best estimator
    and a per-combo results table sorted by rank.

    Keywords: grid search, hyperparameter tuning, cross validation, ML
    """
    __identifier__ = 'plugins.ML.Evaluation'
    NODE_NAME = 'Grid Search Classifier'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    _MODELS = CLASSIFIER_MODELS
    _SCORINGS = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro',
                  'recall_macro', 'roc_auc', 'roc_auc_ovr', 'balanced_accuracy']
    _DEFAULT_SCORING = 'accuracy'
    _TASK = 'classification'


class GridSearchRegressorNode(_GridSearchBase):
    """K-fold grid search across hyperparameters for a regressor.

    Pick a model, fill in values to sweep (one value = fixed; multiple =
    swept), run K-fold cross-validation.  Outputs the refit best estimator
    and a per-combo results table sorted by rank.

    Keywords: grid search, hyperparameter tuning, cross validation, regression
    """
    __identifier__ = 'plugins.ML.Evaluation'
    NODE_NAME = 'Grid Search Regressor'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    _MODELS = REGRESSOR_MODELS
    _SCORINGS = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error',
                  'neg_root_mean_squared_error', 'explained_variance']
    _DEFAULT_SCORING = 'r2'
    _TASK = 'regression'
