"""
regressor_nodes.py
==================
Regression model training nodes.  Classifiers live in classifier_nodes.py.
Each subclasses ``_BaseRegressorNode`` (from base_model_node.py) which
handles the train table → ``X``/``y`` → fit → ``SklearnModelData`` plumbing.
"""
from nodes.base import PORT_COLORS

from .base_model_node import _BaseRegressorNode
from .ml_data import SKLEARN_PORT_COLOR


class LinearRegressionNode(_BaseRegressorNode):
    """Trains an ordinary least-squares Linear Regression model.

    Options:
      - **target_column** — column to predict
      - **feature_columns** — feature columns (blank → all numeric)
      - **fit_intercept** — whether to calculate the intercept

    Keywords: linear regression, OLS, regression, ML
    """
    __identifier__ = 'plugins.ML.Regression'
    NODE_NAME = 'Linear Regression'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self.add_combo_menu('fit_intercept', 'Fit Intercept', items=['True', 'False'])
        self.output_values = {}

    def _build_model(self):
        from sklearn.linear_model import LinearRegression as LR
        fit_int = str(self.get_property('fit_intercept') or 'True') == 'True'
        return LR(fit_intercept=fit_int, n_jobs=-1)


class RandomForestRegressorNode(_BaseRegressorNode):
    """Trains a Random Forest regressor.

    Keywords: random forest, ensemble, regression, ML
    """
    __identifier__ = 'plugins.ML.Regression'
    NODE_NAME = 'RF Regressor'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self._add_int_spinbox('n_estimators', 'Trees', value=100, min_val=1, max_val=10000)
        self._add_int_spinbox('max_depth', 'Max Depth (0=auto)', value=0, min_val=0, max_val=100)
        self._add_int_spinbox('random_seed', 'Random Seed', value=42, min_val=0, max_val=99999)
        self.output_values = {}

    def _build_model(self):
        from sklearn.ensemble import RandomForestRegressor
        n = int(self.get_property('n_estimators') or 100)
        depth = int(self.get_property('max_depth') or 0) or None
        seed = int(self.get_property('random_seed') or 42)
        return RandomForestRegressor(
            n_estimators=n, max_depth=depth,
            random_state=seed if seed > 0 else None, n_jobs=-1)


class SVRNode(_BaseRegressorNode):
    """Trains a Support Vector Regression model.

    Keywords: SVR, support vector regression, kernel, ML
    """
    __identifier__ = 'plugins.ML.Regression'
    NODE_NAME = 'SVR'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self.add_combo_menu('kernel', 'Kernel', items=['rbf', 'linear', 'poly', 'sigmoid'])
        self._add_float_spinbox('C', 'C (Regularization)', value=1.0,
                                min_val=0.001, max_val=10000.0, step=0.1, decimals=3)
        self._add_float_spinbox('epsilon', 'Epsilon', value=0.1,
                                min_val=0.0, max_val=100.0, step=0.01, decimals=3)
        self.output_values = {}

    def _build_model(self):
        from sklearn.svm import SVR
        kernel = str(self.get_property('kernel') or 'rbf')
        C = float(self.get_property('C') or 1.0)
        eps = float(self.get_property('epsilon') or 0.1)
        return SVR(kernel=kernel, C=C, epsilon=eps)


class GradientBoostingRegressorNode(_BaseRegressorNode):
    """Trains a Gradient Boosting regressor.

    Keywords: gradient boosting, GBM, ensemble, regression, ML
    """
    __identifier__ = 'plugins.ML.Regression'
    NODE_NAME = 'GB Regressor'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self._add_int_spinbox('n_estimators', 'Estimators', value=100, min_val=1, max_val=10000)
        self._add_int_spinbox('max_depth', 'Max Depth', value=3, min_val=1, max_val=100)
        self._add_float_spinbox('learning_rate', 'Learning Rate', value=0.1,
                                min_val=0.001, max_val=10.0, step=0.01, decimals=3)
        self._add_int_spinbox('random_seed', 'Random Seed', value=42, min_val=0, max_val=99999)
        self.output_values = {}

    def _build_model(self):
        from sklearn.ensemble import GradientBoostingRegressor
        n = int(self.get_property('n_estimators') or 100)
        depth = int(self.get_property('max_depth') or 3)
        lr = float(self.get_property('learning_rate') or 0.1)
        seed = int(self.get_property('random_seed') or 42)
        return GradientBoostingRegressor(
            n_estimators=n, max_depth=depth, learning_rate=lr,
            random_state=seed if seed > 0 else None)


# ─── Additional regressors ────────────────────────────────────────────────────

class RidgeRegressorNode(_BaseRegressorNode):
    """Trains a Ridge (L2-regularized) regression model.

    Keywords: ridge, L2, regularization, regression, ML
    """
    __identifier__ = 'plugins.ML.Regression'
    NODE_NAME = 'Ridge'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self._add_float_spinbox('alpha', 'Alpha (L2)', value=1.0,
                                min_val=0.0, max_val=10000.0, step=0.1, decimals=3)
        self._add_int_spinbox('random_seed', 'Random Seed', value=42, min_val=0, max_val=99999)
        self.output_values = {}

    def _build_model(self):
        from sklearn.linear_model import Ridge
        return Ridge(
            alpha=float(self.get_property('alpha') or 1.0),
            random_state=int(self.get_property('random_seed') or 42) or None)


class LassoNode(_BaseRegressorNode):
    """Trains a Lasso (L1-regularized) regression model.

    Keywords: lasso, L1, sparse, regression, ML
    """
    __identifier__ = 'plugins.ML.Regression'
    NODE_NAME = 'Lasso'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self._add_float_spinbox('alpha', 'Alpha (L1)', value=1.0,
                                min_val=0.0, max_val=10000.0, step=0.1, decimals=3)
        self._add_int_spinbox('max_iter', 'Max Iterations', value=1000,
                              min_val=10, max_val=1000000)
        self._add_int_spinbox('random_seed', 'Random Seed', value=42, min_val=0, max_val=99999)
        self.output_values = {}

    def _build_model(self):
        from sklearn.linear_model import Lasso
        return Lasso(
            alpha=float(self.get_property('alpha') or 1.0),
            max_iter=int(self.get_property('max_iter') or 1000),
            random_state=int(self.get_property('random_seed') or 42) or None)


class ElasticNetNode(_BaseRegressorNode):
    """Trains an Elastic Net (mixed L1/L2) regression model.

    Keywords: elastic net, L1, L2, regression, ML
    """
    __identifier__ = 'plugins.ML.Regression'
    NODE_NAME = 'Elastic Net'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self._add_float_spinbox('alpha', 'Alpha', value=1.0,
                                min_val=0.0, max_val=10000.0, step=0.1, decimals=3)
        self._add_float_spinbox('l1_ratio', 'L1 Ratio', value=0.5,
                                min_val=0.0, max_val=1.0, step=0.05, decimals=3)
        self._add_int_spinbox('max_iter', 'Max Iterations', value=1000,
                              min_val=10, max_val=1000000)
        self._add_int_spinbox('random_seed', 'Random Seed', value=42, min_val=0, max_val=99999)
        self.output_values = {}

    def _build_model(self):
        from sklearn.linear_model import ElasticNet
        return ElasticNet(
            alpha=float(self.get_property('alpha') or 1.0),
            l1_ratio=float(self.get_property('l1_ratio') or 0.5),
            max_iter=int(self.get_property('max_iter') or 1000),
            random_state=int(self.get_property('random_seed') or 42) or None)


class BayesianRidgeNode(_BaseRegressorNode):
    """Trains a Bayesian Ridge regression model (uncertainty-aware linear).

    Keywords: bayesian, ridge, uncertainty, regression, ML
    """
    __identifier__ = 'plugins.ML.Regression'
    NODE_NAME = 'Bayesian Ridge'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self._add_int_spinbox('max_iter', 'Max Iterations', value=300,
                              min_val=10, max_val=100000)
        self.output_values = {}

    def _build_model(self):
        from sklearn.linear_model import BayesianRidge
        # sklearn renamed n_iter -> max_iter in 1.3.  Try the new name first,
        # fall back gracefully so we work on slightly older sklearns.
        n = int(self.get_property('max_iter') or 300)
        try:
            return BayesianRidge(max_iter=n)
        except TypeError:
            return BayesianRidge(n_iter=n)


class KNNRegressorNode(_BaseRegressorNode):
    """Trains a K-Nearest-Neighbours regressor.

    Keywords: KNN, k-nearest, neighbours, regression, ML
    """
    __identifier__ = 'plugins.ML.Regression'
    NODE_NAME = 'KNN Regressor'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self._add_int_spinbox('n_neighbors', 'Neighbours', value=5, min_val=1, max_val=10000)
        self.add_combo_menu('weights', 'Weights', items=['uniform', 'distance'])
        self.output_values = {}

    def _build_model(self):
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(
            n_neighbors=int(self.get_property('n_neighbors') or 5),
            weights=str(self.get_property('weights') or 'uniform'),
            n_jobs=-1)


class ExtraTreesRegressorNode(_BaseRegressorNode):
    """Trains an Extra-Trees regressor.

    Keywords: extra trees, randomized forest, ensemble, regression, ML
    """
    __identifier__ = 'plugins.ML.Regression'
    NODE_NAME = 'Extra Trees Regressor'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self._add_int_spinbox('n_estimators', 'Trees', value=100, min_val=1, max_val=10000)
        self._add_int_spinbox('max_depth', 'Max Depth (0=auto)', value=0, min_val=0, max_val=100)
        self._add_int_spinbox('random_seed', 'Random Seed', value=42, min_val=0, max_val=99999)
        self.output_values = {}

    def _build_model(self):
        from sklearn.ensemble import ExtraTreesRegressor
        n = int(self.get_property('n_estimators') or 100)
        depth = int(self.get_property('max_depth') or 0) or None
        seed = int(self.get_property('random_seed') or 42)
        return ExtraTreesRegressor(
            n_estimators=n, max_depth=depth,
            random_state=seed if seed > 0 else None, n_jobs=-1)


class HistGradientBoostingRegressorNode(_BaseRegressorNode):
    """Trains a histogram-based gradient boosting regressor (very fast on
    large tables).

    Keywords: histogram gradient boosting, lightgbm, ensemble, regression
    """
    __identifier__ = 'plugins.ML.Regression'
    NODE_NAME = 'Hist Gradient Boosting Regressor'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self._add_int_spinbox('max_iter', 'Iterations', value=100, min_val=1, max_val=10000)
        self._add_float_spinbox('learning_rate', 'Learning Rate', value=0.1,
                                min_val=0.001, max_val=10.0, step=0.01, decimals=3)
        self._add_int_spinbox('max_depth', 'Max Depth (0=auto)', value=0, min_val=0, max_val=100)
        self._add_int_spinbox('random_seed', 'Random Seed', value=42, min_val=0, max_val=99999)
        self.output_values = {}

    def _build_model(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        n = int(self.get_property('max_iter') or 100)
        lr = float(self.get_property('learning_rate') or 0.1)
        depth = int(self.get_property('max_depth') or 0) or None
        seed = int(self.get_property('random_seed') or 42)
        return HistGradientBoostingRegressor(
            max_iter=n, learning_rate=lr, max_depth=depth,
            random_state=seed if seed > 0 else None)


class MLPRegressorNode(_BaseRegressorNode):
    """Trains a multi-layer perceptron regressor.

    Hidden layer sizes are entered as a comma-separated list of ints, e.g.
    "100" for one 100-neuron layer or "100, 50" for two layers.

    Keywords: MLP, neural network, perceptron, regression, ML
    """
    __identifier__ = 'plugins.ML.Regression'
    NODE_NAME = 'MLP Regressor'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self.add_text_input('hidden_layer_sizes', 'Hidden Layers', text='100')
        self.add_combo_menu('activation', 'Activation',
                            items=['relu', 'tanh', 'logistic', 'identity'])
        self._add_float_spinbox('alpha', 'L2 Penalty', value=0.0001,
                                min_val=0.0, max_val=1.0, step=0.0001, decimals=5)
        self._add_int_spinbox('max_iter', 'Max Iterations', value=200,
                              min_val=10, max_val=100000)
        self._add_int_spinbox('random_seed', 'Random Seed', value=42, min_val=0, max_val=99999)
        self.output_values = {}

    def _build_model(self):
        from sklearn.neural_network import MLPRegressor
        hidden = self.get_property('hidden_layer_sizes') or '100'
        try:
            sizes = tuple(int(s.strip()) for s in str(hidden).split(',') if s.strip())
            if not sizes:
                sizes = (100,)
        except ValueError:
            sizes = (100,)
        return MLPRegressor(
            hidden_layer_sizes=sizes,
            activation=str(self.get_property('activation') or 'relu'),
            alpha=float(self.get_property('alpha') or 0.0001),
            max_iter=int(self.get_property('max_iter') or 200),
            random_state=int(self.get_property('random_seed') or 42) or None)


class DecisionTreeRegressorNode(_BaseRegressorNode):
    """Trains a single Decision Tree regressor.

    Keywords: decision tree, CART, regression, ML
    """
    __identifier__ = 'plugins.ML.Regression'
    NODE_NAME = 'Decision Tree Regressor'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self._add_int_spinbox('max_depth', 'Max Depth (0=auto)', value=0, min_val=0, max_val=100)
        self._add_int_spinbox('min_samples_split', 'Min Samples Split',
                              value=2, min_val=2, max_val=1000)
        self._add_int_spinbox('random_seed', 'Random Seed', value=42, min_val=0, max_val=99999)
        self.output_values = {}

    def _build_model(self):
        from sklearn.tree import DecisionTreeRegressor
        depth = int(self.get_property('max_depth') or 0) or None
        return DecisionTreeRegressor(
            max_depth=depth,
            min_samples_split=int(self.get_property('min_samples_split') or 2),
            random_state=int(self.get_property('random_seed') or 42) or None)


class XGBoostRegressorNode(_BaseRegressorNode):
    """Trains an XGBoost regressor (gradient-boosted trees, optimized).

    Requires the ``xgboost`` package — install with ``pip install xgboost``.

    Keywords: xgboost, gradient boosting, ensemble, regression, ML
    """
    __identifier__ = 'plugins.ML.Regression'
    NODE_NAME = 'XGBoost Regressor'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self._add_int_spinbox('n_estimators', 'Trees', value=100, min_val=1, max_val=10000)
        self._add_int_spinbox('max_depth', 'Max Depth', value=6, min_val=1, max_val=100)
        self._add_float_spinbox('learning_rate', 'Learning Rate', value=0.3,
                                min_val=0.001, max_val=10.0, step=0.01, decimals=3)
        self._add_float_spinbox('subsample', 'Subsample', value=1.0,
                                min_val=0.1, max_val=1.0, step=0.05, decimals=2)
        self._add_float_spinbox('colsample_bytree', 'Col Subsample', value=1.0,
                                min_val=0.1, max_val=1.0, step=0.05, decimals=2)
        self._add_float_spinbox('reg_alpha', 'L1 (alpha)', value=0.0,
                                min_val=0.0, max_val=1000.0, step=0.1, decimals=3)
        self._add_float_spinbox('reg_lambda', 'L2 (lambda)', value=1.0,
                                min_val=0.0, max_val=1000.0, step=0.1, decimals=3)
        self._add_int_spinbox('random_seed', 'Random Seed', value=42, min_val=0, max_val=99999)
        self.output_values = {}

    def _build_model(self):
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise RuntimeError(
                "xgboost is not installed.  Run `pip install xgboost`."
            ) from e
        return XGBRegressor(
            n_estimators=int(self.get_property('n_estimators') or 100),
            max_depth=int(self.get_property('max_depth') or 6),
            learning_rate=float(self.get_property('learning_rate') or 0.3),
            subsample=float(self.get_property('subsample') or 1.0),
            colsample_bytree=float(self.get_property('colsample_bytree') or 1.0),
            reg_alpha=float(self.get_property('reg_alpha') or 0.0),
            reg_lambda=float(self.get_property('reg_lambda') or 1.0),
            random_state=int(self.get_property('random_seed') or 42) or None,
            n_jobs=-1,
        )
