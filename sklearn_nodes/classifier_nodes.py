"""
classifier_nodes.py
===================
Classification model training nodes.  Regressors live in regressor_nodes.py.
"""
from nodes.base import PORT_COLORS

from .base_model_node import _BaseClassifierNode
from .ml_data import SKLEARN_PORT_COLOR


class RandomForestClassifierNode(_BaseClassifierNode):
    """
    Trains a Random Forest classifier.

    Options:

    - **target_column** — column to predict
    - **n_estimators** — number of trees (default 100)
    - **max_depth** — max tree depth (0 = unlimited)
    - **random_seed** — for reproducibility

    Keywords: random forest, ensemble, decision tree, classification, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Classification'
    NODE_NAME = 'Random Forest'
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
        from sklearn.ensemble import RandomForestClassifier
        n = int(self.get_property('n_estimators') or 100)
        depth = int(self.get_property('max_depth') or 0) or None
        seed = int(self.get_property('random_seed') or 42)
        return RandomForestClassifier(
            n_estimators=n, max_depth=depth,
            random_state=seed if seed > 0 else None, n_jobs=-1)


class SVMClassifierNode(_BaseClassifierNode):
    """
    Trains a Support Vector Machine classifier.

    Options:

    - **target_column** — column to predict
    - **kernel** — kernel type (rbf, linear, poly, sigmoid)
    - **C** — regularization parameter
    - **gamma** — kernel coefficient (scale or auto)

    Keywords: SVM, support vector machine, classification, kernel, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Classification'
    NODE_NAME = 'SVM Classifier'
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
        self.add_combo_menu('gamma', 'Gamma', items=['scale', 'auto'])
        self.output_values = {}

    def _build_model(self):
        from sklearn.svm import SVC
        kernel = str(self.get_property('kernel') or 'rbf')
        C = float(self.get_property('C') or 1.0)
        gamma = str(self.get_property('gamma') or 'scale')
        return SVC(kernel=kernel, C=C, gamma=gamma)


class KNNClassifierNode(_BaseClassifierNode):
    """
    Trains a K-Nearest Neighbors classifier.

    Options:

    - **target_column** — column to predict
    - **n_neighbors** — number of neighbors (default 5)
    - **weights** — weight function (uniform or distance)

    Keywords: KNN, k-nearest neighbors, classification, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Classification'
    NODE_NAME = 'KNN Classifier'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self._add_int_spinbox('n_neighbors', 'K Neighbors', value=5, min_val=1, max_val=500)
        self.add_combo_menu('weights', 'Weights', items=['uniform', 'distance'])
        self.output_values = {}

    def _build_model(self):
        from sklearn.neighbors import KNeighborsClassifier
        k = int(self.get_property('n_neighbors') or 5)
        w = str(self.get_property('weights') or 'uniform')
        return KNeighborsClassifier(n_neighbors=k, weights=w, n_jobs=-1)


class LogisticRegressionNode(_BaseClassifierNode):
    """
    Trains a Logistic Regression classifier.

    Options:

    - **target_column** — column to predict
    - **C** — inverse regularization strength
    - **max_iter** — maximum iterations
    - **solver** — optimization algorithm

    Keywords: logistic regression, classification, linear model, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Classification'
    NODE_NAME = 'Logistic Regression'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self._add_float_spinbox('C', 'C (Regularization)', value=1.0,
                                min_val=0.001, max_val=10000.0, step=0.1, decimals=3)
        self._add_int_spinbox('max_iter', 'Max Iterations', value=100, min_val=10, max_val=100000)
        self.add_combo_menu('solver', 'Solver', items=['lbfgs', 'liblinear', 'newton-cg', 'saga'])
        self.output_values = {}

    def _build_model(self):
        from sklearn.linear_model import LogisticRegression
        C = float(self.get_property('C') or 1.0)
        mi = int(self.get_property('max_iter') or 100)
        solver = str(self.get_property('solver') or 'lbfgs')
        return LogisticRegression(C=C, max_iter=mi, solver=solver, n_jobs=-1)


class GradientBoostingClassifierNode(_BaseClassifierNode):
    """
    Trains a Gradient Boosting classifier.

    Options:

    - **target_column** — column to predict
    - **n_estimators** — number of boosting stages (default 100)
    - **max_depth** — max depth of individual trees (default 3)
    - **learning_rate** — shrinkage factor (default 0.1)
    - **random_seed** — for reproducibility

    Keywords: gradient boosting, GBM, ensemble, classification, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Classification'
    NODE_NAME = 'Gradient Boosting'
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
        from sklearn.ensemble import GradientBoostingClassifier
        n = int(self.get_property('n_estimators') or 100)
        depth = int(self.get_property('max_depth') or 3)
        lr = float(self.get_property('learning_rate') or 0.1)
        seed = int(self.get_property('random_seed') or 42)
        return GradientBoostingClassifier(
            n_estimators=n, max_depth=depth, learning_rate=lr,
            random_state=seed if seed > 0 else None)


class AdaBoostClassifierNode(_BaseClassifierNode):
    """
    Trains an AdaBoost classifier.

    Options:

    - **target_column** — column to predict
    - **n_estimators** — number of weak learners (default 50)
    - **learning_rate** — weight applied to each classifier (default 1.0)
    - **random_seed** — for reproducibility

    Keywords: AdaBoost, boosting, ensemble, classification, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Classification'
    NODE_NAME = 'AdaBoost'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self._add_int_spinbox('n_estimators', 'Estimators', value=50, min_val=1, max_val=10000)
        self._add_float_spinbox('learning_rate', 'Learning Rate', value=1.0,
                                min_val=0.001, max_val=10.0, step=0.01, decimals=3)
        self._add_int_spinbox('random_seed', 'Random Seed', value=42, min_val=0, max_val=99999)
        self.output_values = {}

    def _build_model(self):
        from sklearn.ensemble import AdaBoostClassifier
        n = int(self.get_property('n_estimators') or 50)
        lr = float(self.get_property('learning_rate') or 1.0)
        seed = int(self.get_property('random_seed') or 42)
        return AdaBoostClassifier(
            n_estimators=n, learning_rate=lr,
            random_state=seed if seed > 0 else None)


class DecisionTreeClassifierNode(_BaseClassifierNode):
    """
    Trains a Decision Tree classifier.

    Options:

    - **target_column** — column to predict
    - **max_depth** — max tree depth (0 = unlimited)
    - **min_samples_split** — minimum samples to split a node (default 2)
    - **random_seed** — for reproducibility

    Keywords: decision tree, CART, classification, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Classification'
    NODE_NAME = 'Decision Tree'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self._add_int_spinbox('max_depth', 'Max Depth (0=auto)', value=0, min_val=0, max_val=100)
        self._add_int_spinbox('min_samples_split', 'Min Samples Split', value=2, min_val=2, max_val=1000)
        self._add_int_spinbox('random_seed', 'Random Seed', value=42, min_val=0, max_val=99999)
        self.output_values = {}

    def _build_model(self):
        from sklearn.tree import DecisionTreeClassifier
        depth = int(self.get_property('max_depth') or 0) or None
        mss = int(self.get_property('min_samples_split') or 2)
        seed = int(self.get_property('random_seed') or 42)
        return DecisionTreeClassifier(
            max_depth=depth, min_samples_split=mss,
            random_state=seed if seed > 0 else None)


class NaiveBayesNode(_BaseClassifierNode):
    """
    Trains a Gaussian Naive Bayes classifier.

    Options:

    - **target_column** — column to predict

    Keywords: naive bayes, Gaussian, probabilistic, classification, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Classification'
    NODE_NAME = 'Naive Bayes'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self.output_values = {}

    def _build_model(self):
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()


# ─── Additional classifiers ────────────────────────────────────────────────────

class ExtraTreesClassifierNode(_BaseClassifierNode):
    """Trains an Extra-Trees classifier (randomized RF variant — often faster).

    Keywords: extra trees, randomized forest, ensemble, classification, ML
    """
    __identifier__ = 'plugins.ML.Classification'
    NODE_NAME = 'Extra Trees'
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
        from sklearn.ensemble import ExtraTreesClassifier
        n = int(self.get_property('n_estimators') or 100)
        depth = int(self.get_property('max_depth') or 0) or None
        seed = int(self.get_property('random_seed') or 42)
        return ExtraTreesClassifier(
            n_estimators=n, max_depth=depth,
            random_state=seed if seed > 0 else None, n_jobs=-1)


class HistGradientBoostingClassifierNode(_BaseClassifierNode):
    """Trains a histogram-based gradient boosting classifier (sklearn's
    LightGBM-equivalent; very fast on large tables).

    Keywords: histogram gradient boosting, lightgbm, ensemble, classification
    """
    __identifier__ = 'plugins.ML.Classification'
    NODE_NAME = 'Hist Gradient Boosting'
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
        from sklearn.ensemble import HistGradientBoostingClassifier
        n = int(self.get_property('max_iter') or 100)
        lr = float(self.get_property('learning_rate') or 0.1)
        depth = int(self.get_property('max_depth') or 0) or None
        seed = int(self.get_property('random_seed') or 42)
        return HistGradientBoostingClassifier(
            max_iter=n, learning_rate=lr, max_depth=depth,
            random_state=seed if seed > 0 else None)


class MLPClassifierNode(_BaseClassifierNode):
    """Trains a multi-layer perceptron classifier.

    Hidden layer sizes are entered as a comma-separated list of ints, e.g.
    "100" for one 100-neuron layer or "100, 50" for two layers.

    Keywords: MLP, neural network, perceptron, classification, ML
    """
    __identifier__ = 'plugins.ML.Classification'
    NODE_NAME = 'MLP Classifier'
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
        from sklearn.neural_network import MLPClassifier
        hidden = self.get_property('hidden_layer_sizes') or '100'
        try:
            sizes = tuple(int(s.strip()) for s in str(hidden).split(',') if s.strip())
            if not sizes:
                sizes = (100,)
        except ValueError:
            sizes = (100,)
        return MLPClassifier(
            hidden_layer_sizes=sizes,
            activation=str(self.get_property('activation') or 'relu'),
            alpha=float(self.get_property('alpha') or 0.0001),
            max_iter=int(self.get_property('max_iter') or 200),
            random_state=int(self.get_property('random_seed') or 42) or None)


class RidgeClassifierNode(_BaseClassifierNode):
    """Trains a Ridge-regularized linear classifier (fast linear baseline).

    Keywords: ridge, linear, L2, classification, ML
    """
    __identifier__ = 'plugins.ML.Classification'
    NODE_NAME = 'Ridge Classifier'
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
        from sklearn.linear_model import RidgeClassifier
        return RidgeClassifier(
            alpha=float(self.get_property('alpha') or 1.0),
            random_state=int(self.get_property('random_seed') or 42) or None)


class LDANode(_BaseClassifierNode):
    """Trains a Linear Discriminant Analysis classifier.

    Keywords: LDA, linear discriminant, classification, ML
    """
    __identifier__ = 'plugins.ML.Classification'
    NODE_NAME = 'LDA'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self.add_combo_menu('solver', 'Solver', items=['svd', 'lsqr', 'eigen'])
        self.output_values = {}

    def _build_model(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis(
            solver=str(self.get_property('solver') or 'svd'))


class QDANode(_BaseClassifierNode):
    """Trains a Quadratic Discriminant Analysis classifier.

    Keywords: QDA, quadratic discriminant, classification, ML
    """
    __identifier__ = 'plugins.ML.Classification'
    NODE_NAME = 'QDA'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['sklearn_model', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('train', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_column_selector('feature_columns', label='Feature Columns (blank=all numeric)', mode='multi')
        self._add_float_spinbox('reg_param', 'Regularization', value=0.0,
                                min_val=0.0, max_val=1.0, step=0.01, decimals=4)
        self.output_values = {}

    def _build_model(self):
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        return QuadraticDiscriminantAnalysis(
            reg_param=float(self.get_property('reg_param') or 0.0))


class XGBoostClassifierNode(_BaseClassifierNode):
    """Trains an XGBoost classifier (gradient-boosted trees, optimized).

    Requires the ``xgboost`` package — install with ``pip install xgboost``.

    Keywords: xgboost, gradient boosting, ensemble, classification, ML
    """
    __identifier__ = 'plugins.ML.Classification'
    NODE_NAME = 'XGBoost Classifier'
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
            from xgboost import XGBClassifier
        except ImportError as e:
            raise RuntimeError(
                "xgboost is not installed.  Run `pip install xgboost`."
            ) from e
        return XGBClassifier(
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

