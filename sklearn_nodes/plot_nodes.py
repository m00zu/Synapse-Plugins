"""
plot_nodes.py
=============
ML visualization nodes: ROC curve, precision-recall, feature importance,
learning curve, cluster visualization, regression scatter.
"""
import numpy as np
import pandas as pd
from nodes.base import BaseExecutionNode, PORT_COLORS
from data_models import TableData, FigureData
from .ml_data import SklearnModelData, SKLEARN_PORT_COLOR, build_xy


class ROCCurveNode(BaseExecutionNode):
    """
    Plots a Receiver Operating Characteristic (ROC) curve with AUC.

    For binary classification, plots a single ROC curve. For multi-class,
    plots one-vs-rest curves for each class.

    Options:

    - **true_column** — column with true labels
    - **pred_column** — column with prediction probabilities or scores

    Keywords: ROC, AUC, receiver operating characteristic, classification, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Visualization'
    NODE_NAME = 'ROC Curve'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('figure', color=PORT_COLORS['figure'])

        self._add_column_selector('true_column', label='True Labels', mode='single')
        self._add_column_selector('pred_column', label='Pred / Prob Column', mode='single')
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        df = self._get_input_df()
        if df is None:
            self.mark_error()
            return False, "No table connected"

        true_col = str(self.get_property('true_column') or '').strip()
        pred_col = str(self.get_property('pred_column') or '').strip()

        if not true_col or true_col not in df.columns:
            self.mark_error()
            return False, f"True column '{true_col}' not found"
        if not pred_col or pred_col not in df.columns:
            self.mark_error()
            return False, f"Prediction column '{pred_col}' not found"

        self.set_progress(20)

        y_true = df[true_col]
        y_score = df[pred_col]
        classes = sorted(y_true.unique(), key=str)

        fig, ax = plt.subplots(figsize=(7, 5))

        if len(classes) <= 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=classes[-1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        else:
            # Multi-class one-vs-rest
            y_bin = label_binarize(y_true, classes=classes)
            for i, cls in enumerate(classes):
                if y_bin.shape[1] > i:
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score == cls)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, lw=2, label=f'{cls} (AUC = {roc_auc:.3f})')

        self.set_progress(70)

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        fig.tight_layout()

        self.output_values['figure'] = FigureData(payload=fig)
        self.mark_clean()
        self.set_progress(100)
        return True, None

    def _get_input_df(self):
        port = self.inputs().get('table')
        if not port or not port.connected_ports():
            return None
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if isinstance(data, TableData):
            return data.df
        return None


class PrecisionRecallCurveNode(BaseExecutionNode):
    """
    Plots a Precision-Recall curve with AUPRC.

    For binary classification, plots a single curve. For multi-class,
    plots one-vs-rest curves for each class.

    Options:

    - **true_column** — column with true labels
    - **pred_column** — column with prediction probabilities or scores

    Keywords: precision, recall, AUPRC, PR curve, classification, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Visualization'
    NODE_NAME = 'Precision-Recall Curve'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('figure', color=PORT_COLORS['figure'])

        self._add_column_selector('true_column', label='True Labels', mode='single')
        self._add_column_selector('pred_column', label='Pred / Prob Column', mode='single')
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        from sklearn.metrics import precision_recall_curve, average_precision_score
        from sklearn.preprocessing import label_binarize
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        df = self._get_input_df()
        if df is None:
            self.mark_error()
            return False, "No table connected"

        true_col = str(self.get_property('true_column') or '').strip()
        pred_col = str(self.get_property('pred_column') or '').strip()

        if not true_col or true_col not in df.columns:
            self.mark_error()
            return False, f"True column '{true_col}' not found"
        if not pred_col or pred_col not in df.columns:
            self.mark_error()
            return False, f"Prediction column '{pred_col}' not found"

        self.set_progress(20)

        y_true = df[true_col]
        y_score = df[pred_col]
        classes = sorted(y_true.unique(), key=str)

        fig, ax = plt.subplots(figsize=(7, 5))

        if len(classes) <= 2:
            precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=classes[-1])
            ap = average_precision_score(y_true == classes[-1], y_score)
            ax.plot(recall, precision, lw=2, label=f'PR (AP = {ap:.3f})')
        else:
            y_bin = label_binarize(y_true, classes=classes)
            for i, cls in enumerate(classes):
                if y_bin.shape[1] > i:
                    precision, recall, _ = precision_recall_curve(y_bin[:, i], (y_score == cls).astype(float))
                    ap = average_precision_score(y_bin[:, i], (y_score == cls).astype(float))
                    ax.plot(recall, precision, lw=2, label=f'{cls} (AP = {ap:.3f})')

        self.set_progress(70)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')
        fig.tight_layout()

        self.output_values['figure'] = FigureData(payload=fig)
        self.mark_clean()
        self.set_progress(100)
        return True, None

    def _get_input_df(self):
        port = self.inputs().get('table')
        if not port or not port.connected_ports():
            return None
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if isinstance(data, TableData):
            return data.df
        return None


class FeatureImportanceNode(BaseExecutionNode):
    """
    Plots feature importance from a trained model as a horizontal bar chart.

    Works with tree-based models (Random Forest, Gradient Boosting, etc.)
    that expose a `feature_importances_` attribute, or linear models with
    `coef_`.

    Keywords: feature importance, bar chart, model interpretation, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Visualization'
    NODE_NAME = 'Feature Importance'
    PORT_SPEC = {'inputs': ['sklearn_model'], 'outputs': ['figure', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('model', color=SKLEARN_PORT_COLOR)
        self.add_output('figure', color=PORT_COLORS['figure'])
        self.add_output('importance', color=PORT_COLORS['table'])
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        model_data = self._get_input('model', SklearnModelData)
        if model_data is None:
            self.mark_error()
            return False, "No model connected"

        model = model_data.payload
        features = model_data.feature_names

        self.set_progress(20)

        # Extract importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)
        else:
            self.mark_error()
            return False, "Model does not expose feature importances or coefficients"

        if not features:
            features = [f'feature_{i}' for i in range(len(importances))]

        self.set_progress(50)

        imp_df = pd.DataFrame({
            'feature': features,
            'importance': importances,
        }).sort_values('importance', ascending=True).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(7, max(5, len(features) * 0.3)))
        ax.barh(imp_df['feature'], imp_df['importance'], color='#3498db')
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importance ({model_data.model_type})')
        fig.tight_layout()

        self.set_progress(90)
        self.output_values['figure'] = FigureData(payload=fig)
        self.output_values['importance'] = TableData(
            payload=imp_df.sort_values('importance', ascending=False).reset_index(drop=True))
        self.mark_clean()
        self.set_progress(100)
        return True, None

    def _get_input(self, port_name, expected_type):
        port = self.inputs().get(port_name)
        if not port or not port.connected_ports():
            return None
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        return data if isinstance(data, expected_type) else None


class LearningCurveNode(BaseExecutionNode):
    """
    Plots training vs validation score as a function of training set size.

    Helps diagnose overfitting or underfitting.

    Options:

    - **target_column** — column to predict
    - **cv_folds** — number of cross-validation folds (default 5)
    - **scoring** — scoring metric (accuracy, f1_macro, r2, etc.)

    Keywords: learning curve, overfitting, underfitting, bias, variance, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Visualization'
    NODE_NAME = 'Learning Curve'
    PORT_SPEC = {'inputs': ['sklearn_model', 'table'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__()
        self.add_input('model', color=SKLEARN_PORT_COLOR)
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('figure', color=PORT_COLORS['figure'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_int_spinbox('cv_folds', 'CV Folds', value=5, min_val=2, max_val=50)
        self.add_combo_menu('scoring', 'Scoring',
                            items=['accuracy', 'f1_macro', 'f1_weighted',
                                   'r2', 'neg_mean_squared_error'])
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        from sklearn.model_selection import learning_curve
        from sklearn.base import clone
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        model_data = self._get_input('model', SklearnModelData)
        if model_data is None:
            self.mark_error()
            return False, "No model connected"

        df = self._get_input_df('table')
        if df is None:
            self.mark_error()
            return False, "No table connected"

        target = str(self.get_property('target_column') or '').strip()
        if not target or target not in df.columns:
            self.mark_error()
            return False, f"Target column '{target}' not found"

        cv = int(self.get_property('cv_folds') or 5)
        scoring = str(self.get_property('scoring') or 'accuracy')

        self.set_progress(10)

        # Rebuild X with the same column expansion the model was trained on.
        source_columns = list(model_data.feature_columns or model_data.feature_names)
        try:
            X, y, _, _ = build_xy(df, target, source_columns)
        except ValueError as e:
            self.mark_error()
            return False, str(e)

        self.set_progress(20)

        model = clone(model_data.payload)
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=cv, scoring=scoring,
                train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)
        except Exception as e:
            self.mark_error()
            return False, f"Learning curve failed: {e}"

        self.set_progress(80)

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color='#3498db')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color='#e74c3c')
        ax.plot(train_sizes, train_mean, 'o-', color='#3498db', label='Training score')
        ax.plot(train_sizes, val_mean, 'o-', color='#e74c3c', label='Validation score')
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel(scoring)
        ax.set_title('Learning Curve')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self.output_values['figure'] = FigureData(payload=fig)
        self.mark_clean()
        self.set_progress(100)
        return True, None

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


class ClusterVisualizationNode(BaseExecutionNode):
    """
    2D scatter plot, optionally colored by cluster or class labels.

    Pick the X / Y axes from existing columns (e.g. ``umap_0`` / ``umap_1``
    after running the UMAP node).  If left blank, PCA is run on all numeric
    columns to produce 2D coordinates automatically.

    Coloring is optional:
      - ``class_column``: a true class / category column (e.g. ``activity``,
        ``label``).  Takes precedence when set.  Legend shows the raw value;
        title reads "Class Visualization".
      - ``cluster_column``: integer cluster IDs from K-Means / DBSCAN /
        Agglomerative.  Used when ``class_column`` is blank.  Legend shows
        ``Cluster N`` (and ``-1`` is rendered as ``Noise`` for DBSCAN).
      - If neither selector is set, falls back to a column named ``cluster``
        if one exists.  Otherwise a plain (uncolored) scatter is drawn.

    Options:

    - **x_column** — column for X axis (blank = auto PCA on numeric columns)
    - **y_column** — column for Y axis (blank = auto PCA on numeric columns)
    - **cluster_column** — column with cluster labels (optional)
    - **class_column** — column with true class labels (overrides cluster_column)

    Keywords: cluster plot, class plot, scatter, PCA, UMAP, visualization, ML
    """
    __identifier__ = 'plugins.ML.Visualization'
    NODE_NAME = 'Cluster Visualization'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('figure', color=PORT_COLORS['figure'])

        self._add_column_selector('x_column', label='X Column (blank=PCA)', mode='single')
        self._add_column_selector('y_column', label='Y Column (blank=PCA)', mode='single')
        self._add_column_selector('cluster_column', label='Cluster Column', mode='single')
        self._add_column_selector('class_column',
                                   label='Class Column',
                                   mode='single')
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        df = self._get_input_df()
        if df is None:
            self.mark_error()
            return False, "No table connected"

        # Color column is optional. class_column takes precedence over
        # cluster_column; falls back to a 'cluster' column if present;
        # otherwise plot an uncolored scatter.
        class_col = str(self.get_property('class_column') or '').strip()
        cluster_col = str(self.get_property('cluster_column') or '').strip()
        if class_col and class_col in df.columns:
            color_col = class_col
            color_kind = 'class'
        elif cluster_col and cluster_col in df.columns:
            color_col = cluster_col
            color_kind = 'cluster'
        elif 'cluster' in df.columns:
            color_col = 'cluster'
            color_kind = 'cluster'
        else:
            color_col = None
            color_kind = None

        x_col = str(self.get_property('x_column') or '').strip()
        y_col = str(self.get_property('y_column') or '').strip()

        self.set_progress(20)

        use_pca = (not x_col or x_col not in df.columns or
                   not y_col or y_col not in df.columns)

        if use_pca:
            from sklearn.decomposition import PCA
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                           if c != color_col]
            if len(numeric_cols) < 2:
                self.mark_error()
                return False, "Need at least 2 numeric columns for visualization"

            pca = PCA(n_components=2)
            coords = pca.fit_transform(df[numeric_cols].values)
            x_vals = coords[:, 0]
            y_vals = coords[:, 1]
            x_label = f'PC1 ({pca.explained_variance_ratio_[0]:.1%})'
            y_label = f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
        else:
            x_vals = df[x_col].values
            y_vals = df[y_col].values
            x_label = x_col
            y_label = y_col

        self.set_progress(60)

        fig, ax = plt.subplots(figsize=(7, 5))

        if color_col is None:
            ax.scatter(x_vals, y_vals, alpha=0.7, s=30, edgecolors='none')
            title = 'Scatter'
        else:
            labels = df[color_col]
            unique_labels = sorted(set(labels), key=lambda x: (isinstance(x, str), x))
            cmap = plt.cm.get_cmap('tab10', max(len(unique_labels), 1))

            for i, lbl in enumerate(unique_labels):
                mask = labels == lbl
                color = cmap(i)
                if color_kind == 'cluster' and not isinstance(lbl, str):
                    label_str = 'Noise' if lbl == -1 else f'Cluster {lbl}'
                else:
                    label_str = str(lbl)
                ax.scatter(x_vals[mask], y_vals[mask], c=[color], label=label_str,
                           alpha=0.7, s=30, edgecolors='none')

            ax.legend(loc='best', fontsize=8)
            title = ('Class Visualization' if color_kind == 'class'
                     else 'Cluster Visualization')
            title = f'{title}  (by {color_col})'

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self.output_values['figure'] = FigureData(payload=fig)
        self.mark_clean()
        self.set_progress(100)
        return True, None

    def _get_input_df(self):
        port = self.inputs().get('table')
        if not port or not port.connected_ports():
            return None
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if isinstance(data, TableData):
            return data.df
        return None


class RegressionScatterNode(BaseExecutionNode):
    """
    Scatter plot of true vs predicted values with an identity line.

    Useful for evaluating regression model performance visually.

    Options:

    - **true_column** — column with true values
    - **pred_column** — column with predicted values

    Keywords: regression scatter, true vs predicted, residual, regression, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Visualization'
    NODE_NAME = 'Regression Scatter'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('figure', color=PORT_COLORS['figure'])

        self._add_column_selector('true_column', label='True Values', mode='single')
        self._add_column_selector('pred_column', label='Predictions', mode='single')
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        df = self._get_input_df()
        if df is None:
            self.mark_error()
            return False, "No table connected"

        true_col = str(self.get_property('true_column') or '').strip()
        pred_col = str(self.get_property('pred_column') or '').strip()

        if not true_col or true_col not in df.columns:
            self.mark_error()
            return False, f"True column '{true_col}' not found"
        if not pred_col or pred_col not in df.columns:
            self.mark_error()
            return False, f"Prediction column '{pred_col}' not found"

        self.set_progress(20)

        y_true = df[true_col].values.astype(float)
        y_pred = df[pred_col].values.astype(float)

        # Compute R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        self.set_progress(50)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='none', color='#3498db')

        # Identity line
        lims = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
        ]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='Identity')

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f'True ({true_col})')
        ax.set_ylabel(f'Predicted ({pred_col})')
        ax.set_title(f'True vs Predicted (R\u00b2 = {r2:.4f})')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self.output_values['figure'] = FigureData(payload=fig)
        self.mark_clean()
        self.set_progress(100)
        return True, None

    def _get_input_df(self):
        port = self.inputs().get('table')
        if not port or not port.connected_ports():
            return None
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if isinstance(data, TableData):
            return data.df
        return None
