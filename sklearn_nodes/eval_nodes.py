"""
eval_nodes.py
=============
Model evaluation nodes: predict, confusion matrix, classification report,
cross-validation.
"""
import numpy as np
import pandas as pd
from nodes.base import BaseExecutionNode, PORT_COLORS
from data_models import TableData, FigureData
from .ml_data import SklearnModelData, SKLEARN_PORT_COLOR, build_xy


class PredictNode(BaseExecutionNode):
    """
    Applies a trained model to new data and outputs predictions.

    Connects a trained model and a table. The node uses the model's stored
    feature_names to select the right columns automatically.

    Keywords: predict, inference, apply model, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Evaluation'
    NODE_NAME = 'Predict'
    PORT_SPEC = {'inputs': ['sklearn_model', 'table'], 'outputs': ['table']}

    def __init__(self):
        super().__init__()
        self.add_input('model', color=SKLEARN_PORT_COLOR)
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('result', color=PORT_COLORS['table'])
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

        model = model_data.payload
        target = model_data.target_name
        # Prefer the original source-column list (for ndarray-aware rebuild);
        # fall back to expanded feature_names for older models.
        source_columns = list(model_data.feature_columns or model_data.feature_names)

        missing = [c for c in source_columns if c not in df.columns]
        if missing:
            self.mark_error()
            return False, f"Missing columns: {', '.join(missing)}"

        self.set_progress(40)

        try:
            X, _, _, _ = build_xy(df, target='', feature_columns=source_columns,
                                   require_target=False)
        except ValueError as e:
            self.mark_error()
            return False, str(e)
        try:
            preds = model.predict(X)
        except Exception as e:
            self.mark_error()
            return False, f"Prediction failed: {e}"

        self.set_progress(80)

        result = df.copy()
        result['prediction'] = preds

        # If the table has the true target column, add a 'correct' column
        if target and target in df.columns:
            result['correct'] = result[target] == result['prediction']

        self.output_values['result'] = TableData(payload=result)
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


class ConfusionMatrixNode(BaseExecutionNode):
    """
    Generates a confusion matrix from predictions.

    Expects a table with true labels and predicted labels columns.
    Outputs a matrix table and a heatmap figure.

    Options:

    - **true_column** — column with true labels
    - **pred_column** — column with predicted labels
    - **normalize** — normalize matrix values (row-wise)

    Keywords: confusion matrix, accuracy, classification, evaluate, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Evaluation'
    NODE_NAME = 'Confusion Matrix'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table', 'figure']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('matrix', color=PORT_COLORS['table'])
        self.add_output('figure', color=PORT_COLORS['figure'])

        self._add_column_selector('true_column', label='True Labels', mode='single')
        self._add_column_selector('pred_column', label='Predictions', mode='single')
        self.add_combo_menu('normalize', 'Normalize', items=['False', 'True'])
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

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

        self.set_progress(30)

        y_true = df[true_col]
        y_pred = df[pred_col]
        labels = sorted(set(y_true.unique()) | set(y_pred.unique()), key=str)

        normalize = str(self.get_property('normalize') or 'False') == 'True'
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.where(row_sums > 0, cm / row_sums, 0)

        self.set_progress(60)

        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.index.name = 'True'
        cm_df.columns.name = 'Predicted'

        # Generate heatmap
        fig, ax = plt.subplots(figsize=(6, 5))
        fmt = '.2f' if normalize else 'd'
        sns.heatmap(cm_df, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    linewidths=0.5, linecolor='#333')
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
        fig.tight_layout()

        self.set_progress(90)
        self.output_values['matrix'] = TableData(payload=cm_df.reset_index())
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


class ClassificationReportNode(BaseExecutionNode):
    """
    Generates a classification report (precision, recall, F1, accuracy).

    Options:

    - **true_column** — column with true labels
    - **pred_column** — column with predicted labels

    Keywords: classification report, precision, recall, F1, accuracy, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Evaluation'
    NODE_NAME = 'Classification Report'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('report', color=PORT_COLORS['table'])

        self._add_column_selector('true_column', label='True Labels', mode='single')
        self._add_column_selector('pred_column', label='Predictions', mode='single')
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        from sklearn.metrics import classification_report

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

        self.set_progress(40)

        report = classification_report(
            df[true_col], df[pred_col], output_dict=True, zero_division=0)

        self.set_progress(80)

        report_df = pd.DataFrame(report).T
        report_df.index.name = 'class'
        report_df = report_df.reset_index()

        self.output_values['report'] = TableData(payload=report_df)
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


class CrossValidationNode(BaseExecutionNode):
    """
    Runs K-fold cross-validation on a model and dataset.

    Options:

    - **target_column** — the column to predict
    - **cv_folds** — number of folds (default 5)
    - **scoring** — metric to evaluate (accuracy, f1_macro, r2, etc.)

    Keywords: cross validation, k-fold, CV, model selection, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Evaluation'
    NODE_NAME = 'Cross Validation'
    PORT_SPEC = {'inputs': ['sklearn_model', 'table'], 'outputs': ['table']}

    def __init__(self):
        super().__init__()
        self.add_input('model', color=SKLEARN_PORT_COLOR)
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('scores', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_int_spinbox('cv_folds', 'Folds', value=5, min_val=2, max_val=50)
        self.add_combo_menu('scoring', 'Scoring',
                            items=['accuracy', 'f1_macro', 'f1_weighted',
                                   'precision_macro', 'recall_macro', 'r2',
                                   'neg_mean_squared_error'])
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        from sklearn.model_selection import cross_val_score
        from sklearn.base import clone

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

        self.set_progress(20)

        # Rebuild X with the same column expansion the model was trained on.
        source_columns = list(model_data.feature_columns or model_data.feature_names)
        try:
            X, y, _, _ = build_xy(df, target, source_columns)
        except ValueError as e:
            self.mark_error()
            return False, str(e)

        self.set_progress(30)

        model = clone(model_data.payload)
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        except Exception as e:
            self.mark_error()
            return False, f"Cross-validation failed: {e}"

        self.set_progress(90)

        scores_df = pd.DataFrame({
            'fold': list(range(1, len(scores) + 1)),
            'score': scores,
        })
        scores_df.loc[len(scores_df)] = {'fold': 'mean', 'score': scores.mean()}
        scores_df.loc[len(scores_df)] = {'fold': 'std', 'score': scores.std()}

        self.output_values['scores'] = TableData(payload=scores_df)
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


# ─── Model evaluation (auto-dispatch by task) ────────────────────────────────

class ModelEvaluationNode(BaseExecutionNode):
    """Comprehensive evaluation of a fitted model on a held-out table.

    Auto-detects task from the model's ``SklearnModelData.task`` field and
    computes the appropriate metric set.

    Classification metrics:
      - accuracy, balanced accuracy
      - precision / recall / f1 (macro)
      - matthews correlation, cohen kappa
      - roc_auc (binary or one-vs-rest), log_loss (when predict_proba available)

    Regression metrics:
      - r2, explained variance
      - rmse, mse, mae, median absolute error
      - mape (mean absolute percentage error)

    Output is a 2-column TableData (``metric``, ``value``).

    Keywords: model evaluation, metrics, classification, regression, ML
    """
    __identifier__ = 'plugins.ML.Evaluation'
    NODE_NAME = 'Model Evaluation'
    PORT_SPEC = {'inputs': ['sklearn_model', 'table'], 'outputs': ['table']}

    def __init__(self):
        super().__init__()
        self.add_input('model', color=SKLEARN_PORT_COLOR)
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('metrics', color=PORT_COLORS['table'])
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

        target = model_data.target_name
        if not target or target not in df.columns:
            self.mark_error()
            return False, f"Target column '{target}' not found in table"

        self.set_progress(20)

        source_columns = list(model_data.feature_columns or model_data.feature_names)
        try:
            X, y_true, _, _ = build_xy(df, target, source_columns)
        except ValueError as e:
            self.mark_error()
            return False, str(e)

        self.set_progress(40)

        model = model_data.payload
        try:
            y_pred = model.predict(X)
        except Exception as e:
            self.mark_error()
            return False, f"Predict failed: {e}"

        self.set_progress(70)

        task = (model_data.task or '').lower()
        if task == 'regression':
            metrics = self._regression_metrics(y_true, y_pred)
        else:
            metrics = self._classification_metrics(model, X, y_true, y_pred)

        df_out = pd.DataFrame(
            [(k, float(v)) for k, v in metrics.items()],
            columns=['metric', 'value'],
        )
        self.output_values['metrics'] = TableData(payload=df_out)
        self.mark_clean()
        self.set_progress(100)

        first = next(iter(metrics))
        return True, f"{task or 'classification'}: {first}={metrics[first]:.4f}"

    @staticmethod
    def _classification_metrics(model, X, y_true, y_pred):
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score,
            precision_score, recall_score, f1_score,
            matthews_corrcoef, cohen_kappa_score,
            roc_auc_score, log_loss,
        )
        metrics = {
            'accuracy':            accuracy_score(y_true, y_pred),
            'balanced_accuracy':   balanced_accuracy_score(y_true, y_pred),
            'precision_macro':     precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro':        recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro':            f1_score(y_true, y_pred, average='macro', zero_division=0),
            'matthews_corrcoef':   matthews_corrcoef(y_true, y_pred),
            'cohen_kappa':         cohen_kappa_score(y_true, y_pred),
        }
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X)
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                elif y_proba.ndim == 2 and y_proba.shape[1] > 2:
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        y_true, y_proba, multi_class='ovr')
                metrics['log_loss'] = log_loss(y_true, y_proba)
            except Exception:
                pass
        return metrics

    @staticmethod
    def _regression_metrics(y_true, y_pred):
        import numpy as np
        from sklearn.metrics import (
            r2_score, explained_variance_score,
            mean_squared_error, mean_absolute_error,
            median_absolute_error, mean_absolute_percentage_error,
        )
        mse = mean_squared_error(y_true, y_pred)
        return {
            'r2':                  r2_score(y_true, y_pred),
            'explained_variance':  explained_variance_score(y_true, y_pred),
            'rmse':                float(np.sqrt(mse)),
            'mse':                 mse,
            'mae':                 mean_absolute_error(y_true, y_pred),
            'median_ae':           median_absolute_error(y_true, y_pred),
            'mape':                mean_absolute_percentage_error(y_true, y_pred),
        }

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
