"""
preprocess_nodes.py
===================
Data preprocessing nodes: train/test split, scaling, feature selection.
"""
import numpy as np
from nodes.base import BaseExecutionNode, PORT_COLORS
from data_models import TableData
from .ml_data import SklearnModelData, SKLEARN_PORT_COLOR


class TrainTestSplitNode(BaseExecutionNode):
    """
    Splits a table into training and testing sets.

    Options:

    - **target_column** — the column to predict
    - **test_size** — fraction of data for testing (0.0–1.0)
    - **random_seed** — for reproducibility (0 = random)
    - **stratify** — preserve class proportions in the split

    Keywords: train test split, holdout, validation, cross validation, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Preprocessing'
    NODE_NAME = 'Train/Test Split'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('train', color=PORT_COLORS['table'])
        self.add_output('test', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_float_spinbox('test_size', 'Test Size', value=0.2,
                                min_val=0.01, max_val=0.99, step=0.05, decimals=2)
        self._add_int_spinbox('random_seed', 'Random Seed', value=42, min_val=0, max_val=99999)
        self.add_combo_menu('stratify', 'Stratify', items=['True', 'False'])
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        from sklearn.model_selection import train_test_split

        df = self._get_input_df()
        if df is None:
            self.mark_error()
            return False, "No table connected"

        target = str(self.get_property('target_column') or '').strip()
        if not target or target not in df.columns:
            self.mark_error()
            return False, f"Target column '{target}' not found"

        test_size = float(self.get_property('test_size') or 0.2)
        seed = int(self.get_property('random_seed') or 42)
        stratify_on = str(self.get_property('stratify') or 'True') == 'True'

        self.set_progress(30)

        strat = None
        if stratify_on:
            counts = df[target].value_counts()
            n_classes = len(counts)
            min_per_class = int(counts.min()) if n_classes else 0
            # Fail fast with a readable hint instead of sklearn's giant dump.
            if n_classes >= 0.9 * len(df) or min_per_class < 2:
                self.mark_error()
                return False, (
                    f"Cannot stratify on '{target}': {n_classes} unique values "
                    f"in {len(df)} rows (min per class = {min_per_class}). "
                    f"Looks like an ID/continuous column — turn off Stratify "
                    f"or pick a class column."
                )
            strat = df[target]

        try:
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=seed if seed > 0 else None,
                stratify=strat)
        except ValueError as e:
            self.mark_error()
            msg = str(e).splitlines()[0][:240]
            return False, msg

        self.set_progress(80)
        self.output_values['train'] = TableData(payload=train_df.reset_index(drop=True))
        self.output_values['test'] = TableData(payload=test_df.reset_index(drop=True))
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


class StandardScalerNode(BaseExecutionNode):
    """
    Scales numeric columns to zero mean and unit variance.

    Outputs the scaled table and the fitted scaler model (for applying
    the same transform to test data).

    Options:

    - **columns** — columns to scale (blank = all numeric)

    Keywords: normalize, standardize, z-score, scale, preprocessing, ML
    """
    __identifier__ = 'plugins.ML.Preprocessing'
    NODE_NAME = 'Standard Scaler'
    PORT_SPEC = {'inputs': ['table', 'sklearn_model'], 'outputs': ['table', 'sklearn_model']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_input('fitted_scaler', color=SKLEARN_PORT_COLOR)
        self.add_output('scaled', color=PORT_COLORS['table'])
        self.add_output('scaler', color=SKLEARN_PORT_COLOR)

        self._add_column_selector('columns', label='Columns', mode='multi')
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        from sklearn.preprocessing import StandardScaler

        df = self._get_input_df('table')
        if df is None:
            self.mark_error()
            return False, "No table connected"

        cols_str = str(self.get_property('columns') or '').strip()
        if cols_str:
            cols = [c.strip() for c in cols_str.split(',') if c.strip() and c.strip() in df.columns]
        else:
            cols = list(df.select_dtypes(include=[np.number]).columns)

        if not cols:
            self.mark_error()
            return False, "No numeric columns to scale"

        self.set_progress(30)

        # Check for upstream fitted scaler (for applying to test data)
        scaler_data = self._get_input_model('fitted_scaler')
        if scaler_data and hasattr(scaler_data.payload, 'transform'):
            scaler = scaler_data.payload
            result = df.copy()
            result[cols] = scaler.transform(df[cols])
        else:
            scaler = StandardScaler()
            result = df.copy()
            result[cols] = scaler.fit_transform(df[cols])

        self.set_progress(80)
        self.output_values['scaled'] = TableData(payload=result)
        self.output_values['scaler'] = SklearnModelData(
            payload=scaler, model_type='StandardScaler',
            feature_names=cols, task='preprocessing')
        self.mark_clean()
        self.set_progress(100)
        return True, None

    def _get_input_df(self, port_name):
        port = self.inputs().get(port_name)
        if not port or not port.connected_ports():
            return None
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if isinstance(data, TableData):
            return data.df
        return None

    def _get_input_model(self, port_name):
        port = self.inputs().get(port_name)
        if not port or not port.connected_ports():
            return None
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if isinstance(data, SklearnModelData):
            return data
        return None


class FeatureSelectionNode(BaseExecutionNode):
    """
    Selects the top K features based on statistical tests.

    Options:

    - **target_column** — the column to predict
    - **k** — number of top features to keep
    - **method** — scoring function (f_classif, mutual_info_classif, f_regression)

    Keywords: feature selection, SelectKBest, mutual information, feature importance, ML
    """
    __identifier__ = 'plugins.ML.Preprocessing'
    NODE_NAME = 'Feature Selection'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('selected', color=PORT_COLORS['table'])
        self.add_output('scores', color=PORT_COLORS['table'])

        self._add_column_selector('target_column', label='Target Column', mode='single')
        self._add_int_spinbox('k', 'Top K Features', value=5, min_val=1, max_val=1000)
        self.add_combo_menu('method', 'Method',
                            items=['f_classif', 'mutual_info_classif', 'f_regression'])
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression
        import pandas as pd

        df = self._get_input_df()
        if df is None:
            self.mark_error()
            return False, "No table connected"

        target = str(self.get_property('target_column') or '').strip()
        if not target or target not in df.columns:
            self.mark_error()
            return False, f"Target column '{target}' not found"

        k = int(self.get_property('k') or 5)
        method = str(self.get_property('method') or 'f_classif')

        score_funcs = {
            'f_classif': f_classif,
            'mutual_info_classif': mutual_info_classif,
            'f_regression': f_regression,
        }
        score_func = score_funcs.get(method, f_classif)

        self.set_progress(30)

        X = df.drop(columns=[target]).select_dtypes(include=[np.number])
        y = df[target]
        k = min(k, X.shape[1])

        selector = SelectKBest(score_func, k=k)
        selector.fit(X, y)

        self.set_progress(60)

        selected_cols = X.columns[selector.get_support()].tolist()
        selected_df = df[selected_cols + [target]]

        scores_df = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_,
            'selected': selector.get_support(),
        }).sort_values('score', ascending=False).reset_index(drop=True)

        self.set_progress(90)
        self.output_values['selected'] = TableData(payload=selected_df)
        self.output_values['scores'] = TableData(payload=scores_df)
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


class MinMaxScalerNode(BaseExecutionNode):
    """
    Scales numeric columns to a given range (default 0–1).

    Outputs the scaled table and the fitted scaler model (for applying
    the same transform to test data).

    Options:

    - **columns** — columns to scale (blank = all numeric)

    Keywords: normalize, min-max, scale, range, preprocessing, ML
    """
    __identifier__ = 'plugins.ML.Preprocessing'
    NODE_NAME = 'MinMax Scaler'
    PORT_SPEC = {'inputs': ['table', 'sklearn_model'], 'outputs': ['table', 'sklearn_model']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_input('fitted_scaler', color=SKLEARN_PORT_COLOR)
        self.add_output('scaled', color=PORT_COLORS['table'])
        self.add_output('scaler', color=SKLEARN_PORT_COLOR)

        self._add_column_selector('columns', label='Columns', mode='multi')
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        from sklearn.preprocessing import MinMaxScaler

        df = self._get_input_df('table')
        if df is None:
            self.mark_error()
            return False, "No table connected"

        cols_str = str(self.get_property('columns') or '').strip()
        if cols_str:
            cols = [c.strip() for c in cols_str.split(',') if c.strip() and c.strip() in df.columns]
        else:
            cols = list(df.select_dtypes(include=[np.number]).columns)

        if not cols:
            self.mark_error()
            return False, "No numeric columns to scale"

        self.set_progress(30)

        # Check for upstream fitted scaler (for applying to test data)
        scaler_data = self._get_input_model('fitted_scaler')
        if scaler_data and hasattr(scaler_data.payload, 'transform'):
            scaler = scaler_data.payload
            result = df.copy()
            result[cols] = scaler.transform(df[cols])
        else:
            scaler = MinMaxScaler()
            result = df.copy()
            result[cols] = scaler.fit_transform(df[cols])

        self.set_progress(80)
        self.output_values['scaled'] = TableData(payload=result)
        self.output_values['scaler'] = SklearnModelData(
            payload=scaler, model_type='MinMaxScaler',
            feature_names=cols, task='preprocessing')
        self.mark_clean()
        self.set_progress(100)
        return True, None

    def _get_input_df(self, port_name):
        port = self.inputs().get(port_name)
        if not port or not port.connected_ports():
            return None
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if isinstance(data, TableData):
            return data.df
        return None

    def _get_input_model(self, port_name):
        port = self.inputs().get(port_name)
        if not port or not port.connected_ports():
            return None
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if isinstance(data, SklearnModelData):
            return data
        return None


class LabelEncoderNode(BaseExecutionNode):
    """
    Encodes categorical columns to integer labels.

    Each unique value in the selected columns is mapped to an integer
    (0, 1, 2, ...). Useful for converting string labels before training.

    Options:

    - **columns** — columns to encode (comma-separated)

    Keywords: label encoder, categorical, encode, ordinal, preprocessing, ML
    """
    __identifier__ = 'plugins.ML.Preprocessing'
    NODE_NAME = 'Label Encoder'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('encoded', color=PORT_COLORS['table'])

        self._add_column_selector('columns', label='Columns', mode='multi')
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        from sklearn.preprocessing import LabelEncoder
        import pandas as pd

        df = self._get_input_df()
        if df is None:
            self.mark_error()
            return False, "No table connected"

        cols_str = str(self.get_property('columns') or '').strip()
        if not cols_str:
            self.mark_error()
            return False, "No columns specified"

        cols = [c.strip() for c in cols_str.split(',') if c.strip() and c.strip() in df.columns]
        if not cols:
            self.mark_error()
            return False, "None of the specified columns found in table"

        self.set_progress(30)

        result = df.copy()
        for col in cols:
            le = LabelEncoder()
            result[col] = le.fit_transform(result[col].astype(str))

        self.set_progress(90)
        self.output_values['encoded'] = TableData(payload=result)
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
