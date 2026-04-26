"""
cluster_nodes.py
================
Clustering nodes: K-Means, DBSCAN, Agglomerative.
"""
import numpy as np
import pandas as pd
from nodes.base import BaseExecutionNode, PORT_COLORS
from data_models import TableData
from .ml_data import SklearnModelData, SKLEARN_PORT_COLOR, build_xy


class KMeansNode(BaseExecutionNode):
    """
    Clusters data using K-Means algorithm.

    Adds a 'cluster' column to the output table with the assigned cluster
    label for each row. Also outputs the fitted model.

    Options:

    - **columns** — columns to cluster on (blank = all numeric)
    - **n_clusters** — number of clusters (default 3)
    - **random_seed** — for reproducibility

    Keywords: k-means, clustering, unsupervised, centroid, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Clustering'
    NODE_NAME = 'K-Means'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table', 'sklearn_model']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('result', color=PORT_COLORS['table'])
        self.add_output('model', color=SKLEARN_PORT_COLOR)

        self._add_column_selector('columns', label='Columns', mode='multi')
        self._add_int_spinbox('n_clusters', 'Clusters', value=3, min_val=2, max_val=1000)
        self._add_int_spinbox('random_seed', 'Random Seed', value=42, min_val=0, max_val=99999)
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        from sklearn.cluster import KMeans

        df = self._get_input_df()
        if df is None:
            self.mark_error()
            return False, "No table connected"

        feat_text = str(self.get_property('columns') or '').strip()
        try:
            X, _, feature_names, used_columns = build_xy(
                df, target='', feature_columns=feat_text,
                require_target=False,
            )
        except ValueError as e:
            self.mark_error()
            return False, str(e)

        self.set_progress(20)

        k = int(self.get_property('n_clusters') or 3)
        seed = int(self.get_property('random_seed') or 42)

        model = KMeans(n_clusters=k, random_state=seed if seed > 0 else None, n_init='auto')

        try:
            labels = model.fit_predict(X)
        except Exception as e:
            self.mark_error()
            return False, f"Clustering failed: {e}"

        self.set_progress(80)

        result = df.copy()
        result['cluster'] = labels

        self.output_values['result'] = TableData(payload=result)
        self.output_values['model'] = SklearnModelData(
            payload=model, model_type='KMeans',
            feature_names=feature_names, feature_columns=used_columns,
            task='clustering')
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


class DBSCANNode(BaseExecutionNode):
    """
    Clusters data using the DBSCAN density-based algorithm.

    Adds a 'cluster' column to the output table. Noise points are
    labelled -1.

    Options:

    - **columns** — columns to cluster on (blank = all numeric)
    - **eps** — maximum distance between neighbours (default 0.5)
    - **min_samples** — minimum points to form a cluster (default 5)

    Keywords: DBSCAN, density, clustering, unsupervised, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Clustering'
    NODE_NAME = 'DBSCAN'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('columns', label='Columns', mode='multi')
        self._add_float_spinbox('eps', 'Epsilon', value=0.5,
                                min_val=0.001, max_val=1000.0, step=0.1, decimals=3)
        self._add_int_spinbox('min_samples', 'Min Samples', value=5, min_val=1, max_val=1000)
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        from sklearn.cluster import DBSCAN

        df = self._get_input_df()
        if df is None:
            self.mark_error()
            return False, "No table connected"

        feat_text = str(self.get_property('columns') or '').strip()
        try:
            X, _, feature_names, used_columns = build_xy(
                df, target='', feature_columns=feat_text,
                require_target=False,
            )
        except ValueError as e:
            self.mark_error()
            return False, str(e)

        self.set_progress(20)

        eps = float(self.get_property('eps') or 0.5)
        min_samples = int(self.get_property('min_samples') or 5)

        model = DBSCAN(eps=eps, min_samples=min_samples)

        try:
            labels = model.fit_predict(X)
        except Exception as e:
            self.mark_error()
            return False, f"Clustering failed: {e}"

        self.set_progress(80)

        result = df.copy()
        result['cluster'] = labels

        self.output_values['result'] = TableData(payload=result)
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


class AgglomerativeNode(BaseExecutionNode):
    """
    Clusters data using Agglomerative (hierarchical) clustering.

    Adds a 'cluster' column to the output table.

    Options:

    - **columns** — columns to cluster on (blank = all numeric)
    - **n_clusters** — number of clusters (default 3)
    - **linkage** — linkage criterion (ward, complete, average, single)

    Keywords: agglomerative, hierarchical, clustering, dendrogram, ML, machine learning
    """
    __identifier__ = 'plugins.ML.Clustering'
    NODE_NAME = 'Agglomerative'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        self.add_output('result', color=PORT_COLORS['table'])

        self._add_column_selector('columns', label='Columns', mode='multi')
        self._add_int_spinbox('n_clusters', 'Clusters', value=3, min_val=2, max_val=1000)
        self.add_combo_menu('linkage', 'Linkage', items=['ward', 'complete', 'average', 'single'])
        self.output_values = {}

    def evaluate(self):
        self.reset_progress()
        from sklearn.cluster import AgglomerativeClustering

        df = self._get_input_df()
        if df is None:
            self.mark_error()
            return False, "No table connected"

        feat_text = str(self.get_property('columns') or '').strip()
        try:
            X, _, feature_names, used_columns = build_xy(
                df, target='', feature_columns=feat_text,
                require_target=False,
            )
        except ValueError as e:
            self.mark_error()
            return False, str(e)

        self.set_progress(20)

        k = int(self.get_property('n_clusters') or 3)
        linkage = str(self.get_property('linkage') or 'ward')

        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)

        try:
            labels = model.fit_predict(X)
        except Exception as e:
            self.mark_error()
            return False, f"Clustering failed: {e}"

        self.set_progress(80)

        result = df.copy()
        result['cluster'] = labels

        self.output_values['result'] = TableData(payload=result)
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
