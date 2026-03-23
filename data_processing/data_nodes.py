"""
plugins/data_processing/data_nodes.py
=====================================
Data transformation nodes — blank subtraction, normalization, etc.
"""
import pandas as pd
import numpy as np
from data_models import TableData
from nodes.base import BaseExecutionNode, PORT_COLORS


def _read_upstream_df(node, port_name='in'):
    """Read a DataFrame from the upstream node without running evaluate."""
    in_port = node.inputs().get(port_name)
    if not in_port or not in_port.connected_ports():
        return None
    cp = in_port.connected_ports()[0]
    up_val = cp.node().output_values.get(cp.name())
    if isinstance(up_val, TableData):
        return up_val.df
    if isinstance(up_val, pd.DataFrame):
        return up_val
    return None


class BlankSubtractNode(BaseExecutionNode):
    """
    Subtract a reference row's value from all rows in a column.

    Common use: subtract background (BG) measurement from all cells.

    - **Reference Column** — the column containing group/cell labels (e.g. 'cell')
    - **Reference Value** — the label of the reference row (e.g. 'BG')
    - **Target Columns** — columns to subtract from (comma-separated, or leave empty for all numeric)

    The reference row's value is subtracted from every row in each target column.
    The reference row itself is kept (will become 0).

    Keywords: blank, subtract, background, correction, normalize, baseline, 空白, 扣除, 背景校正
    """
    __identifier__ = 'plugins.Plugins.data_processing'
    NODE_NAME = 'Blank Subtract'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('table', color=PORT_COLORS['table'])

        self._add_column_selector('ref_column', 'Reference Column')
        self.add_text_input('ref_value', 'Reference Value', text='BG')
        self.add_text_input('target_columns', 'Target Columns',
                            text='')  # empty = all numeric

    def on_input_connected(self, in_port, out_port):
        df = _read_upstream_df(self)
        if df is not None:
            self._refresh_column_selectors(df, 'ref_column')

    def evaluate(self):
        self.reset_progress()

        df = _read_upstream_df(self)
        if df is None:
            return False, "No input or not a table"
        df = df.copy()

        self._refresh_column_selectors(df, 'ref_column')
        self.set_progress(20)

        ref_col = str(self.get_property('ref_column')).strip()
        ref_val = str(self.get_property('ref_value')).strip()
        target_str = str(self.get_property('target_columns')).strip()

        if not ref_col or ref_col not in df.columns:
            return False, f"Reference column '{ref_col}' not found"

        # Find the reference row
        ref_rows = df[df[ref_col].astype(str).str.strip() == ref_val]
        if ref_rows.empty:
            return False, f"No row with {ref_col}='{ref_val}'"

        self.set_progress(40)

        # Determine target columns
        if target_str:
            targets = [c.strip() for c in target_str.split(',') if c.strip()]
            missing = [c for c in targets if c not in df.columns]
            if missing:
                return False, f"Columns not found: {missing}"
        else:
            targets = df.select_dtypes(include=[np.number]).columns.tolist()
            targets = [c for c in targets if c != ref_col]

        if not targets:
            return False, "No numeric target columns found"

        self.set_progress(60)

        # Subtract reference row's values
        ref_row = ref_rows.iloc[0]
        for col in targets:
            ref_value = ref_row[col]
            if pd.notna(ref_value):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce') - float(ref_value)
                except (ValueError, TypeError):
                    pass  # skip non-numeric columns

        self.output_values['table'] = TableData(payload=df)
        self.set_progress(100)
        self.mark_clean()
        return True, None


class RowNormalizeNode(BaseExecutionNode):
    """
    Normalize all rows by a reference row's value (divide instead of subtract).

    Useful for fold-change calculations: value / reference.

    - **Reference Column** — column containing group labels
    - **Reference Value** — label of the reference row (e.g. 'Control', 'BG')
    - **Target Columns** — columns to normalize (comma-separated, or empty for all numeric)

    Keywords: normalize, fold change, ratio, reference, control, 正規化, 倍數變化
    """
    __identifier__ = 'plugins.Plugins.data_processing'
    NODE_NAME = 'Row Normalize'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('table', color=PORT_COLORS['table'])

        self._add_column_selector('ref_column', 'Reference Column')
        self.add_text_input('ref_value', 'Reference Value', text='Control')
        self.add_text_input('target_columns', 'Target Columns', text='')

    def on_input_connected(self, in_port, out_port):
        df = _read_upstream_df(self)
        if df is not None:
            self._refresh_column_selectors(df, 'ref_column')

    def evaluate(self):
        self.reset_progress()

        df = _read_upstream_df(self)
        if df is None:
            return False, "No input or not a table"
        df = df.copy()

        self._refresh_column_selectors(df, 'ref_column')
        self.set_progress(20)

        ref_col = str(self.get_property('ref_column')).strip()
        ref_val = str(self.get_property('ref_value')).strip()
        target_str = str(self.get_property('target_columns')).strip()

        if not ref_col or ref_col not in df.columns:
            return False, f"Reference column '{ref_col}' not found"

        ref_rows = df[df[ref_col].astype(str).str.strip() == ref_val]
        if ref_rows.empty:
            return False, f"No row with {ref_col}='{ref_val}'"

        self.set_progress(40)

        if target_str:
            targets = [c.strip() for c in target_str.split(',') if c.strip()]
            missing = [c for c in targets if c not in df.columns]
            if missing:
                return False, f"Columns not found: {missing}"
        else:
            targets = df.select_dtypes(include=[np.number]).columns.tolist()
            targets = [c for c in targets if c != ref_col]

        if not targets:
            return False, "No numeric target columns found"

        self.set_progress(60)

        ref_row = ref_rows.iloc[0]
        for col in targets:
            ref_value = ref_row[col]
            if pd.notna(ref_value) and float(ref_value) != 0:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce') / float(ref_value)
                except (ValueError, TypeError):
                    pass

        self.output_values['table'] = TableData(payload=df)
        self.set_progress(100)
        self.mark_clean()
        return True, None


class DropRowsNode(BaseExecutionNode):
    """
    Drop rows where a column matches any of the specified values.

    - **Column** — which column to check
    - **Values to Drop** — comma-separated list of values to remove (e.g. 'BG, Artifact (BG)')

    Keywords: drop, remove, filter, exclude, delete rows, 刪除, 移除, 過濾
    """
    __identifier__ = 'plugins.Plugins.data_processing'
    NODE_NAME = 'Drop Rows'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}
    _collection_aware = True

    def __init__(self):
        super().__init__()
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('table', color=PORT_COLORS['table'])

        self._add_column_selector('column', 'Column')
        self.add_text_input('drop_values', 'Values to Drop', text='')

    def evaluate(self):
        self.reset_progress()

        df = _read_upstream_df(self)
        if df is None:
            return False, "No input or not a table"
        df = df.copy()

        self._refresh_column_selectors(df, 'column')
        self.set_progress(20)

        col = str(self.get_property('column')).strip()
        drop_str = str(self.get_property('drop_values')).strip()

        if not col or col not in df.columns:
            return False, f"Column '{col}' not found"

        if not drop_str:
            self.output_values['table'] = TableData(payload=df)
            self.set_progress(100)
            self.mark_clean()
            return True, None

        drop_vals = {v.strip() for v in drop_str.split(',') if v.strip()}

        self.set_progress(50)

        mask = df[col].astype(str).str.strip().isin(drop_vals)
        df = df[~mask].reset_index(drop=True)

        self.output_values['table'] = TableData(payload=df)
        self.set_progress(100)
        self.mark_clean()
        return True, None
