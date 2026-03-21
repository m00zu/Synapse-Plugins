"""
nodes/analysis_nodes.py
=======================
Statistical analysis nodes.
"""
import NodeGraphQt, json
from data_models import TableData, FigureData, StatData
from PIL import Image
import pandas as pd
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
from NodeGraphQt.nodes.base_node import NodeBaseWidget
from nodes.base import BaseExecutionNode, PORT_COLORS


class DataSummaryNode(BaseExecutionNode):
    """
    Computes pixel intensity histograms for images or descriptive statistics for DataFrames.

    Inputs:
    - **any** — an image (grayscale or RGB) or a pandas DataFrame
    - **mask** — optional mask to restrict image histograms to the masked region

    Outputs:
    - **table** — histogram bin counts (images) or `describe()` summary (DataFrames)
    - **figure** — distribution plot of the input data

    Keywords: summary, histogram, describe, statistics, dataframe profile, 統計, 直方圖, 摘要, 分析, 描述統計
    """
    __identifier__ = 'nodes.analysis'
    NODE_NAME = 'Data Summary'
    PORT_SPEC = {'inputs': ['in', 'mask'], 'outputs': ['table', 'fig']}
    OUTPUT_COLUMNS = {
        'table': {
            'grayscale_image': ['Pixel_Value', 'Intensity'],
            'rgb_image':       ['Pixel_Value', 'Red', 'Green', 'Blue'],
            'dataframe':       ['index', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
        }
    }

    def __init__(self):
        super(DataSummaryNode, self).__init__()
        self.add_input('in', multi_input=True, color=PORT_COLORS['any'])
        self.add_input('mask', color=PORT_COLORS['mask'])
        self.add_output('table', multi_output=True, color=PORT_COLORS['table'])
        self.add_output('fig', multi_output=True, color=PORT_COLORS['figure'])

    def _img_table(self, data, mask_np=None):
        img_np = np.array(data)

        max_pixel = img_np.max()
        if max_pixel <= 255: max_val = 255
        elif max_pixel <= 4095: max_val = 4095
        else: max_val = 65535

        num_bins = int(max_val) + 1
        hist_dict = {'Pixel_Value': np.arange(num_bins)}

        bool_mask = None
        if mask_np is not None:
            if mask_np.shape[:2] != img_np.shape[:2]:
                m = Image.fromarray(mask_np.astype(np.uint8))
                m = m.resize((img_np.shape[1], img_np.shape[0]), Image.NEAREST)
                mask_np = np.array(m)
            bool_mask = mask_np > 0

        if data.mode == 'RGB' and len(img_np.shape) == 3:
            for ch, name in zip(range(3), ['Red', 'Green', 'Blue']):
                px = img_np[:, :, ch][bool_mask] if bool_mask is not None else img_np[:, :, ch].ravel()
                hist_dict[name], _ = np.histogram(px, bins=num_bins, range=(0, num_bins))
        else:
            px = img_np[bool_mask] if bool_mask is not None else img_np.ravel()
            hist_dict['Intensity'], _ = np.histogram(px, bins=num_bins, range=(0, num_bins))
        return pd.DataFrame(hist_dict)

    def _df_summary_plot(self, df):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return None
        
        cols = numeric_df.columns[:10]
        n_cols = len(cols)
        
        fig, axes = plt.subplots(n_cols, 1, figsize=(8, 3 * n_cols), squeeze=False)
        sns.set_theme(style="whitegrid")
        
        for i, col in enumerate(cols):
            sns.histplot(numeric_df[col], ax=axes[i, 0], kde=True, color='#4c72b0')
            axes[i, 0].set_title(f"Distribution of '{col}'")
            axes[i, 0].set_xlabel("Value")
            axes[i, 0].set_ylabel("Frequency")
            
        fig.tight_layout()
        plt.close(fig)
        return fig
    
    def _img_summary(self, data, mask_np=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        img_np = np.array(data)

        bool_mask = None
        if mask_np is not None:
            if mask_np.shape[:2] != img_np.shape[:2]:
                m = Image.fromarray(mask_np.astype(np.uint8))
                m = m.resize((img_np.shape[1], img_np.shape[0]), Image.NEAREST)
                mask_np = np.array(m)
            bool_mask = mask_np > 0

        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 8))

        max_pixel = img_np.max()
        if max_pixel <= 255:
            max_val = 255
        elif max_pixel <= 4095:
            max_val = 4095
        else:
            max_val = 65535

        suffix = ' (masked)' if bool_mask is not None else ''

        if data.mode == 'RGB' and len(img_np.shape) == 3:
            colors = ['red', 'green', 'blue']
            channel_names = ['Red', 'Green', 'Blue']
            for i, color in enumerate(colors):
                px = img_np[:, :, i][bool_mask] if bool_mask is not None else img_np[:, :, i].ravel()
                ax.hist(px, bins=256, range=(0, max_val), color=color, alpha=0.5, label=channel_names[i] + suffix, log=True, bottom=0.5)
            ax.legend()
        else:
            px = img_np[bool_mask] if bool_mask is not None else img_np.ravel()
            ax.hist(px, bins=256, range=(0, max_val), color='gray', alpha=0.7, label='Intensity' + suffix, log=True, bottom=0.5)

        ax.set_ylim(bottom=0.5)
        ax.set_title(f"Pixel Intensity Distribution ({data.mode}){suffix}")
        ax.set_xlim(0, max_val)
        ax.set_xlabel(f"Pixel Value (0-{max_val})")
        ax.set_ylabel("Count")

        fig.set_tight_layout(True)
        plt.close(fig)
        return fig

    def evaluate(self):
        self.reset_progress()
        in_values = []
        in_port = self.inputs().get('in')
        if in_port and in_port.connected_ports():
            for connected in in_port.connected_ports():
                upstream_node = connected.node()
                up_val = upstream_node.output_values.get(connected.name(), None)
                if isinstance(up_val, TableData):
                    up_val = up_val.df
                elif hasattr(up_val, 'payload'):
                    up_val = up_val.payload
                in_values.append(up_val)

        if not in_values or in_values[0] is None:
            self.mark_error()
            return False, "No input data"

        data = in_values[0]
        if not isinstance(data, (pd.DataFrame, Image.Image)):
            self.mark_error()
            return False, "Input must be a pandas DataFrame or PIL Image"

        # Read optional mask
        mask_np = None
        mask_port = self.inputs().get('mask')
        if mask_port and mask_port.connected_ports():
            connected = mask_port.connected_ports()[0]
            up_val = connected.node().output_values.get(connected.name(), None)
            if up_val is not None and hasattr(up_val, 'payload'):
                raw = up_val.payload
                if isinstance(raw, Image.Image):
                    mask_np = np.array(raw)
                elif isinstance(raw, np.ndarray):
                    mask_np = raw

        try:
            self.set_progress(10)
            self.output_values['table'] = TableData(payload=pd.DataFrame())
            self.output_values['fig'] = FigureData(payload=None)

            if isinstance(data, pd.DataFrame):
                self.set_progress(30)
                summary = data.describe().reset_index().round(4)
                self.output_values['table'] = TableData(payload=summary)

                self.set_progress(60)
                fig = self._df_summary_plot(data)
                self.output_values['fig'] = FigureData(payload=fig)

                self.set_progress(100)
            elif isinstance(data, Image.Image):
                self.set_progress(30)
                table = self._img_table(data, mask_np)
                self.output_values['table'] = TableData(payload=table)

                self.set_progress(60)
                fig = self._img_summary(data, mask_np)
                self.output_values['fig'] = FigureData(payload=fig)

                self.set_progress(100)

            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)


class OutlierDetectionNode(BaseExecutionNode):
    """
    Detects and removes outliers in numerical data using statistical tests.

    Methods:
    - *ROUT (Prism Regression)* — robust nonlinear regression-based detection
    - *ROUT (Fast Math)* — faster variant of the ROUT method
    - *Grubbs* — classical single-outlier test applied iteratively

    **Threshold** — Q value (ROUT) or alpha significance level (Grubbs).

    Outputs two tables: rows kept and rows removed.

    Keywords: outlier, rout, grubbs, anomaly, remove extremes, 異常值, 極端值, 移除, 統計, 篩選
    """
    __identifier__ = 'nodes.analysis'
    NODE_NAME = 'Outlier Detection'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table', 'table']}

    def __init__(self):
        super(OutlierDetectionNode, self).__init__()
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('kept', color=PORT_COLORS['table'])
        self.add_output('removed', color=PORT_COLORS['table'])
        self.add_combo_menu('method', 'Method', items=['ROUT (Prism Regression)', 'ROUT (Fast Math)', 'Grubbs'])
        self.add_text_input('threshold', 'Threshold (Q / Alpha)', text='0.01')

    def _detect_outliers_grubbs(self, data, threshold):
        from scipy import stats
        n = len(data)
        outlier_mask = np.zeros(n, dtype=bool)
        if n < 3:
            return outlier_mask
            
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return outlier_mask
            
        abs_dev = np.abs(data - mean)
        G = np.max(abs_dev) / std
        max_idx = np.argmax(abs_dev)
        
        t_dist = stats.t.ppf(1 - threshold / (2 * n), n - 2)
        G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))
        
        if G > G_crit:
            outlier_mask[max_idx] = True
            
        return outlier_mask

    def _detect_outliers_rout(self, data, Q):
        from scipy import stats
        from scipy.optimize import least_squares
        
        n = len(data)
        outlier_mask = np.zeros(n, dtype=bool)
        if n < 3:
            return outlier_mask
            
        n_float = float(n)
        correction = np.sqrt(n_float / (n_float - 1.0)) if n_float > 1 else 1.0
        
        initial_guess = [np.median(data)]
        
        def objective(params):
            return data - params[0]
            
        mad = np.median(np.abs(data - np.median(data)))
        current_rs = max(mad * 1.4826, 1e-6)
        
        prev_rs = 0
        max_iter = 20
        robust_mean = initial_guess[0]
        
        for _ in range(max_iter):
            if np.abs(current_rs - prev_rs) < 1e-5 * current_rs:
                break
            
            prev_rs = current_rs
            
            res_lsq = least_squares(objective, [robust_mean], loss='cauchy', f_scale=current_rs)
            robust_mean = res_lsq.x[0]
            
            res = np.abs(data - robust_mean)
            current_rs = np.percentile(res, 68.27) * correction
            if current_rs == 0:
                current_rs = np.std(data, ddof=1)
                
            current_rs = max(current_rs, 1e-6)
            
        res = np.abs(data - robust_mean)
        RS = current_rs
        
        if RS == 0:
            return outlier_mask
                
        t_ratios = res / RS
        df = n - 1
        p_values = 2 * (1 - stats.t.cdf(t_ratios, df))
        
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        i_vals = np.arange(1, n + 1)
        crit_p = (i_vals / n) * Q
        
        discoveries = sorted_p <= crit_p
        if np.any(discoveries):
            largest_discovery_idx = np.where(discoveries)[0][-1]
            outlier_indices = sorted_indices[:largest_discovery_idx + 1]
            outlier_mask[outlier_indices] = True
            
        return outlier_mask

    def _detect_outliers_rout_fast(self, data, Q):
        from scipy import stats
        
        n = len(data)
        outlier_mask = np.zeros(n, dtype=bool)
        if n < 3:
            return outlier_mask
            
        median = np.median(data)
        res = np.abs(data - median)
        
        n_float = float(n)
        correction = np.sqrt(n_float / (n_float - 1.0)) if n_float > 1 else 1.0
        
        mad_rs = np.median(res) * 1.4826
        perc_rs = np.percentile(res, 68.27) * correction
        
        RS = max(mad_rs, perc_rs)
        
        if RS == 0:
            RS = np.std(data, ddof=1)
            if RS == 0:
                return outlier_mask
                
        t_ratios = res / RS
        df = n - 1
        p_values = 2 * (1 - stats.t.cdf(t_ratios, df))
        
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        i_vals = np.arange(1, n + 1)
        crit_p = (i_vals / n) * Q
        
        discoveries = sorted_p <= crit_p
        if np.any(discoveries):
            largest_discovery_idx = np.where(discoveries)[0][-1]
            outlier_indices = sorted_indices[:largest_discovery_idx + 1]
            outlier_mask[outlier_indices] = True
            
        return outlier_mask

    def evaluate(self):
        self.reset_progress()
        
        in_values = []
        in_port = self.inputs().get('in')
        if in_port and in_port.connected_ports():
            for connected in in_port.connected_ports():
                upstream_node = connected.node()
                up_val = upstream_node.output_values.get(connected.name(), None)
                if isinstance(up_val, TableData):
                    up_val = up_val.df
                elif hasattr(up_val, 'payload'):
                    up_val = up_val.payload
                in_values.append(up_val)
        
        if not in_values or in_values[0] is None:
            self.mark_error()
            return False, "No input data"
        
        data = in_values[0]
        method = self.get_property('method') or 'ROUT (Prism Regression)'
        if method == 'ROUT': # Handle legacy workflows
            method = 'ROUT (Prism Regression)'

        try:
            threshold = float(self.get_property('threshold') or 0.01)
            if threshold >= 0.5:
                threshold = threshold / 100.0
        except ValueError:
            threshold = 0.01
        
        try:
            if isinstance(data, pd.DataFrame):
                df = data
                num_cols = df.select_dtypes(include=[np.number]).columns
                if len(num_cols) == 0:
                    self.output_values['kept'] = TableData(payload=df)
                    self.output_values['removed'] = TableData(payload=pd.DataFrame(columns=df.columns))
                    self.mark_clean()
                    return True, None

                # Detect long format: one numeric value column + a group column
                group_col = None
                for col in df.columns:
                    if str(col).lower() in ['group', 'class', 'treatment']:
                        group_col = col
                        break

                if group_col and len(num_cols) == 1:
                    # Long format — run outlier detection per group so each group
                    # is judged against its own distribution (not the global pool)
                    val_col = num_cols[0]
                    kept_mask = pd.Series(True, index=df.index)
                    groups = df[group_col].dropna().unique()
                    for gi, grp in enumerate(groups):
                        grp_idx = df.index[df[group_col] == grp]
                        valid = df.loc[grp_idx, val_col].dropna()
                        if len(valid) < 3:
                            continue
                        arr = valid.values.astype(float)
                        if method == 'ROUT (Prism Regression)':
                            col_mask = self._detect_outliers_rout(arr, threshold)
                        elif method == 'ROUT (Fast Math)':
                            col_mask = self._detect_outliers_rout_fast(arr, threshold)
                        else:
                            col_mask = self._detect_outliers_grubbs(arr, threshold)
                        kept_mask.loc[valid.index[col_mask]] = False
                        self.set_progress(int(90 * (gi + 1) / len(groups)))

                    self.output_values['kept'] = TableData(payload=df[kept_mask].reset_index(drop=True))
                    self.output_values['removed'] = TableData(payload=df[~kept_mask].reset_index(drop=True))
                else:
                    # Wide format — run outlier detection per column (each column = one group)
                    df_kept = pd.DataFrame(index=df.index, columns=df.columns)
                    df_removed = pd.DataFrame(index=df.index, columns=df.columns)

                    non_num_cols = df.columns.difference(num_cols)
                    for col in non_num_cols:
                        df_kept[col] = df[col]
                        df_removed[col] = df[col]

                    for i, col in enumerate(num_cols):
                        col_data = df[col]
                        valid_data = col_data.dropna()

                        if len(valid_data) < 3:
                            df_kept[col] = col_data
                            continue

                        arr = valid_data.values
                        if method == 'ROUT (Prism Regression)':
                            col_mask = self._detect_outliers_rout(arr, threshold)
                        elif method == 'ROUT (Fast Math)':
                            col_mask = self._detect_outliers_rout_fast(arr, threshold)
                        else:
                            col_mask = self._detect_outliers_grubbs(arr, threshold)

                        full_mask = pd.Series(False, index=df.index)
                        full_mask.loc[valid_data.index] = col_mask

                        df_kept[col] = df[col].where(~full_mask, np.nan)
                        df_removed[col] = df[col].where(full_mask, np.nan)

                        self.set_progress(int(90 * (i + 1) / len(num_cols)))

                    df_kept = df_kept.dropna(subset=num_cols, how='all')
                    df_removed = df_removed.dropna(subset=num_cols, how='all')

                    self.output_values['kept'] = TableData(payload=df_kept)
                    self.output_values['removed'] = TableData(payload=df_removed)
                
            else:
                arr = np.array(data)
                if arr.dtype.kind not in 'iuf':
                    return False, "Input data is not numerical"
                
                if method == 'ROUT (Prism Regression)':
                    mask = self._detect_outliers_rout(arr, threshold)
                elif method == 'ROUT (Fast Math)':
                    mask = self._detect_outliers_rout_fast(arr, threshold)
                else:
                    mask = self._detect_outliers_grubbs(arr, threshold)
                    
                kept = arr[~mask].tolist()
                removed = arr[mask].tolist()
                
                self.output_values['kept'] = kept
                self.output_values['removed'] = removed
            
            self.set_progress(100)
            self.mark_clean()
            return True, None
            
        except Exception as e:
            self.mark_error()
            return False, str(e)


class GroupedComparisonNode(BaseExecutionNode):
    """
    Tests whether there are significant differences among two or more groups.

    Tests:
    - *One-Way ANOVA* — parametric, assumes normal distribution and equal variances
    - *Kruskal-Wallis* — non-parametric rank-based alternative to ANOVA

    Outputs a summary table with test statistic, p-value, and significance flag.

    Keywords: anova, kruskal, group comparison, omnibus test, significance, 統計, 比較, 分組, 顯著性, 變異數分析
    """
    __identifier__ = 'nodes.analysis'
    NODE_NAME = 'Grouped Comparison'
    PORT_SPEC = {'inputs': ['in'], 'outputs': ['stats_table']}
    
    def __init__(self):
        super(GroupedComparisonNode, self).__init__()
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('stats_table', color=PORT_COLORS['table'])
        
        methods = ['One-Way ANOVA', 'Kruskal-Wallis']
        self.add_combo_menu('method', 'Statistical Method', items=methods)
        self.add_text_input('target_column', 'Target Column', text='')
        self.add_text_input('group_column', 'Group Column', text='Group')
        self.add_text_input('reference_group', 'Reference Group (for detection)', text='DMSO')
        self._fix_widget_z_order()
        
    def evaluate(self):
        self.reset_progress()
        from scipy.stats import f_oneway, kruskal
        
        if hasattr(self, '_test_data'):
            up_val = self._test_data
        else:
            in_port = self.inputs().get('in')
            if not in_port or not in_port.connected_ports():
                self.mark_error()
                return False, "No input data"
            upstream_node = in_port.connected_ports()[0].node()
            up_val = upstream_node.output_values.get(in_port.connected_ports()[0].name(), None)
            
        if isinstance(up_val, TableData):
            df = up_val.df.copy()
        elif isinstance(up_val, pd.DataFrame):
            df = up_val.copy()
        else:
            self.mark_error()
            return False, "Expected TableData or DataFrame input"
            
        method = self.get_property('method')
        target_col = str(self.get_property('target_column')).strip()
        group_col = str(self.get_property('group_column')).strip()
        
        if not group_col:
            for col in df.columns:
                if str(col).lower() in ['group', 'class', 'treatment']:
                    group_col = col
                    break
        
        is_wide_format = False
        if not group_col or group_col not in df.columns:
            ref_group = str(self.get_property('reference_group')).strip()
            if ref_group in df.columns:
                is_wide_format = True
                melt_vars = df.select_dtypes(include=[np.number]).columns.tolist()
                df = df.melt(value_vars=melt_vars, var_name='_auto_group', value_name='_auto_target')
                group_col = '_auto_group'
                target_col = '_auto_target'
        
        if not is_wide_format and not target_col:
            num_cols = df.select_dtypes(include=[np.number]).columns
            val_cols = [c for c in num_cols if c != group_col]
            if val_cols:
                target_col = val_cols[0]
            else:
                self.mark_error()
                return False, "No numerical target column found"
                
        if group_col not in df.columns or target_col not in df.columns:
            self.mark_error()
            return False, f"Columns '{group_col}' or '{target_col}' not found"

        try:
            df_clean = df.dropna(subset=[target_col, group_col])
            groups = df_clean[group_col].unique()
            
            if len(groups) < 2:
                self.mark_error()
                return False, "At least two groups are required"
                
            data_groups = [df_clean[df_clean[group_col] == g][target_col].values for g in groups]
            
            if 'ANOVA' in method:
                stat, p_val = f_oneway(*data_groups)
                test_name = 'One-Way ANOVA'
                stat_name = 'F-Statistic'
            else:
                stat, p_val = kruskal(*data_groups)
                test_name = 'Kruskal-Wallis'
                stat_name = 'H-Statistic'
            
            res_df = pd.DataFrame([{
                'Test': test_name,
                'Target': target_col,
                stat_name: stat,
                'p-value': p_val,
                'Significant': p_val < 0.05
            }])
            
            self.output_values['stats_table'] = TableData(payload=res_df)
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)


class _RotatedHeaderView(QtWidgets.QHeaderView):
    """Horizontal header that draws group names rotated -90° (reads bottom-to-top)."""

    section_clicked_signal = QtCore.Signal(int)
    _HEADER_H = 80

    def __init__(self, parent=None):
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self.setMinimumSectionSize(20)
        self.setDefaultSectionSize(24)
        self.setFixedHeight(self._HEADER_H)
        self.setSectionsClickable(True)
        self.sectionClicked.connect(self.section_clicked_signal)

    def paintSection(self, painter, rect, logicalIndex):
        painter.save()
        painter.setClipRect(rect)
        painter.fillRect(rect, QtGui.QColor('#2d2d2d'))
        painter.setPen(QtGui.QColor('#3a3a3a'))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

        model = self.model()
        text = ''
        if model:
            text = str(model.headerData(
                logicalIndex, QtCore.Qt.Orientation.Horizontal,
                QtCore.Qt.ItemDataRole.DisplayRole) or '')

        painter.setPen(QtGui.QColor('#d4d4d4'))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        fm = QtGui.QFontMetrics(font)

        # Translate to bottom-left corner of cell, rotate -90° so text reads upward
        painter.translate(rect.x() + 3, rect.y() + rect.height() - 4)
        painter.rotate(-90)
        avail = rect.height() - 8
        elided = fm.elidedText(text, QtCore.Qt.TextElideMode.ElideRight, avail)
        painter.drawText(0, 0, avail, rect.width(),
                         QtCore.Qt.AlignmentFlag.AlignLeft |
                         QtCore.Qt.AlignmentFlag.AlignVCenter, elided)
        painter.restore()


class PairwiseMatrixWidget(NodeBaseWidget):
    """
    Interactive NxN matrix grid for selecting which group pairs to compare.

    Interaction:
    - Upper-triangle cells are clickable toggles; diagonal is disabled
    - Column headers are rotated 90 degrees
    - Click a row or column header to toggle all pairs involving that group
    """
    pairs_changed = QtCore.Signal(str)        # emits "A|B, C|D" string
    update_groups_signal = QtCore.Signal(list) # receives list[str] of group names

    _SEL_COLOR   = QtGui.QColor(70, 160, 255, 180)
    _UNSEL_COLOR = QtGui.QColor(40, 40, 40)
    _DIAG_COLOR  = QtGui.QColor(30, 30, 30)
    _REF_COLOR   = QtGui.QColor(255, 180, 60, 180)

    def __init__(self, parent=None, name='', label=''):
        super().__init__(parent, name, label)
        self._groups: list[str] = []
        self._selected: set[tuple[str, str]] = set()
        self._ref_group: str = ''
        self._is_updating = False

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._table = QtWidgets.QTableWidget()
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

        # Rotated column headers
        self._col_header = _RotatedHeaderView(self._table)
        self._table.setHorizontalHeader(self._col_header)
        self._col_header.section_clicked_signal.connect(self._on_header_clicked)

        # Row headers — click to toggle all pairs for that group
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.verticalHeader().setMinimumSectionSize(20)
        self._table.verticalHeader().setSectionsClickable(True)
        self._table.verticalHeader().sectionClicked.connect(self._on_header_clicked)

        self._table.setFixedHeight(140)
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e; color: #d4d4d4;
                gridline-color: #3a3a3a; border: 1px solid #3a3a3a;
                font-size: 10px;
            }
            QHeaderView::section {
                background-color: #2d2d2d; color: #d4d4d4;
                border: 1px solid #3a3a3a; padding: 2px; font-size: 10px;
            }
        """)
        self._table.cellClicked.connect(self._on_cell_clicked)
        layout.addWidget(self._table)

        # quick-action buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(4)
        btn_all = QtWidgets.QPushButton('All')
        btn_none = QtWidgets.QPushButton('None')
        btn_all.setFixedHeight(20)
        btn_none.setFixedHeight(20)
        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._select_none)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.set_custom_widget(container)
        self.update_groups_signal.connect(self._rebuild_grid)

    # ── public helpers ────────────────────────────────────────────────
    def set_reference(self, ref: str):
        self._ref_group = ref

    def selected_pairs_str(self) -> str:
        return ', '.join(f'{a}|{b}' for a, b in sorted(self._selected))

    def set_pairs_from_str(self, s: str):
        """Parse 'A|B, C|D' into _selected set."""
        self._selected.clear()
        if not s:
            return
        for tok in s.split(','):
            parts = [p.strip() for p in tok.split('|')]
            if len(parts) == 2 and all(parts):
                self._selected.add(tuple(sorted(parts)))
        self._repaint_cells()

    # ── grid management ───────────────────────────────────────────────
    def _rebuild_grid(self, groups: list[str]):
        self._is_updating = True
        self._groups = list(groups)
        n = len(groups)
        self._table.setRowCount(n)
        self._table.setColumnCount(n)

        self._table.setVerticalHeaderLabels(groups)
        self._table.setHorizontalHeaderLabels(groups)

        for r in range(n):
            for c in range(n):
                item = QtWidgets.QTableWidgetItem()
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                if r == c:
                    item.setBackground(self._DIAG_COLOR)
                    item.setText('—')
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                elif r < c:
                    pair = tuple(sorted((groups[r], groups[c])))
                    if pair in self._selected:
                        item.setBackground(self._SEL_COLOR)
                        item.setText('✓')
                    else:
                        item.setBackground(self._UNSEL_COLOR)
                        item.setText('')
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    item.setToolTip(f'{groups[r]}  vs  {groups[c]}')
                else:
                    pair = tuple(sorted((groups[r], groups[c])))
                    if pair in self._selected:
                        item.setBackground(
                            QtGui.QColor(self._SEL_COLOR.red(),
                                         self._SEL_COLOR.green(),
                                         self._SEL_COLOR.blue(), 60))
                    else:
                        item.setBackground(self._DIAG_COLOR)
                    item.setText('')
                    item.setToolTip(f'{groups[c]}  vs  {groups[r]}')
                self._table.setItem(r, c, item)

        # Column width: fixed narrow cells (just enough for ✓)
        cell_w = 26
        for c in range(n):
            self._table.setColumnWidth(c, cell_w)

        # Header height: based on longest group name (rotated text)
        fm = QtGui.QFontMetrics(QtGui.QFont('sans-serif', 8))
        max_text_w = max((fm.horizontalAdvance(g) for g in groups), default=40)
        header_h = min(max(max_text_w + 12, 40), 120)
        self._col_header.setFixedHeight(header_h)

        # Compact table size
        row_header_w = self._table.verticalHeader().sizeHint().width()
        self._table.setFixedWidth(row_header_w + n * cell_w + 4)
        table_h = min(n * 24 + header_h + 4, 300)
        self._table.setFixedHeight(table_h)
        self._is_updating = False

        # Defer resize — Qt needs to process the layout changes first
        QtCore.QTimer.singleShot(0, self._deferred_resize)

    def _deferred_resize(self):
        if self.node and hasattr(self.node, 'view'):
            self.widget().adjustSize()
            self.node.view.draw_node()

    def _repaint_cells(self):
        n = len(self._groups)
        for r in range(n):
            for c in range(n):
                if r == c:
                    continue
                item = self._table.item(r, c)
                if not item:
                    continue
                pair = tuple(sorted((self._groups[r], self._groups[c])))
                sel = pair in self._selected
                if r < c:
                    item.setBackground(self._SEL_COLOR if sel else self._UNSEL_COLOR)
                    item.setText('✓' if sel else '')
                else:
                    item.setBackground(
                        QtGui.QColor(self._SEL_COLOR.red(),
                                     self._SEL_COLOR.green(),
                                     self._SEL_COLOR.blue(), 60)
                        if sel else self._DIAG_COLOR)

    # ── interaction ───────────────────────────────────────────────────
    def _on_header_clicked(self, index):
        """Toggle all pairs involving the group at `index`."""
        if self._is_updating or index >= len(self._groups):
            return
        group = self._groups[index]
        # Collect all pairs involving this group
        all_pairs = set()
        for i, g in enumerate(self._groups):
            if i != index:
                all_pairs.add(tuple(sorted((group, g))))
        # If all are selected, deselect all; otherwise select all
        if all_pairs <= self._selected:
            self._selected -= all_pairs
        else:
            self._selected |= all_pairs
        self._repaint_cells()
        self.pairs_changed.emit(self.selected_pairs_str())

    def _on_cell_clicked(self, row, col):
        if self._is_updating or row == col or row >= len(self._groups) or col >= len(self._groups):
            return
        # clicking lower triangle still toggles the pair
        pair = tuple(sorted((self._groups[row], self._groups[col])))
        if pair in self._selected:
            self._selected.discard(pair)
        else:
            self._selected.add(pair)
        self._repaint_cells()
        self.pairs_changed.emit(self.selected_pairs_str())

    def _select_all(self):
        self._selected.clear()
        for i in range(len(self._groups)):
            for j in range(i + 1, len(self._groups)):
                self._selected.add(tuple(sorted((self._groups[i], self._groups[j]))))
        self._repaint_cells()
        self.pairs_changed.emit(self.selected_pairs_str())

    def _select_none(self):
        self._selected.clear()
        self._repaint_cells()
        self.pairs_changed.emit(self.selected_pairs_str())

    # ── NodeBaseWidget interface ──────────────────────────────────────
    def get_value(self):
        return self.selected_pairs_str()

    def set_value(self, value):
        if isinstance(value, str):
            self.set_pairs_from_str(value)


class PairwiseComparisonNode(BaseExecutionNode):
    """
    Performs pairwise comparisons between groups using parametric or non-parametric tests.

    Tests:
    - *Student's T-test* — parametric, assumes equal variance and normal distribution
    - *Welch's T-test* — parametric, does not assume equal variance
    - *Mann-Whitney U* — non-parametric rank-based test
    - *Kolmogorov-Smirnov* — tests whether two groups come from the same distribution
    - *Tukey HSD* — post-hoc test after ANOVA
    - *Dunn* — non-parametric post-hoc test (requires scikit-posthocs)
    - *Fisher's Z* — compare correlation coefficients between groups (target column = r values)

    **Alternative** — two-sided (default), greater (group1 > group2), or less (group1 < group2). Tukey HSD and Dunn are always two-sided.

    **P-Adj Method** — multiple comparison correction (Bonferroni, Holm, BH).

    Keywords: pairwise, t-test, welch, mann-whitney, tukey, fisher, kolmogorov, ks, correlation, one-sided, 兩兩比較, 統計檢定, 分析, 顯著性, 比較
    """
    __identifier__ = 'nodes.analysis'
    NODE_NAME = 'Pairwise Comparison'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}

    def __init__(self):
        super(PairwiseComparisonNode, self).__init__()
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('stats_table', color=PORT_COLORS['stat'])

        methods = ["Student's T-test", "Welch's T-test", 'Mann-Whitney U', 'Kolmogorov-Smirnov', 'Tukey HSD', 'Dunn', "Fisher's Z (corr.)"]
        self.add_combo_menu('method', 'Statistical Method', items=methods)

        self.add_combo_menu('alternative', 'Alternative',
                            items=['two-sided', 'greater', 'less'])

        p_adj_methods = ['none', 'bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky']
        self.add_combo_menu('p_adj_method', 'P-Adj Method', items=p_adj_methods)

        self.add_text_input('target_column', 'Target Column', text='')
        self.add_text_input('group_column', 'Group Column', text='')

        # Hidden property to persist selected pairs across save/load
        self.create_property('selected_pairs', '',
                             widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value)

        self._matrix_widget = PairwiseMatrixWidget(
            self.view, name='pair_matrix', label='Select Pairs')
        self._matrix_widget.pairs_changed.connect(self._on_pairs_changed)
        self.add_custom_widget(self._matrix_widget, tab='Parameters')
        self._fix_widget_z_order()

    def _on_pairs_changed(self, pairs_str: str):
        self.set_property('selected_pairs', pairs_str)

    def _detect_groups_and_populate(self, df):
        """Detect unique groups from input data and populate the matrix widget."""
        group_col = str(self.get_property('group_column')).strip()
        if not group_col:
            for col in df.columns:
                if str(col).lower() in ['group', 'class', 'treatment']:
                    group_col = col
                    break
        if group_col and group_col in df.columns:
            groups = sorted(df[group_col].dropna().astype(str).str.strip().unique().tolist())
            if groups:
                # Restore saved selection before rebuilding grid
                saved = str(self.get_property('selected_pairs')).strip()
                if saved:
                    self._matrix_widget.set_pairs_from_str(saved)
                self._matrix_widget.update_groups_signal.emit(groups)
        
    def evaluate(self):
        self.reset_progress()
        from scipy.stats import ttest_ind, mannwhitneyu, tukey_hsd
        from statsmodels.stats.multitest import multipletests

        if hasattr(self, '_test_data'):
            up_val = self._test_data
        else:
            in_port = self.inputs().get('in')
            if not in_port or not in_port.connected_ports():
                self.mark_error()
                return False, "No input data"
            upstream_node = in_port.connected_ports()[0].node()
            up_val = upstream_node.output_values.get(in_port.connected_ports()[0].name(), None)

        if isinstance(up_val, TableData):
            df = up_val.df.copy()
        elif isinstance(up_val, pd.DataFrame):
            df = up_val.copy()
        else:
            self.mark_error()
            return False, "Expected TableData or DataFrame input"

        method = self.get_property('method')
        alternative = self.get_property('alternative') or 'two-sided'
        p_adj_method = self.get_property('p_adj_method')
        target_col = str(self.get_property('target_column')).strip()
        group_col = str(self.get_property('group_column')).strip()

        if not group_col:
            for col in df.columns:
                if str(col).lower() in ['group', 'class', 'treatment']:
                    group_col = col
                    break

        # Wide-format detection: if group_col not found, try melt
        selected_str = str(self.get_property('selected_pairs')).strip()
        is_wide_format = False
        if not group_col or group_col not in df.columns:
            potential_groups = []
            if selected_str:
                for tok in selected_str.split(','):
                    potential_groups.extend([p.strip() for p in tok.split('|') if p.strip()])
            found_cols = [c for c in potential_groups if c in df.columns]
            if found_cols:
                is_wide_format = True
                melt_vars = df.select_dtypes(include=[np.number]).columns.tolist()
                df = df.melt(value_vars=melt_vars, var_name='_auto_group', value_name='_auto_target')
                group_col = '_auto_group'
                target_col = '_auto_target'

        if not is_wide_format and not target_col:
            num_cols = df.select_dtypes(include=[np.number]).columns
            val_cols = [c for c in num_cols if c != group_col]
            if val_cols:
                target_col = val_cols[0]
            else:
                self.mark_error()
                return False, "No numerical target column found"

        if group_col not in df.columns or target_col not in df.columns:
            self.mark_error()
            return False, f"Columns '{group_col}' or '{target_col}' not found"

        try:
            df_clean = df.dropna(subset=[target_col, group_col])
            df_clean[group_col] = df_clean[group_col].astype(str).str.strip()
            unique_groups = df_clean[group_col].unique()

            if len(unique_groups) < 2:
                self.mark_error()
                return False, "At least two groups are required"

            # Populate the matrix widget with discovered groups
            self._detect_groups_and_populate(df_clean if not is_wide_format else df)

            # Build pair list from matrix selection
            pairs_set = set()
            if selected_str:
                for tok in selected_str.split(','):
                    parts = [p.strip() for p in tok.split('|')]
                    if len(parts) == 2 and all(parts):
                        if parts[0] in unique_groups and parts[1] in unique_groups:
                            pairs_set.add(tuple(sorted(parts)))

            if not pairs_set:
                res_df = pd.DataFrame(columns=['group1', 'group2', 'p-value', 'p-adj', 'Significant'])
                self.output_values['stats_table'] = StatData(payload=res_df)
                self.set_progress(100)
                self.mark_clean()
                return True, None

            pairs = list(pairs_set)
            results = []

            if 'Tukey' in method:
                group_list = list(unique_groups)
                data_args = [df_clean[df_clean[group_col] == g][target_col].values for g in group_list]
                res = tukey_hsd(*data_args)

                idx_map = {g: i for i, g in enumerate(group_list)}
                for g1, g2 in pairs:
                    i, j = idx_map.get(g1), idx_map.get(g2)
                    if i is not None and j is not None:
                        results.append({
                            'group1': g1, 'group2': g2,
                            'p-value': res.pvalue[i, j],
                            'p-adj': res.pvalue[i, j],
                            'Significant': res.pvalue[i, j] < 0.05
                        })

            elif 'T-test' in method or 'Mann-Whitney' in method or 'Kolmogorov' in method:
                from scipy.stats import ks_2samp
                raw_pvals = []
                for g1, g2 in pairs:
                    d1 = df_clean[df_clean[group_col] == g1][target_col].values
                    d2 = df_clean[df_clean[group_col] == g2][target_col].values
                    if 'T-test' in method:
                        equal_var = 'Welch' not in method
                        stat, p = ttest_ind(d1, d2, nan_policy='omit',
                                            equal_var=equal_var,
                                            alternative=alternative)
                    elif 'Kolmogorov' in method:
                        stat, p = ks_2samp(d1, d2, alternative=alternative)
                    else:
                        stat, p = mannwhitneyu(d1, d2, alternative=alternative)
                    raw_pvals.append(p)
                    results.append({'group1': g1, 'group2': g2, 'statistic': round(stat, 4), 'p-value': p})

                if p_adj_method != 'none' and results:
                    reject, pvals_corrected, _, _ = multipletests(raw_pvals, method=p_adj_method)
                    for i, r in enumerate(results):
                        r['p-adj'] = pvals_corrected[i]
                        r['Significant'] = reject[i]
                else:
                    for r in results:
                        r['p-adj'] = r['p-value']
                        r['Significant'] = r['p-value'] < 0.05

            elif 'Fisher' in method:
                from scipy.stats import norm
                raw_pvals = []
                for g1, g2 in pairs:
                    d1 = df_clean[df_clean[group_col] == g1][target_col].dropna().values
                    d2 = df_clean[df_clean[group_col] == g2][target_col].dropna().values
                    n1, n2 = len(d1), len(d2)
                    if n1 < 4 or n2 < 4:
                        raw_pvals.append(float('nan'))
                        results.append({'group1': g1, 'group2': g2,
                                        'r1': float('nan'), 'r2': float('nan'),
                                        'n1': n1, 'n2': n2,
                                        'z_diff': float('nan'), 'p-value': float('nan')})
                        continue
                    r1 = np.mean(d1)
                    r2 = np.mean(d2)
                    # Clamp r values to avoid arctanh(±1) = inf
                    r1_c = np.clip(r1, -0.9999, 0.9999)
                    r2_c = np.clip(r2, -0.9999, 0.9999)
                    z1 = np.arctanh(r1_c)  # Fisher z-transform
                    z2 = np.arctanh(r2_c)
                    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
                    z_diff = (z1 - z2) / se
                    if alternative == 'greater':
                        p = norm.sf(z_diff)
                    elif alternative == 'less':
                        p = norm.cdf(z_diff)
                    else:
                        p = 2.0 * norm.sf(abs(z_diff))
                    raw_pvals.append(p)
                    results.append({'group1': g1, 'group2': g2,
                                    'r1': r1, 'r2': r2,
                                    'n1': n1, 'n2': n2,
                                    'z_diff': round(z_diff, 4), 'p-value': p})

                if p_adj_method != 'none' and results:
                    valid_mask = [not np.isnan(p) for p in raw_pvals]
                    valid_pvals = [p for p, v in zip(raw_pvals, valid_mask) if v]
                    if valid_pvals:
                        reject, pvals_corrected, _, _ = multipletests(valid_pvals, method=p_adj_method)
                        j = 0
                        for i, r in enumerate(results):
                            if valid_mask[i]:
                                r['p-adj'] = pvals_corrected[j]
                                r['Significant'] = reject[j]
                                j += 1
                            else:
                                r['p-adj'] = float('nan')
                                r['Significant'] = False
                    else:
                        for r in results:
                            r['p-adj'] = float('nan')
                            r['Significant'] = False
                else:
                    for r in results:
                        r['p-adj'] = r['p-value']
                        r['Significant'] = r['p-value'] < 0.05 if not np.isnan(r['p-value']) else False

            elif 'Dunn' in method:
                try:
                    import scikit_posthocs as sp
                except ImportError:
                    self.mark_error()
                    return False, "Dunn's test requires 'scikit-posthocs'. Install with: pip install scikit-posthocs"

                dunn_p = sp.posthoc_dunn(
                    df_clean, val_col=target_col, group_col=group_col, p_adjust=None)

                raw_pvals = []
                for g1, g2 in pairs:
                    p = dunn_p.loc[g1, g2] if g1 in dunn_p.index and g2 in dunn_p.columns else float('nan')
                    raw_pvals.append(p)
                    results.append({'group1': g1, 'group2': g2, 'p-value': p})

                if p_adj_method != 'none' and results:
                    reject, pvals_corrected, _, _ = multipletests(raw_pvals, method=p_adj_method)
                    for i, r in enumerate(results):
                        r['p-adj'] = pvals_corrected[i]
                        r['Significant'] = reject[i]
                else:
                    for r in results:
                        r['p-adj'] = r['p-value']
                        r['Significant'] = r['p-value'] < 0.05

            res_df = pd.DataFrame(results)
            self.output_values['stats_table'] = StatData(payload=res_df)
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)


class NormalityTestNode(BaseExecutionNode):
    """
    Tests whether each numerical column in a DataFrame follows a normal distribution.

    Tests:
    - *Shapiro-Wilk* — recommended for small to moderate samples
    - *Kolmogorov-Smirnov* — compares against a theoretical normal CDF
    - *Anderson-Darling* — weighted variant sensitive to distribution tails

    Outputs a summary table with test statistic, p-value (where applicable), and pass/fail result.

    Keywords: normality, shapiro, kolmogorov-smirnov, anderson-darling, gaussian check, 常態分佈, 統計檢定, 分析, 高斯, 正態性
    """
    __identifier__ = 'nodes.analysis'
    NODE_NAME = 'Normality Test'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}

    def __init__(self):
        super(NormalityTestNode, self).__init__()
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('results', color=PORT_COLORS['table'])

        tests = ['All (Shapiro-Wilk + KS + Anderson-Darling)',
                 'Shapiro-Wilk',
                 'Kolmogorov-Smirnov (KS)',
                 'Anderson-Darling']
        self.add_combo_menu('test', 'Test(s)', items=tests)
        self.add_text_input('alpha', 'Significance Level (α)', text='0.05')
        self.add_text_input('group_column', 'Group Column (optional)', text='')

    def evaluate(self):
        self.reset_progress()
        from scipy import stats

        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No input data"

        upstream_node = in_port.connected_ports()[0].node()
        up_val = upstream_node.output_values.get(in_port.connected_ports()[0].name(), None)

        if isinstance(up_val, TableData):
            df = up_val.df.copy()
        elif isinstance(up_val, pd.DataFrame):
            df = up_val.copy()
        else:
            self.mark_error()
            return False, "Expected TableData or DataFrame input"

        try:
            alpha = float(self.get_property('alpha') or 0.05)
        except ValueError:
            alpha = 0.05

        chosen_test = self.get_property('test') or 'All'
        group_col_name = str(self.get_property('group_column')).strip()

        # Detect group column
        group_col = None
        if group_col_name and group_col_name in df.columns:
            group_col = group_col_name
        else:
            for col in df.columns:
                if str(col).lower() in ['group', 'class', 'treatment']:
                    group_col = col
                    break

        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != group_col]

        if not num_cols:
            self.mark_error()
            return False, "No numerical columns found"

        run_sw  = 'Shapiro' in chosen_test or 'All' in chosen_test
        run_ks  = 'KS' in chosen_test or 'Kolmogorov' in chosen_test or 'All' in chosen_test
        run_ad  = 'Anderson' in chosen_test or 'All' in chosen_test

        rows = []
        groups = df[group_col].unique() if group_col else [None]
        n_groups = len(groups)
        n_cols   = len(num_cols)

        for gi, grp in enumerate(groups):
            sub = df[df[group_col] == grp] if group_col else df

            for ci, col in enumerate(num_cols):
                data = sub[col].dropna().values
                label = col if group_col is None else f"{grp} — {col}"
                n = len(data)

                if n < 3:
                    rows.append({'Column': label, 'Test': 'N/A', 'n': n,
                                 'Statistic': np.nan, 'p-value': np.nan,
                                 'Normal': None, 'Note': f'n={n} (too small)'})
                    continue

                # ── Shapiro-Wilk ────────────────────────────────────────────
                if run_sw:
                    if n > 5000:
                        note = 'n>5000 (low power)'
                    else:
                        note = ''
                    sw_stat, sw_p = stats.shapiro(data)
                    rows.append({
                        'Column': label, 'Test': 'Shapiro-Wilk', 'n': n,
                        'Statistic': round(sw_stat, 6),
                        'p-value':   round(sw_p,   6),
                        'Normal': sw_p > alpha,
                        'Note': note,
                    })

                # ── Kolmogorov-Smirnov ──────────────────────────────────────
                if run_ks:
                    # Fit a normal distribution to the data first, then compare
                    mu, sigma = np.mean(data), np.std(data, ddof=1)
                    if sigma == 0:
                        rows.append({'Column': label, 'Test': 'KS', 'n': n,
                                     'Statistic': np.nan, 'p-value': np.nan,
                                     'Normal': None, 'Note': 'σ=0 (constant data)'})
                    else:
                        ks_stat, ks_p = stats.kstest(data, 'norm', args=(mu, sigma))
                        rows.append({
                            'Column': label, 'Test': 'Kolmogorov-Smirnov', 'n': n,
                            'Statistic': round(ks_stat, 6),
                            'p-value':   round(ks_p,   6),
                            'Normal': ks_p > alpha,
                            'Note': '',
                        })

                # ── Anderson-Darling ────────────────────────────────────────
                if run_ad:
                    ad_result = stats.anderson(data, dist='norm', method='interpolate')
                    # method='interpolate' returns a SignificanceResult with .pvalue directly,
                    # capped at [0.15, 0.01] by SciPy's interpolation tables.
                    ad_p = float(ad_result.pvalue)
                    rows.append({
                        'Column':    label,
                        'Test':      'Anderson-Darling',
                        'n':         n,
                        'Statistic': round(float(ad_result.statistic), 6),
                        'p-value':   round(ad_p, 6),
                        'Normal':    ad_p > alpha,
                        'Note':      'p capped at [0.01, 0.15] by interpolation',
                    })

                self.set_progress(int(90 * (gi * n_cols + ci + 1) / (n_groups * n_cols)))

        result_df = pd.DataFrame(rows, columns=['Column', 'Test', 'n',
                                                'Statistic', 'p-value', 'Normal', 'Note'])
        self.output_values['results'] = TableData(payload=result_df)
        self.set_progress(100)
        self.mark_clean()
        return True, None


# ===========================================================================
# PairwiseMatrixNode
# ===========================================================================

class PairwiseMatrixNode(BaseExecutionNode):
    """
    Computes a pairwise correlation or distance matrix for all numeric columns and visualises it as a heatmap.

    Correlation methods:
    - *Pearson* — linear correlation coefficient, assumes normality
    - *Spearman* — rank-based, robust to outliers and non-normal distributions
    - *Kendall* — rank-based, slower but more exact for small sample sizes

    Outputs a matrix table (for further analysis) and an annotated heatmap figure.

    Keywords: pairwise, matrix, correlation, pearson, spearman, kendall, distance, euclidean,
              cosine, manhattan, heatmap, 相關性, 相關係數, 距離, 熱圖, 統計, 分析
    """
    __identifier__ = 'nodes.analysis'
    NODE_NAME      = 'Pairwise Matrix'
    PORT_SPEC      = {'inputs': ['table'], 'outputs': ['table', 'figure']}

    def __init__(self):
        super().__init__()
        self.add_input('in',     color=PORT_COLORS['table'])
        self.add_output('table', color=PORT_COLORS['table'])
        self.add_output('figure', color=PORT_COLORS['figure'])
        self.add_combo_menu('method', 'Metric',
                            items=['pearson', 'spearman', 'kendall',
                                   'euclidean (distance)',
                                   'cosine (distance)',
                                   'cityblock (distance)',
                                   'chebyshev (distance)',
                                   'correlation (distance)'])
        self.add_combo_menu('colormap', 'Colormap',
                            items=['coolwarm', 'RdBu_r', 'vlag', 'seismic', 'bwr',
                                   'viridis', 'magma', 'inferno', 'plasma', 'cividis',
                                   'Blues', 'Reds', 'Greens', 'Purples', 'Oranges',
                                   'YlOrRd', 'YlGnBu', 'PuBuGn', 'PuOr', 'Spectral',
                                   'PRGn', 'PiYG'])
        self.add_checkbox('annot', '', text='Show values', state=True)
        self.add_checkbox('mask_upper', '', text='Hide upper triangle', state=False)

    def evaluate(self):
        self.reset_progress()
        import seaborn as sns
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        port = self.inputs().get('in')
        if not port or not port.connected_ports():
            self.mark_error()
            return False, "No input connected"
        cp   = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if not isinstance(data, TableData):
            self.mark_error()
            return False, "Input must be TableData"

        df_num = data.df.select_dtypes(include='number')
        if df_num.shape[1] < 2:
            self.mark_error()
            return False, "Need at least 2 numeric columns"

        method  = str(self.get_property('method') or 'pearson')
        cmap    = str(self.get_property('colormap') or 'coolwarm')
        annot   = bool(self.get_property('annot'))
        mask_up = bool(self.get_property('mask_upper'))
        self.set_progress(20)

        is_distance = 'distance' in method
        actual_method = method.split(' ')[0] # Extract metric name 'euclidean', 'pearson', etc.

        if is_distance:
            from scipy.spatial.distance import pdist, squareform
            # Distance metrics are typically transposed compared to correlation formats, 
            # where you want the distance between columns (features)
            # pdist processes rows as observations. To get column-vs-column distance, transpose:
            dist_array = pdist(df_num.T.values, metric=actual_method)
            matrix_arr = squareform(dist_array)
            matrix_df = pd.DataFrame(matrix_arr, index=df_num.columns, columns=df_num.columns).round(3)
        else:
            matrix_df = df_num.corr(method=actual_method).round(3)

        self.output_values['table'] = TableData(payload=matrix_df)
        self.set_progress(50)

        mask = np.triu(np.ones_like(matrix_df, dtype=bool)) if mask_up else None
        n    = matrix_df.shape[0]
        size = max(4, n * 0.8)

        fig    = Figure(figsize=(size, size * 0.85))
        canvas = FigureCanvasAgg(fig)
        ax     = fig.add_subplot(111)

        fmt = '.2f' if annot else ''
        
        # Color bar styling defaults dependent on type
        if is_distance:
            # Distance: 0 is local (same), larger is far.
            vmin, vmax, center = 0, None, None
            if cmap == 'coolwarm': cmap = 'Reds' # Default visual shift for distances
        else:
            # Correlation: -1, 1 scale center 0
            vmin, vmax, center = -1, 1, 0

        sns.heatmap(matrix_df, ax=ax, mask=mask, cmap=cmap,
                    vmin=vmin, vmax=vmax, center=center,
                    annot=annot, fmt=fmt, annot_kws={'size': 8},
                    linewidths=0.5, square=True,
                    cbar_kws={'shrink': 0.7})

        title_label = method.capitalize()
        if not is_distance and actual_method in ['pearson', 'spearman', 'kendall']:
            method_names = {'pearson': "Pearson's r", 'spearman': "Spearman's ρ", 'kendall': "Kendall's τ"}
            title_label = f"{method_names[actual_method]} Correlation"
            
        ax.set_title(title_label, fontweight='bold', pad=10)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        fig.tight_layout()

        self.output_values['figure'] = FigureData(payload=fig)
        self.set_progress(100)
        self.mark_clean()
        return True, None
