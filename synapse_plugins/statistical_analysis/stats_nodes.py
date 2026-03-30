"""
nodes/stats_nodes.py
====================
Advanced statistical analysis nodes for Synapse — Prism-equivalent features:

  1. LinearRegressionNode       — OLS simple/multiple regression
  2. NonlinearRegressionNode    — Curve fitting (4PL, Hill, exponential, MM, …)
  3. TwoWayANOVANode            — Two-way ANOVA with interaction (Type II SS)
  4. ContingencyAnalysisNode    — Chi-square + Fisher's exact
  5. SurvivalAnalysisNode       — Kaplan-Meier + log-rank test
  6. PCANode                    — Principal component analysis
"""

import NodeGraphQt
import pandas as pd
import numpy as np

from data_models import TableData, FigureData, StatData, ModelData
from nodes.base import BaseExecutionNode, PORT_COLORS


# ── Shared helper ──────────────────────────────────────────────────────────────

def _get_table_df(node, port_name='in'):
    """Return (DataFrame copy, None) or (None, error_string)."""
    in_port = node.inputs().get(port_name)
    if not in_port or not in_port.connected_ports():
        return None, "No input connected"
    cp  = in_port.connected_ports()[0]
    val = cp.node().output_values.get(cp.name())
    if isinstance(val, TableData):
        return val.df.copy(), None
    if isinstance(val, pd.DataFrame):
        return val.copy(), None
    return None, "Input must be a TableData (connect a table output)"


# ── 1. Linear Regression ──────────────────────────────────────────────────────

class LinearRegressionNode(BaseExecutionNode):
    """
    Performs ordinary least-squares (OLS) linear or polynomial regression.

    Set **Degree** > 1 for polynomial regression (e.g. 2 = quadratic, 3 = cubic).
    With degree 1 (default), this is standard linear regression.

    Outputs:
    - **coefficients** — slope, intercept, standard error, 95% CI, and p-values per parameter
    - **residuals** — fitted values, residuals, and standardized residuals for downstream plotting

    Summary statistics: R², adjusted R², F-statistic, and F p-value.
    
    Keywords: linear regression, polynomial regression, OLS, slope, intercept,
              R-squared, coefficient, residuals, predict, fitted values,
              multiple regression, quadratic, cubic, standard curve, Bradford,
              線性回歸, 多項式迴歸, 迴歸分析, 最小二乘法, 斜率, 截距, 決定係數
    """
    __identifier__ = 'nodes.analysis'
    NODE_NAME = 'Linear Regression'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['stat', 'table', 'table', 'model']}

    def __init__(self):
        super().__init__()
        self.add_input('in',            color=PORT_COLORS['table'])
        self.add_output('coefficients', color=PORT_COLORS['stat'])
        self.add_output('residuals',    color=PORT_COLORS['table'])
        self.add_output('curve',        color=PORT_COLORS['table'])
        self.add_output('model',        color=PORT_COLORS['model'])

        self._add_column_selector('x_cols', 'X Column(s)', text='', mode='multi', tab='Parameters')
        self._add_column_selector('y_col',  'Y Column',    text='', mode='single', tab='Parameters')
        self._add_int_spinbox('degree', 'Polynomial Degree', value=1, min_val=1, max_val=10, step=1)
        self.add_checkbox('intercept',  '', text='Include Intercept', state=True)

    def evaluate(self):
        self.reset_progress()
        import statsmodels.api as sm

        df, err = _get_table_df(self)
        if df is None:
            self.mark_error(); return False, err

        self._refresh_column_selectors(df, 'x_cols', 'y_col')

        x_cols_raw = str(self.get_property('x_cols') or '').strip()
        y_col      = str(self.get_property('y_col')  or '').strip()
        intercept  = bool(self.get_property('intercept'))

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not y_col or y_col not in df.columns:
            y_col = num_cols[-1] if num_cols else None
        if not y_col:
            self.mark_error(); return False, "Y column not found"

        if not x_cols_raw:
            x_cols = [c for c in num_cols if c != y_col][:1]
        else:
            x_cols = [c.strip() for c in x_cols_raw.split(',')
                      if c.strip() in df.columns]
        if not x_cols:
            self.mark_error(); return False, "X column(s) not found in the table"

        degree = int(self.get_property('degree') or 1)
        degree = max(1, min(degree, 6))  # clamp to 1-6

        try:
            self.set_progress(20)
            df_c = df[x_cols + [y_col]].dropna()
            X    = df_c[x_cols].astype(float)
            y    = df_c[y_col].astype(float)

            # Polynomial expansion: add x², x³, ... columns
            if degree > 1 and len(x_cols) == 1:
                base_col = x_cols[0]
                x_base = X[base_col]
                for d in range(2, degree + 1):
                    X[f'{base_col}^{d}'] = x_base ** d

            if intercept:
                X = sm.add_constant(X, has_constant='add')

            model = sm.OLS(y, X).fit()
            self.set_progress(70)

            ci   = model.conf_int(alpha=0.05)
            rows = [
                {
                    'Parameter':   pname,
                    'Coefficient': round(float(coef), 6),
                    'Std Error':   round(float(se),   6),
                    't-value':     round(float(tv),   4),
                    'p-value':     round(float(pv),   6),
                    '95% CI Lo':   round(float(lo),   6),
                    '95% CI Hi':   round(float(hi),   6),
                }
                for pname, coef, se, tv, pv, lo, hi in zip(
                    model.params.index, model.params, model.bse,
                    model.tvalues, model.pvalues,
                    ci.iloc[:, 0], ci.iloc[:, 1]
                )
            ]
            coef_df = pd.DataFrame(rows)
            n = int(model.nobs)
            pad = [np.nan] * (len(rows) - 1)
            rmse = float(np.sqrt(model.ssr / max(1, n - len(model.params))))
            coef_df.insert(0, 'n',           [n]                                  + pad)
            coef_df.insert(1, 'R²',          [round(model.rsquared,     6)]       + pad)
            coef_df.insert(2, 'Adj R²',      [round(model.rsquared_adj, 6)]       + pad)
            coef_df.insert(3, 'RMSE',        [round(rmse,               6)]       + pad)
            coef_df.insert(4, 'AIC',         [round(float(model.aic),   2)]       + pad)
            coef_df.insert(5, 'BIC',         [round(float(model.bic),   2)]       + pad)
            coef_df.insert(6, 'F-statistic', [round(float(model.fvalue),    4)]   + pad)
            coef_df.insert(7, 'F p-value',   [round(float(model.f_pvalue),  6)]   + pad)

            resid_df                        = df_c.copy()
            resid_df['Predicted']           = model.fittedvalues.values
            resid_df['Residual']            = model.resid.values
            std_r = model.resid.std()
            if std_r > 0:
                resid_df['Standardized Residual'] = (model.resid / std_r).values
            else:
                resid_df['Standardized Residual'] = np.zeros(len(model.resid))

            # Smooth fitted curve with 95% CI for RegressionPlotNode
            if len(x_cols) == 1:
                from scipy.stats import t as _t
                x_raw = df_c[x_cols[0]].astype(float)
                x_curve = np.linspace(float(x_raw.min()), float(x_raw.max()), 200)
                X_curve = pd.DataFrame({x_cols[0]: x_curve})
                if degree > 1:
                    for d in range(2, degree + 1):
                        X_curve[f'{x_cols[0]}^{d}'] = x_curve ** d
                if intercept:
                    X_curve = sm.add_constant(X_curve, has_constant='add')
                y_curve = model.predict(X_curve)

                # CI band from OLS prediction standard error
                n_obs = int(model.nobs)
                n_params = len(model.params)
                _dof = max(1, n_obs - n_params)
                s_err = float(np.sqrt(model.ssr / _dof))
                t_c = _t.ppf(0.975, _dof)
                x_vals = x_raw.values
                x_mean = float(x_vals.mean())
                ss_xx = float(np.sum((x_vals - x_mean) ** 2))
                se_curve = s_err * np.sqrt(
                    1.0 / n_obs + (x_curve - x_mean) ** 2 / max(ss_xx, 1e-300)
                )
                ci_lo = y_curve - t_c * se_curve
                ci_hi = y_curve + t_c * se_curve

                curve_df = pd.DataFrame({
                    x_cols[0]:        x_curve,
                    f'{y_col}_fit':   y_curve,
                    f'{y_col}_ci_lo': ci_lo,
                    f'{y_col}_ci_hi': ci_hi,
                })
            else:
                # Multiple X — can't generate a smooth 1D curve; output fitted values instead
                curve_df = resid_df[[*x_cols, 'Predicted']].rename(
                    columns={'Predicted': f'{y_col}_fit'})

            self.output_values['coefficients'] = TableData(payload=coef_df)
            self.output_values['residuals']    = TableData(payload=resid_df)
            self.output_values['curve']        = TableData(payload=curve_df)
            self.output_values['model']        = ModelData(
                payload=model,
                metadata={
                    'model_type':  'linear',
                    'x_columns':   x_cols,
                    'y_column':    y_col,
                    'degree':      degree,
                    'intercept':   intercept,
                    'model_name':  f'OLS (degree={degree})',
                },
            )
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error(); return False, str(e)


# ── 2. Nonlinear Regression ────────────────────────────────────────────────────

class NonlinearRegressionNode(BaseExecutionNode):
    """
    Fits nonlinear curves to XY data using `scipy.optimize.curve_fit`.

    Built-in models:
    - *4PL (EC50 / Dose-Response)* — four-parameter logistic for IC50/EC50
    - *Hill Equation* — sigmoidal binding/dose-response
    - *One-Phase Exponential Decay* — single-rate decay to plateau
    - *Two-Phase Exponential Decay* — fast + slow decay components
    - *Exponential Growth* — unbounded exponential increase
    - *Michaelis-Menten* — enzyme kinetics saturation curve
    - *Gompertz Growth* — asymmetric sigmoidal growth
    - *Sigmoidal (Logistic)* — symmetric S-curve

    Outputs best-fit parameters with 95% CI and a smooth predicted curve table.

    Keywords: curve fitting, nonlinear regression, EC50, IC50, Hill equation,
              dose-response, exponential, Michaelis-Menten, Gompertz, logistic,
              4-parameter logistic, sigmoid, 非線性迴歸, 劑量反應曲線, 曲線擬合
    """
    __identifier__ = 'nodes.analysis'
    NODE_NAME = 'Nonlinear Regression'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['stat', 'table', 'model']}

    MODELS = [
        '4PL (EC50 / Dose-Response)',
        'Hill Equation',
        'One-Phase Exponential Decay',
        'Two-Phase Exponential Decay',
        'Exponential Growth',
        'Michaelis-Menten',
        'Gompertz Growth',
        'Sigmoidal (Logistic)',
    ]

    def __init__(self):
        super().__init__()
        self.add_input('in',         color=PORT_COLORS['table'])
        self.add_output('parameters', color=PORT_COLORS['stat'])
        self.add_output('curve',      color=PORT_COLORS['table'])
        self.add_output('model',      color=PORT_COLORS['model'])

        self._add_column_selector('x_col', 'X Column', text='', mode='single', tab='Parameters')
        self._add_column_selector('y_col', 'Y Column', text='', mode='single', tab='Parameters')
        self.add_combo_menu('model',    'Model',                 items=self.MODELS)
        import NodeGraphQt as _nq
        self.create_property('n_points', 200,
                             widget_type=_nq.constants.NodePropWidgetEnum.QSPIN_BOX.value,
                             tab='Parameters')
        self._add_float_spinbox('x_min', 'X Min (0=auto)', value=0.0, min_val=-1e9, max_val=1e9, step=0.1, decimals=4)
        self._add_float_spinbox('x_max', 'X Max (0=auto)', value=0.0, min_val=-1e9, max_val=1e9, step=0.1, decimals=4)

    # ── model functions ────────────────────────────────────────────────────────

    @staticmethod
    def _4pl(x, bottom, top, ec50, hill):
        return bottom + (top - bottom) / (1.0 + (ec50 / np.clip(x, 1e-300, None)) ** hill)

    @staticmethod
    def _hill(x, vmax, k, n):
        return vmax * x ** n / (k ** n + x ** n)

    @staticmethod
    def _one_exp_decay(x, y0, plateau, k):
        return plateau + (y0 - plateau) * np.exp(-k * x)

    @staticmethod
    def _two_exp_decay(x, y0, plateau, k1, k2, pct_fast):
        f     = np.clip(pct_fast / 100.0, 0.0, 1.0)
        span1 = (y0 - plateau) * f
        span2 = (y0 - plateau) * (1.0 - f)
        return plateau + span1 * np.exp(-k1 * x) + span2 * np.exp(-k2 * x)

    @staticmethod
    def _exp_growth(x, y0, k):
        return y0 * np.exp(k * x)

    @staticmethod
    def _mm(x, vmax, km):
        return vmax * x / (km + x)

    @staticmethod
    def _gompertz(x, a, b, c):
        return a * np.exp(-b * np.exp(-c * x))

    @staticmethod
    def _logistic(x, l, k, x0):
        return l / (1.0 + np.exp(-k * (x - x0)))

    def _get_model(self, name):
        return {
            '4PL (EC50 / Dose-Response)':  (self._4pl,           ['Bottom', 'Top', 'EC50', 'Hill Slope']),
            'Hill Equation':               (self._hill,          ['Vmax', 'K (half-max)', 'n (Hill)']),
            'One-Phase Exponential Decay': (self._one_exp_decay,  ['Y0', 'Plateau', 'k']),
            'Two-Phase Exponential Decay': (self._two_exp_decay,  ['Y0', 'Plateau', 'k1', 'k2', 'Pct Fast (%)']),
            'Exponential Growth':          (self._exp_growth,    ['Y0', 'k']),
            'Michaelis-Menten':            (self._mm,            ['Vmax', 'Km']),
            'Gompertz Growth':             (self._gompertz,      ['A', 'B', 'C']),
            'Sigmoidal (Logistic)':        (self._logistic,      ['L', 'k', 'x0']),
        }[name]

    def _auto_p0(self, name, x, y):
        med_x = float(np.median(x))
        ymin, ymax = float(y.min()), float(y.max())
        return {
            '4PL (EC50 / Dose-Response)':  [ymin, ymax, max(med_x, 1e-9), 1.0],
            'Hill Equation':               [ymax, max(med_x, 1e-9), 1.0],
            'One-Phase Exponential Decay': [ymax, ymin, 0.1],
            'Two-Phase Exponential Decay': [ymax, ymin, 0.5, 0.05, 50.0],
            'Exponential Growth':          [max(abs(ymin), 1e-6), 0.1],
            'Michaelis-Menten':            [ymax, max(med_x, 1e-9)],
            'Gompertz Growth':             [ymax, 1.0, 0.5],
            'Sigmoidal (Logistic)':        [ymax - ymin, 1.0, med_x],
        }.get(name)

    def evaluate(self):
        self.reset_progress()
        from scipy.optimize import curve_fit
        from scipy.stats   import t as t_dist

        df, err = _get_table_df(self)
        if df is None:
            self.mark_error(); return False, err

        self._refresh_column_selectors(df, 'x_col', 'y_col')

        x_col      = str(self.get_property('x_col') or '').strip()
        y_col      = str(self.get_property('y_col') or '').strip()
        model_name = self.get_property('model') or self.MODELS[0]

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not x_col or x_col not in df.columns:
            x_col = num_cols[0] if num_cols else None
        if not y_col or y_col not in df.columns:
            y_col = num_cols[1] if len(num_cols) > 1 else None
        if x_col is None or y_col is None:
            self.mark_error(); return False, "Need at least two numeric columns"

        try:
            df_c   = df[[x_col, y_col]].dropna()
            xdata  = df_c[x_col].astype(float).values
            ydata  = df_c[y_col].astype(float).values
            n      = len(xdata)
            if n < 4:
                self.mark_error()
                return False, "Need at least 4 data points for curve fitting"

            self.set_progress(20)
            func, param_names = self._get_model(model_name)
            p0 = self._auto_p0(model_name, xdata, ydata)

            popt, pcov = curve_fit(func, xdata, ydata, p0=p0, maxfev=20000)
            self.set_progress(60)

            perr  = np.sqrt(np.diag(pcov))
            dof   = max(1, n - len(popt))
            t_val = t_dist.ppf(0.975, dof)
            ci_lo = popt - t_val * perr
            ci_hi = popt + t_val * perr

            y_pred = func(xdata, *popt)
            ss_res = float(np.sum((ydata - y_pred) ** 2))
            ss_tot = float(np.sum((ydata - ydata.mean()) ** 2))
            r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            rmse   = float(np.sqrt(ss_res / n))
            k      = len(popt)

            # Additional fit statistics
            reduced_chi2 = ss_res / dof if dof > 0 else float('nan')
            ln_lik = -0.5 * n * (np.log(2 * np.pi * ss_res / n) + 1)
            aic = 2 * k - 2 * ln_lik
            bic = k * np.log(n) - 2 * ln_lik
            # F-test: model vs null (mean-only)
            if dof > 0 and ss_tot > 0 and k > 1:
                f_stat = ((ss_tot - ss_res) / (k - 1)) / (ss_res / dof)
                from scipy.stats import f as f_dist
                f_pval = 1.0 - f_dist.cdf(f_stat, k - 1, dof)
            else:
                f_stat, f_pval = float('nan'), float('nan')

            # Per-parameter t-values and p-values
            t_values = popt / np.where(perr > 0, perr, np.nan)
            p_values = 2.0 * (1.0 - t_dist.cdf(np.abs(t_values), dof))

            rows = [
                {
                    'Parameter':      pname,
                    'Best-fit Value': round(float(val), 6),
                    'Std Error':      round(float(se),  6),
                    't-value':        round(float(tv),  4),
                    'p-value':        round(float(pv),  6),
                    '95% CI Lo':      round(float(lo),  6),
                    '95% CI Hi':      round(float(hi),  6),
                }
                for pname, val, se, tv, pv, lo, hi in zip(
                    param_names, popt, perr, t_values, p_values, ci_lo, ci_hi)
            ]
            params_df = pd.DataFrame(rows)
            pad = [np.nan] * (len(rows) - 1)
            params_df.insert(0, 'Model',       [model_name]                       + [''] * (len(rows) - 1))
            params_df.insert(1, 'n',           [int(n)]                           + pad)
            params_df.insert(2, 'R²',          [round(r2,                6)]      + pad)
            params_df.insert(3, 'RMSE',        [round(rmse,              6)]      + pad)
            params_df.insert(4, 'Reduced χ²',  [round(reduced_chi2,      6)]      + pad)
            params_df.insert(5, 'AIC',         [round(float(aic),        2)]      + pad)
            params_df.insert(6, 'BIC',         [round(float(bic),        2)]      + pad)
            params_df.insert(7, 'F-statistic', [round(float(f_stat),     4)]      + pad)
            params_df.insert(8, 'F p-value',   [round(float(f_pval),     6)]      + pad)

            # Smooth fitted curve with 95% CI band (delta method)
            n_pts = max(50, int(self.get_property('n_points') or 200))
            x_min_v = float(self.get_property('x_min') or 0.0)
            x_max_v = float(self.get_property('x_max') or 0.0)
            x_lo = x_min_v if x_min_v != 0.0 else float(xdata.min())
            x_hi = x_max_v if x_max_v != 0.0 else float(xdata.max())
            x_curve = np.linspace(x_lo, x_hi, n_pts)
            y_curve = func(x_curve, *popt)

            # Delta method: var(f(x)) ≈ J(x) @ pcov @ J(x).T
            # J = Jacobian of f w.r.t. parameters via finite differences
            eps = np.sqrt(np.finfo(float).eps)
            J = np.zeros((n_pts, k))
            for i in range(k):
                p_up = popt.copy()
                delta = max(abs(popt[i]) * eps, eps)
                p_up[i] += delta
                J[:, i] = (func(x_curve, *p_up) - y_curve) / delta
            y_var = np.array([float(J[j, :] @ pcov @ J[j, :]) for j in range(n_pts)])
            y_se  = np.sqrt(np.clip(y_var, 0, None))
            ci_band_lo = y_curve - t_val * y_se
            ci_band_hi = y_curve + t_val * y_se

            curve_df = pd.DataFrame({
                x_col:            x_curve,
                f'{y_col}_fit':   y_curve,
                f'{y_col}_ci_lo': ci_band_lo,
                f'{y_col}_ci_hi': ci_band_hi,
            })

            self.output_values['parameters'] = StatData(payload=params_df)
            self.output_values['curve']      = TableData(payload=curve_df)

            # Wrap the fitted function + params as a ModelData for prediction
            _func_copy = func
            _popt_copy = popt.copy()
            def _predict_fn(x_arr):
                return _func_copy(np.asarray(x_arr, dtype=float), *_popt_copy)
            self.output_values['model'] = ModelData(
                payload=_predict_fn,
                metadata={
                    'model_type':   'nonlinear',
                    'x_columns':    [x_col],
                    'y_column':     y_col,
                    'model_name':   model_name,
                    'param_names':  param_names,
                    'param_values': [float(v) for v in popt],
                },
            )
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error(); return False, str(e)


# ── 2b. Model Predict ─────────────────────────────────────────────────────────

class ModelPredictNode(BaseExecutionNode):
    """
    Predicts Y values from a fitted model and a new data table.

    Connect the **model** output from Linear Regression or Nonlinear Regression,
    then provide a table with the X column to predict on.

    The node auto-detects the X column name from the model metadata.
    Override with the **X Column** field if the new table uses a different name.

    Outputs the input table with an added **Predicted** column.
    
    Keywords: predict, interpolate, standard curve, estimate, concentration,
              Bradford, ELISA, 預測, 插值, 標準曲線
    """
    __identifier__ = 'nodes.analysis'
    NODE_NAME = 'Model Predict'
    PORT_SPEC = {'inputs': ['model', 'table'], 'outputs': ['table']}

    def __init__(self):
        super().__init__()
        self.add_input('model', color=PORT_COLORS['model'])
        self.add_input('data',  color=PORT_COLORS['table'])
        self.add_output('out',  color=PORT_COLORS['table'])

        self._add_column_selector('x_col', 'X Column (blank=auto)', text='', mode='single', tab='Parameters')
        self.add_text_input('pred_name',  'Output Column Name',    text='Predicted')
        self.add_checkbox('inverse',      '', text='Inverse predict (given Y → predict X)', state=False)
        self._add_float_spinbox('inv_x_min', 'Inverse X Min (0=auto)', value=0.0, min_val=-1e9, max_val=1e9, step=0.1, decimals=4)
        self._add_float_spinbox('inv_x_max', 'Inverse X Max (0=auto)', value=0.0, min_val=-1e9, max_val=1e9, step=0.1, decimals=4)


    def evaluate(self):
        self.reset_progress()
        import statsmodels.api as sm

        # ── Get model ────────────────────────────────────────────────
        model_port = self.inputs().get('model')
        if not model_port or not model_port.connected_ports():
            self.mark_error()
            return False, "No model connected."
        cp = model_port.connected_ports()[0]
        model_data = cp.node().output_values.get(cp.name())
        if not isinstance(model_data, ModelData):
            self.mark_error()
            return False, "Input is not a ModelData."

        # ── Get data table ───────────────────────────────────────────
        df, err = _get_table_df(self, port_name='data')
        if df is None:
            self.mark_error()
            return False, err or "No data table connected."

        self._refresh_column_selectors(df, 'x_col')

        meta       = model_data.metadata or {}
        model_type = meta.get('model_type', 'unknown')
        x_columns  = meta.get('x_columns', [])
        y_column   = meta.get('y_column', 'Y')
        degree     = meta.get('degree', 1)
        intercept  = meta.get('intercept', True)

        x_col_override = str(self.get_property('x_col') or '').strip()
        pred_name      = str(self.get_property('pred_name') or 'Predicted').strip()
        inverse        = bool(self.get_property('inverse'))
        inv_x_min      = float(self.get_property('inv_x_min') or 0.0)
        inv_x_max      = float(self.get_property('inv_x_max') or 0.0)

        self.set_progress(20)

        try:
            if model_type == 'linear':
                # statsmodels OLS result
                ols_model = model_data.payload

                if inverse:
                    # Inverse prediction: given Y, solve for X
                    # Only works for single-X linear/polynomial models
                    if len(x_columns) != 1:
                        self.mark_error()
                        return False, "Inverse prediction only works with single-X models."
                    y_col_in = x_col_override or y_column
                    if y_col_in not in df.columns:
                        self.mark_error()
                        return False, f"Column '{y_col_in}' not found for inverse prediction."
                    y_vals = df[y_col_in].astype(float).values
                    params = ols_model.params.values
                    from scipy.optimize import brentq
                    x_pred = []
                    # Use user-specified range, or auto from training data
                    if inv_x_min != 0.0 or inv_x_max != 0.0:
                        x_lo, x_hi = inv_x_min, inv_x_max
                    else:
                        x_range = ols_model.model.exog[:, -1 if not intercept else 1]
                        x_lo, x_hi = float(x_range.min()), float(x_range.max())
                        x_lo -= abs(x_lo) * 0.1 + 1e-9
                        x_hi += abs(x_hi) * 0.1 + 1e-9
                    for y_target in y_vals:
                        def _resid(x_val):
                            row = [x_val ** d for d in range(1, degree + 1)]
                            if intercept:
                                row = [1.0] + row
                            return float(np.dot(params, row)) - y_target
                        try:
                            x_pred.append(float(brentq(_resid, x_lo, x_hi)))
                        except ValueError:
                            x_pred.append(float('nan'))
                    result = df.copy()
                    result[pred_name] = x_pred
                else:
                    # Forward prediction: given X, predict Y
                    x_col_name = x_col_override or (x_columns[0] if x_columns else '')
                    if x_col_name not in df.columns:
                        self.mark_error()
                        return False, f"X column '{x_col_name}' not found. Set 'X Column' manually."
                    X_new = df[[x_col_name]].astype(float).copy()
                    if degree > 1:
                        base = X_new[x_col_name]
                        for d in range(2, degree + 1):
                            X_new[f'{x_col_name}^{d}'] = base ** d
                    if intercept:
                        X_new = sm.add_constant(X_new, has_constant='add')
                    result = df.copy()
                    result[pred_name] = ols_model.predict(X_new)

            elif model_type == 'nonlinear':
                predict_fn = model_data.payload  # callable
                x_col_name = x_col_override or (x_columns[0] if x_columns else '')

                if inverse:
                    # Inverse: given Y, find X numerically
                    y_col_in = x_col_override or meta.get('y_column', '')
                    if y_col_in not in df.columns:
                        self.mark_error()
                        return False, f"Column '{y_col_in}' not found for inverse prediction."
                    from scipy.optimize import brentq
                    y_vals = df[y_col_in].astype(float).values
                    # Use user-specified range, or auto-estimate from model params
                    if inv_x_min != 0.0 or inv_x_max != 0.0:
                        x_lo, x_hi = inv_x_min, inv_x_max
                    else:
                        pvals = meta.get('param_values', [])
                        x_lo = 1e-12
                        x_hi = max(abs(v) for v in pvals) * 100 if pvals else 1e6
                    x_pred = []
                    for y_target in y_vals:
                        try:
                            x_pred.append(float(brentq(
                                lambda x: float(predict_fn(np.array([x]))[0]) - y_target,
                                x_lo, x_hi,
                            )))
                        except (ValueError, IndexError):
                            x_pred.append(float('nan'))
                    result = df.copy()
                    result[pred_name] = x_pred
                else:
                    if x_col_name not in df.columns:
                        self.mark_error()
                        return False, f"X column '{x_col_name}' not found. Set 'X Column' manually."
                    x_vals = df[x_col_name].astype(float).values
                    result = df.copy()
                    result[pred_name] = predict_fn(x_vals)
            else:
                self.mark_error()
                return False, f"Unsupported model type: {model_type}"

            self.output_values['out'] = TableData(payload=result)
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)


# ── 3. Two-Way ANOVA ───────────────────────────────────────────────────────────

class TwoWayANOVANode(BaseExecutionNode):
    """
    Performs two-way analysis of variance with interaction term (Type II SS).

    Input must be in long format with two factor columns and one numeric value column.

    Outputs:
    - **anova_table** — sum of squares, df, F-statistic, and p-value per source
    - **group_means** — mean, SD, SEM, and N for every factor combination

    Keywords: two-way anova, 2-way anova, factorial anova, interaction effect,
              main effects, between subjects, F-test, repeated measures,
              雙因子變異數分析, 交互作用, F檢定, 因子分析
    """
    __identifier__ = 'nodes.analysis'
    NODE_NAME = 'Two-Way ANOVA'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['stat', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('in',           color=PORT_COLORS['table'])
        self.add_output('anova_table', color=PORT_COLORS['stat'])
        self.add_output('group_means', color=PORT_COLORS['table'])

        self.add_text_input('factor1',   'Factor A Column', text='')
        self.add_text_input('factor2',   'Factor B Column', text='')
        self.add_text_input('value_col', 'Value Column',    text='')

    def evaluate(self):
        self.reset_progress()
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        df, err = _get_table_df(self)
        if df is None:
            self.mark_error(); return False, err

        factor1   = str(self.get_property('factor1')   or '').strip()
        factor2   = str(self.get_property('factor2')   or '').strip()
        value_col = str(self.get_property('value_col') or '').strip()

        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not factor1 or factor1 not in df.columns:
            factor1 = cat_cols[0] if cat_cols else (num_cols[0] if len(num_cols) > 2 else None)
        if not factor2 or factor2 not in df.columns:
            factor2 = (cat_cols[1] if len(cat_cols) > 1
                       else (num_cols[1] if len(num_cols) > 2 else None))
        if not value_col or value_col not in df.columns:
            value_col = next((c for c in num_cols if c not in (factor1, factor2)), None)

        if not all([factor1, factor2, value_col]):
            self.mark_error()
            return False, ("Could not identify Factor A, Factor B, and Value columns. "
                           "Please fill in the column names.")

        try:
            self.set_progress(20)
            df_c = df[[factor1, factor2, value_col]].dropna().copy()
            # Use safe internal names for the statsmodels formula
            df_c.columns = ['_F1', '_F2', '_V']
            df_c['_F1'] = df_c['_F1'].astype(str)
            df_c['_F2'] = df_c['_F2'].astype(str)

            formula = '_V ~ C(_F1) + C(_F2) + C(_F1):C(_F2)'
            model   = smf.ols(formula, data=df_c).fit()
            aov     = anova_lm(model, typ=2)
            self.set_progress(70)

            aov_out = aov.reset_index()
            aov_out.columns = ['Source', 'Sum of Squares', 'df', 'F', 'p-value']
            aov_out['Source'] = (
                aov_out['Source']
                .str.replace('C(_F1)', factor1, regex=False)
                .str.replace('C(_F2)', factor2, regex=False)
                .str.replace(':',      ' × ',   regex=False)
            )
            aov_out['Significant'] = aov_out['p-value'] < 0.05

            means = (df_c.groupby(['_F1', '_F2'])['_V']
                     .agg(Mean='mean', SD='std', N='count')
                     .reset_index())
            means.columns = [factor1, factor2, 'Mean', 'SD', 'N']
            means['SEM']  = means['SD'] / np.sqrt(means['N'])

            self.output_values['anova_table'] = StatData(payload=aov_out.round(6))
            self.output_values['group_means'] = TableData(payload=means.round(6))
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error(); return False, str(e)


# ── 4. Contingency Analysis ────────────────────────────────────────────────────

class ContingencyAnalysisNode(BaseExecutionNode):
    """
    Tests categorical association using chi-square and Fisher's exact tests.

    Input types:
    - *Raw Data (two columns)* — a crosstab is built automatically from two categorical columns
    - *Contingency Matrix* — a pre-built matrix of observed counts

    Outputs:
    - **test_results** — Pearson chi-square, Yates-corrected chi-square, and Fisher's exact (2x2)
    - **observed_counts** — the observed contingency table
    - **expected_counts** — expected counts under the null hypothesis

    Keywords: chi-square, chi2, fisher exact, contingency table, odds ratio,
              relative risk, categorical, crosstab, association,
              卡方檢定, 費雪精確檢定, 列聯表, 勝算比, 類別資料
    """
    __identifier__ = 'nodes.analysis'
    NODE_NAME = 'Contingency Analysis'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['stat', 'table', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('in',               color=PORT_COLORS['table'])
        self.add_output('test_results',    color=PORT_COLORS['stat'])
        self.add_output('observed_counts', color=PORT_COLORS['table'])
        self.add_output('expected_counts', color=PORT_COLORS['table'])

        self.add_combo_menu('input_type', 'Input Type',
                            items=['Raw Data (two columns)', 'Contingency Matrix'])
        self.add_text_input('col1', 'Row Variable Column', text='')
        self.add_text_input('col2', 'Col Variable Column', text='')

    def evaluate(self):
        self.reset_progress()
        from scipy.stats import chi2_contingency, fisher_exact

        df, err = _get_table_df(self)
        if df is None:
            self.mark_error(); return False, err

        input_type = self.get_property('input_type') or 'Raw Data (two columns)'
        col1       = str(self.get_property('col1') or '').strip()
        col2       = str(self.get_property('col2') or '').strip()

        try:
            self.set_progress(20)
            if 'Raw Data' in input_type:
                if not col1 or col1 not in df.columns:
                    col1 = df.columns[0]
                if not col2 or col2 not in df.columns:
                    col2 = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                ct = pd.crosstab(df[col1], df[col2])
            else:
                first = df.columns[0]
                if not pd.api.types.is_numeric_dtype(df[first]):
                    ct = df.set_index(first)
                else:
                    ct = df.copy()
                ct = ct.select_dtypes(include=[np.number])

            observed = ct.values.astype(float)
            self.set_progress(40)

            rows = []
            chi2, p_chi2, dof, expected = chi2_contingency(observed, correction=False)
            rows.append({
                'Test': 'Pearson Chi-square',
                'Statistic': round(chi2, 4), 'df': int(dof),
                'p-value': round(p_chi2, 6), 'Significant': p_chi2 < 0.05,
            })

            chi2_y, p_y, _, _ = chi2_contingency(observed, correction=True)
            rows.append({
                'Test': "Chi-sq (Yates' corr)",
                'Statistic': round(chi2_y, 4), 'df': int(dof),
                'p-value': round(p_y, 6), 'Significant': p_y < 0.05,
            })

            if observed.shape == (2, 2):
                odds, p_fish = fisher_exact(observed.astype(int))
                rows.append({
                    'Test': "Fisher's Exact",
                    'Statistic': round(float(odds), 4), 'df': 1,
                    'p-value': round(float(p_fish), 6),
                    'Significant': p_fish < 0.05,
                    'Note': f'Odds Ratio = {round(float(odds), 4)}',
                })

            test_df = pd.DataFrame(rows)
            exp_df  = pd.DataFrame(
                expected.round(2), index=ct.index, columns=ct.columns
            )

            self.output_values['test_results']    = StatData(payload=test_df)
            self.output_values['observed_counts'] = TableData(payload=ct)
            self.output_values['expected_counts'] = TableData(payload=exp_df)
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error(); return False, str(e)


# ── 5. Survival Analysis ───────────────────────────────────────────────────────

class SurvivalAnalysisNode(BaseExecutionNode):
    """
    Performs Kaplan-Meier survival analysis with log-rank test.

    Input columns:
    - **Time Column** — duration or follow-up time
    - **Event Column** — `1` = event occurred, `0` = censored
    - **Group Column** (optional) — categorical grouping for multi-group comparison

    Outputs:
    - **km_table** — survival function with 95% CI (feed into SurvivalPlotNode)
    - **log_rank** — omnibus log-rank test statistic and p-value
    - **pairwise_stat** — pairwise log-rank results with optional p-value adjustment

    **P-Adj Method** — multiple comparison correction for pairwise tests.

    Keywords: kaplan meier, survival analysis, log-rank, censored, time-to-event,
              hazard, mortality, 生存分析, 存活曲線, 卡普蘭-邁耶, 截尾
    """
    __identifier__ = 'nodes.analysis'
    NODE_NAME = 'Survival Analysis'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table', 'stat', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('in',        color=PORT_COLORS['table'])
        self.add_output('km_table', color=PORT_COLORS['table'])
        self.add_output('log_rank', color=PORT_COLORS['stat'])
        self.add_output('pairwise_stat', color=PORT_COLORS['table'])

        self.add_text_input('time_col',  'Time Column',                        text='')
        self.add_text_input('event_col', 'Event Column (1=event, 0=censored)', text='')
        self.add_text_input('group_col', 'Group Column (optional)',             text='')

        p_adj_methods = ['none', 'bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky']
        self.add_combo_menu('p_adj_method', 'P-Adj Method (Pairwise)', items=p_adj_methods)
        
        self.add_text_input('custom_comparisons', 'Comparisons (A|B, C|D)', text='')

    # ── Kaplan-Meier estimator ─────────────────────────────────────────────────

    @staticmethod
    def _km_estimate(times, events):
        """
        Compute the KM survival function (Greenwood variance → 95% CI).
        Returns a DataFrame: Time, At_Risk, Events, Censored, Survival,
                             Lower_95CI, Upper_95CI.
        """
        times  = np.asarray(times,  dtype=float)
        events = np.asarray(events, dtype=float)
        order  = np.argsort(times)
        times  = times[order]
        events = events[order]
        n      = len(times)

        rows     = [{'Time': 0.0, 'At_Risk': n, 'Events': 0, 'Censored': 0,
                     'Survival': 1.0, 'Lower_95CI': 1.0, 'Upper_95CI': 1.0}]
        survival = 1.0
        var_sum  = 0.0
        event_times_set = set()

        for t in np.unique(times[events == 1]):
            at_risk = int(np.sum(times >= t))
            d       = int(np.sum((times == t) & (events == 1)))
            if at_risk == 0 or d == 0:
                continue
            survival *= (1.0 - d / at_risk)
            denom     = at_risk * (at_risk - d)
            var_sum  += d / denom if denom > 0 else 0.0
            se        = survival * np.sqrt(var_sum)
            # Complementary log-log CI: stays within (0,1) without clamping,
            # better coverage at the tails than the linear ±1.96·SE formula.
            if 0.0 < survival < 1.0:
                log_s   = np.log(survival)          # negative
                theta   = np.exp(1.96 * se / (survival * abs(log_s)))
                lo      = survival ** theta          # > survival (wrong dir) — swap
                hi      = survival ** (1.0 / theta)
                lo, hi  = min(lo, hi), max(lo, hi)  # ensure lo < hi
                lo      = max(0.0, lo)
                hi      = min(1.0, hi)
            else:
                lo, hi  = survival, survival
            censored_t = int(np.sum((times == t) & (events == 0)))
            rows.append({
                'Time': float(t), 'At_Risk': at_risk, 'Events': d,
                'Censored': censored_t,
                'Survival': round(survival, 6),
                'Lower_95CI': round(lo, 6), 'Upper_95CI': round(hi, 6),
            })
            event_times_set.add(float(t))

        # Add rows for censored observations whose time falls between event times.
        # rows[] is already in event-time order; build lookup arrays from it.
        event_arr = np.array([r['Time']       for r in rows])
        surv_arr  = np.array([r['Survival']   for r in rows])
        lo_arr    = np.array([r['Lower_95CI'] for r in rows])
        hi_arr    = np.array([r['Upper_95CI'] for r in rows])

        for t in np.unique(times[events == 0]):
            if float(t) in event_times_set:
                continue  # already counted in the coincident event row
            idx = max(0, int(np.searchsorted(event_arr, t, side='right')) - 1)
            cnt = int(np.sum((times == t) & (events == 0)))
            rows.append({
                'Time': float(t), 'At_Risk': int(np.sum(times >= t)),
                'Events': 0, 'Censored': cnt,
                'Survival': float(surv_arr[idx]),
                'Lower_95CI': float(lo_arr[idx]),
                'Upper_95CI': float(hi_arr[idx]),
            })

        return pd.DataFrame(rows).sort_values('Time', ignore_index=True)

    # ── Log-rank test ──────────────────────────────────────────────────────────

    @staticmethod
    def _log_rank(groups_data):
        """
        Log-rank (Mantel-Cox) test for 2+ groups.
        groups_data : list of (times_array, events_array).
        Returns (chi2_stat, p_value, degrees_of_freedom).

        Uses the hypergeometric variance-covariance matrix (not Pearson E),
        matching scipy.stats.logrank / lifelines for 2-group comparisons.
        """
        from scipy.stats import chi2 as chi2_dist

        all_event_times = np.unique(
            np.concatenate([g[0][g[1] == 1] for g in groups_data])
        )
        k = len(groups_data)
        O = np.zeros(k)
        E = np.zeros(k)
        V = np.zeros((k, k))   # variance-covariance matrix

        for t in all_event_times:
            n_i = np.array([int(np.sum(g[0] >= t)) for g in groups_data])
            d_i = np.array([int(np.sum((g[0] == t) & (g[1] == 1))) for g in groups_data])
            n   = int(n_i.sum())
            d   = int(d_i.sum())
            if n == 0:
                continue
            O += d_i
            E += n_i * (d / n)
            if n <= 1:
                continue  # variance term has n*(n-1) in denominator → skip
            factor = d * (n - d) / (n ** 2 * (n - 1))
            for i in range(k):
                V[i, i] += n_i[i] * (n - n_i[i]) * factor
                for j in range(i + 1, k):
                    off = -n_i[i] * n_i[j] * factor
                    V[i, j] += off
                    V[j, i] += off

        # The system has rank k-1 (rows/cols sum to zero), so drop the last group.
        z    = (O - E)[:k - 1]
        Vsub = V[:k - 1, :k - 1]
        try:
            stat = float(z @ np.linalg.solve(Vsub, z))
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular (e.g. no variance at all)
            stat = 0.0

        dof = k - 1
        p   = float(1.0 - chi2_dist.cdf(max(stat, 0.0), dof))
        return stat, p, dof

    def evaluate(self):
        self.reset_progress()

        df, err = _get_table_df(self)
        if df is None:
            self.mark_error(); return False, err

        time_col  = str(self.get_property('time_col')  or '').strip()
        event_col = str(self.get_property('event_col') or '').strip()
        group_col = str(self.get_property('group_col') or '').strip()
        
        p_adj_method = self.get_property('p_adj_method')
        custom_str = str(self.get_property('custom_comparisons')).strip()

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not time_col  or time_col  not in df.columns:
            time_col  = num_cols[0] if num_cols else None
        if not event_col or event_col not in df.columns:
            event_col = num_cols[1] if len(num_cols) > 1 else None
        if not time_col or not event_col:
            self.mark_error(); return False, "Need time and event columns"
        if group_col and group_col not in df.columns:
            group_col = ''

        try:
            self.set_progress(20)
            cols = [c for c in [time_col, event_col, group_col] if c]
            df_c = df[cols].dropna().copy()
            df_c[time_col]  = df_c[time_col].astype(float)
            df_c[event_col] = df_c[event_col].astype(float)

            km_frames = []
            if group_col:
                groups      = df_c[group_col].unique()
                groups_data = []
                for i, g in enumerate(groups):
                    sub = df_c[df_c[group_col] == g]
                    km  = self._km_estimate(sub[time_col].values, sub[event_col].values)
                    km.insert(0, 'Group', g)
                    km_frames.append(km)
                    groups_data.append((sub[time_col].values, sub[event_col].values))
                    self.set_progress(20 + 50 * (i + 1) // len(groups))

                km_table     = pd.concat(km_frames, ignore_index=True)
                stat, p, dof = self._log_rank(groups_data)
                lr_df = pd.DataFrame([{
                    'Test': 'Log-Rank (Omnibus)', 'Chi-square': round(stat, 4),
                    'df': dof, 'p-value': round(p, 6), 'Significant': p < 0.05,
                }])

                # Pairwise evaluation
                unique_groups = list(groups)
                pairs = []
                
                if custom_str:
                    for pair in custom_str.split(','):
                        parts = [p.strip() for p in pair.split('|')]
                        if len(parts) == 2 and all(parts):
                            g1, g2 = parts[0], parts[1]
                            if g1 in unique_groups and g2 in unique_groups:
                                cp = tuple(sorted((g1, g2)))
                                if all(tuple(sorted(existing)) != cp for existing in pairs):
                                    pairs.append((g1, g2))

                pairwise_results = []
                if pairs:
                    raw_pvals = []
                    for g1, g2 in pairs:
                        sub1 = df_c[df_c[group_col] == g1]
                        sub2 = df_c[df_c[group_col] == g2]
                        # Run log rank between these two
                        sub1_data = (sub1[time_col].values, sub1[event_col].values)
                        sub2_data = (sub2[time_col].values, sub2[event_col].values)
                        st, pval, _ = self._log_rank([sub1_data, sub2_data])
                        raw_pvals.append(pval)
                        pairwise_results.append({'group1': g1, 'group2': g2, 'Statistic': st, 'p-value': pval})
                        
                    if p_adj_method != 'none':
                        from statsmodels.stats.multitest import multipletests
                        reject, pvals_corrected, _, _ = multipletests(raw_pvals, method=p_adj_method)
                        for i, res in enumerate(pairwise_results):
                            res['p-adj'] = pvals_corrected[i]
                            res['Significant'] = reject[i]
                    else:
                        for res in pairwise_results:
                            res['p-adj'] = res['p-value']
                            res['Significant'] = res['p-value'] < 0.05
                            
                pairwise_df = pd.DataFrame(pairwise_results)
                if pairwise_df.empty:
                    pairwise_df = pd.DataFrame(columns=['group1', 'group2', 'Statistic', 'p-value', 'p-adj', 'Significant'])
            else:
                km_table = self._km_estimate(
                    df_c[time_col].values, df_c[event_col].values
                )
                lr_df = pd.DataFrame([{
                    'Test': 'Log-Rank', 'Chi-square': float('nan'),
                    'df': float('nan'), 'p-value': float('nan'),
                    'Significant': float('nan'),
                    'Note': 'Add a Group column to enable the log-rank test',
                }])
                pairwise_df = pd.DataFrame(columns=['group1', 'group2', 'Statistic', 'p-value', 'p-adj', 'Significant'])

            self.output_values['km_table'] = TableData(payload=km_table)
            self.output_values['log_rank'] = StatData(payload=lr_df)
            self.output_values['pairwise_stat'] = TableData(payload=pairwise_df)
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error(); return False, str(e)


# ── 6. PCA ─────────────────────────────────────────────────────────────────────

class PCANode(BaseExecutionNode):
    """
    Performs principal component analysis (PCA) for multivariate data exploration.

    Outputs:
    - **transformed** — PC coordinates per sample (connect to ScatterPlotNode for PC1 vs PC2)
    - **loadings** — feature contributions per principal component
    - **variance** — eigenvalues and cumulative variance explained per component

    **Standardize** — when enabled, applies Z-score normalization before decomposition.

    Keywords: PCA, principal component analysis, dimensionality reduction,
              biplot, variance explained, loadings, transformed, multivariate,
              主成分分析, 降維, 主分量, 特徵值, 多變量
    """
    __identifier__ = 'nodes.analysis'
    NODE_NAME = 'PCA'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table', 'table', 'stat']}

    def __init__(self):
        super().__init__()
        self.add_input('in',        color=PORT_COLORS['table'])
        self.add_output('transformed', color=PORT_COLORS['table'])
        self.add_output('loadings', color=PORT_COLORS['table'])
        self.add_output('variance', color=PORT_COLORS['stat'])

        self.add_text_input('n_components',  'N Components (blank=all)',   text='')
        self.add_text_input('sample_id_col', 'Sample ID Column (opt.)',    text='')
        self.add_text_input('feature_cols',  'Features (empty=all num)',   text='')
        self.add_checkbox('standardize', '', text='Standardize (Z-score)', state=True)

    def evaluate(self):
        self.reset_progress()
        try:
            import numpy as np
            import pandas as pd
        except ImportError:
            self.mark_error()
            return False, "numpy and pandas are required for PCA."

        df, err = _get_table_df(self)
        if df is None:
            self.mark_error(); return False, err

        sample_id_col  = str(self.get_property('sample_id_col')  or '').strip()
        n_comp_s       = str(self.get_property('n_components')   or '').strip()
        feature_cols_s = str(self.get_property('feature_cols')   or '').strip()
        standardize    = bool(self.get_property('standardize'))

        try:
            self.set_progress(20)
            if feature_cols_s:
                feature_cols = [c.strip() for c in feature_cols_s.split(',')
                                if c.strip() in df.columns]
            else:
                feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not feature_cols:
                self.mark_error(); return False, "No numeric feature columns found"

            df_feat    = df[feature_cols].dropna()
            X          = df_feat.values.astype(float)

            if standardize:
                # Standardize (Z-score)
                means = np.nanmean(X, axis=0)
                stds = np.nanstd(X, axis=0, ddof=0)
                # Avoid division by zero
                stds[stds == 0] = 1.0
                X = (X - means) / stds
            else:
                # Mean center only (PCA requirement)
                means = np.nanmean(X, axis=0)
                X = X - means

            n_samples, n_features = X.shape
            n_max  = min(n_samples, n_features)
            n_comp = min(int(n_comp_s), n_max) if n_comp_s else n_max

            self.set_progress(40)
            
            # Pure Numpy PCA via Singular Value Decomposition (SVD)
            # This is numerically more stable than eigendecomposition of the covariance matrix
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            
            # Principal axes in feature space (loadings)
            components_ = Vt[:n_comp]
            
            # Eigenvalues (variance)
            explained_variance_ = (S ** 2) / (n_samples - 1)
            
            # Explained variance ratio
            total_var = np.sum((S ** 2) / (n_samples - 1))
            explained_variance_ratio_ = explained_variance_ / total_var
            
            # Transform data to principal component space
            scores = np.dot(X, components_.T)
            
            self.set_progress(70)

            pc_names   = [f'PC{i + 1}' for i in range(n_comp)]
            scores_df  = pd.DataFrame(scores, columns=pc_names)
            
            # Reattach any non-feature columns (e.g. Group, ID)
            non_feature_cols = [c for c in df.columns if c not in feature_cols]
            if sample_id_col and sample_id_col in df.columns and sample_id_col not in non_feature_cols:
                non_feature_cols.insert(0, sample_id_col)
                
            for i, col in enumerate(non_feature_cols):
                scores_df.insert(i, col, df.loc[df_feat.index, col].values)

            loadings_df = pd.DataFrame(
                components_.T, index=feature_cols, columns=pc_names
            )
            loadings_df.index.name = 'Feature'
            loadings_df = loadings_df.reset_index()

            var_pct = explained_variance_ratio_[:n_comp] * 100
            var_df  = pd.DataFrame({
                'Component':              pc_names,
                'Eigenvalue':             explained_variance_[:n_comp].round(6),
                'Variance Explained (%)': var_pct.round(2)
            })
            var_df['Cumulative (%)'] = var_pct.cumsum().round(2)

            self.output_values['transformed'] = TableData(payload=scores_df)
            self.output_values['loadings'] = TableData(payload=loadings_df)
            self.output_values['variance'] = StatData(payload=var_df)
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error(); return False, str(e)
