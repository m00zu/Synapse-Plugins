"""
script_node.py
==============
PythonScriptNode — run custom Python code with dynamic input/output ports.
"""
import pandas as pd
import numpy as np
import NodeGraphQt
from PySide6 import QtWidgets, QtCore, QtGui

from data_models import TableData, ImageData, MaskData, FigureData, StatData
from nodes.base import (
    BaseExecutionNode, PORT_COLORS,
    NodeBaseWidget
)


class _CodeEditorDialog(QtWidgets.QDialog):
    """Popup code editor with syntax highlighting and line numbers."""

    def __init__(self, code: str = '', parent=None):
        super().__init__(parent)
        self.setWindowTitle('Python Script Editor')
        self.resize(700, 500)

        self._editor = QtWidgets.QPlainTextEdit()
        self._editor.setPlainText(code)
        self._editor.setFont(QtGui.QFont('Consolas', 11) if QtGui.QFontDatabase.hasFamily('Consolas')
                             else QtGui.QFont('Courier', 11))
        self._editor.setTabStopDistance(
            QtGui.QFontMetricsF(self._editor.font()).horizontalAdvance(' ') * 4
        )
        self._editor.setStyleSheet(
            'QPlainTextEdit { background: #1e1e1e; color: #d4d4d4; '
            'selection-background-color: #264f78; border: none; }'
        )
        self._editor.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)

        # Help label
        help_text = (
            '<b>Available variables:</b> <code>in_1</code>, <code>in_2</code>, … '
            '(DataFrame / ndarray / value from each input port)<br>'
            '<b>Set outputs:</b> <code>out_1 = …</code>, <code>out_2 = …</code><br>'
            '<b>Pre-imported:</b> <code>pd</code>, <code>np</code>, '
            '<code>scipy</code>, <code>skimage</code>, '
            '<code>cv2</code>, <code>PIL</code>, <code>plt</code><br>'
            '<b>Type wrappers:</b> <code>TableData</code>, <code>ImageData</code>, '
            '<code>MaskData</code>, <code>FigureData</code>, <code>StatData</code> '
            '— use to control output type (e.g. <code>out_1 = MaskData(payload=arr)</code>)'
        )
        help_label = QtWidgets.QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setStyleSheet('QLabel { color: #999; font-size: 11px; padding: 4px; }')

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(help_label)
        layout.addWidget(self._editor, 1)
        layout.addWidget(btn_box)

    def get_code(self) -> str:
        return self._editor.toPlainText()


class _CodePreviewWidget(NodeBaseWidget):
    """Inline code preview on the node card with an 'Edit' button."""

    def __init__(self, parent=None, name='script_code', label=''):
        super().__init__(parent, name=name, label=label)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._preview = QtWidgets.QPlainTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setMaximumHeight(80)
        self._preview.setFont(QtGui.QFont('Consolas', 9) if QtGui.QFontDatabase.hasFamily('Consolas')
                              else QtGui.QFont('Courier', 9))
        self._preview.setStyleSheet(
            'QPlainTextEdit { background: #2a2a2a; color: #aaa; border: 1px solid #444; }'
        )

        self._edit_btn = QtWidgets.QPushButton('Edit Script…')
        self._edit_btn.setFixedHeight(24)
        self._edit_btn.clicked.connect(self._open_editor)

        layout.addWidget(self._preview)
        layout.addWidget(self._edit_btn)
        self.set_custom_widget(container)

    def get_value(self):
        return self._preview.toPlainText()

    def set_value(self, value):
        self._preview.setPlainText(str(value or ''))

    def _open_editor(self):
        parent_win = QtWidgets.QApplication.activeWindow()
        dlg = _CodeEditorDialog(self.get_value(), parent=parent_win)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            code = dlg.get_code()
            self.set_value(code)
            self.on_value_changed(code)


_output_helper = None


def _show_script_output(node_name: str, text: str):
    """Thread-safe: emit signal to show popup on main thread."""
    global _output_helper
    if _output_helper is None:
        # Lazy init — only created when first print() output occurs
        class _Helper(QtCore.QObject):
            _show = QtCore.Signal(str, str)
            def __init__(self):
                super().__init__()
                self._show.connect(self._on_show)
            def _on_show(self, title, text):
                QtWidgets.QMessageBox.information(
                    QtWidgets.QApplication.activeWindow(), title, text)
        _output_helper = _Helper()
    _output_helper._show.emit(f'Python Script — {node_name}', text)


class PythonScriptNode(BaseExecutionNode):
    """
    Run custom Python code with dynamic input and output ports.

    Use this node for operations that no dedicated node covers — custom
    formulas, advanced scipy/skimage functions, string parsing, conditional
    logic, or any one-off data transformation.

    ### Setup

    - **Inputs / Outputs** spinboxes control how many ports the node has.
    - Click **Edit Script…** to open the full code editor (dark theme).
    - The inline preview on the node card shows the current script.
    - `print()` output is shown as a popup after execution.

    ### Variables

    | Variable | Description |
    |----------|-------------|
    | `in_1`, `in_2`, … | Data from each input port (DataFrame, ndarray, or raw value). Unconnected = `None`. |
    | `out_1`, `out_2`, … | Assign results here to send downstream. |
    | `pd` | pandas |
    | `np` | numpy |
    | `scipy` | scipy (use `scipy.stats`, `scipy.ndimage`, etc.) |
    | `skimage` | scikit-image (use `skimage.filters`, etc.) |
    | `cv2` | OpenCV |
    | `PIL` | Pillow |
    | `plt` | matplotlib.pyplot |
    | `set_progress(0-100)` | Update the node's progress bar during long operations |

    You can `import` any additional module installed in your environment.

    ### Output types

    Results are auto-wrapped: DataFrame → TableData, 2D ndarray → ImageData,
    Figure → FigureData, scalar → single-cell TableData.
    To force a type, use: `out_1 = MaskData(payload=arr)` or `ImageData(payload=arr, bit_depth=16)`.

    ### Examples

    **Fold-change** (qPCR) — `df['fold_change'] = 2 ** (-df['ddCt'])`:

    - `df = in_1.copy()`
    - `df['fold_change'] = 2 ** (-df['ddCt'])`
    - `out_1 = df`

    **Column ratio** — `df['ratio'] = df['intensity'] / df['area']`:

    - `df = in_1.copy()`
    - `df['ratio'] = df['intensity'] / df['area']`
    - `out_1 = df`

    **Split by median** (set Outputs to 2):

    - `med = in_1['value'].median()`
    - `out_1 = in_1[in_1['value'] > med]`
    - `out_2 = in_1[in_1['value'] <= med]`

    **Custom scipy test**:

    - `from scipy.stats import mannwhitneyu`
    - `g1 = in_1[in_1['group']=='A']['value']`
    - `u, p = mannwhitneyu(g1, g2)`
    - `out_1 = pd.DataFrame({'U': [u], 'p': [p]})`

    **Image filter**:

    - `from scipy.ndimage import gaussian_filter`
    - `out_1 = gaussian_filter(in_1, sigma=3)`

    Keywords: python, script, code, custom, formula, expression, exec, compute, 自定義, 腳本, 程式, 自訂公式
    """
    __identifier__ = 'nodes.utility'
    NODE_NAME = 'Python Script'
    PORT_SPEC = {'inputs': ['any'], 'outputs': ['any']}

    _UI_PROPS = frozenset({
        'color', 'pos', 'selected', 'name', 'progress',
        'n_inputs', 'n_outputs',
    })

    def __init__(self):
        super().__init__(use_progress=True)
        self.set_port_deletion_allowed(True)

        # Start with 1 input, 1 output
        self.add_input('in_1', color=PORT_COLORS['any'])
        self.add_output('out_1', color=PORT_COLORS['any'])

        # Port count spinboxes
        self._add_int_spinbox('n_inputs',  'Inputs',  value=1, min_val=1, max_val=8, step=1)
        self._add_int_spinbox('n_outputs', 'Outputs', value=1, min_val=1, max_val=8, step=1)

        # Code editor (add_custom_widget creates the property automatically)
        H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value
        code_w = _CodePreviewWidget(self.view, name='script_code', label='')
        code_w.set_value('# Write your code here\nout_1 = in_1')
        self.add_custom_widget(code_w, widget_type=H, tab='Code')

        self._current_n_in = 1
        self._current_n_out = 1

    def set_property(self, name, value, push_undo=True):
        super().set_property(name, value, push_undo)
        if name == 'n_inputs':
            self._sync_ports('input', int(value or 1))
        elif name == 'n_outputs':
            self._sync_ports('output', int(value or 1))

    def _sync_ports(self, direction, target):
        """Add or remove ports to match the target count."""
        if direction == 'input':
            current = self._current_n_in
            for i in range(current + 1, target + 1):
                pname = f'in_{i}'
                if pname not in self.inputs():
                    self.add_input(pname, color=PORT_COLORS['any'])
            for i in range(current, target, -1):
                pname = f'in_{i}'
                if pname in self.inputs():
                    port = self.get_input(pname)
                    for cp in port.connected_ports():
                        port.disconnect_from(cp)
                    self.delete_input(port)
            self._current_n_in = target
        else:
            current = self._current_n_out
            for i in range(current + 1, target + 1):
                pname = f'out_{i}'
                if pname not in self.outputs():
                    self.add_output(pname, color=PORT_COLORS['any'])
            for i in range(current, target, -1):
                pname = f'out_{i}'
                if pname in self.outputs():
                    port = self.get_output(pname)
                    for cp in port.connected_ports():
                        port.disconnect_from(cp)
                    self.delete_output(port)
            self._current_n_out = target

    def _unwrap(self, data):
        """Unwrap NodeData to its payload (DataFrame, ndarray, etc.)."""
        if data is None:
            return None
        if hasattr(data, 'df'):
            return data.df.copy()
        if hasattr(data, 'payload'):
            p = data.payload
            if isinstance(p, np.ndarray):
                return p.copy()
            return p
        return data

    def _wrap(self, value):
        """Wrap a raw value into the appropriate NodeData type."""
        if value is None:
            return None
        if isinstance(value, (TableData, ImageData, MaskData, FigureData, StatData)):
            return value  # already wrapped
        if isinstance(value, pd.DataFrame):
            return TableData(payload=value)
        if isinstance(value, pd.Series):
            return TableData(payload=value.to_frame())
        if isinstance(value, np.ndarray):
            if value.ndim >= 2:
                return ImageData(payload=value)
            # 1D array → table
            return TableData(payload=pd.DataFrame({'value': value}))
        # Scalar or other → single-cell table
        return TableData(payload=pd.DataFrame({'result': [value]}))

    def evaluate(self):
        self.reset_progress()
        code = str(self.get_property('script_code') or '').strip()
        if not code:
            self.mark_error()
            return False, 'No script code provided.'

        # Pre-import common libraries (lazy — skip if not installed)
        # Data type wrappers — users can explicitly set output types:
        #   out_1 = TableData(payload=df)
        #   out_1 = ImageData(payload=arr)
        #   out_1 = MaskData(payload=bool_arr)
        #   out_1 = FigureData(payload=fig)
        env = {
            'pd': pd, 'np': np,
            'TableData': TableData, 'ImageData': ImageData,
            'MaskData': MaskData, 'FigureData': FigureData,
            'StatData': StatData,
            'set_progress': self.set_progress,
        }
        for mod_name, alias in [
            ('scipy', 'scipy'),
            ('skimage', 'skimage'),
            ('cv2', 'cv2'),
            ('PIL', 'PIL'),
            ('matplotlib', 'matplotlib'),
            ('matplotlib.pyplot', 'plt'),
        ]:
            try:
                env[alias] = __import__(mod_name, fromlist=[''])
            except ImportError:
                pass

        # Gather inputs
        for port_name, port_obj in self.inputs().items():
            if port_obj.connected_ports():
                cp = port_obj.connected_ports()[0]
                raw = cp.node().output_values.get(cp.name())
                env[port_name] = self._unwrap(raw)
            else:
                env[port_name] = None

        # Pre-declare output variables as None
        for port_name in self.outputs():
            env[port_name] = None

        # Execute — capture print() output via StringIO
        import io, contextlib
        _stdout_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(_stdout_capture):
                exec(code, {'__builtins__': __builtins__}, env)
        except Exception as e:
            printed = _stdout_capture.getvalue().strip()
            err_msg = f'Script error: {e}'
            if printed:
                err_msg += f'\n\nPrint output:\n{printed}'
            self.mark_error()
            return False, err_msg
        self._print_output = _stdout_capture.getvalue().strip()

        # Collect outputs and check for mismatched port names
        any_output = False
        output_names = set(self.outputs().keys())
        warnings = []
        for port_name in output_names:
            val = env.get(port_name)
            if val is not None:
                self.output_values[port_name] = self._wrap(val)
                any_output = True

        # Detect assignments to non-existent output ports (e.g. out_3 when only 1 output)
        import re
        assigned_outs = set(re.findall(r'\b(out_\d+)\b', code)) - output_names
        if assigned_outs:
            warnings.append(
                f"Assigned to {', '.join(sorted(assigned_outs))} but port(s) don't exist. "
                f"Increase Outputs to {max(int(n.split('_')[1]) for n in assigned_outs)}."
            )

        if self._print_output:
            # Store for retrieval; show popup via main-thread signal
            _show_script_output(self.name(), self._print_output)

        if not any_output:
            self.mark_error()
            msg = 'Script produced no output. Assign to out_1, out_2, etc.'
            if warnings:
                msg += '\n' + '\n'.join(warnings)
            return False, msg

        self.set_progress(100)
        self.mark_clean()
        if warnings:
            return True, '\n'.join(warnings)
        return True, None
