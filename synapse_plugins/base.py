"""
nodes/base.py
=============
Shared widgets, signals, PORT_COLORS, and BaseExecutionNode.
All other node modules import from here.
"""
from concurrent.futures import ProcessPoolExecutor
import threading
import NodeGraphQt
from PySide6 import QtCore, QtWidgets, QtGui
from PIL import Image
import pandas as pd
import numpy as np
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget, NodeLineEdit, NodeCheckBox
from NodeGraphQt.widgets.dialogs import FileDialog
from pathlib import Path
from ..data_models import TableData, ImageData, FigureData, ConfocalDatasetData
import traceback
from ..i18n import tr

# ── Port type color scheme ──────────────────────────────────────────────────
# All colors are (R, G, B) tuples, accepted by add_input/add_output color arg.
PORT_COLORS = {
    'table':    (52,  152, 219),  # Blue       – pandas DataFrame / TableData
    'stat':     (65,  105, 225),  # RoyalBlue  – StatData
    'image':    (46,  204, 113),  # Green      – PIL Image / ImageData
    'mask':     (28, 125, 72),    # Forest Green – Boolean mask ImageData (clearly distinct)
    'skeleton': (180, 230, 100),  # Yellow-green  – SkeletonData (thinned skeleton mask)
    'label':       (160, 220,  40),  # Chartreuse – LabelData (integer label array)
    'label_image': (160, 220,  40),  # Chartreuse alias – port name used in PORT_SPEC
    'figure':   (155,  89, 182),  # Purple     – matplotlib Figure / FigureData
    'confocal': (230, 126,  34),  # Orange     – ConfocalDatasetData
    'path':     (149, 165, 166),  # Grey       – file / folder path string
    'collection': (230, 180, 50), # Gold       – CollectionData (named bundle)
    'model':    (255, 140,  66),  # Coral/Orange – fitted model (regression, ML)
    'any':      ( 95, 106, 106),  # Dark grey  – generic / unknown type
}

class ColorPickerButtonWidget(QtWidgets.QWidget):
    """
    Inner PySide6 widget containing the label and color picker button.
    """
    value_changed = QtCore.Signal(list)

    def __init__(self, parent=None):
        super(ColorPickerButtonWidget, self).__init__(parent)
        self._color = QtGui.QColor(105, 105, 105, 255) # Default dimgray
        
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn = QtWidgets.QPushButton()
        self.btn.setMinimumHeight(24)
        self.btn.clicked.connect(self._on_btn_clicked)
        self._update_btn_style()
        
        layout.addWidget(self.btn)

    def _update_btn_style(self):
        self.btn.setStyleSheet(
            f"background-color: {self._color.name()}; "
            f"border: 1px solid #222; border-radius: 3px;"
        )

    def _on_btn_clicked(self):
        parent = QtWidgets.QApplication.activeWindow()
        color = QtWidgets.QColorDialog.getColor(
            self._color, parent, tr("Select Dot Color"),
        )
        if color.isValid():
            self._color = color
            self._update_btn_style()
            # NodeGraphQt requires basic python types for serialization
            self.value_changed.emit([color.red(), color.green(), color.blue(), color.alpha()])
    
    def get_value(self):
        return [self._color.red(), self._color.green(), self._color.blue(), self._color.alpha()]

    def set_value(self, value):
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            r, g, b = value[:3]
            a = value[3] if len(value) > 3 else 255
            new_color = QtGui.QColor(int(r), int(g), int(b), int(a))
            
            # GUARD: Only update and emit if the color actually changed
            if new_color != self._color:
                self._color = new_color
                self._update_btn_style()
                self.value_changed.emit(self.get_value())


class NodeColorPickerWidget(NodeBaseWidget):
    """
    NodeGraphQt wrapper for the ColorPickerButtonWidget to embed in nodes.
    """
    def __init__(self, parent=None, name='color', label=''):
        super(NodeColorPickerWidget, self).__init__(parent, name, label)
        self.set_custom_widget(ColorPickerButtonWidget())
        
        # NodeGraphQt requires the property name as the first argument to value_changed
        self.get_custom_widget().value_changed.connect(
            lambda val: self.value_changed.emit(self.get_name(), val)
        )

    def get_value(self):
        return self.get_custom_widget().get_value()

    def set_value(self, value):
        self.get_custom_widget().set_value(value)

class NodeIntSpinBoxWidget(NodeBaseWidget):
    """A custom node widget with an integer spinbox."""
    def __init__(self, parent=None, name='', label='', value=0, min_val=0, max_val=99999, step=1):
        super(NodeIntSpinBoxWidget, self).__init__(parent, name, label)
        self._spin = QtWidgets.QSpinBox()
        self._spin.setMinimum(min_val)
        self._spin.setMaximum(max_val)
        self._spin.setSingleStep(step)
        self._spin.setValue(int(value))
        self._spin.setMinimumWidth(80)
        self._spin.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self._spin.valueChanged.connect(lambda v: self.value_changed.emit(self.get_name(), v))
        self.set_custom_widget(self._spin)

    def get_value(self):
        return self._spin.value()

    def set_value(self, value):
        self._spin.blockSignals(True)
        try:
            self._spin.setValue(int(value))
        except (ValueError, TypeError):
            pass
        self._spin.blockSignals(False)


class NodeFloatSpinBoxWidget(NodeBaseWidget):
    """A custom node widget with a floating-point spinbox."""
    def __init__(self, parent=None, name='', label='', value=0.0, min_val=0.0,
                 max_val=99999.0, step=0.1, decimals=3):
        super(NodeFloatSpinBoxWidget, self).__init__(parent, name, label)
        self._spin = QtWidgets.QDoubleSpinBox()
        self._spin.setMinimum(min_val)
        self._spin.setMaximum(max_val)
        self._spin.setValue(float(value))
        self._spin.setSingleStep(step)
        self._spin.setDecimals(decimals)
        self._spin.setMinimumWidth(80)
        self._spin.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self._spin.valueChanged.connect(lambda v: self.value_changed.emit(self.get_name(), v))
        self.set_custom_widget(self._spin)

    def get_value(self):
        return self._spin.value()

    def set_value(self, value):
        self._spin.blockSignals(True)
        try:
            self._spin.setValue(float(value))
        except (ValueError, TypeError):
            pass
        self._spin.blockSignals(False)


class NodeColumnSelectorWidget(NodeBaseWidget):
    """Text input with a dropdown button that shows available DataFrame columns.

    Modes:
        'single'  — clicking a column replaces the text field
        'multi'   — clicking a column toggles it in a comma-separated list

    Thread safety: set_columns() only stores the list (no Qt calls).
    The QMenu is rebuilt lazily on the main thread when the user clicks ▼.
    """

    def __init__(self, parent=None, name='', label='', text='', mode='single'):
        super().__init__(parent, name, label)
        self._mode = mode
        self._columns: list[str] = []

        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._edit = QtWidgets.QLineEdit(text)
        self._edit.setMinimumWidth(60)
        self._edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._edit.editingFinished.connect(self._on_text_changed)
        layout.addWidget(self._edit)

        self._btn = QtWidgets.QToolButton()
        self._btn.setText('▼')
        self._btn.setFixedWidth(22)
        self._btn.clicked.connect(self._show_menu)
        layout.addWidget(self._btn)

        self.set_custom_widget(container)

    def get_value(self):
        return self._edit.text()

    def set_value(self, value):
        self._edit.blockSignals(True)
        self._edit.setText(str(value) if value else '')
        self._edit.blockSignals(False)

    def set_columns(self, columns: list[str]):
        """Store column names (thread-safe — no Qt widget calls)."""
        self._columns = list(columns)

    def _show_menu(self):
        """Build and show the menu on demand (always on main thread)."""
        menu = QtWidgets.QMenu(QtWidgets.QApplication.activeWindow())
        for col in self._columns:
            action = menu.addAction(col)
            action.triggered.connect(lambda checked, c=col: self._on_column_clicked(c))
        if not self._columns:
            action = menu.addAction("(run graph first to populate)")
            action.setEnabled(False)
        menu.exec(QtGui.QCursor.pos())

    def _on_column_clicked(self, col: str):
        if self._mode == 'single':
            self._edit.setText(col)
        else:
            current = [c.strip() for c in self._edit.text().split(',') if c.strip()]
            if col in current:
                current.remove(col)
            else:
                current.append(col)
            self._edit.setText(','.join(current))
        self._on_text_changed()

    def _on_text_changed(self):
        self.value_changed.emit(self.get_name(), self.get_value())


class NodeChannelSelectorWidget(NodeBaseWidget):
    """Dropdown toggle selector for image channels (1-4), max 3 selections.

    Clicking a channel in the dropdown toggles it. The text field is read-only
    and shows the current selection as comma-separated numbers.
    Defaults to '1,2,3' if empty.
    """

    def __init__(self, parent=None, name='', label='', text='1,2,3', max_channels=4):
        super().__init__(parent, name, label)
        self._max_ch = max_channels

        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._edit = QtWidgets.QLineEdit(text)
        self._edit.setReadOnly(True)
        self._edit.setMinimumWidth(60)
        self._edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(self._edit)

        self._btn = QtWidgets.QToolButton()
        self._btn.setText('▼')
        self._btn.setFixedWidth(22)
        self._btn.clicked.connect(self._show_menu)
        layout.addWidget(self._btn)

        self.set_custom_widget(container)

    def get_value(self):
        return self._edit.text()

    def set_value(self, value):
        self._edit.blockSignals(True)
        self._edit.setText(str(value) if value else '')
        self._edit.blockSignals(False)

    def _current_selection(self):
        return [c.strip() for c in self._edit.text().split(',') if c.strip()]

    def _show_menu(self):
        menu = QtWidgets.QMenu(QtWidgets.QApplication.activeWindow())
        current = self._current_selection()
        for ch in range(1, self._max_ch + 1):
            ch_str = str(ch)
            action = menu.addAction(f'Ch{ch}')
            action.setCheckable(True)
            action.setChecked(ch_str in current)
            action.triggered.connect(lambda checked, c=ch_str: self._on_channel_toggled(c))
        menu.addSeparator()
        pad_action = menu.addAction('Pad (black)')
        pad_action.triggered.connect(self._on_pad_clicked)
        menu.addSeparator()
        menu.addAction('Clear').triggered.connect(self._clear)
        menu.exec(QtGui.QCursor.pos())

    def _on_channel_toggled(self, ch: str):
        current = self._current_selection()
        if ch in current:
            current.remove(ch)
        else:
            if len(current) >= 3:
                return
            current.append(ch)
        self._edit.setText(','.join(current))
        self.value_changed.emit(self.get_name(), self.get_value())

    def _on_pad_clicked(self):
        """Append a pad (0) channel. Always appends, allows duplicates."""
        current = self._current_selection()
        if len(current) >= 3:
            return
        current.append('0')
        self._edit.setText(','.join(current))
        self.value_changed.emit(self.get_name(), self.get_value())

    def _clear(self):
        self._edit.setText('')
        self.value_changed.emit(self.get_name(), self.get_value())


class NodeRowWidget(NodeBaseWidget):
    """Compact horizontal row of labeled spinboxes, each mapped to its own property."""

    def __init__(self, parent=None, name='', label='', fields=None):
        """
        Args:
            fields: list of dicts, each with:
                name (str): property name
                label (str): short label
                type ('int' | 'float'): spinbox type
                value: default value
                min_val, max_val, step, decimals (optional)
        """
        super().__init__(parent, name, label)
        fields = fields or []
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self._fields = {}  # name → spin
        for f in fields:
            lbl = QtWidgets.QLabel(f['label'])
            lbl.setStyleSheet('color: #999; font-size: 10px;')
            layout.addWidget(lbl)

            if f.get('type') == 'float':
                spin = QtWidgets.QDoubleSpinBox()
                spin.setDecimals(f.get('decimals', 1))
            else:
                spin = QtWidgets.QSpinBox()
            spin.setMinimum(f.get('min_val', 0))
            spin.setMaximum(f.get('max_val', 99999))
            spin.setSingleStep(f.get('step', 1))
            spin.setValue(f['value'])
            spin.setMinimumWidth(50)
            fname = f['name']
            spin.valueChanged.connect(
                lambda v, n=fname: self.value_changed.emit(n, v))
            layout.addWidget(spin)
            self._fields[fname] = spin

        self.set_custom_widget(container)

    def get_value(self):
        return {n: s.value() for n, s in self._fields.items()}

    def set_value(self, value):
        if isinstance(value, dict):
            for n, v in value.items():
                if n in self._fields:
                    self._fields[n].blockSignals(True)
                    try:
                        self._fields[n].setValue(v)
                    except (ValueError, TypeError):
                        pass
                    self._fields[n].blockSignals(False)

    def set_field(self, name, value):
        if name in self._fields:
            self._fields[name].blockSignals(True)
            try:
                self._fields[name].setValue(value)
            except (ValueError, TypeError):
                pass
            self._fields[name].blockSignals(False)


class NodeVec3Widget(NodeBaseWidget):
    """Compact X / Y / Z triple-spinbox widget for 3D coordinates or sizes."""

    def __init__(self, parent=None, name='', label='',
                 value=(0.0, 0.0, 0.0), min_val=-999.0, max_val=999.0,
                 step=1.0, decimals=3, axis_labels=('X', 'Y', 'Z')):
        super().__init__(parent, name, label)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._spins = []
        for i, axis in enumerate(axis_labels):
            lbl = QtWidgets.QLabel(axis)
            lbl.setFixedWidth(12)
            lbl.setStyleSheet('color: #999; font-size: 10px;')
            layout.addWidget(lbl)

            spin = QtWidgets.QDoubleSpinBox()
            spin.setMinimum(min_val)
            spin.setMaximum(max_val)
            spin.setValue(float(value[i]))
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setMinimumWidth(55)
            spin.valueChanged.connect(
                lambda _v: self.value_changed.emit(self.get_name(), self.get_value()))
            layout.addWidget(spin)
            self._spins.append(spin)

        self.set_custom_widget(container)

    def get_value(self):
        return [s.value() for s in self._spins]

    def set_value(self, value):
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            for i, s in enumerate(self._spins):
                s.blockSignals(True)
                try:
                    s.setValue(float(value[i]))
                except (ValueError, TypeError):
                    pass
                s.blockSignals(False)

    def set_axis(self, index, value):
        """Set a single axis (0=X, 1=Y, 2=Z) without triggering signals."""
        if 0 <= index < len(self._spins):
            self._spins[index].blockSignals(True)
            self._spins[index].setValue(float(value))
            self._spins[index].blockSignals(False)


class NodeSignals(QtCore.QObject):
    """
    Signal relay to bridge communication between background evaluation threads
    and the main UI thread.
    """
    progress_updated = QtCore.Signal(str, int)  # node_id, progress_value
    status_updated = QtCore.Signal(str, str)    # node_id, status_type ('clean', 'dirty', 'error')
    display_requested = QtCore.Signal(str, object) # node_id, data
    
# Global signal relay instance
NODE_SIGNALS = NodeSignals()

class NodeFileSelector(NodeBaseWidget):
    """
    A custom node widget that provides a text input and a browse button
    directly on the node surface.
    """
    def __init__(self, parent=None, name='', label='', file_dir=str(Path.home()), ext_filter='*'):
        super(NodeFileSelector, self).__init__(parent, name, label)
        
        base_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(base_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        self._ledit = QtWidgets.QLineEdit()
        self._ledit.setPlaceholderText(tr("Select file..."))
        self._ledit.setMinimumWidth(200)
        self._ledit.editingFinished.connect(self.on_value_changed)

        # Use a standard Qt icon for the browse button
        icon = base_widget.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon)
        self._btn = QtWidgets.QPushButton()
        self._btn.setIcon(icon)
        self._btn.setFixedSize(34, 24)
        self._btn.setIconSize(QtCore.QSize(16, 16))
        self._btn.setProperty('compact', True)
        self._btn.setProperty('pathButton', True)
        self._btn.clicked.connect(self._on_select_file)

        layout.addWidget(self._ledit)
        layout.addWidget(self._btn)
        layout.addStretch()

        self.set_custom_widget(base_widget)
        self._file_directory = file_dir
        self._ext = ext_filter

    def _on_select_file(self):
        file_path = FileDialog.getOpenFileName(None, tr("Select File"), file_dir=self._file_directory, ext_filter=self._ext)
        file = file_path[0] or None
        if file:
            self.set_value(file)
            from pathlib import Path
            self._file_directory = str(Path(file).parent)

    def get_value(self):
        return self._ledit.text()

    def set_value(self, value):
        if value != self.get_value():
            self._ledit.setText(str(value))
            self.on_value_changed()

    def set_custom_widget(self, widget):
        super(NodeFileSelector, self).set_custom_widget(widget)


class NodeFileSaver(NodeBaseWidget):
    """
    A custom node widget that provides a text input and a save button
    directly on the node surface.
    """
    def __init__(self, parent=None, name='', label='', file_dir=str(Path.home()), ext_filter='*'):
        super(NodeFileSaver, self).__init__(parent, name, label)
        
        base_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(base_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        self._ledit = QtWidgets.QLineEdit()
        self._ledit.setPlaceholderText(tr("Select file..."))
        self._ledit.setMinimumWidth(200)
        self._ledit.editingFinished.connect(self.on_value_changed)

        # Use a standard Qt icon for the browse button
        icon = base_widget.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon)
        self._btn = QtWidgets.QPushButton()
        self._btn.setIcon(icon)
        self._btn.setFixedSize(34, 24)
        self._btn.setIconSize(QtCore.QSize(16, 16))
        self._btn.setProperty('compact', True)
        self._btn.setProperty('pathButton', True)
        self._btn.clicked.connect(self._on_select_file)

        layout.addWidget(self._ledit)
        layout.addWidget(self._btn)
        layout.addStretch()

        self.set_custom_widget(base_widget)
        self._file_directory = file_dir
        self._ext = ext_filter

    def _on_select_file(self):
        file_path = FileDialog.getSaveFileName(None, tr("Select File"), file_dir=self._file_directory, ext_filter=self._ext)
        file = file_path[0] or None
        if file:
            self.set_value(file)
            from pathlib import Path
            self._file_directory = str(Path(file).parent)

    def get_value(self):
        return self._ledit.text()

    def set_value(self, value):
        if value != self.get_value():
            self._ledit.setText(str(value))
            self.on_value_changed()

    def set_custom_widget(self, widget):
        super(NodeFileSaver, self).set_custom_widget(widget)

class NodeDirSelector(NodeBaseWidget):
    """
    A custom node widget that provides a text input and a browse button
    for selecting directories directly on the node surface.
    """
    def __init__(self, parent=None, name='', label='', start_dir=str(Path.home())):
        super(NodeDirSelector, self).__init__(parent, name, label)
        
        base_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(base_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        self._ledit = QtWidgets.QLineEdit()
        self._ledit.setPlaceholderText(tr("Select directory..."))
        self._ledit.setMinimumWidth(200)
        self._ledit.editingFinished.connect(self.on_value_changed)
        
        icon = base_widget.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirIcon)
        self._btn = QtWidgets.QPushButton()
        self._btn.setIcon(icon)
        self._btn.setFixedSize(34, 24)
        self._btn.setIconSize(QtCore.QSize(16, 16))
        self._btn.setProperty('compact', True)
        self._btn.setProperty('pathButton', True)
        self._btn.clicked.connect(self._on_select_dir)
        
        layout.addWidget(self._ledit)
        layout.addWidget(self._btn)
        layout.addStretch()
        
        self.set_custom_widget(base_widget)
        self._start_directory = start_dir

    def _on_select_dir(self):
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(None, tr("Select Directory"), self._start_directory)
        if dir_path:
            self.set_value(dir_path)
            self._start_directory = dir_path

    def get_value(self):
        return self._ledit.text()

    def set_value(self, value):
        if value != self.get_value():
            self._ledit.setText(str(value))
            self.on_value_changed()

    def set_custom_widget(self, widget):
        super(NodeDirSelector, self).set_custom_widget(widget)
        # Install event filter to deselect node on focus/click
        self._ledit.installEventFilter(self)
        self._btn.installEventFilter(self)


class NodeProgressBar(NodeBaseWidget):
    """
    A custom progress bar widget for nodes.
    """
    def __init__(self, parent=None):
        super(NodeProgressBar, self).__init__(parent, name='progress', label=tr('Progress'))
        
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFixedHeight(12)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #222;
                text-align: center;
                color: white;
                font-size: 7pt;
            }
            QProgressBar::chunk {
                background-color: #2e7d32;
                width: 1px;
            }
        """)
        layout.addWidget(self._progress_bar)
        self.set_custom_widget(container)

    def get_value(self):
        return self._progress_bar.value()

    def set_value(self, value):
        self._progress_bar.setValue(int(value))


class NodeTableWidget(NodeBaseWidget):
    """
    A custom node widget that displays a pandas DataFrame directly on the node.
    """
    def __init__(self, parent=None):
        super(NodeTableWidget, self).__init__(parent, name='table_view', label='')
        
        self._table = QtWidgets.QTableWidget()
        self._table.setMinimumSize(400, 300)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        
        # Style the table for dark mode consistency
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: #222;
                color: #ddd;
                gridline-color: #444;
                border: 1px solid #555;
            }
            QHeaderView::section {
                background-color: #333;
                color: #fff;
                padding: 4px;
                border: 1px solid #444;
            }
        """)
        
        self.set_custom_widget(self._table)

    def set_value(self, df):
        if not isinstance(df, pd.DataFrame):
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            return
            
        # Dynamically calculate width based on columns
        num_cols = len(df.columns)
        base_width = 400
        calc_width = min(base_width + (num_cols * 30), 800) # Max width of 1200
        self._table.setMinimumWidth(calc_width)

        self._table.setRowCount(df.shape[0])
        self._table.setColumnCount(df.shape[1])
        self._table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        self._table.setVerticalHeaderLabels([str(i) for i in df.index])
        
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                val = df.iat[row, col]
                # Round floats for display
                if isinstance(val, (float, np.floating)):
                    if 0 < abs(val) < 0.0001:
                        display_text = "< 0.0001" if val > 0 else "> -0.0001"
                    else:
                        display_text = f"{val:.4f}"
                else:
                    display_text = str(val)
                    
                item = QtWidgets.QTableWidgetItem(display_text)
                self._table.setItem(row, col, item)
                
        # Force proxy widget to update its size
        if self.widget():
            self.widget().adjustSize()

    def get_value(self):
        return None  # Display only


class NodeImageWidget(NodeBaseWidget):
    """
    A custom node widget that displays an image or figure directly on the node.
    The display area auto-sizes to match the input's aspect ratio, capped at a
    maximum width/height so changing figsize in the Figure Editor does not cause
    the node to grow unboundedly ("explode").
    """
    _DISPLAY_MAX_W = 520
    _DISPLAY_MAX_H = 600
    _DISPLAY_MIN_W = 200
    _DISPLAY_MIN_H = 150
    _DISPLAY_DEFAULT_W = 500
    _DISPLAY_DEFAULT_H = 400

    def __init__(self, parent=None):
        super(NodeImageWidget, self).__init__(parent, name='image_view', label='')

        self._layout_widget = QtWidgets.QWidget()
        self._layout_widget.setFixedSize(self._DISPLAY_DEFAULT_W, self._DISPLAY_DEFAULT_H)
        self._layout = QtWidgets.QVBoxLayout(self._layout_widget)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QtWidgets.QStackedWidget()

        self._image_label = QtWidgets.QLabel()
        self._image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background-color: #111; border: 1px solid #444;")

        from PySide6.QtSvgWidgets import QSvgWidget
        self._svg_widget = QSvgWidget()
        self._svg_widget.setStyleSheet("background-color: #111; border: 1px solid #444;")

        self._stack.addWidget(self._image_label)
        self._stack.addWidget(self._svg_widget)

        self._layout.addWidget(self._stack)
        self.set_custom_widget(self._layout_widget)

    def _compute_display_size(self, w, h):
        """Return (dw, dh) that fits w×h within the display max, preserving aspect ratio."""
        if w <= 0 or h <= 0:
            return self._DISPLAY_DEFAULT_W, self._DISPLAY_DEFAULT_H
        ar = w / h
        dw = self._DISPLAY_MAX_W
        dh = int(dw / ar)
        if dh > self._DISPLAY_MAX_H:
            dh = self._DISPLAY_MAX_H
            dw = int(dh * ar)
        dw = max(dw, self._DISPLAY_MIN_W)
        dh = max(dh, self._DISPLAY_MIN_H)
        return dw, dh

    def set_value(self, data):
        import matplotlib.figure
        import tempfile
        import os
        from PIL import Image
        import numpy as np

        if isinstance(data, matplotlib.figure.Figure):
            # Use figsize directly — stable dimensions across runs.
            # No bbox_inches='tight': tight_layout() is called by the caller,
            # so content fits within the figure bounds without it.
            w_in, h_in = data.get_size_inches()
            dw, dh = self._compute_display_size(w_in, h_in)
            self._layout_widget.setFixedSize(dw, dh)

            fd, path = tempfile.mkstemp(suffix=".svg")
            os.close(fd)
            try:
                data.savefig(path, format='svg')
                self._svg_widget.load(path)
            finally:
                if os.path.exists(path):
                    os.remove(path)

            renderer = self._svg_widget.renderer()
            if renderer.isValid():
                renderer.setAspectRatioMode(QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                default_size = renderer.defaultSize()
                view_box = QtCore.QRectF(0, 0, default_size.width(), default_size.height())
                renderer.setViewBox(view_box)

            self._stack.setCurrentWidget(self._svg_widget)

        elif isinstance(data, bytes):
            # Raw SVG bytes — load directly into the QSvgWidget
            self._layout_widget.setFixedSize(self._DISPLAY_DEFAULT_W, self._DISPLAY_DEFAULT_H)
            self._svg_widget.load(QtCore.QByteArray(data))
            renderer = self._svg_widget.renderer()
            if renderer.isValid():
                renderer.setAspectRatioMode(QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                default_size = renderer.defaultSize()
                view_box = QtCore.QRectF(0, 0, default_size.width(), default_size.height())
                renderer.setViewBox(view_box)
            self._stack.setCurrentWidget(self._svg_widget)

        elif isinstance(data, np.ndarray):
            # Numpy array (multi-bit-depth support)
            from data_models import array_to_qpixmap
            if data.ndim == 2:
                h, w = data.shape
            else:
                h, w = data.shape[:2]
            dw, dh = self._compute_display_size(w, h)
            self._layout_widget.setFixedSize(dw, dh)

            pixmap = array_to_qpixmap(data)
            scaled_pixmap = pixmap.scaled(
                dw * 2, dh * 2,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
            scaled_pixmap.setDevicePixelRatio(2.0)
            self._image_label.setPixmap(scaled_pixmap)
            self._stack.setCurrentWidget(self._image_label)

        elif isinstance(data, Image.Image):
            # PIL Image (backward compat)
            dw, dh = self._compute_display_size(data.width, data.height)
            self._layout_widget.setFixedSize(dw, dh)

            if data.mode not in ('RGB', 'RGBA'):
                data = data.convert('RGBA')
            from PIL.ImageQt import ImageQt
            qimage = ImageQt(data)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(
                dw * 2, dh * 2,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
            scaled_pixmap.setDevicePixelRatio(2.0)
            self._image_label.setPixmap(scaled_pixmap)
            self._stack.setCurrentWidget(self._image_label)

        else:
            self._layout_widget.setFixedSize(self._DISPLAY_DEFAULT_W, self._DISPLAY_DEFAULT_H)
            self._image_label.clear()
            self._image_label.setText(tr("No Preview"))
            self._stack.setCurrentWidget(self._image_label)

        # Force proxy widget to update its geometry so the node redraws at the new size
        if self.widget():
            self.widget().adjustSize()

    def get_value(self):
        return None  # Display only


class NodeToolBoxWidget(NodeBaseWidget):
    """
    A custom node widget that embeds a QToolBox to organize other widgets into collapsible sections.
    """
    def __init__(self, parent=None, name='', label=''):
        super(NodeToolBoxWidget, self).__init__(parent, name, label)
        
        self._toolbox = QtWidgets.QToolBox()
        self._toolbox.setMinimumWidth(280)
        self._toolbox.setMinimumHeight(240)
        
        # Apply theme-consistent stylesheet
        self._toolbox.setStyleSheet("""
            QToolBox {
                background-color: #333;
                border: 1px solid #222;
            }
            QToolBox::tab {
                background: #2b2b2b;
                color: #ddd;
                border: 1px solid #1a1a1a;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
                padding: 4px;
                margin-top: 1px;
            }
            QToolBox::tab:selected {
                background: #444;
                color: #fff;
                border-bottom: 2px solid #2e7d32;
            }
            QWidget#toolbox_page {
                background-color: #333;
            }
            QLabel {
                padding-top: 5px;
            }
        """)
        
        self.set_custom_widget(self._toolbox)
        
    def add_widget_to_page(self, page_name, widget):
        """
        Adds a widget to a specific page in the toolbox. 
        Creates the page if it doesn't exist.
        """
        page = None
        for i in range(self._toolbox.count()):
            if self._toolbox.itemText(i) == page_name:
                page = self._toolbox.widget(i)
                break
                
        if page is None:
            page = QtWidgets.QWidget()
            page.setObjectName("toolbox_page")
            layout = QtWidgets.QVBoxLayout(page)
            layout.setContentsMargins(5, 5, 5, 5)
            layout.setSpacing(2)
            layout.addStretch()
            self._toolbox.addItem(page, page_name)
            
        layout = page.layout()
        # Insert before the stretch
        layout.insertWidget(layout.count() - 1, widget)

    def get_value(self):
        return None

    def set_value(self, value):
        pass


class NodeListWidget(NodeBaseWidget):
    """
    A custom node widget that provides a draggable list of items for reordering.
    """
    def __init__(self, parent=None, name='', label='', items=None):
        super(NodeListWidget, self).__init__(parent, name, label)
        
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.setMinimumHeight(100)
        
        # Connect to the model's row moved signal to detect drag-and-drop reorders
        self.list_widget.model().rowsMoved.connect(self._on_rows_moved)
        
        if items:
            self.list_widget.addItems(items)
            
        self.set_custom_widget(self.list_widget)
        
    def _on_rows_moved(self, parent, start, end, destination, row):
        # Emit the new order whenever a drag-and-drop occurs
        self.on_value_changed()
        
    def get_value(self):
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count())]

    def set_value(self, items):
        if not items:
            items = []
        if items != self.get_value():
            self.list_widget.clear()
            self.list_widget.addItems(items)
            self.on_value_changed()

    def add_items(self, items):
        """Append items to the list if they are not already present."""
        current = set(self.get_value())
        for item in items:
            if item not in current:
                self.list_widget.addItem(item)
                
    def clear(self):
        self.list_widget.clear()
        self.on_value_changed()


class BaseExecutionNode(NodeGraphQt.BaseNode):
    """
    Base node with optional progress bar and dirty-state tracking.
    """

    # Class-level cancellation event — shared by all nodes, set by workers.
    _cancel_event = threading.Event()

    @classmethod
    def request_cancel(cls):
        """Signal all running nodes to stop at the next check point."""
        cls._cancel_event.set()

    @classmethod
    def clear_cancel(cls):
        """Reset the cancellation flag (call before starting execution)."""
        cls._cancel_event.clear()

    @property
    def cancel_requested(self) -> bool:
        """True if the user has clicked Stop."""
        return type(self)._cancel_event.is_set()

    def __init__(self, use_progress=True):
        super(BaseExecutionNode, self).__init__()
        if use_progress:
            self._progress_widget = NodeProgressBar(self.view)
            self.add_custom_widget(self._progress_widget)
        else:
            self._progress_widget = None
        
        self.is_dirty = True
        self.is_disabled = False
        self.output_values = {}
        self._active_dialogs = [] # Keep references to popups to prevent GC
        self._eval_version = 0    # incremented on each parameter change, for stale eval detection
        self._eval_version_at_start = 0  # captured at evaluate start
        
        # Ensure we start in the "Dirty" (Blue) state immediately.
        # Since nodes are always created on the Main Thread, we can call this directly.
        self._mark_dirty_ui()
        
    def _add_list_input(self, name, label='', items=None, tab=None):
        self.create_property(name, value=items or [], widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab=tab)
        widget = NodeListWidget(self.view, name, label, items)
        widget.value_changed.connect(lambda k, v: self.set_property(k, v))
        self.view.add_widget(widget)
        self.view.draw_node()
        return widget

    # ── Convenience widget helpers (available to ALL node subclasses) ──────────

    def _add_int_spinbox(self, name, label, value=0, min_val=0, max_val=99999, step=1, tab=None):
        """Embed an integer spinbox widget on the node card."""
        w = NodeIntSpinBoxWidget(self.view, name=name, label=label,
                                 value=value, min_val=min_val, max_val=max_val, step=step)
        self.add_custom_widget(w, tab=tab)

    def _add_float_spinbox(self, name, label, value=0.0, min_val=0.0, max_val=99999.0,
                           step=0.1, decimals=3, tab=None):
        """Embed a floating-point spinbox widget on the node card."""
        w = NodeFloatSpinBoxWidget(self.view, name=name, label=label,
                                   value=value, min_val=min_val, max_val=max_val,
                                   step=step, decimals=decimals)
        self.add_custom_widget(w, tab=tab)

    def _add_row(self, row_name, label, fields, tab=None):
        """Add a compact horizontal row of labeled spinboxes.

        Each field dict: {name, label, type('int'|'float'), value, min_val, max_val, step, decimals}
        Creates individual properties for each field name.
        """
        w = NodeRowWidget(self.view, name=row_name, label=label, fields=fields)
        # Register the row widget's own compound name (needed by update_model)
        self.create_property(row_name, w.get_value())
        # Create individual properties for each field
        for f in fields:
            self.create_property(f['name'], f['value'])
        # Wire value_changed → set_property (emits per-field name)
        w.value_changed.connect(lambda k, v: self.set_property(k, v))
        w._node = self
        self.view.add_widget(w)
        self.view.draw_node()

    _column_selector_names = None  # lazily initialized set of property names

    def _add_column_selector(self, name, label='', text='', mode='single', tab=None):
        """Embed a text input with a column dropdown button.

        Args:
            name:  property name
            label: display label
            text:  default text
            mode:  'single' (replace on click) or 'multi' (toggle comma-separated)
            tab:   properties panel tab
        """
        w = NodeColumnSelectorWidget(self.view, name=name, label=label,
                                     text=text, mode=mode)
        self.add_custom_widget(w, tab=tab)
        if self._column_selector_names is None:
            self._column_selector_names = set()
        self._column_selector_names.add(name)
        return w

    def _refresh_column_selectors(self, df, *prop_names):
        """Update column selector dropdowns with columns from a DataFrame.

        Thread-safe: only stores the column list. The QMenu is built
        lazily on the main thread when the user clicks the dropdown.
        """
        if df is None:
            return
        columns = list(df.columns)
        for prop_name in prop_names:
            widget = self.get_widget(prop_name)
            if widget and hasattr(widget, 'set_columns'):
                widget.set_columns(columns)

    # ── Collection auto-loop ────────────────────────────────────────────────
    _handles_collection = False  # True for nodes that natively handle CollectionData (Collect, Select)
    _collection_aware = False    # True for stateless nodes safe to auto-loop over collections

    def _check_collection_inputs(self):
        """Check if any input port has a CollectionData. Returns (port_name, CollectionData) or None."""
        from data_models import CollectionData
        for port_name, port in self.inputs().items():
            for connected in port.connected_ports():
                data = connected.node().output_values.get(connected.name())
                if isinstance(data, CollectionData):
                    return port_name, data
        return None

    def _evaluate_collection_loop(self, col_port_name, collection):
        """Run evaluate() once per collection item, repack outputs."""
        from data_models import CollectionData

        # Save original output_values
        all_outputs = {}  # {port_name: {item_name: data}}
        total = len(collection)
        self.set_progress(0)

        # Suppress per-item progress and display calls during the loop
        _orig_set_progress = self.set_progress
        _orig_set_display = self.set_display
        _last_display = [None]  # capture last display data
        self.set_progress = lambda v: None
        self.set_display = lambda d: _last_display.__setitem__(0, d)

        for idx, (item_name, item_data) in enumerate(collection.payload.items()):
            _orig_set_progress(int(idx / total * 100))
            # Temporarily inject the single item as the port's output
            # by patching the upstream node's output_values
            port = self.inputs().get(col_port_name)
            if not port:
                continue
            connected_ports = port.connected_ports()
            if not connected_ports:
                continue

            # Find all collection inputs and single inputs
            # For collection inputs: swap in the current item
            # For single inputs: leave as-is (broadcast)
            originals = {}
            for cp in connected_ports:
                upstream = cp.node()
                up_port = cp.name()
                orig_val = upstream.output_values.get(up_port)
                originals[(id(upstream), up_port)] = orig_val
                if isinstance(orig_val, CollectionData):
                    # Swap collection with single item
                    upstream.output_values[up_port] = orig_val.get(item_name) or item_data

            # Also check other input ports for collections and swap them
            for other_name, other_port in self.inputs().items():
                if other_name == col_port_name:
                    continue
                for cp in other_port.connected_ports():
                    upstream = cp.node()
                    up_port = cp.name()
                    orig_val = upstream.output_values.get(up_port)
                    if isinstance(orig_val, CollectionData):
                        originals[(id(upstream), up_port)] = orig_val
                        # Pair by name if available, otherwise broadcast first item
                        upstream.output_values[up_port] = (
                            orig_val.get(item_name) or
                            next(iter(orig_val.payload.values()), None)
                        )

            # Run the node's normal evaluate
            self.output_values = {}
            success, err = self.evaluate()

            # Restore original upstream values
            for (uid, up_port), orig_val in originals.items():
                for cp in port.connected_ports():
                    if id(cp.node()) == uid:
                        cp.node().output_values[up_port] = orig_val
                for other_name, other_port in self.inputs().items():
                    for cp in other_port.connected_ports():
                        if id(cp.node()) == uid:
                            cp.node().output_values[up_port] = orig_val

            if not success:
                continue  # skip failed items

            # Collect outputs per port
            for out_name, out_val in self.output_values.items():
                if out_name not in all_outputs:
                    all_outputs[out_name] = {}
                all_outputs[out_name][item_name] = out_val

        # Restore progress and display functions
        self.set_progress = _orig_set_progress
        self.set_display = _orig_set_display
        self.set_progress(100)
        # Show the last item's display data
        if _last_display[0] is not None:
            self.set_display(_last_display[0])

        # Repack: each output port gets a CollectionData
        self.output_values = {}
        for out_name, items_dict in all_outputs.items():
            if len(items_dict) == 0:
                continue
            self.output_values[out_name] = CollectionData(payload=items_dict)

        return True, None

    def set_display(self, data):
        """Signals that this node wants to display data (Main Thread only)."""
        NODE_SIGNALS.display_requested.emit(self.id, data)

    def _get_input_image_data(self):
        """Find the first connected ImageData from input ports."""
        from data_models import ImageData
        for port in self.inputs().values():
            for connected in port.connected_ports():
                data = connected.node().output_values.get(connected.name())
                if isinstance(data, ImageData):
                    return data
        return None

    def _make_image_output(self, img, port_name='image'):
        """Create ImageData with all upstream metadata propagated automatically."""
        from data_models import ImageData
        upstream = self._get_input_image_data()
        if upstream:
            kwargs = {f: getattr(upstream, f, None)
                      for f in upstream.model_fields if f != 'payload'}
            out = ImageData(payload=img, **kwargs)
        else:
            out = ImageData(payload=img)
        self.output_values[port_name] = out
        return out

    def _display_ui(self, data):
        """Actual UI logic for display, to be overridden by subclasses."""
        pass
        
    def set_progress(self, value):
        """Sets the progress bar value (0-100) via signals."""
        NODE_SIGNALS.progress_updated.emit(self.id, int(value))

    def _set_progress_ui(self, value):
        """Internal UI update for progress, must run on main thread."""
        if self._progress_widget:
            self._progress_widget.set_value(value)

    def reset_progress(self):
        if self._progress_widget:
            self.set_progress(0)
        
    def mark_dirty(self):
        """Marks the node as dirty, cascades the dirty state to all downstream nodes."""
        if self.is_dirty:
            return  # Already dirty — no need to re-propagate
        self.is_dirty = True
        NODE_SIGNALS.status_updated.emit(self.id, 'dirty')
        # Cascade: any node that depends on our output must also re-evaluate
        for out_port in self.outputs().values():
            for in_port in out_port.connected_ports():
                dn = in_port.node()
                if hasattr(dn, 'mark_dirty'):
                    dn.mark_dirty()
        
    def _mark_dirty_ui(self):
        """Performs actual UI updates for dirty state (Main Thread only)."""
        self.set_color(30, 60, 100) # Deep Royal Blue for dirty
        self.view.border_color = (40, 80, 150, 255) # Matching border
        self.view.update()

    def _fix_widget_z_order(self):
        """
        Manually adjust Z-order of widgets to ensure top-to-bottom layering.
        Widgets added earlier (higher on the node) will get higher Z-values.
        """
        from NodeGraphQt.constants import Z_VAL_NODE_WIDGET
        widgets = list(self.view.widgets.values())
        for i, widget in enumerate(widgets):
            # Assign descending Z-values from Z_VAL_NODE_WIDGET
            # This ensures popups from upper widgets render above lower widgets.
            widget.setZValue(Z_VAL_NODE_WIDGET + (len(widgets) - i))
        
    def clear_cache(self):
        """Manually clears the output memory and forces recalculation on next run."""
        self.output_values = {}
        self.reset_progress()
        self.mark_dirty()
        
    def mark_clean(self):
        """Emits a signal to mark the node as clean."""
        self.is_dirty = False # Set immediately for thread-safety in execution loop
        NODE_SIGNALS.status_updated.emit(self.id, 'clean')

    def _mark_clean_ui(self):
        """Performs actual UI updates for clean state (Main Thread only)."""
        self.set_color(35, 38, 45) # Dark Obsidian for clean
        self.view.border_color = (74, 84, 85, 255) # Default NodeGraphQt border color
        self.view.update()
        
    def mark_error(self):
        """Emits a signal to mark the node as in error state."""
        NODE_SIGNALS.status_updated.emit(self.id, 'error')

    def _mark_error_ui(self):
        """Performs actual UI updates for error state (Main Thread only)."""
        self.set_color(80, 25, 30) # Dark Maroon for error
        self.view.update()

    def mark_skipped(self):
        """Marks the node as disabled/skipped (won't run until re-enabled)."""
        self.is_dirty = False
        NODE_SIGNALS.status_updated.emit(self.id, 'skipped')

    def _mark_skipped_ui(self):
        """Disabled visual: muted olive — distinct from all execution states."""
        self.set_color(45, 45, 45)         # Very dark grey
        self.view.border_color = (80, 80, 80, 255)  # Dim border
        self.view.update()

    def mark_disabled(self):
        """Explicitly disables this node — it will be skipped on every run."""
        self.is_disabled = True
        NODE_SIGNALS.status_updated.emit(self.id, 'disabled')

    def mark_enabled(self):
        """Re-enables this node so it runs again."""
        self.is_disabled = False
        NODE_SIGNALS.status_updated.emit(self.id, 'enabled')
        self.mark_dirty()   # Force re-evaluation on next run

    def _mark_disabled_ui(self):
        """Disabled visual: flat grey with dashed-style dim border."""
        self.set_color(40, 40, 40)         # Almost black
        self.view.border_color = (75, 75, 75, 180)  # Dim border
        self.view.update()

    def _mark_enabled_ui(self):
        """Re-enabled = back to dirty state."""
        self._mark_dirty_ui()
        
    def _is_eval_stale(self) -> bool:
        """Check if parameters changed since evaluate started.
        Call this during long evaluations to bail early."""
        return self._eval_version != self._eval_version_at_start

    def on_input_connected(self, in_port, out_port):
        self.mark_dirty()
        # Auto-populate column selectors from upstream DataFrame
        has_col_selectors = self._column_selector_names
        has_tb_col_buttons = hasattr(self, '_tb_col_buttons') and self._tb_col_buttons
        if has_col_selectors or has_tb_col_buttons:
            self._auto_refresh_column_selectors()

    def _auto_refresh_column_selectors(self):
        """Try to read a DataFrame from the first connected table port and refresh all column selectors."""
        try:
            import pandas as _pd
            for port_name, port in self.inputs().items():
                for cp in port.connected_ports():
                    data = cp.node().output_values.get(cp.name())
                    df = None
                    if hasattr(data, 'df'):
                        df = data.df
                    elif hasattr(data, 'payload') and isinstance(data.payload, _pd.DataFrame):
                        df = data.payload
                    elif isinstance(data, _pd.DataFrame):
                        df = data
                    if df is not None:
                        # Refresh NodeColumnSelectorWidget instances
                        if self._column_selector_names:
                            self._refresh_column_selectors(df, *self._column_selector_names)
                        # Refresh toolbox column selector buttons (plot nodes)
                        if hasattr(self, '_tb_col_buttons') and self._tb_col_buttons:
                            columns = list(df.columns)
                            for btn in self._tb_col_buttons.values():
                                btn._columns = columns
                        return
        except Exception:
            pass
        
    def on_input_disconnected(self, in_port, out_port):
        self.mark_dirty()
        
    def set_property(self, name, value, push_undo=True):
        # Only invalidate if the value actually changes.
        cur_val = self.get_property(name)
        if isinstance(cur_val, (list, tuple)) and isinstance(value, (list, tuple)):
            if list(cur_val) == list(value):
                return
        elif cur_val == value:
            return
            
        super(BaseExecutionNode, self).set_property(name, value, push_undo)
        if name not in ['color', 'pos', 'selected', 'name', 'progress', 'table_view', 'image_view', 'show_preview', 'live_preview']:
            self._eval_version += 1
            self.mark_dirty()


# ---------------------------------------------------------------------------
# Shared image-processing base (used by image_process_nodes, mask_nodes,
# vision_nodes, filopodia_nodes plugins)
# ---------------------------------------------------------------------------

def _arr_to_pil(arr, mode=None):
    """
    Create a PIL Image from a numpy array using frombytes instead of fromarray.
    PIL's _fromarray_typemap can be uninitialized on the first call in Nuitka
    macOS frozen builds (Python 3.14), causing a KeyError. frombytes takes an
    explicit mode and bypasses the typemap entirely.

    When mode is None, it is inferred from the array shape:
      2D         → 'L'
      3D, ch=3   → 'RGB'
      3D, ch=4   → 'RGBA'
    """
    import numpy as np
    from PIL import Image
    if arr.dtype == np.uint16:
        arr = (arr / 256).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    if mode is None:
        if arr.ndim == 2:
            mode = 'L'
        elif arr.ndim == 3 and arr.shape[2] == 3:
            mode = 'RGB'
        elif arr.ndim == 3 and arr.shape[2] == 4:
            mode = 'RGBA'
        else:
            raise ValueError(f"Cannot infer PIL mode from array shape {arr.shape}")
    return Image.frombytes(mode, (arr.shape[1], arr.shape[0]), arr.tobytes())


class BaseImageProcessNode(BaseExecutionNode):
    """
    Base class for image processing nodes. Provides a standardized image preview
    widget that can be toggled via a checkbox to save canvas space.
    """
    _UI_PROPS = {'color', 'pos', 'selected', 'name', 'progress',
                 'table_view', 'image_view', 'show_preview', 'live_preview'}

    def __init__(self):
        super(BaseImageProcessNode, self).__init__()

    def create_preview_widgets(self):
        self._last_display_data = None
        self.add_checkbox('live_preview', '', text='Live Update', state=True)
        self.add_checkbox('show_preview', '', text='Show Preview', state=True)
        self._image_widget = NodeImageWidget(self.view)
        self.add_custom_widget(self._image_widget, tab='View')

        # Patch the view's set_proxy_mode so that when proxy mode turns off
        # (zoom back in), the image widget respects the show_preview state
        # instead of becoming unconditionally visible.
        _original_set_proxy = self.view.set_proxy_mode
        _node_ref = self
        def _patched_set_proxy(mode):
            _original_set_proxy(mode)
            if not mode and hasattr(_node_ref, '_image_widget'):
                show = bool(_node_ref.get_property('show_preview'))
                _node_ref._image_widget.setVisible(show)
        self.view.set_proxy_mode = _patched_set_proxy

    def set_property(self, name, value, push_undo=True):
        super(BaseImageProcessNode, self).set_property(name, value, push_undo)
        if name == 'show_preview':
            is_visible = bool(value)
            if hasattr(self, '_image_widget'):
                self._image_widget.setVisible(is_visible)
                if hasattr(self.view, 'draw_node'):
                    self.view.draw_node()
        elif name == 'live_preview':
            if bool(value) and hasattr(self, '_last_display_data') and self._last_display_data is not None:
                self._display_ui(self._last_display_data)
        else:
            if name not in self._UI_PROPS and self.get_property('live_preview'):
                # Capture version before evaluate — if it changes during
                # evaluation (user moved slider again), discard the result
                ver = self._eval_version
                col_info = self._check_collection_inputs() if self._collection_aware else None
                if col_info:
                    success, err = self._evaluate_collection_loop(*col_info)
                else:
                    success, err = self.evaluate()
                if self._eval_version != ver:
                    return  # stale — a newer evaluation will follow
                if success:
                    self.mark_clean()
                else:
                    self.mark_error()

    def _display_ui(self, data):
        from PIL import Image
        import numpy as np
        self._last_display_data = data
        if not self.get_property('live_preview'):
            return
        if isinstance(data, (Image.Image, np.ndarray)):
            self._image_widget.set_value(data)
        else:
            self._image_widget.set_value(None)
        if hasattr(self, '_image_widget'):
            self._image_widget.setVisible(bool(self.get_property('show_preview')))
        if hasattr(self.view, 'draw_node'):
            self.view.draw_node()
