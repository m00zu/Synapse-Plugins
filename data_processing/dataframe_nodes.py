"""
nodes/dataframe_nodes.py
=========================
Nodes for wrangling, processing, and editing tabular TableData.
"""
import fnmatch
import pandas as pd
import numpy as np
import NodeGraphQt, json
from PySide6 import QtWidgets, QtCore, QtGui

from data_models import TableData
from nodes.base import (
    BaseExecutionNode, PORT_COLORS,
    NodeBaseWidget
)


def _col_letter(n: int) -> str:
    """Convert 0-based column index to Excel-style letter (A, B, ..., Z, AA, ...)."""
    result = ''
    while True:
        result = chr(65 + n % 26) + result
        n = n // 26 - 1
        if n < 0:
            break
    return result


def _is_cell_empty(val) -> bool:
    """Return True if a cell value should be considered empty/blank."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str) and val.strip() in ('', 'nan', 'None'):
        return True
    return False


class _SpreadsheetModel(QtCore.QAbstractTableModel):
    """
    Virtual spreadsheet model: a real DataFrame plus a buffer zone of empty
    rows/columns.  Typing into the buffer auto-expands the DataFrame.
    """
    BUFFER_ROWS = 50
    BUFFER_COLS = 10

    _BG_DATA   = QtGui.QColor(0x22, 0x22, 0x22)
    _BG_BUFFER = QtGui.QColor(0x1a, 0x1a, 0x1a)
    _FG        = QtGui.QColor(0xdd, 0xdd, 0xdd)
    _FG_BUFFER = QtGui.QColor(0x55, 0x55, 0x55)

    # Emitted when a single cell is edited via the QTableView delegate.
    # Separate from dataChanged so the widget can distinguish user edits
    # from programmatic model resets without double-emitting.
    cell_edited = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df = pd.DataFrame()
        self._sort_column: int | None = None
        self._sort_ascending: bool = True

    # ── public helpers ───────���─────────────────────────────────────────
    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def set_dataframe(self, df: pd.DataFrame | None):
        self.beginResetModel()
        self._df = df.copy() if df is not None else pd.DataFrame()
        self._sort_column = None
        self.endResetModel()

    def trim(self) -> pd.DataFrame:
        """Return the DataFrame with trailing all-empty rows/columns removed."""
        df = self._df.copy()
        while len(df) > 0 and df.iloc[-1].apply(_is_cell_empty).all():
            df = df.iloc[:-1]
        while len(df.columns) > 0 and df.iloc[:, -1].apply(_is_cell_empty).all():
            df = df.iloc[:, :-1]
        if df.empty:
            return pd.DataFrame()
        return df.reset_index(drop=True)

    # ── QAbstractTableModel interface ──────────────────────────────────
    def rowCount(self, parent=None):
        return len(self._df) + self.BUFFER_ROWS

    def columnCount(self, parent=None):
        return max(len(self._df.columns), 1) + self.BUFFER_COLS

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        r, c = index.row(), index.column()
        in_data = r < len(self._df) and c < len(self._df.columns)

        if role in (QtCore.Qt.ItemDataRole.DisplayRole,
                    QtCore.Qt.ItemDataRole.EditRole):
            if not in_data:
                return ''
            val = self._df.iat[r, c]
            if _is_cell_empty(val):
                return ''
            if role == QtCore.Qt.ItemDataRole.EditRole:
                return str(val)
            if isinstance(val, (float, np.floating)):
                return f'{val:.2e}' if 0 < abs(val) < 0.0001 else f'{val:.4f}'
            return str(val)

        if role == QtCore.Qt.ItemDataRole.BackgroundRole:
            return self._BG_DATA if in_data else self._BG_BUFFER
        if role == QtCore.Qt.ItemDataRole.ForegroundRole:
            return self._FG if in_data else self._FG_BUFFER
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            return int(QtCore.Qt.AlignmentFlag.AlignLeft |
                       QtCore.Qt.AlignmentFlag.AlignVCenter)
        return None

    def setData(self, index, value, role=QtCore.Qt.ItemDataRole.EditRole):
        if role != QtCore.Qt.ItemDataRole.EditRole:
            return False
        r, c = index.row(), index.column()
        text = str(value).strip() if value is not None else ''

        # Don't expand the DataFrame for empty edits in the buffer zone
        if (r >= len(self._df) or c >= len(self._df.columns)) and not text:
            return False

        if r >= len(self._df) or c >= len(self._df.columns):
            self._expand_to(r, c)

        parsed = self._parse_value(text, c)

        col_name = self._df.columns[c]
        if (not isinstance(parsed, str)
                and not pd.api.types.is_numeric_dtype(self._df[col_name].dtype)):
            self._df[col_name] = self._df[col_name].astype(object)

        self._df.iat[r, c] = parsed

        # Promote column to numeric if every value converts cleanly
        if not pd.api.types.is_numeric_dtype(self._df[col_name].dtype):
            promoted = pd.to_numeric(self._df[col_name], errors='coerce')
            if promoted.notna().all():
                self._df[col_name] = promoted

        self.dataChanged.emit(index, index, [role])
        self.cell_edited.emit()
        return True

    def flags(self, index):
        return (QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsEditable)

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            if section < len(self._df.columns):
                label = str(self._df.columns[section])
                if section == self._sort_column:
                    label += (' \u2191' if self._sort_ascending else ' \u2193')
                return label
            return _col_letter(section)
        return str(section)

    # ── mutation helpers (called by the widget) ────────────────────────
    def sort_by_column(self, col_idx):
        if col_idx >= len(self._df.columns):
            return
        col_name = self._df.columns[col_idx]
        if self._sort_column == col_idx:
            self._sort_ascending = not self._sort_ascending
        else:
            self._sort_column = col_idx
            self._sort_ascending = True
        self.beginResetModel()
        self._df = self._df.sort_values(
            by=col_name, ascending=self._sort_ascending
        ).reset_index(drop=True)
        self.endResetModel()

    def rename_column(self, col_idx, new_name):
        if col_idx >= len(self._df.columns) or new_name in self._df.columns:
            return False
        self._df.rename(
            columns={self._df.columns[col_idx]: new_name}, inplace=True)
        self.headerDataChanged.emit(
            QtCore.Qt.Orientation.Horizontal, col_idx, col_idx)
        return True

    def insert_rows(self, at, count=1):
        at = min(at, len(self._df))
        new = pd.DataFrame({c: [None] * count for c in self._df.columns})
        self.beginResetModel()
        self._df = pd.concat(
            [self._df.iloc[:at], new, self._df.iloc[at:]], ignore_index=True)
        self.endResetModel()

    def insert_columns(self, at, count=1):
        self.beginResetModel()
        for i in range(count):
            name = _col_letter(len(self._df.columns))
            while name in self._df.columns:
                name += '_'
            if at + i >= len(self._df.columns):
                self._df[name] = pd.Series([None] * len(self._df), dtype=object)
            else:
                self._df.insert(at + i, name,
                                pd.Series([None] * len(self._df), dtype=object))
        self.endResetModel()

    def delete_rows(self, rows):
        rows = sorted(set(r for r in rows if r < len(self._df)), reverse=True)
        if not rows:
            return
        self.beginResetModel()
        self._df = self._df.drop(self._df.index[rows]).reset_index(drop=True)
        self.endResetModel()

    def delete_columns(self, cols):
        cols = sorted(set(c for c in cols if c < len(self._df.columns)),
                      reverse=True)
        if not cols:
            return
        names = [self._df.columns[c] for c in cols]
        self.beginResetModel()
        self._df = self._df.drop(columns=names)
        self.endResetModel()

    def clear_cells(self, indices):
        changed = False
        for r, c in indices:
            if r < len(self._df) and c < len(self._df.columns):
                self._df.iat[r, c] = None
                changed = True
        if changed:
            tl = self.index(min(r for r, _ in indices),
                            min(c for _, c in indices))
            br = self.index(max(r for r, _ in indices),
                            max(c for _, c in indices))
            self.dataChanged.emit(tl, br)

    def paste_block(self, start_row, start_col, text):
        """Parse tab/newline-separated *text* and fill cells from (start_row, start_col)."""
        lines = text.rstrip('\n').split('\n')
        rows_data = [line.split('\t') for line in lines]
        if not rows_data:
            return
        max_r = start_row + len(rows_data) - 1
        max_c = start_col + max(len(r) for r in rows_data) - 1
        if max_r >= len(self._df) or max_c >= len(self._df.columns):
            self._expand_to(max_r, max_c)
        for ri, row_vals in enumerate(rows_data):
            for ci, val in enumerate(row_vals):
                r, c = start_row + ri, start_col + ci
                parsed = self._parse_value(val.strip(), c)
                col_name = self._df.columns[c]
                if (not isinstance(parsed, str)
                        and not pd.api.types.is_numeric_dtype(
                            self._df[col_name].dtype)):
                    self._df[col_name] = self._df[col_name].astype(object)
                self._df.iat[r, c] = parsed
        self.beginResetModel()
        self.endResetModel()

    # ── internal ─────────────��─────────────────────────────────────────
    def _expand_to(self, row, col):
        cur_rows, cur_cols = len(self._df), len(self._df.columns)
        if col >= cur_cols:
            for i in range(cur_cols, col + 1):
                name = _col_letter(i)
                while name in self._df.columns:
                    name += '_'
                self._df[name] = pd.Series(
                    [None] * max(len(self._df), 1), dtype=object)
        if row >= len(self._df):
            n_new = row - len(self._df) + 1
            new = pd.DataFrame(
                {c: [None] * n_new for c in self._df.columns})
            self._df = pd.concat([self._df, new], ignore_index=True)
        self.layoutChanged.emit()

    def _parse_value(self, text, col):
        if not text:
            return None
        if col < len(self._df.columns):
            ctype = self._df.dtypes.iloc[col]
            if pd.api.types.is_numeric_dtype(ctype):
                try:
                    return int(text) if pd.api.types.is_integer_dtype(ctype) else float(text)
                except (ValueError, TypeError):
                    return text
        try:
            return int(text)
        except (ValueError, TypeError):
            try:
                return float(text)
            except (ValueError, TypeError):
                return text


class _SpreadsheetView(QtWidgets.QTableView):
    """QTableView subclass that adds copy / paste / delete key handling."""
    paste_requested = QtCore.Signal(str)
    delete_requested = QtCore.Signal()
    copy_requested = QtCore.Signal()

    def keyPressEvent(self, event):
        if event.matches(QtGui.QKeySequence.StandardKey.Paste):
            text = QtWidgets.QApplication.clipboard().text()
            if text:
                self.paste_requested.emit(text)
            return
        if event.matches(QtGui.QKeySequence.StandardKey.Copy):
            self.copy_requested.emit()
            return
        if event.key() in (QtCore.Qt.Key.Key_Delete,
                           QtCore.Qt.Key.Key_Backspace):
            if not self.state() == QtWidgets.QAbstractItemView.State.EditingState:
                self.delete_requested.emit()
                return
        super().keyPressEvent(event)


class EditableNodeTableWidget(NodeBaseWidget):
    """
    Excel-like spreadsheet widget embedded in a node.

    Shows a large virtual grid (data + buffer zone). Users can type anywhere
    to expand the DataFrame, double-click column headers to rename them,
    single-click headers to sort, and right-click for insert/delete operations.
    """
    dataframe_edited = QtCore.Signal(object)
    update_table_signal = QtCore.Signal(object)

    def __init__(self, parent=None):
        super(EditableNodeTableWidget, self).__init__(
            parent, name='editable_table', label='')

        self._model = _SpreadsheetModel()
        self._is_updating_ui = False
        self._header_editor = None

        # ── view ───────────────────────────────────────────────────────
        self._table = _SpreadsheetView()
        self._table.setModel(self._model)

        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.AnyKeyPressed
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed)
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self._table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self._table.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollPerPixel)
        self._table.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollPerPixel)
        self._table.horizontalHeader().setDefaultSectionSize(80)
        self._table.horizontalHeader().setMinimumSectionSize(40)
        self._table.verticalHeader().setDefaultSectionSize(24)

        self._table.setStyleSheet("""
            QTableView {
                background-color: #1e1e1e; color: #ddd;
                gridline-color: #333; border: 1px solid #555;
                selection-background-color: #264f78;
                selection-color: #fff;
            }
            QHeaderView::section {
                background-color: #2d2d2d; color: #fff;
                padding: 3px; border: 1px solid #444; font-size: 10px;
            }
        """)
        self._table.setFixedSize(480, 360)

        # ── header interactions ────────────────────────────────────────
        h = self._table.horizontalHeader()
        h.setSectionsClickable(True)
        h.sectionClicked.connect(self._on_header_click)
        h.sectionDoubleClicked.connect(self._on_header_double_click)

        self._sort_timer = QtCore.QTimer()
        self._sort_timer.setSingleShot(True)
        self._sort_timer.setInterval(300)
        self._pending_sort_col = None
        self._sort_timer.timeout.connect(self._do_sort)

        # ── keyboard shortcuts ─────────────────────────────────────────
        self._table.paste_requested.connect(self._on_paste)
        self._table.delete_requested.connect(self._on_delete)
        self._table.copy_requested.connect(self._on_copy)

        # ── context menu ────────────��───────────────────────���──────────
        self._table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_context_menu)

        # ── data-change tracking (cell_edited fires only for user edits) ──
        self._model.cell_edited.connect(self._emit_edit)

        # ── layout ��────────────────────────────────────────────────────
        container = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(container)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(0)
        lay.addWidget(self._table)
        self.set_custom_widget(container)

        self.update_table_signal.connect(
            self._set_df_main_thread, QtCore.Qt.QueuedConnection)

    # ── header sort / rename ────────────���──────────────────────────────
    def _on_header_click(self, section):
        self._pending_sort_col = section
        self._sort_timer.start()

    def _on_header_double_click(self, section):
        self._sort_timer.stop()
        self._pending_sort_col = None
        if section >= len(self._model.df.columns):
            return
        self._start_header_edit(section)

    def _do_sort(self):
        if self._pending_sort_col is None:
            return
        col = self._pending_sort_col
        self._pending_sort_col = None
        self._model.sort_by_column(col)
        self._emit_edit()

    def _start_header_edit(self, section):
        header = self._table.horizontalHeader()
        x = header.sectionPosition(section) - header.offset()
        w = header.sectionSize(section)
        h = header.height()
        old_name = str(self._model.df.columns[section])

        editor = QtWidgets.QLineEdit(header)
        editor.setGeometry(x, 0, w, h)
        editor.setText(old_name)
        editor.selectAll()
        editor.setStyleSheet(
            'QLineEdit { background: #1a1a2e; color: #fff; '
            'border: 2px solid #4a9eff; padding: 2px; font-size: 10px; }')

        def _finish():
            new_name = editor.text().strip()
            editor.deleteLater()
            self._header_editor = None
            if new_name and new_name != old_name:
                if self._model.rename_column(section, new_name):
                    self._emit_edit()

        editor.editingFinished.connect(_finish)
        editor.show()
        editor.setFocus()
        self._header_editor = editor

    # ── keyboard actions ───────────────────────────────────────────────
    def _on_paste(self, text):
        idx = self._table.currentIndex()
        if not idx.isValid():
            return
        self._model.paste_block(idx.row(), idx.column(), text)
        self._emit_edit()

    def _on_copy(self):
        sel = self._table.selectionModel().selectedIndexes()
        if not sel:
            return
        rows = sorted(set(i.row() for i in sel))
        cols = sorted(set(i.column() for i in sel))
        lines = []
        for r in rows:
            cells = []
            for c in cols:
                idx = self._model.index(r, c)
                val = self._model.data(idx, QtCore.Qt.ItemDataRole.DisplayRole)
                cells.append(str(val) if val else '')
            lines.append('\t'.join(cells))
        QtWidgets.QApplication.clipboard().setText('\n'.join(lines))

    def _on_delete(self):
        sel = self._table.selectionModel().selectedIndexes()
        if not sel:
            return
        indices = [(i.row(), i.column()) for i in sel]
        self._model.clear_cells(indices)
        self._emit_edit()

    # ── context menu ──────────────────────────────���────────────────────
    def _show_context_menu(self, pos):
        sel = self._table.selectionModel().selectedIndexes()
        rows = sorted(set(i.row() for i in sel)) if sel else []
        cols = sorted(set(i.column() for i in sel)) if sel else []
        n_data_rows = len(self._model.df)
        n_data_cols = len(self._model.df.columns)

        menu = QtWidgets.QMenu(self._table)
        if rows:
            menu.addAction('Insert Row Above',
                           lambda: (self._model.insert_rows(rows[0]),
                                    self._emit_edit()))
            menu.addAction('Insert Row Below',
                           lambda: (self._model.insert_rows(rows[-1] + 1),
                                    self._emit_edit()))
            data_rows = [r for r in rows if r < n_data_rows]
            if data_rows:
                menu.addAction(
                    f'Delete Row{"s" if len(data_rows) > 1 else ""}',
                    lambda: (self._model.delete_rows(data_rows),
                             self._emit_edit()))
        menu.addSeparator()
        if cols:
            menu.addAction('Insert Column Left',
                           lambda: (self._model.insert_columns(cols[0]),
                                    self._emit_edit()))
            menu.addAction('Insert Column Right',
                           lambda: (self._model.insert_columns(cols[-1] + 1),
                                    self._emit_edit()))
            data_cols = [c for c in cols if c < n_data_cols]
            if data_cols:
                menu.addAction(
                    f'Delete Column{"s" if len(data_cols) > 1 else ""}',
                    lambda: (self._model.delete_columns(data_cols),
                             self._emit_edit()))
        menu.addSeparator()
        if sel:
            menu.addAction('Clear Cell(s)', self._on_delete)
        menu.exec_(self._table.viewport().mapToGlobal(pos))

    # ── data change tracking ───────────────────────────────────────────
    def _emit_edit(self):
        trimmed = self._model.trim()
        self.dataframe_edited.emit(trimmed)

    # ── NodeBaseWidget interface ───────────────��───────────────────────
    def set_value(self, value):
        """Accept a DataFrame or a saved JSON dict snapshot."""
        restored_from_json = isinstance(value, dict)

        if isinstance(value, dict) and 'columns' in value:
            try:
                new_df = pd.DataFrame(
                    value['data'],
                    index=value.get('index'),
                    columns=value['columns'],
                )
            except Exception:
                new_df = None
        elif isinstance(value, pd.DataFrame):
            new_df = value.copy()
        else:
            new_df = None

        import threading
        if threading.current_thread() is threading.main_thread():
            self._set_df_main_thread(new_df)
        else:
            self.update_table_signal.emit(new_df)

        if restored_from_json and new_df is not None:
            self.dataframe_edited.emit(new_df)

    @QtCore.Slot(object)
    def _set_df_main_thread(self, df):
        self._is_updating_ui = True
        self._model.set_dataframe(df)
        self._is_updating_ui = False

    def get_dataframe(self):
        """Return the trimmed live DataFrame."""
        return self._model.trim()

    def get_value(self):
        """Return a JSON-serializable snapshot for workflow save."""
        df = self._model.trim()
        if df is None or df.empty:
            return None
        try:
            return json.loads(df.to_json(orient='split'))
        except Exception:
            return None


class EditableTableNode(BaseExecutionNode):
    """
    Displays an input table in an editable spreadsheet widget and outputs the modified result.

    The node accepts a TableData input and presents it in an Excel-like spreadsheet
    where you can type anywhere to add data, double-click column headers to rename
    them, single-click headers to sort, and right-click for insert/delete operations.
    Copy/paste and Delete key are supported. Changes are pushed downstream automatically.

    Parameters:
    - **Reset Edits on Next Run** — when checked, discards local edits and reloads from upstream on the next evaluation

    Keywords: table editor, manual edit, interactive dataframe, inline spreadsheet, clean data by hand, 表格編輯, 手動編輯, 資料清理, 選擇欄位, 互動式
    """
    __identifier__ = 'nodes.dataframe.Util'
    NODE_NAME = 'Editable Table'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}
    
    def __init__(self):
        super(EditableTableNode, self).__init__(use_progress=False)
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('out', color=PORT_COLORS['table'])
        
        self.add_checkbox('reset_edits', 'Reset Edits on Next Run', text='Reset', state=False, tab='Editor')
        
        self._last_input_df = None
        
        self.editable_widget = EditableNodeTableWidget(self.view)
        self.add_custom_widget(self.editable_widget, tab='Editor')
        
        # When manual edits occur, mark the node dirty to propagate changes down
        self.editable_widget.dataframe_edited.connect(self._on_table_edited)

        # Start with an empty spreadsheet — the buffer zone provides visual cells
        self.output_values['out'] = TableData(payload=pd.DataFrame())
        
    def _on_table_edited(self, new_df):
        """Triggered asynchronously via Qt Signals when the user clicks out of a table cell."""
        # Update output
        self.output_values['out'] = TableData(payload=new_df)
        
        # Manually trigger downstream refresh
        self.mark_dirty()
        for out_port in self.outputs().values():
            for connected_port in out_port.connected_ports():
                connected_port.node().mark_dirty()

    def evaluate(self):
        self.reset_progress()
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            # No input — retain whatever the user has typed in the spreadsheet
            edited_df = self.editable_widget.get_dataframe()
            if edited_df is None:
                edited_df = pd.DataFrame()
            self.output_values['out'] = TableData(payload=edited_df)
            self._last_input_df = None
            self.set_progress(100)
            self.mark_clean()
            return True, None
            
        up_node = in_port.connected_ports()[0].node()
        data = up_node.output_values.get(in_port.connected_ports()[0].name())
        
        if not isinstance(data, TableData):
            self.editable_widget.set_value(None)
            self._last_input_df = None
            self.mark_error()
            return False, "Input must be TableData."
            
        df = data.df.copy()
        reset_requested = self.get_property('reset_edits')
        
        # If upstream data hasn't changed, and the user hasn't requested a reset, keep our local edits
        if self._last_input_df is not None and df.equals(self._last_input_df) and not reset_requested:
            edited_df = self.editable_widget.get_dataframe()
            if edited_df is not None:
                self.output_values['out'] = TableData(payload=edited_df)
                self.set_progress(100)
                self.mark_clean()
                return True, None
                
        # If we reach here, upstream data changed OR reset was requested OR first run
        self._last_input_df = df.copy()
        
        if reset_requested:
            self.set_property('reset_edits', False)
        
        # Push to UI
        self.editable_widget.set_value(df)
        
        # Push to output
        self.output_values['out'] = TableData(payload=df)
        
        self.set_progress(100)
        self.mark_clean()
        return True, None


class FilterTableNode(BaseExecutionNode):
    """
    Filters rows in a TableData object using a pandas query string.

    Examples:
    - `Area > 100` — keep rows where Area is greater than 100
    - `Area > 50 and Circularity > 0.8` — multiple conditions
    - `Group == "Control"` — match a specific text value
    - `Group != "Background"` — exclude rows
    - `Area > Area.mean()` — compare to column statistics
    - `label in [1, 2, 5]` — match specific values from a list

    Uses pandas `DataFrame.query()` syntax. Column names with spaces need backticks: `` `Column Name` > 10 ``

    Keywords: filter rows, query, pandas query, where clause, subset table, 篩選, 過濾, 查詢, 條件, 子集
    """
    __identifier__ = 'nodes.dataframe.Filter'
    NODE_NAME = 'Filter Table'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}
    _collection_aware = True

    def __init__(self):
        super(FilterTableNode, self).__init__(use_progress=False)
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('out', color=PORT_COLORS['table'])
        
        self.add_text_input('query', 'Query String', text='Value > 0.05', tab='Parameters')
        
    def evaluate(self):
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No input connected."
            
        up_node = in_port.connected_ports()[0].node()
        data = up_node.output_values.get(in_port.connected_ports()[0].name())
        
        if not isinstance(data, TableData):
            self.mark_error()
            return False, "Input must be TableData."
            
        query_str = self.get_property('query')
        if not query_str.strip():
            self.output_values['out'] = data
            self.mark_clean()
            return True, None
            
        try:
            # Evaluate pandas query
            filtered_df = data.df.query(query_str).copy()
            self.output_values['out'] = TableData(payload=filtered_df)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, f"Invalid filter query: {str(e)}"


class MathColumnNode(BaseExecutionNode):
    """
    Creates or modifies a column using a pandas eval expression.

    Examples:
    - `Ratio = Intensity_Ch1 / Intensity_Ch2` — create a new column
    - `Area_um2 = Area * 0.065 * 0.065` — convert pixels to physical units
    - `Normalized = Intensity / Intensity.mean()` — normalize to mean
    - `Log_Area = @np.log10(Area)` — use numpy functions with `@` prefix

    Uses pandas `DataFrame.eval()` syntax. The left side of `=` is the new column name.

    Keywords: column math, eval, formula, derived column, expression, 計算, 公式, 欄位, 衍生欄位, 運算
    """
    __identifier__ = 'nodes.dataframe.Compute'
    NODE_NAME = 'Single Table Math'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}
    _collection_aware = True

    def __init__(self):
        super(MathColumnNode, self).__init__(use_progress=False)
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('out', color=PORT_COLORS['table'])
        
        self.add_text_input('eval_str', 'Equation', text='NewCol = ColA + ColB', tab='Parameters')
        
    def evaluate(self):
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No input connected."
            
        up_node = in_port.connected_ports()[0].node()
        data = up_node.output_values.get(in_port.connected_ports()[0].name())
        
        if not isinstance(data, TableData):
            self.mark_error()
            return False, "Input must be TableData."
            
        eval_str = self.get_property('eval_str')
        if not eval_str.strip():
            self.output_values['out'] = data
            self.mark_clean()
            return True, None
            
        try:
            # Eval allows creating new columns like "A = B + C"
            new_df = data.df.copy()
            new_df.eval(eval_str, inplace=True)
            self.output_values['out'] = TableData(payload=new_df)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, f"Eval failed: {str(e)}"


class AggregateTableNode(BaseExecutionNode):
    """
    Reduces a table to aggregate statistics across rows, optionally grouped by column.

    Without grouping, all numeric columns are reduced to a single row.
    With grouping, each unique group gets its own summary row.

    | Group   | Area |
    |---------|------|
    | Control | 110  |
    | Treated | 190  |

    Parameters:
    - **Operation** — sum, mean, median, min, max, count, std, var, auc
    - **Group By** — column name(s) to group by (comma-separated, leave empty for no grouping)
    - **Columns** — restrict to specific columns (comma-separated, leave empty = all numeric)
    - **Sort By** — column to sort by before computing AUC (required for auc, e.g. time column)

    The **auc** operation computes the area under the curve using the trapezoidal
    rule.  Rows are first sorted by the *Sort By* column, which serves as the
    x-axis (e.g. time).  Each selected numeric column is then integrated against
    that x-axis.

    Keywords: aggregate, groupby, mean, sum, median, auc, area under curve, trapezoidal, 聚合, 分組, 計算, 平均值, 統計
    """
    __identifier__ = 'nodes.dataframe.Compute'
    NODE_NAME = 'Aggregate Table'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}
    _collection_aware = True
    OUTPUT_COLUMNS = {
        'out': {
            'no_group':   ['stat', '...numeric columns (one aggregated value each)'],
            'with_group': ['group_by_col', '...numeric columns (one value per group)'],
        }
    }

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('out', color=PORT_COLORS['table'])
        self.add_combo_menu('operation', 'Operation',
                            items=['sum', 'mean', 'median', 'min', 'max', 'count', 'std', 'var', 'auc'])
        self._add_column_selector('group_by', 'Group By (optional)', text='', mode='single', tab='Parameters')
        self._add_column_selector('columns', 'Columns (A,B)', text='', mode='multi', tab='Parameters')
        self._add_column_selector('sort_by', 'Sort By (for AUC)', text='', mode='single', tab='Parameters')

    def evaluate(self):
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No input connected."

        cp = in_port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if not isinstance(data, TableData):
            self.mark_error()
            return False, "Input must be TableData."

        self._refresh_column_selectors(data.df, 'group_by', 'columns', 'sort_by')

        df = data.df.copy()
        operation   = self.get_property('operation') or 'sum'
        group_by_str = str(self.get_property('group_by') or '').strip()
        columns_str  = str(self.get_property('columns')  or '').strip()
        sort_by_str  = str(self.get_property('sort_by')  or '').strip()

        group_cols = [c.strip() for c in group_by_str.split(',') if c.strip() and c.strip() in df.columns] if group_by_str else []

        if columns_str:
            agg_cols = [c.strip() for c in columns_str.split(',')
                        if c.strip() in df.columns and pd.api.types.is_numeric_dtype(df[c.strip()])]
        else:
            agg_cols = [c for c in df.select_dtypes(include='number').columns
                        if c not in group_cols and c != sort_by_str]

        if not agg_cols:
            self.mark_error()
            return False, "No numeric columns to aggregate."

        try:
            if operation == 'auc':
                # AUC via trapezoidal rule — requires a sort column as x-axis
                sort_col = sort_by_str if sort_by_str in df.columns else None
                if sort_col is None:
                    # Fall back to first numeric column not in agg_cols
                    for c in df.select_dtypes(include='number').columns:
                        if c not in agg_cols and c not in group_cols:
                            sort_col = c
                            break
                if sort_col is None:
                    self.mark_error()
                    return False, "AUC requires a Sort By column (e.g. time)."

                import numpy as np

                def _trapz_auc(sub_df):
                    sub_sorted = sub_df.sort_values(sort_col)
                    x = sub_sorted[sort_col].to_numpy(dtype=float)
                    row = {}
                    for col in agg_cols:
                        y = sub_sorted[col].to_numpy(dtype=float)
                        mask = ~(np.isnan(x) | np.isnan(y))
                        row[col] = float(np.trapz(y[mask], x[mask])) if mask.sum() >= 2 else float('nan')
                    return pd.Series(row)

                if group_cols:
                    result = df.groupby(group_cols).apply(_trapz_auc, include_groups=False).reset_index()
                else:
                    auc_row = _trapz_auc(df)
                    result = auc_row.to_frame().T.reset_index(drop=True)
                    result.insert(0, 'stat', 'auc')
            else:
                if group_cols:
                    result = df.groupby(group_cols)[agg_cols].agg(operation).reset_index()
                else:
                    agg_series = df[agg_cols].agg(operation)   # Series: index=col_names
                    result = agg_series.to_frame(name=operation).T.reset_index(drop=True)
                    result.insert(0, 'stat', operation)

            self.output_values['out'] = TableData(payload=result)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)


class RenameGroupMappingWidget(NodeBaseWidget):
    """
    A custom node widget providing a 2-column table to map Original names to New names.
    Updates internally to a JSON property when edited.
    """
    mapping_edited = QtCore.Signal(dict)
    update_table_signal = QtCore.Signal(dict)
    
    def __init__(self, parent=None):
        super(RenameGroupMappingWidget, self).__init__(parent, name='group_mapping_table', label='Group Mapping')
        
        self._table = QtWidgets.QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(['Original Name', 'New Name'])
        
        # Style
        self._table.setStyleSheet("""
            QTableWidget { background-color: #222; color: #ddd; gridline-color: #444; border: 1px solid #555; }
            QHeaderView::section { background-color: #333; color: #fff; padding: 4px; border: 1px solid #444; }
        """)
        self._table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        
        self._is_updating = False
        self._table.itemChanged.connect(self._on_item_changed)
        
        self.update_table_signal.connect(self._sync_table_ui, QtCore.Qt.QueuedConnection)
        
        # Container
        self._container = QtWidgets.QWidget()
        self._layout = QtWidgets.QVBoxLayout(self._container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._table)
        
        self._table.setMinimumHeight(120)
        self._table.setMaximumHeight(200)
        
        self.set_custom_widget(self._container)
        
    def _sync_table_ui(self, mapping_dict):
        self._is_updating = True
        self._table.setRowCount(0)
        
        for k, v in mapping_dict.items():
            row = self._table.rowCount()
            self._table.insertRow(row)
            
            # Key (Read Only)
            item_k = QtWidgets.QTableWidgetItem(str(k))
            item_k.setFlags(item_k.flags() ^ QtCore.Qt.ItemFlag.ItemIsEditable)
            item_k.setBackground(QtGui.QColor("#2a2a2a"))
            self._table.setItem(row, 0, item_k)
            
            # Value (Editable)
            item_v = QtWidgets.QTableWidgetItem(str(v) if v else "")
            self._table.setItem(row, 1, item_v)
            
        self._is_updating = False
        self.mapping_edited.emit(mapping_dict)
        
    def _on_item_changed(self, item):
        if self._is_updating: return
        
        mapping = {}
        for row in range(self._table.rowCount()):
            k_item = self._table.item(row, 0)
            v_item = self._table.item(row, 1)
            if k_item:
                k = k_item.text().strip()
                v = v_item.text().strip() if v_item else ""
                mapping[k] = v
                
        self.mapping_edited.emit(mapping)

    def get_value(self):
        return ""
        
    def set_value(self, value):
        pass


class RenameGroupNode(BaseExecutionNode):
    """
    Renames values in a target column based on a mapping string or the built-in mapping table.

    Mapping syntax:
    - `OldName : NewName` — rename a single value
    - `OldA | OldB : Combined` — merge multiple values into one
    - Comma-separated for multiple rules: `A : Control, B : Treated`

    Example: Target Column = `Group`, Mapping = `ctrl : Control, exp1 | exp2 : Experimental`

    | Group (before) | Group (after)  |
    |----------------|----------------|
    | ctrl           | Control        |
    | exp1           | Experimental   |
    | exp2           | Experimental   |

    You can also use the mapping table widget below the text input for a visual editor.

    Keywords: rename groups, recode labels, map values, category rename, group relabel, 重新命名, 群組, 對應, 分類, 標籤
    """
    __identifier__ = 'nodes.dataframe.Transform'
    NODE_NAME = 'Rename Group'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}
    
    def __init__(self):
        super(RenameGroupNode, self).__init__(use_progress=False)
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('out', color=PORT_COLORS['table'])
        
        self._add_column_selector('target_column', 'Target Column', text='Group', mode='single', tab='Parameters')
        self.add_text_input('mapping', 'Mapping ( Old:New, A|B )', text='', tab='Parameters')
        
        self.create_property('mapping_json', '{}', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value)
        
        self._mapping_widget = RenameGroupMappingWidget(self.view)
        self._mapping_widget.mapping_edited.connect(self._on_table_edited)
        self.add_custom_widget(self._mapping_widget)
        
    def _on_table_edited(self, new_mapping):
        self.set_property('mapping_json', json.dumps(new_mapping))
        
    def update(self):
        super().update()
        try:
            m = json.loads(self.get_property('mapping_json'))
            self._mapping_widget.update_table_signal.emit(m)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        
    def evaluate(self):
        self.reset_progress()
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No input connected."

        up_node = in_port.connected_ports()[0].node()
        data = up_node.output_values.get(in_port.connected_ports()[0].name())

        if not isinstance(data, TableData):
            self.mark_error()
            return False, "Input must be TableData."

        self._refresh_column_selectors(data.df, 'target_column')

        target_col = self.get_property('target_column').strip()
        df = data.df.copy()

        if target_col not in df.columns:
            self.mark_error()
            return False, f"Column '{target_col}' not found in input data."
            
        # 1. Update JSON map safely from incoming unique values
        try:
            current_mapping = json.loads(self.get_property('mapping_json'))
        except (json.JSONDecodeError, TypeError, ValueError):
            current_mapping = {}

        unique_vals_set = set(str(v) for v in df[target_col].dropna().unique())
        
        new_mapping = {}
        mapping_changed = False
        
        for val_str in unique_vals_set:
            if val_str in current_mapping:
                new_mapping[val_str] = current_mapping[val_str]
            else:
                new_mapping[val_str] = ""
                mapping_changed = True
                
        # Verify if any old keys existed that aren't in the new set
        if len(new_mapping) != len(current_mapping):
            mapping_changed = True
            
        current_mapping = new_mapping
                
        if mapping_changed:
            # Emit to main thread. The UI update will emit mapping_edited, updating the property safely
            self._mapping_widget.update_table_signal.emit(current_mapping)
            
        # 2. Extract final replace dict
        # First from UI Table (fall back to original if empty)
        replace_dict = {}
        for k, v in current_mapping.items():
            if v:
                replace_dict[k] = v
                
        # Then override with string mapping
        mapping_str = self.get_property('mapping').strip()
        if mapping_str:
            pairs = [p.strip() for p in mapping_str.split(',') if p.strip()]
            for pair in pairs:
                if ':' in pair: old, new = pair.split(':', 1)
                elif '|' in pair: old, new = pair.split('|', 1)
                else: continue
                replace_dict[old.strip()] = new.strip()
                
        # 3. Apply mapping
        if replace_dict:
            # We enforce astype(str) temporarily to ensure string keys match
            df[target_col] = df[target_col].astype(str).replace(replace_dict)
            
        self.output_values['out'] = TableData(payload=df)
        self.mark_clean()
        self.set_progress(100)
        return True, None


class ReshapeTableNode(BaseExecutionNode):
    """
    Converts a table between wide and long format.

    Modes:
    - *Wide to Long* (melt) — unpivots multiple value columns into rows
    - *Long to Wide* (pivot) — spreads row values back into columns
    - *Collect by Group* — gathers values by group into side-by-side columns

    **Wide to Long example:**

    | Sample | Ch1 | Ch2 | Ch3 |
    |--------|-----|-----|-----|
    | A      | 10  | 20  | 30  |
    | B      | 15  | 25  | 35  |

    Settings: ID Columns = `Sample`, Group Column = `Channel`, Value Column = `Intensity`

    | Sample | Channel | Intensity |
    |--------|---------|-----------|
    | A      | Ch1     | 10        |
    | A      | Ch2     | 20        |
    | A      | Ch3     | 30        |
    | B      | Ch1     | 15        |
    | B      | Ch2     | 25        |
    | B      | Ch3     | 35        |

    **Long to Wide** reverses the above. Settings: Index Columns = `Sample`, Pivot Column = `Channel`, Value Column = `Intensity`

    Parameters:
    - **ID Columns** — columns to keep as-is (comma-separated). Leave empty to melt all non-numeric columns.
    - **Value Columns** — which columns to unpivot (leave empty = all remaining).
    - **Group Column Name** — name for the new column holding the original column names (default: `Group`).
    - **Value Column Name** — name for the new column holding the values (default: `Value`).
    - **Index Columns (pivot)** — columns that identify each row in the wide output.
    - **Pivot Column (pivot)** — column whose unique values become new column headers.
    - **Value Column (pivot)** — column whose values fill the new columns.

    Keywords: reshape, melt, pivot, wide to long, long to wide, 重塑, 轉置, 樞紐分析, 寬轉長, 長轉寬
    """
    __identifier__ = 'nodes.dataframe.Transform'
    NODE_NAME = 'Reshape Table'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}
    _collection_aware = True

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('out', color=PORT_COLORS['table'])
        self.add_combo_menu('mode', 'Mode', items=['Wide → Long', 'Long → Wide', 'Collect by Group'])
        # Melt params
        self._add_column_selector('id_vars',    'ID Columns (empty = melt all)', text='', mode='multi', tab='Parameters')
        self._add_column_selector('value_vars', 'Value Columns (optional)',      text='', mode='multi', tab='Parameters')
        self.add_text_input('var_name',   'New Group Column Name',         text='Group', tab='Parameters')
        self.add_text_input('value_name', 'New Value Column Name',         text='Value', tab='Parameters')
        # Pivot params
        self._add_column_selector('index_columns', 'Index Columns (pivot)',  text='', mode='multi', tab='Parameters')
        self._add_column_selector('pivot_column',  'Pivot Column (pivot)',   text='Group', mode='single', tab='Parameters')
        self._add_column_selector('value_column',  'Value Column (pivot)',   text='Value', mode='single', tab='Parameters')

    def evaluate(self):
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No input connected."

        up_val = in_port.connected_ports()[0].node().output_values.get(
            in_port.connected_ports()[0].name())
        if isinstance(up_val, TableData):
            df = up_val.df.copy()
        elif isinstance(up_val, pd.DataFrame):
            df = up_val.copy()
        else:
            self.mark_error()
            return False, "Input must be TableData or DataFrame."

        self._refresh_column_selectors(df, 'id_vars', 'value_vars', 'index_columns', 'pivot_column', 'value_column')

        mode = str(self.get_property('mode') or 'Wide → Long')

        try:
            if mode.startswith('Wide'):
                id_vars_str  = str(self.get_property('id_vars')    or '').strip()
                val_vars_str = str(self.get_property('value_vars') or '').strip()
                var_name     = str(self.get_property('var_name')   or 'Group').strip() or 'Group'
                value_name   = str(self.get_property('value_name') or 'Value').strip() or 'Value'

                id_cols  = [c.strip() for c in id_vars_str.split(',')  if c.strip() and c.strip() in df.columns] if id_vars_str  else []
                val_cols = [c.strip() for c in val_vars_str.split(',') if c.strip() and c.strip() in df.columns] if val_vars_str else None

                result = pd.melt(df, id_vars=id_cols, value_vars=val_cols,
                                 var_name=var_name, value_name=value_name).dropna(subset=[value_name])
            elif mode.startswith('Long'):
                idx_str    = str(self.get_property('index_columns') or '').strip()
                pivot_col  = str(self.get_property('pivot_column')  or 'Group').strip()
                val_col    = str(self.get_property('value_column')  or 'Value').strip()

                idx_cols = [c.strip() for c in idx_str.split(',') if c.strip()] if idx_str else []
                missing  = [c for c in idx_cols + [pivot_col, val_col] if c not in df.columns]
                if missing:
                    self.mark_error()
                    return False, f"Columns not found: {', '.join(missing)}"

                result = df.pivot_table(index=idx_cols or None, columns=pivot_col,
                                        values=val_col).reset_index()
                result.columns.name = None

            else:  # Collect by Group
                pivot_col = str(self.get_property('pivot_column') or 'Group').strip()
                val_col   = str(self.get_property('value_column') or 'Value').strip()
                missing   = [c for c in [pivot_col, val_col] if c not in df.columns]
                if missing:
                    self.mark_error()
                    return False, f"Columns not found: {', '.join(missing)}"

                groups = df[pivot_col].dropna().unique()
                series = {str(g): df.loc[df[pivot_col] == g, val_col].reset_index(drop=True)
                          for g in groups}
                result = pd.concat(series, axis=1)
                result.columns.name = None

            self.output_values['out'] = TableData(payload=result)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)


class SortTableNode(BaseExecutionNode):
    """
    Sorts a table by one or more columns in ascending or descending order.

    Parameters:
    - **Sort By** — comma-separated column names to sort by (applied in order)
    - **Order** — ascending or descending

    Keywords: sort, order rows, ascending, descending, rank table, 排序, 遞增, 遞減, 排列, 資料表
    """
    __identifier__ = 'nodes.dataframe.Transform'
    NODE_NAME = 'Sort Table'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}
    _collection_aware = True

    def __init__(self):
        super(SortTableNode, self).__init__(use_progress=False)
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('out', color=PORT_COLORS['table'])

        self._add_column_selector('sort_by', 'Sort By (comma-separated columns)', text='', mode='multi', tab='Parameters')
        self.add_combo_menu('order', 'Order', items=['Ascending', 'Descending'], tab='Parameters')

    def evaluate(self):
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No input connected"

        up_val = in_port.connected_ports()[0].node().output_values.get(
            in_port.connected_ports()[0].name()
        )
        if isinstance(up_val, TableData):
            df = up_val.df.copy()
        elif isinstance(up_val, pd.DataFrame):
            df = up_val.copy()
        else:
            self.mark_error()
            return False, "Input must be TableData or DataFrame"

        self._refresh_column_selectors(df, 'sort_by')

        sort_str = str(self.get_property('sort_by')).strip()
        ascending = self.get_property('order') != 'Descending'

        if not sort_str:
            # No sort columns specified — pass through unchanged
            self.output_values['out'] = TableData(payload=df)
            self.mark_clean()
            return True, None

        cols = [c.strip() for c in sort_str.split(',') if c.strip() in df.columns]
        if not cols:
            self.mark_error()
            return False, f"None of the specified columns exist. Available: {list(df.columns)}"

        try:
            df_sorted = df.sort_values(by=cols, ascending=ascending).reset_index(drop=True)
            self.output_values['out'] = TableData(payload=df_sorted)
            self.mark_clean()
            self.set_progress(100)
            return True, None
        except Exception as e:
            self.mark_error()
            return False, f"Sort failed: {str(e)}"


class TopNNode(BaseExecutionNode):
    """
    Extracts the top (or bottom) N rows ranked by a numeric column.

    Outputs:
    - **top_n** — the selected N rows
    - **rest** — all remaining rows not in top_n

    Parameters:
    - **Rank By Column** — numeric column to rank by
    - **N** — number of rows to select
    - **Select** — *Top (largest)* or *Bottom (smallest)*

    Keywords: top n, bottom n, ranking, largest, smallest, 排序, 前幾名, 最大值, 最小值, 篩選
    """
    __identifier__ = 'nodes.dataframe.Filter'
    NODE_NAME = 'Top N'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table', 'table']}
    _collection_aware = True

    def __init__(self):
        super(TopNNode, self).__init__(use_progress=False)
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('top_n', color=PORT_COLORS['table'])
        self.add_output('rest', color=PORT_COLORS['table'])

        self._add_column_selector('column', 'Rank By Column', text='', mode='single', tab='Parameters')
        self._add_int_spinbox('n', 'N (number of rows)', value=5, min_val=1, max_val=100000, tab='Parameters')
        self.add_combo_menu('order', 'Select', items=['Top (largest)', 'Bottom (smallest)'], tab='Parameters')

    def evaluate(self):
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No input connected"

        up_val = in_port.connected_ports()[0].node().output_values.get(
            in_port.connected_ports()[0].name()
        )
        if isinstance(up_val, TableData):
            df = up_val.df.copy()
        elif isinstance(up_val, pd.DataFrame):
            df = up_val.copy()
        else:
            self.mark_error()
            return False, "Input must be TableData or DataFrame"

        self._refresh_column_selectors(df, 'column')

        col = str(self.get_property('column')).strip()
        n = max(1, int(self.get_property('n')))
        largest = self.get_property('order') != 'Bottom (smallest)'

        if not col:
            self.mark_error()
            return False, "Please specify a column to rank by."

        if col not in df.columns:
            self.mark_error()
            return False, f"Column '{col}' not found. Available: {list(df.columns)}"

        try:
            if largest:
                top_idx = df[col].nlargest(n).index
            else:
                top_idx = df[col].nsmallest(n).index

            df_top = df.loc[top_idx].reset_index(drop=True)
            df_rest = df.drop(index=top_idx).reset_index(drop=True)

            self.output_values['top_n'] = TableData(payload=df_top)
            self.output_values['rest'] = TableData(payload=df_rest)
            self.mark_clean()
            self.set_progress(100)
            return True, None
        except Exception as e:
            self.mark_error()
            return False, f"Top-N failed: {str(e)}"


class ColumnValueSplitNode(BaseExecutionNode):
    """
    Splits a table into two outputs based on whether a column's value matches a list of specified values.

    **Values** — comma-separated. `*` anywhere triggers glob matching:
    - `Control*` — starts with "Control"
    - `*treated` — ends with "treated"
    - `*GFP*` — contains "GFP"
    - Entries without `*` are exact matches

    Outputs:
    - **matched** — rows where the column value matches any entry
    - **rest** — all other rows

    Keywords: split table, match values, wildcard filter, include/exclude groups, partition rows, 分割, 篩選, 萬用字元, 分組, 資料表
    """
    __identifier__ = 'nodes.dataframe.Filter'
    NODE_NAME = 'Column Value Split'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table', 'table']}
    _collection_aware = True

    def __init__(self):
        super(ColumnValueSplitNode, self).__init__(use_progress=False)
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('matched', color=PORT_COLORS['table'])
        self.add_output('rest', color=PORT_COLORS['table'])

        self._add_column_selector('column', 'Column', text='', mode='single', tab='Parameters')
        self.add_text_input('values', 'Values (comma-separated)', text='', tab='Parameters')
        self.add_checkbox('case_sensitive', '', text='Case Sensitive', state=True, tab='Parameters')

    def evaluate(self):
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No input connected"

        up_val = in_port.connected_ports()[0].node().output_values.get(
            in_port.connected_ports()[0].name()
        )
        if isinstance(up_val, TableData):
            df = up_val.df.copy()
        elif isinstance(up_val, pd.DataFrame):
            df = up_val.copy()
        else:
            self.mark_error()
            return False, "Input must be TableData or DataFrame"

        self._refresh_column_selectors(df, 'column')

        col = str(self.get_property('column')).strip()
        values_str = str(self.get_property('values')).strip()
        case_sens = bool(self.get_property('case_sensitive'))

        if not col:
            self.mark_error()
            return False, "Please specify a column to filter on."
        if col not in df.columns:
            self.mark_error()
            return False, f"Column '{col}' not found. Available: {list(df.columns)}"
        if not values_str:
            self.mark_error()
            return False, "Please specify at least one value to match."

        patterns = [v.strip() for v in values_str.split(',') if v.strip()]
        if not case_sens:
            patterns = [p.lower() for p in patterns]

        def _matches(cell: str) -> bool:
            s = cell if case_sens else cell.lower()
            return any(
                fnmatch.fnmatch(s, p) if '*' in p else s == p
                for p in patterns
            )

        try:
            col_series = df[col].astype(str)
            mask = col_series.apply(_matches)

            df_matched = df[mask].reset_index(drop=True)
            df_rest = df[~mask].reset_index(drop=True)

            self.output_values['matched'] = TableData(payload=df_matched)
            self.output_values['rest'] = TableData(payload=df_rest)
            self.mark_clean()
            self.set_progress(100)
            return True, None
        except Exception as e:
            self.mark_error()
            return False, f"Split failed: {str(e)}"


class TwoTableMathNode(BaseExecutionNode):
    """
    Computes a scalar arithmetic operation between one value from each of two input tables.

    For each input table the node picks the first numeric column (or the column
    named in the matching property) and uses row 0 as the scalar value.
    Designed for comparing scalar outputs such as stained-area measurements.

    Outputs a single-row result table: `left_value | right_value | operation | result`

    Parameters:
    - **Operation** — `left / right`, `left * right`, `left + right`, or `left - right`
    - **Left Column** — column name in the left table (blank = first numeric)
    - **Right Column** — column name in the right table (blank = first numeric)

    Keywords: two-table math, ratio, divide tables, scalar compare, combine metrics, 計算, 比值, 兩表, 比較, 合併
    """
    __identifier__ = 'nodes.dataframe.Compute'
    NODE_NAME = 'Two Table Math'
    PORT_SPEC = {'inputs': ['table', 'table'], 'outputs': ['table']}
    _collection_aware = True
    OUTPUT_COLUMNS = {
        'result': ['left_value', 'right_value', 'operation', 'result']
    }

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('left',  color=PORT_COLORS['table'])
        self.add_input('right', color=PORT_COLORS['table'])
        self.add_output('result', color=PORT_COLORS['table'])
        self.add_combo_menu('operation', 'Operation', items=['left / right', 'left * right', 'left + right', 'left - right'])
        self._add_column_selector('left_column',  'Left Column  (blank = first numeric)',  text='', mode='single')
        self._add_column_selector('right_column', 'Right Column (blank = first numeric)', text='', mode='single')
 
    def _extract_scalar(self, data, col_name: str):
        """Return (float_value, resolved_col_name) from a TableData."""
        if isinstance(data, TableData):
            df = data.df
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError(f"Expected TableData, got {type(data).__name__}")

        if col_name and col_name in df.columns:
            col = col_name
        else:
            num_cols = df.select_dtypes(include='number').columns.tolist()
            if not num_cols:
                raise ValueError("No numeric column found in table")
            col = num_cols[0]

        series = df[col].dropna()
        if series.empty:
            raise ValueError(f"Column '{col}' contains no non-NaN values")

        return float(series.iloc[0]), col

    def evaluate(self):
        left_port  = self.inputs().get('left')
        right_port = self.inputs().get('right')

        if not left_port or not left_port.connected_ports():
            self.mark_error()
            return False, "Left input not connected"
        if not right_port or not right_port.connected_ports():
            self.mark_error()
            return False, "Right input not connected"

        lcp = left_port.connected_ports()[0]
        rcp = right_port.connected_ports()[0]
        left_data  = lcp.node().output_values.get(lcp.name())
        right_data = rcp.node().output_values.get(rcp.name())

        _ldf = left_data.df if isinstance(left_data, TableData) else (left_data if isinstance(left_data, pd.DataFrame) else None)
        if _ldf is not None:
            self._refresh_column_selectors(_ldf, 'left_column', 'right_column')

        left_col_hint  = str(self.get_property('left_column')  or '').strip()
        right_col_hint = str(self.get_property('right_column') or '').strip()
        operation = str(self.get_property('operation') or 'left / right')

        try:
            lv, lcol = self._extract_scalar(left_data,  left_col_hint)
            rv, rcol = self._extract_scalar(right_data, right_col_hint)

            if   'left / right' in operation:
                if rv == 0:
                    raise ZeroDivisionError("Right value is 0")
                result = lv / rv
                op_str = '/'
            elif 'left * right' in operation:
                result = lv * rv
                op_str = '*'
            elif 'left + right' in operation:
                result = lv + rv
                op_str = '+'
            else:
                result = lv - rv
                op_str = '-'

            out_df = pd.DataFrame([{
                'left_value':  lv,
                'right_value': rv,
                'operation':   f"{lcol} {op_str} {rcol}",
                'result':      result,
            }])
            self.output_values['result'] = TableData(payload=out_df)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)


class SelectColumnsNode(BaseExecutionNode):
    """
    Keeps only the columns listed in 'Columns' and drops everything else.

    **Columns** — comma-separated list of column names to keep. `*` anywhere in a name triggers glob matching:
    - `*Intensity` — ends with "Intensity"
    - `Intensity*` — starts with "Intensity"
    - `*Intensity*` — contains "Intensity"

    **Drop mode** — when checked, the listed columns are DROPPED instead of kept.

    Keywords: select columns, drop columns, keep subset, wildcard columns, schema trim, 選擇欄位, 刪除欄位, 子集, 萬用字元, 資料表
    """
    __identifier__ = 'nodes.dataframe.Transform'
    NODE_NAME = 'Select Columns'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}
    _collection_aware = True

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('in',  color=PORT_COLORS['table'])
        self.add_output('out', color=PORT_COLORS['table'])
        self._add_column_selector('columns', 'Columns (comma-separated)', text='', mode='multi')
        self.add_checkbox('drop_mode', '', text='Drop listed columns instead', state=False)

    def evaluate(self):
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No input connected"

        cp = in_port.connected_ports()[0]
        up_val = cp.node().output_values.get(cp.name())
        if isinstance(up_val, TableData):
            df = up_val.df.copy()
        elif isinstance(up_val, pd.DataFrame):
            df = up_val.copy()
        else:
            self.mark_error()
            return False, "Input must be TableData or DataFrame"

        self._refresh_column_selectors(df, 'columns')

        cols_str = str(self.get_property('columns') or '').strip()
        drop_mode = bool(self.get_property('drop_mode'))

        if not cols_str:
            self.mark_error()
            return False, "Please specify at least one column."

        entries = [c.strip() for c in cols_str.split(',') if c.strip()]

        # Resolve each entry — '*' anywhere triggers glob matching
        resolved: list[str] = []
        for entry in entries:
            if '*' in entry:
                resolved.extend(c for c in df.columns if fnmatch.fnmatch(c, entry))
            else:
                resolved.append(entry)

        # Deduplicate while preserving order
        seen: set[str] = set()
        resolved = [c for c in resolved if not (c in seen or seen.add(c))]

        missing = [c for c in resolved if c not in df.columns]
        if missing:
            self.mark_error()
            return False, f"Column(s) not found: {missing}. Available: {list(df.columns)}"

        try:
            if drop_mode:
                out_df = df.drop(columns=resolved)
            else:
                out_df = df[resolved]

            self.output_values['out'] = TableData(payload=out_df)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, f"Select columns failed: {str(e)}"


class ExtractObjectNode(BaseExecutionNode):
    """
    Extracts a single object (image, figure, label, etc.) from a table's object column.

    After batch-accumulating images or figures, this node lets you pick one
    item by row index and outputs it as its original data type.

    Parameters:
    - **Row Index** — 1-based row number to extract from
    - **Object Column** — name of the column containing the objects (default: `object`)

    Keywords: extract, pick, select, object, image, figure, row, index, 提取, 選取
    """
    __identifier__ = 'nodes.dataframe.Util'
    NODE_NAME = 'Extract Object'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['any']}
    _collection_aware = True

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('in', color=PORT_COLORS.get('table'))
        self.add_output('out', multi_output=True, color=PORT_COLORS.get('any'))

        self._add_int_spinbox('row_index', 'Row Index (1-based)', value=1, min_val=1, max_val=1000000, tab='Parameters')
        self._add_column_selector('object_column', 'Object Column', text='object', mode='single')

    def evaluate(self):
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No input connected"

        cp = in_port.connected_ports()[0]
        up_val = cp.node().output_values.get(cp.name())
        if isinstance(up_val, TableData):
            df = up_val.df
        elif isinstance(up_val, pd.DataFrame):
            df = up_val
        else:
            self.mark_error()
            return False, "Input must be TableData or DataFrame"

        self._refresh_column_selectors(df, 'object_column')

        col = self.get_property('object_column') or 'object'
        if col not in df.columns:
            self.mark_error()
            return False, f"Column '{col}' not found. Available: {list(df.columns)}"

        idx = max(1, int(self.get_property('row_index'))) - 1  # 1-based → 0-based
        if idx >= len(df):
            self.mark_error()
            return False, f"Row {idx+1} out of range (table has {len(df)} rows)"

        obj = df.iloc[idx][col]
        self.output_values['out'] = obj
        self.mark_clean()
        return True, None


class RandomSampleNode(BaseExecutionNode):
    """
    Randomly samples N rows from the input table.

    If N exceeds the table size, the full table is returned (no error).

    Parameters:
    - **N** — number of rows to draw
    - **Seed** — random seed for reproducibility; leave at `-1` for a different sample each run

    Outputs:
    - **sampled** — the N randomly selected rows
    - **rest** — all remaining rows not in the sample

    Keywords: random sample, subsample, shuffle, draw, pick, 隨機, 抽樣
    """
    __identifier__ = 'nodes.dataframe.Filter'
    NODE_NAME = 'Random Sample'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table', 'table']}
    _collection_aware = True

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('sampled', color=PORT_COLORS['table'])
        self.add_output('rest', color=PORT_COLORS['table'])

        self._add_int_spinbox('n', 'N (rows to sample)', value=10, min_val=1, max_val=1000000, tab='Parameters')
        self._add_int_spinbox('seed', 'Random Seed (-1=random)', value=-1, min_val=-1, max_val=9999999, tab='Parameters')

    def evaluate(self):
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No input connected"

        cp = in_port.connected_ports()[0]
        up_val = cp.node().output_values.get(cp.name())
        if isinstance(up_val, TableData):
            df = up_val.df.copy()
        elif isinstance(up_val, pd.DataFrame):
            df = up_val.copy()
        else:
            self.mark_error()
            return False, "Input must be TableData or DataFrame"

        n = max(1, int(self.get_property('n')))
        seed_val = int(self.get_property('seed'))
        rng = seed_val if seed_val > 0 else None

        n = min(n, len(df))

        try:
            sampled_idx = df.sample(n=n, random_state=rng).index
            df_sampled = df.loc[sampled_idx].reset_index(drop=True)
            df_rest = df.drop(index=sampled_idx).reset_index(drop=True)

            self.output_values['sampled'] = TableData(payload=df_sampled)
            self.output_values['rest'] = TableData(payload=df_rest)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, f"Random sample failed: {str(e)}"


# ---------------------------------------------------------------------------
# High-priority new nodes
# ---------------------------------------------------------------------------

class ConcatTablesNode(BaseExecutionNode):
    """
    Concatenates two tables by stacking rows (vertical) or columns (horizontal).

    Modes:
    - *Vertical (stack rows)* — both tables should have the same columns; mismatched columns are filled with NaN when 'Fill missing columns' is checked
    - *Horizontal (side by side)* — both tables are placed side by side; shorter side is padded with NaN

    Parameters:
    - **Direction** — vertical or horizontal
    - **Fill missing columns with NaN** — when unchecked, only common columns are kept in vertical mode

    Keywords: concat, append, stack rows, merge rows, combine tables, 合併, 連接, 堆疊, 附加, 資料表
    """
    __identifier__ = 'nodes.dataframe.Combine'
    NODE_NAME = 'Concat Tables'
    PORT_SPEC = {'inputs': ['table', 'table'], 'outputs': ['table']}
    _collection_aware = True

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('top',    color=PORT_COLORS['table'])
        self.add_input('bottom', color=PORT_COLORS['table'])
        self.add_output('out',   color=PORT_COLORS['table'])
        self.add_combo_menu('axis', 'Direction',
                            items=['Vertical (stack rows)', 'Horizontal (side by side)'])
        self.add_checkbox('fill_missing', '', text='Fill missing columns with NaN', state=True)

    def _get_df(self, port_name):
        port = self.inputs().get(port_name)
        if not port or not port.connected_ports():
            return None
        cp = port.connected_ports()[0]
        val = cp.node().output_values.get(cp.name())
        if isinstance(val, TableData):
            return val.df.copy()
        if isinstance(val, pd.DataFrame):
            return val.copy()
        return None

    def evaluate(self):
        df_top    = self._get_df('top')
        df_bottom = self._get_df('bottom')

        if df_top is None and df_bottom is None:
            self.mark_error()
            return False, "No inputs connected"

        axis_str     = str(self.get_property('axis') or 'Vertical (stack rows)')
        fill_missing = bool(self.get_property('fill_missing'))
        horizontal   = axis_str.startswith('Horizontal')

        try:
            if horizontal:
                dfs = [d for d in (df_top, df_bottom) if d is not None]
                result = pd.concat(dfs, axis=1, ignore_index=False).reset_index(drop=True)
            else:
                dfs = [d for d in (df_top, df_bottom) if d is not None]
                result = pd.concat(dfs, axis=0,
                                   ignore_index=True,
                                   sort=False).reset_index(drop=True)
                if not fill_missing:
                    # Drop columns not present in both
                    if df_top is not None and df_bottom is not None:
                        common = [c for c in df_top.columns if c in df_bottom.columns]
                        result = result[common]

            self.output_values['out'] = TableData(payload=result)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)


class JoinTablesNode(BaseExecutionNode):
    """
    Merges two tables on a shared key column (like SQL JOIN).

    Example: Left key = `particle_id`, Right key = `id` to match particles to metadata.

    Parameters:
    - **Key Column (left)** — column name to join on in the left table
    - **Key Column (right)** — column name in the right table (leave blank to use the same name as left)
    - **Join Type** — *inner*, *left*, *right*, or *outer*

    Keywords: join, merge, SQL join, inner join, left join, lookup, 合併, 連接, 鍵值, 對應, 資料表
    """
    __identifier__ = 'nodes.dataframe.Combine'
    NODE_NAME = 'Join Tables'
    PORT_SPEC = {'inputs': ['table', 'table'], 'outputs': ['table']}
    _collection_aware = True

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('left',  color=PORT_COLORS['table'])
        self.add_input('right', color=PORT_COLORS['table'])
        self.add_output('out',  color=PORT_COLORS['table'])
        self.add_text_input('left_key',  'Key Column (left)',              text='id')
        self.add_text_input('right_key', 'Key Column (right, blank=same)', text='')
        self.add_combo_menu('how', 'Join Type',
                            items=['inner', 'left', 'right', 'outer'])

    def _get_df(self, port_name):
        port = self.inputs().get(port_name)
        if not port or not port.connected_ports():
            return None
        cp = port.connected_ports()[0]
        val = cp.node().output_values.get(cp.name())
        if isinstance(val, TableData):
            return val.df.copy()
        if isinstance(val, pd.DataFrame):
            return val.copy()
        return None

    def evaluate(self):
        df_left  = self._get_df('left')
        df_right = self._get_df('right')

        if df_left is None:
            self.mark_error(); return False, "Left input not connected"
        if df_right is None:
            self.mark_error(); return False, "Right input not connected"

        lk  = str(self.get_property('left_key')  or '').strip()
        rk  = str(self.get_property('right_key') or '').strip() or lk
        how = str(self.get_property('how') or 'inner')

        if not lk:
            self.mark_error(); return False, "Key Column (left) is required"
        if lk not in df_left.columns:
            self.mark_error()
            return False, f"Column '{lk}' not found in left table. Available: {list(df_left.columns)}"
        if rk not in df_right.columns:
            self.mark_error()
            return False, f"Column '{rk}' not found in right table. Available: {list(df_right.columns)}"

        try:
            result = pd.merge(df_left, df_right, left_on=lk, right_on=rk, how=how)
            self.output_values['out'] = TableData(payload=result)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)


class DropFillNaNNode(BaseExecutionNode):
    """
    Removes or fills NaN values in a table.

    Modes:
    - *Drop rows* — remove any row containing at least one NaN in the specified columns
    - *Fill constant* — replace NaN with a fixed value (e.g. `0` or `"unknown"`)
    - *Fill mean / median / mode* — replace with column statistics
    - *Forward fill / Back fill* — propagate the last or next valid value

    **Columns** — comma-separated column names to act on. Leave empty to apply to all columns.

    Keywords: drop NaN, fill NaN, missing values, impute, clean table, 缺失值, 填補, 刪除, 空值, 資料清理
    """
    __identifier__ = 'nodes.dataframe.Transform'
    NODE_NAME = 'Drop / Fill NaN'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}
    _collection_aware = True

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('in',  color=PORT_COLORS['table'])
        self.add_output('out', color=PORT_COLORS['table'])
        self.add_combo_menu('mode', 'Mode',
                            items=['Drop rows', 'Fill constant', 'Fill mean',
                                   'Fill median', 'Fill mode', 'Forward fill', 'Back fill'])
        self.add_text_input('fill_value', 'Fill Value (constant mode)', text='0')
        self._add_column_selector('columns',   'Columns (blank = all)',       text='', mode='multi')

    def evaluate(self):
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error(); return False, "No input connected"
        cp  = in_port.connected_ports()[0]
        val = cp.node().output_values.get(cp.name())
        if isinstance(val, TableData):
            df = val.df.copy()
        elif isinstance(val, pd.DataFrame):
            df = val.copy()
        else:
            self.mark_error(); return False, "Input must be TableData"

        self._refresh_column_selectors(df, 'columns')

        mode      = str(self.get_property('mode') or 'Drop rows')
        cols_str  = str(self.get_property('columns') or '').strip()
        fill_val  = str(self.get_property('fill_value') or '0')

        cols = [c.strip() for c in cols_str.split(',') if c.strip() and c.strip() in df.columns] \
               if cols_str else list(df.columns)

        try:
            if mode == 'Drop rows':
                df = df.dropna(subset=cols).reset_index(drop=True)
            elif mode == 'Fill constant':
                try:
                    v = float(fill_val) if '.' in fill_val else int(fill_val)
                except ValueError:
                    v = fill_val
                df[cols] = df[cols].fillna(v)
            elif mode == 'Fill mean':
                for c in cols:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        df[c] = df[c].fillna(df[c].mean())
            elif mode == 'Fill median':
                for c in cols:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        df[c] = df[c].fillna(df[c].median())
            elif mode == 'Fill mode':
                for c in cols:
                    m = df[c].mode()
                    if not m.empty:
                        df[c] = df[c].fillna(m.iloc[0])
            elif mode == 'Forward fill':
                df[cols] = df[cols].ffill()
            elif mode == 'Back fill':
                df[cols] = df[cols].bfill()

            self.output_values['out'] = TableData(payload=df)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)


class NormalizeColumnNode(BaseExecutionNode):
    """
    Normalizes one or more numeric columns.

    Methods:
    - *Min-Max (0-1)* — scales each column to [0, 1]
    - *Z-score* — subtracts mean and divides by std (standard score)
    - *Log10 / Log2 / Ln* — log transform (adds 1 before log to handle zeros)
    - *Robust (IQR)* — subtracts median, divides by IQR; robust to outliers

    Parameters:
    - **Columns** — comma-separated names. Leave empty to normalize all numeric columns.
    - **Suffix** — text appended to new column names (e.g. `_norm`). Leave empty to overwrite in-place.

    Keywords: normalize, z-score, min-max, log transform, scale, standardize, 標準化, 正規化, 對數, 縮放, 統計
    """
    __identifier__ = 'nodes.dataframe.Compute'
    NODE_NAME = 'Normalize Column'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}
    _collection_aware = True

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('in',  color=PORT_COLORS['table'])
        self.add_output('out', color=PORT_COLORS['table'])
        self.add_combo_menu('method', 'Method',
                            items=['Min-Max (0–1)', 'Z-score', 'Log10', 'Log2',
                                   'Ln (natural)', 'Robust (IQR)'])
        self._add_column_selector('columns', 'Columns (blank = all numeric)', text='', mode='multi')
        self.add_text_input('suffix',  'New Column Suffix (blank = overwrite)', text='_norm')

    def evaluate(self):
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error(); return False, "No input connected"
        cp  = in_port.connected_ports()[0]
        val = cp.node().output_values.get(cp.name())
        if isinstance(val, TableData):
            df = val.df.copy()
        elif isinstance(val, pd.DataFrame):
            df = val.copy()
        else:
            self.mark_error(); return False, "Input must be TableData"

        self._refresh_column_selectors(df, 'columns')

        method   = str(self.get_property('method') or 'Min-Max (0–1)')
        cols_str = str(self.get_property('columns') or '').strip()
        suffix   = str(self.get_property('suffix') or '')

        if cols_str:
            cols = [c.strip() for c in cols_str.split(',') if c.strip() and c.strip() in df.columns]
        else:
            cols = list(df.select_dtypes(include='number').columns)

        if not cols:
            self.mark_error(); return False, "No numeric columns to normalize"

        try:
            for c in cols:
                s = df[c].astype(float)
                if method == 'Min-Max (0–1)':
                    mn, mx = s.min(), s.max()
                    out = (s - mn) / (mx - mn) if mx != mn else s * 0.0
                elif method == 'Z-score':
                    out = (s - s.mean()) / s.std(ddof=1) if s.std(ddof=1) != 0 else s * 0.0
                elif method == 'Log10':
                    out = np.log10(s + 1)
                elif method == 'Log2':
                    out = np.log2(s + 1)
                elif method == 'Ln (natural)':
                    out = np.log1p(s)
                else:  # Robust IQR
                    med = s.median()
                    q75, q25 = s.quantile(0.75), s.quantile(0.25)
                    iqr = q75 - q25
                    out = (s - med) / iqr if iqr != 0 else s * 0.0

                dest = c + suffix if suffix else c
                df[dest] = out

            self.output_values['out'] = TableData(payload=df)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)


# ---------------------------------------------------------------------------
# Medium-priority new nodes
# ---------------------------------------------------------------------------

class ValueCountsNode(BaseExecutionNode):
    """
    Counts occurrences of each unique value in a column.

    Outputs a two-column table with the original column name and a `count` column,
    sorted by count descending by default.

    Parameters:
    - **Column** — the column to count unique values in
    - **Sort by count (descending)** — sort results by frequency
    - **Add percentage column** — include a `pct` column with relative frequencies

    Keywords: value counts, frequency, count unique, histogram categorical, 計數, 頻率, 唯一值, 分組計數, 資料表
    """
    __identifier__ = 'nodes.dataframe.Compute'
    NODE_NAME = 'Value Counts'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}
    _collection_aware = True

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('in',  color=PORT_COLORS['table'])
        self.add_output('out', color=PORT_COLORS['table'])
        self._add_column_selector('column', 'Column', text='', mode='single')
        self.add_checkbox('sort_desc', '', text='Sort by count (descending)', state=True)
        self.add_checkbox('include_pct', '', text='Add percentage column', state=False)

    def evaluate(self):
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error(); return False, "No input connected"
        cp  = in_port.connected_ports()[0]
        val = cp.node().output_values.get(cp.name())
        if isinstance(val, TableData):
            df = val.df
        elif isinstance(val, pd.DataFrame):
            df = val
        else:
            self.mark_error(); return False, "Input must be TableData"

        self._refresh_column_selectors(df, 'column')

        col       = str(self.get_property('column') or '').strip()
        sort_desc = bool(self.get_property('sort_desc'))
        add_pct   = bool(self.get_property('include_pct'))

        if not col:
            self.mark_error(); return False, "Please specify a column"
        if col not in df.columns:
            self.mark_error()
            return False, f"Column '{col}' not found. Available: {list(df.columns)}"

        try:
            vc = df[col].value_counts(sort=sort_desc, ascending=False)
            result = pd.DataFrame({col: vc.index, 'count': vc.values})
            if add_pct:
                result['pct'] = (result['count'] / result['count'].sum() * 100).round(2)
            self.output_values['out'] = TableData(payload=result)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)


class DropDuplicatesNode(BaseExecutionNode):
    """
    Removes duplicate rows from a table.

    Parameters:
    - **Subset Columns** — comma-separated columns to consider when checking for duplicates. Leave empty to compare all columns.
    - **Keep** — which duplicate to keep: *first* occurrence, *last*, or *none*

    Outputs:
    - **unique** — rows after removing duplicates
    - **dropped** — the removed duplicate rows

    Keywords: drop duplicates, unique rows, deduplicate, remove repeated, 去重, 重複值, 唯一, 資料清理, 刪除重複
    """
    __identifier__ = 'nodes.dataframe.Filter'
    NODE_NAME = 'Drop Duplicates'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table', 'table']}
    _collection_aware = True

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('in',     color=PORT_COLORS['table'])
        self.add_output('unique',  color=PORT_COLORS['table'])
        self.add_output('dropped', color=PORT_COLORS['table'])
        self._add_column_selector('subset', 'Subset Columns (blank = all)', text='', mode='multi')
        self.add_combo_menu('keep', 'Keep', items=['first', 'last', 'none'])

    def evaluate(self):
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error(); return False, "No input connected"
        cp  = in_port.connected_ports()[0]
        val = cp.node().output_values.get(cp.name())
        if isinstance(val, TableData):
            df = val.df.copy()
        elif isinstance(val, pd.DataFrame):
            df = val.copy()
        else:
            self.mark_error(); return False, "Input must be TableData"

        self._refresh_column_selectors(df, 'subset')

        subset_str = str(self.get_property('subset') or '').strip()
        keep_str   = str(self.get_property('keep') or 'first')
        keep       = False if keep_str == 'none' else keep_str

        subset = [c.strip() for c in subset_str.split(',')
                  if c.strip() and c.strip() in df.columns] if subset_str else None

        try:
            dup_mask   = df.duplicated(subset=subset, keep=keep)
            df_unique  = df[~dup_mask].reset_index(drop=True)
            df_dropped = df[dup_mask].reset_index(drop=True)
            self.output_values['unique']  = TableData(payload=df_unique)
            self.output_values['dropped'] = TableData(payload=df_dropped)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)


class TypeCastColumnNode(BaseExecutionNode):
    """
    Converts the data type of one or more columns.

    Target types:
    - *float* — convert to floating-point number
    - *int* — convert to integer (rounds, then casts)
    - *str* — convert to string
    - *bool* — convert to boolean (`0`/`False`/`false`/`no` become False, else True)
    - *category* — pandas Categorical (saves memory for low-cardinality columns)
    - *datetime* — parse as datetime using pandas `to_datetime`

    Parameters:
    - **Columns** — comma-separated column names to cast
    - **Coerce errors to NaN** — when checked, unparseable values become NaN instead of raising an error

    Keywords: type cast, convert dtype, int to float, string column, category, 型別轉換, 資料型別, 整數, 字串, 類別
    """
    __identifier__ = 'nodes.dataframe.Transform'
    NODE_NAME = 'Type Cast Column'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}
    _collection_aware = True

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('in',  color=PORT_COLORS['table'])
        self.add_output('out', color=PORT_COLORS['table'])
        self._add_column_selector('columns', 'Columns (comma-separated)', text='', mode='multi')
        self.add_combo_menu('dtype', 'Target Type',
                            items=['float', 'int', 'str', 'bool', 'category', 'datetime'])
        self.add_checkbox('errors_coerce', '', text='Coerce errors to NaN (not raise)', state=True)

    def evaluate(self):
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error(); return False, "No input connected"
        cp  = in_port.connected_ports()[0]
        val = cp.node().output_values.get(cp.name())
        if isinstance(val, TableData):
            df = val.df.copy()
        elif isinstance(val, pd.DataFrame):
            df = val.copy()
        else:
            self.mark_error(); return False, "Input must be TableData"

        self._refresh_column_selectors(df, 'columns')

        cols_str = str(self.get_property('columns') or '').strip()
        dtype    = str(self.get_property('dtype') or 'float')
        coerce   = bool(self.get_property('errors_coerce'))
        errors   = 'coerce' if coerce else 'raise'

        if not cols_str:
            self.mark_error(); return False, "Please specify at least one column"
        cols = [c.strip() for c in cols_str.split(',') if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            self.mark_error()
            return False, f"Columns not found: {missing}"

        try:
            for c in cols:
                if dtype == 'float':
                    df[c] = pd.to_numeric(df[c], errors=errors).astype(float)
                elif dtype == 'int':
                    df[c] = pd.to_numeric(df[c], errors=errors).round().astype('Int64')
                elif dtype == 'str':
                    df[c] = df[c].astype(str)
                elif dtype == 'bool':
                    s = df[c].astype(str).str.lower().str.strip()
                    df[c] = ~s.isin({'0', 'false', 'no', 'none', 'nan', ''})
                elif dtype == 'category':
                    df[c] = df[c].astype('category')
                elif dtype == 'datetime':
                    df[c] = pd.to_datetime(df[c], errors=errors)

            self.output_values['out'] = TableData(payload=df)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)


class StringColumnOpsNode(BaseExecutionNode):
    """
    Applies a string operation to a text column.

    Operations:
    - *Strip whitespace* — remove leading/trailing spaces
    - *To upper / To lower / Title case* — change case
    - *Replace* — replace a substring or regex pattern with another string
    - *Extract regex group* — extract first capture group; non-matching rows become NaN
    - *Split to two columns* — split on a delimiter and put left/right parts into two new columns
    - *Pad / Zfill* — left-pad with zeros to a fixed width

    Parameters:
    - **Column** — source text column to operate on
    - **Pattern / Delimiter / Width** — context-dependent input for the selected operation
    - **Replace With** — replacement string (for Replace operation)
    - **Result Column** — name for the output column. Leave empty to overwrite the source column.

    Keywords: string ops, text column, replace, extract, split, strip, regex, 字串, 文字, 取代, 提取, 分割
    """
    __identifier__ = 'nodes.dataframe.Transform'
    NODE_NAME = 'String Column Ops'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['table']}
    _collection_aware = True

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('in',  color=PORT_COLORS['table'])
        self.add_output('out', color=PORT_COLORS['table'])
        self._add_column_selector('column',     'Column',         text='', mode='single')
        self.add_combo_menu('operation',  'Operation',
                            items=['Strip whitespace', 'To upper', 'To lower', 'Title case',
                                   'Replace', 'Extract regex group', 'Split → two columns',
                                   'Pad / Zfill'])
        self.add_text_input('pattern',    'Pattern / Delimiter / Width', text='')
        self.add_text_input('replace_with', 'Replace With',             text='')
        self.add_text_input('result_col', 'Result Column (blank = overwrite)', text='')
        self.add_checkbox('use_regex', '', text='Pattern is regex (Replace / Extract)', state=False)

    def evaluate(self):
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error(); return False, "No input connected"
        cp  = in_port.connected_ports()[0]
        val = cp.node().output_values.get(cp.name())
        if isinstance(val, TableData):
            df = val.df.copy()
        elif isinstance(val, pd.DataFrame):
            df = val.copy()
        else:
            self.mark_error(); return False, "Input must be TableData"

        self._refresh_column_selectors(df, 'column')

        col       = str(self.get_property('column') or '').strip()
        op        = str(self.get_property('operation') or 'Strip whitespace')
        pattern   = str(self.get_property('pattern') or '')
        repl      = str(self.get_property('replace_with') or '')
        res_col   = str(self.get_property('result_col') or '').strip() or col
        use_regex = bool(self.get_property('use_regex'))

        if not col:
            self.mark_error(); return False, "Please specify a column"
        if col not in df.columns:
            self.mark_error()
            return False, f"Column '{col}' not found. Available: {list(df.columns)}"

        try:
            s = df[col].astype(str)
            if op == 'Strip whitespace':
                df[res_col] = s.str.strip()
            elif op == 'To upper':
                df[res_col] = s.str.upper()
            elif op == 'To lower':
                df[res_col] = s.str.lower()
            elif op == 'Title case':
                df[res_col] = s.str.title()
            elif op == 'Replace':
                df[res_col] = s.str.replace(pattern, repl, regex=use_regex)
            elif op == 'Extract regex group':
                if not pattern:
                    self.mark_error(); return False, "Pattern is required for Extract"
                # If pattern already has a capture group, use as-is; otherwise wrap
                if '(' in pattern and ')' in pattern:
                    df[res_col] = s.str.extract(pattern, expand=False)
                else:
                    df[res_col] = s.str.extract(f'({pattern})', expand=False)
            elif op == 'Split → two columns':
                delim = pattern if pattern else ' '
                split = s.str.split(delim, n=1, expand=True)
                left_col  = res_col + '_left'  if res_col == col else res_col
                right_col = res_col + '_right' if res_col == col else res_col + '_right'
                df[left_col]  = split[0] if 0 in split.columns else pd.NA
                df[right_col] = split[1] if 1 in split.columns else pd.NA
            elif op == 'Pad / Zfill':
                try:
                    width = int(pattern) if pattern else 4
                except ValueError:
                    width = 4
                df[res_col] = s.str.zfill(width)

            self.output_values['out'] = TableData(payload=df)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)


class NormalizeMappingWidget(NodeBaseWidget):
    """
    A custom node widget providing a 2-column table to map features (numerical columns)
    to specific control groups for normalization.
    """
    mapping_edited = QtCore.Signal(dict)
    update_table_signal = QtCore.Signal(dict)

    def __init__(self, parent=None, name='', label=''):
        super(NormalizeMappingWidget, self).__init__(parent, name, label)

        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(['Group', 'Normalize By'])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(150)

        self._table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
                gridline-color: #3f3f3f;
                border: 1px solid #3f3f3f;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #d4d4d4;
                border: 1px solid #3f3f3f;
                padding: 4px;
            }
        """)

        self.set_custom_widget(self._table)

        self._table.itemChanged.connect(self._on_table_edited)
        self.update_table_signal.connect(self._sync_table_ui)
        self._is_updating = False

    def _sync_table_ui(self, mapping_dict):
        self._is_updating = True
        self._table.setRowCount(0)

        for k, v in mapping_dict.items():
            row = self._table.rowCount()
            self._table.insertRow(row)

            item_k = QtWidgets.QTableWidgetItem(str(k))
            item_k.setFlags(item_k.flags() ^ QtCore.Qt.ItemFlag.ItemIsEditable)
            item_k.setBackground(QtGui.QColor("#2a2a2a"))
            self._table.setItem(row, 0, item_k)

            item_v = QtWidgets.QTableWidgetItem(str(v) if v else "")
            self._table.setItem(row, 1, item_v)

        self._is_updating = False
        self.mapping_edited.emit(mapping_dict)

    def _on_table_edited(self, item):
        if self._is_updating:
            return

        col = item.column()
        if col == 1:
            key_item = self._table.item(item.row(), 0)
            if not key_item:
                return

            current_dict = {}
            for r in range(self._table.rowCount()):
                k = self._table.item(r, 0).text()
                v = self._table.item(r, 1).text().strip()
                current_dict[k] = v

            self.mapping_edited.emit(current_dict)

    def get_value(self):
        return ""

    def set_value(self, value):
        pass


class GroupNormalizationNode(BaseExecutionNode):
    """
    Normalizes numerical columns relative to a specified control group mean.

    Each numeric column is divided by the mean of its corresponding control group,
    producing fold-change values. A mapping table widget lets you assign a different
    control group for each unique group in the data.

    Parameters:
    - **Global Control Group** — default control group name used for normalization
    - **Target Column** — column containing group labels (e.g. `Group`)

    Keywords: normalization, control group, fold change, relative expression, group-wise scaling, 正規化, 對照組, 倍數變化, 分組, 相對表現
    """
    __identifier__ = 'nodes.dataframe.Compute'
    NODE_NAME      = 'Group Normalization'
    PORT_SPEC      = {'inputs': ['table'], 'outputs': ['table']}

    def __init__(self):
        super(GroupNormalizationNode, self).__init__()
        self.add_input('in', color=PORT_COLORS['table'])
        self.add_output('out', color=PORT_COLORS['table'])
        self.add_text_input('control_group', 'Global Control Group', text='')
        self._add_column_selector('target_column', 'Target Column', text='Group', mode='single')

        self.create_property('normalization_mapping_json', '{}',
                             widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value)
        self._mapping_widget = NormalizeMappingWidget(
            self.view, name='norm_mapping_widget', label='Group Targets')
        self._mapping_widget.mapping_edited.connect(self._on_mapping_edited)

        self.add_custom_widget(self._mapping_widget, tab='Parameters')
        self._fix_widget_z_order()

    def _on_mapping_edited(self, new_dict):
        self.set_property('normalization_mapping_json', json.dumps(new_dict))

    def update(self):
        super().update()
        try:
            m = json.loads(self.get_property('normalization_mapping_json'))
            self._mapping_widget.update_table_signal.emit(m)
        except Exception:
            pass

    def evaluate(self):
        self.reset_progress()

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

        self._refresh_column_selectors(df, 'target_column')

        global_control = str(self.get_property('control_group')).strip()
        target_col_name = str(self.get_property('target_column')).strip()

        target_col = None
        if target_col_name and target_col_name in df.columns:
            target_col = target_col_name
        else:
            for col in df.columns:
                if str(col).lower() in ['group', 'class', 'treatment']:
                    target_col = col
                    break

        if not target_col:
            if global_control and global_control in df.columns:
                try:
                    control_mean = df[global_control].mean(skipna=True)
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    for col in num_cols:
                        df[col] = df[col] / control_mean
                    self.output_values['out'] = TableData(payload=df)
                    self.set_progress(100)
                    self.mark_clean()
                    return True, None
                except Exception as e:
                    self.mark_error()
                    return False, str(e)
            else:
                self.mark_error()
                return False, f"Could not find 'Group' column or column named '{global_control}'"

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        unique_groups = [str(v) for v in df[target_col].dropna().unique()]

        try:
            current_mapping = json.loads(self.get_property('normalization_mapping_json'))
        except Exception:
            current_mapping = {}

        new_mapping = {}
        mapping_changed = False

        for g in unique_groups:
            if g in current_mapping and current_mapping[g].strip():
                new_mapping[g] = current_mapping[g]
            else:
                new_mapping[g] = global_control
                mapping_changed = True

        if len(new_mapping) != len(current_mapping):
            mapping_changed = True

        current_mapping = new_mapping

        if mapping_changed:
            self._mapping_widget.update_table_signal.emit(current_mapping)

        df_orig = df.copy()
        df_group_str = df_orig[target_col].astype(str)

        try:
            for g in unique_groups:
                norm_control = current_mapping.get(g, "")
                if not norm_control.strip():
                    norm_control = global_control
                if not norm_control.strip():
                    continue

                control_data = df_orig[df_group_str == norm_control]
                if control_data.empty:
                    continue

                group_mask = df_group_str == g
                if not group_mask.any():
                    continue

                for col in num_cols:
                    control_mean = control_data[col].mean(skipna=True)
                    if control_mean != 0 and not pd.isna(control_mean):
                        df.loc[group_mask, col] = df_orig.loc[group_mask, col] / control_mean
                    else:
                        df.loc[group_mask, col] = np.nan

            self.output_values['out'] = TableData(payload=df)
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            return False, str(e)


# ── Save Table Node ──────────────────────────────────────────────────────────

class SaveTableNode(BaseExecutionNode):
    """
    Saves a table to disk. Click Browse to choose file location and format.

    Inputs:
    - **table** — TableData to save

    Supported formats: CSV, TSV, Excel (.xlsx), GraphPad Prism (.pzfx).
    Users can also type any path with a custom extension directly.

    Keywords: save, export, csv, tsv, excel, xlsx, pzfx, table, write, 儲存, 匯出, 表格
    """
    __identifier__ = 'nodes.dataframe.IO'
    NODE_NAME = 'Save Table'
    PORT_SPEC = {'inputs': ['table'], 'outputs': []}
    _collection_aware = True

    _EXT_FILTER = (
        'CSV Files (*.csv);;'
        'TSV Files (*.tsv);;'
        'Excel Files (*.xlsx);;'
        'GraphPad Prism (*.pzfx);;'
        'All Files (*)'
    )

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])

        from nodes.base import NodeFileSaver
        saver = NodeFileSaver(self.view, name='file_path', label='Save Path',
                              ext_filter=self._EXT_FILTER)
        self.add_custom_widget(saver,
                               widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value)

    def evaluate(self):
        self.reset_progress()

        port = self.inputs().get('table')
        if not port or not port.connected_ports():
            self.mark_error()
            return False, "No input connected"
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if not isinstance(data, TableData):
            self.mark_error()
            return False, "Input must be TableData"

        file_path = str(self.get_property('file_path') or '').strip()
        if not file_path:
            self.mark_error()
            return False, "No file path specified"

        import os
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        ext = os.path.splitext(file_path)[1].lower()
        df = data.df

        self.set_progress(30)
        try:
            if ext == '.xlsx':
                df.to_excel(file_path, index=False)
            elif ext == '.tsv':
                df.to_csv(file_path, sep='\t', index=False)
            elif ext == '.pzfx':
                from nodes.io_nodes import write_pzfx
                write_pzfx({"Data 1": df}, file_path)
            else:
                df.to_csv(file_path, index=False)

            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)
