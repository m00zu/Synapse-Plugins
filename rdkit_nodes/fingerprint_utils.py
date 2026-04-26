"""
fingerprint_utils.py — Fingerprint method + param registry and an inline
NodeBaseWidget (method combo + dynamic param table) shared by the
clustering / similarity / search nodes.

Design: every FP method has a dict of canonical parameter name -> default
value. A ``settings`` dict ``{'method': str, 'params': {...}}`` fully
describes a fingerprint configuration, and ``retrieve_fp_generator(settings)``
returns a callable ``mol -> fingerprint`` ready for batch use.

Atom/bond invariant-generator sub-chaining is intentionally omitted from v1
to keep the inline UI simple; add it if a user asks.
"""
from __future__ import annotations

from typing import Callable

from PySide6 import QtCore, QtWidgets

import numpy as np

from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdmolops import LayeredFingerprint, PatternFingerprint
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint

try:
    from rdkit.Avalon.pyAvalonTools import GetAvalonFP as _GetAvalonFP
    _HAS_AVALON = True
except ImportError:
    _HAS_AVALON = False
    _GetAvalonFP = None  # type: ignore

try:
    from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder
    _HAS_MHFP = True
except ImportError:
    _HAS_MHFP = False
    MHFPEncoder = None  # type: ignore

from NodeGraphQt.widgets.node_widgets import NodeBaseWidget


# ── Public constants ──────────────────────────────────────────────────────────

FP_METHODS: list[str] = [
    'Morgan', 'RDKit', 'Topological Torsion', 'Atom Pair',
    'Layered', 'Pattern', 'MACCS', 'ErG',
]
if _HAS_AVALON:
    FP_METHODS.append('Avalon')
if _HAS_MHFP:
    FP_METHODS.extend(['MHFP', 'SECFP'])

SIMILARITY_METRICS: list[str] = [
    'Tanimoto', 'Dice', 'Braun-Blanquet', 'Cosine', 'Kulczynski',
    'McConnaughey', 'Rogot-Goldberg', 'Russel', 'Sokal', 'Tversky',
]

# Default parameter dict per method. Insertion order determines table order.
# Each value's type (bool / int / str) drives the widget kind in the table;
# string-valued params must have their option list declared in FP_ENUMS.
FP_DEFAULTS: dict[str, dict] = {
    'Morgan': {
        'fpSize':                2048,
        'radius':                2,
        'variant':               'ECFP',
        'countSimulation':       False,
        'includeChirality':      False,
        'useBondTypes':          True,
        'onlyNonzeroInvariants': False,
        'includeRingMembership': True,
    },
    'RDKit': {
        'fpSize':            2048,
        'minPath':           1,
        'maxPath':           7,
        'useHs':             True,
        'branchedPaths':     True,
        'useBondOrder':      True,
        'countSimulation':   False,
        'numBitsPerFeature': 2,
    },
    'Topological Torsion': {
        'fpSize':           2048,
        'includeChirality': False,
        'torsionAtomCount': 4,
        'countSimulation':  True,
    },
    'Atom Pair': {
        'fpSize':           2048,
        'minDistance':      1,
        'maxDistance':      30,
        'includeChirality': False,
        'use2D':            True,
        'countSimulation':  True,
    },
    'Layered': {
        'fpSize':        2048,
        'minPath':       1,
        'maxPath':       7,
        'branchedPaths': True,
    },
    'Pattern': {
        'fpSize':               2048,
        'tautomerFingerprints': False,
    },
    'Avalon': {
        'nBits':     512,
        'isQuery':   False,
        'resetVect': False,
    },
    'MACCS': {},
    'ErG': {
        'fuzzIncrement': 0.3,
        'minPath':       1,
        'maxPath':       15,
    },
    'MHFP': {
        'fpSize':    2048,
        'radius':    3,
        'minRadius': 1,
        'rings':     True,
        'isomeric':  False,
        'kekulize':  True,
    },
    'SECFP': {
        'fpSize':    2048,
        'radius':    3,
        'minRadius': 1,
        'rings':     True,
        'isomeric':  False,
        'kekulize':  True,
    },
}

# String-valued params must declare their enum here. Maps (method, param) to
# the list of legal option strings (first entry is the UI default).
FP_ENUMS: dict[str, dict[str, list[str]]] = {
    'Morgan': {
        # ECFP = connectivity-based atom invariants; FCFP = feature-based.
        'variant': ['ECFP', 'FCFP'],
    },
}


# ── Generator resolution ──────────────────────────────────────────────────────

def default_settings(method: str = 'Morgan') -> dict:
    """Return a fresh settings dict for the given method."""
    return {'method': method, 'params': dict(FP_DEFAULTS.get(method, {}))}


def retrieve_fp_generator(settings: dict) -> Callable:
    """Build a callable ``mol -> fingerprint`` from a settings dict.

    The returned callable is built once and reused across molecules, so
    generator-style FPs (Morgan, RDKit, Atom Pair, Topological Torsion)
    are constructed exactly once.
    """
    method = settings.get('method', 'Morgan')
    params = dict(settings.get('params') or {})

    if method == 'Morgan':
        variant = params.pop('variant', 'ECFP')
        if variant == 'FCFP':
            params['atomInvariantsGenerator'] = (
                rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
            )
        # ECFP is the RDKit default; leave atomInvariantsGenerator unset.
        gen = rdFingerprintGenerator.GetMorganGenerator(**params)
        return lambda mol: gen.GetFingerprint(mol)
    if method == 'RDKit':
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(**params)
        return lambda mol: gen.GetFingerprint(mol)
    if method == 'Topological Torsion':
        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(**params)
        return lambda mol: gen.GetFingerprint(mol)
    if method == 'Atom Pair':
        gen = rdFingerprintGenerator.GetAtomPairGenerator(**params)
        return lambda mol: gen.GetFingerprint(mol)
    if method == 'Layered':
        return lambda mol: LayeredFingerprint(mol, **params)
    if method == 'Pattern':
        return lambda mol: PatternFingerprint(mol, **params)
    if method == 'Avalon':
        if not _HAS_AVALON:
            raise RuntimeError('Avalon fingerprints are not available in this RDKit build.')
        return lambda mol: _GetAvalonFP(mol, **params)
    if method == 'MACCS':
        return lambda mol: GetMACCSKeysFingerprint(mol)
    # The non-bit-vector methods (ErG, MHFP6, SECFP4) return numpy arrays
    # directly, so retrieve_fp_generator returns an ndarray rather than an
    # RDKit ExplicitBitVect.  compute_fingerprint_matrix normalises.
    if method == 'ErG':
        fuzz = float(params.get('fuzzIncrement', 0.3))
        minP = int(params.get('minPath', 1))
        maxP = int(params.get('maxPath', 15))
        return lambda mol: np.asarray(GetErGFingerprint(
            mol, fuzzIncrement=fuzz, minPath=minP, maxPath=maxP),
            dtype=np.float64)
    if method in ('MHFP', 'SECFP'):
        if not _HAS_MHFP:
            raise RuntimeError('MHFPEncoder not available in this RDKit build.')
        fpSize   = int(params.get('fpSize', 2048))
        radius   = int(params.get('radius', 3 if method == 'MHFP' else 2))
        minRad   = int(params.get('minRadius', 1))
        rings    = bool(params.get('rings', True))
        isomeric = bool(params.get('isomeric', False))
        kekulize = bool(params.get('kekulize', True))
        # Seed is fixed at 0 so the permutation table is reproducible across
        # runs. (MHFPEncoder signature: (n_permutations, seed).)
        enc = MHFPEncoder(fpSize, 0)
        if method == 'MHFP':
            return lambda mol: np.asarray(enc.EncodeMol(
                mol, radius=radius, rings=rings, isomeric=isomeric,
                kekulize=kekulize, min_radius=minRad), dtype=np.uint32)
        return lambda mol: np.asarray(enc.EncodeSECFPMol(
            mol, radius=radius, rings=rings, isomeric=isomeric,
            kekulize=kekulize, min_radius=minRad), dtype=bool)
    raise ValueError(f'Unknown fingerprint method: {method!r}')


# ── Inline Qt widget ──────────────────────────────────────────────────────────

class FingerprintParamsWidget(NodeBaseWidget):
    """Inline method-combo + dynamic param-table widget.

    The widget's ``get_value()`` / ``set_value()`` round-trip via a plain
    ``{'method': str, 'params': {...}}`` dict so the node's property
    serialization survives workflow save / reload.
    """

    _MIN_TABLE_HEIGHT = 200

    def __init__(self, parent=None, name: str = 'fp_settings',
                 label: str = 'Fingerprint',
                 default_method: str = 'Morgan'):
        super().__init__(parent, name, label)

        self._current_method = default_method
        self._value_widgets: dict[str, QtWidgets.QWidget] = {}
        self._suppress_emit = False

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # Method row
        method_row = QtWidgets.QHBoxLayout()
        method_row.setContentsMargins(0, 0, 0, 0)
        method_label = QtWidgets.QLabel('Method')
        method_label.setStyleSheet('color: #ccc; font-size: 11px;')
        method_label.setFixedWidth(48)
        self._method_combo = QtWidgets.QComboBox()
        self._method_combo.addItems(FP_METHODS)
        self._method_combo.setCurrentText(default_method)
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        method_row.addWidget(method_label)
        method_row.addWidget(self._method_combo, 1)
        layout.addLayout(method_row)

        # Param table
        self._table = QtWidgets.QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(['Param', 'Value'])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QtWidgets.QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QtWidgets.QTableWidget.SelectionMode.NoSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self._table.setMinimumHeight(self._MIN_TABLE_HEIGHT)
        layout.addWidget(self._table)

        self._build_table(default_method)

        self.set_custom_widget(container)

    # ── Internal ────────────────────────────────────────────────────────────
    def _on_method_changed(self, method: str) -> None:
        self._current_method = method
        self._build_table(method)
        self._emit_change()

    def _build_table(self, method: str) -> None:
        """Rebuild the param rows for the given method, using defaults."""
        defaults = FP_DEFAULTS.get(method, {})
        enums = FP_ENUMS.get(method, {})
        self._value_widgets = {}
        self._table.clearContents()
        self._table.setRowCount(len(defaults))

        for row, (key, default) in enumerate(defaults.items()):
            name_item = QtWidgets.QTableWidgetItem(key)
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, name_item)

            w = self._make_value_widget(default, options=enums.get(key))
            self._table.setCellWidget(row, 1, w)
            self._value_widgets[key] = w

    def _make_value_widget(self, default, options: list[str] | None = None) -> QtWidgets.QWidget:
        if isinstance(default, bool):
            cb = QtWidgets.QCheckBox()
            cb.setChecked(default)
            cb.stateChanged.connect(lambda _: self._emit_change())
            return cb
        if isinstance(default, int):
            sb = QtWidgets.QSpinBox()
            sb.setRange(0, 1 << 16)
            sb.setValue(int(default))
            sb.valueChanged.connect(lambda _: self._emit_change())
            return sb
        if isinstance(default, float):
            ds = QtWidgets.QDoubleSpinBox()
            ds.setRange(0.0, 1e6)
            ds.setDecimals(4)
            ds.setSingleStep(0.1)
            ds.setValue(float(default))
            ds.valueChanged.connect(lambda _: self._emit_change())
            return ds
        if isinstance(default, str) and options:
            cmb = QtWidgets.QComboBox()
            cmb.addItems(options)
            cmb.setCurrentText(default if default in options else options[0])
            cmb.currentTextChanged.connect(lambda _: self._emit_change())
            return cmb
        # Fallback for str without an enum — free-form text edit.
        le = QtWidgets.QLineEdit(str(default))
        le.editingFinished.connect(self._emit_change)
        return le

    def _emit_change(self) -> None:
        if self._suppress_emit:
            return
        self.value_changed.emit(self.get_name(), self.get_value())

    # ── Persistence API used by NodeGraphQt ─────────────────────────────────
    def get_value(self):
        params = {}
        for key, w in self._value_widgets.items():
            if isinstance(w, QtWidgets.QCheckBox):
                params[key] = w.isChecked()
            elif isinstance(w, QtWidgets.QDoubleSpinBox):
                params[key] = float(w.value())
            elif isinstance(w, QtWidgets.QSpinBox):
                params[key] = w.value()
            elif isinstance(w, QtWidgets.QComboBox):
                params[key] = w.currentText()
            elif isinstance(w, QtWidgets.QLineEdit):
                text = w.text().strip()
                try:
                    params[key] = int(text)
                except ValueError:
                    params[key] = text
        return {'method': self._current_method, 'params': params}

    def set_value(self, value):
        if not isinstance(value, dict):
            return
        method = value.get('method') or 'Morgan'
        params = value.get('params') or {}

        self._suppress_emit = True
        try:
            self._method_combo.blockSignals(True)
            self._method_combo.setCurrentText(method)
            self._method_combo.blockSignals(False)
            self._current_method = method
            self._build_table(method)

            for key, val in params.items():
                w = self._value_widgets.get(key)
                if w is None:
                    continue
                if isinstance(w, QtWidgets.QCheckBox):
                    w.setChecked(bool(val))
                elif isinstance(w, QtWidgets.QDoubleSpinBox):
                    try:
                        w.setValue(float(val))
                    except (TypeError, ValueError):
                        pass
                elif isinstance(w, QtWidgets.QSpinBox):
                    try:
                        w.setValue(int(val))
                    except (TypeError, ValueError):
                        pass
                elif isinstance(w, QtWidgets.QComboBox):
                    if val in [w.itemText(i) for i in range(w.count())]:
                        w.setCurrentText(str(val))
                elif isinstance(w, QtWidgets.QLineEdit):
                    w.setText(str(val))
        finally:
            self._suppress_emit = False
