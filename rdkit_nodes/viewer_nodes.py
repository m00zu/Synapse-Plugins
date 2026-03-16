"""
viewer_nodes.py — 3D structure viewer node using 3Dmol.js + QWebEngineView.

Provides:
  - ViewerBridge           QObject bridge for JS↔Python communication
  - Node3DViewerWidget     QWebEngineView widget with 3Dmol.js viewer
  - StructureViewerNode    Passive inline 3D molecular viewer
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from nodes.base import BaseExecutionNode, PORT_COLORS, NodeBaseWidget
from data_models import NodeData

from .protein_data import ProteinData, ReceptorData, DockingResultData

try:
    from .chem_nodes import MoleculeData, MolTableData
except ImportError:
    MoleculeData = None
    MolTableData = None

# ── Paths ────────────────────────────────────────────────────────────────────
_VIEWER_DIR = Path(__file__).parent / 'viewer'
_HTML_PATH = _VIEWER_DIR / 'viewer.html'
_JS_PATH = _VIEWER_DIR / '3Dmol-min.js'


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pdbqt_to_pdb_string(pdbqt_str: str) -> str:
    """Convert PDBQT back to PDB format for 3Dmol.js display."""
    from .protein_utils import pdbqt_to_pdb
    _data, pdb_str = pdbqt_to_pdb(pdbqt_str)
    return pdb_str


def _extract_first_pose(multi_pdbqt: str) -> str:
    """Extract the first MODEL from a multi-pose PDBQT string."""
    lines = []
    in_model = False
    for line in multi_pdbqt.splitlines():
        if line.startswith('MODEL'):
            in_model = True
            continue
        if line.startswith('ENDMDL'):
            break
        if in_model or not multi_pdbqt.lstrip().startswith('MODEL'):
            # If no MODEL record, take everything up to END
            lines.append(line)
            if line.startswith('END') and not line.startswith('ENDMDL'):
                break
    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  ViewerBridge — QObject for JS↔Python communication via QWebChannel
# ══════════════════════════════════════════════════════════════════════════════

class ViewerBridge(QtCore.QObject):
    """Bidirectional bridge between 3Dmol.js and Python via QWebChannel.

    JavaScript calls ``bridge.onAtomClicked(jsonStr)`` when the user clicks
    an atom.  Python receives the data through the ``atom_clicked`` signal.
    """
    atom_clicked = QtCore.Signal(dict)

    @QtCore.Slot(str)
    def onAtomClicked(self, json_str):
        """Called from JavaScript when the user clicks an atom."""
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return
        self.atom_clicked.emit(data)


# ══════════════════════════════════════════════════════════════════════════════
#  Node3DViewerWidget — QWebEngineView embedded in a NodeBaseWidget
# ══════════════════════════════════════════════════════════════════════════════

class Node3DViewerWidget(NodeBaseWidget):
    """
    Embeds a QWebEngineView showing a 3Dmol.js molecular viewer.
    Accepts dict data: {'protein_pdb': str, 'ligand_pdb': str}

    Optionally sets up a QWebChannel for bidirectional JS↔Python
    communication (click handling, shape rendering).
    """
    _DISPLAY_W = 520
    _DISPLAY_H = 450

    def __init__(self, parent=None, enable_bridge=False):
        super().__init__(parent, name='viewer_3d', label='')

        self._page_ready = False
        self._pending_data = None
        self._pending_js = []

        # Container widget for sizing
        self._container = QtWidgets.QWidget()
        self._container.setFixedSize(self._DISPLAY_W, self._DISPLAY_H)

        layout = QtWidgets.QVBoxLayout(self._container)
        layout.setContentsMargins(0, 0, 0, 0)

        from PySide6.QtWebEngineWidgets import QWebEngineView
        self._web = QWebEngineView()
        self._web.setFixedSize(self._DISPLAY_W, self._DISPLAY_H)
        layout.addWidget(self._web)

        # ── Optional QWebChannel for click handling ──────────────────
        self._bridge = None
        if enable_bridge:
            from PySide6.QtWebChannel import QWebChannel
            self._bridge = ViewerBridge()
            self._channel = QWebChannel()
            self._channel.registerObject('bridge', self._bridge)
            self._web.page().setWebChannel(self._channel)

        self.set_custom_widget(self._container)

        # Prevent "WebEnginePage still not deleted" warning on teardown
        self._container.destroyed.connect(self._cleanup_web)

        # Load the viewer HTML
        self._load_viewer_html()

    def _cleanup_web(self):
        """Ensure the web page is deleted before the profile."""
        if self._web is not None:
            self._web.setPage(None)
            self._web = None

    def _load_viewer_html(self):
        """Load viewer.html with the correct base URL so 3Dmol-min.js resolves."""
        if not _HTML_PATH.exists():
            self._web.setHtml(
                '<html><body style="background:#1a1a2e;color:#aaa;font-family:sans-serif">'
                '<p>viewer.html not found</p></body></html>'
            )
            return

        html = _HTML_PATH.read_text(encoding='utf-8')
        base_url = QtCore.QUrl.fromLocalFile(str(_VIEWER_DIR) + '/')
        self._web.setHtml(html, base_url)
        self._web.loadFinished.connect(self._on_load_finished)

    def _on_load_finished(self, ok):
        """Called when the HTML page finishes loading."""
        self._page_ready = ok
        if ok and self._pending_data is not None:
            self._push_data(self._pending_data)
            self._pending_data = None
        if ok and self._pending_js:
            for js in self._pending_js:
                self._web.page().runJavaScript(js)
            self._pending_js.clear()

    def set_value(self, data):
        """Update the 3D viewer with new data.

        Args:
            data: dict with optional keys 'protein_pdb' and 'ligand_pdb'
        """
        if not isinstance(data, dict):
            return
        if self._page_ready:
            self._push_data(data)
        else:
            self._pending_data = data

    def _push_data(self, data):
        """Call the JS loadStructure function with PDB strings."""
        protein_pdb = data.get('protein_pdb', '') or ''
        ligand_pdb = data.get('ligand_pdb', '') or ''
        js = f'loadStructure({json.dumps(protein_pdb)}, {json.dumps(ligand_pdb)});'
        self._web.page().runJavaScript(js)

    def run_js(self, js_code):
        """Execute arbitrary JavaScript in the viewer. Queues if page not ready."""
        if self._page_ready and self._web is not None:
            self._web.page().runJavaScript(js_code)
        else:
            self._pending_js.append(js_code)

    def get_value(self):
        return None  # Display only


# ══════════════════════════════════════════════════════════════════════════════
#  StructureViewerNode
# ══════════════════════════════════════════════════════════════════════════════

class StructureViewerNode(BaseExecutionNode):
    """Interactive 3D molecular structure viewer.

    Displays proteins as cartoon and ligands as ball-and-stick using 3Dmol.js.
    Connect any combination of protein, receptor, molecule, or docking result
    inputs to visualize the structure.

    Keywords: 3D viewer, protein, ligand, structure, molecular viewer, 3Dmol,
              cartoon, docking pose, complex, 蛋白質, 分子, 檢視器, 對接
    """
    __identifier__ = 'nodes.Cheminformatics.Viewer'
    NODE_NAME = '3D Structure Viewer'
    PORT_SPEC = {
        'inputs': ['protein', 'receptor', 'molecule', 'docking_result'],
        'outputs': [],
    }

    def __init__(self):
        super().__init__(use_progress=False)

        # Input ports — all optional
        self.add_input('protein', color=PORT_COLORS.get('protein', (34, 139, 34)))
        self.add_input('receptor', color=PORT_COLORS.get('receptor', (0, 128, 128)))
        self.add_input('molecule', color=PORT_COLORS.get('molecule', (100, 100, 255)))
        self.add_input('docking_result', color=PORT_COLORS.get('docking_result', (255, 140, 0)))

        # Embedded 3D viewer widget
        self._viewer_widget = Node3DViewerWidget(self.view)
        self.add_custom_widget(self._viewer_widget, tab='View')

    def evaluate(self):
        self.reset_progress()
        try:
            protein_pdb = ''
            ligand_pdb = ''

            # ── Collect protein/receptor PDB string ──────────────────────
            protein_port = self.inputs().get('protein')
            if protein_port and protein_port.connected_ports():
                src = protein_port.connected_ports()[0]
                val = src.node().output_values.get(src.name())
                if isinstance(val, ProteinData) and val.payload:
                    protein_pdb = val.payload

            receptor_port = self.inputs().get('receptor')
            if receptor_port and receptor_port.connected_ports():
                src = receptor_port.connected_ports()[0]
                val = src.node().output_values.get(src.name())
                if isinstance(val, ReceptorData) and val.payload:
                    # PDBQT → PDB for display
                    protein_pdb = _pdbqt_to_pdb_string(val.payload)

            # ── Collect ligand PDB string ────────────────────────────────
            mol_port = self.inputs().get('molecule')
            if mol_port and mol_port.connected_ports():
                src = mol_port.connected_ports()[0]
                val = src.node().output_values.get(src.name())
                if MoleculeData is not None and isinstance(val, MoleculeData):
                    from rdkit import Chem
                    mol = val.payload
                    if mol is not None:
                        ligand_pdb = Chem.MolToPDBBlock(mol)

            dock_port = self.inputs().get('docking_result')
            if dock_port and dock_port.connected_ports():
                src = dock_port.connected_ports()[0]
                val = src.node().output_values.get(src.name())
                if isinstance(val, DockingResultData) and val.payload:
                    first_pose = _extract_first_pose(val.payload)
                    if first_pose.strip():
                        ligand_pdb = _pdbqt_to_pdb_string(first_pose)

            if not protein_pdb and not ligand_pdb:
                self.mark_clean()
                return True, None

            # Send data to the viewer widget via display signal
            self.set_display({'protein_pdb': protein_pdb, 'ligand_pdb': ligand_pdb})
            self.mark_clean()
            return True, None

        except Exception as e:
            self.mark_error()
            return False, f'{e}\n{traceback.format_exc()}'

    def _display_ui(self, data):
        """Updates the embedded 3D viewer (Main Thread only)."""
        if isinstance(data, dict):
            self._viewer_widget.set_value(data)
            self.view.draw_node()
