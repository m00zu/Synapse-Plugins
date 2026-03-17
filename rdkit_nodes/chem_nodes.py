"""
chem_nodes.py — RDKit node definitions for the Synapse rdkit_nodes plugin.

All node classes live here; __init__.py handles vendor injection and
re-exports everything via ``from .chem_nodes import *``.
"""
from __future__ import annotations

import NodeGraphQt
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from typing import Any

from PySide6 import QtCore, QtGui, QtSvg
from rdkit import Chem, DataStructs
from rdkit.Chem.rdchem import Mol as RDMol
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.rdMolDraw2D import SetDarkMode
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.rdmolops import LayeredFingerprint, PatternFingerprint
try:
    from rdkit.Avalon.pyAvalonTools import GetAvalonFP as _GetAvalonFP
    _HAS_AVALON = True
except ImportError:
    _HAS_AVALON = False
import sdfrust

from .meeko_ported import MoleculePreparation, PDBQTWriterLegacy

from nodes.base import (BaseExecutionNode, PORT_COLORS,
                         NodeFileSelector, NodeFileSaver, NodeDirSelector)
from data_models import NodeData, ImageData, TableData


# ── Custom port colors ────────────────────────────────────────────────────────
PORT_COLORS.setdefault('molecule',  (205, 92, 92))    # Indian Red
PORT_COLORS.setdefault('mol_table', (178, 102, 178))  # Muted Purple


# ── Custom data types ─────────────────────────────────────────────────────────
class MoleculeData(NodeData):
    """Wraps an RDKit Mol object (or None for an invalid molecule)."""
    payload: RDMol
    name:    str = ''

    @classmethod
    def merge(cls, items: list) -> list:
        """Collect molecules into a list."""
        return [i.payload for i in items]


class MolTableData(TableData):
    """DataFrame with an RDKit Mol object column for batch cheminformatics.

    The ``payload`` DataFrame has at least ``name``, ``smiles``, and a Mol
    column (default ``'ROMol'``).  Downstream nodes can add arbitrary extra
    columns (descriptors, fingerprints, etc.).
    """
    payload: Any          # pd.DataFrame
    mol_col: str = 'ROMol'

    @classmethod
    def merge(cls, items: list) -> 'MolTableData':
        dfs = [i.payload for i in items]
        return cls(payload=pd.concat(dfs, ignore_index=True),
                   mol_col=items[0].mol_col)

# ── Placeholder / error SVG constants ─────────────────────────────────────────
_PLACEHOLDER_SVG: bytes = b"""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300">
  <rect width="300" height="300" fill="none" stroke="#555" stroke-width="1.5"/>
  <text x="150" y="135" font-family="sans-serif" font-size="15"
        text-anchor="middle" fill="#777">No Molecule</text>
  <text x="150" y="160" font-family="sans-serif" font-size="15"
        text-anchor="middle" fill="#777">Selected</text>
</svg>
"""

_FAILED_SVG: bytes = b"""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300">
  <rect width="300" height="300" fill="none" stroke="#555" stroke-width="1.5"/>
  <text x="150" y="135" font-family="sans-serif" font-size="15"
        text-anchor="middle" fill="#c0392b">Invalid</text>
  <text x="150" y="160" font-family="sans-serif" font-size="15"
        text-anchor="middle" fill="#c0392b">SMILES</text>
</svg>
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _draw_mol_svg(mol, size: int = 400, dark: bool = False) -> bytes:
    """Render an RDKit Mol to SVG bytes with a transparent background."""
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    opts = drawer.drawOptions()
    if dark:
        SetDarkMode(opts)
        bg = '#000000'
    else:
        bg = '#FFFFFF'
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace(
        f"<rect style='opacity:1.0;fill:{bg}",
        "<rect style='opacity:1.0;fill:none",
        1,
    )
    return svg.encode('utf-8')


def _svg_bytes_to_pil(svg_bytes: bytes, size: int = 400) -> Image.Image:
    """Rasterise SVG bytes to a PIL RGB Image using Qt's SVG renderer."""
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg_bytes))
    img = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32)
    img.fill(QtCore.Qt.GlobalColor.white)
    painter = QtGui.QPainter(img)
    renderer.render(painter)
    painter.end()
    arr = np.frombuffer(img.bits(), dtype=np.uint8).reshape((size, size, 4)).copy()
    arr = arr[:, :, [2, 1, 0, 3]]   # BGRA -> RGBA
    return Image.fromarray(arr, 'RGBA').convert('RGB')


# ── 3D embedding & format conversion ─────────────────────────────────────────

_BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE:   sdfrust.BondOrder.single,
    Chem.rdchem.BondType.DOUBLE:   sdfrust.BondOrder.double,
    Chem.rdchem.BondType.TRIPLE:   sdfrust.BondOrder.triple,
    Chem.rdchem.BondType.AROMATIC: sdfrust.BondOrder.aromatic,
}

_FORMAT_EXTENSIONS = {
    'MOL2':  '.mol2',
    'SDF':   '.sdf',
    'PDB':   '.pdb',
    'XYZ':   '.xyz',
    'PDBQT': '.pdbqt',
}


def _rdkit_mol_to_sdfrust(mol, name: str = '') -> sdfrust.Molecule:
    """Convert an RDKit Mol (with 3D conformer) to a sdfrust.Molecule.

    The molecule is kekulized first so aromatic bonds become explicit
    single/double alternation.  MOL2's ``ar`` bond type causes kekulization
    failures when read back by RDKit, so explicit Kekulé form is safer.
    """
    mol = Chem.RWMol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=False)

    conf = mol.GetConformer()
    sdf_mol = sdfrust.Molecule(name or Chem.MolToSmiles(Chem.RemoveHs(mol)))

    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        sdf_atom = sdfrust.Atom(i, atom.GetSymbol(), pos.x, pos.y, pos.z)
        charge = atom.GetFormalCharge()
        if charge != 0:
            sdf_atom.formal_charge = charge
        sdf_mol.add_atom(sdf_atom)

    for bond in mol.GetBonds():
        order_fn = _BOND_TYPE_MAP.get(bond.GetBondType(), sdfrust.BondOrder.single)
        sdf_mol.add_bond(sdfrust.Bond(
            bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), order_fn(),
        ))

    return sdf_mol


def embed_mol_3d(
    mol,
    keep_hs: bool = False,
    optimize: bool = True,
    force_field: str = 'MMFF',
    num_confs: int = 1,
    max_iters: int = 0,
    prune_rms: float = -1.0,
    random_seed: int = -1,
    timeout: int = 0,
    random_coords_fallback: bool = False,
    num_threads: int = 0,
):
    """Add explicit Hs, embed in 3D (ETKDGv3), optionally optimize.

    Parameters
    ----------
    mol : RDKit Mol
    keep_hs : bool
        If False, remove explicit Hs from the result.
    optimize : bool
        Run force-field minimisation after embedding.
    force_field : str
        ``'MMFF'`` (Merck) or ``'UFF'`` (Universal).  Falls back to UFF
        automatically when MMFF lacks parameters for the molecule.
    num_confs : int
        Number of conformers to generate.  All conformers are kept on the
        output molecule.
    max_iters : int
        Maximum ETKDG iterations (0 = default ~1000).
    prune_rms : float
        RMSD threshold for pruning similar conformers (< 0 = no pruning).
    random_seed : int
        Seed for the random number generator (−1 = non-deterministic).
    timeout : int
        Per-conformer timeout in seconds (0 = no limit).
    random_coords_fallback : bool
        If True and distance-geometry embedding fails, automatically retry
        with random starting coordinates.
    num_threads : int
        Threads for embedding / optimisation (0 = use all cores).

    Returns the modified mol with all generated conformers, or None on
    failure.
    """
    mol = Chem.RWMol(mol)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = num_threads
    if max_iters > 0:
        params.maxIterations = max_iters
    if prune_rms >= 0:
        params.pruneRmsThresh = prune_rms
    if timeout > 0:
        params.timeout = timeout

    # --- Embed ----------------------------------------------------------------
    if num_confs <= 1:
        result = AllChem.EmbedMolecule(mol, params)
        if result < 0 and random_coords_fallback:
            params.useRandomCoords = True
            result = AllChem.EmbedMolecule(mol, params)
        if result < 0:
            return None
    else:
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        if not cids and random_coords_fallback:
            params.useRandomCoords = True
            cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        if not cids:
            return None

    # --- Optimize -------------------------------------------------------------
    if optimize:
        try:
            use_mmff = (force_field.upper() == 'MMFF'
                        and AllChem.MMFFHasAllMoleculeParams(mol))
            if use_mmff:
                AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_threads)
            else:
                AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=num_threads)
        except Exception:
            pass

    if not keep_hs:
        mol = Chem.RemoveHs(mol)
    return mol


def mol_to_format(mol, fmt: str, name: str = '') -> str:
    """Convert an RDKit Mol (with 3D conformer) to a string in the given format.

    Supported formats: 'mol2', 'sdf', 'pdb', 'xyz', 'smi', 'pdbqt' (requires meeko).
    """
    fmt = fmt.lower().strip()
    if fmt == 'mol2':
        sdf_mol = _rdkit_mol_to_sdfrust(mol, name=name)
        return sdfrust.write_mol2_string(sdf_mol)
    elif fmt == 'sdf':
        return Chem.MolToMolBlock(mol) + '$$$$\n'
    elif fmt == 'pdb':
        return Chem.MolToPDBBlock(mol)
    elif fmt == 'xyz':
        return Chem.MolToXYZBlock(mol)
    elif fmt == 'smi':
        return f'{Chem.MolToSmiles(mol)} {name}\n'
    elif fmt == 'pdbqt':
        prep = MoleculePreparation(rigid_macrocycles=False)
        mol_setups = prep.prepare(mol)
        for setup in mol_setups:
            pdbqt_str, is_ok, err_msg = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                return pdbqt_str
        raise RuntimeError(f"Meeko PDBQT conversion failed: {err_msg}")
    else:
        raise ValueError(f"Unsupported format: '{fmt}'")


# ── Node definitions ──────────────────────────────────────────────────────────

class SMILESInputNode(BaseExecutionNode):
    """Parse a SMILES string and output a Molecule object."""

    __identifier__ = 'nodes.Cheminformatics.Mol'
    NODE_NAME      = 'SMILES Input'
    PORT_SPEC      = {'inputs': [], 'outputs': ['molecule']}

    def __init__(self):
        super().__init__()
        self.add_text_input('smiles', 'SMILES', text='')
        self.add_output('molecule', color=PORT_COLORS['molecule'])

    def evaluate(self):
        smiles = (self.get_property('smiles') or '').strip()
        if not smiles:
            return False, "Enter a SMILES string."

        self.set_progress(10)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, f"Invalid SMILES: '{smiles}'"
        self.output_values['molecule'] = MoleculeData(payload=mol)
        self.set_progress(100)
        self.mark_clean()
        return True, None


class SMILESViewerNode(BaseExecutionNode):
    """Display a 2-D molecule structure diagram.

    Type a SMILES string (e.g. ``c1ccc2ccccc2c1`` for naphthalene) directly
    into the node. The structure renders in the image viewer panel and is also
    available on the ``image`` output port for downstream processing.
    """

    __identifier__ = 'nodes.Cheminformatics.Mol'
    NODE_NAME      = 'SMILES Viewer'
    PORT_SPEC      = {'inputs': [], 'outputs': ['image']}

    def __init__(self):
        super().__init__()
        self.add_text_input('smiles', 'SMILES', text='')
        self.add_checkbox('dark_mode', '', 'Dark Mode', state=False)
        self.add_output('image', color=PORT_COLORS['image'])

    def evaluate(self):
        smiles = (self.get_property('smiles') or '').strip()
        dark   = bool(self.get_property('dark_mode'))

        if not smiles:
            self.set_display(_PLACEHOLDER_SVG)
            self.output_values.pop('image', None)
            self.mark_clean()
            return True, None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self.set_display(_FAILED_SVG)
            return False, f"Invalid SMILES: '{smiles}'"

        self.set_progress(20)
        svg_bytes = _draw_mol_svg(mol, size=600, dark=dark)
        self.set_display(svg_bytes)
        self.set_progress(60)
        pil_img = _svg_bytes_to_pil(svg_bytes, size=600)
        self.output_values['image'] = ImageData(payload=pil_img)
        self.mark_clean()
        self.set_progress(100)
        return True, None


class MoleculeToImageNode(BaseExecutionNode):
    """Render a molecule structure as a 2-D diagram image.

    Accepts a Molecule object from SMILES Input and outputs an ImageData.
    """

    __identifier__ = 'nodes.Cheminformatics.Mol'
    NODE_NAME      = 'Molecule to Image'
    PORT_SPEC      = {'inputs': ['molecule'], 'outputs': ['image']}

    def __init__(self):
        super().__init__()
        self.add_checkbox('dark_mode', '', 'Dark Mode', state=False)
        self._add_int_spinbox('size', 'Size (px)', value=600, min_val=100, max_val=4096)
        self.add_input('molecule', color=PORT_COLORS['molecule'])
        self.add_output('image',   color=PORT_COLORS['image'])

    def evaluate(self):
        in_port = self.inputs().get('molecule')
        if not (in_port and in_port.connected_ports()):
            return False, "No molecule connected."
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, MoleculeData) or val.payload is None:
            return False, "Expected a valid MoleculeData on 'molecule'."

        mol  = val.payload
        size = max(100, int(self.get_property('size') or 600))
        dark = bool(self.get_property('dark_mode'))

        self.set_progress(20)
        svg_bytes = _draw_mol_svg(mol, size=size, dark=dark)
        self.set_display(svg_bytes)
        self.set_progress(60)
        pil_img = _svg_bytes_to_pil(svg_bytes, size=size)
        self.set_progress(90)
        self.output_values['image'] = ImageData(payload=pil_img)
        self.mark_clean()
        self.set_progress(100)
        return True, None


class MolecularDescriptorsNode(BaseExecutionNode):
    """Compute a table of physicochemical descriptors for a molecule.

    Outputs a DataFrame with one row containing: smiles, mol_weight, logp,
    hbd, hba, tpsa, rotatable_bonds, rings, aromatic_rings.
    """

    __identifier__ = 'nodes.Cheminformatics.Mol'
    NODE_NAME      = 'Molecular Descriptors'
    PORT_SPEC      = {'inputs': ['molecule'], 'outputs': ['table']}

    def __init__(self):
        super().__init__()
        self.add_input('molecule', color=PORT_COLORS['molecule'])
        self.add_output('table',   color=PORT_COLORS['table'])

    def evaluate(self):
        in_port = self.inputs().get('molecule')
        if not (in_port and in_port.connected_ports()):
            return False, "No molecule connected."
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, MoleculeData) or val.payload is None:
            return False, "Expected a valid MoleculeData on 'molecule'."

        mol = val.payload
        self.set_progress(30)

        row = {
            'smiles':            Chem.MolToSmiles(mol),
            'mol_weight':        round(Descriptors.MolWt(mol), 4),
            'logp':              round(Descriptors.MolLogP(mol), 4),
            'hbd':               rdMolDescriptors.CalcNumHBD(mol),
            'hba':               rdMolDescriptors.CalcNumHBA(mol),
            'tpsa':              round(Descriptors.TPSA(mol), 4),
            'rotatable_bonds':   rdMolDescriptors.CalcNumRotatableBonds(mol),
            'rings':             rdMolDescriptors.CalcNumRings(mol),
            'aromatic_rings':    rdMolDescriptors.CalcNumAromaticRings(mol),
        }
        df = pd.DataFrame([row])
        self.set_progress(90)
        self.output_values['table'] = TableData(payload=df)
        self.mark_clean()
        self.set_progress(100)
        return True, None


class SubstructureFilterNode(BaseExecutionNode):
    """Filter rows of a SMILES table by a SMARTS substructure pattern.

    Outputs two tables: matches (has the substructure) and rejects (does not).
    """

    __identifier__ = 'nodes.Cheminformatics.Convert'
    NODE_NAME      = 'Substructure Filter'
    PORT_SPEC      = {'inputs': ['table'], 'outputs': ['table', 'table']}

    def __init__(self):
        super().__init__()
        self.add_text_input('smiles_column', 'SMILES col', text='smiles')
        self.add_text_input('smarts',        'SMARTS',     text='')
        self.add_input('table',    color=PORT_COLORS['table'])
        self.add_output('matches', color=PORT_COLORS['table'])
        self.add_output('rejects', color=PORT_COLORS['table'])

    def evaluate(self):
        in_port = self.inputs().get('table')
        if not (in_port and in_port.connected_ports()):
            return False, "No table connected."
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, TableData):
            return False, "Expected TableData on 'table'."

        df          = val.payload.copy()
        col         = (self.get_property('smiles_column') or 'smiles').strip()
        smarts_str  = (self.get_property('smarts') or '').strip()

        if col not in df.columns:
            return False, f"Column '{col}' not found in table."
        if not smarts_str:
            return False, "Enter a SMARTS pattern."

        query = Chem.MolFromSmarts(smarts_str)
        if query is None:
            return False, f"Invalid SMARTS: '{smarts_str}'"

        self.set_progress(20)

        def _matches(smi):
            mol = Chem.MolFromSmiles(str(smi))
            return mol is not None and mol.HasSubstructMatch(query)

        mask = df[col].map(_matches)
        self.set_progress(90)
        self.output_values['matches'] = TableData(payload=df[mask].reset_index(drop=True))
        self.output_values['rejects'] = TableData(payload=df[~mask].reset_index(drop=True))
        self.mark_clean()
        self.set_progress(100)
        return True, None


class Mol3DEmbedNode(BaseExecutionNode):
    """Embed a molecule in 3D using ETKDGv3 and optionally optimize.

    Generates one or more 3D conformers, optionally runs force-field
    minimisation (MMFF or UFF).  All conformers are kept on the output
    molecule.
    """

    __identifier__ = 'nodes.Cheminformatics.Mol'
    NODE_NAME      = 'Mol 3D Embed'
    PORT_SPEC      = {'inputs': ['molecule'], 'outputs': ['molecule']}

    def __init__(self):
        super().__init__()
        self.add_checkbox('keep_hs', '', 'Keep Hydrogens', state=False)
        self.add_checkbox('optimize', '', 'Optimize', state=True)
        self.add_combo_menu('force_field', 'Force Field', items=['MMFF', 'UFF'])
        self._add_int_spinbox('num_confs', 'Num Conformers',
                              value=1, min_val=1, max_val=1000)
        self._add_int_spinbox('max_iters', 'Max Iterations (0=default)',
                              value=0, min_val=0, max_val=100000)
        self._add_float_spinbox('prune_rms', 'Prune RMSD (-1=off)',
                                value=-1.0, min_val=-1.0, max_val=100.0)
        self._add_int_spinbox('random_seed', 'Random Seed (-1=random)',
                              value=-1, min_val=-1, max_val=999999)
        self._add_int_spinbox('timeout', 'Timeout sec (0=none)',
                              value=0, min_val=0, max_val=3600)
        self.add_checkbox('random_coords_fallback', '',
                          'Random Coords Fallback', state=False)
        self.add_input('molecule',  color=PORT_COLORS['molecule'])
        self.add_output('molecule', color=PORT_COLORS['molecule'])

    def evaluate(self):
        in_port = self.inputs().get('molecule')
        if not (in_port and in_port.connected_ports()):
            return False, "No molecule connected."
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, MoleculeData) or val.payload is None:
            return False, "Expected a valid MoleculeData on 'molecule'."

        self.set_progress(10)

        mol3d = embed_mol_3d(
            val.payload,
            keep_hs=bool(self.get_property('keep_hs')),
            optimize=bool(self.get_property('optimize')),
            force_field=(self.get_property('force_field') or 'MMFF'),
            num_confs=max(1, int(self.get_property('num_confs') or 1)),
            max_iters=max(0, int(self.get_property('max_iters') or 0)),
            prune_rms=float(self.get_property('prune_rms') if
                            self.get_property('prune_rms') is not None else -1.0),
            random_seed=int(self.get_property('random_seed') if
                            self.get_property('random_seed') is not None else -1),
            timeout=max(0, int(self.get_property('timeout') or 0)),
            random_coords_fallback=bool(
                self.get_property('random_coords_fallback')),
        )
        if mol3d is None:
            return False, "3D embedding failed (ETKDG could not find a conformer)."

        n = mol3d.GetNumConformers()
        self.set_progress(90)
        self.output_values['molecule'] = MoleculeData(payload=mol3d)
        self.mark_clean()
        self.set_progress(100)
        return True, f"{n} conformer{'s' if n != 1 else ''} generated."


class MolFileWriterNode(BaseExecutionNode):
    """Write a molecule to a 3D file format.

    Accepts a Molecule with a 3D conformer (e.g. from Mol 3D Embed) and
    writes it to disk. MOL2 uses sdfrust; SDF/PDB/XYZ use RDKit;
    PDBQT uses Meeko (optional).
    """

    __identifier__ = 'nodes.Cheminformatics.Mol'
    NODE_NAME      = 'Mol File Writer'
    PORT_SPEC      = {'inputs': ['molecule'], 'outputs': []}

    def __init__(self):
        super().__init__()
        self.add_combo_menu('format', 'Format',
                            items=['MOL2', 'SDF', 'PDB', 'XYZ', 'PDBQT'])
        file_selector = NodeFileSaver(self.view, name='file_path', label='Save Path')
        self.add_custom_widget(
            file_selector,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties'
        )
        self.add_input('molecule', color=PORT_COLORS['molecule'])

    def evaluate(self):
        in_port = self.inputs().get('molecule')
        if not (in_port and in_port.connected_ports()):
            return False, "No molecule connected."
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, MoleculeData) or val.payload is None:
            return False, "Expected a valid MoleculeData on 'molecule'."

        mol      = val.payload
        fmt      = (self.get_property('format') or 'MOL2').upper()
        out_path = (self.get_property('file_path') or '').strip()

        if not out_path:
            return False, "Choose an output file path."

        if mol.GetNumConformers() == 0:
            return False, "Molecule has no 3D conformer. Connect a Mol 3D Embed node first."

        ext = _FORMAT_EXTENSIONS.get(fmt, '.mol2')
        out_file = Path(out_path).expanduser()
        if out_file.suffix.lower() != ext:
            out_file = out_file.with_suffix(ext)

        self.set_progress(20)

        name = Chem.MolToSmiles(Chem.RemoveHs(mol))
        text = mol_to_format(mol, fmt, name=name)

        self.set_progress(70)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(text)

        self.mark_clean()
        self.set_progress(100)
        return True, None


# ── Batch processing helpers ─────────────────────────────────────────────────

_READER_EXTENSIONS = {
    '.sdf', '.sd', '.mol', '.mol2', '.smi', '.smiles',
    '.csv', '.tsv', '.txt', '.pdb', '.xyz',
}


def _name_from_mol(mol, fallback: str = '') -> str:
    """Extract a name from an RDKit Mol's properties, or return *fallback*."""
    for prop in ('_Name', 'IDNUMBER', 'Name', 'name', 'ID', 'id',
                 'ChemicalName', 'CHEMBL_ID', 'zinc_id', 'ZINC_ID'):
        if mol.HasProp(prop):
            v = mol.GetProp(prop).strip()
            if v:
                return v
    return fallback


def _read_mols_from_file(path: Path) -> list[tuple[str, RDMol]]:
    """Read molecules from a single file.  Returns list of (name, mol) tuples.

    Uses MultithreadedSDMolSupplier / MultithreadedSmilesMolSupplier where
    available for .sdf and .smi files.  Other formats use the appropriate
    single-molecule reader.
    """
    ext = path.suffix.lower()
    results: list[tuple[str, RDMol]] = []
    stem = path.stem

    if ext in ('.sdf', '.sd'):
        try:
            supp = Chem.MultithreadedSDMolSupplier(str(path))
        except Exception:
            supp = Chem.SDMolSupplier(str(path))
        for i, mol in enumerate(supp):
            if mol is None:
                continue
            name = _name_from_mol(mol, f'{stem}_{i + 1}')
            results.append((name, mol))

    elif ext in ('.smi', '.smiles', '.txt'):
        try:
            supp = Chem.MultithreadedSmilesMolSupplier(str(path))
        except Exception:
            supp = Chem.SmilesMolSupplier(str(path), titleLine=False)
        for i, mol in enumerate(supp):
            if mol is None:
                continue
            name = _name_from_mol(mol, f'{stem}_{i + 1}')
            results.append((name, mol))

    elif ext in ('.csv', '.tsv'):
        sep = '\t' if ext == '.tsv' else ','
        df = pd.read_csv(str(path), sep=sep)
        # Auto-detect SMILES column
        smi_col = None
        for col in df.columns:
            if col.lower() in ('smiles', 'smi', 'canonical_smiles', 'isosmiles'):
                smi_col = col
                break
        if smi_col is None:
            smi_col = df.columns[0]
        # Auto-detect name column
        name_col = None
        for col in df.columns:
            if col.lower() in ('name', 'id', '_name', 'chembl_id', 'title',
                                'compound_name', 'mol_name', 'catalog id'):
                name_col = col
                break
        for i, row in df.iterrows():
            smi = str(row[smi_col]).strip()
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            n = str(row[name_col]).strip() if name_col else f'{stem}_{i + 1}'
            results.append((n, mol))

    elif ext == '.mol2':
        mol = Chem.MolFromMol2File(str(path))
        if mol is not None:
            results.append((_name_from_mol(mol, stem), mol))

    elif ext == '.mol':
        mol = Chem.MolFromMolFile(str(path))
        if mol is not None:
            results.append((_name_from_mol(mol, stem), mol))

    elif ext == '.pdb':
        mol = Chem.MolFromPDBFile(str(path))
        if mol is not None:
            results.append((_name_from_mol(mol, stem), mol))

    elif ext == '.pdbqt':
        from .protein_utils import PDBQTMolecule, RDKitMolCreate
        with open(path) as f:
            pdbqt_str = f.read()
        try:
            pdbqt_mol = PDBQTMolecule(pdbqt_str, skip_typing=True)
            output_string, _ = RDKitMolCreate.write_sd_string(pdbqt_mol)
            supp = Chem.SDMolSupplier()
            supp.SetData(output_string)
            for i, mol in enumerate(supp):
                if mol is None:
                    continue
                name = _name_from_mol(mol, f'{stem}_{i + 1}' if i > 0 else stem)
                results.append((name, mol))
        except Exception:
            pass

    elif ext == '.xyz':
        mol = Chem.MolFromXYZFile(str(path))
        if mol is not None:
            results.append((stem, mol))

    return results


def _build_mol_table(pairs: list[tuple[str, RDMol]]) -> pd.DataFrame:
    """Build a DataFrame with name, smiles, and ROMol columns from (name, mol) pairs."""
    names, smiles_list, mols = [], [], []
    seen_names: dict[str, int] = {}
    for name, mol in pairs:
        # Deduplicate names
        if name in seen_names:
            seen_names[name] += 1
            name = f'{name}_{seen_names[name]}'
        else:
            seen_names[name] = 0
        names.append(name)
        smiles_list.append(Chem.MolToSmiles(mol))
        mols.append(mol)
    return pd.DataFrame({'name': names, 'smiles': smiles_list, 'ROMol': mols})


# ── Batch node definitions ───────────────────────────────────────────────────

class MolReaderNode(BaseExecutionNode):
    """Read molecules from a file or all files in a directory.

    Supported formats: SDF, SMI, CSV/TSV, MOL, MOL2, PDB, XYZ.
    Uses RDKit's threaded suppliers (MultithreadedSDMolSupplier,
    MultithreadedSmilesMolSupplier) for SDF and SMILES files.
    For directories, reads all matching files with ThreadPoolExecutor.
    """

    __identifier__ = 'nodes.Cheminformatics.IO'
    NODE_NAME      = 'Mol Reader'
    PORT_SPEC      = {'inputs': [], 'outputs': ['mol_table']}

    def __init__(self):
        super().__init__()
        self.add_combo_menu('source_type', 'Source', items=['File', 'Directory'])
        file_selector = NodeFileSelector(
            self.view, name='file_path', label='File',
            ext_filter='Molecule Files (*.sdf *.sd *.smi *.smiles *.csv *.tsv '
                        '*.txt *.mol *.mol2 *.pdb *.xyz);;All Files (*)')
        self.add_custom_widget(
            file_selector,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties')
        dir_selector = NodeDirSelector(self.view, name='dir_path', label='Directory')
        self.add_custom_widget(
            dir_selector,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties')
        self.add_text_input('ext_filter', 'Extensions', text='.sdf,.smi,.mol2')
        self.add_output('mol_table', color=PORT_COLORS['mol_table'])

    def evaluate(self):
        source = self.get_property('source_type') or 'File'
        self.set_progress(5)

        if source == 'File':
            fpath = (self.get_property('file_path') or '').strip()
            if not fpath:
                return False, "Select a molecule file."
            p = Path(fpath).expanduser()
            if not p.is_file():
                return False, f"File not found: {p}"
            pairs = _read_mols_from_file(p)
        else:
            dpath = (self.get_property('dir_path') or '').strip()
            if not dpath:
                return False, "Select a directory."
            d = Path(dpath).expanduser()
            if not d.is_dir():
                return False, f"Directory not found: {d}"
            ext_str = (self.get_property('ext_filter') or '.sdf,.smi,.mol2').strip()
            exts = {e.strip().lower() if e.strip().startswith('.')
                    else '.' + e.strip().lower()
                    for e in ext_str.split(',')}
            files = sorted(f for f in d.iterdir()
                           if f.is_file() and f.suffix.lower() in exts)
            if not files:
                return False, f"No matching files in {d}"
            self.set_progress(10)
            pairs: list[tuple[str, RDMol]] = []
            with ThreadPoolExecutor() as pool:
                futs = {pool.submit(_read_mols_from_file, f): f for f in files}
                for fut in futs:
                    pairs.extend(fut.result())

        if not pairs:
            return False, "No valid molecules found."

        self.set_progress(80)
        df = _build_mol_table(pairs)
        self.output_values['mol_table'] = MolTableData(payload=df)
        self.mark_clean()
        self.set_progress(100)
        return True, f"{len(df)} molecule{'s' if len(df) != 1 else ''} loaded."


class BatchDescriptorsNode(BaseExecutionNode):
    """Compute physicochemical descriptors for every molecule in a MolTable.

    Toggle common descriptors via checkboxes.  For any RDKit descriptor not
    listed, type comma-separated names in the *Custom* field (e.g.
    ``BalabanJ, FractionCSP3, ExactMolWt``).

    Uses ThreadPoolExecutor for parallelism (RDKit releases the GIL).
    """

    __identifier__ = 'nodes.Cheminformatics.Batch'
    NODE_NAME      = 'Batch Descriptors'
    PORT_SPEC      = {'inputs': ['mol_table'], 'outputs': ['mol_table']}

    # (checkbox_name, column_name, descriptor_function)
    _BUILTIN = [
        ('MW',              'mol_weight',      Descriptors.MolWt),
        ('LogP',            'logp',            Descriptors.MolLogP),
        ('HBD',             'hbd',             Descriptors.NumHDonors),
        ('HBA',             'hba',             Descriptors.NumHAcceptors),
        ('TPSA',            'tpsa',            Descriptors.TPSA),
        ('Rotatable Bonds', 'rotatable_bonds', Descriptors.NumRotatableBonds),
        ('Rings',           'rings',           rdMolDescriptors.CalcNumRings),
        ('Aromatic Rings',  'aromatic_rings',  rdMolDescriptors.CalcNumAromaticRings),
        ('Heavy Atoms',     'heavy_atoms',     Descriptors.HeavyAtomCount),
        ('Formal Charge',   'formal_charge',   lambda m: sum(a.GetFormalCharge() for a in m.GetAtoms())),
        ('Molar Refract.',  'molar_refract',   Descriptors.MolMR),
        ('QED',             'qed',             Descriptors.qed),
    ]

    def __init__(self):
        super().__init__()
        for label, _col, _fn in self._BUILTIN:
            default_on = label in ('MW', 'LogP', 'HBD', 'HBA', 'TPSA',
                                   'Rotatable Bonds', 'Rings', 'Aromatic Rings')
            self.add_checkbox(f'desc_{label}', '', label, state=default_on)

        self.add_text_input('custom_descs', 'Custom', text='')

        self.add_input('mol_table',  color=PORT_COLORS['mol_table'])
        self.add_output('mol_table', color=PORT_COLORS['mol_table'])

    def evaluate(self):
        in_port = self.inputs().get('mol_table')
        if not (in_port and in_port.connected_ports()):
            return False, "No mol_table connected."
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, MolTableData):
            return False, "Expected MolTableData."

        df = val.payload.copy()
        mol_col = val.mol_col
        if mol_col not in df.columns:
            return False, f"Mol column '{mol_col}' not in table."

        # Gather enabled built-in descriptors
        active = []  # list of (column_name, callable)
        for label, col, fn in self._BUILTIN:
            if self.get_property(f'desc_{label}'):
                active.append((col, fn))

        # Parse custom descriptors
        custom_raw = (self.get_property('custom_descs') or '').strip()
        bad_names = []
        if custom_raw:
            for name in (n.strip() for n in custom_raw.split(',')):
                if not name:
                    continue
                fn = _DESC_DICT.get(name)
                if fn is not None:
                    active.append((name, fn))
                elif name == 'NumAtoms':
                    active.append((name, lambda m: float(m.GetNumAtoms())))
                elif name == 'FormalCharge':
                    active.append((name, lambda m: float(sum(a.GetFormalCharge() for a in m.GetAtoms()))))
                else:
                    bad_names.append(name)

        if bad_names:
            return False, f"Unknown descriptors: {', '.join(bad_names)}"
        if not active:
            return False, "No descriptors selected."

        self.set_progress(10)
        mols = df[mol_col].tolist()

        def _compute(mol):
            row = {}
            for col, fn in active:
                try:
                    v = fn(mol)
                    row[col] = round(v, 4) if isinstance(v, float) else v
                except Exception:
                    row[col] = np.nan
            return row

        with ThreadPoolExecutor() as pool:
            rows = list(pool.map(_compute, mols))

        desc_df = pd.DataFrame(rows)
        for col in desc_df.columns:
            df[col] = desc_df[col].values
        self.set_progress(90)
        self.output_values['mol_table'] = MolTableData(payload=df, mol_col=mol_col)
        self.mark_clean()
        self.set_progress(100)
        return True, f"{len(active)} descriptors × {len(df)} molecules"


class Batch3DEmbedNode(BaseExecutionNode):
    """Embed all molecules in a MolTable in 3D using ETKDGv3.

    Failed embeddings are dropped from the output table.
    Uses ThreadPoolExecutor for parallelism (RDKit releases the GIL
    during embedding/optimisation).
    """

    __identifier__ = 'nodes.Cheminformatics.Batch'
    NODE_NAME      = 'Batch 3D Embed'
    PORT_SPEC      = {'inputs': ['mol_table'], 'outputs': ['mol_table']}

    def __init__(self):
        super().__init__()
        self.add_checkbox('keep_hs', '', 'Keep Hydrogens', state=False)
        self.add_checkbox('optimize', '', 'Optimize', state=True)
        self.add_combo_menu('force_field', 'Force Field', items=['MMFF', 'UFF'])
        self._add_int_spinbox('timeout', 'Timeout sec (0=none)',
                              value=0, min_val=0, max_val=3600)
        self.add_checkbox('random_coords_fallback', '',
                          'Random Coords Fallback', state=False)
        self.add_input('mol_table',  color=PORT_COLORS['mol_table'])
        self.add_output('mol_table', color=PORT_COLORS['mol_table'])

    def evaluate(self):
        in_port = self.inputs().get('mol_table')
        if not (in_port and in_port.connected_ports()):
            return False, "No mol_table connected."
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, MolTableData):
            return False, "Expected MolTableData."

        df = val.payload.copy()
        mol_col = val.mol_col
        if mol_col not in df.columns:
            return False, f"Mol column '{mol_col}' not in table."

        keep_hs  = bool(self.get_property('keep_hs'))
        optimize = bool(self.get_property('optimize'))
        ff       = self.get_property('force_field') or 'MMFF'
        timeout  = max(0, int(self.get_property('timeout') or 0))
        fallback = bool(self.get_property('random_coords_fallback'))

        self.set_progress(5)

        def _embed_one(mol):
            return embed_mol_3d(mol, keep_hs=keep_hs, optimize=optimize,
                                force_field=ff, timeout=timeout,
                                random_coords_fallback=fallback)

        mols = df[mol_col].tolist()
        with ThreadPoolExecutor() as pool:
            embedded = list(pool.map(_embed_one, mols))

        self.set_progress(85)
        # Keep only rows where embedding succeeded
        mask = [m is not None for m in embedded]
        df = df[mask].copy()
        df[mol_col] = [m for m in embedded if m is not None]
        df = df.reset_index(drop=True)

        n_ok = len(df)
        n_fail = len(mols) - n_ok
        self.output_values['mol_table'] = MolTableData(payload=df, mol_col=mol_col)
        self.mark_clean()
        self.set_progress(100)
        msg = f"{n_ok} embedded"
        if n_fail:
            msg += f", {n_fail} failed"
        return True, msg


class BatchFileWriterNode(BaseExecutionNode):
    """Write all molecules in a MolTable to disk.

    Can write a single multi-record file (SDF, SMI) or individual files
    per molecule (MOL2, SDF, PDB, XYZ) into a chosen directory.
    """

    __identifier__ = 'nodes.Cheminformatics.Batch'
    NODE_NAME      = 'Batch File Writer'
    PORT_SPEC      = {'inputs': ['mol_table'], 'outputs': []}

    def __init__(self):
        super().__init__()
        self.add_combo_menu('mode', 'Mode',
                            items=['Single File', 'One File Per Mol'])
        self.add_combo_menu('format', 'Format',
                            items=['SDF', 'SMI', 'MOL2', 'PDB', 'XYZ', 'PDBQT'])
        file_selector = NodeFileSaver(self.view, name='file_path', label='Output File')
        self.add_custom_widget(
            file_selector,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties')
        dir_selector = NodeDirSelector(self.view, name='dir_path', label='Output Dir')
        self.add_custom_widget(
            dir_selector,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties')
        self.add_input('mol_table', color=PORT_COLORS['mol_table'])

    def evaluate(self):
        in_port = self.inputs().get('mol_table')
        if not (in_port and in_port.connected_ports()):
            return False, "No mol_table connected."
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, MolTableData):
            return False, "Expected MolTableData."

        df       = val.payload
        mol_col  = val.mol_col
        mode     = self.get_property('mode') or 'Single File'
        fmt      = (self.get_property('format') or 'SDF').upper()

        if mol_col not in df.columns:
            return False, f"Mol column '{mol_col}' not in table."

        self.set_progress(5)

        if mode == 'Single File':
            out_path = (self.get_property('file_path') or '').strip()
            if not out_path:
                return False, "Choose an output file path."
            ext = _FORMAT_EXTENSIONS.get(fmt, '.sdf')
            out_file = Path(out_path).expanduser()
            if out_file.suffix.lower() != ext:
                out_file = out_file.with_suffix(ext)
            out_file.parent.mkdir(parents=True, exist_ok=True)

            parts = []
            for _, row in df.iterrows():
                mol = row[mol_col]
                name = row.get('name', '')
                if mol is None:
                    continue
                parts.append(mol_to_format(mol, fmt, name=name))
            out_file.write_text(''.join(parts))
            self.mark_clean()
            self.set_progress(100)
            return True, f"{len(parts)} molecules written to {out_file.name}"

        else:  # One File Per Mol
            dir_path = (self.get_property('dir_path') or '').strip()
            if not dir_path:
                return False, "Choose an output directory."
            out_dir = Path(dir_path).expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            ext = _FORMAT_EXTENSIONS.get(fmt, '.sdf')
            written = 0
            for _, row in df.iterrows():
                mol = row[mol_col]
                name = row.get('name', f'mol_{written}')
                if mol is None:
                    continue
                # Sanitize filename
                safe = "".join(c if c.isalnum() or c in '-_.' else '_' for c in name)
                fpath = out_dir / (safe + ext)
                text = mol_to_format(mol, fmt, name=name)
                fpath.write_text(text)
                written += 1
            self.mark_clean()
            self.set_progress(100)
            return True, f"{written} files written to {out_dir.name}/"


class SubstructureFilterBatchNode(BaseExecutionNode):
    """Filter a MolTable by SMARTS substructure pattern.

    Splits into two outputs: matches (has substructure) and rejects.
    Operates directly on the Mol objects — no SMILES re-parsing.
    """

    __identifier__ = 'nodes.Cheminformatics.Batch'
    NODE_NAME      = 'Batch Substructure Filter'
    PORT_SPEC      = {'inputs': ['mol_table'],
                      'outputs': ['mol_table', 'mol_table']}

    def __init__(self):
        super().__init__()
        self.add_text_input('smarts', 'SMARTS', text='')
        self.add_input('mol_table', color=PORT_COLORS['mol_table'])
        self.add_output('matches',  color=PORT_COLORS['mol_table'])
        self.add_output('rejects',  color=PORT_COLORS['mol_table'])

    def evaluate(self):
        in_port = self.inputs().get('mol_table')
        if not (in_port and in_port.connected_ports()):
            return False, "No mol_table connected."
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, MolTableData):
            return False, "Expected MolTableData."

        smarts_str = (self.get_property('smarts') or '').strip()
        if not smarts_str:
            return False, "Enter a SMARTS pattern."
        query = Chem.MolFromSmarts(smarts_str)
        if query is None:
            return False, f"Invalid SMARTS: '{smarts_str}'"

        df = val.payload.copy()
        mol_col = val.mol_col

        self.set_progress(10)
        mask = df[mol_col].apply(lambda m: m is not None and m.HasSubstructMatch(query))
        self.set_progress(90)

        self.output_values['matches'] = MolTableData(
            payload=df[mask].reset_index(drop=True), mol_col=mol_col)
        self.output_values['rejects'] = MolTableData(
            payload=df[~mask].reset_index(drop=True), mol_col=mol_col)
        self.mark_clean()
        self.set_progress(100)
        n = mask.sum()
        return True, f"{n} match, {len(df) - n} reject"


class MolTableToMoleculeNode(BaseExecutionNode):
    """Pick a single molecule from a MolTable by row index.

    Bridges batch (mol_table) to single-molecule nodes (molecule port).
    """

    __identifier__ = 'nodes.Cheminformatics.Convert'
    NODE_NAME      = 'MolTable to Molecule'
    PORT_SPEC      = {'inputs': ['mol_table'], 'outputs': ['molecule']}

    def __init__(self):
        super().__init__()
        self._add_int_spinbox('row_index', 'Row Index', value=0, min_val=0, max_val=999999)
        self.add_input('mol_table', color=PORT_COLORS['mol_table'])
        self.add_output('molecule', color=PORT_COLORS['molecule'])

    def evaluate(self):
        in_port = self.inputs().get('mol_table')
        if not (in_port and in_port.connected_ports()):
            return False, "No mol_table connected."
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, MolTableData):
            return False, "Expected MolTableData."

        df = val.payload
        mol_col = val.mol_col
        idx = int(self.get_property('row_index') or 0)

        if idx < 0 or idx >= len(df):
            return False, f"Row {idx} out of range (table has {len(df)} rows)."

        row = df.iloc[idx]
        mol = row[mol_col]
        name = row.get('name', '')
        if mol is None:
            return False, f"Row {idx} has no valid molecule."

        self.output_values['molecule'] = MoleculeData(payload=mol, name=name)
        self.mark_clean()
        self.set_progress(100)
        return True, f"'{name}' (row {idx})"


class MolTableToTableNode(BaseExecutionNode):
    """Convert a MolTable to a plain Table by dropping the Mol column.

    Useful for connecting to existing table nodes (Sort, Filter, Plot, etc.).
    """

    __identifier__ = 'nodes.Cheminformatics.Convert'
    NODE_NAME      = 'MolTable to Table'
    PORT_SPEC      = {'inputs': ['mol_table'], 'outputs': ['table']}

    def __init__(self):
        super().__init__()
        self.add_input('mol_table', color=PORT_COLORS['mol_table'])
        self.add_output('table',    color=PORT_COLORS['table'])

    def evaluate(self):
        in_port = self.inputs().get('mol_table')
        if not (in_port and in_port.connected_ports()):
            return False, "No mol_table connected."
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, MolTableData):
            return False, "Expected MolTableData."

        df = val.payload.drop(columns=[val.mol_col], errors='ignore')
        self.output_values['table'] = TableData(payload=df)
        self.mark_clean()
        self.set_progress(100)
        return True, f"{len(df)} rows"


# ── Preset property names for the combo box ──────────────────────────────────
# Maps friendly label → RDKit descriptor name (or callable for special cases)
_PRESET_PROPERTIES = [
    ('MW',              'MolWt'),
    ('LogP',            'MolLogP'),
    ('HBD',             'NumHDonors'),
    ('HBA',             'NumHAcceptors'),
    ('TPSA',            'TPSA'),
    ('Rotatable Bonds', 'NumRotatableBonds'),
    ('Rings',           'RingCount'),
    ('Heavy Atoms',     'HeavyAtomCount'),
    ('Num Atoms',       'NumAtoms'),          # special: mol.GetNumAtoms()
    ('Formal Charge',   'FormalCharge'),       # special: sum of atom charges
    ('Molar Refract.',  'MolMR'),
    ('Fraction CSP3',   'FractionCSP3'),
    ('QED',             'qed'),
    ('Custom',          None),
]
_PRESET_LABELS = [p[0] for p in _PRESET_PROPERTIES]

# Build fast lookup: descriptor name → callable
_DESC_DICT = dict(Descriptors.descList)  # name → function for all ~200 descs

# Map RDKit descriptor name → column name produced by BatchDescriptorsNode.
# When a column already exists in the table, we skip recomputation.
_DESC_TO_COLUMN = {
    'MolWt':              'mol_weight',
    'MolLogP':            'logp',
    'NumHDonors':         'hbd',
    'NumHAcceptors':      'hba',
    'TPSA':               'tpsa',
    'NumRotatableBonds':  'rotatable_bonds',
    'RingCount':          'rings',
    'NumAromaticRings':   'aromatic_rings',
    'HeavyAtomCount':     'heavy_atoms',
    'FormalCharge':       'formal_charge',
    'MolMR':              'molar_refract',
    'qed':                'qed',
}

def _compute_descriptor(mol, desc_name: str) -> float:
    """Compute a single descriptor value by name."""
    if desc_name == 'NumAtoms':
        return float(mol.GetNumAtoms())
    if desc_name == 'FormalCharge':
        return float(sum(a.GetFormalCharge() for a in mol.GetAtoms()))
    fn = _DESC_DICT.get(desc_name)
    if fn is not None:
        return float(fn(mol))
    raise KeyError(f"Unknown descriptor: {desc_name}")


_CMP_OPS = {
    '<':  lambda v, t: v < t,
    '>':  lambda v, t: v > t,
    '\u2264': lambda v, t: v <= t,
    '\u2265': lambda v, t: v >= t,
    '=':  lambda v, t: abs(v - t) < 1e-6,
}


class PropertyFilterNode(BaseExecutionNode):
    """Filter a MolTable by a single molecular property.

    Pick a common property from the dropdown **or** select *Custom* and type
    any RDKit descriptor name (e.g. ``BalabanJ``, ``ExactMolWt``).

    Choose a comparison operator (<, >, \u2264, \u2265, =) and a threshold value.

    Combine multiple PropertyFilterNodes with a MolTable Merge node to build
    complex AND / OR filter chains.
    """

    __identifier__ = 'nodes.Cheminformatics.Batch'
    NODE_NAME      = 'Property Filter'
    PORT_SPEC      = {'inputs': ['mol_table'],
                      'outputs': ['mol_table', 'mol_table']}

    def __init__(self):
        super().__init__()
        self.add_combo_menu('property', 'Property', items=_PRESET_LABELS)
        self.add_text_input('custom_desc', 'Descriptor', text='')
        self.add_combo_menu('operator', 'Operator',
                            items=['<', '>', '\u2264', '\u2265', '='])
        self._add_float_spinbox('threshold', 'Value',
                                value=500.0, min_val=-1e6, max_val=1e6,
                                step=1.0, decimals=3)

        self.add_input('mol_table',  color=PORT_COLORS['mol_table'])
        self.add_output('matches',   color=PORT_COLORS['mol_table'])
        self.add_output('rejects',   color=PORT_COLORS['mol_table'])

    def evaluate(self):
        in_port = self.inputs().get('mol_table')
        if not (in_port and in_port.connected_ports()):
            return False, "No mol_table connected."
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, MolTableData):
            return False, "Expected MolTableData."

        df = val.payload
        mol_col = val.mol_col
        if len(df) == 0:
            self.output_values['matches'] = MolTableData(payload=df.copy(), mol_col=mol_col)
            self.output_values['rejects'] = MolTableData(payload=df.copy(), mol_col=mol_col)
            self.mark_clean()
            self.set_progress(100)
            return True, "Empty table."

        # Resolve descriptor
        prop_label = self.get_property('property') or 'MW'
        if prop_label == 'Custom':
            desc_name = (self.get_property('custom_desc') or '').strip()
            if not desc_name:
                return False, "Select Custom but no descriptor name entered."
            if desc_name not in _DESC_DICT and desc_name not in ('NumAtoms', 'FormalCharge'):
                return False, f"Unknown descriptor: '{desc_name}'"
        else:
            desc_name = dict(_PRESET_PROPERTIES).get(prop_label)
            if desc_name is None:
                return False, f"No descriptor mapped for '{prop_label}'."

        op_str = self.get_property('operator') or '\u2265'
        cmp_fn = _CMP_OPS.get(op_str)
        if cmp_fn is None:
            return False, f"Unknown operator: '{op_str}'"
        threshold = float(self.get_property('threshold') or 0.0)

        self.set_progress(10)

        n = len(df)
        # Check if a precomputed column exists (from BatchDescriptorsNode)
        col_name = _DESC_TO_COLUMN.get(desc_name)
        # Also check if Custom descriptor name matches a column directly
        if col_name is None and desc_name in df.columns:
            col_name = desc_name

        if col_name and col_name in df.columns:
            # Reuse precomputed values — vectorised, very fast
            vals = pd.to_numeric(df[col_name], errors='coerce').values
            pass_mask = np.array([cmp_fn(v, threshold) if np.isfinite(v) else False
                                  for v in vals], dtype=bool)
        else:
            # Compute from Mol objects
            pass_mask = np.zeros(n, dtype=bool)
            for i, mol in enumerate(df[mol_col]):
                if mol is not None:
                    try:
                        val_i = _compute_descriptor(mol, desc_name)
                        pass_mask[i] = cmp_fn(val_i, threshold)
                    except Exception:
                        pass_mask[i] = False

        self.set_progress(80)

        match_df = df[pass_mask].reset_index(drop=True)
        reject_df = df[~pass_mask].reset_index(drop=True)

        self.output_values['matches'] = MolTableData(payload=match_df, mol_col=mol_col)
        self.output_values['rejects'] = MolTableData(payload=reject_df, mol_col=mol_col)
        self.mark_clean()
        self.set_progress(100)
        return True, (f"{prop_label if prop_label != 'Custom' else desc_name} "
                      f"{op_str} {threshold:g} \u2192 "
                      f"{len(match_df)} match, {len(reject_df)} reject")


# ── Drug-likeness preset filter rules ────────────────────────────────────────
# Each preset is a list of (descriptor_name, operator_str, value).
# All rules within a preset are AND'd.
_DRUGLIKE_PRESETS = {
    "Lipinski's": [
        ('MolWt', '\u2264', 500), ('MolLogP', '\u2264', 5),
        ('NumHDonors', '\u2264', 5), ('NumHAcceptors', '\u2264', 10),
    ],
    'Veber': [
        ('TPSA', '\u2264', 140), ('NumRotatableBonds', '\u2264', 10),
    ],
    'Egan': [
        ('MolLogP', '\u2264', 5.88), ('TPSA', '\u2264', 131.6),
    ],
    'Ghose': [
        ('MolWt', '>', 160), ('MolWt', '<', 480),
        ('MolLogP', '>', -0.4), ('MolLogP', '<', 5.6),
        ('MolMR', '>', 40), ('MolMR', '<', 130),
        ('NumAtoms', '>', 20), ('NumAtoms', '<', 70),
    ],
    'Lead-like': [
        ('MolWt', '\u2264', 300), ('MolLogP', '\u2264', 3),
        ('NumHDonors', '\u2264', 3), ('NumHAcceptors', '\u2264', 3),
    ],
    'REOS': [
        ('MolWt', '>', 200), ('MolWt', '<', 500),
        ('MolLogP', '>', -5), ('MolLogP', '<', 5),
        ('NumHDonors', '<', 5), ('NumHAcceptors', '<', 10),
        ('NumRotatableBonds', '<', 8),
        ('FormalCharge', '>', -2), ('FormalCharge', '<', 2),
        ('HeavyAtomCount', '>', 15), ('HeavyAtomCount', '<', 50),
    ],
    'Murcko': [
        ('MolWt', '\u2265', 200), ('MolWt', '\u2264', 400),
        ('MolLogP', '\u2264', 5.2), ('NumHDonors', '\u2264', 3),
        ('NumHAcceptors', '\u2264', 4), ('NumRotatableBonds', '\u2264', 7),
    ],
    'Van de Waterbeemd': [
        ('MolWt', '\u2264', 450), ('TPSA', '\u2264', 90),
    ],
    'Palm': [
        ('TPSA', '\u2264', 140),
    ],
    'PPI': [
        ('MolWt', '\u2265', 400), ('NumHAcceptors', '\u2265', 4),
        ('MolLogP', '\u2265', 4), ('RingCount', '\u2265', 4),
    ],
}


class DrugLikenessFilterNode(BaseExecutionNode):
    """Apply a classic drug-likeness rule set to a MolTable.

    All rules within the chosen preset are AND'd.
    Outputs *matches* (pass all rules) and *rejects* (fail at least one).
    """

    __identifier__ = 'nodes.Cheminformatics.Batch'
    NODE_NAME      = 'Drug-likeness Filter'
    PORT_SPEC      = {'inputs': ['mol_table'],
                      'outputs': ['mol_table', 'mol_table']}

    def __init__(self):
        super().__init__()
        self.add_combo_menu('preset', 'Preset',
                            items=list(_DRUGLIKE_PRESETS.keys()))

        self.add_input('mol_table',  color=PORT_COLORS['mol_table'])
        self.add_output('matches',   color=PORT_COLORS['mol_table'])
        self.add_output('rejects',   color=PORT_COLORS['mol_table'])

    def evaluate(self):
        in_port = self.inputs().get('mol_table')
        if not (in_port and in_port.connected_ports()):
            return False, "No mol_table connected."
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, MolTableData):
            return False, "Expected MolTableData."

        df = val.payload
        mol_col = val.mol_col
        if len(df) == 0:
            self.output_values['matches'] = MolTableData(payload=df.copy(), mol_col=mol_col)
            self.output_values['rejects'] = MolTableData(payload=df.copy(), mol_col=mol_col)
            self.mark_clean()
            self.set_progress(100)
            return True, "Empty table."

        preset_name = self.get_property('preset') or "Lipinski's"
        rules = _DRUGLIKE_PRESETS.get(preset_name)
        if not rules:
            return False, f"Unknown preset: '{preset_name}'"

        self.set_progress(10)
        n = len(df)
        pass_mask = np.ones(n, dtype=bool)

        for desc_name, op_str, threshold in rules:
            cmp_fn = _CMP_OPS.get(op_str)
            if cmp_fn is None:
                continue

            # Try to reuse a precomputed column
            col_name = _DESC_TO_COLUMN.get(desc_name)
            if col_name and col_name in df.columns:
                vals = pd.to_numeric(df[col_name], errors='coerce').values
                for i in range(n):
                    if not pass_mask[i]:
                        continue
                    if not np.isfinite(vals[i]) or not cmp_fn(vals[i], threshold):
                        pass_mask[i] = False
            else:
                for i, mol in enumerate(df[mol_col]):
                    if not pass_mask[i]:
                        continue
                    if mol is None:
                        pass_mask[i] = False
                        continue
                    try:
                        v = _compute_descriptor(mol, desc_name)
                        if not cmp_fn(v, threshold):
                            pass_mask[i] = False
                    except Exception:
                        pass_mask[i] = False

        self.set_progress(80)

        match_df = df[pass_mask].reset_index(drop=True)
        reject_df = df[~pass_mask].reset_index(drop=True)

        self.output_values['matches'] = MolTableData(payload=match_df, mol_col=mol_col)
        self.output_values['rejects'] = MolTableData(payload=reject_df, mol_col=mol_col)
        self.mark_clean()
        self.set_progress(100)
        return True, f"{preset_name}: {len(match_df)} pass, {len(reject_df)} fail"


class MolTableMergeNode(BaseExecutionNode):
    """Combine two MolTables with AND or OR logic.

    **AND** — keep only molecules whose *name* appears in **both** inputs
    (intersection).  Rows are taken from input A.

    **OR** — keep molecules from **either** input (union, duplicates by
    name removed, first occurrence kept).

    Pair with PropertyFilterNode / DrugLikenessFilterNode to build
    complex filter chains.
    """

    __identifier__ = 'nodes.Cheminformatics.Batch'
    NODE_NAME      = 'MolTable Merge'
    PORT_SPEC      = {'inputs': ['mol_table', 'mol_table'],
                      'outputs': ['mol_table']}

    def __init__(self):
        super().__init__()
        self.add_combo_menu('logic', 'Logic', items=['AND', 'OR'])

        self.add_input('mol_table_a', color=PORT_COLORS['mol_table'])
        self.add_input('mol_table_b', color=PORT_COLORS['mol_table'])
        self.add_output('mol_table',  color=PORT_COLORS['mol_table'])

    def evaluate(self):
        def _read_input(port_name):
            p = self.inputs().get(port_name)
            if not (p and p.connected_ports()):
                return None
            src = p.connected_ports()[0]
            v = src.node().output_values.get(src.name())
            return v if isinstance(v, MolTableData) else None

        a = _read_input('mol_table_a')
        b = _read_input('mol_table_b')

        if a is None and b is None:
            return False, "Connect at least one mol_table input."

        mol_col = (a or b).mol_col
        logic = self.get_property('logic') or 'AND'

        # If only one side is connected, pass it through
        if a is None or b is None:
            sole = (a or b).payload.copy()
            self.output_values['mol_table'] = MolTableData(
                payload=sole, mol_col=mol_col)
            self.mark_clean()
            self.set_progress(100)
            return True, f"Only one input — {len(sole)} rows passed through."

        self.set_progress(10)
        df_a, df_b = a.payload, b.payload

        if logic == 'AND':
            common = set(df_a['name']) & set(df_b['name'])
            result = df_a[df_a['name'].isin(common)].reset_index(drop=True)
        else:  # OR
            result = pd.concat([df_a, df_b]).drop_duplicates(
                subset='name', keep='first').reset_index(drop=True)

        self.output_values['mol_table'] = MolTableData(
            payload=result, mol_col=mol_col)
        self.mark_clean()
        self.set_progress(100)
        return True, f"{logic}: {len(result)} rows"


# ── Catalog filter names ─────────────────────────────────────────────────────
_CATALOG_FILTERS = [
    'PAINS_A', 'PAINS_B', 'PAINS_C',
    'BRENK',
    'NIH',
    'ZINC',
    'CHEMBL_BMS',
    'CHEMBL_Dundee',
    'CHEMBL_Glaxo',
    'CHEMBL_Inpharmatica',
    'CHEMBL_LINT',
    'CHEMBL_MLSMR',
    'CHEMBL_SureChEMBL',
]

# Map human-readable names to FilterCatalogParams enum values
_CATALOG_ENUM_MAP = {
    'PAINS_A':               FilterCatalogParams.FilterCatalogs.PAINS_A,
    'PAINS_B':               FilterCatalogParams.FilterCatalogs.PAINS_B,
    'PAINS_C':               FilterCatalogParams.FilterCatalogs.PAINS_C,
    'BRENK':                 FilterCatalogParams.FilterCatalogs.BRENK,
    'NIH':                   FilterCatalogParams.FilterCatalogs.NIH,
    'ZINC':                  FilterCatalogParams.FilterCatalogs.ZINC,
    'CHEMBL_BMS':            FilterCatalogParams.FilterCatalogs.CHEMBL_BMS,
    'CHEMBL_Dundee':         FilterCatalogParams.FilterCatalogs.CHEMBL_Dundee,
    'CHEMBL_Glaxo':          FilterCatalogParams.FilterCatalogs.CHEMBL_Glaxo,
    'CHEMBL_Inpharmatica':   FilterCatalogParams.FilterCatalogs.CHEMBL_Inpharmatica,
    'CHEMBL_LINT':           FilterCatalogParams.FilterCatalogs.CHEMBL_LINT,
    'CHEMBL_MLSMR':          FilterCatalogParams.FilterCatalogs.CHEMBL_MLSMR,
    'CHEMBL_SureChEMBL':     FilterCatalogParams.FilterCatalogs.CHEMBL_SureChEMBL,
}


class BatchCatalogFilterNode(BaseExecutionNode):
    """Filter a MolTable using RDKit's built-in structural-alert catalogs.

    Enable one or more catalogs (PAINS, BRENK, NIH, ZINC, CHEMBL variants).
    A molecule is flagged if it matches *any* enabled catalog.

    Include mode keeps clean molecules (no alerts); Exclude mode keeps
    only flagged molecules.

    Outputs two MolTables: *matches* (kept) and *rejects* (removed).
    """

    __identifier__ = 'nodes.Cheminformatics.Batch'
    NODE_NAME      = 'Batch Catalog Filter'
    PORT_SPEC      = {'inputs': ['mol_table'],
                      'outputs': ['mol_table', 'mol_table']}

    def __init__(self):
        super().__init__()
        self.add_combo_menu('mode', 'Mode', items=['Include Clean', 'Exclude Clean'])

        for cat_name in _CATALOG_FILTERS:
            default_on = cat_name in ('PAINS_A', 'PAINS_B', 'PAINS_C', 'BRENK')
            self.add_checkbox(f'cat_{cat_name}', '', cat_name, state=default_on)

        self.add_input('mol_table',  color=PORT_COLORS['mol_table'])
        self.add_output('matches',   color=PORT_COLORS['mol_table'])
        self.add_output('rejects',   color=PORT_COLORS['mol_table'])

    def evaluate(self):
        in_port = self.inputs().get('mol_table')
        if not (in_port and in_port.connected_ports()):
            return False, "No mol_table connected."
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, MolTableData):
            return False, "Expected MolTableData."

        df = val.payload
        mol_col = val.mol_col
        if len(df) == 0:
            self.output_values['matches'] = MolTableData(payload=df.copy(), mol_col=mol_col)
            self.output_values['rejects'] = MolTableData(payload=df.copy(), mol_col=mol_col)
            self.mark_clean()
            self.set_progress(100)
            return True, "Empty table."

        # Collect enabled catalogs
        enabled = []
        for cat_name in _CATALOG_FILTERS:
            if self.get_property(f'cat_{cat_name}'):
                enabled.append(cat_name)

        if not enabled:
            self.output_values['matches'] = MolTableData(payload=df.copy(), mol_col=mol_col)
            self.output_values['rejects'] = MolTableData(
                payload=df.iloc[:0].copy(), mol_col=mol_col)
            self.mark_clean()
            self.set_progress(100)
            return True, "No catalogs enabled — all rows pass."

        self.set_progress(10)

        # Build combined FilterCatalog from all enabled catalogs
        params = FilterCatalogParams()
        for cat_name in enabled:
            params.AddCatalog(_CATALOG_ENUM_MAP[cat_name])
        catalog = FilterCatalog(params)

        self.set_progress(30)

        # Screen each molecule
        n = len(df)
        flagged = np.zeros(n, dtype=bool)
        for i, mol in enumerate(df[mol_col]):
            if mol is not None:
                if catalog.HasMatch(mol):
                    flagged[i] = True

        self.set_progress(80)

        mode = self.get_property('mode') or 'Include Clean'
        if mode == 'Include Clean':
            # matches = molecules that are clean (no alerts)
            match_mask = ~flagged
        else:
            # Exclude Clean = keep only flagged molecules
            match_mask = flagged

        match_df = df[match_mask].reset_index(drop=True)
        reject_df = df[~match_mask].reset_index(drop=True)

        self.output_values['matches'] = MolTableData(payload=match_df, mol_col=mol_col)
        self.output_values['rejects'] = MolTableData(payload=reject_df, mol_col=mol_col)
        self.mark_clean()
        self.set_progress(100)
        n_flagged = int(flagged.sum())
        return True, f"{len(match_df)} match, {len(reject_df)} reject ({n_flagged} flagged)"


# ── Fingerprint helpers ──────────────────────────────────────────────────────

_FP_GENERATORS = {
    'Morgan':              lambda **kw: rdFingerprintGenerator.GetMorganGenerator(
                               radius=kw.get('radius', 2), fpSize=kw.get('n_bits', 2048)),
    'RDKit':               lambda **kw: rdFingerprintGenerator.GetRDKitFPGenerator(
                               fpSize=kw.get('n_bits', 2048)),
    'Topological Torsion': lambda **kw: rdFingerprintGenerator.GetTopologicalTorsionGenerator(
                               fpSize=kw.get('n_bits', 2048)),
    'Atom Pair':           lambda **kw: rdFingerprintGenerator.GetAtomPairGenerator(
                               fpSize=kw.get('n_bits', 2048)),
}

_FP_NAMES = [
    'Morgan', 'RDKit', 'Topological Torsion', 'Atom Pair',
    'Layered', 'Pattern', 'MACCS',
]
if _HAS_AVALON:
    _FP_NAMES.append('Avalon')

_SIM_METRICS = [
    'Tanimoto', 'Dice', 'Braun-Blanquet', 'Cosine', 'Kulczynski',
    'McConnaughey', 'Rogot-Goldberg', 'Russel', 'Sokal', 'Tversky',
]

# Crossover point: below this numpy float32 matmul is faster;
# above this Rust packed-u64 + rayon wins.
_RUST_THRESHOLD = 1500


def _numpy_pairwise_tanimoto(fp_bool: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto via float32 matrix multiply (fast for small N)."""
    fp = fp_bool.astype(np.float32)
    c = fp @ fp.T
    counts = fp.sum(axis=1)
    unions = counts[:, None] + counts[None, :] - c
    return np.divide(c, unions, out=np.zeros_like(c), where=unions > 0).astype(np.float64)


def _pairwise_similarity(
    fp_matrix: np.ndarray,
    metric: str = 'tanimoto',
) -> np.ndarray:
    """Dispatch to numpy (small N, Tanimoto) or Rust (large N / any metric)."""
    n = fp_matrix.shape[0]
    metric_low = metric.lower()
    if n < _RUST_THRESHOLD and metric_low == 'tanimoto':
        return _numpy_pairwise_tanimoto(fp_matrix)
    return sdfrust.pairwise_similarity(fp_matrix, metric_low)


def _compute_fp_matrix(
    mols: list[RDMol],
    method: str = 'Morgan',
    n_bits: int = 2048,
    radius: int = 2,
) -> np.ndarray:
    """Compute (N, n_bits) boolean fingerprint matrix from RDKit Mol objects.

    Returns a numpy bool array suitable for sdfrust.pairwise_similarity().
    """
    gen_factory = _FP_GENERATORS.get(method)

    def _fp_to_numpy(fp) -> np.ndarray:
        arr = np.zeros(fp.GetNumBits(), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(bool)

    fps = []
    for mol in mols:
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=bool))
            continue

        if gen_factory is not None:
            # Generator-based FP types
            gen = gen_factory(radius=radius, n_bits=n_bits)
            fp = gen.GetFingerprint(mol)
        elif method == 'Layered':
            fp = LayeredFingerprint(mol, fpSize=n_bits)
        elif method == 'Pattern':
            fp = PatternFingerprint(mol, fpSize=n_bits)
        elif method == 'MACCS':
            fp = GetMACCSKeysFingerprint(mol)
        elif method == 'Avalon' and _HAS_AVALON:
            fp = _GetAvalonFP(mol, nBits=n_bits)
        else:
            # Fallback: Morgan
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
            fp = gen.GetFingerprint(mol)

        fps.append(_fp_to_numpy(fp))

    return np.stack(fps)


# ── Pairwise similarity node ─────────────────────────────────────────────────

class PairwiseSimilarityNode(BaseExecutionNode):
    """Compute an NxN pairwise similarity matrix for all molecules in a MolTable.

    Fingerprints are computed with RDKit; the NxN pairwise calculation runs in
    Rust (sdfrust) with rayon parallelism and hardware popcount.

    Output is a Table whose first column is the molecule name and remaining
    columns are named after each molecule (suitable for Heatmap).
    """

    __identifier__ = 'nodes.Cheminformatics.Batch'
    NODE_NAME      = 'Pairwise Similarity'
    PORT_SPEC      = {'inputs': ['mol_table'], 'outputs': ['table']}

    def __init__(self):
        super().__init__()
        self.add_combo_menu('fp_method', 'Fingerprint', items=_FP_NAMES)
        self.add_combo_menu('metric', 'Metric', items=_SIM_METRICS)
        self._add_int_spinbox('n_bits', 'Bits', value=2048, min_val=64, max_val=16384)
        self._add_int_spinbox('radius', 'Radius', value=2, min_val=1, max_val=6)

        self.add_input('mol_table', color=PORT_COLORS['mol_table'])
        self.add_output('table',    color=PORT_COLORS['table'])

    def evaluate(self):
        in_port = self.inputs().get('mol_table')
        if not (in_port and in_port.connected_ports()):
            return False, "No mol_table connected."
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, MolTableData):
            return False, "Expected MolTableData."

        df = val.payload
        mol_col = val.mol_col
        n = len(df)
        if n == 0:
            return False, "Empty table."
        if n > 10000:
            return False, f"Table has {n:,} rows — limit is 10 000 for pairwise similarity."

        method = self.get_property('fp_method') or 'Morgan'
        metric = self.get_property('metric') or 'Tanimoto'
        n_bits = int(self.get_property('n_bits') or 2048)
        radius = int(self.get_property('radius') or 2)

        self.set_progress(10)
        mols = df[mol_col].tolist()
        names = df['name'].tolist() if 'name' in df.columns else [str(i) for i in range(n)]

        # Compute fingerprints (RDKit)
        fp_matrix = _compute_fp_matrix(mols, method=method, n_bits=n_bits, radius=radius)
        self.set_progress(40)

        # Pairwise similarity (Rust + rayon)
        sim_matrix = _pairwise_similarity(fp_matrix, metric)
        self.set_progress(90)

        # Build output table
        out_df = pd.DataFrame(sim_matrix, columns=names)
        out_df.insert(0, 'name', names)

        self.output_values['table'] = TableData(payload=out_df)
        self.mark_clean()
        self.set_progress(100)
        return True, f"{n}×{n} {metric} ({method})"


# ── Similarity search node ───────────────────────────────────────────────────

class SimilaritySearchNode(BaseExecutionNode):
    """Rank all molecules in a MolTable by similarity to a query molecule.

    Adds a ``similarity`` column and sorts descending.  Optionally filters
    by a minimum similarity threshold.
    """

    __identifier__ = 'nodes.Cheminformatics.Batch'
    NODE_NAME      = 'Similarity Search'
    PORT_SPEC      = {'inputs': ['molecule', 'mol_table'],
                      'outputs': ['mol_table']}

    def __init__(self):
        super().__init__()
        self.add_combo_menu('fp_method', 'Fingerprint', items=_FP_NAMES)
        self.add_combo_menu('metric', 'Metric', items=_SIM_METRICS)
        self._add_int_spinbox('n_bits', 'Bits', value=2048, min_val=64, max_val=16384)
        self._add_int_spinbox('radius', 'Radius', value=2, min_val=1, max_val=6)
        self._add_float_spinbox('min_sim', 'Min Similarity',
                                value=0.0, min_val=0.0, max_val=1.0,
                                step=0.05, decimals=3)

        self.add_input('molecule',  color=PORT_COLORS['molecule'])
        self.add_input('mol_table', color=PORT_COLORS['mol_table'])
        self.add_output('mol_table', color=PORT_COLORS['mol_table'])

    def evaluate(self):
        # Read query molecule
        q_port = self.inputs().get('molecule')
        if not (q_port and q_port.connected_ports()):
            return False, "No query molecule connected."
        q_src = q_port.connected_ports()[0]
        q_val = q_src.node().output_values.get(q_src.name())
        if not isinstance(q_val, MoleculeData) or q_val.payload is None:
            return False, "Expected a valid MoleculeData on 'molecule'."

        # Read mol table
        t_port = self.inputs().get('mol_table')
        if not (t_port and t_port.connected_ports()):
            return False, "No mol_table connected."
        t_src = t_port.connected_ports()[0]
        t_val = t_src.node().output_values.get(t_src.name())
        if not isinstance(t_val, MolTableData):
            return False, "Expected MolTableData."

        df = t_val.payload.copy()
        mol_col = t_val.mol_col
        n = len(df)
        if n == 0:
            return False, "Empty table."

        method = self.get_property('fp_method') or 'Morgan'
        metric = self.get_property('metric') or 'Tanimoto'
        n_bits = int(self.get_property('n_bits') or 2048)
        radius = int(self.get_property('radius') or 2)
        min_sim = float(self.get_property('min_sim') or 0.0)

        self.set_progress(10)

        # Compute fingerprints: query + all library mols
        all_mols = [q_val.payload] + df[mol_col].tolist()
        fp_matrix = _compute_fp_matrix(all_mols, method=method, n_bits=n_bits, radius=radius)
        self.set_progress(50)

        # Pairwise of query (row 0) vs all — only need first row
        sim_full = _pairwise_similarity(fp_matrix, metric)
        sims = sim_full[0, 1:]  # exclude self-similarity
        self.set_progress(80)

        df['similarity'] = np.round(sims, 4)

        # Filter & sort
        if min_sim > 0:
            df = df[df['similarity'] >= min_sim].reset_index(drop=True)
        df = df.sort_values('similarity', ascending=False).reset_index(drop=True)

        self.output_values['mol_table'] = MolTableData(payload=df, mol_col=mol_col)
        self.mark_clean()
        self.set_progress(100)
        return True, f"{len(df)} hits (≥{min_sim:.2f}) sorted by {metric}"


# ── Butina clustering node ───────────────────────────────────────────────────

class ButinaClusterNode(BaseExecutionNode):
    """Cluster molecules using Taylor–Butina algorithm.

    Computes fingerprints, pairwise similarity (Rust), then clusters.
    Adds ``cluster_id`` and ``is_centroid`` columns to the output.
    """

    __identifier__ = 'nodes.Cheminformatics.Batch'
    NODE_NAME      = 'Butina Cluster'
    PORT_SPEC      = {'inputs': ['mol_table'], 'outputs': ['mol_table']}

    _CLUSTER_METHODS = ['Auto', 'Matrix (fast)', 'Low Memory']
    _AUTO_FPS_THRESHOLD = 10000   # Auto switches to FP-based above this N
    _MATRIX_LIMIT       = 20000  # hard cap for matrix path (NxN = 3.2 GB)
    _LOW_MEM_LIMIT      = 100000 # hard cap for low-memory path

    def __init__(self):
        super().__init__()
        self.add_combo_menu('fp_method', 'Fingerprint', items=_FP_NAMES)
        self.add_combo_menu('metric', 'Metric', items=_SIM_METRICS)
        self.add_combo_menu('cluster_method', 'Cluster Method',
                            items=self._CLUSTER_METHODS)
        self._add_int_spinbox('n_bits', 'Bits', value=2048, min_val=64, max_val=16384)
        self._add_int_spinbox('radius', 'Radius', value=2, min_val=1, max_val=6)
        self._add_float_spinbox('threshold', 'Similarity Threshold',
                                value=0.35, min_val=0.0, max_val=1.0,
                                step=0.05, decimals=3)

        self.add_input('mol_table',  color=PORT_COLORS['mol_table'])
        self.add_output('mol_table', color=PORT_COLORS['mol_table'])

    def evaluate(self):
        in_port = self.inputs().get('mol_table')
        if not (in_port and in_port.connected_ports()):
            return False, "No mol_table connected."
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, MolTableData):
            return False, "Expected MolTableData."

        df = val.payload.copy()
        mol_col = val.mol_col
        n = len(df)
        if n == 0:
            return False, "Empty table."

        method    = self.get_property('fp_method') or 'Morgan'
        metric    = self.get_property('metric') or 'Tanimoto'
        n_bits    = int(self.get_property('n_bits') or 2048)
        radius    = int(self.get_property('radius') or 2)
        threshold = float(self.get_property('threshold') or 0.35)
        cluster_method = self.get_property('cluster_method') or 'Auto'
        metric_low = metric.lower()

        # Decide path: matrix vs fingerprint-based
        use_fps = (cluster_method == 'Low Memory'
                   or (cluster_method == 'Auto'
                       and n > self._AUTO_FPS_THRESHOLD))

        # Enforce size limits
        if use_fps:
            if n > self._LOW_MEM_LIMIT:
                return False, (f"Table has {n:,} rows — limit is "
                               f"{self._LOW_MEM_LIMIT:,} for Low Memory mode.")
        else:
            if n > self._MATRIX_LIMIT:
                return False, (f"Table has {n:,} rows — limit is "
                               f"{self._MATRIX_LIMIT:,} for Matrix mode. "
                               f"Try 'Low Memory' or 'Auto'.")

        self.set_progress(10)
        mols = df[mol_col].tolist()

        # Fingerprints (RDKit)
        fp_matrix = _compute_fp_matrix(mols, method=method, n_bits=n_bits, radius=radius)
        self.set_progress(30)

        if use_fps:
            # Cluster directly from fingerprints — no NxN matrix
            labels = sdfrust.butina_cluster_fps(fp_matrix, threshold, metric=metric_low)
            path_label = "low-mem"
        else:
            # Build full similarity matrix, then cluster
            sim_matrix = _pairwise_similarity(fp_matrix, metric)
            self.set_progress(70)
            labels = sdfrust.butina_cluster(sim_matrix, threshold)
            path_label = "matrix"
        self.set_progress(90)

        df['cluster_id'] = labels

        # Mark centroids (first molecule assigned to each cluster)
        centroids = set()
        is_centroid = []
        for lab in labels:
            if lab not in centroids:
                centroids.add(lab)
                is_centroid.append(True)
            else:
                is_centroid.append(False)
        df['is_centroid'] = is_centroid

        self.output_values['mol_table'] = MolTableData(payload=df, mol_col=mol_col)
        self.mark_clean()
        self.set_progress(100)
        n_clusters = len(set(labels))
        return True, (f"{n_clusters} clusters (threshold={threshold:.2f}, "
                      f"{metric}, {path_label})")
