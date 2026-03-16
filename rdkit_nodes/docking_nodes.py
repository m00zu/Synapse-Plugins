"""
docking_nodes.py — Protein preparation and molecular docking nodes.

Nodes:
  - PDBLoaderNode        Load PDB/CIF files
  - PDBDownloaderNode    Fetch from RCSB PDB / AlphaFold DB
  - ProteinPrepNode      Fix structure + add hydrogens + generate PDBQT
  - DockingBoxNode       Define search box (manual or auto from ligand)
  - VinaDockNode         Single-ligand docking (outputs MoleculeData)
  - BatchDockNode        Dock every molecule in a MolTable
  - GNINARescoreNode     CNN rescoring (accepts MoleculeData or MolTableData)
  - DrugCLIPScreenNode   Contrastive virtual screening (pocket vs molecules)
  - StructureWriterNode  Export protein/receptor to PDB/PDBQT file
"""
from __future__ import annotations

import ast
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import NodeGraphQt
from PySide6 import QtCore, QtWidgets, QtGui
from rdkit import Chem

from nodes.base import (BaseExecutionNode, PORT_COLORS,
                         NodeFileSelector, NodeFileSaver, NodeVec3Widget)
from data_models import NodeData, TableData
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget

from .protein_data import ProteinData, ReceptorData, DockingResultData
from .protein_utils import (
    clean_pdb, read_pdb_string, fix_and_convert, fix_and_convert_from_string,
    fix_pdb_missing_atoms, fix_pdb_missing_atoms_from_string,
    check_amino_acids, get_pdb_string, write_to_pdbqt,
    PDBQTReceptor, PDBQTMolecule, RDKitMolCreate,
    process_rigid_flex, _check_for_aa_count,
)
from .docking_backend import BACKENDS, get_backend

# Lazy import — only needed for ligand PDBQT conversion (fallback)
from .meeko_ported import MoleculePreparation, PDBQTWriterLegacy

try:
    import sdfrust as _sdfrust
except ImportError:
    _sdfrust = None

# ── Port colors ──────────────────────────────────────────────────────────────
PORT_COLORS.setdefault('protein',        (34, 139, 34))    # Forest green
PORT_COLORS.setdefault('receptor',       (0, 128, 128))    # Teal
PORT_COLORS.setdefault('docking_result', (255, 140, 0))    # Dark orange
PORT_COLORS.setdefault('box_config',     (210, 180, 140))  # Tan

# Re-use existing chem_nodes types when they're imported
try:
    from .chem_nodes import MoleculeData, MolTableData, mol_to_format, embed_mol_3d
except ImportError:
    MoleculeData = None
    MolTableData = None
    embed_mol_3d = None

# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_smiles_and_order(mol):
    """Get canonical SMILES (heavy-atom only) and atom-order mapping.

    If the molecule has explicit Hs, they are removed temporarily.
    Returns (smiles_str, list_of_original_atom_indices_in_smiles_order).
    """
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        mol_noH = Chem.RemoveAllHs(mol)
        smi = Chem.MolToSmiles(mol_noH, canonical=True)
        order_noH = ast.literal_eval(mol_noH.GetProp('_smilesAtomOutputOrder'))
        heavy_idx = [i for i in range(mol.GetNumAtoms())
                     if mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
        return smi, [heavy_idx[j] for j in order_noH]
    smi = Chem.MolToSmiles(mol, canonical=True)
    order = ast.literal_eval(mol.GetProp('_smilesAtomOutputOrder'))
    return smi, list(order)


def _mol_to_pdbqt(mol, confId=-1) -> str:
    """Convert RDKit Mol → PDBQT string.

    Uses the Rust sdfrust backend when available (20x+ faster, writes
    REMARK SMILES / SMILES IDX / H PARENT for reconstruction).
    Falls back to Meeko if sdfrust is not installed.

    *confId* selects which conformer to convert (-1 = default/first).
    """
    if _sdfrust is not None:
        smi, order = _get_smiles_and_order(mol)
        sdf_str = Chem.MolToMolBlock(mol, confId=confId)
        rust_mol = _sdfrust.parse_sdf_string(sdf_str)
        arom = [a.GetIsAromatic() for a in mol.GetAtoms()]
        sym = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        return _sdfrust.mol_to_pdbqt(
            rust_mol, smiles=smi, smiles_atom_order=order,
            aromatic_atoms=arom, symmetry_classes=sym)
    # Fallback: Meeko
    prep = MoleculePreparation(rigid_macrocycles=False)
    mol_setups = prep.prepare(mol)
    for setup in mol_setups:
        pdbqt_str, is_ok, err_msg = PDBQTWriterLegacy.write_string(setup)
        if is_ok:
            return pdbqt_str
    raise RuntimeError(f'Meeko PDBQT conversion failed: {err_msg}')


def _split_pdbqt_models(pdbqt_str):
    """Split multi-model PDBQT into individual MODEL block strings."""
    models = []
    current = []
    for line in pdbqt_str.splitlines():
        if line.startswith('MODEL'):
            current = [line]
        elif line.startswith('ENDMDL'):
            current.append(line)
            models.append('\n'.join(current))
            current = []
        else:
            current.append(line)
    return models


def _merge_dock_results(results, n_poses):
    """Merge (poses_pdbqt, energies) from multiple conformers; keep top *n_poses*."""
    all_poses = []
    for poses_pdbqt, energies in results:
        if not poses_pdbqt or not energies:
            continue
        models = _split_pdbqt_models(poses_pdbqt)
        for block, e_row in zip(models, energies):
            all_poses.append((e_row, block))
    if not all_poses:
        return '', []
    all_poses.sort(key=lambda x: x[0][0])
    top = all_poses[:n_poses]
    parts = []
    merged_e = []
    for i, (e_row, block) in enumerate(top, 1):
        lines = block.split('\n')
        if lines and lines[0].startswith('MODEL'):
            lines[0] = f'MODEL {i}'
        parts.append('\n'.join(lines))
        merged_e.append(e_row)
    return '\n'.join(parts), merged_e


def _parse_flex_residues(text: str):
    """Parse flexible residues string like 'A:LYS:417, B:ASP:100'."""
    if not text.strip():
        return set()
    residues = set()
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(':')
        if len(tokens) == 3:
            chain, resname, resnum = tokens
            residues.add((chain.strip(), resname.strip().upper(), int(resnum.strip())))
        elif len(tokens) == 2:
            resname, resnum = tokens
            residues.add(('', resname.strip().upper(), int(resnum.strip())))
    return residues


# ══════════════════════════════════════════════════════════════════════════════
#  Node 1: PDB Loader
# ══════════════════════════════════════════════════════════════════════════════

class PDBLoaderNode(BaseExecutionNode):
    """Load a PDB or CIF file and output cleaned protein data.

    Removes non-protein atoms (water, ligands), handles multi-model files.
    Optionally returns HETATM ligand bounding-box info for auto-boxing.
    """

    __identifier__ = 'nodes.Cheminformatics.Protein'
    NODE_NAME      = 'PDB Loader'
    PORT_SPEC      = {'inputs': [], 'outputs': ['protein']}

    def __init__(self):
        super().__init__()
        file_selector = NodeFileSelector(
            self.view, name='file_path', label='PDB/CIF File',
            ext_filter='Structure Files (*.pdb *.cif *.ent *.pdb.gz);;All Files (*)')
        self.add_custom_widget(
            file_selector,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties')
        self.add_combo_menu('clean_mode', 'Clean', items=['Auto', 'Keep HETATMs'])
        self.add_output('protein', color=PORT_COLORS['protein'])

    def evaluate(self):
        fpath = (self.get_property('file_path') or '').strip()
        if not fpath:
            return False, 'Select a PDB or CIF file.'
        p = Path(fpath).expanduser()
        if not p.is_file():
            return False, f'File not found: {p}'

        self.set_progress(10)

        # Read raw content
        if p.suffix == '.gz':
            import gzip
            with gzip.open(p, 'rt') as f:
                raw = f.read()
        else:
            with open(p) as f:
                raw = f.read()

        # Detect format
        fmt = 'cif' if raw.lstrip().startswith('data_') else 'pdb'
        name = p.stem.replace('.pdb', '').replace('.cif', '')

        self.set_progress(30)

        clean_mode = self.get_property('clean_mode') or 'Auto'
        return_hetatm = (clean_mode == 'Keep HETATMs')

        result = clean_pdb(raw, return_hetatm=return_hetatm, format=fmt)
        if return_hetatm:
            cleaned, hetatm_dict = result
            metadata = {'ligands': hetatm_dict, 'format': fmt}
        else:
            cleaned = result
            metadata = {'format': fmt}

        if not cleaned.strip():
            return False, 'No protein atoms found in file.'

        # Count chains and residues for status message
        n_lines = sum(1 for l in cleaned.splitlines()
                      if l.startswith('ATOM') or l.startswith('HETATM'))

        self.set_progress(90)
        self.output_values['protein'] = ProteinData(
            payload=cleaned, name=name, format=fmt,
            metadata=metadata, source_path=str(p))

        self.mark_clean()
        self.set_progress(100)
        return True, f'{name}: {n_lines} atoms loaded ({fmt.upper()})'


# ══════════════════════════════════════════════════════════════════════════════
#  Node 1b: PDB Downloader (fetch from RCSB PDB / AlphaFold)
# ══════════════════════════════════════════════════════════════════════════════

_DB_URLS = {
    'RCSB PDB':       'https://files.rcsb.org/download/{id}.pdb',
    'AlphaFold DB':   'https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_v4.pdb',
}

_TITLE_RE = {
    'pdb':  r'TITLE\s+(.*)',
    'cif':  r'_citation\.title\s+(.*)',
}


class PDBDownloaderNode(BaseExecutionNode):
    """Download a protein structure from RCSB PDB or AlphaFold Database.

    Enter a PDB ID (e.g. ``1AKE``) or UniProt ID (for AlphaFold) and the
    structure is fetched, cleaned, and output as ProteinData.  Automatically
    falls back to CIF format when the PDB file is not available.

    HETATM ligands are extracted with bounding-box info (useful for
    auto-centering a docking box on a co-crystallised ligand).

    Keywords: download, fetch, RCSB, PDB ID, AlphaFold, UniProt, web,
              下載, 蛋白質, 結構
    """

    __identifier__ = 'nodes.Cheminformatics.Protein'
    NODE_NAME      = 'PDB Downloader'
    PORT_SPEC      = {'inputs': [], 'outputs': ['protein', 'table']}

    def __init__(self):
        super().__init__()
        self.add_text_input('pdb_id', 'PDB / UniProt ID', text='')
        self.add_combo_menu('database', 'Database', items=list(_DB_URLS.keys()))
        self.add_output('protein', color=PORT_COLORS['protein'])
        self.add_output('ligands', color=PORT_COLORS.get('table', (180, 180, 180)))

    def evaluate(self):
        import re
        import requests

        pdb_id = (self.get_property('pdb_id') or '').strip().upper()
        if not pdb_id:
            return False, 'Enter a PDB or UniProt ID.'

        db = self.get_property('database') or 'RCSB PDB'
        url = _DB_URLS[db].format(id=pdb_id)

        self.set_progress(10)

        # ── Download ─────────────────────────────────────────────────
        cif = False
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404 and not url.endswith('.cif'):
                # Fallback: try CIF
                cif_url = url.rsplit('.', 1)[0] + '.cif'
                r = requests.get(cif_url, timeout=30)
                cif = True
            if r.status_code != 200:
                return False, (
                    f'Failed to download {pdb_id} from {db}. '
                    f'HTTP {r.status_code}.'
                )
        except Exception as e:
            return False, f'Connection error: {e}'

        raw_text = r.text
        fmt = 'cif' if cif or raw_text.lstrip().startswith('data_') else 'pdb'

        self.set_progress(40)

        # ── Extract title ────────────────────────────────────────────
        title_re = _TITLE_RE.get(fmt, _TITLE_RE['pdb'])
        title_parts = re.findall(title_re, raw_text)
        title = ' '.join(p.strip() for p in title_parts).strip() or pdb_id

        # ── Clean ────────────────────────────────────────────────────
        result = clean_pdb(raw_text, return_hetatm=True, format=fmt)
        cleaned, hetatm_dict = result

        if not cleaned.strip():
            return False, f'{pdb_id} contains no protein atoms.'

        self.set_progress(70)

        # ── Build ligand table from HETATM bounding boxes ────────────
        if hetatm_dict:
            rows = []
            for lig_name, info in hetatm_dict.items():
                rows.append({
                    'Ligand': lig_name,
                    'Center X': info['Center'][0],
                    'Center Y': info['Center'][1],
                    'Center Z': info['Center'][2],
                    'Size X': info['Box'][0],
                    'Size Y': info['Box'][1],
                    'Size Z': info['Box'][2],
                    'Volume': info['Volume'],
                })
            lig_df = pd.DataFrame(rows)
            self.output_values['ligands'] = TableData(payload=lig_df)
        else:
            self.output_values['ligands'] = TableData(
                payload=pd.DataFrame(columns=[
                    'Ligand', 'Center X', 'Center Y', 'Center Z',
                    'Size X', 'Size Y', 'Size Z', 'Volume']))

        n_atoms = sum(1 for l in cleaned.splitlines()
                      if l.startswith('ATOM') or l.startswith('HETATM'))

        self.set_progress(90)
        self.output_values['protein'] = ProteinData(
            payload=cleaned, name=pdb_id, format=fmt,
            metadata={'title': title, 'ligands': hetatm_dict,
                      'database': db, 'format': fmt})

        self.mark_clean()
        self.set_progress(100)
        n_lig = len(hetatm_dict)
        return True, f'{pdb_id}: {n_atoms} atoms, {n_lig} ligand(s) — {title[:60]}'


# ══════════════════════════════════════════════════════════════════════════════
#  Node 1c: Protein Editor (chain / residue filter)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_residue_ranges(text: str):
    """Parse residue range string like '1-100, 200-300' into a set of ints."""
    if not text.strip():
        return None  # None means "keep all"
    nums = set()
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-', 1)
            try:
                nums.update(range(int(lo.strip()), int(hi.strip()) + 1))
            except ValueError:
                continue
        else:
            try:
                nums.add(int(part))
            except ValueError:
                continue
    return nums


class ProteinEditorNode(BaseExecutionNode):
    """Filter a protein structure by chain and residue range.

    Useful for trimming multi-chain complexes, keeping only the chain(s)
    of interest, or restricting to a residue range for focused docking.

    Leave a field empty to keep everything (no filter on that axis).

    Keywords: protein editor, chain filter, residue range, trim, select,
              蛋白質編輯, 鏈, 殘基, 篩選
    """

    __identifier__ = 'nodes.Cheminformatics.Protein'
    NODE_NAME      = 'Protein Editor'
    PORT_SPEC      = {'inputs': ['protein'], 'outputs': ['protein']}

    def __init__(self):
        super().__init__()
        self.add_text_input('chains', 'Keep Chains',
                            text='', placeholder_text='A, B  (empty = all)')
        self.add_text_input('residue_range', 'Residue Range',
                            text='', placeholder_text='1-100, 200-300  (empty = all)')
        self.add_text_input('remove_residues', 'Remove Residues',
                            text='', placeholder_text='A:HOH:, B:LYS:417')
        self.add_checkbox('remove_water', 'Remove Water', state=True)
        self.add_checkbox('remove_hetatm', 'Remove HETATM', state=False)

        self.add_input('protein', color=PORT_COLORS['protein'])
        self.add_output('protein', color=PORT_COLORS['protein'])

    def evaluate(self):
        in_port = self.inputs().get('protein')
        if not (in_port and in_port.connected_ports()):
            return False, 'No protein connected.'
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, ProteinData):
            return False, 'Expected ProteinData input.'

        pdb_str = val.payload
        if not pdb_str:
            return False, 'Empty protein data.'

        self.set_progress(10)

        # ── Parse filter parameters ──────────────────────────────────
        chains_text = (self.get_property('chains') or '').strip()
        keep_chains = None
        if chains_text:
            keep_chains = {c.strip().upper() for c in chains_text.split(',')}

        residue_range = _parse_residue_ranges(
            self.get_property('residue_range') or '')

        remove_water = bool(self.get_property('remove_water'))
        remove_hetatm = bool(self.get_property('remove_hetatm'))

        # Parse residues to remove: "A:HOH:, B:LYS:417"
        remove_set = set()
        remove_text = (self.get_property('remove_residues') or '').strip()
        if remove_text:
            for part in remove_text.split(','):
                part = part.strip()
                if not part:
                    continue
                tokens = part.split(':')
                if len(tokens) >= 2:
                    remove_set.add(tuple(t.strip().upper() for t in tokens))

        self.set_progress(30)

        # ── Filter PDB lines ────────────────────────────────────────
        kept = []
        n_removed = 0
        for line in pdb_str.splitlines():
            is_atom = line.startswith('ATOM')
            is_hetatm = line.startswith('HETATM')

            if not (is_atom or is_hetatm):
                # Keep non-coordinate lines (HEADER, REMARK, TER, END, etc.)
                if line.startswith('TER'):
                    # Only keep TER if its chain is in kept chains
                    if keep_chains is not None and len(line) > 21:
                        chain = line[21].strip().upper()
                        if chain and chain not in keep_chains:
                            continue
                kept.append(line)
                continue

            # Extract PDB fields
            chain = line[21].strip().upper() if len(line) > 21 else ''
            try:
                resnum = int(line[22:26].strip()) if len(line) > 26 else 0
            except ValueError:
                resnum = 0
            resname = line[17:20].strip().upper() if len(line) > 20 else ''

            # Apply filters
            if remove_water and resname in ('HOH', 'WAT', 'H2O', 'DOD'):
                n_removed += 1
                continue

            if remove_hetatm and is_hetatm:
                n_removed += 1
                continue

            if keep_chains is not None and chain not in keep_chains:
                n_removed += 1
                continue

            if residue_range is not None and resnum not in residue_range:
                n_removed += 1
                continue

            # Check explicit removal list
            removed = False
            for rm in remove_set:
                if len(rm) == 2:
                    # chain:resname — remove all of that residue type on that chain
                    if rm[0] == chain and (not rm[1] or rm[1] == resname):
                        removed = True
                        break
                elif len(rm) == 3:
                    # chain:resname:resnum
                    rm_chain, rm_resname, rm_resnum = rm
                    if rm_chain == chain:
                        if rm_resname and rm_resname != resname:
                            continue
                        if rm_resnum:
                            try:
                                if int(rm_resnum) != resnum:
                                    continue
                            except ValueError:
                                continue
                        removed = True
                        break
            if removed:
                n_removed += 1
                continue

            kept.append(line)

        self.set_progress(80)

        filtered = '\n'.join(kept)
        n_atoms = sum(1 for l in kept
                      if l.startswith('ATOM') or l.startswith('HETATM'))

        if n_atoms == 0:
            return False, 'All atoms were removed by the filters.'

        self.output_values['protein'] = ProteinData(
            payload=filtered, name=val.name, format=val.format,
            metadata=val.metadata, source_path=val.source_path)

        self.mark_clean()
        self.set_progress(100)
        return True, (f'{val.name}: {n_atoms} atoms kept, '
                      f'{n_removed} removed')


# ══════════════════════════════════════════════════════════════════════════════
#  Node 2: Protein Preparation
# ══════════════════════════════════════════════════════════════════════════════

class ProteinPrepNode(BaseExecutionNode):
    """Prepare a protein for docking: fix structure, add H, generate PDBQT.

    Pipeline: PDBFixer (fix + add H) → protonation checks → PDBQT typing.
    Requires OpenMM for hydrogen addition.
    """

    __identifier__ = 'nodes.Cheminformatics.Protein'
    NODE_NAME      = 'Protein Prep'
    PORT_SPEC      = {'inputs': ['protein'], 'outputs': ['receptor']}

    def __init__(self):
        super().__init__()
        self._add_float_spinbox('ph', 'pH', value=7.0,
                                min_val=0.0, max_val=14.0, step=0.5, decimals=1)
        self.add_checkbox('fix_missing', 'Fix Missing Atoms', state=True)
        self.add_checkbox('fill_gap', 'Fill Gaps', state=False)
        self.add_text_input('flex_residues', 'Flex Residues',
                            text='', placeholder_text='A:LYS:417, B:ASP:100')

        self.add_input('protein', color=PORT_COLORS['protein'])
        self.add_output('receptor', color=PORT_COLORS['receptor'])

    def evaluate(self):
        in_port = self.inputs().get('protein')
        if not (in_port and in_port.connected_ports()):
            return False, 'No protein connected.'
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())
        if not isinstance(val, ProteinData):
            return False, 'Expected ProteinData input.'

        ph = float(self.get_property('ph') or 7.0)
        fill_gap = bool(self.get_property('fill_gap'))
        flex_text = (self.get_property('flex_residues') or '').strip()
        flex_res = _parse_flex_residues(flex_text)

        self.set_progress(10)

        pdb_str = val.payload
        name = val.name
        fmt = val.format  # 'pdb' or 'cif'

        # Write to temp file for PDBFixer (it wants a file path)
        import tempfile
        suffix = '.cif' if fmt == 'cif' else '.pdb'
        with tempfile.NamedTemporaryFile(
                mode='w', suffix=suffix, delete=False) as tmp:
            tmp.write(pdb_str)
            tmp_path = tmp.name

        try:
            self.set_progress(20)
            result = fix_and_convert(tmp_path, fill_gap=fill_gap, ph=ph)
        except ImportError as e:
            return False, str(e)
        except Exception as e:
            return False, f'Protein preparation failed:\n{e}'
        finally:
            os.unlink(tmp_path)

        if isinstance(result, tuple):
            return False, f'Preparation failed: {result[0]}'

        pdbqt_str = result
        self.set_progress(80)

        # Handle flexible residues
        flex_pdbqt = ''
        if flex_res:
            rigid_str, flex_pdbqt, ok, err = process_rigid_flex(pdbqt_str, flex_res)
            if not ok:
                return False, f'Flexible residue error: {err}'
            pdbqt_str = rigid_str

        self.set_progress(95)
        self.output_values['receptor'] = ReceptorData(
            payload=pdbqt_str, name=name,
            flex_pdbqt=flex_pdbqt,
            flex_residues=list(str(r) for r in flex_res),
            source_path=val.source_path)

        self.mark_clean()
        self.set_progress(100)
        n_atoms = sum(1 for l in pdbqt_str.splitlines() if l.startswith('ATOM'))
        msg = f'{name}: {n_atoms} atoms in PDBQT'
        if flex_res:
            msg += f' ({len(flex_res)} flexible residue{"s" if len(flex_res) > 1 else ""})'
        return True, msg


# ══════════════════════════════════════════════════════════════════════════════
#  Node 3: Docking Box
# ══════════════════════════════════════════════════════════════════════════════

class DockingBoxNode(BaseExecutionNode):
    """Define the docking search box with an integrated 3D viewer.

    Click on the protein structure to set the docking center.  The docking
    box is drawn in the viewer in real-time as you adjust center/size values.
    Flexible residues can be selected by clicking in "Add Flexible" mode.

    Accepts either raw ProteinData or prepared ReceptorData (PDBQT) for
    display.  The receptor is passed through for downstream docking nodes.

    Modes:
      - Manual        — enter center/size directly in spinboxes
      - Auto from Ligand — compute box from a connected molecule's coordinates

    Keywords: docking box, search box, center, size, click, interactive,
              對接框, 搜索框, 中心, 大小
    """

    __identifier__ = 'nodes.Cheminformatics.Docking'
    NODE_NAME      = 'Docking Box'
    PORT_SPEC      = {'inputs': ['protein', 'receptor', 'molecule'],
                      'outputs': ['receptor', 'box_config']}

    _UI_PROPS = frozenset({
        'click_mode', 'center', 'box_size', 'padding', 'flex_residues',
    })

    def __init__(self):
        super().__init__()
        from .viewer_nodes import Node3DViewerWidget, _pdbqt_to_pdb_string

        self.add_combo_menu('mode', 'Mode', items=['Manual', 'Auto from Ligand'])
        self.add_combo_menu('click_mode', 'Click Mode',
                            items=['Disable', 'Set Center', 'Add Flexible'])

        # Compact Vec3 widgets: one row for Center, one row for Size
        self._center_w = NodeVec3Widget(
            self.view, name='center', label='Center',
            value=(0.0, 0.0, 0.0), min_val=-999.0, max_val=999.0,
            step=1.0, decimals=3)
        self.add_custom_widget(self._center_w)

        self._size_w = NodeVec3Widget(
            self.view, name='box_size', label='Size',
            value=(20.0, 20.0, 20.0), min_val=1.0, max_val=200.0,
            step=1.0, decimals=1)
        self.add_custom_widget(self._size_w)

        self._add_float_spinbox('padding', 'Padding', value=5.0,
                                min_val=0.0, max_val=50.0, step=1.0, decimals=1)

        self.add_text_input('flex_residues', 'Flexible Residues', text='')

        self.add_input('protein', color=PORT_COLORS['protein'])
        self.add_input('receptor', color=PORT_COLORS['receptor'])
        self.add_input('molecule', color=PORT_COLORS.get('molecule', (205, 92, 92)))
        self.add_output('receptor', color=PORT_COLORS['receptor'])
        self.add_output('box_config', color=PORT_COLORS['box_config'])

        # ── Embedded 3D viewer with click bridge ─────────────────────
        self._viewer_widget = Node3DViewerWidget(self.view, enable_bridge=True)
        self.add_custom_widget(self._viewer_widget, tab='View')

        if self._viewer_widget._bridge is not None:
            self._viewer_widget._bridge.atom_clicked.connect(self._on_atom_click)

        self._pdbqt_to_pdb = _pdbqt_to_pdb_string

    # ── Property change hook (live box update) ───────────────────────

    def set_property(self, name, value, push_undo=True):
        super().set_property(name, value, push_undo)
        if name in ('center', 'box_size'):
            self._update_box_visual()
        elif name == 'click_mode':
            mode_map = {
                'Disable': 'disable',
                'Set Center': 'center',
                'Add Flexible': 'flexible',
            }
            js_mode = mode_map.get(str(value), 'disable')
            self._viewer_widget.run_js(f'setClickMode("{js_mode}");')

    # ── Atom click handler ───────────────────────────────────────────

    def _on_atom_click(self, data):
        """Handle atom click from the 3D viewer."""
        click_mode = self.get_property('click_mode') or 'Disable'
        if click_mode == 'Set Center':
            x = round(float(data.get('x', 0)), 3)
            y = round(float(data.get('y', 0)), 3)
            z = round(float(data.get('z', 0)), 3)
            self.set_property('center', [x, y, z])
        elif click_mode == 'Add Flexible':
            chain = data.get('chain', '')
            resn = data.get('resn', '')
            resi = data.get('resi', '')
            entry = f'{chain}:{resn}:{resi}'
            current = (self.get_property('flex_residues') or '').strip()
            if entry not in current:
                new_val = (current + ', ' + entry).strip(', ')
                self.set_property('flex_residues', new_val)
                self._viewer_widget.run_js(
                    f'highlightResidue("{chain}", {resi});')

    # ── Visual update ────────────────────────────────────────────────

    def _update_box_visual(self):
        """Push current center+size to the JS viewer as sphere+box."""
        center = self.get_property('center') or [0, 0, 0]
        size = self.get_property('box_size') or [20, 20, 20]
        cx, cy, cz = [float(v) for v in center]
        sx, sy, sz = [float(v) for v in size]
        js = (f'drawCenter({cx},{cy},{cz});'
              f'drawBox({cx},{cy},{cz},{sx},{sy},{sz});')
        self._viewer_widget.run_js(js)

    # ── Evaluate ─────────────────────────────────────────────────────

    def evaluate(self):
        # ── Collect protein / receptor ───────────────────────────────
        protein_data = None
        protein_port = self.inputs().get('protein')
        if protein_port and protein_port.connected_ports():
            src = protein_port.connected_ports()[0]
            protein_data = src.node().output_values.get(src.name())

        receptor_data = None
        rec_port = self.inputs().get('receptor')
        if rec_port and rec_port.connected_ports():
            src = rec_port.connected_ports()[0]
            receptor_data = src.node().output_values.get(src.name())
            if receptor_data is not None:
                self.output_values['receptor'] = receptor_data

        mode = self.get_property('mode') or 'Manual'
        self.set_progress(10)

        if mode == 'Auto from Ligand':
            mol_port = self.inputs().get('molecule')
            if not (mol_port and mol_port.connected_ports()):
                return False, 'Connect a molecule for auto-boxing.'
            src = mol_port.connected_ports()[0]
            mol_data = src.node().output_values.get(src.name())

            coords = None

            if MoleculeData is not None and isinstance(mol_data, MoleculeData):
                mol = mol_data.payload
                if mol.GetNumConformers() > 0:
                    conf = mol.GetConformer()
                    coords = np.array([conf.GetAtomPosition(i)
                                       for i in range(mol.GetNumAtoms())])

            meta = {}
            if receptor_data is not None:
                meta = getattr(receptor_data, 'metadata', {}) or {}
            elif protein_data is not None:
                meta = getattr(protein_data, 'metadata', {}) or {}

            if coords is None and meta.get('ligands'):
                first = next(iter(meta['ligands'].values()))
                cx, cy, cz = first['Center']
                sx, sy, sz = first['Box']
            elif coords is not None:
                padding = float(self.get_property('padding') or 5.0)
                min_xyz = coords.min(axis=0)
                max_xyz = coords.max(axis=0)
                center = (min_xyz + max_xyz) / 2.0
                size_arr = (max_xyz - min_xyz) + padding * 2
                cx, cy, cz = center.tolist()
                sx, sy, sz = size_arr.tolist()
            else:
                return False, 'Cannot determine box from ligand.'

            self.set_property('center', [cx, cy, cz])
            self.set_property('box_size', [sx, sy, sz])
        else:
            center = self.get_property('center') or [0, 0, 0]
            size = self.get_property('box_size') or [20, 20, 20]
            cx, cy, cz = [float(v) for v in center]
            sx, sy, sz = [float(v) for v in size]

        self.set_progress(60)

        # ── Resolve PDB string for the viewer ────────────────────────
        protein_pdb = ''
        if protein_data is not None and isinstance(protein_data, ProteinData):
            protein_pdb = protein_data.payload or ''
        elif receptor_data is not None:
            if isinstance(receptor_data, ReceptorData) and receptor_data.payload:
                protein_pdb = self._pdbqt_to_pdb(receptor_data.payload)
            elif isinstance(receptor_data, ProteinData) and receptor_data.payload:
                protein_pdb = receptor_data.payload

        ligand_pdb = ''
        mol_port = self.inputs().get('molecule')
        if mol_port and mol_port.connected_ports():
            src = mol_port.connected_ports()[0]
            mol_data = src.node().output_values.get(src.name())
            if MoleculeData is not None and isinstance(mol_data, MoleculeData):
                mol_obj = mol_data.payload
                if mol_obj is not None:
                    ligand_pdb = Chem.MolToPDBBlock(mol_obj)

        self.set_display({
            'protein_pdb': protein_pdb,
            'ligand_pdb': ligand_pdb,
            'cx': cx, 'cy': cy, 'cz': cz,
            'sx': sx, 'sy': sy, 'sz': sz,
        })

        self.set_progress(80)

        box_df = pd.DataFrame([{
            'center_x': cx, 'center_y': cy, 'center_z': cz,
            'size_x': sx, 'size_y': sy, 'size_z': sz,
        }])
        self.output_values['box_config'] = TableData(payload=box_df)

        self.mark_clean()
        self.set_progress(100)
        return True, (f'Box: center=({cx:.1f}, {cy:.1f}, {cz:.1f}) '
                      f'size=({sx:.1f}, {sy:.1f}, {sz:.1f})')

    def _display_ui(self, data):
        """Load structure + draw box in the embedded viewer (Main Thread)."""
        if not isinstance(data, dict):
            return
        self._viewer_widget.set_value(data)
        cx = data.get('cx', 0)
        cy = data.get('cy', 0)
        cz = data.get('cz', 0)
        sx = data.get('sx', 20)
        sy = data.get('sy', 20)
        sz = data.get('sz', 20)
        if cx or cy or cz:
            import functools
            from PySide6.QtCore import QTimer
            QTimer.singleShot(300, functools.partial(
                self._viewer_widget.run_js,
                f'drawCenter({cx},{cy},{cz});drawBox({cx},{cy},{cz},{sx},{sy},{sz});'))
        self.view.draw_node()


# ══════════════════════════════════════════════════════════════════════════════
#  Node 4: Vina Dock (single ligand)
# ══════════════════════════════════════════════════════════════════════════════

class VinaDockNode(BaseExecutionNode):
    """Dock a single ligand against a prepared receptor.

    Supports Vina CLI and QVina2 (Rust) backends.
    """

    __identifier__ = 'nodes.Cheminformatics.Docking'
    NODE_NAME      = 'Vina Dock'
    PORT_SPEC      = {'inputs': ['receptor', 'molecule', 'box_config'],
                      'outputs': ['molecule', 'table']}

    def __init__(self):
        super().__init__()
        self.add_combo_menu('backend', 'Backend', items=list(BACKENDS.keys()))
        self._add_row('dock_params', 'Docking', [
            {'name': 'exhaustiveness', 'label': 'Exh', 'type': 'int',
             'value': 8, 'min_val': 1, 'max_val': 128},
            {'name': 'n_poses', 'label': 'Poses', 'type': 'int',
             'value': 9, 'min_val': 1, 'max_val': 100},
            {'name': 'energy_range', 'label': 'E-range', 'type': 'float',
             'value': 3.0, 'min_val': 0.5, 'max_val': 20.0, 'step': 0.5, 'decimals': 1},
        ])
        self._add_row('dock_misc', '', [
            {'name': 'seed', 'label': 'Seed', 'type': 'int',
             'value': 42, 'min_val': 0, 'max_val': 999999},
            {'name': 'cpu', 'label': 'CPU', 'type': 'int',
             'value': 0, 'min_val': 0, 'max_val': 128},
        ])
        self.add_combo_menu('scoring', 'Scoring', items=['vina', 'vinardo', 'ad4'])

        self.add_input('receptor',   color=PORT_COLORS['receptor'])
        self.add_input('molecule',   color=PORT_COLORS.get('molecule', (205, 92, 92)))
        self.add_input('box_config', color=PORT_COLORS['box_config'])
        self.add_output('molecule',  color=PORT_COLORS.get('molecule', (205, 92, 92)))
        self.add_output('energies',  color=PORT_COLORS['table'])

    def evaluate(self):
        # Get receptor
        rec_port = self.inputs().get('receptor')
        if not (rec_port and rec_port.connected_ports()):
            return False, 'No receptor connected.'
        rec_src = rec_port.connected_ports()[0]
        rec_val = rec_src.node().output_values.get(rec_src.name())
        if not isinstance(rec_val, ReceptorData):
            return False, 'Expected ReceptorData.'

        # Get ligand
        mol_port = self.inputs().get('molecule')
        if not (mol_port and mol_port.connected_ports()):
            return False, 'No molecule connected.'
        mol_src = mol_port.connected_ports()[0]
        mol_val = mol_src.node().output_values.get(mol_src.name())
        if MoleculeData is None or not isinstance(mol_val, MoleculeData):
            return False, 'Expected MoleculeData.'
        mol = mol_val.payload
        if mol is None:
            return False, 'Invalid molecule.'

        # Get box config
        box_port = self.inputs().get('box_config')
        if not (box_port and box_port.connected_ports()):
            return False, 'No box_config connected.'
        box_src = box_port.connected_ports()[0]
        box_val = box_src.node().output_values.get(box_src.name())
        if not isinstance(box_val, TableData):
            return False, 'Expected TableData for box_config.'
        box_df = box_val.payload
        center = (box_df['center_x'].iloc[0], box_df['center_y'].iloc[0],
                  box_df['center_z'].iloc[0])
        size = (box_df['size_x'].iloc[0], box_df['size_y'].iloc[0],
                box_df['size_z'].iloc[0])

        # Ensure ligand has 3D conformer
        if mol.GetNumConformers() == 0:
            if embed_mol_3d is not None:
                mol_3d = embed_mol_3d(
                    mol, keep_hs=True, num_confs=1, random_seed=42,
                    random_coords_fallback=True)
                if mol_3d is None:
                    return False, 'Could not generate 3D conformer for ligand.'
            else:
                from rdkit.Chem import AllChem
                mol_3d = Chem.AddHs(mol)
                if AllChem.EmbedMolecule(mol_3d, randomSeed=42) < 0:
                    return False, 'Could not generate 3D conformer for ligand.'
        else:
            mol_3d = mol

        self.set_progress(10)

        # Convert each conformer to PDBQT
        conf_ids = [c.GetId() for c in mol_3d.GetConformers()]
        ligand_pdbqts = []
        for cid in conf_ids:
            try:
                ligand_pdbqts.append(_mol_to_pdbqt(mol_3d, confId=cid))
            except Exception:
                pass
        if not ligand_pdbqts:
            return False, 'Ligand PDBQT conversion failed for all conformers.'

        self.set_progress(20)

        # Set up backend
        backend_name = self.get_property('backend') or 'QVina2'
        exhaustiveness = int(self.get_property('exhaustiveness') or 8)
        n_poses = int(self.get_property('n_poses') or 9)
        energy_range = float(self.get_property('energy_range') or 3.0)
        seed = int(self.get_property('seed') or 42)
        cpu = int(self.get_property('cpu') or 0)
        scoring = self.get_property('scoring') or 'vina'

        backend = get_backend(backend_name)

        try:
            def _progress(**kwargs):
                pct = kwargs.get('percent_complete', 0)
                self.set_progress(int(20 + pct * 0.7))

            if len(ligand_pdbqts) == 1:
                poses_pdbqt, energies = backend.dock(
                    rec_val.payload, ligand_pdbqts[0], center, size,
                    exhaustiveness=exhaustiveness,
                    n_poses=n_poses,
                    energy_range=energy_range,
                    seed=seed,
                    cpu=cpu,
                    scoring=scoring,
                    flex_pdbqt=rec_val.flex_pdbqt,
                    progress_callback=_progress,
                )
            else:
                # Multiple conformers → batch dock + merge best poses
                results = backend.batch_dock(
                    rec_val.payload, ligand_pdbqts, center, size,
                    exhaustiveness=exhaustiveness,
                    n_poses=n_poses,
                    energy_range=energy_range,
                    seed=seed,
                    cpu=cpu,
                    scoring=scoring,
                    progress_callback=_progress,
                    stream_results=False,
                )
                poses_pdbqt, energies = _merge_dock_results(
                    results, n_poses)
                if not energies:
                    return False, 'Docking failed for all conformers.'
        except Exception as e:
            return False, f'Docking failed: {e}'

        self.set_progress(92)

        lig_name = mol_val.name or 'ligand'

        # Convert docked PDBQT → RDKit Mol (multi-conformer)
        docked_mol = None
        try:
            pdbqt_mol = PDBQTMolecule(poses_pdbqt, skip_typing=True)
            mol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
            docked_mol = RDKitMolCreate.combine_rdkit_mols(mol_list)
        except Exception:
            pass

        if docked_mol is not None and MoleculeData is not None:
            self.output_values['molecule'] = MoleculeData(
                payload=docked_mol, name=f'{lig_name}_docked')

        # Build energies table
        if energies:
            cols = ['affinity']
            if len(energies[0]) > 1:
                cols.extend(['dist_lb', 'dist_ub'])
            energy_df = pd.DataFrame(energies, columns=cols[:len(energies[0])])
            energy_df.index.name = 'pose'
            energy_df.index += 1
        else:
            energy_df = pd.DataFrame(columns=['affinity'])
        self.output_values['energies'] = TableData(payload=energy_df)

        self.mark_clean()
        self.set_progress(100)
        best = energies[0][0] if energies else 0.0
        n_out = docked_mol.GetNumConformers() if docked_mol else 0
        n_input_confs = len(ligand_pdbqts)
        msg = f'{lig_name}: {n_out} poses, best={best:.2f} kcal/mol'
        if n_input_confs > 1:
            msg += f' (from {n_input_confs} input conformers)'
        return True, msg


# ══════════════════════════════════════════════════════════════════════════════
#  Node 6: Batch Dock
# ══════════════════════════════════════════════════════════════════════════════

class _ProgressBarDelegate(QtWidgets.QStyledItemDelegate):
    """Paints a visual progress bar in the Progress column."""

    def paint(self, painter, option, index):
        value = index.data(QtCore.Qt.ItemDataRole.DisplayRole)
        # Status from column 1 of the same row
        model = index.model()
        status = model.data(
            model.index(index.row(), 1),
            QtCore.Qt.ItemDataRole.DisplayRole) or ''

        painter.save()
        rect = option.rect.adjusted(4, 4, -4, -4)

        # Background track
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor('#3a3a3a'))
        painter.drawRoundedRect(QtCore.QRectF(rect), 3, 3)

        if status in ('Docking', 'Docking...', 'Scoring', 'Scoring...'):
            pct = max(0, min(100, int(value or 0)))
            if pct > 0:
                fill = QtCore.QRectF(rect)
                fill.setWidth(rect.width() * pct / 100.0)
                painter.setBrush(QtGui.QColor('#42a5f5'))
                painter.drawRoundedRect(fill, 3, 3)
            painter.setPen(QtGui.QColor('#fff'))
            painter.drawText(option.rect,
                             QtCore.Qt.AlignmentFlag.AlignCenter, f'{pct}%')

        elif status in ('Refining', 'Refining...'):
            fill = QtCore.QRectF(rect)
            fill.setWidth(rect.width() * 0.99)
            painter.setBrush(QtGui.QColor('#ff9800'))
            painter.drawRoundedRect(fill, 3, 3)
            painter.setPen(QtGui.QColor('#fff'))
            painter.drawText(option.rect,
                             QtCore.Qt.AlignmentFlag.AlignCenter, 'Refining...')

        elif status == 'Done':
            painter.setBrush(QtGui.QColor('#4caf50'))
            painter.drawRoundedRect(QtCore.QRectF(rect), 3, 3)
            painter.setPen(QtGui.QColor('#fff'))
            if isinstance(value, float) and value == value:  # not NaN
                text = f'{value:.2f}'
            else:
                text = 'Done'
            painter.drawText(option.rect,
                             QtCore.Qt.AlignmentFlag.AlignCenter, text)

        elif status == 'Failed':
            painter.setBrush(QtGui.QColor('#4a2020'))
            painter.drawRoundedRect(QtCore.QRectF(rect), 3, 3)
            painter.setPen(QtGui.QColor('#e57373'))
            painter.drawText(option.rect,
                             QtCore.Qt.AlignmentFlag.AlignCenter, 'Failed')

        else:
            # Pending, Queued, Preparing
            painter.setPen(QtGui.QColor('#555'))
            painter.drawText(option.rect,
                             QtCore.Qt.AlignmentFlag.AlignCenter, '\u2014')

        painter.restore()

    def sizeHint(self, option, index):
        return QtCore.QSize(120, 22)


class _DockProgressModel(QtCore.QAbstractTableModel):
    """Table model for batch docking/scoring progress with progress bars."""

    _HEADERS = ['Name', 'Status', 'Progress']
    _STATUS_COLORS = {
        'Pending':     '#999999',
        'Queued':      '#b0bec5',
        'Preparing':   '#78909c',
        'Docking':     '#42a5f5',
        'Docking...':  '#42a5f5',
        'Scoring':     '#42a5f5',
        'Scoring...':  '#42a5f5',
        'Refining':    '#ff9800',
        'Refining...': '#ff9800',
        'Done':        '#4caf50',
        'Failed':      '#e57373',
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[list] = []

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 3

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._rows):
            return None
        row, col = index.row(), index.column()
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return self._rows[row][col]
        if role == QtCore.Qt.ItemDataRole.ForegroundRole and col == 1:
            status = self._rows[row][1]
            return QtGui.QColor(self._STATUS_COLORS.get(status, '#ddd'))
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole and col == 1:
            return QtCore.Qt.AlignmentFlag.AlignCenter
        return None

    def headerData(self, section, orientation,
                   role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return self._HEADERS[section]
            if orientation == QtCore.Qt.Orientation.Vertical:
                return str(section + 1)
        return None

    def set_rows(self, rows_data: list[dict]):
        """Accept list of dicts: [{name, status, score, progress}, ...]."""
        new_rows = []
        for row in rows_data:
            status = str(row.get('status', 'Pending'))
            if status == 'Done':
                val = row.get('score')
                if val is None:
                    val = 0
            else:
                p = row.get('progress')
                val = int(p) if p is not None else 0
            new_rows.append([str(row.get('name', '')), status, val])

        if len(new_rows) != len(self._rows):
            self.beginResetModel()
            self._rows = new_rows
            self.endResetModel()
        else:
            for i in range(len(new_rows)):
                if self._rows[i] != new_rows[i]:
                    self._rows[i] = new_rows[i]
                    self.dataChanged.emit(
                        self.index(i, 0), self.index(i, 2),
                        [QtCore.Qt.ItemDataRole.DisplayRole,
                         QtCore.Qt.ItemDataRole.ForegroundRole])


class _BatchProgressWidget(NodeBaseWidget):
    """Embedded progress table with visual progress bars for docking nodes."""

    def __init__(self, parent=None):
        super().__init__(parent, name='batch_progress', label='')
        self._view = QtWidgets.QTableView()
        self._view.setMinimumSize(420, 260)
        self._view.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._view.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._view.verticalHeader().setDefaultSectionSize(24)
        self._view.setStyleSheet("""
            QTableView {
                background-color: #222;
                color: #ddd;
                gridline-color: #444;
                border: 1px solid #555;
                font-size: 9pt;
            }
            QHeaderView::section {
                background-color: #333;
                color: #fff;
                padding: 3px;
                border: 1px solid #444;
            }
        """)
        self._model = _DockProgressModel(self._view)
        self._view.setModel(self._model)
        self._view.setItemDelegateForColumn(
            2, _ProgressBarDelegate(self._view))
        header = self._view.horizontalHeader()
        header.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.set_custom_widget(self._view)

    def set_value(self, rows):
        """Accept list of dicts: [{name, status, score, progress}, ...]."""
        if isinstance(rows, list):
            self._model.set_rows(rows)

    def get_value(self):
        return None


class BatchDockNode(BaseExecutionNode):
    """Dock every molecule in a MolTable against a prepared receptor.

    Docked poses are converted back to RDKit Mol objects and stored in
    the output mol_table (as conformers on the original molecule).
    A real-time progress table shows each molecule's docking status.
    """

    __identifier__ = 'nodes.Cheminformatics.Docking'
    NODE_NAME      = 'Batch Dock'
    PORT_SPEC      = {'inputs': ['receptor', 'mol_table', 'box_config'],
                      'outputs': ['table', 'mol_table']}

    def __init__(self):
        super().__init__()
        self.add_combo_menu('backend', 'Backend', items=list(BACKENDS.keys()))
        self._add_row('dock_params', 'Docking', [
            {'name': 'exhaustiveness', 'label': 'Exh', 'type': 'int',
             'value': 8, 'min_val': 1, 'max_val': 128},
            {'name': 'n_poses', 'label': 'Poses', 'type': 'int',
             'value': 9, 'min_val': 1, 'max_val': 100},
            {'name': 'energy_range', 'label': 'E-range', 'type': 'float',
             'value': 3.0, 'min_val': 0.5, 'max_val': 20.0, 'step': 0.5, 'decimals': 1},
        ])
        self._add_row('dock_misc', '', [
            {'name': 'seed', 'label': 'Seed', 'type': 'int',
             'value': 42, 'min_val': 0, 'max_val': 999999},
            {'name': 'cpu', 'label': 'CPU', 'type': 'int',
             'value': 0, 'min_val': 0, 'max_val': 128},
        ])
        self.add_combo_menu('scoring', 'Scoring', items=['vina', 'vinardo', 'ad4'])
        self.add_input('receptor',   color=PORT_COLORS['receptor'])
        self.add_input('mol_table',  color=PORT_COLORS.get('mol_table', (178, 102, 178)))
        self.add_input('box_config', color=PORT_COLORS['box_config'])
        self.add_output('results',   color=PORT_COLORS['table'])
        self.add_output('mol_table', color=PORT_COLORS.get('mol_table', (178, 102, 178)))

        # Embedded progress table
        self._batch_progress_w = _BatchProgressWidget(self.view)
        self.add_custom_widget(self._batch_progress_w, tab='Progress')

    def _get_mol_name(self, mol, idx):
        """Extract a display name from a molecule."""
        if mol is None:
            return f'mol_{idx + 1}'
        if mol.HasProp('_Name') and mol.GetProp('_Name').strip():
            return mol.GetProp('_Name').strip()
        if mol.HasProp('CHEMBL_ID'):
            return mol.GetProp('CHEMBL_ID')
        return f'mol_{idx + 1}'

    def evaluate(self):
        # Get receptor
        rec_port = self.inputs().get('receptor')
        if not (rec_port and rec_port.connected_ports()):
            return False, 'No receptor connected.'
        rec_src = rec_port.connected_ports()[0]
        rec_val = rec_src.node().output_values.get(rec_src.name())
        if not isinstance(rec_val, ReceptorData):
            return False, 'Expected ReceptorData.'

        # Get mol_table
        mt_port = self.inputs().get('mol_table')
        if not (mt_port and mt_port.connected_ports()):
            return False, 'No mol_table connected.'
        mt_src = mt_port.connected_ports()[0]
        mt_val = mt_src.node().output_values.get(mt_src.name())
        if MolTableData is None or not isinstance(mt_val, MolTableData):
            return False, 'Expected MolTableData.'
        df = mt_val.payload.copy()
        mol_col = mt_val.mol_col
        n = len(df)
        if n == 0:
            return False, 'Empty table.'

        # Get box config
        box_port = self.inputs().get('box_config')
        if not (box_port and box_port.connected_ports()):
            return False, 'No box_config connected.'
        box_src = box_port.connected_ports()[0]
        box_val = box_src.node().output_values.get(box_src.name())
        if not isinstance(box_val, TableData):
            return False, 'Expected TableData for box_config.'
        box_df = box_val.payload
        center = (box_df['center_x'].iloc[0], box_df['center_y'].iloc[0],
                  box_df['center_z'].iloc[0])
        size = (box_df['size_x'].iloc[0], box_df['size_y'].iloc[0],
                box_df['size_z'].iloc[0])

        # Set up backend
        backend_name = self.get_property('backend') or 'QVina2'
        exhaustiveness = int(self.get_property('exhaustiveness') or 8)
        n_poses = int(self.get_property('n_poses') or 9)
        energy_range = float(self.get_property('energy_range') or 3.0)
        seed = int(self.get_property('seed') or 42)
        cpu = int(self.get_property('cpu') or 0)
        scoring = self.get_property('scoring') or 'vina'

        self.set_progress(5)

        # Build name list and initial progress rows
        mols = df[mol_col].tolist()
        # Prefer the DataFrame 'name' column (has dedup suffixes) over mol _Name
        if 'name' in df.columns:
            names = df['name'].astype(str).tolist()
        else:
            names = [self._get_mol_name(m, i) for i, m in enumerate(mols)]
        progress_rows = [{'name': nm, 'status': 'Pending', 'score': None}
                         for nm in names]
        self.set_display(list(progress_rows))

        # ── Phase 1: Prepare all molecules (3D + H + PDBQT) ──────────────
        lig_pdbqts: list[list[str] | None] = [None] * n  # per-conformer PDBQTs
        prep_statuses: list[str | None] = [None] * n  # failure reason or None
        prepared_mols: list = [None] * n  # mol_3d with explicit Hs

        # 1a. Per-molecule: 3D embed + add Hs
        for i, mol in enumerate(mols):
            if mol is None:
                prep_statuses[i] = 'invalid_mol'
                progress_rows[i] = {'name': names[i], 'status': 'Failed',
                                    'score': None}
                self.set_display(list(progress_rows))
                continue

            progress_rows[i] = {'name': names[i], 'status': 'Preparing',
                                'score': None}
            self.set_display(list(progress_rows))

            # Ensure 3D
            if mol.GetNumConformers() == 0:
                if embed_mol_3d is not None:
                    mol_3d = embed_mol_3d(
                        mol, keep_hs=True, num_confs=1, random_seed=seed,
                        random_coords_fallback=True)
                    if mol_3d is None:
                        prep_statuses[i] = 'embed_failed'
                        progress_rows[i] = {'name': names[i],
                                            'status': 'Failed', 'score': None}
                        self.set_display(list(progress_rows))
                        continue
                else:
                    from rdkit.Chem import AllChem
                    mol_3d = Chem.AddHs(mol)
                    if AllChem.EmbedMolecule(mol_3d, randomSeed=seed) < 0:
                        prep_statuses[i] = 'embed_failed'
                        progress_rows[i] = {'name': names[i],
                                            'status': 'Failed', 'score': None}
                        self.set_display(list(progress_rows))
                        continue
            else:
                mol_3d = mol

            # Ensure explicit H (required for PDBQT conversion)
            if not any(a.GetAtomicNum() == 1 for a in mol_3d.GetAtoms()):
                mol_3d = Chem.AddHs(mol_3d, addCoords=True)

            prepared_mols[i] = mol_3d

        # 1b. Batch PDBQT conversion (one entry per conformer)
        valid_prep = [i for i in range(n) if prepared_mols[i] is not None]
        if valid_prep and _sdfrust is not None:
            # Collect sdfrust molecules and precomputed data
            rust_mols = []
            arom_list = []; sym_list = []
            smi_list = []; order_list = []
            idx_map = []  # batch position → original mol index
            for i in valid_prep:
                mol_3d = prepared_mols[i]
                try:
                    smi, order = _get_smiles_and_order(mol_3d)
                    arom = [a.GetIsAromatic() for a in mol_3d.GetAtoms()]
                    sym = list(Chem.CanonicalRankAtoms(mol_3d, breakTies=False))
                except Exception:
                    prep_statuses[i] = 'pdbqt_failed'
                    progress_rows[i] = {'name': names[i], 'status': 'Failed',
                                        'score': None}
                    continue
                # Each conformer → separate batch entry
                for conf in mol_3d.GetConformers():
                    try:
                        sdf_str = Chem.MolToMolBlock(
                            mol_3d, confId=conf.GetId())
                        rust_mol = _sdfrust.parse_sdf_string(sdf_str)
                        rust_mols.append(rust_mol)
                        arom_list.append(arom)
                        sym_list.append(sym)
                        smi_list.append(smi)
                        order_list.append(order)
                        idx_map.append(i)
                    except Exception:
                        pass  # skip this conformer
            if rust_mols:
                batch_results = _sdfrust.batch_mol_to_pdbqt(
                    rust_mols,
                    aromatic_atoms=arom_list,
                    symmetry_classes=sym_list,
                    smiles=smi_list,
                    smiles_atom_orders=order_list,
                )
                for j, pdbqt in enumerate(batch_results):
                    orig_i = idx_map[j]
                    if pdbqt is not None:
                        if lig_pdbqts[orig_i] is None:
                            lig_pdbqts[orig_i] = []
                        lig_pdbqts[orig_i].append(pdbqt)
            # Mark molecules with zero successful conformers as failed
            for i in valid_prep:
                if not lig_pdbqts[i]:
                    lig_pdbqts[i] = None
                    prep_statuses[i] = 'pdbqt_failed'
                    progress_rows[i] = {'name': names[i], 'status': 'Failed',
                                        'score': None}
            self.set_display(list(progress_rows))
        else:
            # Fallback: per-molecule conversion (Meeko or single sdfrust)
            for i in valid_prep:
                pdbqts = []
                for conf in prepared_mols[i].GetConformers():
                    try:
                        pdbqts.append(_mol_to_pdbqt(
                            prepared_mols[i], confId=conf.GetId()))
                    except Exception:
                        pass
                if pdbqts:
                    lig_pdbqts[i] = pdbqts
                else:
                    prep_statuses[i] = 'pdbqt_failed'
                    progress_rows[i] = {'name': names[i], 'status': 'Failed',
                                        'score': None}
                    self.set_display(list(progress_rows))

        # ── Phase 2: Dock (parallel or sequential) ────────────────────────
        best_energies = [np.nan] * n
        n_poses_list = [0] * n
        statuses = [s or 'pending' for s in prep_statuses]
        dock_pdbqts: list[str | None] = [None] * n  # raw docked poses PDBQT

        # Indices of molecules that have valid PDBQTs
        dock_indices = [i for i in range(n) if lig_pdbqts[i] is not None]

        # Flatten multi-conformer PDBQTs for batch docking
        flat_pdbqts = []
        flat_to_orig = []  # flat index → original molecule index
        for i in dock_indices:
            for pdbqt in lig_pdbqts[i]:
                flat_pdbqts.append(pdbqt)
                flat_to_orig.append(i)
        conf_counts = {i: len(lig_pdbqts[i]) for i in dock_indices}

        if len(flat_pdbqts) > 1:
            # ── Rust batch_dock (grid maps + outer Rayon) ─────────────
            for i in dock_indices:
                progress_rows[i] = {'name': names[i], 'status': 'Queued',
                                    'score': None}
            self.set_display(list(progress_rows))

            n_flat = len(flat_pdbqts)

            import time as _time
            _last_update = [0.0]
            _lig_pcts = {}  # flat_idx → last known per-ligand %

            # Buffers for merging conformer results per molecule
            conf_results_buf = {i: [] for i in dock_indices}
            conf_done = {i: 0 for i in dock_indices}

            def _batch_progress(**kwargs):
                flat_idx = int(kwargs.get('ligand_index', 0))
                if flat_idx >= len(flat_to_orig):
                    return
                orig_i = flat_to_orig[flat_idx]
                # Rust sends overall = (flat_idx*100 + per_lig%) / total
                overall_pct = kwargs.get('percent_complete', 0)
                per_lig = max(0.0, min(100.0,
                    overall_pct * n_flat - flat_idx * 100))
                _lig_pcts[flat_idx] = per_lig

                stage = kwargs.get('stage', '')
                if stage == 'LigandDone':
                    # Buffer conformer result
                    poses_pdbqt = kwargs.get('result_pdbqt', '')
                    energies = kwargs.get('result_energies', [])
                    if poses_pdbqt and energies:
                        conf_results_buf[orig_i].append(
                            (poses_pdbqt, [tuple(e) for e in energies]))
                    conf_done[orig_i] += 1

                    # When all conformers for this molecule are done, merge
                    if conf_done[orig_i] >= conf_counts[orig_i]:
                        if conf_results_buf[orig_i]:
                            merged_pdbqt, merged_e = _merge_dock_results(
                                conf_results_buf[orig_i], n_poses)
                            if merged_e:
                                best_e = merged_e[0][0]
                                best_energies[orig_i] = best_e
                                n_poses_list[orig_i] = len(merged_e)
                                statuses[orig_i] = 'success'
                                dock_pdbqts[orig_i] = merged_pdbqt
                                score_display = best_e
                            else:
                                statuses[orig_i] = 'failed'
                                score_display = None
                        else:
                            statuses[orig_i] = 'failed'
                            score_display = None
                        progress_rows[orig_i] = {
                            'name': names[orig_i],
                            'status': 'Done' if statuses[orig_i] == 'success' else 'Failed',
                            'score': score_display}
                        _last_update[0] = _time.monotonic()
                        true_overall = sum(_lig_pcts.values()) / n_flat
                        self.set_progress(int(5 + true_overall * 0.9))
                        self.set_display(list(progress_rows))
                        return

                    # Not all conformers done yet — keep showing Docking
                    progress_rows[orig_i]['status'] = 'Docking'
                    _last_update[0] = _time.monotonic()
                    true_overall = sum(_lig_pcts.values()) / n_flat
                    self.set_progress(int(5 + true_overall * 0.9))
                    self.set_display(list(progress_rows))
                    return

                progress_rows[orig_i]['status'] = 'Docking'

                # Throttle UI updates
                now = _time.monotonic()
                if now - _last_update[0] > 0.3:
                    _last_update[0] = now
                    true_overall = sum(_lig_pcts.values()) / n_flat
                    self.set_progress(int(5 + true_overall * 0.9))
                    self.set_display(list(progress_rows))

            backend = get_backend(backend_name)
            try:
                backend.batch_dock(
                    rec_val.payload, flat_pdbqts, center, size,
                    exhaustiveness=exhaustiveness,
                    n_poses=n_poses,
                    energy_range=energy_range,
                    seed=seed,
                    cpu=cpu,
                    scoring=scoring,
                    progress_callback=_batch_progress,
                    stream_results=True,
                )
            except Exception:
                # Mark any un-completed molecules as failed
                for i in dock_indices:
                    if statuses[i] not in ('success', 'failed'):
                        statuses[i] = 'failed'
                        progress_rows[i] = {'name': names[i],
                                            'status': 'Failed', 'score': None}
            self.set_display(list(progress_rows))
            self.set_progress(95)
        else:
            # ── Sequential docking (single PDBQT entry) ──────────────
            backend = get_backend(backend_name)
            for flat_idx in range(len(flat_pdbqts)):
                orig_i = flat_to_orig[flat_idx]
                progress_rows[orig_i] = {'name': names[orig_i],
                                         'status': 'Docking',
                                         'score': None, 'progress': 0}
                self.set_display(list(progress_rows))

                def _mol_progress(_idx=orig_i, **kwargs):
                    pct = kwargs.get('percent_complete', 0)
                    progress_rows[_idx]['progress'] = min(int(pct), 99)
                    if int(pct) % 10 == 0:
                        self.set_display(list(progress_rows))

                try:
                    poses_pdbqt, energies = backend.dock(
                        rec_val.payload, flat_pdbqts[flat_idx], center, size,
                        exhaustiveness=exhaustiveness,
                        n_poses=n_poses,
                        energy_range=energy_range,
                        seed=seed,
                        cpu=cpu,
                        scoring=scoring,
                        progress_callback=_mol_progress,
                    )
                    best_e = energies[0][0] if energies else np.nan
                    best_energies[orig_i] = best_e
                    n_poses_list[orig_i] = len(energies)
                    statuses[orig_i] = 'success'
                    dock_pdbqts[orig_i] = poses_pdbqt
                    score_display = best_e if not np.isnan(best_e) else None
                    progress_rows[orig_i] = {'name': names[orig_i],
                                             'status': 'Done',
                                             'score': score_display}
                except Exception:
                    statuses[orig_i] = 'failed'
                    progress_rows[orig_i] = {'name': names[orig_i],
                                             'status': 'Failed',
                                             'score': None}

                self.set_display(list(progress_rows))
                pct = int(5 + (flat_idx + 1) /
                          max(len(flat_pdbqts), 1) * 90)
                self.set_progress(pct)

        # ── Phase 3: Convert docked PDBQT → RDKit Mol ────────────────────
        docked_mols: list = [None] * n
        for i in range(n):
            if dock_pdbqts[i] is None:
                continue
            try:
                pdbqt_mol = PDBQTMolecule(dock_pdbqts[i], skip_typing=True)
                mol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
                docked_mols[i] = RDKitMolCreate.combine_rdkit_mols(mol_list)
            except Exception:
                pass

        df['dock_score'] = best_energies
        df['dock_n_poses'] = n_poses_list
        df['dock_status'] = statuses
        df['dock_mol'] = docked_mols

        self.set_progress(97)

        # Results table (without Mol columns)
        result_df = df.drop(columns=[mol_col, 'dock_mol'], errors='ignore')
        self.output_values['results'] = TableData(payload=result_df)
        self.output_values['mol_table'] = MolTableData(payload=df, mol_col=mol_col)

        self.mark_clean()
        self.set_progress(100)
        n_ok = statuses.count('success')
        valid_e = [e for e in best_energies if not np.isnan(e)]
        best_overall = min(valid_e, default=0.0)
        n_total_confs = len(flat_pdbqts) if flat_pdbqts else n
        msg = f'Docked {n_ok}/{n} molecules, best={best_overall:.2f} kcal/mol'
        if n_total_confs > n:
            msg += f' ({n_total_confs} total conformers)'
        return True, msg

    def _display_ui(self, data):
        """Update the batch progress table (Main Thread)."""
        if isinstance(data, list):
            self._batch_progress_w.set_value(data)
            self.view.draw_node()


# ══════════════════════════════════════════════════════════════════════════════
#  Node 8: GNINA CNN Rescore
# ══════════════════════════════════════════════════════════════════════════════

class GNINARescoreNode(BaseExecutionNode):
    """Rescore docking poses with GNINA CNN models.

    Accepts:
      - MoleculeData (single docked molecule from VinaDockNode, conformers = poses)
      - MolTableData (batch results from BatchDockNode, uses dock_mol conformers)
      - DockingResultData (legacy PDBQT poses)

    Outputs a scores table and (for batch mode) an updated mol_table with
    CNN scores added as columns.
    """

    __identifier__ = 'nodes.Cheminformatics.Docking'
    NODE_NAME      = 'GNINA Rescore'
    PORT_SPEC      = {'inputs': ['receptor', 'mol_table'],
                      'outputs': ['table', 'mol_table']}

    def __init__(self):
        super().__init__()
        self.add_combo_menu(
            'ensemble', 'CNN Ensemble',
            items=['default', 'dense', 'dense_1.3', 'dense_1.3_PT_KD',
                   'dense_1.3_PT_KD_def2018',
                   'crossdock_default2018', 'crossdock_default2018_1.3',
                   'crossdock_default2018_KD',
                   'general_default2018', 'general_default2018_KD',
                   'redock_default2018', 'redock_default2018_1.3',
                   'redock_default2018_KD'])
        self.add_combo_menu(
            'score_mode', 'Score Mode',
            items=['best_pose', 'all_poses'])
        self._add_int_spinbox('gnina_workers', 'Scoring Workers',
                              value=4, min_val=1, max_val=16)

        self.add_input('receptor', color=PORT_COLORS['receptor'])
        self.add_input('result',   color=PORT_COLORS.get('molecule', (205, 92, 92)))
        self.add_output('scores',  color=PORT_COLORS['table'])
        self.add_output('mol_table',
                        color=PORT_COLORS.get('mol_table', (178, 102, 178)))

        self._batch_progress_w = _BatchProgressWidget(self.view)
        self.add_custom_widget(self._batch_progress_w, tab='Progress')

        self._gnina_model = None
        self._loaded_ensemble = None

    def _get_model(self, ensemble_name):
        """Lazy-load GNINA model (reuse if same ensemble)."""
        if self._gnina_model is None or self._loaded_ensemble != ensemble_name:
            from .gnina_scorer import GNINAModel
            self._gnina_model = GNINAModel(ensemble=ensemble_name)
            self._loaded_ensemble = ensemble_name
        return self._gnina_model

    def _display_ui(self, data):
        """Update the batch progress table (Main Thread)."""
        if isinstance(data, list):
            self._batch_progress_w.set_value(data)
            self.view.draw_node()

    def _parse_receptor(self, rec_val):
        """Return (rec_coords, rec_types) from ProteinData or ReceptorData."""
        if isinstance(rec_val, ProteinData):
            from .gnina_scorer import parse_pdb_string
            return parse_pdb_string(rec_val.payload)
        elif isinstance(rec_val, ReceptorData):
            from .gnina_scorer import parse_pdbqt_string
            return parse_pdbqt_string(rec_val.payload)
        raise TypeError('Expected ProteinData or ReceptorData.')

    def evaluate(self):
        # ── Get receptor ──
        rec_port = self.inputs().get('receptor')
        if not (rec_port and rec_port.connected_ports()):
            return False, 'No receptor connected.'
        rec_src = rec_port.connected_ports()[0]
        rec_val = rec_src.node().output_values.get(rec_src.name())

        try:
            rec_coords, rec_types = self._parse_receptor(rec_val)
        except Exception as e:
            return False, f'Failed to parse receptor: {e}'

        self.set_progress(5)

        # ── Get docking result ──
        res_port = self.inputs().get('result')
        if not (res_port and res_port.connected_ports()):
            return False, 'No docking result connected.'
        res_src = res_port.connected_ports()[0]
        res_val = res_src.node().output_values.get(res_src.name())

        # ── Load GNINA model ──
        ensemble_name = self.get_property('ensemble') or 'default'
        try:
            model = self._get_model(ensemble_name)
        except Exception as e:
            return False, f'Failed to load GNINA models: {e}'

        # ── Dispatch based on input type ──
        if MoleculeData is not None and isinstance(res_val, MoleculeData):
            return self._score_molecule(model, rec_coords, rec_types, res_val)
        elif MolTableData is not None and isinstance(res_val, MolTableData):
            return self._score_batch(model, rec_coords, rec_types, res_val)
        elif isinstance(res_val, DockingResultData):
            return self._score_pdbqt(model, rec_coords, rec_types, res_val)
        else:
            return False, 'Expected MoleculeData, MolTableData, or DockingResultData.'

    def _score_mol_conformers(self, model, rec_coords, rec_types, mol,
                              all_poses=False):
        """Score conformers of an RDKit Mol. Returns list of result dicts."""
        from .gnina_scorer import process_molecule

        if mol is None or mol.GetNumConformers() == 0:
            return []

        conf_ids = [c.GetId() for c in mol.GetConformers()]
        if not all_poses:
            conf_ids = conf_ids[:1]

        poses = []
        valid_conf_ids = []
        for conf_id in conf_ids:
            try:
                coords, types = process_molecule(
                    mol, remove_h=True, conf_id=conf_id)
                poses.append((coords, types))
                valid_conf_ids.append(conf_id)
            except Exception:
                pass

        if not poses:
            return []

        center = np.mean(poses[0][0], axis=0)
        results = model.score_poses(rec_coords, rec_types, poses, center=center)

        # Attach conf_id to each result
        for i, res in enumerate(results):
            res['conf_id'] = valid_conf_ids[i]
        return results

    def _score_molecule(self, model, rec_coords, rec_types, mol_data):
        """Score a single MoleculeData (docked conformers from VinaDockNode)."""
        mol = mol_data.payload
        if mol is None or mol.GetNumConformers() == 0:
            return False, 'No docked conformers in molecule.'

        score_mode = self.get_property('score_mode') or 'best_pose'
        all_poses = (score_mode == 'all_poses')

        self.set_progress(15)

        results = self._score_mol_conformers(
            model, rec_coords, rec_types, mol, all_poses=all_poses)

        if not results:
            return False, 'No valid poses could be scored.'

        self.set_progress(90)

        rows = []
        for res in results:
            rows.append({
                'pose': res['conf_id'] + 1,
                'CNNscore': round(res['CNNscore'], 4),
                'CNNaffinity': round(res['CNNaffinity'], 2),
            })

        df = pd.DataFrame(rows)
        self.output_values['scores'] = TableData(payload=df)

        self.mark_clean()
        self.set_progress(100)

        best_cnn = max(r['CNNscore'] for r in results)
        best_aff = max(r['CNNaffinity'] for r in results)
        return True, (f'{len(results)} poses rescored, '
                      f'best CNNscore={best_cnn:.3f}, '
                      f'CNNaffinity={best_aff:.1f}')

    def _score_pdbqt(self, model, rec_coords, rec_types, res_val):
        """Score a legacy DockingResultData (PDBQT poses)."""
        from .gnina_scorer import parse_pdbqt_poses
        poses = parse_pdbqt_poses(res_val.payload)
        if not poses:
            return False, 'No poses found in docking result.'

        self.set_progress(15)

        center = np.mean(poses[0][0], axis=0)
        results = model.score_poses(rec_coords, rec_types, poses, center=center)

        self.set_progress(90)

        rows = []
        vina_energies = res_val.energies or []
        for i, res in enumerate(results):
            row = {'pose': i + 1}
            if i < len(vina_energies):
                e = vina_energies[i]
                row['vina_affinity'] = e[0] if isinstance(e, (list, tuple)) else e
            row['CNNscore'] = round(res['CNNscore'], 4)
            row['CNNaffinity'] = round(res['CNNaffinity'], 2)
            rows.append(row)

        df = pd.DataFrame(rows)
        self.output_values['scores'] = TableData(payload=df)

        self.mark_clean()
        self.set_progress(100)

        best_cnn = max(r['CNNscore'] for r in results) if results else 0
        best_aff = max(r['CNNaffinity'] for r in results) if results else 0
        return True, (f'{len(poses)} poses rescored, '
                      f'best CNNscore={best_cnn:.3f}, '
                      f'CNNaffinity={best_aff:.1f}')

    def _score_batch(self, model, rec_coords, rec_types, mt_val):
        """Score batch docking results from MolTableData with dock_mol column."""
        from .gnina_scorer import process_molecule

        df = mt_val.payload.copy()
        mol_col = mt_val.mol_col

        if 'dock_mol' not in df.columns:
            return False, 'MolTableData has no dock_mol column (run BatchDock first).'

        score_mode = self.get_property('score_mode') or 'best_pose'
        all_poses = (score_mode == 'all_poses')
        n = len(df)

        # Build molecule names
        if 'name' in df.columns:
            names = df['name'].astype(str).tolist()
        else:
            names = [f'mol_{i + 1}' for i in range(n)]

        # Initialize progress table
        progress_rows = [{'name': nm, 'status': 'Pending', 'score': None}
                         for nm in names]
        self.set_display(list(progress_rows))
        self.set_progress(5)

        # Pre-extract poses for all molecules (CPU-bound prep, no model needed)
        mol_pose_data: list[tuple[int, list, list] | None] = []
        # Each entry: (mol_idx, mol_poses, valid_conf_ids) or None
        for mol_idx in range(n):
            dock_mol = df['dock_mol'].iloc[mol_idx]
            if dock_mol is None or dock_mol.GetNumConformers() == 0:
                mol_pose_data.append(None)
                progress_rows[mol_idx] = {'name': names[mol_idx],
                                          'status': 'Failed', 'score': None}
                self.set_display(list(progress_rows))
                continue

            conf_ids = [c.GetId() for c in dock_mol.GetConformers()]
            if not all_poses:
                conf_ids = conf_ids[:1]

            mol_poses = []
            valid_conf_ids = []
            for conf_id in conf_ids:
                try:
                    coords, types = process_molecule(
                        dock_mol, remove_h=True, conf_id=conf_id)
                    mol_poses.append((coords, types))
                    valid_conf_ids.append(conf_id)
                except Exception:
                    pass

            if not mol_poses:
                mol_pose_data.append(None)
                progress_rows[mol_idx] = {'name': names[mol_idx],
                                          'status': 'Failed', 'score': None}
                self.set_display(list(progress_rows))
            else:
                mol_pose_data.append((mol_idx, mol_poses, valid_conf_ids))

        # Indices of scorable molecules
        score_indices = [i for i in range(n) if mol_pose_data[i] is not None]

        all_rows = []
        best_cnn_scores = {}
        best_cnn_affinities = {}
        gnina_workers = max(1, int(self.get_property('gnina_workers') or 4))

        def _score_one_mol(idx):
            """Score a single molecule's poses (thread-safe, shared model)."""
            _, mol_poses, valid_conf_ids = mol_pose_data[idx]
            center = np.mean(mol_poses[0][0], axis=0)
            results = model.score_poses(
                rec_coords, rec_types, mol_poses, center=center)
            return idx, results, valid_conf_ids

        if gnina_workers > 1 and len(score_indices) > 1:
            # ── Threaded GNINA scoring ────────────────────────────────
            for i in score_indices:
                progress_rows[i] = {'name': names[i], 'status': 'Queued',
                                    'score': None}
            self.set_display(list(progress_rows))

            done_count = 0
            with ThreadPoolExecutor(max_workers=gnina_workers) as pool:
                futures = {pool.submit(_score_one_mol, i): i
                           for i in score_indices}
                for future in as_completed(futures):
                    mol_idx = futures[future]
                    done_count += 1
                    try:
                        _, results, valid_conf_ids = future.result()
                        for conf_id, res in zip(valid_conf_ids, results):
                            row = {
                                'molecule': f"{names[mol_idx]}_p{conf_id + 1}",
                                'mol_idx': mol_idx,
                                'pose': conf_id + 1,
                                'CNNscore': round(res['CNNscore'], 4),
                                'CNNaffinity': round(res['CNNaffinity'], 2),
                            }
                            if 'dock_score' in df.columns:
                                row['vina_affinity'] = df['dock_score'].iloc[mol_idx]
                            all_rows.append(row)
                            if mol_idx not in best_cnn_scores or res['CNNscore'] > best_cnn_scores[mol_idx]:
                                best_cnn_scores[mol_idx] = round(res['CNNscore'], 4)
                                best_cnn_affinities[mol_idx] = round(res['CNNaffinity'], 2)

                        best_s = best_cnn_scores.get(mol_idx)
                        progress_rows[mol_idx] = {'name': names[mol_idx],
                                                  'status': 'Done', 'score': best_s}
                    except Exception:
                        progress_rows[mol_idx] = {'name': names[mol_idx],
                                                  'status': 'Failed', 'score': None}
                    self.set_display(list(progress_rows))
                    self.set_progress(int(5 + done_count / n * 85))
        else:
            # ── Sequential GNINA scoring ──────────────────────────────
            for count, mol_idx in enumerate(score_indices):
                progress_rows[mol_idx] = {'name': names[mol_idx],
                                          'status': 'Scoring', 'score': None}
                self.set_display(list(progress_rows))

                try:
                    _, results, valid_conf_ids = _score_one_mol(mol_idx)
                    for conf_id, res in zip(valid_conf_ids, results):
                        row = {
                            'molecule': f"{names[mol_idx]}_p{conf_id + 1}",
                            'mol_idx': mol_idx,
                            'pose': conf_id + 1,
                            'CNNscore': round(res['CNNscore'], 4),
                            'CNNaffinity': round(res['CNNaffinity'], 2),
                        }
                        if 'dock_score' in df.columns:
                            row['vina_affinity'] = df['dock_score'].iloc[mol_idx]
                        all_rows.append(row)
                        if mol_idx not in best_cnn_scores or res['CNNscore'] > best_cnn_scores[mol_idx]:
                            best_cnn_scores[mol_idx] = round(res['CNNscore'], 4)
                            best_cnn_affinities[mol_idx] = round(res['CNNaffinity'], 2)

                    best_s = best_cnn_scores.get(mol_idx)
                    progress_rows[mol_idx] = {'name': names[mol_idx],
                                              'status': 'Done', 'score': best_s}
                except Exception:
                    progress_rows[mol_idx] = {'name': names[mol_idx],
                                              'status': 'Failed', 'score': None}
                self.set_display(list(progress_rows))
                self.set_progress(int(5 + (count + 1) / n * 85))

        if not all_rows:
            return False, 'No valid poses found in dock_mol column.'

        scores_df = pd.DataFrame(all_rows)
        scores_df = scores_df.sort_values(
            ['mol_idx', 'pose'], ignore_index=True)
        scores_df = scores_df.drop(columns=['mol_idx'])
        self.output_values['scores'] = TableData(payload=scores_df)

        # Add best CNN scores as columns in the mol_table
        df['cnn_score'] = [best_cnn_scores.get(i, np.nan) for i in range(n)]
        df['cnn_affinity'] = [best_cnn_affinities.get(i, np.nan) for i in range(n)]
        self.output_values['mol_table'] = MolTableData(
            payload=df, mol_col=mol_col)

        self.mark_clean()
        self.set_progress(100)

        n_scored = len(best_cnn_scores)
        total_poses = len(all_rows)
        best_s = max((r['CNNscore'] for r in all_rows), default=0)
        best_a = max((r['CNNaffinity'] for r in all_rows), default=0)
        return True, (f'{n_scored} molecules, {total_poses} poses rescored, '
                      f'best CNNscore={best_s:.3f}, '
                      f'CNNaffinity={best_a:.1f}')


# ══════════════════════════════════════════════════════════════════════════════
#  Node: Structure Writer — export protein / receptor to file
# ══════════════════════════════════════════════════════════════════════════════

class StructureWriterNode(BaseExecutionNode):
    """Write a protein (PDB) or prepared receptor (PDBQT) to a file.

    Accepts ProteinData or ReceptorData on the *structure* input.
    When connected to a ReceptorData with flexible residues, the flex PDBQT
    is written to a separate ``*_flex.pdbqt`` file alongside the rigid one.
    """

    __identifier__ = 'nodes.Cheminformatics.Protein'
    NODE_NAME      = 'Structure Writer'
    PORT_SPEC      = {'inputs': ['receptor'], 'outputs': []}

    def __init__(self):
        super().__init__()
        saver = NodeFileSaver(
            self.view, name='save_path', label='Output File',
            ext_filter='PDBQT (*.pdbqt);;PDB (*.pdb);;CIF (*.cif);;All Files (*)')
        self.add_custom_widget(
            saver,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties')
        self.add_checkbox('auto_ext', 'Auto Extension', state=True)
        self.add_input('structure', color=PORT_COLORS.get('receptor', (0, 128, 128)))

    def evaluate(self):
        in_port = self.inputs().get('structure')
        if not (in_port and in_port.connected_ports()):
            return False, 'No structure connected.'
        src = in_port.connected_ports()[0]
        val = src.node().output_values.get(src.name())

        if not isinstance(val, (ProteinData, ReceptorData)):
            return False, 'Expected ProteinData or ReceptorData.'

        save_path = (self.get_property('save_path') or '').strip()
        if not save_path:
            return False, 'Select an output file path.'

        auto_ext = bool(self.get_property('auto_ext'))
        p = Path(save_path).expanduser()

        # Determine correct extension if auto
        if auto_ext and not p.suffix:
            if isinstance(val, ReceptorData):
                p = p.with_suffix('.pdbqt')
            else:
                ext = '.cif' if getattr(val, 'format', 'pdb') == 'cif' else '.pdb'
                p = p.with_suffix(ext)

        self.set_progress(10)

        # Ensure parent directory exists
        p.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(val, ReceptorData):
            # Write rigid PDBQT
            with open(p, 'w') as f:
                f.write(val.payload)
            n_atoms = sum(1 for l in val.payload.splitlines()
                         if l.startswith('ATOM') or l.startswith('HETATM'))
            msg = f'Wrote {n_atoms} atoms → {p.name}'

            # Write flex PDBQT alongside if present
            if val.flex_pdbqt and val.flex_pdbqt.strip():
                flex_path = p.with_name(p.stem + '_flex' + p.suffix)
                with open(flex_path, 'w') as f:
                    f.write(val.flex_pdbqt)
                msg += f' + {flex_path.name}'

        else:
            # ProteinData — write PDB/CIF
            with open(p, 'w') as f:
                f.write(val.payload)
            n_atoms = sum(1 for l in val.payload.splitlines()
                         if l.startswith('ATOM') or l.startswith('HETATM'))
            msg = f'Wrote {n_atoms} atoms → {p.name}'

        self.mark_clean()
        self.set_progress(100)
        return True, msg


# ══════════════════════════════════════════════════════════════════════════════
#  DrugCLIP Virtual Screening
# ══════════════════════════════════════════════════════════════════════════════

class DrugCLIPScreenNode(BaseExecutionNode):
    """Screen molecules against a protein pocket using DrugCLIP embeddings.

    Computes contrastive similarity between a protein binding pocket and
    molecules via the DrugCLIP dual-encoder model (ONNX Runtime).

    Inputs:
      - receptor  (ProteinData or ReceptorData)
      - box_config (TableData from DockingBoxNode — defines pocket center)
      - mol_table  (MolTableData — batch of molecules)

    Outputs:
      - mol_table  (MolTableData with ``drugclip_score`` column, sorted desc)
      - table      (TableData — summary scores)

    Keywords: DrugCLIP, virtual screening, contrastive, embedding, similarity
    """

    __identifier__ = 'nodes.Cheminformatics.Docking'
    NODE_NAME      = 'DrugCLIP Screen'
    PORT_SPEC      = {'inputs': ['receptor', 'box_config', 'mol_table'],
                      'outputs': ['mol_table', 'scores']}

    def __init__(self):
        super().__init__()
        self._add_float_spinbox(
            'box_padding', 'Box Padding (A)',
            value=4.0, min_val=0.0, max_val=20.0, step=0.5, decimals=1)
        self._add_int_spinbox(
            'max_pocket_atoms', 'Max Pocket Atoms',
            value=256, min_val=32, max_val=512)
        self._add_int_spinbox(
            'num_confs', 'Conformers (if no 3D)',
            value=5, min_val=1, max_val=20)
        self._add_int_spinbox(
            'screen_workers', 'Workers',
            value=4, min_val=1, max_val=16)

        self.add_input('receptor',  color=PORT_COLORS['receptor'])
        self.add_input('box_config', color=PORT_COLORS['box_config'])
        self.add_input('mol_table',
                       color=PORT_COLORS.get('mol_table', (178, 102, 178)))
        self.add_output('mol_table',
                        color=PORT_COLORS.get('mol_table', (178, 102, 178)))
        self.add_output('scores', color=PORT_COLORS['table'])

        self._batch_progress_w = _BatchProgressWidget(self.view)
        self.add_custom_widget(self._batch_progress_w, tab='Progress')

        self._drugclip_model = None
        self._pocket_emb_cache = None
        self._pocket_cache_key = None

    # ── Lazy model loading ───────────────────────────────────────────────

    def _get_model(self):
        if self._drugclip_model is None:
            from .drugclip_scorer import DrugCLIPModel
            self._drugclip_model = DrugCLIPModel()
        return self._drugclip_model

    def _display_ui(self, data):
        if isinstance(data, list):
            self._batch_progress_w.set_value(data)
            self.view.draw_node()

    # ── Evaluate ─────────────────────────────────────────────────────────

    def evaluate(self):
        from .drugclip_scorer import (
            extract_pocket_from_pdb, extract_pocket_from_pdbqt,
            mol_to_atoms_coords,
        )

        # ── 1. Receptor ──────────────────────────────────────────────────
        rec_port = self.inputs().get('receptor')
        if not (rec_port and rec_port.connected_ports()):
            return False, 'No receptor connected.'
        rec_src = rec_port.connected_ports()[0]
        rec_val = rec_src.node().output_values.get(rec_src.name())
        if not isinstance(rec_val, (ProteinData, ReceptorData)):
            return False, 'Expected ProteinData or ReceptorData.'

        # ── 2. Box config (center + size) ─────────────────────────────────
        box_port = self.inputs().get('box_config')
        if not (box_port and box_port.connected_ports()):
            return False, 'No box_config connected.'
        box_src = box_port.connected_ports()[0]
        box_val = box_src.node().output_values.get(box_src.name())
        if not isinstance(box_val, TableData):
            return False, 'Expected TableData for box_config.'
        box_df = box_val.payload
        center = np.array([
            box_df['center_x'].iloc[0],
            box_df['center_y'].iloc[0],
            box_df['center_z'].iloc[0],
        ], dtype=np.float64)
        box_size = np.array([
            box_df['size_x'].iloc[0],
            box_df['size_y'].iloc[0],
            box_df['size_z'].iloc[0],
        ], dtype=np.float64)

        # ── 3. Molecule table ────────────────────────────────────────────
        mt_port = self.inputs().get('mol_table')
        if not (mt_port and mt_port.connected_ports()):
            return False, 'No mol_table connected.'
        mt_src = mt_port.connected_ports()[0]
        mt_val = mt_src.node().output_values.get(mt_src.name())
        if MolTableData is None or not isinstance(mt_val, MolTableData):
            return False, 'Expected MolTableData.'

        self.set_progress(5)

        # ── 4. Load model ────────────────────────────────────────────────
        try:
            model = self._get_model()
        except Exception as e:
            return False, f'Failed to load DrugCLIP: {e}'

        # ── 5. Encode pocket (cached) ────────────────────────────────────
        box_padding = float(self.get_property('box_padding') or 4.0)
        max_pocket_atoms = int(self.get_property('max_pocket_atoms') or 256)
        cache_key = (hash(rec_val.payload), tuple(center),
                     tuple(box_size), box_padding)

        if self._pocket_cache_key != cache_key:
            try:
                if isinstance(rec_val, ReceptorData):
                    pkt_atoms, pkt_coords = extract_pocket_from_pdbqt(
                        rec_val.payload, center, box_size, box_padding)
                else:
                    pkt_atoms, pkt_coords = extract_pocket_from_pdb(
                        rec_val.payload, center, box_size, box_padding)
                pocket_emb = model.encode_pocket(
                    pkt_atoms, pkt_coords, max_atoms=max_pocket_atoms)
            except Exception as e:
                return False, f'Pocket extraction failed: {e}'
            self._pocket_emb_cache = pocket_emb
            self._pocket_cache_key = cache_key
        else:
            pocket_emb = self._pocket_emb_cache

        self.set_progress(15)

        # ── 6. Prepare molecules ─────────────────────────────────────────
        df = mt_val.payload.copy()
        mol_col = mt_val.mol_col
        n = len(df)
        num_confs = int(self.get_property('num_confs') or 5)

        if 'name' in df.columns:
            names = df['name'].astype(str).tolist()
        elif 'smiles' in df.columns:
            names = df['smiles'].astype(str).tolist()
        else:
            names = [f'mol_{i+1}' for i in range(n)]

        progress_rows = [{'name': nm, 'status': 'Pending', 'score': None}
                         for nm in names]
        self.set_display(list(progress_rows))

        # ── 7. Encode molecules ──────────────────────────────────────────
        embeddings = [None] * n
        workers = int(self.get_property('screen_workers') or 4)

        def _encode_one(idx):
            mol = df[mol_col].iloc[idx]
            result = mol_to_atoms_coords(mol, num_confs=num_confs)
            if result is None:
                return idx, None
            atoms, coords = result
            return idx, model.encode_molecule(atoms, coords)

        done = 0
        if workers > 1 and n > 1:
            with ThreadPoolExecutor(max_workers=min(workers, n)) as pool:
                futures = {pool.submit(_encode_one, i): i for i in range(n)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        _, emb = fut.result()
                        embeddings[idx] = emb
                        status = 'Done' if emb is not None else 'Failed'
                    except Exception:
                        status = 'Failed'
                    progress_rows[idx] = {
                        'name': names[idx], 'status': status, 'score': None}
                    done += 1
                    self.set_display(list(progress_rows))
                    self.set_progress(int(15 + done / n * 75))
        else:
            for i in range(n):
                try:
                    _, emb = _encode_one(i)
                    embeddings[i] = emb
                    status = 'Done' if emb is not None else 'Failed'
                except Exception:
                    status = 'Failed'
                progress_rows[i] = {
                    'name': names[i], 'status': status, 'score': None}
                done += 1
                self.set_display(list(progress_rows))
                self.set_progress(int(15 + done / n * 75))

        # ── 8. Score ─────────────────────────────────────────────────────
        scores = np.full(n, np.nan, dtype=np.float32)
        valid_mask = []
        valid_embs = []
        for i, emb in enumerate(embeddings):
            if emb is not None:
                valid_mask.append(i)
                valid_embs.append(emb)
        if valid_embs:
            mol_embs = np.stack(valid_embs)
            sim = model.score(pocket_emb, mol_embs)
            for j, idx in enumerate(valid_mask):
                scores[idx] = float(sim[j])

        # Update progress with scores
        for i in range(n):
            sc = None if np.isnan(scores[i]) else round(float(scores[i]), 4)
            progress_rows[i]['score'] = sc
        self.set_display(list(progress_rows))

        # ── 9. Build outputs ─────────────────────────────────────────────
        df['drugclip_score'] = scores
        df = df.sort_values('drugclip_score', ascending=False, na_position='last')
        df = df.reset_index(drop=True)

        self.output_values['mol_table'] = MolTableData(
            payload=df, mol_col=mol_col)

        # Summary table
        summary_cols = ['name', 'smiles', 'drugclip_score']
        avail = [c for c in summary_cols if c in df.columns]
        if not avail:
            avail = ['drugclip_score']
        self.output_values['scores'] = TableData(payload=df[avail].copy())

        n_valid = int((~np.isnan(scores)).sum())
        top = float(np.nanmax(scores)) if n_valid > 0 else 0.0
        self.mark_clean()
        self.set_progress(100)
        return True, (f'Scored {n_valid}/{n} molecules, '
                      f'top similarity={top:.4f}')
