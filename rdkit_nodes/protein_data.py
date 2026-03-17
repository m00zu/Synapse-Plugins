"""
Protein & docking data types for the rdkit_nodes plugin.
"""
from __future__ import annotations

from typing import Any, List, Optional

from data_models import NodeData


class ProteinData(NodeData):
    """Raw PDB/CIF string with metadata."""
    payload: str            # PDB-format string
    name: str = ''
    format: str = 'pdb'     # 'pdb' or 'cif'


class ReceptorData(NodeData):
    """Prepared receptor PDBQT string ready for docking."""
    payload: str            # PDBQT-format string
    name: str = ''
    flex_pdbqt: str = ''    # flexible residue PDBQT block (if any)
    flex_residues: List[str] = []


class DockingResultData(NodeData):
    """Docking output: poses + energies."""
    payload: str            # multi-pose PDBQT string
    energies: List[Any] = []   # [(affinity, dist1, dist2), ...]
    receptor_name: str = ''
    ligand_name: str = ''
    n_poses: int = 0
