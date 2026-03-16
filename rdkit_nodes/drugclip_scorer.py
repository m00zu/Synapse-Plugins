"""
drugclip_scorer — Standalone DrugCLIP inference via ONNX Runtime.

Dependencies: onnxruntime, numpy, scipy, rdkit.
No torch or unicore required at runtime.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
#  Atom dictionaries (matches DrugCLIP training — includes [MASK] token)
# ══════════════════════════════════════════════════════════════════════════

_MOL_SYMBOLS = [
    "[PAD]", "[CLS]", "[SEP]", "[UNK]",
    "C", "N", "O", "S", "H", "Cl", "F", "Br", "I", "Si", "P", "B",
    "Na", "K", "Al", "Ca", "Sn", "As", "Hg", "Fe", "Zn", "Cr", "Se",
    "Gd", "Au", "Li", "[MASK]",
]
MOL_DICT = {s: i for i, s in enumerate(_MOL_SYMBOLS)}
MOL_NUM_TYPES = len(_MOL_SYMBOLS)  # 31
MOL_PAD = 0
MOL_CLS = 1
MOL_SEP = 2
MOL_UNK = 3

_PKT_SYMBOLS = [
    "[PAD]", "[CLS]", "[SEP]", "[UNK]",
    "C", "N", "O", "S", "H", "[MASK]",
]
PKT_DICT = {s: i for i, s in enumerate(_PKT_SYMBOLS)}
PKT_NUM_TYPES = len(_PKT_SYMBOLS)  # 10
PKT_PAD = 0
PKT_CLS = 1
PKT_SEP = 2
PKT_UNK = 3


# ══════════════════════════════════════════════════════════════════════════
#  Preprocessing helpers
# ══════════════════════════════════════════════════════════════════════════

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def preprocess_molecule(
    atom_symbols: np.ndarray,
    coordinates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess a single molecule for DrugCLIP mol encoder.

    Args:
        atom_symbols: (N,) str array of element symbols
        coordinates:  (N, 3) float array

    Returns:
        tokens (1, L) int64, distances (1, L, L) float32,
        edge_types (1, L, L) int64  — padded to multiple of 8.
    """
    from scipy.spatial import distance_matrix

    # Remove hydrogens
    mask = atom_symbols != "H"
    atoms = atom_symbols[mask]
    coords = coordinates[mask].astype(np.float32)

    # Normalize (center at origin)
    coords = coords - coords.mean(axis=0)

    # Tokenize: [CLS] atom1 atom2 ... atomN [SEP]
    tok = [MOL_CLS]
    for a in atoms:
        tok.append(MOL_DICT.get(a, MOL_UNK))
    tok.append(MOL_SEP)
    tokens = np.array(tok, dtype=np.int64)

    # Distance matrix (atoms only, then pad for CLS/SEP)
    n = len(coords)
    raw_dist = distance_matrix(coords, coords).astype(np.float32)
    padded_dist = np.zeros((n + 2, n + 2), dtype=np.float32)
    padded_dist[1 : n + 1, 1 : n + 1] = raw_dist

    # Edge types
    edge_types = tokens[:, None] * MOL_NUM_TYPES + tokens[None, :]

    # Pad to multiple of 8
    seq_len = len(tokens)
    pad_to = ((seq_len + 7) // 8) * 8
    if pad_to > seq_len:
        p = pad_to - seq_len
        tokens = np.pad(tokens, (0, p), constant_values=MOL_PAD)
        padded_dist = np.pad(padded_dist, ((0, p), (0, p)), constant_values=0)
        edge_types = np.pad(edge_types, ((0, p), (0, p)), constant_values=0)

    return (
        tokens[None, :],
        padded_dist[None, :, :],
        edge_types[None, :, :].astype(np.int64),
    )


def preprocess_pocket(
    atom_symbols: np.ndarray,
    coordinates: np.ndarray,
    max_atoms: int = 256,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess pocket atoms for DrugCLIP pocket encoder.

    Same pipeline as molecule but with PKT_DICT and optional cropping.
    """
    from scipy.spatial import distance_matrix

    # Remove hydrogens
    mask = atom_symbols != "H"
    atoms = atom_symbols[mask]
    coords = coordinates[mask].astype(np.float32)

    # Crop to max_atoms (distance-weighted softmax sampling, keep inner atoms)
    if len(atoms) > max_atoms:
        rng = np.random.RandomState(seed)
        center = coords.mean(axis=0)
        dists_to_center = np.linalg.norm(coords - center, axis=1)
        weights = _softmax(1.0 / (dists_to_center + 1.0))
        idx = rng.choice(len(atoms), max_atoms, replace=False, p=weights)
        atoms = atoms[idx]
        coords = coords[idx]

    # Normalize
    coords = coords - coords.mean(axis=0)

    # Tokenize
    tok = [PKT_CLS]
    for a in atoms:
        tok.append(PKT_DICT.get(a, PKT_UNK))
    tok.append(PKT_SEP)
    tokens = np.array(tok, dtype=np.int64)

    # Distance matrix
    n = len(coords)
    raw_dist = distance_matrix(coords, coords).astype(np.float32)
    padded_dist = np.zeros((n + 2, n + 2), dtype=np.float32)
    padded_dist[1 : n + 1, 1 : n + 1] = raw_dist

    # Edge types
    edge_types = tokens[:, None] * PKT_NUM_TYPES + tokens[None, :]

    # Pad to multiple of 8
    seq_len = len(tokens)
    pad_to = ((seq_len + 7) // 8) * 8
    if pad_to > seq_len:
        p = pad_to - seq_len
        tokens = np.pad(tokens, (0, p), constant_values=PKT_PAD)
        padded_dist = np.pad(padded_dist, ((0, p), (0, p)), constant_values=0)
        edge_types = np.pad(edge_types, ((0, p), (0, p)), constant_values=0)

    return (
        tokens[None, :],
        padded_dist[None, :, :],
        edge_types[None, :, :].astype(np.int64),
    )


# ══════════════════════════════════════════════════════════════════════════
#  Pocket extraction from PDB
# ══════════════════════════════════════════════════════════════════════════

def _pocket_atom_element(atom_name: str) -> str:
    """Map PDB atom name to element symbol (DrugCLIP convention).

    If the first character is a digit (e.g. '1HB'), skip it.
    """
    name = atom_name.strip()
    if not name:
        return "X"
    if name[0].isdigit():
        return name[1] if len(name) > 1 else "X"
    return name[0]


def _in_box(
    pos: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> bool:
    """Check if a point lies within the axis-aligned box [box_min, box_max]."""
    return bool(np.all(pos >= box_min) and np.all(pos <= box_max))


def extract_pocket_from_pdb(
    pdb_string: str,
    center: np.ndarray,
    size: np.ndarray,
    padding: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract pocket atoms from a PDB string within the docking box + padding.

    Args:
        center: (3,) box center.
        size:   (3,) box dimensions (full widths, same as DockingBoxNode).
        padding: extra angstroms added to each side of the box.

    Returns (atom_symbols, coordinates) — element symbols suitable for PKT_DICT.
    Only considers protein ATOM records (not HETATM).
    """
    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(size, dtype=np.float64) / 2.0 + padding
    box_min = center - half
    box_max = center + half
    atoms = []
    coords = []
    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue
        pos = np.array([x, y, z], dtype=np.float64)
        if not _in_box(pos, box_min, box_max):
            continue
        # Element symbol: use columns 76-78 if available, else atom name
        elem = line[76:78].strip() if len(line) >= 78 else ""
        if not elem:
            elem = _pocket_atom_element(line[12:16])
        atoms.append(elem)
        coords.append(pos)

    if not atoms:
        raise ValueError(
            f"No protein atoms found within box "
            f"(center={center}, size={size}, padding={padding})"
        )
    return np.array(atoms), np.array(coords, dtype=np.float32)


def extract_pocket_from_pdbqt(
    pdbqt_string: str,
    center: np.ndarray,
    size: np.ndarray,
    padding: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract pocket atoms from a PDBQT string within the docking box + padding."""
    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(size, dtype=np.float64) / 2.0 + padding
    box_min = center - half
    box_max = center + half
    atoms = []
    coords = []
    for line in pdbqt_string.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue
        pos = np.array([x, y, z], dtype=np.float64)
        if not _in_box(pos, box_min, box_max):
            continue
        # PDBQT element is in column 77-78 or parse from atom type (col 77+)
        elem = line[76:78].strip() if len(line) >= 78 else ""
        if not elem:
            elem = _pocket_atom_element(line[12:16])
        atoms.append(elem)
        coords.append(pos)

    if not atoms:
        raise ValueError(
            f"No atoms found within box "
            f"(center={center}, size={size}, padding={padding})"
        )
    return np.array(atoms), np.array(coords, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════
#  Molecule → atoms + coords (from RDKit Mol)
# ══════════════════════════════════════════════════════════════════════════

def mol_to_atoms_coords(
    mol,
    conf_id: int = -1,
    num_confs: int = 5,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract atom symbols and 3D coordinates from an RDKit Mol.

    If no conformer exists, generates one with ETKDGv3 + MMFF.
    Returns None on failure.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if mol is None:
        return None

    # Use existing conformer if available
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer(conf_id)
        symbols = np.array([a.GetSymbol() for a in mol.GetAtoms()])
        coords = np.array(conf.GetPositions(), dtype=np.float32)
        return symbols, coords

    # Generate conformer
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.maxIterations = 500
    params.pruneRmsThresh = 0.5
    params.randomSeed = 42
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
    if mol.GetNumConformers() == 0:
        return None

    try:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
        # Pick lowest energy conformer
        best = min(
            range(len(results)),
            key=lambda i: results[i][1] if results[i][0] == 0 else float("inf"),
        )
    except Exception:
        best = 0

    mol = Chem.RemoveHs(mol)
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer(best if best < mol.GetNumConformers() else 0)
    symbols = np.array([a.GetSymbol() for a in mol.GetAtoms()])
    coords = np.array(conf.GetPositions(), dtype=np.float32)
    return symbols, coords


# ══════════════════════════════════════════════════════════════════════════
#  DrugCLIP ONNX Model
# ══════════════════════════════════════════════════════════════════════════

class DrugCLIPModel:
    """DrugCLIP dual-encoder using ONNX Runtime.

    No torch/unicore dependency.  Loads ``drugclip_mol.onnx`` and
    ``drugclip_pocket.onnx`` plus a scalar ``logit_scale.npy``.
    """

    DEFAULT_DIR = Path(__file__).parent / "data" / "drugclip"

    def __init__(
        self,
        mol_onnx: Optional[str] = None,
        pocket_onnx: Optional[str] = None,
        logit_scale_path: Optional[str] = None,
        num_threads: int = 4,
    ):
        import onnxruntime as ort

        d = self.DEFAULT_DIR
        mol_path = mol_onnx or str(d / "drugclip_mol.onnx")
        pkt_path = pocket_onnx or str(d / "drugclip_pocket.onnx")
        scale_path = logit_scale_path or str(d / "logit_scale.npy")

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = num_threads

        self._mol_sess = ort.InferenceSession(mol_path, sess_options=opts)
        self._pkt_sess = ort.InferenceSession(pkt_path, sess_options=opts)
        self.logit_scale = float(np.load(scale_path).flat[0])
        log.info(
            "DrugCLIP loaded: mol=%s, pocket=%s, logit_scale=%.2f",
            mol_path, pkt_path, self.logit_scale,
        )

    # ── Encoding ─────────────────────────────────────────────────────────

    def encode_pocket(
        self,
        atom_symbols: np.ndarray,
        coordinates: np.ndarray,
        max_atoms: int = 256,
    ) -> np.ndarray:
        """Encode pocket → (128,) L2-normalised embedding."""
        tokens, dists, etypes = preprocess_pocket(
            atom_symbols, coordinates, max_atoms=max_atoms
        )
        emb = self._pkt_sess.run(
            ["embedding"],
            {
                "src_tokens": tokens,
                "src_distance": dists,
                "src_edge_type": etypes,
            },
        )[0]
        return emb[0]  # (128,)

    def encode_molecule(
        self,
        atom_symbols: np.ndarray,
        coordinates: np.ndarray,
    ) -> np.ndarray:
        """Encode one molecule → (128,) L2-normalised embedding."""
        tokens, dists, etypes = preprocess_molecule(atom_symbols, coordinates)
        emb = self._mol_sess.run(
            ["embedding"],
            {
                "src_tokens": tokens,
                "src_distance": dists,
                "src_edge_type": etypes,
            },
        )[0]
        return emb[0]  # (128,)

    def encode_molecules(
        self,
        mol_data: List[Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Encode multiple molecules → (N, 128)."""
        embeddings = []
        for atoms, coords in mol_data:
            embeddings.append(self.encode_molecule(atoms, coords))
        return np.stack(embeddings) if embeddings else np.empty((0, 128))

    # ── Scoring ──────────────────────────────────────────────────────────

    @staticmethod
    def score(
        pocket_emb: np.ndarray,
        mol_embs: np.ndarray,
    ) -> np.ndarray:
        """Cosine similarity between pocket and molecules.

        Both inputs are L2-normalised, so dot product = cosine similarity.

        Returns (N,) float array in [-1, 1].
        """
        return mol_embs @ pocket_emb  # (N,)


# ══════════════════════════════════════════════════════════════════════════
#  Quick self-test
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from rdkit import Chem

    model = DrugCLIPModel()

    # Test molecule encoding
    smi = "c1ccc(CC(=O)O)cc1"  # phenylacetic acid
    mol = Chem.MolFromSmiles(smi)
    result = mol_to_atoms_coords(mol)
    if result is not None:
        atoms, coords = result
        emb = model.encode_molecule(atoms, coords)
        print(f"Molecule: {smi}")
        print(f"  embedding shape: {emb.shape}, norm: {np.linalg.norm(emb):.6f}")
        print(f"  first 5 values:  {emb[:5]}")
    else:
        print("Failed to generate conformer")
