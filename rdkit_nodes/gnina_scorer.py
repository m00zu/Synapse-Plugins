"""
GNINA CNN Scoring — Pure Python/NumPy reimplementation for CPU inference.

Loads pre-trained GNINA models (ONNX format) and scores protein-ligand poses
using Gaussian density voxelization + 3D CNN inference.

No CUDA, no libmolgrid, no OpenBabel, no PyTorch required.
Dependencies: onnxruntime, numpy, rdkit
"""

import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

# ─── SMINA Atom Types ────────────────────────────────────────────────────────

SMINA_TYPES = [
    "Hydrogen",                     # 0
    "PolarHydrogen",                # 1
    "AliphaticCarbonXSHydrophobe",  # 2
    "AliphaticCarbonXSNonHydrophobe",  # 3
    "AromaticCarbonXSHydrophobe",   # 4
    "AromaticCarbonXSNonHydrophobe",  # 5
    "Nitrogen",                     # 6
    "NitrogenXSDonor",              # 7
    "NitrogenXSDonorAcceptor",      # 8
    "NitrogenXSAcceptor",           # 9
    "Oxygen",                       # 10
    "OxygenXSDonor",                # 11
    "OxygenXSDonorAcceptor",        # 12
    "OxygenXSAcceptor",             # 13
    "Sulfur",                       # 14
    "SulfurAcceptor",               # 15
    "Phosphorus",                   # 16
    "Fluorine",                     # 17
    "Chlorine",                     # 18
    "Bromine",                      # 19
    "Iodine",                       # 20
    "Magnesium",                    # 21
    "Manganese",                    # 22
    "Zinc",                         # 23
    "Calcium",                      # 24
    "Iron",                         # 25
    "GenericMetal",                  # 26
    "Boron",                        # 27
]

# XS radii (Angstroms) — used for Gaussian density voxelization
XS_RADII = {
    "Hydrogen": 0.37,
    "PolarHydrogen": 0.37,
    "AliphaticCarbonXSHydrophobe": 1.90,
    "AliphaticCarbonXSNonHydrophobe": 1.90,
    "AromaticCarbonXSHydrophobe": 1.90,
    "AromaticCarbonXSNonHydrophobe": 1.90,
    "Nitrogen": 1.80,
    "NitrogenXSDonor": 1.80,
    "NitrogenXSDonorAcceptor": 1.80,
    "NitrogenXSAcceptor": 1.80,
    "Oxygen": 1.70,
    "OxygenXSDonor": 1.70,
    "OxygenXSDonorAcceptor": 1.70,
    "OxygenXSAcceptor": 1.70,
    "Sulfur": 2.00,
    "SulfurAcceptor": 2.00,
    "Phosphorus": 2.10,
    "Fluorine": 1.50,
    "Chlorine": 1.80,
    "Bromine": 2.00,
    "Iodine": 2.20,
    "Magnesium": 1.20,
    "Manganese": 1.20,
    "Zinc": 1.20,
    "Calcium": 1.20,
    "Iron": 1.20,
    "GenericMetal": 1.20,
    "Boron": 1.92,
}

# Element Z → SMINA type name mapping (for elements not C/N/O/S/H)
_ELEMENT_TO_SMINA = {
    9: "Fluorine", 17: "Chlorine", 35: "Bromine", 53: "Iodine",
    15: "Phosphorus", 5: "Boron",
    12: "Magnesium", 25: "Manganese", 30: "Zinc",
    20: "Calcium", 26: "Iron",
}

# Metals that map to GenericMetal
_METAL_Z = {3, 4, 11, 13, 19, 22, 23, 24, 27, 28, 29, 31, 33, 34, 37, 38, 39,
             40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57,
             72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83}


def assign_smina_type(atom):
    """Assign SMINA atom type to an RDKit atom. Returns type name string."""
    z = atom.GetAtomicNum()

    if z == 1:
        # All hydrogens → PolarHydrogen initially, then adjust
        # (GNINA maps all H to "HD" which becomes PolarHydrogen)
        return "PolarHydrogen"

    # Check neighbor properties for adjustment
    has_h_neighbor = False
    has_hetero_neighbor = False
    for nbr in atom.GetNeighbors():
        nz = nbr.GetAtomicNum()
        if nz == 1:
            has_h_neighbor = True
        elif nz != 6:
            has_hetero_neighbor = True

    if z == 6:
        # Carbon: aromatic vs aliphatic, hydrophobe vs non-hydrophobe
        if atom.GetIsAromatic():
            return "AromaticCarbonXSNonHydrophobe" if has_hetero_neighbor else "AromaticCarbonXSHydrophobe"
        else:
            return "AliphaticCarbonXSNonHydrophobe" if has_hetero_neighbor else "AliphaticCarbonXSHydrophobe"

    if z == 7:
        # Nitrogen: check H-bond acceptor status
        # GNINA: if OB says acceptor → NA → NitrogenXSAcceptor/DonorAcceptor
        #         else → N → Nitrogen/NitrogenXSDonor
        # In RDKit, nitrogen is acceptor if it has a lone pair (most N except quaternary N+)
        is_acceptor = _is_nitrogen_acceptor(atom)
        if is_acceptor:
            return "NitrogenXSDonorAcceptor" if has_h_neighbor else "NitrogenXSAcceptor"
        else:
            return "NitrogenXSDonor" if has_h_neighbor else "Nitrogen"

    if z == 8:
        # Oxygen: always starts as OA (acceptor), then adjust for donor
        if has_h_neighbor:
            return "OxygenXSDonorAcceptor"
        else:
            return "OxygenXSAcceptor"

    if z == 16:
        # Sulfur: check acceptor
        is_acceptor = _is_sulfur_acceptor(atom)
        return "SulfurAcceptor" if is_acceptor else "Sulfur"

    # Other elements
    if z in _ELEMENT_TO_SMINA:
        return _ELEMENT_TO_SMINA[z]

    if z in _METAL_Z:
        return "GenericMetal"

    # Fallback
    return "GenericMetal"


def _is_nitrogen_acceptor(atom):
    """Check if nitrogen is H-bond acceptor (has lone pair available).
    Matches OpenBabel's IsHbondAcceptor for nitrogen."""
    # Quaternary nitrogen (formal charge +1, 4 bonds) is not acceptor
    if atom.GetFormalCharge() >= 1 and atom.GetDegree() >= 4:
        return False
    # Amide N bonded to C=O is typically a weak acceptor but OB counts it
    # Simple heuristic: N is acceptor unless it's sp3 with 4 substituents
    return True


def _is_sulfur_acceptor(atom):
    """Check if sulfur is H-bond acceptor."""
    # Thiol/thioether sulfur with lone pairs
    # OB considers S as acceptor if it has available lone pairs
    if atom.GetDegree() <= 2:
        return True
    return False


# ─── Channel Mapping ─────────────────────────────────────────────────────────

# Default maps from GNINA (each line = one channel, space-separated aliases)
DEFAULT_RECMAP = """AliphaticCarbonXSHydrophobe
AliphaticCarbonXSNonHydrophobe
AromaticCarbonXSHydrophobe
AromaticCarbonXSNonHydrophobe
Bromine Iodine Chlorine Fluorine
Nitrogen NitrogenXSAcceptor
NitrogenXSDonor NitrogenXSDonorAcceptor
Oxygen OxygenXSAcceptor
OxygenXSDonorAcceptor OxygenXSDonor
Sulfur SulfurAcceptor
Phosphorus
Calcium
Zinc
GenericMetal Boron Manganese Magnesium Iron"""

DEFAULT_LIGMAP = """AliphaticCarbonXSHydrophobe
AliphaticCarbonXSNonHydrophobe
AromaticCarbonXSHydrophobe
AromaticCarbonXSNonHydrophobe
Bromine Iodine
Chlorine
Fluorine
Nitrogen NitrogenXSAcceptor
NitrogenXSDonor NitrogenXSDonorAcceptor
Oxygen OxygenXSAcceptor
OxygenXSDonorAcceptor OxygenXSDonor
Sulfur SulfurAcceptor
Phosphorus
GenericMetal Boron Manganese Magnesium Zinc Calcium Iron"""


def parse_channel_map(mapstr):
    """Parse a GNINA channel map string.
    Returns: (type_to_channel dict, channel_to_radius dict, num_channels)
    Each line = one channel; space-separated names on a line are aliases.
    """
    type_to_channel = {}
    channel_radii = {}
    lines = [l.strip() for l in mapstr.strip().split('\n') if l.strip()]
    for ch_idx, line in enumerate(lines):
        names = line.split()
        # Use the first atom type's radius as the channel radius
        first_radius = XS_RADII.get(names[0], 1.8)
        channel_radii[ch_idx] = first_radius
        for name in names:
            type_to_channel[name] = ch_idx
    return type_to_channel, channel_radii, len(lines)


# ─── Gaussian Density Voxelization ───────────────────────────────────────────

# Precomputed density coefficients for the quadratic tail region.
# Matches libmolgrid's GridMaker::initialize() with gaussian_radius_multiple=1.0.
# The density function has two regions:
#   Gaussian:  dist <= radius * G        → exp(-2 * dist² / radius²)
#   Quadratic: radius * G < dist <= radius * F → A*(dist/r)² + B*(dist/r) + C
# where G = gaussian_radius_multiple = 1.0, F = final_radius_multiple = 1.5
_DENSITY_G = 1.0   # gaussian_radius_multiple (default)
_DENSITY_F = (1 + 2 * _DENSITY_G**2) / (2 * _DENSITY_G)  # 1.5
_DENSITY_A = np.exp(-2 * _DENSITY_G**2) * 4 * _DENSITY_G**2
_DENSITY_B = -np.exp(-2 * _DENSITY_G**2) * (4 * _DENSITY_G + 8 * _DENSITY_G**3)
_DENSITY_C = np.exp(-2 * _DENSITY_G**2) * (4 * _DENSITY_G**4 + 4 * _DENSITY_G**2 + 1)


def _calc_density(dist, radius):
    """Compute atom density at distance `dist` from atom center.

    Matches libmolgrid's calc_point<false>():
      - Gaussian for dist <= radius * G (G=1.0)
      - Quadratic tail for radius * G < dist <= radius * F (F=1.5)
      - Zero beyond radius * F
    """
    ar = radius  # already includes radius_scale from caller
    cutoff = ar * _DENSITY_F
    if dist >= cutoff:
        return 0.0
    if dist <= ar * _DENSITY_G:
        return np.exp(-2.0 * dist * dist / (ar * ar))
    # Quadratic tail
    dr = dist / ar
    q = (_DENSITY_A * dr + _DENSITY_B) * dr + _DENSITY_C
    return max(q, 0.0)


def _calc_density_vectorized(r, radius):
    """Vectorized density matching libmolgrid's calc_point<false>().

    Args:
        r: array of distances
        radius: scalar atomic radius (already includes radius_scale)
    Returns:
        array of density values
    """
    ar = radius
    cutoff = ar * _DENSITY_F
    gauss_cutoff = ar * _DENSITY_G

    dr = r / ar
    # Gaussian region
    gauss = np.exp(-2.0 * dr * dr)
    # Quadratic tail region
    quad = (_DENSITY_A * dr + _DENSITY_B) * dr + _DENSITY_C
    quad = np.maximum(quad, 0.0)

    density = np.where(r <= gauss_cutoff, gauss,
              np.where(r <= cutoff, quad, 0.0))
    return density


def voxelize(coords, smina_types, type_to_channel, num_channels,
             center, resolution=0.5, dimension=23.5, radius_scale=1.0,
             radius_multiple=1.5):
    """Create 3D Gaussian density grid from atom coordinates and types.

    Args:
        coords: (N, 3) array of atom coordinates
        smina_types: list of N SMINA type name strings
        type_to_channel: dict mapping type name → channel index
        num_channels: total number of channels
        center: (3,) grid center coordinates
        resolution: grid spacing in Angstroms
        dimension: grid edge length in Angstroms
        radius_scale: scale factor for VDW radii
        radius_multiple: cutoff at radius * radius_multiple (used for bounds)

    Returns:
        (num_channels, gd, gd, gd) numpy array
    """
    gd = 1 + round(dimension / resolution)  # 48 for default params
    grid = np.zeros((num_channels, gd, gd, gd), dtype=np.float32)

    # Grid origin (corner)
    origin = np.array(center) - dimension / 2.0

    for i in range(len(coords)):
        stype = smina_types[i]
        if stype not in type_to_channel:
            continue  # skip unmapped types (e.g. hydrogen)
        ch = type_to_channel[stype]
        radius = XS_RADII.get(stype, 1.8) * radius_scale
        cutoff = radius * _DENSITY_F  # match libmolgrid final_radius_multiple

        # Atom position in grid coordinates
        atom_pos = coords[i]
        rel = (atom_pos - origin) / resolution

        # Bounding box of affected voxels
        cutoff_voxels = int(np.ceil(cutoff / resolution))
        i0 = max(0, int(np.floor(rel[0])) - cutoff_voxels)
        i1 = min(gd, int(np.ceil(rel[0])) + cutoff_voxels + 1)
        j0 = max(0, int(np.floor(rel[1])) - cutoff_voxels)
        j1 = min(gd, int(np.ceil(rel[1])) + cutoff_voxels + 1)
        k0 = max(0, int(np.floor(rel[2])) - cutoff_voxels)
        k1 = min(gd, int(np.ceil(rel[2])) + cutoff_voxels + 1)

        # Fill voxels with density (Gaussian + quadratic tail)
        for vi in range(i0, i1):
            dx = (vi * resolution + origin[0]) - atom_pos[0]
            dx2 = dx * dx
            for vj in range(j0, j1):
                dy = (vj * resolution + origin[1]) - atom_pos[1]
                dy2 = dy * dy
                for vk in range(k0, k1):
                    dz = (vk * resolution + origin[2]) - atom_pos[2]
                    dist = np.sqrt(dx2 + dy2 + dz * dz)
                    density = _calc_density(dist, radius)
                    if density > 0:
                        grid[ch, vi, vj, vk] += density

    return grid


def voxelize_fast(coords, smina_types, type_to_channel, num_channels,
                  center, resolution=0.5, dimension=23.5, radius_scale=1.0,
                  radius_multiple=1.5):
    """Vectorized voxelization — much faster than the triple-loop version.
    Uses NumPy broadcasting to compute all voxel distances at once per atom.
    Density function matches libmolgrid: Gaussian core + quadratic tail.
    """
    gd = 1 + round(dimension / resolution)  # 48
    grid = np.zeros((num_channels, gd, gd, gd), dtype=np.float32)
    origin = np.array(center, dtype=np.float64) - dimension / 2.0

    # Precompute voxel center positions along each axis
    ax = origin[0] + np.arange(gd) * resolution
    ay = origin[1] + np.arange(gd) * resolution
    az = origin[2] + np.arange(gd) * resolution

    for i in range(len(coords)):
        stype = smina_types[i]
        if stype not in type_to_channel:
            continue
        ch = type_to_channel[stype]
        radius = XS_RADII.get(stype, 1.8) * radius_scale
        cutoff = radius * _DENSITY_F  # match libmolgrid final_radius_multiple

        atom_pos = coords[i]

        # Find affected voxel range per axis
        i0 = max(0, int(np.searchsorted(ax, atom_pos[0] - cutoff)) - 1)
        i1 = min(gd, int(np.searchsorted(ax, atom_pos[0] + cutoff)) + 1)
        j0 = max(0, int(np.searchsorted(ay, atom_pos[1] - cutoff)) - 1)
        j1 = min(gd, int(np.searchsorted(ay, atom_pos[1] + cutoff)) + 1)
        k0 = max(0, int(np.searchsorted(az, atom_pos[2] - cutoff)) - 1)
        k1 = min(gd, int(np.searchsorted(az, atom_pos[2] + cutoff)) + 1)

        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            continue

        # Vectorized distance computation
        dx = ax[i0:i1] - atom_pos[0]
        dy = ay[j0:j1] - atom_pos[1]
        dz = az[k0:k1] - atom_pos[2]

        # 3D distance grid via broadcasting
        r2 = dx[:, None, None]**2 + dy[None, :, None]**2 + dz[None, None, :]**2
        r = np.sqrt(r2)

        # Two-region density: Gaussian core + quadratic tail
        density = _calc_density_vectorized(r, radius)
        grid[ch, i0:i1, j0:j1, k0:k1] += density.astype(np.float32)

    return grid


# ─── Molecule Processing ─────────────────────────────────────────────────────

def process_molecule(mol, remove_h=False, conf_id=-1):
    """Extract coordinates and SMINA types from an RDKit molecule.

    Args:
        mol: RDKit Mol object with 3D coordinates
        remove_h: if True, skip hydrogen atoms from returned coords/types
        conf_id: conformer index to use (-1 = default/first)

    Returns:
        coords: (N, 3) numpy array (heavy atoms only if remove_h)
        smina_types: list of N SMINA type name strings
    """
    conf = mol.GetConformer(conf_id)
    coords = []
    smina_types = []

    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if remove_h and z == 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
        smina_types.append(assign_smina_type(atom))

    return np.array(coords, dtype=np.float64), smina_types


def _mol_centroid(mol, conf_id=-1):
    """Compute centroid from ALL atoms (including H), matching gnina/libmolgrid."""
    conf = mol.GetConformer(conf_id)
    coords = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
    return np.mean(coords, axis=0)


def read_pdb(path, sanitize=True, remove_h=False):
    """Read PDB file and return (mol, coords, smina_types).

    When remove_h=True, hydrogens are kept in the RDKit molecule for
    neighbor analysis but excluded from the returned coordinates and types.
    """
    # Always read with H present so neighbor analysis works correctly
    mol = Chem.MolFromPDBFile(str(path), sanitize=sanitize, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to read PDB: {path}")
    coords, types = process_molecule(mol, remove_h=remove_h)
    return mol, coords, types


def read_sdf(path, remove_h=False):
    """Read first molecule from SDF file and return (mol, coords, smina_types).

    When remove_h=True, hydrogens are kept in the RDKit molecule for
    neighbor analysis (donor/acceptor typing) but excluded from the
    returned coordinates and types.
    """
    # Always read with H present so neighbor analysis works correctly
    suppl = Chem.SDMolSupplier(str(path), removeHs=False)
    mol = next(iter(suppl))
    if mol is None:
        raise ValueError(f"Failed to read SDF: {path}")
    coords, types = process_molecule(mol, remove_h=remove_h)
    return mol, coords, types


def read_sdf_all(path, remove_h=False):
    """Read all molecules from SDF file.

    Returns list of (mol, coords, smina_types, centroid) tuples.
    centroid is computed from ALL atoms (including H) to match gnina/libmolgrid.
    """
    # Always read with H present so neighbor analysis works correctly
    suppl = Chem.SDMolSupplier(str(path), removeHs=False)
    results = []
    for mol in suppl:
        if mol is None:
            continue
        centroid = _mol_centroid(mol)
        coords, types = process_molecule(mol, remove_h=remove_h)
        results.append((mol, coords, types, centroid))
    return results


# AutoDock atom type → element mapping
_AD_TYPE_TO_ELEMENT = {
    "H": "H", "HD": "H", "HS": "H",
    "C": "C", "A": "C",  # A = aromatic carbon
    "N": "N", "NA": "N", "NS": "N",
    "O": "O", "OA": "O", "OS": "O",
    "S": "S", "SA": "S",
    "P": "P",
    "F": "F", "Cl": "Cl", "Br": "Br", "I": "I",
    "Zn": "Zn", "Fe": "Fe", "Mg": "Mg", "Mn": "Mn",
    "Ca": "Ca", "Cu": "Cu", "Co": "Co", "Ni": "Ni",
    "B": "B", "Si": "Si", "Se": "Se",
}


def read_pdbqt(path, remove_h=True):
    """Read PDBQT file and return (coords, smina_types).

    Uses RDKit for molecular graph construction and hybrid SMINA typing
    (AD type for aromaticity + RDKit neighbors for XS classification).

    Returns:
        coords: (N, 3) numpy array
        smina_types: list of SMINA type name strings
    """
    with open(str(path)) as f:
        lines = f.readlines()

    # Only read first MODEL
    filtered = []
    for line in lines:
        if line.startswith("ENDMDL"):
            break
        filtered.append(line)

    coords, smina_types = _process_pdbqt_via_rdkit(filtered, remove_h)
    if not coords:
        raise ValueError(f"No atoms found in PDBQT: {path}")
    return np.array(coords, dtype=np.float64), smina_types


def _parse_pdbqt_lines(lines, remove_h=True):
    """Parse ATOM/HETATM lines from a PDBQT block, return (coords, smina_types)."""
    coords = []
    smina_types = []
    for line in lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        ad_type = line[77:79].strip() if len(line) > 77 else ""
        if not ad_type:
            ad_type = line[12:16].strip()
            elem = ""
            for ch in ad_type:
                if ch.isalpha():
                    elem += ch
                elif elem:
                    break
            ad_type = elem
        element = _AD_TYPE_TO_ELEMENT.get(ad_type, ad_type)
        z_num = _ELEMENT_SYMBOL_TO_Z.get(element, 0)
        if z_num == 0:
            continue
        if remove_h and z_num == 1:
            continue
        coords.append([x, y, z])
        smina_types.append(_ad_type_to_smina(ad_type, z_num))
    return coords, smina_types


def parse_pdbqt_string(pdbqt_str, remove_h=True):
    """Parse a single-model PDBQT string, return (coords, smina_types).

    Uses RDKit for molecular graph + hybrid SMINA typing when possible.
    """
    lines = pdbqt_str.splitlines()
    coords, types = _process_pdbqt_via_rdkit(lines, remove_h)
    if not coords:
        raise ValueError("No atoms found in PDBQT string")
    return np.array(coords, dtype=np.float64), types


def parse_pdbqt_poses(pdbqt_str, remove_h=True):
    """Parse multi-MODEL PDBQT string, return list of (coords, smina_types).

    Each MODEL/ENDMDL block is one pose. If no MODEL tags are present,
    the entire string is treated as one pose.
    """
    poses = []
    lines = pdbqt_str.splitlines()
    current_model = []
    in_model = False
    has_model_tags = any(l.startswith("MODEL") for l in lines)

    for line in lines:
        if line.startswith("MODEL"):
            in_model = True
            current_model = []
        elif line.startswith("ENDMDL"):
            if current_model:
                coords, types = _parse_pdbqt_lines(current_model, remove_h)
                if coords:
                    poses.append((np.array(coords, dtype=np.float64), types))
            current_model = []
            in_model = False
        else:
            if in_model or not has_model_tags:
                current_model.append(line)

    # Handle case with no MODEL/ENDMDL tags
    if not poses and current_model:
        coords, types = _parse_pdbqt_lines(current_model, remove_h)
        if coords:
            poses.append((np.array(coords, dtype=np.float64), types))

    return poses


def parse_pdb_string(pdb_str, remove_h=True):
    """Parse PDB-format string via RDKit, return (coords, smina_types)."""
    mol = Chem.MolFromPDBBlock(pdb_str, sanitize=True, removeHs=remove_h)
    if mol is None:
        mol = Chem.MolFromPDBBlock(pdb_str, sanitize=False, removeHs=remove_h)
    if mol is None:
        raise ValueError("Failed to parse PDB string")
    return process_molecule(mol, remove_h=remove_h)


_ELEMENT_SYMBOL_TO_Z = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
    "Cl": 17, "Br": 35, "I": 53, "B": 5,
    "Mg": 12, "Mn": 25, "Zn": 30, "Ca": 20, "Fe": 26,
    "Cu": 29, "Co": 27, "Ni": 28, "Si": 14, "Se": 34,
}


def _ad_type_to_smina(ad_type, z_num):
    """Convert AutoDock atom type string to SMINA type name (no neighbor info).
    Fallback for when RDKit parsing is unavailable.
    """
    if z_num == 1:
        return "PolarHydrogen"

    if z_num == 6:
        if ad_type == "A":
            return "AromaticCarbonXSHydrophobe"
        return "AliphaticCarbonXSHydrophobe"

    if z_num == 7:
        if ad_type in ("NA", "NS"):
            return "NitrogenXSAcceptor"
        return "Nitrogen"

    if z_num == 8:
        if ad_type in ("OA", "OS"):
            return "OxygenXSAcceptor"
        return "Oxygen"

    if z_num == 16:
        if ad_type == "SA":
            return "SulfurAcceptor"
        return "Sulfur"

    if z_num in _ELEMENT_TO_SMINA:
        return _ELEMENT_TO_SMINA[z_num]

    return "GenericMetal"


def _assign_smina_hybrid(atom, ad_type):
    """Assign SMINA type using RDKit neighbor info + PDBQT AD type for aromaticity.

    This combines the best of both worlds:
    - AD type 'A' distinguishes aromatic vs aliphatic carbon
    - AD types 'NA'/'NS'/'OA'/'OS'/'SA' encode acceptor status
    - RDKit molecular graph provides neighbor analysis for:
      - Carbon hydrophobe/nonhydrophobe (bonded to heteroatom?)
      - Nitrogen/oxygen donor (bonded to H?)
    """
    z = atom.GetAtomicNum()
    if z == 1:
        return "PolarHydrogen"

    has_h = any(n.GetAtomicNum() == 1 for n in atom.GetNeighbors())
    has_het = any(n.GetAtomicNum() not in (1, 6) for n in atom.GetNeighbors())

    if z == 6:
        if ad_type == "A":
            return "AromaticCarbonXSNonHydrophobe" if has_het else "AromaticCarbonXSHydrophobe"
        return "AliphaticCarbonXSNonHydrophobe" if has_het else "AliphaticCarbonXSHydrophobe"

    if z == 7:
        # AD type is authoritative for acceptor status in PDBQT
        is_acc = ad_type in ("NA", "NS")
        if is_acc:
            return "NitrogenXSDonorAcceptor" if has_h else "NitrogenXSAcceptor"
        return "NitrogenXSDonor" if has_h else "Nitrogen"

    if z == 8:
        # AD type "OA"/"OS" = acceptor; "O" = non-acceptor (rare in PDBQT)
        is_acc = ad_type in ("OA", "OS")
        if is_acc:
            return "OxygenXSDonorAcceptor" if has_h else "OxygenXSAcceptor"
        return "OxygenXSDonor" if has_h else "Oxygen"

    if z == 16:
        return "SulfurAcceptor" if ad_type == "SA" else "Sulfur"

    if z in _ELEMENT_TO_SMINA:
        return _ELEMENT_TO_SMINA[z]

    return "GenericMetal"


def _pdbqt_to_rdkit_mol(pdbqt_lines):
    """Convert PDBQT ATOM/HETATM lines to an RDKit Mol via PDB parsing.

    Returns (mol, serial_to_ad_type) or (None, {}) on failure.
    The mol has proximity-bonded connectivity, suitable for neighbor analysis.
    """
    _ad_to_elem = {
        "H": "H", "HD": "H", "HS": "H", "C": "C", "A": "C",
        "N": "N", "NA": "N", "NS": "N", "O": "O", "OA": "O", "OS": "O",
        "S": "S", "SA": "S", "P": "P", "F": "F", "Cl": "Cl", "Br": "Br",
        "I": "I", "Zn": "Zn", "Fe": "Fe", "Mg": "Mg", "Mn": "Mn",
        "Ca": "Ca", "Cu": "Cu", "Co": "Co", "Ni": "Ni", "B": "B",
        "Si": "Si", "Se": "Se",
    }

    serial_to_ad = {}
    pdb_lines = []
    for line in pdbqt_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            serial = int(line[6:11].strip())
            ad_type = line[77:79].strip() if len(line) > 77 else ""
            if not ad_type:
                ad_type = line[12:16].strip()
                elem = ""
                for ch in ad_type:
                    if ch.isalpha():
                        elem += ch
                    elif elem:
                        break
                ad_type = elem
            serial_to_ad[serial] = ad_type
            elem = _ad_to_elem.get(ad_type, "C")
            pdb_lines.append(line[:54].ljust(76) + elem.rjust(2))
        elif line.startswith("END"):
            pdb_lines.append("END")

    pdb_str = "\n".join(pdb_lines)
    mol = Chem.MolFromPDBBlock(
        pdb_str, sanitize=False, removeHs=False, proximityBonding=True)
    if mol is not None:
        try:
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
        except Exception:
            pass  # best-effort sanitize
    return mol, serial_to_ad


def _process_pdbqt_via_rdkit(pdbqt_lines, remove_h=True):
    """Parse PDBQT lines using RDKit for molecular graph + hybrid SMINA typing.

    Falls back to simple AD-type mapping if RDKit parsing fails.
    """
    mol, serial_to_ad = _pdbqt_to_rdkit_mol(pdbqt_lines)
    if mol is None:
        # Fallback to simple parsing
        return _parse_pdbqt_lines(pdbqt_lines, remove_h)

    conf = mol.GetConformer()
    coords = []
    smina_types = []
    for atom in mol.GetAtoms():
        if remove_h and atom.GetAtomicNum() == 1:
            continue
        info = atom.GetPDBResidueInfo()
        serial = info.GetSerialNumber() if info else -1
        ad = serial_to_ad.get(serial, "C")
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
        smina_types.append(_assign_smina_hybrid(atom, ad))

    return coords, smina_types


# ─── GNINA Model ─────────────────────────────────────────────────────────────

class GNINAModel:
    """GNINA CNN scoring model — loads ONNX models and handles voxelization."""

    DEFAULT_MODELS_DIR = Path(__file__).parent / "data" / "gnina_models"
    DEFAULT_ENSEMBLE = ["dense_1.3.onnx", "dense_1.3_PT_KD_3.onnx", "crossdock_default2018_KD_4.onnx"]

    # Named ensembles matching GNINA's --cnn options
    ENSEMBLES = {
        "default":              ["dense_1.3.onnx", "dense_1.3_PT_KD_3.onnx", "crossdock_default2018_KD_4.onnx"],
        "dense":                ["dense.onnx"] + [f"dense_{i}.onnx" for i in ["1", "2", "3", "4"]],
        "dense_1.3":            [f"dense_1.3_{i}.onnx" if i else "dense_1.3.onnx" for i in ["", "1", "2", "3", "4"]],
        "dense_1.3_PT_KD":      [f"dense_1.3_PT_KD_{i}.onnx" if i else "dense_1.3_PT_KD.onnx" for i in ["", "1", "2", "3", "4"]],
        "dense_1.3_PT_KD_def2018": [f"dense_1.3_PT_KD_def2018_{i}.onnx" if i else "dense_1.3_PT_KD_def2018.onnx" for i in ["", "1", "2", "3", "4"]],
        "crossdock_default2018":   [f"crossdock_default2018_{i}.onnx" if i else "crossdock_default2018.onnx" for i in ["", "1", "2", "3", "4"]],
        "crossdock_default2018_1.3": [f"crossdock_default2018_1.3_{i}.onnx" if i else "crossdock_default2018_1.3.onnx" for i in ["", "1", "2", "3", "4"]],
        "crossdock_default2018_KD":  [f"crossdock_default2018_KD_{i}.onnx" for i in ["1", "2", "3", "4", "5"]],
        "general_default2018":     [f"general_default2018_{i}.onnx" if i else "general_default2018.onnx" for i in ["", "1", "2", "3", "4"]],
        "general_default2018_KD":  [f"general_default2018_KD_{i}.onnx" for i in ["1", "2", "3", "4", "5"]],
        "redock_default2018":      [f"redock_default2018_{i}.onnx" if i else "redock_default2018.onnx" for i in ["", "1", "2", "3", "4"]],
        "redock_default2018_1.3":  [f"redock_default2018_1.3_{i}.onnx" if i else "redock_default2018_1.3.onnx" for i in ["", "1", "2", "3", "4"]],
        "redock_default2018_KD":   [f"redock_default2018_KD_{i}.onnx" for i in ["1", "2", "3", "4", "5"]],
    }

    # URL for auto-downloading model weights (GitHub Release or HuggingFace)
    MODELS_URL = "https://github.com/m00zu/Synapse-Plugins/releases/download/v0.1.0/gnina_models.zip"

    @classmethod
    def _ensure_models(cls):
        """Download GNINA models if not present locally."""
        if cls.DEFAULT_MODELS_DIR.exists() and any(cls.DEFAULT_MODELS_DIR.glob("*.onnx")):
            return
        import requests, zipfile, io
        print(f"GNINA models not found. Downloading...")
        resp = requests.get(cls.MODELS_URL, stream=True, allow_redirects=True)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        buf = io.BytesIO()
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            buf.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                mb = downloaded / 1024 / 1024
                total_mb = total / 1024 / 1024
                print(f"\r  Downloading GNINA models: {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)
        print()
        buf.seek(0)
        cls.DEFAULT_MODELS_DIR.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(buf) as zf:
            zf.extractall(cls.DEFAULT_MODELS_DIR.parent)
        print(f"  Extracted to {cls.DEFAULT_MODELS_DIR}")

    def __init__(self, model_paths=None, ensemble=None):
        """Load one or more GNINA ONNX models.

        Args:
            model_paths: list of paths to .onnx files
            ensemble: name of a preset ensemble (e.g. 'default', 'dense_1.3',
                      'crossdock_default2018_KD'). See GNINAModel.ENSEMBLES.
            If both are None, uses the default 3-model ensemble.
        """
        self._ensure_models()

        if model_paths is None:
            if ensemble is not None:
                if ensemble not in self.ENSEMBLES:
                    avail = ", ".join(sorted(self.ENSEMBLES.keys()))
                    raise ValueError(f"Unknown ensemble '{ensemble}'. Available: {avail}")
                model_paths = [self.DEFAULT_MODELS_DIR / n for n in self.ENSEMBLES[ensemble]]
            else:
                model_paths = [self.DEFAULT_MODELS_DIR / name for name in self.DEFAULT_ENSEMBLE]

        self.models = []

        # Suppress ONNX Runtime warnings
        opts = ort.SessionOptions()
        opts.log_severity_level = 3

        for path in model_paths:
            path = Path(path)

            # Load metadata from sidecar JSON
            json_path = path.with_suffix(".json")
            if json_path.exists():
                with open(json_path) as f:
                    meta = json.load(f)
            else:
                meta = {}

            resolution = meta.get("resolution", 0.5)
            dimension = meta.get("dimension", 23.5)
            recmap = meta.get("recmap", DEFAULT_RECMAP)
            ligmap = meta.get("ligmap", DEFAULT_LIGMAP)
            radius_scale = meta.get("radius_scaling", 1.0)
            skip_softmax = meta.get("skip_softmax", False)

            rec_t2c, _, rec_nch = parse_channel_map(recmap)
            lig_t2c, _, lig_nch = parse_channel_map(ligmap)

            session = ort.InferenceSession(str(path), sess_options=opts)

            self.models.append({
                "session": session,
                "resolution": resolution,
                "dimension": dimension,
                "radius_scale": radius_scale,
                "skip_softmax": skip_softmax,
                "rec_map": rec_t2c,
                "rec_nch": rec_nch,
                "lig_map": lig_t2c,
                "lig_nch": lig_nch,
                "name": path.stem,
            })

        print(f"Loaded {len(self.models)} GNINA model(s): "
              f"{[m['name'] for m in self.models]}")

    def _models_share_grid_params(self):
        """Check if all models use the same grid parameters (common case)."""
        if len(self.models) <= 1:
            return True
        ref = self.models[0]
        return all(
            m["resolution"] == ref["resolution"] and
            m["dimension"] == ref["dimension"] and
            m["radius_scale"] == ref["radius_scale"] and
            m["rec_map"] == ref["rec_map"] and
            m["lig_map"] == ref["lig_map"]
            for m in self.models[1:]
        )

    def _voxelize_combined(self, rec_coords, rec_types, lig_coords, lig_types,
                           center, minfo):
        """Voxelize receptor + ligand and combine into input tensor."""
        res = minfo["resolution"]
        dim = minfo["dimension"]
        rscale = minfo["radius_scale"]

        rec_grid = voxelize_fast(
            rec_coords, rec_types, minfo["rec_map"], minfo["rec_nch"],
            center, res, dim, rscale
        )
        lig_grid = voxelize_fast(
            lig_coords, lig_types, minfo["lig_map"], minfo["lig_nch"],
            center, res, dim, rscale
        )
        return np.concatenate([rec_grid, lig_grid], axis=0)

    def _infer(self, grid, minfo):
        """Run CNN inference and return (score, affinity)."""
        input_arr = grid[np.newaxis, ...].astype(np.float32)  # (1, C, 48, 48, 48)
        pose_logit, affinity = minfo["session"].run(None, {"input": input_arr})

        if minfo["skip_softmax"]:
            score = float(pose_logit[0, 1])
        else:
            # softmax over axis 1
            logit = pose_logit[0]
            exp_l = np.exp(logit - np.max(logit))
            softmax = exp_l / exp_l.sum()
            score = float(softmax[1])

        return score, float(affinity[0])

    def score(self, rec_coords, rec_types, lig_coords, lig_types, center=None):
        """Score a protein-ligand pose.

        Args:
            rec_coords: (N, 3) receptor coordinates
            rec_types: list of N SMINA type names
            lig_coords: (M, 3) ligand coordinates
            lig_types: list of M SMINA type names
            center: (3,) grid center; if None, uses ligand centroid

        Returns:
            dict with 'CNNscore', 'CNNaffinity', and per-model details
        """
        if center is None:
            center = np.mean(lig_coords, axis=0)

        scores = []
        affinities = []

        # Optimize: share voxelization across models with same grid params
        shared_grid = None
        if self._models_share_grid_params():
            shared_grid = self._voxelize_combined(
                rec_coords, rec_types, lig_coords, lig_types,
                center, self.models[0]
            )

        for minfo in self.models:
            if shared_grid is not None:
                combined = shared_grid
            else:
                combined = self._voxelize_combined(
                    rec_coords, rec_types, lig_coords, lig_types,
                    center, minfo
                )

            s, a = self._infer(combined, minfo)
            scores.append(s)
            affinities.append(a)

        return {
            "CNNscore": float(np.mean(scores)),
            "CNNaffinity": float(np.mean(affinities)),
            "per_model": [
                {"name": m["name"], "score": s, "affinity": a}
                for m, s, a in zip(self.models, scores, affinities)
            ]
        }

    def score_poses(self, rec_coords, rec_types, poses, center=None):
        """Score multiple ligand poses against the same receptor (batch).

        Shares receptor voxelization across all poses for efficiency.

        Args:
            rec_coords: (N, 3) receptor coordinates
            rec_types: list of N SMINA type names
            poses: list of (lig_coords, lig_types) tuples
            center: (3,) grid center; if None, uses first ligand centroid

        Returns:
            list of result dicts, one per pose
        """
        if not poses:
            return []

        if center is None:
            center = np.mean(poses[0][0], axis=0)

        # Pre-voxelize receptor (shared across all poses)
        minfo0 = self.models[0]
        shared = self._models_share_grid_params()
        rec_grid = None
        if shared:
            rec_grid = voxelize_fast(
                rec_coords, rec_types, minfo0["rec_map"], minfo0["rec_nch"],
                center, minfo0["resolution"], minfo0["dimension"],
                minfo0["radius_scale"]
            )

        results = []
        for lig_coords, lig_types in poses:
            scores = []
            affinities = []

            for minfo in self.models:
                if shared:
                    lig_grid = voxelize_fast(
                        lig_coords, lig_types, minfo["lig_map"], minfo["lig_nch"],
                        center, minfo["resolution"], minfo["dimension"],
                        minfo["radius_scale"]
                    )
                    combined = np.concatenate([rec_grid, lig_grid], axis=0)
                else:
                    combined = self._voxelize_combined(
                        rec_coords, rec_types, lig_coords, lig_types,
                        center, minfo
                    )

                s, a = self._infer(combined, minfo)
                scores.append(s)
                affinities.append(a)

            results.append({
                "CNNscore": float(np.mean(scores)),
                "CNNaffinity": float(np.mean(affinities)),
                "per_model": [
                    {"name": m["name"], "score": s, "affinity": a}
                    for m, s, a in zip(self.models, scores, affinities)
                ]
            })

        return results

    def score_from_files(self, rec_path, lig_path, center=None, remove_h=True):
        """Score from PDB/PDBQT (receptor) and SDF/PDB/PDBQT (ligand) files.

        For multi-molecule SDF files, scores ALL poses (each with its own
        centroid as grid center, matching gnina's --score_only behavior).

        Args:
            rec_path: path to receptor PDB or PDBQT file
            lig_path: path to ligand SDF, PDB, or PDBQT file
            center: optional (3,) center; defaults to each ligand's centroid
            remove_h: skip hydrogen atoms (GNINA default skips H for grid)

        Returns:
            For single-pose: dict with CNNscore, CNNaffinity, per_model
            For multi-pose SDF: list of such dicts
        """
        rec_path = Path(rec_path)
        if rec_path.suffix.lower() == ".pdbqt":
            rec_coords, rec_types = read_pdbqt(rec_path, remove_h=remove_h)
        else:
            _, rec_coords, rec_types = read_pdb(rec_path, remove_h=remove_h)

        lig_path = Path(lig_path)
        if lig_path.suffix.lower() == ".sdf":
            all_mols = read_sdf_all(lig_path, remove_h=remove_h)
            if len(all_mols) == 1:
                _, lig_coords, lig_types, lig_centroid = all_mols[0]
                pose_center = center if center is not None else lig_centroid
                return self.score(rec_coords, rec_types, lig_coords, lig_types, pose_center)
            # Multi-pose: score each with its own all-atom centroid
            results = []
            for _, lig_coords, lig_types, lig_centroid in all_mols:
                pose_center = center if center is not None else lig_centroid
                results.append(self.score(rec_coords, rec_types, lig_coords, lig_types, pose_center))
            return results
        elif lig_path.suffix.lower() == ".pdbqt":
            lig_coords, lig_types = read_pdbqt(lig_path, remove_h=remove_h)
        else:
            _, lig_coords, lig_types = read_pdb(lig_path, remove_h=remove_h)

        return self.score(rec_coords, rec_types, lig_coords, lig_types, center)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 3:
        print("Usage: python gnina_scorer.py <receptor.pdb> <ligand.sdf> [cx cy cz]")
        sys.exit(1)

    rec_path = sys.argv[1]
    lig_path = sys.argv[2]
    center = None
    if len(sys.argv) >= 6:
        center = np.array([float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])])

    ensemble = None
    if "--ensemble" in sys.argv:
        idx = sys.argv.index("--ensemble")
        ensemble = sys.argv[idx + 1]
        # Remove these args so they don't interfere with positional parsing
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    print(f"Loading GNINA models...")
    t0 = time.time()
    model = GNINAModel(ensemble=ensemble)
    print(f"Models loaded in {time.time()-t0:.1f}s")

    print(f"\nScoring: {rec_path} + {lig_path}")
    t0 = time.time()
    result = model.score_from_files(rec_path, lig_path, center=center)
    elapsed = time.time() - t0

    if isinstance(result, list):
        print(f"\n{len(result)} poses scored (in {elapsed:.1f}s):")
        for pi, r in enumerate(result):
            print(f"\n  Pose {pi+1}:")
            print(f"    CNNscore:    {r['CNNscore']:.5f}")
            print(f"    CNNaffinity: {r['CNNaffinity']:.5f}")
            for pm in r["per_model"]:
                print(f"      {pm['name']:30s}  score={pm['score']:.5f}  affinity={pm['affinity']:.5f}")
    else:
        print(f"\nResults (in {elapsed:.1f}s):")
        print(f"  CNNscore:    {result['CNNscore']:.5f}")
        print(f"  CNNaffinity: {result['CNNaffinity']:.5f}")
        for pm in result["per_model"]:
            print(f"      {pm['name']:30s}  score={pm['score']:.5f}  affinity={pm['affinity']:.5f}")
