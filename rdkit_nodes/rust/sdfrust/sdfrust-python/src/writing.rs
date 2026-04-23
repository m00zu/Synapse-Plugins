//! Python bindings for SDF writing functions.

use pyo3::prelude::*;
use std::path::Path;

use sdfrust::{
    mol_to_pdbqt, mol_to_pdbqt_with_remarks, mol_to_pdbqt_ext, write_mol2_file,
    write_mol2_file_multi, write_mol2_string, write_pdbqt_file, write_sdf_auto_file,
    write_sdf_auto_string, write_sdf_file, write_sdf_file_multi, write_sdf_string,
    write_sdf_v3000_file, write_sdf_v3000_file_multi, write_sdf_v3000_string,
};

use crate::error::convert_error;
use crate::molecule::PyMolecule;

// ============================================================
// SDF V2000 Writing
// ============================================================

/// Write a molecule to an SDF file (V2000 format).
///
/// Args:
///     molecule: The molecule to write.
///     path: Path to the output file.
///
/// Raises:
///     IOError: If the file cannot be written.
#[pyfunction]
#[pyo3(name = "write_sdf_file")]
pub fn py_write_sdf_file(molecule: &PyMolecule, path: &str) -> PyResult<()> {
    write_sdf_file(Path::new(path), &molecule.inner).map_err(convert_error)
}

/// Write a molecule to an SDF string (V2000 format).
///
/// Args:
///     molecule: The molecule to write.
///
/// Returns:
///     The SDF content as a string.
#[pyfunction]
#[pyo3(name = "write_sdf_string")]
pub fn py_write_sdf_string(molecule: &PyMolecule) -> PyResult<String> {
    write_sdf_string(&molecule.inner).map_err(convert_error)
}

/// Write multiple molecules to an SDF file (V2000 format).
///
/// Args:
///     molecules: List of molecules to write.
///     path: Path to the output file.
///
/// Raises:
///     IOError: If the file cannot be written.
#[pyfunction]
#[pyo3(name = "write_sdf_file_multi")]
pub fn py_write_sdf_file_multi(molecules: Vec<PyMolecule>, path: &str) -> PyResult<()> {
    let mols: Vec<_> = molecules.into_iter().map(|m| m.inner).collect();
    write_sdf_file_multi(Path::new(path), &mols).map_err(convert_error)
}

// ============================================================
// SDF V3000 Writing
// ============================================================

/// Write a molecule to an SDF file (V3000 format).
///
/// Args:
///     molecule: The molecule to write.
///     path: Path to the output file.
///
/// Raises:
///     IOError: If the file cannot be written.
#[pyfunction]
#[pyo3(name = "write_sdf_v3000_file")]
pub fn py_write_sdf_v3000_file(molecule: &PyMolecule, path: &str) -> PyResult<()> {
    write_sdf_v3000_file(Path::new(path), &molecule.inner).map_err(convert_error)
}

/// Write a molecule to an SDF string (V3000 format).
///
/// Args:
///     molecule: The molecule to write.
///
/// Returns:
///     The SDF content as a string.
#[pyfunction]
#[pyo3(name = "write_sdf_v3000_string")]
pub fn py_write_sdf_v3000_string(molecule: &PyMolecule) -> PyResult<String> {
    write_sdf_v3000_string(&molecule.inner).map_err(convert_error)
}

/// Write multiple molecules to an SDF file (V3000 format).
///
/// Args:
///     molecules: List of molecules to write.
///     path: Path to the output file.
///
/// Raises:
///     IOError: If the file cannot be written.
#[pyfunction]
#[pyo3(name = "write_sdf_v3000_file_multi")]
pub fn py_write_sdf_v3000_file_multi(molecules: Vec<PyMolecule>, path: &str) -> PyResult<()> {
    let mols: Vec<_> = molecules.into_iter().map(|m| m.inner).collect();
    write_sdf_v3000_file_multi(Path::new(path), &mols).map_err(convert_error)
}

// ============================================================
// SDF Auto-Format Writing (V2000/V3000)
// ============================================================

/// Write a molecule to an SDF file with automatic format selection.
///
/// Uses V3000 format if the molecule has >999 atoms/bonds or V3000-only features,
/// otherwise uses V2000 format.
///
/// Args:
///     molecule: The molecule to write.
///     path: Path to the output file.
///
/// Raises:
///     IOError: If the file cannot be written.
#[pyfunction]
#[pyo3(name = "write_sdf_auto_file")]
pub fn py_write_sdf_auto_file(molecule: &PyMolecule, path: &str) -> PyResult<()> {
    write_sdf_auto_file(Path::new(path), &molecule.inner).map_err(convert_error)
}

/// Write a molecule to an SDF string with automatic format selection.
///
/// Uses V3000 format if the molecule has >999 atoms/bonds or V3000-only features,
/// otherwise uses V2000 format.
///
/// Args:
///     molecule: The molecule to write.
///
/// Returns:
///     The SDF content as a string.
#[pyfunction]
#[pyo3(name = "write_sdf_auto_string")]
pub fn py_write_sdf_auto_string(molecule: &PyMolecule) -> PyResult<String> {
    write_sdf_auto_string(&molecule.inner).map_err(convert_error)
}

// ============================================================
// MOL2 Writing
// ============================================================

/// Write a molecule to a MOL2 file.
///
/// Args:
///     molecule: The molecule to write.
///     path: Path to the output file.
///
/// Raises:
///     IOError: If the file cannot be written.
#[pyfunction]
#[pyo3(name = "write_mol2_file")]
pub fn py_write_mol2_file(molecule: &PyMolecule, path: &str) -> PyResult<()> {
    write_mol2_file(Path::new(path), &molecule.inner).map_err(convert_error)
}

/// Write a molecule to a MOL2 string.
///
/// Args:
///     molecule: The molecule to write.
///
/// Returns:
///     The MOL2 content as a string.
#[pyfunction]
#[pyo3(name = "write_mol2_string")]
pub fn py_write_mol2_string(molecule: &PyMolecule) -> PyResult<String> {
    write_mol2_string(&molecule.inner).map_err(convert_error)
}

/// Write multiple molecules to a MOL2 file.
///
/// Args:
///     molecules: List of molecules to write.
///     path: Path to the output file.
///
/// Raises:
///     IOError: If the file cannot be written.
#[pyfunction]
#[pyo3(name = "write_mol2_file_multi")]
pub fn py_write_mol2_file_multi(molecules: Vec<PyMolecule>, path: &str) -> PyResult<()> {
    let mols: Vec<_> = molecules.into_iter().map(|m| m.inner).collect();
    write_mol2_file_multi(Path::new(path), &mols).map_err(convert_error)
}

// ============================================================
// PDBQT Writing
// ============================================================

/// Convert a molecule to PDBQT format string.
///
/// The molecule must have 3D coordinates and explicit hydrogens.
/// Non-polar hydrogens are merged (charge transferred to parent).
/// Polar hydrogens (on N/O/F/P/S) are kept as HD atoms.
///
/// Args:
///     molecule: The molecule to convert.
///     smiles: Optional SMILES string for REMARK SMILES line.
///     smiles_idx: Optional flat list of (orig_atom_idx, pdbqt_serial) pairs for REMARK SMILES IDX.
///     h_parent: Optional flat list of (parent_serial, h_serial) pairs for REMARK H PARENT.
///     aromatic_atoms: Optional list of booleans (one per atom) indicating aromaticity.
///         When provided, overrides the built-in Hückel aromaticity detection.
///         Use RDKit's aromaticity for best accuracy with Kekulized SDF files.
///     symmetry_classes: Optional list of unsigned ints (one per atom) from
///         ``Chem.CanonicalRankAtoms(mol, breakTies=False)``.  When provided,
///         these are used for the tertiary-amide equivalence check instead of
///         the built-in Morgan-style ranking (which lacks chirality info).
///     smiles_atom_order: Optional list of original atom indices in SMILES output
///         order.  When provided together with ``smiles``, the REMARK SMILES IDX
///         and REMARK H PARENT lines are computed automatically from the internal
///         atom numbering (overrides ``smiles_idx`` and ``h_parent``).
///
/// Returns:
///     PDBQT string with ROOT/BRANCH/ENDBRANCH torsion tree.
#[pyfunction]
#[pyo3(name = "mol_to_pdbqt")]
#[pyo3(signature = (molecule, smiles=None, smiles_idx=None, h_parent=None, aromatic_atoms=None, symmetry_classes=None, smiles_atom_order=None))]
pub fn py_mol_to_pdbqt(
    molecule: &PyMolecule,
    smiles: Option<&str>,
    smiles_idx: Option<Vec<(usize, usize)>>,
    h_parent: Option<Vec<(usize, usize)>>,
    aromatic_atoms: Option<Vec<bool>>,
    symmetry_classes: Option<Vec<u32>>,
    smiles_atom_order: Option<Vec<usize>>,
) -> PyResult<String> {
    mol_to_pdbqt_ext(
        &molecule.inner,
        smiles,
        smiles_idx.as_deref(),
        h_parent.as_deref(),
        aromatic_atoms.as_deref(),
        symmetry_classes.as_deref(),
        smiles_atom_order.as_deref(),
    )
    .map_err(convert_error)
}

/// Write a molecule as PDBQT to a file.
///
/// Args:
///     molecule: The molecule to write.
///     path: Path to the output file.
#[pyfunction]
#[pyo3(name = "write_pdbqt_file")]
pub fn py_write_pdbqt_file(molecule: &PyMolecule, path: &str) -> PyResult<()> {
    write_pdbqt_file(Path::new(path), &molecule.inner).map_err(convert_error)
}

/// Convert multiple molecules to PDBQT format in parallel.
///
/// Uses Rayon to parallelize the conversion across all CPU cores.
/// Returns None for molecules that fail conversion.
///
/// An optional callback is called for each molecule as it completes:
///     callback(index: int, total: int, pdbqt: Optional[str])
///
/// Args:
///     molecules: List of molecules to convert.
///     aromatic_atoms: Optional list of per-molecule aromaticity lists.
///         Each element is either a list of booleans (one per atom) or None.
///         When provided, overrides built-in aromaticity detection per molecule.
///     symmetry_classes: Optional list of per-molecule symmetry class lists.
///         Each element is either a list of unsigned ints or None.
///         From ``Chem.CanonicalRankAtoms(mol, breakTies=False)``.
///     smiles: Optional list of per-molecule SMILES strings (or None per entry).
///         Used for REMARK SMILES line in the PDBQT output.
///     smiles_atom_orders: Optional list of per-molecule atom order lists (or None per entry).
///         Each is a list of original atom indices in SMILES output order.
///         When provided together with ``smiles``, REMARK SMILES IDX and REMARK H PARENT
///         are computed automatically from the internal atom numbering.
///     callback: Optional callback called per molecule with (index, total, pdbqt_or_none).
///
/// Returns:
///     List of PDBQT strings (None for failed conversions).
#[pyfunction]
#[pyo3(name = "batch_mol_to_pdbqt")]
#[pyo3(signature = (molecules, aromatic_atoms=None, symmetry_classes=None, smiles=None, smiles_atom_orders=None, callback=None))]
pub fn py_batch_mol_to_pdbqt(
    py: Python<'_>,
    molecules: Vec<PyRef<'_, PyMolecule>>,
    aromatic_atoms: Option<Vec<Option<Vec<bool>>>>,
    symmetry_classes: Option<Vec<Option<Vec<u32>>>>,
    smiles: Option<Vec<Option<String>>>,
    smiles_atom_orders: Option<Vec<Option<Vec<usize>>>>,
    callback: Option<PyObject>,
) -> PyResult<Vec<Option<String>>> {
    let mols: Vec<sdfrust::Molecule> = molecules.iter().map(|m| m.inner.clone()).collect();
    let total = mols.len();

    // Pre-extract per-molecule optional slices
    let arom_data: Vec<Option<Vec<bool>>> = match aromatic_atoms {
        Some(v) => {
            if v.len() != total {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("aromatic_atoms length ({}) != molecules length ({})", v.len(), total)
                ));
            }
            v
        }
        None => (0..total).map(|_| None).collect(),
    };
    let sym_data: Vec<Option<Vec<u32>>> = match symmetry_classes {
        Some(v) => {
            if v.len() != total {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("symmetry_classes length ({}) != molecules length ({})", v.len(), total)
                ));
            }
            v
        }
        None => (0..total).map(|_| None).collect(),
    };
    let smi_data: Vec<Option<String>> = match smiles {
        Some(v) => {
            if v.len() != total {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("smiles length ({}) != molecules length ({})", v.len(), total)
                ));
            }
            v
        }
        None => (0..total).map(|_| None).collect(),
    };
    let smi_order_data: Vec<Option<Vec<usize>>> = match smiles_atom_orders {
        Some(v) => {
            if v.len() != total {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("smiles_atom_orders length ({}) != molecules length ({})", v.len(), total)
                ));
            }
            v
        }
        None => (0..total).map(|_| None).collect(),
    };

    let results: Vec<Option<String>> = py.allow_threads(|| {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            mols.par_iter()
                .enumerate()
                .map(|(i, m)| {
                    let result = mol_to_pdbqt_ext(
                        m,
                        smi_data[i].as_deref(),
                        None,
                        None,
                        arom_data[i].as_deref(),
                        sym_data[i].as_deref(),
                        smi_order_data[i].as_deref(),
                    ).ok();
                    if let Some(ref cb) = callback {
                        Python::with_gil(|py| {
                            let _ = cb.call1(py, (i, total, &result));
                        });
                    }
                    result
                })
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            mols.iter()
                .enumerate()
                .map(|(i, m)| {
                    let result = mol_to_pdbqt_ext(
                        m,
                        smi_data[i].as_deref(),
                        None,
                        None,
                        arom_data[i].as_deref(),
                        sym_data[i].as_deref(),
                        smi_order_data[i].as_deref(),
                    ).ok();
                    if let Some(ref cb) = callback {
                        Python::with_gil(|py| {
                            let _ = cb.call1(py, (i, total, &result));
                        });
                    }
                    result
                })
                .collect()
        }
    });

    Ok(results)
}
