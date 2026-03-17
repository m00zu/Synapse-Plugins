pub mod common;
pub mod atom;
pub mod scoring;
pub mod precalculate;
pub mod conf;
pub mod grid;
pub mod tree;
pub mod model;
pub mod parse_pdbqt;
pub mod cache;
pub mod bfgs;
pub mod visited;
pub mod monte_carlo;
pub mod parallel;
pub mod python;
pub mod rxdock_atom;
pub mod rxdock_cavity;
pub mod parse_sdf;
pub mod rxdock_scoring;
pub mod rxdock_chrom;
pub mod rxdock_simplex;
pub mod rxdock_sa;
pub mod rxdock_ga;
pub mod rxdock_dihedral;
pub mod rxdock_search;

use pyo3::prelude::*;

/// AutoDock Vina + QuickVina 2 reimplemented in Rust — fast molecular docking with progress reporting
#[pymodule]
fn vina_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyVina>()?;
    m.add_class::<python::PyQVina2>()?;
    m.add_class::<python::PySmina>()?;
    m.add_class::<python::PyRxDock>()?;
    Ok(())
}
